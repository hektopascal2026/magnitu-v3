# Implementation Plan: Magnitu scoring-pipeline issues

Audit date: 2026-09-11. Source: code audit of `pipeline.py`, `ml_window.py`,
`distiller.py`, `explainer.py`, `sampler.py`, `db.py`, `main.py`, `sync.py`,
`model_manager.py` at `47374fc`, focused on "what makes the pushed ranking less
useful to a journalist than it could be".

Findings marked **verified** were reproduced numerically; the reproduction is
quoted in the WP. Findings marked **measure first** are structural readings of
the code whose magnitude depends on live data — those WPs add the measurement
before changing behaviour.

## Global constraints (apply to every work package)

- Python 3.9 only: `Optional[X]` / `typing`, no `X | Y` unions.
- Do **not** change: the recipe JSON `keywords` / `source_weights` /
  `classes` / `class_weights` contract, the score push payload shape,
  `entry_type` values, any Seismo PHP. Adding keys under the recipe's existing
  `metrics` block is allowed (PHP ignores it) and is used by WP6.
- Do **not** change the `models` table column names (`util_at_30`,
  `precision_at_30`, …). Where a WP needs a different `k`, it reports the
  effective `k` alongside and leaves the stored top-30 columns as they are.
- Do **not** rename existing public functions. Add parameters with defaults
  that preserve current behaviour; rewrite bodies in place.
- Seismo HTTP stays in `sync.py`.
- After each WP, run
  `python -m pytest test_magnitu2.py test_ml_window.py test_common_eval.py test_recipe_parity.py test_scoring_plan.py -x -q`
  and fix regressions before moving on.
- Order matters. WP1 is a dependency of WP2 and WP8. WP6 is a dependency of
  WP7. Everything else is independent and can be reordered freely.

## Severity ordering

| WP | Issue | Effect on the desk a journalist reads |
|---|---|---|
| 1 | UTIL@30 gate is a no-op on small reserves | Regressions promote silently |
| 2 | Gate scores the incumbent in-sample | Real improvements get rejected |
| 3 | Explanations use the wrong feature space | "Why" contradicts the score |
| 4 | Smart queue recipe scorer is out of parity | Wrong entries get labelled |
| 5 | Score push drops entries without embeddings | Fresh items have no score |
| 6–7 | Recipe and model composites on different scales | Unscored rows top the feed |
| 8 | `@30` metrics measure the holdout, not the model | Model page misleads |
| 9 | Calibration telemetry is write-only | Miscalibration invisible |
| 10 | `recipe_quality` measured on an unguarded slice | Recipe withheld wrongly |
| 11 | Embedding representation drift on analytical sources | Mixed feature space |
| 12 | Assorted small wiring bugs | Various |

---

## WP1 — Make the promote gate discriminative on small eval reserves

**Verified.** `pipeline._ranking_metrics` (line 800) clips
`k_eff = min(k, n)` at line 870. When the holdout has `n <= 30` rows, "top-30"
is the whole holdout, so `util_at_30` reduces to the mean true class weight of
the eval set — identical for any two models regardless of how they rank.
`bootstrap_util_delta` (line 1720) recomputes `_util` with
`k_eff = min(k, len(idx))`, so every resample yields delta exactly 0, `tie` is
True, and `ml_window.evaluate_recent_gate` (line 351) promotes.

`db.EVAL_RESERVE_MAX_FRACTION = 0.3` (line 1282) caps the reserve at 30% of
confirmed labels, so the reserve cannot exceed 30 rows until a desk has more
than 100 labels — the gate is blind over exactly the range where most desks
live, and near-blind (k_eff within a few rows of n) up to ~200 labels.

Reproduction — perfect ranker as incumbent, perfectly inverted ranker as
challenger, same holdout:

```
n=  28  util perfect=0.307 inverted=0.307  AUC 1.00/0.00  -> gate promotes INVERTED model: True
n=  56  util perfect=0.560 inverted=0.040  AUC 1.00/0.00  -> gate promotes INVERTED model: False
n= 112  util perfect=0.880 inverted=0.000  AUC 1.00/0.00  -> gate promotes INVERTED model: False
```

`precision_at_30` (0.2857 both arms) and `lead_recall_at_30` (1.0000 both arms)
are degenerate in the same way, so the documented diagnostics cannot catch it
either. `ranking_auc` (1.00 vs 0.00) and `ndcg_at_30` (1.00 vs 0.47) both
separate the two models cleanly and are already computed but never consulted.

**Changes:**

1. `pipeline.py`, new helper next to `_RANKING_K`:

```python
def gate_k(n: int) -> int:
    """Top-k for the promote gate: never the whole holdout.

    UTIL@k is rank-invariant when k >= n (top-k is every row), which makes the
    gate a no-op on the <=30-row eval reserves that desks under ~100 labels
    have. Half the holdout keeps the metric discriminative while still scoring
    the part of the ranking an operator actually reads.
    """
    n = int(n)
    if n < 2:
        return 1
    return max(5, min(_RANKING_K, n // 2))
```

2. `pipeline._ranking_metrics`: add two non-persisted keys to `out`, set after
   `k_eff` is computed — `out["k_eff"] = int(k_eff)` and
   `out["util_degenerate"] = bool(k_eff >= n)`. Leave every existing key and
   its rounding untouched.

3. `pipeline._holdout_classification_metrics` (line 1689): add
   `k: int = _RANKING_K`, pass it to `_ranking_metrics`, and copy `k_eff` /
   `util_degenerate` into the returned dict alongside the existing
   `_composites` / `_true_weights` extras (they are already excluded from JSON
   serialisation by callers).

4. `pipeline.evaluate_fitted_model` (line 1771) and `pipeline.evaluate_on_recent`
   (line 2001): add `k: Optional[int] = None`, thread it through. `None` keeps
   today's `_RANKING_K`.

5. `ml_window.evaluate_recent_gate`: after `n` is read, compute
   `k = pipeline.gate_k(n)` and pass `k=k` to `pipeline.bootstrap_util_delta`.
   Recompute `old_util` / `new_util` for the log line at the same `k` rather
   than reading the stored `util_at_30` (they now mean different things).
   Log `gate_k` in the decision line and add `report["gate_k"] = k` in
   `main()`'s desk loop.

6. `ml_window.evaluate_recent_gate`: keep a belt-and-braces branch — if the
   metrics still report `util_degenerate` at the gate `k`, decide on
   `ndcg_at_30` instead and log `decision_metric="ndcg"`. With `gate_k` this
   should be unreachable for `n >= GATE_MIN_RECENT`; it exists so a future
   change to `gate_k` cannot silently reintroduce a blind gate.

Do **not** raise `EVAL_RESERVE_MAX_FRACTION` to fix this — that starves
training, which is the constraint the cap exists to protect.

**Acceptance (`test_ml_window.py`):**

- `pipeline.gate_k(28) == 14`, `gate_k(9) == 5`, `gate_k(200) == 30`.
- `_ranking_metrics` on a 28-row holdout reports `util_degenerate is True` at
  `k=30` and `False` at `k=14`.
- The perfect-vs-inverted table above: `evaluate_recent_gate` returns `False`
  at every `n` in `(28, 56, 112, 224)`.
- A genuinely tied pair (identical composites) still promotes at `n=28`.

---

## WP2 — Stop scoring the incumbent in-sample at the gate

**Structural, verified by reading `roll_eval_reserve`.** `db.roll_eval_reserve`
(line 1323) admits labels once `created_at` is `admit_days` (7) old. A row
admitted at retrain time `T1` has `created_at <= T1 - 7d`. If the incumbent was
trained at `T0 < T1`, any row with `T0 - 7d < created_at <= T1 - 7d` existed as
a label at `T0` but was not in the reserve the incumbent excluded — so it was
in the incumbent's **training set**. The challenger always excludes the full
current reserve.

`evaluate_on_recent` therefore scores the incumbent partly in-sample and the
challenger fully out-of-sample, on every cycle. The bias suppresses promotions,
which is the opposite direction from the gate's stated "on a tie the new model
wins" doctrine. On a desk retraining every 15 labels it is a standing handicap
on the challenger.

`eval_reserve.added_at` already exists and is indexed
(`idx_eval_reserve_profile ON eval_reserve(profile_id, added_at)`), and
`models.trained_at` exists, so no schema change is needed.

**Changes:**

1. `db.get_eval_reserve_rows` (line 1296): add
   `added_before: Optional[str] = None`. When set, add
   `AND r.added_at < ?` to the WHERE clause. Default `None` keeps today's
   behaviour.

2. `pipeline.recent_holdout_features` (line 1790): add
   `added_before: Optional[str] = None` and `roll: bool = True`.
   - Pass `added_before` to `get_eval_reserve_rows`.
   - Skip the `db.roll_eval_reserve` call when `roll` is False. The gate calls
     this function twice (once per arm) and each call currently re-rolls the
     reserve, so the set can change between arms; the second arm must not roll.

3. `pipeline.evaluate_on_recent` (line 2001): add
   `added_before: Optional[str] = None` and `roll: bool = True`, thread both
   through.

4. `ml_window.py`, in the train branch of `main()`: compute the cut once and
   pass the **same value to both arms** so the two `_composites` arrays
   describe identical rows.

```python
gate_cut = (current_model or {}).get("trained_at") or None
new_recent = pipeline.evaluate_on_recent(
    res, profile_id=profile_id, n_recent=GATE_N_RECENT,
    added_before=gate_cut, roll=True,
)
old_recent = None
if current_model:
    old_recent = pipeline.evaluate_on_recent(
        current_model, profile_id=profile_id, n_recent=GATE_N_RECENT,
        added_before=gate_cut, roll=False,
    )
```

5. Fallback when the filtered set is too small: if `new_recent` reports
   `n_recent < GATE_MIN_RECENT` **and** the unfiltered reserve is larger, redo
   both arms with `added_before=None`, set `report["gate_eval_biased"] = True`,
   and log that the incumbent is being scored partly in-sample. Do not refuse
   to promote — that would freeze desks whose reserve is younger than their
   active model. Making the bias explicit in the window report is the
   deliverable; the tie-promotes rule already leans the right way.

**Acceptance (`test_common_eval.py`):**

- With a reserve containing rows added both before and after a model's
  `trained_at`, `get_eval_reserve_rows(..., added_before=trained_at)` returns
  only the earlier rows.
- Both gate arms receive the same row count and the same `y` sequence, so
  `bootstrap_util_delta` never falls back to the one-item slack path purely
  because of a length mismatch.
- With `roll=False`, `recent_holdout_features` does not call
  `roll_eval_reserve` (monkeypatch a counter).
- A reserve entirely newer than `trained_at` sets `gate_eval_biased` and still
  produces a decision.

---

## WP3 — Score explanations in the same feature space as the score

**Verified.** `pipeline._score_transformer` (line 2604) applies `_l2_normalize`
when the model record says so (lines 2678-2679, reading the per-model
`embedding_l2_normalize` flag added in `ffbaa8c`).
`explainer._explain_transformer` (line 177) does not — it goes straight from
`bytes_to_embedding` to `emb.reshape(1, -1)` to `classifier_probabilities`
(lines 200-208). The model was fit on normalized vectors, so the explainer
hands the `StandardScaler` a representation it never saw.

`main.py`'s push loop (lines 446-460) copies `exp["prediction"]` and
`exp["confidence"]` into `score["explanation"]`, which goes to Seismo. So the
desk shows a "why" whose predicted class can contradict the `predicted_label`
and `relevance_score` next to it.

Measured on synthetic E5-like vectors with realistic norm spread (mean-pooled,
un-normalized norms 0.4–3.5): different argmax label on **6.2%** of entries,
mean absolute composite difference **0.098**, max **0.274**.

Two further wiring bugs in the same function: it reads `embedding_dim` from the
global `get_config()` instead of `db.get_effective_config(profile_id)`, and it
ignores the per-model flag entirely.

**Changes — `explainer.py`:**

1. `explain_entry`: pass `profile_id` into `_explain_transformer` (it currently
   only forwards `model_info`).

2. `_explain_transformer(entry, model_info, profile_id=1)`:

```python
config = db.get_effective_config(profile_id)
embedding_dim = config.get("embedding_dim", 768)
...
X = emb.reshape(1, -1)
if bool(model_info.get("embedding_l2_normalize", 0)):
    X = _l2_normalize(X)
```

   Import `_l2_normalize` from `pipeline` alongside the existing imports.

3. Pass the model's normalization decision, not the config's — same rule as
   `_score_transformer`, for the same reason (`P0-2`).

**Acceptance (`test_recipe_quality.py` or a new `test_explainer_parity.py`):**

- For a model record with `embedding_l2_normalize=1`,
  `explain_entry(entry)["prediction"]` equals
  `score_entries([entry])[0]["predicted_label"]`, and
  `explain_entry(entry)["relevance_score"]` equals the pushed
  `relevance_score`, for a set of entries with widely varying embedding norms.
- For a record with `embedding_l2_normalize=0`, behaviour is byte-identical to
  today.

---

## WP4 — Put the smart queue's recipe scorer back in PHP parity

**Verified.** `sampler._recipe_predict` (line 36) is an independent
re-implementation of the recipe scorer that diverges from
`distiller._accumulate_recipe_class_scores` in three ways: it splits on
whitespace without stripping punctuation, it accumulates every repeat hit
instead of once per keyword, and it reads `content` for all entry types instead
of synopsis-only for `lex_item` / `calendar_event`.

Same entry, same recipe, both scorers:

```
sampler bigrams near the keywords: ['market access', 'a third', 'third country.', 'market access,', ...]
sampler matched keys:              ['market access']        # 'third country' invisible (trailing period)
PHP-parity class sums:             {'investigation_lead': 0.8, ...}   # both phrases, once each

lex_item sampler (reads content) -> investigation_lead
lex_item parity (synopsis only)  -> {'investigation_lead': 0.0, 'important': 0.0, ...}
```

Consequence: the `conflict` bucket of the smart queue compares the model
against a broken scorer. Real signal is dropped on punctuation, repeated
phrases double-count, and every `lex_item` is a phantom disagreement. Since
this queue decides what a journalist labels next, the error compounds into the
training set. `distiller` and `explainer` already use the parity helpers; only
`sampler` is out of parity — WP2 of `scoring-fix-plan.md` built these helpers
and this call site was missed.

**Changes — `sampler.py`:**

1. Replace the body of `_recipe_predict` with a call through the parity path
   (keep the signature and the `Optional[str]` return):

```python
from distiller import _accumulate_recipe_class_scores

def _recipe_predict(entry: dict, recipe: dict) -> Optional[str]:
    """Predicted label from the recipe, using the PHP-parity scorer.

    Must match what Seismo actually runs, or the `conflict` queue surfaces
    phantom disagreements instead of real ones.
    """
    classes = recipe.get("classes", [
        "investigation_lead", "important", "background", "noise",
    ])
    class_scores = _accumulate_recipe_class_scores(
        entry, recipe.get("keywords", {}), recipe.get("source_weights", {}),
        classes,
    )
    if not class_scores:
        return None
    max_s = max(class_scores.values())
    exp_scores = {c: math.exp(class_scores.get(c, 0.0) - max_s) for c in classes}
    exp_sum = sum(exp_scores.values())
    if exp_sum == 0:
        return None
    probs = {c: exp_scores[c] / exp_sum for c in classes}
    return max(probs, key=probs.get)
```

2. `distiller` is already imported at the top of `sampler.py`, so no new
   import cycle. Keep the local softmax rather than reusing
   `_recipe_composite` — the sampler needs the argmax label, not the composite.

**Acceptance (extend `test_recipe_parity.py`):**

- A keyword followed by punctuation (`"third country."`) matches.
- A phrase repeated 3× in one entry contributes its weight once, so the
  predicted label matches `_accumulate_recipe_class_scores` argmax.
- A `lex_item` whose signal lives only in `content` yields the same prediction
  from `_recipe_predict` as from `_accumulate_recipe_class_scores` (both see
  synopsis only).

---

## WP5 — Stop dropping entries from the score push, and stop loading BLOBs

**Verified by reading the call sites.** `ml_window.main()`'s score-push block
(lines 858-879) calls `pipeline.score_entries(recent_entries, …)` directly.
`_score_transformer` caps on-the-fly embedding at
`MAX_ONTHEFLY_EMBEDDINGS = 10` and silently omits the rest, and `push_scores`
only sends what it got — so Seismo keeps a stale score, or none. In score-only
mode `embed_cap = 200` per 15-minute tick, so a backlog larger than the tick
rate leaves a standing set of recent entries with no current score at all,
invisible in the desk ranking.

`main.py`'s interactive push drains embeddings first (lines 400-421) and logs
the shortfall (lines 437-442); the headless window does neither.

Separately, three `distiller.py` call sites and one in `main.py` load the whole
entries table **with embedding BLOBs**:

| Site | Call |
|---|---|
| `distiller.py:444` (`_normalize_weights`) | `db.get_all_entries()` |
| `distiller.py:923` (`_optimize_recipe_caps`) | `db.get_all_entries()` |
| `distiller.py:1032` (`evaluate_recipe_rank_metrics`) | `db.get_all_entries()[:sample_size]` |
| `main.py:395` (push) | `db.get_all_entries()` |

`train_tfidf_student` was rewritten to sample keys and use
`get_entries_text_by_keys` precisely to avoid this, and
`_distill_recipe_in_subprocess` plus `distillation_max_entries` exist because of
the 4 GB cgroup OOM. At ~44k entries each call is roughly 130 MB of BLOBs plus
full bodies, so the OOM path the subprocess isolation was built for is still
open — and when the child dies, Seismo keeps the stale recipe. `main.py:395`
additionally holds two copies: `get_all_entries()` with BLOBs, then
`score_entries` reloads the same embeddings via `_embedding_blobs_for_entries`.

**Changes:**

1. `ml_window.py`, before `pipeline.score_entries` in the score-push block:
   drain embeddings for the push window, then report coverage.

```python
push_days = _score_push_days()
recent_entries = db.get_recent_entries(days=push_days, include_embedding=False)
if recent_entries:
    _embed_pending_until_done(max_entries=embed_cap)
    scores = pipeline.score_entries(recent_entries, profile_id=profile_id)
    report["entries_in_window"] = len(recent_entries)
    report["scores_pushed"] = len(scores)
    if len(scores) < len(recent_entries):
        missing = len(recent_entries) - len(scores)
        report["scores_missing"] = missing
        logger.warning(
            "Score push covered %d / %d window entries; %d lack embeddings "
            "(desk %s). Seismo keeps stale scores for those rows.",
            len(scores), len(recent_entries), missing, url,
        )
```

   `_embed_pending_until_done` respects `embed_cap`, so a score-only tick still
   fits its 300s budget; the shortfall is now reported instead of silent.

2. Add `scores_pushed` / `scores_missing` / `entries_in_window` to the desk
   report dict initialiser in `main()` so the window report always carries the
   keys.

3. Switch all four `get_all_entries()` sites above to
   `db.get_all_entries(include_embedding=False)`. None of them touch the
   `embedding` column: `_normalize_weights` and `_optimize_recipe_caps` only
   score text through the recipe, `evaluate_recipe_rank_metrics` only needs
   text, and `main.py`'s push passes entries to `score_entries`, which loads
   embeddings itself via `_embedding_blobs_for_entries`.

4. `main.py:393` — `db.get_recent_entries(days=pruning_days)` in the same
   branch also defaults to `include_embedding=True`. Add
   `include_embedding=False`.

**Acceptance:**

- `test_ml_window.py`: with a window containing entries that have no
  embeddings, the desk report carries a non-zero `scores_missing` and the
  warning fires.
- `test_magnitu2.py`: `_normalize_weights` and `evaluate_recipe_rank_metrics`
  produce byte-identical output when entries are loaded without embeddings.
- Memory smoke: on a store with ≥10k embedded entries, peak RSS of
  `distiller.distill_recipe` drops. Record before/after in the commit message.

---

## WP6 — Measure the recipe-vs-model level gap (diagnostic only)

**Measure first.** Structurally, the recipe composite for an entry with no
keyword hits — or with hits that balance across classes — is exactly the mean
of `class_weights`:

```
(1.0 + 0.80 + 0.20 + 0.0) / 4 = 0.50
```

`README.md:303` states "Seismo evaluates the recipe on new/unscored rows;
Magnitu-pushed scores take precedence when present. Both are absolute
class-weighted composites." Measured on the model side with the shipped config
(`classifier_c: 0.01`, `embedding_l2_normalize: true`, balanced LogReg, OOF
temperature) on synthetic data at a 4% lead / 12% important prior:

| | mean composite | 99th percentile |
|---|---|---|
| model, 200 training labels | 0.133 | 0.453 |
| model, 800 training labels | 0.176 | 0.798 |
| recipe, no keyword match | 0.500 | — |

If the live numbers look like this, a freshly pulled entry the recipe knows
nothing about outranks nearly everything the model scored, and the top of the
feed fills with unscored novelty rather than leads. Rank normalization used to
mask the gap and was removed on 2026-09-01 (`6a33db7`).

Nothing currently measures it. `recipe_quality` is Spearman rank correlation
(`distiller.evaluate_recipe_rank_metrics`, line 1028), which is scale-invariant
by construction, and `recipe_quality_floor` gates on it. `_apply_floor_weights`
already names "recipe scores clustering around the 0.5 no-signal attractor" as
a known root cause but addresses it by boosting a dozen legal phrases rather
than by re-centring the scale.

A second, related quantity is unmeasured: `_normalize_weights` (line 434)
scales weights so the median max `|class_score|` hits
`recipe_normalize_target` (2.0), and **then** `_stabilize_export_weights` clips
each weight to `recipe_max_unigram_abs` (0.12) / `recipe_max_phrase_abs`
(0.24). Clipping after scaling can destroy the target, and whether it does
depends on how hard the caps bite on real coefficients. If the achieved
magnitude is far below 2.0, the recipe is compressed toward 0.50 and the level
gap follows from that.

This WP adds the three numbers and changes no behaviour.

**Changes — `distiller.py`:**

1. New helper, used by both the cap search and the reported metrics:

```python
def recipe_level_metrics(paired_model, paired_recipe, achieved_magnitude,
                         target_magnitude) -> dict:
    """Level (not rank) agreement between recipe and model composites.

    Spearman is scale-invariant, so it cannot see that Seismo's recipe pins
    no-signal rows at mean(class_weights) while pushed model scores centre far
    lower. These are the numbers that can.
    """
    m = np.asarray(paired_model, dtype=np.float64)
    r = np.asarray(paired_recipe, dtype=np.float64)
    no_signal = float(np.mean(np.abs(r - 0.5) <= 0.01)) if r.size else 0.0
    return {
        "level_offset": float(r.mean() - m.mean()) if r.size else 0.0,
        "level_mae": float(np.abs(r - m).mean()) if r.size else 0.0,
        "no_signal_share": no_signal,
        "achieved_magnitude": float(achieved_magnitude),
        "target_magnitude": float(target_magnitude),
    }
```

2. `_normalize_weights`: return the computed `median_mag` as a third element
   (callers in `distill_recipe` unpack two today — update the one call site).
   Add a second magnitude measurement **after** capping so
   `achieved_magnitude` reflects what ships, not what was aimed at.

3. `evaluate_recipe_rank_metrics`: also return the `recipe_level_metrics`
   block and `n_eval` (the number of paired rows, see WP10).

4. `distill_recipe`: write the block into `recipe["metrics"]` under
   `recipe_level` — a new key inside the existing free-form `metrics` block,
   which Seismo ignores. Do not add top-level keys.

5. `ml_window._post_promote_recipe_and_vault`: log the level block and add it
   to the desk report so it lands in the Seismo Diagnostics window report.

**Acceptance:**

- A recipe with an empty `keywords` dict reports
  `no_signal_share == 1.0` and `level_offset == 0.5 - mean(model composite)`.
- `recipe["metrics"]["recipe_level"]` is present and the recipe still parses
  under the existing contract (no top-level key added; `test_recipe_parity.py`
  schema assertions unchanged).
- Run on each live desk and record the five numbers in the PR description.
  **WP7 does not start until those numbers exist.**

---

## WP7 — Align the recipe level (requires WP6 numbers and sign-off)

**Blocked on WP6.** Which fix is right depends on what WP6 measures, so this WP
specifies the decision rather than pre-committing to an implementation.

- **If `achieved_magnitude` is far below `target_magnitude`** the caps are the
  binding constraint and the fix is internal to the cap search: widen the upper
  bounds of `unigram_grid` / `phrase_grid` in `_optimize_recipe_caps`
  (line 911) and change the objective from pure Spearman to Spearman with
  `level_mae` as the tie-break in place of `top30_overlap`. Contract-safe, no
  Seismo change, no config change.
- **If `achieved_magnitude` is at target but `no_signal_share` is high** the
  problem is keyword coverage, not scale: entries genuinely match nothing.
  The lever is `recipe_top_keywords` and the distillation corpus, not the caps.
- **If the level gap persists at target magnitude and low no-signal share**
  then the two scorers disagree on level by construction and the fix belongs on
  the Seismo side (a baseline offset applied before softmax, or scoring
  unscored rows as "unranked" rather than 0.50). That requires PHP changes in
  lockstep per the integration rules and is **out of scope for this plan** —
  open it as a Seismo issue with the WP6 numbers attached.

Do not attempt to force the level with `source_weights`: for an entry with no
keyword hits the class scores are exactly `source_weights[source_type]`, so
moving the no-signal point to ~0.20 needs a noise weight near +1.95, which is
~8× the phrase cap and would swamp every keyword the distiller learned.

**Acceptance:** whichever branch applies, `level_mae` and `no_signal_share`
improve on every desk without `recipe_quality` (Spearman) regressing by more
than 0.02, measured on the same eval set before and after.

---

## WP8 — Report the `@30` metrics honestly

**Verified.** `_holdout_test_fraction` yields a 10–20% test fold, so a
200-label desk has ~20–40 test rows and the same `k_eff >= n` collapse from WP1
applies to the metrics stored in the `models` table and shown on the model
page: `precision_at_30` tends to the share of relevant rows in the fold and
`util_at_30` to its mean class weight.

`lead_recall_at_30` is worse. At a ~4% lead rate on a 40-row fold the expected
lead count is 1–2, so the metric moves in steps of 0.5–1.0 or is 0.0 outright.
`_ranking_metrics` already records `n_leads` for exactly this reason but
nothing acts on it.

**Changes:**

1. `pipeline._train_transformer` / `_train_tfidf`: include `k_eff` (from WP1)
   in the returned dict and append it to `ranking_note` when
   `util_degenerate` is true, e.g. `"@30 metrics degenerate (holdout n=24)"`.
   Keep the stored column values and names unchanged — the fix is disclosure,
   not a new definition.

2. `pipeline._ranking_metrics`: when `n_lead_total < 3`, leave
   `lead_recall_at_30` at `0.0` (the documented "not available" value) and add
   `"lead recall not available (n_leads=N)"` to `ranking_note`, instead of
   reporting a one-lead coin flip as a rate.

3. Model page template: render `ranking_note` next to the `@30` block (it is
   already in the train result and the `models` row) so a degenerate holdout is
   visible where the number is read. Render `0.0` as "—" as the docstring
   already specifies.

**Acceptance:**

- A 24-row holdout produces a `ranking_note` naming the degeneracy.
- A holdout with 1 lead reports `lead_recall_at_30 == 0.0` and the
  not-available note; with 3 leads it reports the rate as today.
- `test_common_eval.py` assertions on existing metric values still pass.

---

## WP9 — Wire up the calibration telemetry that already exists

**Verified.** `pipeline._write_calibration_report` (line 231) computes ECE, a
10-bucket reliability table and composite quintiles into
`<model>.calreport.json` on every train. `grep` finds no reader anywhere — not
the UI, not the window report, not the gate. `cal_dict["temperature_clamped"]`
is likewise written and never consulted, so a temperature that landed on the
`[0.25, 12.0]` grid endpoint (meaning the optimum was outside the search range)
is invisible.

`model_manager._write_package` (lines 158-165) copies only `calibration.json`,
so `.calreport.json` and `.isotonic.json` are absent from `.magnitu` packages.
After an import or fork, `attach_display_scores` silently falls back to the raw
composite because the isotonic sidecar is gone.

**Changes:**

1. `ml_window.py`: after a successful train, read
   `pipeline.calibration_report_path(res["model_path"])` and
   `pipeline.load_calibration(res["model_path"])`, and add
   `report["ece"]`, `report["observed_relevant_rate"]`,
   `report["calibration_temperature"]`, `report["temperature_clamped"]` to the
   desk report. Wrap in try/except — telemetry must not fail a window.

2. Model page: surface ECE and a clamp warning from the sidecars next to the
   existing `calibration_temperature` display.

3. `model_manager.py`: copy both sidecars into the package
   (`calibration_report.json`, `isotonic.json`) and restore them on import next
   to the destination model path, mirroring the existing `calibration.json`
   handling at lines 338-341.

4. Per the `.magnitu` rule, adding files changes the archive layout: bump
   `manifest_format_version` from 1 to 2 and make the reader tolerate 1
   (sidecars absent) so old packages keep importing.

**Acceptance:**

- A window report for a trained desk carries `ece` and
  `temperature_clamped`.
- Export → import round-trip reproduces `display_score` identically (the
  isotonic sidecar survives).
- A `manifest_format_version: 1` package still imports, with
  `display_score == relevance_score`.

---

## WP10 — Guard the `recipe_quality` evaluation set

**Verified.** `distiller.evaluate_recipe_rank_metrics` (line 1028) uses
`db.get_all_entries()[:sample_size]` — the 100 newest entries, since
`get_all_entries` orders by `published_date DESC`. `_optimize_recipe_caps`
selected the caps on **labeled** entries, so the caps are tuned on one
distribution and graded on another.

Worse, `score_entries` caps on-the-fly embedding at `MAX_ONTHEFLY_EMBEDDINGS`
(10), and the newest entries are exactly the ones most likely to lack cached
embeddings. When that happens, Spearman is computed on ~10 rows and
`recipe_quality_floor` (0.30) withholds the recipe push from Seismo on that
basis. `train_tfidf_student` has an explicit guard for this exact failure (the
P1-3 `min_distill_scores` check, `pipeline.py:3113`); the quality evaluator has
none.

**Changes — `distiller.py`:**

1. `evaluate_recipe_rank_metrics`: after building `model_score_map`, require
   coverage before trusting the number.

```python
min_paired = max(30, sample_size // 3)
if len(paired_model) < min_paired:
    logger.warning(
        "Recipe quality evaluated on only %d / %d sampled entries (min=%d); "
        "embeddings likely not cached. Reporting quality_unavailable.",
        len(paired_model), len(entries), min_paired,
    )
    out = dict(empty)
    out["n_eval"] = len(paired_model)
    out["quality_unavailable"] = True
    return out
```

2. Return `n_eval` and `quality_unavailable` in every path.

3. `ml_window._hold_recipe_below_floor`: when
   `metrics.recipe_quality_unavailable` is true, do **not** hold the recipe on
   a below-floor score — an unmeasurable recipe is not a bad recipe. Log
   `recipe_quality_unavailable` to `sync_log` instead and push. Holding on a
   10-row Spearman is the current behaviour and is the wrong default.

4. Sample the eval set from a stable slice rather than "whatever is newest":
   keep `published_date DESC` ordering but skip entries without a cached
   embedding when selecting, so the 100 rows are 100 *scorable* rows. Record
   `n_eval` so a shortfall is visible.

**Acceptance:**

- With fewer than `min_paired` scorable entries, `recipe_quality` is reported
  as unavailable and `_hold_recipe_below_floor` returns `False`.
- With full embedding coverage, the returned Spearman is unchanged from today
  (regression guard in `test_recipe_quality.py`).
- `n_eval` appears in `recipe["metrics"]`.

---

## WP11 — Re-unify the embedding representation (operational, needs a window)

**Verified from git history.** Commit `78698cb` (2026-08-31) raised the content
cap for `substack` / `scraper` from 3000 to 7000 chars and the chunk budget
from 4 to 6 (`pipeline._content_cap_for_entry` line 1260,
`embed_entries` line 1372, `magnitu/entry_preview.ANALYTICAL_SOURCE_TYPES`),
after `EMBEDDING_STACK_GENERATION` was already pinned at `e5-v3` (`1591763`) —
with no generation bump, no `embedding_analytical_content_cap` in
`config.DEFAULTS`, and therefore no trigger in `main._migrate_config` (which
does check `embedding_content_cap` and `embedding_legal_content_cap`).

`config.py:84-85` says "Bump when backbone, prefix rules, or content/token caps
change (triggers re-embed)." Cached embeddings are only invalidated when entry
text changes, so every Substack entry embedded before that date still carries a
3000-char / 4-chunk vector while later ones carry 7000 / 6 — two different
representations of the same source type in one feature space, on the sources
that carry the most analysis.

Nothing at scoring time compares `models.embedding_stack_generation` against
the current config either; only the `embedding_l2_normalize` flag travels with
the model. And `ml_window.py` never imports `main`, so `_migrate_config` never
runs on the VPS: a generation bump takes effect only when someone opens the web
UI, and when it does it deactivates every model, so the next window pushes
nothing until each desk retrains.

Two further consistency bugs in the same area, fix while here:

- For legal entries the content cap is 12000 but the chunk budget is
  6 × 1800 = 10800 chars, so the last ~1200 chars of the configured cap are
  never embedded.
- Only the first chunk carries the natural-language source context and the
  doubled title (`_build_entry_text`), so on long statutory documents the title
  signal that function deliberately repeats is diluted to roughly 1/6 by the
  length-weighted mean in `embed_entries`.

**Changes:**

1. `config.py`: add `"embedding_analytical_content_cap": 7000` to `DEFAULTS`
   and bump `EMBEDDING_STACK_GENERATION` to `"e5-v4"`.

2. Move the stack-currency check out of `main.py` into
   `pipeline.ensure_embedding_stack_current()` (pipeline already imports `db`
   and `config`, so there is no cycle; `main` imports `pipeline`). Add
   `embedding_analytical_content_cap` to its trigger list.
   `main._migrate_config` delegates to it; `ml_window.main()` calls it once
   right after `enforce_embedding_store_cap()`.

3. `pipeline.embed_entries`: derive `max_chunks` from the entry's content cap
   rather than from three constants, so the chunk budget always covers the cap:
   `max_chunks = max(1, -(-content_cap // EMBED_CHUNK_CHARS))`, clamped to a
   ceiling. Keep `EMBED_CHUNK_CHARS = 1800`.

4. `pipeline._build_entry_text` / `embed_entries`: prepend the context block and
   the doubled title to **every** chunk, not just the first, so the length
   weighting cannot dilute them. This is the only behavioural change to the
   embedded text and is the reason the generation bump is needed anyway.

**Sequencing — this WP is operational, not just code.** Bumping the generation
invalidates every embedding and deactivates every model, so all desks stop
receiving scores until they re-embed and retrain. Land it in its own release,
during a maintenance window, with `MAGNITU_ML_FULL_ENTRY_DRAIN=1` and
`MAGNITU_ML_FORCE_RETRAIN=1` on the first window afterwards, and expect the
first post-bump window to take hours on CPU. Do **not** bundle it with any
other WP.

**Acceptance:**

- `ensure_embedding_stack_current()` is called by both `main` startup and
  `ml_window.main()`; a test asserts the second call is a no-op.
- A 12000-char legal entry produces chunks covering all 12000 chars.
- Every chunk of a multi-chunk entry contains the source-context prefix.
- After the re-embed, `SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL`
  matches the entry count and every desk has a fresh promoted model.

---

## WP12 — Small wiring bugs (independent, low risk)

1. **Constant sample weights are silently discarded.**
   `pipeline._transformer_fit_kwargs` (line 639) returns `{}` when
   `std(sw) <= 1e-6`. If every label on a desk is a confirmed Gemini label,
   `synthetic_label_weight: 0.5` produces a constant vector and is dropped —
   while `build_prior_fit` still receives it, so the sidecar's `prior_fit`
   describes weights the fit never used. Change the guard to drop only when the
   vector is all-ones (`np.allclose(sw, 1.0)`), and pass the same effective
   vector to `build_prior_fit`.

2. **Dashboard keywords ignore the profile.**
   `explainer.global_keywords` (line 233) calls `get_feature_importance()` with
   no `profile_id` on the TF-IDF branch, so the learned-phrase panel always
   reports profile 1. Pass the `profile_id` the function already receives.

3. **Binary student crashes the distiller.**
   `distiller._distill_from_transformer` (line 583) iterates
   `enumerate(class_names)` against `coef_matrix[i]`. A student that ends up
   with two classes has `coef_` of shape `(1, n_features)` and raises
   `IndexError` on the second class. Guard with
   `if coef_matrix.shape[0] == 1` and expand to
   `[-coef, +coef]` aligned to `classifier.classes_`, mirroring
   `pipeline._as_2d_logits`.

4. **Displayed probabilities do not reproduce the displayed score.**
   `_score_transformer` (line 2696) rounds `relevance_score` and
   `probabilities` independently, so the per-class numbers Seismo shows do not
   sum to the composite next to them. Compute the composite from the rounded
   probabilities, or document the discrepancy in the explanation payload.
   Cosmetic; pick one and be consistent with `_score_tfidf` (line 2945), which
   has the same pattern.

**Acceptance:** one focused test per item; `test_training_knobs.py` gains a
case for the all-Gemini constant-weight path.

---

## Final verification (after all WPs except WP11)

1. `python -m pytest -x -q` — full suite.
2. Gate replay: run the existing promote-gate replay script against each desk's
   history and confirm no decision flips in a direction the WP did not intend
   (WP1 should reject previously-promoted regressions on small reserves; WP2
   should promote some previously-rejected candidates).
3. Push smoke on one desk: train → push → confirm the window report carries
   `gate_k`, `scores_pushed`, `scores_missing`, `ece`,
   `temperature_clamped`, and `recipe_level`.
4. Confirm `explain_entry(entry)["prediction"]` equals the pushed
   `predicted_label` for 20 sampled entries spanning short news and long lex
   bodies.
5. Record the WP6 level numbers per desk in the PR description and open the
   Seismo-side issue if WP7's third branch applies.

## Explicitly out of scope (do not attempt)

- Any change to Seismo's PHP, the recipe/score JSON contract's scoring keys, or
  `scoreEntryWithRecipe` semantics. WP7's third branch is filed as a Seismo
  issue, not implemented here.
- Re-introducing rank normalization at push time (removed 2026-09-01; it masked
  WP6 rather than fixing it).
- Raising `EVAL_RESERVE_MAX_FRACTION` or lowering `min_labels_to_train` to make
  the gate work — WP1 fixes the metric instead.
- Modelling Seismo's Swiss-dictionary keyword expansion in Python.
- Re-introducing the score-drift EWMA monitor removed in `d714f5b`. WP6's
  level metrics cover the same ground with less machinery; revisit only if they
  prove insufficient.
- Changing `CLASS_WEIGHT_MAP`, `classifier_c`, or `classifier_apply_prior`
  defaults. Those were validated in the D2/D3 evaluations and any change needs
  its own evaluation round, not a bug-fix WP.
