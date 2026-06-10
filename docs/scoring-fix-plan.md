# Implementation Plan: Magnitu scoring & recipe distillation fixes

Audit date: 2026-06-10. Source: code audit of `pipeline.py`, `distiller.py`,
`explainer.py`, `sync.py`, `main.py`, `db.py` against Seismo 0.6
(`src/Core/Scoring/RecipeScorer.php`, `src/Core/Scoring/ScoringService.php`).

## Global constraints (apply to every work package)

- Python 3.9 only: `Optional[X]`/`typing`, no `X | Y` unions.
- Do **not** change: the recipe JSON schema, the score push payload shape,
  `entry_type` values, any Seismo PHP code, any HTTP behavior outside `sync.py`.
- Do **not** rename existing public functions. Add new helpers; rewrite bodies
  in place.
- After each WP, run
  `python -m pytest test_magnitu2.py test_training_knobs.py test_stable_split.py -x -q`
  and fix regressions before moving on.
- WPs are ordered; implement in this order. WP2 is a dependency of WP3, WP5, WP6.

---

## WP1 — Per-profile model artifacts (critical bug)

**Problem:** model versions are per-profile (`db.get_next_model_version(profile_id)`),
but files are named `model_v{version}.joblib` / `recipe_v{version}.json` in a single
shared `MODELS_DIR`. Two profiles overwrite each other's files: profile B's
`model_v1.joblib` clobbers profile A's, and profile A then silently scores with
profile B's classifier.

**Changes:**

1. `pipeline.py`, in `_train_transformer` (around line 1322) and `_train_tfidf`
   (around line 1582), change:

```python
model_filename = "model_v{}.joblib".format(version)
```

to:

```python
model_filename = "model_p{}_v{}.joblib".format(profile_id, version)
```

2. `model_manager.py` (around lines 319–333), same pattern:
   `model_v{}.joblib` → `model_p{}_v{}.joblib` and
   `recipe_v{}.json` → `recipe_p{}_v{}.json`, using the `profile_id` already in
   scope (`store_version = db.get_next_model_version(profile_id)` is right above).

3. `distiller.py`, in `distill_recipe` (around line 252):

```python
recipe_filename = "recipe_p{}_v{}.json".format(profile_id, model_info["version"])
```

4. `distiller.py`, fix the unscoped UPDATE (around lines 258–263):

```python
conn.execute(
    "UPDATE models SET recipe_path = ? WHERE version = ? AND profile_id = ?",
    (recipe_path, model_info["version"], profile_id)
)
```

**Notes:** No migration needed — all loads go through `model_path`/`recipe_path`
stored in the `models` table, so existing rows keep working. The calibration
sidecar path is derived from the model path (`calibration_sidecar_path`), so it
scopes automatically.

**Acceptance:** call the `db.save_model_record` flow via two trainings with
`profile_id=1` and `profile_id=2` (each version 1); assert the two `model_path`
values differ and both files exist.

---

## WP2 — PHP-parity recipe scorer (foundation for everything else)

**Problem:** `distiller._recipe_composite`, `_normalize_weights`,
`_optimize_recipe_caps`, and `evaluate_recipe_quality` simulate Seismo with
(a) every-occurrence keyword counting, (b) a different tokenizer, (c) full
`title+description+content` text for all entry types. Seismo's
`RecipeScorer.php` counts each keyword **once per document**
(`MAX_HITS_PER_TOKEN = 1`), keeps 1-char tokens and accents, splits on a
specific delimiter class, and scores lex/calendar entries on **synopsis only**
(`ScoringService.php`). Consequence: `_normalize_weights` calibrates to a
repeat-counted magnitude, so real Seismo logits land far lower, softmax
flattens, and scores cluster at the 0.4975 no-signal attractor. Cap
optimization and the reported recipe quality optimize/measure the wrong
objective.

**Changes — all in `distiller.py`:**

1. Add module-level constants and two helpers:

```python
# Mirrors RecipeScorer.php tokenizer: split on anything outside this class.
_SEISMO_WORD_RE = re.compile(
    r"[^a-zA-Z0-9äöüàéèêïôùûçÄÖÜÀÉÈÊÏÔÙÛÇß]+"
)
_SEISMO_MAX_NGRAM = 3  # RecipeScorer::MAX_NGRAM


def _seismo_tokenize(text: str) -> list:
    """PHP-parity word split: lowercase, keep 1-char tokens and accents."""
    return [w for w in _SEISMO_WORD_RE.split((text or "").lower()) if w]


def _seismo_tokens(text: str) -> list:
    """Unigrams through trigrams, space-joined, PHP-parity."""
    words = _seismo_tokenize(text)
    out = []
    n = len(words)
    for i in range(n):
        chunk = words[i]
        out.append(chunk)
        for span in range(2, min(_SEISMO_MAX_NGRAM, n - i) + 1):
            chunk = chunk + " " + words[i + span - 1]
            out.append(chunk)
    return out
```

2. Add an entry-text helper mirroring `ScoringService.php`:

```python
def _seismo_score_text(entry: dict) -> str:
    """Text Seismo's PHP actually scores, per entry family.

    feed_item/email: title + (content or description).
    lex_item/calendar_event: title + description ONLY (synopsis;
    full content is for Magnitu export, not PHP recipe scoring).
    """
    et = (entry.get("entry_type") or "").strip()
    title = entry.get("title") or ""
    if et in ("lex_item", "calendar_event"):
        body = entry.get("description") or ""
    else:
        body = entry.get("content") or entry.get("description") or ""
    return title + " " + body
```

3. Add a keyword-key normalizer mirroring `normalizeKeywordKey()`
   (hyphens → spaces etc.):

```python
def _normalize_recipe_key(keyword: str) -> str:
    return " ".join(_seismo_tokenize(keyword))
```

4. Rewrite `_recipe_composite` as a faithful port of `RecipeScorer::score`:
   - Build a lookup `{_normalize_recipe_key(k): class_weights}` from
     `keywords` (merge duplicate normalized keys by summing per-class weights,
     same as PHP).
   - `text = _seismo_score_text(entry)`, `tokens = _seismo_tokens(text)`.
   - **Once-per-token:** keep a `set` of matched tokens; each recipe key
     contributes at most once.
   - Source weights, softmax (max-subtracted), composite `sum(prob * class_wt)`
     — keep as is.
   - Do **not** model the Swiss dictionary expansion; add a one-line comment
     saying PHP additionally expands keywords via `swiss_dictionary.json`
     (acceptable conservative divergence).

5. In `_normalize_weights`, replace the token-matching block (lines ~286–309)
   to use the same logic: `_seismo_score_text(entry)` + `_seismo_tokens` +
   once-per-token set + normalized keyword lookup. Keep the
   median/target/scale logic unchanged.

6. `evaluate_recipe_quality` and `_optimize_recipe_caps` need no internal
   change beyond what they inherit from the new `_recipe_composite`
   (WP5 changes `_optimize_recipe_caps` further).

7. In `explainer.py`, `_recipe_phrase_contributions` has its own copy of
   `_extract_recipe_tokens`: change it to import and use `_seismo_tokens` and
   `_seismo_score_text` from `distiller` so explanations match what Seismo
   matches. Match each phrase once (use a set).

**Keep** the old `_tokenize_text` / `_compose_ngrams` / `_extract_recipe_tokens`
in `distiller.py` — they are still used by `_boost_from_reasoning`. Do not
delete them.

**Acceptance (new file `test_recipe_parity.py`):**
- A keyword appearing 5× in one entry contributes its weight exactly once.
- Entry `{"entry_type": "lex_item", "title": "t", "description": "third country",
  "content": "member states only"}`: "third country" matches,
  "member states only" does not.
- `_seismo_tokenize("E-Commerce-Verordnung")` == `["e", "commerce", "verordnung"]`;
  bigram `"e commerce"` is produced.
- `_normalize_recipe_key("Third-Country")` == `"third country"`.
- Accented text matches accented keyword: keyword `"überwachung"` matches text
  `"Überwachung"`.

---

## WP3 — Stop stripping accents from recipe keywords

**Problem:** `strip_accents="unicode"` in the TF-IDF vectorizers produces
keywords like `uberwachung` that can never match accented document text in
Seismo's PHP (which preserves umlauts/accents in both document tokens and
keyword keys). This silently kills most German/French distilled signal.

**Changes — `pipeline.py`:** remove `strip_accents="unicode"` (delete the line,
leaving the sklearn default `None`) in all three places:

1. `build_tfidf_pipeline` (line ~1000)
2. `_relaxed_tfidf_vectorizer` (line ~1035)
3. The inline fallback `TfidfVectorizer` inside `_train_tfidf`'s
   `except ValueError` block (line ~1553)

Leave `token_pattern` at the sklearn default. Recipe keys are reconciled at
export time via `_normalize_recipe_key` from WP2.

**Acceptance:** retrain the TF-IDF student on a tiny synthetic corpus
containing "Überwachung" labeled rows; assert `get_feature_names_out()`
contains a feature with `"ü"`.

---

## WP4 — Fix `_boost_from_reasoning` (compounding, sign, confirmed-only)

**Problem:** the boost multiplies once per reasoning row containing a token →
×1.5ⁿ exponential blowup across rows; multiplying a **negative** coefficient
makes it more negative (opposite of user intent); pending Gemini labels'
reasoning leaks into recipes; phrases use 1.8 instead of the documented 1.5.

**Changes:**

1. `db.py`, `get_all_reasoning_texts` (line ~1005): add the confirmed filter to
   the WHERE clause:

```python
WHERE profile_id = ? AND reasoning IS NOT NULL AND reasoning != ''
      AND """ + _labels_confirmed_sql("") + """
```

2. `distiller.py`, rewrite the `_boost_from_reasoning` body with this exact
   algorithm (keep the signature):

```python
BOOST_FACTOR = 1.5  # single documented multiplier; applies to unigrams AND phrases

# Pass 1: collect distinct (token, label) pairs. Dedupe within and across
# reasoning rows so the boost is applied exactly once per pair.
pairs = set()
for rl in reasoning_labels:
    reasoning = rl.get("reasoning", "")
    label = rl.get("label", "")
    if not reasoning or not label:
        continue
    tokens = _tokenize_text(reasoning)
    for token in set(tokens + _compose_ngrams(tokens, max_n=3)):
        if " " not in token and len(token) < 3:
            continue
        pairs.add((token, label))

# Pass 2: apply once per pair.
for token, label in pairs:
    existing = keywords.get(token, {}).get(label)
    if existing is not None and existing > 0:
        new_w = existing * BOOST_FACTOR          # amplify supporting signal
    else:
        # Negative or absent coefficient: reasoning explicitly supports this
        # label, so seed a positive base instead of amplifying a negative.
        new_w = 0.16 if " " in token else 0.10
        if existing is not None:
            new_w = max(existing, new_w)
    keywords.setdefault(token, {})[label] = round(new_w, 4)
```

**Acceptance:**
- Token in 10 reasoning rows with the same label → weight boosted exactly once
  (not ×1.5¹⁰).
- Pre-existing weight −0.2 for (token, label) with supporting reasoning →
  becomes +0.10 (unigram), never −0.3.
- A label row with non-empty `pending_gemini_job_id` contributes nothing.

---

## WP5 — Optimize the recipe that actually ships (floor inside cap search)

**Problem:** `_optimize_recipe_caps` grid-searches caps for best model↔recipe
correlation, then `_apply_floor_weights` mutates the recipe **afterwards** — so
the reported `export_caps.quality` describes a different artifact than the one
pushed.

**Changes — `distiller.py`:**

1. In `_optimize_recipe_caps`, inside the triple loop, after
   `t_kw, t_sw = _stabilize_export_weights(...)`, add:

```python
t_kw = _apply_floor_weights({k: dict(v) for k, v in t_kw.items()})
```

(The dict-copy is required: `_apply_floor_weights` mutates its argument; trials
must not contaminate each other.)

2. In `distill_recipe`, the final `keywords = _apply_floor_weights(keywords)`
   call (line ~225) stays — it is now idempotent for the optimized path (floors
   already applied to `best_kw`) and still needed for the
   `_stabilize_export_weights`-only path.

3. Update the long comment above the `_apply_floor_weights` call in
   `distill_recipe` to state floors are now part of the optimization objective.

**Acceptance:** unit test asserting the keywords dict returned from
`_optimize_recipe_caps` already satisfies
`keywords["member states only"]["investigation_lead"] >= 0.55`.

---

## WP6 — Push-flow explanation performance (no per-entry joblib load)

**Problem:** `_sync_push_impl` calls `explainer.explain_entry` per entry;
`_explain_transformer` does `joblib.load` + one SQL query **per entry** —
thousands of model loads per push.

**Changes — `explainer.py`:**

1. Add a module-level classifier cache:

```python
_CLF_CACHE = {"path": None, "mtime": None, "clf": None}


def _load_classifier_cached(model_path: str):
    mtime = Path(model_path).stat().st_mtime
    if _CLF_CACHE["path"] == model_path and _CLF_CACHE["mtime"] == mtime:
        return _CLF_CACHE["clf"]
    clf = joblib.load(model_path)
    _CLF_CACHE.update({"path": model_path, "mtime": mtime, "clf": clf})
    return clf
```

2. In `_explain_transformer`, replace `clf = joblib.load(model_path)` with
   `clf = _load_classifier_cached(model_path)`.

3. Same caching pattern for the recipe JSON in `_recipe_phrase_contributions`
   (cache key: `recipe_path` + mtime).

**Acceptance:** call `explain_entry` twice; assert `joblib.load` runs once
(monkeypatch a counter in a test).

---

## WP7 — Label-sync timestamp hygiene (low risk, contained)

**Problem:** `pull_labels` compares remote `labeled_at` with local `updated_at`
by raw string. ISO `T` separators or timezone suffixes break ordering
(`"2026-06-10T09:00:00" > "2026-06-10 09:59:59"` is true). Also, updating from
remote resets `label_source` to `""`, washing Gemini provenance.

**Changes — `sync.py`, inside `pull_labels` only (Seismo HTTP stays in this
file per rules):**

1. Add a local helper:

```python
def _normalize_label_ts(value: str) -> str:
    """Canonicalize 'YYYY-MM-DD HH:MM:SS' vs ISO-T variants for string compare."""
    s = (value or "").strip().replace("T", " ")
    if "+" in s:
        s = s.split("+", 1)[0]
    return s.rstrip("Z").strip()
```

2. Compare with
   `_normalize_label_ts(remote_time) > _normalize_label_ts(local_time)`.
   Tie (equal) keeps local — unchanged behavior, now documented in the
   docstring.

3. Preserve provenance on remote update: `existing` already contains
   `label_source` (see `db.get_label_with_reasoning`). Change the update call
   to:

```python
db.set_label(entry_type, entry_id, label, reasoning=reasoning,
             profile_id=profile_id,
             label_source=existing.get("label_source", ""))
```

**Acceptance:** remote `"2026-06-10T08:00:00"` does **not** beat local
`"2026-06-10 09:00:00"`; a local Gemini-sourced label updated from remote keeps
`label_source == "Gemini"`.

---

## WP8 (optional, last) — Flag resubstitution metrics

In `_train_transformer` and `_train_tfidf`, in the `min_class_count < 2`
branch, change `split_note` to:

```python
split_note = "RESUBSTITUTION: train==test (some classes have <2 samples); metrics are optimistic"
```

No schema changes; the note already flows into the train result and UI.

---

## Final verification (run after all WPs)

1. `python -m pytest -x -q` — all tests including the new
   `test_recipe_parity.py`.
2. Smoke: train (any profile), then `distiller.distill_recipe(profile_id=1)`;
   assert the recipe file is `recipe_p1_v*.json`, contains `export_caps`, all
   `LEGAL_TEMPLATE_PHRASES` at ≥ seeded weights, and at least one accented
   keyword if the corpus has accented labeled text.
3. Manual diff: `evaluate_recipe_quality` before/after on the same DB — expect
   the number to *drop* (it was previously measured against a flattering, wrong
   proxy); that is expected and correct. Record both values in the commit
   message.

## Explicitly out of scope (do not attempt)

- Modeling Seismo's Swiss-dictionary keyword expansion in Python.
- Changing Seismo's PHP, the recipe/score JSON contracts, or
  `scoreEntryWithRecipe` semantics.
- Changing the `_normalize_weights` target value or cap grid values.
- Any timezone fix on the Seismo server side (`labeled_at` fill-in) —
  Magnitu-side normalization only.
