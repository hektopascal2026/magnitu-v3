# Magnitu v3 — Gemini Synthetic Scorer & Model Library

Working document to implement v3 **in order**, without destabilizing Magnitu 2 behavior. Check boxes as you complete work; add dates or PR links in notes if useful.

## Goals

- Accelerate the learning loop: **Gemini** acts as a synthetic journalist—auto-label entries with **reasoning notes** so the system can reach **100+ labels** with less manual work.
- Add a **Model Library** (multiple `.magnitu` profiles) and, later, **per-profile Seismo** recipe pushes.

## Constraints (do not break)

- **Python 3.9** — use `Optional`, `Union`, etc.; no `X | Y` syntax.
- **Seismo API contract** — keep HTTP to Seismo in **`sync.py`** (single point of contact). Do not change score/recipe/label JSON shapes without updating Seismo’s PHP engine.
- **Label classes** (exact strings): `investigation_lead`, `important`, `background`, `noise`.
- **Entry types** (exact): `feed_item`, `email`, `lex_item`.

## How to use this file

1. Complete **Phase 1** before wiring UI that calls Gemini.
2. Within a phase, work top to bottom unless a task explicitly depends on another.
3. When a slice is done, fill **Acceptance** and note any follow-ups under **Deferred**.

---

## Phase 1 — Integration core (the bridge)

**Objective:** Port **GeminiClient** + **JSON repair / lenient parse** into Magnitu; add **prompt factory** and **Swiss persona** system prompt.

### Tasks

- [x] **1.1** Locate reference code (interview project): `mac/v3/gemini.py`, `mac/v3/extraction/prompts.py` — extract REST patterns, not project-specific cruft.
- [x] **1.2** Add `GeminiClient` using **Google AI v1beta** REST with `responseMimeType: "application/json"` and **response schema** enforcement where supported.
- [x] **1.3** Implement **parse_json_lenient** (or equivalent) for truncated / slightly invalid JSON from the model.
- [x] **1.4** Create **`magnitu/prompts.py`** (or `prompts/` package) with:
  - [x] **1.4a** A **prompt factory** mapping the four Magnitu categories ↔ structured Gemini request (single source of truth for enum ↔ display names).
  - [x] **1.4b** **System prompt**: Swiss economic / trade analyst persona — surface **exclusion language** (e.g. EU/EEA-only, Binnenmarkt, Drittstaaten) that affects Switzerland even when “Switzerland” is not in the text.
- [x] **1.5** Define the **Gemini output contract**: `{ "label": "<enum>", "reasoning": "<string>" }` — validate after parse; reject or retry on bad labels.
- [x] **1.6** Configuration: API key / env var (document in `magnitu_config.example.json` or existing config pattern); no secrets in git.
- [x] **1.7** **Tests**: unit tests for JSON repair + label validation (mock HTTP if needed).

### Acceptance

- [x] Can call Gemini from Python with **strict JSON** path + **fallback repair**, returning validated `label` + `reasoning`.
- [x] No new Seismo endpoints required for this phase.

### Notes / files (fill as you implement)

| Item | Location |
|------|----------|
| Client module | `magnitu/gemini.py`, `magnitu/gemini_config.py` |
| Prompts | `magnitu/prompts.py` |
| Entrypoint (validate + one retry for empty `investigation_lead` reasoning) | `magnitu/synthetic_scorer.py` |
| Tests | `test_gemini.py` |
| Env | `.env.example` (`GEMINI_*`) |

---

## Phase 2 — “Gemini” tab (synthetic labeling)

**Objective:** Dedicated UI for **automated** labeling beside manual **Label**; batch processing with **feedback loop** into `labels.json`.

### Tasks

- [x] **2.1** Navigation: add **“Gemini”** tab next to **“Label”** (same patterns as existing templates / `main.py` routes).
- [x] **2.2** **Target header**: show **active model profile** name (ties to Phase 3; can stub with current single profile first if needed).
- [x] **2.3** **Batch scoring**: pull entries from **“Uncertain”** and/or **“New”** queues (reuse existing queue logic; document which queues).
- [x] **2.4** For each entry: call Phase 1 pipeline → `predicted_label` + `reasoning_note`.
- [x] **2.5** **Persist**: save to **`labels.json`** in the **established format** so **Train** / recipe pipeline applies **1.5× reasoning weight** correctly.
- [x] **2.6** **Idempotency / safety**: define behavior when an entry is already labeled (skip, overwrite, or tag source=`synthetic`) — implement consistently.
- [x] **2.7** Surface errors (rate limits, partial batch failure) in UI; allow retry of failed subset.

### Acceptance

- [x] User can run a batch from the Gemini tab and see labels + reasoning persisted for training.
- [x] `labels.json` remains compatible with existing train/recipe code paths.

### Notes (implementation)

- **Queues:** `sampler.get_gemini_synthetic_batch_entries` reuses `get_smart_entries`, then orders **uncertain → new → conflict → diverse** so the batch favours uncertain/new per roadmap.
- **Persistence:** SQLite `labels` table (training reads via `db.get_all_labels`); **`label_source`** column stores `Gemini`; **`db.export_labels`** adds **`source: Gemini`** for `.magnitu` / `labels.json` export. Seismo push payload unchanged (no `source` field).
- **Batch transport:** `POST /p/{slug}/api/gemini/batch` returns `job_id`; worker thread runs `magnitu/synthetic_batch.run_gemini_synthetic_batch_job`; UI polls `GET /api/jobs/{job_id}` (same pattern as sync).
- **Idempotency:** Skip any row that already has a label with empty `label_source` (human). Optional **`replace_gemini`** body flag re-processes rows where `label_source == Gemini` (retry failures).

### Deferred (optional follow-ups)

- _…_

---

## Phase 3 — Model library (registry)

**Objective:** Move from a **single active model** to a **manageable set** of `.magnitu` profiles under `/models`.

### Tasks

- [ ] **3.1** **Library tab**: scan **`models/`** for `.magnitu` archives (zip); list them.
- [ ] **3.2** **Metadata per model** (from **manifest** inside profile or sidecar — agree schema):
  - [ ] Accuracy / F1 (from `manifest.json` or equivalent).
  - [ ] **Label distribution** counts for all four categories (imbalance visible).
- [ ] **3.3** **Set as active**: copy or symlink **`model.joblib`** + **`labels.json`** (and any paired files) so the app uses the selected profile for training/inference UI.
- [ ] **3.4** Refactor **model loading** so “active profile” is a named path, not hard-coded single files.
- [ ] **3.5** Ensure **Gemini tab** “target” reflects active library selection (align with 2.2).

### Acceptance

- [ ] User can switch profiles from Library; dashboard/training use the active profile.
- [ ] Manifest fields are versioned or documented so UI does not show bogus metrics.

### Manifest schema (fill when fixed)

```json
{
  "version": 1,
  "comment": "Define fields when implementing 3.2"
}
```

---

## Phase 4 — Multi-Seismo deployment

**Objective:** Each model profile can push its **recipe** to a **specific Seismo instance**, not one global URL.

### Tasks

- [ ] **4.1** Extend **model profile metadata** with **`seismo_target_url`** (and auth if needed — align with `?api_key=` pattern).
- [ ] **4.2** **Push workflow**: when user triggers **Push recipe**, resolve URL from **active (or selected) profile**, not only global config.
- [ ] **4.3** Keep all push logic in **`sync.py`** — add functions/parameters for per-target base URL; avoid duplicating HTTP in templates.
- [ ] **4.4** **Knowledge distillation bootstrap**: use Gemini to label **50–100** items for **new** models so the student/distillation path has enough variety for a strong day-one keyword recipe (document the flow in README or internal doc).

### Acceptance

- [ ] Recipe push goes to the Seismo instance configured on the profile used for the push.
- [ ] Global default remains a workable fallback only if explicitly desired.

---

## Cross-cutting checklist

- [ ] **Rate limits & batching**: backoff, max concurrent requests, user-visible progress.
- [ ] **Audit trail**: optional `source` / `labeled_by` for synthetic vs manual (local DB or labels file — decide once).
- [ ] **Security**: API keys only via env / local config; never committed.

---

## Reference — Gemini response shape (synthetic scorer)

Target parsed object after repair + validation:

```json
{
  "label": "investigation_lead",
  "reasoning": "Short justification tied to exclusion language or Swiss relevance."
}
```

Map display names ↔ enums in one module (Phase 1 prompt factory).

---

## Session log (optional)

| Date | Phase | What shipped |
|------|-------|----------------|
| 2026-04-20 | 1 | `validate_synthetic_label_output`, `synthetic_scorer.call_gemini_for_synthetic_label`, `test_gemini.py`, `.env.example` Gemini knobs documented |
| 2026-04-20 | 2 | `/p/{slug}/gemini` UI, background batch job, `label_source` / export `source`, `magnitu/synthetic_batch.py` |
