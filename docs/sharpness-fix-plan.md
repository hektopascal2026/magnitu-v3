# Plan: Sharper scores — rank normalize, synthetic down-weight, chunk pooling

## WP1 — Percentile-rank scores at push time

**Where:** `main.py` only. Local UI and distiller cap optimization keep raw composites.

Add `_rank_normalize_push_scores(scores)` — replace `relevance_score` with mean-rank / N (ties share mean rank). Call in `_sync_push_impl` after explanations, before `push_scores`. Do not touch `predicted_label`, `probabilities`, or `explanation`.

## WP2 — Down-weight Gemini labels in training

**Where:** `pipeline.py` `compute_sample_weights` + `train_tfidf_student`.

- Config key `synthetic_label_weight` (default `0.5`, range 0–1; `1.0` = legacy behavior).
- Multiply weight when `label_source == "Gemini"`.
- Distillation: human hard labels use `HUMAN_DISTILL_WEIGHT * syn_w` for Gemini rows.

Add to `config.py` DEFAULTS and `PROFILE_TRAINING_SETTINGS_KEYS`.

## WP3 — Chunk pooling for long documents

**Where:** `pipeline.py` `embed_entries` only.

Constants: `EMBED_CHUNK_CHARS=1800`, `MAX_EMBED_CHUNKS=4`, `MAX_EMBED_CHUNKS_LEGAL=6`.

`_split_text_chunks(text, chunk_chars, max_chunks)` — ≤max_chunks windows, snap cuts to whitespace.

`embed_entries`: flatten chunks → one `compute_embeddings` call → length-weighted mean per entry.

Short texts (≤1800 chars) → one chunk → identical to today. No embedding-stack bump; recompute optional for long lex bodies.

**Tests:** `test_chunk_pooling.py` — split, pool, rank normalize, synthetic weights.
