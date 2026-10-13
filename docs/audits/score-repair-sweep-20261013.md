# Repair sweep and promotion fairness — 13 October 2026

The 17:07 UTC score-only window admitted 50 EU repair keys, acknowledged 36 and
deferred 14 for the embedding budget. Exact readback of all 50 keys at 18:06 UTC
found Magnitu model 49 for every key: 36 retained their 17:08:46 delivery clock,
ten were delivered at 17:23:43 and four at 17:42:46. The companion JSON retains
the original batch, inferred deferred subset and exact score metadata; no bodies
or credentials are stored. All 14 deferred deliveries recovered.

At 18:06, all six sweeps retained their original 12:38–12:39 UTC fences and models.
Root, Digital, Security and Finance had completed four source lanes with cached
feeds remaining. EU and Energy had completed lex and calendar; three lanes
remained. No complete sweep has yet been observed. Completion monitoring must
read both full and score-only service journals and distinguish an actual
`completed=1` event from a changed fence, expiration or manual reset. Queue
emptiness and source rows aging out do not prove completion or delivery.

An isolated regression demonstrated that the previous model-version mismatch
returned an empty cursor and queue, so repeated promotions could restart the
prefix and discard known repair retries. The correction carries the existing
cursor and bounded queue into the new active model on admission, preserves retry
ages and progress clocks, and rejects stale-model acknowledgements. A fixture
with six consecutive promotions verifies cursor and pending-key continuity;
a worker integration fixture verifies prediction uses the new active model.
These are isolated tests; no production model was forced or promoted for proof.

Future promotions still require measured capacity: preserving traversal does
not mean every score always matches the newest model. Completed/expired source
scans restart at the current horizon and revisit the earlier prefix.
