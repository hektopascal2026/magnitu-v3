# Repair admission backpressure — 29 October 2026

Requested investigation: EU remains at the 200-key cap; make the remaining ML
backlog drain without clearing paid/durable work or masking warnings.

## Verified baseline and cause

Local clean worktree `/private/tmp/magnitu-vps-repair-20261029` started at
`21d82e8f6c4c6d50cc87c3288fd752fa208a143d`, a documentation-only descendant
of installed worker `2889432c744b9328da2fed85074b030540949c94`. Original checkout
was preserved. Before editing, installed hashes matched local ml_window.py,
score_repair_scan.py, score_text_scan.py and score_delivery.py. Seismo stayed at
8.5.8 `d4b197dbf843bdf0f46e1c660043918e41c67e5f`.

At 21:46 UTC, EU/52 and Energy/5 both had 200 pending keys; Finance/3 had 66.
Every retained cause was embedding_budget, not a delivery/transport failure.
EU's 21:41:17 page delivered ten retries, reducing 200 → 190. Later pages,
after its ten-embedding allowance was spent, admitted 10 + 3 + 1 new keys,
delivered only 3 + 1 + 0, and ended at 200 again. Row headroom was wrongly
treated as capacity to process new unknown embedding work. Prior successful
cohort deliveries were real, but did not establish net backlog reduction.

The shared embedding phase is active: the prior normal window stored 200
vectors under its existing 150-second admission deadline. A read-only exact-ID
comparison of one queued row found no changed scoring fields, excluding
refetch invalidation for that sample. No global cache clearing was observed.

## Narrow repair

Within the existing page/queue headroom, reserve remaining on-the-fly capacity
for the next uncached durable retry batch. When a backlog exists, new unknown
keys may use only the remaining unreserved allowance. After the allowance is
spent, discovery performs no new fetch and does not advance the source cursor;
cached durable retries still proceed. Fresh delivery remains independent and
cached-only discovery with an empty backlog retains four full fifty-key pages.

The actual post-refetch embedding check remains authoritative: a body revision
that invalidates a previously cached vector can still defer new keys safely.
Nothing is marked acknowledged until a successful source/score acknowledgement.
Pending keys, ages, retry rotation, model adoption and actual failure states
remain intact. No cap, model, prompt, deadline, concurrency or wire format
changes. Shared embedding remains 200; repair embedding remains ten/profile/
window; repair queue remains 200 and each page at most fifty.

## Verification

Regressions cover a saturated uncached queue with continuous new discovery
(200 → 190 → 180 → 170 across three windows), preservation of the held cursor,
cached retry delivery after budget exhaustion, reservation for mixed cached/
uncached retries, and post-refetch invalidation. Existing acknowledgement,
poison rotation, partial failure, model transition and all-cached throughput
tests remain required. Local/native gates and actual activation/drain receipts
are recorded after execution. These tests use isolated MAGNITU_DATA_DIR stores,
fake HTTP and local scoring stubs, with no provider or paid model calls.

Rollback is a forward revert while both ML units are fully idle. Never reset
the repair queues/cursors or delete checkpoints to make telemetry green.
