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

## First activation and shared-allocation follow-up

PR 32 passed CI and 187 local/native tests plus 33 sync-contract checks. Merge
`b09174f59e537c6e3a047cd4f5eaf26e83214940` exactly matched tested candidate
`93347d21ce60785c5565653a88f20533b894b329`; all seven changed live files were
verified before/after the clean fast-forward under the shared deploy lock.
The first normal score-only window reduced EU 195 → 185, Energy 200 → 187,
Finance 67 → 27, exit 0. No keys or checkpoints were reset. Activation and
original-key cohorts are retained privately in the VPS release-receipts folder.

During that run, most original EU/Energy keys still lacked vectors after the
shared phase. Its newest-first allocation made older durable work rely largely
on ten inline embeddings. A separate allocation fix reserves **at most 100 of
the existing 200 shared slots**, in profile round-robin order, skipping cached,
missing and duplicate keys. The remainder fills by the existing newest order.
Complete entry bodies and metadata are preserved; selection uses narrow id
probes before loading the selected full rows. No embedding/model arithmetic
changes. Existing 150-second admission deadline, ten-entry commit sub-batches,
four repair pages and ten retry keys/page remain. The effective score-only unit
retains its 3.5-GiB memory limit; the full-training unit is separate.

The standalone catch-up launcher was paused by exact PID/start-time identity
before this follow-up; the running scoring service was allowed to finish.
The original acceptance cohort is retained across both code revisions.
Tests cover cross-profile fairness, shared-key reuse, cached/missing exclusion,
invalid durable-state visibility, bounded payloads, body preservation, ordering,
total-budget forwarding and partial embedding commits.

## Allocation activation and observed drain

PR 33 passed [CI](https://github.com/hektopascal2026/magnitu-v3/actions/runs/40963891409)
and **196 tests locally and on native Python 3.14**, with existing NumPy/sklearn
warnings; the 33 local sync-contract checks also passed. Merge
`69330a6ba40f01c4abcb747dcb896d1b42967748` exactly matched the tested tree of
`a446f60cbc3ee7e1072abae48f3c7fe0b195f493`. It was activated at **22:15:01 UTC**,
with both ML units fully idle, under the shared deployment lock, after verifying
all eight changed files before/after the clean fast-forward. Seismo stayed on
8.5.8. No paid model or editorial calls, new training or mail sending.

Normal bounded score-only catch-up windows retained the original acceptance
cohort from first activation: EU/52 195, Energy/5 200, Finance/3 67. No durable
keys, cursors or checkpoints were reset.

| Allocation-patch window | EU pending | Energy pending | Finance pending |
| --- | ---: | ---: | ---: |
| Before | 172 | 174 | 23 |
| 1 | 162 | 164 | 23 |
| 2 | 132 | 134 | 0 |
| 3 | 92 | 94 | 22 |

All three completed successfully, with no reported repair failures. Finance's
22 in the third window are newly discovered keys: none belongs to its original
cohort. A strict indexed source-database lookup independently confirmed all
67 original Finance keys had model 3 Magnitu scores stamped after first
activation. The first window committed 110 shared vectors before its existing
admission deadline; the second committed 180; the third committed all 121 then
missing. EU's complete original cohort had cached vectors by the third window.
That cache progress alone is not acknowledgement or complete-cohort acceptance.

The continuing original-cohort delivery/queue acceptance is recorded in the
[Seismo operator receipt](https://github.com/hektopascal2026/seismo_1.0/blob/main/docs/audits/ml-repair-drain-20261029.md).
Private activation manifests, original key lists, per-window receipts and strict
source-score counts remain under `/var/lib/seismo/release-receipts/` on the VPS.
The operator receipt distinguishes current queues from completed original keys
and does not claim sustained future capacity or complete historical-sweep coverage.
