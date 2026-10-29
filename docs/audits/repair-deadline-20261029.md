# Bounded repair deadline reporting — 29 October 2026

Authorized during Seismo VPS patch repair. Installed worker baseline and clean
source: `f4d20cff5695be47e6cbff9af8f6e9d83e0195c7`. Live SHA-256 for
ml_window.py and test_ml_window.py matched that revision. Original local main
at `35f718d` was clean and preserved; this work uses an isolated temporary Git
worktree at the installed source, branch `codex/repair-deadline`.

After Seismo 8.5.5 fixed the statement timeouts, a real score-only window on
29 October at 17:26:39 UTC successfully fetched repair pages but reached its
existing time deadline before scoring profile 5. Durable keys were retained,
yet the expected bound became worker_error/exit 1. That is separate from the
earlier HTTP 503 failures; it does not demonstrate new query timeouts.

The two internal deadline exits now use a typed ScoreRepairDeferred exception.
The ordinary repair handler and full-window initial repair catch only that
type as deferral, stop that desk's repair opportunity, and retain an explicit
deferred log flag. No limits, concurrency, retry counts, source/cursor identity,
admission order, embedding allowance or paid calls changed. Pending/retry cause
and queue-age warnings remain. An ok window is not proof of an empty queue.

Failure to persist the deferral cause or to read queue health remains a real
worker error. Transport, scoring omissions and unacknowledged writes remain
errors. Existing actual exceptions are not replaced by telemetry exceptions.

Offline Python 3.9 regression tests: 160 passed, with one existing NumPy
warning, across the CI worker/repair/checkpoint/delivery/recipe/eval set.
The sync contract script passed 33 checks. An optional broad model integration
script attempted to obtain uncached HuggingFace weights and was stopped;
that model-dependent integration suite is not claimed passed. Tests use a
private MAGNITU_DATA_DIR, no live DB or secrets, and no paid Gemini calls.

Native tests, clean fast-forward deployment while services are idle, source
hash checks and the bounded real score-only smoke outcome are recorded in the
coordinated Seismo release receipt after execution. Rollback: fast-forward a
revert of this code; do not delete or clear pending repair keys/checkpoints.
