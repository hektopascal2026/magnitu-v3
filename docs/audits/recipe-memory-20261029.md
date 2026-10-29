# Recipe memory repair — 29 October 2026

User explicitly requested actual full training and repair of remaining faults.
Installed worker baseline: clean b5ba4aed6f0786a8510305a0b4054ca7689f2219.
Initial live ml_window.py hash matched before the run; db.py/distiller.py live
hashes were checked after the first local edits and matched their unchanged
baseline Git blobs before any integration/activation. Original local main and
all durable live state are preserved.

Actual full invocation 21b4d3d37461480a98aea488dd8b698d began 18:32:04 UTC
using existing one-shot force-retrain control, normal 1800-second short-slot
budget, 4 GiB cap and unchanged quality gates. Digital/root candidates were
rejected correctly; EU v51 passed, activated and delivered scores (including
50 acknowledged repairs). Recipe distillation began 18:38:19. At 18:40:31
the kernel killed its Python child (PID 2430215, approx 3.1 GiB anonymous RSS)
under the service memory cgroup. The whole service stopped with oom-kill/143;
its normal cleanup restored translation health. This is a real failure, not
successful training. Previous active models/scores and pending work remain.

The cache contains 506,721 entries. Student training already samples keys,
but three later recipe paths still call get_all_entries(): normalization
materializes the entire body/embedding corpus, cap optimization loads the
whole corpus to find labeled keys, and quality evaluation loads everything
before slicing 100 rows. The full corpus has no publication-date index.

Repair: stream every text row by primary key in small fetchmany batches for
order-independent normalization; fetch only the original labeled keys for
cap optimization; fetch only the same ordered sample keys for quality.
No embedding blobs or wide date-sort temporary files in the stream. Reuse
normalized keyword lookup within each fixed recipe/cap trial. No text is
truncated, no normalization population/sample membership/order, student
allowance, gate, model, prompt, paid call or cgroup limit is changed.

Tests verify real SQLite full-row population equality, exclusion of blobs,
bounded fetch/early close/error close, exact normalization arithmetic and
order independence, original quality sample binding and existing recipe
parity/floors. Add these to CI. Full worker/repair/recipe gate initially passed
170 tests locally before the extra resource-lifecycle cases. Final set: 172
pass locally (one NumPy warning) and 172 pass on native Python 3.14 (37
dependency warnings), using private test data and offline model flags.
CI, activation and actual full rerun remain pending.

Version impact PATCH (coordinated Seismo 8.5.7). Risk: normalization still
visits the complete population and memory depends on the largest full body;
this is not a reduced-corpus experiment. Rollback: revert and deploy worker
source without deleting models, checkpoint, queue or labels; previous code's
whole-corpus OOM risk returns. No schema/API/dependency change.
