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
Complete [CI passed](https://github.com/hektopascal2026/magnitu-v3/actions/runs/40934054784)
on source `58ff14f3eb8289be3cc11a4c735726deb797fb65`. PR 30 merged normally
as `2889432c744b9328da2fed85074b030540949c94`, source-identical to that tested
head. Activation used the shared Seismo deployment lock while both ML units
were idle; the installed baseline was clean and post-activation hashes were
checked. No state, models, checkpoints, labels or pending keys were reset.

A bounded frozen comparison on the VPS used 128 complete live text rows,
the stored EU recipe (597 keywords) and the same default normalization target.
Old and new normalization results matched exactly, including reversed input
order. SHA-256 of the result:
`f09447dc61cd548c7d4527281c65c5ca3136803f8eb44c89a3cc3bc17672448f`.
Only counts/hashes were exported; raw source prose stayed on the VPS.

## Actual full acceptance

User-authorized rerun `5159a50c5b114747bb292d00b603528b` began 18:51:51 UTC
and exited zero at 19:22:20 UTC (30 min 30 sec). The normal calendar policy
selected the existing 3600-second ad-hoc allowance; no timeout override or
memory increase. The exact unit result was success, the received Seismo
report finished at 19:21:55Z/status ok with all six desks trained, and kernel
evidence showed no OOM in this invocation. Memory peak reached the unchanged
4 GiB cap, with 395.8 MiB swap peak; do not infer new capacity/headroom.

Digital v111, root v1084 and Security v123 were rejected under existing gates
and retained their active models. Energy v5, EU v52 and Finance v3 were
promoted; all three delivered recipes and model-vault backups with HTTP 200.
The previously failing EU recipe path completed. Finance had only four recent
evaluation rows and used the unchanged small-sample policy; this is technical
acceptance, not strong model-quality evidence for that desk. Translation
resumed and returned HTTP 200/status ok.

Budgeted repair backlog remains (one Energy observation: 176 pending, after
24 acknowledgements). No queue was emptied to hide its age warnings. No paid
editorial generation, mail send, destructive restore or sustained-capacity
certification is included. Private operator evidence is retained under
`/var/lib/seismo/release-receipts/v8.5.7*.json`. Coordinated Seismo application
8.5.7 deployed/tagged `880370780464b5b732ff463b93ab4b8a0d9e49d9` and passed
all 24 authenticated feed export checks across six desks (26.5–148.8 ms).
The documentation-only follow-up does not change either live source revision.

Version impact PATCH (coordinated Seismo 8.5.7). Risk: normalization still
visits the complete population and memory depends on the largest full body;
this is not a reduced-corpus experiment. Rollback: revert and deploy worker
source without deleting models, checkpoint, queue or labels; previous code's
whole-corpus OOM risk returns. No schema/API/dependency change.
