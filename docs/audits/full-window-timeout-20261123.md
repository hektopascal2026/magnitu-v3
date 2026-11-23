# Full Magnitu window timeout — 23 November 2026

The installed worker was clean at 6c6e68d266c1a036c6ba4b6e75edf8318e16d3cf.
The live ml_window.py SHA-256 was
40d9104f597c90a88e147a81a8564dfeaf35dd4902886c53cdf5e9cfe77bdc42 and
sync.py was 13ece460a4fab7b8049c719bdd4f76bbe54881a686159c308991744f2dd6c95e.
The local main checkout was clean and behind production. This fix uses an
isolated temporary checkout of the installed revision.

The 01:00 UTC full window reached shared embeddings at 01:04:24 and repeatedly
computed 1,000-entry batches through 01:59:15. The outer 3,600-second worker
budget expired before desk preparation or training. Partial embedding work
survived. This was general pending work, processed newest first.

The fix supplies one 600-second absolute admission deadline to full-window
shared embeddings and subsequent orphan repair. It uses the existing deadline
checks and ten-entry commit batches. Expiry leaves pending work intact and
continues to desks. The training pipeline separately computes missing labelled
embeddings. The regression simulates an overrun and orphan repair, verifying
that the same deadline stops repeated embedding admission and labels are reached.
