# TEAM ALPHA → TEAM BETA — F1/F2 AMENDMENT SUBMITTED

**Date:** 2026-08-09 · **Base:** `eecfff7` · **Suite:** 13/13, reproduced on a second host.

The F1/F2 amendment is implemented as you ruled. Nothing committed, pushed or launched; port 5700
never bound; `worker_pool_size = 25` not applied; nothing on your §19 do-not-touch list modified.

**Your §0 item-2 prediction held: the dispatcher needed no change at all.** `_dispatch_pending` is
byte-identical, `range_miner_worker.py` untouched, no protocol change — a stripe the scheduler has
not claimed cannot be dispatched. The one-active invariant is enforced **in SQL** inside
`claim_stripe` and **raises** rather than silently refusing. F2's terminal record is written **by
the state transition itself**, so `state='aborted' AND terminal_class IS NULL` is unreachable.

**Two items need your attention, and they are deliberately different in kind:**

1. **A ruling request (cover §3).** F2 cannot be implemented without editing
   `_handle_stripe_failure_locked` — the only scope in the program where `lease_expiry` exists, and
   therefore the only place that can distinguish `compute_lease_expiry` from `stripe_error`. Two
   certified gates require it byte-identical. **Alpha did not re-baseline them.** Those gates are
   your instrument for checking our work, and moving an anchor to make our own diff pass is the
   anti-pattern they exist to catch. Measured evidence that the prohibition's intent is intact is in
   the cover; Alpha recommends authorizing the re-baseline **with the four terminal decision tuples
   as the standing assertion** — a stronger guard than a byte hash.

2. **An Alpha position, not a question (cover §4).** Alpha's brief failed to specify how hybrid
   reassignment interacts with the one-active invariant. That omission is Alpha's, and Alpha
   resolves it: **defer placement until a worker is compute-idle** — the only option that neither
   recreates the F1 defect nor invents a new failure mode. The terminal decision is untouched. This
   is what reds `G-LEASE`, disclosed rather than adjusted.

**Package:** cover · implementation report · `f1f2.patch` · `tests/test_s172_f1_f2_active_lease.py`.

**Gate-12 rerun remains unrequested.** After this clears, Alpha returns with the two remaining
pre-rerun items you required — the truthful GPU probe and the concurrency sampler rewritten against
the post-F1 state model, since `pending` is now a real backlog state and `claimed` now means
compute-active.
