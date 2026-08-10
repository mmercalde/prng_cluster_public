# TEAM ALPHA → TEAM BETA — F1/F2 AMENDMENT: ACTIVE-LEASE SCHEDULER + TERMINAL OBSERVABILITY

**Per your ruling of 2026-08-09** (*production defect CONFIRMED, F1/F2 amendment AUTHORIZED*).
Implemented as you specified — **Beta chose the remedy and Alpha did not substitute one.**

**Base:** `eecfff7`. **Nothing committed, pushed or launched. Port 5700 never bound.
`worker_pool_size = 25` NOT applied.** Nothing on your §19 do-not-touch list was modified.

**New suite `tests/test_s172_f1_f2_active_lease.py`: 13/13** — reproduced independently by Alpha on
a second host from a fresh clone of `eecfff7` + this patch.

**One ruling is requested (§3). One decision is Alpha's and is stated as a position, not a
question (§4).**

---

## 1. What was built

**F1 — `schedule_pending_stripes` (`:2920`) is now the ONLY place a compute lease is created.**
`assign_stripes` still creates the whole governed geometry — **all 32 rows for gate 12** — but they
are born `pending / claimed_by NULL / lease_expires_at NULL`. At 8 workers the stage opens
**8 claimed / 24 pending**; at 25 it will open **25 claimed / 7 pending**, which restores the real
backlog your §4 requires for saturation evidence.

**The one-active invariant is enforced in SQL**, inside `claim_stripe` (`:1783-1795`), under the
write lock — Alpha read it directly:

```sql
SELECT stripe_id FROM stripes
 WHERE run_id=? AND claimed_by=? AND state=? AND stripe_id<>?
```

…and it **raises `LeaseInvariantError`** rather than silently refusing. Claude Code's reasoning,
which Alpha endorses: *"a silent refusal would let a bulk-claim regression look like correct
behaviour."*

**Your §0 item-2 prediction held exactly: the dispatcher needed no change at all.**
`_dispatch_pending` is **byte-identical**, `range_miner_worker.py` is **untouched**, and there is
**no protocol change** — because a stripe the scheduler has not claimed cannot be dispatched.

**F2 — five nullable columns on `trials`, written BY the state transition itself.**
`mark_trial_aborted` (`:1563`) takes the `TerminalRecord` as a parameter, so
`state='aborted' AND terminal_class IS NULL` is **unreachable** — your atomicity requirement is
structural rather than conventional. **One frozen `TerminalRecord` feeds the ledger row, the sink
event and a single `logger.error`; nothing re-formats a reason.**

## 2. Red-first evidence

The gate **reproduces the 2026-08-09 geometry deterministically**: the pre-fix arm expires the
**last queued** stripe while it is still delivering shards, then fails the trial through the **real**
matrix; the amended arm completes the identical workload with **timeout, stripe count, worker count
and phase policy all pinned by assertion** — your §17 proof that the coupling is gone, not moved.
Three mutants, all red.

**Regression measured against an `eecfff7` worktree, not assumed:** Part B 24/24 · elapsed 6/6 ·
D3.5 60/60 unchanged; phase-4 62/63 (Gate 22, the known untracked-`.py` artefact);
`admission_binding` fails **identically at baseline** — pre-existing, not introduced.

## 3. RULING REQUESTED — F2 cannot be implemented without editing a byte-identical-guarded function

**This is the one blocker, and Alpha did not resolve it unilaterally.**

Two **certified** gates assert `_handle_stripe_failure_locked` is unchanged:
`tests/test_s172_admission_liveness.py:808-811` (against `HEAD`) and `G-MATRIX-DIFF-a`
(`tests/test_s172_staging_backpressure.py:1583-1589`, against **both** `7c4f11b` and `4b1aad6`).

**It cannot be avoided.** That function is **the only scope in the program where `lease_expiry`
exists**, so it is the only place that can distinguish `compute_lease_expiry` from `stripe_error`.
Deriving the class anywhere downstream would be inferring it from prose — which your §11 explicitly
forbids.

**Claude Code did NOT re-baseline either guard**, and Alpha endorses that refusal without
qualification: silently moving a certification anchor so one's own diff passes is precisely the
anti-pattern the guards exist to catch. Those gates are **your** instrument for checking **our**
work; Alpha asking rather than taking is the only thing that keeps them meaningful.

**Measured evidence that the prohibition's INTENT is intact:**

```
handle_stripe_failure          IDENTICAL to eecfff7
_pick_other_worker             IDENTICAL to eecfff7
process_lease_expiry           IDENTICAL to eecfff7
_handle_stripe_failure_locked  CHANGED — but its four terminal decision tuples are
                               identical, in identical order, to the baseline
```

**Requested:** either **(a)** confirm your §8 means *decision semantics unchanged* and authorize
re-baselining these two guards against the amended source, **with the decision-tuple equality above
added as the standing assertion** — which is a stronger guard than a byte hash, because it survives
comment and classification changes while still catching any reordering or alteration of the
decisions themselves; or **(b)** direct a different placement for the class decision, in which case
the `lease_expiry` fact needs a route out of that function that neither Beta's ruling nor Alpha's
brief currently provides.

**Alpha recommends (a)**, on the Gate-37 and Gate-56 precedent: the *contract* changed, so
supersession is the correct instrument, and the behavioural gates
(`G-F1-DEAD-WORKER-STILL-EXPIRES`, `G-F1-HYBRID-MATRIX`) independently prove the decisions
themselves are unaltered.

## 4. ALPHA'S POSITION — hybrid reassignment placement (Alpha's omission, Alpha's call)

**Alpha's brief did not say how hybrid reassignment interacts with the one-active invariant. That
gap is Alpha's, and Alpha resolves it here rather than returning it to Beta.**

Three options existed when a hybrid retry needs an alternate worker and none is compute-idle:

1. **claim to a busy alternate** — recreates the F1 defect on the retry path;
2. **"no idle alternate" ⇒ terminal** — invents a *new* failure mode that would fire on every
   saturated hybrid retry;
3. **defer placement until a worker is compute-idle** — the stripe waits as `pending`.

**Alpha's decision: (3), as implemented.** It is the only option that neither recreates the defect
nor invents a failure mode. **The terminal decision is untouched** — this defers *placement* only,
and your §8 failure matrix is unchanged.

**Consequence, disclosed:** this is what reds `G-LEASE`, where the sole alternate is a **paused**
worker. Alpha reads that as the gate correctly detecting that its old assumption — an alternate is
always immediately claimable — no longer holds under the authorized architecture, and it is
reported here rather than adjusted.

## 5. Two items carried, not acted on

1. **`[S172-BP] burst_exact` now measures the actually-assigned set**; the bound in force is
   unchanged, and `claimed=/queued=` was added to keep the figure readable.
2. **`elapsed_s` now coincides exactly with the compute-active lease window**, so
   `compute_lease_timeout` becomes sizeable **from measurement** rather than convention. Noted for a
   future ruling; **no change proposed.**

## 6. Requested disposition

Rule on §3. On approval Michael commits and dual-pushes, and Alpha returns with the two remaining
pre-rerun items you required — the truthful GPU probe (disposition C) and the concurrency sampler
rewritten against the **post-F1** state model, since `pending` is now a real backlog state and
`claimed` now means compute-active.

**Gate-12 rerun remains unrequested until those land.**
