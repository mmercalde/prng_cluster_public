# CLAUDE CODE INSTRUCTIONS — F1/F2 AMENDMENT: ACTIVE-LEASE SCHEDULER + TERMINAL OBSERVABILITY

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD `eecfff7`.
`source ~/venvs/torch/bin/activate` before every test. Long suites:
`python3 -u <suite> | tee /tmp/<name>.log` — never `| tail`.

**Authority:** Team Beta ruling *"GATE-12 F1 FORENSICS / LEASE AMENDMENT"* (2026-08-09) —
**production defect CONFIRMED, F1/F2 amendment AUTHORIZED, Gate-12 rerun REMAINS HELD.**

**Hard constraints:** no commit, no push, **no pipeline launch, no fleet launch, no port 5700
bind.** Gate 12 and the Phase-7 soak are HELD. **Do not apply `worker_pool_size = 25` and do not
launch anything** — Beta: that shape comes *after* F1 is repaired and certified.

**Do not touch** (Beta §19): sieve mathematics · threshold policy · seed-domain authority ·
coverage cursor · D3.5 artifact semantics · WATCHER policy · PRNG kernels · Phase-7 autonomy.
If a concrete dependency is discovered, **disclose it — do not act on it.**

**Base verification:** `git log --oneline -1` = `eecfff7`; tracked tree clean; the existing suites
green. Untracked runtime residue is expected and is not a stop condition.

---

## 0. THE DEFECT, AND WHY BETA REJECTED EACH SINGLE REMEDY

`assign_stripes` (`:2680-2705`) claims **every** stripe of a stage in one loop with **one `now`**
(set once at `:2671`), stamping each `now + compute_lease_timeout` (`:2695`; 300 s at `:245`).
Workers execute serially, so at 4 stripes/worker the last begins with ~40-70 s of its lease left.
On 2026-08-09 three workers **actively streaming results** (last shards 12:47:11.3 / 12.1 / 12.6)
had leases that expired at 12:47:05.487, and the constant-phase policy correctly failed the trial
on that bad input.

**Alpha proposed three candidate remedies. Beta rejected all three individually:**

- *stamp at dispatch* — `_dispatch_pending` still sends the whole batch; the fourth message is
  "dispatched" while sitting behind three jobs **inside the worker**. Moves the false clock, keeps
  the coupling.
- *renew on traffic* — traffic from the current stripe says **nothing** about the three queued
  stripes whose leases are also aging. Broadening renewal to every stripe a worker owns would
  **hide** the modeling error.
- *claim only what can start* — fixes the queue clock but leaves heartbeat-only renewal exposed:
  the result stream and heartbeat share **one ordered TCP path**, so a visibly progressing worker
  can have its heartbeat stuck behind its own results.

**Alpha verified three of Beta's premises against live source before this brief** — record them,
they make the design provably correct rather than merely reasonable:

1. **Serialization is STRUCTURAL, not conventional.** `range_miner_worker.py:1424-1436` calls
   `handle_stripe(msg)` **inline in the receive loop** — the worker cannot read the next message
   until the current stripe finishes. A second assignment sent early simply sits unread in the TCP
   buffer.
2. **The coordinator PUSHES.** `_dispatch_pending` (`:7004-7016`) iterates ledger rows in
   `ST_CLAIMED` and sends `stripe_assign` per worker. **Consequence: if only one stripe per worker
   is ever in `ST_CLAIMED`, the dispatcher cannot send a second one** — the claim policy does the
   work, and a dispatcher rewrite may not be needed at all. Prefer that.
3. **`StripeComplete` is worker-side** (`range_miner_worker.py:1339-1348`), sent after the worker's
   own sub-stripe loop; **coordinator staging is asynchronous**, so compute and staging genuinely
   overlap. Beta's "do not wait for staging" is achievable.

---

## PART F1 — THE ACTIVE-LEASE SCHEDULER

### F1.1 The invariant (Beta §3)

> **A compute lease exists only for work the coordinator has handed to a worker that is presently
> able to execute it. Undispatched backlog has NO compute lease.**

For the current serial worker: **maximum compute-active claimed stripes per worker = 1.**
Completed stripes may remain in `staging`; that does **not** consume the worker's compute slot.

### F1.2 Stage creation and backlog (Beta §4)

Create the **entire governed stripe geometry exactly as today** — for gate 12, all **32 stripe rows
must still exist**. But they begin:

```
state = pending · claimed_by = NULL · lease_expires_at = NULL
```

except the subset handed to currently schedulable workers. For a frozen 8-worker cohort that is
**8 claimed / 24 pending**, not 32 claimed. At 25 workers: **25 claimed / 7 pending**. Beta notes
this also **restores the natural meaning of the queue** that gate-12 saturation evidence needs.

### F1.3 Claim/dispatch semantics (Beta §5)

`pending → claimed` **only** when the scheduler has an eligible, **currently compute-idle** worker
**from the frozen stage cohort**. At that handoff:

```
claimed_by = worker_id
lease_expires_at = now + compute_lease_timeout
```

The lease begins at the **real handoff to an idle worker**, not at stage-wide plan creation. **The
coordinator must not pre-queue a second compute stripe behind a first one.**

**The compute slot is released** when the coordinator accepts terminal compute disposition for the
current stripe — `StripeCompleteMessage`, or the corresponding terminal stripe-error/failure path.
**It is NOT necessary to wait for that stripe's asynchronous staging work to finish** before handing
the worker its next compute stripe.

### F1.4 Active-lease renewal — progress is liveness (Beta §6)

Heartbeat remains valid but is **no longer the only** renewal mechanism. For the **currently active
stripe only**, a successfully validated progress message renews
`lease_expires_at = now + compute_lease_timeout`. At minimum:

- `MinerHeartbeatMessage` when `current_stripe_id` matches the active claim;
- `SubStripeResultMessage` when **run_id / stripe_id / attempt / worker identity all match** the
  active claim.

`StripeCompleteMessage` **leaves** compute-active state and therefore **clears** the active lease
rather than extending it.

**EXPLICITLY FORBIDDEN — none of these may renew:** wrong worker · wrong stripe · stale attempt ·
invalid/rejected result · `status` frame · `register` · traffic for another stripe · a late result
from a prior attempt.

> **The rule is: progress on THIS active attempt renews THIS active attempt.**
> Not: any traffic from the host keeps everything it owns alive.
> **Beta calls this distinction load-bearing.**

### F1.5 Keep the certified back-pressure grace (Beta §7)

**Do not remove** the S172 pause/resume lease protection — it solves a different problem
(coordinator-caused silence must not enter the worker-failure matrix). The two become
complementary:

```
normal active work        → heartbeat OR progress renews the lease
coordinator-caused pause  → bounded grace bridges the silence
actually dead worker      → neither progress nor heartbeat → lease expires
```

After a paused connection resumes, the **first valid heartbeat OR valid active-stripe progress**
performs the real renewal and clears the temporary grace. A worker that resumes and then produces
neither still expires after the bounded grace.

### F1.6 The failure matrix is UNCHANGED (Beta §8)

This amendment is **not** permission to weaken phase policy. A genuinely active stripe with no
valid liveness/progress evidence through the full lease period, after all coordinator-caused
exemptions: **phases 1-2 → fail trial; phases 3-4 → existing hybrid retry/reassignment.**
The failed run exposed **a bad input to the matrix, not a wrong matrix.**

### F1.7 The frozen cohort must survive (Beta §9, mandatory)

Dynamic one-at-a-time assignment **does not reopen worker eligibility.** Every `pending → claimed`
transition must choose from the cohort frozen at successful preflight. Workers registering after
freeze stay globally connected but **cannot receive pending work for that trial.** The positive
behaviour observed in the failed run — 22 late workers excluded — **must survive unchanged.**

### F1.8 Abort cleanup must now include pending backlog (Beta §10)

A direct consequence of replacing bulk claim with coordinator-owned backlog: a terminal trial may
now leave substantial **pending** work. **Terminal cleanup must leave no live runnable stripe** —
all nonterminal states, **including pending**, transition to the terminal cancellation state, and
**a later scheduler loop must be unable to claim one of those rows after termination.**

---

## PART F2 — TERMINAL OBSERVABILITY (rides with F1, Beta §11)

`_handle_stripe_failure_locked:5106-5107` builds a precise reason and emits **no log record**;
`fail_trial:5342-5348`, `abort_trial:5350-5423`, `cancel_active_stripes:1546-1556` emit none;
`trials` has no column for it (discarded `:5406-5407`). Result: **five minutes of coordinator
silence** on 2026-08-09 and a downstream `MinerIngressError` as the operator's only signal.

Beta requires F2 in the same amendment because the F1 lease path **is** the path needing
observability, and the acceptance gates need an authoritative terminal reason to distinguish
genuine worker death from a bad lease transition.

**Required durable terminal record** — narrow additive shape on `trials`:

```
terminal_class TEXT NULL · terminal_reason TEXT NULL · terminal_stripe_id TEXT NULL
terminal_worker_id TEXT NULL · terminal_attempt INTEGER NULL
```

`finalized_at` already provides terminal time; fields that do not apply may be NULL.
Example classes: `compute_lease_expiry` · `stripe_error` · `staging_capacity_timeout` ·
`staging_sizing_failure` · `worker_admission_timeout` · `explicit_abort` · `coordinator_error`.
**Do not infer the class later from prose.**

**Atomicity:** the terminal state transition and its durable record **commit together**. A crash
must not leave `state = aborted` with `terminal_reason = NULL` on a path that possessed a reason.

**Log parity:** one **synchronous ERROR-level** record from the same canonical terminal event. The
durable ledger record, the coordinator log and `Phase5Sink.abort_trial(event)` must agree.
**Do not construct three independent reason strings.**

---

## ACCEPTANCE GATES (Beta §16, minimum set)

- **G-F1-QUEUE-NO-LEASE** — with `N > W` stripes and W workers, exactly W enter compute-active
  claimed; the remaining `N−W` stay `pending` with `claimed_by = NULL` **and**
  `lease_expires_at = NULL`.
- **G-F1-ONE-ACTIVE** — no serial worker holds more than one compute-active claim. **A mutation
  restoring bulk claim must red this gate.**
- **G-F1-FRESH-HANDOFF** — when worker A completes X and receives pending Y,
  `Y.lease_expires_at ≈ handoff_time + timeout`, **not** stage-creation time.
- **G-F1-PROGRESS-RENEWAL** — a valid accepted `SubStripeResultMessage` for the active attempt
  extends its lease.
- **G-F1-HEARTBEAT-RENEWAL** — existing valid active-stripe heartbeat renewal still works.
- **G-F1-SCOPE-RENEWAL** — none of these renew: wrong worker · wrong stripe · stale attempt ·
  invalid result · status frame · late result from a prior attempt.
- **G-F1-LIVE-STREAM-NO-EXPIRY** — reproduce the defect geometry with an intentionally slow result
  stream lasting **beyond** the original stage-wide 300 s stamp; a worker continuously delivering
  valid active-stripe progress **must not** enter the lease-expiry matrix. **Do not solve this by
  increasing the timeout.**
- **G-F1-DEAD-WORKER-STILL-EXPIRES** — the clean control: a genuinely silent active worker still
  expires, and a constant phase still fails the trial immediately. **This proves F1 did not disable
  fault detection.**
- **G-F1-HYBRID-MATRIX** — a genuine hybrid active-stripe expiry still enters the certified hybrid
  retry/reassignment path.
- **G-F1-BACKPRESSURE-HANDOFF** — the pause/resume protection stays green; after resume, valid
  active progress performs the real renewal and ends the grace; a permanently silent worker still
  expires after the bounded grace.
- **G-F1-FROZEN-COHORT** — a late worker may not receive pending work from an already-frozen trial;
  reconnect under the certified same-identity/capability rule remains valid.
- **G-F1-ABORT-PENDING** — abort a trial holding **both** claimed work and pending backlog; prove
  no nonterminal/runnable stripe remains afterward.
- **G-F2-TERMINAL-DURABILITY** — inject a genuine lease failure; the same canonical
  class/reason/stripe/worker/attempt appears in **(1)** durable trial state, **(2)** the abort
  event, **(3)** the coordinator ERROR log. **A mutation dropping the durable reason or the log
  must red.**

### Red-first requirement (Beta §17)

The gate-12 incident is an unusually strong real-world red case — **use it.** Recreate the causal
geometry deterministically: multiple stripes per serial worker · bulk-claim mutant · delivery slower
than the stage-wide lease · worker still making progress. **The pre-fix behaviour must fail by lease
expiry**, and the amended scheduler must complete the same workload **without** raising
`compute_lease_timeout`, reducing stripe count, increasing worker count, or weakening constant-phase
policy. **That is the proof the coupling is gone.**

### Regression package (Beta §18)

Rerun in full, **staging suites sequentially** (the free-space race is separately backlogged):
S172 staging back-pressure · S172 Part B · S172 elapsed roundtrip · D3.5 finalizer · phase-4
coordinator · the new F1/F2 suite. **No gate-12 fleet execution belongs in this verification.**

### Production scope (Beta §19)

Primary `miner/range_miner_coordinator.py`; `miner/range_miner_worker.py` **only** if required to
preserve/clarify active-stripe signalling. **The preferred design needs no new worker protocol
message** — the coordinator can simply not send the next assignment until the current compute
stripe terminates (see §0 item 2). **Any protocol-schema change requires explicit disclosure.**
Additive ledger migration for F2 is authorized.

---

## REPORT

`docs/CLAUDE_CODE_REPORT_F1_F2_ACTIVE_LEASE.md`:

1. Per-section implementation notes with `file:line`.
2. Whether the dispatcher needed changing at all, or whether the claim policy alone was sufficient.
3. The renewal predicate as built, and **each forbidden case with the branch that rejects it.**
4. How abort cleanup reaches pending backlog, and how a post-termination scheduler loop is
   prevented from claiming a cancelled row.
5. The F2 atomicity mechanism, and proof the three reason surfaces derive from **one** construction.
6. Red-first evidence per gate; mutation evidence where specified.
7. Full regression results.
8. Files changed — anything beyond §19's scope justified.
9. **Any disagreement with this brief reported, not worked around.**

**One observation for the report, not a task:** the `elapsed_s` certified under S172-R4 is
`time.time() - t0` across the worker's whole stripe (`range_miner_worker.py:1345`) — which under
this design becomes exactly the compute-active lease window, and is therefore directly useful for
sizing `compute_lease_timeout` from measurement rather than convention. Note it; do not act on it.
