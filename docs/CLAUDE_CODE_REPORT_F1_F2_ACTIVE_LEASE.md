# CLAUDE CODE REPORT — F1/F2 AMENDMENT: ACTIVE-LEASE SCHEDULER + TERMINAL OBSERVABILITY

**Host:** VM101 · repo `~/distributed_prng_analysis` · base HEAD `eecfff7` (tracked tree clean at start)
**Authority:** Team Beta *"GATE-12 F1 FORENSICS / LEASE AMENDMENT"* (2026-08-09).
**Constraints honoured:** no commit · no push · no pipeline launch · no fleet launch · **port 5700 never bound**
(verified before any work: `ss -ltn | grep 5700` empty, no `watcher_agent` / `window_optimizer` /
`range_miner_worker` processes). **`worker_pool_size = 25` was NOT applied.** Nothing in §19's
do-not-touch list was modified.

**Beta chose the remedy; this report implements it and does not substitute one.** Three points where
the brief did not reach are flagged in §9 as ruling requests, not resolved unilaterally.

> **⚠ HEADLINE FOR THE REVIEWER — THREE CERTIFIED GATES NOW RED, AND I DID NOT RE-BASELINE THEM.**
> Two of them assert that `_handle_stripe_failure_locked` is **byte-identical** to a certified
> revision. **F2 cannot be implemented without editing that function** — it is the only scope in
> which `lease_expiry` exists, and therefore the only place that can tell
> `compute_lease_expiry` from `stripe_error`. This is a direct collision between §F1.6 as the gates
> encode it and §F2 as the brief specifies it. **§9.1 has the evidence and the decision Beta owns.**

---

## 1. Implementation notes, per section

### F1.1 / F1.2 — the invariant and the backlog

| what | anchor |
|---|---|
| stripe rows are born `pending / claimed_by NULL / lease_expires_at NULL` | `range_miner_coordinator.py:1699` (`add_stripe`, unchanged — it already did this) |
| the whole governed geometry is still created | `:2842-2866` — `assign_stripes` creates **every** planned row before any claiming; for gate 12 all 32 rows still exist |
| only compute-idle workers receive a claim | `:2904` — `assign_stripes` delegates the initial handoff to the scheduler |
| **one compute-active claim per worker, enforced in SQL** | `:1783-1795` — `claim_stripe` refuses (raises) if the worker already holds a `claimed` stripe for the run |

At 8 workers / 32 stripes the stage now opens **8 claimed / 24 pending**; at 25 workers it would be
**25 claimed / 7 pending**. Proven by `G-F1-QUEUE-NO-LEASE` at W=3, N=8.

**Why the invariant lives in the ledger and not in the scheduler** (owner rule, skill §7 —
structurally stronger over structurally smaller): a caller-side check makes the property hold *by
inference*, and a second claimer racing the check would defeat it. Inside `claim_stripe` the read
and the write are the same statement under the same `_write_lock`, so it holds *by construction*.

**It RAISES rather than returning False** (`LeaseInvariantError`, `:203`). A silent refusal would
make a regression that restores bulk claim look like correct behaviour — the ledger would quietly
decline the extra claims and the ledger state would be indistinguishable from the amended one. That
is precisely what `G-F1-ONE-ACTIVE` must be able to detect, and the mutant in that gate confirms it
does.

### F1.3 — claim/dispatch semantics

`schedule_pending_stripes` (`:2920-3010`) is **the only place in the coordinator that creates a
compute lease.** Three filters, in order:

1. **frozen cohort** — `cohort_filter`, the identical predicate initial assignment used before this
   amendment (§F1.7);
2. **compute-idle** — `ledger.compute_busy_worker_ids` (`:2049`) returns workers holding a `claimed`
   stripe. **`staging` is deliberately excluded**, which is §5's "do not wait for staging" in one
   place: a stripe whose `StripeComplete` has been accepted has released its worker's compute slot;
3. **not the prior claimer** — see §F1.6 below.

The compute slot is released by `record_stripe_complete` (`:1919-1932`, unchanged), which moves
`claimed → staging` **and clears the lease**. No wait for staging anywhere.

### F1.4 — active-lease renewal

`_renew_active_lease` (`:7465-7523`) is the single renewal predicate; both liveness sources call it.
Full treatment in §3.

### F1.5 — the certified back-pressure grace is retained

Not removed, not weakened. The `[S172-BP] resume_grace_cleared` clear moved verbatim into
`_renew_active_lease` (`:7513-7522`) so that **either** liveness source can now end the bridge, which
is exactly the composition Beta describes. `G-F1-BACKPRESSURE-HANDOFF` drives all three rows of the
§F1.5 table: paused → exempt; resumed + progress → real renewal, grace cleared; resumed + permanent
silence → still expires.

> ⚠ **Naming collision, flagged so nobody conflates them.** The S172-BP amendment already has an
> item called **F2** (the resume grace). *This* brief's **F2** is terminal observability. The code
> comment at `:7508-7512` says so explicitly at the one place they touch.

### F1.6 — the failure matrix

**Three of the four matrix functions are byte-identical to `eecfff7`**, verified by AST extraction
this session:

```
handle_stripe_failure          IDENTICAL
_pick_other_worker             IDENTICAL
process_lease_expiry           IDENTICAL
_handle_stripe_failure_locked  CHANGED
```

And the **decision structure of the changed one is unchanged** — same branches, same order, same
terminal tuples:

```
baseline: [('fail_trial','non_retryable'), ('fail_trial','constant_phase'),
           ('fail_trial','no_alternate_worker'), ('fail_trial','hybrid_second_failure')]
live    : [('fail_trial','non_retryable'), ('fail_trial','constant_phase'),
           ('fail_trial','no_alternate_worker'), ('fail_trial','hybrid_second_failure')]
```

What changed inside it: (a) a `TerminalRecord` at each `fail_trial` (F2 — the class distinction
`compute_lease_expiry` vs `stripe_error` is made at `:5395` where `lease_expiry` is in scope, and
nowhere else can make it); (b) the hybrid reassignment **requeues instead of claiming directly**
(`:5479-5497`), because a direct claim would hand the stripe to a worker whether or not it is
compute-idle — re-creating the defect one stripe deep on the retry path. **The terminal decision is
untouched:** `_pick_other_worker` still answers "does any alternate eligible cohort worker exist at
all", deliberately not "is one free right now", so a saturated retry does not become a trial failure.

### F1.7 — the frozen cohort survives

`schedule_pending_stripes` calls `cohort_filter` on **every** pass, so dynamic one-at-a-time handoff
does not reopen eligibility. `G-F1-FROZEN-COHORT` proves a post-freeze joiner receives nothing on the
initial handoff **and** on a later scheduler pass after capacity frees.

### F1.8 — abort cleanup reaches pending backlog

See §4.

### F2 — terminal observability

See §5.

---

## 2. Did the dispatcher need changing? **No — the claim policy alone was sufficient.**

`_dispatch_pending` (`:7524-7540`) is **byte-identical to `eecfff7`**. Beta's §0 item 2 predicted
this and it held: the dispatcher iterates ledger rows in `ST_CLAIMED` and sends one `stripe_assign`
per row, so **if only one stripe per worker is ever `ST_CLAIMED`, it cannot send a second one.** The
claim policy does the work.

`miner/range_miner_worker.py` is **untouched**, and **no protocol message was added or changed** —
the §19 preferred design. The coordinator simply does not send the next assignment until the current
compute stripe terminates.

One ordering decision in `serve_trial` (`:6834`): the scheduler pass runs **immediately above**
`_dispatch_pending`, so a stripe handed off on a given iteration is also dispatched on that
iteration and a freed worker never idles a whole poll interval. It runs **below** the loop's terminal
check, and is paired with `claim_stripe`'s own terminal guard (§4).

---

## 3. The renewal predicate as built, and every forbidden case

`_renew_active_lease(wconn, run_id, stripe_id, worker_id, source, now)` — `:7465`.

```
if not stripe_id: return False
ok, why = accept_stripe_message(wconn, run_id, stripe_id, worker_id, (ST_CLAIMED,))
if not ok: log DEBUG; return False
renewed = ledger.renew_lease(run_id, stripe_id, worker_id, now + compute_lease_timeout)
if renewed and clear_capacity_resume_grace(worker_id): log the S172-BP grace clear
return renewed
```

Call sites — exactly two, asserted by AST in `G-F1-SCOPE-RENEWAL`:
`:7011` heartbeat branch · `:7060` sub_stripe_result branch, **after** `record_substripe_result`
returns `inserted=True`.

| forbidden case | branch that rejects it | gate |
|---|---|---|
| **wrong worker** | `accept_stripe_message:2794` `conn.worker_id != msg_worker_id`, and `:2803` `claimed_by != msg_worker_id` | G-F1-SCOPE-RENEWAL (1) |
| **wrong stripe** | `accept_stripe_message:2800` unknown stripe / `:2803` `claimed_by` mismatch | (2) |
| **stale attempt** | `accept_stripe_message:2807` ledger `current_attempt` vs the **connection's** recorded assignment attempt | (4) |
| **late result from a prior attempt** | same attempt check; the retry path also re-claims to a **different** worker, so `claimed_by` fails first | (4) |
| **invalid / rejected result** | caller: renewal is unreachable unless the L1 fence passed **and** the row was newly inserted — a duplicate returns before it (`:7053-7057`) | by construction |
| **not compute-active (`staging`)** | permitted-states `(ST_CLAIMED,)` at `:7469`, and `renew_lease`'s own `WHERE state='claimed'` | (5) |
| **queued backlog** | same — a `pending` row is not `ST_CLAIMED`; the gate asserts its lease stays NULL | (3) |
| **`status` frame** | `_serve_dispatch:7016` returns for every type outside the two branches | (7), AST-asserted |
| **`register`** | handled in the register branch; never reaches renewal | (7), AST-asserted |
| **idle heartbeat, no active stripe** | `if not stripe_id: return False` | (6) |

**The final authority is `renew_lease` itself** (`:1826-1838`, unchanged): its `WHERE` re-tests
`state='claimed' AND claimed_by=worker` in the same statement that writes the new expiry, so even a
caller that reasoned wrongly cannot extend a lease the worker does not currently own.

**`StripeComplete` deliberately does NOT renew** — `record_stripe_complete` clears the lease as it
leaves compute-active state, which is §F1.4's requirement.

---

## 4. Abort cleanup, backlog, and the post-termination claim

**Reaching pending backlog.** `cancel_active_stripes` (`:1671-1681`) already transitioned
`pending | claimed | staging → cancelled`. That was previously incidental; it is now load-bearing,
because a terminal trial routinely holds substantial pending work. Comment added at `:5794-5804`
recording *why* the state list matters now. `G-F1-ABORT-PENDING` aborts a trial holding 1 claimed +
5 pending and asserts the resulting state set is exactly `{cancelled}`.

**Preventing a later scheduler loop from claiming a cancelled row** — two independent mechanisms, and
the second is the one that holds under a race:

1. `schedule_pending_stripes` finds no `pending` rows after cleanup, so it places nothing.
2. **`claim_stripe` refuses structurally** (`:1799-1812`): the UPDATE carries
   `AND NOT EXISTS (SELECT 1 FROM trials t WHERE t.run_id=? AND t.state IN ('aborted','committed'))`.
   Read and write in one statement, so a scheduler pass racing the abort cannot re-arm a row between
   the check and the claim.

Expressed as `NOT EXISTS` rather than "require running" **on purpose**: many gates and bare-API paths
never create a trial row at all, and absence of a trial row is not termination. `G-F1-ABORT-PENDING`
drives mechanism 2 directly — it forces a cancelled row back to `pending` by hand and asserts
`claim_stripe` still returns False.

---

## 5. F2 atomicity, and proof of one construction

**Atomicity.** The terminal record is written **by the state transition itself**, not by a follow-up
UPDATE: `mark_trial_aborted` (`:1563-1601`) sets `state='aborted'`, `finalized_at`, and all five
terminal columns in **one statement, one commit, one rowcount**. A crash cannot interleave between
"the trial is aborted" and "this is why", so `state='aborted' AND terminal_class IS NULL` is
unreachable for any path that possessed a record. `G-F2-TERMINAL-DURABILITY` asserts that invariant
explicitly.

**Schema.** Five nullable columns on `trials` (`:1013-1024`), plus the additive migration at
`:1211-1228` using the established idiom in this file — `PRAGMA table_info` guard then
`ALTER TABLE … ADD COLUMN`, guarded on the **live table shape**, never on a version counter nobody
maintains. Old rows backfill as NULL, which is honest: a trial that terminated before this amendment
genuinely has no recorded class.

**One construction, three surfaces.** `TerminalRecord` (`:168-201`) is a **frozen** dataclass —
immutable so the object handed to the three surfaces is provably the same value, where a mutable
record could be edited between them and reintroduce the divergence the type exists to prevent.

| surface | derivation |
|---|---|
| (1) durable ledger | `abort_trial:5776` → `mark_trial_aborted(..., terminal=terminal)` |
| (2) `Phase5Sink.abort_trial(event)` | `:5806-5808` → `**terminal.as_event_fields()` |
| (3) coordinator ERROR log | `:5799` → `terminal.log_line()` |

All three read the same object; **nothing re-formats a reason**. `G-F2-TERMINAL-DURABILITY` proves it
by asserting field-by-field equality across (1) and (2) and then asserting
`trial["terminal_reason"] in log_line` — i.e. the log did not compose its own text.

**The class is recorded at the site that knows** — never inferred downstream from prose, which is the
failure mode Beta named. The `compute_lease_expiry` / `stripe_error` distinction is made at `:5395`
from the `lease_expiry` parameter; nowhere else in the program has that fact.

**The log is gated on `first`** (`:5793`), so an idempotent re-abort cannot double-log.

Classes defined at `:154-165`: `compute_lease_expiry` · `stripe_error` · `staging_capacity_timeout` ·
`staging_capacity_invariant` · `staging_sizing_failure` · `worker_admission_timeout` ·
`no_eligible_worker` · `threshold_provenance_violation` · `serve_trial_timeout` · `explicit_abort` ·
`coordinator_error`. **All eleven `fail_trial` / `abort_trial` call sites now pass an explicit
record**; a caller that supplies only prose gets `coordinator_error` **explicitly**, never a class
parsed out of the text.

---

## 6. Red-first and mutation evidence

`tests/test_s172_f1_f2_active_lease.py` — **13/13 green**. Every gate is deterministic: time is
passed as `now` to `process_lease_expiry`, `schedule_pending_stripes` and `_renew_active_lease`, so
"delivery slower than the stage-wide lease" is arithmetic, not a race the host can lose.

### G-F1-LIVE-STREAM-NO-EXPIRY — the red-first gate (Beta §17)

The 2026-08-09 geometry, reproduced deterministically: **4 stripes on one serial worker · 90 s
delivery each = 360 s > the 300 s lease · the worker never stops delivering results.** Two arms, same
workload:

**PRE-FIX arm** — bulk claim at one `now`, and **no renewal call at all**. That is the honest model:
pre-amendment the only renewal path was the heartbeat, and the forensics established none ever landed
(the three expired stripes still carried assign-time + 300 to the microsecond). The arm asserts the
incident's exact shape: the stripe that expires is **the last in the worker's queue** (`idx == N-1`),
the expiry instant is **past the stage-wide deadline**, and the stripe **has delivered shards** —
`"the expired stripe had delivered no results — that is a dead worker, not the defect"`. It then runs
the real matrix and asserts `fail_trial` + `terminal_class == compute_lease_expiry`. **The defect
reproduces.**

**AMENDED arm** — identical workload, and the gate pins every escape Beta forbade:
`assert coord.config.compute_lease_timeout == LEASE` (timeout not raised) ·
`assert len(assigns) == N` (stripe count unchanged) · `assert len(conns) == 1` (worker count
unchanged) · constant phase throughout (policy unweakened). Progress is delivered at 25/50/75/100% of
each stripe, and at **every** tick the gate asserts both
`expired_claimed_stripes(...) == []` and `process_lease_expiry(...) == []`. The whole workload
completes past the stage-wide deadline with the trial still `running`.

### Mutation evidence

| mutant | gate | result |
|---|---|---|
| **bulk claim restored** — claim every pending stripe of the stage to one worker in one pass | `G-F1-ONE-ACTIVE` | **RED** (`LeaseInvariantError`) |
| **durable terminal record dropped** — `mark_trial_aborted` ignores `terminal` | `G-F2-TERMINAL-DURABILITY` | **RED** |
| **terminal ERROR log dropped** — `logger.error` silenced | `G-F2-TERMINAL-DURABILITY` | **RED** |

`_mutant_red` asserts the mutant **raises**; a surviving mutant fails the gate with
*"MUTANT SURVIVED … the gate is vacuous and proves nothing"* (VIR-2 positive control).

### Clean control

`G-F1-DEAD-WORKER-STILL-EXPIRES`: a genuinely silent active worker still expires, the constant phase
still fails the trial immediately, and the durable record is `compute_lease_expiry` with the right
stripe and worker. **F1 did not disable fault detection.**

### The remaining eleven

`G-F1-QUEUE-NO-LEASE` (W=3/N=8; backlog has NULL claimer **and** NULL lease) · `G-F1-FRESH-HANDOFF`
(Y's lease is `handoff + 300`, and the gate states the defect as arithmetic: under the old stamp Y
would have begun with 50 s left) · `G-F1-PROGRESS-RENEWAL` · `G-F1-HEARTBEAT-RENEWAL` (through the
**real** `_serve_dispatch`, not the helper) · `G-F1-SCOPE-RENEWAL` (seven forbidden cases + an AST
assertion that exactly two renewal call sites exist) · `G-F1-HYBRID-MATRIX` · `G-F1-BACKPRESSURE-HANDOFF` ·
`G-F1-FROZEN-COHORT` · `G-F1-ABORT-PENDING` · `G-F2-TERMINAL-DURABILITY`.

---

## 7. Regression results — measured, with a baseline differential

Run **sequentially** (the staging free-space race is separately backlogged), on the **final** tree
after the last change, per Beta §6 evidence discipline. Baseline measured on
`git worktree add /tmp/f1_base eecfff7` in the same environment — **not assumed**.

| suite | baseline `eecfff7` | final tree | chargeable to this change? |
|---|---|---|---|
| **F1/F2 active lease** (new) | — | **13/13** ✅ | new |
| S172 staging back-pressure | **50/50** ✅ | **48/50** ❌ | **YES** — §9.1, §9.2 |
| S172 Part B | **24/24** ✅ | **24/24** ✅ | no |
| S172 elapsed roundtrip | 6/6 ✅ | **6/6** ✅ | no |
| D3.5 finalizer | 60/60 ✅ | **60/60** ✅ | no |
| phase-4 coordinator | **63/63** ✅ | **62/63** ⚠ | **no** — Gate 22 only, see below |
| S172 admission liveness | **16/16** ✅ | **FAIL** ❌ | **YES** — §9.1 |
| S172 admission binding | **FAIL** ❌ | **FAIL** ❌ | **NO — pre-existing** |

**phase-4 62/63 is Gate 22 and nothing else:**
`unexpected changed .py files: {'tests/test_s172_f1_f2_active_lease.py'}`. Gate 22 builds
`changed_py` from `git status --porcelain`, which includes **untracked** files, so any new test file
reds it. This is the known, twice-recorded behaviour (skill §7) — *expected during development, not a
regression, and not a reason to widen Gate 22*. It resolves when the file is committed. **With the
new suite's own 13 gates, the phase-4 arm is 62/63 + 13/13.**

**`admission_binding` fails identically at the baseline** — same gate (`g_c5_missing_capacity_hits_the_existing_failure`),
same assertion, same empty `reasons`. It is **pre-existing at `eecfff7` and not attributable to this
work.** Reported here because a red suite in the package needs an explanation, not because it is
mine. It is not otherwise diagnosed and I did not touch it.

---

## 8. Files changed

| file | scope | note |
|---|---|---|
| `miner/range_miner_coordinator.py` | **§19 primary** | the whole amendment |
| `tests/test_s172_f1_f2_active_lease.py` | new | the F1/F2 gate suite |
| `tests/test_s172_phase4_coordinator.py` | **beyond §19 — justified below** | fixture repairs to 7 gates |

**`miner/range_miner_worker.py` NOT touched. No protocol schema change.** §19's preferred design held.

### Why `tests/test_s172_phase4_coordinator.py` had to change — every edit disclosed

Seven certified gates went red on the F1 invariant. **None of their assertions was weakened**; the
suite is back to 63/63 (Gate 22 aside). Two distinct categories:

**(a) Three gates asserted the behaviour Beta ordered removed** — "all N stripes claimed" with fewer
workers than stripes. Re-expressed to assert `min(W,N)` claimed + the rest `pending` with NULL
claimer and NULL lease, while keeping each gate's actual property:

| gate | property preserved | how |
|---|---|---|
| Gate 1 macro-stripe partition | exact coverage, span list, `expected_substripes` from the advertised cap | assertions kept; cap assertion moved to the claimed row + `expected_substripes_for(500, 5M) == 1` |
| Gate 57 variant-filtered scheduling | **nothing is ever routed to the incompatible worker** | kept, and **strengthened**: the backlog is driven to a second scheduler pass, proving it is queued rather than stranded. The old proxy `stripes_by_state(PENDING) == []` was only meaningful under bulk claim |
| Gate 61 disconnected worker | the disconnected worker receives nothing | kept for the initial handoff **and** a later pass |

**(b) Four staging gates were built on a fixture F1 makes impossible** — two *compute-active* claims
on one worker, used only to create concurrent staging traffic. That traffic pattern is still a real
production shape (`staging` does not occupy the compute slot), just not reachable by claiming twice.
Fixed with one shared helper, `_compute_done`, which calls the **production**
`record_stripe_complete` — the same transition accepting a worker's `StripeComplete` performs. Gates
43, 49, 56, 63. Using a hand-written `state=STAGING` instead would have left
`stripe_complete_seen` unset and silently broken the publication those gates assert; that was caught
by the gates themselves during this work.

**One further gate needed a poll fix, not a fixture fix.** Gate 42 polled for
`current_attempt == 1` as a proxy for "reassigned". The requeue is now two steps, so that is briefly
true while `claimed_by` is still the **failed** worker. The poll was always racy; before F1 the
transition was a single UPDATE so the race could not be observed. Now waits for
`current_attempt == 1 and state == ST_CLAIMED`.

---

## 9. Disagreements and ruling requests — reported, not worked around

### 9.1 — F2 requires editing a function two certified gates require to be byte-identical ⛔ BLOCKING

**The conflict.** Two gates assert `_handle_stripe_failure_locked` is unchanged:

- `tests/test_s172_admission_liveness.py:808-811` — *"the Blocker-3 matrix must be untouched"*,
  against `HEAD`;
- `tests/test_s172_staging_backpressure.py:1583-1589` (`G-MATRIX-DIFF-a`) — against **both**
  `7c4f11b` and `4b1aad6`.

**Why it cannot be avoided.** This brief's §F2 names `_handle_stripe_failure_locked:5106-5107` as
*the* site that "builds a precise reason and emits no log record", and requires an authoritative
`terminal_class`. **That function is the only scope in the program where `lease_expiry` exists**, so
it is the only place that can distinguish `compute_lease_expiry` from `stripe_error`. Deriving the
class anywhere downstream would be inferring it from prose — which §F2 explicitly forbids.

**What I did NOT do:** re-baseline either guard. They are doing their job, and silently moving a
certification anchor to make my own diff pass is the exact anti-pattern the guards exist to catch.

**Evidence that the prohibition's INTENT is intact**, measured this session:

```
handle_stripe_failure          IDENTICAL to eecfff7
_pick_other_worker             IDENTICAL to eecfff7
process_lease_expiry           IDENTICAL to eecfff7
_handle_stripe_failure_locked  CHANGED — but its four terminal decision tuples are
                               identical, in identical order, to the baseline
```

**Ruling requested.** Either (a) confirm §F1.6 means *decision semantics unchanged* and authorize
re-baselining these two guards against the amended source, with the decision-tuple equality above
added as the standing assertion; or (b) direct a different placement for the class decision, in which
case the `lease_expiry` fact needs a route out of that function that the brief does not currently
provide. **I recommend (a)** — the behavioural gates (`G-F1-DEAD-WORKER-STILL-EXPIRES`,
`G-F1-HYBRID-MATRIX`) prove the matrix's decisions directly, which is stronger evidence than a byte
comparison, and the decision-tuple check preserves the byte guard's real intent.

### 9.2 — the brief does not say how hybrid reassignment interacts with the one-active invariant

**The gap.** §F1.3 says `pending → claimed` happens **only** for a compute-idle worker. §F1.6 says the
hybrid retry path is unchanged, and that path claims to a *different* worker chosen by
`_pick_other_worker` — which has never considered busy-ness. At saturation every alternate is busy, so
the two rules cannot both be satisfied by claiming immediately.

**Both alternatives are bad, so I took neither silently:**

- claim to a busy alternate → the reassigned stripe's lease starts while the worker cannot begin it.
  **That is the repaired defect, one stripe deep, on the retry path** — and in a hybrid phase a lease
  expiry there consumes the single retry and then fails the trial.
- make "no *idle* alternate" terminal → every saturated hybrid retry becomes a trial failure. A far
  worse regression than the defect.

**What I implemented, as the minimal reading faithful to both:** the matrix's **terminal decision is
unchanged** (`_pick_other_worker` still answers "does any alternate exist at all", ignoring
busy-ness); only the **placement** is deferred. The stripe returns to `pending` at attempt 1 with
`phase_degraded=1` and the fence bumped, and the scheduler places it on the next idle non-excluded
alternate. A new non-terminal action, `requeued`, is returned when no alternate is free this instant.

**The exclusion uses a retained `claimed_by` on the pending row** (`:5487`) — inert everywhere
(dispatch, lease expiry and completion all read `claimed`) and the durable record of which worker the
scheduler must not choose, so `_pick_other_worker`'s "a DIFFERENT worker" guarantee survives the
deferral. **Consequence to note:** a *requeued* backlog row therefore carries a non-NULL
`claimed_by`, where *initial* backlog carries NULL. `G-F1-QUEUE-NO-LEASE` asserts NULL for initial
backlog; `G-F1-HYBRID-MATRIX` asserts the retained-claimer shape for the requeued case.

**This is what reds `G-LEASE`** (`test_s172_staging_backpressure.py:1394`): it asserts
`out[0]["action"] == "reassigned"` in a two-worker bench where the only alternate is the **paused**
worker. Everything else in that gate still passes — `current_attempt == 1`, `phase_degraded`, the
trial stays `running`, the paused worker's stripe untouched. Note that under the old behaviour that
gate was asserting a reassignment **to a coordinator-paused worker**, which cannot read the
assignment.

**Ruling requested:** ratify deferred placement + the `requeued` action (and I will submit the
one-line `G-LEASE` update as part of that), or specify the interaction you intend.

### 9.3 — `[S172-BP] burst_exact` now measures a smaller set (disclosed, not a defect)

`staging_burst_bound_exact` is documented as "the EXACT burst for a **known assignment**". The known
assignment at stage setup is now `min(idle workers, planned stripes)`, so at 8 workers / 32 stripes
the logged `exact=` falls from 1008 to ~252. **The bound actually in force is unchanged** — the
conservative bound is still derived over **all** planned stripe spans, so nothing shrank. To keep the
figure interpretable rather than looking like a regression against the pre-amendment logs, the line
now carries `claimed=` and `queued=` (`:6740-6757`). Flagging because §2.19 records the
exact-vs-conservative pair (116 vs 136) as Beta-mandated.

### 9.4 — one dependency discovered, disclosed and NOT acted on (§19)

`docs/CLAUDE_CODE_REPORT_GATE12_FAILURE_FORENSICS.md` §E.2 recorded that the gate-12 concurrency
sampler cannot express Beta's saturation criterion, because under bulk claim `pending` never appears
and `claimed_workers` reports assignment rather than occupancy. **F1 changes that**: `pending` is now
a real, meaningful queue, which is the property Beta's §4 note anticipates ("restores the natural
meaning of the queue that gate-12 saturation evidence needs"). The corrected sampler block already
exists in that report §E.3. **Not applied** — it lives in `gate12_launch.sh`, which is launch
tooling, and launching is held.

---

## Observation carried, not acted on (as instructed)

`elapsed_s` (S172-R4, `range_miner_worker.py:1345`) is `time.time() - t0` across the worker's whole
stripe. **Under this design that interval becomes exactly the compute-active lease window** — it
starts when the worker begins the stripe, which is now also when the lease is stamped. So
`compute_lease_timeout` can be sized from measurement rather than convention. Noted only.

The measured values from the failed run make the point: worker `elapsed_s` was **4.46–6.66 s** per
stripe while the coordinator-side ingest span reached **95–121 s**. Under F1 the lease now covers the
former plus delivery of that stripe alone, not four stripes' queue wait. **The R4 caveat still binds
any consumer: `elapsed_s` is stripe SERVICE TIME, not fleet throughput — concurrent intervals
overlap.**

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** every suite run this session with `python3 -u … > log 2>&1`, logs retained at
  `/tmp/f1_final_*.log` (final tree) and `/tmp/f1_base_*.log` (baseline worktree); counts quoted from
  those files. Source claims from `file:line` read live; the "IDENTICAL / CHANGED" table produced by
  AST extraction against `git show eecfff7:` in this session.
- **clean control:** `G-F1-DEAD-WORKER-STILL-EXPIRES` — a genuinely silent worker still expires and a
  constant phase still fails the trial. Plus the baseline worktree at `eecfff7` as the regression
  control.
- **fault-injection (positive) control:** three mutants, all RED — bulk claim restored; durable
  terminal record dropped; terminal ERROR log dropped. `_mutant_red` fails loudly if a mutant
  survives.
- **completion sentinel:** each suite's own `N/N checks green` line plus process exit code.
- **unavailable-observer behaviour:** stated as such — `admission_binding` C5 is reported as
  **pre-existing and not diagnosed**, not as passing; Gate 22 is reported as a known untracked-file
  artefact, not as green.
- **audit claim scope:** this amendment only. **No claim about production behaviour at scale** — the
  gates are deterministic unit-level reconstructions, and **nothing was executed on the fleet.** The
  25-worker shape is explicitly not evidenced here.
- **searched surfaces:** live `miner/range_miner_coordinator.py`, `miner/range_miner_worker.py`,
  `tests/test_s172_{phase4_coordinator,staging_backpressure,staging_partb,admission_liveness,admission_binding,elapsed_roundtrip,phase5_d3_5_finalizer}.py`;
  `git show eecfff7:` for the baseline; live host process/port state; `docs/` — this brief,
  `CLAUDE_CODE_REPORT_GATE12_FAILURE_FORENSICS.md`, `TB_SUBMISSION_GATE12_FORENSICS_F1_DEFECT.md`;
  `CLAUDE.md`; tfm-project-facts skill v21 (§2.19, §2.25, §2.26 govern this work).
- **unavailable surfaces:** the fleet (launch held); any 25-worker or production-shape behaviour; the
  rigs (not contacted this session); `admission_binding` C5's root cause (pre-existing, out of scope).
- **governance trail searched:** this brief; skill §2.19 (S172-BP three-ruling F1 arc — note the F2
  naming collision), §2.24 (both gate-12 prerequisite amendments), §2.25 (gate-12 attempt 1),
  §2.26 (the forensics that produced this amendment); the S172-BP amendment blocks quoted inline in
  `range_miner_coordinator.py`.
- **chapters searched:** none — no claim here concerns sieve mathematics, feature semantics or
  pipeline intent.
