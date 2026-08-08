# CLAUDE CODE REPORT — S172-BP AMENDMENT (Beta findings F1–F5)

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_BP_AMENDMENT.md`
**Authority:** Team Beta ruling *"S172 STAGING BACK-PRESSURE REMEDIATION — HOLD, TARGETED
FIX-FORWARD REQUIRED"* (2026-08-06), reviewing `4b1aad6`.
**Base:** fix-forward on `4b1aad6` (tip `42bdbb1`). No history rewrite.
**Host:** VM101 (`zeus-ubuntu`, 192.168.3.177), `~/distributed_prng_analysis`, venv
`~/venvs/torch`. **Not committed, not pushed, no pipeline launched, gate 12 / soak NOT run.**

**Result: 28/28 green on VM101** (19 pre-existing + 9 new), five consecutive full runs.
`test_s172_staging_partb.py` **24/24**. `test_s172_phase4_coordinator.py` **63/63 with the
production diff applied**; the one red on the working tree is Gate 22 observing the
*uncommitted* test file and is isolated below (§6.2).

**Files changed — exactly two, as required:**

| file | + / − |
|---|---|
| `miner/range_miner_coordinator.py` | +439 / −57 |
| `tests/test_s172_staging_backpressure.py` | +820 / −12 |

No worker code, no seed caps, no stripe geometry, no `gate_s172_prod_shape.py`. Zero new
module-scope imports (AST-diffed against `4b1aad6`: added NONE, removed NONE). `py_compile`
clean on both.

---

## 1. F1 — one wake per capacity release

**Implemented as an ingress-credit counter under `_pause_lock`, exactly as specified.**

| element | anchor (live) |
|---|---|
| `_resume_credits_outstanding` / `_resume_credit_holder` | `:1997` |
| `_grant_resume_credit()` | `:3141` |
| `_try_self_resume(conn_key)` | `:3187` |
| `_release_resume_credit(conn_key, delivered)` | `:3222` |
| `resume_credits_outstanding()` (observability) | `:3240` |
| `_resume_paused_connections()` — now **one call**, loop deleted | `:3244` |
| reader: defensive poll → `self._try_self_resume(rawsock)` | `:5271` |
| reader: credit consumed after the successful `inbound.put` | `:5300` |
| reader: credit released unconditionally on exit | `:5314` |

1. **`_grant_resume_credit`** takes `_pause_lock`, refuses if `_resume_credits_outstanding != 0`,
   finds the **FIFO-oldest unsignaled** paused record, then checks `staging_can_accept()`,
   increments the counter, sets that one event, logs `[S172-BP] resume_signal … credits_outstanding=1`
   and **returns**. One invocation ⇒ at most one wake. The capacity check is taken *inside*
   `_pause_lock` so the observation and the credit are taken together; that is safe because
   `staging_can_accept()` never takes `_pause_lock` and its semaphore probe is
   acquire-then-immediately-release.
2. **Credit consumption** is reader-side under `_pause_lock`, immediately after the successful
   `inbound.put` (`:5300`), and unconditionally on reader exit (`:5314`). Until cleared, no
   further grant is issued by either door — a wake reserves the observation.
3. **The defensive poll is now a head-only self-grant.** `_try_self_resume` succeeds only when
   the connection is the FIFO-oldest paused entry **and** `_resume_credits_outstanding == 0`
   **and** `staging_can_accept()`; on success it takes the credit itself. The lost-wakeup
   protection is preserved (the head can always escape when capacity truly exists and no grant
   is in flight) and a non-head reader can no longer ride someone else's observation. The
   `resume_event.wait(0.05)` cadence is unchanged.
4. **The documented §1.2 margin is unchanged and nothing was resized.** It is named as the final
   backstop in the `_grant_resume_credit` and `_try_self_resume` docstrings.

**One design detail the brief does not name, and it is load-bearing.** In the reader's `finally`,
**deregistration must precede the credit release** (`:5311`, then `:5314`). A grant can only target a
connection still present in `_paused_connections`; clearing the credit first leaves a window in
which a grant lands on a record that is about to be removed, and then *nobody* ever clears it —
`_resume_credits_outstanding` sticks at 1 and the entire paused fleet wedges. The ordering is
enforced with a comment saying so.

**Gate G-RESUME-CREDIT** — two parts plus mutation evidence.

* **part (a)**, `gate_resume_credit_one_wake_per_release`: two connections registered through the
  real registry API, **capacity held wide open for the whole gate**, so the only thing that can
  stop a second wake is the credit. One `_release_capacity()` ⇒ only the FIFO-first event is set;
  five more release events change nothing while the credit is outstanding; a non-head
  `_try_self_resume` is refused; after the credit clears the head self-grants and the second
  reader resumes; a non-holder cannot clear the credit. Fully deterministic — no reader threads,
  because the decision is synchronous inside the call.
* **part (b)**, `gate_resume_credit_real_readers_fifo`: two **real** reader threads on real framed
  sockets; ≥2 paused connections in asserted FIFO entry order; **exactly one** staging capacity
  unit freed and **exactly one** capacity-release path invoked; the unit is then reclaimed
  (modelling the serve loop staging the resumed envelope, which is what closes capacity in
  production); the FIFO-first reader resumes and delivers, the second **remains paused across a
  0.6 s settling window** (≥ 12 of its 50 ms poll cycles, re-asserted every 20 ms); a second unit
  is freed and the second reader resumes. The gate records the inbound depth at reclaim time so a
  lost capacity race is reported as itself rather than as a confusing "the second reader resumed".
* **mutation evidence**, `gate_resume_credit_mutants` (both doors): **mutant 1 restores the loop**
  as `_resume_paused_connections` — it executes (asserted) and wakes **both** readers on one
  capacity-release invocation, redding part (a)'s credited assertion. **Mutant 2** replaces
  `_try_self_resume` with the bare `staging_can_accept()` escape — it executes (asserted) and
  returns True for a **non-head** connection, redding the head-only assertion.

---

## 2. F2 — lease-exemption resume grace

Implemented as Beta specified, item by item.

| item | implementation | anchor |
|---|---|---|
| 1 | `_capacity_resume_grace: Dict[str, float]` under `_pause_lock` | `:2009` |
| 2 | `deregister_paused_connection`, `reason == "resume"` and `worker_id is not None` ⇒ `now + compute_lease_timeout`, written **inside the same `_pause_lock` critical section** that pops the record | `:3079` |
| 3 | `process_lease_expiry` skips a stripe whose `claimed_by` is actively paused **or** has a live grace entry; `capacity_resume_grace(now)` **prunes expired entries in the same pass** | `:3981`, `:3252` |
| 4 | heartbeat branch of `_serve_dispatch` clears that worker's grace after `renew_lease` **succeeds** | `:5451` |
| 5 | cleared on connection drop (`_drop_conn`, inside the identity-eviction branch, so a fenced replacement is not affected) and at trial-terminal cleanup (`serve_trial`'s `finally`) | `:5345`, `:5132` |
| 6 | grace expiry with no heartbeat ⇒ the skip simply stops matching and normal expiry resumes | `:3252` + G-LEASE-HANDOFF arm 3 |

**No ledger mutation.** The grace is a coordinator dict written under `_pause_lock` on the reader
thread; the reader rule ("touches NO ledger state") is preserved and the comment at `:3079` says
so explicitly.

**Deviation from the brief's literal wording, stated rather than worked around (item 4).** The
brief says *"after `renew_lease` succeeds"*. `MinerLedger.renew_lease` returns a bool and can
legitimately return False (the stripe left `claimed`, or was re-claimed by another worker), so the
clear is gated on that return value, not merely on reaching the call. A renew that did not land has
not restored the lease, so the bridge must stay up until its own bound. This is the stricter
reading of Beta's sentence.

**Gate G-LEASE-HANDOFF** (`gate_lease_handoff_grace`), three arms on the real reader loop:
1. A worker is paused past its compute-lease deadline (lease set to `t0 − 1`, hybrid phase so a
   genuine expiry is observable without the constant-phase row killing the trial). The renewing
   heartbeat is written **while paused**, so TCP ordering leaves it behind the held result. On
   resume the reader deregisters, the result and then the heartbeat arrive in order, and
   `process_lease_expiry` is run **inside that window** with the heartbeat drained but not
   dispatched → **zero matrix entries**, `out == []`, the stripe still `claimed`, attempt 0, not
   degraded.
2. The heartbeat is then dispatched → `lease_expires_at` advances past now (real renewal) and the
   grace is **cleared**.
3. Separate bench: a resumed worker that never heartbeats. `process_lease_expiry` at
   `time.time() + 301` prunes the grace and the stripe **does** reach the matrix, and the entry is
   gone — the exemption is bounded.

**Mutation evidence** (`gate_lease_handoff_mutant`): the grace **recording** is removed (the
pre-amendment state of the world — deregistration wrote nothing). The mutant executes (asserted)
and the resumed worker's expiry lands in the matrix with `lease_expiry=True`, redding the credited
assertion.

---

## 3. F3 — timeout evidence snapshot

`staging_capacity_timeout_expired` (`:3291`) now takes the latch decision **entirely inside
`_pause_lock`**, re-checking the latch under the lock (two readers and the serve loop can all
reach it, and exactly one must take the snapshot), and captures
`_capacity_timeout_snapshot = {latched_at, oldest_since, paused_count, worker_ids}` in the **same
critical section as the oldest-pause read that decided the timeout**.

`staging_capacity_timeout_reason` (`:3335`) and `staging_backpressure_metrics` (`:3369`) use the
snapshot when present and consult the live registry **only** if the timeout never latched. The
reason additionally carries `oldest pause held N.Ns at the latch`. Metrics gain
`capacity_timeout_snapshot`, `paused_at_capacity_timeout` and `capacity_timeout_worker_ids`;
`capacity_timeout_snapshot()` (`:3329`) is the public accessor.

**Gate G-TIMEOUT-SNAPSHOT** (`gate_timeout_snapshot_attributes_the_trigger`):
`staging_capacity_timeout = 0.3`, one paused reader; the gate then **waits for the reader thread to
observe the latch, deregister and fully exit** (`thread.is_alive()` is False and
`paused_connection_count() == 0` are both asserted as preconditions — otherwise the race the gate
is about cannot occur and the gate would be measuring nothing). The abort reason must name
`hostA:gpu0` and `1 connections paused`, must not contain `(none)`, and the metrics must carry the
same `latched_at`. The reason assertions come **before** anything touches the new snapshot API, so
the gate reds against `4b1aad6` on behaviour.

---

## 4. F4 — registered workers only

The reader's pause condition now additionally requires `worker_by_sock.get(rawsock) is not None`
(`:5243`); the bound identity is read once into `bound_worker_id` and reused for
`register_paused_connection`. `worker_by_sock` is written only at registration
(`_serve_register` → `:5404`), so this **is** the bound-worker predicate.

An unbound result under saturation is **not paused and not held**: it flows to `inbound` unchanged
and dies in the existing serve-loop identity rejection (`_serve_dispatch`, `:5429` — `bound_worker_id
is None` ⇒ drop, no ledger mutation). **No new rejection logic was added.**

Note for the record: when `worker_by_sock` is `None` (the parameter's default; production's
`serve_trial` always passes the dict at `:4800`) no connection can pause at all. That is the
fail-closed direction and matches the predicate's meaning — "unregistered" — rather than
introducing a second identity source.

**Gate G-BOUND-PAUSE** (`gate_unbound_result_is_never_paused`): staging saturated, a real `_Peer`
constructed with `bind=False` (a socket that connected but never registered) sends a valid
`sub_stripe_result`. Asserted: the frame is delivered to `inbound` (not intercepted); no pause
record; no `paused_worker_ids` entry; **no grace record**; **no snapshot** and
`staging_capacity_timeout_expired(now + 10_000)` still False — the stray never joined the
oldest-pause clock and cannot time out a trial it was never part of; the message reaches the
existing identity rejection with **zero matrix entries and zero shard rows**. Narrowness is
asserted in the same gate: a **registered** worker under the same saturation still pauses.

---

## 5. F5 — stage-bound derivation failure fails closed

`serve_trial`'s stage-setup block (`:4993`) materializes `_stripe_spans`, `_eligible_records` and
the exact-bound rows **once at entry**, and on any derivation exception logs the full context
(stage, run, family, phase, assignment count, eligible count, spans, central caps) and terminates
via **direct** `fail_trial(run_id, reason="coordinator_staging_sizing: could not derive the staging
deferred bound for stage <n> — <Type>: <cause>")` followed by `continue` — so `stage_assigned` is
never set and `_dispatch_pending` is never reached. **Never the matrix, never a smaller implicit
bound, and before any result traffic for that stage.**

`_derive_bound_from_current_state` (`:2932`) carries an explicit ⚠ comment: it is **not a production
fallback**, it survives only for bare-API/gate contexts, it answers a different question (one
macro-stripe, phase 1) and every production stage installs its bound at setup or fails closed.

**Widening stated rather than worked around.** The brief says *"on any derivation exception"*; the
handler is therefore `except Exception`, not the brief's quoted `(ValueError, TypeError)`. That
matters, and the red run proves why: at `4b1aad6` a malformed cap record raising **KeyError** was
not caught by that tuple at all and escaped `serve_trial` as an unhandled exception. Beta's §6
names the swallow; the same handler had a second hole beside it. Both are closed by failing closed.

**Beta item-3 ratification detail.** `_defer_locked` (`:3419`) records which bound it refused on in
`_last_defer_refusal` (`:2031`) — `derived_count_bound` / `operator_override_count_bound` /
`retained_bytes_high_water` — and the §1.6 invariant reason (`:3572`) leads with
`bound_tripped=<phrase>` and one of three explicit sentences: *"the DERIVED COUNT bound
(burst_conservative + resume_margin) was exceeded — the stage sizing was wrong"*, *"the OPERATOR
OVERRIDE COUNT bound (staging_deferred_max) was exceeded …"*, *"the RETAINED-BYTES HIGH-WATER
(staging_high_water_bytes) was exceeded — retained coordinator RAM, not the count bound"*. They are
three different defects with three different owners. **The arithmetic already carried is retained
unchanged** and now follows the classification.

**Gate G-BOUND-DERIVATION-FAILURE** (`gate_bound_derivation_failure_fails_closed`), through the real
`run_trial_miner` → `serve_trial` path with a real loopback worker, **two arms**:

| arm | malformed worker-cap record | exception |
|---|---|---|
| `zero_caps` | all four advertised caps set to 0 → `expected_substripes_for` refuses a non-positive cap | **ValueError** — the exact class §6's handler swallowed |
| `missing_cap_key` | `amd_hybrid` removed → `advertised_effective_cap` builds `VramCaps` from all four | **KeyError** — the class that escaped the handler entirely |

The record is injected **after** the real `assign_stripes` and **before** the sizing call, so the
failure lands in the seam under test; the injection is undone at the **synchronous L7 abort
discharge** (`_Sink.on_abort` fires at the instant the trial becomes terminal), so the gate measures
the fail-closed decision rather than a teardown artefact of its own fault injection. Both arms
assert: injection executed and was undone; `state == "aborted"`; reason starts
`coordinator_staging_sizing:`, names `stage 0` and carries the exception type; **zero matrix
entries**; every stripe at `current_attempt == 0` and not `phase_degraded`; **the worker was handed
no assignment** (`w.assigned == []`), no fetches, nothing published; and execution never continued
on the one-slot fallback — `derived_bound is None`, `staging_jobs_completed == 0`,
`deferred_high_water == 0`, `capacity_invariant_terminations == 0`.

**`G-BOUND-TRIP-PHRASE`** (`gate_invariant_reason_names_which_bound_tripped`) asserts the three
phrases are present in `enqueue_staging`'s live source **and** that the classification is derived
from the real refusal: three `_defer_locked` drives (operator-override count trip, derived count
trip, retained-bytes trip) each produce the matching `_last_defer_refusal`.

---

## 6. Red / green per gate

### 6.1 The nine new checks

Red runs come from a **git worktree detached at `4b1aad6`** with this suite copied in
(`git worktree add --detach … 4b1aad6`, `git status --porcelain` = only the suite modified). Green
runs are VM101, `~/venvs/torch`, working tree.

| gate | RED at `4b1aad6` — the actual failure text | GREEN |
|---|---|---|
| **G-RESUME-CREDIT-a** | `ONE capacity observation woke MORE THAN ONE reader — the wake did not consume the observation (thundering herd)` | PASS |
| **G-RESUME-CREDIT-b** | `expected ONLY hostB still paused, got frozenset()` — one freed unit resumed the **entire** paused fleet | PASS |
| **G-MUT-RESUME-CREDIT** | `'RangeMinerCoordinator' object has no attribute '_try_self_resume'` (the head-only door does not exist at `4b1aad6`) | PASS |
| **G-LEASE-HANDOFF** | `a coordinator-caused silence entered the matrix during the resume handoff: [{'run_id': 'runLH', 'stripe_id': 'runLH_sA', 'retryable': True, 'lease_expiry': True}]` | PASS |
| **G-MUT-LEASE-HANDOFF** | `'RangeMinerCoordinator' object has no attribute 'capacity_resume_grace'` | PASS |
| **G-TIMEOUT-SNAPSHOT** | `the terminal reason does not name the worker whose pause caused the timeout: 'coordinator_staging_capacity_timeout: staging did not release capacity within 0.3s; 0 connections paused (none)'` | PASS |
| **G-BOUND-PAUSE** | `an UNBOUND sub_stripe_result was intercepted by the capacity gate instead of flowing to the existing identity check: []` | PASS |
| **G-BOUND-DERIVATION-FAILURE** | trial did **not** fail closed — see the trace below | PASS |
| **G-BOUND-TRIP-PHRASE** | `the §1.6 invariant reason cannot say 'DERIVED COUNT bound' — it does not distinguish which bound tripped` | PASS |

Totals: **19/28 at `4b1aad6`** (the 19 pre-existing gates green there, as they must be) →
**28/28 on the working tree**.

**The `zero_caps` red is worth reading in full**, because it is Beta's §6 defect executing end to
end at `4b1aad6`:

```
[S172-BP] could not derive the staging deferred bound for stage 0 — falling back to the on-demand derivation
…
serve_trial → _serve_dispatch → enqueue_staging → _defer_locked
  → staging_deferred_bound() → _derive_bound_from_current_state()
  → staging_burst_bound_conservative → expected_substripes_for
ValueError: effective_cap must be positive, got 0
```

The stage derivation failed, was swallowed, **the whole stage was dispatched, results came back**,
and the on-demand fallback then raised on the dispatch loop under `_admission_lock` — an unhandled
exception out of `serve_trial`, *after* real result traffic. That is precisely what "fail closed
before any result traffic" prevents.

### 6.2 The pre-existing suites, and one thing this amendment had to repair

| suite | result |
|---|---|
| `tests/test_s172_staging_backpressure.py` | **28/28** — five consecutive full runs, VM101 |
| `tests/test_s172_staging_partb.py` | **24/24**, VM101 |
| `tests/test_s172_phase4_coordinator.py` | **63/63** with the production diff applied; **62/63** on the working tree — Gate 22 only |

**Gate 22 (`test_s172_phase4_coordinator.py`) — the documented working-tree condition, isolated.**
Its red is `unexpected changed .py files: {'tests/test_s172_staging_backpressure.py'}`. Gate 22
builds `changed_py` from `git status --porcelain` and compares it against an explicit allowlist
(`:1611`) that contains `miner/range_miner_coordinator.py` but not this suite. Isolation evidence,
run in the `4b1aad6` worktree:

* clean worktree at `4b1aad6` → **63/63**;
* worktree with **only** `miner/range_miner_coordinator.py` replaced by the amendment's version
  (suite file clean) → **63/63**.

So the red is caused solely by the *uncommitted* test file and clears at commit. Gate 22 lives in
an out-of-scope file and was **not touched, and not widened** — consistent with the standing
disposition for this exact condition.

**G-MATRIX-DIFF-a was already red at `42bdbb1` before this amendment, and it had to be repaired.**
It compared the working tree against `git show HEAD:miner/range_miner_coordinator.py`, which was
correct only while the remediation was uncommitted. The moment `4b1aad6` landed, `HEAD` *became* the
post-fix file, so `before` returned 6 and the gate asserted `len(before) == 7` — it red on its own
success. A gate whose baseline moves with the work cannot certify the work.

The fix is in scope (the suite is one of the two permitted files) and pins two revisions instead of
tracking `HEAD`:

```
_PRE_REMEDIATION_REV     = 7c4f11b1b9910f868b56906f05f7269f58fba53e   # parent of 4b1aad6
_AMENDMENT_BASELINE_REV  = 4b1aad6ddfa7e6f6a3082a7850fe71b7ae7825b8   # the ruling's subject
```

and adds the amendment's own claim as an assertion: `after == at_baseline`, plus the three matrix
methods compared against **both** baselines. The gate now passes at `4b1aad6` **and** on the
working tree, which is the property it should always have had.

### 6.3 AST evidence — the six survivors and the matrix are untouched (resubmission item 7)

```
=== G-MATRIX-DIFF AST EVIDENCE (S172-BP amendment) ===
_on_staging_failed call sites: 7c4f11b=7  4b1aad6=6  live=6
removed at 4b1aad6 (1):
   - self._on_staging_failed(run_id, stripe_id, True, eligible_provider,
       'staging deferred queue full — dispatch back-pressure')
six survivors, AST-normalised, identical at 4b1aad6 and live: True
   1. self._on_staging_failed(run_id, stripe_id, False, eligible_provider, f'attempt cannot fit staging …
   2. self._on_staging_failed(run_id, stripe_id, retryable=True, eligible_provider=eligible_provider, …
   3. self._on_staging_failed(run_id, stripe_id, True, eligible_provider, 'hash mismatch on advertised …
   4. self._on_staging_failed(run_id, stripe_id, True, eligible_provider, 'staging timeout')
   5. self._on_staging_failed(run_id, stripe_id, False, eligible_provider, f'staging configuration err…
   6. self._on_staging_failed(run_id, stripe_id, True, eligible_provider, str(e))
_on_staging_failed:            7c4f11b==live True | 4b1aad6==live True
handle_stripe_failure:         7c4f11b==live True | 4b1aad6==live True
_handle_stripe_failure_locked: 7c4f11b==live True | 4b1aad6==live True
```

G-LAW and G-MATRIX-DIFF-b (the behavioural half, all eight classification rows driven through the
real matrix) are green unchanged.

---

## 7. Anchors verified live before writing

Every line number in the brief was checked against the committed tree before any edit. All five
resolved:

| brief anchor | live at `4b1aad6` | verdict |
|---|---|---|
| F2.4 heartbeat branch of `_serve_dispatch` `:5067` | `if mt == "heartbeat":` at `:5067` | ✅ |
| F4 `worker_by_sock` written at `:5035` | `worker_by_sock[rawsock] = msg.worker_id` at `:5035`, sole write | ✅ |
| F1 `_resume_paused_connections` loop | `:3070–3095` | ✅ |
| F1 reader defensive poll `if self.staging_can_accept(): released = True` | `:4922` | ✅ |
| F3 latch / reason / metrics | `:3110`, `:3129`, `:3149` | ✅ |
| F5 stage-setup `except (ValueError, TypeError)` | `:4684–4690` | ✅ |
| F5 `_derive_bound_from_current_state` | `:2899` | ✅ |

---

## 8. Items reported rather than worked around

1. **G-MATRIX-DIFF-a was already broken at `42bdbb1`** by its own `HEAD`-relative baseline (§6.2).
   Repaired in scope by pinning `7c4f11b` and `4b1aad6`. Flagged because the resubmission depends
   on this gate's output and a reviewer should know it was red before F1–F5 were written, not
   because of them.
2. **F5's handler is `except Exception`, not `(ValueError, TypeError)`** (§5). The brief says "any
   derivation exception"; the tuple in the pre-amendment code did not catch KeyError, which escaped
   `serve_trial` uncaught. Both holes close together.
3. **F2 item 4 is gated on `renew_lease`'s return value**, not merely on reaching the call (§2).
   Stricter than the brief's wording, in the fail-closed direction.
4. **Reader-exit ordering (F1)**: deregistration must precede the credit release or the fleet can
   wedge on a credit nobody owns (§1). Not named in the brief; enforced and commented.
5. **Observation, out of scope, not actioned.** `staging_backpressure_metrics()` →
   `staging_deferred_bound()` → `_derive_bound_from_current_state()` can *raise* if a registered
   worker's cap record is malformed, and it is called from the trial-terminal summary at
   `serve_trial`'s tail (`:5151`) — i.e. an unrelated worker-record defect could take out the
   terminal metrics of an otherwise clean trial. F5 makes production stages fail closed long before
   this, and the on-demand derivation is now explicitly not a production fallback, so nothing in
   F1–F5 depends on it. Flagged for a future brief; the G-BOUND-DERIVATION-FAILURE gate works
   around it by undoing its own fault injection at the terminal discharge rather than by changing
   the code.
6. **No other file was needed.** The brief's STOP condition was not reached.

---

## 9. Verification-integrity controls (VIR-1…6)

* **execution proof:** every run piped through `tee` to `/tmp/s172_bp_*.log`; per-gate PASS/FAIL
  lines plus a terminal `28/28 checks green` and a completion sentinel.
* **clean control:** 19 pre-existing gates green at `4b1aad6` in the same red run that reds the nine
  new ones; `partb` 24/24; `phase4` 63/63 clean and 63/63 with the production diff.
* **fault-injection (positive) controls:** four mutants — restore the resume loop; headless
  self-resume; remove the grace recording; two malformed worker-cap records. Each asserts it
  **executed** before asserting it reds.
* **completion sentinel:** `COMPLETION SENTINEL: PASS — S172 staging back-pressure CPU gates green`.
* **unavailable-observer behavior:** none of these gates touches a GPU, rig, fleet or the pipeline;
  nothing reports `UNAVAILABLE`.
* **audit claim scope:** the two in-scope files and the three suites named above, on VM101 only.
  Alpha's independent sandbox re-run is **not** part of this report.
* **searched surfaces:** live `miner/range_miner_coordinator.py` and
  `tests/test_s172_staging_backpressure.py`; `git show` of `7c4f11b` and `4b1aad6`;
  `tests/test_s172_phase4_coordinator.py` (read-only, for Gate 22); `docs/` — the amendment brief
  and `docs/TB_SUBMISSION_S172_STAGING_BACKPRESSURE.md` §4.
* **unavailable surfaces:** Beta's 2026-08-06 ruling text itself is not in the repo — this work is
  driven by the brief, which quotes it. Gate 12 / production-shape and the soak remain **NOT
  AUTHORIZED** and were not run.
* **governance trail searched:** `docs/TB_SUBMISSION_S172_STAGING_BACKPRESSURE.md`,
  `docs/CLAUDE_CODE_INSTRUCTIONS_S172_BP_AMENDMENT.md`,
  `docs/CLAUDE_CODE_INSTRUCTIONS_S172_STAGING_BACKPRESSURE_REMEDIATION.md` (via the suite's own
  header), commit messages `7c4f11b` / `4b1aad6` / `42bdbb1`.
* **chapters searched:** none — this amendment is confined to the coordinator's staging
  back-pressure seam and touches no chapter-governed behaviour.

---

## 10. State on delivery

* **Not committed, not pushed.** Two files modified in the working tree; the amendment brief is
  still untracked. Michael commits and dual-pushes.
* Gate 22 clears at commit (§6.2); everything else is green now.
* **Gate 12 (production shape) and the Phase-7 soak remain NOT AUTHORIZED** pending Beta's review
  of this delta. CPU gates and regression runs were the only things run.
* Suggested `git add` list, built from §"Files changed" and not from recall:
  `miner/range_miner_coordinator.py`, `tests/test_s172_staging_backpressure.py`,
  `docs/CLAUDE_CODE_INSTRUCTIONS_S172_BP_AMENDMENT.md`,
  `docs/CLAUDE_CODE_REPORT_S172_BP_AMENDMENT.md`.
