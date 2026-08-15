# SESSION CHANGELOG — 2026-08-13 — S180 — ATTEMPT-6 REMEDIATION, IMPLEMENTED

**Host:** VM101 `zeus-ubuntu` (192.168.3.177), user `michael`, `~/venvs/torch`.
**Base:** `2b0d2dc`, tree at session start carried one pre-existing untracked document
(`docs/AUDIT_STEP1_OFFSET_REACH.md`) — reported, not assumed away.
**Brief:** `~/dashboard_work/CCODE_BRIEF_ATTEMPT6_IMPLEMENTATION_v1_0.md` (v1.0) — Team Beta RRR
2026-08-13: **DESIGN R3 CERTIFIED, IMPLEMENTATION AUTHORIZED.**
**Design implemented:** `~/dashboard_work/ATTEMPT6_REMEDIATION_DESIGN.md` R3 state — **§11 is the
single operative gate contract**; §4 and the arm tables inside §8 are historical.
**Full report:** `~/dashboard_work/ATTEMPT6_IMPLEMENTATION.md`.
**NOTHING COMMITTED, NOTHING PUSHED, ATTEMPT 6 NOT LAUNCHED.**

---

## 1. What this closes, and what it does not

Gate-12 attempts 4 and 5 died with two facts unrecoverable from their own artifacts:

- **every reader exit was indistinguishable at the point of drop.** Nine ways out of
  `_conn_reader_loop`, none of them recording anything, so a disconnect could only ever be
  described as *"the coordinator performed the final close; the antecedent is UNRESOLVED"*;
- **the serve loop's inbound drain had no time term.** `while drained < 256` let ONE iteration of
  attempt 5 spend **940.971 s** inside the drain, during which lease expiry, admission, dispatch
  and stage advance did not run at all.

This session implements the Beta-certified R3 remediation for both. **It does not diagnose attempt
5.** The initiating cause of the two lost reader sessions remains **UNRESOLVED**, and nothing in
the code, the gates or the report states otherwise: Part A makes the NEXT occurrence
self-describing, and the lost-EOF hazard Part A closes is a derived property of source at `2b0d2dc`,
not a claim about what happened.

## 2. Production changes

**`miner/range_miner_coordinator.py` (+1,472/−72), the only coordinator file touched.**

*Part A — reader-exit cause provenance*
- Ten reader-exit reason constants, `READER_EXIT_REASONS`, and the frozen `ReaderExit` record.
  `READER_EXIT_UNCLASSIFIED` is the INITIAL value — a tenth exit added without a label reports it
  and RXP-1 reds on the next run, where a default of `TRANSPORT_ERROR` would be today's defect
  wearing a reason field.
- `ConnState`: a genuinely **run-scoped `connection_id`** (never `fileno()`, never `id(sock)` —
  both are reused during a long process), the close intent under first-writer-wins, and THE
  per-connection saturation accumulator.
- Every exit of `_conn_reader_loop` assigns its own class. `E3` under `reader_stop` is
  `SHUTDOWN_STOP` (Defect A §10's certified discriminator, applied at the catch); `E4`/`E5` are told
  apart inside one handler because `json.JSONDecodeError` IS a `ValueError`.
- `READER_EXIT` is emitted **at the exit itself** — `WORKER_DISCONNECTED` arrives up to a full drain
  later, and in attempt 5 that gap was 940 s plus 1.46.
- The reasoned EOF stays on the **same `inbound` FIFO** with bounded retry (P-ORD); the
  `timeout=0.5` + `except Exception: pass` swallow is DELETED; a non-`Full` failure is logged at
  ERROR and re-raised.
- `CONNECTION_CLOSE_INTENT` is emitted by `_drop_conn` as its FIRST statement, **bound or not** —
  the R3.4 race (reader fails independently, THEN the read-deadline scan runs) otherwise loses the
  coordinator's own decision entirely.
- Persistent ingress saturation is an **INFRASTRUCTURE TERMINAL for the trial**
  (`TC_INBOUND_SATURATION_TIMEOUT`) over a `SimpleQueue` emergency channel — **no legitimate worker
  is ever shed for it.**

*Part B — control-plane fairness*
- M-1 monotonic drain deadline; **the 256 count is RETAINED as a secondary ceiling**, not lowered
  and not deleted. M-2 folds the accept poll into the same budget. R2.3 clamps the first `get` by
  the remaining budget, so the structural claim holds for every `D < poll` too.
- P-1 first-frame REGISTER priority on an admission `SimpleQueue`, with a per-connection fence, and
  its own bounded service discipline `D_adm` + `A_max` — **the deadline tested only from the SECOND
  disposition**, so one disposition per turn is the progress FLOOR.
- `_serve_register_frame`: the register block **extracted, not copied**; `_serve_register` itself is
  digest-identical.
- Observability that keeps the two unbounded terms nameable: composite `control_block_max`/`_at`,
  `slow_control` and `slow_msg` records, drain-stop counters, `drain_deadline_hits`, in-drain
  occupancy sampling, reader-side saturation counters, admission-queue high-water.
- Four config terms validated FAIL-CLOSED at entry to `serve_trial`, in the same place and shape as
  `worker_admission_timeout`. **`A_max` must be an integer `>= 1`**: `A_max = 0` would silently
  restore admission starvation while every other term still looked correct.

**`miner/range_miner_worker.py` (+190/−0).** `prepare()` (a hoist), `emit_startup_sentinel()` **through
`_emit_session_event`**, `await_session_release()` **failing closed**, three optional CLI arguments,
and `main()` re-ordered to `prepare → sentinel → BARRIER → connect → register → serve`.

**`scripts/gate12_sentinel_gate.py` (NEW).** The 25/25 delivery gate and the release-token writer,
reusing `preflight_check`'s certified three-outcome vocabulary — **a count · `UNAVAILABLE` ·
`ERROR`** — where `UNAVAILABLE` is never rendered as `0`.

**`gate12_launch.sh`, `scripts/launch_fleet_manual.sh`.** The two-phase launch: fleet first
(parked at the barrier), 25/25 verification **outside** the 180 s admission window, then the
coordinator, then the release. The old *"start the coordinator FIRST"* comment is corrected in the
same change — an operator following it would launch into an unreleased fleet.

## 3. Gates — the ten-gate §11 battery

`tests/test_s172_attempt6_remediation.py` (NEW): **71/71 green**, counted from the runner rather
than transcribed — RXP-1 13 · RXP-2 10 (9 arms + clean control) · RXP-3 8 · FAIR-7 7 (6 + clean
control) · FAIR-1/2 5 · FAIR-6 11 · FAIR-3 5 · FAIR-4 6 · FAIR-5 2 (its 8 anchor assertions plus
the self-protection arm) · 4 RED arms on the pinned commit.

RED-arm discipline applied unasked: the FULL 40-character SHA
`2b0d2dc5268946d6b1a44e268573e816b7cdcb83`; **the pinned object is verified to still carry every
defect surface before any RED arm is credited**; a drifted anchor terminates **UNAVAILABLE**, which
never accepts. The probes run over **comment-stripped executable source**, because the repaired file
QUOTES the old surface in its own docstrings and a text probe would credit a drifted anchor.

## 4. Regression battery at final state

phase-4 **62/63** · F1 lease-origin **18/18** · F1/F2 active lease **16/16** · Defect A **29/29** ·
admission-liveness **16/16** · execution-set **34/34** · elapsed-roundtrip **6/6** ·
back-pressure **50/50** · Part B **24/24** · `test_s172_admission_binding` 11/20 **pre-existing**
(differential-identical against `213bfff`, not chargeable).

**Plus three suites the brief's list does not name, run because the WORKER changed:** phase-3
worker (all gates green), phase-2 protocol **6/6**, phase-1 scaffolding **6/6**.

**The one phase-4 red is Gate 22**, on the two NEW untracked `.py` deliverables. **Expected, not a
regression, and NOT a reason to widen the allowlist** (Beta rejected permanent allowlisting); it
self-clears on a clean committed tree. Fifth occurrence.

## 5. Scope proof

Per-definition AST digests against pinned `2b0d2dc`: coordinator **10 changed, 19 added, 0
removed** of 229→248; worker **2 changed, 3 added, 0 removed** of 70→73. Every no-touch surface —
`claim_stripe`, `schedule_pending_stripes`, `renew_lease`, `_renew_active_lease`,
`process_lease_expiry`, `_handle_stripe_failure_locked`, `_execution_set_expected_workers`,
`_serve_register`, `_serve_dispatch`, `dispatch_inbound_result`, `assign_stripes`,
`enqueue_staging`, the credit machinery, `fail_trial`, `commit_trial`, `abort_trial` — is
digest-IDENTICAL, and the §4.3 bounded-admission block is compared as an AST subtree and is
identical too. Gated by FAIR-3 arm 2, so it is a test that runs rather than a claim in a report.

## 6. One harness change that is NOT a production change

`tests/test_s172_staging_backpressure.py` (+16/−16): **sixteen tuple-unpack patterns widened from
four names to five.** The reader-exit record rides in a fifth tuple field and Python cannot unpack a
5-tuple into four names, so the certified battery could not run at all without it. **Proven
mechanical:** mapping the five-wide patterns back to their four-wide originals reproduces the
committed file **byte-for-byte**. No assertion, threshold or expectation changed. Same class as the
G3/G5/G6 bench resequencing Beta ratified on a programmatic identity proof.

## 7. Standing limits, restated so nothing reads as more than it is

- **Attempt 5's initiating reader cause is UNRESOLVED** and is stated as known nowhere.
- **`M_i`, `K_i` and `m_i` are unbounded by this repair.** The certified claims are structural:
  the drain contributes `<= D + one in-flight message`, REGISTER delay is independent of
  data-queue depth, and each admission turn contributes `<= D_adm + at most one overrun
  registration`. **`D_adm + A_max` prevents cumulative admission-queue monopolization; it does NOT
  preempt a single synchronous `m_i`,** and admission latency is never described as absolutely
  bounded. Production latency stays **observable rather than mathematically capped**.
- **The sentinel proves the channel at T0, not for the next four hours.** What it converts is
  "unobserved, cause unknown" into "observed at T0, so any later silence is a CHANGE".
- Nothing was committed, nothing was pushed, and **attempt 6 was not launched.**

## 8. Next

Michael reviews → **Beta certifies the implementation** → Michael commits and dual-pushes → clean
tree → prelaunch battery → only then attempt 6.
