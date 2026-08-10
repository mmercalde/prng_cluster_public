# CLAUDE CODE INSTRUCTIONS — DEFECT A: RANGE-MINER TRANSPORT-SESSION RECOVERY

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD **`4c76f42`**. `source
~/venvs/torch/bin/activate` before every test.

**Authority:** Team Beta ruling *"GATE-12 ATTEMPT-2 FORENSIC RULING"* (2026-08-10), §§7–16. This
brief implements **Defect A only**. Defect B (sampler turnover aggregation) is a separate brief.

**Hard constraints — no commit, no push, no launch, no fleet, no real-rig SSH.** Gate 12 and
attempt 3 HELD. **NO-TOUCH surfaces (Beta §25) — if implementation proves any must change, STOP and
return the conflict, do not edit to make a gate pass:** `expected_workers` semantics · frozen cohort
membership · the *live*-eligibility requirement · constant-phase failure policy · hybrid retry budget
· lease semantics · `worker_admission_timeout=180s` · staging/backpressure · seed domain · coverage
authority · dataset authority · publication authority.

---

## THE DEFECT, READ FROM SOURCE (do not re-derive from the ruling's prose — confirm these lines)

`miner/range_miner_worker.py` `serve_forever()` (~`:1423-1431`):

```python
while not self._stop.is_set():
    try:
        msg = self.conn.recv_msg()
    except (ConnectionError, ValueError, OSError):
        break                      # ← transport loss and explicit shutdown exit at the SAME point
    self._dispatch(msg)
finally:
    self.shutdown()
```

`main()` (`:1490-1521`) then `return 0` — so a transport blip terminates the process **silently and
successfully**. `_dispatch` sets `_stop` on a `shutdown` frame; `_handle_sig` sets it on SIGTERM/SIGINT.
**All three exits collapse into one.** That collapse is the defect: there is no reconnect path, so a
permitted worker that loses its session can never re-register, and a later stage's admission is
permanently short (attempt-2's 23/25).

**The coordinator already has the useful half (Beta §7) — do NOT rebuild it.** `_drop_conn`
(`range_miner_coordinator.py:7365-7389`) evicts a dead socket's identity from `wconn_by_worker` /
`registered`, and already guards the fenced-replacement case (`fs_by_worker.get(wid) is fs`) so a
same-worker_id rebind on a *different* live socket is not evicted. The duplicate-rejection path
(Defect 3, Gate 58) handles a reconnect that races the eviction. **The missing seam is worker-side
only.** Confirm this before writing; if the coordinator does NOT accept a same-identity
re-registration after eviction, that is a finding — stop and report it.

## WHAT TO BUILD — Beta §10 three-way session state machine

The worker must distinguish, and act differently on, three exit causes:

| cause | detection | action |
|---|---|---|
| **EXPLICIT SHUTDOWN** | `shutdown` frame → `_stop` set via `_dispatch` | exit, **NO reconnect** |
| **SIGNAL** | SIGTERM/SIGINT → `_stop` set via `_handle_sig` | exit, **NO reconnect** |
| **TRANSPORT LOSS** | `ConnectionError`/`OSError`/`ValueError`/unexpected EOF/framing failure, **with `_stop` NOT set** | close dead session → bounded-backoff reconnect → re-register SAME worker_id → resume idle service loop |

**The load-bearing discriminator:** on catching the transport exception, branch on `self._stop.is_set()`.
If `_stop` is set, a shutdown frame or signal already fired — exit, no reconnect. If `_stop` is clear,
it is a genuine transport loss — recover. Do not reconnect after an explicit stop; that is §10's
no-reconnect requirement and A6 gates it.

## RECONNECT MECHANICS — Beta §§11–14, every one is a gate

- **§11 NO STALE-WORK REPLAY (load-bearing).** Reconnect is transport recovery, NOT
  assignment-recovery. A worker that reconnects must **not** replay an old `SubStripeResult`, resume a
  stripe from private state, declare an old assignment complete, or self-create a retry. The certified
  F1/F2 machinery alone decides the abandoned assignment (constant loss → terminal; hybrid first loss →
  certified retry; hybrid second → terminal). On reconnect the worker re-registers to an **idle**
  service state and waits for new dispatch. Gate A3/A4.
- **§12 ONE ACTIVE CONNECTION PER IDENTITY.** If the reconnect races the coordinator's eviction of the
  old socket, the coordinator rejects the duplicate registration. The worker must treat rejection as a
  **retryable session-establishment condition** and retry after backoff — NOT force-replace, NOT open a
  second live socket. Gate A5.
- **§13 FROZEN COHORT REMAINS AUTHORITY.** Re-registration carries the **same** worker_id, node
  identity, backend, GPU/device identity, supported variants, capability signature. A reconnect with an
  altered capability identity must fail closed. No cohort expansion. Gate A7.
- **§14 RECONNECT BOUND.** A **positive finite run-scoped** bound with bounded backoff — no immortal
  orphan that could attach to a later unrelated trial. Tie the bound to the existing liveness contract
  (the `worker_admission_timeout`/`_tcp_wait_ready` 180 s authority is the natural anchor — a worker
  whose recovery exceeds the window the coordinator would wait cannot help this trial anyway) or another
  **derived** run-scoped authority. **Do not invent an unrelated giant timeout, and do not change
  `worker_admission_timeout` itself** (no-touch). On exhaustion: worker exits cleanly. Gate A8.

## OBSERVABILITY — Beta §15 (required WITH the fix, not after)

Attempt 2 could not name the two lost workers because teardown was silent. Add explicit records; **no
high-rate heartbeat noise.**

- **Worker-side:** worker_id · connection/session generation · disconnect reason / exception class ·
  whether an assignment was active at loss · reconnect attempt number · reconnect success · reconnect
  exhaustion · explicit-shutdown-vs-transport-loss classification.
- **Coordinator-side:** `WORKER_DISCONNECTED{worker_id, stage_idx, stage_assigned,
  eligible_count_after_drop}` and `WORKER_REGISTERED/RECONNECTED{worker_id,
  eligible_count_after_register}`. These bracket the drop so a future 23/25 event names the two IDs and
  the transition, without indirect reconstruction. The coordinator emits these at `_drop_conn` and at
  the registration handler — small additions at existing seams, not a new subsystem.

## REQUIRED ADVERSARIAL GATES — Beta §16, all eight, at 25 logical workers × 4 stages

**No GPUs required** — controlled workers/executors (loopback sockets, stub executors) are authorized.
The fixture must be a real 25-worker, four-stage trial shape, because small-count fixtures are what
missed this (§27 correlated-blind-spot). Build the harness so the dimension list is exercised, not
just the motivating case (self-check #14).

- **A1 — exact production blind spot (the reproduction):** 25 admitted; stages 1-2-3 complete; drop
  transport for **two idle** workers at the stage-3→4 boundary; the **same two processes** reconnect and
  re-register the same identities; stage 4 admits 25, completes; trial completes.
- **A2 — no dynamic downsizing:** same fixture, reconnection **disabled** → 23/25 → admission timeout →
  `worker_admission_timeout`. Stage 4 NOT assigned at 23. (Proves the fix is what closes it, and that
  §4.3 admission is untouched.)
- **A3 — constant-phase active loss unchanged:** drop transport while a **constant-phase** stripe is
  active → the existing F1/F2 terminal-failure semantics are NOT erased by reconnect.
- **A4 — hybrid retry unchanged:** drop an **active hybrid** worker → retry consumed exactly once, the
  failed worker cannot self-reclaim its failed attempt, an alternate executes the certified retry;
  reconnect restores **future eligibility only**.
- **A5 — duplicate-socket race:** reconnect **before** old-socket eviction → duplicate rejected, old
  identity remains singular, worker retries, eventual reconnect after eviction succeeds.
- **A6 — explicit shutdown:** `shutdown` frame → worker exits → **zero** reconnect attempts.
- **A7 — frozen-cohort wall:** reconnect with same identity succeeds; reconnect with changed/non-frozen
  identity fails closed.
- **A8 — retry exhaustion:** coordinator unavailable past the recovery bound → worker exits cleanly,
  does not haunt a future run.

Red-first + mutation evidence per arm. A2 is the clean control for A1; A6 is the clean control for the
reconnect branch (it must NOT fire on a real stop).

## ROOT-CAUSE FORENSICS FIRST — Beta §23 (read-only, before or alongside)

Perform a read-only pass to distinguish transport exception / worker process termination / remote OS
kill / coordinator close for the two attempt-2 workers. Seek: worker logs around the disappearance,
kernel/OOM records, process-termination records, coordinator-side EOF timing, network reset evidence.
**Do NOT claim "TCP idle timeout" or any initiating cause until evidence establishes it** — this is the
correction Beta issued against Alpha's forensic report. If the evidence is silent, report it silent.
The amendment is justified independently (the no-reconnect gap is a defect regardless), so do **not**
block implementation on finding a perfect explanation.

## VERIFICATION

New worker-recovery suite (A1–A8) + full re-run of the certified suites this touches:
`test_s172_f1_f2_active_lease.py` (16/16) · `test_s172_phase4_coordinator.py` (63/63) — because the
coordinator observability additions and the duplicate-identity path border them. Confirm byte-unchanged
`sha256` for every no-touch file that borders the change. Long suites `python3 -u <suite> | tee
/tmp/<n>.log`, never `tail`.

## REPORT — `docs/CLAUDE_CODE_REPORT_DEFECT_A_TRANSPORT_RECOVERY.md`

1. The §10 state machine as implemented, with the `_stop`-discriminator and the source lines.
2. §§11–14 compliance, one subsection each, with the gate that proves it.
3. §15 observability records added and their emit sites.
4. §23 forensic pass: what the evidence shows about the two workers, or an explicit statement that it
   is silent — **no unproven initiating cause.**
5. A1–A8 with red-first/mutation evidence; A2/A6 as the controls.
6. Byte-unchanged confirmation for the no-touch list. Files changed from `git status`.
7. Any no-touch conflict encountered — **returned, not worked around** (Beta §25).
