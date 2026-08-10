# CLAUDE CODE REPORT — DEFECT A: RANGE-MINER TRANSPORT-SESSION RECOVERY

**Authority:** Team Beta, *"GATE-12 ATTEMPT-2 FORENSIC RULING"* (2026-08-10) §§7–16, §23, §25.
**Host:** VM101, `~/distributed_prng_analysis`. **Base HEAD at start: `f216475`** (confirmed first,
not reverted). **HEAD advanced to `990af60` mid-session — by Michael, not by me:** a docs-only commit
(`docs/TB_SUBMISSION_DEFECT_B_CLOSURE_EVIDENCE.md`, +101 lines, one file). `f216475` remains an
ancestor, no source file was touched by it, and this work's two production edits were unaffected. All
`sha256` comparisons below were re-run against the new HEAD and still hold.
**Nothing committed, nothing pushed, nothing launched. No fleet, no real-rig SSH.**
Gate 12 and attempt 3 remain HELD per §24. Port 5700 unbound; `miner_ledger.db` byte-unchanged
(`72d77527…`, mtime still 2026-08-04 18:09:56); no watcher/miner/optimizer process started.

**One deviation from the brief's header, stated up front:** the brief cites HEAD `4c76f42`. Live
HEAD is `f216475` — Defect B landed on top in the interim, as instructed. Every line the brief cites
was re-confirmed against the working tree before editing; all still present, unmoved in substance.
No conflict.

---

## 1. The §10 state machine as implemented

**The defect, re-read from live source before editing** (`miner/range_miner_worker.py`, pre-fix
`serve_forever` at `:1417-1431` of `f216475`) — confirmed verbatim as the brief describes:

```python
while not self._stop.is_set():
    try:    msg = self.conn.recv_msg()
    except (ConnectionError, ValueError, OSError):
        break                      # transport loss AND explicit shutdown exit HERE
    self._dispatch(msg)
finally:
    self.shutdown()
```

`main()` then `return 0` (`:1490-1520`), so a transport blip ended a permitted worker **silently and
successfully**. `_dispatch` set `_stop` on a `shutdown` frame; `_handle_sig` set it on SIGTERM/SIGINT.
All three collapsed into that one `break`. Confirmed.

**One finding the brief did not name, found by reading the live code:** `self._dispatch(msg)` sat
*outside* the inner `try`. So the collapse was into **two** silent exits, not one, and they behaved
differently: an **idle** loss broke out and returned 0, while a loss **while streaming results**
(the `_send` inside `handle_stripe`) propagated out of `serve_forever` entirely as an uncaught
traceback with a non-zero exit. A1's production shape is the idle one; A3/A4's active loss took the
second path. Both are now classified through one seam.

### The implementation

| cause | detection | action | site |
|---|---|---|---|
| **EXPLICIT SHUTDOWN** | `shutdown` frame → `_set_stop_cause("explicit_shutdown")` + `_stop.set()` | exit, **no reconnect** | `:1831` |
| **SIGNAL** | `_handle_sig` → `shutdown(cause="signal")` | exit, **no reconnect** | `:1902` |
| **TRANSPORT LOSS** | transport exception with `_stop` **clear** | close dead session → bounded-backoff reconnect → re-register SAME identity → resume **idle** | `:1731` |

- `serve_forever` (`:1614`) now serves **sessions**, not one socket: run a session, and reconnect only
  if its outcome is recoverable. The certified `finally: self.shutdown()` is preserved.
- `_run_session` (`:1640`) runs one session and classifies its end.
- **`_classify_session_end` (`:1670`) is the load-bearing discriminator:**
  ```python
  cause = (self._stop_cause or STOP_CAUSE_EXPLICIT_SHUTDOWN
           if self._stop.is_set() else SESSION_END_TRANSPORT_LOSS)
  ```
  A transport exception alone authorises nothing. If `_stop` is set, a shutdown frame or a signal
  already decided the daemon is finished and the dying socket is a *consequence* of that decision.
- `_stop_cause` (`_set_stop_cause`, `:1848`) is **first-writer-wins**, so `serve_forever`'s
  `finally: shutdown()` cannot overwrite the real cause with a generic one — which is what keeps
  EXPLICIT SHUTDOWN and SIGNAL distinguishable rather than merely both "not a transport loss".

**The stop is defended twice.** `_recover_session` re-checks `_stop` before its first attempt
(`:1745`). Gate `A6 stop defended twice` asserts the second guard directly. This matters for reading
the mutation evidence: a mutation of the classifier alone cannot make the worker actually reconnect,
so the A6 mutant targets the *classification*, not the attempt count (§5).

**Two narrowing decisions, both deliberate:**

1. **The send-side exception surface is narrower than the receive-side** (`:1229`, `:1237`).
   `_dispatch` was previously outside any guard, so guarding it needs care. On the read path
   `ValueError` means a framing failure and is a genuine transport fault (the certified tuple, kept).
   On the *write* path a `ValueError` is `message_to_bytes` refusing an oversized frame — a
   payload-contract violation, not a dead socket. Recovering from it would reconnect the worker and
   swallow a real defect, so `SEND_TRANSPORT_EXCEPTIONS = (ConnectionError, OSError)` excludes it and
   it keeps propagating exactly as today. Gated in `G-DA-STATE-MACHINE`.
2. **`MinerFramedSocket.close()` now shuts down before closing** (`:1138`), for the same reason the
   coordinator's `_drop_conn` already does (Defect 6 C3): a bare `close()` on a socket with a
   concurrent **blocked** `recv` may defer both the wake-up and the peer's FIN. The worker's control
   loop is normally parked in `recv_msg`, so without this a signal-driven `shutdown()` could leave the
   daemon blocked on an already-closed fd instead of exiting — i.e. the SIGNAL leg of §10 would not
   actually terminate. Found by gate `A6 signal`, which failed before this change.

### §7 precondition confirmed — the coordinator already has the useful half

Confirmed by reading, not assumed, and **not rebuilt**:
- `_drop_conn` (`range_miner_coordinator.py:7384`) evicts the dead socket's identity from
  `fs_by_worker` / `wconn_by_worker` / `self.connections` / `registered`, guarded by
  `fs_by_worker.get(wid) is fs` so a fenced replacement on a different live socket is not evicted.
- `_serve_register` (`:7456`) **accepts a same-identity re-registration once evicted** (no bound
  socket, no live `fs_by_worker` entry → it registers normally, and `_execution_set_admission`
  re-admits it because the frozen set names it). **This is not a finding — the precondition holds.**
- A duplicate that races the eviction is refused with `reject_dup_worker` and the serve loop drops
  that new socket, leaving the original identity intact.

---

## 2. §§11–14 compliance

### §11 — NO STALE-WORK REPLAY (load-bearing)

`_abandon_assignment_no_replay` (`:1699`) resets `state`/`current_stripe_id`/`current_sub_index`/
`progress` and records the abandonment. The worker does **not** re-send the `SubStripeResult` whose
write failed, resume the stripe from private state, declare the old assignment complete, or create a
retry for itself. It returns **idle** and waits for new dispatch. The certified F1/F2 machinery alone
decides the abandoned assignment.

*Proven by:* **A3** (constant phase) and **A4** (hybrid). Both assert `assignment_active_at_loss is
True`, `ASSIGNMENT_ABANDONED{replayed: False}`, `state == "idle"`, and — the direct check — that **no**
`sub_stripe_result` / `stripe_complete` / `stripe_error` frame for the abandoned `stripe_id` ever
arrives after the reconnect. `A3/M` is the red-first control: a replayed completion is detected.

### §12 — ONE ACTIVE CONNECTION PER IDENTITY

`_close_dead_session` (`:1719`) drops the dead socket **without** setting `_stop` (which
`shutdown()` would do) and never opens a second one; one socket at a time, always.

**How the refusal actually reaches the worker, which differs from the brief's wording and is worth
recording:** the coordinator does not send a rejection frame. `_serve_register` returns
`reject_dup_worker` and the serve loop **drops** the duplicate socket. So the worker experiences the
rejection as *the next session's transport loss* and recovers again under the same run-scoped budget —
which is precisely §12's "retryable session-establishment condition, retried after backoff", reached
by a different mechanism than a refusal message. No force-replace, no second live socket.

*Proven by:* **A5**, which induces the real race (see §5 for how) and asserts the duplicate was
rejected, the identity stayed singular (`len(bound) <= 1` throughout), the worker retried
(`reconnect_attempts_total >= 2`, ≥2 loss/reconnect cycles, never `RECONNECT_ABANDONED`), and the
retry after eviction succeeded.

### §13 — FROZEN COHORT REMAINS AUTHORITY

The first `register()` freezes the identity the cohort was admitted under; every reconnect re-sends
**that** frame, and a drifted identity fails closed with `WorkerIdentityChanged` (`:1428`).

`IDENTITY_FIELDS` (`:1295`) is **enumerated**, not "every field of `RegisterMessage`":
`worker_id`, `hostname`, `gpu_id`, `gpu_name`, `backend`, `vram_bytes`, `capabilities`.
**This is a defect I introduced and then caught.** My first version compared
`dataclasses.asdict(msg)` whole. `RegisterMessage` carries a per-message `timestamp`, which
necessarily differs between the first registration and any reconnect — so the wall fired on **every**
recovery and silently re-created the no-reconnect defect wearing a §13 costume. Gate `A7 same
identity` red-first caught it; `changed_fields: ["timestamp"]` is in the run log. `message_type` /
`protocol_version` are framing, not identity, and are likewise excluded.

*Proven by:* **A7 same identity** (frozen frame unchanged across sessions, re-admitted, generation 2);
**A7 changed identity** (device identity drifts → fails closed, never re-enters the cohort,
`IDENTITY_REFUSED{changed_fields: [backend, gpu_name]}`); **A7 non-frozen identity** (a stranger
against a genuinely frozen execution set registers but is **quarantined** and never eligible, while a
set member is — certified exec-set behaviour, gated here because a reconnect must not become a way
in); `A7/M` red-first (forgetting the freeze lets drift through). No cohort expansion anywhere.

### §14 — RECONNECT BOUND

**Positive, finite, run-scoped, and derived — not invented.** `default_recovery_budget_s()` (`:1248`)
reads `DEFAULT_WORKER_ADMISSION_TIMEOUT` (= 180.0) from the coordinator: a worker whose recovery
exceeds the window the coordinator would wait for it cannot help this trial anyway.
**`worker_admission_timeout` itself is untouched** — read, never redefined.

- The import is **function-local by necessity, not by style**: `miner/__init__.py` imports the
  coordinator, and the coordinator imports this module at its line 55, so a module-level import here
  would resolve against a half-initialised coordinator (the constant is defined *below* that import)
  and raise `ImportError`. At call time the package is fully loaded. Verified.
- `recovery_budget_s()` (`:1393`) validates **positive and finite**, exactly as the coordinator
  validates a supplied `worker_admission_timeout`.
- The budget is **cumulative across the worker's life**, not per-episode. A per-episode budget would
  reset on every session and re-create the immortal orphan §14 forbids — a duplicate-rejection
  ping-pong would retry for ever. Only time spent *inside* recovery is charged; productive time is not.
- Backoff (`:1726`) is bounded and derived from the budget: ceiling = budget/12, so at the 180 s
  anchor the ceiling is 15 s and ≥12 attempts fit inside the bound.
- On exhaustion the worker **exits cleanly** (`RECONNECT_EXHAUSTED`, then `shutdown()`).

*Proven by:* **A8 exhaustion** (coordinator gone → exits cleanly inside its bound, `conn is None`, does
not haunt a later run) and **A8 bound derivation** (the 180 s anchor, the derived default, refusal of
`0.0` / `-1.0` / `inf` / `nan`, monotonic bounded backoff, and that the derivation is still *read*
from the coordinator); `A8/M` red-first (an infinite bound must be refused).

---

## 3. §15 observability — records added and their emit sites

Emitted **only on session transitions**. Gate `G-DA-OBS-FIELDS` asserts ~10 heartbeats at a 0.05 s
interval produce **zero** session records, so the no-high-rate-noise bar is checked, not just claimed.
Structured fields carry machine truth (`worker.session_events`, and JSON in the log line); prose
carries diagnostics — the same split as the certified F2 work.

**Worker-side** — `_emit_session_event` (`:1377`), every record carrying `worker_id` and
`session_generation`:

| record | fields | site |
|---|---|---|
| `TRANSPORT_LOSS` | `classification`, `exc_class`, `exc_text`, `assignment_active_at_loss`, `stripe_id_at_loss`, `recovery_spent_s` | `:1737` |
| `ASSIGNMENT_ABANDONED` | `stripe_id`, `sub_index`, `replayed: False`, `authority` | `:1713` |
| `RECONNECT_FAILED` | `attempt`, `attempts_total`, `exc_class`, `reconnect_success: False` | `:1801` |
| `RECONNECTED` | `attempt`, `attempts_total`, `reconnect_success: True`, `resumed_state` | `:1813` |
| `RECONNECT_EXHAUSTED` | `attempts`, `recovery_budget_s`, `recovery_spent_s` | `:1762` |
| `RECONNECT_ABANDONED` | `reason` (the stop cause), `attempt` | `:1750` |
| `RECONNECT_DISABLED` | `reconnect_attempted: False` | `:1774` |
| `IDENTITY_REFUSED` | `changed_fields`, `reason` | `:1443` |
| `SESSION_END` | `classification` (explicit-shutdown / signal / identity-refused), `reconnect_attempted` | `:1626` |

**Coordinator-side** — small additions at existing seams, no new subsystem:

- `WORKER_DISCONNECTED{worker_id, stage_idx, stage_assigned, identity_evicted, obs_status,
  eligible_count_after_drop}` at `_drop_conn` (`:7433`), emitted **after** the eviction so the count is
  the pool the next admission check will see. `stage_idx`/`stage_assigned` come from the serve loop's
  own existing locals (`stage_assigned` already existed at `:6541`), passed from all four call sites
  (`:6662`, `:6678`, `:6691`, `:6720`).
- `WORKER_REGISTERED` / `WORKER_RECONNECTED{worker_id, registration_generation, quarantined,
  obs_status, eligible_count_after_register}` at `_serve_register` (`:7521`), keyed off a run-scoped
  `_registration_generation` counter (`:2543`, cleared per trial at `:6543`) so a first registration is
  never mislabelled a reconnect. It is record-keeping **only** — nothing reads it to decide eligibility.

**Two honesty properties, both gated** (`G-DA-OBS-UNOBSERVED`), carried over from the S4/Defect-B lesson:
- An unmeasurable eligible count is `obs_status: UNOBSERVED` with `null`, **never 0** — a genuinely
  empty pool is a different and real fact, and it still reports `0` when actually observed.
- `identity_evicted: False` distinguishes the fenced-replacement case (this socket died but the
  worker_id is live on another socket, so the pool did **not** shrink) from a real lost worker.

**This is the record attempt 2 did not have.** From the A1 run log, the bracket now reads the pool
descending and names both IDs:

```
WORKER_DISCONNECTED {"worker_id": "rrig6600:gpu0", "stage_idx": 2, "stage_assigned": true,
                     "identity_evicted": true, "obs_status": "OBSERVED",
                     "eligible_count_after_drop": 24}
WORKER_DISCONNECTED {"worker_id": "rrig6600:gpu1", ... "eligible_count_after_drop": 23}
WORKER_RECONNECTED  {"worker_id": "rrig6600:gpu0", "registration_generation": 2,
                     "quarantined": false, "eligible_count_after_register": 24}
```

---

## 4. §23 root-cause forensics — READ-ONLY, and **PARTLY SILENT**

Read-only throughout. The preserved TSV was copied to scratchpad and the copy was read
(`sha256 4f69dba7…` identical); `miner_ledger.db` is byte-unchanged.

### What the evidence establishes

1. **All 25 frozen-cohort identities were present and working in stage 3.** The cohort is named by
   `[S172-CAP] cohort frozen` at 09:25:08 for all four stages, and all 25 appear in the sampler's
   `active_workers_json`.
2. **The two lost workers are NOT nameable from any local artifact.** Between stage 3's start
   (09:47:29) and the terminal (10:44:16) the coordinator log contains **zero** WARNING or ERROR
   records — the teardown was completely silent, and `_drop_conn` logged nothing on the eviction path.
   The eligible set at stage-4 admission was never recorded. The trial's own terminal names
   "23 admitted" and not *which* 23. **This is §15's complaint, confirmed empirically, and it is
   exactly what the amendment fixes.**
3. **The loss window is bounded by measurement.** Sampler occupancy over stage 3:

   | time | fact |
   |---|---|
   | 09:47:29 | stage 3 starts, `active=25 pending=7` (F1 geometry at W=25) |
   | 09:48:51–09:50:54 | 20 workers finish their last stage-3 stripe and go idle |
   | 09:56:40 | `zeus-ubuntu-vm:gpu0` finishes |
   | 10:13:03 | the last 4 (all on `rrig6600`) finish; `compute_active` = 0 thereafter |
   | 10:41:17 | staging drains to 0 (3948 staging jobs, `deferred_high_water=1597`) |
   | 10:44:16 | admission fails at 23/25 after 180.1 s |

   So the earliest-finishing workers held an **idle session for up to ~52 minutes** (09:48:51 →
   10:41:16). The loss must have occurred inside that interval. `compute_active = 0` during the
   10:13→10:41 staging drain is **expected**, not evidence of loss: the certified sampler measures
   `state='claimed'` only and deliberately excludes staging.
4. **The local kernel ring is silent for the run window.** VM101's `dmesg` holds 1192 entries, all from
   the 06:53 boot, and **none** after 07:00 — no OOM, no process kill, no NIC reset recorded locally.
5. **Rig-side kernel evidence via netconsole is empty for the window.** `logs/netconsole_all_rigs.log`
   (a local file, reachable without SSH) ends at `2026-08-10 06:53:34 LISTENER STARTED`, with zero
   entries in 09:23–10:44. **Caveat, stated rather than glossed:** that silence is equally consistent
   with "no rig kernel event" and with "netconsole was not active on the rigs during this run". It does
   not establish the former.
6. **No worker process spawn/exit records exist locally** — the coordinator log captures no per-worker
   stdout, and the workers were one-shot (`use_persistent_workers: False`).

### What the evidence does NOT establish — and is therefore not claimed

- **Which of the four discriminants applies** (transport exception / worker process termination /
  remote OS kill / coordinator close). No local artifact distinguishes them.
- **No initiating cause.** In particular I do **not** claim TCP idle timeout. The ~52-minute idle
  interval is a *measured exposure window*, not a mechanism; asserting causation from it would repeat
  precisely the error Beta corrected in Alpha's earlier report.
- **Rig-side worker logs and process-termination records were not consulted at all** — §24 forbids
  real-rig SSH this session. That surface is **unexamined, not silent**, and is the one place a
  cause might still be recoverable.

**The amendment does not rest on this.** The no-reconnect gap is a defect regardless of what
disconnected those two sockets, and implementation was not blocked on finding an explanation.

---

## 5. A1–A8 — results, with red-first and mutation evidence

New suite: `tests/test_s172_defect_a_transport_recovery.py` — **26/26 green, twice consecutively**
(`/tmp/da7.log`, `/tmp/da8.log`).

**Fixture fidelity (§16, §27).** The fleet arms run a real **25-worker, four-stage** shape on loopback
with stub executors — no GPUs. The 25 identities are **derived** from
`resolve_execution_set(backend="miner", rig_profile="proxmox", admission_count=25)`, which reads the
committed `distributed_config.json`; they are never a hardcoded list, and they are the same 25 the
attempt-2 log names. Identity binding, eviction and duplicate rejection are decided by the **real**
coordinator methods (`_serve_register`, `_drop_conn`) driven with the real serve-loop structures, and
A3/A4 drive the **real** certified retry machinery (`assign_stripes`, `process_lease_expiry`,
`schedule_pending_stripes`). The harness supplies the accept loop and the stub executor, not the
decisions.

| arm | result | evidence |
|---|---|---|
| **A1/RED red-first** | PASS | the **verbatim pre-fix `serve_forever` body** is restored from `f216475` inside the gate; one idle loss ends the daemon silently and successfully, the identity never returns, and the gate detects it |
| **A1 reproduction** | PASS | 25 admitted; stages 1-2-3 complete; 2 **idle** workers dropped at the stage-3→4 boundary (`assignment_active_at_loss: False`); the **same two** identities reconnect (generation 2, `resumed_state: idle`); stage 4 admits **25** and completes; coordinator bracket names both IDs with the pool descending 24 → 23 |
| **A1/M mutation** | PASS | recovery disabled → pool stays 23 |
| **A2 control** | PASS | identical fixture, reconnection disabled → 23/25, admission **refused**, **stage 4 never assigned**, dropped workers exit cleanly (`RECONNECT_DISABLED`, no `RECONNECTED`); expectation never renegotiated downward; `DEFAULT_WORKER_ADMISSION_TIMEOUT == 180.0` re-asserted |
| **A2/M mutation** | PASS | 23 must not satisfy a 25-worker expectation |
| **A3 constant phase** | PASS | active constant-phase loss → certified policy still **fails the trial** (`aborted`), `current_attempt == 0` (no retry consumed), worker returns **idle**, `ASSIGNMENT_ABANDONED{replayed: False}`, **no** frame for the abandoned stripe after reconnect, terminal state not reopened |
| **A3/M mutation** | PASS | a replayed completion for an abandoned stripe is detected |
| **A4 hybrid retry** | PASS | active hybrid loss → `phase_degraded == 1`, `current_attempt == 1` (**exactly once**), trial stays `running`, the **alternate** holds the retry (`claimed_by == alt`), the failed worker cannot self-reclaim, reconnect restores **future eligibility only** (a *new* stripe completes on it) |
| **A4/M mutation** | PASS | two mutants: "retry consumed twice" and "failed worker self-reclaims" both red |
| **A5 duplicate race** | PASS | duplicate rejected (`reject_dup_worker`), identity singular throughout, worker retries, retry after eviction succeeds |
| **A5/M mutation** | PASS | a second live socket for one identity must be refused |
| **A6 explicit shutdown** | PASS | `shutdown` frame → exits, `returned is True`, **zero** reconnect attempts, no `TRANSPORT_LOSS`, `SESSION_END{classification: explicit_shutdown, reconnect_attempted: False}` |
| **A6 signal** | PASS | signal → exits, **zero** reconnect attempts, cause recorded as `signal` |
| **A6 stop defended twice** | PASS | `_recover_session` refuses to attempt anything once `_stop` is set, even when handed a recoverable outcome |
| **A6/M mutation** | PASS | a `_stop`-blind classifier misreads a signal as a transport loss |
| **A7 same / changed / non-frozen identity** | PASS ×3 | §13, detailed in §2 above |
| **A7/M mutation** | PASS | forgetting the frozen frame lets drift through |
| **A8 exhaustion / bound derivation** | PASS ×2 | §14, detailed in §2 above |
| **A8/M mutation** | PASS | an infinite bound must be refused |
| **G-DA-STATE-MACHINE** | PASS | three causes, three outcomes, one discriminator; first-writer-wins; send-surface narrower than read-surface |
| **G-DA-OBS-FIELDS / -UNOBSERVED / -NO-TOUCH** | PASS ×3 | §15 fields with zero heartbeat noise; UNOBSERVED ≠ 0; admission/cohort/lease authorities read, not redefined |

**A2 is the clean control for A1** (same fixture, only recovery disabled) and **A6 is the clean control
for the reconnect branch** (the socket dies there too, but `_stop` is set, so recovery must not fire).

### Two harness-fidelity decisions worth Beta's eye

1. **A3/A4's active loss is injected at the worker's own socket** (`SHUT_WR`, helper
   `_fail_worker_writes`). A worker computing a stripe is neither reading nor writing, so it cannot
   notice a peer's disappearance until its next I/O — and a peer **FIN does not reliably fail that
   next `send`**: the bytes land in the local kernel buffer, the stripe finishes, and the loss only
   surfaces later at a read, by which time it looks **idle**. An RST is not reliable either while the
   peer's reader is still blocked on the socket. Both were observed during development. The peer-side
   eviction still runs through the real `_drop_conn`; the injection only fixes *when* the worker finds
   out, and the observable the code reacts to — a transport exception raised while `state == "mining"`
   — is identical.
2. **A5 suppresses the harness reader-thread's eviction** to open the race window. This models
   production faithfully rather than weakening it: in the real serve loop the reader's `eof` goes onto
   the inbound **queue** and `_drop_conn` runs later in the dispatch loop, so a reconnect can genuinely
   beat the eviction. A1 exercises the eviction path in full.

### Suite tallies

| suite | result | note |
|---|---|---|
| `test_s172_defect_a_transport_recovery.py` | **26/26** | new; green on two consecutive runs |
| `test_s172_f1_f2_active_lease.py` | **16/16** | certified baseline, unchanged |
| `test_s172_phase4_coordinator.py` | **62/63** | see below — expected, self-clearing |
| `test_s172_phase3_worker.py` | **17/17** | drives `serve_forever` directly |
| `test_s172_admission_liveness.py` | **16/16** | admission authority |
| `test_s172_resolved_execution_set.py` | **34/34** | frozen-cohort authority |
| `test_s172_staging_backpressure.py` | **50/50** | `_drop_conn` borders the capacity grace |

Suites were run **sequentially** (concurrent runs flake Part B G-VAL-6 on a free-space race).

**The phase-4 62/63 is the Gate 22 untracked-`.py` sensitivity — fourth occurrence, same answer.**
Gate 22 compares `git status --porcelain` `.py` entries against a hardcoded allowlist. Both files I
edited (`miner/range_miner_coordinator.py`, `miner/range_miner_worker.py`) are **already** in that
allowlist; the only entry tripping it is the new, still-untracked
`tests/test_s172_defect_a_transport_recovery.py`. **Proven self-clearing rather than asserted:** with
that one file temporarily moved out of the tree, phase-4 is **63/63 with Gate 22 PASS**
(`/tmp/gate22_demo.log`), and the file was restored immediately. **The allowlist was NOT widened** —
per the standing rule, it self-clears on the commit that tracks the file.

---

## 6. Byte-unchanged confirmation and files changed

**Files changed — from `git status`, not recall:**

```
 M miner/range_miner_coordinator.py                  (+90 / 12 hunks)
 M miner/range_miner_worker.py                       (+431)
?? tests/test_s172_defect_a_transport_recovery.py    (new suite)
?? docs/CLAUDE_CODE_REPORT_DEFECT_A_TRANSPORT_RECOVERY.md   (this report)
?? miner_ledger.db-shm, miner_ledger.db-wal, optimal_window_config.json.stale_1786149572
```

The three untracked non-`.py` entries pre-date this session (present at `f216475`).

**Byte-unchanged (`sha256`, live vs `HEAD`) for every no-touch file bordering the change:**

| file | |
|---|---|
| `scripts/gate12_concurrency_sampler.py` | BYTE-UNCHANGED `700023a4fd117325` |
| `tests/test_gate12_concurrency_sampler.py` | BYTE-UNCHANGED `99d191c4b08e1bb9` |
| `execution_set.py` | BYTE-UNCHANGED `d21614701c31a7b4` |
| `miner/range_miner_protocol.py` | BYTE-UNCHANGED `7b7f8e1979141789` |
| `miner/__init__.py` | BYTE-UNCHANGED `699e930e379d07a7` |
| `miner/dataset_authority.py` | BYTE-UNCHANGED `365c8e3ee9abf80a` |
| `persistent_worker_coordinator.py` | BYTE-UNCHANGED `fc89a3739e88480a` |
| `window_optimizer.py` | BYTE-UNCHANGED `f849816953c07182` |
| `watcher_agent.py` | BYTE-UNCHANGED `fc874f21d702a7d3` |
| `distributed_config.json` | BYTE-UNCHANGED `ac4ba07ca3f35d90` |
| `rig_profiles_config.json` | BYTE-UNCHANGED `c69a26443cf74ec6` |
| `gate12_launch.sh` | BYTE-UNCHANGED `6fcd1468fa9e8ca0` |
| `miner_ledger.db` | BYTE-UNCHANGED `72d77527…`, mtime still 2026-08-04 |

**The sampler was not touched** (Defect B is committed and separate).

For the one no-touch-adjacent file that *is* edited — `range_miner_coordinator.py` — the 12 hunks are
confined to: `__init__` (the generation counter), one serve_trial local reset, the four
`_drop_conn`/`_serve_register` call sites, the `_drop_conn` body, and the `_serve_register` signature
and tail. **No staging, backpressure, lease, admission-policy, seed-domain, coverage, dataset or
publication code is inside any hunk.** `staging_backpressure` 50/50, `admission_liveness` 16/16 and
F1/F2 16/16 corroborate.

---

## 7. NO-TOUCH conflicts encountered

**None.** No item on Beta's §25 list had to change for any gate to pass:

`expected_workers` semantics · frozen cohort membership · the *live*-eligibility requirement ·
constant-phase failure policy · hybrid retry budget · lease semantics ·
`worker_admission_timeout = 180 s` · staging/backpressure · seed domain · coverage authority ·
dataset authority · publication authority — all **read**, none redefined. The 180 s constant is read
as the §14 anchor and is still asserted at `180.0`. The live-eligibility expression
(`[w for w in wconn_by_worker.values() if not w.quarantined]`) is asserted unchanged in
`G-DA-NO-TOUCH`.

**Three things flagged for Beta's judgement — none of them a no-touch edit, all of them decisions I
made and could reverse:**

1. **`MinerFramedSocket.close()` now shuts down before closing** (§1). Needed for the SIGNAL leg of
   §10 to actually terminate; mirrors the coordinator's own Defect-6 C3 reasoning. It touches a
   certified class, so it is named explicitly rather than buried. Phase-3 17/17 and phase-4 63/63
   cover it.
2. **The send-side exception surface is narrower than the read-side** (§1). A judgement call: an
   oversized-frame `ValueError` is a contract violation, not a transport loss, and must not be
   swallowed into a reconnect.
3. **The recovery budget is cumulative over the worker's life, not per recovery episode** (§14). A
   per-episode budget is the reading closer to the brief's literal wording, but it re-creates the
   immortal orphan §14 forbids under a duplicate-rejection ping-pong. If Beta prefers per-episode
   semantics with a separate episode cap, that is a one-line change to `_recover_session`.

---

## 8. Not done

- **Nothing launched, nothing committed, nothing pushed.** Gate 12 and attempt 3 remain HELD (§24).
- **No real-rig SSH**, so the rig-side worker-log and process-termination surfaces of the §23 pass are
  **unexamined** — reported as unexamined, not as silent.
- Session changelog (CLAUDE.md §1.8) not written — Michael commits; say the word and I will draft it.
