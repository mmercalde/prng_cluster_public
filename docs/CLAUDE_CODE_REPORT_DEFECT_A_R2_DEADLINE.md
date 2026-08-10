# CLAUDE CODE REPORT — DEFECT A R2: §14 RECOVERY-DEADLINE ENFORCEMENT

**Host:** VM101, `~/distributed_prng_analysis`. **Base HEAD `acd6f13`** (verified
`git log --oneline -1`, unchanged, not reverted). **Nothing committed, nothing pushed, nothing
launched. Gate 12 and attempt 3 remain HELD.**

**Authority:** Team Beta *"Defect A Transport-Session Recovery Certification Ruling"* (2026-08-10),
RETURN FOR NARROW REVISION, §14 only, per
`docs/CLAUDE_CODE_INSTRUCTIONS_DEFECT_A_R2_DEADLINE.md`.

**Scope actually delivered:** the three §15 items and the three §16 gates, nothing else. §§10–13
and §15 are untouched (proved function-by-function in §3 below). The cumulative budget, the
backoff shape, the same-identity rule, the state machine, `DEFAULT_WORKER_ADMISSION_TIMEOUT` and
all coordinator behaviour are exactly as certified. **No no-touch conflict arose.** One
expectation in the brief did not hold and is returned rather than worked around — §6.

**Files changed (`git status --porcelain`):**

```
 M miner/range_miner_worker.py                      (+65 / -3)
 M tests/test_s172_defect_a_transport_recovery.py   (+411 / -0)
```

`miner/range_miner_coordinator.py` is **byte-unchanged** — this revision has no coordinator diff at
all. sha256 of the two changed files at the state everything below was measured on:

```
21c8cb35538a483e3502a3822db09f3901a3699d8aa285ce5489a479fef2672b  miner/range_miner_worker.py
18fd199814b69ba9bb49d24cd5e88049ee165548b6e2f263a532449128a93bac  tests/test_s172_defect_a_transport_recovery.py
```

---

## 1. THE THREE FIXES, AT THEIR SOURCE LINES

The invariant being enforced: *no new recovery operation may begin when cumulative remaining ≤ 0,
and no blocking recovery operation may block past the remaining recovery deadline.*

### A. Post-backoff exhaustion re-check — `range_miner_worker.py:1805-1819`

Immediately after the backoff wait is charged and **before** `connect()`:

```python
remaining = budget - self._recovery_spent_s
if remaining <= 0.0:
    self._emit_session_event(
        "RECONNECT_EXHAUSTED", attempts=attempt,
        recovery_budget_s=budget,
        recovery_spent_s=round(self._recovery_spent_s, 3),
        reconnect_success=False)
    return False
```

The existing `RECONNECT_EXHAUSTED` event, the existing field set, the same clean `return False`.
`attempts` reports attempts actually made (the counter has not been incremented yet at this point),
consistent with the pre-wait check the loop already had. This closes escape #1: with
`remaining = 0.2 s`, the wait consumes the remainder and the loop no longer buys another
`connect()` with an allowance of zero.

### B. Bounded recovery connection establishment — `range_miner_worker.py:1406-1432` and `:1827`

`connect()` gains **one optional parameter**:

```python
def connect(self, timeout: Optional[float] = None) -> None:
    if timeout is None:
        sock = socket.create_connection((self.host, self.port))
    else:
        sock = socket.create_connection((self.host, self.port), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    self.conn = MinerFramedSocket(sock)
```

- **TCP_NODELAY retained**, socket construction otherwise identical.
- **The non-recovery path is unchanged.** Every existing caller — production `main()`
  (`:1981`), `tests/test_s172_phase3_worker.py:319`, `tests/test_s172_phase4_coordinator.py:1534`,
  the Defect-A harness `_spin` (`:511`) — calls `connect()` with no argument and takes the
  `timeout is None` branch, which is the pre-fix statement verbatim. The branch is explicit rather
  than passing `timeout=None` through, so the startup path cannot be perturbed by a future global
  default socket timeout.
- The recovery caller is the only one that passes a deadline: `self.connect(timeout=remaining)`
  at `:1827`, where `remaining` is the value the §15-A re-check just computed.
- A `socket.timeout` from a bounded connect is an **ordinary failed recovery attempt**: on this
  host `socket.timeout is TimeoutError`, a subclass of `OSError`, so it is already inside the
  certified `TRANSPORT_EXCEPTIONS = (ConnectionError, ValueError, OSError)` (verified live) and
  lands on the existing `RECONNECT_FAILED` branch — charged, backed off, retried while budget
  remains. **No exception tuple was widened.**

### C. Bounded recovery registration establishment — `range_miner_worker.py:1828-1838`

```python
register_allowance = remaining - (time.monotonic() - t0)
if register_allowance <= 0.0:
    raise socket.timeout(
        "recovery deadline reached after connect, before REGISTER")
assert self.conn is not None
self.conn.sock.settimeout(register_allowance)
self.register()
```

Connection **and** registration together stay inside the one `remaining` allowance: the REGISTER
send receives the residual the connect did not use, not a fresh interval. If the connect consumed
the whole allowance, the REGISTER is not attempted at all and the raise is handled by the same
`TRANSPORT_EXCEPTIONS` branch, so the attempt is charged and the loop's exhaustion check ends it.

**Charging is unchanged and cumulative.** `self._recovery_spent_s += time.monotonic() - t0` on all
three exits (identity refusal, transport failure, success) is the certified accounting, untouched —
so the deadline is a true wall-clock bound **across** attempts, not per-episode.

### The restore-to-blocking guarantee — `range_miner_worker.py:1839-1845`

```python
self.conn.sock.settimeout(None)
```

executed after `register()` succeeds and **before** the restored session is served. This is
load-bearing, not hygiene: `MinerFramedSocket.recv_msg` would raise `socket.timeout` on any quiet
interval of a timed-out socket, and `_classify_session_end` correctly classifies that as a
TRANSPORT LOSS — so a lingering timeout would put an *idle* worker into a reconnect loop and
manufacture the very failure Defect A exists to prevent. The deadline binds **recovery
establishment only**; the certified session loop is timeout-free before and after this amendment
(§4).

### Interactions considered and deliberately not changed

- **Heartbeat during the register window.** `_heartbeat_loop` and `register()` both go through
  `_send`, which serializes on `self._send_guard`, so a heartbeat cannot interleave inside the
  REGISTER frame. A heartbeat that does land in the timed window is *bounded* rather than
  unbounded — strictly better than pre-fix — and a heartbeat exception has always broken that
  thread, which `_run_session` restarts on the next session. No change made.
- **Failure paths leave no socket behind.** Every exception branch already calls
  `_close_dead_session()`; a socket carrying a recovery timeout is therefore never retained.

---

## 2. THE THREE NEW GATES — A8-B1 / A8-B2 / A8-B3

Added to `tests/test_s172_defect_a_transport_recovery.py`. **Suite: 29/29 (26 existing + 3),
green twice consecutively** (`/tmp/defect_a_r2_final1.log`, `/tmp/defect_a_r2_final2.log`).

### A8-B1 — budget consumed by backoff (`:1703`)

`_recovery_spent_s` is pre-set to `budget - 0.2`, so the next backoff (1.0 s, clamped to the
remainder) consumes exactly the rest. `connect` is replaced by a **counting spy that raises if
called at all**, so *"connect was never called"* is a hard assertion, not an inference. Asserts:
`_recover_session` returns False · **connect call count == 0** · `reconnect_attempts_total == 0` ·
last event is `RECONNECT_EXHAUSTED` with `attempts == 0`, `reconnect_success False`,
`recovery_spent_s >= budget` · `w.conn is None`.

**Mutation (embedded, `_mutant_red`):** the **verbatim pre-fix `_recover_session` body**
(`acd6f13:1731-1821`, reproduced at `tests/…:1592`) restored on the instance. Its only exhaustion
check runs before the wait, so it proceeds into a connect with nothing left — the arm detects it.
**Mutant reds.**

### A8-B2 — blocking-connect mutant, the load-bearing arm (`:1769`)

The gap Beta named: the original A8 kills a **localhost** coordinator, and localhost refuses
*instantly*, so the arm shared the implementation's own assumption that connects return promptly.
A8-B2 removes that assumption with `_BlackHoleSocketModule` (`:1672`) — a stand-in for the worker
module's global `socket` that proxies everything to the real module except `create_connection`,
which **blocks for 30 s when given no deadline** (a black-holed route) and blocks for its deadline
then raises `socket.timeout` when given one (what the OS does). **Production `connect()` runs
unmodified** underneath it.

Budget 2.0 s, block 30 s, join window 6.0 s. Asserts: the recovery thread **finished** (did not
hang in the OS connect) · returned False · a deadline was supplied on **every** recovery connect ·
each deadline is **finite, positive and ≤ the remaining budget at that call** · the deadline
reached `socket.create_connection` itself (`shim.timeouts` equals the spy's record, so the bound is
not merely decorating a wrapper) · last event `RECONNECT_EXHAUSTED` · `_recovery_spent_s` and the
measured wall clock both stay inside the budget.

**Mutation (embedded):** the **verbatim pre-fix `connect()` body** (`acd6f13:1406-1409`,
reproduced at `tests/…:1569`) bound to the instance. It discards the deadline, the shim blocks
30 s against a 2 s budget, and the arm reds on
*"the worker blocked past its recovery deadline inside the OS connect"*. **Mutant reds — this is
the arm that would have caught the shipped defect.**

> **Self-caught defect, reported because it is the same class Beta keeps finding.** The first
> version of that mutant **SURVIVED** (run 1, `/tmp/defect_a_r2_run1.log`, 28/29). The verbatim
> copy was a plain test-module function, so its `socket` resolved in the *test* module's globals,
> escaped the shim entirely, connected nowhere and returned instantly. A red-first control that
> silently tests the wrong module is exactly the vacuous arm `_mutant_red` exists to refuse. Fixed
> by rebinding the copy's global namespace to `WORKER.__dict__` via `types.FunctionType`
> (`tests/…:1587`), so the body resolves `socket` and `MinerFramedSocket` **where the real pre-fix
> body resolved them**. The comment at that site records the trap.

### A8-B3 — success before deadline, the control (`:1858`)

A real `_HarnessCoordinator` session on loopback: register, `drop_transport`, reconnect. Asserts
the fix does **not** turn healthy recovery into a false failure: `RECONNECTED` with
`reconnect_success True` · the startup connect deadline is `None` (unbounded, unchanged) while every
recovery connect carries a finite deadline inside the budget · `_recovery_spent_s < budget` · the
worker thread is still alive · **the restored socket's `gettimeout()` is `None`** · and the restored
session is genuinely serving — a `MinerStatusMessage` query round-trips and comes back
`state == "idle"`, which a half-restored session could not do.

### Red-first differential against the pre-fix source

`git worktree add … acd6f13`, the **new suite file copied in unchanged**, same venv, run, worktree
removed (`git worktree list` now shows only the main tree). Log:
`/tmp/defect_a_r2_prefix_redfirst.log`.

```
26/29 — the 26 certified arms GREEN, all three new arms RED:
  A8-B1  FAIL  connect must not begin with the budget spent
  A8-B2  FAIL  the worker blocked past its recovery deadline inside the OS connect
  A8-B3  FAIL  [None, None]   (the recovery connect carried no deadline)
```

Every red is **semantic, not a signature artifact**: the spies call through exactly as the caller
did (`real_connect(timeout) if timeout is not None else real_connect()`), which the pre-fix
one-argument signature also accepts. The differential is exactly the three new arms — no certified
arm moved.

*Note on A8-B3's red:* against pre-fix it **did reconnect successfully** and reds only on the
missing deadline. That is the correct reading of its control role — normal recovery works before
and after; B3's job is to prove the deadline fix did not break it, and it additionally asserts the
deadline and the restore.

---

## 3. §§10–13 AND §15 ARE UNCHANGED — FUNCTION-LEVEL PROOF

Not a claim from recall. `ast` was used to extract every top-level and method source segment from
`git show acd6f13:miner/range_miner_worker.py` and from the working tree, and to sha256 each:

```
top-level+method definitions: old=90  new=90
CHANGED (3):
  RangeMinerWorker            93f7c6f83e03 -> bb2e8b5f127b   (contains the two below)
  RangeMinerWorker.connect            ac3878632e5c -> b08651e95af3
  RangeMinerWorker._recover_session   c86159a07ca5 -> 8bcae8121f96
UNCHANGED: 87
module-level statements (imports/constants): 3c91b9390955 -> 3c91b9390955  IDENTICAL
```

**Only `connect()` and `_recover_session()` differ**, exactly as Beta required. The class hash moves
only because it contains them. Byte-identical, among the 87: `serve_forever`, `_run_session`,
`_classify_session_end`, `_set_stop_cause`, `shutdown` (§10) · `_abandon_assignment_no_replay`
(§11) · `_close_dead_session` (§12) · `register`, `identity_projection`, `_build_register_message`
(§13) · `recovery_budget_s`, `default_recovery_budget_s`, `_backoff_delay` (§14 authorities —
enforced here, not redefined) · `_emit_session_event` (§15).

The identical module-level hash covers `DEFAULT_WORKER_ADMISSION_TIMEOUT`'s lazy derivation import,
`TRANSPORT_EXCEPTIONS`, `SEND_TRANSPORT_EXCEPTIONS`, `RECONNECT_BACKOFF_*` and `IDENTITY_FIELDS` —
**no constant was added, moved or changed**, and no new import was needed (`socket` and
`Optional` were already imported). `DEFAULT_WORKER_ADMISSION_TIMEOUT` remains read, never
redefined; `G-DA-NO-TOUCH` re-ran green.

`miner/range_miner_coordinator.py` does not appear in `git status` — the coordinator is untouched.

---

## 4. THE RESTORED SESSION CARRIES NO SOCKET TIMEOUT

Three independent statements of the same fact:

1. **Source** — `settimeout(None)` at `:1845` runs after `register()` succeeds and before
   `_recover_session` returns True, i.e. before `serve_forever` enters the next `_run_session`.
2. **Measured behaviour** — A8-B3 asserts `w.conn.sock.gettimeout() is None` on the live restored
   socket, after waiting for the `RECONNECTED` event (which is emitted *after* the restore, so the
   read cannot race it). It also asserts the same for the first startup session, so the arm would
   catch a timeout arriving on either path.
3. **Functional proof** — the restored session serves a status round-trip in A8-B3, and A1 (the
   25-worker × 4-stage reproduction, unchanged) still completes stage 4 across reconnected
   sessions. A socket left with a 6 s test budget would have raised `socket.timeout` inside the
   read loop, been classified as a transport loss and produced a reconnect loop; nothing of the
   kind appears in any arm.

The certified loop therefore behaves exactly as before: bounded blocking exists only between the
recovery `connect()` and the end of the recovery `register()`.

---

## 5. VERIFICATION

| suite | result | log |
|---|---|---|
| Defect A (26 existing + A8-B1/B2/B3) | **29/29**, green twice consecutively | `/tmp/defect_a_r2_final1.log`, `/tmp/defect_a_r2_final2.log` |
| Defect A against **pre-fix source** (worktree `acd6f13`) | **26/29** — the three new arms red, all 26 certified arms green | `/tmp/defect_a_r2_prefix_redfirst.log` |
| F1/F2 active-lease | **16/16** | `/tmp/r2_f1f2.log` |
| admission-liveness | **16/16** | `/tmp/r2_adm.log` |
| resolved execution set | **34/34** | `/tmp/r2_exec.log` |
| phase-4 coordinator | **62/63** — sole failure Gate 22, see §6 | `/tmp/r2_phase4.log` |
| phase-3 worker (not in the brief's battery; run because this file is its subject) | **17/17** | `/tmp/r2_phase3.log` |

All runs `source ~/venvs/torch/bin/activate` first, `python3 -u … | tee`, never `tail`, and
sequentially (never concurrently — concurrent S172 runs flake on a free-space race).

**Verification-integrity controls (VIR-1…6)**
- *execution proof:* every suite printed its own tally to a `tee`d log; the pass counts above are
  read from those logs, not asserted.
- *clean control:* A8-B3 (healthy recovery still succeeds); the 26 certified arms green on both the
  patched tree and the pre-fix worktree.
- *fault-injection control:* the embedded verbatim pre-fix `connect()` and `_recover_session()`
  mutants, plus the whole-tree pre-fix worktree differential. One mutant **survived first pass and
  was fixed** (§2) — recorded, not silently repaired.
- *completion sentinel:* each suite's final `N/N checks green` line.
- *unavailable-observer behavior:* none reached in this revision; no fleet, GPU or rig surface was
  touched.
- *audit claim scope:* the worker module's function-level diff versus `acd6f13`, and the four
  battery suites. No claim is made about anything else.
- *searched surfaces:* `docs/CLAUDE_CODE_INSTRUCTIONS_DEFECT_A_R2_DEADLINE.md` (in full),
  `miner/range_miner_worker.py`, `miner/range_miner_coordinator.py` (read for `TRANSPORT_EXCEPTIONS`
  and the no-touch check), `miner/range_miner_protocol.py`,
  `tests/test_s172_defect_a_transport_recovery.py`, `tests/test_s172_phase4_coordinator.py`
  (Gate 22 allowlist), `git show acd6f13:` for both verbatim pre-fix bodies, live Python for
  `socket.timeout` ancestry and `inspect.signature`.
- *unavailable surfaces:* none required. No rig, no coordinator process, no ledger was consulted.
- *governance trail searched:* the R2 brief itself (the binding ruling for this revision) and the
  `acd6f13` commit message; the project-facts skill §§2.19, 2.27–2.30 for the no-touch and
  correlated-blind-spot context. **This revision reopens no prior ruling.**
- *chapters searched:* none — no chapter covers worker transport recovery, and no chapter claim is
  made.

---

## 6. RETURNED, NOT WORKED AROUND — the brief's phase-4 expectation

The brief states: *"phase-4 63/63 (the Gate-22 self-clear is already resolved now that the test
file is committed — confirm 63/63 directly)."* **That expectation does not hold, and the difference
is real rather than cosmetic.**

Gate 22 builds `changed_py` from `git status --porcelain`, which reports **modified tracked files as
well as untracked ones**. `tests/test_s172_defect_a_transport_recovery.py` is committed at
`acd6f13`, so at that commit it appeared in `git status` **not at all** — which is why the gate read
63/63 there, and why the file was never added to the allowlist. This revision *edits* it, so it
reappears as a changed `.py` that the allowlist does not name:

```
Gate 22: coexistence — unexpected changed .py files:
    {'tests/test_s172_defect_a_transport_recovery.py'}
```

It is the **only** failure in the suite (one `FAIL` line in 63). `miner/range_miner_worker.py` **is**
allowlisted (`tests/test_s172_phase4_coordinator.py`, allowlist line 2), so the production diff is
in scope and is not what trips it.

**The allowlist was NOT widened** — standing rule, and this is the 5th occurrence of the Gate-22
sensitivity (the first four were the untracked-`.py` variant; this is the modified-tracked variant
of the same mechanism). It self-clears the moment Michael commits the two changed files. **Beta's
call, flagged not decided:** whether the Defect-A harness should be added to the Gate-22 allowlist
permanently, as every other deliverable's edited test file has been. I have not done so, because
adding it would change the gate's scope on my own authority and the condition disappears at commit
regardless.

---

## 7. JUDGEMENT CALLS (all reversible, flagged for Beta)

1. **`register_allowance` is the residual of `remaining`, not a second full `remaining`.** Beta §15-C
   says connection + registration together must stay inside `remaining`; charging the REGISTER only
   what the connect left over is the strict reading. The looser reading (a fresh `remaining` for the
   REGISTER) would permit up to 2× the deadline. Chose the strict one — the structurally stronger
   mechanism, per the standing owner rule.
2. **Explicit `if timeout is None` branch in `connect()`** rather than forwarding `timeout=None` to
   `create_connection`. Behaviourally identical today; the branch keeps the startup path immune to a
   process-wide `socket.setdefaulttimeout`, and keeps the pre-fix statement literally intact.
3. **The exhausted-before-REGISTER case raises `socket.timeout`** so it lands on the existing
   `RECONNECT_FAILED` branch (charged, then ended by the loop's own exhaustion check) rather than
   returning early with a second `RECONNECT_EXHAUSTED` emit site. One exhaustion emit site per
   loop pass; no new event kind, no new field.

---

## 8. STATE ON EXIT

- **HEAD is `acd6f13`**, not reverted, nothing committed, nothing pushed, nothing launched.
- Working tree carries exactly the two modified files above, plus two untracked docs (the R2 brief
  and this report) and the pre-existing untracked entries from the session start (two sqlite
  WAL/SHM sidecars, one `.stale_*` file). Gate 22 inspects `.py` only, so the two docs do not
  affect §6.
- The temporary `acd6f13` worktree used for the red-first differential has been removed;
  `git worktree list` shows only the main tree.
- **Gate 12 and attempt 3 remain HELD** pending Beta's certification of this revision.
