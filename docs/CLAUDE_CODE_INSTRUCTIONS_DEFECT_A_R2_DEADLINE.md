# CLAUDE CODE INSTRUCTIONS — DEFECT A REVISION: §14 RECOVERY-DEADLINE ENFORCEMENT

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD **`acd6f13`** (the certified-except-§14
Defect-A commit). `source ~/venvs/torch/bin/activate` before every test.

**Authority:** Team Beta *"Defect A Transport-Session Recovery Certification Ruling"* (2026-08-10) —
**RETURN FOR NARROW REVISION, §14 ONLY.** Beta accepted §§10, 11, 12, 13, 15, ratified all three
judgement calls and the self-caught identity fix. **This is a three-item deadline-enforcement fix. Do
NOT redesign reconnect and do NOT reopen §§10–13 or §15.**

**Hard constraints — no commit, no push, no launch.** Gate 12 and attempt 3 HELD. **§25 no-touch,
plus Beta's explicit "do not change" list for THIS revision:** `expected_workers` · live eligibility ·
frozen cohort · F1/F2 · lease semantics · `worker_admission_timeout` · **the recovery-budget anchor**
(`DEFAULT_WORKER_ADMISSION_TIMEOUT`, still read never redefined) · staging/backpressure · seed domain ·
dataset/publication/coverage authority. Keep the cumulative budget, the backoff, the same-identity
rule, the state machine, and all coordinator behaviour **exactly as certified.** If the fix appears to
require changing any of these, STOP and return the conflict.

---

## THE DEFECT (Beta §13), CONFIRMED IN SOURCE

`_recover_session()` checks `remaining <= 0` **before** the backoff wait, waits, charges the spend,
then calls `connect()` — but:

1. **No re-check after the wait, before `connect()`.** With `remaining = 0.2s`: wait 0.2s → budget now
   exhausted → the code still proceeds into another `connect()`. A boundary violation on its own.
2. **`connect()` is unbounded** (`range_miner_worker.py:1407`):
   `socket.create_connection((self.host, self.port))` with **no `timeout`**. A clean refusal returns
   fast (which is why A8 passes), but a **black-holed route blocks in the OS connect** — and while
   blocked the worker cannot inspect `_recovery_spent_s`, `remaining`, or `_stop`. So
   `recovery_budget_s = 180` does **not** guarantee recovery terminates near 180 s. The budget is
   bookkeeping, not an enforced deadline. **This violates §14.**
3. **`register()`'s send is likewise unbounded** — if `connect()` succeeds near the deadline, the
   REGISTER send must not gain a fresh unlimited blocking interval (Beta §15-C).

**Why A8 missed it (Beta §14 — the correlated blind spot, 4th in this arc):** A8 connects to a killed
**localhost** coordinator, which refuses *immediately*. The test shares the implementation's assumption
that connects return promptly, so it can never exercise a blocking-connect overrun. The new A8-B2
mutant fixes exactly that gap.

## THE FIX — Beta §15, three items, nothing more

**Required invariant:** *no new recovery operation may begin when cumulative remaining ≤ 0, and no
blocking recovery operation may block past the remaining recovery deadline.*

### A. Post-backoff exhaustion re-check

Immediately after charging the backoff spend and before `connect()`, recompute
`remaining = budget - self._recovery_spent_s`. If `remaining <= 0`: emit `RECONNECT_EXHAUSTED` (the
existing event) and `return False` **without another `connect()`**. This closes escape #1.

### B. Bound the recovery connection establishment

`connect()` during recovery must receive the **current remaining budget** as its connect timeout.
Concretely: `connect()` takes an optional `timeout: Optional[float]` and passes it to
`socket.create_connection((host, port), timeout=timeout)` when set; on `socket.timeout`/`OSError` the
attempt is a normal failed recovery attempt (charged, backed off, retried while budget remains). The
timeout applies to **recovery establishment only** — after a successful reconnect, restore ordinary
blocking behaviour (`create_connection` returns a socket already in blocking mode; if you set a timeout
on the socket object, clear it back to `None`/blocking before handing it to `MinerFramedSocket`, so the
restored session behaves byte-for-byte as a normal session — the certified session loop must not gain a
socket timeout). The normal (non-recovery) `connect()` path — first startup — keeps its current
behaviour; only the recovery caller passes a deadline.

### C. Bound the recovery registration establishment

If `connect()` succeeds close to the deadline, the REGISTER send must stay inside the remaining
allowance too. Connection + registration together must remain inside `remaining`. After successful
establishment, restore ordinary session behaviour (no lingering socket timeout on the live session).

**Charge all of it to `_recovery_spent_s`** — the cumulative accounting Beta ratified — so the
deadline is a true wall-clock bound across attempts, not per-episode.

## READ BEFORE WRITING

- `connect()` at `:1406-1409` — the unbounded `create_connection`; add the optional timeout, keep TCP
  NODELAY, keep the restore-to-blocking guarantee.
- `_recover_session()` — the loop that computes `remaining`, waits, charges `_recovery_spent_s`, then
  `connect()`/`register()`. The re-check goes between the wait-charge and `connect()`; the remaining
  budget is what you hand `connect(timeout=...)`.
- `recovery_budget_s()` / `_recovery_spent_s` — unchanged; you are enforcing them, not redefining them.

## NEW GATES — Beta §16, exactly these three, added to the existing suite

Keep all 26 existing arms green. Add:

- **A8-B1 — budget consumed by backoff.** `remaining < next backoff`; after the wait consumes the
  remainder, assert **connect call count == 0**, `RECONNECT_EXHAUSTED` emitted, worker exits. Catches
  the missing post-wait re-check (item A). Instrument a connect-call counter (monkeypatch or a spy on
  `connect`) so "connect was never called" is a hard assertion, not an inference.
- **A8-B2 — blocking-connect mutant (the load-bearing one).** Inject a connect implementation that
  **blocks beyond the remaining allowance unless passed a finite timeout** (e.g. a fake that sleeps, or
  points at a non-routable/black-hole address). Prove: a finite timeout is supplied, `timeout <=
  remaining`, and the worker **exits on the deadline** rather than hanging on the OS-global connect.
  **Mutation: `connect` called with no deadline must RED** — this is the arm that would have caught the
  shipped defect, so it must fail against the pre-fix `connect()`.
- **A8-B3 — success before deadline (control).** Connection + registration complete **inside** the
  remaining budget → still reconnects successfully. The deadline fix must not turn normal recovery into
  a false failure. This is the B4-style control that guards against over-correction.

## VERIFICATION

New suite: **29/29** (26 existing + A8-B1/B2/B3), green twice consecutively. Certified battery
unchanged and re-run: F1/F2 16/16 · admission-liveness 16/16 · exec-set 34/34 · phase-4 63/63 (the
Gate-22 self-clear is already resolved now that the test file is committed — confirm 63/63 directly).
Long suites `python3 -u <suite> | tee /tmp/<n>.log`, never `tail`.

## REPORT — `docs/CLAUDE_CODE_REPORT_DEFECT_A_R2_DEADLINE.md`

1. The three fixes at their source lines: post-backoff re-check, bounded recovery `connect(timeout=)`,
   bounded registration — and the **restore-to-blocking** guarantee that keeps the certified session
   loop timeout-free.
2. A8-B1/B2/B3 with red-first/mutation evidence; **A8-B2's no-deadline mutant shown failing** against
   the pre-fix `connect()`.
3. Confirmation §§10–13 and §15 are **byte-unchanged** (sha256 the untouched regions or state
   explicitly which functions changed — only `connect()` and `_recover_session()` should differ, plus
   the new gates).
4. Confirmation the restored (post-reconnect) session carries **no socket timeout** — the certified
   loop behaves exactly as before.
5. Files changed from `git status`. Any no-touch conflict returned, not worked around.
