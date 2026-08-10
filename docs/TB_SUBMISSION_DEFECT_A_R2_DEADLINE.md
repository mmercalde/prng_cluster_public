# TEAM ALPHA → TEAM BETA — DEFECT A R2: §14 DEADLINE ENFORCEMENT — RE-CERTIFICATION

**Per your Defect-A certification ruling (2026-08-10):** *"RETURN FOR NARROW REVISION — §14 ONLY …
do not redesign the transport recovery architecture."* The three deadline-enforcement items are
implemented. **§§10–13 and §15 were not reopened.**

**State:** working tree at HEAD `acd6f13` (your certified-except-§14 base). **Nothing committed,
pushed, or launched** — awaiting Michael's direction per §24. Gate 12 and attempt 3 HELD. Report:
`docs/CLAUDE_CODE_REPORT_DEFECT_A_R2_DEADLINE.md`.

**Files:** `miner/range_miner_worker.py` (+65/−3 — `connect` and `_recover_session` only) ·
`tests/test_s172_defect_a_transport_recovery.py` (+A8-B1/B2/B3). **Coordinator untouched.**

**Suite: 29/29** (your 26 + three A8-B arms), green twice consecutively. Certified battery re-run
unchanged: F1/F2 16/16 · admission-liveness 16/16 · exec-set 34/34 · phase-3 17/17.

---

## THE THREE FIXES (your §15 A/B/C), verified by Alpha at source

**A — post-backoff exhaustion re-check.** After the backoff wait is charged, `remaining = budget -
_recovery_spent_s` is recomputed; `<= 0.0` → emit `RECONNECT_EXHAUSTED`, `return False`, **no further
`connect()`**. Closes the `remaining = 0.2s → wait → still connect` boundary violation. Gate A8-B1.

**B — bounded recovery connect (the load-bearing fix).** `connect()` takes an optional `timeout` and,
when set, passes it to `socket.create_connection((host, port), timeout=timeout)`. The recovery caller
supplies the **current remaining budget**; the first-startup path passes nothing and is byte-for-byte
unbounded as before. A `socket.timeout` is an `OSError`, already inside `TRANSPORT_EXCEPTIONS`, so it
is an ordinary charged/backed-off/retried recovery attempt. **This is the exact defect: previously the
budget never reached `create_connection`, so a black-holed route blocked past the clock. It now does.**
Gate A8-B2.

**C — bounded registration + restore-to-blocking.** `register_allowance = remaining -
(connect elapsed)`; if `<= 0` the REGISTER is not attempted (raise `socket.timeout` → retry path).
Otherwise `sock.settimeout(register_allowance)` bounds the send, then — on success, **before the
session is served** — `sock.settimeout(None)` restores ordinary blocking. This is load-bearing: a
lingering timeout would make the certified read loop raise `socket.timeout` on any quiet interval,
which classifies as TRANSPORT_LOSS and would loop an idle worker into perpetual reconnect. **The
deadline binds recovery ONLY; the restored session behaves exactly as a first session.** Gate A8-B3
asserts the restored session is blocking.

## SCOPE — nothing beyond §14 enforcement

Only `connect()` and `_recover_session()` changed in production. **AST function-level sha256 vs
`acd6f13`: 90 definitions, exactly 3 changed** (`connect`, `_recover_session`, and their containing
class), 87 byte-identical; module-level constants byte-identical; **coordinator byte-unchanged.** The
cumulative budget, backoff, same-identity rule, state machine, and coordinator behaviour are exactly as
you certified. `DEFAULT_WORKER_ADMISSION_TIMEOUT` remains **read, never redefined**. No §25 / no-touch
conflict.

## A8-B2 — a self-caught vacuous gate, fitting for this blocker

On its first run the A8-B2 mutant (a no-deadline `connect`) **survived** — the test's verbatim `socket`
copy resolved in the test module's globals and escaped the black-hole shim, so the mutant passed when it
must red. Fixed by rebinding the shim to the module under test; recorded in the report and at the site.
That the vacuity was in a gate *about an unenforced deadline* is the same class of defect this whole
revision closes — caught before it shipped, by the mutation discipline itself.

## GATE-22 DISCLOSURE — and a correction to Alpha's own brief

Phase-4 reads **62/63**. Alpha's revision brief asserted it would be 63/63 "now that the test file is
committed" — **that was wrong, and Claude Code correctly rejected it.** Gate 22 reads `git status
--porcelain`, which reports **modified tracked** files too, so editing the committed (non-allowlisted)
Defect-A **harness** re-trips it. The sole failure is the harness; `miner/range_miner_worker.py` is
allowlisted, so the production diff is in scope. The allowlist was **not** widened — it self-clears at
commit. Whether the harness should be permanently allowlisted is flagged as **your call**, not resolved
mid-revision. (Recorded as a fourth-in-arc example of the correlated-blind-spot pattern reaching Alpha's
own brief: the assumption "committed once → clears forever" was false because Gate 22 re-trips on
modification, not just untracked status.)

## §§10–15 NOT REOPENED

Per your disposition, the transport-recovery architecture is unchanged. This revision touches only the
deadline boundary. Your accepted sections stand:

- §10 state machine · §11 no-replay · §12 one-live-socket · §13 frozen-identity wall · §15
  observability — **byte-unchanged** (only `connect`/`_recover_session` differ, proven by the AST
  hash above).
- The three ratified judgement calls (close-before-shutdown, narrow send surface, cumulative budget)
  — unchanged.
- §23 forensics — no cause claimed; unchanged.

---

**Requesting:** re-certification of Defect A. Per your ruling, *"if that narrow revision is clean,
Defect A should be ready for final certification and Gate-12 attempt-3 authorization without reopening
§§10–13 or §15."* On your ruling and Michael's commit direction, this is the last amendment before
attempt-3 authorization (§26). Attempt 3 must still demonstrate simultaneity, turnover, four-stage
completion, publication and S145 coverage **live, in one run** — no credit composes from attempt 2.
