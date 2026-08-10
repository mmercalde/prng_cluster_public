# TEAM ALPHA → TEAM BETA — DEFECT A: TRANSPORT-SESSION RECOVERY — CERTIFICATION REQUEST

**Per your Gate-12 attempt-2 forensic ruling §§7–16 and §24.** Defect A (RANGE-MINER
transport-session reconnect/re-register) is implemented, gated, and independently verified by Alpha.
**Defect B is CLOSED/CERTIFIED separately; this submission is Defect A only.**

**State:** uncommitted in the VM101 working tree at HEAD `990af60`. **Nothing committed, pushed, or
launched** — per §24, commit awaits Michael's direction. Gate 12 and attempt 3 remain HELD. Report:
`docs/CLAUDE_CODE_REPORT_DEFECT_A_TRANSPORT_RECOVERY.md`.

**Files:** `miner/range_miner_worker.py` (the §10 state machine, +431/−20) ·
`miner/range_miner_coordinator.py` (§15 emits only, +90) · new suite
`tests/test_s172_defect_a_transport_recovery.py`. Two-file diff for your inspection:

```bash
git diff -- miner/range_miner_worker.py miner/range_miner_coordinator.py
```

*(HEAD moved to `990af60` mid-session — your docs-only Defect-B closure commit. `f216475` remains an
ancestor; the worker/coordinator edits were unaffected, and every sha256 was re-run against the new
HEAD.)*

---

## SUITES — verified by Alpha in the VM101 working tree

- **New Defect-A suite: 26/26**, green twice consecutively, every arm paired with a `/M` mutation.
- **Certified battery, all confirmed by Alpha this session:** F1/F2 **16/16** · admission-liveness
  **16/16** · resolved-execution-set **34/34** · phase-4 **63/63** (see Gate-22 disclosure §6).

## 1. §10 THREE-WAY STATE MACHINE — the discriminator

`serve_forever` now serves **sessions**, not one socket. The load-bearing discriminator is
`_classify_session_end` (`range_miner_worker.py`): a transport exception authorises recovery **only
with `_stop` clear**. `_stop_cause` is **first-writer-wins** (`_set_stop_cause`), so an explicit
`shutdown` frame (`_dispatch`) and a signal (`_handle_sig`, now `shutdown(cause=STOP_CAUSE_SIGNAL)`)
stay distinguishable rather than collapsing into the one exit the defect was made of. `finally:
shutdown()` cannot overwrite the real cause. Gates: `G-DA-STATE-MACHINE`, A6 (+A6/M: a `_stop`-blind
classification misreads a signal).

## 2. §11 NO STALE-WORK REPLAY

`_abandon_assignment_no_replay` returns the worker to **idle** and re-sends nothing: no replayed
`SubStripeResult`, no stripe resumed from private state, no self-declared completion, no self-created
retry. The certified F1/F2 machinery alone decides the abandoned assignment. Gates A3 (constant-phase
loss stays TERMINAL, +A3/M a replayed completion is detected) and A4 (hybrid retry consumed exactly
once by the alternate, +A4/M budget stays exactly one).

## 3. §12 ONE ACTIVE CONNECTION PER IDENTITY

A duplicate registration that races the coordinator's eviction arrives as a transport failure; the
worker treats it as a **retryable session-establishment condition** — back off, retry the SAME
identity — never force-replace, never hold a second live socket. `_close_dead_session` drops the old
socket without setting `_stop`. Gate A5 (rejected → singular → retried → admitted, +A5/M a duplicate
binding must be refused).

## 4. §13 FROZEN COHORT REMAINS AUTHORITY — and a defect Alpha introduced, then the gate caught

Every reconnect re-sends the identity frozen at first registration; a drifted capability identity
raises `WorkerIdentityChanged` and fails closed. Gate A7 (same identity reconnects · changed identity
fails closed · non-frozen identity registered-but-never-eligible, +A7/M forgetting the freeze lets
drift through).

**DISCLOSURE — a §13 defect Alpha introduced and the red-first surfaced.** The first §13 wall compared
`dataclasses.asdict()` **whole**. `RegisterMessage` carries a per-message `timestamp`, so that
comparison would have failed closed on **every** reconnect — silently re-creating the no-reconnect
defect in a §13 costume (the exact correlated-blind-spot shape: the check encoding the same assumption
as the bug). **Gate A7's same-identity red-first is what surfaced it.** The fix is `IDENTITY_FIELDS`,
an enumerated allowlist of exactly the admission fields (worker_id, hostname, gpu_id, gpu_name,
backend, vram_bytes, capabilities); `timestamp`/`message_type`/`protocol_version` are envelope/framing,
correctly excluded. A gate catching Alpha's own mistake before it shipped is the mechanism working.

## 5. §14 RECONNECT BOUND

`recovery_budget_s()` is **positive-finite**, validated exactly as the coordinator validates
`worker_admission_timeout`, and **derived** from `DEFAULT_WORKER_ADMISSION_TIMEOUT` (read via lazy
import to avoid the coordinator↔worker circular import; the constant is on your §25 no-touch list and
is **read, never redefined**). The spend is **cumulative across all episodes**, not per-episode — a
per-episode budget resets each session and re-creates the immortal orphan §14 forbids under a
duplicate-rejection ping-pong. Exhaustion → clean exit, no orphan that could attach to a later trial.
Gate A8 (exits at the bound · derivation positive-finite from the 180 s anchor, +A8/M an infinite
bound must be refused).

## 6. §15 OBSERVABILITY — worker-side and coordinator-side, transition-only

- **Worker:** `_emit_session_event` on session transitions only (TRANSPORT_LOSS, RECONNECTED,
  RECONNECT_EXHAUSTED, IDENTITY_REFUSED, ASSIGNMENT_ABANDONED, SESSION_END) — worker_id, session
  generation, disconnect reason/class, assignment-active-at-loss, attempt number, success/exhaustion,
  explicit-shutdown-vs-loss. **No per-heartbeat noise.**
- **Coordinator (emits only, no logic change):** `WORKER_DISCONNECTED{worker_id, stage_idx,
  stage_assigned, identity_evicted, eligible_count_after_drop}` at `_drop_conn`, emitted **after**
  eviction so the count is what the next admission check sees; `WORKER_REGISTERED` /
  `WORKER_RECONNECTED{worker_id, registration_generation, eligible_count_after_register}` at
  `_serve_register`. **The S4 lesson is carried:** an unmeasurable eligible count is reported
  `UNOBSERVED`, never `0` (`G-DA-OBS-UNOBSERVED`) — a zero pool is a real and different fact from an
  unmeasured one. `identity_evicted=False` distinguishes the fenced-replacement case (socket died, id
  live on another socket, pool did not shrink) from a real loss. **The goal is met: a future 23/25
  names its two IDs and the transition directly, instead of being reconstructed indirectly** — which
  is precisely what attempt 2 could not do.

**§25 NO-TOUCH — verified clean.** `_registration_generation` is a **record-keeping counter only**,
run-scoped (cleared at each `serve_trial`); nothing reads it to decide eligibility. `_eligible`, the
admission gate, `expected_workers`, lease semantics, retry budget, `worker_admission_timeout`,
staging/backpressure, and all data/coverage/publication authority are **read, never redefined**. Gate
`G-DA-NO-TOUCH` asserts this. No conflict encountered.

## 7. A1/RED — mutant authenticity

`_PREFIX_serve_forever` is the pre-fix `serve_forever` body **copied verbatim from `f216475`** (the
defect site you named — the `while not _stop.is_set(): try: recv except: break` loop with `_dispatch`
outside the try). The A1/RED arm swaps it in and proves the suite **reds against the real defect**: the
pre-fix daemon exits silently and successfully and the identity is lost for good. A second red-first
covers the `_stop`-blind classification. This is the same mutant-authenticity discipline that closed
the R3 P1 hole — the 26 green arms test the real bug, not a strawman.

## 8. A FINDING THE BRIEF DID NOT NAME

`self._dispatch(msg)` sat **outside** the inner `try` in the pre-fix loop, so the single collapse was
really into **two** silent exits: an idle (recv-side) loss returned 0, while a loss **mid-result-stream**
(send-side) propagated as an **uncaught traceback**. A1's shape was the first; A3/A4's the second. Both
now classify through one seam (`SEND_TRANSPORT_EXCEPTIONS` on the dispatch path, deliberately narrower
than the read side — an oversized-frame `ValueError` on write is a payload-contract violation, not a
dead socket, and stays uncaught so it cannot be swallowed by a reconnect).

## 9. §23 FORENSICS — partly silent, NO cause claimed

Per your §23 and the standing correction against unproven causes: **Alpha claims no initiating cause
for the two lost workers.**

- All 25 identities were working in stage 3.
- The two lost workers are **not nameable from any local artifact** — zero coordinator WARN/ERROR
  between 09:47:29 and 10:44:16. This is §15's own complaint, confirmed, and the reason the emits above
  exist.
- The loss window is bounded: the earliest finishers held an idle session up to ~52 min
  (09:48:51 → 10:41:16).
- Local kernel ring: zero entries in the window.
- Rig netconsole: empty — **with the caveat that this cannot distinguish "no event" from "not
  active".**
- **No TCP idle timeout claimed.**
- Rig-side worker logs are **unexamined** (§24 forbids SSH) — **not silent.** Available for a future
  authorized pass.

The amendment is justified independently of this: the no-reconnect gap is a defect regardless of what
initiated the two losses, and §15 ensures a recurrence is nameable.

## 10. THREE JUDGEMENT CALLS — flagged for your ruling, reversible

1. **`MinerFramedSocket.close()` now shuts down before closing** — required for the SIGNAL leg to
   actually terminate a daemon parked in a blocked `recv`; mirrors the coordinator's own Defect-6 C3
   reasoning.
2. **Send-side exception surface is narrower than the read side** (`ValueError` excluded) — an
   oversized-frame `ValueError` is a contract violation, not a dead socket; recovering from it would
   swallow a real defect.
3. **Recovery budget is cumulative, not per-episode** — a per-episode budget re-creates the §14
   immortal orphan under a duplicate-rejection ping-pong. Reversible in one line if you prefer
   per-episode.

## 11. GATE-22 DISCLOSURE

Phase-4 read **62/63** with the new test file present — solely the Gate-22 untracked-`.py`
sensitivity (fourth occurrence in this arc). Alpha proved it self-clears (**63/63** with the file
temporarily parked out of the tree) and **did NOT widen the allowlist** — the standing rule (commit
the file, it self-clears on tracking) applies. No real phase-4 regression.

---

**Requesting:** Defect A certification. On your ruling and Michael's commit direction, this is the
final amendment before Gate-12 attempt-3 authorization (§26). Attempt 3 must still demonstrate
simultaneity, turnover, four-stage completion, publication and S145 coverage **live, in one run** — no
credit composes from attempt 2.
