# TEAM ALPHA → TEAM BETA — F1-R DELTA (round 2, final)

**Per your ruling of 2026-08-06 (HOLD — F1-R + evidence repair).** Scope held to your §8
exactly: the reservation correction, G-RESUME-HANDOFF + live mutant, G-SUMMARY-NO-MASK, two
metrics fields, updated report. Two files. F2–F5 and matrix-diff gates UNTOUCHED — verified
below, not asserted. Nothing committed, nothing launched.

**Process note per your §6:** this cover was written AFTER the final-state runs on both
hosts. The evidence below describes the artifact submitted, nothing earlier.

## 1. F1-R correction — Beta Option A, disposition-bounded

The credit no longer clears at `inbound.put`; it rides with the envelope and is released by
the single-threaded serve path at DISPOSITION. Every clear path, enumerated:

| your §4 disposition | clear site |
|---|---|
| (i)/(ii)/(iii) admission / deferred / fenced | `dispatch_inbound_result` — a SEAM the serve loop now calls; `_serve_dispatch` runs verbatim inside it and the clear lives in its `finally`, firing only after dispatch returns |
| (iv) connection terminated | the serve loop's eof arm (FIFO of `inbound` guarantees a delivered credited envelope was dispatched before its eof, so this fires only for undispatched-and-gone) |
| (iv) trial terminated | `clear_any_resume_credit` at trial-terminal cleanup |
| (iv) wake delivered nothing | the reader's `finally`, now CONDITIONAL on `not credit_delivered` — a delivered wake hands its clear to the serve loop; clearing at exit would be round 1's defect one thread later |
| (iv) socket already reaped | the serve loop's `rawsock not in fs_by_sock` skip — **ratification item 1**: your (iv) one line before the eof arm. NECESSARY, not stylistic: a socket dropped by the READ-DEADLINE sweep (not eof) leaves queued envelopes with no eof tuple ever arriving; without this arm a credited envelope from a deadline-dropped socket leaks its reservation permanently |

**§4 tail:** the reader's result gate blocks a connection holding an undisposed reservation
from processing a FURTHER `sub_stripe_result` (`holds_resume_credit` /
`_await_resume_credit_clear`, same cadence/stop/discard semantics as the pause loop, no
ledger state). Heartbeats and completions flow — the §1.4 lease exemption depends on that.
No-wedge argument, in-source: the holder's envelope is already in `inbound`, dispatch is
single-threaded and unconditional, and eof / drop / terminal all clear serve-side.
**Fixed-delay was not used anywhere**, per your §4 last line.

**Metrics (§8.4):** `resume_credit_holder_worker` + `resume_credit_age_s` — needed because
"outstanding" alone can no longer distinguish a healthy in-flight handoff from a wedged one.

## 2. G-RESUME-HANDOFF — your §5, all eleven steps

Two real reader threads; saturate; pause A then B (FIFO asserted); free ONE unit, ONE
release path; wait until A's envelope is in `inbound` and A has left the registry; **no
dispatch, no test-thread semaphore touch**; hold 0.6 s (12 poll cycles) → B still paused, no
second envelope, reservation outstanding; dispatch A's envelope through the REAL serve path
(`dispatch_inbound_result`, the exact production call) → clear only after disposition;
release the second unit → B resumes second, FIFO preserved.

**Mutant (your §5):** the round-1 clear-at-`inbound.put` is restored at exactly the
differing instruction (the bench's `inbound` is injectable for this one purpose), proven to
execute, and **B resumes during the hold window on the still-unconsumed unit** — your §2
schedule, reproduced then closed. The round-1 G-RESUME-CREDIT-b remains as
capacity-accounting only; its docstring now states it does not cover the handoff invariant.

## 3. G-SUMMARY-NO-MASK — your §7, exactly

Malformed caps left in place through terminal construction (the F5 gate, which restores
them from the abort callback, is untouched — this is a separate gate): `run_trial_miner`
returns normally; the primary reason still leads `coordinator_staging_sizing:`;
`bound_in_force is None`; `bound_in_force_error` names the derivation exception; the
`[S172-BP] summary` line still emits. The Alpha guard is now directly executed by the suite.

## 4. "F2–F5 and existing gates unchanged" — verified, with one disclosed mechanic

**Ratification item 2:** G3, G5, G6 needed bench RE-SEQUENCING — the §4-tail rule
legitimately holds a connection's second result until disposition, so the bench's old
drain-everything-then-dispatch-everything can no longer observe it; the bench now pumps
interleaved, as the serve loop actually runs (and G3 uses the disposition clear alone, its
subject being wire order, not the handoff). **Alpha verified programmatically, not from the
report: the assertion lines of all three gates are byte-identical between rounds (6/4/6
assertions each).** All F2–F5 and matrix-diff gate assertions untouched; production
`_serve_dispatch` byte-unchanged (the seam wraps it).

**Ratification item 3:** the suite is **31 gates, not 30** — the required
G-MUT-RESUME-HANDOFF is its own check, as every other mutation gate in this suite is
(28 + G-RESUME-HANDOFF + G-MUT-RESUME-HANDOFF + G-SUMMARY-NO-MASK).

## 5. Final-state evidence (your §8 list)

- **VM101 (canonical): 31/31, three consecutive full runs, executed AFTER the last change**
  (Claude Code report §6; the base-state verification proved the round-1 + Alpha-guard tree
  before work began).
- **Alpha independent host: 31/31** on a fresh clone of `42bdbb1` + the cumulative patch —
  the identical bytes VM101 runs.
- `test_s172_staging_partb.py`: **24/24 VM101**; Alpha host 23/24 with the single red
  IDENTICAL at the clean `42bdbb1` baseline (the recorded environmental item).
- `test_s172_phase4_coordinator.py`: **63/63 VM101 by the accepted isolated
  production-diff method** (62/63 working tree = Gate 22's documented uncommitted-suite
  condition, untouched and unwidened); Alpha-host line-diff vs clean baseline shows ZERO
  differences beyond Gate 22 and the environmental Gate 54.
- Red-first: G-RESUME-HANDOFF red against the ROUND-1 worktree (the mutant IS round-1
  behavior); G-SUMMARY-NO-MASK red against round-1 with only the guard removed — round-1
  there raises `ValueError` out of `serve_trial` and destroys the honest terminal, which is
  the masking your §7 requires disproven.

## 6. Requested disposition

Approve the delta and, per your §9, authorize: the commit of the cumulative amendment
(clearing Gate 22), dual-push, the owner-initiated 4-stripe/25-daemon production-shape
trial (gate 12), and the Phase-7 soak behind it.
