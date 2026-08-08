# CLAUDE CODE INSTRUCTIONS — S172-BP AMENDMENT ROUND 2 (Beta F1-R)

**Authority:** Team Beta ruling *"S172-BP F1–F5 AMENDMENT DELTA — HOLD"* (2026-08-06).
F2/F3/F4/F5 and the matrix isolation are **APPROVED AND CLOSED — do not touch them.**
This round's scope is Beta §8, exactly: (1) the F1 reader-to-dispatch reservation
correction, (2) G-RESUME-HANDOFF + live mutant, (3) G-SUMMARY-NO-MASK, (4) at most a
narrow metrics field for reservation state, (5) updated report. Files:
`miner/range_miner_coordinator.py` and `tests/test_s172_staging_backpressure.py` ONLY.

**Base:** the current VM101 working tree (42bdbb1 + round-1 amendment + Alpha's
terminal-summary patch, the state that ran 28/28 on VM101). Verify before starting:
`git stash list` empty, `git diff --stat` shows exactly the two files, and
`grep -c "bound_in_force_error" miner/range_miner_coordinator.py` returns ≥1 — if the
Alpha patch is absent, STOP and report.

**Hard constraints:** no commit, no push, no launch. F2–F5 gates and G-MATRIX-DIFF must
remain green and UNCHANGED (Beta §8: "all existing F2–F5 and matrix-diff gates unchanged
and green"). `_handle_stripe_failure_locked`, the six `_on_staging_failed` callers, worker
code, seed caps, geometry: untouchable.

---

## 1. F1-R — the defect, so the fix is understood not transcribed

The credit currently clears at `inbound.put`. But the freed staging slot is consumed only
when the SERVE LOOP later dispatches that envelope into `enqueue_staging`. In the gap —
envelope in `inbound`, slot still physically free — reader B's `_try_self_resume` finds:
B is FIFO head (A deregistered), credits == 0 (A cleared at put), `staging_can_accept()`
true **on the same slot**. Two wakes, one slot. Beta §2 names the schedule reachable, and
the round-1 gate was falsely green because it reacquired the slot from the test thread
("modelling the serve loop") — deleting exactly the interval under proof.

**The invariant Beta requires (§4):** the reservation survives until the single-threaded
serve path has DEFINITIVELY DISPOSED of the credited envelope — one of: (i)
`enqueue_staging` acquired admission; (ii) the envelope was retained in the bounded
deferred queue; (iii) the existing identity/attempt/dedup/terminal fence rejected it;
(iv) the connection or trial terminated and the envelope was discarded. A fixed delay is
explicitly NOT acceptable.

## 2. Required implementation — Beta Option A (dispatch acknowledgment)

Design, using the fact that at most ONE credit exists globally and
`_resume_credit_holder` already names its connection:

1. **Reader no longer clears on delivery.** After the successful `inbound.put` of the
   held envelope, the reader marks (locally) `credit_delivered = True` and does NOT call
   `_release_resume_credit`. The reservation now rides with the envelope.
2. **Serve-side disposition clear.** In the serve loop, wrap the dispatch of each
   dequeued `("msg", rawsock, msg)` so that — in a `finally`, covering accepted,
   deferred, fenced, rejected, and exception paths alike — if `rawsock` is the current
   `_resume_credit_holder` and `msg.message_type == "sub_stripe_result"`, call
   `_release_resume_credit(rawsock, delivered=True)`. Because dispatch is single-threaded
   and the credited envelope is by construction the FIRST result from that connection
   after its resume, this fires on exactly the credited envelope, after `enqueue_staging`
   (or a fence/rejection) has run — i.e., after disposition (i)/(ii)/(iii). Place the
   clear so it runs AFTER the dispatch call returns, never before.
3. **eof clear.** In the serve loop's eof handling for `rawsock`: if that connection
   holds the credit, clear it (disposition iv — connection terminated; FIFO ordering of
   `inbound` guarantees the credited envelope, if delivered, was already dispatched
   before its eof, so this only fires for undispatched-and-gone cases).
4. **Terminal clear.** In the trial-terminal cleanup (where
   `clear_all_capacity_resume_grace` already runs), clear any outstanding credit
   unconditionally (disposition iv — trial terminated).
5. **Reader-exit clear becomes conditional.** The unconditional
   `_release_resume_credit(rawsock, delivered=False)` in the reader's `finally` now runs
   ONLY IF `credit_delivered` is False — a wake that delivered nothing must not reserve
   the observation forever (unchanged), but a wake that DID deliver hands the clear to
   the serve loop; clearing at exit would reopen the exact F1-R window. Keep the
   deregister-before-clear ordering comment; extend it with this rule.
6. **No second result while the reservation is out (Beta §4 tail).** At the reader's
   result gate: if this connection is `_resume_credit_holder`, do not process a further
   `sub_stripe_result` — wait (50 ms cadence, honoring `reader_stop` and trial-terminal,
   discard-and-exit semantics identical to the pause loop) until the credit clears, then
   fall through to the normal capacity gate. Without this, one connection can stream
   several results against the same apparent capacity. Heartbeats/completions pass
   through unchanged.
7. **Metrics (§8.4 allowance):** add `resume_credit_holder_worker` and
   `resume_credit_age_s` (None when clear) to `staging_backpressure_metrics`; keep the
   existing counters. Nothing else.

Comment the invariant at `_release_resume_credit`: *"the reservation ends at DISPOSITION
(Beta F1-R §4 i–iv), never at ingress — inbound.put moves the envelope, it does not
consume the slot."*

## 3. Gate G-RESUME-HANDOFF — Beta §5, all eleven steps, verbatim

Two REAL paused reader threads. Saturate staging; pause A then B; assert FIFO. Free
exactly ONE unit; invoke ONE release path. Wait until A's envelope is in `inbound` and A
has left the pause registry. **Do not dispatch it; do not touch the semaphore from the
test thread.** Hold ≥ 0.6 s (≥ 12 defensive poll cycles). Assert: B still paused; no
second envelope in `inbound`; the reservation still outstanding
(`resume_credits_outstanding() == 1`). Then dispatch A's envelope through the REAL serve
path; assert the credit cleared only after disposition. Release the next unit; assert B
resumes second; FIFO preserved.

**Required mutant:** restore the round-1 clear-at-`inbound.put`; prove it executes; prove
B resumes DURING the hold window on the still-unconsumed unit. The round-1
G-RESUME-CREDIT-b stays as a capacity-accounting gate but its docstring must state it
does NOT cover the handoff invariant (Beta §5 last line).

## 4. Gate G-SUMMARY-NO-MASK — Beta §7, exactly

Fail stage sizing with malformed caps and LEAVE the record malformed through terminal
summary construction (the existing F5 gate restores it from the abort callback — do not
modify that gate; this is a separate one). Assert: `run_trial_miner` returns normally;
the primary abort reason still leads `coordinator_staging_sizing:`; the summary's
`bound_in_force is None`; `bound_in_force_error` names the derivation exception; the
`[S172-BP] summary` line still emits. This is the direct execution of the Alpha guard
that the round-1 suite never reached.

## 5. Evidence and report

Beta §6 lesson is binding on process: **the final canonical-host run happens AFTER the
last change, and the report is written after that run** — evidence must describe the
final artifact, nothing earlier. Deliver
`docs/CLAUDE_CODE_REPORT_S172_BP_AMENDMENT_R2.md` with: the disposition-clear design and
every clear path enumerated (dispatch/eof/terminal/undelivered-exit); red-first evidence
for both new gates against the round-1 state (worktree carrying round-1 + Alpha patch —
NOT bare 42bdbb1, since the mutant IS the round-1 behavior); the mutant execution proof;
full suite result (expect 30/30) on VM101; `test_s172_staging_partb.py` on VM101; the
phase-4 isolated production-diff method result; confirmation F2–F5 and matrix-diff gates
byte-unchanged (diff the suite against round 1 and list touched gate functions);
files-changed (exactly two); disagreements reported, not worked around.
