# CLAUDE CODE INSTRUCTIONS — S172-BP AMENDMENT (Beta findings F1–F5)

**Authority:** Team Beta ruling *"S172 STAGING BACK-PRESSURE REMEDIATION — HOLD, TARGETED
FIX-FORWARD REQUIRED"* (2026-08-06), reviewing commit `4b1aad6`. The architecture is APPROVED;
this amendment lands the five localized corrections and five targeted gates the ruling
requires. Fix-forward on `4b1aad6` (now `42bdbb1` tip); **no history rewrite**.

**Host:** VM101, repo `~/distributed_prng_analysis`. `source ~/venvs/torch/bin/activate` before
every test command. Long suites: `python3 -u <suite> | tee /tmp/<name>.log`, never `| tail`.

**Hard constraints (Beta §8.8):** NO `git commit` / `git push` / pipeline launch. **No changes
to worker code, seed caps, stripe geometry, or `gate_s172_prod_shape.py`.** The six surviving
`_on_staging_failed` callers and `_handle_stripe_failure_locked` remain byte-identical —
G-MATRIX-DIFF must stay green and its AST evidence is part of the resubmission. Files in scope:
`miner/range_miner_coordinator.py` and `tests/test_s172_staging_backpressure.py` ONLY. If a
fix appears to need any other file, STOP and report.

Read every anchor live before writing. Line numbers below are from the committed tree at
`4b1aad6`; verify each.

---

## F1 — one wake per capacity release (BLOCKING)

**Defect (Beta §2):** `_resume_paused_connections` sets one event, re-checks
`staging_can_accept()`, and loops — but a wake does not CONSUME the observed capacity, so one
freed slot can satisfy the check repeatedly and wake the whole paused fleet.
**Second herd vector Beta's text does not name but its gate will red:** the paused reader's own
50 ms defensive poll (`if self.staging_can_accept(): released = True`) lets EVERY paused reader
self-release on the same observation. Both doors must close.

**Required semantics (implement with an ingress-credit counter under `_pause_lock`):**

1. `_grant_resume_credit()` — under `_pause_lock`: if `staging_can_accept()` AND there exists
   an unsignaled paused reader AND `_resume_credits_outstanding == 0` for that observation
   window, increment `_resume_credits_outstanding`, set the **FIFO-oldest unsignaled** reader's
   event, log `[S172-BP] resume_signal`, and **return immediately**. One invocation ⇒ at most
   one wake (Beta's minimum rule). `_resume_paused_connections` becomes exactly one call to
   this; the while-loop is deleted. Multiple release events (each `_on_done` → `_pump_deferred`
   → `finally`) each grant at most one — liveness is per-event, not per-loop.
2. **Credit consumption:** the credit is cleared when the woken reader hands its held envelope
   to `inbound` (reader-side, under `_pause_lock`, immediately after the successful
   `inbound.put`) or when that reader exits without delivering. Until cleared, no further grant
   is issued — a wake reserves the observation.
3. **Defensive poll becomes head-only self-grant:** replace the reader's bare
   `staging_can_accept()` escape with `self._try_self_resume(rawsock)` — under `_pause_lock`:
   succeeds ONLY if this connection is the FIFO-oldest paused entry AND
   `_resume_credits_outstanding == 0` AND `staging_can_accept()`; on success it takes the
   credit itself. This preserves the lost-wakeup protection (the head can always escape when
   capacity truly exists and no grant is in flight) while making it impossible for a non-head
   reader to ride someone else's observation. The event-wait remains the primary path;
   `resume_event.wait(0.05)` cadence unchanged.
4. The documented margin (§1.2) is unchanged and remains the final backstop; say so in the
   comment rather than resizing anything.

**Gate G-RESUME-CREDIT (mutation evidence REQUIRED):** ≥2 registered paused connections; free
exactly ONE staging capacity unit; invoke exactly ONE capacity-release path; assert exactly one
reader (the FIFO-first) resumes and the other REMAINS paused across a real settling window;
free a second unit; assert the second resumes; assert FIFO order throughout. The mutant
(restore the loop, or let non-head self-resume) must execute and red the gate.

## F2 — lease-exemption resume grace (BLOCKING; Beta's §1.4 amendment)

**Defect (Beta §3):** exemption covers only live membership in `_paused_connections`. On
resume the reader deregisters FIRST, delivers the envelope, and only later reads the delayed
heartbeat — leaving a window (worker unpaused, lease long expired at up to 600 s pause vs
300 s lease, heartbeat unprocessed) where `process_lease_expiry` routes the stripe into the
matrix.

**Required mechanism (Beta specified it; implement as written):**

1. New coordinator dict `_capacity_resume_grace: Dict[str, float]` under `_pause_lock`.
2. In `deregister_paused_connection`, when `reason == "resume"` and `worker_id` is not None:
   `_capacity_resume_grace[worker_id] = now + compute_lease_timeout`. Reader-thread write to a
   coordinator dict under `_pause_lock` — **no ledger mutation**, preserving the reader rule;
   state this in the comment.
3. `process_lease_expiry` skips a stripe if its `claimed_by` is actively paused **or** has a
   live grace entry (prune expired entries in the same pass).
4. The heartbeat branch of `_serve_dispatch` (`:5067`), after `renew_lease` succeeds, clears
   that worker's grace entry — the real lease is renewed; the bridge is no longer needed.
5. Clear the worker's grace entry on connection drop (the serve loop's eof/cleanup path for
   that worker) and at trial-terminal cleanup, so no grace outlives its connection or trial.
6. Grace expiry with no heartbeat ⇒ normal expiry handling resumes (the skip simply stops
   matching) — this is what keeps the exemption bounded and narrow.

**Gate G-LEASE-HANDOFF (mutation evidence REQUIRED):** pause a worker past its compute-lease
deadline; resume while the heartbeat is still queued behind the held result; run
`process_lease_expiry` inside that window → assert zero matrix entries; then process the
heartbeat → assert normal renewal and grace cleared; separately, a resumed worker that never
heartbeats expires after the grace bound. Mutant (grace recording removed) must execute and
red the gate.

## F3 — timeout evidence snapshot

**Defect (Beta §4):** a reader can observe `staging_capacity_timeout_expired()`, latch it,
deregister and exit before the serve loop builds the terminal reason —
`staging_capacity_timeout_reason()` then reads the CURRENT registry and can truthfully report
`0 connections paused (none)` about a timeout that paused workers caused.

**Fix:** at the moment the latch is first set inside `staging_capacity_timeout_expired`,
atomically (under `_pause_lock`, same critical section as the oldest-pause read) capture
`_capacity_timeout_snapshot = {latched_at, oldest_since, paused_count, worker_ids}`.
`staging_capacity_timeout_reason()` and `staging_backpressure_metrics()` use the snapshot when
present; the live registry only if the timeout never latched. The count/identities in the
terminal reason must be the TRIGGERING ones.

**Gate G-TIMEOUT-SNAPSHOT:** short `staging_capacity_timeout`; force the reader thread to
observe the latch and fully deregister/exit BEFORE the serve loop terminates; assert the abort
reason names the actual worker id and a nonzero paused count, and the metrics carry the same
snapshot values.

## F4 — registered workers only

**Defect (Beta §5):** the reader's pause condition tests message type + capacity but not
identity, so an unregistered socket sending a well-formed `sub_stripe_result` under saturation
acquires pause state (`worker_id=None`), consumes the envelope allowance, joins the oldest-
pause clock, and is held BEFORE the serve loop's identity rejection can see it.

**Fix:** the pause path additionally requires `worker_by_sock.get(rawsock) is not None`
(`worker_by_sock` is written only at registration, `:5035`, so this IS the bound-worker
predicate). An unbound result under saturation is NOT paused and NOT held — it flows to
`inbound` unchanged and dies in the existing serve-loop identity/protocol rejection, exactly
as pre-amendment. No new rejection logic; the point is to stop intercepting the message before
the existing guard.

**Gate G-BOUND-PAUSE:** saturate staging; send a valid `sub_stripe_result` on a connection
that never registered; assert no pause record, no grace record, no snapshot/timeout
attribution to that socket, and that the message reached the existing identity rejection path.

## F5 — stage-bound derivation failure fails closed

**Defect (Beta §6):** the stage-setup `except (ValueError, TypeError)` falls back to the
on-demand derivation — one macro-stripe, phase 1 — which can be MATERIALLY SMALLER than the
failed stage derivation (multi-stripe stages, hybrid caps). A sizing failure silently re-arms
the undersized-queue condition.

**Fix:** in the stage-setup block, materialize `stripe_spans` and the eligible-worker records
ONCE at entry; on any derivation exception, log the full context and terminate via **direct**
`fail_trial(run_id, reason="coordinator_staging_sizing: could not derive the staging deferred
bound for stage <n> — <cause>")` — never the matrix, never a smaller implicit bound, and
BEFORE any result traffic for that stage. `_derive_bound_from_current_state` survives ONLY for
bare-API/gate contexts where no stage derivation was attempted; add a comment stating it is
not a production fallback, and that every production stage installs its bound at setup or
fails closed.
**Also (Beta item-3 ratification detail):** the §1.6 invariant reason must distinguish which
bound tripped — derived-count vs operator-override-count vs retained-bytes high-water — as
three explicit phrases in the reason string. The arithmetic already carried is retained.

**Gate G-BOUND-DERIVATION-FAILURE:** inject a malformed worker-cap record at stage setup;
assert the trial terminates with the `coordinator_staging_sizing:` reason before any staging
of that stage's results, no matrix entry, no retry consumed — and that execution never
continued on the one-slot fallback (assert `_derived_deferred_bound` was never silently None
while results flowed).

## Resubmission evidence (Beta §8, all eight items)

1. Production corrections F1–F5 — `miner/range_miner_coordinator.py` only.
2. Five gates above appended to `tests/test_s172_staging_backpressure.py`, each proven RED
   against the `4b1aad6` behavior first (worktree at `4b1aad6` for the red runs).
3. Mutation evidence for G-RESUME-CREDIT and G-LEASE-HANDOFF (mutant applied, executed,
   reached the credited assertion).
4. Full revised suite green on VM101 (Alpha re-runs on the sandbox host independently).
5. `test_s172_staging_partb.py` green (24/24 on VM101).
6. `test_s172_phase4_coordinator.py` green (63/63 on VM101; the new-file Gate-22 red clears at
   commit as before).
7. AST evidence (G-MATRIX-DIFF re-run output) that the six `_on_staging_failed` callers and
   the retry matrix are unchanged by this amendment.
8. Report: `docs/CLAUDE_CODE_REPORT_S172_BP_AMENDMENT.md` — per-finding fix description with
   anchors, red/green per gate, files-changed (expect exactly two), any disagreement with this
   brief reported rather than worked around.

Production-shape (gate 12) and the soak remain **NOT AUTHORIZED** until Beta reviews this
delta. CPU gates and regression runs are authorized.
