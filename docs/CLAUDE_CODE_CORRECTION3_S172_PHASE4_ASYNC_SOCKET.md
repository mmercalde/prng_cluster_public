# Claude Code Instructions — S172 Phase 4 CORRECTION 3 (Beta: async-staging + socket defects)

**From:** Team Alpha lead
**For:** Claude Code on VM 101, `~/distributed_prng_analysis`, user `michael`
**Date:** 2026-07-19
**Status:** Team Beta REJECTED the resubmission — SIX remaining release blockers, all in
asynchronous staging and socket event handling (NOT the ledger or retry model, which Beta
accepted). Plus a production-timeout fix. Do NOT start Phase 5. Do NOT commit/push.

---

## Read first

Beta confirmed the previous six defects are genuinely fixed and passed gates 37–47. These
six are DEEPER — concurrency and socket-level failures that only surface under adversarial
load. Beta reproduced every one. Critically, Beta found that **gate 43 cheats**: it manually
acks a shard that in the real publish lifecycle cannot be acked until the whole attempt
completes — so the "back-pressure resumes" behavior was proven by a test doing something the
real system can't. That gate must be rewritten to use the real lifecycle (see #3).

Same rule as before: each defect gets (a) the fix, (b) a gate that FAILS on the current
code and passes fixed, using the REAL lifecycle — no manual acks of unpublished shards, no
test-only shortcuts. Do NOT weaken existing gates. If a fix conflicts with an existing gate,
STOP and report.

The core ledger / retry-matrix / resolver logic is accepted — do not redesign it. These
fixes are in the staging executor, the admission/high-water scheduler, `finalize_stripe`
failure routing, the hash-mismatch retry flag, the socket read loop, and the serve timeout.

---

## Defect 1 — timed-out transfer leaves an orphan file later (Beta-reproduced)

`_fetch_with_timeout` starts a `daemon=True` thread and, on `join(timeout)` expiry, raises
`StagingTimeout` and ABANDONS the still-running thread. The abandoned fetch can write the
temp file AFTER the failure path deleted it → orphan `.json.tmp.<pid>`. The L5 fence stops a
late *publish* but not a late *file write*.

**Fix:** an abandoned uncontrolled writer is not acceptable. Either:
- give the transfer real cancellation (pass a cancel token / use a fetch API that can be
  aborted), OR
- have the fetch write to a path uniquely owned by the task key and register a TRACKED
  completion callback that, when the late thread finishes, removes any artifact if the task
  is no longer active (fence-checked). The cleanup must catch the artifact whenever the late
  write lands, not just at timeout instant.

Prefer cancellation if `TransferAdapter` can support it; if not, the tracked-callback cleanup
must be provably complete (the callback runs on the staging executor and re-checks
`_attempt_active` + removes the temp path).

**Gate:** inject a slow `fetch_remote` that completes AFTER the timeout fired. Assert: after
the late completion, there is NO orphan temp/staged file and no reservation held (Beta's
probe: "files after late transfer" must be ledger-only / none).

---

## Defect 2 — staging executor queue is unbounded (Beta-reproduced)

`enqueue_staging` calls `self._staging_exec().submit(...)` directly. `ThreadPoolExecutor`
caps active threads but its work queue is UNBOUNDED, and the reservation happens only when a
job STARTS. 100 queued inline results each retain a survivor payload near the 48 MiB ceiling
→ ~GB of unaccounted RAM. High-water never sees queued jobs.

**Fix:** an explicitly bounded submission path. Options:
- a bounded semaphore sized to (max_workers + a small queue depth), acquired at
  `enqueue_staging` BEFORE retaining the payload; if it can't be acquired, apply dispatch-
  level deferral/back-pressure (do NOT retain the full payload in an unbounded queue), OR
- account pending-task COUNT and BYTES against high-water at submission time (before the
  message payload is retained), so a queued-but-not-started job counts toward capacity.

The invariant: total retained inline bytes (queued + active) must be bounded by the
high-water config, not just active threads.

**Gate:** submit far more inline tasks than `max_workers` with a tiny high-water; assert the
number of retained/pending payloads (and their bytes) never exceeds the bound — excess is
deferred at dispatch, not queued in memory. Beta's probe: queued tasks must NOT sit
unaccounted (the old probe showed 98 queued, 0 reservations).

---

## Defect 3 — attempt-scoped publish deadlocks against high-water (Beta-reproduced) + gate 43 is invalid

Circular dependency: shards stay reserved until Phase 5 acks; Phase 5 gets no manifests
until the whole attempt completes; but if early shards consume all high-water capacity, the
remaining shards of the SAME attempt can't stage → attempt never completes → nothing
published → nothing acked → capacity never frees. Beta reproduced with a 2-shard stripe and
`staging_high_water_files=1`: the second shard timed out and the whole stripe was reassigned
instead of waiting.

**Fix:** stripe-level ADMISSION accounting + a nonblocking deferred scheduler. At minimum:
- an admitted stripe attempt must be GUARANTEED enough high-water capacity to stage its
  COMPLETE attempt (all its sub-stripes) before it is admitted. Do not admit a stripe whose
  full attempt can't fit under high-water.
- within an admitted attempt, a sub-stripe that can't reserve yet must WAIT and resume
  (deferred, nonblocking), NOT time out and trigger reassignment. Timeout/reassign is for
  genuine worker failure, not self-inflicted capacity starvation.

**Gate 43 rewrite (mandatory):** the current gate manually acks the first shard before the
attempt is published — impossible in the real lifecycle. Rewrite it to use the ACTUAL
publish/ack path: a multi-shard attempt under constrained high-water must complete by
admission + deferred resume, with acks happening only AFTER `finalize_stripe` publishes the
whole attempt. No manual ack of an unpublished shard.

---

## Defect 4 — failed completion reconciliation never enters the matrix (Beta-reproduced)

`finalize_stripe` computes a `CompletionCheck`, but on a DEFINITIVE reconciliation failure
(StripeComplete arrived AND the structural result set is wrong — e.g. seed_count sum !=
stripe.seed_count, coverage not exact) it does nothing — the stripe sits in `staging` until
the global trial timeout, never retried.

**Fix:** distinguish "incomplete, still waiting" from "complete-but-invalid." If
StripeComplete has arrived and reconciliation is DEFINITIVELY failed (structural mismatch,
not just missing shards), call `handle_stripe_failure(...)` exactly once (phase-specific
matrix). Incomplete-but-consistent staging keeps waiting. Do not call the matrix while
shards are still legitimately in flight.

**Gate:** a hybrid stripe whose StripeComplete reports coverage that doesn't reconcile (bad
seed_count sum) → asserts `handle_stripe_failure` fired exactly once and the stripe entered
the matrix (reassigned attempt 1), NOT left in `staging`.

---

## Defect 5 — hash mismatch marked non-retryable (Beta-reproduced; contradicts a ruling)

The async staging handler does `except StagingHashMismatch: self._on_staging_failed(...,
retryable=False, ...)`. For a workflow Phase 3/4 (hybrid) stripe this aborts the trial on
attempt 0. The approved brief says a hash mismatch is a failed sub-stripe that feeds the
one-retry path.

**Fix:** `except StagingHashMismatch: self._on_staging_failed(..., retryable=True, ...)`. The
phase-specific matrix then does the right thing — constant phases still fail closed, hybrid
gets its one retry to a different worker.

**Gate:** a hybrid-phase stripe with a hash mismatch on attempt 0 → asserts reassignment to
attempt 1 on a DIFFERENT worker (`phase_degraded=True`), NOT trial abort. (And confirm a
constant-phase hash mismatch still fails closed, so the existing behavior for phase 1/2 is
preserved.)

---

## Defect 6 — server can block indefinitely on a silent/partial client (Beta-reproduced)

The server does a blocking `recv_msg()` after `select()` says a socket is merely readable —
but `select` readable means "some bytes," not "a full frame." A client that connects and
sends nothing, or sends a header + partial body, blocks ALL registration/heartbeats/dispatch
AND prevents the serve timeout from running. Beta reproduced: a silent client before a valid
worker → server thread alive, zero workers registered, timeout never fired until the silent
socket was closed.

**Fix:** nonblocking incremental frame ingestion. Either:
- per-connection reader threads that do blocking reads but feed a bounded coordinator queue
  (so one slow/silent connection can't stall the others or the timeout loop), OR
- nonblocking sockets + incremental frame parsing in the select loop (accumulate bytes per
  connection, only dispatch a message when a COMPLETE frame has arrived; never block on a
  partial frame).

The serve/timeout loop must keep running regardless of any one connection's state, and a
per-connection read deadline should drop a connection that never completes a frame.

**Gates:** (a) a silent client connects before a valid worker → the serve timeout still
fires and the trial aborts (server not wedged); (b) a partial-frame client (header + half
body) → does not block other connections' registration/dispatch, and the stuck connection is
eventually dropped on its read deadline.

---

## Production timeout correction (Beta-required)

`run_trial_miner` sets `"serve_timeout": kwargs.get("serve_timeout", 30.0)` and the
production integration never overrides it — a real multi-billion-seed scan takes far longer
than 30s, so the production path would abort valid trials.

**Fix:** default the serve timeout to `None`/unbounded, OR wire an explicit production trial
timeout through config. The serve loop enforces the timeout ONLY when one is configured (a
`None` timeout means run until terminal state). Ensure the `_use_miner` integration in
`window_optimizer_integration_final.py` passes an appropriate production value (or None).
Keep a short timeout available for the gates (they inject their own).

**Gate:** confirm `run_trial_miner` with no `serve_timeout` does not impose a 30s abort;
confirm a configured timeout is still honored (the gates rely on this).

---

## Verify + report

- Full harness green, exit 0, including the rewritten gate 43 (real lifecycle) and the new
  gates for defects 1/2/4/5/6 + the production-timeout gate. Each new/rewritten gate must
  FAIL on the current code.
- Phase-3 harness still 17/17.
- `git status` — expect the same file set (coordinator, worker, phase-1/3/4 tests,
  integration) + changelog; note any new file.
- Update the changelog: a "Correction 3" section per defect + gate, and confirm the
  gate-43-was-invalid finding is recorded honestly (it manually acked an unpublished shard;
  now uses the real lifecycle).

Report per defect: the fix, the gate, why it would have caught the original, and — for gate
43 — how the rewrite uses the real publish/ack path. Then STOP. Team Alpha re-reviews
adversarially (tracing the orphan-write, the queue RAM bound, the deadlock lifecycle, the
silent-socket timeout), then Team Beta. Do NOT commit/push.
