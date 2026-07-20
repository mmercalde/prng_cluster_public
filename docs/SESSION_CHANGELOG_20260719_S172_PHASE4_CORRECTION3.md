# SESSION CHANGELOG — 2026-07-19 — S172 Phase 4 CORRECTION 3 (async-staging + socket defects)

**Team Alpha (Claude) implementing. Team Beta is the binding approval authority.**
Instructions: `docs/CLAUDE_CODE_CORRECTION3_S172_PHASE4_ASYNC_SOCKET.md`.
Scope: fix SIX Beta-reproduced release blockers (all in asynchronous staging +
socket event handling) + the production-timeout issue, and REWRITE the invalid
gate 43. The accepted ledger / retry-matrix / resolver logic was **not** redesigned.

**Status: harness GREEN — Phase 4 = 54/54 (stable across 3 runs), Phase 3 = 17/17.**
Every new/rewritten gate was verified to FAIL on the pre-fix code (targeted per-defect
reverts) and pass fixed. NOT committed / pushed. WATCHER pipeline NOT run.

Files changed: `miner/range_miner_coordinator.py`,
`tests/test_s172_phase4_coordinator.py` (gate 43 rewrite + gates 48–54),
`window_optimizer_integration_final.py` (production serve_timeout wiring).
fallback parity: code=current, env=ok (no dependency change this session).

---

## The gate-43-was-invalid finding (recorded honestly)

Beta was correct: the **old gate 43 CHEATED**. It proved "back-pressure resumes"
by manually calling `ack_shard(...)` + `release_after_ack(...)` on sub-stripe 0 of
an attempt **before that attempt was ever published**. In the real lifecycle a
shard can only be acked by Phase 5 *after* `finalize_stripe` publishes the WHOLE
attempt (`publish_attempt`), so the manual ack of an unpublished shard is something
the real system can never do. The gate is now rewritten
(`gate43_admission_deferred_resume_real_lifecycle`) to use the ACTUAL publish/ack
path: a multi-shard attempt completes by **admission + nonblocking deferred resume**,
and acks happen ONLY via `ack_by_event_id(...)` on the manifests that
`finalize_stripe` actually published. No manual ack of an unpublished shard remains
anywhere in the harness.

---

## Defect 1 — timed-out transfer leaves a later orphan file

**Fix** (`_fetch_with_timeout`): the fetch daemon is no longer merely abandoned. A
tiny lock serializes an `abandon`-vs-`done` decision; on timeout the caller marks
the fetch ABANDONED and the fetch thread's own `finally` observes that and removes
any temp artifact it produced — so the cleanup fires **whenever** the late write
lands, not only at the timeout instant. `temp_path` is attempt/generation-PRIVATE
(`_staged_path`), so this can only ever remove the task's own file.

**Gate 48** (`gate48_late_transfer_no_orphan`): a `fetch_remote` that writes the
temp 1.0 s AFTER a 0.3 s `fetch_timeout` fires. Asserts the reservation is released
at the timeout AND, after the late write lands, the staging dir is empty (no orphan).
**Would have caught it:** pre-fix the abandoned daemon's late write orphaned
`…_a0_s0_g0_…json.tmp.<pid>` that nothing removed (gate saw exactly that file).

## Defect 2 — unbounded staging executor queue (RAM exhaustion)

**Fix**: `enqueue_staging` now acquires one slot of a bounded
`BoundedSemaphore(staging_workers + staging_queue_depth)` BEFORE the payload is
retained, submitting via `_submit_staging` (slot released on job completion). The
number of in-flight retained payloads (queued + active) is therefore capped — excess
is deferred at dispatch (the producer blocks on the slot), never queued unbounded in
memory. New config: `staging_queue_depth` (default 2).

**Gate 49** (`gate49_bounded_staging_queue`): a producer submits 20 remote tasks
(fetch held on a gate) against `staging_workers=2, staging_queue_depth=2`. Asserts
retained/submitted payloads never exceed the bound (4) and only `workers` fetches
actually run; on release all 20 drain. **Would have caught it:** pre-fix the producer
submitted all 20 immediately (20 > 4) — Beta's "98 queued, 0 reservations".

## Defect 3 — attempt-scoped publish deadlocks against high-water (+ gate 43 invalid)

**Fix**: stripe-level ADMISSION + nonblocking deferred resume.
- `_try_admit_locked` admits an attempt only if its WHOLE footprint (one file per
  expected sub-stripe, each ≤ the §15 inline byte ceiling) still fits under
  high-water alongside already-committed attempts. A non-admissible attempt's
  sub-stripes are DEFERRED (`_deferred`, bounded — not submitted to an unbounded
  queue) and resume via `_pump_deferred` when capacity frees (ack / release /
  cleanup / abort / job completion).
- Within an ADMITTED attempt, a sub-stripe that can't reserve yet WAITS and resumes
  (nonblocking, off the dispatch loop) — it is no longer timed out into the retry
  matrix. Timeout/reassign is reserved for genuine worker failure, not self-inflicted
  capacity starvation.
- Admission budgets are de-committed when the attempt is terminal (done — reservations
  will be acked) or superseded (`_prune_admitted_locked`, `_release_admission`).

**Gate 43 REWRITE** (`gate43_admission_deferred_resume_real_lifecycle`): two 2-shard
stripes contend for a high-water that fits exactly ONE full attempt
(`staging_high_water_files=2`), delivered in the poison interleaving A.0,B.0,A.1,B.1.
Attempt A is admitted and completes; attempt B is DEFERRED until A is published and
its shards acked via the REAL `ack_by_event_id` path — only then does B resume and
complete. **Would have caught it:** pre-fix (no admission) A.0+B.0 fill capacity and
A.1+B.1 starve forever → neither attempt completes → nothing published/acked → the
gate's `.result(timeout=5)` raises (self-deadlock).

## Defect 4 — failed completion reconciliation never enters the matrix

**Fix** (`finalize_stripe`): distinguishes "incomplete, still waiting" from
"complete-but-invalid." When StripeComplete has arrived and ALL expected shards are
present (`substripes_match`) but the accounting does not reconcile (seed_count sum /
survivor sum / coverage), it routes through the phase-specific matrix EXACTLY ONCE
instead of parking in `staging` until the global trial timeout. Routing requires the
eligible-worker set, so it fires only when the real lifecycle (serve dispatch /
staging-job completion) drives finalize — a bare predicate call still just evaluates
(preserving gates 3/4/6 that call `finalize_stripe` without an eligible provider).

**Gate 50** (`gate50_definitive_reconciliation_to_matrix`): a hybrid stripe with all
3 expected shards verified but a seed_count sum of 29 ≠ 30 → reassigned to attempt 1
on a DIFFERENT worker (phase_degraded), NOT left in staging; a second finalize is a
no-op (fired exactly once). **Would have caught it:** pre-fix finalize did nothing on
a complete-but-invalid reconcile → stripe stuck in staging at attempt 0.

## Defect 5 — hash mismatch marked non-retryable (contradicted a ruling)

**Fix** (`_run_staging_job`): `except StagingHashMismatch` now calls
`_on_staging_failed(..., retryable=True, ...)`. The phase-specific matrix then does
the right thing — constant phases (1/2) still fail CLOSED, a hybrid stripe (3/4) gets
its single retry to a DIFFERENT worker.

**Gate 51** (`gate51_hash_mismatch_retryable_hybrid`): (a) a hybrid-phase hash
mismatch on attempt 0 reassigns to attempt 1 on a different worker (trial still
running); (b) a constant-phase hash mismatch still fails closed (trial aborted, attempt
not consumed). **Would have caught it:** pre-fix `retryable=False` aborted the hybrid
trial on attempt 0.

## Defect 6 — server blocks indefinitely on a silent/partial client

**Fix** (`serve_trial`): each accepted connection now gets its OWN reader thread doing
blocking full-frame reads (a legitimately slow-but-complete frame is never corrupted)
into a bounded coordinator queue; the serve loop only ever DRAINS that queue — it
never does a blocking `recv` on the dispatch thread. A per-connection read deadline
drops any connection that never completes a frame (silent or partial), and `_drop_conn`
now `shutdown()`s before `close()` so a reader blocked in `recv` is woken and the peer
promptly sees EOF. The timeout / assignment loop keeps running regardless of any one
connection's state.

**Gate 52** (`gate52_silent_client_timeout_and_deadline`): a silent client connecting
before any worker is dropped on its read deadline (EOF well before the serve timeout)
AND the serve timeout still fires → trial aborts (server not wedged, zero workers).
**Gate 53** (`gate53_partial_frame_nonblocking_and_dropped`): a header + partial body
client connecting before a valid worker does NOT block the valid worker's registration
+ commit, and the stuck connection is dropped (EOF). **Would have caught them:** pre-fix
the blocking `recv_msg()` right after accept wedged registration/dispatch/timeout until
the socket closed (both gates hang / never terminate pre-fix).

## Production timeout correction

**Fix**: `run_trial_miner` now defaults `serve_timeout` to `None` (unbounded — a real
multi-billion-seed scan far exceeds any fixed 30 s); `serve_trial` enforces a timeout
ONLY when one is configured. `window_optimizer_integration_final.py:_use_miner` passes
`serve_timeout=getattr(coordinator, 'serve_timeout', None)`. Gates inject their own
short timeout.

**Gate 54** (`gate54_production_serve_timeout_unbounded`): (a) `run_trial_miner` with
no `serve_timeout` resolves a context timeout of `None`; (b) a configured 1 s timeout
is still honored (no-worker trial aborts at ~1 s). **Would have caught it:** pre-fix
the resolved default was 30.0, aborting valid long scans.

---

## Verification

- `PYTHONPATH=. python3 tests/test_s172_phase4_coordinator.py` → **54/54 green**
  (36 brief + gate 37 serve-path + gates 38–47 C2 + gate 43 rewrite + gates 48–54 C3),
  stable across 3 consecutive runs. gate 23 re-runs the Phase 0/1/2/3 subprocess
  non-regression.
- `PYTHONPATH=. python3 tests/test_s172_phase3_worker.py` → **17/17 green**.
- Pre-fix failure confirmed per defect via targeted reverts:
  D1→orphan file, D2→"20 exceeds bound 4", D3→`.result` TimeoutError (deadlock),
  D4→not reassigned, D5→not reassigned, D6→wedge (both gates hang), prod→`serve_timeout`
  30.0 ≠ None.

Next: Team Alpha adversarial re-review (orphan-write, queue RAM bound, deadlock
lifecycle, silent-socket timeout), then Team Beta. Do NOT commit/push/run WATCHER.
