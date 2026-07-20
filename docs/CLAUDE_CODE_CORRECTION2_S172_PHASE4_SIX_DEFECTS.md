# Claude Code Instructions — S172 Phase 4 CORRECTION 2 (Beta: six serve-path defects)

**From:** Team Alpha lead
**For:** Claude Code on VM 101, `~/distributed_prng_analysis`, user `michael`
**Date:** 2026-07-18
**Status:** Team Beta REJECTED the resubmission — SIX release-blocking defects (two
reproduced dynamically). This corrects all six. Do NOT start Phase 5. Do NOT commit
or push.

---

## Read first

The serve path itself is accepted; these are correctness defects in it + the ledger +
the production wiring. Beta reproduced #1 and #2 with live probes. These defects exist
because the 37-gate harness tests single-attempt, happy-path, in-process flows — the
new gates below MUST exercise the adversarial cases (two attempts sharing state,
duplicate messages, cross-socket spoofing, blocking transfers, terminal-state races,
consecutive production-shaped trials). A fix without a gate that would have *caught*
the original defect is not accepted.

Each defect gets: (a) the fix, (b) a new gate that fails on the OLD code and passes on
the fixed code. Implement in the order below; they build on each other (#1 changes the
path scheme that #2's reservation uniqueness also touches).

Do NOT weaken or delete any existing gate to make these pass. If an existing gate
conflicts with a fix, STOP and report it — do not silently edit it.

---

## Defect 1 — stale attempt can delete the current attempt's file (Beta-reproduced)

**Root cause (two parts):**
- `_staged_path(stripe_id, sub_index, sha256)` omits `run_id`, `attempt`,
  `staging_generation`. A re-dispatched attempt covering the same seed range yields
  the same sha → the SAME path. Two attempts collide on one file.
- `_finalize_stage` RENAMES the temp file onto `staged_path` (the
  `_write_bytes_atomic`/`_atomic_replace` call) BEFORE the L5 fence check. A stale
  attempt-0 callback materializes onto the shared path (clobbering attempt 1), THEN
  detects stale, THEN `_fail_and_release(..., staged_path)` deletes that shared path —
  destroying attempt 1's live file and leaving attempt 1's ledger pointing at nothing.

**Fix:**
1. Make staged AND temp paths attempt/generation-private. `_staged_path` (and the temp
   path) must include `run_id`, `attempt`, and `staging_generation` in the filename (or
   a per-attempt subdirectory). No two distinct immutable task keys
   `(run_id, stripe_id, attempt, sub_index, staging_generation)` may ever resolve to
   the same path. Keep sha in the name if you like, but identity must come from the key.
2. A stale callback in `_finalize_stage` may remove ONLY a path uniquely owned by its
   own task key. Because the path is now attempt/generation-private, the existing
   `_fail_and_release(..., task.staged_path)` becomes safe — but ALSO re-verify the
   fence semantics: the stale branch must never touch a path that another live attempt
   could own. (Renaming before the fence is now acceptable ONLY because the target path
   is private to this task; confirm that invariant in a comment and the gate.)

**Gate (must fail on old code):** two attempts, same stripe/sub_index/seed range (same
sha). Drive attempt-0's `_finalize_stage` to completion AFTER attempt 1 has staged its
file. Assert: attempt-1's staged file still exists after the stale attempt-0 finish;
attempt-1's ledger path is intact and on disk; the stale finish returns `stale` and
removed only its OWN (private) path. This is Beta's exact reproduction —
`attempt1_file_after_stale_finish` must be True.

---

## Defect 2 — duplicate results create duplicate reservations (Beta-reproduced)

**Root cause:** `_serve_dispatch` calls `record_substripe_result(...)` (which correctly
returns False for a duplicate `(attempt, sub_index)`) but IGNORES the return value and
stages the duplicate anyway → two held reservations for one logical shard / event_id →
leak or wrong-resource release.

**Fix:**
```python
inserted = self.ledger.record_substripe_result(...)
if not inserted:
    logger.warning("duplicate sub-stripe result %s/%s dropped", msg.stripe_id, msg.sub_index)
    return
```
before ANY staging. AND add a `UNIQUE` constraint on `reservations.event_id` (defense in
depth — a second reserve for the same event_id must fail at the DB, not just be avoided
by the guard). Make `reserve()` tolerate/handle the uniqueness violation cleanly (return
None or the existing reservation, not crash).

**Gate (must fail on old code):** deliver the same `SubStripeResultMessage` twice through
`_serve_dispatch`. Assert exactly ONE held reservation for that event_id after the
duplicate; assert the second `record_substripe_result` returned False and no second
stage occurred. Beta's probe: `held_reservations_after_duplicate` must be 1.

---

## Defect 3 — connection-bound identity bypassed at dispatch

**Root cause:** the socket→worker_id binding lives in `worker_by_sock`, but
`_serve_dispatch(msg, run_id, wconn_by_worker)` is called WITHOUT the receiving socket's
bound identity. It resolves the connection from `msg.worker_id` alone, so worker A's
socket can send a message claiming worker B's id and the L1 check passes against B's
connection. Violates Decision A (binding connection identity).

**Fix:** pass the receiving socket's bound worker_id into `_serve_dispatch` and reject
any message whose `msg.worker_id` != the socket's bound identity BEFORE resolving the
connection or touching the ledger. The bound identity (from `worker_by_sock`), not the
message field, is authoritative.

**Gate (must fail on old code):** register two workers on two real framed sockets. From
worker A's socket, send a `SubStripeResultMessage` with `worker_id` = B. Assert it is
rejected (logged/dropped, no ledger mutation, no reservation, no stage) — the spoof does
not reach B's connection.

---

## Defect 4 — synchronous remote staging blocks the dispatcher; failure policy incomplete

**Root cause:** the only executor is the abort-cleanup one. `_serve_dispatch` calls
`stage_remote_shard` → `fetch_remote()` synchronously in the receive loop — a large
transfer blocks registration, heartbeats, results, dispatch (exactly what S175
prohibits). Also: `staging_timeout` never enforced; back-pressure logged-and-dropped
instead of postponed; hash/staging failures never routed into `handle_stripe_failure`;
spooled result silently ignored when `transfer is None`; inline write+fsync also run in
the dispatch loop.

**Fix:**
- Add a BOUNDED staging executor/queue (separate from the abort-cleanup executor).
  Remote fetch + verify + rename runs there, not in the receive loop. The dispatch loop
  enqueues a staging task and returns immediately.
- Enforce `staging_timeout`: a staging task exceeding it is failed and its cleanup runs
  (remove file first, release reservation — the L8 order), and the stripe is routed
  through the phase-specific matrix.
- Back-pressure (reserve returns None) must POSTPONE and resume (re-queue), not abandon
  the result.
- Hash-mismatch / staging failure must call `handle_stripe_failure(...)` with the right
  `retryable` (hash mismatch on advertised bytes → treat per spec; a transient fetch
  failure → retryable) so the matrix governs it — not a bare log.
- A spooled result with `transfer is None` is a configuration error → fail the
  stripe/trial explicitly, not silent ignore.
- Inline write + fsync also move off the dispatch loop (into the staging executor).

**Gate(s):** (a) a slow `fetch_remote` (injected sleep) must NOT block a concurrent
heartbeat/register/result on another connection — assert the dispatcher stays responsive
(e.g. a second worker's message is processed while the first's fetch is in flight).
(b) a staging task that exceeds `staging_timeout` is failed and routed through the matrix
(stripe reassigned or trial failed per phase), with the file removed and reservation
released (zero leak). (c) back-pressure postpones then resumes: with a tiny high-water,
a result that can't reserve is re-tried and eventually staged, not dropped.

---

## Defect 5 — terminal trial state not mutually exclusive

**Root cause:** `mark_trial_aborted` uses `WHERE run_id=? AND state != 'aborted'` — a
COMMITTED trial matches and can be flipped to aborted. Plus: `TrialCommit` has no
immutable event_id / durable delivery status; `commit_trial` persists committed before
calling the sink; the real path calls `abort_trial` synchronously via
`handle_stripe_failure` instead of the off-dispatch `submit_abort`.

**Fix:**
- Terminal transitions ONLY from `state='running'`: `mark_trial_aborted` →
  `WHERE run_id=? AND state='running'`; likewise `mark_trial_committed` →
  `WHERE run_id=? AND state='running'`. A committed trial can never become aborted and
  vice-versa; both are terminal and mutually exclusive.
- Give `TrialCommit` an immutable `event_id` (e.g. `{run_id}:commit`) and a durable
  commit-delivery status column, mirroring the abort event. Sink delivery idempotent by
  that event id.
- Order commit like abort: decide terminal state, then deliver to the sink, with durable
  status so a sink failure is retryable and doesn't leave an inconsistent state. (Match
  the L7 pattern you already have for abort.)
- The real dispatch/terminal path must use the OFF-dispatch abort route (`submit_abort`
  onto the cleanup executor), not a synchronous `abort_trial` call inside the receive
  loop. `serve_trial`'s terminal-accounting `fail_trial` and the matrix's terminal
  failures must go through `submit_abort` (and the loop waits on the future) so the
  synchronous discharge never runs on the dispatcher.

**Gate(s):** (a) commit a trial, then attempt to abort it → abort is refused, state stays
committed; and the reverse (abort then attempt commit → refused, already have gate 8 for
part of this, extend it). (b) the terminal abort in the real serve path runs off the
dispatch thread (assert it went through the executor, not inline). (c) commit-sink
idempotency: duplicate commit delivery is a no-op by event_id.

---

## Defect 6 — production call is mis-wired (run_id from config filename; params dropped)

**Root cause:** `run_id = str(coordinator_cfg)` — the config FILENAME becomes the run_id,
so every optimizer trial reuses e.g. `distributed_config.json`, and stripe IDs repeat →
PK collisions on trial 2. The real caller
(`window_optimizer_integration_final.py:_use_miner` gate, matching
`run_trial_persistent`'s call shape) also does not pass workflow phase, concrete family,
window params (window_size/sessions/offset), hybrid caps, staging settings, miner
host/port. Consequences: defaults to workflow Phase 1, window_size 1, sessions/offset
lost, staging_dir None even when miner_output_dir given, bind 127.0.0.1 (remote rigs
can't connect), test_both_modes doesn't drive the four families. Gate 37 hid this by
passing custom kwargs the real caller doesn't.

**Fix:**
- Derive a UNIQUE run_id per trial — e.g. `f"{study_or_cfg_stem}_t{trial_number}_{short_uuid}"`
  — NEVER the raw config filename. Stripe IDs derive from that unique run_id.
- Wire the ACTUAL integration call in `window_optimizer_integration_final.py` (the
  `_use_miner` / miner-gate path that mirrors `run_trial_persistent`): propagate the
  resolved `WindowConfig` (window_size, sessions, offset), the concrete family /
  `test_both_modes` families, workflow phase, hybrid caps, staging settings
  (staging_dir defaulting from miner_output_dir when set), and miner host/port. Default
  bind must be reachable by remote rigs (e.g. `0.0.0.0`), not `127.0.0.1`, in production
  config — keep loopback only for tests.
- `test_both_modes` must drive the four required families (per the brief), not a single
  java_lcg.

**Gate:** call `run_trial_miner` with the EXACT production call shape (the argument set
the real `_use_miner` caller passes — no test-only kwargs) for TWO consecutive trials.
Assert: distinct run_ids; no stripe-ID/PK collision across the two trials; workflow
phase / window params / staging_dir / bind address are the resolved production values,
not the defaults. If wiring the real integration file, run its relevant path or a focused
harness that constructs the same call.

---

## Verify + report

- `PYTHONPATH=. python3 tests/test_s172_phase4_coordinator.py` → all existing gates +
  the ~8-10 new defect gates green, exit 0. Each new gate must be one that FAILS on the
  pre-fix code (state that you checked this — ideally by running the new gate against a
  stashed copy of the old function, or by reasoning explicitly why it would have failed).
- `PYTHONPATH=. python3 tests/test_s172_phase3_worker.py` → still 17/17.
- If you touch `window_optimizer_integration_final.py`, that is a NEW changed file —
  note it; its coexistence with PWC/ZMQ must hold (the miner path is behind the
  `use_range_miner` gate).
- Update the changelog: a "Correction 2" section documenting all six fixes + their gates,
  and fix the two carried-over nits (header "three flagged decisions"; "5 files" — now
  possibly 6 if the integration file changed).

Report per defect: the fix, the new gate, and why that gate would have caught the
original. Then STOP. Team Alpha re-reviews at source — this time tracing the adversarial
case for each, not the happy path — then Team Beta binding re-review. Do NOT commit/push.
