# SESSION_CHANGELOG 2026-07-18 — S172 Phase 4 (RANGE-MINER coordinator)

**Team:** Alpha (Claude, implements). **Authority for all binding rulings + rejections:**
Team Beta (binding rulings in `docs/TB_BINDING_RULINGS_S172_PHASE4.md`).
**Box:** VM 101 (`zeus-ubuntu`, 192.168.3.177), user `michael`, venv `~/venvs/torch`.
**HEAD at start:** `b8fda2f` = `6661b04` + 3 docs-only commits (verified
`git diff --name-only 6661b04 b8fda2f` = 4 `docs/` files only; no source drift).

Spec: `docs/S172_PHASE4_BRIEF.md` (rev-4, 36 gates) + `PROPOSAL_S172_RANGE_MINER_v1_4_5.md`
§3.A/§6.7/§12.3/§12.4/§15. Staged implementation per
`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE4.md`.

## Result

`PYTHONPATH=. python3 tests/test_s172_phase4_coordinator.py` → **37/37 checks green, exit 0**
(CPU-only, loopback, stubbed transfer + stubbed Phase5Sink). That is the **36 brief
Phase-4 gates PLUS gate 37** — the Team-Beta-required real-serve-path gate — a total of
**37 `_check` calls**; gate 23 additionally re-runs Phase 0/1/2/3 as a subprocess
non-regression check (all green, INCLUDING the Stage-0-patched Phase-3 resolver, whose
harness grew 14 → **17** gates).

> **rev-2 (2026-07-18, post-Beta serve-path rejection):** Team Beta REJECTED the initial
> submission on one release blocker — `run_trial_miner()` raised `NotImplementedError`
> when `_serve` was not injected and `_default_serve` was a stub that also raised. Beta
> ruled the coordinator's real serve path is the central Phase-4 deliverable. Corrected:
> `RangeMinerCoordinator.serve_trial()` is now a real framed-TCP server loop and the
> default; both `NotImplementedError` raises are deleted; gate 37 exercises the real path
> with two loopback workers over real `MinerFramedSocket` framing (incl. hybrid
> reassignment). See deviation #2 (rewritten) and the new file note below.

## Files changed (write-only; Michael commits after review)

| File | Action |
|---|---|
| `miner/range_miner_worker.py` | ONE surgical patch — Blocker-6 `ResidueResolver` (Stage 0). |
| `tests/test_s172_phase3_worker.py` | Phase-3 harness enlarged 14 → 17 (Beta-authorized). |
| `miner/range_miner_coordinator.py` | Fleshed out from the 63-line stub → full coordinator, incl. the real `serve_trial()` framed-TCP server (rev-2). |
| `tests/test_s172_phase4_coordinator.py` | New — the 36 brief gates + gate 37 (Beta serve-path, rev-2). |
| `tests/test_s172_phase1_scaffolding.py` | rev-2: gate 2 updated — it asserted `run_trial_miner` raises `NotImplementedError`, which Beta's binding ruling ordered deleted; it now verifies `run_trial_miner` is the real wired entrypoint (drives the coordinator + creates the trial via the injectable `_serve` seam). Flagged for review. |

## Both binding rulings — implemented as specified

**Ruling 1 (Blocker 6) — Option C (reject-on-absence AND reject-on-mismatch, both
non-retryable).** This is a **Phase-4-driven Phase-3 patch**: `ResidueResolver.resolve()`
(`range_miner_worker.py`) now, BEFORE any cache return and BEFORE `self._loader(...)`,
requires `dataset_sha256` (absent → `ResidueResolutionError`) and compares it against the
locally-computed hash (mismatch → `ResidueVerificationError`), reusing the existing
cache-key hash (file hashed once). Both inherit `ResidueError` → route to
`stripe_error(retryable=False)` at `range_miner_worker.py:1152-1154` (routing verified
unchanged post-patch). Plain `!=` (integrity id, not a secret). Phase-3 harness updated:
existing valid fixtures now supply `dataset_sha256`; added gate 15 (missing → non-retryable,
resolver + behavioral loopback), gate 16 (mismatch → non-retryable), gate 17 (a warm cache
cannot bypass a later mismatch). Narrowly limited to `dataset_sha256`; `residue_sha256`
stays mandatory on every production assignment.

**Ruling 2 (L7) — Option A (synchronous `abort_trial()`, no async ack).** `abort_trial()`
order: persist trial=aborted → fence/cancel all active assignments → **synchronous**
`Phase5Sink.abort_trial(event)` → ONLY on its successful return delete staged files +
release reservations. Retained-on-failure: if the sink raises/times out, the trial stays
terminally aborted, `TrialCommit` stays prohibited, staged files + reservations are
**RETAINED**, `abort_cleanup_status='failed'`, retried idempotently (idempotent by
`(event_id, run_id)`). No `TrialAbortAck` / async callback / abort-timeout state machine
added. Dispatch-thread requirement met via `submit_abort()` (a single-thread cleanup
executor) so the discharge never runs in the socket dispatch loop. Gate 34 confirms a
consumed shard's local file survives *while* the sync stub runs and is deleted only after
it returns; gate 28 confirms the retain-on-failure + idempotent retry.

## Per-stage summary

- **Stage 0** — Blocker-6 patch + enlarged Phase-3 harness. 17/17 green.
- **Stage 1** — durable SQLite ledger (sole-writer `_write_lock` + WAL, adapted from
  `zmq_sqlite_coordinator.py`; NO `MAX_ATTEMPTS`, NO one-row result table); shard-level
  ledger keyed `(run_id, stripe_id, attempt, sub_index)` (B1); state machine
  `pending→claimed→staging→done/failed`(+`cancelled`) (B5); macro-stripe partitioner (B7);
  L8 completion predicate. Gates 1-6, 35.
- **Stage 2** — connection-bound identity (Decision A); four-cap registration validation +
  quarantine, durable `workers` table (B7/L4); L1 stale-attempt fencing; spool-root
  restriction; L4 config wiring into `run_trial_miner`. Gates 17-19, 24, 29, 30.
- **Stage 3** — transfer adapter (Decision B, `delete_remote` only after local verify);
  byte/file reservations + high-water back-pressure (L2/§15); inline normalization to
  byte-identical `s172_substripe_v1` (B4, imports `build_substripe_payload_bytes`); L5
  async-task fencing via the immutable `(run,stripe,attempt,sub,staging_generation)` key;
  SC1 durable remote-delete; L8 failure-path cleanup (file removed FIRST, then capacity).
  Gates 13-16, 25-27, 31, 32, 36.
- **Stage 4** — Blocker-3 retry matrix (constant fail-closed; hybrid one-retry-to-a-
  DIFFERENT-worker + `phase_degraded`; `retryable=False` immediate; lease-expiry same
  policy); Blocker-2 attempt-scoped publish; L6 `Phase5Sink` + `event_id`-keyed idempotent
  ack; L3 whole-trial abort + `commit_trial`; durable `trials` table; mandatory
  dataset_sha256 + residue_sha256 assign-payload builder. Gates 7-12, 28, 33, 34.
- **Stage 5** — `run_trial_miner` wired to build+drive the coordinator (via injectable
  `_serve`); end-to-end gate 20 through a REAL worker daemon + REAL patched resolver
  (wrong `dataset_sha256` → `stripe_error(retryable=False)` → matrix row 4 fails trial);
  no-Phase-5-assembly (§3.A); coexistence (only the miner deliverables changed; PWC+ZMQ
  import unmodified); Phase 0/1/2/3 non-regression. Gates 20-23.

## Deviations / decisions flagged for review

1. **Blocker-2 publish timing (a correction, not a deviation).** The Stage-3 draft
   enqueued each shard on verify; Blocker 2 (brief lines 72-80) mandates publishing an
   attempt's manifests **only when whole** (all verified + StripeComplete + L8), and gate 7
   requires a failed attempt to have published nothing. Corrected: `_finalize_stage` now
   *holds* a verified shard (reserved); `publish_shard` fires in `finalize_stripe` at
   completion. Stage-3 gates 13/25/26/27 were updated to drive a completed stripe.
2. **Real default serve path — implemented per Beta's binding rejection (rev-2).** The
   initial submission deferred the live serve loop to Phase 6/7 and had `run_trial_miner`
   raise `NotImplementedError` when `_serve` was not injected. Team Beta REJECTED that:
   the coordinator's real serve path is the central Phase-4 deliverable. Now
   `RangeMinerCoordinator.serve_trial(context)` is a real framed-TCP server loop over the
   Phase-2 `MinerFramedSocket` protocol — bind/listen (ephemeral port selectable via a
   pre-bound `listen_sock`) → accept → `register_worker` (Decision A) → `assign_stripes`
   (B7) → **every** inbound message gated through `accept_stripe_message` (L1) before any
   ledger mutation → `record_*`/`stage_inline_shard`/`stage_remote_shard`/`finalize_stripe`/
   `handle_stripe_failure`/`process_lease_expiry` → `commit_trial` or (matrix) `abort_trial`
   — and it is the DEFAULT (`run_trial_miner` calls `coordinator.serve_trial` when no
   `_serve` is injected). Both `NotImplementedError` raises are DELETED. It wires the
   already-verified handlers together; it does not re-implement ledger/matrix/staging/
   abort/resolver logic (Beta did not contest those). Gate 37 drives the real default path
   with two loopback workers over real framed sockets and proves all six Beta points +
   hybrid reassignment to the DIFFERENT worker (`phase_degraded=True`). **Only the real
   worker FLEET / CT100 keys remain Phase-6/7** — a coordinator server binding to loopback
   workers is in scope and done.

   *Necessary consequence:* `test_s172_phase1_scaffolding.py` gate 2 asserted
   `run_trial_miner` raises `NotImplementedError` — exactly the raise Beta ordered deleted —
   so that one gate was updated to verify the new wired contract instead. Flagged for review.
3. **`remote_delete_attempts` counts every attempt including the successful one** (gate 31:
   stage-fail→1, retry-fail→2, retry-success→3); an already-deleted no-op retry does not
   increment. Chosen as the honest attempt count.

## Correction 2 (2026-07-18, post-Beta six-defect rejection)

Team Beta REJECTED the resubmission on SIX release-blocking defects in the serve path,
ledger, and production wiring (#1 and #2 reproduced dynamically). All six are fixed, each
with a NEW gate that fails on the pre-fix code and passes on the fixed code (harness now 47
`_check` calls: 36 brief + gate 37 + gates 38-47). No existing gate was weakened or deleted.
The accepted ledger/matrix/staging/abort/resolver logic was not redesigned.

1. **Stale attempt could delete the current attempt's file (Beta-reproduced).** `_staged_path`
   (and the temp path) are now ATTEMPT/GENERATION-PRIVATE — identity comes from the immutable
   task key `(run_id, stripe_id, attempt, sub_index, staging_generation)`, never the sha — so
   two attempts covering the same seed range can never collide on one file. The stale branch
   of `_finalize_stage` therefore removes only its own private path. **Gate 38** stages
   attempt 1, then drives attempt 0's stale finish, and asserts
   `attempt1_file_after_stale_finish is True` (Beta's exact probe).
2. **Duplicate results created duplicate reservations (Beta-reproduced).** `_serve_dispatch`
   now checks `record_substripe_result`'s return and drops a duplicate `(attempt, sub_index)`
   BEFORE staging; `reservations.event_id` gained a `UNIQUE` constraint (defense in depth) and
   `reserve()` returns None on the violation. **Gate 39** delivers the same result twice and
   asserts exactly ONE held reservation for the event_id.
3. **Connection-bound identity bypassed at dispatch.** `_serve_dispatch` now takes the
   RECEIVING socket's bound worker_id and rejects any message whose `worker_id` != that bound
   id BEFORE resolving a connection or touching the ledger (Decision A at dispatch). **Gate 40**
   sends a frame on worker A's real framed socket claiming worker B and asserts it is dropped
   with no ledger mutation.
4. **Synchronous staging blocked the dispatcher; failure policy incomplete.** A BOUNDED
   staging executor (separate from the abort-cleanup executor) now runs fetch/verify/rename +
   inline write/fsync OFF the dispatch loop. `staging_timeout` is enforced (bounded fetch);
   back-pressure POSTPONES + resumes (re-queue) instead of dropping; hash/staging/timeout
   failures route through `handle_stripe_failure` (right `retryable`); a spooled result with
   no transfer adapter fails the stripe explicitly. **Gates 41/42/43**: slow fetch doesn't
   block a second connection; a staging timeout is matrix-reassigned with zero leak;
   back-pressure postpones then resumes.
5. **Terminal trial state not mutually exclusive.** `mark_trial_aborted` and
   `mark_trial_committed` now transition ONLY from `state='running'` (committed and aborted are
   mutually exclusive); `TrialCommit` gained an immutable `event_id` + durable
   `commit_delivery_status` (deliver after the terminal decision, idempotent by event_id); the
   real terminal path routes abort through the OFF-dispatch `submit_abort` (`fail_trial` now
   waits on the cleanup-executor future). **Gates 44/45/46**: commit-then-abort refused (and
   reverse); abort runs on the `miner-cleanup` thread, not inline; duplicate commit delivered
   once.
6. **Production call mis-wired (run_id from config filename; params dropped).** `run_trial_miner`
   derives a UNIQUE run_id per trial (`{cfg_stem}_t{n}_{uuid8}`, never the filename) and
   resolves the workflow STAGES from `prng_base + test_both_modes` (all four §6.8 families when
   `test_both_modes`); staging_dir defaults from `miner_output_dir`. The real `_use_miner`
   integration call in `window_optimizer_integration_final.py` now propagates window params
   (window_size/sessions/offset), hybrid caps, staging settings, and a remote-reachable bind
   (`0.0.0.0`). **Gate 47** calls the production shape for two consecutive trials and asserts
   distinct run_ids + no stripe-ID/PK collision + resolved production values.

## Files changed (write-only; Michael commits after review)

| File | Action |
|---|---|
| `miner/range_miner_worker.py` | Blocker-6 patch (Stage 0) + rev label. |
| `tests/test_s172_phase3_worker.py` | 14 → 17 gates (Stage 0) + rev label. |
| `miner/range_miner_coordinator.py` | Full coordinator + real `serve_trial` + Correction-2 six-defect fixes. |
| `tests/test_s172_phase4_coordinator.py` | 47 checks (36 brief + gate 37 + gates 38-47). |
| `tests/test_s172_phase1_scaffolding.py` | gate 2 updated (Beta serve-path ruling). |
| `window_optimizer_integration_final.py` | Correction-2 Defect 6: real `_use_miner` wiring (inside the `use_range_miner` gate only; PWC/ZMQ paths untouched → coexistence holds). |

## Non-regression (gate 23 subprocess)

`test_prng_encoding.py`, `test_s172_phase1_scaffolding.py`, `test_s172_phase2_protocol.py`,
`test_s172_phase3_worker.py` (17 gates incl. the Stage-0 resolver) — all exit 0.

fallback parity: code=pending-commit (6 files uncommitted; one `git pull` from current once
Michael dual-pushes), env=ok (no new runtime deps — stdlib `sqlite3`/`concurrent.futures`/
`uuid` + the existing `miner` modules only).

## Next

STOP for Team Alpha file-vs-source review, then Team Beta binding review, then Michael
commits + dual-pushes. Phase 5 (NPZ writer / 22-array assembly / contract wall) remains
out of scope.
