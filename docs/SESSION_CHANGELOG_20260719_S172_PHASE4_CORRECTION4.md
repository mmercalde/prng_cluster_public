# SESSION CHANGELOG — 2026-07-19 — S172 Phase 4 CORRECTION 4 (overload + heterogeneous workers)

**Team Alpha (Claude) implementing. Team Beta is the binding approval authority.**
Instructions: `docs/CLAUDE_CODE_CORRECTION4_S172_PHASE4_OVERLOAD.md`.
Scope: fix FOUR Beta-reproduced release blockers — the coordinator dispatch thread
could still block/deadlock under staging overload, variant-aware scheduling could
strand stripes forever, connection identity could be rebound, and timed-out fetch
threads weren't tracked. The unifying rule: **the dispatch thread must NEVER block
or wait.** The accepted ledger / retry-matrix / resolver logic was **not** redesigned.

**Status: harness GREEN — Phase 4 = 59/59 (stable), Phase 3 = 17/17.** gate 49 was
REPLACED; every new/changed gate was verified to FAIL on the pre-C4 (Correction-3)
code. NOT committed/pushed; WATCHER not run.

Files changed: `miner/range_miner_coordinator.py`,
`tests/test_s172_phase4_coordinator.py` (gate 49 replaced + gates 55–59),
`window_optimizer_integration_final.py` (unchanged this round).
fallback parity: code=current, env=ok (no dependency change this session).

---

## gate 49 was REPLACED (recorded honestly)

Beta was right: the old gate 49 validated the WRONG behavior. It saturated the
staging semaphore, ran a producer that BLOCKED on `_staging_slots().acquire()`, and
asserted the block ("submitted == bound") as *success* — but the producer IS the
dispatch thread, so it was asserting that the dispatch thread blocks. The gate is
replaced by `gate49_dispatch_not_blocked_when_staging_saturated`, which proves the
opposite invariant with the REAL dispatch path: with every staging slot saturated, a
saturating `sub_stripe_result` delivered on the dispatch thread must NOT block — the
DIFFERENT worker's heartbeat delivered right after it is still processed (lease
renewed). On the C3 code the saturating result blocks on the semaphore and the
heartbeat is never reached (Beta's `third_dispatch_blocked=True`).

## Dispatch-never-blocks audit

Everything reachable from `_serve_dispatch` is nonblocking: `enqueue_staging` uses
`acquire(blocking=False)` only; the deferred-queue add is O(1) under a briefly-held
lock; `_pump_deferred` uses nonblocking slot acquires; the inbound `queue.get` (serve
loop, not dispatch) always has a timeout. The ONLY wait reachable from dispatch is
the accepted **terminal** `fail_trial → submit_abort().result()` — a bounded wait on
the off-dispatch cleanup executor at end-of-trial (unchanged, Beta-accepted L7 path),
never a steady-state/overload wait.

---

## Defect 1 — staging still blocked/deadlocked the dispatch thread (three sub-parts)

**1a — blocking semaphore acquire on the dispatch path.** `_submit_staging` did a
blocking `_staging_slots().acquire()`, so a full staging queue stalled the whole
dispatch thread. **Fix:** `enqueue_staging` is fully nonblocking —
`acquire(blocking=False)`; on failure the work is DEFERRED (bounded, 1c) and resumed
OFF the dispatch thread by `_pump_deferred` (invoked from a staging-completion
callback / matrix / ack). `_submit_with_slot` assumes the slot is already held and
NEVER acquires. **Gate 49** (above).

**1b — a single oversized attempt self-deadlocked.** `_try_admit_locked` always
admitted the first attempt even if its footprint exceeded high-water. **Fix:** removed
the escape hatch; `_attempt_exceeds_highwater` rejects an attempt whose whole
footprint can't fit the configured files OR bytes high-water — it **fails fast** as a
capacity/config error (`handle_stripe_failure`, non-retryable) instead of waiting
forever. **Gate 55** (`gate55_oversized_attempt_fails_fast`): a 2-substripe stripe
under `staging_high_water_files=1` → trial explicitly failed, stripe NOT in `staging`,
nothing admitted/deferred. Pre-fix: admitted then parked in `staging` forever.

**1c — the deferred queue was unbounded.** `_deferred` was a plain list. **Fix:**
`_defer_locked` enforces a COUNT bound (`staging_deferred_max`) and a retained-BYTES
bound (reusing `staging_high_water_bytes`, accounting only inline payloads — remote
results keep their payload on the spool/wire); exceeding either bound applies
dispatch back-pressure via the retry matrix (retryable) rather than retaining another
payload. **Gate 56** (`gate56_bounded_deferred_queue`): with one admitted attempt
holding the single high-water file, 12 un-admittable attempts are submitted →
`len(_deferred) ≤ 2` and retained bytes ≤ the byte bound; the excess is matrix-
back-pressured (hybrid reassign, trial runs on). Pre-fix: `deferred_len=12`.

## Defect 2 — variant-aware scheduling stranded stripes forever

**Fix:** `assign_stripes` now FILTERS the pool to variant-compatible workers BEFORE
round-robin (round-robin only across compatible workers); `serve_trial` FAILS THE
TRIAL EXPLICITLY when no eligible worker supports a stage's family — never leaves
stripes `pending` indefinitely (which, with the now-unbounded timeout, would hang).
**Gate 57** (`gate57_variant_filtered_scheduling`): (a) mixed pool (A=pcg32,
B=java_lcg), two java_lcg stripes → BOTH assigned to B, none pending; (b) a family no
worker supports (`mt19937`) → the serve trial aborts explicitly, not hangs. Pre-fix:
round-robin handed stripe 0 to the pcg32 worker → refused/pending forever.

## Defect 3 — connection identity could be rebound after registration

**Fix:** `_serve_register` returns a status and (a) REJECTS a REGISTER on an
already-bound socket claiming a different id (the socket stays bound to its original
id — no `[A,B]`), and (b) REJECTS a second socket registering an already-live
worker_id (the serve loop drops that socket) — one worker_id ↔ one live socket.
**Gate 58** (`gate58_one_socket_per_worker`): (a) re-register A→B on one socket →
`reject_rebind`, socket stays bound to A, no `[A,B]`; (b) a second socket for
worker A → `reject_dup_worker`, the original socket remains the sole mapping.
Pre-fix: the second register rebound the socket and left a stale `[A,B]` mapping.

## Defect 4 — permanently-hung fetch accumulated untracked daemon threads

**Fix:** `_call_fetch` PREFERS native adapter cancellation (passes `timeout` to
`fetch_remote` when the adapter accepts it, so the transfer itself aborts). When the
adapter can't cancel, a timed-out ('abandoned') `miner-fetch` thread is recorded in a
BOUNDED registry (`_orphan_fetch_threads`, cap `staging_orphan_fetch_max`); exceeding
the cap raises a capacity error instead of accumulating threads, and
`account_orphan_fetches` (called from `serve_trial`'s finally) prunes/joins/reports
them at shutdown. **Gate 59** (`gate59_orphan_fetch_threads_bounded`): N permanently-
blocked fetches → the registry stays ≤ cap, the cap breach surfaces a capacity error,
and shutdown accounting reports a bounded residual. Pre-fix: no tracking, no cap.

---

## New config (all injectable, L4)

`staging_deferred_max` (default 64), `staging_orphan_fetch_max` (default 8). The
deferred-bytes bound reuses `staging_high_water_bytes`.

## Verification

- `PYTHONPATH=. python3 tests/test_s172_phase4_coordinator.py` → **59/59 green**
  (36 brief + gate 37 + gates 38–47 C2 + 48–54 C3 + gate 49 replaced + gates 55–59 C4);
  gate 23 re-runs the Phase 0/1/2/3 subprocess non-regression.
- `PYTHONPATH=. python3 tests/test_s172_phase3_worker.py` → **17/17 green**.
- Pre-fix failure confirmed per defect: gates 49/57/58 fail against the full C3
  coordinator (block / stranded / rebind); gates 55/56/59 fail under targeted reverts
  (fail-fast disabled → not aborted; unbounded defer → `deferred_len=12`; no orphan
  tracking → no cap error).

## Delta-archive packaging note (for Beta)

Beta could not run the full suite from a delta-only archive last round because
committed deps were absent. For this resubmission INCLUDE (even though unchanged, so
Beta can execute everything): `miner/range_miner_protocol.py`, `prng_registry.py`, and
the Phase 0/1/2 harnesses `tests/test_prng_encoding.py`,
`tests/test_s172_phase1_scaffolding.py`, `tests/test_s172_phase2_protocol.py`. This is
packaging only — no new edits to those files.

Next: Team Alpha adversarial re-review (saturated-slots dispatch liveness, oversized
fail-fast, deferred bound, variant strand, socket rebind, thread accumulation), then
Team Beta. Do NOT commit/push/run WATCHER.
