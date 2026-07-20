# Claude Code Instructions — S172 Phase 4 CORRECTION 4 (Beta: overload + heterogeneous-worker freezes)

**From:** Team Alpha lead
**For:** Claude Code on VM 101, `~/distributed_prng_analysis`, user `michael`
**Date:** 2026-07-19
**Status:** Team Beta REJECTED resubmission 3 — the prior seven defects are fixed, but four
NEW release blockers remain: staging can still block/deadlock the dispatch thread,
variant-aware scheduling can strand stripes forever, connection identity can be rebound, and
timed-out fetch threads aren't tracked. Do NOT start Phase 5. Do NOT commit/push.

---

## Read first

Beta reproduced all four. The unifying rule: **the coordinator dispatch thread must NEVER
block or wait.** Everything below serves that plus two correctness fixes (variant scheduling,
one-socket-per-worker). The ledger/retry/resolver logic is accepted — do not redesign it.

Same discipline: each defect gets a fix AND a gate that FAILS on current code, using the REAL
lifecycle (the dispatch thread is the producer — a gate that "blocks the producer and calls
it success" is testing the wrong thing; Beta explicitly called out gate 49 for this). Do not
weaken existing gates. If a fix conflicts with one, STOP and report.

---

## Defect 1 — staging still blocks/deadlocks the dispatch thread (three sub-parts)

**1a — blocking semaphore acquire on the dispatch path.** `_submit_staging` does
`self._staging_slots().acquire()` (blocking), and `_serve_dispatch` calls `enqueue_staging`
synchronously. A full staging queue therefore blocks the coordinator's dispatch thread —
heartbeats, completions, errors, other workers' results all stall. Beta reproduced
`third_dispatch_blocked=True`.

**Fix:** `enqueue_staging` (and everything it calls on the dispatch path) must be
NONBLOCKING. Use `acquire(blocking=False)`; if the slot can't be acquired, DEFER the work
(bounded — see 1c), do not block. The dispatch thread returns immediately in all cases. A
background pump (`_pump_deferred`, invoked off the dispatch thread — e.g. from a staging-
completion callback or a small timer) resubmits deferred work when a slot frees.

**Gate:** saturate all staging slots, then deliver another worker's heartbeat/result on a
different connection and assert the dispatch loop STILL processes it (not blocked). This
replaces gate 49's wrong assertion (which validated blocking the producer).

**1b — a single oversized attempt self-deadlocks.** `_try_admit_locked` ALWAYS admits the
first attempt even when its whole footprint exceeds high-water. Beta reproduced with
`staging_high_water_files=1` and a 2-substripe stripe: second shard can't stage → attempt
never completes → nothing published → Phase 5 can't ack → waits forever. The byte estimate
is also unsafe for high-survivor remote spools (assumes `INLINE_BYTE_LIMIT`/file; remote
spool files can exceed it → admit then deadlock on the byte high-water).

**Fix:** an attempt whose full footprint CANNOT fit within the configured high-water limits
(files OR bytes) must **fail fast** as a capacity/configuration error — call
`handle_stripe_failure` / fail the trial explicitly with a clear message — NOT be admitted to
wait forever. Remove the "always admit the first attempt" escape hatch; if even a single
attempt can't fit, that's a misconfiguration the operator must fix, surfaced immediately. For
the byte estimate: use the best available per-file size bound (if the stripe/attempt carries
expected remote sizes, use them; otherwise document that the files limit is the hard bound
and the byte check must not admit an attempt whose worst-case bytes exceed high-water).

**Gate:** an attempt whose footprint exceeds either high-water limit → fails explicitly
(trial failed / stripe errored with a capacity reason), NO perpetual `staging`, nothing left
waiting.

**1c — the deferred queue is unbounded.** `self._deferred` is a plain list; Beta submitted
100 incompatible attempts and got `deferred_len=100` against `configured_queue_bound=2`, each
retaining its full inline payload.

**Fix:** enforce a bound on `_deferred` — both a COUNT bound and a retained-BYTES bound
(config: e.g. `staging_deferred_max` count + reuse/extend the bytes high-water). When the
deferred bound is reached, apply dispatch-level back-pressure — do NOT retain more payloads.
The correct back-pressure for a miner is to NOT read the next result off that worker's socket
until capacity frees (so the payload stays on the wire / at the worker, not in coordinator
RAM), or to reject with a retryable error that the matrix handles. Pick one, keep it
nonblocking on the dispatch thread, and account retained bytes accurately.

**Gate:** hold one slot, submit far more incompatible attempts than the bound → assert
`len(_deferred)` and retained bytes never exceed the configured bounds.

---

## Defect 2 — variant-aware scheduling strands stripes forever (Beta-reproduced)

`assign_stripes` round-robins `workers[i % len(workers)]` and only THEN checks
`can_assign_variant`; on mismatch it leaves the stripe `pending` with no later path to a
compatible worker. Beta: worker A=pcg32, B=java_lcg, two java_lcg stripes →
`run_s0 pending, no worker` forever (and with the timeout now unbounded, the trial hangs).

**Fix:** filter the scheduling pool to workers that support the EXACT concrete variant BEFORE
round-robin. Round-robin only across compatible workers. If NO compatible (eligible,
non-quarantined) worker exists for a stripe's family, FAIL THE TRIAL EXPLICITLY with a clear
"no worker supports variant X" error — never leave stripes pending indefinitely.

**Gate:** mixed pool (A supports pcg32, B supports java_lcg), two java_lcg stripes → BOTH
stripes assigned to B (the compatible worker), none left pending. And: a family with no
compatible worker → trial fails explicitly, not hangs.

---

## Defect 3 — connection identity can be rebound after registration (Beta-reproduced)

`_serve_register` accepts additional `RegisterMessage` frames on an already-registered
socket. Beta registered one socket as A then B → socket rebinds to B while the old A mapping
lingers (`fs_by_worker=[A,B]`). A second socket can also register an already-connected
worker_id → multiple sockets share one logical identity.

**Fix:**
- Reject any `RegisterMessage` received on an ALREADY-BOUND socket (a socket registers exactly
  once; a second register on it is a protocol violation → drop/close).
- Enforce ONE live socket per worker_id: if a `worker_id` is already connected on another live
  socket, either reject the duplicate registration, OR atomically fence + close the OLD
  connection before accepting the replacement (pick one; if you fence-and-replace, the old
  socket's in-flight assignments must be handled via the existing attempt-fencing so no ledger
  corruption). No state where two sockets map to one worker_id or one socket maps to two.

**Gate:** (a) re-register on one socket (A then B) → the second register is rejected, socket
stays bound to A, no `[A,B]`. (b) a second socket registering an already-connected worker_id →
rejected (or old fenced+closed, exactly one live socket remains for that worker_id).

---

## Defect 4 — permanently-hung fetch accumulates untracked daemon threads (Beta caveat)

The late-write cleanup works when `fetch_remote` eventually RETURNS. A transfer that never
returns leaves its `miner-fetch` daemon thread alive forever — across a 50-trial soak,
repeated network hangs accumulate threads.

**Fix:** prefer a native `TransferAdapter` timeout/cancellation (pass a timeout the adapter
honors, so the fetch actually aborts). If the adapter can't cancel, keep every timed-out
fetch thread in a TRACKED registry (e.g. `self._orphan_fetch_threads`) with its task key, and
on coordinator shutdown join/account them; a bounded registry with a cap that, when exceeded,
surfaces a capacity error rather than silently growing. The goal: no unbounded thread growth
under repeated hangs.

**Gate:** simulate N permanently-blocked fetches → assert the live `miner-fetch` thread count
stays bounded (tracked registry size ≤ cap), not N-unbounded, and shutdown accounts for them.

---

## Note on the delta archive (for the resubmission)

Beta could NOT run the full 54-gate suite from the delta-only archive because committed deps
(`range_miner_protocol.py`, `prng_registry.py`, earlier harnesses) weren't included. For THIS
resubmission, include enough for Beta to run the full suite: add `miner/range_miner_protocol.py`,
`prng_registry.py`, and the Phase 0/1/2 harnesses (`tests/test_prng_encoding.py`,
`tests/test_s172_phase1_scaffolding.py`, `tests/test_s172_phase2_protocol.py`) to the archive —
even though they're unchanged — so Beta can execute everything. (They're already committed;
this is packaging only, not new edits.)

---

## Verify + report

- Full harness green, exit 0, including gate 49 REPLACED (dispatch-not-blocked) and new gates
  for 1b, 1c, 2, 3, 4. Each new/changed gate must FAIL on current code.
- Phase-3 still 17/17.
- Confirm the dispatch thread never blocks: no blocking `.acquire()`, `.join()` (except
  bounded shutdown), `.get()` without timeout, or unbounded wait anywhere reachable from
  `_serve_dispatch`.
- Update the changelog: "Correction 4" per defect + gate; record that gate 49 was replaced
  (it validated blocking the producer, which is the dispatch thread).

Report per defect: fix, gate, why it catches the original, and — for D1a — how the dispatch
path is now provably nonblocking. Then STOP. Team Alpha adversarial re-review (tracing:
saturated-slots dispatch liveness, oversized-attempt fail-fast, deferred bound, variant
strand, socket rebind, thread accumulation), then Team Beta. Do NOT commit/push.
