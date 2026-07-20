# Team Alpha Review Record — S172 Phase 4 (RANGE-MINER coordinator) — REV 4
**Reviewer:** Team Alpha (lead dev)
**Date:** 2026-07-19
**Verdict:** PASS — ready for Team Beta binding re-review.
**Supersedes:** rev-3 (first six-defect correction). Rev-4 covers Correction 3 — the
six async-staging / socket defects Beta found on adversarial probing, plus the
production-timeout fix.
**Method:** file-vs-source, adversarial. Each defect was traced against the delivered
source along Beta's specific attack (orphan late-write, unbounded-queue RAM, capacity
deadlock lifecycle, reconcile→matrix routing, hybrid hash-mismatch retry, silent/partial
socket), not the happy path.

---

## Scope reviewed (Correction-3 delta)

| File | Change |
|---|---|
| `miner/range_miner_coordinator.py` | Tracked fetch-timeout cleanup; bounded staging semaphore; stripe-level admission + deferred resume; `finalize_stripe` matrix routing; hash-mismatch `retryable=True`; per-connection reader threads + read deadline; serve_timeout default `None`. |
| `tests/test_s172_phase4_coordinator.py` | Gate 43 REWRITTEN (real publish/ack lifecycle); gates 48–54 added. Harness now 55 `_check` (54 gates + subprocess non-regression). |
| `window_optimizer_integration_final.py` | Passes `serve_timeout` (None default) through the `_use_miner` path. |

Harness: **54/54 green, stable across 4 runs, exit 0.** Phase-3 still 17/17. Each new /
rewritten gate was verified to FAIL on pre-fix code via per-defect reverts.

---

## Per-defect adversarial verification

**Defect 1 — timed-out transfer orphans a late write.** `_fetch_with_timeout` runs the
fetch on a daemon thread; a shared lock + `state["phase"]` serializes abandon-vs-complete.
On timeout the caller marks `abandoned`; the fetch thread's OWN `finally` clause, whenever
it later lands, observes `abandoned` under the lock and removes its temp artifact. The temp
path is attempt/generation-private, so the cleanup can only remove this task's own file.
The cleanup fires whenever the late write completes, not just at the timeout instant.
**Gate 48:** fetch completes 1s after a 0.3s timeout → staging dir is empty, no reservation
(Beta's orphan probe).

**Defect 2 — unbounded staging queue → RAM.** `_submit_staging` acquires one slot on a
`BoundedSemaphore(staging_workers + staging_queue_depth)` BEFORE the payload is retained /
submitted; the slot releases when the job finishes. Retained payloads (queued + active) are
capped, not just active threads. **Gate 49:** 20 tasks with bound 4 → retained ≤ 4 (Beta's
probe showed 20 > 4 pre-fix).

**Defect 3 — attempt-scoped publish deadlock; gate 43 was invalid.** `_try_admit_locked`
admits an attempt only if its WHOLE footprint (`need_files`, `need_files ×
INLINE_BYTE_LIMIT`) fits under both high-water limits alongside already-committed attempts;
the first attempt is always admitted (a legitimately-large attempt then waits nonblocking
rather than being starved). A sub-stripe whose attempt can't be admitted is DEFERRED to a
bounded `_deferred` list (not the executor's queue) and resumes via `_pump_deferred` when
`_release_admission` frees capacity. **Gate 43 rewrite (Beta-flagged):** the old gate
manually acked a shard before its attempt was published — impossible in the real lifecycle.
The rewrite admits attempt A (footprint fits), DEFERS attempt B, runs the real
`record_stripe_complete` → `finalize_stripe` → publish, then acks A's PUBLISHED manifests
via the real `ack_by_event_id`, which frees capacity so B's deferred sub-stripes resume and
stage. No manual ack of an unpublished shard anywhere; the deadlock is broken by admission,
not by the test cheating.

**Defect 4 — definitive reconcile failure never entered the matrix.** `finalize_stripe`
now distinguishes "incomplete, still waiting" from "complete-but-invalid": when
`substripes_match and not reconciled` (all shards present, accounting doesn't reconcile) and
an `eligible_provider` is supplied, it calls `handle_stripe_failure(...)` exactly once
(phase-specific matrix). Incomplete-but-consistent staging keeps waiting. **Gate 50:** a
hybrid stripe with seed-sum 29≠30 → reassigned to attempt 1, not left in `staging`.

**Defect 5 — hash mismatch marked non-retryable (contradicted a ruling).** The async
handler's `except StagingHashMismatch:` now uses `retryable=True`, feeding the phase matrix
— constant phases still fail closed, hybrid gets its one retry. **Gates 51a/51b:** hybrid
hash mismatch → reassign to attempt 1 on a different worker (`phase_degraded=True`);
constant hash mismatch → fail closed.

**Defect 6 — silent/partial client wedges the server.** Each accepted connection gets its
own `_conn_reader_loop` thread feeding the coordinator, plus a per-connection
`serve_read_deadline`; the serve/timeout loop never blocks on any one connection's `recv`,
and a connection that never completes a frame is dropped on its deadline. **Gates 52/53:** a
silent client before a valid worker → the serve timeout still fires and the trial aborts (no
wedge); a partial-frame client → does not block other connections' registration/dispatch and
is dropped on its read deadline.

**Production timeout.** `serve_timeout` defaults to `None` (unbounded) in both
`run_trial_miner` and the serve context; the loop enforces a timeout only when one is
configured. The `_use_miner` integration passes `serve_timeout` (None by default) through.
**Gate 54:** no-arg call → context timeout is None (no 30s abort); a configured timeout is
still honored (gates rely on this).

---

## Documented bound (honestly flagged, not a gap)

The admission byte-estimate uses `INLINE_BYTE_LIMIT` per file. For remote shards larger than
the 48 MiB inline ceiling this can under-count bytes — but the FILES constraint (default 512)
is exact and is the binding real-world limit (512 files vs 16 GiB). This is a conservative
documented bound on the admission estimate, NOT a change to the accepted reservation logic
and NOT a reintroduction of the deadlock (admission still guarantees the file footprint fits
before admitting).

---

## Cross-cutting confirmations

- The accepted ledger / retry-matrix / resolver logic (rev-1..rev-3) was NOT redesigned;
  these fixes are localized to fetch-timeout cleanup, the staging semaphore, the admission
  scheduler, `finalize_stripe` routing, the hash-mismatch flag, the socket reader threads,
  and the timeout default.
- No existing gate was weakened or deleted; gate 43 was rewritten to the REAL lifecycle at
  Beta's direction.
- Every new/rewritten gate was verified to fail on pre-fix code (per-defect reverts).

## Open items for Michael before commit (housekeeping, non-blocking)

- `python3_with_venv.sh` stays out of the Phase-4 commit (own commit later).
- Confirm the changelog's files-table + fallback-parity line count the current file set.
- Working-prompt / review docs in `docs/` — commit for the trail or leave out (Alpha's lean
  is out).

## Standing

Team **Alpha** pass — adversarial file-vs-source verification that all six async-staging /
socket defects and the production-timeout issue are fixed and gate-covered, with gate 43
rewritten to the real publish/ack lifecycle. NOT the binding gate. Sequence: **Team Beta
binding re-review → Michael commits + dual-pushes.** Given Beta reproduced these with
dynamic probes, gates 48–54 and the rewritten 43 now encode those exact attack shapes.
