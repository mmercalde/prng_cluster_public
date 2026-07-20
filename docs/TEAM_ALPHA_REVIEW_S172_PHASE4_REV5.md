# Team Alpha Review Record — S172 Phase 4 (RANGE-MINER coordinator) — REV 5
**Reviewer:** Team Alpha (lead dev)
**Date:** 2026-07-19
**Verdict:** PASS — ready for Team Beta binding re-review.
**Supersedes:** rev-4 (async-staging / socket correction). Rev-5 covers Correction 4 —
the four overload / heterogeneous-worker freezes Beta reproduced against the delivered
coordinator.
**Method:** file-vs-source, adversarial. Each defect was traced against source along
Beta's specific attack: dispatch-thread liveness under staging saturation, oversized-
attempt fail-fast, bounded deferred queue, variant strand, socket rebind, and hung-fetch
thread accumulation.

---

## Scope reviewed (Correction-4 delta)

| File | Change |
|---|---|
| `miner/range_miner_coordinator.py` | Nonblocking `enqueue_staging`; fail-fast oversized-attempt admission; bounded `_deferred` (count + bytes); variant-filtered `assign_stripes`; `_serve_register` rebind/dup guards; bounded orphan-fetch registry. |
| `tests/test_s172_phase4_coordinator.py` | Gate 49 REPLACED (dispatch-liveness); gates 55–59 added. Harness now 60 `_check` (59 gates + subprocess non-regression). |
| `window_optimizer_integration_final.py` | unchanged from rev-4 (serve_timeout wiring). |

Harness: **59/59 green, stable across 4 runs, exit 0.** Phase-3 still 17/17. Each new /
replaced gate verified to FAIL on the pre-C4 (rev-4) code via targeted reverts.

---

## Per-defect adversarial verification

**Defect 1a — staging blocked the dispatch thread.** `enqueue_staging` is now fully
nonblocking: under a briefly-held `_admission_lock` it checks fail-fast, tries admission,
and attempts a NONBLOCKING `_staging_slots().acquire(blocking=False)`; if no slot, it
defers (bounded) — the dispatch thread returns immediately in every branch (submit /
defer / failfast / backpressure). Slot release and `_pump_deferred` run in the staging
executor's done-callback (off the dispatch thread). Audit of every wait reachable from
`_serve_dispatch`: the only blocking wait is the accepted terminal
`fail_trial → submit_abort().result()` (a bounded end-of-trial wait on the off-dispatch
cleanup executor), never a steady-state/overload wait. **Gate 49 (replaced):** the OLD
gate blocked the producer (which IS the dispatch thread) and called it success — Beta's
exact objection. The rewrite saturates both staging slots with blocked fetches, delivers
a saturating result on the dispatch thread, then a DIFFERENT worker's heartbeat, and
asserts the heartbeat is processed (lease renewed) — proving dispatch did not block.

**Defect 1b — oversized attempt waited forever.** `_attempt_exceeds_highwater` returns a
reason if the attempt's whole footprint exceeds high-water FILES or BYTES; `enqueue_staging`
then fails fast (non-retryable capacity error via `_on_staging_failed`), never admitting it
to wait. The prior "always admit the first attempt" escape hatch is removed. **Gate 55:** a
2-substripe stripe under `staging_high_water_files=1` → trial fails explicitly, NOT parked
in `staging` (pre-fix: admitted then hangs).

**Defect 1c — deferred queue unbounded.** `_defer_locked` enforces BOTH a count cap
(`staging_deferred_max=64`) and a retained-bytes cap (`staging_high_water_bytes`); on
overflow it returns False and `enqueue_staging` applies dispatch-level back-pressure via
the retry matrix (retryable) rather than retaining another payload. **Gate 56:** 12
un-admittable attempts against a bound of 2 → `len(_deferred) ≤ 2` (pre-fix: 12).

**Defect 2 — variant scheduling stranded stripes.** `assign_stripes` now builds
`compatible = [w for w in workers if can_assign_variant(w, family)]` BEFORE the round-robin
and assigns `compatible[i % len(compatible)]`. If no compatible (eligible, non-quarantined)
worker exists, the stripe is refused with a reason that `serve_trial` turns into an EXPLICIT
trial failure — never an indefinite strand. **Gate 57:** mixed pool (A=pcg32, B=java_lcg),
two java_lcg stripes → BOTH assigned to B; a family with no compatible worker → trial
aborts (pre-fix: stripe 0 pending forever, and with the now-unbounded timeout the trial
hangs).

**Defect 3 — connection identity could be rebound.** `_serve_register` returns explicit
statuses: `reject_rebind` (a REGISTER on an already-bound socket claiming a DIFFERENT id →
socket stays bound to the original), `reject_dup_worker` (the worker_id is already live on
another socket → duplicate rejected), and `ok` (new, or an idempotent re-send of the SAME
id). No `[A,B]` rebind, no two sockets sharing one worker_id. **Gate 58:** A→B on one socket
→ rebind rejected (stays A); a second socket for A → `reject_dup_worker` (pre-fix: socket
rebound, `fs_by_worker=[A,B]`).

**Defect 4 — hung fetch accumulated threads.** `_call_fetch` prefers a native adapter
timeout; if the adapter can't cancel, a timed-out ('abandoned') fetch thread is recorded in
a BOUNDED `_orphan_fetch_threads` registry that prunes finished threads on every add and
raises a capacity error when `staging_orphan_fetch_max=8` is exceeded — no unbounded thread
growth across a soak. `account_orphan_fetches` runs from `serve_trial`'s finally for
shutdown accounting. **Gate 59:** N permanently-blocked fetches → registry size ≤ cap,
cap breach → capacity error (pre-fix: untracked, no cap).

---

## Packaging (Beta's prior blocker on execution)

Beta could not run the full 54-gate suite from the delta-only archive because committed
deps weren't included. THIS resubmission archive includes the unchanged committed deps —
`miner/range_miner_protocol.py`, `prng_registry.py`, and the Phase 0/1/2 harnesses
(`tests/test_prng_encoding.py`, `tests/test_s172_phase1_scaffolding.py`,
`tests/test_s172_phase2_protocol.py`) — so Beta can execute the full suite. These are
packaging-only (already committed; no new edits).

---

## Cross-cutting confirmations

- Dispatch-never-blocks audit: no blocking `.acquire()`, unbounded `.get()`, or steady-
  state `.join()` reachable from `_serve_dispatch`; the only wait is the terminal
  off-dispatch abort future.
- The accepted ledger / retry-matrix / resolver logic (rev-1..rev-4) was NOT redesigned;
  fixes are localized to the staging admission/defer path, `assign_stripes` pool
  construction, `_serve_register` guards, and the orphan-fetch registry.
- No existing gate weakened; gate 49 was REPLACED at Beta's direction (it validated
  blocking the producer). Gate 19's variant-refusal path and the C2/C3 gates still pass.

## Open items for Michael before commit (housekeeping, non-blocking)

- `python3_with_venv.sh` stays out of the Phase-4 commit.
- Confirm the changelog files-table + fallback-parity line count the current file set.
- Working-prompt / review docs in `docs/` — commit for the trail or leave out.

## Standing

Team **Alpha** pass — adversarial file-vs-source verification that all four overload /
heterogeneous-worker defects are fixed and gate-covered, with gate 49 replaced to prove
dispatch liveness under saturation. NOT the binding gate. Sequence: **Team Beta binding
re-review → Michael commits + dual-pushes.** Gates 49(new)/55/56/57/58/59 encode Beta's
reproduced attack shapes; the archive now carries the deps needed to run the full suite.
