# Team Alpha Review Record — S172 Phase 4 (RANGE-MINER coordinator) — REV 6
**Reviewer:** Team Alpha (lead dev)
**Date:** 2026-07-19
**Verdict:** PASS — ready for Team Beta binding re-review (Beta stated this round is at
approval threshold).
**Supersedes:** rev-5 (overload / heterogeneous-worker correction). Rev-6 covers
Correction 5 — the three real-resource-bound defects Beta reproduced against resubmission 4.
**Method:** file-vs-source, adversarial. Traced Beta's exact attack for each: a 70 MiB
remote spool vs a 60 MiB high-water, live `miner-fetch` thread accumulation past cap, and a
disconnected worker remaining in the eligible pool.

---

## The theme Beta identified (and it's fixed)

Two of the three prior failures were gates that checked BOOKKEEPING (registry entry count,
a static byte estimate) instead of the REAL RESOURCE (bytes on disk, live threads) — so the
gate passed while the real resource leaked. Every fix + gate below now asserts the real
resource: actual advertised `size_bytes`, live threads via `threading.enumerate()`, and the
actual eligible pool.

---

## Scope reviewed (Correction-5 delta)

| File | Change |
|---|---|
| `miner/range_miner_coordinator.py` | Actual-`size_bytes` admission guard; orphan-slot reservation BEFORE thread start; `_drop_conn` full identity eviction with fenced-replacement guard. |
| `tests/test_s172_phase4_coordinator.py` | Gate 59 REPLACED (live-thread count); gates 60, 61 added. Harness now 62 `_check` (61 gates + subprocess non-regression). |

Harness: **61/61 green from a CLEAN /tmp extraction of the archive** (verified by Claude
Code: build tar → extract to fresh dir → run suite there), Phase-3 17/17 from the same
extraction. Each new/replaced gate verified to FAIL on the pre-C5 (rev-5) code.

---

## Per-defect adversarial verification

**Defect 1 — large remote spools bypassed the oversized-attempt check.** The prior
`_attempt_footprint` used `INLINE_BYTE_LIMIT` (48 MiB) per file — valid only for inline
messages. `enqueue_staging` now applies the BYTE guard using the shard's ACTUAL advertised
`msg.size_bytes`: a single shard whose `size_bytes > staging_high_water_bytes`, or an attempt
whose accumulated advertised bytes exceed it, fails fast (non-retryable capacity error via
`_on_staging_failed`) and is never admitted — so it can never loop on `StagingBackPressure`.
Inline results keep the inline-ceiling bound; remote results use their real advertised size.
**Gate 60:** a 70 MiB remote result with a 60 MiB high-water → the stripe fails IMMEDIATELY,
`fetch_calls == []` (the fetch is never even attempted), no perpetual back-pressure. (Pre-fix:
admitted, then `.result()` dead-loops on TimeoutError.)

**Defect 2 — orphan-fetch registry was bounded but the actual threads were not.** The prior
code started the fetch thread BEFORE the cap check, so a refused registration left a live,
untracked thread. Now `_reserve_orphan_slot` prunes the registry to LIVE threads
(`t.is_alive()`) and admits only if the live count is below cap, and this reservation happens
BEFORE `th.start()` (coordinator.py: reserve at ~1902, `th.start()` at ~1908). If the budget
is exhausted, the thread is never started and the job fails with a `StagingError` capacity
error. A completed-in-time fetch releases its slot; a timed-out one keeps it (still alive →
correctly counted). **Gate 59 (replaced):** N = 7 hung fetches with cap = 2 → asserts the
number of LIVE `miner-fetch` threads counted via `threading.enumerate()` (filtered by name +
`is_alive()`) stays ≤ baseline + cap, and the excess jobs fail with a capacity error. This is
Beta's EXACT probe — it counts real threads, not `len(registry)` (the hole the old gate
missed).

**Defect 3 — disconnected workers remained eligible.** The prior `_drop_conn` removed the
socket from `fs_by_sock`/`worker_by_sock`/`fs_by_worker` but not from `wconn_by_worker`,
`self.connections`, or `registered` — the structures `_eligible()` is built from — so a
worker whose socket was gone still received new stripes. `_drop_conn` now evicts the worker_id
from ALL of `fs_by_worker`, `wconn_by_worker`, `self.connections`, and `registered`. The
eviction is guarded: it only clears the identity if THIS dropped socket is the one currently
bound to the worker_id (`fs is None or fs_by_worker.get(wid) is fs`), so a fenced replacement
that legitimately rebound the same worker_id to a DIFFERENT live socket is NOT evicted. All
serve-loop callers pass the extra structures. **Gate 61:** A and B register; A is dropped
BEFORE assignment; every new compatible stripe goes to B, none to A, and none is left
claimed-by-A-and-unsendable. (Pre-fix: A stayed in `wconn_by_worker`, run_s0 claimed by A and
never sendable.)

---

## Packaging (Beta's recurring execution blocker — resolved and verified)

Prior archives omitted committed deps (`utils/prng_encoding.py`,
`persistent_worker_coordinator.py`, etc.), so Beta couldn't run the full suite (Gate 22/23
and Phase-3 failed on imports, not code). This round the self-contained set is 33 files,
enumerated in the changelog, and Claude Code VERIFIED it by extracting the tar to a fresh
`/tmp` dir and running the suite there (61/61, 17/17) — not merely from the live repo. Note:
Gate 22's `git status` diff check is vacuous in a clean extraction (no `.git`), so it passes
trivially there; the archive verifies imports/execution, which is what Beta needs to run
everything. The full working-tree diff remains available in the live repo.

---

## Cross-cutting confirmations

- The accepted ledger / retry-matrix / resolver / dispatch logic (rev-1..rev-5) was NOT
  redesigned; these are three localized fixes (byte-guard source, orphan reserve-before-start,
  `_drop_conn` eviction).
- No existing gate weakened; gate 59 was REPLACED at Beta's direction (it counted registry
  entries, not live threads).
- Both bookkeeping-vs-real-resource holes Beta flagged are closed: D1 uses actual advertised
  bytes, D2 bounds live threads.

## Open items for Michael before commit (housekeeping, non-blocking)

- `python3_with_venv.sh` stays out of the Phase-4 commit.
- The COMMIT set is the files changed this session (`range_miner_coordinator.py`,
  `test_s172_phase4_coordinator.py`) plus the accumulated Phase-4 deliverables from prior
  corrections — the 33-file archive is for Beta's execution, NOT the commit set. Commit only
  the actual changed/new deliverables; the rest are already committed.
- Confirm the changelog files-table + fallback-parity line.

## Standing

Team **Alpha** pass — adversarial file-vs-source verification that all three real-resource
defects are fixed and gate-covered (actual bytes, live threads, eligible-pool eviction), with
gate 59 replaced to count live threads and the archive verified self-contained from a clean
extraction. NOT the binding gate. Sequence: **Team Beta binding re-review → Michael commits +
dual-pushes.** Beta indicated this round is at approval threshold.
