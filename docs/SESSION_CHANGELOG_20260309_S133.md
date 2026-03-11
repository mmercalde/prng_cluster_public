# SESSION CHANGELOG — S133
**Date:** 2026-03-09/10
**Session:** S133 (includes S133-A and S133-B sub-sessions)
**Author:** Team Alpha (Claude)
**Status:** CLOSED — rig-6600c crash fixed, hybrid kernel root cause identified

---

## Summary

S133 resolved the rig-6600c persistent worker crash in two phases. S133-A fixed a semaphore regression in coordinator.py. S133-B identified the deeper root cause: `java_lcg_hybrid` kernel dispatch was sending strategy buffers to `sieve_gpu_worker.py` which lacked the hybrid kernel signature — crashing the worker process mid-pass.

---

## S133-A: Semaphore Regression Fix

### Problem
rig-6600c workers were dying during the forward sieve pass with `libamdhip64` segfaults. Pattern: all 4 rig-6600c workers alive at pool startup, dead by end of first chunk dispatch round.

### Root Cause
S130 persistent worker integration inadvertently re-introduced a semaphore acquisition path for the local Zeus sieve that blocked the persistent worker dispatch loop. Under concurrent chunk dispatch, Zeus local path acquired a semaphore that was never released when chunks completed, causing a deadlock that manifested as worker death timeouts on rig-6600c (the last worker to receive chunks in round-robin order).

### Fix Applied (`coordinator.py`)
Semaphore acquisition/release block moved outside the persistent worker branch. Persistent worker path now bypasses semaphore entirely (as originally designed in S130 Gate 4).

**Deployed to Zeus:** `ssh rzeus "..."` — pipeline re-test confirmed rig-6600c workers surviving pass 1.

---

## S133-B: Hybrid Kernel Signature Mismatch

### Problem
After semaphore fix, rig-6600c workers survived pass 1 (java_lcg) but died on pass 3 (java_lcg_hybrid). Workers on rigs 120 and 154 also affected. Pattern: all chunk results succeed for passes 1-2, all fail for passes 3-4.

### Root Cause
`persistent_worker_coordinator.py` dispatched `java_lcg_hybrid` and `java_lcg_hybrid_reverse` passes by sending a strategy list to `sieve_gpu_worker.py`. However, `sieve_gpu_worker.py`'s `run_sieve()` function only accepted `prng_type`, `seed_start`, `seed_count`, `dataset_path` — no `strategies` parameter. The worker raised `TypeError: run_sieve() got an unexpected keyword argument 'strategies'`, which closed the pipe and triggered empty-response handling, marking all 12 workers dead.

### Fix
Added `strategies` parameter support to `sieve_gpu_worker.py`'s `run_sieve()` dispatch. When `prng_type` ends in `_hybrid`, strategies are loaded from the passed list rather than the default single-strategy path.

**Deployed to all 3 rigs** (192.168.3.120, 192.168.3.154, 192.168.3.162).

---

## Z10×Z10×Z10 Kernel Gap Discovery

During S133-B investigation, Team Beta flagged that `sieve_gpu_worker.py` also lacks the Z10×Z10×Z10 digit kernel (added to `sieve_filter.py` in S119). This means digit feature scoring is not available via the persistent worker path — workers will silently produce lower-quality survivor scores when PRNG type requires digit features.

**Status:** Documented. TB proposal required before implementing. Carry forward as PENDING.

---

## Files Modified

| File | Change | Deployed To |
|------|--------|-------------|
| `coordinator.py` | Semaphore regression fix (S133-A) | Zeus |
| `sieve_gpu_worker.py` | Hybrid kernel strategies parameter (S133-B) | All 3 rigs |

---

## Pipeline State at Session Close

- rig-6600c: workers survive all 4 sieve passes ✅
- 14/14 chunks succeeding for passes 1-2 ✅
- Passes 3-4 (hybrid): workers survive but not yet clean-verified in production run
- Full end-to-end with persistent workers: not yet clean-completed

---

## Carry-Forward to S134

- Full end-to-end persistent worker clean run not yet confirmed
- Z10×Z10×Z10 kernel missing from `sieve_gpu_worker.py` (TB proposal needed)
- `window_trials` → `trials` manifest key stale
- Chapter 13 / selfplay WATCHER wire-up
- 200-trial Step 1 run

---

*Session S133 — 2026-03-09/10 — Team Alpha*
*rig-6600c crash resolved. Hybrid kernel gap closed. Workers survive all 4 passes.*
