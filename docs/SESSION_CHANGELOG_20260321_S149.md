# SESSION CHANGELOG — S149
**Date:** 2026-03-21  
**Commit range:** `c1c698d` → `0b8210d`  
**Status:** Run 1 active (fresh start, study `window_opt_1774109563`)

---

## Summary

S149 resolved the AMD 4-worker ceiling (S149-B fix validated live), discovered
and fixed the NPZ end-of-run data loss risk (per-trial checkpoint), diagnosed
the IPC serialization bottleneck on high-survivor passes, and obtained TB
approval for the slim_v1 fix (implementation in S150). Multiple cleanup and
correctness issues were also resolved including the WATCHER freshness skip,
manifest success_condition bug, and a stale Optuna study.

---

## Fixes Deployed

### 1. S149-B: Direct Device(gpu_id) — AMD 4-worker ceiling removed
**Commit:** `c1c698d`  
**Files:** `sieve_gpu_worker.py`, `persistent_worker_coordinator.py`  
**Patch:** `apply_s149b_device_gpu_id.py` (20/20 verified)

**Problem:** Workers were hardcoded to `cp.cuda.Device(0)` and coordinator
set `HIP_VISIBLE_DEVICES={gpu_id}` per worker. On AMD rigs, HIP masking is
unreliable for GPUs 4-7 — workers crashed on startup, causing a 4-worker
ceiling. Throughput was ~1,050 s/s per rig (1 worker active).

**Fix:** Remove `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` masking from
`_spawn_worker()`. Workers now see all GPUs. `sieve_gpu_worker.py` binds via
`cp.cuda.Device(gpu_id)` directly. `run_sieve_job(job, gpu_id)` takes explicit
parameter with mismatch assertion.

**Result:** All 3 RX 6600 rigs reached 8 active workers. Cluster throughput:
- Before: ~31,000 s/s (4 workers total across 3 rigs)
- After: ~1,990,000 s/s burst (24 AMD workers + 2 Zeus)
- **46x improvement**

**TB validation:** GPU 0 and GPU 7 spawn test passed on rrig6600. Preprod
soak 5/5 trials clean, 24/24 workers alive across all trials.

---

### 2. Manifest success_condition fix
**Commit:** `265889c` (via manifest edit)  
**File:** `agent_manifests/window_optimizer.json`

**Problem:** `success_condition` still listed `bidirectional_survivors.json`
which was replaced by NPZ in S145. WATCHER halted every run because the file
never exists.

**Fix:** `success_condition` updated to `['optimal_window_config.json']` only.

---

### 3. Per-trial NPZ checkpoint — eliminate end-of-run data loss
**Commits:** `5320108` (initial), `d7ae4d2` (wiring fix), `c39bba0` (order fix)  
**Files:** `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py`  
**Patches:** `apply_s149_npz_checkpoint.py`, `apply_s149_npz_checkpoint_fix2.py`

**Problem:** The NPZ accumulator (`bidirectional_survivors_all.npz`) was only
written at the end of a full 50-trial run. A crash at trial 49 would lose all
accumulated survivors from trials 1-48. This was discovered when Run 1 crashed
during Trial 7's hybrid reverse pass — 22,145 bidirectional survivors were lost.

**Fix:** `create_incremental_save_callback()` in `window_optimizer_bayesian.py`
now accepts a `survivor_accumulator` reference. After each trial that produces
survivors, the callback writes an atomic NPZ checkpoint:
- `bidirectional_survivors_all.npz` — merged with prior data, best-score-wins
- `bidirectional_survivors_binary.npz` — for Steps 2-6 consumption

Two bugs were found and fixed during implementation:
1. `_survivor_accumulator` was set on `WindowOptimizer` instance but read from
   `BayesianOptimization` instance — different objects. Fixed by setting on
   `strategy` (the `BayesianOptimization` instance).
2. `strategy._survivor_accumulator` was set before `strategy` was assigned
   from `strategy_map`. Fixed by moving assignment after `strategy_map.get()`.

**Behavior after fix:**
- Checkpoint writes after every trial with survivors (atomic, non-fatal)
- End-of-run write still runs (both paths active)
- Pruned/zero-survivor trials: no write (no-op)
- Crash at trial N-1: survivors from trials 1 through N-2 are safe on disk

---

### 4. IPC serialization bottleneck diagnosed + TB ruling obtained
**Commit:** `0b8210d`  
**Files:** `docs/IPC_SERIALIZATION_FIX_IMPLEMENTATION_GUIDE_S150.md`,
           `docs/TB_RULING_REQUEST_IPC_SERIALIZATION_S150.md`

**Problem:** On high-survivor passes (Pass 3/4 hybrid), workers return results
as lists of Python dicts — one dict per survivor with 6-7 fields. At 1,400
survivors per chunk × ~150 bytes/survivor = ~210KB JSON per chunk result.
Serialization + SSH write + Zeus parse takes several seconds per chunk. GPU
sits idle during this period. Pass 3/4 throughput: ~73,000 s/s vs ~2,000,000
s/s on Pass 1/2 — a 27x degradation.

**TB ruling:** Option A approved — slim_v1 parallel array payload:
```json
{
  "status": "ok",
  "job_id": "...",
  "result": {
    "format": "slim_v1",
    "seeds": [...],
    "match_rates": [...],
    "skip_sequences": [...],
    "strategy_ids": [...]
  }
}
```
Coordinator accepts both legacy and slim_v1 during rollout. Binary framing
(Option B) deferred. Shared memory (Option C) rejected for remote rigs.

**Implementation:** S150 P1. See implementation guide for full deploy sequence.

---

### 5. Optuna study management cleanup
- Old study DBs archived to `optuna_studies/archive/`
- Fresh Run 1 started with clean study `window_opt_1774109563`
- Run 1 survivors now safe per-trial via NPZ checkpoint

---

## Architecture Invariants Added S149

- **[S149-B]** Workers see all GPUs — no HIP/CUDA/ROCR visibility masking in spawner
- **[S149-B]** `Device(gpu_id)` direct selection in `sieve_gpu_worker.py`
- **[S149-B]** `run_sieve_job(job, gpu_id)` explicit parameter with mismatch assertion
- **[S149-CKPT]** NPZ accumulator written after every survivor-producing trial (atomic)
- **[S149-CKPT]** `strategy._survivor_accumulator` must be set AFTER `strategy_map.get()`
- **[S150-pending]** slim_v1 IPC format approved — implement before next production run

---

## Commit Log

| Commit | Description |
|---|---|
| `c1c698d` | fix(s149b): direct Device(gpu_id) — remove HIP/CUDA masking — 46x throughput gain |
| `265889c` | data(s149): run1 progress — 666 NPZ seeds, manifest fix, TB ruling docs |
| `5320108` | fix(s149): per-trial NPZ checkpoint — eliminate end-of-run data loss |
| `d7ae4d2` | fix(s149): NPZ checkpoint wiring — set on strategy not optimizer |
| `c39bba0` | fix(s149): NPZ checkpoint order fix — strategy assigned before _survivor_accumulator |
| `0b8210d` | docs(s149): IPC serialization fix guide + TB ruling — slim_v1 approved for S150 |

---

## Run 1 Status

| Item | Value |
|---|---|
| Study | `window_opt_1774109563` |
| Seed range | 660,000,000 → 1,733,741,824 |
| Trials | 50 (pruning enabled) |
| Completed trials | ~2 (as of session end) |
| NPZ accumulator | 666 seeds (pre-S149) — checkpoint active from trial 1 |
| Best config so far | W12/T0.30 (S148 calibration) |
| Worker pool | 8 per rig × 3 rigs + Zeus = 26 GPUs |
| Burst throughput | ~1,990,000 s/s (Pass 1/2) |

---

## S150 Priority List

1. **P0** — `--force-step` flag for WATCHER (freshness skip blocking every resume)
2. **P1** — slim_v1 IPC serialization fix (TB approved, implement before Run 1 completes)
3. **P2** — `sweep_run2.sh` with `enqueue_trial()` warm-start from Run 1 best params
4. **P3** — Measure actual GPU utilization during Pass 3/4 with 1s rocm-smi polling
5. **Backlog** — S110 root cleanup, sklearn warnings, CSV writer removal,
                  Chapter 13 wire-up, selfplay NN fix, walk-forward simulation
