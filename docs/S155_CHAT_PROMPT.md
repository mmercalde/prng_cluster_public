# S155 CHAT PROMPT
**Date:** 2026-03-23  
**Last commit:** `7182ab5` (+ docs commit pending)  
**Status:** Production blocked — OOM fix pending TB review

---

## State Coming In

rrig6600c (and all AMD rigs) crash with OOM during production runs. Root cause
confirmed via kern.log. Production is fully blocked.

**DO NOT LAUNCH ANY PRODUCTION RUN until CuPy pool fix is approved and deployed.**

---

## Root Cause (CONFIRMED)

`CUPY_CUDA_MEMORY_POOL_TYPE=none` (added S151) causes CuPy to mmap full 8GB
VRAM per worker at device init. 8 workers × 8GB = 64GB VA on a 7.7GB RAM machine
→ OOM killer fires.

kern.log evidence — 3 separate crashes, all identical:
```
Out of memory: Killed process XXXX (python) total-vm:41452828kB
```

---

## P0 — First Thing S155

### 1. Get TB ruling on PROPOSAL_S155_CUPY_POOL_FIX.md

TB must answer 4 questions before implementation:
1. Is `cp.get_default_memory_pool().set_limit()` effective on ROCm/HIP backend?
2. Does removing `CUPY_CUDA_MEMORY_POOL_TYPE=none` reintroduce the S151 pool race? (Note: cache race already fixed via per-worker CUPY_CACHE_DIR in S152)
3. Is 256MB pool limit appropriate? (working set ~42MB per job, 256MB = 6× headroom)
4. Should pool limit be configurable via `PRNG_CUPY_POOL_LIMIT_MB` env var?

### 2. After TB approval — implement fix

**File 1: `persistent_worker_coordinator.py`**  
Remove `CUPY_CUDA_MEMORY_POOL_TYPE=none` from ROCM_ENV_VARS (~line 97)

**File 2: `sieve_gpu_worker.py`**  
Add after GPU warmup in `run_worker()`:
```python
# [S155] Cap CuPy pool — prevents OOM. 256MB = 6× working set per worker.
cp.get_default_memory_pool().set_limit(256 * 1024 * 1024)
cp.get_default_pinned_memory_pool().set_limit(64 * 1024 * 1024)
```

### 3. Deploy and verify
- Apply patch → deploy to all rigs → 3-trial verification run
- Watch `free -h` on rrig6600c during run — RAM usage should stay flat
- If stable → restore 50-trial manifest and launch Run 1

---

## Cluster State

| Node | Status |
|------|--------|
| Zeus | Clean — no processes running |
| rrig6600 | Clean (1 zombie — harmless) |
| rrig6600b | Clean (1 zombie — harmless) |
| rrig6600c | Up, SSH responsive, no workers running |

| Item | Value |
|------|-------|
| Last commit | `7182ab5` |
| NPZ seeds | 676 |
| Coverage pointer | 660,000,000 |
| Run 1 | NOT complete — blocked on OOM fix |

---

## Key Commands

```bash
# Verify clean before starting
ssh rzeus "pgrep -f 'window_optimizer\|watcher_agent' | wc -l"
ssh rrig6600 "pgrep -f sieve_gpu_worker | wc -l"
ssh rrig6600b "pgrep -f sieve_gpu_worker | wc -l"
ssh rrig6600c "pgrep -f sieve_gpu_worker | wc -l"

# Deploy crash monitors (persistent log)
ssh rrig6600 "nohup ~/rig_crash_monitor.sh >> ~/rig_crash_monitor_persistent.log 2>&1 &"
ssh rrig6600b "nohup ~/rig_crash_monitor.sh >> ~/rig_crash_monitor_persistent.log 2>&1 &"
ssh rrig6600c "nohup ~/rig_crash_monitor.sh >> ~/rig_crash_monitor_persistent.log 2>&1 &"

# Launch (ONLY after OOM fix deployed)
ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
  python3 reset_coverage_s152.py && \
  rm -f optimal_window_config.json /tmp/agent_halt && \
  bash sweep_run1.sh"

# Monitor
ssh rzeus "tail -f ~/distributed_prng_analysis/logs/sweep_run1_production.log"

# RAM watch on rrig6600c
ssh rrig6600c "watch -n5 free -h"
```

---

## Docs to Review

- `docs/PROPOSAL_S155_CUPY_POOL_FIX.md` — TB proposal (read before starting)
- `docs/SESSION_CHANGELOG_20260323_S153.md` — diagnostic work (kernel args/IPC ruled out)
- `docs/SESSION_CHANGELOG_20260323_S154.md` — OOM confirmed, partial fix, real fix analysis
- `docs/TODO_MASTER_S154.md` — full priority backlog
