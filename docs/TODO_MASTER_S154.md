# TODO MASTER — S154
**Generated:** 2026-03-23 (end of S153/S154)  
**Last commit:** `7182ab5`

---

## 🔴 P0 — Fix OOM Before Any Production Run

### THE REAL FIX — CuPy Memory Pool (NOT YET IMPLEMENTED)

**Root cause confirmed:** `CUPY_CUDA_MEMORY_POOL_TYPE=none` causes CuPy to
mmap full 8GB VRAM per worker into process VA space at device init.
8 workers × 8GB = 64GB VA on a 7.7GB RAM machine → OOM killer.

**Required fix (2 parts):**

1. Remove `CUPY_CUDA_MEMORY_POOL_TYPE=none` from `persistent_worker_coordinator.py`
   ROCM_ENV_VARS list (line ~97)

2. Add pool limit inside `run_worker()` in `sieve_gpu_worker.py` after GPU warmup:
   ```python
   # [S154-fix] Cap CuPy pool to prevent OOM — 256MB per worker
   # (6× working set: 2M seeds × ~40 bytes = ~40MB per job)
   cp.get_default_memory_pool().set_limit(256 * 1024 * 1024)
   cp.get_default_pinned_memory_pool().set_limit(64 * 1024 * 1024)
   ```

**⚠️ REQUIRES TEAM BETA REVIEW** — reverses S151 architectural decision
(`CUPY_CUDA_MEMORY_POOL_TYPE=none` was added S151 for race condition fix).
TB must confirm pool limit approach is safe before implementation.

**Memory math:**
- Per job: ~40MB GPU (2M seeds × 40 bytes)
- Pool limit: 256MB = 6× headroom
- 8 workers × 256MB = 2GB total → fits in 7.7GB RAM

### Additional OOM Mitigations

2. **Disable syncthing on all rigs** — syncthing invoked OOM killer at 10:09
   on rrig6600c (unrelated to sieve workers but competes for RAM):
   ```bash
   ssh rrig6600 "systemctl --user stop syncthing; systemctl --user disable syncthing"
   ssh rrig6600b "systemctl --user stop syncthing; systemctl --user disable syncthing"
   ssh rrig6600c "systemctl --user stop syncthing; systemctl --user disable syncthing"
   ```

3. **Increase swap on all rigs** from 2GB to 8GB as safety net:
   ```bash
   # On each rig:
   sudo swapoff -a
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **Add vm.overcommit_memory=2 to sysctl** — prevents kernel from
   over-promising VA space it can't back:
   ```bash
   ssh rrig6600c "sudo sysctl vm.overcommit_memory=2"
   ```
   Note: test carefully — may cause other allocations to fail.

---

## 🔴 P1 — After OOM Fixed

5. **Run 1 fresh launch** — reset coverage to 660M, 3 trials
   ```bash
   ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
     python3 reset_coverage_s152.py && \
     rm -f optimal_window_config.json /tmp/agent_halt && \
     bash sweep_run1.sh"
   ```

6. **Deploy crash monitors with persistent logging before any launch**:
   ```bash
   ssh rrig6600 "pkill -f rig_crash_monitor; nohup ~/rig_crash_monitor.sh >> ~/rig_crash_monitor_persistent.log 2>&1 &"
   ssh rrig6600b "pkill -f rig_crash_monitor; nohup ~/rig_crash_monitor.sh >> ~/rig_crash_monitor_persistent.log 2>&1 &"
   ssh rrig6600c "pkill -f rig_crash_monitor; nohup ~/rig_crash_monitor.sh >> ~/rig_crash_monitor_persistent.log 2>&1 &"
   ```

7. **java_lcg_reverse kernel arg mismatch** — code correctness bug
   (not crash cause, ROCm tolerates it silently):
   Patch: `apply_s153_java_lcg_reverse_args_fix.py` (already written, not applied)

---

## 🟡 P2 — After Run 1 Stable

8. **Write SESSION_CHANGELOG for S152** and upload to project database
   (exists on Zeus at `docs/SESSION_CHANGELOG_20260322_S152.md`)

9. **Respawn storm fix** — 537 threads simultaneously detect dead workers
   and queue up on per-node respawn lock. Need max_respawn_queue or
   quarantine-after-N-failures-in-window logic.

10. **sweep_run2.sh** — warm-start from Run 1 best params

11. **Add OOM monitoring to preflight** — check `free -h` available RAM
    on each rig before launching workers. Fail preflight if < 3GB free.

12. **Add syncthing check to preflight** — verify it's not running on rigs.

---

## 🟢 Backlog (unchanged from S152)

13. S110 root cleanup (884 files in project root)
14. sklearn warnings in Step 5
15. Remove dead CSV writer from `coordinator.py`
16. Regression diagnostics gate → set to True
17. S103 Part 2
18. Chapter 13 wire-up: `dispatch_selfplay()`, `dispatch_learning_loop()`
19. Selfplay NN two-part fix
20. Walk-forward simulation
21. Remove legacy dict-list coordinator parser (after slim_v1 stable)
22. Circuit breaker v3 (TB-approved design pending)
23. Telegram alert on worker quarantine / rig reboot detection

---

## Current State

| Item | Value |
|------|-------|
| Last commit | `7182ab5` |
| NPZ seeds | 676 |
| Coverage | 660,000,000 |
| OOM status | Root cause confirmed — fix pending TB review |
| S154 fix | Deployed (partial — del GPU arrays, does not resolve root cause) |
| rrig6600c | Stable after reboot — OOM fix needed before next run |
| Crash monitor | Persistent log deployed on all rigs |
| Syncthing | Running on rrig6600c — needs disabling |
| Run 1 | NOT complete — blocked on OOM fix |
