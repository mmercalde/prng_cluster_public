# SESSION CHANGELOG — S163-KARG (Part 2)
**Date:** 2026-04-19
**Session:** S163-KARG (continuation)
**Focus:** DPM Pin Service, Skip-Dedup Fix, NPZ Vectorization, Worker Reconnect Fix, Production Runs
**Author:** Claude (Team Alpha Lead Dev)
**Status:** CLOSED
**HEAD:** `884b424`

---

## 🏆 Session Achievements

1. Deployed `amdgpu-dpm-pin.service` — all 24 AMD GPUs pinned to `manual` DPM at boot
2. Fixed 2.5h post-trial CPU bottleneck (skip-dedup fix)
3. Vectorized NPZ accumulator merge (lexsort + searchsorted)
4. Fixed `_superseded` NameError in NPZ accumulator path
5. Fixed worker reconnect timeout — workers now survive rig reboots (5min retry window)
6. Identified and fixed `node_allowlist` not propagating into `PersistentWorkerCoordinator`
7. Scrapped `n_parallel=2` — concept was wrong (was splitting GPU cluster, not CPU)
8. Deployed `reset_seed_coverage.py` utility
9. rrig6600 card0 PCIe link identified as bad (`Unknown 63`) — physical reseat needed
10. Production run 2 completed: 2,851 bidirectional survivors accumulated

---

## Fixes This Session

### Fix 1 — `[S163-KARG-DEDUP]` Skip fwd/rev dedup when summary-only
**Commit:** `c7f00a6`
- Root cause: 2.5h CPU burn deduplicating 2.8M Python dicts after each trial
- Fix: Skip forward/reverse dedup when `>100K` survivors (summary-only JSON path)
- Bidirectional always deduped — load-bearing path unchanged
- Harness A: 4/4 dedup branch tests PASS

### Fix 2 — `[S163-KARG]` Vectorized NPZ accumulator merge
**Commit:** `c7f00a6`
- `deduplicate_survivors()`: numpy lexsort replaces O(N²) Python dict loop
- NPZ merge: searchsorted replaces O(N) dict lookup
- Schema backfill for missing fields in older prior NPZ schemas
- Harness A: 22/22 PASS on Zeus

### Fix 3 — `[S163-KARG-FIX1]` n_parallel flag inheritance
**Commit:** `c4698c8`
- 8-hop propagation chain for transport/runtime flags
- Post-construction attribute assignment for child coordinator
- Harness B: 9/9 PASS on Zeus
- NOTE: n_parallel=2 scrapped immediately after — concept was wrong

### Fix 4 — `[S163-KARG-PWC]` node_allowlist propagated into PersistentWorkerCoordinator
**Commit:** `c7f00a6`
- Root cause: both P0 and P1 read all 4 nodes from config, competing for same workers
- Fix: filter nodes in `_load_config()` before S156 cleanup/worker launch
- TB-approved: filter before S156, not after

### Fix 5 — `[S163-KARG-PORT]` Pre-fork TCP port cleanup
**Commit:** `c7f00a6`
- `fuser -k 5600/tcp` and `fuser -k 5601/tcp` before forking partition processes
- Prevents zombie TCP sockets from blocking next run

### Fix 6 — `[S163-KARG-KILL]` Pre-fork worker kill
**Commit:** `c7f00a6`
- `pkill -9 -f pwc_worker_service` on all rigs before fork
- Prevents stale workers from connecting to wrong partition port

### Fix 7 — n_parallel reverted to 1
**Commit:** `c7f00a6`
- `n_parallel=2` scrapped — was splitting GPU cluster, not Zeus CPU
- Original bottleneck (post-trial CPU) already fixed by skip-dedup fix
- `agent_manifests/window_optimizer.json`: n_parallel reset to 1

### Fix 8 — Worker reconnect timeout increase
**Commit:** `884b424`
- `RECONNECT_DELAY_S`: 2.0 → 5.0s
- `_MAX_REFUSED`: 5 → 60 attempts
- Total retry window: 10s → 5 minutes
- Workers now survive rig reboots and reconnect to coordinator

### Fix 9 — `_superseded` NameError in NPZ accumulator
**Applied this session (not yet committed separately)**
- `_superseded_count = len(_superseded)` → `int(_superseded_mask.sum()) if _prior_count > 0 else 0`
- Prevents fallback to `convert_survivors_to_binary.py` on runs with survivors

---

## Infrastructure

### DPM Pin Service
- `amdgpu-dpm-pin.service` deployed to all 3 rigs
- Pins all 24 AMD GPUs to `manual` DPM + sclk state 1 + mclk state 3 at boot
- `ExecStartPre=/bin/sleep 10` — waits for GPU sysfs ready
- `gpu-enum-heal.service` removed (stale 12-GPU config)
- `rocm-perf-auto` cron removed (was overriding DPM pin)
- GDM disabled + multi-user.target permanent on all 3 rigs

### rrig6600b GRUB Recovery
- Drive removed → fsck via ser8 USB adapter (clean)
- chroot GRUB reinstall → `update-initramfs -u` → boots clean

### Zeus nvidia-compute-mode
- systemd service installed — DEFAULT mode persists across reboots

### rrig6600 card0 PCIe Issue
- `0000:03:00.0` shows `current_link_speed: Unknown 63`
- All other GPUs show `5.0 GT/s PCIe` (correct for Gen1 riser)
- Root cause: bad riser cable or PCIe slot on card0
- Fix: physical reseat — PENDING

### reset_seed_coverage.py
- Deployed to Zeus `~/distributed_prng_analysis/`
- Resets `exhaustive_progress` table in `prng_analysis.db`
- Usage: `python3 reset_seed_coverage.py java_lcg`

---

## Production Runs This Session

| Run | Seed Start | Trials | GPUs | Bidirectional | Notes |
|-----|-----------|--------|------|---------------|-------|
| s163_karg_production_1433 | 0 | 5 | 26 | 0 | Wrong seed range, dedup not tested |
| s163_karg_production2_1508 | 1,073,741,824 | 5 | 10-26 | 2,851 | rrig6600/c crashed Trial 3, skip-dedup + NPZ fix confirmed working |

**Accumulated NPZ total:** 20,914 + 2,851 = 23,765 seeds

---

## Key Learnings

- `n_parallel=2` was conceptually wrong — was splitting GPU cluster instead of parallelizing Zeus CPU post-processing
- Post-trial CPU bottleneck was already solved by skip-dedup fix (2.5h → seconds)
- `_MAX_REFUSED=5` (10s retry) was too short to survive rig reboots (~60-90s)
- rrig6600 card0 has bad PCIe link — intermittent crash source, needs physical reseat
- DPM pin service + GFXOFF=0 + snd-power.conf = stable 26 GPU operation when hardware is healthy
- `crash_forensic_daemon.py` runs on ser8, not Zeus
- Coverage tracker uses `exhaustive_progress` table in `prng_analysis.db` (not a JSON file)

---

## Git State

| Ref | Commit | Description |
|-----|--------|-------------|
| HEAD | `884b424` | Worker reconnect timeout fix |
| Previous | `c7f00a6` | n_parallel=1 + port/worker cleanup + node_allowlist |
| Previous | `c4698c8` | Fix 1 + Fix 2 + Harness A/B |

---

## Pending for Next Session

1. **Reseat rrig6600 card0 PCIe riser** — fix `Unknown 63` link speed
2. **Launch clean production run** — all 26 GPUs, 100K seed cap, 5 trials
3. **Run Step 2** — Scorer Meta-Optimizer with 23,765 accumulated survivors
4. **Fix crash_forensic_daemon.py** — 3 bugs (startup false-DOWN, log-caching, dmesg empty)
5. **Web dashboard auto-start** — wire into watcher startup
6. **TB proposals needed:**
   - Zeus TCP worker path (remove `_is_localhost` bypass)
   - TCP-PWC job pre-fetch in `pwc_worker_service.py`

---

*Session S163-KARG Part 2 — 2026-04-19 — Team Alpha*
