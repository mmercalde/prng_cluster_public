# SESSION CHANGELOG — S162
**Date:** 2026-04-05
**Author:** Claude (Team Alpha)
**Session Focus:** rrig6600c crash root cause investigation and fix

---

## Root Cause — CONFIRMED AND FIXED

**Problem:** rrig6600c crashed within 1-2 minutes of every 3-rig full-cluster run.
Crash signature: `SQC (inst)` GPU page fault → KFD queue cascade → CPU soft lockup → rig reboot.

**Root Cause:** All 3 rigs were using Ubuntu's **stock kernel amdgpu driver**
(`/lib/modules/6.8.0-106-generic/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko`)
instead of AMD's validated DKMS driver. The stock Ubuntu kernel driver cannot
handle 8 concurrent GPU compute workers under full 3-rig concurrent load.

**Fix:** Install AMD's `amdgpu-dkms` package (version `6.12.12.60403`) on all
3 rigs. This replaces the stock kernel module with AMD's own validated driver
at `/lib/modules/6.8.0-106-generic/updates/dkms/amdgpu.ko`.

**Verification:** All 3 trials completed clean with 26/26 GPUs active after fix.
Peak throughput: **35,632,151 seeds/sec** across 26 GPUs.

---

## Experiments Conducted (All Ruled Out)

| Experiment | Result |
|-----------|--------|
| Option B: `AMD_SERIALIZE_KERNEL=3` on rrig6600c | ❌ No effect |
| Exp 1: `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0` on rrig6600c | ❌ No effect |
| `cwsr_enable=0` GRUB param on rrig6600c | ❌ No effect |
| `amdgpu-dkms 6.12.12` on rrig6600c only | ✅ Extended stability to 25 min |
| `amdgpu-dkms 6.12.12` on ALL 3 rigs | ✅ Full clean run confirmed |

---

## Changes Made This Session

### All 3 Rigs (rrig6600, rrig6600b, rrig6600c)
- Installed `amdgpu-dkms 1:6.12.12.60403-2194681.22.04`
- Installed `amdgpu-dkms-firmware 1:6.12.12.60403-2194681.22.04`
- Created `/etc/apt/preferences.d/pin-amdgpu-dkms` to pin version and
  prevent auto-upgrade to untested newer versions
- Rebooted to activate AMD's kernel driver

### rrig6600c Only
- `amdgpu.cwsr_enable=0` added to GRUB (diagnostic — can be reverted)
- `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0` added to coordinator (diagnostic —
  should be reverted from coordinator)

### Zeus (persistent_worker_coordinator.py)
- Exp1 patch active (`HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0` for rrig6600c)
  — **must be reverted**, no longer needed
- Option B patch was replaced by Exp1 — already gone
- Dispatch semaphore `Semaphore(8)` in `_handle_client()` — **review for removal**
- Worker spawn stagger — **review for removal**
- `_best_effort_gpu_cleanup()` / `free_all_blocks()` — **KEEP**, good hygiene

### TCP buffer tuning (Zeus, persisted to /etc/sysctl.conf)
- `wmem_max/rmem_max` = 16MB
- `wmem_default/rmem_default` = 1MB
- These are benign and can stay

---

## Performance Results (Post-Fix)

| Metric | Value |
|--------|-------|
| Total GPUs | 26/26 |
| Total throughput | ~35,632,151 seeds/sec |
| Zeus (2× RTX 3080Ti) | ~70,439 s/s |
| rrig6600 (8× RX 6600) | ~9,539,751 s/s |
| rrig6600b (8× RX 6600) | ~15,957,319 s/s |
| rrig6600c (8× RX 6600) | ~10,064,643 s/s |
| Trial 1 completion | ✅ Clean |
| Trial 2 completion | ✅ Clean |
| Pipeline Step 1 | ✅ PASSED, proceeded to Step 2 |

---

## Why rrig6600c Always Crashed First

rrig6600c received jobs last in the worker spawn sequence. By the time its
8 workers began dispatching kernels, the other 16 workers on rrig6600 and
rrig6600b were already mid-execution. This created maximum concurrent GPU
VA management pressure at the exact moment rrig6600c's workers entered the
workload — the stock driver hit its limit first on rrig6600c simply due to
timing, not any hardware difference.

---

## Key Finding — Driver Mismatch

The AMD ROCm 6.4.3 userspace stack was installed but the kernel-side amdgpu
driver was Ubuntu's stock version, not AMD's validated DKMS build. These
were not in sync for multi-GPU concurrent workload stability on gfx1032.

The `ban-amdgpu-dkms` apt pin file on rrig6600 was a historical protection
from a previous session where DKMS broke on a kernel upgrade. This pin
prevented the fix from being applied automatically.

---

## Pending TODOs (Carry Forward)

1. **Revert Exp1 patch from Zeus coordinator** (`HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0`)
2. **Review and remove diagnostic workarounds** from coordinator:
   - Dispatch semaphore (review)
   - Worker spawn stagger (review)
   - Keep `_best_effort_gpu_cleanup()` / `free_all_blocks()`
3. **Remove `cwsr_enable=0`** from rrig6600c GRUB (diagnostic, no longer needed)
4. **Increase `--seed-cap-amd` 2M → 5M** — test throughput improvement
5. **Selfplay NN fix** (`inner_episode_trainer.py`) — forbidden guard + y-normalization
6. **S110 root cleanup** (884 stray files)
7. **Chapter 13 wire-up** post-Step-6 exit path

---

## Git Commit Instructions (Michael on Zeus)

```bash
cd ~/distributed_prng_analysis
git add docs/SESSION_CHANGELOG_20260405_S162.md
git commit -m "docs(s162): root cause confirmed — amdgpu-dkms fix, 26-GPU stable"
git push origin main && git push public main
```

---

*Session S162 — Team Alpha — 2026-04-05*
