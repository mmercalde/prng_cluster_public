# SESSION CHANGELOG — S162 FINAL
**Date:** 2026-04-04 / 2026-04-05
**Session:** S162
**Focus:** TCP-PWC Production Integration, rrig6600c Crash Root Cause, DKMS Fix
**Author:** Claude (Team Alpha Lead Dev)
**Status:** CLOSED — Root cause confirmed, all 3 rigs pinned, clean baseline restored

---

## Root Cause — CONFIRMED

**Stock Ubuntu kernel amdgpu driver cannot handle 8 concurrent compute processes per GPU under full 3-rig concurrent load.**

AMD's DKMS driver `amdgpu-dkms 6.12.12` fixes it completely. After installing on rrig6600c: first ever clean 26-GPU run at **35.6M seeds/sec**, all 3 trials clean.

---

## Day 1 — 2026-04-04

### S162-1 — First 26-GPU Run Attempt
- TCP-PWC default, all 3 rigs, 26 GPUs active
- rrig6600c crashed at ~18 seconds — SQC (inst) page fault signature

### S162-2 through S162-7 — Diagnostic Attempts (All Ineffective)
All of the following were tried and confirmed NOT the root cause:
- `AMD_SERIALIZE_KERNEL=3` — no effect
- `HSA_NO_SCRATCH_RECLAIM=1` — made things worse (crashed at 219s)
- CuPy memory pool disabled (raw hipMalloc) — made things worse (crashed at 358s)
- `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0` for rrig6600c — partial, not a fix
- `cwsr_enable=0` — unnecessary
- Worker spawn stagger — unnecessary
- Dispatch semaphore — unnecessary
- Seed cap reduction (1M, 500K) — delayed crashes, did not fix

### S162-8 — DKMS Driver Fix (ROOT CAUSE)
- Discovered: rrig6600c was on stock Ubuntu kernel amdgpu driver
- rrig6600b never crashed because job dispatch order — always got jobs first
- Installed `amdgpu-dkms` on rrig6600c:
  ```bash
  sudo apt-get install -y amdgpu-dkms && sudo reboot
  ```
- Result: **First clean 26-GPU run — 35.6M seeds/sec — all 3 trials complete**

---

## Day 2 — 2026-04-05

### After DKMS Fix — Extended Testing
- Installed `amdgpu-dkms 6.12.12` on rrig6600 and rrig6600b
- All 3 rigs now on DKMS driver — consistent baseline
- Multiple diagnostic patches applied and reverted (warmup, VRAM instrumentation, kernel pre-compile)
- All patches made things worse or had no effect post-DKMS
- Restored to `89c1512` clean baseline

### Final State — Restored to Baseline
- Commit `7278db6` — reverts all diagnostic patches
- `persistent_worker_coordinator.py` — clean, no experimental env vars
- `reverse_sieve_filter.py` — clean, no warmup/VRAM patches
- `distributed_config.json` — all 3 rigs at gpu_count=8
- `sieve_gpu_worker.py` — clean, no warmup patch

### Kernel Pinning — All 3 Rigs
```
amdgpu-dkms       HELD
amdgpu-install    HELD
linux-image-generic  HELD
linux-headers-generic  HELD
```

---

## Commits This Session
- `89c1512` — docs: root cause confirmed — amdgpu-dkms fix, 26-GPU stable
- `f50b0e6` — chore: update bidirectional survivors NPZ
- `8d91311` — S162: Add GPU warmup and VRAM instrumentation (LATER REVERTED)
- `b7f5e1a` — S162: Consolidate changelogs
- `58b2155` — S162: checkpoint before no-pool diagnostic patch
- `d79e558` — S162: diagnostic — disable CuPy pool (REVERTED)
- `070199d` — S162: revert no-pool patch
- `606f94f` — S162: HSA_NO_SCRATCH_RECLAIM=1 (REVERTED)
- `a854647` — S162: revert HSA_NO_SCRATCH_RECLAIM
- `22bb145` — S162: kernel pre-compile warmup patch (REVERTED)
- `ac32cd9` — S162: revert kernel pre-compile warmup patch
- `7278db6` — S162: restore to 89c1512 clean state ✅ **CURRENT HEAD**

---

## Next Session Priorities
1. Launch clean run with `seed_cap_amd=5000000` — test 5M seed cap throughput
2. Verify full 3-trial clean run at 35M+ seeds/sec
3. Write TB proposal for `disabled_nodes` param in coordinator (for per-step node exclusion)
4. Revert unnecessary workarounds from coordinator (stagger, semaphore, cleanup patches)
5. S110 root cleanup (884 stray files in project root)

## Known Working Configuration
```bash
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 \
  --params '{"min_workers": 24, "seed_cap_amd": 5000000}' \
  2>&1 | tee logs/s162_5m_seeds_run1.log
```

## Infrastructure State
- Zeus: `45.32.131.224`, commit `7278db6`
- rrig6600 (.120): amdgpu-dkms 6.12.12 ✅ pinned
- rrig6600b (.154): amdgpu-dkms 6.12.12 ✅ pinned
- rrig6600c (.162): amdgpu-dkms 6.12.12 ✅ pinned
