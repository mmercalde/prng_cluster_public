# SESSION CHANGELOG — S155
**Date:** 2026-03-24 to 2026-03-25  
**Commits:** `7182ab5` → `5d8d10d` → (M.2 SATA install) → `89c7597` → `8069692`  
**Status:** CLOSED — OOM root cause exhausted at Python/CuPy layer; hardware interventions attempted; run completed on 18-22 GPUs

---

## Summary

S155 was a multi-day extended debugging session continuing the rrig6600c OOM investigation from S154. Three software fix attempts were made (CuPy pool cap, CUPY_GPU_MEMORY_LIMIT, ROCR_VISIBLE_DEVICES isolation), all failing to change the `total-vm:41452828kB` crash signature. Physical hardware interventions were then attempted: M.2 SATA drive replacement (eliminated BIOS e820 fragmentation hole at `0x37700000`) and CPU upgrade (i5-8400T → i5-9400). Neither resolved the OOM. Session concluded with rrig6600c running at 4 workers as stable operating point, and a 3-trial production run completed on 22 GPUs.

---

## OOM Investigation — Complete Failure History

By end of S155, every theory had been tested and ruled out:

| Session | Theory | Result |
|---------|--------|--------|
| S151 | MaxSessions 5 in sshd | Fixed — not the OOM cause |
| S151 | slim_v1 | Reverted — not the crash cause |
| S152/S153 | java_lcg_reverse kernel arg mismatch | ROCm tolerates it — not crash cause |
| S152/S153 | IPC path | Stage 9 passed — not crash cause |
| S153/S154 | GPU array accumulation per job | Fixed with `del` — identical 41GB VM, not cause |
| S154/S155 | CuPy memory pool cap (256MB) | Deployed TB-approved fix — OOM persists |
| S155 | CUPY_GPU_MEMORY_LIMIT env var | Added to ROCM_ENV_VARS — OOM persists |
| S155 | ROCR_VISIBLE_DEVICES isolation | Restored per-worker GPU isolation — OOM persists |
| S155 | M.2 SATA adapter e820 fragmentation | Replaced with Samsung 860 EVO — e820 cleaner, OOM persists |
| S155 | CPU upgrade (i5-8400T → i5-9400) | Installed — improved throughput, OOM persists |

The `total-vm:41452828kB` signature was identical across ALL crash events regardless of fix applied. The VA bloat is at the ROCm/amdgpu driver level, below Python/CuPy control.

---

## Fixes Deployed This Session

### Fix 1 — CuPy Memory Pool Cap (TB-Approved)
**Commits:** Multiple rounds — final commit `5d8d10d`  
**Files:** `persistent_worker_coordinator.py`, `sieve_gpu_worker.py`

- Removed `CUPY_CUDA_MEMORY_POOL_TYPE=none` from ROCM_ENV_VARS
- Added `cp.get_default_memory_pool().set_limit(256 * 1024 * 1024)` inside worker
- Added `CUPY_GPU_MEMORY_LIMIT=268435456` and `PRNG_CUPY_POOL_LIMIT_MB=256` to ROCM_ENV_VARS
- Required 3 rounds of TB review fixing: sequencing, pinned pool API, remote env propagation

**Result:** OOM persists — no change to crash signature

### Fix 2 — ROCR_VISIBLE_DEVICES Restoration
**File:** `persistent_worker_coordinator.py`, `sieve_gpu_worker.py`

Restored `ROCR_VISIBLE_DEVICES={gpu_id}` in `_spawn_worker()`. Theory: without per-GPU
isolation each worker maps all 8 GPU apertures at HSA init.

**Result:** OOM persists

### Fix 3 — Hybrid IPC Key Fix (unrelated to OOM, correct fix)
Fixed `job["hybrid"]` key mismatch in IPC path — was using broken `prng_type`/`skip_mode`
keys. Separate correctness fix, not related to OOM.

---

## Hardware Interventions

### M.2 SATA Drive Replacement
- Replaced axGear M.2 SATA adapter (suspected BIOS e820 hole cause) with Samsung 860 EVO M.2
- Pre-swap e820: fragmented hole at `0x0000000037700000-0x0000000037700fff`
- Post-swap e820: hole gone, cleaner memory map
- **Result:** OOM persists — e820 fragmentation was not the cause

### CPU Upgrade
- Replaced Intel i5-8400T (1.70GHz, 35W) with Intel i5-9400 (2.90GHz, 65W)
- Eliminates CPU timing differential vs rrig6600/b
- Improves rrig6600c throughput parity
- **Result:** OOM persists

---

## Production Run Status

After exhausting OOM investigation, rrig6600c set to `gpu_count=4` as stable operating point.

3-trial production run completed:
- 22 GPUs (Zeus 2 + rrig6600 8 + rrig6600b 8 + rrig6600c 4)
- All 3 trials completed
- rrig6600c stable throughout at 4 workers
- NPZ accumulator: 676 seeds (no new survivors — window configs produced 0 bidirectional)

---

## rrig6600c Stable Operating Point

`gpu_count=4` is the confirmed stable point on rrig6600c. 8 workers always OOMs.
Root cause of why rrig6600 and rrig6600b handle 8 workers on identical hardware
remains genuinely unknown at end of S155.

---

## Commits This Session

| Commit | Description |
|--------|-------------|
| `7182ab5` | fix(s154): explicit GPU array del + gc.collect() |
| `5d8d10d` | fix(s155): CuPy pool cap + CUPY_GPU_MEMORY_LIMIT + ROCR restore |
| `89c7597` | config(s155): rrig6600c gpu_count=4 stable operating point |
| `8069692` | S130 persistent GPU sieve worker (merged) |

---

## Architecture Invariants Added S155

- **[S155]** `CUPY_CUDA_MEMORY_POOL_TYPE=none` REMOVED — caused 41GB VM mmap OOM
- **[S155]** `CUPY_GPU_MEMORY_LIMIT=268435456` in ROCM_ENV_VARS (256MB hard cap)
- **[S155]** `PRNG_CUPY_POOL_LIMIT_MB=256` in ROCM_ENV_VARS
- **[S155]** `cp.get_default_memory_pool().set_limit(256MB)` in `run_worker()` after warmup
- **[S155]** `ROCR_VISIBLE_DEVICES={gpu_id}` restored in `_spawn_worker()` for per-GPU isolation
- **[S155]** rrig6600c stable at `gpu_count=4` — do NOT set to 8 until root cause resolved
- **[S155]** Crash monitor MUST use `~/rig_crash_monitor_persistent.log` with `>>` append

---

## Open Issues at S155 Close

1. rrig6600c OOM root cause unknown — below Python/CuPy layer, in ROCm/amdgpu driver
2. PWC lifecycle bug — per-trial PWC creation (identified, fix pending S156)
3. Respawn storm — 537 threads simultaneously detect dead workers (known, not fixed)
4. java_lcg_reverse kernel arg mismatch — code bug, not crash cause, fix patch exists
5. Syncthing on all rigs — competes for RAM, should be disabled

---

## Next Session (S156)

1. Investigate PWC lifecycle bug — session-scoped PWC architectural fix
2. Monitor PageTables via remote crash monitor during spawn
3. Determine if rrig6600c crash is PWC accumulation or hardware init cliff
