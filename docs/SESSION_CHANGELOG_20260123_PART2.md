# Session Changelog - January 23, 2026 (Part 2: GPU Failure Investigation)

## Summary
Investigated persistent first-wave GPU failures on ROCm nodes. Root cause identified as **Linux OOM killer** due to insufficient system RAM.

## Root Cause Analysis

### Symptoms
- 1-3 GPU job failures per Step 3 run on rig-6600/rig-6600b
- Failures only in first parallel wave
- Jobs succeed on retry (to Zeus)
- No HIP error in stderr (process killed externally)

### Investigation Path
1. ❌ HIP kernel cache corruption → Fixed but failures continued
2. ❌ Parallel HIP initialization race → Warmup barrier didn't help
3. ❌ Missing HSA_OVERRIDE_GFX_VERSION → Fixed but failures continued
4. ✅ **OOM killer** → `journalctl -k` showed kernel killing Python workers

### Root Cause
```
rig-6600 kernel: Out of memory: Killed process 13994 (python)
rig-6600 kernel: total-vm:13611472kB, anon-rss:1491948kB
```

**Math:**
- rig-6600 RAM: 7.7GB
- Workers launched: 7 parallel
- RAM per worker: ~1.5GB
- Total needed: ~10.5GB
- Shortfall: ~3GB → OOM kill

## Fixes Applied (Evaluate for Retention)

| Fix | Status | Overhead | Keep? |
|-----|--------|----------|-------|
| HIP cache clear (PRE-FLIGHT) | Deployed | ~2s | Optional |
| GPU warmup barrier | Deployed | ~15s | Remove |
| HSA_OVERRIDE in warmup | Deployed | - | Remove if warmup removed |
| Debug stderr capture | Deployed | 0 | Keep |

## Real Fix Required

**Option A: Add RAM** (Recommended)
- Upgrade rig-6600/rig-6600b to 16GB
- Cost: ~$20-40 per node
- Result: Full 12-GPU parallelism

**Option B: Limit Concurrency** (Temporary)
- Cap workers to 5 per node in distributed_config.json
- Cost: Free
- Result: Slower but stable

## Files Modified
- `scripts_coordinator.py` - HIP cache clear, warmup barrier, debug output
- Backups: `.bak_warmup`, `.bak_hsa`, `.bak_debug`

## Key Learning
Always check `journalctl -k` or `dmesg` for kernel-level process kills. Python/HIP errors may not appear in stderr if the kernel OOM killer terminates the process externally.
