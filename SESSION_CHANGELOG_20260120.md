# SESSION CHANGELOG - 2026-01-20

## Step 2 Crash Root Cause Identified & Fixed

### Root Cause
SMU (System Management Unit) exhaustion from excessive polling, NOT SDMA timeout or memory pressure.

### Contributing Factors

| Source | Rate | Impact |
|--------|------|--------|
| `watch -n 0.5 rocm-smi` (2 instances) | ~240-360 reads/min | **Primary cause** |
| Fan service loop (3s) | ~240 writes/min | Contributing |
| **Combined** | **~500+ SMU ops/min** | Crash |

### Fixes Applied

1. **Killed watch processes** - Removed aggressive `watch -n 0.5 rocm-smi` monitoring
2. **Fan script updated** - Changed `LOOP_SECS` from 3 to 20
   - Old: ~240 SMU writes/min
   - New: ~36 SMU writes/min
   - **~14x reduction in SMU load**

### Files Modified
- `/usr/local/bin/rocm-fan-curve.sh` on both rigs (LOOP_SECS=20)

### Kernel Log Signature (for future reference)
```
SMU: response:0xFFFFFFFF for message:TransferTableSmu2Dram
Failed to export SMU metrics table
failed to write reg 2890 wait reg 28a2
```

### Validation
- Both rigs showing healthy fan control (Level 71, RPM 1400-1700)
- No SMU errors in dmesg post-fix
- Ready for Step 2 retry

### Key Learning
ROCm SMU is sensitive to polling frequency. For 12-GPU rigs:
- Fan control loop: 15-30 seconds minimum
- Monitoring: 10+ seconds between `rocm-smi` calls
- Never use `watch -n 0.5` or similar aggressive polling
