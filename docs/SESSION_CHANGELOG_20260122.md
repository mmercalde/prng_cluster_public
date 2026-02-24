# SESSION CHANGELOG - 2026-01-22

## Summary

Implemented Team Beta's Step 3 stability ruling: step-aware batching to prevent RX 6600 node overload.

---

## Root Cause Analysis (Team Beta)

**Unified root cause:** Step 3 exceeds the validated operational envelope of RX 6600 nodes.

Step 3 is **6–7× heavier** per job than Step 2.5:
- Larger survivor sets per job
- Higher memory footprint
- More concurrent file I/O
- Simultaneous job startup across GPUs

| Node | Failure Mode | Mechanism |
|------|--------------|-----------|
| **rig-6600b** | Kernel deadlock/crash | SMU-faulty GPU[4] amplifies overload |
| **rig-6600** | GPU 11 "unknown" state | Driver reset failure under I/O + HIP storms |

**Key insight:** These are two failure modes of the **same overload event**, not separate bugs.

---

## Changes Applied

### 1. Step-Aware Batching (scripts_coordinator.py)

| Parameter | Step 2.5 (default) | Step 3 (Full Scoring) |
|-----------|-------------------|----------------------|
| `MAX_JOBS_PER_NODE_PER_BATCH` | 6 | **2** |
| `INTER_BATCH_COOLDOWN` | 5.0s | **10.0s** |

**Detection method:** Automatic via job file name and/or `job_type` field in job specs.

**New constants:**
```python
# Step 2.5 and other steps (default)
DEFAULT_MAX_JOBS_PER_NODE_PER_BATCH = 6
DEFAULT_INTER_BATCH_COOLDOWN = 5.0

# Step 3 (Full Scoring) - heavier workload
STEP3_MAX_JOBS_PER_NODE_PER_BATCH = 2
STEP3_INTER_BATCH_COOLDOWN = 10.0
```

**New functions:**
- `detect_job_step()` - Identifies Step 3 vs Step 2.5 jobs
- `get_step_aware_limits()` - Returns appropriate batching limits

### 2. Tarball Chunk Distribution

**Problem:** Step 3 ships hundreds of small JSON chunk files, causing SSD + kernel I/O storms.

**Solution:** Tar + single SCP + extract instead of individual file transfers.

**Helper script:** `step3_tarball_helpers.sh`

```bash
# Usage
source step3_tarball_helpers.sh
distribute_chunks_all_rigs scoring_chunks
```

### 3. SMU Polling Protection (rig-6600b)

**Status:** `rocm-fan-curve.service` remains masked.

```bash
sudo systemctl disable --now rocm-fan-curve.service
sudo systemctl mask rocm-fan-curve.service
```

---

## Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `scripts_coordinator.py` | MODIFIED | Step-aware batching logic |
| `step3_tarball_helpers.sh` | NEW | Tarball distribution helper |

---

## Verification Commands

### Check patch applied:
```bash
grep -A 5 'STEP3_MAX_JOBS' scripts_coordinator.py
```

### Expected output:
```python
STEP3_MAX_JOBS_PER_NODE_PER_BATCH = 2     # Team Beta ruling: cap=2 for Step 3
STEP3_INTER_BATCH_COOLDOWN = 10.0         # Team Beta ruling: cooldown=10s for Step 3
```

### Check runtime behavior:
Look for these log lines when running Step 3:
```
[STEP-AWARE] Detected: Step 3 (Full Scoring)
[STEP-AWARE] Limits: max_per_node=2, cooldown=10.0s
```

---

## Decisions Explicitly NOT Taken

| Option | Status | Rationale |
|--------|--------|-----------|
| Persistent GPU Workers | ❌ Deferred | Revisit after stability restored |
| Additional NPZ conversion | ❌ Not needed | Current NPZ coverage sufficient |
| Hardware replacement | ❌ Not needed | Unless GPU[4] required for capacity |

---

## Pending Items

| Item | Status | Notes |
|------|--------|-------|
| GPU[4] SMU investigation | 🟡 Deferred | Power/riser/ppfeaturemask? |
| Re-run Step 3 full cluster | ⏳ Waiting | After ramdisk repopulation |
| Ramdisk repopulation on rig-6600b | ⏳ Required | Before next pipeline run |

---

## Expected Outcome

After applying step-aware batching:

- ✅ No further GPU "unknown" states during Step 3
- ✅ No system-level crashes on rig-6600b
- ✅ Stable Step 3 execution comparable to Step 2.5 behavior
- ✅ Clear logging of effective limits for debugging

---

## Git Commit Message

```
fix: Step-aware batching for Step 3 stability (Team Beta 2026-01-22)

- Step 3: max_per_node=2, cooldown=10s (was 6/5s)
- Step 2.5: unchanged (6/5s)  
- Add detect_job_step() for automatic step detection
- Add tarball helper for chunk distribution
- Prevents RX 6600 overload (GPU unknown states, crashes)

Files: scripts_coordinator.py, step3_tarball_helpers.sh
```

---

*Team Beta Stability & Infrastructure Review - 2026-01-22*
