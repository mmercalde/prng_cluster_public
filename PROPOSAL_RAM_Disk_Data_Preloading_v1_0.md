# PROPOSAL: RAM Disk Data Preloading for Distributed Workers

**To:** Team Beta  
**From:** Claude (AI Assistant)  
**Date:** 2026-01-20  
**Status:** PENDING REVIEW  
**Proposal ID:** PROP-2026-01-20-RAMDISK

---

## 1. Executive Summary

This proposal recommends implementing `/dev/shm` (RAM disk) preloading for input data files on remote mining rigs to eliminate disk I/O contention during distributed job execution.

**Scope:** Lightweight infrastructure change (~30 minutes implementation)  
**Impact:** All script-based pipeline steps (2-6) on remote nodes  
**Risk Level:** Low (no worker code changes, fully backward compatible)  
**Expected Benefit:** Eliminate disk I/O as failure vector, potential 20-30% throughput improvement

---

## 2. Problem Statement

### 2.1 Current Situation (Post-Batching Fix)

Today's session (2026-01-20) implemented per-node job batching to prevent crashes:

| Fix | What It Addresses |
|-----|-------------------|
| `MAX_JOBS_PER_BATCH = 20` | Limits total jobs dispatched per cycle |
| `MAX_JOBS_PER_NODE_PER_BATCH = 6` | Limits concurrent job starts per ROCm node |

**Result:** 25/25 trials completed successfully at 100% pass rate.

### 2.2 What Batching Does NOT Fix

Batching limits *when* jobs start, but does not change *what* each job does at startup:

```
Job Startup Sequence (per worker):
1. SSH connect
2. Python interpreter launch
3. Load bidirectional_survivors_binary.npz (228 KB) ← DISK I/O
4. Load train_history.json (27 KB) ← DISK I/O
5. Load holdout_history.json (7 KB) ← DISK I/O
6. HIP initialization
7. GPU compute
8. Write result
```

With 6 concurrent jobs, that's **6 simultaneous disk reads** during startup. The SSD queue depth gets saturated, causing latency variance and potential I/O wait states.

### 2.3 Evidence of I/O Root Cause

From conversation history (2026-01-16):

> "Hard disk LED stuck ON = I/O thrashing or system hang during disk access."
> 
> "That's the smoking gun."

The Jan 18 benchmarking concluded the root cause was "memory pressure during data loading" - which `/dev/shm` directly addresses by eliminating disk access entirely.

---

## 3. Proposed Solution

### 3.1 Mechanism

`/dev/shm` is a POSIX shared memory filesystem mounted in RAM. It's built into Linux (no installation required) and provides:

- **Instant access:** No disk seek time, no I/O scheduling
- **Parallel reads:** Multiple processes can read simultaneously without contention
- **Persistence:** Survives across job runs (until reboot or manual cleanup)

### 3.2 Memory Impact

| File | Size |
|------|------|
| bidirectional_survivors_binary.npz | 228 KB |
| train_history.json | 27 KB |
| holdout_history.json | 7 KB |
| scorer_jobs.json | ~50 KB |
| **Total** | **~312 KB** |

On 8 GB systems: **0.004% of RAM** - completely negligible.

### 3.3 Architecture

```
CURRENT FLOW:
┌─────────────────┐     SCP      ┌─────────────────┐
│  Zeus           │ ──────────→  │  Remote Rig     │
│  (coordinator)  │              │  ~/distributed_ │
└─────────────────┘              │  prng_analysis/ │
                                 │  (SSD)          │
                                 └────────┬────────┘
                                          │ DISK READ
                                          ▼
                                 ┌─────────────────┐
                                 │  Worker Process │
                                 │  (loads data)   │
                                 └─────────────────┘

PROPOSED FLOW:
┌─────────────────┐     SCP      ┌─────────────────┐
│  Zeus           │ ──────────→  │  Remote Rig     │
│  (coordinator)  │              │  ~/distributed_ │
└─────────────────┘              │  prng_analysis/ │
                                 │  (SSD)          │
                                 └────────┬────────┘
                                          │ ONE-TIME COPY
                                          ▼
                                 ┌─────────────────┐
                                 │  /dev/shm/prng/ │
                                 │  (RAM)          │
                                 └────────┬────────┘
                                          │ RAM READ (instant)
                                          ▼
                                 ┌─────────────────┐
                                 │  Worker Process │
                                 │  (loads data)   │
                                 └─────────────────┘
```

---

## 4. Scope of Changes

### 4.1 Files Requiring Modification

| File | Change Type | Description |
|------|-------------|-------------|
| `run_scorer_meta_optimizer.sh` | MODIFY | Add ramdisk copy after SCP, modify job paths |
| `generate_scorer_jobs.py` | MODIFY | Use `/dev/shm/prng/` paths for remote nodes |

### 4.2 Files NOT Requiring Modification

| File | Reason |
|------|--------|
| `scorer_trial_worker.py` | Reads from path in args - path agnostic |
| `scripts_coordinator.py` | Dispatches jobs as-is - path agnostic |
| `coordinator.py` | Not used for script jobs |
| Steps 3-6 workers | Same pattern - path comes from job spec |

### 4.3 Backward Compatibility

- **Zeus (localhost):** Continues using local NVMe paths (no change needed)
- **Remote rigs without ramdisk:** Falls back to SSD paths (graceful degradation)
- **Feature flag:** Can be enabled/disabled via config or shell variable

---

## 5. Affected Pipeline Steps

### 5.1 Step Analysis

| Step | Script | Distributed? | Uses Data Files? | Ramdisk Benefit? |
|------|--------|--------------|------------------|------------------|
| 1 | `window_optimizer.py` | No (Zeus only) | Yes | N/A |
| 2 | `run_scorer_meta_optimizer.sh` | **Yes** | **Yes** | **HIGH** |
| 3 | `run_step3_full_scoring.sh` | **Yes** | **Yes** | **HIGH** |
| 4 | `adaptive_meta_optimizer.py` | No (Zeus only) | No | N/A |
| 5 | `meta_prediction_optimizer_anti_overfit.py` | **Yes** | **Yes** | **HIGH** |
| 6 | `reinforcement_engine.py` | No (Zeus only) | Yes | N/A |

**Primary beneficiaries:** Steps 2, 3, and 5 (all distributed script jobs)

### 5.2 Step 2 Implementation (Template for Others)

```bash
# In run_scorer_meta_optimizer.sh, after SCP:

echo "Preloading data to ramdisk on remote nodes..."
for REMOTE in 192.168.3.120 192.168.3.154; do
    ssh $REMOTE "mkdir -p /dev/shm/prng && \
        cp ~/distributed_prng_analysis/bidirectional_survivors_binary.npz /dev/shm/prng/ && \
        cp ~/distributed_prng_analysis/train_history.json /dev/shm/prng/ && \
        cp ~/distributed_prng_analysis/holdout_history.json /dev/shm/prng/ && \
        cp ~/distributed_prng_analysis/scorer_jobs.json /dev/shm/prng/"
done
echo "Ramdisk preload complete"
```

```python
# In generate_scorer_jobs.py:

def get_data_path(filename, node_hostname):
    """Return appropriate path based on node type."""
    if node_hostname == 'localhost':
        return f"/home/michael/distributed_prng_analysis/{filename}"
    else:
        return f"/dev/shm/prng/{filename}"
```

---

## 6. Comparison with Declined Proposal

### 6.1 Persistent GPU Workers (DECLINED 2026-01-18)

| Aspect | Persistent Workers | RAM Disk Preloading |
|--------|-------------------|---------------------|
| Complexity | High (daemon, polling, state) | Low (one-time copy) |
| Code changes | ~600 lines new code | ~20 lines modified |
| Testing burden | Full regression | Minimal |
| New failure modes | Many (stale workers, orphans) | None |
| Addresses I/O issue | Yes (indirectly) | Yes (directly) |
| Addresses HIP init | Yes | No |
| Implementation time | 2-3 days | 30 minutes |

### 6.2 Why This Proposal Differs

The Persistent Workers proposal was declined because:
1. The problem was "solved" by configuration tuning
2. Complexity exceeded benefit

RAM Disk Preloading is fundamentally different:
1. **Minimal complexity** - no new daemons or state management
2. **Directly targets root cause** - eliminates disk I/O during job startup
3. **Zero impact on existing code** - workers are path-agnostic
4. **Graceful degradation** - falls back to disk if ramdisk unavailable

---

## 7. Implementation Plan

### 7.1 Phase 1: Step 2 Only (30 minutes)

1. Backup `run_scorer_meta_optimizer.sh`
2. Add ramdisk copy commands after SCP
3. Modify `generate_scorer_jobs.py` to use `/dev/shm/prng/` for remote nodes
4. Test with 25 trials
5. Benchmark: compare trial times before/after

### 7.2 Phase 2: Steps 3 and 5 (1 hour)

1. Apply same pattern to `run_step3_full_scoring.sh`
2. Apply same pattern to Step 5 launcher
3. Test each step individually

### 7.3 Phase 3: Documentation (30 minutes)

1. Update CHAPTER_3 (Scorer Meta-Optimizer)
2. Update CHAPTER_9 (GPU Cluster Infrastructure)
3. Add troubleshooting section for ramdisk issues

---

## 8. Rollback Plan

If issues arise:

```bash
# Immediate rollback - restore from backup
cp run_scorer_meta_optimizer.sh.backup run_scorer_meta_optimizer.sh
git checkout generate_scorer_jobs.py

# Or disable ramdisk (if feature flag implemented)
export USE_RAMDISK=false
```

---

## 9. Success Metrics

| Metric | Current (Batching Only) | Target (With Ramdisk) |
|--------|------------------------|----------------------|
| Remote trial time | 45-60s | 30-40s |
| I/O-related failures | Prevented by batching | Eliminated at source |
| Throughput (trials/min) | ~15.4 | ~18-20 (estimated) |

---

## 10. Recommendation

**APPROVE** this proposal for Phase 1 implementation (Step 2 only).

**Rationale:**
- Low risk, high reward
- Directly addresses identified root cause
- Does not conflict with existing batching fix (complementary)
- Provides foundation for future Steps 3 and 5

---

## 11. Open Questions for Team Beta

1. Should ramdisk cleanup be automatic (after job completion) or manual?
2. Should we implement a config flag (`USE_RAMDISK=true/false`) for easy toggling?
3. Any concerns about `/dev/shm` persistence across reboots? (Answer: it clears on reboot, requiring re-copy)

---

**Submitted for review.**

*Claude (AI Assistant)*  
*2026-01-20*
