# PROPOSAL: Job Batching for Pipeline Stability

**Author:** Claude (with Michael)  
**Date:** 2026-01-20  
**Status:** PROPOSAL  
**Priority:** HIGH - Blocking pipeline execution

---

## 1. Executive Summary

The WATCHER agent pipeline crashes both RX 6600 mining rigs when dispatching high job counts, while the benchmark script runs identical workloads successfully. Root cause analysis identifies **job volume per dispatch cycle** as the critical factor, not sample size, GPU concurrency, or SMU polling.

**Recommendation:** Implement job batching in `scripts_coordinator.py` with inter-batch cooldown and allocator reset.

---

## 2. Problem Statement

### Symptoms
- WATCHER pipeline with 120 trials crashes both rigs within 60 seconds
- Rigs become unresponsive (HDD light solid, SSH hung)
- Requires HiveOS reboot to recover

### What Works
- Benchmark script: 20 trials × 5 sample sizes = 100 total trials
- 100% success rate across all tests
- No crashes, no hangs

### What Fails
- WATCHER pipeline: 120 trials dispatched at once
- Immediate rig lockup

---

## 3. Root Cause Analysis

### Job Distribution Comparison

| Scenario | Total Jobs | Jobs per Rig | Result |
|----------|------------|--------------|--------|
| Benchmark | 20 | 7 | ✅ 100% success |
| Pipeline | 120 | 40 | ❌ Crash |

### Key Differences

| Factor | Benchmark | Pipeline |
|--------|-----------|----------|
| Jobs per dispatch | 20 | 120 |
| Inter-batch cooldown | 15s | None |
| Allocator reset | Yes (drop_caches) | No |
| Node stagger | Implicit (small batches) | None |

### Eliminated Causes

| Suspected Cause | Status | Evidence |
|-----------------|--------|----------|
| Sample size | ❌ Eliminated | All sizes 350-750 stable |
| GPU concurrency | ❌ Eliminated | 12 concurrent validated |
| SMU polling | ❌ Eliminated | Fixed with LOOP_SECS=20 |
| HIP init collision | ❌ Eliminated | Stagger already in place |
| Memory pressure | ❌ Eliminated | sample_size=350-750 all stable |

### Actual Cause

**Simultaneous job initialization overload.** When 40 jobs hit each rig:
1. All 40 SSH connections establish nearly simultaneously
2. All 40 processes begin data loading
3. NVMe I/O saturates
4. System becomes unresponsive

---

## 4. Proposed Solution

### Option A: Job Batching in scripts_coordinator.py (RECOMMENDED)

Add automatic batching with configurable batch size and inter-batch delay.

**Changes to `scripts_coordinator.py`:**

```python
# New configuration parameters
MAX_JOBS_PER_BATCH = 20  # Validated stable limit
INTER_BATCH_COOLDOWN = 5  # Seconds between batches
ENABLE_ALLOCATOR_RESET = True  # Reset between batches

def execute_jobs_batched(self, jobs: List[Job]):
    """Execute jobs in batches to prevent rig overload."""
    batches = [jobs[i:i + MAX_JOBS_PER_BATCH] 
               for i in range(0, len(jobs), MAX_JOBS_PER_BATCH)]
    
    for batch_num, batch in enumerate(batches):
        if batch_num > 0:
            if ENABLE_ALLOCATOR_RESET:
                self.reset_allocator_state()
            time.sleep(INTER_BATCH_COOLDOWN)
        
        self.execute_batch(batch)
```

**Effort:** ~2-4 hours  
**Risk:** Low - additive change, preserves existing behavior for small jobs  
**Autonomy Impact:** None - transparent to WATCHER

### Option B: WATCHER Trial Limit

Cap trials per WATCHER invocation at 20.

**Changes to `agents/watcher_agent.py`:**

```python
MAX_TRIALS_PER_STEP = 20

def run_step(self, step: int, params: dict):
    if step == 2 and params.get('trials', 0) > MAX_TRIALS_PER_STEP:
        params['trials'] = MAX_TRIALS_PER_STEP
        logger.warning(f"Capped trials to {MAX_TRIALS_PER_STEP}")
```

**Effort:** ~30 minutes  
**Risk:** Low  
**Downside:** Requires multiple WATCHER invocations for high trial counts

### Option C: Add Node-Level Stagger

Delay job dispatch between nodes (not just GPUs within a node).

**Changes to `scripts_coordinator.py`:**

```python
INTER_NODE_STAGGER = 5.0  # Seconds between starting each node

def execute_on_all_nodes(self, node_assignments):
    for i, (node, jobs) in enumerate(node_assignments.items()):
        if i > 0:
            time.sleep(INTER_NODE_STAGGER)
        self.execute_on_node(node, jobs)
```

**Effort:** ~1 hour  
**Risk:** Low  
**Downside:** May not fully solve issue if jobs per node still too high

---

## 5. Recommendation

**Implement Option A (Job Batching)** as the primary fix.

This mirrors what the benchmark does:
- Process jobs in manageable batches
- Reset allocator state between batches
- Cooldown period for system recovery

### Implementation Priority

1. **Phase 1:** Add batching to `scripts_coordinator.py` (Option A)
2. **Phase 2:** Add configurable parameters via `distributed_config.json`
3. **Phase 3:** Consider Option C (node stagger) if issues persist

---

## 6. Validation Plan

After implementation:

1. Run `./run_scorer_meta_optimizer.sh 60` (3 batches of 20)
2. Run `./run_scorer_meta_optimizer.sh 100` (5 batches of 20)
3. Run WATCHER pipeline with 100 trials
4. Confirm 100% success rate, no rig crashes

### Success Criteria

- [ ] 100 trials complete without crash
- [ ] Both rigs remain responsive throughout
- [ ] Throughput degradation < 20% vs unbatched (due to cooldown overhead)

---

## 7. Benchmark Data (2026-01-20)

```
sample_size  trials  throughput_trials_per_min  success_rate
350          20      15.34                      1.0000
450          20      13.52                      1.0000
550          20      14.66                      1.0000
650          20      13.05                      1.0000
750          20      12.24                      1.0000
```

All tests: **100% success rate** with 20 jobs per batch.

---

## 8. Files to Modify

| File | Change |
|------|--------|
| `scripts_coordinator.py` | Add batching logic, allocator reset |
| `distributed_config.json` | Add `max_jobs_per_batch`, `inter_batch_cooldown` |
| `CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` | Document batching behavior |
| `SESSION_CHANGELOG_20260120.md` | Record fix |

---

## 9. Decision Requested

- [ ] **APPROVED** - Implement Option A (Job Batching)
- [ ] **APPROVED WITH MODIFICATIONS** - (specify changes)
- [ ] **DECLINED** - (specify reason)
- [ ] **NEEDS MORE INFO** - (specify questions)

---

*Filed: 2026-01-20*  
*Review Required: Team Beta*
