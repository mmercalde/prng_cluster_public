# PROPOSAL: Infrastructure Improvements & Stability Enhancements
## Version 1.0 | January 24, 2026 | Team Beta Review

---

## 📋 Executive Summary

This proposal addresses critical infrastructure gaps identified during the January 24, 2026 debugging session. A 2-hour WATCHER pipeline run failed silently due to: (1) insufficient Optuna trials finding zero survivors, (2) rig-6600 crash mid-run with no preflight detection, and (3) GPU[10] SMU hardware degradation. These issues highlight the need for systematic preflight checks, smarter parameter tuning, and improved GPU health monitoring.

---

## 🎯 Proposal Items

### ITEM A: Preflight Connectivity & Health Check (HIGH PRIORITY)

**Problem Statement:**
WATCHER launched a 2-hour Step 1 run that failed immediately because rig-6600 was unreachable. No preflight check detected this, wasting time and leaving the system in an inconsistent state.

**Proposed Solution:**
Implement `preflight_check.py` module called by WATCHER before ANY step execution:

```python
class PreflightCheck:
    """Run before each pipeline step to ensure cluster health."""
    
    def check_all(self) -> PreflightResult:
        results = PreflightResult()
        
        # 1. SSH connectivity to all nodes
        results.ssh = self.check_ssh_connectivity([
            "192.168.3.120",  # rig-6600
            "192.168.3.154"   # rig-6600b
        ])
        
        # 2. GPU availability (rocm-smi responsive)
        results.gpus = self.check_gpu_health()
        
        # 3. Ramdisk populated (for Steps 2.5+)
        if step >= 2:
            results.ramdisk = self.check_ramdisk([
                "/dev/shm/prng/step3/train_history.json",
                "/dev/shm/prng/step3/holdout_history.json"
            ])
        
        # 4. Required input files exist
        results.inputs = self.check_step_inputs(step)
        
        return results
```

**Integration Point:**
- `watcher_agent.py` calls `PreflightCheck.check_all()` before `_execute_step()`
- Fail fast with actionable error message if any check fails
- Log all preflight results for debugging

**Acceptance Criteria:**
- [ ] SSH connectivity verified before job dispatch
- [ ] GPU health confirmed (no 0xFFFFFFFF SMU responses)
- [ ] Ramdisk population verified for Steps 2.5+
- [ ] Clear error messages on preflight failure

---

### ITEM B: LLM-Adjustable Trial and Seed Parameters (HIGH PRIORITY)

**Problem Statement:**
January 19 run: 100 trials × 100K seeds → 98,172 survivors ✅
January 24 run: 50 trials × 10M seeds → 0 survivors ❌

More trials enables better Optuna exploration. More seeds just wastes time on bad parameter combinations. Currently, these values are hardcoded or manually specified.

**Proposed Solution:**
Add `trials` and `seed_count` to LLM-adjustable parameters with decision logic:

```python
# In agents/contexts/window_optimizer_context.py

def suggest_parameters(self, history: List[RunResult]) -> Dict:
    """LLM-guided parameter selection."""
    
    # Default starting point
    params = {
        "trials": 50,
        "max_seeds": 100000  # Start conservative
    }
    
    # Adaptive logic based on history
    if history:
        last_run = history[-1]
        
        # If 0 survivors after 30+ trials, increase trials
        if last_run.survivors == 0 and last_run.completed_trials >= 30:
            params["trials"] = 100
            params["max_seeds"] = 100000  # Keep seeds low
        
        # If survivors < 50K but > 0, could try more seeds
        elif 0 < last_run.survivors < 50000:
            params["max_seeds"] = min(last_run.max_seeds * 2, 10000000)
    
    return params
```

**Key Insight:**
> "More trials, fewer seeds" - Optuna needs exploration budget, not brute-force seed counts.

**Acceptance Criteria:**
- [ ] `trials` parameter exposed to LLM decision layer
- [ ] `seed_count` parameter exposed to LLM decision layer
- [ ] Default: 50 trials, 100K seeds (proven working configuration)
- [ ] Adaptive increase logic when survivors = 0

---

### ITEM C: Post-Job GPU Cleanup & Health Verification (MEDIUM PRIORITY)

**Problem Statement:**
GPU[10] on rig-6600 experienced SMU failures (0xFFFFFFFF) that cascaded into system crash. No cleanup or health check runs after jobs complete, potentially leaving HIP contexts or GPU state corrupted.

**Proposed Solution:**
Add GPU cleanup and verification after each job batch:

```bash
# cleanup_gpus.sh - Run after job completion on each node

#!/bin/bash
# Reset GPU clocks to default
rocm-smi --resetclocks

# Clear any HIP cache
rm -rf ~/.cache/hip_* 2>/dev/null

# Verify all GPUs responding
rocm-smi --showuse | grep -q "N/A" && exit 1

# Check for SMU errors (0xFFFFFFFF pattern)
dmesg | tail -50 | grep -q "0xFFFFFFFF" && echo "WARNING: SMU errors detected"

exit 0
```

**Integration Point:**
- `scripts_coordinator.py` calls cleanup after each batch completes
- Log warnings if cleanup detects issues
- Optional: Skip problematic GPUs in subsequent batches

**Acceptance Criteria:**
- [ ] GPU clocks reset after job batches
- [ ] HIP cache cleared between runs
- [ ] SMU error detection with warning log
- [ ] No orphaned VRAM confirmed

---

### ITEM D: GPU[10] Monitoring & Isolation (MEDIUM PRIORITY)

**Problem Statement:**
GPU[10] (PCI 0000:22:00.0) on rig-6600 shows same SMU 0xFFFFFFFF pattern as rig-6600b GPU[4]. This is a hardware defect that can cause system-wide crashes.

**Current Known Bad GPUs:**

| Rig | GPU | PCI Address | Issue | Current Workaround |
|-----|-----|-------------|-------|-------------------|
| rig-6600b | GPU[4] | 0F:00.0 | SMU 0xFFFFFFFF | Fan service disabled |
| rig-6600 | GPU[10] | 22:00.0 | SMU 0xFFFFFFFF | **None (NEW)** |

**Proposed Solution:**

**Option D1: Monitor Only (Recommended for now)**
- Add GPU[10] to watch list
- Log SMU errors if detected during preflight
- Escalate if errors occur

**Option D2: Exclude from Job Distribution**
```python
# In distributed_config.json
{
  "nodes": {
    "192.168.3.120": {
      "excluded_gpus": [10],  # Skip GPU[10]
      "reason": "SMU 0xFFFFFFFF errors"
    }
  }
}
```

**Option D3: Hardware Fix**
- Reseat GPU[10] physically
- Check power connections
- Consider RMA if under warranty

**Recommendation:** Start with D1 (monitor), escalate to D2 if crashes continue.

**Acceptance Criteria:**
- [ ] GPU[10] added to monitoring watchlist
- [ ] SMU error detection in preflight check
- [ ] Documented in CHAPTER_9 hardware issues section

---

### ITEM E: Optuna Study Persistence (LOW PRIORITY - FUTURE)

**Problem Statement:**
Each WATCHER run creates a NEW Optuna study:
```python
study_name = f"window_opt_{int(time.time())}"  # Fresh study every time!
```

This discards all learned knowledge from previous runs. A 100-trial run that found optimal parameters is forgotten on the next run.

**Proposed Solution (Future Enhancement):**
```python
# Option 1: Named persistent study
study_name = "window_optimizer_main"
study = optuna.load_study(study_name, storage) if exists else optuna.create_study(...)

# Option 2: Warm-start from previous best
if previous_best_params:
    study.enqueue_trial(previous_best_params)  # Start near known good region
```

**Note:** This is a future enhancement. Current workaround is to run sufficient trials (100+) each time.

**Acceptance Criteria:**
- [ ] Design document for Optuna persistence strategy
- [ ] Backward compatibility with existing runs

---

### ITEM F: Debug Logging for Intermittent Failures (LOW PRIORITY)

**Problem Statement:**
From userMemories: "Investigate GPU2 failure on rig-6600 during Step 3 - add debug logging to capture error before retry masks it."

**Proposed Solution:**
Add verbose logging before retry logic:

```python
# In scripts_coordinator.py

def execute_job(self, job):
    try:
        result = self._run_job(job)
    except Exception as e:
        # Log BEFORE retry masks the error
        logger.error(f"Job {job.job_id} failed on {job.gpu}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        logger.error(f"GPU state: {self._get_gpu_state(job.gpu)}")
        
        # Now retry...
        result = self._retry_job(job)
```

**Acceptance Criteria:**
- [ ] Full error captured before retry
- [ ] GPU state logged on failure
- [ ] Errors queryable in logs post-hoc

---

## 📊 Implementation Priority Matrix

| Item | Priority | Effort | Impact | Dependencies |
|------|----------|--------|--------|--------------|
| A. Preflight Check | HIGH | Medium | HIGH | None |
| B. LLM Parameters | HIGH | Low | HIGH | Item A |
| C. GPU Cleanup | MEDIUM | Low | MEDIUM | None |
| D. GPU[10] Monitor | MEDIUM | Low | MEDIUM | Item A |
| E. Optuna Persistence | LOW | High | MEDIUM | None |
| F. Debug Logging | LOW | Low | LOW | None |

---

## 🚀 Proposed Implementation Order

1. **Phase 1 (Immediate):** Items A + B - Preflight checks and parameter tuning
2. **Phase 2 (Next Session):** Items C + D - GPU cleanup and monitoring
3. **Phase 3 (Future):** Items E + F - Optuna persistence and debug logging

---

## ✅ Team Beta Approval Request

**Requesting approval to:**

1. Implement preflight connectivity check (Item A)
2. Add trials/seeds to LLM-adjustable parameters (Item B)
3. Add GPU cleanup after job batches (Item C)
4. Monitor GPU[10] on rig-6600 (Item D)

**Deferred to future session:**
- Optuna study persistence (Item E)
- Debug logging enhancement (Item F)

---

## 📝 Session Context

**Date:** January 24, 2026
**Issue:** WATCHER Step 1-3 pipeline failed after 2 hours
**Root Causes Identified:**
1. Optuna found 0 survivors (34 trials insufficient exploration)
2. rig-6600 crashed mid-run (GPU[10] SMU failure)
3. No preflight check detected unreachable node

**Key Learning:** More trials (100) with fewer seeds (100K) outperforms fewer trials (50) with more seeds (10M) for Optuna exploration.

---

*Submitted for Team Beta review - January 24, 2026*
