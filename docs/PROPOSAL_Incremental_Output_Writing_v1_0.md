# PROPOSAL: Incremental Output Writing for Window Optimizer

**Version:** 1.0.0  
**Date:** 2026-01-18  
**Author:** Claude (on behalf of Michael)  
**Status:** Awaiting Team Beta Approval  
**Affects:** `window_optimizer.py`, `window_optimizer_bayesian.py`

---

## 1. Problem Statement

### 1.1 Observed Failure

On 2026-01-18, a 50-trial Bayesian window optimization run was interrupted after 2 hours (21 trials completed). Despite accumulating **1.8M+ bidirectional survivors** across successful trials, **zero output files were created**.

```
📊 Accumulated totals:
Forward: 3096688 total
Reverse: 3050783 total
Bidirectional: 1814332 total
Trial 21: W67_O86_midday_S1-118_FT0.39_RT0.19 → Score: 0.00

⚠️  HUMAN REVIEW REQUIRED
Reason: File validation failed: ['optimal_window_config.json: File does not exist']
```

### 1.2 Root Cause

The current architecture writes output files **only after ALL trials complete**:

```
Trial 1  → memory only
Trial 2  → memory only
...
Trial 49 → memory only
Trial 50 → FINALLY write optimal_window_config.json
         → FINALLY write bidirectional_survivors.json
```

**Failure modes this creates:**
- Crash/timeout at trial N → lose ALL work from trials 1 to N-1
- WATCHER cannot evaluate progress mid-run
- No crash recovery possible
- Long runs have catastrophic failure risk

### 1.3 Impact

| Scenario | Current Behavior | Expected Behavior |
|----------|------------------|-------------------|
| 50 trials, crash at trial 49 | 0 output, 0 survivors | Best from 48 trials preserved |
| WATCHER 2-hour timeout | Escalate, no data | Can evaluate partial results |
| User checks progress | No visibility | Can see current best |
| Power failure mid-run | Total loss | Resume from checkpoint |

---

## 2. Proposed Solution

### 2.1 Design Principle

**Write incrementally, update on improvement.**

After each trial:
1. If new best found → update `optimal_window_config.json`
2. If new best found → update `bidirectional_survivors.json` with that trial's survivors
3. Always update progress metadata

### 2.2 Implementation: Optuna Callback

Optuna natively supports callbacks invoked after each trial. We add a `save_best_so_far_callback`:

```python
# In window_optimizer_bayesian.py

from datetime import datetime
from pathlib import Path
import json

def create_incremental_save_callback(
    output_config_path: str = "optimal_window_config.json",
    output_survivors_path: str = "bidirectional_survivors.json",
    total_trials: int = 50
):
    """
    Factory function that creates an Optuna callback for incremental saving.
    
    The callback writes best-so-far results after each completed trial,
    ensuring crash recovery and WATCHER visibility.
    
    Args:
        output_config_path: Path to write optimal config JSON
        output_survivors_path: Path to write bidirectional survivors JSON
        total_trials: Total planned trials (for progress reporting)
    
    Returns:
        Callable suitable for study.optimize(callbacks=[...])
    """
    
    def save_best_so_far_callback(study, trial):
        """Invoked after each trial completes."""
        
        completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
        
        # Build progress metadata (always written)
        progress = {
            "status": "in_progress",
            "completed_trials": completed,
            "total_trials": total_trials,
            "last_updated": datetime.now().isoformat(),
            "last_trial_number": trial.number,
            "last_trial_value": trial.value if trial.value is not None else None,
        }
        
        # If we have a best trial, include it
        if study.best_trial is not None:
            best_config = {
                "best_trial_number": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_bidirectional_count": int(study.best_value) if study.best_value else 0,
                
                # Extract window parameters for downstream compatibility
                "window_size": study.best_params.get("window_size"),
                "offset": study.best_params.get("offset"),
                "skip_min": study.best_params.get("skip_min"),
                "skip_max": study.best_params.get("skip_max"),
                "forward_threshold": study.best_params.get("forward_threshold"),
                "reverse_threshold": study.best_params.get("reverse_threshold"),
                "time_of_day": study.best_params.get("time_of_day", "all"),
                
                # Progress tracking
                **progress
            }
            
            # Write config atomically (write to temp, then rename)
            temp_path = Path(output_config_path).with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(best_config, f, indent=2)
            temp_path.rename(output_config_path)
            
            # If THIS trial is the new best, save its survivors
            if trial.number == study.best_trial.number:
                survivors = trial.user_attrs.get("bidirectional_survivors", [])
                if survivors:
                    temp_surv = Path(output_survivors_path).with_suffix(".tmp")
                    with open(temp_surv, "w") as f:
                        json.dump(survivors, f)
                    temp_surv.rename(output_survivors_path)
        
        else:
            # No successful trial yet - write progress-only file
            progress["best_trial_number"] = None
            progress["best_value"] = None
            progress["note"] = "No successful trials yet"
            
            temp_path = Path(output_config_path).with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(progress, f, indent=2)
            temp_path.rename(output_config_path)
    
    return save_best_so_far_callback
```

### 2.3 Integration Point

In `OptunaBayesianSearch.search()` or equivalent:

```python
# BEFORE (current):
study.optimize(objective, n_trials=max_iterations)

# AFTER (with incremental saving):
callback = create_incremental_save_callback(
    output_config_path=output_file,
    output_survivors_path="bidirectional_survivors.json",
    total_trials=max_iterations
)
study.optimize(objective, n_trials=max_iterations, callbacks=[callback])
```

### 2.4 Storing Survivors in Trial

The objective function must store survivors in trial user_attrs:

```python
def objective(trial):
    # ... run sieve, get results ...
    
    bidirectional_survivors = results.get("bidirectional_survivors", [])
    
    # Store in trial for callback access
    trial.set_user_attr("bidirectional_survivors", bidirectional_survivors)
    
    # Return score (survivor count)
    return len(bidirectional_survivors)
```

---

## 3. Output File Schema

### 3.1 optimal_window_config.json (During Run)

```json
{
  "status": "in_progress",
  "completed_trials": 21,
  "total_trials": 50,
  "last_updated": "2026-01-18T19:45:00",
  "last_trial_number": 20,
  "last_trial_value": 0,
  
  "best_trial_number": 15,
  "best_value": 1814332,
  "best_bidirectional_count": 1814332,
  "best_params": {
    "window_size": 512,
    "offset": 0,
    "skip_min": 1,
    "skip_max": 20,
    "forward_threshold": 0.45,
    "reverse_threshold": 0.55,
    "time_of_day": "all"
  },
  
  "window_size": 512,
  "offset": 0,
  "skip_min": 1,
  "skip_max": 20,
  "forward_threshold": 0.45,
  "reverse_threshold": 0.55,
  "time_of_day": "all"
}
```

### 3.2 optimal_window_config.json (After Completion)

```json
{
  "status": "complete",
  "completed_trials": 50,
  "total_trials": 50,
  "last_updated": "2026-01-18T22:30:00",
  
  "best_trial_number": 37,
  "best_value": 2145678,
  "best_bidirectional_count": 2145678,
  "best_params": { ... },
  
  "window_size": 256,
  "offset": 10,
  ...
}
```

### 3.3 Backward Compatibility

The output schema is **additive**:
- All existing fields preserved (`window_size`, `offset`, etc.)
- New fields added (`status`, `completed_trials`, `total_trials`)
- Downstream steps (2.5, 3, etc.) read the same fields they always did

---

## 4. WATCHER Integration

### 4.1 Updated Evaluation Logic

WATCHER can now evaluate Step 1 **during execution**:

```python
def evaluate_step_1(self, results_path="optimal_window_config.json"):
    """Evaluate window optimizer - supports in-progress evaluation."""
    
    if not Path(results_path).exists():
        return Decision(action="retry", confidence=0.0, 
                       reasoning="Output file not yet created")
    
    with open(results_path) as f:
        config = json.load(f)
    
    status = config.get("status", "complete")  # Default for old format
    
    if status == "in_progress":
        # Mid-run evaluation
        completed = config.get("completed_trials", 0)
        total = config.get("total_trials", 50)
        best_value = config.get("best_value")
        
        if best_value and best_value > 1000:
            # Good progress - let it continue
            return Decision(action="wait", confidence=0.7,
                           reasoning=f"In progress: {completed}/{total} trials, "
                                    f"best={best_value} survivors")
        elif completed > total * 0.5:
            # Past halfway with poor results
            return Decision(action="retry", confidence=0.4,
                           reasoning=f"Poor results after {completed} trials")
        else:
            # Still early - let it run
            return Decision(action="wait", confidence=0.5,
                           reasoning=f"Early stage: {completed}/{total} trials")
    
    else:
        # Completed - use existing evaluation logic
        survivors = config.get("best_bidirectional_count", 0)
        if survivors > 10000:
            return Decision(action="proceed", confidence=0.95, ...)
        elif survivors > 1000:
            return Decision(action="proceed", confidence=0.80, ...)
        # ... etc
```

### 4.2 New WATCHER Action: "wait"

| Action | Meaning |
|--------|---------|
| `proceed` | Step complete, move to next |
| `retry` | Step failed, retry |
| `escalate` | Need human review |
| `wait` | **NEW** - Step in progress, check again later |

---

## 5. Files Modified

| File | Change |
|------|--------|
| `window_optimizer_bayesian.py` | Add `create_incremental_save_callback()` |
| `window_optimizer.py` | Pass callback to optimizer |
| `agents/watcher_agent.py` | Add "wait" action, update Step 1 evaluation |
| `agent_manifests/window_optimizer.json` | Document incremental output behavior |

---

## 6. Testing Plan

### 6.1 Unit Tests

```python
def test_incremental_save_callback():
    """Verify callback writes after each trial."""
    study = optuna.create_study()
    callback = create_incremental_save_callback(
        output_config_path="test_config.json",
        total_trials=5
    )
    
    # Simulate trials
    for i in range(5):
        trial = study.ask()
        # ... mock objective ...
        study.tell(trial, value=i * 100)
        callback(study, study.trials[-1])
        
        # Verify file exists and has correct progress
        assert Path("test_config.json").exists()
        with open("test_config.json") as f:
            config = json.load(f)
        assert config["completed_trials"] == i + 1
```

### 6.2 Integration Test

```bash
# Run with 5 trials, kill at trial 3
timeout 5m python3 window_optimizer.py --strategy bayesian \
    --lottery-file daily3.json --trials 10 --max-seeds 1000000

# Verify partial output exists
cat optimal_window_config.json | jq '.status, .completed_trials'
# Expected: "in_progress", 3 (or similar)
```

### 6.3 WATCHER Test

```bash
# Start long-running optimization
python3 window_optimizer.py --strategy bayesian --trials 50 &

# In another terminal, test WATCHER evaluation
sleep 120  # Wait for some trials
python3 -c "
from agents.watcher_agent import WatcherAgent
w = WatcherAgent()
result = w.evaluate_step_1()
print(f'Action: {result.action}, Confidence: {result.confidence}')
print(f'Reasoning: {result.reasoning}')
"
# Expected: Action: wait, Confidence: 0.7
```

---

## 7. Rollback Plan

If issues arise:
1. Remove callback from `study.optimize()` call
2. Revert to end-of-run file writing
3. No schema changes needed (new fields are additive)

---

## 8. Approval Checklist

| Requirement | Status |
|-------------|--------|
| No breaking changes to output schema | ✅ Additive only |
| Backward compatible with existing WATCHER | ✅ Default status="complete" |
| Atomic file writes (no corruption) | ✅ Write-to-temp, rename |
| Optuna-native implementation | ✅ Uses callbacks API |
| WATCHER can evaluate mid-run | ✅ New "wait" action |
| Crash recovery possible | ✅ Best-so-far preserved |
| Team Beta principles followed | ✅ Immutable structure, configurable |

---

## 9. Approval

| Team | Status | Date | Notes |
|------|--------|------|-------|
| Team Alpha | ✅ Proposed | 2026-01-18 | Root cause identified, fix designed |
| Team Beta | ⏳ Pending | | |
| Michael | ⏳ Pending | | |

---

## 10. Implementation Priority

**HIGH** - This bug causes complete data loss on any interrupted optimization run. The 2-hour failure today lost 21 trials worth of work and 1.8M accumulated survivors.

---

**END OF PROPOSAL v1.0.0**
