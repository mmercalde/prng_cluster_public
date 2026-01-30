# WATCHER Phase 1 Patch - Stale Output Prevention

**Version:** 1.0.0  
**Date:** 2026-01-27  
**Author:** Project Lead + Claude  
**Status:** PENDING TEAM BETA REVIEW  
**Target File:** `agents/watcher_agent.py`

---

## Problem Statement

WATCHER skipped Step 3 execution because:
1. A stale output file existed (`survivors_with_scores.json` from Jan 25)
2. `_evaluate_step_result()` only checks file existence, not freshness
3. Preflight failure (ramdisk missing) was logged but didn't prevent skip

Result: Pipeline continued with 35,453 survivors instead of 75,396 — **silent data corruption**.

---

## Design Principles

1. **Hard vs Soft Failures** — Not all preflight failures should block execution
2. **Timestamp Authority** — Output must be newer than all inputs to be valid
3. **Format Agnostic** — WATCHER never parses data files (NPZ-compatible)
4. **Graceful Degradation** — Run on available resources when possible

---

## Failure Classification

| Category | Examples | Action |
|----------|----------|--------|
| **HARD** | SSH unreachable, input file missing, no GPUs | ❌ BLOCK |
| **SOFT** | Ramdisk missing on some nodes, GPU count mismatch | ⚠️ WARN + CONTINUE |

---

## Code Changes

### Change 1: Add Constants (after imports, ~line 85)

```python
# =============================================================================
# PREFLIGHT FAILURE CLASSIFICATION (Phase 1 Patch - 2026-01-27)
# =============================================================================

PREFLIGHT_HARD_FAILURES = [
    "ssh", "unreachable", "connection refused", "connection timed out",
    "no such file", "input file missing", "no gpus available",
    "bidirectional_survivors_binary.npz not found",
    "train_history.json not found", "holdout_history.json not found"
]

PREFLIGHT_SOFT_FAILURES = [
    "ramdisk", "gpu count", "mismatch", "remediation failed", "degraded"
]


def classify_preflight_failure(failure_msg: str) -> str:
    """
    Classify preflight failure as HARD (block) or SOFT (warn).
    
    HARD = Cannot proceed safely (missing critical resources)
    SOFT = Can proceed with reduced capacity (graceful degradation)
    """
    msg_lower = failure_msg.lower()
    for keyword in PREFLIGHT_HARD_FAILURES:
        if keyword in msg_lower:
            return "HARD"
    return "SOFT"
```

---

### Change 2: Add Freshness Check Method (in WatcherAgent class, ~line 950)

```python
def _output_is_fresh(self, step: int) -> Tuple[bool, str]:
    """
    Check if output file is newer than all input files.
    
    Returns:
        (is_fresh: bool, reason: str)
        
    WATCHER must never skip a step if output is stale relative to inputs.
    This prevents silent data corruption from prior runs.
    """
    from datetime import datetime
    
    # Input -> Output mappings for each step
    STEP_IO_MAP = {
        1: {
            "inputs": ["synthetic_lottery.json"],
            "output": "optimal_window_config.json"
        },
        2: {
            "inputs": [
                "bidirectional_survivors_binary.npz",
                "train_history.json",
                "holdout_history.json"
            ],
            "output": "optimal_scorer_config.json"
        },
        3: {
            "inputs": [
                "bidirectional_survivors_binary.npz",
                "train_history.json",
                "holdout_history.json"
            ],
            "output": "survivors_with_scores.json"
        },
        4: {
            "inputs": ["survivors_with_scores.json"],
            "output": "reinforcement_engine_config.json"
        },
        5: {
            "inputs": [
                "survivors_with_scores.json",
                "train_history.json"
            ],
            "output": "models/reinforcement/best_model.meta.json"
        },
        6: {
            "inputs": [
                "models/reinforcement/best_model.meta.json",
                "forward_survivors.json"
            ],
            "output": "predictions/next_draw_prediction.json"
        }
    }
    
    io_map = STEP_IO_MAP.get(step)
    if not io_map:
        logger.warning(f"No IO mapping for step {step} - assuming fresh")
        return True, "Unknown step (no IO mapping)"
    
    output_file = io_map["output"]
    
    # Check output exists
    if not os.path.exists(output_file):
        return False, f"Output missing: {output_file}"
    
    output_mtime = os.path.getmtime(output_file)
    output_time_str = datetime.fromtimestamp(output_mtime).strftime("%Y-%m-%d %H:%M:%S")
    
    # Check each input
    for inp in io_map["inputs"]:
        if os.path.exists(inp):
            input_mtime = os.path.getmtime(inp)
            if input_mtime > output_mtime:
                input_time_str = datetime.fromtimestamp(input_mtime).strftime("%Y-%m-%d %H:%M:%S")
                return False, f"STALE: {output_file} ({output_time_str}) older than {inp} ({input_time_str})"
    
    return True, f"Fresh: {output_file} ({output_time_str})"
```

---

### Change 3: Modify `_run_preflight_check` (replace existing, ~line 953)

**BEFORE:**
```python
def _run_preflight_check(self, step: int) -> Tuple[bool, str]:
```

**AFTER:**
```python
def _run_preflight_check(self, step: int) -> Tuple[bool, str, bool]:
    """
    Run preflight checks before executing a step.
    
    Returns:
        (passed: bool, message: str, is_hard_failure: bool)
        
        - passed=True → All checks passed
        - passed=False, is_hard_failure=True → BLOCK execution (critical failure)
        - passed=False, is_hard_failure=False → WARN and continue (degraded mode)
    """
    if not PREFLIGHT_AVAILABLE:
        logger.debug("Preflight check not available - skipping")
        return True, "Preflight skipped (module not available)", False
    
    try:
        checker = PreflightChecker()
        result = checker.check_all(step)
        
        # DISPLAY FIX: Print preflight results visibly BEFORE Rich display
        print("\n" + "="*70)
        print(f"PREFLIGHT CHECK - Step {step}")
        print("="*70)
        print(result.summary())
        print("="*70 + "\n")
        
        # No failures = all good
        if not result.failures:
            return True, "All checks passed", False
        
        # Categorize failures
        hard_failures = []
        soft_failures = []
        
        for failure in result.failures:
            if classify_preflight_failure(failure) == "HARD":
                hard_failures.append(failure)
            else:
                soft_failures.append(failure)
        
        # Hard failures = BLOCK
        if hard_failures:
            msg = f"HARD FAILURE (blocking): {'; '.join(hard_failures)}"
            logger.error(msg)
            return False, msg, True
        
        # Soft failures = WARN + CONTINUE
        if soft_failures:
            msg = f"SOFT FAILURE (degraded mode): {'; '.join(soft_failures)}"
            logger.warning(msg)
            print(f"⚠️  {msg}")
            print("    Continuing with available resources...")
            return False, msg, False
        
        return True, "OK", False
        
    except Exception as e:
        logger.warning(f"Preflight check error (non-blocking): {e}")
        return True, f"Preflight error (proceeding anyway): {e}", False
```

---

### Change 4: Modify `_run_step` to Use Freshness Check (~line 1030)

**Find this section:**
```python
# PREFLIGHT CHECK (Team Beta Item A)
preflight_passed, preflight_msg = self._run_preflight_check(step)
if not preflight_passed:
    return {
        "success": False,
        "error": preflight_msg,
        "blocked_by": "preflight_check"
    }
```

**Replace with:**
```python
# PREFLIGHT CHECK (Team Beta Item A) - Phase 1 Patch
preflight_passed, preflight_msg, is_hard_failure = self._run_preflight_check(step)

# Hard failure = BLOCK immediately
if is_hard_failure:
    logger.error(f"Step {step} blocked by hard preflight failure")
    return {
        "success": False,
        "error": preflight_msg,
        "blocked_by": "preflight_hard_failure"
    }

# FRESHNESS CHECK - Never skip stale outputs (Phase 1 Patch)
is_fresh, freshness_msg = self._output_is_fresh(step)
if is_fresh:
    logger.info(f"Step {step}: {freshness_msg} - checking if skip is valid")
    # Output exists and is fresh - allow normal evaluation to decide skip
else:
    logger.warning(f"Step {step}: {freshness_msg} - will re-run")
    # Force re-run by deleting stale output
    # (Or just let it proceed - the step will overwrite)

# Soft failure = log warning, continue with available resources
if not preflight_passed and not is_hard_failure:
    logger.warning(f"Step {step} proceeding in degraded mode: {preflight_msg}")
```

---

### Change 5: Update Imports (if not present)

```python
from datetime import datetime  # Add if not present
```

---

## Testing Checklist

| Test Case | Expected Result |
|-----------|-----------------|
| Fresh output, all preflight pass | ✅ Skip step |
| Fresh output, soft preflight fail | ⚠️ Skip step (output valid) |
| Fresh output, hard preflight fail | ❌ Block step |
| Stale output, all preflight pass | 🔄 Run step |
| Stale output, soft preflight fail | 🔄 Run step (degraded) |
| Stale output, hard preflight fail | ❌ Block step |
| Missing output, any preflight | 🔄 Run step |

---

## Rollback Plan

If issues discovered:
```bash
cd ~/distributed_prng_analysis
git checkout agents/watcher_agent.py
```

---

## Files Affected

| File | Change Type |
|------|-------------|
| `agents/watcher_agent.py` | MODIFIED |

No other files require changes for Phase 1.

---

## Phase 2 Preview (Future)

- Step 3 sidecar metadata (`survivors_with_scores.meta.json`)
- Count validation (input survivors = output survivors)
- Hash validation (input file hash matches expected)

---

## Approval

| Reviewer | Status | Date |
|----------|--------|------|
| Team Beta | ⏳ PENDING | |
| Project Lead | ⏳ PENDING | |

---

**END OF PATCH DOCUMENT**
