# WATCHER Phase 1 Patch v1.1 — Stale Output Prevention

**Version:** 1.1.0  
**Date:** 2026-01-27  
**Authors:** Project Lead + Claude  
**Status:** PENDING TEAM BETA FINAL APPROVAL  
**Supersedes:** WATCHER_PHASE1_PATCH_FOR_REVIEW.md (v1.0)

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-27 | Initial draft |
| 1.1 | 2026-01-27 | Incorporated Team Beta corrections; manifest-derived IO |

---

## Problem Statement

WATCHER skipped Step 3 execution on 2026-01-27 because:

1. A stale output file existed (`survivors_with_scores.json` from Jan 25)
2. `_evaluate_step_result()` only checked file existence, not freshness
3. Preflight failure (ramdisk missing) was logged but didn't affect skip decision
4. Input had 75,396 survivors; stale output had 35,453 — **silent data corruption**

---

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Hard vs Soft Failures** | Not all preflight failures should block; degraded operation is acceptable |
| **Timestamp Authority** | Output must be newer than all inputs to be considered valid |
| **Manifest Authority** | IO paths derived from manifests, not hard-coded (single source of truth) |
| **Format Agnostic** | WATCHER never parses data files (NPZ-compatible) |
| **No Implicit Deletion** | WATCHER decides run/skip; never deletes artifacts |

---

## Team Beta Required Corrections (All Addressed)

| # | Requirement | Status | Solution |
|---|-------------|--------|----------|
| 1 | No hard-coded IO paths | ✅ | Added `required_inputs` / `primary_output` to manifests |
| 2 | Warn on degraded skip | ✅ | Explicit warning when skipping under soft failure |
| 3 | WATCHER never deletes | ✅ | Rule documented; no delete logic |
| 4 | Document limitation | ✅ | Freshness ≠ semantic correctness (Phase 2 adds sidecar) |

---

## Part 1: Manifest Schema Updates

### Discovery

Manifests already have `inputs` and `outputs` fields, but they mix parameter names with file paths. We add two new standardized fields for WATCHER freshness checking.

### New Fields

| Field | Type | Purpose |
|-------|------|---------|
| `required_inputs` | `string[]` | Actual file paths WATCHER checks for freshness |
| `primary_output` | `string` | Main output file WATCHER checks timestamp against |

### Manifest Updates

#### Step 1: `agent_manifests/window_optimizer.json`

Add after `"version"` line:

```json
  "required_inputs": [
    "synthetic_lottery.json"
  ],
  "primary_output": "optimal_window_config.json",
```

#### Step 2: `agent_manifests/scorer_meta.json`

Add after `"version"` line:

```json
  "required_inputs": [
    "bidirectional_survivors_binary.npz",
    "train_history.json",
    "holdout_history.json"
  ],
  "primary_output": "optimal_scorer_config.json",
```

#### Step 3: `agent_manifests/full_scoring.json`

Add after `"version"` line:

```json
  "required_inputs": [
    "bidirectional_survivors_binary.npz",
    "optimal_scorer_config.json",
    "train_history.json",
    "holdout_history.json"
  ],
  "primary_output": "survivors_with_scores.json",
```

#### Step 4: `agent_manifests/ml_meta.json`

Add after `"version"` line:

```json
  "required_inputs": [
    "optimal_window_config.json",
    "train_history.json"
  ],
  "primary_output": "reinforcement_engine_config.json",
```

#### Step 5: `agent_manifests/reinforcement.json`

Add after `"version"` line:

```json
  "required_inputs": [
    "survivors_with_scores.json",
    "train_history.json",
    "reinforcement_engine_config.json"
  ],
  "primary_output": "models/reinforcement/best_model.meta.json",
```

#### Step 6: `agent_manifests/prediction.json`

Add after `"version"` line:

```json
  "required_inputs": [
    "models/reinforcement/best_model.meta.json",
    "survivors_with_scores.json",
    "forward_survivors.json",
    "optimal_window_config.json"
  ],
  "primary_output": "predictions/next_draw_prediction.json",
```

---

## Part 2: WATCHER Code Changes

### Target File: `agents/watcher_agent.py`

---

### Change 1: Add Failure Classification Constants (~line 85, after imports)

```python
# =============================================================================
# PREFLIGHT FAILURE CLASSIFICATION (Phase 1 Patch v1.1 - 2026-01-27)
# =============================================================================
# 
# HARD failures = Cannot proceed safely (missing critical resources)
# SOFT failures = Can proceed with reduced capacity (graceful degradation)
#
# WATCHER RULES:
#   - WATCHER may refuse to skip (force re-run)
#   - WATCHER may allow step to overwrite existing output
#   - WATCHER must NEVER delete artifacts
# =============================================================================

PREFLIGHT_HARD_FAILURES = [
    "ssh", "unreachable", "connection refused", "connection timed out",
    "no such file", "input file missing", "no gpus available",
    "bidirectional_survivors_binary.npz not found",
    "train_history.json not found", "holdout_history.json not found",
    "primary input missing"
]

PREFLIGHT_SOFT_FAILURES = [
    "ramdisk", "gpu count", "mismatch", "remediation failed", "degraded"
]


def classify_preflight_failure(failure_msg: str) -> str:
    """
    Classify preflight failure as HARD (block) or SOFT (warn + continue).
    
    HARD = Cannot proceed safely (missing critical resources)
    SOFT = Can proceed with reduced capacity (graceful degradation)
    
    Returns: "HARD" or "SOFT"
    """
    msg_lower = failure_msg.lower()
    for keyword in PREFLIGHT_HARD_FAILURES:
        if keyword in msg_lower:
            return "HARD"
    return "SOFT"
```

---

### Change 2: Add Manifest IO Loader Method (in WatcherAgent class, ~line 920)

```python
def _get_step_io_from_manifest(self, step: int) -> Tuple[List[str], str]:
    """
    Get required inputs and primary output from step manifest.
    
    Returns:
        (required_inputs: List[str], primary_output: str)
        
    Raises:
        ValueError if manifest missing required fields
        
    This ensures WATCHER uses manifest as single source of truth for IO paths.
    (Team Beta Requirement #1: No hard-coded IO paths)
    """
    manifest_name = STEP_MANIFESTS.get(step)
    if not manifest_name:
        raise ValueError(f"No manifest defined for step {step}")
    
    manifest_path = os.path.join(self.config.manifests_dir, manifest_name)
    if not os.path.exists(manifest_path):
        raise ValueError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    required_inputs = manifest.get("required_inputs")
    primary_output = manifest.get("primary_output")
    
    if not required_inputs:
        raise ValueError(
            f"Manifest {manifest_name} missing 'required_inputs' field. "
            f"Phase 1 patch requires this field for freshness checking."
        )
    
    if not primary_output:
        raise ValueError(
            f"Manifest {manifest_name} missing 'primary_output' field. "
            f"Phase 1 patch requires this field for freshness checking."
        )
    
    return required_inputs, primary_output
```

---

### Change 3: Add Freshness Check Method (in WatcherAgent class, ~line 950)

```python
def _output_is_fresh(self, step: int) -> Tuple[bool, str]:
    """
    Check if output file is newer than all input files.
    
    Returns:
        (is_fresh: bool, reason: str)
        
    WATCHER must never skip a step if output is stale relative to inputs.
    This prevents silent data corruption from prior runs.
    
    NOTE: Freshness ≠ semantic correctness. Phase 2 will add sidecar
    validation for count/hash verification. (Team Beta Requirement #4)
    """
    from datetime import datetime
    
    try:
        required_inputs, primary_output = self._get_step_io_from_manifest(step)
    except ValueError as e:
        logger.error(f"Cannot check freshness: {e}")
        return False, str(e)
    
    # Check output exists
    if not os.path.exists(primary_output):
        return False, f"Output missing: {primary_output}"
    
    output_mtime = os.path.getmtime(primary_output)
    output_time_str = datetime.fromtimestamp(output_mtime).strftime("%Y-%m-%d %H:%M:%S")
    
    # Check each input - output must be newer than ALL inputs
    for inp in required_inputs:
        if os.path.exists(inp):
            input_mtime = os.path.getmtime(inp)
            if input_mtime > output_mtime:
                input_time_str = datetime.fromtimestamp(input_mtime).strftime("%Y-%m-%d %H:%M:%S")
                return False, (
                    f"STALE: {primary_output} ({output_time_str}) "
                    f"older than {inp} ({input_time_str})"
                )
        else:
            # Input doesn't exist - this is actually a hard failure
            # but freshness check isn't the place to enforce it
            logger.warning(f"Input file not found during freshness check: {inp}")
    
    return True, f"Fresh: {primary_output} ({output_time_str})"
```

---

### Change 4: Modify `_run_preflight_check` (replace existing, ~line 953)

**Find and replace the existing `_run_preflight_check` method:**

```python
def _run_preflight_check(self, step: int) -> Tuple[bool, str, bool]:
    """
    Run preflight checks before executing a step.
    
    Returns:
        (passed: bool, message: str, is_hard_failure: bool)
        
        - passed=True → All checks passed
        - passed=False, is_hard_failure=True → BLOCK execution (critical)
        - passed=False, is_hard_failure=False → WARN and continue (degraded)
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

### Change 5: Modify `_run_step` to Use New Logic (~line 1030)

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
# =========================================================================
# PREFLIGHT + FRESHNESS CHECK (Phase 1 Patch v1.1)
# =========================================================================

# 1. Run preflight check (returns 3-tuple now)
preflight_passed, preflight_msg, is_hard_failure = self._run_preflight_check(step)

# 2. Hard failure = BLOCK immediately (cannot proceed safely)
if is_hard_failure:
    logger.error(f"Step {step} blocked by hard preflight failure: {preflight_msg}")
    return {
        "success": False,
        "error": preflight_msg,
        "blocked_by": "preflight_hard_failure"
    }

# 3. Check output freshness (Team Beta: timestamp authority)
is_fresh, freshness_msg = self._output_is_fresh(step)
logger.info(f"Step {step} freshness: {freshness_msg}")

# 4. Decide: skip or run
if is_fresh:
    # Output exists and is fresh
    if not preflight_passed:
        # SOFT failure + fresh output = skip with warning (Team Beta Requirement #2)
        warning_msg = (
            f"⚠️  DEGRADED SKIP: Step {step} using existing output under soft preflight failure.\n"
            f"    Preflight issue: {preflight_msg}\n"
            f"    Output status: {freshness_msg}\n"
            f"    Recommend manual verification if results seem unexpected."
        )
        logger.warning(warning_msg)
        print(warning_msg)
    
    # Check if we should skip (existing evaluation logic)
    # ... (let existing _evaluate_step_result handle skip decision)
else:
    # Output missing or stale - MUST run
    logger.info(f"Step {step} will run: {freshness_msg}")
    # Soft failure here just means degraded mode, still proceed
    if not preflight_passed and not is_hard_failure:
        logger.warning(f"Step {step} running in degraded mode: {preflight_msg}")

# =========================================================================
```

---

### Change 6: Add Import (if not present)

At top of file with other imports:

```python
from datetime import datetime
```

---

## Part 3: Testing Checklist

| Test Case | Preflight | Output State | Expected Result |
|-----------|-----------|--------------|-----------------|
| Normal run | ✅ Pass | Missing | 🔄 Run step |
| Normal skip | ✅ Pass | Fresh | ✅ Skip step |
| Stale detection | ✅ Pass | Stale (old timestamp) | 🔄 Run step |
| Degraded run | ⚠️ Soft fail | Missing | 🔄 Run (degraded) |
| Degraded skip | ⚠️ Soft fail | Fresh | ⚠️ Skip + warning |
| Blocked | ❌ Hard fail | Any | ❌ Block step |
| Missing manifest field | Any | Any | ❌ Error + block |

---

## Part 4: Rollback Plan

If issues discovered after deployment:

```bash
# Revert watcher_agent.py
cd ~/distributed_prng_analysis
git checkout agents/watcher_agent.py

# Revert manifests (if needed)
git checkout agent_manifests/*.json
```

---

## Part 5: Files Affected

| File | Change Type | Description |
|------|-------------|-------------|
| `agents/watcher_agent.py` | MODIFIED | Add freshness check, preflight classification |
| `agent_manifests/window_optimizer.json` | MODIFIED | Add required_inputs, primary_output |
| `agent_manifests/scorer_meta.json` | MODIFIED | Add required_inputs, primary_output |
| `agent_manifests/full_scoring.json` | MODIFIED | Add required_inputs, primary_output |
| `agent_manifests/ml_meta.json` | MODIFIED | Add required_inputs, primary_output |
| `agent_manifests/reinforcement.json` | MODIFIED | Add required_inputs, primary_output |
| `agent_manifests/prediction.json` | MODIFIED | Add required_inputs, primary_output |

**Total: 7 files**

---

## Part 6: Limitations (Phase 2 Scope)

This patch addresses timestamp freshness only. It does NOT address:

| Limitation | Phase 2 Solution |
|------------|------------------|
| Survivor count mismatch | Sidecar metadata validation |
| Cross-run artifact reuse | Run-scoped directories |
| Parameter drift | Input hash verification |
| Semantic correctness | Sidecar count/hash fields |

**Freshness ≠ Semantic Correctness.** A fresh file may still be wrong if it was produced with different parameters. Phase 2 sidecar validation will address this.

---

## Part 7: Approval

| Reviewer | Status | Date | Notes |
|----------|--------|------|-------|
| Team Beta | ⏳ PENDING | | Corrections incorporated |
| Project Lead | ✅ APPROVED | 2026-01-27 | Ready for Team Beta review |

---

## Appendix: Quick Apply Commands

After Team Beta approval, apply with:

```bash
# 1. Update manifests (run from zeus:~/distributed_prng_analysis)
# See Part 1 for exact JSON to add to each file

# 2. Backup watcher_agent.py
cp agents/watcher_agent.py agents/watcher_agent.py.bak_phase1

# 3. Apply code changes from Part 2
# (Manual edit recommended for accuracy)

# 4. Test
PYTHONPATH=. python3 -c "
from agents.watcher_agent import WatcherAgent
w = WatcherAgent()
print('Manifest IO test:')
for step in [1,2,3,4,5,6]:
    try:
        inputs, output = w._get_step_io_from_manifest(step)
        print(f'  Step {step}: {len(inputs)} inputs → {output}')
    except Exception as e:
        print(f'  Step {step}: ERROR - {e}')
"

# 5. Run pipeline test
rm -f survivors_with_scores.json  # Force fresh run
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 3 --end-step 3
```

---

**END OF PATCH DOCUMENT v1.1**
