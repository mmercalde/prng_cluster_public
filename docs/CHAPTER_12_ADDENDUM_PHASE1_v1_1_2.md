# Chapter 12 Addendum: Phase 1 Freshness Check (v1.1.2)

**Date:** January 27, 2026  
**Version:** 1.1.2  
**Applies to:** WATCHER Agent, Agent Manifests

---

## Overview

Phase 1 adds **stale output detection** and **HARD/SOFT preflight classification** to the WATCHER Agent, preventing silent data corruption from outdated files.

---

## Problem Solved

**Bug:** WATCHER accepted stale `survivors_with_scores.json` (from Jan 25) when running Step 3 on Jan 27, causing the step to skip with incorrect data.

**Root Cause:** `_evaluate_step_result()` only checked file existence, not timestamps.

---

## New Module-Level Functions

Added to `agents/watcher_agent.py` after line 341:

### check_output_freshness(step: int) → tuple
```python
def check_output_freshness(step: int) -> tuple:
    """
    Check if output file is newer than all input files.
    Returns: (is_fresh: bool, reason: str, is_hard_failure: bool)
    
    NOTE: Freshness != semantic correctness. Phase 2 adds sidecar validation.
    """
```

**Behavior:**
- Returns `(True, "Fresh: ...", False)` if output newer than all inputs
- Returns `(False, "STALE: ...", False)` if any input newer than output
- Returns `(False, "HARD: ...", True)` if inputs missing or manifest invalid

### classify_preflight_failure(failure_msg: str) → str
```python
def classify_preflight_failure(failure_msg: str) -> str:
    """Classify preflight failure as HARD (block) or SOFT (warn + continue)."""
```

**Classification:**

| Type | Keywords | Action |
|------|----------|--------|
| HARD | ssh, unreachable, input missing, no gpus | ❌ Block execution |
| SOFT | ramdisk, gpu count, mismatch, degraded | ⚠️ Warn + continue |

### get_step_io_from_manifest(step: int) → tuple
```python
def get_step_io_from_manifest(step: int) -> tuple:
    """
    Get required inputs and primary output from step manifest.
    Returns: (required_inputs: List[str], primary_output: str)
    """
```

### resolve_repo_path(p: str) → str
```python
def resolve_repo_path(p: str) -> str:
    """Resolve path relative to REPO_ROOT (not os.getcwd())."""
```

---

## New Constants
```python
# Derived from __file__, not os.getcwd() - works under cron/systemd
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREFLIGHT_HARD_FAILURES = [
    "ssh", "unreachable", "connection refused", "connection timed out",
    "no such file", "input file missing", "no gpus available",
    "bidirectional_survivors_binary.npz not found",
    "train_history.json not found", "holdout_history.json not found",
    "primary input missing", "manifest missing"
]

PREFLIGHT_SOFT_FAILURES = [
    "ramdisk", "gpu count", "mismatch", "remediation failed", "degraded"
]
```

---

## Manifest Schema Extension

All 6 step manifests now include:
```json
{
  "required_inputs": ["file1.json", "file2.npz"],
  "primary_output": "output.json"
}
```

### Manifest IO Mapping

| Step | Manifest | required_inputs | primary_output |
|------|----------|-----------------|----------------|
| 1 | window_optimizer.json | synthetic_lottery.json | optimal_window_config.json |
| 2 | scorer_meta.json | npz, train, holdout | optimal_scorer_config.json |
| 3 | full_scoring.json | npz, config, train, holdout | survivors_with_scores.json |
| 4 | ml_meta.json | window_config, train | reinforcement_engine_config.json |
| 5 | reinforcement.json | scores, train, config | best_model.meta.json |
| 6 | prediction.json | model, scores, forward, config | next_draw_prediction.json |

---

## Modified _run_step() Behavior

### Before (v1.0)
```
Preflight failed? → Block (any failure)
Output exists? → Skip (no timestamp check)
```

### After (v1.1.2)
```
Preflight failed?
  → HARD failure? → Block
  → SOFT failure? → Warn + continue
Output fresh? (newer than all inputs)
  → Yes → Skip
  → No (stale/missing) → Run
```

---

## Integration Points

### In _run_step() (line ~1150)
```python
# PREFLIGHT CHECK with HARD/SOFT classification
preflight_passed, preflight_msg = self._run_preflight_check(step)
if not preflight_passed:
    failure_type = classify_preflight_failure(preflight_msg)
    if failure_type == "HARD":
        return {"success": False, "error": preflight_msg, "blocked_by": "preflight_hard_failure"}
    else:
        logger.warning(f"SOFT preflight failure (continuing): {preflight_msg}")

# FRESHNESS CHECK
is_fresh, freshness_msg, is_hard_freshness = check_output_freshness(step)

if is_hard_freshness:
    return {"success": False, "error": freshness_msg, "blocked_by": "freshness_hard_failure"}

if is_fresh:
    return {"success": True, "skipped": True, "reason": freshness_msg}
```

---

## Phase 2 (Future)

Phase 1 validates **timestamps only**. Phase 2 will add:
- Sidecar metadata files (`.meta.json`)
- Survivor count validation
- Input hash verification
- Parameter drift detection

---

## Verification Commands
```bash
# Test freshness check for all steps
python3 -c "
from agents.watcher_agent import check_output_freshness
for step in range(1, 7):
    fresh, msg, hard = check_output_freshness(step)
    print(f'Step {step}: {\"FRESH\" if fresh else \"STALE\"} - {msg[:60]}...')
"

# Validate manifest IO
python3 -c "
from agents.watcher_agent import get_step_io_from_manifest
for step in range(1, 7):
    ri, po = get_step_io_from_manifest(step)
    print(f'Step {step}: {len(ri)} inputs → {po.split(\"/\")[-1]}')
"
```

---

*End of Phase 1 Addendum*

---

## Addendum: Step 6 Output Path Fix (2026-01-28)

### Problem
Manifest expected `predictions/next_draw_prediction.json`, but script saved to `results/predictions/predictions_YYYYMMDD.json`.

### Team Beta Ruling: Option B
- **Canonical:** `predictions/next_draw_prediction.json` (WATCHER contract)
- **Archive:** `predictions/history/predictions_YYYYMMDD.json` (optional)

### Files Modified
- `prediction_generator.py` - `_save_predictions()` method
- `agent_manifests/prediction.json` - `outputs`, `success_condition`
