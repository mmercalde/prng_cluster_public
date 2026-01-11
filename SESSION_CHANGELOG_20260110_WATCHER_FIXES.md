# Session Changelog: January 10, 2026
## Watcher Agent Step 2.5 & Step 3 Fixes

**Session Goal:** Fix WATCHER's inability to correctly execute and evaluate Steps 2.5 and 3

---

## Executive Summary

Steps 2.5 and 3 executed perfectly when run directly, but WATCHER failed due to:
1. Incorrect argument passing (shell scripts vs Python scripts)
2. Incorrect evaluation method (heuristic/LLM vs file-based)
3. Scripts rejecting unknown parameters from manifest `default_params`

All issues resolved. Both steps now pass through WATCHER autonomously.

---

## Issues & Fixes

### Issue 1: Step 2.5 Argument Style Mismatch
**Problem:** WATCHER passed `--key value` args to shell script expecting positional args.

**Fix:** Added `arg_style: positional` to manifest.

**File:** `agent_manifests/scorer_meta.json`
```json
{
  "arg_style": "positional",
  "positional_args": ["trials"],
  "flag_args": ["legacy_scoring"]
}
```

### Issue 2: Step 2.5 Wrong Evaluation Type
**Problem:** WATCHER used LLM/heuristic evaluation, but Step 2.5 is file-validated.

**Fix:** Added file-based evaluation config.

**File:** `agent_manifests/scorer_meta.json`
```json
{
  "evaluation_type": "file_exists",
  "success_condition": ["optimal_scorer_config.json"],
  "disable_llm_parsing": true,
  "disable_heuristic_parsing": true,
  "retry_policy": "none"
}
```

### Issue 3: WATCHER Missing File-Exists Evaluation Logic
**Problem:** WATCHER had no code path for `evaluation_type: file_exists`.

**Fix:** Added short-circuit evaluation in `evaluate_results()`.

**File:** `agents/watcher_agent.py` (lines 257-287)
```python
# Check for file-based evaluation (Step 2.5 style)
if manifest_path and os.path.exists(manifest_path):
    with open(manifest_path) as f:
        manifest_data = json.load(f)
    
    if manifest_data.get('evaluation_type') == 'file_exists':
        required_files = manifest_data.get('success_condition', [])
        if isinstance(required_files, str):
            required_files = [required_files]
        
        missing = [p for p in required_files if not Path(p).exists()]
        success = not missing
        
        decision = AgentDecision(
            success_condition_met=success,
            confidence=1.0 if success else 0.0,
            reasoning="All required files exist" if success else f"Missing required files: {missing}",
            recommended_action="proceed" if success else "escalate",
            parse_method="file_exists"
        )
        context = build_full_context(...)
        return decision, context
```

### Issue 4: Step 3 Wrong Script Mapping
**Problem:** `STEP_SCRIPTS[3]` pointed to job generator only, not full orchestration.

**Fix:** Changed to wrapper script.

**File:** `agents/watcher_agent.py`
```python
STEP_SCRIPTS = {
    ...
    3: "run_step3_full_scoring.sh",  # Was: "generate_full_scoring_jobs.py"
    ...
}
```

### Issue 5: Step 3 Python Script Rejects Unknown Args
**Problem:** `generate_full_scoring_jobs.py` used `parse_args()` which fails on unknown flags.

**Fix:** Changed to `parse_known_args()`.

**File:** `generate_full_scoring_jobs.py` (line 30)
```python
args, unknown = parser.parse_known_args()
if unknown:
    print(f"Note: Ignoring unknown args: {unknown}")
```

### Issue 6: Step 3 Shell Script Rejects Unknown Flags
**Problem:** `run_step3_full_scoring.sh` exited on unknown flags from manifest.

**Fix:** Changed to ignore and continue.

**File:** `run_step3_full_scoring.sh` (lines 78-81)
```bash
*)
    echo "Note: Ignoring unknown option: $1"
    shift
    ;;
```

### Issue 7: Step 3 Wrong Evaluation Type
**Problem:** WATCHER used heuristic evaluation, but Step 3 is file-validated.

**Fix:** Added file-based evaluation config.

**File:** `agent_manifests/full_scoring.json`
```json
{
  "evaluation_type": "file_exists",
  "success_condition": ["survivors_with_scores.json"],
  "disable_llm_parsing": true,
  "disable_heuristic_parsing": true,
  "retry_policy": "none"
}
```

---

## Architecture Pattern Established

| Step | Nature | Evaluation Type | Success Condition |
|------|--------|-----------------|-------------------|
| 2.5 | Shell wrapper (multi-phase) | `file_exists` | `optimal_scorer_config.json` |
| 3 | Shell wrapper (multi-phase) | `file_exists` | `survivors_with_scores.json` |
| 4+ | Python scripts | `heuristic` / `llm` | Metrics-based |

**Key Insight:** Shell-orchestrated multi-phase steps should use `file_exists` evaluation, not stdout parsing.

---

## Design Principle (Team Beta)

> Scripts must accept a superset of parameters and consume only what they understand.

This enables:
- Manifest configurability without script surgery
- Forward compatibility when new params are added
- Autonomous orchestration without contract drift

---

## Test Results

### Step 2.5
```
Parse Method: file_exists
Confidence:   1.00
Action:       PROCEED
Step 2: Scorer Meta-Optimizer  ✅  Done  score: 1.0000
```

### Step 3
```
Parse Method: file_exists
Confidence:   1.00
Action:       PROCEED
Step 3: Full Scoring  ✅  Done  score: 1.0000  Time: 0:01:33

Output: survivors_with_scores.json (101,592 survivors, 64 features each)
```

---

## Files Modified

| File | Changes |
|------|---------|
| `agents/watcher_agent.py` | File-exists evaluation logic, STEP_SCRIPTS[3] mapping, debug logging |
| `agent_manifests/scorer_meta.json` | arg_style, evaluation_type, success_condition |
| `agent_manifests/full_scoring.json` | evaluation_type, success_condition, args_map |
| `generate_full_scoring_jobs.py` | parse_known_args() |
| `run_step3_full_scoring.sh` | Ignore unknown flags |

---

*Session completed: January 10, 2026*
