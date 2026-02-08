# Session Changelog — 2026-02-07 (Session 64)

**Focus:** Schema Regression & Recovery  
**Duration:** ~4 hours  
**Outcome:** Reverted to known-good state, Step 1 re-running

---

## Critical Incident: Schema Regression

### What Happened

Claude introduced a schema regression that broke the Step 1 → Step 2 contract. The `bidirectional_survivors.json` file was populated with raw integers instead of enriched 22-field dicts.

### Root Cause Analysis

Claude misdiagnosed a "two-mode gap" that didn't exist and attempted to fix it by:

1. Adding `bidirectional_survivors: list = None` field to `TestResult` dataclass in `window_optimizer.py`
2. Populating it with `list(bidirectional_constant)` — raw integers from set intersection
3. This bypassed the accumulator path which correctly builds enriched dicts

### The Actual Architecture (Correct)

```
Bayesian Optimization
  ├── Trial 1 → survivors added to accumulator (enriched dicts)
  ├── Trial 2 → survivors added to accumulator (enriched dicts)
  ├── Trial N → survivors added to accumulator (enriched dicts)
  └── END → deduplicate accumulator → write bidirectional_survivors.json ✅
```

The accumulator at lines 249-255 of `window_optimizer_integration_final.py` already builds enriched dicts:
```python
for seed in bidirectional_constant:
    accumulator['bidirectional'].append({'seed': seed, **metadata_constant})
```

**TestResult is only for per-trial metrics reporting to Optuna, NOT for survivor data flow.**

### Impact

| Component | Impact |
|-----------|--------|
| `bidirectional_survivors.json` | Corrupted (raw ints instead of dicts) |
| `convert_survivors_to_binary.py` | Would fail (expects dict keys) |
| Step 2+ pipeline | Broken |
| Time wasted | ~4 hours |

---

## Recovery Actions

### 1. Surgical Revert

Reverted only Step 1 files to commit `0b4e8f6` (last known-good):

```bash
git checkout 0b4e8f6 -- window_optimizer.py window_optimizer_integration_final.py
```

### 2. Commit Revert

```bash
git commit -m "revert: Restore Step 1 to known-good state (pre-schema regression)

Reverts window_optimizer.py and window_optimizer_integration_final.py to 0b4e8f6.

Root cause: bidirectional_survivors was incorrectly populated with raw integers
instead of enriched dicts, breaking Step 1 → Step 2 schema contract.

This revert restores the canonical survivor schema (list[dict] with 'seed' key).
Clean fix will be applied separately after baseline verification."
```

Commit: `ac1e51e`

### 3. Re-run Step 1

Deleted corrupted output files and re-ran Step 1 through WATCHER:

```bash
rm -f optimal_window_config.json bidirectional_survivors.json
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 --no-llm
```

---

## Mistakes Made (Full Accountability)

| # | Mistake | Impact |
|---|---------|--------|
| 1 | Diagnosed non-existent "two-mode gap" | Invented a problem |
| 2 | Added TestResult field without understanding data flow | Schema corruption |
| 3 | Populated field with raw integers instead of dicts | Broke downstream |
| 4 | Told user "architecture is correct" when asked to review | Delayed fix |
| 5 | Proposed manual fix instead of revert | More wasted time |
| 6 | Suggested running LLM alongside GPU cluster | Would cause failures |
| 7 | Failed to research project files before giving commands | Repeated errors |
| 8 | Did not create session number at start | Poor tracking |

---

## Lessons Learned

### 1. Understand Before Changing

The accumulator path was the correct architecture all along. TestResult is for metrics, not data flow.

### 2. Revert First

When state is contaminated by multiple fix attempts, revert to known-good baseline before attempting any fix.

### 3. Research Before Answering

Always check project files, chat history, and documentation before proposing solutions.

### 4. Schema Boundaries Are Sacred

Changes that affect data shape between pipeline steps must be treated as high-risk modifications requiring careful review.

---

## Feature Clarification: Auto-Consolidation NOT Needed

The "auto-consolidation" feature Claude was trying to add is **unnecessary**:

- Bayesian optimization already generates survivors via the accumulator
- Survivors are written at the END of optimization (lines 548-571)
- TestResult only reports metrics to Optuna
- No separate "consolidation" step is required

---

## Recovery Successful ✅

### First Attempt (Manifest Defaults) - FAILED

```bash
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 --no-llm
```

**Result:** 0 survivors, 2-byte file (empty `[]`), ESCALATE after 37 minutes.

**Root cause:** Manifest defaults had insufficient search space.

### Second Attempt (Custom Params) - SUCCESS

```bash
PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt --run-pipeline --start-step 1 --end-step 1 --params '{"trials":100,"max_seeds":50000}'
```

**Result:**

| Metric | Value |
|--------|-------|
| Bidirectional survivors | **24,628** |
| Total unique (deduplicated) | **48,896** |
| Best window | W3, O62, S[10-234] |
| Skip mode distribution | 44,502 constant / 4,394 variable |
| Time | 44:15 |
| WATCHER evaluation | ✅ PROCEED (confidence 1.00) |

### Schema Verification

```
Total survivors: 48896
Element type: <class 'dict'>
Keys: ['seed', 'window_size', 'offset', 'skip_min', 'skip_max', 'skip_range', 
       'sessions', 'trial_number', 'prng_base', 'skip_mode', 'prng_type', 
       'forward_count', 'reverse_count', 'bidirectional_count', 
       'bidirectional_selectivity', 'score', 'intersection_count', 
       'intersection_ratio', 'forward_only_count', 'reverse_only_count', 
       'survivor_overlap_ratio', 'intersection_weight']
✅ Schema is CORRECT (22 enriched fields)
```

### WATCHER Decision Log

```json
{"step": 1, "action": "proceed", "confidence": 1.0, "success": true, 
 "reasoning": "All required files valid", "timestamp": "2026-02-07T19:08:51"}
```

---

## Current State

| Item | Status |
|------|--------|
| `window_optimizer.py` | ✅ Reverted to 0b4e8f6 |
| `window_optimizer_integration_final.py` | ✅ Reverted to 0b4e8f6 |
| Step 1 execution | ✅ 48,896 survivors |
| Schema validation | ✅ 22 enriched fields |
| WATCHER state | ✅ PROCEED, ready for Step 2 |

---

## Timeline

| Time (Local) | Event | Result |
|--------------|-------|--------|
| ~09:00 | Session start, schema regression discovered | - |
| ~09:30 | Team Beta review, revert decision | - |
| 09:32 | Git revert to 0b4e8f6 | Commit ac1e51e |
| 10:24 | First re-run (manifest defaults) | ESCALATE (0 survivors) |
| 10:24 | Second re-run (100 trials, 50K seeds) | Started |
| 11:08 | Step 1 complete | ✅ PROCEED (48,896 survivors) |

---

## Parameter Discovery

Working configuration for synthetic data:
- **Trials:** 100 (Bayesian optimization)
- **Seeds:** 50,000 per trial
- **Total search:** 5M seed-trial combinations
- **Time:** ~44 minutes

This should be documented for future runs.

---

## Git Commands

```bash
cd ~/distributed_prng_analysis
git add docs/SESSION_CHANGELOG_20260207_S64.md
git commit -m "docs: Session 64 - Schema regression incident and recovery

INCIDENT:
- Claude introduced schema regression breaking Step 1 → Step 2 contract
- bidirectional_survivors.json had raw integers instead of enriched dicts

RECOVERY:
- Reverted window_optimizer.py and window_optimizer_integration_final.py to 0b4e8f6
- Re-ran Step 1 with params: trials=100, max_seeds=50000
- Result: 48,896 survivors with correct 22-field schema

WATCHER: PROCEED with confidence 1.0"

git push origin main
```

---

## Copy Command (ser8 → Zeus)

```bash
scp ~/Downloads/SESSION_CHANGELOG_20260207_S64.md rzeus:~/distributed_prng_analysis/docs/
```

---

**END OF SESSION 64**
