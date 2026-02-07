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

## Current State

| Item | Status |
|------|--------|
| `window_optimizer.py` | ✅ Reverted to 0b4e8f6 |
| `window_optimizer_integration_final.py` | ✅ Reverted to 0b4e8f6 |
| Step 1 | 🔄 Running via WATCHER |
| Schema verification | ⏳ Pending Step 1 completion |

---

## Verification Commands (After Step 1 Completes)

```bash
# Verify schema is correct
python3 -c "
import json
d=json.load(open('bidirectional_survivors.json'))
print(f'Total survivors: {len(d)}')
if len(d) > 0:
    print(f'Element type: {type(d[0])}')
    if isinstance(d[0], dict):
        print(f'Keys: {list(d[0].keys())}')
        print('✅ Schema is CORRECT (enriched dicts)')
    else:
        print('❌ Schema is BROKEN (raw ints)')
"
```

---

## Git Commands

```bash
cd ~/distributed_prng_analysis
git add docs/SESSION_CHANGELOG_20260207_S64.md
git commit -m "docs: Session 64 - Schema regression incident report"
git push origin main
```

---

## Copy Command (ser8 → Zeus)

```bash
scp ~/Downloads/SESSION_CHANGELOG_20260207_S64.md rzeus:~/distributed_prng_analysis/docs/
```

---

**END OF SESSION 64**
