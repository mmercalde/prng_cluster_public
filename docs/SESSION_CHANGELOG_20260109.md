# SESSION CHANGELOG - January 9, 2026

## Bug Fix: Watcher Agent Step 2.5 Script Mapping

**Problem:** `watcher_agent.py` line 138 had hardcoded `run_scorer_meta_optimizer.py` which uses `/shared/ml/` paths (non-existent NFS mount).

**Root Cause:** `STEP_SCRIPTS` dict (line 136-145) overrides manifest-defined scripts. The `.py` version was never updated to use PULL architecture.

**Fix:** Changed line 138 from `.py` to `.sh`:
```python
# Before
2: "run_scorer_meta_optimizer.py",

# After  
2: "run_scorer_meta_optimizer.sh",
```

**Why `.sh` is correct:**
- `run_scorer_meta_optimizer.sh` implements PULL architecture
- Uses `scripts_coordinator.py` (per Jan 3 fix)
- Workers write locally, coordinator pulls via SCP
- No shared filesystem required

**Documentation Gap Identified:**
- `STEP_SCRIPTS` dict not documented in CHAPTER_12
- Relationship between manifests and hardcoded scripts unclear
- Should consider: make watcher read scripts FROM manifests instead of hardcoded dict

## Files Changed

| File | Change |
|------|--------|
| `agents/watcher_agent.py` | Line 138: `.py` → `.sh` |
