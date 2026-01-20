# Architectural Note: Step 7 Not Required for Autonomy

**Date:** 2026-01-18  
**Author:** Team Beta (Claude)  
**Status:** DECISION RECORDED  
**Replaces:** PROPOSAL_Step7_PostPipeline_Export_v1_1.md (draft)

---

## Executive Summary

After thorough code review, **Step 7 (Post-Pipeline Export) is NOT required** for Chapter 13 autonomy. Chapter 13 has built-in rolling baseline functionality that handles all drift detection and comparison needs.

Step 7 remains a **nice-to-have** for external analysis workflows (Jupyter, pandas) but is not on the critical path.

---

## The Original Assumption (Incorrect)

We assumed Chapter 13 needed an external baseline to compute drift metrics like:
- `entropy_change`
- `top_feature_turnover`
- `survivor_churn`

The proposed Step 7 would create `baseline_snapshot.json` after Step 6, which Chapter 13 would compare against.

---

## What the Code Review Revealed

### Finding 1: Rolling Baseline Built In

**File:** `chapter_13_diagnostics.py`

```python
# Line 60
PREVIOUS_DIAGNOSTICS = ".previous_diagnostics.json"

# Line 171-172
def load_previous_diagnostics() -> Optional[Dict[str, Any]]:
    """Load previous diagnostics for comparison."""
```

Chapter 13 maintains its own rolling baseline via `.previous_diagnostics.json`. Each diagnostic cycle:
1. Loads previous diagnostics (if exists)
2. Computes current metrics
3. Calculates deltas (drift)
4. Saves current as new `.previous_diagnostics.json`

### Finding 2: First-Run Handling

**Lines 431-432:**
```python
entropy_change = 0.0
turnover = 0.0
```

On first run (no previous diagnostics), drift metrics default to 0.0. This is correct behavior - there's nothing to compare against yet.

### Finding 3: Comparison Logic is Self-Contained

**Lines 435-478:**
```python
if previous_diagnostics:
    prev_feature = previous_diagnostics.get("feature_diagnostics", {})
    # ... compute turnover, entropy_change, etc.
```

All comparison logic checks `if previous_diagnostics:` before computing deltas. No external baseline file is needed.

### Finding 4: History Archival Exists

**Lines 851-878:**
```python
history_dir: str = DEFAULT_HISTORY_DIR  # "diagnostics_history"

# Create history directory
Path(history_dir).mkdir(exist_ok=True)

# Archive with timestamp
archive_path = os.path.join(history_dir, archive_name)
```

Chapter 13 already archives diagnostics to `diagnostics_history/` for audit trail and historical analysis.

---

## What Already Exists (Complete Inventory)

| Capability | File(s) | Status |
|------------|---------|--------|
| Feature importance (3 methods) | `feature_importance.py` | ✅ Complete |
| 13 chart visualizations | `feature_visualizer.py` | ✅ Complete |
| Web dashboard (14+ charts) | `web_dashboard.py` `/features` | ✅ Complete |
| LLM interpretation | `feature_importance_ai_interpreter.py` | ✅ Complete |
| WATCHER heuristics (6 steps) | `watcher_agent.py` | ✅ Complete |
| Chapter 13 diagnostics | `chapter_13_diagnostics.py` | ✅ Complete |
| Chapter 13 triggers | `chapter_13_triggers.py` | ✅ Complete |
| Chapter 13 LLM advisor | `chapter_13_llm_advisor.py` | ✅ Complete |
| Rolling baseline | `.previous_diagnostics.json` | ✅ Built in |
| History archival | `diagnostics_history/` | ✅ Built in |

---

## Step 7 Reclassification

### NOT Required For:
- Chapter 13 autonomy
- WATCHER decision-making
- Drift detection
- Retrain triggers
- LLM advisor context

### Would Be Useful For (Optional):
- **NPZ/CSV exports** for external ML tools (Jupyter notebooks, pandas DataFrames)
- **Point-in-time snapshots** for cross-run comparison outside the system
- **Standardized archive format** for long-term storage

---

## Recommendation

1. **Remove Step 7 from critical path** - It does not block Chapter 13 testing or autonomy
2. **Defer to "nice-to-have" backlog** - Implement when external analysis workflows are needed
3. **Update documentation** - Mark Step 7 as optional enhancement, not pipeline step
4. **Focus on actual blockers:**
   - File validation fix (2-byte empty files)
   - Chapter 13 testing (`--once` mode)
   - Hit rate measurement

---

## If Step 7 Is Ever Implemented

Scope should be minimal (~150 lines):

```python
# post_pipeline_export.py (OPTIONAL)

def export_for_external_analysis():
    """Export data for Jupyter/pandas workflows."""
    
    # Read existing outputs (don't recompute)
    importance = load_json("visualization_outputs/importance.json")
    survivors = load_json("survivors_with_scores.json")
    
    # Export formats for external tools
    export_npz(survivors, "exports/survivor_features.npz")
    export_csv(importance, "exports/feature_importance.csv")
    
    # Point-in-time snapshot (optional)
    create_snapshot("exports/snapshot_YYYYMMDD.json")
```

**Estimate:** 2 hours (reduced from original 6-hour estimate)

---

## Code References

| Location | What It Shows |
|----------|---------------|
| `chapter_13_diagnostics.py:60` | `PREVIOUS_DIAGNOSTICS` constant |
| `chapter_13_diagnostics.py:171-172` | `load_previous_diagnostics()` function |
| `chapter_13_diagnostics.py:431-432` | Default values for first run |
| `chapter_13_diagnostics.py:435-478` | Conditional drift computation |
| `chapter_13_diagnostics.py:851-878` | History archival logic |

---

## Conclusion

**Step 7 was solving a problem that doesn't exist.** Chapter 13 is architecturally complete and self-contained for autonomy purposes.

The correct priority order is:
1. ✅ File validation fix (prevents silent failures)
2. ✅ Chapter 13 testing (validate the loop works)
3. 🔲 Optional: Step 7 exports (when external analysis is needed)

---

*This note supersedes any previous proposals for Step 7 as a required pipeline step.*
