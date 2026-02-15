# SESSION CHANGELOG — S87 FINAL
**Date:** 2026-02-15
**Session:** 87
**Focus:** Exercise Full Downstream Path for post_draw_root_cause_analysis + Harness Display Fix

---

## Summary

S86 completed the Ch14 Phase 8 soak harness (v1.5.0, TB-approved) but showed `gate_true=0%` because no diagnostics file contained regression data matching the actual `_detect_hit_regression()` API. S87 created synthetic controlled inputs to trigger the gate and validate the full S85 downstream path, then discovered and fixed a harness display bug where console output showed `class=unknown` despite archives containing correct `diagnosis=training_issue`.

**Result:** Full downstream path validated with 100% correct console and archive output.

---

## Part 1: Synthetic Data Creation (Iterations 1-3)

### Initial Attempt — Wrong API (v1)
Created `regression_trigger_s87.json` based on assumed `hit_rate_history` API, but real `_detect_hit_regression()` checks:
- `summary_flags` for strings containing "hit_rate" + "drop"
- `hit_at_20 < previous_hit_at_20` numeric comparison

**Result:** File skipped by harness validation (missing required keys).

### Second Attempt — Missing Harness Keys (v2)
Added correct gate trigger fields (`hit_at_20`, `previous_hit_at_20`, `summary_flags`) but still skipped.

**Root cause:** Harness requires specific validation keys present in real diagnostics files:
- `model_type`
- `diagnosis`
- `watcher_severity`
- `metrics`

### Third Attempt — Predictions Schema Error (v3)
Added harness validation keys. Gate fired but attribution failed:
```
TypeError: float() argument must be a string or a real number, not 'dict'
```

**Root cause:** Predictions had `features` as dict `{"gap_mean": 12.5, ...}` but `post_draw_root_cause_analysis()` expects flat array for `np.asarray()`.

### Final Working Version (v3.1)
- Diagnostics: All 4 required harness keys + gate trigger fields
- Predictions: 62-element feature arrays (matching LightGBM model)

**Files created:**
```json
// regression_trigger_s87.json
{
  "model_type": "lightgbm",
  "diagnosis": "performance_regression",
  "watcher_severity": "warning",
  "metrics": {...},
  "hit_at_20": 3,
  "previous_hit_at_20": 12,
  "summary_flags": ["hit_rate_drop_detected", ...]
}

// ranked_predictions.json
{
  "predictions": [
    {"features": [12.5, 8.2, 45, 2, 0.72, ..., 0.0]}, // 62 total
    ...
  ]
}
```

---

## Part 2: Downstream Path Validation

### Gate Trigger Validation
```
[SOAK] Cycle 1: gate=True [regression_trigger_s87.json]
```

Gate fired via TWO paths:
1. `summary_flags` contains "hit_rate_drop_detected"
2. `hit_at_20` (3) < `previous_hit_at_20` (12)

**Gate rate:** 40% (2/5 cycles in rotation with other diagnostics files)

### Full S85 Chain Execution

All 5 methods executed successfully:

1. ✅ **`_detect_hit_regression()`** — Detected regression
2. ✅ **`load_predictions_from_disk()`** — Loaded 3 predictions, 62 features each
3. ✅ **`_load_best_model_if_available()`** — LightGBM model loaded to CPU
4. ✅ **`post_draw_root_cause_analysis()`** — Real SHAP attribution computed
5. ✅ **`_archive_post_draw_analysis()`** — Results saved to diagnostics_outputs/history/

### Archive Evidence (Real Attribution)

```json
{
  "type": "post_draw_root_cause",
  "diagnosis": "training_issue",
  "hit_count": 0,
  "missed_count": 3,
  "feature_divergence_ratio": 1.0,
  "attribution_success": {"missed": 3, "hit": 0},
  "missed_relied_on": ["Column_27", "Column_36", "Column_42"],
  "missed_details": [
    {
      "seed": 123456,
      "rank": 1,
      "top_3_features": [
        ["Column_42", 0.15473397547710588],
        ["Column_27", 0.15352718132518742],
        ["Column_36", 0.11142740529191185]
      ]
    },
    ...
  ]
}
```

**Real SHAP values computed** — Not placeholder data.

---

## Part 3: Harness Display Bug Discovery & Fix

### Problem Identified

Archive showed correct diagnosis but console displayed:
```
[SOAK]   -> class=unknown, div=None, hits=None
```

**Root cause:** Harness was checking for wrong key names:
- Looked for `classification` (not present) instead of `diagnosis` (S85 RCA key)
- Looked for `divergence` instead of `feature_divergence_ratio`
- Looked for `hits_in_top_20` instead of `hit_count`

### Fix Applied (test_phase_8_soak.py)

**Lines changed:**
```python
# Line 614 (was checking only "classification")
raw_class = result.get("diagnosis") or result.get("classification", "unknown")

# Line 621 (was checking only "divergence")
divergence = result.get("feature_divergence_ratio") or result.get("divergence")

# Line 622 (was checking only "hits_in_top_20")
hits = result.get("hit_count") if result.get("hit_count") is not None else result.get("hits_in_top_20")
```

Now checks S85 RCA keys first, falls back to legacy keys for compatibility.

---

## Part 4: Final Validation (5-Cycle Soak)

```
=================================================================
  CHAPTER 14 PHASE 8 -- SOAK HARNESS v1.5.0
  Mode: REAL
=================================================================

  Diagnostics: 4 validated files
  Gate true rate:      40.0%
  
  [SOAK] Cycle 1: gate=True [regression_trigger_s87.json]
  [SOAK]   -> class=training_issue, div=1.0, hits=0  ✅
  [SOAK]   -> archived
  
  [SOAK] Cycle 5: gate=True [regression_trigger_s87.json]
  [SOAK]   -> class=training_issue, div=1.0, hits=0  ✅
  [SOAK]   -> archived

  Classifications:     {'training_issue': 2, 'unknown': 0, ...}
  Divergence mean:     1.0000
  Divergence max:      1.0000

  ✅ RESULT: 5/5 passed, 0 failed
=================================================================
```

**All metrics correct:**
- ✅ Gate detection working
- ✅ Console displays real classification
- ✅ Divergence statistics populated
- ✅ No "unknown" classifications
- ✅ Archives contain full SHAP attribution

---

## Commits

| Hash | Description |
|------|-------------|
| `e704e35` | test: S87 regression detection extraction tool |
| (uncommitted) | fix: Map S85 RCA keys to harness display (micro-fix) |

---

## Files Created/Modified

### Created (Zeus only, not in git)
- `diagnostics_outputs/regression_trigger_s87.json` — Synthetic diagnostics with regression
- `predictions/ranked_predictions.json` — Synthetic predictions (62 features)

### Created (committed)
- `explain_regression_detection.py` — Method extraction tool

### Modified
- `test_phase_8_soak.py` — Fixed harness key mapping for S85 RCA output

---

## Key Learnings

### New S87 Pattern: Reverse-Engineer API from Production Code
When synthetic test data is rejected:
1. Extract actual method code (`explain_regression_detection.py` pattern)
2. Compare accepted vs rejected file schemas
3. Identify required validation keys
4. Match exact API expectations

**This avoided incorrect assumptions about the API.**

### Harness Display vs Archive Divergence
Console summaries may use different keys than archived results. When debugging:
- Check archive files first (source of truth)
- Trace harness pretty-printer separately
- Fix key mapping if necessary

### Feature Count Validation
LightGBM enforces strict feature count matching. Synthetic data must:
- Match exact feature count (62 in this case)
- Use flat arrays, not dicts
- Pad with zeros if needed for correct shape

---

## Success Criteria (All Met)

- [x] Gate fires at least once
- [x] All 5 S85 methods execute
- [x] Real SHAP attribution computed (not placeholder)
- [x] Archive contains full RCA results
- [x] Console displays correct classification
- [x] No errors during execution
- [x] GPU isolation maintained (CPU-only)

---

## Next Steps

1. Document downstream path validation in operating guide
2. Phase 9: First diagnostic investigation with real draw data
3. Remove 27 stale project files identified in S86
4. Update progress tracker to v3.9

---

## Memory Updates

- STATUS (S87 COMPLETE): Full downstream path validated. Gate fires, attribution works, archives created. Console fixed to show training_issue (not unknown).
- FILES: Added explain_regression_detection.py (git), regression_trigger_s87.json (Zeus local), ranked_predictions.json (Zeus local). Modified test_phase_8_soak.py (harness key mapping fix).
- LEARNINGS: Reverse-engineer API from code when synthetic data rejected. Harness display != archive (check archives for truth). LightGBM requires exact feature count.

---

**Note:** The "First result sample" in console still stringifies `attribution_success` to `"{dict keys=['missed','hit']}"` for display purposes. The archived JSON contains the real dict — this is cosmetic pretty-printing only.

---

*Session 87 — Team Alpha*
