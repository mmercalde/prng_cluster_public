# PROPOSAL: Step 7 Post-Pipeline Analysis & File Validation Fix

**Version:** 1.1.0  
**Date:** 2026-01-18  
**Author:** Team Alpha  
**Status:** DRAFT - Awaiting Team Beta Review

---

## Executive Summary

This proposal addresses two issues discovered during Chapter 13 testing:

1. **Critical Bug:** Empty files (2 bytes) pass `file_exists` validation, causing silent pipeline failures
2. **Missing Capability:** No structured export of visualization data for external ML tools

**Proposed Solution:** Add content validation to WATCHER evaluation and introduce Step 7 as a lightweight post-pipeline export phase.

**Key Change from v1.0.0:** Removed all overlap with Chapter 13. Step 7 is now **export-only** - no hit rate measurement, no LLM interpretation, no drift analysis (Chapter 13 handles all of these).

---

## Part 1: File Validation Fix

### 1.1 Problem Statement

Current `file_exists` evaluation in `watcher_agent.py` only checks `os.path.exists()`:

```python
# Current (BROKEN)
if os.path.exists(filepath):
    return True, "File exists"
```

**Failure Case (2026-01-18):**
```
-rw-rw-r-- 1 michael michael 2 Jan 18 14:03 bidirectional_survivors.json
```

A 2-byte file (`[]`) passed validation, causing Steps 2-6 to run on empty data. Pipeline reported "success" with zero predictions generated.

### 1.2 Proposed Fix

Add minimum file size validation with step-specific thresholds:

```python
# watcher_agent.py - Enhanced file_exists evaluation

FILE_SIZE_THRESHOLDS = {
    # Step 1 outputs
    "optimal_window_config.json": 500,
    "bidirectional_survivors.json": 100,
    "train_history.json": 500,
    "holdout_history.json": 500,
    
    # Step 2 outputs
    "optimal_scorer_config.json": 200,
    
    # Step 3 outputs
    "survivors_with_scores.json": 1000,
    
    # Step 5 outputs
    "models/reinforcement/best_model.json": 1000,
    "models/reinforcement/best_model.meta.json": 200,
    
    # Step 6 outputs
    "results/predictions/predictions_*.json": 500,
    
    # Default
    "_default": 100
}

def evaluate_file_exists(filepath: str) -> Tuple[bool, str]:
    """Enhanced file existence check with size validation."""
    
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    size = os.path.getsize(filepath)
    threshold = FILE_SIZE_THRESHOLDS.get(
        os.path.basename(filepath),
        FILE_SIZE_THRESHOLDS["_default"]
    )
    
    if size < threshold:
        return False, f"File too small: {size} bytes (min: {threshold})"
    
    # Validate JSON structure for .json files
    if filepath.endswith('.json'):
        try:
            with open(filepath) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) == 0:
                return False, "JSON file contains empty array"
            if isinstance(data, dict) and len(data) == 0:
                return False, "JSON file contains empty object"
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
    
    return True, f"File valid: {size} bytes"
```

### 1.3 Implementation Effort

| Task | Estimate |
|------|----------|
| Add `FILE_SIZE_THRESHOLDS` constant | 10 min |
| Modify `evaluate_file_exists()` | 20 min |
| Add JSON structure validation | 15 min |
| Test with empty/valid files | 15 min |
| **Total** | **~1 hour** |

---

## Part 2: Step 7 - Post-Pipeline Export

### 2.1 Purpose (Revised Scope)

**Step 7 is EXPORT ONLY.** It produces data artifacts for:

1. **External ML tools** - Feature matrices, correlations in NPZ/CSV format
2. **Visualization dashboards** - Static chart images
3. **Chapter 13 baseline** - Snapshot for drift comparison

### 2.2 What Step 7 Does NOT Do

These are handled by **Chapter 13** (already implemented):

| Capability | Owner | Notes |
|------------|-------|-------|
| Hit rate measurement | Chapter 13 | `post_draw_diagnostics.json` |
| Confidence calibration | Chapter 13 | `confidence_calibration` section |
| LLM interpretation | Chapter 13 | `chapter_13_llm_advisor.py` |
| Feature drift analysis | Chapter 13 | `feature_diagnostics` section |
| Retrain triggers | Chapter 13 | `chapter_13_triggers.py` |

### 2.3 Architecture Position

```
Steps 1-6 (Existing)
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Post-Pipeline Export (NEW)                             │
│                                                                 │
│  Purpose: Export data for external tools & Chapter 13 baseline  │
│                                                                 │
│  Outputs:                                                       │
│    • analysis/feature_importance.json    (ML-ready)             │
│    • analysis/correlation_matrix.npz     (ML-ready)             │
│    • analysis/survivor_features.npz      (ML-ready)             │
│    • analysis/baseline_snapshot.json     (For Chapter 13)       │
│    • analysis/charts/*.png               (Visualization)        │
│                                                                 │
│  NO LLM calls. NO hit rate analysis. Just export.               │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
Chapter 13: Live Feedback Loop
  (Uses baseline_snapshot.json for drift detection)
  (Handles all hit rate, confidence, LLM analysis)
```

### 2.4 Output Specifications

#### 2.4.1 feature_importance.json (ML-ready)

```json
{
  "timestamp": "2026-01-18T14:30:00Z",
  "model_type": "xgboost",
  "features": [
    {"name": "residue_8_match_rate", "importance": 0.182, "category": "residue"},
    {"name": "skip_entropy", "importance": 0.156, "category": "skip"}
  ],
  "total_features": 62,
  "top_10_cumulative": 0.65
}
```

#### 2.4.2 correlation_matrix.npz (ML-ready)

```python
# NumPy archive - loadable by any ML framework
{
    "matrix": np.array(...),       # (62, 62) correlation matrix
    "feature_names": [...],        # List of 62 feature names
}
```

#### 2.4.3 survivor_features.npz (ML-ready)

```python
# For external clustering, PCA, t-SNE analysis
{
    "X": np.array(...),            # (n_survivors, n_features)
    "seeds": np.array(...),        # Seed IDs
    "scores": np.array(...),       # Quality scores
    "feature_names": [...]         # Column labels
}
```

#### 2.4.4 baseline_snapshot.json (For Chapter 13)

```json
{
  "created_at": "2026-01-18T14:30:00Z",
  "pipeline_run_id": "step6_20260118_143000",
  "model": {
    "type": "xgboost",
    "r2_score": -0.08,
    "signal_quality": "weak"
  },
  "feature_summary": {
    "top_10": ["residue_8_match_rate", "skip_entropy", "..."],
    "top_10_cumulative": 0.65
  },
  "predictions": {
    "count": 20,
    "mean_confidence": 0.72
  },
  "survivors": {
    "count": 752,
    "score_mean": 0.45
  }
}
```

Chapter 13's `chapter_13_diagnostics.py` can compare current state against this baseline to detect drift.

#### 2.4.5 charts/*.png (Visualization)

Generated from existing `web_dashboard.py` chart functions:

| Chart | Source Function | Output |
|-------|-----------------|--------|
| Feature importance bar | `generate_feature_importance_chart()` | `importance_bar.png` |
| Correlation heatmap | `generate_heatmap_plotly()` | `correlation_heatmap.png` |
| Radar chart | `generate_radar_chart()` | `feature_radar.png` |
| Convergence plot | `generate_convergence_plotly()` | `convergence.png` |

### 2.5 Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `post_pipeline_export.py` | ~200 | Main Step 7 script |
| `ml_export.py` | ~150 | NPZ/CSV export utilities |
| `agent_manifests/post_pipeline_export.json` | ~30 | Agent manifest |

**Note:** No `chart_to_llm.py` needed - Chapter 13's `chapter_13_llm_advisor.py` handles LLM interpretation.

### 2.6 CLI Interface

```bash
# Run as Step 7 (after Step 6)
python3 post_pipeline_export.py \
    --survivors survivors_with_scores.json \
    --model-meta models/reinforcement/best_model.meta.json \
    --predictions results/predictions/predictions_latest.json \
    --output-dir analysis/

# Options
  --charts-only        Only generate chart images
  --export-only        Only export ML data (no charts)
  --skip-baseline      Skip baseline snapshot generation
```

### 2.7 Integration with Chapter 13

Chapter 13 uses Step 7's `baseline_snapshot.json` for drift detection:

```python
# In chapter_13_diagnostics.py

def compute_drift():
    """Compare current metrics to Step 7 baseline."""
    
    baseline_path = "analysis/baseline_snapshot.json"
    if not os.path.exists(baseline_path):
        return None  # No baseline yet, skip drift calculation
    
    baseline = load_json(baseline_path)
    current = compute_current_metrics()
    
    drift = {
        "feature_shift": compare_top_features(
            baseline["feature_summary"]["top_10"],
            current["feature_summary"]["top_10"]
        ),
        "survivor_count_change": (
            current["survivors"]["count"] - baseline["survivors"]["count"]
        ) / baseline["survivors"]["count"],
        "confidence_change": (
            current["predictions"]["mean_confidence"] - 
            baseline["predictions"]["mean_confidence"]
        )
    }
    
    return drift
```

### 2.8 Implementation Effort

| Task | Estimate |
|------|----------|
| `post_pipeline_export.py` | 2 hours |
| `ml_export.py` | 1.5 hours |
| Agent manifest | 20 min |
| WATCHER integration | 30 min |
| Chapter 13 baseline integration | 30 min |
| Testing | 1 hour |
| **Total** | **~6 hours** |

**Reduced from 12 hours** (v1.0.0) by removing LLM interpretation and duplicate analytics.

---

## Part 3: Implementation Priority

### Phase 1: File Validation Fix (Critical)
- Prevents silent failures
- ~1 hour implementation
- **Deploy immediately**

### Phase 2: ML Export + Baseline (High Value)
- Enables external ML analysis
- Provides Chapter 13 baseline
- ~4 hours implementation

### Phase 3: Chart Export (Enhancement)
- Dashboard visualizations
- ~2 hours implementation

---

## Part 4: Questions for Team Beta

1. **File Validation Thresholds:** Are the proposed thresholds reasonable? Should they be configurable in `distributed_config.json`?

2. **Step 7 Triggering:** Should Step 7 run automatically after Step 6, or be opt-in via `--end-step 7`?

3. **Export Formats:** Are JSON/NPZ/CSV sufficient, or should we add Parquet for larger datasets?

4. **Chapter 13 Dependency:** Should Chapter 13 require Step 7 baseline, or work without it (degraded mode)?

5. **Chart Selection:** Which of the 14 existing chart types should Step 7 export by default?

---

## Part 5: Summary of Changes from v1.0.0

| Component | v1.0.0 | v1.1.0 (This Version) |
|-----------|--------|------------------------|
| Hit rate measurement | Included | **Removed** (Ch13 handles) |
| Confidence calibration | Included | **Removed** (Ch13 handles) |
| LLM interpretation | Included | **Removed** (Ch13 handles) |
| Feature drift analysis | Included | **Removed** (Ch13 handles) |
| ML export (NPZ/CSV) | Included | **Kept** |
| Chart generation | Included | **Kept** |
| Baseline snapshot | Included | **Kept** |
| `chart_to_llm.py` | Proposed | **Removed** |
| Estimated effort | 12 hours | **6 hours** |

---

## Appendix A: Relationship to Chapter 13

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 7 (Export Only)                         │
│                                                                 │
│  • feature_importance.json                                      │
│  • correlation_matrix.npz                                       │
│  • survivor_features.npz                                        │
│  • baseline_snapshot.json ──────────────────────┐               │
│  • charts/*.png                                 │               │
└─────────────────────────────────────────────────│───────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CHAPTER 13 (Analysis)                        │
│                                                                 │
│  Uses baseline for drift detection:                             │
│    • Compare current feature importance to baseline             │
│    • Compare current survivor count to baseline                 │
│    • Compare current confidence to baseline                     │
│                                                                 │
│  Handles ALL analytics:                                         │
│    • Hit rate measurement (post_draw_diagnostics.json)          │
│    • Confidence calibration                                     │
│    • LLM interpretation (chapter_13_llm_advisor.py)             │
│    • Retrain triggers (chapter_13_triggers.py)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix B: File Validation Test Cases

```python
# test_file_validation.py

def test_empty_json_array():
    """Empty array should fail validation."""
    with open("test.json", "w") as f:
        f.write("[]")
    assert evaluate_file_exists("test.json") == (False, "JSON file contains empty array")

def test_empty_json_object():
    """Empty object should fail validation."""
    with open("test.json", "w") as f:
        f.write("{}")
    assert evaluate_file_exists("test.json") == (False, "JSON file contains empty object")

def test_small_file():
    """File below threshold should fail."""
    with open("test.json", "w") as f:
        f.write('{"a": 1}')  # 8 bytes
    assert evaluate_file_exists("test.json")[0] == False

def test_valid_file():
    """Valid file should pass."""
    with open("survivors.json", "w") as f:
        json.dump([{"seed": 123, "score": 0.5}] * 100, f)
    assert evaluate_file_exists("survivors.json")[0] == True
```

---

## Appendix C: Stale Files to Archive

These proposal files reference the old Qwen dual-model architecture (replaced by DeepSeek-R1-14B + Claude Opus in commit `07bfd79`, Jan 7 2026):

- `PROPOSAL_Schema_v1_0_4_Dual_LLM_Architecture.md`
- `PROPOSAL_Schema_v1_0_4_Dual_LLM_Architecture.txt`

**Recommendation:** Move to `archive/` or add "SUPERSEDED" header.

---

*End of Proposal v1.1.0*
