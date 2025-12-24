# Unified Agent Context Framework v3.2.8

**Document Version:** 3.2.8  
**Date:** December 24, 2025  
**Author:** Claude (AI Assistant)  
**Status:** PRODUCTION-READY  
**Supersedes:** v3.2.7  
**Patch Focus:** Step 6 Confidence Fix + Raw Scores Contract + Parent Run ID Lineage

---

## Changes from v3.2.7

| Section | Change |
|---------|--------|
| Part 10 | NEW: Step 6 Output Contract (raw_scores, confidence_scores, normalized) |
| Part 11 | NEW: Confidence Calibration (sigmoid z-score) |
| Part 12 | NEW: Parent Run ID Lineage Protocol |
| Part 13 | UPDATED: GlobalStateTracker Module (14 global features) |

---

## Critical Issues Addressed

### Issue 1: All Confidence = 1.0 (FIXED)

**Problem:** Step 6 `prediction_generator.py` displayed `confidence: 1.0000` for all predictions, making ranking meaningless.

**Root Cause Analysis:**
```python
# _build_prediction_pool() returns:
{
    "predictions": [521, 626, 415, ...],      # list of ints
    "confidence_scores": [0.347, 0.335, ...]  # list of floats (raw model output)
}

# generate_predictions() at line 250:
predictions_list = pool_result.get('predictions', [])  # Gets ints only!
# NOTE: pool_result['confidence_scores'] was NEVER read

# The buggy loop (lines 270-277):
for pred in predictions_list[:k]:        # pred = 521 (int)
    if isinstance(pred, dict):           # FALSE
        confidences.append(pred.get('confidence', 0.0))
    elif isinstance(pred, (int, float)): # TRUE - always hits this
        confidences.append(1.0 / len(predictions_list))  # = 0.1 for all

# Normalization makes it worse:
max_conf = max([0.1, 0.1, ...])  # = 0.1
confidences = [0.1/0.1, ...]     # = [1.0, 1.0, 1.0, ...]
```

**Solution:** Fixed extraction to read both arrays from pool_result, applied sigmoid z-score calibration.

### Issue 2: No Lineage Tracking (FIXED)

**Problem:** Step 6 outputs had no reference to which Step 5 training run produced the model.

**Solution:** 
- Auto-read `parent_run_id` from Step 5 sidecar (`best_model.meta.json → agent_metadata.run_id`)
- CLI override: `--parent-run-id`
- Stored in output: `agent_metadata.parent_run_id`

---

## Part 10: Step 6 Output Contract (NEW)

### 10.1 Output JSON Structure
```json
{
    "predictions": [521, 626, 415, 131, 26],
    "raw_scores": [0.127792, 0.108792, 0.057818, 0.057234, 0.056891],
    "confidence_scores": [0.7949, 0.6884, 0.3286, 0.3283, 0.3281],
    "confidence_scores_normalized": [1.0, 0.8513, 0.4524, 0.4478, 0.4451],
    "metadata": {
        "method": "dual_sieve",
        "pool_size": 10,
        "k": 5,
        "forward_count": 50,
        "reverse_count": 50,
        "intersection_count": 25,
        "prng_type": "java_lcg",
        "mod": 1000,
        "skip": 0,
        "dual_sieve": true,
        "gpu_available": true,
        "timestamp": "2025-12-24T09:54:53.973000",
        "score_stats": {
            "raw_min": 0.0000775,
            "raw_max": 0.127792,
            "raw_mean": 0.064026,
            "raw_std": 0.034064,
            "raw_unique": 10
        }
    },
    "agent_metadata": {
        "pipeline_step": 6,
        "pipeline_step_name": "prediction",
        "parent_run_id": "step5_20251223_171709",
        "confidence": 0.4937,
        "reasoning": "Generated 5 predictions with avg confidence 0.4937"
    }
}
```

### 10.2 Field Definitions

| Field | Type | Purpose | Consumer |
|-------|------|---------|----------|
| `raw_scores` | `List[float]` | Unmodified model output | Automation (cross-run comparison) |
| `confidence_scores` | `List[float]` | Calibrated 0-1 via sigmoid z-score | Gating, thresholds |
| `confidence_scores_normalized` | `List[float]` | Max-normalized for display | Human UI |
| `metadata.score_stats` | `Dict` | Debugging/monitoring | Watcher Agent |

### 10.3 Automation Contract

**For Watcher Agent / downstream automation:**
- Use `raw_scores` for cross-run comparability
- Use `confidence_scores` for threshold gating (e.g., "only act if confidence > 0.7")
- Use `score_stats.raw_std` to detect degenerate outputs (std ≈ 0 means model isn't discriminating)
- Use `score_stats.raw_unique` to detect constant outputs (unique = 1 is a bug)

---

## Part 11: Confidence Calibration (NEW)

### 11.1 Why Sigmoid Z-Score?

**Problem with max-normalization:**
- Top prediction always = 1.0, regardless of actual model confidence
- No cross-run comparability
- Masks low-variance outputs

**Sigmoid z-score approach:**
```python
# 1. Compute statistics
mean = sum(raw_scores) / n
std = sqrt(variance)

# 2. Convert to z-scores and apply sigmoid
def sigmoid(z):
    return 1.0 / (1.0 + exp(-z))

confidence = sigmoid((raw_score - mean) / std)
```

**Properties:**
- Scores above mean → confidence > 0.5
- Scores below mean → confidence < 0.5
- Preserves relative ranking
- Bounded [0, 1] without forcing top to 1.0
- Handles constant outputs gracefully (all → 0.5)

### 11.2 Constant Output Handling

If model outputs are constant (std ≈ 0):
- All confidences set to 0.5 (neutral)
- Warning logged: "Model outputs are constant; confidence set to 0.5 for all"
- `score_stats.raw_unique = 1` signals the issue

---

## Part 12: Parent Run ID Lineage Protocol (NEW)

### 12.1 Lineage Chain
```
Step 5 (Training)
├── Output: models/reinforcement/best_model.meta.json
│   └── agent_metadata.run_id: "step5_20251223_171709"
│
└──→ Step 6 (Prediction)
     ├── Input: reads best_model.meta.json
     ├── Extracts: agent_metadata.run_id
     └── Output: predictions_*.json
         └── agent_metadata.parent_run_id: "step5_20251223_171709"
```

### 12.2 Resolution Order

1. CLI argument: `--parent-run-id <value>` (highest priority)
2. Sidecar auto-read: `best_model.meta.json → agent_metadata.run_id`
3. None (warning logged, lineage incomplete)

### 12.3 CLI Usage
```bash
# Auto-read from sidecar (recommended)
python3 prediction_generator.py \
    --survivors-forward survivors.json \
    --lottery-history lottery.json \
    --models-dir models/reinforcement

# Manual override
python3 prediction_generator.py \
    --survivors-forward survivors.json \
    --lottery-history lottery.json \
    --models-dir models/reinforcement \
    --parent-run-id "step5_custom_20251224"
```

---

## Part 13: GlobalStateTracker Module (UPDATED)

### 13.1 Module Location

`models/global_state_tracker.py` - GPU-neutral, importable anywhere

### 13.2 14 Global Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | `global_lottery_mean` | Mean of all draws |
| 2 | `global_lottery_std` | Standard deviation |
| 3 | `global_lottery_min` | Minimum draw value |
| 4 | `global_lottery_max` | Maximum draw value |
| 5 | `global_lottery_median` | Median draw value |
| 6 | `global_lottery_skew` | Distribution skewness |
| 7 | `global_lottery_kurtosis` | Distribution kurtosis |
| 8 | `global_lottery_entropy` | Shannon entropy (SciPy fallback) |
| 9 | `global_lottery_range` | max - min |
| 10 | `global_lottery_iqr` | Interquartile range |
| 11 | `global_lottery_cv` | Coefficient of variation |
| 12 | `global_draw_count` | Total number of draws |
| 13 | `global_unique_ratio` | unique_values / total_draws |
| 14 | `global_repeat_ratio` | 1 - unique_ratio |

### 13.3 Feature Architecture
```
Total Features: 62 (when global features enabled)
├── Per-seed features: 48 (from survivor_scorer.py)
└── Global features: 14 (from GlobalStateTracker)
```

### 13.4 SciPy Fallback

If SciPy unavailable:
```python
def _numpy_entropy(values):
    """Pure numpy entropy calculation"""
    counts = np.bincount(values.astype(int))
    probs = counts[counts > 0] / len(values)
    return -np.sum(probs * np.log2(probs + 1e-10))
```

---

## Summary of Session 14-15 Changes

| Component | Before | After |
|-----------|--------|-------|
| Confidence output | All 1.0 | Differentiated (0.13 - 0.87) |
| Raw scores | Not preserved | `raw_scores` in output |
| Score stats | None | min/max/mean/std/unique |
| Lineage | No parent_run_id | Auto-read from sidecar |
| Intersection | Crashed on dicts | Type-tolerant |
| GlobalStateTracker | Duplicated in 3 files | Single module |

---

## Appendix A: Type-Tolerant Intersection

### A.1 The Problem

Production survivors can be:
- `List[int]` - just seed numbers: `[12345, 67890, ...]`
- `List[dict]` - scored survivors: `[{"seed": 12345, "features": {...}}, ...]`

Old code: `set(survivors)` → `TypeError: unhashable type: 'dict'`

### A.2 The Solution
```python
def _coerce_seed_list(self, items) -> List[int]:
    """Convert mixed list (int or dict with seed) to list of ints."""
    out = []
    for x in items or []:
        if isinstance(x, dict):
            if "seed" in x:
                out.append(int(x["seed"]))
        else:
            out.append(int(x))
    return out

def compute_dual_sieve_intersection(self, forward, reverse):
    # Coerce to seed lists first
    forward = self._coerce_seed_list(forward)
    reverse = self._coerce_seed_list(reverse)
    # Now safe to use set operations
    intersection = set(forward) & set(reverse)
    ...
```

---

## Appendix B: Test Verification

### B.1 Confidence Bug Fix Test
```
=== PREDICTIONS ===
1. 521 (confidence: 0.7949)   # Was 1.0000
2. 626 (confidence: 0.6884)   # Was 1.0000
3. 415 (confidence: 0.3286)   # Was 1.0000
4. 131 (confidence: 0.3283)   # Was 1.0000
5. 026 (confidence: 0.3281)   # Was 1.0000
```

### B.2 Raw Scores Verification
```json
{
    "raw_scores": [0.127792, 0.108792, 0.057818, 0.057234, 0.056891],
    "metadata": {
        "score_stats": {
            "raw_min": 0.0000775,
            "raw_max": 0.127792,
            "raw_std": 0.034064,
            "raw_unique": 10
        }
    }
}
```

### B.3 Parent Run ID Verification
```
INFO -   Parent run ID: step5_20251223_171709
INFO - Using sidecar parent_run_id: step5_20251223_171709

agent_metadata.parent_run_id: "step5_20251223_171709"
```

