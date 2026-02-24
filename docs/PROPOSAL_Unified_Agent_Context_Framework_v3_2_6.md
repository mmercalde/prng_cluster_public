# Unified Agent Context Framework v3.2.6

**Document Version:** 3.2.6  
**Date:** December 20, 2025  
**Author:** Claude (AI Assistant)  
**Status:** PRODUCTION-READY  
**Supersedes:** v3.2.5  
**Patch Focus:** Multi-Model Architecture v3.1.2, scripts_coordinator.py corrections (Addendums M, N)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v3.2.6 | 2025-12-20 | Added Addendum M (Multi-Model Architecture), Addendum N (scripts_coordinator corrections) |
| v3.2.5 | 2025-12-17 | Added Addendum L (Step 3 Pipeline Reference & Remote Rig Fixes) |
| v3.2.4 | 2025-12-13 | Added Addendum K (Feature Registry & Step 3 Improvements) |
| v3.2.3 | 2025-12-10 | Added Addendum J (Phase 5 Visualization) |
| v3.2.2 | 2025-12-10 | Added Addendum I (Phase 4 Agent Integration) |
| v3.2.1 | 2025-12-09 | Added Addendums G & H (Feature Importance) |
| v3.2.0 | 2025-12-04 | AI Agent Parameter Awareness |
| v3.1.0 | 2025-12-03 | Initial production release |

---

## Addendum Index

| Addendum | Title | Version | Status | Session |
|----------|-------|---------|--------|---------|
| A | BaseAgent Pattern | v1.0 | ✅ APPROVED | 1 |
| B | Manifest v1.3.0 | v1.0 | ✅ APPROVED | 4 |
| C | Centralized Config | v1.0 | ✅ APPROVED | 8 |
| D | Threshold Philosophy | v1.0 | ✅ APPROVED | 8 |
| E | SearchBounds Pattern | v1.0 | ✅ APPROVED | 8 |
| F | agent_metadata Injection | v1.1.0 | ✅ IMPLEMENTED | 9 |
| G | Model-Agnostic Feature Importance | v1.1.0 | ✅ IMPLEMENTED | 10 |
| H | Feature Importance Integration | v1.0.0 | ✅ IMPLEMENTED | 10 |
| I | Phase 4 Agent Integration | v1.0.0 | ✅ IMPLEMENTED | 11 |
| J | Phase 5 Visualization | v1.0.0 | ✅ IMPLEMENTED | 11 |
| K | Feature Registry & Step 3 Improvements | v1.0.0 | ✅ IMPLEMENTED | 12 |
| L | Step 3 Pipeline Reference & Remote Rig Fixes | v1.0.0 | ⚠️ CORRECTED in N | 14 |
| **M** | **Multi-Model ML Architecture v3.1.2** | **v1.0.0** | **✅ IMPLEMENTED** | **15** |
| **N** | **scripts_coordinator.py Universal Orchestrator** | **v1.0.0** | **✅ IMPLEMENTED** | **14-15** |

---

## ⚠️ IMPORTANT: Addendum L Corrections (v3.2.6)

**Addendum L in v3.2.5 references `coordinator.py` for Step 3 execution. This is OUTDATED.**

As of Session 14, `scripts_coordinator.py` v1.4.0 replaces `coordinator.py` for all ML script jobs (Steps 3, 4, 5, 6).

### Corrected Section 2.5 (Script Roles)

| Script | Location | Role | Called By |
|--------|----------|------|-----------|
| `run_step3_full_scoring.sh` | Zeus | Orchestrator | User |
| `generate_step3_scoring_jobs.py` | Zeus | Job generator | Shell script |
| **`scripts_coordinator.py`** | Zeus | **Distributed executor (CURRENT)** | Shell script |
| `coordinator.py` | Zeus | ⚠️ DEPRECATED (legacy) | - |
| `distributed_worker.py` | All nodes | Job router | Coordinator via SSH |
| `full_scoring_worker.py` | All nodes | Feature extractor | distributed_worker.py |
| `survivor_scorer.py` | All nodes | GPU batch processor | full_scoring_worker.py |

### Corrected Section 2.6 (Execution Flow)

```
scripts_coordinator.py (Zeus)    ← CORRECTED from coordinator.py
    │
    ├── Reads scoring_jobs.json (36 jobs)
    │
    ├── 3 Parallel Threads (one per node):
    │   │
    │   ├── Thread 1 (localhost): GPU0 → GPU1 (3s stagger)
    │   ├── Thread 2 (rig-6600): GPU0 → GPU1 → ... → GPU11 (0.5s stagger)
    │   └── Thread 3 (rig-6600b): GPU0 → GPU1 → ... → GPU11 (0.5s stagger)
    │
    ├── For each job:
    │   │
    │   ├── SSH to target node
    │   │   └── Execute: full_scoring_worker.py --gpu-id N ...
    │   │
    │   └── Success Detection (FILE-BASED, not stdout):
    │       ├── Output file exists
    │       ├── Size > 0
    │       └── Valid JSON
    │
    └── Write manifest: scripts_run_manifest.json
```

### Why This Change?

| Metric | coordinator.py | scripts_coordinator.py |
|--------|----------------|------------------------|
| Success Rate | 72% | **100%** |
| Code Lines | ~1700 | **~580** |
| Success Detection | stdout JSON (fragile) | **File-based (robust)** |

---

# Addendum M: Multi-Model ML Architecture v3.1.2

**Addendum Version:** 1.0.0  
**Date:** December 20, 2025  
**Status:** ✅ IMPLEMENTED  
**Session:** 15

---

## M.1 Executive Summary

Multi-Model Architecture v3.1.2 provides:

1. **CRITICAL BUG FIX:** Replaces synthetic random y-labels with real quality scores
2. **4 model types:** neural_net, xgboost, lightgbm, catboost
3. **Sidecar metadata:** `best_model.meta.json` for model identification
4. **Feature schema hashing:** Validates training/prediction consistency
5. **Streaming parsing:** Handles 813MB+ survivor files

---

## M.2 Critical Bug Fixed

**BEFORE (WRONG):** Training on random noise
```python
# OLD CODE - This was meaningless!
actual_quality = np.random.uniform(0.2, 0.8, len(survivors)).tolist()
```

**AFTER (CORRECT):** Training on real quality scores
```python
# NEW CODE - Real scores from Step 3
from models.feature_schema import load_quality_from_survivors
survivors, actual_quality, metadata = load_quality_from_survivors(args.survivors)
```

---

## M.3 Supported Model Types

| Model Type | Framework | Hardware | Extension |
|------------|-----------|----------|-----------|
| `neural_net` | PyTorch | All 26 GPUs (ROCm + CUDA) | `.pth` |
| `xgboost` | XGBoost | Zeus only (CUDA) | `.json` |
| `lightgbm` | LightGBM | Zeus only (CUDA) | `.txt` |
| `catboost` | CatBoost | Zeus both GPUs | `.cbm` |

---

## M.4 Command Line Interface

### Step 5 - Training

```bash
# Default (neural_net) - backward compatible
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data synthetic_lottery.json \
    --trials 50 --k-folds 5

# Specific model type
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data synthetic_lottery.json \
    --model-type xgboost \
    --trials 50

# Compare all 4 models
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data synthetic_lottery.json \
    --compare-models \
    --trials 50
```

**New Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-type` | `neural_net` | Model type: neural_net, xgboost, lightgbm, catboost |
| `--compare-models` | `False` | Train all 4 and select best |
| `--output-dir` | `models/reinforcement` | Where to save model + sidecar |

### Step 6 - Prediction

```bash
# Auto-loads model type from sidecar
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json \
    --lottery-history synthetic_lottery.json
```

**New Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--models-dir` | `models/reinforcement` | Directory with best_model.meta.json |

---

## M.5 Sidecar Metadata (best_model.meta.json)

The sidecar file is the **single source of truth** for model identification:

```json
{
  "schema_version": "3.1.2",
  "model_type": "xgboost",
  "checkpoint_path": "models/reinforcement/best_model.json",
  "checkpoint_format": "json",
  
  "feature_schema": {
    "source_file": "/path/to/survivors_with_scores.json",
    "feature_count": 50,
    "ordering": "lexicographic_by_key",
    "feature_schema_hash": "5026d8e9d692e009"
  },
  
  "y_label_source": {
    "field": "features.score",
    "observed_min": 0.275,
    "observed_max": 0.375,
    "normalization_method": "none",
    "warnings": []
  },
  
  "validation_metrics": {
    "mse": 0.05,
    "mae": 0.15,
    "rmse": 0.22
  },
  
  "training_info": {
    "n_trials": 50,
    "k_folds": 5,
    "model_type": "xgboost"
  }
}
```

**CRITICAL:** Step 6 loads model type from sidecar ONLY. File extensions are NEVER used.

---

## M.6 Feature Schema Validation

The feature schema hash prevents silent failures from schema drift:

```python
# Step 5 writes hash to sidecar
schema = get_feature_schema_with_hash(args.survivors)
# hash = "5026d8e9d692e009"

# Step 6 validates hash before prediction
if runtime_hash != expected_hash:
    raise ValueError("Feature schema mismatch - retrain required")
```

---

## M.7 File Structure

```
models/
├── __init__.py                    # Package exports
├── feature_schema.py              # Streaming schema + hash validation
├── gpu_memory.py                  # GPU memory reporting mixin
├── model_factory.py               # create/load/save functions
├── model_selector.py              # Model comparison
├── wrappers/
│   ├── __init__.py
│   ├── base.py                    # ModelInterface protocol
│   ├── neural_net_wrapper.py      # Wraps SurvivorQualityNet
│   ├── xgboost_wrapper.py         # XGBoost GPU
│   ├── lightgbm_wrapper.py        # LightGBM GPU
│   └── catboost_wrapper.py        # CatBoost multi-GPU
├── reinforcement/                 # Output directory
│   ├── best_model.pth             # (if neural_net)
│   ├── best_model.json            # (if xgboost)
│   ├── best_model.txt             # (if lightgbm)
│   ├── best_model.cbm             # (if catboost)
│   └── best_model.meta.json       # Sidecar (REQUIRED)
├── Qwen2.5-Coder-14B-*.gguf       # LLM models (separate system)
└── Qwen2.5-Math-7B-*.gguf         # LLM models (separate system)
```

**Note:** The `.gguf` files are for the Dual-LLM agent system (autonomous pipeline orchestration) and are completely separate from the ML model wrappers.

---

## M.8 Manifest Updates

### reinforcement.json (v1.4.0)

```json
{
  "agent_name": "reinforcement_agent",
  "version": "1.4.0",
  "outputs": [
    "models/reinforcement/best_model.pth",
    "models/reinforcement/best_model.meta.json",
    "models/reinforcement/training_history.json"
  ],
  "parameter_bounds": {
    "model_type": {
      "type": "categorical",
      "choices": ["neural_net", "xgboost", "lightgbm", "catboost"],
      "default": "neural_net"
    },
    "compare_models": {
      "type": "bool",
      "default": false
    },
    "output_dir": {
      "type": "string",
      "default": "models/reinforcement"
    }
  }
}
```

### prediction.json (v1.4.0)

```json
{
  "agent_name": "prediction_agent",
  "version": "1.4.0",
  "inputs": [
    "models/reinforcement/best_model.meta.json",
    "survivors_with_scores.json"
  ],
  "parameter_bounds": {
    "models_dir": {
      "type": "string",
      "default": "models/reinforcement"
    }
  }
}
```

---

## M.9 Programmatic Usage

```python
from models import create_model, load_model, ModelSelector

# Create a model
model = create_model('xgboost', config={'n_estimators': 200})

# Train
model.fit(X_train, y_train, X_val, y_val)

# Save with sidecar
from models.model_factory import save_model_with_sidecar
save_model_with_sidecar(model, 'models/reinforcement', feature_schema, y_metadata, ...)

# Load (type from sidecar)
from models.model_factory import load_model_from_sidecar
model, meta = load_model_from_sidecar('models/reinforcement')

# Compare multiple models
selector = ModelSelector()
results = selector.train_and_compare(X_train, y_train, X_val, y_val,
                                      model_types=['neural_net', 'xgboost'],
                                      metric='mse')
print(f"Best: {results['best_model']} ({results['best_score']:.4f})")
```

---

## M.10 Approval

| Role | Name | Date | Status |
|------|------|------|--------|
| Team Alpha | Claude | 2025-12-20 | ✅ Proposed |
| Team Beta | Claude | 2025-12-20 | ✅ Approved |
| Implementation | Claude | 2025-12-20 | ✅ Complete |
| Testing | Michael | 2025-12-20 | ✅ Passed |

---

**End of Addendum M v1.0.0 - Multi-Model ML Architecture v3.1.2 IMPLEMENTED**

---

# Addendum N: scripts_coordinator.py Universal Orchestrator

**Addendum Version:** 1.0.0  
**Date:** December 20, 2025  
**Status:** ✅ IMPLEMENTED  
**Session:** 14-15

---

## N.1 Executive Summary

**IMPORTANT:** As of Session 14, `scripts_coordinator.py` v1.4.0 replaces `coordinator.py` as the primary job orchestrator for Steps 3, 4, 5, and 6.

This addendum corrects references in Addendum L and documents the architectural shift.

---

## N.2 Architectural Change

### Before (coordinator.py)

```
coordinator.py (~1700 lines)
├── Complex stdout JSON parsing
├── 72% success rate under concurrency
├── SSH connection overload issues
└── Tightly coupled to job format
```

### After (scripts_coordinator.py)

```
scripts_coordinator.py (~580 lines)
├── File-based success detection (.done files)
├── 100% success rate
├── Sequential per-node, parallel across nodes
├── ML-agnostic (Steps 3, 4, 5, 6 compatible)
└── Clean, maintainable code
```

### Performance Comparison

| Metric | coordinator.py | scripts_coordinator.py |
|--------|----------------|------------------------|
| Success Rate | 72% (26/36) | **100% (36/36)** |
| Runtime | ~500s | **261s** |
| Code Lines | ~1700 | **~580** |
| Concurrency Model | Parallel all | Sequential/node, parallel/cluster |

---

## N.3 Updated Pipeline Flow

### Step 3: Full Scoring

```bash
# DEPRECATED
python3 coordinator.py --jobs-file scoring_jobs.json

# CURRENT (use this)
python3 scripts_coordinator.py --jobs-file scoring_jobs.json
```

### Step 5: Anti-Overfit Training

```bash
# Generate jobs
python3 generate_kfold_jobs.py --config reinforcement_engine_config.json

# Execute via scripts_coordinator
python3 scripts_coordinator.py \
    --jobs-file anti_overfit_jobs.json \
    --output-dir anti_overfit_results \
    --preserve-paths
```

### Step 6: Prediction

```bash
# Direct execution (no coordinator needed for single job)
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json \
    --lottery-history synthetic_lottery.json
```

---

## N.4 scripts_coordinator.py Usage

### Basic Usage

```bash
# Step 3 - Full Scoring (default)
python3 scripts_coordinator.py --jobs-file scoring_jobs.json

# Step 5 - Anti-Overfit Trials
python3 scripts_coordinator.py --jobs-file anti_overfit_jobs.json \
    --output-dir anti_overfit_results --preserve-paths

# Dry run (preview only)
python3 scripts_coordinator.py --jobs-file scoring_jobs.json --dry-run
```

### Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--jobs-file` | Required | Path to jobs JSON file |
| `--output-dir` | `full_scoring_results` | Base output directory |
| `--preserve-paths` | `False` | Don't rewrite paths (for Step 5) |
| `--dry-run` | `False` | Preview only, no execution |

---

## N.5 File-Based Success Detection

The key architectural improvement is **file-based success detection**:

```python
# Worker writes output file on success
output_path = Path(args.output_file)
with open(output_path, 'w') as f:
    json.dump(results, f)

# Coordinator checks for output file
if output_path.exists() and output_path.stat().st_size > 0:
    return JobResult.SUCCESS
```

This eliminates stdout JSON parsing failures under concurrency.

---

## N.6 coordinator_adapter.py

For backward compatibility, `coordinator_adapter.py` v2.0.0 bridges both formats:

```bash
# Automatically routes to correct coordinator
python3 coordinator_adapter.py --jobs-file scoring_jobs.json

# Detects job format and calls:
# - scripts_coordinator.py for script-based jobs
# - coordinator.py for legacy CuPy jobs (rare)
```

---

## N.7 Integration with Multi-Model Architecture

The scripts_coordinator.py works seamlessly with Multi-Model v3.1.2:

```bash
# Step 3: Score survivors (scripts_coordinator)
python3 scripts_coordinator.py --jobs-file scoring_jobs.json

# Step 5: Train model (direct, with model selection)
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data synthetic_lottery.json \
    --model-type xgboost

# Step 6: Generate predictions (direct, loads from sidecar)
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json
```

---

## N.8 Deprecation Notice

| Component | Status | Notes |
|-----------|--------|-------|
| `coordinator.py` | ⚠️ DEPRECATED | Use scripts_coordinator.py for script jobs |
| `run_step3_full_scoring.sh` (old) | ⚠️ DEPRECATED | Updated v2.0.0 available |
| Stdout JSON parsing | ❌ REMOVED | File-based detection only |

---

## N.9 Approval

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | Claude | 2025-12-20 | ✅ |
| Implementation | Claude | 2025-12-18 | ✅ |
| Documentation | Claude | 2025-12-20 | ✅ |
| Final Approval | Michael | 2025-12-20 | Pending |

---

**End of Addendum N v1.0.0 - scripts_coordinator.py Universal Orchestrator**

---

## Master Document Approval

| Role | Name | Date | Approval |
|------|------|------|----------|
| Author | Claude | 2025-12-09 | ✓ |
| Phase 2 Implementation | Claude | 2025-12-09 | ✓ |
| Phase 4 Implementation | Claude | 2025-12-10 | ✓ |
| Phase 5 Implementation | Claude | 2025-12-10 | ✓ |
| Session 12 Implementation | Claude | 2025-12-13 | ✓ |
| Session 14 Implementation | Claude | 2025-12-17 | ✓ |
| **Session 15 Implementation** | **Claude** | **2025-12-20** | **✓** |
| Syntax/Import Tests | Michael | 2025-12-10 | ✓ |
| Functional Test | Michael | 2025-12-20 | ✓ |
| Final Approval | Michael | 2025-12-20 | Pending |

---

**End of Unified Agent Context Framework v3.2.6**

---

## APPENDIX: Integration Note

This v3.2.6 document should be used in conjunction with v3.2.5. The v3.2.5 document contains:
- Parts 1-5 (Core Framework)
- Addendums A-L (Sessions 1-14)

This v3.2.6 document adds:
- Corrections to Addendum L references (coordinator.py → scripts_coordinator.py)
- Addendum M (Multi-Model Architecture v3.1.2)
- Addendum N (scripts_coordinator.py Universal Orchestrator)

To create a complete single document, append Addendums M and N to v3.2.5, and apply the Addendum L corrections noted in the "IMPORTANT" section above.
