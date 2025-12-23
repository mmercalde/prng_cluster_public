# Unified Agent Context Framework v3.2.7

**Document Version:** 3.2.7  
**Date:** December 22, 2025  
**Author:** Claude (AI Assistant)  
**Status:** PRODUCTION-READY  
**Supersedes:** v3.2.6  
**Patch Focus:** Subprocess Isolation + Model Checkpoint Persistence

---

## Changes from v3.2.6

| Section | Change |
|---------|--------|
| Part 6 | NEW: Subprocess Isolation Architecture for Step 5 |
| Part 7 | NEW: Model Checkpoint Persistence & Handoff |
| Part 8 | UPDATED: Agent Manifest for Step 5 (reinforcement.json v1.5.0) |
| Part 9 | NEW: Step 5 → Step 6 Handoff Protocol |

---

## Critical Issues Addressed

### Issue 1: OpenCL/CUDA GPU Conflict (NEW)

**Problem:** When running `--compare-models` in Step 5, LightGBM (OpenCL) fails with error -9999 if any CUDA model has run first. This is because:
- PyTorch/XGBoost/CatBoost use CUDA
- LightGBM uses OpenCL
- CUDA context blocks OpenCL initialization
- Single Python process cannot "un-initialize" CUDA

**Solution:** Subprocess isolation - each trial runs in a fresh Python interpreter with no inherited GPU context.

### Issue 2: Model Checkpoint Loss (NEW)

**Problem:** Subprocess trains model, returns metrics via JSON, exits → model object is garbage collected. Step 6 has no model to load.

**Solution:** Worker script saves checkpoint before exit, coordinator copies winner to output directory.

---

## Part 6: Subprocess Isolation Architecture (NEW)

### 6.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│           meta_prediction_optimizer_anti_overfit.py                  │
│                    (Main Process - NO GPU imports)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SubprocessTrialCoordinator                                         │
│  ├── Saves X_train, y_train, X_val, y_val to .npz (once)           │
│  ├── For each Optuna trial:                                        │
│  │   └── subprocess.run(train_single_trial.py ...)                 │
│  │       ├── Fresh Python interpreter                              │
│  │       ├── NO GPU context inherited                              │
│  │       ├── Imports only required ML library                      │
│  │       ├── Trains model → saves checkpoint                       │
│  │       ├── Returns JSON with metrics + checkpoint_path           │
│  │       └── Exits → GPU memory freed                              │
│  │                                                                  │
│  ├── Tracks best_result across all trials                          │
│  └── save_best_model() → copies winner to output_dir              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Key Files

| File | Version | Purpose |
|------|---------|---------|
| `train_single_trial.py` | v1.0.1 | Isolated worker (NO GPU imports at module level) |
| `subprocess_trial_coordinator.py` | v1.0.1 | Manages subprocess execution, tracks best model |
| `meta_prediction_optimizer_anti_overfit.py` | v2.0.0 | Main script with conditional CUDA init |

### 6.3 Worker Script Contract

**Input (CLI args):**
```bash
python3 train_single_trial.py \
    --model-type {lightgbm|xgboost|catboost|neural_net} \
    --data-path /tmp/trial_data.npz \
    --params '{"n_estimators": 100, ...}' \
    --trial-number 5 \
    --save-model \
    --model-output-dir /tmp/trial_models/
```

**Output (JSON to stdout):**
```json
{
  "success": true,
  "model_type": "catboost",
  "trial_number": 5,
  "val_mse": 7.97e-12,
  "r2": 1.0,
  "train_mse": 1.23e-12,
  "val_mae": 0.000001,
  "device": "cuda:0:1",
  "duration": 3.92,
  "checkpoint_path": "/tmp/trial_models/catboost_trial5.cbm",
  "params": {"n_estimators": 337, ...}
}
```

### 6.4 Model File Extensions

| Model Type | Extension | Format |
|------------|-----------|--------|
| `neural_net` | `.pth` | PyTorch state dict |
| `xgboost` | `.json` | XGBoost JSON |
| `lightgbm` | `.txt` | LightGBM text |
| `catboost` | `.cbm` | CatBoost binary |

---

## Part 7: Model Checkpoint Persistence (NEW)

### 7.1 Checkpoint Flow

```
Trial runs in subprocess
    ↓
model.save(checkpoint_path)  # e.g., /tmp/xyz/catboost_trial5.cbm
    ↓
JSON output includes checkpoint_path
    ↓
Coordinator tracks best_result
    ↓
Optimization completes
    ↓
coordinator.save_best_model()
    ↓
shutil.copy2(temp_checkpoint, output_dir/best_model.{ext})
    ↓
Sidecar updated with actual checkpoint_path
```

### 7.2 Output Directory Structure

```
models/reinforcement/
├── best_model.cbm              # Winner checkpoint (actual model file)
├── best_model.meta.json        # Sidecar with checkpoint_path
└── (optional) all_models/      # If --save-all-models used
    ├── lightgbm_trial7.txt
    ├── xgboost_trial2.json
    ├── catboost_trial5.cbm
    └── neural_net_trial0.pth
```

### 7.3 Sidecar Schema (Updated)

```json
{
  "schema_version": "3.2.7",
  "model_type": "catboost",
  "checkpoint_path": "models/reinforcement/best_model.cbm",
  "checkpoint_format": "cbm",
  
  "feature_schema": {
    "source_file": "/path/to/survivors_with_scores.json",
    "feature_count": 48,
    "feature_names": ["actual_mean", "actual_std", ...],
    "ordering": "lexicographic_by_key",
    "feature_schema_hash": "779dd68f4bf7554a"
  },
  
  "y_label_source": {
    "field": "features.score",
    "observed_min": 0.0,
    "observed_max": 0.375,
    "observed_range": 0.375,
    "normalization_method": "none",
    "output_range": [0.0, 1.0],
    "warnings": []
  },
  
  "training_params": {
    "model_type": "catboost",
    "cb_n_estimators": 337,
    "cb_max_depth": 10,
    "cb_lr": 0.0498,
    ...
  },
  
  "validation_metrics": {
    "val_mse": 7.97e-12,
    "r2": 1.0,
    "train_mse": 1.23e-12,
    "val_mae": 0.000001
  },
  
  "provenance": {
    "cli_args": {...},
    "git_commit": "5583c23",
    "n_survivors_loaded": 395211,
    "timestamp": "2025-12-22T19:21:23.546349"
  },
  
  "agent_metadata": {
    "pipeline_step": 5,
    "pipeline_step_name": "anti_overfit_training",
    "run_id": "step5_20251222_192123"
  }
}
```

---

## Part 8: Agent Manifest Update (Step 5)

### 8.1 reinforcement.json v1.5.0

```json
{
  "manifest_version": "1.5.0",
  "agent_name": "reinforcement_agent",
  "description": "Step 5: Anti-Overfit Training with Multi-Model Comparison",
  "pipeline_step": 5,
  
  "inputs": [
    "survivors_with_scores.json",
    "synthetic_lottery.json"
  ],
  
  "outputs": [
    "models/reinforcement/best_model.{ext}",
    "models/reinforcement/best_model.meta.json"
  ],
  
  "actions": [
    {
      "type": "run_script",
      "script": "meta_prediction_optimizer_anti_overfit.py",
      "args_map": {
        "survivors": "survivors_file",
        "lottery-data": "lottery_file",
        "trials": "optuna_trials",
        "model-type": "model_type",
        "compare-models": "compare_all_models",
        "output-dir": "model_output_dir"
      },
      "timeout_minutes": 60
    }
  ],
  
  "parameter_bounds": {
    "optuna_trials": {"min": 10, "max": 200, "default": 50},
    "model_type": {"choices": ["neural_net", "xgboost", "lightgbm", "catboost"]},
    "compare_all_models": {"type": "bool", "default": true}
  },
  
  "follow_up_agents": ["prediction_agent"],
  
  "subprocess_isolation": {
    "enabled": true,
    "reason": "OpenCL/CUDA GPU conflict resolution",
    "worker_script": "train_single_trial.py",
    "coordinator": "subprocess_trial_coordinator.py"
  }
}
```

### 8.2 New Manifest Fields

| Field | Purpose |
|-------|---------|
| `subprocess_isolation.enabled` | Indicates subprocess architecture is used |
| `subprocess_isolation.reason` | Documents why it's needed |
| `subprocess_isolation.worker_script` | Worker script name |
| `subprocess_isolation.coordinator` | Coordinator module name |

---

## Part 9: Step 5 → Step 6 Handoff Protocol (NEW)

### 9.1 Handoff Requirements

For Step 6 (`prediction_generator.py`) to load the trained model:

1. **Model checkpoint must exist** at `checkpoint_path`
2. **Sidecar must exist** with valid `model_type` field
3. **Feature schema hash must match** between training and inference data
4. **Model type determines loader:**

```python
# prediction_generator.py model loading
def load_model(meta_path: str):
    with open(meta_path) as f:
        meta = json.load(f)
    
    model_type = meta['model_type']
    checkpoint = meta['checkpoint_path']
    
    if model_type == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(checkpoint)
    
    elif model_type == 'xgboost':
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(checkpoint)
    
    elif model_type == 'lightgbm':
        import lightgbm as lgb
        model = lgb.Booster(model_file=checkpoint)
    
    elif model_type == 'neural_net':
        import torch
        checkpoint_data = torch.load(checkpoint)
        # Reconstruct model from architecture info
        model = build_neural_net(
            input_dim=checkpoint_data['input_dim'],
            hidden_layers=checkpoint_data['hidden_layers'],
            dropout=checkpoint_data['dropout']
        )
        model.load_state_dict(checkpoint_data['model_state_dict'])
    
    return model, meta
```

### 9.2 Validation Checklist

```python
def validate_step5_output(output_dir: str) -> bool:
    """Validate Step 5 outputs before Step 6."""
    meta_path = Path(output_dir) / 'best_model.meta.json'
    
    # 1. Sidecar exists
    if not meta_path.exists():
        raise FileNotFoundError(f"Sidecar not found: {meta_path}")
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    # 2. Required fields present
    required = ['model_type', 'checkpoint_path', 'feature_schema']
    for field in required:
        if field not in meta:
            raise ValueError(f"Missing required field: {field}")
    
    # 3. Checkpoint exists
    checkpoint = Path(meta['checkpoint_path'])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    # 4. Model type is valid
    valid_types = ['neural_net', 'xgboost', 'lightgbm', 'catboost']
    if meta['model_type'] not in valid_types:
        raise ValueError(f"Invalid model_type: {meta['model_type']}")
    
    return True
```

---

## Part 10: Watcher Agent Integration

### 10.1 Pipeline Execution Flow

```
Watcher Agent
    ↓
Step 5: meta_prediction_optimizer_anti_overfit.py --compare-models
    ↓
SubprocessTrialCoordinator manages trials
    ↓
Best model saved to models/reinforcement/best_model.{ext}
    ↓
Sidecar saved to models/reinforcement/best_model.meta.json
    ↓
Watcher Agent validates outputs
    ↓
Step 6: prediction_generator.py --models-dir models/reinforcement
    ↓
Model loaded from checkpoint_path
    ↓
Predictions generated
```

### 10.2 Error Handling

| Error | Watcher Agent Action |
|-------|---------------------|
| No checkpoint file | Retry Step 5 with more trials |
| Schema hash mismatch | FATAL - data inconsistency |
| Model load failure | Check model_type, retry if corrupted |
| GPU memory error | Clear cache, retry with smaller batch |

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | Claude (AI) | 2025-12-22 | ✓ |
| Implemented | Michael | 2025-12-22 | ✓ |
| Team Beta | | | |
| Team Alpha | | | |

---

## Appendix A: GPU Framework Compatibility Matrix

| Framework | GPU Backend | Can coexist with CUDA? |
|-----------|-------------|------------------------|
| PyTorch | CUDA | ✅ Yes (same backend) |
| XGBoost | CUDA | ✅ Yes (same backend) |
| CatBoost | CUDA | ✅ Yes (same backend) |
| LightGBM | **OpenCL** | ❌ No (requires subprocess) |

## Appendix B: Performance Benchmarks

### Subprocess Overhead

| Metric | Value | Notes |
|--------|-------|-------|
| Subprocess spawn | ~0.5s | One-time per trial |
| Data load (.npz) | ~1.5s | 252K samples × 48 features |
| Total overhead | ~2.0s | Negligible vs training |

### Model Training Times (395K samples)

| Model | Avg Time | Device |
|-------|----------|--------|
| CatBoost | 3.9s | cuda:0:1 |
| XGBoost | 2.5s | cuda:0 |
| LightGBM | 2.5s | gpu (OpenCL) |
| Neural Net | 240s | cuda:0,1 (DataParallel) |

---

**End of v3.2.7 Specification**
