# Multi-Model Architecture Integration with Autonomy
**Updated:** December 24, 2025 (Session 15)

## Overview

This document explains how the Multi-Model Architecture integrates with the autonomous pipeline via `watcher_agent.py`.

---

## Current Architecture
````
watcher_agent.py
    │
    ├── Monitors pipeline steps (1 → 2 → 2.5 → 3 → 4 → 5 → 6)
    │
    ├── Reads agent_manifests/*.json for each step
    │
    ├── Calls LLM (Qwen) for decisions when needed
    │
    └── Executes scripts based on manifest actions
````

---

## Integration Points for Multi-Model

### 1. Agent Manifests (Updated ✅)

**agent_manifests/reinforcement.json (v1.5.0):**
````json
{
  "version": "1.5.0",
  "parameter_bounds": {
    "model_type": {
      "type": "categorical",
      "choices": ["neural_net", "xgboost", "lightgbm", "catboost"],
      "default": "neural_net"
    },
    "compare_models": {
      "type": "bool", 
      "default": false
    }
  }
}
````

**agent_manifests/prediction.json (v1.5.0):**
````json
{
  "version": "1.5.0",
  "parameter_bounds": {
    "models_dir": {
      "type": "string",
      "default": "models/reinforcement"
    },
    "parent_run_id": {
      "type": "string",
      "default": null,
      "description": "Auto-read from sidecar if not provided"
    }
  }
}
````

### 2. Watcher Agent Decision Flow
````
Step 5 Triggered
    │
    ├── watcher_agent reads reinforcement.json manifest
    │
    ├── LLM analyzes previous step results:
    │   │
    │   ├── "survivors_with_scores.json has 395K entries"
    │   ├── "Feature distribution shows high variance"
    │   └── "Previous neural_net had overfit_ratio > 1.5"
    │
    ├── LLM suggests parameters:
    │   {
    │     "model_type": "xgboost",
    │     "compare_models": true,
    │     "trials": 50,
    │     "k_folds": 5
    │   }
    │
    └── watcher_agent executes:
        python3 meta_prediction_optimizer_anti_overfit.py \
            --model-type xgboost \
            --compare-models \
            --trials 50
````

### 3. Step 5 → Step 6 Handoff (NEW in v2.2)
````
Step 5 Output:
├── models/reinforcement/best_model.json (or .pth)
└── models/reinforcement/best_model.meta.json (sidecar)
    ├── model_type: "xgboost"
    ├── feature_schema: { per_seed_feature_names: [...], total_features: 62 }
    └── agent_metadata:
        └── run_id: "step5_20251223_171709"

Step 6 Input:
├── Reads sidecar → auto-detects model type
├── Extracts parent_run_id → links to training run
└── Loads model via model_factory.load_model_from_sidecar()

Step 6 Output:
├── predictions_*.json
│   ├── predictions: [521, 626, 415]
│   ├── raw_scores: [0.127, 0.108, 0.057]      # Machine truth
│   ├── confidence_scores: [0.79, 0.68, 0.32]  # Calibrated
│   └── agent_metadata:
│       └── parent_run_id: "step5_20251223_171709"  # Lineage!
````

### 4. Agent Context Injection
````python
# In watcher_agent.py - build_step5_command()

def build_step5_command(self, context: dict) -> list:
    """Build Step 5 command with model selection."""
    
    cmd = [
        'python3', 'meta_prediction_optimizer_anti_overfit.py',
        '--survivors', context['survivors_file'],
        '--lottery-data', context['lottery_file'],
        '--trials', str(context.get('trials', 50)),
        '--k-folds', str(context.get('k_folds', 5)),
        '--output-dir', 'models/reinforcement',
    ]
    
    # Model type selection (from LLM or default)
    model_type = context.get('model_type', 'neural_net')
    cmd.extend(['--model-type', model_type])
    
    # Compare all models if suggested
    if context.get('compare_models', False):
        cmd.append('--compare-models')
    
    return cmd
````

### 5. Step 6 Command (Auto-Detection)
````python
# In watcher_agent.py - build_step6_command()

def build_step6_command(self, context: dict) -> list:
    """Build Step 6 command - model type auto-detected from sidecar."""
    
    cmd = [
        'python3', 'prediction_generator.py',
        '--models-dir', 'models/reinforcement',  # Sidecar here
        '--survivors-forward', context['forward_survivors'],
        '--lottery-history', context['lottery_file'],
    ]
    
    # Optional: explicit parent_run_id (normally auto-read from sidecar)
    if context.get('parent_run_id'):
        cmd.extend(['--parent-run-id', context['parent_run_id']])
    
    return cmd
````

---

## LLM Prompt for Step 5 Model Selection
````python
STEP5_CONTEXT_PROMPT = """
## Step 5: Anti-Overfit Training

You are selecting ML model configuration for survivor quality prediction.

### Available Model Types:
- neural_net: PyTorch NN, runs on all 26 GPUs (ROCm + CUDA)
- xgboost: Gradient boosting, Zeus only (CUDA), good for tabular data
- lightgbm: Fast gradient boosting, Zeus only, handles large datasets
- catboost: Gradient boosting, multi-GPU on Zeus, handles categorical features

### When to use each:
- neural_net: Default, good for complex patterns, distributed training
- xgboost/lightgbm: When data has high feature variance, tree models resist overfitting
- catboost: When skip_mode categorical feature is important
- compare_models=true: When unsure, let the system pick best

### Previous Results:
{previous_step_results}

### Current Data:
- Survivors: {survivor_count}
- Features: {feature_count}
- Score range: [{score_min}, {score_max}]

### Suggest parameters:
```json
{
  "model_type": "...",
  "compare_models": true/false,
  "trials": ...,
  "k_folds": ...
}
```
"""
````

---

## Complete Autonomous Flow
````
┌─────────────────────────────────────────────────────────────────────┐
│                     watcher_agent.py                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 3 Complete → survivors_with_scores.json (813MB)               │
│       │                                                              │
│       ▼                                                              │
│  Step 4 Complete → reinforcement_engine_config.json                 │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ STEP 5: Model Selection Decision                            │    │
│  │                                                              │    │
│  │  LLM analyzes:                                              │    │
│  │  • Survivor count (395K)                                    │    │
│  │  • Feature variance                                         │    │
│  │  • Previous overfit ratios                                  │    │
│  │                                                              │    │
│  │  LLM suggests: {model_type: "xgboost", compare: true}       │    │
│  │                                                              │    │
│  │  Execute:                                                    │    │
│  │  meta_prediction_optimizer_anti_overfit.py                  │    │
│  │    --model-type xgboost --compare-models                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  Output: models/reinforcement/                                       │
│          ├── best_model.json (xgboost won)                          │
│          └── best_model.meta.json (sidecar with run_id)             │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ STEP 6: Prediction (Auto-loads from sidecar)                │    │
│  │                                                              │    │
│  │  prediction_generator.py --models-dir models/reinforcement  │    │
│  │                                                              │    │
│  │  → Reads best_model.meta.json                               │    │
│  │  → Sees model_type: "xgboost"                               │    │
│  │  → Extracts parent_run_id: "step5_20251223_171709"          │    │
│  │  → Loads best_model.json (not .pth!)                        │    │
│  │  → Validates feature hash                                    │    │
│  │  → Generates predictions with lineage                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
````

---

## Subprocess Isolation (OpenCL/CUDA Conflict)

When `--compare-models` is used, Step 5 runs each model in a separate subprocess to avoid GPU backend conflicts:
````
Main Process (subprocess_trial_coordinator.py)
    │ ← NO GPU imports!
    │
    ├── Trial 0: subprocess → train_single_trial.py → LightGBM (OpenCL) → exits
    ├── Trial 1: subprocess → train_single_trial.py → PyTorch (CUDA) → exits
    ├── Trial 2: subprocess → train_single_trial.py → XGBoost (CUDA) → exits
    └── Trial N: Fresh GPU state each time!
````

This solves the "Error Code: -9999" issue when LightGBM runs after CUDA models.

---

## Step 6 Output Contract (v2.2)
````json
{
    "predictions": [521, 626, 415],
    "raw_scores": [0.127792, 0.108792, 0.057818],
    "confidence_scores": [0.7949, 0.6884, 0.3286],
    "confidence_scores_normalized": [1.0, 0.8513, 0.4524],
    "metadata": {
        "method": "dual_sieve",
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
        "parent_run_id": "step5_20251223_171709",
        "confidence": 0.4937,
        "reasoning": "Generated 5 predictions with avg confidence 0.4937"
    }
}
````

### Field Usage for Autonomy:
- `raw_scores` - Cross-run comparison (never normalized)
- `confidence_scores` - Threshold gating (e.g., "act if confidence > 0.7")
- `score_stats.raw_std` - Detect degenerate outputs (std ≈ 0 = no discrimination)
- `parent_run_id` - Trace predictions back to training run

---

## The Key Insight

The **sidecar pattern** makes autonomy easier because:

1. Step 5 can try ANY model type
2. Step 6 doesn't need to know which model won
3. The sidecar is the contract between steps
4. Feature hash validates consistency automatically
5. Parent run ID enables full lineage tracking

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Multi-model wrappers | ✅ Complete | 4 models: neural_net, xgboost, lightgbm, catboost |
| Subprocess isolation | ✅ Complete | Resolves OpenCL/CUDA conflict |
| Sidecar generation | ✅ Complete | model_type, feature_schema, run_id |
| Step 6 model loading | ✅ Complete | Auto-detects from sidecar |
| Confidence calibration | ✅ Complete | Sigmoid z-score, raw scores preserved |
| Parent run ID lineage | ✅ Complete | Auto-read from sidecar, CLI override |
| Watcher agent integration | 🔄 In Progress | Manifest updates done, full integration pending |
