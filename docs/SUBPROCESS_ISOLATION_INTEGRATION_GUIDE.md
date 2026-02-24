# Subprocess Isolation Integration Guide
## For meta_prediction_optimizer_anti_overfit.py

**Version:** 1.0.0  
**Date:** December 2025  
**Status:** Ready for Integration

---

## Overview

This guide explains how to integrate subprocess isolation into the existing
`meta_prediction_optimizer_anti_overfit.py` script to solve the OpenCL/CUDA
conflict when using `--compare-models`.

## Files Created

| File | Purpose |
|------|---------|
| `train_single_trial.py` | Isolated worker script (runs in subprocess) |
| `subprocess_trial_coordinator.py` | Coordinator module (imported by main script) |

## Integration Steps

### Step 1: Copy Files to Zeus

```bash
# Copy the new files
scp train_single_trial.py zeus:~/distributed_prng_analysis/
scp subprocess_trial_coordinator.py zeus:~/distributed_prng_analysis/
```

### Step 2: Create Backup of Existing Script

```bash
ssh zeus
cd ~/distributed_prng_analysis

# Create timestamped backup
cp meta_prediction_optimizer_anti_overfit.py \
   meta_prediction_optimizer_anti_overfit.py.bak_$(date +%Y%m%d_%H%M%S)
```

### Step 3: Modify meta_prediction_optimizer_anti_overfit.py

The modifications are minimal and maintain full backward compatibility:

#### 3.1 Remove Early CUDA Initialization (Around Line 40-59)

**BEFORE:**
```python
def initialize_cuda_early():
    """Initialize CUDA before any model operations"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            _ = torch.zeros(1).to(device)
            # ...
    except:
        pass
    return False

CUDA_INITIALIZED = initialize_cuda_early()  # <-- REMOVE THIS LINE
```

**AFTER:**
```python
def initialize_cuda_early():
    """Initialize CUDA before any model operations"""
    # NOTE: Only called when NOT using --compare-models
    # Subprocess isolation handles GPU init for --compare-models
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            _ = torch.zeros(1).to(device)
            # ...
    except:
        pass
    return False

# REMOVED: CUDA_INITIALIZED = initialize_cuda_early()
# Now called conditionally in main() based on args
CUDA_INITIALIZED = False
```

#### 3.2 Add Import for Subprocess Coordinator (Top of File)

```python
# Add after existing imports
from subprocess_trial_coordinator import (
    SubprocessTrialCoordinator,
    create_optuna_objective,
    run_isolated_comparison,
    SAFE_MODEL_ORDER
)
```

#### 3.3 Modify main() to Use Subprocess for --compare-models

**Find the section where --compare-models is handled and replace with:**

```python
def main():
    import argparse
    global CUDA_INITIALIZED

    parser = argparse.ArgumentParser(
        description='Anti-Overfit Meta-Prediction Optimizer (IMPROVED)'
    )
    parser.add_argument('--survivors', required=True)
    parser.add_argument('--lottery-data', required=True)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--test-holdout', type=float, default=0.2)
    parser.add_argument('--study-name', type=str, help='Optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_studies.db')
    # Multi-model args
    parser.add_argument('--model-type', type=str, default='neural_net',
                        choices=['neural_net', 'xgboost', 'lightgbm', 'catboost'])
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare all 4 model types using subprocess isolation')
    parser.add_argument('--output-dir', type=str, default='models/reinforcement')

    args = parser.parse_args()

    # CONDITIONAL CUDA initialization
    # Only init CUDA if NOT comparing models (subprocess handles it otherwise)
    if not args.compare_models:
        CUDA_INITIALIZED = initialize_cuda_early()
        print(f"✅ CUDA initialized: {CUDA_INITIALIZED}")
    else:
        print("⚡ Using subprocess isolation for --compare-models")
        print("   GPU initialization deferred to subprocesses")

    # ... rest of data loading ...

    # If comparing models, use subprocess isolation
    if args.compare_models:
        print("\n" + "="*70)
        print("MULTI-MODEL COMPARISON (Subprocess Isolation)")
        print("="*70)
        print("This ensures LightGBM (OpenCL) works alongside CUDA models")
        
        # Prepare features (X) and labels (y) for comparison
        # Assuming X_train, y_train, X_val, y_val are prepared above
        
        with SubprocessTrialCoordinator(
            X_train, y_train, X_val, y_val,
            worker_script='train_single_trial.py',
            verbose=True
        ) as coordinator:
            
            # Run Optuna with subprocess isolation
            study = optuna.create_study(
                study_name=args.study_name or f"compare_models_{datetime.now():%Y%m%d_%H%M%S}",
                direction='minimize',
                sampler=TPESampler(seed=42),
                storage=args.storage,
                load_if_exists=True
            )
            
            objective = create_optuna_objective(
                coordinator,
                model_types=SAFE_MODEL_ORDER,
                metric='val_mse'
            )
            
            study.optimize(objective, n_trials=args.trials)
            
            # Get best result
            best_trial = study.best_trial
            print(f"\n🏆 Best Trial: {best_trial.number}")
            print(f"   Model: {best_trial.params['model_type']}")
            print(f"   Val MSE: {best_trial.value:.6f}")
        
        return  # Exit after comparison

    # Original single-model training continues here...
    # ... existing code ...
```

---

## Verification

### Test Subprocess Isolation

```bash
# Run the test scripts first
python3 test_subprocess_isolation.py
python3 test_subprocess_isolation_gpu.py

# Both should show:
# 🎉 TEST PASSED! Subprocess isolation works correctly.
```

### Test Integration

```bash
# Test single model (backward compatible)
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data synthetic_lottery.json \
    --model-type xgboost \
    --trials 5

# Test model comparison (uses subprocess isolation)
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data synthetic_lottery.json \
    --compare-models \
    --trials 20
```

---

## How It Works

### Without --compare-models (Original Behavior)

```
┌─────────────────────────────────────────────────────────────┐
│  meta_prediction_optimizer_anti_overfit.py                  │
│                                                              │
│  1. CUDA initializes at startup                             │
│  2. Single model type trains                                │
│  3. Uses ReinforcementEngine directly                       │
│  4. Everything in one process                               │
│                                                              │
│  ✅ Works for: neural_net, xgboost, catboost               │
│  ❌ LightGBM would fail if run after CUDA init             │
└─────────────────────────────────────────────────────────────┘
```

### With --compare-models (New Subprocess Isolation)

```
┌─────────────────────────────────────────────────────────────┐
│  meta_prediction_optimizer_anti_overfit.py (COORDINATOR)    │
│                                                              │
│  1. NO CUDA initialization at startup                       │
│  2. Creates SubprocessTrialCoordinator                      │
│  3. Saves training data to temp file                        │
│  4. For each Optuna trial:                                  │
│       └─→ subprocess.run(train_single_trial.py)            │
│           └─→ Fresh Python process                         │
│           └─→ Clean GPU state                              │
│           └─→ Returns JSON result                          │
│  5. Optuna compares results fairly                         │
│                                                              │
│  ✅ Works for ALL models including LightGBM                │
│  ✅ True fair comparison                                    │
│  ✅ OpenCL/CUDA conflict solved                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Backward Compatibility

| Command | Before | After |
|---------|--------|-------|
| `--model-type neural_net` | ✅ Works | ✅ Works (same) |
| `--model-type xgboost` | ✅ Works | ✅ Works (same) |
| `--model-type lightgbm` | ✅ Works (if first) | ✅ Works (same) |
| `--compare-models` | ❌ LightGBM fails | ✅ All 4 work |
| No model args | ✅ Default neural_net | ✅ Same |
| Output files | `best_model.*` | Same structure |
| Sidecar JSON | `best_model.meta.json` | Same structure |

---

## Troubleshooting

### Error: Worker script not found

```
FileNotFoundError: Worker script 'train_single_trial.py' not found
```

**Solution:** Ensure `train_single_trial.py` is in `~/distributed_prng_analysis/`

### Error: Timeout on trials

```
TrialResult(success=False, error='Timeout after 300s')
```

**Solution:** Increase timeout in coordinator initialization or reduce model complexity

### Error: JSON parsing failed

```
Invalid JSON output: Expecting value
```

**Solution:** Check `train_single_trial.py` for print statements that aren't JSON

---

## Next Steps After Integration

1. **Run Step 4:** `adaptive_meta_optimizer.py` for architecture recommendations
2. **Full Step 5:** Use `--compare-models` with 395K survivors
3. **Implement `--save-all-models`:** Save all 4 trained models for AI analysis
4. **Run Step 6:** `prediction_generator.py` with best model

---

## Files Summary

```
~/distributed_prng_analysis/
├── train_single_trial.py              # NEW: Isolated worker
├── subprocess_trial_coordinator.py    # NEW: Coordinator module
├── meta_prediction_optimizer_anti_overfit.py  # MODIFIED: Uses coordinator
├── models/
│   ├── model_factory.py              # UNCHANGED
│   ├── model_selector.py             # UNCHANGED (optional future update)
│   └── wrappers/                     # UNCHANGED
│       ├── neural_net_wrapper.py
│       ├── xgboost_wrapper.py
│       ├── lightgbm_wrapper.py
│       └── catboost_wrapper.py
└── agent_manifests/
    └── reinforcement.json            # UNCHANGED
```

---

**END OF INTEGRATION GUIDE**
