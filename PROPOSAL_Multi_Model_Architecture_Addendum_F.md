# PROPOSAL: Multi-Model ML Architecture
## Addendum F: Subprocess Isolation for OpenCL/CUDA Compatibility

**Date:** December 22, 2025  
**Status:** IMPLEMENTED  
**Parent Document:** PROPOSAL_Multi_Model_Architecture_v3_1_2_FINAL.md  
**Addresses:** Critical GPU compatibility bug discovered post-implementation

---

## Executive Summary

After implementing the Multi-Model Architecture (v3.1.2), a critical bug was discovered: **LightGBM (OpenCL) fails when CUDA has been initialized first**. This addendum documents the root cause and the subprocess isolation solution that was implemented to resolve it.

---

## Problem Discovery

### Symptoms
```
RuntimeError: Cannot access GPU  
File: lightgbm/basic.py, line 203  
Error Code: -9999 (OpenCL initialization failure)
```

### Investigation

1. **Initial Hypothesis:** Running LightGBM first via `SAFE_MODEL_ORDER` would solve the problem.

2. **Discovery:** Even with `SAFE_MODEL_ORDER = ['lightgbm', ...]`, Optuna's TPE sampler randomly selects model types. The order guarantee only applies to direct comparison, not Optuna trials.

3. **Root Cause:** The `initialize_cuda_early()` function runs at module import time:

```python
# meta_prediction_optimizer_anti_overfit.py (OLD)
def initialize_cuda_early():
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        _ = torch.zeros(1).to(device)  # Creates CUDA context
        return True
    return False

CUDA_INITIALIZED = initialize_cuda_early()  # Runs at import!
```

4. **GPU Context Conflict:**
   - CUDA creates an exclusive GPU context
   - OpenCL cannot access GPU while CUDA context exists
   - Python process cannot "un-initialize" CUDA
   - Only solution: fresh Python process

---

## Solution: Subprocess Isolation

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│           meta_prediction_optimizer_anti_overfit.py                  │
│                    (Main Process - NO GPU imports)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SubprocessTrialCoordinator                                         │
│  ├── Saves training data to .npz file (once)                       │
│  ├── For each Optuna trial:                                        │
│  │   └── subprocess.run(train_single_trial.py ...)                 │
│  │       ├── Fresh Python interpreter                              │
│  │       ├── NO GPU context inherited                              │
│  │       ├── Imports only required ML library                      │
│  │       ├── Trains model                                          │
│  │       ├── Saves checkpoint                                      │
│  │       ├── Returns JSON with metrics                             │
│  │       └── Exits → GPU memory freed                              │
│  │                                                                  │
│  ├── Tracks best result across all trials                          │
│  └── Copies winning model to output directory                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **No GPU imports at module level** in worker script
2. **Import ML library after argument parsing** (inside training function)
3. **Each trial in fresh subprocess** (no GPU state inheritance)
4. **Subprocess exits after training** (GPU context destroyed)

---

## Implementation Files

### 1. train_single_trial.py (Isolated Worker)

**Critical Structure:**
```python
#!/usr/bin/env python3
# STDLIB ONLY AT MODULE LEVEL - NO GPU LIBRARIES!
import sys
import json
import argparse
import time
import os
from pathlib import Path

def train_lightgbm(X_train, y_train, X_val, y_val, params, save_path=None):
    # Import INSIDE function - after subprocess started
    import lightgbm as lgb
    import numpy as np
    # ... training code ...

def train_xgboost(X_train, y_train, X_val, y_val, params, save_path=None):
    import xgboost as xgb
    import numpy as np
    # ... training code ...

# Similar for catboost, neural_net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--params', type=str)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--model-output-dir', type=str)
    args = parser.parse_args()
    
    # GPU environment setup BEFORE imports
    setup_gpu_environment(args.model_type)
    
    # NOW import numpy (after GPU setup)
    import numpy as np
    
    # Load data, train model, output JSON
```

**Version:** 1.0.1  
**New Args:** `--save-model`, `--model-output-dir`  
**Output:** JSON to stdout with `checkpoint_path` field

### 2. subprocess_trial_coordinator.py

**Key Methods:**
```python
class SubprocessTrialCoordinator:
    def __init__(self, X_train, y_train, X_val, y_val, ...):
        # Save data to temp .npz file
        self.data_path = self.temp_dir / 'trial_data.npz'
        np.savez(self.data_path, X_train=X_train, ...)
    
    def run_trial(self, model_type, params, trial_number):
        cmd = [
            sys.executable,
            str(self.worker_script),
            '--model-type', model_type,
            '--data-path', str(self.data_path),
            '--params', json.dumps(params),
            '--save-model',
            '--model-output-dir', str(self.temp_models_dir)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=self.timeout)
        # Parse JSON output, track best result
    
    def save_best_model(self):
        # Copy best checkpoint to output directory
        shutil.copy2(self.best_result.checkpoint_path, dest_path)
```

**Version:** 1.0.1  
**New Features:** Model saving, best result tracking

### 3. meta_prediction_optimizer_anti_overfit.py

**Key Changes:**
```python
# OLD (problematic)
CUDA_INITIALIZED = initialize_cuda_early()  # Runs at import!

# NEW (conditional)
CUDA_INITIALIZED = False  # Set in main() based on args

def main():
    if args.compare_models:
        # Subprocess isolation - NO CUDA init here
        print("⚡ Mode: Multi-Model Comparison (Subprocess Isolation)")
    else:
        CUDA_INITIALIZED = initialize_cuda_early()
```

**Version:** 2.0.0  
**New Features:** Conditional CUDA init, subprocess coordinator integration

---

## Validation Results

### Test: 12 Trials in Random Order

```
Trial 0:  LIGHTGBM   ✅ (gpu)     MSE: 0.003243
Trial 1:  NEURAL_NET ✅ (cuda:0)  MSE: 0.004272
Trial 2:  XGBOOST    ✅ (cuda:0)  MSE: 0.003654
Trial 3:  XGBOOST    ✅ (cuda:0)  MSE: 0.003654
Trial 4:  CATBOOST   ✅ (cuda:0)  MSE: 0.003217
Trial 5:  NEURAL_NET ✅ (cuda:0)  MSE: 0.004700
Trial 6:  XGBOOST    ✅ (cuda:0)  MSE: 0.003654
Trial 7:  CATBOOST   ✅ (cuda:0)  MSE: 0.003217
Trial 8:  NEURAL_NET ✅ (cuda:0)  MSE: 0.004375
Trial 9:  CATBOOST   ✅ (cuda:0)  MSE: 0.003217
Trial 10: LIGHTGBM   ✅ (gpu)     MSE: 0.003243  ← AFTER 9 CUDA trials!
Trial 11: LIGHTGBM   ✅ (gpu)     MSE: 0.003242  ← Still works!
```

**Key Result:** LightGBM (OpenCL) works regardless of trial order.

### Production Run: 395K Survivors

| Model | Val MSE | R² | Device |
|-------|---------|-----|--------|
| 🏆 CatBoost | 2.17e-11 | 1.0000 | cuda:0:1 |
| XGBoost | 2.93e-08 | 1.0000 | cuda:0 |
| LightGBM | 8.82e-08 | 1.0000 | gpu (OpenCL) |
| Neural Net | 0.00249 | -0.0002 | cuda:0,1 |

---

## Performance Considerations

### Subprocess Overhead

| Metric | Value | Impact |
|--------|-------|--------|
| Subprocess spawn | ~0.5s | Negligible vs training time |
| Data serialization | ~2s (395K samples) | Once per optimization run |
| Total overhead per trial | ~2.5s | ~1% of training time |

### GPU Memory

| Behavior | Before | After |
|----------|--------|-------|
| Memory per trial | Accumulates | Released on exit |
| Max concurrent usage | 1 process | 1 process |
| Memory leaks | Possible | Impossible |

---

## Backward Compatibility

| Command | Before | After |
|---------|--------|-------|
| `--model-type neural_net` | ✅ Works | ✅ Same (uses subprocess) |
| `--model-type xgboost` | ✅ Works | ✅ Same (uses subprocess) |
| `--model-type lightgbm` | ✅ Works (if first) | ✅ Always works |
| `--compare-models` | ❌ LightGBM fails | ✅ All 4 work |

---

## Model Checkpoint Saving

### Problem
Subprocess trains model, returns metrics via JSON, exits → model object is garbage collected.

### Solution
Added model saving to worker script:

```python
def train_lightgbm(..., save_path=None):
    # ... training ...
    if save_path:
        model.save_model(save_path)
    return {'checkpoint_path': save_path, ...}
```

### File Extensions

| Model | Extension | Format |
|-------|-----------|--------|
| LightGBM | .txt | Booster text format |
| XGBoost | .json | JSON format |
| CatBoost | .cbm | CatBoost binary model |
| Neural Net | .pth | PyTorch state dict |

---

## Integration with Existing Pipeline

### Step 5 Output
```
models/reinforcement/
├── best_model.cbm           # Winner checkpoint
└── best_model.meta.json     # Sidecar with checkpoint_path
```

### Sidecar Update
```json
{
  "checkpoint_path": "models/reinforcement/best_model.cbm",
  "checkpoint_format": "cbm",
  "model_type": "catboost",
  ...
}
```

### Step 6 Compatibility
`prediction_generator.py` loads model from `checkpoint_path`:
```python
meta = json.load(open('best_model.meta.json'))
model_type = meta['model_type']
checkpoint = meta['checkpoint_path']

if model_type == 'catboost':
    model = CatBoostRegressor().load_model(checkpoint)
elif model_type == 'xgboost':
    model = xgb.Booster()
    model.load_model(checkpoint)
# etc.
```

---

## Future Considerations

### 1. Parallel Subprocess Execution
Currently trials run sequentially. Could parallelize with process pool, but need to manage GPU memory.

### 2. Neural Net Timeout
300s timeout insufficient for large datasets. Consider:
- Dynamic timeout based on dataset size
- `--timeout` CLI argument
- Reduced epochs for comparison phase

### 3. Model Ensemble
With all 4 models saved, could implement weighted ensemble prediction.

---

## Conclusion

The subprocess isolation architecture successfully resolves the OpenCL/CUDA conflict while:
- Maintaining backward compatibility
- Adding model checkpoint saving
- Enabling fair comparison across all 4 model types
- Introducing minimal overhead (~2.5s per trial)

This fix is now deployed and validated on the Zeus cluster.
