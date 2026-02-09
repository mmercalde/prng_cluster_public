# Design Invariant: GPU Process Isolation

**Status:** MANDATORY | **Enforced Since:** Session 72 (Feb 8, 2026)

---

## The Invariant

> **GPU-accelerated code must NEVER run in the coordinating process when using subprocess isolation.**

This is NON-NEGOTIABLE. Violations will cause:
- LightGBM OpenCL errors (-9999)
- CatBoost "CUDA device busy" failures
- Neural net silent crashes
- Non-deterministic GPU failures

---

## Why This Matters

### The Problem

When running multiple GPU-accelerated ML frameworks in a single Python process:

| Framework | GPU Runtime |
|-----------|-------------|
| LightGBM | OpenCL |
| CatBoost | CUDA |
| XGBoost | CUDA |
| PyTorch | CUDA |

These runtimes **do not coordinate VRAM ownership**:
- CUDA frameworks retain VRAM via caching allocators
- LightGBM's OpenCL context cannot reliably initialize after CUDA touches GPU
- Cleanup APIs (`gc.collect()`, cache clears) are **ineffective**
- Once GPU context is initialized, it cannot be safely reset

### The Solution

**Hard process isolation per model:**
- Parent coordinator NEVER imports GPU libraries
- Each model trains in its own subprocess
- OS-enforced teardown on process exit
- Results returned via JSON

---

## Implementation

### 1. Deferred CUDA Initialization
```python
# At module level - DO NOT initialize GPU
CUDA_INITIALIZED = False  # Deferred - set in main()

# In main() - conditional based on mode
if args.compare_models:
    print("⚡ Mode: Multi-Model Comparison (Subprocess Isolation)")
    print("   GPU initialization DEFERRED to subprocesses")
    CUDA_INITIALIZED = False  # Parent stays GPU-clean
else:
    CUDA_INITIALIZED = initialize_cuda_early()
```

### 2. Subprocess Architecture
```
Parent Process (meta_prediction_optimizer_anti_overfit.py)
    ├── NO GPU imports
    ├── NO torch.cuda calls
    ├── NO catboost GPU init
    └── Spawns subprocesses via subprocess_trial_coordinator.py
        │
        ├── Subprocess 1: neural_net (fresh CUDA context) → exits
        ├── Subprocess 2: lightgbm (fresh OpenCL context) → exits
        ├── Subprocess 3: xgboost (fresh CUDA context) → exits
        └── Subprocess 4: catboost (fresh CUDA context) → exits
```

### 3. Key Files

| File | Role |
|------|------|
| `meta_prediction_optimizer_anti_overfit.py` | Coordinator (NO GPU) |
| `subprocess_trial_coordinator.py` | Subprocess orchestration |
| `train_single_trial.py` | Single model worker (HAS GPU) |

---

## What NOT To Do

❌ **NEVER** initialize CUDA at module import time when using `--compare-models`
```python
# BAD - This breaks subprocess isolation
CUDA_INITIALIZED = initialize_cuda_early()  # Runs at import!
```

❌ **NEVER** import torch/catboost/xgboost in coordinator when using subprocess mode
```python
# BAD - GPU context leaks to subprocesses
import torch
torch.cuda.is_available()  # Initializes CUDA context!
```

❌ **NEVER** rely on SAFE_MODEL_ORDER for correctness
```python
# This is a WORKAROUND, not a solution
SAFE_MODEL_ORDER = ['lightgbm', ...]  # Order doesn't fix isolation
```

---

## Verification

When `--compare-models` is active, you should see:
```
⚡ Mode: Multi-Model Comparison (Subprocess Isolation)
   GPU initialization DEFERRED to subprocesses
✅ CUDA initialized: False  ← Parent is GPU-clean
```

Each subprocess should report its own GPU:
```
Trial 0: NEURAL_NET
  ✅ SUCCESS 🚀
  Device: cuda:0,1 (DataParallel)  ← Subprocess owns GPU

Trial 1: LIGHTGBM  
  ✅ SUCCESS 🚀
  Device: gpu  ← OpenCL works because no prior CUDA
```

---

## History

| Date | Event |
|------|-------|
| Dec 2025 | Original subprocess isolation implemented |
| Jan 2026 | Integration lost during refactoring |
| Feb 8, 2026 | Issue rediscovered, root cause identified |
| Feb 8, 2026 | Fix applied, invariant documented (Session 72) |

---

## References

- Team Beta Analysis: "GPU memory isolation cannot be safely achieved inside one Python process"
- Session 72 changelog
- `subprocess_trial_coordinator.py` header documentation

---

**This invariant is now consistently applied across:**
- Multi-model training (Step 5)
- Distributed scoring
- Selfplay training loops  
- WATCHER dispatch phases

**Violations of this invariant are considered CRITICAL BUGS.**
