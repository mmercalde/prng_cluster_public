# Session Changelog - S72 (February 8, 2026)

## Summary
Fixed critical LightGBM OpenCL GPU isolation issue. Root cause: parent process was initializing CUDA before subprocess isolation could take effect.

---

## Issue: LightGBM "Unknown OpenCL Error (-9999)"

### Symptoms
- LightGBM failed with OpenCL error -9999
- CatBoost reported "CUDA device busy"
- Neural net crashed silently in 1.7s (should take 100s+)
- All models falling back to CPU

### Root Cause
Parent process initialized CUDA at module import time:
```python
CUDA_INITIALIZED = initialize_cuda_early()  # Ran at import!
```
This defeated subprocess isolation - subprocesses inherited dirty GPU state.

### Solution
1. Defer CUDA initialization until `main()` parses args
2. Skip GPU init entirely when `--compare-models` is active
3. Each subprocess gets clean GPU context

---

## Fixes Applied

### 1. meta_prediction_optimizer_anti_overfit.py
- Changed `CUDA_INITIALIZED = initialize_cuda_early()` → `CUDA_INITIALIZED = False`
- Added conditional init in `main()` based on `--compare-models`
- Restored subprocess isolation wiring (was disconnected)
- Added missing `import os` and `import shutil`
- Fixed SAFE_MODEL_ORDER (LightGBM first)

### 2. agent_manifests/reinforcement.json  
- Changed `"model_type": "xgboost"` → `"compare_models": true`
- WATCHER now invokes multi-model comparison by default

### 3. Documentation Created/Updated
- NEW: `docs/DESIGN_INVARIANT_GPU_ISOLATION.md`
- Updated: Chapter 6 (Section 5.4 added)
- Updated: Chapter 9 (Section 6 added)
- Updated: Chapter 12 (Section 6.4 added)
- Updated: COMPLETE_OPERATING_GUIDE_v1_1.md

---

## Test Results (After Fix)

| Model | Status | Device | R² |
|-------|--------|--------|-----|
| neural_net | ✅ SUCCESS | cuda:0,1 (DataParallel) | -0.0019 |
| lightgbm | ✅ SUCCESS | gpu (OpenCL) | -0.0001 |
| xgboost | ✅ SUCCESS | cuda:0 | -0.0009 |
| catboost | ✅ SUCCESS 🏆 | cuda:0:1 | 0.0003 |

Both WATCHER and direct CLI paths verified working.

---

## Design Invariant Established

> **GPU-accelerated code must NEVER run in the coordinating process when using subprocess isolation.**

This is now documented in `docs/DESIGN_INVARIANT_GPU_ISOLATION.md` and referenced in Chapters 6, 9, and 12.

---

## Lesson Learned

The solution was **already documented** in COMPLETE_OPERATING_GUIDE_v1_1.md (Session 9+). The subprocess isolation architecture existed but:
1. The wiring was lost during refactoring
2. Parent CUDA initialization was added later, defeating isolation
3. Nobody checked the guide when debugging

**Action:** Always search existing documentation before debugging.
