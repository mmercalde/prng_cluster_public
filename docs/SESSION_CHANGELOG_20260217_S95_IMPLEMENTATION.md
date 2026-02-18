# SESSION_CHANGELOG_20260217_S95_IMPLEMENTATION.md

## Session 95 — Dual-GPU Parallel NN Optuna + Per-Step Timeout
**Date:** 2026-02-17  
**Operator:** Team Alpha (Claude)  
**Reviewer:** Team Beta (3 review rounds, clean approve)

---

## Summary

Implemented dual-GPU parallelism for Step 5 NN Optuna training on Zeus (2× RTX 3080 Ti).
Added per-step WATCHER timeout support. Step 5 timeout raised to 360 minutes.
Enforced single-GPU policy across all non-leased code paths.

---

## S95 Production Run Results (Pre-Implementation Baseline)

The S95 baseline run (20 NN Optuna trials, n_jobs=1, old code) was killed by the
WATCHER 120-minute timeout at trial 6 of 20. No final model was saved.

| Trial | R² | Notes |
|-------|-----|-------|
| 1 | -5.53e-05 | Random exploration |
| 2 | -0.005977 | Bad region — high LR overshoot |
| 3 | -0.000143 | TPE adjusting |
| 4 | -0.000060 | Approaching zero |
| 5 | +1.84e-06 | Noise around zero (not meaningful signal) |
| 6 | (killed) | Timeout at 120 min |

Per Team Beta: R² values are all clustered around zero — the NN is performing
indistinguishably from the mean predictor. The sign flip at Trial 5 is noise, not
signal. Pipeline is stable (no NaNs, no explosions), but predictive signal is
extremely weak under this formulation.

The NN's purpose is NOT R² competition with tree models — it's the Chapter 14
diagnostics probe (gradient flow, weight distributions, activation statistics)
that drives autonomous WATCHER decision-making.

Each trial took ~35 min (5 folds × 7 min). At n_jobs=1, 20 trials = ~11.7 hours.
The 120-min timeout never had a chance.

---

## Files Modified

### meta_prediction_optimizer_anti_overfit.py (2595 → 2706 lines, +111)

| Change | Lines | Description |
|--------|-------|-------------|
| Imports | 53-54 | Added `import queue`, `import uuid` |
| `_s95_detect_cuda_gpus_no_torch()` | 1630-1664 | CUDA-clean GPU count (env → nvidia-smi → fallback 1). **NO torch import** (S72 invariant) |
| `_s95_build_gpu_queue()` | 1666-1674 | Thread-safe `queue.Queue` of GPU id strings |
| `_run_optuna_optimization()` | 1713-1751 | GPU detection, JournalFileBackend when n_jobs>1, `study.optimize(..., n_jobs=n_jobs)` |
| `_optuna_objective()` | 1893-1952 | GPU lease via queue.get() for entire trial (all folds), try/finally return |
| `_run_nn_optuna_trial()` | 1808, 1817-1818, 1842-1850 | Added `gpu_id=None` param, set `CUDA_VISIBLE_DEVICES` in sub_env, pass `env=sub_env` |
| `_export_split_npz()` | 2002-2035 | UUID-based naming with `tempfile.mkdtemp()`, `trial_number`/`fold_idx` params |
| NPZ cleanup | 1892-1898 | Clean up per-export temp directory after NPZ removal |
| optuna_info | 1792 | Added `n_jobs` field, use `storage_label` string (not object) |
| Inline CatBoost | 1088 | `devices='0:1'` → `devices='0'` (TB review round 2) |

### train_single_trial.py (904 → 903 lines, -1)

| Change | Lines | Description |
|--------|-------|-------------|
| DataParallel | 509-510 | Removed 4-line block → 2-line comment. Subprocess pinned via CUDA_VISIBLE_DEVICES |
| CatBoost devices | 371-372 | `'0:1'` → `'0'` (defensive: subprocess sees single GPU) |
| CatBoost device_used | 375 | `'cuda:0:1'` → `'cuda:0'` (telemetry fix, TB review round 3) |
| CVD default | 78 | `'0,1'` → `'0'` (single GPU default, lease path overrides) |

### agents/watcher_agent.py (+13 lines)

| Change | Lines | Description |
|--------|-------|-------------|
| `step_timeout_overrides` | WatcherConfig | New `Optional[Dict[int, int]]` field for per-step timeouts |
| `get_step_timeout_minutes()` | WatcherConfig | Method to resolve per-step timeout with fallback to global |
| `to_dict()` | WatcherConfig | Include new field in serialization |
| `execute_step()` | ~1404-1409 | Use `get_step_timeout_minutes(step)` instead of global |
| Timeout error msg | ~1436 | Show actual per-step timeout in error |
| `main()` | ~2674 | Set `step_timeout_overrides={5: 360}` |

**Effect:** Step 5 gets 360 minutes (6 hours). All other steps remain at 120 minutes.

---

## Team Beta Critical Fixes Addressed

| Fix | Requirement | Implementation | Review |
|-----|-------------|----------------|--------|
| **A** | GPU detection must NOT import torch | `_s95_detect_cuda_gpus_no_torch()`: env → nvidia-smi → fallback 1 | Round 1 ✅ |
| **B** | Temp NPZ collision-proof under threading | UUID + `tempfile.mkdtemp()` per export | Round 1 ✅ |
| **C** | GPU lease scope: whole trial, not per fold | `queue.get()` before fold loop, `queue.put()` in finally | Round 1 ✅ |
| **D** | Exception safety: prevent lost GPU deadlock | try/finally wraps entire fold loop + return | Round 1 ✅ |
| **E** | CatBoost devices defensive fix | `'0:1'` → `'0'` in train_single_trial.py AND MultiModelTrainer | Rounds 1+2 ✅ |
| **F** | CVD default in worker | `'0,1'` → `'0'` (lease path overrides explicitly) | Round 2 ✅ |
| **G** | CatBoost telemetry `device_used` | `'cuda:0:1'` → `'cuda:0'` | Round 3 ✅ |

## GPU Policy (Enforced)

- **NN Optuna trials:** GPU assigned via S95 lease queue (`CUDA_VISIBLE_DEVICES` per subprocess)
- **Subprocess workers (non-leased):** Default to GPU 0 via CVD='0'
- **Inline tree training (parent):** Inherits environment (no explicit constraint in parent)
- **Only the S95 lease queue assigns multi-GPU work**

## Items Parked (Not S95 Scope)

- Compare-models `best_model = None` disk-first pattern (pre-existing, not broken)
- Composite diagnostic objective for NN (future enhancement)

---

## Existing Bug Fixed

`_run_nn_optuna_trial` created `sub_env = os.environ.copy()` but never passed it to
`subprocess.run()`. Now passes `env=sub_env` — this also enables the GPU pinning.

---

## Invariants Preserved

- ✅ GPU Isolation (S72): Parent process never imports torch or touches CUDA
- ✅ Category B flags: Unchanged (--normalize-features, --use-leaky-relu)
- ✅ Skip registry: Checked before Optuna (S90)
- ✅ TB Trim #1: No --enable-diagnostics on Optuna folds
- ✅ TB Trim #2: NPZ cleanup in finally block
- ✅ Single-writer: save_best_model() after study.optimize()
- ✅ Freshness invalidation (Bug A): File timestamp based
- ✅ WATCHER retry loop: Same artifact paths
- ✅ Graceful degradation: 1 GPU → n_jobs=1 (identical to current behavior)

---

## Deployment Commands

```bash
# From ser8 (files downloaded to ~/Downloads/)
scp ~/Downloads/meta_prediction_optimizer_anti_overfit.py rzeus:~/distributed_prng_analysis/
scp ~/Downloads/train_single_trial.py rzeus:~/distributed_prng_analysis/
scp ~/Downloads/watcher_agent.py rzeus:~/distributed_prng_analysis/agents/
scp ~/Downloads/SESSION_CHANGELOG_20260217_S95_IMPLEMENTATION.md rzeus:~/distributed_prng_analysis/docs/

# On Zeus — verify syntax
cd ~/distributed_prng_analysis
source ~/venvs/torch/bin/activate
python3 -c "import py_compile; py_compile.compile('meta_prediction_optimizer_anti_overfit.py', doraise=True); py_compile.compile('train_single_trial.py', doraise=True); py_compile.compile('agents/watcher_agent.py', doraise=True); print('syntax OK')"

# Smoke test (2 trials, verify dual-GPU)
time PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \
    --start-step 5 --end-step 5 \
    --params '{"compare_models": true, "trials": 2, "enable_diagnostics": true}' \
    2>&1 | tee /tmp/s95_dual_gpu_smoke.log

# Verify in log:
#   "Parallel GPU Optuna: n_jobs=2"
#   "[S95] Leased GPU 0 for trial ..."
#   "[S95] Leased GPU 1 for trial ..."
#   NPZ paths contain UUID, not timestamp+pid
```

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| NN Optuna (20 trials) | ~11.7h (n_jobs=1) | ~5.8h (n_jobs=2) |
| Step 5 timeout | 120 min (kills at trial 6) | 360 min (completes all 20) |
| GPU 0 utilization | 100% | 100% |
| GPU 1 utilization | 0% | ~95% |
| CatBoost GPU policy | devices='0:1' | devices='0' everywhere |

## Git Commit

```bash
git add meta_prediction_optimizer_anti_overfit.py train_single_trial.py agents/watcher_agent.py
git commit -m "S95: Dual-GPU parallel NN Optuna + per-step timeout

Team Beta approved (3 rounds). Changes:
- CUDA-clean GPU detection (no torch in parent, S72 invariant)
- UUID-based NPZ naming (collision-proof under threading)
- JournalFileBackend when n_jobs>1 (avoids SQLite lock)
- Whole-trial GPU lease with exception-safe queue
- DataParallel removed, CatBoost devices='0' everywhere
- CatBoost telemetry device_used fixed
- CVD default '0' in worker (lease path overrides)
- Per-step timeout: Step 5 = 360min for NN Optuna
- All other steps remain at 120min default"
git push public main
```
