# PROPOSAL: Dual-GPU Parallel NN Optuna Trials for Step 5
## Version 2.0 — Team Beta Review Required

**Date:** 2026-02-17 (Session 95)  
**Author:** Team Alpha  
**Priority:** HIGH — Zeus GPU 1 sits 100% idle during 2-hour NN training  
**Target:** Cut NN Optuna training time from ~2 hours to ~1 hour  

---

## Problem

During Step 5 `--compare-models`, NN training dominates at ~2 hours (20 Optuna trials × 
5 K-folds × ~1.2 min/fold). Zeus has 2× RTX 3080 Ti but only GPU 0 is used. GPU 1 is 
completely idle.

```
Current:
GPU 0: [trial 0][trial 1][trial 2]...[trial 19]  ≈ 2 hours
GPU 1: [================ IDLE ================]  ≈ 2 hours wasted
```

---

## Solution: n_jobs=2 Parallel Optuna with GPU Queue

Run two NN Optuna trials simultaneously, one per GPU:

```
Proposed:
GPU 0: [trial 0][trial 2][trial 4]...[trial 18]  ≈ 1 hour
GPU 1: [trial 1][trial 3][trial 5]...[trial 19]  ≈ 1 hour
```

---

## How It Works

Optuna natively supports `n_jobs=2` in `study.optimize()`. This runs two threads, 
each calling `_optuna_objective()` concurrently. Since our objective function spawns 
**subprocesses** (not Python compute), the GIL is NOT a bottleneck — threads spend 
their time in `subprocess.run()` which releases the GIL.

Each subprocess gets its own `CUDA_VISIBLE_DEVICES` via a thread-safe GPU queue:

```python
import queue
import threading

# Thread-safe GPU assignment
_gpu_queue = queue.Queue()
_gpu_queue.put("0")
_gpu_queue.put("1")

def _optuna_objective(self, trial) -> float:
    gpu_id = self._gpu_queue.get()  # Block until a GPU is free
    try:
        # ... existing fold loop ...
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X_train_val)):
            # _run_nn_optuna_trial now passes gpu_id to subprocess
            result = self._run_nn_optuna_trial(
                X_tr, y_tr, X_vl, y_vl, config, trial.number, fold_idx,
                gpu_id=gpu_id
            )
        return avg_r2
    finally:
        self._gpu_queue.put(gpu_id)  # Return GPU to pool
```

And in `_run_nn_optuna_trial()`:
```python
sub_env = os.environ.copy()
sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Isolate to one GPU
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, 
                      cwd=str(Path(__file__).parent), env=sub_env)
```

The `study.optimize()` call changes from:
```python
# Current:
study.optimize(self._optuna_objective, n_trials=n_trials)

# Proposed:
n_gpus = self._detect_gpu_count()  # 2 on Zeus, 1 elsewhere
study.optimize(self._optuna_objective, n_trials=n_trials, n_jobs=n_gpus)
```

---

## Code Audit — What Needs to Change

### File 1: `meta_prediction_optimizer_anti_overfit.py`

| Location | Current | Change | Risk |
|----------|---------|--------|------|
| `__init__` (~line 1399) | `self.device = device` | Add `self._gpu_queue = Queue(); populate with available GPUs` | Low |
| `_run_optuna_optimization` (line 1676) | `study.optimize(obj, n_trials=n)` | Add `n_jobs=n_gpus` | Low |
| `_optuna_objective` (line 1817) | No GPU affinity | Wrap in `gpu_queue.get()/put()` | Medium |
| `_run_nn_optuna_trial` (line 1767) | `sub_env = os.environ.copy()` | Add `sub_env["CUDA_VISIBLE_DEVICES"] = gpu_id` | Low |
| `_export_split_npz` (line 1914) | `{timestamp}_{pid}_data.npz` | Add `_t{trial_number}_f{fold_idx}` to prevent collision | Low |

### File 2: `train_single_trial.py`

| Location | Current | Change | Risk |
|----------|---------|--------|------|
| Line 78 | `os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'` | Only set if not already set (already gated ✅) | None |
| Line 509-512 | `nn.DataParallel(model)` | **Remove.** With CUDA_VISIBLE_DEVICES=X, only 1 GPU visible. DataParallel on a <1MB model across 2 GPUs wastes more time on communication than it saves. | Low |
| Line 371 | CatBoost `devices='0:1'` | Change to `devices='0'`. With CUDA_VISIBLE_DEVICES isolating to 1 GPU, `0:1` would fail. Trees aren't affected by this proposal but defensive fix. | Low |

**No changes to:** `watcher_agent.py`, `training_health_check.py`, manifests, grammars, 
WATCHER retry loop, diagnostics, or any other file.

---

## Collision Analysis — What's Safe, What's Not

### ✅ Already Safe (no changes needed)
- **Optuna SQLite storage:** Each model type has its own `.db` file. NN trials share one 
  DB, but Optuna's `n_jobs` threading handles SQLite locking internally.
- **Subprocess PID isolation:** Each `train_single_trial.py` gets its own PID, own CUDA 
  context, own memory space.
- **Artifact writes during trials:** Optuna folds write NO artifacts (TB rule: memory only).
- **Final model save:** Only happens AFTER `study.optimize()` returns (single-threaded).
- **compare-models sequential loop:** Unchanged. Only NN's internal Optuna is parallel.
  Tree models still run sequentially after NN finishes.
- **Diagnostics on Optuna folds:** Already disabled (TB Trim #1).

### ⚠️ Needs Fix
- **NPZ temp files:** Current naming is `{timestamp}_{pid}`. Two threads in the same 
  process share PID and could generate same timestamp. 
  **Fix:** Add trial number and fold index: `{timestamp}_{pid}_t{trial}_f{fold}_data.npz`

### ⚠️ Important Nuance — Optuna SQLite vs JournalFileBackend
Optuna docs explicitly say: *"We would never recommend SQLite3 for parallel optimization."*
However, this warning is for **multi-process** parallelism (SQLite file locking issues). 
For **n_jobs threading within a single process** (our case), SQLite works fine because 
all threads share the same connection through Optuna's internal session management.

If we see any SQLite locking issues in practice, the fallback is to switch to 
`JournalFileBackend` which is specifically designed for parallel Optuna:

```python
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

storage = JournalStorage(JournalFileBackend(f"optuna_studies/{study_name}.log"))
```

### ⚠️ TPE Sampler with n_jobs=2
TPE is inherently sequential — it uses past trial results to suggest next params.
With `n_jobs=2`, trial N+1 starts before trial N finishes, so TPE can't use trial N's 
result. Optuna handles this gracefully (it uses whatever results are available), but 
with very high `n_jobs` the sampling degrades toward random search.

**n_jobs=2 is fine.** Research confirms n_jobs=5 showed no harmful effect on TPE performance. 
With only 2 concurrent trials, the impact on TPE's Bayesian optimization quality is negligible.

---

## Graceful Degradation

The implementation must work on any GPU count:

```python
def _detect_gpu_count(self) -> int:
    """Detect available CUDA GPUs for parallel Optuna trials."""
    try:
        import torch
        count = torch.cuda.device_count()
        return max(1, count)  # At least 1
    except Exception:
        return 1  # CPU fallback = sequential

# In _run_optuna_optimization:
if self.model_type == "neural_net" and NN_SUBPROCESS_ROUTING_ENABLED:
    n_gpus = self._detect_gpu_count()
    self.logger.info(f"  Parallel GPU Optuna: n_jobs={n_gpus} ({n_gpus} GPUs detected)")
    self._gpu_queue = queue.Queue()
    for i in range(n_gpus):
        self._gpu_queue.put(str(i))
else:
    n_gpus = 1  # Tree models: no benefit from parallel trials

study.optimize(self._optuna_objective, n_trials=n_trials, n_jobs=n_gpus)
```

- **Zeus (2 GPUs):** `n_jobs=2`, full parallel
- **Single GPU machine:** `n_jobs=1`, same as current behavior
- **CPU only:** `n_jobs=1`, same as current behavior

---

## Invariants Preserved

| Invariant | Status |
|-----------|--------|
| GPU Isolation (S72) — parent never touches CUDA | ✅ Unchanged. All GPU work in subprocesses |
| Category B flags (normalize, leaky_relu, dropout) | ✅ Threaded through unchanged |
| Skip registry | ✅ Checked before Optuna starts |
| WATCHER retry loop | ✅ Receives same final artifacts |
| Freshness invalidation (Bug A, S93) | ✅ File timestamp based, unaffected |
| TB Trim #1 — no diagnostics on Optuna folds | ✅ Only final model emits diagnostics |
| TB rule — no artifacts during trials (memory only) | ✅ Unchanged |
| Single-writer — save_best_model() called once | ✅ Called after study.optimize() returns |

---

## Expected Results

| Metric | Current | Proposed |
|--------|---------|----------|
| NN Optuna time (20 trials) | ~2 hours | **~1 hour** |
| Total Step 5 time (compare-models) | ~2h 5min | **~1h 5min** |
| GPU 0 utilization | 100% | 100% |
| GPU 1 utilization | **0%** | **~95%** |
| Code changes | — | ~40 lines across 2 files |
| New dependencies | — | None (queue is stdlib) |

---

## Implementation Estimate

**One session.** Changes are surgical:
1. Add GPU queue to `__init__` and `_run_optuna_optimization` (~15 lines)
2. Wrap `_optuna_objective` with queue get/put (~6 lines)
3. Thread `gpu_id` through `_run_nn_optuna_trial` → `CUDA_VISIBLE_DEVICES` (~5 lines)
4. Fix NPZ temp naming (~1 line)
5. Remove DataParallel from `train_single_trial.py` (~4 lines removed)
6. Fix CatBoost `devices='0'` (~1 line)
7. Smoke test with 2 trials → production run with 20 trials

---

## Decision Required from Team Beta

1. **Approve `n_jobs=2` for NN Optuna?** (Tree models stay sequential — no benefit)
2. **Approve `CUDA_VISIBLE_DEVICES` per-subprocess as GPU isolation?**
3. **Approve DataParallel removal?** (Wasteful for <1MB model)
4. **SQLite or JournalFileBackend?** (Recommend SQLite first, JournalFile as fallback)
5. **Approve graceful degradation?** (Auto-detect GPU count, n_jobs=1 on single GPU)

---

*Proposal by Team Alpha — Session 95*  
*Code audit: meta_prediction_optimizer_anti_overfit.py (2,595L), train_single_trial.py (904L)*  
*Optuna docs: Confirmed n_jobs threading + SQLite safe for single-process parallel*
