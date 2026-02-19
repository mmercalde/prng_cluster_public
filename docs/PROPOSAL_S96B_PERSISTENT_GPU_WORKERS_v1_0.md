# PROPOSAL: S96B — Persistent GPU Worker Processes for NN Training
**Version:** 1.0  
**Date:** 2026-02-18  
**Author:** Claude (Team Alpha Lead Dev)  
**Status:** Submitted for Team Beta review  
**Depends On:** S96A (Full-Batch NN Training) — DEPLOYED AND VERIFIED  
**Scope:** `train_single_trial.py`, `meta_prediction_optimizer_anti_overfit.py`

---

## 1. Problem Statement

S96A eliminated the mini-batch DataLoader overhead, reducing per-trial NN training from 5-10 minutes to ~4 seconds. However, production profiling reveals that **85% of the 4-second trial time is subprocess overhead, not GPU compute:**

| Component | Time | % of Trial |
|-----------|------|------------|
| Python interpreter boot | ~0.5s | 13% |
| `import torch` + CUDA init | ~1.5s | 38% |
| NPZ load from disk | ~0.3s | 8% |
| StandardScaler + tensor creation | ~0.2s | 5% |
| **Actual GPU training** | **~0.5s** | **13%** |
| Result serialization + exit | ~0.3s | 8% |
| Subprocess spawn/wait overhead | ~0.7s | 18% |

With 20 Optuna trials × 5-fold CV = 100 subprocess launches, the total overhead is **~350 seconds of process management** for **~50 seconds of actual GPU work**. GPU utilization during a compare-models NN pass is approximately **5%**.

### Evidence (S96A Production Run)

```
50 trials completed: 15:53:13 → 16:02:58 = 9m 45s
Per trial average: 11.7 seconds (including 5-fold CV)
Per fold actual: ~0.5s GPU compute
Theoretical minimum (zero overhead): ~50 seconds total
```

---

## 2. Proposed Solution: Persistent GPU Worker Processes

### 2.1 Architecture

Replace per-trial subprocess spawning with **long-lived worker processes** — one per GPU — that boot once and process multiple trials via stdin/stdout IPC.

```
                    Optuna Optimizer (parent process — no torch)
                           │
                ┌──────────┴──────────┐
                │                     │
         GPU-0 Worker            GPU-1 Worker
       (persistent subprocess)  (persistent subprocess)
                │                     │
         ┌──────┤                ┌──────┤
         │      │                │      │
      T0/F0  T0/F2  T1/F1    T0/F1  T0/F3  T1/F0
      T1/F3  T2/F0  T2/F2    T1/F2  T2/F1  T2/F3
         │      │                │      │
         └──────┘                └──────┘
              │                       │
        (stays alive              (stays alive
         until optimize()          until optimize()
         completes)                completes)
```

### 2.2 Worker Lifecycle

1. **Spawn:** At start of `_run_optuna_optimization()`, spawn one worker per available GPU
2. **Initialize:** Each worker boots Python, imports torch, initializes CUDA — **once**
3. **Ready signal:** Worker writes `{"status": "ready"}` to stdout
4. **Trial loop:** Parent writes trial config JSON to worker's stdin, worker trains and writes result JSON to stdout
5. **Shutdown:** Parent writes `{"command": "shutdown"}` to stdin, worker exits cleanly

### 2.3 IPC Protocol

**Parent → Worker (stdin, one JSON line per trial):**
```json
{
  "command": "train",
  "X_train_path": "/tmp/step5_nn_xxx/split_t0_f0.npz",
  "params": {"hidden_layers": [128, 64], "learning_rate": 0.001, "dropout": 0.3, "epochs": 100},
  "trial_number": 0,
  "fold_idx": 0,
  "normalize_features": true,
  "use_leaky_relu": true,
  "batch_mode": "auto"
}
```

**Worker → Parent (stdout, one JSON line per result):**
```json
{
  "status": "complete",
  "model_type": "neural_net",
  "r2": -0.0001,
  "val_mse": 0.000182,
  "train_mse": 0.000175,
  "duration": 0.47,
  "device": "cuda:0 (NVIDIA GeForce RTX 3080 Ti)",
  "trial_number": 0,
  "fold_idx": 0
}
```

**Shutdown:**
```json
{"command": "shutdown"}
```

### 2.4 NPZ Data Sharing

NPZ files continue to be written to disk by the parent (unchanged from S96A). Workers load NPZ per trial. Future optimization: shared memory (`/dev/shm`) for zero-copy, but disk is fine for Phase 2 since NPZ load is only 0.3s and amortized.

---

## 3. Files Modified

### 3.1 New File: `nn_gpu_worker.py`

Persistent worker process. Responsibilities:
- Boot torch, CUDA once
- Enter stdin read loop
- For each trial config: load NPZ, normalize, build model, train, report results
- Clean up GPU memory between trials (`torch.cuda.empty_cache()`)
- Handle errors per-trial (report failure, continue processing)

**Key invariant:** This file imports torch at module level (it IS the subprocess). Parent never imports torch.

### 3.2 Modified: `meta_prediction_optimizer_anti_overfit.py`

Changes to `AntiOverfitMetaOptimizer`:

| Method | Change |
|--------|--------|
| `_run_optuna_optimization()` | Spawn persistent workers before `study.optimize()`, shutdown after |
| `_run_nn_optuna_trial()` | Replace subprocess.run() with write-to-worker-stdin + read-result |
| `_run_nn_via_subprocess()` | Final model: send to worker if alive, fall back to subprocess |

### 3.3 Unchanged: `train_single_trial.py`

No modifications. Remains available as fallback if worker dies.

---

## 4. Invariants Preserved

| Invariant | How |
|-----------|-----|
| Parent stays CUDA-clean (S72/S73) | Workers are subprocesses; parent never imports torch |
| Subprocess isolation | Workers are separate processes with own CUDA contexts |
| GPU lease queue (S95) | Workers are assigned GPUs at spawn time via `CUDA_VISIBLE_DEVICES` |
| Category B (normalize + LeakyReLU) | Worker reads flags from trial config JSON |
| Diagnostics non-fatal | Worker wraps diagnostics in try/except, same as train_single_trial.py |
| Optuna study integrity | Trial results returned same as before; Optuna doesn't know about workers |

---

## 5. Fallback Strategy

If a worker crashes (OOM, CUDA error, segfault):

1. Parent detects broken pipe / EOF on worker stdout
2. Parent logs `[S96B] Worker GPU-{N} died, falling back to subprocess mode`
3. Remaining trials for that GPU fall back to `train_single_trial.py` subprocess (S96A path)
4. Other GPU's worker continues unaffected
5. No data loss — trial simply retries via subprocess

This means S96B is **never worse** than S96A. Worker crash = graceful degradation to proven subprocess mode.

---

## 6. Expected Performance

| Metric | S95 (old) | S96A (current) | S96B (proposed) |
|--------|-----------|----------------|-----------------|
| Per-fold time | 5-10 min | ~4 sec | ~0.5 sec |
| 20 trials × 5 folds | 4-8 hours | ~10 min | ~1-2 min |
| Compare-models (4 types) | 8+ hours | ~15 min | ~5-7 min |
| GPU utilization (NN) | <5% | ~5% | ~40-60% |
| Subprocess launches | 100 | 100 | 2 (workers only) |

**Conservative estimate: 5-10x speedup on top of S96A.**

---

## 7. Kill Switch

If workers cause issues in production:

```python
# In watcher_policies.json
{
  "training": {
    "use_persistent_workers": false  # Falls back to S96A subprocess-per-trial
  }
}
```

Default: `true` when present, `false` when absent (backward compatible).

---

## 8. Acceptance Tests

### Test 0: Syntax verification
```bash
python3 -c "import ast; ast.parse(open('nn_gpu_worker.py').read()); print('OK')"
```

### Test 1: Worker standalone smoke test
```bash
echo '{"command":"train","X_train_path":"/tmp/test_s96a.npz","params":{"hidden_layers":[64,32],"learning_rate":0.001,"epochs":5},"trial_number":0,"fold_idx":0,"normalize_features":true,"use_leaky_relu":true,"batch_mode":"full"}' | \
CUDA_VISIBLE_DEVICES=0 python3 nn_gpu_worker.py
```
Expected: Worker prints ready signal, processes trial, prints result JSON, waits for more input.

### Test 2: Multi-trial sequential
```bash
# Send 3 trials, then shutdown
printf '{"command":"train",...trial0...}\n{"command":"train",...trial1...}\n{"command":"train",...trial2...}\n{"command":"shutdown"}\n' | \
CUDA_VISIBLE_DEVICES=0 python3 nn_gpu_worker.py
```
Expected: 3 result lines + clean exit. No CUDA reinit between trials.

### Test 3: Fallback on worker death
```bash
# Kill worker mid-run, verify parent recovers to subprocess mode
python3 meta_prediction_optimizer_anti_overfit.py \
  --survivors survivors_with_scores.json \
  --lottery-data train_history.json \
  --model-type neural_net \
  --trials 5
# (manually kill worker during run, verify trials complete via fallback)
```

### Test 4: Full compare-models through WATCHER
```bash
rm diagnostics_outputs/model_skip_registry.json 2>/dev/null
rm models/reinforcement/best_model.* 2>/dev/null
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \
  --start-step 5 --end-step 5 \
  --params '{"compare_models": true, "trials": 20}'
```
Expected: All 4 models complete. NN timing < 2 minutes. No worker crashes.

### Test 5: Timing comparison
```bash
# Run 20 NN trials, compare wall clock to S96A baseline
time python3 meta_prediction_optimizer_anti_overfit.py \
  --survivors survivors_with_scores.json \
  --lottery-data train_history.json \
  --model-type neural_net \
  --trials 20
```
Expected: < 2 minutes (vs ~10 minutes S96A baseline).

---

## 9. Scope Boundary — What This Does NOT Include

- **Phase 3 (trial batching on single GPU):** Packing multiple models into one CUDA context. Deferred — requires architecture bucketing and concurrent stream management. Separate proposal if needed after Phase 2 validates.
- **ROCm rig workers:** Phase 2 targets Zeus 3080 Ti GPUs only. Extending to ROCm rigs is a separate effort if Step 5 ever moves to distributed execution.
- **Tree model workers:** XGBoost/LightGBM/CatBoost don't have the subprocess spawn overhead problem (they run inline). No changes to tree model paths.

---

## 10. Implementation Estimate

| Task | Effort |
|------|--------|
| `nn_gpu_worker.py` | ~150 lines, 2 hours |
| `meta_prediction_optimizer_anti_overfit.py` modifications | ~80 lines changed, 1 hour |
| Testing (Tests 0-5) | 1 hour |
| Documentation + changelog | 30 min |
| **Total** | **~4-5 hours** |

---

## 11. Decision Requested

- [ ] **APPROVED** — Implement as proposed
- [ ] **APPROVED WITH MODIFICATIONS** — (specify changes)
- [ ] **DECLINED** — (specify reason)
- [ ] **NEEDS MORE INFO** — (specify questions)

---

*Filed: 2026-02-18*  
*Review Required: Team Beta*  
*Prerequisite: S96A deployed and verified (COMPLETE)*
