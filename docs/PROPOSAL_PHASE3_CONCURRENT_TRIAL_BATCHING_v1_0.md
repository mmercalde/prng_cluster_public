# PROPOSAL: Phase 3 — Concurrent NN Trial Batching + Tree Model Parallelism
**Version:** 1.0.0
**Date:** 2026-02-19
**Author:** Team Alpha (Michael)
**Prerequisite:** S96B deployed and verified ✅
**Status:** DRAFT — Awaiting Team Beta Review

---

## 1. Context & Progression

```
S96A:    Full-batch training (all 98K survivors as one tensor, no DataLoader)
         Result: 75-150× speedup per trial. NN trial: 5-10 min → ~4 seconds.

S96B:    Persistent GPU workers (CUDA init once, serve all trials via IPC)
         Result: 2 concurrent trials (1 per GPU). 20 NN trials → ~1-2 min.
         GPU utilization per GPU: still LOW — one trial at a time, GPU idles
         between forward/backward passes.

Phase 3: Concurrent trial batching — run N trials SIMULTANEOUSLY on same GPU
         Pack multiple SurvivorQualityNet instances into 12GB VRAM using
         CUDA streams. GPU utilization approaches 100%.
         20 NN trials / (2 GPUs × N concurrent) → target: < 30 seconds.
```

S96A changelog explicitly noted: *"Phase 3 — if GPU utilization still low after
Phase 2."* S96B confirmed utilization is still low — Phase 3 trigger met.

---

## 2. Why GPU Utilization Is Still Low After S96B

`SurvivorQualityNet` is tiny:

| Metric | Value |
|--------|-------|
| Input features | 62 |
| Typical hidden layers | [256, 128, 64] |
| Parameters | ~50K–200K |
| VRAM per instance | ~150–250 MB |
| Available VRAM (3080 Ti) | 12 GB |
| Theoretical fit | **5–6 instances simultaneously** |

With S96B, GPU 0 runs Trial 181, GPU 1 runs Trial 182. Both GPUs have ~11.5GB
of VRAM sitting idle while each tiny network trains. That's the waste Phase 3
eliminates.

---

## 3. Phase 3A: Concurrent Trial Batching (NN)

### 3.1 Architecture

```
S96B (current):
    GPU-0 Worker:  Trial 181 → done → Trial 183 → done → ...  (serial)
    GPU-1 Worker:  Trial 182 → done → Trial 184 → done → ...  (serial)

Phase 3 (proposed):
    GPU-0 Worker:  [Trial 181 | Trial 183 | Trial 185] → all running simultaneously
    GPU-1 Worker:  [Trial 182 | Trial 184 | Trial 186] → all running simultaneously
                    ↑ 3 CUDA streams per GPU, N models in parallel
```

### 3.2 CUDA Stream Model

Each concurrent trial gets its own CUDA stream:

```python
# Inside nn_gpu_worker.py — concurrent batch execution
streams = [torch.cuda.Stream() for _ in range(n_concurrent)]
models  = []
optims  = []

for i, trial_job in enumerate(batch):
    stream = streams[i % n_concurrent]
    with torch.cuda.stream(stream):
        model = SurvivorQualityNet(**trial_job['arch']).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=trial_job['lr'])
        models.append((model, optim, stream, trial_job))

# Run all epochs concurrently
for epoch in range(max_epochs):
    for model, optim, stream, job in models:
        with torch.cuda.stream(stream):
            pred = model(X_train)
            loss = criterion(pred, y_train)
            optim.zero_grad()
            loss.backward()
            optim.step()

# Synchronize before reading results
torch.cuda.synchronize()
```

### 3.3 Architecture Bucketing (Team Beta Requirement)

Different Optuna trials propose different layer configs. Models with different
architectures cannot share tensor operations, but they CAN share a CUDA stream
safely if launched independently.

**Bucketing strategy:**
- Group incoming trial jobs by hidden layer count (e.g., 2-layer, 3-layer, 4-layer)
- Within each bucket, assign to streams in round-robin
- Prevents CUDA graph recompilation between stream switches

```python
def bucket_trials(trial_jobs):
    """Group by architecture signature for CUDA efficiency."""
    buckets = defaultdict(list)
    for job in trial_jobs:
        sig = tuple(job['params']['hidden_layers'])  # e.g., (256, 128, 64)
        buckets[sig].append(job)
    return buckets
```

### 3.4 VRAM Budget Enforcer

Hard limit per concurrent instance to prevent OOM:

```python
VRAM_BUDGET_MB = 1800  # per concurrent slot (conservative for 12GB / 6 slots)

def estimate_vram_mb(hidden_layers, n_survivors):
    """Rough VRAM estimate for one SurvivorQualityNet instance."""
    params = 62 * hidden_layers[0]
    for i in range(len(hidden_layers) - 1):
        params += hidden_layers[i] * hidden_layers[i+1]
    params += hidden_layers[-1]  # output
    
    # Model weights + activations (forward + backward) + optimizer states (Adam: 2×)
    # Full-batch: activations = n_survivors × hidden_layers[0] × 4 bytes
    model_mb  = params * 4 / 1024 / 1024 * 3   # weights + grad + Adam
    activ_mb  = n_survivors * hidden_layers[0] * 4 / 1024 / 1024 * 2  # fwd + bwd
    return model_mb + activ_mb

def max_concurrent_for_budget(trial_jobs, vram_total_mb=11000):
    """How many trials fit given worst-case VRAM estimate?"""
    worst = max(
        estimate_vram_mb(job['params']['hidden_layers'], 79027)  # train set size
        for job in trial_jobs
    )
    return max(1, int(vram_total_mb / worst))
```

### 3.5 OOM Recovery

```python
try:
    run_concurrent_batch(batch, n_concurrent)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    gc.collect()
    # Retry with half concurrency
    run_concurrent_batch(batch, max(1, n_concurrent // 2))
```

### 3.6 Integration With S96B Workers

Phase 3 extends `nn_gpu_worker.py`, not replaces it:

```
New IPC command: "train_batch"  (vs current "train" for single trial)

Request:
{
  "command": "train_batch",
  "jobs": [
    {"trial_number": 181, "fold_idx": 0, "params": {...}, "X_train_path": "..."},
    {"trial_number": 183, "fold_idx": 0, "params": {...}, "X_train_path": "..."},
    {"trial_number": 185, "fold_idx": 0, "params": {...}, "X_train_path": "..."}
  ],
  "n_concurrent": 3
}

Response (one JSON line per completed trial):
{"trial_number": 181, "fold_idx": 0, "r2": 0.00012, "status": "ok"}
{"trial_number": 183, "fold_idx": 0, "r2": -0.00003, "status": "ok"}
{"trial_number": 185, "fold_idx": 0, "r2": 0.00008, "status": "ok"}
```

The existing `"train"` single-trial command remains unchanged — full backward compat.

### 3.7 Expected Speedup

| Configuration | NN Trials Time | Notes |
|---------------|---------------|-------|
| S96A baseline | ~10 min | Subprocess per trial |
| S96B | ~1–2 min | 2 concurrent (1 per GPU), serial within GPU |
| Phase 3 (N=3) | ~20–40 sec | 6 concurrent (3 per GPU) |
| Phase 3 (N=5) | ~15–25 sec | 10 concurrent (5 per GPU) |

**Best case: 20 NN trials in under 30 seconds.**

---

## 4. Phase 3B: Tree Model Parallelism (Bonus)

Tree models (CatBoost, LightGBM, XGBoost) are now the dominant bottleneck at
~24 min of the ~26 min total. They run sequentially because the GPU isolation
invariant requires process separation between CUDA frameworks.

However, **within each model type**, trials ARE independent and can run in parallel.

### 4.1 Why Trees Can't Use CUDA Streams

Tree models use GPU for matrix ops but manage their own device contexts (not
PyTorch CUDA streams). Multiple CatBoost instances in the same process will
fight over GPU memory management — the isolation invariant exists exactly to
prevent this.

### 4.2 Solution: Multi-Process Tree Workers (One Process Per Concurrent Trial)

```
Current (sequential):
    CatBoost trial 0 → subprocess exits → trial 1 → subprocess exits → ...

Phase 3B (parallel):
    CatBoost trial 0 → subprocess (GPU 0)  ┐
    CatBoost trial 1 → subprocess (GPU 1)  ┘ both running simultaneously
    → both exit → trial 2 (GPU 0) + trial 3 (GPU 1) → ...
```

Each subprocess gets its own `CUDA_VISIBLE_DEVICES`, so there's zero context
sharing — isolation invariant is preserved.

### 4.3 Implementation

```python
# In meta_prediction_optimizer_anti_overfit.py
# Replace sequential tree trial loop with ProcessPoolExecutor

from concurrent.futures import ProcessPoolExecutor, as_completed

def run_tree_trials_parallel(trials, model_type, n_gpu=2):
    """
    Run tree model Optuna trials in parallel, one subprocess per GPU.
    Preserves GPU isolation invariant: each process owns exactly one GPU.
    """
    futures = {}
    with ProcessPoolExecutor(max_workers=n_gpu) as pool:
        for i, trial in enumerate(trials):
            gpu_id = i % n_gpu
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            fut = pool.submit(run_single_tree_trial, trial, model_type, env)
            futures[fut] = trial

        for fut in as_completed(futures):
            result = fut.result()
            optuna_study.tell(futures[fut], result['r2'])
```

### 4.4 Expected Speedup

| Model | Current | With Phase 3B (2 parallel) |
|-------|---------|---------------------------|
| CatBoost (20 trials) | ~8–10 min | ~4–5 min |
| LightGBM (20 trials) | ~3–4 min | ~2 min |
| XGBoost (20 trials) | ~2–3 min | ~1–1.5 min |
| **Total tree time** | **~24 min** | **~12 min** |

Combined with Phase 3A (NN < 30 sec), total Step 5 compare_models:

| Stage | Current | After Phase 3A+3B |
|-------|---------|-------------------|
| NN (20 trials) | ~2 min (S96B) | ~30 sec |
| Tree models (60 trials) | ~24 min | ~12 min |
| **Total** | **~26 min** | **~12–13 min** |

**~50% overall reduction** just from parallelizing trees. The NN gain is now
irrelevant in comparison — trees dominate and that's where the leverage is.

### 4.5 Risk: CatBoost Multi-GPU Awareness

CatBoost has built-in multi-GPU support and may auto-detect both GPUs even when
`CUDA_VISIBLE_DEVICES=0`. Must pin explicitly:

```python
CatBoostRegressor(
    task_type="GPU",
    devices="0",          # NOT "0:1" — pin to single device
    ...
)
```

---

## 5. Implementation Order

### Phase 3A — Concurrent NN Batching (MEDIUM PRIORITY)
NN is already fast (~2 min). This is a nice-to-have, not urgent.

1. Add `"train_batch"` command to `nn_gpu_worker.py`
2. Add `bucket_trials()` and `estimate_vram_mb()` utilities
3. Add OOM recovery with halved concurrency
4. Add `n_concurrent` param to `reinforcement.json` manifest
5. Soak test: 20 trials batch mode, verify R² identical to serial
6. Measure actual GPU utilization improvement with `nvidia-smi dmon`

### Phase 3B — Tree Model Parallel Workers (HIGH PRIORITY)
Trees are 93% of runtime. This is the high-leverage target.

1. Wrap tree trial execution in `ProcessPoolExecutor(max_workers=2)`
2. Pin `CUDA_VISIBLE_DEVICES` per worker process
3. Pin CatBoost `devices="0"` to prevent auto multi-GPU
4. Add `tree_parallel_workers: 1` param to manifest (default=1, backward compat)
5. Soak test: parallel tree run, verify Optuna study results match serial
6. Team Beta review required (changes GPU resource allocation pattern)

---

## 6. Acceptance Tests

### Phase 3A Tests
```bash
# Test 1: Batch command smoke test
echo '{"command":"train_batch","jobs":[...trial0...,...trial1...],"n_concurrent":2}' | \
  CUDA_VISIBLE_DEVICES=0 python3 nn_gpu_worker.py

# Test 2: OOM recovery (force with n_concurrent=10 on small GPU)
# Expected: graceful fallback to n_concurrent=5, then 2

# Test 3: Full 20-trial NN run, compare R² to S96B baseline
# Expected: identical Optuna study results, < 30 sec wall clock
```

### Phase 3B Tests
```bash
# Test 1: Two CatBoost trials simultaneously, verify no GPU contention
CUDA_VISIBLE_DEVICES=0 python3 -c "
import subprocess, time
p0 = subprocess.Popen(['python3', 'run_tree_trial.py', '--gpu', '0', '--trial', '0'])
p1 = subprocess.Popen(['python3', 'run_tree_trial.py', '--gpu', '1', '--trial', '1'])
p0.wait(); p1.wait()
print('Both completed without error')
"

# Test 2: Full compare_models with tree_parallel_workers=2
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \
  --start-step 5 --end-step 5 \
  --params '{"compare_models": true, "trials": 20,
             "persistent_workers": true, "tree_parallel_workers": 2}'
# Expected: total wall clock ~12-14 min (down from 26 min)
```

---

## 7. Files To Create/Modify

| File | Change |
|------|--------|
| `nn_gpu_worker.py` | Add `train_batch` command + CUDA stream concurrency |
| `meta_prediction_optimizer_anti_overfit.py` | Add ProcessPoolExecutor for tree trials |
| `agent_manifests/reinforcement.json` | Add `n_concurrent_nn` + `tree_parallel_workers` params |
| `CHAPTER_6_ANTI_OVERFIT_TRAINING.md` | Document Phase 3 architecture |

---

## 8. Success Criteria

| Metric | Target |
|--------|--------|
| NN 20-trial time | < 30 seconds |
| Tree 20-trial time per model | < 5 min |
| Total compare_models | < 14 min (down from 26 min) |
| R² results | Numerically identical to serial baseline |
| GPU isolation invariant | Zero violations (confirmed by no OpenCL errors) |
| OOM recovery | Graceful degradation, no crashes |

---

## 9. Priority Summary

| Phase | Target | Priority | Blocker |
|-------|--------|----------|---------|
| 3A: NN concurrent batching | < 30 sec NN | Medium | Team Beta review |
| 3B: Tree parallel workers | ~12 min trees | **High** | Team Beta review |

Phase 3B gives 2× more wall-clock improvement than Phase 3A because trees
dominate. However both can be developed in parallel — they touch different
code paths.

---

*Team Alpha / S97*
*Prerequisite: Fix survivors_with_scores.json symlink → run real Step 3 → retrain first*
*Phase 3 should be benchmarked on real data, not test data*
