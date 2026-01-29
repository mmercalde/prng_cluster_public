# Selfplay Architecture Proposal v1.0
## Multi-Model Inner Episode Training with Full CPU/GPU Utilization

**Date:** January 29, 2026  
**Status:** APPROVED by Team Beta + User  
**Document Version:** 1.0

---

## Executive Summary

This proposal defines the selfplay architecture for the distributed PRNG analysis system, specifically addressing how inner episodes (ML model training) should leverage cluster resources. Based on extensive benchmarking conducted on January 28-29, 2026, we recommend:

1. **Tree models (LightGBM, XGBoost, CatBoost) on CPU** for inner episodes
2. **24 rig GPUs for sieving** in outer episodes only
3. **Zeus GPUs (CUDA) for CatBoost/XGBoost** when dataset size warrants
4. **NO neural_net for inner episodes** due to 500,000x worse MSE on tabular data

---

## Part 1: Hardware Inventory

### 1.1 Compute Resources

| Node | CPU | Cores/Threads | GPUs | GPU Framework |
|------|-----|---------------|------|---------------|
| **Zeus** | Intel i9-9920X | 12c/24t | 2× RTX 3080 Ti | CUDA |
| **rig-6600** | Intel i5-9400 | 6c/6t | 12× RX 6600 | ROCm 6.4.3 |
| **rig-6600b** | Intel i5-8400 | 6c/6t | 12× RX 6600 | ROCm 6.4.3 |

**Totals:**
- CPU: 24 cores / 36 threads
- GPU: 26 GPUs (~285 TFLOPS)

### 1.2 Software Environment

| Package | rig-6600 | rig-6600b | Zeus |
|---------|----------|-----------|------|
| PyTorch | 2.7.0+rocm6.3 | 2.7.0+rocm6.3 | 2.x+cu1xx |
| LightGBM | 4.6.0 | 4.6.0 | 4.6.0 |
| XGBoost | 3.1.3 | 3.1.3 | 3.1.3 |
| CatBoost | 1.2.8 | 1.2.8 | 1.2.8 |

---

## Part 2: Benchmark Results (January 29, 2026)

### 2.1 Model Performance on Tabular Data (95K Survivors, Dec 21 2025)

| Model | MSE | Rank | Notes |
|-------|-----|------|-------|
| CatBoost | 1.77e-9 | 🏆 #1 | Best for tabular |
| XGBoost | 9.32e-9 | #2 | Fast, good accuracy |
| LightGBM | 1.06e-8 | #3 | Fastest training |
| Neural Net | 9.32e-4 | #4 | **500,000x worse** |

**Critical Finding:** Neural networks are fundamentally unsuited for tabular/structured data. Tree models must be used for inner episodes.

### 2.2 CPU vs GPU Training Speed (2K samples × 47 features)

| Rig | CPU (12 models) | GPU (12 models) | CPU Advantage |
|-----|-----------------|-----------------|---------------|
| rig-6600 | **1.12s** | 8.79s | **7.9x faster** |
| rig-6600b | **1.08s** | 11.92s | **11x faster** |

| Metric | CPU | GPU |
|--------|-----|-----|
| Throughput | **10-11 models/sec** | 1-1.4 models/sec |
| Per-model time | **~0.09s** | ~0.75s |

### 2.3 Individual Model Training Times (CPU, 2K samples)

| Model | rig-6600 | rig-6600b |
|-------|----------|-----------|
| LightGBM | 0.10s | 0.09s |
| XGBoost | 0.32s | 0.50s |
| CatBoost | 0.36s | 0.34s |

### 2.4 GPU Parallel Training Issues

**Problem:** ROCm OpenCL has contention issues when multiple processes access GPUs simultaneously.

| Test | Result |
|------|--------|
| Single GPU | ✅ Works (1.15s) |
| 12 GPUs parallel (processes) | ❌ Hangs indefinitely |
| 4 GPUs parallel (threads) | ⚠️ Contention (1.3-5.5s) |

**Root Cause:** OpenCL context creation serialization + shared resource contention.

### 2.5 LightGBM GPU Crossover Analysis

| Samples | CPU | GPU (best) | Winner |
|---------|-----|------------|--------|
| 2,000 | 0.10s | 1.61s | CPU (16x) |
| 5,000 | 0.11s | 0.70s | CPU (6x) |
| 10,000 | 0.13s | 0.75s | CPU (6x) |
| 20,000 | 0.16s | 0.79s | CPU (5x) |
| 50,000 | 0.27s | 0.93s | CPU (3x) |
| 100,000 | 0.50s | 1.21s | CPU (2.4x) |

**Conclusion:** LightGBM GPU only becomes competitive at 1M+ samples (per official docs). CPU wins at all selfplay-relevant dataset sizes.

---

## Part 3: Approved Architecture

### 3.1 Two-Phase Selfplay Execution

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SELFPLAY ORCHESTRATOR                           │
│                                                                     │
│  CRITICAL: Does NOT directly SSH to rigs for GPU work!             │
│  Uses existing proven coordinators to prevent ROCm/SSH storms.     │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OUTER EPISODE (Sieving)                          │
│                                                                     │
│  COORDINATION: Uses coordinator.py / scripts_coordinator.py        │
│  (Existing proven infrastructure - DO NOT BYPASS)                  │
│                                                                     │
│  These coordinators handle:                                         │
│    ✅ SSH batching (max concurrent connections)                    │
│    ✅ ROCm init stagger (0.3s delays between workers)              │
│    ✅ Pull-based work distribution                                 │
│    ✅ Cooldowns between batches                                    │
│    ✅ Failure handling and retries                                 │
│                                                                     │
│  Purpose: Generate survivor candidates via bidirectional sieve     │
│  Framework: PyTorch (GPU vectorized operations)                    │
│                                                                     │
│  Zeus:      2× RTX 3080 Ti (CUDA)     → Sieving workers            │
│  rig-6600:  12× RX 6600 (ROCm)        → Sieving workers            │
│  rig-6600b: 12× RX 6600 (ROCm)        → Sieving workers            │
│                                                                     │
│  Total: 26 GPU workers (coordinated, not direct SSH)               │
│  Output: 100-5,000 survivors per episode                           │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INNER EPISODE (ML Training)                      │
│                                                                     │
│  COORDINATION: Lightweight (no ROCm concerns for CPU work)         │
│                                                                     │
│  Why simpler coordination is OK for inner episodes:                │
│    • No OpenCL context creation (GPU-specific issue)               │
│    • No HIP initialization (GPU-specific issue)                    │
│    • No VRAM allocation conflicts (CPU uses RAM only)              │
│    • CPU processes don't compete for shared GPU resources          │
│                                                                     │
│  Options for inner episode coordination:                            │
│    A) Persistent workers spawned once at selfplay start            │
│    B) Lightweight SSH calls (no GPU init overhead)                 │
│    C) Local execution on each node (no cross-node coordination)    │
│                                                                     │
│  Purpose: Evaluate survivor quality via ML model training          │
│  Models: LightGBM, XGBoost, CatBoost ONLY (no neural_net)         │
│                                                                     │
│  Zeus i9-9920X (24 threads):                                        │
│    → 3× CPU workers (8 threads each)                               │
│    → Optionally: 2× GPU workers (CatBoost CUDA)                    │
│                                                                     │
│  rig-6600 i5-9400 (6 threads):                                      │
│    → 2× CPU workers (3 threads each)                               │
│                                                                     │
│  rig-6600b i5-8400 (6 threads):                                     │
│    → 2× CPU workers (3 threads each)                               │
│                                                                     │
│  Total: 7-9 parallel inner episode workers                          │
│  Training time: 0.1-0.5s per model                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Model Support Matrix

| Model | Zeus GPU (CUDA) | Zeus CPU | Rig GPUs (ROCm) | Rig CPUs |
|-------|-----------------|----------|-----------------|----------|
| neural_net | ✅ (sieving only) | ❌ | ✅ (sieving only) | ❌ |
| xgboost | ✅ | ✅ | ❌ | ✅ |
| lightgbm | ✅ (OpenCL) | ✅ | ❌ (too slow) | ✅ |
| catboost | ✅ | ✅ | ❌ | ✅ |

### 3.3 Thread Allocation Strategy

```python
INNER_EPISODE_WORKERS = {
    "zeus": {
        "cpu_workers": 3,
        "threads_per_worker": 8,
        "gpu_workers": 2,  # Optional: CatBoost/XGBoost CUDA
        "models": ["lightgbm", "xgboost", "catboost"]
    },
    "rig-6600": {
        "cpu_workers": 2,
        "threads_per_worker": 3,
        "gpu_workers": 0,  # GPUs idle during inner episode
        "models": ["lightgbm", "xgboost", "catboost"]
    },
    "rig-6600b": {
        "cpu_workers": 2,
        "threads_per_worker": 3,
        "gpu_workers": 0,  # GPUs idle during inner episode
        "models": ["lightgbm", "xgboost", "catboost"]
    }
}
```

### 3.4 Model Configuration

```python
# LightGBM (CPU)
lgb.LGBMRegressor(
    n_estimators=100,
    n_jobs=3,  # Threads per worker (rig) or 8 (Zeus)
    device="cpu",
    verbose=-1
)

# XGBoost (CPU)
xgb.XGBRegressor(
    n_estimators=100,
    n_jobs=3,  # Threads per worker
    tree_method="hist",  # CPU optimized
    verbosity=0
)

# CatBoost (CPU)
CatBoostRegressor(
    iterations=100,
    thread_count=3,  # Threads per worker
    verbose=0
)

# CatBoost (Zeus GPU - optional)
CatBoostRegressor(
    iterations=100,
    task_type="GPU",
    devices="0:1",  # Dual 3080 Ti
    verbose=0
)
```

---

## Part 4: Implementation Requirements

### 4.1 Worker Spawning

Inner episode workers should be spawned as **separate processes** (not threads) to avoid GIL contention and ensure clean resource isolation.

```python
from concurrent.futures import ProcessPoolExecutor

def run_inner_episode(config):
    """Run single inner episode on assigned resources."""
    model_type = config["model_type"]
    n_jobs = config["threads"]
    
    # Train model
    if model_type == "lightgbm":
        model = lgb.LGBMRegressor(n_estimators=100, n_jobs=n_jobs)
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(n_estimators=100, n_jobs=n_jobs)
    elif model_type == "catboost":
        model = CatBoostRegressor(iterations=100, thread_count=n_jobs)
    
    model.fit(X, y)
    return evaluate_model(model, X_test, y_test)
```

### 4.2 Staggered Startup (CRITICAL)

To avoid ROCm initialization storms during sieving phase:

```python
STAGGER_DELAY = 0.3  # seconds between worker starts

for i, worker in enumerate(workers):
    time.sleep(i * STAGGER_DELAY)
    worker.start()
```

### 4.3 Environment Variables

```bash
# On rig-6600 and rig-6600b
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Limit OpenMP threads to prevent oversubscription
export OMP_NUM_THREADS=3  # Match threads_per_worker

# Disable GPU for tree models on rigs
export CUDA_VISIBLE_DEVICES=""  # Not applicable (ROCm)
```

---

## Part 5: Expected Performance

### 5.1 Inner Episode Throughput

| Node | Workers | Models/sec | Notes |
|------|---------|------------|-------|
| Zeus CPU | 3 | ~30 | 8 threads each |
| Zeus GPU | 2 | ~2 | CatBoost CUDA (optional) |
| rig-6600 | 2 | ~20 | 3 threads each |
| rig-6600b | 2 | ~20 | 3 threads each |
| **Total** | **7-9** | **~70** | All CPU |

### 5.2 Complete Selfplay Cycle

```
Outer Episode (26 GPUs):
  - Sieve 100K seeds → 2,000 survivors
  - Time: ~10-30 seconds (parallelized)

Inner Episode (7 CPU workers):
  - Train model on 2,000 survivors
  - Time: ~0.3 seconds per model
  - 7 parallel = 7 models in ~0.3s

Feedback Integration:
  - Evaluate results
  - Update Optuna study
  - Time: ~0.1 seconds

Total Cycle: ~15-35 seconds
```

---

## Part 6: Coordination Requirements (CRITICAL)

### 6.1 GPU Work MUST Use Existing Coordinators

**RULE: Selfplay MUST NOT directly SSH to rigs for GPU sieving work.**

The existing coordinators exist specifically to prevent:
- **SSH storms** (too many concurrent connections)
- **ROCm init storms** (HIP/OpenCL context creation bottleneck)
- **Memory exhaustion** (too many concurrent GPU workers)

| Coordinator | Use For | Storm Prevention |
|-------------|---------|------------------|
| `coordinator.py` | Seed-based jobs (Steps 1-2) | SSH batching, pull-based |
| `scripts_coordinator.py` | Script-based jobs (Steps 3-6) | Batching, cooldowns, stagger |

### 6.2 Selfplay Orchestrator Flow

```python
# CORRECT: Selfplay uses coordinators
def run_outer_episode(params):
    """Outer episode - GPU sieving via coordinator."""
    
    # Generate job definitions
    jobs = generate_sieve_jobs(params)
    
    # Submit to coordinator (handles batching, stagger, etc.)
    results = scripts_coordinator.submit_jobs(
        jobs=jobs,
        max_concurrent=4,      # Per-rig limit
        stagger_delay=0.3,     # ROCm init protection
        cooldown=2.0           # Between batches
    )
    
    return results

# WRONG: Direct SSH causes storms
def run_outer_episode_BAD(params):
    """DO NOT DO THIS - causes ROCm/SSH storms!"""
    for seed_range in seed_ranges:
        for rig in rigs:
            for gpu in range(12):
                # WRONG! Direct SSH per job
                ssh(rig, f"python sieve.py --gpu {gpu}")  # STORM!
```

### 6.3 CPU Work - Simpler Coordination OK

Inner episodes (CPU tree model training) do NOT require the full coordinator infrastructure because:

| GPU Issue | CPU Equivalent | Problem? |
|-----------|----------------|----------|
| OpenCL context creation | N/A | ❌ No |
| HIP initialization | N/A | ❌ No |
| VRAM allocation | RAM allocation | ❌ No (plenty of RAM) |
| GPU resource contention | CPU scheduling | ❌ No (OS handles it) |

**Acceptable approaches for inner episodes:**

```python
# Option A: Persistent workers (recommended)
# Spawn once at selfplay start, workers pull from queue
workers = spawn_persistent_workers(config)
results = workers.process_batch(survivors)

# Option B: Simple SSH (OK for CPU work)
# No batching needed, but don't spawn 100s simultaneously
for rig in rigs:
    ssh(rig, f"python inner_episode.py --threads 3")

# Option C: Local execution only
# Each node runs inner episodes on its own survivors
```

### 6.4 Coordination Decision Matrix

| Work Type | Direct SSH OK? | Use Coordinator? | Stagger Required? |
|-----------|----------------|------------------|-------------------|
| GPU Sieving | ❌ NO | ✅ YES (mandatory) | ✅ YES (0.3s) |
| CPU ML Training | ✅ YES | Optional | ❌ NO |
| Data Transfer | ✅ YES | Optional | ❌ NO |

---

## Part 7: Restrictions and Warnings

### 7.1 DO NOT USE

| Item | Reason |
|------|--------|
| neural_net for inner episodes | 500,000x worse MSE on tabular data |
| LightGBM GPU on rigs | 8-11x slower than CPU |
| Parallel GPU processes on ROCm | OpenCL contention causes hangs |
| More than 3 threads/worker on rigs | Oversubscription (only 6 cores) |
| **Direct SSH for GPU sieving jobs** | **Causes ROCm/SSH storms - use coordinators!** |

### 7.2 REQUIRED

| Item | Reason |
|------|--------|
| **coordinator.py / scripts_coordinator.py for GPU work** | **Prevents ROCm/SSH storms** |
| Staggered worker startup (GPU only) | Prevents ROCm initialization storms |
| Process isolation | Avoids GIL and resource conflicts |
| HSA_OVERRIDE_GFX_VERSION=10.3.0 | Required for RX 6600 on ROCm |
| Kill zombie processes between runs | Prevents OpenCL resource exhaustion |

### 7.3 Zombie Process Cleanup

After failed runs, always clean up:

```bash
# On each rig
pkill -9 -f python3
rocm-smi  # Verify GPUs show 0% usage, 0% VRAM
```

---

## Part 8: Configuration Files

### 7.1 selfplay_config.json

```json
{
  "outer_episode": {
    "framework": "pytorch",
    "device": "gpu",
    "seed_range": [0, 100000],
    "batch_size": 10000,
    "stagger_delay": 0.3
  },
  "inner_episode": {
    "models": ["lightgbm", "xgboost", "catboost"],
    "device": "cpu",
    "dataset_size_threshold": 100000,
    "use_zeus_gpu_above_threshold": true
  },
  "workers": {
    "zeus": {
      "cpu_workers": 3,
      "cpu_threads_per_worker": 8,
      "gpu_workers": 0
    },
    "rig-6600": {
      "cpu_workers": 2,
      "cpu_threads_per_worker": 3,
      "gpu_workers": 0
    },
    "rig-6600b": {
      "cpu_workers": 2,
      "cpu_threads_per_worker": 3,
      "gpu_workers": 0
    }
  }
}
```

### 7.2 Integration with WATCHER Agent

The WATCHER agent should be updated to:

1. Spawn inner episode workers based on `selfplay_config.json`
2. Monitor CPU usage (not GPU) during inner episodes
3. Handle zombie process cleanup on failures
4. Report model type and training times in diagnostics

---

## Part 9: Learning Telemetry (Optional but Recommended)

### 9.1 Purpose

Learning telemetry provides **observability without control**. It does not affect decisions — it gives humans early warning and helps explain why learning stalled or accelerated.

### 9.2 Telemetry Schema

```json
{
  "learning_health": {
    "timestamp": "2026-01-29T12:00:00Z",
    "inner_episode_throughput": 68.2,
    "policy_entropy": 0.41,
    "recent_reward_trend": "+3.2%",
    "last_promotion_days_ago": 4,
    "models_trained_total": 1247,
    "current_best_policy": "policy_v3_2_1",
    "survivor_count_avg": 2340,
    "training_time_avg_ms": 142
  }
}
```

### 9.3 Metric Definitions

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `inner_episode_throughput` | Models trained per second (cluster-wide) | 50-80 |
| `policy_entropy` | Diversity of parameter exploration (0=converged, 1=random) | 0.2-0.6 |
| `recent_reward_trend` | Rolling 10-trial reward change | > -5% |
| `last_promotion_days_ago` | Days since Chapter 13 promoted a policy | < 14 |
| `models_trained_total` | Cumulative models trained this session | Increasing |
| `current_best_policy` | Active policy identifier | Valid ID |
| `survivor_count_avg` | Average survivors per outer episode | 100-5000 |
| `training_time_avg_ms` | Average inner episode time | < 500 |

### 9.4 Warning Thresholds

| Condition | Warning | Action |
|-----------|---------|--------|
| `throughput < 30` | ⚠️ Low throughput | Check for stuck workers |
| `entropy < 0.1` | ⚠️ Premature convergence | Increase exploration |
| `reward_trend < -10%` | ⚠️ Learning regression | Review recent changes |
| `promotion_days > 21` | ⚠️ Stalled promotion | Investigate pipeline |

### 9.5 Integration Points

| Component | Telemetry Role |
|-----------|----------------|
| Selfplay Orchestrator | Writes telemetry after each cycle |
| WATCHER Agent | Displays telemetry in status |
| Web Dashboard | Visualizes trends |
| Chapter 13 | Reads for diagnostics (no control) |

### 9.6 Critical Constraint

**Telemetry is READ-ONLY for all automated systems.**

- ✅ Humans can view and interpret
- ✅ Dashboards can display
- ✅ Logs can record
- ❌ No system may use telemetry to make decisions
- ❌ No LLM may use telemetry to control learning

This maintains the separation established in earlier proposals:
- Learning happens **statistically** (tree models + bandit policy)
- Verification happens **deterministically** (Chapter 13)
- Telemetry happens **observationally** (no control path)

---

## Part 10: Future Considerations

### 8.1 When to Re-evaluate GPU Training

GPU training should be reconsidered if:
- Survivor counts exceed 100K consistently
- AMD releases improved OpenCL drivers
- LightGBM adds native ROCm/HIP support

### 8.2 Potential Optimizations

1. **Persistent workers:** Keep tree model workers alive between cycles
2. **Model caching:** Reuse trained models across similar parameter sets
3. **Adaptive threading:** Adjust thread counts based on load

---

## Approval Signatures

| Role | Status | Date |
|------|--------|------|
| Team Beta | ✅ APPROVED | 2026-01-29 |
| User (Michael) | ✅ APPROVED | 2026-01-29 |
| Claude | ✅ DRAFTED | 2026-01-29 |

---

## Appendix A: Raw Benchmark Data

### A.1 LightGBM CPU Training (rig-6600)

```
Training tests (2000 samples × 47 features):
LightGBM CPU:  0.10s
XGBoost CPU:   0.32s
CatBoost CPU:  0.36s
```

### A.2 LightGBM GPU Training (rig-6600)

```
=== Finding GPU/CPU Crossover Point ===
Samples    CPU      GPU(255)   GPU(63)   GPU Winner?
-------------------------------------------------------
  2000    0.10s    1.61s      7.57s      ❌ NO
  5000    0.11s    1.25s      0.70s      ❌ NO
 10000    0.13s    1.23s      0.75s      ❌ NO
 20000    0.16s    1.30s      0.82s      ❌ NO
 50000    0.27s    1.48s      0.93s      ❌ NO
100000    0.50s    1.76s      1.21s      ❌ NO
```

### A.3 Sequential Throughput Comparison

```
=== CPU Sequential (12 models) ===
Total: 1.12s = 10.7 models/sec

=== GPU 0 Sequential (12 models) ===
Total: 8.79s = 1.4 models/sec

=== WINNER: CPU (7.9x faster) ===
```

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-29 | Initial proposal based on benchmark results |
| 1.1 | 2026-01-29 | Added Part 6 (Coordination Requirements), Part 9 (Learning Telemetry) |

---

**END OF PROPOSAL**
