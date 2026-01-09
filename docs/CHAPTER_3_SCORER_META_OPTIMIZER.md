# Chapter 3: Scorer Meta-Optimizer (Step 2.5)

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 3.4  
**File:** `scorer_trial_worker.py`  
**Lines:** ~350  
**Purpose:** Execute single scorer meta-optimization trial on remote GPU

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Position](#2-architecture-position)
3. [Pull Architecture](#3-pull-architecture)
4. [Version History](#4-version-history)
5. [Execution Flow](#5-execution-flow)
6. [Trial Parameters](#6-trial-parameters)
7. [Training/Holdout Split](#7-trainingholdout-split)
8. [GPU-Vectorized Scoring](#8-gpu-vectorized-scoring)
9. [Adaptive Memory Batching](#9-adaptive-memory-batching)
10. [Result Serialization](#10-result-serialization)
11. [CLI Interface](#11-cli-interface)
12. [Integration with Coordinator](#12-integration-with-coordinator)
13. [Troubleshooting](#13-troubleshooting)
14. [Complete Method Reference](#14-complete-method-reference)

---

## 1. Overview

### 1.1 What the Scorer Trial Worker Does

The scorer trial worker executes a **single Optuna trial** for scorer hyperparameter optimization:

- **Receives trial parameters** from coordinator via JSON file or CLI args
- **Scores survivors** using specified residue/temporal parameters
- **Evaluates on holdout set** to measure generalization
- **Writes result JSON** locally for coordinator to pull

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Single Trial Execution** | One GPU, one parameter configuration |
| **Pull Architecture** | Results written locally, coordinator pulls via SCP |
| **GPU-Vectorized Scoring** | PyTorch/CuPy batch processing |
| **Adaptive Memory Batching** | Adjusts batch size for 8GB VRAM |
| **Holdout Evaluation** | v3.4 critical fix for proper validation |

### 1.3 Pipeline Position

```
Step 2.5: Scorer Meta-Optimizer
    │
    ├── run_scorer_meta_optimizer.py (orchestrator)
    │       │
    │       ├── generate_scorer_jobs.py (creates trial jobs)
    │       │
    │       └── coordinator.py (distributes to 26 GPUs)
    │               │
    │               └── scorer_trial_worker.py ◄── THIS FILE
    │                       │
    │                       └── Writes: scorer_trial_results/trial_*.json
    │
    └── coordinator.collect_scorer_results() (pulls results)
```

---

## 2. Architecture Position

### 2.1 In the Optimization Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SCORER META-OPTIMIZER (Step 2.5)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Optuna Study (Zeus)                                                     │
│       │                                                                  │
│       ├── Trial 0 ──► GPU 0 ──► scorer_trial_worker.py                  │
│       ├── Trial 1 ──► GPU 1 ──► scorer_trial_worker.py                  │
│       ├── Trial 2 ──► GPU 2 ──► scorer_trial_worker.py                  │
│       │   ...                                                            │
│       └── Trial 99 ──► GPU 25 ──► scorer_trial_worker.py                │
│                                                                          │
│  Each worker:                                                            │
│   1. Receives params (residue_mod_1/2/3, max_offset, temporal_window)   │
│   2. Scores survivors with those params                                  │
│   3. Evaluates on holdout data                                          │
│   4. Writes result JSON locally                                         │
│                                                                          │
│  Coordinator pulls all results → Optuna aggregates → Best params        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Why Separate Worker?

| Approach | Problem |
|----------|---------|
| Direct Optuna workers | Database contention with 26 concurrent connections |
| Shared storage | NFS/Samba bottleneck, file locking issues |
| **Pull architecture** | ✅ Workers write locally, coordinator pulls via SCP |

---

## 3. Pull Architecture

### 3.1 How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                         PULL ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  STEP 1: Coordinator dispatches trial                            │
│          Zeus ───SSH───► rig-6600 (GPU 5)                        │
│                          scorer_trial_worker.py --trial-id 42    │
│                                                                   │
│  STEP 2: Worker executes and writes locally                      │
│          rig-6600:/home/michael/distributed_prng_analysis/       │
│                   scorer_trial_results/trial_42.json             │
│                                                                   │
│  STEP 3: Coordinator pulls result via SCP                        │
│          Zeus ───SCP───► rig-6600                                │
│          GET scorer_trial_results/trial_42.json                  │
│          DELETE remote file after successful pull                │
│                                                                   │
│  STEP 4: Coordinator aggregates all results                      │
│          Optuna study updated with trial results                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Result File Location

```
~/distributed_prng_analysis/
    scorer_trial_results/
        trial_0.json
        trial_1.json
        trial_2.json
        ...
        trial_99.json
```

### 3.3 Why Pull Instead of Push?

| Push (worker → coordinator) | Pull (coordinator → worker) |
|----------------------------|----------------------------|
| Workers need coordinator address | Workers only need local filesystem |
| Network errors lose results | Results persist until pulled |
| Complex retry logic | Simple: pull, verify, delete |
| Coordinator must be listening | Coordinator polls when ready |

---

## 4. Version History

### 4.1 Version Timeline

```
Version 3.4 - December 2025 (CURRENT)
├── CRITICAL FIX: Holdout evaluation uses sampled seeds
├── Previous: Holdout evaluated on ALL seeds (wrong!)
└── Now: Holdout evaluated on SAME sampled seeds as training

Version 3.3 - November 2025
├── GPU-vectorized scoring via SurvivorScorer.extract_ml_features_batch()
├── Adaptive memory batching for 8GB VRAM
└── 3.8x performance improvement

Version 3.2 - November 2025
├── --params-file support for shorter SSH commands
├── Previous: All params passed via CLI args (SSH command too long)
└── Now: Params written to JSON file, worker reads file

Version 3.1 - October 2025
├── ROCm environment setup at file top
└── Mining mode support for AMD rigs

Version 3.0 - October 2025
├── Initial distributed implementation
└── Basic trial execution
```

### 4.2 v3.4 Critical Fix Details

**The Bug (v3.3 and earlier):**
```python
# WRONG: Training used sampled seeds, holdout used ALL seeds
train_score = score_survivors(sampled_seeds, train_history)
holdout_score = score_survivors(ALL_survivors, holdout_history)  # BUG!
```

**The Fix (v3.4):**
```python
# CORRECT: Both use the SAME sampled seeds
sampled_seeds = random.sample(survivors, sample_size)
train_score = score_survivors(sampled_seeds, train_history)
holdout_score = score_survivors(sampled_seeds, holdout_history)  # FIXED!
```

**Why It Matters:**
- Holdout evaluation must measure how well the model generalizes
- Using different seeds biases the holdout metric
- v3.4 ensures apples-to-apples comparison

---

## 5. Execution Flow

### 5.1 Main Flow Diagram

```
scorer_trial_worker.py main()
    │
    ├──► Parse arguments (--params-file or CLI args)
    │
    ├──► Load data files:
    │    ├─ survivors.json (from Step 1)
    │    ├─ train_history.json (80% of draws)
    │    └─ holdout_history.json (20% of draws)
    │
    ├──► Initialize SurvivorScorer with trial params:
    │    ├─ residue_mod_1 (5-20)
    │    ├─ residue_mod_2 (50-150)
    │    ├─ residue_mod_3 (500-1500)
    │    ├─ max_offset (1-15)
    │    └─ temporal_window_size (50-200)
    │
    ├──► Sample survivors (if too many):
    │    └─ sample_size = min(10000, len(survivors))
    │
    ├──► Score on training data:
    │    └─ GPU-vectorized batch scoring
    │
    ├──► Evaluate on holdout data (v3.4 fix):
    │    └─ SAME sampled seeds, different history
    │
    ├──► Compute trial metrics:
    │    ├─ mean_train_score
    │    ├─ mean_holdout_score
    │    ├─ generalization_gap
    │    └─ top_k_holdout_score
    │
    └──► Write result JSON:
         └─ scorer_trial_results/trial_{trial_id}.json
```

### 5.2 Code Structure

```python
def main():
    args = parse_arguments()
    
    # Load params from file or CLI
    if args.params_file:
        params = load_params_from_file(args.params_file, args.trial_id)
    else:
        params = extract_params_from_args(args)
    
    # Load data
    survivors = load_json(args.survivors_file)
    train_history = load_json(args.train_history_file)
    holdout_history = load_json(args.holdout_history_file)
    
    # Run trial
    result = run_trial(
        survivors=survivors,
        train_history=train_history,
        holdout_history=holdout_history,
        params=params,
        trial_id=args.trial_id
    )
    
    # Save result
    save_result(result, args.trial_id)
```

---

## 6. Trial Parameters

### 6.1 Optuna Search Space

| Parameter | Range | Purpose |
|-----------|-------|---------|
| `residue_mod_1` | 5-20 | Small residue modulus |
| `residue_mod_2` | 50-150 | Medium residue modulus |
| `residue_mod_3` | 500-1500 | Large residue modulus |
| `max_offset` | 1-15 | Temporal alignment offset |
| `temporal_window_size` | [50, 100, 150, 200] | Stability analysis window |

### 6.2 Parameter File Format (v3.2+)

**scorer_jobs.json:**
```json
{
    "study_name": "scorer_meta_v1",
    "study_db": "sqlite:///optuna_scorer.db",
    "trials": [
        {
            "trial_id": 0,
            "params": {
                "residue_mod_1": 14,
                "residue_mod_2": 137,
                "residue_mod_3": 1136,
                "max_offset": 3,
                "temporal_window_size": 50
            }
        },
        {
            "trial_id": 1,
            "params": {
                "residue_mod_1": 8,
                "residue_mod_2": 92,
                "residue_mod_3": 847,
                "max_offset": 7,
                "temporal_window_size": 100
            }
        }
    ]
}
```

### 6.3 Why --params-file?

```bash
# OLD (v3.1): Very long SSH command
ssh rig-6600 'python scorer_trial_worker.py survivors.json train.json holdout.json \
    --trial-id 42 --residue-mod-1 14 --residue-mod-2 137 --residue-mod-3 1136 \
    --max-offset 3 --temporal-window-size 50 --optuna-study-name scorer_meta_v1 \
    --optuna-study-db sqlite:///optuna.db'

# NEW (v3.2+): Short command, params in file
ssh rig-6600 'python scorer_trial_worker.py survivors.json train.json holdout.json \
    42 --params-file scorer_jobs.json --optuna-study-name scorer_meta_v1 \
    --optuna-study-db sqlite:///optuna.db'
```

---

## 7. Training/Holdout Split

### 7.1 Data Flow from Step 1

```
Step 1 (Window Optimizer) outputs:
    │
    ├── bidirectional_survivors.json (intersection of forward+reverse)
    │
    ├── train_history.json (80% of lottery draws)
    │
    └── holdout_history.json (20% of lottery draws)

Step 2.5 (Scorer Meta-Optimizer) uses:
    │
    ├── survivors from Step 1
    │
    ├── train_history for scoring during optimization
    │
    └── holdout_history for validation (unseen data)
```

### 7.2 Holdout Evaluation (v3.4)

```python
def run_trial(survivors, train_history, holdout_history, params, trial_id):
    """
    Run single scorer trial with proper holdout evaluation.
    
    v3.4 CRITICAL FIX: Use SAME sampled seeds for both train and holdout.
    """
    # Initialize scorer with trial params
    scorer = SurvivorScorer(
        prng_type='java_lcg',
        mod=1000,
        config_dict={
            'residue_mod_1': params['residue_mod_1'],
            'residue_mod_2': params['residue_mod_2'],
            'residue_mod_3': params['residue_mod_3'],
            'max_offset': params['max_offset'],
            'temporal_window_size': params['temporal_window_size']
        }
    )
    
    # Sample survivors (same sample for both evaluations!)
    sample_size = min(10000, len(survivors))
    sampled_seeds = random.sample([s['seed'] for s in survivors], sample_size)
    
    # Score on training data
    train_features = scorer.extract_ml_features_batch(
        seeds=sampled_seeds,
        lottery_history=train_history
    )
    train_scores = [f['score'] for f in train_features]
    
    # Score on holdout data (SAME seeds, different history)
    holdout_features = scorer.extract_ml_features_batch(
        seeds=sampled_seeds,  # v3.4 FIX: Same seeds!
        lottery_history=holdout_history
    )
    holdout_scores = [f['score'] for f in holdout_features]
    
    # Compute metrics
    return {
        'trial_id': trial_id,
        'params': params,
        'mean_train_score': np.mean(train_scores),
        'mean_holdout_score': np.mean(holdout_scores),
        'generalization_gap': np.mean(train_scores) - np.mean(holdout_scores),
        'std_train_score': np.std(train_scores),
        'std_holdout_score': np.std(holdout_scores),
        'n_survivors_scored': len(sampled_seeds),
        'top_k_holdout_score': np.mean(sorted(holdout_scores, reverse=True)[:100])
    }
```

### 7.3 Why Holdout Matters

| Metric | Meaning |
|--------|---------|
| `mean_train_score` | How well params fit the training data |
| `mean_holdout_score` | How well params generalize to unseen data |
| `generalization_gap` | Train - Holdout (large = overfitting) |
| `top_k_holdout_score` | Best survivors on holdout (what we care about) |

---

## 8. GPU-Vectorized Scoring

### 8.1 Batch Processing (v3.3+)

```python
def score_survivors_gpu(scorer, seeds, lottery_history):
    """
    GPU-vectorized scoring using SurvivorScorer.extract_ml_features_batch()
    
    v3.3: Crypto miner style - all parallel, single CPU transfer at end
    """
    features_list = scorer.extract_ml_features_batch(
        seeds=seeds,
        lottery_history=lottery_history
    )
    return features_list
```

### 8.2 Performance Comparison

| Method | 10K Seeds | Notes |
|--------|-----------|-------|
| Single-seed loop | ~300 sec | CPU-bound |
| GPU batch (v3.3) | ~80 sec | 3.8x faster |

### 8.3 Under the Hood

```
SurvivorScorer.extract_ml_features_batch():
    │
    ├── STEP 1: Generate ALL predictions on GPU
    │   └── PyTorch tensors, single kernel launch
    │
    ├── STEP 2: Compute ALL features on GPU
    │   ├── Base matching (vectorized)
    │   ├── Stats (mean, std, etc.)
    │   ├── Residuals (vectorized histogram via scatter_add)
    │   └── Temporal stability
    │
    └── STEP 3: SINGLE CPU transfer
        └── Stack all tensors → one .cpu().numpy() call
```

---

## 9. Adaptive Memory Batching

### 9.1 The Problem

```
8GB VRAM (RX 6600):
    10K seeds × 5000 draws × float32 = 200MB predictions
    + features = ~400MB
    + overhead = ~600MB
    
    Fine for 10K seeds, but what about 100K?
```

### 9.2 Adaptive Batching

```python
def score_with_adaptive_batching(scorer, seeds, lottery_history, target_vram_mb=4000):
    """
    Adaptively batch to fit in VRAM.
    
    v3.3: Starts with estimated batch size, reduces on OOM.
    """
    # Estimate batch size
    bytes_per_seed = len(lottery_history) * 4 * 3  # float32 × 3 tensors
    estimated_batch_size = int(target_vram_mb * 1024 * 1024 / bytes_per_seed)
    batch_size = min(estimated_batch_size, 10000)  # Cap at 10K
    
    all_features = []
    
    for i in range(0, len(seeds), batch_size):
        batch_seeds = seeds[i:i+batch_size]
        
        try:
            features = scorer.extract_ml_features_batch(
                seeds=batch_seeds,
                lottery_history=lottery_history
            )
            all_features.extend(features)
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # Reduce batch size and retry
                batch_size = batch_size // 2
                torch.cuda.empty_cache()
                # Retry with smaller batch
                ...
            else:
                raise
        
        # Clear cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_features
```

### 9.3 Memory Settings for RX 6600

```python
# At top of scorer_trial_worker.py

import os
os.environ["PYTORCH_HIP_ALLOC_CONF"] = \
    "garbage_collection_threshold:0.8,max_split_size_mb:128"

import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of 8GB = 6.4GB
```

---

## 10. Result Serialization

### 10.1 Result JSON Format

```json
{
    "trial_id": 42,
    "status": "success",
    "params": {
        "residue_mod_1": 14,
        "residue_mod_2": 137,
        "residue_mod_3": 1136,
        "max_offset": 3,
        "temporal_window_size": 50
    },
    "metrics": {
        "mean_train_score": 0.0847,
        "mean_holdout_score": 0.0823,
        "generalization_gap": 0.0024,
        "std_train_score": 0.0312,
        "std_holdout_score": 0.0298,
        "top_k_holdout_score": 0.1456,
        "n_survivors_scored": 10000
    },
    "runtime_seconds": 78.4,
    "gpu_id": 5,
    "hostname": "rig-6600",
    "timestamp": "2025-12-15T14:32:45"
}
```

### 10.2 save_result()

```python
def save_result(result: dict, trial_id: int):
    """
    Save trial result to local filesystem.
    
    Path: ~/distributed_prng_analysis/scorer_trial_results/trial_{trial_id}.json
    """
    results_dir = Path.home() / "distributed_prng_analysis" / "scorer_trial_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = results_dir / f"trial_{trial_id}.json"
    
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Result saved: {result_path}")
```

### 10.3 Error Handling

```python
def run_trial_safe(survivors, train_history, holdout_history, params, trial_id):
    """Run trial with error handling."""
    try:
        result = run_trial(survivors, train_history, holdout_history, params, trial_id)
        result['status'] = 'success'
        
    except Exception as e:
        result = {
            'trial_id': trial_id,
            'status': 'failed',
            'error': str(e),
            'params': params
        }
    
    return result
```

---

## 11. CLI Interface

### 11.1 Arguments

```bash
python3 scorer_trial_worker.py <survivors> <train_history> <holdout_history> <trial_id> [options]

Positional Arguments:
  survivors           Path to survivors JSON from Step 1
  train_history       Path to training lottery history (80%)
  holdout_history     Path to holdout lottery history (20%)
  trial_id            Unique trial identifier

Options:
  --params-file FILE          JSON file with trial parameters
  --residue-mod-1 INT         Small residue modulus (5-20)
  --residue-mod-2 INT         Medium residue modulus (50-150)
  --residue-mod-3 INT         Large residue modulus (500-1500)
  --max-offset INT            Temporal alignment offset (1-15)
  --temporal-window-size INT  Stability window (50-200)
  --optuna-study-name NAME    Optuna study name
  --optuna-study-db PATH      Optuna database path
  --gpu-id INT                GPU device ID (default: 0)
  -h, --help                  Show help
```

### 11.2 Usage Examples

**With params file (recommended):**
```bash
python3 scorer_trial_worker.py \
    bidirectional_survivors.json \
    train_history.json \
    holdout_history.json \
    42 \
    --params-file scorer_jobs.json \
    --optuna-study-name scorer_meta_v1 \
    --optuna-study-db sqlite:///optuna_scorer.db
```

**With CLI params:**
```bash
python3 scorer_trial_worker.py \
    bidirectional_survivors.json \
    train_history.json \
    holdout_history.json \
    42 \
    --residue-mod-1 14 \
    --residue-mod-2 137 \
    --residue-mod-3 1136 \
    --max-offset 3 \
    --temporal-window-size 50
```

---

## 12. Integration with scripts_coordinator.py

### 12.1 Architectural Change (January 2026)

**IMPORTANT:** Step 2.5 now uses `scripts_coordinator.py` instead of `coordinator.py`.

| Aspect | Old (Deprecated) | New (Current) |
|--------|------------------|---------------|
| Coordinator | `coordinator.py` | `scripts_coordinator.py` |
| Job dispatch | Parallel SSH storm | Staggered, controlled |
| Success detection | stdout parsing | File-based |
| Failure modes | Ambiguous | Explicit (MISSING, EMPTY, TIMEOUT) |

**Rationale:** Script-based jobs require autonomy-safe execution semantics.

### 12.2 Job Creation (generate_scorer_jobs.py)
```python
def generate_scorer_jobs(study_name, n_trials):
    """Generate job specifications for all trials."""
    jobs = []
    
    for trial_id in range(n_trials):
        job = {
            "job_id": f"scorer_trial_{trial_id}",
            "script": "scorer_trial_worker.py",
            "args": [
                "bidirectional_survivors_binary.npz",  # NPZ format (88x faster)
                "train_history.json",
                "holdout_history.json",
                str(trial_id),
                "--params-file", "scorer_jobs.json",
                "--optuna-study-name", study_name,
                "--optuna-study-db", f"sqlite:///optuna_{study_name}.db"
            ],
            "expected_output": f"scorer_trial_results/trial_{trial_id:04d}.json",
            "timeout": 3600
        }
        jobs.append(job)
    
    return jobs
```

### 12.3 Coordinator Dispatch
```bash
# Step 2.5 uses scripts_coordinator.py (NOT coordinator.py)
python3 scripts_coordinator.py \
    --jobs-file scorer_jobs.json \
    --output-dir scorer_trial_results \
    --preserve-paths
```

### 12.4 Result Collection

Result collection still uses `coordinator.py` as a transport utility (SCP pull only):
```python
from coordinator import MultiGPUCoordinator

coord = MultiGPUCoordinator('ml_coordinator_config.json')
all_results = coord.collect_scorer_results(total_trials)
```

### 12.5 Binary Survivor Loading (NPZ Format)

| Format | File Size | Load Time |
|--------|-----------|-----------|
| JSON | 258 MB | 4.2s |
| NPZ | 0.6 MB | 0.05s |

**Conversion:**
```bash
python3 convert_survivors_to_binary.py bidirectional_survivors.json
```

---


## 13. Troubleshooting

### 13.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "No result files found" | Worker crashed | Check stderr, look for Python errors |
| "OOM" | Batch too large | Reduce sample size or batch size |
| "Holdout score = 0" | Wrong holdout file | Verify holdout_history.json exists |
| "Trial timeout" | Slow scoring | Increase timeout, reduce sample size |
| "Permission denied" | SCP auth | Check SSH keys on coordinator |

### 13.2 Debug Mode

```bash
# Run locally with verbose output
python3 scorer_trial_worker.py \
    test_survivors.json \
    train_history.json \
    holdout_history.json \
    0 \
    --residue-mod-1 10 \
    --residue-mod-2 100 \
    --residue-mod-3 1000 \
    --max-offset 5 \
    --temporal-window-size 100 \
    2>&1 | tee worker_debug.log
```

### 13.3 Verify Result Files

```bash
# Check local results
ls -la ~/distributed_prng_analysis/scorer_trial_results/

# Check remote results
ssh rig-6600 'ls -la ~/distributed_prng_analysis/scorer_trial_results/'

# View a result
cat ~/distributed_prng_analysis/scorer_trial_results/trial_0.json | jq .
```

---

## 14. Complete Method Reference

### 14.1 Main Functions

| Function | Purpose |
|----------|---------|
| `main()` | Entry point, parse args, run trial |
| `parse_arguments()` | Parse CLI arguments |
| `load_params_from_file(path, trial_id)` | Load params from JSON file |
| `run_trial(...)` | Execute single trial |
| `save_result(result, trial_id)` | Write result JSON |

### 14.2 Scoring Functions

| Function | Purpose |
|----------|---------|
| `score_survivors_gpu(scorer, seeds, history)` | GPU-vectorized batch scoring |
| `score_with_adaptive_batching(...)` | OOM-safe batched scoring |
| `compute_metrics(train_scores, holdout_scores)` | Compute trial metrics |

### 14.3 Utility Functions

| Function | Purpose |
|----------|---------|
| `sample_survivors(survivors, n)` | Random sample for scoring |
| `load_json(path)` | Load JSON file |
| `setup_gpu(gpu_id)` | Initialize GPU device |

---

## 15. Chapter Summary

**Chapter 13: Scorer Trial Worker** covers single trial execution:

| Component | Lines | Purpose |
|-----------|-------|---------|
| ROCm setup | ~20 | Environment before imports |
| Argument parsing | ~50 | CLI and params-file support |
| Trial execution | ~100 | Score + evaluate + metrics |
| GPU batching | ~80 | Vectorized scoring |
| Result handling | ~30 | JSON serialization |

**Key Points:**
- Part of pull architecture (write locally, coordinator pulls)
- v3.4 critical fix: Holdout uses SAME sampled seeds
- v3.3: GPU-vectorized scoring (3.8x faster)
- v3.2: --params-file for shorter SSH commands
- Adaptive batching for 8GB VRAM

---

## Next Chapter

**Chapter 14: Feature Importance** will cover:
- `feature_importance.py` — Permutation, gradient, SHAP methods
- `feature_visualizer.py` — 13 chart types
- AI-powered interpretation

---

*End of Chapter 13: Scorer Trial Worker*
