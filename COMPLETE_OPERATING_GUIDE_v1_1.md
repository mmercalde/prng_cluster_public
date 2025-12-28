# COMPLETE OPERATING GUIDE
## Distributed PRNG Analysis System
**Version 1.1.0**  
**December 2025**  
**Updated: Session 17 (Dec 27, 2025)**

### 26-GPU Cluster Architecture
Zeus (2× RTX 3080 Ti) + rig-6600 (12× RX 6600) + rig-6600b (12× RX 6600)  
~285 TFLOPS Combined Computing Power

---

## Table of Contents

1. System Overview
2. Pipeline Steps
3. Core Modules
4. Distributed Workers
5. Configuration System
6. Operational Procedures
7. Monitoring & Debugging
8. Appendix A: Data Models
9. Appendix B: File Inventory

---

# PART 1: SYSTEM OVERVIEW

## 1.1 Architecture Overview

The Distributed PRNG Analysis System is a sophisticated GPU-accelerated platform for analyzing pseudo-random number generator patterns in lottery data. The system combines statistical sieving, machine learning, and Bayesian optimization across a 26-GPU cluster.

### Core Architecture Components

| Component | File | Purpose |
|-----------|------|---------|
| Coordinator | coordinator.py | Central job distribution and fault tolerance manager |
| Distributed Workers | distributed_worker.py | GPU-bound execution agents on each node |
| GPU Sieve | sieve_filter.py | CuPy-based residue sieve for seed filtering |
| PRNG Registry | prng_registry.py | 46 PRNG algorithm implementations (CuPy kernels + PyTorch GPU) |
| Survivor Scorer | survivor_scorer.py | 50-feature ML scoring engine |
| Global State Tracker | global_state_tracker.py | 14 global lottery features (NEW Session 14) |
| Reinforcement Engine | reinforcement_engine.py | Neural network for survivor quality prediction |
| Multi-Model Comparison | meta_prediction_optimizer_anti_overfit.py | 4 ML models with subprocess isolation |

## 1.2 Hardware Configuration

| Node | GPU Type | Count | Backend |
|------|----------|-------|---------|
| Zeus (localhost) | RTX 3080 Ti (12GB) | 2 | CUDA / PyTorch |
| rig-6600 | RX 6600 (8GB) | 12 | ROCm / HIP |
| rig-6600b | RX 6600 (8GB) | 12 | ROCm / HIP |

**Network:** All nodes connected via SSH with key-based authentication.  
**IP Addresses:** rig-6600 (192.168.3.120), rig-6600b (192.168.3.154)

## 1.3 Software Dependencies

- Python 3.10+ with PyTorch, CuPy, NumPy, SciPy, Optuna, scikit-learn
- XGBoost, LightGBM, CatBoost for multi-model comparison (NEW Session 9)
- CUDA 12.x on Zeus for NVIDIA GPUs
- ROCm 5.7+ on mining rigs for AMD GPUs
- Paramiko for SSH connection pooling

## 1.4 Data Flow Overview

The system processes lottery data through a multi-stage pipeline:
```
Raw draws → GPU sieve filtering → Survivor scoring → ML training → Prediction generation
```

Each stage can run distributed across all 26 GPUs with automatic fault tolerance and job recovery.

**Note:** Step 0 (PRNG Fingerprinting) was investigated in Session 17 and **ARCHIVED** - mathematically impossible under mod1000 projection.

---

# PART 2: PIPELINE STEPS

The complete prediction pipeline consists of 6 major steps, each building on the previous results.

## 2.0 Step 0: PRNG Fingerprinting — ARCHIVED

**Status:** ARCHIVED (Session 17)

**Original Purpose:** Classify unknown PRNGs by comparing behavioral fingerprints against a library of known PRNGs.

**Investigation Results:**
- Tested 52 mod1000-specific features, 30 curated features, 64 original features
- SNR (Signal-to-Noise Ratio) < 0.15 for ALL features
- Within-PRNG variance 4-7× larger than between-PRNG variance

**Verdict:** Under mod1000 projection, all PRNGs become statistically indistinguishable. Fingerprinting is mathematically impossible.

**Alternative:** Trust the bidirectional sieve - wrong PRNG produces 0 survivors, right PRNG produces survivors.

## 2.1 Step 1: Window Optimizer

**Purpose:** Find optimal sieve parameters (window_size, offset, skip_range, thresholds) using Bayesian optimization with Optuna's Tree-Parzen Estimator algorithm.

**Primary Script:** `window_optimizer.py`

### Key Classes & Functions

| Class/Function | Purpose |
|----------------|---------|
| WindowConfig | Dataclass holding window_size, offset, skip_min, skip_max, thresholds |
| SearchBounds | Defines parameter ranges from distributed_config.json |
| BayesianOptimization | Search strategy using Optuna TPE sampler |
| run_bayesian_optimization() | Main entry point for distributed optimization |

### Configuration

Search bounds are loaded from `distributed_config.json` → `search_bounds` section.

**Key parameters:** window_size (2-500), offset (0-100), skip_min (0-10), skip_max (10-500), forward_threshold (0.001-0.1), reverse_threshold (0.001-0.1)

### Output

`optimal_window_config.json`

### Example Usage
```bash
python3 window_optimizer.py --strategy bayesian --lottery-file daily3.json --trials 50
python3 window_optimizer.py --strategy bayesian --lottery-file daily3.json --trials 50 --test-both-modes
```

## 2.2 Step 2.5: Scorer Meta-Optimizer

**Purpose:** Optimize the survivor_scorer.py hyperparameters (residue_mods, max_offset, temporal windows) using distributed Bayesian optimization across all 26 GPUs.

**Primary Scripts:**
- `run_scorer_meta_optimizer.py`: Orchestrator that batches trials across GPUs
- `generate_scorer_jobs.py`: Creates job JSON files for each trial
- `scorer_trial_worker.py`: Executes single trial on remote GPU

### Architecture: Pull Model

Workers do NOT access the Optuna database directly. Instead, they write JSON results to local filesystem, and the coordinator (Zeus) pulls results via SCP. This prevents database contention with 26 concurrent workers.

### Search Space

| Parameter | Range | Purpose |
|-----------|-------|---------|
| residue_mod_1 | 5-20 | Small residue modulus |
| residue_mod_2 | 50-150 | Medium residue modulus |
| residue_mod_3 | 500-1500 | Large residue modulus |
| max_offset | 1-15 | Temporal alignment offset |
| temporal_window_size | [50,100,150,200] | Stability analysis window |

### Output

`optimal_scorer_config.json`

## 2.3 Step 3: Full Distributed Scoring

**Purpose:** Apply the optimal scorer configuration to ALL survivors from Step 1, extracting 64 ML features per seed (50 per-seed + 14 global) for downstream model training.

**Primary Scripts:**
- `generate_full_scoring_jobs.py`: Creates chunked jobs (1000 survivors/job)
- `run_step3_full_scoring.sh`: Orchestrates distributed execution and aggregation
- `survivor_scorer.py`: Core scoring engine (50 per-seed features)
- `global_state_tracker.py`: Global lottery features (14 features)

### Feature Architecture (Updated Session 17)
```
Total Features: 64 (62 for training after excluding score, confidence)
├── Per-seed features: 50 (from survivor_scorer.py)
│   ├── Residue features: 12
│   ├── Temporal features: 20
│   ├── Statistical features: 12
│   ├── Metadata features: 4 (skip_min, skip_max, bidirectional_count, bidirectional_selectivity)
│   └── Score metrics: 2 (excluded from training)
│
└── Global features: 14 (from GlobalStateTracker, prefixed with 'global_')
    ├── Residue entropy: 3 (global_residue_8/125/1000_entropy)
    ├── Bias detection: 3 (global_power_of_two_bias, global_frequency_bias_ratio, global_suspicious_gap_percentage)
    ├── Regime detection: 3 (global_regime_change_detected, global_regime_age, global_reseed_probability)
    ├── Marker analysis: 4 (global_marker_390/804/575_variance, global_high_variance_count)
    └── Stability: 1 (global_temporal_stability)
```

### Global Features Integration (NEW Session 17)

Global features are added at Step 3 Phase 5 (Aggregation):
- Computed once from lottery history (O(1) not O(N))
- Identical for all survivors
- Prefixed with `global_` to prevent namespace collision
- Now available to ALL model types

### Parallel Execution (NEW Session 16)

Step 3 now uses ThreadPoolExecutor for GPU-aware parallelism within each node:
```
🔀 PARALLEL: localhost | 2 GPU workers | 4 jobs | distribution: {0: 2, 1: 2}
🔀 PARALLEL: 192.168.3.120 | 12 GPU workers | 7 jobs | distribution: {0: 1, 1: 1, ...}
```

### Output

`survivors_with_scores.json`

## 2.4 Step 4: ML Meta-Optimizer

**Purpose:** Optimize the neural network architecture and training hyperparameters for the reinforcement_engine.py model using Bayesian optimization.

**Primary Script:** `adaptive_meta_optimizer.py`

### Key Classes

| Class | Purpose |
|-------|---------|
| PredictionMetrics | Dataclass with variance, MAE, discrimination_power, calibration_error |
| MetaPredictionOptimizer | Main optimizer class with Optuna integration |

### Composite Score Formula
```
score = variance * 10.0 + discrimination_power * 5.0 + (1/(MAE+0.01)) * 2.0 + (1/(calibration_error+0.01)) * 1.0
```

### Output

`reinforcement_engine_config_optimized.json`

## 2.5 Step 5: Anti-Overfit Training

**Purpose:** Train the final prediction model with K-fold cross-validation and anti-overfitting measures to ensure generalization.

**Primary Scripts:**
- `meta_prediction_optimizer_anti_overfit.py`: Orchestrator with overfit detection
- `subprocess_trial_coordinator.py`: Subprocess isolation for multi-model comparison
- `train_single_trial.py`: Single trial worker

### Multi-Model Architecture (Session 9+)

Step 5 now supports 4 ML model types with subprocess isolation:

| Model | Backend | Speed | Session 17 Results |
|-------|---------|-------|-------------------|
| neural_net | PyTorch (CUDA/ROCm) | Slow (253s+) | R²=0.0000 |
| xgboost | XGBoost (CUDA) | Fast (1.8s) | R²=1.0000 |
| lightgbm | LightGBM (OpenCL) | Fast (2.9s) | R²=0.9999 |
| **catboost** | CatBoost (CUDA) | Fast (4.8s) | **R²=1.0000 🏆** |

### Subprocess Isolation

Each trial runs in a separate subprocess to prevent OpenCL/CUDA conflicts:
```
Main Process (subprocess_trial_coordinator.py)
    ├── Trial 0: subprocess → LightGBM (OpenCL) → exits
    ├── Trial 1: subprocess → PyTorch (CUDA) → exits
    ├── Trial 2: subprocess → XGBoost (CUDA) → exits
    └── Trial 3: subprocess → CatBoost (CUDA) → exits
```

### Timeout CLI (NEW Session 17)
```bash
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data train_history.json \
    --compare-models \
    --trials 8 \
    --timeout 900  # 15 minutes per trial (default: 600s)
```

### Anti-Overfit Metrics

- `overfit_ratio`: train_score / validation_score (target: close to 1.0)
- `is_overfitting()`: Returns True if overfit_ratio > threshold

### Output

- `models/reinforcement/best_model.{pth|json|cbm|txt}` (model file)
- `models/reinforcement/best_model.meta.json` (sidecar with metadata)

## 2.6 Step 6: Prediction Generation

**Purpose:** Generate final predictions using the trained model, producing ranked survivor seeds with confidence scores.

**Primary Script:** `prediction_generator.py`

### Sidecar Pattern

Step 6 auto-detects model type from sidecar:
```
Step 5 Output:
├── best_model.cbm (CatBoost model)
└── best_model.meta.json
    ├── model_type: "catboost"
    ├── feature_schema: {...}
    └── agent_metadata:
        └── run_id: "step5_20251226_235017"

Step 6:
├── Reads sidecar → auto-detects model type
├── Extracts parent_run_id → links to training run
└── Generates predictions with lineage
```

### Output Contract (v2.2)
```json
{
    "predictions": [521, 626, 415],
    "raw_scores": [0.127792, 0.108792, 0.057818],
    "confidence_scores": [0.7949, 0.6884, 0.3286],
    "confidence_scores_normalized": [1.0, 0.8513, 0.4524],
    "metadata": {
        "method": "dual_sieve",
        "score_stats": {...}
    },
    "agent_metadata": {
        "pipeline_step": 6,
        "parent_run_id": "step5_20251226_235017"
    }
}
```

---

# PART 3: CORE MODULES

## 3.1 survivor_scorer.py — Scoring Engine

**Purpose:** Computes similarity scores between PRNG-generated sequences and actual lottery draws. Extracts 50 per-seed ML features for downstream model training.

### Key Class: SurvivorScorer

| Method | Purpose |
|--------|---------|
| `__init__(prng_type, mod, residue_mods, config_dict)` | Initialize scorer with PRNG config |
| `extract_ml_features(seed, lottery_history, skip)` | Extract 50 ML features for single seed |
| `batch_score_vectorized(seeds, lottery_history)` | GPU-accelerated batch scoring |

### ROCm Compatibility

The scorer includes critical VRAM management for RX 6600 rigs:
- `PYTORCH_HIP_ALLOC_CONF` for memory pooling
- 80% VRAM limit via `torch.cuda.set_per_process_memory_fraction(0.8)`
- Explicit two-step NumPy→GPU tensor transfer for ROCm stability

## 3.2 global_state_tracker.py — Global Features (NEW Session 14)

**Purpose:** Computes 14 lottery-wide features from historical data. These features capture global patterns like regime changes, marker number anomalies, and distribution biases.

### Key Class: GlobalStateTracker

| Method | Purpose |
|--------|---------|
| `__init__(lottery_history, config)` | Initialize with lottery data |
| `get_global_state()` | Returns dict of 14 global features |
| `compute_marker_variance(marker)` | Variance analysis for suspicious numbers |
| `detect_regime_change()` | Detect PRNG state changes |

### Global Features (14 total, prefixed with `global_`)

| Category | Features |
|----------|----------|
| Residue Entropy | global_residue_8_entropy, global_residue_125_entropy, global_residue_1000_entropy |
| Bias Detection | global_power_of_two_bias, global_frequency_bias_ratio, global_suspicious_gap_percentage |
| Regime Detection | global_regime_change_detected, global_regime_age, global_reseed_probability |
| Marker Analysis | global_marker_390_variance, global_marker_804_variance, global_marker_575_variance, global_high_variance_count |
| Stability | global_temporal_stability |

## 3.3 reinforcement_engine.py — ML Model

**Purpose:** Neural network for survivor quality prediction. Uses extracted features to predict which seeds are most likely to generate accurate lottery predictions.

### Network Architecture: SurvivorQualityNet
```
Input (62 features) → Dense(128) → Dense(64) → Dense(32) → Output(1)
```

All layers use ReLU activation except output (linear for regression).

### Distributed Training Support

Supports PyTorch DistributedDataParallel (DDP) for multi-GPU training via `--distributed` flag. Compatible with both CUDA (Zeus) and ROCm (mining rigs).

## 3.4 Multi-Model Wrappers (NEW Session 9)

**Purpose:** Unified interface for 4 ML model types, enabling fair comparison and subprocess isolation.

### Model Wrappers

| File | Model | Backend |
|------|-------|---------|
| `neural_net_wrapper.py` | PyTorch NN | CUDA/ROCm |
| `xgboost_wrapper.py` | XGBoost | CUDA |
| `lightgbm_wrapper.py` | LightGBM | OpenCL |
| `catboost_wrapper.py` | CatBoost | CUDA |

### Model Factory
```python
from models.model_factory import create_model, load_model_from_sidecar

# Create new model
model = create_model('catboost', n_features=62)

# Load from sidecar (auto-detect type)
model = load_model_from_sidecar('models/reinforcement')
```

## 3.5 prng_registry.py — PRNG Algorithms

**Purpose:** Central registry of 46 PRNG algorithm implementations. Provides both CuPy GPU kernels for high-throughput sieving and PyTorch GPU functions for ML scoring.

### Supported PRNGs (Partial List)

| Family | PRNGs |
|--------|-------|
| LCG | java_lcg, java_lcg_hybrid, minstd, lcg32, lcg32_hybrid |
| XorShift | xorshift32, xorshift64, xorshift128, xorshift32_hybrid |
| PCG | pcg32, pcg32_hybrid |
| Mersenne Twister | mt19937, mt19937_simple (624-word state) |
| Modern | xoshiro256++, philox4x32, sfc64 |

### API Functions
```python
get_kernel_info(prng_family)     # Returns kernel code and metadata
get_cpu_reference(prng_family)   # Returns CPU reference implementation
get_pytorch_gpu_reference(prng)  # Returns PyTorch GPU function
has_pytorch_gpu(prng_family)     # Check if PyTorch version exists
list_available_prngs()           # List all registered PRNGs
```

## 3.6 sieve_filter.py — GPU Sieve

**Purpose:** GPU-accelerated residue sieve for filtering candidate seeds. Uses CuPy kernels to test millions of seeds per second against lottery patterns.

### Key Class: GPUSieve

| Method | Purpose |
|--------|---------|
| `__init__(gpu_id)` | Bind to specific GPU |
| `run_sieve(seeds, draws, prng_family, ...)` | Standard constant-skip sieve |
| `run_hybrid_sieve(seeds, draws, ...)` | Variable skip pattern search |

### Window Size Limits

v2.3+ supports window sizes up to 2048 draws (previously 512). All hardcoded buffer sizes were replaced with dynamic sizing to prevent GPU crashes with large windows.

## 3.7 coordinator.py — Distributed Job Management

**Purpose:** Central coordinator managing job distribution, SSH connection pooling, fault tolerance, and result aggregation across the 26-GPU cluster.

### Key Class: MultiGPUCoordinator

Version 1.8.2 with per-node concurrency limits for script jobs to prevent CPU overload on mining rig CPUs.

### Key Methods

| Method | Purpose |
|--------|---------|
| `execute_truly_parallel_dynamic()` | Main distributed execution with work queue pattern |
| `execute_local_job()` | Execute job on Zeus GPUs |
| `execute_remote_job()` | Execute job via SSH on mining rigs |
| `collect_scorer_results()` | Pull architecture result aggregation |

### Fault Tolerance

- Automatic job recovery with progress persistence
- Failed jobs are retried on alternate workers
- Per-node concurrency limits prevent SSH connection overload (max 26 concurrent connections via work queue pattern)

---

# PART 4: DISTRIBUTED WORKERS

## 4.1 distributed_worker.py

**Purpose:** GPU-bound execution agent that runs on each node. Handles job deserialization, GPU binding, environment setup, and result serialization.

### Key Features

- **ROCm Environment Setup:** Sets `HSA_OVERRIDE_GFX_VERSION`, `HSA_ENABLE_SDMA` before GPU imports
- **Lazy CuPy Import:** `_ensure_cupy()` for environment-safe GPU initialization
- **Script Job Support:** v1.8.0 restored GPU init skip for script jobs (they handle own setup)

## 4.2 subprocess_trial_coordinator.py (NEW Session 11)

**Purpose:** Coordinates multi-model trials using subprocess isolation to prevent CUDA/OpenCL conflicts.

### Key Features

- Each trial runs in fresh subprocess
- No GPU imports in coordinator process
- Model checkpoints saved per trial
- Configurable timeout per trial (NEW Session 17)

## 4.3 train_single_trial.py (NEW Session 11)

**Purpose:** Executes single training trial in isolated subprocess. Handles GPU initialization, model training, and result serialization.

### Usage
```bash
python3 train_single_trial.py \
    --data-path /tmp/trial_data.npz \
    --trial-id 0 \
    --model-type catboost \
    --output-dir /tmp/trial_models \
    --params '{"cb_n_estimators": 500, ...}'
```

## 4.4 GPU Binding & Memory Management

### CUDA (Zeus)
```bash
CUDA_VISIBLE_DEVICES=0 python3 distributed_worker.py --gpu-id 0
```

Workers use logical device ID 0 after CUDA_VISIBLE_DEVICES isolation.

### ROCm (Mining Rigs)
```bash
HIP_VISIBLE_DEVICES=5 python3 distributed_worker.py --gpu-id 5
```

ROCm workers set both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` for PyTorch compatibility.

**Critical environment variables:**
- `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- `HSA_ENABLE_SDMA=0`

### VRAM Management for RX 6600

8GB VRAM requires careful management:
- `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128`
- `torch.cuda.set_per_process_memory_fraction(0.8)`

---

# PART 5: CONFIGURATION SYSTEM

## 5.1 distributed_config.json

**Purpose:** Central configuration file defining cluster topology, sieve defaults, and search bounds. Single source of truth for all scripts.

### Structure
```json
{
  "nodes": [{
    "hostname": "localhost",
    "username": "michael",
    "gpu_count": 2,
    "gpu_type": "RTX 3080 Ti",
    "script_path": "/home/michael/distributed_prng_analysis",
    "python_env": "/home/michael/torch/bin/python"
  }, ...],
  "sieve_defaults": {...},
  "search_bounds": {...}
}
```

## 5.2 config_manifests/feature_registry.json (NEW Session 17)

**Purpose:** Documents all 64 features (50 per-seed + 14 global) with metadata.

### Structure
```json
{
  "version": "1.0.0",
  "per_seed_features": {...},
  "global_features": {
    "global_residue_8_entropy": {...},
    "global_regime_change_detected": {...},
    ...
  },
  "feature_combination": {
    "per_seed_count": 50,
    "global_count": 14,
    "total": 64,
    "training_features": 62
  }
}
```

---

# PART 6: OPERATIONAL PROCEDURES

## 6.1 Starting the Cluster

### Prerequisites

- SSH keys configured for passwordless access to rig-6600 and rig-6600b
- ROCm drivers installed and working on mining rigs
- Python environments activated (torch on Zeus, rocm_env on rigs)
- Shared filesystem or synchronized code across nodes

### Connectivity Test
```bash
python3 coordinator.py --test-connectivity
```

## 6.2 Running a Full Pipeline

### Step-by-Step
```bash
# Step 1: Window Optimization
python3 window_optimizer.py --strategy bayesian --lottery-file daily3.json --trials 100

# Step 2.5: Scorer Meta-Optimization
python3 run_scorer_meta_optimizer.py --batches 10

# Step 3: Full Distributed Scoring
python3 generate_full_scoring_jobs.py --survivors bidirectional_survivors.json
bash run_step3_full_scoring.sh

# Step 4: ML Meta-Optimization
python3 adaptive_meta_optimizer.py --mode full --lottery-data train_history.json --survivor-data survivors_with_scores.json

# Step 5: Anti-Overfit Training (with multi-model comparison)
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data train_history.json \
    --compare-models \
    --trials 8 \
    --timeout 900

# Step 6: Generate Predictions
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward survivors_with_scores.json \
    --lottery-history train_history.json
```

## 6.3 Data Quality Check (NEW Session 17)

The lottery scraper may create duplicates. Check and clean:
```bash
python3 -c "
import json
with open('daily3.json') as f:
    data = json.load(f)
seen = set()
clean = []
for entry in data:
    key = (entry['date'], tuple(entry['numbers']))
    if key not in seen:
        seen.add(key)
        clean.append(entry)
print(f'Original: {len(data)}, Clean: {len(clean)}, Duplicates: {len(data)-len(clean)}')
with open('daily3_clean.json', 'w') as f:
    json.dump(clean, f)
"
```

---

# PART 7: MONITORING & DEBUGGING

## 7.1 GPU Monitoring

### NVIDIA (Zeus)
```bash
nvidia-smi --loop=1  # Continuous monitoring
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
```

### AMD (Mining Rigs)
```bash
rocm-smi --showuse  # GPU utilization
rocm-smi --showmeminfo vram  # VRAM usage
watch -n1 rocm-smi  # Continuous monitoring
```

## 7.2 Common Errors & Solutions

| Error | Solution |
|-------|----------|
| CUDA OOM on RTX 3080 Ti | Reduce batch size, increase empty_cache() calls |
| ROCm OOM on RX 6600 | Check PYTORCH_HIP_ALLOC_CONF, verify 80% VRAM limit |
| SSH connection refused | Check SSH keys, verify sshd running on target |
| NoneType is not iterable | Check payload checks: 'if job.payload and ...' |
| cuBLAS no CUDA context | Call initialize_cuda() before any CUDA operations |
| HSA memory allocation failed | Set HSA_OVERRIDE_GFX_VERSION=10.3.0 for RX 6600 |
| Neural net timeout | Increase --timeout (default 600s) |
| OpenCL/CUDA conflict | Use --compare-models for subprocess isolation |

---

# APPENDIX A: DATA MODELS

## models.py — Core Data Structures

### WorkerNode
```python
@dataclass
class WorkerNode:
    hostname: str
    username: str
    gpu_count: int
    gpu_type: str
    python_env: str
    script_path: str
```

### JobSpec
```python
@dataclass
class JobSpec:
    job_id: str
    prng_type: str
    mapping_type: str
    seeds: List[int]
    samples: int
    payload: Optional[Dict[str, Any]] = None
    analysis_type: str = 'statistical'
    attempt: int = 0
```

---

# APPENDIX B: FILE INVENTORY

## Core Pipeline Scripts

| File | Purpose |
|------|---------|
| window_optimizer.py | Step 1: Bayesian window optimization |
| run_scorer_meta_optimizer.py | Step 2.5: Scorer hyperparameter tuning |
| generate_full_scoring_jobs.py | Step 3: Job generation for full scoring |
| run_step3_full_scoring.sh | Step 3: Orchestration with global features |
| adaptive_meta_optimizer.py | Step 4: ML architecture optimization |
| meta_prediction_optimizer_anti_overfit.py | Step 5: Anti-overfit training |
| prediction_generator.py | Step 6: Final prediction generation |

## Core Modules

| File | Purpose |
|------|---------|
| coordinator.py | Distributed job coordination (v1.8.2) |
| distributed_worker.py | GPU-bound execution agent (v1.8.0) |
| subprocess_trial_coordinator.py | Multi-model subprocess isolation |
| train_single_trial.py | Single trial worker |
| sieve_filter.py | GPU residue sieve engine (v2.3.1) |
| prng_registry.py | 46 PRNG implementations (v2.4) |
| survivor_scorer.py | 50-feature per-seed scoring engine |
| global_state_tracker.py | 14-feature global scoring engine |
| reinforcement_engine.py | Neural network model (v1.4.0) |
| models/model_factory.py | Multi-model factory |
| models/wrappers/*.py | XGBoost, LightGBM, CatBoost, Neural Net wrappers |

## Configuration Files

| File | Purpose |
|------|---------|
| distributed_config.json | Cluster topology and search bounds |
| config_manifests/feature_registry.json | Feature documentation |
| optimal_window_config.json | Output from Step 1 |
| optimal_scorer_config.json | Output from Step 2.5 |

---

**— End of Document —**
