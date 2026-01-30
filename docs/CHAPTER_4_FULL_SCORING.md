# Chapter 4: Full Scoring & Feature Extraction (Step 3)

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 2.0.0 (Holdout Integration)  
**File:** `survivor_scorer.py`, `full_scoring_worker.py`  
**Lines:** ~550  
**Purpose:** GPU-accelerated ML feature extraction for survivor seeds

---

## Table of Contents

1. [Overview](#1-overview)
2. [Environment Setup](#2-environment-setup)
3. [Constants and Configuration](#3-constants-and-configuration)
4. [SurvivorScorer Class](#4-survivorscorer-class)
5. [Feature Extraction](#5-feature-extraction)
6. [Dual Sieve Intersection](#6-dual-sieve-intersection)
7. [Batch Scoring](#7-batch-scoring)
8. [GPU-Batched Feature Extraction](#8-gpu-batched-feature-extraction)
9. [Complete Feature Reference](#9-complete-feature-reference)
10. [Integration Points](#10-integration-points)
11. [Method Reference](#11-method-reference)

---

## 1. Overview

### 1.1 What the Survivor Scorer Does

The Survivor Scorer extracts **47 ML features** from survivor seeds for quality prediction:

- **PRNG Sequence Generation:** CPU reference or PyTorch GPU
- **Match Analysis:** Exact matches, residue matching, lane agreement
- **Temporal Stability:** Window-based consistency tracking
- **Statistical Features:** Mean, std, entropy, KL divergence
- **Bidirectional Features:** Forward/reverse intersection analysis

### 1.2 Key Fixes in Final Version

```
BUG FIX 1: Added residue_mod_1/2/3 translation for Optuna
BUG FIX 2: Temporal stability optimization (reuse seq, no duplicate generation)
BUG FIX 3: Team Beta's targeted VRAM limit for RX 6600 rigs only
BUG FIX 4: Consolidated launch contention fix (PYTORCH_HIP_ALLOC_CONF)
BUG FIX 5: Explicit two-step NumPy to GPU tensor transfer (ROCm stability)
```

### 1.3 Key Features

| Feature | Description |
|---------|-------------|
| **47 ML Features** | Comprehensive seed quality metrics + holdout_hits |
| **GPU Acceleration** | PyTorch tensors for parallel computation |
| **ROCm Support** | AMD RX 6600 compatibility |
| **Batch Processing** | Vectorized scoring for entire pools |
| **Memory Management** | VRAM limits for 8GB GPUs |
| **Holdout Integration** | Computes y-label for Step 5 ML training |

### 1.4 HOLDOUT_HITS Integration (v2.0)

The scoring pipeline now computes `holdout_hits` — the ML y-label for Step 5:

```
┌─────────────────────────────────────────────────────────────┐
│  TRAIN HISTORY (used for sieving & feature extraction)      │
│  → Features extracted from THESE draws                      │
│  → Using features as y-label = CIRCULAR / OVERFIT           │
├─────────────────────────────────────────────────────────────┤
│  HOLDOUT HISTORY (never seen during sieving)                │
│  → Measures TRUE predictive power                           │
│  → Using this as y-label = HONEST EVALUATION                │
└─────────────────────────────────────────────────────────────┘
```

**Key Implementation Details:**

| Component | Value | Source |
|-----------|-------|--------|
| `offset` | `len(train_history)` | DERIVED, not configurable |
| `holdout_hits` | 0.0 - 1.0 | Hits / total holdout draws |

**Critical Rule (per Team Beta):**
```python
# OFFSET IS A LAW, NOT A PARAMETER
offset = len(train_history)  # THE ONLY SOURCE
```

This ensures temporal causality: holdout data is the CONTINUATION of training data in the PRNG sequence.

---

---

## 1.5 Configuration Sources (CRITICAL)

**Added: 2026-01-25** — Understand where parameters come from.

### Parameter Source by Run Method

| Run Method | Config Source | chunk_size Location |
|------------|---------------|---------------------|
| Manual: `bash run_step3_full_scoring.sh` | Script default | Line 70: `CHUNK_SIZE=1000` |
| Manual with override: `bash run_step3_full_scoring.sh --chunk-size 500` | CLI argument | Passed directly |
| WATCHER: `--start-step 3 --end-step 3` | Manifest | `agent_manifests/full_scoring.json` |
| WATCHER with override: `--params '{"chunk_size": 500}'` | CLI params | Overrides manifest |

### Key Insight

**WATCHER ignores script defaults.** It reads `default_params` from the manifest and passes them explicitly to the script.

### To Change chunk_size Permanently

Update **BOTH** files:

```bash
# 1. Update script default (for manual runs)
sed -i 's/^CHUNK_SIZE=.*/CHUNK_SIZE=1000/' run_step3_full_scoring.sh

# 2. Update manifest default (for WATCHER runs)
sed -i 's/"chunk_size": [0-9]*/"chunk_size": 1000/' agent_manifests/full_scoring.json

# 3. Verify
grep "CHUNK_SIZE=" run_step3_full_scoring.sh
grep '"chunk_size"' agent_manifests/full_scoring.json
```

### OOM Prevention Reminder

Use `chunk_size=1000` (not 5000) to prevent OOM on 7.7GB mining rigs:
- 1000 seeds/chunk × ~500MB = safe for 12 concurrent workers
- 5000 seeds/chunk × ~1.5GB = OOM with 7+ concurrent workers


## 2. Environment Setup

### 2.1 ROCm Configuration (CRITICAL)

**MUST BE BEFORE ANY IMPORTS:**

```python
import sys, os, socket
HOST = socket.gethostname()

# AMD ROCm fixes + CRITICAL VRAM LAUNCH FIX
if HOST in ["rig-6600", "rig-6600b"]:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
    # Forces allocator to use small chunks (128MB) and aggressively free memory
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", 
                          "garbage_collection_threshold:0.8,max_split_size_mb:128")

os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")
os.environ.setdefault('CUPY_CUDA_MEMORY_POOL_TYPE', 'none')
```

### 2.2 VRAM Limiting for RX 6600

```python
# Team Beta's targeted VRAM limit for RX 6600 rigs only
if HOST in ["rig-6600", "rig-6600b"] and TORCH_AVAILABLE:
    if torch.cuda.is_available():
        # Limit PyTorch to 80% (6.4GB of 8GB) VRAM
        torch.cuda.set_per_process_memory_fraction(0.8)
        # Disable benchmark mode to reduce memory fragmentation
        torch.backends.cudnn.benchmark = False
```

**Why This Matters:**
- RX 6600 has 8GB VRAM
- 12-worker launch can cause instantaneous VRAM spike
- Linux OOM Killer intervention prevented by limiting to 6.4GB

### 2.3 Safe Entropy Function

```python
from scipy.stats import entropy as _entropy

# CRITICAL: Safe entropy — fixes CuPy → NumPy implicit conversion
def entropy(p, q=None, *args, **kwargs):
    p = p.get() if hasattr(p, 'get') else p  # CuPy → NumPy
    if q is not None:
        q = q.get() if hasattr(q, 'get') else q
    return _entropy(p, q, *args, **kwargs)
```

---

---

## 2.5 Ramdisk Prerequisites (CRITICAL)

**Added: 2026-01-22**

Step 3 jobs expect training data at `/dev/shm/prng/step3/` on **ALL nodes**.

### Required Files

| File | Purpose |
|------|---------|
| `train_history.json` | Training draws for scoring |
| `holdout_history.json` | Holdout draws for validation |

### Why Ramdisk?

Jobs reference paths like `/dev/shm/prng/step3/train_history.json` to avoid:
- Disk I/O contention with 12 concurrent GPU jobs
- NFS/network latency on distributed nodes
- JSON parsing overhead on repeated loads

### Verification (Before Running Step 3)

```bash
# Check all nodes
ssh 192.168.3.120 "ls -la /dev/shm/prng/step3/"
ssh 192.168.3.154 "ls -la /dev/shm/prng/step3/"
ls -la /dev/shm/prng/step3/  # Zeus
```

**Expected:** Both files present on all three nodes.

### Manual Population (If Missing)

```bash
# On Zeus
mkdir -p /dev/shm/prng/step3
cp train_history.json holdout_history.json /dev/shm/prng/step3/

# On rig-6600
ssh 192.168.3.120 "mkdir -p /dev/shm/prng/step3"
scp train_history.json holdout_history.json 192.168.3.120:/dev/shm/prng/step3/

# On rig-6600b
ssh 192.168.3.154 "mkdir -p /dev/shm/prng/step3"
scp train_history.json holdout_history.json 192.168.3.154:/dev/shm/prng/step3/
```

### Automatic Preload

The WATCHER agent runs ramdisk preload but in **"Standalone mode"** — it only populates the local node. For distributed execution, you must manually populate remote nodes or run the preload script on each node.

### Common Failure Pattern

If Step 3 jobs fail instantly (~3 seconds) with no output file:
1. Check ramdisk exists on failing node
2. Worker expects `--train-history /dev/shm/prng/step3/train_history.json`
3. Missing file = immediate argument parsing failure


## 3. Constants and Configuration

### 3.1 Default Constants

```python
DEFAULT_MOD = 1000              # PRNG output modulo
DEFAULT_RESIDUE_MODS = [8, 125, 1000]  # 3-lane validation
DEFAULT_MAX_OFFSET = 5          # Maximum offset for alignment
DEFAULT_TEMPORAL_WINDOW = 100   # Window size for stability
DEFAULT_TEMPORAL_WINDOWS = 5    # Number of windows
DEFAULT_MIN_CONFIDENCE = 0.1    # Minimum confidence threshold
```

### 3.2 Why [8, 125, 1000]?

| Mod | Purpose | Bit Pattern |
|-----|---------|-------------|
| **8** | Fast power-of-2 check | Low 3 bits |
| **125** | Coprime to 8 | Different bit range |
| **1000** | Full 3-digit match | Complete residue |

**Mathematical Property:** 8 × 125 = 1000, so checking all three provides orthogonal validation.

### 3.3 Java LCG Fallback

```python
def java_lcg_sequence(seed: int, n: int, mod: int) -> np.ndarray:
    """CPU-only fallback for tiny sequences"""
    arr = np.empty(n, dtype=np.int64)
    state = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
    for i in range(n):
        state = (state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        arr[i] = (state >> 16) % mod
    return arr
```

---

## 4. SurvivorScorer Class

### 4.1 Constructor

```python
class SurvivorScorer:
    def __init__(self, 
                 prng_type: str = 'java_lcg', 
                 mod: int = 1000, 
                 residue_mods: List[int] = None, 
                 config_dict: Optional[Dict] = None):
```

### 4.2 Optuna Parameter Translation (BUG FIX 1)

```python
# Optuna passes: {"residue_mod_1": 14, "residue_mod_2": 137, "residue_mod_3": 1136}
# But self.residue_mods expects a LIST: [14, 137, 1136]
if 'residue_mod_1' in config_dict:
    residue_mods = [
        config_dict.get('residue_mod_1', 8),
        config_dict.get('residue_mod_2', 125),
        config_dict.get('residue_mod_3', 1000)
    ]
```

### 4.3 Instance Attributes

| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `prng_type` | str | 'java_lcg' | PRNG algorithm |
| `mod` | int | 1000 | Output modulo |
| `residue_mods` | List[int] | [8, 125, 1000] | Validation lanes |
| `max_offset` | int | 5 | Alignment offset |
| `temporal_window_size` | int | 100 | Window size |
| `temporal_num_windows` | int | 5 | Number of windows |
| `min_confidence_threshold` | float | 0.1 | Minimum confidence |
| `device` | str | 'cuda'/'cpu' | PyTorch device |

### 4.4 Sequence Generation

```python
def _generate_sequence(self, seed: int, n: int, skip: int = 0) -> np.ndarray:
    """
    Generate PRNG sequence using configured prng_type.
    Uses prng_registry for dynamic PRNG lookup - NO HARDCODING.
    """
    raw = self._cpu_func(seed=int(seed), n=n, skip=skip)
    return np.array([v % self.mod for v in raw], dtype=np.int64)
```

### 4.5 Seed Coercion

```python
def _coerce_seed_list(self, items) -> List[int]:
    """Convert mixed list (int or dict with seed) to list of ints."""
    out = []
    for x in items or []:
        if isinstance(x, dict):
            if "seed" in x:
                out.append(int(x["seed"]))
        else:
            out.append(int(x))
    return out
```

---

## 5. Feature Extraction

### 5.1 extract_ml_features()

```python
def extract_ml_features(self, 
                        seed: int, 
                        lottery_history: List[int], 
                        forward_survivors=None, 
                        reverse_survivors=None, 
                        skip: int = 0) -> Dict[str, float]:
    """
    Extract 46 ML features for a single seed.
    Uses PyTorch tensors — proven stable on ROCm.
    """
```

### 5.2 Feature Extraction Flow

```
1. Generate PRNG sequence
2. Convert to PyTorch tensors (two-step transfer for ROCm)
3. Compute base matching score
4. Compute residue features for each mod
5. Compute temporal stability across windows
6. Compute statistical features
7. Compute lane agreement
8. Fill defaults for bidirectional features
9. Clear VRAM
10. Return feature dict
```

### 5.3 Two-Step Tensor Transfer (BUG FIX 5)

```python
# CRITICAL: Explicit two-step transfer (NumPy -> CPU Tensor -> GPU Tensor)
# This stabilizes data transfer on the crowded, low-bandwidth RX 6600 PCIe bus.
pred_cpu = torch.from_numpy(seq).to(torch.long)
pred = pred_cpu.to(self.device)
act = torch.tensor(hist_np, device=self.device, dtype=torch.long)
```

### 5.4 Residue Features

```python
for mod in self.residue_mods:
    p_res = pred % mod
    a_res = act % mod
    match_rate = (p_res == a_res).float().mean().item()
    
    # KL divergence via histogram
    p_hist = torch.histc(p_res.float(), bins=mod, min=0, max=mod-1)
    a_hist = torch.histc(a_res.float(), bins=mod, min=0, max=mod-1)
    p_dist = (p_hist / p_hist.sum()).clamp(min=1e-10)
    a_dist = (a_hist / a_hist.sum()).clamp(min=1e-10)
    
    kl = entropy(p_dist.cpu().numpy(), a_dist.cpu().numpy())
    
    features[f'residue_{mod}_match_rate'] = match_rate
    features[f'residue_{mod}_coherence'] = 1.0 / (1.0 + kl)
    features[f'residue_{mod}_kl_divergence'] = kl
```

### 5.5 Temporal Stability (BUG FIX 2)

```python
# OPTIMIZED: Reuse seq already generated, no duplicate generation
scores = []
stride = max(1, (n - self.temporal_window_size) // self.temporal_num_windows)

for i in range(self.temporal_num_windows):
    s = i * stride
    e = min(s + self.temporal_window_size, n)
    if e - s < self.temporal_window_size // 2:
        break
    # Use seq[s:e] instead of regenerating
    scores.append(np.mean(seq[s:e] == hist_np[s:e]))

if scores:
    arr = np.array(scores)
    trend = np.polyfit(np.arange(len(arr)), arr, 1)[0] if len(arr) > 1 else 0.0
    features.update({
        'temporal_stability_mean': float(arr.mean()),
        'temporal_stability_std': float(arr.std()),
        'temporal_stability_min': float(arr.min()),
        'temporal_stability_max': float(arr.max()),
        'temporal_stability_trend': float(trend)
    })
```

---

## 6. Dual Sieve Intersection

### 6.1 compute_dual_sieve_intersection()

```python
def compute_dual_sieve_intersection(
    self,
    forward_survivors: List[int],
    reverse_survivors: List[int]
) -> Dict[str, Any]:
    """
    Compute intersection of forward and reverse sieve survivors.
    Per Team Beta: NEVER discard valid intersection, Jaccard is metadata.
    """
```

### 6.2 Return Structure

```python
{
    "intersection": [123, 456, 789],  # Sorted list of common seeds
    "jaccard": 0.15,                   # Jaccard similarity coefficient
    "counts": {
        "forward": 1000,
        "reverse": 800,
        "intersection": 150,
        "union": 1650
    }
}
```

### 6.3 Jaccard Similarity

```python
forward_set = set(forward_survivors)
reverse_set = set(reverse_survivors)
intersection = forward_set & reverse_set
union = forward_set | reverse_set
jaccard = len(intersection) / len(union) if union else 0.0
```

---

## 7. Batch Scoring

### 7.1 batch_score_vectorized()

```python
def batch_score_vectorized(self, 
                           seeds: Union[List[int], torch.Tensor], 
                           lottery_history: Union[List[int], torch.Tensor],
                           device: Optional[str] = None, 
                           return_dict: bool = False):
    """
    Vectorized scoring using GPU-accelerated PRNG generation.
    Returns either tensor of scores or list of dicts.
    """
```

### 7.2 Vectorized Kernel

```python
def _vectorized_scoring_kernel(self, seeds_tensor, lottery_history_tensor, device):
    batch_size = seeds_tensor.shape[0]
    history_len = lottery_history_tensor.shape[0]
    
    if has_pytorch_gpu(self.prng_type):
        prng_func = get_pytorch_gpu_reference(self.prng_type)
        predictions = prng_func(
            seeds=seeds_tensor, 
            n=history_len, 
            mod=self.mod,
            device=device, 
            skip=0
        )
    else:
        # CPU fallback
        predictions = self._cpu_batch_generate(...)
    
    # Vectorized matching
    matches = (predictions == lottery_history_tensor.unsqueeze(0))
    scores = matches.float().sum(dim=1) / history_len
    
    return scores
```

### 7.3 Legacy batch_score()

```python
def batch_score(self, 
                seeds: List[int], 
                lottery_history: List[int], 
                use_dual_gpu: bool = False, 
                window_metadata=None) -> List[Dict]:
    """
    Legacy per-seed scoring with full feature extraction.
    Slower but provides complete feature dicts.
    """
    results = []
    for seed in seeds:
        features = self.extract_ml_features(seed, lottery_history)
        results.append({
            'seed': seed, 
            'features': features, 
            'score': features['score']
        })
    return results
```

---

## 8. GPU-Batched Feature Extraction

### 8.1 extract_ml_features_batch()

```python
def extract_ml_features_batch(self, 
                               seeds: List[int], 
                               lottery_history: List[int], 
                               forward_survivors=None, 
                               reverse_survivors=None, 
                               survivor_metadata=None) -> List[Dict[str, float]]:
    """
    GPU-BATCHED ML feature extraction - CRYPTO MINER STYLE
    Processes ALL seeds in parallel on GPU, single CPU transfer at end.
    """
```

### 8.2 Batch Processing Flow

```
STEP 1: Generate ALL predictions on GPU
    └─→ seeds_t, hist_t tensors
    └─→ PyTorch GPU PRNG or CPU fallback
    └─→ predictions shape: (batch_size, n)

STEP 2: Compute ALL features on GPU
    └─→ Base matching
    └─→ Stats (mean, std, min, max)
    └─→ Residuals
    └─→ Lane agreement
    └─→ Residue features (vectorized histogram)
    └─→ Temporal stability

STEP 3: SINGLE CPU TRANSFER
    └─→ Stack all tensors
    └─→ One .cpu().numpy() call
    └─→ Build result dicts
```

### 8.3 Vectorized Histogram (No Python Loops)

```python
# VECTORIZED batch histogram using scatter_add - NO PYTHON LOOPS!

# Create batch indices: [0,0,0,...,1,1,1,...,2,2,2,...]
batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, n)

# Flatten for scatter: (batch_size * n,)
p_flat = p_res.reshape(-1)
batch_flat = batch_idx.reshape(-1)

# Compute combined index: batch_idx * mod + residue_value
p_scatter_idx = batch_flat * mod + p_flat

# Scatter to build histograms: (batch_size * mod,)
p_hists = torch.zeros(batch_size * mod, device=device)
ones = torch.ones_like(p_flat, dtype=torch.float32)
p_hists.scatter_add_(0, p_scatter_idx, ones)

# Reshape to (batch_size, mod)
p_hists = p_hists.reshape(batch_size, mod)
```

### 8.4 Single Transfer Optimization

```python
# Collect all tensors and transfer once
results_gpu = {
    'score': base_scores * 100,
    'confidence': torch.clamp(base_scores, min=self.min_confidence_threshold),
    'exact_matches': match_counts,
    # ... all other features
}

# Stack all into one tensor, transfer, unpack
keys = list(results_gpu.keys())
stacked = torch.stack([results_gpu[k] for k in keys], dim=1)
stacked_cpu = stacked.cpu().numpy()  # SINGLE TRANSFER!
```

### 8.5 Metadata Integration

```python
# Add metadata if available
if survivor_metadata and seed in survivor_metadata:
    meta = survivor_metadata[seed]
    for field in ['forward_count', 'reverse_count', 'bidirectional_count',
                  'intersection_count', 'intersection_ratio', 
                  'skip_min', 'skip_max', 'skip_range', 
                  'skip_mean', 'skip_std', 'skip_entropy', ...]:
        if field in meta and meta[field] is not None:
            features[field] = float(meta[field])
```

---

## 9. Complete Feature Reference

### 9.1 Base Features (5)

| Feature | Description | Range |
|---------|-------------|-------|
| `score` | Base match rate × 100 | 0-100 |
| `confidence` | Clamped match rate | 0.1-1.0 |
| `exact_matches` | Count of exact matches | 0-n |
| `total_predictions` | Sequence length | n |
| `best_offset` | Best alignment offset | 0-5 |

### 9.2 Residue Features (9 per default config)

For each mod in [8, 125, 1000]:

| Feature | Description |
|---------|-------------|
| `residue_{mod}_match_rate` | Match rate at this modulo |
| `residue_{mod}_coherence` | 1 / (1 + KL divergence) |
| `residue_{mod}_kl_divergence` | KL divergence of distributions |

### 9.3 Temporal Stability Features (5)

| Feature | Description |
|---------|-------------|
| `temporal_stability_mean` | Average window match rate |
| `temporal_stability_std` | Std dev of window rates |
| `temporal_stability_min` | Minimum window rate |
| `temporal_stability_max` | Maximum window rate |
| `temporal_stability_trend` | Linear regression slope |

### 9.4 Statistical Features (10)

| Feature | Description |
|---------|-------------|
| `pred_mean` | Mean of predictions |
| `pred_std` | Std dev of predictions |
| `pred_min` | Minimum prediction |
| `pred_max` | Maximum prediction |
| `actual_mean` | Mean of actual values |
| `actual_std` | Std dev of actual values |
| `residual_mean` | Mean of (pred - actual) |
| `residual_std` | Std dev of residuals |
| `residual_abs_mean` | Mean absolute residual |
| `residual_max_abs` | Maximum absolute residual |

### 9.5 Lane Agreement Features (3)

| Feature | Description |
|---------|-------------|
| `lane_agreement_8` | Match rate mod 8 |
| `lane_agreement_125` | Match rate mod 125 |
| `lane_consistency` | Average of lane agreements |

### 9.6 Bidirectional/Metadata Features (14)

| Feature | Source | Description |
|---------|--------|-------------|
| `forward_count` | Metadata | Forward sieve survivors |
| `reverse_count` | Metadata | Reverse sieve survivors |
| `intersection_count` | Metadata | Bidirectional intersection |
| `intersection_ratio` | Metadata | intersection / union |
| `survivor_overlap_ratio` | Metadata | Overlap ratio |
| `forward_only_count` | Metadata | Forward-only seeds |
| `reverse_only_count` | Metadata | Reverse-only seeds |
| `bidirectional_count` | Metadata | Bidirectional seeds |
| `bidirectional_selectivity` | Metadata | Selectivity score |
| `skip_min` | Metadata | Minimum skip value |
| `skip_max` | Metadata | Maximum skip value |
| `skip_range` | Metadata | skip_max - skip_min |
| `skip_mean` | Metadata | Average skip |
| `skip_std` | Metadata | Skip std dev |
| `skip_entropy` | Metadata | Skip entropy |
| `survivor_velocity` | Metadata | Survivor count velocity |
| `velocity_acceleration` | Metadata | Velocity change rate |
| `intersection_weight` | Metadata | Weighted intersection |

### 9.7 Holdout Performance Feature (1)

| Feature | Description | Range | Purpose |
|---------|-------------|-------|---------|
| `holdout_hits` | Match rate on holdout data | 0.0 - 1.0 | ML y-label for Step 5 |

**Why This Feature Exists:**

This is the ONLY non-circular training target. All other features measure training performance (what the sieve already validated). `holdout_hits` measures future performance (generalization).

### 9.8 Total Feature Count: 47

```
Base:                5
Residue (3 mods):    9
Temporal:            5
Statistical:        10
Lane Agreement:      3
Bidirectional:      14
Holdout:             1   ← NEW
─────────────────────────
TOTAL:              47
```

---

## 10. Integration Points

### 10.1 Required Files

| File | Purpose |
|------|---------|
| `prng_registry.py` | PRNG implementations |

### 10.2 Imports from prng_registry

```python
from prng_registry import (
    get_cpu_reference,        # CPU PRNG function
    get_pytorch_gpu_reference, # PyTorch GPU function
    has_pytorch_gpu,          # Check GPU support
    list_pytorch_gpu_prngs,   # List GPU PRNGs
    get_kernel_info           # Kernel metadata
)
```

### 10.3 Usage by Other Modules

| Module | Usage |
|--------|-------|
| `reinforcement_engine.py` | Feature extraction |
| `meta_prediction_optimizer.py` | Scoring |
| `coordinator.py` | Batch processing |

### 10.4 Input Data Formats

**Seeds:**
```python
# Format 1: List of ints
seeds = [123, 456, 789]

# Format 2: List of dicts
seeds = [{'seed': 123, 'match_rate': 0.8}, ...]
```

**Lottery History:**
```python
lottery_history = [123, 456, 789, ...]  # List of draw values
```

**Survivor Metadata:**
```python
survivor_metadata = {
    123: {
        'forward_count': 50,
        'reverse_count': 30,
        'intersection_count': 15,
        'skip_min': 1,
        'skip_max': 10,
        ...
    },
    456: {...}
}
```

### 10.5 Holdout History Integration

**Files Added/Modified:**

| File | Change |
|------|--------|
| `full_scoring_worker.py` | Added `--holdout-history` CLI, `compute_holdout_hits_batch()` |
| `generate_step3_scoring_jobs.py` | Added `--holdout-history` to job args |
| `run_step3_full_scoring.sh` | Added `HOLDOUT_HISTORY` variable, scp to remotes |

**compute_holdout_hits_batch() Function:**
```python
def compute_holdout_hits_batch(
    seeds: List[int],
    holdout_history: List[int],
    train_history_len: int,      # Used to derive offset
    prng_type: str = 'java_lcg',
    mod: int = 1000
) -> Dict[int, float]:
    """
    Compute holdout Hit@K for multiple seeds.
    
    CRITICAL: offset is DERIVED from train_history_len, not configurable.
    """
    offset = train_history_len  # THE LAW
    
    for seed in seeds:
        predictions = prng_func(seed, n_holdout, skip=offset)
        hits = sum(1 for pos, actual in enumerate(holdout_history) 
                   if predictions[pos] % mod == actual)
        results[seed] = hits / n_holdout
    
    return results
```

**Output Format (per survivor):**
```json
{
  "seed": 12345,
  "holdout_hits": 1.0,
  "score": 85.5,
  "features": { ... }
}
```

---

## 11. Method Reference

### 11.1 SurvivorScorer Methods

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__()` | `prng_type, mod, residue_mods, config_dict` | — | Initialize |
| `_generate_sequence()` | `seed, n, skip` | `np.ndarray` | Generate PRNG sequence |
| `_coerce_seed_list()` | `items` | `List[int]` | Convert mixed to ints |
| `compute_dual_sieve_intersection()` | `forward, reverse` | `Dict` | Intersection analysis |
| `extract_ml_features()` | `seed, lottery_history, ...` | `Dict[str, float]` | Single seed features |
| `extract_ml_features_batch()` | `seeds, lottery_history, ...` | `List[Dict]` | Batch GPU features |
| `batch_score_vectorized()` | `seeds, lottery_history, ...` | `Tensor/List` | Fast vectorized scoring |
| `batch_score()` | `seeds, lottery_history, ...` | `List[Dict]` | Legacy full features |
| `_vectorized_scoring_kernel()` | `seeds_t, hist_t, device` | `Tensor` | Core scoring kernel |
| `_cpu_batch_generate()` | `seeds, n` | `np.ndarray` | CPU fallback generation |
| `_empty_ml_features()` | — | `Dict` | Zero-filled features |

### 11.2 Module-Level Functions

| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `entropy()` | `p, q, ...` | `float` | Safe CuPy → NumPy entropy |
| `java_lcg_sequence()` | `seed, n, mod` | `np.ndarray` | CPU fallback LCG |

---

## 12. Example Usage

### 12.1 Basic Feature Extraction

```python
from survivor_scorer import SurvivorScorer

# Initialize
scorer = SurvivorScorer(prng_type='java_lcg', mod=1000)

# Single seed
seed = 12345
lottery_history = [123, 456, 789, ...]  # 5000 draws
features = scorer.extract_ml_features(seed, lottery_history)

print(f"Score: {features['score']:.2f}")
print(f"Temporal stability: {features['temporal_stability_mean']:.4f}")
```

### 12.2 Batch Scoring

```python
# Vectorized (fast, scores only)
seeds = list(range(1000000, 1001000))
scores = scorer.batch_score_vectorized(seeds, lottery_history)
print(f"Top score: {scores.max():.4f}")

# Full features (slower)
results = scorer.batch_score(seeds[:100], lottery_history)
for r in results[:5]:
    print(f"Seed {r['seed']}: {r['score']:.2f}")
```

### 12.3 GPU-Batched Features

```python
# Crypto miner style - all parallel
features_list = scorer.extract_ml_features_batch(
    seeds=seeds[:1000],
    lottery_history=lottery_history,
    survivor_metadata=metadata_dict
)

for features in features_list[:3]:
    print(f"Score: {features['score']:.2f}, "
          f"Residue 8: {features['residue_8_match_rate']:.4f}")
```

### 12.4 Dual Sieve Intersection

```python
forward = [123, 456, 789, 101, 102]
reverse = [789, 101, 102, 103, 104]

result = scorer.compute_dual_sieve_intersection(forward, reverse)

print(f"Intersection: {result['intersection']}")  # [789, 101, 102]
print(f"Jaccard: {result['jaccard']:.4f}")        # 0.375
```

### 12.5 With Optuna Config

```python
# Optuna passes individual params
config_dict = {
    'residue_mod_1': 14,
    'residue_mod_2': 137,
    'residue_mod_3': 1136,
    'max_offset': 3,
    'temporal_window_size': 50
}

scorer = SurvivorScorer(
    prng_type='java_lcg',
    mod=1000,
    config_dict=config_dict
)

# residue_mods is now [14, 137, 1136]
```

---

## 13. Performance Notes

### 13.1 Memory Management

```python
# After batch operations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 13.2 RX 6600 Specific

- VRAM limited to 80% (6.4GB)
- PYTORCH_HIP_ALLOC_CONF for fragmentation
- Two-step tensor transfer for PCIe stability
- cudnn.benchmark disabled

### 13.3 Batch vs Single

| Method | Speed | Features | Use Case |
|--------|-------|----------|----------|
| `batch_score_vectorized()` | Fastest | Score only | Initial filtering |
| `extract_ml_features_batch()` | Fast | All 46 | GPU training |
| `extract_ml_features()` | Slow | All 46 | Single seed analysis |
| `batch_score()` | Slowest | All 46 | Legacy compatibility |

---

## 14. Chapter Summary

**Chapter 6: Survivor Scorer** covers the feature extraction pipeline:

| Component | Lines | Purpose |
|-----------|-------|---------|
| Environment setup | ~30 | ROCm/VRAM configuration |
| SurvivorScorer class | ~100 | Core scorer |
| `extract_ml_features()` | ~100 | Single seed features |
| `extract_ml_features_batch()` | ~150 | GPU batch features |
| Batch scoring | ~50 | Vectorized scoring |
| Helpers | ~30 | Utilities |

**Key Achievements:**
- 46 ML features per seed
- GPU-accelerated batch processing
- ROCm compatibility for AMD GPUs
- Memory management for 8GB GPUs
- Single transfer optimization

---

## Next Chapter

**Chapter 7: GPU Optimizer** will cover:
- `gpu_optimizer.py` — GPU binding and distribution
- Platform detection (CUDA/ROCm)
- Memory management
- Job allocation

---

*End of Chapter 6: Survivor Scorer*
