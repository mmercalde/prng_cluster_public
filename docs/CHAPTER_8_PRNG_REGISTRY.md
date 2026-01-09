# Chapter 8: PRNG Registry

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 2.4  
**File:** `prng_registry.py`  
**Lines:** 4,323  
**Purpose:** Complete PRNG kernel library with GPU sieve implementations

---

## Table of Contents

1. [Overview](#1-overview)
2. [Registry Structure](#2-registry-structure)
3. [PRNG Categories](#3-prng-categories)
4. [CPU Reference Implementations](#4-cpu-reference-implementations)
5. [GPU Kernel Architecture](#5-gpu-kernel-architecture)
6. [Kernel Memory Layout](#6-kernel-memory-layout)
7. [Standard Sieve Kernels](#7-standard-sieve-kernels)
8. [Hybrid Sieve Kernels](#8-hybrid-sieve-kernels)
9. [Reverse Sieve Kernels](#9-reverse-sieve-kernels)
10. [PyTorch GPU Implementations](#10-pytorch-gpu-implementations)
11. [Adding New PRNGs](#11-adding-new-prngs)
12. [Helper Functions](#12-helper-functions)
13. [Complete PRNG Reference](#13-complete-prng-reference)

---

## 1. Overview

### 1.1 What the PRNG Registry Does

The PRNG Registry is the **kernel library** for the entire system. It contains:

- **46 PRNG implementations** across 14 families
- **GPU kernels** for forward and reverse sieves (CuPy/CUDA/ROCm)
- **CPU reference implementations** for verification
- **PyTorch GPU implementations** for ML scoring (v2.4)

### 1.2 Version History

```
Version 2.4 - November 27, 2025
- ENHANCEMENT: PyTorch GPU implementations for Step 2.5 ML scoring
  * Works on both CUDA (NVIDIA) and ROCm (AMD)
  * Phase 1: java_lcg, java_lcg_hybrid implemented

Version 2.3 - October 29, 2025
- CRITICAL FIX: Replaced hardcoded 512 buffer sizes with dynamic sizing
  * Changed 71 instances across all hybrid kernels
  * Arrays now: best_skip_seq[2048], current_skip_seq[2048]
  * Window sizes up to 2048 draws now supported

- CRITICAL FIX: Changed skip_sequences stride from hardcoded to dynamic
  * Changed: skip_sequences[pos * k + i] (was pos * 512 + i)
  * Eliminated illegal memory access causing GPU crashes
```

### 1.3 Key Statistics

| Metric | Count |
|--------|-------|
| Total PRNGs | 46 |
| Fixed Skip | 14 |
| Hybrid (Variable Skip) | 14 |
| Reverse Sieve | 18 |
| Max Window Size | 2048 draws |
| Seed Types | uint32, uint64 |

---

## 2. Registry Structure

### 2.1 KERNEL_REGISTRY Dictionary

```python
KERNEL_REGISTRY = {
    'prng_name': {
        'kernel_source': KERNEL_SOURCE_STRING,  # CUDA/HIP kernel code
        'kernel_name': 'kernel_function_name',  # Entry point name
        'cpu_reference': cpu_function,          # Python CPU implementation
        'pytorch_gpu': pytorch_function,        # PyTorch GPU (if available)
        'default_params': {                     # PRNG-specific parameters
            'a': 25214903917,
            'c': 11,
        },
        'description': 'Human-readable description',
        'seed_type': 'uint32' | 'uint64',       # Seed data type
        'state_size': 4,                        # State size in bytes
        'variable_skip': True | False,          # Supports variable skip?
        'multi_strategy': True | False,         # Multi-strategy hybrid?
    },
    ...
}
```

### 2.2 Naming Convention

| Pattern | Meaning | Example |
|---------|---------|---------|
| `{prng}` | Standard forward sieve | `java_lcg` |
| `{prng}_hybrid` | Variable skip forward | `java_lcg_hybrid` |
| `{prng}_reverse` | Standard reverse sieve | `java_lcg_reverse` |
| `{prng}_hybrid_reverse` | Variable skip reverse | `java_lcg_hybrid_reverse` |

---

## 3. PRNG Categories

### 3.1 Fixed Skip PRNGs (14 total)

Standard sieves with constant skip between outputs.

| PRNG | Seed Type | State Size | Description |
|------|-----------|------------|-------------|
| `xorshift32` | uint32 | 4 bytes | Xorshift32 with configurable shifts |
| `xorshift64` | uint64 | 8 bytes | Xorshift64 with Weyl sequence |
| `xorshift128` | uint32 | 16 bytes | 128-bit state xorshift |
| `pcg32` | uint32 | 8 bytes | PCG-XSH-RR 32-bit |
| `lcg32` | uint32 | 4 bytes | Generic 32-bit LCG |
| `java_lcg` | uint64 | 8 bytes | Java Random (48-bit) |
| `minstd` | uint32 | 4 bytes | Park & Miller minimal standard |
| `mt19937` | uint32 | 2496 bytes | Mersenne Twister (624-word state) |
| `xoshiro256pp` | uint64 | 32 bytes | Modern high-quality PRNG |
| `philox4x32` | uint64 | 8 bytes | Counter-based (GPU-optimized) |
| `sfc64` | uint64 | 32 bytes | Small Fast Counting |

### 3.2 Hybrid PRNGs (14 total)

Variable skip pattern detection with multi-strategy search.

| PRNG | Base | Description |
|------|------|-------------|
| `xorshift32_hybrid` | xorshift32 | Variable skip detection |
| `xorshift64_hybrid` | xorshift64 | Variable skip detection |
| `xorshift128_hybrid` | xorshift128 | Variable skip detection |
| `pcg32_hybrid` | pcg32 | Variable skip detection |
| `lcg32_hybrid` | lcg32 | Variable skip detection |
| `java_lcg_hybrid` | java_lcg | Variable skip detection |
| `minstd_hybrid` | minstd | Variable skip detection |
| `mt19937_hybrid` | mt19937 | Variable skip detection |
| `xoshiro256pp_hybrid` | xoshiro256pp | Variable skip detection |
| `philox4x32_hybrid` | philox4x32 | Variable skip detection |
| `sfc64_hybrid` | sfc64 | Variable skip detection |

### 3.3 Reverse Sieve PRNGs (18 total)

Backward validation from candidate seeds.

| Forward | Reverse | Hybrid Reverse |
|---------|---------|----------------|
| `xorshift32` | `xorshift32_reverse` | `xorshift32_hybrid_reverse` |
| `xorshift64` | `xorshift64_reverse` | `xorshift64_hybrid_reverse` |
| `xorshift128` | `xorshift128_reverse` | `xorshift128_hybrid_reverse` |
| `pcg32` | `pcg32_reverse` | `pcg32_hybrid_reverse` |
| `lcg32` | `lcg32_reverse` | `lcg32_hybrid_reverse` |
| `java_lcg` | `java_lcg_reverse` | `java_lcg_hybrid_reverse` |
| `minstd` | `minstd_reverse` | `minstd_hybrid_reverse` |
| `mt19937` | `mt19937_reverse` | `mt19937_hybrid_reverse` |
| `philox4x32` | `philox4x32_reverse` | `philox4x32_hybrid_reverse` |

---

## 4. CPU Reference Implementations

### 4.1 Purpose

CPU implementations serve as:
1. **Verification** — Validate GPU results
2. **Fallback** — When GPU unavailable
3. **Documentation** — Reference for algorithm behavior

### 4.2 Common Signature

```python
def prng_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """
    Generate n outputs from PRNG starting at seed.
    
    Args:
        seed: Initial seed value
        n: Number of outputs to generate
        skip: Number of outputs to skip before first
        **kwargs: PRNG-specific parameters
        
    Returns:
        List of n integer outputs
    """
```

### 4.3 Java LCG Example

```python
def java_lcg_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """Java LCG (java.util.Random) CPU reference"""
    a = kwargs.get('a', 25214903917)
    c = kwargs.get('c', 11)
    m = (1 << 48)  # 2^48
    
    state = seed & (m - 1)
    
    # Skip initial outputs
    for _ in range(skip):
        state = (a * state + c) & (m - 1)
    
    outputs = []
    for _ in range(n):
        state = (a * state + c) & (m - 1)
        output = (state >> 16) & 0xFFFFFFFF  # Extract bits 16-47
        outputs.append(output)
    
    return outputs
```

### 4.4 MT19937 Example

```python
def mt19937_cpu_simple(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """MT19937 with init_genrand (matches GPU kernel)"""
    
    # Initialize 624-word state array
    state = [0] * 624
    state[0] = seed & 0xFFFFFFFF
    for i in range(1, 624):
        state[i] = (1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i) & 0xFFFFFFFF
    
    index = 624  # Force twist on first extract
    
    def mt19937_extract():
        nonlocal index
        if index >= 624:
            # Twist operation
            for i in range(624):
                y = (state[i] & 0x80000000) + (state[(i+1) % 624] & 0x7FFFFFFF)
                state[i] = state[(i + 397) % 624] ^ (y >> 1)
                if y % 2 != 0:
                    state[i] ^= 0x9908B0DF
            index = 0
        
        # Extract and temper
        y = state[index]
        index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF
    
    # Skip outputs
    for _ in range(skip):
        mt19937_extract()
    
    # Generate outputs
    return [mt19937_extract() for _ in range(n)]
```

---

## 5. GPU Kernel Architecture

### 5.1 Common Kernel Structure

All GPU kernels follow this pattern:

```c
extern "C" __global__
void prng_flexible_sieve(
    // Input arrays
    TYPE* seeds,              // Candidate seeds
    unsigned int* residues,   // Target draws to match
    
    // Output arrays
    TYPE* survivors,          // Seeds that passed threshold
    float* match_rates,       // Match rate for each survivor
    unsigned char* best_skips,// Best skip value found
    unsigned int* survivor_count,  // Atomic counter
    
    // Parameters
    int n_seeds,              // Number of seeds to test
    int k,                    // Number of residues (window size)
    int skip_min,             // Minimum skip to try
    int skip_max,             // Maximum skip to try
    float threshold,          // Minimum match rate
    
    // PRNG-specific parameters...
    int offset                // Window offset (always LAST)
) {
    // Thread index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    // Get this thread's seed
    TYPE seed = seeds[idx];
    
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    // Try each skip value
    for (int skip = skip_min; skip <= skip_max; skip++) {
        // Initialize PRNG state
        TYPE state = seed;
        
        // Pre-advance by offset
        for (int o = 0; o < offset; o++) {
            state = prng_step(state);
        }
        
        // Burn skip values before first draw
        for (int s = 0; s < skip; s++) {
            state = prng_step(state);
        }
        
        int matches = 0;
        
        // Generate and compare
        for (int i = 0; i < k; i++) {
            state = prng_step(state);
            unsigned int output = extract_output(state);
            
            // 3-lane modulo validation
            if (((output % 1000) == (residues[i] % 1000)) &&
                ((output % 8) == (residues[i] % 8)) &&
                ((output % 125) == (residues[i] % 125))) {
                matches++;
            }
            
            // Skip between draws
            for (int s = 0; s < skip; s++) {
                state = prng_step(state);
            }
        }
        
        float rate = (float)matches / (float)k;
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    // Store survivor if above threshold
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seed;
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
```

### 5.2 3-Lane Modulo Validation

All kernels use a 3-lane modulo check for robustness:

```c
if (((output % 1000) == (residues[i] % 1000)) &&
    ((output % 8) == (residues[i] % 8)) &&
    ((output % 125) == (residues[i] % 125))) {
    matches++;
}
```

**Why 3 lanes?**
- `% 1000`: Matches the 3-digit draw directly
- `% 8`: Fast power-of-2 check
- `% 125`: Coprime to 8, covers different bit patterns

---

## 6. Kernel Memory Layout

### 6.1 Standard Sieve Memory

| Array | Size | Type | Purpose |
|-------|------|------|---------|
| `seeds` | n_seeds | uint32/uint64 | Input seeds |
| `residues` | k | uint32 | Target draws |
| `survivors` | n_seeds | uint32/uint64 | Output seeds |
| `match_rates` | n_seeds | float32 | Output rates |
| `best_skips` | n_seeds | uint8 | Output skip values |
| `survivor_count` | 1 | uint32 | Atomic counter |

### 6.2 Hybrid Sieve Memory

Additional arrays for variable skip:

| Array | Size | Type | Purpose |
|-------|------|------|---------|
| `skip_sequences` | n_seeds × k | uint32 | Skip pattern per survivor |
| `strategy_ids` | n_seeds | uint32 | Winning strategy ID |

### 6.3 Local Arrays (Per Thread)

```c
// Version 2.3+: Dynamic sizing up to 2048
unsigned int best_skip_seq[2048];     // Best skip sequence found
unsigned int current_skip_seq[2048];  // Current trial sequence
```

**CRITICAL FIX (v2.3):** Changed from `[512]` to `[2048]` to support larger windows.

### 6.4 Skip Sequence Stride

```c
// FIXED in v2.3: Dynamic stride based on k (window size)
skip_sequences[pos * k + i] = best_skip_seq[i];

// BROKEN (before v2.3): Hardcoded 512 stride
skip_sequences[pos * 512 + i] = best_skip_seq[i];  // WRONG!
```

---

## 7. Standard Sieve Kernels

### 7.1 Xorshift32 Kernel

```c
extern "C" __global__
void xorshift32_flexible_sieve(
    unsigned int* seeds, unsigned int* residues, unsigned int* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold,
    int shift_a, int shift_b, int shift_c, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    unsigned int seed = seeds[idx];
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    for (int skip = skip_min; skip <= skip_max; skip++) {
        unsigned int state = seed;
        
        // Pre-advance by offset
        for (int o = 0; o < offset; o++) {
            state ^= state << shift_a;
            state ^= state >> shift_b;
            state ^= state << shift_c;
        }
        
        // Burn skip values
        for (int s = 0; s < skip; s++) {
            state ^= state << shift_a;
            state ^= state >> shift_b;
            state ^= state << shift_c;
        }
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            // Generate output
            state ^= state << shift_a;
            state ^= state >> shift_b;
            state ^= state << shift_c;
            
            // 3-lane check
            if (((state % 1000) == (residues[i] % 1000)) &&
                ((state % 8) == (residues[i] % 8)) &&
                ((state % 125) == (residues[i] % 125))) {
                matches++;
            }
            
            // Skip between draws
            for (int s = 0; s < skip; s++) {
                state ^= state << shift_a;
                state ^= state >> shift_b;
                state ^= state << shift_c;
            }
        }
        
        float rate = (float)matches / (float)k;
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seed;
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
```

### 7.2 Java LCG Kernel

```c
extern "C" __global__
void java_lcg_flexible_sieve(
    unsigned long long* seeds, unsigned int* residues, unsigned long long* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold,
    unsigned long long a, unsigned long long c, int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    unsigned long long seed = seeds[idx];
    const unsigned long long m = 0xFFFFFFFFFFFFULL;  // 2^48 - 1
    
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    for (int skip = skip_min; skip <= skip_max; skip++) {
        unsigned long long state = seed & m;
        
        // Pre-advance and burn
        for (int o = 0; o < offset; o++) state = (a * state + c) & m;
        for (int s = 0; s < skip; s++) state = (a * state + c) & m;
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            state = (a * state + c) & m;
            unsigned int output = (state >> 16) & 0xFFFFFFFF;
            
            if (((output % 1000) == (residues[i] % 1000)) &&
                ((output % 8) == (residues[i] % 8)) &&
                ((output % 125) == (residues[i] % 125))) {
                matches++;
            }
            
            for (int s = 0; s < skip; s++) state = (a * state + c) & m;
        }
        
        float rate = (float)matches / (float)k;
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seed;
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
```

---

## 8. Hybrid Sieve Kernels

### 8.1 Hybrid Kernel Architecture

Hybrid kernels search for variable skip patterns using multiple strategies:

```c
extern "C" __global__
void prng_hybrid_multi_strategy_sieve(
    TYPE* seeds, unsigned int* residues, TYPE* survivors,
    float* match_rates, unsigned int* skip_sequences, unsigned int* strategy_ids,
    unsigned int* survivor_count, int n_seeds, int k,
    int* strategy_max_misses, int* strategy_tolerances, int n_strategies,
    float threshold, /* PRNG params... */ int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    TYPE seed = seeds[idx];
    float best_match_rate = 0.0f;
    int best_strategy_id = 0;
    unsigned int best_skip_seq[2048];
    
    // Test each strategy
    for (int strat_id = 0; strat_id < n_strategies; strat_id++) {
        int max_misses = strategy_max_misses[strat_id];
        int skip_tolerance = strategy_tolerances[strat_id];
        
        TYPE state = seed;
        int matches = 0;
        int consecutive_misses = 0;
        int expected_skip = 5;  // Initial guess
        unsigned int current_skip_seq[2048];
        
        for (int draw_idx = 0; draw_idx < k && draw_idx < 2048; draw_idx++) {
            TYPE state_backup = state;
            bool found = false;
            int actual_skip = expected_skip;
            
            // Search window around expected skip
            int search_min = (expected_skip > skip_tolerance) ? 
                             (expected_skip - skip_tolerance) : 0;
            int search_max = expected_skip + skip_tolerance;
            
            for (int test_skip = search_min; test_skip <= search_max; test_skip++) {
                state = state_backup;
                
                // Advance by test_skip
                for (int j = 0; j < test_skip; j++) {
                    state = prng_step(state);
                }
                
                // Generate output
                TYPE temp_state = prng_step(state);
                unsigned int output = extract_output(temp_state);
                
                // Check match
                if (((output % 1000) == (residues[draw_idx] % 1000)) &&
                    ((output % 8) == (residues[draw_idx] % 8)) &&
                    ((output % 125) == (residues[draw_idx] % 125))) {
                    matches++;
                    consecutive_misses = 0;
                    actual_skip = test_skip;
                    expected_skip = test_skip;  // Adapt expectation
                    found = true;
                    state = temp_state;
                    break;
                }
            }
            
            if (draw_idx < 2048) current_skip_seq[draw_idx] = actual_skip;
            
            if (!found) {
                consecutive_misses++;
                if (consecutive_misses >= max_misses) break;  // Abandon strategy
            }
        }
        
        float match_rate = (float)matches / k;
        if (match_rate > best_match_rate) {
            best_match_rate = match_rate;
            best_strategy_id = strat_id;
            for (int i = 0; i < k && i < 2048; i++) {
                best_skip_seq[i] = current_skip_seq[i];
            }
        }
    }
    
    // Store survivor
    if (best_match_rate >= threshold) {
        int pos = atomicAdd(survivor_count, 1);
        if (pos < n_seeds) {
            survivors[pos] = seed;
            match_rates[pos] = best_match_rate;
            strategy_ids[pos] = best_strategy_id;
            
            // Store skip sequence with dynamic stride
            int seq_size = (k < 2048) ? k : 2048;
            for (int i = 0; i < seq_size; i++) {
                skip_sequences[pos * k + i] = best_skip_seq[i];
            }
        }
    }
}
```

### 8.2 Strategy Parameters

```c
// Typical strategy configurations
strategies = [
    {'max_consecutive_misses': 3, 'skip_tolerance': 5},   // Tight
    {'max_consecutive_misses': 5, 'skip_tolerance': 10},  // Medium
    {'max_consecutive_misses': 8, 'skip_tolerance': 15}   // Loose
]
```

| Parameter | Meaning |
|-----------|---------|
| `max_consecutive_misses` | Abort strategy after this many consecutive failures |
| `skip_tolerance` | Search window: `expected_skip ± tolerance` |

---

## 9. Reverse Sieve Kernels

### 9.1 Purpose

Reverse sieve kernels validate candidates from forward sieves by testing them against a **reversed** temporal sequence (looking backward in time).

### 9.2 Temporal Reversal

**CRITICAL:** The residue array is reversed BEFORE being passed to the kernel (handled in `sieve_filter.py`):

```python
# In sieve_filter.py
if '_reverse' in prng_family:
    residues_reversed = residues[::-1]
    residues_gpu = cp.array(residues_reversed, dtype=residue_dtype)
```

### 9.3 Reverse Kernel Example

```c
extern "C" __global__
void java_lcg_reverse_sieve(
    unsigned long long* candidate_seeds, unsigned int* residues,
    unsigned long long* survivors, float* match_rates,
    unsigned char* best_skips, unsigned int* survivor_count,
    int n_candidates, int k, int skip_min, int skip_max,
    float threshold, int offset
) {
    // Identical logic to forward kernel
    // Residues are already reversed by caller
    // ...
}
```

---

## 10. PyTorch GPU Implementations

### 10.1 Purpose (v2.4)

PyTorch GPU implementations enable:
- **ML scoring** in Step 2.5 without CuPy
- **Cross-platform** support (CUDA + ROCm)
- **Batch processing** with native PyTorch tensors

### 10.2 Lazy Import Pattern

**CRITICAL:** PyTorch must not be imported at module load to avoid breaking CuPy on ROCm:

```python
# DO NOT: import torch  # Breaks CuPy kernel compilation on ROCm

# DO: Lazy import
def _torch_available() -> bool:
    """Check if PyTorch is installed without importing it."""
    return importlib.util.find_spec("torch") is not None

def _get_torch():
    """Lazy import torch - only when actually needed."""
    if not _torch_available():
        raise RuntimeError("PyTorch not available")
    import torch
    return torch
```

### 10.3 PyTorch GPU Function Signature

```python
def java_lcg_pytorch_gpu(
    seeds: 'torch.Tensor',    # [N] tensor of seeds
    n: int,                   # Number of outputs
    mod: int,                 # Modulo for output
    device: str = 'cuda',     # 'cuda' or 'cuda:0', etc.
    skip: int = 0,            # Initial skip
    **kwargs                  # PRNG-specific params
) -> 'torch.Tensor':          # [N, n] output tensor
```

### 10.4 Java LCG PyTorch Implementation

```python
def java_lcg_pytorch_gpu(
    seeds: 'torch.Tensor',
    n: int,
    mod: int,
    device: str = 'cuda',
    skip: int = 0,
    **kwargs
) -> 'torch.Tensor':
    """PyTorch GPU implementation of Java LCG"""
    torch = _get_torch()
    
    a = kwargs.get('a', 25214903917)
    c = kwargs.get('c', 11)
    mask = (1 << 48) - 1
    
    seeds = seeds.to(device).long()
    N = seeds.shape[0]
    
    # Initialize state (no XOR - matches CPU reference)
    state = seeds & mask
    
    # Apply initial skip
    for _ in range(skip):
        state = (a * state + c) & mask
    
    # Preallocate output
    output = torch.zeros((N, n), dtype=torch.int64, device=device)
    
    # Generate sequence (vectorized over seeds)
    for i in range(n):
        state = (a * state + c) & mask
        output[:, i] = ((state >> 16) & 0xFFFFFFFF) % mod
    
    return output
```

### 10.5 Helper Functions

```python
def get_pytorch_gpu_reference(prng_family: str) -> Callable:
    """Get PyTorch GPU implementation for PRNG family."""
    info = get_kernel_info(prng_family)
    if 'pytorch_gpu' not in info:
        raise ValueError(f"PyTorch GPU not implemented for '{prng_family}'")
    return info['pytorch_gpu']

def has_pytorch_gpu(prng_family: str) -> bool:
    """Check if PRNG has PyTorch GPU implementation."""
    return 'pytorch_gpu' in get_kernel_info(prng_family)

def list_pytorch_gpu_prngs() -> List[str]:
    """List all PRNGs with PyTorch GPU support."""
    return [name for name, info in KERNEL_REGISTRY.items() 
            if 'pytorch_gpu' in info]
```

---

## 11. Adding New PRNGs

### 11.1 Step-by-Step Guide

1. **CPU Reference Implementation**
2. **GPU Kernel (Standard)**
3. **GPU Kernel (Hybrid)** — if variable skip needed
4. **GPU Kernel (Reverse)** — for bidirectional validation
5. **Registry Entry**
6. **Testing**

### 11.2 Template: CPU Reference

```python
def my_prng_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """My PRNG CPU reference implementation"""
    # Get parameters
    param1 = kwargs.get('param1', DEFAULT_VALUE)
    
    # Initialize state
    state = seed
    
    # Skip initial outputs
    for _ in range(skip):
        state = step_function(state, param1)
    
    # Generate outputs
    outputs = []
    for _ in range(n):
        state = step_function(state, param1)
        outputs.append(extract_output(state))
    
    return outputs
```

### 11.3 Template: GPU Kernel

```c
extern "C" __global__
void my_prng_flexible_sieve(
    TYPE* seeds, unsigned int* residues, TYPE* survivors,
    float* match_rates, unsigned char* best_skips, unsigned int* survivor_count,
    int n_seeds, int k, int skip_min, int skip_max, float threshold,
    /* PRNG params */ int offset
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_seeds) return;
    
    TYPE seed = seeds[idx];
    float best_rate = 0.0f;
    int best_skip_val = 0;
    
    for (int skip = skip_min; skip <= skip_max; skip++) {
        TYPE state = seed;
        
        // Pre-advance by offset
        for (int o = 0; o < offset; o++) {
            // PRNG step function here
        }
        
        // Burn skip
        for (int s = 0; s < skip; s++) {
            // PRNG step function here
        }
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            // Generate output
            // PRNG step function here
            unsigned int output = /* extract */;
            
            // 3-lane check
            if (((output % 1000) == (residues[i] % 1000)) &&
                ((output % 8) == (residues[i] % 8)) &&
                ((output % 125) == (residues[i] % 125))) {
                matches++;
            }
            
            // Skip between draws
            for (int s = 0; s < skip; s++) {
                // PRNG step function here
            }
        }
        
        float rate = (float)matches / (float)k;
        if (rate > best_rate) {
            best_rate = rate;
            best_skip_val = skip;
        }
    }
    
    if (best_rate >= threshold) {
        unsigned int pos = atomicAdd(survivor_count, 1);
        survivors[pos] = seed;
        match_rates[pos] = best_rate;
        best_skips[pos] = (unsigned char)best_skip_val;
    }
}
```

### 11.4 Registry Entry

```python
KERNEL_REGISTRY['my_prng'] = {
    'kernel_source': MY_PRNG_KERNEL,
    'kernel_name': 'my_prng_flexible_sieve',
    'cpu_reference': my_prng_cpu,
    'default_params': {
        'param1': DEFAULT_VALUE,
    },
    'description': 'My PRNG description',
    'seed_type': 'uint32',  # or 'uint64'
    'state_size': 4,        # bytes
}
```

---

## 12. Helper Functions

### 12.1 get_kernel_info()

```python
def get_kernel_info(prng_family: str) -> Dict[str, Any]:
    """Get kernel configuration for PRNG family"""
    if prng_family not in KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown PRNG family: {prng_family}. "
            f"Available: {list_available_prngs()}"
        )
    return KERNEL_REGISTRY[prng_family]
```

### 12.2 list_available_prngs()

```python
def list_available_prngs() -> List[str]:
    """List all available PRNG families"""
    return list(KERNEL_REGISTRY.keys())
```

### 12.3 get_cpu_reference()

```python
def get_cpu_reference(prng_family: str) -> Callable:
    """Get CPU reference implementation for PRNG"""
    return get_kernel_info(prng_family)['cpu_reference']
```

---

## 13. Complete PRNG Reference

### 13.1 All 46 PRNGs

| # | Name | Type | Seed | State | Variable Skip |
|---|------|------|------|-------|---------------|
| 1 | `xorshift32` | Forward | uint32 | 4B | No |
| 2 | `xorshift32_hybrid` | Forward | uint32 | 4B | **Yes** |
| 3 | `xorshift32_reverse` | Reverse | uint32 | 4B | No |
| 4 | `xorshift32_hybrid_reverse` | Reverse | uint32 | 4B | **Yes** |
| 5 | `xorshift64` | Forward | uint64 | 8B | No |
| 6 | `xorshift64_hybrid` | Forward | uint64 | 8B | **Yes** |
| 7 | `xorshift64_reverse` | Reverse | uint64 | 8B | No |
| 8 | `xorshift64_hybrid_reverse` | Reverse | uint64 | 8B | **Yes** |
| 9 | `xorshift128` | Forward | uint32 | 16B | No |
| 10 | `xorshift128_hybrid` | Forward | uint32 | 16B | **Yes** |
| 11 | `xorshift128_reverse` | Reverse | uint32 | 16B | No |
| 12 | `xorshift128_hybrid_reverse` | Reverse | uint32 | 16B | **Yes** |
| 13 | `pcg32` | Forward | uint32 | 8B | No |
| 14 | `pcg32_hybrid` | Forward | uint32 | 8B | **Yes** |
| 15 | `pcg32_reverse` | Reverse | uint32 | 8B | No |
| 16 | `pcg32_hybrid_reverse` | Reverse | uint32 | 8B | **Yes** |
| 17 | `lcg32` | Forward | uint32 | 4B | No |
| 18 | `lcg32_hybrid` | Forward | uint32 | 4B | **Yes** |
| 19 | `lcg32_reverse` | Reverse | uint32 | 4B | No |
| 20 | `lcg32_hybrid_reverse` | Reverse | uint32 | 4B | **Yes** |
| 21 | `java_lcg` | Forward | uint64 | 8B | No |
| 22 | `java_lcg_hybrid` | Forward | uint64 | 8B | **Yes** |
| 23 | `java_lcg_reverse` | Reverse | uint64 | 8B | No |
| 24 | `java_lcg_hybrid_reverse` | Reverse | uint64 | 8B | **Yes** |
| 25 | `minstd` | Forward | uint32 | 4B | No |
| 26 | `minstd_hybrid` | Forward | uint32 | 4B | **Yes** |
| 27 | `minstd_reverse` | Reverse | uint32 | 4B | No |
| 28 | `minstd_hybrid_reverse` | Reverse | uint32 | 4B | **Yes** |
| 29 | `mt19937` | Forward | uint32 | 2496B | No |
| 30 | `mt19937_hybrid` | Forward | uint32 | 2496B | **Yes** |
| 31 | `mt19937_reverse` | Reverse | uint32 | 2496B | No |
| 32 | `mt19937_hybrid_reverse` | Reverse | uint32 | 2496B | **Yes** |
| 33 | `xoshiro256pp` | Forward | uint64 | 32B | No |
| 34 | `xoshiro256pp_hybrid` | Forward | uint64 | 32B | **Yes** |
| 35 | `xoshiro256pp_reverse` | Reverse | uint64 | 32B | No |
| 36 | `xoshiro256pp_hybrid_reverse` | Reverse | uint64 | 32B | **Yes** |
| 37 | `philox4x32` | Forward | uint64 | 8B | No |
| 38 | `philox4x32_hybrid` | Forward | uint64 | 8B | **Yes** |
| 39 | `philox4x32_reverse` | Reverse | uint64 | 8B | No |
| 40 | `philox4x32_hybrid_reverse` | Reverse | uint64 | 8B | **Yes** |
| 41 | `sfc64` | Forward | uint64 | 32B | No |
| 42 | `sfc64_hybrid` | Forward | uint64 | 32B | **Yes** |
| 43 | `sfc64_reverse` | Reverse | uint64 | 32B | No |
| 44 | `sfc64_hybrid_reverse` | Reverse | uint64 | 32B | **Yes** |

### 13.2 PRNG Default Parameters

| PRNG | Parameters |
|------|------------|
| `xorshift32` | `shift_a=13, shift_b=17, shift_c=5` |
| `pcg32` | `increment=1442695040888963407` |
| `lcg32` | `a=1103515245, c=12345, m=0x7FFFFFFF` |
| `java_lcg` | `a=25214903917, c=11` |
| `minstd` | `a=48271, m=2147483647` |
| `mt19937` | (none - uses standard initialization) |
| `xoshiro256pp` | (none - uses standard constants) |
| `philox4x32` | (none - uses standard constants) |
| `sfc64` | (none - uses standard constants) |

---

## 14. Chapter Summary

**Chapter 3: PRNG Registry** covers the complete kernel library:

| Category | Count | Purpose |
|----------|-------|---------|
| **Fixed Skip** | 14 | Standard forward sieves |
| **Hybrid** | 14 | Variable skip detection |
| **Reverse** | 18 | Backward validation |
| **Total** | 46 | Complete PRNG coverage |

**Key Features:**
- GPU kernels (CUDA/ROCm via CuPy)
- CPU reference implementations
- PyTorch GPU implementations (v2.4)
- Support for window sizes up to 2048 draws
- 3-lane modulo validation
- Multi-strategy hybrid search

---

## Next Chapter

**Chapter 4: Meta-Optimizer (Step 2.5)** will cover:
- `meta_prediction_optimizer_anti_overfit.py` — ML model training
- Bayesian hyperparameter optimization
- 4 ML models: PyTorch, XGBoost, LightGBM, CatBoost
- Feature importance analysis

---

*End of Chapter 3: PRNG Registry*
