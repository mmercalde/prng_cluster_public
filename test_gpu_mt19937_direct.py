#!/usr/bin/env python3
"""
Direct GPU test - bypass all infrastructure
Test if GPU kernel can find a known seed
"""

import cupy as cp
from prng_registry import get_kernel_info
import numpy as np

# Get MT19937 kernel
config = get_kernel_info('mt19937')
kernel = cp.RawKernel(config['kernel_source'], config['kernel_name'])

# Test parameters
seed = 1706817600
skip = 5
k = 10  # Test with just 10 draws first

# Generate expected outputs with CPU
from prng_registry import mt19937_cpu_simple
all_outputs = mt19937_cpu_simple(seed, k * (skip + 1), skip=0)
residues = []
for i in range(k):
    idx = i * (skip + 1) + skip
    residues.append(all_outputs[idx])

print(f"=== Direct GPU MT19937 Test ===")
print(f"Seed: {seed}")
print(f"Skip: {skip}")
print(f"Expected residues: {[r % 1000 for r in residues[:5]]}")

# Prepare GPU arrays
seeds_gpu = cp.array([seed], dtype=cp.uint32)
residues_gpu = cp.array(residues, dtype=cp.uint32)
survivors_gpu = cp.zeros(1, dtype=cp.uint32)
match_rates_gpu = cp.zeros(1, dtype=cp.float32)
best_skips_gpu = cp.zeros(1, dtype=cp.uint8)
survivor_count_gpu = cp.zeros(1, dtype=cp.uint32)

# Run kernel
block_size = 256
grid_size = 1

kernel(
    (grid_size,), (block_size,),
    (
        seeds_gpu, residues_gpu, survivors_gpu,
        match_rates_gpu, best_skips_gpu, survivor_count_gpu,
        1,  # n_seeds
        k,  # k (window size)
        skip,  # skip_min
        skip,  # skip_max (test exact skip)
        0.5,  # threshold (50%)
        0  # offset
    )
)

# Get results
count = int(survivor_count_gpu[0])
print(f"\nGPU Results:")
print(f"  Survivors found: {count}")

if count > 0:
    survivor_seed = int(survivors_gpu[0])
    match_rate = float(match_rates_gpu[0])
    best_skip = int(best_skips_gpu[0])
    
    print(f"  Survivor seed: {survivor_seed}")
    print(f"  Match rate: {match_rate:.1%}")
    print(f"  Best skip: {best_skip}")
    
    if survivor_seed == seed and best_skip == skip:
        print("\n✅ SUCCESS! GPU kernel works correctly!")
    else:
        print(f"\n⚠️  Found survivor but wrong values")
        print(f"     Expected: seed={seed}, skip={skip}")
else:
    print("\n❌ FAIL! GPU kernel found nothing")
    print("   Debugging needed in GPU kernel code")
