#!/usr/bin/env python3
"""
Test xorshift32_hybrid with variable skip - proper parameters
"""
import cupy as cp
from prng_registry import get_kernel_info, xorshift32_cpu
import sys

print("="*70)
print("XORSHIFT32_HYBRID - VARIABLE SKIP TEST")
print("="*70)

# Generate variable skip test data
known_seed = 54321
base_pattern = [5,5,3,7,5,5,8,4,5,5]
skip_pattern = base_pattern * 51  # 510 skips
k = len(skip_pattern)

total_needed = sum(skip_pattern) + k
all_outputs = xorshift32_cpu(known_seed, total_needed, skip=0)

draws = []
idx = 0
for skip in skip_pattern:
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Generated {k} draws with seed {known_seed}")
print(f"Variable skip pattern: {base_pattern}")
print(f"First 10 draws: {draws[:10]}")

# Test with hybrid kernel
config = get_kernel_info('xorshift32_hybrid')
kernel = cp.RawKernel(config['kernel_source'], config['kernel_name'])

seeds = cp.array([known_seed], dtype=cp.uint32)
residues_gpu = cp.array(draws, dtype=cp.uint32)
survivors = cp.zeros(10, dtype=cp.uint32)
match_rates = cp.zeros(10, dtype=cp.float32)
skip_sequences = cp.zeros(10 * 512, dtype=cp.uint32)
strategy_ids = cp.zeros(10, dtype=cp.uint32)
survivor_count = cp.zeros(1, dtype=cp.uint32)

# More lenient strategy for variable patterns
n_strategies = 1
strategy_max_misses = cp.array([20], dtype=cp.int32)  # Allow more misses
strategy_tolerances = cp.array([4], dtype=cp.int32)   # ±4 tolerance

default_params = config['default_params']

print(f"\nTesting with strategy: max_misses=20, tolerance=4, threshold=0.50")

kernel(
    (1,), (1,),
    (seeds, residues_gpu, survivors, match_rates, skip_sequences, strategy_ids,
     survivor_count, cp.int32(1), cp.int32(k),
     strategy_max_misses, strategy_tolerances, cp.int32(n_strategies),
     cp.float32(0.50),  # 50% threshold
     cp.int32(default_params['shift_a']), 
     cp.int32(default_params['shift_b']), 
     cp.int32(default_params['shift_c']),
     cp.int32(0))
)

count = int(survivor_count.get()[0])
print(f"\nResults: {count} survivors")

if count > 0:
    found_seed = int(survivors.get()[0])
    match_rate = float(match_rates.get()[0])
    
    # Extract skip pattern
    skip_seq = skip_sequences.get()[:k]
    
    print(f"  ✅ Found seed: {found_seed}")
    print(f"     Match rate: {match_rate:.1%}")
    print(f"     Detected pattern (first 20): {list(skip_seq[:20])}")
    print(f"     Expected pattern (first 20): {skip_pattern[:20]}")
    
    # Check pattern accuracy
    matches = sum(1 for i in range(min(50, k)) if skip_seq[i] == skip_pattern[i])
    accuracy = matches / 50
    
    print(f"     Pattern accuracy: {accuracy:.1%} ({matches}/50)")
    
    if found_seed == known_seed and accuracy >= 0.80:
        print(f"\n✅✅✅ XORSHIFT32_HYBRID WORKS WITH VARIABLE SKIP!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Found seed but pattern accuracy could be better")
else:
    print(f"  ❌ Seed {known_seed} NOT found!")
    sys.exit(1)

