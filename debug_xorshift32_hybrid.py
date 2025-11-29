#!/usr/bin/env python3
"""
Debug xorshift32_hybrid kernel
"""
import cupy as cp
from prng_registry import get_kernel_info, xorshift32_cpu

print("="*70)
print("DEBUGGING XORSHIFT32_HYBRID")
print("="*70)

# Generate SIMPLE constant skip test first (easier to debug)
known_seed = 12345
skip = 5
n_draws = 100

total_outputs = n_draws * (skip + 1)
all_outputs = xorshift32_cpu(known_seed, total_outputs, skip=0)

draws = []
idx = 0
for i in range(n_draws):
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Testing CONSTANT skip first: seed={known_seed}, skip={skip}")
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

# Very lenient strategy
n_strategies = 1
strategy_max_misses = cp.array([50], dtype=cp.int32)  # Allow many misses
strategy_tolerances = cp.array([5], dtype=cp.int32)   # Wide tolerance

default_params = config['default_params']

print(f"\nTrying with very lenient parameters...")
print(f"  max_misses=50, tolerance=5, threshold=0.01")

kernel(
    (1,), (1,),
    (seeds, residues_gpu, survivors, match_rates, skip_sequences, strategy_ids,
     survivor_count, cp.int32(1), cp.int32(n_draws),
     strategy_max_misses, strategy_tolerances, cp.int32(n_strategies),
     cp.float32(0.01),  # Very low threshold
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
    print(f"  ✅ Found seed: {found_seed}, match_rate: {match_rate:.1%}")
    
    # Check pattern
    skip_seq = skip_sequences.get()[:20]
    print(f"  Detected pattern: {skip_seq[:10]}")
else:
    print("  ❌ Still not found!")
    print("\nLet's check if regular xorshift32 flexible sieve works...")
    
    # Try the regular flexible sieve for comparison
    config2 = get_kernel_info('xorshift32')
    kernel2 = cp.RawKernel(config2['kernel_source'], config2['kernel_name'])
    
    survivors2 = cp.zeros(10, dtype=cp.uint32)
    match_rates2 = cp.zeros(10, dtype=cp.float32)
    best_skips2 = cp.zeros(10, dtype=cp.uint8)
    survivor_count2 = cp.zeros(1, dtype=cp.uint32)
    
    kernel2(
        (1,), (1,),
        (seeds, residues_gpu, survivors2, match_rates2, best_skips2, survivor_count2,
         cp.int32(1), cp.int32(n_draws), cp.int32(0), cp.int32(10), 
         cp.float32(0.01),
         cp.int32(default_params['shift_a']), 
         cp.int32(default_params['shift_b']), 
         cp.int32(default_params['shift_c']),
         cp.int32(0))
    )
    
    count2 = int(survivor_count2.get()[0])
    if count2 > 0:
        print(f"  ✅ Regular flexible sieve WORKS: found {count2} survivors")
        print(f"     This means the hybrid kernel has a bug!")
    else:
        print(f"  ❌ Regular flexible sieve also fails!")
        print(f"     Something wrong with xorshift32 implementation")

