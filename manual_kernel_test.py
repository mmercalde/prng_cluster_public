#!/usr/bin/env python3
"""
Manually test xorshift32_hybrid kernel with explicit parameters
"""
import cupy as cp
from prng_registry import get_kernel_info, xorshift32_cpu

print("="*70)
print("MANUAL KERNEL TEST")
print("="*70)

# Generate test data
known_seed = 54321
skip = 5  # Use CONSTANT skip first (easier)
n_draws = 100

total_outputs = n_draws * (skip + 1)
all_outputs = xorshift32_cpu(known_seed, total_outputs, skip=0)

draws = []
idx = 0
for i in range(n_draws):
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Generated {n_draws} draws, seed={known_seed}, constant skip={skip}")
print(f"First 10: {draws[:10]}")

# Get kernel
config = get_kernel_info('xorshift32_hybrid')
kernel = cp.RawKernel(config['kernel_source'], config['kernel_name'])

print(f"\nKernel: {config['kernel_name']}")
print(f"Default params: {config['default_params']}")

# Setup arrays
n_seeds = 10000
seeds_gpu = cp.arange(50000, 50000 + n_seeds, dtype=cp.uint32)
residues_gpu = cp.array(draws, dtype=cp.uint32)
survivors_gpu = cp.zeros(n_seeds, dtype=cp.uint32)
match_rates_gpu = cp.zeros(n_seeds, dtype=cp.float32)
skip_sequences_gpu = cp.zeros(n_seeds * 512, dtype=cp.uint32)
strategy_ids_gpu = cp.zeros(n_seeds, dtype=cp.uint32)
survivor_count_gpu = cp.zeros(1, dtype=cp.uint32)

# Strategy params (1 simple strategy)
n_strategies = 1
strategy_max_misses = cp.array([20], dtype=cp.int32)
strategy_tolerances = cp.array([5], dtype=cp.int32)

# PRNG params
shift_a = config['default_params']['shift_a']
shift_b = config['default_params']['shift_b']
shift_c = config['default_params']['shift_c']

print(f"\nKernel parameters:")
print(f"  n_seeds: {n_seeds}")
print(f"  k (draws): {n_draws}")
print(f"  n_strategies: {n_strategies}")
print(f"  threshold: 0.5")
print(f"  shift_a: {shift_a}, shift_b: {shift_b}, shift_c: {shift_c}")
print(f"  offset: 0")

# Build kernel_args EXACTLY as kernel expects
kernel_args = [
    seeds_gpu,                    # 1
    residues_gpu,                 # 2
    survivors_gpu,                # 3
    match_rates_gpu,              # 4
    skip_sequences_gpu,           # 5
    strategy_ids_gpu,             # 6
    survivor_count_gpu,           # 7
    cp.int32(n_seeds),           # 8
    cp.int32(n_draws),           # 9
    strategy_max_misses,          # 10
    strategy_tolerances,          # 11
    cp.int32(n_strategies),      # 12
    cp.float32(0.5),             # 13 - threshold
    cp.int32(shift_a),           # 14
    cp.int32(shift_b),           # 15
    cp.int32(shift_c),           # 16
    cp.int32(0)                  # 17 - offset
]

print(f"\nTotal kernel_args: {len(kernel_args)}")
print("Launching kernel...")

try:
    threads = 256
    blocks = (n_seeds + threads - 1) // threads
    kernel((blocks,), (threads,), tuple(kernel_args))
    cp.cuda.Device().synchronize()
    
    count = int(survivor_count_gpu[0].get())
    print(f"\n✅ Kernel executed successfully!")
    print(f"   Survivors: {count}")
    
    if count > 0:
        survivors = survivors_gpu[:count].get()
        if 54321 in survivors:
            print(f"   🎯 FOUND SEED 54321!")
        else:
            print(f"   Found seeds: {survivors[:5]}")
    
except Exception as e:
    print(f"\n❌ Kernel failed: {e}")
    import traceback
    traceback.print_exc()

