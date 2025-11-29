#!/usr/bin/env python3
"""
Validate xorshift32 on local GPU only
"""
import cupy as cp
from prng_registry import get_kernel_info, xorshift32_cpu
import sys

print("="*70)
print("XORSHIFT32 - LOCAL GPU TEST")
print("="*70)

# Generate constant skip test data
known_seed = 12345
skip = 5
n_draws = 512

total_outputs = n_draws * (skip + 1)
all_outputs = xorshift32_cpu(known_seed, total_outputs, skip=0)

draws = []
idx = 0
for i in range(n_draws):
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Generated {n_draws} draws with seed {known_seed}, skip {skip}")
print(f"First 10: {draws[:10]}")

# Test on GPU
config = get_kernel_info('xorshift32')
kernel = cp.RawKernel(config['kernel_source'], config['kernel_name'])

seeds = cp.array([known_seed], dtype=cp.uint32)
residues_gpu = cp.array(draws, dtype=cp.uint32)
survivors = cp.zeros(10, dtype=cp.uint32)
match_rates = cp.zeros(10, dtype=cp.float32)
best_skips = cp.zeros(10, dtype=cp.uint8)
survivor_count = cp.zeros(1, dtype=cp.uint32)

# xorshift32 needs 3 shift parameters
default_params = config['default_params']
shift_a = default_params['shift_a']
shift_b = default_params['shift_b']
shift_c = default_params['shift_c']

print(f"\nUsing shift params: a={shift_a}, b={shift_b}, c={shift_c}")
print("Testing on local GPU...")

kernel(
    (1,), (1,),
    (seeds, residues_gpu, survivors, match_rates, best_skips, survivor_count,
     cp.int32(1), cp.int32(n_draws), cp.int32(5), cp.int32(5), 
     cp.float32(0.01),
     cp.int32(shift_a), cp.int32(shift_b), cp.int32(shift_c),  # ADD THESE
     cp.int32(0))  # offset
)

count = int(survivor_count.get()[0])
print(f"\nResults:")
print(f"  Survivors found: {count}")

if count > 0:
    found_seed = int(survivors.get()[0])
    match_rate = float(match_rates.get()[0])
    found_skip = int(best_skips.get()[0])
    
    print(f"  ✅ Found seed: {found_seed}")
    print(f"     Match rate: {match_rate:.1%}")
    print(f"     Best skip: {found_skip}")
    
    if found_seed == known_seed and found_skip == skip:
        print(f"\n✅✅✅ XORSHIFT32 WORKS LOCALLY!")
        sys.exit(0)
    else:
        print(f"\n❌ Mismatch!")
        sys.exit(1)
else:
    print(f"  ❌ Seed {known_seed} NOT found!")
    sys.exit(1)

