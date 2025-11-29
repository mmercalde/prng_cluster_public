#!/usr/bin/env python3
import cupy as cp
from prng_registry import get_kernel_info
import json

# Load data
with open('test_seed42_fullstate.json', 'r') as f:
    data = json.load(f)

residues = [entry['full_state'] for entry in data[-30:]]
print(f"Loaded {len(residues)} residues")
print(f"First 3: {residues[:3]}")

# Get kernel
config = get_kernel_info('xorshift32')
kernel = cp.RawKernel(config['kernel_source'], config['kernel_name'])

# Setup arrays
seeds_gpu = cp.array([42], dtype=cp.uint32)
residues_gpu = cp.array(residues, dtype=cp.uint32)
survivors_gpu = cp.zeros(1, dtype=cp.uint32)
match_rates_gpu = cp.zeros(1, dtype=cp.float32)
best_skips_gpu = cp.zeros(1, dtype=cp.uint8)
survivor_count_gpu = cp.zeros(1, dtype=cp.uint32)

# Call kernel
kernel(
    (1,), (1,),
    (seeds_gpu, residues_gpu, survivors_gpu, match_rates_gpu, best_skips_gpu, survivor_count_gpu,
     cp.int32(1), cp.int32(30), cp.int32(0), cp.int32(1), cp.float32(0.9),
     cp.uint32(13), cp.uint32(17), cp.uint32(5), cp.int32(0))
)

count = int(survivor_count_gpu.get())
print(f"\nSurvivors found: {count}")
if count > 0:
    print(f"Seed: {int(survivors_gpu[0].get())}")
    print(f"Match rate: {float(match_rates_gpu[0].get())}")
    print(f"Best skip: {int(best_skips_gpu[0].get())}")
else:
    print(f"Match rate for seed 42: {float(match_rates_gpu[0].get())}")
