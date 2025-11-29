#!/usr/bin/env python3
"""
Call sieve_filter.py directly (not through coordinator) to see debug output
"""
import sys
sys.path.insert(0, '.')
import cupy as cp
cp.cuda.Device(0).use()

from sieve_filter import GPUSieve
from prng_registry import xorshift32_cpu

# Generate test data - use 100 draws like manual test (which works)
known_seed = 54321
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

print(f"Testing with {len(draws)} draws, seed in range 50000-60000")

# Create a simple strategy
strategies = [{
    'name': 'Test',
    'max_consecutive_misses': 20,
    'skip_tolerance': 5,
    'enable_reseed_search': False,
    'skip_learning_rate': 0.3,
    'breakpoint_threshold': 0.5
}]

sieve = GPUSieve(gpu_id=0)

print("\nCalling run_hybrid_sieve...")
try:
    result = sieve.run_hybrid_sieve(
        prng_family='xorshift32_hybrid',
        seed_start=50000,
        seed_end=60000,
        residues=draws,
        strategies=strategies,
        min_match_threshold=0.5,
        chunk_size=10000,
        offset=0
    )
    
    print(f"\n✅ Success!")
    print(f"Survivors: {len(result.get('survivors', []))}")
    
    for s in result.get('survivors', []):
        if s['seed'] == 54321:
            print(f"🎯 FOUND {s['seed']}!")
            
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

