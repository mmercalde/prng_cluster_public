#!/usr/bin/env python3
"""
Test direct with 670 draws (same as coordinator)
"""
import sys
sys.path.insert(0, '.')
import cupy as cp
cp.cuda.Device(0).use()

from sieve_filter import GPUSieve
from prng_registry import xorshift32_cpu

# Generate SAME data as coordinator test
known_seed = 54321
base_pattern = [5,5,3,7,5,5,8,4,5,5]
skip_pattern = base_pattern * 67  # 670 skips
k = len(skip_pattern)

total_needed = sum(skip_pattern) + k
all_outputs = xorshift32_cpu(known_seed, total_needed, skip=0)

draws = []
idx = 0
for skip in skip_pattern:
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Testing with {len(draws)} draws (VARIABLE SKIP)")
print(f"First 10: {draws[:10]}")

strategies = [{
    'name': 'Test',
    'max_consecutive_misses': 20,
    'skip_tolerance': 5,
    'enable_reseed_search': False,
    'skip_learning_rate': 0.3,
    'breakpoint_threshold': 0.5
}]

sieve = GPUSieve(gpu_id=0)

print("\nCalling run_hybrid_sieve with 670 draws...")
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
            print(f"   Match rate: {s.get('match_rate', 0):.1%}")
            
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

