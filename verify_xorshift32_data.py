#!/usr/bin/env python3
"""
Verify xorshift32 test data is correct
"""
from prng_registry import xorshift32_cpu
import json

# Load the test data
with open('test_xorshift32_const_skip5.json', 'r') as f:
    test_data = json.load(f)

draws = [d['draw'] for d in test_data]
print(f"Test data has {len(draws)} draws")
print(f"First 10: {draws[:10]}")

# Regenerate with seed 12345, skip 5
seed = 12345
skip = 5
n_draws = 100

all_outputs = xorshift32_cpu(seed, n_draws * (skip + 1), skip=0)
expected_draws = []
idx = 0
for i in range(n_draws):
    idx += skip
    expected_draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"\nExpected from xorshift32_cpu:")
print(f"First 10: {expected_draws[:10]}")

# Compare
matches = sum(1 for i in range(len(draws)) if draws[i] == expected_draws[i])
print(f"\nMatch: {matches}/{len(draws)} draws match")

if matches == len(draws):
    print("✅ Test data is correct!")
else:
    print("❌ Test data is WRONG!")
    print(f"\nFirst mismatch:")
    for i in range(min(20, len(draws))):
        if draws[i] != expected_draws[i]:
            print(f"  Index {i}: got {draws[i]}, expected {expected_draws[i]}")
            break

# Now test if the sieve SHOULD find it
print(f"\n{'='*70}")
print("Testing with GPUSieve directly...")
print(f"{'='*70}")

import cupy as cp
cp.cuda.Device(0).use()

from sieve_filter import GPUSieve

sieve = GPUSieve(gpu_id=0)

result = sieve.run_sieve(
    prng_family='xorshift32',
    seed_start=10000,
    seed_end=15000,
    residues=draws,
    skip_range=(0, 10),
    min_match_threshold=0.01,
    offset=0
)

survivors = result.get('survivors', [])
print(f"Survivors: {len(survivors)}")

match = [s for s in survivors if s['seed'] == 12345]
if match:
    print(f"✅ FOUND seed 12345 in direct test!")
    print(f"   Match rate: {match[0].get('match_rate', 0):.1%}")
    print(f"   Skip: {match[0].get('best_skip')}")
else:
    print(f"❌ Seed 12345 NOT found in direct test")
    if survivors:
        print(f"   Sample found: {[s['seed'] for s in survivors[:5]]}")

