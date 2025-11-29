#!/usr/bin/env python3
from prng_registry import mt19937_cpu
import json

# Generate what SHOULD be in the file
known_seed = 12345
k = 2000
total_needed = k * (constant_skip := 5) + k

all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)

expected_draws = []
idx = 0
for i in range(k):
    idx += constant_skip
    expected_draws.append(all_outputs[idx] % 1000)
    idx += 1

# Load file
with open('test_26gpu_large.json', 'r') as f:
    file_draws = [d['draw'] for d in json.load(f)]

# Check ALL 2000 draws
matches = sum(1 for i in range(len(file_draws)) if expected_draws[i] == file_draws[i])
print(f"Total matches: {matches}/{len(file_draws)}")

# Check first 512 specifically (the window)
matches_512 = sum(1 for i in range(512) if expected_draws[i] == file_draws[i])
print(f"First 512 (window): {matches_512}/512")

if matches == 2000 and matches_512 == 512:
    print("\n✅ File is PERFECT - all 2000 draws match seed 12345")
    print("❌ So why isn't the sieve finding it?")
    print("\nPossible issues:")
    print("1. Sieve is using wrong skip range (checking 0-10 instead of exactly 5)")
    print("2. Threshold too high even at 1%")
    print("3. Offset mismatch")
