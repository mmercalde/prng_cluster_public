#!/usr/bin/env python3
from prng_registry import mt19937_cpu

known_seed = 12345
constant_skip = 5

# Generate what the sieve will test
outputs = mt19937_cpu(known_seed, 300, skip=0)

# What skip=5 produces
draws_skip5 = []
idx = 0
for i in range(100):
    idx += 5
    draws_skip5.append(outputs[idx] % 1000)
    idx += 1

print("First 20 draws with skip=5:")
print(draws_skip5[:20])

# Now check what the file has
import json
with open('test_26gpu_align.json', 'r') as f:
    file_draws = [d['draw'] for d in json.load(f)]

print(f"\nFirst 20 draws in file:")
print(file_draws[:20])

# Check if they match
matches = sum(1 for i in range(min(100, len(file_draws))) if draws_skip5[i] == file_draws[i])
print(f"\nMatches in first 100: {matches}/100 = {matches/100:.1%}")

# What about if we use window_size=250?
matches_250 = sum(1 for i in range(min(250, len(file_draws))) if i < len(draws_skip5) and draws_skip5[i] == file_draws[i])
print(f"Matches in first 250: {matches_250}/250 = {matches_250/250:.1%}")

if matches_250 == 100:
    print("\n✅ File matches perfectly for first 100 draws")
    print("⚠️ But draws 101-250 don't exist in our generated sequence!")
    print("   The sieve is looking at draws we didn't generate correctly")
