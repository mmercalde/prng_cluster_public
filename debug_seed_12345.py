#!/usr/bin/env python3
from prng_registry import mt19937_cpu
import json

# What we generated for the file
known_seed = 12345
constant_skip = 5
k = 2000

total_needed = k * (constant_skip + 1)
all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)

draws_file = []
idx = 0
for i in range(k):
    idx += constant_skip
    draws_file.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Generated {len(draws_file)} draws for the file")
print(f"First 10: {draws_file[:10]}")

# What the sieve will generate when testing seed 12345
sieve_outputs = mt19937_cpu(12345, 600, skip=0)
sieve_draws = []
idx = 0
for i in range(100):
    idx += 5
    if idx < len(sieve_outputs):
        sieve_draws.append(sieve_outputs[idx] % 1000)
    idx += 1

print(f"\nWhat sieve generates for seed 12345:")
print(f"First 10: {sieve_draws[:10]}")

# Load actual file
with open('test_26gpu_large.json', 'r') as f:
    file_data = json.load(f)
    file_draws = [d['draw'] for d in file_data]

print(f"\nActual file:")
print(f"Total draws: {len(file_draws)}")
print(f"First 10: {file_draws[:10]}")

# Check matches in first 512 (window_size)
matches = sum(1 for i in range(min(512, len(file_draws), len(draws_file))) 
              if draws_file[i] == file_draws[i])
print(f"\nFile vs generated: {matches}/512 matches")

# Check if sieve output matches file
sieve_matches = sum(1 for i in range(min(100, len(file_draws))) 
                    if i < len(sieve_draws) and sieve_draws[i] == file_draws[i])
print(f"Sieve output vs file: {sieve_matches}/100 matches")

if sieve_matches == 100:
    print("\n✅ Sieve should find seed 12345 with 100% match!")
    print(f"   But with window_size=512, it needs to match 512 draws")
    print(f"   We only generated 100 matching draws, rest are mismatches!")
    print("\n❌ BUG: File has 2000 draws but only first 100 match seed 12345!")
