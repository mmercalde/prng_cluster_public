#!/usr/bin/env python3
"""Recreate test data correctly with FULL MT19937"""
import json
from datetime import datetime, timedelta
from prng_registry import mt19937_cpu

SEED = 1706817600  # Feb 1, 2024 12:00:00
SKIP = 5
NUM_DRAWS = 600

print(f"Creating test data:")
print(f"  Seed: {SEED}")
print(f"  Skip: {SKIP}")
print(f"  Draws: {NUM_DRAWS}")

# Generate all outputs needed
total_outputs = NUM_DRAWS * (SKIP + 1)
all_outputs = mt19937_cpu(SEED, total_outputs, skip=0)

# Take every (SKIP+1)th output
draws = []
base_date = datetime(2024, 1, 1)

for i in range(NUM_DRAWS):
    idx = i * (SKIP + 1) + SKIP
    draw = all_outputs[idx] % 1000
    draws.append({
        'date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
        'session': 'evening',
        'draw': draw
    })

with open('test_our_mt19937.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_our_mt19937.json")
print(f"  First 10: {[d['draw'] for d in draws[:10]]}")
