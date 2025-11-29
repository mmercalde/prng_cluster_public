#!/usr/bin/env python3
"""
Create test dataset using OUR mt19937_cpu implementation
This will match our GPU kernel!
"""

import json
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '.')
from prng_registry import mt19937_cpu

# Use timestamp seed INSIDE the date range
test_date = datetime(2024, 2, 1, 12, 0, 0)
KNOWN_SEED = int(test_date.timestamp())  # 1706817600

KNOWN_SKIP = 5
NUM_DRAWS = 600

print(f"Creating test dataset with OUR MT19937:")
print(f"  Timestamp: {KNOWN_SEED}")
print(f"  Date: {test_date}")
print(f"  Skip: {KNOWN_SKIP}")
print(f"  Draws: {NUM_DRAWS}")

# Generate using OUR mt19937_cpu
total_outputs = NUM_DRAWS * (KNOWN_SKIP + 1)
all_outputs = mt19937_cpu(KNOWN_SEED, total_outputs, skip=0)

# Take every (KNOWN_SKIP+1)th output as the "published" draw
draws = []
base_date = datetime(2024, 1, 1)

for i in range(NUM_DRAWS):
    # Skip outputs, take the next one
    output_index = i * (KNOWN_SKIP + 1) + KNOWN_SKIP
    draw = all_outputs[output_index] % 1000
    
    draws.append({
        'date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
        'session': 'evening',
        'draw': draw
    })

# Save test dataset
with open('test_our_mt19937.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✓ Created test_our_mt19937.json")
print(f"  Seed (timestamp): {KNOWN_SEED}")
print(f"  Skip: {KNOWN_SKIP}")
print(f"  Draws: {len(draws)}")
print(f"  Sample: {[d['draw'] for d in draws[:5]]}")
print(f"\nExpected result: Should find seed {KNOWN_SEED} with skip={KNOWN_SKIP}")
