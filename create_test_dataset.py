#!/usr/bin/env python3
"""
Create a synthetic test dataset with known PRNG seed
This verifies the sieve can find seeds we KNOW are correct
"""

import json
from datetime import datetime, timedelta
import random

# Create test with KNOWN seed and parameters
KNOWN_SEED = 1234567890  # Unix timestamp for Feb 13, 2009
KNOWN_SKIP = 5  # 5 discarded draws between each published draw
NUM_DRAWS = 50  # Small test dataset

print(f"Creating test dataset with KNOWN parameters:")
print(f"  Seed: {KNOWN_SEED}")
print(f"  Skip: {KNOWN_SKIP}")
print(f"  Draws: {NUM_DRAWS}")

# Initialize MT19937 with known seed
random.seed(KNOWN_SEED)

# Generate draws with skip
draws = []
base_date = datetime(2024, 1, 1)

for i in range(NUM_DRAWS):
    # Skip draws (unpublished test draws)
    for _ in range(KNOWN_SKIP):
        _ = random.randint(0, 999)
    
    # Published draw
    draw = random.randint(0, 999)
    date = base_date + timedelta(days=i)
    
    draws.append({
        'date': date.strftime('%Y-%m-%d'),
        'session': 'evening',
        'draw': draw
    })

# Save test dataset
with open('test_known_prng.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✓ Created test_known_prng.json")
print(f"  Contains {len(draws)} draws")
print(f"  Sample: {[d['draw'] for d in draws[:5]]}")
print(f"\nNow run:")
print(f"  python3 timestamp_search.py test_known_prng.json \\")
print(f"    --mode second \\")
print(f"    --window 50 \\")
print(f"    --threshold 0.8 \\")
print(f"    --prngs mt19937 \\")
print(f"    --skip-max 10")
print(f"\nExpected result: Should find seed {KNOWN_SEED} with skip={KNOWN_SKIP}")
