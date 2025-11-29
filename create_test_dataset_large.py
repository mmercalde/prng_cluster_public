#!/usr/bin/env python3
"""Create larger test dataset with timestamp seed IN the date range"""
import json
from datetime import datetime, timedelta
import random

# Use a timestamp that falls WITHIN the dataset's date range
# Dataset spans 2024-01-01 to ~2024-03-15 (600 days)
# Pick a timestamp in the middle: 2024-02-01 12:00:00
test_date = datetime(2024, 2, 1, 12, 0, 0)
KNOWN_SEED = int(test_date.timestamp())  # 1706792400

KNOWN_SKIP = 5
NUM_DRAWS = 600

print(f"Creating large test dataset with TIMESTAMP seed:")
print(f"  Timestamp: {KNOWN_SEED}")
print(f"  Date: {test_date}")
print(f"  Skip: {KNOWN_SKIP}")
print(f"  Draws: {NUM_DRAWS}")

random.seed(KNOWN_SEED)
draws = []
base_date = datetime(2024, 1, 1)

for i in range(NUM_DRAWS):
    for _ in range(KNOWN_SKIP):
        _ = random.randint(0, 999)
    draw = random.randint(0, 999)
    draws.append({
        'date': (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
        'session': 'evening',
        'draw': draw
    })

with open('test_known_large.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✓ Created test_known_large.json")
print(f"  Seed (timestamp): {KNOWN_SEED}")
print(f"  Skip: {KNOWN_SKIP}")
print(f"  Draws: {len(draws)}")
print(f"  Sample: {[d['draw'] for d in draws[:5]]}")
print(f"\nExpected result: Should find seed {KNOWN_SEED} with skip={KNOWN_SKIP}")
