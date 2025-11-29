#!/usr/bin/env python3
"""Create test dataset with VARIABLE skip pattern using FULL MT19937"""
import json
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '.')
from prng_registry import mt19937_cpu

# Use timestamp seed
test_date = datetime(2024, 2, 1, 12, 0, 0)
KNOWN_SEED = int(test_date.timestamp())  # 1706817600
NUM_DRAWS = 512

# Variable skip pattern: [5, 7, 3, 6, 5, 8, 4, 9, 5, 6...]
SKIP_PATTERN = [5, 7, 3, 6, 5, 8, 4, 9, 5, 6] * 52  # Repeat to cover 512 draws

print(f"Creating VARIABLE SKIP test dataset:")
print(f"  Seed: {KNOWN_SEED} (timestamp)")
print(f"  Skip pattern: {SKIP_PATTERN[:10]}... (repeating)")
print(f"  Draws: {NUM_DRAWS}")

# Generate enough outputs
max_skip = max(SKIP_PATTERN)
total_needed = sum(SKIP_PATTERN[:NUM_DRAWS]) + NUM_DRAWS
all_outputs = mt19937_cpu(KNOWN_SEED, total_needed, skip=0)

# Apply variable skip
draws = []
base_date = datetime(2024, 1, 1)
output_idx = 0

for draw_idx in range(NUM_DRAWS):
    skip = SKIP_PATTERN[draw_idx % len(SKIP_PATTERN)]
    output_idx += skip  # Skip outputs
    draw = all_outputs[output_idx] % 1000
    output_idx += 1  # Move to next
    
    draws.append({
        'date': (base_date + timedelta(days=draw_idx)).strftime('%Y-%m-%d'),
        'session': 'evening',
        'draw': draw
    })

# Save
with open('test_variable_skip_512.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_variable_skip_512.json")
print(f"  Seed: {KNOWN_SEED}")
print(f"  Pattern: Variable {SKIP_PATTERN[:10]}... (repeating)")
print(f"  Draws: {len(draws)}")
print(f"  Sample: {[d['draw'] for d in draws[:5]]}")
print(f"\nExpected: Hybrid mode should find seed {KNOWN_SEED}")
