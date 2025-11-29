#!/usr/bin/env python3
import json

def lcg32_step(state, a=1103515245, c=12345, m=0x7FFFFFFF):
    return ((a * state) + c) % m if m > 0 else ((a * state) + c) & 0xFFFFFFFF

SEED = 54321
NUM_DRAWS = 512  # Increased to 512
skip_pattern = [3, 6, 9] * (NUM_DRAWS // 3 + 1)
skip_pattern = skip_pattern[:NUM_DRAWS]

print(f"Generating LCG32 Hybrid test data (512 draws)...")
print(f"Seed: {SEED}")

state = SEED
draws = []

for i, skip in enumerate(skip_pattern):
    for _ in range(skip):
        state = lcg32_step(state)
    state = lcg32_step(state)
    draw = state % 1000
    draws.append({'draw': draw, 'session': 'evening', 'timestamp': 6000000 + i})

with open('test_lcg32_hybrid.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_lcg32_hybrid.json with {len(draws)} draws")
