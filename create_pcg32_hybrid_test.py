#!/usr/bin/env python3
import json

def pcg32_step(state, inc):
    multiplier = 6364136223846793005
    state = (state * multiplier + inc) & 0xFFFFFFFFFFFFFFFF
    return state

def pcg32_output(state):
    xorshifted = ((state >> 18) ^ state) >> 27
    rot = state >> 59
    return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

SEED = 1234
INCREMENT = 1442695040888963407
NUM_DRAWS = 512  # Increased to 512

# Variable skip pattern: [5, 5, 7, 7, 3, 3] repeating
skip_pattern = [5, 5, 7, 7, 3, 3] * (NUM_DRAWS // 6 + 1)
skip_pattern = skip_pattern[:NUM_DRAWS]

print(f"Generating PCG32 Hybrid test data (512 draws)...")
print(f"Seed: {SEED}")

state = SEED
draws = []

for i, skip in enumerate(skip_pattern):
    for _ in range(skip):
        state = pcg32_step(state, INCREMENT)
    state = pcg32_step(state, INCREMENT)
    output = pcg32_output(state)
    draw = output % 1000
    draws.append({'draw': draw, 'session': 'midday', 'timestamp': 5000000 + i})

with open('test_pcg32_hybrid.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_pcg32_hybrid.json with {len(draws)} draws")
