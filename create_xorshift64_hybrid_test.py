#!/usr/bin/env python3
import json

def xorshift64_step(state):
    state ^= (state >> 12) & 0xFFFFFFFFFFFFFFFF
    state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 27) & 0xFFFFFFFFFFFFFFFF
    state = (state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
    return state

SEED = 0x123456789ABCDEF0
NUM_DRAWS = 512  # Increased to 512
skip_pattern = [1, 2, 3, 5, 8, 5, 3, 2] * (NUM_DRAWS // 8 + 1)
skip_pattern = skip_pattern[:NUM_DRAWS]

print(f"Generating Xorshift64 Hybrid test data (512 draws)...")
print(f"Seed: {hex(SEED)}")

state = SEED
draws = []

for i, skip in enumerate(skip_pattern):
    for _ in range(skip):
        state = xorshift64_step(state)
    state = xorshift64_step(state)
    draw = (state & 0xFFFFFFFF) % 1000
    draws.append({'draw': draw, 'session': 'midday', 'timestamp': 7000000 + i})

with open('test_xorshift64_hybrid.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_xorshift64_hybrid.json with {len(draws)} draws")
