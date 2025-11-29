#!/usr/bin/env python3
"""
Create synthetic dataset with FULL 32-bit state values (not just mod 1000)
This allows the multi-modulo sieve to work correctly.
"""

import json

def xorshift32_step(state):
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF

# Generate 100 draws from seed 42 with NO skip
# Store FULL state values, not just mod 1000
state = 42
draws = []

for i in range(100):
    state = xorshift32_step(state)
    draws.append({
        "date": f"2020-{(i//30)+1:02d}-{(i%30)+1:02d}",
        "session": "midday",
        "draw": state % 1000,
        "full_state": int(state)  # Store full 32-bit state
    })

with open('test_seed42_fullstate.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"Created test_seed42_fullstate.json with {len(draws)} draws")
print(f"First 5 draws (mod 1000): {[d['draw'] for d in draws[:5]]}")
print(f"First 5 full states: {[d['full_state'] for d in draws[:5]]}")
