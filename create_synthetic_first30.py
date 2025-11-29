#!/usr/bin/env python3
import json

def xorshift32_step(state):
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF

# Generate ONLY 30 draws from seed 42
state = 42
draws = []
for i in range(30):
    state = xorshift32_step(state)
    draws.append({
        "date": f"2020-01-{i+1:02d}",
        "session": "midday",
        "draw": state % 1000,
        "full_state": int(state)
    })

with open('test_seed42_first30.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"Created test_seed42_first30.json with 30 draws")
print(f"Draws (mod 1000): {[d['draw'] for d in draws[:10]]}")
