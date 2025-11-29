import json

def xorshift32_step(state):
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF

SEED = 12345
SKIP = 5
NUM_DRAWS = 600

state = SEED
draws = []
for i in range(NUM_DRAWS):
    for _ in range(SKIP):
        state = xorshift32_step(state)
    state = xorshift32_step(state)
    draws.append({
        'draw': state % 1000,
        'session': 'midday',
        'timestamp': 5000000 + i
    })

with open('test_multi_prng_xorshift32.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Test created: seed {SEED}, skip {SKIP}, {len(draws)} draws")
