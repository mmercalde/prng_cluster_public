import json

def xorshift64_step(state):
    """xorshift64 step function"""
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
    return x

SEED = 1234
SKIP = 5
NUM_DRAWS = 512

state = SEED
draws = []
for i in range(NUM_DRAWS):
    for _ in range(SKIP):
        state = xorshift64_step(state)
    state = xorshift64_step(state)
    draws.append({
        'draw': state % 1000,
        'session': 'midday',
        'timestamp': 5000000 + i
    })

with open('test_multi_prng_xorshift64.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_multi_prng_xorshift64.json")
print(f"   Seed: {SEED}, Skip: {SKIP}, Draws: {len(draws)}")
