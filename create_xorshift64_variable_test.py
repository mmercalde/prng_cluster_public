import json

def xorshift64_step(state):
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
    return x

SEED = 1234
VARIABLE_SKIP = [5, 5, 3, 7, 5, 5, 8, 4]  # Variable pattern
NUM_DRAWS = 512

state = SEED
draws = []

for i in range(NUM_DRAWS):
    # Apply variable skip
    skip = VARIABLE_SKIP[i % len(VARIABLE_SKIP)]
    for _ in range(skip):
        state = xorshift64_step(state)
    # Generate draw
    state = xorshift64_step(state)
    draws.append({
        'draw': state % 1000,
        'session': 'midday',
        'timestamp': 5000000 + i
    })

with open('test_multi_prng_xorshift64_variable.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_multi_prng_xorshift64_variable.json")
print(f"   Seed: {SEED}, Variable skip pattern: {VARIABLE_SKIP}")
print(f"   Draws: {len(draws)}")
