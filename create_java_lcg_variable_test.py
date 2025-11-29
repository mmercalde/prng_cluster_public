import json

def java_lcg_step(state):
    a = 25214903917
    c = 11
    m = 0xFFFFFFFFFFFF
    state = (a * state + c) & m
    output = (state >> 16) & 0xFFFFFFFF
    return state, output

SEED = 1234
VARIABLE_SKIP = [5, 5, 3, 7, 5, 5, 8, 4]
NUM_DRAWS = 512

state = SEED & 0xFFFFFFFFFFFF
draws = []

for i in range(NUM_DRAWS):
    skip = VARIABLE_SKIP[i % len(VARIABLE_SKIP)]
    for _ in range(skip):
        state, _ = java_lcg_step(state)
    state, output = java_lcg_step(state)
    draws.append({
        'draw': output % 1000,
        'session': 'midday',
        'timestamp': 5000000 + i
    })

with open('test_multi_prng_java_lcg_variable.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_multi_prng_java_lcg_variable.json")
