import json

def java_lcg_step(state):
    """Java LCG step function"""
    a = 25214903917
    c = 11
    m = 0xFFFFFFFFFFFF  # 48-bit mask
    
    state = (a * state + c) & m
    output = (state >> 16) & 0xFFFFFFFF
    
    return state, output

SEED = 1234
SKIP = 5
NUM_DRAWS = 512

state = SEED & 0xFFFFFFFFFFFF
draws = []
for i in range(NUM_DRAWS):
    # Apply skip
    for _ in range(SKIP):
        state, _ = java_lcg_step(state)
    # Generate draw
    state, output = java_lcg_step(state)
    draws.append({
        'draw': output % 1000,
        'session': 'midday',
        'timestamp': 5000000 + i
    })

with open('test_multi_prng_java_lcg.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_multi_prng_java_lcg.json")
print(f"   Seed: {SEED}, Skip: {SKIP}, Draws: {len(draws)}")
