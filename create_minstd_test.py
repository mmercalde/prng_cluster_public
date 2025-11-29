import json

def minstd_step(state):
    """MINSTD step function"""
    a = 48271
    m = 2147483647
    
    if state == 0:
        state = 1
    
    temp = (a * state) % m
    return temp

SEED = 1234
SKIP = 5
NUM_DRAWS = 512

state = SEED % 2147483647
if state == 0:
    state = 1

draws = []
for i in range(NUM_DRAWS):
    # Apply skip
    for _ in range(SKIP):
        state = minstd_step(state)
    # Generate draw
    state = minstd_step(state)
    draws.append({
        'draw': state % 1000,
        'session': 'midday',
        'timestamp': 5000000 + i
    })

with open('test_multi_prng_minstd.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_multi_prng_minstd.json")
print(f"   Seed: {SEED}, Skip: {SKIP}, Draws: {len(draws)}")
