import json

def pcg32_step(state):
    """PCG32 step function"""
    multiplier = 6364136223846793005
    increment = 1442695040888963407
    
    oldstate = state
    state = (oldstate * multiplier + increment) & 0xFFFFFFFFFFFFFFFF
    
    xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & 0xFFFFFFFF
    rot = (oldstate >> 59) & 0xFFFFFFFF
    output = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF
    
    return state, output

SEED = 1234
SKIP = 5
NUM_DRAWS = 512

state = SEED
draws = []
for i in range(NUM_DRAWS):
    # Apply skip
    for _ in range(SKIP):
        state, _ = pcg32_step(state)
    # Generate draw
    state, output = pcg32_step(state)
    draws.append({
        'draw': output % 1000,
        'session': 'midday',
        'timestamp': 5000000 + i
    })

with open('test_multi_prng_pcg32.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_multi_prng_pcg32.json")
print(f"   Seed: {SEED}, Skip: {SKIP}, Draws: {len(draws)}")
