import json

def xorshift128_step(x, y, z, w):
    """xorshift128 step function - returns new (x,y,z,w) state"""
    t = x ^ ((x << 11) & 0xFFFFFFFF)
    x = y
    y = z
    z = w
    w = (w ^ (w >> 19) ^ (t ^ (t >> 8))) & 0xFFFFFFFF
    return x, y, z, w

SEED = 1234
SKIP = 5
NUM_DRAWS = 512

# Initialize state
x = SEED
y = 362436069
z = 521288629
w = 88675123

draws = []
for i in range(NUM_DRAWS):
    # Apply skip
    for _ in range(SKIP):
        x, y, z, w = xorshift128_step(x, y, z, w)
    # Generate draw
    x, y, z, w = xorshift128_step(x, y, z, w)
    draws.append({
        'draw': w % 1000,
        'session': 'midday',
        'timestamp': 5000000 + i
    })

with open('test_multi_prng_xorshift128.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_multi_prng_xorshift128.json")
print(f"   Seed: {SEED}, Skip: {SKIP}, Draws: {len(draws)}")
