import json

def rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF

def xoshiro256pp_step(s0, s1, s2, s3):
    """xoshiro256++ step function"""
    result = (rotl(s0 + s3, 23) + s0) & 0xFFFFFFFFFFFFFFFF
    t = (s1 << 17) & 0xFFFFFFFFFFFFFFFF
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = rotl(s3, 45)
    return s0, s1, s2, s3, result

SEED = 1234
SKIP = 5
NUM_DRAWS = 512

# Initialize state
s0 = SEED
s1 = 0x9E3779B97F4A7C15
s2 = 0x6A09E667F3BCC908
s3 = 0xBB67AE8584CAA73B

draws = []
for i in range(NUM_DRAWS):
    # Apply skip
    for _ in range(SKIP):
        s0, s1, s2, s3, _ = xoshiro256pp_step(s0, s1, s2, s3)
    # Generate draw
    s0, s1, s2, s3, output = xoshiro256pp_step(s0, s1, s2, s3)
    draws.append({
        'draw': output % 1000,
        'session': 'midday',
        'timestamp': 5000000 + i
    })

with open('test_multi_prng_xoshiro256pp.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_multi_prng_xoshiro256pp.json")
print(f"   Seed: {SEED}, Skip: {SKIP}, Draws: {len(draws)}")
