import json

def rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF

def sfc64_step(a, b, c, counter):
    tmp = (a + b + counter) & 0xFFFFFFFFFFFFFFFF
    counter = (counter + 1) & 0xFFFFFFFFFFFFFFFF
    a = (b ^ (b >> 11)) & 0xFFFFFFFFFFFFFFFF
    b = (c + (c << 3)) & 0xFFFFFFFFFFFFFFFF
    c = (rotl(c, 24) + tmp) & 0xFFFFFFFFFFFFFFFF
    return a, b, c, counter, tmp

SEED = 1234
VARIABLE_SKIP = [5, 5, 3, 7, 5, 5, 8, 4]
NUM_DRAWS = 512

a = SEED
b = 0x9E3779B97F4A7C15
c = 0x6A09E667F3BCC908
counter = 1
draws = []

for i in range(NUM_DRAWS):
    skip = VARIABLE_SKIP[i % len(VARIABLE_SKIP)]
    for _ in range(skip):
        a, b, c, counter, _ = sfc64_step(a, b, c, counter)
    a, b, c, counter, output = sfc64_step(a, b, c, counter)
    draws.append({
        'draw': output % 1000,
        'session': 'midday',
        'timestamp': 5000000 + i
    })

with open('test_multi_prng_sfc64_variable.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_multi_prng_sfc64_variable.json")
