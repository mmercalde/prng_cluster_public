import json

def philox4x32_generate(key0, key1, counter):
    """Philox4x32 generation function"""
    PHILOX_M4x32_0 = 0xD2511F53
    PHILOX_M4x32_1 = 0xCD9E8D57
    PHILOX_W32_0 = 0x9E3779B9
    PHILOX_W32_1 = 0xBB67AE85
    
    ctr = [counter, 0, 0, 0]
    k = [key0, key1]
    
    for round in range(10):
        prod0 = ctr[0] * PHILOX_M4x32_0
        prod1 = ctr[2] * PHILOX_M4x32_1
        
        hi0 = (prod0 >> 32) & 0xFFFFFFFF
        lo0 = prod0 & 0xFFFFFFFF
        hi1 = (prod1 >> 32) & 0xFFFFFFFF
        lo1 = prod1 & 0xFFFFFFFF
        
        ctr[0] = (hi1 ^ ctr[1] ^ k[0]) & 0xFFFFFFFF
        ctr[1] = lo1 & 0xFFFFFFFF
        ctr[2] = (hi0 ^ ctr[3] ^ k[1]) & 0xFFFFFFFF
        ctr[3] = lo0 & 0xFFFFFFFF
        
        k[0] = (k[0] + PHILOX_W32_0) & 0xFFFFFFFF
        k[1] = (k[1] + PHILOX_W32_1) & 0xFFFFFFFF
    
    return ctr[0]

SEED = 1234
KEY0 = SEED & 0xFFFFFFFF
KEY1 = (SEED >> 32) & 0xFFFFFFFF
SKIP = 5
NUM_DRAWS = 512

draws = []
counter = SKIP
for i in range(NUM_DRAWS):
    output = philox4x32_generate(KEY0, KEY1, counter)
    draws.append({
        'draw': output % 1000,
        'session': 'midday',
        'timestamp': 5000000 + i
    })
    counter += SKIP + 1

with open('test_multi_prng_philox4x32.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_multi_prng_philox4x32.json")
print(f"   Seed: {SEED} (key0={KEY0}, key1={KEY1}), Skip: {SKIP}, Draws: {len(draws)}")
