#!/usr/bin/env python3
import os

# PCG32 Hybrid Test Generator
pcg32_code = """#!/usr/bin/env python3
import json

def pcg32_step(state, inc):
    multiplier = 6364136223846793005
    state = (state * multiplier + inc) & 0xFFFFFFFFFFFFFFFF
    return state

def pcg32_output(state):
    xorshifted = ((state >> 18) ^ state) >> 27
    rot = state >> 59
    return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

SEED = 12345
INCREMENT = 1442695040888963407
NUM_DRAWS = 20
skip_pattern = [5, 5, 7, 7, 3, 3] * 4

print(f"Generating PCG32 Hybrid test data...")
print(f"Seed: {SEED}")

state = SEED
draws = []

for i, skip in enumerate(skip_pattern[:NUM_DRAWS]):
    for _ in range(skip):
        state = pcg32_step(state, INCREMENT)
    state = pcg32_step(state, INCREMENT)
    output = pcg32_output(state)
    draw = output % 1000
    draws.append({'draw': draw, 'session': 'midday', 'timestamp': 5000000 + i})
    print(f"  Draw {i+1}: skip={skip}, draw={draw}")

with open('test_pcg32_hybrid.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_pcg32_hybrid.json")
"""

# LCG32 Hybrid Test Generator
lcg32_code = """#!/usr/bin/env python3
import json

def lcg32_step(state, a=1103515245, c=12345, m=0x7FFFFFFF):
    return ((a * state) + c) % m if m > 0 else ((a * state) + c) & 0xFFFFFFFF

SEED = 54321
NUM_DRAWS = 20
skip_pattern = [3, 6, 9] * 7

print(f"Generating LCG32 Hybrid test data...")
print(f"Seed: {SEED}")

state = SEED
draws = []

for i, skip in enumerate(skip_pattern[:NUM_DRAWS]):
    for _ in range(skip):
        state = lcg32_step(state)
    state = lcg32_step(state)
    draw = state % 1000
    draws.append({'draw': draw, 'session': 'evening', 'timestamp': 6000000 + i})
    print(f"  Draw {i+1}: skip={skip}, draw={draw}")

with open('test_lcg32_hybrid.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_lcg32_hybrid.json")
"""

# Xorshift64 Hybrid Test Generator
xorshift64_code = """#!/usr/bin/env python3
import json

def xorshift64_step(state):
    state ^= (state >> 12) & 0xFFFFFFFFFFFFFFFF
    state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 27) & 0xFFFFFFFFFFFFFFFF
    state = (state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
    return state

SEED = 0x123456789ABCDEF0
NUM_DRAWS = 25
skip_pattern = [1, 2, 3, 5, 8, 5, 3, 2] * 4

print(f"Generating Xorshift64 Hybrid test data...")
print(f"Seed: {hex(SEED)}")

state = SEED
draws = []

for i, skip in enumerate(skip_pattern[:NUM_DRAWS]):
    for _ in range(skip):
        state = xorshift64_step(state)
    state = xorshift64_step(state)
    draw = (state & 0xFFFFFFFF) % 1000
    draws.append({'draw': draw, 'session': 'midday', 'timestamp': 7000000 + i})
    print(f"  Draw {i+1}: skip={skip}, draw={draw}")

with open('test_xorshift64_hybrid.json', 'w') as f:
    json.dump(draws, f, indent=2)

print(f"✅ Created test_xorshift64_hybrid.json")
"""

# Write all files
files = {
    'create_pcg32_hybrid_test.py': pcg32_code,
    'create_lcg32_hybrid_test.py': lcg32_code,
    'create_xorshift64_hybrid_test.py': xorshift64_code
}

print("Installing Hybrid Test Generators...")
for fname, code in files.items():
    with open(fname, 'w') as f:
        f.write(code)
    os.chmod(fname, 0o755)
    print(f"✅ {fname}")

print("\nRun: python3 create_pcg32_hybrid_test.py")
