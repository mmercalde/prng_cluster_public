#!/usr/bin/env python3
"""
Fix: Use a proper 64-bit seed for xorshift64 that's in our test range
"""

import sys
sys.path.insert(0, '.')
from prng_registry import xorshift64_cpu
import json
import subprocess

# Use a seed that's definitely in our 0-100,000 range
# AND is a "good" xorshift64 seed
seed = 88675  # A nice 5-digit number well within range
skip = 5
n_draws = 2000
total_outputs = n_draws * (skip + 1)

print(f"Using seed {seed} for xorshift64 test")

# Generate test data
all_outputs = xorshift64_cpu(seed, total_outputs, skip=0)

draws = []
for i in range(n_draws):
    idx = i * (skip + 1) + skip
    draws.append(all_outputs[idx] % 1000)

test_data = [{'draw': d, 'session': 'midday', 'timestamp': 3000000 + i} for i, d in enumerate(draws)]

# Save test file
with open('test_multi_prng_xorshift64.json', 'w') as f:
    json.dump(test_data, f)

print(f"✅ Generated test data with seed {seed}")
print(f"   First 10 residues: {draws[:10]}")

# Copy to remote nodes
for host in ['192.168.3.120', '192.168.3.154']:
    subprocess.run(['scp', 'test_multi_prng_xorshift64.json', 
                   f'{host}:/home/michael/distributed_prng_analysis/'], 
                   capture_output=True)
    print(f"   Copied to {host}")

# Update the test script to use this seed
with open('test_all_prngs_distributed_proper.py', 'r') as f:
    content = f.read()

# Replace the seed value
content = content.replace('known_seed = 12345', f'known_seed = {seed}')

with open('test_all_prngs_distributed_proper.py', 'w') as f:
    f.write(content)

print(f"\n✅ Updated test script to use seed {seed}")

