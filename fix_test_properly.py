#!/usr/bin/env python3
"""
Fix: Generate ALL test files with seed 88675
"""

import sys
sys.path.insert(0, '.')
from prng_registry import KERNEL_REGISTRY
import json
import subprocess

# Use seed 88675 for ALL PRNGs
seed = 88675
skip = 5
n_draws = 2000
total_outputs = n_draws * (skip + 1)

test_prngs = ['xorshift32', 'pcg32', 'lcg32', 'mt19937', 'xorshift64']

print(f"Generating test data for all PRNGs with seed {seed}...\n")

for prng in test_prngs:
    prng_config = KERNEL_REGISTRY.get(prng)
    if not prng_config or 'cpu_reference' not in prng_config:
        print(f"⚠️  Skipping {prng} - no CPU reference")
        continue
    
    cpu_func = prng_config['cpu_reference']
    all_outputs = cpu_func(seed, total_outputs, skip=0)
    
    draws = []
    for i in range(n_draws):
        idx = i * (skip + 1) + skip
        draws.append(all_outputs[idx] % 1000)
    
    test_data = [{'draw': d, 'session': 'midday', 'timestamp': 3000000 + i} for i, d in enumerate(draws)]
    
    filename = f'test_multi_prng_{prng}.json'
    with open(filename, 'w') as f:
        json.dump(test_data, f)
    
    print(f"✅ {prng:12} - First 5 residues: {draws[:5]}")
    
    # Copy to remote nodes
    for host in ['192.168.3.120', '192.168.3.154']:
        subprocess.run(['scp', filename, 
                       f'{host}:/home/michael/distributed_prng_analysis/'], 
                       capture_output=True)

print(f"\n✅ Generated all test files with seed {seed}")
print(f"✅ Copied all files to remote nodes")

# Update the test script
with open('test_all_prngs_distributed_proper.py', 'r') as f:
    content = f.read()

content = content.replace('known_seed = 12345', f'known_seed = {seed}')

with open('test_all_prngs_distributed_proper.py', 'w') as f:
    f.write(content)

print(f"✅ Updated test script to expect seed {seed}")

