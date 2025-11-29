#!/usr/bin/env python3
"""Add mt19937_hybrid to KERNEL_REGISTRY"""

with open('prng_registry.py', 'r') as f:
    lines = f.readlines()

# Find line with 'xorshift64': (should be around line 646)
insert_before = None
for i, line in enumerate(lines):
    if "'xorshift64':" in line:
        insert_before = i
        break

if not insert_before:
    print("❌ Could not find xorshift64")
    exit(1)

print(f"✅ Inserting before line {insert_before + 1}")

registry_entry = """    'mt19937_hybrid': {
        'kernel_source': MT19937_HYBRID_KERNEL,
        'kernel_name': 'mt19937_hybrid_multi_strategy_sieve',
        'cpu_reference': mt19937_cpu,
        'default_params': {},
        'description': 'MT19937 with hybrid variable skip detection (multi-strategy)',
        'seed_type': 'uint32',
        'state_size': 2496,
        'variable_skip': True,
        'multi_strategy': True,
    },
"""

new_lines = lines[:insert_before] + [registry_entry] + lines[insert_before:]

with open('prng_registry.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Added to registry")

import importlib, sys
if 'prng_registry' in sys.modules:
    del sys.modules['prng_registry']
import prng_registry
print(f"✅ Available PRNGs: {prng_registry.list_available_prngs()}")
if 'mt19937_hybrid' in prng_registry.list_available_prngs():
    print("✅ mt19937_hybrid registered!")
