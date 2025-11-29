#!/usr/bin/env python3
"""
Add hybrid metadata flags to xorshift32_hybrid registry entry
"""

with open('prng_registry.py', 'r') as f:
    content = f.read()

# Find xorshift32_hybrid entry and add the flags
old_entry = """    'xorshift32_hybrid': {
        'kernel_source': XORSHIFT32_HYBRID_KERNEL,
        'kernel_name': 'xorshift32_hybrid_multi_strategy_sieve',
        'cpu_reference': xorshift32_cpu,
        'default_params': {
            'shift_a': 13,
            'shift_b': 17,
            'shift_c': 5,
        },
        'description': 'Xorshift32 Hybrid - Variable skip pattern detection',
        'seed_type': 'uint32',
        'state_size': 4,
    },"""

new_entry = """    'xorshift32_hybrid': {
        'kernel_source': XORSHIFT32_HYBRID_KERNEL,
        'kernel_name': 'xorshift32_hybrid_multi_strategy_sieve',
        'cpu_reference': xorshift32_cpu,
        'default_params': {
            'shift_a': 13,
            'shift_b': 17,
            'shift_c': 5,
        },
        'description': 'Xorshift32 Hybrid - Variable skip pattern detection',
        'seed_type': 'uint32',
        'state_size': 4,
        'variable_skip': True,
        'multi_strategy': True,
    },"""

if old_entry in content:
    content = content.replace(old_entry, new_entry)
    print("✅ Added variable_skip and multi_strategy flags to xorshift32_hybrid")
else:
    print("❌ Could not find xorshift32_hybrid entry")
    exit(1)

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Updated prng_registry.py")

