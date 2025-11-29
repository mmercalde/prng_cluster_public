#!/usr/bin/env python3
"""
Fix the mt19937_hybrid entry and properly add xorshift32_hybrid
"""

with open('prng_registry.py', 'r') as f:
    content = f.read()

# Find and fix the mt19937_hybrid entry
# It should end with }, not just 'default_params': {},
old_mt19937 = """    'mt19937_hybrid': {
        'kernel_source': MT19937_HYBRID_KERNEL,
        'kernel_name': 'mt19937_hybrid_multi_strategy_sieve',
        'cpu_reference': mt19937_cpu,
        'default_params': {},
    'xorshift32_hybrid': {"""

new_mt19937 = """    'mt19937_hybrid': {
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
    'xorshift32_hybrid': {"""

content = content.replace(old_mt19937, new_mt19937)

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Fixed mt19937_hybrid entry")
print("✅ xorshift32_hybrid should now be properly in the registry")

