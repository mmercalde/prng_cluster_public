#!/usr/bin/env python3
"""Add mt19937_hybrid to KERNEL_REGISTRY - simple line insertion"""

with open('prng_registry.py', 'r') as f:
    lines = f.readlines()

# Find line with 'xorshift64': (it's at line 646)
insert_before = 646  # Line number for 'xorshift64':

registry_entry = """    'mt19937_hybrid': {
        'kernel_source': MT19937_HYBRID_KERNEL,
        'kernel_name': 'mt19937_hybrid_multi_strategy_sieve',
        'cpu_reference': mt19937_cpu_simple,
        'default_params': {},
        'description': 'MT19937 with hybrid variable skip detection (multi-strategy)',
        'seed_type': 'uint32',
        'state_size': 2496,
        'variable_skip': True,
        'multi_strategy': True,
    },
"""

# Insert before xorshift64
new_lines = lines[:insert_before-1] + [registry_entry] + lines[insert_before-1:]

with open('prng_registry.py', 'w') as f:
    f.writelines(new_lines)

print(f"✅ Inserted mt19937_hybrid entry before line {insert_before}")

# Test
try:
    import importlib
    import sys
    
    # Clear any cached imports
    if 'prng_registry' in sys.modules:
        del sys.modules['prng_registry']
    
    import prng_registry
    
    print("✅ File parses correctly!")
    print(f"✅ Available PRNGs: {prng_registry.list_available_prngs()}")
    
    if 'mt19937_hybrid' in prng_registry.list_available_prngs():
        print("✅ mt19937_hybrid successfully registered!")
    else:
        print("❌ mt19937_hybrid not found in registry")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
