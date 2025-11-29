#!/usr/bin/env python3
"""Fix the registry entry to use correct CPU reference"""

with open('prng_registry.py', 'r') as f:
    content = f.read()

# Replace mt19937_cpu_simple with mt19937_cpu
content = content.replace("'cpu_reference': mt19937_cpu_simple,", "'cpu_reference': mt19937_cpu,")

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Fixed cpu_reference to use mt19937_cpu")

# Test
try:
    import importlib
    import sys
    
    if 'prng_registry' in sys.modules:
        del sys.modules['prng_registry']
    
    import prng_registry
    
    print("✅ File parses correctly!")
    print(f"✅ Available PRNGs: {prng_registry.list_available_prngs()}")
    
    if 'mt19937_hybrid' in prng_registry.list_available_prngs():
        print("✅ mt19937_hybrid successfully registered!")
        config = prng_registry.get_kernel_info('mt19937_hybrid')
        print(f"   Description: {config['description']}")
        print(f"   Variable skip: {config.get('variable_skip', False)}")
        print(f"   Multi-strategy: {config.get('multi_strategy', False)}")
    else:
        print("❌ mt19937_hybrid not found in registry")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
