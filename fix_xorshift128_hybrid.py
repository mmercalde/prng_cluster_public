with open('prng_registry.py', 'r') as f:
    content = f.read()

old = """    'xorshift128_hybrid': {
        'kernel_source': XORSHIFT128_HYBRID_KERNEL,
        'kernel_name': 'xorshift128_hybrid_multi_strategy_sieve',
        'cpu_reference': xorshift128_cpu,
        'default_params': {},
        'description': 'Xorshift128 Hybrid - Variable skip pattern detection',
        'seed_type': 'uint32',
        'state_size': 16,
        'variable_skip': True,
    },"""

new = """    'xorshift128_hybrid': {
        'kernel_source': XORSHIFT128_HYBRID_KERNEL,
        'kernel_name': 'xorshift128_hybrid_multi_strategy_sieve',
        'cpu_reference': xorshift128_cpu,
        'default_params': {},
        'description': 'Xorshift128 Hybrid - Variable skip pattern detection',
        'seed_type': 'uint32',
        'state_size': 16,
        'variable_skip': True,
        'multi_strategy': True,
    },"""

if old in content:
    content = content.replace(old, new)
    with open('prng_registry.py', 'w') as f:
        f.write(content)
    print("✅ Fixed xorshift128_hybrid - added multi_strategy flag")
else:
    print("❌ Pattern not found")
