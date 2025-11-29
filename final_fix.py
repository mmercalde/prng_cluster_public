#!/usr/bin/env python3
"""Final fix: remove config from run_hybrid_sieve call"""

with open('sieve_filter.py', 'r') as f:
    content = f.read()

# Find and replace the run_hybrid_sieve call
# Remove config, restore prng_family
content = content.replace(
    '''result = sieve.run_hybrid_sieve(config,

                            seed_start=seed_start,''',
    '''result = sieve.run_hybrid_sieve(
                            prng_family=family_name,
                            seed_start=seed_start,'''
)

# Also remove the config definition lines we added
content = content.replace(
    '''phase_start = time.time()
                        from prng_registry import get_kernel_info
                        config = get_kernel_info(family_name)
                        result = sieve.run_hybrid_sieve(''',
    '''phase_start = time.time()
                        result = sieve.run_hybrid_sieve('''
)

with open('sieve_filter.py', 'w') as f:
    f.write(content)

print("✅ FINAL FIX: config removed, prng_family restored")
