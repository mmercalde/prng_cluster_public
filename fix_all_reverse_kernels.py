#!/usr/bin/env python3
"""Fix ALL reverse kernels to iterate backwards"""

import re

with open('prng_registry.py', 'r') as f:
    content = f.read()

# Backup
with open('prng_registry.py.bak_reverse_fix', 'w') as f:
    f.write(content)

print("✅ Created backup: prng_registry.py.bak_reverse_fix\n")

# Find all reverse kernel functions
reverse_kernels = [
    'java_lcg_reverse_sieve',
    'java_lcg_hybrid_reverse_sieve',
    'mt19937_reverse_sieve',
    'mt19937_hybrid_reverse_sieve',
    'xorshift32_reverse_sieve',
    'xorshift32_hybrid_reverse_sieve',
    'xorshift64_reverse_sieve',
    'xorshift64_hybrid_reverse_sieve',
    'xorshift128_reverse_sieve',
    'xorshift128_hybrid_reverse_sieve',
    'pcg32_reverse_sieve',
    'pcg32_hybrid_reverse_sieve',
    'lcg32_reverse_sieve',
    'lcg32_hybrid_reverse_sieve',
    'minstd_reverse_sieve',
    'minstd_hybrid_reverse_sieve',
    'philox4x32_reverse_sieve',
    'philox4x32_hybrid_reverse_sieve',
    'xoshiro256pp_reverse_sieve',
    'sfc64_reverse_sieve'
]

lines = content.split('\n')
fixed_count = 0
current_kernel = None

for i, line in enumerate(lines):
    # Track which kernel we're in
    for kernel_name in reverse_kernels:
        if f'void {kernel_name}(' in line:
            current_kernel = kernel_name
            break
    
    # Check if we're leaving a reverse kernel
    if current_kernel and 'void ' in line and current_kernel not in line:
        if not any(k in line for k in reverse_kernels):
            current_kernel = None
    
    # Fix the loop if we're in a reverse kernel
    if current_kernel and 'for (int i = 0; i < k; i++)' in line:
        old_line = line
        lines[i] = line.replace('for (int i = 0; i < k; i++)', 
                               'for (int i = k-1; i >= 0; i--)')
        fixed_count += 1
        print(f"✅ Fixed {current_kernel}")
        print(f"   Line {i+1}: {old_line.strip()}")
        print(f"   →         {lines[i].strip()}\n")

# Write back
with open('prng_registry.py', 'w') as f:
    f.write('\n'.join(lines))

print(f"\n{'='*60}")
print(f"SUMMARY: Fixed {fixed_count} reverse kernels")
print(f"{'='*60}")

# Verify
print("\nVerifying fixes...")
with open('prng_registry.py', 'r') as f:
    content = f.read()

forward_loops = content.count('for (int i = 0; i < k; i++)')
reverse_loops = content.count('for (int i = k-1; i >= 0; i--)')

print(f"Forward loops remaining: {forward_loops}")
print(f"Reverse loops created: {reverse_loops}")

if fixed_count == 0:
    print("\n⚠️  WARNING: No fixes applied! Check if kernels already fixed or pattern different.")
else:
    print(f"\n✅ Successfully fixed {fixed_count} kernels!")
    print("   Backup saved to: prng_registry.py.bak_reverse_fix")

