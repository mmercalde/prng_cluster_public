#!/usr/bin/env python3
"""
Fix MT19937 CPU reference - remove unused parameters from nested function
"""

with open('prng_registry.py', 'r') as f:
    content = f.read()

# The nested function signature has parameters it doesn't use
old_signature = "    def mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A):"
new_signature = "    def mt19937_extract():"

content = content.replace(old_signature, new_signature)

# Now fix the calls to not pass those parameters
old_call = "mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A)"
new_call = "mt19937_extract()"

content = content.replace(old_call, new_call)

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Fixed mt19937_cpu_simple!")
print("   - Removed unused parameters from mt19937_extract()")
print("   - Updated all calls to mt19937_extract()")

