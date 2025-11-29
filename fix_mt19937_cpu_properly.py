#!/usr/bin/env python3
"""
Fix mt19937_cpu_simple: ADD parameters back to match what the calls expect
"""

with open('prng_registry.py', 'r') as f:
    content = f.read()

# The nested function signature needs parameters
old_signature = "    def mt19937_extract():"
new_signature = "    def mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A):"

content = content.replace(old_signature, new_signature)

# The function needs to use the OUTER scope variables (state, index)
# not the parameters. The parameters are just there for the signature.

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Added parameters back to mt19937_extract() signature")
print("   (Function still uses state/index from outer scope)")

