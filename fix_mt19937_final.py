#!/usr/bin/env python3
"""
Final fix: CPU function should call mt19937_extract with state and index
"""

with open('prng_registry.py', 'r') as f:
    content = f.read()

# In the CPU function, the calls should use state and index, not mt and mti
# Find the mt19937_cpu_simple function and fix the calls

# The function uses 'state' and 'index' but calls with 'mt' and 'mti'
old_call = "mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A)"
new_call = "mt19937_extract(state, index, 624, 397, 0x80000000, 0x7FFFFFFF, 0x9908B0DF)"

# Only replace in CPU code, not in GPU kernel
lines = content.split('\n')
for i in range(len(lines)):
    # Only fix if we're in the Python code (not in the kernel string)
    if old_call in lines[i] and 'r"""' not in lines[i] and "r'''" not in lines[i]:
        lines[i] = lines[i].replace(old_call, new_call)
        print(f"Fixed line {i+1}: {lines[i].strip()}")

content = '\n'.join(lines)

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("\n✅ Fixed CPU function calls to use state and index")

