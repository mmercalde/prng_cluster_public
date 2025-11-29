#!/usr/bin/env python3
"""Fix both CPU and GPU MT19937 implementations"""

with open('prng_registry.py', 'r') as f:
    content = f.read()

print("Applying fixes...")

# 1. Fix GPU device function - replace #include with typedef
old_header = '''#include <stdint.h>
#ifndef __cplusplus
typedef unsigned int uint32_t;
#endif'''

new_header = '''typedef unsigned int uint32_t;'''

content = content.replace(old_header, new_header)
print("✅ Fixed GPU header (typedef instead of #include)")

# 2. Replace mt19937_cpu function with pure Python implementation
import re

# Find and replace the entire mt19937_cpu function
cpu_func_pattern = r'def mt19937_cpu\(.*?\):\s*""".*?""".*?return.*?\n(?=\ndef |\nclass |\n[A-Z])'

new_cpu_func = '''def mt19937_cpu(seed: int, n: int, skip: int = 0):
    """Pure Python canonical MT19937 for CPU verification."""
    N, M = 624, 397
    MATRIX_A = 0x9908B0DF
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7FFFFFFF

    # Initialize state array
    mt = [0] * N
    mt[0] = seed & 0xFFFFFFFF
    for i in range(1, N):
        x = mt[i - 1] ^ (mt[i - 1] >> 30)
        mt[i] = (1812433253 * x + i) & 0xFFFFFFFF
    mti = N

    def extract():
        nonlocal mti
        if mti >= N:
            for kk in range(N - M):
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ (MATRIX_A if y & 1 else 0)
            for kk in range(N - M, N - 1):
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)
                mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ (MATRIX_A if y & 1 else 0)
            y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK)
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ (MATRIX_A if y & 1 else 0)
            mti = 0
        y = mt[mti]
        mti += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & 0xFFFFFFFF

    # Burn skip draws
    for _ in range(skip):
        extract()

    return [extract() for _ in range(n)]

'''

content = re.sub(cpu_func_pattern, new_cpu_func, content, flags=re.DOTALL)
print("✅ Replaced mt19937_cpu with pure Python implementation")

# Save
with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ All fixes applied")
