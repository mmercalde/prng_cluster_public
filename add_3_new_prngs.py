#!/usr/bin/env python3
"""
Add 3 new PRNGs to prng_registry.py
Following EXACT pattern of xorshift32
"""

print("""
================================================================================
STEP 1: Add these CPU functions to prng_registry.py
(Add after xorshift32_cpu, around line 50)
================================================================================
""")

print(r'''
def java_lcg_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """Java LCG (java.util.Random) CPU reference"""
    a = kwargs.get('a', 25214903917)
    c = kwargs.get('c', 11)
    m = (1 << 48)  # 2^48
    
    state = seed & (m - 1)
    
    # Apply skip
    for _ in range(skip):
        state = (a * state + c) & (m - 1)
    
    outputs = []
    for _ in range(n):
        state = (a * state + c) & (m - 1)
        output = (state >> 16) & 0xFFFFFFFF  # Use upper 32 bits like Java
        outputs.append(output)
    
    return outputs


def minstd_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """MINSTD (Park & Miller) CPU reference"""
    a = kwargs.get('a', 48271)
    m = kwargs.get('m', 2147483647)  # 2^31 - 1
    
    state = seed % m
    if state == 0:
        state = 1
    
    # Apply skip
    for _ in range(skip):
        state = (a * state) % m
    
    outputs = []
    for _ in range(n):
        state = (a * state) % m
        outputs.append(state)
    
    return outputs


def xorshift128_cpu(seed: int, n: int, skip: int = 0, **kwargs) -> List[int]:
    """Xorshift128 CPU reference"""
    # Initialize 128-bit state from seed
    x = seed & 0xFFFFFFFF
    y = 362436069
    z = 521288629
    w = 88675123
    
    # Apply skip
    for _ in range(skip):
        t = x ^ (x << 11)
        t &= 0xFFFFFFFF
        x, y, z = y, z, w
        w = (w ^ (w >> 19) ^ (t ^ (t >> 8))) & 0xFFFFFFFF
    
    outputs = []
    for _ in range(n):
        t = x ^ (x << 11)
        t &= 0xFFFFFFFF
        x, y, z = y, z, w
        w = (w ^ (w >> 19) ^ (t ^ (t >> 8))) & 0xFFFFFFFF
        outputs.append(w)
    
    return outputs
''')

print("✅ CPU functions complete")
