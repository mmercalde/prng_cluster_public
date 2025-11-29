#!/usr/bin/env python3
"""
Python-compatible MT19937 implementation
Matches Python's random.seed() and random.getrandbits(32)
"""

from typing import List

def mt19937_python_compatible(seed: int, n: int, skip: int = 0) -> List[int]:
    """
    MT19937 that matches Python's random module
    Uses init_by_array like Python does
    """
    N = 624
    M = 397
    MATRIX_A = 0x9908B0DF
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7FFFFFFF
    
    # Convert seed to init_key array (like Python does)
    def seed_to_init_key(s):
        if s < 0:
            s = -s
        if s == 0:
            seed_bytes = b'\x00'
        else:
            seed_bytes = s.to_bytes((s.bit_length() + 7) // 8, byteorder='big')
        
        if len(seed_bytes) % 4:
            seed_bytes = b'\x00' * (4 - len(seed_bytes) % 4) + seed_bytes
        
        init_key = []
        for i in range(0, len(seed_bytes), 4):
            word = int.from_bytes(seed_bytes[i:i+4], byteorder='big')
            init_key.append(word)
        return init_key
    
    # Initialize using init_by_array
    def init_by_array(init_key, key_length):
        state = [0] * N
        state[0] = 19650218
        for i in range(1, N):
            state[i] = (1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i) & 0xFFFFFFFF
        
        i = 1
        j = 0
        k = max(N, key_length)
        
        for _ in range(k):
            state[i] = ((state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1664525)) + init_key[j] + j) & 0xFFFFFFFF
            i += 1
            j += 1
            if i >= N:
                state[0] = state[N-1]
                i = 1
            if j >= key_length:
                j = 0
        
        for _ in range(N-1):
            state[i] = ((state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1566083941)) - i) & 0xFFFFFFFF
            i += 1
            if i >= N:
                state[0] = state[N-1]
                i = 1
        
        state[0] = 0x80000000
        return state
    
    # Initialize state
    init_key = seed_to_init_key(seed)
    state = init_by_array(init_key, len(init_key))
    index = N  # Force initial twist
    
    # Twist function
    def twist():
        nonlocal state
        for i in range(N):
            x = (state[i] & UPPER_MASK) | (state[(i+1) % N] & LOWER_MASK)
            xA = x >> 1
            if x & 1:
                xA ^= MATRIX_A
            state[i] = state[(i + M) % N] ^ xA
    
    # Extract function
    def extract():
        nonlocal index
        if index >= N:
            twist()
            index = 0
        
        y = state[index]
        index += 1
        
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        
        return y & 0xFFFFFFFF
    
    # Skip outputs
    for _ in range(skip):
        extract()
    
    # Generate n outputs
    outputs = []
    for _ in range(n):
        outputs.append(extract())
    
    return outputs


# Test against Python
if __name__ == '__main__':
    import random
    
    print("=== Testing Python Compatibility ===")
    
    for seed in [0, 1, 12345, 0xFFFFFFFF]:
        random.seed(seed)
        python_out = [random.getrandbits(32) for _ in range(10)]
        
        our_out = mt19937_python_compatible(seed, 10, skip=0)
        
        match = python_out == our_out
        print(f"Seed {seed:10d}: {'✅ MATCH' if match else '❌ FAIL'}")
        if not match:
            print(f"  Python: {python_out[:3]}")
            print(f"  Ours:   {our_out[:3]}")
    
    print("\n=== Testing with skip ===")
    seed = 12345
    skip = 5
    
    random.seed(seed)
    for _ in range(skip):
        random.getrandbits(32)
    python_after_skip = [random.getrandbits(32) for _ in range(5)]
    
    our_after_skip = mt19937_python_compatible(seed, 5, skip=skip)
    
    match = python_after_skip == our_after_skip
    print(f"Skip test: {'✅ MATCH' if match else '❌ FAIL'}")
    if not match:
        print(f"  Python: {python_after_skip}")
        print(f"  Ours:   {our_after_skip}")
