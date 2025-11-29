#!/usr/bin/env python3
"""
Implement MT19937 init_by_array to match Python's random.seed()
Based on the original MT19937 reference implementation
"""

def init_by_array(init_key, key_length):
    """
    Initialize MT19937 state using an array
    This is what Python's random.seed() uses internally
    """
    N = 624
    state = [0] * N
    
    # Step 1: Initialize with default seed
    state[0] = 19650218
    for i in range(1, N):
        state[i] = (1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i) & 0xFFFFFFFF
    
    # Step 2: Mix in the init_key array
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
    
    # Step 3: Final mixing
    for _ in range(N-1):
        state[i] = ((state[i] ^ ((state[i-1] ^ (state[i-1] >> 30)) * 1566083941)) - i) & 0xFFFFFFFF
        i += 1
        if i >= N:
            state[0] = state[N-1]
            i = 1
    
    state[0] = 0x80000000  # MSB is 1; assuring non-zero initial array
    
    return state


def seed_to_init_key(seed):
    """Convert Python integer seed to init_key array like Python does"""
    if seed < 0:
        seed = -seed
    
    # Convert to bytes
    if seed == 0:
        seed_bytes = b'\x00'
    else:
        seed_bytes = seed.to_bytes((seed.bit_length() + 7) // 8, byteorder='big')
    
    # Pad to 4-byte boundary
    if len(seed_bytes) % 4:
        seed_bytes = b'\x00' * (4 - len(seed_bytes) % 4) + seed_bytes
    
    # Convert to 32-bit words
    init_key = []
    for i in range(0, len(seed_bytes), 4):
        word = int.from_bytes(seed_bytes[i:i+4], byteorder='big')
        init_key.append(word)
    
    return init_key


# Test it
if __name__ == '__main__':
    import random
    
    seed = 12345
    
    # Python's output
    random.seed(seed)
    python_state = random.getstate()
    
    # Our implementation
    init_key = seed_to_init_key(seed)
    our_state = init_by_array(init_key, len(init_key))
    
    print(f"Seed: {seed}")
    print(f"Init key: {init_key}")
    print()
    print("State comparison:")
    print(f"  Python state[0]: {python_state[1][0]}")
    print(f"  Our state[0]:    {our_state[0]}")
    print(f"  Match: {python_state[1][0] == our_state[0]}")
    print()
    print(f"  Python state[1]: {python_state[1][1]}")
    print(f"  Our state[1]:    {our_state[1]}")
    print(f"  Match: {python_state[1][1] == our_state[1]}")
    print()
    
    # Check first 10
    matches = sum(1 for i in range(10) if python_state[1][i] == our_state[i])
    print(f"First 10 matches: {matches}/10")
