#!/usr/bin/env python3
"""Move device function INSIDE the kernel string"""

with open('prng_registry.py', 'r') as f:
    content = f.read()

# Remove the device function that's outside the string
device_func_outside = '''// Device function for MT19937 extract (replaces lambda)
__device__ __forceinline__ unsigned int mt19937_extract(
    unsigned int* mt, int& mti, const int N, const int M,
    const unsigned int UPPER_MASK, const unsigned int LOWER_MASK,
    const unsigned int MATRIX_A
) {
    if (mti >= N) {
        for (int i = 0; i < N; i++) {
            unsigned int y = (mt[i] & UPPER_MASK) | (mt[(i+1) % N] & LOWER_MASK);
            mt[i] = mt[(i + M) % N] ^ (y >> 1);
            if (y & 1) mt[i] ^= MATRIX_A;
        }
        mti = 0;
    }
    unsigned int y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    y ^= (y >> 18);
    return y;
}

'''

content = content.replace(device_func_outside, '')
print("✅ Removed device function from outside")

# Now add it INSIDE the kernel string, right after the opening r'''
# Find: MT19937_HYBRID_KERNEL = r'''\nextern "C" __global__
# Insert device function before the kernel function

old_start = '''MT19937_HYBRID_KERNEL = r\'\'\'
extern "C" __global__'''

new_start = '''MT19937_HYBRID_KERNEL = r\'\'\'
// Device function for MT19937 extract (replaces lambda)
__device__ __forceinline__ unsigned int mt19937_extract(
    unsigned int* mt, int& mti, const int N, const int M,
    const unsigned int UPPER_MASK, const unsigned int LOWER_MASK,
    const unsigned int MATRIX_A
) {
    if (mti >= N) {
        for (int i = 0; i < N; i++) {
            unsigned int y = (mt[i] & UPPER_MASK) | (mt[(i+1) % N] & LOWER_MASK);
            mt[i] = mt[(i + M) % N] ^ (y >> 1);
            if (y & 1) mt[i] ^= MATRIX_A;
        }
        mti = 0;
    }
    unsigned int y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    y ^= (y >> 18);
    return y;
}

extern "C" __global__'''

content = content.replace(old_start, new_start)
print("✅ Added device function INSIDE kernel string")

with open('prng_registry.py', 'w') as f:
    f.write(content)

print("✅ Device function now in correct location!")
