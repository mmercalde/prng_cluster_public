#!/bin/bash
set -e

echo "=========================================================================="
echo "COMPLETE MT19937 LAMBDA FIX - PRODUCTION VERSION"
echo "=========================================================================="
echo ""

# 1. CREATE BACKUP
echo "1. Creating backup..."
backup_file="prng_registry_before_fix_$(date +%Y%m%d_%H%M%S).py"
cp prng_registry.py "$backup_file"
echo "   ✅ Backup: $backup_file"

# 2. APPLY FIX
echo ""
echo "2. Applying fix to both MT19937 kernels..."
python3 << 'PYEOF'
import re

with open('prng_registry.py', 'r') as f:
    content = f.read()

# Device function with fallback typedef
device_function_block = '''#include <stdint.h>
#ifndef __cplusplus
typedef unsigned int uint32_t;
#endif

// MT19937 extract - FULL algorithm, static device function (no lambda)
static __device__ __forceinline__ uint32_t mt19937_extract(
    uint32_t* mt, int& mti, const int N, const int M,
    const uint32_t UPPER_MASK, const uint32_t LOWER_MASK,
    const uint32_t MATRIX_A
) {
    if (mti >= N) {
        int kk = 0;
        uint32_t y;
        for (; kk < N - M; ++kk) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        }
        for (; kk < N - 1; ++kk) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        }
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        mti = 0;
    }
    uint32_t y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    y ^= (y >> 18);
    return y;
}

'''

# IDEMPOTENCY: Check if already applied
marker1 = "MT19937_KERNEL = r'''"
marker2 = "MT19937_HYBRID_KERNEL = r'''"

kernel1_section = content.split(marker1, 1)[1].split("extern \"C\" __global__", 1)[0]
kernel2_section = content.split(marker2, 1)[1].split("extern \"C\" __global__", 1)[0]

needs_fix_kernel1 = "mt19937_extract(" not in kernel1_section
needs_fix_kernel2 = "mt19937_extract(" not in kernel2_section

if needs_fix_kernel1:
    marker = "MT19937_KERNEL = r'''\nextern \"C\" __global__"
    content = content.replace(marker, "MT19937_KERNEL = r'''\n" + device_function_block + "extern \"C\" __global__")
    print("   ✅ Fixed MT19937_KERNEL")
else:
    print("   ⏭️  MT19937_KERNEL already fixed")

if needs_fix_kernel2:
    marker = "MT19937_HYBRID_KERNEL = r'''\nextern \"C\" __global__"
    content = content.replace(marker, "MT19937_HYBRID_KERNEL = r'''\n" + device_function_block + "extern \"C\" __global__")
    print("   ✅ Fixed MT19937_HYBRID_KERNEL")
else:
    print("   ⏭️  MT19937_HYBRID_KERNEL already fixed")

# Remove lambdas (only if they exist)
old_lambda = '''        auto extract = [&]() -> unsigned int {
            if (mti >= N) {
                // Twist
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
        };'''

lambda_count = content.count(old_lambda)
if lambda_count > 0:
    content = content.replace(old_lambda, '        // Using mt19937_extract() device function')
    print(f"   ✅ Removed {lambda_count} lambdas")
else:
    print("   ⏭️  No lambdas to remove")

# Replace extract() calls
extract_count = content.count('extract()')
if extract_count > 0:
    content = content.replace('extract()', 'mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A)')
    print(f"   ✅ Replaced {extract_count} extract() calls")
else:
    print("   ⏭️  No extract() calls to replace")

# Robust type update using regex
content = re.sub(r'\bunsigned\s+int\s+mt\[624\];?', 'uint32_t mt[624]', content)
print("   ✅ Updated mt[624] to uint32_t")

with open('prng_registry.py', 'w') as f:
    f.write(content)

try:
    compile(content, 'prng_registry.py', 'exec')
    print("   ✅ Python syntax valid")
except SyntaxError as e:
    print(f"   ❌ Syntax error: {e}")
    exit(1)
PYEOF

# 3. VERIFICATION
echo ""
echo "3. Verification checks..."
echo ""

echo "   a) Lambda check:"
if grep -q "auto extract" prng_registry.py; then
    echo "      ❌ Lambdas still present!"
    exit 1
else
    echo "      ✅ No lambdas remain"
fi

echo ""
echo "   b) Device function count:"
count=$(grep -c "static __device__.*mt19937_extract" prng_registry.py || echo "0")
if [ "$count" -eq 2 ]; then
    echo "      ✅ Found 2 device functions (correct)"
else
    echo "      ❌ Expected 2 device functions, found $count"
    exit 1
fi

echo ""
echo "   c) Stray extract() calls:"
if grep -n "extract()" prng_registry.py; then
    echo "      ⚠️  Found extract() calls - verify these are safe"
else
    echo "      ✅ No stray extract() calls"
fi

echo ""
echo "   d) Module load check:"
python3 -c "import prng_registry; print('      ✅ Module loads')" || { echo "      ❌ Module failed to load"; exit 1; }

# 4. CPU TEST
echo ""
echo "4. CPU test (seed=5489, first 10 outputs)..."
python3 << 'PYEOF'
from prng_registry import mt19937_cpu

SEED = 5489
EXPECTED = [3499211612, 581869302, 3890346734, 3586334585, 545404204,
            4161255391, 3922919429, 949333985, 2715962298, 1323567403]

cpu_out = mt19937_cpu(SEED, 10, skip=0)
match = (cpu_out[:10] == EXPECTED)
if match:
    print("   ✅ CPU PASS")
else:
    print("   ❌ CPU FAIL")
    print(f"      Expected: {EXPECTED[:5]}...")
    print(f"      Got:      {cpu_out[:5]}...")
    exit(1)
PYEOF

# 5. GPU SMOKE TEST (dedicated kernel)
echo ""
echo "5. GPU smoke test (dedicated MT test kernel)..."
python3 << 'PYEOF'
import cupy as cp

# Standalone MT test kernel
code = r'''
#include <stdint.h>
#ifndef __cplusplus
typedef unsigned int uint32_t;
#endif

static __device__ __forceinline__ uint32_t mt19937_extract(
    uint32_t* mt, int& mti, const int N, const int M,
    const uint32_t UPPER_MASK, const uint32_t LOWER_MASK,
    const uint32_t MATRIX_A
) {
    if (mti >= N) {
        int kk = 0;
        uint32_t y;
        for (; kk < N - M; ++kk) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        }
        for (; kk < N - 1; ++kk) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        }
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1U) ? MATRIX_A : 0U);
        mti = 0;
    }
    uint32_t y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    y ^= (y >> 18);
    return y;
}

extern "C" __global__
void mt_test(uint32_t seed, uint32_t* out, int n, int burn) {
    const int N = 624, M = 397;
    const uint32_t UPPER_MASK = 0x80000000U;
    const uint32_t LOWER_MASK = 0x7FFFFFFFU;
    const uint32_t MATRIX_A = 0x9908B0DFU;
    
    uint32_t mt[624];
    int mti = N;
    
    // Canonical seeding
    mt[0] = seed;
    for (int i = 1; i < N; ++i) {
        uint32_t x = mt[i-1] ^ (mt[i-1] >> 30);
        mt[i] = 1812433253U * x + (uint32_t)i;
    }
    
    // Burn-in
    for (int i = 0; i < burn; ++i) {
        (void)mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A);
    }
    
    // Generate outputs
    for (int i = 0; i < n; ++i) {
        out[i] = mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A);
    }
}
'''

try:
    mod = cp.RawModule(code=code, backend='nvrtc')
    kernel = mod.get_function('mt_test')
    
    SEED = 5489
    N = 10
    out = cp.zeros(N, dtype=cp.uint32)
    
    kernel((1,), (1,), (SEED, out, N, 0))
    
    gpu = out.get().tolist()
    expected = [3499211612, 581869302, 3890346734, 3586334585, 545404204,
                4161255391, 3922919429, 949333985, 2715962298, 1323567403]
    
    if gpu == expected:
        print("   ✅ GPU PASS")
    else:
        print("   ❌ GPU FAIL")
        print(f"      Expected: {expected[:5]}...")
        print(f"      Got:      {gpu[:5]}...")
        exit(1)
except Exception as e:
    print(f"   ❌ GPU test failed: {e}")
    exit(1)
PYEOF

echo ""
echo "=========================================================================="
echo "✅ ALL TESTS PASSED! FIX COMPLETE!"
echo "=========================================================================="
echo ""
echo "Next steps:"
echo "  1. Deploy to workers:"
echo "     scp prng_registry.py michael@192.168.3.120:~/distributed_prng_analysis/"
echo "     scp prng_registry.py michael@192.168.3.154:~/distributed_prng_analysis/"
echo ""
echo "  2. Test with constant skip data:"
echo "     python3 timestamp_search.py test_our_mt19937.json \\"
echo "       --mode second --window 512 --threshold 0.8 \\"
echo "       --prngs mt19937 --skip-max 10 --resume-policy restart"
