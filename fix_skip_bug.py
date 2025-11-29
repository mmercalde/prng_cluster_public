#!/usr/bin/env python3
"""
Backup prng_registry.py and fix xoshiro256pp_reverse and sfc64_reverse kernels
BUG: Skip only happens once before all draws, should happen BEFORE EACH draw
"""

import shutil
from datetime import datetime

# Backup first!
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_file = f'prng_registry.py.backup_{timestamp}'
shutil.copy('prng_registry.py', backup_file)
print(f"✅ Backup created: {backup_file}")
print()

# Read the current file
with open('prng_registry.py', 'r') as f:
    content = f.read()

# Define the fix for xoshiro256pp_reverse
XOSHIRO_OLD = '''        // Burn skip_val outputs before first draw
        for (int s = 0; s < skip_val; s++) {
            unsigned long long temp = s0 + s3;
            unsigned long long result = ((temp << 23) | (temp >> 41)) + s0;
            unsigned long long t = s1 << 17;
            s2 ^= s0;
            s3 ^= s1;
            s1 ^= s2;
            s0 ^= s3;
            s2 ^= t;
            s3 = ((s3 << 45) | (s3 >> 19));
        }
        int matches = 0;
        for (int i = 0; i < k; i++) {
            // Generate output
            unsigned long long temp = s0 + s3;
            unsigned long long result = ((temp << 23) | (temp >> 41)) + s0;
            unsigned long long t = s1 << 17;
            s2 ^= s0;
            s3 ^= s1;
            s1 ^= s2;
            s0 ^= s3;
            s2 ^= t;
            s3 = ((s3 << 45) | (s3 >> 19));'''

XOSHIRO_NEW = '''        int matches = 0;
        for (int i = 0; i < k; i++) {
            // Burn skip_val outputs BEFORE EACH draw
            for (int s = 0; s < skip_val; s++) {
                unsigned long long temp = s0 + s3;
                unsigned long long result = ((temp << 23) | (temp >> 41)) + s0;
                unsigned long long t = s1 << 17;
                s2 ^= s0;
                s3 ^= s1;
                s1 ^= s2;
                s0 ^= s3;
                s2 ^= t;
                s3 = ((s3 << 45) | (s3 >> 19));
            }
            // Generate output
            unsigned long long temp = s0 + s3;
            unsigned long long result = ((temp << 23) | (temp >> 41)) + s0;
            unsigned long long t = s1 << 17;
            s2 ^= s0;
            s3 ^= s1;
            s1 ^= s2;
            s0 ^= s3;
            s2 ^= t;
            s3 = ((s3 << 45) | (s3 >> 19));'''

# Fix xoshiro256pp_reverse
if XOSHIRO_OLD in content:
    content = content.replace(XOSHIRO_OLD, XOSHIRO_NEW)
    print("✅ Fixed xoshiro256pp_reverse kernel")
else:
    print("⚠️  Could not find xoshiro256pp_reverse pattern to fix")
    print("    (kernel may already be fixed or pattern changed)")

# Define the fix for sfc64_reverse
SFC64_OLD = '''        // Burn skip_val outputs before first draw
        for (int s = 0; s < skip_val; s++) {
            unsigned long long result = a + b + counter;
            unsigned long long t = b << 24;
            c ^= a;
            d ^= b;
            b ^= c;
            a ^= d;
            c ^= t;
            d = (d << 11) | (d >> 53);
            counter++;
        }
        int matches = 0;
        for (int i = 0; i < k; i++) {
            // Generate output
            unsigned long long result = a + b + counter;
            unsigned long long t = b << 24;
            c ^= a;
            d ^= b;
            b ^= c;
            a ^= d;
            c ^= t;
            d = (d << 11) | (d >> 53);
            counter++;'''

SFC64_NEW = '''        int matches = 0;
        for (int i = 0; i < k; i++) {
            // Burn skip_val outputs BEFORE EACH draw
            for (int s = 0; s < skip_val; s++) {
                unsigned long long result = a + b + counter;
                unsigned long long t = b << 24;
                c ^= a;
                d ^= b;
                b ^= c;
                a ^= d;
                c ^= t;
                d = (d << 11) | (d >> 53);
                counter++;
            }
            // Generate output
            unsigned long long result = a + b + counter;
            unsigned long long t = b << 24;
            c ^= a;
            d ^= b;
            b ^= c;
            a ^= d;
            c ^= t;
            d = (d << 11) | (d >> 53);
            counter++;'''

# Fix sfc64_reverse
if SFC64_OLD in content:
    content = content.replace(SFC64_OLD, SFC64_NEW)
    print("✅ Fixed sfc64_reverse kernel")
else:
    print("⚠️  Could not find sfc64_reverse pattern to fix")
    print("    (kernel may already be fixed or pattern changed)")

# Write the fixed content
with open('prng_registry.py', 'w') as f:
    f.write(content)

print()
print("="*70)
print("FIXES APPLIED!")
print("="*70)
print()
print("What was fixed:")
print("  • xoshiro256pp_reverse: Skip now happens BEFORE EACH draw")
print("  • sfc64_reverse: Skip now happens BEFORE EACH draw")
print()
print("Next steps:")
print("  1. Clear kernel cache on ALL nodes:")
print("     rm -rf ~/.cupy/kernel_cache/*")
print("     ssh 192.168.3.120 'rm -rf ~/.cupy/kernel_cache/*'")
print("     ssh 192.168.3.154 'rm -rf ~/.cupy/kernel_cache/*'")
print()
print("  2. Deploy to remote nodes:")
print("     scp prng_registry.py 192.168.3.120:~/distributed_prng_analysis/")
print("     scp prng_registry.py 192.168.3.154:~/distributed_prng_analysis/")
print()
print("  3. Rerun tests:")
print("     bash test_last_four_prngs.sh")
print()
print(f"Backup saved as: {backup_file}")
