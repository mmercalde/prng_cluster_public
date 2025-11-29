#!/usr/bin/env python3
"""
Fix ONLY the GPU kernel, not the Python CPU function!
"""

with open('prng_registry.py', 'r') as f:
    lines = f.readlines()

# Find the MT19937_KERNEL string and fix only within it
in_kernel = False
kernel_start = -1
kernel_end = -1

for i, line in enumerate(lines):
    if "MT19937_KERNEL = r'''" in line:
        in_kernel = True
        kernel_start = i
        print(f"Found MT19937_KERNEL start at line {i+1}")
    elif in_kernel and "'''" in line and i > kernel_start:
        in_kernel = False
        kernel_end = i
        print(f"Found MT19937_KERNEL end at line {i+1}")
        break

if kernel_start > 0 and kernel_end > 0:
    # Fix only within the kernel
    for i in range(kernel_start, kernel_end + 1):
        # Replace mt19937_extract() with proper C call
        if "mt19937_extract(mt, &mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A)" not in lines[i]:
            lines[i] = lines[i].replace(
                "mt19937_extract()",
                "mt19937_extract(mt, &mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A)"
            )
    
    print(f"✅ Fixed GPU kernel (lines {kernel_start+1} to {kernel_end+1})")
    
    # Now revert the Python CPU function - remove the C syntax we accidentally added
    for i in range(len(lines)):
        if i < kernel_start or i > kernel_end:
            # Outside kernel - fix Python code
            lines[i] = lines[i].replace(
                "mt19937_extract(mt, &mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A)",
                "mt19937_extract(mt, mti, N, M, UPPER_MASK, LOWER_MASK, MATRIX_A)"
            )
    
    with open('prng_registry.py', 'w') as f:
        f.writelines(lines)
    
    print("✅ Fixed Python CPU function (removed & syntax)")
else:
    print("❌ Could not find MT19937_KERNEL boundaries!")

