#!/usr/bin/env python3
"""
CORRECT FIX: Reverse residues array for _reverse kernels
"""

import re

# Step 1: Revert CUDA loop changes
print("Step 1: Reverting CUDA loop direction changes...")
with open('prng_registry.py', 'r') as f:
    prng_content = f.read()

# Count how many reverse loops exist
reverse_loops = prng_content.count('for (int i = k-1; i >= 0; i--)')
print(f"   Found {reverse_loops} reverse loops to revert")

# Revert them all
prng_content = prng_content.replace('for (int i = k-1; i >= 0; i--)', 'for (int i = 0; i < k; i++)')

with open('prng_registry.py', 'w') as f:
    f.write(prng_content)

print(f"✅ Reverted {reverse_loops} CUDA loops back to forward direction\n")

# Step 2: Fix sieve_filter.py
print("Step 2: Adding residue reversal logic to sieve_filter.py...")

with open('sieve_filter.py', 'r') as f:
    sieve_content = f.read()

# Fix location 1 (line 122)
old_122 = """            residues_gpu = cp.array(residues, dtype=cp.uint32)"""

new_122 = """            # TEMPORAL REVERSAL: Reverse residues for _reverse kernels
            if '_reverse' in prng_family:
                residues_reversed = residues[::-1]
                residues_gpu = cp.array(residues_reversed, dtype=cp.uint32)
            else:
                residues_gpu = cp.array(residues, dtype=cp.uint32)"""

# Fix location 2 (line 291)
old_291 = """            residues_gpu = cp.array(residues, dtype=cp.uint32)"""

new_291 = """            # TEMPORAL REVERSAL: Reverse residues for _reverse kernels
            if '_reverse' in prng_family:
                residues_reversed = residues[::-1]
                residues_gpu = cp.array(residues_reversed, dtype=cp.uint32)
            else:
                residues_gpu = cp.array(residues, dtype=cp.uint32)"""

# Apply fixes
fixes_applied = 0

# We need to replace them one at a time to avoid replacing the same line twice
lines = sieve_content.split('\n')
new_lines = []
already_fixed = False

for i, line in enumerate(lines):
    if 'residues_gpu = cp.array(residues, dtype=cp.uint32)' in line and not already_fixed:
        # Check if not already fixed
        if i > 0 and 'TEMPORAL REVERSAL' not in lines[i-1]:
            indent = len(line) - len(line.lstrip())
            spaces = ' ' * indent
            new_lines.append(f"{spaces}# TEMPORAL REVERSAL: Reverse residues for _reverse kernels")
            new_lines.append(f"{spaces}if '_reverse' in prng_family:")
            new_lines.append(f"{spaces}    residues_reversed = residues[::-1]")
            new_lines.append(f"{spaces}    residues_gpu = cp.array(residues_reversed, dtype=cp.uint32)")
            new_lines.append(f"{spaces}else:")
            new_lines.append(line)
            fixes_applied += 1
            continue
    new_lines.append(line)

with open('sieve_filter.py', 'w') as f:
    f.write('\n'.join(new_lines))

print(f"✅ Added residue reversal at {fixes_applied} location(s)\n")

print("="*60)
print("CORRECT FIX COMPLETED!")
print("="*60)
print("Changes made:")
print("1. ✅ CUDA kernels: Reverted to forward loops (i=0 to k-1)")
print("2. ✅ Python code: Reverses residues array for _reverse kernels")
print("")
print("How it works:")
print("  Forward kernel: Checks residues[0, 1, 2, ..., k-1]")
print("  Reverse kernel: Checks residues[k-1, k-2, ..., 1, 0]")
print("  (by reversing the array before passing to kernel)")
print("="*60)

