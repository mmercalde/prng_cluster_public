#!/usr/bin/env python3
"""
Fix sieve_filter.py to properly cast all integer parameters to cp.int32
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the line where kernel_args is built
for i, line in enumerate(lines):
    if 'n_seeds, k, skip_min, skip_max, cp.float32(min_match_threshold)' in line:
        print(f"Found kernel_args construction at line {i+1}")
        
        # Replace the line to cast all ints to cp.int32
        old_line = line
        new_line = line.replace(
            'n_seeds, k, skip_min, skip_max, cp.float32(min_match_threshold)',
            'cp.int32(n_seeds), cp.int32(k), cp.int32(skip_min), cp.int32(skip_max), cp.float32(min_match_threshold)'
        )
        
        lines[i] = new_line
        
        print(f"OLD: {old_line.strip()}")
        print(f"NEW: {new_line.strip()}")
        break

# Write back
with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("\n✅ Fixed sieve_filter.py to cast integer parameters to cp.int32!")

