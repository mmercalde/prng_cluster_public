#!/usr/bin/env python3
"""
Add the missing strategy parameters to kernel_args
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the kernel_args line with cp.int32(k),
for i in range(len(lines)):
    if 'survivor_count_gpu, cp.int32(n_seeds), cp.int32(k),' in lines[i]:
        print(f"✅ Found the line at {i+1}")
        
        # The next line should be cp.float32(min_match_threshold)
        # We need to INSERT strategy params BEFORE it
        
        indent = '                    '
        new_lines = [
            lines[i],  # Keep: survivor_count_gpu, cp.int32(n_seeds), cp.int32(k),
            f"{indent}strategy_max_misses, strategy_tolerances, cp.int32(n_strategies),\n",
            f"{indent}cp.float32(min_match_threshold),\n",
        ]
        
        # Replace current line + next line (which has min_match_threshold)
        lines[i:i+2] = new_lines
        
        print(f"✅ Added strategy_max_misses, strategy_tolerances, n_strategies")
        break

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed kernel_args - now has all required parameters!")

