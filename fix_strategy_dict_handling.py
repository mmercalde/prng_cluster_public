#!/usr/bin/env python3
"""
Fix sieve_filter.py to handle strategies as dicts (not just objects)
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the lines that access strategy attributes
for i in range(len(lines)):
    if 'strategy_max_misses = cp.array([s.max_consecutive_misses for s in strategies]' in lines[i]:
        print(f"✅ Found strategy_max_misses at line {i+1}")
        
        # Replace both lines to handle dicts
        old_line1 = lines[i]
        old_line2 = lines[i+1]
        
        indent = '            '
        new_lines = [
            f"{indent}# Handle strategies as dicts or objects\n",
            f"{indent}strategy_max_misses = cp.array([\n",
            f"{indent}    s['max_consecutive_misses'] if isinstance(s, dict) else s.max_consecutive_misses\n",
            f"{indent}    for s in strategies\n",
            f"{indent}], dtype=cp.int32)\n",
            f"{indent}strategy_tolerances = cp.array([\n",
            f"{indent}    s['skip_tolerance'] if isinstance(s, dict) else s.skip_tolerance\n",
            f"{indent}    for s in strategies\n",
            f"{indent}], dtype=cp.int32)\n",
        ]
        
        # Replace 2 lines with new code
        lines[i:i+2] = new_lines
        
        print(f"✅ Updated to handle dict or object strategies")
        break

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed sieve_filter.py to handle strategy dicts")

