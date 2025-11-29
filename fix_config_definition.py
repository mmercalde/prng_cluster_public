#!/usr/bin/env python3
"""Add missing config definition before run_hybrid_sieve calls"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find lines that call run_hybrid_sieve and add config definition before them
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Check if this line calls run_hybrid_sieve
    if 'run_hybrid_sieve(config,' in line and i > 0:
        # Look backwards to find where we should insert the config definition
        # Insert it after the print statement and before the call
        if 'print(f"Testing' in lines[i-2]:
            # Insert after the print and phase_start lines
            new_lines.append(lines[i-2])  # print statement
            new_lines.append(lines[i-1])  # phase_start or phase2_threshold
            new_lines.append('                        from prng_registry import get_kernel_info\n')
            new_lines.append('                        config = get_kernel_info(family_name)\n')
            new_lines.append(line)  # The run_hybrid_sieve call
            i += 1
            continue
    
    new_lines.append(line)
    i += 1

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Added config definition before run_hybrid_sieve calls")
