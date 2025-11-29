#!/usr/bin/env python3
"""
Fix execute_sieve_job to support all hybrid PRNGs, not just mt19937
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the line: if use_hybrid and family_name == 'mt19937':
for i in range(len(lines)):
    if "if use_hybrid and family_name == 'mt19937':" in lines[i]:
        print(f"✅ Found hardcoded mt19937 check at line {i+1}")
        
        # Replace with check for any hybrid PRNG
        indent = '            '
        old_line = lines[i]
        new_line = f"{indent}# Check if this family supports hybrid mode\n"
        
        # Add metadata check
        check_lines = [
            new_line,
            f"{indent}from prng_registry import get_kernel_info\n",
            f"{indent}family_config = get_kernel_info(family_name)\n",
            f"{indent}supports_hybrid = family_config.get('variable_skip', False)\n",
            f"{indent}\n",
            f"{indent}if use_hybrid and supports_hybrid:\n",
        ]
        
        lines[i:i+1] = check_lines
        
        print(f"✅ Changed to check 'variable_skip' metadata")
        print(f"   Now all hybrid PRNGs (mt19937_hybrid, xorshift32_hybrid, etc.) will work!")
        break

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed execute_sieve_job to support all hybrid PRNGs")

