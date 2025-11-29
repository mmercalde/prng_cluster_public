#!/usr/bin/env python3
"""
Fix sieve_filter.py to check PRNG metadata instead of hardcoding
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the hybrid check we just added
for i in range(len(lines)):
    if "if not prng_family.endswith('_hybrid'):" in lines[i]:
        print(f"✅ Found the check at line {i+1}")
        
        # Replace with metadata-based check
        indent = '        '
        new_check = [
            f"{indent}# Check if this PRNG supports hybrid mode (variable skip patterns)\n",
            f"{indent}from prng_registry import get_kernel_info\n",
            f"{indent}prng_config = get_kernel_info(prng_family)\n",
            f"{indent}if not prng_config.get('variable_skip', False):\n",
            f"{indent}    raise ValueError(f\"Hybrid sieve requires a PRNG with variable_skip support. \"\n",
            f"{indent}                     f\"{{prng_family}} does not support variable skip patterns. \"\n",
            f"{indent}                     f\"Try using {{prng_family.replace('_hybrid', '')}} for constant skip.\")\n",
        ]
        
        # Replace the 3 lines
        lines[i-1:i+2] = new_check
        print(f"✅ Changed to metadata-based check")
        break

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed sieve_filter.py to use PRNG metadata")
print("\nNow the system will check the registry for 'variable_skip' capability!")

