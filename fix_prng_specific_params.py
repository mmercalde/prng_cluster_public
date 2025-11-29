#!/usr/bin/env python3
"""
Fix to explicitly handle PRNG-specific parameters
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the kernel_args with the unpacking line
for i in range(len(lines)):
    if '*[cp.int32(v) for v in config.get' in lines[i]:
        print(f"✅ Found unpacking at line {i+1}")
        
        # Replace the unpacking with conditional logic
        # We need to check the PRNG family and add params accordingly
        
        indent = '                    '
        
        # Replace the line with conditional param addition
        new_code = [
            f"{indent}cp.float32(min_match_threshold),\n",
        ]
        
        # Add conditional params based on PRNG type
        # For now, hardcode the known PRNGs (we'll make this more dynamic later)
        new_code.extend([
            f"{indent}# Add PRNG-specific params based on family\n",
            f"{indent}*([\n",
            f"{indent}    cp.int32(config['default_params']['shift_a']),\n",
            f"{indent}    cp.int32(config['default_params']['shift_b']),\n",
            f"{indent}    cp.int32(config['default_params']['shift_c'])\n",
            f"{indent}] if 'xorshift' in prng_family else []),\n",
        ])
        
        # Keep the offset line (next line)
        # Remove the old unpacking line and threshold line before it
        lines[i-1:i+1] = new_code
        
        print(f"✅ Replaced with conditional PRNG param handling")
        break

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed sieve_filter.py - xorshift PRNGs will get shift params, mt19937 won't")

