#!/usr/bin/env python3
"""
Fix sieve_filter.py to pass PRNG-specific parameters to hybrid kernels
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the kernel_args for hybrid sieve (around line 285)
for i in range(len(lines)):
    if 'kernel_args = [' in lines[i] and i > 260 and i < 300:
        # Check if this is the hybrid kernel_args (has strategy_max_misses)
        context = ''.join(lines[max(0, i-5):i+15])
        if 'strategy_max_misses' in context and 'strategy_tolerances' in context:
            print(f"✅ Found hybrid kernel_args at line {i+1}")
            
            # Find the closing ] of kernel_args
            j = i
            while j < len(lines) and ']' not in lines[j]:
                j += 1
            
            if j >= len(lines):
                print("❌ Could not find end of kernel_args")
                continue
            
            # The current last line should be: cp.float32(min_match_threshold), cp.int32(offset)
            # We need to add PRNG params before offset
            
            # Find the line with offset
            for k in range(i, j+1):
                if 'cp.int32(offset)' in lines[k]:
                    print(f"✅ Found offset at line {k+1}")
                    
                    # Replace the line to add PRNG-specific params
                    indent = '                    '
                    old_line = lines[k]
                    
                    # New lines with PRNG params
                    new_lines = [
                        f"{indent}cp.float32(min_match_threshold),\n",
                    ]
                    
                    # Add PRNG-specific params based on config
                    # We need to insert code BEFORE this section to get the params
                    # Let's insert it before kernel_args
                    
                    insert_before_args = i
                    param_code = [
                        f"{indent}# Get PRNG-specific parameters\n",
                        f"{indent}prng_params = config.get('default_params', {{}})\n",
                    ]
                    
                    lines[insert_before_args:insert_before_args] = param_code
                    
                    # Now modify the offset line to include params
                    # For xorshift32: shift_a, shift_b, shift_c
                    # For mt19937: no extra params
                    
                    new_offset_lines = [
                        f"{indent}cp.float32(min_match_threshold),\n",
                        f"{indent}# Add PRNG-specific params\n",
                        f"{indent}*[cp.int32(v) for v in prng_params.values()],\n",
                        f"{indent}cp.int32(offset)\n",
                    ]
                    
                    lines[k+2:k+3] = new_offset_lines  # +2 because we inserted 2 lines before
                    
                    print(f"✅ Added PRNG-specific params before offset")
                    break
            break

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed sieve_filter.py to pass PRNG-specific params to hybrid kernels")

