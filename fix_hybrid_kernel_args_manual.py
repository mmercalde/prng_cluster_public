#!/usr/bin/env python3
"""
Manually fix the hybrid kernel_args to include PRNG-specific params
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the specific kernel_args section in run_hybrid_sieve
# It should be after "kernel, config = self._get_kernel" and contain strategy_max_misses
found = False
for i in range(len(lines)):
    if 'kernel_args = [' in lines[i]:
        # Check context - look for strategy_max_misses nearby
        context = ''.join(lines[max(0,i-10):min(len(lines),i+15)])
        if 'strategy_max_misses' in context and 'strategy_tolerances' in context:
            print(f"✅ Found hybrid kernel_args at line {i+1}")
            
            # Find the line with offset (should be within 10 lines)
            for j in range(i, min(i+12, len(lines))):
                if 'cp.int32(offset)' in lines[j]:
                    print(f"✅ Found offset line at {j+1}")
                    
                    # Replace this line to add PRNG params
                    # Old: cp.float32(min_match_threshold), cp.int32(offset)
                    # New: cp.float32(min_match_threshold),
                    #      *[cp.int32(v) for v in config.get('default_params', {}).values()],
                    #      cp.int32(offset)
                    
                    indent = '                    '
                    
                    # Replace the offset line and the line before
                    # Line j-1 should have: cp.float32(min_match_threshold), cp.int32(offset)
                    # We need to split this into 3 lines
                    
                    old_line = lines[j]
                    
                    new_lines = [
                        f"{indent}cp.float32(min_match_threshold),\n",
                        f"{indent}*[cp.int32(v) for v in config.get('default_params', {{}}).values()],\n",
                        f"{indent}cp.int32(offset)\n",
                    ]
                    
                    # Replace the line that has both threshold and offset
                    lines[j-1:j+1] = new_lines
                    
                    found = True
                    print(f"✅ Inserted PRNG params between threshold and offset")
                    break
            
            if found:
                break

if not found:
    print("❌ Could not find the correct location!")
    exit(1)

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed sieve_filter.py - PRNG params will now be passed to hybrid kernel")

