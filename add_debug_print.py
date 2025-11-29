#!/usr/bin/env python3
"""
Add debug printing to sieve_filter to see what it's actually passing
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the kernel call line
for i in range(len(lines)):
    if 'kernel((blocks,), (threads_per_block,), tuple(kernel_args))' in lines[i] and i > 300:
        print(f"✅ Found kernel call at line {i+1}")
        
        # Insert debug print BEFORE the kernel call
        indent = '                '
        debug_lines = [
            f"{indent}# DEBUG: Print kernel_args\n",
            f"{indent}print(f'DEBUG: Calling kernel with {{len(kernel_args)}} args', file=sys.stderr)\n",
            f"{indent}for idx, arg in enumerate(kernel_args):\n",
            f"{indent}    if hasattr(arg, 'shape'):\n",
            f"{indent}        print(f'  {{idx}}: {{type(arg).__name__}} shape={{arg.shape}}', file=sys.stderr)\n",
            f"{indent}    else:\n",
            f"{indent}        print(f'  {{idx}}: {{type(arg).__name__}} = {{arg}}', file=sys.stderr)\n",
        ]
        
        lines[i:i] = debug_lines
        print(f"✅ Added debug printing before kernel call")
        break

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Added debug output to sieve_filter.py")
print("   Now run the test and we'll see EXACTLY what's being passed!")

