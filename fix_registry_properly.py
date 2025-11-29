#!/usr/bin/env python3
"""
Properly fix the KERNEL_REGISTRY structure
"""

with open('prng_registry.py', 'r') as f:
    lines = f.readlines()

# Find the problem area (around line 1140-1155)
# Remove the orphaned lines and fix indentation

fixed_lines = []
skip_until = -1

for i, line in enumerate(lines):
    line_num = i + 1
    
    # Skip the orphaned description lines (1146-1151)
    if 1146 <= line_num <= 1151:
        continue
    
    # Fix xorshift64 indentation (should be 4 spaces, currently more)
    if line_num == 1152 and "'xorshift64':" in line:
        fixed_lines.append("    'xorshift64': {\n")
        continue
    
    fixed_lines.append(line)

with open('prng_registry.py', 'w') as f:
    f.writelines(fixed_lines)

print("✅ Removed orphaned lines")
print("✅ Fixed xorshift64 indentation")

