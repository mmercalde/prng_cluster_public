#!/usr/bin/env python3
"""
Fix ALL indentation issues in the two-phase block
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Lines 507-620 need proper indentation based on nesting level
# Find all lines that start "if", "for", "elif", "else" and ensure next line is indented

fixes = 0
for i in range(495, 620):
    if i >= len(lines):
        break
        
    line = lines[i].rstrip()
    
    # If this line ends with ":", next non-empty line needs more indent
    if line.endswith(':') and i+1 < len(lines):
        current_indent = len(line) - len(line.lstrip())
        next_line = lines[i+1]
        next_indent = len(next_line) - len(next_line.lstrip())
        
        # If next line is not empty and not indented more
        if next_line.strip() and next_indent <= current_indent:
            # Add 4 spaces to next line
            lines[i+1] = '    ' + next_line
            fixes += 1
            print(f"Fixed line {i+2}: indented after '{line[-40:].strip()}'")

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print(f"\n✅ Fixed {fixes} indentation issues")

