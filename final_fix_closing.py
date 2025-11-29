#!/usr/bin/env python3
"""Add missing ) after offset=offset"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'offset=offset' in line:
        new_lines.append(line)
        new_lines.append('                        )\n')
    else:
        new_lines.append(line)

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("MISSING ) ADDED!")
