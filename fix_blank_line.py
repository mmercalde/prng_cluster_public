#!/usr/bin/env python3
"""Remove blank line after max_pilot_seeds=128"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    if 'max_pilot_seeds=128' in lines[i]:
        new_lines.append(lines[i])
        i += 1
        # Skip blank lines
        while i < len(lines) and lines[i].strip() == '':
            i += 1
        new_lines.append(lines[i])
        i += 1
    else:
        new_lines.append(lines[i])
        i += 1

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("BLANK LINE AFTER max_pilot_seeds REMOVED!")
