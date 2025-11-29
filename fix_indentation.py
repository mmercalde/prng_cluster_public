#!/usr/bin/env python3
"""Fix unexpected indent in run_hybrid_sieve call"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith('    min_match_threshold='):
        new_lines.append('                            ' + line[4:])  # 28 spaces
    else:
        new_lines.append(line)

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("INDENTATION FIXED!")
