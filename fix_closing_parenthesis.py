#!/usr/bin/env python3
"""FINAL FIX: ADD MISSING ) AFTER pilot_duration"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'pilot_duration = (time.time() - pilot_start_time) * 1000' in line:
        new_lines.append(line)
        new_lines.append('                        )\n')  # ADD THE MISSING )
        i += 1
        # Now skip to pilot_info and dedent it
        while i < len(lines) and 'pilot_info = {' not in lines[i]:
            i += 1
        if i < len(lines):
            new_lines.append('                        ' + lines[i].lstrip())
        i += 1
        continue
    new_lines.append(line)
    i += 1

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("MISSING ) ADDED AFTER pilot_duration!")
