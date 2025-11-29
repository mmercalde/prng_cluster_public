#!/usr/bin/env python3
"""FINAL FIX: Move pilot_info outside estimate_background_thresholds"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'max_pilot_seeds=128' in line:
        new_lines.append(line)
        i += 1
        while i < len(lines) and lines[i].strip() == '':
            i += 1
        new_lines.append('                            pilot_duration = (time.time() - pilot_start_time) * 1000\n')
        i += 1
        while i < len(lines) and 'pilot_info = {' not in lines[i]:
            i += 1
        if i < len(lines):
            new_lines.append('                        ' + lines[i].lstrip())  # dedent pilot_info
        i += 1
        continue
    new_lines.append(line)
    i += 1

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("PILOT_INFO FINAL FIX APPLIED!")
