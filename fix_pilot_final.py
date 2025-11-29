#!/usr/bin/env python3
"""FINAL FIX: Move pilot_duration into estimate_background_thresholds"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'max_pilot_seeds=128' in line:
        new_lines.append(line)  # keep max_pilot_seeds line
        i += 1
        # skip blank lines
        while i < len(lines) and lines[i].strip() == '':
            i += 1
        # add pilot_duration as argument
        new_lines.append('                            pilot_duration = (time.time() - pilot_start_time) * 1000\n')
        i += 1  # skip the old pilot_duration line
        continue
    new_lines.append(line)
    i += 1

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("PILOT_DURATION FINAL FIX APPLIED!")
