#!/usr/bin/env python3
"""ULTIMATE FIX: CORRECT estimate_background_thresholds CALL"""

import re

with open('sieve_filter.py', 'r') as f:
    content = f.read()

# Find the broken block
pattern = r"""
(
    est_p1,\ est_p2\ =\ estimate_background_thresholds\(
    .*?
    max_pilot_seeds=128,
    .*?
    pilot_duration\ =\ \(time\.time\(\)\ -\ pilot_start_time\)\ *\ 1000
    .*?
    pilot_info\ =.*?
)
"""

match = re.search(pattern, content, re.DOTALL | re.VERBOSE)
if match:
    broken = match.group(1)
    fixed = re.sub(
        r'(max_pilot_seeds=128,)\s*pilot_duration.*?\n\s*pilot_info.*',
        r'\1\n                            pilot_duration = (time.time() - pilot_start_time) * 1000\n                        )\n                        pilot_info =',
        broken,
        flags=re.DOTALL
    )
    content = content.replace(broken, fixed)

with open('sieve_filter.py', 'w') as f:
    f.write(content)

print("ULTIMATE FIX APPLIED!")
