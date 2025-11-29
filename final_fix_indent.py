#!/usr/bin/env python3
"""Remove extra ) after run_hybrid_sieve call"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

with open('sieve_filter.py', 'w') as f:
    for line in lines:
        if line.strip() == ')':
            # Skip lines that are just a closing parenthesis with only indentation
            if line.strip() == ')':
                continue
        f.write(line)

print("EXTRA ) REMOVED!")
