#!/usr/bin/env python3
"""Fix syntax error in run_hybrid_sieve call"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'result = sieve.run_hybrid_sieve(' in line:
        new_lines.append('                        result = sieve.run_hybrid_sieve(\n')
        new_lines.append('                            prng_family=family_name,\n')
        i += 1
        # Skip until we find the last argument
        while i < len(lines) and 'offset=offset' not in lines[i]:
            i += 1
        if i < len(lines):
            new_lines.append('                            ' + lines[i].strip() + '\n')
            new_lines.append('                        )\n')
        i += 1
        continue
    new_lines.append(line)
    i += 1

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("SYNTAX FIXED!")
