#!/usr/bin/env python3
"""Fix run_sieve call indentation and structure"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'result = sieve.run_sieve(' in line:
        new_lines.append('            result = sieve.run_sieve(\n')
        new_lines.append('                prng_family=family_name,\n')
        new_lines.append('                seed_start=seed_start,\n')
        new_lines.append('                seed_end=seed_end,\n')
        new_lines.append('                residues=draws,\n')
        new_lines.append('                skip_range=skip_range,\n')
        new_lines.append('                offset=offset,\n')
        new_lines.append('                min_match_threshold=min_match_threshold,\n')
        new_lines.append('                custom_params=custom_params\n')
        new_lines.append('            )\n')
        i += 1
        # Skip all old lines until after the call
        while i < len(lines) and 'per_family_results.append(result)' not in lines[i]:
            i += 1
        continue
    new_lines.append(line)
    i += 1

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("run_sieve CALL FIXED!")
