#!/usr/bin/env python3
"""Complete fix: full run_hybrid_sieve call"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'result = sieve.run_hybrid_sieve(' in line:
        new_lines.append('                        result = sieve.run_hybrid_sieve(\n')
        new_lines.append('                            prng_family=family_name,\n')
        new_lines.append('                            seed_start=seed_start,\n')
        new_lines.append('                            seed_end=seed_end,\n')
        new_lines.append('                            residues=draws,\n')
        new_lines.append('                            strategies=strategies,\n')
        new_lines.append('                            min_match_threshold=phase2_threshold,\n')
        new_lines.append('                            offset=offset\n')
        new_lines.append('                        )\n')
        # Skip to after the old call
        while i < len(lines) and ')' not in lines[i]:
            i += 1
        i += 1
        continue
    new_lines.append(line)
    i += 1

with open('sieve_filter.py', 'w') as f:
    f.writelines(new_lines)

print("COMPLETE FIX: All args restored!")
