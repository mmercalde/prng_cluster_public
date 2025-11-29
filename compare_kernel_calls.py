#!/usr/bin/env python3
"""
Compare what manual test does vs what sieve_filter does
"""

print("MANUAL TEST (WORKS):")
print("  n_draws = 100")
print("  n_seeds = 10000") 
print("  seed range: 50000-60000")
print("  cp.int32(n_draws) for k parameter")
print("")

print("SIEVE_FILTER:")
with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()
    
# Find where k is passed
for i, line in enumerate(lines):
    if 'cp.int32(k),' in line and i > 300:
        print(f"  Line {i+1}: {line.strip()}")
        # Check what k is
        for j in range(max(0, i-30), i):
            if 'k = ' in lines[j] or 'len(residues)' in lines[j]:
                print(f"  Line {j+1}: {lines[j].strip()}")

