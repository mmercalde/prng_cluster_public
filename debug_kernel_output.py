#!/usr/bin/env python3
"""Debug: manually replicate what the kernel should do"""

# Expected residues (last 30 from file)
import json
with open('test_seed42_fullstate.json', 'r') as f:
    data = json.load(f)
residues = [entry['full_state'] for entry in data[-30:]]

# Replicate kernel logic for seed 42, skip=0, offset=0
def xorshift32(state, shift_a=13, shift_b=17, shift_c=5):
    state = state & 0xFFFFFFFF
    state ^= (state << shift_a) & 0xFFFFFFFF
    state ^= (state >> shift_b) & 0xFFFFFFFF
    state ^= (state << shift_c) & 0xFFFFFFFF
    return state & 0xFFFFFFFF

seed = 42
skip = 0
offset = 0

# PRE-ADVANCE BY OFFSET (offset * (skip+1) = 0 * 1 = 0 times)
state = seed
for o in range(offset * (skip + 1)):
    state = xorshift32(state)

# Skip advancement (skip = 0 times)
for b in range(skip):
    state = xorshift32(state)

# Now check matches
matches = 0
print("Checking matches:")
for i in range(min(5, len(residues))):
    # Advance state FIRST
    state = xorshift32(state)
    
    # Check matches
    m1 = (state % 1000) == (residues[i] % 1000)
    m2 = (state % 8) == (residues[i] % 8)
    m3 = (state % 125) == (residues[i] % 125)
    
    if m1 and m2 and m3:
        matches += 1
        
    print(f"{i}: state={state}, residue={residues[i]}")
    print(f"   %1000: {state%1000} vs {residues[i]%1000} {'✓' if m1 else '✗'}")
    print(f"   %8:    {state%8} vs {residues[i]%8} {'✓' if m2 else '✗'}")
    print(f"   %125:  {state%125} vs {residues[i]%125} {'✓' if m3 else '✗'}")
    print(f"   Match: {'YES' if (m1 and m2 and m3) else 'NO'}")

print(f"\nTotal matches in first 5: {matches}/5")
