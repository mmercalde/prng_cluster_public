#!/usr/bin/env python3
"""
Fix sieve_filter.py to support all hybrid PRNGs, not just mt19937_hybrid
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find and replace the hardcoded check (around line 252-254)
fixed = False
for i in range(len(lines)):
    if "if prng_family != 'mt19937_hybrid':" in lines[i]:
        print(f"✅ Found hardcoded check at line {i+1}")
        
        # Replace the check to allow all *_hybrid PRNGs
        indent = '        '
        new_check = [
            f"{indent}# Check if this is a hybrid PRNG\n",
            f"{indent}if not prng_family.endswith('_hybrid'):\n",
            f"{indent}    raise ValueError(f\"Hybrid sieve requires a hybrid PRNG (e.g., mt19937_hybrid, xorshift32_hybrid), got {{prng_family}}\")\n",
        ]
        
        # Replace 3 lines (the comment, if statement, and raise)
        lines[i-1:i+2] = new_check
        fixed = True
        print(f"✅ Changed check to accept all *_hybrid PRNGs")
        break

if not fixed:
    print("❌ Could not find the hardcoded check!")
    exit(1)

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed sieve_filter.py to support all hybrid PRNGs")

