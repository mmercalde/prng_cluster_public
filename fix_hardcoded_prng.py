#!/usr/bin/env python3
"""
Fix hardcoded prng_family='mt19937' in sieve_filter.py
Replace it with the variable family_name from the loop
"""

with open('sieve_filter.py', 'r') as f:
    content = f.read()

# Find and replace the hardcoded line
old_line = "                        prng_family='mt19937',  # FULL MT19937 (same as test data)"
new_line = "                        prng_family=family_name,  # Use PRNG from job spec"

if old_line in content:
    content = content.replace(old_line, new_line)
    
    with open('sieve_filter.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed hardcoded prng_family!")
    print(f"   OLD: prng_family='mt19937'")
    print(f"   NEW: prng_family=family_name")
else:
    print("❌ Could not find the hardcoded line - may have already been fixed")

