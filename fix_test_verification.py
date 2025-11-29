#!/usr/bin/env python3
"""
Fix the test verification to check for the correct seed (88675)
"""

with open('test_all_prngs_distributed_proper.py', 'r') as f:
    content = f.read()

# Replace ALL occurrences of hardcoded 12345 with the variable known_seed
content = content.replace("s['seed'] == 12345", "s['seed'] == known_seed")
content = content.replace("if s['seed'] == 12345", "if s['seed'] == known_seed")
content = content.replace('print(f"   ✅ Found seed 12345', 'print(f"   ✅ Found seed {known_seed}')
content = content.replace('print(f"   ❌ Did NOT find seed 12345', 'print(f"   ❌ Did NOT find seed {known_seed}')

with open('test_all_prngs_distributed_proper.py', 'w') as f:
    f.write(content)

print("✅ Fixed test verification to use known_seed variable instead of hardcoded 12345")

