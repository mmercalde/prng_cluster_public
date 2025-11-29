#!/usr/bin/env python3
"""
Fix strategy.name access to handle dicts
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find the line with strategies[strat_id].name
for i in range(len(lines)):
    if "strategies[strat_id].name" in lines[i]:
        print(f"✅ Found .name access at line {i+1}")
        
        # Replace with dict access
        old_line = lines[i]
        new_line = old_line.replace(
            "strategies[strat_id].name",
            "strategies[strat_id]['name'] if isinstance(strategies[strat_id], dict) else strategies[strat_id].name"
        )
        
        lines[i] = new_line
        print(f"✅ Changed to handle dict or object")
        break

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed strategy name access")

