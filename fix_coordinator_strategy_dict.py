#!/usr/bin/env python3
"""
Fix coordinator.py - strategies are already dicts, don't call .to_dict()
"""

with open('coordinator.py', 'r') as f:
    content = f.read()

# Replace the .to_dict() calls with direct dict usage
old_pattern1 = "'strategies': [s.to_dict() for s in self._sieve_config['strategies']]"
new_pattern1 = "'strategies': self._sieve_config['strategies']"

count = content.count(old_pattern1)
print(f"Found {count} occurrences of .to_dict() on strategies")

content = content.replace(old_pattern1, new_pattern1)

with open('coordinator.py', 'w') as f:
    f.write(content)

print(f"✅ Fixed {count} occurrences in coordinator.py")
print("   Strategies are already dicts, removed .to_dict() calls")

