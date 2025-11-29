#!/usr/bin/env python3
"""Fix StrategyConfig object access in sieve_filter.py"""

with open('sieve_filter.py', 'r') as f:
    content = f.read()

# Replace all dictionary-style accesses with attribute accesses
replacements = [
    ("s['max_consecutive_misses']", "s.max_consecutive_misses"),
    ("s['skip_tolerance']", "s.skip_tolerance"),
    ("s['name']", "s.name"),
    ("s['allow_reseed']", "s.allow_reseed"),
]

print("Fixing StrategyConfig access patterns...")
fixes_made = 0

for old, new in replacements:
    count = content.count(old)
    if count > 0:
        content = content.replace(old, new)
        print(f"  ✅ Replaced {count}x: {old} → {new}")
        fixes_made += count

if fixes_made == 0:
    print("  ⚠️  No replacements needed - might already be fixed")
else:
    print(f"\nTotal fixes: {fixes_made}")

# Write back
with open('sieve_filter.py', 'w') as f:
    f.write(content)

print("✅ sieve_filter.py updated")

# Validate syntax
try:
    compile(content, 'sieve_filter.py', 'exec')
    print("✅ File syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    import traceback
    traceback.print_exc()
