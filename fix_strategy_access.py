#!/usr/bin/env python3
"""Fix StrategyConfig access in run_hybrid_sieve"""

with open('sieve_filter.py', 'r') as f:
    content = f.read()

# Fix the array creation lines to use attribute access instead of dict access
old_code = """            strategy_max_misses = cp.array([s['max_consecutive_misses'] for s in strategies], dtype=cp.int32)
            strategy_tolerances = cp.array([s['skip_tolerance'] for s in strategies], dtype=cp.int32)"""

new_code = """            strategy_max_misses = cp.array([s.max_consecutive_misses for s in strategies], dtype=cp.int32)
            strategy_tolerances = cp.array([s.skip_tolerance for s in strategies], dtype=cp.int32)"""

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Fixed strategy attribute access")
else:
    print("⚠️ Could not find exact match, trying line by line...")
    # Try individual replacements
    content = content.replace("s['max_consecutive_misses']", "s.max_consecutive_misses")
    content = content.replace("s['skip_tolerance']", "s.skip_tolerance")
    print("✅ Replaced individual occurrences")

with open('sieve_filter.py', 'w') as f:
    f.write(content)

print("✅ sieve_filter.py updated")

# Test syntax
try:
    compile(content, 'sieve_filter.py', 'exec')
    print("✅ File syntax is valid!")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
