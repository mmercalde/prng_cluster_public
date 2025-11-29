import sys, json
sys.path.insert(0, '.')
from reverse_sieve_filter import load_draws_from_daily3

draws = load_draws_from_daily3('daily3.json', 768, ['midday', 'evening'], 0)

print(f"Total draws loaded: {len(draws)}")
print(f"None values: {draws.count(None)}")
print(f"Zero values: {draws.count(0)}")

# Find positions of None/invalid values
for i, d in enumerate(draws):
    if d is None or d == 0:
        print(f"  Position {i}: {d}")
        if i > 0:
            print(f"    Previous: {draws[i-1]}")
        if i < len(draws) - 1:
            print(f"    Next: {draws[i+1]}")
