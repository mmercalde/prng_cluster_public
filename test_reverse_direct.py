import sys
sys.path.insert(0, '.')
import cupy as cp
from reverse_sieve_filter import GPUReverseSieve

# Test data
candidate_seeds = [
    {'seed': 87, 'skip': 4, 'match_rate': 0.01}
]

# Load actual draws properly
import json
with open('daily3.json') as f:
    data = json.load(f)
    # Get 768 midday draws
    if isinstance(data, dict) and 'draws' in data:
        draws = [d['midday'] for d in data['draws'][:768]]
    else:
        draws = [d['midday'] for d in data[:768]]

print(f"Testing {len(candidate_seeds)} candidates against {len(draws)} draws")
print(f"Candidate: {candidate_seeds[0]}")

sieve = GPUReverseSieve(gpu_id=0)
result = sieve.run_reverse_sieve(
    candidate_seeds=candidate_seeds,
    prng_family='lcg32',
    draws=draws,
    skip_range=(0, 20),
    min_match_threshold=0.01,
    offset=0
)

print(f"\nResult: {result}")
print(f"Survivors: {len(result['survivors'])}")
if result['survivors']:
    print(f"First survivor: {result['survivors'][0]}")
