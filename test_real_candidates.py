import sys
sys.path.insert(0, '.')
from reverse_sieve_filter import GPUReverseSieve, load_draws_from_daily3

candidates = [
    {'seed': 235667, 'skip': 19},
    {'seed': 265005, 'skip': 19}
]

draws = load_draws_from_daily3('daily3.json', 768, ['midday', 'evening'], 0)
sieve = GPUReverseSieve(gpu_id=0)

print("Testing the actual skip=19 candidates:")
for c in candidates:
    result = sieve.run_reverse_sieve([c], 'lcg32', draws, (0,20), 0.01, 0)
    found = len(result['survivors'])
    match_rate = result['survivors'][0]['match_rate'] if found else 0
    print(f"  Seed {c['seed']}: {'✅ PASS' if found else '❌ FAIL'} (rate={match_rate:.6f})")
