import sys
sys.path.insert(0, '.')
from reverse_sieve_filter import GPUReverseSieve, load_draws_from_daily3

candidate_seeds = [{'seed': 87, 'skip': 4}]
draws = load_draws_from_daily3('daily3.json', 768, ['midday', 'evening'], 0)
print(f"Loaded {len(draws)} draws")

sieve = GPUReverseSieve(gpu_id=0)
result = sieve.run_reverse_sieve(candidate_seeds, 'lcg32', draws, (0,20), 0.01, 0)
print(f"Survivors: {len(result['survivors'])}")
print(result)
