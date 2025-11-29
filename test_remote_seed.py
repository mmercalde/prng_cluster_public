import sys, json
sys.path.insert(0, '/home/michael/distributed_prng_analysis')
from reverse_sieve_filter import GPUReverseSieve, load_draws_from_daily3

print("Loading draws...")
draws = load_draws_from_daily3('daily3.json', 768, ['midday', 'evening'], 0)
print(f"Loaded {len(draws)} draws, first 5: {draws[:5]}")

candidate = {'seed': 208989, 'skip': 19}
print(f"Testing candidate: {candidate}")

sieve = GPUReverseSieve(gpu_id=0)
result = sieve.run_reverse_sieve([candidate], 'lcg32', draws, (19, 19), 0.01, 0)
print(f"Result: {len(result['survivors'])} survivors")
if result['survivors']:
    print(f"Survivor: {result['survivors'][0]}")
