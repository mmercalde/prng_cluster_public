import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from reverse_sieve_filter import GPUReverseSieve, load_draws_from_daily3

# Test seed 629100, skip 4 - forward found it, reverse didn't
candidate = {'seed': 629100, 'skip': 4}

# Load draws EXACTLY as forward sieve does
coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Manually run LCG32 forward to see what it matches
a, c, m = 1103515245, 12345, 0x7FFFFFFF
state = candidate['seed']

# Apply skip before first draw
for s in range(candidate['skip']):
    state = (a * state + c) % m

# Load draws
draws = load_draws_from_daily3('daily3.json', 768, ['midday', 'evening'], 0)

# Count matches going forward
matches = 0
for i in range(768):
    state = (a * state + c) % m
    if (state % 1000 == draws[i] % 1000 and 
        state % 8 == draws[i] % 8 and 
        state % 125 == draws[i] % 125):
        matches += 1
    # Apply skip after each draw
    for s in range(candidate['skip']):
        state = (a * state + c) % m

print(f"Manual forward test: {matches}/768 = {matches/768:.6f}")
print(f"Should be >= 0.01 threshold: {matches/768 >= 0.01}")

# Now test with reverse sieve
sieve = GPUReverseSieve(gpu_id=0)
result = sieve.run_reverse_sieve([candidate], 'lcg32', draws, (0,20), 0.01, 0)
print(f"Reverse sieve found: {len(result['survivors'])} survivors")
if result['survivors']:
    print(f"Match rate: {result['survivors'][0]['match_rate']}")
