import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer
from reverse_sieve_filter import GPUReverseSieve, load_draws_from_daily3

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Manually test each of the 6 skip=19 candidates locally
candidates = [
    {'seed': 39270, 'skip': 19},
    {'seed': 135971, 'skip': 19},
    {'seed': 208989, 'skip': 19},
    {'seed': 385530, 'skip': 19},
    {'seed': 461147, 'skip': 19},
    {'seed': 534165, 'skip': 19}
]

draws = load_draws_from_daily3('daily3.json', 768, ['midday', 'evening'], 0)
sieve = GPUReverseSieve(gpu_id=0)

print("Testing skip=19 candidates locally:")
for c in candidates:
    result = sieve.run_reverse_sieve([c], 'lcg32', draws, (0,20), 0.01, 0)
    found = len(result['survivors'])
    print(f"  Seed {c['seed']}: {'✅ PASS' if found else '❌ FAIL'}")
