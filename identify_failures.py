import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Capture forward survivors
forward_survivors = []
original_execute = coordinator.execute_gpu_job
def capture_forward(job, worker):
    result = original_execute(job, worker)
    if job.search_type == 'residue_sieve' and result.results:
        if 'survivors' in result.results:
            forward_survivors.extend(result.results['survivors'])
    return result
coordinator.execute_gpu_job = capture_forward

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

# Get forward seeds
forward_seeds = set(s['seed'] for s in forward_survivors)
print(f"Forward found {len(forward_seeds)} unique seeds:")
for s in sorted(forward_seeds):
    print(f"  {s}")

# Now run reverse manually and see which pass
from reverse_sieve_filter import GPUReverseSieve, load_draws_from_daily3
draws = load_draws_from_daily3('daily3.json', 768, ['midday', 'evening'], 0)
sieve = GPUReverseSieve(gpu_id=0)

print(f"\nTesting each seed locally:")
failed = []
for s in forward_survivors:
    result = sieve.run_reverse_sieve([s], 'lcg32', draws, (0,20), 0.01, 0)
    if len(result['survivors']) == 0:
        failed.append(s['seed'])
        print(f"  ❌ FAIL: seed={s['seed']}, skip={s['best_skip']}, rate={s['match_rate']}")

print(f"\n{len(failed)} seeds failed reverse verification:")
print(failed)
