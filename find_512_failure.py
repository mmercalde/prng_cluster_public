import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

forward_survivors = []
verified_survivors = []

original_execute = coordinator.execute_gpu_job
def capture_execute(job, worker):
    result = original_execute(job, worker)
    if result.results:
        if job.search_type == 'residue_sieve':
            forward_survivors.extend(result.results.get('survivors', []))
        elif job.search_type == 'reverse_sieve':
            verified_survivors.extend(result.results.get('survivors', []))
    return result

coordinator.execute_gpu_job = capture_execute

optimizer = WindowOptimizer(coordinator, test_seeds=100_000)
result = optimizer.evaluate_window('lcg32', 512, use_all_gpus=True)

forward_seeds = {s['seed'] for s in forward_survivors}
verified_seeds = {s['seed'] for s in verified_survivors}

missing = sorted(forward_seeds - verified_seeds)
print(f"\nWindow 512 - Missing seed:")
for seed in missing:
    fwd = [s for s in forward_survivors if s['seed'] == seed][0]
    print(f"  seed={seed}, skip={fwd['best_skip']}")
