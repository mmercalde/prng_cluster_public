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

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

forward_seeds = {s['seed'] for s in forward_survivors}
verified_seeds = {s['seed'] for s in verified_survivors}

print(f"\n{'='*60}")
print(f"Forward found: {len(forward_seeds)} unique seeds")
print(f"Reverse verified: {len(verified_seeds)} unique seeds")
print(f"\nMISSING seeds (in forward but not in reverse):")
missing = sorted(forward_seeds - verified_seeds)
for seed in missing:
    fwd = [s for s in forward_survivors if s['seed'] == seed][0]
    print(f"  seed={seed}, skip={fwd['best_skip']}, rate={fwd['match_rate']}")
