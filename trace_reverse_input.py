import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Patch _create_reverse_sieve_jobs to see candidates
original_create = coordinator._create_reverse_sieve_jobs
def debug_create(args, candidate_seeds):
    print(f"\n🔍 Reverse job with {len(candidate_seeds)} candidates:")
    for c in candidate_seeds[:3]:
        print(f"  {c}")
    return original_create(args, candidate_seeds)
coordinator._create_reverse_sieve_jobs = debug_create

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
print(f"\nResult: {result.forward_survivors} forward, {result.verified_survivors} verified")
