import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

original_create = coordinator._create_reverse_sieve_jobs
def debug_create(args, candidate_seeds):
    skip = args.skip_min
    if skip == 19 and len(candidate_seeds) == 6:
        print(f"\n🔍 ALL 6 candidates with skip=19:")
        for c in candidate_seeds:
            print(f"  {c}")
    return original_create(args, candidate_seeds)
coordinator._create_reverse_sieve_jobs = debug_create

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
optimizer.cache = {}  # Clear cache
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
