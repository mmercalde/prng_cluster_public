import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'
optimizer = WindowOptimizer(coordinator, test_seeds=100000)  # Smaller test

# Monkey patch to see what's being sent
original_create = coordinator._create_reverse_sieve_jobs
def debug_create(args, candidate_seeds):
    print(f"\n🔍 REVERSE SIEVE DEBUG:")
    print(f"  Candidates: {len(candidate_seeds)}")
    if candidate_seeds:
        print(f"  First candidate: {candidate_seeds[0]}")
        print(f"  Type: {type(candidate_seeds[0])}")
    return original_create(args, candidate_seeds)
coordinator._create_reverse_sieve_jobs = debug_create

result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=False)
print(f"\nForward: {result.forward_survivors}, Verified: {result.verified_survivors}")
