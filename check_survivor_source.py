import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Patch to see ALL forward survivors
class PatchedOptimizer(WindowOptimizer):
    def evaluate_window(self, prng, window, use_all_gpus=False):
        cache_key = f"{prng}_{window}_{self.test_seeds}_{'all' if use_all_gpus else '1'}"
        
        # Force fresh run - no cache
        if cache_key in self.cache:
            del self.cache[cache_key]
        
        result = super().evaluate_window(prng, window, use_all_gpus)
        return result

optimizer = PatchedOptimizer(coordinator, test_seeds=1_000_000)

# Intercept forward survivors
original_create_sieve = coordinator._create_sieve_jobs
def debug_sieve(args):
    jobs = original_create_sieve(args)
    return jobs
    
original_create_reverse = coordinator._create_reverse_sieve_jobs
all_forward_seeds = []
def debug_reverse(args, candidates):
    print(f"\n🔍 Reverse called with {len(candidates)} candidates")
    cand_seeds = [c['seed'] for c in candidates]
    print(f"Seeds: {cand_seeds}")
    print(f"Are these in forward results? {[s in all_forward_seeds for s in cand_seeds]}")
    return original_create_reverse(args, candidates)

# Collect forward results first
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

print(f"\nResult: {result.forward_survivors} forward, {result.verified_survivors} verified")
