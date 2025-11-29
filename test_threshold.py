import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Temporarily patch to use 0.009 threshold for reverse
original_eval = WindowOptimizer.evaluate_window
def patched_eval(self, prng, window, use_all_gpus=False):
    # Modify reverse threshold
    import types
    result = original_eval(self, prng, window, use_all_gpus)
    return result

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

print(f"Forward: {result.forward_survivors}, Verified: {result.verified_survivors}")
print(f"Rate: {result.verified_survivors/result.forward_survivors*100:.1f}%")
