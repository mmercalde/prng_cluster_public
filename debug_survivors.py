import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'
optimizer = WindowOptimizer(coordinator, test_seeds=100000)

# Patch to capture forward survivors
original_eval = optimizer.evaluate_window
def debug_eval(prng, window, use_all_gpus):
    result = original_eval(prng, window, use_all_gpus)
    return result

result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=False)

print(f"\nForward: {result.forward_survivors}")
print(f"Verified: {result.verified_survivors}")
print(f"\nMissing: {result.forward_survivors - result.verified_survivors} survivors")
