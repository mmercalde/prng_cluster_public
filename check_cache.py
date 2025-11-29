import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)

# Clear cache
optimizer.cache = {}
print("Cache cleared")

# Run once
result1 = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
print(f"Run 1: Forward={result1.forward_survivors}, Verified={result1.verified_survivors}")

# Check cache
print(f"Cache size: {len(optimizer.cache)}")

# Run again (should use cache)
result2 = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
print(f"Run 2: Forward={result2.forward_survivors}, Verified={result2.verified_survivors}")
