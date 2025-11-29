import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

# Create coordinator but override to use ONLY localhost
coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Filter out remote nodes - keep ONLY localhost
coordinator.gpu_workers = [w for w in coordinator.gpu_workers if w.node.hostname == 'localhost']
print(f"Using {len(coordinator.gpu_workers)} LOCAL GPUs only\n")

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
optimizer.cache = {}

result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

print(f"\n{'='*60}")
print(f"LOCALHOST ONLY RESULTS:")
print(f"  Forward: {result.forward_survivors}")
print(f"  Verified: {result.verified_survivors}")
print(f"  Rate: {result.verified_survivors/result.forward_survivors*100:.1f}%")
print(f"{'='*60}")
