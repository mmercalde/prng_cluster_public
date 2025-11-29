import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

print("="*70)
print("QUICK TEST: Window 512 with 100k seeds")
print("="*70)

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

optimizer = WindowOptimizer(coordinator, test_seeds=100_000)
result = optimizer.evaluate_window('lcg32', 512, use_all_gpus=True)

print("\n" + "="*70)
print("RESULTS:")
print(f"  Window: 512")
print(f"  Forward survivors: {result.forward_survivors}")
print(f"  Verified survivors: {result.verified_survivors}")
vr = result.verified_survivors / result.forward_survivors if result.forward_survivors > 0 else 0
print(f"  Verification rate: {vr*100:.1f}%")
print(f"  Runtime: {result.runtime:.1f}s")
print("="*70)
