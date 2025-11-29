import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

print("="*70)
print("FULL TEST: 1M seeds, Window 768, ALL GPUs")
print("="*70)

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

print("\n" + "="*70)
print("RESULTS:")
print(f"  Forward survivors: {result.forward_survivors}")
print(f"  Verified survivors: {result.verified_survivors}")
vr = result.verified_survivors / result.forward_survivors if result.forward_survivors > 0 else 0
print(f"  Verification rate: {vr*100:.1f}%")
print("="*70)
print(f"\nExpected: 25 forward, 25 verified")
