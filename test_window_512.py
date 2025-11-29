import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

print("="*70)
print("VERIFICATION TEST: Window 512")
print("="*70)

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 512, use_all_gpus=True)

print("\n" + "="*70)
print("RESULTS:")
print(f"  Window: 512")
print(f"  Forward survivors: {result['forward_survivors']}")
print(f"  Verified survivors: {result['verified_survivors']}")
print(f"  Verification rate: {result['verification_rate']*100:.1f}%")
print(f"  Runtime: {result['runtime']:.1f}s")
print(f"  Signal strength: {result['signal_strength']:.2f}")
print("="*70)
