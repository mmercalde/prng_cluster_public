#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

print("="*70)
print("FAST TEST: Window 768 with 100K seeds")
print("="*70)

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Only 100K seeds = MUCH faster
optimizer = WindowOptimizer(coordinator, test_seeds=100_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

print("\n" + "="*70)
print("RESULTS:")
print(f"  Forward survivors: {result.forward_survivors}")
print(f"  Verified survivors: {result.verified_survivors}")
if result.forward_survivors > 0:
    print(f"  Verification rate: {result.verified_survivors/result.forward_survivors*100:.1f}%")
print(f"  Runtime: {result.runtime:.1f}s")
print("="*70)

if result.verified_survivors == result.forward_survivors and result.forward_survivors > 0:
    print("\n✅ Reverse sieve is WORKING (100% verification)")
    print(f"   Found {result.forward_survivors} survivors with 100K seeds")
    print(f"   Extrapolated to 1M: ~{result.forward_survivors * 10} survivors expected")
else:
    print("\n❌ Something is wrong")
