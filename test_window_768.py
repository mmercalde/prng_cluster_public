#!/usr/bin/env python3
"""
Quick test: Does window=768 find survivors like the original test?
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer
print("="*70)
print("VERIFICATION TEST: Window 768")
print("Should match original test results (25 survivors)")
print("="*70)
coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'
optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
print("\nTesting window=768 with ALL GPUs...")
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
print(f"\n{'='*70}")
print(f"RESULTS:")
print(f"  Window: 768")
print(f"  Forward survivors: {result.forward_survivors}")
print(f"  Verified survivors: {result.verified_survivors}")
print(f"  Verification rate: {result.verified_survivors/result.forward_survivors*100 if result.forward_survivors > 0 else 0:.1f}%")
print(f"  Runtime: {result.runtime:.1f}s")
print(f"  Signal strength: {result.signal_strength:.2f}")
print(f"{'='*70}")
if result.verified_survivors > 0:
    print("\n✅ SUCCESS! Ready to use full cluster")
else:
    print("\n❌ PROBLEM: No verified survivors found")
print(f"\nOriginal: 25 survivors | This test: {result.verified_survivors}")
