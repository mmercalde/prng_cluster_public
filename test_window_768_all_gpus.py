#!/usr/bin/env python3
"""
TEST: Window 768 with ALL 26 GPUs (Forward + Reverse Sieves)
Expected: 25 forward survivors, 25 verified survivors
"""
import sys
sys.path.insert(0, '.')

from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

print("="*70)
print("FULL TEST: Window 768 with ALL 26 GPUs")
print("="*70)
print("Test Configuration:")
print("  PRNG: lcg32")
print("  Window: 768")
print("  Seeds: 1,000,000")
print("  GPUs: ALL (2 local + 24 remote)")
print("  Forward sieve: YES")
print("  Reverse sieve: YES")
print("  Expected: 25 forward → 25 verified")
print("="*70)

# Initialize coordinator
coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Create optimizer
optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)

# Run test with ALL GPUs
print("\nExecuting test...\n")
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

# Display results
print("\n" + "="*70)
print("RESULTS:")
print("="*70)
print(f"  Window size: {result.window_size}")
print(f"  Forward survivors: {result.forward_survivors}")
print(f"  Verified survivors: {result.verified_survivors}")

if result.forward_survivors > 0:
    verification_rate = (result.verified_survivors / result.forward_survivors) * 100
    print(f"  Verification rate: {verification_rate:.1f}%")
else:
    print(f"  Verification rate: N/A (no forward survivors)")

print(f"  Runtime: {result.runtime:.1f}s")
print(f"  Signal strength: {result.signal_strength:.2f}")
print("="*70)

# Analysis
print("\nANALYSIS:")
if result.forward_survivors == 25 and result.verified_survivors == 25:
    print("  ✅ PERFECT! Got expected 25/25")
elif result.verified_survivors == result.forward_survivors and result.verified_survivors > 0:
    print(f"  ✅ Reverse sieve working (100% verification)")
    print(f"  ⚠️  But only found {result.forward_survivors} forward survivors (expected 25)")
    print(f"  → This suggests not all jobs executed or seed space incomplete")
elif result.forward_survivors > 0 and result.verified_survivors == 0:
    print(f"  ❌ Reverse sieve FAILED")
    print(f"  → Found {result.forward_survivors} forward survivors but verified 0")
else:
    print(f"  ❌ Something went wrong")
    print(f"  → Check logs above for errors")

print("="*70)
