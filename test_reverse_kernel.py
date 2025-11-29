#!/usr/bin/env python3
"""
Test reverse sieve kernels - FIXED to use raw MT19937 output
"""

import cupy as cp
import numpy as np
import sys

# Load the kernel
exec(open('reverse_kernels_addition.py').read())

# Import CPU reference from prng_registry
try:
    from prng_registry import mt19937_cpu
    print("✅ Using mt19937_cpu from prng_registry")
except ImportError:
    print("❌ Cannot import mt19937_cpu from prng_registry")
    sys.exit(1)


def test_mt19937_reverse_kernel():
    """Test MT19937 reverse kernel with known seed"""
    
    print("=" * 70)
    print("TESTING MT19937 REVERSE KERNEL")
    print("=" * 70)
    
    # Generate known sequence using SAME method as kernel
    known_seed = 12345
    skip = 5
    k = 30
    
    # Use CPU reference - this matches the GPU kernel EXACTLY
    total_needed = k * (skip + 1)
    all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)
    
    # Extract draws with skip pattern (same logic as kernel)
    draws = []
    idx = 0
    for i in range(k):
        idx += skip  # Skip
        draws.append(all_outputs[idx] % 1000)
        idx += 1
    
    print(f"\nKnown seed: {known_seed}")
    print(f"Skip: {skip}")
    print(f"Generated {len(draws)} draws: {draws[:10]}...")
    
    # Test if reverse kernel can validate this seed
    print(f"\nTesting reverse kernel validation...")
    
    # Compile kernel
    kernel = cp.RawKernel(MT19937_REVERSE_KERNEL, 'mt19937_reverse_sieve')
    
    # Prepare inputs
    candidate_seeds = cp.array([known_seed], dtype=cp.uint32)
    residues = cp.array(draws, dtype=cp.uint32)
    survivors = cp.zeros(1, dtype=cp.uint32)
    match_rates = cp.zeros(1, dtype=cp.float32)
    best_skips = cp.zeros(1, dtype=cp.uint8)
    survivor_count = cp.zeros(1, dtype=cp.uint32)
    
    # Launch kernel
    kernel(
        (1,), (1,),
        (candidate_seeds, residues, survivors, match_rates, best_skips,
         survivor_count, cp.int32(1), cp.int32(len(draws)),
         cp.int32(0), cp.int32(20), cp.float32(0.01), cp.int32(0))
    )
    
    # Check results
    count = int(survivor_count[0].get())
    
    if count > 0:
        seed_out = int(survivors[0].get())
        rate = float(match_rates[0].get())
        skip_out = int(best_skips[0].get())
        
        print(f"\n✅ KERNEL TEST PASSED!")
        print(f"   Survivor seed: {seed_out}")
        print(f"   Match rate: {rate:.3f} ({rate*100:.1f}%)")
        print(f"   Best skip: {skip_out}")
        
        if seed_out == known_seed and skip_out == skip and rate > 0.95:
            print(f"\n🎉 PERFECT MATCH - Kernel correctly validated the seed!")
            return True
        else:
            print(f"\n⚠️  PARTIAL - Found seed but skip/rate unexpected")
            print(f"   Expected skip: {skip}, got: {skip_out}")
            print(f"   Expected rate: >0.95, got: {rate:.3f}")
            return False
    else:
        print(f"\n❌ KERNEL TEST FAILED - No survivors found")
        return False


def test_wrong_seed():
    """Test that kernel rejects wrong seeds"""
    
    print("\n" + "=" * 70)
    print("TESTING WRONG SEED REJECTION")
    print("=" * 70)
    
    # Generate draws with seed 12345
    known_seed = 12345
    wrong_seed = 99999
    skip = 5
    k = 30
    
    total_needed = k * (skip + 1)
    all_outputs = mt19937_cpu(known_seed, total_needed, skip=0)
    
    draws = []
    idx = 0
    for i in range(k):
        idx += skip
        draws.append(all_outputs[idx] % 1000)
        idx += 1
    
    print(f"\nCorrect seed: {known_seed}")
    print(f"Testing wrong seed: {wrong_seed}")
    
    kernel = cp.RawKernel(MT19937_REVERSE_KERNEL, 'mt19937_reverse_sieve')
    
    candidate_seeds = cp.array([wrong_seed], dtype=cp.uint32)
    residues = cp.array(draws, dtype=cp.uint32)
    survivors = cp.zeros(1, dtype=cp.uint32)
    match_rates = cp.zeros(1, dtype=cp.float32)
    best_skips = cp.zeros(1, dtype=cp.uint8)
    survivor_count = cp.zeros(1, dtype=cp.uint32)
    
    kernel(
        (1,), (1,),
        (candidate_seeds, residues, survivors, match_rates, best_skips,
         survivor_count, cp.int32(1), cp.int32(len(draws)),
         cp.int32(0), cp.int32(20), cp.float32(0.90), cp.int32(0))  # High threshold
    )
    
    count = int(survivor_count[0].get())
    
    if count == 0:
        print(f"\n✅ CORRECT - Kernel rejected wrong seed")
        return True
    else:
        rate = float(match_rates[0].get())
        print(f"\n⚠️  Wrong seed survived with rate {rate:.3f}")
        if rate < 0.10:
            print(f"   (Low rate - likely noise, acceptable)")
            return True
        else:
            print(f"   ❌ ERROR - High match rate on wrong seed!")
            return False


def main():
    try:
        print("\n🔬 REVERSE KERNEL VALIDATION TEST")
        print("Using mt19937_cpu reference for test data generation\n")
        
        test1 = test_mt19937_reverse_kernel()
        test2 = test_wrong_seed()
        
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Known seed validation: {'✅ PASS' if test1 else '❌ FAIL'}")
        print(f"Wrong seed rejection: {'✅ PASS' if test2 else '❌ FAIL'}")
        
        if test1 and test2:
            print(f"\n🎉 ALL TESTS PASSED - Kernel is ready for integration!")
            return 0
        else:
            print(f"\n⚠️  TESTS FAILED - Kernel needs debugging")
            return 1
            
    except Exception as e:
        print(f"\n💥 TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
