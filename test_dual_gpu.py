#!/usr/bin/env python3
"""
Test Script for Dual GPU Survivor Scorer
Demonstrates native dual GPU support in survivor_scorer.py
"""

import json
from survivor_scorer import SurvivorScorer

def test_dual_gpu():
    print("=" * 80)
    print("DUAL GPU SURVIVOR SCORER - TEST")
    print("=" * 80)
    
    # Initialize scorer
    scorer = SurvivorScorer(prng_type='java_lcg', mod=1000)
    
    # Generate test data
    print("\n📊 Generating test data...")
    test_seed = 42424242
    test_history = scorer._generate_predictions(test_seed, 1000, skip=0)
    print(f"✅ Generated {len(test_history)} test lottery draws")
    
    # Create test seeds (simulate 510K survivors with smaller sample)
    print(f"\n📊 Creating test survivor seeds...")
    import random
    random.seed(42)
    test_seeds = [random.randint(1, 2**32-1) for _ in range(10000)]
    print(f"✅ Created {len(test_seeds):,} test seeds")
    
    # Test 1: Dual GPU mode (default)
    print("\n" + "=" * 80)
    print("TEST 1: DUAL GPU MODE (use_dual_gpu=True)")
    print("=" * 80)
    
    import time
    start = time.time()
    results_dual = scorer.batch_score(test_seeds, test_history, use_dual_gpu=True)
    elapsed_dual = time.time() - start
    
    print(f"\n✅ Dual GPU completed in {elapsed_dual:.2f}s")
    print(f"   Top seed: {results_dual[0]['seed']} with score {results_dual[0]['score']:.2f}%")
    
    # Test 2: Single GPU mode
    print("\n" + "=" * 80)
    print("TEST 2: SINGLE GPU MODE (use_dual_gpu=False)")
    print("=" * 80)
    
    start = time.time()
    results_single = scorer.batch_score(test_seeds, test_history, use_dual_gpu=False)
    elapsed_single = time.time() - start
    
    print(f"\n✅ Single GPU completed in {elapsed_single:.2f}s")
    print(f"   Top seed: {results_single[0]['seed']} with score {results_single[0]['score']:.2f}%")
    
    # Compare
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"Single GPU time: {elapsed_single:.2f}s")
    print(f"Dual GPU time:   {elapsed_dual:.2f}s")
    
    if elapsed_single > elapsed_dual:
        speedup = elapsed_single / elapsed_dual
        print(f"🚀 Speedup:      {speedup:.2f}x faster with dual GPU!")
    else:
        print(f"⚠️  Single GPU was faster (test size may be too small)")
    
    # Verify results match
    same_top_seed = results_dual[0]['seed'] == results_single[0]['seed']
    print(f"\n✅ Results consistent: {same_top_seed}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)
    print("\n💡 Usage in your pipeline:")
    print("   scorer.batch_score(survivors, history, use_dual_gpu=True)  # Dual GPU")
    print("   scorer.batch_score(survivors, history, use_dual_gpu=False) # Single GPU")
    print("\n💡 CLI usage:")
    print("   python3 survivor_scorer.py --test --dual-gpu     # Enable dual GPU")
    print("   python3 survivor_scorer.py --test --no-dual-gpu  # Disable dual GPU")

if __name__ == "__main__":
    try:
        test_dual_gpu()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
