#!/usr/bin/env python3
"""
Debug script to diagnose why survivor_scorer.py isn't matching correctly
"""

import sys
from survivor_scorer import SurvivorScorer

# Test parameters
SEED = 42424242
NUM_DRAWS = 20  # Fewer draws to see what's happening

print("="*60)
print("SURVIVOR SCORER - DEBUG TEST")
print("="*60)

# Step 1: Generate test data using Java LCG
print("\nStep 1: Generating test data with Java LCG...")
try:
    from prng_registry import get_cpu_reference
    java_lcg = get_cpu_reference('java_lcg')
    
    # Generate draws
    draws = []
    state = SEED
    print(f"Initial seed: {state}")
    
    for i in range(NUM_DRAWS):
        state, value = java_lcg(state, skip=1)
        draw = value % 1000
        draws.append(draw)
        if i < 5:
            print(f"  Draw {i+1}: state={state}, value={value}, draw={draw}")
    
    print(f"\n✅ Generated {len(draws)} draws")
    print(f"   First 10: {draws[:10]}")
    
except Exception as e:
    print(f"❌ Error generating test data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Initialize scorer
print("\nStep 2: Initializing survivor_scorer...")
try:
    scorer = SurvivorScorer(prng_type='java_lcg', mod=1000)
    print("✅ Scorer initialized")
except Exception as e:
    print(f"❌ Error initializing scorer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Score with the same seed
print(f"\nStep 3: Scoring with seed {SEED}...")
try:
    result = scorer.score_survivor(
        seed=SEED,
        lottery_history=draws,
        max_draws=NUM_DRAWS
    )
    
    print(f"✅ Scoring complete")
    print(f"\n📊 Results:")
    print(f"   Score: {result['score']:.2f}%")
    print(f"   Matches: {result['exact_matches']}/{result['total_predictions']}")
    print(f"   Confidence: {result['confidence']:.1f}/100")
    
    # Show details
    if 'details' in result:
        details = result['details']
        print(f"\n📋 Details:")
        print(f"   Predictions made: {len(details['predictions'])}")
        print(f"   Actuals received: {len(details['actuals'])}")
        
        # Show first 10 comparisons
        print(f"\n🔍 First 10 Predictions vs Actuals:")
        print(f"   {'Index':<6} {'Predicted':<10} {'Actual':<10} {'Match':<6}")
        print(f"   {'-'*6} {'-'*10} {'-'*10} {'-'*6}")
        
        for i in range(min(10, len(details['predictions']))):
            pred = details['predictions'][i]
            actual = details['actuals'][i] if i < len(details['actuals']) else 'N/A'
            match = '✓' if pred == actual else '✗'
            print(f"   {i:<6} {pred:<10} {actual:<10} {match:<6}")
    
except Exception as e:
    print(f"❌ Error during scoring: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Verify PRNG is working correctly
print(f"\n" + "="*60)
print("Step 4: Direct PRNG verification...")
print("="*60)

try:
    # Generate a few values directly
    print(f"\nDirect generation from seed {SEED}:")
    state = SEED
    for i in range(5):
        state, value = java_lcg(state, skip=1)
        draw = value % 1000
        print(f"  Step {i+1}: state={state}, value={value}, draw={draw}")
    
    # Now let scorer generate
    print(f"\nScorer's predictions from seed {SEED}:")
    scorer_result = scorer.score_survivor(
        seed=SEED,
        lottery_history=draws[:5],  # Just first 5
        max_draws=5
    )
    
    if 'details' in scorer_result:
        preds = scorer_result['details']['predictions']
        print(f"  Scorer predictions: {preds[:5]}")
    
    print("\n🔍 Analysis:")
    print(f"  Expected draws: {draws[:5]}")
    print(f"  Scorer predictions: {preds[:5] if 'details' in scorer_result else 'N/A'}")
    
    if draws[:5] == preds[:5]:
        print("  ✅ MATCH! Scorer is working correctly.")
    else:
        print("  ❌ MISMATCH! There's a discrepancy.")
        print("\n  Possible issues:")
        print("  1. Different PRNG implementation")
        print("  2. Different skip pattern")
        print("  3. Different modulo operation")
        print("  4. State initialization difference")
    
except Exception as e:
    print(f"❌ Error in verification: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DEBUG TEST COMPLETE")
print("="*60)
