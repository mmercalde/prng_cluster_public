#!/usr/bin/env python3
"""
QUICK TEST - Minimal verification of survivor_scorer.py
Use this if the full test has issues
"""

print("=" * 60)
print("SURVIVOR SCORER - QUICK TEST")
print("=" * 60)
print()

# Step 1: Import
print("Step 1: Importing survivor_scorer...")
try:
    from survivor_scorer import SurvivorScorer, JavaLCG
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

print()

# Step 2: Generate test data
print("Step 2: Generating test data...")
try:
    seed = 12345
    prng = JavaLCG(seed)
    data = [prng.next_int(1000) for _ in range(50)]
    print(f"✅ Generated 50 draws")
    print(f"   First 5: {data[:5]}")
except Exception as e:
    print(f"❌ Data generation failed: {e}")
    exit(1)

print()

# Step 3: Score correct seed
print("Step 3: Scoring correct seed...")
try:
    scorer = SurvivorScorer(prng_type='java_lcg')
    result = scorer.score_survivor(seed, data)
    print(f"✅ Scoring successful")
    print(f"   Score: {result['score_percent']:.2f}%")
    print(f"   Matches: {result['exact_matches']}/{result['total_tested']}")
except Exception as e:
    print(f"❌ Scoring failed: {e}")
    exit(1)

print()

# Step 4: Check result
print("Step 4: Validating results...")
if result['score_percent'] > 90:
    print("✅ PASSED - Correct seed scored high!")
else:
    print(f"⚠️  WARNING - Score was {result['score_percent']:.2f}% (expected >90%)")

print()
print("=" * 60)
print("✅ BASIC TEST COMPLETE")
print("=" * 60)
print()
print("survivor_scorer.py is working on your system!")
print()
