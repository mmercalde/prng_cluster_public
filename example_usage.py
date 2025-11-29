#!/usr/bin/env python3
"""
Example: Score Your Bidirectional Survivors Against Real Lottery Data

This script shows how to use survivor_scorer.py with your actual lottery data
and bidirectional survivors from your window optimization.
"""

from survivor_scorer import SurvivorScorer
import json
from pathlib import Path

print("=" * 80)
print("EXAMPLE: Score Real Survivors Against Lottery Data")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load Your Lottery Data
# ============================================================================
print("STEP 1: Loading lottery data...")

# Option A: Load from JSON file (like daily3.json)
lottery_file = 'daily3.json'

if Path(lottery_file).exists():
    with open(lottery_file) as f:
        data = json.load(f)
    
    # Extract just the draw numbers
    if isinstance(data, list):
        if isinstance(data[0], dict):
            lottery_history = [entry['draw'] for entry in data]
        else:
            lottery_history = data
    
    print(f"✅ Loaded {len(lottery_history)} draws from {lottery_file}")
    print(f"   First 5 draws: {lottery_history[:5]}")
    print(f"   Last 5 draws:  {lottery_history[-5:]}")
else:
    # Option B: Manual data entry (if you don't have daily3.json)
    print("⚠️  daily3.json not found, using sample data...")
    lottery_history = [
        767, 475, 904, 134, 840, 994, 303, 618, 450,
        # ... add your actual lottery draws here
    ]
    print(f"✅ Using {len(lottery_history)} sample draws")

print()

# ============================================================================
# STEP 2: Configure Scorer
# ============================================================================
print("STEP 2: Configuring scorer...")

# Create scorer with your PRNG type
prng_type = 'java_lcg'  # Change this to match your analysis
                        # Options: java_lcg, xorshift32, pcg32, mt19937, etc.

config = {
    'max_offset': 100,        # Test up to 100 offset positions
    'gap_tolerance': 5,       # Allow up to 5 gaps in sequence
    'near_miss_credit': True, # Give credit for near misses
    'window_size': None,      # None = use all data, or set to 512 for rolling windows
}

scorer = SurvivorScorer(prng_type=prng_type, config=config)
print(f"✅ Scorer configured for {prng_type}")
print(f"   Max offset: {config['max_offset']}")
print(f"   Gap tolerance: {config['gap_tolerance']}")
print()

# ============================================================================
# STEP 3: Score Your Bidirectional Survivors
# ============================================================================
print("STEP 3: Scoring bidirectional survivors...")

# Replace these with YOUR actual bidirectional survivors from window optimization
bidirectional_survivors = [
    20045392,  # Example seed - replace with your actual survivors
    # Add more survivors here if you have them:
    # 30012345,
    # 40067890,
]

print(f"Testing {len(bidirectional_survivors)} survivors:")
for seed in bidirectional_survivors:
    print(f"   - {seed}")

print()

# Score all survivors
results = scorer.batch_score_survivors(
    bidirectional_survivors, 
    lottery_history,
    top_k=None  # Return all results, sorted by score
)

print()

# ============================================================================
# STEP 4: Display Results
# ============================================================================
print("STEP 4: Results")
print("=" * 80)
print()
print(f"{'Rank':<6} {'Seed':<15} {'Score %':<10} {'Exact':<8} {'Partial':<9} {'Conf':<8} {'Offset':<8}")
print("-" * 80)

for r in results:
    print(f"{r['rank']:<6} {r['seed']:<15} {r['score_percent']:<10.2f} "
          f"{r['exact_matches']:<8} {r['partial_matches']:<9} "
          f"{r['confidence']:<8.1f} {r['best_offset']:<8}")

print()

# ============================================================================
# STEP 5: Detailed Analysis of Best Survivor
# ============================================================================
print("STEP 5: Detailed Analysis of Best Survivor")
print("=" * 80)

best_survivor = results[0]
print()
print(f"🏆 BEST SURVIVOR: {best_survivor['seed']}")
print()
print(f"📊 SCORING METRICS:")
print(f"   Overall Score:        {best_survivor['score_percent']:.2f}%")
print(f"   Confidence:           {best_survivor['confidence']:.1f}/100")
print(f"   Alignment Quality:    {best_survivor['alignment_quality']:.2f}")
print()
print(f"📈 MATCH BREAKDOWN:")
print(f"   Exact Matches:        {best_survivor['exact_matches']}/{best_survivor['total_tested']}")
print(f"   Partial Matches:      {best_survivor['partial_matches']}")
print(f"   Match Rate:           {(best_survivor['exact_matches']/best_survivor['total_tested']*100):.2f}%")
print()
print(f"🔍 TECHNICAL DETAILS:")
print(f"   Best Offset:          {best_survivor['best_offset']}")
print(f"   Gaps Found:           {best_survivor['gaps_found']}")
print()

# Show ML features
print(f"🤖 ML FEATURES (for reinforcement learning):")
for feature_name, value in best_survivor['ml_features'].items():
    print(f"   {feature_name:<25} {value:.6f}")

print()

# ============================================================================
# STEP 6: Generate Predictions (If High Score)
# ============================================================================
if best_survivor['score_percent'] > 50.0:
    print("STEP 6: Generate Predictions")
    print("=" * 80)
    print()
    print(f"⚠️  Score is {best_survivor['score_percent']:.2f}% - HIGH CONFIDENCE!")
    print()
    print("To generate predictions for the next lottery draw:")
    print()
    print(f"from prng_registry import get_cpu_reference")
    print()
    print(f"# Get PRNG function")
    print(f"prng_func = get_cpu_reference('{prng_type}')")
    print()
    print(f"# Generate next 10 predictions")
    print(f"seed = {best_survivor['seed']}")
    print(f"offset = {best_survivor['best_offset']}")
    print(f"skip = {len(lottery_history) + offset}")
    print()
    print(f"predictions = [val % 1000 for val in prng_func(seed, 10, skip=skip)]")
    print(f"print('Next 10 predicted draws:', predictions)")
    print()
else:
    print("STEP 6: Score Analysis")
    print("=" * 80)
    print()
    print(f"ℹ️  Score is {best_survivor['score_percent']:.2f}% - MODERATE/LOW confidence")
    print()
    print("This survivor may not be the correct seed. Consider:")
    print("   1. Testing more bidirectional survivors")
    print("   2. Running window optimization with more seeds")
    print("   3. Trying different window configurations")
    print("   4. Using a different PRNG type")
    print()

# ============================================================================
# STEP 7: Export Results
# ============================================================================
print("STEP 7: Export Results")
print("=" * 80)
print()

# Export to JSON
scorer.export_results(results, 'survivor_scores.json', format='json')
print("✅ Results exported to: survivor_scores.json")

# Export to CSV
scorer.export_results(results, 'survivor_scores.csv', format='csv')
print("✅ Results exported to: survivor_scores.csv")

print()
print("=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
print()
print("📊 Summary:")
print(f"   - Tested {len(bidirectional_survivors)} survivors")
print(f"   - Against {len(lottery_history)} lottery draws")
print(f"   - Best score: {best_survivor['score_percent']:.2f}%")
print(f"   - Best seed: {best_survivor['seed']}")
print()
print("Next steps:")
print("   - Review survivor_scores.json for full details")
print("   - Open survivor_scores.csv in Excel/LibreOffice")
print("   - If score >50%, consider generating predictions")
print("   - If score <50%, test more survivors or different configurations")
print()
