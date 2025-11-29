#!/usr/bin/env python3
"""
Examine EXACTLY what ML/AI features are returned
Let's see the actual data structure and values
"""
import json
import sys
sys.path.insert(0, '.')

from survivor_scorer import SurvivorScorer

print("=" * 70)
print("EXAMINING ML/AI FEATURE OUTPUT")
print("=" * 70)

# Initialize scorer
scorer = SurvivorScorer(prng_type='java_lcg', mod=1000)

# Create small test dataset
print("\n[Step 1] Creating test data...")
test_seed = 42424242
test_draws = scorer._generate_predictions(test_seed, 100, skip=0)
print(f"✅ Created {len(test_draws)} test draws")

# Simulate forward/reverse survivors
forward_survivors = [test_seed, 12345, 67890]
reverse_survivors = [test_seed, 11111, 67890]

print("\n[Step 2] Extracting ML features...")
features = scorer.extract_ml_features(
    seed=test_seed,
    lottery_history=test_draws,
    forward_survivors=forward_survivors,
    reverse_survivors=reverse_survivors,
    skip=0
)

print(f"✅ Extracted {len(features)} features")

# Display EVERY feature with its value
print("\n" + "=" * 70)
print("COMPLETE FEATURE SET (ALL 46 FEATURES)")
print("=" * 70)

# Sort by category
categories = {
    'Basic Scoring': [],
    'Residue Coherence': [],
    'Skip Entropy': [],
    'Temporal Stability': [],
    'Survivor Velocity': [],
    'Intersection Weights': [],
    'Lane Agreement': [],
    'Statistical': []
}

for key, value in sorted(features.items()):
    if key in ['score', 'exact_matches', 'total_predictions', 'confidence', 'best_offset']:
        categories['Basic Scoring'].append((key, value))
    elif 'residue' in key:
        categories['Residue Coherence'].append((key, value))
    elif 'skip' in key:
        categories['Skip Entropy'].append((key, value))
    elif 'temporal' in key:
        categories['Temporal Stability'].append((key, value))
    elif 'velocity' in key or 'acceleration' in key:
        categories['Survivor Velocity'].append((key, value))
    elif 'intersection' in key or 'forward_count' in key or 'reverse_count' in key or 'overlap' in key:
        categories['Intersection Weights'].append((key, value))
    elif 'lane' in key:
        categories['Lane Agreement'].append((key, value))
    else:
        categories['Statistical'].append((key, value))

for category, items in categories.items():
    if items:
        print(f"\n{category} ({len(items)} features):")
        print("-" * 70)
        for key, value in items:
            if isinstance(value, float):
                print(f"  {key:30s}: {value:10.4f}")
            else:
                print(f"  {key:30s}: {value}")

# Export to JSON for ML training
output_file = 'ml_features_sample.json'
with open(output_file, 'w') as f:
    json.dump(features, f, indent=2)

print("\n" + "=" * 70)
print("DATA STRUCTURE FOR ML/AI")
print("=" * 70)
print(f"\n✅ Saved complete feature set to: {output_file}")
print(f"\nFeature vector shape: {len(features)} features")
print(f"All numeric: {all(isinstance(v, (int, float)) for v in features.values())}")

# Show what PyTorch would receive
print("\n" + "=" * 70)
print("PYTORCH INPUT FORMAT")
print("=" * 70)
print("\nFeature names (sorted for consistency):")
feature_names = sorted(features.keys())
for i, name in enumerate(feature_names):
    print(f"  {i:2d}. {name}")

print("\nFeature values (as tensor input):")
feature_vector = [features[k] for k in feature_names]
print(f"  Vector length: {len(feature_vector)}")
print(f"  First 10 values: {[f'{v:.4f}' if isinstance(v, float) else v for v in feature_vector[:10]]}")

print("\n" + "=" * 70)
print("CRITICAL QUESTIONS FOR ML/AI")
print("=" * 70)
print("\n1. Are all features numeric? ", "YES" if all(isinstance(v, (int, float)) for v in features.values()) else "NO")
print("2. Any NaN or Inf values? ", "YES" if any(str(v) in ['nan', 'inf', '-inf'] for v in features.values()) else "NO")
print("3. Feature ranges reasonable? Check above")
print("4. All features present? ", "YES" if len(features) >= 46 else f"NO - only {len(features)}")

# Test with WRONG seed to compare
print("\n" + "=" * 70)
print("COMPARISON: CORRECT vs WRONG SEED FEATURES")
print("=" * 70)

wrong_features = scorer.extract_ml_features(
    seed=99999999,  # Wrong seed
    lottery_history=test_draws,
    forward_survivors=forward_survivors,
    reverse_survivors=reverse_survivors,
    skip=0
)

print("\nKey differences (Correct vs Wrong):")
print("-" * 70)
key_features = ['score', 'residue_8_coherence', 'intersection_weight', 'temporal_stability_mean']
for key in key_features:
    correct_val = features.get(key, 0)
    wrong_val = wrong_features.get(key, 0)
    diff = abs(correct_val - wrong_val)
    print(f"{key:30s}: {correct_val:8.4f} vs {wrong_val:8.4f}  (diff: {diff:8.4f})")

print("\n✅ Feature extraction complete!")
print(f"✅ Data ready for ML/AI training")
print(f"✅ Check {output_file} for complete JSON output")
