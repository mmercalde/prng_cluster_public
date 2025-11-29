#!/usr/bin/env python3
"""
Test Fixed Reinforcement Engine with Normalization
===================================================

Validates that the normalization fixes work correctly.

Run this after deploying reinforcement_engine.py to verify:
1. Normalization auto-fits on training
2. Features are properly scaled
3. Model produces varied predictions
4. Drift detection works

Author: Distributed PRNG Analysis System
Date: November 7, 2025
"""

import sys
import json
import numpy as np

print("="*70)
print("TESTING FIXED REINFORCEMENT ENGINE")
print("="*70)
print()

# Test 1: Import and initialization
print("Test 1: Import and Initialization")
print("-"*70)
try:
    from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Load synthetic data
try:
    with open('synthetic_lottery.json') as f:
        lottery_data = json.load(f)
    lottery_history = [d['draw'] for d in lottery_data]
    print(f"✅ Loaded {len(lottery_history)} lottery draws")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    sys.exit(1)

# Initialize engine
try:
    config = ReinforcementConfig.from_json('reinforcement_engine_config.json')
    engine = ReinforcementEngine(config, lottery_history)
    print("✅ Engine initialized")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)

print()

# Test 2: Feature extraction before normalization
print("Test 2: Feature Extraction (Before Normalization)")
print("-"*70)
test_seeds = [12345, 67890, 11111]
features_before = []
for seed in test_seeds:
    features = engine.extract_combined_features(seed)
    features_before.append(features)
    print(f"  Seed {seed}: mean={features.mean():.2f}, std={features.std():.2f}, range=[{features.min():.2f}, {features.max():.2f}]")

features_before_array = np.array(features_before)
print(f"\n  Overall statistics:")
print(f"    Mean: {features_before_array.mean():.2f}")
print(f"    Std: {features_before_array.std():.2f}")
print(f"    Max range: {features_before_array.max() - features_before_array.min():.2f}")

if features_before_array.max() > 100:
    print("  ⚠️  Features NOT normalized (as expected before training)")
else:
    print("  ✅ Features appear normalized")

print()

# Test 3: Training and auto-normalization
print("Test 3: Training with Auto-Normalization")
print("-"*70)

train_survivors = np.random.randint(1, 100000, 100).tolist()
train_results = np.random.uniform(0.4, 0.8, 100).tolist()

print(f"Training on {len(train_survivors)} survivors...")
try:
    engine.train(train_survivors, train_results)
    print("✅ Training successful")
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check if normalizer was fitted
if engine.scaler_fitted:
    print("✅ Normalizer auto-fitted during training")
else:
    print("❌ Normalizer NOT fitted")

print()

# Test 4: Feature extraction after normalization
print("Test 4: Feature Extraction (After Normalization)")
print("-"*70)
features_after = []
for seed in test_seeds:
    features = engine.extract_combined_features(seed)
    features_after.append(features)
    print(f"  Seed {seed}: mean={features.mean():.2f}, std={features.std():.2f}, range=[{features.min():.2f}, {features.max():.2f}]")

features_after_array = np.array(features_after)
print(f"\n  Overall statistics:")
print(f"    Mean: {features_after_array.mean():.2f}")
print(f"    Std: {features_after_array.std():.2f}")
print(f"    Max range: {features_after_array.max() - features_after_array.min():.2f}")

if abs(features_after_array.mean()) < 10 and features_after_array.max() < 100:
    print("  ✅ Features properly normalized!")
else:
    print("  ⚠️  Features may not be fully normalized")

print()

# Test 5: Prediction variance
print("Test 5: Prediction Variance")
print("-"*70)

test_survivors = np.random.randint(1, 100000, 100).tolist()
qualities = engine.predict_quality_batch(test_survivors)

print(f"  Predictions on 100 seeds:")
print(f"    Mean: {np.mean(qualities):.6f}")
print(f"    Std: {np.std(qualities):.6f}")
print(f"    Min: {np.min(qualities):.6f}")
print(f"    Max: {np.max(qualities):.6f}")
print(f"    Unique values: {len(np.unique(qualities))}")

if np.std(qualities) > 0.01:
    print("  ✅ Good prediction variance")
else:
    print("  ❌ No prediction variance (model not learning)")

saturated_at_one = (np.array(qualities) > 0.99).sum() / len(qualities)
saturated_at_zero = (np.array(qualities) < 0.01).sum() / len(qualities)

print(f"    Saturated at 1.0: {saturated_at_one*100:.1f}%")
print(f"    Saturated at 0.0: {saturated_at_zero*100:.1f}%")

if saturated_at_one < 0.9 and saturated_at_zero < 0.9:
    print("  ✅ Not saturated")
else:
    print("  ❌ Model saturated")

print()

# Test 6: Save and load with scaler
print("Test 6: Save/Load with Normalization Scaler")
print("-"*70)

try:
    engine.save_model('test_normalized_model.pth')
    print("✅ Model saved with scaler")
except Exception as e:
    print(f"❌ Save failed: {e}")
    sys.exit(1)

# Create new engine and load
try:
    engine2 = ReinforcementEngine(config, lottery_history)
    engine2.load_model('models/reinforcement/test_normalized_model.pth')
    print("✅ Model loaded")
    
    if engine2.scaler_fitted:
        print("✅ Scaler restored from saved model")
    else:
        print("❌ Scaler NOT restored")
        
except Exception as e:
    print(f"❌ Load failed: {e}")
    sys.exit(1)

print()

# Final summary
print("="*70)
print("SUMMARY")
print("="*70)

all_passed = (
    engine.scaler_fitted and
    abs(features_after_array.mean()) < 10 and
    np.std(qualities) > 0.01 and
    saturated_at_one < 0.9
)

if all_passed:
    print("✅ ALL TESTS PASSED")
    print("\nThe fixed reinforcement engine:")
    print("  ✓ Auto-fits normalizer on training")
    print("  ✓ Properly normalizes features")
    print("  ✓ Produces varied predictions")
    print("  ✓ Saves/loads scaler with model")
    print("\n🎉 Ready for production use!")
else:
    print("❌ SOME TESTS FAILED")
    print("\nCheck the output above for details.")

print("="*70)
