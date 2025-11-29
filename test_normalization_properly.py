#!/usr/bin/env python3
"""
Test normalization is working correctly
"""

import json
import numpy as np
from reinforcement_engine import ReinforcementEngine, ReinforcementConfig

print("=" * 70)
print("NORMALIZATION VALIDATION TEST")
print("=" * 70)

# Load config and data
config = ReinforcementConfig.from_json('reinforcement_engine_config.json')
with open('synthetic_lottery.json') as f:
    lottery = json.load(f)
    lottery_nums = [d['draw'] for d in lottery]

# Initialize engine
engine = ReinforcementEngine(config, lottery_nums)

# Create test survivors
survivors = list(range(12345, 12445))
scores = [0.5 + (i % 10) * 0.05 for i in range(100)]

print("\n1. Before Training:")
print(f"   Scaler fitted: {engine.scaler_fitted}")

# Extract raw features before training
test_seed = 12345
raw_features_before = engine.extract_combined_features(test_seed, None, None)
print(f"   Raw feature sample: mean={raw_features_before.mean():.2f}, std={raw_features_before.std():.2f}")

# Train (this triggers normalization)
print("\n2. Training...")
engine.train(survivors, scores)

print("\n3. After Training:")
print(f"   Scaler fitted: {engine.scaler_fitted}")

# Check scaler statistics
if engine.scaler_fitted:
    print(f"   Scaler mean (first 5): {engine.feature_scaler.mean_[:5]}")
    print(f"   Scaler std (first 5): {engine.feature_scaler.scale_[:5]}")

# Extract features again - these SHOULD be normalized
normalized_features = engine.extract_combined_features(test_seed, None, None)
print(f"\n4. Normalized Features Check:")
print(f"   Mean: {normalized_features.mean():.4f} (should be ~0)")
print(f"   Std: {normalized_features.std():.4f} (should be ~1)")
print(f"   Range: [{normalized_features.min():.2f}, {normalized_features.max():.2f}]")

# Test predictions
print(f"\n5. Prediction Variance:")
test_seeds = list(range(99999, 99999+20))
predictions = [engine.predict_quality(s) for s in test_seeds]
print(f"   Mean: {np.mean(predictions):.4f}")
print(f"   Std: {np.std(predictions):.4f}")
print(f"   Range: [{min(predictions):.4f}, {max(predictions):.4f}]")

# Final verdict
print("\n" + "=" * 70)
if abs(normalized_features.mean()) < 0.5 and 0.5 < normalized_features.std() < 1.5:
    print("✅ NORMALIZATION WORKING CORRECTLY!")
else:
    print("⚠️ Normalization may need adjustment")
    print(f"   Expected: mean≈0, std≈1")
    print(f"   Got: mean={normalized_features.mean():.4f}, std={normalized_features.std():.4f}")

if np.std(predictions) > 0.01:
    print("✅ PREDICTIONS HAVE GOOD VARIANCE!")
else:
    print("⚠️ Predictions are saturated")
    
print("=" * 70)
