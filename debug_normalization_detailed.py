#!/usr/bin/env python3
"""
Detailed normalization debugging
"""
import json
import numpy as np
from reinforcement_engine import ReinforcementEngine, ReinforcementConfig

config = ReinforcementConfig.from_json('reinforcement_engine_config.json')
with open('synthetic_lottery.json') as f:
    lottery = [d['draw'] for d in json.load(f)]

engine = ReinforcementEngine(config, lottery)

# Train to fit scaler
print("Training to fit scaler...")
survivors = list(range(12345, 12445))
scores = [0.5 + (i % 10) * 0.05 for i in range(100)]
engine.train(survivors, scores)

print("\n" + "="*70)
print("DETAILED NORMALIZATION DIAGNOSTICS")
print("="*70)

# Check scaler parameters
print("\n1. Scaler Parameters:")
print(f"   Scaler fitted: {engine.scaler_fitted}")
print(f"   Total features: {len(engine.feature_scaler.scale_)}")

# Find zero-variance features
zero_var_mask = engine.feature_scaler.scale_ == 1.0
zero_var_indices = np.where(zero_var_mask)[0]

print(f"\n2. Zero-Variance Features:")
print(f"   Count: {np.sum(zero_var_mask)}")
print(f"   Indices: {zero_var_indices.tolist()}")

if np.sum(zero_var_mask) > 0:
    print(f"\n   Details of zero-variance features:")
    for idx in zero_var_indices[:5]:  # Show first 5
        print(f"     Feature {idx}: mean={engine.feature_scaler.mean_[idx]:.2f}, scale={engine.feature_scaler.scale_[idx]}")

# Extract features for a test seed WITHOUT normalization temporarily
print(f"\n3. Testing Feature Extraction (seed=12345):")

# Temporarily disable normalization to get raw features
temp_fitted = engine.scaler_fitted
engine.scaler_fitted = False
raw_features = engine.extract_combined_features(12345, None, None)
engine.scaler_fitted = temp_fitted

print(f"   Raw features (first 10): {raw_features[:10]}")
print(f"   Raw mean: {raw_features.mean():.4f}")
print(f"   Raw std: {raw_features.std():.4f}")

# Now get normalized features
normalized_features = engine.extract_combined_features(12345, None, None)
print(f"\n4. Normalized features (first 10): {normalized_features[:10]}")
print(f"   Normalized mean: {normalized_features.mean():.4f}")
print(f"   Normalized std: {normalized_features.std():.4f}")

# Check what happened to zero-variance features specifically
if np.sum(zero_var_mask) > 0:
    print(f"\n5. Zero-Variance Features After Normalization:")
    for idx in zero_var_indices[:5]:
        print(f"   Feature {idx}:")
        print(f"     Raw value: {raw_features[idx]:.4f}")
        print(f"     Scaler mean: {engine.feature_scaler.mean_[idx]:.4f}")
        print(f"     Expected normalized: {raw_features[idx] - engine.feature_scaler.mean_[idx]:.4f}")
        print(f"     Actual normalized: {normalized_features[idx]:.4f}")
        print(f"     Match: {'✅' if abs((raw_features[idx] - engine.feature_scaler.mean_[idx]) - normalized_features[idx]) < 0.01 else '❌'}")

# Check non-zero-variance features
non_zero_mask = ~zero_var_mask
non_zero_indices = np.where(non_zero_mask)[0]

print(f"\n6. Sample Non-Zero-Variance Features:")
for idx in non_zero_indices[:3]:
    expected = (raw_features[idx] - engine.feature_scaler.mean_[idx]) / engine.feature_scaler.scale_[idx]
    print(f"   Feature {idx}:")
    print(f"     Raw: {raw_features[idx]:.4f}")
    print(f"     Mean: {engine.feature_scaler.mean_[idx]:.4f}, Scale: {engine.feature_scaler.scale_[idx]:.4f}")
    print(f"     Expected: {expected:.4f}")
    print(f"     Actual: {normalized_features[idx]:.4f}")
    print(f"     Match: {'✅' if abs(expected - normalized_features[idx]) < 0.01 else '❌'}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

# Overall stats check
if abs(normalized_features.mean()) < 0.1 and abs(normalized_features.std() - 1.0) < 0.1:
    print("✅ Normalization is working correctly!")
    print(f"   Mean: {normalized_features.mean():.4f} (≈0)")
    print(f"   Std: {normalized_features.std():.4f} (≈1)")
else:
    print("❌ Normalization is NOT working correctly!")
    print(f"   Mean: {normalized_features.mean():.4f} (expected ≈0)")
    print(f"   Std: {normalized_features.std():.4f} (expected ≈1)")
    
    # Analyze why
    if np.sum(zero_var_mask) > 10:
        print(f"\n⚠️ Too many zero-variance features: {np.sum(zero_var_mask)}/{len(engine.feature_scaler.scale_)}")
        print("   This suggests features aren't varying across survivors")
