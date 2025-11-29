import json
import numpy as np
from reinforcement_engine import ReinforcementEngine, ReinforcementConfig

config = ReinforcementConfig.from_json('reinforcement_engine_config.json')
with open('synthetic_lottery.json') as f:
    lottery = [d['draw'] for d in json.load(f)]

engine = ReinforcementEngine(config, lottery)

# Extract features from 10 survivors WITHOUT normalization
survivors = list(range(12345, 12355))
features_list = []

# Temporarily disable normalization to see raw features
engine.normalization_enabled = False
for seed in survivors:
    feats = engine.extract_combined_features(seed, None, None)
    features_list.append(feats)
    
features_array = np.array(features_list)

print("Raw Features (10 survivors, 60 features each):")
print(f"  Shape: {features_array.shape}")
print(f"  Mean per feature (first 10): {features_array.mean(axis=0)[:10]}")
print(f"  Std per feature (first 10): {features_array.std(axis=0)[:10]}")
print(f"  Overall mean: {features_array.mean():.2f}")
print(f"  Overall std: {features_array.std():.2f}")

# Now fit scaler manually
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(features_array)

print("\nScaler after fitting:")
print(f"  Mean (first 10): {scaler.mean_[:10]}")
print(f"  Std (first 10): {scaler.scale_[:10]}")

# Transform one sample
test_features = features_array[0]
normalized = scaler.transform([test_features])[0]

print(f"\nTest normalization:")
print(f"  Before: mean={test_features.mean():.2f}, std={test_features.std():.2f}")
print(f"  After: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
print(f"  After range: [{normalized.min():.2f}, {normalized.max():.2f}]")
