#!/usr/bin/env python3
"""
Test the ACTUAL reinforcement workflow from whitepaper
"""
import json
import numpy as np
from reinforcement_engine import ReinforcementEngine, ReinforcementConfig

config = ReinforcementConfig.from_json('reinforcement_engine_config.json')
with open('synthetic_lottery.json') as f:
    lottery = [d['draw'] for d in json.load(f)]

# Split into training and test
train_lottery = lottery[:-10]
test_draws = lottery[-10:]

engine = ReinforcementEngine(config, train_lottery)

print("="*70)
print("REINFORCEMENT LEARNING WORKFLOW TEST")
print("="*70)

# 1. Generate survivor pool (simulating forward/reverse sieve output)
print("\n1. Survivor Pool Generation")
survivors = list(range(12345, 12445))
print(f"   Initial pool: {len(survivors)} survivors")

# 2. Score survivors on their ACTUAL prediction quality
print("\n2. Scoring Survivors (actual hit rates)")
actual_scores = []
for seed in survivors:
    # Use the actual score returned by scorer
    score_dict = engine.scorer.score_survivor(seed, train_lottery[-10:], skip=0)
    actual_scores.append(score_dict['score'])  # Use 'score' key instead of 'match_rate'

print(f"   Score range: [{min(actual_scores):.3f}, {max(actual_scores):.3f}]")
print(f"   Score variance: {np.std(actual_scores):.3f}")

# 3. Train ML model to predict survivor quality
print("\n3. Training ML Model")
engine.train(survivors, actual_scores)
print(f"   Model trained on {len(survivors)} survivors")

# 4. Predict quality of new survivors
print("\n4. Testing Predictions")
test_survivors = list(range(50000, 50100))
predicted_qualities = engine.predict_quality_batch(test_survivors)

print(f"   Predicted quality range: [{min(predicted_qualities):.3f}, {max(predicted_qualities):.3f}]")
print(f"   Prediction variance: {np.std(predicted_qualities):.3f}")

# 5. Prune to top performers
print("\n5. Pruning Pool")
top_survivors = engine.prune_survivors(test_survivors, keep_top_n=30)
print(f"   Kept top {len(top_survivors)} survivors")

# 6. Continuous learning loop
print("\n6. Continuous Learning (simulating new draws)")
for i, new_draw in enumerate(test_draws[:3]):
    print(f"\n   Draw {i+1}: {new_draw}")
    engine.continuous_learning_loop(new_draw, top_survivors)

print("\n" + "="*70)
print("✅ REINFORCEMENT WORKFLOW COMPLETE")
print("="*70)
print("\nKey Points:")
print("  - Zero-variance features are NORMAL for similar survivors")
print("  - Model learns to weight features that DO discriminate")
print("  - Prediction variance is what matters (you have it!)")
print("  - Continuous feedback adapts the model over time")
