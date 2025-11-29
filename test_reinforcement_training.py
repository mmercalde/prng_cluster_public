#!/usr/bin/env python3
"""Test reinforcement engine training on synthetic data"""

from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
import json
import numpy as np

# Load synthetic data
print("Loading synthetic data...")
with open('synthetic_lottery.json') as f:
    lottery_data = json.load(f)
lottery_history = [d['draw'] for d in lottery_data]

# Load config and initialize
print("Initializing engine...")
config = ReinforcementConfig.from_json('reinforcement_engine_config.json')
engine = ReinforcementEngine(config, lottery_history)

print("✅ Engine initialized with synthetic data\n")

# Generate 1000 test survivors
print("Generating 1000 test survivors...")
test_survivors = np.random.randint(1, 100000, size=1000).tolist()

# Get initial predictions
print("Getting initial predictions...")
initial_qualities = engine.predict_quality_batch(test_survivors)

# Show top 10
top_10_idx = np.argsort(initial_qualities)[-10:][::-1]
print("\n🏆 TOP 10 SEEDS (before training):")
for i, idx in enumerate(top_10_idx, 1):
    print(f"  {i}. Seed {test_survivors[idx]:6d}: quality={initial_qualities[idx]:.4f}")

# Simulate training data (use top performers)
print("\n📚 Training on top 100 performers...")
train_survivors = [test_survivors[i] for i in np.argsort(initial_qualities)[-100:]]
train_results = np.random.uniform(0.6, 0.9, 100)  # Simulate good hit rates

# Train the model
print("Training model...")
engine.train(train_survivors, train_results)

# Re-evaluate
print("\n🔄 Re-scoring after training...")
trained_qualities = engine.predict_quality_batch(test_survivors)

# Show new top 10
new_top_10_idx = np.argsort(trained_qualities)[-10:][::-1]
print("\n🏆 TOP 10 SEEDS (after training):")
for i, idx in enumerate(new_top_10_idx, 1):
    print(f"  {i}. Seed {test_survivors[idx]:6d}: quality={trained_qualities[idx]:.4f}")

# Show improvement
print(f"\n📊 Quality score statistics:")
print(f"  Before training: mean={np.mean(initial_qualities):.4f}, std={np.std(initial_qualities):.4f}")
print(f"  After training:  mean={np.mean(trained_qualities):.4f}, std={np.std(trained_qualities):.4f}")

# Save the trained model
print("\n💾 Saving trained model...")
engine.save_model('synthetic_trained_model.pth')

print("\n✅ Training test complete!")
