#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/michael/distributed_prng_analysis')

from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
import json

print("✅ Testing imports...")

# Test 1: Load data
with open('bidirectional_survivors.json') as f:
    survivors = json.load(f)[:100]  # Just 100 for speed
print(f"✅ Loaded {len(survivors)} survivors")

# Test 2: Create config
config = ReinforcementConfig()
print("✅ Config created")

# Test 3: Create engine
with open('train_history.json') as f:
    history = json.load(f)
engine = ReinforcementEngine(config, history)
print("✅ Engine created")

# Test 4: Train for 2 epochs
engine.train([s if isinstance(s, int) else s['seed'] for s in survivors[:50]], 
             [0.5] * 50, 
             epochs=2)
print("✅ Training works!")
