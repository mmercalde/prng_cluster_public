#!/usr/bin/env python3
"""
Quick Fix: Generate Proper Survivor Files
Creates survivor files in the correct format with realistic test data
"""

import json
import random

print("="*80)
print("GENERATING PROPER SURVIVOR FILES")
print("="*80)

# Generate forward survivors
forward = {
    'survivors': [
        {
            'seed': random.randint(0, 1000000),
            'score': random.uniform(0.3, 0.9),
            'exact_matches': random.randint(10, 50),
            'total_predictions': random.randint(100, 500)
        }
        for _ in range(150)
    ],
    'metadata': {
        'total_tested': 1000000,
        'threshold': 0.01,
        'direction': 'forward'
    }
}

# Generate reverse survivors
reverse = {
    'survivors': [
        {
            'seed': random.randint(0, 1000000),
            'score': random.uniform(0.3, 0.9),
            'exact_matches': random.randint(10, 50),
            'total_predictions': random.randint(100, 500)
        }
        for _ in range(140)
    ],
    'metadata': {
        'total_tested': 1000000,
        'threshold': 0.01,
        'direction': 'reverse'
    }
}

# Generate bidirectional (overlap)
forward_seeds = [s['seed'] for s in forward['survivors']]
overlap_seeds = random.sample(forward_seeds, 45)

bidirectional = {
    'survivors': [
        {
            'seed': seed,
            'forward_score': random.uniform(0.3, 0.9),
            'reverse_score': random.uniform(0.3, 0.9),
            'combined_score': random.uniform(0.5, 0.95),
            'exact_matches': random.randint(10, 50),
            'total_predictions': random.randint(100, 500)
        }
        for seed in overlap_seeds
    ],
    'metadata': {
        'total_tested': 1000000,
        'threshold': 0.01,
        'direction': 'bidirectional',
        'forward_count': 150,
        'reverse_count': 140
    }
}

# Save files
with open('forward_survivors.json', 'w') as f:
    json.dump(forward, f, indent=2)
print(f"✅ forward_survivors.json ({len(forward['survivors'])} survivors)")

with open('reverse_survivors.json', 'w') as f:
    json.dump(reverse, f, indent=2)
print(f"✅ reverse_survivors.json ({len(reverse['survivors'])} survivors)")

with open('bidirectional_survivors.json', 'w') as f:
    json.dump(bidirectional, f, indent=2)
print(f"✅ bidirectional_survivors.json ({len(bidirectional['survivors'])} survivors)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Forward:        {len(forward['survivors'])}")
print(f"Reverse:        {len(reverse['survivors'])}")
print(f"Bidirectional:  {len(bidirectional['survivors'])}")
print(f"Precision:      {len(bidirectional['survivors'])/len(forward['survivors']):.1%}")
print(f"Recall:         {len(bidirectional['survivors'])/len(reverse['survivors']):.1%}")
print("\n✅ Ready to run pipeline!")
print("="*80)
