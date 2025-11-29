#!/usr/bin/env python3
import json
import optuna

# Create quick test study with just 3 trials
study = optuna.create_study(
    study_name="quick_test",
    direction='maximize',
    storage="sqlite:///optuna_studies/quick_test.db",
    load_if_exists=True
)

# Generate 3 simple configs
jobs = []
for trial_id in range(3):
    config = {
        'residue_mod_1': 10,
        'residue_mod_2': 100,
        'residue_mod_3': 1000,
        'max_offset': 10,
        'temporal_window_size': 100,
        'temporal_num_windows': 5,
        'min_confidence_threshold': 0.2,
        'hidden_layers': '128_64',
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'optuna_trial_number': trial_id
    }
    jobs.append({
        'trial_id': trial_id,
        'config': config
    })

with open('scorer_jobs.json', 'w') as f:
    json.dump(jobs, f, indent=2)

print(f"✅ Generated {len(jobs)} quick test jobs")
