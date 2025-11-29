#!/usr/bin/env python3
"""
generate_scorer_jobs.py (v3 - With sample_size parameter)
==========================================================
Generates job specifications for scorer meta-optimization with Optuna pruning support.

CHANGELOG:
----------
v3 (2025-11-29):
- CRITICAL FIX: Added sample_size parameter to search space
- Added --sample-size CLI argument (default: 25000)
- Prevents 742K seed processing, enables fast trials (~30s instead of 30min)

v2:
- Added --legacy-scoring flag for backward compatibility

v1:
- Initial version with Optuna pruning support
"""

import optuna
import json
import argparse
from pathlib import Path


def define_search_space(trial: optuna.Trial, sample_size: int = 25000):
    """
    Define the search space for scorer meta-optimization.
    
    Args:
        trial: Optuna trial object
        sample_size: Number of seeds to sample for training (v3 addition)
    """
    params = {
        # Scorer parameters
        'residue_mod_1': trial.suggest_int('residue_mod_1', 5, 20),
        'residue_mod_2': trial.suggest_int('residue_mod_2', 50, 150),
        'residue_mod_3': trial.suggest_int('residue_mod_3', 500, 1500),
        'max_offset': trial.suggest_int('max_offset', 1, 15),
        'temporal_window_size': trial.suggest_int('temporal_window_size', 50, 100),
        'temporal_num_windows': trial.suggest_int('temporal_num_windows', 3, 10),
        'min_confidence_threshold': trial.suggest_float('min_confidence_threshold', 0.05, 0.25),

        # Neural network architecture
        'hidden_layers': trial.suggest_categorical('hidden_layers', ['128_64', '256_128_64', '512_256_128']),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        
        # v3 CRITICAL ADDITION: Training sample size
        'sample_size': sample_size,
    }
    return params


def main():
    parser = argparse.ArgumentParser(description="Generate scorer meta-optimization jobs with pruning")
    parser.add_argument('--trials', type=int, required=True, help='Number of trials to generate')
    parser.add_argument('--survivors', type=str, required=True, help='Path to survivors JSON')
    parser.add_argument('--train-history', type=str, required=True, help='Path to training history JSON')
    parser.add_argument('--holdout-history', type=str, required=True, help='Path to holdout history JSON')
    parser.add_argument('--study-name', type=str, required=True, help='Optuna study name')
    parser.add_argument('--study-db', type=str, required=True, help='Optuna storage URL')
    parser.add_argument('--output', type=str, default='scorer_jobs.json', help='Output jobs file')
    parser.add_argument('--legacy-scoring', action='store_true', help='Use legacy batch_score() instead of vectorized')
    # v3 ADDITION: sample_size parameter
    parser.add_argument('--sample-size', type=int, default=25000, 
                        help='Number of seeds to sample for training (default: 25000)')

    args = parser.parse_args()

    # Ensure study directory exists
    Path(args.study_db.replace('sqlite:///', '')).parent.mkdir(parents=True, exist_ok=True)

    # Create Optuna study WITH CONSERVATIVE PRUNER
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=6,
        interval_steps=2,
        n_min_trials=5
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.study_db,
        direction='maximize',
        pruner=pruner,
        load_if_exists=True
    )

    print(f"Pre-sampling {args.trials} parameter sets from Optuna with MedianPruner...")
    print(f"Pruner settings: startup={pruner._n_startup_trials}, warmup={pruner._n_warmup_steps}")
    print(f"Sample size: {args.sample_size} seeds per trial")  # v3 addition

    remote_data_path = "/home/michael/distributed_prng_analysis"
    jobs = []

    for i in range(args.trials):
        trial = study.ask()
        params = define_search_space(trial, sample_size=args.sample_size)  # v3: pass sample_size
        params['optuna_trial_number'] = trial.number

        job = {
            "job_id": f"scorer_trial_{i}",
            "analysis_type": "script",
            "script": "scorer_trial_worker.py",
            "args": [
                f"{remote_data_path}/{Path(args.survivors).name}",
                f"{remote_data_path}/{Path(args.train_history).name}",
                f"{remote_data_path}/{Path(args.holdout_history).name}",
                str(i),
                json.dumps(params),
                "--optuna-study-name", args.study_name,
                "--optuna-study-db", args.study_db
            ],
            "expected_output": f"scorer_trial_results/trial_{i:04d}.json",
            "timeout": 3600
        }

        # Add --use-legacy-scoring flag if requested
        if args.legacy_scoring:
            job["args"].append("--use-legacy-scoring")

        jobs.append(job)

    # Write jobs file
    with open(args.output, 'w') as f:
        json.dump(jobs, f, indent=2)

    print(f"✅ Generated {len(jobs)} jobs with pre-sampled parameters")
    print(f"   Output: {args.output}")
    print(f"   Optuna study: {args.study_name} ({args.study_db})")
    print(f"   Sample size: {args.sample_size} seeds")  # v3 addition
    if args.legacy_scoring:
        print(f"   Scoring method: LEGACY (batch_score)")
    else:
        print(f"   Scoring method: VECTORIZED (batch_score_vectorized)")
    print(f"\n📋 Sample job:")
    print(json.dumps(jobs[0], indent=2))


if __name__ == "__main__":
    main()
