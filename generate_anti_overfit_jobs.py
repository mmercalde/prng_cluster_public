#!/usr/bin/env python3
"""
generate_anti_overfit_jobs.py (v1.0)
====================================
Generates job specifications for anti-overfit meta-optimization (Step 5).
Pre-samples Optuna trials and creates script-based jobs for 26-GPU distribution.

Based on generate_scorer_jobs.py pattern from Step 2.5.

PULL ARCHITECTURE:
- Jobs created on Zeus with pre-sampled Optuna parameters
- Workers write results locally
- Coordinator pulls results via SCP
"""

import optuna
import json
import argparse
from pathlib import Path


def define_search_space(trial: optuna.Trial):
    """
    Define the search space for anti-overfit meta-optimization.
    Matches the _sample_config() method in meta_prediction_optimizer_anti_overfit.py
    """
    params = {
        # Neural network architecture (stored as string for JSON serialization)
        'hidden_layers': trial.suggest_categorical('hidden_layers', [
            '[32]',
            '[64, 32]',
            '[128, 64]',
            '[128, 64, 32]',
            '[256, 128, 64]',
            '[256, 128, 64, 32]'
        ]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        
        # Training parameters
        'epochs': trial.suggest_int('epochs', 50, 300),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
        'early_stopping_patience': trial.suggest_int('early_stopping_patience', 5, 30),
        
        # Regularization
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    }
    return params


def main():
    parser = argparse.ArgumentParser(description="Generate anti-overfit meta-optimization jobs (Step 5)")
    parser.add_argument('--trials', type=int, required=True, help='Number of trials to generate')
    parser.add_argument('--survivors', type=str, required=True, help='Path to survivors JSON')
    parser.add_argument('--lottery-data', type=str, required=True, help='Path to lottery data JSON')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of K-fold CV splits')
    parser.add_argument('--test-holdout', type=float, default=0.2, help='Test holdout percentage')
    parser.add_argument('--study-name', type=str, required=True, help='Optuna study name')
    parser.add_argument('--study-db', type=str, required=True, help='Optuna storage URL')
    parser.add_argument('--output', type=str, default='anti_overfit_jobs.json', help='Output jobs file')

    args = parser.parse_args()

    # Ensure study directory exists
    db_path = args.study_db.replace('sqlite:///', '')
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create Optuna study with pruner (same pattern as Step 2.5)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,
        interval_steps=1,
        n_min_trials=3
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.study_db,
        direction='maximize',
        pruner=pruner,
        load_if_exists=True
    )

    print(f"======================================================================")
    print(f"Generating Anti-Overfit Jobs (Step 5 Distributed)")
    print(f"======================================================================")
    print(f"Pre-sampling {args.trials} parameter sets from Optuna...")
    print(f"K-folds: {args.k_folds}, Test holdout: {args.test_holdout}")
    print(f"Study: {args.study_name}")
    print(f"")

    # Remote path (same on all nodes)
    remote_data_path = "/home/michael/distributed_prng_analysis"
    
    # Get just filenames (not full paths) for remote execution
    survivors_filename = Path(args.survivors).name
    lottery_filename = Path(args.lottery_data).name
    
    jobs = []

    for i in range(args.trials):
        trial = study.ask()
        params = define_search_space(trial)
        
        # Job spec follows Step 2.5 pattern
        # Args passed directly to anti_overfit_trial_worker.py
        job = {
            "job_id": f"anti_overfit_trial_{trial.number}",
            "script": "anti_overfit_trial_worker.py",
            "args": [
                survivors_filename,           # arg[0]: survivors file
                lottery_filename,             # arg[1]: lottery data file  
                str(trial.number),            # arg[2]: trial ID
                json.dumps(params),           # arg[3]: params JSON
                str(args.k_folds),            # arg[4]: k-folds
                str(args.test_holdout),       # arg[5]: test holdout pct
                args.study_name,              # arg[6]: study name
                args.study_db                 # arg[7]: study db path
            ],
            "expected_output": f"anti_overfit_results/trial_{trial.number:04d}.json",
            "timeout": 1800  # 30 minutes max per trial
        }
        jobs.append(job)
        
        print(f"  Trial {trial.number}: arch={params['hidden_layers']}, lr={params['learning_rate']:.6f}, epochs={params['epochs']}")

    # Save jobs file
    with open(args.output, 'w') as f:
        json.dump(jobs, f, indent=2)

    print(f"")
    print(f"✅ Generated {len(jobs)} jobs to {args.output}")
    print(f"   Study: {args.study_name}")
    print(f"   Database: {args.study_db}")


if __name__ == "__main__":
    main()
