#!/usr/bin/env python3
"""
Scorer Meta-Optimizer (Step 2.5)
================================

Orchestrates the distributed hyperparameter optimization for survivor_scorer.py.

This script, run from the head node (zeus), performs the following loop:
1. Asks Optuna for a batch of 26 new parameter sets (one per GPU).
2. Calls `generate_scorer_evaluation_jobs.py` to create the JSON job files.
3. Calls `run_scorer_evaluation_distributed.sh` to execute these 26 jobs in parallel
   across the full cluster (using coordinator.py and distributed_worker.py).
4. Calls `aggregate_scorer_evaluation_results.py` to gather the 26 accuracy scores.
5. "Tells" Optuna the results of all 26 trials.
6. Repeats this process for a set number of batches.
7. Saves the single best parameter configuration to 'optimal_scorer_config.json'.
"""

import os
import json
import subprocess
import time
import argparse
from pathlib import Path
import optuna

# --- Configuration ---
# The total number of experiments to run (e.g., 10 batches * 26 GPUs = 260 trials)
N_BATCHES = 10
GPUS_PER_BATCH = 26 # This should match your cluster size

# File paths
JOB_DIR = Path("/shared/ml/scorer_evaluation_jobs")
RESULTS_DIR = Path("/shared/ml/scorer_evaluation_results")
AGGREGATED_RESULTS_FILE = RESULTS_DIR / "aggregated_scores.json"
FINAL_CONFIG_FILE = Path("optimal_scorer_config.json")

# Inputs from the main workflow
SURVIVOR_FILE = Path("bidirectional_survivors.json")
TRAIN_HISTORY_FILE = Path("train_history.json")
HOLDOUT_HISTORY_FILE = Path("holdout_history.py") # Used by the worker


def define_search_space(trial: optuna.trial.Trial) -> dict:
    """
    Defines the hyperparameter search space for survivor_scorer.py.
    This is where we define what to "test" in the contest.
    """
    params = {
        # Suggest three integers for residue_mods
        "residue_mod_1": trial.suggest_int("residue_mod_1", 5, 20),
        "residue_mod_2": trial.suggest_int("residue_mod_2", 50, 150),
        "residue_mod_3": trial.suggest_int("residue_mod_3", 500, 1500),
        
        # Suggest a max_offset
        "max_offset": trial.suggest_int("max_offset", 3, 15),

        # Suggest parameters for temporal stability
        "temporal_window_size": trial.suggest_categorical("temporal_window_size", [50, 100, 150, 200]),
        "temporal_num_windows": trial.suggest_int("temporal_num_windows", 3, 10),

        # Suggest a confidence threshold
        "min_confidence_threshold": trial.suggest_float("min_confidence_threshold", 0.05, 0.3, log=True)
    }
    
    # Re-format residue_mods into a list for the JSON
    params["residue_mods"] = [
        params.pop("residue_mod_1"),
        params.pop("residue_mod_2"),
        params.pop("residue_mod_3")
    ]
    
    return params

def run_subprocess(command: list):
    """Helper to run a shell command and check for errors."""
    print(f"\n[Orchestrator] Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR ---")
        print(e.stdout)
        print(e.stderr)
        raise

def main():
    start_time = time.time()
    print("="*80)
    print("🚀 Starting Step 2.5: Scorer Meta-Optimizer")
    print("="*80)

    # Ensure directories exist
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "shards").mkdir(parents=True, exist_ok=True) # For worker output

    # Initialize the Optuna study
    # We use a database for persistence, so it can be stopped and resumed.
    storage_name = "sqlite:///scorer_optimizer.db"
    study = optuna.create_study(
        study_name="survivor_scorer_optimization",
        storage=storage_name,
        direction="maximize",  # We want to maximize accuracy
        load_if_exists=True
    )

    # Main optimization loop
    for batch_num in range(N_BATCHES):
        batch_start_time = time.time()
        print("\n" + "-"*80)
        print(f"Starting Batch {batch_num + 1}/{N_BATCHES}")
        print(f"Total trials so far: {len(study.trials)}")
        print(f"Best score so far: {study.best_value or 'N/A'}")
        print(f"Best params so far: {study.best_params or 'N/A'}")
        print("-"*80)

        # 1. Ask Optuna for a batch of 26 trials
        trials_to_run = []
        for i in range(GPUS_PER_BATCH):
            trial = study.ask(define_search_space)
            trials_to_run.append(trial)
        
        # Prepare parameters to be written to jobs
        batch_params = {trial.number: trial.params for trial in trials_to_run}
        params_file = JOB_DIR / "batch_params.json"
        with open(params_file, 'w') as f:
            json.dump(batch_params, f, indent=2)

        # 2. Generate the 26 JSON job files
        cmd_generate = [
            "python3", "generate_scorer_evaluation_jobs.py",
            "--params-file", str(params_file),
            "--job-dir", str(JOB_DIR),
            "--results-dir", str(RESULTS_DIR / "shards"),
            "--survivor-file", str(SURVIVOR_FILE),
            "--train-history", str(TRAIN_HISTORY_FILE),
            "--holdout-history", str(HOLDOUT_HISTORY_FILE),
            "--gpus", str(GPUS_PER_BATCH)
        ]
        run_subprocess(cmd_generate)

        # 3. Run the 26 jobs in parallel on the cluster
        cmd_run_distributed = [
            "bash", "run_scorer_evaluation_distributed.sh",
            str(JOB_DIR), # The directory containing the job files
        ]
        run_subprocess(cmd_run_distributed)

        # 4. Aggregate the 26 result files
        cmd_aggregate = [
            "python3", "aggregate_scorer_evaluation_results.py",
            "--results-dir", str(RESULTS_DIR / "shards"),
            "--output-file", str(AGGREGATED_RESULTS_FILE)
        ]
        run_subprocess(cmd_aggregate)

        # 5. Tell Optuna the results of this batch
        try:
            with open(AGGREGATED_RESULTS_FILE, 'r') as f:
                aggregated_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: Could not read aggregated results: {e}")
            # Tell Optuna all trials in this batch failed
            for trial in trials_to_run:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue

        print("\n[Orchestrator] Telling Optuna results...")
        for trial in trials_to_run:
            result = aggregated_results["results"].get(str(trial.number))
            
            if result and result["status"] == "success":
                accuracy = result["accuracy"]
                study.tell(trial, accuracy)
                print(f"  Trial {trial.number}: Success (Accuracy: {accuracy:.4f})")
            else:
                error_msg = result.get('error', 'Unknown error')
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                print(f"  Trial {trial.number}: Failed ({error_msg})")
        
        print(f"Batch {batch_num + 1} completed in {time.time() - batch_start_time:.1f}s")

    # All batches done. Save the final best configuration.
    print("\n" + "="*80)
    print("✅ Optimization Complete!")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best Accuracy: {study.best_value:.4f}")
    print("Best Parameters:")
    print(json.dumps(study.best_params, indent=2))
    print("="*80)

    # Save the winning parameters to the file
    with open(FINAL_CONFIG_FILE, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"Saved best configuration to {FINAL_CONFIG_FILE}")

if __name__ == "__main__":
    main()
