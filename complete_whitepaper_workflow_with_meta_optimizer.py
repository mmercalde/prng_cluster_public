#!/usr/bin/env python3
"""
Complete Whitepaper Workflow with Meta-Optimizer - WITH VARIABLE SKIP SUPPORT
==============================================================================
Version: 2.0
Date: 2025-11-15

NEW IN V2.0:
- Added --test-both-modes flag to test constant AND variable skip patterns
- Added --prng-type argument (no longer hardcoded)
- Displays skip mode distribution in summaries
- Passes --test-both-modes to window optimizer

This script orchestrates the full, distributed ML pipeline in the correct order.

Steps:
1.  (26-GPU) Bayesian Window Optimizer → Finds optimal params + generates survivors
2.5 (26-GPU) Scorer Meta-Optimizer → Finds optimal scorer parameters
3.  (26-GPU) Full Distributed Scoring → Scores all survivors
4.  (Local)  Adaptive Meta-Optimizer → Derives optimal training architecture
5.  (26-GPU) Anti-Overfit Optimizer → Trains final model with K-Fold validation
6.  (Local)  Quality Prediction → Tests the final model
"""

import subprocess
import sys
import json
from pathlib import Path
import time
import argparse
import os

def launch_progress_monitor():
    """Launch progress monitor - auto-starts tmux if needed"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Check if monitor was already launched (to prevent duplicates)
        if os.environ.get('PRNG_MONITOR_LAUNCHED'):
            return True
        
        # Check if we're already in tmux
        if os.environ.get('TMUX'):
            # Already in tmux but monitor not launched yet - don't split
            # (tmux command already created the split)
            return True
        else:
            # Not in tmux - start tmux with both workflow and monitor
            print("📊 Starting tmux session with progress monitor...")
            
            # Re-launch this script inside tmux with the same arguments
            args_str = ' '.join(sys.argv)
            
            # Set env var to prevent duplicate monitor launch
            # Create tmux session, run workflow, split and run monitor
            tmux_cmd = (
                f'tmux new-session -d -s prng -c {script_dir} '
                f'"PRNG_MONITOR_LAUNCHED=1 python3 {args_str}; read -p Press_Enter_to_close" \\; '
                f'split-window -h -p 40 "python3 progress_monitor.py" \\; '
                f'select-pane -t 0 \\; '
                f'attach'
            )
            
            os.execvp('bash', ['bash', '-c', tmux_cmd])
            # Note: execvp replaces current process, so we won't return here
            
    except Exception as e:
        print(f"⚠️  Could not auto-launch monitor: {e}")
        print("   Run manually in another terminal: python3 progress_monitor.py")
        return False
import os

def run_command(cmd, description):
    """Run a shell command, stream output, and check return code."""
    print("\n" + "="*70)
    print(f"🚀 STARTING: {description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}\n")

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, stdin=subprocess.DEVNULL, env=env)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        returncode = process.poll()
        if returncode != 0:
            print(f"\n❌ FAILED (Code {returncode}): {description}")
            return False
        print(f"\n✅ COMPLETE: {description}")
        return True
    except Exception as e:
        print(f"\n❌ FAILED (Exception): {description}\n{e}")
        return False

def main(args):
    start_time = time.time()
    
    # Auto-launch progress monitor if in tmux
    launch_progress_monitor()


    print("="*70)
    print("COMPLETE WHITEPAPER WORKFLOW V2.0 (WITH SKIP MODE SUPPORT)")
    print("="*70)
    print("\nOrchestrating full, distributed pipeline:")
    print("  1. Bayesian Window Optimizer (26-GPU, runs real sieves!)")
    if args.test_both_modes:
        print("     ⚡ TESTING BOTH MODES: constant AND variable skip")
    else:
        print("     ℹ️  Testing constant skip only (use --test-both-modes for both)")
    print("  2.5 Scorer Meta-Optimizer (26-GPU)")
    print("  3. Full Distributed Scoring (26-GPU)")
    print("  4. Adaptive Meta-Optimizer (Local)")
    print("  5. Anti-Overfit Optimizer (26-GPU)")
    print("  6. Final Model Prediction (Local)")
    print("="*70)

    # --- PREREQUISITES ---
    print("\n" + "="*70)
    print("CHECKING PREREQUISITES")
    print("="*70)

    # Define paths
    survivor_file = "bidirectional_survivors.json"
    train_history_file = "train_history.json"
    holdout_history_file = "holdout_history.json"
    optimal_window_config = "optimal_window_config.json"
    optimal_scorer_config = "optimal_scorer_config.json"
    scored_survivor_file = "survivors_with_scores.json"
    optimal_training_config = "reinforcement_engine_config.json"
    final_model_path = "models/anti_overfit/best_model.pth"

    required_files = [
        # Step 1 - Window Optimizer
        'window_optimizer.py',
        'window_optimizer_integration_final.py',
        'window_optimizer_bayesian.py',
        'coordinator.py',
        args.lottery_file,

        # Step 2.5 - Scorer Meta-Optimizer
        'run_scorer_meta_optimizer.sh',
        'generate_scorer_jobs.py',
        'scorer_trial_worker.py',
        'ml_coordinator_config.json',

        # Step 3 - Full Scoring
        'run_full_scoring.sh',
        'generate_full_scoring_jobs.py',

        # Step 4 - Adaptive Optimizer
        'adaptive_meta_optimizer.py',

        # Step 5 - Anti-Overfit
        'meta_prediction_optimizer_anti_overfit.py',
        'run_ml_distributed.sh',
        'anti_overfit_trial_worker.py',

        # Core Dependencies
        'reinforcement_engine.py',
        'survivor_scorer.py'
    ]

    missing = []
    for f in required_files:
        if Path(f).exists():
            print(f"  [✅] {f}")
        else:
            print(f"  [❌] {f} (MISSING)")
            missing.append(f)

    if missing:
        print(f"\n❌ Missing required files: {missing}")
        return 1
    else:
        print("\n✅ All prerequisites found.")

    # --- STEP 1: Bayesian Window Optimizer ---
    print("\n\n" + "="*70)
    print("STEP 1: BAYESIAN WINDOW OPTIMIZER (26-GPU, REAL SIEVES)")
    print("="*70)
    print(f"\nLaunching {args.window_opt_trials} Bayesian trials...")
    print(f"PRNG: {args.prng_type}")
    
    if args.test_both_modes:
        print(f"⚡ BOTH MODES ENABLED:")
        print(f"   • Constant skip: {args.prng_type}")
        print(f"   • Variable skip: {args.prng_type}_hybrid")
    else:
        print(f"ℹ️  Constant skip only: {args.prng_type}")
        print(f"   (Use --test-both-modes to test variable skip too)")
    
    print("\nThis will:")
    print("  • Run REAL forward/reverse sieves for each trial")
    print("  • Accumulate bidirectional survivors across all trials")
    print("  • Find optimal window parameters")
    print("  • Generate train/holdout splits")
    print("")

    # NEW IN V2.0: Build command with configurable prng-type and test-both-modes
    cmd = [
        'python3', 'window_optimizer.py',
        '--strategy', 'bayesian',
        '--lottery-file', args.lottery_file,
        '--trials', str(args.window_opt_trials),
        '--output', optimal_window_config,
        '--max-seeds', str(args.seed_count),
        '--prng-type', args.prng_type  # ✅ NOW CONFIGURABLE!
    ]
    
    # NEW IN V2.0: Add --test-both-modes flag if requested
    if args.test_both_modes:
        cmd.append('--test-both-modes')
    
    if not run_command(cmd, "Running: 26-GPU Bayesian Window Optimization with Real Sieves"):
        return 1

    # Verify outputs
    if not Path(optimal_window_config).exists():
        print(f"\n❌ {optimal_window_config} was not created by Step 1.")
        return 1

    if not Path(survivor_file).exists():
        print(f"\n❌ {survivor_file} was not created by Step 1.")
        return 1

    # NEW IN V2.0: Display skip mode distribution
    try:
        with open(survivor_file, 'r') as f:
            survivors = json.load(f)
            survivor_count = len(survivors) if isinstance(survivors, list) else len(survivors.get('survivors', []))

        # Count survivors by skip mode
        skip_mode_counts = {}
        if isinstance(survivors, list):
            for s in survivors:
                mode = s.get('skip_mode', 'unknown')
                skip_mode_counts[mode] = skip_mode_counts.get(mode, 0) + 1

        print(f"\n✅ Step 1 Complete:")
        print(f"   Optimal config: {optimal_window_config}")
        print(f"   Survivors generated: {survivor_count:,}")
        
        # NEW IN V2.0: Show skip mode distribution
        if skip_mode_counts:
            print(f"\n   Skip Mode Distribution:")
            for mode, count in sorted(skip_mode_counts.items()):
                pct = (count / survivor_count * 100) if survivor_count > 0 else 0
                print(f"     • {mode:12s}: {count:6,} ({pct:5.1f}%)")
        
        print(f"\n   Train data: {train_history_file}")
        print(f"   Holdout data: {holdout_history_file}")
        
    except Exception as e:
        print(f"\n⚠️  Could not read survivor file: {e}")
        survivor_count = 0

    # --- STEP 2.5: Scorer Meta-Optimizer (Distributed) ---
    print("\n\n" + "="*70)
    print("STEP 2.5: SCORER META-OPTIMIZER (Distributed, 26 GPUs)")
    print("="*70)
    print(f"\nLaunching {args.scorer_trials} distributed trials to find optimal scorer parameters...")
    print("ℹ️  This will automatically use skip_mode as an ML feature")

    if not run_command(
        ['bash', 'run_scorer_meta_optimizer.sh', str(args.scorer_trials)],
        "Running: 26-GPU Scorer Meta-Optimization"
    ):
        return 1

    if not Path(optimal_scorer_config).exists():
        print(f"\n❌ {optimal_scorer_config} was not created by Step 2.5.")
        return 1
    else:
        print(f"\n✅ Optimal scorer config saved to {optimal_scorer_config}")

    # --- STEP 3: Full Distributed Scoring ---
    print("\n\n" + "="*70)
    print("STEP 3: FULL DISTRIBUTED SCORING (Distributed, 26 GPUs)")
    print("="*70)
    print(f"\nLaunching distributed run to score all {survivor_count:,} survivors...")
    print("ℹ️  This will extract skip_mode features for each survivor")

    if not run_command(
        ['bash', 'run_full_scoring.sh'],
        "Running: 26-GPU Full Scoring Run"
    ):
        return 1

    if not Path(scored_survivor_file).exists():
        print(f"\n❌ {scored_survivor_file} was not created by Step 3.")
        return 1
    else:
        print(f"\n✅ All survivors scored and aggregated into {scored_survivor_file}")

    # --- STEP 4: Adaptive Meta-Optimizer (Derive Training Params) ---
    print("\n\n" + "="*70)
    print("STEP 4: ADAPTIVE META-OPTIMIZER (Parameter Derivation)")
    print("="*70)
    print("\nDeriving optimal training parameters (network architecture, epochs, etc.)...")

    if not run_command(
        ['python3', 'adaptive_meta_optimizer.py',
         '--mode', 'full',
         '--lottery-data', train_history_file,
         '--survivor-data', scored_survivor_file,
         '--apply'],
        "Running: Adaptive Meta-Optimizer"
    ):
        print("\n⚠️  Meta-optimizer failed, training will use default parameters...")
    else:
        print(f"\n✅ Optimal training config saved to {optimal_training_config}")

    # --- STEP 5: Anti-Overfit Optimizer (Distributed Final Training) ---
    print("\n\n" + "="*70)
    print("STEP 5: ANTI-OVERFIT OPTIMIZER (Distributed, 26 GPUs)")
    print("="*70)
    print(f"\nLaunching {args.anti_overfit_trials} distributed K-Fold trials to train final model...")
    print("ℹ️  Model will train on features including skip_mode")

    if not run_command(
        ['python3', 'meta_prediction_optimizer_anti_overfit.py',
         '--survivors', scored_survivor_file,
         '--lottery-data', train_history_file,
         '--trials', str(args.anti_overfit_trials),
         '--k-folds', str(args.k_folds),
         '--study-name', 'final_model_anti_overfit',
         '--distributed'],
        "Running: 26-GPU Anti-Overfit Final Model Training"
    ):
        return 1

    if not Path(final_model_path).exists():
        print(f"\n❌ Final model {final_model_path} was not created by Step 5.")
        return 1
    else:
        print(f"\n✅ Final trained model saved to {final_model_path}")

    # --- STEP 6: Quality Prediction (Local Test) ---
    print("\n\n" + "="*70)
    print("STEP 6: FINAL MODEL PREDICTION (Local Test)")
    print("="*70)
    print(f"\nLoading final model from {final_model_path} to make predictions...")

    try:
        from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
        from survivor_scorer import SurvivorScorer

        with open(holdout_history_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                holdout_history = [d['draw'] for d in data]
            else:
                holdout_history = data

            if not holdout_history:
                print("⚠️ Holdout history is empty, using train history for prediction test.")
                with open(train_history_file, 'r') as f_train:
                    train_data = json.load(f_train)
                    if isinstance(train_data, list) and len(train_data) > 0 and isinstance(train_data[0], dict):
                        holdout_history = [d['draw'] for d in train_data]
                    else:
                        holdout_history = train_data

        engine = ReinforcementEngine(
            config=ReinforcementConfig(),
            lottery_history=holdout_history
        )

        engine.load_model(final_model_path)
        print("✅ Final model loaded successfully.")

        import random
        test_survivors = [random.randint(0, 1000000) for _ in range(20)]
        print("✅ Predicting quality for 20 random survivors...")

        predictions = engine.predict_quality_batch(
            survivors=test_survivors,
            lottery_history=holdout_history
        )

        print("\n--- Sample Predictions ---")
        for seed, quality in zip(test_survivors[:5], predictions[:5]):
            print(f"  Seed {seed:<10} -> Quality: {quality:.4f}")
        print("--------------------------\n")

    except Exception as e:
        print(f"\n❌ Error during Step 6 (Prediction): {e}")
        import traceback
        traceback.print_exc()
        return 1

    # --- SUCCESS! ---
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("✅✅✅ COMPLETE DISTRIBUTED WORKFLOW PASSED ✅✅✅")
    print("="*70)
    print("\nWorkflow Summary:")
    print(f"  1. ✅ Bayesian Window Optimizer: {args.window_opt_trials} trials, {survivor_count:,} survivors")
    if args.test_both_modes:
        print(f"      ⚡ Tested BOTH constant AND variable skip modes")
    else:
        print(f"      ℹ️  Tested constant skip only")
    print(f"  2.5 ✅ Scorer Meta-Optimizer: {args.scorer_trials} distributed trials")
    print(f"  3. ✅ Full Scoring Run: All survivors scored")
    print(f"  4. ✅ Adaptive Optimizer: Training params derived")
    print(f"  5. ✅ Anti-Overfit Optimizer: {args.anti_overfit_trials} distributed trials")
    print(f"  6. ✅ Final Prediction: Model working")
    print(f"\nTest completed in {elapsed / 60:.1f} minutes")
    print("="*70 + "\n")

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Complete Whitepaper Workflow Orchestrator V2.0 (WITH SKIP MODE SUPPORT)'
    )

    # --- General ---
    parser.add_argument('--lottery-file', type=str, default='synthetic_lottery.json',
                       help='Initial lottery data file')

    # --- Step 1: Window Optimizer (NEW IN V2.0!) ---
    parser.add_argument('--window-opt-trials', type=int, default=50,
                       help='Number of Bayesian trials for Window Optimizer (Step 1)')
    parser.add_argument('--seed-count', type=int, default=10_000_000,
                       help='Max seeds per Bayesian trial (Step 1)')
    
    # NEW IN V2.0: Configurable PRNG type
    parser.add_argument('--prng-type', type=str, default='java_lcg',
                       help='Base PRNG type from prng_registry (e.g., java_lcg, xorshift32, mt19937)')
    
    # NEW IN V2.0: Test both constant AND variable skip modes
    parser.add_argument('--test-both-modes', action='store_true',
                       help='Test BOTH constant and variable skip patterns (runs 4 sieves per trial instead of 2)')

    # --- Step 2.5 ---
    parser.add_argument('--scorer-trials', type=int, default=100,
                       help='Number of trials for Scorer Meta-Optimizer (Step 2.5)')

    # --- Step 5 ---
    parser.add_argument('--anti-overfit-trials', type=int, default=50,
                       help='Number of trials for Anti-Overfit Optimizer (Step 5)')
    parser.add_argument('--k-folds', type=int, default=5,
                       help='Number of K-Folds for Anti-Overfit Optimizer (Step 5)')

    parsed_args = parser.parse_args()

    os.environ["DISTRIBUTED_WORKFLOW"] = "true"
    sys.exit(main(parsed_args))
