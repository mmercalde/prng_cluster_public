#!/usr/bin/env python3
"""
MASTER PIPELINE RUNNER
======================
Calls all your existing scripts in the correct order

Usage:
  python3 run_complete_pipeline.py --lottery-file synthetic_lottery.json --seed-start 0 --seed-count 100000
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and check for errors"""
    print("\n" + "="*80)
    print(f"🚀 {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        sys.exit(1)
    
    print(f"\n✅ COMPLETE: {description}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Complete Pipeline Runner')
    parser.add_argument('--lottery-file', type=str, default='synthetic_lottery.json',
                       help='Lottery data file')
    parser.add_argument('--seed-start', type=int, default=0,
                       help='Starting seed')
    parser.add_argument('--seed-count', type=int, default=100000,
                       help='Number of seeds to test')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='Match threshold (0.01 = 1%)')
    parser.add_argument('--window-iterations', type=int, default=10,
                       help='Window optimizer iterations')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPLETE WHITEPAPER PIPELINE - MASTER RUNNER")
    print("="*80)
    print(f"Lottery file: {args.lottery_file}")
    print(f"Seeds: {args.seed_start} to {args.seed_start + args.seed_count}")
    print(f"Threshold: {args.threshold*100}%")
    print("="*80)
    
    # STEP 1: Window Optimizer (runs sieves internally)
    run_command([
        'python3', 'window_optimizer.py',
        '--lottery-file', args.lottery_file,
        '--seed-start', str(args.seed_start),
        '--seed-count', str(args.seed_count),
        '--threshold', str(args.threshold),
        '--iterations', str(args.window_iterations)
    ], "STEP 1: Window Optimizer (Finding Best Windows & Survivors)")
    
    # STEP 2: Forward Sieve
    run_command([
        'python3', 'sieve_filter.py',
        '--lottery-data', args.lottery_file,
        '--seed-start', str(args.seed_start),
        '--seed-end', str(args.seed_start + args.seed_count),
        '--threshold', str(args.threshold)
    ], "STEP 2: Forward Sieve")
    
    # STEP 3: Reverse Sieve
    run_command([
        'python3', 'reverse_sieve_filter.py',
        '--lottery-data', args.lottery_file,
        '--seed-start', str(args.seed_start),
        '--seed-end', str(args.seed_start + args.seed_count),
        '--threshold', str(args.threshold)
    ], "STEP 3: Reverse Sieve")
    
    # STEP 4: Adaptive Meta-Optimizer
    run_command([
        'python3', 'adaptive_meta_optimizer.py',
        '--mode', 'full',
        '--lottery-data', args.lottery_file,
        '--apply'
    ], "STEP 4: Adaptive Meta-Optimizer (Derive Training Parameters)")
    
    # STEP 5: Survivor Scorer
    run_command([
        'python3', 'survivor_scorer.py',
        '--lottery-data', args.lottery_file,
        '--survivors', 'bidirectional_survivors.json',
        '--batch-mode'
    ], "STEP 5: Survivor Scoring (Extract ML Features)")
    
    # STEP 6: Reinforcement Engine Training
    run_command([
        'python3', 'reinforcement_engine.py',
        '--mode', 'train',
        '--lottery-data', args.lottery_file,
        '--survivors', 'bidirectional_survivors.json'
    ], "STEP 6: Reinforcement Engine (Train Neural Network)")
    
    # STEP 7: Quality Prediction
    run_command([
        'python3', 'reinforcement_engine.py',
        '--mode', 'predict',
        '--lottery-data', args.lottery_file,
        '--survivors', 'bidirectional_survivors.json'
    ], "STEP 7: Quality Prediction")
    
    print("\n" + "="*80)
    print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*80)
    print("\nResults:")
    print("  - Window optimizer results: optimization_results/window_optimizer_results.json")
    print("  - Survivors: forward_survivors.json, reverse_survivors.json, bidirectional_survivors.json")
    print("  - Meta-optimizer config: reinforcement_engine_config.json (updated)")
    print("  - Trained model: reinforcement_model.pth")
    print("  - Predictions: predictions.json")
    print("="*80)

if __name__ == '__main__':
    main()
