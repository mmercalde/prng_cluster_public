#!/usr/bin/env python3
"""
Generate Full Scoring Jobs (v1.0)
=================================
Generates jobs for a full, distributed scoring run *using*
the optimal parameters found by the Scorer Meta-Optimizer.

This script creates one job per N survivors.
"""

import json
from utils.survivor_loader import load_survivors
import argparse
from pathlib import Path
import math
import sys

# Define a chunk size. 1000 survivors per job is a good balance.
# 97k survivors / 1000 = 97 jobs. Your 26 GPUs will handle this in ~4 passes.
DEFAULT_CHUNK_SIZE = 1000

def main():
    parser = argparse.ArgumentParser(description='Generate Full Scoring Jobs (Step 3.5)')
    parser.add_argument('--survivors', type=str, required=True, help='Path to bidirectional_survivors.json')
    parser.add_argument('--config', type=str, default='optimal_scorer_config.json', help='Path to optimal_scorer_config.json')
    parser.add_argument('--train-history', type=str, default='train_history.json', help='Path to train_history.json')
    parser.add_argument('--holdout-history', type=str, default='holdout_history.json', help='Path to holdout_history.json')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE, help='Number of survivors per job')
    parser.add_argument('--jobs-file', type=str, default='scoring_jobs.json', help='Output JSON file for coordinator')
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Note: Ignoring unknown args: {unknown}")

    print("="*70)
    print("Generating Full Distributed Scoring Jobs (Step 3.5)")
    print("="*70)

    # 1. Load the optimal config
    print(f"Loading optimal config from {args.config}...")
    try:
        with open(args.config, 'r') as f:
            optimal_params_str = json.dumps(json.load(f))
    except FileNotFoundError:
        print(f"❌ ERROR: Optimal config file not found: {args.config}")
        print("Please run the Scorer Meta-Optimizer (Step 3) first.")
        return 1
    
    # 2. Load survivors using modular loader (NPZ/JSON auto-detect)
    print(f"Loading survivors from {args.survivors}...")
    try:
        result = load_survivors(args.survivors, return_format="array")
        survivor_seeds = result.data['seeds'].tolist()
        print(f"Loaded {len(survivor_seeds)} survivor seeds from {result.format} "
              f"(fallback={result.fallback_used})")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Survivor file not found: {args.survivors}")
        return 1
    
    # 3. Create job chunks
    jobs = []
    num_chunks = math.ceil(len(survivor_seeds) / args.chunk_size)
    print(f"Splitting {len(survivor_seeds)} seeds into {num_chunks} jobs of {args.chunk_size} seeds each...")
    
    remote_data_path = "/home/michael/distributed_prng_analysis"
    chunk_files_to_create = []
    
    for i in range(num_chunks):
        chunk_start = i * args.chunk_size
        chunk_end = (i + 1) * args.chunk_size
        seeds_chunk = survivor_seeds[chunk_start:chunk_end]
        
        # Create a temporary file for this chunk's seeds
        chunk_file_name = f"chunk_scoring_seeds_{i:04d}.json"
        with open(chunk_file_name, 'w') as f:
            json.dump(seeds_chunk, f)
        chunk_files_to_create.append(chunk_file_name)
            
        remote_chunk_path = f"{remote_data_path}/{chunk_file_name}"

        job = {
            "job_id": f"scoring_chunk_{i:04d}",
            "script": "scorer_trial_worker.py", 
            "args": [
                remote_chunk_path, # Arg 1: survivors_file
                f"{remote_data_path}/{args.train_history}", # Arg 2: train_history_file
                f"{remote_data_path}/{args.holdout_history}", # Arg 3: holdout_history_file
                str(i), # Arg 4: trial_id (used for output filename)
                optimal_params_str, # Arg 5: params
            ],
            "expected_output": f"scorer_trial_results/trial_{i:04d}.json",
            "timeout": 3600,
            "cleanup_files": [remote_chunk_path] # Add chunk file to cleanup
        }
        jobs.append(job)

    # 4. Save jobs file
    with open(args.jobs_file, 'w') as f:
        json.dump(jobs, f, indent=2)
        
    print(f"\n✅ Generated {len(jobs)} jobs and {num_chunks} seed-chunk files.")
    print(f"   Jobs file: {args.jobs_file}")
    print("   Seed chunks: chunk_scoring_seeds_*.json (these must be copied to nodes)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
