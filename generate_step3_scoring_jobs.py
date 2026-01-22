#!/usr/bin/env python3
"""
Generate Full Scoring Jobs - Step 3 Job Generator
===================================================
Creates distributed job specifications for the full scoring pass.

CRITICAL FIX: Previous version called scorer_trial_worker.py which was designed
for Step 2.5 meta-optimization and returns only prediction floats. This version
calls full_scoring_worker.py which extracts full 46-feature objects.

Output: scoring_jobs.json with jobs that call full_scoring_worker.py

Usage:
    python3 generate_full_scoring_jobs.py \
        --survivors bidirectional_survivors.json \
        --config optimal_scorer_config.json \
        --train-history train_history.json \
        --chunk-size 5000 \
        --output-file scoring_jobs.json

Author: Distributed PRNG Analysis System
Date: December 12, 2025
Version: 2.0.0 - FIXED to use full_scoring_worker.py
"""

import sys
import os
import json
from utils.survivor_loader import load_survivors
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import math


def load_json(filepath: str) -> Any:
    """Load JSON file with error handling."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Any, filepath: str):
    """Save JSON file with pretty formatting."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def extract_seeds(survivors_data: Any) -> List[int]:
    """
    Extract seed values from various input formats.
    
    Supports:
    - Flat list: [12345, 67890, ...]
    - Object list: [{"seed": 12345}, {"candidate_seed": 67890}, ...]
    """
    if not survivors_data:
        return []
    
    if isinstance(survivors_data[0], dict):
        seeds = []
        for s in survivors_data:
            seed = s.get('seed', s.get('candidate_seed', s.get('survivor_seed')))
            if seed is not None:
                seeds.append(int(seed))
        return seeds
    else:
        return [int(s) for s in survivors_data]


def calculate_smart_chunk_size(total_seeds: int, num_gpus: int = 26) -> int:
    """
    Calculate optimal chunk size based on GPU count and total seeds.
    
    Goal: Create approximately num_gpus chunks (one per GPU) with slight
    overhead for fault tolerance, but not so many that SSH capacity is overwhelmed.
    
    Args:
        total_seeds: Total number of seeds to process
        num_gpus: Number of GPUs in cluster (default: 26)
    
    Returns:
        Optimal chunk size
    """
    # Target: ~1.5x GPU count for some fault tolerance buffer
    # but not more than 2x to avoid SSH overload
    target_chunks = int(num_gpus * 1.5)
    
    # Calculate base chunk size
    chunk_size = max(1000, total_seeds // target_chunks)
    
    # Round up to nice number for cleaner distribution
    chunk_size = ((chunk_size + 999) // 1000) * 1000
    
    return chunk_size


def generate_jobs(
    survivors_file: str,
    train_history_file: str,
    holdout_history_file: str,
    config_file: Optional[str],
    chunk_size: int = 5000,
    prng_type: str = 'java_lcg',
    mod: int = 1000,
    remote_base_path: str = '/home/michael/distributed_prng_analysis',
    timeout: int = 7200,
    forward_survivors_file: Optional[str] = None,
    reverse_survivors_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate job specifications for distributed full scoring.
    
    Args:
        survivors_file: Path to bidirectional survivors JSON
        train_history_file: Path to training lottery history JSON
        config_file: Optional path to optimal scorer config
        chunk_size: Seeds per job chunk
        prng_type: PRNG type (can be overridden by config)
        mod: Modulo value (can be overridden by config)
        remote_base_path: Base path on remote workers
        timeout: Job timeout in seconds
        forward_survivors_file: Optional path to forward survivors
        reverse_survivors_file: Optional path to reverse survivors
    
    Returns:
        List of job specifications for coordinator
    """
    
    # Load survivors
    print(f"Loading survivors from {survivors_file}...")
    result = load_survivors(survivors_file)
    survivors_data = result.data if hasattr(result, "data") else result
    seeds = extract_seeds(survivors_data)
    print(f"  Loaded {len(seeds)} survivor seeds")
    
    if not seeds:
        raise ValueError("No survivor seeds found in input file")
    
    # Load config to get optimized parameters (if available)
    if config_file and os.path.exists(config_file):
        print(f"Loading optimal config from {config_file}...")
        config = load_json(config_file)
        
        # Extract PRNG settings from config
        prng_type = config.get('prng_type', config.get('prng', {}).get('type', prng_type))
        mod = config.get('mod', config.get('prng', {}).get('mod', mod))
        
        print(f"  Using prng_type={prng_type}, mod={mod}")
    
    # Split seeds into chunks
    seed_chunks = chunk_list(seeds, chunk_size)
    num_chunks = len(seed_chunks)
    print(f"Split {len(seeds)} seeds into {num_chunks} chunks of ~{chunk_size} each")
    
    # Create chunk files directory
    chunk_dir = Path("scoring_chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # Results directory
    results_dir = Path("full_scoring_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    jobs = []
    
    for i, chunk in enumerate(seed_chunks):
        # Save chunk to local file
        chunk_filename = f"chunk_{i:04d}.json"
        local_chunk_path = chunk_dir / chunk_filename
        
        with open(local_chunk_path, 'w') as f:
            json.dump(chunk, f)
        
        # Remote paths (will be copied by run_full_scoring.sh)
        remote_chunk_path = f"{remote_base_path}/scoring_chunks/{chunk_filename}"
        remote_history_path = f"{remote_base_path}/{Path(train_history_file).name}"
        remote_holdout_path = f"{remote_base_path}/{Path(holdout_history_file).name}"
        remote_output_path = f"{remote_base_path}/full_scoring_results/chunk_{i:04d}.json"
        
        # Build job arguments
        args = [
            "--seeds-file", remote_chunk_path,
            "--train-history", remote_history_path,
            "--holdout-history", remote_holdout_path,
            "--output-file", remote_output_path,
            "--prng-type", prng_type,
            "--mod", str(mod)
        ]
        
        # Add optional dual-sieve files if provided
        if forward_survivors_file:
            args.extend(["--forward-survivors", 
                        f"{remote_base_path}/{Path(forward_survivors_file).name}"])
        
        if reverse_survivors_file:
            args.extend(["--reverse-survivors",
                        f"{remote_base_path}/{Path(reverse_survivors_file).name}"])
        
        # Create job specification
        job = {
            "job_id": f"full_scoring_{i:04d}",
            "script": "full_scoring_worker.py",  # <-- FIXED: Was scorer_trial_worker.py
            "args": args,
            "chunk_file": str(local_chunk_path),
            "seed_count": len(chunk),
            "expected_output": f"full_scoring_results/chunk_{i:04d}.json",
            "timeout": timeout
        }
        
        jobs.append(job)
        
        if (i + 1) % 10 == 0 or (i + 1) == num_chunks:
            print(f"  Generated jobs: {i+1}/{num_chunks}")
    
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description='Generate Full Scoring Jobs for Step 3 (calls full_scoring_worker.py)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script generates job specifications for distributed full scoring.
Jobs call full_scoring_worker.py which extracts 46 ML features per survivor.

Example:
    python3 generate_full_scoring_jobs.py \\
        --survivors bidirectional_survivors.json \\
        --train-history train_history.json \\
        --config optimal_scorer_config.json \\
        --chunk-size 5000 \\
        --output-file scoring_jobs.json
        """
    )
    
    # Required arguments
    parser.add_argument('--survivors', required=True,
                       help='JSON file containing survivor seeds')
    parser.add_argument('--train-history', required=True,
                       help='JSON file containing training lottery history')
    
    # Optional arguments
    parser.add_argument('--config', default=None,
                       help='Optimal scorer config from Step 2.5')
    parser.add_argument('--chunk-size', type=str, default='auto',
                       help='Seeds per job chunk (default: auto = smart sizing for 26 GPUs)')
    parser.add_argument('--num-gpus', type=int, default=26,
                       help='Number of GPUs in cluster for auto chunk sizing (default: 26)')
    parser.add_argument('--prng-type', default='java_lcg',
                       help='PRNG type if not in config (default: java_lcg)')
    parser.add_argument('--mod', type=int, default=1000,
                       help='Modulo value if not in config (default: 1000)')
    parser.add_argument('--output-file', '-o', default='scoring_jobs.json',
                       help='Output jobs file (default: scoring_jobs.json)')
    parser.add_argument('--remote-base-path', default='/home/michael/distributed_prng_analysis',
                       help='Base path on remote workers')
    parser.add_argument('--timeout', type=int, default=7200,
                       help='Job timeout in seconds (default: 7200)')
    
    # Optional dual-sieve files
    parser.add_argument('--forward-survivors', default=None,
                       help='Forward sieve survivors for dual-sieve scoring')
    parser.add_argument('--reverse-survivors', default=None,
                       help='Reverse sieve survivors for dual-sieve scoring')
    parser.add_argument('--holdout-history', default='holdout_history.json',
                       help='Holdout history for computing holdout_hits y-label')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GENERATE FULL SCORING JOBS - Step 3")
    print("=" * 60)
    print(f"  Survivors: {args.survivors}")
    print(f"  Train History: {args.train_history}")
    print(f"  Config: {args.config or 'None (using defaults)'}")
    print(f"  Chunk Size: {args.chunk_size} {'(smart GPU-aware sizing)' if args.chunk_size == 'auto' else '(manual)'}")
    print(f"  Num GPUs: {args.num_gpus}")
    print(f"  PRNG Type: {args.prng_type}")
    print(f"  Mod: {args.mod}")
    print(f"  Output: {args.output_file}")
    print("=" * 60)
    
    # Validate input files
    if not os.path.exists(args.survivors):
        print(f"ERROR: Survivors file not found: {args.survivors}")
        sys.exit(1)
    
    if not os.path.exists(args.train_history):
        print(f"ERROR: Training history file not found: {args.train_history}")
        sys.exit(1)
    
    if args.config and not os.path.exists(args.config):
        print(f"WARNING: Config file not found: {args.config} - using defaults")
        args.config = None
    
    # Load survivors using modular loader (NPZ/JSON auto-detect)
    result = load_survivors(args.survivors, return_format="array")
    survivors_data = result.data['seeds'].tolist()
    total_seeds = result.count
    print(f"  Loaded {total_seeds:,} survivors from {result.format} (fallback={result.fallback_used})")
    
    # Calculate chunk size (auto or manual)
    if args.chunk_size == 'auto':
        chunk_size = calculate_smart_chunk_size(total_seeds, args.num_gpus)
        print(f"  Auto chunk size: {chunk_size:,} (for {args.num_gpus} GPUs, {total_seeds:,} seeds)")
    else:
        chunk_size = int(args.chunk_size)
    
    num_chunks = (total_seeds + chunk_size - 1) // chunk_size
    print(f"  Total chunks: {num_chunks} (target: ~{args.num_gpus * 1.5:.0f} for {args.num_gpus} GPUs)")
    
    # Generate jobs
    try:
        jobs = generate_jobs(
            survivors_file=args.survivors,
            train_history_file=args.train_history,
            holdout_history_file=args.holdout_history,
            config_file=args.config,
            chunk_size=chunk_size,
            prng_type=args.prng_type,
            mod=args.mod,
            remote_base_path=args.remote_base_path,
            timeout=args.timeout,
            forward_survivors_file=args.forward_survivors,
            reverse_survivors_file=args.reverse_survivors
        )
    except Exception as e:
        print(f"ERROR: Failed to generate jobs: {e}")
        sys.exit(1)
    
    # Save jobs file
    save_json(jobs, args.output_file)
    
    # Calculate totals
    total_seeds = sum(j['seed_count'] for j in jobs)
    
    print()
    print("=" * 60)
    print("JOB GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total Jobs: {len(jobs)}")
    print(f"  Total Seeds: {total_seeds:,}")
    print(f"  Output File: {args.output_file}")
    print()
    print("Next steps:")
    print(f"  1. Copy chunk files to remote nodes")
    print(f"  2. Run: python3 coordinator.py --jobs-file {args.output_file}")
    print(f"  3. Aggregate results from full_scoring_results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
