#!/usr/bin/env python3
"""
Full Scoring Worker - Step 3 Distributed Scoring
=================================================
Scores survivor seeds and extracts full 50-feature vectors for ML pipeline.

This script is designed for Step 3 of the whitepaper pipeline.
Uses SurvivorScorer.extract_ml_features() for complete feature extraction.

CRITICAL FIX: scorer_trial_worker.py was designed for Step 2.5 meta-optimization
and only returns prediction floats. This worker returns full survivor objects
with all 50 features required by Steps 4-6 (Reinforcement Engine).

PULL Architecture:
- Reads seeds from local file (pre-copied by coordinator)
- Writes results to local file
- Coordinator pulls results via SCP

Output Format:
[
  {
    "seed": 12345,
    "score": 0.847,
    "features": {
      "lane_agreement_8": 0.75,
      "residue_8_match_rate": 0.91,
      ... // all 50 features
    },
    "metadata": {
      "prng_type": "java_lcg",
      "mod": 1000,
      "worker_hostname": "zeus",
      "worker_gpu": 0,
      "timestamp": 1765517223.5
    }
  },
  ...
]

Usage:
    python3 full_scoring_worker.py \
        --seeds-file chunk_0001.json \
        --train-history train_history.json \
        --output-file scoring_results/chunk_0001.json \
        --prng-type java_lcg \
        --gpu-id 0

Author: Distributed PRNG Analysis System
Date: December 12, 2025
Version: 1.0.0
"""

import sys
import os
import json
import time
import socket
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
# GlobalStateTracker - GPU-neutral module for global features (14 features)
from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_NAMES

# ============================================================================
# ROCm/CUDA Environment Setup - MUST BE BEFORE ANY GPU IMPORTS
# ============================================================================
HOST = socket.gethostname()

# ROCm setup for AMD GPUs (rig-6600, rig-6600b)
if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
    os.environ.setdefault("ROCM_PATH", "/opt/rocm")
    os.environ.setdefault("HIP_PATH", "/opt/rocm")

# GPU isolation - MUST be set before torch import
def setup_gpu_environment(gpu_id: int):
    """Configure GPU isolation for this worker process."""
    if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]:
        # AMD ROCm
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        # NVIDIA CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# ============================================================================
# Logging Setup
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FullScoringWorker")

# ============================================================================
# SurvivorScorer will be imported AFTER GPU environment setup in main()
# This prevents CUDA/ROCm initialization before GPU isolation is configured
# ============================================================================
SurvivorScorer = None  # Placeholder, imported in main()



def _best_effort_gpu_cleanup():
    """Post-trial GPU memory cleanup (safe, non-invasive)"""
    try:
        import gc
        gc.collect()
    except Exception:
        pass

    # CuPy ROCm allocator cleanup
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    # PyTorch ROCm allocator cleanup
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


import atexit
atexit.register(_best_effort_gpu_cleanup)

def load_seeds(seeds_file: str) -> List[int]:
    """
    Load seeds from JSON file.
    Handles both flat list [12345, 67890, ...] and object list [{seed: 12345}, ...]
    """
    with open(seeds_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        return []
    
    # Handle different formats
    if isinstance(data[0], dict):
        # Object format: [{seed: 12345, ...}, ...]
        return [int(s.get('seed', s.get('candidate_seed', 0))) for s in data]
    else:
        # Flat format: [12345, 67890, ...]
        return [int(s) for s in data]



def load_survivors_with_metadata(seeds_file: str) -> List[Dict]:
    """
    Load survivors preserving full metadata (forward_count, reverse_count, etc.)
    Returns list of dicts, each with at least 'seed' key.
    """
    with open(seeds_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        return []
    
    # Handle different formats
    if isinstance(data[0], dict):
        # Object format: [{seed: 12345, forward_count: 100, ...}, ...]
        result = []
        for s in data:
            obj = dict(s)  # Copy
            if 'candidate_seed' in obj and 'seed' not in obj:
                obj['seed'] = obj['candidate_seed']
            result.append(obj)
        return result
    else:
        # Flat format: [12345, 67890, ...] - convert to objects
        return [{'seed': int(s)} for s in data]

def load_lottery_history(history_file: str) -> List[int]:
    """
    Load lottery history from JSON file.
    Handles both flat list [978, 973, ...] and object list [{"draw": 978}, ...]
    """
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        return []
    
    # Handle different formats
    if isinstance(data[0], dict):
        # Object format with various possible keys
        for key in ['draw', 'number', 'value', 'result']:
            if key in data[0]:
                return [int(entry[key]) for entry in data]
        # Fallback: try first numeric value
        for entry in data:
            for v in entry.values():
                if isinstance(v, (int, float)):
                    return [int(entry[list(entry.keys())[0]]) for entry in data]
        raise ValueError(f"Cannot parse lottery history format: {list(data[0].keys())}")
    else:
        # Flat format: [978, 973, ...]
        return [int(x) for x in data]




def compute_holdout_hits(
    seed: int,
    holdout_history: List[int],
    prng_type: str = 'java_lcg',
    mod: int = 1000
) -> float:
    """
    Compute holdout Hit@K for a single seed.
    
    This is the y-label for ML training (Step 5).
    Measures how many holdout draws the seed correctly predicts.
    
    Args:
        seed: The survivor seed to evaluate
        holdout_history: List of holdout lottery draws (NOT used in sieving)
        prng_type: PRNG algorithm type
        mod: Modulo value (default: 1000)
    
    Returns:
        Hit rate as float [0.0, 1.0] = hits / total_holdout_draws
    """
    from prng_registry import get_cpu_reference
    
    prng_func = get_cpu_reference(prng_type)
    if prng_func is None:
        logger.warning(f"Unknown PRNG type '{prng_type}', cannot compute holdout_hits")
        return 0.0
    
    # Generate all predictions at once (prng_func returns list of N values)
    num_draws = len(holdout_history)
    if num_draws == 0:
        return 0.0
    
    predictions = prng_func(seed, num_draws)
    hits = 0
    for position, actual_draw in enumerate(holdout_history):
        predicted = predictions[position] % mod
        if predicted == actual_draw:
            hits += 1
    
    return hits / num_draws


def compute_holdout_hits_batch(
    seeds: List[int],
    holdout_history: List[int],
    train_history_len: int,
    prng_type: str = 'java_lcg',
    mod: int = 1000
) -> Dict[int, float]:
    """
    Compute holdout Hit@K for multiple seeds efficiently.
    
    CRITICAL: offset is DERIVED from train_history_len, not configurable.
    holdout data is positions [train_history_len : train_history_len + holdout_len]
    
    Returns dict mapping seed -> holdout_hits rate.
    """
    from prng_registry import get_cpu_reference
    
    prng_func = get_cpu_reference(prng_type)
    if prng_func is None:
        logger.warning(f"Unknown PRNG type '{prng_type}', cannot compute holdout_hits")
        return {seed: 0.0 for seed in seeds}
    
    results = {}
    n_holdout = len(holdout_history)
    
    # OFFSET DERIVED - THIS IS THE LAW (per Team Beta)
    offset = train_history_len
    logger.info(f"[HOLDOUT] Offset derived from train_history_len: {offset}")
    
    for seed in seeds:
        # Generate predictions starting at position OFFSET (skip training positions)
        predictions = prng_func(seed, n_holdout, skip=offset)
        hits = sum(1 for pos, actual in enumerate(holdout_history) 
                   if predictions[pos] % mod == actual)
        results[seed] = hits / n_holdout if n_holdout > 0 else 0.0
    
    return results

def score_survivors(
    seeds: List[int],
    train_history: List[int],
    prng_type: str = 'java_lcg',
    mod: int = 1000,
    batch_size: int = 100,
    forward_survivors: Optional[List[int]] = None,
    reverse_survivors: Optional[List[int]] = None,
    scorer_class = None,
    survivor_metadata: Dict[int, Dict] = None,  # v1.8.2: seed -> metadata mapping
    holdout_history: Optional[List[int]] = None  # v2.0: For computing holdout_hits (y-label)
) -> List[Dict[str, Any]]:
    """
    Score all seeds with full 50-feature extraction.
    
    Args:
        seeds: List of survivor seeds to score
        train_history: Training lottery history
        prng_type: PRNG algorithm type (default: java_lcg)
        mod: Modulo value (default: 1000 for Pick 3)
        batch_size: Progress reporting interval
        forward_survivors: Optional forward sieve survivors for dual-sieve scoring
        reverse_survivors: Optional reverse sieve survivors for dual-sieve scoring
        scorer_class: SurvivorScorer class (passed after delayed import)
    
    Returns:
        List of scored survivor objects with full features
    """
    if scorer_class is None:
        raise RuntimeError("SurvivorScorer class not provided")
    
    # Initialize scorer
    scorer = scorer_class(prng_type=prng_type, mod=mod)
    logger.info(f"Initialized SurvivorScorer with prng_type={prng_type}, mod={mod}")
    
    # Initialize GlobalStateTracker for global features (14 features)
    global_tracker = GlobalStateTracker(train_history, {"mod": mod})
    global_features = global_tracker.get_global_state()
    logger.info(f"GlobalStateTracker initialized: {len(global_features)} global features")
    
    results = []
    total = len(seeds)
    start_time = time.time()
    
    # v1.9.0: GPU-BATCHED scoring - crypto miner style!
    # Process ALL seeds in parallel on GPU, single CPU transfer at end
    logger.info(f"[BATCH-GPU] Starting batched feature extraction for {total} seeds...")
    
    # v2.0: Compute holdout_hits for y-label (if holdout_history provided)
    holdout_hits_map = {}
    if holdout_history:
        logger.info(f"[HOLDOUT] Computing holdout_hits for {total} seeds...")
        holdout_hits_map = compute_holdout_hits_batch(
            seeds=seeds,
            holdout_history=holdout_history,
            train_history_len=len(train_history),  # DERIVED - not configurable
            prng_type=prng_type,
            mod=mod
        )
        logger.info(f"[HOLDOUT] Complete. Mean holdout_hits: {sum(holdout_hits_map.values())/len(holdout_hits_map):.4f}")

    try:
        # Call the new batched method - processes all seeds on GPU in parallel
        all_features = scorer.extract_ml_features_batch(
            seeds=seeds,
            lottery_history=train_history,
            forward_survivors=forward_survivors,
            reverse_survivors=reverse_survivors,
            survivor_metadata=survivor_metadata
        )
        
        # Build result objects from batch results
        gpu_id = int((os.environ.get('CUDA_VISIBLE_DEVICES') or 
                     os.environ.get('HIP_VISIBLE_DEVICES') or 
                     '0').split(',')[0])
        
        for i, (seed, features) in enumerate(zip(seeds, all_features)):
            score = features.get('score', features.get('confidence', 0.0))
            result = {
                'seed': seed,
                'score': float(score),
                'features': {k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in features.items()},
                'metadata': {
                    'prng_type': prng_type,
                    'mod': mod,
                    'worker_hostname': HOST,
                    'worker_gpu': gpu_id,
                    'timestamp': time.time()
                }
            }
            # Merge global features (14 features with global_ prefix)
            for gkey, gval in global_features.items():
                result["features"][f"global_{gkey}"] = float(gval) if isinstance(gval, (int, float)) else gval
            # v2.0: Add holdout_hits (y-label for Step 5)
            result["holdout_hits"] = holdout_hits_map.get(seed, 0.0)
            results.append(result)
        
        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 0
        logger.info(f"[BATCH-GPU] Complete: {total} seeds in {elapsed:.1f}s ({rate:.0f} seeds/sec)")
        
    except Exception as e:
        logger.error(f"[BATCH-GPU] Batch processing failed: {e}, falling back to sequential")
        import traceback
        traceback.print_exc()
        # Fallback to sequential processing if batch fails
        for i, seed in enumerate(seeds):
            try:
                features = scorer.extract_ml_features(
                    seed=seed,
                    lottery_history=train_history,
                    forward_survivors=forward_survivors,
                    reverse_survivors=reverse_survivors
                )
                if survivor_metadata and seed in survivor_metadata:
                    meta = survivor_metadata[seed]
                    for field in ['forward_count', 'reverse_count', 'bidirectional_count',
                                 'skip_min', 'skip_max', 'skip_range']:
                        if field in meta and meta[field] is not None:
                            features[field] = float(meta[field])
                score = features.get('score', features.get('confidence', 0.0))
                # Merge global features (14 features with global_ prefix)
                for gkey, gval in global_features.items():
                    features[f"global_{gkey}"] = float(gval) if isinstance(gval, (int, float)) else gval
                results.append({
                    'seed': seed,
                    'score': float(score),
                    'features': features,
                    'metadata': {'prng_type': prng_type, 'mod': mod, 
                                'worker_hostname': HOST, 'timestamp': time.time()},
                    'holdout_hits': holdout_hits_map.get(seed, 0.0),
                })
            except Exception as e2:
                logger.warning(f"Failed to score seed {seed}: {e2}")
                results.append({'seed': seed, 'score': 0.0, 'features': {}, 'error': str(e2)})
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"[FALLBACK] Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%) | {rate:.1f} seeds/sec")
    
    return results



def main():
    parser = argparse.ArgumentParser(
        description='Full Scoring Worker (Step 3) - Extracts 50 ML features per survivor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python3 full_scoring_worker.py \\
        --seeds-file bidirectional_survivors.json \\
        --train-history train_history.json \\
        --output-file scoring_results/chunk_0001.json \\
        --prng-type java_lcg \\
        --gpu-id 0
        """
    )
    
    # Required arguments
    parser.add_argument('--seeds-file', required=True,
                       help='JSON file containing survivor seeds')
    parser.add_argument('--train-history', required=True,
                       help='JSON file containing training lottery history')
    parser.add_argument('--output-file', required=True,
                       help='Output file for scored survivors')
    
    # Optional arguments
    parser.add_argument('--prng-type', default='java_lcg',
                       help='PRNG type (default: java_lcg)')
    parser.add_argument('--mod', type=int, default=1000,
                       help='Modulo value (default: 1000)')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Progress reporting batch size (default: 100)')
    
    # Optional dual-sieve files
    parser.add_argument('--forward-survivors', default=None,
                       help='Optional: JSON file with forward sieve survivors')
    parser.add_argument('--reverse-survivors', default=None,
                       help='Optional: JSON file with reverse sieve survivors')
    
    # v2.0: Holdout data for y-label
    parser.add_argument('--holdout-history', default=None,
                       help='JSON file with holdout lottery draws for computing holdout_hits (y-label)')
    # Verbose mode
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup GPU environment FIRST - before any CUDA/ROCm imports
    setup_gpu_environment(args.gpu_id)
    logger.info(f"Worker starting on {HOST}, GPU {args.gpu_id}")
    
    # NOW import SurvivorScorer - after GPU environment is configured
    # This ensures CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES is set before torch loads
    try:
        from survivor_scorer import SurvivorScorer
        logger.info("SurvivorScorer imported successfully (after GPU setup)")
    except ImportError as e:
        error_msg = f"Cannot import SurvivorScorer: {e}"
        logger.error(error_msg)
        logger.error("Make sure survivor_scorer.py is in the same directory or PYTHONPATH")
        print(json.dumps({
            "status": "error",
            "error": error_msg,
            "hostname": HOST
        }))
        sys.exit(1)
    
    # Load input data
    try:
        logger.info(f"Loading seeds from {args.seeds_file}...")
        # v1.8.2: Load with metadata for feature merging
        survivors_full = load_survivors_with_metadata(args.seeds_file)
        seeds = [s['seed'] for s in survivors_full]
        survivor_metadata = {s['seed']: s for s in survivors_full}
        logger.info(f"Loaded {len(seeds)} seeds")
        
        logger.info(f"Loading training history from {args.train_history}...")
        train_history = load_lottery_history(args.train_history)
        logger.info(f"Loaded {len(train_history)} lottery draws")
        
        # v1.9.1: Removed loading of forward/reverse survivor files (1.7GB)
        # The metadata is already in the chunk file via survivor_metadata
        forward_survivors = None
        reverse_survivors = None
        logger.info("Using chunk metadata for bidirectional features (no large file loading)")
        
        # v2.0: Load holdout history for computing holdout_hits (y-label)
        holdout_history = None
        if args.holdout_history:
            logger.info(f"Loading holdout history from {args.holdout_history}...")
            holdout_history = load_lottery_history(args.holdout_history)
            logger.info(f"Loaded {len(holdout_history)} holdout draws for y-label computation")
        else:
            logger.warning("No --holdout-history provided, holdout_hits will be 0.0 for all survivors")
            
    except Exception as e:
        error_msg = f"Failed to load input data: {e}"
        logger.error(error_msg)
        print(json.dumps({
            "status": "error",
            "error": error_msg,
            "hostname": HOST
        }))
        sys.exit(1)
    
    # Validate inputs
    if not seeds:
        error_msg = "No seeds found in input file"
        logger.error(error_msg)
        print(json.dumps({
            "status": "error",
            "error": error_msg,
            "hostname": HOST
        }))
        sys.exit(1)
    
    if not train_history:
        error_msg = "No lottery history found in training file"
        logger.error(error_msg)
        print(json.dumps({
            "status": "error",
            "error": error_msg,
            "hostname": HOST
        }))
        sys.exit(1)
    
    # Score all survivors
    try:
        logger.info(f"Starting full scoring of {len(seeds)} survivors...")
        start_time = time.time()
        
        results = score_survivors(
            seeds=seeds,
            train_history=train_history,
            prng_type=args.prng_type,
            mod=args.mod,
            batch_size=args.batch_size,
            forward_survivors=forward_survivors,
            reverse_survivors=reverse_survivors,
            scorer_class=SurvivorScorer,  # Pass the class imported after GPU setup
            survivor_metadata=survivor_metadata,  # v1.8.2: pass metadata for merging
            holdout_history=holdout_history  # v2.0: for computing y-label
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Scoring complete: {len(results)} survivors in {elapsed:.1f}s")
        
    except Exception as e:
        error_msg = f"Scoring failed: {e}"
        logger.error(error_msg)
        print(json.dumps({
            "status": "error",
            "error": error_msg,
            "hostname": HOST
        }))
        sys.exit(1)
    
    # Save results
    try:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        error_msg = f"Failed to save results: {e}"
        logger.error(error_msg)
        print(json.dumps({
            "status": "error",
            "error": error_msg,
            "hostname": HOST
        }))
        sys.exit(1)
    
    # Calculate statistics for summary
    valid_results = [r for r in results if 'error' not in r]
    error_count = len(results) - len(valid_results)
    
    if valid_results:
        scores = [r['score'] for r in valid_results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        # Count features in first valid result
        feature_count = len(valid_results[0].get('features', {}))
    else:
        avg_score = max_score = min_score = 0.0
        feature_count = 0
    
    # Print JSON status for coordinator
    status = {
        "status": "success",
        "count": len(results),
        "valid_count": len(valid_results),
        "error_count": error_count,
        "feature_count": feature_count,
        "avg_score": avg_score,
        "max_score": max_score,
        "min_score": min_score,
        "output_file": str(output_path),
        "hostname": HOST,
        "gpu_id": args.gpu_id,
        "elapsed_seconds": elapsed,
        "seeds_per_second": len(results) / elapsed if elapsed > 0 else 0
    }
    
    print(json.dumps(status))
    
    logger.info("=" * 60)
    logger.info(f"FULL SCORING COMPLETE")
    logger.info(f"  Survivors: {len(results)}")
    logger.info(f"  Features per survivor: {feature_count}")
    logger.info(f"  Avg Score: {avg_score:.4f}")
    logger.info(f"  Score Range: [{min_score:.4f}, {max_score:.4f}]")
    if error_count > 0:
        logger.warning(f"  Errors: {error_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
