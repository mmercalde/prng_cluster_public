#!/usr/bin/env python3
"""
scorer_trial_worker.py (v3.4 - Holdout Sampling Fix)
==================================================
Runs ONE pre-defined scorer trial on a remote worker with intelligent early stopping.

CHANGELOG:
---------
v3.4 (2025-11-29):
- CRITICAL FIX: Holdout evaluation now uses SAMPLED seeds (not full 742K set)
- Training sample_size now correctly applied to BOTH training AND holdout
- Prevents 30+ minute hangs during holdout prediction
- Added sampled_seeds_tensor for GPU-vectorized holdout scoring

v3.3:
- GPU-vectorized scoring with batch_score_vectorized()
- Adaptive memory batching for large seed sets
- Debug logging throughout scoring pipeline

v3.2:
- Added --params-file support for shorter SSH commands
- Backward compatible with inline JSON

v3.1:
- Returns full score list for aggregation
- Saves scores to local JSON result file

PULL ARCHITECTURE:
- Workers do NOT access Optuna database
- Results written to local filesystem
- Coordinator pulls results via SCP
"""
# =============================================================================
# PULL ARCHITECTURE MODIFICATION (2025-11-17)
# =============================================================================
# Workers run trials and write JSON results locally.
# Coordinator (zeus) pulls results from all nodes.
# No Optuna pruning - all trials run to completion.
# =============================================================================

import os
import sys
import json
from utils.survivor_loader import load_survivors
import time
import socket
import logging
from pathlib import Path
from typing import Optional, List, Tuple

# =============================================================================
# GPU ID INJECTION - Must be FIRST before any CUDA/ROCm imports
# =============================================================================
gpu_id = None
hostname = socket.gethostname()

# Extract gpu_id from CLI args
for i, arg in enumerate(sys.argv):
    if arg in ("--gpu-id", "--gpu", "-g") and i + 1 < len(sys.argv):
        gpu_id = int(sys.argv[i + 1])
        break

if gpu_id is not None:
    # AMD ROCm nodes: rig-6600, rig-6600b, rig-6600xt
    if any(x in hostname for x in ["rig-6600", "rig-6600b", "rig-6600xt"]):
        # On ROCm, ALWAYS set HIP from gpu_id, independent of CUDA env
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[Worker Init] {hostname} (ROCm) bound to GPU {gpu_id} via HIP_VISIBLE_DEVICES={gpu_id}")
    else:
        # CUDA host (Zeus): only set CUDA if parent didn't already isolate it
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"[Worker Init] {hostname} (CUDA) bound to GPU {gpu_id} via CUDA_VISIBLE_DEVICES={gpu_id}")
        else:
            print(f"[Worker Init] {hostname} inheriting parent mapping: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}")

# =============================================================================
# Now safe to import CUDA-dependent modules
# =============================================================================
from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
from survivor_scorer import SurvivorScorer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Global data cache (loaded once per worker)
# =============================================================================
survivors = None
train_history = None
holdout_history = None
seeds_to_score = None



def _best_effort_gpu_cleanup():
    """Post-trial GPU memory cleanup (safe, non-invasive)"""
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

def load_data(survivors_file: str, train_history_file: str, holdout_history_file: str):
    """Load data files (cached for reuse across trials on same worker)."""
    global survivors, train_history, holdout_history, seeds_to_score

    if survivors is None:
        logger.info("Loading data (one time)...")
        try:
            survivors_file = os.path.expanduser(survivors_file)
            train_history_file = os.path.expanduser(train_history_file)
            holdout_history_file = os.path.expanduser(holdout_history_file)

            # Load survivors using modular loader (NPZ/JSON auto-detect)
            survivor_result = load_survivors(survivors_file, return_format="array")
            survivors = survivor_result.data
            logger.info(f"Loaded {survivor_result.count:,} survivors from {survivor_result.format} "
                       f"(fallback={survivor_result.fallback_used})")

            with open(train_history_file) as f:
                train_data = json.load(f)
                if isinstance(train_data, list) and len(train_data) > 0 and isinstance(train_data[0], dict):
                    train_history = [d['draw'] for d in train_data]
                else:
                    train_history = train_data

            with open(holdout_history_file) as f:
                holdout_data = json.load(f)
                if isinstance(holdout_data, list) and len(holdout_data) > 0 and isinstance(holdout_data[0], dict):
                    holdout_history = [d['draw'] for d in holdout_data]
                else:
                    holdout_history = holdout_data

            # Extract seeds (modular loader returns array format)
            seeds_to_score = survivors['seeds'].tolist()

            logger.info(f"Loaded {len(seeds_to_score)} survivors/seeds from {survivors_file}.")
            logger.info(f"Loaded {len(train_history)} training draws from {train_history_file}.")
            logger.info(f"Loaded {len(holdout_history)} holdout draws from {holdout_history_file}.")

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            raise

    # Extract PRNG type from survivor metadata
    prng_type = 'java_lcg'
    mod = 1000
    if isinstance(survivors, dict) and 'seeds' in survivors:
        # NPZ format - prng_type from metadata if available
        pass  # Keep defaults, NPZ doesn't store per-survivor prng_type
    elif survivors and len(survivors) > 0 and isinstance(survivors[0], dict):
        prng_type = survivors[0].get('prng_type', 'java_lcg')
        if '_' in prng_type and prng_type.split('_')[-1].isdigit():
            mod = int(prng_type.split('_')[-1])

    return seeds_to_score, train_history, holdout_history, prng_type, mod


def run_trial(seeds_to_score: List[int], train_history: List[int], holdout_history: List[int], 
              params: dict, prng_type: str = 'java_lcg', mod: int = 1000, 
              trial=None, use_legacy_scoring: bool = False) -> Tuple[float, List[float]]:
    """
    Runs the full trial: score seeds, train model, evaluate on holdout.
    
    v3.4 FIX: Holdout evaluation now uses SAMPLED seeds (same as training).
    
    Args:
        seeds_to_score: Full list of seed values
        train_history: Training lottery draws
        holdout_history: Holdout lottery draws  
        params: Trial parameters dict
        prng_type: PRNG type (from config, NOT hardcoded)
        mod: Modulo value for PRNG
        trial: Optional Optuna trial (disabled in PULL mode)
        use_legacy_scoring: If True, use CPU scoring instead of GPU
        
    Returns:
        Tuple[float, List[float]]: (accuracy, holdout_predictions)
    """
    try:
        import torch
        import numpy as np
        
        # =============================================================================
        # CONFIGURATION
        # =============================================================================
        config = ReinforcementConfig()
        
        config.training.update({
            'epochs': 25,
            'batch_size': params.get('batch_size', 128),
            'learning_rate': params.get('learning_rate', 0.001),
            'early_stopping_patience': 5,
            'sample_size': params.get('sample_size', 50000)
        })
        
        config.model.update({
            'hidden_layers': [int(x) for x in params.get('hidden_layers', '128_64').split('_')],
            'dropout': params.get('dropout', 0.3)
        })

        # =============================================================================
        # INITIALIZE ENGINE
        # =============================================================================
        logger.info("Initializing Mini-Run Engine...")
        engine = ReinforcementEngine(config, lottery_history=train_history)
        
        # Determine device
        device = engine.device if hasattr(engine, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configure scorer with trial parameters (PRNG type from config!)
        trial_scorer_params = {
            'residue_mod_1': params.get('residue_mod_1', 10),
            'residue_mod_2': params.get('residue_mod_2', 100),
            'residue_mod_3': params.get('residue_mod_3', 1000),
            'max_offset': params.get('max_offset', 10),
            'temporal_window_size': params.get('temporal_window_size', 100),
            'temporal_num_windows': params.get('temporal_num_windows', 5),
            'min_confidence_threshold': params.get('min_confidence_threshold', 0.15)
        }
        
        engine.scorer = SurvivorScorer(prng_type=prng_type, mod=mod, config_dict=trial_scorer_params)
        logger.info(f"Scorer configured with PRNG={prng_type}, mod={mod}, params={trial_scorer_params}")

        # =============================================================================
        # SCORING SECTION (GPU-Vectorized)
        # =============================================================================
        logger.info(f"🔍 [DEBUG] Scoring {len(seeds_to_score)} seeds on training data...")
        logger.info(f"🔍 [DEBUG] Use legacy scoring: {use_legacy_scoring}")
        
        if use_legacy_scoring:
            # Legacy CPU scoring (for debugging/comparison)
            logger.info("⚠️ Using LEGACY CPU scoring method")
            t_start = time.time()
            y_train = engine.scorer.batch_score(
                seeds=seeds_to_score,
                lottery_history=train_history,
                use_dual_gpu=False
            )
            if y_train and isinstance(y_train[0], dict):
                y_train = [item["score"] for item in y_train]
            logger.info(f"Legacy scoring completed in {time.time()-t_start:.2f}s")
        else:
            # GPU-vectorized scoring (default - fast!)
            logger.info("✅ [DEBUG-VECTOR] Using GPU-vectorized batch_score_vectorized() method")
            logger.info(f"🔍 [DEBUG-VECTOR] PyTorch version: {torch.__version__}")
            logger.info(f"🔍 [DEBUG-VECTOR] CUDA available: {torch.cuda.is_available()}")
            logger.info(f"🔍 [DEBUG-VECTOR] Selected device: {device}")
            
            # Create tensors on GPU
            logger.info(f"🔍 [DEBUG-VECTOR] Creating tensors...")
            logger.info(f"🔍 [DEBUG-VECTOR]   seeds_to_score: {len(seeds_to_score)} items")
            logger.info(f"🔍 [DEBUG-VECTOR]   train_history: {len(train_history)} items")
            
            t_start = time.time()
            seeds_tensor = torch.tensor(seeds_to_score, dtype=torch.int64, device=device)
            logger.info(f"🔍 [DEBUG-VECTOR] Seeds tensor created in {time.time()-t_start:.3f}s, shape={seeds_tensor.shape}, device={seeds_tensor.device}")
            
            t_start = time.time()
            history_tensor = torch.tensor(train_history, dtype=torch.int64, device=device)
            logger.info(f"🔍 [DEBUG-VECTOR] History tensor created in {time.time()-t_start:.3f}s, shape={history_tensor.shape}, device={history_tensor.device}")
            
            # Adaptive batching for memory safety
            logger.info(f"🔍 [DEBUG-VECTOR] Using adaptive batching for memory safety...")
            
            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                history_len = len(history_tensor)
                bytes_per_seed = max(history_len, 100) * 8
                allocated_gb = torch.cuda.memory_allocated(0) / 1e9
                free_mem_gb = gpu_mem_gb - allocated_gb
                usable_mem = free_mem_gb * 0.30 * 1e9
                batch_size = int(usable_mem / bytes_per_seed)
                batch_size = max(10_000, min(batch_size, 100_000))
            else:
                batch_size = 50_000
            
            # Process in batches
            t_start = time.time()
            total_seeds = len(seeds_tensor)
            num_batches = (total_seeds + batch_size - 1) // batch_size
            all_scores = []
            
            logger.info(f"🔍 [DEBUG-VECTOR] Processing {total_seeds:,} seeds in {num_batches} batch(es)...")
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_seeds)
                batch_seeds = seeds_tensor[batch_start:batch_end]
                
                logger.info(f"🔍 [DEBUG-VECTOR] Batch {batch_idx+1}/{num_batches}: {batch_end - batch_start:,} seeds...")
                
                batch_t_start = time.time()
                batch_scores = engine.scorer.batch_score_vectorized(
                    seeds=batch_seeds,
                    lottery_history=history_tensor,
                    device=device,
                    return_dict=False
                )
                batch_elapsed = time.time() - batch_t_start
                
                logger.info(f"🔍 [DEBUG-VECTOR] Batch {batch_idx+1} completed in {batch_elapsed:.2f}s ({(batch_end-batch_start)/batch_elapsed:.0f} seeds/sec)")
                all_scores.append(batch_scores)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate all batch scores
            scores_tensor = torch.cat(all_scores)
            elapsed = time.time() - t_start
            logger.info(f"🔍 [DEBUG-VECTOR] Total scoring completed in {elapsed:.2f}s ({total_seeds/elapsed:.0f} seeds/sec overall)")
            logger.info(f"🔍 [DEBUG-VECTOR] scores_tensor shape: {scores_tensor.shape}, device: {scores_tensor.device}")
            
            # Convert to numpy
            logger.info(f"🔍 [DEBUG-VECTOR] Converting scores to numpy...")
            t_start = time.time()
            y_train = scores_tensor.cpu().numpy().tolist()
            logger.info(f"🔍 [DEBUG-VECTOR] Conversion complete in {time.time()-t_start:.3f}s")
            logger.info(f"🔍 [DEBUG-VECTOR] y_train length: {len(y_train)}, mean score: {np.mean(y_train):.6f}")

        # =============================================================================
        # SAMPLING (if configured)
        # =============================================================================
        sample_size = config.training.get('sample_size', None)
        
        if sample_size and len(seeds_to_score) > sample_size:
            import random
            logger.info(f"📊 Sampling {sample_size:,} seeds from {len(seeds_to_score):,} for training")
            random.seed(42)  # Reproducible sampling
            sample_indices = random.sample(range(len(seeds_to_score)), sample_size)
            sampled_seeds = [seeds_to_score[i] for i in sample_indices]
            sampled_scores = [y_train[i] for i in sample_indices]
            
            # v3.4 FIX: Create sampled tensor for holdout evaluation
            sampled_seeds_tensor = torch.tensor(sampled_seeds, dtype=torch.int64, device=device)
            logger.info(f"🔍 [DEBUG] Created sampled_seeds_tensor: {sampled_seeds_tensor.shape}")
        else:
            logger.info(f"📊 Using all {len(seeds_to_score):,} seeds for training")
            sampled_seeds = seeds_to_score
            sampled_scores = y_train
            sampled_seeds_tensor = seeds_tensor  # Use full tensor

        # =============================================================================
        # TRAINING SECTION
        # =============================================================================
        logger.info(f"🔍 [DEBUG] Starting mini-run training on {len(sampled_seeds):,} seeds...")
        start_train = time.time()
        
        try:
            engine.train(
                survivors=sampled_seeds,
                actual_results=sampled_scores,
                lottery_history=train_history,
                epoch_callback=None  # No Optuna pruning in PULL mode
            )
            train_time = time.time() - start_train
            logger.info(f"🔍 [DEBUG] Training completed in {train_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

        # =============================================================================
        # HOLDOUT EVALUATION (v3.4 FIX: Use SAMPLED seeds!)
        # =============================================================================
        logger.info("🔍 [DEBUG] Evaluating on holdout (vectorized)...")
        logger.info(f"🔍 [DEBUG] Using {len(sampled_seeds):,} SAMPLED seeds for holdout (NOT full {len(seeds_to_score):,})")
        
        # Create holdout history tensor
        holdout_tensor = torch.tensor(holdout_history, dtype=torch.int64, device=device)
        
        # v3.4 FIX: Score SAMPLED seeds on holdout (not full set!)
        holdout_scores_tensor = engine.scorer.batch_score_vectorized(
            seeds=sampled_seeds_tensor,  # v3.4 FIX: Use sampled tensor!
            lottery_history=holdout_tensor,
            device=device,
            return_dict=False
        )
        
        y_holdout = holdout_scores_tensor.cpu().numpy().tolist()
        logger.info(f"🔍 [DEBUG] Holdout scoring completed (vectorized), {len(y_holdout)} scores")
        
        # v3.4 FIX: Predict on SAMPLED seeds (not full set!)
        y_pred_holdout = engine.predict_quality_batch(
            survivors=sampled_seeds,  # v3.4 FIX: Use sampled seeds list!
            lottery_history=holdout_history
        )
        
        logger.info(f"🔍 [DEBUG] Holdout prediction completed, {len(y_pred_holdout)} predictions")
        
        # Calculate MSE
        holdout_mse = float(torch.nn.functional.mse_loss(
            torch.tensor(y_pred_holdout),
            torch.tensor(y_holdout)
        ))
        
        accuracy = -holdout_mse  # Negative MSE (higher is better)
        
        logger.info(f"Holdout MSE: {holdout_mse:.6f}, Accuracy (NegMSE): {accuracy:.6f}")
        
        return accuracy, y_pred_holdout if isinstance(y_pred_holdout, list) else y_pred_holdout.tolist()

    except Exception as e:
        logger.error(f"Trial execution failed: {e}", exc_info=True)
        raise


def save_local_result(trial_id: int, params: dict, accuracy: float, state: str, 
                      error: Optional[str], scores: Optional[list] = None):
    """Save trial result to local filesystem for coordinator to pull later."""
    local_results_dir = Path.home() / "distributed_prng_analysis" / "scorer_trial_results"
    local_results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = local_results_dir / f"trial_{trial_id:04d}.json"
    
    result = {
        "trial_id": trial_id,
        "params": params,
        "accuracy": accuracy,
        "status": state,
        "error": error,
        "hostname": socket.gethostname(),
        "timestamp": time.time()
    }
    
    if scores is not None:
        result['scores'] = scores
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✅ Result saved locally to {output_file}")


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    
    survivors_file = sys.argv[1]
    train_history_file = sys.argv[2]
    holdout_history_file = sys.argv[3]
    trial_id = int(sys.argv[4])
    
    params_json = None
    params_file = None
    study_name = None
    study_db = None
    use_legacy_scoring = False
    
    # Parse optional arguments
    i = 5
    while i < len(sys.argv):
        if sys.argv[i] == '--params-file' and i + 1 < len(sys.argv):
            params_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--params-json' and i + 1 < len(sys.argv):
            params_json = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--optuna-study-name' and i + 1 < len(sys.argv):
            study_name = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--optuna-study-db' and i + 1 < len(sys.argv):
            study_db = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--gpu-id':
            # Already handled at top of file
            i += 2
        elif sys.argv[i] == '--use-legacy-scoring':
            use_legacy_scoring = True
            i += 1
        elif i == 5 and not sys.argv[i].startswith('--'):
            # BACKWARD COMPATIBILITY: Position 5 without flag = inline JSON
            params_json = sys.argv[i]
            i += 1
        else:
            i += 1
    
    # Load parameters
    params = None
    
    if params_file:
        # v3.2+: Load from job file
        try:
            with open(params_file) as f:
                jobs = json.load(f)
            for job in jobs:
                if job.get('job_id') == f'scorer_trial_{trial_id}':
                    args = job.get('args', [])
                    for idx, arg in enumerate(args):
                        if arg.startswith('{') and arg.endswith('}'):
                            params = json.loads(arg)
                            break
                    break
            if params is None:
                raise ValueError(f"Trial {trial_id} not found in {params_file}")
            logger.info(f"Loading parameters from file: {params_file}")
            logger.info(f"Parameters loaded from file: {params}")
        except Exception as e:
            logger.error(f"Failed to load params from file: {e}")
            raise
            
    elif params_json:
        # Legacy: Inline JSON
        params = json.loads(params_json)
        logger.info(f"Parameters: {params}")
    else:
        raise ValueError("Must provide either --params-file or --params-json (or inline JSON at position 5)")
    
    # Run trial
    try:
        seeds, train_hist, holdout_hist, prng_type, mod = load_data(
            survivors_file, train_history_file, holdout_history_file
        )
        
        trial = None  # No Optuna in PULL mode
        
        accuracy, scores = run_trial(
            seeds, train_hist, holdout_hist, params,
            prng_type=prng_type, mod=mod, trial=trial,
            use_legacy_scoring=use_legacy_scoring
        )
        
        save_local_result(trial_id, params, accuracy, "success", None, scores=scores)
        
        _best_effort_gpu_cleanup()
        print(json.dumps({"status": "success", "trial_id": trial_id, "accuracy": accuracy}))
        sys.stdout.flush()
        sys.exit(0)
        
    except Exception as e:
        if "pruned" in str(e).lower():
            logger.info(f"⚡ Trial {trial_id} was pruned")
            save_local_result(trial_id, params, float('-inf'), "pruned", "Trial pruned", scores=None)
            _best_effort_gpu_cleanup()
            print(json.dumps({"status": "pruned", "trial_id": trial_id}))
            sys.stdout.flush()
            sys.exit(0)
        else:
            error_msg = str(e)
            logger.error(f"❌ Trial {trial_id} failed: {error_msg}", exc_info=True)
            if params is None:
                params = {"error": "Failed to parse params"}
            save_local_result(trial_id, params, float('-inf'), "error", error_msg, scores=None)
            _best_effort_gpu_cleanup()
            print(json.dumps({"status": "error", "trial_id": trial_id, "error": error_msg}))
            sys.stdout.flush()
            sys.exit(1)


if __name__ == "__main__":
    main()
