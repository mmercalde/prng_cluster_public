#!/usr/bin/env python3
"""
scorer_trial_worker.py (v3.4-DEBUG - Comprehensive Logging BOTH METHODS)
==================================================
DEBUG VERSION: Adds extensive logging to BOTH legacy and vectorized methods

All original functionality preserved - only adds logging!
"""
# =============================================================================
# ROCm environment setup - MUST BE FIRST
# =============================================================================
import os
import socket
HOST = socket.gethostname()

if HOST in ["rig-6600", "rig-6600b"]:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")

os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")

# =============================================================================
# GPU BINDING - MUST COME AFTER ROCm ENV SETUP
# =============================================================================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", type=int, default=None)
args, _ = parser.parse_known_args()

if args.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu_id)

print(f"[Worker Init] {socket.gethostname()} bound to GPU {args.gpu_id} "
      f"via CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

import sys
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
from survivor_scorer import SurvivorScorer

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global data
survivors = None
train_history = None
holdout_history = None
seeds_to_score = None


def load_data(survivors_file, train_history_file, holdout_history_file):
    """Load all necessary data from local paths."""
    global survivors, train_history, holdout_history, seeds_to_score

    if survivors is None:
        logger.info("Loading data (one time)...")
        try:
            survivors_file = os.path.expanduser(survivors_file)
            train_history_file = os.path.expanduser(train_history_file)
            holdout_history_file = os.path.expanduser(holdout_history_file)

            with open(survivors_file) as f:
                survivors = json.load(f)

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

            if isinstance(survivors, list) and len(survivors) > 0:
                if isinstance(survivors[0], dict):
                    seeds_to_score = [s.get('seed', s) for s in survivors]
                else:
                    seeds_to_score = survivors
            else:
                 seeds_to_score = []

            logger.info(f"Loaded {len(seeds_to_score)} survivors/seeds from {survivors_file}.")
            logger.info(f"Loaded {len(train_history)} training draws from {train_history_file}.")
            logger.info(f"Loaded {len(holdout_history)} holdout draws from {holdout_history_file}.")

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            raise

    # Extract PRNG type
    prng_type = 'java_lcg'
    mod = 1000
    if survivors and len(survivors) > 0 and isinstance(survivors[0], dict):
        prng_type = survivors[0].get('prng_type', 'java_lcg')
        if '_' in prng_type and prng_type.split('_')[-1].isdigit():
            mod = int(prng_type.split('_')[-1])

    return seeds_to_score, train_history, holdout_history, prng_type, mod


def run_trial(seeds_to_score, train_history, holdout_history, params, prng_type='java_lcg', mod=1000, trial=None, use_legacy_scoring=False):
    """
    Runs the full test and returns the accuracy score AND the full score list.
    
    DEBUG VERSION: Extensive logging added to BOTH legacy and vectorized paths!
    """
    try:
        from dataclasses import replace
        config = ReinforcementConfig()

        config.training.update({
            'epochs': 25,
            'batch_size': params.get('batch_size', 128),
            'learning_rate': params.get('learning_rate', 0.001),
            'early_stopping_patience': 5,
            'sample_size': params.get('sample_size', 50000)  # Configurable training sample size
        })

        config.model.update({
            'hidden_layers': [int(x) for x in params.get('hidden_layers', '128_64').split('_')],
            'dropout': params.get('dropout', 0.3)
        })

        if hasattr(config, 'survivor_pool'):
            config.survivor_pool.update({
                'prune_threshold': 0.3,
                'cache_scores': True
            })

        logger.info("Initializing Mini-Run Engine...")
        engine = ReinforcementEngine(config, lottery_history=train_history)

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
        # SCORING SECTION - DEBUG INSTRUMENTATION FOR BOTH METHODS
        # =============================================================================
        logger.info(f"🔍 [DEBUG] Scoring {len(seeds_to_score)} seeds on training data...")
        logger.info(f"🔍 [DEBUG] Use legacy scoring: {use_legacy_scoring}")

        if use_legacy_scoring:
            # =============================================================================
            # LEGACY METHOD WITH DEBUG LOGGING
            # =============================================================================
            logger.info("⚠️  [DEBUG-LEGACY] Using LEGACY batch_score() method")
            logger.info(f"🔍 [DEBUG-LEGACY] Starting batch_score call...")
            logger.info(f"🔍 [DEBUG-LEGACY]   Seeds count: {len(seeds_to_score)}")
            logger.info(f"🔍 [DEBUG-LEGACY]   History length: {len(train_history)}")
            logger.info(f"🔍 [DEBUG-LEGACY]   use_dual_gpu: False")
            
            try:
                t_start = time.time()
                y_train = engine.scorer.batch_score(
                    seeds=seeds_to_score,
                    lottery_history=train_history,
                    use_dual_gpu=False
                )
                elapsed = time.time() - t_start
                logger.info(f"🔍 [DEBUG-LEGACY] batch_score completed in {elapsed:.3f}s")
                logger.info(f"🔍 [DEBUG-LEGACY] Result type: {type(y_train)}, length: {len(y_train)}")
                
                # Extract scores
                if y_train and isinstance(y_train[0], dict):
                    logger.info(f"🔍 [DEBUG-LEGACY] Extracting scores from dict format...")
                    y_train = [item["score"] if isinstance(item, dict) else item for item in y_train]
                    logger.info(f"🔍 [DEBUG-LEGACY] Extraction complete, scores count: {len(y_train)}")
                
                import numpy as np
                logger.info(f"🔍 [DEBUG-LEGACY] Score statistics: mean={np.mean(y_train):.6f}, min={np.min(y_train):.6f}, max={np.max(y_train):.6f}")
                
            except Exception as e:
                logger.error(f"❌ [DEBUG-LEGACY] batch_score FAILED!")
                logger.error(f"❌ [DEBUG-LEGACY] Exception type: {type(e).__name__}")
                logger.error(f"❌ [DEBUG-LEGACY] Exception message: {str(e)}")
                logger.error(f"❌ [DEBUG-LEGACY] Full traceback:", exc_info=True)
                raise
                
        else:
            # =============================================================================
            # VECTORIZED METHOD WITH DEBUG LOGGING
            # =============================================================================
            logger.info("✅ [DEBUG-VECTOR] Using GPU-vectorized batch_score_vectorized() method")
            
            try:
                import torch
                import numpy as np
                
                logger.info(f"🔍 [DEBUG-VECTOR] PyTorch imported successfully")
                logger.info(f"🔍 [DEBUG-VECTOR] PyTorch version: {torch.__version__}")
                logger.info(f"🔍 [DEBUG-VECTOR] CUDA available: {torch.cuda.is_available()}")
                
                if torch.cuda.is_available():
                    logger.info(f"🔍 [DEBUG-VECTOR] CUDA device count: {torch.cuda.device_count()}")
                    logger.info(f"🔍 [DEBUG-VECTOR] Current CUDA device: {torch.cuda.current_device()}")
                    try:
                        logger.info(f"🔍 [DEBUG-VECTOR] CUDA device name: {torch.cuda.get_device_name()}")
                    except:
                        logger.warning(f"🔍 [DEBUG-VECTOR] Could not get CUDA device name")
                
                logger.info(f"🔍 [DEBUG-VECTOR] Environment variables:")
                logger.info(f"🔍 [DEBUG-VECTOR]   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
                logger.info(f"🔍 [DEBUG-VECTOR]   HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'NOT SET')}")
                
                # Determine device
                logger.info(f"🔍 [DEBUG-VECTOR] Checking engine.device attribute...")
                device = engine.device if hasattr(engine, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
                logger.info(f"🔍 [DEBUG-VECTOR] Selected device: {device}")
                
                # Create tensors
                logger.info(f"🔍 [DEBUG-VECTOR] Creating tensors...")
                logger.info(f"🔍 [DEBUG-VECTOR]   seeds_to_score: {len(seeds_to_score)} items")
                logger.info(f"🔍 [DEBUG-VECTOR]   train_history: {len(train_history)} items")
                
                t_start = time.time()
                seeds_tensor = torch.tensor(seeds_to_score, dtype=torch.int64, device=device)
                logger.info(f"🔍 [DEBUG-VECTOR] Seeds tensor created in {time.time()-t_start:.3f}s, shape={seeds_tensor.shape}, device={seeds_tensor.device}")
                
                t_start = time.time()
                history_tensor = torch.tensor(train_history, dtype=torch.int64, device=device)
                logger.info(f"🔍 [DEBUG-VECTOR] History tensor created in {time.time()-t_start:.3f}s, shape={history_tensor.shape}, device={history_tensor.device}")
                
                # Call vectorized scoring
                # Call vectorized scoring with ADAPTIVE BATCHING
                logger.info(f"🔍 [DEBUG-VECTOR] Using adaptive batching for memory safety...")
                
                # Calculate optimal batch size based on GPU memory
                if torch.cuda.is_available():
                    gpu_props = torch.cuda.get_device_properties(0)
                    gpu_mem_gb = gpu_props.total_memory / 1e9
                    history_len = len(history_tensor)
                    
                    # Estimate: ~8 bytes per seed per history item (float64)
                    bytes_per_seed = max(history_len, 100) * 8  # Min 100 to avoid edge case
                    
                    # Check existing memory usage and use conservative estimate
                    allocated_gb = torch.cuda.memory_allocated(0) / 1e9
                    # Use only 30% of remaining free memory (conservative for pre-loaded engine)
                    free_mem_gb = gpu_mem_gb - allocated_gb
                    usable_mem = free_mem_gb * 0.30 * 1e9
                    batch_size = int(usable_mem / bytes_per_seed)

                    # Clamp to reasonable range: min 10K, max 100K (safer for pre-loaded engine)
                    batch_size = max(10_000, min(batch_size, 100_000))
                else:
                    batch_size = 50_000  # CPU fallback
                    logger.info(f"🔍 [DEBUG-VECTOR] CPU mode: using {batch_size:,} batch size")
                
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
                    
                    # Clear GPU cache between batches to prevent fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Concatenate all batch scores into single tensor
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
                
            except Exception as e:
                logger.error(f"❌ [DEBUG-VECTOR] Vectorized scoring FAILED!")
                logger.error(f"❌ [DEBUG-VECTOR] Exception type: {type(e).__name__}")
                logger.error(f"❌ [DEBUG-VECTOR] Exception message: {str(e)}")
                logger.error(f"❌ [DEBUG-VECTOR] Full traceback:", exc_info=True)
                raise

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
        else:
            logger.info(f"📊 Using all {len(seeds_to_score):,} seeds for training")
            sampled_seeds = seeds_to_score
            sampled_scores = y_train

        # =============================================================================
        # TRAINING SECTION
        # =============================================================================
        logger.info(f"🔍 [DEBUG] Starting mini-run training on {len(sampled_seeds):,} seeds...")
        start_train = time.time()

        def pruning_callback(epoch: int, val_loss: float) -> bool:
            if trial is not None and epoch % 2 == 0 and epoch >= 6:
                trial.report(-val_loss, epoch)
                if trial.should_prune():
                    logger.info(f"⚡ Trial pruned at epoch {epoch+1} (val_loss={val_loss:.6f})")
                    return False
            return True

        try:
            engine.train(
                survivors=sampled_seeds,
                actual_results=sampled_scores,
                lottery_history=train_history,
                epoch_callback=pruning_callback if trial else None
            )
            train_time = time.time() - start_train
            logger.info(f"🔍 [DEBUG] Training completed in {train_time:.1f}s")

        except Exception as e:
            # Just re-raise - don't try to convert to TrialPruned since optuna not imported
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

        # =============================================================================
        # HOLDOUT EVALUATION (GPU-VECTORIZED)
        # =============================================================================
        logger.info("🔍 [DEBUG] Evaluating on holdout (vectorized)...")
        
        # Use the SAME vectorized method as training scoring
        import torch
        holdout_tensor = torch.tensor(holdout_history, dtype=torch.int64, device=device)
        
        # Score on holdout using GPU vectorization (fast!)
        holdout_scores_tensor = engine.scorer.batch_score_vectorized(
            seeds=seeds_tensor,  # Already on GPU from training
            lottery_history=holdout_tensor,
            device=device,
            return_dict=False
        )
        
        # Convert to list
        y_holdout = holdout_scores_tensor.cpu().numpy().tolist()
        logger.info(f"🔍 [DEBUG] Holdout scoring completed (vectorized)")

        y_pred_holdout = engine.predict_quality_batch(
            survivors=seeds_to_score,
            lottery_history=holdout_history
        )

        import torch
        holdout_mse = float(torch.nn.functional.mse_loss(
            torch.tensor(y_pred_holdout),
            torch.tensor(y_holdout)
        ))

        accuracy = -holdout_mse

        logger.info(f"Holdout MSE: {holdout_mse:.6f}, Accuracy (NegMSE): {accuracy:.6f}")

        return accuracy, y_pred_holdout if isinstance(y_pred_holdout, list) else y_pred_holdout.tolist()

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial execution failed: {e}", exc_info=True)
        raise


def save_local_result(trial_id: int, params: dict, accuracy: float, state: str, error: Optional[str], scores: Optional[list] = None):
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
        elif sys.argv[i] == '--use-legacy-scoring':
            use_legacy_scoring = True
            i += 1
        elif sys.argv[i] == '--gpu-id':
            i += 2
        elif i == 5 and not sys.argv[i].startswith('--'):
            params_json = sys.argv[i]
            i += 1
        else:
            i += 1

    try:
        gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', -1))
    except (ValueError, TypeError):
        gpu_id = -1

    hostname = socket.gethostname()
    logger.info(f"--- Scorer Trial Worker {trial_id} Starting on {hostname} (GPU:{gpu_id}) ---")
    if study_name and study_db:
        logger.info(f"   Optuna Study: {study_name} @ {study_db}")

    params = None
    scores = None

    try:
        if params_file:
            logger.info(f"Loading parameters from file: {params_file}")
            with open(params_file, 'r') as f:
                jobs_data = json.load(f)
            my_job = None
            for job in jobs_data:
                if job['job_id'] == f'scorer_trial_{trial_id}':
                    my_job = job
                    break
            if my_job is None:
                raise ValueError(f"Could not find job 'scorer_trial_{trial_id}' in {params_file}")
            params = json.loads(my_job['args'][4])
            logger.info(f"Parameters loaded from file: {params}")
        elif params_json:
            params = json.loads(params_json)
            logger.info(f"Parameters: {params}")
        else:
            raise ValueError("Must provide either --params-file or --params-json (or inline JSON at position 5)")

        seeds, train_hist, holdout_hist, prng_type, mod = load_data(survivors_file, train_history_file, holdout_history_file)

        trial = None

        accuracy, scores = run_trial(seeds, train_hist, holdout_hist, params, prng_type=prng_type, mod=mod, trial=trial, use_legacy_scoring=use_legacy_scoring)

        save_local_result(trial_id, params, accuracy, "success", None, scores=scores)

        print(json.dumps({"status": "success", "trial_id": trial_id, "accuracy": accuracy}))
        sys.exit(0)

    except Exception as e:
        if "TrialPruned" in str(type(e).__name__) or "pruned" in str(e).lower():
            logger.info(f"⚡ Trial {trial_id} was pruned (early stopped)")
        save_local_result(trial_id, params, float('-inf'), "pruned", "Trial pruned by Optuna", scores=None)
        print(json.dumps({"status": "pruned", "trial_id": trial_id}))
        sys.exit(0)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Trial {trial_id} failed: {error_msg}", exc_info=True)
        if params is None:
            params = {"error": "Failed to parse params_json"}
        save_local_result(trial_id, params, float('-inf'), "error", error_msg, scores=None)
        print(json.dumps({"status": "error", "trial_id": trial_id, "error": error_msg}))
        sys.exit(1)


if __name__ == "__main__":
    main()
