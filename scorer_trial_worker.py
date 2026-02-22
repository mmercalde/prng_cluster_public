#!/usr/bin/env python3
"""
scorer_trial_worker.py (v4.0 - WSI objective, draw-history-free)
==================================================
v4.0 (2026-02-21):
- ARCHITECTURE: Remove draw history from Step 2 (TB ruling S102/S103)
- NEW OBJECTIVE: WSI bounded [-1,1]; all 5 Optuna params active
- REMOVE: ReinforcementEngine, SurvivorScorer
- PRESERVE: per-trial RNG (S101), prng_type from config (S102)
- CLI: positional args 2+3 accepted but ignored

v3.6 (2026-02-21):
- BUG FIX: NPZ branch never read prng_type from config

v3.5 (2026-02-20):
- BUG FIX: Replace neg-MSE objective with Spearman rank correlation
  (MSE collapsed to constant on low-variance score distributions — S101)
- BUG FIX: Remove random.seed(42) — replaced with per-trial seed
  (seed=42 locked all 100 trials to identical 450 seeds, 2.6% pool coverage)
  New: random.seed(params['optuna_trial_number']) — unique per trial,
  stable for retries, ~93% survivor pool coverage across 100 trials
- Team Beta Mod 1: guard y_holdout degeneracy (constant scorer output)
- Team Beta Mod 2: runtime SciPy import guard, best-effort non-fatal
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
import atexit
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
    if any(x in hostname for x in ["rig-6600", "rig-6600b", "rig-6600c", "rig-6600xt"]):
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
# v4.0: ReinforcementEngine + SurvivorScorer removed (WSI uses NPZ signals only)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Global data cache (loaded once per worker)
# =============================================================================
survivors = None
seeds_to_score = None
npz_forward_matches = None   # float32 ndarray -- quality signal from NPZ
npz_reverse_matches = None   # float32 ndarray -- quality signal from NPZ



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
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass

# Register cleanup for crash safety
atexit.register(_best_effort_gpu_cleanup)

def load_data(survivors_file: str,
              train_history_file: str = None,
              holdout_history_file: str = None):
    """
    v4.0: Load NPZ survivor data only -- draw history ignored.
    train_history_file / holdout_history_file accepted for CLI compat.
    """
    global survivors, seeds_to_score, npz_forward_matches, npz_reverse_matches

    if survivors is None:
        logger.info('Loading NPZ survivor data (one time)...')
        try:
            survivors_file = os.path.expanduser(survivors_file)

            # survivor_loader.data is plain Dict[str, np.ndarray] (no structured array)
            survivor_result = load_survivors(survivors_file, return_format='array')
            survivors = survivor_result.data
            logger.info(
                f'Loaded {survivor_result.count:,} survivors from '
                f'{survivor_result.format} (fallback={survivor_result.fallback_used})'
            )

            if not isinstance(survivors, dict) or 'seeds' not in survivors:
                raise ValueError(
                    f'Unexpected survivors type: {type(survivors)}. '
                    'Expected Dict[str, np.ndarray] from survivor_loader.'
                )

            seeds_to_score = survivors['seeds'].tolist()
            logger.info(f'Loaded {len(seeds_to_score):,} seeds.')

            import numpy as _np
            if 'forward_matches' in survivors and 'reverse_matches' in survivors:
                npz_forward_matches = survivors['forward_matches'].astype(_np.float32)
                npz_reverse_matches = survivors['reverse_matches'].astype(_np.float32)
                logger.info(
                    f'NPZ quality signals: '
                    f'fwd mean={npz_forward_matches.mean():.4f}  '
                    f'rev mean={npz_reverse_matches.mean():.4f}'
                )
            else:
                raise RuntimeError(
                    'NPZ missing forward_matches or reverse_matches. '
                    'Re-run convert_survivors_to_binary.py with NPZ v3.0+ format. '
                    f'Available NPZ keys: {list(survivors.keys())}'
                )

        except Exception as e:
            logger.error(f'Failed to load data: {e}', exc_info=True)
            raise

    if npz_forward_matches is None or npz_reverse_matches is None:
        raise RuntimeError('NPZ quality signals are None after load -- cannot compute WSI.')

    # prng_type from optimal_window_config.json (canonical source, S102)
    prng_type = 'java_lcg'
    mod = 1000
    wc_path = os.path.join(
        os.path.dirname(os.path.abspath(survivors_file)), 'optimal_window_config.json'
    )
    if os.path.exists(wc_path):
        try:
            with open(wc_path) as _wf:
                _wc = json.load(_wf)
            prng_type = _wc.get('prng_type') or 'java_lcg'
            mod       = _wc.get('mod')       or 1000
            logger.info(
                f'Pipeline config: prng_type={prng_type}, mod={mod} '
                '(from optimal_window_config.json)'
            )
        except Exception as _e:
            logger.warning(f'Could not read optimal_window_config.json: {_e} -- using defaults')

    return seeds_to_score, npz_forward_matches, npz_reverse_matches, prng_type, mod



def run_trial(seeds_to_score,
              npz_forward_matches,
              npz_reverse_matches,
              params,
              prng_type='java_lcg',
              mod=1000,
              trial=None,
              use_legacy_scoring=False):
    """
    v4.0: WSI (Weighted Separation Index) objective -- draw-history-free.

    ALL 5 Optuna params affect the objective (TB issue 3 resolved):
      rm1, rm2, rm3  -> normalized mixture weights for fwd/rev/interaction
      max_offset     -> squared-interaction term weight (wi = offset/15)
      temporal_window_size -> temporal smoothing weight (tw = size/200)

    Scoring:
        scores = wf*fwd + wr*rev + w3*(fwd*rev) + tw*(fwd+rev)/2 + wi*(fwd*rev)**2

    WSI formula (bounded [-1,1], TB-approved S103):
        quality = fwd * rev
        WSI = cov(scores,quality) / ((std_s+eps)*(std_q+eps))

    Degenerate guard: std(scores) < 1e-12 -> WSI = -1.0
    """
    try:
        import numpy as np
        import random

        # Sampling -- per-trial RNG preserved from S101
        n_seeds     = len(seeds_to_score)
        sample_size = params.get('sample_size', 50000)

        if n_seeds > sample_size:
            random.seed(params.get('optuna_trial_number', 0))
            sample_idx  = random.sample(range(n_seeds), sample_size)
            sample_idx  = np.array(sample_idx, dtype=np.int64)
            sampled_fwd = npz_forward_matches[sample_idx]
            sampled_rev = npz_reverse_matches[sample_idx]
            logger.info(
                f'Sampled {sample_size:,} / {n_seeds:,} seeds '
                f'(rng_seed={params.get("optuna_trial_number", 0)})'
            )
        else:
            sampled_fwd = npz_forward_matches
            sampled_rev = npz_reverse_matches
            logger.info(f'Using all {n_seeds:,} seeds')

        # Parametric scoring -- all 5 params active
        rm1        = float(params.get('residue_mod_1',   10))
        rm2        = float(params.get('residue_mod_2',  100))
        rm3        = float(params.get('residue_mod_3', 1000))
        max_offset = float(params.get('max_offset',       10))
        tw_size    = float(params.get('temporal_window_size', 100))

        eps    = 1e-10
        w_sum  = rm1 + rm2 + rm3 + eps
        wf     = rm1 / w_sum          # forward weight
        wr     = rm2 / w_sum          # reverse weight
        w3     = rm3 / w_sum          # intersection weight
        tw     = tw_size / 200.0      # temporal smoothing weight
        wi     = max_offset / 15.0    # squared interaction weight

        fwd_rev = sampled_fwd * sampled_rev

        scores = (
            wf * sampled_fwd
            + wr * sampled_rev
            + w3 * fwd_rev
            + tw * (sampled_fwd + sampled_rev) / 2.0
            + wi * fwd_rev ** 2
        )

        logger.info(
            f'Parametric scoring: wf={wf:.3f}  wr={wr:.3f}  w3={w3:.3f}  '
            f'tw={tw:.3f}  wi={wi:.3f}  '
            f'score_mean={scores.mean():.4f}  score_std={scores.std():.4f}'
        )

        # WSI objective -- bounded [-1, 1]
        quality = fwd_rev
        std_s   = float(scores.std())
        std_q   = float(quality.std())

        if std_s < 1e-12:
            wsi = -1.0
            logger.warning(
                f'Degenerate scores (std={std_s:.2e}) -> WSI=-1.0.  '
                f'rm1={rm1:.0f}  rm2={rm2:.0f}  rm3={rm3:.0f}  '
                f'offset={max_offset:.0f}  tw={tw_size:.0f}'
            )
        else:
            centered_s = scores  - scores.mean()
            centered_q = quality - quality.mean()
            covariance = float(np.mean(centered_s * centered_q))
            wsi        = covariance / ((std_s + eps) * (std_q + eps))
            wsi        = float(np.clip(wsi, -1.0, 1.0))
            logger.info(
                f'WSI = {wsi:.6f}  '
                f'(cov={covariance:.6f}  std_s={std_s:.4f}  std_q={std_q:.4f}  '
                f'quality_mean={quality.mean():.4f})'
            )

        return wsi, scores.tolist()

    except Exception as e:
        logger.error(f'Trial execution failed: {e}', exc_info=True)
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
    
    survivors_file       = sys.argv[1]
    # v4.0: args 2+3 accepted for WATCHER/shell compat but ignored in load_data
    train_history_file   = sys.argv[2] if len(sys.argv) > 2 else None
    holdout_history_file = sys.argv[3] if len(sys.argv) > 3 else None
    trial_id             = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    
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
        # v4.0: draw history files passed for compat but unused
        seeds, fwd_matches, rev_matches, prng_type, mod = load_data(
            survivors_file, train_history_file, holdout_history_file
        )

        trial = None  # No Optuna in PULL mode

        accuracy, scores = run_trial(
            seeds, fwd_matches, rev_matches, params,
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
