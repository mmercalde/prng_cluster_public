#!/usr/bin/env python3
"""
scorer_trial_worker.py (v4.2 - Subset-Selection, bidirectional_count signal, TB S107)
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
npz_forward_matches     = None   # float32 ndarray -- quality signal from NPZ
npz_reverse_matches     = None   # float32 ndarray -- quality signal from NPZ
npz_bidirectional_count = None   # float32 ndarray -- survival frequency (v4.2)
npz_intersection_ratio  = None   # float32 ndarray -- bidirectional tightness (v4.2)
npz_trial_number        = None   # int32   ndarray -- trial_number per seed (v4.1)



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

            # v4.2: load bidirectional_count + intersection_ratio (TB ruling S107 Q1-Q3)
            # bidirectional_selectivity dropped -- 98.8% at floor, unusable as signal
            global npz_bidirectional_count, npz_intersection_ratio, npz_trial_number
            if 'bidirectional_count' in survivors:
                npz_bidirectional_count = survivors['bidirectional_count'].astype(_np.float32)
                logger.info(
                    f'NPZ bidirectional_count: min={npz_bidirectional_count.min():.0f}  '
                    f'median={float(_np.median(npz_bidirectional_count)):.0f}  '
                    f'max={npz_bidirectional_count.max():.0f}  '
                    f'std={npz_bidirectional_count.std():.1f}'
                )
            else:
                logger.warning('NPZ missing bidirectional_count -- using ones fallback')
                npz_bidirectional_count = _np.ones(len(seeds_to_score), dtype=_np.float32)

            if 'intersection_ratio' in survivors:
                npz_intersection_ratio = survivors['intersection_ratio'].astype(_np.float32)
                logger.info(
                    f'NPZ intersection_ratio: min={npz_intersection_ratio.min():.4f}  '
                    f'median={float(_np.median(npz_intersection_ratio)):.4f}  '
                    f'max={npz_intersection_ratio.max():.4f}'
                )
            else:
                logger.warning('NPZ missing intersection_ratio -- ir_score will be 0.0')
                npz_intersection_ratio = _np.zeros(len(seeds_to_score), dtype=_np.float32)

            if 'trial_number' in survivors:
                npz_trial_number = survivors['trial_number'].astype(_np.int32)
                logger.info(
                    f'NPZ trial_number: min={npz_trial_number.min()}  '
                    f'max={npz_trial_number.max()}  '
                    f'unique={len(_np.unique(npz_trial_number))}'
                )
            else:
                logger.warning('NPZ missing trial_number -- coverage bonus disabled')
                npz_trial_number = _np.zeros(len(seeds_to_score), dtype=_np.int32)

        except Exception as e:
            logger.error(f'Failed to load data: {e}', exc_info=True)
            raise

    if npz_forward_matches is None or npz_reverse_matches is None:
        raise RuntimeError('NPZ quality signals are None after load -- cannot compute WSI.')
    if npz_bidirectional_count is None or npz_trial_number is None:
        raise RuntimeError(
            'NPZ bidirectional_count/trial_number are None after load -- v4.2 cannot run.'
        )

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

    return seeds_to_score, npz_forward_matches, npz_reverse_matches, npz_bidirectional_count, npz_intersection_ratio, npz_trial_number, prng_type, mod



def run_trial(seeds_to_score,
              npz_forward_matches,
              npz_reverse_matches,
              npz_bidirectional_count,
              npz_intersection_ratio,
              npz_trial_number,
              params,
              prng_type='java_lcg',
              mod=1000,
              trial=None,
              use_legacy_scoring=False):
    """
    v4.2: Subset-Selection Objective -- bidirectional_count primary signal.
    Last modified : 2026-02-22
    Session       : S107
    Expected lines: ~155

    CHANGE FROM v4.1:
        bidirectional_selectivity dropped (98.8% at floor -- unusable).
        Primary: bidirectional_count (survival frequency, std=722).
        Secondary bonus: intersection_ratio (bidirectional tightness, weight=0.10).
        Median used (robust against heavy-tail counts -- TB Q2).
        Percentile-rank vs full global arrays (stable across trials).
        ir_disabled guard: if IR array all zeros, ir_score=0.0 with warning.

    TB FORMULA (final v4.2):
        mask     = vote_count >= 2  (k-of-3 residue filter)
        bc_stat  = median(bidirectional_count[mask])
        bc_score = P(bc_global < bc_stat)              in [0,1]
        ir_stat  = median(intersection_ratio[mask])
        ir_score = P(ir_global < ir_stat)              in [0,1]
        bal      = 1 - abs(mean(fwd[mask]) - mean(rev[mask]))
        coverage = unique(trial_number[mask]) / unique(trial_number[sample])
        tw_weight= clip(temporal_window_size/1000, 0.05, 0.20)
        size_pen = min(|log(keep/0.10)|, 5.0)
        objective= clip(bc_score*(0.75+0.25*bal) + tw_weight*coverage
                        + 0.10*ir_score - 0.30*size_pen, -1, 1)

    PRESERVED:
        S101: random.seed(optuna_trial_number) per-trial unique sampling
        All degenerate guards (too_small, keep_too_low, keep_too_high)
        full-length scores array (Option B)
        WATCHER CLI compatibility
    """
    import numpy as np
    import random
    import math

    EPS            = 1e-9
    TARGET_KEEP    = 0.10
    MIN_KEEP_FRAC  = 0.01
    MAX_KEEP_FRAC  = 0.40
    MIN_KEEP_COUNT = 10
    LAMBDA_SIZE    = 0.30
    SIZE_PEN_CAP   = 5.0
    IR_WEIGHT      = 0.10

    rm1         = int(params.get('residue_mod_1',   10))
    rm2         = int(params.get('residue_mod_2',  100))
    rm3         = int(params.get('residue_mod_3', 1000))
    max_offset  = int(params.get('max_offset',       5))
    tw_size     = int(params.get('temporal_window_size', 100))
    trial_num   = int(params.get('optuna_trial_number',   0))
    sample_size = int(params.get('sample_size', 50000))

    n_seeds = len(seeds_to_score)

    # S101: per-trial unique sampling
    if n_seeds > sample_size:
        random.seed(trial_num)
        sample_idx = np.array(
            random.sample(range(n_seeds), sample_size), dtype=np.int64
        )
        seeds_arr = np.array(seeds_to_score, dtype=np.int64)[sample_idx]
        bc_arr    = npz_bidirectional_count[sample_idx]
        ir_arr    = npz_intersection_ratio[sample_idx]
        fwd_arr   = npz_forward_matches[sample_idx]
        rev_arr   = npz_reverse_matches[sample_idx]
        tn_arr    = npz_trial_number[sample_idx]
        logger.info(f'Sampled {sample_size:,} / {n_seeds:,} seeds (rng_seed={trial_num})')
    else:
        sample_idx = None
        seeds_arr  = np.array(seeds_to_score, dtype=np.int64)
        bc_arr     = npz_bidirectional_count
        ir_arr     = npz_intersection_ratio
        fwd_arr    = npz_forward_matches
        rev_arr    = npz_reverse_matches
        tn_arr     = npz_trial_number

    N = len(seeds_arr)

    # Bound offset vs modulus so filter always has teeth (TB Tweak 6)
    off1 = max(1, min(max_offset, max(rm1 - 1, 1)))
    off2 = max(1, min(max_offset, max(rm2 - 1, 1)))
    off3 = max(1, min(max_offset, max(rm3 - 1, 1)))

    # k-of-3 mask: seed passes if >= 2 of 3 residue conditions met
    m1 = (seeds_arr % max(rm1, 1)) < off1
    m2 = (seeds_arr % max(rm2, 1)) < off2
    m3 = (seeds_arr % max(rm3, 1)) < off3
    vote_count = m1.astype(np.int32) + m2.astype(np.int32) + m3.astype(np.int32)
    mask = vote_count >= 2

    subset_n = int(mask.sum())
    keep     = subset_n / max(N, 1)

    logger.info(
        f'Mask: rm=({rm1},{rm2},{rm3}) off=({off1},{off2},{off3}) '
        f'subset_n={subset_n} keep={keep:.4f} ({keep*100:.1f}%)'
    )

    # Degenerate guards
    def _reject(reason):
        logger.warning(f'Rejected: {reason} subset_n={subset_n} keep={keep:.4f} -> -1.0')
        _log_trial_metrics(trial_num, params, subset_n, keep,
                           objective=-1.0, reason=reason)
        return -1.0, np.zeros(n_seeds, dtype=np.float32).tolist()

    if subset_n < MIN_KEEP_COUNT:
        return _reject('too_small')
    if keep < MIN_KEEP_FRAC:
        return _reject('keep_too_low')
    if keep > MAX_KEEP_FRAC:
        return _reject('keep_too_high')

    # Primary: bidirectional_count -- median (robust vs heavy tail, TB Q2)
    bc_subset = bc_arr[mask]
    bc_stat   = float(np.median(bc_subset))
    bc_score  = float(np.mean(npz_bidirectional_count < bc_stat))  # global percentile

    # Secondary: intersection_ratio (TB Q3, optional bonus weight=0.10)
    ir_disabled = bool(np.all(npz_intersection_ratio == 0))
    if ir_disabled:
        logger.warning('intersection_ratio all zeros -- ir_score=0.0 for this trial')
        ir_stat  = 0.0
        ir_score = 0.0
    else:
        ir_subset = ir_arr[mask]
        ir_stat   = float(np.median(ir_subset))
        ir_score  = float(np.mean(npz_intersection_ratio < ir_stat))  # global percentile

    # Balance bonus
    fwd_mean = float(fwd_arr[mask].mean())
    rev_mean = float(rev_arr[mask].mean())
    bal      = float(np.clip(1.0 - abs(fwd_mean - rev_mean), 0.0, 1.0))

    # Temporal coverage via trial_number
    uniq_total = max(len(np.unique(tn_arr)), 1)
    uniq_sel   = len(np.unique(tn_arr[mask]))
    coverage   = uniq_sel / uniq_total
    tw_weight  = float(np.clip(tw_size / 1000.0, 0.05, 0.20))

    # Size penalty, capped
    size_penalty = min(
        abs(math.log((keep + EPS) / TARGET_KEEP)),
        SIZE_PEN_CAP
    )

    # TB v4.2 composite objective
    objective = (
        bc_score * (0.75 + 0.25 * bal)
        + tw_weight * coverage
        + IR_WEIGHT * ir_score
        - LAMBDA_SIZE * size_penalty
    )
    objective = float(np.clip(objective, -1.0, 1.0))

    logger.info(
        f'Objective={objective:.6f}  bc_stat={bc_stat:.0f}  bc_score={bc_score:.4f}  '
        f'ir_stat={ir_stat:.4f}  ir_score={ir_score:.4f}  '
        f'bal={bal:.4f}  coverage={coverage:.4f}  tw_weight={tw_weight:.3f}  '
        f'size_pen={size_penalty:.4f}'
    )

    _log_trial_metrics(
        trial_num, params, subset_n, keep,
        bc_stat=bc_stat, bc_score=bc_score,
        ir_stat=ir_stat, ir_score=ir_score,
        fwd_mean=fwd_mean, rev_mean=rev_mean, bal=bal,
        coverage=coverage, tw_weight=tw_weight,
        size_penalty=size_penalty, objective=objective, reason='ok'
    )

    # Option B: full-length scores array
    full = np.zeros(n_seeds, dtype=np.float32)
    if sample_idx is not None:
        full[sample_idx] = mask.astype(np.float32)
    else:
        full[:] = mask.astype(np.float32)

    return objective, full.tolist()


def _log_trial_metrics(trial_num, params, subset_n, keep,
                       bc_stat=None, bc_score=None,
                       ir_stat=None, ir_score=None,
                       fwd_mean=None, rev_mean=None, bal=None,
                       coverage=None, tw_weight=None,
                       size_penalty=None, objective=None, reason='ok'):
    """
    Per-trial diagnostic metrics (TB S107 requirement).
    v4.2: sel_* replaced with bc_* and ir_*.
    Key signal: bc_score must vary across trials for real landscape.
    Session: S107  Expected lines: ~25
    """
    metrics = {
        'trial_num'   : trial_num,
        'params'      : params,
        'subset_n'    : subset_n,
        'keep'        : round(keep, 6) if keep is not None else None,
        'bc_stat'     : round(bc_stat, 2) if bc_stat is not None else None,
        'bc_score'    : round(bc_score, 6) if bc_score is not None else None,
        'ir_stat'     : round(ir_stat, 6) if ir_stat is not None else None,
        'ir_score'    : round(ir_score, 6) if ir_score is not None else None,
        'fwd_mean'    : round(fwd_mean, 6) if fwd_mean is not None else None,
        'rev_mean'    : round(rev_mean, 6) if rev_mean is not None else None,
        'bal'         : round(bal, 6) if bal is not None else None,
        'coverage'    : round(coverage, 6) if coverage is not None else None,
        'tw_weight'   : round(tw_weight, 6) if tw_weight is not None else None,
        'size_penalty': round(size_penalty, 6) if size_penalty is not None else None,
        'objective'   : round(objective, 6) if objective is not None else None,
        'reason'      : reason,
    }
    logger.info(f'[TRIAL_METRICS] {metrics}')


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
        seeds, fwd_matches, rev_matches, bc, ir, tn, prng_type, mod = load_data(
            survivors_file, train_history_file, holdout_history_file
        )

        trial = None  # No Optuna in PULL mode

        accuracy, scores = run_trial(
            seeds, fwd_matches, rev_matches, bc, ir, tn, params,
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
