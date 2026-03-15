#!/usr/bin/env python3
"""
Real Bayesian Optimization for Window Parameter Search
Uses Optuna (Tree-Parzen Estimator) and scikit-learn (Gaussian Processes)
"""

import json
import numpy as np
from typing import Callable, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING
import random

# Type checking import (no runtime circular dependency)
if TYPE_CHECKING:
    from window_optimizer import SearchBounds

# Runtime import function for when we actually need the classes
def _get_search_bounds():
    from window_optimizer import SearchBounds, load_search_bounds_from_config
    return SearchBounds, load_search_bounds_from_config


# [S121] TRSE context loader — passive, never raises
def _load_trse_context(context_file: str = 'trse_context.json') -> Optional[dict]:
    """
    Load trse_context.json if present. Returns None silently if absent or invalid.
    Step 1 runs normally with full default bounds when context is absent.
    """
    if not context_file:
        return None
    try:
        import os
        if not os.path.exists(context_file):
            return None
        with open(context_file) as f:
            ctx = json.load(f)
        # Version guard — require v1.15+ for regime_type field
        version = ctx.get('trse_version', '0.0.0')
        major, minor = int(version.split('.')[0]), int(version.split('.')[1])
        if (major, minor) < (1, 15):
            print(f"[TRSE] Context version {version} < 1.15 — skipping bounds narrowing")
            return None
        return ctx
    except Exception as e:
        print(f"[TRSE] Could not load context: {e} — running with default bounds")
        return None

# Try importing Optuna (preferred)
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not available - install with: pip install optuna")

# Try importing sklearn for Gaussian Processes (fallback)
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF
    SKLEARN_GP_AVAILABLE = True
except ImportError:
    SKLEARN_GP_AVAILABLE = False
    print("⚠️  scikit-learn GP not available")


# === INCREMENTAL OUTPUT IMPORTS (Patch 2026-01-18) ===
from datetime import datetime as _patch_datetime
from pathlib import Path as _patch_Path
import json as _patch_json
# === END INCREMENTAL IMPORTS ===


# ============================================================================
# DATA STRUCTURES (copied from window_optimizer.py)
# ============================================================================

@dataclass
class WindowConfig:
    """Complete window and skip configuration"""
    window_size: int
    offset: int
    sessions: List[str]
    skip_min: int
    skip_max: int
    forward_threshold: float = 0.72
    reverse_threshold: float = 0.81
    
    def __hash__(self):
        return hash((self.window_size, self.offset, tuple(self.sessions), 
                    self.skip_min, self.skip_max, self.forward_threshold, self.reverse_threshold))
    
    def description(self) -> str:
        sess = '+'.join(self.sessions)
        return f"W{self.window_size}_O{self.offset}_{sess}_S{self.skip_min}-{self.skip_max}_FT{self.forward_threshold}_RT{self.reverse_threshold}"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_vector(self) -> np.ndarray:
        """Convert to numerical vector for optimization"""
        session_encoding = {
            ('midday', 'evening'): 0,
            ('midday',): 1,
            ('evening',): 2
        }
        sess_tuple = tuple(sorted(self.sessions))
        return np.array([
            self.window_size,
            self.offset,
            session_encoding.get(sess_tuple, 0),
            self.skip_min,
            self.skip_max
        ], dtype=float)



# SearchBounds imported from window_optimizer.py (single source of truth)

@dataclass
class OptimizationResult:
    """Result from testing a window configuration"""
    config: WindowConfig
    forward_count: int
    reverse_count: int
    bidirectional_count: int
    iteration: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'config': self.config.to_dict(),
            'forward_count': self.forward_count,
            'reverse_count': self.reverse_count,
            'bidirectional_count': self.bidirectional_count,
            'iteration': self.iteration
        }


class ResultScorer:
    """Score optimization results (higher is better)"""
    
    def __init__(self, strategy='bidirectional'):
        self.strategy = strategy
    
    def score(self, result: OptimizationResult) -> float:
        """Score a result (higher is better)"""
        if self.strategy == 'bidirectional':
            # Primary: bidirectional survivors
            return result.bidirectional_count if result.bidirectional_count > 0 else -1000
        elif self.strategy == 'balanced':
            # Balance between forward, reverse, and bidirectional
            return (result.forward_count * 0.3 + 
                   result.reverse_count * 0.3 + 
                   result.bidirectional_count * 0.4) if result.bidirectional_count > 0 else -1000
        elif self.strategy == 'conservative':
            # Prefer configurations with consistent forward/reverse
            if result.forward_count == 0 or result.reverse_count == 0:
                return -1000
            ratio = min(result.forward_count, result.reverse_count) / max(result.forward_count, result.reverse_count)
            return result.bidirectional_count * ratio if result.bidirectional_count > 0 else -1000
        else:
            return result.bidirectional_count if result.bidirectional_count > 0 else -1000


# ============================================================================
# INCREMENTAL OUTPUT CALLBACK (Patch 2026-01-18)
# ============================================================================

def create_incremental_save_callback(
    output_config_path: str = "optimal_window_config.json",
    output_survivors_path: str = "bidirectional_survivors.json",
    total_trials: int = 50,
    trial_history_context: dict = None,  # [S140b]
):
    """
    Optuna callback that saves best-so-far results after each trial.
    Ensures crash recovery and WATCHER visibility.
    [S140b] Writes per-trial data to step1_trial_history when context provided.
    """
    def save_best_so_far(study, trial):
        completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
        
        progress = {
            "status": "in_progress",
            "completed_trials": completed,
            "total_trials": total_trials,
            "last_updated": _patch_datetime.now().isoformat(),
            "last_trial_number": trial.number,
            "last_trial_value": trial.value,
        }
        
        if study.best_trial is not None:
            best_params = study.best_params or {}
            best_config = {
                **progress,
                "best_trial_number": study.best_trial.number,
                "best_value": study.best_value,
                "best_bidirectional_count": int(study.best_value) if study.best_value and study.best_value > 0 else 0,
                "best_params": best_params,
                "window_size": best_params.get("window_size"),
                "offset": best_params.get("offset"),
                "skip_min": best_params.get("skip_min"),
                "skip_max": best_params.get("skip_max"),
                "forward_threshold": best_params.get("forward_threshold"),
                "reverse_threshold": best_params.get("reverse_threshold"),
            }
            
            # Atomic write
            temp_path = _patch_Path(output_config_path).with_suffix(".tmp")
            with open(temp_path, "w") as f:
                _patch_json.dump(best_config, f, indent=2)
            temp_path.rename(output_config_path)
            
            print(f"[SAVE] Trial {trial.number}: config saved (best={study.best_value:.0f} @ trial {study.best_trial.number})")
            
            # Save survivors if this trial stored them and is the new best
            if trial.number == study.best_trial.number:
                survivors = trial.user_attrs.get("bidirectional_survivors")
                if survivors and len(survivors) > 0:
                    temp_surv = _patch_Path(output_survivors_path).with_suffix(".tmp")
                    with open(temp_surv, "w") as f:
                        _patch_json.dump(survivors, f)
                    temp_surv.rename(output_survivors_path)
                    print(f"[SAVE] Trial {trial.number}: {len(survivors)} survivors saved")
        else:
            progress["note"] = "No successful trials yet"
            temp_path = _patch_Path(output_config_path).with_suffix(".tmp")
            with open(temp_path, "w") as f:
                _patch_json.dump(progress, f, indent=2)
            temp_path.rename(output_config_path)

        # [S140b] per-trial history write
        # [S142] Skip when n_parallel>1 — _worker_obj owns writes with correct session strings.
        # If we write here too, INSERT OR IGNORE silently blocks the NP2 write (same PK, NULL session).
        if trial_history_context and not trial_history_context.get('n_parallel_gt1'):
            try:
                from database_system import DistributedPRNGDatabase as _DBTH
                _db_th = _DBTH()
                _params = trial.params if trial.params else {}
                _score  = trial.value if trial.value is not None else 0.0
                _pruned = trial.state.name == 'PRUNED'
                _surv   = trial.user_attrs.get('bidirectional_survivors', [])
                _bidi   = len(_surv) if isinstance(_surv, list) else 0
                _db_th.write_step1_trial(
                    run_id=trial_history_context.get('run_id','unknown'),
                    study_name=trial_history_context.get('study_name','unknown'),
                    trial_number=trial.number,
                    prng_type=trial_history_context.get('prng_type','java_lcg'),
                    seed_range_start=trial_history_context.get('seed_start',0),
                    seed_range_end=trial_history_context.get('seed_end',0),
                    params=_params,
                    trial_score=_score,
                    forward_survivors=trial.user_attrs.get('forward_count',0),
                    reverse_survivors=trial.user_attrs.get('reverse_count',0),
                    bidirectional_survivors=_bidi,
                    pruned=_pruned)
            except Exception as _e_th:
                print(f"   [TRIAL_HISTORY] write failed (non-fatal): {_e_th}")
    
    return save_best_so_far


def finalize_incremental_output(study, output_config_path: str = "optimal_window_config.json"):
    """Mark output as complete after study.optimize() finishes."""
    config_path = _patch_Path(output_config_path)
    if not config_path.exists():
        return
    
    with open(config_path, "r") as f:
        config = _patch_json.load(f)
    
    config["status"] = "complete"
    config["completed_at"] = _patch_datetime.now().isoformat()
    
    with open(config_path, "w") as f:
        _patch_json.dump(config, f, indent=2)
    
    print(f"[SAVE] Finalized {output_config_path} (status=complete)")


# ============================================================================
# OPTUNA-BASED BAYESIAN OPTIMIZATION (PREFERRED)
# ============================================================================

class OptunaBayesianSearch:
    """Bayesian optimization using Optuna's TPE sampler"""
    
    def __init__(self, n_startup_trials=5, seed=None,
                 enable_pruning=False, n_parallel=1):  # S115 R3
        """
        Args:
            n_startup_trials: Number of random trials before using TPE
            seed: Random seed for reproducibility
            enable_pruning: If True enable forward_count==0 pruning (S115 M2)
            n_parallel: Number of parallel partitions (S115 M1)
        """
        self.enable_pruning = enable_pruning
        self.n_parallel = n_parallel
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        self.n_startup_trials = n_startup_trials
        self.seed = seed
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def search(self, 
               objective_function: Callable,
               bounds: 'SearchBounds',
               max_iterations: int,
               scorer: ResultScorer,
               resume_study: bool = False,
               study_name: str = '',
               trse_context_file: str = 'trse_context.json',
               trial_history_context: dict = None) -> Dict:  # S121, [S140b]
        """
        Run Bayesian optimization using Optuna
        
        Args:
            objective_function: Function that takes WindowConfig and returns OptimizationResult
            bounds: Search space boundaries
            max_iterations: Number of iterations
            scorer: Function to score results
            
        Returns:
            Dictionary with best config, all results, etc.
        """
        print(f"\n{'='*80}")
        print(f"🎯 BAYESIAN OPTIMIZATION (Optuna TPE)")
        print(f"   Startup trials: {self.n_startup_trials}")
        print(f"   Total trials: {max_iterations}")
        print(f"{'='*80}\n")
        
        # Storage for all results
        all_results = []
        best_result = None
        best_score = float('-inf')
        
        def optuna_objective(trial):
            """Optuna objective function"""
            # Sample parameters from search space
            window_size = trial.suggest_int('window_size', 
                                           bounds.min_window_size, 
                                           bounds.max_window_size)
            offset = trial.suggest_int('offset', 
                                      bounds.min_offset, 
                                      bounds.max_offset)
            session_idx = trial.suggest_int('session_idx', 
                                           0, 
                                           len(bounds.session_options) - 1)
            skip_min = trial.suggest_int('skip_min', 
                                        bounds.min_skip_min, 
                                        bounds.max_skip_min)
            skip_max = trial.suggest_int('skip_max', 
                                        max(skip_min, bounds.min_skip_max),
                                        bounds.max_skip_max)

            # Suggest thresholds (Optuna optimizes these!)
            forward_threshold = trial.suggest_float('forward_threshold',
                                                   bounds.min_forward_threshold,
                                                   bounds.max_forward_threshold)
            reverse_threshold = trial.suggest_float('reverse_threshold',
                                                   bounds.min_reverse_threshold,
                                                   bounds.max_reverse_threshold)
            
            # Create configuration
            config = WindowConfig(
                window_size=window_size,
                offset=offset,
                sessions=bounds.session_options[session_idx],
                skip_min=skip_min,
                skip_max=skip_max,
                forward_threshold=round(forward_threshold, 2),
                reverse_threshold=round(reverse_threshold, 2)
            )
            
            # Evaluate configuration — S115 M2: pass trial for pruning hook
            result = objective_function(config, optuna_trial=trial)
            result.iteration = trial.number
            
            # Store result
            all_results.append(result)
            score = scorer.score(result)
            
            # Store result data for incremental callback
            trial.set_user_attr("bidirectional_survivors", 
                               getattr(result, 'bidirectional_survivors', []))
            trial.set_user_attr("result_dict", result.to_dict())
            
            # Track best
            nonlocal best_result, best_score
            if score > best_score:
                best_score = score
                best_result = result
                print(f"✨ NEW BEST [Trial {trial.number + 1}]: {config.description()}")
                print(f"   Score: {score:.2f} (Bidirectional: {result.bidirectional_count})")
                print(f"   Forward: {result.forward_count}, Reverse: {result.reverse_count}\n")
            else:
                print(f"   Trial {trial.number + 1}: {config.description()} → Score: {score:.2f}")
            
            return score
        
        # [S121] TRSE Step 0 — Load regime context and narrow search bounds if available
        # Passive: if context absent or confidence low, bounds unchanged.
        # Rule A only: window ceiling narrowed when regime_type=short_persistence
        #              AND regime_type_confidence >= 0.70 (TB guardrail).
        # Rules B (skip) and C (offset) are logged only — not applied.
        # Shuffle test (S121) confirmed density_proxy measures digit bias, not
        # temporal correlation, so skip/offset advisory fields are unreliable
        # as hard bounds constraints.
        trse_ctx = _load_trse_context(trse_context_file)
        if trse_ctx:
            _regime_type  = trse_ctx.get('regime_type', 'unknown')
            _type_conf    = trse_ctx.get('regime_type_confidence', 0.0)
            _regime_stable = trse_ctx.get('regime_stable', False)
            _rec_ws       = trse_ctx.get('recommended_window_size', None)
            _w3_w8_ratio  = trse_ctx.get('w3_w8_ratio', None)

            print(f"\n[TRSE] Context loaded — regime={trse_ctx.get('current_regime')} "
                  f"stable={_regime_stable} type={_regime_type} "
                  f"type_conf={_type_conf:.3f} w3_w8_ratio={_w3_w8_ratio}")

            # Rule A — window ceiling from regime_type (TB approved, S121)
            if (_regime_type == 'short_persistence'
                    and _type_conf >= 0.70
                    and _regime_stable):
                old_max = bounds.max_window_size
                new_max = max(bounds.min_window_size + 1,
                              min(32, bounds.max_window_size))
                bounds.max_window_size = new_max  # SearchBounds is a dataclass, not namedtuple
                print(f"[TRSE] Rule A ACTIVE: short_persistence "
                      f"(conf={_type_conf:.3f}) → "
                      f"window_size ceiling {old_max} → {new_max}")
            else:
                print(f"[TRSE] Rule A SKIPPED: type={_regime_type} "
                      f"conf={_type_conf:.3f} stable={_regime_stable} "
                      f"(requires short_persistence + conf>=0.70 + stable)")

            # Rule B — skip bounds advisory (logged only, disabled per TB S121)
            _skip_prof = trse_ctx.get('skip_entropy_profile', {})
            if _skip_prof.get('consistent_with_known_skip'):
                print(f"[TRSE] Rule B (skip): consistent with [5,56] — "
                      f"LOGGED ONLY (disabled: shuffle test refuted temporal correlation)")

            # Rule C — offset advisory (logged only, disabled per TB S121)
            _off = trse_ctx.get('dominant_offset_lag', {})
            if _off.get('confident'):
                print(f"[TRSE] Rule C (offset): dominant_lag={_off.get('dominant_lag')} — "
                      f"LOGGED ONLY (disabled: FFT on digit-bias signal is fragile)")
        else:
            print("[TRSE] No context found — running Step 1 with default bounds")

        # Create study with TPE sampler
        # S119: multivariate=True — models param correlations jointly (window_size↔skip_max etc.)
        # Safe: search space is static (skip_max lower bound always=10), Optuna 4.4.0 tested.
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.filterwarnings('ignore', message='.*multivariate.*')
            sampler = TPESampler(
                n_startup_trials=self.n_startup_trials,
                seed=self.seed,
                multivariate=True   # S119
            )
        
        # Create persistent storage for the study
        import time
        import glob as _glob
        import os as _os

        # --- Resume logic (S114/S116 patch, upgraded S115 R4) ---
        # resume_study=True: find most recent incomplete DB and continue
        # resume_study=False (default): always create fresh study
        # S115 R4: n_parallel>1 uses JournalFileBackend (no SQLite write-lock contention)
        _resume = False
        _resumed_completed = 0
        _fresh_study_name = f"window_opt_{int(time.time())}"
        _fresh_storage_path = f"sqlite:////home/michael/distributed_prng_analysis/optuna_studies/{_fresh_study_name}.db"

        if resume_study:
            # specific study_name takes priority over auto-select
            if study_name:
                _candidate_dbs = [f"optuna_studies/{study_name}.db"]
                print(f"   🔄 Requested study: {study_name}")
            else:
                _candidate_dbs = sorted(
                    _glob.glob("optuna_studies/window_opt_*.db"),
                    key=_os.path.getmtime,
                    reverse=True
                )
            _resume_found = False
            for _candidate_db in _candidate_dbs:
                if not _os.path.exists(_candidate_db):
                    print(f"   ⚠️  Study DB not found: {_candidate_db}")
                    continue
                _candidate_name = _os.path.splitext(_os.path.basename(_candidate_db))[0]
                _candidate_storage = f"sqlite:////home/michael/distributed_prng_analysis/{_candidate_db}"
                try:
                    _candidate_study = optuna.load_study(
                        study_name=_candidate_name,
                        storage=_candidate_storage
                    )
                    _candidate_completed = len([
                        t for t in _candidate_study.trials
                        if t.state.name == "COMPLETE"
                    ])
                    if _candidate_completed > 0 and (_candidate_completed < max_iterations or study_name):
                        _resume = True
                        _resumed_completed = _candidate_completed
                        study_name = _candidate_name
                        storage_path = _candidate_storage
                        print(f"   🔄 RESUMING study: {_candidate_name}")
                        print(f"   🔄 Completed: {_candidate_completed}/{max_iterations} trials")
                        print(f"   🔄 Remaining: {max_iterations - _candidate_completed} trials")
                        _resume_found = True
                        break
                    else:
                        print(f"   📊 Study {_candidate_name}: {_candidate_completed} trials — skipping")
                except Exception as _e:
                    print(f"   ⚠️  Could not load {_candidate_name} ({_e})")
            if not _resume_found:
                print(f"   📊 No resumable study found — starting fresh")

        if not _resume:
            # S125: always SQLite (JournalFileBackend removed -- n_parallel parallelism
            # now owned by multiprocessing dispatcher in integration layer; n_jobs=1 here)
            study_name = _fresh_study_name
            storage_path = _fresh_storage_path

        # S115 M2: ThresholdPruner as secondary safety net
        _pruner = optuna.pruners.ThresholdPruner(lower=1.0) if self.enable_pruning else optuna.pruners.NopPruner()

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction='maximize',
            sampler=sampler,
            pruner=_pruner,
            load_if_exists=_resume
        )
        print(f"   📊 Optuna study: optuna_studies/{study_name}.db")
        self._trial_history_context = trial_history_context  # [S140b]

        # [S144] Warm-start: enqueue from trial_history_context ONLY.
        # No hardcoded fallback -- CA-specific W8_O43 removed.
        # New/different datasets get no warm-start and Optuna explores freely.
        if not _resume:
            if trial_history_context:
                _ww=trial_history_context.get('warm_start_window')
                _wo=trial_history_context.get('warm_start_offset')
                _wsk=trial_history_context.get('warm_start_skip_min')
                _wsx=trial_history_context.get('warm_start_skip_max')
                _wf=trial_history_context.get('warm_start_fwd_thresh')
                _wr=trial_history_context.get('warm_start_rev_thresh')
                if all(v is not None for v in [_ww,_wo,_wsk,_wsx,_wf,_wr]):
                    _ws_params={'window_size':int(_ww),'offset':int(_wo),
                               'skip_min':int(_wsk),'skip_max':int(_wsx),
                               'forward_threshold':float(_wf),'reverse_threshold':float(_wr)}
                    _ws_source=f'step1_trial_history (W{_ww}_O{_wo})'
                    study.enqueue_trial(_ws_params)
                    print(f"   🌡️  Warm-start: enqueued {_ws_source} as trial 0")  # [S144]
                else:
                    print("   ℹ️  Warm-start skipped: trial_history_context incomplete -- Optuna explores freely")
            else:
                print("   ℹ️  Warm-start skipped: no trial_history_context -- Optuna explores freely")
        else:
            print("   ✅ Resume mode: skipping warm-start (already in DB)")

        # Trials remaining: full count on fresh, remainder on resume
        # S125: n_parallel>1 dispatched externally; this path always n_jobs=1
        _trials_to_run = max_iterations - _resumed_completed
        
        # Run optimization with incremental save callback
        # [S140b] trial_history_context flows from optimize_window
        _th_context = self._trial_history_context if hasattr(self, '_trial_history_context') else None
        _incremental_callback = create_incremental_save_callback(
            output_config_path="optimal_window_config.json",
            output_survivors_path="bidirectional_survivors.json",
            total_trials=max_iterations,
            trial_history_context=_th_context
        )
        # S115 R1: prune telemetry callback
        def _prune_telemetry(study, trial):
            _nt = len(study.trials)
            if _nt > 0 and _nt % 10 == 0:
                _np = sum(1 for t in study.trials if t.state.name=='PRUNED')
                _nc = sum(1 for t in study.trials if t.state.name=='COMPLETE')
                print(f"   📊 Prune telemetry ({_nt} trials): complete={_nc}  pruned={_np}  rate={_np/_nt*100:.0f}%")

        study.optimize(optuna_objective, n_trials=_trials_to_run,
                       callbacks=[_incremental_callback, _prune_telemetry],
                       n_jobs=1)  # S125: always 1 -- parallelism via multiprocessing.Process

        _nt = len(study.trials); _np = sum(1 for t in study.trials if t.state.name=='PRUNED')
        _nc = sum(1 for t in study.trials if t.state.name=='COMPLETE')
        if _nt > 0:
            print(f"\nPRUNING SUMMARY\n  Total: {_nt}  Pruned: {_np} ({_np/_nt*100:.1f}%)  Complete: {_nc}")
        
        # Finalize output (mark status=complete)
        finalize_incremental_output(study, "optimal_window_config.json")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"🏆 OPTIMIZATION COMPLETE")
        print(f"   Best score: {best_score:.2f}")
        print(f"   Best config: {best_result.config.description()}")
        print(f"   Bidirectional survivors: {best_result.bidirectional_count}")
        print(f"   📊 Optuna-optimized thresholds:")
        print(f"      Forward threshold: {best_result.config.forward_threshold}")
        print(f"      Reverse threshold: {best_result.config.reverse_threshold}")
        print(f"{'='*80}\n")
        
        return {
            'strategy': 'optuna_bayesian',
            'best_config': best_result.config.to_dict(),
            'best_result': best_result.to_dict(),
            'best_score': best_score,
            'all_results': [r.to_dict() for r in all_results],
            'iterations': len(all_results),
            'optuna_study': {
                'best_trial': study.best_trial.number,
                'best_value': study.best_value,
                'best_params': study.best_params
            }
        }


# ============================================================================
# SKLEARN GAUSSIAN PROCESS OPTIMIZATION (FALLBACK)
# ============================================================================

class GaussianProcessBayesianSearch:
    """Bayesian optimization using sklearn Gaussian Processes"""
    
    def __init__(self, n_initial_points=5, acquisition='ucb', seed=None):
        """
        Args:
            n_initial_points: Number of random points before using GP
            acquisition: 'ucb', 'ei', or 'pi'
            seed: Random seed
        """
        if not SKLEARN_GP_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        self.n_initial_points = n_initial_points
        self.acquisition = acquisition
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def _config_to_vector(self, config: WindowConfig, bounds: 'SearchBounds') -> np.ndarray:
        """Convert config to normalized vector [0, 1]"""
        session_idx = bounds.session_options.index(config.sessions)
        return np.array([
            (config.window_size - bounds.min_window_size) / (bounds.max_window_size - bounds.min_window_size),
            (config.offset - bounds.min_offset) / (bounds.max_offset - bounds.min_offset),
            session_idx / max(1, len(bounds.session_options) - 1),
            (config.skip_min - bounds.min_skip_min) / max(1, bounds.max_skip_min - bounds.min_skip_min),
            (config.skip_max - bounds.min_skip_max) / max(1, bounds.max_skip_max - bounds.min_skip_max)
        ])
    
    def _vector_to_config(self, vec: np.ndarray, bounds: 'SearchBounds') -> WindowConfig:
        """Convert normalized vector to config"""
        window_size = int(np.clip(
            vec[0] * (bounds.max_window_size - bounds.min_window_size) + bounds.min_window_size,
            bounds.min_window_size,
            bounds.max_window_size
        ))
        offset = int(np.clip(
            vec[1] * (bounds.max_offset - bounds.min_offset) + bounds.min_offset,
            bounds.min_offset,
            bounds.max_offset
        ))
        session_idx = int(np.clip(
            vec[2] * len(bounds.session_options),
            0,
            len(bounds.session_options) - 1
        ))
        skip_min = int(np.clip(
            vec[3] * (bounds.max_skip_min - bounds.min_skip_min) + bounds.min_skip_min,
            bounds.min_skip_min,
            bounds.max_skip_min
        ))
        skip_max = int(np.clip(
            vec[4] * (bounds.max_skip_max - bounds.min_skip_max) + bounds.min_skip_max,
            bounds.min_skip_max,
            bounds.max_skip_max
        ))
        
        # Ensure skip_max >= skip_min
        skip_max = max(skip_max, skip_min)
        
        return WindowConfig(
            window_size=window_size,
            offset=offset,
            sessions=bounds.session_options[session_idx],
            skip_min=skip_min,
            skip_max=skip_max,
            forward_threshold=bounds.default_forward_threshold,
            reverse_threshold=bounds.default_reverse_threshold
        )
    
    def _acquisition_function(self, gp, X_train, y_train, X_new, kappa=2.0):
        """Upper Confidence Bound acquisition function"""
        mu, sigma = gp.predict(X_new.reshape(1, -1), return_std=True)
        if self.acquisition == 'ucb':
            return mu + kappa * sigma
        elif self.acquisition == 'ei':
            # Expected Improvement
            best = np.max(y_train)
            z = (mu - best) / (sigma + 1e-9)
            from scipy.stats import norm
            return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)
        else:  # pi - probability of improvement
            from scipy.stats import norm
            best = np.max(y_train)
            z = (mu - best) / (sigma + 1e-9)
            return norm.cdf(z)
    
    def search(self,
               objective_function: Callable,
               bounds: 'SearchBounds',
               max_iterations: int,
               scorer: ResultScorer) -> Dict:
        """Run Gaussian Process Bayesian optimization"""
        
        print(f"\n{'='*80}")
        print(f"🎯 BAYESIAN OPTIMIZATION (Gaussian Process)")
        print(f"   Initial random points: {self.n_initial_points}")
        print(f"   Acquisition: {self.acquisition}")
        print(f"{'='*80}\n")
        
        all_results = []
        best_result = None
        best_score = float('-inf')
        
        X_train = []
        y_train = []
        
        # Phase 1: Random initialization
        for i in range(min(self.n_initial_points, max_iterations)):
            # Random config
            config = WindowConfig(
                window_size=random.randint(bounds.min_window_size, bounds.max_window_size),
                offset=random.randint(bounds.min_offset, bounds.max_offset),
                sessions=random.choice(bounds.session_options),
                skip_min=random.randint(bounds.min_skip_min, bounds.max_skip_min),
                skip_max=random.randint(bounds.min_skip_max, bounds.max_skip_max),
                forward_threshold=round(random.uniform(bounds.min_forward_threshold, bounds.max_forward_threshold), 2),
                reverse_threshold=round(random.uniform(bounds.min_reverse_threshold, bounds.max_reverse_threshold), 2)
            )
            config.skip_max = max(config.skip_max, config.skip_min)
            
            result = objective_function(config)
            result.iteration = i
            score = scorer.score(result)
            
            all_results.append(result)
            X_train.append(self._config_to_vector(config, bounds))
            y_train.append(score)
            
            if score > best_score:
                best_score = score
                best_result = result
                print(f"✨ NEW BEST [Init {i + 1}]: {config.description()}")
                print(f"   Score: {score:.2f} (Bidirectional: {result.bidirectional_count})\n")
            else:
                print(f"   Init {i + 1}: {config.description()} → Score: {score:.2f}")
        
        # Phase 2: GP-guided search
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        for i in range(self.n_initial_points, max_iterations):
            # Fit GP to current data
            gp.fit(X_train, y_train)
            
            # Find best point via acquisition function
            n_candidates = 1000
            X_candidates = np.random.rand(n_candidates, 5)
            
            acquisition_values = []
            for x in X_candidates:
                acq = self._acquisition_function(gp, X_train, y_train, x)
                acquisition_values.append(acq[0] if hasattr(acq, '__len__') else acq)
            
            best_idx = np.argmax(acquisition_values)
            x_next = X_candidates[best_idx]
            
            # Convert to config
            config = self._vector_to_config(x_next, bounds)
            
            # Evaluate
            result = objective_function(config)
            result.iteration = i
            score = scorer.score(result)
            
            all_results.append(result)
            X_train = np.vstack([X_train, x_next])
            y_train = np.append(y_train, score)
            
            if score > best_score:
                best_score = score
                best_result = result
                print(f"✨ NEW BEST [GP {i + 1}]: {config.description()}")
                print(f"   Score: {score:.2f} (Bidirectional: {result.bidirectional_count})\n")
            else:
                print(f"   GP {i + 1}: {config.description()} → Score: {score:.2f}")
        
        print(f"\n{'='*80}")
        print(f"🏆 OPTIMIZATION COMPLETE")
        print(f"   Best score: {best_score:.2f}")
        print(f"   Best config: {best_result.config.description()}")
        print(f"{'='*80}\n")
        
        return {
            'strategy': 'gaussian_process_bayesian',
            'best_config': best_result.config.to_dict(),
            'best_result': best_result.to_dict(),
            'best_score': best_score,
            'all_results': [r.to_dict() for r in all_results],
            'iterations': len(all_results)
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_bayesian_optimizer(method='auto', **kwargs):
    """
    Create appropriate Bayesian optimizer
    
    Args:
        method: 'auto', 'optuna', or 'sklearn'
        **kwargs: Additional arguments for the optimizer
        
    Returns:
        Bayesian optimizer instance
    """
    if method == 'auto':
        if OPTUNA_AVAILABLE:
            return OptunaBayesianSearch(**kwargs)
        elif SKLEARN_GP_AVAILABLE:
            return GaussianProcessBayesianSearch(**kwargs)
        else:
            raise ImportError("Neither Optuna nor scikit-learn available for Bayesian optimization")
    elif method == 'optuna':
        return OptunaBayesianSearch(**kwargs)
    elif method == 'sklearn':
        return GaussianProcessBayesianSearch(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    print("🎯 Bayesian Optimization Module")
    print(f"   Optuna available: {OPTUNA_AVAILABLE}")
    print(f"   sklearn GP available: {SKLEARN_GP_AVAILABLE}")
    
    if OPTUNA_AVAILABLE:
        print("\n✅ Recommended: Use Optuna-based Bayesian optimization")
    elif SKLEARN_GP_AVAILABLE:
        print("\n✅ Available: sklearn Gaussian Process optimization")
    else:
        print("\n❌ Install Optuna: pip install optuna")
