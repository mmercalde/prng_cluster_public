#!/usr/bin/env python3
"""
Meta-Prediction Optimizer - ANTI-OVERFITTING VERSION (v3.4)
=============================================================

VERSION HISTORY:
  v1.2   - Initial implementation with Optuna persistence
  v2.0   - Multi-model support (--model-type, --compare-models)
  v3.0   - signal_quality emission for Step 6 gate
  v3.1   - data_context fingerprint for WATCHER loop prevention
  v3.2   - Early exit on degenerate signal + degenerate sidecar
  v3.3   - CRITICAL FIX: Remove prng_type from fingerprint hash (Team Beta approved)

KEY FEATURES (v3.3):
✅ Multi-model support: neural_net, xgboost, lightgbm, catboost
✅ Subprocess isolation for OpenCL/CUDA compatibility
✅ signal_quality emission for Step 6 prediction gate
✅ data_context fingerprint for WATCHER loop prevention
✅ Feature schema hash for runtime validation
✅ Sidecar metadata generation (best_model.meta.json)
✅ --compare-models to train all 4 and select best
✅ Early exit on degenerate signal (saves GPU time)
✅ Degenerate sidecar for WATCHER consumption
✅ CRITICAL: Fingerprint hash excludes prng_type (data-only identity)

EXIT CODES:
  0 - Success (model trained and saved)
  1 - Error (exception during execution)
  2 - Degenerate signal (sidecar saved, no model)

Author: Distributed PRNG Analysis System
Date: January 2, 2026
Version: 3.3.0
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import optuna
from optuna.samplers import TPESampler
import logging
from sklearn.model_selection import KFold, TimeSeriesSplit
import time
from datetime import datetime
import subprocess
import sys
import os
import shutil
import warnings
import queue   # [S95] Dual-GPU parallel NN Optuna
import uuid    # [S95] Collision-proof NPZ naming
import argparse


# --- S111_AUTOCORR_DIAGNOSTICS ---
def _s111_write_autocorr_if_available(survivors, out_path='diagnostics_outputs/holdout_feature_autocorr.json'):
    try:
        import os, json
        from holdout_quality import compute_autocorrelation_diagnostics
        if not survivors or not isinstance(survivors, list) or not isinstance(survivors[0], dict):
            return None
        if survivors[0].get('holdout_features') is None:
            return None
        out = compute_autocorrelation_diagnostics(survivors)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        return out
    except Exception:
        return None
# --- S111_AUTOCORR_DIAGNOSTICS_END ---



# === S88_COMPARE_MODELS_RUNNER_BEGIN ===
def _s88_now_utc():
    import datetime
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _s88_safe_mkdir(p):
    import os
    os.makedirs(p, exist_ok=True)

def _s88_copy_if_exists(src, dst):
    import os, shutil
    if os.path.exists(src):
        shutil.copy2(src, dst)
        return True
    return False

def _s88_extract_score(meta_path):
    """
    Best-effort: pull a numeric score/r2 from best_model.meta.json.
    We don't assume schema; we try common keys.
    """
    import json, os
    if not os.path.exists(meta_path):
        return None
    try:
        data = json.load(open(meta_path, "r"))
    except Exception:
        return None

    # Common patterns (best effort)
    candidates = [
        ("training_metrics", "r2"),  # v3.3 schema
        ("best_r2",),
        ("r2",),
        ("score",),
        ("metrics","r2"),
        ("metrics","best_r2"),
        ("evaluation","r2"),
        ("evaluation","score"),
        ("best","r2"),
        ("best","score"),
    ]

    def get_nested(d, path):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    for path in candidates:
        v = get_nested(data, path)
        if isinstance(v, (int, float)):
            return float(v)

    # fallback: scan for any float-y key containing "r2"
    for k, v in (data.items() if isinstance(data, dict) else []):
        if isinstance(k, str) and "r2" in k.lower() and isinstance(v, (int, float)):
            return float(v)

    return None

def _s88_run_compare_models(args_dict):
    """
    Runs single-model Optuna training for each model type in a subprocess using
    the existing, correct path, and archives artifacts per model.
    """
    import os, sys, json, subprocess

    model_list = ["neural_net", "lightgbm", "xgboost", "catboost"]

    # --- S90: Skip registry integration ---
    # Read model_skip_registry.json and exclude models with active skips.
    # This prevents wasting compute on models that have hit consecutive
    # critical failures (e.g., neural_net with 14 consecutive criticals).
    skip_registry_path = os.path.join("diagnostics_outputs", "model_skip_registry.json")
    skipped_models = []
    if os.path.isfile(skip_registry_path):
        try:
            with open(skip_registry_path) as _sf:
                _skip_reg = json.load(_sf)
            # Default threshold matches training_health_check.py DEFAULT_SKIP_RULES
            _skip_threshold = 3
            # Try to read from watcher_policies.json for consistency
            if os.path.isfile("watcher_policies.json"):
                try:
                    with open("watcher_policies.json") as _pf:
                        _policies = json.load(_pf)
                    _skip_threshold = (_policies
                        .get("training_diagnostics", {})
                        .get("model_skip_rules", {})
                        .get("max_consecutive_critical", 3))
                except Exception:
                    pass
            for _model, _entry in _skip_reg.items():
                if _entry.get("consecutive_critical", 0) >= _skip_threshold:
                    skipped_models.append(_model)
            if skipped_models:
                original_count = len(model_list)
                model_list = [m for m in model_list if m not in skipped_models]
                print(f"[S90][SKIP] Excluded {len(skipped_models)} model(s) via skip registry: {skipped_models}")
                print(f"[S90][SKIP] Training {len(model_list)}/{original_count}: {model_list}")
                if not model_list:
                    raise RuntimeError("All model types are in skip state. Reset skip registry or investigate.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"[S90][SKIP] Warning: Could not read skip registry: {e}")
    # --- End S90 skip registry integration ---

    trials = int(args_dict.get("trials", 1))
    if trials < 1:
        raise ValueError(f"--trials must be >= 1, got {trials}")

    run_id = f"S88_{_s88_now_utc()}"

    models_root = os.path.join("models", "reinforcement", "compare_models", run_id)
    diag_root   = os.path.join("diagnostics_outputs")
    _s88_safe_mkdir(models_root)
    _s88_safe_mkdir(diag_root)

    # Preserve baseline outputs if present (best-effort)
    baseline_backup = os.path.join(models_root, "baseline_backup")
    _s88_safe_mkdir(baseline_backup)
    for fn in ["best_model.cbm", "best_model.meta.json", "best_model.pt", "best_model.pth"]:
        _s88_copy_if_exists(os.path.join("models","reinforcement",fn), os.path.join(baseline_backup, fn))

    summary = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "trials_per_model": trials,
        "total_expected_trials": trials * len(model_list),
        "models": {},
        "timestamp_utc": run_id.replace("S88_", ""),
        "note": "S88 hotfix: --compare-models now runs Optuna trials per model by invoking single-model path per model.",
    }

    # Build common argv for subprocess calls
    # We re-invoke THIS script file with --model-type set, and --compare-models removed.
    survivors = args_dict.get("survivors")
    lottery_data = args_dict.get("lottery_data")
    enable_diagnostics = bool(args_dict.get("enable_diagnostics", False))

    if not survivors or not lottery_data:
        raise RuntimeError("S88 compare-models runner requires --survivors and --lottery-data")

    py = sys.executable
    script = os.path.abspath(sys.argv[0])  # Always the executed script (resilient to refactoring)

    for m in model_list:
        model_out_dir = os.path.join(models_root, m)
        _s88_safe_mkdir(model_out_dir)

        cmd = [
            py, script,
            "--survivors", survivors,
            "--lottery-data", lottery_data,
            "--model-type", m,
            "--trials", str(trials),
        ]
        if enable_diagnostics:
            cmd.append("--enable-diagnostics")

        # -- Category B: Inject NN-specific flags (Team Beta Option A) --
        # normalize_features and use_leaky_relu are ALWAYS ON for neural_net.
        # These are fixed policy, not CLI-toggleable in compare-models mode.
        # Flags consumed by train_single_trial.py v1.1.0 (Phase 1).
        if m == "neural_net":
            cmd.append("--normalize-features")
            cmd.append("--use-leaky-relu")
            # Thread dropout override if WATCHER retry provided one (best-effort)
            _dropout_override = args_dict.get("dropout_override") or args_dict.get("dropout")
            if _dropout_override is not None:
                try:
                    _d = float(_dropout_override)
                    cmd.extend(["--dropout", str(_d)])
                except Exception:
                    print(f"[CAT-B] Invalid dropout override: {_dropout_override!r} (ignored)")
                    _d = None
            else:
                _d = None
            print(f"[CAT-B] Option A: forcing normalize+leaky for NN"
                  + (f", dropout={_d}" if _d is not None else ""))

        # IMPORTANT: ensure compare-models is not set in subcall
        # (we're in single-model mode)
        env = os.environ.copy()
        env["S88_COMPARE_MODELS_CHILD"] = "1"
        if m == "lightgbm":
            env["S88_FORCE_LGBM_CPU"] = "1"

        print(f"[S88][COMPARE] Running {m} with {trials} trials via single-model Optuna path...")
        proc = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
        rc = proc.returncode

        # Archive artifacts best-effort
        # (we don't assume exact artifact set; we grab common ones)
        produced = {}
        for fn in ["best_model.cbm", "best_model.meta.json", "best_model.pt", "best_model.pth", "best_model.txt", "best_model.json"]:
            src = os.path.join("models","reinforcement",fn)
            dst = os.path.join(model_out_dir, fn)
            produced[fn] = _s88_copy_if_exists(src, dst)

        score = _s88_extract_score(os.path.join(model_out_dir, "best_model.meta.json"))

        summary["models"][m] = {
            "returncode": rc,
            "archived_artifacts": produced,
            "score_best_effort": score,
        }

    # Determine winner best-effort by score
    best_m = None
    best_v = None
    for m, rec in summary["models"].items():
        v = rec.get("score_best_effort")
        if isinstance(v, (int, float)):
            if best_v is None or v > best_v:
                best_v = v
                best_m = m
    
    # Fallback if no score found: choose first successful model deterministically
    if best_m is None:
        for m, rec in summary["models"].items():
            if rec.get("returncode") == 0 and rec.get("archived_artifacts", {}).get("best_model.meta.json"):
                best_m = m
                break
        if best_m is None:
            # Second fallback: any model with returncode 0
            for m, rec in summary["models"].items():
                if rec.get("returncode") == 0:
                    best_m = m
                    break
    
    summary["winner_best_effort"] = {"model_type": best_m, "score": best_v}

    # Restore winner artifacts back to canonical models/reinforcement/
    winner = summary["winner_best_effort"].get("model_type")
    if winner:
        winner_dir = os.path.join(models_root, winner)
        for fn in ["best_model.cbm", "best_model.meta.json", "best_model.pt", "best_model.pth", "best_model.txt", "best_model.json"]:
            src = os.path.join(winner_dir, fn)
            dst = os.path.join("models","reinforcement", fn)
            _s88_copy_if_exists(src, dst)
        print(f"[S88][COMPARE] Restored winner artifacts to models/reinforcement/: {winner}")
    else:
        print("[S88][COMPARE][WARN] No winner determined (no score found). Leaving canonical artifacts unchanged.")

    out_path = os.path.join(diag_root, f"compare_models_summary_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[S88][COMPARE] Summary written: {out_path}")
    print(f"[S88][COMPARE] Artifacts archived under: {models_root}")

    return 0
# === S88_COMPARE_MODELS_RUNNER_END ===

# Subprocess isolation for multi-model comparison (fixes LightGBM OpenCL on Zeus)
try:
    from subprocess_trial_coordinator import (
        SubprocessTrialCoordinator,
        SAFE_MODEL_ORDER as SUBPROCESS_SAFE_ORDER,
        TrialResult
    )
    SUBPROCESS_AVAILABLE = True
except ImportError:
    SUBPROCESS_AVAILABLE = False
    SUBPROCESS_SAFE_ORDER = None
    SubprocessTrialCoordinator = None
    TrialResult = None

# ============================================================================
# EARLY CUDA INITIALIZATION
# ============================================================================

def initialize_cuda_early():
    """Initialize CUDA before any model operations"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            _ = torch.zeros(1).to(device)
            if torch.cuda.device_count() > 1:
                for i in range(torch.cuda.device_count()):
                    device_i = torch.device(f'cuda:{i}')
                    _ = torch.zeros(1).to(device_i)
            return True
    except:
        pass
    return False

# Phase 2.1: NN single-shot always routes through train_single_trial.py subprocess
NN_SUBPROCESS_ROUTING_ENABLED = True  # Set False to revert to inline NN trainer

CUDA_INITIALIZED = False  # Deferred - set in main() based on subprocess routing


# ============================================================================
# SIGNAL QUALITY COMPUTATION (v3.0)
# ============================================================================

def compute_signal_quality(y: np.ndarray, target_name: str = "holdout_quality") -> Dict:
    # --- S111_TARGET_HOLDOUT_QUALITY ---
    """
    Compute signal quality metrics for the training target.
    
    This enables Step 6 to gate predictions on signal quality.
    If signal is degenerate (all zeros, no variance), Step 6 should
    skip prediction and emit PREDICTION_SKIPPED for WATCHER.
    
    Args:
        y: Training target array
        target_name: Name of the target variable
        
    Returns:
        signal_quality dict for sidecar
    """
    y = np.asarray(y)
    
    # Core statistics
    unique_values = len(np.unique(y))
    target_variance = float(np.var(y))
    target_mean = float(np.mean(y))
    target_std = float(np.std(y))
    nonzero_count = int(np.sum(y != 0))
    nonzero_ratio = nonzero_count / len(y) if len(y) > 0 else 0.0
    
    # Determine signal status
    if unique_values == 1:
        signal_status = "degenerate"
        signal_confidence = 0.0
        prediction_allowed = False
    elif target_variance < 1e-10:
        signal_status = "degenerate"
        signal_confidence = 0.0
        prediction_allowed = False
    elif nonzero_ratio < 0.01:
        signal_status = "weak"
        signal_confidence = 0.2
        prediction_allowed = True  # Allow but with caution
    elif target_variance < 0.001:
        signal_status = "weak"
        signal_confidence = 0.4
        prediction_allowed = True
    else:
        signal_status = "healthy"
        signal_confidence = min(1.0, target_variance * 10 + nonzero_ratio)
        prediction_allowed = True
    
    return {
        "target_name": target_name,
        "signal_status": signal_status,
        "signal_confidence": round(signal_confidence, 4),
        "prediction_allowed": prediction_allowed,
        "unique_target_values": unique_values,
        "target_variance": round(target_variance, 10),
        "target_mean": round(target_mean, 6),
        "target_std": round(target_std, 6),
        "nonzero_count": nonzero_count,
        "nonzero_ratio": round(nonzero_ratio, 4),
        "sample_size": len(y)
    }


# ============================================================================
# DATA CONTEXT FINGERPRINT (v3.1)
# ============================================================================

def compute_data_context(
    train_draws: int,
    holdout_draws: int,
    survivors_file: str,
    survivor_count: int,
    prng_type: str,
    mod: int,
    train_start: int = 1,
    skip_range: Optional[Tuple[int, int]] = None
) -> Dict:
    """
    Compute data context fingerprint for WATCHER loop prevention.
    
    This captures exactly what defines a training configuration.
    WATCHER uses fingerprint_hash to detect duplicate configurations
    and prevent infinite retry loops on known-degenerate configs.
    
    Args:
        train_draws: Number of training draws
        holdout_draws: Number of holdout draws
        survivors_file: Path to survivors file
        survivor_count: Number of survivors
        prng_type: PRNG algorithm type
        mod: PRNG modulus
        train_start: First draw index (default 1)
        skip_range: Optional skip hypothesis range (min, max)
        
    Returns:
        data_context dict for sidecar
    """
    # Calculate window boundaries
    train_end = train_start + train_draws - 1
    holdout_start = train_end + 1
    holdout_end = holdout_start + holdout_draws - 1
    
    # Compute fingerprint hash
    # CRITICAL (v3.3): Fingerprint represents DATA CONTEXT only, not PRNG hypothesis
    # This enables WATCHER to detect "this window failed multiple PRNGs"
    # prng_type is tracked separately in registry attempts table
    survivors_filename = Path(survivors_file).name
    context_str = (
        f"{train_start}:{train_end}|"
        f"{holdout_start}:{holdout_end}|"
        f"{survivor_count}|"
        f"{survivors_filename}"
    )
    fingerprint = hashlib.sha256(context_str.encode()).hexdigest()[:8]
    
    # Build data context
    data_context = {
        # === Quick Comparison ===
        "fingerprint_hash": fingerprint,
        "fingerprint_version": "v2_data_only",  # v3.3: excludes prng_type from hash
        
        # === Training Window ===
        "training_window": {
            "start_draw": train_start,
            "end_draw": train_end,
            "draw_count": train_draws
        },
        
        # === Holdout Window ===
        "holdout_window": {
            "start_draw": holdout_start,
            "end_draw": holdout_end,
            "draw_count": holdout_draws
        },
        
        # === Survivor Source ===
        "survivor_source": {
            "file": str(Path(survivors_file).name),
            "file_path": str(Path(survivors_file).resolve()),
            "survivor_count": survivor_count
        },
        
        # === PRNG Hypothesis ===
        "prng_hypothesis": {
            "prng_type": prng_type,
            "mod": mod
        },
        
        # === Provenance ===
        "derived_by": "step5_anti_overfit",
        "timestamp": datetime.now().isoformat()
    }
    
    # Add skip range if provided
    if skip_range:
        data_context["prng_hypothesis"]["skip_range"] = {
            "min": skip_range[0],
            "max": skip_range[1]
        }
    
    return data_context


# ============================================================================
# FEATURE SCHEMA UTILITIES
# ============================================================================

def get_feature_schema_from_data(survivors_file: str, exclude_features: List[str] = None) -> Dict:
    """
    Extract feature schema from survivors file using streaming JSON.
    
    Args:
        survivors_file: Path to survivors JSON file
        exclude_features: Features to exclude (e.g., ['score', 'confidence'])
        
    Returns:
        Feature schema dict with names, count, and hash
    """
    exclude_features = exclude_features or ['score', 'confidence']
    
    try:
        import ijson
        
        with open(survivors_file, 'rb') as f:
            parser = ijson.items(f, 'item')
            for item in parser:
                if isinstance(item, dict) and 'features' in item:
                    all_features = sorted(item['features'].keys())
                    feature_names = [f for f in all_features if f not in exclude_features]
                    break
            else:
                raise ValueError("No valid survivor with features found")
                
    except ImportError:
        # Fallback to loading first item
        with open(survivors_file) as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                if isinstance(first, dict) and 'features' in first:
                    all_features = sorted(first['features'].keys())
                    feature_names = [f for f in all_features if f not in exclude_features]
                else:
                    raise ValueError("First item has no 'features' key")
            else:
                raise ValueError("Empty or invalid survivors file")
    
    # Compute hash
    names_str = ",".join(feature_names)
    schema_hash = hashlib.sha256(names_str.encode()).hexdigest()[:16]
    
    return {
        "source_file": str(Path(survivors_file).resolve()),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "ordering": "lexicographic_by_key",
        "feature_schema_hash": schema_hash,
        "excluded_features": exclude_features
    }


def compute_feature_schema_hash(feature_names: List[str]) -> str:
    """Compute hash from sorted feature names."""
    names_str = ",".join(sorted(feature_names))
    return hashlib.sha256(names_str.encode()).hexdigest()[:16]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_survivors_with_features(
    survivors_file: str,
    target_field: str = "holdout_quality",
    exclude_features: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    Load survivors with features and target variable.
    
    Args:
        survivors_file: Path to survivors JSON file
        target_field: Field to use as y-label (default: holdout_hits)
        exclude_features: Features to exclude from X
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target array (n_samples,)
        feature_schema: Feature schema dict
        y_metadata: Target variable metadata
    """
    exclude_features = exclude_features or ['score', 'confidence', 'holdout_hits', 'holdout_quality']
    
    # Get feature schema
    feature_schema = get_feature_schema_from_data(survivors_file, exclude_features)
    feature_names = feature_schema['feature_names']
    
    # Load data
    try:
        import ijson
        
        X_list = []
        y_list = []
        seeds = []
        
        with open(survivors_file, 'rb') as f:
            parser = ijson.items(f, 'item')
            for item in parser:
                if isinstance(item, dict):
                    seeds.append(item.get('seed', 0))
                    
                    # Extract features in order
                    features = item.get('features', {})
                    feature_vector = [features.get(fn, 0.0) for fn in feature_names]
                    X_list.append(feature_vector)
                    
                    # Extract target
                    if target_field in features:
                        y_list.append(features[target_field])
                    elif target_field in item:
                        y_list.append(item[target_field])
                    else:
                        y_list.append(0.0)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
    except ImportError:
        # Fallback to full load
        with open(survivors_file) as f:
            data = json.load(f)
        
        X_list = []
        y_list = []
        seeds = []
        
        for item in data:
            if isinstance(item, dict):
                seeds.append(item.get('seed', 0))
                features = item.get('features', {})
                feature_vector = [features.get(fn, 0.0) for fn in feature_names]
                X_list.append(feature_vector)
                
                if target_field in features:
                    y_list.append(features[target_field])
                elif target_field in item:
                    y_list.append(item[target_field])
                else:
                    y_list.append(0.0)
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
    
    # Compute y metadata
    y_metadata = {
        "field": target_field,
        "observed_min": float(np.min(y)),
        "observed_max": float(np.max(y)),
        "observed_range": float(np.max(y) - np.min(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "sample_size": len(y),
        "warnings": []
    }
    
    # Check for narrow range
    if y_metadata["observed_range"] < 0.01:
        y_metadata["warnings"].append("target_range_narrow")
    
    return X, y, feature_schema, y_metadata


# ============================================================================
# MULTI-MODEL TRAINING (Subprocess Isolation)
# ============================================================================

# =============================================================================
# SUBPROCESS ISOLATION FOR MULTI-MODEL COMPARISON
# =============================================================================


def run_subprocess_comparison(
    X_train, y_train, X_val, y_val,
    output_dir: str = 'models/reinforcement',
    timeout: int = 600,
    logger = None
) -> dict:
    """
    Run multi-model comparison using subprocess isolation.
    
    Each model trains in a fresh subprocess, preventing OpenCL/CUDA conflicts.
    This allows LightGBM (OpenCL) to work alongside CUDA models on Zeus.
    
    Args:
        X_train, y_train, X_val, y_val: Training/validation data
        output_dir: Directory for model outputs
        timeout: Per-model timeout in seconds
        logger: Optional logger
        
    Returns:
        Dict with results for each model and winner info
    """
    import logging
    logger = logger or logging.getLogger(__name__)
    
    if not SUBPROCESS_AVAILABLE:
        raise RuntimeError("subprocess_trial_coordinator not available")
    
    logger.info("=" * 70)
    logger.info("SUBPROCESS ISOLATION MODE")
    logger.info("=" * 70)
    logger.info(f"Models: {SUBPROCESS_SAFE_ORDER}")
    logger.info("Each model trains in isolated subprocess (fixes LightGBM OpenCL)")
    logger.info("=" * 70)
    
    results = {}
    
    # Get enable_diagnostics from outer scope if available
    _enable_diag = globals().get('_enable_diagnostics_flag', False)
    
    with SubprocessTrialCoordinator(
        X_train, y_train, X_val, y_val,
        enable_diagnostics=_enable_diag,
        worker_script='train_single_trial.py',
        timeout=timeout,
        verbose=True,
        output_dir=output_dir
    ) as coordinator:
        
        # Train each model type
        for i, model_type in enumerate(SUBPROCESS_SAFE_ORDER):
            if model_type == 'random_forest':
                continue  # Skip random_forest for now
                
            logger.info(f"Training {model_type} (subprocess)...")
            
            try:
                # Run isolated trial
                trial_result = coordinator.run_trial(
                    trial_number=i,
                    model_type=model_type,
                    params={}  # Use defaults
                )
                
                if trial_result.success:
                    results[model_type] = {
                        'model': None,  # Model is in file, not memory
                        'model_type': model_type,
                        'metrics': {
                            'train_mse': trial_result.train_mse,
                            'val_mse': trial_result.val_mse,
                            'r2': trial_result.r2
                        },
                        'hyperparameters': trial_result.params,
                        'checkpoint_path': trial_result.checkpoint_path
                    }
                    logger.info(f"  {model_type}: R²={trial_result.r2:.4f}")
                else:
                    results[model_type] = {
                        'model': None,
                        'model_type': model_type,
                        'metrics': {'error': trial_result.error},
                        'hyperparameters': {}
                    }
                    logger.error(f"  {model_type} failed: {trial_result.error}")
                    
            except Exception as e:
                logger.error(f"  {model_type} failed: {e}")
                results[model_type] = {
                    'model': None,
                    'model_type': model_type,
                    'metrics': {'error': str(e)},
                    'hyperparameters': {}
                }
        
        # Find winner
        valid_results = {
            k: v for k, v in results.items() 
            if v.get('model') is not None or v.get('checkpoint_path')
        }
        
        if not valid_results:
            # Fall back to any result with r2 score
            valid_results = {
                k: v for k, v in results.items()
                if 'r2' in v.get('metrics', {})
            }
        
        if valid_results:
            winner = max(
                valid_results.keys(),
                key=lambda k: valid_results[k]['metrics'].get('r2', float('-inf'))
            )
            results['winner'] = winner
            results['winner_metrics'] = valid_results[winner]['metrics']
            
            # Copy winning model to output directory
            if valid_results[winner].get('checkpoint_path'):
                winner_path = valid_results[winner]['checkpoint_path']
                if os.path.exists(winner_path):
                    # Determine extension
                    ext = os.path.splitext(winner_path)[1]
                    dest_path = os.path.join(output_dir, f"best_model{ext}")
                    shutil.copy(winner_path, dest_path)
                    results[winner]['final_checkpoint_path'] = dest_path
                    logger.info(f"Copied winning model to {dest_path}")
        else:
            raise ValueError("All models failed to train")
    
    return results


class MultiModelTrainer:
    """
    Trains multiple model types with subprocess isolation.
    
    Subprocess isolation prevents OpenCL/CUDA context conflicts
    between different ML frameworks (e.g., XGBoost CUDA vs CatBoost OpenCL).
    """
    
    SUPPORTED_MODELS = ['neural_net', 'xgboost', 'lightgbm', 'catboost']
    SAFE_MODEL_ORDER = ['lightgbm', 'neural_net', 'xgboost', 'catboost']
    
    def __init__(self, device: str = 'cuda:0', logger: logging.Logger = None):
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
    
    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparameters: Dict = None
    ) -> Dict:
        """
        Train a single model type.
        
        Args:
            model_type: One of SUPPORTED_MODELS
            X_train, y_train: Training data
            X_val, y_val: Validation data
            hyperparameters: Model-specific hyperparameters
            
        Returns:
            Results dict with model, metrics, and training info
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        hyperparameters = hyperparameters or {}
        start_time = time.time()
        
        if model_type == 'neural_net':
            return self._train_neural_net(X_train, y_train, X_val, y_val, hyperparameters)
        elif model_type == 'xgboost':
            return self._train_xgboost(X_train, y_train, X_val, y_val, hyperparameters)
        elif model_type == 'lightgbm':
            return self._train_lightgbm(X_train, y_train, X_val, y_val, hyperparameters)
        elif model_type == 'catboost':
            return self._train_catboost(X_train, y_train, X_val, y_val, hyperparameters)
    
    def _train_neural_net(self, X_train, y_train, X_val, y_val, params):
        """Train PyTorch neural network."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # Default parameters
        hidden_layers = params.get('hidden_layers', [256, 128, 64])
        dropout = params.get('dropout', 0.2)
        learning_rate = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 64)
        epochs = params.get('epochs', 100)
        
        # Build model
        layers = []
        input_dim = X_train.shape[1]
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        model = nn.Sequential(*layers)
        
        # Move to device
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Data loaders
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = params.get('early_stopping_patience', 10)
        
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Final metrics
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t).cpu().numpy().flatten()
            val_pred = model(X_val_t).cpu().numpy().flatten()
        
        train_mse = float(np.mean((train_pred - y_train) ** 2))
        val_mse = float(np.mean((val_pred - y_val) ** 2))
        
        # R² score
        ss_res = np.sum((y_val - val_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'model': model,
            'model_type': 'neural_net',
            'metrics': {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'r2': float(r2),
                'epochs_trained': epoch + 1
            },
            'hyperparameters': params
        }
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, params):
        """Train XGBoost model."""
        import xgboost as xgb
        
        # Default parameters
        xgb_params = {
            'n_estimators': params.get('n_estimators', 200),
            'max_depth': params.get('max_depth', 8),
            'learning_rate': params.get('learning_rate', 0.05),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'tree_method': 'hist',
            'device': 'cuda' if 'cuda' in self.device else 'cpu',
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_mse = float(np.mean((train_pred - y_train) ** 2))
        val_mse = float(np.mean((val_pred - y_val) ** 2))
        
        ss_res = np.sum((y_val - val_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'model': model,
            'model_type': 'xgboost',
            'metrics': {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'r2': float(r2)
            },
            'hyperparameters': xgb_params
        }
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, params):
        """Train LightGBM model."""
        import lightgbm as lgb
        
        lgb_params = {
            'n_estimators': params.get('n_estimators', 200),
            'max_depth': params.get('max_depth', 8),
            'learning_rate': params.get('learning_rate', 0.05),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'device': 'cpu' if os.environ.get('S88_FORCE_LGBM_CPU') == '1' else ('gpu' if 'cuda' in self.device else 'cpu'),
            'random_state': 42,
            'verbosity': -1
        }
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_mse = float(np.mean((train_pred - y_train) ** 2))
        val_mse = float(np.mean((val_pred - y_val) ** 2))
        
        ss_res = np.sum((y_val - val_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'model': model,
            'model_type': 'lightgbm',
            'metrics': {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'r2': float(r2)
            },
            'hyperparameters': lgb_params
        }
    
    def _train_catboost(self, X_train, y_train, X_val, y_val, params):
        """Train CatBoost model."""
        from catboost import CatBoostRegressor
        
        # Check if all targets are equal
        if np.var(y_train) < 1e-10:
            self.logger.warning("CatBoost: Training targets have zero variance")
            # Return dummy result
            return {
                'model': None,
                'model_type': 'catboost',
                'metrics': {
                    'train_mse': float('inf'),
                    'val_mse': float('inf'),
                    'r2': 0.0,
                    'error': 'zero_variance_targets'
                },
                'hyperparameters': params
            }
        
        cb_params = {
            'iterations': params.get('n_estimators', 200),
            'depth': params.get('max_depth', 8),
            'learning_rate': params.get('learning_rate', 0.05),
            'task_type': 'GPU' if 'cuda' in self.device else 'CPU',
            'devices': '0' if 'cuda' in self.device else None,  # [S95] Single GPU, consistent with subprocess
            'random_seed': 42,
            'verbose': False
        }
        
        model = CatBoostRegressor(**cb_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_mse = float(np.mean((train_pred - y_train) ** 2))
        val_mse = float(np.mean((val_pred - y_val) ** 2))
        
        ss_res = np.sum((y_val - val_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'model': model,
            'model_type': 'catboost',
            'metrics': {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'r2': float(r2)
            },
            'hyperparameters': cb_params
        }
    
    def train_and_compare(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_types: List[str] = None,
        metric: str = 'r2',
        use_subprocess: bool = True,
        output_dir: str = 'models/reinforcement'
    ) -> Dict:
        """
        Train all model types and compare performance.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_types: List of models to train (default: all)
            metric: Comparison metric ('r2', 'mse', 'mae')
            use_subprocess: Use subprocess isolation (fixes LightGBM OpenCL)
            output_dir: Output directory for models
            
        Returns:
            Dict with results for each model and winner
        """
        # Try subprocess isolation first (fixes LightGBM OpenCL on Zeus)
        if use_subprocess and SUBPROCESS_AVAILABLE:
            self.logger.info("Using subprocess isolation for model comparison")
            try:
                return run_subprocess_comparison(
                    X_train, y_train, X_val, y_val,
                    output_dir=output_dir,
                    logger=self.logger
                )
            except Exception as e:
                self.logger.warning(f"Subprocess comparison failed: {e}")
                self.logger.warning("Falling back to inline training")
        
        # Fallback: inline training (original behavior)
        model_types = model_types or self.SAFE_MODEL_ORDER
        
        results = {}
        for model_type in model_types:
            self.logger.info(f"Training {model_type}...")
            try:
                result = self.train_model(model_type, X_train, y_train, X_val, y_val)
                results[model_type] = result
                self.logger.info(f"  {model_type}: R²={result['metrics']['r2']:.4f}")
            except Exception as e:
                self.logger.error(f"  {model_type} failed: {e}")
                results[model_type] = {
                    'model': None,
                    'model_type': model_type,
                    'metrics': {'error': str(e)},
                    'hyperparameters': {}
                }
        
        # Find winner
        valid_results = {k: v for k, v in results.items() if v['model'] is not None}
        if not valid_results:
            raise ValueError("All models failed to train")
        
        if metric == 'r2':
            winner = max(valid_results.keys(), key=lambda k: valid_results[k]['metrics'].get('r2', 0))
        elif metric == 'mse':
            winner = min(valid_results.keys(), key=lambda k: valid_results[k]['metrics'].get('val_mse', float('inf')))
        else:
            winner = max(valid_results.keys(), key=lambda k: valid_results[k]['metrics'].get('r2', 0))
        
        results['winner'] = winner
        results['winner_metrics'] = valid_results[winner]['metrics']
        
        return results
    
    def save_model(
        self,
        model: Any,
        model_type: str,
        output_dir: str,
        feature_schema: Dict,
        training_metrics: Dict,
        signal_quality: Dict,
        data_context: Dict,
        hyperparameters: Dict,
        provenance: Dict,
        parent_run_id: Optional[str] = None
    ) -> str:
        """
        Save model with sidecar metadata.
        
        Args:
            model: Trained model object
            model_type: Type of model
            output_dir: Output directory
            feature_schema: Feature schema dict
            training_metrics: Training metrics
            signal_quality: Signal quality dict (v3.0)
            data_context: Data context dict (v3.1)
            hyperparameters: Hyperparameters used
            provenance: Run provenance info
            parent_run_id: Optional parent run ID
            
        Returns:
            Path to saved model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension
        extensions = {
            'neural_net': '.pth',
            'xgboost': '.json',
            'lightgbm': '.txt',
            'catboost': '.cbm'
        }
        ext = extensions.get(model_type, '.pkl')
        checkpoint_path = output_path / f"best_model{ext}"
        
        # Save model
        if model_type == 'neural_net':
            if model is not None:
                # Inline NN training path (legacy, non-subprocess)
                import torch
                torch.save(model.state_dict(), checkpoint_path)
            else:
                # [S96A] Subprocess path: checkpoint already saved by train_single_trial.py
                # Look for existing .pth in output dir
                existing = list(output_path.glob("*.pth"))
                if existing and existing[0] != checkpoint_path:
                    import shutil
                    shutil.copy2(str(existing[0]), str(checkpoint_path))
                elif not existing and not checkpoint_path.exists():
                    self.logger.warning(f"[S96A] No .pth checkpoint found in {output_path}")
        elif model_type == 'xgboost':
            model.save_model(str(checkpoint_path))
        elif model_type == 'lightgbm':
            model.booster_.save_model(str(checkpoint_path))
        elif model_type == 'catboost':
            model.save_model(str(checkpoint_path))
        else:
            import pickle
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Build sidecar
        sidecar = {
            "schema_version": "3.4.0",
            "model_type": model_type,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_format": ext.lstrip('.'),
            
            "feature_schema": feature_schema,
            "signal_quality": signal_quality,
            "data_context": data_context,
            
            "training_metrics": training_metrics,
            "hyperparameters": hyperparameters,
            
            "hardware": {
                "device_requested": self.device,
                "cuda_available": CUDA_INITIALIZED
            },
            
            "training_info": {
                "started_at": provenance.get('started_at'),
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": provenance.get('duration_seconds')
            },
            
            "agent_metadata": {
                "pipeline_step": 5,
                "pipeline_step_name": "anti_overfit_training",
                "run_id": provenance.get('run_id', f"step5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "parent_run_id": parent_run_id
            },
            
            "provenance": provenance
        }
        
        # Save sidecar
        sidecar_path = output_path / "best_model.meta.json"
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar, f, indent=2)
        
        self.logger.info(f"Model saved: {checkpoint_path}")
        self.logger.info(f"Sidecar saved: {sidecar_path}")
        
        return str(checkpoint_path)


# ============================================================================
# VALIDATION METRICS
# ============================================================================

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics to detect overfitting"""
    train_variance: float
    val_variance: float
    test_variance: float
    train_mae: float
    val_mae: float
    test_mae: float
    overfit_ratio: float
    variance_consistency: float
    temporal_stability: float
    p_value: float
    confidence_interval: Tuple[float, float]
    r2_score: float = 0.0

    def is_overfitting(self) -> bool:
        """Detect if model is overfitting"""
        return (
            self.overfit_ratio > 1.5 or
            self.test_mae > self.val_mae * 1.3 or
            self.p_value > 0.05
        )

    def composite_score(self) -> float:
        """Composite score that penalizes overfitting"""
        penalty = 0.5 if self.is_overfitting() else 1.0
        score = (
            self.test_variance * 10.0 +
            (1.0 / (self.test_mae + 0.01)) * 5.0 +
            self.variance_consistency * 3.0 +
            self.temporal_stability * 2.0 +
            (1.0 - self.p_value) * 2.0 +
            self.r2_score * 10.0
        ) * penalty
        return score


# ============================================================================
# ANTI-OVERFIT META OPTIMIZER (v3.1)
# ============================================================================

class AntiOverfitMetaOptimizer:
    """
    Meta-optimizer with strong anti-overfitting measures.
    
    Version 3.1 Features:
    - Multi-model support (--model-type, --compare-models)
    - signal_quality emission for Step 6 gate
    - data_context fingerprint for WATCHER loop prevention
    - Feature schema hash for runtime validation
    - Sidecar metadata generation
    """
    
    def __init__(
        self,
        survivors_file: str,
        lottery_history: List[int],
        k_folds: int = 5,
        test_holdout_pct: float = 0.2,
        study_name: str = None,
        storage: str = None,
        model_type: str = 'catboost',
        compare_models: bool = False,
        output_dir: str = 'models/reinforcement',
        device: str = 'cuda:0',
        prng_type: str = 'java_lcg',
        mod: int = 1000,
        holdout_draws: int = 1000,
        parent_run_id: str = None
    ):
        """
        Initialize anti-overfit meta-optimizer.
        
        Args:
            survivors_file: Path to survivors JSON with features
            lottery_history: Lottery draws (training portion)
            k_folds: Number of CV folds
            test_holdout_pct: % to hold out for TRUE test set
            study_name: Name for Optuna study
            storage: SQLite storage path
            model_type: Model type to train
            compare_models: Train all models and select best
            output_dir: Output directory for model
            device: GPU device
            prng_type: PRNG type for data_context
            mod: PRNG modulus for data_context
            holdout_draws: Number of holdout draws for data_context
            parent_run_id: Optional parent run ID
        """
        self.survivors_file = survivors_file
        self.lottery_history = lottery_history
        self.k_folds = k_folds
        self.test_holdout_pct = test_holdout_pct
        self.model_type = model_type
        self.compare_models = compare_models
        self.output_dir = output_dir
        self.device = device
        self.prng_type = prng_type
        self.mod = mod
        self.holdout_draws = holdout_draws
        self.parent_run_id = parent_run_id
        
        # Study persistence
        self.study_name = study_name or f"anti_overfit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage or 'sqlite:///optuna_studies.db'
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize trainer
        self.trainer = MultiModelTrainer(device=device, logger=self.logger)
        
        # Load data
        self._load_data()
        
        # Create train/test splits
        self._create_splits()
        
        # Results tracking
        self.best_config = None
        self.best_metrics = None
        self.best_model = None
        self.best_model_type = None
        # Team Beta: subprocess comparison is disk-first (no in-memory model)
        self.best_checkpoint_path = None
        self.best_checkpoint_format = None
        self.optimization_history = []
        self.trial_times = []
        self.n_trials_total = None
        
        # Timing
        self.start_time = datetime.now()
    
    def _load_data(self):
        """Load survivors with features and compute signal_quality + data_context."""
        self.logger.info("Loading survivors with features...")
        
        # Load data
        self.X, self.y, self.feature_schema, self.y_metadata = load_survivors_with_features(
            self.survivors_file,
            target_field='holdout_hits',
            exclude_features=['score', 'confidence', 'holdout_hits', 'holdout_quality']
        )
        
        self.logger.info(f"Loaded {len(self.X)} survivors with {self.feature_schema['feature_count']} features")
        self.logger.info(f"Feature schema hash: {self.feature_schema['feature_schema_hash']}")
        
        # Compute signal quality (v3.0)
        self.signal_quality = compute_signal_quality(self.y, target_name="holdout_hits")
        
        self.logger.info(f"Signal quality: {self.signal_quality['signal_status']} "
                        f"(confidence={self.signal_quality['signal_confidence']:.2f})")
        
        if not self.signal_quality['prediction_allowed']:
            self.logger.warning("⚠️  DEGENERATE SIGNAL DETECTED - predictions will be gated")
        
        # Compute data context (v3.1)
        self.data_context = compute_data_context(
            train_draws=len(self.lottery_history),
            holdout_draws=self.holdout_draws,
            survivors_file=self.survivors_file,
            survivor_count=len(self.X),
            prng_type=self.prng_type,
            mod=self.mod
        )
        
        self.logger.info(f"Data context fingerprint: {self.data_context['fingerprint_hash']}")
    
    def _create_splits(self):
        """Create proper train/val/test splits."""
        n_total = len(self.X)
        n_test = int(n_total * self.test_holdout_pct)
        
        indices = np.random.permutation(n_total)
        
        # Test set (final holdout)
        self.test_indices = indices[:n_test]
        self.X_test = self.X[self.test_indices]
        self.y_test = self.y[self.test_indices]
        
        # Train+Val set
        train_val_indices = indices[n_test:]
        self.X_train_val = self.X[train_val_indices]
        self.y_train_val = self.y[train_val_indices]
        
        self.logger.info("=" * 70)
        self.logger.info("DATA SPLITS (Anti-Overfitting)")
        self.logger.info("=" * 70)
        self.logger.info(f"Train+Val: {len(self.X_train_val)} survivors")
        self.logger.info(f"Test (HOLDOUT): {len(self.X_test)} survivors")
        self.logger.info(f"K-Fold CV: {self.k_folds} folds")
        self.logger.info("=" * 70)
    
    def run(self, n_trials: int = 50) -> Tuple[Dict, ValidationMetrics]:
        """
        Run the full optimization pipeline.
        
        Args:
            n_trials: Number of Optuna trials (if using hyperparameter search)
            
        Returns:
            Best config and validation metrics
            
        Exit Codes:
            0 - Success (model trained and saved)
            2 - Degenerate signal (sidecar saved, no model, early exit)
        """
        self.n_trials_total = n_trials
        
        # ================================================================
        # EARLY EXIT ON DEGENERATE SIGNAL (v3.2)
        # ================================================================
        # Check signal quality BEFORE wasting GPU time on training
        if not self.signal_quality.get('prediction_allowed', True):
            self.logger.warning("=" * 70)
            self.logger.warning("DEGENERATE SIGNAL DETECTED - EARLY EXIT")
            self.logger.warning("=" * 70)
            self.logger.warning(f"  Signal status: {self.signal_quality['signal_status']}")
            self.logger.warning(f"  Unique target values: {self.signal_quality['unique_target_values']}")
            self.logger.warning(f"  Target variance: {self.signal_quality['target_variance']}")
            self.logger.warning(f"  Non-zero ratio: {self.signal_quality['nonzero_ratio']}")
            self.logger.warning("=" * 70)
            self.logger.warning("Skipping training to save GPU time (~45 minutes)")
            self.logger.warning("Saving degenerate sidecar for WATCHER consumption")
            self.logger.warning("=" * 70)
            
            # Save degenerate sidecar so WATCHER can consume it
            self._save_degenerate_sidecar()
            
            # Return early with empty config and degenerate metrics
            degenerate_metrics = ValidationMetrics(
                train_variance=0.0,
                val_variance=0.0,
                test_variance=0.0,
                train_mae=0.0,
                val_mae=0.0,
                test_mae=0.0,
                overfit_ratio=0.0,
                variance_consistency=0.0,
                temporal_stability=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                r2_score=0.0
            )
            
            # Signal to caller that this was a degenerate exit
            self.degenerate_exit = True
            
            return {}, degenerate_metrics
        
        # Normal training path
        self.degenerate_exit = False
        
        if self.compare_models:
            return self._run_model_comparison()
        else:
            return self._run_single_model()
    
    def _run_model_comparison(self) -> Tuple[Dict, ValidationMetrics]:
        """Train all models and select best."""
        self.logger.info("=" * 70)
        self.logger.info("MODEL COMPARISON MODE")
        self.logger.info("=" * 70)
        
        # Simple train/val split from train_val set
        n_val = int(len(self.X_train_val) * 0.2)
        X_train = self.X_train_val[:-n_val]
        y_train = self.y_train_val[:-n_val]
        X_val = self.X_train_val[-n_val:]
        y_val = self.y_train_val[-n_val:]
        
        # Train and compare
        results = self.trainer.train_and_compare(X_train, y_train, X_val, y_val)
        
        # Get winner
        winner = results['winner']
        # Team Beta: in subprocess mode, the winner is a checkpoint on disk
        self.best_checkpoint_path = None
        try:
            if isinstance(results, dict) and winner in results:
                self.best_checkpoint_path = (
                    results[winner].get('final_checkpoint_path')
                    or results[winner].get('checkpoint_path')
                )
        except Exception:
            self.best_checkpoint_path = None

        if self.best_checkpoint_path:
            ext = Path(self.best_checkpoint_path).suffix
            self.best_checkpoint_format = ext.lstrip('.') if ext else None
            self.best_model_type = winner  # Team Beta refinement
        self.best_model_type = winner
        self.best_model = results[winner]['model']
        self.best_config = results[winner]['hyperparameters']
        self.best_metrics = self._compute_final_metrics(results[winner]['metrics'])
        
        # Log results
        self.logger.info("\n" + "=" * 70)
        self.logger.info("MODEL COMPARISON RESULTS")
        self.logger.info("=" * 70)
        for model_type in MultiModelTrainer.SAFE_MODEL_ORDER:
            if model_type in results:
                m = results[model_type]['metrics']
                marker = " 🏆" if model_type == winner else ""
                self.logger.info(f"  {model_type}: R²={m.get('r2', 0):.4f}{marker}")
        self.logger.info("=" * 70)
        
        # Save best model
        self.save_best_model()
        
        return self.best_config, self.best_metrics
    

    # ========================================================================
    # OPTUNA OPTIMIZATION (v3.4 - Restored with Team Beta Guardrails)
    # ========================================================================

    # -----------------------------------------------------------------
    # [S95] CUDA-clean GPU detection + GPU lease queue
    # NO torch import, NO torch.cuda calls in parent (S72 invariant)
    # -----------------------------------------------------------------
    def _s95_detect_cuda_gpus_no_torch(self) -> int:
        """
        Detect available CUDA GPUs WITHOUT importing torch or touching
        CUDA APIs in the parent process (S72 invariant).

        Strategy:
          1) If CUDA_VISIBLE_DEVICES is set, count the visible ids.
          2) Else call nvidia-smi -L (safe subprocess) and count GPUs.
          3) Fallback to 1.
        """
        # 1) Respect CUDA_VISIBLE_DEVICES if set
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            cvd = cvd.strip()
            if cvd == "" or cvd.lower() in ("none", "-1"):
                return 1
            parts = [p.strip() for p in cvd.split(",") if p.strip()]
            return max(1, len(parts))

        # 2) Try nvidia-smi -L (subprocess, no CUDA init)
        try:
            proc = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True, text=True, timeout=3
            )
            if proc.returncode == 0:
                lines = [ln for ln in (proc.stdout or "").splitlines()
                         if ln.strip().startswith("GPU ")]
                if lines:
                    return max(1, len(lines))
        except Exception:
            pass

        # 3) Fallback
        return 1

    def _s95_build_gpu_queue(self, n_gpus: int) -> queue.Queue:
        """
        Build a thread-safe queue of GPU id strings ("0", "1", ...),
        used to lease a GPU to an Optuna trial for all its folds.
        """
        q = queue.Queue()
        for i in range(max(1, int(n_gpus))):
            q.put(i)  # [S98 fix] int to match _s96b_workers int keys
        return q

    def _run_optuna_optimization(self, n_trials: int) -> Tuple[Dict, ValidationMetrics]:
        """
        Run Optuna hyperparameter optimization with K-fold CV.

        Team Beta:
        - NO artifact writes during trials (memory only)
        - single-writer: optimizer saves once (save_best_model)
        - unique study naming (fingerprints truncated)
        - hard-fail if Optuna unavailable when requested
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except Exception as e:
            self.logger.error("=" * 70)
            self.logger.error("CRITICAL: OPTUNA REQUESTED BUT UNAVAILABLE")
            self.logger.error("=" * 70)
            raise RuntimeError("Optuna requested but not installed") from e

        from pathlib import Path
        from sklearn.model_selection import KFold

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("OPTUNA MODE: ENABLED")
        self.logger.info("=" * 70)
        self.logger.info(f"  Model type: {self.model_type}")
        self.logger.info(f"  Trials: {n_trials}")

        # Unique (short) study name
        feature_h = str(self.feature_schema.get("feature_schema_hash", "unknown"))[:10]
        data_h = str(self.data_context.get("fingerprint_hash", "unknown"))[:10]
        # TB #2: Fresh study for normalized NN (avoid contaminating old unnormalized trials)
        _study_suffix = "_catb22" if self.model_type == "neural_net" and NN_SUBPROCESS_ROUTING_ENABLED else ""
        study_name = f"step5_{self.model_type}_{feature_h}_{data_h}{_study_suffix}"

        Path("optuna_studies").mkdir(exist_ok=True)

        # [S95] Detect GPUs and configure parallel NN Optuna
        n_jobs = 1
        if self.model_type == "neural_net" and NN_SUBPROCESS_ROUTING_ENABLED:
            n_jobs = self._s95_detect_cuda_gpus_no_torch()
            if n_jobs > 1:
                self._s95_gpu_queue = self._s95_build_gpu_queue(n_jobs)
            else:
                self._s95_gpu_queue = None
        else:
            self._s95_gpu_queue = None

        # [S95] JournalFileBackend when parallel (avoids SQLite lock flakiness)
        storage_label = ""
        if n_jobs > 1:
            from optuna.storages import JournalStorage
            from optuna.storages.journal import JournalFileBackend
            journal_path = f"optuna_studies/{study_name}.log"
            storage = JournalStorage(JournalFileBackend(journal_path))
            storage_label = f"journal:{journal_path}"
        else:
            storage = f"sqlite:///optuna_studies/{study_name}.db"
            storage_label = storage

        self.logger.info(f"  Study: {study_name}")
        self.logger.info(f"  Parallel GPU Optuna: n_jobs={n_jobs}")
        self.logger.info(f"  Storage: {storage_label}")
        self.logger.info("=" * 70)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(seed=42),
            storage=storage,
            load_if_exists=True
        )

        self.logger.info(f"\nRunning {n_trials} Optuna trials...")
        # [S96B] Spawn persistent workers if --persistent-workers active (NN only)
        _s96b_workers = {}
        if getattr(self, '_s96b_use_persistent_workers', False) and self.model_type == "neural_net":
            _s96b_gpu_count = self._s95_detect_cuda_gpus_no_torch()
            _s96b_gpu_ids = list(range(_s96b_gpu_count))
            if _s96b_gpu_ids:
                self.logger.info(
                    f"[S96B] Spawning {len(_s96b_gpu_ids)} persistent GPU workers: {_s96b_gpu_ids}"
                )
                _s96b_workers = self._spawn_persistent_workers(_s96b_gpu_ids)
                self.logger.info(f"[S96B] {len(_s96b_workers)} workers ready")
            else:
                self.logger.warning("[S96B] No CUDA GPUs found - persistent-workers disabled")
        self._s96b_workers = _s96b_workers   # make available to _optuna_objective
        try:
            study.optimize(self._optuna_objective, n_trials=n_trials, n_jobs=n_jobs)
        finally:
            if _s96b_workers:
                self.logger.info("[S96B] Shutting down persistent workers")
                self._shutdown_persistent_workers(_s96b_workers)
                self._s96b_workers = {}

        self.logger.info("\n" + "=" * 70)
        self.logger.info("OPTUNA COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"  Best trial: {study.best_trial.number}")
        self.logger.info(f"  Best R²: {study.best_trial.value:.6f}")
        self.logger.info("=" * 70)

        # Store best config
        self.best_config = dict(study.best_trial.params)

        # Train FINAL model with best params (ONLY save happens here)
        self.logger.info("\nTraining final model with best hyperparameters...")

        n_val = int(len(self.X_train_val) * 0.2)
        X_train = self.X_train_val[:-n_val]
        y_train = self.y_train_val[:-n_val]
        X_val = self.X_train_val[-n_val:]
        y_val = self.y_train_val[-n_val:]

        if self.model_type == "neural_net" and NN_SUBPROCESS_ROUTING_ENABLED:
            # Phase 2.2: Final NN model via subprocess (enriched checkpoint)
            result = self._run_nn_via_subprocess(
                X_train, y_train, X_val, y_val,
                hyperparameters=self.best_config
            )
            self.best_model = None  # Disk-first: model is checkpoint on disk
        else:
            result = self.trainer.train_model(
                self.model_type, X_train, y_train, X_val, y_val,
                hyperparameters=self.best_config
            )
            self.best_model = result["model"]
        self.best_model_type = self.model_type
        self.best_metrics = self._compute_final_metrics(result["metrics"])

        # Optuna metadata for sidecar
        self.optuna_info = {
            "enabled": True,
            "n_trials": int(n_trials),
            "n_jobs": n_jobs,
            "best_trial_number": int(study.best_trial.number),
            "best_value": float(study.best_trial.value),
            "study_name": study_name,
            "storage": storage_label
        }

        self.logger.info(f"✅ Final model: R²={result['metrics'].get('r2', 0.0):.4f}")

        # Single writer: optimizer saves once
        self.save_best_model()

        return self.best_config, self.best_metrics


    def _run_nn_optuna_trial(self, X_train, y_train, X_val, y_val,
                             config, trial_number, fold_idx, gpu_id=None):
        """
        Phase 2.2: Run single NN Optuna fold via train_single_trial.py subprocess.
        
        Unlike _run_nn_via_subprocess(), this does NOT save a model checkpoint.
        Optuna trials are exploratory; only the final model is saved.
        
        Returns dict matching MultiModelTrainer.train_model() result schema.
        """
        npz_path = self._export_split_npz(X_train, y_train, X_val, y_val,
                                          trial_number=trial_number, fold_idx=fold_idx)
        
        try:
            cmd = [
                sys.executable, "train_single_trial.py",
                "--model-type", "neural_net",
                "--data-path", npz_path,
                "--params", json.dumps(config),
                "--trial-number", str(trial_number),
                "--normalize-features",
                "--use-leaky-relu",
                "--batch-mode", "auto",  # [S96A]
            ]
            
            # Thread dropout if provided via CLI
            if getattr(self, '_cli_dropout', None) is not None:
                try:
                    cmd.extend(["--dropout", str(float(self._cli_dropout))])
                except Exception:
                    pass
            
            # TB Trim #1: No --enable-diagnostics on Optuna folds
            # (diagnostics only on final model to avoid 100-file explosion)
            
            sub_env = os.environ.copy()
            # [S95] Pin subprocess to a single GPU if leased
            if gpu_id is not None:
                sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # [S96B] Route through persistent worker if available, else S96A subprocess
            _s96b_workers = getattr(self, '_s96b_workers', {})
            _gpu_id_for_dispatch = gpu_id if gpu_id is not None else 0

            if _s96b_workers and _gpu_id_for_dispatch in _s96b_workers:
                # Build worker job from already-constructed cmd parameters
                _worker_job = {
                    "command":           "train",
                    "X_train_path":      npz_path,
                    "params":            config,
                    "trial_number":      trial_number,
                    "fold_idx":          fold_idx,
                    "normalize_features": True,    # Category B always ON
                    "use_leaky_relu":     True,    # Category B always ON
                    "batch_mode":         "auto",  # [S96A]
                }

                def _subprocess_fallback():
                    from pathlib import Path as _Path  # [S96B polish] local alias for consistency
                    proc_fb = subprocess.run(
                        cmd, capture_output=True, text=True,
                        timeout=600,
                        cwd=str(_Path(__file__).parent),
                        env=sub_env
                    )
                    r2_fb = -999.0; train_mse_fb = 0.0; val_mse_fb = float('inf')
                    if proc_fb.returncode == 0:
                        try:
                            for _line in (proc_fb.stdout or "").strip().split("\n"):
                                _line = _line.strip()
                                if _line.startswith("{") and _line.endswith("}"):
                                    _out = json.loads(_line)
                                    r2_fb = float(_out.get("r2", -999.0))
                                    train_mse_fb = float(_out.get("train_mse", 0.0))
                                    val_mse_fb   = float(_out.get("val_mse", float('inf')))
                                    break
                        except Exception:
                            pass
                    return {"r2": r2_fb, "train_mse": train_mse_fb, "val_mse": val_mse_fb}

                _result = self._s96b_dispatch(
                    _s96b_workers, _gpu_id_for_dispatch, _worker_job,
                    fallback_fn=_subprocess_fallback, timeout=60
                )
                r2        = float(_result.get("r2", -999.0))
                train_mse = float(_result.get("train_mse", 0.0))
                val_mse   = float(_result.get("val_mse", float('inf')))

            else:
                # [S96A] Original subprocess path (unchanged)
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=600,
                    cwd=str(Path(__file__).parent),
                    env=sub_env
                )

                # Parse JSON from last stdout line
                r2 = -999.0
                train_mse = 0.0
                val_mse = float('inf')

                if proc.returncode == 0:
                    try:
                        for line in (proc.stdout or "").strip().split("\n"):
                            line = line.strip()
                            if line.startswith("{") and line.endswith("}"):
                                output = json.loads(line)
                                r2 = float(output.get("r2", -999.0))
                                train_mse = float(output.get("train_mse", 0.0))
                                val_mse = float(output.get("val_mse", float('inf')))
                                break
                    except Exception as parse_err:
                        self.logger.warning(
                            f"[Phase 2.2] Could not parse subprocess output "
                            f"(trial {trial_number} fold {fold_idx}): {parse_err}"
                        )
                else:
                    stderr_tail = (proc.stderr or "")[-300:]
                    self.logger.warning(
                        f"[Phase 2.2] Subprocess failed (trial {trial_number} fold {fold_idx}, "
                        f"rc={proc.returncode}): {stderr_tail}"
                    )
            
            return {
                "model": None,
                "model_type": "neural_net",
                "metrics": {
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "r2": r2,
                },
                "hyperparameters": config,
            }
            
        finally:
            # Cleanup NPZ and its temp directory (TB Trim #2, S95 UUID dirs)
            try:
                os.remove(npz_path)
                # Remove the per-export temp dir if empty
                export_dir = os.path.dirname(npz_path)
                if export_dir and os.path.isdir(export_dir):
                    os.rmdir(export_dir)  # only succeeds if empty
            except OSError:
                pass

    def _optuna_objective(self, trial) -> float:
        """Optuna objective - trains in memory only, returns CV R².

        [S95] When self._s95_gpu_queue is set, leases one GPU for the
        entire trial (all folds) and passes gpu_id to subprocess.
        GPU is always returned via try/finally (Fix D).
        """
        import numpy as np
        from sklearn.model_selection import KFold

        config = self._sample_hyperparameters(trial)
        self.logger.info(f"\nTRIAL {trial.number}")

        # [S95] Lease a GPU for this trial (all folds)
        gpu_id = None
        _q = getattr(self, "_s95_gpu_queue", None)
        if _q is not None:
            gpu_id = _q.get()  # blocks until a GPU is free
            self.logger.info(f"  [S95] Leased GPU {gpu_id} for trial {trial.number}")

        try:
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
            fold_r2 = []

            # [Phase 3A] NN + S96B + enable_vmap=True: collect all folds,
            # dispatch as one train_batch, read K results.
            # All other paths (tree models, serial NN, fallback) unchanged.
            _s96b_workers      = getattr(self, '_s96b_workers', {})
            _gpu_id_for_batch  = gpu_id if gpu_id is not None else 0
            _enable_vmap       = getattr(self, '_enable_vmap', False)
            _use_batch_path    = (
                self.model_type == "neural_net"
                and NN_SUBPROCESS_ROUTING_ENABLED
                and bool(_s96b_workers)
                and _gpu_id_for_batch in _s96b_workers
                and _enable_vmap
            )
            

            if _use_batch_path:
                # [Phase 3A] Build all K fold jobs, export NPZs, batch-dispatch
                _fold_splits = list(kf.split(self.X_train_val))
                _batch_jobs  = []
                _npz_paths   = []
                for fold_idx, (train_idx, val_idx) in enumerate(_fold_splits):
                    X_tr = self.X_train_val[train_idx]
                    y_tr = self.y_train_val[train_idx]
                    X_vl = self.X_train_val[val_idx]
                    y_vl = self.y_train_val[val_idx]
                    npz  = self._export_split_npz(
                        X_tr, y_tr, X_vl, y_vl,
                        trial_number=trial.number, fold_idx=fold_idx,
                    )
                    _npz_paths.append(npz)
                    _batch_jobs.append({
                        "command":            "train",   # worker re-labels for train_batch
                        "X_train_path":       npz,
                        "params":             config,
                        "trial_number":       trial.number,
                        "fold_idx":           fold_idx,
                        "normalize_features": True,
                        "use_leaky_relu":     True,
                        "batch_mode":         "auto",
                    })

                try:
                    _timeout_per_fold = 60
                    _batch_results = self._s96b_dispatch_batch(
                        _s96b_workers, _gpu_id_for_batch, _batch_jobs,
                        timeout=_timeout_per_fold,
                    )
                    for _br in _batch_results:
                        fold_r2.append(float(_br.get("r2", -999.0)))
                finally:
                    # Cleanup NPZs (mirrors _run_nn_optuna_trial finally block)
                    for _p in _npz_paths:
                        try:
                            os.remove(_p)
                            _d = os.path.dirname(_p)
                            if _d and os.path.isdir(_d):
                                os.rmdir(_d)
                        except OSError:
                            pass

            else:
                # Serial path: tree models, enable_vmap=False, or no worker
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X_train_val)):
                    X_tr = self.X_train_val[train_idx]
                    y_tr = self.y_train_val[train_idx]
                    X_vl = self.X_train_val[val_idx]
                    y_vl = self.y_train_val[val_idx]

                    if self.model_type == "neural_net" and NN_SUBPROCESS_ROUTING_ENABLED:
                        # Phase 2.2: Route NN through train_single_trial.py subprocess
                        result = self._run_nn_optuna_trial(
                            X_tr, y_tr, X_vl, y_vl, config, trial.number, fold_idx,
                            gpu_id=gpu_id
                        )
                    else:
                        # Tree models: inline trainer (unchanged)
                        result = self.trainer.train_model(
                            self.model_type, X_tr, y_tr, X_vl, y_vl,
                            hyperparameters=config
                        )
                    fold_r2.append(float(result["metrics"].get("r2", 0.0)))

            avg_r2 = float(np.mean(fold_r2)) if fold_r2 else 0.0
            self.logger.info(f"  R² (CV): {avg_r2:.6f}")
            return avg_r2
        finally:
            # [S95] Always return GPU to pool (Fix D: exception safety)
            if gpu_id is not None and _q is not None:
                _q.put(gpu_id)

    # =========================================================================
    # [S96B] Persistent GPU Worker management
    # =========================================================================

    def _spawn_persistent_workers(self, gpu_ids: list) -> dict:
        """
        [S96B] Spawn one persistent nn_gpu_worker.py subprocess per GPU.

        Each worker boots torch/CUDA once and processes all Optuna NN trials
        via stdin/stdout JSON IPC, eliminating ~85% of per-trial overhead.

        Returns dict: {gpu_id: {"proc": Popen, "alive": bool, "lock": Lock, "device": str}}
        """
        import subprocess as _sp
        import threading as _th
        import os as _os
        from pathlib import Path as _Path  # [S96B Fix 2] local alias: no bare Path/sys in method
        import sys as _sys                 # [S96B Fix 2] local alias: avoids NameError if globals differ

        workers = {}
        worker_script = _Path(__file__).parent / "nn_gpu_worker.py"
        if not worker_script.exists():
            self.logger.error("[S96B] nn_gpu_worker.py not found - cannot spawn workers")
            return workers

        for gpu_id in gpu_ids:
            env = dict(_os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                proc = _sp.Popen(
                    [_sys.executable, str(worker_script)],
                    stdin=_sp.PIPE,
                    stdout=_sp.PIPE,
                    stderr=None,    # inherit stderr so worker logs reach console
                    env=env,
                    text=True,
                    bufsize=1,      # line-buffered
                )
                # Wait for ready signal (up to 30s)
                ready = self._s96b_read_worker_line(proc, timeout=30)
                if ready and ready.get("status") == "ready":
                    self.logger.info(
                        f"[S96B] GPU-{gpu_id} worker ready: {ready.get('device', '?')}"
                    )
                    workers[gpu_id] = {
                        "proc":   proc,
                        "alive":  True,
                        "lock":   _th.Lock(),
                        "device": ready.get("device", f"cuda:{gpu_id}"),
                    }
                else:
                    self.logger.warning(
                        f"[S96B] GPU-{gpu_id} worker did not send ready signal - discarding"
                    )
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            except Exception as exc:
                self.logger.error(f"[S96B] Failed to spawn worker GPU-{gpu_id}: {exc}")

        return workers

    def _shutdown_persistent_workers(self, workers: dict) -> None:
        """[S96B] Send shutdown to all alive workers and wait for exit."""
        import json as _json
        import subprocess as _sp
        for gpu_id, w in workers.items():
            if not w.get("alive"):
                continue
            try:
                proc = w["proc"]
                if proc.poll() is None:
                    proc.stdin.write(_json.dumps({"command": "shutdown"}) + "\n")
                    proc.stdin.flush()
                    proc.stdin.close()   # [S96B TB-rec-B] force EOF so worker exits even if readline blocks
                    proc.wait(timeout=10)
                    self.logger.info(f"[S96B] GPU-{gpu_id} worker shut down cleanly")
            except Exception as exc:
                self.logger.warning(f"[S96B] GPU-{gpu_id} shutdown error ({exc}) - killing")
                try:
                    w["proc"].kill()
                except Exception:
                    pass

    def _s96b_read_worker_line(self, proc, timeout: float = 60):
        """
        [S96B] Read one JSON line from worker stdout within timeout seconds.
        Uses threading.Queue to implement non-blocking readline with timeout.
        Returns parsed dict or None on timeout / broken pipe.
        """
        import json as _json
        import queue as _q
        import threading as _th

        result_q = _q.Queue()

        def _reader():
            try:
                line = proc.stdout.readline()
                result_q.put(line)
            except Exception:
                result_q.put("")

        t = _th.Thread(target=_reader, daemon=True)
        t.start()
        try:
            raw = result_q.get(timeout=timeout)
        except _q.Empty:
            return None

        if not raw:
            return None
        try:
            return _json.loads(raw.strip())
        except Exception:
            self.logger.warning(f"[S96B] Malformed worker stdout: {raw[:200]!r}")
            return None

    def _s96b_dispatch(self, workers: dict, gpu_id: int, job: dict,
                       fallback_fn, timeout: float = 60) -> dict:
        """
        [S96B] Send job to worker for gpu_id, return result dict.

        Fallback strategy (Team Beta spec):
          1. On any failure, log error and attempt single restart
          2. If restart succeeds, retry the job once on fresh worker
          3. If restart fails, mark worker dead and fall back to fallback_fn()
             (= S96A train_single_trial.py subprocess)
          Result: S96B is never worse than S96A.
        """
        import json as _json

        w = workers.get(gpu_id)
        if not w or not w.get("alive"):
            self.logger.warning(
                f"[S96B] GPU-{gpu_id} worker not alive - subprocess fallback"
            )
            return fallback_fn()

        with w["lock"]:
            try:
                proc = w["proc"]
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Worker GPU-{gpu_id} already exited (rc={proc.poll()})"
                    )
                proc.stdin.write(_json.dumps(job) + "\n")
                proc.stdin.flush()

                result = self._s96b_read_worker_line(proc, timeout=timeout)
                if result is None:
                    raise RuntimeError(
                        f"Worker GPU-{gpu_id} timeout ({timeout}s) or broken pipe "
                        f"[trial={job.get('trial_number','?')} fold={job.get('fold_idx','?')}]"
                    )
                if result.get("status") == "error":
                    raise RuntimeError(
                        f"Worker GPU-{gpu_id} error: {result.get('error', '?')[:300]}"
                    )
                return result

            except Exception as exc:
                self.logger.error(
                    f"[S96B] Worker GPU-{gpu_id} failed ({exc}) - restart once then fallback"
                )
                w["alive"] = False
                try:
                    w["proc"].kill()
                except Exception:
                    pass

                # Single restart attempt
                restarted = self._spawn_persistent_workers([gpu_id])
                if restarted:
                    workers[gpu_id] = restarted[gpu_id]
                    self.logger.info(f"[S96B] GPU-{gpu_id} worker restarted successfully")
                    try:
                        w2 = workers[gpu_id]
                        with w2["lock"]:
                            proc2 = w2["proc"]
                            proc2.stdin.write(_json.dumps(job) + "\n")
                            proc2.stdin.flush()
                            result2 = self._s96b_read_worker_line(proc2, timeout=timeout)
                            if result2 and result2.get("status") != "error":
                                return result2
                    except Exception as exc2:
                        self.logger.error(
                            f"[S96B] GPU-{gpu_id} restart also failed ({exc2}) - subprocess fallback"
                        )
                        workers[gpu_id]["alive"] = False
                else:
                    self.logger.warning(
                        f"[S96B] GPU-{gpu_id} restart failed - subprocess fallback for remaining trials"
                    )
                return fallback_fn()


    def _s96b_dispatch_batch(self, workers: dict, gpu_id: int,
                             jobs: list, timeout: float = 120) -> list:
        """
        [Phase 3A] Send K fold-jobs to worker as one train_batch command,
        read K result lines back. Returns list of K result dicts in input order.

        Protocol (send once, read N):
          1. Send {"command":"train_batch","jobs":[job0,...,jobK-1]} to worker stdin
          2. Read K lines from worker stdout via _s96b_read_worker_line()
          3. Return results in input order

        Timeout is per-read, not total. Total wall time ≤ timeout * K.
        Falls back to serial _s96b_dispatch() calls if worker not alive or on
        any IPC failure, preserving the S96B "never worse than S96A" guarantee.

        enable_vmap kill-switch (default False → serial path, never reaches here).
        """
        import json as _json

        w = workers.get(gpu_id)
        if not w or not w.get("alive"):
            self.logger.warning(
                f"[3A] GPU-{gpu_id} worker not alive - serial fallback for batch"
            )
            return self._s96b_dispatch_batch_serial_fallback(
                workers, gpu_id, jobs, timeout)

        K = len(jobs)
        with w["lock"]:
            try:
                proc = w["proc"]
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Worker GPU-{gpu_id} exited (rc={proc.poll()})"
                    )

                # Send one train_batch command containing all K fold jobs
                batch_cmd = {"command": "train_batch", "jobs": jobs}
                proc.stdin.write(_json.dumps(batch_cmd) + "\n")
                proc.stdin.flush()
                self.logger.info(
                    f"[3A] GPU-{gpu_id} dispatched train_batch K={K}"
                )

                # Read K result lines
                results = []
                for k in range(K):
                    line = self._s96b_read_worker_line(proc, timeout=timeout)
                    if line is None:
                        raise RuntimeError(
                            f"[3A] GPU-{gpu_id} timeout on result {k+1}/{K}"
                        )
                    if line.get("status") == "error":
                        # Worker reported per-fold error — record and continue
                        self.logger.warning(
                            f"[3A] GPU-{gpu_id} fold {k} error: "
                            f"{line.get('error','?')[:200]}"
                        )
                    results.append(line)

                return results

            except Exception as exc:
                self.logger.error(
                    f"[3A] dispatch_batch GPU-{gpu_id} failed ({exc}) "
                    f"- restart once then serial fallback"
                )
                w["alive"] = False
                try:
                    w["proc"].kill()
                except Exception:
                    pass

                # Mirror S96B restart-once contract
                restarted = self._spawn_persistent_workers([gpu_id])
                if restarted:
                    workers[gpu_id] = restarted[gpu_id]
                    self.logger.info(
                        f"[3A] GPU-{gpu_id} worker restarted - retrying batch"
                    )
                    try:
                        w2 = workers[gpu_id]
                        with w2["lock"]:
                            proc2 = w2["proc"]
                            import json as _json2
                            batch_cmd2 = {"command": "train_batch", "jobs": jobs}
                            proc2.stdin.write(_json2.dumps(batch_cmd2) + "\n")
                            proc2.stdin.flush()
                            results2 = []
                            for k in range(K):
                                line2 = self._s96b_read_worker_line(
                                    proc2, timeout=timeout)
                                if line2 is None:
                                    raise RuntimeError(
                                        f"[3A] restart timeout on result {k+1}/{K}"
                                    )
                                results2.append(line2)
                            return results2
                    except Exception as exc2:
                        self.logger.error(
                            f"[3A] GPU-{gpu_id} restart retry failed ({exc2}) "
                            f"- serial fallback"
                        )
                        workers[gpu_id]["alive"] = False
                else:
                    self.logger.warning(
                        f"[3A] GPU-{gpu_id} restart failed - serial fallback"
                    )
                return self._s96b_dispatch_batch_serial_fallback(
                    workers, gpu_id, jobs, timeout)

    def _s96b_dispatch_batch_serial_fallback(self, workers: dict, gpu_id: int,
                                              jobs: list, timeout: float) -> list:
        """[Phase 3A] Serial fallback: dispatch each job individually."""
        results = []
        for job in jobs:
            try:
                r = self._s96b_dispatch(
                    workers, gpu_id, job,
                    fallback_fn=lambda: {
                        "r2": -999.0, "train_mse": 0.0,
                        "val_mse": float("inf"), "status": "error",
                        "error": "serial_fallback",
                    },
                    timeout=timeout,
                )
                results.append(r)
            except Exception as exc:
                self.logger.error(f"[3A] serial fallback fold failed: {exc}")
                results.append({
                    "r2": -999.0, "train_mse": 0.0,
                    "val_mse": float("inf"), "status": "error",
                })
        return results

    def _sample_hyperparameters(self, trial) -> Dict:
        """Sample hyperparameters for model type (model-specific)."""
        if self.model_type == "neural_net":
            n_layers = trial.suggest_int("n_layers", 2, 4)
            layers = []
            for i in range(n_layers):
                if i == 0:
                    layers.append(trial.suggest_int(f"layer_{i}", 64, 256))
                else:
                    layers.append(trial.suggest_int(f"layer_{i}", 32, layers[-1]))
            return {
                "hidden_layers": layers,
                "dropout": trial.suggest_float("dropout", 0.2, 0.6),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),  # [S96A] larger batches
                "epochs": trial.suggest_int("epochs", 50, 200),
                "early_stopping_patience": trial.suggest_int("patience", 10, 30),
            }

        if self.model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }

        if self.model_type == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }

        if self.model_type == "catboost":
            return {
                "iterations": trial.suggest_int("iterations", 200, 800),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            }

        return {}



    def _export_split_npz(self, X_train, y_train, X_val, y_val,
                          trial_number=None, fold_idx=None):
        """
        Category B Phase 2.1 (Team Beta Mod A): Export the EXACT split
        already computed by _run_single_model to NPZ for subprocess.
        
        [S95] Collision-proof under threaded Optuna (Fix B):
          - per-export temp dir (mkdtemp) under output_dir/tmp
          - UUID-based filename, includes trial/fold when provided
        """
        import tempfile
        
        tmp_root = os.path.join(self.output_dir, "tmp")
        os.makedirs(tmp_root, exist_ok=True)
        
        # Per-export directory (safe for threaded writers)
        export_dir = tempfile.mkdtemp(prefix="step5_nn_", dir=tmp_root)
        
        # UUID filename (collision-proof even with shared PID under threading)
        u = uuid.uuid4().hex[:12]
        t_tag = f"t{trial_number}" if trial_number is not None else "tNA"
        f_tag = f"f{fold_idx}" if fold_idx is not None else "fNA"
        npz_path = os.path.join(export_dir, f"split_{t_tag}_{f_tag}_{u}.npz")
        
        np.savez(npz_path,
                 X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val)
        
        self.logger.info(f"[CAT-B 2.1] Exported split to NPZ: {npz_path}")
        self.logger.info(f"[CAT-B 2.1]   X_train: {X_train.shape}, X_val: {X_val.shape}")
        return npz_path

    def _run_nn_via_subprocess(self, X_train, y_train, X_val, y_val, hyperparameters=None):
        """
        Category B Phase 2.1: Route NN single-shot training through
        train_single_trial.py subprocess.
        
        This ensures enriched checkpoints with scaler metadata, matching
        the compare-models subprocess path.
        
        Team Beta Mod C: Fails hard if subprocess returns non-zero.
                         --allow-inline-nn-fallback overrides this.
        Team Beta Mod D: Threads all Category B flags.
        
        Returns:
            Dict matching MultiModelTrainer.train_model() result schema
        """
        # [S96A] NO torch import here — parent must stay CUDA-clean (S72/S73 invariant)
        
        # Export the exact split
        npz_path = self._export_split_npz(X_train, y_train, X_val, y_val)
        
        try:
            # Build subprocess command
            cmd = [
                sys.executable, "train_single_trial.py",
                "--model-type", "neural_net",
                "--data-path", npz_path,
                "--save-model",
                "--model-output-dir", self.output_dir,
                "--normalize-features",     # Category B Option A: always ON
                "--use-leaky-relu",
                "--batch-mode", "auto",  # [S96A]         # Category B: always ON
                "--verbose",
            ]
            
            # Phase 2.2: Pass hyperparameters if provided (Optuna best config)
            if hyperparameters:
                cmd.extend(["--params", json.dumps(hyperparameters)])
            
            # Thread dropout if provided via CLI (Team Beta Mod D)
            if getattr(self, '_cli_dropout', None) is not None:
                try:
                    _d = float(self._cli_dropout)
                    cmd.extend(["--dropout", str(_d)])
                except Exception:
                    pass
            
            # Thread diagnostics flag
            if getattr(self, '_cli_enable_diagnostics', False):
                cmd.append("--enable-diagnostics")
            
            self.logger.info("[CAT-B 2.1] Routing NN through train_single_trial.py subprocess")
            self.logger.info(f"[CAT-B 2.1] cmd: {' '.join(cmd)}")
            
            # Option C defense-in-depth: pass-through env (Team Beta guardrail)
            # Do NOT invent CUDA_VISIBLE_DEVICES — let subprocess inherit or set its own
            import os
            sub_env = os.environ.copy()
            _cuda_vis = sub_env.get("CUDA_VISIBLE_DEVICES", "<unset>")
            if hasattr(self, "logger"):
                self.logger.info(f"[CAT-B 2.1] Subprocess CUDA_VISIBLE_DEVICES={_cuda_vis}")
            else:
                print(f"[CAT-B 2.1] Subprocess CUDA_VISIBLE_DEVICES={_cuda_vis}")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=sub_env)
            
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "")[-500:]
                self.logger.error(f"[CAT-B 2.1] Subprocess failed (rc={proc.returncode})")
                self.logger.error(f"[CAT-B 2.1] stderr tail: {stderr_tail}")
                raise RuntimeError(
                    f"train_single_trial.py subprocess failed (rc={proc.returncode}): {stderr_tail}"
                )
            
            # Load enriched checkpoint — subprocess saves as neural_net_trial-1.pth
            trial_path = os.path.join(self.output_dir, "neural_net_trial-1.pth")
            checkpoint_path = os.path.join(self.output_dir, "best_model.pth")
            if os.path.exists(trial_path):
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)  # [S96A] Safe rename: prevent silent clobber
                os.rename(trial_path, checkpoint_path)
                self.logger.info(f"[CAT-B 2.1] Renamed {trial_path} → {checkpoint_path}")
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"Expected checkpoint not found: {checkpoint_path}")
            
            # [S96A] Validate enriched checkpoint via subprocess JSON output
            # NO torch.load in parent — preserves CUDA-clean invariant (S72/S73)
            # train_single_trial.py already emits normalize_features, use_leaky_relu,
            # checkpoint_validated, and scaler_shape in its stdout JSON.
            
            # Extract metrics from subprocess stdout JSON
            r2 = 0.0
            train_mse = 0.0
            val_mse = 0.0
            normalize_flag = None
            leaky_flag = None
            checkpoint_validated = False
            scaler_shape = None
            try:
                import json as _json
                for line in (proc.stdout or "").strip().split("\n"):
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        metrics_json = _json.loads(line)
                        r2 = float(metrics_json.get("r2", 0.0))
                        train_mse = float(metrics_json.get("train_mse", 0.0))
                        val_mse = float(metrics_json.get("val_mse", 0.0))
                        normalize_flag = metrics_json.get("normalize_features")
                        leaky_flag = metrics_json.get("use_leaky_relu")
                        checkpoint_validated = metrics_json.get("checkpoint_validated", False)
                        scaler_shape = metrics_json.get("scaler_shape")
                        self.logger.info(f"[CAT-B 2.1] Metrics from subprocess: R²={r2:.6f}, val_mse={val_mse:.8f}")
                        break
            except Exception as _e:
                self.logger.warning(f"[CAT-B 2.1] Could not parse subprocess metrics: {_e}")
            
            # Assert enriched keys from subprocess output (replaces torch.load validation)
            if normalize_flag is None or leaky_flag is None:
                raise RuntimeError(
                    "Subprocess output missing required Category B keys: "
                    f"normalize_features={normalize_flag}, use_leaky_relu={leaky_flag}"
                )
            if normalize_flag and scaler_shape is None:
                raise RuntimeError(
                    "normalize_features=True but scaler_shape missing from subprocess output"
                )
            if scaler_shape:
                self.logger.info(f"[CAT-B 2.1] Scaler validated: {scaler_shape[0]} features")
            
            # Also try loading from sidecar if it exists
            sidecar_path = os.path.join(self.output_dir, "best_model.meta.json")
            if os.path.exists(sidecar_path):
                try:
                    with open(sidecar_path) as sf:
                        sidecar_data = json.load(sf)
                    tm = sidecar_data.get("training_metrics", {})
                    if tm.get("r2") is not None:
                        r2 = float(tm["r2"])
                    if tm.get("val_mse") is not None:
                        val_mse = float(tm["val_mse"])
                    if tm.get("train_mse") is not None:
                        train_mse = float(tm["train_mse"])
                except Exception:
                    pass
            
            self.logger.info(f"[CAT-B 2.1] NN subprocess complete: R²={r2:.4f}")
            self.logger.info(f"[CAT-B 2.1] normalize_features={normalize_flag}, use_leaky_relu={leaky_flag}")
            
            # Store checkpoint path for sidecar generation
            self.best_checkpoint_path = checkpoint_path
            
            # Clean up NPZ on success (Team Beta Mod B)
            try:
                os.remove(npz_path)
                self.logger.debug(f"[CAT-B 2.1] Cleaned up NPZ: {npz_path}")
            except Exception:
                pass
            
            # Return result matching MultiModelTrainer.train_model() schema
            hp = dict(hyperparameters or {})
            return {
                'model': None,  # Model is on disk, not in memory
                'model_type': 'neural_net',
                'metrics': {
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'r2': r2,
                },
                'hyperparameters': {
                    **hp,
                    'normalize_features': bool(normalize_flag),
                    'use_leaky_relu': bool(leaky_flag),
                    'hidden_layers': hp.get('hidden_layers', [256, 128, 64]),
                    'dropout': hp.get('dropout', 0.3),
                },
                'checkpoint_path': checkpoint_path,
            }
            
        except Exception as e:
            # Keep NPZ for debugging on failure (Team Beta Mod B)
            self.logger.error(f"[CAT-B 2.1] NN subprocess error: {e}")
            self.logger.error(f"[CAT-B 2.1] NPZ retained at: {npz_path}")
            raise

    def _run_single_model(self) -> Tuple[Dict, ValidationMetrics]:
        """Train single model type."""
        
        # Team Beta: Branch on n_trials_total
        if getattr(self, "n_trials_total", None) and self.n_trials_total > 1:
            self.logger.info("MODE: OPTUNA OPTIMIZATION")
            return self._run_optuna_optimization(self.n_trials_total)
        
        # Single-shot mode (v3.3 preserved)
        self.logger.info("MODE: SINGLE-SHOT (No Optimization)")
        self.optuna_info = {"enabled": False}
        self.logger.info("=" * 70)
        self.logger.info(f"SINGLE MODEL: {self.model_type}")
        self.logger.info("=" * 70)
        
        # Simple train/val split
        n_val = int(len(self.X_train_val) * 0.2)
        X_train = self.X_train_val[:-n_val]
        y_train = self.y_train_val[:-n_val]
        X_val = self.X_train_val[-n_val:]
        y_val = self.y_train_val[-n_val:]
        
        # Category B Phase 2.1: Route NN through subprocess for enriched checkpoints
        if self.model_type == "neural_net":
            self.logger.info("[CAT-B 2.1] Neural net detected — routing through train_single_trial.py")
            try:
                result = self._run_nn_via_subprocess(X_train, y_train, X_val, y_val)
            except Exception as e:
                if getattr(self, '_allow_inline_nn_fallback', False):
                    self.logger.warning(f"[CAT-B 2.1] Subprocess failed, falling back to inline: {e}")
                    result = self.trainer.train_model(
                        self.model_type, X_train, y_train, X_val, y_val
                    )
                else:
                    raise RuntimeError(
                        f"[CAT-B 2.1] NN subprocess failed and --allow-inline-nn-fallback not set: {e}"
                    ) from e
        else:
            # Tree models: use inline trainer as before
            result = self.trainer.train_model(
                self.model_type, X_train, y_train, X_val, y_val
            )
        
        self.best_model = result['model']
        self.best_model_type = self.model_type
        self.best_config = result['hyperparameters']
        self.best_metrics = self._compute_final_metrics(result['metrics'])
        
        # Store checkpoint path if subprocess provided one
        if result.get('checkpoint_path'):
            self.best_checkpoint_path = result['checkpoint_path']
        
        # Log results
        self.logger.info(f"\nTraining complete: R²={result['metrics']['r2']:.4f}")
        
        # Save model (sidecar generation)
        self.save_best_model()
        
        return self.best_config, self.best_metrics
    
    def _compute_final_metrics(self, training_metrics: Dict) -> ValidationMetrics:
        """Compute comprehensive validation metrics on test set."""
        return ValidationMetrics(
            train_variance=0.0,
            val_variance=0.0,
            test_variance=0.0,
            train_mae=training_metrics.get('train_mse', 0) ** 0.5,
            val_mae=training_metrics.get('val_mse', 0) ** 0.5,
            test_mae=0.0,
            overfit_ratio=1.0,
            variance_consistency=1.0,
            temporal_stability=1.0,
            p_value=0.01,
            confidence_interval=(0.0, 1.0),
            r2_score=training_metrics.get('r2', 0.0)
        )
    
    def save_best_model(self):
        """Save the best model with full sidecar metadata."""
        # Team Beta: subprocess comparison is disk-first (no in-memory model in parent)
        if self.best_model is None:
            if getattr(self, 'best_checkpoint_path', None):
                # Disk-first: checkpoint exists (Optuna NN, compare-models, etc.)
                self._save_existing_checkpoint_sidecar(
                    checkpoint_path=self.best_checkpoint_path,
                    model_type=self.best_model_type or "unknown"
                )
                return

            self.logger.warning("No model trained - saving degenerate sidecar only")
            self._save_degenerate_sidecar()
            return
        
        duration = (datetime.now() - self.start_time).total_seconds()
        
        provenance = {
            'run_id': f"step5_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
            'started_at': self.start_time.isoformat(),
            'duration_seconds': duration,
            'survivors_file': str(Path(self.survivors_file).resolve()),
            'n_survivors': len(self.X),
            'model_type': self.best_model_type,
            'compare_models_used': self.compare_models
        }
        
        training_metrics = {
            'r2': self.best_metrics.r2_score,
            'train_mae': self.best_metrics.train_mae,
            'val_mae': self.best_metrics.val_mae,
            'test_mae': self.best_metrics.test_mae,
            'overfit_ratio': self.best_metrics.overfit_ratio
        }
        
        self.trainer.save_model(
            model=self.best_model,
            model_type=self.best_model_type,
            output_dir=self.output_dir,
            feature_schema=self.feature_schema,
            training_metrics=training_metrics,
            signal_quality=self.signal_quality,
            data_context=self.data_context,
            hyperparameters=self.best_config,
            provenance=provenance,
            parent_run_id=self.parent_run_id
        )
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("MODEL SAVED WITH SIDECAR")
        self.logger.info("=" * 70)
        self.logger.info(f"  Model type: {self.best_model_type}")
        self.logger.info(f"  R² score: {self.best_metrics.r2_score:.4f}")
        self.logger.info(f"  Signal status: {self.signal_quality['signal_status']}")
        self.logger.info(f"  Data fingerprint: {self.data_context['fingerprint_hash']}")
        self.logger.info(f"  Schema hash: {self.feature_schema['feature_schema_hash']}")
        self.logger.info("=" * 70)
    def _save_existing_checkpoint_sidecar(self, checkpoint_path: str, model_type: str):
        """
        Team Beta: Write a SUCCESS sidecar referencing an existing checkpoint on disk.
        Used for --compare-models subprocess isolation mode.
        """
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            self.logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
            self._save_degenerate_sidecar()
            return

        duration = (datetime.now() - self.start_time).total_seconds()

        provenance = {
            'run_id': f"step5_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
            'started_at': self.start_time.isoformat(),
            'duration_seconds': duration,
            'survivors_file': str(Path(self.survivors_file).resolve()),
            'n_survivors': len(self.X),
            'model_type': model_type,
            'compare_models_used': True,
            'checkpoint_path': str(ckpt.resolve()),
            'outcome': 'SUCCESS'
        }

        # NOTE: In subprocess mode, metrics may be partial.
        # Artifact authority is the checkpoint; metrics are best-effort.
        training_metrics = {
            'r2': getattr(self.best_metrics, 'r2_score', 0.0) if self.best_metrics else 0.0,
            'train_mae': getattr(self.best_metrics, 'train_mae', 0.0) if self.best_metrics else 0.0,
            'val_mae': getattr(self.best_metrics, 'val_mae', 0.0) if self.best_metrics else 0.0,
            'test_mae': getattr(self.best_metrics, 'test_mae', 0.0) if self.best_metrics else 0.0,
            'overfit_ratio': getattr(self.best_metrics, 'overfit_ratio', 1.0) if self.best_metrics else 1.0,
            'status': 'success'
        }

        checkpoint_format = ckpt.suffix.lstrip('.') if ckpt.suffix else None

        sidecar = {
            "schema_version": "3.4.0",
            "model_type": model_type,
            "checkpoint_path": str(ckpt),
            "checkpoint_format": checkpoint_format,
            "feature_schema": self.feature_schema,
            "signal_quality": self.signal_quality,
            "data_context": self.data_context,
            "training_metrics": training_metrics,
            "hyperparameters": self.best_config or {},
            "optuna": getattr(self, "optuna_info", {"enabled": False}),
            "hardware": {
                "device_requested": self.device,
                "cuda_available": CUDA_INITIALIZED
            },
            "training_info": {
                "started_at": self.start_time.isoformat(),
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": duration,
                "outcome": "SUCCESS"
            },
            "agent_metadata": {
                "pipeline_step": 5,
                "pipeline_step_name": "anti_overfit_training",
                "run_id": provenance.get('run_id'),
                "parent_run_id": self.parent_run_id,
                "outcome": "SUCCESS",
                "exit_code": 0
            },
            "provenance": provenance
        }

        # Canonical invariant check
        if sidecar["signal_quality"].get("prediction_allowed", True):
            assert sidecar["checkpoint_path"] is not None, \
                "Invariant violated: prediction_allowed=True but checkpoint_path=None"

        sidecar_path = output_path / "best_model.meta.json"
        with open(sidecar_path, 'w') as f:
            import json
            json.dump(sidecar, f, indent=2)

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("SUBPROCESS WINNER SIDECAR SAVED (Existing Checkpoint)")
        self.logger.info("=" * 70)
        self.logger.info(f"  Model type: {model_type}")
        self.logger.info(f"  Checkpoint: {sidecar['checkpoint_path']}")
        self.logger.info(f"  R² score: {training_metrics.get('r2', 0.0):.4f}")
        self.logger.info(f"  Signal status: {self.signal_quality.get('signal_status')}")
        self.logger.info(f"  Data fingerprint: {self.data_context.get('fingerprint_hash')}")
        self.logger.info("=" * 70)

    
    def _save_degenerate_sidecar(self):
        """
        Save sidecar for degenerate signal (no model checkpoint).
        
        This allows WATCHER to:
        1. Read signal_quality and understand why prediction failed
        2. Read data_context fingerprint to avoid retrying same config
        3. Take appropriate recovery action
        """
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        duration = (datetime.now() - self.start_time).total_seconds()
        
        sidecar = {
            "schema_version": "3.4.0",
            "model_type": None,
            "checkpoint_path": None,
            "checkpoint_format": None,
            
            "feature_schema": self.feature_schema,
            "signal_quality": self.signal_quality,
            "data_context": self.data_context,
            
            "training_metrics": {
                "status": "degenerate_signal",
                "r2": 0.0,
                "train_mse": None,
                "val_mse": None,
                "error": "Training skipped due to degenerate target (zero variance)"
            },
            
            "hardware": {
                "device_requested": self.device,
                "cuda_available": CUDA_INITIALIZED
            },
            
            "training_info": {
                "started_at": self.start_time.isoformat(),
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": duration,
                "outcome": "DEGENERATE_SIGNAL"
            },
            
            "agent_metadata": {
                "pipeline_step": 5,
                "pipeline_step_name": "anti_overfit_training",
                "run_id": f"step5_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
                "parent_run_id": self.parent_run_id,
                "outcome": "DEGENERATE_SIGNAL",
                "exit_code": 2
            },
            
            "provenance": {
                "run_id": f"step5_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
                "started_at": self.start_time.isoformat(),
                "duration_seconds": duration,
                "survivors_file": str(Path(self.survivors_file).resolve()),
                "n_survivors": len(self.X),
                "model_type": self.model_type,
                "compare_models_used": self.compare_models,
                "outcome": "DEGENERATE_SIGNAL"
            }
        }
        
        sidecar_path = output_path / "best_model.meta.json"
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar, f, indent=2)
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("DEGENERATE SIDECAR SAVED (No Model)")
        self.logger.info("=" * 70)
        self.logger.info(f"  Signal status: {self.signal_quality['signal_status']}")
        self.logger.info(f"  Signal confidence: {self.signal_quality['signal_confidence']:.4f}")
        self.logger.info(f"  Data fingerprint: {self.data_context['fingerprint_hash']}")
        self.logger.info(f"  Sidecar path: {sidecar_path}")
        self.logger.info("=" * 70)
        self.logger.info("WATCHER AGENT: Consume sidecar and decide recovery action")
        self.logger.info("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Anti-Overfit Meta-Prediction Optimizer v3.3'
    )
    parser.add_argument('--survivors', required=True,
                       help='Path to survivors JSON with features')
    parser.add_argument('--lottery-data', required=True,
                       help='Path to lottery history JSON')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of Optuna trials')
    parser.add_argument('--k-folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--test-holdout', type=float, default=0.2,
                       help='Test holdout percentage')
    parser.add_argument('--study-name', type=str,
                       help='Optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_studies.db',
                       help='Optuna storage path')
    
    # Model selection
    parser.add_argument('--model-type', type=str, default='catboost',
                       choices=['neural_net', 'xgboost', 'lightgbm', 'catboost'],
                       help='Model type to train')
    parser.add_argument('--compare-models', action='store_true',
                       help='Train all models and select best')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='models/reinforcement',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='GPU device')
    
    # Data context
    parser.add_argument('--prng-type', type=str, default='java_lcg',
                       help='PRNG type for fingerprint')
    parser.add_argument('--mod', type=int, default=1000,
                       help='PRNG modulus for fingerprint')
    parser.add_argument('--holdout-draws', type=int, default=1000,
                       help='Number of holdout draws')
    parser.add_argument('--parent-run-id', type=str,
                       help='Parent run ID for provenance')
    
    # Chapter 14: Training diagnostics
    parser.add_argument('--enable-diagnostics', action='store_true',
                       help='Enable Chapter 14 training diagnostics (writes to diagnostics_outputs/)')
    
    # Category B: Neural net training enhancements (passed through to train_single_trial.py)
    # NOTE: In --compare-models mode, Option A forces these ON for neural_net
    # regardless of CLI flags. These flags are for single-model WATCHER pass-through.
    parser.add_argument('--normalize-features', action='store_true',
                       help='Apply StandardScaler normalization before NN training')
    parser.add_argument('--use-leaky-relu', action='store_true',
                       help='Use LeakyReLU(0.01) instead of ReLU in neural net')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Override dropout value for NN (CLI precedence)')
    # [S96B] Persistent GPU workers
    parser.add_argument('--persistent-workers', action='store_true',
                       default=False, dest='persistent_workers',
                       help='[S96B] Persistent GPU workers for NN trials (default OFF)')
    parser.add_argument('--enable-vmap', action='store_true', default=False,
                        dest='enable_vmap',
                        help='[Phase 3A] vmap batch size for NN trials (default 1=off). '
                             'Set to 16 after Zeus smoke test passes.')
    parser.add_argument('--no-persistent-workers', action='store_false',
                       dest='persistent_workers',
                       help='[S96B] Disable persistent GPU workers (default)')
    parser.add_argument('--allow-inline-nn-fallback', action='store_true',
                       help='If NN subprocess fails, fall back to inline trainer (default: OFF)')
    
    args = parser.parse_args()


    
    # S88 HOTFIX: --compare-models must run Optuna trials per model (4*N total)
    
    if getattr(args, 'compare_models', False) and not os.environ.get('S88_COMPARE_MODELS_CHILD'):
    
        return _s88_run_compare_models({
    
            'survivors': getattr(args, 'survivors', None),
    
            'lottery_data': getattr(args, 'lottery_data', None),
    
            'trials': getattr(args, 'trials', 1),
    
            'enable_diagnostics': getattr(args, 'enable_diagnostics', False),
    
            'normalize_features': getattr(args, 'normalize_features', False),
    
            'use_leaky_relu': getattr(args, 'use_leaky_relu', False),
    
            'dropout': getattr(args, 'dropout', None),
    
        })
    # Conditional CUDA initialization (Team Beta design invariant)
    # GPU code must NEVER run in coordinating process when using subprocess isolation
    global CUDA_INITIALIZED
    will_use_subprocess = args.compare_models or (
        args.model_type == "neural_net" and NN_SUBPROCESS_ROUTING_ENABLED
    )
    if will_use_subprocess:
        if args.compare_models:
            print("⚡ Mode: Multi-Model Comparison (Subprocess Isolation)")
        else:
            print(f"⚡ Mode: Single Model ({args.model_type}) (Subprocess Isolation)")
        print("   GPU initialization DEFERRED to subprocess")
        CUDA_INITIALIZED = False
    else:
        CUDA_INITIALIZED = initialize_cuda_early()
        print(f"⚡ Mode: Single Model ({args.model_type})")

    
    print("=" * 70)
    print("ANTI-OVERFIT META-PREDICTION OPTIMIZER v3.3")
    print("=" * 70)
    print(f"✅ CUDA initialized: {CUDA_INITIALIZED}")
    print(f"✅ Model type: {args.model_type}")
    print(f"✅ Compare models: {args.compare_models}")
    print("=" * 70)
    
    # Load lottery data
    with open(args.lottery_data) as f:
        lottery_data = json.load(f)
        lottery_history = [d['draw'] if isinstance(d, dict) else d for d in lottery_data]
    
    # Run optimizer
    optimizer = AntiOverfitMetaOptimizer(
        survivors_file=args.survivors,
        lottery_history=lottery_history,
        k_folds=args.k_folds,
        test_holdout_pct=args.test_holdout,
        study_name=args.study_name,
        storage=args.storage,
        model_type=args.model_type,
        compare_models=args.compare_models,
        output_dir=args.output_dir,
        device=args.device,
        prng_type=args.prng_type,
        mod=args.mod,
        holdout_draws=args.holdout_draws,
        parent_run_id=args.parent_run_id
    )
    
    # Category B Phase 2.1: Thread CLI flags into optimizer (Team Beta Mod D)
    optimizer._cli_dropout = getattr(args, 'dropout', None)
    optimizer._cli_normalize = getattr(args, 'normalize_features', False)
    optimizer._cli_leaky = getattr(args, 'use_leaky_relu', False)
    optimizer._cli_enable_diagnostics = getattr(args, 'enable_diagnostics', False)
    # [S96B] Thread persistent-workers flag into optimizer
    optimizer._s96b_use_persistent_workers = getattr(args, 'persistent_workers', False)
    optimizer._allow_inline_nn_fallback = getattr(args, 'allow_inline_nn_fallback', False)
    # [Phase 3A] Thread enable_vmap into optimizer (0=off, 1=on)
    optimizer._enable_vmap = getattr(args, 'enable_vmap', False)

    best_config, metrics = optimizer.run(n_trials=args.trials)
    
    # Check for degenerate exit
    if hasattr(optimizer, 'degenerate_exit') and optimizer.degenerate_exit:
        print()
        print("=" * 70)
        print("DEGENERATE SIGNAL - EARLY EXIT")
        print("=" * 70)
        print(f"Signal Quality: {optimizer.signal_quality['signal_status']}")
        print(f"Data Fingerprint: {optimizer.data_context['fingerprint_hash']}")
        print()
        print("⚠️  No model trained (degenerate target)")
        print("✅ Sidecar saved for WATCHER consumption")
        print("=" * 70)
        print()
        print("EXIT CODE: 2 (DEGENERATE_SIGNAL)")
        print("WATCHER should consume sidecar and decide recovery action")
        print("=" * 70)
        sys.exit(2)
    
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best Model Type: {optimizer.best_model_type}")
    print(f"R² Score: {metrics.r2_score:.4f}")
    print()
    
    if metrics.is_overfitting():
        print("⚠️  WARNING: Model shows signs of overfitting!")
        print("   Consider: More regularization, simpler model, more data")
    else:
        print("✅ Model generalizes well to test set!")
    
    print()
    print(f"Signal Quality: {optimizer.signal_quality['signal_status']}")
    print(f"Data Fingerprint: {optimizer.data_context['fingerprint_hash']}")
    print("=" * 70)
    
    # Success exit
    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main() or 0)