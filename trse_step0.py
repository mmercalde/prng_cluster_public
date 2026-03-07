#!/usr/bin/env python3
"""
TRSE — Temporal Regime Segmentation Engine (Step 0)
=====================================================
Version: 1.0.0
Session: S121

PURPOSE
-------
Pre-pipeline sidecar that characterises the *current draw regime* before
Step 1 runs.  It does NOT filter seeds or touch survivors; it produces a
JSON context file consumed by Step 1 (window parameter adjustment) and
optionally by Step 5 (regime-conditioned training gate).

DESIGN (approved S119)
-----------------------
Feature set:
  • Entropy drift   — mod8 / mod125 / mod1000 residue entropy per window
  • Digit transition fingerprints — 3 × 10×10 transition matrices (H→H,
    T→T, O→O) per window, flattened to 300 floats
  • Lag structure    — EXCLUDED (autocorrelation probe S119 refuted signal)

Probe params (from S119 machine fingerprint probe):
  window_size = 400  draws
  stride      =  50  draws
  k_clusters  =   5  (silhouette-optimal)

OUTPUT
------
  trse_context.json   (written to cwd, same dir as other pipeline outputs)

  {
    "trse_version": "1.0.0",
    "timestamp": "...",
    "n_draws": 18068,
    "n_windows": 354,
    "window_size": 400,
    "stride": 50,
    "k_clusters": 5,
    "current_regime": 2,          # cluster id of most recent window
    "regime_age": 12,             # consecutive windows in current regime
    "switch_rate": 0.062,         # global fraction of windows that switch
    "silhouette": 0.079,
    "regime_counts": [29,78,93,95,59],
    "regime_entropy_profile": {   # mean entropy features per regime
        "0": {"entropy_mod8": ..., "entropy_mod125": ..., "entropy_mod1000": ...},
        ...
    },
    "current_window_features": {  # raw features of last window
        "entropy_mod8": ...,
        "entropy_mod125": ...,
        "entropy_mod1000": ...,
        "digit_transition_H": [[...], ...],   # 10×10
        "digit_transition_T": [[...], ...],
        "digit_transition_O": [[...], ...]
    },
    "recommended_window_size": 8,  # pass-through to Step 1 (from Optuna best)
    "regime_stable": true          # age >= 3 windows → regime is mature
  }

USAGE
-----
  # Standalone
  python3 trse_step0.py --lottery-data daily3.json --output trse_context.json

  # Inside WATCHER (Step 0)
  PYTHONPATH=. python3 agents/watcher_agent.py \
      --run-pipeline --start-step 0 --end-step 0 \
      --params '{"lottery_data": "daily3.json"}'

  # Force re-run even if context is fresh
  python3 trse_step0.py --lottery-data daily3.json --force

DEPENDENCIES
------------
  numpy, scikit-learn (KMeans, silhouette_score)
  Both already present in torch venv on Zeus.

Author: Team Alpha S121
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__version__ = "1.0.0"

# ── Probe parameters (S119 validated) ────────────────────────────────────────
DEFAULT_WINDOW_SIZE = 400
DEFAULT_STRIDE      = 50
DEFAULT_K_CLUSTERS  = 5
DEFAULT_OUTPUT      = "trse_context.json"

# Regime is considered "stable / mature" if it has been active this many
# consecutive windows or more.
REGIME_STABLE_THRESHOLD = 3


# =============================================================================
# Feature extraction
# =============================================================================

def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy (bits) from a raw count array. Safe for zero counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def extract_entropy_features(window: np.ndarray) -> Dict[str, float]:
    """
    Compute residue entropy for mod8, mod125, mod1000 over a draw window.

    Args:
        window: 1-D integer array of draw values (0–999)

    Returns:
        dict with keys entropy_mod8, entropy_mod125, entropy_mod1000
    """
    feats: Dict[str, float] = {}
    for mod, name in [(8, "entropy_mod8"), (125, "entropy_mod125"), (1000, "entropy_mod1000")]:
        bins = np.bincount(window % mod, minlength=mod).astype(np.float64)
        feats[name] = _entropy(bins)
    return feats


def extract_digit_transition(window: np.ndarray) -> Dict[str, List[List[float]]]:
    """
    Build 10×10 first-order transition matrices for each digit position
    (hundreds H, tens T, ones O) independently.

    Entry [i][j] = fraction of consecutive-draw pairs where digit went i→j.
    Row-normalised so each row sums to 1 (or 0 if digit i never appeared).

    Args:
        window: 1-D integer array of draw values (0–999)

    Returns:
        dict with keys digit_transition_H, digit_transition_T, digit_transition_O
        each a 10×10 list of floats
    """
    if len(window) < 2:
        empty = [[0.0] * 10 for _ in range(10)]
        return {
            "digit_transition_H": empty,
            "digit_transition_T": empty,
            "digit_transition_O": empty,
        }

    hundreds = (window // 100) % 10
    tens     = (window //  10) % 10
    ones     =  window         % 10

    result = {}
    for name, digits in [
        ("digit_transition_H", hundreds),
        ("digit_transition_T", tens),
        ("digit_transition_O", ones),
    ]:
        mat = np.zeros((10, 10), dtype=np.float64)
        for prev, nxt in zip(digits[:-1], digits[1:]):
            mat[prev, nxt] += 1.0
        # Row-normalise
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0   # avoid divide-by-zero
        mat /= row_sums
        result[name] = mat.tolist()

    return result


def build_feature_vector(entropy: Dict[str, float],
                          transitions: Dict[str, List]) -> np.ndarray:
    """
    Concatenate entropy (3) + flattened transition matrices (3 × 100 = 300)
    into a 303-dim numpy vector for clustering.
    """
    parts = [
        entropy["entropy_mod8"],
        entropy["entropy_mod125"],
        entropy["entropy_mod1000"],
    ]
    for key in ["digit_transition_H", "digit_transition_T", "digit_transition_O"]:
        parts.extend(np.array(transitions[key]).flatten().tolist())
    return np.array(parts, dtype=np.float64)


# =============================================================================
# Windowing
# =============================================================================

def compute_windows(draws: np.ndarray,
                    window_size: int,
                    stride: int) -> Tuple[np.ndarray, List[int]]:
    """
    Slide a window over the draw history and extract feature vectors.

    Returns:
        X       — (n_windows, 303) feature matrix
        offsets — list of start indices for each window
    """
    n = len(draws)
    offsets: List[int] = []
    vectors: List[np.ndarray] = []

    start = 0
    while start + window_size <= n:
        w = draws[start: start + window_size]
        entropy     = extract_entropy_features(w)
        transitions = extract_digit_transition(w)
        vec         = build_feature_vector(entropy, transitions)
        vectors.append(vec)
        offsets.append(start)
        start += stride

    if not vectors:
        raise ValueError(
            f"Not enough draws ({n}) for window_size={window_size}. "
            f"Need at least {window_size} draws."
        )

    return np.vstack(vectors), offsets


# =============================================================================
# Clustering
# =============================================================================

def cluster_windows(X: np.ndarray,
                    k: int,
                    random_state: int = 42) -> Tuple[np.ndarray, float]:
    """
    KMeans clustering on the window feature matrix.

    Returns:
        labels     — (n_windows,) cluster assignment per window
        silhouette — float score (-1 to 1); 0 if only one cluster populated
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    # Standardise before clustering (entropy and transition features on
    # different scales)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)

    n_unique = len(np.unique(labels))
    if n_unique < 2:
        sil = 0.0
    else:
        sil = float(silhouette_score(X_scaled, labels))

    return labels, sil


# =============================================================================
# Regime summary
# =============================================================================

def compute_switch_rate(labels: np.ndarray) -> float:
    """Fraction of consecutive window pairs that change cluster."""
    if len(labels) < 2:
        return 0.0
    switches = np.sum(labels[1:] != labels[:-1])
    return float(switches) / (len(labels) - 1)


def compute_regime_age(labels: np.ndarray) -> int:
    """
    Number of consecutive trailing windows that share the same cluster as
    the most recent window.
    """
    if len(labels) == 0:
        return 0
    current = labels[-1]
    age = 0
    for lbl in reversed(labels):
        if lbl == current:
            age += 1
        else:
            break
    return age


def compute_regime_entropy_profile(X: np.ndarray,
                                   labels: np.ndarray,
                                   k: int) -> Dict[str, Dict[str, float]]:
    """
    Mean entropy features (first 3 dims of X = mod8/125/1000 entropy)
    per regime cluster.
    """
    profile: Dict[str, Dict[str, float]] = {}
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            profile[str(c)] = {
                "entropy_mod8": 0.0,
                "entropy_mod125": 0.0,
                "entropy_mod1000": 0.0,
                "n_windows": 0,
            }
        else:
            mean_feats = X[mask, :3].mean(axis=0)
            profile[str(c)] = {
                "entropy_mod8":    float(mean_feats[0]),
                "entropy_mod125":  float(mean_feats[1]),
                "entropy_mod1000": float(mean_feats[2]),
                "n_windows":       int(mask.sum()),
            }
    return profile


# =============================================================================
# Main analysis
# =============================================================================

def run_trse(draws: np.ndarray,
             window_size: int = DEFAULT_WINDOW_SIZE,
             stride: int = DEFAULT_STRIDE,
             k_clusters: int = DEFAULT_K_CLUSTERS,
             recommended_window_size: int = 8,
             verbose: bool = True) -> dict:
    """
    Full TRSE analysis pipeline.

    Args:
        draws                   : 1-D integer array, chronological order
        window_size             : draws per analysis window
        stride                  : step between windows
        k_clusters              : number of regime clusters
        recommended_window_size : pass-through to Step 1 (from Optuna best)
        verbose                 : print progress

    Returns:
        context dict ready to serialise as trse_context.json
    """
    t0 = time.time()

    if verbose:
        print(f"[TRSE v{__version__}] Starting regime analysis")
        print(f"  draws={len(draws)}  window={window_size}  stride={stride}  k={k_clusters}")

    # 1. Extract windowed features
    X, offsets = compute_windows(draws, window_size, stride)
    n_windows = len(offsets)

    if verbose:
        print(f"  Windows computed: {n_windows}  feature_dim={X.shape[1]}")

    # 2. Cluster
    labels, sil = cluster_windows(X, k_clusters)

    if verbose:
        counts = np.bincount(labels, minlength=k_clusters).tolist()
        print(f"  Clustering done: silhouette={sil:.4f}  counts={counts}")

    # 3. Regime summary
    current_regime = int(labels[-1])
    regime_age     = compute_regime_age(labels)
    switch_rate    = compute_switch_rate(labels)
    regime_counts  = np.bincount(labels, minlength=k_clusters).tolist()
    regime_stable  = regime_age >= REGIME_STABLE_THRESHOLD

    # 4. Current window features (last window)
    last_window  = draws[offsets[-1]: offsets[-1] + window_size]
    curr_entropy = extract_entropy_features(last_window)
    curr_trans   = extract_digit_transition(last_window)

    # 5. Regime entropy profile
    entropy_profile = compute_regime_entropy_profile(X, labels, k_clusters)

    elapsed = time.time() - t0

    context = {
        "trse_version":          __version__,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":       round(elapsed, 3),
        "n_draws":               int(len(draws)),
        "n_windows":             n_windows,
        "window_size":           window_size,
        "stride":                stride,
        "k_clusters":            k_clusters,
        "silhouette":            round(sil, 6),
        "switch_rate":           round(switch_rate, 6),
        "current_regime":        current_regime,
        "regime_age":            regime_age,
        "regime_stable":         regime_stable,
        "regime_counts":         regime_counts,
        "regime_entropy_profile": entropy_profile,
        "current_window_features": {
            **curr_entropy,
            **curr_trans,
        },
        "recommended_window_size": recommended_window_size,
    }

    if verbose:
        print(f"  Current regime: {current_regime}  "
              f"age={regime_age}  stable={regime_stable}  "
              f"switch_rate={switch_rate:.4f}")
        print(f"  Done in {elapsed:.2f}s")

    return context


# =============================================================================
# I/O helpers
# =============================================================================

def load_draws(lottery_file: str) -> np.ndarray:
    """
    Load draw history from daily3.json (or any pipeline-compatible JSON).

    Handles both formats:
      - list of ints:  [472, 385, ...]
      - list of dicts: [{"draw": 472, ...}, ...]
    """
    path = Path(lottery_file)
    if not path.exists():
        raise FileNotFoundError(f"Lottery data not found: {lottery_file}")

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Expected non-empty list in {lottery_file}")

    if isinstance(data[0], dict):
        draws = [int(d["draw"]) for d in data]
    else:
        draws = [int(d) for d in data]

    return np.array(draws, dtype=np.int32)


def load_context(context_file: str) -> Optional[dict]:
    """Load existing trse_context.json if it exists, else return None."""
    path = Path(context_file)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def save_context(context: dict, output_file: str) -> None:
    """Write context dict to JSON."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(context, f, indent=2)
    print(f"[TRSE] Context saved → {output_file}")


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="TRSE v1.0.0 — Temporal Regime Segmentation Engine (Step 0)"
    )
    p.add_argument("--lottery-data", default="daily3.json",
                   help="Path to lottery draw history JSON (default: daily3.json)")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"Output context file (default: {DEFAULT_OUTPUT})")
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
                   help=f"Window size in draws (default: {DEFAULT_WINDOW_SIZE})")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                   help=f"Stride between windows (default: {DEFAULT_STRIDE})")
    p.add_argument("--k-clusters", type=int, default=DEFAULT_K_CLUSTERS,
                   help=f"Number of regime clusters (default: {DEFAULT_K_CLUSTERS})")
    p.add_argument("--recommended-window-size", type=int, default=8,
                   help="Pass-through window_size for Step 1 (default: 8, from Optuna)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if trse_context.json is already fresh")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    verbose = not args.quiet

    # Freshness check — skip if output exists and is newer than lottery data
    if not args.force:
        existing = load_context(args.output)
        if existing is not None:
            lottery_mtime = Path(args.lottery_data).stat().st_mtime
            output_mtime  = Path(args.output).stat().st_mtime
            if output_mtime > lottery_mtime:
                if verbose:
                    print(f"[TRSE] Context is fresh ({args.output}) — use --force to re-run")
                # Still print summary for WATCHER parsing
                print(f"[TRSE] current_regime={existing['current_regime']}  "
                      f"regime_age={existing['regime_age']}  "
                      f"stable={existing['regime_stable']}  "
                      f"silhouette={existing['silhouette']:.4f}")
                return 0

    # Load draws
    try:
        draws = load_draws(args.lottery_data)
    except Exception as e:
        print(f"[TRSE] ERROR loading draws: {e}", file=sys.stderr)
        return 1

    if verbose:
        print(f"[TRSE] Loaded {len(draws)} draws from {args.lottery_data}")

    # Run analysis
    try:
        context = run_trse(
            draws,
            window_size=args.window_size,
            stride=args.stride,
            k_clusters=args.k_clusters,
            recommended_window_size=args.recommended_window_size,
            verbose=verbose,
        )
    except Exception as e:
        print(f"[TRSE] ERROR during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Save
    save_context(context, args.output)

    # Summary line for WATCHER log parsing
    print(f"[TRSE] COMPLETE  "
          f"regime={context['current_regime']}  "
          f"age={context['regime_age']}  "
          f"stable={context['regime_stable']}  "
          f"silhouette={context['silhouette']:.4f}  "
          f"switch_rate={context['switch_rate']:.4f}  "
          f"n_windows={context['n_windows']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
