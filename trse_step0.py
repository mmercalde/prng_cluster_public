#!/usr/bin/env python3
"""
TRSE — Temporal Regime Segmentation Engine (Step 0)
=====================================================
Version: 1.15.0
Session: S121

CHANGES FROM v1.1.0 (TB S121 review — v1.15 spec)
---------------------------------------------------
ADDITION 1 — regime_type via 4-point duality score (TB improvement)
  classify_regime_type() tests the EXACT S114 geometry:
    D3 high + D8 high + D31 low + D64 low → short_persistence
  Uses density_proxy() (residue histogram overlap, pure numpy, ~0.3s)
  Outputs: regime_type, regime_type_confidence, window_density_profile

ADDITION 2 — skip_entropy_profile
  Analyzes inter-draw gap structure vs known [5,56] ADM skip range
  Advisory only — confident=False expected often (TB caution)

ADDITION 3 — dominant_offset_lag
  FFT on draw sequence to detect periodic lag (target ~43)
  Advisory only — confident=False expected on noisy data (TB caution)

CHANGES FROM v1.1.0 (multi-scale clustering)
  All v1.1 fields preserved for backward compat
  Three new fields added to context dict
  trse_version bumped to 1.15.0

DESIGN (approved S119 → S121)
-------------------------------
Feature set per window (clustering layer):
  Entropy drift   — mod8/mod125/mod1000
  Digit transitions — 3×10×10 matrices
  Lag structure   — EXCLUDED (autocorr probe S119 refuted signal)

Scales (clustering layer):
  Fine   W=200, S=50
  Mid    W=400, S=50  (S119 validated primary)
  Coarse W=800, S=100

Regime-type layer (new v1.15):
  Density probes at W=3, W=8, W=31, W=64
  Duality score = min(D3,D8) - max(D31,D64)
  Directly tests S114 geometry

OUTPUT trse_context.json
--------------------------
{
  "trse_version": "1.15.0",
  "current_regime": 0,
  "regime_age": 5,
  "regime_stable": true,
  "regime_confidence": 0.73,

  "regime_type": "short_persistence",
  "regime_type_confidence": 0.81,
  "window_density_profile": {
    "w3": 0.72, "w8": 0.69, "w31": 0.08, "w64": 0.03
  },

  "skip_entropy_profile": {
    "draw_gap_entropy": 0.847,
    "draw_gap_mean": 31.2,
    "draw_gap_std": 14.6,
    "gap_range_min": 5,
    "gap_range_max": 58,
    "consistent_with_known_skip": true,
    "known_skip_range": [5, 56]
  },

  "dominant_offset_lag": {
    "dominant_lag": 43,
    "lag_strength": 0.73,
    "secondary_lag": 21,
    "secondary_strength": 0.41,
    "confident": true
  },

  "scales": { "w200": {...}, "w400": {...}, "w800": {...} },
  "recommended_window_size": 8,
  "window_coherence_ceiling": null,
  "window_confidence": null,
  ...
}

USAGE
-----
  python3 trse_step0.py --lottery-data daily3.json
  python3 trse_step0.py --lottery-data daily3.json --force

Author: Team Alpha S121 (TB v1.15 recommendations incorporated)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__version__ = "1.15.0"

# ── Scale definitions (v1.1 multi-scale) ─────────────────────────────────────
SCALES = [
    {"name": "w200", "window_size": 200, "stride":  50},
    {"name": "w400", "window_size": 400, "stride":  50},   # S119 primary
    {"name": "w800", "window_size": 800, "stride": 100},
]

# ── Density probe windows (v1.15 regime_type) ────────────────────────────────
# W=3 and W=8: S114 validated survivor windows
# W=31 and W=64: should be dead (S114 confirmed 0 survivors)
DENSITY_PROBE_WINDOWS = [3, 8, 31, 64]

# regime_type classification thresholds (TB sketch, validated on real data)
T_HIGH = 0.50   # density above this = strong signal
T_MID  = 0.30   # density above this = moderate signal
T_LOW  = 0.25   # density below this = collapsed

DEFAULT_K_CLUSTERS  = 5
DEFAULT_OUTPUT      = "trse_context.json"
REGIME_STABLE_AGE   = 3
CONFIDENCE_AGE_BOOST_MAX = 0.15
KNOWN_SKIP_RANGE    = [5, 56]   # S112 validated


# =============================================================================
# Feature extraction  (unchanged from v1.0/v1.1)
# =============================================================================

def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def extract_entropy_features(window: np.ndarray) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for mod, name in [(8, "entropy_mod8"), (125, "entropy_mod125"),
                      (1000, "entropy_mod1000")]:
        bins = np.bincount(window % mod, minlength=mod).astype(np.float64)
        feats[name] = _entropy(bins)
    return feats


def extract_digit_transition(window: np.ndarray) -> Dict[str, List[List[float]]]:
    if len(window) < 2:
        empty = [[0.0] * 10 for _ in range(10)]
        return {"digit_transition_H": empty,
                "digit_transition_T": empty,
                "digit_transition_O": empty}
    hundreds = (window // 100) % 10
    tens     = (window //  10) % 10
    ones     =  window         % 10
    result = {}
    for name, digits in [("digit_transition_H", hundreds),
                         ("digit_transition_T", tens),
                         ("digit_transition_O", ones)]:
        mat = np.zeros((10, 10), dtype=np.float64)
        for prev, nxt in zip(digits[:-1], digits[1:]):
            mat[prev, nxt] += 1.0
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat /= row_sums
        result[name] = mat.tolist()
    return result


def build_feature_vector(entropy: Dict[str, float],
                          transitions: Dict[str, List]) -> np.ndarray:
    parts = [entropy["entropy_mod8"],
             entropy["entropy_mod125"],
             entropy["entropy_mod1000"]]
    for key in ["digit_transition_H", "digit_transition_T",
                "digit_transition_O"]:
        parts.extend(np.array(transitions[key]).flatten().tolist())
    return np.array(parts, dtype=np.float64)


# =============================================================================
# Windowing / clustering  (unchanged from v1.1)
# =============================================================================

def compute_windows(draws: np.ndarray,
                    window_size: int,
                    stride: int) -> Tuple[np.ndarray, List[int]]:
    n = len(draws)
    offsets: List[int] = []
    vectors: List[np.ndarray] = []
    start = 0
    while start + window_size <= n:
        w = draws[start: start + window_size]
        entropy     = extract_entropy_features(w)
        transitions = extract_digit_transition(w)
        vectors.append(build_feature_vector(entropy, transitions))
        offsets.append(start)
        start += stride
    if not vectors:
        raise ValueError(
            f"Not enough draws ({n}) for window_size={window_size}.")
    return np.vstack(vectors), offsets


def cluster_windows(X: np.ndarray, k: int,
                    random_state: int = 42) -> Tuple[np.ndarray, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km       = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels   = km.fit_predict(X_scaled)
    n_unique = len(np.unique(labels))
    sil = float(silhouette_score(X_scaled, labels)) if n_unique >= 2 else 0.0
    return labels, sil


def compute_switch_rate(labels: np.ndarray) -> float:
    if len(labels) < 2:
        return 0.0
    return float(np.sum(labels[1:] != labels[:-1])) / (len(labels) - 1)


def compute_regime_age(labels: np.ndarray) -> int:
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


def compute_regime_entropy_profile(X: np.ndarray, labels: np.ndarray,
                                   k: int) -> Dict:
    profile: Dict = {}
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            profile[str(c)] = {"entropy_mod8": 0.0, "entropy_mod125": 0.0,
                                "entropy_mod1000": 0.0, "n_windows": 0}
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
# Per-scale analysis  (unchanged from v1.1)
# =============================================================================

def run_trse_scale(draws: np.ndarray, window_size: int, stride: int,
                   k_clusters: int, verbose: bool = False) -> dict:
    X, offsets = compute_windows(draws, window_size, stride)
    labels, sil = cluster_windows(X, k_clusters)
    current_regime = int(labels[-1])
    regime_age     = compute_regime_age(labels)
    switch_rate    = compute_switch_rate(labels)
    regime_counts  = np.bincount(labels, minlength=k_clusters).tolist()
    age_stable     = regime_age >= REGIME_STABLE_AGE
    if verbose:
        print(f"  [W{window_size}] regime={current_regime}  age={regime_age}  "
              f"stable={age_stable}  sil={sil:.4f}  "
              f"switch_rate={switch_rate:.4f}  n_windows={len(offsets)}")
    return {
        "regime": current_regime, "regime_age": regime_age,
        "stable": age_stable, "silhouette": round(sil, 6),
        "switch_rate": round(switch_rate, 6), "n_windows": len(offsets),
        "regime_counts": regime_counts,
        "_X": X, "_labels": labels, "_offsets": offsets,
    }


# =============================================================================
# Multi-scale fusion  (unchanged from v1.1)
# =============================================================================

def fuse_scales(scale_results: Dict[str, dict], k_clusters: int,
                regime_age_primary: int) -> Tuple[int, bool, float]:
    w400 = scale_results["w400"]
    w800 = scale_results["w800"]
    primary_regime    = w400["regime"]
    cross_scale_agree = (w400["regime"] == w800["regime"])
    regime_stable     = cross_scale_agree and w400["stable"]
    weights = {"w200": 0.2, "w400": 0.5, "w800": 0.3}
    weighted_sil = 0.0
    total_weight = 0.0
    for name, res in scale_results.items():
        if res["regime"] == primary_regime:
            w = weights.get(name, 0.2)
            weighted_sil += w * res["silhouette"]
            total_weight += w
    base_confidence = (weighted_sil / total_weight) if total_weight > 0 else 0.0
    base_confidence = math.tanh(base_confidence * 10)
    age_boost   = min(CONFIDENCE_AGE_BOOST_MAX, regime_age_primary * 0.02)
    confidence  = round(min(1.0, base_confidence + age_boost), 4)
    return primary_regime, regime_stable, confidence


# =============================================================================
# NEW v1.15 — Density proxy (core building block)
# =============================================================================

def density_proxy(draws: np.ndarray, window_size: int,
                  n_sample_windows: int = 20) -> float:
    """
    Cheap residue-based survivor-density proxy for a given window_size.

    Instead of running the full GPU sieve, we measure how much the residue
    distribution in sliding windows of size `window_size` deviates from the
    expected uniform distribution.

    High deviation = structured PRNG signal at this window size
    Low deviation  = noise / no signal

    Returns a normalised score in [0, 1].

    Method:
      1. Sample up to n_sample_windows non-overlapping windows of size W
      2. For each window compute chi-square statistic vs uniform for
         mod8, mod125 residues
      3. Average and normalise to [0,1] using a soft sigmoid

    This is a proxy, not a true survivor count. It correlates with
    survivor density but is not equivalent to it.
    """
    n = len(draws)
    if n < window_size * 2:
        return 0.0

    # Sample windows evenly across the draw history
    max_start   = n - window_size
    step        = max(1, max_start // n_sample_windows)
    starts      = list(range(0, max_start, step))[:n_sample_windows]

    chi_scores = []
    for s in starts:
        w = draws[s: s + window_size]

        # mod8 chi-square
        obs8   = np.bincount(w % 8, minlength=8).astype(np.float64)
        exp8   = np.full(8, len(w) / 8.0)
        chi8   = float(np.sum((obs8 - exp8) ** 2 / exp8))

        # mod125 chi-square (only if window large enough)
        if window_size >= 50:
            obs125 = np.bincount(w % 125, minlength=125).astype(np.float64)
            exp125 = np.full(125, len(w) / 125.0)
            chi125 = float(np.sum((obs125 - exp125) ** 2 / exp125))
        else:
            chi125 = 0.0

        chi_scores.append(chi8 + chi125 * 0.1)

    if not chi_scores:
        return 0.0

    mean_chi = float(np.mean(chi_scores))

    # Soft normalisation — sigmoid centred at a moderate chi value
    # chi=0 (perfect uniform) → score≈0
    # chi=large → score→1
    # Calibrated so that random data gives ~0.5 at W=8 (expected)
    # and structured PRNG data gives >0.6
    score = math.tanh(mean_chi / (window_size * 2.0))
    return round(min(1.0, max(0.0, score)), 4)


# =============================================================================
# NEW v1.15 — Regime type classification (TB 4-point duality improvement)
# =============================================================================

def classify_regime_type(draws: np.ndarray,
                          verbose: bool = False) -> dict:
    """
    Classify PRNG regime type using the S114 4-point duality pattern.

    Tests the exact geometry discovered in S114:
      D3  = high  (short reseed cycle)
      D8  = high  (long persistence state)
      D31 = low   (confirmed 0 survivors in S114)
      D64 = low   (confirmed 0 survivors in S114)

    Duality score = min(D3, D8) - max(D31, D64)
    Positive duality = short_persistence signal

    Returns:
      regime_type            : str
      regime_type_confidence : float [0,1]
      window_density_profile : dict
      duality_score          : float
    """
    d3  = density_proxy(draws, 3)
    d8  = density_proxy(draws, 8)
    d31 = density_proxy(draws, 31)
    d64 = density_proxy(draws, 64)

    short_pair = min(d3, d8)
    long_pair  = max(d31, d64)
    duality_score = round(short_pair - long_pair, 4)

    if verbose:
        print(f"  [TRSE] density_proxy: W3={d3:.3f}  W8={d8:.3f}  "
              f"W31={d31:.3f}  W64={d64:.3f}  duality={duality_score:.3f}")

    # Classification (TB logic, thresholds from spec)
    if d3 > T_HIGH and d8 > T_HIGH and d31 < T_LOW and d64 < T_LOW:
        regime_type = "short_persistence"
    elif d31 > T_MID or d64 > T_MID:
        regime_type = "long_persistence"
    elif (d3 > T_MID or d8 > T_MID) and (d31 > T_MID or d64 > T_MID):
        regime_type = "mixed"
    else:
        regime_type = "unknown"

    # Confidence: sigmoid of duality score (TB recommendation)
    # duality_score in (-1, 1) → confidence in (0, 1)
    confidence = round(1.0 / (1.0 + math.exp(-duality_score * 6)), 4)

    if verbose:
        print(f"  [TRSE] regime_type={regime_type}  "
              f"confidence={confidence:.4f}")

    return {
        "regime_type":            regime_type,
        "regime_type_confidence": confidence,
        "duality_score":          duality_score,
        "window_density_profile": {
            "w3":  d3,
            "w8":  d8,
            "w31": d31,
            "w64": d64,
        },
    }


# =============================================================================
# NEW v1.15 — Skip entropy profile (advisory)
# =============================================================================

def analyze_skip_entropy(draws: np.ndarray) -> dict:
    """
    Analyze inter-draw gap structure vs known [5,56] ADM skip range.

    TB caution: skip range cannot be reliably inferred from observed draw
    values alone if PRNG output is uniform. This field is ADVISORY only.
    consistent_with_known_skip=False is expected and acceptable.

    Returns:
      draw_gap_entropy          : float
      draw_gap_mean             : float
      draw_gap_std              : float
      gap_range_min             : int
      gap_range_max             : int
      consistent_with_known_skip: bool
      known_skip_range          : [int, int]
    """
    if len(draws) < 10:
        return {
            "draw_gap_entropy": 0.0, "draw_gap_mean": 0.0,
            "draw_gap_std": 0.0, "gap_range_min": 0, "gap_range_max": 0,
            "consistent_with_known_skip": False,
            "known_skip_range": KNOWN_SKIP_RANGE,
        }

    diffs = np.abs(np.diff(draws.astype(np.int32))) % 1000
    bins  = np.bincount(diffs, minlength=1000).astype(np.float64)
    entropy = _entropy(bins)

    mean_gap = float(np.mean(diffs))
    std_gap  = float(np.std(diffs))

    # Estimate empirical gap range via 5th/95th percentile
    p5  = int(np.percentile(diffs, 5))
    p95 = int(np.percentile(diffs, 95))

    # Consistency check: does the empirical range overlap with known [5,56]?
    known_lo, known_hi = KNOWN_SKIP_RANGE
    overlap = (p5 <= known_hi) and (p95 >= known_lo)

    return {
        "draw_gap_entropy":           round(entropy, 6),
        "draw_gap_mean":              round(mean_gap, 3),
        "draw_gap_std":               round(std_gap, 3),
        "gap_range_min":              p5,
        "gap_range_max":              p95,
        "consistent_with_known_skip": bool(overlap),
        "known_skip_range":           KNOWN_SKIP_RANGE,
    }


# =============================================================================
# NEW v1.15 — Dominant offset lag (advisory)
# =============================================================================

def detect_offset_periodicity(draws: np.ndarray) -> dict:
    """
    FFT on draw sequence to detect dominant periodic lag.

    TB caution: FFT on integer draw values (mod 1000) is fragile.
    Spectral peaks can appear from digit structure / modulo wrapping.
    confident=False is expected on noisy data and is handled gracefully
    by Step 1 (which ignores the field when confident=False).

    Target: dominant_lag near 43 (S112 validated maintenance cycle).
    Validation: dominant_lag in [20, 80] OR confident=False.

    Returns:
      dominant_lag       : int
      lag_strength       : float
      secondary_lag      : int
      secondary_strength : float
      confident          : bool
    """
    if len(draws) < 100:
        return {"dominant_lag": -1, "lag_strength": 0.0,
                "secondary_lag": -1, "secondary_strength": 0.0,
                "confident": False}

    # FFT on mod-1000 draw values
    signal = (draws % 1000).astype(np.float64)
    signal -= signal.mean()

    fft_vals  = np.fft.rfft(signal)
    power     = np.abs(fft_vals) ** 2
    freqs     = np.fft.rfftfreq(len(signal))

    # Convert frequency → lag in draw units
    # Skip DC component (freq=0) and very high freq (noise)
    min_lag, max_lag = 10, 200
    valid_mask = (freqs > 0) & (1.0 / (freqs + 1e-12) >= min_lag) & \
                 (1.0 / (freqs + 1e-12) <= max_lag)

    if valid_mask.sum() < 2:
        return {"dominant_lag": -1, "lag_strength": 0.0,
                "secondary_lag": -1, "secondary_strength": 0.0,
                "confident": False}

    valid_power = power[valid_mask]
    valid_freqs = freqs[valid_mask]
    total_power = valid_power.sum()

    if total_power == 0:
        return {"dominant_lag": -1, "lag_strength": 0.0,
                "secondary_lag": -1, "secondary_strength": 0.0,
                "confident": False}

    # Normalise power
    norm_power = valid_power / total_power

    # Top 2 peaks
    sorted_idx = np.argsort(norm_power)[::-1]

    dom_idx   = sorted_idx[0]
    dom_freq  = valid_freqs[dom_idx]
    dom_lag   = int(round(1.0 / dom_freq)) if dom_freq > 0 else -1
    dom_str   = round(float(norm_power[dom_idx]), 4)

    sec_idx   = sorted_idx[1] if len(sorted_idx) > 1 else dom_idx
    sec_freq  = valid_freqs[sec_idx]
    sec_lag   = int(round(1.0 / sec_freq)) if sec_freq > 0 else -1
    sec_str   = round(float(norm_power[sec_idx]), 4)

    # Confidence: dominant peak must stand clearly above noise floor
    # TB: confident=False is acceptable — Step 1 ignores field gracefully
    confident = bool(dom_str > 0.15 and min_lag <= dom_lag <= max_lag)

    return {
        "dominant_lag":        dom_lag,
        "lag_strength":        dom_str,
        "secondary_lag":       sec_lag,
        "secondary_strength":  sec_str,
        "confident":           confident,
    }


# =============================================================================
# Main multi-scale entry point (v1.1 + v1.15 additions)
# =============================================================================

def run_trse_multiscale(draws: np.ndarray,
                        k_clusters: int = DEFAULT_K_CLUSTERS,
                        recommended_window_size: int = 8,
                        verbose: bool = True) -> dict:
    """
    Full TRSE analysis: multi-scale clustering + v1.15 structural probes.
    """
    t0 = time.time()

    if verbose:
        print(f"[TRSE v{__version__}] Multi-scale regime analysis")
        print(f"  draws={len(draws)}  scales=W200/W400/W800  k={k_clusters}")

    # ── v1.1: Multi-scale clustering ─────────────────────────────────────────
    scale_results: Dict[str, dict] = {}
    for scale in SCALES:
        name        = scale["name"]
        window_size = scale["window_size"]
        stride      = scale["stride"]
        if verbose:
            print(f"  Running scale {name} (W={window_size}, S={stride})...")
        try:
            result = run_trse_scale(draws, window_size, stride,
                                    k_clusters, verbose=verbose)
            scale_results[name] = result
        except ValueError as e:
            if verbose:
                print(f"  [TRSE] Scale {name} skipped: {e}")
            scale_results[name] = {
                "regime": -1, "regime_age": 0, "stable": False,
                "silhouette": 0.0, "switch_rate": 0.0, "n_windows": 0,
                "regime_counts": [],
                "_X": None, "_labels": None, "_offsets": None,
            }

    primary = scale_results["w400"]
    current_regime, regime_stable, regime_confidence = fuse_scales(
        scale_results, k_clusters, primary["regime_age"]
    )

    if verbose:
        print(f"  Fusion: regime={current_regime}  stable={regime_stable}  "
              f"confidence={regime_confidence:.4f}")

    # Primary scale features
    primary_X       = primary["_X"]
    primary_labels  = primary["_labels"]
    primary_offsets = primary["_offsets"]
    last_window     = draws[primary_offsets[-1]: primary_offsets[-1] + 400]
    curr_entropy    = extract_entropy_features(last_window)
    curr_trans      = extract_digit_transition(last_window)
    entropy_profile = compute_regime_entropy_profile(
        primary_X, primary_labels, k_clusters
    )

    # ── v1.15: Structural probes ──────────────────────────────────────────────
    if verbose:
        print(f"  Running v1.15 structural probes...")

    regime_type_result  = classify_regime_type(draws, verbose=verbose)
    skip_profile        = analyze_skip_entropy(draws)
    offset_lag          = detect_offset_periodicity(draws)

    if verbose:
        print(f"  skip_entropy: consistent={skip_profile['consistent_with_known_skip']}  "
              f"range=[{skip_profile['gap_range_min']},{skip_profile['gap_range_max']}]")
        print(f"  offset_lag:   confident={offset_lag['confident']}  "
              f"dominant_lag={offset_lag['dominant_lag']}")

    elapsed = time.time() - t0

    # ── Build scales summary (strip internal numpy arrays) ───────────────────
    scales_summary = {}
    for name, res in scale_results.items():
        scales_summary[name] = {
            "regime":      res["regime"],
            "stable":      res["stable"],
            "silhouette":  res["silhouette"],
            "switch_rate": res["switch_rate"],
            "n_windows":   res["n_windows"],
            "regime_age":  res["regime_age"],
        }

    context = {
        # Identity
        "trse_version":            __version__,
        "timestamp":               datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":         round(elapsed, 3),
        "n_draws":                 int(len(draws)),
        "k_clusters":              k_clusters,

        # Consensus (fused from all scales)
        "current_regime":          current_regime,
        "regime_age":              primary["regime_age"],
        "regime_stable":           regime_stable,
        "regime_confidence":       regime_confidence,

        # v1.15 regime type (TB 4-point duality)
        "regime_type":             regime_type_result["regime_type"],
        "regime_type_confidence":  regime_type_result["regime_type_confidence"],
        "duality_score":           regime_type_result["duality_score"],
        "window_density_profile":  regime_type_result["window_density_profile"],

        # v1.15 structural probes (advisory)
        "skip_entropy_profile":    skip_profile,
        "dominant_offset_lag":     offset_lag,

        # Primary scale (W400) stats — backward compat
        "silhouette":              primary["silhouette"],
        "switch_rate":             primary["switch_rate"],
        "regime_counts":           primary["regime_counts"],

        # Multi-scale detail
        "scales":                  scales_summary,

        # Step 1 consumption
        "recommended_window_size": recommended_window_size,
        "window_coherence_ceiling": None,   # TB future hook
        "window_confidence":        None,   # TB future hook

        # Current window features (W400)
        "regime_entropy_profile":  entropy_profile,
        "current_window_features": {
            **curr_entropy,
            **curr_trans,
        },
    }

    if verbose:
        print(f"  Done in {elapsed:.2f}s")
        print(f"[TRSE] COMPLETE  "
              f"regime={current_regime}  "
              f"age={primary['regime_age']}  "
              f"stable={regime_stable}  "
              f"confidence={regime_confidence:.4f}  "
              f"regime_type={regime_type_result['regime_type']}  "
              f"type_confidence={regime_type_result['regime_type_confidence']:.4f}")

    return context


# =============================================================================
# I/O helpers  (unchanged)
# =============================================================================

def load_draws(lottery_file: str) -> np.ndarray:
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
    path = Path(context_file)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def save_context(context: dict, output_file: str) -> None:
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
        description=f"TRSE v{__version__} — Temporal Regime Segmentation Engine (Step 0)"
    )
    p.add_argument("--lottery-data",  default="daily3.json")
    p.add_argument("--output",        default=DEFAULT_OUTPUT)
    p.add_argument("--k-clusters",    type=int, default=DEFAULT_K_CLUSTERS)
    p.add_argument("--recommended-window-size", type=int, default=8)
    p.add_argument("--force",         action="store_true")
    p.add_argument("--quiet",         action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args    = parse_args(argv)
    verbose = not args.quiet

    # Freshness check
    if not args.force:
        existing = load_context(args.output)
        if existing is not None:
            try:
                lottery_mtime = Path(args.lottery_data).stat().st_mtime
                output_mtime  = Path(args.output).stat().st_mtime
                if output_mtime > lottery_mtime:
                    ver = existing.get("trse_version", "?")
                    if verbose:
                        print(f"[TRSE] Context is fresh (v{ver}) — use --force to re-run")
                    print(f"[TRSE] COMPLETE  "
                          f"regime={existing.get('current_regime','?')}  "
                          f"age={existing.get('regime_age','?')}  "
                          f"stable={existing.get('regime_stable','?')}  "
                          f"confidence={existing.get('regime_confidence','?')}  "
                          f"regime_type={existing.get('regime_type','?')}  "
                          f"type_confidence={existing.get('regime_type_confidence','?')}")
                    return 0
            except Exception:
                pass

    try:
        draws = load_draws(args.lottery_data)
    except Exception as e:
        print(f"[TRSE] ERROR loading draws: {e}", file=sys.stderr)
        return 1

    if verbose:
        print(f"[TRSE] Loaded {len(draws)} draws from {args.lottery_data}")

    try:
        context = run_trse_multiscale(
            draws,
            k_clusters=args.k_clusters,
            recommended_window_size=args.recommended_window_size,
            verbose=verbose,
        )
    except Exception as e:
        print(f"[TRSE] ERROR during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    save_context(context, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
