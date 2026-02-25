#!/usr/bin/env python3
"""
holdout_quality.py — Holdout Validation Quality Computation
============================================================

S111: Holdout Validation Redesign (v1.1 approved)

Standalone utility module providing:
  - get_survivor_skip(metadata) → per-survivor skip from NPZ metadata
  - compute_holdout_quality(holdout_features) → composite score [0,1]

Composite formula (v1.1 §4):
  50% CRT match quality
  30% distributional coherence
  20% temporal stability

This module is intentionally dependency-free (no SurvivorScorer import)
so it can be used by both Step 3 workers and Step 5 diagnostics.

Author: Team Alpha
Date: 2026-02-25
Session: S111
"""


def get_survivor_skip(metadata: dict) -> int:
    """
    Extract per-survivor skip value from NPZ metadata.

    Priority: skip_best → skip_min → 0

    The sieve discovers skip rhythms per-survivor. The holdout feature
    extraction MUST use the same skip the sieve found, not a uniform
    default. This was the root cause of the Poisson(λ≈1) noise floor
    in the old holdout_hits computation.

    Args:
        metadata: dict from survivor's NPZ metadata or features dict
                  Expected keys: skip_best, skip_min (float or int)

    Returns:
        int: skip value >= 0
    """
    if not isinstance(metadata, dict):
        return 0

    for key in ("skip_best", "skip_min"):
        val = metadata.get(key)
        if val is None:
            continue
        try:
            ival = int(float(val))  # handle both int and float storage
            if ival >= 0:
                return ival
        except (ValueError, TypeError, OverflowError):
            continue

    return 0


def compute_holdout_quality(holdout_features: dict) -> float:
    """
    Compute composite holdout quality score from holdout feature vector.

    This uses the SAME feature methodology as training (SurvivorScorer),
    applied to holdout data. The composite weights are:

      CRT match quality (50%):
        40% residue_1000_match_rate
        20% lane_agreement_8
        20% lane_agreement_125
        20% lane_consistency

      Distributional coherence (30%):
        34% residue_1000_coherence
        33% residue_8_coherence
        33% residue_125_coherence

      Temporal stability (20%):
        100% temporal_stability_mean

    Per v1.1 proposal §4 (Team Beta approved).

    Args:
        holdout_features: dict of feature_name → float, as returned by
                          SurvivorScorer.extract_ml_features()

    Returns:
        float: composite quality score in [0.0, 1.0]
    """
    hf = holdout_features or {}

    def _get(name: str, default: float = 0.0) -> float:
        try:
            return float(hf.get(name, default))
        except (ValueError, TypeError):
            return float(default)

    # --- CRT match quality (50%) ---
    crt_score = (
        0.40 * _get("residue_1000_match_rate")
        + 0.20 * _get("lane_agreement_8")
        + 0.20 * _get("lane_agreement_125")
        + 0.20 * _get("lane_consistency")
    )

    # --- Distributional coherence (30%) ---
    dist_score = (
        0.34 * _get("residue_1000_coherence")
        + 0.33 * _get("residue_8_coherence")
        + 0.33 * _get("residue_125_coherence")
    )

    # --- Temporal stability (20%) ---
    temp_score = _get("temporal_stability_mean")

    # --- Composite ---
    holdout_quality = (
        0.50 * crt_score
        + 0.30 * dist_score
        + 0.20 * temp_score
    )

    # Clamp to [0, 1] defensively
    return max(0.0, min(1.0, float(holdout_quality)))


def compute_autocorrelation_diagnostics(survivors: list) -> dict:
    """
    Compute per-feature Pearson correlation between training features
    and holdout_features. Required by v1.1 §5 for autocorrelation
    detection.

    If R² > 0.30 and median autocorrelation is high, the holdout_quality
    signal may be feature autocorrelation rather than genuine temporal
    persistence. Step 6 + Chapter 13 external validation resolves this.

    Args:
        survivors: list of survivor dicts, each with "features" and
                   optionally "holdout_features"

    Returns:
        dict with:
          - n_survivors_with_holdout_features: int
          - feature_corr: {feature_name: pearson_r}
          - median_corr: float
          - warning: str or None
    """
    import math

    # Collect paired series
    series = {}
    n = 0

    for s in survivors:
        hf = s.get("holdout_features") or {}
        tf = s.get("features") or {}
        if not hf or not tf:
            continue

        common = set(tf.keys()) & set(hf.keys())
        for k in common:
            try:
                x = float(tf[k])
                y = float(hf[k])
            except (ValueError, TypeError):
                continue
            series.setdefault(k, ([], []))
            series[k][0].append(x)
            series[k][1].append(y)
        n += 1

    def _pearson(xs, ys):
        if len(xs) < 3:
            return None
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        vx = sum((a - mx) ** 2 for a in xs)
        vy = sum((b - my) ** 2 for b in ys)
        if vx <= 0 or vy <= 0:
            return None
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        return float(cov / math.sqrt(vx * vy))

    feature_corr = {}
    for k, (xs, ys) in series.items():
        r = _pearson(xs, ys)
        if r is not None:
            feature_corr[k] = round(r, 6)

    # Median correlation
    corr_values = list(feature_corr.values())
    if corr_values:
        sorted_corr = sorted(corr_values)
        mid = len(sorted_corr) // 2
        median_corr = sorted_corr[mid] if len(sorted_corr) % 2 else (sorted_corr[mid - 1] + sorted_corr[mid]) / 2
    else:
        median_corr = 0.0

    # Warning check
    warning = None
    if median_corr > 0.70:
        warning = (
            f"HIGH AUTOCORRELATION: median corr={median_corr:.3f}. "
            "holdout_quality may reflect scorer self-consistency, not "
            "temporal persistence. Validate with Step 6 + Chapter 13."
        )

    return {
        "n_survivors_with_holdout_features": n,
        "feature_corr": feature_corr,
        "median_corr": round(median_corr, 6),
        "warning": warning,
    }
