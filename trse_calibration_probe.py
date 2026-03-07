#!/usr/bin/env python3
"""
TRSE Calibration Probe
======================
Version: 1.0.0
Session: S121

PURPOSE
-------
Validate whether density_proxy() reliably distinguishes validated survivor
windows from confirmed-dead windows using ground truth from S114 Optuna runs.

GROUND TRUTH (S114 real CA Daily 3 data):
  W=3   → 143,959 bidirectional survivors  (SHORT reseed regime)
  W=8   → 43-85 bidirectional survivors    (LONG persistence regime)
  W=31  → 0 survivors
  W=64  → 0 survivors
  W=243 → 0 survivors
  W=489 → 0 survivors

TEST 1 — Ranking test
  density_proxy(W) should rank: W3 > W8 >> W31/W64/W243/W489
  If ranking holds → relative normalization is justified
  If ranking fails → TB conservative position is correct

TEST 2 — Temporal stability
  Run sweep across 3 time slices (early/mid/late draw history)
  If W3/W8 consistently beat W31/W64 across all periods → signal is stable
  If inconsistent → signal may be era-specific

TEST 3 — Separation margin
  Compute gap between short-window pair and long-window pair
  gap = min(D3,D8) - max(D31,D64,D243,D489)
  If gap > 0.10 consistently → threshold recalibration justified
  If gap < 0.05 → signal too weak to rely on

TEST 4 — Relative normalization preview
  Apply relative normalization (max=1.0) and show what
  classify_regime_type() would return with recalibrated thresholds

OUTPUT
------
Printed report + probe_results.json for TB submission

USAGE
-----
  python3 trse_calibration_probe.py --lottery-data daily3.json
  python3 trse_calibration_probe.py --lottery-data daily3.json --output probe_results.json

Author: Team Alpha S121
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

__version__ = "1.0.0"

# ── Ground truth from S114 Optuna runs ───────────────────────────────────────
GROUND_TRUTH = {
    3:   143959,   # W=3  Trial 3, S114
    8:   53,       # W=8  Trial 0 warm-start, S112/S114
    31:  0,        # W=31 Trial 4, S114
    64:  0,        # W=64 (interpolated from W=243=0, W=31=0)
    243: 0,        # W=243 Trial 1, S114
    489: 0,        # W=489 Trial 2, S114
}

# Expected ranking by actual survivors
EXPECTED_HIGH  = [3, 8]        # should have high density_proxy
EXPECTED_LOW   = [31, 64, 243, 489]  # should have low density_proxy

# Full probe window set
PROBE_WINDOWS  = [3, 8, 16, 31, 64, 128, 243, 489]

# Time slice labels
SLICE_LABELS   = ["early (2000-2009)", "mid (2009-2018)", "late (2018-2026)"]


# =============================================================================
# density_proxy — identical to trse_step0.py v1.15
# =============================================================================

def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def density_proxy(draws: np.ndarray, window_size: int,
                  n_sample_windows: int = 20) -> float:
    """
    Residue chi-square based survivor density proxy.
    Identical implementation to trse_step0.py v1.15.
    """
    n = len(draws)
    if n < window_size * 2:
        return 0.0

    max_start = n - window_size
    step      = max(1, max_start // n_sample_windows)
    starts    = list(range(0, max_start, step))[:n_sample_windows]

    chi_scores = []
    for s in starts:
        w = draws[s: s + window_size]

        obs8  = np.bincount(w % 8,   minlength=8).astype(np.float64)
        exp8  = np.full(8, len(w) / 8.0)
        chi8  = float(np.sum((obs8 - exp8) ** 2 / exp8))

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
    score    = math.tanh(mean_chi / (window_size * 2.0))
    return round(min(1.0, max(0.0, score)), 4)


# =============================================================================
# Relative normalization (proposed fix)
# =============================================================================

def relative_normalize(profile: Dict[int, float]) -> Dict[int, float]:
    """Normalize density scores so max = 1.0 across the 4-point profile."""
    max_d = max(profile.values()) or 1.0
    return {w: round(v / max_d, 4) for w, v in profile.items()}


def classify_with_relative_norm(profile: Dict[int, float]) -> Tuple[str, float, float]:
    """
    Classify regime_type using relative normalization.
    Uses W3, W8, W31, W64 only (the validated 4-point set).

    Returns: (regime_type, confidence, duality_score)
    """
    norm = relative_normalize({w: profile[w] for w in [3, 8, 31, 64]
                                if w in profile})

    d3  = norm.get(3,  0.0)
    d8  = norm.get(8,  0.0)
    d31 = norm.get(31, 0.0)
    d64 = norm.get(64, 0.0)

    short_pair    = min(d3, d8)
    long_pair     = max(d31, d64)
    duality_score = round(short_pair - long_pair, 4)

    # Relative thresholds — calibrated for normalized scores
    T_HIGH_REL = 0.35   # normalized score above this = signal present
    T_LOW_REL  = 0.25   # normalized score below this = collapsed

    if d3 > T_HIGH_REL and d8 > T_HIGH_REL and d31 < T_LOW_REL and d64 < T_LOW_REL:
        regime_type = "short_persistence"
    elif d31 > T_HIGH_REL or d64 > T_HIGH_REL:
        regime_type = "long_persistence"
    elif (d3 > T_HIGH_REL or d8 > T_HIGH_REL) and (d31 > T_HIGH_REL or d64 > T_HIGH_REL):
        regime_type = "mixed"
    else:
        regime_type = "unknown"

    confidence = round(1.0 / (1.0 + math.exp(-duality_score * 6)), 4)
    return regime_type, confidence, duality_score


# =============================================================================
# Test 1 — Full-history ranking test
# =============================================================================

def test_ranking(draws: np.ndarray) -> dict:
    """
    Test whether density_proxy ranks validated windows correctly
    against ground truth from S114.
    """
    print("\n" + "="*60)
    print("TEST 1 — Full-history ranking vs S114 ground truth")
    print("="*60)
    print(f"  {'Window':>8} | {'density_proxy':>14} | {'Actual survivors':>17} | {'Rank OK?':>9}")
    print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*17}-+-{'-'*9}")

    profile: Dict[int, float] = {}
    for w in PROBE_WINDOWS:
        d = density_proxy(draws, w)
        profile[w] = d
        actual = GROUND_TRUTH.get(w, "?")
        expected_high = w in EXPECTED_HIGH

        if actual == "?":
            rank_ok = "N/A"
        elif expected_high:
            rank_ok = "✅ HIGH" if d > 0.20 else "❌ should be HIGH"
        else:
            rank_ok = "✅ LOW " if d < 0.30 else "⚠️  unexpectedly high"

        print(f"  W={w:>4}   | {d:>14.4f} | {str(actual):>17} | {rank_ok}")

    # Separation margin
    high_scores = [profile[w] for w in EXPECTED_HIGH if w in profile]
    low_scores  = [profile[w] for w in EXPECTED_LOW  if w in profile]
    gap = min(high_scores) - max(low_scores) if high_scores and low_scores else 0.0

    print(f"\n  Separation gap: min(W3,W8) - max(W31,W64,W243,W489) = {gap:.4f}")
    if gap > 0.10:
        print("  → GAP > 0.10: relative normalization JUSTIFIED ✅")
    elif gap > 0.05:
        print("  → GAP 0.05-0.10: marginal signal, proceed with caution ⚠️")
    else:
        print("  → GAP < 0.05: signal too weak, TB caution VALIDATED ❌")

    # Ranking check
    correct_rank = all(profile.get(h, 0) > profile.get(l, 1)
                       for h in EXPECTED_HIGH
                       for l in EXPECTED_LOW if l in profile)
    print(f"  Ranking holds (all high > all low): {'✅ YES' if correct_rank else '❌ NO'}")

    return {
        "profile":      profile,
        "gap":          round(gap, 4),
        "ranking_holds": correct_rank,
        "high_mean":    round(float(np.mean(high_scores)), 4) if high_scores else 0.0,
        "low_mean":     round(float(np.mean(low_scores)),  4) if low_scores  else 0.0,
    }


# =============================================================================
# Test 2 — Temporal stability across 3 time slices
# =============================================================================

def test_temporal_stability(draws: np.ndarray) -> dict:
    """
    Run density probe across 3 chronological thirds of draw history.
    Tests whether W3/W8 consistently beat W31/W64 across all eras.
    """
    print("\n" + "="*60)
    print("TEST 2 — Temporal stability (3 time slices)")
    print("="*60)

    n      = len(draws)
    slices = [
        draws[:n//3],
        draws[n//3: 2*n//3],
        draws[2*n//3:],
    ]

    slice_results = []
    all_stable    = True

    for i, (label, sl) in enumerate(zip(SLICE_LABELS, slices)):
        profile: Dict[int, float] = {}
        for w in [3, 8, 31, 64]:
            profile[w] = density_proxy(sl, w)

        gap = min(profile[3], profile[8]) - max(profile[31], profile[64])
        regime_type, confidence, duality = classify_with_relative_norm(profile)

        stable_this = (profile[3] > profile[31] and
                       profile[3] > profile[64] and
                       profile[8] > profile[31] and
                       profile[8] > profile[64])

        if not stable_this:
            all_stable = False

        print(f"\n  [{label}]  n={len(sl)} draws")
        print(f"    W3={profile[3]:.4f}  W8={profile[8]:.4f}  "
              f"W31={profile[31]:.4f}  W64={profile[64]:.4f}")
        print(f"    gap={gap:.4f}  "
              f"regime_type={regime_type}  confidence={confidence:.4f}")
        print(f"    W3/W8 > W31/W64: {'✅ YES' if stable_this else '❌ NO'}")

        slice_results.append({
            "label":        label,
            "n_draws":      len(sl),
            "profile":      profile,
            "gap":          round(gap, 4),
            "regime_type":  regime_type,
            "confidence":   confidence,
            "signal_holds": stable_this,
        })

    print(f"\n  Signal consistent across ALL 3 periods: "
          f"{'✅ YES — stable signal' if all_stable else '⚠️  INCONSISTENT — era-specific'}")

    return {
        "slices":     slice_results,
        "all_stable": all_stable,
    }


# =============================================================================
# Test 3 — Separation margin analysis
# =============================================================================

def test_separation_margin(draws: np.ndarray) -> dict:
    """
    Compute separation margin across multiple window sample sizes
    to check if gap is robust or sample-size dependent.
    """
    print("\n" + "="*60)
    print("TEST 3 — Separation margin vs sample windows")
    print("="*60)
    print(f"  {'n_samples':>10} | {'D3':>8} | {'D8':>8} | "
          f"{'D31':>8} | {'D64':>8} | {'gap':>8}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    sample_sizes = [10, 20, 40, 80]
    margins = []

    for ns in sample_sizes:
        d3  = density_proxy(draws, 3,  n_sample_windows=ns)
        d8  = density_proxy(draws, 8,  n_sample_windows=ns)
        d31 = density_proxy(draws, 31, n_sample_windows=ns)
        d64 = density_proxy(draws, 64, n_sample_windows=ns)
        gap = min(d3, d8) - max(d31, d64)
        margins.append(gap)
        print(f"  {ns:>10} | {d3:>8.4f} | {d8:>8.4f} | "
              f"{d31:>8.4f} | {d64:>8.4f} | {gap:>8.4f}")

    mean_margin = float(np.mean(margins))
    std_margin  = float(np.std(margins))
    print(f"\n  Mean gap: {mean_margin:.4f}  Std: {std_margin:.4f}")
    if std_margin < 0.05:
        print("  → Gap is STABLE across sample sizes ✅")
    else:
        print("  → Gap is VARIABLE — sample-size sensitive ⚠️")

    return {
        "sample_sizes":  sample_sizes,
        "margins":       [round(m, 4) for m in margins],
        "mean_margin":   round(mean_margin, 4),
        "std_margin":    round(std_margin,  4),
        "margin_stable": bool(std_margin < 0.05),
    }


# =============================================================================
# Test 4 — Relative normalization preview
# =============================================================================

def test_relative_normalization(draws: np.ndarray, rank_result: dict) -> dict:
    """
    Apply relative normalization to full-history profile and show
    what classify_regime_type() would return with recalibrated thresholds.
    """
    print("\n" + "="*60)
    print("TEST 4 — Relative normalization preview")
    print("="*60)

    profile = rank_result["profile"]
    core    = {w: profile[w] for w in [3, 8, 31, 64]}
    norm    = relative_normalize(core)

    print(f"  Raw scores:        "
          f"W3={core[3]:.4f}  W8={core[8]:.4f}  "
          f"W31={core[31]:.4f}  W64={core[64]:.4f}")
    print(f"  Normalized (max=1):"
          f"W3={norm[3]:.4f}  W8={norm[8]:.4f}  "
          f"W31={norm[31]:.4f}  W64={norm[64]:.4f}")

    regime_type, confidence, duality = classify_with_relative_norm(core)

    print(f"\n  With relative normalization:")
    print(f"    regime_type            = {regime_type}")
    print(f"    regime_type_confidence = {confidence:.4f}")
    print(f"    duality_score          = {duality:.4f}")

    current_type = "unknown"  # current absolute threshold result
    print(f"\n  Current (absolute T_HIGH=0.50): {current_type}")
    print(f"  Proposed (relative T_HIGH=0.35): {regime_type}")

    if regime_type == "short_persistence":
        print(f"\n  → Relative normalization recovers 'short_persistence' ✅")
        print(f"    Step 1 Rule A would activate: max_window_size → 32")
    else:
        print(f"\n  → Relative normalization still returns '{regime_type}' ⚠️")
        print(f"    Further calibration needed before Rule A activation")

    return {
        "raw_profile":      core,
        "normalized":       norm,
        "regime_type":      regime_type,
        "confidence":       confidence,
        "duality_score":    duality,
        "step1_rule_a_fires": regime_type == "short_persistence",
    }


# =============================================================================
# Summary verdict
# =============================================================================

def print_verdict(t1: dict, t2: dict, t3: dict, t4: dict) -> dict:
    print("\n" + "="*60)
    print("VERDICT — Is relative normalization justified?")
    print("="*60)

    evidence = {
        "ranking_holds":          t1["ranking_holds"],
        "separation_gap":         t1["gap"],
        "gap_meaningful":         t1["gap"] > 0.10,
        "temporal_stability":     t2["all_stable"],
        "margin_stable":          t3["margin_stable"],
        "relative_norm_recovers": t4["step1_rule_a_fires"],
    }

    score = sum([
        evidence["ranking_holds"],
        evidence["gap_meaningful"],
        evidence["temporal_stability"],
        evidence["margin_stable"],
        evidence["relative_norm_recovers"],
    ])

    print(f"\n  Ranking holds (W3/W8 > W31/W64):  {'✅' if evidence['ranking_holds'] else '❌'}")
    print(f"  Separation gap > 0.10:             {'✅' if evidence['gap_meaningful'] else '❌'}  (gap={evidence['separation_gap']:.4f})")
    print(f"  Temporal stability (all 3 eras):   {'✅' if evidence['temporal_stability'] else '❌'}")
    print(f"  Margin stable across sample sizes: {'✅' if evidence['margin_stable'] else '❌'}")
    print(f"  Relative norm → short_persistence: {'✅' if evidence['relative_norm_recovers'] else '❌'}")

    print(f"\n  Evidence score: {score}/5")

    if score >= 4:
        verdict = "PROCEED"
        print(f"\n  VERDICT: PROCEED with relative normalization ✅")
        print(f"  Rule A can be activated in Step 1.")
        print(f"  regime_type=short_persistence is defensible on real data.")
    elif score >= 3:
        verdict = "CONDITIONAL"
        print(f"\n  VERDICT: CONDITIONAL ⚠️")
        print(f"  Signal present but not fully stable.")
        print(f"  Wire Step 1 with confidence gate >= 0.70.")
    else:
        verdict = "DEFER"
        print(f"\n  VERDICT: DEFER ❌")
        print(f"  TB conservative position validated.")
        print(f"  Use clustering fields only for Step 1 conditioning.")

    evidence["verdict"] = verdict
    evidence["score"]   = score
    return evidence


# =============================================================================
# I/O
# =============================================================================

def load_draws(lottery_file: str) -> np.ndarray:
    path = Path(lottery_file)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {lottery_file}")
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Expected non-empty list")
    if isinstance(data[0], dict):
        draws = [int(d["draw"]) for d in data]
    else:
        draws = [int(d) for d in data]
    return np.array(draws, dtype=np.int32)


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="TRSE Calibration Probe v1.0.0 — validate density_proxy against S114 ground truth"
    )
    p.add_argument("--lottery-data", default="daily3.json")
    p.add_argument("--output", default="probe_results.json",
                   help="Output JSON for TB submission (default: probe_results.json)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"TRSE Calibration Probe v{__version__}")
    print(f"Lottery data: {args.lottery_data}")
    print(f"Ground truth: S114 CA Daily 3 Optuna trial results")

    try:
        draws = load_draws(args.lottery_data)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(draws)} draws\n")

    t0 = time.time()

    t1 = test_ranking(draws)
    t2 = test_temporal_stability(draws)
    t3 = test_separation_margin(draws)
    t4 = test_relative_normalization(draws, t1)
    verdict = print_verdict(t1, t2, t3, t4)

    elapsed = round(time.time() - t0, 2)
    print(f"\n  Total runtime: {elapsed}s")

    # Save results
    results = {
        "probe_version":      __version__,
        "lottery_file":       args.lottery_data,
        "n_draws":            int(len(draws)),
        "ground_truth":       GROUND_TRUTH,
        "test1_ranking":      {k: v for k, v in t1.items() if k != "profile"},
        "test1_profile":      {str(k): v for k, v in t1["profile"].items()},
        "test2_temporal":     t2,
        "test3_separation":   t3,
        "test4_relative_norm": t4,
        "verdict":            verdict,
        "elapsed_seconds":    elapsed,
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {args.output}")
    print(f"  Submit probe_results.json to TB for review.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
