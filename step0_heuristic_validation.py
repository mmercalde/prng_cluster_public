#!/usr/bin/env python3
"""
step0_heuristic_validation.py
=============================
Track A — Real-data validation of TRSE (Step 0) advisory heuristics.

WHAT THIS DOES
--------------
Loads the REAL draw dataset and asks, for each Step-0 advisory heuristic,
two questions the trse_context.json values alone cannot answer:

  1. SIGNIFICANCE (permutation-null): Is the observed statistic real temporal
     structure, or the kind of value you'd get from ANY ordering of these
     draws? We shuffle the draw order N times, recompute the statistic each
     time to build a null distribution, and locate the real value in it.
     - offset periodicity  -> lag_strength (FFT power). Shuffling destroys
       periodicity, so a real dominant lag must sit above the shuffle null.
     - skip entropy        -> draw_gap_entropy. Order-dependent; two-sided.
     - regime duality      -> duality_score / density_proxy(W). Tests whether
       short-window density reflects LOCAL structure vs global composition
       (which shuffling preserves).

  2. STABILITY (time-split): Do the heuristics computed on an earlier slice of
     the series still hold on a later slice? A heuristic that flips across time
     is not a stable property of the process.

IMPORTANT — this tests YOUR code, not a reimplementation. It imports
density_proxy / classify_regime_type / analyze_skip_entropy /
detect_offset_periodicity / load_draws directly from trse_step0 and exercises
those. If a signature has drifted, it says so and stops rather than guessing.

HONEST SCOPE
------------
These heuristics were DERIVED from this dataset, so this harness measures
significance-vs-noise and temporal stability — NOT true out-of-sample validity.
For genuine out-of-sample confirmation, re-run --lottery-data against a SECOND
dataset (e.g. PA Pick 3). This is CPU-only; no cluster, no GPU.

USAGE
-----
    python3 step0_heuristic_validation.py --lottery-data daily3.json \
        --permutations 1000 --out step0_validation_findings.json

Read-only: computes and reports; writes only the findings file.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

try:
    import trse_step0 as t0
except Exception as e:  # pragma: no cover
    print(f"[FATAL] Could not import trse_step0: {e}\n"
          f"Run this from the project root where trse_step0.py lives.",
          file=sys.stderr)
    sys.exit(2)


# ── Signature guard (Team Beta standard) ─────────────────────────────────────
REQUIRED = {
    "load_draws": ["lottery_file"],
    "density_proxy": ["draws", "window_size"],
    "classify_regime_type": ["draws"],
    "analyze_skip_entropy": ["draws"],
    "detect_offset_periodicity": ["draws"],
}


def assert_api():
    problems = []
    for name, params in REQUIRED.items():
        fn = getattr(t0, name, None)
        if fn is None:
            problems.append(f"missing function: {name}")
            continue
        sig = inspect.signature(fn)
        for p in params:
            if p not in sig.parameters:
                problems.append(f"{name}: expected param '{p}' not in {list(sig.parameters)}")
    if problems:
        print("[FATAL] trse_step0 API drift detected:", file=sys.stderr)
        for p in problems:
            print("   -", p, file=sys.stderr)
        sys.exit(2)


def call_filtered(fn: Callable, draws: np.ndarray, **maybe):
    """Call fn(draws, **kw) passing only kwargs the signature accepts."""
    sig = inspect.signature(fn)
    kw = {k: v for k, v in maybe.items() if k in sig.parameters}
    return fn(draws, **kw)


# ── Permutation engine ───────────────────────────────────────────────────────
def permutation_test(
    draws: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_perm: int,
    obs_repeats: int,
    rng: np.random.Generator,
    side: str,  # "high" | "low" | "two"
) -> Dict:
    # Observed: averaged over repeats to absorb any internal sampling noise.
    obs_samples = [float(stat_fn(draws)) for _ in range(obs_repeats)]
    observed = float(np.mean(obs_samples))

    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        shuffled = rng.permutation(draws)
        null[i] = float(stat_fn(shuffled))

    null_mean = float(np.mean(null))
    null_std = float(np.std(null))
    # Empirical p-values with +1 correction (never reports p=0).
    p_high = (np.sum(null >= observed) + 1) / (n_perm + 1)
    p_low = (np.sum(null <= observed) + 1) / (n_perm + 1)
    if side == "high":
        p = p_high
    elif side == "low":
        p = p_low
    else:
        p = min(1.0, 2.0 * min(p_high, p_low))
    z = (observed - null_mean) / null_std if null_std > 0 else float("inf")
    pct = float((np.sum(null <= observed) / n_perm) * 100.0)
    return {
        "observed": round(observed, 6),
        "observed_spread": round(float(np.std(obs_samples)), 6),
        "null_mean": round(null_mean, 6),
        "null_std": round(null_std, 6),
        "z_vs_null": round(z, 3) if np.isfinite(z) else None,
        "percentile_in_null": round(pct, 2),
        "p_value": round(float(p), 5),
        "side": side,
        "n_permutations": n_perm,
    }


def verdict_sig(p: float, alpha: float) -> str:
    return "SIGNIFICANT" if p < alpha else "NOT-DISTINGUISHABLE-FROM-NULL"


# ── Heuristic statistic extractors (call the REAL functions) ─────────────────
def stat_offset_strength(draws: np.ndarray) -> float:
    return float(t0.detect_offset_periodicity(draws).get("lag_strength", 0.0))


def stat_skip_entropy(draws: np.ndarray) -> float:
    return float(t0.analyze_skip_entropy(draws).get("draw_gap_entropy", 0.0))


def stat_regime_duality(draws: np.ndarray) -> float:
    r = call_filtered(t0.classify_regime_type, draws, verbose=False)
    return float(r.get("duality_score", 0.0))


def make_density_stat(window: int) -> Callable[[np.ndarray], float]:
    def _f(draws: np.ndarray) -> float:
        return float(t0.density_proxy(draws, window))
    return _f


# ── Time-split stability ─────────────────────────────────────────────────────
def time_split_report(draws: np.ndarray, frac: float) -> Dict:
    n = len(draws)
    cut = int(n * frac)
    early, late = draws[:cut], draws[cut:]
    out = {"split_fraction": frac, "n_early": len(early), "n_late": len(late)}
    for label, seg in (("early", early), ("late", late)):
        if len(seg) < 64:
            out[label] = {"error": "segment too short for probes"}
            continue
        off = t0.detect_offset_periodicity(seg)
        skp = t0.analyze_skip_entropy(seg)
        reg = call_filtered(t0.classify_regime_type, seg, verbose=False)
        out[label] = {
            "dominant_lag": off.get("dominant_lag"),
            "lag_strength": off.get("lag_strength"),
            "offset_confident": off.get("confident"),
            "gap_p5_p95": [skp.get("gap_range_min"), skp.get("gap_range_max")],
            "skip_consistent": skp.get("consistent_with_known_skip"),
            "regime_type": reg.get("regime_type"),
            "w3_w8_ratio": reg.get("w3_w8_ratio"),
        }
    e, l = out.get("early", {}), out.get("late", {})
    if "error" not in e and "error" not in l:
        out["stability"] = {
            "regime_type_agrees": e.get("regime_type") == l.get("regime_type"),
            "offset_both_confident": bool(e.get("offset_confident") and l.get("offset_confident")),
            "dominant_lag_delta": (
                abs((e.get("dominant_lag") or 0) - (l.get("dominant_lag") or 0))
            ),
            "skip_consistent_both": bool(e.get("skip_consistent") and l.get("skip_consistent")),
        }
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Track A: real-data Step-0 heuristic validation")
    ap.add_argument("--lottery-data", required=True, help="Path to draw dataset (e.g. daily3.json)")
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--observed-repeats", type=int, default=20,
                    help="Repeats for the observed statistic (absorbs density_proxy sampling noise)")
    ap.add_argument("--time-split", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="step0_validation_findings.json")
    args = ap.parse_args()

    assert_api()
    rng = np.random.default_rng(args.seed)
    draws = t0.load_draws(args.lottery_data)

    print("=" * 74)
    print("TRACK A — REAL-DATA STEP-0 HEURISTIC VALIDATION")
    print("=" * 74)
    print(f"dataset: {args.lottery_data}   draws: {len(draws)}   "
          f"permutations: {args.permutations}   alpha: {args.alpha}")
    print("NOTE: heuristics were derived from this dataset. This tests "
          "significance-vs-noise + temporal stability, NOT out-of-sample "
          "validity. For that, re-run on a second dataset (PA Pick 3).")
    print("-" * 74)

    # Reference: what the heuristics actually report on the full real series.
    off_full = t0.detect_offset_periodicity(draws)
    skip_full = t0.analyze_skip_entropy(draws)
    reg_full = call_filtered(t0.classify_regime_type, draws, verbose=False)
    print(f"reported offset: dominant_lag={off_full.get('dominant_lag')} "
          f"strength={off_full.get('lag_strength')} confident={off_full.get('confident')}")
    print(f"reported skip:   gap[p5,p95]=[{skip_full.get('gap_range_min')},"
          f"{skip_full.get('gap_range_max')}] consistent={skip_full.get('consistent_with_known_skip')}")
    print(f"reported regime: type={reg_full.get('regime_type')} "
          f"w3_w8_ratio={reg_full.get('w3_w8_ratio')} duality={reg_full.get('duality_score')}")
    print(f"density W3/W8/W31/W64: {reg_full.get('window_density_profile')}")
    print("-" * 74)

    tests = [
        ("offset_periodicity.lag_strength", stat_offset_strength, "high"),
        ("skip_entropy.draw_gap_entropy", stat_skip_entropy, "two"),
        ("regime.duality_score", stat_regime_duality, "high"),
        ("regime.density_W3", make_density_stat(3), "high"),
        ("regime.density_W8", make_density_stat(8), "high"),
    ]

    findings: List[Dict] = []
    for name, fn, side in tests:
        res = permutation_test(draws, fn, args.permutations, args.observed_repeats, rng, side)
        res["heuristic"] = name
        res["verdict"] = verdict_sig(res["p_value"], args.alpha)
        findings.append(res)
        print(f"[{res['verdict']:>28}]  {name}")
        print(f"    observed={res['observed']}  null={res['null_mean']}±{res['null_std']}  "
              f"z={res['z_vs_null']}  p={res['p_value']} ({side})")

    print("-" * 74)
    stab = time_split_report(draws, args.time_split)
    s = stab.get("stability", {})
    print(f"TIME-SPLIT STABILITY (first {int(args.time_split*100)}% vs rest):")
    if s:
        print(f"    regime_type agrees: {s['regime_type_agrees']}  |  "
              f"offset both confident: {s['offset_both_confident']}  |  "
              f"dominant_lag delta: {s['dominant_lag_delta']}  |  "
              f"skip consistent both: {s['skip_consistent_both']}")
    else:
        print("    (segments too short to split)")

    out = {
        "probe": "step0_heuristic_validation_v1",
        "dataset": args.lottery_data,
        "n_draws": int(len(draws)),
        "alpha": args.alpha,
        "reported_full_series": {
            "offset": off_full, "skip": skip_full, "regime": reg_full,
        },
        "permutation_findings": findings,
        "time_split": stab,
        "caveat": ("Heuristics derived from this dataset; permutation=significance, "
                   "time-split=stability. Out-of-sample requires a second dataset."),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("-" * 74)
    print(f"[done] findings written to {args.out}  (read-only; no data modified)")


if __name__ == "__main__":
    main()
