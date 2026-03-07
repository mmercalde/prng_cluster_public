#!/usr/bin/env python3
"""
W8 Correlation Test
===================
Version: 1.0.0
Session: S121

PURPOSE
-------
Determine whether the W8 density_proxy signal reflects real temporal
structure in the draw sequence, or is an artifact of digit frequency
bias / modulo distribution.

METHOD
------
Compare density_proxy(W) on original draws vs shuffled draws.
Shuffling destroys time-order structure but preserves digit distribution.

If signal collapses after shuffling → real temporal correlation
If signal survives shuffling       → window artifact / digit bias

Run across W=3, W=8, W=31, W=64 to get full picture.

EXPECTED OUTCOME (based on calibration probe)
---------------------------------------------
Real temporal correlation:
  Original W8 ≈ 0.36
  Shuffled W8 ≈ 0.05-0.12
  Ratio       > 2.0  → REAL CORRELATION

Digit bias artifact:
  Original W8 ≈ shuffled W8
  Ratio       ≈ 1.0  → ARTIFACT

USAGE
-----
  python3 w8_correlation_test.py --lottery-data daily3.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Import density_proxy directly from trse_step0.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "trse_step0",
    Path(__file__).parent / "trse_step0.py"
)
trse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trse)
density_proxy = trse.density_proxy

__version__ = "1.0.0"

PROBE_WINDOWS = [3, 8, 31, 64]
N_SHUFFLE_TRIALS = 20   # enough for stable mean/std, still fast


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


def run_shuffle_test(draws: np.ndarray,
                     window_size: int,
                     n_trials: int = N_SHUFFLE_TRIALS) -> dict:
    """
    Run density_proxy on original + n_trials shuffled versions.
    Returns real score, shuffled mean/std, ratio, and verdict.
    """
    real_score = density_proxy(draws, window_size)

    shuffled_scores = []
    for _ in range(n_trials):
        shuffled = draws.copy()
        np.random.shuffle(shuffled)
        shuffled_scores.append(density_proxy(shuffled, window_size))

    shuf_mean = float(np.mean(shuffled_scores))
    shuf_std  = float(np.std(shuffled_scores))
    ratio     = real_score / (shuf_mean + 1e-9)

    # How many std deviations above shuffled baseline?
    z_score = (real_score - shuf_mean) / (shuf_std + 1e-9)

    if ratio > 2.0 and z_score > 3.0:
        verdict = "REAL TEMPORAL CORRELATION"
        symbol  = "✅"
    elif ratio > 1.3 and z_score > 1.5:
        verdict = "WEAK CORRELATION"
        symbol  = "⚠️ "
    else:
        verdict = "LIKELY ARTIFACT"
        symbol  = "❌"

    return {
        "window_size":   window_size,
        "real_score":    round(real_score,  4),
        "shuf_mean":     round(shuf_mean,   4),
        "shuf_std":      round(shuf_std,    4),
        "ratio":         round(ratio,       3),
        "z_score":       round(z_score,     2),
        "verdict":       verdict,
        "symbol":        symbol,
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=f"W8 Correlation Test v{__version__}"
    )
    p.add_argument("--lottery-data", default="daily3.json")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--output", default="w8_correlation_results.json")
    args = p.parse_args(argv)

    np.random.seed(args.seed)

    print(f"W8 Correlation Test v{__version__}")
    print(f"Lottery data : {args.lottery_data}")
    print(f"Shuffle trials: {N_SHUFFLE_TRIALS} per window")
    print(f"Random seed  : {args.seed}")

    try:
        draws = load_draws(args.lottery_data)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(draws)} draws\n")

    t0 = time.time()

    results = []
    print(f"  {'Window':>8} | {'Real':>8} | {'Shuf mean':>10} | "
          f"{'Shuf std':>9} | {'Ratio':>7} | {'Z-score':>8} | Verdict")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*9}-+-"
          f"{'-'*7}-+-{'-'*8}-+-{'-'*26}")

    for w in PROBE_WINDOWS:
        r = run_shuffle_test(draws, w)
        results.append(r)
        print(f"  W={w:>4}   | {r['real_score']:>8.4f} | "
              f"{r['shuf_mean']:>10.4f} | {r['shuf_std']:>9.4f} | "
              f"{r['ratio']:>7.2f} | {r['z_score']:>8.2f} | "
              f"{r['symbol']} {r['verdict']}")

    elapsed = round(time.time() - t0, 3)

    # Overall verdict
    w8_result = next(r for r in results if r["window_size"] == 8)
    w3_result = next(r for r in results if r["window_size"] == 3)

    print(f"\n{'='*70}")
    print(f"OVERALL VERDICT")
    print(f"{'='*70}")

    if w8_result["verdict"] == "REAL TEMPORAL CORRELATION":
        print(f"\n  W8 signal is REAL TEMPORAL STRUCTURE ✅")
        print(f"  ratio={w8_result['ratio']:.2f}x above shuffled baseline")
        print(f"  z={w8_result['z_score']:.1f} standard deviations above noise")
        print(f"\n  Implication: Step-1's W8 discovery reflects genuine")
        print(f"  cross-draw RNG state persistence, not a windowing artifact.")
        print(f"  Step-0 regime_type=short_persistence classification is valid.")
    elif w8_result["verdict"] == "WEAK CORRELATION":
        print(f"\n  W8 signal is WEAKLY CORRELATED ⚠️")
        print(f"  Proceed with caution. Use confidence gate >= 0.80.")
    else:
        print(f"\n  W8 signal is LIKELY ARTIFACT ❌")
        print(f"  TB conservative position validated.")
        print(f"  Step-0 should use clustering fields only.")

    print(f"\n  W3 result: {w3_result['symbol']} {w3_result['verdict']} "
          f"(ratio={w3_result['ratio']:.2f}x, z={w3_result['z_score']:.1f})")
    print(f"  Runtime: {elapsed}s")

    # Save
    output = {
        "test_version":   __version__,
        "lottery_file":   args.lottery_data,
        "n_draws":        int(len(draws)),
        "n_shuffle_trials": N_SHUFFLE_TRIALS,
        "random_seed":    args.seed,
        "results":        results,
        "w8_verdict":     w8_result["verdict"],
        "elapsed_seconds": elapsed,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {args.output}")
    print(f"  Submit to TB with probe_results.json.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
