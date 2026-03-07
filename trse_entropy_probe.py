#!/usr/bin/env python3
"""
TRSE Entropy Drift Probe — Pre-validation Test
===============================================
Runs BEFORE building the full TRSE engine.
Purpose: Detect candidate regime boundaries in draw history
         using Shannon entropy drift across 3 CRT lanes.

Team Beta prescription:
  - mod-8   entropy  (session/day-of-week structure)
  - mod-125 entropy  (mid-range residue structure)
  - mod-1000 entropy (full residue structure)

Runtime: <30 seconds on Zeus, CPU only, no GPU required.

Usage:
    python3 trse_entropy_probe.py
    python3 trse_entropy_probe.py --file daily3.json
    python3 trse_entropy_probe.py --file daily3.json --window 800 --stride 100
    python3 trse_entropy_probe.py --file daily3.json --plot

Output:
    trse_boundary_candidates.json   — machine-readable results
    Console summary                 — human-readable interpretation
"""

import json
import math
import argparse
import sys
import os
from collections import Counter
from datetime import datetime

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FILE   = "daily3.json"
DEFAULT_WINDOW = 500          # draws per sliding window
DEFAULT_STRIDE = 50           # step size between windows
DEFAULT_Z      = 2.0          # z-score threshold for boundary flagging
OUTPUT_FILE    = "trse_boundary_candidates.json"


# ─────────────────────────────────────────────────────────────────────────────
# DRAW LOADER  (matches window_optimizer.py format exactly)
# ─────────────────────────────────────────────────────────────────────────────

def load_draws(path: str) -> list:
    """
    Load daily3.json in any of its known formats.
    Returns flat list of 3-digit integers (one per draw).
    """
    with open(path) as f:
        data = json.load(f)

    # Format A: list of {"draw": [d1, d2, d3], ...}
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if "draw" in data[0]:
            raw = [d["draw"] for d in data]
            # draw field may be a list [d1,d2,d3] or a single int
            draws = []
            for r in raw:
                if isinstance(r, list):
                    # Combine 3-digit draw: e.g. [4,7,2] -> 472
                    draws.append(int("".join(str(x) for x in r)))
                else:
                    draws.append(int(r))
            return draws
        # Other dict formats
        for k in ("draws", "numbers", "values", "history", "results"):
            if k in data[0]:
                return [int(x) for x in data[0][k]]

    # Format B: plain list of ints or strings
    if isinstance(data, list):
        return [int(x) for x in data]

    # Format C: top-level dict with known keys
    if isinstance(data, dict):
        for k in ("draws", "numbers", "values", "history"):
            if k in data and isinstance(data[k], list):
                return [int(x) for x in data[k]]

    raise ValueError(
        f"Unsupported format in {path}. "
        "Expected list of draw dicts or plain int list."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTROPY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def shannon_entropy(values: list) -> float:
    """Shannon entropy in bits."""
    c = Counter(values)
    n = len(values)
    if n == 0:
        return 0.0
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def lane_entropies(window_draws: list) -> tuple:
    """
    Compute entropy across all 3 CRT lanes simultaneously.
    Matches existing residue feature architecture in Step 3.

    Returns: (H_mod8, H_mod125, H_mod1000)
    """
    vals = [int(x) for x in window_draws]
    return (
        shannon_entropy([x % 8   for x in vals]),
        shannon_entropy([x % 125 for x in vals]),
        shannon_entropy([x % 1000 for x in vals]),
    )


def combined_drift_score(delta8: float, delta125: float, delta1000: float) -> float:
    """
    Weighted combination of 3-lane entropy deltas.
    mod-1000 carries most information; mod-8 is most sensitive to
    session/structural changes.
    """
    return 0.20 * delta8 + 0.30 * delta125 + 0.50 * delta1000


# ─────────────────────────────────────────────────────────────────────────────
# BOUNDARY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_boundaries(draws: list, window: int, stride: int, z_thresh: float) -> dict:
    """
    Slide window across draw history, compute 3-lane entropy per window,
    detect sharp drift transitions as boundary candidates.

    Returns full results dict ready for JSON output.
    """
    n = len(draws)
    windows = []
    h8_series, h125_series, h1000_series = [], [], []

    for i in range(0, n - window + 1, stride):
        segment = draws[i:i + window]
        h8, h125, h1000 = lane_entropies(segment)
        center = i + window // 2
        windows.append(center)
        h8_series.append(h8)
        h125_series.append(h125)
        h1000_series.append(h1000)

    h8    = np.array(h8_series)
    h125  = np.array(h125_series)
    h1000 = np.array(h1000_series)

    # Per-lane absolute deltas
    d8    = np.abs(np.diff(h8))
    d125  = np.abs(np.diff(h125))
    d1000 = np.abs(np.diff(h1000))

    # Combined weighted drift score
    combined = np.array([
        combined_drift_score(d8[i], d125[i], d1000[i])
        for i in range(len(d8))
    ])

    # Z-score normalise combined drift
    z = (combined - combined.mean()) / (combined.std() + 1e-12)

    # Flag boundary candidates where z > threshold
    candidates = []
    for i, score in enumerate(z):
        if score > z_thresh:
            draw_idx = windows[i]   # centre of the transition window
            candidates.append({
                "draw_index":       int(draw_idx),
                "z_score":          round(float(score), 4),
                "combined_drift":   round(float(combined[i]), 6),
                "delta_mod8":       round(float(d8[i]), 6),
                "delta_mod125":     round(float(d125[i]), 6),
                "delta_mod1000":    round(float(d1000[i]), 6),
                "lanes_triggered":  int(
                    (d8[i]    > d8.mean()    + 1.5 * d8.std())    +
                    (d125[i]  > d125.mean()  + 1.5 * d125.std())  +
                    (d1000[i] > d1000.mean() + 1.5 * d1000.std())
                ),
            })

    # Sort by z-score descending
    candidates.sort(key=lambda x: x["z_score"], reverse=True)

    return {
        "meta": {
            "total_draws":   n,
            "window_size":   window,
            "stride":        stride,
            "z_threshold":   z_thresh,
            "total_windows": len(windows),
            "generated_at":  datetime.now().isoformat(),
        },
        "series": {
            "window_centers": [int(x) for x in windows],
            "entropy_mod8":    [round(x, 6) for x in h8_series],
            "entropy_mod125":  [round(x, 6) for x in h125_series],
            "entropy_mod1000": [round(x, 6) for x in h1000_series],
            "combined_drift":  [round(x, 6) for x in combined.tolist()],
            "z_scores":        [round(x, 4) for x in z.tolist()],
        },
        "stats": {
            "entropy_mod8_mean":    round(float(h8.mean()), 4),
            "entropy_mod8_std":     round(float(h8.std()),  4),
            "entropy_mod125_mean":  round(float(h125.mean()), 4),
            "entropy_mod125_std":   round(float(h125.std()),  4),
            "entropy_mod1000_mean": round(float(h1000.mean()), 4),
            "entropy_mod1000_std":  round(float(h1000.std()),  4),
            "combined_drift_mean":  round(float(combined.mean()), 6),
            "combined_drift_std":   round(float(combined.std()),  6),
            "combined_drift_max":   round(float(combined.max()),  6),
            "z_score_max":          round(float(z.max()), 4),
        },
        "boundary_candidates": candidates,
        "candidate_count":     len(candidates),
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def interpret(results: dict) -> str:
    """
    Human-readable interpretation of boundary candidates.
    Tells Michael directly what the numbers mean for TRSE.
    """
    meta       = results["meta"]
    stats      = results["stats"]
    candidates = results["boundary_candidates"]
    n          = meta["total_draws"]
    count      = len(candidates)

    lines = []
    lines.append("=" * 65)
    lines.append("TRSE ENTROPY DRIFT PROBE — RESULTS")
    lines.append("=" * 65)
    lines.append(f"Total draws:      {n:,}")
    lines.append(f"Windows scanned:  {meta['total_windows']}")
    lines.append(f"Window size:      {meta['window_size']} draws")
    lines.append(f"Stride:           {meta['stride']} draws")
    lines.append(f"Z threshold:      {meta['z_threshold']}")
    lines.append("")
    lines.append("Entropy statistics (stability indicators):")
    lines.append(f"  mod-8    mean={stats['entropy_mod8_mean']:.4f}  std={stats['entropy_mod8_std']:.4f}")
    lines.append(f"  mod-125  mean={stats['entropy_mod125_mean']:.4f}  std={stats['entropy_mod125_std']:.4f}")
    lines.append(f"  mod-1000 mean={stats['entropy_mod1000_mean']:.4f}  std={stats['entropy_mod1000_std']:.4f}")
    lines.append("")
    lines.append(f"Combined drift max z-score: {stats['z_score_max']:.4f}")
    lines.append("")
    lines.append("─" * 65)

    if count == 0:
        lines.append("RESULT: No candidate boundaries found.")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  Entropy is stable across the full draw history.")
        lines.append("  TRSE may still find subtle regime shifts via survivor")
        lines.append("  overlap, but this probe does not support a strong case.")
        lines.append("  Consider increasing window size and re-running before")
        lines.append("  committing to full TRSE implementation.")

    elif count <= 3:
        lines.append(f"RESULT: {count} candidate boundary(ies) — STRONG SIGNAL.")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  Sharp, localised entropy discontinuities detected.")
        lines.append("  This is consistent with real generator regime changes.")
        lines.append("  TRSE is strongly worth building.")
        lines.append("")
        lines.append("Candidate boundaries (draw index → approx calendar zone):")
        for c in candidates:
            lanes = c["lanes_triggered"]
            lane_str = f"  [{lanes}/3 lanes agree]"
            lines.append(
                f"  draw {c['draw_index']:6,}   z={c['z_score']:.3f}"
                f"   drift={c['combined_drift']:.5f}{lane_str}"
            )

    elif count <= 8:
        lines.append(f"RESULT: {count} candidate boundaries — MODERATE SIGNAL.")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  Multiple entropy shifts detected. Some may be real")
        lines.append("  regime boundaries; others may be noise.")
        lines.append("  Focus on candidates where lanes_triggered >= 2.")
        lines.append("  TRSE is worth building — use 2nd-pass confirmation.")
        lines.append("")
        lines.append("Top candidates (all):")
        for c in candidates:
            lanes = c["lanes_triggered"]
            conf = "STRONG" if lanes >= 2 else "WEAK"
            lines.append(
                f"  draw {c['draw_index']:6,}   z={c['z_score']:.3f}"
                f"   [{conf} — {lanes}/3 lanes]"
            )

    else:
        lines.append(f"RESULT: {count} candidates — NOISY. Window may be too small.")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  Too many candidates likely means the window size is")
        lines.append("  capturing short-term variance rather than regime shifts.")
        lines.append("  Re-run with --window 800 --stride 100 to filter noise.")
        lines.append("")
        lines.append(f"  Top 5 by z-score:")
        for c in candidates[:5]:
            lines.append(
                f"  draw {c['draw_index']:6,}   z={c['z_score']:.3f}"
                f"   [{c['lanes_triggered']}/3 lanes]"
            )

    lines.append("")
    lines.append("─" * 65)
    lines.append(f"Full results saved to: {OUTPUT_FILE}")
    lines.append("─" * 65)

    # Draw-to-date estimation (CA Daily 3 ~2 draws/day from 2000)
    if count > 0:
        lines.append("")
        lines.append("Approximate calendar mapping (2 draws/day from Jan 2000):")
        lines.append("  (Cross-reference with known CA Lottery equipment history)")
        import datetime as dt
        base = dt.date(2000, 1, 1)
        for c in candidates[:10]:
            approx_days = c["draw_index"] // 2
            approx_date = base + dt.timedelta(days=approx_days)
            lines.append(
                f"  draw {c['draw_index']:6,}  →  ~{approx_date.strftime('%b %Y')}"
                f"   z={c['z_score']:.3f}"
            )

    lines.append("=" * 65)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict):
    """Plot entropy series and boundary candidates if matplotlib available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    series     = results["series"]
    candidates = results["boundary_candidates"]
    centers    = series["window_centers"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("TRSE Entropy Drift Probe", fontsize=13, fontweight="bold")

    axes[0].plot(centers, series["entropy_mod8"],    color="steelblue",  lw=1.2)
    axes[0].set_ylabel("H mod-8",   fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(centers, series["entropy_mod125"],  color="darkorange", lw=1.2)
    axes[1].set_ylabel("H mod-125", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(centers, series["entropy_mod1000"], color="darkgreen",  lw=1.2)
    axes[2].set_ylabel("H mod-1000", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(centers[1:], series["z_scores"],    color="crimson",    lw=1.2)
    axes[3].axhline(results["meta"]["z_threshold"], color="black",
                    ls="--", lw=0.8, label=f"z={results['meta']['z_threshold']}")
    axes[3].set_ylabel("Z-score", fontsize=9)
    axes[3].set_xlabel("Draw index", fontsize=9)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=8)

    # Mark candidates on all panels
    for c in candidates:
        for ax in axes:
            ax.axvline(c["draw_index"], color="red", alpha=0.35, lw=1.0, ls=":")

    plt.tight_layout()
    plot_path = "trse_entropy_probe.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TRSE Entropy Drift Probe — pre-validation test for regime boundary detection"
    )
    parser.add_argument("--file",   default=DEFAULT_FILE,
                        help=f"Path to draw history JSON (default: {DEFAULT_FILE})")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"Sliding window size in draws (default: {DEFAULT_WINDOW})")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help=f"Stride between windows (default: {DEFAULT_STRIDE})")
    parser.add_argument("--z",      type=float, default=DEFAULT_Z,
                        help=f"Z-score threshold for boundary flagging (default: {DEFAULT_Z})")
    parser.add_argument("--plot",   action="store_true",
                        help="Generate entropy drift plot (requires matplotlib)")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help=f"Output JSON path (default: {OUTPUT_FILE})")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}")
        print("Run from ~/distributed_prng_analysis/ or pass --file path/to/daily3.json")
        sys.exit(1)

    print(f"Loading draw history from: {args.file} ...", flush=True)
    draws = load_draws(args.file)
    print(f"Loaded {len(draws):,} draws.")

    if len(draws) < args.window * 2:
        print(f"ERROR: Too few draws ({len(draws)}) for window size {args.window}.")
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"Scanning with window={args.window}, stride={args.stride}, z={args.z} ...",
          flush=True)

    results = detect_boundaries(draws, args.window, args.stride, args.z)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print ─────────────────────────────────────────────────────────────────
    print()
    print(interpret(results))

    # ── Second pass at wider window if noisy ──────────────────────────────────
    if results["candidate_count"] > 8:
        w2, s2 = 800, 100
        print(f"\nAuto-running second pass: window={w2}, stride={s2} ...", flush=True)
        r2 = detect_boundaries(draws, w2, s2, args.z)
        out2 = args.output.replace(".json", "_wide.json")
        with open(out2, "w") as f:
            json.dump(r2, f, indent=2)
        print(interpret(r2))

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
