#!/usr/bin/env python3
"""
digit_sequential_sieve.py
=========================
Digit-Sequential Filter for PRNG Survivor Analysis

Instead of testing: prng_output % 1000 == draw_value
Tests:              three sequential generator calls
                    call1 % 10 == hundreds_digit
                    call2 % 10 == tens_digit
                    call3 % 10 == ones_digit

Input:  results/window_opt_forward_8_43_t1.json  (2016 survivors)
        data/daily3.json                          (draw history)

Output: digit_sequential_survivors.json

Usage:
    python3 digit_sequential_sieve.py [--base-dir ~/distributed_prng_analysis]
                                      [--threshold 3]
                                      [--require-all]

Threshold modes:
    --threshold 3   : all 3 digits must match (strict)
    --threshold 2   : at least 2 of 3 digits must match (relaxed)
    --require-all   : shorthand for --threshold 3

Author: Team Alpha
Session: S134
"""

import json
import argparse
import os
from pathlib import Path


# ─── java_lcg parameters (from prng_registry.py) ─────────────────────────────
A = 25214903917
C = 11
M = (1 << 48)  # 2^48


def lcg_step(state: int) -> tuple[int, int]:
    """Single LCG step. Returns (new_state, output)."""
    state = (A * state + C) & (M - 1)
    output = (state >> 16) & 0xFFFFFFFF
    return state, output


def run_to_offset(seed: int, offset: int) -> int:
    """Advance generator from seed through `offset` steps, return state."""
    state = seed & (M - 1)
    for _ in range(offset):
        state = (A * state + C) & (M - 1)
    return state


def test_seed_digit_sequential(
    seed: int,
    best_skip: int,
    draws: list[int],
    window_size: int,
    offset: int,
    digit_threshold: int,
) -> dict:
    """
    Test a single seed using digit-sequential matching.

    For each draw, advance generator 3 times:
        state1 → hundreds digit = (output1 % 10)
        state2 → tens digit     = (output2 % 10)
        state3 → ones digit     = (output3 % 10)

    Between draws, skip `best_skip` extra steps (matching original pipeline).

    Returns dict with match stats.
    """
    # Advance to offset position
    state = run_to_offset(seed, offset)

    # Apply best_skip before first draw
    for _ in range(best_skip):
        state = (A * state + C) & (M - 1)

    window_draws = draws[:window_size]

    total_draws = 0
    full_matches = 0      # all 3 digits correct
    partial_matches = 0   # >= digit_threshold digits correct
    per_draw_details = []

    for draw_val in window_draws:
        # Extract expected digits
        h_exp = (draw_val // 100) % 10
        t_exp = (draw_val // 10) % 10
        o_exp = draw_val % 10

        # Three sequential generator calls
        state, out1 = lcg_step(state)
        h_got = out1 % 10

        state, out2 = lcg_step(state)
        t_got = out2 % 10

        state, out3 = lcg_step(state)
        o_got = out3 % 10

        # Count digit matches
        digit_hits = (
            (1 if h_got == h_exp else 0) +
            (1 if t_got == t_exp else 0) +
            (1 if o_got == o_exp else 0)
        )

        if digit_hits == 3:
            full_matches += 1
        if digit_hits >= digit_threshold:
            partial_matches += 1

        per_draw_details.append({
            "draw": draw_val,
            "predicted": h_got * 100 + t_got * 10 + o_got,
            "digit_hits": digit_hits,
        })

        total_draws += 1

        # Skip between draws
        for _ in range(best_skip):
            state = (A * state + C) & (M - 1)

    return {
        "total_draws": total_draws,
        "full_matches": full_matches,          # all 3 digits
        "partial_matches": partial_matches,    # >= threshold digits
        "full_match_rate": full_matches / total_draws if total_draws > 0 else 0.0,
        "partial_match_rate": partial_matches / total_draws if total_draws > 0 else 0.0,
        "details": per_draw_details,
    }


def load_draws(draws_path: str, session: str = None) -> list[int]:
    """Load daily3.json and return list of integer draw values.
    
    If session is specified (e.g. 'midday'), filter to that session only.
    """
    with open(draws_path) as f:
        data = json.load(f)

    draws = []
    # Handle various formats
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                # Session filter
                if session and entry.get("session", "") != session:
                    continue
                # Try common field names
                val = (entry.get("draw") or entry.get("value") or
                       entry.get("result") or entry.get("number"))
                if val is not None:
                    draws.append(int(val))
            elif isinstance(entry, (int, float)):
                draws.append(int(entry))
    elif isinstance(data, dict):
        # Try 'draws' key
        raw = data.get("draws") or data.get("results") or []
        for entry in raw:
            if isinstance(entry, dict):
                val = (entry.get("draw") or entry.get("value") or
                       entry.get("result") or entry.get("number"))
                if val is not None:
                    draws.append(int(val))
            elif isinstance(entry, (int, float)):
                draws.append(int(entry))

    if not draws:
        raise ValueError(f"Could not parse draws from {draws_path}. "
                         f"Check file structure. First 200 chars: "
                         f"{str(data)[:200]}")
    return draws


def main():
    parser = argparse.ArgumentParser(description="Digit-Sequential PRNG Sieve")
    parser.add_argument(
        "--base-dir",
        default=os.path.expanduser("~/distributed_prng_analysis"),
        help="Project base directory on Zeus",
    )
    parser.add_argument(
        "--survivors-file",
        default=None,
        help="Path to forward survivor JSON (default: results/window_opt_forward_8_43_t1.json)",
    )
    parser.add_argument(
        "--draws-file",
        default=None,
        help="Path to draws JSON (default: data/daily3.json)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Number of draws to test (default: 8, matches W8_O43)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=43,
        help="Draw offset into history (default: 43, matches W8_O43)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Minimum digit matches required per draw (default: 3 = all digits)",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Require all 3 digits to match (same as --threshold 3)",
    )
    parser.add_argument(
        "--session",
        default="midday",
        help="Filter draws by session (default: midday, matches optimal config). Use 'all' for no filter.",
    )
    parser.add_argument(
        "--skip-min",
        type=int,
        default=5,
        help="Min inter-draw skip to test (default: 5, matches original trial)",
    )
    parser.add_argument(
        "--skip-max",
        type=int,
        default=56,
        help="Max inter-draw skip to test (default: 56, matches original trial)",
    )
    parser.add_argument(
        "--min-draw-hits",
        type=int,
        default=2,
        help="Minimum draws that must pass digit threshold to be a survivor (default: 2)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: results/digit_sequential_survivors.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-seed details",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    survivors_file = Path(args.survivors_file) if args.survivors_file else \
        base / "results" / "window_opt_forward_8_43_t1.json"
    draws_file = Path(args.draws_file) if args.draws_file else \
        base / "data" / "daily3.json"
    output_file = Path(args.output) if args.output else \
        base / "results" / "digit_sequential_survivors.json"

    digit_threshold = 3 if args.require_all else args.threshold

    print(f"=== Digit-Sequential Sieve ===")
    print(f"Survivors input : {survivors_file}")
    print(f"Draws input     : {draws_file}")
    print(f"Window          : W{args.window_size}_O{args.offset}")
    print(f"Digit threshold : {digit_threshold}/3 per draw")
    print(f"Min draw hits   : {args.min_draw_hits}/{args.window_size}")
    print(f"Output          : {output_file}")
    print()

    # ── Load inputs ──────────────────────────────────────────────────────────
    print(f"Loading survivors from {survivors_file}...")
    with open(survivors_file) as f:
        fwd_data = json.load(f)
    candidates = fwd_data.get("survivors", [])
    print(f"  {len(candidates)} candidates loaded")

    print(f"Loading draws from {draws_file}...")
    session_filter = None if args.session == "all" else args.session
    draws = load_draws(str(draws_file), session=session_filter)
    print(f"  {len(draws)} draws loaded (session={args.session})")
    print(f"  Using draws[{args.offset}:{args.offset + args.window_size}] = "
          f"{draws[args.offset:args.offset + args.window_size]}")
    print()

    window_draws = draws[args.offset: args.offset + args.window_size]
    if len(window_draws) < args.window_size:
        raise ValueError(f"Not enough draws: need {args.window_size}, "
                         f"got {len(window_draws)} starting at offset {args.offset}")

    # ── Run digit-sequential filter ──────────────────────────────────────────
    ds_survivors = []
    all_results = []

    # Normalize candidates — may be raw ints or dicts
    skip_min = args.skip_min
    skip_max = args.skip_max
    match_rates = fwd_data.get("match_rates", [])

    for i, cand in enumerate(candidates):
        if isinstance(cand, dict):
            seed = cand["seed"]
            family = cand.get("family", "java_lcg")
            mod1000_match_rate = cand.get("match_rate", 0)
            mod1000_matches = cand.get("matches", 0)
        else:
            seed = int(cand)
            family = fwd_data.get("prng_type", "java_lcg")
            mod1000_match_rate = match_rates[i] if i < len(match_rates) else 0
            mod1000_matches = 0

        # Test all skips in range, keep best
        best_result = None
        best_skip_used = skip_min
        for skip in range(skip_min, skip_max + 1):
            result = test_seed_digit_sequential(
                seed=seed,
                best_skip=skip,
                draws=window_draws,
                window_size=args.window_size,
                offset=args.offset,
                digit_threshold=digit_threshold,
            )
            if best_result is None or result["partial_matches"] > best_result["partial_matches"]:
                best_result = result
                best_skip_used = skip

        result = best_result
        passes = result["partial_matches"] >= args.min_draw_hits

        record = {
            "seed": seed,
            "family": family,
            "best_skip": best_skip_used,
            "mod1000_match_rate": mod1000_match_rate,
            "mod1000_matches": mod1000_matches,
            "ds_full_matches": result["full_matches"],
            "ds_partial_matches": result["partial_matches"],
            "ds_full_match_rate": round(result["full_match_rate"], 4),
            "ds_partial_match_rate": round(result["partial_match_rate"], 4),
            "ds_passes": passes,
        }

        all_results.append(record)
        if passes:
            ds_survivors.append(record)

        if args.verbose and passes:
            print(f"  PASS  seed={seed:>10}  skip={best_skip:>2}  "
                  f"full={result['full_matches']}/{args.window_size}  "
                  f"partial={result['partial_matches']}/{args.window_size}")

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(candidates)}  "
                  f"DS survivors so far: {len(ds_survivors)}")

    print()
    print(f"=== Results ===")
    print(f"Input candidates      : {len(candidates)}")
    print(f"DS survivors          : {len(ds_survivors)}")
    print(f"Digit threshold used  : {digit_threshold}/3")
    print(f"Min draw hits         : {args.min_draw_hits}")

    # ── Intersection analysis ─────────────────────────────────────────────────
    # Which DS survivors also have mod1000_matches >= 2 (i.e. were in original 85)?
    # We don't have the 85 list here directly but we can note high mod1000 scorers
    high_mod1000 = [r for r in ds_survivors if r["mod1000_matches"] >= 3]
    print(f"DS survivors with mod1000_matches>=3 : {len(high_mod1000)}")
    if high_mod1000:
        print("  Seeds:", [r["seed"] for r in high_mod1000])

    # Distribution of DS full matches
    dist = {}
    for r in all_results:
        k = r["ds_full_matches"]
        dist[k] = dist.get(k, 0) + 1
    print(f"\nDistribution of full digit matches (all 3/draw):")
    for k in sorted(dist.keys(), reverse=True):
        bar = "█" * min(dist[k] // 5, 60)
        print(f"  {k}/{args.window_size} draws : {dist[k]:>5}  {bar}")

    # ── Save output ──────────────────────────────────────────────────────────
    output_data = {
        "meta": {
            "window_size": args.window_size,
            "offset": args.offset,
            "digit_threshold": digit_threshold,
            "min_draw_hits": args.min_draw_hits,
            "input_candidates": len(candidates),
            "ds_survivor_count": len(ds_survivors),
            "draws_tested": window_draws,
        },
        "ds_survivors": ds_survivors,
        "all_results_summary": [
            {k: v for k, v in r.items() if k != "details"}
            for r in all_results
        ],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nOutput saved: {output_file}")


if __name__ == "__main__":
    main()
