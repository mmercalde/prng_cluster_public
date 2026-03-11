#!/usr/bin/env python3
"""
ds_cross_compare.py
===================
Cross-compare digit-sequential survivors against all 4 sieve passes:
  - forward constant skip
  - forward hybrid (variable skip)
  - reverse constant skip
  - reverse hybrid (variable skip)

Reports which seeds appear in BOTH digit-sequential AND any sieve pass.

Usage:
    python3 ds_cross_compare.py [--base-dir ~/distributed_prng_analysis]
                                [--session all]
                                [--threshold 2]
                                [--min-draw-hits 2]

Session S134
"""

import json
import argparse
import os
from pathlib import Path

A = 25214903917
C = 11
M = (1 << 48)


def lcg_step(state):
    state = (A * state + C) & (M - 1)
    return state, (state >> 16) & 0xFFFFFFFF


def load_draws(path, session=None):
    with open(path) as f:
        data = json.load(f)
    draws = []
    if isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                if session and e.get("session", "") != session:
                    continue
                val = e.get("draw") or e.get("value") or e.get("result")
                if val is not None:
                    draws.append(int(val))
            elif isinstance(e, (int, float)):
                draws.append(int(e))
    return draws


def load_sieve_file(path):
    """Load survivors from any of the 4 sieve output files. Returns set of seed ints."""
    with open(path) as f:
        d = json.load(f)
    raw = d.get("survivors", [])
    seeds = set()
    for s in raw:
        if isinstance(s, dict):
            seeds.add(int(s.get("seed", s)))
        else:
            seeds.add(int(s))
    return seeds


def test_seed_ds(seed, skip, draws, digit_threshold):
    """Test one seed+skip combo. Returns (full_matches, partial_matches)."""
    state = seed & (M - 1)
    for _ in range(skip):
        state = (A * state + C) & (M - 1)

    full = 0
    partial = 0
    for draw in draws:
        h_exp = (draw // 100) % 10
        t_exp = (draw // 10) % 10
        o_exp = draw % 10

        state, o1 = lcg_step(state)
        state, o2 = lcg_step(state)
        state, o3 = lcg_step(state)

        hits = (int(o1 % 10 == h_exp) +
                int(o2 % 10 == t_exp) +
                int(o3 % 10 == o_exp))

        if hits == 3:
            full += 1
        if hits >= digit_threshold:
            partial += 1

        for _ in range(skip):
            state = (A * state + C) & (M - 1)

    return full, partial


def run_ds_on_seed_set(seeds, draws, skip_min, skip_max,
                       digit_threshold, min_draw_hits):
    """
    Run DS filter on a set of seeds.
    Returns dict: seed -> {best_skip, full_matches, partial_matches}
    for seeds that pass min_draw_hits.
    """
    survivors = {}
    for seed in seeds:
        best_full = 0
        best_partial = 0
        best_skip = skip_min
        for skip in range(skip_min, skip_max + 1):
            f, p = test_seed_ds(seed, skip, draws, digit_threshold)
            if p > best_partial or (p == best_partial and f > best_full):
                best_partial = p
                best_full = f
                best_skip = skip
        if best_partial >= min_draw_hits:
            survivors[seed] = {
                "best_skip": best_skip,
                "full_matches": best_full,
                "partial_matches": best_partial,
            }
    return survivors


def main():
    parser = argparse.ArgumentParser(description="DS Cross-Compare vs 4 Sieve Files")
    parser.add_argument("--base-dir", default=os.path.expanduser(
        "~/distributed_prng_analysis"))
    parser.add_argument("--draws-file", default=None)
    parser.add_argument("--session", default="all",
                        help="Session filter: 'midday', 'evening', 'all' (default: all)")
    parser.add_argument("--offset", type=int, default=43)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--skip-min", type=int, default=5)
    parser.add_argument("--skip-max", type=int, default=56)
    parser.add_argument("--digit-threshold", type=int, default=3,
                        choices=[1, 2, 3],
                        help="Min digit matches per draw (default 3=all)")
    parser.add_argument("--min-draw-hits", type=int, default=2,
                        help="Min draws that must pass digit threshold (default 2)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = Path(args.base_dir)
    draws_file = Path(args.draws_file) if args.draws_file else base / "daily3.json"
    output_file = Path(args.output) if args.output else \
        base / "results" / "ds_cross_compare.json"

    session_filter = None if args.session == "all" else args.session

    # ── Load draws ────────────────────────────────────────────────────────────
    draws_all = load_draws(str(draws_file), session=session_filter)
    window_draws = draws_all[args.offset: args.offset + args.window_size]
    print(f"=== DS Cross-Compare ===")
    print(f"Session filter : {args.session}")
    print(f"Window         : W{args.window_size}_O{args.offset}")
    print(f"Draws          : {window_draws}")
    print(f"Digit threshold: {args.digit_threshold}/3 per draw")
    print(f"Min draw hits  : {args.min_draw_hits}/{args.window_size}")
    print(f"Skip range     : {args.skip_min}-{args.skip_max}")
    print()

    # ── Load all 4 sieve files ────────────────────────────────────────────────
    sieve_files = {
        "forward_constant":  base / "results" / "window_opt_forward_8_43_t1.json",
        "forward_hybrid":    base / "results" / "window_opt_forward_hybrid_8_43_t1.json",
        "reverse_constant":  base / "results" / "window_opt_reverse_8_43_t1.json",
        "reverse_hybrid":    base / "results" / "window_opt_reverse_hybrid_8_43_t1.json",
    }

    sieve_sets = {}
    all_sieve_seeds = set()
    for name, path in sieve_files.items():
        seeds = load_sieve_file(str(path))
        sieve_sets[name] = seeds
        all_sieve_seeds |= seeds
        print(f"  {name:25s}: {len(seeds):>5} seeds")

    print(f"  {'UNION (all 4)':25s}: {len(all_sieve_seeds):>5} seeds")
    print()

    # Bidirectional intersection (forward_constant ∩ reverse_constant)
    bidi_constant = sieve_sets["forward_constant"] & sieve_sets["reverse_constant"]
    bidi_hybrid   = sieve_sets["forward_hybrid"]   & sieve_sets["reverse_hybrid"]
    bidi_all      = sieve_sets["forward_constant"] & sieve_sets["reverse_constant"] & \
                    sieve_sets["forward_hybrid"]   & sieve_sets["reverse_hybrid"]
    print(f"  {'bidi_constant (F∩R)':25s}: {len(bidi_constant):>5} seeds")
    print(f"  {'bidi_hybrid (FH∩RH)':25s}: {len(bidi_hybrid):>5} seeds")
    print(f"  {'bidi_all (F∩R∩FH∩RH)':25s}: {len(bidi_all):>5} seeds")
    print()

    # ── Run DS on UNION of all sieve seeds ────────────────────────────────────
    print(f"Running DS filter on {len(all_sieve_seeds)} unique seeds "
          f"(skip {args.skip_min}-{args.skip_max})...")
    ds_survivors = run_ds_on_seed_set(
        seeds=all_sieve_seeds,
        draws=window_draws,
        skip_min=args.skip_min,
        skip_max=args.skip_max,
        digit_threshold=args.digit_threshold,
        min_draw_hits=args.min_draw_hits,
    )
    ds_seed_set = set(ds_survivors.keys())
    print(f"DS survivors: {len(ds_seed_set)}")
    print()

    # ── Cross-compare ─────────────────────────────────────────────────────────
    print("=== Intersection Analysis ===")
    results = {}
    for name, sieve_set in sieve_sets.items():
        intersection = ds_seed_set & sieve_set
        results[name] = sorted(intersection)
        print(f"  DS ∩ {name:25s}: {len(intersection):>4} seeds  {sorted(intersection)[:10]}")

    # Special intersections
    ds_bidi_constant = ds_seed_set & bidi_constant
    ds_bidi_hybrid   = ds_seed_set & bidi_hybrid
    ds_bidi_all      = ds_seed_set & bidi_all
    ds_any_sieve     = ds_seed_set & all_sieve_seeds

    print()
    print(f"  DS ∩ bidi_constant          : {len(ds_bidi_constant):>4} seeds  {sorted(ds_bidi_constant)}")
    print(f"  DS ∩ bidi_hybrid            : {len(ds_bidi_hybrid):>4} seeds  {sorted(ds_bidi_hybrid)}")
    print(f"  DS ∩ bidi_all               : {len(ds_bidi_all):>4} seeds  {sorted(ds_bidi_all)}")
    print(f"  DS ∩ any_sieve (union)      : {len(ds_any_sieve):>4} seeds  {sorted(ds_any_sieve)[:20]}")

    # ── Save output ───────────────────────────────────────────────────────────
    output_data = {
        "meta": {
            "session": args.session,
            "window_size": args.window_size,
            "offset": args.offset,
            "window_draws": window_draws,
            "digit_threshold": args.digit_threshold,
            "min_draw_hits": args.min_draw_hits,
            "skip_range": [args.skip_min, args.skip_max],
        },
        "sieve_counts": {k: len(v) for k, v in sieve_sets.items()},
        "sieve_union_count": len(all_sieve_seeds),
        "bidi_constant_count": len(bidi_constant),
        "bidi_hybrid_count": len(bidi_hybrid),
        "bidi_all_count": len(bidi_all),
        "ds_survivor_count": len(ds_seed_set),
        "ds_survivors_detail": {
            str(s): ds_survivors[s] for s in sorted(ds_seed_set)
        },
        "intersections": {
            k: sorted(v) for k, v in results.items()
        },
        "ds_bidi_constant": sorted(ds_bidi_constant),
        "ds_bidi_hybrid": sorted(ds_bidi_hybrid),
        "ds_bidi_all": sorted(ds_bidi_all),
        "ds_any_sieve": sorted(ds_any_sieve),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nOutput saved: {output_file}")


if __name__ == "__main__":
    main()
