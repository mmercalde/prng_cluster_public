#!/usr/bin/env python3
"""
Build prediction pools from sieve survivors.

Inputs:
  - survivors JSON(s) from residue_sieve/coordinator (glob supported)
    Expected minimal fields per file:
      {
        "version": 1,
        "run_id": "...",
        "survivors": [<seed>, ...],                 # required
        "per_seed": {                               # optional meta per seed
            "<seed>": {
                "composite": <float>,               # optional, default 1.0
                "offset_conf": <float>,             # optional, default 0.5
                "gap_stability": <float>,          # optional, default 0.5
                "best_skip": <int>                 # optional, default 0
            },
            ...
        },
        "fit_draw_count": <int>                     # optional, default 0
      }

Outputs (prediction_pools.json):
  {
    "version": 1,
    "run_id": "ISO-8601 timestamp",
    "source_runs": ["..."],
    "pool_sizes": [20,100,300],
    "survivor_count": N,
    "mapping": "mod1000",
    "prng_type": "xorshift32",
    "weighting": {"composite": 0.6, "offset_conf": 0.25, "gap_stability": 0.15},
    "ranked": [[number, weight_share], ...],
    "pools": {
      "20": {"numbers": [...], "coverage": 0.28},
      "100": {"numbers": [...], "coverage": 0.62},
      "300": {"numbers": [...], "coverage": 0.88}
    }
  }

Notes:
- This script includes a minimal PRNG (xorshift32) for next-step prediction.
- If you have a canonical PRNG implementation elsewhere, replace `prng_next_batch` with your adapter.
"""
import argparse
import glob
import json
import os
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Tuple

# ----------------------------
# Minimal PRNGs (extend as needed)
# ----------------------------

def xorshift32_step(state: int) -> int:
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def prng_next_batch(prng_type: str, seeds: List[int], steps: int) -> List[int]:
    """Return raw PRNG values after advancing each seed by `steps`.
    Replace/extend with your cluster's canonical PRNG implementation.
    """
    if prng_type.lower() == "xorshift32":
        out = []
        for s in seeds:
            x = s & 0xFFFFFFFF
            for _ in range(steps):
                x = xorshift32_step(x)
            out.append(x)
        return out
    raise ValueError(f"Unsupported prng_type: {prng_type}")


# ----------------------------
# Utilities
# ----------------------------

def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_survivor_files(patterns: List[str]) -> Tuple[List[int], Dict[str, dict], List[str], int]:
    """Load survivor JSONs. Supports native and coordinator schemas.
       Returns (seeds, per_seed_meta, run_ids, fit_draw_count)."""
    seeds_set = set()
    per_seed: Dict[str, dict] = {}
    run_ids: List[str] = []
    fit_draw_count = 0

    files: List[str] = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise FileNotFoundError(f"No input files match: {patterns}")

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Coordinator format:
        # {
        #   "results": [
        #     {"survivors": [{"seed":42,"family":"xorshift32","match_rate":1.0,"best_skip":5}, ...]}
        #   ]
        # }
        if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
            run_ids.append(data.get("run_id", os.path.basename(fp)))
            for result in data.get("results", []):
                for surv in result.get("survivors", []):
                    seed = int(surv.get("seed"))
                    seeds_set.add(seed)
                    meta = per_seed.get(str(seed), {}).copy() if str(seed) in per_seed else {}
                    meta.update({
                        "best_skip": int(surv.get("best_skip", 0)),
                        "composite": float(surv.get("match_rate", 1.0)),
                        "family": str(surv.get("family", "unknown")),
                        # temporary fallbacks so weighting works today
                        "offset_conf": float(surv.get("match_rate", 0.5)),
                        "gap_stability": 1.0 / (1 + int(surv.get("best_skip", 0))),
                    })
                    per_seed[str(seed)] = meta
            continue  # skip native parsing for this file

        # Native schema:
        # {"survivors":[42,...], "per_seed":{"42":{"best_skip":5,...}}, "fit_draw_count":...}
        run_ids.append(data.get("run_id", os.path.basename(fp)))
        fit_draw_count = max(fit_draw_count, int(data.get("fit_draw_count", 0)))
        surv = data.get("survivors") or data.get("survivor_seeds") or []
        for s in surv:
            seeds_set.add(int(s))
        meta = data.get("per_seed", {})
        for k, v in meta.items():
            per_seed[str(int(k))] = v

    return sorted(seeds_set), per_seed, run_ids, fit_draw_count


def map_to_draw_space(raw: int, mapping: str = "mod1000", ts: int = 0) -> int:
    if mapping == "mod1000":
        return int(raw % 1000)
    if mapping == "xor_ts_mod1000":
        return int((raw ^ ts) % 1000)
    raise ValueError(f"Unknown mapping: {mapping}")


def compute_weight(seed: int, per_seed: Dict[str, dict], w_comp: float, w_off: float, w_gap: float) -> float:
    m = per_seed.get(str(seed), {})
    base = float(m.get("composite", m.get("match_rate", 1.0)))
    comp = float(m.get("composite", base))
    offc = float(m.get("offset_conf", base))  # fallback to match_rate if missing
    gap  = float(m.get("gap_stability", 1.0 / (1 + int(m.get("best_skip", 0)))))  # derived if missing

    # clamp to [0,1]
    comp = max(0.0, min(1.0, comp))
    offc = max(0.0, min(1.0, offc))
    gap  = max(0.0, min(1.0, gap))

    return w_comp * comp + w_off * offc + w_gap * gap


def main():
    ap = argparse.ArgumentParser(description="Build prediction pools from sieve survivors.")
    ap.add_argument("--survivors", nargs="+", required=True, help="Path(s) or glob(s) to results/multi_gpu_analysis_*.json")
    ap.add_argument("--prng-type", default="xorshift32", help="PRNG type (default: xorshift32)")
    ap.add_argument("--mapping", default="mod1000", help="Mapping: mod1000 | xor_ts_mod1000")
    ap.add_argument("--pools", default="20,100,300", help="Comma-separated pool sizes, e.g., 20,100,300")
    ap.add_argument("--out", default="prediction_pools.json", help="Output JSON path")
    ap.add_argument("--best-skip-field", default="best_skip", help="Per-seed field name for best skip")
    ap.add_argument("--fit-draw-count", type=int, default=None, help="Override fit_draw_count if not present in file(s)")
    ap.add_argument("--next-timestamp", type=int, default=0, help="Timestamp for xor_ts_mod1000 mapping")
    # Weights
    ap.add_argument("--w-comp", type=float, default=0.6, help="Weight for composite score")
    ap.add_argument("--w-off", type=float, default=0.25, help="Weight for offset confidence")
    ap.add_argument("--w-gap", type=float, default=0.15, help="Weight for gap stability")
    args = ap.parse_args()

    pool_sizes = [int(x) for x in args.pools.split(",") if x.strip()]

    seeds, per_seed, run_ids, fit_draw_ct = load_survivor_files(args.survivors)
    if args.fit_draw_count is not None:
        fit_draw_ct = args.fit_draw_count

    # Handle edge case: no survivors
    if len(seeds) == 0:
        out = {
            "version": 1,
            "run_id": iso_now(),
            "source_runs": run_ids,
            "pool_sizes": pool_sizes,
            "survivor_count": 0,
            "mapping": args.mapping,
            "prng_type": args.prng_type,
            "weighting": {"composite": args.w_comp, "offset_conf": args.w_off, "gap_stability": args.w_gap},
            "ranked": [],
            "pools": {str(k): {"numbers": [], "coverage": 0.0} for k in pool_sizes}
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("No survivors found - wrote empty pools")
        return 0

    # Determine steps to advance per seed: best_skip + 1 (fallback 1)
    steps_per_seed = []
    for s in seeds:
        meta = per_seed.get(str(s), {})
        bs = int(meta.get(args.best_skip_field, meta.get("skip", 0)))
        steps_per_seed.append(max(1, bs + 1))

    # Advance each seed by its own steps and map to draw space
    raw_next = []
    for s, st in zip(seeds, steps_per_seed):
        val = s
        for _ in range(st):
            if args.prng_type.lower() == "xorshift32":
                val = xorshift32_step(val)
            else:
                # fall back to batch function (can be extended)
                val = prng_next_batch(args.prng_type, [val], 1)[0]
        raw_next.append(val)

    # Map to 0..999
    pred_numbers = [map_to_draw_space(v, args.mapping, args.next_timestamp) for v in raw_next]

    # Aggregate weighted votes
    weights = defaultdict(float)
    total_weight = 0.0
    for seed, num in zip(seeds, pred_numbers):
        w = compute_weight(seed, per_seed, args.w_comp, args.w_off, args.w_gap)
        weights[num] += w
        total_weight += w
    if total_weight <= 0:
        total_weight = 1.0

    # Deterministic sort: descending weight, then ascending number for ties
    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))

    # Build pools and coverages
    pools = {}
    for k in pool_sizes:
        topk = ranked[:k]
        cov = sum(w for _, w in topk) / total_weight if ranked else 0.0
        pools[str(k)] = {
            "numbers": [int(n) for n, _ in topk],
            "coverage": round(cov, 3),
        }

    out = {
        "version": 1,
        "run_id": iso_now(),
        "source_runs": run_ids,
        "pool_sizes": pool_sizes,
        "survivor_count": len(seeds),
        "mapping": args.mapping,
        "prng_type": args.prng_type,
        "weighting": {"composite": args.w_comp, "offset_conf": args.w_off, "gap_stability": args.w_gap},
        "ranked": [[int(n), round(w/total_weight, 6)] for n, w in ranked],
        "pools": pools,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Console summary
    cov_summary = {k: f"{v['coverage']*100:.1f}%" for k, v in pools.items()}
    print(f"Survivors={len(seeds)} | Coverage: {cov_summary}")
    
    # Print top 10 for quick eyeballing
    if ranked:
        print("\nTop 10 predictions:")
        for n, w in ranked[:10]:
            print(f"  {n:03d}\t{w/total_weight:.3%}")
    
    return 0


if __name__ == "__main__":
    exit(main())
