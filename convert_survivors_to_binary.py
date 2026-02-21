#!/usr/bin/env python3
"""
convert_survivors_to_binary.py - Convert JSON survivors to NPZ binary format

Performance: 88x faster loading (4.2s → 0.05s), ~25x smaller (258MB → ~10MB)

Usage:
    python3 convert_survivors_to_binary.py bidirectional_survivors.json
    python3 convert_survivors_to_binary.py bidirectional_survivors.json --output /tmp/survivors.npz

Version History:
  1.0.0 - Initial (3 arrays only)
  2.0.0 - Added --output flag for atomic write support
  3.0.0 - CRITICAL FIX: Preserve ALL 22 metadata fields (Team Beta Jan 23, 2026)
          Previous versions silently dropped 19 fields, causing 14/47 ML features to be 0.0
  3.1.0 - S103 FIX: forward_matches/reverse_matches now map to per-seed match rates
          (forward_match_rate, reverse_match_rate) written by integration v3.0.
          Previously mapped to forward_count/reverse_count (trial-level aggregates),
          making all quality fields identical for every seed in the same trial.
          These are the surface fingerprint signals that ML uses to rank survivors.
"""
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

VERSION = "3.1.0"

SKIP_MODE_ENCODING = {'constant': 0, 'variable': 1}
PRNG_TYPE_ENCODING = {
    'java_lcg': 0, 'java_lcg_reverse': 1,
    'mt19937': 2, 'mt19937_reverse': 3,
    'xorshift128': 4, 'xorshift128_reverse': 5,
    'lcg32': 6, 'lcg32_reverse': 7,
    'minstd': 8, 'minstd_reverse': 9,
    'randu': 10, 'randu_reverse': 11,
}


def convert_json_to_npz(input_file: str, output_file: str, meta_file: str) -> dict:
    """
    Convert JSON survivors to compressed NPZ format with FULL metadata.

    v3.1 FIX: forward_matches and reverse_matches now correctly map to per-seed
    match rates (forward_match_rate, reverse_match_rate) from the GPU sieve kernel.
    These are genuine per-seed quality signals (0.0-1.0) with real variance across
    the survivor population - the surface fingerprint for ML ranking.

    Requires Step 1 to have been run with window_optimizer_integration_final.py v3.0+
    """
    print(f"Loading {input_file}...")
    with open(input_file) as f:
        survivors = json.load(f)

    n = len(survivors)
    print(f"Loaded {n:,} survivors")

    if n > 0:
        available = set(survivors[0].keys())
        print(f"Available fields: {sorted(available)}")
        if 'forward_match_rate' not in available:
            print("WARNING: forward_match_rate missing - Step 1 integration v3.0+ required")
        if 'reverse_match_rate' not in available:
            print("WARNING: reverse_match_rate missing - Step 1 integration v3.0+ required")

    # === CORE ARRAYS ===
    seeds = np.array([s['seed'] for s in survivors], dtype=np.uint32)

    # v3.1 FIX: per-seed match rates from GPU kernel (0.0-1.0)
    forward_matches = np.array([
        s.get('forward_match_rate', s.get('reverse_match_rate', 0.0))
        for s in survivors
    ], dtype=np.float32)

    reverse_matches = np.array([
        s.get('reverse_match_rate', s.get('forward_match_rate', 0.0))
        for s in survivors
    ], dtype=np.float32)

    # === METADATA ARRAYS (v3.0) ===
    window_size = np.array([s.get('window_size', 0) for s in survivors], dtype=np.int32)
    offset = np.array([s.get('offset', 0) for s in survivors], dtype=np.int32)
    trial_number = np.array([s.get('trial_number', 0) for s in survivors], dtype=np.int32)
    skip_min = np.array([s.get('skip_min', 0) for s in survivors], dtype=np.int32)
    skip_max = np.array([s.get('skip_max', 0) for s in survivors], dtype=np.int32)
    skip_range = np.array([s.get('skip_range', 0) for s in survivors], dtype=np.int32)

    # Trial-level context (retained for reference)
    forward_count = np.array([s.get('forward_count', 0.0) for s in survivors], dtype=np.float32)
    reverse_count = np.array([s.get('reverse_count', 0.0) for s in survivors], dtype=np.float32)
    bidirectional_count = np.array([s.get('bidirectional_count', 0.0) for s in survivors], dtype=np.float32)

    # Per-seed score: avg(fwd_rate, rev_rate) in v3.0+, trial count in older data
    score = np.array([s.get('score', 0.0) for s in survivors], dtype=np.float32)

    intersection_count = np.array([s.get('intersection_count', 0.0) for s in survivors], dtype=np.float32)
    intersection_ratio = np.array([s.get('intersection_ratio', 0.0) for s in survivors], dtype=np.float32)
    intersection_weight = np.array([s.get('intersection_weight', 0.0) for s in survivors], dtype=np.float32)
    bidirectional_selectivity = np.array([s.get('bidirectional_selectivity', 0.0) for s in survivors], dtype=np.float32)
    forward_only_count = np.array([s.get('forward_only_count', 0.0) for s in survivors], dtype=np.float32)
    reverse_only_count = np.array([s.get('reverse_only_count', 0.0) for s in survivors], dtype=np.float32)
    survivor_overlap_ratio = np.array([s.get('survivor_overlap_ratio', 0.0) for s in survivors], dtype=np.float32)

    skip_mode = np.array([
        SKIP_MODE_ENCODING.get(s.get('skip_mode', 'constant'), 0)
        for s in survivors
    ], dtype=np.uint8)

    prng_type = np.array([
        PRNG_TYPE_ENCODING.get(s.get('prng_type', s.get('prng_base', 'java_lcg')), 0)
        for s in survivors
    ], dtype=np.uint8)

    # === VERIFY variance ===
    fwd_unique = len(set(forward_matches.tolist()))
    rev_unique = len(set(reverse_matches.tolist()))
    print(f"\n📊 forward_matches: min={forward_matches.min():.4f} max={forward_matches.max():.4f} unique={fwd_unique}")
    print(f"📊 reverse_matches: min={reverse_matches.min():.4f} max={reverse_matches.max():.4f} unique={rev_unique}")
    # Warn if unique values < 10% of survivors (suggests trial-level aggregates)
    if n > 0 and fwd_unique < max(3, n * 0.10):
        print(f"⚠️  WARNING: Low variance ({fwd_unique} unique values for {n} survivors) - check Step 1 integration version")
    else:
        print(f"✅ Good per-seed variance ({fwd_unique} unique values for {n} survivors)")

    # === SAVE ===
    print(f"\nSaving {output_file}...")
    np.savez_compressed(
        output_file,
        seeds=seeds,
        forward_matches=forward_matches,
        reverse_matches=reverse_matches,
        window_size=window_size,
        offset=offset,
        trial_number=trial_number,
        skip_min=skip_min,
        skip_max=skip_max,
        skip_range=skip_range,
        forward_count=forward_count,
        reverse_count=reverse_count,
        bidirectional_count=bidirectional_count,
        intersection_count=intersection_count,
        intersection_ratio=intersection_ratio,
        intersection_weight=intersection_weight,
        bidirectional_selectivity=bidirectional_selectivity,
        forward_only_count=forward_only_count,
        reverse_only_count=reverse_only_count,
        survivor_overlap_ratio=survivor_overlap_ratio,
        score=score,
        skip_mode=skip_mode,
        prng_type=prng_type,
    )

    input_size = Path(input_file).stat().st_size
    output_size = Path(output_file).stat().st_size
    ratio = input_size / output_size if output_size > 0 else 0

    metadata = {
        "version": VERSION,
        "source_file": str(Path(input_file).resolve()),
        "output_file": str(Path(output_file).resolve()),
        "survivor_count": n,
        "forward_matches_source": "forward_match_rate per-seed (v3.1+)",
        "reverse_matches_source": "reverse_match_rate per-seed (v3.1+)",
        "arrays": {
            "core": ["seeds", "forward_matches", "reverse_matches"],
            "metadata_int": ["window_size", "offset", "trial_number", "skip_min", "skip_max", "skip_range"],
            "metadata_float": [
                "forward_count", "reverse_count", "bidirectional_count",
                "intersection_count", "intersection_ratio", "intersection_weight",
                "bidirectional_selectivity", "forward_only_count", "reverse_only_count",
                "survivor_overlap_ratio", "score"
            ],
            "categorical": ["skip_mode", "prng_type"]
        },
        "array_count": 22,
        "input_size_bytes": input_size,
        "output_size_bytes": output_size,
        "compression_ratio": ratio,
        "converted_at": datetime.now().isoformat(),
        "encodings": {
            "skip_mode": SKIP_MODE_ENCODING,
            "prng_type": PRNG_TYPE_ENCODING
        }
    }

    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Conversion complete (v{VERSION}):")
    print(f"  Input:  {input_size:,} bytes ({input_size/1024/1024:.1f} MB)")
    print(f"  Output: {output_size:,} bytes ({output_size/1024:.1f} KB)")
    print(f"  Ratio:  {ratio:.1f}x compression")
    print(f"  Meta:   {meta_file}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description=f"Convert JSON survivors to NPZ binary format (v{VERSION})"
    )
    parser.add_argument("input_file", help="Input JSON file")
    parser.add_argument("--output", "-o", help="Output NPZ file (default: input_binary.npz)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_binary.npz"

    meta_path = output_path.parent / f"{output_path.stem}.meta.json"
    convert_json_to_npz(str(input_path), str(output_path), str(meta_path))
    return 0


if __name__ == "__main__":
    exit(main())
