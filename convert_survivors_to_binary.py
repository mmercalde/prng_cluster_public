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

# [S172 Phase-5 D3.0] Canonical encoding seam. The local 12-entry
# PRNG_TYPE_ENCODING / SKIP_MODE_ENCODING tables that used to live here were a
# pre-Phase-0 fork of the registry: they disagreed with canonical on seven
# shared keys, lacked 'java_lcg_hybrid' entirely (so `.get(..., 0)` silently
# relabelled every hybrid survivor as java_lcg), and carried two ids —
# randu(10)/randu_reverse(11) — that are not in KERNEL_REGISTRY at all.
# utils/prng_encoding is now the single source of truth and hard-fails on an
# unknown identity; the ValueError it raises propagates unwrapped by design.
from utils.prng_encoding import (  # noqa: E402
    PRNG_TYPE_ENCODING,
    SKIP_MODE_ENCODING,
    encode_prng_type,
    encode_skip_mode,
)

# [S172 Phase-5 D3.0] Frozen on-disk NPZ contract — the 22 arrays in their
# savez order with their exact dtypes. Used to emit a RECTANGULAR empty
# artifact: structurally indistinguishable from a non-empty one except for
# length. The six `*_count` arrays are float32 despite being logically
# integral; that is the existing contract and is reproduced deliberately.
_EMPTY_NPZ_DTYPES = {
    'seeds':                     np.uint32,
    'forward_matches':           np.float32,
    'reverse_matches':           np.float32,
    'window_size':               np.int32,
    'offset':                    np.int32,
    'trial_number':              np.int32,
    'skip_min':                  np.int32,
    'skip_max':                  np.int32,
    'skip_range':                np.int32,
    'forward_count':             np.float32,
    'reverse_count':             np.float32,
    'bidirectional_count':       np.float32,
    'intersection_count':        np.float32,
    'intersection_ratio':        np.float32,
    'intersection_weight':       np.float32,
    'bidirectional_selectivity': np.float32,
    'forward_only_count':        np.float32,
    'reverse_only_count':        np.float32,
    'survivor_overlap_ratio':    np.float32,
    'score':                     np.float32,
    'skip_mode':                 np.uint8,
    'prng_type':                 np.uint8,
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

    if n == 0:
        print("⚠️  No survivors — skipping NPZ conversion (empty input is valid)")
        # Write empty NPZ so downstream steps don't fail on missing file.
        # [S172 Phase-5 D3.0] All 22 arrays, each length 0 with its frozen
        # dtype. The previous one-array form (`seeds=[]`) was a defect, not an
        # alternate valid representation: it made an empty artifact structurally
        # different from a non-empty one, so every downstream reader had to
        # special-case it.
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_path), **{
            name: np.array([], dtype=dtype)
            for name, dtype in _EMPTY_NPZ_DTYPES.items()
        })
        print(f"✅ Empty NPZ written to {output_file} "
              f"({len(_EMPTY_NPZ_DTYPES)} rectangular zero-length arrays)")
        return
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
    def _parse_skip_range(val):
        """Handle skip_range as int, list, or 'min-max' string."""
        if isinstance(val, int):
            return val
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return int(val[1]) - int(val[0])
        if isinstance(val, str) and '-' in val:
            parts = val.split('-')
            try:
                return int(parts[1]) - int(parts[0])
            except (ValueError, IndexError):
                return 0
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0
    skip_range = np.array([_parse_skip_range(s.get('skip_range', 0)) for s in survivors], dtype=np.int32)

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

    # [S172 Phase-5 D3.0] Resolution stays the caller's job (the canonical
    # docstring says so): resolve prng_type -> prng_base -> 'java_lcg' exactly
    # as before, THEN hand the resolved string to the canonical encoder. The
    # only change is that an unresolvable identity now raises instead of
    # silently becoming 0.
    skip_mode = np.array([
        encode_skip_mode(s.get('skip_mode', 'constant'))
        for s in survivors
    ], dtype=np.uint8)

    prng_type = np.array([
        encode_prng_type(s.get('prng_type', s.get('prng_base', 'java_lcg')))
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
