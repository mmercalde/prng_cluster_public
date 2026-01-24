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
"""
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

VERSION = "3.0.0"

# Categorical encodings
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
    
    v3.0: Now preserves all 22 fields from the source JSON.
    
    Returns metadata dict for verification.
    """
    print(f"Loading {input_file}...")
    with open(input_file) as f:
        survivors = json.load(f)
    
    n = len(survivors)
    print(f"Loaded {n:,} survivors")
    
    # Show available fields
    if n > 0:
        available = set(survivors[0].keys())
        print(f"Available fields: {sorted(available)}")
    
    # === CORE ARRAYS (v1.0 compatibility) ===
    seeds = np.array([s['seed'] for s in survivors], dtype=np.uint32)
    
    forward_matches = np.array([
        s.get('forward_count', s.get('forward_matches', s.get('score', 0.0))) 
        for s in survivors
    ], dtype=np.float32)
    
    reverse_matches = np.array([
        s.get('reverse_count', s.get('reverse_matches', s.get('score', 0.0))) 
        for s in survivors
    ], dtype=np.float32)
    
    # === METADATA ARRAYS (v3.0 addition) ===
    # Integer fields
    window_size = np.array([s.get('window_size', 0) for s in survivors], dtype=np.int32)
    offset = np.array([s.get('offset', 0) for s in survivors], dtype=np.int32)
    trial_number = np.array([s.get('trial_number', 0) for s in survivors], dtype=np.int32)
    
    # Skip analysis
    skip_min = np.array([s.get('skip_min', 0) for s in survivors], dtype=np.int32)
    skip_max = np.array([s.get('skip_max', 0) for s in survivors], dtype=np.int32)
    skip_range = np.array([s.get('skip_range', 0) for s in survivors], dtype=np.int32)
    
    # Sieve counts (may duplicate forward/reverse_matches but with correct names)
    forward_count = np.array([s.get('forward_count', 0.0) for s in survivors], dtype=np.float32)
    reverse_count = np.array([s.get('reverse_count', 0.0) for s in survivors], dtype=np.float32)
    bidirectional_count = np.array([s.get('bidirectional_count', 0.0) for s in survivors], dtype=np.float32)
    
    # Intersection metrics
    intersection_count = np.array([s.get('intersection_count', 0.0) for s in survivors], dtype=np.float32)
    intersection_ratio = np.array([s.get('intersection_ratio', 0.0) for s in survivors], dtype=np.float32)
    intersection_weight = np.array([s.get('intersection_weight', 0.0) for s in survivors], dtype=np.float32)
    
    # Derived metrics
    bidirectional_selectivity = np.array([s.get('bidirectional_selectivity', 0.0) for s in survivors], dtype=np.float32)
    forward_only_count = np.array([s.get('forward_only_count', 0.0) for s in survivors], dtype=np.float32)
    reverse_only_count = np.array([s.get('reverse_only_count', 0.0) for s in survivors], dtype=np.float32)
    survivor_overlap_ratio = np.array([s.get('survivor_overlap_ratio', 0.0) for s in survivors], dtype=np.float32)
    
    # Score
    score = np.array([s.get('score', 0.0) for s in survivors], dtype=np.float32)
    
    # Categorical fields (encoded as uint8)
    skip_mode = np.array([
        SKIP_MODE_ENCODING.get(s.get('skip_mode', 'constant'), 0) 
        for s in survivors
    ], dtype=np.uint8)
    
    prng_type = np.array([
        PRNG_TYPE_ENCODING.get(s.get('prng_type', s.get('prng_base', 'java_lcg')), 0)
        for s in survivors
    ], dtype=np.uint8)
    
    # === SAVE NPZ ===
    print(f"Saving {output_file}...")
    np.savez_compressed(
        output_file,
        # Core (v1.0)
        seeds=seeds,
        forward_matches=forward_matches,
        reverse_matches=reverse_matches,
        # Metadata (v3.0)
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
    
    # Metadata for verification
    input_size = Path(input_file).stat().st_size
    output_size = Path(output_file).stat().st_size
    ratio = input_size / output_size if output_size > 0 else 0
    
    metadata = {
        "version": VERSION,
        "source_file": str(Path(input_file).resolve()),
        "output_file": str(Path(output_file).resolve()),
        "survivor_count": n,
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
    
    # Save metadata
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Conversion complete (v{VERSION}):")
    print(f"  Input:  {input_size:,} bytes ({input_size/1024/1024:.1f} MB)")
    print(f"  Output: {output_size:,} bytes ({output_size/1024:.1f} KB)")
    print(f"  Ratio:  {ratio:.1f}x compression")
    print(f"  Arrays: 22 (was 3 in v2.0)")
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
    
    # Default output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_binary.npz"
    
    meta_path = output_path.parent / f"{output_path.stem}.meta.json"
    
    convert_json_to_npz(str(input_path), str(output_path), str(meta_path))
    return 0


if __name__ == "__main__":
    exit(main())
