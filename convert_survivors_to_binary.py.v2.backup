#!/usr/bin/env python3
"""
convert_survivors_to_binary.py - Convert JSON survivors to NPZ binary format

Performance: 88x faster loading (4.2s → 0.05s), 400x smaller (258MB → 0.6MB)

Usage:
    python3 convert_survivors_to_binary.py bidirectional_survivors.json
    python3 convert_survivors_to_binary.py bidirectional_survivors.json --output /tmp/survivors.npz

Version: 2.0.0 (January 19, 2026)
  - Added --output flag for atomic write support
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime


def convert_json_to_npz(input_file: str, output_file: str, meta_file: str) -> dict:
    """
    Convert JSON survivors to compressed NPZ format.
    
    Returns metadata dict for verification.
    """
    print(f"Loading {input_file}...")
    with open(input_file) as f:
        survivors = json.load(f)
    
    print(f"Loaded {len(survivors):,} survivors")
    
    # Extract arrays
    seeds = np.array([s['seed'] for s in survivors], dtype=np.uint32)
    
    # Handle optional fields with defaults
    forward_matches = np.array([
        s.get('forward_count', s.get('score', 0)) 
        for s in survivors
    ], dtype=np.float32)
    
    reverse_matches = np.array([
        s.get('reverse_count', s.get('score', 0)) 
        for s in survivors
    ], dtype=np.float32)
    
    # Save compressed NPZ
    print(f"Saving {output_file}...")
    np.savez_compressed(
        output_file,
        seeds=seeds,
        forward_matches=forward_matches,
        reverse_matches=reverse_matches
    )
    
    # Metadata for verification
    metadata = {
        "source_file": str(Path(input_file).resolve()),
        "output_file": str(Path(output_file).resolve()),
        "survivor_count": len(survivors),
        "seed_dtype": str(seeds.dtype),
        "seed_min": int(seeds.min()),
        "seed_max": int(seeds.max()),
        "converted_at": datetime.now().isoformat(),
        "format_version": "1.0"
    }
    
    # Save metadata
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Report sizes
    input_size = Path(input_file).stat().st_size
    output_size = Path(output_file).stat().st_size
    ratio = input_size / output_size if output_size > 0 else 0
    
    print(f"✓ Conversion complete:")
    print(f"  Input:  {input_size:,} bytes ({input_size/1024/1024:.1f} MB)")
    print(f"  Output: {output_size:,} bytes ({output_size/1024:.1f} KB)")
    print(f"  Ratio:  {ratio:.0f}x compression")
    print(f"  Meta:   {meta_file}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Convert bidirectional_survivors.json to NPZ binary format'
    )
    parser.add_argument('input_file', 
                        help='Input JSON file (e.g., bidirectional_survivors.json)')
    parser.add_argument('--output', '-o',
                        help='Output NPZ file path (default: derived from input)')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    
    # Verify input exists
    if not Path(input_file).exists():
        print(f"ERROR: Input file not found: {input_file}")
        return 1
    
    # Derive output paths
    if args.output:
        output_file = args.output
        # Meta file alongside output
        meta_file = str(Path(args.output).with_suffix('.meta.json'))
    else:
        # Default: derive from input name
        base = input_file.replace('.json', '')
        output_file = f"{base}_binary.npz"
        meta_file = f"{base}_binary.meta.json"
    
    try:
        convert_json_to_npz(input_file, output_file, meta_file)
        return 0
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
