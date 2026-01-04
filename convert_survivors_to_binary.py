#!/usr/bin/env python3
"""
Convert survivors JSON to binary NPZ format for fast loading.
"""
import json
import numpy as np
import hashlib
from datetime import datetime
from pathlib import Path

def convert_survivors(json_path, output_path=None):
    json_path = Path(json_path)
    if output_path is None:
        output_path = json_path.stem + "_binary.npz"
    
    print(f"Loading {json_path}...")
    with open(json_path) as f:
        data = json.load(f)
    
    n = len(data)
    print(f"Converting {n:,} survivors...")
    
    # Core fields
    seeds = np.array([s['seed'] for s in data], dtype=np.int64)
    
    # Numeric scoring fields
    scores = np.array([s.get('score', 0.0) for s in data], dtype=np.float32)
    forward_count = np.array([s.get('forward_count', 0) for s in data], dtype=np.int32)
    reverse_count = np.array([s.get('reverse_count', 0) for s in data], dtype=np.int32)
    bidirectional_count = np.array([s.get('bidirectional_count', 0) for s in data], dtype=np.int32)
    intersection_count = np.array([s.get('intersection_count', 0) for s in data], dtype=np.int32)
    intersection_ratio = np.array([s.get('intersection_ratio', 0.0) for s in data], dtype=np.float32)
    bidirectional_selectivity = np.array([s.get('bidirectional_selectivity', 0.0) for s in data], dtype=np.float32)
    
    # Numeric PRNG/Skip fields
    skip_min = np.array([s.get('skip_min', 0) for s in data], dtype=np.int32)
    skip_max = np.array([s.get('skip_max', 0) for s in data], dtype=np.int32)
    skip_range = np.array([s.get('skip_range', 0) for s in data], dtype=np.int32)
    window_size = np.array([s.get('window_size', 0) for s in data], dtype=np.int32)
    offset = np.array([s.get('offset', 0) for s in data], dtype=np.int32)
    trial_number = np.array([s.get('trial_number', 0) for s in data], dtype=np.int32)
    
    # Additional ratios
    survivor_overlap_ratio = np.array([s.get('survivor_overlap_ratio', 0.0) for s in data], dtype=np.float32)
    intersection_weight = np.array([s.get('intersection_weight', 0.0) for s in data], dtype=np.float32)
    forward_only_count = np.array([s.get('forward_only_count', 0) for s in data], dtype=np.int32)
    reverse_only_count = np.array([s.get('reverse_only_count', 0) for s in data], dtype=np.int32)
    
    # String fields -> indexed lookup
    # prng_type
    prng_types = sorted(set(s.get('prng_type', 'mt') for s in data))
    prng_type_map = {t: i for i, t in enumerate(prng_types)}
    prng_type_idx = np.array([prng_type_map[s.get('prng_type', 'mt')] for s in data], dtype=np.int8)
    
    # prng_base (same as prng_type in your data)
    prng_bases = sorted(set(s.get('prng_base', 'mt') for s in data))
    prng_base_map = {t: i for i, t in enumerate(prng_bases)}
    prng_base_idx = np.array([prng_base_map[s.get('prng_base', 'mt')] for s in data], dtype=np.int8)
    
    # skip_mode
    skip_modes = sorted(set(s.get('skip_mode', 'range') for s in data))
    skip_mode_map = {m: i for i, m in enumerate(skip_modes)}
    skip_mode_idx = np.array([skip_mode_map[s.get('skip_mode', 'range')] for s in data], dtype=np.int8)
    
    # sessions (list -> bitmask: midday=1, evening=2)
    def sessions_to_mask(sess_list):
        mask = 0
        if 'midday' in sess_list: mask |= 1
        if 'evening' in sess_list: mask |= 2
        return mask
    sessions_mask = np.array([sessions_to_mask(s.get('sessions', [])) for s in data], dtype=np.int8)
    
    # Schema hash
    fields = ['seed', 'score', 'forward_count', 'reverse_count', 'bidirectional_count',
              'intersection_count', 'intersection_ratio', 'bidirectional_selectivity',
              'skip_min', 'skip_max', 'skip_range', 'window_size', 'offset', 'trial_number',
              'survivor_overlap_ratio', 'intersection_weight', 'forward_only_count', 
              'reverse_only_count', 'prng_type_idx', 'prng_base_idx', 'skip_mode_idx', 'sessions_mask']
    schema_hash = hashlib.md5(json.dumps(fields).encode()).hexdigest()[:8]
    
    # Metadata
    metadata = {
        'source_json': str(json_path.name),
        'row_count': n,
        'schema_hash': schema_hash,
        'created_at': datetime.now().isoformat(),
        'fields': fields,
        'prng_types': prng_types,
        'prng_bases': prng_bases,
        'skip_modes': skip_modes,
        'sessions_encoding': {'midday': 1, 'evening': 2}
    }
    
    # Save
    np.savez_compressed(
        output_path,
        seeds=seeds,
        scores=scores,
        forward_count=forward_count,
        reverse_count=reverse_count,
        bidirectional_count=bidirectional_count,
        intersection_count=intersection_count,
        intersection_ratio=intersection_ratio,
        bidirectional_selectivity=bidirectional_selectivity,
        skip_min=skip_min,
        skip_max=skip_max,
        skip_range=skip_range,
        window_size=window_size,
        offset=offset,
        trial_number=trial_number,
        survivor_overlap_ratio=survivor_overlap_ratio,
        intersection_weight=intersection_weight,
        forward_only_count=forward_only_count,
        reverse_only_count=reverse_only_count,
        prng_type_idx=prng_type_idx,
        prng_base_idx=prng_base_idx,
        skip_mode_idx=skip_mode_idx,
        sessions_mask=sessions_mask,
        metadata=np.array([json.dumps(metadata)])
    )
    
    size_json = json_path.stat().st_size / 1e6
    size_npz = Path(output_path).stat().st_size / 1e6
    
    print(f"\n✅ Saved to {output_path}")
    print(f"   Rows: {n:,}")
    print(f"   Fields: {len(fields)}")
    print(f"   Schema hash: {schema_hash}")
    print(f"   JSON: {size_json:.1f} MB → NPZ: {size_npz:.1f} MB ({size_json/size_npz:.1f}x smaller)")
    print(f"   PRNG types: {prng_types}")
    print(f"   Skip modes: {skip_modes}")
    
    # Sidecar
    meta_path = Path(output_path).with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata: {meta_path}")
    
    return output_path

if __name__ == "__main__":
    import sys
    json_path = sys.argv[1] if len(sys.argv) > 1 else "bidirectional_survivors.json"
    convert_survivors(json_path)
