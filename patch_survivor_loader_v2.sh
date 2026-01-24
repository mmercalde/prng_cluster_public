#!/bin/bash
# patch_survivor_loader_v2.sh - Update survivor_loader.py to v2.0
# Adds support for NPZ v3.0 full metadata reconstruction
#
# USAGE:
#   cd ~/distributed_prng_analysis
#   bash patch_survivor_loader_v2.sh

set -e
cd ~/distributed_prng_analysis

echo "=============================================="
echo "Patching utils/survivor_loader.py → v2.0"
echo "=============================================="

# Backup
cp utils/survivor_loader.py utils/survivor_loader.py.v1.backup
echo "✓ Backed up to utils/survivor_loader.py.v1.backup"

# Create updated file
cat > utils/survivor_loader.py << 'LOADER_EOF'
#!/usr/bin/env python3
"""
Modular Survivor Loader - Auto-detects NPZ vs JSON
===================================================
Provides consistent interface for loading survivor data across all pipeline scripts.

Usage:
    from utils.survivor_loader import load_survivors
    
    # Simple usage (auto-detect format)
    result = load_survivors("bidirectional_survivors_binary.npz")
    survivors = result.data  # numpy arrays or list of dicts
    
    # Check metadata (WATCHER visibility)
    print(f"Format: {result.format}, Fallback: {result.fallback_used}")
    
    # Force specific return format
    result = load_survivors("survivors.json", return_format="array")
    result = load_survivors("survivors.npz", return_format="dict")

Version History:
  1.0.0 - Initial modular loader (Jan 3, 2026)
  2.0.0 - NPZ v3.0 support: full metadata reconstruction (Jan 23, 2026)
          Now reconstructs all 22 fields from NPZ v3.0 format
          
Approved: Team Beta
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Literal
from dataclasses import dataclass
import logging

VERSION = "2.0.0"

# Default logger
_default_logger = logging.getLogger(__name__)

# Categorical decodings (reverse of encoding in converter)
SKIP_MODE_DECODING = {0: 'constant', 1: 'variable'}
PRNG_TYPE_DECODING = {
    0: 'java_lcg', 1: 'java_lcg_reverse',
    2: 'mt19937', 3: 'mt19937_reverse',
    4: 'xorshift128', 5: 'xorshift128_reverse',
    6: 'lcg32', 7: 'lcg32_reverse',
    8: 'minstd', 9: 'minstd_reverse',
    10: 'randu', 11: 'randu_reverse',
}


@dataclass
class SurvivorData:
    """
    Container for survivor data with WATCHER-visible metadata.
    
    Attributes:
        data: The survivor data (arrays or list of dicts)
        format: Source format ("npz" or "json")
        source_path: Resolved absolute path that was loaded
        fallback_used: True if fell back from NPZ to JSON
        count: Number of survivors loaded
        npz_version: NPZ schema version (1 or 3) if loaded from NPZ
    """
    data: Union[Dict[str, np.ndarray], List[Dict[str, Any]]]
    format: Literal["npz", "json"]
    source_path: str
    fallback_used: bool
    count: int
    npz_version: Optional[int] = None
    
    def __repr__(self):
        ver = f", npz_v{self.npz_version}" if self.npz_version else ""
        return f"SurvivorData(format={self.format}{ver}, count={self.count}, fallback={self.fallback_used})"


def detect_npz_version(data: Dict[str, np.ndarray]) -> int:
    """
    Detect NPZ schema version based on available arrays.
    
    v1: seeds, forward_matches, reverse_matches (3 arrays)
    v3: All 22 metadata fields including skip_min, forward_count, etc.
    """
    keys = set(data.keys())
    
    # v3.0 has metadata fields
    v3_indicators = {'skip_min', 'skip_max', 'forward_count', 'bidirectional_count'}
    if v3_indicators.issubset(keys):
        return 3
    
    return 1


def load_survivors(
    path: str,
    *,
    allow_fallback: bool = True,
    return_format: Literal["array", "dict", "auto"] = "auto",
    logger: Optional[logging.Logger] = None
) -> SurvivorData:
    """
    Load survivors from NPZ (fast) or JSON (fallback).
    
    Detection order (deterministic):
        1. If path ends with .npz → load NPZ
        2. If path ends with .json → load JSON
        3. If allow_fallback and .npz exists for .json path → load NPZ
        4. If allow_fallback and .json exists for .npz path → load JSON
        5. Else → hard error (FileNotFoundError)
    
    Args:
        path: File path (.npz or .json)
        allow_fallback: If True, try alternate format if primary missing
        return_format: Output format preference
            - "array": Always return numpy arrays (fast)
            - "dict": Always return list of dicts (legacy compatible)
            - "auto": Return native format (NPZ→array, JSON→dict)
        logger: Optional logger for visibility
    
    Returns:
        SurvivorData with:
            - data: numpy arrays dict OR list of dicts (based on return_format)
            - format: "npz" or "json" (what was actually loaded)
            - source_path: Resolved path that was loaded
            - fallback_used: True if fallback occurred
            - count: Number of survivors
            - npz_version: 1 or 3 if loaded from NPZ
    
    Raises:
        FileNotFoundError: If file not found and no fallback available
        ValueError: If file format is invalid
    
    Example:
        result = load_survivors("bidirectional_survivors_binary.npz")
        print(f"Loaded {result.count} survivors from {result.format}")
        
        if result.fallback_used:
            logger.warning("NPZ missing, fell back to JSON")
    """
    log = logger or _default_logger
    path = Path(path)
    fallback_used = False
    resolved_path = None
    source_format = None
    
    # === DETECTION LOGIC (deterministic order) ===
    
    if path.suffix == ".npz":
        if path.exists():
            resolved_path = path
            source_format = "npz"
        elif allow_fallback:
            # Try JSON fallback
            json_path = path.with_suffix(".json")
            if json_path.exists():
                resolved_path = json_path
                source_format = "json"
                fallback_used = True
                log.warning(f"NPZ not found, falling back to JSON: {json_path}")
    
    elif path.suffix == ".json":
        if path.exists():
            resolved_path = path
            source_format = "json"
        elif allow_fallback:
            # Try NPZ (preferred, faster)
            npz_path = path.with_suffix(".npz")
            # Also check for _binary.npz naming convention
            npz_binary_path = Path(str(path).replace(".json", "_binary.npz"))
            
            if npz_binary_path.exists():
                resolved_path = npz_binary_path
                source_format = "npz"
                fallback_used = True
                log.info(f"Found NPZ binary version: {npz_binary_path}")
            elif npz_path.exists():
                resolved_path = npz_path
                source_format = "npz"
                fallback_used = True
                log.info(f"Found NPZ version: {npz_path}")
    
    else:
        # Unknown extension - try to find any version
        if path.exists():
            # Guess format from content
            resolved_path = path
            source_format = "json"  # Default assumption
        elif allow_fallback:
            npz_path = path.with_suffix(".npz")
            json_path = path.with_suffix(".json")
            if npz_path.exists():
                resolved_path = npz_path
                source_format = "npz"
            elif json_path.exists():
                resolved_path = json_path
                source_format = "json"
    
    # === HARD ERROR if not found ===
    
    if resolved_path is None or not resolved_path.exists():
        raise FileNotFoundError(
            f"Survivor file not found: {path}\n"
            f"Checked: {path}, {path.with_suffix('.npz')}, {path.with_suffix('.json')}"
        )
    
    log.debug(f"Loading survivors: {resolved_path} (format={source_format})")
    
    # === LOAD DATA ===
    
    npz_version = None
    if source_format == "npz":
        raw_data = _load_npz(resolved_path)
        npz_version = detect_npz_version(raw_data)
        native_format = "array"
        log.debug(f"NPZ version detected: v{npz_version}")
    else:
        raw_data = _load_json(resolved_path)
        native_format = "dict"
    
    # === CONVERT IF NEEDED ===
    
    if return_format == "auto":
        data = raw_data
    elif return_format == "array" and native_format == "dict":
        data = _dict_to_array(raw_data)
    elif return_format == "dict" and native_format == "array":
        data = _array_to_dict(raw_data, npz_version)
    else:
        data = raw_data
    
    # === COUNT ===
    
    if isinstance(data, dict):
        count = len(data.get('seeds', []))
    else:
        count = len(data)
    
    log.info(f"Loaded {count} survivors from {source_format} ({resolved_path.name})")
    
    return SurvivorData(
        data=data,
        format=source_format,
        source_path=str(resolved_path.absolute()),
        fallback_used=fallback_used,
        count=count,
        npz_version=npz_version
    )


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load from NPZ binary format (fast).
    
    Handles multiple NPZ schemas:
    - v1: seeds, forward_matches, reverse_matches
    - v3: seeds + 21 metadata arrays (full preservation)
    """
    data = np.load(path, allow_pickle=True)
    
    # Return all arrays as dict (flexible schema)
    result = {key: data[key] for key in data.files if key != 'metadata'}
    
    # Ensure 'seeds' key exists
    if 'seeds' not in result:
        raise ValueError(f"NPZ missing required 'seeds' array: {path}")
    
    return result


def _load_json(path: Path) -> List[Dict[str, Any]]:
    """Load from JSON format (legacy)."""
    with open(path, 'r') as f:
        return json.load(f)


def _dict_to_array(survivors: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Convert list of dicts to numpy arrays.
    
    v2.0: Now extracts all available fields, not just seeds/matches.
    """
    if not survivors:
        return {'seeds': np.array([], dtype=np.uint32)}
    
    # Collect all numeric fields from first survivor
    sample = survivors[0]
    
    # Core fields (always present)
    seeds = []
    
    # Build arrays for all numeric fields
    numeric_fields = {}
    for key, val in sample.items():
        if key == 'seed':
            continue  # Handled separately
        if isinstance(val, (int, float)):
            numeric_fields[key] = []
    
    # Extract values
    for s in survivors:
        if isinstance(s, dict):
            seeds.append(s.get('seed', s.get('candidate_seed', 0)))
            for key in numeric_fields:
                numeric_fields[key].append(s.get(key, 0.0))
        else:
            # Just a seed integer
            seeds.append(int(s))
            for key in numeric_fields:
                numeric_fields[key].append(0.0)
    
    result = {'seeds': np.array(seeds, dtype=np.uint32)}
    
    for key, vals in numeric_fields.items():
        # Determine dtype
        if all(isinstance(v, int) and -2**31 <= v < 2**31 for v in vals):
            result[key] = np.array(vals, dtype=np.int32)
        else:
            result[key] = np.array(vals, dtype=np.float32)
    
    return result


def _array_to_dict(data: Dict[str, np.ndarray], npz_version: int = 1) -> List[Dict[str, Any]]:
    """Convert numpy arrays to list of dicts.
    
    v2.0: Supports both v1 (3 fields) and v3 (22 fields) NPZ formats.
    """
    n = len(data['seeds'])
    survivors = []
    
    for i in range(n):
        survivor = {'seed': int(data['seeds'][i])}
        
        # Core match fields (v1 compatible)
        if 'forward_matches' in data:
            survivor['forward_matches'] = float(data['forward_matches'][i])
        if 'reverse_matches' in data:
            survivor['reverse_matches'] = float(data['reverse_matches'][i])
        
        # v3.0 metadata fields
        if npz_version >= 3:
            # Integer fields
            for field in ['window_size', 'offset', 'trial_number', 'skip_min', 'skip_max', 'skip_range']:
                if field in data:
                    survivor[field] = int(data[field][i])
            
            # Float fields
            for field in ['forward_count', 'reverse_count', 'bidirectional_count',
                         'intersection_count', 'intersection_ratio', 'intersection_weight',
                         'bidirectional_selectivity', 'forward_only_count', 'reverse_only_count',
                         'survivor_overlap_ratio', 'score']:
                if field in data:
                    survivor[field] = float(data[field][i])
            
            # Categorical fields (decode from int)
            if 'skip_mode' in data:
                survivor['skip_mode'] = SKIP_MODE_DECODING.get(int(data['skip_mode'][i]), 'constant')
            if 'prng_type' in data:
                survivor['prng_type'] = PRNG_TYPE_DECODING.get(int(data['prng_type'][i]), 'java_lcg')
        
        survivors.append(survivor)
    
    return survivors


# === CONVENIENCE FUNCTIONS ===

def get_survivor_count(path: str) -> int:
    """Quick count without loading full data."""
    path = Path(path)
    
    if path.suffix == ".npz" and path.exists():
        data = np.load(path)
        return len(data['seeds'])
    elif path.exists():
        with open(path, 'r') as f:
            return len(json.load(f))
    else:
        raise FileNotFoundError(f"Survivor file not found: {path}")


def get_survivor_metadata(path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load survivors and return as dict keyed by seed.
    Useful for Step 3 metadata merge.
    
    Returns:
        {seed_value: {field: value, ...}, ...}
    """
    result = load_survivors(path, return_format="dict")
    return {s['seed']: s for s in result.data}
LOADER_EOF

echo "✓ Updated utils/survivor_loader.py to v2.0"

# Verify
echo ""
echo "Verifying..."
python3 -c "from utils.survivor_loader import VERSION; print(f'Version: {VERSION}')"
python3 -c "from utils.survivor_loader import detect_npz_version; print('✓ detect_npz_version available')"
python3 -c "from utils.survivor_loader import get_survivor_metadata; print('✓ get_survivor_metadata available')"

echo ""
echo "=============================================="
echo "PATCH COMPLETE"
echo "=============================================="
