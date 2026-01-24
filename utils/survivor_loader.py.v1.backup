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

Version: 1.0.0
Date: January 3, 2026
Approved: Team Beta
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Literal
from dataclasses import dataclass
import logging

# Default logger
_default_logger = logging.getLogger(__name__)


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
    """
    data: Union[Dict[str, np.ndarray], List[Dict[str, Any]]]
    format: Literal["npz", "json"]
    source_path: str
    fallback_used: bool
    count: int
    
    def __repr__(self):
        return f"SurvivorData(format={self.format}, count={self.count}, fallback={self.fallback_used})"


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
    
    if source_format == "npz":
        raw_data = _load_npz(resolved_path)
        native_format = "array"
    else:
        raw_data = _load_json(resolved_path)
        native_format = "dict"
    
    # === CONVERT IF NEEDED ===
    
    if return_format == "auto":
        data = raw_data
    elif return_format == "array" and native_format == "dict":
        data = _dict_to_array(raw_data)
    elif return_format == "dict" and native_format == "array":
        data = _array_to_dict(raw_data)
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
        count=count
    )


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load from NPZ binary format (fast).
    
    Handles multiple NPZ schemas:
    - v1: seeds, forward_matches, reverse_matches
    - v2: seeds, scores, forward_count, reverse_count, + metadata
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
    """Convert list of dicts to numpy arrays."""
    seeds = []
    forward_matches = []
    reverse_matches = []
    
    for s in survivors:
        if isinstance(s, dict):
            seeds.append(s.get('seed', s.get('candidate_seed', 0)))
            forward_matches.append(s.get('forward_match_rate', s.get('forward_match', 0.0)))
            reverse_matches.append(s.get('reverse_match_rate', s.get('reverse_match', 0.0)))
        else:
            # Just a seed integer
            seeds.append(int(s))
            forward_matches.append(0.0)
            reverse_matches.append(0.0)
    
    return {
        'seeds': np.array(seeds, dtype=np.uint32),
        'forward_matches': np.array(forward_matches, dtype=np.float32),
        'reverse_matches': np.array(reverse_matches, dtype=np.float32)
    }


def _array_to_dict(data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Convert numpy arrays to list of dicts."""
    return [
        {
            'seed': int(seed),
            'forward_match_rate': float(fwd),
            'reverse_match_rate': float(rev)
        }
        for seed, fwd, rev in zip(
            data['seeds'],
            data['forward_matches'],
            data['reverse_matches']
        )
    ]


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
