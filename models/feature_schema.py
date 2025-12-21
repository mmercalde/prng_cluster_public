"""
Feature Schema Derivation and Validation (v3.1.2)

Provides streaming-based schema extraction from large survivor files
without loading entire file into memory.

Key Features:
- Uses ijson for memory-safe parsing (813MB+ files)
- Lexicographic feature ordering (matches reinforcement_engine.py)
- SHA256-based schema hash for validation
- Narrow range warning for y-labels
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any, Optional

logger = logging.getLogger(__name__)

# Try ijson first (preferred), fallback to json.JSONDecoder
try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False
    logger.warning("ijson not available, using json.JSONDecoder fallback")


def get_feature_schema_from_data(survivors_file: str) -> Dict[str, Any]:
    """
    Derive feature schema by streaming only the first record.
    
    Uses ijson for memory-safe parsing of large files (813MB+).
    Does NOT load entire file into memory.
    
    Args:
        survivors_file: Path to survivors JSON file (Step 3 output)
        
    Returns:
        Dict with feature_count, feature_names, ordering, etc.
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or has no features
    """
    path = Path(survivors_file)
    if not path.exists():
        raise FileNotFoundError(f"Cannot derive schema: {survivors_file} not found")
    
    if IJSON_AVAILABLE:
        return _get_schema_ijson(path)
    else:
        return _get_schema_decoder(path)


def _get_schema_ijson(path: Path) -> Dict[str, Any]:
    """Extract schema using ijson streaming parser."""
    with open(path, 'rb') as f:
        parser = ijson.items(f, 'item')
        try:
            first_record = next(parser)
        except StopIteration:
            raise ValueError(f"Empty survivors file: {path}")
    
    features = first_record.get('features', {})
    if not features:
        raise ValueError(f"First record has no 'features' key: {path}")
    
    return {
        "version": "dynamic",
        "source_file": str(path.resolve()),
        "feature_count": len(features),
        "feature_names": sorted(features.keys()),  # Lexicographic ordering
        "ordering": "lexicographic_by_key",
        "derived_from": "ijson_first_record"
    }


def _get_schema_decoder(path: Path) -> Dict[str, Any]:
    """
    Extract schema using json.JSONDecoder (fallback).
    
    Reads file in chunks until first complete JSON object is decoded.
    Does NOT use regex. Does NOT load entire file.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    chunk_size = 8192  # 8KB chunks
    
    with open(path, 'r', encoding='utf-8') as f:
        # Skip opening bracket of array
        while True:
            char = f.read(1)
            if char == '[':
                break
            if not char:
                raise ValueError(f"Invalid JSON array: {path}")
        
        # Read until we can decode first object
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"Could not find first record in: {path}")
            
            buffer += chunk
            buffer = buffer.lstrip()
            
            if buffer.startswith('{'):
                try:
                    first_record, _ = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    continue
    
    features = first_record.get('features', {})
    if not features:
        raise ValueError(f"First record has no 'features' key: {path}")
    
    return {
        "version": "dynamic",
        "source_file": str(path.resolve()),
        "feature_count": len(features),
        "feature_names": sorted(features.keys()),
        "ordering": "lexicographic_by_key",
        "derived_from": "json_decoder_buffered"
    }


def get_feature_schema_with_hash(survivors_file: str) -> Dict[str, Any]:
    """
    Get feature schema with validation hash.
    
    The hash is computed from the sorted feature names, providing
    a way to detect schema drift between training and prediction.
    
    Args:
        survivors_file: Path to survivors JSON file
        
    Returns:
        Schema dict with feature_schema_hash added
    """
    schema = get_feature_schema_from_data(survivors_file)
    
    # Canonical hash from sorted feature names
    names_str = ",".join(schema["feature_names"])
    schema["feature_schema_hash"] = hashlib.sha256(
        names_str.encode('utf-8')
    ).hexdigest()[:16]
    
    return schema


def validate_feature_schema_hash(expected_hash: str, feature_names: List[str]) -> bool:
    """
    Validate that runtime features match training schema.
    
    Called by Step 6 before prediction to ensure feature ordering
    hasn't changed between training and inference.
    
    Args:
        expected_hash: Hash from training (stored in sidecar)
        feature_names: Runtime feature names
        
    Returns:
        True if hashes match, False otherwise
    """
    names_str = ",".join(sorted(feature_names))
    runtime_hash = hashlib.sha256(names_str.encode('utf-8')).hexdigest()[:16]
    return runtime_hash == expected_hash


def get_feature_count(survivors_file: str = "survivors_with_scores.json") -> int:
    """
    Get feature count from actual data.
    
    Convenience function for model initialization.
    """
    schema = get_feature_schema_from_data(survivors_file)
    return schema["feature_count"]


def load_quality_from_survivors(survivors_file: str) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """
    Load survivors and quality scores with range validation.
    
    Streams the file to avoid memory issues with large files.
    Auto-detects score normalization based on observed range.
    
    Args:
        survivors_file: Path to survivors JSON file
        
    Returns:
        Tuple of (survivors, quality, metadata)
        - survivors: List of seed integers
        - quality: List of normalized quality scores [0, 1]
        - metadata: Dict with source info, range, normalization, warnings
    """
    if not IJSON_AVAILABLE:
        raise ImportError("ijson required for streaming quality loading. Install with: pip install ijson")
    
    path = Path(survivors_file)
    if not path.exists():
        raise FileNotFoundError(f"Survivors file not found: {survivors_file}")
    
    # First pass: sample to determine actual range
    scores_sample = []
    with open(path, 'rb') as f:
        parser = ijson.items(f, 'item')
        for i, item in enumerate(parser):
            if i >= 1000:  # Sample first 1000
                break
            if 'features' in item:
                scores_sample.append(item['features'].get('score', 0.0))
            elif 'score' in item:
                scores_sample.append(item.get('score', 0.0))
    
    if not scores_sample:
        raise ValueError(f"No scores found in {survivors_file}")
    
    observed_min = min(scores_sample)
    observed_max = max(scores_sample)
    observed_range = observed_max - observed_min
    
    # Initialize warnings list
    warnings = []
    
    # Check for narrow range (Team Beta Condition #3)
    NARROW_RANGE_THRESHOLD = 0.01
    if observed_range < NARROW_RANGE_THRESHOLD:
        warnings.append("score_range_narrow")
        logger.warning(f"Score range is very narrow ({observed_range:.4f})")
        logger.warning(f"  Min: {observed_min:.4f}, Max: {observed_max:.4f}")
        logger.warning(f"  This may indicate 'score' is not the right y-label field")
    
    # Determine normalization strategy
    if observed_max <= 1.0:
        # Already normalized [0, 1]
        normalization = "none"
        normalize_fn: Callable[[float], float] = lambda x: x
    elif observed_max <= 100.0:
        # Percentage [0, 100] -> [0, 1]
        normalization = "divide_by_100"
        normalize_fn = lambda x: x / 100.0
    else:
        # Arbitrary scale -> normalize by observed max
        normalization = f"divide_by_{observed_max:.4f}"
        max_val = observed_max  # Capture for closure
        normalize_fn = lambda x: x / max_val
    
    # Second pass: load all data with streaming
    survivors: List[int] = []
    quality: List[float] = []
    
    logger.info(f"Loading survivors from {survivors_file}...")
    with open(path, 'rb') as f:
        parser = ijson.items(f, 'item')
        for item in parser:
            # Get seed
            seed = item.get('seed')
            if seed is None:
                continue
            survivors.append(int(seed))
            
            # Get raw score
            if 'features' in item:
                raw_score = item['features'].get('score', 0.0)
            elif 'score' in item:
                raw_score = item.get('score', 0.0)
            else:
                raw_score = 0.5
                if "missing_score_field" not in warnings:
                    warnings.append("missing_score_field")
            
            # Normalize and clamp to [0, 1]
            normalized = normalize_fn(float(raw_score))
            clamped = max(0.0, min(1.0, normalized))
            quality.append(clamped)
    
    logger.info(f"Loaded {len(survivors)} survivors")
    
    # Metadata for sidecar
    metadata = {
        "field": "features.score",
        "observed_min": observed_min,
        "observed_max": observed_max,
        "observed_range": observed_range,
        "normalization_method": normalization,
        "output_range": [0.0, 1.0],
        "sample_size": len(scores_sample),
        "total_samples": len(survivors),
        "warnings": warnings
    }
    
    return survivors, quality, metadata
