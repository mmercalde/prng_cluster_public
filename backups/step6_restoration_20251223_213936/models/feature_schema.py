"""
Feature Schema Derivation and Validation (v3.3.0)

Provides streaming-based schema extraction from large survivor files
without loading entire file into memory.

Key Features:
- Uses ijson for memory-safe parsing (813MB+ files)
- Lexicographic feature ordering (matches reinforcement_engine.py)
- SHA256-based schema hash for validation
- Narrow range warning for y-labels
- v3.2.0: Returns full survivor dicts with pre-computed features
- v3.3.0: LABEL LEAKAGE FIX - excludes 'score' and 'confidence' from features

Changes in v3.3.0:
- Added exclude_from_features parameter (default: ['score', 'confidence'])
- 'score' is the y-label - MUST NOT be in X features
- 'confidence' is constant (0.1) - non-informative, removed for cleanliness
- Schema hash now reflects actual training features (48, not 50)
- Added logging of excluded features
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any, Optional, Union

logger = logging.getLogger(__name__)

# Try ijson first (preferred), fallback to json.JSONDecoder
try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False
    logger.warning("ijson not available, using json.JSONDecoder fallback")

# Default fields to exclude from training features
# - 'score': This is the Y-LABEL - including it causes catastrophic leakage
# - 'confidence': Constant value (0.1) across all records - non-informative
DEFAULT_EXCLUDE_FEATURES = ['score', 'confidence']


def get_feature_schema_from_data(
    survivors_file: str,
    exclude_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Derive feature schema by streaming only the first record.
    
    Uses ijson for memory-safe parsing of large files (813MB+).
    Does NOT load entire file into memory.
    
    Args:
        survivors_file: Path to survivors JSON file (Step 3 output)
        exclude_features: Features to exclude from schema (default: ['score', 'confidence'])
        
    Returns:
        Dict with feature_count, feature_names, ordering, etc.
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or has no features
    """
    if exclude_features is None:
        exclude_features = DEFAULT_EXCLUDE_FEATURES
    
    path = Path(survivors_file)
    if not path.exists():
        raise FileNotFoundError(f"Cannot derive schema: {survivors_file} not found")
    
    if IJSON_AVAILABLE:
        return _get_schema_ijson(path, exclude_features)
    else:
        return _get_schema_decoder(path, exclude_features)


def _get_schema_ijson(path: Path, exclude_features: List[str]) -> Dict[str, Any]:
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
    
    # v3.3.0: Exclude label and non-informative features
    filtered_features = {k: v for k, v in features.items() if k not in exclude_features}
    
    return {
        "version": "dynamic",
        "source_file": str(path.resolve()),
        "feature_count": len(filtered_features),
        "feature_names": sorted(filtered_features.keys()),  # Lexicographic ordering
        "ordering": "lexicographic_by_key",
        "derived_from": "ijson_first_record",
        "excluded_features": exclude_features,  # v3.3.0: Track what was excluded
        "original_feature_count": len(features)  # v3.3.0: For reference
    }


def _get_schema_decoder(path: Path, exclude_features: List[str]) -> Dict[str, Any]:
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
    
    # v3.3.0: Exclude label and non-informative features
    filtered_features = {k: v for k, v in features.items() if k not in exclude_features}
    
    return {
        "version": "dynamic",
        "source_file": str(path.resolve()),
        "feature_count": len(filtered_features),
        "feature_names": sorted(filtered_features.keys()),
        "ordering": "lexicographic_by_key",
        "derived_from": "json_decoder_buffered",
        "excluded_features": exclude_features,
        "original_feature_count": len(features)
    }


def get_feature_schema_with_hash(
    survivors_file: str,
    exclude_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get feature schema with validation hash.
    
    The hash is computed from the sorted feature names (AFTER exclusion),
    providing a way to detect schema drift between training and prediction.
    
    Args:
        survivors_file: Path to survivors JSON file
        exclude_features: Features to exclude (default: ['score', 'confidence'])
        
    Returns:
        Schema dict with feature_schema_hash added
    """
    if exclude_features is None:
        exclude_features = DEFAULT_EXCLUDE_FEATURES
        
    schema = get_feature_schema_from_data(survivors_file, exclude_features)
    
    # Canonical hash from sorted feature names (AFTER exclusion)
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


def get_feature_count(
    survivors_file: str = "survivors_with_scores.json",
    exclude_features: Optional[List[str]] = None
) -> int:
    """
    Get feature count from actual data.
    
    Convenience function for model initialization.
    """
    if exclude_features is None:
        exclude_features = DEFAULT_EXCLUDE_FEATURES
    schema = get_feature_schema_from_data(survivors_file, exclude_features)
    return schema["feature_count"]


def load_quality_from_survivors(
    survivors_file: str,
    return_features: bool = True,
    max_survivors: Optional[int] = None,
    exclude_from_features: Optional[List[str]] = None
) -> Tuple[List[Union[int, Dict[str, Any]]], List[float], Dict[str, Any]]:
    """
    Load survivors and quality scores with range validation.
    
    v3.3.0: Now excludes 'score' and 'confidence' from features by default.
    - 'score' is the Y-LABEL - including it is catastrophic leakage
    - 'confidence' is constant (0.1) - non-informative
    
    Streams the file to avoid memory issues with large files.
    Auto-detects score normalization based on observed range.
    
    Args:
        survivors_file: Path to survivors JSON file
        return_features: If True (default), return full survivor dicts with features.
                        If False, return only seed integers (legacy mode).
        max_survivors: Optional limit on number of survivors to load (for testing).
                      If None, loads all survivors.
        exclude_from_features: Features to exclude from X (default: ['score', 'confidence'])
                              CRITICAL: 'score' must always be excluded (it's the y-label)
        
    Returns:
        Tuple of (survivors, quality, metadata)
        - survivors: List of survivor dicts (if return_features=True) or seed integers
        - quality: List of normalized quality scores [0, 1]
        - metadata: Dict with source info, range, normalization, warnings
    """
    if not IJSON_AVAILABLE:
        raise ImportError("ijson required for streaming quality loading. Install with: pip install ijson")
    
    # v3.3.0: Default exclusions - CRITICAL for preventing label leakage
    if exclude_from_features is None:
        exclude_from_features = DEFAULT_EXCLUDE_FEATURES.copy()
    
    # SAFETY: Always ensure 'score' is excluded (it's the y-label!)
    if 'score' not in exclude_from_features:
        logger.warning("Adding 'score' to exclusion list - it is the y-label!")
        exclude_from_features = list(exclude_from_features) + ['score']
    
    path = Path(survivors_file)
    if not path.exists():
        raise FileNotFoundError(f"Survivors file not found: {survivors_file}")
    
    # Log exclusions
    logger.info(f"Loading survivors from {survivors_file}...")
    logger.info(f"  Excluded from features: {exclude_from_features}")
    
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
    survivors: List[Union[int, Dict[str, Any]]] = []
    quality: List[float] = []
    
    if max_survivors:
        logger.info(f"  Limited to {max_survivors} survivors (--max-survivors)")
    
    with open(path, 'rb') as f:
        parser = ijson.items(f, 'item')
        for i, item in enumerate(parser):
            # Check max_survivors limit
            if max_survivors and i >= max_survivors:
                logger.info(f"  Reached max_survivors limit ({max_survivors})")
                break
            
            # Get seed
            seed = item.get('seed')
            if seed is None:
                continue
            
            # v3.3.0: Return full survivor dict or just seed
            if return_features:
                # v3.3.0: EXCLUDE label and non-informative features
                raw_features = item.get('features', {})
                filtered_features = {
                    k: v for k, v in raw_features.items() 
                    if k not in exclude_from_features
                }
                
                survivor_dict = {
                    'seed': int(seed),
                    'features': filtered_features
                }
                survivors.append(survivor_dict)
            else:
                # Legacy mode: just seed integer
                survivors.append(int(seed))
            
            # Get raw score (always from original features, not filtered)
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
    if return_features and survivors:
        actual_feature_count = len(survivors[0].get('features', {}))
        logger.info(f"  Features per survivor: {actual_feature_count} (after exclusions)")
    
    # Get schema info for metadata (with exclusions applied)
    schema = get_feature_schema_with_hash(survivors_file, exclude_from_features)
    
    # Metadata for sidecar
    metadata = {
        "field": "features.score",
        "observed_min": float(observed_min),
        "observed_max": float(observed_max),
        "observed_range": float(observed_range),
        "normalization_method": normalization,
        "output_range": [0.0, 1.0],
        "sample_size": len(scores_sample),
        "total_samples": len(survivors),
        "warnings": warnings,
        "return_features": return_features,
        "excluded_from_features": exclude_from_features,  # v3.3.0
        "feature_schema": schema  # v3.3.0: Schema reflects exclusions
    }
    
    return survivors, quality, metadata


def load_survivors_with_features(
    survivors_file: str,
    max_survivors: Optional[int] = None,
    exclude_features: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load survivors with their pre-computed features (no quality scores).
    
    v3.3.0: Excludes 'score' and 'confidence' by default.
    
    Args:
        survivors_file: Path to survivors JSON file
        max_survivors: Optional limit on number of survivors to load
        exclude_features: Features to exclude (default: ['score', 'confidence'])
        
    Returns:
        Tuple of (survivors, schema)
        - survivors: List of dicts with 'seed' and 'features' keys
        - schema: Feature schema with hash
    """
    if not IJSON_AVAILABLE:
        raise ImportError("ijson required. Install with: pip install ijson")
    
    if exclude_features is None:
        exclude_features = DEFAULT_EXCLUDE_FEATURES.copy()
    
    path = Path(survivors_file)
    if not path.exists():
        raise FileNotFoundError(f"Survivors file not found: {survivors_file}")
    
    survivors: List[Dict[str, Any]] = []
    
    logger.info(f"Loading survivors with features from {survivors_file}...")
    logger.info(f"  Excluded features: {exclude_features}")
    if max_survivors:
        logger.info(f"  Limited to {max_survivors} survivors")
    
    with open(path, 'rb') as f:
        parser = ijson.items(f, 'item')
        for i, item in enumerate(parser):
            if max_survivors and i >= max_survivors:
                break
            
            seed = item.get('seed')
            if seed is None:
                continue
            
            # v3.3.0: Filter out excluded features
            raw_features = item.get('features', {})
            filtered_features = {
                k: v for k, v in raw_features.items()
                if k not in exclude_features
            }
            
            survivor_dict = {
                'seed': int(seed),
                'features': filtered_features
            }
            survivors.append(survivor_dict)
    
    logger.info(f"Loaded {len(survivors)} survivors with features")
    
    schema = get_feature_schema_with_hash(survivors_file, exclude_features)
    
    return survivors, schema


def extract_feature_matrix(
    survivors: List[Dict[str, Any]],
    feature_names: Optional[List[str]] = None
) -> Tuple[List[List[float]], List[str]]:
    """
    Extract feature matrix from survivor dicts.
    
    Helper to convert survivor dicts to feature matrix.
    Uses lexicographic ordering for consistency.
    
    Args:
        survivors: List of survivor dicts with 'features' key
        feature_names: Optional list of feature names (auto-derived if None)
        
    Returns:
        Tuple of (feature_matrix, feature_names)
        - feature_matrix: List of feature vectors (N x F)
        - feature_names: Sorted list of feature names
    """
    if not survivors:
        raise ValueError("Empty survivors list")
    
    # Derive feature names from first survivor if not provided
    if feature_names is None:
        first_features = survivors[0].get('features', {})
        feature_names = sorted(first_features.keys())
    
    # Extract feature matrix
    feature_matrix = []
    for survivor in survivors:
        features = survivor.get('features', {})
        row = [float(features.get(name, 0.0)) for name in feature_names]
        feature_matrix.append(row)
    
    return feature_matrix, feature_names


def get_seeds_from_survivors(survivors: List[Union[int, Dict[str, Any]]]) -> List[int]:
    """
    Extract seed integers from survivors list.
    
    Helper to handle both legacy (int) and new (dict) formats.
    
    Args:
        survivors: List of seed integers or survivor dicts
        
    Returns:
        List of seed integers
    """
    seeds = []
    for s in survivors:
        if isinstance(s, dict):
            seeds.append(int(s['seed']))
        else:
            seeds.append(int(s))
    return seeds
