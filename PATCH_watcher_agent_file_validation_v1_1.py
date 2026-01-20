#!/usr/bin/env python3
"""
PATCH: File Validation Fix for watcher_agent.py
================================================

Version: 1.1.0 (Post Team Beta Review)
Date: 2026-01-18

Changes from v1.0.0:
  - CRITICAL: Removed torch.load() - replaced with header-only check
  - CRITICAL: Added fnmatch pattern matching for filenames
  - ADVISORY: NPZ validation now uses mmap_mode="r" for lazy loading
  - ADVISORY: Renamed confidence -> validation_confidence for clarity

Problem: Empty files (2 bytes like "[]" or "{}") pass current file_exists validation,
         causing silent pipeline failures downstream.

Solution: Add size thresholds and JSON structure validation with pattern matching.

Installation:
    1. Find the evaluate_file_exists() function in watcher_agent.py
    2. Replace it with the version below
    3. Add the helper functions before evaluate_file_exists()

Author: Team Alpha + Team Beta Review
"""

import os
import json
import fnmatch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

# =============================================================================
# CONFIGURATION - Add to top of watcher_agent.py or to distributed_config.json
# =============================================================================

FILE_VALIDATION_CONFIG = {
    # Minimum file sizes by extension (bytes)
    "min_sizes": {
        ".json": 50,      # Minimum meaningful JSON
        ".npz": 100,      # Minimum meaningful NPZ
        ".pth": 1000,     # PyTorch models are larger
        ".pt": 1000,      # PyTorch models
        ".xgb": 1000,     # XGBoost models
        ".lgb": 1000,     # LightGBM models
        ".cbm": 1000,     # CatBoost models
        "default": 10     # Fallback for unknown extensions
    },
    
    # JSON files that must contain arrays with minimum counts
    # Uses glob/fnmatch patterns for flexible matching
    "json_array_minimums": {
        "bidirectional_survivors*.json": 100,
        "forward_survivors*.json": 100,
        "reverse_survivors*.json": 100,
        "survivors_with_scores*.json": 100,
        "train_history*.json": 10,
        "holdout_history*.json": 5,
    },
    
    # JSON files that must contain specific keys
    # Uses glob/fnmatch patterns for flexible matching
    "json_required_keys": {
        "optimal_window_config*.json": ["window_size", "offset"],
        "optimal_scorer_config*.json": ["best_trial"],
        "best_model*.meta.json": ["model_type", "feature_schema"],
        "reinforcement_engine_config*.json": ["survivor_count"],
        "predictions*.json": ["predictions"],
        "prediction_pool*.json": ["predictions"],
    }
}


# =============================================================================
# HELPER FUNCTIONS - Add before evaluate_file_exists()
# =============================================================================

def get_min_file_size(filepath: str) -> int:
    """Get minimum expected file size based on extension."""
    ext = Path(filepath).suffix.lower()
    return FILE_VALIDATION_CONFIG["min_sizes"].get(
        ext, 
        FILE_VALIDATION_CONFIG["min_sizes"]["default"]
    )


def match_config_by_pattern(filename: str, table: Dict[str, Any]) -> Optional[Any]:
    """
    Match filename against pattern-based config table.
    
    Uses fnmatch for glob-style pattern matching to handle:
    - predictions_latest.json
    - predictions_20260118_143000.json
    - predictions_run42.json
    
    Args:
        filename: Just the filename (not full path)
        table: Dict mapping patterns to values
    
    Returns:
        Matched value or None
    """
    for pattern, value in table.items():
        if fnmatch.fnmatch(filename, pattern):
            return value
    return None


def validate_json_structure(filepath: str) -> Tuple[bool, str]:
    """
    Validate JSON file has meaningful content.
    
    Uses pattern matching for flexible filename handling.
    
    Returns:
        (is_valid, reason)
    """
    filename = Path(filepath).name
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Read error: {e}"
    
    # Check for empty structures
    if data is None:
        return False, "JSON contains null"
    
    if isinstance(data, list):
        if len(data) == 0:
            return False, "JSON array is empty"
        
        # Check minimum counts using pattern matching
        min_count = match_config_by_pattern(
            filename, 
            FILE_VALIDATION_CONFIG["json_array_minimums"]
        )
        if min_count and len(data) < min_count:
            return False, f"JSON array has {len(data)} items, expected >= {min_count}"
    
    elif isinstance(data, dict):
        if len(data) == 0:
            return False, "JSON object is empty"
        
        # Check required keys using pattern matching
        required_keys = match_config_by_pattern(
            filename,
            FILE_VALIDATION_CONFIG["json_required_keys"]
        )
        if required_keys:
            missing = [k for k in required_keys if k not in data]
            if missing:
                return False, f"Missing required keys: {missing}"
    
    return True, "Valid JSON structure"


def validate_file_content(filepath: str) -> Tuple[bool, str]:
    """
    Validate file has meaningful content based on type.
    
    IMPORTANT: WATCHER must NOT fully deserialize models.
    Model loading happens safely in Step 6, not here.
    
    Returns:
        (is_valid, reason)
    """
    ext = Path(filepath).suffix.lower()
    
    if ext == ".json":
        return validate_json_structure(filepath)
    
    elif ext == ".npz":
        try:
            import numpy as np
            # Use mmap_mode="r" for lazy loading - avoids pulling large arrays into memory
            with np.load(filepath, mmap_mode="r") as data:
                if len(data.files) == 0:
                    return False, "NPZ contains no arrays"
                # Check at least one array has data (cheap with mmap)
                for key in data.files:
                    if data[key].size > 0:
                        return True, f"NPZ valid with {len(data.files)} arrays"
                return False, "All NPZ arrays are empty"
        except Exception as e:
            return False, f"NPZ read error: {e}"
    
    elif ext in [".pth", ".pt"]:
        # CRITICAL: Do NOT use torch.load() here!
        # torch.load() executes pickle deserialization which can:
        #   - allocate large memory
        #   - execute arbitrary constructors
        #   - hang or crash WATCHER
        # Full model loading happens safely in Step 6, not WATCHER.
        try:
            with open(filepath, "rb") as f:
                # Just read first few bytes to verify file is readable
                magic = f.read(16)
            if len(magic) < 16:
                return False, "PyTorch file too small to be valid"
            return True, "PyTorch checkpoint present (not deserialized)"
        except Exception as e:
            return False, f"PyTorch file read error: {e}"
    
    elif ext in [".xgb", ".lgb", ".cbm"]:
        # Same principle: don't deserialize, just verify readable
        try:
            with open(filepath, "rb") as f:
                magic = f.read(16)
            if len(magic) < 16:
                return False, f"{ext} model file too small"
            return True, f"{ext} model file present (not deserialized)"
        except Exception as e:
            return False, f"{ext} file read error: {e}"
    
    # For other file types, size check is sufficient
    return True, "Content validation skipped (unknown type)"


# =============================================================================
# REPLACEMENT FUNCTION - Replace existing evaluate_file_exists()
# =============================================================================

def evaluate_file_exists(
    filepath: str,
    validate_content: bool = True,
    min_size_override: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Evaluate if a file exists AND has meaningful content.
    
    This is the FIXED version that catches empty files.
    
    Args:
        filepath: Path to check
        validate_content: Whether to validate JSON/NPZ structure (default: True)
        min_size_override: Override minimum size check (None = use defaults)
    
    Returns:
        (success, reason) tuple
    """
    # Gate 1: File exists
    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"
    
    # Gate 2: Is a file (not directory)
    if not os.path.isfile(filepath):
        return False, f"Path is not a file: {filepath}"
    
    # Gate 3: File size check
    file_size = os.path.getsize(filepath)
    min_size = min_size_override if min_size_override is not None else get_min_file_size(filepath)
    
    if file_size < min_size:
        return False, f"File too small: {file_size} bytes (min: {min_size})"
    
    # Gate 4: Content validation (optional but recommended)
    if validate_content:
        is_valid, content_reason = validate_file_content(filepath)
        if not is_valid:
            return False, f"Content validation failed: {content_reason}"
        # Include content validation details in success message
        return True, f"Valid file: {file_size} bytes - {content_reason}"
    
    return True, f"Valid file: {file_size} bytes"


# =============================================================================
# INTEGRATION WITH HEURISTIC EVALUATION
# =============================================================================

def evaluate_step_outputs(step: int, manifest: Dict[str, Any]) -> Tuple[bool, float, str]:
    """
    Evaluate all outputs for a pipeline step.
    
    Returns:
        (all_valid, validation_confidence, summary)
        
    Note: validation_confidence is about file validation success,
          NOT model prediction confidence. These are distinct concepts.
    """
    outputs = manifest.get("outputs", [])
    if not outputs:
        return True, 0.5, "No outputs to validate"
    
    results = []
    for output in outputs:
        # Handle both string and dict output formats
        if isinstance(output, dict):
            filepath = output.get("file_pattern", output.get("name", ""))
            required = output.get("required", True)
        else:
            filepath = output
            required = True
        
        is_valid, reason = evaluate_file_exists(filepath)
        results.append({
            "file": filepath,
            "valid": is_valid,
            "required": required,
            "reason": reason
        })
    
    # Calculate results
    required_files = [r for r in results if r["required"]]
    required_valid = [r for r in required_files if r["valid"]]
    
    all_required_valid = len(required_valid) == len(required_files)
    
    # validation_confidence = file validation success rate
    # This is NOT the same as model prediction confidence
    if len(required_files) == 0:
        validation_confidence = 0.5
    else:
        validation_confidence = len(required_valid) / len(required_files)
    
    # Build summary
    failed = [r for r in results if not r["valid"]]
    if failed:
        summary = f"Failed: {', '.join(f['file'] + ' (' + f['reason'] + ')' for f in failed)}"
    else:
        summary = f"All {len(results)} outputs valid"
    
    return all_required_valid, validation_confidence, summary


# =============================================================================
# TEST CASES - Run with: python3 PATCH_watcher_agent_file_validation_v1_1.py
# =============================================================================

def run_tests():
    """Test the validation functions."""
    import tempfile
    import shutil
    
    print("=" * 70)
    print("FILE VALIDATION PATCH v1.1.0 - TEST SUITE (Post Team Beta Review)")
    print("=" * 70)
    
    # Create temp directory
    test_dir = tempfile.mkdtemp(prefix="validation_test_")
    
    try:
        # Test 1: Empty JSON array (THE BUG)
        print("\nTest 1: Empty JSON array (should FAIL)")
        empty_array = os.path.join(test_dir, "bidirectional_survivors.json")
        with open(empty_array, 'w') as f:
            f.write("[]")
        result, reason = evaluate_file_exists(empty_array)
        status = "PASS" if not result else "FAIL"
        print(f"  Result: {status} - {reason}")
        assert not result, "Empty array should fail validation!"
        
        # Test 2: Empty JSON object
        print("\nTest 2: Empty JSON object (should FAIL)")
        empty_obj = os.path.join(test_dir, "test.json")
        with open(empty_obj, 'w') as f:
            f.write("{}")
        result, reason = evaluate_file_exists(empty_obj)
        status = "PASS" if not result else "FAIL"
        print(f"  Result: {status} - {reason}")
        assert not result, "Empty object should fail validation!"
        
        # Test 3: Small file below threshold
        print("\nTest 3: Small file below threshold (should FAIL)")
        small_file = os.path.join(test_dir, "tiny.json")
        with open(small_file, 'w') as f:
            f.write('{"a":1}')  # 7 bytes, below 50 byte minimum
        result, reason = evaluate_file_exists(small_file)
        status = "PASS" if not result else "FAIL"
        print(f"  Result: {status} - {reason}")
        assert not result, "Small file should fail validation!"
        
        # Test 4: Valid JSON with content
        print("\nTest 4: Valid JSON with content (should PASS)")
        valid_json = os.path.join(test_dir, "valid.json")
        with open(valid_json, 'w') as f:
            json.dump({"key": "value", "data": list(range(100))}, f)
        result, reason = evaluate_file_exists(valid_json)
        status = "PASS" if result else "FAIL"
        print(f"  Result: {status} - {reason}")
        assert result, "Valid JSON should pass validation!"
        
        # Test 5: Missing required keys
        print("\nTest 5: Missing required keys (should FAIL)")
        missing_keys = os.path.join(test_dir, "optimal_window_config.json")
        with open(missing_keys, 'w') as f:
            json.dump({"some_key": "value", "padding": "x" * 100}, f)
        result, reason = evaluate_file_exists(missing_keys)
        status = "PASS" if not result else "FAIL"
        print(f"  Result: {status} - {reason}")
        assert not result, "Missing required keys should fail!"
        
        # Test 6: Valid config with required keys
        print("\nTest 6: Valid config with required keys (should PASS)")
        valid_config = os.path.join(test_dir, "optimal_window_config.json")
        with open(valid_config, 'w') as f:
            json.dump({
                "window_size": 256,
                "offset": 50,
                "other_data": "x" * 100
            }, f)
        result, reason = evaluate_file_exists(valid_config)
        status = "PASS" if result else "FAIL"
        print(f"  Result: {status} - {reason}")
        assert result, "Valid config should pass!"
        
        # Test 7: Array below minimum count
        print("\nTest 7: Array below minimum count (should FAIL)")
        small_array = os.path.join(test_dir, "bidirectional_survivors.json")
        with open(small_array, 'w') as f:
            json.dump([{"seed": i} for i in range(10)], f)  # Only 10, need 100
        result, reason = evaluate_file_exists(small_array)
        status = "PASS" if not result else "FAIL"
        print(f"  Result: {status} - {reason}")
        assert not result, "Small array should fail minimum count check!"
        
        # Test 8: Non-existent file
        print("\nTest 8: Non-existent file (should FAIL)")
        result, reason = evaluate_file_exists("/nonexistent/file.json")
        status = "PASS" if not result else "FAIL"
        print(f"  Result: {status} - {reason}")
        assert not result, "Non-existent file should fail!"
        
        # Test 9: PATTERN MATCHING - Timestamped predictions file
        print("\nTest 9: Timestamped predictions file (pattern matching)")
        timestamped = os.path.join(test_dir, "predictions_20260118_143000.json")
        with open(timestamped, 'w') as f:
            json.dump({"predictions": [1, 2, 3], "padding": "x" * 100}, f)
        result, reason = evaluate_file_exists(timestamped)
        status = "PASS" if result else "FAIL"
        print(f"  Result: {status} - {reason}")
        assert result, "Timestamped predictions should pass with pattern matching!"
        
        # Test 10: PATTERN MATCHING - Timestamped predictions missing key
        print("\nTest 10: Timestamped predictions missing required key (should FAIL)")
        bad_timestamped = os.path.join(test_dir, "predictions_run42.json")
        with open(bad_timestamped, 'w') as f:
            json.dump({"wrong_key": [1, 2, 3], "padding": "x" * 100}, f)
        result, reason = evaluate_file_exists(bad_timestamped)
        status = "PASS" if not result else "FAIL"
        print(f"  Result: {status} - {reason}")
        assert not result, "Timestamped predictions without 'predictions' key should fail!"
        
        # Test 11: PyTorch file (header check only, no torch.load)
        print("\nTest 11: PyTorch file validation (header only, no deserialization)")
        fake_pth = os.path.join(test_dir, "model.pth")
        with open(fake_pth, 'wb') as f:
            f.write(b'\x80\x04\x95' + b'\x00' * 1000)  # Fake pickle header + padding
        result, reason = evaluate_file_exists(fake_pth)
        status = "PASS" if result else "FAIL"
        print(f"  Result: {status} - {reason}")
        assert result, "PyTorch file with valid header should pass!"
        assert "not deserialized" in reason, "Should confirm no deserialization!"
        
        # Test 12: Pattern matching helper function
        print("\nTest 12: Pattern matching helper function")
        test_cases = [
            ("predictions_latest.json", "predictions*.json", True),
            ("predictions_20260118.json", "predictions*.json", True),
            ("prediction_pool_v2.json", "prediction_pool*.json", True),
            ("other_file.json", "predictions*.json", False),
            ("bidirectional_survivors_backup.json", "bidirectional_survivors*.json", True),
        ]
        all_passed = True
        for filename, pattern, should_match in test_cases:
            table = {pattern: ["test_key"]}
            result = match_config_by_pattern(filename, table)
            matched = result is not None
            if matched == should_match:
                print(f"    ✅ '{filename}' vs '{pattern}' = {matched}")
            else:
                print(f"    ❌ '{filename}' vs '{pattern}' = {matched} (expected {should_match})")
                all_passed = False
        assert all_passed, "Pattern matching tests failed!"
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✅")
        print("=" * 70)
        print("\nChanges in v1.1.0 (Team Beta Review):")
        print("  ✅ Removed torch.load() - header check only")
        print("  ✅ Added fnmatch pattern matching for filenames")
        print("  ✅ NPZ uses mmap_mode='r' for lazy loading")
        print("  ✅ Renamed confidence -> validation_confidence")
        print("=" * 70)
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    run_tests()
