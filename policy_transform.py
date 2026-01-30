#!/usr/bin/env python3
"""
policy_transform.py — Phase 9B.1: Policy-Conditioned Data Transforms

Version: 1.0.0
Created: 2026-01-30
Team Beta Approved: 2026-01-30

This module implements pure functional transforms for policy-conditioned learning.
Policies decide what data is seen next, not outcomes.

INVARIANTS (violation = PolicyViolationError):
1. Stateless: Same inputs → Same outputs (deterministic)
2. Never fabricates: Only filters/weights/masks/windows existing data
3. Preserves originals: Returns new list, doesn't mutate input
4. Auditable: Logs every transformation applied
5. Safe: Never drops below ABSOLUTE_MIN_SURVIVORS

TRANSFORM ORDER (fixed, not policy-specified):
    filter → weight → mask → window

AUTHORITY BOUNDARIES:
- This module has NO decision authority
- Chapter 13 validates policies
- WATCHER executes policies
- Selfplay proposes policies
"""

import hashlib
import json
import copy
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timezone


# =============================================================================
# CONSTANTS
# =============================================================================

VERSION = "1.0.0"

# Global safety floor — policies cannot go below this
ABSOLUTE_MIN_SURVIVORS = 50

# Fields that CANNOT be masked (per Team Beta ruling)
FORBIDDEN_MASK_FIELDS = frozenset({"score", "holdout_hits", "seed"})

# Valid filter operators
VALID_OPERATORS = frozenset({"gt", "gte", "lt", "lte", "eq"})

# Valid weight methods
VALID_WEIGHT_METHODS = frozenset({"linear", "exponential", "step"})


# =============================================================================
# EXCEPTIONS
# =============================================================================

class PolicyViolationError(RuntimeError):
    """
    Raised when a policy would violate Phase 9B invariants.
    
    Use cases:
    - min_survivors < ABSOLUTE_MIN_SURVIVORS
    - Empty survivors after transform
    - Invalid field paths
    - Attempt to mask forbidden fields
    - Invalid operator or method
    """
    pass


class PolicyValidationError(ValueError):
    """
    Raised when a policy fails schema validation.
    
    Less severe than PolicyViolationError — indicates malformed input
    rather than invariant violation.
    """
    pass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PolicyTransformResult:
    """Result of applying a policy to survivors."""
    
    survivors: List[Dict]           # Transformed survivors
    original_count: int             # Before transform
    filtered_count: int             # After transform
    transform_log: List[str]        # Audit trail of operations
    policy_fingerprint: str         # Hash for deduplication
    
    # Detailed metrics
    filter_removed: int = 0
    weight_adjusted: int = 0
    mask_features_removed: int = 0
    window_trimmed: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize for telemetry/logging."""
        return {
            "original_count": self.original_count,
            "filtered_count": self.filtered_count,
            "transform_log": self.transform_log,
            "policy_fingerprint": self.policy_fingerprint,
            "metrics": {
                "filter_removed": self.filter_removed,
                "weight_adjusted": self.weight_adjusted,
                "mask_features_removed": self.mask_features_removed,
                "window_trimmed": self.window_trimmed,
            }
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_nested_value(d: Dict, path: str, default: Any = None) -> Any:
    """
    Get a nested value from a dict using dot notation.
    
    Example:
        _get_nested_value({"features": {"score": 0.5}}, "features.score")
        → 0.5
    """
    keys = path.split(".")
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _set_nested_value(d: Dict, path: str, value: Any) -> Dict:
    """
    Set a nested value in a dict using dot notation.
    Returns a NEW dict (does not mutate input).
    """
    keys = path.split(".")
    result = copy.deepcopy(d)
    current = result
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return result


def _normalize_values(values: List[float]) -> Tuple[List[float], float, float]:
    """
    Normalize values to 0-1 range.
    
    Returns:
        (normalized_values, min_val, max_val)
    """
    if not values:
        return [], 0.0, 0.0
    
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    
    if range_val == 0:
        # All values identical — return 0.5 for all
        return [0.5] * len(values), min_val, max_val
    
    normalized = [(v - min_val) / range_val for v in values]
    return normalized, min_val, max_val


# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def _apply_filter(
    survivors: List[Dict],
    config: Dict,
    strict_mode: bool = True
) -> Tuple[List[Dict], str, int]:
    """
    Filter survivors based on field threshold.
    
    Config schema:
    {
        "enabled": bool,
        "field": str,              # e.g., "features.holdout_hits"
        "operator": str,           # "gt", "gte", "lt", "lte", "eq"
        "threshold": float,
        "min_survivors": int       # Never drop below this (policy-level)
    }
    
    Returns:
        (filtered_survivors, log_message, removed_count)
    """
    if not config.get("enabled", False):
        return survivors, "filter: skipped (disabled)", 0
    
    # Validate config
    field_path = config.get("field")
    if not field_path:
        if strict_mode:
            raise PolicyValidationError("filter: missing 'field' in config")
        return survivors, "filter: skipped (missing field)", 0
    
    operator = config.get("operator", "gte")
    if operator not in VALID_OPERATORS:
        if strict_mode:
            raise PolicyValidationError(f"filter: invalid operator '{operator}'")
        return survivors, f"filter: skipped (invalid operator '{operator}')", 0
    
    threshold = config.get("threshold", 0.0)
    policy_min = config.get("min_survivors", ABSOLUTE_MIN_SURVIVORS)
    
    # Enforce absolute floor
    effective_min = max(policy_min, ABSOLUTE_MIN_SURVIVORS)
    
    # Define filter predicate
    def passes_filter(s: Dict) -> bool:
        value = _get_nested_value(s, field_path)
        if value is None:
            return False  # Missing field = excluded
        
        if operator == "gt":
            return value > threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "lte":
            return value <= threshold
        elif operator == "eq":
            return abs(value - threshold) < 1e-9
        return True
    
    # Apply filter
    filtered = [s for s in survivors if passes_filter(s)]
    removed = len(survivors) - len(filtered)
    
    # Safety: Never drop below min_survivors
    if len(filtered) < effective_min and len(survivors) >= effective_min:
        # Sort by field value, keep top effective_min
        # Direction depends on operator
        reverse = operator in ("gt", "gte")
        
        def sort_key(s):
            val = _get_nested_value(s, field_path)
            return val if val is not None else float('-inf')
        
        sorted_survivors = sorted(survivors, key=sort_key, reverse=reverse)
        filtered = sorted_survivors[:effective_min]
        removed = len(survivors) - len(filtered)
        
        log = (f"filter: {len(survivors)} → {len(filtered)} "
               f"(field={field_path}, {operator} {threshold}, "
               f"safety floor applied: min={effective_min})")
    else:
        log = (f"filter: {len(survivors)} → {len(filtered)} "
               f"(field={field_path}, {operator} {threshold})")
    
    # Final safety check
    if len(filtered) < ABSOLUTE_MIN_SURVIVORS:
        raise PolicyViolationError(
            f"filter: would reduce survivors to {len(filtered)}, "
            f"below absolute minimum {ABSOLUTE_MIN_SURVIVORS}"
        )
    
    return filtered, log, removed


def _apply_weight(
    survivors: List[Dict],
    config: Dict,
    strict_mode: bool = True
) -> Tuple[List[Dict], str, int]:
    """
    Adjust survivor scores based on field value.
    
    Config schema:
    {
        "enabled": bool,
        "field": str,              # Field to use for weighting
        "method": str,             # "linear", "exponential", "step"
        "params": {
            # linear: score *= (1 + alpha * normalized_value)
            "alpha": float,        # Strength of adjustment
            
            # exponential: score *= exp(alpha * normalized_value)
            "alpha": float,
            
            # step: score *= multiplier if value > threshold
            "threshold": float,
            "multiplier": float
        }
    }
    
    NOTE: Values are always normalized to 0-1 before applying.
    
    Returns:
        (weighted_survivors, log_message, adjusted_count)
    """
    if not config.get("enabled", False):
        return survivors, "weight: skipped (disabled)", 0
    
    field_path = config.get("field")
    if not field_path:
        if strict_mode:
            raise PolicyValidationError("weight: missing 'field' in config")
        return survivors, "weight: skipped (missing field)", 0
    
    method = config.get("method", "linear")
    if method not in VALID_WEIGHT_METHODS:
        if strict_mode:
            raise PolicyValidationError(f"weight: invalid method '{method}'")
        return survivors, f"weight: skipped (invalid method '{method}')", 0
    
    params = config.get("params", {})
    
    # Extract field values for normalization
    raw_values = []
    for s in survivors:
        val = _get_nested_value(s, field_path)
        raw_values.append(val if val is not None else 0.0)
    
    # Normalize to 0-1
    normalized, min_val, max_val = _normalize_values(raw_values)
    
    # Apply weighting
    weighted = []
    adjusted_count = 0
    
    for i, s in enumerate(survivors):
        new_s = copy.deepcopy(s)
        old_score = new_s.get("score", 0.0)
        norm_val = normalized[i]
        
        if method == "linear":
            alpha = params.get("alpha", 0.1)
            multiplier = 1.0 + alpha * norm_val
        elif method == "exponential":
            alpha = params.get("alpha", 0.1)
            multiplier = math.exp(alpha * norm_val)
        elif method == "step":
            threshold = params.get("threshold", 0.5)
            step_mult = params.get("multiplier", 1.5)
            # Use raw value for step comparison
            raw_val = _get_nested_value(s, field_path) or 0.0
            multiplier = step_mult if raw_val > threshold else 1.0
        else:
            multiplier = 1.0
        
        new_score = old_score * multiplier
        new_s["score"] = new_score
        
        if abs(multiplier - 1.0) > 1e-9:
            adjusted_count += 1
        
        weighted.append(new_s)
    
    log = (f"weight: method={method}, field={field_path}, "
           f"norm_range=[{min_val:.4f}, {max_val:.4f}], "
           f"adjusted={adjusted_count}/{len(survivors)}")
    
    return weighted, log, adjusted_count


def _apply_mask(
    survivors: List[Dict],
    config: Dict,
    strict_mode: bool = True
) -> Tuple[List[Dict], str, int]:
    """
    Remove specified features from each survivor.
    
    Config schema:
    {
        "enabled": bool,
        "exclude_features": list[str]  # Feature names to exclude
    }
    
    NOTE: Only features.* can be masked. score, holdout_hits, seed are FORBIDDEN.
    
    Returns:
        (masked_survivors, log_message, features_removed_count)
    """
    if not config.get("enabled", False):
        return survivors, "mask: skipped (disabled)", 0
    
    exclude = set(config.get("exclude_features", []))
    
    if not exclude:
        return survivors, "mask: skipped (empty exclude list)", 0
    
    # Check for forbidden fields
    forbidden_found = exclude & FORBIDDEN_MASK_FIELDS
    if forbidden_found:
        raise PolicyViolationError(
            f"mask: cannot mask forbidden fields: {forbidden_found}. "
            f"Only features.* can be masked."
        )
    
    # Apply masking
    masked = []
    total_removed = 0
    
    for s in survivors:
        new_s = copy.deepcopy(s)
        features = new_s.get("features", {})
        
        # Remove excluded features
        removed_this = 0
        for feat in list(features.keys()):
            if feat in exclude:
                del features[feat]
                removed_this += 1
        
        new_s["features"] = features
        total_removed += removed_this
        masked.append(new_s)
    
    log = f"mask: excluded {len(exclude)} feature names, removed {total_removed} total values"
    
    return masked, log, total_removed


def _apply_window(
    survivors: List[Dict],
    config: Dict,
    strict_mode: bool = True
) -> Tuple[List[Dict], str, int]:
    """
    Restrict to index range (for temporal analysis).
    
    Config schema:
    {
        "enabled": bool,
        "start_index": int | None,  # None = from beginning
        "end_index": int | None     # None = to end
    }
    
    NOTE: Assumes survivors are ordered by some temporal or quality key.
    
    Returns:
        (windowed_survivors, log_message, trimmed_count)
    """
    if not config.get("enabled", False):
        return survivors, "window: skipped (disabled)", 0
    
    start = config.get("start_index") or 0
    end = config.get("end_index") or len(survivors)
    
    # Clamp to valid range
    start = max(0, start)
    end = min(len(survivors), end)
    
    if start >= end:
        raise PolicyViolationError(
            f"window: invalid range [{start}:{end}] would produce empty result"
        )
    
    windowed = survivors[start:end]
    trimmed = len(survivors) - len(windowed)
    
    # Safety check
    if len(windowed) < ABSOLUTE_MIN_SURVIVORS:
        raise PolicyViolationError(
            f"window: would reduce survivors to {len(windowed)}, "
            f"below absolute minimum {ABSOLUTE_MIN_SURVIVORS}"
        )
    
    log = f"window: [{start}:{end}] ({len(windowed)} survivors, trimmed {trimmed})"
    
    return windowed, log, trimmed


# =============================================================================
# MAIN API
# =============================================================================

def apply_policy(
    survivors: List[Dict],
    policy: Dict,
    *,
    strict_mode: bool = True
) -> PolicyTransformResult:
    """
    Phase 9B.1: Pure functional transform.
    
    INVARIANTS:
    - Stateless: Same inputs → Same outputs (deterministic)
    - Never fabricates: Only filters/weights/masks/windows existing data
    - Preserves originals: Returns new list, doesn't mutate input
    - Auditable: Logs every transformation applied
    
    Operations (fixed order: filter → weight → mask → window):
    - filter: Remove survivors below threshold
    - weight: Adjust survivor scores
    - mask: Hide certain features
    - window: Restrict to index range
    
    Args:
        survivors: List of survivor dicts with seed, score, features
        policy: Policy dict matching PolicySchema
        strict_mode: If True, raise on invalid policy; if False, skip invalid ops
    
    Returns:
        PolicyTransformResult with transformed survivors and metadata
    
    Raises:
        PolicyViolationError: If transform would violate invariants
        PolicyValidationError: If policy is malformed and strict_mode=True
    """
    if not survivors:
        raise PolicyViolationError("apply_policy: empty survivors list")
    
    original_count = len(survivors)
    transform_log = []
    
    # Compute fingerprint first (before any transforms)
    fingerprint = compute_policy_fingerprint(policy)
    transform_log.append(f"policy_fingerprint: {fingerprint}")
    
    # Get transforms config
    transforms = policy.get("transforms", {})
    
    # Work on a copy (never mutate input)
    current = copy.deepcopy(survivors)
    
    # Metrics tracking
    filter_removed = 0
    weight_adjusted = 0
    mask_features_removed = 0
    window_trimmed = 0
    
    # === FIXED ORDER: filter → weight → mask → window ===
    
    # 1. Filter
    filter_config = transforms.get("filter", {})
    current, log, removed = _apply_filter(current, filter_config, strict_mode)
    transform_log.append(log)
    filter_removed = removed
    
    # 2. Weight
    weight_config = transforms.get("weight", {})
    current, log, adjusted = _apply_weight(current, weight_config, strict_mode)
    transform_log.append(log)
    weight_adjusted = adjusted
    
    # 3. Mask
    mask_config = transforms.get("mask", {})
    current, log, masked = _apply_mask(current, mask_config, strict_mode)
    transform_log.append(log)
    mask_features_removed = masked
    
    # 4. Window
    window_config = transforms.get("window", {})
    current, log, trimmed = _apply_window(current, window_config, strict_mode)
    transform_log.append(log)
    window_trimmed = trimmed
    
    # Final safety check
    if len(current) < ABSOLUTE_MIN_SURVIVORS:
        raise PolicyViolationError(
            f"apply_policy: final survivor count {len(current)} "
            f"below absolute minimum {ABSOLUTE_MIN_SURVIVORS}"
        )
    
    return PolicyTransformResult(
        survivors=current,
        original_count=original_count,
        filtered_count=len(current),
        transform_log=transform_log,
        policy_fingerprint=fingerprint,
        filter_removed=filter_removed,
        weight_adjusted=weight_adjusted,
        mask_features_removed=mask_features_removed,
        window_trimmed=window_trimmed,
    )


def compute_policy_fingerprint(policy: Dict) -> str:
    """
    Compute deterministic hash of policy parameters.
    
    Used for:
    - Detecting semantic duplicates
    - Cache key for memoization
    - Lineage tracking
    
    INCLUDES (per Team Beta):
    - All transform parameters
    - Safety parameters (min_survivors, etc.)
    
    EXCLUDES:
    - policy_id
    - parent_policy_id
    - created_at
    - episode_number
    - fitness metrics
    """
    transforms = policy.get("transforms", {})
    
    # Build canonical representation
    canonical = {
        "transforms": {
            "filter": {
                "enabled": transforms.get("filter", {}).get("enabled", False),
                "field": transforms.get("filter", {}).get("field"),
                "operator": transforms.get("filter", {}).get("operator"),
                "threshold": transforms.get("filter", {}).get("threshold"),
                "min_survivors": transforms.get("filter", {}).get("min_survivors"),
            },
            "weight": {
                "enabled": transforms.get("weight", {}).get("enabled", False),
                "field": transforms.get("weight", {}).get("field"),
                "method": transforms.get("weight", {}).get("method"),
                "params": transforms.get("weight", {}).get("params", {}),
            },
            "mask": {
                "enabled": transforms.get("mask", {}).get("enabled", False),
                "exclude_features": sorted(
                    transforms.get("mask", {}).get("exclude_features", [])
                ),
            },
            "window": {
                "enabled": transforms.get("window", {}).get("enabled", False),
                "start_index": transforms.get("window", {}).get("start_index"),
                "end_index": transforms.get("window", {}).get("end_index"),
            },
        }
    }
    
    # Deterministic JSON serialization
    canonical_str = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    
    # SHA256, truncated to 16 chars
    return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()[:16]


def validate_policy_schema(policy: Dict) -> Tuple[bool, List[str]]:
    """
    Validate policy against schema.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check top-level structure
    if not isinstance(policy, dict):
        errors.append("policy must be a dict")
        return False, errors
    
    transforms = policy.get("transforms", {})
    
    # Validate filter
    filter_cfg = transforms.get("filter", {})
    if filter_cfg.get("enabled"):
        if not filter_cfg.get("field"):
            errors.append("filter.field is required when enabled")
        if filter_cfg.get("operator") and filter_cfg["operator"] not in VALID_OPERATORS:
            errors.append(f"filter.operator must be one of {VALID_OPERATORS}")
        min_surv = filter_cfg.get("min_survivors")
        if min_surv is not None and min_surv < ABSOLUTE_MIN_SURVIVORS:
            errors.append(
                f"filter.min_survivors ({min_surv}) cannot be below "
                f"ABSOLUTE_MIN_SURVIVORS ({ABSOLUTE_MIN_SURVIVORS})"
            )
    
    # Validate weight
    weight_cfg = transforms.get("weight", {})
    if weight_cfg.get("enabled"):
        if not weight_cfg.get("field"):
            errors.append("weight.field is required when enabled")
        if weight_cfg.get("method") and weight_cfg["method"] not in VALID_WEIGHT_METHODS:
            errors.append(f"weight.method must be one of {VALID_WEIGHT_METHODS}")
    
    # Validate mask
    mask_cfg = transforms.get("mask", {})
    if mask_cfg.get("enabled"):
        exclude = set(mask_cfg.get("exclude_features", []))
        forbidden = exclude & FORBIDDEN_MASK_FIELDS
        if forbidden:
            errors.append(f"mask.exclude_features contains forbidden fields: {forbidden}")
    
    # Validate window
    window_cfg = transforms.get("window", {})
    if window_cfg.get("enabled"):
        start = window_cfg.get("start_index", 0) or 0
        end = window_cfg.get("end_index")
        if end is not None and start >= end:
            errors.append(f"window: start_index ({start}) must be < end_index ({end})")
    
    return len(errors) == 0, errors


def create_empty_policy(policy_id: str = None) -> Dict:
    """
    Create a minimal valid policy with all transforms disabled.
    
    Useful for:
    - Baseline comparisons
    - Testing
    - Initial episode (no conditioning)
    """
    return {
        "policy_id": policy_id or f"policy_empty_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "parent_policy_id": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episode_number": 0,
        "transforms": {
            "filter": {"enabled": False},
            "weight": {"enabled": False},
            "mask": {"enabled": False},
            "window": {"enabled": False},
        },
        "fitness": None,
        "fitness_delta": None,
        "val_r2": None,
        "train_val_gap": None,
        "survivor_count_after": None,
        "fingerprint": None,  # Computed on apply
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI interface for testing and validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 9B.1: Policy Transform Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run unit tests
  python policy_transform.py --test
  
  # Validate a policy file
  python policy_transform.py --validate policy.json
  
  # Apply policy to survivors
  python policy_transform.py --apply policy.json --survivors survivors.json --output result.json
        """
    )
    
    parser.add_argument('--test', action='store_true',
                        help='Run unit tests')
    parser.add_argument('--validate', type=str,
                        help='Validate a policy JSON file')
    parser.add_argument('--apply', type=str,
                        help='Apply a policy JSON file')
    parser.add_argument('--survivors', type=str,
                        help='Survivors JSON file (for --apply)')
    parser.add_argument('--output', type=str,
                        help='Output file for results')
    parser.add_argument('--version', action='store_true',
                        help='Show version')
    
    args = parser.parse_args()
    
    if args.version:
        print(f"policy_transform.py v{VERSION}")
        return
    
    if args.test:
        run_tests()
        return
    
    if args.validate:
        with open(args.validate) as f:
            policy = json.load(f)
        is_valid, errors = validate_policy_schema(policy)
        if is_valid:
            print(f"✅ Policy is valid")
            print(f"   Fingerprint: {compute_policy_fingerprint(policy)}")
        else:
            print(f"❌ Policy validation failed:")
            for err in errors:
                print(f"   - {err}")
        return
    
    if args.apply:
        if not args.survivors:
            print("Error: --survivors required with --apply")
            return
        
        with open(args.apply) as f:
            policy = json.load(f)
        with open(args.survivors) as f:
            survivors = json.load(f)
        
        result = apply_policy(survivors, policy)
        
        print(f"Transform complete:")
        print(f"  Original: {result.original_count}")
        print(f"  Final: {result.filtered_count}")
        print(f"  Fingerprint: {result.policy_fingerprint}")
        print(f"\nTransform log:")
        for log in result.transform_log:
            print(f"  {log}")
        
        if args.output:
            output = {
                "survivors": result.survivors,
                "metadata": result.to_dict()
            }
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nOutput written to: {args.output}")
        
        return
    
    parser.print_help()


# =============================================================================
# UNIT TESTS
# =============================================================================

def run_tests():
    """Run unit tests for policy_transform module."""
    print("=" * 60)
    print("POLICY TRANSFORM — UNIT TESTS")
    print("=" * 60)
    print()
    
    passed = 0
    failed = 0
    
    # Test helpers
    def assert_eq(actual, expected, msg):
        nonlocal passed, failed
        if actual == expected:
            print(f"  ✅ {msg}")
            passed += 1
        else:
            print(f"  ❌ {msg}")
            print(f"     Expected: {expected}")
            print(f"     Actual:   {actual}")
            failed += 1
    
    def assert_true(condition, msg):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {msg}")
            passed += 1
        else:
            print(f"  ❌ {msg}")
            failed += 1
    
    def assert_raises(exc_type, func, msg):
        nonlocal passed, failed
        try:
            func()
            print(f"  ❌ {msg} (no exception raised)")
            failed += 1
        except exc_type:
            print(f"  ✅ {msg}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {msg} (wrong exception: {type(e).__name__})")
            failed += 1
    
    # === Test Data ===
    def make_test_survivors(n=100):
        """Generate test survivor data."""
        return [
            {
                "seed": 1000 + i,
                "score": 50.0 + i * 0.5,
                "holdout_hits": 0.3 + (i / n) * 0.5,
                "features": {
                    "temporal_stability": 0.5 + (i / n) * 0.4,
                    "residue_coherence": 0.1 + (i / n) * 0.3,
                    "lane_agreement": 0.05 + (i / n) * 0.1,
                }
            }
            for i in range(n)
        ]
    
    # === Test 1: Empty policy passthrough ===
    print("[Test 1] Empty policy passthrough")
    survivors = make_test_survivors(100)
    policy = create_empty_policy()
    result = apply_policy(survivors, policy)
    assert_eq(result.original_count, 100, "original_count = 100")
    assert_eq(result.filtered_count, 100, "filtered_count = 100 (no change)")
    assert_eq(len(result.survivors), 100, "survivors unchanged")
    print()
    
    # === Test 2: Filter transform ===
    print("[Test 2] Filter transform")
    survivors = make_test_survivors(100)
    policy = {
        "transforms": {
            "filter": {
                "enabled": True,
                "field": "holdout_hits",
                "operator": "gte",
                "threshold": 0.6,
                "min_survivors": 50
            }
        }
    }
    result = apply_policy(survivors, policy)
    assert_true(result.filtered_count < 100, f"filtered_count < 100 (actual: {result.filtered_count})")
    assert_true(result.filtered_count >= 50, f"filtered_count >= 50 (safety floor)")
    # All remaining survivors should have holdout_hits >= 0.6 OR be top-50 by holdout_hits
    print()
    
    # === Test 3: Weight transform (linear) ===
    print("[Test 3] Weight transform (linear)")
    survivors = make_test_survivors(100)
    original_scores = [s["score"] for s in survivors]
    policy = {
        "transforms": {
            "weight": {
                "enabled": True,
                "field": "features.temporal_stability",
                "method": "linear",
                "params": {"alpha": 0.5}
            }
        }
    }
    result = apply_policy(survivors, policy)
    # Scores should be adjusted
    new_scores = [s["score"] for s in result.survivors]
    scores_changed = any(abs(new_scores[i] - original_scores[i]) > 0.001 for i in range(len(new_scores)))
    assert_true(scores_changed, "scores were adjusted")
    print()
    
    # === Test 4: Mask transform ===
    print("[Test 4] Mask transform")
    survivors = make_test_survivors(100)
    policy = {
        "transforms": {
            "mask": {
                "enabled": True,
                "exclude_features": ["lane_agreement", "residue_coherence"]
            }
        }
    }
    result = apply_policy(survivors, policy)
    first_features = result.survivors[0]["features"]
    assert_true("lane_agreement" not in first_features, "lane_agreement masked")
    assert_true("residue_coherence" not in first_features, "residue_coherence masked")
    assert_true("temporal_stability" in first_features, "temporal_stability preserved")
    print()
    
    # === Test 5: Window transform ===
    print("[Test 5] Window transform")
    survivors = make_test_survivors(100)
    policy = {
        "transforms": {
            "window": {
                "enabled": True,
                "start_index": 20,
                "end_index": 80
            }
        }
    }
    result = apply_policy(survivors, policy)
    assert_eq(result.filtered_count, 60, "windowed to 60 survivors")
    assert_eq(result.survivors[0]["seed"], 1020, "first seed is 1020 (index 20)")
    print()
    
    # === Test 6: Combined transforms ===
    print("[Test 6] Combined transforms (fixed order)")
    survivors = make_test_survivors(200)
    policy = {
        "transforms": {
            "filter": {
                "enabled": True,
                "field": "holdout_hits",
                "operator": "gte",
                "threshold": 0.5,
                "min_survivors": 80
            },
            "weight": {
                "enabled": True,
                "field": "features.temporal_stability",
                "method": "linear",
                "params": {"alpha": 0.3}
            },
            "mask": {
                "enabled": True,
                "exclude_features": ["lane_agreement"]
            },
            "window": {
                "enabled": True,
                "start_index": 0,
                "end_index": 100
            }
        }
    }
    result = apply_policy(survivors, policy)
    assert_true(result.filtered_count >= ABSOLUTE_MIN_SURVIVORS, "above safety floor")
    assert_true("lane_agreement" not in result.survivors[0]["features"], "mask applied")
    print()
    
    # === Test 7: Fingerprint determinism ===
    print("[Test 7] Fingerprint determinism")
    policy1 = {
        "policy_id": "test_1",
        "created_at": "2026-01-01T00:00:00Z",
        "transforms": {
            "filter": {"enabled": True, "field": "score", "operator": "gte", "threshold": 0.5}
        }
    }
    policy2 = {
        "policy_id": "test_2",  # Different ID
        "created_at": "2026-01-02T00:00:00Z",  # Different timestamp
        "transforms": {
            "filter": {"enabled": True, "field": "score", "operator": "gte", "threshold": 0.5}
        }
    }
    fp1 = compute_policy_fingerprint(policy1)
    fp2 = compute_policy_fingerprint(policy2)
    assert_eq(fp1, fp2, "same params → same fingerprint (ignores ID/timestamp)")
    print()
    
    # === Test 8: Fingerprint sensitivity ===
    print("[Test 8] Fingerprint sensitivity")
    policy_a = {
        "transforms": {
            "filter": {"enabled": True, "field": "score", "threshold": 0.5}
        }
    }
    policy_b = {
        "transforms": {
            "filter": {"enabled": True, "field": "score", "threshold": 0.6}  # Different!
        }
    }
    fp_a = compute_policy_fingerprint(policy_a)
    fp_b = compute_policy_fingerprint(policy_b)
    assert_true(fp_a != fp_b, "different params → different fingerprint")
    print()
    
    # === Test 9: Validation ===
    print("[Test 9] Schema validation")
    valid_policy = {
        "transforms": {
            "filter": {"enabled": True, "field": "score", "operator": "gte", "threshold": 0.5}
        }
    }
    is_valid, errors = validate_policy_schema(valid_policy)
    assert_true(is_valid, "valid policy passes validation")
    
    invalid_policy = {
        "transforms": {
            "filter": {"enabled": True}  # Missing required field
        }
    }
    is_valid, errors = validate_policy_schema(invalid_policy)
    assert_true(not is_valid, "invalid policy fails validation")
    print()
    
    # === Test 10: Safety violations ===
    print("[Test 10] Safety violations")
    
    # Test empty survivors
    assert_raises(
        PolicyViolationError,
        lambda: apply_policy([], create_empty_policy()),
        "empty survivors raises PolicyViolationError"
    )
    
    # Test masking forbidden fields
    def try_mask_forbidden():
        survivors = make_test_survivors(100)
        policy = {
            "transforms": {
                "mask": {"enabled": True, "exclude_features": ["score"]}
            }
        }
        apply_policy(survivors, policy)
    
    assert_raises(
        PolicyViolationError,
        try_mask_forbidden,
        "masking 'score' raises PolicyViolationError"
    )
    print()
    
    # === Test 11: Input immutability ===
    print("[Test 11] Input immutability")
    survivors = make_test_survivors(100)
    original_first_score = survivors[0]["score"]
    policy = {
        "transforms": {
            "weight": {
                "enabled": True,
                "field": "features.temporal_stability",
                "method": "linear",
                "params": {"alpha": 1.0}
            }
        }
    }
    result = apply_policy(survivors, policy)
    assert_eq(survivors[0]["score"], original_first_score, "original survivors unchanged")
    print()
    
    # === Summary ===
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED")
    else:
        print(f"\n❌ {failed} TESTS FAILED")
    
    return failed == 0


if __name__ == "__main__":
    main()
