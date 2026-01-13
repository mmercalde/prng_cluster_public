#!/usr/bin/env python3
"""
Chapter 13 Diagnostics Engine — Phase 2
Compares predictions to reality and generates structured diagnostics

RESPONSIBILITIES:
1. Compare Step 6 predictions against actual draw outcome
2. Compute confidence calibration metrics
3. Track survivor performance (hits, decays, reinforcements)
4. Detect feature drift and pipeline health issues
5. Generate summary flags and recommended actions
6. Archive diagnostics history

VERSION: 1.0.0
DATE: 2026-01-12
DEPENDS ON: prediction_pool.json, lottery_history.json, survivors_with_scores.json, watcher_policies.json

INPUTS (Read-Only):
- prediction_pool.json (Step 6) — Predictions to validate
- confidence_map.json (Step 6) — Confidence scores
- lottery_history.json — Ground truth with latest draw
- best_model.meta.json (Step 5) — Model provenance
- survivors_with_scores.json (Step 3) — Current survivor state
- watcher_policies.json — Thresholds for flags

OUTPUTS:
- post_draw_diagnostics.json — Current run diagnostics
- diagnostics_history/<timestamp>_diagnostics.json — Archived copy
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from statistics import mean, median, stdev
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

SCHEMA_VERSION = "1.0.0"

# Default file paths
DEFAULT_PREDICTION_POOL = "prediction_pool.json"
DEFAULT_CONFIDENCE_MAP = "confidence_map.json"
DEFAULT_LOTTERY_HISTORY = "lottery_history.json"
DEFAULT_MODEL_META = "best_model.meta.json"
DEFAULT_SURVIVORS = "survivors_with_scores.json"
DEFAULT_POLICIES = "watcher_policies.json"
DEFAULT_OUTPUT = "post_draw_diagnostics.json"
DEFAULT_HISTORY_DIR = "diagnostics_history"

# Previous diagnostics for comparison
PREVIOUS_DIAGNOSTICS = ".previous_diagnostics.json"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_json_safe(path: str, default: Any = None) -> Any:
    """Load JSON file with graceful fallback."""
    if not os.path.exists(path):
        if default is not None:
            return default
        raise FileNotFoundError(f"Required file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def load_predictions(path: str) -> Dict[str, Any]:
    """
    Load prediction pool from Step 6.
    
    Expected format:
    {
        "predictions": [
            {"value": 123, "confidence": 0.85, "rank": 1, "survivor_id": ...},
            ...
        ],
        "pool_size": 20,
        "generated_at": "..."
    }
    """
    data = load_json_safe(path, {"predictions": [], "pool_size": 0})
    
    # Handle alternate formats
    if "predictions" not in data and isinstance(data, list):
        data = {"predictions": data, "pool_size": len(data)}
    
    return data


def load_confidence_map(path: str) -> Dict[str, float]:
    """
    Load confidence scores from Step 6.
    
    Expected format: {"value_or_id": confidence_score, ...}
    """
    return load_json_safe(path, {})


def load_latest_draw(history_path: str) -> Dict[str, Any]:
    """
    Extract the most recent draw from lottery history.
    
    Returns:
        Dict with draw info including value and metadata
    """
    history = load_json_safe(history_path)
    
    draws = history.get("draws", [])
    if not draws:
        raise ValueError(f"No draws found in {history_path}")
    
    latest = draws[-1]
    
    # Normalize draw value
    draw_digits = latest.get("draw", latest.get("digits", latest.get("value", [])))
    if isinstance(draw_digits, int):
        # Convert single int to digit list
        draw_digits = [
            (draw_digits // 100) % 10,
            (draw_digits // 10) % 10,
            draw_digits % 10
        ]
    
    raw_value = latest.get("raw_value")
    if raw_value is None and draw_digits:
        raw_value = sum(d * (10 ** (len(draw_digits) - 1 - i)) 
                       for i, d in enumerate(draw_digits))
    
    return {
        "draw_id": latest.get("draw_id", f"draw_{len(draws)}"),
        "draw": draw_digits,
        "raw_value": raw_value,
        "timestamp": latest.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "draw_source": latest.get("draw_source", "external"),
        "draw_index": len(draws) - 1
    }


def load_survivors(path: str) -> List[Dict[str, Any]]:
    """Load survivors with scores from Step 3."""
    data = load_json_safe(path, {"survivors": []})
    
    if "survivors" in data:
        return data["survivors"]
    elif isinstance(data, list):
        return data
    else:
        return []


def load_model_meta(path: str) -> Dict[str, Any]:
    """Load model metadata from Step 5."""
    return load_json_safe(path, {
        "model_type": "unknown",
        "feature_hash": None,
        "trained_at": None
    })


def load_previous_diagnostics() -> Optional[Dict[str, Any]]:
    """Load previous diagnostics for comparison."""
    if os.path.exists(PREVIOUS_DIAGNOSTICS):
        return load_json_safe(PREVIOUS_DIAGNOSTICS)
    return None


# =============================================================================
# PREDICTION VALIDATION
# =============================================================================

def compute_prediction_validation(
    predictions: List[Dict[str, Any]],
    actual_draw: Dict[str, Any],
    near_threshold: int = 5
) -> Dict[str, Any]:
    """
    Compare predictions against actual draw outcome.
    
    Args:
        predictions: List of predicted values with confidence/rank
        actual_draw: Actual draw result
        near_threshold: Distance threshold for "near hit"
    
    Returns:
        Prediction validation metrics
    """
    actual_value = actual_draw.get("raw_value", 0)
    pool_size = len(predictions)
    
    if pool_size == 0:
        return {
            "pool_size": 0,
            "exact_hits": 0,
            "near_hits_within_threshold": 0,
            "near_threshold": near_threshold,
            "best_rank": None,
            "median_rank": None,
            "pool_coverage": 0.0,
            "hit_values": [],
            "near_hit_values": []
        }
    
    exact_hits = 0
    near_hits = 0
    hit_values = []
    near_hit_values = []
    hit_ranks = []
    
    for pred in predictions:
        pred_value = pred.get("value", pred.get("raw_value", pred.get("prediction", 0)))
        rank = pred.get("rank", predictions.index(pred) + 1)
        
        distance = abs(pred_value - actual_value)
        
        if distance == 0:
            exact_hits += 1
            hit_values.append(pred_value)
            hit_ranks.append(rank)
        elif distance <= near_threshold:
            near_hits += 1
            near_hit_values.append(pred_value)
            hit_ranks.append(rank)
    
    # Compute rank statistics
    best_rank = min(hit_ranks) if hit_ranks else None
    median_rank = median(hit_ranks) if hit_ranks else None
    
    # Pool coverage: fraction of prediction space covered
    unique_predictions = len(set(
        p.get("value", p.get("raw_value", p.get("prediction", 0))) 
        for p in predictions
    ))
    pool_coverage = unique_predictions / 1000  # Assuming mod 1000
    
    return {
        "pool_size": pool_size,
        "exact_hits": exact_hits,
        "near_hits_within_threshold": near_hits,
        "near_threshold": near_threshold,
        "best_rank": best_rank,
        "median_rank": median_rank,
        "pool_coverage": round(pool_coverage, 4),
        "hit_values": hit_values,
        "near_hit_values": near_hit_values,
        "actual_value": actual_value
    }


# =============================================================================
# CONFIDENCE CALIBRATION
# =============================================================================

def compute_confidence_calibration(
    predictions: List[Dict[str, Any]],
    actual_draw: Dict[str, Any],
    confidence_map: Dict[str, float]
) -> Dict[str, Any]:
    """
    Analyze confidence score accuracy.
    
    Checks if high-confidence predictions actually hit more often.
    
    Returns:
        Confidence calibration metrics
    """
    actual_value = actual_draw.get("raw_value", 0)
    
    confidences = []
    hit_confidences = []
    miss_confidences = []
    
    for pred in predictions:
        pred_value = pred.get("value", pred.get("raw_value", pred.get("prediction", 0)))
        
        # Get confidence from prediction or confidence map
        conf = pred.get("confidence", 0.0)
        if conf == 0.0 and confidence_map:
            conf = confidence_map.get(str(pred_value), 0.0)
        
        confidences.append(conf)
        
        if pred_value == actual_value:
            hit_confidences.append(conf)
        else:
            miss_confidences.append(conf)
    
    if not confidences:
        return {
            "mean_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0,
            "confidence_spread": 0.0,
            "predicted_vs_actual_correlation": 0.0,
            "overconfidence_detected": False,
            "underconfidence_detected": False,
            "hit_mean_confidence": None,
            "miss_mean_confidence": None
        }
    
    mean_conf = mean(confidences)
    max_conf = max(confidences)
    min_conf = min(confidences)
    spread = max_conf - min_conf
    
    # Compare hit vs miss confidence
    hit_mean = mean(hit_confidences) if hit_confidences else None
    miss_mean = mean(miss_confidences) if miss_confidences else None
    
    # Correlation: do higher confidence predictions hit more?
    # Simplified: compare hit confidence to overall mean
    if hit_mean is not None and miss_mean is not None:
        # Positive if hits have higher confidence than misses
        correlation = (hit_mean - miss_mean) / max(spread, 0.01)
        correlation = max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]
    else:
        correlation = 0.0
    
    # Detect calibration issues
    overconfidence = max_conf > 0.8 and len(hit_confidences) == 0
    underconfidence = max_conf < 0.5 and len(hit_confidences) > 0
    
    return {
        "mean_confidence": round(mean_conf, 4),
        "max_confidence": round(max_conf, 4),
        "min_confidence": round(min_conf, 4),
        "confidence_spread": round(spread, 4),
        "predicted_vs_actual_correlation": round(correlation, 4),
        "overconfidence_detected": overconfidence,
        "underconfidence_detected": underconfidence,
        "hit_mean_confidence": round(hit_mean, 4) if hit_mean is not None else None,
        "miss_mean_confidence": round(miss_mean, 4) if miss_mean is not None else None
    }


# =============================================================================
# SURVIVOR PERFORMANCE
# =============================================================================

def compute_survivor_performance(
    survivors: List[Dict[str, Any]],
    actual_draw: Dict[str, Any],
    previous_diagnostics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Track which survivors predicted correctly.
    
    Identifies:
    - Survivors that hit this draw
    - Candidates for reinforcement (multiple recent hits)
    - Candidates for decay (consistently wrong)
    
    Returns:
        Survivor performance metrics
    """
    actual_value = actual_draw.get("raw_value", 0)
    
    hit_survivors = []
    top_10_hits = 0
    
    # Track survivor predictions
    for i, survivor in enumerate(survivors[:100]):  # Check top 100
        # Get survivor's predicted value (if available)
        pred_value = survivor.get("predicted_value", survivor.get("value"))
        survivor_id = survivor.get("id", survivor.get("seed", i))
        
        if pred_value is not None and pred_value == actual_value:
            hit_survivors.append(survivor_id)
            if i < 10:
                top_10_hits += 1
    
    # Identify decay candidates (survivors with low recent performance)
    decay_candidates = []
    reinforce_candidates = []
    
    # Get previous hit survivors for comparison
    prev_hits = set()
    if previous_diagnostics:
        prev_perf = previous_diagnostics.get("survivor_performance", {})
        prev_hits = set(prev_perf.get("hit_survivors", []))
    
    # Survivors that hit now but didn't before = reinforce
    for sid in hit_survivors:
        if sid not in prev_hits:
            reinforce_candidates.append(sid)
    
    # For decay: would need historical tracking (simplified here)
    # In production, track consecutive misses per survivor
    
    top_10_rate = top_10_hits / 10 if len(survivors) >= 10 else 0.0
    
    return {
        "hit_survivors": hit_survivors,
        "hit_count": len(hit_survivors),
        "top_10_hit_rate": round(top_10_rate, 4),
        "decay_candidates": decay_candidates,
        "reinforce_candidates": reinforce_candidates,
        "total_survivors_checked": min(len(survivors), 100)
    }


# =============================================================================
# FEATURE DIAGNOSTICS
# =============================================================================

def compute_feature_diagnostics(
    model_meta: Dict[str, Any],
    previous_diagnostics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Detect feature importance drift and schema changes.
    
    Returns:
        Feature diagnostic metrics
    """
    current_hash = model_meta.get("feature_hash", model_meta.get("schema_hash"))
    current_importance = model_meta.get("feature_importance", {})
    
    # Default values
    dominant_shift = False
    entropy_change = 0.0
    turnover = 0.0
    schema_match = True
    
    if previous_diagnostics:
        prev_feature = previous_diagnostics.get("feature_diagnostics", {})
        prev_hash = prev_feature.get("current_hash")
        prev_importance = prev_feature.get("top_features", {})
        
        # Check schema hash
        if prev_hash and current_hash:
            schema_match = prev_hash == current_hash
        
        # Check feature importance turnover
        if current_importance and prev_importance:
            # Get top 10 features from each
            curr_top = set(list(current_importance.keys())[:10])
            prev_top = set(list(prev_importance.keys())[:10])
            
            if curr_top and prev_top:
                changed = len(curr_top - prev_top)
                turnover = changed / 10.0
            
            # Check for dominant feature shift
            curr_top_val = max(current_importance.values()) if current_importance else 0
            prev_top_val = max(prev_importance.values()) if prev_importance else 0
            
            if prev_top_val > 0:
                change_ratio = abs(curr_top_val - prev_top_val) / prev_top_val
                dominant_shift = change_ratio > 0.3
    
    # Compute entropy of current feature importance
    if current_importance:
        values = list(current_importance.values())
        total = sum(values)
        if total > 0:
            normalized = [v / total for v in values]
            # Shannon entropy (simplified)
            entropy = -sum(p * (p if p > 0 else 1e-10) for p in normalized)
        else:
            entropy = 0.0
    else:
        entropy = 0.0
    
    # Entropy change from previous
    if previous_diagnostics:
        prev_entropy = previous_diagnostics.get("feature_diagnostics", {}).get("entropy", 0.0)
        entropy_change = entropy - prev_entropy
    
    return {
        "dominant_feature_shift": dominant_shift,
        "entropy": round(entropy, 4),
        "entropy_change": round(entropy_change, 4),
        "top_feature_turnover": round(turnover, 4),
        "schema_hash_match": schema_match,
        "current_hash": current_hash,
        "top_features": dict(list(current_importance.items())[:5]) if current_importance else {}
    }


# =============================================================================
# PIPELINE HEALTH
# =============================================================================

def compute_pipeline_health(
    prediction_validation: Dict[str, Any],
    survivors: List[Dict[str, Any]],
    previous_diagnostics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Assess overall pipeline health metrics.
    
    Returns:
        Pipeline health indicators
    """
    # Window decay: how well are predictions performing over time?
    exact_hits = prediction_validation.get("exact_hits", 0)
    pool_size = prediction_validation.get("pool_size", 1)
    current_hit_rate = exact_hits / max(pool_size, 1)
    
    # Default values
    window_decay = 0.0
    survivor_churn = 0.0
    consecutive_misses = 0
    stability = "stable"
    
    if previous_diagnostics:
        prev_health = previous_diagnostics.get("pipeline_health", {})
        
        # Track consecutive misses
        prev_misses = prev_health.get("consecutive_misses", 0)
        if exact_hits == 0:
            consecutive_misses = prev_misses + 1
        else:
            consecutive_misses = 0
        
        # Compute window decay (rolling hit rate decline)
        prev_hit_rate = prev_health.get("current_hit_rate", current_hit_rate)
        if prev_hit_rate > 0:
            window_decay = (prev_hit_rate - current_hit_rate) / prev_hit_rate
            window_decay = max(0.0, window_decay)  # Only track decay, not improvement
        
        # Survivor churn: how many survivors changed?
        prev_survivor_count = prev_health.get("survivor_count", len(survivors))
        # Simplified: would need survivor ID tracking for proper churn
        if prev_survivor_count > 0:
            survivor_churn = abs(len(survivors) - prev_survivor_count) / prev_survivor_count
    
    # Determine stability status
    if consecutive_misses >= 5:
        stability = "degraded"
    elif consecutive_misses >= 3 or window_decay > 0.3:
        stability = "warning"
    elif survivor_churn > 0.3:
        stability = "unstable"
    else:
        stability = "stable"
    
    return {
        "current_hit_rate": round(current_hit_rate, 4),
        "window_decay": round(window_decay, 4),
        "survivor_churn": round(survivor_churn, 4),
        "survivor_count": len(survivors),
        "model_stability": stability,
        "consecutive_misses": consecutive_misses
    }


# =============================================================================
# SUMMARY FLAGS & RECOMMENDATIONS
# =============================================================================

def generate_summary_flags(
    prediction_validation: Dict[str, Any],
    confidence_calibration: Dict[str, Any],
    survivor_performance: Dict[str, Any],
    feature_diagnostics: Dict[str, Any],
    pipeline_health: Dict[str, Any],
    policies: Dict[str, Any]
) -> List[str]:
    """
    Generate summary flags based on diagnostic results.
    
    Returns:
        List of flag strings indicating issues/status
    """
    flags = []
    
    # Get thresholds from policies
    triggers = policies.get("retrain_triggers", {})
    regime_triggers = policies.get("regime_shift_triggers", {})
    
    # Prediction quality flags
    if prediction_validation.get("exact_hits", 0) == 0:
        flags.append("NO_EXACT_HITS")
    
    if prediction_validation.get("pool_coverage", 0) < 0.01:
        flags.append("LOW_POOL_COVERAGE")
    
    hit_rate = prediction_validation.get("exact_hits", 0) / max(prediction_validation.get("pool_size", 1), 1)
    if hit_rate < triggers.get("hit_rate_collapse_threshold", 0.01):
        flags.append("WEAK_SIGNAL")
    
    # Confidence calibration flags
    if confidence_calibration.get("overconfidence_detected", False):
        flags.append("OVERCONFIDENT_MODEL")
    
    if confidence_calibration.get("underconfidence_detected", False):
        flags.append("UNDERCONFIDENT_MODEL")
    
    correlation = confidence_calibration.get("predicted_vs_actual_correlation", 0)
    if correlation < triggers.get("confidence_drift_threshold", 0.2):
        flags.append("CONFIDENCE_DRIFT")
    
    # Survivor flags
    if survivor_performance.get("hit_count", 0) == 0:
        flags.append("NO_SURVIVOR_HITS")
    
    if len(survivor_performance.get("reinforce_candidates", [])) > 0:
        flags.append("NEW_STRONG_SURVIVORS")
    
    # Feature flags
    if feature_diagnostics.get("dominant_feature_shift", False):
        flags.append("FEATURE_DRIFT")
    
    if not feature_diagnostics.get("schema_hash_match", True):
        flags.append("SCHEMA_MISMATCH")
    
    if feature_diagnostics.get("top_feature_turnover", 0) > 0.3:
        flags.append("HIGH_FEATURE_TURNOVER")
    
    # Pipeline health flags
    consecutive_misses = pipeline_health.get("consecutive_misses", 0)
    if consecutive_misses >= triggers.get("max_consecutive_misses", 5):
        flags.append("CONSECUTIVE_MISS_LIMIT")
    
    if pipeline_health.get("model_stability") == "degraded":
        flags.append("MODEL_DEGRADED")
    elif pipeline_health.get("model_stability") == "warning":
        flags.append("MODEL_WARNING")
    
    if pipeline_health.get("window_decay", 0) > regime_triggers.get("window_decay_threshold", 0.5):
        flags.append("HIGH_WINDOW_DECAY")
    
    if pipeline_health.get("survivor_churn", 0) > regime_triggers.get("survivor_churn_threshold", 0.4):
        flags.append("HIGH_SURVIVOR_CHURN")
    
    # Summary recommendation flags
    if "CONSECUTIVE_MISS_LIMIT" in flags or "CONFIDENCE_DRIFT" in flags:
        flags.append("RETRAIN_RECOMMENDED")
    
    if "HIGH_WINDOW_DECAY" in flags and "HIGH_SURVIVOR_CHURN" in flags:
        flags.append("REGIME_SHIFT_POSSIBLE")
    
    return sorted(set(flags))


def generate_recommended_actions(
    summary_flags: List[str],
    pipeline_health: Dict[str, Any],
    policies: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate recommended actions based on flags.
    
    Returns:
        Dict of recommended actions and parameter adjustments
    """
    actions = {
        "retrain_step_5": False,
        "rerun_step_3": False,
        "rerun_step_1": False,
        "human_review_required": False,
        "parameter_adjustments": {}
    }
    
    # Determine required actions
    if "RETRAIN_RECOMMENDED" in summary_flags:
        actions["retrain_step_5"] = True
        actions["rerun_step_3"] = True
    
    if "REGIME_SHIFT_POSSIBLE" in summary_flags:
        actions["rerun_step_1"] = True
        actions["human_review_required"] = True
    
    if "SCHEMA_MISMATCH" in summary_flags:
        actions["human_review_required"] = True
    
    if "MODEL_DEGRADED" in summary_flags:
        actions["retrain_step_5"] = True
        actions["rerun_step_3"] = True
    
    # Suggest parameter adjustments
    if "OVERCONFIDENT_MODEL" in summary_flags:
        actions["parameter_adjustments"]["confidence_threshold"] = "-0.1"
    
    if "UNDERCONFIDENT_MODEL" in summary_flags:
        actions["parameter_adjustments"]["confidence_threshold"] = "+0.1"
    
    if "WEAK_SIGNAL" in summary_flags:
        actions["parameter_adjustments"]["pool_size"] = "+10"
    
    if "LOW_POOL_COVERAGE" in summary_flags:
        actions["parameter_adjustments"]["pool_size"] = "+5"
    
    # Check v1 approval requirements
    v1_approval = policies.get("v1_approval_required", {})
    if v1_approval.get("retrain_execution", True) and actions["retrain_step_5"]:
        actions["human_review_required"] = True
    
    if v1_approval.get("regime_reset", True) and actions["rerun_step_1"]:
        actions["human_review_required"] = True
    
    return actions


# =============================================================================
# FINGERPRINT
# =============================================================================

def compute_data_fingerprint(
    actual_draw: Dict[str, Any],
    predictions: List[Dict[str, Any]]
) -> str:
    """Compute fingerprint for this diagnostic run."""
    data = {
        "draw": actual_draw.get("raw_value"),
        "draw_id": actual_draw.get("draw_id"),
        "prediction_count": len(predictions)
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]


# =============================================================================
# MAIN DIAGNOSTIC GENERATOR
# =============================================================================

def generate_diagnostics(
    prediction_pool_path: str = DEFAULT_PREDICTION_POOL,
    confidence_map_path: str = DEFAULT_CONFIDENCE_MAP,
    lottery_history_path: str = DEFAULT_LOTTERY_HISTORY,
    model_meta_path: str = DEFAULT_MODEL_META,
    survivors_path: str = DEFAULT_SURVIVORS,
    policies_path: str = DEFAULT_POLICIES
) -> Dict[str, Any]:
    """
    Generate complete post-draw diagnostics.
    
    Returns:
        Full diagnostics structure per Chapter 13 schema
    """
    timestamp = datetime.now(timezone.utc)
    run_id = f"chapter13_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'='*60}")
    print(f"CHAPTER 13 DIAGNOSTICS — {run_id}")
    print(f"{'='*60}")
    
    # Load all inputs
    print("\n📂 Loading inputs...")
    
    predictions_data = load_predictions(prediction_pool_path)
    predictions = predictions_data.get("predictions", [])
    print(f"   ✅ Predictions: {len(predictions)} loaded")
    
    confidence_map = load_confidence_map(confidence_map_path)
    print(f"   ✅ Confidence map: {len(confidence_map)} entries")
    
    actual_draw = load_latest_draw(lottery_history_path)
    print(f"   ✅ Actual draw: {actual_draw['draw']} (ID: {actual_draw['draw_id']})")
    
    model_meta = load_model_meta(model_meta_path)
    print(f"   ✅ Model meta: {model_meta.get('model_type', 'unknown')}")
    
    survivors = load_survivors(survivors_path)
    print(f"   ✅ Survivors: {len(survivors)} loaded")
    
    policies = load_json_safe(policies_path, {})
    print(f"   ✅ Policies: loaded")
    
    previous = load_previous_diagnostics()
    if previous:
        print(f"   ✅ Previous diagnostics: {previous.get('run_id', 'unknown')}")
    else:
        print(f"   ⚠️  No previous diagnostics (first run)")
    
    # Compute all metrics
    print("\n📊 Computing metrics...")
    
    prediction_validation = compute_prediction_validation(predictions, actual_draw)
    print(f"   ✅ Prediction validation: {prediction_validation['exact_hits']} hits")
    
    confidence_calibration = compute_confidence_calibration(
        predictions, actual_draw, confidence_map
    )
    print(f"   ✅ Confidence calibration: mean={confidence_calibration['mean_confidence']}")
    
    survivor_performance = compute_survivor_performance(
        survivors, actual_draw, previous
    )
    print(f"   ✅ Survivor performance: {survivor_performance['hit_count']} hits")
    
    feature_diagnostics = compute_feature_diagnostics(model_meta, previous)
    print(f"   ✅ Feature diagnostics: turnover={feature_diagnostics['top_feature_turnover']}")
    
    pipeline_health = compute_pipeline_health(
        prediction_validation, survivors, previous
    )
    print(f"   ✅ Pipeline health: {pipeline_health['model_stability']}")
    
    # Generate flags and recommendations
    print("\n🚩 Generating flags...")
    
    summary_flags = generate_summary_flags(
        prediction_validation,
        confidence_calibration,
        survivor_performance,
        feature_diagnostics,
        pipeline_health,
        policies
    )
    print(f"   Flags: {summary_flags if summary_flags else '(none)'}")
    
    recommended_actions = generate_recommended_actions(
        summary_flags, pipeline_health, policies
    )
    print(f"   Actions: retrain={recommended_actions['retrain_step_5']}, "
          f"step3={recommended_actions['rerun_step_3']}, "
          f"step1={recommended_actions['rerun_step_1']}")
    
    # Build final diagnostics
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "draw_id": actual_draw["draw_id"],
        "draw_timestamp": actual_draw["timestamp"],
        "draw_source": actual_draw.get("draw_source", "external"),
        "data_fingerprint": compute_data_fingerprint(actual_draw, predictions),
        "generated_at": timestamp.isoformat(),
        
        "prediction_validation": prediction_validation,
        "confidence_calibration": confidence_calibration,
        "survivor_performance": survivor_performance,
        "feature_diagnostics": feature_diagnostics,
        "pipeline_health": pipeline_health,
        
        "summary_flags": summary_flags,
        "recommended_actions": recommended_actions
    }
    
    return diagnostics


# =============================================================================
# OUTPUT & ARCHIVAL
# =============================================================================

def save_diagnostics(
    diagnostics: Dict[str, Any],
    output_path: str = DEFAULT_OUTPUT,
    history_dir: str = DEFAULT_HISTORY_DIR,
    archive: bool = True
) -> Tuple[str, Optional[str]]:
    """
    Save diagnostics to file and optionally archive.
    
    Returns:
        Tuple of (output_path, archive_path or None)
    """
    # Save current diagnostics
    with open(output_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n💾 Saved: {output_path}")
    
    # Save as previous for next comparison
    with open(PREVIOUS_DIAGNOSTICS, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    archive_path = None
    if archive:
        # Create history directory
        Path(history_dir).mkdir(exist_ok=True)
        
        # Archive with timestamp
        timestamp = diagnostics.get("generated_at", datetime.now(timezone.utc).isoformat())
        ts_clean = timestamp.replace(":", "").replace("-", "").replace("T", "_")[:15]
        archive_name = f"{ts_clean}_diagnostics.json"
        archive_path = os.path.join(history_dir, archive_name)
        
        shutil.copy(output_path, archive_path)
        print(f"📦 Archived: {archive_path}")
    
    return output_path, archive_path


def show_status(
    output_path: str = DEFAULT_OUTPUT,
    history_dir: str = DEFAULT_HISTORY_DIR
) -> None:
    """Show current diagnostics status."""
    print(f"\n{'='*60}")
    print("CHAPTER 13 DIAGNOSTICS — Status")
    print(f"{'='*60}")
    
    # Current diagnostics
    if os.path.exists(output_path):
        diag = load_json_safe(output_path)
        print(f"\n📄 Current Diagnostics ({output_path}):")
        print(f"   Run ID: {diag.get('run_id', 'N/A')}")
        print(f"   Draw ID: {diag.get('draw_id', 'N/A')}")
        print(f"   Generated: {diag.get('generated_at', 'N/A')}")
        
        pv = diag.get("prediction_validation", {})
        print(f"   Exact Hits: {pv.get('exact_hits', 'N/A')}/{pv.get('pool_size', 'N/A')}")
        
        ph = diag.get("pipeline_health", {})
        print(f"   Stability: {ph.get('model_stability', 'N/A')}")
        print(f"   Consecutive Misses: {ph.get('consecutive_misses', 'N/A')}")
        
        flags = diag.get("summary_flags", [])
        print(f"   Flags: {flags if flags else '(none)'}")
    else:
        print(f"\n📄 No current diagnostics ({output_path} not found)")
    
    # History
    if os.path.exists(history_dir):
        files = sorted(Path(history_dir).glob("*_diagnostics.json"))
        print(f"\n📦 Diagnostics History ({history_dir}):")
        print(f"   Total archived: {len(files)}")
        if files:
            print(f"   Latest: {files[-1].name}")
            print(f"   Oldest: {files[0].name}")
    else:
        print(f"\n📦 No history directory ({history_dir})")
    
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chapter 13 Diagnostics Engine — Compare predictions to reality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --generate         Generate diagnostics for latest draw (default)
  --status           Show current diagnostics status
  --dry-run          Generate but don't save diagnostics

Inputs (all optional, use defaults if not specified):
  --predictions      Prediction pool from Step 6 (default: prediction_pool.json)
  --confidence       Confidence map from Step 6 (default: confidence_map.json)
  --history          Lottery history with latest draw (default: lottery_history.json)
  --model-meta       Model metadata from Step 5 (default: best_model.meta.json)
  --survivors        Survivors from Step 3 (default: survivors_with_scores.json)
  --policies         WATCHER policies (default: watcher_policies.json)

Output:
  --output           Output file (default: post_draw_diagnostics.json)
  --history-dir      Archive directory (default: diagnostics_history/)
  --no-archive       Don't archive to history directory

Examples:
  python3 chapter_13_diagnostics.py --generate
  python3 chapter_13_diagnostics.py --status
  python3 chapter_13_diagnostics.py --dry-run
  python3 chapter_13_diagnostics.py --generate --output custom_diagnostics.json
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--generate", action="store_true", default=True,
                           help="Generate diagnostics (default)")
    mode_group.add_argument("--status", action="store_true",
                           help="Show current status")
    mode_group.add_argument("--dry-run", action="store_true",
                           help="Generate but don't save")
    
    # Input files
    parser.add_argument("--predictions", type=str, default=DEFAULT_PREDICTION_POOL,
                       help=f"Prediction pool file (default: {DEFAULT_PREDICTION_POOL})")
    parser.add_argument("--confidence", type=str, default=DEFAULT_CONFIDENCE_MAP,
                       help=f"Confidence map file (default: {DEFAULT_CONFIDENCE_MAP})")
    parser.add_argument("--history", type=str, default=DEFAULT_LOTTERY_HISTORY,
                       help=f"Lottery history file (default: {DEFAULT_LOTTERY_HISTORY})")
    parser.add_argument("--model-meta", type=str, default=DEFAULT_MODEL_META,
                       help=f"Model metadata file (default: {DEFAULT_MODEL_META})")
    parser.add_argument("--survivors", type=str, default=DEFAULT_SURVIVORS,
                       help=f"Survivors file (default: {DEFAULT_SURVIVORS})")
    parser.add_argument("--policies", type=str, default=DEFAULT_POLICIES,
                       help=f"Policies file (default: {DEFAULT_POLICIES})")
    
    # Output options
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                       help=f"Output file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--history-dir", type=str, default=DEFAULT_HISTORY_DIR,
                       help=f"Archive directory (default: {DEFAULT_HISTORY_DIR})")
    parser.add_argument("--no-archive", action="store_true",
                       help="Don't archive to history directory")
    
    args = parser.parse_args()
    
    try:
        if args.status:
            show_status(args.output, args.history_dir)
            return 0
        
        # Generate diagnostics
        diagnostics = generate_diagnostics(
            prediction_pool_path=args.predictions,
            confidence_map_path=args.confidence,
            lottery_history_path=args.history,
            model_meta_path=args.model_meta,
            survivors_path=args.survivors,
            policies_path=args.policies
        )
        
        if args.dry_run:
            print(f"\n{'='*60}")
            print("DRY RUN — Diagnostics not saved")
            print(f"{'='*60}")
            print(json.dumps(diagnostics, indent=2))
            return 0
        
        # Save and archive
        save_diagnostics(
            diagnostics,
            output_path=args.output,
            history_dir=args.history_dir,
            archive=not args.no_archive
        )
        
        print(f"\n✅ Diagnostics complete")
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ Value error: {e}")
        return 2
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 99


if __name__ == "__main__":
    sys.exit(main())
