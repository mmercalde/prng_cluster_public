"""
Metrics Extractor v1.0
Computes derived metrics from raw counts - NO semantic interpretation.

Team Beta Approved: 2026-01-04
"""
from typing import Dict, Any, Optional
import json
import os


def load_evaluation_thresholds(
    config_path: str = "distributed_config.json"
) -> Dict[str, Any]:
    """Load evaluation thresholds from config."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get("evaluation_thresholds", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_step_thresholds(
    step_id: str,
    data_source_type: str,
    config_path: str = "distributed_config.json"
) -> Dict[str, Any]:
    """
    Get thresholds for a specific step and data source type.
    
    Args:
        step_id: e.g., "step_1_window_optimizer"
        data_source_type: "synthetic", "real", or "hybrid"
        config_path: Path to distributed_config.json
    
    Returns:
        Threshold priors for this step/source combination
    """
    thresholds = load_evaluation_thresholds(config_path)
    step_thresholds = thresholds.get(step_id, {})
    
    # Get source-specific thresholds, fall back to "real" (strictest)
    return step_thresholds.get(
        data_source_type, 
        step_thresholds.get("real", {})
    )


def extract_step1_derived_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute derived metrics for Step 1 (Window Optimizer).
    
    Raw inputs:
        seeds_tested, forward_count, reverse_count, bidirectional_count
    
    Derived outputs:
        forward_rate, reverse_rate, bidirectional_rate, overlap_ratio
    """
    seeds = raw.get("seeds_tested", raw.get("seed_count", 1))
    forward = raw.get("forward_count", 0)
    reverse = raw.get("reverse_count", 0)
    bidirectional = raw.get("bidirectional_count", 0)
    
    # Avoid division by zero
    seeds = max(seeds, 1)
    min_directional = max(min(forward, reverse), 1)
    
    return {
        "forward_rate": forward / seeds,
        "reverse_rate": reverse / seeds,
        "bidirectional_rate": bidirectional / seeds,
        "overlap_ratio": bidirectional / min_directional,
        "forward_reverse_ratio": forward / max(reverse, 1),
    }


def extract_step2_derived_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute derived metrics for Step 2 (Scorer Meta-Optimizer).
    """
    trials = raw.get("trials", raw.get("n_trials", 1))
    best_score = raw.get("best_score", 0)
    convergence_trial = raw.get("convergence_trial", trials)
    initial_score = raw.get("initial_score", 0)
    
    return {
        "score_improvement": best_score - initial_score,
        "convergence_rate": convergence_trial / max(trials, 1),
        "trials_efficiency": best_score / max(trials, 1),
    }


def extract_step3_derived_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute derived metrics for Step 3 (Full Scoring).
    """
    total = raw.get("survivors_total", raw.get("total_survivors", 1))
    scored = raw.get("survivors_scored", 0)
    features = raw.get("features_extracted", 0)
    failed = raw.get("failed_chunks", 0)
    total_chunks = raw.get("total_chunks", 1)
    
    return {
        "completion_rate": scored / max(total, 1),
        "feature_coverage": features,  # Absolute, but compared to expected 62
        "chunk_success_rate": (total_chunks - failed) / max(total_chunks, 1),
    }


def extract_step4_derived_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute derived metrics for Step 4 (ML Meta-Optimizer).
    """
    score_min = raw.get("score_min", 0)
    score_max = raw.get("score_max", 0)
    feature_variances = raw.get("feature_variances", [0.1])
    
    avg_variance = sum(feature_variances) / max(len(feature_variances), 1)
    
    return {
        "score_range": score_max - score_min,
        "avg_feature_variance": avg_variance,
        "score_midpoint": (score_min + score_max) / 2,
    }


def extract_step5_derived_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute derived metrics for Step 5 (Anti-Overfit Training).
    """
    train_loss = raw.get("train_loss", 0)
    val_loss = raw.get("val_loss", 0)
    holdout_hits = raw.get("holdout_hits", 0)
    holdout_total = raw.get("holdout_total", 1)
    
    return {
        "val_train_gap": abs(val_loss - train_loss),
        "holdout_accuracy": holdout_hits / max(holdout_total, 1),
        "overfitting_risk": (val_loss - train_loss) / max(train_loss, 0.001),
    }


def extract_step6_derived_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute derived metrics for Step 6 (Prediction Generator).
    """
    import math
    
    predictions = raw.get("predictions", [])
    confidence_scores = raw.get("confidence_scores", [0.5])
    
    n = len(predictions) if predictions else 1
    conf_mean = sum(confidence_scores) / max(len(confidence_scores), 1)
    
    # Entropy calculation (simplified)
    if confidence_scores:
        # Treat as probability distribution
        entropy = -sum(
            p * math.log(p + 1e-10) for p in confidence_scores if p > 0
        ) / max(len(confidence_scores), 1)
    else:
        entropy = 0
    
    return {
        "confidence_mean": conf_mean,
        "prediction_entropy": entropy,
        "prediction_count": n,
    }


# Step extractor registry
STEP_EXTRACTORS = {
    1: extract_step1_derived_metrics,
    2: extract_step2_derived_metrics,
    3: extract_step3_derived_metrics,
    4: extract_step4_derived_metrics,
    5: extract_step5_derived_metrics,
    6: extract_step6_derived_metrics,
}

STEP_IDS = {
    1: "step_1_window_optimizer",
    2: "step_2_scorer_meta",
    3: "step_3_full_scoring",
    4: "step_4_ml_meta",
    5: "step_5_anti_overfit",
    6: "step_6_prediction",
}


def extract_derived_metrics(step: int, raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract derived metrics for any step.
    
    Args:
        step: Pipeline step number (1-6)
        raw: Raw metrics dict
    
    Returns:
        Derived metrics dict (rates, ratios - no interpretation)
    """
    extractor = STEP_EXTRACTORS.get(step)
    if extractor:
        return extractor(raw)
    return {}
