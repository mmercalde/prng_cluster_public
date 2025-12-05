#!/usr/bin/env python3
"""
Prediction Context - Step 6 specialized agent context.

Provides domain expertise for evaluating prediction pool generation
and final output quality.

Version: 3.2.0
"""

from typing import Dict, Any, List
from agents.contexts.base_agent_context import BaseAgentContext


class PredictionContext(BaseAgentContext):
    """
    Specialized context for Step 6: Prediction Generator.
    
    Key focus areas:
    - Prediction pool size and diversity
    - Confidence distribution
    - Coverage analysis
    - Output quality metrics
    """
    
    agent_name: str = "prediction_agent"
    pipeline_step: int = 6
    
    def get_key_metrics(self) -> List[str]:
        """Key metrics for prediction generation."""
        return [
            "pool_size",
            "mean_confidence",
            "confidence_std",
            "min_confidence",
            "max_confidence",
            "diversity_score",
            "coverage_pct",
            "unique_predictions",
            "execution_time_seconds"
        ]
    
    def get_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Evaluation thresholds for prediction generator."""
        return {
            "pool_size": {
                "excellent": {"min": 100, "max": 300},
                "good": {"min": 50, "max": 100},
                "acceptable": {"min": 300, "max": 500},
                "poor": {"min": 20, "max": 50},
                "fail": {"min": 0, "max": 20}
            },
            "mean_confidence": {
                "excellent": {"min": 0.80, "max": 1.0},
                "good": {"min": 0.65, "max": 0.80},
                "acceptable": {"min": 0.50, "max": 0.65},
                "poor": {"min": 0.35, "max": 0.50},
                "fail": {"min": 0.0, "max": 0.35}
            },
            "diversity_score": {
                "excellent": {"min": 0.80, "max": 1.0},
                "good": {"min": 0.60, "max": 0.80},
                "acceptable": {"min": 0.40, "max": 0.60},
                "poor": {"min": 0.20, "max": 0.40},
                "fail": {"min": 0.0, "max": 0.20}
            },
            "coverage_pct": {
                "excellent": {"min": 80, "max": 100},
                "good": {"min": 60, "max": 80},
                "acceptable": {"min": 40, "max": 60},
                "poor": {"min": 20, "max": 40},
                "fail": {"min": 0, "max": 20}
            }
        }
    
    def interpret_results(self) -> str:
        """Interpret prediction generation results."""
        pool_size = self.results.get("pool_size", 0)
        mean_conf = self.results.get("mean_confidence", 0)
        conf_std = self.results.get("confidence_std", 0)
        min_conf = self.results.get("min_confidence", 0)
        max_conf = self.results.get("max_confidence", 0)
        diversity = self.results.get("diversity_score", 0)
        coverage = self.results.get("coverage_pct", 0)
        unique = self.results.get("unique_predictions", 0)
        
        interpretation_parts = []
        
        # Pool size analysis
        if pool_size >= 100 and pool_size <= 300:
            interpretation_parts.append(f"Optimal pool size ({pool_size} predictions) - good balance of quantity and quality.")
        elif pool_size < 50:
            interpretation_parts.append(f"Small pool ({pool_size} predictions) - may miss opportunities.")
        elif pool_size > 500:
            interpretation_parts.append(f"Large pool ({pool_size} predictions) - consider filtering for quality.")
        else:
            interpretation_parts.append(f"Pool contains {pool_size} predictions.")
        
        # Confidence analysis
        if mean_conf >= 0.80:
            interpretation_parts.append(f"High confidence predictions (mean={mean_conf:.3f}).")
        elif mean_conf >= 0.65:
            interpretation_parts.append(f"Good confidence level (mean={mean_conf:.3f}).")
        elif mean_conf >= 0.50:
            interpretation_parts.append(f"Moderate confidence (mean={mean_conf:.3f}) - use with caution.")
        else:
            interpretation_parts.append(f"Low confidence predictions (mean={mean_conf:.3f}) - model may need improvement.")
        
        # Confidence distribution
        conf_range = max_conf - min_conf
        if conf_std > 0.15:
            interpretation_parts.append(f"Wide confidence range ({min_conf:.2f}-{max_conf:.2f}) - mixed quality.")
        else:
            interpretation_parts.append(f"Consistent confidence ({min_conf:.2f}-{max_conf:.2f}).")
        
        # Diversity analysis
        if diversity >= 0.80:
            interpretation_parts.append(f"Excellent diversity (score={diversity:.3f}) - predictions cover number space well.")
        elif diversity >= 0.60:
            interpretation_parts.append(f"Good diversity (score={diversity:.3f}).")
        elif diversity >= 0.40:
            interpretation_parts.append(f"Moderate diversity (score={diversity:.3f}) - some clustering.")
        else:
            interpretation_parts.append(f"Low diversity (score={diversity:.3f}) - predictions may be too similar.")
        
        # Coverage
        if coverage > 0:
            interpretation_parts.append(f"Number space coverage: {coverage:.1f}%.")
        
        # Unique count
        if unique > 0 and unique != pool_size:
            interpretation_parts.append(f"Unique predictions: {unique}/{pool_size}.")
        
        return " ".join(interpretation_parts)
    
    def get_retry_suggestions(self) -> List[Dict[str, Any]]:
        """Suggest parameter adjustments for retry."""
        suggestions = []
        pool_size = self.results.get("pool_size", 0)
        mean_conf = self.results.get("mean_confidence", 0)
        diversity = self.results.get("diversity_score", 0)
        conf_threshold = self.results.get("config", {}).get("confidence_threshold", 0.5)
        current_pool_size = self.results.get("config", {}).get("pool_size", 200)
        diversity_weight = self.results.get("config", {}).get("diversity_weight", 0.5)
        
        # Too few predictions
        if pool_size < 50:
            suggestions.append({
                "param": "confidence_threshold",
                "suggestion": max(conf_threshold - 0.1, 0.3),
                "reason": f"Small pool ({pool_size}) - lower threshold to include more predictions"
            })
            suggestions.append({
                "param": "pool_size",
                "suggestion": min(current_pool_size * 2, 500),
                "reason": "Increase target pool size"
            })
        
        # Low confidence
        if mean_conf < 0.50:
            suggestions.append({
                "param": "confidence_threshold",
                "suggestion": min(conf_threshold + 0.1, 0.7),
                "reason": f"Low mean confidence ({mean_conf:.3f}) - raise threshold for quality"
            })
        
        # Low diversity
        if diversity < 0.40:
            suggestions.append({
                "param": "diversity_weight",
                "suggestion": min(diversity_weight + 0.2, 1.0),
                "reason": f"Low diversity ({diversity:.3f}) - increase diversity weight"
            })
        
        # Too many predictions (quality concern)
        if pool_size > 500:
            suggestions.append({
                "param": "confidence_threshold",
                "suggestion": min(conf_threshold + 0.15, 0.8),
                "reason": f"Large pool ({pool_size}) - raise threshold for quality"
            })
            suggestions.append({
                "param": "pool_size",
                "suggestion": 300,
                "reason": "Reduce target pool size for focus"
            })
        
        return suggestions


def create_prediction_context(
    results: Dict[str, Any],
    run_number: int = 1,
    manifest_path: str = None
) -> PredictionContext:
    """Factory function to create prediction context."""
    ctx = PredictionContext(
        run_number=run_number,
        results=results
    )
    
    if manifest_path:
        ctx.load_manifest(manifest_path)
    
    return ctx
