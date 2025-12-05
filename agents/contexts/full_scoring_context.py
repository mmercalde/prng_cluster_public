#!/usr/bin/env python3
"""
Full Scoring Context - Step 3 specialized agent context.

Provides domain expertise for evaluating full survivor scoring,
feature extraction, and scoring completion across distributed GPUs.

Version: 3.2.0
"""

from typing import Dict, Any, List
from agents.contexts.base_agent_context import BaseAgentContext


class FullScoringContext(BaseAgentContext):
    """
    Specialized context for Step 3: Full Scoring.
    
    Key focus areas:
    - Scoring completion rate
    - Feature vector quality
    - Score distribution analysis
    - Processing efficiency
    """
    
    agent_name: str = "full_scoring_agent"
    pipeline_step: int = 3
    
    def get_key_metrics(self) -> List[str]:
        """Key metrics for full scoring."""
        return [
            "completion_rate",
            "survivors_scored",
            "survivors_total",
            "feature_dimensions",
            "mean_score",
            "score_std",
            "top_candidates",
            "execution_time_seconds"
        ]
    
    def get_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Evaluation thresholds for full scoring."""
        return {
            "completion_rate": {
                "excellent": {"min": 1.0, "max": 1.0},
                "good": {"min": 0.99, "max": 1.0},
                "acceptable": {"min": 0.95, "max": 0.99},
                "poor": {"min": 0.90, "max": 0.95},
                "fail": {"min": 0.0, "max": 0.90}
            },
            "feature_dimensions": {
                "excellent": {"min": 64, "max": 128},
                "good": {"min": 32, "max": 64},
                "acceptable": {"min": 16, "max": 32},
                "poor": {"min": 8, "max": 16},
                "fail": {"min": 0, "max": 8}
            },
            "score_std": {
                "excellent": {"min": 0.15, "max": 0.30},
                "good": {"min": 0.10, "max": 0.15},
                "acceptable": {"min": 0.05, "max": 0.10},
                "poor": {"min": 0.30, "max": 0.50},
                "fail": {"min": 0.0, "max": 0.05}  # Too uniform = no signal
            }
        }
    
    def interpret_results(self) -> str:
        """Interpret full scoring results."""
        completion = self.results.get("completion_rate", 0)
        scored = self.results.get("survivors_scored", 0)
        total = self.results.get("survivors_total", 0)
        features = self.results.get("feature_dimensions", 0)
        mean_score = self.results.get("mean_score", 0)
        score_std = self.results.get("score_std", 0)
        top_candidates = self.results.get("top_candidates", 0)
        
        interpretation_parts = []
        
        # Completion analysis
        if completion >= 1.0:
            interpretation_parts.append(f"Complete scoring: {scored}/{total} survivors processed (100%).")
        elif completion >= 0.99:
            interpretation_parts.append(f"Near-complete scoring: {scored}/{total} survivors ({completion:.1%}).")
        elif completion >= 0.95:
            interpretation_parts.append(f"Acceptable completion: {scored}/{total} survivors ({completion:.1%}).")
        else:
            interpretation_parts.append(f"Incomplete scoring: Only {scored}/{total} survivors ({completion:.1%}) - investigate failures.")
        
        # Feature analysis
        if features >= 64:
            interpretation_parts.append(f"Rich feature set ({features} dimensions) - good for ML.")
        elif features >= 32:
            interpretation_parts.append(f"Adequate feature set ({features} dimensions).")
        else:
            interpretation_parts.append(f"Limited feature set ({features} dimensions) - may reduce ML accuracy.")
        
        # Score distribution analysis
        if score_std >= 0.15 and score_std <= 0.30:
            interpretation_parts.append(f"Healthy score distribution (std={score_std:.3f}) - good differentiation.")
        elif score_std < 0.05:
            interpretation_parts.append(f"Very uniform scores (std={score_std:.3f}) - weak signal, may need parameter adjustment.")
        elif score_std > 0.30:
            interpretation_parts.append(f"High score variance (std={score_std:.3f}) - strong differentiation but verify quality.")
        else:
            interpretation_parts.append(f"Score distribution std={score_std:.3f}.")
        
        # Top candidates
        if top_candidates > 0:
            interpretation_parts.append(f"Identified {top_candidates} top candidates for ML training.")
        
        return " ".join(interpretation_parts)
    
    def get_retry_suggestions(self) -> List[Dict[str, Any]]:
        """Suggest parameter adjustments for retry."""
        suggestions = []
        completion = self.results.get("completion_rate", 0)
        score_std = self.results.get("score_std", 0)
        features = self.results.get("feature_dimensions", 0)
        batch_size = self.results.get("config", {}).get("batch_size", 1000)
        
        # Incomplete scoring
        if completion < 0.95:
            suggestions.append({
                "param": "batch_size",
                "suggestion": max(batch_size // 2, 100),
                "reason": f"Low completion rate ({completion:.1%}) - smaller batches may reduce failures"
            })
            suggestions.append({
                "param": "distributed",
                "suggestion": True,
                "reason": "Enable distributed processing for large survivor sets"
            })
        
        # Uniform scores (no signal)
        if score_std < 0.05:
            suggestions.append({
                "param": "threshold_min",
                "suggestion": 0.001,
                "reason": f"Uniform scores (std={score_std:.3f}) - adjust scoring threshold"
            })
        
        # Limited features
        if features < 32:
            suggestions.append({
                "param": "feature_config",
                "suggestion": "expanded",
                "reason": f"Limited features ({features}) - use expanded feature extraction"
            })
        
        return suggestions


def create_full_scoring_context(
    results: Dict[str, Any],
    run_number: int = 1,
    manifest_path: str = None
) -> FullScoringContext:
    """Factory function to create full scoring context."""
    ctx = FullScoringContext(
        run_number=run_number,
        results=results
    )
    
    if manifest_path:
        ctx.load_manifest(manifest_path)
    
    return ctx
