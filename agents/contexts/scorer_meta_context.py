#!/usr/bin/env python3
"""
Scorer Meta-Optimizer Context - Step 2 specialized agent context.

Provides domain expertise for evaluating ML scorer hyperparameter
optimization results across distributed GPUs.

Version: 3.2.0
"""

from typing import Dict, Any, List
from agents.contexts.base_agent_context import BaseAgentContext


class ScorerMetaContext(BaseAgentContext):
    """
    Specialized context for Step 2: Scorer Meta-Optimizer.
    
    Key focus areas:
    - Validation score convergence
    - Hyperparameter optimization quality
    - Cross-validation stability
    - Distributed execution efficiency
    """
    
    agent_name: str = "scorer_meta_agent"
    pipeline_step: int = 2
    
    def get_key_metrics(self) -> List[str]:
        """Key metrics for scorer meta-optimization."""
        return [
            "best_validation_score",
            "convergence_trial",
            "total_trials",
            "cv_std",
            "best_threshold",
            "best_k_folds",
            "distributed_speedup",
            "execution_time_seconds"
        ]
    
    def get_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Evaluation thresholds for scorer meta-optimizer."""
        return {
            "best_validation_score": {
                "excellent": {"min": 0.95, "max": 1.0},
                "good": {"min": 0.85, "max": 0.95},
                "acceptable": {"min": 0.70, "max": 0.85},
                "poor": {"min": 0.50, "max": 0.70},
                "fail": {"min": 0.0, "max": 0.50}
            },
            "cv_std": {
                "excellent": {"min": 0.0, "max": 0.02},
                "good": {"min": 0.02, "max": 0.05},
                "acceptable": {"min": 0.05, "max": 0.10},
                "poor": {"min": 0.10, "max": 0.20},
                "fail": {"min": 0.20, "max": None}
            },
            "convergence_trial": {
                "excellent": {"min": 0, "max": 20},
                "good": {"min": 21, "max": 50},
                "acceptable": {"min": 51, "max": 100},
                "poor": {"min": 101, "max": 200},
                "fail": {"min": 201, "max": None}
            }
        }
    
    def interpret_results(self) -> str:
        """Interpret scorer meta-optimization results."""
        val_score = self.results.get("best_validation_score", 0)
        cv_std = self.results.get("cv_std", 0)
        convergence = self.results.get("convergence_trial", 0)
        total_trials = self.results.get("total_trials", 0)
        threshold = self.results.get("best_threshold", "unknown")
        
        interpretation_parts = []
        
        # Validation score analysis
        if val_score >= 0.95:
            interpretation_parts.append(f"Excellent validation score ({val_score:.3f}) indicates strong scorer configuration.")
        elif val_score >= 0.85:
            interpretation_parts.append(f"Good validation score ({val_score:.3f}) - scorer should perform well.")
        elif val_score >= 0.70:
            interpretation_parts.append(f"Acceptable validation score ({val_score:.3f}) - may need refinement.")
        else:
            interpretation_parts.append(f"Low validation score ({val_score:.3f}) - scorer configuration needs improvement.")
        
        # CV stability analysis
        if cv_std <= 0.02:
            interpretation_parts.append(f"Very stable cross-validation (std={cv_std:.3f}).")
        elif cv_std <= 0.05:
            interpretation_parts.append(f"Stable cross-validation (std={cv_std:.3f}).")
        elif cv_std <= 0.10:
            interpretation_parts.append(f"Moderate CV variance (std={cv_std:.3f}) - results may vary.")
        else:
            interpretation_parts.append(f"High CV variance (std={cv_std:.3f}) - unstable configuration.")
        
        # Convergence analysis
        if total_trials > 0:
            convergence_pct = (convergence / total_trials) * 100
            if convergence_pct < 30:
                interpretation_parts.append(f"Quick convergence at trial {convergence}/{total_trials} ({convergence_pct:.0f}%).")
            elif convergence_pct < 70:
                interpretation_parts.append(f"Normal convergence at trial {convergence}/{total_trials}.")
            else:
                interpretation_parts.append(f"Late convergence at trial {convergence}/{total_trials} - more trials may help.")
        
        # Optimal parameters
        interpretation_parts.append(f"Optimal threshold={threshold}.")
        
        return " ".join(interpretation_parts)
    
    def get_retry_suggestions(self) -> List[Dict[str, Any]]:
        """Suggest parameter adjustments for retry."""
        suggestions = []
        val_score = self.results.get("best_validation_score", 0)
        cv_std = self.results.get("cv_std", 0)
        convergence = self.results.get("convergence_trial", 0)
        total_trials = self.results.get("total_trials", 50)
        current_k_folds = self.results.get("best_k_folds", 5)
        
        # Low validation score
        if val_score < 0.70:
            suggestions.append({
                "param": "n_trials",
                "suggestion": min(total_trials * 2, 500),
                "reason": f"Low validation score ({val_score:.3f}) - increase trials for better optimization"
            })
            suggestions.append({
                "param": "threshold_max",
                "suggestion": 0.3,
                "reason": "Expand threshold search range"
            })
        
        # High CV variance
        if cv_std > 0.10:
            suggestions.append({
                "param": "k_folds",
                "suggestion": min(current_k_folds + 2, 10),
                "reason": f"High CV variance ({cv_std:.3f}) - more folds may stabilize"
            })
        
        # Late convergence
        if total_trials > 0 and convergence > total_trials * 0.7:
            suggestions.append({
                "param": "n_trials",
                "suggestion": min(total_trials + 50, 500),
                "reason": f"Late convergence at trial {convergence} - optimization may not have plateaued"
            })
        
        return suggestions


def create_scorer_meta_context(
    results: Dict[str, Any],
    run_number: int = 1,
    manifest_path: str = None
) -> ScorerMetaContext:
    """Factory function to create scorer meta context."""
    ctx = ScorerMetaContext(
        run_number=run_number,
        results=results
    )
    
    if manifest_path:
        ctx.load_manifest(manifest_path)
    
    return ctx
