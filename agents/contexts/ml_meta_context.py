#!/usr/bin/env python3
"""
ML Meta-Optimizer Context - Step 4 specialized agent context.

Provides domain expertise for evaluating neural network architecture
optimization results.

Version: 3.2.0
"""

from typing import Dict, Any, List
from agents.contexts.base_agent_context import BaseAgentContext


class MLMetaContext(BaseAgentContext):
    """
    Specialized context for Step 4: ML Meta-Optimizer.
    
    Key focus areas:
    - Architecture optimization score
    - Layer configuration quality
    - Model complexity vs performance
    - Training feasibility
    """
    
    agent_name: str = "ml_meta_agent"
    pipeline_step: int = 4
    
    def get_key_metrics(self) -> List[str]:
        """Key metrics for ML meta-optimization."""
        return [
            "architecture_score",
            "best_layers",
            "best_neurons",
            "best_dropout",
            "best_learning_rate",
            "validation_loss",
            "model_parameters",
            "convergence_trial",
            "execution_time_seconds"
        ]
    
    def get_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Evaluation thresholds for ML meta-optimizer."""
        return {
            "architecture_score": {
                "excellent": {"min": 0.90, "max": 1.0},
                "good": {"min": 0.75, "max": 0.90},
                "acceptable": {"min": 0.60, "max": 0.75},
                "poor": {"min": 0.40, "max": 0.60},
                "fail": {"min": 0.0, "max": 0.40}
            },
            "validation_loss": {
                "excellent": {"min": 0.0, "max": 0.1},
                "good": {"min": 0.1, "max": 0.3},
                "acceptable": {"min": 0.3, "max": 0.5},
                "poor": {"min": 0.5, "max": 1.0},
                "fail": {"min": 1.0, "max": None}
            },
            "best_layers": {
                "excellent": {"min": 2, "max": 4},
                "good": {"min": 4, "max": 6},
                "acceptable": {"min": 1, "max": 2},
                "poor": {"min": 6, "max": 8},
                "fail": {"min": 8, "max": None}
            },
            "model_parameters": {
                "excellent": {"min": 10000, "max": 100000},
                "good": {"min": 100000, "max": 500000},
                "acceptable": {"min": 5000, "max": 10000},
                "poor": {"min": 500000, "max": 2000000},
                "fail": {"min": 2000000, "max": None}
            }
        }
    
    def interpret_results(self) -> str:
        """Interpret ML meta-optimization results."""
        arch_score = self.results.get("architecture_score", 0)
        layers = self.results.get("best_layers", 0)
        neurons = self.results.get("best_neurons", [])
        dropout = self.results.get("best_dropout", 0)
        lr = self.results.get("best_learning_rate", 0)
        val_loss = self.results.get("validation_loss", 0)
        params = self.results.get("model_parameters", 0)
        
        interpretation_parts = []
        
        # Architecture score analysis
        if arch_score >= 0.90:
            interpretation_parts.append(f"Excellent architecture found (score={arch_score:.3f}).")
        elif arch_score >= 0.75:
            interpretation_parts.append(f"Good architecture (score={arch_score:.3f}) - should train well.")
        elif arch_score >= 0.60:
            interpretation_parts.append(f"Acceptable architecture (score={arch_score:.3f}) - may need refinement.")
        else:
            interpretation_parts.append(f"Poor architecture score ({arch_score:.3f}) - optimization needs more trials.")
        
        # Layer analysis
        if layers >= 2 and layers <= 4:
            interpretation_parts.append(f"Optimal depth ({layers} layers) - good complexity balance.")
        elif layers < 2:
            interpretation_parts.append(f"Shallow network ({layers} layers) - may underfit.")
        else:
            interpretation_parts.append(f"Deep network ({layers} layers) - watch for overfitting.")
        
        # Parameter count analysis
        if params > 0:
            if params < 100000:
                interpretation_parts.append(f"Lightweight model ({params:,} params) - fast inference.")
            elif params < 500000:
                interpretation_parts.append(f"Medium model ({params:,} params) - good balance.")
            else:
                interpretation_parts.append(f"Large model ({params:,} params) - may need regularization.")
        
        # Validation loss
        if val_loss > 0:
            if val_loss < 0.1:
                interpretation_parts.append(f"Low validation loss ({val_loss:.4f}) - strong fit.")
            elif val_loss < 0.3:
                interpretation_parts.append(f"Acceptable validation loss ({val_loss:.4f}).")
            else:
                interpretation_parts.append(f"High validation loss ({val_loss:.4f}) - model struggles to fit data.")
        
        # Configuration summary
        if neurons:
            neuron_str = "→".join(str(n) for n in neurons[:4])
            interpretation_parts.append(f"Architecture: {neuron_str}, dropout={dropout}, lr={lr}.")
        
        return " ".join(interpretation_parts)
    
    def get_retry_suggestions(self) -> List[Dict[str, Any]]:
        """Suggest parameter adjustments for retry."""
        suggestions = []
        arch_score = self.results.get("architecture_score", 0)
        val_loss = self.results.get("validation_loss", 0)
        layers = self.results.get("best_layers", 3)
        convergence = self.results.get("convergence_trial", 0)
        total_trials = self.results.get("config", {}).get("trials", 100)
        
        # Low architecture score
        if arch_score < 0.60:
            suggestions.append({
                "param": "trials",
                "suggestion": min(total_trials * 2, 200),
                "reason": f"Low architecture score ({arch_score:.3f}) - more trials for better search"
            })
            suggestions.append({
                "param": "max_layers",
                "suggestion": 6,
                "reason": "Expand architecture search space"
            })
        
        # High validation loss
        if val_loss > 0.5:
            suggestions.append({
                "param": "min_layers",
                "suggestion": 2,
                "reason": f"High validation loss ({val_loss:.4f}) - try deeper networks"
            })
            suggestions.append({
                "param": "mode",
                "suggestion": "deep",
                "reason": "Use deep search mode for better architecture exploration"
            })
        
        # Shallow network
        if layers < 2:
            suggestions.append({
                "param": "min_layers",
                "suggestion": 2,
                "reason": f"Shallow network ({layers} layers) - enforce minimum depth"
            })
        
        # Late convergence
        if total_trials > 0 and convergence > total_trials * 0.8:
            suggestions.append({
                "param": "trials",
                "suggestion": min(total_trials + 50, 200),
                "reason": "Late convergence - optimization may benefit from more trials"
            })
        
        return suggestions


def create_ml_meta_context(
    results: Dict[str, Any],
    run_number: int = 1,
    manifest_path: str = None
) -> MLMetaContext:
    """Factory function to create ML meta context."""
    ctx = MLMetaContext(
        run_number=run_number,
        results=results
    )
    
    if manifest_path:
        ctx.load_manifest(manifest_path)
    
    return ctx
