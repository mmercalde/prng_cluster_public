#!/usr/bin/env python3
"""
Anti-Overfit Context - Step 5 specialized agent context.

Provides domain expertise for evaluating anti-overfitting training
results, including k-fold validation and generalization metrics.

Version: 3.2.0
"""

from typing import Dict, Any, List
from agents.contexts.base_agent_context import BaseAgentContext


class AntiOverfitContext(BaseAgentContext):
    """
    Specialized context for Step 5: Anti-Overfit Training.
    
    Key focus areas:
    - Train/validation loss ratio (overfit detection)
    - K-fold cross-validation stability
    - Generalization gap analysis
    - Early stopping effectiveness
    """
    
    agent_name: str = "anti_overfit_agent"
    pipeline_step: int = 5
    
    def get_key_metrics(self) -> List[str]:
        """Key metrics for anti-overfit training."""
        return [
            "overfit_ratio",
            "train_loss",
            "validation_loss",
            "kfold_mean",
            "kfold_std",
            "best_epoch",
            "total_epochs",
            "early_stopped",
            "dropout_used",
            "execution_time_seconds"
        ]
    
    def get_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Evaluation thresholds for anti-overfit training."""
        return {
            "overfit_ratio": {
                # overfit_ratio = validation_loss / train_loss
                # 1.0 = perfect, >1.0 = overfitting
                "excellent": {"min": 0.95, "max": 1.05},
                "good": {"min": 1.05, "max": 1.15},
                "acceptable": {"min": 1.15, "max": 1.30},
                "poor": {"min": 1.30, "max": 1.50},
                "fail": {"min": 1.50, "max": None}
            },
            "kfold_std": {
                "excellent": {"min": 0.0, "max": 0.02},
                "good": {"min": 0.02, "max": 0.05},
                "acceptable": {"min": 0.05, "max": 0.08},
                "poor": {"min": 0.08, "max": 0.15},
                "fail": {"min": 0.15, "max": None}
            },
            "validation_loss": {
                "excellent": {"min": 0.0, "max": 0.1},
                "good": {"min": 0.1, "max": 0.25},
                "acceptable": {"min": 0.25, "max": 0.4},
                "poor": {"min": 0.4, "max": 0.6},
                "fail": {"min": 0.6, "max": None}
            }
        }
    
    def interpret_results(self) -> str:
        """Interpret anti-overfit training results."""
        overfit_ratio = self.results.get("overfit_ratio", 1.0)
        train_loss = self.results.get("train_loss", 0)
        val_loss = self.results.get("validation_loss", 0)
        kfold_mean = self.results.get("kfold_mean", 0)
        kfold_std = self.results.get("kfold_std", 0)
        best_epoch = self.results.get("best_epoch", 0)
        total_epochs = self.results.get("total_epochs", 0)
        early_stopped = self.results.get("early_stopped", False)
        dropout = self.results.get("dropout_used", 0)
        
        interpretation_parts = []
        
        # Overfit ratio analysis
        if overfit_ratio <= 1.05:
            interpretation_parts.append(f"Excellent generalization (overfit_ratio={overfit_ratio:.3f}) - model generalizes well.")
        elif overfit_ratio <= 1.15:
            interpretation_parts.append(f"Good generalization (overfit_ratio={overfit_ratio:.3f}) - minimal overfitting.")
        elif overfit_ratio <= 1.30:
            interpretation_parts.append(f"Acceptable generalization (overfit_ratio={overfit_ratio:.3f}) - some overfitting present.")
        elif overfit_ratio <= 1.50:
            interpretation_parts.append(f"Poor generalization (overfit_ratio={overfit_ratio:.3f}) - significant overfitting detected.")
        else:
            interpretation_parts.append(f"Severe overfitting (overfit_ratio={overfit_ratio:.3f}) - model memorizing training data.")
        
        # Loss values
        interpretation_parts.append(f"Train loss={train_loss:.4f}, Val loss={val_loss:.4f}.")
        
        # K-fold stability
        if kfold_std <= 0.02:
            interpretation_parts.append(f"Highly stable k-fold results (std={kfold_std:.4f}).")
        elif kfold_std <= 0.05:
            interpretation_parts.append(f"Stable k-fold results (std={kfold_std:.4f}).")
        elif kfold_std <= 0.08:
            interpretation_parts.append(f"Moderate k-fold variance (std={kfold_std:.4f}).")
        else:
            interpretation_parts.append(f"High k-fold variance (std={kfold_std:.4f}) - model performance varies by fold.")
        
        # Training progress
        if total_epochs > 0:
            progress_pct = (best_epoch / total_epochs) * 100
            if early_stopped:
                interpretation_parts.append(f"Early stopped at epoch {best_epoch}/{total_epochs} ({progress_pct:.0f}%).")
            else:
                interpretation_parts.append(f"Best at epoch {best_epoch}/{total_epochs}.")
        
        # Regularization
        if dropout > 0:
            interpretation_parts.append(f"Dropout={dropout:.2f} applied.")
        
        return " ".join(interpretation_parts)
    
    def get_retry_suggestions(self) -> List[Dict[str, Any]]:
        """Suggest parameter adjustments for retry."""
        suggestions = []
        overfit_ratio = self.results.get("overfit_ratio", 1.0)
        kfold_std = self.results.get("kfold_std", 0)
        val_loss = self.results.get("validation_loss", 0)
        dropout = self.results.get("dropout_used", 0)
        current_epochs = self.results.get("config", {}).get("epochs", 100)
        current_k = self.results.get("config", {}).get("k_folds", 5)
        
        # Severe overfitting
        if overfit_ratio > 1.30:
            suggestions.append({
                "param": "dropout_min",
                "suggestion": min(dropout + 0.1, 0.5),
                "reason": f"Overfitting detected (ratio={overfit_ratio:.3f}) - increase dropout"
            })
            suggestions.append({
                "param": "dropout_max",
                "suggestion": min(dropout + 0.2, 0.7),
                "reason": "Expand dropout search range for better regularization"
            })
        
        # High k-fold variance
        if kfold_std > 0.08:
            suggestions.append({
                "param": "k_folds",
                "suggestion": min(current_k + 2, 10),
                "reason": f"High fold variance (std={kfold_std:.4f}) - more folds for stability"
            })
        
        # High validation loss
        if val_loss > 0.4:
            suggestions.append({
                "param": "epochs",
                "suggestion": min(current_epochs + 100, 500),
                "reason": f"High validation loss ({val_loss:.4f}) - more training may help"
            })
        
        # Underfitting (ratio too low means train loss is high)
        if overfit_ratio < 0.95:
            suggestions.append({
                "param": "dropout_max",
                "suggestion": max(dropout - 0.1, 0.1),
                "reason": f"Possible underfitting (ratio={overfit_ratio:.3f}) - reduce regularization"
            })
            suggestions.append({
                "param": "epochs",
                "suggestion": min(current_epochs * 2, 500),
                "reason": "Train longer to reduce underfitting"
            })
        
        return suggestions


def create_anti_overfit_context(
    results: Dict[str, Any],
    run_number: int = 1,
    manifest_path: str = None
) -> AntiOverfitContext:
    """Factory function to create anti-overfit context."""
    ctx = AntiOverfitContext(
        run_number=run_number,
        results=results
    )
    
    if manifest_path:
        ctx.load_manifest(manifest_path)
    
    return ctx
