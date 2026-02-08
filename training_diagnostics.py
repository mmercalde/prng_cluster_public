#!/usr/bin/env python3
"""
Training Diagnostics — Chapter 14 Phase 1
==========================================

Core diagnostic module for live training introspection across all 4 model types.
Uses PyTorch dynamic computational graph hooks for neural networks and native
callbacks for tree models (XGBoost, LightGBM, CatBoost).

Design Invariants (non-negotiable):
1. PASSIVE OBSERVER — Never modifies gradients, weights, or training behavior
2. BEST-EFFORT, NON-FATAL — All code paths wrapped in try/except
3. ABSENT ≠ FAILURE — Missing diagnostics maps to PROCEED, not BLOCK

Schema Version: 1.1.0 (Multi-model design per Option D)

Usage:
    # Single model
    diag = TrainingDiagnostics.create('catboost')
    diag.attach(model)
    # ... training loop calls diag.on_round_end() ...
    diag.detach()
    diag.save()
    
    # Multi-model (--compare-models)
    collector = MultiModelDiagnostics()
    for model_type in ['neural_net', 'xgboost', 'lightgbm', 'catboost']:
        diag = collector.create_for_model(model_type)
        # ... training ...
        collector.finalize_model(model_type, diag, metrics)
    collector.save()

Author: Distributed PRNG Analysis System
Date: February 2026

Version History:
    1.0.0   2026-02-08  Session 69  Initial implementation (Phase 1-2)
                                    - TrainingDiagnostics ABC with factory
                                    - NNDiagnostics with PyTorch hooks
                                    - TreeDiagnostics for XGB/LGB/CatBoost
                                    - MultiModelDiagnostics collector
                                    - Severity classification system
    1.1.0   2026-02-08  Session 71  Phase 5: FIFO history pruning
                                    - Added MAX_HISTORY_FILES constant (100)
                                    - Added _prune_history_fifo() function
                                    - Glob narrowed to compare_models_*.json
                                    - Added is_dir() defensive check
                                    - Team Beta approved with refinements
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

SCHEMA_VERSION = "1.1.0"
DEFAULT_OUTPUT_DIR = "diagnostics_outputs"
DEFAULT_OUTPUT_FILE = "training_diagnostics.json"
DEFAULT_HISTORY_DIR = "diagnostics_outputs/history"
MAX_HISTORY_FILES = 100  # FIFO pruning limit (Team Beta Session 69)

# Severity thresholds (can be overridden via config)
SEVERITY_THRESHOLDS = {
    "dead_neuron_pct": {"warning": 15.0, "critical": 25.0},
    "gradient_spread_ratio": {"warning": 1000.0, "critical": 10000.0},
    "overfit_gap_ratio": {"warning": 0.3, "critical": 0.5},
    "gradient_norm_min": {"warning": 1e-6, "critical": 1e-8},
}


# =============================================================================
# BASE CLASS
# =============================================================================

class TrainingDiagnostics(ABC):
    """
    Abstract base class for model-specific training diagnostics.
    
    Subclasses implement attach/detach/capture logic for their model type.
    All subclasses share the same output schema and severity classification.
    """
    
    model_type: str = "base"
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        capture_every_n: int = 5,
        max_rounds: int = 500,
    ):
        """
        Args:
            feature_names: Names of input features. If None, uses generic names.
            capture_every_n: Capture detailed snapshot every N rounds.
            max_rounds: Safety cap to prevent unbounded memory growth.
        """
        self.feature_names = feature_names or []
        self.capture_every_n = capture_every_n
        self.max_rounds = max_rounds
        
        # State
        self._attached = False
        self._round_data: List[Dict[str, Any]] = []
        self._training_start: Optional[datetime] = None
        self._training_end: Optional[datetime] = None
        
        # Final metrics (set after training)
        self._final_metrics: Dict[str, Any] = {}
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> "TrainingDiagnostics":
        """
        Factory method — returns appropriate subclass instance.
        
        Args:
            model_type: One of 'neural_net', 'xgboost', 'lightgbm', 'catboost'
            **kwargs: Passed to subclass __init__
            
        Returns:
            TrainingDiagnostics subclass instance
        """
        subclasses = {
            "neural_net": NNDiagnostics,
            "xgboost": XGBDiagnostics,
            "lightgbm": LGBDiagnostics,
            "catboost": CatBoostDiagnostics,
        }
        
        if model_type not in subclasses:
            logger.warning(f"Unknown model_type '{model_type}', using base diagnostics")
            # Return a minimal diagnostics that just tracks losses
            return MinimalDiagnostics(**kwargs)
        
        return subclasses[model_type](**kwargs)
    
    @abstractmethod
    def attach(self, model, context: Optional[Dict] = None) -> "TrainingDiagnostics":
        """
        Register hooks/callbacks on the model.
        
        Args:
            model: The model instance to attach to
            context: Optional training context (e.g., data loaders)
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def detach(self) -> None:
        """Remove all hooks/callbacks. Safe to call multiple times."""
        pass
    
    @abstractmethod
    def on_round_end(
        self,
        round_num: int,
        train_loss: float,
        val_loss: float,
        **kwargs
    ) -> None:
        """
        Called after each training epoch/round.
        
        Args:
            round_num: Current epoch/round number (0-indexed)
            train_loss: Training loss for this round
            val_loss: Validation loss for this round
            **kwargs: Model-specific additional metrics
        """
        pass
    
    def set_final_metrics(self, metrics: Dict[str, Any]) -> None:
        """Set final training metrics (MSE, R², etc.) after training completes."""
        self._final_metrics = metrics
        self._training_end = datetime.now()
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate complete diagnostics report.
        
        Returns:
            Dict matching the Phase 1 schema
        """
        try:
            training_summary = self._compute_training_summary()
            diagnosis = self._diagnose(training_summary)
            model_specific = self._get_model_specific()
            
            return {
                "status": "complete" if self._round_data else "partial",
                "training_summary": training_summary,
                "diagnosis": diagnosis,
                "model_specific": model_specific,
                "round_data_truncated": len(self._round_data) > 50,
                "round_data_sample": self._round_data[-10:] if self._round_data else [],
            }
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "diagnosis": {
                    "severity": "absent",
                    "issues": [f"Diagnostics failed: {e}"],
                    "suggested_fixes": [],
                    "confidence": 0.0
                }
            }
    
    def _compute_training_summary(self) -> Dict[str, Any]:
        """Compute summary statistics from round data."""
        if not self._round_data:
            return {}
        
        train_losses = [r.get("train_loss", 0) for r in self._round_data]
        val_losses = [r.get("val_loss", 0) for r in self._round_data]
        
        best_val_idx = int(np.argmin(val_losses)) if val_losses else 0
        
        return {
            "rounds_captured": len(self._round_data),
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": min(val_losses) if val_losses else None,
            "best_val_round": best_val_idx,
            "overfit_gap": (val_losses[-1] - train_losses[-1]) if train_losses and val_losses else None,
        }
    
    def _diagnose(self, training_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify severity and detect issues.
        
        Override in subclasses for model-specific diagnosis.
        """
        issues = []
        suggested_fixes = []
        severity = "ok"
        
        # Check overfit gap
        overfit_gap = training_summary.get("overfit_gap")
        if overfit_gap is not None:
            final_train = training_summary.get("final_train_loss", 1)
            if final_train > 0:
                gap_ratio = overfit_gap / final_train
                if gap_ratio > SEVERITY_THRESHOLDS["overfit_gap_ratio"]["critical"]:
                    issues.append(f"Severe overfitting (gap ratio: {gap_ratio:.2f})")
                    suggested_fixes.append("Increase regularization or reduce model complexity")
                    severity = "critical"
                elif gap_ratio > SEVERITY_THRESHOLDS["overfit_gap_ratio"]["warning"]:
                    issues.append(f"Moderate overfitting (gap ratio: {gap_ratio:.2f})")
                    suggested_fixes.append("Consider early stopping or dropout")
                    severity = "warning"
        
        return {
            "severity": severity,
            "issues": issues,
            "suggested_fixes": suggested_fixes,
            "confidence": 0.8 if self._round_data else 0.0
        }
    
    def _get_model_specific(self) -> Dict[str, Any]:
        """Return model-specific diagnostics. Override in subclasses."""
        return {}
    
    @property
    def severity(self) -> str:
        """Quick access to current severity level."""
        try:
            report = self.get_report()
            return report.get("diagnosis", {}).get("severity", "absent")
        except:
            return "absent"
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save diagnostics to JSON file.
        
        Args:
            path: Output path. If None, uses default location.
            
        Returns:
            Path where file was written
        """
        if path is None:
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            path = os.path.join(DEFAULT_OUTPUT_DIR, f"{self.model_type}_diagnostics.json")
        
        report = self.get_report()
        report["model_type"] = self.model_type
        report["generated_at"] = datetime.now().isoformat()
        report["schema_version"] = SCHEMA_VERSION
        
        try:
            with open(path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Diagnostics saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save diagnostics: {e}")
        
        return path


# =============================================================================
# NEURAL NET DIAGNOSTICS — PyTorch Dynamic Graph Hooks
# =============================================================================

class NNDiagnostics(TrainingDiagnostics):
    """
    Neural network diagnostics using PyTorch hooks.
    
    Hooks into register_forward_hook() and register_full_backward_hook()
    to capture activations, gradients, and dead neuron statistics.
    """
    
    model_type = "neural_net"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Hook handles (for cleanup)
        self._forward_hooks = []
        self._backward_hooks = []
        
        # Per-forward-pass storage (cleared each round)
        self._activations: Dict[str, Any] = {}
        self._gradients: Dict[str, Any] = {}
        self._input_gradients: Optional[Any] = None
        
        # Model reference
        self._model = None
        self._layer_names: List[str] = []
        
        # Accumulated stats
        self._dead_neuron_history: Dict[str, List[float]] = defaultdict(list)
        self._gradient_norm_history: Dict[str, List[float]] = defaultdict(list)
    
    def attach(self, model, context: Optional[Dict] = None) -> "NNDiagnostics":
        """Register forward and backward hooks on all Linear layers."""
        try:
            import torch.nn as nn
            
            if self._attached:
                self.detach()
            
            self._model = model
            self._layer_names = []
            self._training_start = datetime.now()
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    self._layer_names.append(name)
                    
                    # Forward hook: captures activations
                    fh = module.register_forward_hook(self._make_forward_hook(name))
                    self._forward_hooks.append(fh)
                    
                    # Backward hook: captures gradients
                    bh = module.register_full_backward_hook(self._make_backward_hook(name))
                    self._backward_hooks.append(bh)
            
            self._attached = True
            logger.info(f"NNDiagnostics attached to {len(self._layer_names)} layers: {self._layer_names}")
            
        except Exception as e:
            logger.error(f"Failed to attach NN hooks: {e}")
            self._attached = False
        
        return self
    
    def detach(self) -> None:
        """Remove all hooks."""
        for h in self._forward_hooks:
            try:
                h.remove()
            except:
                pass
        for h in self._backward_hooks:
            try:
                h.remove()
            except:
                pass
        
        self._forward_hooks.clear()
        self._backward_hooks.clear()
        self._attached = False
        logger.info("NNDiagnostics detached")
    
    def _make_forward_hook(self, layer_name: str):
        """Create forward hook that captures activation statistics."""
        def hook(module, input_tensor, output_tensor):
            try:
                output = output_tensor.detach()
                self._activations[layer_name] = {
                    "mean": float(output.mean().item()),
                    "std": float(output.std().item()),
                    "dead_pct": float((output == 0).float().mean().item() * 100),
                    "neuron_count": output.shape[-1] if len(output.shape) > 1 else 1,
                }
            except Exception as e:
                logger.debug(f"Forward hook error for {layer_name}: {e}")
        return hook
    
    def _make_backward_hook(self, layer_name: str):
        """Create backward hook that captures gradient statistics."""
        def hook(module, grad_input, grad_output):
            try:
                if grad_output[0] is not None:
                    grad = grad_output[0].detach()
                    self._gradients[layer_name] = {
                        "norm": float(grad.norm().item()),
                        "mean": float(grad.abs().mean().item()),
                        "max": float(grad.abs().max().item()),
                    }
                
                # Capture input gradients for feature attribution (first layer only)
                if layer_name == self._layer_names[0] and grad_input[0] is not None:
                    self._input_gradients = grad_input[0].detach()
            except Exception as e:
                logger.debug(f"Backward hook error for {layer_name}: {e}")
        return hook
    
    def on_round_end(
        self,
        round_num: int,
        train_loss: float,
        val_loss: float,
        learning_rate: Optional[float] = None,
        **kwargs
    ) -> None:
        """Capture snapshot after each epoch."""
        try:
            # Only capture detailed data every N rounds (to save memory)
            capture_detail = (round_num % self.capture_every_n == 0)
            
            snapshot = {
                "round": round_num,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            
            if learning_rate is not None:
                snapshot["learning_rate"] = float(learning_rate)
            
            if capture_detail and self._activations:
                snapshot["layers"] = {}
                for name in self._layer_names:
                    act = self._activations.get(name, {})
                    grad = self._gradients.get(name, {})
                    snapshot["layers"][name] = {**act, **grad}
                    
                    # Track history for analysis
                    if "dead_pct" in act:
                        self._dead_neuron_history[name].append(act["dead_pct"])
                    if "norm" in grad:
                        self._gradient_norm_history[name].append(grad["norm"])
                
                # Feature gradients (if available)
                if self._input_gradients is not None:
                    try:
                        feat_grads = self._input_gradients.abs().mean(dim=0)
                        top_k = min(10, len(feat_grads))
                        top_indices = feat_grads.argsort(descending=True)[:top_k]
                        
                        snapshot["feature_gradients"] = {
                            "top_10": [
                                {
                                    "feature": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                                    "magnitude": float(feat_grads[i].item())
                                }
                                for i in top_indices.tolist()
                            ],
                            "spread_ratio": float(feat_grads.max().item() / (feat_grads.min().item() + 1e-10))
                        }
                    except Exception as e:
                        logger.debug(f"Feature gradient capture failed: {e}")
            
            # Respect max_rounds cap
            if len(self._round_data) < self.max_rounds:
                self._round_data.append(snapshot)
            
            # Clear per-pass storage
            self._activations.clear()
            self._gradients.clear()
            self._input_gradients = None
            
        except Exception as e:
            logger.error(f"on_round_end failed: {e}")
    
    def _diagnose(self, training_summary: Dict[str, Any]) -> Dict[str, Any]:
        """NN-specific diagnosis including gradient health and dead neurons."""
        base_diagnosis = super()._diagnose(training_summary)
        issues = base_diagnosis["issues"].copy()
        suggested_fixes = base_diagnosis["suggested_fixes"].copy()
        severity = base_diagnosis["severity"]
        
        # Check dead neurons
        for layer_name, dead_history in self._dead_neuron_history.items():
            if dead_history:
                final_dead_pct = dead_history[-1]
                if final_dead_pct > SEVERITY_THRESHOLDS["dead_neuron_pct"]["critical"]:
                    issues.append(f"Dead neurons critical in {layer_name}: {final_dead_pct:.1f}%")
                    suggested_fixes.append("Switch ReLU → LeakyReLU or GELU")
                    severity = "critical"
                elif final_dead_pct > SEVERITY_THRESHOLDS["dead_neuron_pct"]["warning"]:
                    issues.append(f"Dead neurons elevated in {layer_name}: {final_dead_pct:.1f}%")
                    suggested_fixes.append("Consider LeakyReLU or add BatchNorm")
                    if severity == "ok":
                        severity = "warning"
        
        # Check gradient health
        for layer_name, grad_history in self._gradient_norm_history.items():
            if grad_history:
                final_grad_norm = grad_history[-1]
                if final_grad_norm < SEVERITY_THRESHOLDS["gradient_norm_min"]["critical"]:
                    issues.append(f"Vanishing gradients in {layer_name}: norm={final_grad_norm:.2e}")
                    suggested_fixes.append("Reduce network depth or add skip connections")
                    severity = "critical"
                elif final_grad_norm < SEVERITY_THRESHOLDS["gradient_norm_min"]["warning"]:
                    issues.append(f"Weak gradients in {layer_name}: norm={final_grad_norm:.2e}")
                    if severity == "ok":
                        severity = "warning"
        
        # Check gradient spread (feature scale imbalance)
        if self._round_data and "feature_gradients" in self._round_data[-1]:
            spread = self._round_data[-1]["feature_gradients"].get("spread_ratio", 1)
            if spread > SEVERITY_THRESHOLDS["gradient_spread_ratio"]["critical"]:
                issues.append(f"Feature scale imbalance: gradient spread {spread:.0f}x")
                suggested_fixes.append("Add BatchNorm to input layer or normalize features")
                severity = "critical"
            elif spread > SEVERITY_THRESHOLDS["gradient_spread_ratio"]["warning"]:
                issues.append(f"Moderate feature scale imbalance: gradient spread {spread:.0f}x")
                if severity == "ok":
                    severity = "warning"
        
        return {
            "severity": severity,
            "issues": issues,
            "suggested_fixes": suggested_fixes,
            "confidence": 0.9 if self._round_data else 0.0
        }
    
    def _get_model_specific(self) -> Dict[str, Any]:
        """Return NN-specific gradient health and layer statistics."""
        model_specific = {
            "gradient_health": {
                "vanishing": False,
                "exploding": False,
                "dead_neuron_pct": 0.0,
                "gradient_spread_ratio": 1.0
            },
            "layer_health": {}
        }
        
        # Compute final dead neuron percentage (average across layers)
        total_dead = []
        for layer_name, dead_history in self._dead_neuron_history.items():
            if dead_history:
                model_specific["layer_health"][layer_name] = {
                    "dead_pct": dead_history[-1],
                    "dead_trend": "increasing" if len(dead_history) > 1 and dead_history[-1] > dead_history[0] else "stable"
                }
                total_dead.append(dead_history[-1])
        
        if total_dead:
            model_specific["gradient_health"]["dead_neuron_pct"] = np.mean(total_dead)
        
        # Gradient norms
        for layer_name, grad_history in self._gradient_norm_history.items():
            if grad_history:
                if layer_name in model_specific["layer_health"]:
                    model_specific["layer_health"][layer_name]["gradient_norm"] = grad_history[-1]
                else:
                    model_specific["layer_health"][layer_name] = {"gradient_norm": grad_history[-1]}
                
                # Detect vanishing/exploding
                if grad_history[-1] < 1e-8:
                    model_specific["gradient_health"]["vanishing"] = True
                if grad_history[-1] > 1e6:
                    model_specific["gradient_health"]["exploding"] = True
        
        # Feature gradient spread
        if self._round_data and "feature_gradients" in self._round_data[-1]:
            model_specific["gradient_health"]["gradient_spread_ratio"] = \
                self._round_data[-1]["feature_gradients"].get("spread_ratio", 1.0)
        
        return model_specific


# =============================================================================
# TREE MODEL DIAGNOSTICS — Native Callbacks
# =============================================================================

class TreeDiagnostics(TrainingDiagnostics):
    """Base class for tree model diagnostics (XGBoost, LightGBM, CatBoost)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_importance: Dict[str, float] = {}
        self._evals_result: Dict[str, List[float]] = {}
        self._best_iteration: Optional[int] = None
    
    def attach(self, model, context: Optional[Dict] = None) -> "TreeDiagnostics":
        """Tree models don't need pre-attach hooks — data collected after training."""
        self._attached = True
        self._training_start = datetime.now()
        logger.info(f"{self.model_type} diagnostics attached (post-training collection)")
        return self
    
    def detach(self) -> None:
        """Nothing to detach for tree models."""
        self._attached = False
    
    def on_round_end(
        self,
        round_num: int,
        train_loss: float,
        val_loss: float,
        **kwargs
    ) -> None:
        """Record round data from callbacks."""
        try:
            if len(self._round_data) < self.max_rounds:
                snapshot = {
                    "round": round_num,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
                snapshot.update({k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))})
                self._round_data.append(snapshot)
        except Exception as e:
            logger.error(f"Tree on_round_end failed: {e}")
    
    def set_evals_result(self, evals_result: Dict) -> None:
        """Set evaluation results from model.evals_result() or equivalent."""
        self._evals_result = evals_result
    
    def set_feature_importance(self, importance: Dict[str, float]) -> None:
        """Set feature importance scores."""
        self._feature_importance = importance
    
    def set_best_iteration(self, iteration: int) -> None:
        """Set best iteration (early stopping point)."""
        self._best_iteration = iteration
    
    def _get_model_specific(self) -> Dict[str, Any]:
        """Return tree-specific feature importance and early stopping info."""
        model_specific = {
            "early_stopping_triggered": self._best_iteration is not None,
            "best_iteration": self._best_iteration,
        }
        
        if self._feature_importance:
            # Sort by importance
            sorted_features = sorted(
                self._feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            model_specific["feature_importance"] = {
                "top_10": [{"feature": k, "importance": v} for k, v in sorted_features[:10]],
                "concentration_ratio": sorted_features[0][1] / (sum(v for _, v in sorted_features) + 1e-10)
                    if sorted_features else 0.0
            }
        
        return model_specific


class XGBDiagnostics(TreeDiagnostics):
    """XGBoost-specific diagnostics using evals_result() and feature_importances_."""
    model_type = "xgboost"


class LGBDiagnostics(TreeDiagnostics):
    """LightGBM-specific diagnostics using record_evaluation and feature_importance()."""
    model_type = "lightgbm"


class CatBoostDiagnostics(TreeDiagnostics):
    """CatBoost-specific diagnostics using get_evals_result() and get_feature_importance()."""
    model_type = "catboost"


class MinimalDiagnostics(TrainingDiagnostics):
    """Fallback diagnostics for unknown model types — tracks losses only."""
    model_type = "unknown"
    
    def attach(self, model, context: Optional[Dict] = None) -> "MinimalDiagnostics":
        self._attached = True
        self._training_start = datetime.now()
        return self
    
    def detach(self) -> None:
        self._attached = False
    
    def on_round_end(self, round_num: int, train_loss: float, val_loss: float, **kwargs) -> None:
        if len(self._round_data) < self.max_rounds:
            self._round_data.append({
                "round": round_num,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            })


# =============================================================================
# MULTI-MODEL COLLECTOR (Option D: Compare Models Support)
# =============================================================================

class MultiModelDiagnostics:
    """
    Collector for multi-model diagnostics during --compare-models runs.
    
    Aggregates diagnostics from all trained models into a single output file
    with comparison metadata.
    """
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        self._model_reports: Dict[str, Dict[str, Any]] = {}
        self._comparison: Dict[str, Any] = {}
        self._mode = "compare_models"
        self._start_time = datetime.now()
    
    def create_for_model(self, model_type: str, **kwargs) -> TrainingDiagnostics:
        """Create diagnostics instance for a specific model type."""
        return TrainingDiagnostics.create(model_type, **kwargs)
    
    def finalize_model(
        self,
        model_type: str,
        diagnostics: TrainingDiagnostics,
        final_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Finalize and store diagnostics for a model after training completes.
        
        Args:
            model_type: The model type identifier
            diagnostics: The diagnostics instance
            final_metrics: Final training metrics (MSE, R², etc.)
        """
        try:
            if final_metrics:
                diagnostics.set_final_metrics(final_metrics)
            
            report = diagnostics.get_report()
            self._model_reports[model_type] = report
            
            logger.info(f"Finalized diagnostics for {model_type}: severity={report.get('diagnosis', {}).get('severity', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to finalize {model_type} diagnostics: {e}")
            self._model_reports[model_type] = {
                "status": "failed",
                "error": str(e),
                "diagnosis": {"severity": "absent", "issues": [], "suggested_fixes": [], "confidence": 0.0}
            }
    
    def set_comparison_result(
        self,
        winner: str,
        ranking: List[str],
        winner_metric: str = "val_loss",
        winner_value: Optional[float] = None
    ) -> None:
        """Set the model comparison results."""
        self._comparison = {
            "winner": winner,
            "winner_metric": winner_metric,
            "winner_value": winner_value,
            "ranking": ranking,
        }
        
        # Compute NN gap to winner (for Option D tracking)
        if "neural_net" in self._model_reports and winner != "neural_net":
            nn_report = self._model_reports["neural_net"]
            nn_val_loss = nn_report.get("training_summary", {}).get("final_val_loss")
            if nn_val_loss is not None and winner_value is not None and winner_value > 0:
                self._comparison["nn_gap_to_winner"] = nn_val_loss / winner_value
    
    def get_combined_report(self) -> Dict[str, Any]:
        """Generate combined multi-model report."""
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "mode": self._mode,
            "models": self._model_reports,
            "comparison": self._comparison,
            "meta": {
                "diagnostics_enabled": True,
                "total_models": len(self._model_reports),
                "total_time_seconds": (datetime.now() - self._start_time).total_seconds()
            }
        }
    
    def save(self, path: Optional[str] = None) -> str:
        """Save combined diagnostics to JSON."""
        if path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, DEFAULT_OUTPUT_FILE)
        
        report = self.get_combined_report()
        
        try:
            with open(path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Multi-model diagnostics saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save multi-model diagnostics: {e}")
        
        # Also save to history
        try:
            os.makedirs(DEFAULT_HISTORY_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_path = os.path.join(DEFAULT_HISTORY_DIR, f"compare_models_{timestamp}.json")
            with open(history_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # FIFO pruning (Phase 5 - Session 71)
            _prune_history_fifo()
            
        except Exception as e:
            logger.debug(f"Failed to save to history: {e}")
        
        return path
    
    def get_nn_diagnostic_summary(self) -> Optional[Dict[str, Any]]:
        """
        Quick access to NN diagnostics for Strategy Advisor.
        
        Returns None if NN wasn't trained or diagnostics failed.
        """
        if "neural_net" not in self._model_reports:
            return None
        
        nn_report = self._model_reports["neural_net"]
        return {
            "severity": nn_report.get("diagnosis", {}).get("severity", "absent"),
            "issues": nn_report.get("diagnosis", {}).get("issues", []),
            "suggested_fixes": nn_report.get("diagnosis", {}).get("suggested_fixes", []),
            "gap_to_winner": self._comparison.get("nn_gap_to_winner"),
            "dead_neuron_pct": nn_report.get("model_specific", {}).get("gradient_health", {}).get("dead_neuron_pct", 0),
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_diagnostics(path: str = None) -> Optional[Dict[str, Any]]:
    """
    Load diagnostics from JSON file.
    
    Args:
        path: Path to diagnostics file. If None, uses default location.
        
    Returns:
        Diagnostics dict or None if file doesn't exist/is invalid
    """
    if path is None:
        path = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILE)
    
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load diagnostics from {path}: {e}")
    
    return None


def get_severity(path: str = None) -> str:
    """
    Quick check of diagnostics severity.
    
    Returns: 'ok' | 'warning' | 'critical' | 'absent'
    """
    diag = load_diagnostics(path)
    
    if diag is None:
        return "absent"
    
    # Multi-model format
    if "models" in diag:
        # Return worst severity across all models
        severities = []
        for model_report in diag["models"].values():
            sev = model_report.get("diagnosis", {}).get("severity", "absent")
            severities.append(sev)
        
        if "critical" in severities:
            return "critical"
        if "warning" in severities:
            return "warning"
        if "ok" in severities:
            return "ok"
        return "absent"
    
    # Single-model format
    return diag.get("diagnosis", {}).get("severity", "absent")


def _prune_history_fifo(history_dir: str = DEFAULT_HISTORY_DIR, max_files: int = MAX_HISTORY_FILES) -> None:
    """
    FIFO pruning for diagnostics history directory.
    
    Keeps the newest max_files, deletes oldest by mtime.
    Single log line per prune event (not per file).
    
    Args:
        history_dir: Path to history directory
        max_files: Maximum files to keep
    """
    try:
        history_path = Path(history_dir)
        if not history_path.exists() or not history_path.is_dir():
            return
        
        # Get only compare_models history files (not other artifacts)
        files = [(f, f.stat().st_mtime) for f in history_path.glob("compare_models_*.json")]
        
        if len(files) <= max_files:
            return  # Nothing to prune
        
        # Sort by mtime (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Delete oldest files
        to_delete = files[:len(files) - max_files]
        for f, _ in to_delete:
            f.unlink()
        
        logger.info(f"FIFO pruned {len(to_delete)} diagnostics files (kept {max_files})")
        
    except Exception as e:
        logger.debug(f"History pruning failed (non-fatal): {e}")


# =============================================================================
# CLI INTERFACE (for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Diagnostics CLI")
    parser.add_argument("--test-nn", action="store_true", help="Run NN diagnostics test")
    parser.add_argument("--test-tree", action="store_true", help="Run tree diagnostics test")
    parser.add_argument("--test-fifo", action="store_true", help="Test FIFO pruning")
    parser.add_argument("--load", type=str, help="Load and display diagnostics from file")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.load:
        diag = load_diagnostics(args.load)
        if diag:
            print(json.dumps(diag, indent=2))
        else:
            print(f"Failed to load diagnostics from {args.load}")
    
    elif args.test_nn:
        print("Testing NN Diagnostics...")
        try:
            import torch
            import torch.nn as nn
            
            # Create simple model
            model = nn.Sequential(
                nn.Linear(47, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
            # Attach diagnostics
            diag = NNDiagnostics(capture_every_n=1)
            diag.attach(model)
            
            # Simulate training
            for epoch in range(10):
                x = torch.randn(32, 47)
                y = model(x)
                loss = y.sum()
                loss.backward()
                
                diag.on_round_end(epoch, train_loss=0.5 - epoch*0.04, val_loss=0.6 - epoch*0.03)
            
            diag.detach()
            
            # Get report
            report = diag.get_report()
            print(f"Status: {report['status']}")
            print(f"Severity: {report['diagnosis']['severity']}")
            print(f"Issues: {report['diagnosis']['issues']}")
            print(f"Rounds captured: {report['training_summary']['rounds_captured']}")
            
            # Save
            path = diag.save()
            print(f"Saved to: {path}")
            
        except ImportError:
            print("PyTorch not available for NN test")
    
    elif args.test_tree:
        print("Testing Tree Diagnostics...")
        
        diag = CatBoostDiagnostics()
        diag.attach(None)
        
        # Simulate training rounds
        for round_num in range(20):
            diag.on_round_end(round_num, train_loss=0.01 - round_num*0.0004, val_loss=0.012 - round_num*0.0003)
        
        diag.set_feature_importance({
            "intersection_weight": 0.234,
            "bidirectional_selectivity": 0.189,
            "skip_entropy": 0.145,
        })
        diag.set_best_iteration(15)
        diag.detach()
        
        report = diag.get_report()
        print(f"Status: {report['status']}")
        print(f"Severity: {report['diagnosis']['severity']}")
        print(f"Best iteration: {report['model_specific']['best_iteration']}")
        
        path = diag.save()
        print(f"Saved to: {path}")
    
    elif args.test_fifo:
        print("Testing FIFO Pruning...")
        print(f"MAX_HISTORY_FILES = {MAX_HISTORY_FILES}")
        print(f"History dir: {DEFAULT_HISTORY_DIR}")
        
        # Check current state
        history_path = Path(DEFAULT_HISTORY_DIR)
        if history_path.exists():
            files = list(history_path.glob("*.json"))
            print(f"Current files: {len(files)}")
            
            # Run pruning
            _prune_history_fifo()
            
            files_after = list(history_path.glob("*.json"))
            print(f"Files after prune: {len(files_after)}")
        else:
            print(f"History directory does not exist: {DEFAULT_HISTORY_DIR}")
    
    else:
        parser.print_help()
