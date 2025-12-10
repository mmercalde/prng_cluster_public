#!/usr/bin/env python3
"""
Feature Importance Extraction Module (Model-Agnostic)
======================================================

Computes and tracks feature importance for any supported ML model.
Supports: PyTorch Neural Networks, XGBoost, LightGBM, RandomForest, sklearn models.

Design Principle (Addendum G):
    "Model detection belongs ONLY in this module.
     Pipeline scripts NEVER know what model they're using."

This module provides a single entry point: get_feature_importance()
All model-specific logic is encapsulated here.

Author: Distributed PRNG Analysis System
Date: December 9, 2025
Version: 1.1.0 - With Team Beta corrections
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field

# Setup module logger
logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
TORCH_AVAILABLE = False
SKLEARN_AVAILABLE = False
XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis results."""
    
    computation_method: str
    model_version: str
    timestamp: str
    total_features: int
    importance_by_feature: Dict[str, float]
    importance_by_category: Dict[str, float]
    top_10_features: List[Dict[str, Any]]
    bottom_10_features: List[Dict[str, Any]]
    model_type_detected: str = ""  # NEW: Team Beta Correction #4
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "FeatureImportanceResult":
        """Load results from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# FEATURE NAME CONSTANTS
# ============================================================================

STATISTICAL_FEATURES = [
    'score', 'confidence', 'exact_matches', 'total_predictions', 'best_offset',
    'residue_8_match_rate', 'residue_8_coherence', 'residue_8_kl_divergence',
    'residue_125_match_rate', 'residue_125_coherence', 'residue_125_kl_divergence',
    'residue_1000_match_rate', 'residue_1000_coherence', 'residue_1000_kl_divergence',
    'temporal_stability_mean', 'temporal_stability_std', 'temporal_stability_min',
    'temporal_stability_max', 'temporal_stability_trend',
    'pred_mean', 'pred_std', 'actual_mean', 'actual_std',
    'lane_agreement_8', 'lane_agreement_125', 'lane_consistency',
    'skip_entropy', 'skip_mean', 'skip_std', 'skip_range',
    'survivor_velocity', 'velocity_acceleration',
    'intersection_weight', 'survivor_overlap_ratio',
    'forward_count', 'reverse_count', 'intersection_count', 'intersection_ratio',
    'pred_min', 'pred_max',
    'residual_mean', 'residual_std', 'residual_abs_mean', 'residual_max_abs',
    'forward_only_count', 'reverse_only_count'
]

GLOBAL_STATE_FEATURES = [
    'residue_8_entropy', 'residue_125_entropy', 'residue_1000_entropy',
    'power_of_two_bias', 'frequency_bias_ratio', 'suspicious_gap_percentage',
    'regime_change_detected', 'regime_age', 'high_variance_count',
    'marker_390_variance', 'marker_804_variance', 'marker_575_variance',
    'reseed_probability', 'temporal_stability'
]


# ============================================================================
# MODEL-AGNOSTIC ENTRY POINT
# ============================================================================

def get_feature_importance(
    model: Any,
    X: np.ndarray,
    y: Optional[np.ndarray],  # Team Beta Correction #2: Made optional
    feature_names: List[str],
    method: str = 'auto',
    n_repeats: int = 10,
    device: str = 'cuda:0',
    normalize: bool = True  # Team Beta Correction #3: Add normalization option
) -> Dict[str, float]:
    """
    Model-agnostic feature importance extraction.
    
    Automatically detects model type and uses the optimal extraction method:
    - XGBoost/LightGBM/RandomForest: Native feature_importances_ attribute
    - Neural Networks (PyTorch): Permutation or gradient-based importance
    
    Args:
        model: Trained model (any supported type)
        X: Feature matrix for importance calculation (N, F)
        y: Target values (N,) - Optional for gradient method
        feature_names: List of feature names matching X columns
        method: Extraction method
                - 'auto': Detect best method for model type (RECOMMENDED)
                - 'native': Use model's built-in importance (tree models only)
                - 'permutation': Permutation importance (any model)
                - 'gradient': Gradient saliency (neural networks only)
        n_repeats: Number of permutation repeats (if using permutation method)
        device: CUDA device for neural network computation
        normalize: Whether to normalize importance values to sum to 1.0
    
    Returns:
        Dict mapping feature names to importance scores (sorted descending)
    
    Examples:
        # Works with Neural Network (current)
        importance = get_feature_importance(nn_model, X, y, feature_names)
        
        # Works with XGBoost (future) - SAME CALL, NO CHANGES
        importance = get_feature_importance(xgb_model, X, y, feature_names)
        
        # Gradient method (no y needed)
        importance = get_feature_importance(nn_model, X, None, feature_names, method='gradient')
    
    Raises:
        ValueError: If model type not supported or invalid method
        RuntimeError: If extraction fails
    """
    
    # =========================================================================
    # MODEL TYPE DETECTION (Encapsulated here - nowhere else in codebase)
    # Team Beta Correction #4: Log detection path
    # =========================================================================
    
    model_type = _detect_model_type(model)
    logger.info(f"Detected model type: {model_type}")
    
    # Determine method to use
    actual_method = _resolve_method(model_type, method)
    logger.info(f"Using extraction method: {actual_method}")
    
    # =========================================================================
    # EXTRACTION DISPATCH
    # =========================================================================
    
    if actual_method == 'native':
        importance = _extract_native_importance(model, feature_names, normalize)
    
    elif actual_method == 'permutation':
        if y is None:
            raise ValueError("Permutation importance requires y (target values)")
        importance = _extract_permutation_importance(model, X, y, feature_names, n_repeats, model_type, device, normalize)
    
    elif actual_method == 'gradient':
        if model_type != 'pytorch':
            raise ValueError("Gradient method only supported for PyTorch models")
        importance = _extract_gradient_importance(model, X, feature_names, device, normalize)
    
    else:
        raise ValueError(f"Unknown method: {actual_method}")
    
    # Sort by importance descending
    sorted_importance = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
    
    logger.info(f"Extracted importance for {len(sorted_importance)} features")
    logger.info(f"Top 3: {list(sorted_importance.keys())[:3]}")
    
    return sorted_importance


# ============================================================================
# MODEL TYPE DETECTION
# ============================================================================

def _detect_model_type(model: Any) -> str:
    """
    Detect the type of ML model.
    
    Returns:
        One of: 'pytorch', 'xgboost', 'sklearn_tree', 'sklearn_other', 'unknown'
    """
    # Check for PyTorch model
    if _is_pytorch_model(model):
        return 'pytorch'
    
    # Check for XGBoost
    if XGBOOST_AVAILABLE:
        if isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)):
            return 'xgboost'
        if hasattr(model, 'get_booster'):
            return 'xgboost'
    
    # Check for sklearn tree-based models with native importance
    if hasattr(model, 'feature_importances_'):
        return 'sklearn_tree'
    
    # Check for other sklearn models (use permutation)
    if hasattr(model, 'predict'):
        return 'sklearn_other'
    
    return 'unknown'


def _is_pytorch_model(model: Any) -> bool:
    """Check if model is a PyTorch module."""
    if not TORCH_AVAILABLE:
        return False
    
    # Handle DataParallel wrapped models
    if hasattr(model, 'module'):
        return isinstance(model.module, nn.Module)
    
    return isinstance(model, nn.Module)


def _resolve_method(model_type: str, requested_method: str) -> str:
    """
    Resolve the actual method to use based on model type and request.
    
    Args:
        model_type: Detected model type
        requested_method: User-requested method ('auto', 'native', 'permutation', 'gradient')
    
    Returns:
        Actual method to use
    """
    if requested_method == 'auto':
        # Choose best method for model type
        if model_type in ('xgboost', 'sklearn_tree'):
            return 'native'
        elif model_type == 'pytorch':
            return 'permutation'  # More reliable than gradient
        else:
            return 'permutation'
    
    elif requested_method == 'native':
        if model_type not in ('xgboost', 'sklearn_tree'):
            raise ValueError(f"Native importance not available for {model_type}. Use 'permutation' or 'gradient'.")
        return 'native'
    
    elif requested_method == 'gradient':
        if model_type != 'pytorch':
            raise ValueError(f"Gradient method only available for PyTorch models, not {model_type}")
        return 'gradient'
    
    elif requested_method == 'permutation':
        return 'permutation'
    
    else:
        raise ValueError(f"Unknown method: {requested_method}")


# ============================================================================
# EXTRACTION METHODS
# ============================================================================

def _extract_native_importance(
    model: Any, 
    feature_names: List[str],
    normalize: bool = True
) -> Dict[str, float]:
    """
    Extract importance from models with feature_importances_ attribute.
    
    Team Beta Correction #3: Normalize values with zero-check
    """
    # Handle XGBoost Booster object
    if hasattr(model, 'get_booster'):
        return _extract_xgboost_booster_importance(model, feature_names, normalize)
    
    # Standard sklearn-style feature_importances_
    importance = model.feature_importances_
    
    # Team Beta Correction #3: Normalize with zero-check
    if normalize:
        total = sum(importance)
        if total > 0:
            importance = [v / total for v in importance]
        else:
            logger.warning("Total importance is 0, skipping normalization")
    
    result = dict(zip(feature_names, importance))
    return result


def _extract_xgboost_booster_importance(
    model: Any, 
    feature_names: List[str],
    normalize: bool = True
) -> Dict[str, float]:
    """
    Extract importance from XGBoost Booster object.
    
    Team Beta Correction #5: Handle feature name mismatch (f0, f1, etc.)
    """
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type='gain')
    
    # Map to our feature names
    result = {}
    for i, name in enumerate(feature_names):
        # XGBoost uses f0, f1, f2... if no names set
        key = name if name in importance_dict else f'f{i}'
        result[name] = float(importance_dict.get(key, 0.0))
    
    # Team Beta Correction #3: Normalize
    if normalize:
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        else:
            logger.warning("Total importance is 0, skipping normalization")
    
    return result


def _extract_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int,
    model_type: str,
    device: str,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Permutation importance for any model.
    
    For PyTorch models, uses custom GPU-accelerated implementation.
    For sklearn models, uses sklearn's permutation_importance.
    """
    if model_type == 'pytorch':
        return _extract_permutation_importance_pytorch(
            model, X, y, feature_names, n_repeats, device, normalize
        )
    
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn required for permutation importance on non-PyTorch models")
    
    # Use sklearn's permutation_importance
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        scoring='neg_mean_squared_error'
    )
    
    importance = dict(zip(feature_names, result.importances_mean))
    
    # Normalize (use absolute values for comparison)
    if normalize:
        total = sum(abs(v) for v in importance.values())
        if total > 0:
            importance = {k: abs(v) / total for k, v in importance.items()}
    
    return importance


def _extract_permutation_importance_pytorch(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int,
    device: str,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Permutation importance for PyTorch neural networks.
    
    GPU-accelerated implementation.
    """
    model.eval()
    
    # Move data to device
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    
    # Get baseline predictions
    with torch.no_grad():
        baseline_pred = model(X_tensor)
        if baseline_pred.dim() > 1:
            baseline_pred = baseline_pred.squeeze(-1)
        baseline_mse = torch.mean((baseline_pred - y_tensor) ** 2).item()
    
    # Calculate importance for each feature
    importance = {}
    
    for i, feature_name in enumerate(feature_names):
        mse_increases = []
        
        for _ in range(n_repeats):
            # Shuffle feature i
            X_permuted = X_tensor.clone()
            perm_idx = torch.randperm(X_tensor.shape[0], device=device)
            X_permuted[:, i] = X_tensor[perm_idx, i]
            
            # Get predictions with shuffled feature
            with torch.no_grad():
                permuted_pred = model(X_permuted)
                if permuted_pred.dim() > 1:
                    permuted_pred = permuted_pred.squeeze(-1)
                permuted_mse = torch.mean((permuted_pred - y_tensor) ** 2).item()
            
            # Importance = increase in error
            mse_increases.append(permuted_mse - baseline_mse)
        
        importance[feature_name] = float(np.mean(mse_increases))
    
    # Normalize
    if normalize:
        total = sum(abs(v) for v in importance.values())
        if total > 0:
            importance = {k: abs(v) / total for k, v in importance.items()}
    
    return importance


def _extract_gradient_importance(
    model: nn.Module,
    X: np.ndarray,
    feature_names: List[str],
    device: str,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Gradient saliency importance for PyTorch neural networks.
    
    Team Beta Correction #2: This method doesn't require y
    """
    model.eval()
    
    # Move data to device and enable gradients
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
    
    # Forward pass
    output = model(X_tensor)
    
    # Backward pass from sum of outputs
    output.sum().backward()
    
    # Importance = mean absolute gradient per feature
    gradients = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
    
    importance = dict(zip(feature_names, gradients.astype(float)))
    
    # Normalize
    if normalize:
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
    
    return importance


# ============================================================================
# HIGH-LEVEL EXTRACTION WITH FULL RESULTS
# ============================================================================

class FeatureImportanceExtractor:
    """
    Full-featured extractor that produces FeatureImportanceResult objects.
    
    Use get_feature_importance() for simple Dict[str, float] output.
    Use this class when you need full metadata, categories, and history tracking.
    """
    
    # Class-level feature lists for categorization
    STATISTICAL_FEATURES = STATISTICAL_FEATURES
    GLOBAL_STATE_FEATURES = GLOBAL_STATE_FEATURES
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        device: str = 'cuda:0'
    ):
        """
        Initialize extractor.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            device: CUDA device for computation
        """
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model_type = _detect_model_type(model)
        
        logger.info(f"FeatureImportanceExtractor initialized")
        logger.info(f"  Model type: {self.model_type}")
        logger.info(f"  Features: {len(feature_names)}")
        logger.info(f"  Device: {device}")
    
    def extract(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,  # Team Beta Correction #2
        method: str = 'auto',
        model_version: str = 'unknown',
        n_repeats: int = 10
    ) -> FeatureImportanceResult:
        """
        Extract feature importance with full metadata.
        
        Args:
            X: Feature matrix (N, F)
            y: Target values (N,) - Optional for gradient method
            method: 'auto', 'native', 'permutation', or 'gradient'
            model_version: Version string for tracking
            n_repeats: Repeats for permutation method
        
        Returns:
            FeatureImportanceResult with full metadata
        """
        # Get raw importance
        importance = get_feature_importance(
            model=self.model,
            X=X,
            y=y,
            feature_names=self.feature_names,
            method=method,
            n_repeats=n_repeats,
            device=self.device,
            normalize=True
        )
        
        # Categorize features
        importance_by_category = self._compute_category_importance(importance)
        
        # Get top/bottom features
        sorted_features = list(importance.items())
        top_10 = [
            {'name': k, 'importance': v, 'category': self._get_category(k)}
            for k, v in sorted_features[:10]
        ]
        bottom_10 = [
            {'name': k, 'importance': v, 'category': self._get_category(k)}
            for k, v in sorted_features[-10:]
        ]
        
        return FeatureImportanceResult(
            computation_method=_resolve_method(self.model_type, method),
            model_version=model_version,
            timestamp=datetime.now().isoformat(),
            total_features=len(importance),
            importance_by_feature=importance,
            importance_by_category=importance_by_category,
            top_10_features=top_10,
            bottom_10_features=bottom_10,
            model_type_detected=self.model_type  # Team Beta Correction #4
        )
    
    def _compute_category_importance(self, importance: Dict[str, float]) -> Dict[str, float]:
        """Compute total importance by category."""
        statistical_total = sum(
            importance.get(f, 0.0) for f in self.STATISTICAL_FEATURES
        )
        global_total = sum(
            importance.get(f, 0.0) for f in self.GLOBAL_STATE_FEATURES
        )
        
        # Handle features not in either category
        known_features = set(self.STATISTICAL_FEATURES) | set(self.GLOBAL_STATE_FEATURES)
        other_total = sum(
            v for k, v in importance.items() if k not in known_features
        )
        
        result = {
            'statistical_features': statistical_total,
            'global_state_features': global_total
        }
        
        if other_total > 0:
            result['other_features'] = other_total
        
        return result
    
    def _get_category(self, feature_name: str) -> str:
        """Get category for a feature."""
        if feature_name in self.STATISTICAL_FEATURES:
            return 'statistical'
        elif feature_name in self.GLOBAL_STATE_FEATURES:
            return 'global_state'
        else:
            return 'other'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_importance(
    current: Dict[str, float],
    baseline: Dict[str, float],
    threshold: float = 0.15
) -> Dict[str, Any]:
    """
    Compare current importance to baseline for drift detection.
    
    Args:
        current: Current importance dict
        baseline: Baseline importance dict
        threshold: Drift alert threshold
    
    Returns:
        Dict with drift analysis
    """
    delta = {}
    for feature in current:
        curr_val = current.get(feature, 0.0)
        base_val = baseline.get(feature, 0.0)
        delta[feature] = curr_val - base_val
    
    # Calculate overall drift score
    drift_score = sum(abs(v) for v in delta.values()) / len(delta) if delta else 0.0
    
    # Find top gainers and losers
    sorted_delta = sorted(delta.items(), key=lambda x: x[1], reverse=True)
    top_gainers = [(k, v) for k, v in sorted_delta[:5] if v > 0]
    top_losers = [(k, v) for k, v in sorted_delta[-5:] if v < 0]
    
    return {
        'delta': delta,
        'drift_score': drift_score,
        'drift_alert': drift_score > threshold,
        'top_gainers': top_gainers,
        'top_losers': top_losers
    }


def get_importance_summary_for_agent(importance: Dict[str, float]) -> Dict[str, Any]:
    """
    Create a compact summary suitable for agent_metadata injection.
    
    Args:
        importance: Full importance dict
    
    Returns:
        Compact summary dict
    """
    sorted_features = list(importance.items())
    
    # Compute category weights
    stat_weight = sum(
        importance.get(f, 0.0) for f in STATISTICAL_FEATURES
    )
    global_weight = sum(
        importance.get(f, 0.0) for f in GLOBAL_STATE_FEATURES
    )
    
    return {
        'top_features': [f for f, _ in sorted_features[:5]],
        'top_importance': [round(v, 4) for _, v in sorted_features[:5]],
        'statistical_weight': round(stat_weight, 4),
        'global_weight': round(global_weight, 4),
        'total_features': len(importance)
    }


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == '__main__':
    # Quick self-test
    logging.basicConfig(level=logging.INFO)
    
    print("Feature Importance Module v1.1.0")
    print("=" * 50)
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"sklearn available: {SKLEARN_AVAILABLE}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"Statistical features: {len(STATISTICAL_FEATURES)}")
    print(f"Global state features: {len(GLOBAL_STATE_FEATURES)}")
    print("=" * 50)
    print("Module loaded successfully!")
