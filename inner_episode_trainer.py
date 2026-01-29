#!/usr/bin/env python3
"""
Inner Episode Trainer for Selfplay
===================================

Trains tree models (LightGBM, XGBoost, CatBoost) on CPU for selfplay inner episodes.
Returns PROXY metrics for Optuna optimization - NOT ground truth hit rate.

Per CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md:
- Selfplay optimizes proxy rewards (R², stability, etc)
- Selfplay NEVER sees ground-truth outcomes
- Output is hypothesis (learned_policy_candidate.json), NOT decision

Author: Team Alpha
Version: 1.0.3
Date: 2026-01-29

Changelog:
- v1.0.3: Cap train_val_gap at 5.0 to prevent numerical instability with near-zero MSE
- v1.0.2: Fixed nested features loading for survivors_with_scores.json format
- v1.0.1: Team Beta improvements (R² floor, NaN guard, OMP_NUM_THREADS safety)
- v1.0.0: Initial implementation
"""

import numpy as np
import json
import time
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ML Libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('inner_episode_trainer')


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for inner episode training."""
    
    # Model selection (NO neural_net - 500,000x worse MSE)
    model_types: List[str] = field(default_factory=lambda: ['lightgbm', 'xgboost', 'catboost'])
    
    # Training parameters
    n_estimators: int = 100
    k_folds: int = 3  # Reduced for selfplay speed
    
    # Thread allocation (per worker)
    n_jobs: int = 3  # Default for rigs (i5-9400/8400 have 6 cores, 2 workers)
    
    # Validation
    test_size: float = 0.2
    random_state: int = 42
    
    # Timeout (seconds)
    timeout: float = 60.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProxyMetrics:
    """
    Proxy metrics for selfplay optimization.
    
    CRITICAL: These are PROXY rewards, NOT ground-truth hit rate.
    Per contract, selfplay learns from these signals, not real outcomes.
    
    NOTE on train_val_gap:
    When targets are highly predictable (MSE near machine precision),
    the ratio val_mse/train_mse can explode (e.g., 45x) even when both
    errors are essentially zero. This is numerical instability, not
    real overfitting. The gap is capped at 5.0 to prevent this.
    """
    
    # Core metrics
    val_r2: float = 0.0           # Validation R² (generalization signal)
    val_mae: float = 0.0          # Validation MAE
    val_mse: float = 0.0          # Validation MSE
    
    # Stability metrics
    fold_std: float = 0.0         # Std dev across K folds (lower = more stable)
    train_val_gap: float = 0.0    # Overfit ratio (closer to 1.0 = better)
    
    # Consistency metrics
    prediction_std: float = 0.0   # Std of predictions (spread)
    prediction_range: float = 0.0 # Max - min prediction
    
    # Training info
    model_type: str = ''
    training_time_ms: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    
    # Feature importance (top 5)
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        # Convert tuples to list for JSON serialization
        result['top_features'] = [(k, float(v)) for k, v in self.top_features]
        return result
    
    @property
    def fitness(self) -> float:
        """
        Combined fitness score for Optuna optimization.
        Higher is better.
        
        Balances:
        - Generalization (R²)
        - Stability (low fold_std)
        - No overfitting (train_val_gap close to 1.0)
        """
        # R² is primary signal (0 to 1, higher better)
        # Floor of 0.05 forces minimal explanatory signal before stability can help
        r2_component = max(0, self.val_r2 - 0.05)
        
        # Stability penalty (0 to 1, lower fold_std better)
        stability_penalty = min(1.0, self.fold_std * 10)
        
        # Overfit penalty (1.0 is perfect, >1.5 is bad)
        overfit_penalty = max(0, (self.train_val_gap - 1.0) * 0.5)
        
        # Combined fitness
        fitness = r2_component - 0.2 * stability_penalty - 0.3 * overfit_penalty
        
        return fitness


@dataclass 
class TrainingResult:
    """Result from training a single model."""
    
    success: bool
    model_type: str
    metrics: Optional[ProxyMetrics] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'model_type': self.model_type,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'error': self.error
        }


# =============================================================================
# Model Builders
# =============================================================================

def build_lightgbm(config: TrainerConfig) -> lgb.LGBMRegressor:
    """Build LightGBM regressor for CPU training."""
    return lgb.LGBMRegressor(
        n_estimators=config.n_estimators,
        n_jobs=config.n_jobs,
        device='cpu',  # NEVER 'gpu' on rigs (8-11x slower)
        verbose=-1,
        random_state=config.random_state,
        # Fast defaults for selfplay
        max_bin=63,
        num_leaves=31,
        learning_rate=0.1,
    )


def build_xgboost(config: TrainerConfig) -> xgb.XGBRegressor:
    """Build XGBoost regressor for CPU training."""
    return xgb.XGBRegressor(
        n_estimators=config.n_estimators,
        n_jobs=config.n_jobs,
        tree_method='hist',  # CPU optimized
        verbosity=0,
        random_state=config.random_state,
        # Fast defaults for selfplay
        max_depth=6,
        learning_rate=0.1,
    )


def build_catboost(config: TrainerConfig) -> CatBoostRegressor:
    """Build CatBoost regressor for CPU training."""
    return CatBoostRegressor(
        iterations=config.n_estimators,
        thread_count=config.n_jobs,
        verbose=0,
        random_seed=config.random_state,
        # Fast defaults for selfplay
        depth=6,
        learning_rate=0.1,
    )


MODEL_BUILDERS = {
    'lightgbm': build_lightgbm,
    'xgboost': build_xgboost,
    'catboost': build_catboost,
}


# =============================================================================
# Training Functions
# =============================================================================

def train_single_fold(
    model_type: str,
    config: TrainerConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Tuple[float, float, float, float, np.ndarray, Optional[np.ndarray]]:
    """
    Train single fold and return metrics.
    
    Returns:
        (train_mse, val_mse, val_mae, val_r2, predictions, feature_importance)
    """
    # Build model
    builder = MODEL_BUILDERS.get(model_type)
    if not builder:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = builder(config)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Metrics
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    # Feature importance
    importance = None
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance()
    
    return train_mse, val_mse, val_mae, val_r2, val_pred, importance


def train_model_kfold(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    config: TrainerConfig,
    feature_names: Optional[List[str]] = None
) -> ProxyMetrics:
    """
    Train model with K-fold cross-validation.
    
    Returns proxy metrics for selfplay optimization.
    """
    import os
    # Prevent OpenMP thread explosion if users forget env vars
    os.environ.setdefault("OMP_NUM_THREADS", str(config.n_jobs))
    
    start_time = time.time()
    
    kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=config.random_state)
    
    fold_metrics = {
        'train_mse': [],
        'val_mse': [],
        'val_mae': [],
        'val_r2': [],
    }
    all_predictions = []
    all_importance = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_mse, val_mse, val_mae, val_r2, predictions, importance = train_single_fold(
            model_type, config, X_train, y_train, X_val, y_val, feature_names
        )
        
        fold_metrics['train_mse'].append(train_mse)
        fold_metrics['val_mse'].append(val_mse)
        fold_metrics['val_mae'].append(val_mae)
        fold_metrics['val_r2'].append(val_r2)
        all_predictions.extend(predictions)
        
        if importance is not None:
            all_importance.append(importance)
    
    # Guard against non-finite values (numerical poison prevention)
    if not np.isfinite(fold_metrics['val_r2']).all():
        raise ValueError("Non-finite R² detected in fold results - check for degenerate data")
    
    # Aggregate metrics
    training_time_ms = (time.time() - start_time) * 1000
    
    all_predictions = np.array(all_predictions)
    
    # Calculate mean importance
    top_features = []
    if all_importance and feature_names:
        mean_importance = np.mean(all_importance, axis=0)
        sorted_idx = np.argsort(mean_importance)[::-1][:5]
        top_features = [(feature_names[i], float(mean_importance[i])) for i in sorted_idx]
    
    # Calculate train/val gap (overfit ratio)
    # Cap at 5.0 to prevent numerical instability when MSE is near machine precision
    # (e.g., XGBoost can achieve train_mse ≈ 0, making ratio explode)
    mean_train_mse = np.mean(fold_metrics['train_mse'])
    mean_val_mse = np.mean(fold_metrics['val_mse'])
    raw_gap = mean_val_mse / (mean_train_mse + 1e-15)
    train_val_gap = min(raw_gap, 5.0)  # Cap to prevent extreme penalties
    
    if raw_gap > 5.0:
        logger.warning(
            f"train_val_gap capped: {raw_gap:.2f} -> 5.0 "
            f"(near-zero MSE causes numerical instability)"
        )
    
    return ProxyMetrics(
        val_r2=float(np.mean(fold_metrics['val_r2'])),
        val_mae=float(np.mean(fold_metrics['val_mae'])),
        val_mse=float(mean_val_mse),
        fold_std=float(np.std(fold_metrics['val_r2'])),
        train_val_gap=float(train_val_gap),
        prediction_std=float(np.std(all_predictions)),
        prediction_range=float(np.max(all_predictions) - np.min(all_predictions)),
        model_type=model_type,
        training_time_ms=training_time_ms,
        n_samples=len(X),
        n_features=X.shape[1],
        top_features=top_features,
    )


# =============================================================================
# Inner Episode Trainer Class
# =============================================================================

class InnerEpisodeTrainer:
    """
    Trains tree models for selfplay inner episodes.
    
    CRITICAL CONTRACT REQUIREMENTS:
    - Uses PROXY rewards only (R², stability, etc)
    - NEVER sees ground-truth outcomes (hit rate)
    - Output is hypothesis, not decision
    - CPU only (GPU is 8-11x slower for tree models)
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        """Initialize trainer with configuration."""
        self.config = config or TrainerConfig()
        self.logger = logging.getLogger('inner_episode_trainer')
        
        # Validate model types (NO neural_net)
        forbidden = set(self.config.model_types) & {'neural_net', 'random_forest'}
        if forbidden:
            raise ValueError(
                f"Forbidden model types for selfplay inner episodes: {forbidden}. "
                f"Neural networks are 500,000x worse on tabular data."
            )
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_type: Optional[str] = None
    ) -> TrainingResult:
        """
        Train a single model and return proxy metrics.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (proxy labels, NOT hit rate)
            feature_names: Optional feature names for importance
            model_type: Model to train (default: first in config.model_types)
        
        Returns:
            TrainingResult with proxy metrics
        """
        model_type = model_type or self.config.model_types[0]
        
        if model_type not in MODEL_BUILDERS:
            return TrainingResult(
                success=False,
                model_type=model_type,
                error=f"Unknown model type: {model_type}"
            )
        
        try:
            metrics = train_model_kfold(
                model_type=model_type,
                X=X,
                y=y,
                config=self.config,
                feature_names=feature_names
            )
            
            self.logger.info(
                f"Trained {model_type}: R²={metrics.val_r2:.4f}, "
                f"MAE={metrics.val_mae:.4f}, time={metrics.training_time_ms:.0f}ms"
            )
            
            return TrainingResult(
                success=True,
                model_type=model_type,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Training failed for {model_type}: {e}")
            return TrainingResult(
                success=False,
                model_type=model_type,
                error=str(e)
            )
    
    def train_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, TrainingResult]:
        """
        Train all configured model types.
        
        Returns:
            Dict mapping model_type -> TrainingResult
        """
        results = {}
        
        for model_type in self.config.model_types:
            results[model_type] = self.train(X, y, feature_names, model_type)
        
        return results
    
    def train_best(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[TrainingResult, Dict[str, TrainingResult]]:
        """
        Train all models and return the best one by fitness score.
        
        Returns:
            (best_result, all_results)
        """
        all_results = self.train_all(X, y, feature_names)
        
        # Find best by fitness
        best_result = None
        best_fitness = float('-inf')
        
        for model_type, result in all_results.items():
            if result.success and result.metrics:
                fitness = result.metrics.fitness
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_result = result
        
        if best_result:
            self.logger.info(
                f"Best model: {best_result.model_type} "
                f"(fitness={best_fitness:.4f})"
            )
        
        return best_result, all_results


# =============================================================================
# Utility Functions
# =============================================================================

def load_survivors_for_training(
    survivors_path: str,
    feature_columns: Optional[List[str]] = None,
    target_column: str = 'score'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load survivors data for training.
    
    Handles two formats:
    1. Flat: features at top level
    2. Nested: features inside 'features' dict (survivors_with_scores.json)
    
    Args:
        survivors_path: Path to survivors JSON file
        feature_columns: List of feature columns to extract
        target_column: Column to use as target (MUST be proxy, not hit rate)
    
    Returns:
        (X, y, feature_names)
    """
    with open(survivors_path, 'r') as f:
        survivors = json.load(f)
    
    if not survivors:
        raise ValueError("No survivors loaded")
    
    # Get sample to detect format
    sample = survivors[0] if isinstance(survivors, list) else list(survivors.values())[0]
    
    # Check if features are nested
    nested_features = 'features' in sample and isinstance(sample['features'], dict)
    
    if nested_features:
        logger.info("Detected nested features format (survivors_with_scores.json)")
        feature_source = sample['features']
    else:
        logger.info("Detected flat features format")
        feature_source = sample
    
    # Auto-detect feature columns if not provided
    if not feature_columns:
        # Exclude known non-feature columns
        exclude_cols = {
            'seed', 'score', 'hit', 'hit_rate', 'holdout_hits', 
            'quality_score', 'features', 'metadata'
        }
        feature_columns = [
            k for k in feature_source.keys()
            if isinstance(feature_source[k], (int, float)) 
            and k not in exclude_cols
        ]
    
    logger.info(f"Using {len(feature_columns)} features")
    
    # Extract features and target
    X = []
    y = []
    skipped = 0
    
    items = survivors if isinstance(survivors, list) else survivors.values()
    for item in items:
        # Get features from nested or flat structure
        if nested_features:
            feat_dict = item.get('features', {})
        else:
            feat_dict = item
        
        # Get target - try multiple locations
        if target_column in item:
            target = float(item[target_column])
        elif nested_features and target_column in feat_dict:
            target = float(feat_dict[target_column])
        else:
            skipped += 1
            continue
        
        # Extract feature values
        features = []
        valid = True
        for col in feature_columns:
            val = feat_dict.get(col)
            if val is None or not isinstance(val, (int, float)):
                valid = False
                break
            features.append(float(val))
        
        if valid:
            X.append(features)
            y.append(target)
        else:
            skipped += 1
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} samples with missing features/target")
    
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    
    # Validate target has variance
    if np.std(y_arr) < 1e-10:
        raise ValueError(
            f"Target '{target_column}' has no variance (all values equal). "
            f"Choose a different target column."
        )
    
    logger.info(f"Loaded {len(X_arr)} samples, target='{target_column}' (std={np.std(y_arr):.4f})")
    
    return X_arr, y_arr, feature_columns


def compute_feature_hash(feature_names: List[str]) -> str:
    """Compute hash of feature schema for validation."""
    schema_str = ','.join(sorted(feature_names))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Inner Episode Trainer for Selfplay',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LightGBM with default settings
  python3 inner_episode_trainer.py --survivors survivors.json --model lightgbm
  
  # Train all models and select best
  python3 inner_episode_trainer.py --survivors survivors.json --compare-all
  
  # Custom thread count (Zeus)
  python3 inner_episode_trainer.py --survivors survivors.json --n-jobs 8
        """
    )
    
    parser.add_argument(
        '--survivors', 
        type=str, 
        required=True,
        help='Path to survivors JSON file'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='lightgbm',
        choices=['lightgbm', 'xgboost', 'catboost'],
        help='Model type to train (default: lightgbm)'
    )
    parser.add_argument(
        '--compare-all', 
        action='store_true',
        help='Train all models and select best'
    )
    parser.add_argument(
        '--n-jobs', 
        type=int, 
        default=3,
        help='Number of threads per model (default: 3 for rigs)'
    )
    parser.add_argument(
        '--k-folds', 
        type=int, 
        default=3,
        help='K-fold splits (default: 3 for selfplay speed)'
    )
    parser.add_argument(
        '--n-estimators', 
        type=int, 
        default=100,
        help='Number of estimators/trees (default: 100)'
    )
    parser.add_argument(
        '--target', 
        type=str, 
        default='score',
        help='Target column (default: score - MUST be proxy, not hit rate)'
    )
    parser.add_argument(
        '--output', 
        type=str,
        help='Output path for results JSON'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"\n{'='*60}")
    print("INNER EPISODE TRAINER")
    print(f"{'='*60}")
    print(f"Survivors: {args.survivors}")
    print(f"Model: {args.model}")
    print(f"Threads: {args.n_jobs}")
    print(f"K-Folds: {args.k_folds}")
    
    X, y, feature_names = load_survivors_for_training(
        args.survivors,
        target_column=args.target
    )
    
    print(f"Samples: {len(X)}")
    print(f"Features: {len(feature_names)}")
    print(f"Feature hash: {compute_feature_hash(feature_names)}")
    print(f"{'='*60}\n")
    
    # Configure trainer
    config = TrainerConfig(
        model_types=['lightgbm', 'xgboost', 'catboost'] if args.compare_all else [args.model],
        n_estimators=args.n_estimators,
        k_folds=args.k_folds,
        n_jobs=args.n_jobs,
    )
    
    trainer = InnerEpisodeTrainer(config)
    
    # Train
    if args.compare_all:
        best_result, all_results = trainer.train_best(X, y, feature_names)
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        for model_type, result in all_results.items():
            if result.success and result.metrics:
                m = result.metrics
                print(f"\n{model_type}:")
                print(f"  R²: {m.val_r2:.4f}")
                print(f"  MAE: {m.val_mae:.6f}")
                print(f"  Fold Std: {m.fold_std:.4f}")
                print(f"  Fitness: {m.fitness:.4f}")
                print(f"  Time: {m.training_time_ms:.0f}ms")
            else:
                print(f"\n{model_type}: FAILED - {result.error}")
        
        if best_result:
            print(f"\n{'='*60}")
            print(f"BEST: {best_result.model_type} (fitness={best_result.metrics.fitness:.4f})")
            print(f"{'='*60}")
            
            # Save output
            if args.output:
                output = {
                    'best': best_result.to_dict(),
                    'all': {k: v.to_dict() for k, v in all_results.items()},
                    'config': config.to_dict(),
                }
                with open(args.output, 'w') as f:
                    json.dump(output, f, indent=2)
                print(f"\nResults saved to: {args.output}")
    
    else:
        result = trainer.train(X, y, feature_names, args.model)
        
        if result.success and result.metrics:
            m = result.metrics
            print(f"\n{'='*60}")
            print(f"RESULT: {result.model_type}")
            print(f"{'='*60}")
            print(f"R²: {m.val_r2:.4f}")
            print(f"MAE: {m.val_mae:.6f}")
            print(f"MSE: {m.val_mse:.6f}")
            print(f"Fold Std: {m.fold_std:.4f}")
            print(f"Overfit Ratio: {m.train_val_gap:.2f}")
            print(f"Fitness: {m.fitness:.4f}")
            print(f"Time: {m.training_time_ms:.0f}ms")
            
            if m.top_features:
                print(f"\nTop Features:")
                for name, importance in m.top_features:
                    print(f"  {name}: {importance:.4f}")
            
            # Save output
            if args.output:
                output = {
                    'result': result.to_dict(),
                    'config': config.to_dict(),
                }
                with open(args.output, 'w') as f:
                    json.dump(output, f, indent=2)
                print(f"\nResults saved to: {args.output}")
        else:
            print(f"\nTRAINING FAILED: {result.error}")
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
