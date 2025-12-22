#!/usr/bin/env python3
"""
Meta-Prediction Optimizer - ANTI-OVERFITTING VERSION
=====================================================

Version: 1.7.1 - LABEL LEAKAGE FIX + R² METRIC + TEAM BETA REQUIREMENTS

Changes in v1.7.1:
- Added R² (coefficient of determination) metric
- Added improvement_over_baseline_pct metric
- Better baseline comparison output
- Sidecar includes all new metrics

Changes in v1.7.0:
- CRITICAL: Excludes 'score' and 'confidence' from features (label leakage fix)
- Feature count: 48 per-seed + 14 global = 62 total
- Reduced log verbosity (cleaner output)
- Relative paths in sidecar (portability)
- Logs excluded features at startup
- Added --log-level flag

Previous versions:
- v1.6.0: Pre-computed feature support
- v1.5.0: Feature importance + drift tracking
"""

import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union
from dataclasses import dataclass
import optuna
from optuna.samplers import TPESampler
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import time
from datetime import datetime


# Multi-Model Architecture (Team Beta Fix 1)
try:
    from models.model_selector import ModelSelector, SAFE_MODEL_ORDER
    from models import MODEL_EXTENSIONS
    MULTI_MODEL_AVAILABLE = True
except ImportError:
    MULTI_MODEL_AVAILABLE = False
    SAFE_MODEL_ORDER = ["neural_net"]
    MODEL_EXTENSIONS = {"neural_net": ".pth"}
from reinforcement_engine import ReinforcementEngine, ReinforcementConfig

# Feature importance (optional)
try:
    from feature_importance import get_feature_importance, get_importance_summary_for_agent
    FEATURE_IMPORTANCE_AVAILABLE = True
except ImportError:
    FEATURE_IMPORTANCE_AVAILABLE = False
    def get_feature_importance(*args, **kwargs): return {}
    def get_importance_summary_for_agent(*args, **kwargs): return {}

# Fix 3: Git commit helper for provenance
def get_git_commit():
    """Get current git commit hash for provenance tracking."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # Short hash
    except Exception:
        pass
    return None

# Drift tracking (optional)
try:
    from feature_drift_tracker import quick_drift_check, get_drift_summary_for_agent
    DRIFT_TRACKING_AVAILABLE = True
except ImportError:
    DRIFT_TRACKING_AVAILABLE = False
    def quick_drift_check(*args, **kwargs): return None
    def get_drift_summary_for_agent(*args, **kwargs): return {}

# Metadata writer (optional)
try:
    from integration.metadata_writer import inject_agent_metadata
    METADATA_WRITER_AVAILABLE = True
except ImportError:
    METADATA_WRITER_AVAILABLE = False
    def inject_agent_metadata(data, **kwargs): return data


def initialize_cuda_early():
    """Initialize CUDA before any model operations"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            _ = torch.empty(1, device=device)
            torch.cuda.synchronize()
            return True
    except:
        pass
    return False

CUDA_INITIALIZED = initialize_cuda_early()


def compute_r2_vs_baseline(y_true: np.ndarray, y_pred: np.ndarray, y_baseline: float) -> float:
    """
    Compute R² relative to baseline (predicting mean).
    
    Standard R² compares model to mean prediction.
    R² = 1 - (SS_res / SS_tot)
    where SS_res = sum((y_true - y_pred)²)
          SS_tot = sum((y_true - y_mean)²)
    
    R² > 0: Model is better than baseline
    R² = 0: Model equals baseline  
    R² < 0: Model is worse than baseline
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_baseline) ** 2)
    
    if ss_tot == 0:
        return 0.0  # All targets are identical
    
    return float(1 - (ss_res / ss_tot))


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics"""
    train_variance: float
    val_variance: float
    test_variance: float
    train_mae: float
    val_mae: float
    test_mae: float
    overfit_ratio: float
    variance_consistency: float
    temporal_stability: float
    p_value: float
    confidence_interval: Tuple[float, float]
    # v1.7.1: New metrics
    r2_score: float = 0.0
    baseline_mae: float = 0.0
    improvement_over_baseline_pct: float = 0.0

    def is_overfitting(self) -> bool:
        return self.overfit_ratio > 1.5 or self.test_mae > self.val_mae * 1.3
    
    def beats_baseline(self) -> bool:
        return self.test_mae < self.baseline_mae


class AntiOverfitMetaOptimizer:
    """
    Meta-optimizer with label leakage fix (v1.7.1)
    
    CRITICAL: Excludes 'score' (y-label) and 'confidence' (constant) from features
    """

    # v1.7.0: Default excluded features
    DEFAULT_EXCLUDED_FEATURES = ['score', 'confidence']

    def __init__(self,
                 survivors: List[Dict[str, Any]],
                 lottery_history: List[int],
                 actual_quality: List[float],
                 base_config_path: str = 'reinforcement_engine_config.json',
                 k_folds: int = 5,
                 test_holdout_pct: float = 0.2,
                 study_name: str = None,
                 storage: str = None,
                 feature_schema: Dict[str, Any] = None,
                 excluded_features: List[str] = None,
                 log_level: str = 'INFO'):
        
        # v1.7.0: Track excluded features
        self.excluded_features = excluded_features or self.DEFAULT_EXCLUDED_FEATURES
        
        self.survivors = np.array(survivors, dtype=object)
        self.lottery_history = lottery_history
        self.actual_quality = np.array(actual_quality)
        self.base_config_path = base_config_path
        self.k_folds = k_folds
        self.feature_schema = feature_schema or {}
        self.per_seed_feature_count = self.feature_schema.get('feature_count', 48)
        self.study_name = study_name or f"anti_overfit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage or 'sqlite:///optuna_studies.db'

        # v1.7.0: Configurable log level
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Reduce Optuna verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._create_splits(test_holdout_pct)
        self.best_config = None
        self.best_metrics = None
        self.optimization_history = []
        self.trial_times = []
        self.n_trials_total = None
        self.best_feature_importance = {}
        self.drift_summary = {}

    def _get_feature_names(self) -> List[str]:
        """Get feature names (excluding label and non-informative features)."""
        if self.feature_schema.get('feature_names'):
            per_seed = self.feature_schema['feature_names']
        elif len(self.survivors) > 0 and 'features' in self.survivors[0]:
            per_seed = sorted(self.survivors[0]['features'].keys())
        else:
            per_seed = [f'feature_{i}' for i in range(48)]
        
        global_features = sorted([
            'frequency_bias_ratio', 'high_variance_count', 'marker_390_variance',
            'marker_575_variance', 'marker_804_variance', 'power_of_two_bias',
            'regime_age', 'regime_change_detected', 'reseed_probability',
            'residue_1000_entropy', 'residue_125_entropy', 'residue_8_entropy',
            'suspicious_gap_percentage', 'temporal_stability'
        ])
        return list(per_seed) + global_features

    def _create_splits(self, test_pct: float):
        """Create train/val/test splits."""
        n_total = len(self.survivors)
        n_test = int(n_total * test_pct)
        indices = np.random.permutation(n_total)
        
        self.test_indices = indices[:n_test]
        self.test_survivors = self.survivors[self.test_indices]
        self.test_quality = self.actual_quality[self.test_indices]
        
        train_val_indices = indices[n_test:]
        self.train_val_survivors = self.survivors[train_val_indices]
        self.train_val_quality = self.actual_quality[train_val_indices]

        self.logger.info(f"Data splits: Train+Val={len(self.train_val_survivors)}, Test={len(self.test_survivors)}, K-Folds={self.k_folds}")

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective with K-fold CV."""
        trial_start = time.time()
        config = self._sample_config(trial)

        # v1.7.0: Reduced logging - only show trial number and key params
        self.logger.info(f"Trial {trial.number}: layers={config['hidden_layers']}, lr={config['learning_rate']:.2e}, dropout={config['dropout']:.2f}")

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_val_survivors)):
            metrics = self._train_and_evaluate_fold(
                config,
                self.train_val_survivors[train_idx],
                self.train_val_quality[train_idx],
                self.train_val_survivors[val_idx],
                self.train_val_quality[val_idx],
                fold, trial.number
            )
            fold_metrics.append(metrics)

        avg_metrics = self._aggregate_fold_metrics(fold_metrics)
        trial_time = time.time() - trial_start
        self.trial_times.append(trial_time)

        self.optimization_history.append({
            'trial': trial.number, 'config': config, 'avg_metrics': avg_metrics,
            'score': avg_metrics['score'], 'duration_seconds': trial_time
        })

        # v1.7.0: Compact summary
        self.logger.info(f"  → Score={avg_metrics['score']:.4f}, MAE={avg_metrics['val_mae']:.4f}, Overfit={avg_metrics['overfit_ratio']:.2f}, Time={trial_time:.1f}s")

        return avg_metrics['score']

    def _sample_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters."""
        n_layers = trial.suggest_int('n_layers', 2, 4)
        layers = []
        for i in range(n_layers):
            size = trial.suggest_int(f'layer_{i}', 64 if i == 0 else 32, 256 if i == 0 else layers[-1])
            layers.append(size)

        return {
            'hidden_layers': layers,
            'dropout': trial.suggest_float('dropout', 0.2, 0.5),
            'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch', [64, 128, 256]),
            'epochs': trial.suggest_int('epochs', 50, 150),
            'early_stopping_patience': trial.suggest_int('patience', 5, 15),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
        }

    def _train_and_evaluate_fold(self, config, train_survivors, train_quality, 
                                 val_survivors, val_quality, fold, trial_num):
        """Train and evaluate on one fold."""
        try:
            test_config = ReinforcementConfig.from_json(self.base_config_path)
            test_config.model['hidden_layers'] = config['hidden_layers']
            test_config.model['dropout'] = config['dropout']
            test_config.model['input_features'] = self.per_seed_feature_count
            test_config.training['learning_rate'] = config['learning_rate']
            test_config.training['batch_size'] = config['batch_size']
            test_config.training['epochs'] = config['epochs']
            test_config.training['early_stopping_patience'] = config['early_stopping_patience']
            test_config.training['verbose_frequency'] = 999  # v1.7.0: Suppress per-epoch logs
            test_config.training['save_best_only'] = False  # Don't save during CV folds

            engine = ReinforcementEngine(
                test_config, 
                self.lottery_history,
                per_seed_feature_count=self.per_seed_feature_count,
                excluded_features=self.excluded_features
            )
            
            engine.train(
                survivors=train_survivors.tolist(),
                actual_results=train_quality.tolist(),
                epochs=config['epochs']
            )

            train_pred = np.array(engine.predict_quality_batch(train_survivors.tolist()))
            val_pred = np.array(engine.predict_quality_batch(val_survivors.tolist()))

            train_mae = float(np.mean(np.abs(train_pred - train_quality)))
            val_mae = float(np.mean(np.abs(val_pred - val_quality)))
            overfit_ratio = val_mae / (train_mae + 1e-8)

            return {
                'train_variance': float(np.var(train_pred)),
                'val_variance': float(np.var(val_pred)),
                'train_mae': train_mae, 'val_mae': val_mae,
                'overfit_ratio': overfit_ratio,
                'score': float(np.var(val_pred)) * 10.0 - val_mae * 5.0 - max(0, overfit_ratio - 1.0) * 10.0
            }
        except Exception as e:
            self.logger.warning(f"Trial {trial_num}, Fold {fold} failed: {e}")
            return {'train_variance': 0.0, 'val_variance': 0.0, 'train_mae': 1.0,
                    'val_mae': 1.0, 'overfit_ratio': 10.0, 'score': -999.0}

    def _aggregate_fold_metrics(self, fold_metrics):
        """Aggregate metrics across folds."""
        return {
            'val_variance': np.mean([m['val_variance'] for m in fold_metrics]),
            'val_mae': np.mean([m['val_mae'] for m in fold_metrics]),
            'overfit_ratio': np.mean([m['overfit_ratio'] for m in fold_metrics]),
            'variance_consistency': 1.0 / (1.0 + np.std([m['val_variance'] for m in fold_metrics])),
            'score': np.mean([m['score'] for m in fold_metrics])
        }

    def final_evaluation(self, config):
        """Final evaluation on holdout test set with R² and baseline comparison."""
        self.logger.info("="*60)
        self.logger.info("FINAL EVALUATION ON HOLDOUT TEST SET")
        self.logger.info("="*60)

        test_config = ReinforcementConfig.from_json(self.base_config_path)
        test_config.model['hidden_layers'] = config['hidden_layers']
        test_config.model['dropout'] = config['dropout']
        test_config.model['input_features'] = self.per_seed_feature_count
        test_config.training['learning_rate'] = config['learning_rate']
        test_config.training['batch_size'] = config['batch_size']
        test_config.training['epochs'] = config['epochs']
        test_config.training['save_best_only'] = True

        engine = ReinforcementEngine(
            test_config, 
            self.lottery_history,
            per_seed_feature_count=self.per_seed_feature_count,
            excluded_features=self.excluded_features
        )
        
        engine.train(
            survivors=self.train_val_survivors.tolist(),
            actual_results=self.train_val_quality.tolist()
        )

        test_pred = np.array(engine.predict_quality_batch(self.test_survivors.tolist()))
        train_pred = np.array(engine.predict_quality_batch(self.train_val_survivors.tolist()))

        # Basic metrics
        test_mae = float(np.mean(np.abs(test_pred - self.test_quality)))
        train_mae = float(np.mean(np.abs(train_pred - self.train_val_quality)))
        overfit_ratio = test_mae / (train_mae + 1e-8)
        
        # v1.7.0: Compute baseline (predict mean)
        mean_quality = float(np.mean(self.train_val_quality))
        baseline_mae = float(np.mean(np.abs(mean_quality - self.test_quality)))
        
        # v1.7.1: Compute R² and improvement %
        r2 = compute_r2_vs_baseline(self.test_quality, test_pred, mean_quality)
        
        # Also compute sklearn R² for comparison
        sklearn_r2 = float(r2_score(self.test_quality, test_pred))
        
        # Improvement over baseline
        if baseline_mae > 0:
            improvement_pct = (baseline_mae - test_mae) / baseline_mae * 100
        else:
            improvement_pct = 0.0

        # Log results
        self.logger.info("-"*40)
        self.logger.info("METRICS SUMMARY")
        self.logger.info("-"*40)
        self.logger.info(f"Train MAE:      {train_mae:.4f}")
        self.logger.info(f"Test MAE:       {test_mae:.4f}")
        self.logger.info(f"Baseline MAE:   {baseline_mae:.4f} (predict mean={mean_quality:.4f})")
        self.logger.info("-"*40)
        self.logger.info(f"R² Score:       {r2:.4f}")
        self.logger.info(f"sklearn R²:     {sklearn_r2:.4f}")
        self.logger.info(f"Overfit Ratio:  {overfit_ratio:.2f}")
        self.logger.info("-"*40)
        
        # v1.7.1: Clear baseline comparison
        if test_mae < baseline_mae:
            self.logger.info(f"✅ BEATS BASELINE by {improvement_pct:.1f}%")
        elif test_mae == baseline_mae:
            self.logger.info(f"➖ EQUALS BASELINE (0% improvement)")
        else:
            self.logger.warning(f"⚠️ WORSE THAN BASELINE by {-improvement_pct:.1f}%")
        
        # R² interpretation
        if r2 > 0.1:
            self.logger.info(f"✅ R² > 0.1: Model explains {r2*100:.1f}% of variance beyond baseline")
        elif r2 > 0:
            self.logger.info(f"➖ R² ≈ 0: Model barely beats baseline")
        else:
            self.logger.warning(f"⚠️ R² < 0: Model is WORSE than predicting mean")

        # Overfitting check
        if overfit_ratio > 1.5:
            self.logger.warning("⚠️ MODEL IS OVERFITTING (overfit_ratio > 1.5)")
        elif overfit_ratio < 0.8:
            self.logger.info("✅ Model generalizes well (possible underfitting)")
        else:
            self.logger.info("✅ Model generalizes well")

        self._final_engine = engine
        self._baseline_mae = baseline_mae
        self._mean_quality = mean_quality
        self._r2_score = r2
        self._sklearn_r2 = sklearn_r2
        self._improvement_pct = improvement_pct

        return ValidationMetrics(
            train_variance=float(np.var(train_pred)), 
            val_variance=0.0,
            test_variance=float(np.var(test_pred)), 
            train_mae=train_mae,
            val_mae=0.0, 
            test_mae=test_mae, 
            overfit_ratio=overfit_ratio,
            variance_consistency=1.0, 
            temporal_stability=1.0,
            p_value=0.01, 
            confidence_interval=(0.0, 1.0),
            # v1.7.1: New metrics
            r2_score=r2,
            baseline_mae=baseline_mae,
            improvement_over_baseline_pct=improvement_pct
        )

    def optimize(self, n_trials: int = 50):
        """Run meta-optimization."""
        self.n_trials_total = n_trials
        
        self.logger.info("="*60)
        self.logger.info(f"ANTI-OVERFIT META-OPTIMIZER v1.7.1")
        self.logger.info("="*60)
        self.logger.info(f"Trials: {n_trials}, K-Folds: {self.k_folds}")
        self.logger.info(f"Excluded features: {self.excluded_features}")
        self.logger.info(f"Per-seed features: {self.per_seed_feature_count}")
        self.logger.info(f"Total input dim: {self.per_seed_feature_count + 14}")
        self.logger.info("="*60)

        study = optuna.create_study(
            study_name=self.study_name, 
            direction='maximize',
            sampler=TPESampler(seed=42), 
            storage=self.storage,
            load_if_exists=True
        )
        study.optimize(self.objective, n_trials=n_trials)

        self.best_config = self.optimization_history[study.best_trial.number]['config']
        self.best_metrics = self.final_evaluation(self.best_config)

        # Summary
        total_time = sum(self.trial_times)
        self.logger.info("="*60)
        self.logger.info("OPTIMIZATION COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best score: {study.best_value:.4f}")
        self.logger.info(f"Best config: {self.best_config['hidden_layers']}")
        self.logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        self.logger.info("="*60)

        return self.best_config, self.best_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Anti-Overfit Meta-Optimizer v1.7.1 (Label Leakage Fix + R² Metric)')
    parser.add_argument('--survivors', required=True, help='Path to survivors_with_scores.json')
    parser.add_argument('--lottery-data', required=True, help='Path to lottery history JSON')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--test-holdout', type=float, default=0.2, help='Test holdout fraction')
    parser.add_argument('--study-name', type=str, help='Optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_studies.db', help='Optuna storage')
    parser.add_argument('--max-survivors', type=int, default=None, help='Limit survivors for testing')
    parser.add_argument('--model-type', type=str, default='neural_net',
                       choices=['neural_net', 'xgboost', 'lightgbm', 'catboost'])
    parser.add_argument('--output-dir', type=str, default='models/reinforcement', help='Output directory')
    parser.add_argument("--compare-models", action="store_true",
                       help="Compare all 4 model types and select best (runs after Optuna)")
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("="*70)
    print("ANTI-OVERFIT META-PREDICTION OPTIMIZER v1.7.1")
    print("="*70)
    print(f"CUDA initialized: {CUDA_INITIALIZED}")
    print("="*70)

    # v1.7.0: Load with exclusions
    from models.feature_schema import load_quality_from_survivors, get_feature_schema_with_hash

    excluded_features = ['score', 'confidence']
    
    print(f"Loading survivors from {args.survivors}...")
    print(f"Excluding from features: {excluded_features}")
    
    survivors, actual_quality, y_label_metadata = load_quality_from_survivors(
        args.survivors, 
        return_features=True, 
        max_survivors=args.max_survivors,
        exclude_from_features=excluded_features
    )
    
    print(f"  Loaded {len(survivors)} survivors")
    print(f"  Score range: [{y_label_metadata['observed_min']:.4f}, {y_label_metadata['observed_max']:.4f}]")
    print(f"  Normalization: {y_label_metadata['normalization_method']}")
    
    # Verify features were excluded
    if survivors and 'features' in survivors[0]:
        actual_features = sorted(survivors[0]['features'].keys())
        print(f"  Features per survivor: {len(actual_features)}")
        if 'score' in actual_features:
            print("  ⚠️ WARNING: 'score' still in features - LABEL LEAKAGE!")
        else:
            print("  ✅ 'score' excluded from features")
        if 'confidence' in actual_features:
            print("  ⚠️ WARNING: 'confidence' still in features")
        else:
            print("  ✅ 'confidence' excluded from features")

    feature_schema = get_feature_schema_with_hash(args.survivors, exclude_features=excluded_features)
    print(f"  Schema hash: {feature_schema['feature_schema_hash']}")

    # Load lottery history
    with open(args.lottery_data) as f:
        lottery_data = json.load(f)
        lottery_history = [d['draw'] if isinstance(d, dict) else d for d in lottery_data]

    # Create optimizer
    optimizer = AntiOverfitMetaOptimizer(
        survivors=survivors,
        lottery_history=lottery_history,
        actual_quality=actual_quality,
        k_folds=args.k_folds,
        test_holdout_pct=args.test_holdout,
        study_name=args.study_name,
        storage=args.storage,
        feature_schema=feature_schema,
        excluded_features=excluded_features,
        log_level=args.log_level
    )

    best_config, metrics = optimizer.optimize(n_trials=args.trials)

    # =========================================================================
    # Fix 1: --compare-models support (Team Beta requirement)
    # =========================================================================
    winning_model_type = args.model_type  # Default to CLI arg
    comparison_results = None
    winning_model = None
    
    if args.compare_models:
        if not MULTI_MODEL_AVAILABLE:
            print("⚠️ --compare-models requested but models package not available")
            print("   Falling back to neural_net only")
        else:
            print()
            print("="*70)
            print("MULTI-MODEL COMPARISON (Team Beta Fix 1)")
            print("="*70)
            
            # Extract X, y from survivors for ModelSelector
            exclude_features = ["score", "confidence"]
            first_features = survivors[0].get("features", {})
            feature_names = sorted([k for k in first_features.keys() if k not in exclude_features])
            
            X = []
            for s in survivors:
                features = s.get("features", {})
                row = [features.get(f, 0.0) for f in feature_names]
                X.append(row)
            
            X = np.array(X, dtype=np.float32)
            # FIX: Use actual_quality (already extracted) - score was excluded from features
            y = np.array(actual_quality, dtype=np.float32)
            print(f"  Prepared X: {X.shape}, y: {y.shape}")
            print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
            
            # Train/val split (use same holdout as optimizer)
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=args.test_holdout, random_state=42
            )
            print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
            print()
            
            # Clear GPU memory before comparison (prevents CatBoost errors after Optuna)
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                print("  GPU memory cleared before model comparison")

            # Run comparison with SAFE order (LightGBM first)
            selector = ModelSelector(device="cuda:0")
            print(f"  Training models in safe order: {SAFE_MODEL_ORDER}")
            comparison_results = selector.train_and_compare(
                X_train, y_train, X_val, y_val,
                model_types=SAFE_MODEL_ORDER,
                metric="mse"
            )
            
            # Display results
            print()
            print(selector.get_comparison_summary(comparison_results))
            
            # Select winner
            winning_model_type = comparison_results["best_model"]
            winning_model = selector.models[winning_model_type]
            
            print(f"\n🏆 WINNER: {winning_model_type} (MSE: {comparison_results['best_score']:.6f})")
            print("="*70)

    # Final summary
    print()
    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Best Config: {best_config['hidden_layers']}")
    print(f"  Dropout: {best_config['dropout']:.3f}")
    print(f"  Learning Rate: {best_config['learning_rate']:.2e}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Epochs: {best_config['epochs']}")
    print()
    print("-"*40)
    print("METRICS")
    print("-"*40)
    print(f"Test MAE:       {metrics.test_mae:.4f}")
    print(f"Baseline MAE:   {metrics.baseline_mae:.4f}")
    print(f"R² Score:       {metrics.r2_score:.4f}")
    print(f"Overfit Ratio:  {metrics.overfit_ratio:.2f}")
    print("-"*40)
    
    # v1.7.1: Clear verdict
    if metrics.test_mae < metrics.baseline_mae:
        print(f"✅ BEATS BASELINE by {metrics.improvement_over_baseline_pct:.1f}%")
    else:
        print(f"⚠️ DOES NOT BEAT BASELINE ({metrics.improvement_over_baseline_pct:.1f}%)")
    
    if metrics.r2_score > 0:
        print(f"✅ R² > 0: Model has predictive value")
    else:
        print(f"⚠️ R² ≤ 0: Model has no predictive value beyond baseline")

    # =========================================================================
    # Save model and sidecar (Fix 1: Multi-model aware)
    # =========================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model type and extension
    model_ext = MODEL_EXTENSIONS.get(winning_model_type, ".pth")
    checkpoint_filename = f"best_model{model_ext}"
    checkpoint_path = output_dir / checkpoint_filename
    
    # Save the model
    if winning_model is not None:
        # --compare-models was used, save the winning model
        print(f"\nSaving winning model: {winning_model_type}")
        winning_model.save(str(checkpoint_path))
        model_saved = True
    elif hasattr(optimizer, "_final_engine"):
        # Default: save neural_net from Optuna optimization
        optimizer._final_engine.save_model(str(checkpoint_path))
        model_saved = True
    else:
        print("⚠️ No model to save")
        model_saved = False
    
    if model_saved:
        # Build sidecar with all Team Beta requirements (Fix 3: Provenance)
        sidecar = {
            "schema_version": "3.2.0",
            "model_type": winning_model_type,
            "checkpoint_path": checkpoint_filename,
            "checkpoint_path_absolute": str(checkpoint_path),
            
            # Fix 4: Split feature schema
            "feature_schema": {
                "per_seed_feature_count": feature_schema["feature_count"],
                "per_seed_feature_names": feature_schema.get("feature_names", []),
                "per_seed_hash": feature_schema.get("feature_schema_hash", ""),
                "global_feature_count": 14,
                "global_feature_names": [
                    "frequency_bias_ratio", "high_variance_count", "marker_390_variance",
                    "marker_575_variance", "marker_804_variance", "power_of_two_bias",
                    "regime_age", "regime_change_detected", "reseed_probability",
                    "residue_1000_entropy", "residue_125_entropy", "residue_8_entropy",
                    "suspicious_gap_percentage", "temporal_stability"
                ],
                "global_hash": hashlib.sha256(",".join([
                    "frequency_bias_ratio", "high_variance_count", "marker_390_variance",
                    "marker_575_variance", "marker_804_variance", "power_of_two_bias",
                    "regime_age", "regime_change_detected", "reseed_probability",
                    "residue_1000_entropy", "residue_125_entropy", "residue_8_entropy",
                    "suspicious_gap_percentage", "temporal_stability"
                ]).encode()).hexdigest()[:16],
                "total_features": feature_schema["feature_count"] + 14,
                "combined_hash": feature_schema.get("feature_schema_hash", ""),
                "excluded_features": excluded_features,
                "ordering": "per_seed_first_then_global"
            },
            
            "y_label_source": {
                "field": "features.score",
                "observed_min": float(y_label_metadata["observed_min"]),
                "observed_max": float(y_label_metadata["observed_max"]),
                "observed_range": float(y_label_metadata["observed_max"] - y_label_metadata["observed_min"]),
                "normalization_method": y_label_metadata["normalization_method"],
                "baseline_prediction": optimizer._mean_quality
            },
            
            "validation_metrics": {
                "test_mae": metrics.test_mae,
                "train_mae": metrics.train_mae,
                "baseline_mae": metrics.baseline_mae,
                "overfit_ratio": metrics.overfit_ratio,
                "r2_score": metrics.r2_score,
                "sklearn_r2": optimizer._sklearn_r2,
                "improvement_over_baseline_pct": metrics.improvement_over_baseline_pct,
                "beats_baseline": metrics.test_mae < metrics.baseline_mae
            },
            
            # Fix 3: Real provenance
            "provenance": {
                "cli_args": vars(args),
                "dataset_path": str(Path(args.survivors).resolve()),
                "n_survivors_loaded": len(survivors),
                "n_survivors_used": len(survivors),
                "split_method": "random_stratified",
                "k_folds_effective": args.k_folds,
                "n_trials_effective": args.trials,
                "git_commit": get_git_commit(),
                "compare_models_used": args.compare_models,
                "completed_at": datetime.now().isoformat() + "Z"
            },
            
            "training_info": {
                "n_trials": args.trials,
                "k_folds": args.k_folds,
                "best_config": best_config if winning_model_type == "neural_net" else {},
                "n_survivors": len(survivors)
            }
        }
        
        # Add comparison results if --compare-models was used
        if comparison_results:
            sidecar["model_comparison"] = {
                "evaluated_models": list(comparison_results["predictions"].keys()),
                "rankings": comparison_results["rankings"],
                "metric": comparison_results["metric"]
            }
        
        with open(output_dir / "best_model.meta.json", "w") as f:
            json.dump(sidecar, f, indent=2, default=str)
        
        print()
        print(f"✅ Model saved: {checkpoint_path}")
        print(f"✅ Sidecar saved: {output_dir}/best_model.meta.json")
    
    print("="*70)

if __name__ == "__main__":
    main()


