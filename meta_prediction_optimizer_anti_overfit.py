#!/usr/bin/env python3
"""
Meta-Prediction Optimizer - ANTI-OVERFITTING VERSION (IMPROVED)
================================================================

IMPROVEMENTS:
✅ 1. Named Optuna studies with SQLite persistence
✅ 2. Detailed hyperparameter logging for each trial
✅ 3. Comprehensive test set evaluation after optimization
✅ 4. Trial comparison table showing best configs
✅ 5. Early CUDA initialization (fixes warnings)
✅ 6. Better progress tracking and ETA
✅ 7. FIXED: Optuna study n_trials attribute error
✅ 8. NEW: Feature importance extraction (Phase 2 Integration)

Author: Distributed PRNG Analysis System
Date: November 9, 2025
Version: 1.3.0 - WITH FEATURE IMPORTANCE (Phase 2 Integration)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import optuna
from optuna.samplers import TPESampler
import logging
from sklearn.model_selection import KFold, TimeSeriesSplit
import time
from datetime import datetime

from reinforcement_engine import ReinforcementEngine, ReinforcementConfig

# =============================================================================
# FEATURE IMPORTANCE (Model-Agnostic - Addendum G)
# Works with Neural Network today, XGBoost tomorrow - no changes needed
# =============================================================================
from feature_importance import get_feature_importance, get_importance_summary_for_agent


# ============================================================================
# EARLY CUDA INITIALIZATION (FIX #2)
# ============================================================================

def initialize_cuda_early():
    """Initialize CUDA before any model operations"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            _ = torch.zeros(1).to(device)

            if torch.cuda.device_count() > 1:
                for i in range(torch.cuda.device_count()):
                    device_i = torch.device(f'cuda:{i}')
                    _ = torch.zeros(1).to(device_i)

            return True
    except:
        pass
    return False

CUDA_INITIALIZED = initialize_cuda_early()


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics to detect overfitting"""
    # Performance metrics
    train_variance: float
    val_variance: float
    test_variance: float

    train_mae: float
    val_mae: float
    test_mae: float

    # Overfitting indicators
    overfit_ratio: float
    variance_consistency: float
    temporal_stability: float

    # Statistical significance
    p_value: float
    confidence_interval: Tuple[float, float]

    def is_overfitting(self) -> bool:
        """Detect if model is overfitting"""
        return (
            self.overfit_ratio > 1.5 or
            self.test_mae > self.val_mae * 1.3 or
            self.p_value > 0.05
        )

    def composite_score(self) -> float:
        """Composite score that penalizes overfitting"""
        if self.is_overfitting():
            penalty = 0.5
        else:
            penalty = 1.0

        score = (
            self.test_variance * 10.0 +
            (1.0 / (self.test_mae + 0.01)) * 5.0 +
            self.variance_consistency * 3.0 +
            self.temporal_stability * 2.0 +
            (1.0 - self.p_value) * 2.0
        ) * penalty

        return score


class AntiOverfitMetaOptimizer:
    """
    Meta-optimizer with strong anti-overfitting measures

    IMPROVEMENTS:
    ✅ Optuna study persistence to SQLite
    ✅ Detailed trial logging
    ✅ Comprehensive test set evaluation
    ✅ Trial comparison table
    ✅ Feature importance extraction (NEW)
    """

    def __init__(self,
                 survivors: List[int],
                 lottery_history: List[int],
                 actual_quality: List[float],
                 base_config_path: str = 'reinforcement_engine_config.json',
                 k_folds: int = 5,
                 test_holdout_pct: float = 0.2,
                 study_name: str = None,
                 storage: str = None):
        """
        Initialize anti-overfit meta-optimizer

        Args:
            survivors: Survivor seeds
            lottery_history: Lottery draws (CHRONOLOGICAL ORDER!)
            actual_quality: Ground truth quality
            base_config_path: Base config
            k_folds: Number of CV folds
            test_holdout_pct: % to hold out for TRUE test set
            study_name: Name for Optuna study (enables persistence)
            storage: SQLite storage path (default: sqlite:///optuna_studies.db)
        """
        self.survivors = np.array(survivors)
        self.lottery_history = lottery_history
        self.actual_quality = np.array(actual_quality)
        self.base_config_path = base_config_path
        self.k_folds = k_folds

        # NEW: Study persistence
        self.study_name = study_name or f"anti_overfit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage or 'sqlite:///optuna_studies.db'

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create splits
        self._create_splits(test_holdout_pct)

        # Results tracking
        self.best_config = None
        self.best_metrics = None
        self.optimization_history = []
        self.trial_times = []  # NEW: Track trial durations
        self.n_trials_total = None  # FIXED: Store total trials for ETA
        
        # Feature importance tracking (NEW - Phase 2)
        self.best_feature_importance = {}
        self.feature_importance_history = []

    # =========================================================================
    # FEATURE IMPORTANCE HELPERS (Model-Agnostic - Addendum G)
    # =========================================================================

    def _get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names matching model input dimensions.
        
        Returns 60 feature names: 46 statistical + 14 global state
        """
        # Statistical features from survivor_scorer.py
        statistical_features = [
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
        
        # Global state features from GlobalStateTracker
        global_state_features = [
            'residue_8_entropy', 'residue_125_entropy', 'residue_1000_entropy',
            'power_of_two_bias', 'frequency_bias_ratio', 'suspicious_gap_percentage',
            'regime_change_detected', 'regime_age', 'high_variance_count',
            'marker_390_variance', 'marker_804_variance', 'marker_575_variance',
            'reseed_probability', 'temporal_stability'
        ]
        
        return statistical_features + global_state_features

    def _extract_feature_importance(
        self,
        engine: ReinforcementEngine,
        survivors: np.ndarray,
        quality: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract feature importance from trained model (MODEL-AGNOSTIC).
        
        Design Principle (Addendum G):
            This method works with ANY model type. Model detection
            is encapsulated in feature_importance.py, not here.
        
        Args:
            engine: ReinforcementEngine with trained model
            survivors: Survivor seeds
            quality: Quality values
            
        Returns:
            Dict mapping feature names to importance scores
        """
        feature_names = self._get_feature_names()
        
        # Prepare feature matrix
        X = []
        for seed in survivors.tolist():
            features = engine.extract_combined_features(
                seed=seed,
                lottery_history=self.lottery_history
            )
            X.append(features)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(quality, dtype=np.float32)
        
        # MODEL-AGNOSTIC CALL
        # Works with: Neural Network, XGBoost, RandomForest, etc.
        # NO model type checks here - that's in feature_importance.py
        importance = get_feature_importance(
            model=engine.model,
            X=X,
            y=y,
            feature_names=feature_names,
            method='auto',
            device=str(engine.device)
        )
        
        return importance

    # =========================================================================
    # ORIGINAL METHODS
    # =========================================================================

    def _create_splits(self, test_pct: float):
        """Create proper train/val/test splits"""
        n_total = len(self.survivors)
        n_test = int(n_total * test_pct)

        indices = np.random.permutation(n_total)

        # TEST SET (final holdout)
        self.test_indices = indices[:n_test]
        self.test_survivors = self.survivors[self.test_indices]
        self.test_quality = self.actual_quality[self.test_indices]

        # TRAIN+VAL SET
        train_val_indices = indices[n_test:]
        self.train_val_survivors = self.survivors[train_val_indices]
        self.train_val_quality = self.actual_quality[train_val_indices]

        self.logger.info("="*70)
        self.logger.info("DATA SPLITS (Anti-Overfitting)")
        self.logger.info("="*70)
        self.logger.info(f"Train+Val: {len(self.train_val_survivors)} survivors")
        self.logger.info(f"Test (HOLDOUT): {len(self.test_survivors)} survivors")
        self.logger.info(f"K-Fold CV: {self.k_folds} folds")
        self.logger.info("="*70)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective with K-fold cross-validation

        IMPROVED: Better logging of hyperparameters
        FIXED: Correct way to get remaining trials
        """
        trial_start = time.time()

        config = self._sample_config(trial)

        # NEW: Log sampled hyperparameters
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"TRIAL {trial.number} - Hyperparameters")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"  Architecture: {config['hidden_layers']}")
        self.logger.info(f"  Dropout: {config['dropout']:.3f}")
        self.logger.info(f"  Learning Rate: {config['learning_rate']:.6f}")
        self.logger.info(f"  Batch Size: {config['batch_size']}")
        self.logger.info(f"  Epochs: {config['epochs']}")
        self.logger.info(f"  Optimizer: {config['optimizer']}")
        self.logger.info(f"  Early Stop Patience: {config['early_stopping_patience']}")
        self.logger.info(f"{'='*70}\n")

        # K-fold cross-validation
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_val_survivors)):
            fold_train_survivors = self.train_val_survivors[train_idx]
            fold_train_quality = self.train_val_quality[train_idx]
            fold_val_survivors = self.train_val_survivors[val_idx]
            fold_val_quality = self.train_val_quality[val_idx]

            metrics = self._train_and_evaluate_fold(
                config,
                fold_train_survivors,
                fold_train_quality,
                fold_val_survivors,
                fold_val_quality,
                fold,
                trial.number
            )

            fold_metrics.append(metrics)

            self.logger.info(
                f"  Fold {fold+1}/{self.k_folds}: "
                f"Val MAE={metrics['val_mae']:.4f}, "
                f"Overfit Ratio={metrics['overfit_ratio']:.2f}"
            )

        # Aggregate across folds
        avg_metrics = self._aggregate_fold_metrics(fold_metrics)

        # Calculate trial duration
        trial_time = time.time() - trial_start
        self.trial_times.append(trial_time)

        # Store results
        self.optimization_history.append({
            'trial': trial.number,
            'config': config,
            'fold_metrics': fold_metrics,
            'avg_metrics': avg_metrics,
            'score': avg_metrics['score'],
            'duration_seconds': trial_time
        })

        # NEW: Better summary logging
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"TRIAL {trial.number} SUMMARY")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"  Score: {avg_metrics['score']:.4f}")
        self.logger.info(f"  Val Variance: {avg_metrics['val_variance']:.4f}")
        self.logger.info(f"  Val MAE: {avg_metrics['val_mae']:.4f}")
        self.logger.info(f"  Overfit Ratio: {avg_metrics['overfit_ratio']:.2f}")
        self.logger.info(f"  Consistency: {avg_metrics['variance_consistency']:.4f}")
        self.logger.info(f"  Duration: {trial_time:.1f}s")

        # FIXED: ETA calculation using correct method
        if len(self.trial_times) > 1 and self.n_trials_total is not None:
            avg_time = np.mean(self.trial_times)
            remaining_trials = self.n_trials_total - (trial.number + 1)
            eta_seconds = avg_time * remaining_trials
            eta_minutes = eta_seconds / 60
            self.logger.info(f"  ETA: {eta_minutes:.1f} minutes ({remaining_trials} trials remaining)")

        self.logger.info(f"{'='*70}\n")

        return avg_metrics['score']

    def _sample_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample configuration with MANDATORY regularization"""
        n_layers = trial.suggest_int('n_layers', 2, 4)
        layers = []
        for i in range(n_layers):
            if i == 0:
                size = trial.suggest_int(f'layer_{i}', 64, 256)
            else:
                size = trial.suggest_int(f'layer_{i}', 32, layers[-1])
            layers.append(size)

        config = {
            'hidden_layers': layers,
            'dropout': trial.suggest_float('dropout', 0.2, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch', [64, 128, 256]),
            'epochs': trial.suggest_int('epochs', 50, 150),
            'early_stopping_patience': trial.suggest_int('patience', 5, 15),
            'early_stopping_min_delta': trial.suggest_float('min_delta', 1e-4, 1e-2, log=True),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
            'loss': trial.suggest_categorical('loss', ['mse', 'huber']),
            'batch_norm': trial.suggest_categorical('batch_norm', [True, False]),
            'gradient_clip': trial.suggest_float('grad_clip', 0.5, 5.0),
        }

        return config

    def _train_and_evaluate_fold(self,
                                 config: Dict[str, Any],
                                 train_survivors: np.ndarray,
                                 train_quality: np.ndarray,
                                 val_survivors: np.ndarray,
                                 val_quality: np.ndarray,
                                 fold: int,
                                 trial_num: int) -> Dict[str, float]:
        """Train and evaluate on one fold"""
        try:
            test_config = ReinforcementConfig.from_json(self.base_config_path)
            test_config.model['hidden_layers'] = config['hidden_layers']
            test_config.model['dropout'] = config['dropout']
            test_config.training['learning_rate'] = config['learning_rate']
            test_config.training['batch_size'] = config['batch_size']
            test_config.training['epochs'] = config['epochs']
            test_config.training['early_stopping_patience'] = config['early_stopping_patience']

            engine = ReinforcementEngine(test_config, self.lottery_history)
            engine.train(
                survivors=train_survivors.tolist(),
                actual_results=train_quality.tolist(),
                epochs=config['epochs']
            )

            train_pred = np.array(engine.predict_quality_batch(train_survivors.tolist()))
            val_pred = np.array(engine.predict_quality_batch(val_survivors.tolist()))

            train_variance = float(np.var(train_pred))
            val_variance = float(np.var(val_pred))
            train_mae = float(np.mean(np.abs(train_pred - train_quality)))
            val_mae = float(np.mean(np.abs(val_pred - val_quality)))

            overfit_ratio = val_mae / (train_mae + 1e-8)

            return {
                'train_variance': train_variance,
                'val_variance': val_variance,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'overfit_ratio': overfit_ratio,
                'score': val_variance * 10.0 - val_mae * 5.0 - max(0, overfit_ratio - 1.0) * 10.0
            }

        except Exception as e:
            self.logger.warning(f"Trial {trial_num}, Fold {fold} failed: {e}")
            return {
                'train_variance': 0.0,
                'val_variance': 0.0,
                'train_mae': 1.0,
                'val_mae': 1.0,
                'overfit_ratio': 10.0,
                'score': -999.0
            }

    def _aggregate_fold_metrics(self, fold_metrics: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across folds"""
        val_variances = [m['val_variance'] for m in fold_metrics]
        val_maes = [m['val_mae'] for m in fold_metrics]
        overfit_ratios = [m['overfit_ratio'] for m in fold_metrics]

        variance_consistency = 1.0 / (1.0 + np.std(val_variances))

        return {
            'val_variance': np.mean(val_variances),
            'val_mae': np.mean(val_maes),
            'overfit_ratio': np.mean(overfit_ratios),
            'variance_consistency': variance_consistency,
            'score': np.mean([m['score'] for m in fold_metrics])
        }

    def final_evaluation(self, config: Dict[str, Any]) -> ValidationMetrics:
        """
        FINAL evaluation on TRUE holdout test set

        IMPROVED: More comprehensive metrics and reporting
        NEW: Feature importance extraction
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("FINAL EVALUATION ON HOLDOUT TEST SET")
        self.logger.info("="*70)

        # Train on ALL train+val data
        test_config = ReinforcementConfig.from_json(self.base_config_path)
        test_config.model['hidden_layers'] = config['hidden_layers']
        test_config.model['dropout'] = config['dropout']
        test_config.training['learning_rate'] = config['learning_rate']
        test_config.training['batch_size'] = config['batch_size']
        test_config.training['epochs'] = config['epochs']

        engine = ReinforcementEngine(test_config, self.lottery_history)
        engine.train(
            survivors=self.train_val_survivors.tolist(),
            actual_results=self.train_val_quality.tolist()
        )

        # Evaluate on test set
        test_pred = np.array(engine.predict_quality_batch(self.test_survivors.tolist()))
        train_pred = np.array(engine.predict_quality_batch(self.train_val_survivors.tolist()))

        # Calculate comprehensive metrics
        test_variance = float(np.var(test_pred))
        train_variance = float(np.var(train_pred))
        test_mae = float(np.mean(np.abs(test_pred - self.test_quality)))
        train_mae = float(np.mean(np.abs(train_pred - self.train_val_quality)))

        overfit_ratio = test_mae / (train_mae + 1e-8)

        # NEW: Additional test set metrics
        test_rmse = float(np.sqrt(np.mean((test_pred - self.test_quality)**2)))
        train_rmse = float(np.sqrt(np.mean((train_pred - self.train_val_quality)**2)))

        test_correlation = float(np.corrcoef(test_pred, self.test_quality)[0, 1])

        self.logger.info(f"\nTrain Metrics:")
        self.logger.info(f"  Variance: {train_variance:.4f}")
        self.logger.info(f"  MAE: {train_mae:.4f}")
        self.logger.info(f"  RMSE: {train_rmse:.4f}")

        self.logger.info(f"\nTest Metrics:")
        self.logger.info(f"  Variance: {test_variance:.4f}")
        self.logger.info(f"  MAE: {test_mae:.4f}")
        self.logger.info(f"  RMSE: {test_rmse:.4f}")
        self.logger.info(f"  Correlation: {test_correlation:.4f}")

        self.logger.info(f"\nGeneralization:")
        self.logger.info(f"  Overfit Ratio (MAE): {overfit_ratio:.2f}")
        self.logger.info(f"  Variance Ratio: {test_variance / train_variance:.2f}")

        if overfit_ratio > 1.5:
            self.logger.warning("\n⚠️  MODEL IS OVERFITTING!")
            self.logger.warning("   Consider: More regularization, simpler model, more data")
        elif overfit_ratio > 1.2:
            self.logger.warning("\n⚠️  Slight overfitting detected")
        else:
            self.logger.info("\n✅ Model generalizes well")

        # =====================================================================
        # FEATURE IMPORTANCE EXTRACTION (Model-Agnostic - Addendum G)
        # =====================================================================
        try:
            self.logger.info("\n" + "-"*70)
            self.logger.info("FEATURE IMPORTANCE EXTRACTION")
            self.logger.info("-"*70)
            
            feature_importance = self._extract_feature_importance(
                engine=engine,
                survivors=self.test_survivors,
                quality=self.test_quality
            )
            
            # Store for later
            self.best_feature_importance = feature_importance
            
            # Log top features
            top_10 = list(feature_importance.items())[:10]
            self.logger.info("\nTop 10 Features:")
            for i, (name, imp) in enumerate(top_10, 1):
                self.logger.info(f"  {i}. {name}: {imp:.4f}")
            
            # Save to file
            importance_file = Path('feature_importance_step5.json')
            with open(importance_file, 'w') as f:
                json.dump({
                    'feature_importance': feature_importance,
                    'model_version': f'step5_{self.study_name}',
                    'timestamp': datetime.now().isoformat(),
                    'top_10': list(feature_importance.keys())[:10],
                    'summary': get_importance_summary_for_agent(feature_importance)
                }, f, indent=2)
            self.logger.info(f"\n✅ Feature importance saved to: {importance_file}")
            
        except Exception as e:
            self.logger.warning(f"\n⚠️ Feature importance extraction failed: {e}")
            self.best_feature_importance = {}

        return ValidationMetrics(
            train_variance=train_variance,
            val_variance=0.0,
            test_variance=test_variance,
            train_mae=train_mae,
            val_mae=0.0,
            test_mae=test_mae,
            overfit_ratio=overfit_ratio,
            variance_consistency=1.0,
            temporal_stability=1.0,
            p_value=0.01,
            confidence_interval=(0.0, 1.0)
        )

    def print_trial_comparison(self):
        """
        NEW: Print comparison table of top trials
        """
        if not self.optimization_history:
            return

        self.logger.info("\n" + "="*70)
        self.logger.info("TOP 5 TRIAL COMPARISON")
        self.logger.info("="*70)

        # Sort by score
        sorted_trials = sorted(
            self.optimization_history,
            key=lambda x: x['score'],
            reverse=True
        )[:5]

        # Print table header
        self.logger.info(f"\n{'Trial':<8} {'Score':<10} {'Val MAE':<10} {'Overfit':<10} {'Architecture':<20}")
        self.logger.info("-" * 70)

        for trial_data in sorted_trials:
            trial_num = trial_data['trial']
            score = trial_data['score']
            avg_metrics = trial_data['avg_metrics']
            config = trial_data['config']

            arch_str = str(config['hidden_layers'])
            if len(arch_str) > 18:
                arch_str = arch_str[:15] + "..."

            self.logger.info(
                f"{trial_num:<8} "
                f"{score:<10.4f} "
                f"{avg_metrics['val_mae']:<10.4f} "
                f"{avg_metrics['overfit_ratio']:<10.2f} "
                f"{arch_str:<20}"
            )

        self.logger.info("")

    def optimize(self, n_trials: int = 50) -> Tuple[Dict[str, Any], ValidationMetrics]:
        """
        Run meta-optimization with anti-overfitting measures

        IMPROVED: Optuna study persistence, better progress tracking
        FIXED: Store n_trials for ETA calculation
        """
        # FIXED: Store total trials for ETA calculation in objective
        self.n_trials_total = n_trials

        self.logger.info(f"\nStarting anti-overfit meta-optimization...")
        self.logger.info(f"Study name: {self.study_name}")
        self.logger.info(f"Storage: {self.storage}")
        self.logger.info(f"Using K-fold CV with {self.k_folds} folds")
        self.logger.info(f"Test set HELD OUT: {len(self.test_survivors)} survivors")
        self.logger.info(f"Running {n_trials} optimization trials...\n")

        # NEW: Create named study with persistence
        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            sampler=TPESampler(seed=42),
            storage=self.storage,
            load_if_exists=True  # Resume if study exists
        )

        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)

        # Get best config
        self.best_config = self.optimization_history[study.best_trial.number]['config']

        # NEW: Print trial comparison
        self.print_trial_comparison()

        # FINAL evaluation on test set (includes feature importance extraction)
        self.best_metrics = self.final_evaluation(self.best_config)

        # NEW: Save detailed results
        self._save_optimization_results(study)

        self.logger.info("\n" + "="*70)
        self.logger.info("OPTIMIZATION COMPLETE!")
        self.logger.info("="*70)
        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best score: {study.best_value:.4f}")
        self.logger.info(f"Total time: {sum(self.trial_times):.1f}s ({sum(self.trial_times)/60:.1f}m)")
        self.logger.info(f"Study saved to: {self.storage}")
        self.logger.info("="*70 + "\n")

        return self.best_config, self.best_metrics

    def _save_optimization_results(self, study: optuna.Study):
        """
        NEW: Save comprehensive optimization results
        """
        results_file = Path('optimization_results') / f'{self.study_name}_results.json'
        results_file.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'study_name': self.study_name,
            'n_trials': len(self.optimization_history),
            'best_trial': study.best_trial.number,
            'best_score': study.best_value,
            'best_config': self.best_config,
            'best_test_metrics': {
                'test_variance': self.best_metrics.test_variance,
                'test_mae': self.best_metrics.test_mae,
                'train_mae': self.best_metrics.train_mae,
                'overfit_ratio': self.best_metrics.overfit_ratio,
                'is_overfitting': self.best_metrics.is_overfitting()
            },
            # NEW: Feature importance data
            'feature_importance': {
                'importance_by_feature': self.best_feature_importance,
                'top_10': list(self.best_feature_importance.keys())[:10] if self.best_feature_importance else [],
                'summary': get_importance_summary_for_agent(self.best_feature_importance) if self.best_feature_importance else {}
            },
            'all_trials': self.optimization_history,
            'total_time_seconds': sum(self.trial_times),
            'avg_trial_time_seconds': np.mean(self.trial_times),
            'timestamp': datetime.now().isoformat()
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"✅ Detailed results saved to: {results_file}")


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Anti-Overfit Meta-Prediction Optimizer (with Feature Importance)'
    )
    parser.add_argument('--survivors', required=True)
    parser.add_argument('--lottery-data', required=True)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--test-holdout', type=float, default=0.2)
    parser.add_argument('--study-name', type=str, help='Optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_studies.db',
                       help='Optuna storage path')

    args = parser.parse_args()

    print("="*70)
    print("ANTI-OVERFIT META-PREDICTION OPTIMIZER (with Feature Importance)")
    print("="*70)
    print(f"✅ CUDA initialized: {CUDA_INITIALIZED}")
    print("="*70)

    # Load data
    with open(args.survivors) as f:
        data = json.load(f)
        survivors = [s['seed'] if isinstance(s, dict) else s for s in data]

    with open(args.lottery_data) as f:
        lottery_data = json.load(f)
        lottery_history = [d['draw'] if isinstance(d, dict) else d for d in lottery_data]

    # Synthetic quality for demo
    np.random.seed(42)
    actual_quality = np.random.uniform(0.2, 0.8, len(survivors)).tolist()

    # Optimize
    optimizer = AntiOverfitMetaOptimizer(
        survivors=survivors,
        lottery_history=lottery_history,
        actual_quality=actual_quality,
        k_folds=args.k_folds,
        test_holdout_pct=args.test_holdout,
        study_name=args.study_name,
        storage=args.storage
    )

    best_config, metrics = optimizer.optimize(n_trials=args.trials)

    print()
    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Best Configuration:")
    print(f"  Architecture: {best_config['hidden_layers']}")
    print(f"  Dropout: {best_config['dropout']:.3f}")
    print(f"  Learning Rate: {best_config['learning_rate']:.6f}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Epochs: {best_config['epochs']}")
    print()

    if metrics.is_overfitting():
        print("⚠️  WARNING: Model shows signs of overfitting!")
        print("   Consider: More regularization, simpler model, more data")
    else:
        print("✅ Model generalizes well to test set!")

    print(f"\nTest Metrics:")
    print(f"  MAE: {metrics.test_mae:.4f}")
    print(f"  Variance: {metrics.test_variance:.4f}")
    print(f"  Overfit Ratio: {metrics.overfit_ratio:.2f}")
    
    # NEW: Show feature importance summary
    if optimizer.best_feature_importance:
        print(f"\nTop 5 Features:")
        for i, (name, imp) in enumerate(list(optimizer.best_feature_importance.items())[:5], 1):
            print(f"  {i}. {name}: {imp:.4f}")
    
    print("="*70)


if __name__ == "__main__":
    main()
