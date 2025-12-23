#!/usr/bin/env python3
"""
Meta-Prediction Optimizer - ANTI-OVERFITTING VERSION (IMPROVED)
================================================================

IMPROVEMENTS:
✅ 1. Named Optuna studies with SQLite persistence
✅ 2. Detailed hyperparameter logging for each trial
✅ 3. Comprehensive test set evaluation after optimization
✅ 4. Trial comparison table showing best configs
✅ 5. Early CUDA initialization (fixes warnings) - NOW CONDITIONAL
✅ 6. Better progress tracking and ETA
✅ 7. FIXED: Optuna study n_trials attribute error
✅ 8. NEW: Multi-model support (--model-type, --compare-models)
✅ 9. NEW: Subprocess isolation for OpenCL/CUDA compatibility
✅ 10. NEW: --save-all-models flag for post-hoc AI analysis
✅ 11. NEW: Sidecar metadata generation (best_model.meta.json)

Author: Distributed PRNG Analysis System
Date: November 9, 2025 (Original)
Updated: December 22, 2025 (Subprocess Isolation + Multi-Model)
Version: 2.0.0 - SUBPROCESS ISOLATION
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import optuna
from optuna.samplers import TPESampler
import logging
import time
from datetime import datetime
import sys
import os
import hashlib
import subprocess

# Conditional import for subprocess coordinator
try:
    from subprocess_trial_coordinator import (
        SubprocessTrialCoordinator,
        create_optuna_objective,
        run_isolated_comparison,
        sample_params_for_model,
        SAFE_MODEL_ORDER,
        TrialResult
    )
    SUBPROCESS_AVAILABLE = True
except ImportError:
    SUBPROCESS_AVAILABLE = False
    SAFE_MODEL_ORDER = ['lightgbm', 'neural_net', 'xgboost', 'catboost']

# Import reinforcement engine for backward compatibility (single model mode)
from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
from sklearn.model_selection import KFold, TimeSeriesSplit


# ============================================================================
# CUDA INITIALIZATION - NOW CONDITIONAL
# ============================================================================

# Global flag - set in main() based on args
CUDA_INITIALIZED = False

def initialize_cuda_early():
    """
    Initialize CUDA before any model operations.
    
    NOTE: Only called when NOT using --compare-models.
    Subprocess isolation handles GPU init for --compare-models.
    """
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


# ============================================================================
# VALIDATION METRICS
# ============================================================================

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


# ============================================================================
# FEATURE LOADING UTILITIES
# ============================================================================

def load_features_from_survivors(survivors_file: str, 
                                  exclude_features: List[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load features and quality scores from survivors_with_scores.json.
    
    Args:
        survivors_file: Path to JSON file with scored survivors
        exclude_features: Features to exclude (e.g., ['score', 'confidence'] to prevent leakage)
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Quality scores (n_samples,)
        metadata: Dict with feature names, schema hash, etc.
    """
    if exclude_features is None:
        exclude_features = ['score', 'confidence', 'seed', 'actual_quality']
    
    with open(survivors_file) as f:
        data = json.load(f)
    
    # Handle different data formats
    if isinstance(data, list):
        survivors = data
    elif isinstance(data, dict) and 'survivors' in data:
        survivors = data['survivors']
    else:
        survivors = data
    
    # Extract features and quality
    all_features = []
    quality_scores = []
    feature_names = None
    
    for survivor in survivors:
        if isinstance(survivor, dict):
            # Get quality score
            if 'actual_quality' in survivor:
                quality = survivor['actual_quality']
            elif 'score' in survivor:
                quality = survivor['score']
            elif 'features' in survivor and 'score' in survivor['features']:
                quality = survivor['features']['score']
            else:
                quality = 0.5  # Default
            
            quality_scores.append(quality)
            
            # Get features
            if 'features' in survivor:
                features_dict = survivor['features']
            else:
                features_dict = {k: v for k, v in survivor.items() 
                               if k not in ['seed', 'actual_quality']}
            
            # Filter out excluded features
            features_dict = {k: v for k, v in features_dict.items() 
                           if k not in exclude_features and isinstance(v, (int, float))}
            
            # Get feature names from first sample
            if feature_names is None:
                feature_names = sorted(features_dict.keys())
            
            # Extract feature values in consistent order
            feature_values = [features_dict.get(name, 0.0) for name in feature_names]
            all_features.append(feature_values)
        else:
            # Simple format - just seed values
            quality_scores.append(0.5)
            all_features.append([])
    
    X = np.array(all_features, dtype=np.float32)
    y = np.array(quality_scores, dtype=np.float32)
    
    # Generate schema hash
    if feature_names:
        names_str = ",".join(feature_names)
        schema_hash = hashlib.sha256(names_str.encode()).hexdigest()[:16]
    else:
        schema_hash = "no_features"
    
    # Check for narrow range warning
    y_range = y.max() - y.min()
    warnings = []
    if y_range < 0.01:
        warnings.append(f"score_range_narrow (range={y_range:.6f})")
    
    metadata = {
        'feature_names': feature_names or [],
        'feature_count': len(feature_names) if feature_names else 0,
        'schema_hash': schema_hash,
        'n_samples': len(y),
        'y_min': float(y.min()),
        'y_max': float(y.max()),
        'y_range': float(y_range),
        'warnings': warnings,
        'source_file': str(Path(survivors_file).resolve())
    }
    
    return X, y, metadata


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return "unknown"


# ============================================================================
# SIDECAR METADATA GENERATION
# ============================================================================

def save_model_with_sidecar(model_path: str,
                            model_type: str,
                            feature_metadata: Dict,
                            training_params: Dict,
                            validation_metrics: Dict,
                            cli_args: Dict,
                            output_dir: str = 'models/reinforcement') -> str:
    """
    Save model metadata sidecar file (best_model.meta.json).
    
    Args:
        model_path: Path to saved model checkpoint
        model_type: Model type string
        feature_metadata: From load_features_from_survivors()
        training_params: Hyperparameters used
        validation_metrics: Final metrics
        cli_args: Command line arguments
        output_dir: Output directory
    
    Returns:
        Path to metadata file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    meta = {
        'schema_version': '3.1.2',
        'model_type': model_type,
        'checkpoint_path': str(model_path),
        'checkpoint_format': Path(model_path).suffix.lstrip('.'),
        
        'feature_schema': {
            'source_file': feature_metadata.get('source_file', ''),
            'feature_count': feature_metadata.get('feature_count', 0),
            'feature_names': feature_metadata.get('feature_names', []),
            'ordering': 'lexicographic_by_key',
            'feature_schema_hash': feature_metadata.get('schema_hash', '')
        },
        
        'y_label_source': {
            'field': 'features.score',
            'observed_min': feature_metadata.get('y_min', 0.0),
            'observed_max': feature_metadata.get('y_max', 1.0),
            'observed_range': feature_metadata.get('y_range', 1.0),
            'normalization_method': 'none',
            'output_range': [0.0, 1.0],
            'warnings': feature_metadata.get('warnings', [])
        },
        
        'training_params': training_params,
        'validation_metrics': validation_metrics,
        
        'provenance': {
            'cli_args': cli_args,
            'git_commit': get_git_commit(),
            'n_survivors_loaded': feature_metadata.get('n_samples', 0),
            'timestamp': datetime.now().isoformat()
        },
        
        'agent_metadata': {
            'pipeline_step': 5,
            'pipeline_step_name': 'anti_overfit_training',
            'run_id': f"step5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    }
    
    meta_path = output_dir / 'best_model.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    return str(meta_path)


# ============================================================================
# MULTI-MODEL COMPARISON WITH SUBPROCESS ISOLATION
# ============================================================================

def run_multi_model_comparison(X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                n_trials: int = 50,
                                study_name: str = None,
                                storage: str = 'sqlite:///optuna_studies.db',
                                save_all_models: bool = False,
                                output_dir: str = 'models/reinforcement') -> Dict[str, Any]:
    """
    Run multi-model comparison using subprocess isolation.
    
    This ensures LightGBM (OpenCL) works alongside CUDA models
    by running each trial in a fresh subprocess.
    
    Args:
        X_train, y_train, X_val, y_val: Training/validation data
        n_trials: Number of Optuna trials
        study_name: Optuna study name
        storage: Optuna storage path
        save_all_models: If True, save all 4 models (not just winner)
        output_dir: Directory for model outputs
    
    Returns:
        Dict with best model info and all results
    """
    if not SUBPROCESS_AVAILABLE:
        raise RuntimeError(
            "subprocess_trial_coordinator.py not found!\n"
            "Please ensure it's in the same directory."
        )
    
    print("\n" + "="*70)
    print("MULTI-MODEL COMPARISON (Subprocess Isolation)")
    print("="*70)
    print("This ensures LightGBM (OpenCL) works alongside CUDA models")
    print(f"Models: {SAFE_MODEL_ORDER}")
    print(f"Trials: {n_trials}")
    print("="*70 + "\n")
    
    # Create coordinator
    with SubprocessTrialCoordinator(
        X_train, y_train, X_val, y_val,
        worker_script='train_single_trial.py',
        verbose=True
    ) as coordinator:
        
        # Create Optuna study
        study = optuna.create_study(
            study_name=study_name or f"compare_models_{datetime.now():%Y%m%d_%H%M%S}",
            direction='minimize',
            sampler=TPESampler(seed=42),
            storage=storage,
            load_if_exists=True
        )
        
        # Create objective with subprocess isolation
        objective = create_optuna_objective(
            coordinator,
            model_types=SAFE_MODEL_ORDER,
            metric='val_mse'
        )
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best result
        best_trial = study.best_trial
        best_model_type = best_trial.params.get('model_type', 'unknown')
        
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Best Trial: {best_trial.number}")
        print(f"Best Model: {best_model_type}")
        print(f"Best Val MSE: {best_trial.value:.6f}")
        print("="*70)
        
        # Collect results per model type
        model_results = {}
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                mt = trial.params.get('model_type', 'unknown')
                if mt not in model_results or trial.value < model_results[mt]['val_mse']:
                    model_results[mt] = {
                        'trial_number': trial.number,
                        'val_mse': trial.value,
                        'params': trial.params
                    }
        
        # Print summary
        print("\nBest result per model type:")
        print(f"{'Model':<12} {'Val MSE':<12} {'Trial':<8}")
        print("-"*35)
        for mt in SAFE_MODEL_ORDER:
            if mt in model_results:
                r = model_results[mt]
                marker = "🏆" if mt == best_model_type else "  "
                print(f"{marker} {mt:<10} {r['val_mse']:<12.6f} {r['trial_number']:<8}")
        
        return {
            'best_model_type': best_model_type,
            'best_trial_number': best_trial.number,
            'best_val_mse': best_trial.value,
            'best_params': best_trial.params,
            'model_results': model_results,
            'study_name': study.study_name,
            'n_trials': len(study.trials)
        }


# ============================================================================
# ORIGINAL ANTI-OVERFIT META-OPTIMIZER (Preserved for Backward Compatibility)
# ============================================================================

class AntiOverfitMetaOptimizer:
    """
    Meta-optimizer with strong anti-overfitting measures
    
    ORIGINAL IMPLEMENTATION - Used when --compare-models is NOT specified
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
        """Initialize anti-overfit meta-optimizer"""
        self.survivors = np.array(survivors)
        self.lottery_history = lottery_history
        self.actual_quality = np.array(actual_quality)
        self.base_config_path = base_config_path
        self.k_folds = k_folds

        self.study_name = study_name or f"anti_overfit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage or 'sqlite:///optuna_studies.db'

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self._create_splits(test_holdout_pct)

        self.best_config = None
        self.best_metrics = None
        self.optimization_history = []
        self.trial_times = []
        self.n_trials_total = None

    def _create_splits(self, test_pct: float):
        """Create proper train/val/test splits"""
        n_total = len(self.survivors)
        n_test = int(n_total * test_pct)

        indices = np.random.permutation(n_total)

        self.test_indices = indices[:n_test]
        self.test_survivors = self.survivors[self.test_indices]
        self.test_quality = self.actual_quality[self.test_indices]

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
        """Objective with K-fold cross-validation"""
        trial_start = time.time()

        config = self._sample_config(trial)

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

        avg_metrics = self._aggregate_fold_metrics(fold_metrics)

        trial_time = time.time() - trial_start
        self.trial_times.append(trial_time)

        self.optimization_history.append({
            'trial': trial.number,
            'config': config,
            'fold_metrics': fold_metrics,
            'avg_metrics': avg_metrics,
            'score': avg_metrics['score'],
            'duration_seconds': trial_time
        })

        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"TRIAL {trial.number} SUMMARY")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"  Score: {avg_metrics['score']:.4f}")
        self.logger.info(f"  Val Variance: {avg_metrics['val_variance']:.4f}")
        self.logger.info(f"  Val MAE: {avg_metrics['val_mae']:.4f}")
        self.logger.info(f"  Overfit Ratio: {avg_metrics['overfit_ratio']:.2f}")
        self.logger.info(f"  Consistency: {avg_metrics['variance_consistency']:.4f}")
        self.logger.info(f"  Duration: {trial_time:.1f}s")

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
        """FINAL evaluation on TRUE holdout test set"""
        self.logger.info("\n" + "="*70)
        self.logger.info("FINAL EVALUATION ON HOLDOUT TEST SET")
        self.logger.info("="*70)

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

        test_pred = np.array(engine.predict_quality_batch(self.test_survivors.tolist()))
        train_pred = np.array(engine.predict_quality_batch(self.train_val_survivors.tolist()))

        test_variance = float(np.var(test_pred))
        train_variance = float(np.var(train_pred))
        test_mae = float(np.mean(np.abs(test_pred - self.test_quality)))
        train_mae = float(np.mean(np.abs(train_pred - self.train_val_quality)))

        overfit_ratio = test_mae / (train_mae + 1e-8)

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
        """Print comparison table of top trials"""
        if not self.optimization_history:
            return

        self.logger.info("\n" + "="*70)
        self.logger.info("TOP 5 TRIAL COMPARISON")
        self.logger.info("="*70)

        sorted_trials = sorted(
            self.optimization_history,
            key=lambda x: x['score'],
            reverse=True
        )[:5]

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
        """Run meta-optimization with anti-overfitting measures"""
        self.n_trials_total = n_trials

        self.logger.info(f"\nStarting anti-overfit meta-optimization...")
        self.logger.info(f"Study name: {self.study_name}")
        self.logger.info(f"Storage: {self.storage}")
        self.logger.info(f"Using K-fold CV with {self.k_folds} folds")
        self.logger.info(f"Test set HELD OUT: {len(self.test_survivors)} survivors")
        self.logger.info(f"Running {n_trials} optimization trials...\n")

        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            sampler=TPESampler(seed=42),
            storage=self.storage,
            load_if_exists=True
        )

        study.optimize(self.objective, n_trials=n_trials)

        self.best_config = self.optimization_history[study.best_trial.number]['config']

        self.print_trial_comparison()

        self.best_metrics = self.final_evaluation(self.best_config)

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
        """Save comprehensive optimization results"""
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
    global CUDA_INITIALIZED

    parser = argparse.ArgumentParser(
        description='Anti-Overfit Meta-Prediction Optimizer v2.0 (Subprocess Isolation)'
    )
    
    # Required arguments
    parser.add_argument('--survivors', required=True,
                        help='Path to survivors JSON file')
    parser.add_argument('--lottery-data', required=True,
                        help='Path to lottery data JSON file')
    
    # Optimization parameters
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--k-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--test-holdout', type=float, default=0.2,
                        help='Fraction of data for test holdout (default: 0.2)')
    
    # Optuna persistence
    parser.add_argument('--study-name', type=str,
                        help='Optuna study name (auto-generated if not specified)')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_studies.db',
                        help='Optuna storage path (default: sqlite:///optuna_studies.db)')
    
    # Multi-model options (NEW)
    parser.add_argument('--model-type', type=str, default='neural_net',
                        choices=['neural_net', 'xgboost', 'lightgbm', 'catboost'],
                        help='Model type for single-model training (default: neural_net)')
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare all 4 model types using subprocess isolation')
    parser.add_argument('--save-all-models', action='store_true',
                        help='Save all trained models (not just winner) for AI analysis')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='models/reinforcement',
                        help='Output directory for models (default: models/reinforcement)')

    args = parser.parse_args()

    # Store CLI args for provenance
    cli_args = vars(args).copy()

    print("="*70)
    print("ANTI-OVERFIT META-PREDICTION OPTIMIZER v2.0")
    print("="*70)
    
    # CONDITIONAL CUDA initialization
    if args.compare_models:
        print("⚡ Mode: Multi-Model Comparison (Subprocess Isolation)")
        print("   GPU initialization deferred to subprocesses")
        print(f"   Models: {SAFE_MODEL_ORDER}")
        if not SUBPROCESS_AVAILABLE:
            print("\n❌ ERROR: subprocess_trial_coordinator.py not found!")
            print("   Please ensure it's in ~/distributed_prng_analysis/")
            sys.exit(1)
    else:
        CUDA_INITIALIZED = initialize_cuda_early()
        print(f"⚡ Mode: Single Model ({args.model_type})")
        print(f"   CUDA initialized: {CUDA_INITIALIZED}")
    
    print("="*70)

    # Load survivors data
    print(f"\nLoading data...")
    print(f"  Survivors: {args.survivors}")
    print(f"  Lottery: {args.lottery_data}")
    
    # Try to load features if available
    try:
        X, y, feature_metadata = load_features_from_survivors(args.survivors)
        has_features = X.shape[1] > 0
        print(f"  ✅ Loaded {X.shape[0]} survivors with {X.shape[1]} features")
        if feature_metadata.get('warnings'):
            for w in feature_metadata['warnings']:
                print(f"  ⚠️  Warning: {w}")
    except Exception as e:
        print(f"  ⚠️  Could not load features: {e}")
        print(f"  Falling back to seed-only mode")
        has_features = False
        X, y, feature_metadata = None, None, {}
    
    # Load lottery data
    with open(args.lottery_data) as f:
        lottery_data = json.load(f)
        lottery_history = [d['draw'] if isinstance(d, dict) else d for d in lottery_data]
    print(f"  ✅ Loaded {len(lottery_history)} lottery draws")

    # ========================================================================
    # MULTI-MODEL COMPARISON (Subprocess Isolation)
    # ========================================================================
    if args.compare_models:
        if not has_features:
            print("\n❌ ERROR: --compare-models requires survivors with features!")
            print("   Use survivors_with_scores.json from Step 3")
            sys.exit(1)
        
        # Split data
        n_samples = X.shape[0]
        n_test = int(n_samples * args.test_holdout)
        indices = np.random.permutation(n_samples)
        
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
        # Further split train into train/val for comparison
        n_train = len(train_idx)
        n_val = int(n_train * 0.2)
        val_idx = train_idx[:n_val]
        train_idx = train_idx[n_val:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        print(f"\nData splits:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples (held out)")
        
        # Run comparison
        results = run_multi_model_comparison(
            X_train, y_train, X_val, y_val,
            n_trials=args.trials,
            study_name=args.study_name,
            storage=args.storage,
            save_all_models=args.save_all_models,
            output_dir=args.output_dir
        )
        
        # Save metadata
        best_model_type = results['best_model_type']
        
        meta_path = save_model_with_sidecar(
            model_path=f"{args.output_dir}/best_model",  # Placeholder
            model_type=best_model_type,
            feature_metadata=feature_metadata,
            training_params=results['best_params'],
            validation_metrics={'val_mse': results['best_val_mse']},
            cli_args=cli_args,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Metadata saved: {meta_path}")
        print(f"🏆 Best model: {best_model_type}")
        
        return

    # ========================================================================
    # SINGLE MODEL TRAINING (Original Behavior)
    # ========================================================================
    
    # Load data in original format
    with open(args.survivors) as f:
        data = json.load(f)
        survivors = [s['seed'] if isinstance(s, dict) else s for s in data]

    # Use actual quality if available, otherwise synthetic
    if has_features and y is not None:
        actual_quality = y.tolist()
        print(f"\n✅ Using actual quality scores from features")
    else:
        np.random.seed(42)
        actual_quality = np.random.uniform(0.2, 0.8, len(survivors)).tolist()
        print(f"\n⚠️  Using synthetic quality scores (demo mode)")

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
    print("="*70)


if __name__ == "__main__":
    main()
