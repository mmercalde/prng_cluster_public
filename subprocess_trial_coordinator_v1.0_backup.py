#!/usr/bin/env python3
"""
Subprocess Trial Coordinator (subprocess_trial_coordinator.py)
==============================================================

Coordinates isolated model training via subprocess execution.
This solves the OpenCL/CUDA conflict when using --compare-models.

This module is imported by meta_prediction_optimizer_anti_overfit.py
and provides subprocess-based training that maintains clean GPU state
between trials.

Key Features:
- Each trial runs in isolated subprocess (fresh GPU state)
- LightGBM (OpenCL) works regardless of trial order
- Full backward compatibility with existing API
- Integrates with Optuna for hyperparameter optimization
- Uses existing model wrappers from models/wrappers/*.py

Usage:
    from subprocess_trial_coordinator import SubprocessTrialCoordinator
    
    coordinator = SubprocessTrialCoordinator(
        X_train, y_train, X_val, y_val,
        worker_script='train_single_trial.py'
    )
    
    result = coordinator.run_trial(
        model_type='lightgbm',
        params={'n_estimators': 100},
        trial_number=0
    )

Author: PRNG Analysis System
Date: December 2025
Version: 1.0.0
"""

import subprocess
import sys
import json
import tempfile
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
from datetime import datetime
import logging

# Version
__version__ = "1.0.0"

# Safe model order - LightGBM MUST be first for OpenCL/CUDA compatibility
SAFE_MODEL_ORDER = ['lightgbm', 'neural_net', 'xgboost', 'catboost']

# Default timeout per trial (seconds)
DEFAULT_TIMEOUT = 300


@dataclass
class TrialResult:
    """Result from a single trial execution."""
    success: bool
    model_type: str
    trial_number: int
    
    # Metrics (populated on success)
    train_mse: float = 0.0
    val_mse: float = 0.0
    train_mae: float = 0.0
    val_mae: float = 0.0
    r2: float = 0.0
    
    # Metadata
    device: str = 'unknown'
    duration: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    best_iteration: int = 0
    
    # Error info (populated on failure)
    error: str = ''
    traceback: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrialResult':
        """Create from dictionary."""
        return cls(
            success=data.get('success', False),
            model_type=data.get('model_type', 'unknown'),
            trial_number=data.get('trial_number', -1),
            train_mse=data.get('train_mse', 0.0),
            val_mse=data.get('val_mse', 0.0),
            train_mae=data.get('train_mae', 0.0),
            val_mae=data.get('val_mae', 0.0),
            r2=data.get('r2', 0.0),
            device=data.get('device', 'unknown'),
            duration=data.get('duration', 0.0),
            params=data.get('params', {}),
            best_iteration=data.get('best_iteration', 0),
            error=data.get('error', ''),
            traceback=data.get('traceback', '')
        )


class SubprocessTrialCoordinator:
    """
    Coordinates model training trials via subprocess isolation.
    
    Each trial runs in a fresh Python subprocess, ensuring:
    - Clean GPU state (no CUDA/OpenCL conflicts)
    - LightGBM works regardless of previous trials
    - Memory isolation between trials
    
    Attributes:
        data_path: Path to saved training data (.npz file)
        worker_script: Path to train_single_trial.py
        timeout: Maximum seconds per trial
        verbose: Print progress to stderr
    """
    
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 worker_script: str = 'train_single_trial.py',
                 timeout: int = DEFAULT_TIMEOUT,
                 verbose: bool = True,
                 temp_dir: Optional[str] = None):
        """
        Initialize coordinator with training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            worker_script: Path to isolated worker script
            timeout: Max seconds per trial
            verbose: Print progress
            temp_dir: Directory for temp files (auto-created if None)
        """
        self.timeout = timeout
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Resolve worker script path
        self.worker_script = self._resolve_worker_script(worker_script)
        
        # Create temp directory for data
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_temp = False
        else:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.temp_dir = Path(self._temp_dir_obj.name)
            self._cleanup_temp = True
        
        # Save training data to temp file
        self.data_path = self.temp_dir / 'trial_data.npz'
        np.savez(
            self.data_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        
        # Store shapes for validation
        self.data_shape = {
            'X_train': X_train.shape,
            'y_train': y_train.shape,
            'X_val': X_val.shape,
            'y_val': y_val.shape
        }
        
        if self.verbose:
            self.logger.info(f"SubprocessTrialCoordinator initialized:")
            self.logger.info(f"  Worker script: {self.worker_script}")
            self.logger.info(f"  Data path: {self.data_path}")
            self.logger.info(f"  X_train shape: {X_train.shape}")
            self.logger.info(f"  Timeout: {timeout}s")
    
    def _resolve_worker_script(self, worker_script: str) -> Path:
        """Find the worker script, checking multiple locations."""
        # Check if absolute path
        if Path(worker_script).is_absolute():
            if Path(worker_script).exists():
                return Path(worker_script)
            raise FileNotFoundError(f"Worker script not found: {worker_script}")
        
        # Check relative to current directory
        if Path(worker_script).exists():
            return Path(worker_script).resolve()
        
        # Check in same directory as this module
        module_dir = Path(__file__).parent
        candidate = module_dir / worker_script
        if candidate.exists():
            return candidate.resolve()
        
        # Check in distributed_prng_analysis directory
        home_dir = Path.home() / 'distributed_prng_analysis' / worker_script
        if home_dir.exists():
            return home_dir.resolve()
        
        raise FileNotFoundError(
            f"Worker script '{worker_script}' not found in:\n"
            f"  - Current directory: {Path.cwd()}\n"
            f"  - Module directory: {module_dir}\n"
            f"  - Home directory: {home_dir.parent}"
        )
    
    def run_trial(self,
                  model_type: str,
                  params: Dict[str, Any],
                  trial_number: int = -1) -> TrialResult:
        """
        Run a single trial in isolated subprocess.
        
        Args:
            model_type: One of 'lightgbm', 'neural_net', 'xgboost', 'catboost'
            params: Hyperparameters for the model
            trial_number: Optuna trial number (for logging)
        
        Returns:
            TrialResult with metrics or error information
        """
        if model_type not in SAFE_MODEL_ORDER:
            return TrialResult(
                success=False,
                model_type=model_type,
                trial_number=trial_number,
                error=f"Unknown model type: {model_type}. Valid: {SAFE_MODEL_ORDER}"
            )
        
        start_time = time.time()
        
        if self.verbose:
            self.logger.info(f"\n{'─'*60}")
            self.logger.info(f"Trial {trial_number}: {model_type.upper()}")
            self.logger.info(f"{'─'*60}")
        
        try:
            # Build subprocess command
            cmd = [
                sys.executable,
                str(self.worker_script),
                '--model-type', model_type,
                '--data-path', str(self.data_path),
                '--params', json.dumps(params),
                '--trial-number', str(trial_number)
            ]
            
            if self.verbose:
                cmd.append('--verbose')
            
            # Run subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.worker_script.parent)
            )
            
            elapsed = time.time() - start_time
            
            # Check return code
            if result.returncode != 0:
                if self.verbose:
                    self.logger.error(f"  ❌ FAILED (exit code {result.returncode})")
                    if result.stderr:
                        self.logger.error(f"  stderr: {result.stderr[:500]}")
                
                return TrialResult(
                    success=False,
                    model_type=model_type,
                    trial_number=trial_number,
                    error=f"Exit code {result.returncode}: {result.stderr[:500]}",
                    duration=elapsed,
                    params=params
                )
            
            # Parse JSON output
            try:
                # Find JSON line in output (worker may print other stuff to stderr)
                stdout_lines = result.stdout.strip().split('\n')
                json_lines = [l for l in stdout_lines if l.strip().startswith('{')]
                
                if not json_lines:
                    raise ValueError("No JSON output found")
                
                output = json.loads(json_lines[-1])
                
                # Create result
                trial_result = TrialResult.from_dict(output)
                trial_result.params = params
                
                if self.verbose:
                    device_icon = '🚀' if 'cuda' in trial_result.device or trial_result.device == 'gpu' else '💻'
                    self.logger.info(f"  ✅ SUCCESS {device_icon}")
                    self.logger.info(f"  Device: {trial_result.device}")
                    self.logger.info(f"  Val MSE: {trial_result.val_mse:.6f}")
                    self.logger.info(f"  R²: {trial_result.r2:.4f}")
                    self.logger.info(f"  Duration: {trial_result.duration:.2f}s")
                
                return trial_result
                
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                if self.verbose:
                    self.logger.error(f"  ❌ FAILED (bad JSON output)")
                    self.logger.error(f"  stdout: {result.stdout[:500]}")
                
                return TrialResult(
                    success=False,
                    model_type=model_type,
                    trial_number=trial_number,
                    error=f"Invalid JSON output: {e}",
                    duration=elapsed,
                    params=params
                )
                
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            if self.verbose:
                self.logger.error(f"  ❌ FAILED (timeout after {self.timeout}s)")
            
            return TrialResult(
                success=False,
                model_type=model_type,
                trial_number=trial_number,
                error=f"Timeout after {self.timeout}s",
                duration=elapsed,
                params=params
            )
            
        except Exception as e:
            import traceback
            elapsed = time.time() - start_time
            if self.verbose:
                self.logger.error(f"  ❌ FAILED (exception: {e})")
            
            return TrialResult(
                success=False,
                model_type=model_type,
                trial_number=trial_number,
                error=str(e),
                traceback=traceback.format_exc(),
                duration=elapsed,
                params=params
            )
    
    def run_comparison(self,
                       model_types: Optional[List[str]] = None,
                       params_per_model: Optional[Dict[str, Dict[str, Any]]] = None,
                       default_params: Optional[Dict[str, Any]] = None) -> Dict[str, TrialResult]:
        """
        Run comparison across multiple model types.
        
        Args:
            model_types: List of model types to compare (default: SAFE_MODEL_ORDER)
            params_per_model: Dict mapping model_type to params
            default_params: Default params for all models
        
        Returns:
            Dict mapping model_type to TrialResult
        """
        if model_types is None:
            model_types = SAFE_MODEL_ORDER.copy()
        
        # Ensure safe ordering (LightGBM first)
        if 'lightgbm' in model_types and model_types[0] != 'lightgbm':
            self.logger.warning("⚠️  Reordering models: LightGBM must run first (OpenCL/CUDA conflict)")
            model_types = ['lightgbm'] + [m for m in model_types if m != 'lightgbm']
        
        if default_params is None:
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6
            }
        
        if params_per_model is None:
            params_per_model = {}
        
        results = {}
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("MODEL COMPARISON (Subprocess Isolation)")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Models: {model_types}")
        self.logger.info(f"Order guaranteed: LightGBM first (OpenCL/CUDA safe)")
        
        for i, model_type in enumerate(model_types):
            # Get params for this model
            params = params_per_model.get(model_type, default_params.copy())
            
            # Run trial
            result = self.run_trial(
                model_type=model_type,
                params=params,
                trial_number=i
            )
            
            results[model_type] = result
        
        # Summary
        self._print_comparison_summary(results)
        
        return results
    
    def _print_comparison_summary(self, results: Dict[str, TrialResult]):
        """Print comparison summary."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("COMPARISON SUMMARY")
        self.logger.info(f"{'='*60}")
        
        # Sort by val_mse (lower is better)
        successful = [(k, v) for k, v in results.items() if v.success]
        failed = [(k, v) for k, v in results.items() if not v.success]
        
        if successful:
            sorted_results = sorted(successful, key=lambda x: x[1].val_mse)
            
            self.logger.info(f"\n{'Model':<12} {'Val MSE':<12} {'R²':<10} {'Device':<15} {'Time':<8}")
            self.logger.info(f"{'-'*12} {'-'*12} {'-'*10} {'-'*15} {'-'*8}")
            
            for i, (model_type, result) in enumerate(sorted_results):
                rank = '🏆' if i == 0 else f'#{i+1}'
                self.logger.info(
                    f"{rank} {model_type:<10} {result.val_mse:<12.6f} {result.r2:<10.4f} "
                    f"{result.device:<15} {result.duration:<8.2f}s"
                )
            
            best_model, best_result = sorted_results[0]
            self.logger.info(f"\n✅ Best model: {best_model} (Val MSE: {best_result.val_mse:.6f})")
        
        if failed:
            self.logger.warning(f"\n❌ Failed models:")
            for model_type, result in failed:
                self.logger.warning(f"  {model_type}: {result.error[:60]}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self._cleanup_temp and hasattr(self, '_temp_dir_obj'):
            self._temp_dir_obj.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


def create_optuna_objective(coordinator: SubprocessTrialCoordinator,
                            model_types: Optional[List[str]] = None,
                            metric: str = 'val_mse') -> callable:
    """
    Create an Optuna objective function that uses subprocess isolation.
    
    Args:
        coordinator: SubprocessTrialCoordinator instance
        model_types: List of model types to sample from
        metric: Metric to optimize ('val_mse', 'val_mae', 'r2')
    
    Returns:
        Callable objective function for Optuna
    """
    if model_types is None:
        model_types = SAFE_MODEL_ORDER.copy()
    
    def objective(trial) -> float:
        """Optuna objective with subprocess isolation."""
        
        # Sample model type
        model_type = trial.suggest_categorical('model_type', model_types)
        
        # Sample hyperparameters based on model type
        params = sample_params_for_model(trial, model_type)
        
        # Run trial in subprocess
        result = coordinator.run_trial(
            model_type=model_type,
            params=params,
            trial_number=trial.number
        )
        
        # Return metric (Optuna minimizes by default)
        if not result.success:
            return float('inf')  # Failed trials get worst score
        
        if metric == 'val_mse':
            return result.val_mse
        elif metric == 'val_mae':
            return result.val_mae
        elif metric == 'r2':
            return -result.r2  # Negate because Optuna minimizes
        else:
            return result.val_mse
    
    return objective


def sample_params_for_model(trial, model_type: str) -> Dict[str, Any]:
    """
    Sample hyperparameters for a specific model type.
    
    Args:
        trial: Optuna trial object
        model_type: Model type string
    
    Returns:
        Dict of hyperparameters
    """
    params = {}
    
    if model_type == 'lightgbm':
        params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 500),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 15, 127),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
            'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lgb_lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lgb_lambda_l2', 1e-8, 10.0, log=True),
        }
    
    elif model_type == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 500),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        }
    
    elif model_type == 'catboost':
        params = {
            'n_estimators': trial.suggest_int('cb_n_estimators', 50, 500),
            'max_depth': trial.suggest_int('cb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('cb_lr', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1e-8, 10.0, log=True),
            'random_strength': trial.suggest_float('cb_random_strength', 0.0, 10.0),
            'bagging_temperature': trial.suggest_float('cb_bagging_temp', 0.0, 10.0),
            'border_count': trial.suggest_int('cb_border_count', 32, 255),
        }
    
    elif model_type == 'neural_net':
        # Dynamic architecture
        n_layers = trial.suggest_int('nn_n_layers', 2, 4)
        layers = []
        for i in range(n_layers):
            if i == 0:
                size = trial.suggest_int(f'nn_layer_{i}', 64, 512)
            else:
                size = trial.suggest_int(f'nn_layer_{i}', 32, layers[-1])
            layers.append(size)
        
        params = {
            'hidden_layers': layers,
            'dropout': trial.suggest_float('nn_dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('nn_lr', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('nn_batch_size', [64, 128, 256, 512]),
            'epochs': trial.suggest_int('nn_epochs', 30, 150),
            'early_stopping_patience': trial.suggest_int('nn_patience', 5, 20),
            'optimizer': trial.suggest_categorical('nn_optimizer', ['adam', 'adamw', 'sgd']),
            'weight_decay': trial.suggest_float('nn_weight_decay', 1e-6, 1e-2, log=True),
        }
    
    return params


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_isolated_comparison(X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            model_types: Optional[List[str]] = None,
                            params: Optional[Dict[str, Any]] = None,
                            verbose: bool = True) -> Tuple[str, TrialResult, Dict[str, TrialResult]]:
    """
    Convenience function to run model comparison with subprocess isolation.
    
    Args:
        X_train, y_train, X_val, y_val: Training and validation data
        model_types: Models to compare (default: all 4)
        params: Default hyperparameters
        verbose: Print progress
    
    Returns:
        Tuple of (best_model_type, best_result, all_results)
    """
    with SubprocessTrialCoordinator(
        X_train, y_train, X_val, y_val,
        verbose=verbose
    ) as coordinator:
        
        results = coordinator.run_comparison(
            model_types=model_types,
            default_params=params
        )
        
        # Find best
        successful = {k: v for k, v in results.items() if v.success}
        
        if not successful:
            raise RuntimeError("All models failed! Check error messages above.")
        
        best_model = min(successful.items(), key=lambda x: x[1].val_mse)
        
        return best_model[0], best_model[1], results


if __name__ == '__main__':
    """Test the coordinator with synthetic data."""
    import numpy as np
    
    print("="*60)
    print("SUBPROCESS TRIAL COORDINATOR - TEST")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 5000, 50
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X @ np.random.randn(n_features) + np.random.randn(n_samples) * 0.5
    y = y.astype(np.float32)
    
    # Split
    split = int(n_samples * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"\nData: X_train={X_train.shape}, X_val={X_val.shape}")
    
    # Test comparison
    best_model, best_result, all_results = run_isolated_comparison(
        X_train, y_train, X_val, y_val,
        params={'n_estimators': 50, 'learning_rate': 0.1}
    )
    
    print(f"\n{'='*60}")
    print(f"🏆 WINNER: {best_model}")
    print(f"   Val MSE: {best_result.val_mse:.6f}")
    print(f"   R²: {best_result.r2:.4f}")
    print(f"{'='*60}")
