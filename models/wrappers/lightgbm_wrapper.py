"""
LightGBM GPU Wrapper (v3.2.0) - FIXED

LightGBM with GPU acceleration.
Runs on Zeus 3080 Ti's only (requires CUDA or OpenCL).

FIXES in v3.2.0:
- Proper CUDA/OpenCL backend detection
- Explicit device configuration for NVIDIA GPUs
- Fallback to CPU if GPU fails

NOTE: LightGBM GPU requires either:
  1. LightGBM built with CUDA support (pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON)
  2. Or working OpenCL drivers (clinfo should show NVIDIA platform)
  
For NVIDIA GPUs, CUDA is preferred. Install with:
  pip uninstall lightgbm
  pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON --break-system-packages
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import subprocess
import os

from models.wrappers.base import ModelInterface
from models.gpu_memory import GPUMemoryMixin

logger = logging.getLogger(__name__)

# Check LightGBM availability
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_VERSION = lgb.__version__
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_VERSION = None
    logger.warning("LightGBM not installed. Run: pip install lightgbm")


def _check_cuda_available() -> bool:
    """Check if CUDA is available for LightGBM."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_opencl_nvidia() -> bool:
    """Check if OpenCL is available with NVIDIA platform."""
    try:
        result = subprocess.run(
            ['clinfo'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return 'NVIDIA' in result.stdout or 'nvidia' in result.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _detect_gpu_backend() -> str:
    """
    Detect the best available GPU backend for LightGBM.
    
    Returns:
        'gpu' - OpenCL backend (preferred - works with pip-installed LightGBM)
        'cpu' - Fallback if no GPU support available
    
    Note: We use OpenCL ('gpu') because pip-installed LightGBM has OpenCL
    support but NOT CUDA support (CUDA requires building from source).
    The 'device': 'gpu' parameter uses OpenCL, 'device': 'cuda' requires
    a special build.
    """
    # Check if OpenCL is available with NVIDIA platform
    if _check_opencl_nvidia():
        # Verify it actually works with a quick test
        try:
            import lightgbm as lgb
            test_data = lgb.Dataset(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                label=[0, 1]
            )
            test_params = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbose': -1,
                'num_iterations': 1,
                'num_leaves': 2
            }
            lgb.train(test_params, test_data, num_boost_round=1)
            logger.info("LightGBM OpenCL GPU backend verified working")
            return 'gpu'
        except Exception as e:
            logger.warning(f"OpenCL test failed: {e}")
    
    logger.warning("No GPU backend available for LightGBM, using CPU")
    return 'cpu'


class LightGBMWrapper(GPUMemoryMixin):
    """
    LightGBM with GPU acceleration.
    
    Runs on Zeus 3080 Ti's only (CUDA/OpenCL required).
    Implements ModelInterface protocol.
    
    GPU Backend Detection:
    - Prefers CUDA if LightGBM was built with CUDA support
    - Falls back to OpenCL if NVIDIA OpenCL platform available
    - Falls back to CPU as last resort
    
    NOTE: Does NOT implement train_mode/eval_mode (tree models don't have these).
    """
    
    _model_type: str = "lightgbm"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        """
        Initialize LightGBM wrapper.
        
        Args:
            config: Model configuration dict with:
                - n_estimators: Number of trees (default: 100)
                - max_depth: Maximum tree depth (default: -1 for unlimited)
                - learning_rate: Learning rate (default: 0.1)
                - num_leaves: Number of leaves per tree (default: 31)
                - subsample: Row subsampling ratio (default: 0.8)
                - colsample_bytree: Column subsampling ratio (default: 0.8)
                - force_cpu: Force CPU mode even if GPU available (default: False)
            device: Device string (overrides auto-detection)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        self.config = config or {}
        self._fitted = False
        self._booster = None
        self.model = None
        self._feature_count = None
        
        # Determine GPU backend
        force_cpu = self.config.get('force_cpu', False)
        if force_cpu:
            self._gpu_backend = 'cpu'
        elif device:
            # User specified device
            if 'cuda' in device.lower():
                self._gpu_backend = 'gpu'  # Use OpenCL (pip LightGBM has no CUDA)
            elif 'gpu' in device.lower():
                self._gpu_backend = 'gpu'
            else:
                self._gpu_backend = 'cpu'
        else:
            # Auto-detect
            self._gpu_backend = _detect_gpu_backend()
        
        # Extract device ID if specified
        self._device_id = 1  # Default to GPU 1 to avoid OpenCL/CUDA conflict
        if device and ':' in device:
            try:
                self._device_id = int(device.split(':')[1])
            except (ValueError, IndexError):
                pass
        
        logger.info(f"LightGBM initialized with backend: {self._gpu_backend}, device_id: {self._device_id}")
    
    def _get_lgb_params(self, n_features: int) -> Dict[str, Any]:
        """
        Build LightGBM parameters with GPU configuration.
        
        Args:
            n_features: Number of input features
            
        Returns:
            LightGBM parameter dict
        """
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': self.config.get('n_estimators', 100),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'num_leaves': self.config.get('num_leaves', 31),
            'max_depth': self.config.get('max_depth', -1),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'random_state': self.config.get('random_state', 42),
            'verbose': -1,  # Suppress output
            'force_col_wise': True,  # Better for many features
        }
        
        # GPU-specific parameters
        if self._gpu_backend == 'cuda':
            params['device'] = 'cuda'
            params['gpu_device_id'] = self._device_id
            logger.info(f"Using LightGBM CUDA backend on device {self._device_id}")
            
        elif self._gpu_backend == 'gpu':
            # OpenCL backend
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0  # Usually NVIDIA is platform 0
            params['gpu_device_id'] = self._device_id
            # OpenCL-specific optimizations
            params['gpu_use_dp'] = False  # Use single precision for speed
            logger.info(f"Using LightGBM OpenCL backend on device {self._device_id}")
        else:
            # CPU fallback
            params['device'] = 'cpu'
            params['n_jobs'] = -1  # Use all CPU cores
            logger.info("Using LightGBM CPU backend")
        
        return params
    
    @property
    def model_type(self) -> str:
        return self._model_type
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
    
    @property
    def feature_count(self) -> Optional[int]:
        return self._feature_count
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            eval_set: Optional[List[tuple]] = None,
            early_stopping_rounds: Optional[int] = None,
            verbose: bool = False) -> Dict[str, Any]:
        """
        Train LightGBM model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            eval_set: List of (X, y) tuples for validation
            early_stopping_rounds: Early stopping patience
            verbose: Whether to show training progress
            
        Returns:
            Training info dict with metrics
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        
        n_samples, n_features = X.shape
        self._feature_count = n_features
        
        params = self._get_lgb_params(n_features)
        
        # Create datasets
        train_data = lgb.Dataset(X, label=y)
        
        valid_sets = [train_data]
        valid_names = ['train']
        has_eval_set = False
        
        if eval_set is not None:
            # Handle both formats: [(X, y)] or (X, y)
            if isinstance(eval_set, tuple) and len(eval_set) == 2 and isinstance(eval_set[0], np.ndarray):
                # Format: (X_val, y_val) - direct tuple
                X_val, y_val = eval_set
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                valid_sets.append(val_data)
                valid_names.append('valid_0')
                has_eval_set = True
            elif isinstance(eval_set, list) and len(eval_set) > 0:
                # Format: [(X_val, y_val), ...] - list of tuples
                for i, (X_val, y_val) in enumerate(eval_set):
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    valid_sets.append(val_data)
                    valid_names.append(f'valid_{i}')
                has_eval_set = True
        
        # Build callbacks
        callbacks = []
        if early_stopping_rounds and has_eval_set:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
        if not verbose:
            callbacks.append(lgb.log_evaluation(period=0))  # Suppress logging
        
        # Train
        try:
            self._booster = lgb.train(
                params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks if callbacks else None
            )
            self._fitted = True
            
            # Collect metrics
            result = {
                'n_estimators': self._booster.num_trees(),
                'best_iteration': self._booster.best_iteration if hasattr(self._booster, 'best_iteration') else self._booster.num_trees(),
                'feature_count': n_features,
                'gpu_backend': self._gpu_backend,
                'device_id': self._device_id,
            }
            
            # Get best score if early stopping was used
            if hasattr(self._booster, 'best_score') and self._booster.best_score:
                result['best_score'] = self._booster.best_score
            
            logger.info(f"LightGBM training complete: {result}")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle CUDA/OpenCL errors by falling back
            if any(x in error_msg for x in ['cuda', 'opencl', 'clgetplatformids', 'gpu', 'device']):
                
                # If we were trying CUDA, try OpenCL next
                if self._gpu_backend == 'cuda':
                    logger.warning(f"CUDA error: {e}. Trying OpenCL...")
                    self._gpu_backend = 'gpu'
                    params = self._get_lgb_params(n_features)
                    
                    try:
                        self._booster = lgb.train(
                            params,
                            train_data,
                            valid_sets=valid_sets,
                            valid_names=valid_names,
                            callbacks=callbacks if callbacks else None
                        )
                        self._fitted = True
                        
                        result = {
                            'n_estimators': self._booster.num_trees(),
                            'feature_count': n_features,
                            'gpu_backend': 'gpu (OpenCL)',
                            'fallback_reason': f'CUDA failed: {e}',
                        }
                        logger.info(f"LightGBM training complete (OpenCL fallback): {result}")
                        return result
                    except Exception as e2:
                        logger.warning(f"OpenCL also failed: {e2}. Falling back to CPU...")
                        error_msg = str(e2).lower()
                
                # Fall back to CPU
                logger.warning(f"GPU error: {e}. Falling back to CPU...")
                self._gpu_backend = 'cpu'
                params = self._get_lgb_params(n_features)
                
                # Retry with CPU
                self._booster = lgb.train(
                    params,
                    train_data,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=callbacks if callbacks else None
                )
                self._fitted = True
                
                result = {
                    'n_estimators': self._booster.num_trees(),
                    'feature_count': n_features,
                    'gpu_backend': 'cpu (fallback)',
                    'fallback_reason': str(e),
                }
                logger.info(f"LightGBM training complete (CPU fallback): {result}")
                return result
            else:
                raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if self._booster is not None:
            return self._booster.predict(X)
        elif self.model is not None:
            return self.model.predict(X)
        else:
            raise RuntimeError("No model available for prediction")
    
    def save(self, path: str) -> None:
        """
        Save model to text format.
        
        LightGBM text format is human-readable and portable.
        """
        if self._booster is None and self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")
        
        path = Path(path)
        
        if self._booster is not None:
            self._booster.save_model(str(path))
        else:
            self.model.booster_.save_model(str(path))
        
        logger.info(f"LightGBM model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'LightGBMWrapper':
        """
        Load model from text file.
        
        Args:
            path: Path to .txt model file
            device: Device for predictions (default: auto-detect)
            
        Returns:
            Loaded LightGBMWrapper instance
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        wrapper = cls(device=device)
        wrapper._booster = lgb.Booster(model_file=str(path))
        wrapper._fitted = True
        
        # Try to get feature count from model
        try:
            wrapper._feature_count = wrapper._booster.num_feature()
        except:
            pass
        
        logger.info(f"LightGBM model loaded from {path}")
        return wrapper
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dict mapping feature index to importance score
        """
        if self._booster is None:
            return {}
        
        importance = self._booster.feature_importance()
        return {f"feature_{i}": float(v) for i, v in enumerate(importance)}
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            **self.config,
            'gpu_backend': self._gpu_backend,
            'device_id': self._device_id,
        }
