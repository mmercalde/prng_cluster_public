"""
CatBoost GPU Wrapper (v3.2.0) - FIXED

CatBoost with multi-GPU acceleration.
Uses BOTH Zeus 3080 Ti's (devices='0:1').

FIXES in v3.2.0:
- Bootstrap/subsample conflict resolution
- Proper GPU parameter configuration
- Removed conflicting bootstrap_type with subsample

NOTE: CatBoost does NOT have train/eval modes like PyTorch.
      GPU utilization verification is configuration-only (honest).

CatBoost Bootstrap Rules:
- If subsample < 1.0, you MUST set bootstrap_type='Bernoulli' or 'MVS'
- bootstrap_type='Bayesian' (default) does NOT support subsample
- We use 'MVS' (Minimal Variance Sampling) which is optimal for GPU
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from models.wrappers.base import ModelInterface
from models.gpu_memory import GPUMemoryMixin

logger = logging.getLogger(__name__)

# Check CatBoost availability
try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
    import catboost
    CATBOOST_VERSION = catboost.__version__
except ImportError:
    CATBOOST_AVAILABLE = False
    CATBOOST_VERSION = None
    logger.warning("CatBoost not installed. Run: pip install catboost")


class CatBoostWrapper(GPUMemoryMixin):
    """
    CatBoost with multi-GPU acceleration.
    
    Uses BOTH Zeus 3080 Ti's for training (devices='0:1').
    Implements ModelInterface protocol.
    
    FIXED in v3.2.0:
    - Bootstrap/subsample conflict resolved by using MVS bootstrap
    - Proper GPU device configuration
    
    NOTE: Does NOT implement train_mode/eval_mode (tree models don't have these).
    """
    
    _model_type: str = "catboost"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        """
        Initialize CatBoost wrapper.
        
        Args:
            config: Model configuration dict with:
                - iterations: Number of trees (default: 100)
                - depth: Tree depth (default: 6)
                - learning_rate: Learning rate (default: 0.1)
                - subsample: Row subsampling ratio (default: 0.8)
                - colsample_bylevel: Column subsampling (default: 0.8)
                - random_seed: Random seed (default: 42)
                - use_gpu: Whether to use GPU (default: True)
            device: Device string (e.g., 'cuda:0' or 'cuda:0,1' for multi-GPU)
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        self.config = config or {}
        self.model = None
        self._fitted = False
        self._feature_count = None
        
        # GPU configuration
        self._use_gpu = self.config.get('use_gpu', True)
        
        # Parse device string
        if device:
            if 'cuda' in device.lower() or 'gpu' in device.lower():
                self._use_gpu = True
                # Extract device IDs
                if ':' in device:
                    device_part = device.split(':')[1]
                    # Handle multi-GPU: "cuda:0,1" or "cuda:0:1"
                    if ',' in device_part:
                        self._devices = device_part  # "0,1"
                    else:
                        self._devices = "0:1"  # Always use both GPUs
                else:
                    self._devices = "0:1"  # Always use both GPUs
            else:
                self._use_gpu = False
                self._devices = None
        else:
            # Default: use both Zeus GPUs
            self._devices = "0:1"
        
        logger.info(f"CatBoost initialized: use_gpu={self._use_gpu}, devices={self._devices}")
    
    def _get_catboost_params(self) -> Dict[str, Any]:
        """
        Build CatBoost parameters with proper GPU and subsample configuration.
        
        CRITICAL: CatBoost has specific rules about bootstrap_type and subsample:
        - Bayesian (default): Does NOT support subsample
        - Bernoulli: Supports subsample, basic random sampling
        - MVS: Supports subsample, better variance reduction (recommended for GPU)
        
        Returns:
            CatBoost parameter dict
        """
        params = {
            'iterations': self.config.get('iterations', 100),
            'depth': self.config.get('depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'random_seed': self.config.get('random_seed', 42),
            'loss_function': 'RMSE',
            'verbose': False,
            'allow_writing_files': False,  # Don't create temp files
        }
        
        # Handle subsample - MUST use compatible bootstrap_type
        subsample = self.config.get('subsample', 0.8)
        if subsample < 1.0:
            # MVS (Minimal Variance Sampling) is optimal for GPU with subsample
            params['bootstrap_type'] = 'MVS'
            params['subsample'] = subsample
            logger.debug(f"Using MVS bootstrap with subsample={subsample}")
        # If subsample=1.0, don't set bootstrap_type or subsample (use defaults)
        
        # Column subsampling - ONLY for CPU mode
        # GPU mode does not support rsm (random subspace method) for regression
        colsample = self.config.get('colsample_bylevel', 0.8)
        
        # GPU configuration
        if self._use_gpu:
            params['task_type'] = 'GPU'
            params['devices'] = self._devices
            # GPU-specific optimizations
            params['gpu_ram_part'] = self.config.get('gpu_ram_part', 0.9)
            # NOTE: colsample_bylevel (rsm) NOT supported on GPU for regression
            # So we don't add it here
            logger.info(f"CatBoost GPU mode: devices={self._devices}")
        else:
            params['task_type'] = 'CPU'
            params['thread_count'] = -1  # Use all cores
            # colsample_bylevel only works on CPU
            if colsample < 1.0:
                params['colsample_bylevel'] = colsample
            logger.info("CatBoost CPU mode")
        
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
        Train CatBoost model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            eval_set: List of (X, y) tuples for validation
            early_stopping_rounds: Early stopping patience
            verbose: Whether to show training progress
            
        Returns:
            Training info dict with metrics
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed")
        
        n_samples, n_features = X.shape
        self._feature_count = n_features
        
        params = self._get_catboost_params()
        
        if verbose:
            params['verbose'] = 10  # Print every 10 iterations
        
        # Create model
        self.model = CatBoostRegressor(**params)
        
        # Create pools (CatBoost's dataset format)
        train_pool = Pool(X, label=y)
        
        eval_pool = None
        if eval_set is not None:
            # Handle both formats: [(X, y)] or (X, y)
            if isinstance(eval_set, tuple) and len(eval_set) == 2 and isinstance(eval_set[0], np.ndarray):
                # Format: (X_val, y_val) - direct tuple
                X_val, y_val = eval_set
                eval_pool = Pool(X_val, label=y_val)
            elif isinstance(eval_set, list) and len(eval_set) > 0:
                # Format: [(X_val, y_val)] - list of tuples
                X_val, y_val = eval_set[0]
                eval_pool = Pool(X_val, label=y_val)
        
        # Train
        try:
            self.model.fit(
                train_pool,
                eval_set=eval_pool,
                early_stopping_rounds=early_stopping_rounds,
                use_best_model=True if eval_pool else False
            )
            self._fitted = True
            
            result = {
                'iterations': self.model.tree_count_,
                'best_iteration': self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else self.model.tree_count_,
                'feature_count': n_features,
                'use_gpu': self._use_gpu,
                'devices': self._devices,
                'bootstrap_type': params.get('bootstrap_type', 'Bayesian'),
            }
            
            # Get best score if available
            if hasattr(self.model, 'best_score_') and self.model.best_score_:
                result['best_score'] = self.model.best_score_
            
            logger.info(f"CatBoost training complete: {result}")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle GPU errors by falling back to CPU
            if 'gpu' in error_msg or 'cuda' in error_msg or 'device' in error_msg:
                logger.warning(f"GPU error: {e}. Falling back to CPU...")
                self._use_gpu = False
                params = self._get_catboost_params()
                
                if verbose:
                    params['verbose'] = 10
                
                self.model = CatBoostRegressor(**params)
                self.model.fit(
                    train_pool,
                    eval_set=eval_pool,
                    early_stopping_rounds=early_stopping_rounds,
                    use_best_model=True if eval_pool else False
                )
                self._fitted = True
                
                result = {
                    'iterations': self.model.tree_count_,
                    'feature_count': n_features,
                    'use_gpu': False,
                    'fallback_reason': str(e),
                }
                logger.info(f"CatBoost training complete (CPU fallback): {result}")
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
        
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """
        Save model to CatBoost binary format.
        
        CatBoost's native format (.cbm) is most efficient.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")
        
        path = Path(path)
        self.model.save_model(str(path))
        
        logger.info(f"CatBoost model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'CatBoostWrapper':
        """
        Load model from file.
        
        Args:
            path: Path to .cbm model file
            device: Device for predictions
            
        Returns:
            Loaded CatBoostWrapper instance
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed")
        
        wrapper = cls(device=device)
        wrapper.model = CatBoostRegressor()
        wrapper.model.load_model(str(path))
        wrapper._fitted = True
        
        # Try to get feature count
        try:
            wrapper._feature_count = wrapper.model.feature_count_
        except:
            pass
        
        logger.info(f"CatBoost model loaded from {path}")
        return wrapper
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dict mapping feature index to importance score
        """
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        return {f"feature_{i}": float(v) for i, v in enumerate(importance)}
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            **self.config,
            'use_gpu': self._use_gpu,
            'devices': self._devices,
        }
