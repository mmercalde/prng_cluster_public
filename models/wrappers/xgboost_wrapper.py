"""
XGBoost GPU Wrapper (v3.2.0)

XGBoost with GPU acceleration using gpu_hist tree method.
Runs on Zeus 3080 Ti's only (requires CUDA).

NOTE: XGBoost does NOT have train/eval modes like PyTorch.
      Do NOT call train_mode() or eval_mode() on this wrapper.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from models.wrappers.base import ModelInterface
from models.gpu_memory import GPUMemoryMixin

logger = logging.getLogger(__name__)

# Check XGBoost availability
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    XGBOOST_VERSION = xgb.__version__
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBOOST_VERSION = None
    logger.warning("XGBoost not installed. Run: pip install xgboost")


class XGBoostWrapper(GPUMemoryMixin):
    """
    XGBoost with GPU acceleration.
    
    Runs on Zeus 3080 Ti's only (CUDA required).
    Implements ModelInterface protocol.
    
    NOTE: Does NOT implement train_mode/eval_mode (tree models don't have these).
    """
    
    _model_type: str = "xgboost"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        """
        Initialize XGBoost wrapper.
        
        Args:
            config: Model configuration dict with:
                - n_estimators: Number of trees (default: 100)
                - max_depth: Maximum tree depth (default: 6)
                - learning_rate: Learning rate (default: 0.1)
                - subsample: Row subsampling ratio (default: 0.8)
                - colsample_bytree: Column subsampling ratio (default: 0.8)
            device: Device string (default: 'cuda:0')
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        self.config = config or {}
        self._booster = None
        self._fitted = False
        self._feature_count = None
        
        # Parse device
        self._use_gpu = True
        self._device_id = 0
        
        if device:
            if 'cuda' in device.lower() or 'gpu' in device.lower():
                self._use_gpu = True
                if ':' in device:
                    try:
                        self._device_id = int(device.split(':')[1])
                    except (ValueError, IndexError):
                        pass
            elif device.lower() == 'cpu':
                self._use_gpu = False
        
        logger.info(f"XGBoost initialized: use_gpu={self._use_gpu}, device_id={self._device_id}")
    
    def _get_xgb_params(self) -> Dict[str, Any]:
        """
        Build XGBoost parameters with GPU configuration.
        
        Returns:
            XGBoost parameter dict
        """
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'random_state': self.config.get('random_state', 42),
            'verbosity': 0,
        }
        
        if self._use_gpu:
            # Use gpu_hist for GPU acceleration
            params['tree_method'] = 'gpu_hist'
            params['device'] = f'cuda:{self._device_id}'
            logger.info(f"Using XGBoost GPU (gpu_hist) on device {self._device_id}")
        else:
            params['tree_method'] = 'hist'
            params['device'] = 'cpu'
            logger.info("Using XGBoost CPU (hist)")
        
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
        Train XGBoost model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            eval_set: List of (X, y) tuples for validation
            early_stopping_rounds: Early stopping patience
            verbose: Whether to show training progress
            
        Returns:
            Training info dict with metrics
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        
        n_samples, n_features = X.shape
        self._feature_count = n_features
        
        params = self._get_xgb_params()
        n_estimators = self.config.get('n_estimators', 100)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        evals = [(dtrain, 'train')]
        if eval_set is not None:
            # Handle both formats: [(X, y)] or (X, y)
            if isinstance(eval_set, tuple) and len(eval_set) == 2 and isinstance(eval_set[0], np.ndarray):
                # Format: (X_val, y_val) - direct tuple
                X_val, y_val = eval_set
                dval = xgb.DMatrix(X_val, label=y_val)
                evals.append((dval, 'valid_0'))
            elif isinstance(eval_set, list) and len(eval_set) > 0:
                # Format: [(X_val, y_val), ...] - list of tuples
                for i, (X_val, y_val) in enumerate(eval_set):
                    dval = xgb.DMatrix(X_val, label=y_val)
                    evals.append((dval, f'valid_{i}'))
        
        # Train
        callbacks = []
        if early_stopping_rounds:
            callbacks.append(xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                save_best=True
            ))
        
        try:
            self._booster = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=evals,
                callbacks=callbacks if callbacks else None,
                verbose_eval=verbose
            )
            self._fitted = True
            
            result = {
                'n_estimators': self._booster.num_boosted_rounds(),
                'best_iteration': self._booster.best_iteration if hasattr(self._booster, 'best_iteration') else self._booster.num_boosted_rounds(),
                'feature_count': n_features,
                'use_gpu': self._use_gpu,
                'device_id': self._device_id,
            }
            
            # Get best score
            if hasattr(self._booster, 'best_score'):
                result['best_score'] = self._booster.best_score
            
            logger.info(f"XGBoost training complete: {result}")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle GPU errors by falling back to CPU
            if 'gpu' in error_msg or 'cuda' in error_msg:
                logger.warning(f"GPU error: {e}. Falling back to CPU...")
                self._use_gpu = False
                params = self._get_xgb_params()
                
                self._booster = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=n_estimators,
                    evals=evals,
                    callbacks=callbacks if callbacks else None,
                    verbose_eval=verbose
                )
                self._fitted = True
                
                result = {
                    'n_estimators': self._booster.num_boosted_rounds(),
                    'feature_count': n_features,
                    'use_gpu': False,
                    'fallback_reason': str(e),
                }
                logger.info(f"XGBoost training complete (CPU fallback): {result}")
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
        
        dmatrix = xgb.DMatrix(X)
        return self._booster.predict(dmatrix)
    
    def save(self, path: str) -> None:
        """
        Save model to JSON format.
        
        XGBoost JSON format is portable and human-readable.
        """
        if self._booster is None:
            raise RuntimeError("No model to save. Call fit() first.")
        
        path = Path(path)
        self._booster.save_model(str(path))
        
        logger.info(f"XGBoost model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'XGBoostWrapper':
        """
        Load model from file.
        
        Args:
            path: Path to model file
            device: Device for predictions
            
        Returns:
            Loaded XGBoostWrapper instance
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        
        wrapper = cls(device=device)
        wrapper._booster = xgb.Booster()
        wrapper._booster.load_model(str(path))
        wrapper._fitted = True
        
        # Try to get feature count
        try:
            wrapper._feature_count = wrapper._booster.num_features()
        except:
            pass
        
        logger.info(f"XGBoost model loaded from {path}")
        return wrapper
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dict mapping feature index to importance score
        """
        if self._booster is None:
            return {}
        
        importance = self._booster.get_score(importance_type='gain')
        # Convert from f0, f1, ... to feature_0, feature_1, ...
        return {f"feature_{k[1:]}": float(v) for k, v in importance.items()}
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            **self.config,
            'use_gpu': self._use_gpu,
            'device_id': self._device_id,
        }
