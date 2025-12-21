"""
LightGBM GPU Wrapper (v3.1.2)

LightGBM with GPU acceleration.
Runs on Zeus 3080 Ti's only (requires CUDA).

NOTE: LightGBM does NOT have train/eval modes like PyTorch.
      Do NOT call train_mode() or eval_mode() on this wrapper.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from models.wrappers.base import ModelInterface
from models.gpu_memory import GPUMemoryMixin

logger = logging.getLogger(__name__)

# Check LightGBM availability
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Run: pip install lightgbm")


class LightGBMWrapper(GPUMemoryMixin):
    """
    LightGBM with GPU acceleration.
    
    Runs on Zeus 3080 Ti's only.
    Implements ModelInterface protocol.
    
    NOTE: Does NOT implement train_mode/eval_mode (tree models don't have these).
    """
    
    _model_type: str = "lightgbm"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        """
        Initialize LightGBM wrapper.
        
        Args:
            config: Model configuration dict with:
                - n_estimators: Number of trees (default: 100)
                - max_depth: Maximum tree depth (default: 6)
                - learning_rate: Learning rate (default: 0.1)
                - subsample: Row subsampling ratio (default: 0.8)
                - colsample_bytree: Column subsampling ratio (default: 0.8)
            device: Device string (default: 'cuda:0')
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        self.config = config or {}
        self.device = device or 'cuda:0'
        self.model = None
        self._booster = None  # Store booster separately for save/load
        self._fitted = False
        self._gpu_info: Dict[str, Any] = {}
        
        logger.info(f"LightGBMWrapper initialized for {self.device}")
    
    @property
    def model_type(self) -> str:
        return self._model_type
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'LightGBMWrapper':
        """
        Train LightGBM with GPU acceleration.
        
        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training targets, shape (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            self for method chaining
        """
        # Extract device index
        device_idx = 0
        if ':' in str(self.device):
            device_idx = int(str(self.device).split(':')[1])
        
        # LightGBM parameters with GPU acceleration
        params = {
            'objective': 'regression',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': device_idx,
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'random_state': 42,
            'verbose': -1
        }
        
        logger.info(f"Training LightGBM with GPU acceleration on device {device_idx}")
        logger.info(f"  n_estimators: {params['n_estimators']}")
        logger.info(f"  max_depth: {params['max_depth']}")
        
        self.model = lgb.LGBMRegressor(**params)
        
        # Prepare callbacks and eval set
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set
        )
        
        self._booster = self.model.booster_
        self._fitted = True
        self._gpu_info = self.log_gpu_memory()
        
        logger.info("LightGBM training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples,)
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
        
        if self._booster is not None:
            self._booster.save_model(path)
        else:
            self.model.booster_.save_model(path)
        
        logger.info(f"LightGBM model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'LightGBMWrapper':
        """
        Load model from text file.
        
        Args:
            path: Path to .txt model file
            device: Device for predictions (default: cuda:0)
            
        Returns:
            Loaded LightGBMWrapper instance
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        wrapper = cls(device=device)
        wrapper._booster = lgb.Booster(model_file=path)
        wrapper._fitted = True
        
        logger.info(f"LightGBM model loaded from {path}")
        return wrapper
