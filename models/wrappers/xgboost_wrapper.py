"""
XGBoost GPU Wrapper (v3.1.2)

XGBoost with GPU acceleration using gpu_hist tree method.
Runs on Zeus 3080 Ti's only (requires CUDA).

NOTE: XGBoost does NOT have train/eval modes like PyTorch.
      Do NOT call train_mode() or eval_mode() on this wrapper.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from models.wrappers.base import ModelInterface
from models.gpu_memory import GPUMemoryMixin

logger = logging.getLogger(__name__)

# Check XGBoost availability
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
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
        self.device = device or 'cuda:0'
        self.model = None
        self._fitted = False
        self._gpu_info: Dict[str, Any] = {}
        
        logger.info(f"XGBoostWrapper initialized for {self.device}")
    
    @property
    def model_type(self) -> str:
        return self._model_type
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'XGBoostWrapper':
        """
        Train XGBoost with GPU acceleration.
        
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
        
        # XGBoost parameters with GPU acceleration
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist',  # GPU acceleration
            'device': f'cuda:{device_idx}',
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'random_state': 42,
            'verbosity': 1
        }
        
        logger.info(f"Training XGBoost with GPU acceleration on cuda:{device_idx}")
        logger.info(f"  n_estimators: {params['n_estimators']}")
        logger.info(f"  max_depth: {params['max_depth']}")
        
        self.model = xgb.XGBRegressor(**params)
        
        # Prepare eval set
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self._fitted = True
        self._gpu_info = self.log_gpu_memory()
        
        logger.info("XGBoost training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples,)
        """
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """
        Save model to JSON format.
        
        XGBoost native JSON format preserves all model information.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")
        self.model.save_model(path)
        logger.info(f"XGBoost model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'XGBoostWrapper':
        """
        Load model from JSON file.
        
        Args:
            path: Path to .json model file
            device: Device for predictions (default: cuda:0)
            
        Returns:
            Loaded XGBoostWrapper instance
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        wrapper = cls(device=device)
        wrapper.model = xgb.XGBRegressor()
        wrapper.model.load_model(path)
        wrapper._fitted = True
        
        logger.info(f"XGBoost model loaded from {path}")
        return wrapper
