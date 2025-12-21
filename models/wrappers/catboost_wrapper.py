"""
CatBoost GPU Wrapper (v3.1.2)

CatBoost with multi-GPU acceleration.
Uses BOTH Zeus 3080 Ti's (devices='0:1').

NOTE: CatBoost does NOT have train/eval modes like PyTorch.
      GPU utilization verification is configuration-only (honest).
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from models.wrappers.base import ModelInterface
from models.gpu_memory import GPUMemoryMixin

logger = logging.getLogger(__name__)

# Check CatBoost availability
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not installed. Run: pip install catboost")


class CatBoostWrapper(GPUMemoryMixin):
    """
    CatBoost with multi-GPU acceleration.
    
    Uses BOTH Zeus 3080 Ti's (devices='0:1').
    Implements ModelInterface protocol.
    
    NOTE: Does NOT implement train_mode/eval_mode (tree models don't have these).
    NOTE: GPU verification is configuration_only (CatBoost doesn't expose utilization via Python API).
    """
    
    _model_type: str = "catboost"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        """
        Initialize CatBoost wrapper.
        
        Args:
            config: Model configuration dict with:
                - iterations: Number of trees (default: 100)
                - depth: Maximum tree depth (default: 6)
                - learning_rate: Learning rate (default: 0.1)
                - subsample: Row subsampling ratio (default: 0.8)
            device: Device string - for CatBoost, use '0:1' for multi-GPU (default)
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        self.config = config or {}
        # CatBoost uses different device format
        self.device = device or '0:1'  # Use both GPUs on Zeus
        self._devices_config = self.device
        self.model = None
        self._fitted = False
        self._gpu_info: Dict[str, Any] = {}
        self._gpu_verification: Dict[str, Any] = {}
        
        logger.info(f"CatBoostWrapper initialized for devices {self.device}")
    
    @property
    def model_type(self) -> str:
        return self._model_type
    
    def _verify_multi_gpu_usage(self) -> Dict[str, Any]:
        """
        Verify GPU usage from CatBoost.
        
        NOTE: CatBoost doesn't expose per-GPU utilization via Python API.
        This reports configuration and memory, not verified utilization.
        """
        verification = {
            "devices_requested": self._devices_config,
            "verification_method": "configuration_only",
            "verified_via_logs": False,
            "memory_report": self.log_gpu_memory().get("memory_report", []),
            "warnings": []
        }
        
        verification["warnings"].append(
            "GPU utilization not directly verifiable via CatBoost Python API. "
            "Check CatBoost verbose output or nvidia-smi for actual GPU usage."
        )
        
        return verification
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'CatBoostWrapper':
        """
        Train CatBoost with multi-GPU acceleration.
        
        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training targets, shape (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            self for method chaining
        """
        # CatBoost parameters with multi-GPU acceleration
        params = {
            'task_type': 'GPU',
            'devices': self._devices_config,
            'iterations': self.config.get('iterations', 100),
            'depth': self.config.get('depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'subsample': self.config.get('subsample', 0.8),
            'random_seed': 42,
            'verbose': 100  # Log every 100 iterations for verification
        }
        
        logger.info(f"Training CatBoost with multi-GPU on devices {self._devices_config}")
        logger.info(f"  iterations: {params['iterations']}")
        logger.info(f"  depth: {params['depth']}")
        
        self.model = CatBoostRegressor(**params)
        
        # Prepare eval set
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set
        )
        
        self._fitted = True
        self._gpu_info = self.log_gpu_memory()
        self._gpu_verification = self._verify_multi_gpu_usage()
        
        logger.info("CatBoost training complete")
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
        Save model to CBM (CatBoost binary) format.
        
        CBM format is efficient and preserves all model information.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")
        self.model.save_model(path)
        logger.info(f"CatBoost model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'CatBoostWrapper':
        """
        Load model from CBM file.
        
        Args:
            path: Path to .cbm model file
            device: Device for predictions (default: '0:1' for multi-GPU)
            
        Returns:
            Loaded CatBoostWrapper instance
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        wrapper = cls(device=device)
        wrapper.model = CatBoostRegressor()
        wrapper.model.load_model(path)
        wrapper._fitted = True
        
        logger.info(f"CatBoost model loaded from {path}")
        return wrapper
