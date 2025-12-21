"""
Base Model Interface (v3.1.2)

Defines the protocol that ALL model wrappers must implement.
Separates required methods (all models) from optional methods (PyTorch only).

Required Methods (ALL models):
- fit(X_train, y_train, X_val, y_val) -> self
- predict(X) -> np.ndarray
- save(path) -> None
- load(path, device) -> ModelInterface (classmethod)
- model_type (property)

Optional Methods (PyTorch only):
- train_mode() -> None
- eval_mode() -> None
- state_dict() -> Dict
- load_state_dict(state) -> None
"""

from typing import Protocol, Dict, Any, Optional, runtime_checkable
import numpy as np


@runtime_checkable
class ModelInterface(Protocol):
    """
    Required interface - ALL models must implement these methods.
    
    This is a Protocol class, meaning any class that implements these
    methods is considered compatible, without explicit inheritance.
    """
    
    @property
    def model_type(self) -> str:
        """Return model type identifier (e.g., 'neural_net', 'xgboost')."""
        ...
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'ModelInterface':
        """
        Train the model.
        
        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training targets, shape (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            self for method chaining
        """
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples,)
        """
        ...
    
    def save(self, path: str) -> None:
        """
        Save model to file.
        
        Format depends on model type:
        - neural_net: .pth (PyTorch state dict)
        - xgboost: .json (XGBoost JSON format)
        - lightgbm: .txt (LightGBM text format)
        - catboost: .cbm (CatBoost binary format)
        """
        ...
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'ModelInterface':
        """
        Load model from file.
        
        Args:
            path: Path to saved model file
            device: Device to load model to (e.g., 'cuda:0', 'cpu')
            
        Returns:
            Loaded model instance
        """
        ...


class TorchModelMixin:
    """
    Optional mixin for PyTorch-specific methods.
    
    XGBoost/LightGBM/CatBoost wrappers do NOT implement these methods.
    Use hasattr() to check availability before calling.
    
    Example:
        if hasattr(model, 'eval_mode'):
            model.eval_mode()
    """
    
    def train_mode(self) -> None:
        """Set model to training mode. PyTorch: self.model.train()"""
        if hasattr(self, 'model') and hasattr(self.model, 'train'):
            self.model.train()
    
    def eval_mode(self) -> None:
        """Set model to evaluation mode. PyTorch: self.model.eval()"""
        if hasattr(self, 'model') and hasattr(self.model, 'eval'):
            self.model.eval()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get model state dictionary. PyTorch only."""
        if hasattr(self, 'model') and hasattr(self.model, 'state_dict'):
            return self.model.state_dict()
        raise NotImplementedError("state_dict not available for this model type")
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load model state dictionary. PyTorch only."""
        if hasattr(self, 'model') and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(state)
        else:
            raise NotImplementedError("load_state_dict not available for this model type")
