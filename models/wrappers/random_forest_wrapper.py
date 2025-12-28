"""
Random Forest Wrapper for Multi-Model Architecture
"""
import joblib
import numpy as np
from typing import Optional, Dict, Any

class RandomForestWrapper:
    """Wrapper for sklearn RandomForestRegressor"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = 'cpu'):
        self.model = None
        self.config = config or {}
        self.device = device  # RF is CPU-only but keep interface consistent
    
    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(**self.config)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not fitted or loaded")
        return self.model.predict(X)
    
    def save(self, path: str):
        if not path.endswith('.joblib'):
            path = f"{path}.joblib"
        joblib.dump(self.model, path)
        return path
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'RandomForestWrapper':
        """
        Load model from file.
        
        Args:
            path: Path to .joblib model file
            device: Device for predictions (ignored, RF is CPU-only)
            
        Returns:
            Loaded RandomForestWrapper instance
        """
        wrapper = cls(device=device or 'cpu')
        wrapper.model = joblib.load(path)
        return wrapper
