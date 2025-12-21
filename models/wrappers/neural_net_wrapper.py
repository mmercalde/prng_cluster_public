"""
Neural Network Wrapper (v3.1.2)

Wraps existing SurvivorQualityNet from reinforcement_engine.py.
Runs on ALL 26 GPUs (ROCm + CUDA via PyTorch).

This wrapper does NOT duplicate the model - it imports and uses
the existing SurvivorQualityNet class.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.wrappers.base import ModelInterface, TorchModelMixin
from models.gpu_memory import GPUMemoryMixin

logger = logging.getLogger(__name__)

# Import existing model from reinforcement_engine
import sys
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from reinforcement_engine import SurvivorQualityNet


class NeuralNetWrapper(TorchModelMixin, GPUMemoryMixin):
    """
    Wrapper for existing SurvivorQualityNet.
    
    Runs on ALL 26 GPUs (ROCm + CUDA) via PyTorch.
    Implements ModelInterface protocol.
    """
    
    _model_type: str = "neural_net"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        """
        Initialize neural network wrapper.
        
        Args:
            config: Model configuration dict with:
                - input_size: Number of input features (default: 50)
                - hidden_layers: List of hidden layer sizes (default: [128, 64, 32])
                - dropout: Dropout rate (default: 0.3)
                - learning_rate: Learning rate (default: 0.001)
                - batch_size: Training batch size (default: 256)
                - epochs: Number of training epochs (default: 100)
            device: Device string (default: 'cuda' if available else 'cpu')
        """
        self.config = config or {}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        input_size = self.config.get('input_size', 50)
        hidden_layers = self.config.get('hidden_layers', [128, 64, 32])
        dropout = self.config.get('dropout', 0.3)
        
        # Create model using existing SurvivorQualityNet
        self.model = SurvivorQualityNet(
            input_size=input_size,
            hidden_layers=hidden_layers,
            dropout=dropout
        ).to(torch.device(self.device))
        
        self._fitted = False
        self._gpu_info: Dict[str, Any] = {}
        
        logger.info(f"NeuralNetWrapper initialized on {self.device}")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Hidden layers: {hidden_layers}")
        logger.info(f"  Dropout: {dropout}")
    
    @property
    def model_type(self) -> str:
        return self._model_type
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'NeuralNetWrapper':
        """
        Train the neural network.
        
        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training targets, shape (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation targets
            
        Returns:
            self for method chaining
        """
        device = torch.device(self.device)
        
        # Convert to tensors
        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
        
        if X_val is not None and y_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)
        else:
            X_val_t, y_val_t = None, None
        
        # Training parameters
        learning_rate = self.config.get('learning_rate', 0.001)
        batch_size = self.config.get('batch_size', 256)
        epochs = self.config.get('epochs', 100)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        logger.info(f"Training for {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
        
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_t), batch_size):
                batch_X = X_t[i:i+batch_size]
                batch_y = y_t[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            # Validation
            if X_val_t is not None and y_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                self.model.train()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}")
        
        self._fitted = True
        self._gpu_info = self.log_gpu_memory()
        
        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples,), values in [0, 1]
        """
        if not self._fitted:
            logger.warning("Model not fitted, predictions may be unreliable")
        
        self.model.eval()
        device = torch.device(self.device)
        
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            outputs = self.model(X_t)
            return outputs.cpu().numpy().flatten()
    
    def save(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Saves PyTorch state dict along with config and metadata.
        """
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'config': self.config,
            'model_type': self.model_type,
            'fitted': self._fitted,
            'gpu_info': self._gpu_info
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'NeuralNetWrapper':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to .pth checkpoint file
            device: Device to load model to
            
        Returns:
            Loaded NeuralNetWrapper instance
        """
        checkpoint = torch.load(path, map_location=device or 'cpu', weights_only=False)
        
        wrapper = cls(config=checkpoint.get('config', {}), device=device)
        wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        wrapper._fitted = checkpoint.get('fitted', True)
        wrapper._gpu_info = checkpoint.get('gpu_info', {})
        
        logger.info(f"Model loaded from {path}")
        return wrapper
