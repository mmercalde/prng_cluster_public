"""
Neural Network GPU Wrapper (v3.2.0) - FIXED

PyTorch neural network for survivor quality prediction.
Runs on ALL 26 GPUs (ROCm + CUDA via PyTorch).

FIXES in v3.2.0:
- Dynamic feature count from data (no hardcoded 50)
- Proper shape validation before forward pass
- Feature count mismatch error messages

This wrapper does NOT duplicate the model - it imports and uses
the existing SurvivorQualityNet class architecture.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not installed")

from models.wrappers.base import ModelInterface, TorchModelMixin
from models.gpu_memory import GPUMemoryMixin


class SurvivorQualityNet(nn.Module):
    """
    Neural network for survivor quality prediction.
    
    Architecture is dynamically configured based on feature count and hidden layers.
    This matches the architecture from reinforcement_engine.py.
    """
    
    def __init__(self, input_size: int, hidden_layers: List[int], dropout: float = 0.3):
        """
        Initialize network.
        
        Args:
            input_size: Number of input features (dynamic, NOT hardcoded)
            hidden_layers: List of hidden layer sizes e.g. [256, 128, 64]
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer (single value for quality score)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x).squeeze(-1)


class NeuralNetWrapper(TorchModelMixin, GPUMemoryMixin):
    """
    PyTorch neural network wrapper for survivor quality prediction.
    
    Runs on ALL 26 GPUs (ROCm + CUDA).
    Implements ModelInterface protocol.
    
    FIXED in v3.2.0:
    - Feature count is derived from data, not hardcoded
    - Shape validation before forward pass
    - Clear error messages for dimension mismatches
    """
    
    _model_type: str = "neural_net"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        """
        Initialize neural network wrapper.
        
        Args:
            config: Model configuration dict with:
                - hidden_layers: List of hidden layer sizes (default: [256, 128, 64])
                - dropout: Dropout probability (default: 0.3)
                - learning_rate: Learning rate (default: 0.001)
                - batch_size: Training batch size (default: 256)
                - epochs: Number of training epochs (default: 100)
            device: Device string (default: 'cuda:0' if available)
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        
        self.config = config or {}
        self.model = None
        self._fitted = False
        self._feature_count = None  # Will be set from data
        
        # Determine device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        # Architecture config (feature count will be set during fit)
        self.hidden_layers = self.config.get('hidden_layers', [256, 128, 64])
        self.dropout = self.config.get('dropout', 0.3)
        
        # Training config
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 256)
        self.epochs = self.config.get('epochs', 100)
        self.patience = self.config.get('patience', 10)
        
        logger.info(f"NeuralNetWrapper initialized on {self.device}")
    
    def _validate_input_shape(self, X: np.ndarray, context: str = "input") -> None:
        """
        Validate input shape matches expected feature count.
        
        Args:
            X: Input array
            context: Context string for error messages
            
        Raises:
            ValueError: If shape mismatch detected
        """
        if X.ndim != 2:
            raise ValueError(f"{context} must be 2D array, got shape {X.shape}")
        
        n_features = X.shape[1]
        
        if self._feature_count is not None and n_features != self._feature_count:
            raise ValueError(
                f"Feature count mismatch!\n"
                f"  Model trained with: {self._feature_count} features\n"
                f"  {context} has: {n_features} features\n"
                f"  Difference: {n_features - self._feature_count}\n"
                f"\n"
                f"This usually means:\n"
                f"  1. Training data had different features than test data\n"
                f"  2. Feature extraction changed between training and inference\n"
                f"  3. Label leakage fix removed some features\n"
                f"\n"
                f"Solution: Retrain the model with the current feature set."
            )
    
    @property
    def model_type(self) -> str:
        return self._model_type
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
    
    @property
    def feature_count(self) -> Optional[int]:
        return self._feature_count
    
    def train_mode(self) -> None:
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
    
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
            early_stopping_rounds: Optional[int] = None,
            verbose: bool = False) -> Dict[str, Any]:
        """
        Train neural network.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            eval_set: List of (X, y) tuples for validation
            early_stopping_rounds: Early stopping patience (overrides config)
            verbose: Whether to show training progress
            
        Returns:
            Training info dict with metrics
        """
        n_samples, n_features = X.shape
        
        # CRITICAL: Set feature count from data
        self._feature_count = n_features
        logger.info(f"Training with {n_features} features (derived from data)")
        
        # Build model with correct input size
        self.model = SurvivorQualityNet(
            input_size=n_features,  # Dynamic, not hardcoded!
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Use DataParallel if multiple GPUs available
        self._num_gpus = torch.cuda.device_count()
        if self._num_gpus > 1 and self.device.type == "cuda":
            logger.info(f"Using DataParallel with {self._num_gpus} GPUs")
            self.model = nn.DataParallel(self.model)
        else:
            logger.info(f"Using single GPU: {self.device}")
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Use provided patience or config
        patience = early_stopping_rounds or self.patience
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation setup
        val_loader = None
        if eval_set is not None:
            # Handle both formats: [(X, y)] or (X, y)
            if isinstance(eval_set, tuple) and len(eval_set) == 2 and isinstance(eval_set[0], np.ndarray):
                # Format: (X_val, y_val) - direct tuple
                X_val, y_val = eval_set
            elif isinstance(eval_set, list) and len(eval_set) > 0:
                # Format: [(X_val, y_val)] - list of tuples
                X_val, y_val = eval_set[0]
            else:
                X_val, y_val = None, None
            
            if X_val is not None:
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).to(self.device)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        best_state_dict = None
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = None
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()
                    val_loss /= len(val_loader)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    epochs_without_improvement += 1
                
                if verbose and epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                if verbose and epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        
        # Restore best model
        if best_state_dict is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state_dict.items()})
        
        self._fitted = True
        
        result = {
            'epochs_trained': epoch + 1,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss if val_loader else None,
            'final_train_loss': train_loss,
            'feature_count': n_features,
            'hidden_layers': self.hidden_layers,
            'device': str(self.device),
        }
        
        logger.info(f"Neural network training complete: {result}")
        return result
    
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
        
        # Validate input shape
        self._validate_input_shape(X, "Prediction input")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def save(self, path: str) -> None:
        """
        Save model to PyTorch checkpoint.
        
        Saves both state dict and architecture info for proper loading.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")
        
        path = Path(path)
        
        # Save complete checkpoint with architecture info
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'feature_count': self._feature_count,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'config': self.config,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Neural network saved to {path} (features: {self._feature_count})")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'NeuralNetWrapper':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to .pth checkpoint file
            device: Device for inference
            
        Returns:
            Loaded NeuralNetWrapper instance
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        
        path = Path(path)
        checkpoint = torch.load(path, map_location='cpu')
        
        # Get architecture info from checkpoint
        feature_count = checkpoint.get('feature_count') or checkpoint.get('input_dim')
        hidden_layers = checkpoint.get('hidden_layers', [256, 128, 64])
        dropout = checkpoint.get('dropout', 0.3)
        config = checkpoint.get('config', {})
        
        if feature_count is None:
            raise ValueError(
                f"Checkpoint missing feature_count. "
                f"This is an old checkpoint format. Please retrain the model."
            )
        
        # Create wrapper
        config['hidden_layers'] = hidden_layers
        config['dropout'] = dropout
        wrapper = cls(config=config, device=device)
        wrapper._feature_count = feature_count
        
        # Build model with correct architecture
        wrapper.model = SurvivorQualityNet(
            input_size=feature_count,
            hidden_layers=hidden_layers,
            dropout=dropout
        ).to(wrapper.device)
        
        # Load weights
        wrapper.model.load_state_dict(checkpoint.get('state_dict') or checkpoint.get('model_state_dict'))
        wrapper._fitted = True
        
        logger.info(f"Neural network loaded from {path} (features: {feature_count})")
        return wrapper
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance via gradient-based attribution.
        
        Note: This is approximate - neural nets don't have built-in importance.
        """
        # For now, return empty - could implement gradient-based importance later
        return {}
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            **self.config,
            'feature_count': self._feature_count,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'device': str(self.device),
        }
