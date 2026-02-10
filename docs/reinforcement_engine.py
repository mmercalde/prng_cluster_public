#!/usr/bin/env python3
"""
Reinforcement Engine - ML Training Orchestrator (DISTRIBUTED-READY)
====================================================================

Version: 1.7.0 - CHAPTER 14 TRAINING DIAGNOSTICS INTEGRATION

Changes in v1.7.0:
- Added enable_diagnostics parameter for Chapter 14 live training introspection
- Integrated NNDiagnostics PyTorch hooks for per-epoch capture
- Best-effort, non-fatal diagnostics (never blocks training)
- Diagnostics auto-save to diagnostics_outputs/ after training
- Added --enable-diagnostics CLI flag

Changes in v1.6.1:
- Fixed save spam: now saves best_model.pth only ONCE at end of training
- No more "Model saved" log spam during training loop

Changes in v1.6.0:
- CUDA warmup fix: torch.cuda.set_device() + tensor warmup before any ops
- Updated default feature count: 48 per-seed (removed score, confidence)
- Best model overwrite: saves to best_model.pth only (no epoch spam)
- Added --save-all-epochs flag for explicit checkpoint snapshots
- Improved startup logging (device, GPU count, feature dimensions)
- Suppressed cuBLAS warning via proper initialization order

Previous versions:
- v1.5.0: Pre-computed feature support (dict detection)
- v1.4.0: Optuna pruning hooks
- v1.3.0: Stateless train/predict with lottery_history override
- v1.2.0: ROCm support for AMD GPUs
- v1.1.0: DDP distributed training
"""

import sys
import os
import json
import logging
import warnings

# Suppress cuBLAS warning (we handle CUDA init properly now)
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

# ============================================================================
# ROCm environment setup - MUST BE BEFORE TORCH IMPORT
# ============================================================================
import socket
HOST = socket.gethostname()
if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")
# ============================================================================

import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import Counter, deque
import hashlib

import numpy as np
from scipy.stats import entropy

# GlobalStateTracker - imported from GPU-neutral module (Step 6 Restoration v2.2)
from models.global_state_tracker import GlobalStateTracker

# PyTorch for neural network (NOW SAFE TO IMPORT - AFTER ROCm SETUP)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not available! Install with: pip install torch")
    sys.exit(1)

# DDP support (optional)
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

# Import from pipeline
try:
    from survivor_scorer import SurvivorScorer
    SURVIVOR_SCORER_AVAILABLE = True
except ImportError:
    print("ERROR: Cannot import survivor_scorer.py!")
    print("Make sure survivor_scorer.py is in the same directory")
    sys.exit(1)

# ============================================================================
# TRAINING DIAGNOSTICS (Chapter 14 - v1.7.0)
# ============================================================================
try:
    from training_diagnostics import NNDiagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    NNDiagnostics = None  # Placeholder for type hints


# ============================================================================
# CUDA INITIALIZATION (v1.6.0 - PROPER WARMUP)
# ============================================================================

def initialize_cuda(device: torch.device = None, logger: logging.Logger = None) -> bool:
    """
    Initialize CUDA context with proper warmup.
    
    v1.6.0: Fixes "no current CUDA context" warning by:
    1. Explicitly setting device before any CUDA ops
    2. Allocating a small tensor to force context creation
    3. Synchronizing to ensure context is fully initialized
    
    Args:
        device: Target device (default: cuda:0)
        logger: Optional logger for status messages
        
    Returns:
        True if CUDA initialized successfully, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Default to cuda:0 if not specified
        if device is None:
            device = torch.device('cuda:0')
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Step 1: Explicitly set device
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        
        # Step 2: Force context creation with small allocation
        _ = torch.empty(1, device=device)
        
        # Step 3: Synchronize to ensure context is ready
        torch.cuda.synchronize(device)
        
        if logger:
            logger.info(f"✅ CUDA context initialized on {device}")
            logger.info(f"   Device: {torch.cuda.get_device_name(device)}")
            logger.info(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        
        return True
        
    except RuntimeError as e:
        if "busy or unavailable" in str(e):
            # CUDA already initialized - that's fine
            return True
        else:
            if logger:
                logger.warning(f"CUDA initialization failed: {e}")
            return False


# Global flag - but we now init per-engine with proper device
CUDA_INITIALIZED = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ReinforcementConfig:
    """
    Configuration for reinforcement engine
    
    v1.7.0: Added diagnostics config block
    v1.6.0: Updated default input_features to 48 (removed score, confidence)
    """
    # Model architecture
    model: Dict[str, Any] = field(default_factory=lambda: {
        'input_features': 48,  # v1.6.0: 48 features (removed score, confidence)
        'hidden_layers': [128, 64, 32],
        'dropout': 0.3,
        'activation': 'relu',
        'output_activation': 'sigmoid'
    })

    # Training parameters
    training: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 0.001,
        'batch_size': 256,
        'epochs': 100,
        'optimizer': 'adam',
        'loss_function': 'mse',
        'early_stopping_patience': 10,
        'validation_split': 0.2,
        'save_best_only': True,  # v1.6.0: Only save best model (no epoch spam)
        'save_all_epochs': False,  # v1.6.0: Explicit flag for epoch snapshots
        'save_frequency': 10,
        'verbose_frequency': 10
    })

    # PRNG parameters
    prng: Dict[str, Any] = field(default_factory=lambda: {
        'prng_type': 'java_lcg',
        'mod': 1000,
        'skip': 0
    })

    # Global state tracking
    global_state: Dict[str, Any] = field(default_factory=lambda: {
        'window_size': 1000,
        'anomaly_threshold': 3.0,
        'regime_change_threshold': 0.15,
        'marker_numbers': [390, 804, 575],
        'variance_threshold': 1.0,
        'gap_threshold': 500,
        'frequency_bias_threshold': 3.0
    })

    # Feature normalization
    normalization: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'method': 'standard',
        'auto_fit': True,
        'per_feature': True,
        'save_with_model': True,
        'refit_on_drift': True,
        'drift_threshold': 0.3
    })

    # Survivor management
    survivor_pool: Dict[str, Any] = field(default_factory=lambda: {
        'max_pool_size': 10000,
        'prune_threshold': 0.3,
        'min_confidence': 0.5,
        'update_frequency': 10
    })

    # Output and persistence
    output: Dict[str, Any] = field(default_factory=lambda: {
        'models_dir': 'models/reinforcement',
        'logs_dir': 'logs/reinforcement',
        'save_frequency': 100,
        'log_level': 'INFO'
    })

    # Distributed training settings
    distributed: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'backend': 'nccl',
        'init_method': 'env://',
        'world_size': None,
        'rank': None,
        'local_rank': None,
        'find_unused_parameters': False
    })

    # v1.7.0: Training diagnostics (Chapter 14)
    diagnostics: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'capture_every_n': 5,
        'output_dir': 'diagnostics_outputs',
        'save_history': True
    })

    @classmethod
    def from_json(cls, path: str) -> 'ReinforcementConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            model=data.get('model', {}),
            training=data.get('training', {}),
            prng=data.get('prng', {}),
            global_state=data.get('global_state', {}),
            normalization=data.get('normalization', {}),
            survivor_pool=data.get('survivor_pool', {}),
            output=data.get('output', {}),
            distributed=data.get('distributed', {}),
            diagnostics=data.get('diagnostics', {})  # v1.7.0
        )

    def to_json(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# ============================================================================
# GLOBAL STATE TRACKER
# ============================================================================


# ============================================================================
# PYTORCH NEURAL NETWORK
# ============================================================================

class SurvivorQualityNet(nn.Module):
    """PyTorch neural network for survivor quality prediction"""

    def __init__(self, input_size: int = 62, hidden_layers: List[int] = [128, 64, 32],
                 dropout: float = 0.3):
        super(SurvivorQualityNet, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# REINFORCEMENT ENGINE
# ============================================================================

class ReinforcementEngine:
    """
    Main ML training orchestrator
    
    v1.7.0: Chapter 14 training diagnostics integration
    v1.6.0: CUDA warmup fix, updated feature count, reduced checkpoint spam
    """

    def __init__(self, config: ReinforcementConfig, lottery_history: List[int],
                 logger: Optional[logging.Logger] = None,
                 scorer_config_dict: Optional[Dict] = None,
                 per_seed_feature_count: Optional[int] = None,
                 excluded_features: Optional[List[str]] = None,
                 enable_diagnostics: bool = False):
        """
        Initialize ReinforcementEngine.
        
        Args:
            config: ReinforcementConfig instance
            lottery_history: List of lottery draw integers
            logger: Optional logger instance
            scorer_config_dict: Optional custom scorer parameters
            per_seed_feature_count: Override for per-seed feature count (default: 48)
            excluded_features: Features excluded from training (for logging)
            enable_diagnostics: Enable Chapter 14 training diagnostics (v1.7.0)
        """
        self.config = config
        self.lottery_history = lottery_history
        self.logger = logger or self._setup_logger()
        self.excluded_features = excluded_features or ['score', 'confidence']

        self.logger.info("="*60)
        self.logger.info("Initializing ReinforcementEngine v1.7.0")
        self.logger.info("="*60)

        # Distributed training state
        self.is_distributed = config.distributed.get('enabled', False)
        self.rank = None
        self.world_size = None
        self.local_rank = None

        if self.is_distributed:
            self._init_distributed()
        else:
            self.logger.info("Mode: LOCAL (standard multi-GPU)")

        # Device selection (BEFORE CUDA init)
        self.device = self._get_device()

        # v1.6.0: CUDA warmup with proper initialization
        global CUDA_INITIALIZED
        if not CUDA_INITIALIZED and torch.cuda.is_available():
            CUDA_INITIALIZED = initialize_cuda(self.device, self.logger)

        # Log GPU info
        self._log_gpu_info()

        # Survivor scorer (for legacy mode)
        self.scorer = SurvivorScorer(
            prng_type=config.prng['prng_type'],
            mod=config.prng['mod'],
            config_dict=scorer_config_dict
        )

        if scorer_config_dict:
            self.logger.info(f"Survivor scorer: Custom trial parameters")
        else:
            self.logger.info(f"Survivor scorer: {config.prng['prng_type']}, mod={config.prng['mod']}")

        # Global state tracker
        self.global_tracker = GlobalStateTracker(
            lottery_history=lottery_history,
            config=config.global_state
        )
        self.logger.info(f"Global state tracker: {len(lottery_history)} draws")

        # v1.6.0: Dynamic feature count (default 48 after label leakage fix)
        global_state = self.global_tracker.get_global_state()
        self.global_feature_count = len(global_state)
        
        if per_seed_feature_count is not None:
            self.per_seed_feature_count = per_seed_feature_count
        else:
            self.per_seed_feature_count = config.model.get('input_features', 48)
        
        total_input_size = self.per_seed_feature_count + self.global_feature_count

        # v1.6.0: Clear feature dimension logging
        self.logger.info("-"*60)
        self.logger.info("FEATURE CONFIGURATION")
        self.logger.info("-"*60)
        self.logger.info(f"  Feature mode: PRECOMPUTED")
        self.logger.info(f"  Excluded features: {self.excluded_features}")
        self.logger.info(f"  Per-seed features: {self.per_seed_feature_count}")
        self.logger.info(f"  Global features: {self.global_feature_count}")
        self.logger.info(f"  Total input dim: {total_input_size}")
        self.logger.info("-"*60)

        # Create model
        self.model = SurvivorQualityNet(
            input_size=total_input_size,
            hidden_layers=config.model['hidden_layers'],
            dropout=config.model['dropout']
        ).to(self.device)

        # Model wrapping
        self._wrap_model()

        self.logger.info(f"Neural network: {total_input_size} → {config.model['hidden_layers']} → 1")

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training['learning_rate']
        )

        # Training state
        self.criterion = nn.MSELoss()
        self.training_history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'best_epoch': None,
            'best_val_loss': float('inf')
        }
        self.best_val_loss = float('inf')
        self.best_overfit_ratio = float('inf')
        self.feature_scaler = StandardScaler() if config.normalization.get('enabled', True) else None
        self.scaler_fitted = False
        self.normalization_enabled = config.normalization.get('enabled', True)
        self.feature_stats = {'means': None, 'stds': None, 'n_samples': 0}
        self.per_seed_feature_names: Optional[List[str]] = None

        if self.normalization_enabled:
            self.logger.info("Feature normalization: ENABLED")

        # v1.7.0: Training diagnostics (Chapter 14)
        config_diagnostics_enabled = config.diagnostics.get('enabled', False)
        self.enable_diagnostics = (enable_diagnostics or config_diagnostics_enabled) and DIAGNOSTICS_AVAILABLE
        self._diagnostics: Optional['NNDiagnostics'] = None
        self._last_diagnostics_path: Optional[str] = None
        
        if self.enable_diagnostics:
            self.logger.info("Training diagnostics: ENABLED (Chapter 14)")
        elif enable_diagnostics and not DIAGNOSTICS_AVAILABLE:
            self.logger.warning("Training diagnostics requested but training_diagnostics.py not found")

        Path(config.output['models_dir']).mkdir(parents=True, exist_ok=True)
        Path(config.output['logs_dir']).mkdir(parents=True, exist_ok=True)

        self.logger.info("="*60)
        self.logger.info("ReinforcementEngine initialized successfully!")
        self.logger.info("="*60)

    def _log_gpu_info(self):
        """Log GPU configuration details."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.logger.info(f"CUDA available: Yes")
            self.logger.info(f"GPU count: {gpu_count}")
            self.logger.info(f"Selected device: {self.device}")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                self.logger.info(f"  GPU {i}: {name} ({mem:.1f} GB)")
        else:
            self.logger.info("CUDA available: No (using CPU)")

    def _init_distributed(self):
        """Initialize distributed training"""
        if not DDP_AVAILABLE:
            self.logger.error("DDP not available")
            raise RuntimeError("DDP not available")

        self.rank = int(os.environ.get('RANK', self.config.distributed.get('rank', 0)))
        self.world_size = int(os.environ.get('WORLD_SIZE', self.config.distributed.get('world_size', 1)))
        self.local_rank = int(os.environ.get('LOCAL_RANK', self.config.distributed.get('local_rank', 0)))

        backend = self.config.distributed.get('backend', 'nccl')
        init_method = self.config.distributed.get('init_method', 'env://')

        self.logger.info(f"Mode: DISTRIBUTED")
        self.logger.info(f"  Rank: {self.rank}/{self.world_size}")
        self.logger.info(f"  Local rank: {self.local_rank}")
        self.logger.info(f"  Backend: {backend}")

        try:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=self.rank,
                world_size=self.world_size
            )
            self.logger.info("✅ DDP initialized successfully")
        except Exception as e:
            self.logger.error(f"DDP initialization failed: {e}")
            raise

    def _get_device(self) -> torch.device:
        """Get appropriate device"""
        if not torch.cuda.is_available():
            return torch.device('cpu')

        if self.is_distributed:
            return torch.device(f'cuda:{self.local_rank}')
        else:
            return torch.device('cuda:0')

    def _wrap_model(self):
        """Wrap model with DDP or DataParallel"""
        if self.is_distributed:
            self.logger.info(f"Wrapping model with DistributedDataParallel...")
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.distributed.get('find_unused_parameters', False)
            )
            self.logger.info(f"✅ DDP wrapper active (rank {self.rank})")
        elif torch.cuda.device_count() > 1 and os.environ.get('CUDA_VISIBLE_DEVICES') is None:
            self.logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            device_ids = list(range(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
        else:
            self.logger.info(f"Using single device: {self.device}")

    def _setup_logger(self) -> logging.Logger:
        log_level = self.config.output.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def extract_combined_features(
        self, 
        survivor: Union[int, Dict[str, Any]],
        forward_survivors: Optional[List[int]] = None,
        reverse_survivors: Optional[List[int]] = None,
        lottery_history: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Extract combined features (per-seed + global state).
        
        Accepts either:
        - int (seed): Legacy mode - extracts via SurvivorScorer
        - dict with 'features': Uses pre-computed features directly
        """
        if isinstance(survivor, dict) and 'features' in survivor:
            features_dict = survivor['features']
            feature_names = sorted(features_dict.keys())
            per_seed_values = [float(features_dict.get(k, 0.0)) for k in feature_names]
            
            if self.per_seed_feature_names is None:
                self.per_seed_feature_names = feature_names
                self.per_seed_feature_count = len(feature_names)
        else:
            seed = survivor if isinstance(survivor, int) else int(survivor.get('seed', survivor))
            history_to_use = lottery_history if lottery_history is not None else self.lottery_history

            per_seed_features = self.scorer.extract_ml_features(
                seed=seed,
                lottery_history=history_to_use,
                forward_survivors=forward_survivors,
                reverse_survivors=reverse_survivors
            )

            # Filter out excluded features (score, confidence)
            feature_names = sorted([k for k in per_seed_features.keys() if k not in self.excluded_features])
            per_seed_values = [per_seed_features[k] for k in feature_names]
            
            if self.per_seed_feature_names is None:
                self.per_seed_feature_names = feature_names

        global_state = self.global_tracker.get_global_state()
        global_names = sorted(global_state.keys())
        global_values = [global_state[k] for k in global_names]

        combined = np.array(per_seed_values + global_values, dtype=np.float32)

        if self.normalization_enabled and self.scaler_fitted:
            original_combined = combined.copy()
            normalized = self.feature_scaler.transform([combined])[0]
            zero_var_mask = self.feature_scaler.scale_ == 1.0
            if np.any(zero_var_mask):
                normalized[zero_var_mask] = (original_combined[zero_var_mask] -
                                            self.feature_scaler.mean_[zero_var_mask])
            combined = normalized.astype(np.float32)

        return combined

    def train(
        self, 
        survivors: List[Union[int, Dict[str, Any]]],
        actual_results: List[float],
        forward_survivors: Optional[List[int]] = None,
        reverse_survivors: Optional[List[int]] = None,
        epochs: Optional[int] = None,
        lottery_history: Optional[List[int]] = None,
        epoch_callback: Optional[callable] = None
    ):
        """
        Train model.
        
        v1.7.0: Integrated Chapter 14 diagnostics hooks
        v1.6.0: Only saves best_model.pth (no epoch spam)
        """
        if len(survivors) == 0 or len(actual_results) == 0:
            self.logger.warning("No training data provided")
            return

        epochs = epochs or self.config.training['epochs']
        batch_size = self.config.training['batch_size']
        save_best_only = self.config.training.get('save_best_only', True)
        save_all_epochs = self.config.training.get('save_all_epochs', False)
        verbose_freq = self.config.training.get('verbose_frequency', 10)
        should_log = not self.is_distributed or self.rank == 0

        using_precomputed = isinstance(survivors[0], dict) and 'features' in survivors[0]
        
        if should_log:
            mode = "pre-computed features" if using_precomputed else "scorer extraction"
            self.logger.info(f"Training on {len(survivors)} survivors for {epochs} epochs ({mode})...")

        # Normalization
        if self.normalization_enabled:
            if not self.scaler_fitted:
                if should_log:
                    self.logger.info("Fitting feature normalizer...")
                self._fit_normalizer(survivors, forward_survivors, reverse_survivors, lottery_history)

        # Extract features
        if should_log:
            self.logger.info("Extracting features...")
        X = []
        for survivor in survivors:
            features = self.extract_combined_features(
                survivor, forward_survivors, reverse_survivors,
                lottery_history=lottery_history
            )
            X.append(features)

        X = np.array(X, dtype=np.float32)
        y = np.array(actual_results, dtype=np.float32).reshape(-1, 1)

        # Train/val split
        val_split = self.config.training['validation_split']
        n_val = int(len(X) * val_split)
        indices = np.random.permutation(len(X))
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        X_train_t = torch.tensor(X_train).to(self.device)
        y_train_t = torch.tensor(y_train).to(self.device)
        X_val_t = torch.tensor(X_val).to(self.device)
        y_val_t = torch.tensor(y_val).to(self.device)

        # ====================================================================
        # v1.7.0: Attach diagnostics hooks (best-effort, non-fatal)
        # ====================================================================
        if self.enable_diagnostics and should_log:
            try:
                capture_every_n = self.config.diagnostics.get('capture_every_n', 5)
                self._diagnostics = NNDiagnostics(
                    feature_names=self.per_seed_feature_names,
                    capture_every_n=capture_every_n
                )
                # Get the underlying model if wrapped in DataParallel/DDP
                model_to_hook = self.model.module if hasattr(self.model, 'module') else self.model
                self._diagnostics.attach(model_to_hook)
                self.logger.info(f"✅ Diagnostics hooks attached (capture every {capture_every_n} epochs)")
            except Exception as e:
                self.logger.warning(f"Diagnostics attach failed (non-fatal): {e}")
                self._diagnostics = None
        # ====================================================================

        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            n_batches = len(X_train) // batch_size + (1 if len(X_train) % batch_size else 0)
            epoch_loss = 0.0

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                batch_X = X_train_t[start_idx:end_idx]
                batch_y = y_train_t[start_idx:end_idx]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = self.criterion(val_outputs, y_val_t).item()

            avg_loss = epoch_loss / n_batches

            self.training_history['epoch'].append(epoch)
            self.training_history['loss'].append(avg_loss)
            self.training_history['val_loss'].append(val_loss)

            # ================================================================
            # v1.7.0: Record diagnostics for this epoch (best-effort)
            # ================================================================
            if self._diagnostics is not None:
                try:
                    self._diagnostics.on_round_end(epoch, avg_loss, val_loss)
                except Exception:
                    pass  # Non-fatal - diagnostics failure never blocks training
            # ================================================================

            # Pruning callback
            if epoch_callback is not None:
                should_continue = epoch_callback(epoch, val_loss)
                if not should_continue:
                    if should_log:
                        self.logger.info(f"⚡ Training stopped by callback at epoch {epoch+1}")
                    break

            # Logging
            if should_log and (epoch + 1) % verbose_freq == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Best model tracking
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0

                # v1.6.1: Mark for save at end (no spam during training)
                if save_best_only:
                    self._pending_best_save = True
                    
                # v1.6.0: Optional epoch snapshots (explicit flag)
                if save_all_epochs and should_log:
                    self.save_model(f"best_model_epoch_{epoch+1}.pth")
            else:
                patience_counter += 1

                if patience_counter >= self.config.training['early_stopping_patience']:
                    if should_log:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Training summary
        self.training_history['best_epoch'] = best_epoch
        self.training_history['best_val_loss'] = best_val_loss
        self.best_val_loss = best_val_loss
        self.best_overfit_ratio = val_loss / avg_loss if avg_loss > 0 else float('inf')

        # ====================================================================
        # v1.7.0: Finalize and save diagnostics (best-effort)
        # ====================================================================
        if self._diagnostics is not None and should_log:
            try:
                self._diagnostics.detach()
                self._diagnostics.set_final_metrics({
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'final_train_loss': avg_loss,
                    'final_val_loss': val_loss,
                    'overfit_ratio': self.best_overfit_ratio,
                    'total_epochs': epoch + 1
                })
                diag_path = self._diagnostics.save()
                self._last_diagnostics_path = diag_path
                self.logger.info(f"✅ Training diagnostics saved to {diag_path}")
            except Exception as e:
                self.logger.warning(f"Diagnostics save failed (non-fatal): {e}")
            finally:
                self._diagnostics = None  # Clear reference
        # ====================================================================

        # v1.6.1: Save best model ONCE at end of training (no spam)
        if save_best_only and should_log and getattr(self, '_pending_best_save', False):
            self.save_model("best_model.pth")
            self._pending_best_save = False

        if should_log:
            self.logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
            self.logger.info(f"  Best epoch: {best_epoch}/{epochs}")
            self.logger.info(f"  Final train loss: {avg_loss:.4f}")
            self.logger.info(f"  Final val loss: {val_loss:.4f}")
            self.logger.info(f"  Overfit ratio: {val_loss / avg_loss:.2f}")

    def get_last_diagnostics_path(self) -> Optional[str]:
        """
        Get path to most recent diagnostics file (v1.7.0).
        
        Returns:
            Path to diagnostics JSON file, or None if diagnostics weren't generated
        """
        return self._last_diagnostics_path

    def predict_quality(
        self, 
        survivor: Union[int, Dict[str, Any]],
        forward_survivors: Optional[List[int]] = None,
        reverse_survivors: Optional[List[int]] = None,
        lottery_history: Optional[List[int]] = None
    ) -> float:
        """Predict quality score for a single survivor."""
        self.model.eval()

        with torch.no_grad():
            features = self.extract_combined_features(
                survivor, forward_survivors, reverse_survivors,
                lottery_history=lottery_history
            )
            features_t = torch.tensor(features).unsqueeze(0).to(self.device)
            quality = self.model(features_t).item()

        return quality

    def predict_quality_batch(
        self, 
        survivors: List[Union[int, Dict[str, Any]]],
        forward_survivors: Optional[List[int]] = None,
        reverse_survivors: Optional[List[int]] = None,
        lottery_history: Optional[List[int]] = None
    ) -> List[float]:
        """Predict quality scores for batch of survivors."""
        if not survivors:
            return []

        self.model.eval()

        X = []
        for survivor in survivors:
            features = self.extract_combined_features(
                survivor, forward_survivors, reverse_survivors,
                lottery_history=lottery_history
            )
            X.append(features)

        X = np.array(X, dtype=np.float32)
        X_t = torch.tensor(X).to(self.device)

        with torch.no_grad():
            qualities = self.model(X_t).cpu().numpy().flatten()

        return qualities.tolist()

    def prune_survivors(
        self, 
        survivors: List[Union[int, Dict[str, Any]]],
        keep_top_n: Optional[int] = None,
        forward_survivors: Optional[List[int]] = None,
        reverse_survivors: Optional[List[int]] = None
    ) -> List[Union[int, Dict[str, Any]]]:
        """Prune survivor pool to keep only top performers"""
        if not survivors:
            return []

        keep_top_n = keep_top_n or int(len(survivors) * self.config.survivor_pool['prune_threshold'])
        keep_top_n = max(1, min(keep_top_n, len(survivors)))

        self.logger.info(f"Pruning {len(survivors)} survivors to top {keep_top_n}...")

        qualities = self.predict_quality_batch(survivors, forward_survivors, reverse_survivors)
        ranked = sorted(zip(survivors, qualities), key=lambda x: x[1], reverse=True)
        top_survivors = [s for s, _ in ranked[:keep_top_n]]

        self.logger.info(f"Kept top {len(top_survivors)} survivors")
        return top_survivors

    def _fit_normalizer(
        self, 
        survivors: List[Union[int, Dict[str, Any]]],
        forward_survivors: Optional[List[int]] = None,
        reverse_survivors: Optional[List[int]] = None,
        lottery_history: Optional[List[int]] = None
    ):
        """Fit feature normalizer on survivor pool"""
        if not self.normalization_enabled:
            return

        temp_fitted = self.scaler_fitted
        self.scaler_fitted = False

        features_list = []
        for survivor in survivors:
            features = self.extract_combined_features(
                survivor, forward_survivors, reverse_survivors,
                lottery_history=lottery_history
            )
            features_list.append(features)

        features_array = np.array(features_list)
        self.feature_scaler.fit(features_array)
        self.scaler_fitted = True
        self.feature_stats['means'] = features_array.mean(axis=0)
        self.feature_stats['stds'] = features_array.std(axis=0)
        self.feature_stats['n_samples'] = len(features_array)

        mean_range = [self.feature_stats['means'].min(), self.feature_stats['means'].max()]
        std_range = [self.feature_stats['stds'].min(), self.feature_stats['stds'].max()]

        self.logger.info(f"✅ Normalizer fitted on {len(survivors)} survivors")
        self.logger.info(f"   Feature mean range: [{mean_range[0]:.2f}, {mean_range[1]:.2f}]")
        self.logger.info(f"   Feature std range: [{std_range[0]:.2f}, {std_range[1]:.2f}]")

    def save_model(self, filename: Optional[str] = None):
        """Save model state"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reinforcement_model_{timestamp}.pth"

        filepath = Path(self.config.output['models_dir']) / filename

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'normalization': {
                'scaler_fitted': self.scaler_fitted,
                'feature_stats': self.feature_stats,
                'scaler_params': {
                    'mean_': self.feature_scaler.mean_.tolist() if self.scaler_fitted and self.normalization_enabled else None,
                    'scale_': self.feature_scaler.scale_.tolist() if self.scaler_fitted and self.normalization_enabled else None,
                    'var_': self.feature_scaler.var_.tolist() if self.scaler_fitted and self.normalization_enabled else None,
                    'n_samples_seen_': int(self.feature_scaler.n_samples_seen_) if self.scaler_fitted and self.normalization_enabled else 0
                } if self.normalization_enabled else None
            },
            'feature_schema': {
                'per_seed_feature_count': self.per_seed_feature_count,
                'global_feature_count': self.global_feature_count,
                'per_seed_feature_names': self.per_seed_feature_names,
                'excluded_features': self.excluded_features
            },
            'distributed_info': {
                'is_distributed': self.is_distributed,
                'rank': self.rank,
                'world_size': self.world_size
            } if self.is_distributed else None
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)

        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.training_history = checkpoint.get('training_history', {})

        if 'normalization' in checkpoint and checkpoint['normalization']:
            norm_data = checkpoint['normalization']
            self.scaler_fitted = norm_data.get('scaler_fitted', False)
            self.feature_stats = norm_data.get('feature_stats', {'means': None, 'stds': None, 'n_samples': 0})

            if norm_data.get('scaler_params') and self.normalization_enabled:
                scaler_params = norm_data['scaler_params']
                if scaler_params['mean_'] is not None:
                    self.feature_scaler.mean_ = np.array(scaler_params['mean_'])
                    self.feature_scaler.scale_ = np.array(scaler_params['scale_'])
                    self.feature_scaler.var_ = np.array(scaler_params['var_'])
                    self.feature_scaler.n_samples_seen_ = scaler_params['n_samples_seen_']
                    self.logger.info("  Normalization scaler restored")

        if 'feature_schema' in checkpoint:
            schema = checkpoint['feature_schema']
            self.per_seed_feature_count = schema.get('per_seed_feature_count', self.per_seed_feature_count)
            self.global_feature_count = schema.get('global_feature_count', self.global_feature_count)
            self.per_seed_feature_names = schema.get('per_seed_feature_names')
            self.excluded_features = schema.get('excluded_features', self.excluded_features)

        self.logger.info(f"Model loaded from {filepath}")

    def cleanup_distributed(self):
        """Cleanup distributed resources"""
        if self.is_distributed and dist.is_initialized():
            self.logger.info("Cleaning up distributed resources...")
            dist.destroy_process_group()
            self.logger.info("✅ Distributed cleanup complete")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface for testing"""
    import argparse
    parser = argparse.ArgumentParser(
        description='Reinforcement Engine v1.7.0 - ML Training Orchestrator'
    )
    parser.add_argument('--config', type=str,
                       default='reinforcement_engine_config.json',
                       help='Configuration file')
    parser.add_argument('--lottery-data', type=str,
                       help='Lottery history JSON file')
    parser.add_argument('--test', action='store_true',
                       help='Run self-test')
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training mode')
    parser.add_argument('--enable-diagnostics', action='store_true',
                       help='Enable Chapter 14 training diagnostics (v1.7.0)')

    args = parser.parse_args()

    if args.test:
        print("="*70)
        print("REINFORCEMENT ENGINE v1.7.0 - SELF TEST")
        print("="*70)

        config = ReinforcementConfig()
        if args.distributed:
            config.distributed['enabled'] = True
            print("Testing in DISTRIBUTED mode")
        else:
            print("Testing in LOCAL mode")

        np.random.seed(42)
        lottery_history = np.random.randint(0, 1000, 5000).tolist()

        try:
            engine = ReinforcementEngine(
                config, 
                lottery_history,
                enable_diagnostics=args.enable_diagnostics
            )
            print("✅ Engine initialized successfully")
            print(f"   Device: {engine.device}")
            print(f"   Distributed: {engine.is_distributed}")
            print(f"   CUDA initialized: {CUDA_INITIALIZED}")
            print(f"   GPU available: {GPU_AVAILABLE}")
            print(f"   Hostname: {HOST}")
            print(f"   Per-seed features: {engine.per_seed_feature_count}")
            print(f"   Global features: {engine.global_feature_count}")
            print(f"   Excluded features: {engine.excluded_features}")
            print(f"   Diagnostics enabled: {engine.enable_diagnostics}")
            print(f"   Diagnostics available: {DIAGNOSTICS_AVAILABLE}")

            global_state = engine.global_tracker.get_global_state()
            print(f"✅ Global state computed: {len(global_state)} features")

            # Test legacy mode
            features = engine.extract_combined_features(12345)
            print(f"✅ Feature extraction (legacy): {len(features)} features")

            # Test pre-computed mode
            survivor_dict = {
                'seed': 12345,
                'features': {f'feature_{i}': float(i) for i in range(48)}
            }
            features_precomputed = engine.extract_combined_features(survivor_dict)
            print(f"✅ Feature extraction (pre-computed): {len(features_precomputed)} features")

            quality = engine.predict_quality(12345)
            print(f"✅ Prediction: quality={quality:.4f}")

            if engine.is_distributed:
                engine.cleanup_distributed()

            print("\n✅ All tests passed!")
            return 0

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    if not args.lottery_data:
        parser.error("--lottery-data required (or use --test)")

    try:
        config = ReinforcementConfig.from_json(args.config)
        print(f"✅ Config loaded from {args.config}")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        print("Using default config")
        config = ReinforcementConfig()

    if args.distributed:
        config.distributed['enabled'] = True
        print("Distributed mode: ENABLED")

    with open(args.lottery_data, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            lottery_history = [d['draw'] if isinstance(d, dict) else d for d in data]
        else:
            lottery_history = data.get('draws', [])

    print(f"✅ Loaded {len(lottery_history)} lottery draws")

    engine = ReinforcementEngine(
        config, 
        lottery_history,
        enable_diagnostics=args.enable_diagnostics
    )
    print("✅ ReinforcementEngine initialized")
    print(f"   Device: {engine.device}")
    print(f"   Distributed: {engine.is_distributed}")
    print(f"   Model: {sum(p.numel() for p in engine.model.parameters())} parameters")
    print(f"   Diagnostics: {'ENABLED' if engine.enable_diagnostics else 'DISABLED'}")

    global_state = engine.global_tracker.get_global_state()
    print("\n=== Global Statistical State ===")
    for key, value in sorted(global_state.items()):
        print(f"  {key}: {value:.4f}")

    print("\n=== Ready for Training ===")
    print("Use engine.train(survivors, actual_results) to begin")

    if engine.is_distributed:
        engine.cleanup_distributed()

if __name__ == "__main__":
    sys.exit(main() or 0)
