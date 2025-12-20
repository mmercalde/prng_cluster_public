#!/usr/bin/env python3
"""
Reinforcement Engine - ML Training Orchestrator (DISTRIBUTED-READY)
====================================================================

IMPROVEMENTS:
âœ… 1. Smart model saving - only when validation improves
âœ… 2. CUDA lazy initialization (FIX: no conflict with dual GPU scorer)
âœ… 3. Better epoch logging with progress indicators
âœ… 4. Training summary with best epoch info
âœ… 5. Configurable save frequency
âœ… 6. DISTRIBUTED MODE - DDP support with --distributed flag
âœ… 7. ROCm SUPPORT - Compatible with AMD RX 6600 GPUs
âœ… 8. STATELESS BUG FIX: train(), predict...() now accept lottery_history
âœ… 9. META-OPTIMIZER HOOK: __init__ accepts scorer_config_dict
âœ… 10. OPTUNA PRUNING SUPPORT: train() accepts epoch_callback for early stopping

Author: Distributed PRNG Analysis System
Date: November 12, 2025
Version: 1.4.0 - WITH OPTUNA PRUNING HOOKS
"""

import sys
import os
import json
import logging

# ============================================================================
# ROCm environment setup - MUST BE BEFORE TORCH IMPORT
# ============================================================================
import socket
HOST = socket.gethostname()
if HOST in ["rig-6600", "rig-6600b"]:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")
# ============================================================================

import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import Counter, deque
import hashlib

import numpy as np
from scipy.stats import entropy

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

# DDP support (optional - only imported when distributed mode is enabled)
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
# CUDA INITIALIZATION (FIXED - LAZY INITIALIZATION)
# ============================================================================

def initialize_cuda():
    """
    Initialize CUDA context early to prevent cuBLAS warnings

    This fixes the warning:
    "Attempting to run cuBLAS, but there was no current CUDA context!"
    
    IMPORTANT: Always use cuda:0 (the logical device). CUDA_VISIBLE_DEVICES
    handles mapping to the correct physical GPU. Do NOT iterate over 
    device_count() as that returns physical count, breaking isolation.
    """
    if torch.cuda.is_available():
        try:
            # Always use cuda:0 - CUDA_VISIBLE_DEVICES maps this correctly
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            # Force CUDA context creation with a dummy operation
            _ = torch.zeros(1).to(device)
            return True
        except RuntimeError as e:
            if "busy or unavailable" in str(e):
                # CUDA already initialized by another process/thread - that's fine
                return True
            else:
                raise
    return False

CUDA_INITIALIZED = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ReinforcementConfig:
    """
    Configuration for reinforcement engine

    Whitepaper Section 4: ML Integration Parameters
    """
    # Model architecture
    model: Dict[str, Any] = field(default_factory=lambda: {
        'input_features': 50,  # 50 features from survivors_with_scores.json (Step 3)
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
        'save_best_only': True,
        'save_frequency': 10,
        'verbose_frequency': 10
    })

    # PRNG parameters
    prng: Dict[str, Any] = field(default_factory=lambda: {
        'prng_type': 'java_lcg',
        'mod': 1000,
        'skip': 0
    })

    # Global state tracking (Whitepaper Section 4.1)
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
            distributed=data.get('distributed', {})
        )

    def to_json(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# ============================================================================
# GLOBAL STATE TRACKER
# ============================================================================

class GlobalStateTracker:
    """
    Track system-wide statistical patterns

    Whitepaper Section 4.1: Global Statistical State Vector
    """

    def __init__(self, lottery_history: List[int], config: Dict[str, Any]):
        self.lottery_history = lottery_history
        self.config = config
        self._cache = {}
        self._history_hash = self._compute_hash(lottery_history)
        self.current_regime_start = 0
        self.regime_history = []
        self._initialized = False

    def _compute_hash(self, data: List[int]) -> str:
        return hashlib.md5(str(data).encode()).hexdigest()

    def get_global_state(self) -> Dict[str, float]:
        current_hash = self._compute_hash(self.lottery_history)
        if current_hash != self._history_hash or not self._cache:
            self._cache = self._compute_global_state()
            self._history_hash = current_hash
        return self._cache

    def _compute_global_state(self) -> Dict[str, float]:
        if len(self.lottery_history) < 100:
            return self._default_state()

        state = {}
        state.update(self._compute_residue_distributions())
        state.update(self._detect_power_of_two_bias())
        state.update(self._detect_frequency_anomalies())
        state.update(self._detect_regime_changes())
        state.update(self._track_marker_numbers())
        state.update(self._compute_temporal_stability())
        return state

    def _default_state(self) -> Dict[str, float]:
        return {
            'residue_8_entropy': 0.0,
            'residue_125_entropy': 0.0,
            'residue_1000_entropy': 0.0,
            'power_of_two_bias': 0.0,
            'frequency_bias_ratio': 1.0,
            'suspicious_gap_percentage': 0.0,
            'regime_change_detected': 0.0,
            'regime_age': 0.0,
            'high_variance_count': 0.0,
            'marker_390_variance': 0.0,
            'marker_804_variance': 0.0,
            'marker_575_variance': 0.0,
            'reseed_probability': 0.0,
            'temporal_stability': 1.0
        }

    def _compute_residue_distributions(self) -> Dict[str, float]:
        residues = {}
        for mod in [8, 125, 1000]:
            residue_counts = Counter([x % mod for x in self.lottery_history])
            total = len(self.lottery_history)
            probs = [count / total for count in residue_counts.values()]
            ent = entropy(probs)
            max_entropy = np.log(mod)
            normalized_entropy = ent / max_entropy if max_entropy > 0 else 0
            residues[f'residue_{mod}_entropy'] = float(normalized_entropy)
        return residues

    def _detect_power_of_two_bias(self) -> Dict[str, float]:
        powers_of_two = [2**i for i in range(10) if 2**i < 1000]
        power_two_count = sum(1 for x in self.lottery_history if x in powers_of_two)
        expected_rate = len(powers_of_two) / 1000.0
        actual_rate = power_two_count / len(self.lottery_history)
        bias = actual_rate / expected_rate if expected_rate > 0 else 1.0
        return {'power_of_two_bias': float(bias)}

    def _detect_frequency_anomalies(self) -> Dict[str, float]:
        freq_counter = Counter(self.lottery_history)
        if not freq_counter:
            return {'frequency_bias_ratio': 1.0, 'suspicious_gap_percentage': 0.0}
        max_freq = max(freq_counter.values())
        min_freq = min(freq_counter.values())
        ratio = max_freq / min_freq if min_freq > 0 else 1.0
        gap_threshold = self.config.get('gap_threshold', 500)
        last_seen = {}
        suspicious_count = 0
        for i, num in enumerate(self.lottery_history):
            last_seen[num] = i
        current_index = len(self.lottery_history) - 1
        for num in range(1000):
            if num not in last_seen:
                suspicious_count += 1
            elif current_index - last_seen[num] > gap_threshold:
                suspicious_count += 1
        suspicious_pct = suspicious_count / 1000.0
        return {
            'frequency_bias_ratio': float(ratio),
            'suspicious_gap_percentage': float(suspicious_pct)
        }

    def _detect_regime_changes(self) -> Dict[str, float]:
        window_size = self.config.get('window_size', 1000)
        threshold = self.config.get('regime_change_threshold', 0.15)
        if len(self.lottery_history) < window_size * 2:
            return {'regime_change_detected': 0.0, 'regime_age': 0.0}
        recent = self.lottery_history[-window_size:]
        historical = self.lottery_history[-2*window_size:-window_size]
        recent_dist = Counter(recent)
        historical_dist = Counter(historical)
        divergence = 0.0
        for num in range(1000):
            p = (recent_dist.get(num, 0) + 1) / (window_size + 1000)
            q = (historical_dist.get(num, 0) + 1) / (window_size + 1000)
            divergence += p * np.log(p / q)
        regime_changed = 1.0 if divergence > threshold else 0.0
        if regime_changed > 0.5:
            self.current_regime_start = len(self.lottery_history)
        regime_age = len(self.lottery_history) - self.current_regime_start
        return {
            'regime_change_detected': float(regime_changed),
            'regime_age': float(regime_age)
        }

    def _track_marker_numbers(self) -> Dict[str, float]:
        marker_numbers = self.config.get('marker_numbers', [390, 804, 575])
        metrics = {}
        for marker in marker_numbers:
            appearances = [i for i, x in enumerate(self.lottery_history) if x == marker]
            if len(appearances) < 2:
                metrics[f'marker_{marker}_variance'] = 0.0
                continue
            gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
            if not gaps:
                metrics[f'marker_{marker}_variance'] = 0.0
                continue
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            cv = std_gap / mean_gap if mean_gap > 0 else 0.0
            metrics[f'marker_{marker}_variance'] = float(cv)
        high_variance_count = sum(1 for v in metrics.values() if v > 1.0)
        reseed_prob = high_variance_count / len(marker_numbers) if marker_numbers else 0.0
        metrics['reseed_probability'] = float(reseed_prob)
        metrics['high_variance_count'] = float(high_variance_count)
        return metrics

    def _compute_temporal_stability(self) -> Dict[str, float]:
        window_size = min(100, len(self.lottery_history) // 4)
        if len(self.lottery_history) < window_size * 2:
            return {'temporal_stability': 1.0}
        windows = []
        for i in range(4):
            start = len(self.lottery_history) - (i+1) * window_size
            end = len(self.lottery_history) - i * window_size
            if start >= 0:
                windows.append(Counter(self.lottery_history[start:end]))
        if len(windows) < 2:
            return {'temporal_stability': 1.0}
        overlaps = []
        for i in range(len(windows)-1):
            common = set(windows[i].keys()) & set(windows[i+1].keys())
            overlap = len(common) / 1000.0
            overlaps.append(overlap)
        stability = np.mean(overlaps) if overlaps else 1.0
        return {'temporal_stability': float(stability)}

    def update_history(self, new_draws: List[int]):
        self.lottery_history.extend(new_draws)


# ============================================================================
# PYTORCH NEURAL NETWORK
# ============================================================================

class SurvivorQualityNet(nn.Module):
    """PyTorch neural network for survivor quality prediction"""

    def __init__(self, input_size: int = 60, hidden_layers: List[int] = [128, 64, 32],
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
    Main ML training orchestrator - DISTRIBUTED-READY VERSION with ROCm Support
    NOW WITH OPTUNA PRUNING HOOKS
    """

    def __init__(self, config: ReinforcementConfig, lottery_history: List[int],
                 logger: Optional[logging.Logger] = None,
                 scorer_config_dict: Optional[Dict] = None):
        self.config = config
        self.lottery_history = lottery_history
        self.logger = logger or self._setup_logger()

        self.logger.info("Initializing ReinforcementEngine...")

        # Distributed training state
        self.is_distributed = config.distributed.get('enabled', False)
        self.rank = None
        self.world_size = None
        self.local_rank = None

        # Initialize distributed if enabled
        if self.is_distributed:
            self._init_distributed()
        else:
            self.logger.info("  Mode: LOCAL (standard multi-GPU)")

        # Initialize CUDA lazily
        global CUDA_INITIALIZED
        if not CUDA_INITIALIZED and torch.cuda.is_available():
            self.logger.info("Initializing CUDA context...")
            CUDA_INITIALIZED = initialize_cuda()
            if CUDA_INITIALIZED:
                self.logger.info("âœ… CUDA initialized successfully")

        # Survivor scorer
        self.scorer = SurvivorScorer(
            prng_type=config.prng['prng_type'],
            mod=config.prng['mod'],
            config_dict=scorer_config_dict
        )

        if scorer_config_dict:
            self.logger.info(f"  Survivor scorer: Using custom trial parameters!")
        else:
            self.logger.info(f"  Survivor scorer: {config.prng['prng_type']}, mod={config.prng['mod']}")

        # Global state tracker
        self.global_tracker = GlobalStateTracker(
            lottery_history=lottery_history,
            config=config.global_state
        )
        self.logger.info(f"  Global state tracker: {len(lottery_history)} draws")

        # Device selection
        self.device = self._get_device()

        # Calculate total input size (use known dimensions instead of extracting)
        global_state = self.global_tracker.get_global_state()
        # Known: Step 3 extracts 50 features per seed (46 base + 4 sieve metadata)
        test_features_count = 50
        total_input_size = test_features_count + len(global_state)

        self.logger.info(f"  Feature dimensions: {test_features_count} per-seed + {len(global_state)} global = {total_input_size} total")

        # Create model
        self.model = SurvivorQualityNet(
            input_size=total_input_size,
            hidden_layers=config.model['hidden_layers'],
            dropout=config.model['dropout']
        ).to(self.device)

        # Model wrapping
        self._wrap_model()

        self.logger.info(f"  Neural network: {total_input_size} inputs â†’ {config.model['hidden_layers']}")
        self.logger.info(f"  Device: {self.device}")

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
        self.feature_stats = {
            'means': None,
            'stds': None,
            'n_samples': 0
        }

        if self.normalization_enabled:
            self.logger.info("  Feature normalization: ENABLED")
        else:
            self.logger.warning("  Feature normalization: DISABLED (not recommended)")

        Path(config.output['models_dir']).mkdir(parents=True, exist_ok=True)
        Path(config.output['logs_dir']).mkdir(parents=True, exist_ok=True)

        self.logger.info("ReinforcementEngine initialized successfully!")

    def _init_distributed(self):
        """Initialize distributed training"""
        if not DDP_AVAILABLE:
            self.logger.error("âŒ DDP not available - install PyTorch with distributed support")
            raise RuntimeError("DDP not available")

        self.rank = int(os.environ.get('RANK', self.config.distributed.get('rank', 0)))
        self.world_size = int(os.environ.get('WORLD_SIZE', self.config.distributed.get('world_size', 1)))
        self.local_rank = int(os.environ.get('LOCAL_RANK', self.config.distributed.get('local_rank', 0)))

        backend = self.config.distributed.get('backend', 'nccl')
        init_method = self.config.distributed.get('init_method', 'env://')

        self.logger.info(f"  Mode: DISTRIBUTED")
        self.logger.info(f"    Rank: {self.rank}/{self.world_size}")
        self.logger.info(f"    Local rank: {self.local_rank}")
        self.logger.info(f"    Backend: {backend}")

        try:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=self.rank,
                world_size=self.world_size
            )
            self.logger.info("  âœ… DDP initialized successfully")
        except Exception as e:
            self.logger.error(f"âŒ DDP initialization failed: {e}")
            raise

    def _get_device(self) -> torch.device:
        """Get appropriate device for local or distributed mode"""
        if not torch.cuda.is_available():
            return torch.device('cpu')

        if self.is_distributed:
            device_id = self.local_rank
            return torch.device(f'cuda:{device_id}')
        else:
            return torch.device('cuda:0')

    def _wrap_model(self):
        """Wrap model with DDP or DataParallel"""
        if self.is_distributed:
            self.logger.info(f"  Wrapping model with DistributedDataParallel...")
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.distributed.get('find_unused_parameters', False)
            )
            self.logger.info(f"  âœ… DDP wrapper active (rank {self.rank})")
        elif torch.cuda.device_count() > 1 and os.environ.get('CUDA_VISIBLE_DEVICES') is None:
            # Only use DataParallel when CUDA_VISIBLE_DEVICES is NOT set
            # When set, we're running isolated single-GPU jobs
            self.logger.info(f"  Using {torch.cuda.device_count()} GPUs for training!")
            device_ids = list(range(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            for i in device_ids:
                self.logger.info(f"     GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.logger.info(f"  â„¹ï¸  Using single device: {self.device}")

    def _setup_logger(self) -> logging.Logger:
        log_level = self.config.output.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def extract_combined_features(self, seed: int,
                                  forward_survivors: Optional[List[int]] = None,
                                  reverse_survivors: Optional[List[int]] = None,
                                  lottery_history: Optional[List[int]] = None) -> np.ndarray:
        """
        Extract combined features (per-seed + global state)
        MODIFIED: Accepts lottery_history override for stateless operation.
        """
        history_to_use = lottery_history if lottery_history is not None else self.lottery_history

        per_seed_features = self.scorer.extract_ml_features(
            seed=seed,
            lottery_history=history_to_use,
            forward_survivors=forward_survivors,
            reverse_survivors=reverse_survivors
        )

        global_state = self.global_tracker.get_global_state()

        feature_names = sorted(per_seed_features.keys())
        per_seed_values = [per_seed_features[k] for k in feature_names]

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

    def train(self, survivors: List[int],
             actual_results: List[float],
             forward_survivors: Optional[List[int]] = None,
             reverse_survivors: Optional[List[int]] = None,
             epochs: Optional[int] = None,
             lottery_history: Optional[List[int]] = None,
             epoch_callback: Optional[callable] = None):  # âœ… NEW: Pruning callback
        """
        Train model - DISTRIBUTED-READY VERSION WITH PRUNING SUPPORT
        MODIFIED: Accepts lottery_history override for stateless operation.
        NEW: Accepts epoch_callback for Optuna pruning support.

        Args:
            epoch_callback: Optional function(epoch: int, val_loss: float) -> bool
                           Returns True to continue training, False to stop (prune)
        """
        if len(survivors) == 0 or len(actual_results) == 0:
            self.logger.warning("No training data provided")
            return

        epochs = epochs or self.config.training['epochs']
        batch_size = self.config.training['batch_size']
        save_best_only = self.config.training.get('save_best_only', True)
        verbose_freq = self.config.training.get('verbose_frequency', 10)
        should_log = not self.is_distributed or self.rank == 0

        if should_log:
            self.logger.info(f"Training on {len(survivors)} survivors for {epochs} epochs...")

        # Normalization
        if self.normalization_enabled:
            if not self.scaler_fitted:
                 if should_log:
                    self.logger.info("Fitting feature normalizer (first training)...")
                 self._fit_normalizer(survivors, forward_survivors, reverse_survivors, lottery_history)
            elif self.config.normalization.get('refit_on_drift', True):
                 if self._check_distribution_drift(survivors, forward_survivors, reverse_survivors, lottery_history):
                    if should_log:
                        self.logger.warning("âš ï¸ Distribution drift detected - refitting normalizer")
                    self._fit_normalizer(survivors, forward_survivors, reverse_survivors, lottery_history)

        # Extract features
        if should_log:
            self.logger.info("Extracting features...")
        X = []
        for seed in survivors:
            features = self.extract_combined_features(seed, forward_survivors, reverse_survivors,
                                                    lottery_history=lottery_history)
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

        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        # âœ… MODIFIED: Training loop with pruning callback support
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

            # Update history
            self.training_history['epoch'].append(epoch)
            self.training_history['loss'].append(avg_loss)
            self.training_history['val_loss'].append(val_loss)

            # âœ… NEW: Call pruning callback if provided
            if epoch_callback is not None:
                should_continue = epoch_callback(epoch, val_loss)
                if not should_continue:
                    if should_log:
                        self.logger.info(f"âš¡ Training stopped by callback at epoch {epoch+1}")
                    break

            # Logging
            if should_log and (epoch + 1) % verbose_freq == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Best model tracking
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0

                if save_best_only and should_log:
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

        if should_log:
            self.logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
            self.logger.info(f"  Best epoch: {best_epoch}/{epochs}")
            self.logger.info(f"  Final train loss: {avg_loss:.4f}")
            self.logger.info(f"  Final val loss: {val_loss:.4f}")
            self.logger.info(f"  Overfit ratio: {val_loss / avg_loss:.2f}")

    def predict_quality(self, seed: int,
                       forward_survivors: Optional[List[int]] = None,
                       reverse_survivors: Optional[List[int]] = None,
                       lottery_history: Optional[List[int]] = None) -> float:
        """
        Predict quality score for a single survivor
        MODIFIED: Accepts lottery_history override for stateless operation.
        """
        self.model.eval()

        with torch.no_grad():
            features = self.extract_combined_features(seed, forward_survivors, reverse_survivors,
                                                    lottery_history=lottery_history)
            features_t = torch.tensor(features).unsqueeze(0).to(self.device)
            quality = self.model(features_t).item()

        return quality

    def predict_quality_batch(self, survivors: List[int],
                              forward_survivors: Optional[List[int]] = None,
                              reverse_survivors: Optional[List[int]] = None,
                              lottery_history: Optional[List[int]] = None) -> List[float]:
        """
        Predict quality scores for batch of survivors
        MODIFIED: Accepts lottery_history override for stateless operation.
        """
        if not survivors:
            return []

        self.model.eval()

        X = []
        for seed in survivors:
            features = self.extract_combined_features(seed, forward_survivors, reverse_survivors,
                                                    lottery_history=lottery_history)
            X.append(features)

        X = np.array(X, dtype=np.float32)
        X_t = torch.tensor(X).to(self.device)

        with torch.no_grad():
            qualities = self.model(X_t).cpu().numpy().flatten()

        return qualities.tolist()

    def prune_survivors(self, survivors: List[int],
                       keep_top_n: Optional[int] = None,
                       forward_survivors: Optional[List[int]] = None,
                       reverse_survivors: Optional[List[int]] = None) -> List[int]:
        """Prune survivor pool to keep only top performers"""
        if not survivors:
            return []

        keep_top_n = keep_top_n or int(len(survivors) * self.config.survivor_pool['prune_threshold'])
        keep_top_n = max(1, min(keep_top_n, len(survivors)))

        self.logger.info(f"Pruning {len(survivors)} survivors to top {keep_top_n}...")

        qualities = self.predict_quality_batch(survivors, forward_survivors, reverse_survivors)
        ranked = sorted(zip(survivors, qualities), key=lambda x: x[1], reverse=True)
        top_survivors = [seed for seed, _ in ranked[:keep_top_n]]

        self.logger.info(f"Kept top {len(top_survivors)} survivors")
        return top_survivors

    def _fit_normalizer(self, survivors: List[int],
                       forward_survivors: Optional[List[int]] = None,
                       reverse_survivors: Optional[List[int]] = None,
                       lottery_history: Optional[List[int]] = None):
        """Fit feature normalizer on survivor pool"""
        if not self.normalization_enabled:
            return

        temp_fitted = self.scaler_fitted
        self.scaler_fitted = False

        features_list = []
        for seed in survivors:
            features = self.extract_combined_features(seed, forward_survivors, reverse_survivors,
                                                    lottery_history=lottery_history)
            features_list.append(features)

        features_array = np.array(features_list)
        self.feature_scaler.fit(features_array)
        self.scaler_fitted = True
        self.feature_stats['means'] = features_array.mean(axis=0)
        self.feature_stats['stds'] = features_array.std(axis=0)
        self.feature_stats['n_samples'] = len(features_array)

        mean_range = [self.feature_stats['means'].min(), self.feature_stats['means'].max()]
        std_range = [self.feature_stats['stds'].min(), self.feature_stats['stds'].max()]

        self.logger.info(f"âœ… Normalizer fitted on {len(survivors)} survivors")
        self.logger.info(f"   Feature mean range: [{mean_range[0]:.2f}, {mean_range[1]:.2f}]")
        self.logger.info(f"   Feature std range: [{std_range[0]:.2f}, {std_range[1]:.2f}]")

    def _check_distribution_drift(self, survivors: List[int],
                                  forward_survivors: Optional[List[int]] = None,
                                  reverse_survivors: Optional[List[int]] = None,
                                  lottery_history: Optional[List[int]] = None) -> bool:
        """Check if feature distribution has drifted significantly"""
        if not self.normalization_enabled or not self.scaler_fitted:
            return False

        sample_size = min(100, len(survivors))
        if sample_size == 0:
            return False

        sample_survivors = np.random.choice(survivors, sample_size, replace=False).tolist()

        temp_fitted = self.scaler_fitted
        self.scaler_fitted = False

        features_list = []
        for seed in sample_survivors:
            features = self.extract_combined_features(seed, forward_survivors, reverse_survivors,
                                                    lottery_history=lottery_history)
            features_list.append(features)

        self.scaler_fitted = temp_fitted

        current_features = np.array(features_list)
        current_means = current_features.mean(axis=0)
        old_means = self.feature_stats['means']
        old_stds = self.feature_stats['stds']

        if old_means is None or old_stds is None:
            return False

        mean_shift = np.abs((current_means - old_means) / (old_stds + 1e-8))
        max_shift = mean_shift.max()
        drift_threshold = self.config.normalization.get('drift_threshold', 0.3)

        if max_shift > drift_threshold:
            self.logger.warning(f"Distribution drift: max shift = {max_shift:.2f} std devs")
            return True

        return False

    def save_model(self, filename: Optional[str] = None):
        """Save model state - DDP-aware"""
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
            'distributed_info': {
                'is_distributed': self.is_distributed,
                'rank': self.rank,
                'world_size': self.world_size
            } if self.is_distributed else None
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model state - DDP-aware"""
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

        self.logger.info(f"Model loaded from {filepath}")

    def continuous_learning_loop(self, new_draw: int,
                                 survivors: List[int],
                                 forward_survivors: Optional[List[int]] = None,
                                 reverse_survivors: Optional[List[int]] = None):
        """Continuous learning loop - update model with new draw"""
        self.lottery_history.append(new_draw)
        self.global_tracker.update_history([new_draw])

        actual_results = []
        for seed in survivors:
            predicted = self.scorer.score_survivor(seed, [new_draw], skip=self.config.prng['skip'])
            hit_rate = predicted['score']
            actual_results.append(hit_rate)

        if len(self.lottery_history) % self.config.survivor_pool['update_frequency'] == 0:
            self.logger.info("Retraining model with new data...")
            self.train(
                survivors=survivors,
                actual_results=actual_results,
                forward_survivors=forward_survivors,
                reverse_survivors=reverse_survivors,
                epochs=10,
                lottery_history=self.lottery_history
            )

    def cleanup_distributed(self):
        """Cleanup distributed resources"""
        if self.is_distributed and dist.is_initialized():
            self.logger.info("Cleaning up distributed resources...")
            dist.destroy_process_group()
            self.logger.info("âœ… Distributed cleanup complete")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface for testing"""
    import argparse
    parser = argparse.ArgumentParser(
        description='Reinforcement Engine - ML Training Orchestrator (DISTRIBUTED-READY with ROCm Support)'
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

    args = parser.parse_args()

    if args.test:
        print("="*70)
        print("REINFORCEMENT ENGINE - SELF TEST (DISTRIBUTED-READY with ROCm)")
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
            engine = ReinforcementEngine(config, lottery_history)
            print("âœ… Engine initialized successfully")
            print(f"   Device: {engine.device}")
            print(f"   Distributed: {engine.is_distributed}")
            print(f"   CUDA initialized: {CUDA_INITIALIZED}")
            print(f"   GPU available: {GPU_AVAILABLE}")
            print(f"   Hostname: {HOST}")

            global_state = engine.global_tracker.get_global_state()
            print(f"âœ… Global state computed: {len(global_state)} features")

            features = engine.extract_combined_features(12345)
            print(f"âœ… Feature extraction: {len(features)} features")

            quality = engine.predict_quality(12345)
            print(f"âœ… Prediction: quality={quality:.4f}")

            if engine.is_distributed:
                engine.cleanup_distributed()

            print("\nâœ… All tests passed!")
            return 0

        except Exception as e:
            print(f"âŒ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    if not args.lottery_data:
        parser.error("--lottery-data required (or use --test)")

    try:
        config = ReinforcementConfig.from_json(args.config)
        print(f"âœ… Config loaded from {args.config}")
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

    print(f"âœ… Loaded {len(lottery_history)} lottery draws")

    engine = ReinforcementEngine(config, lottery_history)
    print("âœ… ReinforcementEngine initialized")
    print(f"   Device: {engine.device}")
    print(f"   Distributed: {engine.is_distributed}")
    print(f"   Model: {sum(p.numel() for p in engine.model.parameters())} parameters")

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
