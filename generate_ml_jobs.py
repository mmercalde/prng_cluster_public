#!/usr/bin/env python3
"""
generate_ml_jobs.py - Generate job specs for 26-GPU Optuna training
Creates one job per trial, each will sample different hyperparameters
"""

import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate ML training jobs for distributed Optuna')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--survivors', required=True, help='Path to survivors JSON')
    parser.add_argument('--scores', required=True, help='Path to scores JSON')
    parser.add_argument('--study-name', required=True, help='Optuna study name')
    parser.add_argument('--study-db', required=True, help='Optuna study database URL')
    args = parser.parse_args()

    # Validate input files exist
    if not Path(args.survivors).exists():
        print(f"WARNING: Survivors file not found: {args.survivors}")
    if not Path(args.scores).exists():
        print(f"WARNING: Scores file not found: {args.scores}")

    jobs = []
    for i in range(args.trials):
        job = {
            "job_id": f"ml_trial_{i}",
            "script": "anti_overfit_trial_worker.py",
            "args": [
                args.survivors,
                args.scores,
                args.study_name,
                args.study_db,
                str(i)
            ],
            "gpu_required": True,
            "output_file": f"/shared/ml/results/trial_{i}.json",
            "timeout": 3600,
            "retry_on_failure": False
        }
        jobs.append(job)

    # Write job specifications
    output_file = "ml_jobs.json"
    with open(output_file, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"✅ Generated {len(jobs)} job specifications")
    print(f"   Output: {output_file}")
    print(f"   Study: {args.study_name}")
    print(f"   Database: {args.study_db}")

    # Print first job as sample
    print(f"\n📋 Sample job:")
    print(json.dumps(jobs[0], indent=2))

if __name__ == "__main__":
    main()
michael@zeus:~/distributed_prng_analysis$ cat anti_overfit_trial_worker.py
#!/usr/bin/env python3
"""
anti_overfit_trial_worker.py - Single Optuna trial for distributed ML training
Runs ONE trial with hyperparameters suggested by Optuna
"""

import json
import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    if len(sys.argv) != 6:
        logger.error("Usage: anti_overfit_trial_worker.py <survivors_file> <scores_file> <study_name> <study_db> <trial_id>")
        sys.exit(1)

    survivors_file = sys.argv[1]
    scores_file = sys.argv[2]
    study_name = sys.argv[3]
    study_db = sys.argv[4]
    trial_id = int(sys.argv[5])

    logger.info(f"Starting trial {trial_id}")
    logger.info(f"Survivors: {survivors_file}")
    logger.info(f"Scores: {scores_file}")
    logger.info(f"Study: {study_name}")

    try:
        # Import after args are validated
        import torch
        import optuna
        from reinforcement_engine import ReinforcementEngine, ReinforcementConfig

        # Load data
        logger.info("Loading data...")
        with open(survivors_file) as f:
            survivors = json.load(f)
        with open(scores_file) as f:
            scores = json.load(f)

        logger.info(f"Loaded {len(survivors)} survivors")

        # Load base config
        base_config_path = Path(survivors_file).parent / "reinforcement_engine_config.json"
        if base_config_path.exists():
            base_config = ReinforcementConfig.from_json(str(base_config_path))
            logger.info(f"Loaded base config from {base_config_path}")
        else:
            # Create default config if none exists
            base_config = ReinforcementConfig()
            logger.warning("No base config found, using defaults")

        # Define Optuna objective function
        def objective(trial):
            """Optuna objective - samples hyperparameters and returns validation loss"""

            # Sample hyperparameters
            hidden_layers = trial.suggest_categorical('hidden_layers', [
                [128, 64],
                [256, 128, 64],
                [512, 256, 128],
                [256, 128, 64, 32]
            ])

            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
            epochs = trial.suggest_int('epochs', 30, 100)

            logger.info(f"Trial {trial.number} hyperparameters:")
            logger.info(f"  Hidden layers: {hidden_layers}")
            logger.info(f"  Dropout: {dropout:.3f}")
            logger.info(f"  Learning rate: {learning_rate:.6f}")
            logger.info(f"  Batch size: {batch_size}")
            logger.info(f"  Epochs: {epochs}")

            # Create config with trial's hyperparameters
            config = ReinforcementConfig()
            config.model['hidden_layers'] = hidden_layers
            config.model['dropout'] = dropout
            config.training['learning_rate'] = learning_rate
            config.training['batch_size'] = batch_size
            config.training['epochs'] = epochs

            # Use base config for other settings
            config.training['validation_split'] = base_config.training.get('validation_split', 0.2)
            config.training['early_stopping_patience'] = base_config.training.get('early_stopping_patience', 10)

            # Create model save path
            model_dir = Path("/shared/ml/models")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"trial_{trial.number}_best.pth"

            # Initialize reinforcement engine
            logger.info("Initializing ReinforcementEngine...")
            engine = ReinforcementEngine(
                config=config,
                lottery_history=[0] * 5000  # Dummy history, not used in this context
            )

            # Train the model
            logger.info("Starting training...")
            try:
                engine.train(survivors=survivors, actual_results=scores)

                # Get validation loss
                val_loss = engine.best_val_loss
                overfit_ratio = engine.best_overfit_ratio

                logger.info(f"Training complete:")
                logger.info(f"  Val loss: {val_loss:.6f}")
                logger.info(f"  Overfit ratio: {overfit_ratio:.3f}")

                # Save model if it's good
                if hasattr(engine, 'best_model_path') and engine.best_model_path:
                    import shutil
                    shutil.copy(engine.best_model_path, str(model_path))
                    logger.info(f"Model saved to {model_path}")

                # Return validation loss for Optuna
                return val_loss

            except Exception as e:
                logger.error(f"Training failed: {e}")
                raise optuna.exceptions.TrialPruned()

        # Load or create Optuna study
        logger.info(f"Connecting to Optuna study: {study_name}")
        study = optuna.load_study(
            study_name=study_name,
            storage=study_db
        )

        # Run ONE trial
        logger.info("Starting Optuna optimization (1 trial)...")
        study.optimize(objective, n_trials=1, show_progress_bar=False)

        # Get this trial's results
        trial = study.trials[-1]  # Most recent trial

        # Prepare result JSON
        result = {
            "trial_id": trial_id,
            "trial_number": trial.number,
            "val_loss": trial.value if trial.value is not None else float('inf'),
            "overfit_ratio": None,
            "state": str(trial.state),
            "params": trial.params,
            "model_path": f"/shared/ml/models/trial_{trial.number}_best.pth"
        }

        # Write result to stdout (for coordinator to capture)
        print(json.dumps(result))

        # Also write to file
        output_file = f"/shared/ml/results/trial_{trial_id}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"✅ Trial {trial_id} complete")
        logger.info(f"   Trial number: {trial.number}")
        logger.info(f"   Val loss: {result['val_loss']:.6f}")
        logger.info(f"   Result saved to {output_file}")

    except Exception as e:
        logger.error(f"❌ Trial {trial_id} failed: {e}", exc_info=True)

        # Write error result
        error_result = {
            "trial_id": trial_id,
            "error": str(e),
            "state": "FAILED"
        }
        print(json.dumps(error_result))

        output_file = f"/shared/ml/results/trial_{trial_id}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(error_result, f, indent=2)

        sys.exit(1)

if __name__ == "__main__":
    main()
michael@zeus:~/distributed_prng_analysis$ cat reinforcement_engine.py
#!/usr/bin/env python3
"""
Reinforcement Engine - ML Training Orchestrator (IMPROVED)
===========================================================

IMPROVEMENTS:
✅ 1. Smart model saving - only when validation improves
✅ 2. CUDA lazy initialization (FIX: no conflict with dual GPU scorer)
✅ 3. Better epoch logging with progress indicators
✅ 4. Training summary with best epoch info
✅ 5. Configurable save frequency

Author: Distributed PRNG Analysis System
Date: November 8, 2025
Version: 1.2.1 - FIXED GPU CONFLICT
"""

import sys
import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import Counter, deque
import hashlib

import numpy as np
from scipy.stats import entropy

# PyTorch for neural network
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
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

        # Force CUDA context creation with a dummy operation
        _ = torch.zeros(1).to(device)

        # Initialize both GPUs if available
        if torch.cuda.device_count() > 1:
            for i in range(torch.cuda.device_count()):
                device_i = torch.device(f'cuda:{i}')
                _ = torch.zeros(1).to(device_i)

        return True
    return False


# ✅ FIX: Don't initialize CUDA when module loads
# This prevents conflicts with dual GPU scoring in survivor_scorer.py
# CUDA will be initialized lazily when ReinforcementEngine is created
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
        'input_features': 46,  # From survivor_scorer.py
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
        'save_best_only': True,  # NEW: Only save when val improves
        'save_frequency': 10,     # NEW: Save every N epochs (if not best_only)
        'verbose_frequency': 10   # NEW: Print progress every N epochs
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
        'anomaly_threshold': 3.0,  # Standard deviations
        'regime_change_threshold': 0.15,
        'marker_numbers': [390, 804, 575],  # From discovery
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
            output=data.get('output', {})
        )

    def to_json(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# ============================================================================
# GLOBAL STATE TRACKER (UNCHANGED - keeping your working version)
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
# PYTORCH NEURAL NETWORK (UNCHANGED)
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
# REINFORCEMENT ENGINE (IMPROVED WITH LAZY CUDA INIT)
# ============================================================================

class ReinforcementEngine:
    """
    Main ML training orchestrator - IMPROVED VERSION

    IMPROVEMENTS:
    ✅ Smart model saving (only when validation improves)
    ✅ CUDA lazy initialization (no conflict with dual GPU scorer)
    ✅ Better progress logging
    ✅ Training summary with best epoch
    """

    def __init__(self, config: ReinforcementConfig, lottery_history: List[int],
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.lottery_history = lottery_history
        self.logger = logger or self._setup_logger()

        self.logger.info("Initializing ReinforcementEngine...")

        # ✅ FIX: Initialize CUDA lazily (only when ReinforcementEngine is created)
        global CUDA_INITIALIZED
        if not CUDA_INITIALIZED and torch.cuda.is_available():
            self.logger.info("Initializing CUDA context...")
            CUDA_INITIALIZED = initialize_cuda()
            if CUDA_INITIALIZED:
                self.logger.info("✅ CUDA initialized successfully")

        # Survivor scorer
        self.scorer = SurvivorScorer(
            prng_type=config.prng['prng_type'],
            mod=config.prng['mod']
        )
        self.logger.info(f"  Survivor scorer: {config.prng['prng_type']}, mod={config.prng['mod']}")

        # Global state tracker
        self.global_tracker = GlobalStateTracker(
            lottery_history=lottery_history,
            config=config.global_state
        )
        self.logger.info(f"  Global state tracker: {len(lottery_history)} draws")

        # PyTorch model (CUDA already initialized above)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Calculate total input size
        global_state = self.global_tracker.get_global_state()
        test_features = self.scorer.extract_ml_features(
            seed=0,
            lottery_history=lottery_history[:100] if len(lottery_history) > 100 else lottery_history,
            forward_survivors=None,
            reverse_survivors=None
        )
        total_input_size = len(test_features) + len(global_state)

        self.logger.info(f"  Feature dimensions: {len(test_features)} per-seed + {len(global_state)} global = {total_input_size} total")

        self.model = SurvivorQualityNet(
            input_size=total_input_size,
            hidden_layers=config.model['hidden_layers'],
            dropout=config.model['dropout']
        ).to(self.device)

        # Dual GPU support
        if torch.cuda.device_count() > 1:
            self.logger.info(f"🚀 Using {torch.cuda.device_count()} GPUs for training!")
            self.model = nn.DataParallel(self.model, device_ids=[0, 1])
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.logger.info(f"ℹ️  Using single GPU: {self.device}")

        self.logger.info(f"  Neural network: {total_input_size} inputs → {config.model['hidden_layers']}")
        self.logger.info(f"  Device: {self.device}")

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training['learning_rate']
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training history
        self.training_history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'best_epoch': None,
            'best_val_loss': float('inf')
        }

        # Feature normalization
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

        # Create output directories
        Path(config.output['models_dir']).mkdir(parents=True, exist_ok=True)
        Path(config.output['logs_dir']).mkdir(parents=True, exist_ok=True)

        self.logger.info("ReinforcementEngine initialized successfully!")

    def _setup_logger(self) -> logging.Logger:
        log_level = self.config.output.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def extract_combined_features(self, seed: int,
                                  forward_survivors: Optional[List[int]] = None,
                                  reverse_survivors: Optional[List[int]] = None) -> np.ndarray:
        """Extract combined features (per-seed + global state)"""
        per_seed_features = self.scorer.extract_ml_features(
            seed=seed,
            lottery_history=self.lottery_history,
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
             epochs: Optional[int] = None):
        """
        Train model - IMPROVED VERSION

        IMPROVEMENTS:
        ✅ Only saves models when validation improves
        ✅ Better logging with progress tracking
        ✅ Training summary at end
        """
        if len(survivors) == 0 or len(actual_results) == 0:
            self.logger.warning("No training data provided")
            return

        if len(survivors) != len(actual_results):
            raise ValueError("survivors and actual_results must have same length")

        epochs = epochs or self.config.training['epochs']
        batch_size = self.config.training['batch_size']
        save_best_only = self.config.training.get('save_best_only', True)
        verbose_freq = self.config.training.get('verbose_frequency', 10)

        self.logger.info(f"Training on {len(survivors)} survivors for {epochs} epochs...")

        # Auto-fit normalization
        if self.normalization_enabled:
            if not self.scaler_fitted:
                self.logger.info("Fitting feature normalizer (first training)...")
                self._fit_normalizer(survivors, forward_survivors, reverse_survivors)
            elif self.config.normalization.get('refit_on_drift', True):
                if self._check_distribution_drift(survivors, forward_survivors, reverse_survivors):
                    self.logger.warning("⚠️ Distribution drift detected - refitting normalizer")
                    self._fit_normalizer(survivors, forward_survivors, reverse_survivors)

        # Extract features
        self.logger.info("Extracting features...")
        X = []
        for seed in survivors:
            features = self.extract_combined_features(seed, forward_survivors, reverse_survivors)
            X.append(features)

        X = np.array(X, dtype=np.float32)
        y = np.array(actual_results, dtype=np.float32).reshape(-1, 1)

        # Train/validation split
        val_split = self.config.training['validation_split']
        n_val = int(len(X) * val_split)

        indices = np.random.permutation(len(X))
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Convert to PyTorch tensors
        X_train_t = torch.tensor(X_train).to(self.device)
        y_train_t = torch.tensor(y_train).to(self.device)
        X_val_t = torch.tensor(X_val).to(self.device)
        y_val_t = torch.tensor(y_val).to(self.device)

        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()

            # Mini-batch training
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

            # Log progress
            avg_loss = epoch_loss / n_batches
            self.training_history['epoch'].append(epoch)
            self.training_history['loss'].append(avg_loss)
            self.training_history['val_loss'].append(val_loss)

            # Print progress at intervals
            if (epoch + 1) % verbose_freq == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

            # ✅ IMPROVED MODEL SAVING
            # Only save when validation improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0

                if save_best_only:
                    # Overwrite best model (not create new file)
                    self.save_model(f"best_model_epoch_{epoch+1}.pth")
                else:
                    # Save at configured frequency
                    save_freq = self.config.training.get('save_frequency', 10)
                    if (epoch + 1) % save_freq == 0:
                        self.save_model(f"model_epoch_{epoch+1}.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.config.training['early_stopping_patience']:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Store best epoch info
        self.training_history['best_epoch'] = best_epoch
        self.training_history['best_val_loss'] = best_val_loss

        # ✅ TRAINING SUMMARY
        self.logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
        self.logger.info(f"  Best epoch: {best_epoch}/{epochs}")
        self.logger.info(f"  Final train loss: {avg_loss:.4f}")
        self.logger.info(f"  Final val loss: {val_loss:.4f}")
        self.logger.info(f"  Overfit ratio: {val_loss / avg_loss:.2f}")

    def predict_quality(self, seed: int,
                       forward_survivors: Optional[List[int]] = None,
                       reverse_survivors: Optional[List[int]] = None) -> float:
        """Predict quality score for a single survivor"""
        self.model.eval()

        with torch.no_grad():
            features = self.extract_combined_features(seed, forward_survivors, reverse_survivors)
            features_t = torch.tensor(features).unsqueeze(0).to(self.device)
            quality = self.model(features_t).item()

        return quality

    def predict_quality_batch(self, survivors: List[int],
                              forward_survivors: Optional[List[int]] = None,
                              reverse_survivors: Optional[List[int]] = None) -> List[float]:
        """Predict quality scores for batch of survivors"""
        if not survivors:
            return []

        self.model.eval()

        X = []
        for seed in survivors:
            features = self.extract_combined_features(seed, forward_survivors, reverse_survivors)
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
                       reverse_survivors: Optional[List[int]] = None):
        """Fit feature normalizer on survivor pool"""
        if not self.normalization_enabled:
            return

        temp_fitted = self.scaler_fitted
        self.scaler_fitted = False

        features_list = []
        for seed in survivors:
            features = self.extract_combined_features(seed, forward_survivors, reverse_survivors)
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

    def _check_distribution_drift(self, survivors: List[int],
                                  forward_survivors: Optional[List[int]] = None,
                                  reverse_survivors: Optional[List[int]] = None) -> bool:
        """Check if feature distribution has drifted significantly"""
        if not self.normalization_enabled or not self.scaler_fitted:
            return False

        if self.feature_stats['means'] is None:
            return False

        sample_size = min(100, len(survivors))
        sample_survivors = np.random.choice(survivors, sample_size, replace=False).tolist()

        temp_fitted = self.scaler_fitted
        self.scaler_fitted = False

        features_list = []
        for seed in sample_survivors:
            features = self.extract_combined_features(seed, forward_survivors, reverse_survivors)
            features_list.append(features)

        self.scaler_fitted = temp_fitted

        current_features = np.array(features_list)
        current_means = current_features.mean(axis=0)

        old_means = self.feature_stats['means']
        old_stds = self.feature_stats['stds']

        mean_shift = np.abs((current_means - old_means) / (old_stds + 1e-8))
        max_shift = mean_shift.max()

        drift_threshold = self.config.normalization.get('drift_threshold', 0.3)

        if max_shift > drift_threshold:
            self.logger.warning(f"Distribution drift: max shift = {max_shift:.2f} std devs")
            return True

        return False

    def save_model(self, filename: Optional[str] = None):
        """Save model state"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reinforcement_model_{timestamp}.pth"

        filepath = Path(self.config.output['models_dir']) / filename

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'normalization': {
                'scaler_fitted': self.scaler_fitted,
                'feature_stats': self.feature_stats,
                'scaler_params': {
                    'mean_': self.feature_scaler.mean_.tolist() if self.scaler_fitted else None,
                    'scale_': self.feature_scaler.scale_.tolist() if self.scaler_fitted else None,
                    'var_': self.feature_scaler.var_.tolist() if self.scaler_fitted else None,
                    'n_samples_seen_': int(self.feature_scaler.n_samples_seen_) if self.scaler_fitted else 0
                } if self.normalization_enabled and self.scaler_fitted else None
            }
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
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
                epochs=10
            )


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface for testing"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Reinforcement Engine - ML Training Orchestrator (IMPROVED)'
    )
    parser.add_argument('--config', type=str,
                       default='reinforcement_engine_config.json',
                       help='Configuration file')
    parser.add_argument('--lottery-data', type=str,
                       help='Lottery history JSON file')
    parser.add_argument('--test', action='store_true',
                       help='Run self-test')

    args = parser.parse_args()

    if args.test:
        print("="*70)
        print("REINFORCEMENT ENGINE - SELF TEST (IMPROVED VERSION)")
        print("="*70)

        config = ReinforcementConfig()
        np.random.seed(42)
        lottery_history = np.random.randint(0, 1000, 5000).tolist()

        try:
            engine = ReinforcementEngine(config, lottery_history)
            print("✅ Engine initialized successfully")
            print(f"   Device: {engine.device}")
            print(f"   CUDA initialized: {CUDA_INITIALIZED}")
            print(f"   GPU available: {GPU_AVAILABLE}")

            global_state = engine.global_tracker.get_global_state()
            print(f"✅ Global state computed: {len(global_state)} features")

            features = engine.extract_combined_features(12345)
            print(f"✅ Feature extraction: {len(features)} features")

            quality = engine.predict_quality(12345)
            print(f"✅ Prediction: quality={quality:.4f}")

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

    with open(args.lottery_data, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            lottery_history = [d['draw'] if isinstance(d, dict) else d for d in data]
        else:
            lottery_history = data.get('draws', [])

    print(f"✅ Loaded {len(lottery_history)} lottery draws")

    engine = ReinforcementEngine(config, lottery_history)
    print("✅ ReinforcementEngine initialized")
    print(f"   Device: {engine.device}")
    print(f"   Model: {sum(p.numel() for p in engine.model.parameters())} parameters")

    global_state = engine.global_tracker.get_global_state()
    print("\n=== Global Statistical State ===")
    for key, value in sorted(global_state.items()):
        print(f"  {key}: {value:.4f}")

    print("\n=== Ready for Training ===")
    print("Use engine.train(survivors, actual_results) to begin")


if __name__ == "__main__":
    sys.exit(main() or 0)
