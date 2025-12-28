"""
Model Factory (v3.1.2)

Single entry point for creating and loading ML models.
Supports: neural_net, xgboost, lightgbm, catboost

Usage:
    from models import create_model, load_model
    
    # Create new model
    model = create_model('xgboost', config={'n_estimators': 200})
    
    # Load existing model
    model = load_model('xgboost', 'models/reinforcement/best_model.json')
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Available model types
AVAILABLE_MODELS = ['neural_net', 'xgboost', 'lightgbm', 'catboost', 'random_forest']

# Model file extensions
MODEL_EXTENSIONS = {
    'neural_net': '.pth',
    'xgboost': '.json',
    'lightgbm': '.txt',
    'catboost': '.cbm',
    'random_forest': '.joblib'
}

# Default output directory (matches reinforcement_engine.py)
DEFAULT_MODELS_DIR = 'models/reinforcement'


def _get_model_class(model_type: str):
    """
    Lazy load model class to avoid import errors if dependencies missing.
    
    Args:
        model_type: One of 'neural_net', 'xgboost', 'lightgbm', 'catboost'
        
    Returns:
        Model wrapper class
        
    Raises:
        ValueError: If model_type is unknown
        ImportError: If required package is not installed
    """
    if model_type == 'neural_net':
        from models.wrappers.neural_net_wrapper import NeuralNetWrapper
        return NeuralNetWrapper
    elif model_type == 'xgboost':
        from models.wrappers.xgboost_wrapper import XGBoostWrapper
        return XGBoostWrapper
    elif model_type == 'lightgbm':
        from models.wrappers.lightgbm_wrapper import LightGBMWrapper
        return LightGBMWrapper
    elif model_type == 'catboost':
        from models.wrappers.catboost_wrapper import CatBoostWrapper
        return CatBoostWrapper
    elif model_type == 'random_forest':
        from models.wrappers.random_forest_wrapper import RandomForestWrapper
        return RandomForestWrapper
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {AVAILABLE_MODELS}"
        )


def create_model(model_type: str, config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
    """
    Create a new model instance.
    
    Args:
        model_type: One of 'neural_net', 'xgboost', 'lightgbm', 'catboost'
        config: Model configuration dict (model-specific parameters)
        device: Device string (e.g., 'cuda:0', 'cpu')
        
    Returns:
        Model wrapper instance implementing ModelInterface
        
    Example:
        model = create_model('xgboost', config={'n_estimators': 200})
        model.fit(X_train, y_train)
    """
    logger.info(f"Creating model: {model_type}")
    model_class = _get_model_class(model_type)
    return model_class(config=config, device=device)


def load_model(model_type: str, path: str, device: Optional[str] = None):
    """
    Load a trained model from file.
    
    Args:
        model_type: One of 'neural_net', 'xgboost', 'lightgbm', 'catboost'
        path: Path to saved model file
        device: Device to load model to
        
    Returns:
        Loaded model wrapper instance
        
    Example:
        model = load_model('xgboost', 'models/reinforcement/best_model.json')
        predictions = model.predict(X_test)
    """
    logger.info(f"Loading model: {model_type} from {path}")
    model_class = _get_model_class(model_type)
    return model_class.load(path, device=device)


def list_available_models() -> List[str]:
    """Return list of available model types."""
    return AVAILABLE_MODELS.copy()


def get_model_extension(model_type: str) -> str:
    """
    Get the file extension for a model type.
    
    Args:
        model_type: One of 'neural_net', 'xgboost', 'lightgbm', 'catboost'
        
    Returns:
        File extension (e.g., '.pth', '.json')
    """
    return MODEL_EXTENSIONS.get(model_type, '.bin')


def get_model_defaults(model_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a model type.
    
    Args:
        model_type: One of 'neural_net', 'xgboost', 'lightgbm', 'catboost'
        
    Returns:
        Default configuration dict
    """
    defaults = {
        'neural_net': {
            'input_size': 50,
            'hidden_layers': [128, 64, 32],
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 100
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'catboost': {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8
        }
    }
    return defaults.get(model_type, {}).copy()


def get_zeus_only_models() -> List[str]:
    """Return models that require CUDA (Zeus only)."""
    return ['xgboost', 'lightgbm', 'catboost']


def get_distributed_models() -> List[str]:
    """Return models that can run on all nodes (ROCm + CUDA)."""
    return ['neural_net']


def save_model_with_sidecar(model, output_dir: str, 
                            feature_schema: Dict[str, Any],
                            y_label_metadata: Dict[str, Any],
                            training_info: Dict[str, Any],
                            validation_metrics: Dict[str, Any]) -> str:
    """
    Save model with metadata sidecar file.
    
    This is the canonical way to save models in the pipeline.
    Creates both the model checkpoint and best_model.meta.json.
    
    Args:
        model: Trained model wrapper
        output_dir: Output directory (default: models/reinforcement)
        feature_schema: Feature schema from get_feature_schema_with_hash()
        y_label_metadata: Y-label metadata from load_quality_from_survivors()
        training_info: Training info (started_at, completed_at, k_folds, etc.)
        validation_metrics: Validation metrics (mse, mae, rmse, etc.)
        
    Returns:
        Path to saved checkpoint file
    """
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine checkpoint filename
    model_type = model.model_type
    extension = get_model_extension(model_type)
    checkpoint_name = f"best_model{extension}"
    checkpoint_path = output_path / checkpoint_name
    
    # Save model
    model.save(str(checkpoint_path))
    
    # Generate sidecar metadata
    meta = {
        "schema_version": "3.1.2",
        "model_type": model_type,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_format": extension.lstrip('.'),
        
        "feature_schema": feature_schema,
        
        "y_label_source": y_label_metadata,
        
        "training_params": model.config if hasattr(model, 'config') else {},
        
        "validation_metrics": validation_metrics,
        
        "hardware": model._gpu_info if hasattr(model, '_gpu_info') else {},
        
        "training_info": training_info,
        
        "agent_metadata": {
            "pipeline_step": 5,
            "pipeline_step_name": "anti_overfit_training",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }
    
    # Write sidecar
    meta_path = output_path / "best_model.meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Model saved to {checkpoint_path}")
    logger.info(f"Metadata saved to {meta_path}")
    
    return str(checkpoint_path)


def load_model_from_sidecar(models_dir: str, device: Optional[str] = None,
                            survivors_file: Optional[str] = None,
                            strict: bool = True):
    """
    Load model using sidecar metadata (Step 6 pattern).
    
    CRITICAL (Team Beta Fix 2): Strict validation with FATAL errors.
    - Model type from meta.json ONLY (never file extension)
    - FATAL if meta.json missing
    - FATAL if model_type field missing
    - FATAL if schema hash mismatch (when survivors_file provided)
    - FATAL if feature count mismatch
    
    Args:
        models_dir: Directory containing best_model.meta.json
        device: Device to load model to
        survivors_file: Optional path to survivors for schema validation
        strict: If True, raise on validation failures (default: True)
    
    Returns:
        Tuple of (model, metadata)
    
    Raises:
        FileNotFoundError: If sidecar or checkpoint missing
        ValueError: If required fields missing or schema mismatch
    """
    meta_path = Path(models_dir) / "best_model.meta.json"
    
    # FATAL: Missing sidecar
    if not meta_path.exists():
        raise FileNotFoundError(
            f"FATAL: Missing metadata sidecar: {meta_path}\n"
            "Step 5 MUST generate best_model.meta.json.\n"
            "Model type CANNOT be inferred from file extension."
        )
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    # FATAL: model_type missing
    if "model_type" not in meta:
        raise ValueError(
            f"FATAL: model_type field missing from {meta_path}\n"
            "Sidecar is malformed. Re-run Step 5."
        )
    
    model_type = meta["model_type"]
    
    # FATAL: checkpoint_path missing
    if "checkpoint_path" not in meta:
        raise ValueError(
            f"FATAL: checkpoint_path field missing from {meta_path}\n"
            "Sidecar is malformed. Re-run Step 5."
        )
    
    checkpoint_path = meta["checkpoint_path"]
    
    # Validate checkpoint exists
    if not Path(checkpoint_path).exists():
        # Try relative to models_dir
        alt_path = Path(models_dir) / Path(checkpoint_path).name
        if alt_path.exists():
            checkpoint_path = str(alt_path)
        else:
            raise FileNotFoundError(
                f"FATAL: Checkpoint not found: {checkpoint_path}\n"
                f"Also tried: {alt_path}\n"
                f"Referenced in: {meta_path}"
            )
    
    # Schema validation (if survivors_file provided)
    if survivors_file and strict:
        try:
            from models.feature_schema import get_feature_schema_with_hash
            
            # Get runtime schema
            runtime_schema = get_feature_schema_with_hash(survivors_file)
            
            # Get training schema from meta
            training_schema = meta.get("feature_schema", {})
            training_hash = training_schema.get("combined_hash") or training_schema.get("feature_schema_hash", "")
            training_count = training_schema.get("total_features") or training_schema.get("per_seed_feature_count", 0)
            
            runtime_hash = runtime_schema.get("feature_schema_hash", "")
            runtime_count = runtime_schema.get("feature_count", 0)
            
            # FATAL: Feature count mismatch
            if training_count > 0 and runtime_count > 0 and training_count != runtime_count:
                raise ValueError(
                    f"FATAL: Feature count mismatch\n"
                    f"  Training: {training_count} features\n"
                    f"  Runtime:  {runtime_count} features\n"
                    f"Model was trained on different feature set."
                )
            
            # FATAL: Schema hash mismatch
            if training_hash and runtime_hash and training_hash != runtime_hash:
                raise ValueError(
                    f"FATAL: Feature schema hash mismatch\n"
                    f"  Training: {training_hash}\n"
                    f"  Runtime:  {runtime_hash}\n"
                    f"Features may have different ordering or names."
                )
            
            logger.info(f"Schema validation passed: {runtime_count} features, hash={runtime_hash[:8]}...")
            
        except ImportError:
            logger.warning("feature_schema module not available, skipping schema validation")
    
    # Load model
    model = load_model(model_type, checkpoint_path, device=device)
    
    logger.info(f"Loaded {model_type} from {checkpoint_path}")
    
    return model, meta
