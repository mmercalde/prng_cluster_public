#!/usr/bin/env python3
"""
Isolated Single-Trial Trainer (train_single_trial.py)
=====================================================

Runs in fresh subprocess - NO GPU contamination between trials.
This solves the OpenCL/CUDA conflict when comparing models.

Uses existing model wrappers from models/wrappers/*.py
Maintains full backward compatibility with existing pipeline.

Usage (called by meta_prediction_optimizer_anti_overfit.py):
    python3 train_single_trial.py \
        --model-type lightgbm \
        --data-path /tmp/trial_data.npz \
        --params '{"n_estimators": 100, "learning_rate": 0.1}' \
        --fold-indices /tmp/fold_indices.json

Output:
    JSON to stdout with metrics for Optuna to capture.

Author: PRNG Analysis System
Date: December 2025
Version: 1.0.1 - WITH MODEL SAVING

CRITICAL: NO imports at module level except stdlib!
          This ensures clean GPU state when subprocess starts.
"""

# ==============================================================================
# STDLIB ONLY AT MODULE LEVEL - NO GPU LIBRARIES!
# ==============================================================================
import sys
import json
import argparse
import time
import os
from pathlib import Path
from datetime import datetime
import tempfile

# Version for tracking
__version__ = "1.0.1"

# Model file extensions
MODEL_EXTENSIONS = {
    'neural_net': '.pth',
    'xgboost': '.json',
    'lightgbm': '.txt',
    'catboost': '.cbm'
}


def setup_gpu_environment(model_type: str):
    """
    Setup GPU environment based on model type.
    Called BEFORE importing any ML libraries.
    """
    if model_type == 'lightgbm':
        # LightGBM uses OpenCL - no special setup needed
        pass
    else:
        # CUDA models - set device visibility if needed
        # This runs before torch/xgboost/catboost imports
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def train_lightgbm(X_train, y_train, X_val, y_val, params: dict, save_path: str = None) -> dict:
    """
    Train LightGBM model with GPU (OpenCL).
    
    Imports lightgbm HERE to ensure clean GPU state.
    """
    import lightgbm as lgb
    import numpy as np
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Default params with GPU
    lgb_params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'num_leaves': params.get('num_leaves', 31),
        'learning_rate': params.get('learning_rate', 0.05),
        'max_depth': params.get('max_depth', -1),
        'min_data_in_leaf': params.get('min_data_in_leaf', 20),
        'feature_fraction': params.get('feature_fraction', 0.8),
        'bagging_fraction': params.get('bagging_fraction', 0.8),
        'bagging_freq': params.get('bagging_freq', 5),
        'lambda_l1': params.get('lambda_l1', 0.0),
        'lambda_l2': params.get('lambda_l2', 0.0),
        # GPU settings (OpenCL)
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
    }
    
    n_estimators = params.get('n_estimators', 100)
    
    try:
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        device_used = 'gpu'
    except Exception as e:
        # Fallback to CPU if GPU fails
        print(f"LightGBM GPU failed ({e}), falling back to CPU", file=sys.stderr)
        lgb_params['device'] = 'cpu'
        del lgb_params['gpu_platform_id']
        del lgb_params['gpu_device_id']
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        device_used = 'cpu_fallback'
    
    # Predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Metrics
    train_mse = float(np.mean((train_preds - y_train) ** 2))
    val_mse = float(np.mean((val_preds - y_val) ** 2))
    train_mae = float(np.mean(np.abs(train_preds - y_train)))
    val_mae = float(np.mean(np.abs(val_preds - y_val)))
    
    # R² score
    ss_res = np.sum((y_val - val_preds) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Save model if path provided
    checkpoint_path = None
    if save_path:
        model.save_model(save_path)
        checkpoint_path = save_path
    
    return {
        'model_type': 'lightgbm',
        'device': device_used,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'r2': r2,
        'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else n_estimators,
        'checkpoint_path': checkpoint_path
    }


def train_xgboost(X_train, y_train, X_val, y_val, params: dict, save_path: str = None) -> dict:
    """
    Train XGBoost model with GPU (CUDA).
    
    Imports xgboost HERE to ensure clean GPU state.
    """
    import xgboost as xgb
    import numpy as np
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Default params with GPU
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': params.get('max_depth', 6),
        'learning_rate': params.get('learning_rate', 0.1),
        'min_child_weight': params.get('min_child_weight', 1),
        'subsample': params.get('subsample', 0.8),
        'colsample_bytree': params.get('colsample_bytree', 0.8),
        'reg_alpha': params.get('reg_alpha', 0.0),
        'reg_lambda': params.get('reg_lambda', 1.0),
        'verbosity': 0,
        # GPU settings (CUDA)
        'tree_method': 'hist',
        'device': 'cuda:0',
    }
    
    n_estimators = params.get('n_estimators', 100)
    
    try:
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        device_used = 'cuda:0'
    except Exception as e:
        # Fallback to CPU if GPU fails
        print(f"XGBoost GPU failed ({e}), falling back to CPU", file=sys.stderr)
        xgb_params['device'] = 'cpu'
        xgb_params['tree_method'] = 'hist'
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        device_used = 'cpu_fallback'
    
    # Predictions
    train_preds = model.predict(dtrain)
    val_preds = model.predict(dval)
    
    # Metrics
    train_mse = float(np.mean((train_preds - y_train) ** 2))
    val_mse = float(np.mean((val_preds - y_val) ** 2))
    train_mae = float(np.mean(np.abs(train_preds - y_train)))
    val_mae = float(np.mean(np.abs(val_preds - y_val)))
    
    # R² score
    ss_res = np.sum((y_val - val_preds) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Save model if path provided
    checkpoint_path = None
    if save_path:
        model.save_model(save_path)
        checkpoint_path = save_path
    
    return {
        'model_type': 'xgboost',
        'device': device_used,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'r2': r2,
        'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else n_estimators,
        'checkpoint_path': checkpoint_path
    }


def train_catboost(X_train, y_train, X_val, y_val, params: dict, save_path: str = None) -> dict:
    """
    Train CatBoost model with GPU (CUDA).
    
    Imports catboost HERE to ensure clean GPU state.
    """
    from catboost import CatBoostRegressor
    import numpy as np
    
    try:
        model = CatBoostRegressor(
            iterations=params.get('n_estimators', 100),
            depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
            random_strength=params.get('random_strength', 1.0),
            bagging_temperature=params.get('bagging_temperature', 1.0),
            border_count=params.get('border_count', 254),
            verbose=False,
            early_stopping_rounds=20,
            # GPU settings (CUDA) - use both GPUs
            task_type='GPU',
            devices='0:1',
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        device_used = 'cuda:0:1'
    except Exception as e:
        # Fallback to CPU if GPU fails
        print(f"CatBoost GPU failed ({e}), falling back to CPU", file=sys.stderr)
        model = CatBoostRegressor(
            iterations=params.get('n_estimators', 100),
            depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
            verbose=False,
            early_stopping_rounds=20,
            task_type='CPU',
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        device_used = 'cpu_fallback'
    
    # Predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Metrics
    train_mse = float(np.mean((train_preds - y_train) ** 2))
    val_mse = float(np.mean((val_preds - y_val) ** 2))
    train_mae = float(np.mean(np.abs(train_preds - y_train)))
    val_mae = float(np.mean(np.abs(val_preds - y_val)))
    
    # R² score
    ss_res = np.sum((y_val - val_preds) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Save model if path provided
    checkpoint_path = None
    if save_path:
        model.save_model(save_path)
        checkpoint_path = save_path
    
    return {
        'model_type': 'catboost',
        'device': device_used,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'r2': r2,
        'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') else params.get('n_estimators', 100),
        'checkpoint_path': checkpoint_path
    }


def train_neural_net(X_train, y_train, X_val, y_val, params: dict, save_path: str = None) -> dict:
    """
    Train PyTorch Neural Net with GPU (CUDA).
    
    Imports torch HERE to ensure clean GPU state.
    Uses architecture similar to existing SurvivorQualityNet.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        device_used = f'cuda:0 ({torch.cuda.get_device_name(0)})'
        # Warm up GPU
        _ = torch.zeros(1).to(device)
    else:
        device = torch.device('cpu')
        device_used = 'cpu'
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # Architecture from params (matches SurvivorQualityNet style)
    input_dim = X_train.shape[1]
    hidden_layers = params.get('hidden_layers', [256, 128, 64])
    dropout = params.get('dropout', 0.3)
    
    # Build model dynamically
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        ])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, 1))
    
    model = nn.Sequential(*layers).to(device)
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        device_used = f'cuda:0,1 (DataParallel)'
    
    # Training setup
    optimizer_name = params.get('optimizer', 'adam')
    lr = params.get('learning_rate', 0.001)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=params.get('weight_decay', 1e-5))
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=params.get('weight_decay', 1e-5))
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=params.get('weight_decay', 1e-5))
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    criterion = nn.MSELoss()
    
    # DataLoader
    batch_size = params.get('batch_size', 128)
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop with early stopping
    epochs = params.get('epochs', 100)
    patience = params.get('early_stopping_patience', 15)
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_state_dict = None
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            # Save best state
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    
    # Final predictions
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_t).cpu().numpy().flatten()
        val_preds = model(X_val_t).cpu().numpy().flatten()
    
    # Metrics
    train_mse = float(np.mean((train_preds - y_train) ** 2))
    val_mse = float(np.mean((val_preds - y_val) ** 2))
    train_mae = float(np.mean(np.abs(train_preds - y_train)))
    val_mae = float(np.mean(np.abs(val_preds - y_val)))
    
    # R² score
    ss_res = np.sum((y_val - val_preds) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Save model if path provided
    checkpoint_path = None
    if save_path:
        # Save the model state dict (handles DataParallel)
        state_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            'model_state_dict': state_to_save,
            'input_dim': input_dim,
            'hidden_layers': hidden_layers,
            'dropout': dropout,
            'best_epoch': best_epoch
        }, save_path)
        checkpoint_path = save_path
    
    return {
        'model_type': 'neural_net',
        'device': device_used,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'r2': r2,
        'best_iteration': best_epoch,
        'architecture': hidden_layers,
        'checkpoint_path': checkpoint_path
    }


def main():
    """
    Main entry point for isolated trial worker.
    
    Parses args, loads data, trains model, outputs JSON result.
    """
    parser = argparse.ArgumentParser(
        description='Isolated Single-Trial Trainer - runs in subprocess for GPU isolation'
    )
    parser.add_argument('--model-type', required=True,
                        choices=['lightgbm', 'neural_net', 'xgboost', 'catboost'],
                        help='Model type to train')
    parser.add_argument('--data-path', required=True,
                        help='Path to .npz file with X_train, y_train, X_val, y_val')
    parser.add_argument('--params', type=str, default='{}',
                        help='JSON string of hyperparameters')
    parser.add_argument('--trial-number', type=int, default=-1,
                        help='Optuna trial number (for logging)')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Optional: write result to file instead of stdout')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress to stderr')
    # NEW: Model saving args
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model checkpoint')
    parser.add_argument('--model-output-dir', type=str, default=None,
                        help='Directory for model checkpoint (default: temp dir)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.verbose:
        print(f"[Trial {args.trial_number}] Starting {args.model_type} training...", file=sys.stderr)
    
    # Setup GPU environment BEFORE any ML imports
    setup_gpu_environment(args.model_type)
    
    # NOW import numpy (after GPU setup)
    import numpy as np
    
    # Load data
    try:
        data = np.load(args.data_path)
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
    except Exception as e:
        result = {
            'success': False,
            'model_type': args.model_type,
            'error': f'Failed to load data: {e}',
            'duration': time.time() - start_time
        }
        print(json.dumps(result))
        return 1
    
    # Parse hyperparameters
    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as e:
        result = {
            'success': False,
            'model_type': args.model_type,
            'error': f'Invalid params JSON: {e}',
            'duration': time.time() - start_time
        }
        print(json.dumps(result))
        return 1
    
    if args.verbose:
        print(f"[Trial {args.trial_number}] Data loaded: X_train={X_train.shape}, params={params}", file=sys.stderr)
    
    # Determine save path if saving model
    save_path = None
    if args.save_model:
        ext = MODEL_EXTENSIONS.get(args.model_type, '.bin')
        if args.model_output_dir:
            os.makedirs(args.model_output_dir, exist_ok=True)
            save_path = os.path.join(args.model_output_dir, f"{args.model_type}_trial{args.trial_number}{ext}")
        else:
            # Use temp directory
            save_path = os.path.join(tempfile.gettempdir(), f"{args.model_type}_trial{args.trial_number}_{int(time.time())}{ext}")
        
        if args.verbose:
            print(f"[Trial {args.trial_number}] Will save model to: {save_path}", file=sys.stderr)
    
    # Train based on model type
    try:
        if args.model_type == 'lightgbm':
            result = train_lightgbm(X_train, y_train, X_val, y_val, params, save_path)
        elif args.model_type == 'xgboost':
            result = train_xgboost(X_train, y_train, X_val, y_val, params, save_path)
        elif args.model_type == 'catboost':
            result = train_catboost(X_train, y_train, X_val, y_val, params, save_path)
        elif args.model_type == 'neural_net':
            result = train_neural_net(X_train, y_train, X_val, y_val, params, save_path)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        result['success'] = True
        result['trial_number'] = args.trial_number
        result['params'] = params
        result['duration'] = time.time() - start_time
        result['timestamp'] = datetime.now().isoformat()
        result['data_shape'] = {
            'X_train': list(X_train.shape),
            'X_val': list(X_val.shape)
        }
        
    except Exception as e:
        import traceback
        result = {
            'success': False,
            'model_type': args.model_type,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'duration': time.time() - start_time,
            'trial_number': args.trial_number,
            'checkpoint_path': None
        }
    
    # Output result
    output_json = json.dumps(result, indent=None)  # Compact for stdout parsing
    
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(result, f, indent=2)
        if args.verbose:
            print(f"[Trial {args.trial_number}] Result written to {args.output_path}", file=sys.stderr)
    
    # Always print to stdout for subprocess capture
    print(output_json)
    
    if args.verbose:
        status = '✅' if result.get('success') else '❌'
        print(f"[Trial {args.trial_number}] {status} Completed in {result['duration']:.2f}s", file=sys.stderr)
    
    return 0 if result.get('success') else 1


if __name__ == '__main__':
    sys.exit(main())
