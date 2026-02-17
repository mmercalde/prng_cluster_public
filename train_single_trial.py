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


# Chapter 14: Training diagnostics (best-effort, non-fatal)
try:
    from training_diagnostics import TrainingDiagnostics, TreeDiagnostics, NNDiagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    TreeDiagnostics = None
    NNDiagnostics = None


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
__version__ = "1.1.0"  # Category B Phase 1: normalize + leaky_relu + dropout override

# Model file extensions
MODEL_EXTENSIONS = {
    'neural_net': '.pth',
    'xgboost': '.json',
    'lightgbm': '.txt',
    'catboost': '.cbm',
    'random_forest': '.joblib'
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



# ------------------------------------------------------------------
# Chapter 14: Canonical diagnostics handoff for health check
# ------------------------------------------------------------------
def _write_canonical_diagnostics(src_path: str):
    """Write canonical diagnostics file for health check consumption."""
    try:
        import shutil, os
        os.makedirs("diagnostics_outputs", exist_ok=True)
        dst = "diagnostics_outputs/training_diagnostics.json"
        shutil.copyfile(src_path, dst)
    except Exception:
        pass  # best-effort, non-fatal


def _emit_tree_diagnostics(model, model_type: str, r2: float, mse: float, enable_diagnostics: bool):
    """Chapter 14: Emit tree model diagnostics (best-effort, non-fatal)."""
    if not enable_diagnostics or not DIAGNOSTICS_AVAILABLE:
        return
    try:
        import os
        os.makedirs('diagnostics_outputs', exist_ok=True)
        diag = TrainingDiagnostics.create(model_type)
        diag.attach(model)
        
        # Try to capture evals_result for models that support it
        if model_type == 'catboost' and hasattr(model, 'get_evals_result'):
            try:
                evals = model.get_evals_result()
                if evals:
                    keys = list(evals.keys())
                    learn_key = 'learn' if 'learn' in keys else keys[0]
                    val_key = 'validation' if 'validation' in keys else (keys[-1] if len(keys) > 1 else keys[0])
                    metric_keys = list(evals[learn_key].keys()) if evals.get(learn_key) else []
                    if metric_keys:
                        metric = metric_keys[0]
                        for i in range(len(evals.get(val_key, {}).get(metric, []))):
                            t = evals[learn_key][metric][i] if i < len(evals.get(learn_key, {}).get(metric, [])) else 0
                            v = evals[val_key][metric][i]
                            diag.on_round_end(i, t, v)
            except Exception:
                pass
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            try:
                imp = model.feature_importances_
                importance = {f'f{i}': float(v) for i, v in enumerate(imp)}
                diag.set_feature_importance(importance)
            except Exception:
                pass
        
        diag.set_final_metrics({'r2': r2, 'mse': mse})
        diag.detach()
        diag.save(f'diagnostics_outputs/{model_type}_diagnostics.json')
        _write_canonical_diagnostics(f'diagnostics_outputs/{model_type}_diagnostics.json')
        print(f"[DIAG] {model_type} diagnostics saved", file=sys.stderr)
    except Exception as e:
        print(f"[DIAG] {model_type} diagnostics failed (non-fatal): {e}", file=sys.stderr)


def _emit_nn_diagnostics(model, r2: float, mse: float, enable_diagnostics: bool):
    """Chapter 14: Emit neural net diagnostics (best-effort, non-fatal)."""
    if not enable_diagnostics or not DIAGNOSTICS_AVAILABLE:
        return
    try:
        import os
        os.makedirs('diagnostics_outputs', exist_ok=True)
        diag = NNDiagnostics()
        diag.set_final_metrics({'r2': r2, 'mse': mse})
        diag.save('diagnostics_outputs/neural_net_diagnostics.json')
        _write_canonical_diagnostics(f'diagnostics_outputs/neural_net_diagnostics.json')
        print(f"[DIAG] neural_net diagnostics saved", file=sys.stderr)
    except Exception as e:
        print(f"[DIAG] neural_net diagnostics failed (non-fatal): {e}", file=sys.stderr)


def train_lightgbm(X_train, y_train, X_val, y_val, params: dict, save_path: str = None, enable_diagnostics: bool = False) -> dict:
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
    

    # Chapter 14: Emit diagnostics
    _emit_tree_diagnostics(model, 'lightgbm', r2, val_mse, enable_diagnostics)

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


def train_xgboost(X_train, y_train, X_val, y_val, params: dict, save_path: str = None, enable_diagnostics: bool = False) -> dict:
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
    

    # Chapter 14: Emit diagnostics
    _emit_tree_diagnostics(model, 'xgboost', r2, val_mse, enable_diagnostics)

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


def train_catboost(X_train, y_train, X_val, y_val, params: dict, save_path: str = None, enable_diagnostics: bool = False) -> dict:
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
    

    # Chapter 14: Emit diagnostics
    _emit_tree_diagnostics(model, 'catboost', r2, val_mse, enable_diagnostics)

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


def train_neural_net(X_train, y_train, X_val, y_val, params: dict, save_path: str = None,
                     enable_diagnostics: bool = False, normalize_features: bool = False,
                     use_leaky_relu: bool = False, dropout_override: float = None) -> dict:
    """
    Train PyTorch Neural Net with GPU (CUDA).
    
    Imports torch HERE to ensure clean GPU state.
    Uses architecture similar to existing SurvivorQualityNet.
    
    Category B enhancements (S92, Team Beta approved):
      - normalize_features: StandardScaler on X_train, applied to X_val
      - use_leaky_relu: LeakyReLU(0.01) instead of ReLU
      - dropout_override: CLI value takes precedence over params/Optuna
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
    
    # ── Category B: Input normalization (StandardScaler) ──────────────
    # Team Beta Decision: store mean/scale arrays, NOT pickled sklearn
    scaler_mean = None
    scaler_scale = None
    if normalize_features:
        scaler_mean = X_train.mean(axis=0).astype(np.float32)
        scaler_scale = X_train.std(axis=0).astype(np.float32)
        # Guard: replace zero scale with 1.0 (Team Beta safety requirement)
        scaler_scale[scaler_scale == 0] = 1.0
        X_train = (X_train - scaler_mean) / scaler_scale
        X_val = (X_val - scaler_mean) / scaler_scale
        print(f"[CAT-B] Input normalization applied: {X_train.shape[1]} features, "
              f"scale range [{scaler_scale.min():.4f}, {scaler_scale.max():.4f}]",
              file=sys.stderr)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    # Architecture from params (matches SurvivorQualityNet style)
    input_dim = X_train.shape[1]
    hidden_layers = params.get('hidden_layers', [256, 128, 64])
    
    # ── Category B: Dropout override precedence ──────────────────────
    # CLI --dropout takes precedence over params/Optuna suggestion
    if dropout_override is not None:
        dropout = max(0.0, min(0.9, dropout_override))  # Clamp to [0.0, 0.9]
        if dropout != dropout_override:
            print(f"[CAT-B] Dropout clamped: {dropout_override} -> {dropout}", file=sys.stderr)
        else:
            print(f"[CAT-B] Dropout override: {dropout} (CLI precedence)", file=sys.stderr)
    else:
        dropout = params.get('dropout', 0.3)
    
    # Build model dynamically
    # Import and use existing SurvivorQualityNet for compatibility
    from models.wrappers.neural_net_wrapper import SurvivorQualityNet
    
    # ── Category B: LeakyReLU toggle ─────────────────────────────────
    model = SurvivorQualityNet(
        input_size=input_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
        use_leaky_relu=use_leaky_relu,
    ).to(device)
    
    if use_leaky_relu:
        print(f"[CAT-B] Activation: LeakyReLU(0.01)", file=sys.stderr)
    
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
    
    # S93: Wire NNDiagnostics hooks into training loop
    _nn_diag = None
    _base_model = model.module if hasattr(model, 'module') else model  # TB Fix #1: DataParallel
    if enable_diagnostics and DIAGNOSTICS_AVAILABLE:
        try:
            _nn_diag = NNDiagnostics()
            _nn_diag.attach(_base_model)
            print(f"[DIAG] NNDiagnostics attached ({len(_nn_diag._layer_names)} layers)", file=sys.stderr)
        except Exception as _diag_err:
            print(f"[DIAG] NNDiagnostics attach failed (non-fatal): {_diag_err}", file=sys.stderr)
            _nn_diag = None

    for epoch in range(epochs):
        model.train()
        _epoch_train_loss = 0.0
        _epoch_batches = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            _epoch_train_loss += loss.item()
            _epoch_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
        
        # S93: Record epoch diagnostics
        if _nn_diag is not None:
            try:
                _avg_train_loss = _epoch_train_loss / max(_epoch_batches, 1)
                _nn_diag.on_round_end(
                    round_num=epoch,
                    train_loss=_avg_train_loss,
                    val_loss=val_loss,
                    learning_rate=lr,
                )
            except Exception as _rnd_err:
                if epoch == 0:
                    print(f"[DIAG] on_round_end failed (non-fatal): {_rnd_err}", file=sys.stderr)

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
        checkpoint_data = {
            'state_dict': state_to_save,
            'feature_count': input_dim,
            'hidden_layers': hidden_layers,
            'dropout': dropout,
            'best_epoch': best_epoch,
            # Category B: normalization + activation metadata for Step 6 portability
            'normalize_features': normalize_features,
            'use_leaky_relu': use_leaky_relu,
        }
        # Team Beta Decision: store mean/scale as arrays, NOT pickled sklearn
        if scaler_mean is not None:
            checkpoint_data['scaler_mean'] = scaler_mean  # numpy array
            checkpoint_data['scaler_scale'] = scaler_scale  # numpy array
        torch.save(checkpoint_data, save_path)
        checkpoint_path = save_path
    

    # Chapter 14: Emit diagnostics
    # S93: Save live diagnostics (replaces empty post-hoc stub)
    # TB Fix C: Always detach hooks (separate try)
    if _nn_diag is not None:
        try:
            _nn_diag.detach()
        except Exception:
            pass
    # TB Fix #2: Robust MSE resolution (None if neither exists — v3 nice-to-have B)
    if _nn_diag is not None and enable_diagnostics:
        try:
            _mse_val = locals().get('val_mse', locals().get('mse', None))
            _nn_diag.set_final_metrics({'r2': r2, 'mse': _mse_val})
            os.makedirs('diagnostics_outputs', exist_ok=True)
            _nn_diag.save('diagnostics_outputs/neural_net_diagnostics.json')
            # v3 nice-to-have A: check existence before calling
            if '_write_canonical_diagnostics' in dir() or '_write_canonical_diagnostics' in globals():
                _write_canonical_diagnostics('diagnostics_outputs/neural_net_diagnostics.json')
            # TB Fix B: try-wrapped private attr access
            try:
                _rnd_count = len(_nn_diag._round_data)
                _lyr_count = len(_nn_diag._layer_names)
            except Exception:
                _rnd_count = _lyr_count = '?'
            print(f"[DIAG] NN diagnostics saved: {_rnd_count} rounds, {_lyr_count} layers", file=sys.stderr)
        except Exception as _save_err:
            print(f"[DIAG] NN diagnostics save failed (non-fatal): {_save_err}", file=sys.stderr)
    elif enable_diagnostics:
        _emit_nn_diagnostics(model, r2, locals().get('val_mse', locals().get('mse', None)), enable_diagnostics)

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
        'checkpoint_path': checkpoint_path,
        # Category B metadata
        'normalize_features': normalize_features,
        'use_leaky_relu': use_leaky_relu,
        'dropout_source': 'cli_override' if dropout_override is not None else 'params',
    }




def train_random_forest(X_train, y_train, X_val, y_val, params: dict, save_path: str = None, enable_diagnostics: bool = False) -> dict:
    """
    Train Random Forest model (CPU-based, no GPU conflicts).
    
    Imports sklearn HERE to ensure clean state.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    import joblib
    
    start_time = time.time()
    
    # Extract RF-specific params
    rf_params = {
        "n_estimators": params.get("rf_n_estimators", 100),
        "max_depth": params.get("rf_max_depth", None),
        "min_samples_split": params.get("rf_min_samples_split", 2),
        "min_samples_leaf": params.get("rf_min_samples_leaf", 1),
        "max_features": params.get("rf_max_features", "sqrt"),
        "n_jobs": params.get("rf_n_jobs", -1),
        "random_state": params.get("rf_random_state", 42),
    }
    
    # Train model
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_pred)
    val_r2 = r2_score(y_val, y_pred)
    
    duration = time.time() - start_time
    
    # Save model if path provided
    checkpoint_path = None
    if save_path:
        checkpoint_path = save_path if save_path.endswith(".joblib") else f"{save_path}.joblib"
        joblib.dump(model, checkpoint_path)
    
    return {
        "val_mse": float(val_mse),
        "r2": float(val_r2),
        "duration": duration,
        "device": "cpu",
        "model_type": "random_forest",
        "checkpoint_path": checkpoint_path,
        "feature_importances": model.feature_importances_.tolist() if hasattr(model, "feature_importances_") else None
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
                        choices=['lightgbm', 'neural_net', 'xgboost', 'catboost', 'random_forest'],
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
    # Chapter 14: Training diagnostics
    parser.add_argument('--enable-diagnostics', action='store_true',
                        help='Enable Chapter 14 training diagnostics')
    # Category B: Neural net training enhancements (S92, Team Beta approved)
    parser.add_argument('--normalize-features', action='store_true',
                        help='Apply StandardScaler normalization before NN training (NN-only)')
    parser.add_argument('--use-leaky-relu', action='store_true',
                        help='Use LeakyReLU(0.01) instead of ReLU in neural net (NN-only)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override dropout value (takes precedence over params/Optuna)')
    
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
            result = train_lightgbm(X_train, y_train, X_val, y_val, params, save_path, getattr(args, 'enable_diagnostics', False))
        elif args.model_type == 'xgboost':
            result = train_xgboost(X_train, y_train, X_val, y_val, params, save_path, getattr(args, 'enable_diagnostics', False))
        elif args.model_type == 'catboost':
            result = train_catboost(X_train, y_train, X_val, y_val, params, save_path, getattr(args, 'enable_diagnostics', False))
        elif args.model_type == 'neural_net':
            result = train_neural_net(
                X_train, y_train, X_val, y_val, params, save_path,
                getattr(args, 'enable_diagnostics', False),
                normalize_features=getattr(args, 'normalize_features', False),
                use_leaky_relu=getattr(args, 'use_leaky_relu', False),
                dropout_override=getattr(args, 'dropout', None),
            )
        elif args.model_type == 'random_forest':
            result = train_random_forest(X_train, y_train, X_val, y_val, params, save_path, getattr(args, 'enable_diagnostics', False))
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
