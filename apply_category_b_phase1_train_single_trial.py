#!/usr/bin/env python3
"""
Category B Phase 1 — Patch train_single_trial.py
=================================================

Adds:
  1. --normalize-features flag (StandardScaler, fit on train only)
  2. --use-leaky-relu flag (passed through to SurvivorQualityNet)
  3. --dropout override flag (CLI precedence over Optuna/params)
  4. Checkpoint metadata: scaler_mean, scaler_scale, use_leaky_relu
  5. Scaler applied to X_val consistently

Team Beta Decision: Option A — normalize ON by default (not Optuna-searched)
Team Beta Decision: Store mean/scale arrays, NOT pickled sklearn objects

Author: Team Alpha (S92)
Date: 2026-02-15
"""

import re
import sys
import shutil
from pathlib import Path

TARGET = Path("train_single_trial.py")
BACKUP = TARGET.with_suffix(".pre_category_b_phase1")

def verify_preconditions():
    """Verify the file is in expected state before patching."""
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run from ~/distributed_prng_analysis/")
        return False

    content = TARGET.read_text()

    # Check version marker
    if '__version__ = "1.0.1"' not in content:
        print(f"ERROR: Unexpected version. Expected 1.0.1")
        return False

    # Check we haven't already patched
    if '--normalize-features' in content:
        print("ERROR: --normalize-features already present. Already patched?")
        return False

    if '--use-leaky-relu' in content:
        print("ERROR: --use-leaky-relu already present. Already patched?")
        return False

    # Check target anchor points exist
    anchors = [
        "parser.add_argument('--enable-diagnostics'",
        "def train_neural_net(",
        "model = SurvivorQualityNet(",
        "input_size=input_dim,",
        "'state_dict': state_to_save,",
    ]
    for anchor in anchors:
        if anchor not in content:
            print(f"ERROR: Missing anchor: {anchor}")
            return False

    print("All preconditions PASSED")
    return True


def apply_patch():
    """Apply Category B Phase 1 changes."""
    content = TARGET.read_text()

    # ================================================================
    # PATCH 1: Update version
    # ================================================================
    content = content.replace(
        '__version__ = "1.0.1"',
        '__version__ = "1.1.0"  # Category B Phase 1: normalize + leaky_relu + dropout override'
    )
    print("[1/6] Version bumped to 1.1.0")

    # ================================================================
    # PATCH 2: Add 3 new argparse flags after --enable-diagnostics
    # ================================================================
    old_diag_arg = """    parser.add_argument('--enable-diagnostics', action='store_true',
                        help='Enable Chapter 14 training diagnostics')"""

    new_diag_arg = """    parser.add_argument('--enable-diagnostics', action='store_true',
                        help='Enable Chapter 14 training diagnostics')
    # Category B: Neural net training enhancements (S92, Team Beta approved)
    parser.add_argument('--normalize-features', action='store_true',
                        help='Apply StandardScaler normalization before NN training (NN-only)')
    parser.add_argument('--use-leaky-relu', action='store_true',
                        help='Use LeakyReLU(0.01) instead of ReLU in neural net (NN-only)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override dropout value (takes precedence over params/Optuna)')"""

    if old_diag_arg not in content:
        print("ERROR: Cannot find --enable-diagnostics argument block")
        return False

    content = content.replace(old_diag_arg, new_diag_arg)
    print("[2/6] Added --normalize-features, --use-leaky-relu, --dropout argparse flags")

    # ================================================================
    # PATCH 3: Pass new flags to train_neural_net() call
    # ================================================================
    old_nn_call = """        elif args.model_type == 'neural_net':
            result = train_neural_net(X_train, y_train, X_val, y_val, params, save_path, getattr(args, 'enable_diagnostics', False))"""

    new_nn_call = """        elif args.model_type == 'neural_net':
            result = train_neural_net(
                X_train, y_train, X_val, y_val, params, save_path,
                getattr(args, 'enable_diagnostics', False),
                normalize_features=getattr(args, 'normalize_features', False),
                use_leaky_relu=getattr(args, 'use_leaky_relu', False),
                dropout_override=getattr(args, 'dropout', None),
            )"""

    if old_nn_call not in content:
        print("ERROR: Cannot find train_neural_net() call in main()")
        return False

    content = content.replace(old_nn_call, new_nn_call)
    print("[3/6] Updated train_neural_net() call with new parameters")

    # ================================================================
    # PATCH 4: Update train_neural_net() signature and add normalization
    # ================================================================
    old_nn_sig = """def train_neural_net(X_train, y_train, X_val, y_val, params: dict, save_path: str = None, enable_diagnostics: bool = False) -> dict:
    \"\"\"
    Train PyTorch Neural Net with GPU (CUDA).
    
    Imports torch HERE to ensure clean GPU state.
    Uses architecture similar to existing SurvivorQualityNet.
    \"\"\"
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
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    # Architecture from params (matches SurvivorQualityNet style)
    input_dim = X_train.shape[1]
    hidden_layers = params.get('hidden_layers', [256, 128, 64])
    dropout = params.get('dropout', 0.3)
    
    # Build model dynamically
    # Import and use existing SurvivorQualityNet for compatibility
    from models.wrappers.neural_net_wrapper import SurvivorQualityNet
    
    model = SurvivorQualityNet(
        input_size=input_dim,
        hidden_layers=hidden_layers,
        dropout=dropout
    ).to(device)"""

    new_nn_sig = """def train_neural_net(X_train, y_train, X_val, y_val, params: dict, save_path: str = None,
                     enable_diagnostics: bool = False, normalize_features: bool = False,
                     use_leaky_relu: bool = False, dropout_override: float = None) -> dict:
    \"\"\"
    Train PyTorch Neural Net with GPU (CUDA).
    
    Imports torch HERE to ensure clean GPU state.
    Uses architecture similar to existing SurvivorQualityNet.
    
    Category B enhancements (S92, Team Beta approved):
      - normalize_features: StandardScaler on X_train, applied to X_val
      - use_leaky_relu: LeakyReLU(0.01) instead of ReLU
      - dropout_override: CLI value takes precedence over params/Optuna
    \"\"\"
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
        print(f"[CAT-B] Activation: LeakyReLU(0.01)", file=sys.stderr)"""

    if old_nn_sig not in content:
        print("ERROR: Cannot find train_neural_net() function signature block")
        print("  Looking for exact match of old signature...")
        # Try to find partial match for debugging
        if "def train_neural_net(" in content:
            idx = content.index("def train_neural_net(")
            print(f"  Found at char {idx}, showing context:")
            print(content[idx:idx+200])
        return False

    content = content.replace(old_nn_sig, new_nn_sig)
    print("[4/6] Updated train_neural_net() with normalization + dropout override + leaky_relu")

    # ================================================================
    # PATCH 5: Update checkpoint save to include scaler metadata
    # ================================================================
    old_checkpoint_save = """        state_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            'state_dict': state_to_save,
            'feature_count': input_dim,
            'hidden_layers': hidden_layers,
            'dropout': dropout,
            'best_epoch': best_epoch
        }, save_path)
        checkpoint_path = save_path"""

    new_checkpoint_save = """        state_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
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
        checkpoint_path = save_path"""

    if old_checkpoint_save not in content:
        print("ERROR: Cannot find checkpoint save block")
        return False

    content = content.replace(old_checkpoint_save, new_checkpoint_save)
    print("[5/6] Updated checkpoint save with scaler metadata + activation flag")

    # ================================================================
    # PATCH 6: Add normalization/activation info to return dict
    # ================================================================
    old_return = """    return {
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
    }"""

    new_return = """    return {
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
    }"""

    if old_return not in content:
        print("ERROR: Cannot find neural_net return dict")
        return False

    content = content.replace(old_return, new_return)
    print("[6/6] Added Category B metadata to return dict")

    # ================================================================
    # Write patched file
    # ================================================================
    TARGET.write_text(content)
    print(f"\nAll 6 patches applied to {TARGET}")
    return True


def main():
    print("=" * 60)
    print("Category B Phase 1A: train_single_trial.py")
    print("=" * 60)

    if not verify_preconditions():
        sys.exit(1)

    # Create backup
    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    if not apply_patch():
        # Revert on failure
        shutil.copy2(BACKUP, TARGET)
        print("REVERTED to backup due to patch failure")
        sys.exit(1)

    # Syntax check
    import py_compile
    try:
        py_compile.compile(str(TARGET), doraise=True)
        print(f"Syntax check PASSED")
    except py_compile.PyCompileError as e:
        print(f"SYNTAX ERROR: {e}")
        shutil.copy2(BACKUP, TARGET)
        print("REVERTED to backup due to syntax error")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Phase 1A COMPLETE")
    print("=" * 60)
    print(f"  File: {TARGET}")
    print(f"  Version: 1.0.1 -> 1.1.0")
    print(f"  Backup: {BACKUP}")
    print(f"  New flags: --normalize-features, --use-leaky-relu, --dropout")
    print(f"  Checkpoint: +normalize_features, +use_leaky_relu, +scaler_mean, +scaler_scale")


if __name__ == "__main__":
    main()
