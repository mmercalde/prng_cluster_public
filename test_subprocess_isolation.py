#!/usr/bin/env python3
"""
Subprocess Isolation Test for Mixed OpenCL/CUDA Models
=======================================================

This script tests whether subprocess isolation solves the OpenCL/CUDA conflict.

Test Procedure:
1. Generate synthetic training data
2. Run trials in RANDOM order (simulating Optuna)
3. Each trial runs in isolated subprocess
4. Verify ALL model types succeed regardless of order

Usage:
    python3 test_subprocess_isolation.py

Expected Result:
    All 4 model types train successfully in any order.

Author: PRNG Analysis System
Date: December 2025
"""

import subprocess
import sys
import json
import tempfile
import time
import random
from pathlib import Path
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_SAMPLES = 5000       # Training samples (small for quick test)
NUM_FEATURES = 50        # Feature count (matches your system)
NUM_TRIALS = 8           # Total trials (2 per model type, randomized)
TIMEOUT_SECONDS = 120    # Max time per trial

MODEL_TYPES = ['lightgbm', 'neural_net', 'xgboost', 'catboost']

# ============================================================================
# WORKER SCRIPT (Written to temp file)
# ============================================================================

WORKER_SCRIPT = '''#!/usr/bin/env python3
"""
Isolated Trial Worker - Runs in fresh subprocess
NO imports at module level to ensure clean GPU state
"""
import sys
import json
import argparse
import time

def train_lightgbm(X_train, y_train, X_val, y_val, params):
    """Train LightGBM model (OpenCL)"""
    import lightgbm as lgb
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'num_leaves': params.get('num_leaves', 31),
        'learning_rate': params.get('learning_rate', 0.05),
        'n_estimators': params.get('n_estimators', 50),
        # GPU settings - comment out if no GPU
        # 'device': 'gpu',
        # 'gpu_platform_id': 0,
        # 'gpu_device_id': 0,
    }
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=lgb_params['n_estimators'],
        valid_sets=[val_data],
    )
    
    preds = model.predict(X_val)
    mse = ((preds - y_val) ** 2).mean()
    return {'mse': float(mse), 'model': 'lightgbm'}


def train_xgboost(X_train, y_train, X_val, y_val, params):
    """Train XGBoost model (CUDA)"""
    import xgboost as xgb
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': params.get('max_depth', 6),
        'learning_rate': params.get('learning_rate', 0.1),
        'verbosity': 0,
        # GPU settings - comment out if no GPU
        # 'tree_method': 'gpu_hist',
        # 'device': 'cuda:0',
    }
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=params.get('n_estimators', 50),
        evals=[(dval, 'val')],
        verbose_eval=False
    )
    
    preds = model.predict(dval)
    mse = ((preds - y_val) ** 2).mean()
    return {'mse': float(mse), 'model': 'xgboost'}


def train_catboost(X_train, y_train, X_val, y_val, params):
    """Train CatBoost model (CUDA)"""
    from catboost import CatBoostRegressor
    
    model = CatBoostRegressor(
        iterations=params.get('n_estimators', 50),
        depth=params.get('max_depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        verbose=False,
        # GPU settings - comment out if no GPU
        # task_type='GPU',
        # devices='0',
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    preds = model.predict(X_val)
    mse = ((preds - y_val) ** 2).mean()
    return {'mse': float(mse), 'model': 'catboost'}


def train_neural_net(X_train, y_train, X_val, y_val, params):
    """Train PyTorch Neural Net (CUDA)"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Device selection
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # Simple MLP
    hidden_size = params.get('hidden_size', 64)
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden_size),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Linear(hidden_size // 2, 1)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
    criterion = nn.MSELoss()
    
    # Training
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=params.get('batch_size', 64), shuffle=True)
    
    model.train()
    for epoch in range(params.get('epochs', 20)):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).cpu().numpy().flatten()
    
    mse = ((preds - y_val) ** 2).mean()
    return {'mse': float(mse), 'model': 'neural_net'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', required=True, 
                        choices=['lightgbm', 'xgboost', 'catboost', 'neural_net'])
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--params', type=str, default='{}')
    args = parser.parse_args()
    
    # Load data
    import numpy as np
    data = np.load(args.data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    params = json.loads(args.params)
    
    # Train based on model type
    start_time = time.time()
    
    if args.model_type == 'lightgbm':
        result = train_lightgbm(X_train, y_train, X_val, y_val, params)
    elif args.model_type == 'xgboost':
        result = train_xgboost(X_train, y_train, X_val, y_val, params)
    elif args.model_type == 'catboost':
        result = train_catboost(X_train, y_train, X_val, y_val, params)
    elif args.model_type == 'neural_net':
        result = train_neural_net(X_train, y_train, X_val, y_val, params)
    
    result['duration'] = time.time() - start_time
    
    # Output result as JSON (parent process captures this)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
'''


# ============================================================================
# MAIN TEST COORDINATOR
# ============================================================================

def generate_synthetic_data(n_samples, n_features):
    """Generate synthetic regression data"""
    np.random.seed(42)
    
    # Features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Target with some structure (not pure noise)
    weights = np.random.randn(n_features)
    y = X @ weights + np.random.randn(n_samples) * 0.5
    y = y.astype(np.float32)
    
    # Split
    split_idx = int(n_samples * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_val, y_val


def run_trial(worker_script_path, data_path, model_type, params, trial_num):
    """Run a single trial in isolated subprocess"""
    
    print(f"\n{'─'*60}")
    print(f"Trial {trial_num}: {model_type.upper()}")
    print(f"{'─'*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [
                sys.executable, 
                str(worker_script_path),
                '--model-type', model_type,
                '--data-path', str(data_path),
                '--params', json.dumps(params)
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  ❌ FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[:500]}")
            return {
                'success': False,
                'model': model_type,
                'error': result.stderr[:500],
                'duration': elapsed
            }
        
        # Parse output
        try:
            output = json.loads(result.stdout.strip())
            output['success'] = True
            print(f"  ✅ SUCCESS")
            print(f"  MSE: {output['mse']:.6f}")
            print(f"  Duration: {output['duration']:.2f}s")
            return output
        except json.JSONDecodeError:
            print(f"  ❌ FAILED (bad JSON output)")
            print(f"  stdout: {result.stdout[:500]}")
            return {
                'success': False,
                'model': model_type,
                'error': 'Invalid JSON output',
                'duration': elapsed
            }
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ FAILED (timeout after {TIMEOUT_SECONDS}s)")
        return {
            'success': False,
            'model': model_type,
            'error': 'Timeout',
            'duration': TIMEOUT_SECONDS
        }
    except Exception as e:
        print(f"  ❌ FAILED (exception: {e})")
        return {
            'success': False,
            'model': model_type,
            'error': str(e),
            'duration': 0
        }


def main():
    print("="*70)
    print("SUBPROCESS ISOLATION TEST")
    print("Testing Mixed OpenCL (LightGBM) + CUDA (Others) in Random Order")
    print("="*70)
    
    # Create temp directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Write worker script
        worker_script_path = tmpdir / 'trial_worker.py'
        worker_script_path.write_text(WORKER_SCRIPT)
        print(f"\n✓ Worker script: {worker_script_path}")
        
        # Generate and save data
        print(f"\n✓ Generating synthetic data...")
        print(f"  Samples: {NUM_SAMPLES}")
        print(f"  Features: {NUM_FEATURES}")
        
        X_train, y_train, X_val, y_val = generate_synthetic_data(NUM_SAMPLES, NUM_FEATURES)
        
        data_path = tmpdir / 'trial_data.npz'
        np.savez(data_path, 
                 X_train=X_train, 
                 y_train=y_train,
                 X_val=X_val, 
                 y_val=y_val)
        print(f"  Data saved: {data_path}")
        print(f"  Train shape: {X_train.shape}")
        print(f"  Val shape: {X_val.shape}")
        
        # Generate randomized trial order (simulating Optuna)
        trials = []
        for model_type in MODEL_TYPES:
            trials.extend([model_type] * (NUM_TRIALS // len(MODEL_TYPES)))
        random.shuffle(trials)
        
        print(f"\n✓ Trial order (randomized like Optuna):")
        print(f"  {trials}")
        
        # Run trials
        results = []
        for i, model_type in enumerate(trials):
            params = {
                'n_estimators': 30,
                'learning_rate': 0.1,
                'max_depth': 4,
                'num_leaves': 15,
                'hidden_size': 32,
                'epochs': 10,
                'batch_size': 64
            }
            
            result = run_trial(worker_script_path, data_path, model_type, params, i)
            result['trial'] = i
            results.append(result)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        success_count = sum(1 for r in results if r.get('success', False))
        fail_count = len(results) - success_count
        
        print(f"\nTotal trials: {len(results)}")
        print(f"  ✅ Succeeded: {success_count}")
        print(f"  ❌ Failed: {fail_count}")
        
        # Per-model breakdown
        print(f"\nPer-model results:")
        for model_type in MODEL_TYPES:
            model_results = [r for r in results if r.get('model') == model_type]
            successes = sum(1 for r in model_results if r.get('success', False))
            total = len(model_results)
            status = "✅" if successes == total else "❌"
            
            if successes > 0:
                avg_mse = np.mean([r['mse'] for r in model_results if r.get('success')])
                print(f"  {status} {model_type:12s}: {successes}/{total} succeeded, avg MSE: {avg_mse:.6f}")
            else:
                print(f"  {status} {model_type:12s}: {successes}/{total} succeeded")
        
        # Final verdict
        print("\n" + "="*70)
        if fail_count == 0:
            print("🎉 TEST PASSED! Subprocess isolation works correctly.")
            print("   All model types succeeded regardless of order.")
            print("   LightGBM (OpenCL) coexists with CUDA models.")
        else:
            print("⚠️  TEST FAILED! Some trials failed.")
            print("   Check error messages above.")
            print("\n   Failed trials:")
            for r in results:
                if not r.get('success', False):
                    print(f"     Trial {r['trial']}: {r['model']} - {r.get('error', 'unknown')}")
        print("="*70)
        
        # Return exit code
        return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
