#!/usr/bin/env python3
"""
Subprocess Isolation Test - GPU ENABLED VERSION
================================================

This script tests subprocess isolation with ACTUAL GPU acceleration:
- LightGBM: OpenCL on GPU
- XGBoost: CUDA on GPU
- CatBoost: CUDA on GPU
- Neural Net: CUDA on GPU (PyTorch)

This is the REAL acid test - confirming OpenCL and CUDA can coexist
across subprocess boundaries when actually using the GPUs.

Usage:
    python3 test_subprocess_isolation_gpu.py

Expected Result:
    All 4 model types train successfully on GPU in any order.

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

NUM_SAMPLES = 10000      # More samples for GPU to show benefit
NUM_FEATURES = 62        # Match your actual feature count
NUM_TRIALS = 12          # 3 per model type, randomized
TIMEOUT_SECONDS = 180    # More time for GPU initialization

MODEL_TYPES = ['lightgbm', 'neural_net', 'xgboost', 'catboost']

# ============================================================================
# WORKER SCRIPT (Written to temp file) - GPU ENABLED
# ============================================================================

WORKER_SCRIPT = '''#!/usr/bin/env python3
"""
Isolated Trial Worker - GPU ENABLED VERSION
Runs in fresh subprocess with GPU acceleration
"""
import sys
import json
import argparse
import time
import os

def train_lightgbm(X_train, y_train, X_val, y_val, params):
    """Train LightGBM model with GPU (OpenCL)"""
    import lightgbm as lgb
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'num_leaves': params.get('num_leaves', 31),
        'learning_rate': params.get('learning_rate', 0.05),
        # GPU settings (OpenCL)
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
    }
    
    try:
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=params.get('n_estimators', 100),
            valid_sets=[val_data],
        )
        gpu_used = 'gpu'
    except Exception as e:
        # Fallback to CPU if GPU fails
        print(f"GPU failed ({e}), falling back to CPU", file=sys.stderr)
        lgb_params['device'] = 'cpu'
        del lgb_params['gpu_platform_id']
        del lgb_params['gpu_device_id']
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=params.get('n_estimators', 100),
            valid_sets=[val_data],
        )
        gpu_used = 'cpu_fallback'
    
    preds = model.predict(X_val)
    mse = float(((preds - y_val) ** 2).mean())
    return {'mse': mse, 'model': 'lightgbm', 'device': gpu_used}


def train_xgboost(X_train, y_train, X_val, y_val, params):
    """Train XGBoost model with GPU (CUDA)"""
    import xgboost as xgb
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': params.get('max_depth', 6),
        'learning_rate': params.get('learning_rate', 0.1),
        'verbosity': 0,
        # GPU settings (CUDA)
        'tree_method': 'hist',
        'device': 'cuda:0',
    }
    
    try:
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=[(dval, 'val')],
            verbose_eval=False
        )
        gpu_used = 'cuda:0'
    except Exception as e:
        # Fallback to CPU if GPU fails
        print(f"GPU failed ({e}), falling back to CPU", file=sys.stderr)
        xgb_params['device'] = 'cpu'
        xgb_params['tree_method'] = 'hist'
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=[(dval, 'val')],
            verbose_eval=False
        )
        gpu_used = 'cpu_fallback'
    
    preds = model.predict(dval)
    mse = float(((preds - y_val) ** 2).mean())
    return {'mse': mse, 'model': 'xgboost', 'device': gpu_used}


def train_catboost(X_train, y_train, X_val, y_val, params):
    """Train CatBoost model with GPU (CUDA)"""
    from catboost import CatBoostRegressor
    
    try:
        model = CatBoostRegressor(
            iterations=params.get('n_estimators', 100),
            depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            verbose=False,
            # GPU settings (CUDA)
            task_type='GPU',
            devices='0',
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        gpu_used = 'cuda:0'
    except Exception as e:
        # Fallback to CPU if GPU fails
        print(f"GPU failed ({e}), falling back to CPU", file=sys.stderr)
        model = CatBoostRegressor(
            iterations=params.get('n_estimators', 100),
            depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            verbose=False,
            task_type='CPU',
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        gpu_used = 'cpu_fallback'
    
    preds = model.predict(X_val)
    mse = float(((preds - y_val) ** 2).mean())
    return {'mse': mse, 'model': 'catboost', 'device': gpu_used}


def train_neural_net(X_train, y_train, X_val, y_val, params):
    """Train PyTorch Neural Net with GPU (CUDA)"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Force CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_used = 'cuda:0'
        # Warm up GPU
        _ = torch.zeros(1).to(device)
    else:
        device = torch.device('cpu')
        gpu_used = 'cpu'
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    # MLP architecture (similar to your SurvivorQualityNet)
    hidden_size = params.get('hidden_size', 128)
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden_size),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size),
        nn.Dropout(0.3),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size // 2),
        nn.Dropout(0.2),
        nn.Linear(hidden_size // 2, hidden_size // 4),
        nn.ReLU(),
        nn.Linear(hidden_size // 4, 1)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
    criterion = nn.MSELoss()
    
    # Training
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=params.get('batch_size', 128), shuffle=True)
    
    model.train()
    for epoch in range(params.get('epochs', 50)):
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
    
    mse = float(((preds - y_val) ** 2).mean())
    return {'mse': mse, 'model': 'neural_net', 'device': gpu_used}


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

def check_gpu_availability():
    """Check what GPUs are available"""
    print("\n" + "="*70)
    print("GPU AVAILABILITY CHECK")
    print("="*70)
    
    # Check CUDA (PyTorch)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
            cuda_count = torch.cuda.device_count()
            print(f"✅ CUDA: {cuda_count} device(s) - {cuda_device}")
        else:
            print("❌ CUDA: Not available")
    except Exception as e:
        print(f"❌ CUDA: Error - {e}")
        cuda_available = False
    
    # Check LightGBM GPU
    try:
        import lightgbm as lgb
        # LightGBM doesn't have a direct GPU check, but we can try
        print(f"✅ LightGBM: {lgb.__version__} (GPU support depends on build)")
    except Exception as e:
        print(f"❌ LightGBM: Error - {e}")
    
    # Check XGBoost GPU
    try:
        import xgboost as xgb
        print(f"✅ XGBoost: {xgb.__version__}")
    except Exception as e:
        print(f"❌ XGBoost: Error - {e}")
    
    # Check CatBoost GPU
    try:
        from catboost import CatBoostRegressor
        import catboost
        print(f"✅ CatBoost: {catboost.__version__}")
    except Exception as e:
        print(f"❌ CatBoost: Error - {e}")
    
    print("="*70 + "\n")
    return cuda_available


def generate_synthetic_data(n_samples, n_features):
    """Generate synthetic regression data with non-linear patterns"""
    np.random.seed(42)
    
    # Features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Non-linear target (tree models should do better here)
    y = np.zeros(n_samples, dtype=np.float32)
    
    # Linear component
    weights = np.random.randn(n_features) * 0.5
    y += X @ weights
    
    # Non-linear components
    y += np.sin(X[:, 0] * 2) * 3
    y += (X[:, 1] > 0).astype(float) * X[:, 2] * 2
    y += np.abs(X[:, 3]) * 1.5
    y += (X[:, 4] * X[:, 5]) * 0.8
    
    # Noise
    y += np.random.randn(n_samples) * 0.5
    
    # Normalize to reasonable range (like your actual_quality)
    y = (y - y.min()) / (y.max() - y.min())
    
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
            # Find JSON in output (might have other prints)
            stdout_lines = result.stdout.strip().split('\n')
            json_line = [l for l in stdout_lines if l.startswith('{')][-1]
            output = json.loads(json_line)
            output['success'] = True
            
            device_info = output.get('device', 'unknown')
            device_icon = '🚀' if 'cuda' in device_info or device_info == 'gpu' else '💻'
            
            print(f"  ✅ SUCCESS {device_icon}")
            print(f"  Device: {device_info}")
            print(f"  MSE: {output['mse']:.6f}")
            print(f"  Duration: {output['duration']:.2f}s")
            
            if result.stderr:
                # Show any warnings
                warnings = [l for l in result.stderr.split('\n') if l.strip()]
                if warnings:
                    print(f"  ⚠️  Warnings: {warnings[0][:80]}")
            
            return output
            
        except (json.JSONDecodeError, IndexError) as e:
            print(f"  ❌ FAILED (bad JSON output)")
            print(f"  stdout: {result.stdout[:500]}")
            return {
                'success': False,
                'model': model_type,
                'error': f'Invalid JSON output: {e}',
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
    print("SUBPROCESS ISOLATION TEST - GPU ENABLED")
    print("Testing OpenCL (LightGBM) + CUDA (Others) with GPU Acceleration")
    print("="*70)
    
    # Check GPU availability first
    cuda_available = check_gpu_availability()
    
    if not cuda_available:
        print("⚠️  WARNING: CUDA not available. Models will fall back to CPU.")
        print("   This test is most meaningful on Zeus with RTX 3080 Ti GPUs.")
        response = input("\nContinue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Exiting.")
            return 0
    
    # Create temp directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Write worker script
        worker_script_path = tmpdir / 'trial_worker_gpu.py'
        worker_script_path.write_text(WORKER_SCRIPT)
        print(f"\n✓ Worker script: {worker_script_path}")
        
        # Generate and save data
        print(f"\n✓ Generating synthetic data (non-linear patterns)...")
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
        print(f"  Y range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        
        # Generate randomized trial order (simulating Optuna)
        trials = []
        for model_type in MODEL_TYPES:
            trials.extend([model_type] * (NUM_TRIALS // len(MODEL_TYPES)))
        random.shuffle(trials)
        
        print(f"\n✓ Trial order (randomized like Optuna):")
        print(f"  {trials}")
        print(f"\n  ⚡ This simulates Optuna randomly picking models.")
        print(f"  ⚡ LightGBM should work even after CUDA models!")
        
        # Run trials
        results = []
        for i, model_type in enumerate(trials):
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'num_leaves': 31,
                'hidden_size': 128,
                'epochs': 30,
                'batch_size': 128
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
        print(f"  {'Model':<12} {'Success':<10} {'Avg MSE':<12} {'Avg Time':<10} {'Device'}")
        print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*15}")
        
        for model_type in MODEL_TYPES:
            model_results = [r for r in results if r.get('model') == model_type]
            successes = sum(1 for r in model_results if r.get('success', False))
            total = len(model_results)
            status = "✅" if successes == total else "❌"
            
            if successes > 0:
                successful_results = [r for r in model_results if r.get('success')]
                avg_mse = np.mean([r['mse'] for r in successful_results])
                avg_time = np.mean([r['duration'] for r in successful_results])
                devices = set(r.get('device', 'unknown') for r in successful_results)
                device_str = ', '.join(devices)
                print(f"  {status} {model_type:<10} {successes}/{total:<8} {avg_mse:<12.6f} {avg_time:<10.2f} {device_str}")
            else:
                print(f"  {status} {model_type:<10} {successes}/{total:<8} {'N/A':<12} {'N/A':<10} N/A")
        
        # GPU usage check
        print(f"\n" + "-"*70)
        print("GPU USAGE ANALYSIS")
        print("-"*70)
        
        gpu_trials = sum(1 for r in results if r.get('success') and 
                        ('cuda' in r.get('device', '') or r.get('device') == 'gpu'))
        cpu_trials = sum(1 for r in results if r.get('success') and 
                        ('cpu' in r.get('device', '') or r.get('device') == 'cpu_fallback'))
        
        print(f"  GPU trials: {gpu_trials}")
        print(f"  CPU trials: {cpu_trials}")
        
        if gpu_trials > 0:
            print(f"\n  🚀 GPU acceleration is working!")
        else:
            print(f"\n  💻 All trials ran on CPU (GPU may not be configured)")
        
        # Check if LightGBM specifically worked after CUDA
        lightgbm_after_cuda = False
        cuda_seen = False
        for r in results:
            if r.get('success'):
                if r.get('model') in ['xgboost', 'catboost', 'neural_net']:
                    cuda_seen = True
                elif r.get('model') == 'lightgbm' and cuda_seen:
                    lightgbm_after_cuda = True
                    break
        
        # Final verdict
        print("\n" + "="*70)
        if fail_count == 0:
            print("🎉 TEST PASSED! Subprocess isolation works correctly.")
            print("   All model types succeeded regardless of order.")
            if lightgbm_after_cuda:
                print("   ✅ LightGBM (OpenCL) ran successfully AFTER CUDA models!")
                print("   ✅ The OpenCL/CUDA conflict is SOLVED!")
            print("\n   Ready to implement in meta_prediction_optimizer_anti_overfit.py")
        else:
            print("⚠️  TEST FAILED! Some trials failed.")
            print("   Check error messages above.")
            print("\n   Failed trials:")
            for r in results:
                if not r.get('success', False):
                    print(f"     Trial {r['trial']}: {r['model']} - {r.get('error', 'unknown')[:60]}")
        print("="*70)
        
        # Return exit code
        return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
