#!/usr/bin/env python3
"""
GPU Capability Test for Remote Rigs
Tests PyTorch, XGBoost, LightGBM, CatBoost GPU support on ROCm

Run on rig:
    source ~/rocm_env/bin/activate
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 test_gpu_capability.py
"""

import os
import sys
import socket
import time

# ============================================================
# ROCm Environment Setup - MUST BE FIRST
# ============================================================
HOST = socket.gethostname()
print(f"Host: {HOST}")

if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", 
                          "garbage_collection_threshold:0.8,max_split_size_mb:128")

os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")

print("="*60)
print("GPU CAPABILITY TEST")
print("="*60)

# ============================================================
# Test Data (small, realistic size)
# ============================================================
import numpy as np
np.random.seed(42)

N_SAMPLES = 3000
N_FEATURES = 47

X = np.random.randn(N_SAMPLES, N_FEATURES).astype(np.float32)
y = np.random.randn(N_SAMPLES).astype(np.float32)

print(f"\nTest data: {N_SAMPLES} samples × {N_FEATURES} features")

# ============================================================
# 1. PyTorch / ROCm Test
# ============================================================
print("\n" + "="*60)
print("1. PyTorch (ROCm)")
print("="*60)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA/ROCm available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Device count: {device_count}")
        
        for i in range(min(device_count, 3)):  # Show first 3
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        
        # Limit VRAM for RX 6600
        if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]:
            torch.cuda.set_per_process_memory_fraction(0.8)
            print("VRAM limited to 80% (6.4GB)")
        
        # Quick training test
        print("\nTraining small neural net...")
        
        X_tensor = torch.tensor(X, device='cuda')
        y_tensor = torch.tensor(y, device='cuda')
        
        model = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ).cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        start = time.time()
        for epoch in range(100):
            optimizer.zero_grad()
            pred = model(X_tensor).squeeze()
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        elapsed = time.time() - start
        print(f"✅ PyTorch GPU: {elapsed:.2f}s for 100 epochs")
        print(f"   Final loss: {loss.item():.6f}")
        
        # Cleanup
        del X_tensor, y_tensor, model
        torch.cuda.empty_cache()
    else:
        print("❌ PyTorch: No GPU available")

except Exception as e:
    print(f"❌ PyTorch error: {e}")

# ============================================================
# 2. XGBoost Test
# ============================================================
print("\n" + "="*60)
print("2. XGBoost")
print("="*60)

try:
    import xgboost as xgb
    print(f"XGBoost version: {xgb.__version__}")
    
    # Try GPU first
    gpu_works = False
    
    try:
        print("Attempting GPU training (tree_method='hist', device='cuda')...")
        start = time.time()
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            tree_method='hist',
            device='cuda',
            verbosity=0
        )
        model.fit(X, y)
        
        elapsed = time.time() - start
        print(f"✅ XGBoost GPU: {elapsed:.2f}s for 100 trees")
        gpu_works = True
        
    except Exception as e:
        print(f"⚠️ XGBoost GPU failed: {e}")
        print("Falling back to CPU...")
        
        start = time.time()
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            tree_method='hist',
            device='cpu',
            verbosity=0
        )
        model.fit(X, y)
        
        elapsed = time.time() - start
        print(f"✅ XGBoost CPU: {elapsed:.2f}s for 100 trees")

except Exception as e:
    print(f"❌ XGBoost error: {e}")

# ============================================================
# 3. LightGBM Test
# ============================================================
print("\n" + "="*60)
print("3. LightGBM")
print("="*60)

try:
    import lightgbm as lgb
    print(f"LightGBM version: {lgb.__version__}")
    
    # Try GPU first
    gpu_works = False
    
    try:
        print("Attempting GPU training (device='gpu')...")
        start = time.time()
        
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            device='gpu',
            verbose=-1
        )
        model.fit(X, y)
        
        elapsed = time.time() - start
        print(f"✅ LightGBM GPU: {elapsed:.2f}s for 100 trees")
        gpu_works = True
        
    except Exception as e:
        print(f"⚠️ LightGBM GPU failed: {e}")
        print("Falling back to CPU...")
        
        start = time.time()
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            device='cpu',
            verbose=-1
        )
        model.fit(X, y)
        
        elapsed = time.time() - start
        print(f"✅ LightGBM CPU: {elapsed:.2f}s for 100 trees")

except Exception as e:
    print(f"❌ LightGBM error: {e}")

# ============================================================
# 4. CatBoost Test
# ============================================================
print("\n" + "="*60)
print("4. CatBoost")
print("="*60)

try:
    from catboost import CatBoostRegressor
    import catboost
    print(f"CatBoost version: {catboost.__version__}")
    
    # Try GPU first
    gpu_works = False
    
    try:
        print("Attempting GPU training (task_type='GPU')...")
        start = time.time()
        
        model = CatBoostRegressor(
            iterations=100,
            depth=6,
            task_type='GPU',
            devices='0',
            verbose=0
        )
        model.fit(X, y)
        
        elapsed = time.time() - start
        print(f"✅ CatBoost GPU: {elapsed:.2f}s for 100 iterations")
        gpu_works = True
        
    except Exception as e:
        print(f"⚠️ CatBoost GPU failed: {e}")
        print("Falling back to CPU...")
        
        start = time.time()
        model = CatBoostRegressor(
            iterations=100,
            depth=6,
            task_type='CPU',
            verbose=0
        )
        model.fit(X, y)
        
        elapsed = time.time() - start
        print(f"✅ CatBoost CPU: {elapsed:.2f}s for 100 iterations")

except Exception as e:
    print(f"❌ CatBoost error: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Host: {HOST}")
print(f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
