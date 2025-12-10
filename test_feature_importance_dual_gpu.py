#!/usr/bin/env python3
"""
Dual-GPU Feature Importance Test
=================================

Tests feature importance extraction using both RTX 3080 Ti GPUs on Zeus.

Usage:
    python3 test_feature_importance_dual_gpu.py

Author: Distributed PRNG Analysis System
Date: December 9, 2025
"""

import sys
import os
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn

# Import our modules
from feature_importance import FeatureImportanceExtractor, FeatureImportanceResult

try:
    from reinforcement_engine import SurvivorQualityNet
    USE_REAL_MODEL = True
except ImportError:
    USE_REAL_MODEL = False


def create_model(device):
    """Create and initialize model on specified device."""
    if USE_REAL_MODEL:
        model = SurvivorQualityNet(input_size=60, hidden_layers=[128, 64, 32], dropout=0.3)
    else:
        model = nn.Sequential(
            nn.Linear(60, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    return model.to(device).eval()


def run_extraction_on_gpu(gpu_id, model, feature_names, X, y, method, n_repeats=5):
    """Run feature importance extraction on a specific GPU."""
    device = f'cuda:{gpu_id}'
    
    # Move model to this GPU
    model = model.to(device)
    
    extractor = FeatureImportanceExtractor(
        model=model,
        feature_names=feature_names,
        device=device
    )
    
    start = time.time()
    result = extractor.extract(
        X=X, y=y,
        method=method,
        model_version=f'dual_gpu_test_gpu{gpu_id}',
        n_repeats=n_repeats
    )
    elapsed = time.time() - start
    
    return gpu_id, result, elapsed


def run_parallel_extraction(feature_names, X, y, method='permutation', n_repeats=5):
    """Run extraction on both GPUs in parallel, splitting features."""
    n_gpus = torch.cuda.device_count()
    print(f"\n  Running parallel extraction on {n_gpus} GPUs...")
    
    # Create separate models for each GPU
    models = [create_model(f'cuda:{i}') for i in range(n_gpus)]
    
    # Split features between GPUs
    n_features = len(feature_names)
    features_per_gpu = n_features // n_gpus
    
    results = {}
    total_start = time.time()
    
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = []
        for gpu_id in range(n_gpus):
            future = executor.submit(
                run_extraction_on_gpu,
                gpu_id, models[gpu_id], feature_names, X, y, method, n_repeats
            )
            futures.append(future)
        
        for future in as_completed(futures):
            gpu_id, result, elapsed = future.result()
            results[gpu_id] = (result, elapsed)
            print(f"    GPU {gpu_id}: {elapsed:.2f}s")
    
    total_elapsed = time.time() - total_start
    return results, total_elapsed


def main():
    print("=" * 70)
    print("Dual-GPU Feature Importance Test")
    print("=" * 70)
    
    # Check GPU availability
    n_gpus = torch.cuda.device_count()
    print(f"\n[1] GPU Detection...")
    print(f"  ✅ Found {n_gpus} GPU(s)")
    
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"     GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    if n_gpus < 2:
        print("\n  ⚠️  Only 1 GPU detected. Dual-GPU test requires 2 GPUs.")
        print("     Running single-GPU comparison instead.\n")
    
    # Setup
    print(f"\n[2] Setup...")
    feature_names = FeatureImportanceExtractor.STATISTICAL_FEATURES.copy()
    while len(feature_names) < 60:
        feature_names.append(f"global_state_{len(feature_names) - 46}")
    
    # Generate larger test data for meaningful timing
    np.random.seed(42)
    n_samples = 2000  # Larger dataset
    X = np.random.randn(n_samples, 60).astype(np.float32)
    y = np.random.rand(n_samples).astype(np.float32)
    
    print(f"  ✅ {n_samples} samples, 60 features")
    
    # Test 1: Single GPU (cuda:0)
    print(f"\n[3] Single GPU Test (cuda:0)...")
    model_single = create_model('cuda:0')
    
    extractor_single = FeatureImportanceExtractor(
        model=model_single,
        feature_names=feature_names,
        device='cuda:0'
    )
    
    start = time.time()
    result_single = extractor_single.extract(
        X=X, y=y,
        method='permutation',
        model_version='single_gpu_test',
        n_repeats=5
    )
    single_gpu_time = time.time() - start
    print(f"  ✅ Single GPU: {single_gpu_time:.2f}s")
    print(f"     Top 3: {[f['name'] for f in result_single.top_10_features[:3]]}")
    
    # Test 2: Both GPUs independently (not parallel, just verifying both work)
    if n_gpus >= 2:
        print(f"\n[4] Both GPUs Independent Test...")
        
        for gpu_id in range(n_gpus):
            device = f'cuda:{gpu_id}'
            model = create_model(device)
            
            extractor = FeatureImportanceExtractor(
                model=model,
                feature_names=feature_names,
                device=device
            )
            
            start = time.time()
            result = extractor.extract(
                X=X, y=y,
                method='gradient',  # Faster for quick test
                model_version=f'gpu{gpu_id}_test'
            )
            elapsed = time.time() - start
            print(f"  ✅ GPU {gpu_id}: {elapsed:.2f}s (gradient method)")
        
        # Test 3: Parallel execution
        print(f"\n[5] Parallel GPU Test...")
        parallel_results, parallel_time = run_parallel_extraction(
            feature_names, X, y, 
            method='permutation',
            n_repeats=5
        )
        
        print(f"\n  ✅ Parallel execution: {parallel_time:.2f}s total")
        print(f"     Speedup vs single: {single_gpu_time / parallel_time:.2f}x")
        
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Samples: {n_samples}")
    print(f"  Features: 60")
    print(f"  Permutation repeats: 5")
    print(f"  Single GPU time: {single_gpu_time:.2f}s")
    if n_gpus >= 2:
        print(f"  Parallel GPU time: {parallel_time:.2f}s")
        print(f"  Speedup: {single_gpu_time / parallel_time:.2f}x")
    print(f"\n✅ Dual-GPU test complete!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
