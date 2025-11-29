#!/usr/bin/env python3
"""
Test PyTorch GPU PRNG Implementations
======================================
Tests correctness and performance of PyTorch GPU functions in prng_registry.py

Tests:
1. Correctness: GPU output matches CPU reference
2. Performance: GPU vs CPU benchmark
3. Device compatibility: CUDA and/or CPU
4. Batch sizes: 10, 100, 1000, 10000 seeds

Usage:
    python3 test_pytorch_gpu_prngs.py
    python3 test_pytorch_gpu_prngs.py --prng java_lcg --device cuda
    python3 test_pytorch_gpu_prngs.py --benchmark-only
"""

import sys
import time
import argparse
import numpy as np

# Add current directory to path
sys.path.insert(0, '.')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not installed!")
    print("   Install with: pip install torch --break-system-packages")
    sys.exit(1)

from prng_registry import (
    get_cpu_reference,
    get_pytorch_gpu_reference,
    has_pytorch_gpu,
    list_pytorch_gpu_prngs,
    get_kernel_info,
    PYTORCH_AVAILABLE
)


def test_correctness(prng_name, device='cuda', n_seeds=100, n_draws=500, mod=1000):
    """
    Test that PyTorch GPU output matches CPU reference.
    
    Args:
        prng_name: PRNG to test (e.g., 'java_lcg')
        device: 'cuda' or 'cpu'
        n_seeds: Number of seeds to test
        n_draws: Number of draws per seed
        mod: Modulo for lottery output
    
    Returns:
        Tuple (passed: bool, match_rate: float, time_cpu: float, time_gpu: float)
    """
    print(f"\n{'='*70}")
    print(f"CORRECTNESS TEST: {prng_name} on {device}")
    print(f"{'='*70}")
    
    if not has_pytorch_gpu(prng_name):
        print(f"❌ {prng_name} does not have PyTorch GPU implementation")
        return False, 0.0, 0, 0
    
    # Get functions
    cpu_func = get_cpu_reference(prng_name)
    gpu_func = get_pytorch_gpu_reference(prng_name)
    prng_info = get_kernel_info(prng_name)
    prng_params = prng_info.get('default_params', {})
    
    # Generate test seeds
    np.random.seed(42)
    test_seeds = np.random.randint(1, 1000000, size=n_seeds)
    
    print(f"Testing {n_seeds} seeds × {n_draws} draws each...")
    print(f"PRNG params: {prng_params}")
    
    # CPU Reference
    print(f"\n1. Generating CPU reference...")
    start = time.time()
    cpu_results = []
    for seed in test_seeds:
        sequence = cpu_func(seed=int(seed), n=n_draws, skip=0, **prng_params)
        cpu_results.append([val % mod for val in sequence])
    cpu_results = np.array(cpu_results, dtype=np.int64)
    time_cpu = time.time() - start
    print(f"   CPU: {time_cpu:.3f}s ({n_seeds/time_cpu:.1f} seeds/sec)")
    
    # GPU Implementation
    print(f"\n2. Generating GPU output...")
    seeds_tensor = torch.tensor(test_seeds, dtype=torch.int64, device=device)
    start = time.time()
    gpu_results = gpu_func(
        seeds=seeds_tensor,
        n=n_draws,
        mod=mod,
        device=device,
        skip=0,
        **prng_params
    )
    gpu_results_np = gpu_results.cpu().numpy()
    time_gpu = time.time() - start
    print(f"   GPU: {time_gpu:.3f}s ({n_seeds/time_gpu:.1f} seeds/sec)")
    print(f"   Speedup: {time_cpu/time_gpu:.1f}x")
    
    # Compare
    print(f"\n3. Comparing outputs...")
    matches = (cpu_results == gpu_results_np)
    match_rate = matches.mean()
    
    if match_rate == 1.0:
        print(f"✅ PASS: 100% match ({matches.sum()}/{matches.size} values)")
    else:
        print(f"❌ FAIL: {match_rate*100:.2f}% match ({matches.sum()}/{matches.size} values)")
        
        # Show first mismatch
        mismatch_idx = np.where(~matches)
        if len(mismatch_idx[0]) > 0:
            seed_idx = mismatch_idx[0][0]
            draw_idx = mismatch_idx[1][0]
            print(f"\n   First mismatch:")
            print(f"   Seed #{seed_idx} (value={test_seeds[seed_idx]}), Draw #{draw_idx}")
            print(f"   CPU: {cpu_results[seed_idx, draw_idx]}")
            print(f"   GPU: {gpu_results_np[seed_idx, draw_idx]}")
            
            # Show full sequence for first seed
            print(f"\n   Full sequence for seed #{seed_idx}:")
            print(f"   CPU: {cpu_results[seed_idx, :10]}...")
            print(f"   GPU: {gpu_results_np[seed_idx, :10]}...")
    
    return match_rate == 1.0, match_rate, time_cpu, time_gpu


def benchmark_performance(prng_name, device='cuda', batch_sizes=[100, 1000, 10000], n_draws=500):
    """
    Benchmark GPU performance across different batch sizes.
    
    Args:
        prng_name: PRNG to benchmark
        device: 'cuda' or 'cpu'
        batch_sizes: List of batch sizes to test
        n_draws: Number of draws per seed
    
    Returns:
        Dict with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"PERFORMANCE BENCHMARK: {prng_name} on {device}")
    print(f"{'='*70}")
    
    if not has_pytorch_gpu(prng_name):
        print(f"❌ {prng_name} does not have PyTorch GPU implementation")
        return {}
    
    gpu_func = get_pytorch_gpu_reference(prng_name)
    cpu_func = get_cpu_reference(prng_name)
    prng_info = get_kernel_info(prng_name)
    prng_params = prng_info.get('default_params', {})
    
    results = {}
    
    print(f"\n{'Batch Size':<15} {'CPU Time':<15} {'GPU Time':<15} {'Speedup':<10} {'GPU Seeds/sec':<15}")
    print(f"{'-'*70}")
    
    for batch_size in batch_sizes:
        # Generate test seeds
        test_seeds = np.random.randint(1, 1000000, size=batch_size)
        
        # CPU timing
        start = time.time()
        for seed in test_seeds[:min(batch_size, 100)]:  # Limit CPU to 100 for large batches
            _ = cpu_func(seed=int(seed), n=n_draws, skip=0, **prng_params)
        time_cpu = (time.time() - start) * (batch_size / min(batch_size, 100))
        
        # GPU timing
        seeds_tensor = torch.tensor(test_seeds, dtype=torch.int64, device=device)
        
        # Warmup
        _ = gpu_func(seeds=seeds_tensor[:10], n=n_draws, mod=1000, device=device, skip=0, **prng_params)
        
        # Actual benchmark
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        _ = gpu_func(seeds=seeds_tensor, n=n_draws, mod=1000, device=device, skip=0, **prng_params)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        time_gpu = time.time() - start
        
        speedup = time_cpu / time_gpu if time_gpu > 0 else 0
        seeds_per_sec = batch_size / time_gpu if time_gpu > 0 else 0
        
        print(f"{batch_size:<15} {time_cpu:<15.3f} {time_gpu:<15.3f} {speedup:<10.1f}x {seeds_per_sec:<15.0f}")
        
        results[batch_size] = {
            'cpu_time': time_cpu,
            'gpu_time': time_gpu,
            'speedup': speedup,
            'seeds_per_sec': seeds_per_sec
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test PyTorch GPU PRNG implementations')
    parser.add_argument('--prng', default='java_lcg', help='PRNG to test (default: java_lcg)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to test')
    parser.add_argument('--correctness-only', action='store_true', help='Run only correctness test')
    parser.add_argument('--benchmark-only', action='store_true', help='Run only benchmark')
    parser.add_argument('--all', action='store_true', help='Test all available GPU PRNGs')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"PyTorch GPU PRNG Test Suite")
    print(f"{'='*70}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Device for testing: {args.device}")
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print(f"\n⚠️  WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # List available GPU PRNGs
    available_prngs = list_pytorch_gpu_prngs()
    print(f"\nAvailable GPU PRNGs: {available_prngs}")
    
    if not available_prngs:
        print(f"\n❌ No PRNGs with PyTorch GPU implementation found!")
        print(f"   Make sure prng_registry.py has been updated with PyTorch GPU functions.")
        return 1
    
    # Determine which PRNGs to test
    if args.all:
        test_prngs = available_prngs
    else:
        if args.prng not in available_prngs:
            print(f"\n❌ PRNG '{args.prng}' does not have PyTorch GPU implementation")
            print(f"   Available: {available_prngs}")
            return 1
        test_prngs = [args.prng]
    
    # Run tests
    all_passed = True
    
    for prng_name in test_prngs:
        # Correctness test
        if not args.benchmark_only:
            passed, match_rate, _, _ = test_correctness(
                prng_name=prng_name,
                device=args.device,
                n_seeds=100,
                n_draws=500,
                mod=1000
            )
            all_passed = all_passed and passed
        
        # Benchmark
        if not args.correctness_only:
            _ = benchmark_performance(
                prng_name=prng_name,
                device=args.device,
                batch_sizes=[100, 1000, 10000],
                n_draws=500
            )
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    
    if all_passed:
        print(f"✅ All tests PASSED!")
        return 0
    else:
        print(f"❌ Some tests FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
