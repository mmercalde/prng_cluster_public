#!/usr/bin/env python3
"""
Memory Benchmark & Concurrency Estimator for Step 3 Workers
============================================================
Version: 1.0.0
Date: 2026-01-24

PURPOSE:
  Measures actual memory usage per worker to determine safe concurrency
  levels on memory-constrained mining rigs (7.7GB RAM).

USAGE:
  # Run on mining rig (rig-6600 or rig-6600b)
  python3 benchmark_worker_memory.py --survivors bidirectional_survivors.json

  # Test specific chunk sizes
  python3 benchmark_worker_memory.py --survivors bidirectional_survivors.json --chunk-sizes 500,1000,2000,5000

  # Full benchmark with concurrent worker simulation
  python3 benchmark_worker_memory.py --survivors bidirectional_survivors.json --test-concurrency

OUTPUT:
  - Peak memory per worker at each chunk size
  - Recommended max_concurrent_workers for this node
  - Suggested chunk_size for distributed_config.json
"""

import os
import sys
import json
import time
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import gc

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not installed. Install with: pip install psutil --break-system-packages")


@dataclass
class MemoryProfile:
    """Memory usage profile for a single test."""
    chunk_size: int
    peak_rss_mb: float
    baseline_mb: float
    delta_mb: float
    load_time_sec: float
    

@dataclass
class ConcurrencyRecommendation:
    """Final recommendation for safe concurrency."""
    available_ram_mb: float
    safe_ram_mb: float  # 80% of available
    per_worker_mb: float
    recommended_workers: int
    recommended_chunk_size: int
    safety_margin: float


def get_system_memory() -> Tuple[float, float]:
    """Get total and available system memory in MB."""
    if not PSUTIL_AVAILABLE:
        # Fallback: read from /proc/meminfo
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            mem_total = mem_available = 0
            for line in lines:
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1]) / 1024  # KB to MB
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1]) / 1024
            return mem_total, mem_available
        except Exception as e:
            print(f"⚠️  Could not read /proc/meminfo: {e}")
            return 7700, 6000  # Conservative defaults for mining rigs
    
    mem = psutil.virtual_memory()
    return mem.total / (1024 * 1024), mem.available / (1024 * 1024)


def get_current_rss_mb() -> float:
    """Get current process RSS in MB."""
    if PSUTIL_AVAILABLE:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    else:
        # Fallback: read from /proc/self/status
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024  # KB to MB
        except:
            pass
        return 0


def load_survivors_chunk(survivors_path: str, chunk_size: int, offset: int = 0) -> List[Dict]:
    """Load a chunk of survivors from JSON file."""
    with open(survivors_path, 'r') as f:
        survivors = json.load(f)
    
    # Simulate chunk loading
    end = min(offset + chunk_size, len(survivors))
    return survivors[offset:end]


def simulate_worker_memory_usage(
    survivors_path: str,
    train_history_path: str,
    chunk_size: int
) -> MemoryProfile:
    """
    Simulate a single worker's memory usage for a given chunk size.
    
    This mimics what full_scoring_worker.py does:
    1. Load survivor chunk
    2. Load train_history
    3. Initialize scorer state
    4. Process (we skip actual GPU work)
    """
    gc.collect()
    time.sleep(0.1)  # Let GC settle
    
    baseline_mb = get_current_rss_mb()
    start_time = time.time()
    
    # Step 1: Load survivors chunk
    survivors = load_survivors_chunk(survivors_path, chunk_size)
    
    # Step 2: Load train_history
    with open(train_history_path, 'r') as f:
        train_history = json.load(f)
    
    # Step 3: Simulate feature extraction state (numpy arrays would be here)
    # In real worker, this creates arrays for 64 features × chunk_size
    import numpy as np
    feature_matrix = np.zeros((len(survivors), 64), dtype=np.float32)
    
    # Step 4: Simulate intermediate state
    # (scorer creates prediction arrays, residue buffers, etc.)
    predictions_buffer = np.zeros((len(survivors), len(train_history)), dtype=np.float32)
    
    load_time = time.time() - start_time
    peak_mb = get_current_rss_mb()
    
    # Cleanup
    del survivors, train_history, feature_matrix, predictions_buffer
    gc.collect()
    
    return MemoryProfile(
        chunk_size=chunk_size,
        peak_rss_mb=peak_mb,
        baseline_mb=baseline_mb,
        delta_mb=peak_mb - baseline_mb,
        load_time_sec=load_time
    )


def worker_process(
    survivors_path: str,
    train_history_path: str,
    chunk_size: int,
    worker_id: int,
    result_queue: multiprocessing.Queue
):
    """Worker process for concurrent memory testing."""
    try:
        profile = simulate_worker_memory_usage(
            survivors_path, train_history_path, chunk_size
        )
        result_queue.put((worker_id, profile.peak_rss_mb, None))
    except Exception as e:
        result_queue.put((worker_id, 0, str(e)))


def test_concurrent_workers(
    survivors_path: str,
    train_history_path: str,
    chunk_size: int,
    num_workers: int,
    timeout: int = 60
) -> Tuple[float, bool, Optional[str]]:
    """
    Test memory usage with N concurrent workers.
    
    Returns: (peak_system_memory_mb, success, error_msg)
    """
    result_queue = multiprocessing.Queue()
    processes = []
    
    # Record baseline
    _, baseline_available = get_system_memory()
    
    # Spawn workers
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=worker_process,
            args=(survivors_path, train_history_path, chunk_size, i, result_queue)
        )
        processes.append(p)
    
    # Start all workers simultaneously
    for p in processes:
        p.start()
    
    # Monitor peak memory usage
    peak_used = 0
    start = time.time()
    
    while any(p.is_alive() for p in processes):
        if time.time() - start > timeout:
            for p in processes:
                p.terminate()
            return peak_used, False, "Timeout"
        
        total, available = get_system_memory()
        used = total - available
        peak_used = max(peak_used, used)
        time.sleep(0.1)
    
    # Collect results
    errors = []
    while not result_queue.empty():
        worker_id, peak_mb, error = result_queue.get()
        if error:
            errors.append(f"Worker {worker_id}: {error}")
    
    # Join all
    for p in processes:
        p.join(timeout=5)
    
    if errors:
        return peak_used, False, "; ".join(errors)
    
    return peak_used, True, None


def run_benchmark(
    survivors_path: str,
    train_history_path: str,
    chunk_sizes: List[int],
    test_concurrency: bool = False,
    max_workers_to_test: int = 12
) -> Dict:
    """
    Run full memory benchmark.
    
    Returns dict with profiles and recommendations.
    """
    print("=" * 70)
    print("MEMORY BENCHMARK FOR STEP 3 WORKERS")
    print("=" * 70)
    
    # System info
    total_ram, available_ram = get_system_memory()
    print(f"\n📊 System Memory:")
    print(f"   Total RAM:     {total_ram:.0f} MB")
    print(f"   Available RAM: {available_ram:.0f} MB")
    print(f"   Used RAM:      {total_ram - available_ram:.0f} MB")
    
    # Count survivors
    with open(survivors_path, 'r') as f:
        survivors = json.load(f)
    total_survivors = len(survivors)
    del survivors
    gc.collect()
    
    print(f"\n📁 Input Files:")
    print(f"   Survivors: {survivors_path} ({total_survivors:,} records)")
    print(f"   Train History: {train_history_path}")
    
    # Single-worker chunk size tests
    print(f"\n{'=' * 70}")
    print("PHASE 1: Single Worker Memory Profiles")
    print("=" * 70)
    
    profiles = []
    for chunk_size in chunk_sizes:
        if chunk_size > total_survivors:
            print(f"\n⚠️  Skipping chunk_size={chunk_size} (exceeds survivor count)")
            continue
            
        print(f"\n🔬 Testing chunk_size = {chunk_size:,}...")
        profile = simulate_worker_memory_usage(
            survivors_path, train_history_path, chunk_size
        )
        profiles.append(profile)
        
        print(f"   Baseline:   {profile.baseline_mb:.1f} MB")
        print(f"   Peak RSS:   {profile.peak_rss_mb:.1f} MB")
        print(f"   Delta:      {profile.delta_mb:.1f} MB")
        print(f"   Load time:  {profile.load_time_sec:.2f}s")
    
    # Calculate memory per survivor
    if len(profiles) >= 2:
        # Linear regression to estimate bytes per survivor
        sizes = [p.chunk_size for p in profiles]
        deltas = [p.delta_mb for p in profiles]
        
        # Simple linear fit: delta = m * chunk_size + b
        n = len(sizes)
        sum_x = sum(sizes)
        sum_y = sum(deltas)
        sum_xy = sum(x * y for x, y in zip(sizes, deltas))
        sum_xx = sum(x * x for x in sizes)
        
        m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        b = (sum_y - m * sum_x) / n
        
        mb_per_1k_survivors = m * 1000
        base_overhead_mb = b
        
        print(f"\n📈 Memory Model (linear fit):")
        print(f"   Per 1K survivors: ~{mb_per_1k_survivors:.1f} MB")
        print(f"   Base overhead:    ~{base_overhead_mb:.1f} MB")
    else:
        # Fallback: use single measurement
        if profiles:
            mb_per_1k_survivors = (profiles[0].delta_mb / profiles[0].chunk_size) * 1000
            base_overhead_mb = profiles[0].baseline_mb
        else:
            mb_per_1k_survivors = 150  # Conservative default
            base_overhead_mb = 200
    
    # Concurrent worker tests (optional)
    concurrency_results = {}
    if test_concurrency:
        print(f"\n{'=' * 70}")
        print("PHASE 2: Concurrent Worker Tests")
        print("=" * 70)
        
        # Use middle chunk size for concurrency tests
        test_chunk = chunk_sizes[len(chunk_sizes) // 2]
        print(f"\nUsing chunk_size = {test_chunk:,} for concurrency tests")
        
        for num_workers in range(1, max_workers_to_test + 1):
            print(f"\n🔬 Testing {num_workers} concurrent workers...")
            
            peak_mb, success, error = test_concurrent_workers(
                survivors_path, train_history_path, test_chunk, num_workers
            )
            
            concurrency_results[num_workers] = {
                'peak_system_mb': peak_mb,
                'success': success,
                'error': error
            }
            
            if success:
                print(f"   ✅ Peak system memory: {peak_mb:.0f} MB")
            else:
                print(f"   ❌ Failed: {error}")
                print(f"   📊 Peak before failure: {peak_mb:.0f} MB")
                break  # Stop at first failure
    
    # Generate recommendations
    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    safe_ram = available_ram * 0.80  # 80% safety margin
    
    # Find optimal chunk size for different concurrency levels
    recommendations = []
    for target_workers in [5, 7, 10, 12]:
        ram_per_worker = safe_ram / target_workers
        
        # Solve: base_overhead + (chunk_size/1000) * mb_per_1k = ram_per_worker
        max_chunk = int(((ram_per_worker - base_overhead_mb) / mb_per_1k_survivors) * 1000)
        max_chunk = max(100, min(max_chunk, total_survivors))
        
        jobs_needed = (total_survivors + max_chunk - 1) // max_chunk
        
        recommendations.append({
            'workers': target_workers,
            'chunk_size': max_chunk,
            'ram_per_worker_mb': ram_per_worker,
            'jobs_needed': jobs_needed
        })
    
    print(f"\n📊 For {available_ram:.0f} MB available RAM (using 80% = {safe_ram:.0f} MB):")
    print(f"\n{'Workers':<10} {'Chunk Size':<12} {'RAM/Worker':<12} {'Total Jobs':<12}")
    print("-" * 46)
    for rec in recommendations:
        print(f"{rec['workers']:<10} {rec['chunk_size']:<12,} {rec['ram_per_worker_mb']:<12.0f} {rec['jobs_needed']:<12}")
    
    # Best recommendation
    # Aim for 7 workers (matches current OOM point) with safety
    target_rec = next((r for r in recommendations if r['workers'] == 7), recommendations[0])
    
    print(f"\n✅ RECOMMENDED CONFIGURATION:")
    print(f"   max_concurrent_script_jobs: {target_rec['workers']}")
    print(f"   chunk_size: {target_rec['chunk_size']:,}")
    print(f"   Expected jobs: {target_rec['jobs_needed']}")
    
    # Generate config snippet
    config_snippet = {
        "memory_benchmark_results": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_ram_mb": total_ram,
            "available_ram_mb": available_ram,
            "mb_per_1k_survivors": round(mb_per_1k_survivors, 1),
            "base_overhead_mb": round(base_overhead_mb, 1),
            "recommended_workers": target_rec['workers'],
            "recommended_chunk_size": target_rec['chunk_size']
        }
    }
    
    # Save results
    output_file = "memory_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'system': {
                'total_ram_mb': total_ram,
                'available_ram_mb': available_ram,
                'hostname': os.uname().nodename
            },
            'profiles': [
                {
                    'chunk_size': p.chunk_size,
                    'peak_rss_mb': p.peak_rss_mb,
                    'delta_mb': p.delta_mb,
                    'load_time_sec': p.load_time_sec
                }
                for p in profiles
            ],
            'memory_model': {
                'mb_per_1k_survivors': round(mb_per_1k_survivors, 2),
                'base_overhead_mb': round(base_overhead_mb, 2)
            },
            'concurrency_results': concurrency_results,
            'recommendations': recommendations,
            'config_snippet': config_snippet
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return config_snippet


def main():
    parser = argparse.ArgumentParser(
        description="Memory benchmark for Step 3 distributed workers"
    )
    parser.add_argument(
        '--survivors', '-s',
        default='bidirectional_survivors.json',
        help='Path to survivors JSON file'
    )
    parser.add_argument(
        '--train-history', '-t',
        default='train_history.json',
        help='Path to train_history JSON file'
    )
    parser.add_argument(
        '--chunk-sizes', '-c',
        default='500,1000,2000,5000,10000',
        help='Comma-separated chunk sizes to test'
    )
    parser.add_argument(
        '--test-concurrency',
        action='store_true',
        help='Also test concurrent workers (slower but more accurate)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=12,
        help='Maximum workers to test in concurrency mode'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.survivors).exists():
        print(f"❌ Survivors file not found: {args.survivors}")
        sys.exit(1)
    
    if not Path(args.train_history).exists():
        print(f"❌ Train history file not found: {args.train_history}")
        sys.exit(1)
    
    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(',')]
    
    run_benchmark(
        survivors_path=args.survivors,
        train_history_path=args.train_history,
        chunk_sizes=chunk_sizes,
        test_concurrency=args.test_concurrency,
        max_workers_to_test=args.max_workers
    )


if __name__ == '__main__':
    main()
