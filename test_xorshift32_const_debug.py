#!/usr/bin/env python3
"""
Debug why xorshift32 constant skip fails
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import xorshift32_cpu
import json

# Use the SAME data we created
seed = 12345
skip = 5
n_draws = 100

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

# Try with different thresholds
thresholds = [0.01, 0.05, 0.10, 0.20]

for threshold in thresholds:
    print(f"\n{'='*70}")
    print(f"Testing with threshold={threshold}")
    print(f"{'='*70}")
    
    class Args:
        target_file = 'test_xorshift32_const_skip5.json'
        window_size = 100
        seeds = 20000  # Smaller range for faster testing
        seed_start = 10000  # Includes 12345
        threshold = threshold
        skip_min = 0
        skip_max = 10
        offset = 0
        prng_type = 'xorshift32'
        hybrid = False
        phase1_threshold = 0.01
        phase2_threshold = 0.50
        gpu_id = 0
    
    args = Args()
    coordinator.current_target_file = args.target_file
    
    jobs = coordinator._create_sieve_jobs(args)
    
    # Find and run job with seed 12345
    for job, worker in jobs:
        seed_start, seed_end = job.seeds
        if seed_start <= 12345 < seed_end:
            result = coordinator.execute_gpu_job(job, worker)
            
            if result.success and result.results:
                survivors = result.results.get('survivors', [])
                match = [s for s in survivors if s['seed'] == 12345]
                
                if match:
                    print(f"✅ FOUND with threshold {threshold}")
                    print(f"   Match rate: {match[0].get('match_rate', 0):.1%}")
                    print(f"   Skip: {match[0].get('best_skip')}")
                    sys.exit(0)
                else:
                    print(f"❌ Not found, {len(survivors)} other survivors")
            break

print("\n❌ Seed 12345 not found with any threshold")

