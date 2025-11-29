#!/usr/bin/env python3
"""
Check the actual error in the results
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

class Args:
    target_file = 'test_xorshift32_const_skip5.json'
    window_size = 100
    seeds = 20000
    seed_start = 10000
    threshold = 0.01
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

# Find the job
for job, worker in jobs:
    seed_start, seed_end = job.seeds
    if seed_start <= 12345 < seed_end:
        result = coordinator.execute_gpu_job(job, worker)
        
        print("Full results:")
        print(f"  success: {result.results.get('success')}")
        print(f"  error: {result.results.get('error')}")
        
        if result.results.get('traceback'):
            print(f"\nTraceback:")
            print(result.results['traceback'])
        
        break

