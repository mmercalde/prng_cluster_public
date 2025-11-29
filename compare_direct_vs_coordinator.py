#!/usr/bin/env python3
"""
Compare what's different between direct call (works) vs coordinator (fails)
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json

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

# Find the job with seed 12345
for job, worker in jobs:
    seed_start, seed_end = job.seeds
    if seed_start <= 12345 < seed_end:
        print(f"Job that should contain seed 12345:")
        print(f"  Job ID: {job.job_id}")
        print(f"  Seed range: {seed_start} - {seed_end}")
        print(f"  Worker: {worker.node.hostname} GPU{worker.gpu_id}")
        print(f"  Payload keys: {job.payload.keys() if job.payload else 'None'}")
        
        if job.payload:
            print(f"\nPayload details:")
            for key in ['prng_families', 'window_size', 'min_match_threshold', 'skip_range', 'dataset_path']:
                if key in job.payload:
                    print(f"  {key}: {job.payload[key]}")
        
        print(f"\nExecuting job to see actual error...")
        result = coordinator.execute_gpu_job(job, worker)
        
        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Runtime: {result.runtime:.2f}s")
        
        if result.error:
            print(f"  Error: {result.error}")
        
        if result.results:
            print(f"  Results keys: {result.results.keys()}")
            survivors = result.results.get('survivors', [])
            print(f"  Survivors: {len(survivors)}")
            
            if 'per_family' in result.results:
                print(f"  Per-family results:")
                for family, data in result.results['per_family'].items():
                    print(f"    {family}: {data.get('found', 0)} found, {data.get('tested', 0)} tested")
        
        break

