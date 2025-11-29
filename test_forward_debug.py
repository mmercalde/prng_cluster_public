#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

class Args:
    target_file = 'test_constant_align.json'
    window_size = 30
    seeds = 1000000
    seed_start = 0
    threshold = 0.01
    skip_min = 0
    skip_max = 10
    offset = 0
    prng_type = 'mt19937'
    hybrid = False
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()

print("Creating forward sieve jobs...")
forward_jobs = coordinator._create_sieve_jobs(args)
print(f"Created {len(forward_jobs)} jobs")

if forward_jobs:
    fwd_job, fwd_worker = forward_jobs[0]
    print(f"\nExecuting first job: {fwd_job.job_id}")
    print(f"  Seeds: {len(fwd_job.seeds)}")
    print(f"  Search type: {fwd_job.search_type}")
    
    fwd_result = coordinator.execute_local_job(fwd_job, fwd_worker)
    
    print(f"\nResult:")
    print(f"  Success: {fwd_result.success}")
    print(f"  Error: {fwd_result.error}")
    print(f"  Runtime: {fwd_result.runtime}")
    print(f"  Results: {fwd_result.results}")
