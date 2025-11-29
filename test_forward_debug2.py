#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

class Args:
    target_file = 'test_constant_align.json'
    window_size = 50
    seeds = 100000
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

# CRITICAL: Set the coordinator's current target file
print(f"Setting coordinator target file: {args.target_file}")
coordinator.current_target_file = args.target_file

print("Creating forward sieve jobs...")
forward_jobs = coordinator._create_sieve_jobs(args)
print(f"Created {len(forward_jobs)} jobs")

if forward_jobs:
    fwd_job, fwd_worker = forward_jobs[0]
    print(f"\nJob payload target_file: {fwd_job.payload.get('target_file', 'NOT SET')}")
    print(f"Coordinator current_target_file: {coordinator.current_target_file}")
    
    print(f"\nExecuting first job...")
    fwd_result = coordinator.execute_local_job(fwd_job, fwd_worker)
    
    print(f"\nResult:")
    print(f"  Success: {fwd_result.success}")
    if not fwd_result.success:
        print(f"  Error: {fwd_result.error}")
