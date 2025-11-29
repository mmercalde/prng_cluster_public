#!/usr/bin/env python3
"""
Debug what's being sent to workers
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

class Args:
    target_file = 'test_xorshift32_hybrid_dist.json'
    window_size = 512
    seeds = 100000
    seed_start = 0
    threshold = 0.01
    skip_min = 0
    skip_max = 10
    offset = 0
    prng_type = 'xorshift32_hybrid'
    hybrid = True
    phase1_threshold = 0.01
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()
coordinator.current_target_file = args.target_file

jobs = coordinator._create_sieve_jobs(args)
first_job, first_worker = jobs[0]

print("First job payload:")
import json
print(json.dumps(first_job.payload, indent=2, default=str))

print(f"\nprng_families: {first_job.payload.get('prng_families')}")
print(f"hybrid: {first_job.payload.get('hybrid')}")
print(f"strategies: {len(first_job.payload.get('strategies', []))} strategies")
if first_job.payload.get('strategies'):
    print(f"First strategy: {first_job.payload['strategies'][0]}")

