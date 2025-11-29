#!/usr/bin/env python3
"""
Verify the fix is working - check what command is generated
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

# Find target worker
for w in workers:
    if w.node.hostname == '192.168.3.154' and w.gpu_id == 2:
        target_worker = w
        break

class Args:
    target_file = 'test_26gpu_large.json'
    window_size = 512
    seeds = 20000
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
coordinator.current_target_file = args.target_file

forward_jobs = coordinator._create_sieve_jobs(args)

# Find the job
for job, worker in forward_jobs:
    if worker.node.hostname == '192.168.3.154' and worker.gpu_id == 2:
        seed_start, seed_end = job.seeds
        if seed_start <= 12345 < seed_end:
            print(f"Job payload (checking for search_type):")
            print(json.dumps(job.payload, indent=2))
            
            # Build the command
            cmd = coordinator._build_sh_safe_cmd(
                worker.node,
                f"job_{job.job_id}.json",
                job.payload,
                worker.gpu_id
            )
            
            print(f"\nCommand that will be executed:")
            # Extract the python command line
            lines = cmd.split('\n')
            for line in lines:
                if 'python' in line.lower():
                    print(f"  {line.strip()}")
            
            if 'sieve_filter.py' in cmd:
                print(f"\n✅ FIX WORKED - Using sieve_filter.py")
            elif 'distributed_worker.py' in cmd:
                print(f"\n❌ FIX FAILED - Still using distributed_worker.py")
            
            break

