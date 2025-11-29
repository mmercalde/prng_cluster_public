#!/usr/bin/env python3
"""
Capture actual stdout/stderr from coordinator's remote execution
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json

# Monkey-patch to capture output
original_execute_remote = MultiGPUCoordinator.execute_remote_job

def debug_execute_remote(self, job, worker):
    result = original_execute_remote(self, job, worker)
    
    print(f"\n{'='*70}")
    print(f"COORDINATOR OUTPUT")
    print(f"{'='*70}")
    print(f"Success: {result.success}")
    print(f"Runtime: {result.runtime:.2f}s")
    print(f"Error: {result.error}")
    
    if result.results:
        print(f"\nParsed results:")
        print(json.dumps(result.results, indent=2))
    else:
        print(f"\nNo results parsed!")
    
    print(f"{'='*70}\n")
    return result

MultiGPUCoordinator.execute_remote_job = debug_execute_remote

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

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

# Find and execute the job
for job, worker in forward_jobs:
    if worker.node.hostname == '192.168.3.154' and worker.gpu_id == 2:
        seed_start, seed_end = job.seeds
        if seed_start <= 12345 < seed_end:
            print(f"Executing job {job.job_id} on {worker.node.hostname} GPU{worker.gpu_id}")
            print(f"Seed range: [{seed_start}, {seed_end})")
            
            result = coordinator.execute_remote_job(job, worker)
            
            survivors = result.results.get('survivors', []) if result.results else []
            print(f"\nFinal: {len(survivors)} survivors")
            if survivors:
                for s in survivors[:5]:
                    print(f"  Seed {s['seed']}: {s['match_rate']:.4f}")
            break

