#!/usr/bin/env python3
"""
Debug what command execute_remote_job actually runs
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import os

# Monkey-patch to see the actual SSH command
original_execute_remote = MultiGPUCoordinator.execute_remote_job

def debug_execute_remote(self, job, worker):
    print(f"\n{'='*70}")
    print(f"DEBUGGING execute_remote_job")
    print(f"{'='*70}")
    print(f"Worker: {worker.node.hostname} GPU {worker.gpu_id}")
    print(f"Job: {job.job_id}")
    
    # Get the activate path
    activate_path = os.path.join(os.path.dirname(worker.node.python_env), 'activate')
    print(f"Activate path: {activate_path}")
    
    # Show what command will be constructed
    job_file = f"job_{job.job_id}.json"
    cmd_str = (
        f"source {activate_path} && "
        f"CUDA_VISIBLE_DEVICES={worker.gpu_id} "
        f"python -u sieve_filter.py --job-file {job_file} --gpu-id 0"
    ).strip()
    
    print(f"\nCommand that will be run:")
    print(f"  {cmd_str}")
    print(f"\n{'='*70}\n")
    
    # Call original
    return original_execute_remote(self, job, worker)

MultiGPUCoordinator.execute_remote_job = debug_execute_remote

# Now run the test
coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

# Find GPU 2 on 192.168.3.154
target_worker = None
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

# Find the job with seed 12345
for job, worker in forward_jobs:
    if worker.node.hostname == '192.168.3.154' and worker.gpu_id == 2:
        seed_start, seed_end = job.seeds
        if seed_start <= 12345 < seed_end:
            print(f"Executing job {job.job_id} [{seed_start}, {seed_end})")
            result = coordinator.execute_remote_job(job, worker)
            
            print(f"\nResult:")
            print(f"  Success: {result.success}")
            print(f"  Survivors: {len(result.results.get('survivors', [])) if result.results else 0}")
            break

