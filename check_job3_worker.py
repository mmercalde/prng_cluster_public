#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

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
coordinator.current_target_file = args.target_file

forward_jobs = coordinator._create_sieve_jobs(args)
job_3, worker_3 = forward_jobs[3]

print(f"Job 3 assigned to:")
print(f"  Node: {worker_3.node.hostname}")
print(f"  GPU ID: {worker_3.gpu_id}")
print(f"  Worker ID: {worker_3.worker_id}")

# Check if this is a remote or local job
is_remote = worker_3.node.hostname not in ['localhost', '127.0.0.1']
print(f"\n  Is remote? {is_remote}")
print(f"  Execute method: {'execute_remote_job' if is_remote else 'execute_local_job'}")

if is_remote:
    print("\n❌ BUG: execute_local_job is being called for a REMOTE worker!")
    print("   This will try to run the remote worker's command locally")
    print("   and fail because the remote activate path doesn't exist locally")
