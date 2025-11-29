#!/usr/bin/env python3
"""
Check what job file the coordinator actually creates
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json
import time

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

# Find the target worker
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

# Find the job
for job, worker in forward_jobs:
    if worker.node.hostname == '192.168.3.154' and worker.gpu_id == 2:
        seed_start, seed_end = job.seeds
        if seed_start <= 12345 < seed_end:
            print(f"Job {job.job_id} payload:")
            print(json.dumps(job.payload, indent=2))
            
            # Execute it
            print(f"\nExecuting...")
            result = coordinator.execute_remote_job(job, worker)
            
            # Give it a moment
            time.sleep(1)
            
            # Now check what file was created on remote
            print(f"\nChecking remote job file...")
            import subprocess
            check_result = subprocess.run(
                ['ssh', '192.168.3.154', f'cat /home/michael/distributed_prng_analysis/job_{job.job_id}.json'],
                capture_output=True,
                text=True
            )
            
            if check_result.returncode == 0:
                print(f"Remote job file content:")
                remote_job = json.loads(check_result.stdout)
                print(json.dumps(remote_job, indent=2))
                
                # Compare
                print(f"\nComparing payload vs remote file:")
                for key in job.payload:
                    local_val = job.payload[key]
                    remote_val = remote_job.get(key, "MISSING")
                    match = "✅" if local_val == remote_val else "❌"
                    print(f"  {match} {key}: {local_val} vs {remote_val}")
            else:
                print(f"❌ Could not read remote file: {check_result.stderr}")
            
            print(f"\nCoordinator result:")
            print(f"  Success: {result.success}")
            print(f"  Survivors: {len(result.results.get('survivors', [])) if result.results else 0}")
            break

