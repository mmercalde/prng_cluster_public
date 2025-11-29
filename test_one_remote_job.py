#!/usr/bin/env python3
"""
Run a single remote job and capture what actually happens
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

# Find the worker on 192.168.3.154 GPU 2
target_worker = None
for w in workers:
    if w.node.hostname == '192.168.3.154' and w.gpu_id == 2:
        target_worker = w
        break

if not target_worker:
    print("❌ Worker not found")
    sys.exit(1)

print(f"Testing worker: {target_worker.node.hostname} GPU {target_worker.gpu_id}")

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

# Create just the jobs for this worker's seed range
forward_jobs = coordinator._create_sieve_jobs(args)

# Find the job that should contain seed 12345 on this worker
target_job = None
for job, worker in forward_jobs:
    if worker.node.hostname == '192.168.3.154' and worker.gpu_id == 2:
        seed_start, seed_end = job.seeds
        print(f"\nJob {job.job_id}: seeds [{seed_start}, {seed_end})")
        if seed_start <= 12345 < seed_end:
            target_job = job
            print(f"✅ This job contains seed 12345")
            break

if not target_job:
    print("❌ No job found containing seed 12345 on this worker")
    sys.exit(1)

print(f"\nJob payload:")
print(json.dumps(target_job.payload, indent=2))

print(f"\n🚀 Executing job...")
result = coordinator.execute_remote_job(target_job, target_worker)

print(f"\nResult:")
print(f"  Success: {result.success}")
print(f"  Runtime: {result.runtime:.2f}s")
print(f"  Error: {result.error}")

if result.results:
    survivors = result.results.get('survivors', [])
    print(f"  Survivors: {len(survivors)}")
    
    if survivors:
        print(f"  Top survivors:")
        for s in survivors[:5]:
            marker = "🎯" if s['seed'] == 12345 else "  "
            print(f"    {marker} Seed {s['seed']}: rate={s['match_rate']:.4f}, skip={s['best_skip']}")

