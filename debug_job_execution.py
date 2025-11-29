#!/usr/bin/env python3
"""
Debug why jobs aren't finding anything
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import xorshift32_cpu
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

# Generate simple test data
seed = 98765
skip = 7
n_draws = 100

total_outputs = n_draws * (skip + 1)
all_outputs = xorshift32_cpu(seed, total_outputs, skip=0)

draws = []
idx = 0
for i in range(n_draws):
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Test data: {len(draws)} draws")
print(f"First 10: {draws[:10]}")

test_data = [{'draw': d, 'session': 'midday', 'timestamp': 5000000 + i} 
             for i, d in enumerate(draws)]

with open('test_debug.json', 'w') as f:
    json.dump(test_data, f)

class Args:
    target_file = 'test_debug.json'
    window_size = 512
    seeds = 10000  # Much smaller range for debugging
    seed_start = 90000  # Range that includes 98765
    threshold = 0.01
    skip_min = 0
    skip_max = 15
    offset = 0
    prng_type = 'xorshift32'
    hybrid = False
    phase1_threshold = 0.01
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()
coordinator.current_target_file = args.target_file

jobs = coordinator._create_sieve_jobs(args)
print(f"\nCreated {len(jobs)} jobs")

# Run just ONE job that contains our seed
target_job = None
target_worker = None
for job, worker in jobs:
    seed_start, seed_end = job.seeds
    if seed_start <= 98765 < seed_end:
        target_job = job
        target_worker = worker
        print(f"Job range: {seed_start} - {seed_end}")
        break

if target_job:
    print(f"\nExecuting job on {target_worker.node.hostname} GPU{target_worker.gpu_id}...")
    result = coordinator.execute_gpu_job(target_job, target_worker)
    
    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Runtime: {result.runtime:.2f}s")
    
    if result.error:
        print(f"  Error: {result.error}")
        print(f"\nSTDOUT:")
        print(result.stdout)
        print(f"\nSTDERR:")
        print(result.stderr)
    
    if result.results:
        survivors = result.results.get('survivors', [])
        print(f"  Survivors: {len(survivors)}")
        
        if survivors:
            print(f"  Found seeds: {[s['seed'] for s in survivors[:5]]}")
        
        if any(s['seed'] == 98765 for s in survivors):
            print(f"  ✅ FOUND seed 98765!")
else:
    print(f"❌ No job contains seed 98765!")

