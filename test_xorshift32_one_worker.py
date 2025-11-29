#!/usr/bin/env python3
"""
Test xorshift32_hybrid on ONE worker to see actual errors
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import xorshift32_cpu
import json

print("="*70)
print("XORSHIFT32_HYBRID - SINGLE WORKER TEST")
print("="*70)

# Generate test data
known_seed = 54321
base_pattern = [5,5,3,7,5,5,8,4,5,5]
skip_pattern = base_pattern * 67
k = len(skip_pattern)

total_needed = sum(skip_pattern) + k
all_outputs = xorshift32_cpu(known_seed, total_needed, skip=0)

draws = []
idx = 0
for skip in skip_pattern:
    idx += skip
    draws.append(all_outputs[idx] % 1000)
    idx += 1

print(f"Testing with seed {known_seed} in range")

test_data = [{'draw': d, 'session': 'midday', 'timestamp': 5000000 + i} 
             for i, d in enumerate(draws)]

with open('test_xorshift32_single.json', 'w') as f:
    json.dump(test_data, f)

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

class Args:
    target_file = 'test_xorshift32_single.json'
    window_size = 512
    seeds = 100000
    seed_start = 50000  # Make sure 54321 is in range
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

# Find job containing seed 54321
target_job = None
target_worker = None
for job, worker in jobs:
    seed_start, seed_end = job.seeds
    if seed_start <= 54321 < seed_end:
        target_job = job
        target_worker = worker
        print(f"\n✅ Found job containing seed 54321: {job.job_id}")
        print(f"   Range: {seed_start} to {seed_end}")
        print(f"   Worker: {worker.node.hostname} GPU{worker.gpu_id}")
        break

if not target_job:
    print(f"\n❌ No job contains seed 54321!")
    sys.exit(1)

# Execute just this one job
print(f"\nExecuting job...")
result = coordinator.execute_gpu_job(target_job, target_worker)

print(f"\nResult:")
print(f"  Success: {result.success}")
print(f"  Runtime: {result.runtime:.2f}s")

if result.error:
    print(f"  ❌ Error: {result.error}")
    
if result.results:
    survivors = result.results.get('survivors', [])
    print(f"  Survivors: {len(survivors)}")
    
    for s in survivors:
        if s['seed'] == 54321:
            print(f"\n  🎯 FOUND SEED 54321!")
            print(f"     Match rate: {s.get('match_rate', 0):.1%}")
            print(f"     Pattern: {s.get('skip_pattern', [])[:10]}")
            sys.exit(0)
    
    if survivors:
        print(f"  Found other seeds: {[s['seed'] for s in survivors[:5]]}")

print(f"\n❌ Seed 54321 not found")

