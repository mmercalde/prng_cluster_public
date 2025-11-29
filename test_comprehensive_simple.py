#!/usr/bin/env python3
"""
Simpler comprehensive test - run all jobs and collect results
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import mt19937_cpu, xorshift32_cpu
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

# Test 1: Xorshift32 constant skip (simplest)
print("="*70)
print("TEST 1: Xorshift32 - Constant Skip")
print("="*70)

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

print(f"Known seed: {seed}, skip: {skip}")
print(f"First 10 draws: {draws[:10]}")

test_data = [{'draw': d, 'session': 'midday', 'timestamp': 5000000 + i} 
             for i, d in enumerate(draws)]

with open('test_xor_const.json', 'w') as f:
    json.dump(test_data, f)

class Args:
    target_file = 'test_xor_const.json'
    window_size = 512
    seeds = 100000
    seed_start = 0
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
print(f"Created {len(jobs)} jobs, running all...")

all_survivors = []

def execute_job(job_worker):
    job, worker = job_worker
    result = coordinator.execute_gpu_job(job, worker)
    return result

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(execute_job, jw): jw for jw in jobs}
    
    for future in as_completed(futures):
        result = future.result()
        if result.success and result.results:
            survivors = result.results.get('survivors', [])
            all_survivors.extend(survivors)

print(f"\nTotal survivors across all jobs: {len(all_survivors)}")

found = [s for s in all_survivors if s['seed'] == seed]
if found:
    print(f"✅ FOUND seed {seed}!")
    print(f"   Match rate: {found[0].get('match_rate', 0):.1%}")
    print(f"   Skip: {found[0].get('best_skip', 'N/A')}")
else:
    print(f"❌ Seed {seed} NOT found")
    print(f"   Sample of found seeds: {[s['seed'] for s in all_survivors[:10]]}")

