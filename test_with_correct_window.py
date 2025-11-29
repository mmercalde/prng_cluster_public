#!/usr/bin/env python3
"""
Test with correct window_size matching number of draws
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from prng_registry import xorshift32_cpu
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

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

with open('test_window.json', 'w') as f:
    json.dump(test_data, f)

class Args:
    target_file = 'test_window.json'
    window_size = 100  # MATCH THE NUMBER OF DRAWS!
    seeds = 10000
    seed_start = 90000
    threshold = 0.01
    skip_min = 0
    skip_max = 10  # Include skip=7
    offset = 0
    prng_type = 'xorshift32'
    hybrid = False
    phase1_threshold = 0.01
    phase2_threshold = 0.50
    gpu_id = 0

args = Args()
coordinator.current_target_file = args.target_file

jobs = coordinator._create_sieve_jobs(args)

# Find and run the job with our seed
for job, worker in jobs:
    seed_start, seed_end = job.seeds
    if seed_start <= 98765 < seed_end:
        print(f"\nTesting range {seed_start} - {seed_end}")
        print(f"Window size: {args.window_size}")
        print(f"Skip range: {args.skip_min} - {args.skip_max}")
        
        result = coordinator.execute_gpu_job(job, worker)
        
        print(f"\nResult: Success={result.success}, Runtime={result.runtime:.2f}s")
        
        if result.results:
            survivors = result.results.get('survivors', [])
            print(f"Survivors: {len(survivors)}")
            
            found = [s for s in survivors if s['seed'] == 98765]
            if found:
                print(f"✅ FOUND seed 98765!")
                print(f"   Match rate: {found[0].get('match_rate', 0):.1%}")
                print(f"   Best skip: {found[0].get('best_skip')}")
            else:
                print(f"❌ Seed 98765 not found")
                if survivors:
                    print(f"   Sample found: {[s['seed'] for s in survivors[:5]]}")
        break

