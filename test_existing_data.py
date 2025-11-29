#!/usr/bin/env python3
"""
Test with existing test_known.json that we know works
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json

# Load test_known.json to see what we have
with open('test_known.json', 'r') as f:
    data = json.load(f)

print(f"test_known.json has {len(data)} entries")
print(f"First entry: {data[0]}")
print(f"Keys: {data[0].keys()}")

# Extract just the draws
draws = [d['draw'] for d in data]
print(f"\nFirst 10 draws: {draws[:10]}")
print(f"Total draws: {len(draws)}")

# This file was likely generated with a known seed - let's try common ones
test_seeds = [12345, 54321, 98765, 42, 1337]

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

for test_seed in test_seeds:
    print(f"\n{'='*70}")
    print(f"Testing with seed {test_seed}")
    print(f"{'='*70}")
    
    class Args:
        target_file = 'test_known.json'
        window_size = len(draws)
        seeds = 100000
        seed_start = 0
        threshold = 0.01
        skip_min = 0
        skip_max = 20
        offset = 0
        prng_type = 'mt19937'  # Likely mt19937
        hybrid = False
        phase1_threshold = 0.01
        phase2_threshold = 0.50
        gpu_id = 0
    
    args = Args()
    coordinator.current_target_file = args.target_file
    
    jobs = coordinator._create_sieve_jobs(args)
    
    # Find job containing test_seed
    for job, worker in jobs:
        seed_start, seed_end = job.seeds
        if seed_start <= test_seed < seed_end:
            result = coordinator.execute_gpu_job(job, worker)
            
            if result.success and result.results:
                survivors = result.results.get('survivors', [])
                found = [s for s in survivors if s['seed'] == test_seed]
                
                if found:
                    print(f"✅ FOUND seed {test_seed}!")
                    print(f"   Match rate: {found[0].get('match_rate', 0):.1%}")
                    print(f"   Skip: {found[0].get('best_skip')}")
                    sys.exit(0)
                elif survivors:
                    print(f"❌ Seed {test_seed} not found")
                    print(f"   But found {len(survivors)} other seeds")
                    print(f"   Sample: {[s['seed'] for s in survivors[:5]]}")
            break

print("\n❌ None of the test seeds were found in test_known.json")

