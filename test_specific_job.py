#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json

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

print("Creating forward sieve jobs...")
forward_jobs = coordinator._create_sieve_jobs(args)

# Get job 3 (contains seed 12345)
job_3, worker_3 = forward_jobs[3]

print(f"\nJob 3 details:")
print(f"  Seed range: {job_3.seeds}")
print(f"  Payload window_size: {job_3.payload.get('window_size')}")
print(f"  Payload threshold: {job_3.payload.get('min_match_threshold')}")
print(f"  Payload skip_range: {job_3.payload.get('skip_range')}")

print(f"\nExecuting job 3...")
result = coordinator.execute_local_job(job_3, worker_3)

print(f"\nResult:")
print(f"  Success: {result.success}")
print(f"  Runtime: {result.runtime:.2f}s")

if result.success and result.results:
    survivors = result.results.get('survivors', [])
    print(f"  Total survivors: {len(survivors)}")
    
    # Check for seed 12345
    known = [s for s in survivors if s['seed'] == 12345]
    if known:
        print(f"\n✅ FOUND SEED 12345!")
        print(f"   Match rate: {known[0]['match_rate']:.3f}")
        print(f"   Best skip: {known[0]['best_skip']}")
    else:
        print(f"\n❌ Seed 12345 NOT in survivors")
        print(f"   Testing seed 12345 manually...")
        
        # Manual test with the exact parameters
        from prng_registry import mt19937_cpu
        
        # Generate what seed 12345 SHOULD produce with skip=5
        test_outputs = mt19937_cpu(12345, 600, skip=0)
        test_draws = []
        idx = 0
        for i in range(100):
            idx += 5
            test_draws.append(test_outputs[idx] % 1000)
            idx += 1
        
        # Compare with our test data
        with open('test_constant_align.json', 'r') as f:
            actual_draws = [d['draw'] for d in json.load(f)]
        
        matches = sum(1 for i in range(min(50, len(test_draws), len(actual_draws))) 
                      if test_draws[i] == actual_draws[i])
        match_rate = matches / 50
        
        print(f"   Manual verification:")
        print(f"     Generated draws (first 10): {test_draws[:10]}")
        print(f"     Actual draws (first 10): {actual_draws[:10]}")
        print(f"     Matches in first 50: {matches}/50 = {match_rate:.1%}")
        
        if match_rate >= 0.01:
            print(f"   ✅ Should pass threshold (0.01 = 1%)")
        else:
            print(f"   ❌ Below threshold!")
else:
    print(f"  Error: {result.error}")
