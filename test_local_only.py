#!/usr/bin/env python3
"""
TEST: Only your 2 local RTX 3080 Ti GPUs
Should finish in 5-10 seconds
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

print("="*70)
print("LOCAL ONLY TEST: 2 GPUs, 100K seeds, Window 768")
print("="*70)

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Monkey-patch to use ONLY local GPUs
original_create = coordinator._create_sieve_jobs
def local_only_jobs(args):
    all_jobs = original_create(args)
    local_jobs = [(job, worker) for job, worker in all_jobs 
                  if coordinator.is_localhost(worker.node.hostname)]
    print(f"Filtered to {len(local_jobs)} LOCAL jobs (was {len(all_jobs)})")
    return local_jobs

coordinator._create_sieve_jobs = local_only_jobs

optimizer = WindowOptimizer(coordinator, test_seeds=100_000)
print("\nRunning...\n")

import time
start = time.time()
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
elapsed = time.time() - start

print("\n" + "="*70)
print(f"RESULTS (completed in {elapsed:.1f}s):")
print("="*70)
print(f"  Forward survivors: {result.forward_survivors}")
print(f"  Verified survivors: {result.verified_survivors}")
if result.forward_survivors > 0:
    rate = result.verified_survivors/result.forward_survivors*100
    print(f"  Verification rate: {rate:.1f}%")
print("="*70)

if rate == 100.0:
    print("\n✅ LOCAL TEST PASSED!")
    print("   Forward and reverse sieves working on local GPUs")
    print("   Issue is likely with remote node execution")
else:
    print("\n❌ Problem with forward/reverse sieve logic")
