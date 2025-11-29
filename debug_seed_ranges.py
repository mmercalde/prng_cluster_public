#!/usr/bin/env python3
"""
Check what seed ranges are being created
"""
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

class Args:
    target_file = 'test.json'
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

print(f"Created {len(jobs)} jobs")
print(f"Total seeds to test: {args.seeds}")
print(f"\nFirst 5 job ranges:")
for i, (job, worker) in enumerate(jobs[:5]):
    seed_start, seed_end = job.seeds
    print(f"  Job {i}: {seed_start:6d} - {seed_end:6d} ({seed_end-seed_start} seeds)")

print(f"\nTest seeds:")
print(f"  12345: in range 0-100000? {0 <= 12345 < 100000}")
print(f"  54321: in range 0-100000? {0 <= 54321 < 100000}")
print(f"  98765: in range 0-100000? {0 <= 98765 < 100000}")

