#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
import json

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

class Args:
    target_file = 'test_constant_align.json'
    window_size = 50  # Using 50 out of 100 draws
    seeds = 100000
    seed_start = 0
    threshold = 0.01  # Very low threshold
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
print(f"Created {len(forward_jobs)} jobs")

if forward_jobs:
    # Check first job parameters
    fwd_job, fwd_worker = forward_jobs[0]
    print(f"\nFirst job parameters:")
    print(f"  Seeds to test: {fwd_job.seeds}")
    print(f"  Samples: {fwd_job.samples}")
    print(f"  Window: {fwd_job.lmax}")
    print(f"  Grid size: {fwd_job.grid_size}")
    
    # Execute ALL jobs to find seed 12345
    print(f"\nSearching all {len(forward_jobs)} jobs for seed 12345...")
    
    found_in_job = None
    for i, (job, worker) in enumerate(forward_jobs):
        if 12345 in job.seeds:
            found_in_job = i
            print(f"✅ Seed 12345 is in job {i}: seeds {job.seeds}")
            
            # Execute this specific job
            result = coordinator.execute_local_job(job, worker)
            if result.success and result.results:
                survivors = result.results.get('survivors', [])
                known = [s for s in survivors if s['seed'] == 12345]
                if known:
                    print(f"✅ Found seed 12345 with skip={known[0]['best_skip']}, rate={known[0]['match_rate']:.3f}")
                else:
                    print(f"⚠️ Seed 12345 was tested but didn't survive (threshold too high?)")
                    print(f"   Total survivors from this job: {len(survivors)}")
            break
    
    if found_in_job is None:
        print(f"❌ Seed 12345 is NOT in any job's seed range!")
        print(f"   Job ranges: {[j.seeds for j, w in forward_jobs[:3]]}")
