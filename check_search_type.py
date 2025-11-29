#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

coordinator = MultiGPUCoordinator('distributed_config.json')
workers = coordinator.create_gpu_workers()

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

forward_jobs = coordinator._create_sieve_jobs(args)

# Check the first few jobs
print("Checking job search_type:")
for job, worker in forward_jobs[:5]:
    print(f"  Job {job.job_id}: search_type='{job.search_type}'")
    if worker.node.hostname == '192.168.3.154' and worker.gpu_id == 2:
        print(f"    ^ This is the job with seed 12345!")
        print(f"    Job details:")
        print(f"      prng_type: {job.prng_type}")
        print(f"      mapping_type: {job.mapping_type}")
        print(f"      analysis_type: {job.analysis_type}")

