#!/bin/bash

echo "1. Running job MANUALLY via SSH..."
ssh 192.168.3.154 'cd /home/michael/distributed_prng_analysis && source /home/michael/rocm_env/bin/activate && CUDA_VISIBLE_DEVICES=2 python -u sieve_filter.py --job-file job_test_manual.json --gpu-id 0' > manual_output.txt 2>&1

echo "Manual output:"
cat manual_output.txt | grep -A 2 "survivors"

echo ""
echo "2. Running job via COORDINATOR..."
python3 << 'PYEOF'
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

for job, worker in forward_jobs:
    if worker.node.hostname == '192.168.3.154' and worker.gpu_id == 2:
        seed_start, seed_end = job.seeds
        if seed_start <= 12345 < seed_end:
            result = coordinator.execute_remote_job(job, worker)
            print(f"Coordinator result:")
            print(f"  Success: {result.success}")
            if result.results:
                print(f"  Survivors: {result.results.get('survivors', [])}")
            else:
                print(f"  Results: None")
            break
PYEOF
