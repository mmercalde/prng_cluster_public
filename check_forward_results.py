import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

class Args:
    target_file = 'daily3.json'
    method = 'residue_sieve'
    prng_type = 'lcg32'
    window_size = 768
    seeds = 1_000_000
    seed_start = 0
    offset = 0
    skip_min = 0
    skip_max = 20
    threshold = 0.01
    phase1_threshold = 0.01
    phase2_threshold = 0.50
    session_filter = 'both'
    hybrid = False
    gpu_id = None

jobs = coordinator._create_sieve_jobs(Args())

all_survivors = []
for job, worker in jobs:
    result = coordinator.execute_gpu_job(job, worker)
    if result.results and 'survivors' in result.results:
        all_survivors.extend(result.results['survivors'])

# Check if 461147 and 534165 are in the results
for seed_num in [461147, 534165]:
    found = [s for s in all_survivors if s['seed'] == seed_num]
    if found:
        print(f"Seed {seed_num}: {found[0]}")
    else:
        print(f"Seed {seed_num}: NOT FOUND in forward results")

print(f"\nTotal forward survivors: {len(all_survivors)}")
