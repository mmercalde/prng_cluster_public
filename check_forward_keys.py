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
    seeds = 100000
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
job, worker = jobs[0]
result = coordinator.execute_gpu_job(job, worker)

if result.results and 'survivors' in result.results and result.results['survivors']:
    print("Forward survivor keys:", result.results['survivors'][0].keys())
    print("First survivor:", result.results['survivors'][0])
