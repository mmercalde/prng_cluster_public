import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

coordinator = MultiGPUCoordinator('distributed_config.json')

# Simulate the 6 candidates with skip=19
candidates = [
    {'seed': 39270, 'skip': 19, 'match_rate': 0.010416666977107525},
    {'seed': 135971, 'skip': 19, 'match_rate': 0.010416666977107525},
    {'seed': 208989, 'skip': 19, 'match_rate': 0.010416666977107525},
    {'seed': 385530, 'skip': 19, 'match_rate': 0.010416666977107525},
    {'seed': 461147, 'skip': 19, 'match_rate': 0.010416666977107525},
    {'seed': 534165, 'skip': 19, 'match_rate': 0.010416666977107525}
]

class Args:
    target_file = 'daily3.json'
    method = 'reverse_sieve'
    prng_type = 'lcg32'
    window_size = 768
    threshold = 0.01
    session_filter = 'both'
    hybrid = False
    gpu_id = None
    skip_min = 19
    skip_max = 19
    offset = 0

jobs = coordinator._create_reverse_sieve_jobs(Args(), candidates)
print(f"Created {len(jobs)} jobs for {len(candidates)} candidates\n")

for i, (job, worker) in enumerate(jobs):
    cands = job.payload.get('candidate_seeds', [])
    print(f"Job {i}: {len(cands)} candidates on {worker.node.hostname} GPU {worker.gpu_id}")
    for c in cands:
        print(f"  {c}")
