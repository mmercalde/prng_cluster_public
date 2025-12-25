import sys, json
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from cluster_models import JobSpec, GPUWorker, WorkerNode

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Create a reverse sieve job
payload = {
    'analysis_type': 'reverse_sieve',
    'job_id': 'test_120',
    'dataset_path': 'daily3.json',
    'window_size': 768,
    'min_match_threshold': 0.01,
    'skip_min': 19,
    'skip_max': 19,
    'offset': 0,
    'prng_families': ['lcg32'],
    'sessions': ['midday', 'evening'],
    'candidate_seeds': [{'seed': 208989, 'skip': 19}]
}

job = JobSpec(
    job_id='test_120',
    seeds=[{'seed': 208989, 'skip': 19}],
    prng_type='reverse_sieve',
    mapping_type='reverse_sieve',
    samples=1,
    lmax=1,
    grid_size=1,
    mining_mode=False,
    search_type='reverse_sieve',
    target_draw=None,
    payload=payload
)

# Find .120 worker
node_120 = [n for n in coordinator.nodes if n.hostname == '192.168.3.120'][0]
worker_120 = GPUWorker(node=node_120, gpu_id=0, worker_id='test')

print("Executing seed 208989 on .120 via coordinator...")
result = coordinator.execute_gpu_job(job, worker_120)
print(f"Success: {result.success}")
print(f"Results: {result.results}")
