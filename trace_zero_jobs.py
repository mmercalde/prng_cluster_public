import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

original_execute = coordinator.execute_gpu_job
zero_jobs = []
def trace_execute(job, worker):
    result = original_execute(job, worker)
    if job.search_type == 'reverse_sieve':
        surv = len(result.results.get('survivors', [])) if result.results else 0
        if surv == 0:
            zero_jobs.append({
                'job': job.job_id,
                'node': worker.node.hostname,
                'gpu': worker.gpu_id,
                'payload': job.payload.get('candidate_seeds', [])
            })
    return result
coordinator.execute_gpu_job = trace_execute

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

print(f"\nJobs that returned 0 survivors:")
for z in zero_jobs:
    print(f"  Job {z['job']} on {z['node']} GPU {z['gpu']}: {len(z['payload'])} candidates")
    for c in z['payload']:
        print(f"    seed={c['seed']}, skip={c['skip']}")
