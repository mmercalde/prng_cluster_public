import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

original_execute = coordinator.execute_gpu_job
def trace_execute(job, worker):
    if job.search_type == 'reverse_sieve':
        cands = job.payload.get('candidate_seeds', []) if job.payload else []
        seeds = [c.get('seed') for c in cands]
        print(f"🔍 Executing {job.job_id} on {worker.node.hostname} GPU {worker.gpu_id}, seeds={seeds}")
    result = original_execute(job, worker)
    if job.search_type == 'reverse_sieve':
        surv = len(result.results.get('survivors', [])) if result.results else 0
        print(f"  → Result: {surv} survivors")
    return result

coordinator.execute_gpu_job = trace_execute

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
