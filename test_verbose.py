import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

# Patch to see all job results
original_execute = coordinator.execute_gpu_job
def debug_execute(job, worker):
    result = original_execute(job, worker)
    if job.search_type == 'reverse_sieve':
        print(f"Reverse job {job.job_id}: success={result.success}, survivors={len(result.results.get('survivors', [])) if result.results else 0}")
    return result
coordinator.execute_gpu_job = debug_execute

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

print(f"\n{'='*60}")
print(f"Final: Forward={result.forward_survivors}, Verified={result.verified_survivors}")
