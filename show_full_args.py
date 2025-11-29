import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

original_execute = coordinator.execute_gpu_job
def debug_execute(job, worker):
    if job.search_type == 'reverse_sieve' and job.job_id == 'reverse_003':
        print(f"\n🔍 Full job payload:")
        for k, v in job.payload.items():
            print(f"  {k}: {v}")
    return original_execute(job, worker)
coordinator.execute_gpu_job = debug_execute

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
