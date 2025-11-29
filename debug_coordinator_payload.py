import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

original_execute = coordinator.execute_gpu_job
def debug_execute(job, worker):
    if job.search_type == 'reverse_sieve' and worker.node.hostname == '192.168.3.120':
        if job.job_id == 'reverse_003':  # This is the one with seed 235667
            print(f"\n🔍 Job payload for reverse_003 (seed 235667):")
            print(f"  Payload keys: {job.payload.keys()}")
            print(f"  Args: {job.payload.get('args', {})}")
            print(f"  Candidates: {job.payload.get('candidate_seeds', [])}")
    return original_execute(job, worker)
coordinator.execute_gpu_job = debug_execute

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
