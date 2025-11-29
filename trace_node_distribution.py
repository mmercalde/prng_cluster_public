import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

node_jobs = {'localhost': [], '192.168.3.120': [], '192.168.3.154': []}

original_execute = coordinator.execute_gpu_job
def trace_execute(job, worker):
    if job.search_type == 'reverse_sieve':
        cands = job.payload.get('candidate_seeds', []) if job.payload else []
        seeds = [c.get('seed') for c in cands]
        node_jobs[worker.node.hostname].extend(seeds)
    result = original_execute(job, worker)
    return result

coordinator.execute_gpu_job = trace_execute

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

print("\nSeeds sent to each node:")
for node, seeds in node_jobs.items():
    print(f"  {node}: {len(seeds)} seeds")
    
# Check if failing seeds all went to .120
failing = [208989, 235667, 265005, 385530, 999439]
for node, seeds in node_jobs.items():
    fails_on_node = [s for s in failing if s in seeds]
    if fails_on_node:
        print(f"  {node} got failing seeds: {fails_on_node}")
