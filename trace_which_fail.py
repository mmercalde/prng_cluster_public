import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

failed_seeds = []
original_execute = coordinator.execute_gpu_job
def trace_execute(job, worker):
    result = original_execute(job, worker)
    if job.search_type == 'reverse_sieve':
        surv = len(result.results.get('survivors', [])) if result.results else 0
        cands = job.payload.get('candidate_seeds', [])
        if surv == 0 and cands:
            for c in cands:
                failed_seeds.append({
                    'seed': c['seed'],
                    'skip': c['skip'],
                    'node': worker.node.hostname,
                    'gpu': worker.gpu_id
                })
    return result
coordinator.execute_gpu_job = trace_execute

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)

print(f"\n{'='*60}")
print(f"Failed seeds ({len(failed_seeds)}):")
for f in failed_seeds:
    print(f"  seed={f['seed']}, skip={f['skip']}, node={f['node']}, gpu={f['gpu']}")
