import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

original_build = coordinator._build_sh_safe_cmd
call_count = 0
def debug_build(node, payload_filename, payload_json, gpu_id=0, timeout_s=None):
    global call_count
    if payload_json.get('search_type') == 'reverse_sieve':
        call_count += 1
        cands = payload_json.get('candidate_seeds', [])
        seeds_list = [c.get('seed') for c in cands]
        print(f"\n🔍 Call {call_count} to {node.hostname} GPU {gpu_id}:")
        print(f"  Seeds: {seeds_list}")
        print(f"  skip_min: {payload_json.get('skip_min')}")
        print(f"  skip_max: {payload_json.get('skip_max')}")
    return original_build(node, payload_filename, payload_json, gpu_id, timeout_s)

coordinator._build_sh_safe_cmd = debug_build

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
print(f"\nTotal reverse calls: {call_count}")
