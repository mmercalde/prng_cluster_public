import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator
from modules.window_optimizer import WindowOptimizer

coordinator = MultiGPUCoordinator('distributed_config.json')
coordinator.current_target_file = 'daily3.json'

original_build = coordinator._build_sh_safe_cmd
def debug_build(node, payload_filename, payload_json, gpu_id=0, timeout_s=None):
    if payload_json.get('search_type') == 'reverse_sieve':
        cands = payload_json.get('candidate_seeds', [])
        if cands and any(c.get('seed') in [208989, 235667, 265005, 385530, 999439] for c in cands):
            print(f"\n🔍 Sending to {node.hostname}:")
            print(f"  skip_min: {payload_json.get('skip_min')}")
            print(f"  skip_max: {payload_json.get('skip_max')}")
            print(f"  Candidates: {cands}")
    return original_build(node, payload_filename, payload_json, gpu_id, timeout_s)

coordinator._build_sh_safe_cmd = debug_build

optimizer = WindowOptimizer(coordinator, test_seeds=1_000_000)
result = optimizer.evaluate_window('lcg32', 768, use_all_gpus=True)
