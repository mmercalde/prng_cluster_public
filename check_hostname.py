import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

coordinator = MultiGPUCoordinator('distributed_config.json')
print("GPU Workers:")
for w in coordinator.gpu_workers:
    print(f"  Hostname: '{w.node.hostname}', GPU: {w.gpu_id}, Local: {coordinator.is_localhost(w.node.hostname)}")
