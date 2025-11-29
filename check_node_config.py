#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from coordinator import MultiGPUCoordinator

coordinator = MultiGPUCoordinator('distributed_config.json')

for node in coordinator.nodes:
    print(f"Node: {node.hostname}")
    print(f"  python_env: {node.python_env}")
    print(f"  script_path: {node.script_path}")
    
    import os
    if node.python_env:
        activate_path = os.path.join(os.path.dirname(node.python_env), 'activate')
        print(f"  Constructed activate_path: {activate_path}")
        print(f"  Exists? {os.path.exists(activate_path)}")
    print()
