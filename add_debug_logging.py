#!/usr/bin/env python3
"""Add debug logging to track result collection"""

with open('coordinator.py', 'r') as f:
    content = f.read()

old = """            # Return results for this worker
            results_queue.put(my_results)
            if jobs_completed > 0:
                print(f"🏁 {worker.node.gpu_type}@{worker.node.hostname}(gpu{worker.gpu_id}) completed {jobs_completed} jobs")"""

new = """            # Return results for this worker
            print(f"DEBUG: Worker {worker.node.hostname}(gpu{worker.gpu_id}) putting {len(my_results)} results in queue")
            results_queue.put(my_results)
            print(f"DEBUG: Worker {worker.node.hostname}(gpu{worker.gpu_id}) - put complete")
            if jobs_completed > 0:
                print(f"🏁 {worker.node.gpu_type}@{worker.node.hostname}(gpu{worker.gpu_id}) completed {jobs_completed} jobs")"""

if old in content:
    content = content.replace(old, new)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Added debug logging!")
else:
    print("❌ Pattern not found")
    exit(1)
