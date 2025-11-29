#!/usr/bin/env python3
"""Fix collection timing - collect AFTER all threads finish"""

with open('coordinator.py', 'r') as f:
    content = f.read()

old = """        # Wait for all worker threads to complete
        for thread in threads:
            thread.join() # Wait indefinitely - we need ALL results!
        
        # Give workers a moment to put results in queue
        time.sleep(0.5)
        
        # Collect all results - IMMEDIATE, NO SEQUENTIAL WAITING
        all_results = []
        while not results_queue.empty():
            try:
                worker_results = results_queue.get_nowait()
                all_results.extend(worker_results)
            except queue.Empty:
                break"""

new = """        # Wait for all worker threads to complete
        for thread in threads:
            thread.join() # Wait indefinitely - we need ALL results!
        
        # Give workers a moment to put results in queue
        time.sleep(2.0)  # Increased from 0.5 to 2.0 seconds
        
        # Collect all results - IMMEDIATE, NO SEQUENTIAL WAITING
        all_results = []
        print(f"DEBUG: Starting collection, queue size: {results_queue.qsize()}")
        collected_count = 0
        while not results_queue.empty():
            try:
                worker_results = results_queue.get_nowait()
                all_results.extend(worker_results)
                collected_count += 1
                print(f"DEBUG: Collected batch {collected_count}, total results now: {len(all_results)}")
            except queue.Empty:
                break
        print(f"DEBUG: Collection complete, total results: {len(all_results)}")"""

if old in content:
    content = content.replace(old, new)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Added collection debugging!")
else:
    print("❌ Pattern not found")
    exit(1)
