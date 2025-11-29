#!/usr/bin/env python3
"""Add debug to show why jobs are failing"""

with open('coordinator.py', 'r') as f:
    content = f.read()

old = """        # Collect all results - IMMEDIATE, NO SEQUENTIAL WAITING
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

new = """        # Collect all results - IMMEDIATE, NO SEQUENTIAL WAITING
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
        print(f"DEBUG: Collection complete, total results: {len(all_results)}")
        
        # Debug: Show failed job info
        failed_jobs = [r for r in all_results if not r.success]
        if failed_jobs:
            print(f"\\nDEBUG: {len(failed_jobs)} FAILED jobs found:")
            for i, job in enumerate(failed_jobs[:3]):  # Show first 3
                print(f"  Failed job {i+1}:")
                print(f"    Error: {getattr(job, 'error', 'No error message')}")
                print(f"    Results: {type(getattr(job, 'results', None))}")"""

if old in content:
    content = content.replace(old, new)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Added failed job debugging!")
else:
    print("❌ Pattern not found")
