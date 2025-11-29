#!/usr/bin/env python3
"""Fix result collection race condition"""

with open('coordinator.py', 'r') as f:
    content = f.read()

old = """        # Wait for all worker threads to complete
        for thread in threads:
            thread.join(timeout=30) # Reasonable timeout
        # Collect all results - IMMEDIATE, NO SEQUENTIAL WAITING
        all_results = []
        while not results_queue.empty():"""

new = """        # Wait for all worker threads to complete
        for thread in threads:
            thread.join() # Wait indefinitely - we need ALL results!
        
        # Give workers a moment to put results in queue
        import time
        time.sleep(0.5)
        
        # Collect all results - IMMEDIATE, NO SEQUENTIAL WAITING
        all_results = []
        while not results_queue.empty():"""

if old in content:
    content = content.replace(old, new)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Fixed result collection race condition!")
else:
    print("❌ Pattern not found")
    exit(1)
