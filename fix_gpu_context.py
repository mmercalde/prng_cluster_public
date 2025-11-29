#!/usr/bin/env python3
"""Fix GPU device access in parallel worker threads"""

with open('coordinator.py', 'r') as f:
    content = f.read()

old = """                    # Handle sieve jobs separately
                    if job_spec.get('search_type') == 'residue_sieve':
                        from sieve_filter import execute_sieve_job
                        try:
                            result_dict = execute_sieve_job(job_spec, worker.gpu_id)"""

new = """                    # Handle sieve jobs separately
                    if job_spec.get('search_type') == 'residue_sieve':
                        from sieve_filter import execute_sieve_job
                        import os
                        # Set CUDA device for this thread
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(worker.gpu_id)
                        try:
                            # Use gpu_id=0 since CUDA_VISIBLE_DEVICES limits to one device
                            result_dict = execute_sieve_job(job_spec, 0)"""

if old in content:
    content = content.replace(old, new)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Fixed GPU device context!")
else:
    print("❌ Pattern not found")
