#!/usr/bin/env python3
"""Quick patch to wrap sieve dict results in object format"""

# Read coordinator.py
with open('coordinator.py', 'r') as f:
    content = f.read()

# Find and replace the sieve result handling
old = '''                    if job_spec.get('search_type') == 'residue_sieve':
                        from sieve_filter import execute_sieve_job
                        result = execute_sieve_job(job_spec, worker.gpu_id)
                        my_results.append(result)
                        jobs_completed += 1
                        work_queue.task_done()
                        continue'''

new = '''                    if job_spec.get('search_type') == 'residue_sieve':
                        from sieve_filter import execute_sieve_job
                        result_dict = execute_sieve_job(job_spec, worker.gpu_id)
                        # Wrap dict in simple object for compatibility
                        class SieveResult:
                            def __init__(self, d):
                                self.success = d.get('success', True)
                                self.runtime = d.get('runtime', 0)
                                self.error = d.get('error', None)
                                self.survivors_found = d.get('survivors_found', 0)
                                self.__dict__.update(d)
                        result = SieveResult(result_dict)
                        my_results.append(result)
                        jobs_completed += 1
                        work_queue.task_done()
                        continue'''

if old in content:
    content = content.replace(old, new)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Sieve result format fixed!")
else:
    print("❌ Pattern not found")
    exit(1)
