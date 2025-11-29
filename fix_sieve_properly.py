#!/usr/bin/env python3
"""Fix sieve execution to work with both local and remote"""

with open('coordinator.py', 'r') as f:
    content = f.read()

old = """                    # Handle sieve jobs separately
                    if job_spec.get('search_type') == 'residue_sieve':
                        try:
                            # Check if this is a local or remote worker
                            if self.is_localhost(worker.node.hostname):
                                # Local execution - call directly
                                from sieve_filter import execute_sieve_job
                                result_dict = execute_sieve_job(job_spec, worker.gpu_id)
                            else:
                                # Remote execution - use SSH
                                result = self.execute_remote_sieve_job(job_spec, worker)
                                result_dict = {
                                    'success': result.success,
                                    'runtime': result.runtime,
                                    'error': result.error,
                                    'results': result.results
                                }"""

new = """                    # Handle sieve jobs separately
                    if job_spec.get('search_type') == 'residue_sieve':
                        try:
                            # Check if this is a local or remote worker
                            if self.is_localhost(worker.node.hostname):
                                # Local execution - call directly
                                from sieve_filter import execute_sieve_job
                                result_dict = execute_sieve_job(job_spec, worker.gpu_id)
                            else:
                                # Remote execution via SSH
                                # Create JobSpec object with all required fields
                                from dataclasses import dataclass
                                @dataclass
                                class TempJobSpec:
                                    job_id: str
                                    seeds: list
                                    prng_type: str = 'mt19937'
                                    search_type: str = 'residue_sieve'
                                
                                job_obj = TempJobSpec(
                                    job_id=job_spec['job_id'],
                                    seeds=[job_spec['seeds'][0], job_spec['seeds'][-1] + 1],  # [start, end]
                                    prng_type=job_spec.get('prng_type', 'mt19937'),
                                    search_type='residue_sieve'
                                )
                                result = self.execute_remote_job(job_obj, worker)
                                result_dict = {
                                    'success': result.success,
                                    'runtime': result.runtime,
                                    'error': result.error,
                                    'results': result.results
                                }"""

if old in content:
    content = content.replace(old, new)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Fixed sieve execution with proper remote handling!")
else:
    print("❌ Pattern not found")
