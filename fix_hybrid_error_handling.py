#!/usr/bin/env python3
"""Add error handling for hybrid PRNG failures"""

with open('coordinator.py', 'r') as f:
    content = f.read()

# Exact pattern from lines 1077-1082
old = '''                    if job_spec.get('search_type') == 'residue_sieve':
                        from sieve_filter import execute_sieve_job
                        result_dict = execute_sieve_job(job_spec, worker.gpu_id)
                        # Wrap dict in simple object for compatibility
                        class SieveResult:'''

new = '''                    if job_spec.get('search_type') == 'residue_sieve':
                        from sieve_filter import execute_sieve_job
                        try:
                            result_dict = execute_sieve_job(job_spec, worker.gpu_id)
                        except ValueError as e:
                            # Hybrid mode not supported for this PRNG
                            print(f"⚠️  {worker.node.hostname}(gpu{worker.gpu_id}) | {str(e)[:80]}")
                            result_dict = {
                                'success': False, 
                                'error': str(e), 
                                'runtime': 0, 
                                'survivors_found': 0
                            }
                        # Wrap dict in simple object for compatibility
                        class SieveResult:'''

if old in content:
    content = content.replace(old, new)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Added try/except around execute_sieve_job")
else:
    print("❌ Pattern not found - code may have changed")
    exit(1)
