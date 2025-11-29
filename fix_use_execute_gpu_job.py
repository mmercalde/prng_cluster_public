#!/usr/bin/env python3
"""Use execute_gpu_job instead of direct execute_sieve_job call"""

with open('coordinator.py', 'r') as f:
    content = f.read()

# Revert the CUDA_VISIBLE_DEVICES change first
old1 = """                    # Handle sieve jobs separately
                    if job_spec.get('search_type') == 'residue_sieve':
                        from sieve_filter import execute_sieve_job
                        import os
                        # Set CUDA device for this thread
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(worker.gpu_id)
                        try:
                            # Use gpu_id=0 since CUDA_VISIBLE_DEVICES limits to one device
                            result_dict = execute_sieve_job(job_spec, 0)"""

new1 = """                    # Handle sieve jobs separately
                    if job_spec.get('search_type') == 'residue_sieve':
                        # Convert job_spec dict to JobSpec object
                        from dataclasses import dataclass, field
                        from typing import List, Optional
                        
                        @dataclass
                        class JobSpec:
                            job_id: str
                            seeds: List[int]
                            samples: int
                            lmax: int
                            grid_size: int
                            mining_mode: bool = False
                            search_type: str = 'correlation'
                            target_draw: Optional[List[int]] = None
                            analysis_type: str = 'statistical'
                            attempt: int = 0
                        
                        job = JobSpec(
                            job_id=job_spec.get('job_id', 'unknown'),
                            seeds=job_spec.get('seeds', []),
                            samples=job_spec.get('samples', 1000),
                            lmax=job_spec.get('lmax', 10),
                            grid_size=job_spec.get('grid_size', 50),
                            search_type='residue_sieve'
                        )
                        try:
                            # Use execute_gpu_job which handles local vs remote execution
                            result = self.execute_gpu_job(job, worker)
                            result_dict = {
                                'success': result.success,
                                'runtime': result.runtime,
                                'error': result.error,
                                'results': result.results
                            }"""

# Find and replace
if old1 in content:
    content = content.replace(old1, new1)
    with open('coordinator.py', 'w') as f:
        f.write(content)
    print("✅ Fixed to use execute_gpu_job!")
else:
    print("❌ Pattern not found - trying alternate approach")
    print("Searching for simpler pattern...")
