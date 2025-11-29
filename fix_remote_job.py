with open('coordinator.py', 'r') as f:
    content = f.read()

# Find and replace the else branch for reverse_sieve
old_else = """            else:
                # Standard job structure
                job_data = {
                    'job_id': job.job_id,
                    'search_type': 'residue_sieve',
                    'prng_type': job.prng_type,
                    'mapping_type': job.mapping_type,
                    'seeds': job.seeds,
                    'samples': job.samples,
                    'lmax': job.lmax,
                    'grid_size': job.grid_size,
                    'mining_mode': job.mining_mode,
                    'search_type': job.search_type,
                    'job_type': 'advanced_draw_matching' if job.search_type == 'draw_match' else 'standard_analysis',
                    'target_draw': job.target_draw,
                    'target_file': self.current_target_file,
                    'analysis_type': job.analysis_type,
                }"""

new_else = """            elif job.search_type == 'reverse_sieve':
                # Reverse sieve - use payload directly
                job_data = job.payload if job.payload else {}
                job_data['job_id'] = job.job_id
                job_data['target_file'] = self.current_target_file
            else:
                # Standard job structure
                job_data = {
                    'job_id': job.job_id,
                    'search_type': job.search_type,
                    'prng_type': job.prng_type,
                    'mapping_type': job.mapping_type,
                    'seeds': job.seeds,
                    'samples': job.samples,
                    'lmax': job.lmax,
                    'grid_size': job.grid_size,
                    'mining_mode': job.mining_mode,
                    'job_type': 'advanced_draw_matching' if job.search_type == 'draw_match' else 'standard_analysis',
                    'target_draw': job.target_draw,
                    'target_file': self.current_target_file,
                    'analysis_type': job.analysis_type,
                }"""

content = content.replace(old_else, new_else)

with open('coordinator.py', 'w') as f:
    f.write(content)

print("✅ Fixed execute_remote_job to use payload for reverse_sieve")
