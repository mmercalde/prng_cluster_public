#!/usr/bin/env python3
"""
Dynamic Sieve Fix - Based on Actual Code Inspection
Precise string replacements matching the real code
"""

import shutil
from datetime import datetime

print("╔════════════════════════════════════════════════════════════╗")
print("║    Dynamic Sieve Fix - Verified Against Real Code         ║")
print("╚════════════════════════════════════════════════════════════╝")
print()

# Backup files with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print("📦 Creating backups...")
shutil.copy2('coordinator.py', f'coordinator.py.bak_{timestamp}')
shutil.copy2('sieve_filter.py', f'sieve_filter.py.bak_{timestamp}')
print(f"   ✅ coordinator.py.bak_{timestamp}")
print(f"   ✅ sieve_filter.py.bak_{timestamp}")
print()

# =============================================================================
# FIX 1: sieve_filter.py lines 443-444
# Make execute_sieve_job accept 'seeds' list
# =============================================================================
print("📝 Fix 1: sieve_filter.py - accept 'seeds' list...")

with open('sieve_filter.py', 'r') as f:
    sieve_content = f.read()

# Exact match from line 443-444
old_sieve = '''        seed_start = job.get('seed_start', 0)
        seed_end = job.get('seed_end', 100000)'''

new_sieve = '''        # Accept both 'seeds' list (from dynamic) and seed_start/seed_end (legacy)
        if 'seeds' in job:
            seed_start = job['seeds'][0]
            seed_end = job['seeds'][-1] + 1
        else:
            seed_start = job.get('seed_start', 0)
            seed_end = job.get('seed_end', 100000)'''

if old_sieve in sieve_content:
    sieve_content = sieve_content.replace(old_sieve, new_sieve)
    with open('sieve_filter.py', 'w') as f:
        f.write(sieve_content)
    print("   ✅ execute_sieve_job now accepts 'seeds' list")
else:
    print("   ❌ FAILED - Pattern not found in sieve_filter.py")
    print("   Expected lines 443-444:")
    print("        seed_start = job.get('seed_start', 0)")
    print("        seed_end = job.get('seed_end', 100000)")
    exit(1)

# =============================================================================
# FIX 2: coordinator.py lines 1298-1309
# Remove sieve bypass
# =============================================================================
print("📝 Fix 2: coordinator.py - remove sieve bypass...")

with open('coordinator.py', 'r') as f:
    coord_content = f.read()

# Exact match from lines 1298-1309
old_bypass = '''        # Check for sieve method FIRST before dynamic distribution
        if hasattr(args, 'method') and args.method == 'residue_sieve':
            print("🔬 Using Residue Sieve Method")
            # Sieve uses its own distribution, skip dynamic
            pass  # Fall through to sieve handling below
        else:
            use_dynamic = True # Default to dynamic distribution
            use_parallel_dynamic = True # Use the new parallel implementation
            if use_dynamic and use_parallel_dynamic:
                print("🚀 Using Parallel Dynamic Job Distribution Mode")
                return self.execute_truly_parallel_dynamic(
                    target_file, output_file, args, total_seeds, samples, lmax, grid_size
            )'''

new_bypass = '''        # ALL methods use dynamic distribution (including sieve)
        use_dynamic = True
        use_parallel_dynamic = True
        
        if hasattr(args, 'method') and args.method == 'residue_sieve':
            print("🔬 Using Residue Sieve Method (Dynamic Distribution)")
        
        if use_dynamic and use_parallel_dynamic:
            print("🚀 Using Parallel Dynamic Job Distribution Mode")
            return self.execute_truly_parallel_dynamic(
                target_file, output_file, args, total_seeds, samples, lmax, grid_size
            )'''

if old_bypass in coord_content:
    coord_content = coord_content.replace(old_bypass, new_bypass)
    print("   ✅ Removed sieve bypass")
else:
    print("   ❌ FAILED - Bypass pattern not found")
    exit(1)

# =============================================================================
# FIX 3: coordinator.py lines 1022-1032
# Add sieve job_spec creation
# =============================================================================
print("📝 Fix 3: coordinator.py - add sieve job_spec support...")

# Exact match from lines 1022-1032
old_jobspec = '''            job_spec = {
                'job_id': f"parallel_dynamic_{job_id}",
                'seeds': seed_list,
                'samples': samples,
                'lmax': lmax,
                'grid_size': grid_size,
                'prng_type': 'mt',
                'mapping_type': 'mod',
                'search_type': 'draw_match' if (hasattr(args, 'draw_match') and args.draw_match is not None) else 'correlation',
                'target_draw': [args.draw_match] if (hasattr(args, 'draw_match') and args.draw_match is not None) else None,
                'analysis_type': getattr(args, 'analysis_type', 'statistical')
            }'''

new_jobspec = '''            # Create job_spec based on analysis method
            if hasattr(args, 'method') and args.method == 'residue_sieve':
                # SIEVE job
                job_spec = {
                    'job_id': f"sieve_{job_id:03d}",
                    'seeds': seed_list,
                    'dataset_path': args.target_file,
                    'window_size': getattr(args, 'window_size', 768),
                    'min_match_threshold': getattr(args, 'threshold', 0.01),
                    'skip_range': [getattr(args, 'skip_min', 0), getattr(args, 'skip_max', 20)],
                    'offset': getattr(args, 'offset', 0),
                    'sessions': ['midday', 'evening'],
                    'search_type': 'residue_sieve'
                }
                if hasattr(args, 'prng_type'):
                    job_spec['prng_families'] = [args.prng_type]
            else:
                # STANDARD job
                job_spec = {
                    'job_id': f"parallel_dynamic_{job_id}",
                    'seeds': seed_list,
                    'samples': samples,
                    'lmax': lmax,
                    'grid_size': grid_size,
                    'prng_type': 'mt',
                    'mapping_type': 'mod',
                    'search_type': 'draw_match' if (hasattr(args, 'draw_match') and args.draw_match is not None) else 'correlation',
                    'target_draw': [args.draw_match] if (hasattr(args, 'draw_match') and args.draw_match is not None) else None,
                    'analysis_type': getattr(args, 'analysis_type', 'statistical')
                }'''

if old_jobspec in coord_content:
    coord_content = coord_content.replace(old_jobspec, new_jobspec)
    print("   ✅ Added sieve job_spec creation")
else:
    print("   ❌ FAILED - job_spec pattern not found")
    exit(1)

# =============================================================================
# FIX 4: coordinator.py lines 1058-1074
# Add sieve job execution in worker_loop
# =============================================================================
print("📝 Fix 4: coordinator.py - add sieve execution in worker...")

# Exact match from lines 1058-1074
old_worker = '''                    # Convert to JobSpec object
                    job = JobSpec(
                        job_id=job_spec['job_id'],
                        prng_type=job_spec['prng_type'],
                        mapping_type=job_spec['mapping_type'],
                        seeds=job_spec['seeds'],
                        samples=job_spec['samples'],
                        lmax=job_spec['lmax'],
                        grid_size=job_spec['grid_size'],
                        mining_mode=self.is_mining_node(worker.node),
                        search_type=job_spec.get('search_type', 'correlation'),
                        target_draw=job_spec.get('target_draw', None),
                        analysis_type=job_spec.get('analysis_type', 'statistical'),
                        attempt=0
                    )
                    # Execute job using existing infrastructure
                    result = self.execute_gpu_job(job, worker)'''

new_worker = '''                    # Handle sieve jobs separately
                    if job_spec.get('search_type') == 'residue_sieve':
                        from sieve_filter import execute_sieve_job
                        result = execute_sieve_job(job_spec, worker.gpu_id)
                        my_results.append(result)
                        jobs_completed += 1
                        work_queue.task_done()
                        continue
                    
                    # Convert to JobSpec object (standard jobs)
                    job = JobSpec(
                        job_id=job_spec['job_id'],
                        prng_type=job_spec['prng_type'],
                        mapping_type=job_spec['mapping_type'],
                        seeds=job_spec['seeds'],
                        samples=job_spec['samples'],
                        lmax=job_spec['lmax'],
                        grid_size=job_spec['grid_size'],
                        mining_mode=self.is_mining_node(worker.node),
                        search_type=job_spec.get('search_type', 'correlation'),
                        target_draw=job_spec.get('target_draw', None),
                        analysis_type=job_spec.get('analysis_type', 'statistical'),
                        attempt=0
                    )
                    # Execute job using existing infrastructure
                    result = self.execute_gpu_job(job, worker)'''

if old_worker in coord_content:
    coord_content = coord_content.replace(old_worker, new_worker)
    print("   ✅ Added sieve worker execution")
else:
    print("   ❌ FAILED - Worker pattern not found")
    exit(1)

# Write updated coordinator.py
with open('coordinator.py', 'w') as f:
    f.write(coord_content)

print()
print("=" * 60)
print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
print("=" * 60)
print()
print("Backups:")
print(f"  coordinator.py.bak_{timestamp}")
print(f"  sieve_filter.py.bak_{timestamp}")
print()
print("Changes:")
print("  1. ✅ execute_sieve_job accepts 'seeds' list")
print("  2. ✅ Sieve bypass removed")
print("  3. ✅ Dynamic executor creates sieve job_spec")
print("  4. ✅ Worker executes sieve jobs")
print()
print("Test now:")
print("  python3 coordinator.py --resume-policy restart --max-concurrent 26 \\")
print("      daily3.json --method residue_sieve --prng-type lcg32 \\")
print("      --skip-min 0 --skip-max 20 --threshold 0.01 \\")
print("      --window-size 768 --session-filter both \\")
print("      --seed-start 0 --seeds 10000")
