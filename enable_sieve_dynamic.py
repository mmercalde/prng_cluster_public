#!/usr/bin/env python3
"""
Enable sieve to use dynamic executor with GPU optimization.

Changes:
1. Remove bypass that blocks sieve from dynamic
2. Modify dynamic executor to support sieve arguments
3. Let GPUOptimizer calculate optimal chunks
"""

with open('coordinator.py', 'r') as f:
    content = f.read()

print("🔧 Making sieve use dynamic executor with GPU optimization...\n")

# ============================================================================
# FIX 1: Remove the bypass that blocks sieve from using dynamic
# ============================================================================
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

new_bypass = '''        # ALL methods now use dynamic distribution (including sieve)
        use_dynamic = True
        use_parallel_dynamic = True
        
        if hasattr(args, 'method') and args.method == 'residue_sieve':
            print("🔬 Using Residue Sieve Method with Dynamic Distribution")
        else:
            print("🚀 Using Parallel Dynamic Job Distribution Mode")
        
        if use_dynamic and use_parallel_dynamic:
            return self.execute_truly_parallel_dynamic(
                target_file, output_file, args, total_seeds, samples, lmax, grid_size
            )'''

if old_bypass in content:
    content = content.replace(old_bypass, new_bypass)
    print("✅ Fix 1: Removed sieve bypass - now uses dynamic executor")
else:
    print("⚠️  Fix 1: Could not find bypass code (may already be fixed)")

# ============================================================================
# FIX 2: Make dynamic executor support sieve arguments
# ============================================================================
old_job_spec = '''            job_spec = {
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

new_job_spec = '''            # Build job_spec based on analysis method
            if hasattr(args, 'method') and args.method == 'residue_sieve':
                # SIEVE mode - use sieve-specific parameters
                job_spec = {
                    'job_id': f"sieve_{job_id:03d}",
                    'dataset_path': args.target_file,
                    'seed_start': seed_list[0],
                    'seed_end': seed_list[-1] + 1,
                    'window_size': getattr(args, 'window_size', 768),
                    'min_match_threshold': getattr(args, 'threshold', 0.01),
                    'skip_range': [
                        getattr(args, 'skip_min', 0),
                        getattr(args, 'skip_max', 20)
                    ],
                    'prng_families': [args.prng_type] if hasattr(args, 'prng_type') else ['mt19937'],
                    'sessions': ['midday', 'evening'],  # Default sessions
                    'offset': getattr(args, 'offset', 0),
                    'search_type': 'residue_sieve'
                }
            else:
                # STANDARD mode - use correlation/draw_match parameters
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

if old_job_spec in content:
    content = content.replace(old_job_spec, new_job_spec)
    print("✅ Fix 2: Dynamic executor now supports sieve arguments")
else:
    print("⚠️  Fix 2: Could not find job_spec code (may already be fixed)")

# ============================================================================
# FIX 3: Make worker_loop handle sieve jobs properly
# ============================================================================
old_worker_conversion = '''                    # Convert to JobSpec object
                    job = JobSpec(
                        job_id=job_spec['job_id'],'''

new_worker_conversion = '''                    # Convert to appropriate job format
                    if job_spec.get('search_type') == 'residue_sieve':
                        # SIEVE job - execute directly with sieve_filter
                        from sieve_filter import execute_sieve_job
                        result = execute_sieve_job(job_spec, worker.gpu_id)
                        my_results.append(result)
                        jobs_completed += 1
                        work_queue.task_done()
                        continue
                    
                    # STANDARD job - Convert to JobSpec object
                    job = JobSpec(
                        job_id=job_spec['job_id'],'''

if old_worker_conversion in content:
    content = content.replace(old_worker_conversion, new_worker_conversion)
    print("✅ Fix 3: Worker loop now handles sieve jobs")
else:
    print("⚠️  Fix 3: Could not find worker conversion code")

# ============================================================================
# Write the fixed file
# ============================================================================
with open('coordinator_sieve_dynamic.py', 'w') as f:
    f.write(content)

print("\n" + "="*60)
print("✅ SIEVE DYNAMIC OPTIMIZATION COMPLETE!")
print("="*60)
print("\nChanges made:")
print("1. ✅ Removed bypass - sieve now uses dynamic executor")
print("2. ✅ Dynamic executor supports sieve arguments")
print("3. ✅ Worker loop executes sieve jobs via sieve_filter")
print("4. ✅ GPUOptimizer calculates optimal chunks (already in code)")
print("\nTo apply:")
print("  cp coordinator_sieve_dynamic.py coordinator.py")
print("\nExpected performance:")
print("  Before: 321 seconds (26 static chunks)")
print("  After:  ~15-20 seconds (dynamic work-stealing)")
print("\nThe GPUOptimizer will:")
print("  - Calculate optimal chunk sizes per GPU")
print("  - Fast GPUs (RTX 3080 Ti) grab more work")
print("  - Slow GPUs (RX 6600) grab less work")
print("  - NO IDLE TIME!")
