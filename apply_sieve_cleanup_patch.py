#!/usr/bin/env python3
"""
PATCH: sieve_filter.py Inter-Chunk GPU Cleanup
Approved: Team Beta 2026-01-26

Changes:
1. Add gc.collect() to _best_effort_gpu_cleanup() for consistency
2. Add inter-chunk cleanup calls to both forward sieve loops
"""

import sys
import shutil
from datetime import datetime

TARGET_FILE = "sieve_filter.py"
BACKUP_SUFFIX = f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"

def apply_patch():
    # Read original
    with open(TARGET_FILE, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_file = TARGET_FILE.replace('.py', BACKUP_SUFFIX)
    shutil.copy(TARGET_FILE, backup_file)
    print(f"✓ Backup created: {backup_file}")
    
    # =========================================================================
    # EDIT 1: Update _best_effort_gpu_cleanup() to include gc.collect()
    # =========================================================================
    
    old_cleanup = '''def _best_effort_gpu_cleanup():
    """Clean GPU memory after job completion - safe, best-effort"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass'''
    
    new_cleanup = '''def _best_effort_gpu_cleanup():
    """Clean GPU memory between chunks - safe, best-effort (Team Beta 2026-01-26)"""
    # 1. Python GC first - drop refs before GPU cleanup
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    # 2. PyTorch/ROCm cache clear
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    # 3. CuPy memory pool release
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass'''
    
    if old_cleanup not in content:
        print("✗ ERROR: Could not find _best_effort_gpu_cleanup() function to update")
        print("  Expected pattern not found. Manual intervention required.")
        sys.exit(1)
    
    content = content.replace(old_cleanup, new_cleanup, 1)
    print("✓ Edit 1: Updated _best_effort_gpu_cleanup() with gc.collect()")
    
    # =========================================================================
    # EDIT 2: Add inter-chunk cleanup to forward constant loop (around line 222)
    # =========================================================================
    
    old_loop1 = '''                            all_best_skips.append(skips[i])
                total_tested += n_seeds'''
    
    new_loop1 = '''                            all_best_skips.append(skips[i])
                # Inter-chunk cleanup (skip final chunk - Team Beta 2026-01-26)
                if chunk_start + chunk_size < seed_end:
                    _best_effort_gpu_cleanup()
                total_tested += n_seeds'''
    
    if old_loop1 not in content:
        print("✗ ERROR: Could not find loop 1 pattern (forward constant)")
        print("  Expected pattern not found. Manual intervention required.")
        sys.exit(1)
    
    content = content.replace(old_loop1, new_loop1, 1)
    print("✓ Edit 2: Added inter-chunk cleanup to forward constant loop")
    
    # =========================================================================
    # EDIT 3: Add inter-chunk cleanup to forward hybrid loop (around line 374)
    # =========================================================================
    
    old_loop2 = '''                            all_skip_sequences.append(skip_seq)
                total_tested += n_seeds'''
    
    new_loop2 = '''                            all_skip_sequences.append(skip_seq)
                # Inter-chunk cleanup (skip final chunk - Team Beta 2026-01-26)
                if chunk_start + chunk_size < seed_end:
                    _best_effort_gpu_cleanup()
                total_tested += n_seeds'''
    
    if old_loop2 not in content:
        print("✗ ERROR: Could not find loop 2 pattern (forward hybrid)")
        print("  Expected pattern not found. Manual intervention required.")
        sys.exit(1)
    
    content = content.replace(old_loop2, new_loop2, 1)
    print("✓ Edit 3: Added inter-chunk cleanup to forward hybrid loop")
    
    # =========================================================================
    # Write patched file
    # =========================================================================
    
    with open(TARGET_FILE, 'w') as f:
        f.write(content)
    
    print(f"✓ Patched file written: {TARGET_FILE}")
    
    # =========================================================================
    # Verify syntax
    # =========================================================================
    
    import py_compile
    try:
        py_compile.compile(TARGET_FILE, doraise=True)
        print("✓ Syntax check: PASSED")
    except py_compile.PyCompileError as e:
        print(f"✗ Syntax check: FAILED - {e}")
        print(f"  Restoring from backup: {backup_file}")
        shutil.copy(backup_file, TARGET_FILE)
        sys.exit(1)
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "="*60)
    print("PATCH APPLIED SUCCESSFULLY")
    print("="*60)
    print(f"Backup: {backup_file}")
    print("\nChanges made:")
    print("  1. _best_effort_gpu_cleanup() now includes gc.collect()")
    print("  2. Forward constant loop: inter-chunk cleanup added")
    print("  3. Forward hybrid loop: inter-chunk cleanup added")
    print("\nNext steps:")
    print("  1. Verify: grep -n 'Inter-chunk cleanup' sieve_filter.py")
    print("  2. Deploy: scp sieve_filter.py 192.168.3.120:~/distributed_prng_analysis/")
    print("  3. Deploy: scp sieve_filter.py 192.168.3.154:~/distributed_prng_analysis/")
    print("  4. Test:   ./benchmark_step1_stability_v2.sh 20 500000")

if __name__ == '__main__':
    apply_patch()
