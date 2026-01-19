#!/usr/bin/env python3
"""
SURGICAL PATCH #2: Preserve incremental fields in window_optimizer.py
======================================================================
Makes the final save MERGE with incremental output instead of overwriting.

Usage:
    python3 surgical_patch_window_optimizer.py [--dry-run]
"""

import sys
from pathlib import Path
from datetime import datetime
from shutil import copy2

TARGET_FILE = Path("window_optimizer.py")

# =============================================================================
# EXACT STRING TO FIND AND REPLACE
# =============================================================================

# Find the optimal_config build section and add merge logic before the write
FIND_1 = '''    optimal_config = {
        'window_size': best_config['window_size'],
        'offset': best_config['offset'],
        'skip_min': best_config['skip_min'],
        'skip_max': best_config['skip_max'],
        'sessions': best_config['sessions'],
        'prng_type': prng_type,
        'test_both_modes': test_both_modes,  # NEW: Record whether we tested both modes
        'seed_count': seed_count,
        'optimization_score': results['best_score'],
        # Survivor counts for watcher evaluation
        'forward_count': forward_count,
        'reverse_count': reverse_count,
        'bidirectional_count': bidirectional_count
    }'''

REPLACE_1 = '''    optimal_config = {
        'window_size': best_config['window_size'],
        'offset': best_config['offset'],
        'skip_min': best_config['skip_min'],
        'skip_max': best_config['skip_max'],
        'sessions': best_config['sessions'],
        'prng_type': prng_type,
        'test_both_modes': test_both_modes,  # NEW: Record whether we tested both modes
        'seed_count': seed_count,
        'optimization_score': results['best_score'],
        # Survivor counts for watcher evaluation
        'forward_count': forward_count,
        'reverse_count': reverse_count,
        'bidirectional_count': bidirectional_count
    }
    
    # === MERGE INCREMENTAL OUTPUT FIELDS (Patch 2026-01-18) ===
    # Preserve fields from incremental saves (crash recovery data)
    if Path(output_config).exists():
        try:
            with open(output_config, 'r') as f:
                existing = json.load(f)
            # Preserve incremental tracking fields
            incremental_fields = ['status', 'completed_trials', 'total_trials', 
                                  'best_trial_number', 'best_value', 'best_bidirectional_count',
                                  'last_updated', 'last_trial_number', 'last_trial_value']
            for field in incremental_fields:
                if field in existing:
                    optimal_config[field] = existing[field]
            # Mark as complete since we finished successfully
            optimal_config['status'] = 'complete'
            optimal_config['completed_at'] = datetime.now().isoformat()
        except (json.JSONDecodeError, IOError):
            pass  # If file is corrupt, just use new config
    # === END MERGE ==='''


# =============================================================================
# PATCH APPLICATION
# =============================================================================

def main():
    dry_run = "--dry-run" in sys.argv
    
    print("=" * 70)
    print("SURGICAL PATCH #2: Preserve incremental fields in window_optimizer.py")
    print("=" * 70)
    
    if not TARGET_FILE.exists():
        print(f"\n[ERROR] File not found: {TARGET_FILE}")
        print("        Run this script from ~/distributed_prng_analysis/")
        sys.exit(1)
    
    # Read original
    with open(TARGET_FILE, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "MERGE INCREMENTAL OUTPUT FIELDS" in content:
        print("\n[INFO] File already patched!")
        sys.exit(0)
    
    # Check target exists
    if FIND_1 not in content:
        print("\n[ERROR] Target string not found!")
        print("        File may have been modified. Aborting.")
        sys.exit(1)
    
    # Backup
    if not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = TARGET_FILE.with_suffix(f".py.bak.{timestamp}")
        copy2(TARGET_FILE, backup_path)
        print(f"\n[OK] Backup: {backup_path}")
    else:
        print("\n[DRY-RUN] Would create backup")
    
    # Apply patch
    patched_content = content.replace(FIND_1, REPLACE_1, 1)
    print("[OK] Patch 1: Add merge logic for incremental fields")
    
    # Write patched file
    if not dry_run:
        with open(TARGET_FILE, 'w') as f:
            f.write(patched_content)
        print(f"\n[OK] Wrote patched file: {TARGET_FILE}")
        
        # Validate syntax
        print("\n[...] Validating syntax...")
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', str(TARGET_FILE)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[ERROR] Syntax error!\n{result.stderr}")
            print("\n[RESTORE] Restoring backup...")
            copy2(backup_path, TARGET_FILE)
            print(f"[OK] Restored from {backup_path}")
            sys.exit(1)
        print("[OK] Syntax valid")
    else:
        print("\n[DRY-RUN] Would write patched file")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Patch applied!")
    print("=" * 70)
    print("\nNow the final save will preserve incremental tracking fields:")
    print("  status, completed_trials, total_trials, best_value, etc.")


if __name__ == "__main__":
    main()
