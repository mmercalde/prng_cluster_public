#!/usr/bin/env python3
"""
UNIFIED RAMDISK PATH PATCH - Team Beta Approved Option A
Implements unified /dev/shm/prng paths across all nodes (Zeus + remotes)

Changes:
1. Add localhost preload to run_scorer_meta_optimizer.sh
2. Fix generate_scorer_jobs.py to use DATA_ROOT for job args
3. Update clear_ramdisk.sh to clear localhost sentinel

Usage:
    python3 apply_unified_ramdisk_patch.py
"""

import os
import sys
from datetime import datetime

def create_backup(filepath):
    """Create timestamped backup."""
    if not os.path.exists(filepath):
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    with open(filepath, 'r') as f:
        content = f.read()
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"  ✅ Backup: {backup_path}")
    return backup_path

def patch_shell_script():
    """Add localhost preload to run_scorer_meta_optimizer.sh"""
    filepath = "run_scorer_meta_optimizer.sh"
    
    if not os.path.exists(filepath):
        print(f"  ❌ {filepath} not found")
        return False
    
    create_backup(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the existing ramdisk preload block and add localhost
    old_block = '''[INFO] Preloading data to RAM disk on remote nodes...'''
    new_block = '''[INFO] Preloading data to RAM disk on ALL nodes (unified path)...

# Localhost preload (no SSH)
echo "  → localhost"
mkdir -p /dev/shm/prng
if [ ! -f /dev/shm/prng/.ready ]; then
    cp ~/distributed_prng_analysis/bidirectional_survivors_binary.npz /dev/shm/prng/ &&
    cp ~/distributed_prng_analysis/train_history.json /dev/shm/prng/ &&
    cp ~/distributed_prng_analysis/holdout_history.json /dev/shm/prng/ &&
    cp ~/distributed_prng_analysis/scorer_jobs.json /dev/shm/prng/ 2>/dev/null || true &&
    touch /dev/shm/prng/.ready &&
    echo "    ✓ Ramdisk preload complete"
else
    echo "    ✓ Ramdisk already loaded (skipped)"
fi

# Remote nodes preload'''
    
    if "localhost preload" in content:
        print(f"  ⚠️  {filepath} already has localhost preload. Skipping.")
        return True
    
    content = content.replace(old_block, new_block)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Patched {filepath} with localhost preload")
    return True

def patch_job_generator():
    """Fix generate_scorer_jobs.py to use DATA_ROOT"""
    filepath = "generate_scorer_jobs.py"
    
    if not os.path.exists(filepath):
        print(f"  ❌ {filepath} not found")
        return False
    
    create_backup(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace the helper function with simpler DATA_ROOT constant
    old_helper = '''# ============================================================
# RAMDISK PATH HELPER (Team Beta Approved 2026-01-20)
# ============================================================
RAMDISK_PATH = "/dev/shm/prng"
USE_RAMDISK_FOR_REMOTES = True  # Set False to disable

def get_remote_data_path(filename):
    """Return ramdisk path for remote nodes."""
    if USE_RAMDISK_FOR_REMOTES:
        return f"{RAMDISK_PATH}/{filename}"
    else:
        return f"/home/michael/distributed_prng_analysis/{filename}"
# ============================================================'''

    new_helper = '''# ============================================================
# UNIFIED RAMDISK PATH (Team Beta Approved 2026-01-20 - Option A)
# All nodes use /dev/shm/prng for job data (unified path)
# ============================================================
USE_RAMDISK = True  # Set False to use SSD paths

if USE_RAMDISK:
    DATA_ROOT = "/dev/shm/prng"
else:
    DATA_ROOT = "/home/michael/distributed_prng_analysis"
# ============================================================'''

    content = content.replace(old_helper, new_helper)
    
    # Replace the hardcoded path assignment
    old_path = 'remote_data_path = "/home/michael/distributed_prng_analysis"'
    new_path = '# Path now uses unified DATA_ROOT (ramdisk or SSD based on USE_RAMDISK flag)'
    
    content = content.replace(old_path, new_path)
    
    # Replace path usage in job args
    old_args = 'f"{remote_data_path}/{Path(args.survivors).name}"'
    new_args = 'f"{DATA_ROOT}/{Path(args.survivors).name}"'
    content = content.replace(old_args, new_args)
    
    old_args2 = 'f"{remote_data_path}/{Path(args.train_history).name}"'
    new_args2 = 'f"{DATA_ROOT}/{Path(args.train_history).name}"'
    content = content.replace(old_args2, new_args2)
    
    old_args3 = 'f"{remote_data_path}/{Path(args.holdout_history).name}"'
    new_args3 = 'f"{DATA_ROOT}/{Path(args.holdout_history).name}"'
    content = content.replace(old_args3, new_args3)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Patched {filepath} with unified DATA_ROOT")
    return True

def patch_clear_script():
    """Update clear_ramdisk.sh to include localhost"""
    filepath = "clear_ramdisk.sh"
    
    if not os.path.exists(filepath):
        print(f"  ❌ {filepath} not found")
        return False
    
    create_backup(filepath)
    
    new_content = '''#!/bin/bash
# clear_ramdisk.sh - Force refresh ramdisk data on next run
# Use after Step 1 regenerates survivor data
# Team Beta Approved: Option A (unified paths) - clears ALL nodes including localhost

echo "Clearing ramdisk sentinel on ALL nodes..."

# Clear localhost first (no SSH)
echo "  → localhost"
rm -f /dev/shm/prng/.ready && echo "    ✓ Sentinel cleared" || echo "    ⚠️  No sentinel found"

# Clear remote nodes
REMOTE_NODES=$(python3 -c "
import json
with open('distributed_config.json') as f:
    cfg = json.load(f)
for node in cfg['nodes']:
    if node['hostname'] != 'localhost':
        print(node['hostname'])
")

for REMOTE in $REMOTE_NODES; do
    echo "  → $REMOTE"
    ssh "$REMOTE" "rm -f /dev/shm/prng/.ready && echo '    ✓ Sentinel cleared'" 2>/dev/null || echo "    ⚠️  Could not reach $REMOTE"
done

echo ""
echo "Done. Next job run will refresh ramdisk data on all nodes."
'''
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    os.chmod(filepath, 0o755)
    
    print(f"  ✅ Patched {filepath} with localhost support")
    return True

def main():
    print("=" * 60)
    print("UNIFIED RAMDISK PATH PATCH - Team Beta Option A")
    print("=" * 60)
    print()
    
    if not os.path.exists("distributed_config.json"):
        print("❌ ERROR: Must run from ~/distributed_prng_analysis/")
        sys.exit(1)
    
    print("[1/3] Patching run_scorer_meta_optimizer.sh (localhost preload)...")
    patch_shell_script()
    
    print("\n[2/3] Patching generate_scorer_jobs.py (DATA_ROOT)...")
    patch_job_generator()
    
    print("\n[3/3] Patching clear_ramdisk.sh (localhost support)...")
    patch_clear_script()
    
    print()
    print("=" * 60)
    print("✅ UNIFIED RAMDISK PATCH COMPLETE")
    print()
    print("Verification:")
    print("  grep 'DATA_ROOT' generate_scorer_jobs.py")
    print("  grep 'localhost' run_scorer_meta_optimizer.sh | head -5")
    print()
    print("Test:")
    print("  ./clear_ramdisk.sh              # Clear all sentinels")
    print("  ./run_scorer_meta_optimizer.sh 10  # Test 10 trials")
    print()
    print("Expected in job args:")
    print('  "/dev/shm/prng/bidirectional_survivors_binary.npz"')
    print("=" * 60)

if __name__ == "__main__":
    main()
