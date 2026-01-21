#!/usr/bin/env python3
"""
RAMDISK PRELOAD PATCH - Team Alpha Approved
Applies /dev/shm preloading to run_scorer_meta_optimizer.sh
Implements all Team Beta required revisions:
  - No hardcoded IPs (extracts from distributed_config.json)
  - Copy-once sentinel (.ready file)
  - Config-driven paths (no hostname inference)
  - Sanity checks and explicit logging

Usage:
    python3 apply_ramdisk_preload_patch.py

Backup created automatically before patching.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

SCRIPT_PATH = "run_scorer_meta_optimizer.sh"
CONFIG_PATH = "distributed_config.json"

def create_backup(filepath):
    """Create timestamped backup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    
    with open(filepath, 'r') as f:
        content = f.read()
    with open(backup_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Backup created: {backup_path}")
    return backup_path

def get_ramdisk_preload_block():
    """Return the ramdisk preload shell code block."""
    return '''
# ============================================================
# RAMDISK PRELOAD (Team Beta Approved 2026-01-20)
# Eliminates disk I/O contention during worker startup
# ============================================================
echo ""
echo "[INFO] Preloading data to RAM disk on remote nodes..."

# Extract remote nodes from distributed_config.json (single source of truth)
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
    
    # Sanity check: verify /dev/shm is available
    ssh "$REMOTE" "df -h /dev/shm | grep -q shm" || {
        echo "    ⚠️  WARNING: /dev/shm not available on $REMOTE, skipping ramdisk"
        continue
    }
    
    # Copy-once guard: only copy if .ready sentinel missing
    ssh "$REMOTE" "
        mkdir -p /dev/shm/prng &&
        if [ ! -f /dev/shm/prng/.ready ]; then
            cp ~/distributed_prng_analysis/bidirectional_survivors_binary.npz /dev/shm/prng/ &&
            cp ~/distributed_prng_analysis/train_history.json /dev/shm/prng/ &&
            cp ~/distributed_prng_analysis/holdout_history.json /dev/shm/prng/ &&
            cp ~/distributed_prng_analysis/scorer_jobs.json /dev/shm/prng/ 2>/dev/null || true &&
            touch /dev/shm/prng/.ready &&
            echo '    ✓ Ramdisk preload complete'
        else
            echo '    ✓ Ramdisk already loaded (skipped)'
        fi
    "
done

echo "[INFO] Ramdisk preload phase complete"
echo ""
# ============================================================
'''

def find_insertion_point(content):
    """Find where to insert ramdisk preload (after 'Data copied to remote nodes')."""
    marker = "Data copied to remote nodes"
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if marker in line:
            return i + 1
    
    # Fallback: look for "Pushing latest" which comes after SCP
    for i, line in enumerate(lines):
        if "Pushing latest" in line:
            return i
    
    return None

def patch_shell_script():
    """Apply ramdisk preload patch to run_scorer_meta_optimizer.sh."""
    
    if not os.path.exists(SCRIPT_PATH):
        print(f"❌ ERROR: {SCRIPT_PATH} not found")
        return False
    
    # Create backup
    create_backup(SCRIPT_PATH)
    
    with open(SCRIPT_PATH, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "RAMDISK PRELOAD" in content:
        print(f"⚠️  {SCRIPT_PATH} already contains ramdisk preload. Skipping.")
        return True
    
    # Find insertion point
    lines = content.split('\n')
    insert_idx = find_insertion_point(content)
    
    if insert_idx is None:
        print("❌ ERROR: Could not find insertion point in script")
        print("   Looking for 'Data copied to remote nodes' or 'Pushing latest'")
        return False
    
    # Insert ramdisk block
    ramdisk_block = get_ramdisk_preload_block()
    lines.insert(insert_idx, ramdisk_block)
    
    # Write patched content
    with open(SCRIPT_PATH, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Patched {SCRIPT_PATH} with ramdisk preload")
    return True

def patch_job_generator():
    """
    Patch generate_scorer_jobs.py to use ramdisk paths for remote nodes.
    Uses config-driven approach, not hostname inference.
    """
    generator_path = "generate_scorer_jobs.py"
    
    if not os.path.exists(generator_path):
        print(f"⚠️  {generator_path} not found - will need manual path update in job generation")
        return False
    
    create_backup(generator_path)
    
    with open(generator_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "RAMDISK_PATH" in content or "/dev/shm/prng" in content:
        print(f"⚠️  {generator_path} already contains ramdisk references. Skipping.")
        return True
    
    # Find the data path constant or the place where paths are constructed
    # We need to add a helper function and modify path generation
    
    helper_code = '''
# ============================================================
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
# ============================================================

'''
    
    # Insert after imports
    import_end = 0
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_end = i + 1
        elif import_end > 0 and line.strip() and not line.startswith('#'):
            break
    
    lines.insert(import_end + 1, helper_code)
    
    with open(generator_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Patched {generator_path} with ramdisk helper")
    print(f"   NOTE: Job path generation may need manual update to use get_remote_data_path()")
    return True

def update_distributed_config():
    """Add ramdisk configuration to distributed_config.json."""
    
    if not os.path.exists(CONFIG_PATH):
        print(f"⚠️  {CONFIG_PATH} not found")
        return False
    
    create_backup(CONFIG_PATH)
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Add ramdisk settings to each remote node
    modified = False
    for node in config.get('nodes', []):
        if node.get('hostname') != 'localhost':
            if 'use_ramdisk' not in node:
                node['use_ramdisk'] = True
                node['ramdisk_path'] = "/dev/shm/prng"
                modified = True
    
    if modified:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Updated {CONFIG_PATH} with ramdisk configuration")
    else:
        print(f"⚠️  {CONFIG_PATH} already has ramdisk config or no remote nodes")
    
    return True

def create_ramdisk_clear_script():
    """Create helper script to clear ramdisk on remotes (for data refresh)."""
    script = '''#!/bin/bash
# clear_ramdisk.sh - Force refresh ramdisk data on next run
# Use after Step 1 regenerates survivor data

echo "Clearing ramdisk sentinel on remote nodes..."

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
    ssh "$REMOTE" "rm -f /dev/shm/prng/.ready && echo '    ✓ Sentinel cleared'"
done

echo "Done. Next job run will refresh ramdisk data."
'''
    
    script_path = "clear_ramdisk.sh"
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    print(f"✅ Created {script_path} (use after Step 1 to refresh data)")

def main():
    print("=" * 60)
    print("RAMDISK PRELOAD PATCH - Team Alpha Approved")
    print("=" * 60)
    print()
    
    # Verify we're in the right directory
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ ERROR: Must run from ~/distributed_prng_analysis/")
        print(f"   Could not find {CONFIG_PATH}")
        sys.exit(1)
    
    success = True
    
    # 1. Patch shell script
    print("\n[1/4] Patching run_scorer_meta_optimizer.sh...")
    if not patch_shell_script():
        success = False
    
    # 2. Update distributed_config.json
    print("\n[2/4] Updating distributed_config.json...")
    if not update_distributed_config():
        success = False
    
    # 3. Patch job generator (optional - may need manual work)
    print("\n[3/4] Patching generate_scorer_jobs.py...")
    patch_job_generator()  # Non-fatal if fails
    
    # 4. Create helper script
    print("\n[4/4] Creating clear_ramdisk.sh helper...")
    create_ramdisk_clear_script()
    
    print()
    print("=" * 60)
    if success:
        print("✅ PATCH COMPLETE")
        print()
        print("Next steps:")
        print("  1. Review changes: git diff")
        print("  2. Test with: ./run_scorer_meta_optimizer.sh 25")
        print("  3. If successful, commit: git add -A && git commit -m 'feat: ramdisk preload for remote nodes'")
        print()
        print("To clear ramdisk after Step 1 data refresh:")
        print("  ./clear_ramdisk.sh")
    else:
        print("⚠️  PATCH PARTIALLY COMPLETE - Review errors above")
    print("=" * 60)

if __name__ == "__main__":
    main()
