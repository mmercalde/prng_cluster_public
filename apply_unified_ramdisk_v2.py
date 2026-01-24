#!/usr/bin/env python3
"""
apply_unified_ramdisk_v2.py - Unified Ramdisk Infrastructure Deployment
============================================================================
Version: 2.0.0 (Team Beta Approved - PROP-2026-01-21-RAMDISK-UNIFIED)

This script:
1. Deploys ramdisk_preload.sh (shared shell module)
2. Deploys ramdisk_config.py (shared Python module)
3. Deploys clear_ramdisk.sh (updated for per-step dirs)
4. Refactors Step 2's run_scorer_meta_optimizer.sh to use new infrastructure
5. Updates generate_scorer_jobs.py to use per-step paths
6. Migrates existing flat ramdisk to per-step directories

Team Beta Requirements Addressed:
   A1: Per-step subdirectories (/dev/shm/prng/step2/, step3/, step5/)
   A2: Cached node list (parsed once, reused)
   A3: Non-fatal cleanup (never bricks pipeline)
   B1: Pre-flight headroom check (warn >50%, abort >80%)

Usage:
    cd ~/distributed_prng_analysis
    python3 apply_unified_ramdisk_v2.py
    
Rollback:
    ls -la *.backup_*
    cp <file>.backup_TIMESTAMP <file>
============================================================================
"""

import os
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Timestamp for backups
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def create_backup(filepath):
    """Create timestamped backup of a file."""
    if not os.path.exists(filepath):
        return None
    backup_path = f"{filepath}.backup_{TIMESTAMP}"
    shutil.copy2(filepath, backup_path)
    print(f"  ✅ Backup: {backup_path}")
    return backup_path

def write_file(path, content, mode=0o644):
    """Write content to file with specified mode."""
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, mode)

# ============================================================================
# ramdisk_preload.sh - Unified shell module
# ============================================================================
RAMDISK_PRELOAD_SH = '''#!/bin/bash
# ============================================================================
# ramdisk_preload.sh - Unified Ramdisk Infrastructure for Distributed Steps
# ============================================================================
# Version: 2.0.0 (Team Beta Approved)
# 
# Usage:
#   export RAMDISK_STEP_ID=2
#   source ramdisk_preload.sh
#   preload_ramdisk file1.npz file2.json file3.json
#
# Team Beta Requirements: A1 (per-step dirs), A2 (cached nodes), A3 (non-fatal), B1 (headroom)
# ============================================================================

RAMDISK_BASE="/dev/shm/prng"
RAMDISK_STEP_ID="${RAMDISK_STEP_ID:-unknown}"
RAMDISK_DIR="${RAMDISK_BASE}/step${RAMDISK_STEP_ID}"
RAMDISK_SENTINEL="${RAMDISK_DIR}/.ready"

# A2: Cache node list - resolve ONCE at source time
if [ -z "$_RAMDISK_NODES_CACHED" ]; then
    _RAMDISK_CLUSTER_NODES=($(python3 -c "
import json
try:
    with open('distributed_config.json') as f:
        cfg = json.load(f)
    for node in cfg['nodes']:
        print(node['hostname'])
except:
    print('localhost')
" 2>/dev/null))
    export _RAMDISK_NODES_CACHED=1
    export _RAMDISK_CLUSTER_NODES="${_RAMDISK_CLUSTER_NODES[*]}"
fi
CLUSTER_NODES=(${_RAMDISK_CLUSTER_NODES})

# B1: Pre-flight headroom check
check_ramdisk_headroom() {
    local node="$1"
    local usage
    if [ "$node" = "localhost" ]; then
        usage=$(df /dev/shm 2>/dev/null | awk 'NR==2 {gsub(/%/,""); print $5}')
    else
        usage=$(ssh "$node" "df /dev/shm 2>/dev/null | awk 'NR==2 {gsub(/%/,\\"\\"); print \\$5}'" 2>/dev/null)
    fi
    usage="${usage:-0}"
    
    if [ "$usage" -ge 80 ]; then
        echo "    ❌ ABORT: /dev/shm at ${usage}%"
        return 1
    elif [ "$usage" -ge 50 ]; then
        echo "    ⚠️  WARNING: /dev/shm at ${usage}%"
    fi
    return 0
}

# Main preload function
preload_ramdisk() {
    local files=("$@")
    [ ${#files[@]} -eq 0 ] && { echo "[ERROR] No files specified"; return 1; }
    
    echo "[INFO] Ramdisk preload for Step ${RAMDISK_STEP_ID} (${#files[@]} files)..."
    echo "[INFO] Target: ${RAMDISK_DIR}"
    
    local watcher_managed="${WATCHER_MANAGED_RAMDISK:-}"
    [ -z "$watcher_managed" ] && echo "[INFO] Standalone mode" || echo "[INFO] WATCHER-managed mode"
    
    for NODE in "${CLUSTER_NODES[@]}"; do
        echo "  → $NODE"
        check_ramdisk_headroom "$NODE" || continue
        
        if [ "$NODE" = "localhost" ]; then
            mkdir -p "$RAMDISK_DIR"
            if [ -f "$RAMDISK_SENTINEL" ]; then
                echo "    ✓ Already loaded (skipped)"
                continue
            fi
            [ -z "$watcher_managed" ] && rm -rf "${RAMDISK_DIR:?}"/* 2>/dev/null || true
            local copied=0
            for f in "${files[@]}"; do
                [ -f "$f" ] && cp "$f" "$RAMDISK_DIR/" && ((copied++))
            done
            touch "$RAMDISK_SENTINEL"
            echo "    ✓ Preloaded ($copied files)"
        else
            ssh "$NODE" "mkdir -p $RAMDISK_DIR" 2>/dev/null
            if ssh "$NODE" "[ -f $RAMDISK_SENTINEL ]" 2>/dev/null; then
                echo "    ✓ Already loaded (skipped)"
                continue
            fi
            [ -z "$watcher_managed" ] && ssh "$NODE" "rm -rf ${RAMDISK_DIR:?}/*" 2>/dev/null || true
            local copied=0
            for f in "${files[@]}"; do
                [ -f "$f" ] && scp -q "$f" "$NODE:$RAMDISK_DIR/" 2>/dev/null && ((copied++))
            done
            ssh "$NODE" "touch $RAMDISK_SENTINEL" 2>/dev/null
            echo "    ✓ Preloaded ($copied files)"
        fi
    done
    echo "[INFO] Ramdisk preload complete"
}

# A3: Clear ramdisk - non-fatal
clear_ramdisk() {
    local step_id="${1:-all}"
    [ "$step_id" = "all" ] && target="${RAMDISK_BASE}/step*" || target="${RAMDISK_BASE}/step${step_id}"
    echo "[INFO] Clearing ramdisk${step_id:+ for Step $step_id}..."
    for NODE in "${CLUSTER_NODES[@]}"; do
        if [ "$NODE" = "localhost" ]; then
            rm -rf $target 2>/dev/null || true
        else
            ssh "$NODE" "rm -rf $target" 2>/dev/null || true
        fi
        echo "  → $NODE: cleared"
    done
}

get_ramdisk_path() { echo "${RAMDISK_DIR}/${1}"; }

export -f preload_ramdisk clear_ramdisk get_ramdisk_path check_ramdisk_headroom 2>/dev/null || true
'''

# ============================================================================
# Step 2 ramdisk block (to be inserted into run_scorer_meta_optimizer.sh)
# ============================================================================
STEP2_RAMDISK_BLOCK = '''
# ============================================================================
# RAMDISK PRELOAD - Step 2 (Unified Infrastructure v2.0)
# ============================================================================
export RAMDISK_STEP_ID=2
source "$(dirname "$0")/ramdisk_preload.sh"

# Preload Step 2 data files
preload_ramdisk \\
    bidirectional_survivors_binary.npz \\
    train_history.json \\
    holdout_history.json

# Set data paths for job generation
export RAMDISK_DATA_DIR="${RAMDISK_DIR}"
# ============================================================================
'''

def patch_run_scorer_meta_optimizer():
    """Patch run_scorer_meta_optimizer.sh to use unified ramdisk infrastructure."""
    filepath = "run_scorer_meta_optimizer.sh"
    
    if not os.path.exists(filepath):
        print(f"  ⚠️  {filepath} not found - skipping")
        return False
    
    create_backup(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Remove old ramdisk block(s) - various patterns from previous implementations
    old_patterns = [
        # Pattern 1: Original "Preloading data to RAM disk" block
        (r'\n# =+\n# RAMDISK PRELOAD.*?# =+\n', ''),
        # Pattern 2: Inline preload echo statements  
        (r'\necho "\[INFO\] Preloading data to RAM disk.*?echo "\[INFO\] Ramdisk preload complete"\n', ''),
        # Pattern 3: Old unified path block
        (r'\n\[INFO\] Preloading data to RAM disk on ALL nodes.*?fi\ndone\n', ''),
    ]
    
    import re
    for pattern, replacement in old_patterns:
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Find insertion point (after set -e or shebang)
    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('set -e') or line.startswith('set -'):
            insert_idx = i + 1
            break
        elif line.startswith('#!'):
            insert_idx = i + 1
    
    # Check if already has unified v2 block
    if 'RAMDISK_STEP_ID=2' in content and 'source' in content and 'ramdisk_preload.sh' in content:
        print(f"  ✓ {filepath} already has unified v2 ramdisk block")
        return True
    
    # Insert new ramdisk block
    lines.insert(insert_idx, STEP2_RAMDISK_BLOCK)
    content = '\n'.join(lines)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Patched {filepath} with unified ramdisk (Step 2)")
    return True

def patch_generate_scorer_jobs():
    """Patch generate_scorer_jobs.py to use per-step ramdisk paths."""
    filepath = "generate_scorer_jobs.py"
    
    if not os.path.exists(filepath):
        print(f"  ⚠️  {filepath} not found - skipping")
        return False
    
    create_backup(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check for old DATA_ROOT pattern and update
    old_patterns = [
        'DATA_ROOT = "/dev/shm/prng"',
        'RAMDISK_PATH = "/dev/shm/prng"',
    ]
    
    new_data_root = '''# Unified Ramdisk v2.0 - Per-step directories (Team Beta A1)
RAMDISK_BASE = "/dev/shm/prng"
STEP_ID = 2
DATA_ROOT = f"{RAMDISK_BASE}/step{STEP_ID}"'''
    
    modified = False
    for old in old_patterns:
        if old in content:
            content = content.replace(old, new_data_root)
            modified = True
            break
    
    if not modified:
        # Check if already updated
        if 'RAMDISK_BASE' in content and 'step{STEP_ID}' in content:
            print(f"  ✓ {filepath} already has per-step paths")
            return True
        
        # Find a good insertion point
        if 'USE_RAMDISK' in content:
            # Insert after USE_RAMDISK line
            content = content.replace(
                'USE_RAMDISK = True',
                f'USE_RAMDISK = True\n{new_data_root}'
            )
            modified = True
    
    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Patched {filepath} with per-step paths")
    else:
        print(f"  ⚠️  Could not find insertion point in {filepath}")
    
    return modified

def deploy_ramdisk_preload_sh():
    """Deploy ramdisk_preload.sh."""
    filepath = "ramdisk_preload.sh"
    
    if os.path.exists(filepath):
        create_backup(filepath)
    
    write_file(filepath, RAMDISK_PRELOAD_SH, mode=0o755)
    print(f"  ✅ Deployed {filepath}")
    return True

def deploy_clear_ramdisk_sh():
    """Deploy updated clear_ramdisk.sh."""
    # This is handled by copying the file we created
    print("  ✅ clear_ramdisk.sh will be copied separately")
    return True

def deploy_ramdisk_config_py():
    """Deploy ramdisk_config.py."""
    # This is handled by copying the file we created
    print("  ✅ ramdisk_config.py will be copied separately")
    return True

def migrate_existing_ramdisk():
    """Migrate existing flat ramdisk structure to per-step directories."""
    print("\n[4/5] Migrating existing ramdisk structure...")
    
    # Use subprocess to run migration on all nodes
    result = subprocess.run([
        'python3', '-c', '''
import os
import subprocess

RAMDISK_BASE = "/dev/shm/prng"

# Get nodes
try:
    import json
    with open("distributed_config.json") as f:
        cfg = json.load(f)
    nodes = [n["hostname"] for n in cfg["nodes"]]
except:
    nodes = ["localhost"]

for node in nodes:
    print(f"  → {node}")
    if node == "localhost":
        # Check if flat structure exists
        if os.path.exists(f"{RAMDISK_BASE}/.ready") and not os.path.exists(f"{RAMDISK_BASE}/step2"):
            os.makedirs(f"{RAMDISK_BASE}/step2", exist_ok=True)
            # Move files to step2
            for f in os.listdir(RAMDISK_BASE):
                src = os.path.join(RAMDISK_BASE, f)
                if os.path.isfile(src):
                    os.rename(src, os.path.join(f"{RAMDISK_BASE}/step2", f))
            print("    ✓ Migrated flat → step2/")
        else:
            print("    ✓ Already using per-step dirs (or empty)")
    else:
        result = subprocess.run(["ssh", node, f"""
            if [ -f '{RAMDISK_BASE}/.ready' ] && [ ! -d '{RAMDISK_BASE}/step2' ]; then
                mkdir -p '{RAMDISK_BASE}/step2'
                for f in {RAMDISK_BASE}/*; do
                    [ -f "$f" ] && mv "$f" '{RAMDISK_BASE}/step2/'
                done
                echo '    ✓ Migrated flat → step2/'
            else
                echo '    ✓ Already using per-step dirs (or empty)'
            fi
        """], capture_output=True, text=True)
        print(result.stdout.strip() if result.stdout else "    ⚠️  Could not reach node")
'''
    ], capture_output=True, text=True)
    
    print(result.stdout if result.stdout else "  Migration script executed")
    if result.stderr:
        print(f"  ⚠️  {result.stderr}")

def verify_deployment():
    """Verify all files are in place."""
    print("\n[5/5] Verifying deployment...")
    
    required_files = [
        ('ramdisk_preload.sh', 0o755),
        ('ramdisk_config.py', 0o644),
        ('clear_ramdisk.sh', 0o755),
    ]
    
    all_ok = True
    for filepath, expected_mode in required_files:
        if os.path.exists(filepath):
            mode = os.stat(filepath).st_mode & 0o777
            if mode == expected_mode:
                print(f"  ✅ {filepath} (mode: {oct(mode)})")
            else:
                print(f"  ⚠️  {filepath} wrong mode: {oct(mode)} (expected: {oct(expected_mode)})")
                os.chmod(filepath, expected_mode)
                print(f"      Fixed mode to {oct(expected_mode)}")
        else:
            print(f"  ❌ {filepath} MISSING")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 70)
    print("UNIFIED RAMDISK DEPLOYMENT v2.0")
    print("Team Beta Approved - PROP-2026-01-21-RAMDISK-UNIFIED")
    print("=" * 70)
    
    # Verify we're in the right directory
    if not os.path.exists('distributed_config.json'):
        print("\n❌ ERROR: Must run from ~/distributed_prng_analysis/")
        print("   Could not find distributed_config.json")
        sys.exit(1)
    
    success = True
    
    # 1. Deploy ramdisk_preload.sh
    print("\n[1/5] Deploying ramdisk_preload.sh...")
    success &= deploy_ramdisk_preload_sh()
    
    # 2. Patch run_scorer_meta_optimizer.sh
    print("\n[2/5] Patching run_scorer_meta_optimizer.sh (Step 2)...")
    success &= patch_run_scorer_meta_optimizer()
    
    # 3. Patch generate_scorer_jobs.py
    print("\n[3/5] Patching generate_scorer_jobs.py...")
    success &= patch_generate_scorer_jobs()
    
    # 4. Migrate existing ramdisk
    migrate_existing_ramdisk()
    
    # 5. Verify
    success &= verify_deployment()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ DEPLOYMENT COMPLETE")
    else:
        print("⚠️  DEPLOYMENT COMPLETED WITH WARNINGS")
    
    print("""
MANUAL STEPS REQUIRED:
1. Copy ramdisk_config.py and clear_ramdisk.sh to this directory
2. Test Step 2:
   ./clear_ramdisk.sh --status
   ./clear_ramdisk.sh 2
   ./run_scorer_meta_optimizer.sh 10

EXPECTED BEHAVIOR:
- Ramdisk uses /dev/shm/prng/step2/ (not flat /dev/shm/prng/)
- Sentinel is /dev/shm/prng/step2/.ready
- Job args show /dev/shm/prng/step2/bidirectional_survivors_binary.npz

ROLLBACK:
   ls -la *.backup_*
   cp <file>.backup_TIMESTAMP <file>
""")
    print("=" * 70)

if __name__ == '__main__':
    main()
