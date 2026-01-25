#!/bin/bash
# ============================================================================
# ramdisk_preload.sh - Unified Ramdisk Infrastructure for Distributed Steps
# ============================================================================
# Version: 2.1.0 (Team Beta Approved)
# 
# Changes from v2.0.0:
#   - FIX: Only create .ready sentinel if files were actually copied
#   - FIX: Fail loudly if source files don't exist
#   - ADD: Count verification before marking complete
#
# Usage:
#   export RAMDISK_STEP_ID=3
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
if [ -z "${_RAMDISK_NODES_CACHED:-}" ]; then
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
        usage=$(ssh "$node" "df /dev/shm 2>/dev/null | awk 'NR==2 {gsub(/%/,\"\"); print \$5}'" 2>/dev/null)
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

# B2: Verify required files actually exist (not just sentinel)
verify_ramdisk_files() {
    local node="$1"
    shift
    local files=("$@")
    for f in "${files[@]}"; do
        local bname=$(basename "$f")
        if [ "$node" = "localhost" ]; then
            [ ! -f "$RAMDISK_DIR/$bname" ] && return 1
        else
            ssh "$node" "[ ! -f $RAMDISK_DIR/$bname ]" 2>/dev/null && return 1
        fi
    done
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
    
    # v2.1.0: Verify source files exist before attempting any copies
    local missing_sources=0
    for f in "${files[@]}"; do
        if [ ! -f "$f" ]; then
            echo "[ERROR] Source file not found: $f"
            missing_sources=$((missing_sources+1))
        fi
    done
    if [ $missing_sources -gt 0 ]; then
        echo "[ERROR] Cannot proceed - $missing_sources source file(s) missing"
        return 1
    fi
    
    local expected_count=${#files[@]}
    
    for NODE in "${CLUSTER_NODES[@]}"; do
        echo "  → $NODE"
        check_ramdisk_headroom "$NODE" || continue
        
        if [ "$NODE" = "localhost" ]; then
            mkdir -p "$RAMDISK_DIR"
            if [ -f "$RAMDISK_SENTINEL" ] && verify_ramdisk_files "localhost" "${files[@]}"; then
                echo "    ✓ Already loaded (verified)"
                continue
            fi
            [ -z "$watcher_managed" ] && rm -rf "${RAMDISK_DIR:?}"/* 2>/dev/null || true
            local copied=0
            for f in "${files[@]}"; do
                if [ -f "$f" ] && cp "$f" "$RAMDISK_DIR/"; then
                    copied=$((copied+1))
                fi
            done
            # v2.1.0 FIX: Only create .ready if ALL files were copied
            if [ $copied -eq $expected_count ]; then
                touch "$RAMDISK_SENTINEL"
                echo "    ✓ Preloaded ($copied files)"
            else
                echo "    ❌ INCOMPLETE: Only $copied/$expected_count files copied"
                rm -f "$RAMDISK_SENTINEL" 2>/dev/null  # Remove stale sentinel
                return 1
            fi
        else
            ssh "$NODE" "mkdir -p $RAMDISK_DIR" 2>/dev/null
            if ssh "$NODE" "[ -f $RAMDISK_SENTINEL ]" 2>/dev/null && verify_ramdisk_files "$NODE" "${files[@]}"; then
                echo "    ✓ Already loaded (verified)"
                continue
            fi
            [ -z "$watcher_managed" ] && ssh "$NODE" "rm -rf ${RAMDISK_DIR:?}/*" 2>/dev/null || true
            local copied=0
            for f in "${files[@]}"; do
                if [ -f "$f" ] && scp -q "$f" "$NODE:$RAMDISK_DIR/" 2>/dev/null; then
                    copied=$((copied+1))
                fi
            done
            # v2.1.0 FIX: Only create .ready if ALL files were copied
            if [ $copied -eq $expected_count ]; then
                ssh "$NODE" "touch $RAMDISK_SENTINEL" 2>/dev/null
                echo "    ✓ Preloaded ($copied files)"
            else
                echo "    ❌ INCOMPLETE: Only $copied/$expected_count files copied to $NODE"
                ssh "$NODE" "rm -f $RAMDISK_SENTINEL" 2>/dev/null  # Remove stale sentinel
                # Don't return 1 here - continue to other nodes (A3: non-fatal for remotes)
            fi
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
