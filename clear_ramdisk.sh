#!/bin/bash
# ============================================================================
# clear_ramdisk.sh - Clear Ramdisk Data on All Cluster Nodes
# ============================================================================
# Version: 2.0.0 (Team Beta Approved - PROP-2026-01-21-RAMDISK-UNIFIED)
#
# Usage:
#   ./clear_ramdisk.sh           # Clear ALL steps on all nodes
#   ./clear_ramdisk.sh 2         # Clear Step 2 only
#   ./clear_ramdisk.sh 3         # Clear Step 3 only
#   ./clear_ramdisk.sh 5         # Clear Step 5 only
#   ./clear_ramdisk.sh --status  # Show ramdisk status
#   ./clear_ramdisk.sh --migrate # Migrate flat structure to per-step dirs
#
# Team Beta Requirements:
#   A1: Per-step subdirectories (/dev/shm/prng/step2/, step3/, step5/)
#   A3: Non-fatal cleanup (errors logged but don't halt)
# ============================================================================

set -e

RAMDISK_BASE="/dev/shm/prng"

# ============================================================================
# Get cluster nodes (cached for efficiency - A2)
# ============================================================================
get_cluster_nodes() {
    python3 -c "
import json, sys
try:
    with open('distributed_config.json') as f:
        cfg = json.load(f)
    for node in cfg['nodes']:
        print(node['hostname'])
except Exception as e:
    print('localhost')
" 2>/dev/null
}

# Cache nodes once
CLUSTER_NODES=($(get_cluster_nodes))

# ============================================================================
# Show ramdisk status
# ============================================================================
show_status() {
    echo "============================================"
    echo "RAMDISK STATUS"
    echo "============================================"
    
    for NODE in "${CLUSTER_NODES[@]}"; do
        echo ""
        echo "→ $NODE"
        
        if [ "$NODE" = "localhost" ]; then
            echo "  Usage: $(df -h /dev/shm 2>/dev/null | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}')"
            echo "  Steps loaded:"
            for step_dir in ${RAMDISK_BASE}/step*/; do
                if [ -d "$step_dir" ]; then
                    step_name=$(basename "$step_dir")
                    file_count=$(ls -1 "$step_dir" 2>/dev/null | grep -v "^\.ready$" | wc -l)
                    has_sentinel=$([ -f "${step_dir}/.ready" ] && echo "✓" || echo "✗")
                    size=$(du -sh "$step_dir" 2>/dev/null | cut -f1)
                    echo "    ${step_name}: ${file_count} files (${size}), ready: ${has_sentinel}"
                fi
            done
            if [ ! -d "${RAMDISK_BASE}/step2" ] && [ ! -d "${RAMDISK_BASE}/step3" ] && [ ! -d "${RAMDISK_BASE}/step5" ]; then
                # Check for old flat structure
                if [ -f "${RAMDISK_BASE}/.ready" ]; then
                    echo "    ⚠️  OLD FLAT STRUCTURE DETECTED - run: ./clear_ramdisk.sh --migrate"
                else
                    echo "    (none)"
                fi
            fi
        else
            ssh "$NODE" "
                echo \"  Usage: \$(df -h /dev/shm 2>/dev/null | awk 'NR==2 {print \$3 \"/\" \$2 \" (\" \$5 \")\"}')\"
                echo '  Steps loaded:'
                found=0
                for step_dir in ${RAMDISK_BASE}/step*/; do
                    if [ -d \"\$step_dir\" ]; then
                        found=1
                        step_name=\$(basename \"\$step_dir\")
                        file_count=\$(ls -1 \"\$step_dir\" 2>/dev/null | grep -v '^\.ready$' | wc -l)
                        has_sentinel=\$([ -f \"\${step_dir}/.ready\" ] && echo '✓' || echo '✗')
                        size=\$(du -sh \"\$step_dir\" 2>/dev/null | cut -f1)
                        echo \"    \${step_name}: \${file_count} files (\${size}), ready: \${has_sentinel}\"
                    fi
                done
                if [ \$found -eq 0 ]; then
                    if [ -f '${RAMDISK_BASE}/.ready' ]; then
                        echo '    ⚠️  OLD FLAT STRUCTURE DETECTED - run: ./clear_ramdisk.sh --migrate'
                    else
                        echo '    (none)'
                    fi
                fi
            " 2>/dev/null || echo "  ⚠️  Could not reach node"
        fi
    done
    
    echo ""
    echo "============================================"
}

# ============================================================================
# Migrate from flat to per-step directories
# ============================================================================
migrate_to_step_dirs() {
    echo "============================================"
    echo "MIGRATING TO PER-STEP DIRECTORIES"
    echo "============================================"
    
    for NODE in "${CLUSTER_NODES[@]}"; do
        echo "→ $NODE"
        
        if [ "$NODE" = "localhost" ]; then
            if [ -f "${RAMDISK_BASE}/.ready" ] && [ ! -d "${RAMDISK_BASE}/step2" ]; then
                echo "  Migrating flat structure to step2/..."
                mkdir -p "${RAMDISK_BASE}/step2"
                # Move files (not directories) to step2
                for f in ${RAMDISK_BASE}/*; do
                    if [ -f "$f" ]; then
                        mv "$f" "${RAMDISK_BASE}/step2/"
                    fi
                done
                echo "  ✓ Migrated to ${RAMDISK_BASE}/step2/"
            else
                echo "  ✓ Already using per-step directories (or empty)"
            fi
        else
            ssh "$NODE" "
                if [ -f '${RAMDISK_BASE}/.ready' ] && [ ! -d '${RAMDISK_BASE}/step2' ]; then
                    echo '  Migrating flat structure to step2/...'
                    mkdir -p '${RAMDISK_BASE}/step2'
                    for f in ${RAMDISK_BASE}/*; do
                        if [ -f \"\$f\" ]; then
                            mv \"\$f\" '${RAMDISK_BASE}/step2/'
                        fi
                    done
                    echo '  ✓ Migrated'
                else
                    echo '  ✓ Already using per-step directories (or empty)'
                fi
            " 2>/dev/null || echo "  ⚠️  Could not reach node"
        fi
    done
    
    echo ""
    echo "Migration complete. Run './clear_ramdisk.sh --status' to verify."
}

# ============================================================================
# Clear ramdisk - A3: idempotent and non-fatal
# ============================================================================
clear_ramdisk() {
    local step_id="$1"
    
    if [ -z "$step_id" ]; then
        echo "============================================"
        echo "CLEARING ALL RAMDISK DATA"
        echo "============================================"
        target="${RAMDISK_BASE}/step*"
    else
        echo "============================================"
        echo "CLEARING RAMDISK FOR STEP ${step_id}"
        echo "============================================"
        target="${RAMDISK_BASE}/step${step_id}"
    fi
    
    for NODE in "${CLUSTER_NODES[@]}"; do
        echo "→ $NODE"
        
        if [ "$NODE" = "localhost" ]; then
            # A3: Non-fatal - use || true
            rm -rf $target 2>/dev/null || true
            echo "  ✓ Cleared"
        else
            # A3: Non-fatal - catch SSH failures
            if ssh "$NODE" "rm -rf $target" 2>/dev/null; then
                echo "  ✓ Cleared"
            else
                echo "  ⚠️  Warning: cleanup failed (continuing)"
            fi
        fi
    done
    
    echo ""
    echo "Ramdisk cleanup complete."
    
    # Also clear old flat structure if present
    if [ -z "$step_id" ]; then
        echo ""
        echo "Checking for old flat structure..."
        for NODE in "${CLUSTER_NODES[@]}"; do
            if [ "$NODE" = "localhost" ]; then
                if [ -f "${RAMDISK_BASE}/.ready" ]; then
                    rm -f "${RAMDISK_BASE}/.ready" "${RAMDISK_BASE}"/*.npz "${RAMDISK_BASE}"/*.json 2>/dev/null || true
                    echo "→ $NODE: Cleared old flat structure"
                fi
            else
                ssh "$NODE" "
                    if [ -f '${RAMDISK_BASE}/.ready' ]; then
                        rm -f '${RAMDISK_BASE}/.ready' '${RAMDISK_BASE}'/*.npz '${RAMDISK_BASE}'/*.json 2>/dev/null || true
                        echo '→ $NODE: Cleared old flat structure'
                    fi
                " 2>/dev/null || true
            fi
        done
    fi
}

# ============================================================================
# Main
# ============================================================================

# Parse arguments
case "${1:-}" in
    --status|-s)
        show_status
        ;;
    --migrate|-m)
        migrate_to_step_dirs
        ;;
    --help|-h)
        echo "Usage: $0 [STEP_ID | --status | --migrate | --help]"
        echo ""
        echo "Options:"
        echo "  (none)      Clear ALL steps on all nodes"
        echo "  STEP_ID     Clear specific step (2, 3, or 5)"
        echo "  --status    Show ramdisk status across cluster"
        echo "  --migrate   Migrate old flat structure to per-step dirs"
        echo "  --help      Show this help"
        ;;
    [0-9]*)
        clear_ramdisk "$1"
        ;;
    "")
        clear_ramdisk
        ;;
    *)
        echo "Unknown option: $1"
        echo "Run '$0 --help' for usage."
        exit 1
        ;;
esac
