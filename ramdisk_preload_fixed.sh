#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# Ramdisk Preload Script (Fixed)
# ════════════════════════════════════════════════════════════════════════════════
# Version: 1.1.0
# Date: January 25, 2026
#
# FIX: The v1.0 script checked for .ready marker but didn't verify files exist.
#      This version verifies actual files exist before skipping.
#
# Usage:
#   bash ramdisk_preload_fixed.sh [step]
#   bash ramdisk_preload_fixed.sh 3       # Preload Step 3 files
#
# Files for Step 3:
#   - train_history.json
#   - holdout_history.json
#
# Nodes:
#   - localhost
#   - 192.168.3.120 (rig-6600)
#   - 192.168.3.154 (rig-6600b)
#   - 192.168.3.162 (rig-6600c)
#
# ════════════════════════════════════════════════════════════════════════════════

set -euo pipefail

STEP="${1:-3}"
RAMDISK_BASE="/dev/shm/prng"
STEP_DIR="$RAMDISK_BASE/step$STEP"

# Files required for each step
declare -A STEP_FILES
STEP_FILES[3]="train_history.json holdout_history.json"

# All nodes (localhost handled separately)
REMOTE_NODES="192.168.3.120 192.168.3.154"

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

error() {
    echo "[$(date '+%H:%M:%S')] ERROR: $1" >&2
}

# ────────────────────────────────────────────────────────────────────────────────
# Verify files exist on a node
# Returns 0 if all files exist, 1 otherwise
# ────────────────────────────────────────────────────────────────────────────────
verify_files() {
    local node="$1"
    local dir="$2"
    local files="$3"
    local all_exist=0
    
    for file in $files; do
        if [[ "$node" == "localhost" ]]; then
            if [[ ! -f "$dir/$file" ]]; then
                log "  Missing on localhost: $dir/$file"
                all_exist=1
            fi
        else
            if ! ssh -o ConnectTimeout=5 "$node" "test -f $dir/$file" 2>/dev/null; then
                log "  Missing on $node: $dir/$file"
                all_exist=1
            fi
        fi
    done
    
    return $all_exist
}

# ────────────────────────────────────────────────────────────────────────────────
# Preload files to a node
# ────────────────────────────────────────────────────────────────────────────────
preload_to_node() {
    local node="$1"
    local dir="$2"
    local files="$3"
    
    if [[ "$node" == "localhost" ]]; then
        mkdir -p "$dir"
        for file in $files; do
            if [[ -f "$file" ]]; then
                cp "$file" "$dir/"
                log "  Copied $file to $dir/"
            else
                error "Source file not found: $file"
                return 1
            fi
        done
    else
        ssh -o ConnectTimeout=5 "$node" "mkdir -p $dir"
        for file in $files; do
            if [[ -f "$file" ]]; then
                scp -o ConnectTimeout=5 "$file" "$node:$dir/" >/dev/null
                log "  Copied $file to $node:$dir/"
            else
                error "Source file not found: $file"
                return 1
            fi
        done
    fi
}

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

log "════════════════════════════════════════════════════════════════════════════════"
log "Ramdisk Preload - Step $STEP (Fixed Version)"
log "════════════════════════════════════════════════════════════════════════════════"

FILES="${STEP_FILES[$STEP]:-}"
if [[ -z "$FILES" ]]; then
    error "No files defined for Step $STEP"
    exit 1
fi

log "Files to preload: $FILES"
log "Target directory: $STEP_DIR"
log ""

# ────────────────────────────────────────────────────────────────────────────────
# Process localhost
# ────────────────────────────────────────────────────────────────────────────────
log "--- localhost ---"
if verify_files "localhost" "$STEP_DIR" "$FILES"; then
    log "  ✅ All files present (verified)"
else
    log "  Populating ramdisk..."
    if preload_to_node "localhost" "$STEP_DIR" "$FILES"; then
        # Create .ready marker AFTER files are verified
        touch "$STEP_DIR/.ready"
        log "  ✅ Preload complete"
    else
        error "Failed to preload localhost"
        exit 1
    fi
fi

# ────────────────────────────────────────────────────────────────────────────────
# Process remote nodes
# ────────────────────────────────────────────────────────────────────────────────
for node in $REMOTE_NODES; do
    log ""
    log "--- $node ---"
    
    # Check SSH connectivity first
    if ! ssh -o ConnectTimeout=5 "$node" "echo OK" >/dev/null 2>&1; then
        error "$node unreachable via SSH"
        continue
    fi
    
    # FIX: Check actual files, not just .ready marker
    if verify_files "$node" "$STEP_DIR" "$FILES"; then
        log "  ✅ All files present (verified)"
    else
        log "  Populating ramdisk..."
        if preload_to_node "$node" "$STEP_DIR" "$FILES"; then
            ssh "$node" "touch $STEP_DIR/.ready"
            log "  ✅ Preload complete"
        else
            error "Failed to preload $node"
            # Continue to other nodes, don't exit
        fi
    fi
done

log ""
log "════════════════════════════════════════════════════════════════════════════════"
log "Ramdisk preload complete"
log "════════════════════════════════════════════════════════════════════════════════"

# ────────────────────────────────────────────────────────────────────────────────
# Final verification summary
# ────────────────────────────────────────────────────────────────────────────────
log ""
log "Final verification:"
for node in localhost $REMOTE_NODES; do
    if verify_files "$node" "$STEP_DIR" "$FILES" 2>/dev/null; then
        log "  ✅ $node: All files present"
    else
        log "  ❌ $node: Missing files"
    fi
done
