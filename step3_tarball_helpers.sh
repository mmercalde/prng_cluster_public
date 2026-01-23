#!/bin/bash
# =============================================================================
# Step 3 Tarball Distribution Helper
# =============================================================================
# Team Beta Mandatory Action - 2026-01-22
#
# Purpose: Eliminate small-file SCP storms during Step 3 chunk distribution
#
# Instead of:
#   scp scoring_chunks/*.json user@host:path/   # 100s of individual transfers
#
# Use:
#   tar + scp + extract                          # Single transfer
#
# Usage:
#   source step3_tarball_helpers.sh
#   distribute_chunks_tarball "scoring_chunks" "rig-6600" "/home/michael/distributed_prng_analysis/scoring_chunks"
#
# =============================================================================

set -e

# Configuration
TARBALL_NAME="step3_chunks.tar.gz"
REMOTE_USER="${REMOTE_USER:-michael}"

# =============================================================================
# FUNCTIONS
# =============================================================================

# Create tarball from chunk directory
create_chunks_tarball() {
    local chunk_dir="$1"
    local tarball="${2:-$TARBALL_NAME}"
    
    if [[ ! -d "$chunk_dir" ]]; then
        echo "❌ ERROR: Chunk directory not found: $chunk_dir"
        return 1
    fi
    
    local file_count=$(find "$chunk_dir" -name "*.json" -o -name "*.npz" 2>/dev/null | wc -l)
    
    if [[ "$file_count" -eq 0 ]]; then
        echo "❌ ERROR: No chunk files found in $chunk_dir"
        return 1
    fi
    
    echo "📦 Creating tarball from $chunk_dir ($file_count files)..."
    
    # Create tarball (preserving directory structure)
    tar -czf "$tarball" -C "$(dirname "$chunk_dir")" "$(basename "$chunk_dir")"
    
    local size=$(du -h "$tarball" | cut -f1)
    echo "✅ Created: $tarball ($size)"
    
    return 0
}

# Transfer tarball to remote node
transfer_tarball() {
    local tarball="$1"
    local remote_host="$2"
    local remote_path="$3"
    
    if [[ ! -f "$tarball" ]]; then
        echo "❌ ERROR: Tarball not found: $tarball"
        return 1
    fi
    
    echo "📤 Transferring $tarball to $remote_host..."
    
    # Ensure remote directory exists
    ssh "${REMOTE_USER}@${remote_host}" "mkdir -p $remote_path"
    
    # Transfer with compression already done
    scp "$tarball" "${REMOTE_USER}@${remote_host}:${remote_path}/"
    
    echo "✅ Transferred to $remote_host:$remote_path"
    return 0
}

# Extract tarball on remote node
extract_tarball_remote() {
    local remote_host="$1"
    local remote_path="$2"
    local tarball="${3:-$TARBALL_NAME}"
    
    echo "📂 Extracting on $remote_host..."
    
    ssh "${REMOTE_USER}@${remote_host}" "cd $remote_path && tar -xzf $tarball && rm -f $tarball"
    
    echo "✅ Extracted and cleaned up on $remote_host"
    return 0
}

# All-in-one: Create, transfer, extract
distribute_chunks_tarball() {
    local chunk_dir="$1"
    local remote_host="$2"
    local remote_path="$3"
    local tarball="${4:-$TARBALL_NAME}"
    
    echo ""
    echo "========================================"
    echo "Step 3 Tarball Distribution"
    echo "========================================"
    echo "Source: $chunk_dir"
    echo "Target: $remote_host:$remote_path"
    echo ""
    
    # Create
    create_chunks_tarball "$chunk_dir" "$tarball" || return 1
    
    # Transfer
    transfer_tarball "$tarball" "$remote_host" "$remote_path" || return 1
    
    # Extract
    extract_tarball_remote "$remote_host" "$remote_path" "$tarball" || return 1
    
    # Cleanup local tarball
    rm -f "$tarball"
    
    echo ""
    echo "✅ Distribution complete: $chunk_dir → $remote_host"
    return 0
}

# Distribute to all remote rigs
distribute_chunks_all_rigs() {
    local chunk_dir="$1"
    local remote_path="${2:-/home/michael/distributed_prng_analysis/scoring_chunks}"
    
    local rigs=("192.168.3.120" "192.168.3.154")
    local rig_names=("rig-6600" "rig-6600b")
    
    echo ""
    echo "========================================"
    echo "Step 3 Tarball Distribution - All Rigs"
    echo "========================================"
    
    # Create tarball once
    create_chunks_tarball "$chunk_dir" "$TARBALL_NAME" || return 1
    
    local success_count=0
    for i in "${!rigs[@]}"; do
        local host="${rigs[$i]}"
        local name="${rig_names[$i]}"
        
        echo ""
        echo "--- $name ($host) ---"
        
        if transfer_tarball "$TARBALL_NAME" "$host" "$remote_path" && \
           extract_tarball_remote "$host" "$remote_path" "$TARBALL_NAME"; then
            ((success_count++))
        else
            echo "⚠️  Failed on $name"
        fi
    done
    
    # Cleanup local tarball
    rm -f "$TARBALL_NAME"
    
    echo ""
    echo "========================================"
    echo "Distribution Summary: $success_count/${#rigs[@]} rigs successful"
    echo "========================================"
    
    [[ "$success_count" -eq "${#rigs[@]}" ]]
}

# Verify chunks exist on remote
verify_chunks_remote() {
    local remote_host="$1"
    local remote_path="$2"
    
    echo "🔍 Verifying chunks on $remote_host..."
    
    local count=$(ssh "${REMOTE_USER}@${remote_host}" "find $remote_path -name '*.json' -o -name '*.npz' 2>/dev/null | wc -l")
    
    echo "   Found $count chunk files"
    
    [[ "$count" -gt 0 ]]
}

# =============================================================================
# USAGE EXAMPLES (when sourced)
# =============================================================================

show_usage() {
    cat << 'EOF'

Step 3 Tarball Distribution Helper
==================================

Usage (source this file first):
  source step3_tarball_helpers.sh

Functions available:
  distribute_chunks_tarball <chunk_dir> <remote_host> <remote_path>
    - One-shot: create, transfer, extract to single node

  distribute_chunks_all_rigs <chunk_dir> [remote_path]
    - Distribute to both rig-6600 and rig-6600b

  create_chunks_tarball <chunk_dir> [tarball_name]
    - Create tarball only

  verify_chunks_remote <remote_host> <remote_path>
    - Verify chunks exist on remote

Examples:
  # Distribute to single rig
  distribute_chunks_tarball scoring_chunks 192.168.3.120 ~/distributed_prng_analysis/scoring_chunks

  # Distribute to all rigs
  distribute_chunks_all_rigs scoring_chunks

  # Verify
  verify_chunks_remote 192.168.3.120 ~/distributed_prng_analysis/scoring_chunks

EOF
}

# If run directly (not sourced), show usage
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    show_usage
fi
