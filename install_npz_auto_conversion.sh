#!/bin/bash
# =============================================================================
# SELF-INSTALLING PATCH: NPZ Auto-Conversion for Step 2.5
# =============================================================================
# Date: January 19, 2026
# Status: Team Beta APPROVED
# 
# USAGE:
#   chmod +x install_npz_auto_conversion.sh
#   ./install_npz_auto_conversion.sh
#
# WHAT IT DOES:
#   1. Backs up existing files with timestamp
#   2. Patches convert_survivors_to_binary.py (adds --output flag)
#   3. Patches run_scorer_meta_optimizer.sh (adds auto-conversion block)
#   4. Validates patches applied correctly
#   5. Reports success/failure
#
# ROLLBACK:
#   ./install_npz_auto_conversion.sh --rollback
# =============================================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${SCRIPT_DIR}/backups/npz_patch_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# ROLLBACK FUNCTION
# =============================================================================
rollback() {
    echo "============================================"
    echo "NPZ Patch Rollback"
    echo "============================================"
    
    # Find most recent backup
    local latest_backup=$(ls -td "${SCRIPT_DIR}/backups/npz_patch_"* 2>/dev/null | head -1)
    
    if [ -z "$latest_backup" ]; then
        log_error "No backup found to rollback!"
        exit 1
    fi
    
    log_info "Rolling back from: $latest_backup"
    
    if [ -f "${latest_backup}/convert_survivors_to_binary.py" ]; then
        cp "${latest_backup}/convert_survivors_to_binary.py" "${SCRIPT_DIR}/"
        log_info "Restored convert_survivors_to_binary.py"
    fi
    
    if [ -f "${latest_backup}/run_scorer_meta_optimizer.sh" ]; then
        cp "${latest_backup}/run_scorer_meta_optimizer.sh" "${SCRIPT_DIR}/"
        log_info "Restored run_scorer_meta_optimizer.sh"
    fi
    
    log_info "Rollback complete!"
    exit 0
}

# Check for rollback flag
if [ "${1:-}" = "--rollback" ]; then
    rollback
fi

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================
echo "============================================"
echo "NPZ Auto-Conversion Patch Installer"
echo "============================================"
echo ""

log_info "Running pre-flight checks..."

# Check we're in the right directory
if [ ! -f "distributed_config.json" ]; then
    log_error "distributed_config.json not found!"
    log_error "Run this script from ~/distributed_prng_analysis/"
    exit 1
fi

# Check target files exist
if [ ! -f "convert_survivors_to_binary.py" ]; then
    log_warn "convert_survivors_to_binary.py not found - will create new"
fi

if [ ! -f "run_scorer_meta_optimizer.sh" ]; then
    log_error "run_scorer_meta_optimizer.sh not found!"
    exit 1
fi

log_info "Pre-flight checks passed ✓"

# =============================================================================
# CREATE BACKUPS
# =============================================================================
log_info "Creating backups in $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

[ -f "convert_survivors_to_binary.py" ] && cp "convert_survivors_to_binary.py" "$BACKUP_DIR/"
cp "run_scorer_meta_optimizer.sh" "$BACKUP_DIR/"

log_info "Backups created ✓"

# =============================================================================
# PATCH 1: convert_survivors_to_binary.py (FULL REPLACEMENT)
# =============================================================================
log_info "Patching convert_survivors_to_binary.py..."

cat > convert_survivors_to_binary.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
convert_survivors_to_binary.py - Convert JSON survivors to NPZ binary format

Performance: 88x faster loading (4.2s → 0.05s), 400x smaller (258MB → 0.6MB)

Usage:
    python3 convert_survivors_to_binary.py bidirectional_survivors.json
    python3 convert_survivors_to_binary.py bidirectional_survivors.json --output /tmp/survivors.npz

Version: 2.0.0 (January 19, 2026)
  - Added --output flag for atomic write support
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime


def convert_json_to_npz(input_file: str, output_file: str, meta_file: str) -> dict:
    """
    Convert JSON survivors to compressed NPZ format.
    
    Returns metadata dict for verification.
    """
    print(f"Loading {input_file}...")
    with open(input_file) as f:
        survivors = json.load(f)
    
    print(f"Loaded {len(survivors):,} survivors")
    
    # Extract arrays
    seeds = np.array([s['seed'] for s in survivors], dtype=np.uint32)
    
    # Handle optional fields with defaults
    forward_matches = np.array([
        s.get('forward_count', s.get('score', 0)) 
        for s in survivors
    ], dtype=np.float32)
    
    reverse_matches = np.array([
        s.get('reverse_count', s.get('score', 0)) 
        for s in survivors
    ], dtype=np.float32)
    
    # Save compressed NPZ
    print(f"Saving {output_file}...")
    np.savez_compressed(
        output_file,
        seeds=seeds,
        forward_matches=forward_matches,
        reverse_matches=reverse_matches
    )
    
    # Metadata for verification
    metadata = {
        "source_file": str(Path(input_file).resolve()),
        "output_file": str(Path(output_file).resolve()),
        "survivor_count": len(survivors),
        "seed_dtype": str(seeds.dtype),
        "seed_min": int(seeds.min()),
        "seed_max": int(seeds.max()),
        "converted_at": datetime.now().isoformat(),
        "format_version": "1.0"
    }
    
    # Save metadata
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Report sizes
    input_size = Path(input_file).stat().st_size
    output_size = Path(output_file).stat().st_size
    ratio = input_size / output_size if output_size > 0 else 0
    
    print(f"✓ Conversion complete:")
    print(f"  Input:  {input_size:,} bytes ({input_size/1024/1024:.1f} MB)")
    print(f"  Output: {output_size:,} bytes ({output_size/1024:.1f} KB)")
    print(f"  Ratio:  {ratio:.0f}x compression")
    print(f"  Meta:   {meta_file}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Convert bidirectional_survivors.json to NPZ binary format'
    )
    parser.add_argument('input_file', 
                        help='Input JSON file (e.g., bidirectional_survivors.json)')
    parser.add_argument('--output', '-o',
                        help='Output NPZ file path (default: derived from input)')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    
    # Verify input exists
    if not Path(input_file).exists():
        print(f"ERROR: Input file not found: {input_file}")
        return 1
    
    # Derive output paths
    if args.output:
        output_file = args.output
        # Meta file alongside output
        meta_file = str(Path(args.output).with_suffix('.meta.json'))
    else:
        # Default: derive from input name
        base = input_file.replace('.json', '')
        output_file = f"{base}_binary.npz"
        meta_file = f"{base}_binary.meta.json"
    
    try:
        convert_json_to_npz(input_file, output_file, meta_file)
        return 0
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
PYTHON_SCRIPT

chmod +x convert_survivors_to_binary.py
log_info "convert_survivors_to_binary.py patched ✓"

# =============================================================================
# PATCH 2: run_scorer_meta_optimizer.sh (INSERT BLOCK AFTER SHEBANG)
# =============================================================================
log_info "Patching run_scorer_meta_optimizer.sh..."

# Create the NPZ auto-conversion block
NPZ_BLOCK='
# =============================================================================
# NPZ AUTO-CONVERSION BLOCK (Inserted by install_npz_auto_conversion.sh)
# Date: '"${TIMESTAMP}"'
# Team Beta Approved: January 19, 2026
# =============================================================================

set -euo pipefail  # Fail-fast, no undefined vars, pipe failures

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SURVIVORS="bidirectional_survivors_binary.npz"
JSON_SOURCE="bidirectional_survivors.json"
TMP_NPZ="${SURVIVORS}.tmp.$$"
CONFIG_FILE="distributed_config.json"
REMOTE_DIR="distributed_prng_analysis"

trap '"'"'rm -f "$TMP_NPZ"'"'"' EXIT

# -----------------------------------------------------------------------------
# Extract remote nodes from distributed_config.json (no hardcoding)
# -----------------------------------------------------------------------------
get_remote_nodes() {
    python3 << '"'"'PYEOF'"'"'
import json
import sys

CONFIG_FILE = "distributed_config.json"

try:
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    
    nodes = config.get("nodes", [])
    for node in nodes:
        hostname = node.get("hostname", "")
        if hostname and hostname != "localhost" and not hostname.startswith("127."):
            print(hostname)
except FileNotFoundError:
    print(f"ERROR: {CONFIG_FILE} not found", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON in {CONFIG_FILE}: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# -----------------------------------------------------------------------------
# NPZ Conversion Function (atomic write)
# -----------------------------------------------------------------------------
convert_to_npz() {
    echo "============================================"
    echo "NPZ Conversion Required"
    echo "============================================"
    
    if [ ! -f "$JSON_SOURCE" ]; then
        echo "ERROR: $JSON_SOURCE not found!"
        echo "Run Step 1 (window_optimizer.py) first."
        exit 1
    fi
    
    echo "Converting $JSON_SOURCE → $SURVIVORS (atomic)..."
    rm -f "$TMP_NPZ"
    
    if ! python3 convert_survivors_to_binary.py "$JSON_SOURCE" --output "$TMP_NPZ"; then
        echo "ERROR: NPZ conversion failed!"
        exit 1
    fi
    
    if [ ! -s "$TMP_NPZ" ]; then
        echo "ERROR: Conversion produced empty or missing file!"
        exit 1
    fi
    
    mv "$TMP_NPZ" "$SURVIVORS"
    echo "✓ Conversion complete: $SURVIVORS"
}

# -----------------------------------------------------------------------------
# Distribute NPZ to Remote Nodes
# -----------------------------------------------------------------------------
distribute_npz() {
    echo ""
    echo "Distributing NPZ to remote nodes..."
    
    local nodes
    nodes=$(get_remote_nodes)
    
    if [ -z "$nodes" ]; then
        echo "WARNING: No remote nodes found in $CONFIG_FILE"
        echo "Skipping distribution (localhost-only mode)."
        return 0
    fi
    
    local failed=0
    for node in $nodes; do
        echo -n "  → $node: "
        if scp -q "$SURVIVORS" "${node}:~/${REMOTE_DIR}/" 2>/dev/null; then
            echo "✓"
        else
            echo "✗ FAILED"
            failed=1
        fi
    done
    
    if [ $failed -ne 0 ]; then
        echo "ERROR: Distribution failed to one or more nodes!"
        echo "Cluster is in inconsistent state. Aborting."
        exit 1
    fi
    
    echo "✓ Distribution complete to all nodes."
}

# -----------------------------------------------------------------------------
# Main: Auto-Convert if Needed
# -----------------------------------------------------------------------------
if [ ! -f "$SURVIVORS" ]; then
    echo "NPZ file missing - conversion required."
    convert_to_npz
    distribute_npz
elif [ "$JSON_SOURCE" -nt "$SURVIVORS" ]; then
    echo "JSON newer than NPZ - reconversion required."
    convert_to_npz
    distribute_npz
else
    echo "NPZ file up-to-date, skipping conversion."
fi

echo "============================================"
echo ""
# =============================================================================
# END NPZ AUTO-CONVERSION BLOCK
# =============================================================================
'

# Check if patch already applied
if grep -q "NPZ AUTO-CONVERSION BLOCK" run_scorer_meta_optimizer.sh; then
    log_warn "NPZ block already present in run_scorer_meta_optimizer.sh"
    log_warn "Skipping to avoid duplicate insertion"
else
    # Extract shebang line
    SHEBANG=$(head -1 run_scorer_meta_optimizer.sh)
    
    # Extract rest of file (everything after line 1)
    REMAINING=$(tail -n +2 run_scorer_meta_optimizer.sh)
    
    # Reconstruct file: shebang + NPZ block + original content
    {
        echo "$SHEBANG"
        echo "$NPZ_BLOCK"
        echo "$REMAINING"
    } > run_scorer_meta_optimizer.sh.new
    
    mv run_scorer_meta_optimizer.sh.new run_scorer_meta_optimizer.sh
    chmod +x run_scorer_meta_optimizer.sh
    
    log_info "run_scorer_meta_optimizer.sh patched ✓"
fi

# =============================================================================
# VALIDATION
# =============================================================================
echo ""
log_info "Running validation checks..."

# Check 1: Python script has --output
if python3 convert_survivors_to_binary.py --help 2>&1 | grep -q "\-\-output"; then
    log_info "  ✓ convert_survivors_to_binary.py has --output flag"
else
    log_error "  ✗ --output flag not found in Python script!"
    exit 1
fi

# Check 2: Bash script has NPZ block
if grep -q "NPZ AUTO-CONVERSION BLOCK" run_scorer_meta_optimizer.sh; then
    log_info "  ✓ run_scorer_meta_optimizer.sh has NPZ block"
else
    log_error "  ✗ NPZ block not found in bash script!"
    exit 1
fi

# Check 3: Bash script has set -euo pipefail
if grep -q "set -euo pipefail" run_scorer_meta_optimizer.sh; then
    log_info "  ✓ run_scorer_meta_optimizer.sh has fail-fast enabled"
else
    log_error "  ✗ fail-fast not enabled!"
    exit 1
fi

# Check 4: No hardcoded IPs in NPZ block
if grep -A100 "NPZ AUTO-CONVERSION BLOCK" run_scorer_meta_optimizer.sh | grep -q "192.168"; then
    log_error "  ✗ Hardcoded IP addresses found!"
    exit 1
else
    log_info "  ✓ No hardcoded IP addresses"
fi

# =============================================================================
# SUCCESS
# =============================================================================
echo ""
echo "============================================"
log_info "PATCH INSTALLATION COMPLETE ✓"
echo "============================================"
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""
echo "To rollback:  ./install_npz_auto_conversion.sh --rollback"
echo ""
echo "Next steps:"
echo "  1. Test: ./run_scorer_meta_optimizer.sh --dry-run (if supported)"
echo "  2. Or run full pipeline: PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 2 --end-step 6"
echo ""
