#!/bin/bash
# apply_s107_run_scorer_v2_3.sh
# S107 patch for run_scorer_meta_optimizer.sh v2.3
#
# Fixes:
#   1. Wire --sample-size CLI arg so Watcher override propagates (default 5000)
#   2. Add 192.168.3.162 (rig-6600c) to scp push loop
#
# Note: removes orphaned timing comment lines that were attached to hardcoded 450

set -e
TARGET="run_scorer_meta_optimizer.sh"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="${TARGET}.s107_v2_3_backup_${TIMESTAMP}"

echo "=== S107 run_scorer_meta_optimizer.sh v2.3 patcher ==="

# Pre-flight checks
FAIL=0
grep -q 'USE_LEGACY_SCORING=false' "$TARGET"           || { echo "ABORT: USE_LEGACY_SCORING=false not found"; FAIL=1; }
grep -q 'TRIALS=""'                "$TARGET"           || { echo "ABORT: TRIALS=\"\" not found"; FAIL=1; }
grep -q -- '--sample-size 450'     "$TARGET"           || { echo "ABORT: --sample-size 450 not found"; FAIL=1; }
grep -q 'for node in 192.168.3.120 192.168.3.154;' "$TARGET" || { echo "ABORT: scp node list not found"; FAIL=1; }
grep -q '(vs 25000 default'        "$TARGET"           || { echo "ABORT: orphaned comment line not found"; FAIL=1; }

[ $FAIL -ne 0 ] && exit 1
echo "Pre-flight: all 5 targets found"

# Backup
cp "$TARGET" "$BACKUP"
echo "Backup: $BACKUP"

# Patch 1: Add SAMPLE_SIZE=5000 default after USE_LEGACY_SCORING=false
sed -i 's/USE_LEGACY_SCORING=false/USE_LEGACY_SCORING=false\nSAMPLE_SIZE=5000/' "$TARGET"
echo "Patch 1/5: added SAMPLE_SIZE=5000 default"

# Patch 2: Add --sample-size argument parser before --legacy-scoring case
sed -i 's/        --legacy-scoring)/        --sample-size)\n            SAMPLE_SIZE=$2\n            shift 2\n            ;;\n        --legacy-scoring)/' "$TARGET"
echo "Patch 2/5: added --sample-size argument parser"

# Patch 3: Replace --sample-size 450 + comment with --sample-size $SAMPLE_SIZE (both branches)
sed -i 's/        --sample-size 450    # TUNED 2026-01-17: 5000 seeds = ~60-90s trials/        --sample-size $SAMPLE_SIZE/' "$TARGET"
echo "Patch 3/5: replaced hardcoded 450 with \$SAMPLE_SIZE"

# Patch 4: Remove orphaned comment continuation lines (both appear identically in both branches)
sed -i '/^                              # (vs 25000 default = 400-700s trials)$/d' "$TARGET"
sed -i '/^                              # Benefits: Better Bayesian exploration, less GPU stress \\$/d' "$TARGET"
sed -i '/^                              # Benefits: Better Bayesian exploration, less GPU stress$/d' "$TARGET"
echo "Patch 4/5: removed orphaned timing comment lines"

# Patch 5: Add 192.168.3.162 to scp push loop
sed -i 's/for node in 192.168.3.120 192.168.3.154;/for node in 192.168.3.120 192.168.3.154 192.168.3.162;/' "$TARGET"
echo "Patch 5/5: added 192.168.3.162 to scp push loop"

# Post-flight verification
FAIL=0
grep -q 'SAMPLE_SIZE=5000'               "$TARGET" || { echo "FAIL: SAMPLE_SIZE default not found"; FAIL=1; }
grep -q -- '--sample-size)'              "$TARGET" || { echo "FAIL: --sample-size parser not found"; FAIL=1; }
grep -q -- '--sample-size \$SAMPLE_SIZE' "$TARGET" || { echo "FAIL: \$SAMPLE_SIZE not wired in"; FAIL=1; }
grep -q -- '--sample-size 450'           "$TARGET" && { echo "FAIL: hardcoded 450 still present"; FAIL=1; }
grep -q '(vs 25000 default'              "$TARGET" && { echo "FAIL: orphaned comment still present"; FAIL=1; }
grep -q '192.168.3.162'                  "$TARGET" || { echo "FAIL: 192.168.3.162 not added"; FAIL=1; }

[ $FAIL -ne 0 ] && { echo "Post-flight FAILED - rolling back"; cp "$BACKUP" "$TARGET"; exit 1; }

echo "Post-flight: all 6 checks passed"
echo "=== v2.3 patch complete ==="
