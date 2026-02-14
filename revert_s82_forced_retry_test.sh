#!/usr/bin/env bash
# ==============================================================================
# revert_s82_forced_retry_test.sh
# Session 82 Monkey Test -- Revert Forced RETRY Patch
# ==============================================================================
#
# Removes ONLY the content between S82_FORCED_RETRY_TEST_BEGIN and
# S82_FORCED_RETRY_TEST_END markers. Original code is fully preserved.
#
# Safe to run multiple times (idempotent).
# ==============================================================================

set -euo pipefail

TARGET="training_health_check.py"
MARKER_BEGIN="S82_FORCED_RETRY_TEST_BEGIN"
MARKER_END="S82_FORCED_RETRY_TEST_END"

echo "=============================================="
echo " S82 Monkey Test -- Revert Forced RETRY Patch"
echo "=============================================="

# --- Pre-checks ---
if [ ! -f "$TARGET" ]; then
    echo "ERROR: $TARGET not found. Run from ~/distributed_prng_analysis/"
    exit 1
fi

if ! grep -q "$MARKER_BEGIN" "$TARGET"; then
    echo "SKIP: No S82 markers found -- patch not applied or already reverted."
    exit 0
fi

# --- Revert via Python (safe multi-line removal) ---
python3 << 'PYEOF'
target = "training_health_check.py"

with open(target, "r") as f:
    lines = f.readlines()

output = []
inside_patch = False

for line in lines:
    if "S82_FORCED_RETRY_TEST_BEGIN" in line:
        inside_patch = True
        continue
    if "S82_FORCED_RETRY_TEST_END" in line:
        inside_patch = False
        continue
    if inside_patch:
        continue
    output.append(line)

with open(target, "w") as f:
    f.writelines(output)

print("Patch reverted successfully via Python.")
PYEOF

# --- Verify ---
echo ""
echo "--- Verification ---"

if grep -q "$MARKER_BEGIN" "$TARGET"; then
    echo "FAIL: Begin marker still present -- revert incomplete!"
    exit 1
else
    echo "PASS: Begin marker removed"
fi

if grep -q "$MARKER_END" "$TARGET"; then
    echo "FAIL: End marker still present -- revert incomplete!"
    exit 1
else
    echo "PASS: End marker removed"
fi

if grep -q "S82 FORCED RETRY ACTIVE" "$TARGET"; then
    echo "FAIL: Monkey test code still present!"
    exit 1
else
    echo "PASS: Monkey test code removed"
fi

if grep -q "return _check_training_health_impl(diagnostics_path)" "$TARGET"; then
    echo "PASS: Original implementation preserved"
else
    echo "FAIL: Original implementation MISSING -- restore from backup!"
    echo "  cp ${TARGET}.s82_backup $TARGET"
    exit 1
fi

echo ""
echo "=============================================="
echo " REVERT COMPLETE"
echo "=============================================="
echo ""
echo "training_health_check.py is back to production state."
echo ""
echo "Backup file ${TARGET}.s82_backup can be removed:"
echo "  rm ${TARGET}.s82_backup"
echo "=============================================="
