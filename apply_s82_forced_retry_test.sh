#!/usr/bin/env bash
# ==============================================================================
# apply_s82_forced_retry_test.sh
# Session 82 Monkey Test -- Force RETRY from check_training_health()
# ==============================================================================
#
# PURPOSE:
#   Injects a forced RETRY return into check_training_health() to prove:
#   - S76 retry threading works
#   - S81 LLM refinement executes
#   - Clamp enforcement works
#   - Lifecycle invocation works
#   - No daemon regression
#   - Phase 7 is fully closed
#
# ARCHITECTURE:
#   - ONLY touches training_health_check.py
#   - Does NOT modify WATCHER, policies, thresholds, or daemon state
#   - Inserts forced return BEFORE real implementation call
#   - Guarded by S82_FORCED_RETRY_TEST_BEGIN/END markers
#   - Idempotent: safe to re-run
#   - Reversible: revert_s82_forced_retry_test.sh removes cleanly
#
# Team Beta: APPROVED (Session 82)
# ==============================================================================

set -euo pipefail

TARGET="training_health_check.py"
MARKER_BEGIN="# >>> S82_FORCED_RETRY_TEST_BEGIN <<<"
MARKER_END="# >>> S82_FORCED_RETRY_TEST_END <<<"

echo "=============================================="
echo " S82 Monkey Test -- Apply Forced RETRY Patch"
echo "=============================================="

# --- Pre-checks ---
if [ ! -f "$TARGET" ]; then
    echo "ERROR: $TARGET not found. Run from ~/distributed_prng_analysis/"
    exit 1
fi

if grep -q "S82_FORCED_RETRY_TEST_BEGIN" "$TARGET"; then
    echo "SKIP: Patch already applied (markers found)."
    echo "To re-apply, run revert_s82_forced_retry_test.sh first."
    exit 0
fi

# Verify anchor exists
ANCHOR='    try:
        return _check_training_health_impl(diagnostics_path)'

if ! grep -q "return _check_training_health_impl(diagnostics_path)" "$TARGET"; then
    echo "ERROR: Anchor not found in $TARGET."
    echo "Expected: 'return _check_training_health_impl(diagnostics_path)'"
    echo "File may have been modified. Aborting."
    exit 1
fi

# --- Backup ---
cp "$TARGET" "${TARGET}.s82_backup"
echo "Backup: ${TARGET}.s82_backup"

# --- Apply patch ---
# We insert the forced return block INSIDE check_training_health(), right after
# the docstring closes and before the try: block. This uses Python's sed to
# insert before the 'try:' line within the function.

python3 << 'PYEOF'
import re

target = "training_health_check.py"

with open(target, "r") as f:
    content = f.read()

# The anchor: the try/return block inside check_training_health()
# We insert our forced return BEFORE the try: line
anchor = '    try:\n        return _check_training_health_impl(diagnostics_path)'

patch_code = '''    # >>> S82_FORCED_RETRY_TEST_BEGIN <<<
    # MONKEY TEST: Force RETRY to validate full WATCHER retry loop
    # Applied: Session 82 | Revert after test with revert_s82_forced_retry_test.sh
    # This bypasses real health check and returns a synthetic CRITICAL/RETRY dict
    import logging as _s82_log
    _s82_logger = _s82_log.getLogger("S82_MONKEY_TEST")
    _s82_logger.warning("S82 FORCED RETRY ACTIVE -- training_health_check returning synthetic RETRY")
    return {
        'action': 'RETRY',
        'model_type': 'neural_net',
        'severity': 'critical',
        'issues': [
            'S82 MONKEY TEST: Forced critical -- gradient explosion simulated',
            'S82 MONKEY TEST: Forced critical -- severe overfitting (gap ratio: 0.87)',
            'S82 MONKEY TEST: Forced critical -- dead ReLU neurons detected (62%)',
        ],
        'suggested_fixes': [
            'Switch to tree-based model (CatBoost)',
            'Enable feature normalization',
            'Increase dropout regularization',
        ],
        'confidence': 0.9,
        'note': 'S82 FORCED RETRY -- synthetic health check for WATCHER retry loop validation',
    }
    # >>> S82_FORCED_RETRY_TEST_END <<<
'''

if anchor not in content:
    print("ERROR: Anchor not found in file content (Python check)")
    exit(1)

# Insert patch BEFORE the anchor
patched = content.replace(anchor, patch_code + anchor)

with open(target, "w") as f:
    f.write(patched)

print("Patch applied successfully via Python.")
PYEOF

# --- Verify ---
echo ""
echo "--- Verification ---"

if grep -q "S82_FORCED_RETRY_TEST_BEGIN" "$TARGET"; then
    echo "PASS: Begin marker found"
else
    echo "FAIL: Begin marker NOT found"
    exit 1
fi

if grep -q "S82_FORCED_RETRY_TEST_END" "$TARGET"; then
    echo "PASS: End marker found"
else
    echo "FAIL: End marker NOT found"
    exit 1
fi

if grep -q "S82 FORCED RETRY ACTIVE" "$TARGET"; then
    echo "PASS: Log message present"
else
    echo "FAIL: Log message NOT found"
    exit 1
fi

if grep -q "return _check_training_health_impl(diagnostics_path)" "$TARGET"; then
    echo "PASS: Original try/return block preserved"
else
    echo "FAIL: Original implementation block MISSING -- REVERT IMMEDIATELY"
    exit 1
fi

echo ""
echo "=============================================="
echo " PATCH APPLIED SUCCESSFULLY"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Run the test:"
echo "     PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \\"
echo "       --start-step 5 --end-step 6 \\"
echo "       --params '{\"trials\":3,\"max_seeds\":5000,\"enable_diagnostics\":true}'"
echo ""
echo "  2. Look for these log lines:"
echo "     - 'S82 FORCED RETRY ACTIVE'"
echo "     - '[WATCHER][HEALTH] ... requesting RETRY'"
echo "     - 'Applied:' or 'REJECTED (bounds)'"
echo "     - 'Re-running Step 5 with updated params'"
echo ""
echo "  3. REVERT after testing:"
echo "     bash revert_s82_forced_retry_test.sh"
echo ""
echo "  DO NOT FORGET TO REVERT."
echo "=============================================="
