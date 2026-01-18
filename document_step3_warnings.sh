#!/bin/bash
# Document Step 3 "Ignoring Unknown Option" Behavior
# ===================================================
# 
# The "Note: Ignoring unknown option: X" warnings during Step 3 are EXPECTED.
# 
# Explanation:
# - full_scoring.json contains default_params with extra parameters for documentation
# - run_step3_full_scoring.sh only uses a subset of these parameters
# - The shell script's tolerance mechanism (added Jan 10, 2026) logs and skips unknown params
# - This is BY DESIGN for autonomy/configurability - scripts consume only what they understand
#
# This script adds clarifying comments to both the manifest and shell script.

set -e
cd ~/distributed_prng_analysis

echo "============================================================"
echo "Adding documentation for 'Ignoring unknown option' warnings"
echo "============================================================"

# 1. Add comment to run_step3_full_scoring.sh explaining the tolerance behavior
SHELL_SCRIPT="run_step3_full_scoring.sh"

if [[ -f "$SHELL_SCRIPT" ]]; then
    # Check if comment already exists
    if grep -q "TOLERANCE NOTE" "$SHELL_SCRIPT"; then
        echo "✅ Shell script already documented"
    else
        # Find the catch-all case and add comment before it
        python3 << 'PYEOF'
with open("run_step3_full_scoring.sh", "r") as f:
    content = f.read()

# Add documentation comment before the catch-all case
old_pattern = '''    *)
        echo "Note: Ignoring unknown option: $1"
        shift
        ;;'''

new_pattern = '''    # TOLERANCE NOTE (Jan 2026):
        # This catch-all allows the script to accept a superset of parameters.
        # WATCHER/manifests may pass extra params (prng_type, mod, batch_size, etc.)
        # that are documented in default_params but not used by this script.
        # This is BY DESIGN - scripts consume only what they understand.
        # The warnings are informational, not errors.
    *)
        echo "Note: Ignoring unknown option: $1"
        shift
        ;;'''

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    with open("run_step3_full_scoring.sh", "w") as f:
        f.write(content)
    print("✅ Added tolerance documentation to run_step3_full_scoring.sh")
else:
    print("⚠️  Could not find catch-all pattern - may already be modified")
PYEOF
    fi
else
    echo "⚠️  Shell script not found: $SHELL_SCRIPT"
fi

# 2. Add comment to full_scoring.json explaining extra params
MANIFEST="agent_manifests/full_scoring.json"

if [[ -f "$MANIFEST" ]]; then
    python3 << 'PYEOF'
import json

with open("agent_manifests/full_scoring.json", "r") as f:
    m = json.load(f)

# Add a documentation note about extra params
if "_note_default_params" not in m:
    m["_note_default_params"] = (
        "Some default_params (prng_type, mod, batch_size, jobs_file, output_file) "
        "are included for documentation/future use but are not currently consumed by "
        "run_step3_full_scoring.sh. The shell script's tolerance mechanism logs "
        "'Note: Ignoring unknown option' for these - this is expected behavior, not an error."
    )
    
    with open("agent_manifests/full_scoring.json", "w") as f:
        json.dump(m, f, indent=2)
    
    print("✅ Added documentation note to full_scoring.json")
else:
    print("✅ Manifest already has documentation note")
PYEOF
else
    echo "⚠️  Manifest not found: $MANIFEST"
fi

echo ""
echo "============================================================"
echo "Documentation added. The 'Ignoring unknown option' warnings"
echo "are now documented as expected behavior."
echo "============================================================"
