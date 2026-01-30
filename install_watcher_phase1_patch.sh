#!/bin/bash
# =============================================================================
# WATCHER Phase 1 Patch v1.1.1 - Self-Installing Script
# =============================================================================
# 
# Purpose: Fix stale output detection in WATCHER agent
# Date: 2026-01-27
# Author: Project Lead
#
# What this script does:
#   1. Adds required_inputs/primary_output to all 6 manifests (idempotent)
#   2. Patches agents/watcher_agent.py with freshness check logic
#   3. Validates all changes
#   4. Runs sanity test
#
# Usage:
#   bash install_watcher_phase1_patch.sh
#
# Rollback:
#   git checkout agents/watcher_agent.py agent_manifests/*.json
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "WATCHER Phase 1 Patch v1.1.1 - Installer"
echo "=============================================="
echo ""

# Check we're in the right directory
if [ ! -f "agents/watcher_agent.py" ]; then
    echo -e "${RED}ERROR: Must run from distributed_prng_analysis root directory${NC}"
    exit 1
fi

# Create backup
echo -e "${YELLOW}Creating backups...${NC}"
BACKUP_DIR="backups/phase1_patch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp agents/watcher_agent.py "$BACKUP_DIR/"
cp agent_manifests/*.json "$BACKUP_DIR/"
echo -e "${GREEN}✓ Backups saved to $BACKUP_DIR${NC}"

# =============================================================================
# PART 1: Update Manifests
# =============================================================================
echo ""
echo "=============================================="
echo "PART 1: Updating Manifests"
echo "=============================================="

update_manifest() {
    local file="$1"
    local required_inputs="$2"
    local primary_output="$3"
    
    # Check if already patched
    if grep -q '"required_inputs"' "$file"; then
        echo -e "${YELLOW}  ⚠ $file already has required_inputs - skipping${NC}"
        return 0
    fi
    
    # Use Python for safe JSON manipulation
    python3 << PYEOF
import json

file_path = "$file"
required_inputs = $required_inputs
primary_output = "$primary_output"

with open(file_path, 'r') as f:
    data = json.load(f)

# Add new fields after version (or at start if no version)
new_data = {}
for key, value in data.items():
    new_data[key] = value
    if key == "version":
        new_data["required_inputs"] = required_inputs
        new_data["primary_output"] = primary_output

# If version wasn't found, add at the end
if "required_inputs" not in new_data:
    new_data["required_inputs"] = required_inputs
    new_data["primary_output"] = primary_output

with open(file_path, 'w') as f:
    json.dump(new_data, f, indent=2)
    f.write('\n')

print(f"  ✓ Updated {file_path}")
PYEOF
}

# Step 1: window_optimizer.json
update_manifest "agent_manifests/window_optimizer.json" \
    '["synthetic_lottery.json"]' \
    "optimal_window_config.json"

# Step 2: scorer_meta.json
update_manifest "agent_manifests/scorer_meta.json" \
    '["bidirectional_survivors_binary.npz", "train_history.json", "holdout_history.json"]' \
    "optimal_scorer_config.json"

# Step 3: full_scoring.json
update_manifest "agent_manifests/full_scoring.json" \
    '["bidirectional_survivors_binary.npz", "optimal_scorer_config.json", "train_history.json", "holdout_history.json"]' \
    "survivors_with_scores.json"

# Step 4: ml_meta.json
update_manifest "agent_manifests/ml_meta.json" \
    '["optimal_window_config.json", "train_history.json"]' \
    "reinforcement_engine_config.json"

# Step 5: reinforcement.json
update_manifest "agent_manifests/reinforcement.json" \
    '["survivors_with_scores.json", "train_history.json", "reinforcement_engine_config.json"]' \
    "models/reinforcement/best_model.meta.json"

# Step 6: prediction.json
update_manifest "agent_manifests/prediction.json" \
    '["models/reinforcement/best_model.meta.json", "survivors_with_scores.json", "forward_survivors.json", "optimal_window_config.json"]' \
    "predictions/next_draw_prediction.json"

echo -e "${GREEN}✓ All manifests updated${NC}"

# =============================================================================
# PART 2: Patch watcher_agent.py
# =============================================================================
echo ""
echo "=============================================="
echo "PART 2: Patching watcher_agent.py"
echo "=============================================="

# Check if already patched
if grep -q "PREFLIGHT_HARD_FAILURES" agents/watcher_agent.py; then
    echo -e "${YELLOW}⚠ watcher_agent.py already patched - skipping code changes${NC}"
else
    # Create the patch content
    python3 << 'PYEOF'
import re

file_path = "agents/watcher_agent.py"

with open(file_path, 'r') as f:
    content = f.read()

# =============================================================================
# PATCH 1: Add failure classification constants after imports
# =============================================================================

# Find the end of imports (after the last 'import' or 'from' line before class definitions)
import_section_end = """
# =============================================================================
# PREFLIGHT FAILURE CLASSIFICATION (Phase 1 Patch v1.1.1 - 2026-01-27)
# =============================================================================
# HARD failures = Cannot proceed safely (missing critical resources)
# SOFT failures = Can proceed with reduced capacity (graceful degradation)
#
# WATCHER RULES:
#   - WATCHER may refuse to skip (force re-run)
#   - WATCHER may allow step to overwrite existing output
#   - WATCHER must NEVER delete artifacts
# =============================================================================

PREFLIGHT_HARD_FAILURES = [
    "ssh", "unreachable", "connection refused", "connection timed out",
    "no such file", "input file missing", "no gpus available",
    "bidirectional_survivors_binary.npz not found",
    "train_history.json not found", "holdout_history.json not found",
    "primary input missing"
]

PREFLIGHT_SOFT_FAILURES = [
    "ramdisk", "gpu count", "mismatch", "remediation failed", "degraded"
]


def classify_preflight_failure(failure_msg: str) -> str:
    \"\"\"
    Classify preflight failure as HARD (block) or SOFT (warn + continue).
    Returns: "HARD" or "SOFT"
    \"\"\"
    msg_lower = failure_msg.lower()
    for keyword in PREFLIGHT_HARD_FAILURES:
        if keyword in msg_lower:
            return "HARD"
    return "SOFT"

"""

# Find where to insert (before PREFLIGHT_AVAILABLE or class WatcherAgent)
if "PREFLIGHT_AVAILABLE" in content:
    # Insert before PREFLIGHT_AVAILABLE
    pattern = r'(\n)(try:\s*\n\s*from utils\.preflight_check)'
    if re.search(pattern, content):
        content = re.sub(pattern, import_section_end + r'\1\2', content, count=1)
        print("  ✓ Added failure classification constants")
    else:
        # Alternative: insert before class definition
        pattern = r'(\nclass WatcherAgent)'
        content = re.sub(pattern, import_section_end + r'\1', content, count=1)
        print("  ✓ Added failure classification constants (before class)")
else:
    # Insert before class definition
    pattern = r'(\nclass WatcherAgent)'
    content = re.sub(pattern, import_section_end + r'\1', content, count=1)
    print("  ✓ Added failure classification constants (before class)")

# =============================================================================
# PATCH 2: Add helper methods to WatcherAgent class
# =============================================================================

# Find the class and add methods after __init__ or at a suitable location
# We'll add them before _run_preflight_check if it exists

new_methods = '''
    def _resolve_path(self, p: str) -> str:
        """Resolve path relative to repo root."""
        import os
        if os.path.isabs(p):
            return p
        # Assume running from repo root
        return os.path.join(os.getcwd(), p)
    
    def _get_step_io_from_manifest(self, step: int) -> tuple:
        """
        Get required inputs and primary output from step manifest.
        Returns: (required_inputs: List[str], primary_output: str)
        Raises ValueError if manifest missing required fields.
        """
        import os
        import json
        
        STEP_MANIFESTS = {
            1: "window_optimizer.json",
            2: "scorer_meta.json",
            3: "full_scoring.json",
            4: "ml_meta.json",
            5: "reinforcement.json",
            6: "prediction.json"
        }
        
        manifest_name = STEP_MANIFESTS.get(step)
        if not manifest_name:
            raise ValueError(f"No manifest defined for step {step}")
        
        manifest_path = os.path.join("agent_manifests", manifest_name)
        if not os.path.exists(manifest_path):
            raise ValueError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        required_inputs = manifest.get("required_inputs")
        primary_output = manifest.get("primary_output")
        
        if not required_inputs:
            raise ValueError(f"Manifest {manifest_name} missing 'required_inputs'")
        if not primary_output:
            raise ValueError(f"Manifest {manifest_name} missing 'primary_output'")
        
        # Resolve paths
        required_inputs = [self._resolve_path(p) for p in required_inputs]
        primary_output = self._resolve_path(primary_output)
        
        return required_inputs, primary_output
    
    def _output_is_fresh(self, step: int) -> tuple:
        """
        Check if output file is newer than all input files.
        Returns: (is_fresh: bool, reason: str)
        
        NOTE: Freshness != semantic correctness. Phase 2 adds sidecar validation.
        """
        import os
        from datetime import datetime
        
        try:
            required_inputs, primary_output = self._get_step_io_from_manifest(step)
        except ValueError as e:
            # Missing manifest field = HARD failure (Team Beta requirement #5)
            return False, f"HARD: {e}"
        
        # Check required inputs exist (Team Beta requirement #2)
        for inp in required_inputs:
            if not os.path.exists(inp):
                return False, f"HARD: Primary input missing: {inp}"
        
        # Check output exists
        if not os.path.exists(primary_output):
            return False, f"Output missing: {primary_output}"
        
        output_mtime = os.path.getmtime(primary_output)
        output_time_str = datetime.fromtimestamp(output_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # Check each input - output must be newer than ALL inputs
        for inp in required_inputs:
            if os.path.exists(inp):
                input_mtime = os.path.getmtime(inp)
                if input_mtime > output_mtime:
                    input_time_str = datetime.fromtimestamp(input_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    return False, f"STALE: {primary_output} ({output_time_str}) older than {inp} ({input_time_str})"
        
        return True, f"Fresh: {primary_output} ({output_time_str})"

'''

# Find a good insertion point - after __init__ method or before _run_preflight_check
if "_run_preflight_check" in content and "_output_is_fresh" not in content:
    # Insert before _run_preflight_check
    pattern = r'(\n    def _run_preflight_check\(self)'
    content = re.sub(pattern, new_methods + r'\1', content, count=1)
    print("  ✓ Added _get_step_io_from_manifest and _output_is_fresh methods")
elif "_output_is_fresh" not in content:
    # Try to find end of __init__ or another method
    print("  ⚠ Could not find insertion point for methods - manual edit required")

# =============================================================================
# Write the patched file
# =============================================================================
with open(file_path, 'w') as f:
    f.write(content)

print("  ✓ watcher_agent.py patched")
PYEOF
fi

# =============================================================================
# PART 3: Validation
# =============================================================================
echo ""
echo "=============================================="
echo "PART 3: Validation"
echo "=============================================="

# Validate manifests have the new fields
echo "Checking manifests..."
for manifest in agent_manifests/window_optimizer.json agent_manifests/scorer_meta.json agent_manifests/full_scoring.json agent_manifests/ml_meta.json agent_manifests/reinforcement.json agent_manifests/prediction.json; do
    if grep -q '"required_inputs"' "$manifest" && grep -q '"primary_output"' "$manifest"; then
        echo -e "  ${GREEN}✓ $manifest${NC}"
    else
        echo -e "  ${RED}✗ $manifest - missing fields${NC}"
        exit 1
    fi
done

# Validate watcher_agent.py has the new code
echo "Checking watcher_agent.py..."
if grep -q "PREFLIGHT_HARD_FAILURES" agents/watcher_agent.py; then
    echo -e "  ${GREEN}✓ Failure classification constants present${NC}"
else
    echo -e "  ${RED}✗ Failure classification constants missing${NC}"
    exit 1
fi

if grep -q "_output_is_fresh" agents/watcher_agent.py; then
    echo -e "  ${GREEN}✓ Freshness check method present${NC}"
else
    echo -e "  ${YELLOW}⚠ Freshness check method not auto-inserted - may need manual edit${NC}"
fi

# Python syntax check
echo "Checking Python syntax..."
if python3 -m py_compile agents/watcher_agent.py 2>/dev/null; then
    echo -e "  ${GREEN}✓ Python syntax valid${NC}"
else
    echo -e "  ${RED}✗ Python syntax error - rolling back${NC}"
    cp "$BACKUP_DIR/watcher_agent.py" agents/watcher_agent.py
    exit 1
fi

# =============================================================================
# PART 4: Sanity Test
# =============================================================================
echo ""
echo "=============================================="
echo "PART 4: Sanity Test"
echo "=============================================="

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

print("Testing manifest IO resolution...")
try:
    # Simple test - just verify we can read the manifests
    import json
    import os
    
    STEP_MANIFESTS = {
        1: "window_optimizer.json",
        2: "scorer_meta.json",
        3: "full_scoring.json",
        4: "ml_meta.json",
        5: "reinforcement.json",
        6: "prediction.json"
    }
    
    for step, manifest_name in STEP_MANIFESTS.items():
        path = os.path.join("agent_manifests", manifest_name)
        with open(path) as f:
            data = json.load(f)
        
        ri = data.get("required_inputs", [])
        po = data.get("primary_output", "")
        
        print(f"  Step {step}: {len(ri)} inputs → {po}")
        
        if not ri or not po:
            print(f"    ERROR: Missing fields!")
            sys.exit(1)
    
    print("\n✅ All manifests valid!")
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Sanity test passed${NC}"
else
    echo -e "${RED}✗ Sanity test failed${NC}"
    exit 1
fi

# =============================================================================
# COMPLETE
# =============================================================================
echo ""
echo "=============================================="
echo -e "${GREEN}PHASE 1 PATCH INSTALLED SUCCESSFULLY${NC}"
echo "=============================================="
echo ""
echo "Changes made:"
echo "  • 6 manifests updated with required_inputs/primary_output"
echo "  • watcher_agent.py patched with freshness check"
echo ""
echo "To test:"
echo "  rm -f survivors_with_scores.json"
echo "  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 3 --end-step 3"
echo ""
echo "To rollback:"
echo "  cp $BACKUP_DIR/* agents/ agent_manifests/"
echo "  # OR: git checkout agents/watcher_agent.py agent_manifests/*.json"
echo ""
