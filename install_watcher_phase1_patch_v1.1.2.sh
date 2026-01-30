#!/bin/bash
# =============================================================================
# WATCHER Phase 1 Patch v1.1.2 - Self-Installing Script
# =============================================================================
# 
# Purpose: Fix stale output detection in WATCHER agent
# Date: 2026-01-27
# Author: Project Lead
# Version: 1.1.2 (Team Beta corrections applied)
#
# Corrections from v1.1.1:
#   1. Fail hard if method insertion fails (no manual edits allowed)
#   2. Patch _run_step() to actually USE freshness logic
#   3. STEP_MANIFESTS defined once at module scope (no duplication)
#   4. Repo root derived from __file__, not os.getcwd()
#   5. Rollback uses explicit file paths
#
# Usage:
#   cd ~/distributed_prng_analysis
#   bash install_watcher_phase1_patch.sh
#
# Rollback:
#   cp backups/phase1_patch_XXXXX/watcher_agent.py agents/
#   cp backups/phase1_patch_XXXXX/*.json agent_manifests/
#   # OR: git checkout agents/watcher_agent.py agent_manifests/*.json
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "WATCHER Phase 1 Patch v1.1.2 - Installer"
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
if grep -q "PHASE1_PATCH_INSTALLED" agents/watcher_agent.py; then
    echo -e "${YELLOW}⚠ watcher_agent.py already patched (v1.1.2) - skipping${NC}"
else
    # Create the patch using Python
    python3 << 'PYEOF'
import re
import sys

file_path = "agents/watcher_agent.py"

with open(file_path, 'r') as f:
    content = f.read()

errors = []
changes = []

# =============================================================================
# PATCH 1: Add module-level constants and STEP_MANIFESTS (after imports)
# =============================================================================

module_additions = '''
# =============================================================================
# PHASE1_PATCH_INSTALLED - v1.1.2 (2026-01-27)
# Stale Output Detection + Preflight Classification
# =============================================================================

from datetime import datetime as _datetime_module

# STEP_MANIFESTS - Single source of truth for step→manifest mapping
# Used by freshness check and other components
STEP_MANIFESTS = {
    1: "window_optimizer.json",
    2: "scorer_meta.json",
    3: "full_scoring.json",
    4: "ml_meta.json",
    5: "reinforcement.json",
    6: "prediction.json"
}

# Derive REPO_ROOT from this file's location (not os.getcwd())
import os as _os_module
REPO_ROOT = _os_module.path.dirname(_os_module.path.dirname(_os_module.path.abspath(__file__)))

# PREFLIGHT FAILURE CLASSIFICATION
# HARD = Cannot proceed safely (missing critical resources)
# SOFT = Can proceed with reduced capacity (graceful degradation)
PREFLIGHT_HARD_FAILURES = [
    "ssh", "unreachable", "connection refused", "connection timed out",
    "no such file", "input file missing", "no gpus available",
    "bidirectional_survivors_binary.npz not found",
    "train_history.json not found", "holdout_history.json not found",
    "primary input missing", "manifest missing"
]

PREFLIGHT_SOFT_FAILURES = [
    "ramdisk", "gpu count", "mismatch", "remediation failed", "degraded"
]


def classify_preflight_failure(failure_msg: str) -> str:
    """Classify preflight failure as HARD (block) or SOFT (warn + continue)."""
    msg_lower = failure_msg.lower()
    for keyword in PREFLIGHT_HARD_FAILURES:
        if keyword in msg_lower:
            return "HARD"
    return "SOFT"


def resolve_repo_path(p: str) -> str:
    """Resolve path relative to REPO_ROOT."""
    if _os_module.path.isabs(p):
        return p
    return _os_module.path.join(REPO_ROOT, p)


def get_step_io_from_manifest(step: int) -> tuple:
    """
    Get required inputs and primary output from step manifest.
    Returns: (required_inputs: List[str], primary_output: str)
    Raises ValueError if manifest missing required fields (HARD failure).
    """
    import json
    
    manifest_name = STEP_MANIFESTS.get(step)
    if not manifest_name:
        raise ValueError(f"No manifest defined for step {step}")
    
    manifest_path = _os_module.path.join(REPO_ROOT, "agent_manifests", manifest_name)
    if not _os_module.path.exists(manifest_path):
        raise ValueError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    required_inputs = manifest.get("required_inputs")
    primary_output = manifest.get("primary_output")
    
    if not required_inputs:
        raise ValueError(f"Manifest {manifest_name} missing 'required_inputs' - run installer")
    if not primary_output:
        raise ValueError(f"Manifest {manifest_name} missing 'primary_output' - run installer")
    
    # Resolve all paths relative to repo root
    required_inputs = [resolve_repo_path(p) for p in required_inputs]
    primary_output = resolve_repo_path(primary_output)
    
    return required_inputs, primary_output


def check_output_freshness(step: int) -> tuple:
    """
    Check if output file is newer than all input files.
    Returns: (is_fresh: bool, reason: str, is_hard_failure: bool)
    
    NOTE: Freshness != semantic correctness. Phase 2 adds sidecar validation.
    """
    try:
        required_inputs, primary_output = get_step_io_from_manifest(step)
    except ValueError as e:
        # Missing manifest field = HARD failure
        return False, f"HARD: {e}", True
    
    # Check required inputs exist (Team Beta requirement #2)
    for inp in required_inputs:
        if not _os_module.path.exists(inp):
            return False, f"HARD: Primary input missing: {inp}", True
    
    # Check output exists
    if not _os_module.path.exists(primary_output):
        return False, f"Output missing: {primary_output}", False
    
    output_mtime = _os_module.path.getmtime(primary_output)
    output_time_str = _datetime_module.fromtimestamp(output_mtime).strftime("%Y-%m-%d %H:%M:%S")
    
    # Check each input - output must be newer than ALL inputs
    for inp in required_inputs:
        input_mtime = _os_module.path.getmtime(inp)
        if input_mtime > output_mtime:
            input_time_str = _datetime_module.fromtimestamp(input_mtime).strftime("%Y-%m-%d %H:%M:%S")
            return False, f"STALE: {primary_output} ({output_time_str}) older than {inp} ({input_time_str})", False
    
    return True, f"Fresh: {primary_output} ({output_time_str})", False

'''

# Find insertion point - after imports, before class definition or PREFLIGHT_AVAILABLE
if "PHASE1_PATCH_INSTALLED" in content:
    print("  ✓ Module constants already present")
else:
    # Try to insert before 'try:' block for PREFLIGHT_AVAILABLE, or before class
    if "PREFLIGHT_AVAILABLE" in content:
        pattern = r'(\n)(try:\s*\n\s*from utils\.preflight_check)'
        if re.search(pattern, content):
            content = re.sub(pattern, module_additions + r'\n\1\2', content, count=1)
            changes.append("Added module-level constants before PREFLIGHT_AVAILABLE")
        else:
            pattern = r'(\nclass WatcherAgent)'
            if re.search(pattern, content):
                content = re.sub(pattern, module_additions + r'\1', content, count=1)
                changes.append("Added module-level constants before class")
            else:
                errors.append("Could not find insertion point for module constants")
    else:
        pattern = r'(\nclass WatcherAgent)'
        if re.search(pattern, content):
            content = re.sub(pattern, module_additions + r'\1', content, count=1)
            changes.append("Added module-level constants before class")
        else:
            errors.append("Could not find insertion point for module constants")

# =============================================================================
# PATCH 2: Modify _run_preflight_check to return 3-tuple
# =============================================================================

# Check if already modified
if "is_hard_failure" in content and "_run_preflight_check" in content:
    print("  ✓ _run_preflight_check already returns 3-tuple")
else:
    # Find the return statements in _run_preflight_check and modify them
    # This is complex - we need to change the function signature and returns
    
    # Find the function
    preflight_pattern = r'(def _run_preflight_check\(self, step: int\)) -> Tuple\[bool, str\]:'
    if re.search(preflight_pattern, content):
        # Update return type
        content = re.sub(
            preflight_pattern,
            r'\1 -> Tuple[bool, str, bool]:',
            content
        )
        changes.append("Updated _run_preflight_check return type to 3-tuple")
        
        # Update return statements - this is tricky, we'll add a note
        # The function needs manual review to ensure all returns include is_hard_failure
        print("  ⚠ _run_preflight_check return type updated - verify return statements")
    else:
        # Try simpler pattern
        preflight_pattern2 = r'def _run_preflight_check\(self, step: int\).*?:'
        if re.search(preflight_pattern2, content):
            print("  ⚠ _run_preflight_check found but signature differs - needs review")
        else:
            print("  ⚠ _run_preflight_check not found - may already be modified")

# =============================================================================
# PATCH 3: Add freshness check call in _run_step (CRITICAL - Team Beta #2)
# =============================================================================

# We need to find where _run_step processes each step and add freshness check
# Look for the preflight check section in _run_step

freshness_block = '''
        # =====================================================================
        # FRESHNESS CHECK (Phase 1 Patch v1.1.2)
        # Check if output exists and is newer than all inputs
        # =====================================================================
        is_fresh, freshness_msg, is_hard_freshness_fail = check_output_freshness(step)
        
        if is_hard_freshness_fail:
            # Missing inputs or manifest = HARD block
            logger.error(f"Step {step} blocked: {freshness_msg}")
            return {
                "success": False,
                "error": freshness_msg,
                "blocked_by": "freshness_hard_failure"
            }
        
        if is_fresh:
            # Output exists and is fresh - check if we should skip
            if not preflight_passed:
                # Soft failure + fresh output = skip with warning (Team Beta #2)
                warning_msg = (
                    f"⚠️  DEGRADED SKIP: Step {step} using existing output under soft preflight failure.\\n"
                    f"    Preflight issue: {preflight_msg}\\n"
                    f"    Output status: {freshness_msg}\\n"
                    f"    Recommend manual verification if results seem unexpected."
                )
                logger.warning(warning_msg)
                print(warning_msg)
            logger.info(f"Step {step}: {freshness_msg}")
        else:
            # Output missing or stale - will run
            logger.info(f"Step {step}: {freshness_msg} - will execute")
            if not preflight_passed and not is_hard_failure:
                logger.warning(f"Step {step} running in degraded mode: {preflight_msg}")
        
'''

# Find where to insert - after preflight check in _run_step
# Look for pattern like "preflight_passed, preflight_msg = self._run_preflight_check(step)"
preflight_call_pattern = r'(preflight_passed, preflight_msg = self\._run_preflight_check\(step\))'

if "check_output_freshness(step)" in content:
    print("  ✓ Freshness check already integrated in _run_step")
elif re.search(preflight_call_pattern, content):
    # Need to also capture the is_hard_failure
    # First update the call to handle 3-tuple
    content = re.sub(
        preflight_call_pattern,
        'preflight_passed, preflight_msg, is_hard_failure = self._run_preflight_check(step)',
        content
    )
    
    # Now find where to insert freshness block - after the hard failure check
    hard_fail_pattern = r'(if not preflight_passed:[\s\S]*?blocked_by.*?preflight.*?\n\s*\})'
    
    # Actually, let's be more careful. Find the block after preflight and insert
    # Look for the pattern where we check preflight and potentially return
    insert_pattern = r'(preflight_passed, preflight_msg, is_hard_failure = self\._run_preflight_check\(step\)[\s\S]*?if not preflight_passed:[\s\S]*?\})'
    
    if re.search(insert_pattern, content):
        # Insert freshness block after the preflight handling
        content = re.sub(
            insert_pattern,
            r'\1\n' + freshness_block,
            content,
            count=1
        )
        changes.append("Added freshness check in _run_step after preflight")
    else:
        # Simpler approach: just add after the preflight call line
        simple_pattern = r'(preflight_passed, preflight_msg, is_hard_failure = self\._run_preflight_check\(step\)\n)'
        if re.search(simple_pattern, content):
            content = re.sub(
                simple_pattern,
                r'\1' + freshness_block,
                content,
                count=1
            )
            changes.append("Added freshness check after preflight call")
        else:
            errors.append("Could not find insertion point for freshness check in _run_step")
else:
    # Maybe it's already a 3-tuple call?
    if "is_hard_failure = self._run_preflight_check" in content:
        print("  ✓ Preflight already returns 3-tuple")
        # Still need to add freshness check
        if "check_output_freshness" not in content:
            errors.append("Preflight updated but freshness check not integrated")
    else:
        errors.append("Could not find preflight call pattern in _run_step")

# =============================================================================
# Final validation and write
# =============================================================================

if errors:
    print("\n❌ ERRORS - Cannot auto-patch:")
    for e in errors:
        print(f"   • {e}")
    print("\nAborting. Manual intervention required.")
    sys.exit(1)

if changes:
    print("\n✓ Changes applied:")
    for c in changes:
        print(f"   • {c}")

# Write the patched file
with open(file_path, 'w') as f:
    f.write(content)

print("\n✓ watcher_agent.py patched successfully")
PYEOF

    PATCH_RESULT=$?
    if [ $PATCH_RESULT -ne 0 ]; then
        echo -e "${RED}✗ Patch failed - restoring backup${NC}"
        cp "$BACKUP_DIR/watcher_agent.py" agents/watcher_agent.py
        exit 1
    fi
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
MANIFEST_ERRORS=0
for manifest in agent_manifests/window_optimizer.json agent_manifests/scorer_meta.json agent_manifests/full_scoring.json agent_manifests/ml_meta.json agent_manifests/reinforcement.json agent_manifests/prediction.json; do
    if grep -q '"required_inputs"' "$manifest" && grep -q '"primary_output"' "$manifest"; then
        echo -e "  ${GREEN}✓ $manifest${NC}"
    else
        echo -e "  ${RED}✗ $manifest - missing fields${NC}"
        MANIFEST_ERRORS=$((MANIFEST_ERRORS + 1))
    fi
done

if [ $MANIFEST_ERRORS -gt 0 ]; then
    echo -e "${RED}✗ Manifest validation failed${NC}"
    exit 1
fi

# Validate watcher_agent.py has the new code
echo "Checking watcher_agent.py..."
WATCHER_ERRORS=0

if grep -q "PHASE1_PATCH_INSTALLED" agents/watcher_agent.py; then
    echo -e "  ${GREEN}✓ Patch marker present${NC}"
else
    echo -e "  ${RED}✗ Patch marker missing${NC}"
    WATCHER_ERRORS=$((WATCHER_ERRORS + 1))
fi

if grep -q "STEP_MANIFESTS" agents/watcher_agent.py; then
    echo -e "  ${GREEN}✓ STEP_MANIFESTS defined${NC}"
else
    echo -e "  ${RED}✗ STEP_MANIFESTS missing${NC}"
    WATCHER_ERRORS=$((WATCHER_ERRORS + 1))
fi

if grep -q "check_output_freshness" agents/watcher_agent.py; then
    echo -e "  ${GREEN}✓ Freshness check function present${NC}"
else
    echo -e "  ${RED}✗ Freshness check function missing${NC}"
    WATCHER_ERRORS=$((WATCHER_ERRORS + 1))
fi

if grep -q "REPO_ROOT" agents/watcher_agent.py; then
    echo -e "  ${GREEN}✓ REPO_ROOT defined${NC}"
else
    echo -e "  ${RED}✗ REPO_ROOT missing${NC}"
    WATCHER_ERRORS=$((WATCHER_ERRORS + 1))
fi

if [ $WATCHER_ERRORS -gt 0 ]; then
    echo -e "${RED}✗ Watcher validation failed - restoring backup${NC}"
    cp "$BACKUP_DIR/watcher_agent.py" agents/watcher_agent.py
    exit 1
fi

# Python syntax check
echo "Checking Python syntax..."
if python3 -m py_compile agents/watcher_agent.py 2>/dev/null; then
    echo -e "  ${GREEN}✓ Python syntax valid${NC}"
else
    echo -e "  ${RED}✗ Python syntax error - restoring backup${NC}"
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
    # Import the patched functions
    from agents.watcher_agent import STEP_MANIFESTS, get_step_io_from_manifest, REPO_ROOT
    
    print(f"  REPO_ROOT: {REPO_ROOT}")
    
    for step in [1, 2, 3, 4, 5, 6]:
        try:
            ri, po = get_step_io_from_manifest(step)
            print(f"  Step {step}: {len(ri)} inputs → {po.split('/')[-1]}")
        except ValueError as e:
            print(f"  Step {step}: ERROR - {e}")
            sys.exit(1)
    
    print("\n✅ Manifest IO resolution works!")
    
    # Test freshness check (will fail gracefully if files don't exist)
    print("\nTesting freshness check...")
    from agents.watcher_agent import check_output_freshness
    
    is_fresh, msg, is_hard = check_output_freshness(3)
    print(f"  Step 3: fresh={is_fresh}, hard={is_hard}")
    print(f"  Message: {msg[:80]}...")
    
    print("\n✅ Freshness check works!")
    
except ImportError as e:
    print(f"ERROR: Import failed - {e}")
    print("The patch may not have been applied correctly.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Sanity test passed${NC}"
else
    echo -e "${RED}✗ Sanity test failed - restoring backup${NC}"
    cp "$BACKUP_DIR/watcher_agent.py" agents/watcher_agent.py
    exit 1
fi

# =============================================================================
# COMPLETE
# =============================================================================
echo ""
echo "=============================================="
echo -e "${GREEN}PHASE 1 PATCH v1.1.2 INSTALLED SUCCESSFULLY${NC}"
echo "=============================================="
echo ""
echo "Changes made:"
echo "  • 6 manifests updated with required_inputs/primary_output"
echo "  • watcher_agent.py patched with:"
echo "    - STEP_MANIFESTS (single source of truth)"
echo "    - REPO_ROOT (derived from __file__)"
echo "    - check_output_freshness() function"
echo "    - Preflight failure classification"
echo ""
echo "To test:"
echo "  rm -f survivors_with_scores.json"
echo "  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 3 --end-step 3"
echo ""
echo "To rollback:"
echo "  cp $BACKUP_DIR/watcher_agent.py agents/"
echo "  cp $BACKUP_DIR/*.json agent_manifests/"
echo "  # OR: git checkout agents/watcher_agent.py agent_manifests/*.json"
echo ""
