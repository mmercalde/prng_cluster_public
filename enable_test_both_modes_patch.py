#!/usr/bin/env python3
"""
Patch: Enable test_both_modes by default in window_optimizer.json manifest

Purpose:
- Enables BOTH constant AND variable skip pattern testing in Step 1
- Doubles the search space per trial (4 sieves instead of 2)
- Produces richer survivor metadata for ML feature extraction

Changes:
1. parameter_bounds.test_both_modes.default: false → true
2. default_params: adds "test_both_modes": true

Downstream Compatibility (Verified):
- Step 2.5 (Scorer Meta): ✅ No changes needed
- Step 3 (Full Scoring): ✅ Already extracts skip_* features from metadata
- Step 4 (ML Meta): ✅ No changes needed
- Step 5 (Anti-Overfit): ✅ Uses extracted features
- Step 6 (Prediction): ✅ No changes needed
- WATCHER: ✅ File validation unchanged

Date: 2026-01-19
Author: Claude (Team Alpha)
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

MANIFEST_PATH = Path("agent_manifests/window_optimizer.json")

def apply_patch():
    if not MANIFEST_PATH.exists():
        print(f"❌ Manifest not found: {MANIFEST_PATH}")
        return False
    
    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = MANIFEST_PATH.with_suffix(f".json.bak_{timestamp}")
    shutil.copy(MANIFEST_PATH, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    # Load manifest
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    
    # Track changes
    changes = []
    
    # Change 1: Update parameter_bounds.test_both_modes.default
    if "parameter_bounds" in manifest and "test_both_modes" in manifest["parameter_bounds"]:
        old_value = manifest["parameter_bounds"]["test_both_modes"].get("default", False)
        if old_value != True:
            manifest["parameter_bounds"]["test_both_modes"]["default"] = True
            changes.append(f"parameter_bounds.test_both_modes.default: {old_value} → True")
    
    # Change 2: Add test_both_modes to default_params
    if "default_params" in manifest:
        if "test_both_modes" not in manifest["default_params"]:
            manifest["default_params"]["test_both_modes"] = True
            changes.append("default_params: added test_both_modes: true")
        elif manifest["default_params"]["test_both_modes"] != True:
            old_value = manifest["default_params"]["test_both_modes"]
            manifest["default_params"]["test_both_modes"] = True
            changes.append(f"default_params.test_both_modes: {old_value} → true")
    
    if not changes:
        print("ℹ️  No changes needed - test_both_modes already enabled")
        return True
    
    # Write updated manifest
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Patched {MANIFEST_PATH}")
    print("\nChanges applied:")
    for change in changes:
        print(f"  • {change}")
    
    return True


def verify_patch():
    """Verify the patch was applied correctly."""
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Check parameter_bounds
    pb_default = manifest.get("parameter_bounds", {}).get("test_both_modes", {}).get("default")
    pb_status = "✅" if pb_default == True else "❌"
    print(f"{pb_status} parameter_bounds.test_both_modes.default = {pb_default}")
    
    # Check default_params
    dp_value = manifest.get("default_params", {}).get("test_both_modes")
    dp_status = "✅" if dp_value == True else "❌"
    print(f"{dp_status} default_params.test_both_modes = {dp_value}")
    
    # Show what WATCHER will now do
    print("\n" + "-"*60)
    print("EFFECT: Step 1 will now run per trial:")
    print("  1. Forward sieve (constant skip) - java_lcg")
    print("  2. Reverse sieve (constant skip) - java_lcg")
    print("  3. Forward sieve (variable skip) - java_lcg_hybrid")
    print("  4. Reverse sieve (variable skip) - java_lcg_hybrid")
    print("\nSurvivors will be tagged with skip_mode: 'constant' or 'variable'")
    print("-"*60)
    
    return pb_default == True and dp_value == True


if __name__ == "__main__":
    print("="*60)
    print("PATCH: Enable test_both_modes by default")
    print("="*60)
    
    success = apply_patch()
    if success:
        verify_patch()
        print("\n✅ Patch complete!")
        print("\nTo test:")
        print("  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \\")
        print("      --start-step 1 --end-step 1 \\")
        print("      --params '{\"trials\": 3, \"max_seeds\": 100000}'")
    else:
        print("\n❌ Patch failed")
