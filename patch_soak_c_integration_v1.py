#!/usr/bin/env python3
"""
Soak C Integration Patches

Applies patches to enable true autonomous operation in test mode:
1. chapter_13_acceptance.py — Honor skip_escalation_in_test_mode
2. chapter_13_acceptance.py — Honor auto_approve_in_test_mode

Usage:
    cd ~/distributed_prng_analysis
    python3 patch_soak_c_integration_v1.py --apply
    python3 patch_soak_c_integration_v1.py --revert
    python3 patch_soak_c_integration_v1.py --check

VERSION: 1.1.1
DATE: 2026-02-05
SESSION: 60
TEAM BETA: Approved with refinements (v1.1.1 fixes marker detection + escalation gating)
"""

import argparse
import shutil
import sys
from pathlib import Path

PATCHES = {
    "chapter_13_acceptance.py": {
        "backup_suffix": ".pre_soakc_patch",
        "patches": [
            {
                "name": "skip_escalation_in_test_mode",
                "marker": "SOAK C PATCH v1.1: skip_escalation_in_test_mode",
                "find": """        # If any escalation reasons, escalate
        if escalation_reasons:
            return self._create_decision(
                ValidationResult.ESCALATE,
                "Mandatory escalation",""",
                "replace": """        # If any escalation reasons, escalate (unless test mode skip enabled)
        # === SOAK C PATCH v1.1: skip_escalation_in_test_mode ===
        # Team Beta refinement: explicit logging with count + reasons
        _skip_policies = self.policies if hasattr(self, 'policies') else {}
        _test_mode = _skip_policies.get('test_mode', False)
        _skip_esc = _skip_policies.get('skip_escalation_in_test_mode', False)
        _suppress_escalation = _test_mode and _skip_esc
        
        if _suppress_escalation and escalation_reasons:
            logger.warning(
                "SOAK C: Escalation suppressed (%d reasons): %s",
                len(escalation_reasons),
                escalation_reasons
            )
        # === END SOAK C PATCH ===
        
        if escalation_reasons and not _suppress_escalation:
            return self._create_decision(
                ValidationResult.ESCALATE,
                "Mandatory escalation","""
            },
            {
                "name": "auto_approve_in_test_mode",
                "marker": "SOAK C PATCH v1.1: auto_approve_in_test_mode",
                "find": """        # =====================================================================
        # AUTOMATIC ACCEPTANCE (Section 13.2)
        # =====================================================================""",
                "replace": """        # =====================================================================
        # === SOAK C PATCH v1.1: auto_approve_in_test_mode ===
        # Team Beta refinement: consistent policy access via self.policies
        # =====================================================================
        _auto_policies = self.policies if hasattr(self, 'policies') else {}
        if _auto_policies.get('test_mode') and _auto_policies.get('auto_approve_in_test_mode'):
            logger.info("SOAK C: Auto-approving proposal (test_mode + auto_approve_in_test_mode)")
            return self._create_decision(
                ValidationResult.ACCEPT,
                "Auto-approved in test mode",
                violations=[],
                proposal_id=proposal_id,
                timestamp=timestamp,
            )
        # === END SOAK C PATCH ===
        
        # =====================================================================
        # AUTOMATIC ACCEPTANCE (Section 13.2)
        # ====================================================================="""
            }
        ]
    }
}


def apply_patches():
    """Apply all Soak C integration patches."""
    success_count = 0
    
    for filename, config in PATCHES.items():
        filepath = Path(filename)
        if not filepath.exists():
            print(f"❌ File not found: {filename}")
            print(f"   Run this from ~/distributed_prng_analysis/")
            continue
        
        # Backup
        backup_path = Path(f"{filename}{config['backup_suffix']}")
        if not backup_path.exists():
            shutil.copy(filepath, backup_path)
            print(f"📦 Backed up: {filename} → {backup_path}")
        else:
            print(f"📦 Backup exists: {backup_path}")
        
        # Read content
        content = filepath.read_text()
        patches_applied_this_file = 0
        
        # Apply patches
        for patch in config["patches"]:
            marker = patch.get("marker", f"SOAK C PATCH: {patch['name']}")
            if marker in content:
                print(f"⚠️  Already applied: {patch['name']}")
                patches_applied_this_file += 1
            elif patch["find"] in content:
                content = content.replace(patch["find"], patch["replace"])
                print(f"✅ Applied: {patch['name']}")
                patches_applied_this_file += 1
            else:
                print(f"❌ Could not find target for: {patch['name']}")
                print(f"   Expected to find:")
                print(f"   {patch['find'][:80]}...")
        
        # Write if any patches applied
        if patches_applied_this_file > 0:
            filepath.write_text(content)
            success_count += patches_applied_this_file
    
    print(f"\n{'='*60}")
    if success_count > 0:
        print(f"✅ {success_count} patch(es) applied successfully")
        print(f"\nVerify with:")
        print(f"  grep -n 'SOAK C PATCH' chapter_13_acceptance.py")
        print(f"\nRevert with:")
        print(f"  python3 {sys.argv[0]} --revert")
    else:
        print(f"❌ No patches applied")
    print(f"{'='*60}")
    
    return success_count > 0


def revert_patches():
    """Revert all Soak C integration patches."""
    reverted = 0
    
    for filename, config in PATCHES.items():
        backup_path = Path(f"{filename}{config['backup_suffix']}")
        filepath = Path(filename)
        
        if backup_path.exists():
            shutil.copy(backup_path, filepath)
            print(f"✅ Reverted: {filename}")
            reverted += 1
        else:
            print(f"⚠️  No backup found for: {filename}")
    
    print(f"\n{'='*60}")
    print(f"✅ {reverted} file(s) reverted")
    print(f"{'='*60}")


def check_patches():
    """Check if patches are applied."""
    for filename, config in PATCHES.items():
        filepath = Path(filename)
        if not filepath.exists():
            print(f"❌ File not found: {filename}")
            continue
        
        content = filepath.read_text()
        print(f"\n{filename}:")
        
        for patch in config["patches"]:
            marker = patch.get("marker", f"SOAK C PATCH: {patch['name']}")
            if marker in content:
                print(f"  ✅ {patch['name']}: APPLIED")
            elif patch["find"] in content:
                print(f"  ⬜ {patch['name']}: NOT APPLIED")
            else:
                print(f"  ❓ {patch['name']}: UNKNOWN STATE (source may have changed)")
        
        # Check backup
        backup_path = Path(f"{filename}{config['backup_suffix']}")
        if backup_path.exists():
            print(f"  📦 Backup exists: {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Soak C Integration Patches for chapter_13_acceptance.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 patch_soak_c_integration_v1.py --check   # Check current state
  python3 patch_soak_c_integration_v1.py --apply   # Apply patches
  python3 patch_soak_c_integration_v1.py --revert  # Revert to backup
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true", help="Apply Soak C patches")
    group.add_argument("--revert", action="store_true", help="Revert to backup")
    group.add_argument("--check", action="store_true", help="Check patch status")
    
    args = parser.parse_args()
    
    if args.apply:
        success = apply_patches()
        sys.exit(0 if success else 1)
    elif args.revert:
        revert_patches()
    elif args.check:
        check_patches()


if __name__ == "__main__":
    main()
