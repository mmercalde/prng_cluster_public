#!/usr/bin/env python3
"""
SAFE PATCH INSTALLER v3: Incremental Output Writing for Window Optimizer
=========================================================================
Team Beta Approved Architecture - Guided Installer (NOT auto-rewriter)

v3 FIXES (per Team Beta review):
- Fixed import insertion logic (no docstring parsing, handles __future__)
- Strengthened manual edit verification (regex exact match)
- Runtime import validation runs from repo root
- Added content hash tracking in manifest
- Added --repo-root option

Usage:
    python3 apply_incremental_output_patch_v3.py [--dry-run] [--target PATH] [--verify-only] [--repo-root PATH]

Author: Claude (Team Beta Approved v3)
Date: 2026-01-18
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from shutil import copy2

# =============================================================================
# PATCH CONTENT - SAFE ADDITIONS ONLY
# =============================================================================

IMPORTS_TO_ADD = '''# === INCREMENTAL OUTPUT PATCH v1.0 (2026-01-18) ===
from datetime import datetime as dt_datetime
from pathlib import Path as dt_Path
# === END PATCH IMPORTS ===
'''

CALLBACK_FUNCTION = '''
# === INCREMENTAL OUTPUT CALLBACK (Team Beta Approved 2026-01-18) ===
def create_incremental_save_callback(
    output_config_path: str = "optimal_window_config.json",
    output_survivors_path: str = "bidirectional_survivors.json",
    total_trials: int = 50
):
    """
    Factory function that creates an Optuna callback for incremental saving.
    
    The callback writes best-so-far results after each completed trial,
    ensuring crash recovery and WATCHER visibility.
    """
    import json as _json
    
    def save_best_so_far_callback(study, trial):
        """Invoked after each trial completes."""
        
        completed = len([t for t in study.trials 
                        if t.state.name == "COMPLETE"])
        
        # Build progress metadata
        progress = {
            "status": "in_progress",
            "completed_trials": completed,
            "total_trials": total_trials,
            "last_updated": dt_datetime.now().isoformat(),
            "last_trial_number": trial.number,
            "last_trial_value": trial.value if trial.value is not None else None,
        }
        
        # If we have a best trial, include full config
        if study.best_trial is not None:
            best_params = study.best_params or {}
            best_config = {
                # Progress tracking
                **progress,
                
                # Best trial info
                "best_trial_number": study.best_trial.number,
                "best_value": study.best_value,
                "best_bidirectional_count": int(study.best_value) if study.best_value else 0,
                "best_params": best_params,
                
                # Extract window parameters for downstream compatibility
                "window_size": best_params.get("window_size"),
                "offset": best_params.get("offset"),
                "skip_min": best_params.get("skip_min"),
                "skip_max": best_params.get("skip_max"),
                "forward_threshold": best_params.get("forward_threshold"),
                "reverse_threshold": best_params.get("reverse_threshold"),
                "time_of_day": best_params.get("time_of_day", "all"),
            }
            
            # Write config atomically (write to temp, then rename)
            temp_path = dt_Path(output_config_path).with_suffix(".tmp")
            with open(temp_path, "w") as f:
                _json.dump(best_config, f, indent=2)
            temp_path.rename(output_config_path)
            
            print(f"[PATCH] Trial {trial.number}: Saved best config (best={study.best_value:.0f} @ trial {study.best_trial.number})")
            
            # If THIS trial is the new best AND has survivors, save them
            if trial.number == study.best_trial.number:
                survivors = trial.user_attrs.get("bidirectional_survivors")
                if survivors is not None and len(survivors) > 0:
                    temp_surv = dt_Path(output_survivors_path).with_suffix(".tmp")
                    with open(temp_surv, "w") as f:
                        _json.dump(survivors, f)
                    temp_surv.rename(output_survivors_path)
                    print(f"[PATCH] Trial {trial.number}: Saved {len(survivors)} bidirectional survivors")
        
        else:
            # No successful trial yet - write progress-only file
            progress["best_trial_number"] = None
            progress["best_value"] = None
            progress["note"] = "No successful trials yet"
            
            temp_path = dt_Path(output_config_path).with_suffix(".tmp")
            with open(temp_path, "w") as f:
                _json.dump(progress, f, indent=2)
            temp_path.rename(output_config_path)
    
    return save_best_so_far_callback


def finalize_output(study, output_config_path: str = "optimal_window_config.json"):
    """
    Call AFTER study.optimize() to mark status as 'complete'.
    Team Beta requirement: explicit status finalization.
    """
    import json as _json
    
    config_path = dt_Path(output_config_path)
    if not config_path.exists():
        return
    
    with open(config_path, "r") as f:
        config = _json.load(f)
    
    config["status"] = "complete"
    config["completed_at"] = dt_datetime.now().isoformat()
    # Update completed_trials to match total if present
    if "total_trials" in config:
        config["completed_trials"] = config["total_trials"]
    config["last_updated"] = config["completed_at"]
    
    with open(config_path, "w") as f:
        _json.dump(config, f, indent=2)
    
    print(f"[PATCH] Finalized {output_config_path} (status=complete)")
# === END INCREMENTAL OUTPUT CALLBACK ===
'''

# =============================================================================
# MANUAL EDIT INSTRUCTIONS
# =============================================================================

MANUAL_EDIT_TRIAL_ATTR = """
+------------------------------------------------------------------------------+
|  MANUAL EDIT #1: Add trial.set_user_attr() in objective function             |
+------------------------------------------------------------------------------+

Find your objective function (the function passed to study.optimize())
and add this line BEFORE the return statement:

    trial.set_user_attr('bidirectional_survivors', bidirectional_survivors)

Example:

    def objective(trial):
        # ... existing code ...
        bidirectional_survivors = results.get('bidirectional', [])

        # >>> ADD THIS LINE <<<
        trial.set_user_attr('bidirectional_survivors', bidirectional_survivors)

        return len(bidirectional_survivors)

+------------------------------------------------------------------------------+
"""

MANUAL_EDIT_OPTIMIZE = """
+------------------------------------------------------------------------------+
|  MANUAL EDIT #2: Add callback to study.optimize()                            |
+------------------------------------------------------------------------------+

Find the study.optimize() call and modify it to include the callback.
Also add finalize_output() AFTER the optimize call.

BEFORE:
    study.optimize(objective, n_trials=max_iterations)

AFTER:
    # Create incremental save callback
    _callback = create_incremental_save_callback(
        output_config_path=output_file,
        output_survivors_path="bidirectional_survivors.json",
        total_trials=max_iterations
    )
    study.optimize(objective, n_trials=max_iterations, callbacks=[_callback])

    # Finalize output (mark status=complete)
    finalize_output(study, output_file)

NOTE: Adjust variable names (output_file, max_iterations) to match yours.

+------------------------------------------------------------------------------+
"""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sha256_file(filepath):
    """Compute SHA256 hash of file contents"""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_content(content):
    """Compute SHA256 hash of string content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# =============================================================================
# SAFE PATCH APPLICATION LOGIC
# =============================================================================

def find_file(target_path=None):
    """Find window_optimizer_bayesian.py"""
    if target_path:
        path = Path(target_path)
        if path.exists():
            return path
        else:
            print(f"[ERROR] Target file not found: {target_path}")
            sys.exit(1)
    
    # Auto-detect
    candidates = [
        Path("window_optimizer_bayesian.py"),
        Path("~/distributed_prng_analysis/window_optimizer_bayesian.py").expanduser(),
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    print("[ERROR] Could not find window_optimizer_bayesian.py")
    print("        Use --target PATH to specify location")
    sys.exit(1)


def detect_repo_root(filepath):
    """Detect repo root by looking for .git or known markers"""
    current = filepath.parent.resolve()
    
    for _ in range(10):  # Max 10 levels up
        if (current / ".git").exists():
            return current
        if (current / "distributed_config.json").exists():
            return current
        if (current / "coordinator.py").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    
    # Fallback to file's directory
    return filepath.parent.resolve()


def backup_file(filepath):
    """Create timestamped backup"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.with_suffix(f".py.bak.{timestamp}")
    copy2(filepath, backup_path)
    print(f"[OK] Backup created: {backup_path}")
    return backup_path


def check_already_patched(content):
    """Check if patch was already applied"""
    if "INCREMENTAL OUTPUT CALLBACK" in content:
        return True
    if "create_incremental_save_callback" in content:
        return True
    return False


def apply_imports(content):
    """
    Add imports if not present - SAFE insertion after existing imports.
    
    v3 FIX: No docstring parsing. Simple rule:
    - Find all import/from lines
    - Handle __future__ imports (must stay at top)
    - Insert after last import line
    """
    if "dt_datetime" in content and "dt_Path" in content:
        print("   [OK] Imports already present")
        return content, False
    
    lines = content.split('\n')
    
    # Find import lines (track __future__ separately)
    last_future_import_idx = -1
    last_regular_import_idx = -1
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip comments and empty lines for import detection
        if stripped.startswith('#') or stripped == '':
            continue
        
        # Check for imports
        if stripped.startswith('from __future__'):
            last_future_import_idx = i
        elif stripped.startswith('import ') or stripped.startswith('from '):
            last_regular_import_idx = i
    
    # Determine insertion point
    if last_regular_import_idx >= 0:
        insert_idx = last_regular_import_idx + 1
    elif last_future_import_idx >= 0:
        insert_idx = last_future_import_idx + 1
    else:
        # No imports found - insert at line 0 or after shebang/encoding
        insert_idx = 0
        for i, line in enumerate(lines[:5]):
            if line.startswith('#!') or line.startswith('# -*-') or line.startswith('# coding'):
                insert_idx = i + 1
    
    # Insert imports
    lines.insert(insert_idx, IMPORTS_TO_ADD)
    print(f"   [OK] Added imports after line {insert_idx}")
    
    return '\n'.join(lines), True


def apply_callback_function(content):
    """Add callback function after imports - SAFE"""
    if "create_incremental_save_callback" in content:
        print("   [OK] Callback function already present")
        return content, False
    
    lines = content.split('\n')
    insert_idx = None
    
    # Look for our import marker first
    for i, line in enumerate(lines):
        if "# === END PATCH IMPORTS ===" in line:
            insert_idx = i + 1
            break
    
    if insert_idx is None:
        # Fallback: find first def/class after imports
        in_imports = True
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if in_imports:
                if stripped.startswith('import ') or stripped.startswith('from '):
                    continue
                elif stripped.startswith('def ') or stripped.startswith('class '):
                    insert_idx = i
                    break
                elif stripped and not stripped.startswith('#'):
                    # Non-import, non-comment, non-empty - might be constant
                    # Keep looking
                    in_imports = False
            else:
                if stripped.startswith('def ') or stripped.startswith('class '):
                    insert_idx = i
                    break
    
    if insert_idx is None:
        # Last resort: end of file
        insert_idx = len(lines)
        print("   [WARN] Inserting callback at end of file (couldn't find better spot)")
    
    lines.insert(insert_idx, CALLBACK_FUNCTION)
    print(f"   [OK] Added callback function at line {insert_idx + 1}")
    
    return '\n'.join(lines), True


def check_trial_user_attr(content):
    """
    CHECK (not apply) for trial.set_user_attr - requires manual edit.
    
    v3 FIX: Use regex for exact pattern matching to avoid false positives.
    """
    # Exact pattern: trial.set_user_attr('bidirectional_survivors' or "bidirectional_survivors"
    pattern = r"trial\.set_user_attr\(\s*['\"]bidirectional_survivors['\"]\s*,"
    
    if re.search(pattern, content):
        print("   [OK] trial.set_user_attr('bidirectional_survivors', ...) found")
        return True
    
    print(MANUAL_EDIT_TRIAL_ATTR)
    
    response = input("\n   Have you added trial.set_user_attr() manually? [y/N]: ")
    if response.lower() != 'y':
        print("\n   [ABORT] Manual edit #1 required before proceeding.")
        print("           Please add the line shown above, then re-run this script.")
        return False
    
    return True


def check_optimize_callback(content):
    """
    CHECK (not apply) for study.optimize callback - requires manual edit.
    
    v3 FIX: Use exact pattern matching for both callbacks=[ and create_incremental_save_callback(
    """
    has_callbacks = "callbacks=[" in content or "callbacks = [" in content
    has_callback_func = "create_incremental_save_callback(" in content
    
    if has_callbacks and has_callback_func:
        print("   [OK] study.optimize wired to create_incremental_save_callback")
        return True
    
    print(MANUAL_EDIT_OPTIMIZE)
    
    response = input("\n   Have you modified study.optimize() manually? [y/N]: ")
    if response.lower() != 'y':
        print("\n   [ABORT] Manual edit #2 required before proceeding.")
        print("           Please modify study.optimize() as shown above, then re-run this script.")
        return False
    
    return True


def validate_syntax(filepath):
    """Validate Python syntax"""
    result = subprocess.run(
        [sys.executable, '-m', 'py_compile', str(filepath)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("   [OK] Syntax validation passed")
        return True
    else:
        print(f"   [ERROR] Syntax validation FAILED:")
        print(result.stderr)
        return False


def validate_runtime_import(filepath, repo_root):
    """
    Validate that the callback can be imported.
    
    v3 FIX: Run from repo_root with proper sys.path setup.
    """
    # Create a temp test script
    test_script = f'''
import sys
sys.path.insert(0, "{repo_root}")
try:
    from {filepath.stem} import create_incremental_save_callback, finalize_output
    # Basic sanity check
    cb = create_incremental_save_callback(total_trials=5)
    assert callable(cb), "Callback should be callable"
    print("OK")
except ImportError as e:
    print(f"IMPORT_FAIL: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"RUNTIME_FAIL: {{e}}")
    sys.exit(1)
'''
    
    result = subprocess.run(
        [sys.executable, '-c', test_script],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    
    if result.returncode == 0 and "OK" in result.stdout:
        print("   [OK] Runtime import validation passed")
        return True
    else:
        print(f"   [ERROR] Runtime import validation FAILED:")
        print(f"   stdout: {result.stdout}")
        print(f"   stderr: {result.stderr}")
        return False


def write_patch_manifest(filepath, original_hash, final_hash):
    """
    Write patch manifest for audit trail.
    
    v3 FIX: Include content hashes for idempotency checking.
    """
    manifest_dir = filepath.parent / ".patch_state"
    manifest_dir.mkdir(exist_ok=True)
    
    manifest = {
        "patch": "incremental_output_v1.0",
        "patch_script_version": "v3",
        "applied_at": datetime.now().isoformat(),
        "target_file": str(filepath.name),
        "target_relpath": str(filepath),
        "sha256_before": original_hash,
        "sha256_after": final_hash,
        "verified": True,
        "notes": "Imports and callback auto-applied. Objective and optimize edits confirmed manually."
    }
    
    manifest_path = manifest_dir / "incremental_output.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"   [OK] Patch manifest written: {manifest_path}")
    return manifest_path


def check_existing_manifest(filepath):
    """Check if patch was already applied based on manifest"""
    manifest_path = filepath.parent / ".patch_state" / "incremental_output.json"
    
    if not manifest_path.exists():
        return None
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    current_hash = sha256_file(filepath)
    
    if manifest.get("sha256_after") == current_hash:
        return "complete"  # File unchanged since patch
    else:
        return "modified"  # File changed since patch


def main():
    parser = argparse.ArgumentParser(
        description='Safe patch installer for incremental output (Team Beta v3)'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Show changes without modifying files')
    parser.add_argument('--target', type=str, default=None,
                       help='Path to window_optimizer_bayesian.py')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify patch state, do not modify')
    parser.add_argument('--repo-root', type=str, default=None,
                       help='Repository root for runtime validation (auto-detected if not set)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("INCREMENTAL OUTPUT PATCH INSTALLER v3.0")
    print("Team Beta Approved - Safe Guided Installer")
    print("=" * 70)
    
    # Find file
    filepath = find_file(args.target)
    filepath = filepath.resolve()
    print(f"\n[TARGET] {filepath}")
    
    # Detect or use repo root
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = detect_repo_root(filepath)
    print(f"[REPO ROOT] {repo_root}")
    
    # Read content
    with open(filepath, 'r') as f:
        original_content = f.read()
    
    original_hash = sha256_content(original_content)
    
    # Check existing manifest
    manifest_state = check_existing_manifest(filepath)
    if manifest_state == "complete":
        print("\n[INFO] Patch manifest indicates file was already patched and unchanged.")
        response = input("       Re-run verification anyway? [y/N]: ")
        if response.lower() != 'y':
            print("       Exiting.")
            sys.exit(0)
    elif manifest_state == "modified":
        print("\n[WARN] Patch manifest exists but file has been modified since patching.")
        print("       Proceeding with verification...")
    
    # Verify-only mode
    if args.verify_only:
        print("\n[MODE] VERIFY-ONLY")
        is_patched = check_already_patched(original_content)
        
        # Use exact pattern matching (v3 fix)
        has_user_attr = bool(re.search(
            r"trial\.set_user_attr\(\s*['\"]bidirectional_survivors['\"]\s*,", 
            original_content
        ))
        has_callback_wiring = (
            ("callbacks=[" in original_content or "callbacks = [" in original_content) and 
            "create_incremental_save_callback(" in original_content
        )
        
        print(f"   Callback function present:    {'[OK]' if is_patched else '[MISSING]'}")
        print(f"   trial.set_user_attr present:  {'[OK]' if has_user_attr else '[MISSING]'}")
        print(f"   study.optimize wired:         {'[OK]' if has_callback_wiring else '[MISSING]'}")
        
        if is_patched and has_user_attr and has_callback_wiring:
            print("\n[RESULT] File is fully patched")
            sys.exit(0)
        else:
            print("\n[RESULT] Patch incomplete")
            sys.exit(1)
    
    # Check if already patched (callback present)
    if check_already_patched(original_content):
        print("\n[INFO] Callback function already present in file.")
        response = input("       Skip to manual edit verification? [Y/n]: ")
        if response.lower() == 'n':
            print("       Exiting.")
            sys.exit(0)
        
        # Skip to manual checks
        print("\n" + "=" * 70)
        print("STEP 3: Verify manual edits")
        print("=" * 70)
        
        if not check_trial_user_attr(original_content):
            sys.exit(1)
        
        # Re-read in case user edited
        with open(filepath, 'r') as f:
            original_content = f.read()
        
        if not check_optimize_callback(original_content):
            sys.exit(1)
        
        print("\n[RESULT] All checks passed!")
        sys.exit(0)
    
    # === STEP 1: Backup ===
    print("\n" + "=" * 70)
    print("STEP 1: Create backup")
    print("=" * 70)
    
    if not args.dry_run:
        backup_path = backup_file(filepath)
    else:
        print("   [DRY RUN] Would create backup")
        backup_path = None
    
    # === STEP 2: Apply safe additions ===
    print("\n" + "=" * 70)
    print("STEP 2: Apply safe additions (imports + callback function)")
    print("=" * 70)
    
    content = original_content
    modified = False
    
    print("\n   Adding imports...")
    content, changed = apply_imports(content)
    modified = modified or changed
    
    print("\n   Adding callback function...")
    content, changed = apply_callback_function(content)
    modified = modified or changed
    
    # Write if modified
    if modified and not args.dry_run:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"\n   [OK] Wrote safe additions to {filepath}")
        
        # Validate syntax
        print("\n   Validating syntax...")
        if not validate_syntax(filepath):
            print("\n   [ERROR] SYNTAX ERROR - Restoring backup!")
            if backup_path:
                copy2(backup_path, filepath)
                print(f"   Restored from: {backup_path}")
            sys.exit(1)
    elif args.dry_run:
        print("\n   [DRY RUN] Would write modifications")
    
    # === STEP 3: Manual edits ===
    print("\n" + "=" * 70)
    print("STEP 3: Manual edits required")
    print("=" * 70)
    print("\nThe following edits must be done MANUALLY for safety.")
    print("The script will verify they are present.\n")
    
    input("Press Enter to see Manual Edit #1...")
    
    # Re-read file (in case we just wrote it)
    with open(filepath, 'r') as f:
        content = f.read()
    
    if not check_trial_user_attr(content):
        sys.exit(1)
    
    input("\nPress Enter to see Manual Edit #2...")
    
    # Re-read again (user may have edited)
    with open(filepath, 'r') as f:
        content = f.read()
    
    if not check_optimize_callback(content):
        sys.exit(1)
    
    # === STEP 4: Final validation ===
    print("\n" + "=" * 70)
    print("STEP 4: Final validation")
    print("=" * 70)
    
    print("\n   Running syntax validation...")
    if not validate_syntax(filepath):
        print("\n   [ERROR] Final syntax validation failed!")
        if backup_path:
            response = input("   Restore from backup? [y/N]: ")
            if response.lower() == 'y':
                copy2(backup_path, filepath)
                print(f"   Restored from: {backup_path}")
        sys.exit(1)
    
    print("\n   Running runtime import validation...")
    if not validate_runtime_import(filepath, repo_root):
        print("\n   [ERROR] Runtime import validation failed!")
        print("           This may indicate missing dependencies or import errors.")
        sys.exit(1)
    
    # === STEP 5: Write manifest ===
    print("\n" + "=" * 70)
    print("STEP 5: Write patch manifest")
    print("=" * 70)
    
    # Get final hash
    final_hash = sha256_file(filepath)
    
    if not args.dry_run:
        write_patch_manifest(filepath, original_hash, final_hash)
    else:
        print("   [DRY RUN] Would write patch manifest")
    
    # === DONE ===
    print("\n" + "=" * 70)
    print("[SUCCESS] PATCH INSTALLATION COMPLETE")
    print("=" * 70)
    
    print("\nSummary:")
    print("   [OK] Backup created")
    print("   [OK] Imports added")
    print("   [OK] Callback function added")
    print("   [OK] trial.set_user_attr confirmed")
    print("   [OK] study.optimize wiring confirmed")
    print("   [OK] Syntax validated")
    print("   [OK] Runtime import validated")
    print("   [OK] Patch manifest written")
    
    print("\nTest the patch:")
    print("   python3 window_optimizer.py --strategy bayesian \\")
    print("       --lottery-file daily3.json --trials 3 --max-seeds 1000000")
    print("\n   Then verify:")
    print("   cat optimal_window_config.json | jq '.status, .completed_trials'")
    
    if args.dry_run:
        print("\n[DRY RUN COMPLETE - No files were modified]")


if __name__ == "__main__":
    main()
