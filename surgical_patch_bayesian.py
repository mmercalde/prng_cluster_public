#!/usr/bin/env python3
"""
SURGICAL PATCH: Incremental Output for window_optimizer_bayesian.py
====================================================================
Exact string replacements - no heuristics, no pattern matching.

This script performs 4 precise edits:
1. Add imports after existing imports
2. Add callback function before OptunaBayesianSearch class
3. Add trial.set_user_attr in optuna_objective (stores result for callback)
4. Add callback to study.optimize() + finalize_output() after

Usage:
    python3 surgical_patch_bayesian.py [--dry-run]
"""

import sys
from pathlib import Path
from datetime import datetime
from shutil import copy2

TARGET_FILE = Path("window_optimizer_bayesian.py")

# =============================================================================
# EXACT STRINGS TO FIND AND REPLACE
# =============================================================================

# PATCH 1: Add imports after "SKLEARN_GP_AVAILABLE = False"
FIND_1 = '''    SKLEARN_GP_AVAILABLE = False
    print("⚠️  scikit-learn GP not available")'''

REPLACE_1 = '''    SKLEARN_GP_AVAILABLE = False
    print("⚠️  scikit-learn GP not available")


# === INCREMENTAL OUTPUT IMPORTS (Patch 2026-01-18) ===
from datetime import datetime as _patch_datetime
from pathlib import Path as _patch_Path
import json as _patch_json
# === END INCREMENTAL IMPORTS ==='''


# PATCH 2: Add callback function before the OptunaBayesianSearch class
FIND_2 = '''# ============================================================================
# OPTUNA-BASED BAYESIAN OPTIMIZATION (PREFERRED)
# ============================================================================

class OptunaBayesianSearch:'''

REPLACE_2 = '''# ============================================================================
# INCREMENTAL OUTPUT CALLBACK (Patch 2026-01-18)
# ============================================================================

def create_incremental_save_callback(
    output_config_path: str = "optimal_window_config.json",
    output_survivors_path: str = "bidirectional_survivors.json",
    total_trials: int = 50
):
    """
    Optuna callback that saves best-so-far results after each trial.
    Ensures crash recovery and WATCHER visibility.
    """
    def save_best_so_far(study, trial):
        completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
        
        progress = {
            "status": "in_progress",
            "completed_trials": completed,
            "total_trials": total_trials,
            "last_updated": _patch_datetime.now().isoformat(),
            "last_trial_number": trial.number,
            "last_trial_value": trial.value,
        }
        
        if study.best_trial is not None:
            best_params = study.best_params or {}
            best_config = {
                **progress,
                "best_trial_number": study.best_trial.number,
                "best_value": study.best_value,
                "best_bidirectional_count": int(study.best_value) if study.best_value and study.best_value > 0 else 0,
                "best_params": best_params,
                "window_size": best_params.get("window_size"),
                "offset": best_params.get("offset"),
                "skip_min": best_params.get("skip_min"),
                "skip_max": best_params.get("skip_max"),
                "forward_threshold": best_params.get("forward_threshold"),
                "reverse_threshold": best_params.get("reverse_threshold"),
            }
            
            # Atomic write
            temp_path = _patch_Path(output_config_path).with_suffix(".tmp")
            with open(temp_path, "w") as f:
                _patch_json.dump(best_config, f, indent=2)
            temp_path.rename(output_config_path)
            
            print(f"[SAVE] Trial {trial.number}: config saved (best={study.best_value:.0f} @ trial {study.best_trial.number})")
            
            # Save survivors if this trial stored them and is the new best
            if trial.number == study.best_trial.number:
                survivors = trial.user_attrs.get("bidirectional_survivors")
                if survivors and len(survivors) > 0:
                    temp_surv = _patch_Path(output_survivors_path).with_suffix(".tmp")
                    with open(temp_surv, "w") as f:
                        _patch_json.dump(survivors, f)
                    temp_surv.rename(output_survivors_path)
                    print(f"[SAVE] Trial {trial.number}: {len(survivors)} survivors saved")
        else:
            progress["note"] = "No successful trials yet"
            temp_path = _patch_Path(output_config_path).with_suffix(".tmp")
            with open(temp_path, "w") as f:
                _patch_json.dump(progress, f, indent=2)
            temp_path.rename(output_config_path)
    
    return save_best_so_far


def finalize_incremental_output(study, output_config_path: str = "optimal_window_config.json"):
    """Mark output as complete after study.optimize() finishes."""
    config_path = _patch_Path(output_config_path)
    if not config_path.exists():
        return
    
    with open(config_path, "r") as f:
        config = _patch_json.load(f)
    
    config["status"] = "complete"
    config["completed_at"] = _patch_datetime.now().isoformat()
    
    with open(config_path, "w") as f:
        _patch_json.dump(config, f, indent=2)
    
    print(f"[SAVE] Finalized {output_config_path} (status=complete)")


# ============================================================================
# OPTUNA-BASED BAYESIAN OPTIMIZATION (PREFERRED)
# ============================================================================

class OptunaBayesianSearch:'''


# PATCH 3: Store result in trial.user_attrs (add before "return score")
FIND_3 = '''            # Track best
            nonlocal best_result, best_score
            if score > best_score:'''

REPLACE_3 = '''            # Store result data for incremental callback
            trial.set_user_attr("bidirectional_survivors", 
                               getattr(result, 'bidirectional_survivors', []))
            trial.set_user_attr("result_dict", result.to_dict())
            
            # Track best
            nonlocal best_result, best_score
            if score > best_score:'''


# PATCH 4: Add callback to study.optimize() and finalize after
FIND_4 = '''        # Run optimization
        study.optimize(optuna_objective, n_trials=max_iterations)
        
        # Print summary'''

REPLACE_4 = '''        # Run optimization with incremental save callback
        _incremental_callback = create_incremental_save_callback(
            output_config_path="optimal_window_config.json",
            output_survivors_path="bidirectional_survivors.json",
            total_trials=max_iterations
        )
        study.optimize(optuna_objective, n_trials=max_iterations, callbacks=[_incremental_callback])
        
        # Finalize output (mark status=complete)
        finalize_incremental_output(study, "optimal_window_config.json")
        
        # Print summary'''


# =============================================================================
# PATCH APPLICATION
# =============================================================================

def main():
    dry_run = "--dry-run" in sys.argv
    
    print("=" * 70)
    print("SURGICAL PATCH: Incremental Output for window_optimizer_bayesian.py")
    print("=" * 70)
    
    if not TARGET_FILE.exists():
        print(f"\n[ERROR] File not found: {TARGET_FILE}")
        print("        Run this script from ~/distributed_prng_analysis/")
        sys.exit(1)
    
    # Read original
    with open(TARGET_FILE, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "INCREMENTAL OUTPUT CALLBACK" in content:
        print("\n[INFO] File already patched!")
        print("       Use --verify to check patch status.")
        sys.exit(0)
    
    # Backup
    if not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = TARGET_FILE.with_suffix(f".py.bak.{timestamp}")
        copy2(TARGET_FILE, backup_path)
        print(f"\n[OK] Backup: {backup_path}")
    else:
        print("\n[DRY-RUN] Would create backup")
    
    # Apply patches
    patches = [
        ("Patch 1: Add imports", FIND_1, REPLACE_1),
        ("Patch 2: Add callback function", FIND_2, REPLACE_2),
        ("Patch 3: Store result in trial", FIND_3, REPLACE_3),
        ("Patch 4: Wire callback to study.optimize()", FIND_4, REPLACE_4),
    ]
    
    patched_content = content
    for name, find, replace in patches:
        if find not in patched_content:
            print(f"\n[ERROR] {name}: Target string not found!")
            print(f"        Expected to find:\n{find[:100]}...")
            print("\n        File may have been modified. Aborting.")
            sys.exit(1)
        
        patched_content = patched_content.replace(find, replace, 1)
        print(f"[OK] {name}")
    
    # Write patched file
    if not dry_run:
        with open(TARGET_FILE, 'w') as f:
            f.write(patched_content)
        print(f"\n[OK] Wrote patched file: {TARGET_FILE}")
        
        # Validate syntax
        print("\n[...] Validating syntax...")
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', str(TARGET_FILE)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[ERROR] Syntax error!\n{result.stderr}")
            print("\n[RESTORE] Restoring backup...")
            copy2(backup_path, TARGET_FILE)
            print(f"[OK] Restored from {backup_path}")
            sys.exit(1)
        print("[OK] Syntax valid")
        
        # Quick import test
        print("[...] Testing import...")
        test_result = subprocess.run(
            [sys.executable, '-c', 
             'from window_optimizer_bayesian import create_incremental_save_callback; print("OK")'],
            capture_output=True, text=True, cwd=str(TARGET_FILE.parent)
        )
        if "OK" in test_result.stdout:
            print("[OK] Import test passed")
        else:
            print(f"[WARN] Import test issue: {test_result.stderr}")
    else:
        print("\n[DRY-RUN] Would write patched file")
        print("\n--- DIFF PREVIEW ---")
        print(f"Lines before: {len(content.splitlines())}")
        print(f"Lines after:  {len(patched_content.splitlines())}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Patch applied!")
    print("=" * 70)
    print("\nTest with:")
    print("  python3 window_optimizer.py --strategy bayesian \\")
    print("      --lottery-file daily3.json --trials 3 --max-seeds 1000000")
    print("\nVerify incremental output:")
    print("  cat optimal_window_config.json | jq '.status, .completed_trials'")


if __name__ == "__main__":
    main()
