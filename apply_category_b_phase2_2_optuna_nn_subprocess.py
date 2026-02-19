#!/usr/bin/env python3
"""
apply_category_b_phase2_2_optuna_nn_subprocess.py
==================================================

Phase 2.2: Route Optuna multi-trial NN through train_single_trial.py subprocess.

WHAT THIS DOES:
  1. Adds _run_nn_optuna_trial() helper method for Optuna fold training
  2. Patches _optuna_objective() to route NN through subprocess (Spot 1)
  3. Patches _run_optuna_optimization() final model to use subprocess (Spot 2)
  4. Patches _run_nn_via_subprocess() to accept optional params/trial args
  5. Patches save_best_model() gate to accept disk-first in any mode (TB #1)
  6. Patches study_name to add suffix for normalized NN studies (TB #2)

WHAT THIS DOES NOT CHANGE:
  - Tree model Optuna paths (lightgbm, xgboost, catboost) — unchanged
  - train_single_trial.py — zero modifications
  - _s88_run_compare_models() — unchanged
  - _run_single_model() single-shot branch — unchanged (Phase 2.1)
  - Step 6, WATCHER, or any other pipeline component

PREREQUISITES:
  - Phase 2.1 deployed (commits dd34310, 3c8afca, 3dac87d)
  - _export_split_npz() exists (line ~1793)
  - _run_nn_via_subprocess() exists (line ~1818)
  - NN_SUBPROCESS_ROUTING_ENABLED = True (line ~348)

Team Beta Corrections Applied:
  TB #1: save_best_model() gate — remove compare_models requirement
  TB #2: Fresh Optuna study name suffix for normalized NN trials
  TB Trim #1: Diagnostics only on final model, not per fold
  TB Trim #2: NPZ per fold (simple, delete in finally)

Author: Team Alpha (S94)
Date: 2026-02-16
"""

import re
import sys
import os
import shutil
from datetime import datetime


TARGET = "meta_prediction_optimizer_anti_overfit.py"


def backup(path):
    """Create timestamped backup."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = f"{path}.pre_phase2_2_{ts}"
    shutil.copy2(path, dst)
    print(f"  ✅ Backup: {dst}")
    return dst


def verify_prerequisites(content):
    """Verify Phase 2.1 infrastructure is present."""
    checks = {
        "_export_split_npz": "def _export_split_npz(",
        "_run_nn_via_subprocess": "def _run_nn_via_subprocess(",
        "NN_SUBPROCESS_ROUTING_ENABLED": "NN_SUBPROCESS_ROUTING_ENABLED",
        "train_single_trial.py reference": "train_single_trial.py",
        "_optuna_objective": "def _optuna_objective(self, trial)",
        "_run_optuna_optimization": "def _run_optuna_optimization(",
        "save_best_model": "def save_best_model(self):",
    }
    
    all_ok = True
    for name, pattern in checks.items():
        if pattern not in content:
            print(f"  ❌ Missing prerequisite: {name} (pattern: {pattern})")
            all_ok = False
        else:
            print(f"  ✅ Found: {name}")
    
    return all_ok


def apply_patch_1_new_helper_method(content):
    """
    Add _run_nn_optuna_trial() method BEFORE _optuna_objective().
    
    This helper runs a single NN fold via subprocess without --save-model.
    Returns metrics dict matching inline trainer interface.
    """
    
    new_method = '''
    def _run_nn_optuna_trial(self, X_train, y_train, X_val, y_val,
                             config, trial_number, fold_idx):
        """
        Phase 2.2: Run single NN Optuna fold via train_single_trial.py subprocess.
        
        Unlike _run_nn_via_subprocess(), this does NOT save a model checkpoint.
        Optuna trials are exploratory; only the final model is saved.
        
        Returns dict matching MultiModelTrainer.train_model() result schema.
        """
        npz_path = self._export_split_npz(X_train, y_train, X_val, y_val)
        
        try:
            cmd = [
                sys.executable, "train_single_trial.py",
                "--model-type", "neural_net",
                "--data-path", npz_path,
                "--params", json.dumps(config),
                "--trial-number", str(trial_number),
                "--normalize-features",
                "--use-leaky-relu",
            ]
            
            # Thread dropout if provided via CLI
            if getattr(self, '_cli_dropout', None) is not None:
                try:
                    cmd.extend(["--dropout", str(float(self._cli_dropout))])
                except Exception:
                    pass
            
            # TB Trim #1: No --enable-diagnostics on Optuna folds
            # (diagnostics only on final model to avoid 100-file explosion)
            
            sub_env = os.environ.copy()
            
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=600,
                cwd=str(Path(__file__).parent)
            )
            
            # Parse JSON from last stdout line
            r2 = -999.0
            train_mse = 0.0
            val_mse = float('inf')
            
            if proc.returncode == 0:
                try:
                    for line in (proc.stdout or "").strip().split("\\n"):
                        line = line.strip()
                        if line.startswith("{") and line.endswith("}"):
                            output = json.loads(line)
                            r2 = float(output.get("r2", -999.0))
                            train_mse = float(output.get("train_mse", 0.0))
                            val_mse = float(output.get("val_mse", float('inf')))
                            break
                except Exception as parse_err:
                    self.logger.warning(
                        f"[Phase 2.2] Could not parse subprocess output "
                        f"(trial {trial_number} fold {fold_idx}): {parse_err}"
                    )
            else:
                stderr_tail = (proc.stderr or "")[-300:]
                self.logger.warning(
                    f"[Phase 2.2] Subprocess failed (trial {trial_number} fold {fold_idx}, "
                    f"rc={proc.returncode}): {stderr_tail}"
                )
            
            return {
                "model": None,
                "model_type": "neural_net",
                "metrics": {
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "r2": r2,
                },
                "hyperparameters": config,
            }
            
        finally:
            # Cleanup NPZ (TB Trim #2)
            try:
                os.remove(npz_path)
            except OSError:
                pass

'''
    
    # Insert before _optuna_objective
    anchor = "    def _optuna_objective(self, trial) -> float:"
    if anchor not in content:
        raise ValueError(f"Anchor not found: {anchor}")
    
    content = content.replace(anchor, new_method + anchor)
    print("  ✅ Patch 1: Added _run_nn_optuna_trial() method")
    return content


def apply_patch_2_optuna_objective(content):
    """
    Patch _optuna_objective() to route NN through subprocess (Spot 1).
    
    Replace the inline trainer call with conditional routing.
    """
    
    old_block = '''        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X_train_val)):
            result = self.trainer.train_model(
                self.model_type,
                self.X_train_val[train_idx], self.y_train_val[train_idx],
                self.X_train_val[val_idx], self.y_train_val[val_idx],
                hyperparameters=config
            )
            fold_r2.append(float(result["metrics"].get("r2", 0.0)))'''
    
    new_block = '''        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X_train_val)):
            X_tr = self.X_train_val[train_idx]
            y_tr = self.y_train_val[train_idx]
            X_vl = self.X_train_val[val_idx]
            y_vl = self.y_train_val[val_idx]

            if self.model_type == "neural_net" and NN_SUBPROCESS_ROUTING_ENABLED:
                # Phase 2.2: Route NN through train_single_trial.py subprocess
                result = self._run_nn_optuna_trial(
                    X_tr, y_tr, X_vl, y_vl, config, trial.number, fold_idx
                )
            else:
                # Tree models: inline trainer (unchanged)
                result = self.trainer.train_model(
                    self.model_type, X_tr, y_tr, X_vl, y_vl,
                    hyperparameters=config
                )
            fold_r2.append(float(result["metrics"].get("r2", 0.0)))'''
    
    if old_block not in content:
        raise ValueError("Patch 2 anchor not found: _optuna_objective inline trainer block")
    
    content = content.replace(old_block, new_block)
    print("  ✅ Patch 2: _optuna_objective() now routes NN through subprocess")
    return content


def apply_patch_3_final_model(content):
    """
    Patch _run_optuna_optimization() final model training (Spot 2).
    
    Replace inline trainer with conditional subprocess routing for NN.
    Uses regex to handle whitespace variations.
    """
    
    # Use regex to match the block robustly (handles trailing whitespace, etc.)
    pattern = (
        r'( *)result = self\.trainer\.train_model\(\s*\n'
        r'\s*self\.model_type, X_train, y_train, X_val, y_val,\s*\n'
        r'\s*hyperparameters=self\.best_config\s*\n'
        r'\s*\)\s*\n'
        r'(\s*)self\.best_model = result\["model"\]\s*\n'
        r'(\s*)self\.best_model_type = self\.model_type\s*\n'
        r'(\s*)self\.best_metrics = self\._compute_final_metrics\(result\["metrics"\]\)'
    )
    
    match = re.search(pattern, content)
    if not match:
        raise ValueError("Patch 3 anchor not found: final model trainer block (regex)")
    
    indent = match.group(1)  # Capture leading indentation
    
    replacement = (
        f'{indent}if self.model_type == "neural_net" and NN_SUBPROCESS_ROUTING_ENABLED:\n'
        f'{indent}    # Phase 2.2: Final NN model via subprocess (enriched checkpoint)\n'
        f'{indent}    result = self._run_nn_via_subprocess(\n'
        f'{indent}        X_train, y_train, X_val, y_val,\n'
        f'{indent}        hyperparameters=self.best_config\n'
        f'{indent}    )\n'
        f'{indent}    self.best_model = None  # Disk-first: model is checkpoint on disk\n'
        f'{indent}else:\n'
        f'{indent}    result = self.trainer.train_model(\n'
        f'{indent}        self.model_type, X_train, y_train, X_val, y_val,\n'
        f'{indent}        hyperparameters=self.best_config\n'
        f'{indent}    )\n'
        f'{indent}    self.best_model = result["model"]\n'
        f'{indent}self.best_model_type = self.model_type\n'
        f'{indent}self.best_metrics = self._compute_final_metrics(result["metrics"])'
    )
    
    content = content[:match.start()] + replacement + content[match.end():]
    print("  ✅ Patch 3: Final model training now routes NN through subprocess")
    return content


def apply_patch_4_subprocess_params(content):
    """
    Patch _run_nn_via_subprocess() to accept optional hyperparameters.
    
    When called from Optuna final model, we need to pass the best config
    as --params so the subprocess uses Optuna-selected hyperparameters
    instead of defaults.
    """
    
    old_sig = "    def _run_nn_via_subprocess(self, X_train, y_train, X_val, y_val):"
    new_sig = "    def _run_nn_via_subprocess(self, X_train, y_train, X_val, y_val, hyperparameters=None):"
    
    if old_sig not in content:
        raise ValueError("Patch 4a anchor not found: _run_nn_via_subprocess signature")
    
    content = content.replace(old_sig, new_sig)
    
    # Add --params injection after the --verbose line
    old_verbose = '''                "--verbose",
            ]'''
    
    new_verbose = '''                "--verbose",
            ]
            
            # Phase 2.2: Pass hyperparameters if provided (Optuna best config)
            if hyperparameters:
                cmd.extend(["--params", json.dumps(hyperparameters)])'''
    
    if old_verbose not in content:
        raise ValueError("Patch 4b anchor not found: --verbose block in _run_nn_via_subprocess")
    
    content = content.replace(old_verbose, new_verbose, 1)  # Replace only first occurrence
    print("  ✅ Patch 4: _run_nn_via_subprocess() now accepts hyperparameters arg")
    return content


def apply_patch_5_save_gate(content):
    """
    TB Critical Correction #1: Remove compare_models gate from save_best_model().
    
    Before: requires compare_models=True for disk-first sidecar
    After:  any mode with best_checkpoint_path gets disk-first sidecar
    Uses regex for whitespace robustness.
    """
    
    # Match the specific compare_models gate pattern
    pattern = (
        r"if getattr\(self, 'compare_models', False\) and getattr\(self, 'best_checkpoint_path', None\):"
    )
    
    replacement = "if getattr(self, 'best_checkpoint_path', None):"
    
    match = re.search(pattern, content)
    if not match:
        raise ValueError("Patch 5 anchor not found: save_best_model compare_models gate")
    
    # Also add a comment explaining the change
    old_line = match.group(0)
    new_line = (
        "if getattr(self, 'best_checkpoint_path', None):\n"
        "                # Disk-first: checkpoint exists (Optuna NN, compare-models, etc.)"
    )
    
    content = content[:match.start()] + new_line + content[match.end():]
    print("  ✅ Patch 5: save_best_model() gate now accepts disk-first in any mode (TB #1)")
    return content


def apply_patch_6_study_name(content):
    """
    TB Critical Correction #2: Fresh study name for normalized NN trials.
    
    Append '_catb22' suffix when model is NN to avoid contamination
    from old unnormalized trials.
    """
    
    old_study = '        study_name = f"step5_{self.model_type}_{feature_h}_{data_h}"'
    new_study = '''        # TB #2: Fresh study for normalized NN (avoid contaminating old unnormalized trials)
        _study_suffix = "_catb22" if self.model_type == "neural_net" and NN_SUBPROCESS_ROUTING_ENABLED else ""
        study_name = f"step5_{self.model_type}_{feature_h}_{data_h}{_study_suffix}"'''
    
    if old_study not in content:
        raise ValueError("Patch 6 anchor not found: study_name construction")
    
    content = content.replace(old_study, new_study)
    print("  ✅ Patch 6: NN Optuna study name gets '_catb22' suffix (TB #2)")
    return content


def verify_syntax(path):
    """Compile to verify no syntax errors."""
    import py_compile
    try:
        py_compile.compile(path, doraise=True)
        print(f"  ✅ Syntax OK: {path}")
        return True
    except py_compile.PyCompileError as e:
        print(f"  ❌ Syntax error: {e}")
        return False


def verify_patches(content):
    """Verify all patches are present in final content."""
    checks = {
        "Patch 1 (new method)": "def _run_nn_optuna_trial(",
        "Patch 2 (objective routing)": 'if self.model_type == "neural_net" and NN_SUBPROCESS_ROUTING_ENABLED:',
        "Patch 2 (fold variables)": "X_tr = self.X_train_val[train_idx]",
        "Patch 3 (final model routing)": "# Phase 2.2: Final NN model via subprocess",
        "Patch 3 (hyperparameters pass)": "hyperparameters=self.best_config",
        "Patch 4 (params signature)": "def _run_nn_via_subprocess(self, X_train, y_train, X_val, y_val, hyperparameters=None):",
        "Patch 4 (params injection)": '# Phase 2.2: Pass hyperparameters if provided',
        "Patch 5 (gate fix)": "# Disk-first: checkpoint exists (Optuna NN, compare-models, etc.)",
        "Patch 6 (study suffix)": '_catb22',
        "TB Trim #1 (no diag in folds)": "# TB Trim #1: No --enable-diagnostics on Optuna folds",
    }
    
    all_ok = True
    for name, pattern in checks.items():
        if pattern not in content:
            print(f"  ❌ Missing: {name}")
            all_ok = False
        else:
            print(f"  ✅ Verified: {name}")
    
    # Negative checks — make sure old code is gone
    negative_checks = {
        "Old objective inline call removed": (
            "result = self.trainer.train_model(\n"
            "                self.model_type,\n"
            "                self.X_train_val[train_idx]"
        ),
        "Old save gate removed": "getattr(self, 'compare_models', False) and getattr(self, 'best_checkpoint_path'",
    }
    
    for name, pattern in negative_checks.items():
        if pattern in content:
            print(f"  ❌ Old code still present: {name}")
            all_ok = False
        else:
            print(f"  ✅ Confirmed removed: {name}")
    
    return all_ok


def main():
    print("=" * 70)
    print("Category B Phase 2.2: Optuna NN Subprocess Routing")
    print("=" * 70)
    print()
    
    if not os.path.exists(TARGET):
        print(f"❌ Target file not found: {TARGET}")
        print("   Run this from ~/distributed_prng_analysis/")
        return 1
    
    # Step 1: Backup
    print("[1/8] Creating backup...")
    backup_path = backup(TARGET)
    
    # Step 2: Read file
    print("[2/8] Reading source...")
    with open(TARGET) as f:
        content = f.read()
    original_lines = content.count('\n')
    print(f"  ✅ Read {original_lines} lines")
    
    # Step 3: Verify prerequisites
    print("[3/8] Verifying prerequisites...")
    if not verify_prerequisites(content):
        print("❌ Prerequisites not met. Is Phase 2.1 deployed?")
        return 1
    
    # Step 4: Apply patches
    print("[4/8] Applying patches...")
    try:
        content = apply_patch_1_new_helper_method(content)
        content = apply_patch_2_optuna_objective(content)
        content = apply_patch_3_final_model(content)
        content = apply_patch_4_subprocess_params(content)
        content = apply_patch_5_save_gate(content)
        content = apply_patch_6_study_name(content)
    except ValueError as e:
        print(f"\n❌ Patch failed: {e}")
        print(f"   Restoring backup: {backup_path}")
        shutil.copy2(backup_path, TARGET)
        return 1
    
    # Step 5: Write patched file
    print("[5/8] Writing patched file...")
    with open(TARGET, 'w') as f:
        f.write(content)
    new_lines = content.count('\n')
    print(f"  ✅ Written {new_lines} lines (delta: +{new_lines - original_lines})")
    
    # Step 6: Verify syntax
    print("[6/8] Verifying syntax...")
    if not verify_syntax(TARGET):
        print(f"❌ Syntax error! Restoring backup: {backup_path}")
        shutil.copy2(backup_path, TARGET)
        return 1
    
    # Step 7: Verify patches
    print("[7/8] Verifying patches...")
    with open(TARGET) as f:
        final_content = f.read()
    if not verify_patches(final_content):
        print(f"⚠️  Some verification checks failed. Review manually.")
        print(f"   Backup available: {backup_path}")
        # Don't auto-restore — patches may be partially correct
    
    # Step 8: Summary
    print("[8/8] Summary...")
    print()
    print("=" * 70)
    print("✅ Phase 2.2 APPLIED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Backup: {backup_path}")
    print(f"  Lines added: +{new_lines - original_lines}")
    print()
    print("PATCHES APPLIED:")
    print("  1. _run_nn_optuna_trial() — new helper for Optuna fold training")
    print("  2. _optuna_objective() — NN routes through subprocess (Spot 1)")
    print("  3. _run_optuna_optimization() — final model via subprocess (Spot 2)")
    print("  4. _run_nn_via_subprocess() — accepts hyperparameters argument")
    print("  5. save_best_model() gate — disk-first in any mode (TB #1)")
    print("  6. Study name suffix '_catb22' for normalized NN (TB #2)")
    print()
    print("VERIFICATION COMMANDS:")
    print("  python3 -m py_compile meta_prediction_optimizer_anti_overfit.py")
    print()
    print("SMOKE TEST (2 trials, fast):")
    print("  python3 meta_prediction_optimizer_anti_overfit.py \\")
    print("    --survivors survivors_with_scores.json \\")
    print("    --lottery-data train_history.json \\")
    print("    --trials 2 --model-type neural_net \\")
    print("    --enable-diagnostics 2>&1 | tee /tmp/nn_phase2_2_test.log")
    print()
    print("VERIFY LOG SHOULD SHOW:")
    print("  - [CAT-B] Input normalization applied")
    print("  - [CAT-B] Activation: LeakyReLU(0.01)")
    print("  - Subprocess invocations per fold")
    print("  - R² meaningfully better than -6.0 baseline")
    print("  - Study name ending in '_catb22'")
    print()
    print("TREE MODEL REGRESSION TEST:")
    print("  python3 meta_prediction_optimizer_anti_overfit.py \\")
    print("    --survivors survivors_with_scores.json \\")
    print("    --lottery-data train_history.json \\")
    print("    --trials 2 --model-type catboost 2>&1 | tail -20")
    print()
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
