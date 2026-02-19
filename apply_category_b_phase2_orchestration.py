#!/usr/bin/env python3
"""
Category B Phase 2 — Patch meta_prediction_optimizer_anti_overfit.py
====================================================================

Threads NN-specific Category B flags into the subprocess command builder
in _s88_run_compare_models() so that neural_net training gets:
  --normalize-features (always, Option A: default ON for NN)
  --use-leaky-relu (always, default ON for initial evaluation)
  --dropout <float> (only if retry params provide one, best-effort parse)

Also adds Category B CLI flags to argparse for WATCHER pass-through.

NOTE: This patcher does NOT modify _sample_hyperparameters() or Optuna
search space. Per Team Beta Option A, normalize_features and use_leaky_relu
are fixed ON for neural_net and not searched by Optuna.

Verified against live code fetched from:
  github.com/mmercalde/prng_cluster_public/main/meta_prediction_optimizer_anti_overfit.py
  Commit: 3c3f9ae

Author: Team Alpha (S92)
Date: 2026-02-15
"""

import sys
import shutil
from pathlib import Path

TARGET = Path("meta_prediction_optimizer_anti_overfit.py")
BACKUP = TARGET.with_suffix(".pre_category_b_phase2")


def verify_preconditions():
    """Verify the file is in expected state."""
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run from ~/distributed_prng_analysis/")
        return False

    content = TARGET.read_text()

    # Check we haven't already patched
    if '--normalize-features' in content:
        print("ERROR: --normalize-features already present. Already patched?")
        return False

    # Verify exact anchors from live code
    anchors = [
        'def _s88_run_compare_models(args_dict):',
        '# IMPORTANT: ensure compare-models is not set in subcall',
        'if m == "lightgbm":',
        'env["S88_FORCE_LGBM_CPU"] = "1"',
        'def _sample_hyperparameters(self, trial) -> Dict:',
        'if self.model_type == "neural_net":',
    ]
    for anchor in anchors:
        if anchor not in content:
            print(f"ERROR: Missing anchor: {anchor}")
            return False

    print("All preconditions PASSED")
    return True


def apply_patch():
    content = TARGET.read_text()

    # ================================================================
    # PATCH 1: Inject NN flags into subprocess cmd in _s88_run_compare_models
    #
    # Team Beta Required Fix #2: use "is not None" for dropout log
    # Team Beta Required Fix #3: best-effort parse for dropout
    # Team Beta Recommendation A: log that Option A forces flags
    # ================================================================

    old_cmd_block = """        # IMPORTANT: ensure compare-models is not set in subcall
        # (we're in single-model mode)
        env = os.environ.copy()
        env["S88_COMPARE_MODELS_CHILD"] = "1"
        if m == "lightgbm":
            env["S88_FORCE_LGBM_CPU"] = "1"

        print(f"[S88][COMPARE] Running {m} with {trials} trials via single-model Optuna path...")"""

    new_cmd_block = """        # -- Category B: Inject NN-specific flags (Team Beta Option A) --
        # normalize_features and use_leaky_relu are ALWAYS ON for neural_net.
        # These are fixed policy, not CLI-toggleable in compare-models mode.
        # Flags consumed by train_single_trial.py v1.1.0 (Phase 1).
        if m == "neural_net":
            cmd.append("--normalize-features")
            cmd.append("--use-leaky-relu")
            # Thread dropout override if WATCHER retry provided one (best-effort)
            _dropout_override = args_dict.get("dropout_override") or args_dict.get("dropout")
            if _dropout_override is not None:
                try:
                    _d = float(_dropout_override)
                    cmd.extend(["--dropout", str(_d)])
                except Exception:
                    print(f"[CAT-B] Invalid dropout override: {_dropout_override!r} (ignored)")
                    _d = None
            else:
                _d = None
            print(f"[CAT-B] Option A: forcing normalize+leaky for NN"
                  + (f", dropout={_d}" if _d is not None else ""))

        # IMPORTANT: ensure compare-models is not set in subcall
        # (we're in single-model mode)
        env = os.environ.copy()
        env["S88_COMPARE_MODELS_CHILD"] = "1"
        if m == "lightgbm":
            env["S88_FORCE_LGBM_CPU"] = "1"

        print(f"[S88][COMPARE] Running {m} with {trials} trials via single-model Optuna path...")"""

    if old_cmd_block not in content:
        print("ERROR: Cannot find subprocess cmd/env block in _s88_run_compare_models")
        return False

    content = content.replace(old_cmd_block, new_cmd_block)
    print("[1/3] Injected NN Category B flags into _s88_run_compare_models subprocess builder")
    print("      Fix #2: dropout log uses 'is not None' (0.0 correctly logged)")
    print("      Fix #3: dropout parse is best-effort (bad values ignored, not crash)")
    print("      Rec A: log clearly states 'Option A: forcing normalize+leaky for NN'")

    # ================================================================
    # PATCH 2: Add Category B CLI flags to argparse
    # ================================================================

    old_diag_arg = """    # Chapter 14: Training diagnostics
    parser.add_argument('--enable-diagnostics', action='store_true',
                       help='Enable Chapter 14 training diagnostics (writes to diagnostics_outputs/)')"""

    new_diag_arg = """    # Chapter 14: Training diagnostics
    parser.add_argument('--enable-diagnostics', action='store_true',
                       help='Enable Chapter 14 training diagnostics (writes to diagnostics_outputs/)')
    
    # Category B: Neural net training enhancements (passed through to train_single_trial.py)
    # NOTE: In --compare-models mode, Option A forces these ON for neural_net
    # regardless of CLI flags. These flags are for single-model WATCHER pass-through.
    parser.add_argument('--normalize-features', action='store_true',
                       help='Apply StandardScaler normalization before NN training')
    parser.add_argument('--use-leaky-relu', action='store_true',
                       help='Use LeakyReLU(0.01) instead of ReLU in neural net')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Override dropout value for NN (CLI precedence)')"""

    if old_diag_arg not in content:
        print("ERROR: Cannot find --enable-diagnostics argument in CLI")
        return False

    content = content.replace(old_diag_arg, new_diag_arg)
    print("[2/3] Added Category B CLI flags to meta_prediction_optimizer argparse")

    # ================================================================
    # PATCH 3: Thread CLI flags into _s88_run_compare_models args_dict
    #
    # Team Beta Recommendation A: pass all 3 flags (not just dropout)
    # ================================================================

    old_s88_call = """        return _s88_run_compare_models({
    
            'survivors': getattr(args, 'survivors', None),
    
            'lottery_data': getattr(args, 'lottery_data', None),
    
            'trials': getattr(args, 'trials', 1),
    
            'enable_diagnostics': getattr(args, 'enable_diagnostics', False),
    
        })"""

    new_s88_call = """        return _s88_run_compare_models({
    
            'survivors': getattr(args, 'survivors', None),
    
            'lottery_data': getattr(args, 'lottery_data', None),
    
            'trials': getattr(args, 'trials', 1),
    
            'enable_diagnostics': getattr(args, 'enable_diagnostics', False),
    
            'normalize_features': getattr(args, 'normalize_features', False),
    
            'use_leaky_relu': getattr(args, 'use_leaky_relu', False),
    
            'dropout': getattr(args, 'dropout', None),
    
        })"""

    if old_s88_call not in content:
        print("ERROR: Cannot find _s88_run_compare_models call in main()")
        return False

    content = content.replace(old_s88_call, new_s88_call)
    print("[3/3] Threaded all 3 Category B flags into _s88_run_compare_models args_dict")

    # ================================================================
    # Write patched file
    # ================================================================
    TARGET.write_text(content)
    print(f"\nAll 3 patches applied to {TARGET}")
    return True


def main():
    print("=" * 60)
    print("Category B Phase 2: meta_prediction_optimizer_anti_overfit.py")
    print("  (v2 — with Team Beta required fixes #1-#3 + recs A-B)")
    print("=" * 60)

    if not verify_preconditions():
        sys.exit(1)

    # Create backup
    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    if not apply_patch():
        shutil.copy2(BACKUP, TARGET)
        print("REVERTED to backup due to patch failure")
        sys.exit(1)

    # Syntax check
    import py_compile
    try:
        py_compile.compile(str(TARGET), doraise=True)
        print(f"Syntax check PASSED")
    except py_compile.PyCompileError as e:
        print(f"SYNTAX ERROR: {e}")
        shutil.copy2(BACKUP, TARGET)
        print("REVERTED to backup due to syntax error")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Phase 2 COMPLETE")
    print("=" * 60)
    print(f"  File: {TARGET}")
    print(f"  Backup: {BACKUP}")
    print()
    print("  Patch 1: NN flags in _s88_run_compare_models subprocess builder")
    print("    --normalize-features: ALWAYS for neural_net (Option A)")
    print("    --use-leaky-relu: ALWAYS for neural_net")
    print("    --dropout: best-effort parse from args_dict")
    print()
    print("  Patch 2: Category B CLI flags in argparse")
    print("    (for single-model WATCHER pass-through)")
    print()
    print("  Patch 3: All 3 flags threaded into _s88_run_compare_models call")
    print()
    print("  Team Beta Fixes Applied:")
    print("    #1: Header accurately describes patches (no false Optuna claim)")
    print("    #2: Dropout log uses 'is not None' (0.0 correctly handled)")
    print("    #3: Dropout parse is best-effort (bad values ignored)")
    print()
    print("  NOTE: Single-model NN path (Rec B) deferred to Phase 2.1/Phase 3")
    print("        WATCHER must explicitly pass --normalize-features for now")


if __name__ == "__main__":
    main()
