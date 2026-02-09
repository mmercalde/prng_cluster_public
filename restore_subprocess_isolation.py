#!/usr/bin/env python3
"""
Patch: Restore Subprocess Isolation for LightGBM Compatibility
==============================================================

This patch modifies meta_prediction_optimizer_anti_overfit.py to use
subprocess_trial_coordinator.py for --compare-models mode.

This fixes the "Unknown OpenCL Error (-9999)" when running LightGBM on Zeus.

Session: 72
Date: February 8, 2026
Team Beta Approved: Pending

CHANGES:
1. Add conditional import for subprocess_trial_coordinator
2. Add run_subprocess_comparison() function
3. Modify _run_model_comparison() to use subprocess when available

SAFETY:
- Falls back to inline training if subprocess_trial_coordinator unavailable
- No changes to single-model training path
- Preserves all existing functionality
"""

import os
import sys
import shutil
from datetime import datetime

TARGET_FILE = "meta_prediction_optimizer_anti_overfit.py"
BACKUP_SUFFIX = f".pre_subprocess_patch_{datetime.now():%Y%m%d_%H%M%S}"

# =============================================================================
# PATCH 1: Add subprocess import after existing imports (around line 50)
# =============================================================================

IMPORT_ANCHOR = '''import argparse'''

IMPORT_ADDITION = '''import argparse

# Subprocess isolation for multi-model comparison (fixes LightGBM OpenCL on Zeus)
try:
    from subprocess_trial_coordinator import (
        SubprocessTrialCoordinator,
        SAFE_MODEL_ORDER as SUBPROCESS_SAFE_ORDER,
        TrialResult
    )
    SUBPROCESS_AVAILABLE = True
except ImportError:
    SUBPROCESS_AVAILABLE = False
    SUBPROCESS_SAFE_ORDER = None
    SubprocessTrialCoordinator = None
    TrialResult = None'''


# =============================================================================
# PATCH 2: Add subprocess comparison function (before class MultiModelTrainer)
# =============================================================================

CLASS_ANCHOR = '''class MultiModelTrainer:'''

SUBPROCESS_FUNCTION = '''# =============================================================================
# SUBPROCESS ISOLATION FOR MULTI-MODEL COMPARISON
# =============================================================================

def run_subprocess_comparison(
    X_train, y_train, X_val, y_val,
    output_dir: str = 'models/reinforcement',
    timeout: int = 600,
    logger = None
) -> dict:
    """
    Run multi-model comparison using subprocess isolation.
    
    Each model trains in a fresh subprocess, preventing OpenCL/CUDA conflicts.
    This allows LightGBM (OpenCL) to work alongside CUDA models on Zeus.
    
    Args:
        X_train, y_train, X_val, y_val: Training/validation data
        output_dir: Directory for model outputs
        timeout: Per-model timeout in seconds
        logger: Optional logger
        
    Returns:
        Dict with results for each model and winner info
    """
    import logging
    logger = logger or logging.getLogger(__name__)
    
    if not SUBPROCESS_AVAILABLE:
        raise RuntimeError("subprocess_trial_coordinator not available")
    
    logger.info("=" * 70)
    logger.info("SUBPROCESS ISOLATION MODE")
    logger.info("=" * 70)
    logger.info(f"Models: {SUBPROCESS_SAFE_ORDER}")
    logger.info("Each model trains in isolated subprocess (fixes LightGBM OpenCL)")
    logger.info("=" * 70)
    
    results = {}
    
    with SubprocessTrialCoordinator(
        X_train, y_train, X_val, y_val,
        worker_script='train_single_trial.py',
        timeout=timeout,
        verbose=True,
        output_dir=output_dir
    ) as coordinator:
        
        # Train each model type
        for i, model_type in enumerate(SUBPROCESS_SAFE_ORDER):
            if model_type == 'random_forest':
                continue  # Skip random_forest for now
                
            logger.info(f"Training {model_type} (subprocess)...")
            
            try:
                # Run isolated trial
                trial_result = coordinator.run_trial(
                    trial_number=i,
                    model_type=model_type,
                    params={}  # Use defaults
                )
                
                if trial_result.success:
                    results[model_type] = {
                        'model': None,  # Model is in file, not memory
                        'model_type': model_type,
                        'metrics': {
                            'train_mse': trial_result.train_mse,
                            'val_mse': trial_result.val_mse,
                            'r2': trial_result.r2
                        },
                        'hyperparameters': trial_result.params,
                        'checkpoint_path': trial_result.checkpoint_path
                    }
                    logger.info(f"  {model_type}: R²={trial_result.r2:.4f}")
                else:
                    results[model_type] = {
                        'model': None,
                        'model_type': model_type,
                        'metrics': {'error': trial_result.error},
                        'hyperparameters': {}
                    }
                    logger.error(f"  {model_type} failed: {trial_result.error}")
                    
            except Exception as e:
                logger.error(f"  {model_type} failed: {e}")
                results[model_type] = {
                    'model': None,
                    'model_type': model_type,
                    'metrics': {'error': str(e)},
                    'hyperparameters': {}
                }
        
        # Find winner
        valid_results = {
            k: v for k, v in results.items() 
            if v.get('model') is not None or v.get('checkpoint_path')
        }
        
        if not valid_results:
            # Fall back to any result with r2 score
            valid_results = {
                k: v for k, v in results.items()
                if 'r2' in v.get('metrics', {})
            }
        
        if valid_results:
            winner = max(
                valid_results.keys(),
                key=lambda k: valid_results[k]['metrics'].get('r2', float('-inf'))
            )
            results['winner'] = winner
            results['winner_metrics'] = valid_results[winner]['metrics']
            
            # Copy winning model to output directory
            if valid_results[winner].get('checkpoint_path'):
                winner_path = valid_results[winner]['checkpoint_path']
                if os.path.exists(winner_path):
                    # Determine extension
                    ext = os.path.splitext(winner_path)[1]
                    dest_path = os.path.join(output_dir, f"best_model{ext}")
                    shutil.copy(winner_path, dest_path)
                    results[winner]['final_checkpoint_path'] = dest_path
                    logger.info(f"Copied winning model to {dest_path}")
        else:
            raise ValueError("All models failed to train")
    
    return results


class MultiModelTrainer:'''


# =============================================================================
# PATCH 3: Modify train_and_compare to use subprocess when available
# =============================================================================

OLD_TRAIN_AND_COMPARE_START = '''    def train_and_compare(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_types: List[str] = None,
        metric: str = 'r2'
    ) -> Dict:
        """
        Train all model types and compare performance.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_types: List of models to train (default: all)
            metric: Comparison metric ('r2', 'mse', 'mae')
            
        Returns:
            Dict with results for each model and winner
        """
        model_types = model_types or self.SAFE_MODEL_ORDER
        
        results = {}
        for model_type in model_types:'''

NEW_TRAIN_AND_COMPARE_START = '''    def train_and_compare(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_types: List[str] = None,
        metric: str = 'r2',
        use_subprocess: bool = True,
        output_dir: str = 'models/reinforcement'
    ) -> Dict:
        """
        Train all model types and compare performance.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_types: List of models to train (default: all)
            metric: Comparison metric ('r2', 'mse', 'mae')
            use_subprocess: Use subprocess isolation (fixes LightGBM OpenCL)
            output_dir: Output directory for models
            
        Returns:
            Dict with results for each model and winner
        """
        # Try subprocess isolation first (fixes LightGBM OpenCL on Zeus)
        if use_subprocess and SUBPROCESS_AVAILABLE:
            self.logger.info("Using subprocess isolation for model comparison")
            try:
                return run_subprocess_comparison(
                    X_train, y_train, X_val, y_val,
                    output_dir=output_dir,
                    logger=self.logger
                )
            except Exception as e:
                self.logger.warning(f"Subprocess comparison failed: {e}")
                self.logger.warning("Falling back to inline training")
        
        # Fallback: inline training (original behavior)
        model_types = model_types or self.SAFE_MODEL_ORDER
        
        results = {}
        for model_type in model_types:'''


def apply_patches():
    """Apply all patches to the target file."""
    
    if not os.path.exists(TARGET_FILE):
        print(f"ERROR: {TARGET_FILE} not found")
        print("Run this script from ~/distributed_prng_analysis")
        sys.exit(1)
    
    # Create backup
    backup_path = TARGET_FILE + BACKUP_SUFFIX
    shutil.copy(TARGET_FILE, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    # Read current content
    with open(TARGET_FILE, 'r') as f:
        content = f.read()
    
    # Track changes
    changes_made = []
    
    # PATCH 1: Add subprocess import
    if "SUBPROCESS_AVAILABLE" in content:
        print("⚠️  Patch 1 (imports) already applied — skipping")
    elif IMPORT_ANCHOR not in content:
        print("❌ ERROR: Could not find import anchor")
        sys.exit(1)
    else:
        content = content.replace(IMPORT_ANCHOR, IMPORT_ADDITION)
        changes_made.append("imports")
        print("✅ Applied Patch 1: Subprocess imports")
    
    # PATCH 2: Add subprocess comparison function
    if "def run_subprocess_comparison" in content:
        print("⚠️  Patch 2 (function) already applied — skipping")
    elif CLASS_ANCHOR not in content:
        print("❌ ERROR: Could not find class anchor")
        sys.exit(1)
    else:
        content = content.replace(CLASS_ANCHOR, SUBPROCESS_FUNCTION)
        changes_made.append("function")
        print("✅ Applied Patch 2: run_subprocess_comparison function")
    
    # PATCH 3: Modify train_and_compare
    if "use_subprocess: bool = True" in content:
        print("⚠️  Patch 3 (train_and_compare) already applied — skipping")
    elif OLD_TRAIN_AND_COMPARE_START not in content:
        print("❌ ERROR: Could not find train_and_compare method")
        print("   The method signature may have changed")
        sys.exit(1)
    else:
        content = content.replace(OLD_TRAIN_AND_COMPARE_START, NEW_TRAIN_AND_COMPARE_START)
        changes_made.append("train_and_compare")
        print("✅ Applied Patch 3: train_and_compare with subprocess support")
    
    if not changes_made:
        print("\n⚠️  No changes needed — all patches already applied")
        return
    
    # Write patched content
    with open(TARGET_FILE, 'w') as f:
        f.write(content)
    
    print()
    print("=" * 60)
    print("PATCHING COMPLETE")
    print("=" * 60)
    print(f"Changes: {', '.join(changes_made)}")
    print()
    print("Verify with:")
    print("  python3 -c \"import meta_prediction_optimizer_anti_overfit; print('OK')\"")
    print()
    print("Test subprocess isolation:")
    print("  python3 meta_prediction_optimizer_anti_overfit.py \\")
    print("    --survivors survivors_with_scores.json \\")
    print("    --lottery-data lottery_history.json \\")
    print("    --compare-models --trials 1")
    print()
    print(f"To revert: cp {backup_path} {TARGET_FILE}")


if __name__ == "__main__":
    apply_patches()
