#!/usr/bin/env python3
"""
Category B Phase 3A — Patch prediction_generator.py (Step 6 Scaler Application)
================================================================================

When Step 5 trains with --normalize-features, the checkpoint stores
scaler_mean and scaler_scale arrays. Step 6 MUST apply the same transform
before calling model.predict(), otherwise training/inference mismatch.

This patch:
  1. Loads scaler_mean/scaler_scale from the ACTUAL loaded checkpoint path
  2. Applies (X - mean) / scale before model.predict() in _build_prediction_pool
  3. Best-effort: if scaler fields absent (old checkpoints), proceeds without
     transform and logs a warning (Team Beta mandatory follow-on B)
  4. Guards: zero-scale clamping, model_type=="neural_net" gate, dimension check

Does NOT modify:
  - Signal gate logic
  - Feature schema validation
  - Any tree model paths (scaler is NN-only)

Team Beta Fixes Applied:
  3A-1: Uses self.model_checkpoint_path (the actually-loaded checkpoint)
  3A-2: Best-effort torch import with single-warning log
  3A-3: Scale clamping (zero/tiny scale -> 1.0) prevents NaN/inf
  3A-4: Explicit model_type=="neural_net" gate before applying normalization

Verified against live code:
  github.com/mmercalde/prng_cluster_public/main/prediction_generator.py
  Commit: 3c3f9ae (pre Phase 3)

Author: Team Alpha (S92)
Date: 2026-02-15
"""

import sys
import shutil
from pathlib import Path

TARGET = Path("prediction_generator.py")
BACKUP = TARGET.with_suffix(".pre_category_b_phase3")


def verify_preconditions():
    """Verify the file is in expected state."""
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run from ~/distributed_prng_analysis/")
        return False

    content = TARGET.read_text()

    if 'scaler_mean' in content:
        print("ERROR: scaler_mean already present. Already patched?")
        return False

    anchors = [
        'class PredictionGenerator:',
        'self._signal_gate_blocked = False',
        'self.model_checkpoint_path = checkpoint_path',
        'predicted_quality = self.model.predict(X)',
        'def _build_prediction_pool(',
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
    # PATCH 1: Add scaler instance attributes after model load
    #
    # Fix 3A-1: Uses self.model_checkpoint_path (the actually-loaded path)
    # Fix 3A-4: Only loads scaler when model_type == 'neural_net'
    # ================================================================

    old_model_loaded = """                self.logger.info(f"  Model loaded: {self.model_meta.get('model_type', 'unknown')}")
                self.model_checkpoint_path = checkpoint_path
                
                # v6.0: Extract and validate feature schema
                fs = self.model_meta.get("feature_schema", {})
                self.feature_names = fs.get("per_seed_feature_names", fs.get("feature_names", []))"""

    new_model_loaded = """                self.logger.info(f"  Model loaded: {self.model_meta.get('model_type', 'unknown')}")
                self.model_checkpoint_path = checkpoint_path
                
                # Category B: Load scaler for NN normalization (Phase 3)
                # Fix 3A-1: use self.model_checkpoint_path (the actual loaded path)
                # Fix 3A-4: only for neural_net model type
                model_type_loaded = self.model_meta.get('model_type', 'unknown')
                if model_type_loaded == 'neural_net':
                    self._load_nn_scaler(self.model_checkpoint_path)
                
                # v6.0: Extract and validate feature schema
                fs = self.model_meta.get("feature_schema", {})
                self.feature_names = fs.get("per_seed_feature_names", fs.get("feature_names", []))"""

    if old_model_loaded not in content:
        print("ERROR: Cannot find model loaded + feature schema block")
        return False

    content = content.replace(old_model_loaded, new_model_loaded)
    print("[1/4] Added scaler loading after model load (uses actual checkpoint path)")

    # ================================================================
    # PATCH 2: Add _load_nn_scaler() method
    #
    # Fix 3A-1: Takes checkpoint_path as param (the actual loaded one)
    # Fix 3A-2: Best-effort torch import, single warning log
    # Fix 3A-3: Zero-scale clamping
    # ================================================================

    old_signal_explanation = """    def _get_signal_explanation(self, status: str) -> str:"""

    new_signal_explanation = """    def _load_nn_scaler(self, checkpoint_path: str):
        \"\"\"
        Category B Phase 3: Load scaler parameters from NN checkpoint.
        
        Args:
            checkpoint_path: Path to the .pth checkpoint that was actually loaded.
                             NOT guessed — passed from self.model_checkpoint_path.
        
        Team Beta decisions:
          - Store mean/scale as arrays, not pickled sklearn (portable)
          - Best-effort: if absent (old checkpoints), proceed without transform + warn
          - Zero-scale elements clamped to 1.0 to prevent NaN/inf
          - Only called when model_type == 'neural_net'
        \"\"\"
        try:
            import torch
        except ImportError:
            self.logger.warning("[CAT-B] PyTorch not available — cannot load NN scaler")
            return
        
        if not checkpoint_path or not Path(checkpoint_path).exists():
            self.logger.warning(f"[CAT-B] NN checkpoint not found: {checkpoint_path}")
            self.logger.warning("[CAT-B] Proceeding without scaler (best-effort)")
            return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            self.logger.warning(f"[CAT-B] Cannot read checkpoint: {e}")
            self.logger.warning("[CAT-B] Proceeding without scaler (best-effort)")
            return
        
        if not isinstance(checkpoint, dict):
            self.logger.info("[CAT-B] Checkpoint is not a dict — no scaler metadata")
            return
        
        normalize_flag = checkpoint.get('normalize_features', False)
        if not normalize_flag:
            self.logger.info("[CAT-B] NN checkpoint: normalize_features=False (no scaler needed)")
            return
        
        scaler_mean = checkpoint.get('scaler_mean')
        scaler_scale = checkpoint.get('scaler_scale')
        
        if scaler_mean is None or scaler_scale is None:
            self.logger.warning("[CAT-B] normalize_features=True but scaler arrays missing!")
            self.logger.warning("[CAT-B] Proceeding without scaler (best-effort)")
            return
        
        import numpy as np
        self._scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
        self._scaler_scale = np.asarray(scaler_scale, dtype=np.float32)
        
        # Fix 3A-3: Clamp zero/tiny scale to 1.0 to prevent NaN/inf
        zero_mask = self._scaler_scale < 1e-12
        n_clamped = int(zero_mask.sum())
        if n_clamped > 0:
            self._scaler_scale[zero_mask] = 1.0
            self.logger.warning(f"[CAT-B] Clamped {n_clamped} zero-scale features to 1.0")
        
        self._normalize_features = True
        
        use_leaky = checkpoint.get('use_leaky_relu', False)
        self.logger.info(f"[CAT-B] NN scaler loaded: {len(self._scaler_mean)} features")
        self.logger.info(f"[CAT-B]   scale range: [{self._scaler_scale.min():.4f}, {self._scaler_scale.max():.4f}]")
        self.logger.info(f"[CAT-B]   use_leaky_relu: {use_leaky}")

    def _get_signal_explanation(self, status: str) -> str:"""

    if old_signal_explanation not in content:
        print("ERROR: Cannot find _get_signal_explanation method")
        return False

    content = content.replace(old_signal_explanation, new_signal_explanation)
    print("[2/4] Added _load_nn_scaler() with zero-scale clamping + best-effort guards")

    # ================================================================
    # PATCH 3: Apply scaler before model.predict() — NN-only gate
    #
    # Fix 3A-4: Explicit model_type=="neural_net" check
    # Fix 3A-3: Scale already clamped in _load_nn_scaler, but double-safe here
    # ================================================================

    old_predict = """        # Score using model or fallback
        if self.model is not None:
            predicted_quality = self.model.predict(X)
            model_type = self.model_meta.get("model_type", "unknown")"""

    new_predict = """        # Category B: Apply scaler normalization ONLY for neural_net
        # Fix 3A-4: explicit model_type gate prevents accidental cross-model normalization
        if (getattr(self, '_normalize_features', False)
                and self._scaler_mean is not None
                and self.model_meta.get("model_type") == "neural_net"):
            if X.shape[1] == len(self._scaler_mean):
                X = (X - self._scaler_mean) / self._scaler_scale
                self.logger.debug(f"[CAT-B] Applied input normalization to {X.shape[0]} samples")
            else:
                self.logger.warning(
                    f"[CAT-B] Scaler dimension mismatch: X has {X.shape[1]} features, "
                    f"scaler has {len(self._scaler_mean)}. Skipping normalization."
                )

        # Score using model or fallback
        if self.model is not None:
            predicted_quality = self.model.predict(X)
            model_type = self.model_meta.get("model_type", "unknown")"""

    if old_predict not in content:
        print("ERROR: Cannot find model.predict() block in _build_prediction_pool")
        return False

    content = content.replace(old_predict, new_predict)
    print("[3/4] Added scaler application with model_type=='neural_net' gate")

    # ================================================================
    # PATCH 4: Initialize scaler attrs for non-model path (safety)
    # ================================================================

    old_init_attrs = """        self._signal_gate_blocked = False
        self._signal_gate_result = None"""

    new_init_attrs = """        self._signal_gate_blocked = False
        self._signal_gate_result = None
        # Category B: scaler defaults (set properly after model load for NN)
        self._scaler_mean = None
        self._scaler_scale = None
        self._normalize_features = False"""

    if old_init_attrs not in content:
        print("ERROR: Cannot find _signal_gate_blocked initialization")
        return False

    content = content.replace(old_init_attrs, new_init_attrs)
    print("[4/4] Added safe scaler attribute defaults in __init__")

    TARGET.write_text(content)
    print(f"\nAll 4 patches applied to {TARGET}")
    return True


def main():
    print("=" * 60)
    print("Category B Phase 3A: prediction_generator.py (Step 6 Scaler)")
    print("  (v2 — with Team Beta fixes 3A-1 through 3A-4)")
    print("=" * 60)

    if not verify_preconditions():
        sys.exit(1)

    shutil.copy2(TARGET, BACKUP)
    print(f"Backup: {BACKUP}")

    if not apply_patch():
        shutil.copy2(BACKUP, TARGET)
        print("REVERTED to backup due to patch failure")
        sys.exit(1)

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
    print("Phase 3A COMPLETE")
    print("=" * 60)
    print(f"  File: {TARGET}")
    print(f"  Backup: {BACKUP}")
    print()
    print("  Team Beta Fixes Applied:")
    print("    3A-1: Uses self.model_checkpoint_path (actual loaded path)")
    print("    3A-2: Best-effort torch import, single-warning log")
    print("    3A-3: Zero-scale clamping (< 1e-12 -> 1.0) prevents NaN/inf")
    print("    3A-4: Explicit model_type=='neural_net' gate at both load and apply")
    print()
    print("  Behavior:")
    print("    NN + normalize=True + scaler present -> apply transform")
    print("    NN + normalize=False -> skip, log info")
    print("    NN + scaler missing -> warn, proceed (best-effort)")
    print("    Non-NN model -> scaler never loaded or applied")
    print("    Dimension mismatch -> warn, skip normalization")


if __name__ == "__main__":
    main()
