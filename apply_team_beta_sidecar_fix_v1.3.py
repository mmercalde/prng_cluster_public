#!/usr/bin/env python3
"""
Team Beta Patch v1.3: Step 5 subprocess compare-models sidecar consistency fix

Fixes:
- In --compare-models mode, best model lives on disk (checkpoint), not in memory.
- save_best_model() must write SUCCESS sidecar when checkpoint exists.
- Enforce invariant: prediction_allowed implies checkpoint_path is not None.

v1.3 improvements:
- Fixed string escaping (no embedded newlines in logger calls)
- Flexible anchor for _save_degenerate_sidecar (handles whitespace)
"""

from __future__ import annotations
import re
import shutil
from datetime import datetime
from pathlib import Path
import sys

FILE = Path("meta_prediction_optimizer_anti_overfit.py")

def fail(msg: str) -> None:
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(1)

def main() -> None:
    if not FILE.exists():
        fail(f"Missing file: {FILE.resolve()}")

    original = FILE.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # 1) Add instance fields for disk-first checkpoint in __init__
    # ------------------------------------------------------------------
    anchor_1 = r"(self\.best_model\s*=\s*None\s*\n\s*self\.best_model_type\s*=\s*None\s*\n)"
    if not re.search(anchor_1, original):
        fail("Could not find __init__ anchor for best_model/best_model_type block.")

    insert_1 = (
        r"\1"
        "        # Team Beta: subprocess comparison is disk-first (no in-memory model)\n"
        "        self.best_checkpoint_path = None\n"
        "        self.best_checkpoint_format = None\n"
    )
    patched = re.sub(anchor_1, insert_1, original, count=1)
    print("  ✓ Step 1: Added checkpoint fields to __init__")

    # ------------------------------------------------------------------
    # 2) Capture final checkpoint path in _run_model_comparison
    # ------------------------------------------------------------------
    anchor_2 = r"(winner\s*=\s*results\['winner'\]\s*\n)"
    if not re.search(anchor_2, patched):
        fail("Could not find _run_model_comparison anchor: winner = results['winner'].")

    insert_2 = (
        r"\1"
        "        # Team Beta: in subprocess mode, the winner is a checkpoint on disk\n"
        "        self.best_checkpoint_path = None\n"
        "        try:\n"
        "            if isinstance(results, dict) and winner in results:\n"
        "                self.best_checkpoint_path = (\n"
        "                    results[winner].get('final_checkpoint_path')\n"
        "                    or results[winner].get('checkpoint_path')\n"
        "                )\n"
        "        except Exception:\n"
        "            self.best_checkpoint_path = None\n"
        "\n"
        "        if self.best_checkpoint_path:\n"
        "            ext = Path(self.best_checkpoint_path).suffix\n"
        "            self.best_checkpoint_format = ext.lstrip('.') if ext else None\n"
        "            self.best_model_type = winner  # Team Beta refinement\n"
    )
    patched = re.sub(anchor_2, insert_2, patched, count=1)
    print("  ✓ Step 2: Added checkpoint capture in _run_model_comparison")

    # ------------------------------------------------------------------
    # 3) Add helper to write SUCCESS sidecar for existing checkpoint
    #    Using flexible whitespace anchor
    # ------------------------------------------------------------------
    anchor_3 = r"(\n\s*def _save_degenerate_sidecar\(self\):)"
    if not re.search(anchor_3, patched):
        fail("Could not find anchor for _save_degenerate_sidecar to insert helper.")

    # Helper method - using explicit string concatenation to avoid escaping issues
    helper_lines = [
        "",
        "    def _save_existing_checkpoint_sidecar(self, checkpoint_path: str, model_type: str):",
        '        """',
        "        Team Beta: Write a SUCCESS sidecar referencing an existing checkpoint on disk.",
        "        Used for --compare-models subprocess isolation mode.",
        '        """',
        "        output_path = Path(self.output_dir)",
        "        output_path.mkdir(parents=True, exist_ok=True)",
        "",
        "        ckpt = Path(checkpoint_path)",
        "        if not ckpt.exists():",
        '            self.logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")',
        "            self._save_degenerate_sidecar()",
        "            return",
        "",
        "        duration = (datetime.now() - self.start_time).total_seconds()",
        "",
        "        provenance = {",
        "            'run_id': f\"step5_{self.start_time.strftime('%Y%m%d_%H%M%S')}\",",
        "            'started_at': self.start_time.isoformat(),",
        "            'duration_seconds': duration,",
        "            'survivors_file': str(Path(self.survivors_file).resolve()),",
        "            'n_survivors': len(self.X),",
        "            'model_type': model_type,",
        "            'compare_models_used': True,",
        "            'checkpoint_path': str(ckpt.resolve()),",
        "            'outcome': 'SUCCESS'",
        "        }",
        "",
        "        # NOTE: In subprocess mode, metrics may be partial.",
        "        # Artifact authority is the checkpoint; metrics are best-effort.",
        "        training_metrics = {",
        "            'r2': getattr(self.best_metrics, 'r2_score', 0.0) if self.best_metrics else 0.0,",
        "            'train_mae': getattr(self.best_metrics, 'train_mae', 0.0) if self.best_metrics else 0.0,",
        "            'val_mae': getattr(self.best_metrics, 'val_mae', 0.0) if self.best_metrics else 0.0,",
        "            'test_mae': getattr(self.best_metrics, 'test_mae', 0.0) if self.best_metrics else 0.0,",
        "            'overfit_ratio': getattr(self.best_metrics, 'overfit_ratio', 1.0) if self.best_metrics else 1.0,",
        "            'status': 'success'",
        "        }",
        "",
        "        checkpoint_format = ckpt.suffix.lstrip('.') if ckpt.suffix else None",
        "",
        "        sidecar = {",
        '            "schema_version": "3.2.0",',
        '            "model_type": model_type,',
        '            "checkpoint_path": str(ckpt),',
        '            "checkpoint_format": checkpoint_format,',
        '            "feature_schema": self.feature_schema,',
        '            "signal_quality": self.signal_quality,',
        '            "data_context": self.data_context,',
        '            "training_metrics": training_metrics,',
        '            "hyperparameters": self.best_config or {},',
        '            "hardware": {',
        '                "device_requested": self.device,',
        '                "cuda_available": CUDA_INITIALIZED',
        "            },",
        '            "training_info": {',
        '                "started_at": self.start_time.isoformat(),',
        '                "completed_at": datetime.now().isoformat(),',
        '                "duration_seconds": duration,',
        '                "outcome": "SUCCESS"',
        "            },",
        '            "agent_metadata": {',
        '                "pipeline_step": 5,',
        '                "pipeline_step_name": "anti_overfit_training",',
        "                \"run_id\": provenance.get('run_id'),",
        '                "parent_run_id": self.parent_run_id,',
        '                "outcome": "SUCCESS",',
        '                "exit_code": 0',
        "            },",
        '            "provenance": provenance',
        "        }",
        "",
        "        # Canonical invariant check",
        '        if sidecar["signal_quality"].get("prediction_allowed", True):',
        '            assert sidecar["checkpoint_path"] is not None, \\',
        '                "Invariant violated: prediction_allowed=True but checkpoint_path=None"',
        "",
        '        sidecar_path = output_path / "best_model.meta.json"',
        "        with open(sidecar_path, 'w') as f:",
        "            import json",
        "            json.dump(sidecar, f, indent=2)",
        "",
        '        self.logger.info("")',
        '        self.logger.info("=" * 70)',
        '        self.logger.info("SUBPROCESS WINNER SIDECAR SAVED (Existing Checkpoint)")',
        '        self.logger.info("=" * 70)',
        '        self.logger.info(f"  Model type: {model_type}")',
        "        self.logger.info(f\"  Checkpoint: {sidecar['checkpoint_path']}\")",
        "        self.logger.info(f\"  R² score: {training_metrics.get('r2', 0.0):.4f}\")",
        "        self.logger.info(f\"  Signal status: {self.signal_quality.get('signal_status')}\")",
        "        self.logger.info(f\"  Data fingerprint: {self.data_context.get('fingerprint_hash')}\")",
        '        self.logger.info("=" * 70)',
        "",
    ]
    helper = "\n".join(helper_lines)

    patched = re.sub(anchor_3, helper + r"\1", patched, count=1)
    print("  ✓ Step 3: Added _save_existing_checkpoint_sidecar helper")

    # ------------------------------------------------------------------
    # 4) Modify save_best_model() to use checkpoint-based success path
    # ------------------------------------------------------------------
    anchor_4 = r'(    def save_best_model\(self\):\n        """Save the best model with full sidecar metadata\."""\n        if self\.best_model is None:\n            self\.logger\.warning\("No model trained - saving degenerate sidecar only"\)\n            self\._save_degenerate_sidecar\(\)\n            return\n)'

    if not re.search(anchor_4, patched, flags=re.MULTILINE):
        fail("Could not find save_best_model early-degenerate block to replace.")

    replacement_4 = '''    def save_best_model(self):
        """Save the best model with full sidecar metadata."""
        # Team Beta: subprocess comparison is disk-first (no in-memory model in parent)
        if self.best_model is None:
            if getattr(self, 'compare_models', False) and getattr(self, 'best_checkpoint_path', None):
                self._save_existing_checkpoint_sidecar(
                    checkpoint_path=self.best_checkpoint_path,
                    model_type=self.best_model_type or "unknown"
                )
                return

            self.logger.warning("No model trained - saving degenerate sidecar only")
            self._save_degenerate_sidecar()
            return
'''

    patched = re.sub(anchor_4, replacement_4, patched, count=1, flags=re.MULTILINE)
    print("  ✓ Step 4: Updated save_best_model early guard")

    # ------------------------------------------------------------------
    # Write backup + patched file
    # ------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = FILE.with_suffix(f".py.pre_team_beta_v1.3_{ts}")
    shutil.copy2(FILE, backup)
    FILE.write_text(patched, encoding="utf-8")

    print(f"\n✅ Backup created: {backup}")
    print(f"✅ Patched: {FILE}")
    print(f"✅ Version: 1.3 (line-joined helper, flexible anchors)")

if __name__ == "__main__":
    main()
