#!/usr/bin/env python3
"""
Training Diagnostics — Chapter 14 Phase 1
==========================================

Core diagnostic module for live training introspection across all 4 model types.
Uses PyTorch dynamic computational graph hooks for neural networks and native
callbacks for tree models (XGBoost, LightGBM, CatBoost).

Design Invariants (non-negotiable):
1. PASSIVE OBSERVER — Never modifies gradients, weights, or training behavior
2. BEST-EFFORT, NON-FATAL — All code paths wrapped in try/except
3. ABSENT ≠ FAILURE — Missing diagnostics maps to PROCEED, not BLOCK

Schema Version: 1.1.0 (Multi-model design per Option D)

Usage:
    # Single model
    diag = TrainingDiagnostics.create('catboost')
    diag.attach(model)
    # ... training loop calls diag.on_round_end() ...
    diag.detach()
    diag.save()
    
    # Multi-model (--compare-models)
    collector = MultiModelDiagnostics()
    for model_type in ['neural_net', 'xgboost', 'lightgbm', 'catboost']:
        diag = collector.create_for_model(model_type)
        # ... training ...
        collector.finalize_model(model_type, diag, metrics)
    collector.save()

Author: Distributed PRNG Analysis System
Date: February 2026

Version History:
    1.0.0   2026-02-08  Session 69  Initial implementation (Phase 1-2)
                                    - TrainingDiagnostics ABC with factory
                                    - NNDiagnostics with PyTorch hooks
                                    - TreeDiagnostics for XGB/LGB/CatBoost
                                    - MultiModelDiagnostics collector
                                    - Severity classification system
    1.1.0   2026-02-08  Session 71  Phase 5: FIFO history pruning
                                    - Added MAX_HISTORY_FILES constant (100)
                                    - Added _prune_history_fifo() function
                                    - Glob narrowed to compare_models_*.json
                                    - Added is_dir() defensive check
                                    - Team Beta approved with refinements
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

SCHEMA_VERSION = "1.1.0"
DEFAULT_OUTPUT_DIR = "diagnostics_outputs"
DEFAULT_OUTPUT_FILE = "training_diagnostics.json"
DEFAULT_HISTORY_DIR = "diagnostics_outputs/history"
MAX_HISTORY_FILES = 100  # FIFO pruning limit (Team Beta Session 69)

# Severity thresholds (can be overridden via config)
SEVERITY_THRESHOLDS = {
    "dead_neuron_pct": {"warning": 15.0, "critical": 25.0},
    "gradient_spread_ratio": {"warning": 1000.0, "critical": 10000.0},
    "overfit_gap_ratio": {"warning": 0.3, "critical": 0.5},
    "gradient_norm_min": {"warning": 1e-6, "critical": 1e-8},
}

# [CH14-DIAG wave 1, 2026-11-23] Report-content level (declared setting).
#   legacy (default): report content exactly as before this change.
#   full:   named features (CD-04), NN eval-mode input-gradient probe (CD-03),
#           spread taken from the last round that carries it (CD-02 producer),
#           tree curves/importances on the paths that run (CD-05), and a
#           column-class tag on every attribution line (CD-13).
# Training is never altered at either level (Design Invariant 1).
DIAGNOSTICS_LEVELS = ("legacy", "full")
DEFAULT_DIAGNOSTICS_LEVEL = "legacy"
# Rows of the fixed training slice used by the NN input-gradient probe (CD-03).
DEFAULT_GRADIENT_PROBE_ROWS = 512
# Column classes (CD-13). Computed from the training matrix, never from a list.
COLUMN_CLASSES = ("constant", "run", "trial", "local")


def validate_diagnostics_level(level: Optional[str]) -> str:
    """None -> DEFAULT_DIAGNOSTICS_LEVEL; unknown value -> ValueError (fail closed)."""
    if level is None:
        return DEFAULT_DIAGNOSTICS_LEVEL
    if level not in DIAGNOSTICS_LEVELS:
        raise ValueError(f"diagnostics level must be one of {DIAGNOSTICS_LEVELS}, got {level!r}")
    return level


# [CH14-DIAG wave 2, 2026-11-23] Declared settings. Every one acts at level 'full' only;
# at level 'legacy' none of them is read, so the legacy report is unchanged.
#   dead_unit_definition   CD-23. linear_zero (default, today's: share of exact zeros at the
#                          nn.Linear output, td forward hook) | always_negative_preact (share of
#                          units whose pre-activation, the input of the activation module that
#                          follows the Linear, is <= 0 on every row of the eval-mode probe batch;
#                          needs the CD-03 probe, else the report says linear_zero was used)
#   unused_gradient_epsilon CD-08. NN column counted unused when its mean |input gradient| on the
#                          probe is below this (trees: importance exactly 0)
#   feature_manifest       TODO-CH14-4. path of the feature manifest (format tfm_feature_manifest/1);
#                          its run_context alias names the run-level columns (CD-13 'run' class)
#   run_id / run_started_at CD-07. stamped into every report and the compare-models pointer
DEAD_UNIT_DEFINITIONS = ("linear_zero", "always_negative_preact")
DEFAULT_DEAD_UNIT_DEFINITION = "linear_zero"
DEFAULT_UNUSED_GRADIENT_EPSILON = 1e-8
DEFAULT_FEATURE_MANIFEST_PATH = "models/feature_manifest.json"
FEATURE_MANIFEST_FORMAT = "tfm_feature_manifest/1"
FEATURE_MANIFEST_RUN_ALIAS = "run_context"


def validate_dead_unit_definition(value: Optional[str]) -> str:
    if value is None:
        return DEFAULT_DEAD_UNIT_DEFINITION
    if value not in DEAD_UNIT_DEFINITIONS:
        raise ValueError(f"dead_unit_definition must be one of {DEAD_UNIT_DEFINITIONS}, got {value!r}")
    return value


def new_run_id(now: Optional[datetime] = None) -> Tuple[str, str]:
    """CD-07: (run_id, run_started_at UTC ISO) for one Step 5 run."""
    import uuid
    from datetime import timezone
    now = now or datetime.now(timezone.utc)
    return f"s5-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}", now.isoformat()


# =============================================================================
# [W3 2026-11-27] Flag-to-model binding: run-local staging and identity records.
# Active ONLY with an explicit binding request (--binding-run-id, made by WATCHER). Level 'full'
# and --diagnostics-run-id alone never turn it on. Design:
# docs/drafts/unit_w/DESIGN_FLAG_MODEL_BINDING_20261127_REV2.md section 4.
#   diagnostics_outputs/<run id>/<model id>/   one writer: the final-model writer of that model
#     <family>_diagnostics.json                the report bytes (same bytes as the public copies)
#     best_model.<ext>                         the checkpoint bytes (public file is a copy of these)
#     identity.pending.json                    the pair, written by the final-model writer
#     identity.json                            published only after exit 0 (single) / child rc 0 (compare)
# =============================================================================
W3_SCHEMA = "w3/1"
W3_PENDING = "identity.pending.json"
W3_PUBLISHED = "identity.json"
W3_REQUEST_KEYS = ("binding_run_id", "binding_run_started_at", "binding_model_id")


def w3_safe_id(value: Any) -> Optional[str]:
    """A run id or model id usable as one path component, else None (W3 stays off)."""
    import re
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,95}", value):
        return None
    return value


def w3_request(run_id: Any, started: Any = None, model_id: Any = None, family: str = "model",
               root: str = DEFAULT_OUTPUT_DIR) -> Optional[Dict[str, Any]]:
    """The binding request for one final model, or None (no request or an unsafe id: W3 off)."""
    rid = w3_safe_id(run_id)
    if rid is None:
        if run_id:
            logger.warning("[W3] binding run id %r is not a safe path component; binding off", run_id)
        return None
    if model_id:
        mid = w3_safe_id(model_id)
        if mid is None:
            logger.warning("[W3] binding model id %r is not a safe path component; binding off", model_id)
            return None
    else:
        import uuid
        mid = f"{family}-{uuid.uuid4().hex[:8]}"
    return {"run_id": rid, "run_started_at": started if isinstance(started, str) else None,
            "model_id": mid, "stage_dir": os.path.join(root, rid, mid)}


def w3_sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def w3_atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    """Temp file in the same dir, fsync, os.replace: the file is visible only when complete."""
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def w3_make_record(req: Dict[str, Any], model_type: str, checkpoint_file: str, report_file: str,
                   require_report_checkpoint: bool) -> Dict[str, Any]:
    """The pair, from the staged files the caller (a final-model writer) wrote itself. Raises ValueError with
    the reason when the staged files do not agree (then nothing is recorded: the run stays unbound)."""
    stage = req["stage_dir"]
    for name in (checkpoint_file, report_file):
        if os.path.basename(name) != name:
            raise ValueError(f"staged file name {name!r} is not a basename")
    ck_sha = w3_sha256_file(os.path.join(stage, checkpoint_file))
    with open(os.path.join(stage, report_file), "rb") as f:
        raw = f.read()
    import hashlib
    rep = json.loads(raw)
    if rep.get("diagnostics_run_id") != req["run_id"]:
        raise ValueError(f"staged report run id {rep.get('diagnostics_run_id')!r} != {req['run_id']!r}")
    if rep.get("model_type") != model_type:
        raise ValueError(f"staged report family {rep.get('model_type')!r} != {model_type!r}")
    if rep.get("binding_model_id") != req["model_id"]:
        raise ValueError(f"staged report model id {rep.get('binding_model_id')!r} != {req['model_id']!r}")
    if require_report_checkpoint and rep.get("checkpoint_sha256") != ck_sha:
        raise ValueError(f"report checkpoint_sha256 {rep.get('checkpoint_sha256')!r} != staged checkpoint {ck_sha}")
    return {"schema": W3_SCHEMA, "run_id": req["run_id"], "run_started_at": req.get("run_started_at"),
            "model_id": req["model_id"], "model_type": model_type, "stage_dir": os.path.abspath(stage),
            "checkpoint_file": checkpoint_file, "checkpoint_sha256": ck_sha,
            "report_file": report_file, "report_sha256": hashlib.sha256(raw).hexdigest(),
            "report_checkpoint_sha256": rep.get("checkpoint_sha256"),
            "recorded_at": datetime.utcnow().isoformat() + "Z"}


def w3_write_pending(req: Dict[str, Any], model_type: str, checkpoint_file: str, report_file: str,
                     require_report_checkpoint: bool) -> Optional[Dict[str, Any]]:
    """Best-effort: the record, written atomically as identity.pending.json, or None (logged)."""
    try:
        rec = w3_make_record(req, model_type, checkpoint_file, report_file, require_report_checkpoint)
        w3_atomic_write_json(os.path.join(req["stage_dir"], W3_PENDING), rec)
        return rec
    except Exception as e:
        logger.warning("[W3] identity record not written (run stays unbound): %s: %s", type(e).__name__, e)
        return None


W3_EXPECT_KEYS = ("schema", "run_id", "model_id", "model_type", "stage_dir", "checkpoint_file",
                  "checkpoint_sha256", "report_file", "report_sha256", "recorded_at")


def w3_publish(stage_dir: str, run_id: str, model_type: Optional[str] = None,
               expect: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Publish identity.pending.json as identity.json after re-checking the staged files. Never raises.
    The caller calls this ONLY after the writer's exit 0 (single) or the child's returncode 0 (compare).
    expect (single model): the record the accepted final path wrote in this process; the pending file must be
    that record, else nothing is published (Beta X-W3 R2 Q2)."""
    try:
        with open(os.path.join(stage_dir, W3_PENDING)) as f:
            rec = json.load(f)
        if expect is not None:
            _diff = [k for k in W3_EXPECT_KEYS if rec.get(k) != expect.get(k)]
            if _diff:
                return False, f"pending record is not the accepted final path's pair (differs in {_diff})"
        if rec.get("run_id") != run_id:
            return False, f"pending record run id {rec.get('run_id')!r} != {run_id!r}"
        if model_type is not None and rec.get("model_type") != model_type:
            return False, f"pending record family {rec.get('model_type')!r} != {model_type!r}"
        if w3_sha256_file(os.path.join(stage_dir, rec["checkpoint_file"])) != rec["checkpoint_sha256"]:
            return False, "staged checkpoint changed after it was recorded"
        if w3_sha256_file(os.path.join(stage_dir, rec["report_file"])) != rec["report_sha256"]:
            return False, "staged report changed after it was recorded"
        w3_atomic_write_json(os.path.join(stage_dir, W3_PUBLISHED),
                             dict(rec, published_at=datetime.utcnow().isoformat() + "Z"))
        os.unlink(os.path.join(stage_dir, W3_PENDING))
        return True, "published"
    except Exception as e:
        logger.warning("[W3] identity not published (run stays unbound): %s: %s", type(e).__name__, e)
        return False, f"{type(e).__name__}: {e}"


def w3_sidecar_block(rec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The sidecar's step5_identity block (owner decision 3, 2026-11-27): the pair as recorded."""
    if not rec:
        return None
    return {k: rec.get(k) for k in ("schema", "run_id", "run_started_at", "model_id", "model_type", "stage_dir",
                                    "checkpoint_file", "checkpoint_sha256", "report_file", "report_sha256")}


def diagnostics_options_from_args(args) -> Optional[Dict[str, Any]]:
    """
    One bundle of report options from a parsed Step 5 / trial-worker namespace.
    None at level 'legacy' (nothing downstream changes). At level 'full' a run id and start
    time are generated when the caller did not pass them (CD-07).
    """
    if not getattr(args, "enable_diagnostics", False) or \
            validate_diagnostics_level(getattr(args, "diagnostics_level", None)) != "full":
        return None
    run_id = getattr(args, "diagnostics_run_id", None)
    started = getattr(args, "diagnostics_run_started_at", None)
    if not run_id:
        run_id, gen_started = new_run_id()
        started = started or gen_started
    return {
        "level": "full",
        "gradient_probe_rows": getattr(args, "gradient_probe_rows", None),
        "run_id": run_id,
        "run_started_at": started,
        "dead_unit_definition": validate_dead_unit_definition(getattr(args, "dead_unit_definition", None)),
        "unused_gradient_epsilon": getattr(args, "unused_gradient_epsilon", None),
        "feature_manifest": getattr(args, "feature_manifest", None),
    }


def diagnostics_child_flags(opts: Optional[Dict[str, Any]]) -> List[str]:
    """CLI flags that hand `opts` to a child (Step 5 per-model child or the trial worker). [] at legacy."""
    if not opts or opts.get("level") != "full":
        return []
    flags = ["--diagnostics-level", "full"]
    pairs = (("gradient_probe_rows", "--gradient-probe-rows", int),
             ("run_id", "--diagnostics-run-id", str),
             ("run_started_at", "--diagnostics-run-started-at", str),
             ("dead_unit_definition", "--dead-unit-definition", str),
             ("unused_gradient_epsilon", "--unused-gradient-epsilon", float),
             ("feature_manifest", "--feature-manifest", str))
    for key, flag, cast in pairs:
        if opts.get(key) is not None:
            flags.extend([flag, str(cast(opts[key]))])
    return flags


def load_run_level_names(path: Optional[str] = None) -> Tuple[Optional[List[str]], Dict[str, Any]]:
    """
    TODO-CH14-4: the run-level column names, read from the feature manifest (never a second list).
    Uses models.feature_manifest (LL-1) when importable, else reads the JSON with the same format
    check. Returns (names or None, source record). Missing or invalid manifest -> (None, reason).
    """
    import hashlib
    path = path or DEFAULT_FEATURE_MANIFEST_PATH
    src: Dict[str, Any] = {"path": str(path), "alias": FEATURE_MANIFEST_RUN_ALIAS}
    if not os.path.isfile(path):
        src["status"] = "missing"
        return None, src
    try:
        raw = Path(path).read_bytes()
        src["sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            from models.feature_manifest import load_manifest, expand_groups  # LL-1, when landed
            man = load_manifest(Path(path))
            groups = expand_groups(man, [FEATURE_MANIFEST_RUN_ALIAS])
            src["reader"] = "models.feature_manifest"
        except ImportError:
            man = json.loads(raw)
            if man.get("format") != FEATURE_MANIFEST_FORMAT:
                raise ValueError(f"format {man.get('format')!r} != {FEATURE_MANIFEST_FORMAT!r}")
            groups = list((man.get("aliases") or {}).get(FEATURE_MANIFEST_RUN_ALIAS, []))
            if not groups:
                raise ValueError(f"no {FEATURE_MANIFEST_RUN_ALIAS!r} alias")
            src["reader"] = "json"
        names = sorted({n for g in groups for n in man["groups"][g]})
        src.update({"status": "ok", "format": man.get("format"), "groups": sorted(groups),
                    "count": len(names)})
        return names, src
    except Exception as e:
        src.update({"status": "invalid", "error": str(e)})
        return None, src


# =============================================================================
# BASE CLASS
# =============================================================================

class TrainingDiagnostics(ABC):
    """
    Abstract base class for model-specific training diagnostics.
    
    Subclasses implement attach/detach/capture logic for their model type.
    All subclasses share the same output schema and severity classification.
    """
    
    model_type: str = "base"
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        capture_every_n: int = 5,
        max_rounds: int = 500,
        diagnostics_level: Optional[str] = None,
        names_source: Optional[str] = None,
        column_classes: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        run_started_at: Optional[str] = None,
        unused_gradient_epsilon: Optional[float] = None,
    ):
        """
        Args:
            feature_names: Names of input features. If None, uses generic names.
            capture_every_n: Capture detailed snapshot every N rounds.
            max_rounds: Safety cap to prevent unbounded memory growth.
            diagnostics_level: 'legacy' (default, unchanged report) or 'full' (CH14-DIAG wave 1).
            names_source: where feature_names came from ('npz', 'sidecar', 'caller', 'positional').
            column_classes: output of classify_columns() for the training matrix (CD-13).
            run_id, run_started_at: CD-07 stamp (level 'full'), from diagnostics_options_from_args.
            unused_gradient_epsilon: CD-08 NN threshold (level 'full').
        """
        self.feature_names = feature_names or []
        self.capture_every_n = capture_every_n
        self.max_rounds = max_rounds
        self.diagnostics_level = validate_diagnostics_level(diagnostics_level)
        self.names_source = names_source or ("caller" if feature_names else "positional")
        self.column_classes = column_classes
        self.run_id = run_id
        self.run_started_at = run_started_at
        self.unused_gradient_epsilon = (DEFAULT_UNUSED_GRADIENT_EPSILON if unused_gradient_epsilon is None
                                        else float(unused_gradient_epsilon))
        
        # State
        self._attached = False
        self._round_data: List[Dict[str, Any]] = []
        self._training_start: Optional[datetime] = None
        self._training_end: Optional[datetime] = None
        
        # Final metrics (set after training)
        self._final_metrics: Dict[str, Any] = {}
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> "TrainingDiagnostics":
        """
        Factory method — returns appropriate subclass instance.
        
        Args:
            model_type: One of 'neural_net', 'xgboost', 'lightgbm', 'catboost'
            **kwargs: Passed to subclass __init__
            
        Returns:
            TrainingDiagnostics subclass instance
        """
        subclasses = {
            "neural_net": NNDiagnostics,
            "xgboost": XGBDiagnostics,
            "lightgbm": LGBDiagnostics,
            "catboost": CatBoostDiagnostics,
        }
        
        if model_type not in subclasses:
            logger.warning(f"Unknown model_type '{model_type}', using base diagnostics")
            # Return a minimal diagnostics that just tracks losses
            return MinimalDiagnostics(**kwargs)
        
        return subclasses[model_type](**kwargs)
    
    @abstractmethod
    def attach(self, model, context: Optional[Dict] = None) -> "TrainingDiagnostics":
        """
        Register hooks/callbacks on the model.
        
        Args:
            model: The model instance to attach to
            context: Optional training context (e.g., data loaders)
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def detach(self) -> None:
        """Remove all hooks/callbacks. Safe to call multiple times."""
        pass
    
    @abstractmethod
    def on_round_end(
        self,
        round_num: int,
        train_loss: float,
        val_loss: float,
        **kwargs
    ) -> None:
        """
        Called after each training epoch/round.
        
        Args:
            round_num: Current epoch/round number (0-indexed)
            train_loss: Training loss for this round
            val_loss: Validation loss for this round
            **kwargs: Model-specific additional metrics
        """
        pass
    
    def set_final_metrics(self, metrics: Dict[str, Any]) -> None:
        """Set final training metrics (MSE, R², etc.) after training completes."""
        self._final_metrics = metrics
        self._training_end = datetime.now()
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate complete diagnostics report.
        
        Returns:
            Dict matching the Phase 1 schema
        """
        try:
            training_summary = self._compute_training_summary()
            diagnosis = self._diagnose(training_summary)
            model_specific = self._get_model_specific()

            report = {
                "status": "complete" if self._round_data else "partial",
                "training_summary": training_summary,
                "diagnosis": diagnosis,
                "model_specific": model_specific,
                "round_data_truncated": len(self._round_data) > 50,
                "round_data_sample": self._round_data[-10:] if self._round_data else [],
            }
            if self.diagnostics_level == "full":
                # CH14-DIAG wave 1: additive fields only; legacy report unchanged.
                self._add_full_report_fields(report)
            return report
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "diagnosis": {
                    "severity": "absent",
                    "issues": [f"Diagnostics failed: {e}"],
                    "suggested_fixes": [],
                    "confidence": 0.0
                }
            }
    
    def _compute_training_summary(self) -> Dict[str, Any]:
        """Compute summary statistics from round data."""
        if not self._round_data:
            return {}
        
        train_losses = [r.get("train_loss", 0) for r in self._round_data]
        val_losses = [r.get("val_loss", 0) for r in self._round_data]
        
        best_val_idx = int(np.argmin(val_losses)) if val_losses else 0
        
        return {
            "rounds_captured": len(self._round_data),
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": min(val_losses) if val_losses else None,
            "best_val_round": best_val_idx,
            "overfit_gap": (val_losses[-1] - train_losses[-1]) if train_losses and val_losses else None,
        }
    
    def _diagnose(self, training_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify severity and detect issues.
        
        Override in subclasses for model-specific diagnosis.
        """
        issues = []
        suggested_fixes = []
        severity = "ok"
        
        # Check overfit gap
        overfit_gap = training_summary.get("overfit_gap")
        if overfit_gap is not None:
            final_train = training_summary.get("final_train_loss", 1)
            if final_train > 0:
                gap_ratio = overfit_gap / final_train
                if gap_ratio > SEVERITY_THRESHOLDS["overfit_gap_ratio"]["critical"]:
                    issues.append(f"Severe overfitting (gap ratio: {gap_ratio:.2f})")
                    suggested_fixes.append("Increase regularization or reduce model complexity")
                    severity = "critical"
                elif gap_ratio > SEVERITY_THRESHOLDS["overfit_gap_ratio"]["warning"]:
                    issues.append(f"Moderate overfitting (gap ratio: {gap_ratio:.2f})")
                    suggested_fixes.append("Consider early stopping or dropout")
                    severity = "warning"
        
        return {
            "severity": severity,
            "issues": issues,
            "suggested_fixes": suggested_fixes,
            "confidence": 0.8 if self._round_data else 0.0
        }
    
    def _get_model_specific(self) -> Dict[str, Any]:
        """Return model-specific diagnostics. Override in subclasses."""
        return {}

    # ── CH14-DIAG wave 1 (level 'full' only) ─────────────────────────
    def _attribution(self) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Per-column attribution vector (positional) and its kind. Override in subclasses."""
        return None, None

    def _feature_name(self, i: int) -> str:
        return self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"

    def _settings_record(self) -> Dict[str, Any]:
        src = (self.column_classes or {}).get("run_level_names_source") or {}
        return {"diagnostics_level": self.diagnostics_level,
                "unused_gradient_epsilon": self.unused_gradient_epsilon,
                "feature_manifest": src.get("path"),
                "feature_manifest_sha256": src.get("sha256")}

    def _add_full_report_fields(self, report: Dict[str, Any]) -> None:
        """Additive report fields for level 'full': names source, column classes, class tags."""
        report["diagnostics_level"] = "full"
        # CD-07: run identity, so a health check can tell this run's report from an older one.
        # Named diagnostics_run_* because compare-models summaries already use run_id (S88 id).
        report["diagnostics_run_id"] = self.run_id
        report["diagnostics_run_started_at"] = self.run_started_at
        # Run record of the report settings in force (switch register: every switch stored).
        report["diagnostics_settings"] = self._settings_record()
        report["round_index_base"] = 0  # best_val_round is a 0-based index into the rounds
        report["feature_names_source"] = self.names_source
        cc = self.column_classes or {}
        by_feature = cc.get("by_feature", {})
        if cc:
            report["column_classes"] = {
                k: v for k, v in cc.items() if k != "by_feature"
            }
            report["column_classes"]["by_feature"] = by_feature
        vec, kind = self._attribution()
        if vec is not None:
            vec = np.asarray(vec, dtype=float)
            report["feature_count"] = int(vec.shape[0])
            report["attribution_kind"] = kind
            if by_feature:
                report["attribution_share_by_class"] = attribution_share_by_class(
                    vec, [self._feature_name(i) for i in range(vec.shape[0])], by_feature
                )
            report["unused_features"] = unused_features(
                vec, kind, [self._feature_name(i) for i in range(vec.shape[0])], by_feature,
                self.unused_gradient_epsilon)
            report["unused_feature_pct"] = report["unused_features"]["pct"]
        # Tag every top-10 line with its column class (or 'unclassified').
        ms = report.get("model_specific", {})
        for block_key in ("feature_importance", "feature_gradients"):
            block = ms.get(block_key)
            if isinstance(block, dict):
                for entry in block.get("top_10", []):
                    entry["column_class"] = by_feature.get(entry.get("feature"), "unclassified")
        for snap in report.get("round_data_sample", []):
            fg = snap.get("feature_gradients")
            if isinstance(fg, dict):
                for entry in fg.get("top_10", []):
                    entry["column_class"] = by_feature.get(entry.get("feature"), "unclassified")

    @property
    def severity(self) -> str:
        """Quick access to current severity level."""
        try:
            report = self.get_report()
            return report.get("diagnosis", {}).get("severity", "absent")
        except:
            return "absent"
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save diagnostics to JSON file.
        
        Args:
            path: Output path. If None, uses default location.
            
        Returns:
            Path where file was written
        """
        if path is None:
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            path = os.path.join(DEFAULT_OUTPUT_DIR, f"{self.model_type}_diagnostics.json")
        
        report = self.get_report()
        report["model_type"] = self.model_type
        report["generated_at"] = datetime.now().isoformat()
        report["schema_version"] = SCHEMA_VERSION
        
        try:
            with open(path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Diagnostics saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save diagnostics: {e}")

        return path

    def save_bound(self, stage_path: str, public_paths: List[str], identity: Dict[str, Any]) -> str:
        """[W3] The report serialized ONCE to bytes (the same text save() writes, plus the identity keys) and
        written first to the run-local staging path, then to each public path. Returns the sha256 of the
        bytes. A failure writing the STAGED copy raises (the caller falls back to save(), unbound); a public
        write failure is logged as save() does."""
        import hashlib
        report = self.get_report()
        report["model_type"] = self.model_type
        report["generated_at"] = datetime.now().isoformat()
        report["schema_version"] = SCHEMA_VERSION
        report.update(identity)
        data = json.dumps(report, indent=2, default=str)
        os.makedirs(os.path.dirname(stage_path) or ".", exist_ok=True)
        with open(stage_path, "w") as f:
            f.write(data)
        for p in public_paths:
            try:
                with open(p, "w") as f:
                    f.write(data)
                logger.info(f"Diagnostics saved to {p}")
            except Exception as e:
                logger.error(f"Failed to save diagnostics: {e}")
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# NEURAL NET DIAGNOSTICS — PyTorch Dynamic Graph Hooks
# =============================================================================

class NNDiagnostics(TrainingDiagnostics):
    """
    Neural network diagnostics using PyTorch hooks.
    
    Hooks into register_forward_hook() and register_full_backward_hook()
    to capture activations, gradients, and dead neuron statistics.
    """
    
    model_type = "neural_net"
    
    def __init__(self, dead_unit_definition: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        # CD-23 (wave 2): which dead-unit definition the report uses (default = today's).
        self.dead_unit_definition = validate_dead_unit_definition(dead_unit_definition)
        self._dead_unit_history: Dict[str, List[float]] = defaultdict(list)
        self._dead_unit_no_activation: List[str] = []
        
        # Hook handles (for cleanup)
        self._forward_hooks = []
        self._backward_hooks = []
        
        # Per-forward-pass storage (cleared each round)
        self._activations: Dict[str, Any] = {}
        self._gradients: Dict[str, Any] = {}
        self._input_gradients: Optional[Any] = None
        
        # Model reference
        self._model = None
        self._layer_names: List[str] = []
        
        # Accumulated stats
        self._dead_neuron_history: Dict[str, List[float]] = defaultdict(list)
        self._gradient_norm_history: Dict[str, List[float]] = defaultdict(list)

        # CD-03 (level 'full'): eval-mode input-gradient probe. None = no probe (legacy).
        self._probe: Optional[Tuple[Any, Any, Any]] = None
        self._probing = False
        self._last_feature_gradients: Optional[np.ndarray] = None
        self._last_feature_gradients_round: Optional[int] = None

    def set_gradient_probe(self, X_probe, y_probe, loss_fn) -> None:
        """
        CD-03: register a fixed training slice for the per-feature gradient probe.

        On each detail round the probe runs one extra forward/backward with the model
        in eval() mode on a requires_grad COPY of X_probe and takes
        torch.autograd.grad(loss, x) only: no parameter .grad is written, no optimizer
        step, BatchNorm running stats are not updated (eval), Dropout is identity
        (eval) so no RNG draw. The model's train/eval mode is restored afterwards.
        """
        self._probe = (X_probe, y_probe, loss_fn)

    _ACTIVATION_TYPES = ("ReLU", "LeakyReLU", "ReLU6", "PReLU", "RReLU", "ELU", "SELU", "CELU",
                         "GELU", "SiLU")

    def _preact_hooks(self, store: Dict[str, Any]) -> list:
        """CD-23: temporary probe-only hooks on the first activation module after each Linear.
        Records, per unit, whether its pre-activation is <= 0 on every probe row."""
        import torch.nn as nn
        handles, last_linear, seen = [], None, set()
        self._dead_unit_no_activation = []
        for name, module in self._model.named_modules():
            if isinstance(module, nn.Linear):
                if last_linear is not None and last_linear not in seen:
                    self._dead_unit_no_activation.append(last_linear)
                last_linear = name
            elif type(module).__name__ in self._ACTIVATION_TYPES and last_linear is not None \
                    and last_linear not in seen:
                seen.add(last_linear)
                def hook(mod, inp, out, _layer=last_linear):
                    x = inp[0].detach()
                    store[_layer] = float((x <= 0).all(dim=0).float().mean().item() * 100)
                handles.append(module.register_forward_hook(hook))
        if last_linear is not None and last_linear not in seen:
            self._dead_unit_no_activation.append(last_linear)
        return handles

    def _run_gradient_probe(self):
        import torch
        X_probe, y_probe, loss_fn = self._probe
        model = self._model
        was_training = model.training
        self._probing = True
        preact: Dict[str, float] = {}
        handles = self._preact_hooks(preact) if self.dead_unit_definition == "always_negative_preact" else []
        try:
            model.eval()
            x = X_probe.detach().clone().requires_grad_(True)
            out = model(x)
            loss = loss_fn(out, y_probe)
            (g,) = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)
            for layer, pct in preact.items():
                self._dead_unit_history[layer].append(pct)
            return g.detach().abs().mean(dim=0)
        finally:
            for h in handles:
                h.remove()
            model.train(was_training)
            self._probing = False

    def _settings_record(self) -> Dict[str, Any]:
        rec = super()._settings_record()
        rec["dead_unit_definition"] = self.dead_unit_definition
        rec["gradient_probe_rows"] = int(self._probe[0].shape[0]) if self._probe is not None else 0
        return rec

    def _dead_history(self) -> Tuple[Dict[str, List[float]], str]:
        """CD-23: (per-layer dead-unit history, definition actually used)."""
        if self.dead_unit_definition == "always_negative_preact" and self._dead_unit_history:
            return self._dead_unit_history, "always_negative_preact"
        return self._dead_neuron_history, "linear_zero"

    def _last_round_with(self, key: str) -> Optional[Dict[str, Any]]:
        """CD-02 producer: newest round snapshot that carries `key` (not only the last round)."""
        for snap in reversed(self._round_data):
            if key in snap:
                return snap
        return None

    def _attribution(self):
        if self._last_feature_gradients is None:
            return None, None
        return self._last_feature_gradients, "input_gradient_mean_abs"

    def attach(self, model, context: Optional[Dict] = None) -> "NNDiagnostics":
        """Register forward and backward hooks on all Linear layers."""
        try:
            import torch.nn as nn
            
            if self._attached:
                self.detach()
            
            self._model = model
            self._layer_names = []
            self._training_start = datetime.now()
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    self._layer_names.append(name)
                    
                    # Forward hook: captures activations
                    fh = module.register_forward_hook(self._make_forward_hook(name))
                    self._forward_hooks.append(fh)
                    
                    # Backward hook: captures gradients
                    bh = module.register_full_backward_hook(self._make_backward_hook(name))
                    self._backward_hooks.append(bh)
            
            self._attached = True
            logger.info(f"NNDiagnostics attached to {len(self._layer_names)} layers: {self._layer_names}")
            
        except Exception as e:
            logger.error(f"Failed to attach NN hooks: {e}")
            self._attached = False
        
        return self
    
    def detach(self) -> None:
        """Remove all hooks."""
        for h in self._forward_hooks:
            try:
                h.remove()
            except:
                pass
        for h in self._backward_hooks:
            try:
                h.remove()
            except:
                pass
        
        self._forward_hooks.clear()
        self._backward_hooks.clear()
        self._attached = False
        logger.info("NNDiagnostics detached")
    
    def _make_forward_hook(self, layer_name: str):
        """Create forward hook that captures activation statistics."""
        def hook(module, input_tensor, output_tensor):
            if self._probing:
                return  # CD-03: the probe pass is not a training/validation pass
            try:
                output = output_tensor.detach()
                self._activations[layer_name] = {
                    "mean": float(output.mean().item()),
                    "std": float(output.std().item()),
                    "dead_pct": float((output == 0).float().mean().item() * 100),
                    "neuron_count": output.shape[-1] if len(output.shape) > 1 else 1,
                }
            except Exception as e:
                logger.debug(f"Forward hook error for {layer_name}: {e}")
        return hook
    
    def _make_backward_hook(self, layer_name: str):
        """Create backward hook that captures gradient statistics."""
        def hook(module, grad_input, grad_output):
            if self._probing:
                return  # CD-03: probe gradients are read from autograd.grad, not hooks
            try:
                if grad_output[0] is not None:
                    grad = grad_output[0].detach()
                    self._gradients[layer_name] = {
                        "norm": float(grad.norm().item()),
                        "mean": float(grad.abs().mean().item()),
                        "max": float(grad.abs().max().item()),
                    }
                
                # Capture input gradients for feature attribution (first layer only)
                if layer_name == self._layer_names[0] and grad_input[0] is not None:
                    self._input_gradients = grad_input[0].detach()
            except Exception as e:
                logger.debug(f"Backward hook error for {layer_name}: {e}")
        return hook
    
    def on_round_end(
        self,
        round_num: int,
        train_loss: float,
        val_loss: float,
        learning_rate: Optional[float] = None,
        **kwargs
    ) -> None:
        """Capture snapshot after each epoch."""
        try:
            # Only capture detailed data every N rounds (to save memory)
            capture_detail = (round_num % self.capture_every_n == 0)
            
            snapshot = {
                "round": round_num,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            
            if learning_rate is not None:
                snapshot["learning_rate"] = float(learning_rate)
            
            if capture_detail and self._activations:
                snapshot["layers"] = {}
                for name in self._layer_names:
                    act = self._activations.get(name, {})
                    grad = self._gradients.get(name, {})
                    snapshot["layers"][name] = {**act, **grad}
                    
                    # Track history for analysis
                    if "dead_pct" in act:
                        self._dead_neuron_history[name].append(act["dead_pct"])
                    if "norm" in grad:
                        self._gradient_norm_history[name].append(grad["norm"])
                
                # CD-03 (level 'full'): eval-mode probe gradients take precedence.
                if self._probe is not None:
                    try:
                        feat_grads = self._run_gradient_probe()
                        top_k = min(10, len(feat_grads))
                        top_indices = feat_grads.argsort(descending=True)[:top_k]
                        self._last_feature_gradients = feat_grads.cpu().numpy().astype(float)
                        self._last_feature_gradients_round = round_num
                        snapshot["feature_gradients"] = {
                            "top_10": [
                                {"feature": self._feature_name(i),
                                 "magnitude": float(feat_grads[i].item())}
                                for i in top_indices.tolist()
                            ],
                            "spread_ratio": float(feat_grads.max().item() / (feat_grads.min().item() + 1e-10)),
                            "source": "eval_probe",
                            "probe_rows": int(self._probe[0].shape[0]),
                        }
                    except Exception as e:
                        logger.debug(f"Gradient probe failed (non-fatal): {e}")
                # Feature gradients (if available)
                elif self._input_gradients is not None:
                    try:
                        feat_grads = self._input_gradients.abs().mean(dim=0)
                        top_k = min(10, len(feat_grads))
                        top_indices = feat_grads.argsort(descending=True)[:top_k]
                        
                        snapshot["feature_gradients"] = {
                            "top_10": [
                                {
                                    "feature": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                                    "magnitude": float(feat_grads[i].item())
                                }
                                for i in top_indices.tolist()
                            ],
                            "spread_ratio": float(feat_grads.max().item() / (feat_grads.min().item() + 1e-10))
                        }
                    except Exception as e:
                        logger.debug(f"Feature gradient capture failed: {e}")
            
            # Respect max_rounds cap
            if len(self._round_data) < self.max_rounds:
                self._round_data.append(snapshot)
            
            # Clear per-pass storage
            self._activations.clear()
            self._gradients.clear()
            self._input_gradients = None
            
        except Exception as e:
            logger.error(f"on_round_end failed: {e}")
    
    def _diagnose(self, training_summary: Dict[str, Any]) -> Dict[str, Any]:
        """NN-specific diagnosis including gradient health and dead neurons."""
        base_diagnosis = super()._diagnose(training_summary)
        issues = base_diagnosis["issues"].copy()
        suggested_fixes = base_diagnosis["suggested_fixes"].copy()
        severity = base_diagnosis["severity"]
        
        # Check dead neurons
        for layer_name, dead_history in self._dead_history()[0].items():
            if dead_history:
                final_dead_pct = dead_history[-1]
                if final_dead_pct > SEVERITY_THRESHOLDS["dead_neuron_pct"]["critical"]:
                    issues.append(f"Dead neurons critical in {layer_name}: {final_dead_pct:.1f}%")
                    suggested_fixes.append("Switch ReLU → LeakyReLU or GELU")
                    severity = "critical"
                elif final_dead_pct > SEVERITY_THRESHOLDS["dead_neuron_pct"]["warning"]:
                    issues.append(f"Dead neurons elevated in {layer_name}: {final_dead_pct:.1f}%")
                    suggested_fixes.append("Consider LeakyReLU or add BatchNorm")
                    if severity == "ok":
                        severity = "warning"
        
        # Check gradient health
        for layer_name, grad_history in self._gradient_norm_history.items():
            if grad_history:
                final_grad_norm = grad_history[-1]
                if final_grad_norm < SEVERITY_THRESHOLDS["gradient_norm_min"]["critical"]:
                    issues.append(f"Vanishing gradients in {layer_name}: norm={final_grad_norm:.2e}")
                    suggested_fixes.append("Reduce network depth or add skip connections")
                    severity = "critical"
                elif final_grad_norm < SEVERITY_THRESHOLDS["gradient_norm_min"]["warning"]:
                    issues.append(f"Weak gradients in {layer_name}: norm={final_grad_norm:.2e}")
                    if severity == "ok":
                        severity = "warning"
        
        # Check gradient spread (feature scale imbalance)
        _fg_snap = self._spread_snapshot()
        if _fg_snap is not None:
            spread = _fg_snap["feature_gradients"].get("spread_ratio", 1)
            if spread > SEVERITY_THRESHOLDS["gradient_spread_ratio"]["critical"]:
                issues.append(f"Feature scale imbalance: gradient spread {spread:.0f}x")
                suggested_fixes.append("Add BatchNorm to input layer or normalize features")
                severity = "critical"
            elif spread > SEVERITY_THRESHOLDS["gradient_spread_ratio"]["warning"]:
                issues.append(f"Moderate feature scale imbalance: gradient spread {spread:.0f}x")
                if severity == "ok":
                    severity = "warning"
        
        return {
            "severity": severity,
            "issues": issues,
            "suggested_fixes": suggested_fixes,
            "confidence": 0.9 if self._round_data else 0.0
        }
    
    def _get_model_specific(self) -> Dict[str, Any]:
        """Return NN-specific gradient health and layer statistics."""
        model_specific = {
            "gradient_health": {
                "vanishing": False,
                "exploding": False,
                "dead_neuron_pct": 0.0,
                "gradient_spread_ratio": 1.0
            },
            "layer_health": {}
        }
        
        # Compute final dead neuron percentage (average across layers)
        total_dead = []
        _dead_hist, _dead_used = self._dead_history()
        for layer_name, dead_history in _dead_hist.items():
            if dead_history:
                model_specific["layer_health"][layer_name] = {
                    "dead_pct": dead_history[-1],
                    "dead_trend": "increasing" if len(dead_history) > 1 and dead_history[-1] > dead_history[0] else "stable"
                }
                total_dead.append(dead_history[-1])
        
        if total_dead:
            model_specific["gradient_health"]["dead_neuron_pct"] = np.mean(total_dead)
        if self.dead_unit_definition != DEFAULT_DEAD_UNIT_DEFINITION:
            # CD-23: say which definition was asked for and which one the numbers use.
            model_specific["gradient_health"]["dead_unit_definition"] = self.dead_unit_definition
            model_specific["gradient_health"]["dead_unit_definition_effective"] = _dead_used
            if _dead_used == "always_negative_preact":
                model_specific["gradient_health"]["dead_unit_layers_without_activation"] = \
                    list(self._dead_unit_no_activation)
        
        # Gradient norms
        for layer_name, grad_history in self._gradient_norm_history.items():
            if grad_history:
                if layer_name in model_specific["layer_health"]:
                    model_specific["layer_health"][layer_name]["gradient_norm"] = grad_history[-1]
                else:
                    model_specific["layer_health"][layer_name] = {"gradient_norm": grad_history[-1]}
                
                # Detect vanishing/exploding
                if grad_history[-1] < 1e-8:
                    model_specific["gradient_health"]["vanishing"] = True
                if grad_history[-1] > 1e6:
                    model_specific["gradient_health"]["exploding"] = True
        
        # Feature gradient spread
        _fg_snap = self._spread_snapshot()
        if _fg_snap is not None:
            model_specific["gradient_health"]["gradient_spread_ratio"] = \
                _fg_snap["feature_gradients"].get("spread_ratio", 1.0)
            if self.diagnostics_level == "full":
                # CD-02/CD-03: the named per-feature gradients behind the spread.
                model_specific["feature_gradients"] = dict(_fg_snap["feature_gradients"])
                model_specific["feature_gradients"]["round"] = _fg_snap.get("round")
        
        return model_specific

    def _spread_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Round snapshot the gradient spread is read from.
        legacy: the last round only, as before (td:595-598 at 41ad509).
        full:   the newest round that carries feature_gradients (CD-02 producer, R-61),
                so a run whose last epoch is not a detail epoch still reports it.
        """
        if not self._round_data:
            return None
        if self.diagnostics_level == "full":
            return self._last_round_with("feature_gradients")
        return self._round_data[-1] if "feature_gradients" in self._round_data[-1] else None


# =============================================================================
# TREE MODEL DIAGNOSTICS — Native Callbacks
# =============================================================================

class TreeDiagnostics(TrainingDiagnostics):
    """Base class for tree model diagnostics (XGBoost, LightGBM, CatBoost)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_importance: Dict[str, float] = {}
        self._evals_result: Dict[str, List[float]] = {}
        self._best_iteration: Optional[int] = None
    
    def attach(self, model, context: Optional[Dict] = None) -> "TreeDiagnostics":
        """Tree models don't need pre-attach hooks — data collected after training."""
        self._attached = True
        self._training_start = datetime.now()
        logger.info(f"{self.model_type} diagnostics attached (post-training collection)")
        return self
    
    def detach(self) -> None:
        """Nothing to detach for tree models."""
        self._attached = False
    
    def on_round_end(
        self,
        round_num: int,
        train_loss: float,
        val_loss: float,
        **kwargs
    ) -> None:
        """Record round data from callbacks."""
        try:
            if len(self._round_data) < self.max_rounds:
                snapshot = {
                    "round": round_num,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
                snapshot.update({k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))})
                self._round_data.append(snapshot)
        except Exception as e:
            logger.error(f"Tree on_round_end failed: {e}")
    
    def set_evals_result(self, evals_result: Dict) -> None:
        """Set evaluation results from model.evals_result() or equivalent."""
        self._evals_result = evals_result
    
    def set_feature_importance(self, importance: Dict[str, float]) -> None:
        """Set feature importance scores."""
        self._feature_importance = importance
    
    def set_best_iteration(self, iteration: int) -> None:
        """Set best iteration (early stopping point)."""
        self._best_iteration = iteration

    # ── CH14-DIAG wave 1, CD-05 (level 'full' only) ──────────────────
    def load_curves(self, rounds, train_curve, val_curve, used_rounds: int,
                    curve_source: str, rounds_run: int, round_stride: int = 1) -> None:
        """
        Load per-round train/val MSE computed after training (staged prediction or the
        library's own eval record). train entries may be None where the kept model has
        no trees for that round (LightGBM trims to its best iteration).
        used_rounds = number of rounds in the model whose predictions are scored.
        """
        self._round_data = []
        for r, t, v in zip(rounds, train_curve, val_curve):
            if len(self._round_data) >= self.max_rounds:
                break
            self._round_data.append({
                "round": int(r),
                "train_loss": None if t is None else float(t),
                "val_loss": float(v),
            })
        self._used_rounds = int(used_rounds)
        self._curve_source = curve_source
        self._rounds_run = int(rounds_run)
        self._round_stride = int(round_stride)

    def set_importance_vector(self, vec, kind: str) -> None:
        """Positional importance vector (all columns, zeros kept) and its kind (CD-05)."""
        self._importance_vector = np.asarray(vec, dtype=float)
        self._importance_kind = kind
        self._feature_importance = {
            self._feature_name(i): float(v) for i, v in enumerate(self._importance_vector)
        }

    def _attribution(self):
        vec = getattr(self, "_importance_vector", None)
        if vec is None:
            return None, None
        return vec, self._importance_kind

    def _compute_training_summary(self) -> Dict[str, Any]:
        if self.diagnostics_level != "full" or not getattr(self, "_curve_source", None):
            return super()._compute_training_summary()
        if not self._round_data:
            return {}
        vals = [r["val_loss"] for r in self._round_data]
        best_idx = int(np.argmin(vals))
        # The scored model: newest captured round inside the kept trees, with a train loss.
        used = None
        for snap in reversed(self._round_data):
            if snap["round"] <= self._used_rounds - 1 and snap["train_loss"] is not None:
                used = snap
                break
        final_train = used["train_loss"] if used else None
        final_val = used["val_loss"] if used else None
        return {
            "rounds_captured": len(self._round_data),
            "final_train_loss": final_train,
            "final_val_loss": final_val,
            "best_val_loss": float(min(vals)),
            "best_val_round": best_idx,
            "overfit_gap": (final_val - final_train) if used else None,
            # additive, level 'full'
            "best_val_round_number": int(self._round_data[best_idx]["round"]),
            "used_round_number": int(used["round"]) if used else None,
            "used_rounds": self._used_rounds,
            "rounds_run": self._rounds_run,
            "round_stride": self._round_stride,
            "curve_source": self._curve_source,
            "loss_metric": "mse",
        }

    def _get_model_specific(self) -> Dict[str, Any]:
        model_specific = self._get_model_specific_base()
        if self.diagnostics_level == "full" and getattr(self, "_importance_kind", None):
            fi = model_specific.get("feature_importance")
            if isinstance(fi, dict):
                fi["importance_kind"] = self._importance_kind
                fi["zero_importance_columns"] = int(np.sum(self._importance_vector == 0))
        return model_specific

    def _get_model_specific_base(self) -> Dict[str, Any]:
        """Return tree-specific feature importance and early stopping info."""
        model_specific = {
            "early_stopping_triggered": self._best_iteration is not None,
            "best_iteration": self._best_iteration,
        }
        
        if self._feature_importance:
            # Sort by importance
            sorted_features = sorted(
                self._feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            model_specific["feature_importance"] = {
                "top_10": [{"feature": k, "importance": v} for k, v in sorted_features[:10]],
                "concentration_ratio": sorted_features[0][1] / (sum(v for _, v in sorted_features) + 1e-10)
                    if sorted_features else 0.0
            }
        
        return model_specific


class XGBDiagnostics(TreeDiagnostics):
    """XGBoost-specific diagnostics using evals_result() and feature_importances_."""
    model_type = "xgboost"


class LGBDiagnostics(TreeDiagnostics):
    """LightGBM-specific diagnostics using record_evaluation and feature_importance()."""
    model_type = "lightgbm"


class CatBoostDiagnostics(TreeDiagnostics):
    """CatBoost-specific diagnostics using get_evals_result() and get_feature_importance()."""
    model_type = "catboost"


class MinimalDiagnostics(TrainingDiagnostics):
    """Fallback diagnostics for unknown model types — tracks losses only."""
    model_type = "unknown"
    
    def attach(self, model, context: Optional[Dict] = None) -> "MinimalDiagnostics":
        self._attached = True
        self._training_start = datetime.now()
        return self
    
    def detach(self) -> None:
        self._attached = False
    
    def on_round_end(self, round_num: int, train_loss: float, val_loss: float, **kwargs) -> None:
        if len(self._round_data) < self.max_rounds:
            self._round_data.append({
                "round": round_num,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            })


# =============================================================================
# MULTI-MODEL COLLECTOR (Option D: Compare Models Support)
# =============================================================================

class MultiModelDiagnostics:
    """
    Collector for multi-model diagnostics during --compare-models runs.
    
    Aggregates diagnostics from all trained models into a single output file
    with comparison metadata.
    """
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        self._model_reports: Dict[str, Dict[str, Any]] = {}
        self._comparison: Dict[str, Any] = {}
        self._mode = "compare_models"
        self._start_time = datetime.now()
    
    def create_for_model(self, model_type: str, **kwargs) -> TrainingDiagnostics:
        """Create diagnostics instance for a specific model type."""
        return TrainingDiagnostics.create(model_type, **kwargs)
    
    def finalize_model(
        self,
        model_type: str,
        diagnostics: TrainingDiagnostics,
        final_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Finalize and store diagnostics for a model after training completes.
        
        Args:
            model_type: The model type identifier
            diagnostics: The diagnostics instance
            final_metrics: Final training metrics (MSE, R², etc.)
        """
        try:
            if final_metrics:
                diagnostics.set_final_metrics(final_metrics)
            
            report = diagnostics.get_report()
            self._model_reports[model_type] = report
            
            logger.info(f"Finalized diagnostics for {model_type}: severity={report.get('diagnosis', {}).get('severity', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to finalize {model_type} diagnostics: {e}")
            self._model_reports[model_type] = {
                "status": "failed",
                "error": str(e),
                "diagnosis": {"severity": "absent", "issues": [], "suggested_fixes": [], "confidence": 0.0}
            }
    
    def set_comparison_result(
        self,
        winner: str,
        ranking: List[str],
        winner_metric: str = "val_loss",
        winner_value: Optional[float] = None
    ) -> None:
        """Set the model comparison results."""
        self._comparison = {
            "winner": winner,
            "winner_metric": winner_metric,
            "winner_value": winner_value,
            "ranking": ranking,
        }
        
        # Compute NN gap to winner (for Option D tracking)
        if "neural_net" in self._model_reports and winner != "neural_net":
            nn_report = self._model_reports["neural_net"]
            nn_val_loss = nn_report.get("training_summary", {}).get("final_val_loss")
            if nn_val_loss is not None and winner_value is not None and winner_value > 0:
                self._comparison["nn_gap_to_winner"] = nn_val_loss / winner_value
    
    def get_combined_report(self) -> Dict[str, Any]:
        """Generate combined multi-model report."""
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "mode": self._mode,
            "models": self._model_reports,
            "comparison": self._comparison,
            "meta": {
                "diagnostics_enabled": True,
                "total_models": len(self._model_reports),
                "total_time_seconds": (datetime.now() - self._start_time).total_seconds()
            }
        }
    
    def save(self, path: Optional[str] = None) -> str:
        """Save combined diagnostics to JSON."""
        if path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, DEFAULT_OUTPUT_FILE)
        
        report = self.get_combined_report()
        
        try:
            with open(path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Multi-model diagnostics saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save multi-model diagnostics: {e}")
        
        # Also save to history
        try:
            os.makedirs(DEFAULT_HISTORY_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_path = os.path.join(DEFAULT_HISTORY_DIR, f"compare_models_{timestamp}.json")
            with open(history_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # FIFO pruning (Phase 5 - Session 71)
            _prune_history_fifo()
            
        except Exception as e:
            logger.debug(f"Failed to save to history: {e}")
        
        return path
    
    def get_nn_diagnostic_summary(self) -> Optional[Dict[str, Any]]:
        """
        Quick access to NN diagnostics for Strategy Advisor.
        
        Returns None if NN wasn't trained or diagnostics failed.
        """
        if "neural_net" not in self._model_reports:
            return None
        
        nn_report = self._model_reports["neural_net"]
        return {
            "severity": nn_report.get("diagnosis", {}).get("severity", "absent"),
            "issues": nn_report.get("diagnosis", {}).get("issues", []),
            "suggested_fixes": nn_report.get("diagnosis", {}).get("suggested_fixes", []),
            "gap_to_winner": self._comparison.get("nn_gap_to_winner"),
            "dead_neuron_pct": nn_report.get("model_specific", {}).get("gradient_health", {}).get("dead_neuron_pct", 0),
        }


# =============================================================================
# CH14-DIAG WAVE 1 HELPERS (level 'full'; nothing here runs at level 'legacy')
# =============================================================================

def classify_columns(
    X,
    feature_names: Optional[List[str]] = None,
    groups=None,
    run_level_names: Optional[List[str]] = None,
    run_level_source: Optional[Dict[str, Any]] = None,
    trial_unavailable_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    CD-13: tag each column of the training matrix by how many values it carries.

      constant  one value over every row of X (NaN-only columns count as constant)
      run       constant AND named in run_level_names (caller-declared run-level set)
      trial     one value within every group of `groups` (e.g. trial id per row), not constant
      local     anything else (varies between rows of one group, or no groups given)

    Computed from the data, not from a feature list (system map §9). Without `groups`,
    'trial' cannot be told from 'local' and trial_class is reported 'unavailable'.
    Reads X in place; no copy of X is made.
    """
    import warnings
    X = np.asarray(X)
    n_rows, n_cols = X.shape
    names = [feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
             for i in range(n_cols)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_min = np.nanmin(X, axis=0) if n_rows else np.zeros(n_cols)
        col_max = np.nanmax(X, axis=0) if n_rows else np.zeros(n_cols)
    constant = (col_min == col_max) | (np.isnan(col_min) & np.isnan(col_max))
    trial_const = np.zeros(n_cols, dtype=bool)
    trial_available = groups is not None
    if trial_available:
        g = np.asarray(groups)
        if g.shape[0] != n_rows:
            raise ValueError(f"groups has {g.shape[0]} rows, X has {n_rows}")
        order = np.argsort(g, kind="stable")
        gs = g[order]
        cuts = np.flatnonzero(gs[1:] != gs[:-1]) + 1
        trial_const[:] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for block in np.split(order, cuts):
                sub = X[block]
                trial_const &= (np.nanmin(sub, axis=0) == np.nanmax(sub, axis=0)) | np.all(np.isnan(sub), axis=0)
    run_set = set(run_level_names or [])
    by_feature = {}
    for i, nm in enumerate(names):
        if constant[i]:
            by_feature[nm] = "run" if nm in run_set else "constant"
        elif trial_available and trial_const[i]:
            by_feature[nm] = "trial"
        else:
            by_feature[nm] = "local"
    counts = {c: 0 for c in COLUMN_CLASSES}
    for c in by_feature.values():
        counts[c] += 1
    out = {
        "rows": int(n_rows),
        "columns": int(n_cols),
        "counts": counts,
        "non_local": int(n_cols - counts["local"]),
        "trial_class": "available" if trial_available else "unavailable",
        "computed_from": "training_matrix",
        "by_feature": by_feature,
    }
    # TODO-CH14-4 (wave 2): where the run-level set came from, and why 'trial' is unavailable.
    if run_level_source is not None:
        out["run_level_names_source"] = dict(run_level_source)
        out["run_level_names_absent_from_data"] = sorted(run_set - set(names))
    if not trial_available and trial_unavailable_reason:
        out["trial_class_reason"] = trial_unavailable_reason
    return out


def attribution_share_by_class(vec, names: List[str], by_feature: Dict[str, str]) -> Dict[str, Any]:
    """CD-13: share of total attribution (importance or |gradient|) carried by each column class."""
    vec = np.abs(np.asarray(vec, dtype=float))
    total = float(vec.sum())
    share = {c: 0.0 for c in COLUMN_CLASSES}
    share["unclassified"] = 0.0
    for v, nm in zip(vec, names):
        share[by_feature.get(nm, "unclassified")] += float(v)
    if total > 0:
        share = {k: v / total for k, v in share.items()}
    share["total"] = total
    return share


def unused_features(vec, kind: Optional[str], names: List[str], by_feature: Dict[str, str],
                    epsilon: float = DEFAULT_UNUSED_GRADIENT_EPSILON) -> Dict[str, Any]:
    """
    CD-08 (R-74): share of columns the model does not use, overall and per column class.
      trees  importance exactly 0 (never split on, or zero gain)
      NN     mean |input gradient| on the probe below `epsilon`
    pct_excluding_constant leaves out the 'constant' and 'run' classes, which no model can use
    within one run, so an expected zero is told from an alarming one.
    """
    vec = np.abs(np.asarray(vec, dtype=float))
    if kind == "input_gradient_mean_abs":
        unused = vec < float(epsilon)
        rule = f"mean |input gradient| < {float(epsilon):g}"
    else:
        unused = vec == 0
        rule = "importance == 0"
    n = int(vec.shape[0])
    by_class: Dict[str, Dict[str, Any]] = {}
    for u, nm in zip(unused.tolist(), names):
        c = by_feature.get(nm, "unclassified") if by_feature else "unclassified"
        b = by_class.setdefault(c, {"columns": 0, "unused": 0})
        b["columns"] += 1
        b["unused"] += int(u)
    for b in by_class.values():
        b["pct"] = 100.0 * b["unused"] / b["columns"] if b["columns"] else None
    keep = [not (by_feature and by_feature.get(nm) in ("constant", "run")) for nm in names]
    n_keep = int(sum(keep))
    u_keep = int(sum(u for u, k in zip(unused.tolist(), keep) if k))
    return {
        "rule": rule,
        "attribution_kind": kind,
        "columns": n,
        "unused": int(unused.sum()),
        "pct": 100.0 * float(unused.sum()) / n if n else None,
        "pct_excluding_constant": 100.0 * u_keep / n_keep if n_keep else None,
        "columns_excluding_constant": n_keep,
        "by_class": by_class,
        "unused_names": [nm for u, nm in zip(unused.tolist(), names) if u],
    }


def _tree_kind(model, model_type: str) -> str:
    """'booster' (native) or 'sklearn' API object."""
    if model_type == "xgboost":
        return "sklearn" if hasattr(model, "get_booster") else "booster"
    if model_type == "lightgbm":
        return "sklearn" if hasattr(model, "booster_") else "booster"
    return "sklearn"  # CatBoostRegressor


def _stage_points(total: int, max_points: int) -> Tuple[List[int], int]:
    """Rounds (1-based counts of trees) at which to evaluate, and the stride."""
    if total <= 0:
        return [], 1
    stride = max(1, int(np.ceil(total / max(1, max_points))))
    pts = list(range(stride, total + 1, stride))
    if pts[-1] != total:
        pts.append(total)
    return pts, stride


def tree_curves(model, model_type: str, X_train, y_train, X_val, y_val,
                max_points: int = 500, lgb_val_record: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    CD-05: per-round train and val MSE after training, without touching the fit call.

      xgboost   staged prediction (iteration_range) over every boosted round; the Booster
                keeps all rounds and its default predict uses all of them.
      lightgbm  staged prediction (num_iteration) over the kept trees; the native Booster is
                trimmed to its best iteration by lgb.train, so rounds after it come from
                `lgb_val_record` (record_evaluation callback, val only) when given.
      catboost  get_evals_result() learn/validation RMSE squared (all rounds run; the model
                itself is trimmed to the best iteration by use_best_model).
    """
    y_train = np.asarray(y_train, dtype=float)
    y_val = np.asarray(y_val, dtype=float)

    def mse(p, y):
        return float(np.mean((np.asarray(p, dtype=float).ravel() - y) ** 2))

    if model_type == "xgboost":
        import xgboost as xgb
        booster = model.get_booster() if _tree_kind(model, model_type) == "sklearn" else model
        total = int(booster.num_boosted_rounds())
        pts, stride = _stage_points(total, max_points)
        dtr, dva = xgb.DMatrix(X_train), xgb.DMatrix(X_val)
        tr = [mse(booster.predict(dtr, iteration_range=(0, i)), y_train) for i in pts]
        va = [mse(booster.predict(dva, iteration_range=(0, i)), y_val) for i in pts]
        return {"rounds": [i - 1 for i in pts], "train": tr, "val": va, "used_rounds": total,
                "rounds_run": total, "stride": stride, "source": "staged_prediction"}

    if model_type == "lightgbm":
        booster = model.booster_ if _tree_kind(model, model_type) == "sklearn" else model
        kept = int(booster.current_iteration())
        run = max(kept, len(lgb_val_record or []))
        pts, stride = _stage_points(run, max_points)
        rounds, tr, va = [], [], []
        for i in pts:
            rounds.append(i - 1)
            if i <= kept:
                tr.append(mse(booster.predict(X_train, num_iteration=i), y_train))
                va.append(mse(booster.predict(X_val, num_iteration=i), y_val))
            else:
                tr.append(None)
                va.append(float(lgb_val_record[i - 1]))
        src = "staged_prediction" + ("+record_evaluation" if run > kept else "")
        return {"rounds": rounds, "train": tr, "val": va, "used_rounds": kept,
                "rounds_run": run, "stride": stride, "source": src}

    if model_type == "catboost":
        ev = model.get_evals_result() or {}
        learn = next(iter(ev.get("learn", {}).values()), [])
        val_key = "validation" if "validation" in ev else next((k for k in ev if k != "learn"), None)
        val = next(iter(ev.get(val_key, {}).values()), []) if val_key else []
        run = len(val)
        pts, stride = _stage_points(run, max_points)
        sq = lambda v: float(v) ** 2  # RMSE -> MSE
        tr = [sq(learn[i - 1]) if i - 1 < len(learn) else None for i in pts]
        va = [sq(val[i - 1]) for i in pts]
        return {"rounds": [i - 1 for i in pts], "train": tr, "val": va,
                "used_rounds": int(model.tree_count_), "rounds_run": run, "stride": stride,
                "source": "get_evals_result_rmse_squared"}

    raise ValueError(f"tree_curves: unsupported model_type {model_type!r}")


def tree_importance(model, model_type: str, n_features: int) -> Tuple[np.ndarray, str]:
    """CD-05: positional importance over all columns (never-split columns are 0)."""
    vec = np.zeros(n_features, dtype=float)
    if model_type == "xgboost":
        booster = model.get_booster() if _tree_kind(model, model_type) == "sklearn" else model
        score = booster.get_score(importance_type="gain")
        fn = booster.feature_names
        for k, v in score.items():
            if fn and k in fn:
                idx = fn.index(k)
            elif k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
            else:
                continue
            if idx < n_features:
                vec[idx] = float(v)
        return vec, "gain"
    if model_type == "lightgbm":
        booster = model.booster_ if _tree_kind(model, model_type) == "sklearn" else model
        imp = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
        vec[: min(n_features, imp.shape[0])] = imp[:n_features]
        return vec, "gain"
    if model_type == "catboost":
        imp = np.asarray(model.get_feature_importance(), dtype=float)
        vec[: min(n_features, imp.shape[0])] = imp[:n_features]
        return vec, "prediction_values_change"
    raise ValueError(f"tree_importance: unsupported model_type {model_type!r}")


def tree_best_iteration(model, model_type: str) -> Optional[int]:
    try:
        if model_type == "catboost":
            return model.get_best_iteration()
        bi = getattr(model, "best_iteration", None)
        if bi is None and hasattr(model, "best_iteration_"):
            bi = model.best_iteration_
        return None if bi is None else int(bi)
    except Exception:
        return None


TRIAL_ID_UNAVAILABLE_REASON = ("no trial id: the Step 5 rows (survivors_with_scores.json top-level "
                               "keys seed, score, metadata, features, holdout_*) and the feature "
                               "manifest carry none")


def classify_from_options(X, feature_names: Optional[List[str]],
                          options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """CD-13 with TODO-CH14-4: run-level set from the feature manifest named in `options`."""
    opts = options or {}
    names, src = load_run_level_names(opts.get("feature_manifest"))
    return classify_columns(X, feature_names, run_level_names=names, run_level_source=src,
                            trial_unavailable_reason=TRIAL_ID_UNAVAILABLE_REASON)


def _stamp_kwargs(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    opts = options or {}
    return {"run_id": opts.get("run_id"), "run_started_at": opts.get("run_started_at"),
            "unused_gradient_epsilon": opts.get("unused_gradient_epsilon")}


def make_nn_diagnostics_full(options: Dict[str, Any], X_train, X_probe=None, y_probe=None,
                             loss_fn=None) -> "NNDiagnostics":
    """Level-'full' NNDiagnostics from the options bundle (tst attach block). Probe set when rows > 0."""
    names = options.get("feature_names")
    diag = NNDiagnostics(
        feature_names=names, names_source=options.get("names_source"), diagnostics_level="full",
        column_classes=classify_from_options(X_train, names, options),
        dead_unit_definition=options.get("dead_unit_definition"), **_stamp_kwargs(options),
    )
    rows = options.get("gradient_probe_rows")
    rows = DEFAULT_GRADIENT_PROBE_ROWS if rows is None else int(rows)
    if rows > 0 and X_probe is not None:
        diag.set_gradient_probe(X_probe[:rows], y_probe[:rows], loss_fn)
    return diag


def build_tree_diagnostics_full(
    model, model_type: str, X_train, y_train, X_val, y_val,
    feature_names: Optional[List[str]] = None, names_source: Optional[str] = None,
    final_metrics: Optional[Dict[str, Any]] = None, max_points: Optional[int] = None,
    lgb_val_record: Optional[List[float]] = None, column_classes: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> "TreeDiagnostics":
    """CD-05 + CD-04 + CD-13: a complete level-'full' tree report object, built after training.
    `options` (wave 2): run id stamp (CD-07), unused epsilon (CD-08), feature manifest (TODO-CH14-4)."""
    n_features = int(np.asarray(X_train).shape[1])
    if column_classes is None:
        column_classes = (classify_from_options(X_train, feature_names, options) if options is not None
                          else classify_columns(X_train, feature_names))
    diag = TrainingDiagnostics.create(
        model_type, feature_names=feature_names, diagnostics_level="full",
        names_source=names_source, column_classes=column_classes, **_stamp_kwargs(options),
    )
    diag.attach(model)
    cur = tree_curves(model, model_type, X_train, y_train, X_val, y_val,
                      max_points=max_points or diag.max_rounds, lgb_val_record=lgb_val_record)
    diag.load_curves(cur["rounds"], cur["train"], cur["val"], used_rounds=cur["used_rounds"],
                     curve_source=cur["source"], rounds_run=cur["rounds_run"], round_stride=cur["stride"])
    vec, kind = tree_importance(model, model_type, n_features)
    diag.set_importance_vector(vec, kind)
    bi = tree_best_iteration(model, model_type)
    if bi is not None:
        diag.set_best_iteration(bi)
    if final_metrics:
        diag.set_final_metrics(final_metrics)
    diag.detach()
    return diag


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_diagnostics(path: str = None) -> Optional[Dict[str, Any]]:
    """
    Load diagnostics from JSON file.
    
    Args:
        path: Path to diagnostics file. If None, uses default location.
        
    Returns:
        Diagnostics dict or None if file doesn't exist/is invalid
    """
    if path is None:
        path = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILE)
    
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load diagnostics from {path}: {e}")
    
    return None


def get_severity(path: str = None) -> str:
    """
    Quick check of diagnostics severity.
    
    Returns: 'ok' | 'warning' | 'critical' | 'absent'
    """
    diag = load_diagnostics(path)
    
    if diag is None:
        return "absent"
    
    # Multi-model format
    if "models" in diag:
        # Return worst severity across all models
        severities = []
        for model_report in diag["models"].values():
            sev = model_report.get("diagnosis", {}).get("severity", "absent")
            severities.append(sev)
        
        if "critical" in severities:
            return "critical"
        if "warning" in severities:
            return "warning"
        if "ok" in severities:
            return "ok"
        return "absent"
    
    # Single-model format
    return diag.get("diagnosis", {}).get("severity", "absent")


def _prune_history_fifo(history_dir: str = DEFAULT_HISTORY_DIR, max_files: int = MAX_HISTORY_FILES,
                        pattern: str = "compare_models_*.json", mode: str = "delete",
                        archive_dir: Optional[str] = None,
                        exclude_prefixes: Tuple[str, ...] = ()) -> Optional[Dict[str, Any]]:
    """
    FIFO pruning for diagnostics history directory.
    
    Keeps the newest max_files, deletes oldest by mtime.
    Single log line per prune event (not per file).
    
    Args:
        history_dir: Path to history directory
        max_files: Maximum files to keep
        pattern, mode, archive_dir, exclude_prefixes: [CH14-DIAG CD-10, wave 2] the defaults
            are the original behaviour (compare_models_*.json, delete). mode 'move' moves the
            oldest files into archive_dir instead of deleting them; files whose name starts
            with one of exclude_prefixes are never candidates and are counted as 'excluded'.

    Returns:
        summary dict (candidates, kept, pruned, mode, excluded) or None when nothing to do.
    """
    try:
        if mode not in ("delete", "move"):
            raise ValueError(f"history prune mode must be delete or move, got {mode!r}")
        history_path = Path(history_dir)
        if not history_path.exists() or not history_path.is_dir():
            return None
        
        # Get only compare_models history files (not other artifacts)
        matched = list(history_path.glob(pattern))
        excluded = [f for f in matched if f.name.startswith(tuple(exclude_prefixes))] if exclude_prefixes else []
        files = [(f, f.stat().st_mtime) for f in matched if f not in excluded and f.is_file()]
        
        if len(files) <= max_files:
            return None  # Nothing to prune
        
        # Sort by mtime (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Delete oldest files
        to_delete = files[:len(files) - max_files]
        if mode == "move":
            dest = Path(archive_dir or (str(history_path).rstrip("/") + "_archive"))
            dest.mkdir(parents=True, exist_ok=True)
            for f, _ in to_delete:
                os.replace(f, dest / f.name)
        else:
            for f, _ in to_delete:
                f.unlink()
        
        logger.info(f"FIFO pruned {len(to_delete)} diagnostics files (kept {max_files})"
                    + (f" [{mode}; {len(excluded)} excluded by prefix]" if (mode != "delete" or excluded) else ""))
        return {"candidates": len(files), "kept": max_files, "pruned": len(to_delete), "mode": mode,
                "excluded": len(excluded), "pruned_names": [f.name for f, _ in to_delete]}
        
    except Exception as e:
        logger.debug(f"History pruning failed (non-fatal): {e}")
        return None


# =============================================================================
# CLI INTERFACE (for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Diagnostics CLI")
    parser.add_argument("--test-nn", action="store_true", help="Run NN diagnostics test")
    parser.add_argument("--test-tree", action="store_true", help="Run tree diagnostics test")
    parser.add_argument("--test-fifo", action="store_true", help="Test FIFO pruning")
    parser.add_argument("--load", type=str, help="Load and display diagnostics from file")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.load:
        diag = load_diagnostics(args.load)
        if diag:
            print(json.dumps(diag, indent=2))
        else:
            print(f"Failed to load diagnostics from {args.load}")
    
    elif args.test_nn:
        print("Testing NN Diagnostics...")
        try:
            import torch
            import torch.nn as nn
            
            # Create simple model
            model = nn.Sequential(
                nn.Linear(47, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
            # Attach diagnostics
            diag = NNDiagnostics(capture_every_n=1)
            diag.attach(model)
            
            # Simulate training
            for epoch in range(10):
                x = torch.randn(32, 47)
                y = model(x)
                loss = y.sum()
                loss.backward()
                
                diag.on_round_end(epoch, train_loss=0.5 - epoch*0.04, val_loss=0.6 - epoch*0.03)
            
            diag.detach()
            
            # Get report
            report = diag.get_report()
            print(f"Status: {report['status']}")
            print(f"Severity: {report['diagnosis']['severity']}")
            print(f"Issues: {report['diagnosis']['issues']}")
            print(f"Rounds captured: {report['training_summary']['rounds_captured']}")
            
            # Save
            path = diag.save()
            print(f"Saved to: {path}")
            
        except ImportError:
            print("PyTorch not available for NN test")
    
    elif args.test_tree:
        print("Testing Tree Diagnostics...")
        
        diag = CatBoostDiagnostics()
        diag.attach(None)
        
        # Simulate training rounds
        for round_num in range(20):
            diag.on_round_end(round_num, train_loss=0.01 - round_num*0.0004, val_loss=0.012 - round_num*0.0003)
        
        diag.set_feature_importance({
            "intersection_weight": 0.234,
            "bidirectional_selectivity": 0.189,
            "skip_entropy": 0.145,
        })
        diag.set_best_iteration(15)
        diag.detach()
        
        report = diag.get_report()
        print(f"Status: {report['status']}")
        print(f"Severity: {report['diagnosis']['severity']}")
        print(f"Best iteration: {report['model_specific']['best_iteration']}")
        
        path = diag.save()
        print(f"Saved to: {path}")
    
    elif args.test_fifo:
        print("Testing FIFO Pruning...")
        print(f"MAX_HISTORY_FILES = {MAX_HISTORY_FILES}")
        print(f"History dir: {DEFAULT_HISTORY_DIR}")
        
        # Check current state
        history_path = Path(DEFAULT_HISTORY_DIR)
        if history_path.exists():
            files = list(history_path.glob("*.json"))
            print(f"Current files: {len(files)}")
            
            # Run pruning
            _prune_history_fifo()
            
            files_after = list(history_path.glob("*.json"))
            print(f"Files after prune: {len(files_after)}")
        else:
            print(f"History directory does not exist: {DEFAULT_HISTORY_DIR}")
    
    else:
        parser.print_help()
