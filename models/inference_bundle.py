#!/usr/bin/env python3
"""
models/inference_bundle.py - [LL-6, Ch14 learning loop, 2026-11-23]

ONE way to load a Step 5 model for inference, shared by Step 6
(prediction_generator.py) and the Chapter 13 attribution path.

System map v1.2 §4.2, Blocker 3 ruling: "replace the duplicated loader with the
canonical mechanism returning an explicit inference bundle (model, model_type,
feature_names, feature_schema_hash, feature_count, normalize_features,
scaler_mean, scaler_scale, checkpoint_hash, checkpoint_version)".
Gate A-NORM (same section): for the same survivor, Step 6's prediction equals
the attribution path's forward prediction within a declared tolerance.

- The model is loaded by models.model_factory.load_model_from_sidecar, exactly
  as Step 6 does (prediction_generator.py, __init__ STEP 4).
- The NN scaler is read by load_nn_scaler(), which IS Step 6's former
  _load_nn_scaler body, moved here unchanged (same rules, same log lines);
  PredictionGenerator._load_nn_scaler now calls it.
- Feature names: Step 6's rule, feature_schema.per_seed_feature_names, else
  feature_schema.feature_names (prediction_generator.py, STEP 4).
- Rows -> X by NAME. Step 6 fills a missing name with 0.0; the bundle refuses
  (fail closed) unless missing='zero' is asked for, which reproduces Step 6.

One declared difference from Step 6, kept visible rather than copied:
  Step 6 reads the scaler from the sidecar's checkpoint_path AS WRITTEN, while
  load_model_from_sidecar loads the model from models_dir/<name> when that path
  does not exist. A relocated NN directory therefore gets a model WITHOUT its
  scaler in Step 6. The bundle reads the scaler from the file the model was
  actually loaded from (scaler_source='resolved', default), or reproduces
  Step 6 (scaler_source='as_written'). Gate A-NORM measures both.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

SCALER_SOURCES = ("resolved", "as_written")
MISSING_POLICIES = ("refuse", "zero")


class BundleError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Step 6's scaler loader (moved from prediction_generator.py, body unchanged)
# ---------------------------------------------------------------------------
def load_nn_scaler(checkpoint_path, logger: logging.Logger) -> dict:
    """
    Category B Phase 3: Load scaler parameters from NN checkpoint.

    Team Beta decisions:
      - Store mean/scale as arrays, not pickled sklearn (portable)
      - Best-effort: if absent (old checkpoints), proceed without transform + warn
      - Zero-scale elements clamped to 1.0 to prevent NaN/inf
      - Only called when model_type == 'neural_net'

    Returns {"scaler_mean", "scaler_scale", "normalize_features"}; the two arrays
    are None and the flag False in every best-effort exit, as before.
    """
    out = {"scaler_mean": None, "scaler_scale": None, "normalize_features": False}
    try:
        import torch
    except ImportError:
        logger.warning("[CAT-B] PyTorch not available — cannot load NN scaler")
        return out

    if not checkpoint_path or not Path(checkpoint_path).exists():
        logger.warning(f"[CAT-B] NN checkpoint not found: {checkpoint_path}")
        logger.warning("[CAT-B] Proceeding without scaler (best-effort)")
        return out

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        logger.warning(f"[CAT-B] Cannot read checkpoint: {e}")
        logger.warning("[CAT-B] Proceeding without scaler (best-effort)")
        return out

    if not isinstance(checkpoint, dict):
        logger.info("[CAT-B] Checkpoint is not a dict — no scaler metadata")
        return out

    normalize_flag = checkpoint.get('normalize_features', False)
    if not normalize_flag:
        logger.info("[CAT-B] NN checkpoint: normalize_features=False (no scaler needed)")
        return out

    scaler_mean = checkpoint.get('scaler_mean')
    scaler_scale = checkpoint.get('scaler_scale')

    if scaler_mean is None or scaler_scale is None:
        logger.warning("[CAT-B] normalize_features=True but scaler arrays missing!")
        logger.warning("[CAT-B] Proceeding without scaler (best-effort)")
        return out

    scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
    scaler_scale = np.asarray(scaler_scale, dtype=np.float32)

    # Fix 3A-3: Clamp zero/tiny scale to 1.0 to prevent NaN/inf
    zero_mask = scaler_scale < 1e-12
    n_clamped = int(zero_mask.sum())
    if n_clamped > 0:
        scaler_scale[zero_mask] = 1.0
        logger.warning(f"[CAT-B] Clamped {n_clamped} zero-scale features to 1.0")

    use_leaky = checkpoint.get('use_leaky_relu', False)
    logger.info(f"[CAT-B] NN scaler loaded: {len(scaler_mean)} features")
    logger.info(f"[CAT-B]   scale range: [{scaler_scale.min():.4f}, {scaler_scale.max():.4f}]")
    logger.info(f"[CAT-B]   use_leaky_relu: {use_leaky}")
    out.update(scaler_mean=scaler_mean, scaler_scale=scaler_scale, normalize_features=True)
    return out


def sidecar_feature_names(meta: dict) -> List[str]:
    """Step 6's rule (prediction_generator.py STEP 4)."""
    fs = meta.get("feature_schema", {}) or {}
    return list(fs.get("per_seed_feature_names", fs.get("feature_names", [])) or [])


def resolve_checkpoint_path(models_dir, meta: dict) -> str:
    """The file load_model_from_sidecar actually loads (model_factory.py, same rule:
    the written path if it exists, else models_dir/<basename>)."""
    cp = meta["checkpoint_path"]
    if Path(cp).exists():
        return str(cp)
    alt = Path(models_dir) / Path(cp).name
    if alt.exists():
        return str(alt)
    raise BundleError(f"checkpoint not found: {cp} (also tried {alt})")


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@dataclass
class InferenceBundle:
    model: object                 # the model_factory wrapper (Step 6's self.model)
    model_type: str
    feature_names: List[str]
    feature_schema_hash: Optional[str]
    feature_count: int
    normalize_features: bool
    scaler_mean: Optional[np.ndarray]
    scaler_scale: Optional[np.ndarray]
    checkpoint_path: str
    checkpoint_hash: str
    checkpoint_version: Optional[str]
    models_dir: str
    scaler_source: str
    meta: dict = field(repr=False, default_factory=dict)

    # -- the object each attribution backend expects -----------------------
    @property
    def attribution_model(self):
        """neural_net -> nn.Module; xgboost/lightgbm -> Booster; catboost -> CatBoostRegressor."""
        w = self.model
        if self.model_type == "neural_net":
            return w.model
        if self.model_type in ("xgboost", "lightgbm"):
            return getattr(w, "_booster", None) or w.model
        if self.model_type == "catboost":
            return w.model
        raise BundleError(f"unsupported model_type {self.model_type}")

    @property
    def y_mean(self):
        return getattr(self.model, "_y_mean", None)

    @property
    def y_std(self):
        return getattr(self.model, "_y_std", None)

    def vector(self, features: Dict[str, float], missing: str = "refuse") -> np.ndarray:
        """One row by name, in the sidecar's order."""
        if missing not in MISSING_POLICIES:
            raise BundleError(f"missing={missing!r} not in {MISSING_POLICIES}")
        if not isinstance(features, dict):
            raise BundleError(f"features must be a name-keyed dict, got {type(features).__name__}")
        absent = [n for n in self.feature_names if n not in features]
        if absent and missing == "refuse":
            raise BundleError(f"{len(absent)} sidecar features absent from the row: {absent[:5]}")
        return np.array([float(features.get(n, 0.0)) for n in self.feature_names], dtype=np.float32)

    def matrix(self, rows: Iterable[Dict[str, float]], missing: str = "refuse") -> np.ndarray:
        return np.stack([self.vector(r, missing) for r in rows]).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Step 6's normalisation rule (prediction_generator.py, after X is built):
        NN with a loaded scaler only. A width mismatch fails closed here (Step 6
        logs and skips)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.feature_count:
            raise BundleError(f"X shape {X.shape} != (n, {self.feature_count})")
        if self.normalize_features and self.scaler_mean is not None and self.model_type == "neural_net":
            if X.shape[1] != len(self.scaler_mean):
                raise BundleError(f"scaler width {len(self.scaler_mean)} != {X.shape[1]}")
            X = (X - self.scaler_mean) / self.scaler_scale
        return X

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        """Step 6's model call on Step 6's transform (wrapper.predict)."""
        return np.asarray(self.model.predict(self.transform(X_raw))).reshape(-1)

    def describe(self) -> dict:
        return {"model_type": self.model_type, "feature_count": self.feature_count,
                "feature_schema_hash": self.feature_schema_hash,
                "normalize_features": self.normalize_features,
                "checkpoint_path": self.checkpoint_path, "checkpoint_hash": self.checkpoint_hash,
                "checkpoint_version": self.checkpoint_version, "models_dir": self.models_dir,
                "scaler_source": self.scaler_source,
                "y_mean": None if self.y_mean is None else float(self.y_mean),
                "y_std": None if self.y_std is None else float(self.y_std)}


def load_inference_bundle(models_dir: str, device: str = "cpu",
                          scaler_source: str = "resolved",
                          logger: Optional[logging.Logger] = None) -> InferenceBundle:
    """Load a Step 5 model the way Step 6 does, as one explicit bundle."""
    from models.model_factory import load_model_from_sidecar
    log = logger or logging.getLogger("inference_bundle")
    if scaler_source not in SCALER_SOURCES:
        raise BundleError(f"scaler_source={scaler_source!r} not in {SCALER_SOURCES}")
    meta_path = Path(models_dir) / "best_model.meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    sq = meta.get("signal_quality", {}) or {}
    if not sq.get("prediction_allowed", True):
        raise BundleError(f"sidecar signal gate blocks prediction: {sq.get('signal_status')}")
    if meta.get("checkpoint_path") is None:
        raise BundleError("checkpoint_path is null")
    model, meta2 = load_model_from_sidecar(models_dir=models_dir, device=device)
    model_type = meta2["model_type"]
    resolved = resolve_checkpoint_path(models_dir, meta2)
    names = sidecar_feature_names(meta2)
    if not names:
        raise BundleError("sidecar has no feature names (Step 6 would need require_feature_schema=False)")
    fs = meta2.get("feature_schema", {}) or {}
    sc = {"scaler_mean": None, "scaler_scale": None, "normalize_features": False}
    if model_type == "neural_net":
        src = resolved if scaler_source == "resolved" else meta2["checkpoint_path"]
        sc = load_nn_scaler(src, log)
    fc = int(fs.get("feature_count", len(names)) or len(names))
    if fc != len(names):
        raise BundleError(f"sidecar feature_count {fc} != {len(names)} names")
    return InferenceBundle(
        model=model, model_type=model_type, feature_names=names,
        feature_schema_hash=fs.get("feature_schema_hash"), feature_count=len(names),
        normalize_features=bool(sc["normalize_features"]), scaler_mean=sc["scaler_mean"],
        scaler_scale=sc["scaler_scale"], checkpoint_path=resolved,
        checkpoint_hash=_sha256(resolved), checkpoint_version=meta2.get("schema_version"),
        models_dir=str(models_dir), scaler_source=scaler_source, meta=meta2)
