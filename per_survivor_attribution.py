#!/usr/bin/env python3
"""
Per-Survivor Attribution Module — Chapter 14, Phase 2
=====================================================
Version: 1.0.0
Date: 2026-02-13
Session: 83

PURPOSE:
    Global feature importance (Chapter 11) answers: "Which features matter on average?"
    Per-survivor attribution answers: "For THIS specific seed, which features drove
    its prediction?"

    Critical for pool strategy: if Top 20 survivors are driven by different features
    than Top 300 survivors, they're structurally different populations requiring
    different optimization strategies.

BACKENDS:
    - Neural Net: Input gradient attribution (grad or grad_x_input)
    - XGBoost: pred_contribs (native tree decomposition)
    - LightGBM: pred_contrib (native tree decomposition)
    - CatBoost: ShapValues (C++ SHAP implementation)

INVARIANTS:
    - All backends return dict mapping feature_name → normalized attribution score
    - Scores are absolute values, normalized to sum to 1.0
    - Best-effort, non-fatal (Ch14 invariant) — returns empty dict on failure
    - No side effects on model state

CONSUMED BY:
    - chapter_13_orchestrator.py: post_draw_root_cause_analysis()
    - Strategy Advisor: POOL_PRECISION focus recommendations
    - Web dashboard: /training route (tier comparison chart)
"""

import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger("per_survivor_attribution")

__version__ = "1.0.1"  # TB review fixes: device auto-detect, zero_grad, sampling, failure tracking


# ===========================================================================
# Backend: Neural Net — Input Gradient Attribution
# ===========================================================================

def per_survivor_attribution_nn(
    model,
    features: np.ndarray,
    feature_names: List[str],
    method: str = "grad_x_input",
) -> Dict[str, float]:
    """
    Compute per-seed feature attribution via input gradients.

    Uses PyTorch dynamic graph: forward pass builds graph,
    backward pass computes d(prediction)/d(each_input_feature).

    Device is auto-detected from model parameters (GPU isolation safe).

    Methods:
        'grad'         — |∂y/∂x|  (raw gradient magnitude)
        'grad_x_input' — |x * ∂y/∂x|  (gradient weighted by input value)

    grad_x_input is the default because it handles differently-scaled features
    more stably (avoids over-emphasizing small-magnitude features with large
    gradients). No extra graph cost — same backward pass, one extra multiply.

    Args:
        model: Trained SurvivorQualityNet (or any nn.Module)
        features: np.ndarray shape (N_features,) — one survivor's features
        feature_names: List[str] of feature names
        method: 'grad' or 'grad_x_input' (default)

    Returns:
        dict mapping feature_name → attribution_score (normalized, sums to ~1.0)
    """
    try:
        import torch

        # TB Finding #1: Auto-detect device from model to prevent
        # CUDA init in parent process (GPU isolation invariant, S72)
        try:
            resolved_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            resolved_device = torch.device("cpu")

        was_training = model.training
        model.eval()

        # TB Finding #2: Zero gradients before backward to prevent
        # accumulation when called in loops (compare_pool_tiers)
        model.zero_grad(set_to_none=True)

        x = torch.tensor(features, dtype=torch.float32, device=resolved_device)
        x = x.unsqueeze(0).requires_grad_(True)  # [1, N_features]

        # Forward pass — dynamic graph built
        prediction = model(x)

        # Backward pass — graph traversed, gradients computed
        prediction.backward(retain_graph=False)

        # Attribution method selection
        if method == "grad_x_input":
            grads = (x.grad * x).squeeze().abs()
        else:
            grads = x.grad.squeeze().abs()

        # Normalize to sum to 1
        total = grads.sum()
        if total > 0:
            grads = grads / total

        result = {name: grads[i].item() for i, name in enumerate(feature_names)}

        # Restore model state
        if was_training:
            model.train()

        return result

    except Exception as e:
        logger.warning(f"NN attribution failed: {e}")
        return {}


# ===========================================================================
# Backend: XGBoost — pred_contribs
# ===========================================================================

def per_survivor_attribution_xgb(
    model,
    features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    XGBoost native per-sample contribution.

    Each tree votes on the prediction. pred_contribs decomposes
    the final prediction into per-feature contributions.

    Args:
        model: Trained XGBoost model (Booster or sklearn wrapper)
        features: np.ndarray shape (N_features,)
        feature_names: List[str]

    Returns:
        dict mapping feature_name → attribution_score (normalized)
    """
    try:
        import xgboost as xgb

        # Handle both sklearn wrapper and raw Booster
        booster = model.get_booster() if hasattr(model, "get_booster") else model

        dmatrix = xgb.DMatrix(
            features.reshape(1, -1),
            feature_names=feature_names,
        )
        contributions = booster.predict(dmatrix, pred_contribs=True)
        # contributions shape: [1, N_features + 1] — last is bias term

        raw = contributions[0][:-1]  # Drop bias
        total = np.abs(raw).sum()
        if total > 0:
            normalized = np.abs(raw) / total
        else:
            normalized = raw

        return {name: float(normalized[i]) for i, name in enumerate(feature_names)}

    except Exception as e:
        logger.warning(f"XGBoost attribution failed: {e}")
        return {}


# ===========================================================================
# Backend: LightGBM — pred_contrib
# ===========================================================================

def per_survivor_attribution_lgb(
    model,
    features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    LightGBM native per-sample contribution.
    Same concept as XGBoost — decomposes prediction into feature contributions.

    Args:
        model: Trained LightGBM model (Booster or sklearn wrapper)
        features: np.ndarray shape (N_features,)
        feature_names: List[str]

    Returns:
        dict mapping feature_name → attribution_score (normalized)
    """
    try:
        # Handle both sklearn wrapper and raw Booster
        if hasattr(model, "predict"):
            contributions = model.predict(
                features.reshape(1, -1),
                pred_contrib=True,
            )
        else:
            logger.warning("LightGBM model type not recognized for pred_contrib")
            return {}

        # contributions shape: [1, N_features + 1] — last is bias
        raw = contributions[0][:-1]
        total = np.abs(raw).sum()
        if total > 0:
            normalized = np.abs(raw) / total
        else:
            normalized = raw

        return {name: float(normalized[i]) for i, name in enumerate(feature_names)}

    except Exception as e:
        logger.warning(f"LightGBM attribution failed: {e}")
        return {}


# ===========================================================================
# Backend: CatBoost — Native SHAP (C++ Implementation)
# ===========================================================================

def per_survivor_attribution_catboost(
    model,
    features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    CatBoost native SHAP values — implemented in C++, much faster
    than the Python shap library.

    SHAP provides theoretically grounded attribution:
    each feature's contribution is its average marginal
    contribution across all possible feature coalitions.

    Args:
        model: Trained CatBoost model
        features: np.ndarray shape (N_features,)
        feature_names: List[str]

    Returns:
        dict mapping feature_name → attribution_score (normalized)
    """
    try:
        from catboost import Pool

        pool = Pool(
            features.reshape(1, -1),
            feature_names=feature_names,
        )
        shap_values = model.get_feature_importance(pool, type="ShapValues")
        # shap_values shape: [1, N_features + 1] — last is expected value

        raw = shap_values[0][:-1]
        total = np.abs(raw).sum()
        if total > 0:
            normalized = np.abs(raw) / total
        else:
            normalized = raw

        return {name: float(normalized[i]) for i, name in enumerate(feature_names)}

    except Exception as e:
        logger.warning(f"CatBoost attribution failed: {e}")
        return {}


# ===========================================================================
# Unified Interface
# ===========================================================================

# Backend dispatch table
_ATTRIBUTION_BACKENDS = {
    "neural_net": per_survivor_attribution_nn,
    "xgboost": per_survivor_attribution_xgb,
    "lightgbm": per_survivor_attribution_lgb,
    "catboost": per_survivor_attribution_catboost,
}


def per_survivor_attribution(
    model,
    model_type: str,
    features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Unified per-survivor attribution across all model types.

    Routes to the appropriate backend based on model_type.
    NN backend auto-detects device from model (GPU isolation safe).

    Args:
        model: Trained model (any supported type)
        model_type: str — 'neural_net', 'xgboost', 'lightgbm', or 'catboost'
        features: np.ndarray shape (N_features,) — one survivor's feature vector
        feature_names: List[str] of feature names

    Returns:
        dict mapping feature_name → attribution_score (normalized, sums to ~1.0)
        Returns empty dict on failure (Ch14 best-effort invariant)
    """
    backend = _ATTRIBUTION_BACKENDS.get(model_type)
    if backend is None:
        logger.warning(f"Unknown model_type for attribution: {model_type}")
        return {}

    return backend(model, features, feature_names)


# ===========================================================================
# [LL-7, Ch14 learning loop, 2026-11-23] Signed attribution, by name
# ===========================================================================
# The four functions above are unchanged: they return |contribution| shares
# that sum to 1. The signed path below is added beside them (system map v1.2 §5;
# plan LL-7). For every survivor it returns, in the model's TARGET units:
#   signed[name]          which way and how far each feature pushed
#   baseline              the model's output at the reference point
#   prediction            the attribution path's OWN forward pass (A-NORM uses it)
#   completeness_residual prediction - (baseline + sum(signed))
# Trees (pred_contribs / pred_contrib / ShapValues) are exactly complete up to
# float error. The NN uses integrated gradients from a declared reference
# (input zero in the model's input space = the training mean when a scaler is
# loaded); its residual is the IG discretisation error and is reported, not hidden.
# Inputs are name-keyed: X is built from a {name: value} map in the sidecar's
# order (models.inference_bundle), and counts and names are checked, never zipped
# blind.

SIGNED_VERSION = "1.0.0"
SIGNED_METHODS = {"neural_net": ("integrated_gradients", "grad_x_input"),
                  "xgboost": ("tree_contribs",), "lightgbm": ("tree_contribs",),
                  "catboost": ("tree_shap",)}
COMPLETENESS_ATOL_TREE = 1e-4   # declared before measurement; target units


def _names_sha(names):
    import hashlib
    return hashlib.sha256(",".join(names).encode("utf-8")).hexdigest()


class AttributionInputError(ValueError):
    pass


def _check_names(feature_names, x, expected_count=None, model_names=None):
    names = list(feature_names)
    if len(set(names)) != len(names):
        raise AttributionInputError("duplicate feature names")
    if x.ndim != 1 or x.shape[0] != len(names):
        raise AttributionInputError(f"x has shape {x.shape}, {len(names)} names")
    if expected_count is not None and expected_count != len(names):
        raise AttributionInputError(f"model expects {expected_count} features, got {len(names)} names")
    status = "none"
    if model_names:
        mn = [str(n) for n in model_names]
        positional = all(n == f"Column_{i}" or n == f"f{i}" or n == str(i) for i, n in enumerate(mn))
        if positional:
            status = "positional (model trained on an unnamed matrix in sidecar order)"
            if len(mn) != len(names):
                raise AttributionInputError(f"model has {len(mn)} columns, {len(names)} names")
        elif mn != names:
            raise AttributionInputError("model feature names differ from the sidecar names")
        else:
            status = "equal"
    return status


def _nn_signed(net, z, y_mean, y_std, method, ig_steps):
    import torch
    try:
        dev = next(net.parameters()).device
    except (StopIteration, AttributeError):
        dev = torch.device("cpu")
    was_training = net.training
    net.eval()
    ys = 1.0 if y_std is None else float(y_std)
    ym = 0.0 if y_mean is None else float(y_mean)
    zt = torch.tensor(z, dtype=torch.float32, device=dev).unsqueeze(0)
    with torch.no_grad():
        f_z = float(net(zt).reshape(-1)[0])
        z0 = torch.zeros_like(zt)
        f_0 = float(net(z0).reshape(-1)[0])
    if method == "grad_x_input":
        x = zt.clone().requires_grad_(True)
        net.zero_grad(set_to_none=True)
        net(x).reshape(-1)[0].backward()
        contrib = (x.grad * x).reshape(-1).detach().cpu().numpy() * ys
    else:  # integrated gradients, midpoint rule, reference z0 = 0
        alphas = (torch.arange(ig_steps, dtype=torch.float32, device=dev) + 0.5) / ig_steps
        path = z0 + alphas.reshape(-1, 1) * (zt - z0)
        path.requires_grad_(True)
        net.zero_grad(set_to_none=True)
        net(path).sum().backward()
        avg = path.grad.mean(dim=0)
        contrib = ((zt - z0).reshape(-1) * avg).detach().cpu().numpy() * ys
    if was_training:
        net.train()
    return contrib.astype(np.float64), f_0 * ys + ym, f_z * ys + ym


def per_survivor_attribution_signed(model, model_type: str, x_model_input: np.ndarray,
                                    feature_names: List[str], method: Optional[str] = None,
                                    expected_feature_count: Optional[int] = None,
                                    y_mean=None, y_std=None, ig_steps: int = 64) -> Dict:
    """Signed per-survivor attribution in target units (see block comment).

    x_model_input: ONE row exactly as the model receives it (for the NN: after
    the training scaler; build it with InferenceBundle.vector + transform).
    Raises AttributionInputError on a name or count problem (fail closed),
    unlike the best-effort absolute functions above.
    """
    if model_type not in SIGNED_METHODS:
        raise AttributionInputError(f"unknown model_type {model_type}")
    method = method or SIGNED_METHODS[model_type][0]
    if method not in SIGNED_METHODS[model_type]:
        raise AttributionInputError(f"method {method} not available for {model_type}")
    x = np.asarray(x_model_input, dtype=np.float32).reshape(-1)
    names = list(feature_names)
    exact = model_type != "neural_net"
    if model_type == "neural_net":
        status = _check_names(names, x, expected_feature_count)
        contrib, baseline, pred = _nn_signed(model, x, y_mean, y_std, method, ig_steps)
    elif model_type == "xgboost":
        import xgboost as xgb
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        status = _check_names(names, x, booster.num_features(), booster.feature_names)
        dm = xgb.DMatrix(x.reshape(1, -1))   # unnamed, as XGBoostWrapper.predict builds it
        c = booster.predict(dm, pred_contribs=True)[0].astype(np.float64)
        contrib, baseline = c[:-1], float(c[-1])
        pred = float(booster.predict(dm)[0])
    elif model_type == "lightgbm":
        status = _check_names(names, x, model.num_feature(), model.feature_name())
        c = np.asarray(model.predict(x.reshape(1, -1), pred_contrib=True))[0].astype(np.float64)
        contrib, baseline = c[:-1], float(c[-1])
        pred = float(np.asarray(model.predict(x.reshape(1, -1))).reshape(-1)[0])
    else:  # catboost
        from catboost import Pool
        _cn = list(getattr(model, "feature_names_", None) or [])
        status = _check_names(names, x, len(_cn) if _cn else None, _cn)
        sv = np.asarray(model.get_feature_importance(Pool(x.reshape(1, -1)), type="ShapValues"))[0]
        contrib, baseline = sv[:-1].astype(np.float64), float(sv[-1])
        pred = float(np.asarray(model.predict(x.reshape(1, -1))).reshape(-1)[0])
    if contrib.shape[0] != len(names):
        raise AttributionInputError(f"{contrib.shape[0]} contributions for {len(names)} names")
    residual = pred - (baseline + float(contrib.sum()))
    tot = float(np.abs(contrib).sum())
    signed = {n: float(contrib[i]) for i, n in enumerate(names)}
    return {
        "version": SIGNED_VERSION, "model_type": model_type, "method": method,
        "ig_steps": ig_steps if method == "integrated_gradients" else None,
        "reference": ("model-input zero (training mean when scaled)" if model_type == "neural_net"
                      else "tree expected value"),
        "n_features": len(names), "feature_names_sha256": _names_sha(names),
        "model_feature_names": status,
        "signed": signed,
        "abs_share": {n: (abs(v) / tot if tot > 0 else 0.0) for n, v in signed.items()},
        "baseline": baseline, "prediction": pred,
        "completeness_residual": residual,
        "completeness_exact_method": exact,
        "completeness_ok": (abs(residual) <= COMPLETENESS_ATOL_TREE) if exact else None,
    }


def attribute_row(bundle, features: Dict[str, float], method: Optional[str] = None,
                  ig_steps: int = 64, missing: str = "refuse") -> Dict:
    """[LL-7] One survivor's signed attribution from a name-keyed feature map,
    through a models.inference_bundle.InferenceBundle (LL-6)."""
    z = bundle.transform(bundle.vector(features, missing=missing).reshape(1, -1))[0]
    out = per_survivor_attribution_signed(
        bundle.attribution_model, bundle.model_type, z, bundle.feature_names, method=method,
        expected_feature_count=bundle.feature_count, y_mean=bundle.y_mean, y_std=bundle.y_std,
        ig_steps=ig_steps)
    out["checkpoint_hash"] = bundle.checkpoint_hash
    out["feature_schema_hash"] = bundle.feature_schema_hash
    return out


def per_survivor_attribution_signed_matrix(model, model_type: str, Z: np.ndarray,
                                           feature_names: List[str], method: Optional[str] = None,
                                           expected_feature_count: Optional[int] = None,
                                           y_mean=None, y_std=None, ig_steps: int = 64,
                                           chunk: int = 256) -> Dict:
    """[LL-7] Batched form of per_survivor_attribution_signed for many survivors.
    Z: (n, F) model inputs. Returns arrays: signed (n, F), baseline (n,),
    prediction (n,), completeness_residual (n,). The test proves it equals the
    per-row function."""
    if model_type not in SIGNED_METHODS:
        raise AttributionInputError(f"unknown model_type {model_type}")
    method = method or SIGNED_METHODS[model_type][0]
    if method not in SIGNED_METHODS[model_type]:
        raise AttributionInputError(f"method {method} not available for {model_type}")
    Z = np.asarray(Z, dtype=np.float32)
    names = list(feature_names)
    if Z.ndim != 2:
        raise AttributionInputError(f"Z must be 2-D, got {Z.shape}")
    if model_type == "neural_net":
        status = _check_names(names, Z[0], expected_feature_count)
        import torch
        net = model
        try:
            dev = next(net.parameters()).device
        except (StopIteration, AttributeError):
            dev = torch.device("cpu")
        was_training = net.training
        net.eval()
        ys = 1.0 if y_std is None else float(y_std)
        ym = 0.0 if y_mean is None else float(y_mean)
        S = np.empty(Z.shape, dtype=np.float64)
        P = np.empty(Z.shape[0]); B0 = np.empty(Z.shape[0])
        with torch.no_grad():
            f0 = float(net(torch.zeros((1, Z.shape[1]), device=dev)).reshape(-1)[0])
        step = max(1, chunk // max(1, ig_steps)) if method == "integrated_gradients" else chunk
        for i in range(0, Z.shape[0], step):
            zt = torch.tensor(Z[i:i + step], device=dev)
            with torch.no_grad():
                P[i:i + step] = net(zt).reshape(-1).cpu().numpy() * ys + ym
            B0[i:i + step] = f0 * ys + ym
            if method == "grad_x_input":
                x = zt.clone().requires_grad_(True)
                net.zero_grad(set_to_none=True)
                net(x).sum().backward()
                S[i:i + step] = (x.grad * x).detach().cpu().numpy() * ys
            else:
                m = zt.shape[0]
                alphas = (torch.arange(ig_steps, dtype=torch.float32, device=dev) + 0.5) / ig_steps
                path = (alphas.reshape(1, -1, 1) * zt.reshape(m, 1, -1)).reshape(m * ig_steps, -1)
                path.requires_grad_(True)
                net.zero_grad(set_to_none=True)
                net(path).sum().backward()
                avg = path.grad.reshape(m, ig_steps, -1).mean(dim=1)
                S[i:i + step] = (zt * avg).detach().cpu().numpy() * ys
        if was_training:
            net.train()
    elif model_type == "xgboost":
        import xgboost as xgb
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        status = _check_names(names, Z[0], booster.num_features(), booster.feature_names)
        dm = xgb.DMatrix(Z)
        c = booster.predict(dm, pred_contribs=True).astype(np.float64)
        S, B0 = c[:, :-1], c[:, -1]
        P = booster.predict(dm).astype(np.float64)
    elif model_type == "lightgbm":
        status = _check_names(names, Z[0], model.num_feature(), model.feature_name())
        c = np.asarray(model.predict(Z, pred_contrib=True)).astype(np.float64)
        S, B0 = c[:, :-1], c[:, -1]
        P = np.asarray(model.predict(Z)).astype(np.float64).reshape(-1)
    else:
        from catboost import Pool
        _cn = list(getattr(model, "feature_names_", None) or [])
        status = _check_names(names, Z[0], len(_cn) if _cn else None, _cn)
        sv = np.asarray(model.get_feature_importance(Pool(Z), type="ShapValues")).astype(np.float64)
        S, B0 = sv[:, :-1], sv[:, -1]
        P = np.asarray(model.predict(Z)).astype(np.float64).reshape(-1)
    if S.shape[1] != len(names):
        raise AttributionInputError(f"{S.shape[1]} contribution columns for {len(names)} names")
    R = P - (B0 + S.sum(axis=1))
    return {"version": SIGNED_VERSION, "model_type": model_type, "method": method,
            "ig_steps": ig_steps if method == "integrated_gradients" else None,
            "feature_names": names, "feature_names_sha256": _names_sha(names),
            "model_feature_names": status, "signed": S, "baseline": B0, "prediction": P,
            "completeness_residual": R, "completeness_exact_method": model_type != "neural_net"}


# ===========================================================================
# Pool Tier Comparison
# ===========================================================================

def compare_pool_tiers(
    model,
    model_type: str,
    survivors_with_scores: List[Dict],
    feature_names: List[str],
    tiers: Optional[Dict[str, int]] = None,
    max_samples_per_tier: Optional[int] = None,
    model_input=None,
) -> Dict:
    """
    Compare what drives predictions for tight vs wide pool.
    This is the data the Strategy Advisor uses for POOL_PRECISION focus.

    Computes average attribution for each tier (top_20, top_100, top_300),
    then calculates divergence — which features concentrate in top tier
    vs spread across the wider pool.

    Args:
        model: Trained model
        model_type: str
        survivors_with_scores: List of dicts with 'prediction' and 'features' keys
        feature_names: List[str]
        tiers: Optional custom tier sizes. Default: {top_20: 20, top_100: 100, top_300: 300}
        max_samples_per_tier: Optional cap on samples per tier (TB Finding #3).
            Use to limit computational cost for wide tiers. E.g., max_samples_per_tier=50
            computes attribution for at most 50 survivors even in top_300 tier.
        model_input: [LL-7, Beta X-LL6-BUNDLE R1 Q3] Optional callable(features) -> one
            model-input row. None (default) = unchanged: np.array(s["features"]) as given.
            Chapter 13 under --c13-model-loader canonical passes its _c13_model_input
            (bundle.vector by name, then the training scaler), so the tiers are attributed
            on the same input as the root-cause attribution path.

    Returns:
        dict with keys:
            'top_20': {feature: avg_attribution, ...}
            'top_100': {feature: avg_attribution, ...}
            'top_300': {feature: avg_attribution, ...}
            'divergence': {feature: top_20_attr - top_300_attr, ...}
                Positive = feature concentrates in top tier
                Negative = feature more important in wide tier
            'metadata': {tier_sizes, model_type, timestamp, attribution_failures}
    """
    from datetime import datetime

    if tiers is None:
        tiers = {"top_20": 20, "top_100": 100, "top_300": 300}

    # Sort by prediction score (highest first)
    ranked = sorted(
        survivors_with_scores,
        key=lambda s: s.get("prediction", 0),
        reverse=True,
    )

    # TB Finding #5: Track attribution failures for debugging visibility
    _total_attempts = 0
    _total_failures = 0

    def avg_attribution(pool: List[Dict]) -> Dict[str, float]:
        """Compute mean attribution across a pool of survivors."""
        nonlocal _total_attempts, _total_failures

        # TB Finding #3: Cap samples to limit computational cost
        effective_pool = pool
        if max_samples_per_tier and len(pool) > max_samples_per_tier:
            effective_pool = pool[:max_samples_per_tier]

        attrs = []
        for s in effective_pool:
            _total_attempts += 1
            if model_input is None:
                features_arr = np.array(s["features"], dtype=np.float32)
            else:  # [LL-7] Q3: the same model input as the attribution path
                features_arr = model_input(s["features"])
            attr = per_survivor_attribution(
                model, model_type, features_arr, feature_names
            )
            if attr:  # Skip failed attributions
                attrs.append(attr)
            else:
                _total_failures += 1

        if not attrs:
            return {f: 0.0 for f in feature_names}

        return {
            f: float(np.mean([a.get(f, 0.0) for a in attrs]))
            for f in feature_names
        }

    # Compute per-tier attribution
    tier_comparison = {}
    for tier_name, tier_size in tiers.items():
        pool = ranked[:tier_size]
        if len(pool) < tier_size:
            logger.warning(
                f"Tier '{tier_name}' requested {tier_size} survivors "
                f"but only {len(pool)} available"
            )
        tier_comparison[tier_name] = avg_attribution(pool)
        logger.info(f"Computed attribution for {tier_name} ({len(pool)} survivors)")

    # Compute divergence: top tier vs wide tier
    # Use smallest and largest tier for divergence
    tier_names_sorted = sorted(tiers.keys(), key=lambda k: tiers[k])
    tight_tier = tier_names_sorted[0]   # smallest (e.g., top_20)
    wide_tier = tier_names_sorted[-1]   # largest (e.g., top_300)

    divergence = {}
    for f in feature_names:
        tight_val = tier_comparison.get(tight_tier, {}).get(f, 0.0)
        wide_val = tier_comparison.get(wide_tier, {}).get(f, 0.0)
        divergence[f] = tight_val - wide_val

    tier_comparison["divergence"] = divergence
    tier_comparison["metadata"] = {
        "model_type": model_type,
        "tier_sizes": {k: min(v, len(ranked)) for k, v in tiers.items()},
        "total_survivors": len(ranked),
        "attribution_attempts": _total_attempts,
        "attribution_failures": _total_failures,
        "max_samples_per_tier": max_samples_per_tier,
        "generated_at": datetime.now().isoformat(),
        "version": __version__,
    }

    return tier_comparison


# ===========================================================================
# Utility: Save tier comparison to JSON
# ===========================================================================

def save_tier_comparison(tier_comparison: Dict, path: str = "diagnostics_outputs/tier_comparison.json"):
    """Save tier comparison result to JSON file."""
    import json
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w") as f:
        json.dump(tier_comparison, f, indent=2, default=str)

    logger.info(f"Tier comparison saved to {path}")


# ===========================================================================
# Self-test
# ===========================================================================

def _self_test():
    """Quick self-test — verify all backends handle edge cases."""
    print("=== per_survivor_attribution self-test ===")

    feature_names = [f"feature_{i}" for i in range(10)]
    fake_features = np.random.randn(10).astype(np.float32)

    # Test unified interface with unknown model type
    result = per_survivor_attribution(None, "unknown_model", fake_features, feature_names)
    assert result == {}, f"Expected empty dict for unknown model, got {result}"
    print("  [1] Unknown model type returns empty dict: PASS")

    # Test XGBoost backend with None model (should fail gracefully)
    result = per_survivor_attribution_xgb(None, fake_features, feature_names)
    assert result == {}, f"Expected empty dict for None model, got {result}"
    print("  [2] XGBoost None model returns empty dict: PASS")

    # Test LightGBM backend with None model
    result = per_survivor_attribution_lgb(None, fake_features, feature_names)
    assert result == {}, f"Expected empty dict for None model, got {result}"
    print("  [3] LightGBM None model returns empty dict: PASS")

    # Test CatBoost backend with None model
    result = per_survivor_attribution_catboost(None, fake_features, feature_names)
    assert result == {}, f"Expected empty dict for None model, got {result}"
    print("  [4] CatBoost None model returns empty dict: PASS")

    # Test compare_pool_tiers with empty survivors
    result = compare_pool_tiers(None, "xgboost", [], feature_names)
    assert "divergence" in result, "Expected divergence key"
    assert "metadata" in result, "Expected metadata key"
    print("  [5] compare_pool_tiers with empty survivors: PASS")

    print("\n  All 5 self-tests PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()
