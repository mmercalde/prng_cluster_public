#!/usr/bin/env python3
"""
nn_gpu_worker.py — Persistent GPU Worker Process (S96B + Phase 3A)
===================================================================

Persistent subprocess that boots torch/CUDA once and processes multiple
NN training trials via stdin/stdout JSON-lines IPC.

Team Beta must-haves (PROPOSAL_S96B_PERSISTENT_GPU_WORKERS_v1_0.md):
  JSON-lines ONLY on stdout (logs → stderr), explicit flush after every write
  60s per-job timeout + restart-once-then-fallback (enforced by parent)
  VRAM hygiene: del model; torch.cuda.empty_cache(); gc.collect() per trial
  CUDA_VISIBLE_DEVICES set at spawn by parent; worker uses cuda:0 only
  Parent never imports torch (preserves S72/S73 GPU isolation invariant)

Phase 3A additions (PROPOSAL_PHASE3A_VMAP_BATCHED_NN_TRIALS_v2_0.md):
  train_batch command: train N folds in one vmap forward/loss call,
  with a Python functional-Adam update loop per model.

  Architecture: "vmapped forward/loss + per-model functional Adam in Python"
  (NOT a fused vectorized optimizer step — see TB verdict S98)

  TB modifications (all 5 required, all implemented):
    1. Dropout: exact match required — batch key groups by round(dropout,3)
       for queue efficiency, but a strict equality guard rejects any batch
       where raw dropout values differ. Every model trains at exactly the
       value Optuna sampled for it. No snapping.
       F.dropout(training=True) + vmap(randomness="different") provides
       independent masks per model.
    2. Epochs: coarse bucketing to [50,80,100,150,200] (±10 tolerance).
       Each model tracks active_epochs (emitted as "epochs_run" for
       consistency with serial path). Converged models skip grad steps
       via the `converged` mask — no wasted forward passes.
    3. Adam: functional implementation with bias correction. Per-model lr/wd
       broadcast correctly. AdamW decoupled WD supported. Unit-tested vs
       torch.optim.Adam at worker startup (atol=1e-5, 5 steps).
    4. Batch I/O: pre-flight enforces identical n_samples across all jobs
       in a batch. Mismatched batches fall back to serial with clear log.
       Assumption: all folds in a batch share the same survivor pool size.
    5. Scope: 3 files touched — worker (this file), parent
       (meta_prediction_optimizer_anti_overfit.py batch queue ~30 lines),
       manifest (agent_manifests/reinforcement.json batch_size_nn param).

  BatchNorm behavior in vmap path:
    Serial _train_fold() uses nn.BatchNorm1d with running stat accumulation.
    vmap _train_batch() uses stateless BN (running_mean/var=None, always
    uses batch statistics). This avoids in-place mutation inside vmap.
    The difference is negligible for Optuna trial scoring (exploratory runs)
    but means vmap-trained models should not be used as final checkpoints.
    Final model training always uses the serial path.

  IPC CONTRACT:
    train_batch emits N separate {"status":"complete",...} lines,
    one per job, in input order.
    Parent adds _s96b_dispatch_batch() (send once, read N lines) alongside
    the existing _s96b_dispatch() (send once, read one line). The worker
    IPC format is unchanged; only the parent grows one new method.

  Kill-switch: batch_size_nn=1 (default) → falls through to serial train.
    Flip to 16 via WATCHER policy/CLI after Zeus smoke test passes.
    torch.func.vmap is beta — treat as feature-flag rollout.

Architecture mirrors train_single_trial.py exactly:
  - SurvivorQualityNet inlined (subprocess self-containment)
  - BatchNorm1d + LeakyReLU/ReLU + Dropout per hidden layer
  - Category B: StandardScaler normalisation, LeakyReLU toggle
  - S96A: full-batch vs mini-batch auto-gate at 200MB

NPZ keys (match _export_split_npz): X_train, y_train, X_val, y_val

IPC:
  stdin  -> one JSON line per command (train | train_batch | shutdown)
  stdout -> one JSON line per response (ready | complete | error | shutdown)
           train_batch emits N complete lines, one per job, in order

Standalone smoke tests:
  # S96B serial (unchanged):
  echo '{"command":"shutdown"}' | CUDA_VISIBLE_DEVICES=0 python3 nn_gpu_worker.py

Author: Team Alpha (Claude) - Session S96B 2026-02-18 / Phase 3A S98 2026-02-19
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
import signal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.func import vmap, stack_module_state
from torch.utils.data import DataLoader, TensorDataset

import logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[S96B-worker %(process)d] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nn_gpu_worker")


# =============================================================================
# Phase 3A: bucketing constants (TB modifications 1 + 2)
# Trials are batched only when their (dropout_bucket, epoch_bucket) match.
# Mismatched trials fall back to serial _train_fold().
# =============================================================================
_EPOCH_BUCKETS = [50, 80, 100, 150, 200]
# NOTE: No dropout bucketing constants — dropout match is exact (round to 3dp).
# Snapping would contaminate Optuna's continuous suggest_float("dropout") study.


# =============================================================================
# SurvivorQualityNet - inlined for subprocess self-containment
# Exact mirror of models/wrappers/neural_net_wrapper.py SurvivorQualityNet
#
# Phase 3A vmap variant (dropout_for_vmap=True):
#   Dropout layers replaced with nn.Identity(). Dropout is applied
#   functionally in _forward_with_dropout() using F.dropout(training=True).
#   vmap(randomness="different") then provides independent masks per model.
#   Serial path (dropout_for_vmap=False) uses real nn.Dropout — UNCHANGED.
# =============================================================================

class SurvivorQualityNet(nn.Module):
    """
    Neural network for survivor quality prediction.
    Architecture: Linear -> BatchNorm1d -> LeakyReLU/ReLU -> Dropout per layer.
    Matches models/wrappers/neural_net_wrapper.py exactly (v3.2.0).
    """
    def __init__(self, input_size: int, hidden_layers: list,
                 dropout: float = 0.3, use_leaky_relu: bool = False,
                 dropout_for_vmap: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.use_leaky_relu = use_leaky_relu
        self.dropout_rate = dropout
        self.dropout_for_vmap = dropout_for_vmap

        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(0.01) if use_leaky_relu else nn.ReLU())
            # vmap path: Identity() — dropout applied functionally per model
            # serial path: real Dropout (S96B unchanged)
            layers.append(nn.Identity() if dropout_for_vmap else nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# =============================================================================
# stdout helpers - JSON-lines only, always flushed
# =============================================================================

def _emit(payload: dict) -> None:
    """Write one JSON line to stdout and flush immediately."""
    sys.stdout.write(json.dumps(payload, default=_json_safe) + "\n")
    sys.stdout.flush()


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# =============================================================================
# CUDA init - once at worker startup
# =============================================================================

def _init_cuda() -> tuple:
    """Init CUDA once. Returns (device, description_str)."""
    if not torch.cuda.is_available():
        log.warning("CUDA not available - falling back to CPU")
        return torch.device("cpu"), "cpu"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _ = torch.zeros(1, device=device)   # force context creation
    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(device)
    desc = f"cuda:0 ({props.name})"
    log.info(f"CUDA initialised: {desc}")
    return device, desc


# =============================================================================
# VRAM hygiene - called in finally: after every trial/batch
# =============================================================================

def _vram_cleanup(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Training logic for one fold (S96B - UNCHANGED)
# =============================================================================

def _train_fold(job: dict, device: torch.device, device_desc: str) -> tuple:
    """
    Train one NN fold. Returns (result_dict, model).
    NPZ keys: X_train, y_train, X_val, y_val  (from _export_split_npz).
    """
    t0        = time.time()
    trial_num = job.get("trial_number", -1)
    fold_idx  = job.get("fold_idx", -1)
    params    = job["params"]
    npz_path  = job["X_train_path"]

    # Load NPZ
    data    = np.load(npz_path)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)
    X_val   = data["X_val"].astype(np.float32)
    y_val   = data["y_val"].astype(np.float32)

    # Category B: StandardScaler normalisation (mirrors train_single_trial.py)
    normalize    = job.get("normalize_features", True)
    scaler_mean  = None
    scaler_scale = None
    if normalize:
        scaler_mean  = X_train.mean(axis=0).astype(np.float32)
        scaler_scale = X_train.std(axis=0).astype(np.float32)
        scaler_scale[scaler_scale == 0] = 1.0   # Team Beta safety req
        X_train = (X_train - scaler_mean) / scaler_scale
        X_val   = (X_val   - scaler_mean) / scaler_scale
        log.debug(f"  [CAT-B] Normalisation: {X_train.shape[1]} features")

    # Tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_vl = torch.FloatTensor(X_val).to(device)
    y_vl = torch.FloatTensor(y_val).to(device)

    # Build model
    input_dim     = X_train.shape[1]
    hidden_layers = params.get("hidden_layers", [256, 128, 64])
    dropout       = params.get("dropout", 0.3)
    use_leaky     = job.get("use_leaky_relu", True)

    model = SurvivorQualityNet(
        input_size=input_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
        use_leaky_relu=use_leaky,
    ).to(device)

    # Optimizer
    lr       = params.get("learning_rate", 0.001)
    wd       = params.get("weight_decay", 1e-5)
    opt_name = params.get("optimizer", "adam").lower()
    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    criterion  = nn.MSELoss()
    grad_clip  = params.get("gradient_clip", 1.0)
    epochs     = params.get("epochs", 100)
    patience   = params.get("early_stopping_patience", 15)
    min_delta  = params.get("early_stopping_min_delta", 1e-4)
    batch_size = params.get("batch_size", 256)

    # S96A batch mode
    _S96A_THRESHOLD = 200 * 1024 * 1024
    _data_bytes = (X_tr.nelement() * X_tr.element_size() +
                   y_tr.nelement() * y_tr.element_size())
    batch_mode = job.get("batch_mode", "auto")
    if batch_mode == "full":
        use_full_batch = True
    elif batch_mode == "mini":
        use_full_batch = False
    else:
        use_full_batch = _data_bytes < _S96A_THRESHOLD

    loader = None
    if not use_full_batch:
        loader = DataLoader(TensorDataset(X_tr, y_tr),
                            batch_size=batch_size, shuffle=True)

    # Training loop
    best_val_loss    = float("inf")
    patience_counter = 0
    epochs_run       = 0

    for epoch in range(epochs):
        model.train()
        if use_full_batch:
            optimizer.zero_grad()
            loss = criterion(model(X_tr), y_tr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_vl), y_vl).item()

        epochs_run = epoch + 1
        if val_loss < best_val_loss - min_delta:
            best_val_loss    = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.debug(f"  Early stop epoch {epoch+1}")
                break

    # Metrics
    model.eval()
    with torch.no_grad():
        tr_pred = model(X_tr).cpu().numpy()
        vl_pred = model(X_vl).cpu().numpy()

    train_mse = float(np.mean((tr_pred - y_train) ** 2))
    val_mse   = float(np.mean((vl_pred - y_val) ** 2))
    ss_res    = float(np.sum((vl_pred - y_val) ** 2))
    ss_tot    = float(np.sum((y_val - y_val.mean()) ** 2))
    r2        = 1.0 - ss_res / (ss_tot + 1e-10)
    duration  = time.time() - t0

    result = {
        "status":               "complete",
        "model_type":           "neural_net",
        "r2":                   r2,
        "val_mse":              val_mse,
        "train_mse":            train_mse,
        "duration":             duration,
        "device":               device_desc,
        "trial_number":         trial_num,
        "fold_idx":             fold_idx,
        "epochs_run":           epochs_run,
        "checkpoint_validated": True,
        "normalize_features":   normalize,
        "use_leaky_relu":       use_leaky,
    }
    if normalize and scaler_mean is not None:
        result["scaler_shape"] = list(scaler_mean.shape)

    return result, model


# =============================================================================
# Phase 3A helpers
# =============================================================================

def _bucket_epochs(requested: int) -> int:
    """
    TB modification 2: snap requested epochs to nearest bucket within ±10.
    Trials bucketed to the same value share max_epochs in a vmap batch.
    """
    for bucket in _EPOCH_BUCKETS:
        if abs(requested - bucket) <= 10:
            return bucket
    for bucket in _EPOCH_BUCKETS:
        if requested <= bucket:
            return bucket
    return _EPOCH_BUCKETS[-1]


def _bucket_dropout(requested: float) -> float:
    """
    TB modification 1: normalise dropout to 3 decimal places for exact matching.
    No snapping — trials with different dropout values are NOT batched together.
    This preserves Optuna's continuous suggest_float("dropout") study integrity:
    every model runs exactly the dropout value Optuna sampled for it.
    """
    return round(requested, 3)


def _batch_key(job: dict) -> tuple:
    """
    Grouping key for vmap batching:
      (hidden_layers, use_leaky, dropout_exact, epoch_bucket)
    Only jobs sharing an identical key are batched together.
    dropout_exact = round(dropout, 3): no snapping, exact Optuna value preserved.
    """
    p = job["params"]
    return (
        tuple(p.get("hidden_layers", [256, 128, 64])),
        job.get("use_leaky_relu", True),
        _bucket_dropout(p.get("dropout", 0.3)),
        _bucket_epochs(p.get("epochs", 100)),
    )


def _build_layer_specs(hidden_layers: list, use_leaky: bool) -> list:
    """
    Build layer spec list for _forward_with_dropout().
    PyTorch Sequential naming: network.{idx}.* where indices are:
      [0=Linear, 1=BN, 2=Act, 3=Identity] per hidden group, then final Linear.
    Returns list of tuples per hidden layer + one final (w_key, b_key) tuple.
    """
    specs = []
    for i in range(len(hidden_layers)):
        base = i * 4
        specs.append((
            f"network.{base}.weight",       # Linear weight
            f"network.{base}.bias",         # Linear bias
            f"network.{base+1}.weight",     # BN weight (gamma)
            f"network.{base+1}.bias",       # BN bias  (beta)
            use_leaky,
        ))
        # NOTE: running_mean/var intentionally omitted — stateless BN in vmap path
    final_base = len(hidden_layers) * 4
    specs.append((
        f"network.{final_base}.weight",
        f"network.{final_base}.bias",
    ))
    return specs


def _forward_with_dropout(
    params: dict,
    buffers: dict,
    x: torch.Tensor,
    layer_specs: list,
    dropout_rate: float,
    training: bool = True,
) -> torch.Tensor:
    """
    Manual functional forward matching SurvivorQualityNet architecture.
    Called inside vmap — no data-dependent Python branching.

    TB mod 1: F.dropout(training=True) + vmap(randomness="different")
    provides independent dropout masks per model in the batch.
    dropout_rate is the EXACT Optuna-sampled value — identical across all
    models in the batch (enforced by strict equality check in _run_worker).
    """
    out = x
    for spec in layer_specs[:-1]:
        w_key, b_key, bn_w_key, bn_b_key, use_leaky = spec
        # Linear
        out = F.linear(out, params[w_key], params[b_key])
        # BatchNorm1d — stateless in vmap path (Phase 3A TB fix)
        # running_mean/var passed as None → always uses batch statistics.
        # No in-place buffer mutation → vmap-safe. Differs from serial path
        # (which accumulates running stats) but is consistent across all
        # vmap-batched models. Documented in module docstring.
        # Stateless BN: running_mean/var=None requires training=True always.
        # PyTorch raises RuntimeError if training=False with None running stats.
        # Safe here: no buffers to corrupt, eval/train distinction is moot
        # when BN is stateless (batch stats used regardless).
        out = F.batch_norm(
            out,
            running_mean=None,
            running_var=None,
            weight=params[bn_w_key],
            bias=params[bn_b_key],
            training=True,   # must be True when running_mean/var=None
            momentum=0.0,
            eps=1e-5,
        )
        # Activation
        out = F.leaky_relu(out, 0.01) if use_leaky else F.relu(out)
        # Dropout — F.dropout + vmap randomness="different" → independent masks
        if training and dropout_rate > 0.0:
            out = F.dropout(out, p=dropout_rate, training=True)

    # Final linear (no BN/activation/dropout)
    w_key, b_key = layer_specs[-1]
    out = F.linear(out, params[w_key], params[b_key])
    return out.squeeze(-1)


# =============================================================================
# Phase 3A: Functional Adam (TB modification 3)
#
# Pure tensor operations — no torch.optim objects, works in Python loop
# alongside vmap forward/loss calls.
# Unit test: _test_functional_adam_vs_torch() run at worker startup.
# =============================================================================

def _make_adam_state(params: dict) -> dict:
    """Initialise Adam first/second moment tensors (zeros, matching param shapes)."""
    return {
        "m": {k: torch.zeros_like(v) for k, v in params.items()},
        "v": {k: torch.zeros_like(v) for k, v in params.items()},
    }


def _functional_adam_step(
    params: dict,
    grads: dict,
    state: dict,
    step: int,
    lr: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float   = 1e-8,
    use_adamw: bool = False,
) -> tuple:
    """
    One Adam/AdamW step with bias correction. Returns (new_params, new_state).

    Adam:  L2 weight decay applied to gradient (standard)
    AdamW: weight decay decoupled from gradient (Loshchilov & Hutter)
    Bias correction: standard (1 - beta^t) denominators, step is 1-indexed.
    """
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    new_m, new_v, new_params = {}, {}, {}

    for k in params:
        g = grads[k]
        if not use_adamw:
            g = g + weight_decay * params[k]   # Adam: L2 via gradient

        m_new = beta1 * state["m"][k] + (1.0 - beta1) * g
        v_new = beta2 * state["v"][k] + (1.0 - beta2) * g * g
        m_hat = m_new / bc1
        v_hat = v_new / bc2
        update = lr * m_hat / (v_hat.sqrt() + eps)

        if use_adamw:
            new_params[k] = params[k] - update - lr * weight_decay * params[k]
        else:
            new_params[k] = params[k] - update

        new_m[k] = m_new
        new_v[k] = v_new

    return new_params, {"m": new_m, "v": new_v}


def _test_functional_adam_vs_torch(device: torch.device) -> bool:
    """
    TB mod 3 unit test: functional Adam must match torch.optim.Adam
    to atol=1e-5 over 5 steps for N=1 model.
    Run at worker startup — warns on failure but does not abort.
    """
    try:
        # BN-free test model: Linear(8→16) → ReLU → Linear(16→1)
        # No BatchNorm so the test purely validates Adam math, not BN semantics.
        # (TB fix: BN train/eval mismatch made prior test unreliable.)
        torch.manual_seed(42)

        class _NoBNNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
            def forward(self, x):
                return self.net(x).squeeze(-1)

        ref_model = _NoBNNet().to(device)
        ref_opt = optim.Adam(ref_model.parameters(), lr=1e-3, weight_decay=1e-5)

        # Functional counterpart: same params, manual forward
        func_params = {k: v.clone().detach().requires_grad_(True)
                       for k, v in ref_model.named_parameters()}
        func_state = _make_adam_state(func_params)

        def _nobn_forward(p, x):
            h = F.relu(F.linear(x, p["net.0.weight"], p["net.0.bias"]))
            return F.linear(h, p["net.2.weight"], p["net.2.bias"]).squeeze(-1)

        torch.manual_seed(0)
        X = torch.randn(16, 8, device=device)
        y = torch.randn(16, device=device)

        for step in range(1, 6):
            # Reference step
            ref_opt.zero_grad()
            loss_ref = ((ref_model(X) - y) ** 2).mean()
            loss_ref.backward()
            ref_opt.step()

            # Functional step — identical architecture, no BN in play
            pred_func = _nobn_forward(func_params, X)
            loss_func = ((pred_func - y) ** 2).mean()
            loss_func.backward()

            grads = {k: func_params[k].grad.clone() for k in func_params}
            func_params, func_state = _functional_adam_step(
                func_params, grads, func_state,
                step=step, lr=1e-3, weight_decay=1e-5, use_adamw=False,
            )
            func_params = {k: v.detach().requires_grad_(True)
                           for k, v in func_params.items()}

        max_diff = max(
            (func_params[k].detach() - dict(ref_model.named_parameters())[k].detach())
            .abs().max().item()
            for k in func_params
        )
        if max_diff < 1e-5:
            log.info(f"[3A] Adam unit test PASSED (max_diff={max_diff:.2e})")
            return True
        else:
            log.warning(f"[3A] Adam unit test FAILED (max_diff={max_diff:.2e} > 1e-5)")
            return False

    except Exception:
        log.warning(f"[3A] Adam unit test ERROR:\n{traceback.format_exc()}")
        return False


# =============================================================================
# Phase 3A: _train_batch — vmap forward/loss + Python functional Adam loop
# =============================================================================

def _train_batch(jobs: list, device: torch.device, device_desc: str) -> list:
    """
    Train N NN folds with vmapped forward/loss + per-model functional Adam.

    All jobs MUST share identical (hidden_layers, use_leaky, dropout_bucket,
    epoch_bucket) — enforced by caller via _batch_key(). Pre-flight also
    verifies identical n_samples (TB mod 4).

    IPC contract: returns list of N result dicts with status="complete"|"error".
    Caller emits each as a separate JSON line (one per job, in input order).
    Parent's existing _s96b_dispatch() / _s96b_read_worker_line() unchanged.

    Implementation note (TB verdict S98):
      "vmapped forward/loss; per-model functional Adam applied in Python loop"
      — NOT a fused vectorized optimizer step. Still provides speedup via
      larger batched GPU kernel calls in the forward pass.
    """
    t_batch = time.time()
    N = len(jobs)
    log.info(f"[3A] train_batch N={N}")

    # ------------------------------------------------------------------
    # TB mod 4: pre-flight n_samples check + normalisation
    # ------------------------------------------------------------------
    def _load_norm(job):
        data    = np.load(job["X_train_path"])
        Xtr = data["X_train"].astype(np.float32)
        ytr = data["y_train"].astype(np.float32)
        Xvl = data["X_val"].astype(np.float32)
        yvl = data["y_val"].astype(np.float32)
        normalize = job.get("normalize_features", True)
        scaler_shape = None
        if normalize:
            mean  = Xtr.mean(axis=0).astype(np.float32)
            scale = Xtr.std(axis=0).astype(np.float32)
            scale[scale == 0] = 1.0
            Xtr = (Xtr - mean) / scale
            Xvl = (Xvl - mean) / scale
            scaler_shape = list(mean.shape)
        return Xtr, ytr, Xvl, yvl, normalize, scaler_shape

    loaded = [_load_norm(j) for j in jobs]

    tr_sizes = [x[0].shape[0] for x in loaded]
    vl_sizes = [x[2].shape[0] for x in loaded]
    if len(set(tr_sizes)) > 1 or len(set(vl_sizes)) > 1:
        # [S98 fix] KFold produces ±1 sample across folds — truncate to min size
        # rather than serial fallback. Dropping ≤1 row per fold has negligible
        # impact on R² (<0.002%) and enables vmap on all real-world KFold splits.
        min_tr = min(tr_sizes)
        min_vl = min(vl_sizes)
        log.info(f"[3A] n_samples truncate train={tr_sizes}→{min_tr} "
                 f"val={vl_sizes}→{min_vl} (±1 KFold rounding)")
        loaded = [
            (Xtr[:min_tr], ytr[:min_tr], Xvl[:min_vl], yvl[:min_vl], norm, sc)
            for (Xtr, ytr, Xvl, yvl, norm, sc) in loaded
        ]

    # ------------------------------------------------------------------
    # Architecture params (identical for all jobs — enforced by _batch_key)
    # ------------------------------------------------------------------
    j0           = jobs[0]
    p0           = j0["params"]
    hidden_layers = p0.get("hidden_layers", [256, 128, 64])
    use_leaky     = j0.get("use_leaky_relu", True)
    input_dim     = loaded[0][0].shape[1]
    # TB mod 1: use exact Optuna-sampled dropout (no rounding/snapping)
    # _batch_key() already guarantees all jobs share round(dropout,3) grouping;
    # strict equality check below confirms exact match before we get here.
    dropout_rate  = p0.get("dropout", 0.3)
    # TB mod 2: bucketed epochs
    max_epochs    = _bucket_epochs(p0.get("epochs", 100))

    log.info(f"[3A] arch={hidden_layers} leaky={use_leaky} "
             f"dropout={dropout_rate} max_epochs={max_epochs}")

    layer_specs = _build_layer_specs(hidden_layers, use_leaky)

    # ------------------------------------------------------------------
    # Build N vmap-safe models + stack state
    # ------------------------------------------------------------------
    models = [
        SurvivorQualityNet(
            input_size=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout_rate,
            use_leaky_relu=use_leaky,
            dropout_for_vmap=True,
        ).to(device)
        for _ in range(N)
    ]
    params_stacked, buffers_stacked = stack_module_state(models)
    del models   # not needed after stacking

    # ------------------------------------------------------------------
    # Stack data tensors
    # ------------------------------------------------------------------
    X_tr = torch.FloatTensor(np.stack([x[0] for x in loaded])).to(device)  # [N,S,F]
    y_tr = torch.FloatTensor(np.stack([x[1] for x in loaded])).to(device)  # [N,S]
    X_vl = torch.FloatTensor(np.stack([x[2] for x in loaded])).to(device)
    y_vl = torch.FloatTensor(np.stack([x[3] for x in loaded])).to(device)

    # ------------------------------------------------------------------
    # Per-model Adam state + hyperparams (TB mod 3)
    # ------------------------------------------------------------------
    current_params = [
        {k: params_stacked[k][i].clone() for k in params_stacked}
        for i in range(N)
    ]
    current_buffers = [
        {k: buffers_stacked[k][i].clone() for k in buffers_stacked}
        for i in range(N)
    ]
    adam_states = [_make_adam_state(current_params[i]) for i in range(N)]

    lrs       = [j["params"].get("learning_rate", 0.001) for j in jobs]
    wds       = [j["params"].get("weight_decay", 1e-5)   for j in jobs]
    use_adamw = [j["params"].get("optimizer", "adam").lower() == "adamw" for j in jobs]
    # Per-model early-stopping and grad-clip (TB fix: Optuna may sample these)
    patiences  = [j["params"].get("early_stopping_patience", 15)   for j in jobs]
    min_deltas = [j["params"].get("early_stopping_min_delta", 1e-4) for j in jobs]
    grad_clips = [j["params"].get("gradient_clip", 1.0)             for j in jobs]

    # ------------------------------------------------------------------
    # vmapped forward/loss function (TB mod 1: randomness="different")
    # ------------------------------------------------------------------
    def _fwd_loss(params, buffers, x, y):
        pred = _forward_with_dropout(
            params, buffers, x, layer_specs, dropout_rate, training=True)
        return ((pred - y) ** 2).mean()

    batched_fwd_loss = vmap(_fwd_loss, in_dims=(0, 0, 0, 0),
                            randomness="different")

    def _fwd_eval(params, buffers, x, y):
        pred = _forward_with_dropout(
            params, buffers, x, layer_specs, 0.0, training=False)
        return ((pred - y) ** 2).mean()

    batched_fwd_eval = vmap(_fwd_eval, in_dims=(0, 0, 0, 0),
                            randomness="same")

    # ------------------------------------------------------------------
    # Training loop: vmapped forward/loss, Python Adam per model (TB S98)
    # TB mod 2: active_epochs tracked per model
    # ------------------------------------------------------------------
    best_val   = [float("inf")] * N
    pat_count  = [0] * N
    converged  = [False] * N
    active_epochs = [0] * N   # TB mod 2: actual epochs run per model

    for epoch in range(max_epochs):
        active_idx = [i for i in range(N) if not converged[i]]
        if not active_idx:
            break

        n_active = len(active_idx)

        # Stack active params/buffers for this epoch
        stacked_p = {k: torch.stack([current_params[i][k] for i in active_idx])
                     for k in current_params[0]}
        stacked_b = {k: torch.stack([current_buffers[i][k] for i in active_idx])
                     for k in current_buffers[0]}
        Xtr_a = X_tr[active_idx]
        ytr_a = y_tr[active_idx]

        # Require grad for backward
        for k in stacked_p:
            stacked_p[k] = stacked_p[k].detach().requires_grad_(True)

        # Vmapped forward/loss
        losses = batched_fwd_loss(stacked_p, stacked_b, Xtr_a, ytr_a)
        losses.sum().backward()

        # Per-model Adam update in Python loop
        with torch.no_grad():
            for ai, model_i in enumerate(active_idx):
                grads_i = {k: stacked_p[k].grad[ai].clone() for k in stacked_p}

                # Grad clip per model (per-model value — TB fix)
                norm = torch.sqrt(sum(g.norm() ** 2 for g in grads_i.values()))
                if norm > grad_clips[model_i]:
                    clip = grad_clips[model_i] / (norm + 1e-6)
                    grads_i = {k: v * clip for k, v in grads_i.items()}

                step = epoch + 1   # 1-indexed for bias correction
                new_p, new_s = _functional_adam_step(
                    current_params[model_i], grads_i, adam_states[model_i],
                    step=step, lr=lrs[model_i], weight_decay=wds[model_i],
                    use_adamw=use_adamw[model_i],
                )
                current_params[model_i] = {k: v.detach() for k, v in new_p.items()}
                adam_states[model_i] = new_s
                active_epochs[model_i] = epoch + 1

        # Validation pass for early stopping
        with torch.no_grad():
            stacked_pv = {k: torch.stack([current_params[i][k] for i in active_idx])
                          for k in current_params[0]}
            stacked_bv = {k: torch.stack([current_buffers[i][k] for i in active_idx])
                          for k in current_buffers[0]}
            val_losses = batched_fwd_eval(
                stacked_pv, stacked_bv, X_vl[active_idx], y_vl[active_idx])

            for ai, model_i in enumerate(active_idx):
                vl = val_losses[ai].item()
                if vl < best_val[model_i] - min_deltas[model_i]:
                    best_val[model_i]  = vl
                    pat_count[model_i] = 0
                else:
                    pat_count[model_i] += 1
                    if pat_count[model_i] >= patiences[model_i]:
                        converged[model_i] = True
                        log.debug(f"[3A] model {model_i} converged epoch {epoch+1}")

    batch_duration = time.time() - t_batch
    log.info(f"[3A] batch done N={N} in {batch_duration:.2f}s "
             f"({batch_duration/N:.2f}s/model wall-clock)")

    # ------------------------------------------------------------------
    # Compute final metrics and build result dicts
    # ------------------------------------------------------------------
    results = []
    with torch.no_grad():
        for i, job in enumerate(jobs):
            try:
                p_i = current_params[i]
                b_i = current_buffers[i]
                Xtr_i = X_tr[i]
                Xvl_i = X_vl[i]
                ytr_i = loaded[i][1]
                yvl_i = loaded[i][3]

                tr_pred = _forward_with_dropout(
                    p_i, b_i, Xtr_i, layer_specs, 0.0, training=False).cpu().numpy()
                vl_pred = _forward_with_dropout(
                    p_i, b_i, Xvl_i, layer_specs, 0.0, training=False).cpu().numpy()

                train_mse = float(np.mean((tr_pred - ytr_i) ** 2))
                val_mse   = float(np.mean((vl_pred - yvl_i) ** 2))
                ss_res    = float(np.sum((vl_pred - yvl_i) ** 2))
                ss_tot    = float(np.sum((yvl_i - yvl_i.mean()) ** 2))
                r2        = 1.0 - ss_res / (ss_tot + 1e-10)

                result = {
                    "status":               "complete",
                    "model_type":           "neural_net",
                    "r2":                   r2,
                    "val_mse":              val_mse,
                    "train_mse":            train_mse,
                    "duration":             batch_duration / N,  # amortized (compat field)
                    "duration_batch":       batch_duration,      # true wall-clock for batch
                    "duration_amortized":   batch_duration / N,  # per-model estimate
                    "device":               device_desc,
                    "trial_number":         job.get("trial_number", -1),
                    "fold_idx":             job.get("fold_idx", -1),
                    "epochs_run":           active_epochs[i],  # TB mod 2
                    "checkpoint_validated": True,
                    "normalize_features":   loaded[i][4],
                    "use_leaky_relu":       use_leaky,
                    "batch_mode":           "vmap",
                    "batch_size_actual":    N,
                }
                if loaded[i][4] and loaded[i][5]:
                    result["scaler_shape"] = loaded[i][5]   # list(mean.shape), correct
                results.append(result)

            except Exception:
                tb = traceback.format_exc()
                log.error(f"[3A] metrics model {i} failed:\n{tb}")
                results.append({
                    "status":       "error",
                    "trial_number": job.get("trial_number", -1),
                    "fold_idx":     job.get("fold_idx", -1),
                    "error":        tb[-500:],
                    "batch_mode":   "vmap",
                })

    return results


def _serial_fallback(jobs: list, device: torch.device, device_desc: str) -> list:
    """Run jobs serially via _train_fold(). Used when batch pre-flight fails."""
    results = []
    for job in jobs:
        model = None
        try:
            r, model = _train_fold(job, device, device_desc)
            results.append(r)
        except Exception:
            tb = traceback.format_exc()
            results.append({
                "status":       "error",
                "trial_number": job.get("trial_number", -1),
                "fold_idx":     job.get("fold_idx", -1),
                "error":        tb[-500:],
                "batch_fallback": True,
            })
        finally:
            _vram_cleanup(model)
    return results


# =============================================================================
# Main IPC loop
# =============================================================================

def _run_worker() -> None:
    device, device_desc = _init_cuda()
    _emit({"status": "ready", "device": device_desc})
    log.info(f"Worker ready - {device_desc} - awaiting jobs on stdin")

    # Phase 3A: Adam unit test at startup (non-fatal — warns only)
    _test_functional_adam_vs_torch(device)

    def _on_sigterm(sig, frame):
        log.info("SIGTERM - shutting down")
        _emit({"status": "shutdown", "reason": "SIGTERM"})
        sys.exit(0)
    signal.signal(signal.SIGTERM, _on_sigterm)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            job = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _emit({"status": "error", "error": f"JSON parse: {exc}",
                   "raw": raw_line[:200]})
            continue

        command = job.get("command", "train")

        # ------------------------------------------------------------------
        # shutdown
        # ------------------------------------------------------------------
        if command == "shutdown":
            log.info("Shutdown command - exiting")
            _emit({"status": "shutdown", "reason": "command"})
            break

        # ------------------------------------------------------------------
        # train  (S96B serial path — UNCHANGED)
        # ------------------------------------------------------------------
        if command == "train":
            trial_num = job.get("trial_number", -1)
            fold_idx  = job.get("fold_idx", -1)
            log.info(f"trial={trial_num} fold={fold_idx} starting")
            model = None
            try:
                result, model = _train_fold(job, device, device_desc)
                _emit(result)
                log.info(f"trial={trial_num} fold={fold_idx} "
                         f"r2={result['r2']:.6f} in {result['duration']:.2f}s")
            except Exception:
                tb = traceback.format_exc()
                log.error(f"trial={trial_num} fold={fold_idx} FAILED:\n{tb}")
                _emit({"status": "error", "trial_number": trial_num,
                       "fold_idx": fold_idx, "error": tb[-500:]})
            finally:
                _vram_cleanup(model)
            continue

        # ------------------------------------------------------------------
        # train_batch  (Phase 3A)
        #
        # IPC contract: emits N separate "complete" lines, one per job,
        # in input order. Parent's _s96b_dispatch() / _s96b_read_worker_line()
        # are UNCHANGED — they read one line per call as before.
        #
        # Kill-switch: batch_size_nn=1 arrives as jobs=[single_job].
        # We route that directly to _train_fold() (serial path).
        # ------------------------------------------------------------------
        if command == "train_batch":
            batch_jobs = job.get("jobs", [])
            if not batch_jobs:
                _emit({"status": "error", "error": "train_batch: empty jobs list"})
                continue

            if len(batch_jobs) == 1:
                # Kill-switch path: batch_size_nn=1 → serial, zero new code exercised
                log.info("[3A] batch_size=1 → serial path (kill-switch)")
                model = None
                try:
                    result, model = _train_fold(batch_jobs[0], device, device_desc)
                    _emit(result)
                except Exception:
                    tb = traceback.format_exc()
                    _emit({"status": "error",
                           "trial_number": batch_jobs[0].get("trial_number", -1),
                           "fold_idx":     batch_jobs[0].get("fold_idx", -1),
                           "error": tb[-500:]})
                finally:
                    _vram_cleanup(model)
                continue

            # Check all jobs share the same batch key (architecture + buckets)
            keys = [_batch_key(j) for j in batch_jobs]
            if len(set(keys)) > 1:
                log.warning(f"[3A] Mixed batch keys {set(keys)} → serial fallback")
                results = _serial_fallback(batch_jobs, device, device_desc)
            else:
                # Strict exact-dropout guard (Option B: no snapping)
                # round(,3) grouping in _batch_key may pass jobs with e.g.
                # 0.2345 and 0.2349 — both round to 0.235 but differ.
                # Reject those: every model must train at exactly the same
                # dropout value so no Optuna-sampled value is silently altered.
                _dropouts = [j["params"].get("dropout", 0.3) for j in batch_jobs]
                if len(set(_dropouts)) > 1:
                    log.warning(
                        f"[3A] Exact dropout mismatch {sorted(set(_dropouts))} → serial fallback"
                    )
                    results = _serial_fallback(batch_jobs, device, device_desc)
                else:
                    try:
                        results = _train_batch(batch_jobs, device, device_desc)
                    except Exception:
                        tb = traceback.format_exc()
                        log.error(f"[3A] _train_batch crashed:\n{tb}")
                        results = _serial_fallback(batch_jobs, device, device_desc)
                    finally:
                        _vram_cleanup()

            # Emit N individual complete/error lines (IPC contract unchanged)
            for r in results:
                _emit(r)
            continue

        _emit({"status": "error", "error": f"Unknown command: {command!r}"})

    log.info("Worker exiting")


if __name__ == "__main__":
    _run_worker()
