#!/usr/bin/env python3
"""
nn_gpu_worker.py — Persistent GPU Worker Process (S96B)
=========================================================

Persistent subprocess that boots torch/CUDA once and processes multiple
NN training trials via stdin/stdout JSON-lines IPC.

Team Beta must-haves (PROPOSAL_S96B_PERSISTENT_GPU_WORKERS_v1_0.md):
  JSON-lines ONLY on stdout (logs → stderr), explicit flush after every write
  60s per-job timeout + restart-once-then-fallback (enforced by parent)
  VRAM hygiene: del model; torch.cuda.empty_cache(); gc.collect() per trial
  CUDA_VISIBLE_DEVICES set at spawn by parent; worker uses cuda:0 only
  Parent never imports torch (preserves S72/S73 GPU isolation invariant)

Architecture mirrors train_single_trial.py exactly:
  - Uses SurvivorQualityNet (inlined here for subprocess self-containment)
  - BatchNorm1d + LeakyReLU/ReLU + Dropout per hidden layer
  - Category B: StandardScaler normalisation, LeakyReLU toggle
  - S96A: full-batch vs mini-batch auto-gate at 200MB

NPZ keys (match _export_split_npz): X_train, y_train, X_val, y_val

IPC:
  stdin  -> one JSON line per command (train | shutdown)
  stdout -> one JSON line per response (ready | complete | error | shutdown)

Standalone smoke test:
  echo '{"command":"shutdown"}' | CUDA_VISIBLE_DEVICES=0 python3 nn_gpu_worker.py

Author: Team Alpha (Claude) - Session S96B 2026-02-18
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
# SurvivorQualityNet - inlined for subprocess self-containment
# Exact mirror of models/wrappers/neural_net_wrapper.py SurvivorQualityNet
# =============================================================================

class SurvivorQualityNet(nn.Module):
    """
    Neural network for survivor quality prediction.
    Architecture: Linear -> BatchNorm1d -> LeakyReLU/ReLU -> Dropout per layer.
    Matches models/wrappers/neural_net_wrapper.py exactly (v3.2.0).
    """
    def __init__(self, input_size: int, hidden_layers: list,
                 dropout: float = 0.3, use_leaky_relu: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.use_leaky_relu = use_leaky_relu

        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(0.01) if use_leaky_relu else nn.ReLU())
            layers.append(nn.Dropout(dropout))
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
# VRAM hygiene - called in finally: after every trial
# =============================================================================

def _vram_cleanup(model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Training logic for one fold
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
        "checkpoint_validated": True,       # S96A enriched key (parent validates)
        "normalize_features":   normalize,
        "use_leaky_relu":       use_leaky,
    }
    if normalize and scaler_mean is not None:
        result["scaler_shape"] = list(scaler_mean.shape)

    return result, model


# =============================================================================
# Main IPC loop
# =============================================================================

def _run_worker() -> None:
    device, device_desc = _init_cuda()
    _emit({"status": "ready", "device": device_desc})
    log.info(f"Worker ready - {device_desc} - awaiting jobs on stdin")

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
            _emit({"status": "error", "error": f"JSON parse: {exc}", "raw": raw_line[:200]})
            continue

        command = job.get("command", "train")

        if command == "shutdown":
            log.info("Shutdown command - exiting")
            _emit({"status": "shutdown", "reason": "command"})
            break

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

        _emit({"status": "error", "error": f"Unknown command: {command!r}"})

    log.info("Worker exiting")


if __name__ == "__main__":
    _run_worker()
