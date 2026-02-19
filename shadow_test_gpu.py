#!/usr/bin/env python3

import argparse
import json
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# LOADERS
# ============================================================

def load_blackbox(path):
    with open(path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        if "draw" in obj[0]:
            return [int(d["draw"]) % 1000 for d in obj]

    if isinstance(obj, list):
        return [int(x) % 1000 for x in obj]

    raise ValueError("Unsupported blackbox format")


def load_pool(path, weighted=False):
    with open(path, "r") as f:
        obj = json.load(f)

    nums = []
    weights = []

    for d in obj["predictions"]:
        nums.append(int(d["predicted_value"]) % 1000)
        weights.append(float(d.get("score", 1.0)))

    if not weighted:
        weights = None

    return nums, weights


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def digits(n):
    return (n // 100, (n // 10) % 10, n % 10)


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return -np.sum(p * np.log2(p))


def extract_features(window):
    w = np.array(window)
    W = len(w)
    d = np.array([digits(x) for x in w])

    feats = []

    # Digit distributions
    for pos in range(3):
        counts = np.bincount(d[:, pos], minlength=10).astype(float)
        p = counts / W
        feats.extend(p.tolist())
        feats.append(entropy(p))

    # Residue entropy
    for mod in (8, 5, 125):
        r = w % mod
        counts = np.bincount(r, minlength=mod).astype(float)
        p = counts / W
        feats.append(entropy(p))

    # Autocorrelation
    x = (w - w.mean()) / (w.std() + 1e-9)
    for lag in (1, 2, 3, 5, 8):
        if lag < W:
            feats.append(np.corrcoef(x[:-lag], x[lag:])[0,1])
        else:
            feats.append(0)

    return np.array(feats, dtype=np.float32)


# ============================================================
# WINDOWING
# ============================================================

def build_windows(seq, W):
    return [seq[i:i+W] for i in range(len(seq)-W+1)]


def sample_windows(windows, max_count):
    if len(windows) <= max_count:
        return windows
    idx = np.random.choice(len(windows), max_count, replace=False)
    return [windows[i] for i in idx]


def featurize(windows):
    return np.stack([extract_features(w) for w in windows])


# ============================================================
# GPU METRICS
# ============================================================

def gpu_nearest_neighbor(real, pred, device):
    real_t = torch.tensor(real, device=device)
    pred_t = torch.tensor(pred, device=device)

    chunk = 1000
    mins = []

    for i in range(0, pred_t.shape[0], chunk):
        p_chunk = pred_t[i:i+chunk]
        d = torch.cdist(p_chunk, real_t)
        mins.append(torch.min(d, dim=1).values)

    mins = torch.cat(mins)
    return mins.cpu().numpy()


def gpu_mmd(real, pred, device):
    real_t = torch.tensor(real, device=device)
    pred_t = torch.tensor(pred, device=device)

    Z = torch.cat([real_t, pred_t], dim=0)
    d2 = torch.cdist(Z, Z) ** 2
    gamma = 1.0 / (torch.median(d2) + 1e-9)

    def kernel(A, B):
        d2 = torch.cdist(A, B) ** 2
        return torch.mean(torch.exp(-gamma * d2))

    kxx = kernel(real_t, real_t)
    kyy = kernel(pred_t, pred_t)
    kxy = kernel(real_t, pred_t)

    return (kxx + kyy - 2*kxy).item()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blackbox", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", default="shadow_gpu_out")
    parser.add_argument("--window", type=int, default=80)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--local-size", type=int, default=2000,
                        help="Number of most recent draws to use")
    parser.add_argument("--weighted-pool", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    real_seq = load_blackbox(args.blackbox)

    # 🔥 LOCAL SLICE
    if args.local_size > 0:
        real_seq = real_seq[-args.local_size:]

    pool_nums, pool_weights = load_pool(args.pool, args.weighted_pool)

    pred_seqs = []
    for _ in range(args.K):
        if pool_weights:
            pred_seqs.append(random.choices(pool_nums, weights=pool_weights, k=len(real_seq)))
        else:
            pred_seqs.append(random.choices(pool_nums, k=len(real_seq)))

    real_windows = sample_windows(build_windows(real_seq, args.window), 5000)

    pred_windows = []
    for s in pred_seqs:
        w = sample_windows(build_windows(s, args.window), 500)
        pred_windows.extend(w)

    pred_windows = sample_windows(pred_windows, 5000)

    X_real = featurize(real_windows)
    X_pred = featurize(pred_windows)

    mu = X_real.mean(0)
    sigma = X_real.std(0) + 1e-9
    X_real = (X_real - mu) / sigma
    X_pred = (X_pred - mu) / sigma

    # PCA
    U,S,Vt = np.linalg.svd(X_real - X_real.mean(0), full_matrices=False)
    comps = Vt[:2]
    Z_real = (X_real - X_real.mean(0)) @ comps.T
    Z_pred = (X_pred - X_real.mean(0)) @ comps.T

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    nn = gpu_nearest_neighbor(Z_real, Z_pred, device0)
    mmd = gpu_mmd(Z_real, Z_pred, device1)

    metrics = {
        "nn_median": float(np.median(nn)),
        "nn_p90": float(np.percentile(nn,90)),
        "coverage_rate_eps1": float((nn < 1.0).mean()),
        "mmd_rbf": float(mmd)
    }

    with open(os.path.join(args.out,"shadow_metrics.json"),"w") as f:
        json.dump(metrics,f,indent=2)

    plt.figure(figsize=(8,6))
    plt.scatter(Z_real[:,0], Z_real[:,1], s=12, alpha=0.4, label="Real (local)")
    plt.scatter(Z_pred[:,0], Z_pred[:,1], s=12, alpha=0.4, label="Predicted Pool")
    plt.legend()
    plt.title("Local Shadow Test (Dual GPU)")
    plt.savefig(os.path.join(args.out,"shadow_scatter.png"), dpi=150)
    plt.close()

    print("DONE. Metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
