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

    for pos in range(3):
        counts = np.bincount(d[:, pos], minlength=10).astype(float)
        p = counts / W
        feats.extend(p.tolist())
        feats.append(entropy(p))

    for mod in (8, 5, 125):
        r = w % mod
        counts = np.bincount(r, minlength=mod).astype(float)
        p = counts / W
        feats.append(entropy(p))

    x = (w - w.mean()) / (w.std() + 1e-9)
    for lag in (1,2,3,5,8):
        if lag < W:
            feats.append(np.corrcoef(x[:-lag], x[lag:])[0,1])
        else:
            feats.append(0)

    return np.array(feats, dtype=np.float32)


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
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blackbox", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--window", type=int, default=80)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--weighted-pool", action="store_true")
    parser.add_argument("--out", default="shadow_overlay")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    real_seq_full = load_blackbox(args.blackbox)
    pool_nums, pool_weights = load_pool(args.pool, args.weighted_pool)

    # Generate predicted sequences
    pred_seqs = []
    for _ in range(args.K):
        if pool_weights:
            pred_seqs.append(random.choices(pool_nums, weights=pool_weights, k=2000))
        else:
            pred_seqs.append(random.choices(pool_nums, k=2000))

    pred_windows = []
    for s in pred_seqs:
        w = sample_windows(build_windows(s, args.window), 500)
        pred_windows.extend(w)

    pred_windows = sample_windows(pred_windows, 5000)
    X_pred = featurize(pred_windows)

    # Prepare real slices
    slices = {
        "Last 200": real_seq_full[-200:],
        "Last 500": real_seq_full[-500:],
        "Last 1000": real_seq_full[-1000:]
    }

    real_embeddings = {}

    for label, seq in slices.items():
        wins = sample_windows(build_windows(seq, args.window), 5000)
        real_embeddings[label] = featurize(wins)

    # Standardize using Last 1000 as anchor
    anchor = real_embeddings["Last 1000"]
    mu = anchor.mean(0)
    sigma = anchor.std(0) + 1e-9

    X_pred = (X_pred - mu) / sigma
    for k in real_embeddings:
        real_embeddings[k] = (real_embeddings[k] - mu) / sigma

    # PCA on anchor
    U,S,Vt = np.linalg.svd(real_embeddings["Last 1000"] - real_embeddings["Last 1000"].mean(0),
                           full_matrices=False)
    comps = Vt[:2]

    Z_pred = (X_pred - real_embeddings["Last 1000"].mean(0)) @ comps.T

    Z_real = {}
    for k in real_embeddings:
        Z_real[k] = (real_embeddings[k] - real_embeddings["Last 1000"].mean(0)) @ comps.T

    # Plot
    plt.figure(figsize=(9,7))
    colors = {"Last 200":"green", "Last 500":"blue", "Last 1000":"purple"}

    for k in Z_real:
        plt.scatter(Z_real[k][:,0], Z_real[k][:,1],
                    s=12, alpha=0.4, label=k, c=colors[k])

    plt.scatter(Z_pred[:,0], Z_pred[:,1],
                s=12, alpha=0.4, label="Predicted Pool", c="orange")

    plt.legend()
    plt.title("Overlay: Local Regime Comparison")
    plt.savefig(os.path.join(args.out,"overlay_scatter.png"), dpi=150)
    plt.close()

    print("DONE. Overlay saved.")


if __name__ == "__main__":
    main()
