#!/usr/bin/env python3

import argparse
import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LOADERS
# ============================================================

def load_blackbox(path):
    with open(path, "r") as f:
        obj = json.load(f)

    # Case: synthetic structured list of dicts
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        if "draw" in obj[0]:
            return [int(d["draw"]) % 1000 for d in obj]
    # Case: flat list
    if isinstance(obj, list):
        return [int(x) % 1000 for x in obj]

    raise ValueError("Unsupported blackbox format")


def load_pool(path, weighted=False):
    with open(path, "r") as f:
        obj = json.load(f)

    if "predictions" not in obj:
        raise ValueError("Expected 'predictions' key in pool JSON")

    nums = []
    weights = []

    for d in obj["predictions"]:
        if "predicted_value" in d:
            nums.append(int(d["predicted_value"]) % 1000)
            weights.append(float(d.get("score", 1.0)))
        else:
            continue

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

    # Digit distributions (3x10)
    for pos in range(3):
        counts = np.bincount(d[:, pos], minlength=10).astype(float)
        p = counts / W
        feats.extend(p.tolist())
        feats.append(entropy(p))

    # Residue entropy (mod 8, 5, 125)
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


def featurize(windows):
    return np.stack([extract_features(w) for w in windows])


# ============================================================
# BASELINES
# ============================================================

def baseline_uniform(length, K):
    return [[random.randint(0,999) for _ in range(length)] for _ in range(K)]


def baseline_digit_match(real_seq, length, K):
    d = np.array([digits(x) for x in real_seq])
    probs = []
    for pos in range(3):
        counts = np.bincount(d[:,pos], minlength=10).astype(float)
        probs.append(counts/counts.sum())

    seqs = []
    for _ in range(K):
        s = []
        for _ in range(length):
            h = np.random.choice(10, p=probs[0])
            t = np.random.choice(10, p=probs[1])
            o = np.random.choice(10, p=probs[2])
            s.append(h*100+t*10+o)
        seqs.append(s)
    return seqs


# ============================================================
# METRICS
# ============================================================

def nearest_dist(A, B):
    diffs = A[:,None,:] - B[None,:,:]
    d2 = np.sum(diffs**2, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def mmd_rbf(X, Y):
    Z = np.vstack([X,Y])
    d2 = np.sum((Z[:,None,:]-Z[None,:,:])**2, axis=2)
    gamma = 1.0 / (np.median(d2)+1e-9)

    def k(A,B):
        d2 = np.sum((A[:,None,:]-B[None,:,:])**2, axis=2)
        return np.mean(np.exp(-gamma*d2))

    return k(X,X)+k(Y,Y)-2*k(X,Y)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blackbox", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", default="shadow_out")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--K", type=int, default=150)
    ap.add_argument("--weighted-pool", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    real_seq = load_blackbox(args.blackbox)
    pool_nums, pool_weights = load_pool(args.pool, args.weighted_pool)

    # Synthetic sequences from pool
    pred_seqs = []
    for _ in range(args.K):
        if pool_weights:
            pred_seqs.append(random.choices(pool_nums, weights=pool_weights, k=len(real_seq)))
        else:
            pred_seqs.append(random.choices(pool_nums, k=len(real_seq)))

    # Windows
    real_windows = build_windows(real_seq, args.window)
    pred_windows = []
    for s in pred_seqs:
        pred_windows.extend(build_windows(s, args.window))

    X_real = featurize(real_windows)
    X_pred = featurize(pred_windows)

    # Standardize on real
    mu = X_real.mean(axis=0)
    sigma = X_real.std(axis=0)+1e-9
    X_real = (X_real-mu)/sigma
    X_pred = (X_pred-mu)/sigma

    # PCA 2D
    U,S,Vt = np.linalg.svd(X_real - X_real.mean(0), full_matrices=False)
    comps = Vt[:2]
    Z_real = (X_real - X_real.mean(0)) @ comps.T
    Z_pred = (X_pred - X_real.mean(0)) @ comps.T

    # Metrics
    nn = nearest_dist(Z_pred, Z_real)
    metrics = {
        "nn_median": float(np.median(nn)),
        "nn_p90": float(np.percentile(nn,90)),
        "coverage_rate_eps1": float((nn<1.0).mean()),
        "mmd_rbf": float(mmd_rbf(Z_real, Z_pred))
    }

    with open(os.path.join(args.out,"shadow_metrics.json"),"w") as f:
        json.dump(metrics,f,indent=2)

    # Plot
    plt.figure(figsize=(10,8))
    plt.scatter(Z_real[:,0],Z_real[:,1],s=8,alpha=0.3,label="Real")
    plt.scatter(Z_pred[:,0],Z_pred[:,1],s=8,alpha=0.3,label="Predicted")
    plt.legend()
    plt.title("Shadow Test PCA")
    plt.savefig(os.path.join(args.out,"shadow_scatter.png"),dpi=150)
    plt.close()

    print("Done. Results in:", args.out)


if __name__ == "__main__":
    main()

