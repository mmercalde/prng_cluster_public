#!/usr/bin/env python3

import argparse
import json
import random
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------
# Loaders
# ---------------------------

def load_blackbox(path: str) -> List[int]:
    with open(path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "draw" in obj[0]:
        return [int(d["draw"]) % 1000 for d in obj]

    return [int(x) % 1000 for x in obj]


def load_survivor_seeds(path: str) -> List[int]:
    with open(path, "r") as f:
        data = json.load(f)

    seeds = [int(d["seed"]) for d in data if "seed" in d]
    if not seeds:
        raise ValueError("No seeds found in survivors file")

    return seeds


# ---------------------------
# Java LCG
# ---------------------------

JAVA_MULT = 25214903917
JAVA_ADD  = 11
JAVA_MASK = (1 << 48) - 1

def java_lcg_sequence_mod1000(seed32: int, n: int) -> List[int]:
    seed = (seed32 ^ JAVA_MULT) & JAVA_MASK
    out = []
    for _ in range(n):
        seed = (seed * JAVA_MULT + JAVA_ADD) & JAVA_MASK
        x = (seed >> 17) & 0x7FFFFFFF
        out.append(int(x % 1000))
    return out


# ---------------------------
# Feature extraction
# ---------------------------

def digits(n):
    return (n // 100, (n // 10) % 10, n % 10)

def entropy(p):
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())

def extract_features(window):
    w = np.asarray(window)
    W = len(w)
    d = np.asarray([digits(int(x)) for x in w])

    feats = []

    for pos in range(3):
        counts = np.bincount(d[:, pos], minlength=10)
        p = counts / W
        feats.extend(p.tolist())
        feats.append(entropy(p))

    for mod in (8, 5, 125):
        r = w % mod
        counts = np.bincount(r, minlength=mod)
        p = counts / W
        feats.append(entropy(p))

    x = (w - w.mean()) / (w.std() + 1e-9)
    for lag in (1, 2, 3, 5, 8):
        if lag < W:
            c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            feats.append(0 if not np.isfinite(c) else c)
        else:
            feats.append(0)

    return np.asarray(feats, dtype=np.float32)


def build_windows(seq, W):
    return [seq[i:i+W] for i in range(len(seq) - W + 1)]


def featurize(windows):
    return np.stack([extract_features(w) for w in windows], axis=0)


# ---------------------------
# GPU PCA
# ---------------------------

def project_pca_real_basis_gpu(X_real, X_pred, device="cuda"):
    Xr = torch.tensor(X_real, device=device, dtype=torch.float32)
    Xp = torch.tensor(X_pred, device=device, dtype=torch.float32)

    mu = Xr.mean(dim=0, keepdim=True)
    Xr_c = Xr - mu
    Xp_c = Xp - mu

    U, S, V = torch.pca_lowrank(Xr_c, q=2)
    V2 = V[:, :2]

    Zr = Xr_c @ V2
    Zp = Xp_c @ V2

    return Zr.detach().cpu().numpy(), Zp.detach().cpu().numpy()


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blackbox", required=True)
    ap.add_argument("--survivors", required=True)
    ap.add_argument("--window", type=int, default=80)
    ap.add_argument("--local-size", type=int, default=2000)
    ap.add_argument("--seed-sample", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=2000)
    ap.add_argument("--interval-ms", type=int, default=1500)
    args = ap.parse_args()

    real_full = load_blackbox(args.blackbox)
    survivor_seeds = load_survivor_seeds(args.survivors)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fig, ax = plt.subplots(figsize=(10, 7))
    scat_real = ax.scatter([], [], s=10, alpha=0.4, label="Real")
    scat_pred = ax.scatter([], [], s=10, alpha=0.4, label="Predicted")
    ax.legend()

    def recompute_frame():
        real_seq = real_full[-args.local_size:]
        rw = build_windows(real_seq, args.window)
        rw = random.sample(rw, min(2000, len(rw)))
        X_real = featurize(rw)

        seeds = random.sample(survivor_seeds, min(args.seed_sample, len(survivor_seeds)))
        pw = []

        for s in seeds:
            seq = java_lcg_sequence_mod1000(s, args.seq_len)
            w = build_windows(seq, args.window)
            if w:
                pw.extend(random.sample(w, min(20, len(w))))

        if not pw:
            raise RuntimeError("No predicted windows generated")

        X_pred = featurize(pw)

        mu = X_real.mean(0)
        sigma = X_real.std(0) + 1e-9
        Xr = (X_real - mu) / sigma
        Xp = (X_pred - mu) / sigma

        return project_pca_real_basis_gpu(Xr, Xp, device=device)

    def update(_):
        try:
            Zr, Zp = recompute_frame()
            print("Frame OK | Zr:", Zr.shape, "Zp:", Zp.shape)

            scat_real.set_offsets(Zr[:, :2])
            scat_pred.set_offsets(Zp[:, :2])

            allz = np.vstack([Zr[:, :2], Zp[:, :2]])
            xmin, ymin = allz.min(axis=0)
            xmax, ymax = allz.max(axis=0)
            pad = 0.1
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)

        except Exception as e:
            print("ERROR IN FRAME:", e)

        return scat_real, scat_pred

    FuncAnimation(fig, update, interval=args.interval_ms)
    plt.show()


if __name__ == "__main__":
    main()
