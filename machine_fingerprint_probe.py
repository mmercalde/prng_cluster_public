#!/usr/bin/env python3
"""
Machine Fingerprint Probe — S119
Team Beta authored, Team Alpha packaged.

Tests whether daily3.json behaves like one homogeneous source
or a mixture of multiple interleaved machine/process signatures.

Uses sliding-window digit-transition matrices (3 x 10x10 per window)
clustered with KMeans k=2..5, scored by silhouette.

Usage:
    ssh rzeus "cd ~/distributed_prng_analysis && python3 machine_fingerprint_probe.py"

Output interpretation:
    silhouette < 0.05                         → mostly homogeneous
    silhouette 0.05-0.12                      → weak/possible mixture
    silhouette > 0.12 + switch_rate > 0.35    → strong interleaved mixture signal
    silhouette > 0.12 + switch_rate < 0.20    → chronological regime blocks
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Load data ──────────────────────────────────────────────────────────────
data_path = Path("daily3.json")
if not data_path.exists():
    print(f"ERROR: {data_path} not found. Run from ~/distributed_prng_analysis/", file=sys.stderr)
    sys.exit(1)

raw = json.load(open(data_path))
# Handle both flat list [134, 840, ...] and object list [{"draw": 134, ...}, ...]
if isinstance(raw[0], dict):
    D = [tuple(map(int, str(r['draw']).zfill(3))) for r in raw]
else:
    D = [tuple(map(int, str(x).zfill(3))) for x in raw]
print(f"Loaded {len(D)} draws from {data_path}")

# ── Build sliding-window transition fingerprints ───────────────────────────
W, S = 400, 50   # window size, stride
X = []
for i in range(0, len(D) - W, S):
    M = np.zeros((3, 10, 10), float)
    for a, b in zip(D[i:i+W-1], D[i+1:i+W]):
        for p in range(3):
            M[p, a[p], b[p]] += 1
    # Smoothed row-normalize (Laplace smoothing alpha=0.5)
    M = (M + 0.5) / (M.sum(axis=2, keepdims=True) + 5.0)
    X.append(M.ravel())

X = np.array(X)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
print(f"Built {len(X)} window fingerprints (W={W}, S={S}, features=300)")
print()

# ── KMeans k=2..5, find best silhouette ───────────────────────────────────
print(f"{'k':<4} {'silhouette':<12} {'switch_rate':<13} {'counts'}")
print("-" * 55)

best = None
results = []
for k in range(2, 6):
    km = KMeans(n_clusters=k, n_init=30, random_state=0).fit(X)
    sil = silhouette_score(X, km.labels_, metric="cosine")
    sw  = np.mean(km.labels_[1:] != km.labels_[:-1])
    counts = np.bincount(km.labels_)
    print(f"{k:<4} {sil:<12.3f} {sw:<13.3f} {counts}")
    results.append((k, sil, sw, km.labels_))
    if best is None or sil > best[1]:
        best = (k, sil, sw, km.labels_)

print()
print(f"Best k={best[0]}  silhouette={best[1]:.3f}  switch_rate={best[2]:.3f}")
print()

# ── Interpretation ─────────────────────────────────────────────────────────
sil, sw = best[1], best[2]

if sil < 0.05:
    interpretation = "MOSTLY HOMOGENEOUS — little evidence for multiple sources"
    step0_implication = "Step 0 minimal value. Session split may be sufficient."
elif sil < 0.12:
    interpretation = "WEAK/POSSIBLE MIXTURE — subtle structure present"
    step0_implication = "Step 0 as context clustering layer recommended. Monitor."
elif sw > 0.35:
    interpretation = "STRONG INTERLEAVED MIXTURE — multiple sources rotating"
    step0_implication = "Step 0 should be context clustering engine, NOT temporal boundary detector."
elif sw < 0.20:
    interpretation = "CHRONOLOGICAL REGIME BLOCKS — temporal change-point structure"
    step0_implication = "Original TRSE temporal boundary framing still valid."
else:
    interpretation = "MIXED SIGNAL — both interleaved and temporal structure present"
    step0_implication = "Step 0 should be hybrid: context clustering + temporal boundary."

print(f"Interpretation: {interpretation}")
print(f"Step 0 implication: {step0_implication}")
print()

# ── Temporal label trace (first 60 windows) ───────────────────────────────
print("Temporal label trace (best k, first 60 windows):")
labels = best[3]
trace = "".join(str(l) for l in labels[:60])
print(f"  {trace}")
print()
print("  Interleaved looks like: 010101 or 01201020")
print("  Chronological looks like: 000000111111222222")
