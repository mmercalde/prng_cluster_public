#!/usr/bin/env python3
"""
benchmark_npz_merge.py
Harness A — performance benchmarks for NPZ dedup + merge

Runs both legacy and vectorized implementations at TB-required scales:
  100K, 500K, 700K+700K merge, 1.4M dedup, production-like overlap

Usage: python3 benchmark_npz_merge.py
"""
import time, sys
import numpy as np

# ── Import both implementations from test_npz_merge ───────────────────────────
sys.path.insert(0, '.')
from test_npz_merge import (
    merge_legacy, merge_vectorized, survivors_to_arrays,
    ALL_FIELDS, FIELDS_UINT32, FIELDS_INT32, FIELDS_FLOAT32, FIELDS_UINT8
)
from test_deduplicate import (
    deduplicate_survivors_legacy, deduplicate_survivors_vectorized
)

np.random.seed(42)

def _dtype(fname):
    if fname in FIELDS_UINT32: return np.uint32
    if fname in FIELDS_INT32:  return np.int32
    if fname in FIELDS_UINT8:  return np.uint8
    return np.float32

SKIP_ENC = {'constant': 0, 'variable': 1}
PRNG_ENC = {'java_lcg': 0}

def make_prior_npz(n):
    """Generate a synthetic prior NPZ with n unique seeds."""
    seeds = np.random.choice(50_000_000, n, replace=False).astype(np.uint32)
    arrays = {'seeds': seeds}
    for fn in ALL_FIELDS:
        if fn == 'seeds': continue
        arrays[fn] = np.random.rand(n).astype(_dtype(fn))
    return arrays

def make_new_survivors(n, overlap_seeds=None, overlap_pct=0.5):
    """Generate n unique new survivors with optional overlap."""
    if overlap_seeds is not None:
        n_overlap = min(int(n * overlap_pct), len(overlap_seeds))
        overlap = overlap_seeds[:n_overlap].tolist()
        n_unique = n - n_overlap
        unique = np.random.choice(range(50_000_000, 100_000_000), n_unique, replace=False).tolist()
        all_seeds = overlap + unique
    else:
        all_seeds = np.random.choice(range(50_000_000, 100_000_000), n, replace=False).tolist()
    np.random.shuffle(all_seeds)

    return [{
        "seed": int(s),
        "score": float(np.random.rand()),
        "forward_match_rate": float(np.random.rand()),
        "reverse_match_rate": float(np.random.rand()),
        "window_size": 6, "offset": 54, "trial_number": 1,
        "skip_min": 8, "skip_max": 116, "skip_range": 108,
        "forward_count": 100.0, "reverse_count": 80.0,
        "bidirectional_count": 10.0, "intersection_count": 10.0,
        "intersection_ratio": 0.1, "intersection_weight": 0.05,
        "bidirectional_selectivity": 1.2, "forward_only_count": 90.0,
        "reverse_only_count": 70.0, "survivor_overlap_ratio": 0.11,
        "skip_mode": "constant", "prng_type": "java_lcg"
    } for s in all_seeds]

def bench_dedup(label, n):
    data = [{"seed": int(s), "score": float(np.random.rand())}
            for s in np.random.randint(0, n//2, n)]  # ~50% dupe rate
    t0 = time.perf_counter()
    r_leg = deduplicate_survivors_legacy(data)
    t1 = time.perf_counter()
    r_vec = deduplicate_survivors_vectorized(data)
    t2 = time.perf_counter()
    t_leg = t1 - t0
    t_vec = t2 - t1
    speedup = t_leg / t_vec if t_vec > 0 else 0
    ok = {s['seed']: s['score'] for s in r_leg} == {s['seed']: s['score'] for s in r_vec}
    return t_leg, t_vec, speedup, ok

def bench_merge(label, n_prior, n_new, overlap_pct=0.5):
    prior = make_prior_npz(n_prior) if n_prior > 0 else None
    overlap_seeds = prior['seeds'].astype(int) if prior is not None else None
    new_surv = make_new_survivors(n_new, overlap_seeds=overlap_seeds, overlap_pct=overlap_pct)
    t0 = time.perf_counter()
    r_leg = merge_legacy(prior, new_surv)
    t1 = time.perf_counter()
    r_vec = merge_vectorized(prior, new_surv)
    t2 = time.perf_counter()
    t_leg = t1 - t0
    t_vec = t2 - t1
    speedup = t_leg / t_vec if t_vec > 0 else 0
    ok = len(r_leg['seeds']) == len(r_vec['seeds'])
    return t_leg, t_vec, speedup, ok

# ── Run benchmarks ────────────────────────────────────────────────────────────
print("=" * 70)
print("HARNESS A — Performance Benchmarks")
print("=" * 70)

print("\n── deduplicate_survivors() ─────────────────────────────────────────────")
print(f"{'Label':<35} {'t_legacy':>10} {'t_vector':>10} {'speedup':>9} {'match':>6}")
print("-" * 70)
for n, label in [(100_000, "100K"), (500_000, "500K"), (1_400_000, "1.4M")]:
    t_leg, t_vec, spd, ok = bench_dedup(label, n)
    mark = "OK" if ok else "FAIL"
    print(f"  dedup_{label:<29} {t_leg:>9.3f}s {t_vec:>9.3f}s {spd:>8.1f}x {mark:>6}")

print("\n── NPZ accumulator merge() ────────────────────────────────────────────")
print(f"{'Label':<35} {'t_legacy':>10} {'t_vector':>10} {'speedup':>9} {'match':>6}")
print("-" * 70)

configs = [
    (0,       100_000, 0.0,  "empty_prior+100K_new"),
    (100_000, 100_000, 0.5,  "100K_prior+100K_new_50pct"),
    (500_000, 500_000, 0.5,  "500K_prior+500K_new_50pct"),
    (700_000, 700_000, 0.5,  "700K_prior+700K_new_50pct"),
    (700_000, 700_000, 0.3,  "700K_prior+700K_new_30pct"),
]
for n_prior, n_new, ovlp, label in configs:
    t_leg, t_vec, spd, ok = bench_merge(label, n_prior, n_new, ovlp)
    mark = "OK" if ok else "FAIL"
    print(f"  {label:<35} {t_leg:>9.3f}s {t_vec:>9.3f}s {spd:>8.1f}x {mark:>6}")

print()
print("=" * 70)
print("Benchmark complete.")
print("NOTE: speedup > 1.0x = vectorized is faster.")
print("      All 'match' must be OK for the patch to be safe.")
