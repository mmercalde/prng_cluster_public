#!/usr/bin/env python3
"""
test_deduplicate.py
Harness A — correctness tests for deduplicate_survivors()

Tests both the legacy (live code) and vectorized (patched) implementations
against the full TB-spec correctness matrix.

Usage: python3 test_deduplicate.py
"""
import time, sys, json
import numpy as np
from pathlib import Path

# ── Extract legacy implementation from live code ──────────────────────────────
# Copied verbatim from window_optimizer_integration_final.py line ~1423
def deduplicate_survivors_legacy(survivor_list):
    """Keep survivor with highest per-seed score for each unique seed."""
    seed_map = {}
    for survivor in survivor_list:
        seed = survivor['seed']
        if seed not in seed_map or survivor.get('score', 0) > seed_map[seed].get('score', 0):
            seed_map[seed] = survivor
    return list(seed_map.values())

# ── Vectorized implementation (patch under test) ──────────────────────────────
def deduplicate_survivors_vectorized(survivor_list):
    """[S163-KARG] Vectorized via numpy — replaces O(N) pure Python dict loop."""
    if not survivor_list:
        return []
    import numpy as _np_dedup
    seeds  = _np_dedup.array([s['seed'] for s in survivor_list], dtype=_np_dedup.int64)
    scores = _np_dedup.array([s.get('score', 0.0) for s in survivor_list], dtype=_np_dedup.float32)
    order  = _np_dedup.lexsort((-scores, seeds))
    sorted_seeds = seeds[order]
    keep_mask = _np_dedup.concatenate(([True], sorted_seeds[1:] != sorted_seeds[:-1]))
    keep_idx  = order[keep_mask]
    return [survivor_list[i] for i in keep_idx]

# ── Test helpers ──────────────────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
results = []

def make_s(seed, score, **kwargs):
    d = {"seed": seed, "score": score, "window_size": 6, "offset": 54,
         "trial_number": 1, "prng_type": "java_lcg", "skip_mode": "constant"}
    d.update(kwargs)
    return d

def check(name, survivor_list):
    """Run both implementations and assert they produce identical seed→score mappings."""
    t0 = time.perf_counter()
    leg = deduplicate_survivors_legacy(survivor_list)
    t1 = time.perf_counter()
    vec = deduplicate_survivors_vectorized(survivor_list)
    t2 = time.perf_counter()

    leg_map = {s['seed']: round(float(s.get('score', 0)), 6) for s in leg}
    vec_map = {s['seed']: round(float(s.get('score', 0)), 6) for s in vec}

    ok = leg_map == vec_map
    status = PASS if ok else FAIL
    results.append((name, status))

    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}: {status}  legacy={len(leg)} vec={len(vec)}  "
          f"t_leg={1000*(t1-t0):.2f}ms t_vec={1000*(t2-t1):.2f}ms")
    if not ok:
        only_leg = set(leg_map) - set(vec_map)
        only_vec = set(vec_map) - set(leg_map)
        diff_score = {k for k in leg_map if k in vec_map and leg_map[k] != vec_map[k]}
        if only_leg: print(f"    Seeds only in legacy:     {list(only_leg)[:5]}")
        if only_vec: print(f"    Seeds only in vectorized: {list(only_vec)[:5]}")
        if diff_score: print(f"    Score mismatch seeds:     {list(diff_score)[:5]}")
    return ok

# ── Test cases ────────────────────────────────────────────────────────────────
print("=" * 60)
print("HARNESS A — deduplicate_survivors() correctness")
print("=" * 60)

# TC1: empty list
check("TC1_empty_list", [])

# TC2: single record
check("TC2_single_record", [make_s(42, 0.5)])

# TC3: all unique seeds
check("TC3_all_unique",
      [make_s(i*100, round(0.1 + i*0.05, 3)) for i in range(20)])

# TC4: repeated same seed — keep highest score
check("TC4_same_seed_keep_highest", [
    make_s(999, 0.3),
    make_s(999, 0.9),  # winner
    make_s(999, 0.5),
    make_s(999, 0.7),
])

# TC5: mixed repeated seeds
check("TC5_mixed_repeated", [
    make_s(100, 0.5), make_s(200, 0.3), make_s(100, 0.9),
    make_s(300, 0.1), make_s(200, 0.2), make_s(100, 0.7),
    make_s(400, 0.8),
])

# TC6: score=0 edge case (all zeros)
check("TC6_all_zero_scores",
      [make_s(i, 0.0) for i in range(10)])

# TC7: duplicate seeds, equal scores — any consistent winner is fine
check("TC7_equal_scores_tiebreak", [
    make_s(500, 0.5), make_s(500, 0.5), make_s(500, 0.5)
])

# TC8: missing score key — defaults to 0
check("TC8_missing_score_key", [
    {"seed": 111, "window_size": 4},
    {"seed": 111, "score": 0.7, "window_size": 4},
    {"seed": 222, "window_size": 5},
])

# TC9: large scale — 1.4M records
print("\n  TC9_large_scale_1_4M (this takes a moment)...")
np.random.seed(99)
n = 1_400_000
seeds_large = np.random.randint(0, 500_000, n)
scores_large = np.random.rand(n)
large_data = [{"seed": int(seeds_large[i]), "score": float(scores_large[i])} for i in range(n)]
check("TC9_large_scale_1_4M", large_data)

# TC10: boundary — seeds at int32 max
check("TC10_large_seed_values", [
    make_s(2**31 - 1, 0.5),
    make_s(2**31 - 2, 0.9),
    make_s(2**31 - 1, 0.8),  # should lose to 0.8 wait — 0.8 > 0.5 so wins
])

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
passed = sum(1 for _, s in results if s == PASS)
failed = sum(1 for _, s in results if s == FAIL)
print(f"RESULTS: {passed}/{len(results)} passed  {failed} failed")
for name, status in results:
    mark = "✅" if status == PASS else "❌"
    print(f"  {mark} {name}: {status}")

if failed > 0:
    sys.exit(1)
print("\nAll deduplicate_survivors tests PASSED")
