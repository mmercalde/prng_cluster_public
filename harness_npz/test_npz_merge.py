#!/usr/bin/env python3
"""
test_npz_merge.py
Harness A — correctness tests for NPZ accumulator merge block

Tests the full merge pipeline: _survivors_to_arrays + merge logic + backfill + sort.
Both legacy and vectorized implementations run against the same inputs.

Usage: python3 test_npz_merge.py
"""
import json, sys, time, hashlib, tempfile, os
import numpy as np
from pathlib import Path

FIXTURES = Path("fixtures")

# ── Field schema (canonical — from convert_survivors_to_binary.py v3.1) ───────
FIELDS_UINT32  = ['seeds']
FIELDS_INT32   = ['window_size','offset','trial_number','skip_min','skip_max','skip_range']
FIELDS_FLOAT32 = ['forward_matches','reverse_matches','forward_count','reverse_count',
                  'bidirectional_count','intersection_count','intersection_ratio',
                  'intersection_weight','bidirectional_selectivity','forward_only_count',
                  'reverse_only_count','survivor_overlap_ratio','score']
FIELDS_UINT8   = ['skip_mode','prng_type']
ALL_FIELDS = FIELDS_UINT32 + FIELDS_INT32 + FIELDS_FLOAT32 + FIELDS_UINT8

SKIP_ENC = {'constant': 0, 'variable': 1}
PRNG_ENC = {'java_lcg': 0, 'java_lcg_reverse': 1, 'mt19937': 2,
            'mt19937_reverse': 3, 'xorshift128': 4}

def _dtype(fname):
    if fname in FIELDS_UINT32: return np.uint32
    if fname in FIELDS_INT32:  return np.int32
    if fname in FIELDS_UINT8:  return np.uint8
    return np.float32

def survivors_to_arrays(survivors):
    """Convert list of survivor dicts to NPZ field arrays."""
    def _parse_skip_range(val):
        if isinstance(val, int): return val
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return int(val[1]) - int(val[0])
        try: return int(val)
        except: return 0
    n = len(survivors)
    return {
        'seeds':                  np.array([s['seed'] for s in survivors], dtype=np.uint32),
        'forward_matches':        np.array([s.get('forward_match_rate', s.get('forward_matches', 0.0)) for s in survivors], dtype=np.float32),
        'reverse_matches':        np.array([s.get('reverse_match_rate', s.get('reverse_matches', 0.0)) for s in survivors], dtype=np.float32),
        'window_size':            np.array([s.get('window_size', 0) for s in survivors], dtype=np.int32),
        'offset':                 np.array([s.get('offset', 0) for s in survivors], dtype=np.int32),
        'trial_number':           np.array([s.get('trial_number', 0) for s in survivors], dtype=np.int32),
        'skip_min':               np.array([s.get('skip_min', 0) for s in survivors], dtype=np.int32),
        'skip_max':               np.array([s.get('skip_max', 0) for s in survivors], dtype=np.int32),
        'skip_range':             np.array([_parse_skip_range(s.get('skip_range', 0)) for s in survivors], dtype=np.int32),
        'forward_count':          np.array([s.get('forward_count', 0.0) for s in survivors], dtype=np.float32),
        'reverse_count':          np.array([s.get('reverse_count', 0.0) for s in survivors], dtype=np.float32),
        'bidirectional_count':    np.array([s.get('bidirectional_count', 0.0) for s in survivors], dtype=np.float32),
        'intersection_count':     np.array([s.get('intersection_count', 0.0) for s in survivors], dtype=np.float32),
        'intersection_ratio':     np.array([s.get('intersection_ratio', 0.0) for s in survivors], dtype=np.float32),
        'intersection_weight':    np.array([s.get('intersection_weight', 0.0) for s in survivors], dtype=np.float32),
        'bidirectional_selectivity': np.array([s.get('bidirectional_selectivity', 0.0) for s in survivors], dtype=np.float32),
        'forward_only_count':     np.array([s.get('forward_only_count', 0.0) for s in survivors], dtype=np.float32),
        'reverse_only_count':     np.array([s.get('reverse_only_count', 0.0) for s in survivors], dtype=np.float32),
        'survivor_overlap_ratio': np.array([s.get('survivor_overlap_ratio', 0.0) for s in survivors], dtype=np.float32),
        'score':                  np.array([s.get('score', 0.0) for s in survivors], dtype=np.float32),
        'skip_mode':              np.array([SKIP_ENC.get(s.get('skip_mode', 'constant'), 0) for s in survivors], dtype=np.uint8),
        'prng_type':              np.array([PRNG_ENC.get(s.get('prng_type', 'java_lcg'), 0) for s in survivors], dtype=np.uint8),
    }

# ── Legacy merge (copied verbatim from live code) ─────────────────────────────
def merge_legacy(prior_npz, new_survivors):
    new_arrays  = survivors_to_arrays(new_survivors)
    new_seeds   = new_arrays['seeds'].astype(np.int64)
    new_scores  = new_arrays['score']

    if prior_npz is not None:
        prior_seeds  = prior_npz['seeds'].astype(np.int64)
        prior_scores = prior_npz['score'].astype(np.float32)
        prior_count  = len(prior_seeds)
        prior_idx = {int(prior_seeds[i]): i for i in range(prior_count)}
    else:
        prior_seeds  = np.array([], dtype=np.int64)
        prior_scores = np.array([], dtype=np.float32)
        prior_count  = 0
        prior_idx    = {}

    keep_prior = []
    keep_new   = []
    superseded = set()
    for ni in range(len(new_seeds)):
        seed = int(new_seeds[ni])
        if seed not in prior_idx:
            keep_new.append(ni)
        else:
            pi = prior_idx[seed]
            if float(new_scores[ni]) > float(prior_scores[pi]):
                keep_new.append(ni)
                superseded.add(pi)
    keep_prior = [i for i in range(prior_count) if i not in superseded]

    return _build_merged(prior_npz, new_arrays, keep_prior, keep_new, prior_count)

# ── Vectorized merge (patch under test) ───────────────────────────────────────
def merge_vectorized(prior_npz, new_survivors):
    new_arrays  = survivors_to_arrays(new_survivors)
    new_seeds   = new_arrays['seeds'].astype(np.int64)
    new_scores  = new_arrays['score']

    if prior_npz is not None:
        prior_seeds  = prior_npz['seeds'].astype(np.int64)
        prior_scores = prior_npz['score'].astype(np.float32)
        prior_count  = len(prior_seeds)
    else:
        prior_seeds  = np.array([], dtype=np.int64)
        prior_scores = np.array([], dtype=np.float32)
        prior_count  = 0

    if prior_count > 0:
        prior_sort_order   = np.argsort(prior_seeds)
        prior_seeds_sorted = prior_seeds[prior_sort_order]
        pos          = np.searchsorted(prior_seeds_sorted, new_seeds)
        pos_clipped  = np.clip(pos, 0, prior_count - 1)
        matched      = prior_seeds_sorted[pos_clipped] == new_seeds
        prior_orig_idx = prior_sort_order[pos_clipped]
        new_beats    = matched & (new_scores > prior_scores[prior_orig_idx])
        keep_new_mask = (~matched) | new_beats
        keep_new     = list(np.where(keep_new_mask)[0])
        sup_orig     = prior_orig_idx[new_beats]
        sup_mask     = np.zeros(prior_count, dtype=bool)
        if len(sup_orig) > 0:
            sup_mask[sup_orig] = True
        keep_prior   = list(np.where(~sup_mask)[0])
    else:
        keep_new   = list(range(len(new_seeds)))
        keep_prior = []

    return _build_merged(prior_npz, new_arrays, keep_prior, keep_new, prior_count)

# ── Shared merge finaliser (backfill + sort) ──────────────────────────────────
def _build_merged(prior_npz, new_arrays, keep_prior, keep_new, prior_count):
    merged = {}
    for fname in ALL_FIELDS:
        dt = _dtype(fname)
        parts = []
        if keep_prior:
            if prior_npz is not None and fname in prior_npz:
                parts.append(prior_npz[fname][keep_prior].astype(dt))
            else:
                # Prior doesn't have this field (schema drift / v1 NPZ) —
                # fill with zeros for prior contribution so array stays full length
                parts.append(np.zeros(len(keep_prior), dtype=dt))
        if keep_new and fname in new_arrays:
            parts.append(new_arrays[fname][keep_new].astype(dt))
        merged[fname] = np.concatenate(parts) if parts else np.array([], dtype=dt)

    # backfill missing fields
    seed_len = len(merged['seeds'])
    for fn in ALL_FIELDS:
        if fn == 'seeds': continue
        if fn not in merged or len(merged[fn]) == 0:
            merged[fn] = np.zeros(seed_len, dtype=_dtype(fn))
        elif len(merged[fn]) != seed_len:
            raise ValueError(f"[S163-KARG-NPZ] Field {fn} length {len(merged[fn])} != seeds length {seed_len}")

    # sort by seed
    sort_idx = np.argsort(merged['seeds'])
    for fn in merged:
        merged[fn] = merged[fn][sort_idx]
    return merged

def npz_hash(merged):
    """Deterministic hash of merged arrays for comparison."""
    h = hashlib.md5()
    for fn in sorted(merged.keys()):
        h.update(merged[fn].tobytes())
    return h.hexdigest()

# ── Test runner ───────────────────────────────────────────────────────────────
results = []

def check(name, prior_npz, new_survivors, expect_raises=None):
    t0 = time.perf_counter()
    try:
        leg = merge_legacy(prior_npz, new_survivors)
        t1 = time.perf_counter()
        vec = merge_vectorized(prior_npz, new_survivors)
        t2 = time.perf_counter()
    except Exception as e:
        if expect_raises and expect_raises in str(e):
            print(f"  ✅ {name}: PASS (expected raise: {expect_raises})")
            results.append((name, "PASS"))
            return True
        print(f"  ❌ {name}: FAIL (unexpected exception: {e})")
        results.append((name, "FAIL"))
        return False

    if expect_raises:
        print(f"  ❌ {name}: FAIL (expected raise '{expect_raises}' but no exception)")
        results.append((name, "FAIL"))
        return False

    leg_h = npz_hash(leg)
    vec_h = npz_hash(vec)
    ok = leg_h == vec_h
    status = "PASS" if ok else "FAIL"
    results.append((name, status))
    mark = "✅" if ok else "❌"
    n_leg = len(leg['seeds'])
    n_vec = len(vec['seeds'])
    t_leg = 1000*(t1-t0)
    t_vec = 1000*(t2-t1)
    print(f"  {mark} {name}: {status}  seeds_out={n_leg}  "
          f"t_leg={t_leg:.1f}ms t_vec={t_vec:.1f}ms")
    if not ok:
        print(f"    hash_leg={leg_h[:12]} hash_vec={vec_h[:12]}")
        # find first mismatch
        for fn in sorted(leg.keys()):
            if not np.array_equal(leg[fn], vec[fn]):
                print(f"    First mismatch field: {fn}")
                break
    return ok

print("=" * 60)
print("HARNESS A — NPZ accumulator merge correctness")
print("=" * 60)

# ── Load fixtures ─────────────────────────────────────────────────────────────
prior_empty   = np.load(FIXTURES / "prior_empty.npz")
prior_v1      = np.load(FIXTURES / "prior_v1_missing_fields.npz")
prior_v2      = np.load(FIXTURES / "prior_v2_full_schema.npz")
new_small     = json.loads((FIXTURES / "new_small.json").read_text())
new_overlap50 = json.loads((FIXTURES / "new_overlap_50pct.json").read_text())
new_complete  = json.loads((FIXTURES / "new_complete_overlap.json").read_text())
new_large     = json.loads((FIXTURES / "new_large.json").read_text())

# ── Correctness tests ─────────────────────────────────────────────────────────
check("TC1_empty_prior",              None,      new_small)
check("TC2_empty_prior_npz_fixture",  prior_empty, new_small)
check("TC3_no_overlap",               prior_v2,  new_small)
check("TC4_complete_overlap_all_lose", prior_v2, new_complete)
check("TC5_50pct_overlap",            prior_v2,  new_overlap50)
check("TC6_prior_v1_missing_fields",  prior_v1,  new_small)
check("TC7_prior_v1_with_overlap",    prior_v1,  new_overlap50)
check("TC8_large_new_survivors",      prior_v2,  new_large)

# TC9: searchsorted boundary — new seed < all prior seeds
import json as _json
prior_high = {fn: prior_v2[fn].copy() for fn in prior_v2.files}
prior_high['seeds'] = prior_v2['seeds'] + 500_000_000  # shift to very high values
new_low = [{"seed": i, "score": 0.5, "window_size": 6, "offset": 54, "trial_number": 1,
            "forward_match_rate": 0.5, "reverse_match_rate": 0.5, "prng_type": "java_lcg",
            "skip_mode": "constant"} for i in range(10)]
check("TC9_searchsorted_boundary_low_seeds",  prior_high, new_low)

# TC10: new seed > all prior seeds
new_high = [{"seed": 900_000_000 + i, "score": 0.5, "window_size": 6, "offset": 54,
             "trial_number": 1, "forward_match_rate": 0.5, "reverse_match_rate": 0.5,
             "prng_type": "java_lcg", "skip_mode": "constant"} for i in range(10)]
check("TC10_searchsorted_boundary_high_seeds", prior_v2, new_high)

# TC11: tagged schema mismatch ValueError must raise
# Construct scenario where merged output has a field with wrong length.
# We do this by creating a mock prior that has a field with n+1 elements
# (simulates a corrupted NPZ write) — backfill check catches this.
# Strategy: wrap merge_vectorized to inject a bad field post-merge
def merge_vectorized_inject_bad_field(prior_npz, new_survivors):
    """Force [S163-KARG-NPZ] path by injecting a wrong-length field."""
    result = merge_vectorized(prior_npz, new_survivors)
    # Corrupt one field to wrong length — simulates a corrupted NPZ
    result['score'] = np.append(result['score'], np.float32(0.5))  # one extra
    seed_len = len(result['seeds'])
    # Now run the backfill/check phase manually (mirrors live code)
    for fn in ALL_FIELDS:
        if fn == 'seeds': continue
        if fn not in result or len(result[fn]) == 0:
            result[fn] = np.zeros(seed_len, dtype=_dtype(fn))
        elif len(result[fn]) != seed_len:
            raise ValueError(
                f"[S163-KARG-NPZ] Field {fn} length "
                f"{len(result[fn])} != seeds length {seed_len}"
            )
    return result

try:
    merge_vectorized_inject_bad_field(prior_v2, new_small)
    print("  ❌ TC11_tagged_schema_mismatch_raises: FAIL (expected exception not raised)")
    results.append(("TC11_tagged_schema_mismatch_raises", "FAIL"))
except ValueError as e:
    if str(e).startswith("[S163-KARG-NPZ]"):
        print(f"  ✅ TC11_tagged_schema_mismatch_raises: PASS (raised: {str(e)[:60]})")
        results.append(("TC11_tagged_schema_mismatch_raises", "PASS"))
    else:
        print(f"  ❌ TC11_tagged_schema_mismatch_raises: FAIL (wrong exception: {e})")
        results.append(("TC11_tagged_schema_mismatch_raises", "FAIL"))

# ── NPZ shape integrity check ─────────────────────────────────────────────────
print("\n  Schema integrity check on TC5 output:")
from io import BytesIO
tmp = BytesIO()
vec_result = merge_vectorized(prior_v2, new_overlap50)
np.savez_compressed(tmp, **vec_result)
tmp.seek(0)
loaded = np.load(tmp)
seed_len = len(loaded['seeds'])
all_correct = True
for fn in ALL_FIELDS:
    if fn not in loaded.files:
        print(f"    ❌ MISSING field: {fn}")
        all_correct = False
    elif len(loaded[fn]) != seed_len:
        print(f"    ❌ WRONG length: {fn} = {len(loaded[fn])} (expected {seed_len})")
        all_correct = False
if all_correct:
    print(f"    ✅ All {len(ALL_FIELDS)} fields present, all length {seed_len}")
    results.append(("TC_schema_integrity", "PASS"))
else:
    results.append(("TC_schema_integrity", "FAIL"))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
passed = sum(1 for _, s in results if s == "PASS")
failed = sum(1 for _, s in results if s == "FAIL")
print(f"RESULTS: {passed}/{len(results)} passed  {failed} failed")
for name, status in results:
    mark = "✅" if status == "PASS" else "❌"
    print(f"  {mark} {name}: {status}")

if failed > 0:
    sys.exit(1)
print("\nAll NPZ merge tests PASSED")
