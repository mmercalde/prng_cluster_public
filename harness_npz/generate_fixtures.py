#!/usr/bin/env python3
"""
generate_fixtures.py
Generate all NPZ and JSON fixtures for harness_npz.
Run once: python3 generate_fixtures.py
"""
import json, numpy as np, random
from pathlib import Path

FIXTURES = Path("fixtures")
FIXTURES.mkdir(exist_ok=True)
random.seed(42)
np.random.seed(42)

# ── Canonical field schema (from convert_survivors_to_binary.py v3.1) ──────────
FIELDS_UINT32  = ['seeds']
FIELDS_INT32   = ['window_size','offset','trial_number','skip_min','skip_max','skip_range']
FIELDS_FLOAT32 = ['forward_matches','reverse_matches','forward_count','reverse_count',
                  'bidirectional_count','intersection_count','intersection_ratio',
                  'intersection_weight','bidirectional_selectivity','forward_only_count',
                  'reverse_only_count','survivor_overlap_ratio','score']
FIELDS_UINT8   = ['skip_mode','prng_type']
ALL_FIELDS     = FIELDS_UINT32 + FIELDS_INT32 + FIELDS_FLOAT32 + FIELDS_UINT8

def _dtype(fname):
    if fname in FIELDS_UINT32:  return np.uint32
    if fname in FIELDS_INT32:   return np.int32
    if fname in FIELDS_UINT8:   return np.uint8
    return np.float32

def make_survivor(seed, score=None, trial=1, window=6, offset=54):
    return {
        "seed": int(seed),
        "forward_match_rate": round(random.uniform(0.1, 0.9), 8),
        "reverse_match_rate": round(random.uniform(0.1, 0.9), 8),
        "score": round(score if score is not None else random.uniform(0.1, 0.99), 8),
        "window_size": window,
        "offset": offset,
        "trial_number": trial,
        "skip_min": 8,
        "skip_max": 116,
        "skip_range": 108,
        "forward_count": 100.0,
        "reverse_count": 80.0,
        "bidirectional_count": 10.0,
        "intersection_count": 10.0,
        "intersection_ratio": 0.1,
        "intersection_weight": 0.05,
        "bidirectional_selectivity": 1.2,
        "forward_only_count": 90.0,
        "reverse_only_count": 70.0,
        "survivor_overlap_ratio": 0.11,
        "skip_mode": "constant",
        "prng_type": "java_lcg"
    }

def make_npz(survivors, missing_fields=None):
    """Convert survivor list to NPZ arrays, optionally omitting fields."""
    n = len(survivors)
    arrays = {}
    for fname in ALL_FIELDS:
        if missing_fields and fname in missing_fields:
            continue
        if fname == 'seeds':
            arrays[fname] = np.array([s['seed'] for s in survivors], dtype=np.uint32)
        elif fname == 'forward_matches':
            arrays[fname] = np.array([s.get('forward_match_rate', 0.0) for s in survivors], dtype=np.float32)
        elif fname == 'reverse_matches':
            arrays[fname] = np.array([s.get('reverse_match_rate', 0.0) for s in survivors], dtype=np.float32)
        elif fname == 'skip_mode':
            enc = {'constant': 0, 'variable': 1}
            arrays[fname] = np.array([enc.get(s.get('skip_mode','constant'), 0) for s in survivors], dtype=np.uint8)
        elif fname == 'prng_type':
            enc = {'java_lcg': 0, 'java_lcg_reverse': 1, 'mt19937': 2}
            arrays[fname] = np.array([enc.get(s.get('prng_type','java_lcg'), 0) for s in survivors], dtype=np.uint8)
        else:
            arrays[fname] = np.array([s.get(fname, 0.0) for s in survivors], dtype=_dtype(fname))
    return arrays

# ── prior_empty.npz ───────────────────────────────────────────────────────────
arrays = {f: np.array([], dtype=_dtype(f)) for f in ALL_FIELDS}
np.savez_compressed(FIXTURES / "prior_empty.npz", **arrays)
print("Generated: prior_empty.npz")

# ── prior_v2_full_schema.npz (700 survivors, full schema) ─────────────────────
seeds_v2 = np.random.choice(10_000_000, 700, replace=False)
surv_v2 = [make_survivor(s, score=round(random.uniform(0.1, 0.9), 4)) for s in seeds_v2]
np.savez_compressed(FIXTURES / "prior_v2_full_schema.npz", **make_npz(surv_v2))
print("Generated: prior_v2_full_schema.npz (700 seeds, full schema)")

# ── prior_v1_missing_fields.npz (500 survivors, missing late-added fields) ────
seeds_v1 = np.random.choice(10_000_000, 500, replace=False)
surv_v1 = [make_survivor(s) for s in seeds_v1]
MISSING = ['bidirectional_selectivity','forward_only_count','reverse_only_count','survivor_overlap_ratio']
np.savez_compressed(FIXTURES / "prior_v1_missing_fields.npz", **make_npz(surv_v1, missing_fields=MISSING))
print(f"Generated: prior_v1_missing_fields.npz (500 seeds, missing: {MISSING})")

# ── new_small.json (50 survivors, no overlap with prior) ─────────────────────
new_seeds_small = np.random.choice(range(10_000_000, 20_000_000), 50, replace=False)
new_small = [make_survivor(s) for s in new_seeds_small]
(FIXTURES / "new_small.json").write_text(json.dumps(new_small, indent=2))
print("Generated: new_small.json (50 seeds, no overlap)")

# ── new_overlap_50pct.json (200 survivors, 50% overlap with prior_v2) ─────────
overlap_seeds = seeds_v2[:100]  # first 100 of prior_v2
unique_seeds  = np.random.choice(range(20_000_000, 30_000_000), 100, replace=False)
overlap_surv  = [make_survivor(s, score=round(random.uniform(0.1, 0.99), 4)) for s in overlap_seeds]
unique_surv   = [make_survivor(s) for s in unique_seeds]
new_overlap_50 = overlap_surv + unique_surv
random.shuffle(new_overlap_50)
(FIXTURES / "new_overlap_50pct.json").write_text(json.dumps(new_overlap_50, indent=2))
print("Generated: new_overlap_50pct.json (200 seeds, 50% overlap with prior_v2)")

# ── new_complete_overlap.json (all seeds exist in prior_v2, lower scores) ─────
complete_overlap = []
for s in surv_v2:
    low_score = max(0.01, s['score'] - 0.3)
    complete_overlap.append(make_survivor(s['seed'], score=round(low_score, 4)))
(FIXTURES / "new_complete_overlap.json").write_text(json.dumps(complete_overlap, indent=2))
print("Generated: new_complete_overlap.json (all seeds overlap, all lose to prior)")

# ── new_large.json (50K survivors, ~30% overlap with prior_v2) ────────────────
overlap_large = seeds_v2[:200]  # 200 overlap
unique_large  = np.random.choice(range(30_000_000, 60_000_000), 49800, replace=False)
large_surv = (
    [make_survivor(s, score=round(random.uniform(0.5, 0.99), 4)) for s in overlap_large] +
    [make_survivor(s) for s in unique_large]
)
random.shuffle(large_surv)
(FIXTURES / "new_large.json").write_text(json.dumps(large_surv, indent=2))
print("Generated: new_large.json (50K seeds)")

print("\nAll fixtures generated.")
