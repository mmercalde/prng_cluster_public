#!/usr/bin/env python3
"""
recover_s165_survivors.py

Reconstructs bidirectional survivors from S165 result JSONs and merges
into bidirectional_survivors_all.npz accumulator.

The window_optimizer.py process was killed before completing the NPZ merge.
This script reconstructs the bidirectional intersection from the raw sieve
result files and merges them into the accumulator.

Usage:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    python3 recover_s165_survivors.py
"""

import json
import os
import numpy as np
from pathlib import Path

ACCUM_NPZ  = "bidirectional_survivors_all.npz"
BINARY_NPZ = "bidirectional_survivors_binary.npz"

# S165 result files — Trial 2, W6_O0 config
TRIAL_PAIRS = [
    {
        'label': 'constant_skip_t2',
        'forward': 'results/window_opt_forward_6_0_t2.json',
        'reverse': 'results/window_opt_reverse_6_0_t2.json',
        'skip_mode': 'constant',
        'window_size': 6,
        'offset': 0,
        'trial_number': 2,
    },
    {
        'label': 'variable_skip_t2',
        'forward': 'results/window_opt_forward_hybrid_6_0_t2.json',
        'reverse': 'results/window_opt_reverse_hybrid_6_0_t2.json',
        'skip_mode': 'variable',
        'window_size': 6,
        'offset': 0,
        'trial_number': 2,
    },
]

def load_survivors(path):
    """Load survivor seeds and match rates from result JSON."""
    print(f"  Loading {path}...")
    with open(path) as f:
        d = json.load(f)

    survivors = d.get('survivors', [])
    match_rates = d.get('match_rates', [])
    count = d.get('survivor_count', len(survivors))

    print(f"    survivor_count: {count:,}, list length: {len(survivors):,}")

    if not survivors:
        return {}, count

    # Build seed -> match_rate map
    seed_map = {}
    for i, s in enumerate(survivors):
        seed = int(s)
        rate = float(match_rates[i]) if i < len(match_rates) else 0.5
        seed_map[seed] = rate

    return seed_map, count

def merge_into_npz(new_survivors, accum_path, binary_path):
    """Merge new bidirectional survivors into existing NPZ accumulator."""
    if not new_survivors:
        print("  No new survivors to merge.")
        return 0

    seeds_arr = np.array([s['seed'] for s in new_survivors], dtype=np.int64)
    fwd_rates  = np.array([s['forward_match_rate'] for s in new_survivors], dtype=np.float32)
    rev_rates  = np.array([s['reverse_match_rate'] for s in new_survivors], dtype=np.float32)
    scores     = np.array([s['score'] for s in new_survivors], dtype=np.float32)
    w_sizes    = np.array([s['window_size'] for s in new_survivors], dtype=np.int32)
    offsets    = np.array([s['offset'] for s in new_survivors], dtype=np.int32)
    trials     = np.array([s['trial_number'] for s in new_survivors], dtype=np.int32)

    # Load existing accumulator
    if os.path.exists(accum_path):
        prior = np.load(accum_path)
        prior_seeds  = prior['seeds']
        prior_fwd    = prior.get('forward_match_rate', np.zeros(len(prior_seeds), dtype=np.float32))
        prior_rev    = prior.get('reverse_match_rate', np.zeros(len(prior_seeds), dtype=np.float32))
        prior_scores = prior.get('score',              np.zeros(len(prior_seeds), dtype=np.float32))
        prior_wsizes = prior.get('window_size',        np.zeros(len(prior_seeds), dtype=np.int32))
        prior_offsets= prior.get('offset',             np.zeros(len(prior_seeds), dtype=np.int32))
        prior_trials = prior.get('trial_number',       np.zeros(len(prior_seeds), dtype=np.int32))
        print(f"  Prior accumulator: {len(prior_seeds):,} seeds")
    else:
        prior_seeds   = np.array([], dtype=np.int64)
        prior_fwd     = np.array([], dtype=np.float32)
        prior_rev     = np.array([], dtype=np.float32)
        prior_scores  = np.array([], dtype=np.float32)
        prior_wsizes  = np.array([], dtype=np.int32)
        prior_offsets = np.array([], dtype=np.int32)
        prior_trials  = np.array([], dtype=np.int32)
        print(f"  No prior accumulator — creating new")

    # Merge: highest score wins for duplicate seeds
    merged = {}
    for i, seed in enumerate(prior_seeds):
        merged[int(seed)] = {
            'fwd': float(prior_fwd[i]),
            'rev': float(prior_rev[i]),
            'score': float(prior_scores[i]),
            'wsize': int(prior_wsizes[i]),
            'offset': int(prior_offsets[i]),
            'trial': int(prior_trials[i]),
        }

    new_count = 0
    updated_count = 0
    for i, seed in enumerate(seeds_arr):
        s = int(seed)
        new_score = float(scores[i])
        if s not in merged or new_score > merged[s]['score']:
            merged[s] = {
                'fwd': float(fwd_rates[i]),
                'rev': float(rev_rates[i]),
                'score': new_score,
                'wsize': int(w_sizes[i]),
                'offset': int(offsets[i]),
                'trial': int(trials[i]),
            }
            if s not in merged:
                new_count += 1
            else:
                updated_count += 1

    print(f"  New unique seeds added: {len(merged) - len(prior_seeds):,}")
    print(f"  Total after merge: {len(merged):,}")

    # Write merged NPZ atomically
    all_seeds  = np.array(list(merged.keys()), dtype=np.int64)
    all_fwd    = np.array([v['fwd']    for v in merged.values()], dtype=np.float32)
    all_rev    = np.array([v['rev']    for v in merged.values()], dtype=np.float32)
    all_scores = np.array([v['score']  for v in merged.values()], dtype=np.float32)
    all_wsizes = np.array([v['wsize']  for v in merged.values()], dtype=np.int32)
    all_offsets= np.array([v['offset'] for v in merged.values()], dtype=np.int32)
    all_trials = np.array([v['trial']  for v in merged.values()], dtype=np.int32)

    tmp = accum_path + ".recover.tmp"
    np.savez_compressed(tmp,
        seeds=all_seeds,
        forward_match_rate=all_fwd,
        reverse_match_rate=all_rev,
        score=all_scores,
        window_size=all_wsizes,
        offset=all_offsets,
        trial_number=all_trials,
        schema_version=np.array([3]),
    )
    os.replace(tmp, accum_path)
    print(f"  ✅ Written {accum_path}: {len(merged):,} seeds")

    # Also write binary NPZ (Steps 2-6 format)
    tmp_bin = binary_path + ".recover.tmp"
    np.savez_compressed(tmp_bin,
        seeds=all_seeds,
        forward_match_rate=all_fwd,
        reverse_match_rate=all_rev,
        score=all_scores,
        window_size=all_wsizes,
        offset=all_offsets,
        trial_number=all_trials,
        schema_version=np.array([3]),
    )
    os.replace(tmp_bin, binary_path)
    print(f"  ✅ Written {binary_path}: {len(merged):,} seeds")

    return len(merged)

# ============================================================
# MAIN
# ============================================================
print("=" * 60)
print("S165 Survivor Recovery Script")
print("=" * 60)

all_new_survivors = []

for pair in TRIAL_PAIRS:
    print(f"\n--- {pair['label']} ---")

    if not os.path.exists(pair['forward']):
        print(f"  SKIP — forward file missing: {pair['forward']}")
        continue
    if not os.path.exists(pair['reverse']):
        print(f"  SKIP — reverse file missing: {pair['reverse']}")
        continue

    fwd_map, fwd_count = load_survivors(pair['forward'])
    rev_map, rev_count = load_survivors(pair['reverse'])

    if not fwd_map or not rev_map:
        print(f"  SKIP — one or both passes have 0 survivors")
        continue

    # Compute bidirectional intersection
    fwd_set = set(fwd_map.keys())
    rev_set = set(rev_map.keys())
    bidi = fwd_set & rev_set

    print(f"  Forward: {len(fwd_map):,} | Reverse: {len(rev_map):,} | Bidirectional: {len(bidi):,}")

    for seed in bidi:
        fwd_rate = fwd_map[seed]
        rev_rate = rev_map[seed]
        all_new_survivors.append({
            'seed': seed,
            'forward_match_rate': fwd_rate,
            'reverse_match_rate': rev_rate,
            'score': (fwd_rate + rev_rate) / 2.0,
            'window_size': pair['window_size'],
            'offset': pair['offset'],
            'trial_number': pair['trial_number'],
        })

# Deduplicate across passes (highest score wins)
print(f"\n--- Deduplication ---")
seen = {}
for s in all_new_survivors:
    seed = s['seed']
    if seed not in seen or s['score'] > seen[seed]['score']:
        seen[seed] = s

deduped = list(seen.values())
print(f"  Total before dedup: {len(all_new_survivors):,}")
print(f"  Total after dedup:  {len(deduped):,}")

# Merge into NPZ
print(f"\n--- NPZ Merge ---")
total = merge_into_npz(deduped, ACCUM_NPZ, BINARY_NPZ)

print(f"\n{'=' * 60}")
print(f"Recovery complete. Accumulator now has {total:,} seeds.")
print(f"{'=' * 60}")
