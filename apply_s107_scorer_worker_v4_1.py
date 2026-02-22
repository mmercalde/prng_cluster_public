#!/usr/bin/env python3
"""
apply_s107_scorer_worker_v4_1.py
=================================
Patches scorer_trial_worker.py v4.0 -> v4.1
Session  : S107
Date     : 2026-02-22
Expected lines after patch: ~480

PATCHES (7 static + 2 dynamic):
  P1  globals block      -- add npz_selectivity, npz_trial_number, sel_global_*
  P2  load_data body     -- load selectivity/trial_number, compute global stats
  P3  load_data sanity   -- guard for new fields post-load
  P4  load_data return   -- 5-tuple -> 7-tuple
  P5  version header     -- v4.0 -> v4.1
  P6  main() load_data   -- unpack 7-tuple
  P7  main() run_trial   -- pass sel + tn
  P8  run_trial()        -- full replacement via def boundary (dynamic)
  P9  _log_trial_metrics -- insertion after run_trial (dynamic)

TB ISSUES ADDRESSED:
  Bug 1: npz_selectivity + npz_trial_number defined + loaded (P1, P2, P3)
  Bug 2: MIN_KEEP_FRAC guard enforced in run_trial
  Bug 3: full-length scores array Option B
  Tweak 4: global stats computed live at load time (P2)
  Tweak 5: percentile-rank normalization vs global dist
  Tweak 6: max_offset bounded vs modulus per lane
  Tweak 7: temporal_window_size -> coverage bonus via trial_number
  Tweak 8: size_penalty capped at 5.0
"""

import sys
import os
import re
import ast
import shutil
from datetime import datetime

TARGET = 'scorer_trial_worker.py'
BACKUP = f'scorer_trial_worker.py.bak_s107_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

# ===========================================================================
# PATCH DEFINITIONS
# ===========================================================================

P1_OLD = """\
# Global data cache (loaded once per worker)
# =============================================================================
survivors = None
seeds_to_score = None
npz_forward_matches = None   # float32 ndarray -- quality signal from NPZ
npz_reverse_matches = None   # float32 ndarray -- quality signal from NPZ"""

P1_NEW = """\
# Global data cache (loaded once per worker)
# =============================================================================
survivors = None
seeds_to_score = None
npz_forward_matches = None   # float32 ndarray -- quality signal from NPZ
npz_reverse_matches = None   # float32 ndarray -- quality signal from NPZ
npz_selectivity     = None   # float32 ndarray -- bidirectional_selectivity (v4.1)
npz_trial_number    = None   # int32   ndarray -- trial_number per seed (v4.1)

# v4.1: Global selectivity stats -- computed once at load time, never hardcoded
sel_global_min    = None
sel_global_median = None
sel_global_iqr    = None
sel_global_max    = None"""

# ---------------------------------------------------------------------------
P2_OLD = """\
            else:
                raise RuntimeError(
                    'NPZ missing forward_matches or reverse_matches. '
                    'Re-run convert_survivors_to_binary.py with NPZ v3.0+ format. '
                    f'Available NPZ keys: {list(survivors.keys())}'
                )"""

P2_NEW = """\
            else:
                raise RuntimeError(
                    'NPZ missing forward_matches or reverse_matches. '
                    'Re-run convert_survivors_to_binary.py with NPZ v3.0+ format. '
                    f'Available NPZ keys: {list(survivors.keys())}'
                )

            # v4.1: load selectivity + trial_number for subset-selection objective
            global npz_selectivity, npz_trial_number
            global sel_global_min, sel_global_median, sel_global_iqr, sel_global_max
            if 'bidirectional_selectivity' in survivors:
                npz_selectivity   = survivors['bidirectional_selectivity'].astype(_np.float32)
                sel_global_min    = float(npz_selectivity.min())
                sel_global_max    = float(npz_selectivity.max())
                sel_global_median = float(_np.median(npz_selectivity))
                sel_global_iqr    = float(
                    _np.percentile(npz_selectivity, 75) - _np.percentile(npz_selectivity, 25)
                )
                logger.info(
                    f'NPZ selectivity: min={sel_global_min:.4f}  '
                    f'median={sel_global_median:.4f}  '
                    f'iqr={sel_global_iqr:.4f}  '
                    f'max={sel_global_max:.4f}'
                )
            else:
                logger.warning(
                    'NPZ missing bidirectional_selectivity -- using ones fallback'
                )
                npz_selectivity   = _np.ones(len(seeds_to_score), dtype=_np.float32)
                sel_global_min    = 1.0
                sel_global_median = 1.0
                sel_global_iqr    = 0.1
                sel_global_max    = 1.0

            if 'trial_number' in survivors:
                npz_trial_number = survivors['trial_number'].astype(_np.int32)
                logger.info(
                    f'NPZ trial_number: min={npz_trial_number.min()}  '
                    f'max={npz_trial_number.max()}  '
                    f'unique={len(_np.unique(npz_trial_number))}'
                )
            else:
                logger.warning('NPZ missing trial_number -- coverage bonus disabled')
                npz_trial_number = _np.zeros(len(seeds_to_score), dtype=_np.int32)"""

# ---------------------------------------------------------------------------
P3_OLD = """\
    if npz_forward_matches is None or npz_reverse_matches is None:
        raise RuntimeError('NPZ quality signals are None after load -- cannot compute WSI.')"""

P3_NEW = """\
    if npz_forward_matches is None or npz_reverse_matches is None:
        raise RuntimeError('NPZ quality signals are None after load -- cannot compute WSI.')
    if npz_selectivity is None or npz_trial_number is None:
        raise RuntimeError('NPZ selectivity/trial_number are None after load -- v4.1 cannot run.')"""

# ---------------------------------------------------------------------------
P4_OLD = \
    "    return seeds_to_score, npz_forward_matches, npz_reverse_matches, prng_type, mod"

P4_NEW = \
    "    return seeds_to_score, npz_forward_matches, npz_reverse_matches, " \
    "npz_selectivity, npz_trial_number, prng_type, mod"

# ---------------------------------------------------------------------------
P5_OLD = "scorer_trial_worker.py (v4.0 - WSI objective, draw-history-free)"
P5_NEW = "scorer_trial_worker.py (v4.1 - Subset-Selection objective, TB-approved S107)"

# ---------------------------------------------------------------------------
P6_OLD = """\
        seeds, fwd_matches, rev_matches, prng_type, mod = load_data(
            survivors_file, train_history_file, holdout_history_file
        )"""

P6_NEW = """\
        seeds, fwd_matches, rev_matches, sel, tn, prng_type, mod = load_data(
            survivors_file, train_history_file, holdout_history_file
        )"""

# ---------------------------------------------------------------------------
P7_OLD = """\
        accuracy, scores = run_trial(
            seeds, fwd_matches, rev_matches, params,
            prng_type=prng_type, mod=mod, trial=trial,
            use_legacy_scoring=use_legacy_scoring
        )"""

P7_NEW = """\
        accuracy, scores = run_trial(
            seeds, fwd_matches, rev_matches, sel, tn, params,
            prng_type=prng_type, mod=mod, trial=trial,
            use_legacy_scoring=use_legacy_scoring
        )"""

# ---------------------------------------------------------------------------
# P8: run_trial() -- full replacement (dynamic via def boundary)
RUN_TRIAL_NEW = '''\
def run_trial(seeds_to_score,
              npz_forward_matches,
              npz_reverse_matches,
              npz_selectivity,
              npz_trial_number,
              params,
              prng_type='java_lcg',
              mod=1000,
              trial=None,
              use_legacy_scoring=False):
    """
    v4.1: Subset-Selection Objective (TB ruling S107) -- draw-history-free.
    Last modified : 2026-02-22
    Session       : S107
    Expected lines: ~145

    ARCHITECTURE:
        params define a residue FILTER (k-of-3 mask) over survivor seeds.
        Objective measures QUALITY of the selected subset using
        bidirectional_selectivity -- independent of the filter criterion.
        Optuna landscape: different params -> different subsets -> different
        percentile-rank selectivity -> real gradient to follow.

    PARAMS:
        rm1, rm2, rm3          -> residue moduli for k-of-3 mask
        max_offset             -> offset tolerance (bounded vs modulus per lane)
        temporal_window_size   -> controls trial_number coverage bonus weight

    OBJECTIVE (all 8 TB issues addressed):
        mask     = seeds where vote_count(m1+m2+m3) >= 2  (k-of-3)
        sel_score= P(global_selectivity < subset_mean_selectivity)
        bal      = 1 - abs(mean(fwd[mask]) - mean(rev[mask]))
        coverage = unique_trials[mask] / unique_trials[sample]
        tw_weight= clip(temporal_window_size/1000, 0.05, 0.20)
        keep     = mask.sum() / N
        size_pen = min(|log(keep/0.10)|, 5.0)
        objective= clip(sel_score*(0.75+0.25*bal)+tw_weight*coverage
                        -0.30*size_pen, -1, 1)

    PRESERVED:
        S101: random.seed(optuna_trial_number) per-trial unique sampling
        WATCHER CLI: positional args 2+3 in main() accepted, ignored
    """
    import numpy as np
    import random
    import math

    EPS            = 1e-9
    TARGET_KEEP    = 0.10
    MIN_KEEP_FRAC  = 0.01
    MAX_KEEP_FRAC  = 0.40
    MIN_KEEP_COUNT = 10
    LAMBDA_SIZE    = 0.30
    SIZE_PEN_CAP   = 5.0

    rm1         = int(params.get('residue_mod_1',   10))
    rm2         = int(params.get('residue_mod_2',  100))
    rm3         = int(params.get('residue_mod_3', 1000))
    max_offset  = int(params.get('max_offset',       5))
    tw_size     = int(params.get('temporal_window_size', 100))
    trial_num   = int(params.get('optuna_trial_number',   0))
    sample_size = int(params.get('sample_size', 50000))

    n_seeds = len(seeds_to_score)

    # S101: per-trial unique sampling
    if n_seeds > sample_size:
        random.seed(trial_num)
        sample_idx = np.array(
            random.sample(range(n_seeds), sample_size), dtype=np.int64
        )
        seeds_arr = np.array(seeds_to_score, dtype=np.int64)[sample_idx]
        sel_arr   = npz_selectivity[sample_idx]
        fwd_arr   = npz_forward_matches[sample_idx]
        rev_arr   = npz_reverse_matches[sample_idx]
        tn_arr    = npz_trial_number[sample_idx]
        logger.info(f'Sampled {sample_size:,} / {n_seeds:,} seeds (rng_seed={trial_num})')
    else:
        sample_idx = None
        seeds_arr  = np.array(seeds_to_score, dtype=np.int64)
        sel_arr    = npz_selectivity
        fwd_arr    = npz_forward_matches
        rev_arr    = npz_reverse_matches
        tn_arr     = npz_trial_number

    N = len(seeds_arr)

    # Tweak 6: bound offset vs modulus so filter always has teeth
    off1 = max(1, min(max_offset, max(rm1 - 1, 1)))
    off2 = max(1, min(max_offset, max(rm2 - 1, 1)))
    off3 = max(1, min(max_offset, max(rm3 - 1, 1)))

    # k-of-3 mask
    m1 = (seeds_arr % max(rm1, 1)) < off1
    m2 = (seeds_arr % max(rm2, 1)) < off2
    m3 = (seeds_arr % max(rm3, 1)) < off3
    vote_count = m1.astype(np.int32) + m2.astype(np.int32) + m3.astype(np.int32)
    mask = vote_count >= 2

    subset_n = int(mask.sum())
    keep     = subset_n / max(N, 1)

    logger.info(
        f'Mask: rm=({rm1},{rm2},{rm3}) off=({off1},{off2},{off3}) '
        f'subset_n={subset_n} keep={keep:.4f} ({keep*100:.1f}%)'
    )

    # Degenerate guards (all three bands enforced)
    def _reject(reason):
        logger.warning(f'Rejected: {reason} subset_n={subset_n} keep={keep:.4f} -> -1.0')
        _log_trial_metrics(trial_num, params, subset_n, keep,
                           objective=-1.0, reason=reason)
        full = np.zeros(n_seeds, dtype=np.float32)
        return -1.0, full.tolist()

    if subset_n < MIN_KEEP_COUNT:
        return _reject('too_small')
    if keep < MIN_KEEP_FRAC:
        return _reject('keep_too_low')
    if keep > MAX_KEEP_FRAC:
        return _reject('keep_too_high')

    # Tweak 5: percentile-rank vs GLOBAL distribution (stable across trials)
    sel_subset = sel_arr[mask]
    sel_mean   = float(sel_subset.mean())
    sel_p25    = float(np.percentile(sel_subset, 25))
    sel_p75    = float(np.percentile(sel_subset, 75))
    sel_std    = float(sel_subset.std())
    sel_score  = float(np.mean(npz_selectivity < sel_mean))  # global percentile

    # Balance bonus
    fwd_mean = float(fwd_arr[mask].mean())
    rev_mean = float(rev_arr[mask].mean())
    bal      = float(np.clip(1.0 - abs(fwd_mean - rev_mean), 0.0, 1.0))

    # Tweak 7: temporal coverage via trial_number
    uniq_total = max(len(np.unique(tn_arr)), 1)
    uniq_sel   = len(np.unique(tn_arr[mask]))
    coverage   = uniq_sel / uniq_total
    tw_weight  = float(np.clip(tw_size / 1000.0, 0.05, 0.20))

    # Tweak 8: size penalty, capped
    size_penalty = min(
        abs(math.log((keep + EPS) / TARGET_KEEP)),
        SIZE_PEN_CAP
    )

    # Composite objective
    objective = (
        sel_score * (0.75 + 0.25 * bal)
        + tw_weight * coverage
        - LAMBDA_SIZE * size_penalty
    )
    objective = float(np.clip(objective, -1.0, 1.0))

    logger.info(
        f'Objective={objective:.6f}  sel_score={sel_score:.4f}  '
        f'bal={bal:.4f}  coverage={coverage:.4f}  tw_weight={tw_weight:.3f}  '
        f'size_pen={size_penalty:.4f}'
    )

    _log_trial_metrics(
        trial_num, params, subset_n, keep,
        sel_mean=sel_mean, sel_p25=sel_p25, sel_p75=sel_p75, sel_std=sel_std,
        fwd_mean=fwd_mean, rev_mean=rev_mean, bal=bal,
        coverage=coverage, tw_weight=tw_weight,
        size_penalty=size_penalty, objective=objective, reason='ok'
    )

    # Bug 3 fix Option B: full-length scores array
    full = np.zeros(n_seeds, dtype=np.float32)
    if sample_idx is not None:
        full[sample_idx] = mask.astype(np.float32)
    else:
        full[:] = mask.astype(np.float32)

    return objective, full.tolist()

'''

# ---------------------------------------------------------------------------
# P9: _log_trial_metrics() -- inserted immediately after run_trial()
LOG_METRICS_NEW = '''\
def _log_trial_metrics(trial_num, params, subset_n, keep,
                       sel_mean=None, sel_p25=None, sel_p75=None, sel_std=None,
                       fwd_mean=None, rev_mean=None, bal=None,
                       coverage=None, tw_weight=None,
                       size_penalty=None, objective=None, reason='ok'):
    """
    Per-trial diagnostic metrics (TB S107 requirement).
    Verifies Optuna landscape is non-flat: if objective is constant across
    trials, something is still tautological.
    Session: S107  Expected lines: ~25
    """
    metrics = {
        'trial_num'   : trial_num,
        'params'      : params,
        'subset_n'    : subset_n,
        'keep'        : round(keep, 6) if keep is not None else None,
        'sel_mean'    : round(sel_mean, 6) if sel_mean is not None else None,
        'sel_p25'     : round(sel_p25, 6) if sel_p25 is not None else None,
        'sel_p75'     : round(sel_p75, 6) if sel_p75 is not None else None,
        'sel_std'     : round(sel_std, 6) if sel_std is not None else None,
        'fwd_mean'    : round(fwd_mean, 6) if fwd_mean is not None else None,
        'rev_mean'    : round(rev_mean, 6) if rev_mean is not None else None,
        'bal'         : round(bal, 6) if bal is not None else None,
        'coverage'    : round(coverage, 6) if coverage is not None else None,
        'tw_weight'   : round(tw_weight, 6) if tw_weight is not None else None,
        'size_penalty': round(size_penalty, 6) if size_penalty is not None else None,
        'objective'   : round(objective, 6) if objective is not None else None,
        'reason'      : reason,
    }
    logger.info(f'[TRIAL_METRICS] {metrics}')

'''

# ===========================================================================
# PATCHER
# ===========================================================================

def extract_func_block(content, start_marker, end_marker):
    """Extract function block using top-level def boundaries."""
    si = content.find(start_marker)
    if si == -1:
        return None, None, None
    ei = content.find(end_marker, si + len(start_marker))
    if ei == -1:
        return None, None, None
    return content[si:ei], si, ei


def apply_patch(content, name, old, new):
    count = content.count(old)
    if count == 0:
        print(f'FAIL  {name}: not found')
        print(f'      First 80 chars of old: {repr(old[:80])}')
        return content, False
    if count > 1:
        print(f'FAIL  {name}: found {count}x (not unique)')
        return content, False
    result = content.replace(old, new, 1)
    print(f'OK    {name}')
    return result, True


def main():
    print('=' * 60)
    print('apply_s107_scorer_worker_v4_1.py')
    print(f'Target : {TARGET}')
    print('=' * 60)

    if not os.path.exists(TARGET):
        print(f'ERROR: {TARGET} not found in {os.getcwd()}')
        sys.exit(1)

    # Backup
    shutil.copy2(TARGET, BACKUP)
    print(f'Backup : {BACKUP}')

    with open(TARGET, 'r', encoding='utf-8') as f:
        content = f.read()

    import hashlib
    pre_md5 = hashlib.md5(content.encode()).hexdigest()
    print(f'Pre-MD5: {pre_md5}')
    print()

    failures = []

    # Static patches P1-P7
    for name, old, new in [
        ('P1 globals block',          P1_OLD, P1_NEW),
        ('P2 load_data body',         P2_OLD, P2_NEW),
        ('P3 load_data sanity guard', P3_OLD, P3_NEW),
        ('P4 load_data return',       P4_OLD, P4_NEW),
        ('P5 version header',         P5_OLD, P5_NEW),
        ('P6 main load_data unpack',  P6_OLD, P6_NEW),
        ('P7 main run_trial call',    P7_OLD, P7_NEW),
    ]:
        content, ok = apply_patch(content, name, old, new)
        if not ok:
            failures.append(name)

    # P8: dynamic run_trial() replacement via def boundary
    block, si, ei = extract_func_block(
        content,
        '\ndef run_trial(',
        '\ndef save_local_result('
    )
    if block is None:
        print('FAIL  P8 run_trial (dynamic): boundary markers not found')
        failures.append('P8 run_trial')
    else:
        content = content[:si] + '\n' + RUN_TRIAL_NEW + content[ei:]
        print(f'OK    P8 run_trial (dynamic): replaced {len(block)} chars')

    # P9: insert _log_trial_metrics after run_trial (before save_local_result)
    insert_marker = '\ndef save_local_result('
    pos = content.find(insert_marker)
    if pos == -1:
        print('FAIL  P9 _log_trial_metrics: save_local_result not found')
        failures.append('P9 _log_trial_metrics')
    else:
        content = content[:pos] + '\n' + LOG_METRICS_NEW + content[pos:]
        print(f'OK    P9 _log_trial_metrics: inserted before save_local_result')

    print()

    if failures:
        print(f'FAILED patches: {failures}')
        print('Restoring backup...')
        shutil.copy2(BACKUP, TARGET)
        sys.exit(1)

    # Write patched file
    with open(TARGET, 'w', encoding='utf-8') as f:
        f.write(content)

    post_md5 = hashlib.md5(content.encode()).hexdigest()
    print(f'Post-MD5 : {post_md5}')
    print(f'Lines    : {len(content.splitlines())}')
    print()

    # AST check
    try:
        ast.parse(content)
        print('AST      : OK')
    except SyntaxError as e:
        print(f'AST      : FAIL -- {e}')
        print('Restoring backup...')
        shutil.copy2(BACKUP, TARGET)
        sys.exit(1)

    # Post-patch verification
    print()
    print('=== Verification ===')
    checks = [
        ('v4.1 header present',        'v4.1 - Subset-Selection objective' in content),
        ('npz_selectivity global',      'npz_selectivity     = None' in content),
        ('npz_trial_number global',     'npz_trial_number    = None' in content),
        ('sel_global_min global',       'sel_global_min    = None' in content),
        ('selectivity loaded',          'bidirectional_selectivity' in content),
        ('trial_number loaded',         "'trial_number' in survivors" in content),
        ('sanity guard new fields',     'npz_selectivity is None' in content),
        ('7-tuple return',              'npz_selectivity, npz_trial_number, prng_type, mod' in content),
        ('7-tuple unpack in main',      'seeds, fwd_matches, rev_matches, sel, tn, prng_type, mod' in content),
        ('run_trial has sel param',     'npz_selectivity,\n              npz_trial_number,' in content),
        ('k-of-3 mask',                 'vote_count >= 2' in content),
        ('percentile-rank scoring',     'np.mean(npz_selectivity < sel_mean)' in content),
        ('coverage bonus',              'uniq_sel / uniq_total' in content),
        ('size_penalty cap',            'SIZE_PEN_CAP' in content),
        ('full-length scores array',    'full = np.zeros(n_seeds' in content),
        ('_log_trial_metrics defined',  'def _log_trial_metrics(' in content),
        ('TRIAL_METRICS log tag',       '[TRIAL_METRICS]' in content),
        ('no WSI reference',            'WSI' not in content.split('def run_trial')[1].split('def save_local_result')[0]),
        ('only 1 load_data call site',  content.count('= load_data(') == 1),
    ]

    all_ok = True
    for desc, result in checks:
        status = '✅' if result else '❌'
        print(f'  {status} {desc}')
        if not result:
            all_ok = False

    print()
    if all_ok:
        print('✅ All checks passed. scorer_trial_worker.py is v4.1 ready.')
        print()
        print('Next steps:')
        print('  python3 -c "import ast; ast.parse(open(\'scorer_trial_worker.py\').read()); print(\'AST OK\')"')
        print('  grep -c "npz_selectivity\\|npz_trial_number" scorer_trial_worker.py')
        print()
        print('Smoke test (3 param sets -- objective must VARY):')
        print('  for j in 0 1 2; do')
        print('    PYTHONPATH=. python3 scorer_trial_worker.py \\')
        print('      bidirectional_survivors_binary.npz /dev/null /dev/null $j \\')
        print('      --params-json "{\\"residue_mod_1\\":$((7+7*j)),\\"residue_mod_2\\":$((50+50*j)),\\"residue_mod_3\\":$((300+300*j)),\\"max_offset\\":$((2+2*j)),\\"temporal_window_size\\":$((50+50*j)),\\"optuna_trial_number\\":$j,\\"sample_size\\":500}" \\')
        print('      --gpu-id 0 | tail -n 1')
        print('  done')
        print()
        print('Deploy to rigs:')
        print('  scp scorer_trial_worker.py 192.168.3.120:~/distributed_prng_analysis/')
        print('  scp scorer_trial_worker.py 192.168.3.154:~/distributed_prng_analysis/')
        print('  md5sum scorer_trial_worker.py')
    else:
        print('❌ Some checks failed -- review above before running Step 2.')


if __name__ == '__main__':
    main()
