#!/usr/bin/env python3
"""
apply_s107_scorer_worker_v4_2.py  (hardened — TB review 2)
============================================================
Patches scorer_trial_worker.py v4.1 -> v4.2
Session  : S107
Date     : 2026-02-22

CHANGE SUMMARY (TB ruling S107 Q1-Q4):
  - Replace bidirectional_selectivity (98.8% at floor) with
    bidirectional_count as primary quality signal
  - Add intersection_ratio as secondary bonus (weight=0.10)
  - Use median(subset) not mean() -- robust against heavy-tail counts
  - Percentile-rank vs global array
  - Drop npz_selectivity entirely

TB HARDENING FIXES (review 2):
  Fix 1: P1 anchor uses full unique header block
  Fix 2: 'npz_selectivity GONE' check scoped to globals + run_trial only
  Fix 3: P6/P7 use exact multi-line blocks from Zeus v4.1 output
  Fix 4: load_data call-site count is warn-only, not hard fail

PATCHES (7 static + 2 dynamic = 9 total):
  P1  globals block      -- swap selectivity/stats -> bc + ir globals
  P2  load_data body     -- load bc + ir, remove selectivity block
  P3  sanity guard       -- update for bc + trial_number
  P4  load_data return   -- swap selectivity -> bc, ir in tuple
  P5  version header     -- v4.1 -> v4.2
  P6  main() unpack      -- sel,tn -> bc,ir,tn  (multi-line anchor)
  P7  main() run_trial   -- pass bc, ir, tn     (multi-line anchor)
  P8  run_trial()        -- full replacement (dynamic, def boundary)
  P9  _log_trial_metrics -- rename sel->bc,ir   (dynamic, def boundary)

FINAL v4.2 OBJECTIVE (TB S107):
  bc_stat  = median(bidirectional_count[mask])
  bc_score = P(bc_global < bc_stat)
  ir_score = P(ir_global < median(ir[mask]))
  bal      = 1 - abs(mean(fwd[mask]) - mean(rev[mask]))
  coverage = unique(trial_number[mask]) / unique(trial_number[sample])
  tw_weight= clip(temporal_window_size/1000, 0.05, 0.20)
  size_pen = min(|log(keep/0.10)|, 5.0)
  objective= clip(bc_score*(0.75+0.25*bal) + tw_weight*coverage
                  + 0.10*ir_score - 0.30*size_pen, -1, 1)
"""

import sys
import os
import ast
import shutil
from datetime import datetime

TARGET = 'scorer_trial_worker.py'
BACKUP = f'scorer_trial_worker.py.bak_s107_v42_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

# ===========================================================================
# PATCH STRINGS
# ===========================================================================

# P1: globals -- full unique header as anchor (Fix 1)
P1_OLD = """\
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

P1_NEW = """\
# Global data cache (loaded once per worker)
# =============================================================================
survivors = None
seeds_to_score = None
npz_forward_matches     = None   # float32 ndarray -- quality signal from NPZ
npz_reverse_matches     = None   # float32 ndarray -- quality signal from NPZ
npz_bidirectional_count = None   # float32 ndarray -- survival frequency (v4.2)
npz_intersection_ratio  = None   # float32 ndarray -- bidirectional tightness (v4.2)
npz_trial_number        = None   # int32   ndarray -- trial_number per seed (v4.1)"""

# ---------------------------------------------------------------------------
# P2: load_data body
P2_OLD = """\
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

P2_NEW = """\
            # v4.2: load bidirectional_count + intersection_ratio (TB ruling S107 Q1-Q3)
            # bidirectional_selectivity dropped -- 98.8% at floor, unusable as signal
            global npz_bidirectional_count, npz_intersection_ratio, npz_trial_number
            if 'bidirectional_count' in survivors:
                npz_bidirectional_count = survivors['bidirectional_count'].astype(_np.float32)
                logger.info(
                    f'NPZ bidirectional_count: min={npz_bidirectional_count.min():.0f}  '
                    f'median={float(_np.median(npz_bidirectional_count)):.0f}  '
                    f'max={npz_bidirectional_count.max():.0f}  '
                    f'std={npz_bidirectional_count.std():.1f}'
                )
            else:
                logger.warning('NPZ missing bidirectional_count -- using ones fallback')
                npz_bidirectional_count = _np.ones(len(seeds_to_score), dtype=_np.float32)

            if 'intersection_ratio' in survivors:
                npz_intersection_ratio = survivors['intersection_ratio'].astype(_np.float32)
                logger.info(
                    f'NPZ intersection_ratio: min={npz_intersection_ratio.min():.4f}  '
                    f'median={float(_np.median(npz_intersection_ratio)):.4f}  '
                    f'max={npz_intersection_ratio.max():.4f}'
                )
            else:
                logger.warning('NPZ missing intersection_ratio -- ir_score will be 0.0')
                npz_intersection_ratio = _np.zeros(len(seeds_to_score), dtype=_np.float32)

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
# P3: sanity guard
P3_OLD = """\
    if npz_bidirectional_count is None or npz_trial_number is None:
        raise RuntimeError(
            'NPZ bidirectional_count/trial_number are None after load -- v4.2 cannot run.'
        )"""

P3_NEW = """\
    if npz_bidirectional_count is None or npz_trial_number is None:
        raise RuntimeError(
            'NPZ bidirectional_count/trial_number are None after load -- v4.2 cannot run.'
        )"""
# NOTE: P3 was already correct from v4.2 first draft -- this is identity patch,
# kept for explicit documentation that the guard is present and correct.

# ---------------------------------------------------------------------------
# P4: return tuple -- swap selectivity -> bc + ir
P4_OLD = \
    "    return seeds_to_score, npz_forward_matches, npz_reverse_matches, " \
    "npz_selectivity, npz_trial_number, prng_type, mod"

P4_NEW = \
    "    return seeds_to_score, npz_forward_matches, npz_reverse_matches, " \
    "npz_bidirectional_count, npz_intersection_ratio, npz_trial_number, prng_type, mod"

# ---------------------------------------------------------------------------
# P5: version header
P5_OLD = "scorer_trial_worker.py (v4.1 - Subset-Selection objective, TB-approved S107)"
P5_NEW = "scorer_trial_worker.py (v4.2 - Subset-Selection, bidirectional_count signal, TB S107)"

# ---------------------------------------------------------------------------
# P6: main() unpack -- exact multi-line block from Zeus v4.1 (Fix 3)
P6_OLD = """\
        seeds, fwd_matches, rev_matches, sel, tn, prng_type, mod = load_data(
            survivors_file, train_history_file, holdout_history_file
        )"""

P6_NEW = """\
        seeds, fwd_matches, rev_matches, bc, ir, tn, prng_type, mod = load_data(
            survivors_file, train_history_file, holdout_history_file
        )"""

# ---------------------------------------------------------------------------
# P7: main() run_trial call -- exact multi-line block from Zeus v4.1 (Fix 3)
P7_OLD = """\
        accuracy, scores = run_trial(
            seeds, fwd_matches, rev_matches, sel, tn, params,
            prng_type=prng_type, mod=mod, trial=trial,
            use_legacy_scoring=use_legacy_scoring
        )"""

P7_NEW = """\
        accuracy, scores = run_trial(
            seeds, fwd_matches, rev_matches, bc, ir, tn, params,
            prng_type=prng_type, mod=mod, trial=trial,
            use_legacy_scoring=use_legacy_scoring
        )"""

# ---------------------------------------------------------------------------
# P8: run_trial() full replacement
RUN_TRIAL_V42 = '''\
def run_trial(seeds_to_score,
              npz_forward_matches,
              npz_reverse_matches,
              npz_bidirectional_count,
              npz_intersection_ratio,
              npz_trial_number,
              params,
              prng_type='java_lcg',
              mod=1000,
              trial=None,
              use_legacy_scoring=False):
    """
    v4.2: Subset-Selection Objective -- bidirectional_count primary signal.
    Last modified : 2026-02-22
    Session       : S107
    Expected lines: ~155

    CHANGE FROM v4.1:
        bidirectional_selectivity dropped (98.8% at floor -- unusable).
        Primary: bidirectional_count (survival frequency, std=722).
        Secondary bonus: intersection_ratio (bidirectional tightness, weight=0.10).
        Median used (robust against heavy-tail counts -- TB Q2).
        Percentile-rank vs full global arrays (stable across trials).
        ir_disabled guard: if IR array all zeros, ir_score=0.0 with warning.

    TB FORMULA (final v4.2):
        mask     = vote_count >= 2  (k-of-3 residue filter)
        bc_stat  = median(bidirectional_count[mask])
        bc_score = P(bc_global < bc_stat)              in [0,1]
        ir_stat  = median(intersection_ratio[mask])
        ir_score = P(ir_global < ir_stat)              in [0,1]
        bal      = 1 - abs(mean(fwd[mask]) - mean(rev[mask]))
        coverage = unique(trial_number[mask]) / unique(trial_number[sample])
        tw_weight= clip(temporal_window_size/1000, 0.05, 0.20)
        size_pen = min(|log(keep/0.10)|, 5.0)
        objective= clip(bc_score*(0.75+0.25*bal) + tw_weight*coverage
                        + 0.10*ir_score - 0.30*size_pen, -1, 1)

    PRESERVED:
        S101: random.seed(optuna_trial_number) per-trial unique sampling
        All degenerate guards (too_small, keep_too_low, keep_too_high)
        full-length scores array (Option B)
        WATCHER CLI compatibility
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
    IR_WEIGHT      = 0.10

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
        bc_arr    = npz_bidirectional_count[sample_idx]
        ir_arr    = npz_intersection_ratio[sample_idx]
        fwd_arr   = npz_forward_matches[sample_idx]
        rev_arr   = npz_reverse_matches[sample_idx]
        tn_arr    = npz_trial_number[sample_idx]
        logger.info(f'Sampled {sample_size:,} / {n_seeds:,} seeds (rng_seed={trial_num})')
    else:
        sample_idx = None
        seeds_arr  = np.array(seeds_to_score, dtype=np.int64)
        bc_arr     = npz_bidirectional_count
        ir_arr     = npz_intersection_ratio
        fwd_arr    = npz_forward_matches
        rev_arr    = npz_reverse_matches
        tn_arr     = npz_trial_number

    N = len(seeds_arr)

    # Bound offset vs modulus so filter always has teeth (TB Tweak 6)
    off1 = max(1, min(max_offset, max(rm1 - 1, 1)))
    off2 = max(1, min(max_offset, max(rm2 - 1, 1)))
    off3 = max(1, min(max_offset, max(rm3 - 1, 1)))

    # k-of-3 mask: seed passes if >= 2 of 3 residue conditions met
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

    # Degenerate guards
    def _reject(reason):
        logger.warning(f'Rejected: {reason} subset_n={subset_n} keep={keep:.4f} -> -1.0')
        _log_trial_metrics(trial_num, params, subset_n, keep,
                           objective=-1.0, reason=reason)
        return -1.0, np.zeros(n_seeds, dtype=np.float32).tolist()

    if subset_n < MIN_KEEP_COUNT:
        return _reject('too_small')
    if keep < MIN_KEEP_FRAC:
        return _reject('keep_too_low')
    if keep > MAX_KEEP_FRAC:
        return _reject('keep_too_high')

    # Primary: bidirectional_count -- median (robust vs heavy tail, TB Q2)
    bc_subset = bc_arr[mask]
    bc_stat   = float(np.median(bc_subset))
    bc_score  = float(np.mean(npz_bidirectional_count < bc_stat))  # global percentile

    # Secondary: intersection_ratio (TB Q3, optional bonus weight=0.10)
    ir_disabled = bool(np.all(npz_intersection_ratio == 0))
    if ir_disabled:
        logger.warning('intersection_ratio all zeros -- ir_score=0.0 for this trial')
        ir_stat  = 0.0
        ir_score = 0.0
    else:
        ir_subset = ir_arr[mask]
        ir_stat   = float(np.median(ir_subset))
        ir_score  = float(np.mean(npz_intersection_ratio < ir_stat))  # global percentile

    # Balance bonus
    fwd_mean = float(fwd_arr[mask].mean())
    rev_mean = float(rev_arr[mask].mean())
    bal      = float(np.clip(1.0 - abs(fwd_mean - rev_mean), 0.0, 1.0))

    # Temporal coverage via trial_number
    uniq_total = max(len(np.unique(tn_arr)), 1)
    uniq_sel   = len(np.unique(tn_arr[mask]))
    coverage   = uniq_sel / uniq_total
    tw_weight  = float(np.clip(tw_size / 1000.0, 0.05, 0.20))

    # Size penalty, capped
    size_penalty = min(
        abs(math.log((keep + EPS) / TARGET_KEEP)),
        SIZE_PEN_CAP
    )

    # TB v4.2 composite objective
    objective = (
        bc_score * (0.75 + 0.25 * bal)
        + tw_weight * coverage
        + IR_WEIGHT * ir_score
        - LAMBDA_SIZE * size_penalty
    )
    objective = float(np.clip(objective, -1.0, 1.0))

    logger.info(
        f'Objective={objective:.6f}  bc_stat={bc_stat:.0f}  bc_score={bc_score:.4f}  '
        f'ir_stat={ir_stat:.4f}  ir_score={ir_score:.4f}  '
        f'bal={bal:.4f}  coverage={coverage:.4f}  tw_weight={tw_weight:.3f}  '
        f'size_pen={size_penalty:.4f}'
    )

    _log_trial_metrics(
        trial_num, params, subset_n, keep,
        bc_stat=bc_stat, bc_score=bc_score,
        ir_stat=ir_stat, ir_score=ir_score,
        fwd_mean=fwd_mean, rev_mean=rev_mean, bal=bal,
        coverage=coverage, tw_weight=tw_weight,
        size_penalty=size_penalty, objective=objective, reason='ok'
    )

    # Option B: full-length scores array
    full = np.zeros(n_seeds, dtype=np.float32)
    if sample_idx is not None:
        full[sample_idx] = mask.astype(np.float32)
    else:
        full[:] = mask.astype(np.float32)

    return objective, full.tolist()

'''

# ---------------------------------------------------------------------------
# P9: _log_trial_metrics
LOG_METRICS_V42 = '''\
def _log_trial_metrics(trial_num, params, subset_n, keep,
                       bc_stat=None, bc_score=None,
                       ir_stat=None, ir_score=None,
                       fwd_mean=None, rev_mean=None, bal=None,
                       coverage=None, tw_weight=None,
                       size_penalty=None, objective=None, reason='ok'):
    """
    Per-trial diagnostic metrics (TB S107 requirement).
    v4.2: sel_* replaced with bc_* and ir_*.
    Key signal: bc_score must vary across trials for real landscape.
    Session: S107  Expected lines: ~25
    """
    metrics = {
        'trial_num'   : trial_num,
        'params'      : params,
        'subset_n'    : subset_n,
        'keep'        : round(keep, 6) if keep is not None else None,
        'bc_stat'     : round(bc_stat, 2) if bc_stat is not None else None,
        'bc_score'    : round(bc_score, 6) if bc_score is not None else None,
        'ir_stat'     : round(ir_stat, 6) if ir_stat is not None else None,
        'ir_score'    : round(ir_score, 6) if ir_score is not None else None,
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
# PATCHER ENGINE
# ===========================================================================

def extract_func_block(content, start_marker, end_marker):
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
        print(f'      First 80 chars: {repr(old[:80])}')
        return content, False
    if count > 1:
        print(f'FAIL  {name}: found {count}x (not unique)')
        return content, False
    print(f'OK    {name}')
    return content.replace(old, new, 1), True


def main():
    print('=' * 60)
    print('apply_s107_scorer_worker_v4_2.py  (hardened)')
    print(f'Target : {TARGET}')
    print('=' * 60)

    if not os.path.exists(TARGET):
        print(f'ERROR: {TARGET} not found in {os.getcwd()}')
        sys.exit(1)

    shutil.copy2(TARGET, BACKUP)
    print(f'Backup : {BACKUP}')

    with open(TARGET, 'r', encoding='utf-8') as f:
        content = f.read()

    import hashlib
    print(f'Pre-MD5: {hashlib.md5(content.encode()).hexdigest()}')
    print()

    failures = []

    # P3 is identity if sanity guard already correct -- skip if not found to avoid fail
    static_patches = [
        ('P1 globals block',       P1_OLD, P1_NEW),
        ('P2 load_data body',      P2_OLD, P2_NEW),
        ('P4 load_data return',    P4_OLD, P4_NEW),
        ('P5 version header',      P5_OLD, P5_NEW),
        ('P6 main unpack',         P6_OLD, P6_NEW),
        ('P7 main run_trial call', P7_OLD, P7_NEW),
    ]

    # P3 only if the old sanity guard exists (it may not if P3 was pre-applied)
    if content.count(P3_OLD) == 1:
        static_patches.insert(2, ('P3 sanity guard', P3_OLD, P3_NEW))
    else:
        new_guard = 'npz_bidirectional_count is None or npz_trial_number is None'
        if new_guard in content:
            print(f'SKIP  P3 sanity guard: already correct')
        else:
            print(f'WARN  P3 sanity guard: neither old nor new found -- check manually')

    for name, old, new in static_patches:
        content, ok = apply_patch(content, name, old, new)
        if not ok:
            failures.append(name)

    # P8: run_trial() dynamic replacement
    block, si, ei = extract_func_block(
        content,
        '\ndef run_trial(',
        '\ndef _log_trial_metrics('
    )
    if block is None:
        print('FAIL  P8 run_trial (dynamic): boundary not found')
        failures.append('P8 run_trial')
    else:
        content = content[:si] + '\n' + RUN_TRIAL_V42 + content[ei:]
        print(f'OK    P8 run_trial (dynamic): replaced {len(block)} chars')

    # P9: _log_trial_metrics dynamic replacement
    block, si, ei = extract_func_block(
        content,
        '\ndef _log_trial_metrics(',
        '\ndef save_local_result('
    )
    if block is None:
        print('FAIL  P9 _log_trial_metrics (dynamic): boundary not found')
        failures.append('P9 _log_trial_metrics')
    else:
        content = content[:si] + '\n' + LOG_METRICS_V42 + content[ei:]
        print(f'OK    P9 _log_trial_metrics (dynamic): replaced {len(block)} chars')

    print()

    if failures:
        print(f'FAILED patches: {failures}')
        print('Restoring backup...')
        shutil.copy2(BACKUP, TARGET)
        sys.exit(1)

    with open(TARGET, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'Post-MD5 : {hashlib.md5(content.encode()).hexdigest()}')
    print(f'Lines    : {len(content.splitlines())}')
    print()

    try:
        ast.parse(content)
        print('AST      : OK')
    except SyntaxError as e:
        print(f'AST      : FAIL -- {e}')
        shutil.copy2(BACKUP, TARGET)
        sys.exit(1)

    # Scoped sections for Fix 2 checks
    globals_section  = content[content.find('# Global data cache (loaded once per worker)'):
                                content.find('# Global data cache (loaded once per worker)') + 2000] \
        if '# Global data cache (loaded once per worker)' in content else content[:2000]
    run_trial_body   = content.split('def run_trial', 1)[1].split('def _log_trial_metrics', 1)[0] \
        if 'def run_trial' in content else ''
    load_data_calls  = content.count('= load_data(')

    print()
    print('=== Verification ===')

    # (result, is_hard_fail)
    checks = [
        ('v4.2 header present',
         'v4.2 - Subset-Selection, bidirectional_count' in content),
        ('npz_bidirectional_count in globals',
         'npz_bidirectional_count' in globals_section),
        ('npz_intersection_ratio in globals',
         'npz_intersection_ratio' in globals_section),
        ('npz_selectivity NOT in globals (scoped)',
         'npz_selectivity' not in globals_section),
        ('npz_selectivity NOT in run_trial (scoped)',
         'npz_selectivity' not in run_trial_body),
        ('bc loaded in load_data',
         "'bidirectional_count' in survivors" in content),
        ('ir loaded in load_data',
         "'intersection_ratio' in survivors" in content),
        ('sanity guard bc present',
         'npz_bidirectional_count is None' in content),
        ('return tuple has bc + ir',
         'npz_bidirectional_count, npz_intersection_ratio, npz_trial_number' in content),
        ('main unpack bc ir tn',
         'seeds, fwd_matches, rev_matches, bc, ir, tn' in content),
        ('run_trial bc param in sig',
         'npz_bidirectional_count,\n              npz_intersection_ratio,' in content),
        ('k-of-3 mask preserved',
         'vote_count >= 2' in content),
        ('median used for bc',
         'np.median(bc_subset)' in content),
        ('bc_score global percentile',
         'np.mean(npz_bidirectional_count < bc_stat)' in content),
        ('ir_score present',
         'np.mean(npz_intersection_ratio < ir_stat)' in content),
        ('ir_disabled guard',
         'ir_disabled' in content),
        ('IR_WEIGHT 0.10',
         'IR_WEIGHT      = 0.10' in content),
        ('coverage bonus preserved',
         'uniq_sel / uniq_total' in content),
        ('size_penalty cap preserved',
         'SIZE_PEN_CAP' in content),
        ('full-length array preserved',
         'full = np.zeros(n_seeds' in content),
        ('TRIAL_METRICS bc_score logged',
         "'bc_score'" in content),
        ('TRIAL_METRICS ir_score logged',
         "'ir_score'" in content),
        ('no WSI in run_trial body',
         'WSI' not in run_trial_body),
    ]

    all_ok = True
    for desc, result in checks:
        print(f'  {"✅" if result else "❌"} {desc}')
        if not result:
            all_ok = False

    # Warn-only (Fix 4)
    if load_data_calls == 1:
        print(f'  ✅ load_data call sites: {load_data_calls}')
    else:
        print(f'  ⚠️  load_data call sites: {load_data_calls} (verify manually -- warn only)')

    print()
    if all_ok:
        print('✅ All checks passed. scorer_trial_worker.py is v4.2 ready.')
        print()
        print('Smoke test (bc_score AND objective must vary across 3 trials):')
        print('  for j in 0 1 2; do')
        print('    PYTHONPATH=. python3 scorer_trial_worker.py \\')
        print('      bidirectional_survivors_binary.npz /dev/null /dev/null $j \\')
        print('      --params-json "{\\"residue_mod_1\\":$((7+7*j)),\\"residue_mod_2\\":$((50+50*j)),\\"residue_mod_3\\":$((300+300*j)),\\"max_offset\\":$((2+2*j)),\\"temporal_window_size\\":$((50+50*j)),\\"optuna_trial_number\\":$j,\\"sample_size\\":500}" \\')
        print('      --gpu-id 0 2>&1 | grep -E "TRIAL_METRICS|Objective|Mask"')
        print('  done')
        print()
        print('Deploy to rigs after smoke test passes:')
        print('  scp scorer_trial_worker.py 192.168.3.120:~/distributed_prng_analysis/')
        print('  scp scorer_trial_worker.py 192.168.3.154:~/distributed_prng_analysis/')
        print('  md5sum scorer_trial_worker.py')
    else:
        print('❌ Some checks failed -- review above before proceeding.')


if __name__ == '__main__':
    main()
