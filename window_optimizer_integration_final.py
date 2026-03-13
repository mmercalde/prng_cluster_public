#!/usr/bin/env python3
# S115 N2: guarded optuna import — pruning only fires if optuna present
try:
    import optuna as _optuna_module
    _OPTUNA_AVAILABLE = True
except ImportError:
    _optuna_module = None
    _OPTUNA_AVAILABLE = False
"""
Window Optimizer Integration - WITH VARIABLE SKIP SUPPORT
==========================================================
Version: 3.1
Date: 2026-02-22

CHANGELOG:
  v3.1 (2026-02-22) - S104 FIX: Restore 7 missing intersection fields
    Fields lost during S103 rewrite: intersection_count, intersection_ratio,
    forward_only_count, reverse_only_count, survivor_overlap_ratio,
    bidirectional_selectivity, intersection_weight.
    Formulas restored from v2.0 backup (bak_20260221_pre_s103).
    Variable names updated to match v3.0 naming (forward_records not forward_survivors).
    Applied to both constant skip and variable skip (hybrid) blocks.

  v3.0 (2026-02-21) - S103 FIX: Preserve per-seed match rates from sieve
    CRITICAL BUG FIX: extract_survivors_from_result() was discarding per-seed
    match_rate computed by the GPU kernel and returning only seed integers.
    The accumulator then stamped trial-level aggregate counts onto every survivor,
    making all quality fields (intersection_ratio, bidirectional_selectivity,
    survivor_overlap_ratio, score) identical for all seeds in the same trial.

    FIX:
    - extract_survivors_from_result() renamed to extract_survivor_records()
      Returns List[Dict] with {seed, match_rate} per survivor, not List[int]
    - Accumulator now stores forward_match_rate and reverse_match_rate per seed
    - score field is now the per-seed bidirectional match rate (avg fwd+rev)
    - Trial-level counts retained as context fields (forward_count, etc.)
    - Deduplication updated to use per-seed score (match rate) not trial count

  v2.0 (2025-11-15) - Added variable skip support (test_both_modes flag)
  v1.0 (2025-10-01) - Initial integration

ACCUMULATES ALL BIDIRECTIONAL SURVIVORS WITH RICH METADATA
Saves ALL survivors from ALL trials with window metadata for temporal diversity
"""

from typing import Dict, Any, List, Tuple
import json
import os
from window_optimizer import WindowConfig, TestResult

# S134: Lazy imports for persistent worker path — only loaded when flag is set
try:
    from sieve_filter import load_draws_from_daily3
except ImportError:
    load_draws_from_daily3 = None  # will be imported inside _get_residues_for_config fallback

try:
    from persistent_worker_coordinator import run_trial_persistent
except ImportError:
    run_trial_persistent = None  # only needed when --use-persistent-workers is set


def extract_survivor_records(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract survivor records (seed + match_rate) from coordinator result.

    v3.0: Returns full records [{seed, match_rate}, ...] instead of [int, ...]
    The sieve GPU kernel computes match_rate per seed - this must be preserved
    as it is the primary per-seed quality signal for downstream ML.

    Args:
        result: Dictionary containing job results from coordinator

    Returns:
        List of dicts: [{'seed': int, 'match_rate': float}, ...]
        Deduped by seed, keeping highest match_rate per seed.
    """
    records = {}  # seed -> best match_rate record

    if 'results' in result:
        for job_result in result['results']:
            # Format 1a: Survivors directly in job result
            if 'survivors' in job_result:
                for survivor in job_result['survivors']:
                    seed = survivor.get('seed', survivor.get('id'))
                    if seed is not None:
                        rate = float(survivor.get('match_rate', 0.0))
                        if seed not in records or rate > records[seed]['match_rate']:
                            records[seed] = {'seed': seed, 'match_rate': rate}

            # Format 1b: Survivors grouped by PRNG family
            if 'per_family' in job_result:
                for family, family_data in job_result['per_family'].items():
                    if 'survivors' in family_data:
                        for survivor in family_data['survivors']:
                            seed = survivor.get('seed', survivor.get('id'))
                            if seed is not None:
                                rate = float(survivor.get('match_rate', 0.0))
                                if seed not in records or rate > records[seed]['match_rate']:
                                    records[seed] = {'seed': seed, 'match_rate': rate}

    return list(records.values())


# Keep old name as alias for any callers that used it for seed-only access
def extract_survivors_from_result(result: Dict[str, Any]) -> List[int]:
    """
    Legacy compatibility wrapper - returns seed integers only.
    New code should use extract_survivor_records() to preserve match_rate.
    """
    return [r['seed'] for r in extract_survivor_records(result)]


# ============================================================================
# S134: PERSISTENT WORKER HELPERS
# These are only called when use_persistent_workers=True.
# The original run_bidirectional_test path never calls these.
# ============================================================================

def _get_residues_for_config(config, dataset_path: str):
    """
    Load draw residues from dataset file for the given window config.
    Mirrors what sieve_filter.py does internally via load_draws_from_daily3().
    """
    if load_draws_from_daily3 is not None:
        return load_draws_from_daily3(
            path        = dataset_path,
            window_size = config.window_size,
            offset      = config.offset,
        )
    # Fallback: load manually
    import json as _json
    with open(dataset_path) as f:
        data = _json.load(f)
    draws = data if isinstance(data, list) else data.get("draws", [])
    n = len(draws)
    start = max(0, min(int(config.offset), n - config.window_size))
    end   = start + config.window_size
    window = draws[start:end]
    return [int(e.get("full_state", e["draw"])) if isinstance(e, dict) else int(e)
            for e in window]


def _build_test_result_from_pw(pw_result: dict, accumulator, config,
                                prng_base: str, trial_number: int,
                                optuna_trial=None):
    """
    Convert run_trial_persistent() output into a TestResult and update accumulator.
    Mirrors the accumulator logic in the original run_bidirectional_test path.
    """
    from window_optimizer import TestResult

    bidi_constant = pw_result.get("bidirectional_constant", set())
    bidi_variable = pw_result.get("bidirectional_variable", set())
    fwd_map       = pw_result.get("forward_map", {})
    rev_map       = pw_result.get("reverse_map", {})
    fwd_records   = pw_result.get("forward_records", [])
    rev_records   = pw_result.get("reverse_records", [])
    fwd_h_records = pw_result.get("forward_records_hybrid", [])
    rev_h_records = pw_result.get("reverse_records_hybrid", [])

    total_bidi = len(bidi_constant) + len(bidi_variable)

    # Update accumulator (same logic as original path)
    if accumulator is not None:
        for seed in bidi_constant | bidi_variable:
            fmr = fwd_map.get(seed, 0.0)
            rmr = rev_map.get(seed, 0.0)
            is_var = seed in bidi_variable
            _union = set(fwd_map.keys()) | set(rev_map.keys())
            accumulator['bidirectional'].append({
                'seed':                    seed,
                'forward_match_rate':      fmr,
                'reverse_match_rate':      rmr,
                'score':                   (fmr + rmr) / 2,
                'window_size':             config.window_size,
                'offset':                  config.offset,
                'skip_min':                config.skip_min,
                'skip_max':                config.skip_max,
                'trial_number':            trial_number,
                'prng_type':               prng_base + ("_hybrid" if is_var else ""),
                'prng_base':               prng_base,
                'skip_mode':               "variable" if is_var else "constant",
                'forward_count':           len(fwd_map),
                'reverse_count':           len(rev_map),
                'bidirectional_count':     total_bidi,
                'forward_only_count':      len(set(fwd_map.keys()) - set(rev_map.keys())),
                'reverse_only_count':      len(set(rev_map.keys()) - set(fwd_map.keys())),
                'intersection_count':      len(bidi_constant),
                'intersection_ratio':      len(bidi_constant) / max(len(_union), 1),
                'survivor_overlap_ratio':  len(bidi_constant) / max(len(fwd_map), 1),
                'intersection_weight':     len(bidi_constant) / max(len(fwd_map) + len(rev_map), 1),
                'sessions':                getattr(config, 'sessions', 'all'),
                'skip_range':              f"{config.skip_min}-{config.skip_max}",
                'forward_match_rate':      fmr,
                'reverse_match_rate':      rmr,
            })
        accumulator['forward'].extend(fwd_records + fwd_h_records)
        accumulator['reverse'].extend(rev_records + rev_h_records)

    return TestResult(
        config             = config,
        forward_count      = len(fwd_map),
        reverse_count      = len(rev_map),
        bidirectional_count= total_bidi,
        iteration          = trial_number,
    )


def run_bidirectional_test(coordinator,
                           config: WindowConfig,
                           dataset_path: str,
                           seed_start: int,
                           seed_count: int,
                           prng_base: str = 'java_lcg',
                           test_both_modes: bool = False,
                           forward_threshold: float = 0.01,
                           reverse_threshold: float = 0.01,
                           trial_number: int = 0,
                           accumulator: Dict[str, List] = None,
                           optuna_trial=None) -> TestResult:  # S115 M2
    """
    Run forward + reverse sieve and ACCUMULATE survivors with metadata.

    v3.0: Survivors now carry per-seed forward_match_rate and reverse_match_rate
    from the GPU sieve kernel. These are genuine quality signals (0.0-1.0) that
    vary per seed, enabling downstream ML feature discrimination.

    NEW IN V2.0: Optionally tests BOTH constant and variable skip patterns.
    """

    # ========================================================================
    # S134: PERSISTENT WORKER PATH — activated by use_persistent_workers=True
    # Zero changes to the original path below. This is a purely additive gate.
    # WATCHER compatible — transparent, same output files produced.
    # ========================================================================
    _use_pw = getattr(coordinator, 'use_persistent_workers', False)
    if _use_pw:
        if run_trial_persistent is None:
            raise ImportError("persistent_worker_coordinator.py not found — cannot use --use-persistent-workers")
        _pw_result = run_trial_persistent(
            coordinator_cfg   = getattr(coordinator, 'config_file', 'distributed_config.json'),
            config            = config,
            trial_number      = trial_number,
            prng_base         = prng_base,
            residues          = _get_residues_for_config(config, dataset_path),
            total_seeds       = seed_count,
            forward_threshold = forward_threshold,
            reverse_threshold = reverse_threshold,
            test_both_modes   = test_both_modes,
            dataset_path      = dataset_path,
            worker_pool_size  = getattr(coordinator, 'worker_pool_size', 8),
            seed_cap_nvidia   = getattr(coordinator, 'seed_cap_nvidia', 5_000_000),
            seed_cap_amd      = getattr(coordinator, 'seed_cap_amd',    2_000_000),
        )
        if _pw_result.get("pruned"):
            # Return minimal pruned TestResult — only fields TestResult accepts
            return TestResult(
                config              = config,
                forward_count       = 0,
                reverse_count       = 0,
                bidirectional_count = 0,
                iteration           = trial_number,
            )
        # Build TestResult from persistent worker result
        return _build_test_result_from_pw(_pw_result, accumulator, config,
                                          prng_base, trial_number, optuna_trial)
    # ========================================================================
    # END PERSISTENT WORKER PATH — original path continues unchanged below
    # ========================================================================

    # ========================================================================
    # HELPER: Args Class for Coordinator
    # ========================================================================
    class Args:
        def __init__(self):
            self.target_file = dataset_path
            self.method = 'residue_sieve'
            self.seed_start = seed_start
            self.seeds = seed_count
            self.window_size = config.window_size
            self.offset = config.offset
            self.skip_min = config.skip_min
            self.skip_max = config.skip_max
            self.threshold = forward_threshold
            self.resume_policy = 'restart'
            self.max_concurrent = 26
            self.analysis_type = 'statistical'
            self.draw_match = None

            if set(config.sessions) == {'midday', 'evening'}:
                self.session_filter = 'both'
            elif 'midday' in config.sessions:
                self.session_filter = 'midday'
            else:
                self.session_filter = 'evening'

    print(f"\n  Testing: {config.description()}")

    # ========================================================================
    # PART 1: CONSTANT SKIP TEST (Always runs)
    # ========================================================================

    print(f"    Running FORWARD sieve ({prng_base}) [CONSTANT SKIP]...")
    forward_args = Args()
    forward_args.step_name = f"Forward Sieve ({prng_base})"
    forward_args.prng_type = prng_base

    forward_result = coordinator.execute_distributed_analysis(
        forward_args.target_file,
        f'results/window_opt_forward_{config.window_size}_{config.offset}_t{trial_number}.json',  # S115 M3
        forward_args,
        forward_args.seeds,
        1000,
        8,
        50
    )

    # v3.0: Extract full records with per-seed match_rate
    forward_records = extract_survivor_records(forward_result)
    print(f"      Forward: {len(forward_records):,} survivors")

    # S115 M2: prune dead trials (forward==0) before expensive reverse sieve
    if optuna_trial is not None:
        if not _OPTUNA_AVAILABLE:
            print("      ⚠️  optuna_trial passed but Optuna not installed — pruning disabled.")
        elif len(forward_records) == 0:
            print(f"      ✂️  PRUNED  trial={optuna_trial.number}  "
                  f"window={config.window_size}  offset={config.offset}  "
                  f"skip={config.skip_min}-{config.skip_max}  forward_count=0")
            raise _optuna_module.exceptions.TrialPruned()

    print(f"    Running REVERSE sieve ({prng_base}_reverse) [CONSTANT SKIP]...")
    reverse_args = Args()
    reverse_args.prng_type = prng_base + "_reverse"  # e.g. java_lcg_reverse
    reverse_args.threshold = reverse_threshold
    reverse_args.step_name = f"Reverse Sieve ({prng_base})"

    reverse_result = coordinator.execute_distributed_analysis(
        reverse_args.target_file,
        f'results/window_opt_reverse_{config.window_size}_{config.offset}_t{trial_number}.json',  # S115 M3
        reverse_args,
        reverse_args.seeds,
        1000,
        8,
        50
    )

    reverse_records = extract_survivor_records(reverse_result)
    print(f"      Reverse: {len(reverse_records):,} survivors")

    # Build lookup dicts: seed -> match_rate
    forward_map = {r['seed']: r['match_rate'] for r in forward_records}
    reverse_map = {r['seed']: r['match_rate'] for r in reverse_records}

    forward_set = set(forward_map.keys())
    reverse_set = set(reverse_map.keys())
    bidirectional_constant = forward_set & reverse_set

    print(f"      ✨ Bidirectional (constant): {len(bidirectional_constant):,} survivors")

    # Update dashboard
    if hasattr(coordinator, "_progress_writer") and coordinator._progress_writer:
        best_so_far = getattr(coordinator, "_best_bidirectional", 0)
        if len(bidirectional_constant) > best_so_far:
            coordinator._best_bidirectional = len(bidirectional_constant)
            best_so_far = len(bidirectional_constant)
        acc_fwd = len(accumulator['forward']) if accumulator else 0
        acc_rev = len(accumulator['reverse']) if accumulator else 0
        acc_bid = len(accumulator['bidirectional']) if accumulator else 0
        coordinator._progress_writer.update_trial_stats(
            trial_num=trial_number,
            forward_survivors=len(forward_records),
            reverse_survivors=len(reverse_records),
            bidirectional=len(bidirectional_constant),
            best_bidirectional=best_so_far,
            config_desc=config.description(),
            accumulated_forward=acc_fwd,
            accumulated_reverse=acc_rev,
            accumulated_bidirectional=acc_bid
        )

    # ========================================================================
    # ACCUMULATE CONSTANT SKIP SURVIVORS WITH METADATA
    # v3.0: Per-seed match rates stored individually, not trial aggregates
    # ========================================================================
    if accumulator is not None:
        # Trial-level context (same for all seeds in this trial)
        # v3.1: Compute trial-level intersection statistics
        _union_size = len(forward_set | reverse_set)
        metadata_base = {
            'window_size': config.window_size,
            'offset': config.offset,
            'skip_min': config.skip_min,
            'skip_max': config.skip_max,
            'skip_range': config.skip_max - config.skip_min,
            'sessions': config.sessions,
            'trial_number': trial_number,
            'prng_base': prng_base,
            'skip_mode': 'constant',
            'prng_type': prng_base,
            # Trial-level counts
            'forward_count': len(forward_records),
            'reverse_count': len(reverse_records),
            'bidirectional_count': len(bidirectional_constant),
            # v3.1: Restored intersection fields (were in v2.0, lost in S103 rewrite)
            'intersection_count': len(bidirectional_constant),
            'intersection_ratio': len(bidirectional_constant) / max(_union_size, 1),
            'forward_only_count': len(forward_set - reverse_set),
            'reverse_only_count': len(reverse_set - forward_set),
            'survivor_overlap_ratio': len(bidirectional_constant) / max(len(forward_set), 1),
            'bidirectional_selectivity': len(forward_set) / max(len(reverse_set), 1),
            'intersection_weight': len(bidirectional_constant) / max(len(forward_set) + len(reverse_set), 1),
        }

        for record in forward_records:
            seed = record['seed']
            accumulator['forward'].append({
                'seed': seed,
                'forward_match_rate': record['match_rate'],  # v3.0: per-seed
                **metadata_base
            })

        for record in reverse_records:
            seed = record['seed']
            accumulator['reverse'].append({
                'seed': seed,
                'reverse_match_rate': record['match_rate'],  # v3.0: per-seed
                **metadata_base
            })

        for seed in bidirectional_constant:
            fwd_rate = forward_map[seed]
            rev_rate = reverse_map[seed]
            accumulator['bidirectional'].append({
                'seed': seed,
                'forward_match_rate': fwd_rate,             # v3.0: per-seed
                'reverse_match_rate': rev_rate,             # v3.0: per-seed
                'score': (fwd_rate + rev_rate) / 2.0,      # v3.0: per-seed avg
                **metadata_base
            })

    # ========================================================================
    # PART 2: VARIABLE SKIP TEST (Only if test_both_modes=True)
    # ========================================================================
    # [S124] Track variable-skip bidirectional count separately so Optuna score
    # reflects BOTH constant AND variable survivors.
    _variable_bidi_count = 0  # stays 0 when test_both_modes=False
    if test_both_modes and not prng_base.endswith('_hybrid'):
        prng_hybrid = f"{prng_base}_hybrid"

        print(f"\n    🔄 TESTING VARIABLE SKIP MODE...")
        print(f"    Running FORWARD sieve ({prng_hybrid}) [VARIABLE SKIP]...")

        forward_args_hybrid = Args()
        forward_args_hybrid.prng_type = prng_hybrid
        forward_args_hybrid.step_name = f"Forward Sieve ({prng_hybrid}) [VARIABLE]"

        forward_result_hybrid = coordinator.execute_distributed_analysis(
            forward_args_hybrid.target_file,
            f'results/window_opt_forward_hybrid_{config.window_size}_{config.offset}_t{trial_number}.json',  # S115 M3
            forward_args_hybrid,
            forward_args_hybrid.seeds,
            1000, 8, 50
        )

        forward_records_hybrid = extract_survivor_records(forward_result_hybrid)
        print(f"      Forward (variable): {len(forward_records_hybrid):,} survivors")

        print(f"    Running REVERSE sieve ({prng_hybrid}_reverse) [VARIABLE SKIP]...")
        reverse_args_hybrid = Args()
        reverse_args_hybrid.threshold = reverse_threshold
        reverse_args_hybrid.step_name = f"Reverse Sieve ({prng_hybrid}) [VARIABLE]"
        reverse_args_hybrid.prng_type = prng_hybrid + "_reverse"  # e.g. java_lcg_hybrid_reverse

        reverse_result_hybrid = coordinator.execute_distributed_analysis(
            reverse_args_hybrid.target_file,
            f'results/window_opt_reverse_hybrid_{config.window_size}_{config.offset}_t{trial_number}.json',  # S115 M3
            reverse_args_hybrid,
            reverse_args_hybrid.seeds,
            1000, 8, 50
        )

        reverse_records_hybrid = extract_survivor_records(reverse_result_hybrid)
        print(f"      Reverse (variable): {len(reverse_records_hybrid):,} survivors")

        forward_map_hybrid = {r['seed']: r['match_rate'] for r in forward_records_hybrid}
        reverse_map_hybrid = {r['seed']: r['match_rate'] for r in reverse_records_hybrid}
        forward_set_hybrid = set(forward_map_hybrid.keys())
        reverse_set_hybrid = set(reverse_map_hybrid.keys())
        bidirectional_variable = forward_set_hybrid & reverse_set_hybrid
        _variable_bidi_count = len(bidirectional_variable)   # [S124] wire into Optuna score

        print(f"      ✨ Bidirectional (variable): {len(bidirectional_variable):,} survivors")

        if accumulator is not None:
            # v3.1: Compute trial-level intersection statistics (variable skip)
            _union_size_hybrid = len(forward_set_hybrid | reverse_set_hybrid)
            metadata_base_hybrid = {
                'window_size': config.window_size,
                'offset': config.offset,
                'skip_min': config.skip_min,
                'skip_max': config.skip_max,
                'skip_range': config.skip_max - config.skip_min,
                'sessions': config.sessions,
                'trial_number': trial_number,
                'prng_base': prng_base,
                'skip_mode': 'variable',
                'prng_type': prng_hybrid,
                # Trial-level counts
                'forward_count': len(forward_records_hybrid),
                'reverse_count': len(reverse_records_hybrid),
                'bidirectional_count': len(bidirectional_variable),
                # v3.1: Restored intersection fields (were in v2.0, lost in S103 rewrite)
                'intersection_count': len(bidirectional_variable),
                'intersection_ratio': len(bidirectional_variable) / max(_union_size_hybrid, 1),
                'forward_only_count': len(forward_set_hybrid - reverse_set_hybrid),
                'reverse_only_count': len(reverse_set_hybrid - forward_set_hybrid),
                'survivor_overlap_ratio': len(bidirectional_variable) / max(len(forward_set_hybrid), 1),
                'bidirectional_selectivity': len(forward_set_hybrid) / max(len(reverse_set_hybrid), 1),
                'intersection_weight': len(bidirectional_variable) / max(len(forward_set_hybrid) + len(reverse_set_hybrid), 1),
            }

            for record in forward_records_hybrid:
                seed = record['seed']
                accumulator['forward'].append({
                    'seed': seed,
                    'forward_match_rate': record['match_rate'],
                    **metadata_base_hybrid
                })

            for record in reverse_records_hybrid:
                seed = record['seed']
                accumulator['reverse'].append({
                    'seed': seed,
                    'reverse_match_rate': record['match_rate'],
                    **metadata_base_hybrid
                })

            for seed in bidirectional_variable:
                fwd_rate = forward_map_hybrid[seed]
                rev_rate = reverse_map_hybrid[seed]
                accumulator['bidirectional'].append({
                    'seed': seed,
                    'forward_match_rate': fwd_rate,
                    'reverse_match_rate': rev_rate,
                    'score': (fwd_rate + rev_rate) / 2.0,
                    **metadata_base_hybrid
                })

    # ========================================================================
    # PRINT ACCUMULATOR STATUS
    # ========================================================================
    if accumulator is not None:
        print(f"      📊 Accumulated totals:")
        print(f"         Forward: {len(accumulator['forward'])} total")
        print(f"         Reverse: {len(accumulator['reverse'])} total")
        print(f"         Bidirectional: {len(accumulator['bidirectional'])} total")

    # [S124] Combined bidirectional score: constant + variable skip survivors
    _total_bidi = len(bidirectional_constant) + _variable_bidi_count
    return TestResult(
        config=config,
        forward_count=len(forward_records),
        reverse_count=len(reverse_records),
        bidirectional_count=_total_bidi,   # constant + variable (S124)
        iteration=trial_number
    )


def add_window_optimizer_to_coordinator():
    """
    Add window optimization method to coordinator.
    Monkey-patches MultiGPUCoordinator with optimize_window().
    """
    from coordinator import MultiGPUCoordinator
    from window_optimizer import (WindowOptimizer, SearchBounds,
                                   RandomSearch, GridSearch,
                                   BayesianOptimization, EvolutionarySearch,
                                   BidirectionalCountScorer)

    def optimize_window(self,
                        dataset_path: str,
                        seed_start: int = 0,
                        seed_count: int = 10_000_000,
                        prng_base: str = 'java_lcg',
                        test_both_modes: bool = False,
                        strategy_name: str = 'bayesian',
                        max_iterations: int = 50,
                        output_file: str = 'window_optimization.json',
                        resume_study: bool = False,
                        study_name: str = '',
                        n_parallel: int = 1,
                        enable_pruning: bool = False,
                        trse_context_file: str = 'trse_context.json'):  # S123 TRSE thread
        # S115 M1/M4: Partition map (IPs from distributed_config.json)
        # P0: localhost+192.168.3.120 (10 GPUs, ~141 TFLOPS)
        # P1: 192.168.3.154+192.168.3.162 (16 GPUs, ~142 TFLOPS)
        # M5: imbalance documented — TFLOPS near-equal; logged per trial
        _PARALLEL_PARTITIONS = {
            0: ['localhost', '192.168.3.120'],
            1: ['192.168.3.154', '192.168.3.162'],
        }
        _partition_coordinators = {}

        def _get_partition_coordinator(idx):
            if idx not in _partition_coordinators:
                from coordinator import MultiGPUCoordinator as _MCC
                coord = _MCC(
                    config_file=getattr(self, 'config_file', 'distributed_config.json'),
                    node_allowlist=_PARALLEL_PARTITIONS[idx % len(_PARALLEL_PARTITIONS)],
                    seed_cap_nvidia=5_000_000,
                    seed_cap_amd=2_000_000,
                )
                coord.load_configuration()
                coord.create_gpu_workers()
                _partition_coordinators[idx] = coord
                print(f"   🔀 Partition {idx} coordinator ready: {_PARALLEL_PARTITIONS[idx % len(_PARALLEL_PARTITIONS)]}")
            return _partition_coordinators[idx]

        def _shutdown_partition_coordinators():
            for c in _partition_coordinators.values():
                try: c.ssh_pool.cleanup_all()
                except Exception: pass
            _partition_coordinators.clear()

        # ====================================================================
        # S125 Bug B fix: multiprocessing dispatcher for n_parallel > 1
        # Each Process owns one partition and its own isolated CUDA/ROCm context.
        # Both share the same SQLite Optuna DB via RDBStorage(timeout=20s).
        # Replaces the broken n_jobs=N threading approach (shared CUDA context).
        # ====================================================================
        if n_parallel > 1:
            import multiprocessing as _mp
            import glob as _mpglob
            import time as _mptime

            def _partition_worker(partition_idx, allowlist, config_file_w,
                                   dataset_path_w, seed_start_w, seed_count_w,
                                   prng_base_w, test_both_modes_w,
                                   storage_url, study_name_w, trials_for_worker,
                                   result_queue, temp_file):
                # Runs in a separate process; has its own CUDA context.
                import sys as _sys
                _sys.path.insert(0, '/home/michael/distributed_prng_analysis')  # S137: hardcoded, fork-safe
                try:
                    from coordinator import MultiGPUCoordinator as _WMCC
                    from window_optimizer_integration_final import run_bidirectional_test as _wbt
                    from window_optimizer import (
                        WindowConfig, SearchBounds, BidirectionalCountScorer,
                    )
                    import optuna as _opt2

                    _opt2.logging.set_verbosity(_opt2.logging.WARNING)

                    # Isolated coordinator -- only this partition's nodes
                    _wcoord = _WMCC(
                        config_file=config_file_w,
                        node_allowlist=allowlist,
                        seed_cap_nvidia=5_000_000,
                        seed_cap_amd=2_000_000,
                    )
                    _wcoord.load_configuration()
                    _wcoord.create_gpu_workers()

                    _local_acc = {'forward': [], 'reverse': [], 'bidirectional': []}
                    _local_bounds = SearchBounds.from_config()

                    # S139B: Apply TRSE Rule A in partition worker path
                    # Mirrors OptunaBayesianSearch.search() lines 380-406
                    # Passive: no-op if context absent, stale, or confidence low
                    try:
                        import json as _trse_json
                        import os as _trse_os
                        _trse_path = trse_context_file if trse_context_file else 'trse_context.json'
                        if _trse_os.path.exists(_trse_path):
                            with open(_trse_path) as _tf:
                                _trse_ctx = _trse_json.load(_tf)
                            _trse_ver = _trse_ctx.get('trse_version', '0.0.0')
                            _vmaj, _vmin = int(_trse_ver.split('.')[0]), int(_trse_ver.split('.')[1])
                            if (_vmaj, _vmin) >= (1, 15):
                                _regime_type   = _trse_ctx.get('regime_type', 'unknown')
                                _type_conf     = _trse_ctx.get('regime_type_confidence', 0.0)
                                _regime_stable = _trse_ctx.get('regime_stable', False)
                                _w3_w8_ratio   = _trse_ctx.get('w3_w8_ratio', None)
                                print(f"\n[TRSE][P{partition_idx}] Context loaded — "
                                      f"regime_type={_regime_type} "
                                      f"type_conf={_type_conf:.3f} "
                                      f"stable={_regime_stable} "
                                      f"w3_w8_ratio={_w3_w8_ratio}")
                                if (_regime_type == 'short_persistence'
                                        and _type_conf >= 0.70
                                        and _regime_stable):
                                    _old_max = _local_bounds.max_window_size
                                    _new_max = max(_local_bounds.min_window_size + 1,
                                                   min(32, _local_bounds.max_window_size))
                                    _local_bounds.max_window_size = _new_max
                                    print(f"[TRSE][P{partition_idx}] Rule A ACTIVE: "
                                          f"short_persistence (conf={_type_conf:.3f}) → "
                                          f"window_size ceiling {_old_max} → {_new_max}")
                                else:
                                    print(f"[TRSE][P{partition_idx}] Rule A SKIPPED: "
                                          f"type={_regime_type} conf={_type_conf:.3f} "
                                          f"stable={_regime_stable}")
                            else:
                                print(f"[TRSE][P{partition_idx}] Context version "
                                      f"{_trse_ver} < 1.15 — skipping bounds narrowing")
                        else:
                            print(f"[TRSE][P{partition_idx}] No context found — "
                                  f"running with default bounds")
                    except Exception as _trse_e:
                        print(f"[TRSE][P{partition_idx}] Context load failed "
                              f"(non-fatal): {_trse_e}")

                    _tctr = {'n': 0}

                    def _local_test(cfg, optuna_trial=None):
                        _tctr['n'] += 1
                        return _wbt(
                            coordinator=_wcoord,
                            config=cfg,
                            dataset_path=dataset_path_w,
                            seed_start=seed_start_w,
                            seed_count=seed_count_w,
                            prng_base=prng_base_w,
                            test_both_modes=test_both_modes_w,
                            forward_threshold=_local_bounds.default_forward_threshold,
                            reverse_threshold=_local_bounds.default_reverse_threshold,
                            trial_number=_tctr['n'],
                            accumulator=_local_acc,
                            optuna_trial=optuna_trial,
                        )

                    _pstorage = _opt2.storages.RDBStorage(
                        url=storage_url,
                        engine_kwargs={"connect_args": {"timeout": 20}}
                    )
                    _pstudy = _opt2.load_study(
                        study_name=study_name_w,
                        storage=_pstorage,
                    )

                    def _worker_obj(trial):
                        ws  = trial.suggest_int('window_size',
                                                _local_bounds.min_window_size,
                                                _local_bounds.max_window_size)
                        off = trial.suggest_int('offset',
                                                _local_bounds.min_offset,
                                                _local_bounds.max_offset)
                        si  = trial.suggest_int('session_idx', 0,
                                                len(_local_bounds.session_options) - 1)
                        skn = trial.suggest_int('skip_min',
                                                _local_bounds.min_skip_min,
                                                _local_bounds.max_skip_min)
                        skx = trial.suggest_int('skip_max',
                                                max(skn, _local_bounds.min_skip_max),
                                                _local_bounds.max_skip_max)
                        ft  = trial.suggest_float('forward_threshold',
                                                  _local_bounds.min_forward_threshold,
                                                  _local_bounds.max_forward_threshold)
                        rt  = trial.suggest_float('reverse_threshold',
                                                  _local_bounds.min_reverse_threshold,
                                                  _local_bounds.max_reverse_threshold)
                        cfg = WindowConfig(
                            window_size=ws, offset=off,
                            sessions=_local_bounds.session_options[si],
                            skip_min=skn, skip_max=skx,
                            forward_threshold=round(ft, 2),
                            reverse_threshold=round(rt, 2),
                        )
                        result = _local_test(cfg, optuna_trial=trial)
                        result.iteration = trial.number
                        score = float(result.bidirectional_count)
                        trial.set_user_attr("result_dict", result.to_dict())
                        print(f"   [P{partition_idx}] Trial {trial.number}: "
                              f"{cfg.description()} score={score:.0f}")
                        return score

                    _pstudy.optimize(_worker_obj, n_trials=trials_for_worker, n_jobs=1)

                    # S138: Write accumulator to temp file (avoids 2.4GB pipe deadlock)
                    import json as _json
                    with open(temp_file, 'w') as _tf:
                        _json.dump(_local_acc, _tf)
                    result_queue.put({
                        'partition': partition_idx,
                        'status': 'ok',
                        'temp_file': temp_file,
                    })
                except Exception:
                    import traceback as _tb
                    result_queue.put({
                        'partition': partition_idx,
                        'status': 'error',
                        'error': _tb.format_exc(),
                    })

            # ----------------------------------------------------------------
            # Determine shared Optuna study name + storage URL
            # ----------------------------------------------------------------
            if resume_study and study_name:
                _mp_study_name = study_name
                _mp_storage_url = (
                    "sqlite:////home/michael/distributed_prng_analysis/"
                    f"optuna_studies/{_mp_study_name}.db"
                )
                print(f"   [n_parallel] Workers RESUME study: {_mp_study_name}")
            elif resume_study:
                _mp_dbs = sorted(
                    _mpglob.glob("optuna_studies/window_opt_*.db"),
                    key=os.path.getmtime, reverse=True
                )
                if _mp_dbs:
                    _mp_study_name = os.path.splitext(os.path.basename(_mp_dbs[0]))[0]
                    _mp_storage_url = (
                        "sqlite:////home/michael/distributed_prng_analysis/"
                        f"optuna_studies/{_mp_study_name}.db"
                    )
                    print(f"   [n_parallel] Workers RESUME most recent: {_mp_study_name}")
                else:
                    _mp_study_name = f"window_opt_{int(_mptime.time())}"
                    _mp_storage_url = (
                        "sqlite:////home/michael/distributed_prng_analysis/"
                        f"optuna_studies/{_mp_study_name}.db"
                    )
                    print(f"   [n_parallel] No DB found -- fresh: {_mp_study_name}")
            else:
                _mp_study_name = f"window_opt_{int(_mptime.time())}"
                _mp_storage_url = (
                    "sqlite:////home/michael/distributed_prng_analysis/"
                    f"optuna_studies/{_mp_study_name}.db"
                )
                print(f"   [n_parallel] Fresh study: {_mp_study_name}")

            # Create study + warm-start if fresh
            if not os.path.exists(f"optuna_studies/{_mp_study_name}.db"):
                import optuna as _osetup
                import warnings as _ws2
                from optuna.samplers import TPESampler as _TPS
                _setup_storage = _osetup.storages.RDBStorage(
                    url=_mp_storage_url,
                    engine_kwargs={"connect_args": {"timeout": 20}}
                )
                with _ws2.catch_warnings():
                    _ws2.filterwarnings('ignore', message='.*multivariate.*')
                    _setup_sampler = _TPS(n_startup_trials=3, multivariate=True)
                _setup_study = _osetup.create_study(
                    study_name=_mp_study_name,
                    storage=_setup_storage,
                    direction='maximize',
                    sampler=_setup_sampler,
                    load_if_exists=True,
                )
                if len(_setup_study.trials) == 0:
                    _setup_study.enqueue_trial({
                        'window_size': 8, 'offset': 43,
                        'skip_min': 5, 'skip_max': 56,
                        'forward_threshold': 0.49, 'reverse_threshold': 0.49
                    })
                    print("   [n_parallel] Warm-start enqueued (W8_O43_S5-56)")
                print(f"   [n_parallel] Study ready: {_mp_study_name} "
                      f"({len(_setup_study.trials)} trials)")

            # ----------------------------------------------------------------
            # Divide trials and launch worker processes
            # ----------------------------------------------------------------
            # S137: Initialize accumulator, bounds, optimizer so they exist in n_parallel path
            survivor_accumulator = {'forward': [], 'reverse': [], 'bidirectional': []}
            bounds = SearchBounds.from_config()      # S137-D: needed for session_options after best trial
            optimizer = WindowOptimizer(self, dataset_path)  # S137-E: needed for save_results

            # S138B: Enforce trial ceiling — subtract already-completed trials
            try:
                import optuna as _ocount
                _count_storage = _ocount.storages.RDBStorage(
                    url=_mp_storage_url,
                    engine_kwargs={"connect_args": {"timeout": 20}}
                )
                _count_study = _ocount.load_study(
                    study_name=_mp_study_name,
                    storage=_count_storage,
                )
                _existing_complete = len([
                    t for t in _count_study.trials
                    if t.state == _ocount.trial.TrialState.COMPLETE
                ])
                print(f"   [n_parallel] Existing complete trials: {_existing_complete}")
            except Exception as _ce:
                print(f"   [n_parallel] Could not query existing trials: {_ce} -- assuming 0")
                _existing_complete = 0

            _remaining_trials = max(0, max_iterations - _existing_complete)
            print(f"   [n_parallel] Remaining trials to run: {_remaining_trials} "
                  f"(ceiling={max_iterations}, existing={_existing_complete})")

            if _remaining_trials == 0:
                print(f"   [n_parallel] Trial ceiling already reached -- skipping workers")
                _trials_per_worker = [0] * n_parallel
            else:
                _trials_per_worker = [_remaining_trials // n_parallel] * n_parallel
                for _ri in range(_remaining_trials % n_parallel):
                    _trials_per_worker[_ri] += 1

            if _remaining_trials == 0:
                print(f"\n   [n_parallel] No workers launched — ceiling already met")
            else:
                print(f"\n{'='*60}")
                print(f"LAUNCHING {n_parallel} PARTITION WORKERS (multiprocessing.Process)")
                for _pi in range(n_parallel):
                    print(f"   P{_pi}: {_PARALLEL_PARTITIONS[_pi]}  -> {_trials_per_worker[_pi]} trials")
                print(f"   Study: {_mp_study_name}")
            print(f"{'='*60}\n")

            try:
                _mp.set_start_method('fork', force=True)  # S137: fork avoids pickle on local fn
            except RuntimeError:
                pass  # already set in this process

            _rq = _mp.Queue()
            _procs = []
            for _pi in range(n_parallel):
                _proc = _mp.Process(
                    target=_partition_worker,
                    args=(
                        _pi,
                        _PARALLEL_PARTITIONS[_pi],
                        getattr(self, 'config_file', 'distributed_config.json'),
                        dataset_path, seed_start, seed_count,
                        prng_base, test_both_modes,
                        _mp_storage_url, _mp_study_name,
                        _trials_per_worker[_pi],
                        _rq,
                        f'/tmp/partition_{_pi}_survivors_{_mp_study_name}.json',
                    ),
                    daemon=False,
                )
                _proc.start()
                _procs.append(_proc)
                print(f"   Started Process-{_pi} (pid={_proc.pid}) -> {_PARALLEL_PARTITIONS[_pi]}")

            # Collect status from queue (lightweight — no accumulator payload)
            _collected = 0
            _partition_status = {}
            while _collected < n_parallel:
                try:
                    _res = _rq.get(timeout=7200)
                    _pi = _res['partition']
                    _partition_status[_pi] = _res
                    if _res['status'] == 'ok':
                        print(f"\n   Process-{_pi} signaled OK (survivors in temp file)")
                    else:
                        print(f"\n   Process-{_pi} ERROR:")
                        print(_res.get('error', 'unknown error'))
                    _collected += 1
                except Exception as _qe:
                    print(f"   Queue timeout/error: {_qe}")
                    break

            for _proc in _procs:
                _proc.join(timeout=60)
                if _proc.is_alive():
                    print(f"   Process {_proc.pid} still alive -- terminating")
                    _proc.terminate()

            # S138: Read temp files and merge survivors into accumulator
            import json as _json2
            import os as _os2
            for _pi in range(n_parallel):
                _tf_path = f'/tmp/partition_{_pi}_survivors_{_mp_study_name}.json'
                if _os2.path.exists(_tf_path):
                    print(f"   Process-{_pi} complete -- merging survivors from temp file")
                    try:
                        with open(_tf_path, 'r') as _tf:
                            _res_acc = _json2.load(_tf)
                        for _k in ('forward', 'reverse', 'bidirectional'):
                            survivor_accumulator[_k].extend(_res_acc.get(_k, []))
                        print(f"      Merged: fwd={len(_res_acc.get('forward',[]))} "                              f"rev={len(_res_acc.get('reverse',[]))} "                              f"bid={len(_res_acc.get('bidirectional',[]))}")
                        _os2.remove(_tf_path)
                    except Exception as _tfe:
                        print(f"   ⚠️  Failed to read temp file {_tf_path}: {_tfe}")
                else:
                    print(f"   ⚠️  Process-{_pi} temp file missing: {_tf_path}")

            print(f"\n   All partition workers complete.")
            print(f"      Forward:       {len(survivor_accumulator['forward'])}")
            print(f"      Reverse:       {len(survivor_accumulator['reverse'])}")
            print(f"      Bidirectional: {len(survivor_accumulator['bidirectional'])}")

            # Load best result from study for results dict
            import optuna as _ofin
            _fin_storage = _ofin.storages.RDBStorage(
                url=_mp_storage_url,
                engine_kwargs={"connect_args": {"timeout": 20}}
            )
            _fin_study = _ofin.load_study(
                study_name=_mp_study_name, storage=_fin_storage
            )
            _best_t = _fin_study.best_trial
            print(f"\n   Best trial: #{_best_t.number}  score={_best_t.value:.1f}")

            from window_optimizer import WindowConfig as _WC2
            _bp = _best_t.params
            _si_list = (bounds.session_options
                        if hasattr(bounds, 'session_options')
                        else [['midday'], ['evening']])
            _best_cfg2 = _WC2(
                window_size=_bp['window_size'],
                offset=_bp['offset'],
                sessions=_si_list[_bp.get('session_idx', 0)],
                skip_min=_bp['skip_min'],
                skip_max=_bp['skip_max'],
                forward_threshold=round(_bp.get('forward_threshold', 0.49), 2),
                reverse_threshold=round(_bp.get('reverse_threshold', 0.49), 2),
            )
            results = {
                'strategy': 'optuna_bayesian_parallel',
                'best_config': _best_cfg2.to_dict(),
                'best_result': {
                    'config': _best_cfg2.to_dict(),
                    'bidirectional_count': int(_best_t.value or 0),
                    'forward_count': 0,
                    'reverse_count': 0,
                },
                'best_score': _best_t.value or 0,
                'all_results': [],
                'iterations': len(_fin_study.trials),
                'optuna_study': {
                    'best_trial': _best_t.number,
                    'best_value': _best_t.value,
                    'best_params': _best_t.params,
                }
            }
            optimizer.save_results(results, output_file)
            # Falls through to the dedup+save survivor block below
            # (that block reads survivor_accumulator directly, not 'results')


        if False: pass  # indent anchor

        print(f"\n{'='*80}")
        print(f"WINDOW OPTIMIZATION WITH SURVIVOR ACCUMULATION")
        print(f"Dataset: {dataset_path}")
        print(f"PRNG: {prng_base}")
        if test_both_modes:
            print(f"Mode: TESTING BOTH CONSTANT AND VARIABLE SKIP")
        else:
            print(f"Mode: CONSTANT SKIP ONLY")
        print(f"Seed range: {seed_start:,} → {seed_start + seed_count:,}")
        print(f"Strategy: {strategy_name}")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*80}\n")

        survivor_accumulator = {
            'forward': [],
            'reverse': [],
            'bidirectional': []
        }

        optimizer = WindowOptimizer(self, dataset_path)
        bounds = SearchBounds.from_config()
        trial_counter = {'count': 0}

        def test_config(config,
                        ss=seed_start, sc=seed_count,
                        ft=bounds.default_forward_threshold,
                        rt=bounds.default_reverse_threshold,
                        optuna_trial=None):  # S115 M2
            trial_counter['count'] += 1
            # S115 M1/M5: route to partition coordinator
            if optuna_trial is not None and n_parallel > 1:
                _part = optuna_trial.number % n_parallel
                _coord = _get_partition_coordinator(_part)
                print(f"   🔀 Trial {optuna_trial.number} → Partition {_part} ({_PARALLEL_PARTITIONS[_part]})")
            else:
                _coord = self
            return run_bidirectional_test(
                coordinator=_coord,   # S125: was 'self' -- dead routing var fixed (Bug A)
                config=config,
                dataset_path=dataset_path,
                seed_start=ss,
                seed_count=sc,
                prng_base=prng_base,
                test_both_modes=test_both_modes,
                forward_threshold=ft,
                reverse_threshold=rt,
                trial_number=trial_counter['count'],
                accumulator=survivor_accumulator,
                optuna_trial=optuna_trial          # S119 Gap5
            )

        optimizer.test_configuration = test_config

        strategy_map = {
            'random': RandomSearch(),
            'grid': GridSearch(
                window_sizes=[512, 768, 1024],
                offsets=[0, 100],
                skip_ranges=[(0, 20), (0, 50)]
            ),
            'bayesian': BayesianOptimization(n_initial=3, enable_pruning=enable_pruning, n_parallel=n_parallel),  # S115 wire-up
            'evolutionary': EvolutionarySearch(population_size=10)
        }

        strategy = strategy_map.get(strategy_name, RandomSearch())

        # [S140b] trial history context — flows to Optuna callback
        _trial_history_ctx = {
            'run_id':     f"step1_{prng_base}_{int(seed_start)}",
            'study_name': study_name,
            'prng_type':  prng_base,
            'seed_start': seed_start,
            'seed_end':   seed_start + seed_count,
        }

        results = optimizer.optimize(
            strategy=strategy,
            bounds=bounds,
            max_iterations=max_iterations,
            scorer=BidirectionalCountScorer(),
            seed_start=seed_start,
            seed_count=seed_count,
            resume_study=resume_study,              # S116-Bug5 confirmed
            study_name=study_name,                  # S116-Bug5 confirmed
            trse_context_file=trse_context_file,    # S123 TRSE thread
            trial_history_context=_trial_history_ctx  # [S140b]
        )

        optimizer.save_results(results, output_file)

        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        best = results['best_config']
        print(f"  Window size: {best['window_size']}")
        print(f"  Offset: {best['offset']}")
        print(f"  Sessions: {', '.join(best['sessions'])}")
        print(f"  Skip range: [{best['skip_min']}, {best['skip_max']}]")
        print(f"  Bidirectional survivors: {results['best_result']['bidirectional_count']:,}")
        print(f"{'='*80}\n")

        # ====================================================================
        # SAVE ALL ACCUMULATED SURVIVORS WITH METADATA
        # ====================================================================
        print(f"\n{'='*80}")
        print("SAVING ALL ACCUMULATED SURVIVORS WITH METADATA")
        print(f"{'='*80}")

        try:
            def deduplicate_survivors(survivor_list):
                """Keep survivor with highest per-seed score for each unique seed."""
                seed_map = {}
                for survivor in survivor_list:
                    seed = survivor['seed']
                    if seed not in seed_map or survivor.get('score', 0) > seed_map[seed].get('score', 0):
                        seed_map[seed] = survivor
                return list(seed_map.values())

            forward_deduped = deduplicate_survivors(survivor_accumulator['forward'])
            reverse_deduped = deduplicate_survivors(survivor_accumulator['reverse'])
            bidirectional_deduped = deduplicate_survivors(survivor_accumulator['bidirectional'])

            with open('forward_survivors.json', 'w') as f:
                json.dump(sorted(forward_deduped, key=lambda x: x['seed']), f, indent=2)
            print(f"✅ Saved forward_survivors.json: {len(forward_deduped)} unique seeds")

            with open('reverse_survivors.json', 'w') as f:
                json.dump(sorted(reverse_deduped, key=lambda x: x['seed']), f, indent=2)
            print(f"✅ Saved reverse_survivors.json: {len(reverse_deduped)} unique seeds")

            with open('bidirectional_survivors.json', 'w') as f:
                json.dump(sorted(bidirectional_deduped, key=lambda x: x['seed']), f, indent=2)
            print(f"✅ Saved bidirectional_survivors.json: {len(bidirectional_deduped)} unique seeds")

            # Print sample to confirm per-seed fields present
            if bidirectional_deduped:
                sample = bidirectional_deduped[0]
                print(f"\n📊 Sample survivor:")
                print(f"   seed: {sample['seed']}")
                print(f"   forward_match_rate: {sample.get('forward_match_rate', 'MISSING')}")
                print(f"   reverse_match_rate: {sample.get('reverse_match_rate', 'MISSING')}")
                print(f"   score: {sample.get('score', 'MISSING')}")
                print(f"   window_size: {sample['window_size']}, trial: {sample['trial_number']}")

            if test_both_modes:
                constant_count = sum(1 for s in bidirectional_deduped if s.get('skip_mode') == 'constant')
                variable_count = sum(1 for s in bidirectional_deduped if s.get('skip_mode') == 'variable')
                print(f"\n📈 Skip Mode Distribution:")
                print(f"   Constant skip: {constant_count} survivors")
                print(f"   Variable skip: {variable_count} survivors")

            # Convert to NPZ binary format (required by Step 2)
            from subprocess import run as subprocess_run, CalledProcessError
            try:
                subprocess_run(
                    ["python3", "convert_survivors_to_binary.py", "bidirectional_survivors.json"],
                    check=True
                )
                print(f"✅ Converted to bidirectional_survivors_binary.npz")
            except CalledProcessError as e:
                print(f"❌ NPZ conversion failed: {e}")
                raise RuntimeError("Step 1 incomplete - NPZ conversion required for Step 2")

            print(f"{'='*80}\n")

        except Exception as e:
            print(f"⚠️  Error saving survivors with metadata: {e}")
            import traceback
            traceback.print_exc()

        try:
            from integration.sieve_integration import save_bidirectional_sieve_results
            save_bidirectional_sieve_results(
                forward_survivors=[],
                reverse_survivors=[],
                intersection=[],
                config={
                    'prng_type': prng_base,
                    'seed_start': seed_start,
                    'seed_end': seed_start + seed_count,
                    'total_seeds': seed_count,
                    'window_size': best.get('window_size', 0),
                    'offset': best.get('offset', 0),
                    'skip_min': best.get('skip_min', 0),
                    'skip_max': best.get('skip_max', 0),
                    'forward_threshold': best.get('forward_threshold', 0.01),
                    'reverse_threshold': best.get('reverse_threshold', 0.01),
                    'dataset': dataset_path,
                    'sessions': best.get('sessions', [])
                },
                run_id=f"window_opt_{prng_base}_{strategy_name}"
            )
        except Exception as e:
            print(f"Note: New results format unavailable: {e}")

        return results

    MultiGPUCoordinator.optimize_window = optimize_window
    print("✅ Window optimizer integrated into MultiGPUCoordinator")
