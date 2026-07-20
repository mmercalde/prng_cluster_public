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

try:
    from zmq_sqlite_coordinator import run_trial_zmq_sqlite
except ImportError:
    run_trial_zmq_sqlite = None  # only needed when --use-zmq-sqlite is set

# [S172 Phase 1] RANGE-MINER runner — optional import so this module keeps working
# on hosts without the miner/ package. Enabling --use-range-miner without the
# package raises ImportError inside the gate below, matching the PWC/ZMQ pattern.
try:
    from miner import run_trial_miner
except ImportError:
    run_trial_miner = None


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


# ─────────────────────────────────────────────────────────────────────────────
# [S152] Incremental NPZ flush — write survivors to disk as-found
# ─────────────────────────────────────────────────────────────────────────────
import os as _os_flush
import numpy as _np_flush

# Flush threshold: flush NPZ after this many NEW bidi survivors accumulate.
# Override via env: PRNG_FLUSH_EVERY=1 to flush after every chunk.
_FLUSH_EVERY = int(_os_flush.environ.get("PRNG_FLUSH_EVERY", "10"))

# Tracks how many survivors were present at the last flush (module-level state,
# reset to 0 at process start — safe because each run is a fresh process).
_flush_last_count = 0


def _flush_npz_incremental(accumulator: dict, label: str = "") -> None:
    """
    Atomic merge-write of accumulator bidirectional survivors to NPZ.

    - Deduplicates by seed (highest score wins).
    - Merges with any pre-existing NPZ on disk.
    - Writes atomically via .tmp → rename.
    - Updates both bidirectional_survivors_all.npz  (with scores)
      and    bidirectional_survivors_binary.npz     (Steps 2-6 format).
    - Non-fatal: any write error is logged but does not raise.
    """
    global _flush_last_count

    bidi = accumulator.get("bidirectional", [])
    current_count = len(bidi)

    new_since_last = current_count - _flush_last_count
    if new_since_last < _FLUSH_EVERY:
        return  # not enough new survivors yet

    try:
        _ACCUM_NPZ  = "bidirectional_survivors_all.npz"
        _BINARY_NPZ = "bidirectional_survivors_binary.npz"

        # Deduplicate: highest score per seed wins
        seen: dict = {}
        for s in bidi:
            seed = int(s["seed"])
            if seed not in seen or s.get("score", 0.0) > seen[seed].get("score", 0.0):
                seen[seed] = s

        # Merge with prior NPZ if it exists
        if _os_flush.path.exists(_ACCUM_NPZ):
            try:
                prior = _np_flush.load(_ACCUM_NPZ)
                prior_seeds  = prior["seeds"]
                prior_scores = prior.get("score", _np_flush.zeros(len(prior_seeds)))
                for i, pseed in enumerate(prior_seeds):
                    pseed = int(pseed)
                    pscore = float(prior_scores[i])
                    if pseed not in seen or pscore > seen[pseed].get("score", 0.0):
                        seen[pseed] = {"seed": pseed, "score": pscore}
            except Exception as _me:
                print(f"[S152-FLUSH] Warning: could not read prior NPZ for merge: {_me}")

        all_survivors = list(seen.values())
        seeds  = _np_flush.array([s["seed"]  for s in all_survivors], dtype=_np_flush.uint64)
        scores = _np_flush.array([s.get("score", 0.0) for s in all_survivors], dtype=_np_flush.float32)
        fwd_mr = _np_flush.array([s.get("forward_match_rate", 0.0) for s in all_survivors], dtype=_np_flush.float32)
        rev_mr = _np_flush.array([s.get("reverse_match_rate", 0.0) for s in all_survivors], dtype=_np_flush.float32)

        # Atomic write — accumulator NPZ
        _tmp = _ACCUM_NPZ + ".flush.tmp"
        _np_flush.savez_compressed(_tmp, seeds=seeds, score=scores)
        _os_flush.replace(_tmp, _ACCUM_NPZ)

        # Atomic write — binary NPZ (Steps 2-6)
        _tmp_bin = _BINARY_NPZ + ".flush.tmp"
        _np_flush.savez_compressed(_tmp_bin, seeds=seeds,
                                   forward_match_rate=fwd_mr,
                                   reverse_match_rate=rev_mr,
                                   score=scores)
        _os_flush.replace(_tmp_bin, _BINARY_NPZ)

        _flush_last_count = 0  # [S166] reset — list cleared below
        # [S166] Clear the in-memory list after flush — data is safe in NPZ.
        # Without this, the list grows unboundedly and causes OOM on Zeus.
        accumulator["bidirectional"] = []
        _tag = f" [{label}]" if label else ""
        print(
            f"[S152-FLUSH]{_tag} NPZ flushed: {len(seeds):,} total survivors "
            f"(+{new_since_last} new this flush, threshold={_FLUSH_EVERY})"
        )

    except Exception as _fe:
        print(f"[S152-FLUSH] Warning: incremental flush failed (non-fatal): {_fe}")


# ─────────────────────────────────────────────────────────────────────────────
# END [S152] incremental flush helper
# ─────────────────────────────────────────────────────────────────────────────

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
        # [S166-ACCUM] Count only — full objects not retained for forward/reverse.
        accumulator['forward_count'] = accumulator.get('forward_count', 0) + len(fwd_records) + len(fwd_h_records)
        accumulator['reverse_count'] = accumulator.get('reverse_count', 0) + len(rev_records) + len(rev_h_records)

        # [S152] Flush survivors to disk as-found (incremental, threshold-gated)
        _flush_npz_incremental(accumulator, label=f"chunk/trial-{trial_number}")

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
                           optuna_trial=None,
                           enable_pruning: bool = False) -> TestResult:  # S115 M2, [S145-R1]
    """
    Run forward + reverse sieve and ACCUMULATE survivors with metadata.

    v3.0: Survivors now carry per-seed forward_match_rate and reverse_match_rate
    from the GPU sieve kernel. These are genuine quality signals (0.0-1.0) that
    vary per seed, enabling downstream ML feature discrimination.

    NEW IN V2.0: Optionally tests BOTH constant and variable skip patterns.
    """

    # ========================================================================
    # [S172 Phase 4] RANGE-MINER PATH — activated by use_range_miner=True
    # Mutually exclusive with --use-persistent-workers and --use-zmq-sqlite
    # (enforced at argparse in window_optimizer.py). Placed at the top of the
    # cascade so miner selection wins unambiguously; PWC/ZMQ paths follow.
    # run_trial_miner drives the real coordinator server (serve_trial); it derives
    # a UNIQUE run_id per trial and resolves the workflow stages from
    # prng_base + test_both_modes. Coexistence with PWC/ZMQ holds — this gate is
    # behind use_range_miner and touches neither.
    # ========================================================================
    _use_miner = getattr(coordinator, 'use_range_miner', False)
    if _use_miner:
        if run_trial_miner is None:
            raise ImportError(
                "miner/ package not found — cannot use --use-range-miner. "
                "Ensure miner/__init__.py and miner/range_miner_coordinator.py "
                "are deployed to the project root."
            )
        _miner_result = run_trial_miner(
            coordinator_cfg        = getattr(coordinator, 'config_file', 'distributed_config.json'),
            config                 = config,
            trial_number           = trial_number,
            prng_base              = prng_base,
            residues               = _get_residues_for_config(config, dataset_path),
            total_seeds            = seed_count,
            forward_threshold      = forward_threshold,
            reverse_threshold      = reverse_threshold,
            test_both_modes        = test_both_modes,
            dataset_path           = dataset_path,
            worker_pool_size       = getattr(coordinator, 'worker_pool_size', 8),
            seed_cap_nvidia        = getattr(coordinator, 'seed_cap_nvidia', 5_000_000),
            seed_cap_amd           = getattr(coordinator, 'seed_cap_amd',    2_000_000),
            # Defect 6: hybrid caps, window params, staging + bind must be wired —
            # not left to defaults (which would silently use Phase 1 / window 1 /
            # loopback bind and drop sessions/offset).
            seed_cap_nvidia_hybrid = getattr(coordinator, 'seed_cap_nvidia_hybrid', 2_500_000),
            seed_cap_amd_hybrid    = getattr(coordinator, 'seed_cap_amd_hybrid',    1_000_000),
            miner_stripe_size      = getattr(coordinator, 'miner_stripe_size', 67_108_864),
            miner_substripes       = getattr(coordinator, 'miner_substripes', 8),
            miner_output_dir       = getattr(coordinator, 'miner_output_dir', None),
            staging_dir            = getattr(coordinator, 'staging_dir', None),
            staging_high_water_bytes = getattr(coordinator, 'staging_high_water_bytes', 16 * 1024 ** 3),
            staging_high_water_files = getattr(coordinator, 'staging_high_water_files', 512),
            compute_lease_timeout  = getattr(coordinator, 'compute_lease_timeout', 300.0),
            staging_timeout        = getattr(coordinator, 'staging_timeout', 600.0),
            # Production bind must be reachable by REMOTE rigs (0.0.0.0), not
            # loopback. Tests inject 127.0.0.1 / a pre-bound listen_sock.
            miner_host             = getattr(coordinator, 'miner_host', '0.0.0.0'),
            miner_port             = getattr(coordinator, 'miner_port', 5700),
            node_allowlist         = getattr(coordinator, 'node_allowlist', None),
            # Resolved WindowConfig window params (were dropped before).
            window_size            = getattr(config, 'window_size', 1),
            sessions               = getattr(config, 'sessions', None),
            offset                 = getattr(config, 'offset', 0),
            # Correction-3 production-timeout: a real multi-billion-seed scan far
            # exceeds any fixed 30s. Default UNBOUNDED (None) so the production path
            # runs until a terminal state; an explicit config value still binds.
            serve_timeout          = getattr(coordinator, 'serve_timeout', None),
        )
        if _miner_result.get("pruned"):
            return TestResult(
                config              = config,
                forward_count       = 0,
                reverse_count       = 0,
                bidirectional_count = 0,
                iteration           = trial_number,
            )
        return _build_test_result_from_pw(_miner_result, accumulator, config,
                                          prng_base, trial_number, optuna_trial)
    # ========================================================================
    # END RANGE-MINER PATH — original path continues unchanged below
    # ========================================================================

    # ========================================================================
    # S134: PERSISTENT WORKER PATH — activated by use_persistent_workers=True
    # Zero changes to the original path below. This is a purely additive gate.
    # WATCHER compatible — transparent, same output files produced.
    # ========================================================================
    _use_pw = getattr(coordinator, 'use_persistent_workers', False)
    if _use_pw:
        if run_trial_persistent is None:
            raise ImportError("persistent_worker_coordinator.py not found — cannot use --use-persistent-workers")
        # [S163-KARG-FIX1] Hops 5→8: pass pwc_host and pwc_port so PersistentWorkerCoordinator
        # binds the partition-correct port — not always defaulting to 5600.
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
            pwc_transport     = getattr(coordinator, 'pwc_transport', 'tcp'),
            pwc_host          = getattr(coordinator, 'pwc_host', '0.0.0.0'),  # [S163-KARG-FIX1] hop 5
            pwc_port          = getattr(coordinator, 'pwc_port', 5600),       # [S163-KARG-FIX1] hop 5
            node_allowlist    = getattr(coordinator, 'node_allowlist', None), # [S163-KARG-PWC] hop 6
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
    # [S158D] ZMQ-SQLITE PATH — activated by use_zmq_sqlite=True
    # Zero changes to original path. Purely additive gate.
    # ========================================================================
    _use_zmq = getattr(coordinator, 'use_zmq_sqlite', False)
    if _use_zmq:
        if run_trial_zmq_sqlite is None:
            raise ImportError(
                "zmq_sqlite_coordinator.py not found — cannot use --use-zmq-sqlite"
            )
        # S159E: session-scoped coordinator — created once, reused across all trials
        # Workers stay connected between trials; sockets never close mid-optimization
        if not hasattr(coordinator, '_zmq_session_coord') or coordinator._zmq_session_coord is None:
            from zmq_sqlite_coordinator import ZMQSQLiteCoordinator as _ZMQSC
            coordinator._zmq_session_coord = _ZMQSC(
                config_file      = getattr(coordinator, 'config_file', 'distributed_config.json'),
                seed_cap_amd     = getattr(coordinator, 'seed_cap_amd',    2_000_000),
                seed_cap_nvidia  = getattr(coordinator, 'seed_cap_nvidia', 5_000_000),
                worker_pool_size = getattr(coordinator, 'worker_pool_size', 8),
            )
            print(f'[S159E] Created new ZMQ session coordinator for trial {trial_number}')
        else:
            print(f'[S159E] Reusing existing ZMQ session coordinator for trial {trial_number}')
        _zmq_result = run_trial_zmq_sqlite(
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
            session_coord     = coordinator._zmq_session_coord,
        )
        if _zmq_result.get("pruned"):
            return TestResult(
                config              = config,
                forward_count       = 0,
                reverse_count       = 0,
                bidirectional_count = 0,
                iteration           = trial_number,
            )
        return _build_test_result_from_pw(
            _zmq_result, accumulator, config, prng_base, trial_number, optuna_trial
        )
    # ========================================================================
    # END ZMQ-SQLITE PATH
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
    # [S145-R1] Gate on enable_pruning — when False, always run reverse sieve
    if optuna_trial is not None and enable_pruning:
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
        acc_fwd = accumulator.get('forward_count', 0) if accumulator else 0
        acc_rev = accumulator.get('reverse_count', 0) if accumulator else 0
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

        # [S166-ACCUM] Stop accumulating full forward/reverse objects — RAM bomb.
        # Only bidirectional objects are load-bearing (NPZ + Steps 2-6).
        # Preserve counts for dashboard, logging, and output JSON contract.
        accumulator['forward_count'] = accumulator.get('forward_count', 0) + len(forward_records)
        accumulator['reverse_count'] = accumulator.get('reverse_count', 0) + len(reverse_records)

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

        # [S147 Q0] Gate: skip hybrid reverse if hybrid forward = 0
        # SKIP not prune — constant-skip results preserved.
        if not forward_records_hybrid:
            print(f"      Hybrid forward zero survivors — skipping hybrid reverse (Q0 gate)")
            reverse_records_hybrid = []
        else:
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

            # [S166-ACCUM] Count only — no full object accumulation for forward/reverse.
            accumulator['forward_count'] = accumulator.get('forward_count', 0) + len(forward_records_hybrid)
            accumulator['reverse_count'] = accumulator.get('reverse_count', 0) + len(reverse_records_hybrid)

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
        print(f"         Forward: {accumulator.get('forward_count', 0)} total (count only)")
        print(f"         Reverse: {accumulator.get('reverse_count', 0)} total (count only)")
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

            # [S140b-NP2] Build warm_start_params from DB
            # Read directly — trial_history_context not in optimize_window scope
            _warm_start_params = None
            try:
                from database_system import DistributedPRNGDatabase as _DBNP2
                _db_np2 = _DBNP2()
                _best_np2 = _db_np2.get_best_step1_params(prng_base, limit=1)
                if _best_np2:
                    _bp_np2 = _best_np2[0]
                    if all(_bp_np2.get(k) is not None for k in
                           ['window_size','offset','skip_min','skip_max',
                            'forward_threshold','reverse_threshold']):
                        _warm_start_params = {
                            'window_size':       int(_bp_np2['window_size']),
                            'offset':            int(_bp_np2['offset']),
                            'skip_min':          int(_bp_np2['skip_min']),
                            'skip_max':          int(_bp_np2['skip_max']),
                            'forward_threshold': float(_bp_np2['forward_threshold']),
                            'reverse_threshold': float(_bp_np2['reverse_threshold']),
                        }
            except Exception as _e_np2:
                print(f'   [n_parallel] warm_start DB lookup failed: {_e_np2}')

            def _partition_worker(partition_idx, allowlist, config_file_w,
                                   dataset_path_w, seed_start_w, seed_count_w,
                                   prng_base_w, test_both_modes_w,
                                   storage_url, study_name_w, trials_for_worker,
                                   result_queue, temp_file,
                                   warm_start_params=None,        # [S140b-NP2]
                                   use_persistent_workers_w=False,  # [S163-KARG-FIX1] hop 3
                                   pwc_transport_w='tcp',           # [S163-KARG-FIX1] hop 3
                                   pwc_min_workers_w=1,             # [S163-KARG-FIX1] hop 3 — permissive for partition
                                   worker_pool_size_w=8,            # [S163-KARG-FIX1] hop 3
                                   seed_cap_nvidia_w=5_000_000,     # [S163-KARG-FIX1] hop 3
                                   seed_cap_amd_w=2_000_000,        # [S163-KARG-FIX1] hop 3
                                   pwc_host_w='0.0.0.0',           # [S163-KARG-FIX1] hop 3
                                   pwc_port_w=5600):               # [S163-KARG-FIX1] hop 3
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

                    # [S163-KARG-FIX1] Isolated coordinator — inherit parent runtime flags.
                    # Constructor only accepts config_file, node_allowlist, seed_caps.
                    # Transport/runtime flags set as attributes after construction (TB ruling).
                    _wcoord = _WMCC(
                        config_file=config_file_w,
                        node_allowlist=allowlist,
                        seed_cap_nvidia=seed_cap_nvidia_w,  # inherit from parent (not hardcoded)
                        seed_cap_amd=seed_cap_amd_w,        # inherit from parent (not hardcoded)
                    )
                    # Hops 3→4: set transport/runtime attributes post-construction
                    _wcoord.use_persistent_workers = use_persistent_workers_w
                    _wcoord.pwc_transport          = pwc_transport_w
                    _wcoord.pwc_min_workers        = pwc_min_workers_w
                    _wcoord.worker_pool_size       = worker_pool_size_w
                    _wcoord.pwc_host               = pwc_host_w
                    _wcoord.pwc_port               = pwc_port_w
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
                        # [S142-C] _worker_obj trial history writes removed per TB ruling.
                        # Canonical step1_trial_history is written by backfill from
                        # the shared Optuna study after all partition workers complete.
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
                    # [S140b-NP2] Dynamic warm-start with fallback
                    _ws_trial = dict(_warm_start_params) if _warm_start_params else {}
                    if _ws_trial:
                        _setup_study.enqueue_trial(_ws_trial)
                        print(
                            f"   [n_parallel] Warm-start enqueued "
                            f"(W{_ws_trial.get('window_size')}_"
                            f"O{_ws_trial.get('offset')}_"
                            f"S{_ws_trial.get('skip_min')}-"
                            f"{_ws_trial.get('skip_max')})"  # [S140b-NP2]
                        )
                    else:
                        _setup_study.enqueue_trial({
                            'window_size': 8, 'offset': 43,
                            'skip_min': 5, 'skip_max': 56,
                            'forward_threshold': 0.49, 'reverse_threshold': 0.49
                        })
                        print("   [n_parallel] Warm-start fallback enqueued (W8_O43_S5-56)")
                print(f"   [n_parallel] Study ready: {_mp_study_name} "
                      f"({len(_setup_study.trials)} trials)")

            # ----------------------------------------------------------------
            # Divide trials and launch worker processes
            # ----------------------------------------------------------------
            # S137: Initialize accumulator, bounds, optimizer so they exist in n_parallel path
            # [S166-ACCUM] forward/reverse are now counts not lists
            survivor_accumulator = {'forward_count': 0, 'reverse_count': 0, 'bidirectional': []}
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

            # [S163-KARG-KILL] Kill all stale pwc_worker_service processes on ALL rigs
            # BEFORE forking partition processes. Without this, stale workers from prior
            # runs reconnect to whichever TCP port comes up first, causing cross-partition
            # contamination (P1's workers connecting to P0's port 5600).
            # Must be done in parent before fork — each partition's S156 only sees its own nodes.
            print(f"\n[NP2-KILL] Killing stale pwc_worker_service on all AMD rigs before fork...")
            import subprocess as _pre_kill_sp
            _all_rig_ips = ['192.168.3.120', '192.168.3.154', '192.168.3.162']
            for _rig_ip in _all_rig_ips:
                try:
                    _pre_kill_sp.run(
                        ['ssh', '-q', '-o', 'StrictHostKeyChecking=no',
                         f'michael@{_rig_ip}',
                         'pkill -9 -f pwc_worker_service 2>/dev/null; echo ok'],
                        capture_output=True, timeout=10
                    )
                    print(f"   [NP2-KILL] {_rig_ip}: stale workers killed")
                except Exception as _kill_e:
                    print(f"   [NP2-KILL] {_rig_ip}: kill failed (non-fatal): {_kill_e}")
            import time as _pre_kill_time
            _pre_kill_time.sleep(2)  # allow processes to die before fork
            print(f"[NP2-KILL] Pre-fork cleanup complete")

            # [S163-KARG-PORT] Kill any zombie processes holding TCP ports 5600-5601
            # kill -9 does not close sockets immediately — zombies hold ports across runs
            import subprocess as _port_kill_sp
            for _port in range(5600, 5600 + n_parallel):
                try:
                    _fuser = _port_kill_sp.run(
                        ['fuser', '-k', f'{_port}/tcp'],
                        capture_output=True, timeout=5
                    )
                    print(f"   [NP2-PORT] fuser -k {_port}/tcp: done")
                except Exception as _pe:
                    pass  # fuser may not be installed — non-fatal
            import time as _port_wait
            _port_wait.sleep(1)  # allow sockets to release
            print(f"[NP2-PORT] Port cleanup complete\n")

            _rq = _mp.Queue()
            _procs = []
            for _pi in range(n_parallel):
                # [S163-KARG-FIX1] Hops 1→2: capture parent transport flags before fork.
                # Port is distinct per partition: P0=5600, P1=5601 (TB-approved scheme).
                _pwc_port_base = getattr(self, 'pwc_port', 5600)
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
                        _warm_start_params,                              # [S140b-NP2]
                        getattr(self, 'use_persistent_workers', False),  # [S163-KARG-FIX1] hop 2
                        getattr(self, 'pwc_transport', 'tcp'),           # [S163-KARG-FIX1] hop 2
                        1,                                               # pwc_min_workers — permissive for partition
                        getattr(self, 'worker_pool_size', 8),            # [S163-KARG-FIX1] hop 2
                        getattr(self, 'seed_cap_nvidia', 5_000_000),     # [S163-KARG-FIX1] hop 2
                        getattr(self, 'seed_cap_amd', 2_000_000),        # [S163-KARG-FIX1] hop 2
                        getattr(self, 'pwc_host', '0.0.0.0'),           # [S163-KARG-FIX1] hop 2
                        _pwc_port_base + _pi,                           # [S163-KARG-FIX1] hop 2 — P0=5600, P1=5601
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
            print(f"      Forward:       {survivor_accumulator.get('forward_count', 0)} (count only)")
            print(f"      Reverse:       {survivor_accumulator.get('reverse_count', 0)} (count only)")
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

            # [S142-B] Canonical trial history backfill from shared study
            # _worker_obj writes are opportunistic. This is the authoritative write.
            print(f"\n   [TRIAL_HISTORY] Backfilling from shared study ({len(_fin_study.trials)} trials)...")
            try:
                from database_system import DistributedPRNGDatabase as _DBFILL
                _db_fill = _DBFILL()
                _si_opts = (bounds.session_options
                            if hasattr(bounds, 'session_options')
                            else [['midday'], ['evening'], ['morning']])
                _fill_written = 0
                _fill_skipped = 0
                for _ft in _fin_study.trials:
                    if _ft.state.name != 'COMPLETE':
                        _fill_skipped += 1
                        continue
                    _fparams = _ft.params or {}
                    _fsi = _fparams.get('session_idx', 0)
                    _fsess_raw = (_si_opts[_fsi]
                                  if isinstance(_si_opts, list) and _fsi < len(_si_opts)
                                  else ['unknown'])
                    _fsess = (','.join(_fsess_raw)
                              if isinstance(_fsess_raw, (list, tuple))
                              else str(_fsess_raw))
                    _db_fill.write_step1_trial(
                        run_id=f"step1_{prng_base}_{int(seed_start)}",  # [S142-C] canonical run_id, no suffix
                        study_name=_mp_study_name,
                        trial_number=int(_ft.number),
                        prng_type=str(prng_base),
                        seed_range_start=int(seed_start),
                        seed_range_end=int(seed_start + seed_count - 1),
                        params={
                            'window_size':       _fparams.get('window_size'),
                            'offset':            _fparams.get('offset'),
                            'skip_min':          _fparams.get('skip_min'),
                            'skip_max':          _fparams.get('skip_max'),
                            'time_of_day':       _fsess,
                            'forward_threshold': _fparams.get('forward_threshold', 0.49),
                            'reverse_threshold': _fparams.get('reverse_threshold', 0.49),
                        },
                        trial_score=float(_ft.value or 0.0),
                        forward_survivors=0,
                        reverse_survivors=0,
                        bidirectional_survivors=int(_ft.value or 0),
                        pruned=False
                    )
                    _fill_written += 1
                print(f"   [TRIAL_HISTORY] Backfill complete: "
                      f"{_fill_written} written, {_fill_skipped} skipped (PRUNED)")
            except Exception as _fill_e:
                print(f"   [TRIAL_HISTORY] Backfill failed (non-fatal): {_fill_e}")

            # Falls through to the dedup+save survivor block below
            # (that block reads survivor_accumulator directly, not 'results')
            print(f"\n[NP2] EXIT NP2 PATH — entering shared dedup+save survivor block")

        # [S142-B] NP2 terminal flag — prevents single-process search from running
        _np2_complete = n_parallel > 1  # True when NP2 block ran above

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

        # [S142-B] Skip single-process search when NP2 already ran
        if not _np2_complete:
            print(f"[SINGLE] ENTER SINGLE-PROCESS SEARCH PATH")
        else:
            print(f"[NP2] Single-process search path SKIPPED (n_parallel={n_parallel})")

        if not _np2_complete:
            survivor_accumulator = {
                'forward': [],
                'reverse': [],
                'bidirectional': []
            }

        if not _np2_complete:
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
                optuna_trial=optuna_trial,         # S119 Gap5
                enable_pruning=enable_pruning      # [S145-R1] closure var
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
        strategy._survivor_accumulator = survivor_accumulator  # [S149]

        # [S140b] trial history context — flows to Optuna callback
        _trial_history_ctx = {
            'run_id':       f"step1_{prng_base}_{int(seed_start)}",
            'study_name':   study_name,
            'prng_type':    prng_base,
            'seed_start':   seed_start,
            'seed_end':     seed_start + seed_count,
            'n_parallel_gt1': n_parallel > 1,  # [S142] guard: NP2 owns writes
        }

        # [S166] Add warm_start fields from DB lookup — feeds Optuna enqueue
        # This mirrors the n_parallel DB lookup path above.
        try:
            from database_system import DistributedPRNGDatabase as _DBW
            _db_ws = _DBW()
            _best_ws = _db_ws.get_best_step1_params(prng_base, limit=1)
            if _best_ws:
                _bws = _best_ws[0]
                _trial_history_ctx['warm_start_window']     = _bws.get('window_size')
                _trial_history_ctx['warm_start_offset']     = _bws.get('offset')
                _trial_history_ctx['warm_start_skip_min']   = _bws.get('skip_min')
                _trial_history_ctx['warm_start_skip_max']   = _bws.get('skip_max')
                _trial_history_ctx['warm_start_session']    = _bws.get('session')
                _trial_history_ctx['warm_start_fwd_thresh'] = _bws.get('forward_threshold')
                _trial_history_ctx['warm_start_rev_thresh'] = _bws.get('reverse_threshold')
                print(f"   [WARM_START] loaded W{_bws.get('window_size')}_O{_bws.get('offset')} from step1_trial_history")
            else:
                print(f"   [WARM_START] no prior history for {prng_base}")
        except Exception as _ews:
            print(f"   [WARM_START] DB lookup failed (non-fatal): {_ews}")

        if not _np2_complete:  # [S142-B] skip single-process search for NP2
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
        best = results.get('best_config', {})
        # [S145-R1] Guard: best_config empty when all trials pruned
        if best and 'window_size' in best:
            print(f"  Window size: {best['window_size']}")
            print(f"  Offset: {best['offset']}")
            print(f"  Sessions: {', '.join(best.get('sessions', []))}")
            print(f"  Skip range: [{best['skip_min']}, {best['skip_max']}]")
            print(f"  Bidirectional survivors: {results['best_result'].get('bidirectional_count', 0):,}")
        else:
            print(f"  ⚠️  All trials pruned — no survivors found in this seed range")
            print(f"  Coverage tracker will advance seed_start on next run")
        print(f"{'='*80}\n")

        # ====================================================================
        # SAVE ALL ACCUMULATED SURVIVORS WITH METADATA
        # ====================================================================
        print(f"\n{'='*80}")
        print("SAVING ALL ACCUMULATED SURVIVORS WITH METADATA")
        print(f"{'='*80}")

        try:
            def deduplicate_survivors(survivor_list):
                """Keep survivor with highest per-seed score for each unique seed.
                [S163-KARG] Vectorized via numpy — replaces O(N) pure Python dict loop.
                """
                if not survivor_list:
                    return []
                import numpy as _np_dedup
                seeds  = _np_dedup.array([s['seed'] for s in survivor_list], dtype=_np_dedup.int64)
                scores = _np_dedup.array([s.get('score', 0.0) for s in survivor_list], dtype=_np_dedup.float32)
                # Sort by seed asc, then score desc — argsort stable keeps last (highest score) per seed
                # Strategy: sort by seed, then for ties keep highest score entry
                order  = _np_dedup.lexsort((-scores, seeds))   # primary: seed asc; secondary: score desc
                sorted_seeds = seeds[order]
                # Keep first occurrence of each unique seed (= highest score due to sort)
                keep_mask = _np_dedup.concatenate(([True], sorted_seeds[1:] != sorted_seeds[:-1]))
                keep_idx  = order[keep_mask]
                return [survivor_list[i] for i in keep_idx]

            # [S163-KARG-DEDUP] TB-approved fix: skip forward/reverse dedup when
            # output will be summary-only anyway. Only bidirectional_deduped is
            # load-bearing (feeds NPZ accumulator + Steps 2-6). Forward/reverse
            # dedup on 1.4M+ records was 100% wasted work at scale.
            # Root cause: S162 NPZ accumulator failure masked this — fallback path
            # skipped dedup entirely. S163 NPZ fix exposed the bottleneck.
            _JSON_WRITE_LIMIT = 100_000

            import time as _dedup_time

            _fwd_count = survivor_accumulator.get('forward_count', 0)
            _rev_count = survivor_accumulator.get('reverse_count', 0)
            _bid_count = len(survivor_accumulator['bidirectional'])

            # [S166-ACCUM] forward/reverse full objects no longer retained in bayesian mode.
            # forward/reverse JSON output is always summary-only (count + metadata).
            # bidirectional JSON + NPZ are unaffected — bidirectional objects still accumulated.
            forward_summary_only = True   # [S166] always summary-only
            reverse_summary_only = True   # [S166] always summary-only
            _ = _fwd_count  # suppress unused warning

            print(f"\n[DEDUP] fwd={_fwd_count:,} ({'summary-only — skipping dedup' if forward_summary_only else 'deduping'})  "
                  f"rev={_rev_count:,} ({'summary-only — skipping dedup' if reverse_summary_only else 'deduping'})  "
                  f"bidi={_bid_count:,} (always dedup)")

            # [S166-ACCUM] forward/reverse objects not retained — always summary-only.
            forward_deduped = None
            reverse_deduped = None
            print(f"[DEDUP] forward: {_fwd_count:,} (count only — objects not retained)")
            print(f"[DEDUP] reverse: {_rev_count:,} (count only — objects not retained)")

            # Bidirectional — ALWAYS dedup — load-bearing path
            _t0 = _dedup_time.time()
            bidirectional_deduped = deduplicate_survivors(survivor_accumulator['bidirectional'])
            print(f"[DEDUP] bidi deduped:    {_bid_count:,} → {len(bidirectional_deduped):,}  "
                  f"({_dedup_time.time()-_t0:.3f}s)")

            # Write forward_survivors.json
            if not forward_summary_only and forward_deduped is not None:
                with open('forward_survivors.json', 'w') as f:
                    json.dump(sorted(forward_deduped, key=lambda x: x['seed']), f, indent=2)
                print(f"✅ Saved forward_survivors.json: {len(forward_deduped):,} unique seeds")
            else:
                _summary = {
                    "survivor_count": _fwd_count,
                    "note": f"Full survivors omitted (count > {_JSON_WRITE_LIMIT:,}) — see bidirectional_survivors_all.npz",
                }
                with open('forward_survivors.json', 'w') as f:
                    json.dump(_summary, f, indent=2)
                print(f"⚠️  forward_survivors.json: summary only ({_fwd_count:,} > {_JSON_WRITE_LIMIT:,}) — NPZ has full data")

            # Write reverse_survivors.json
            if not reverse_summary_only and reverse_deduped is not None:
                with open('reverse_survivors.json', 'w') as f:
                    json.dump(sorted(reverse_deduped, key=lambda x: x['seed']), f, indent=2)
                print(f"✅ Saved reverse_survivors.json: {len(reverse_deduped):,} unique seeds")
            else:
                _summary = {
                    "survivor_count": _rev_count,
                    "note": f"Full survivors omitted (count > {_JSON_WRITE_LIMIT:,}) — see bidirectional_survivors_all.npz",
                }
                with open('reverse_survivors.json', 'w') as f:
                    json.dump(_summary, f, indent=2)
                print(f"⚠️  reverse_survivors.json: summary only ({_rev_count:,} > {_JSON_WRITE_LIMIT:,}) — NPZ has full data")

            # bidirectional_survivors.json — always written (canonical input for Steps 2-6)
            # Same 100K guard for safety, but bidirectional count is normally much smaller.
            if len(bidirectional_deduped) <= _JSON_WRITE_LIMIT:
                with open('bidirectional_survivors.json', 'w') as f:
                    json.dump(sorted(bidirectional_deduped, key=lambda x: x['seed']), f)
                print(f"✅ Saved bidirectional_survivors.json: {len(bidirectional_deduped)} unique seeds")
            else:
                _summary = {
                    "survivor_count": len(bidirectional_deduped),
                    "note": f"Full survivors omitted (count > {_JSON_WRITE_LIMIT:,}) — see bidirectional_survivors_binary.npz",
                }
                with open('bidirectional_survivors.json', 'w') as f:
                    json.dump(_summary, f, indent=2)
                print(f"⚠️  bidirectional_survivors.json: summary only ({len(bidirectional_deduped):,} > {_JSON_WRITE_LIMIT:,}) — binary NPZ has full data")

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

            # [S145-R1 v2] NPZ ACCUMULATOR — direct NPZ→NPZ merge
            # Replaces JSON accumulator (v1) — eliminates 700MB+ JSON intermediary
            # Merge policy: best per-seed score wins on conflict (TB ruling S145-R1)
            # Backward compatible: bidirectional_survivors_binary.npz same path/schema/22 fields
            # Steps 2-6 unaffected — they consume bidirectional_survivors_binary.npz exclusively
            import os as _os_s145
            import numpy as _np_s145
            _SKIP_ENC = {'constant': 0, 'variable': 1}
            _PRNG_ENC = {
                'java_lcg': 0, 'java_lcg_reverse': 1,
                'mt19937': 2, 'mt19937_reverse': 3,
                'xorshift128': 4, 'xorshift128_reverse': 5,
                'lcg32': 6, 'lcg32_reverse': 7,
                'minstd': 8, 'minstd_reverse': 9,
                'randu': 10, 'randu_reverse': 11,
            }

            def _survivors_to_arrays(survivors):
                """Convert list of survivor dicts to NPZ field arrays."""
                def _parse_skip_range(val):
                    if isinstance(val, int): return val
                    if isinstance(val, (list, tuple)) and len(val) == 2:
                        return int(val[1]) - int(val[0])
                    if isinstance(val, str) and '-' in val:
                        try: return int(val.split('-')[1]) - int(val.split('-')[0])
                        except: return 0
                    try: return int(val)
                    except: return 0
                n = len(survivors)
                return {
                    'seeds':                  _np_s145.array([s['seed'] for s in survivors], dtype=_np_s145.uint32),
                    'forward_matches':        _np_s145.array([s.get('forward_match_rate', s.get('forward_matches', 0.0)) for s in survivors], dtype=_np_s145.float32),
                    'reverse_matches':        _np_s145.array([s.get('reverse_match_rate', s.get('reverse_matches', 0.0)) for s in survivors], dtype=_np_s145.float32),
                    'window_size':            _np_s145.array([s.get('window_size', 0) for s in survivors], dtype=_np_s145.int32),
                    'offset':                 _np_s145.array([s.get('offset', 0) for s in survivors], dtype=_np_s145.int32),
                    'trial_number':           _np_s145.array([s.get('trial_number', 0) for s in survivors], dtype=_np_s145.int32),
                    'skip_min':               _np_s145.array([s.get('skip_min', 0) for s in survivors], dtype=_np_s145.int32),
                    'skip_max':               _np_s145.array([s.get('skip_max', 0) for s in survivors], dtype=_np_s145.int32),
                    'skip_range':             _np_s145.array([_parse_skip_range(s.get('skip_range', 0)) for s in survivors], dtype=_np_s145.int32),
                    'forward_count':          _np_s145.array([s.get('forward_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'reverse_count':          _np_s145.array([s.get('reverse_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'bidirectional_count':    _np_s145.array([s.get('bidirectional_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'intersection_count':     _np_s145.array([s.get('intersection_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'intersection_ratio':     _np_s145.array([s.get('intersection_ratio', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'intersection_weight':    _np_s145.array([s.get('intersection_weight', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'bidirectional_selectivity': _np_s145.array([s.get('bidirectional_selectivity', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'forward_only_count':     _np_s145.array([s.get('forward_only_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'reverse_only_count':     _np_s145.array([s.get('reverse_only_count', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'survivor_overlap_ratio': _np_s145.array([s.get('survivor_overlap_ratio', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'score':                  _np_s145.array([s.get('score', 0.0) for s in survivors], dtype=_np_s145.float32),
                    'skip_mode':              _np_s145.array([_SKIP_ENC.get(s.get('skip_mode', 'constant'), 0) for s in survivors], dtype=_np_s145.uint8),
                    'prng_type':              _np_s145.array([_PRNG_ENC.get(s.get('prng_type', s.get('prng_base', 'java_lcg')), 0) for s in survivors], dtype=_np_s145.uint8),
                }

            _accum_npz = 'bidirectional_survivors_all.npz'
            try:
                # Load prior accumulated NPZ if exists
                if _os_s145.path.exists(_accum_npz):
                    _prior_npz = _np_s145.load(_accum_npz)
                    _prior_seeds = _prior_npz['seeds'].astype(_np_s145.int64)
                    _prior_scores = _prior_npz['score'].astype(_np_s145.float32)
                    _prior_count = len(_prior_seeds)
                    # [S163-KARG] Sort prior seeds for searchsorted (vectorized lookup)
                    _prior_sort_order = _np_s145.argsort(_prior_seeds)
                    _prior_seeds_sorted = _prior_seeds[_prior_sort_order]
                else:
                    _prior_npz = None
                    _prior_seeds = _np_s145.array([], dtype=_np_s145.int64)
                    _prior_scores = _np_s145.array([], dtype=_np_s145.float32)
                    _prior_sort_order = _np_s145.array([], dtype=_np_s145.int64)
                    _prior_seeds_sorted = _np_s145.array([], dtype=_np_s145.int64)
                    _prior_count = 0

                # Convert current run survivors to arrays
                _new_arrays = _survivors_to_arrays(bidirectional_deduped)
                _new_seeds = _new_arrays['seeds'].astype(_np_s145.int64)
                _new_scores = _new_arrays['score']

                # [S163-KARG] Vectorized merge — replaces pure Python dict loop
                # Use searchsorted on sorted prior seeds for O(N log N) instead of O(N) dict
                if _prior_count > 0:
                    _pos = _np_s145.searchsorted(_prior_seeds_sorted, _new_seeds)
                    _pos_clipped = _np_s145.clip(_pos, 0, _prior_count - 1)
                    _matched = _prior_seeds_sorted[_pos_clipped] == _new_seeds
                    # Original indices in prior arrays for matched seeds
                    _prior_orig_idx = _prior_sort_order[_pos_clipped]
                    # For matched: check if new score beats prior score
                    _new_beats = _matched & (_new_scores > _prior_scores[_prior_orig_idx])
                    _keep_new_mask = (~_matched) | _new_beats
                    _keep_new = list(_np_s145.where(_keep_new_mask)[0])
                    # Superseded prior indices = matched AND new beats
                    _superseded_prior_orig = _prior_orig_idx[_new_beats]
                    _superseded_mask = _np_s145.zeros(_prior_count, dtype=bool)
                    if len(_superseded_prior_orig) > 0:
                        _superseded_mask[_superseded_prior_orig] = True
                    _keep_prior = list(_np_s145.where(~_superseded_mask)[0])
                else:
                    _keep_new = list(range(len(_new_seeds)))
                    _keep_prior = []
                    _superseded_prior_orig = _np_s145.array([], dtype=_np_s145.int64)  # [S166] no prior — nothing superseded

                # Build merged field arrays
                _FIELDS_INT32  = ['window_size','offset','trial_number','skip_min','skip_max','skip_range']
                _FIELDS_FLOAT32 = ['forward_matches','reverse_matches','forward_count','reverse_count',
                                   'bidirectional_count','intersection_count','intersection_ratio',
                                   'intersection_weight','bidirectional_selectivity','forward_only_count',
                                   'reverse_only_count','survivor_overlap_ratio','score']
                _FIELDS_UINT8  = ['skip_mode','prng_type']
                _FIELDS_UINT32 = ['seeds']

                _merged_arrays = {}
                for _fname in _FIELDS_UINT32 + _FIELDS_INT32 + _FIELDS_FLOAT32 + _FIELDS_UINT8:
                    _dtype = (_np_s145.uint32 if _fname in _FIELDS_UINT32 else
                              _np_s145.int32  if _fname in _FIELDS_INT32  else
                              _np_s145.uint8  if _fname in _FIELDS_UINT8  else
                              _np_s145.float32)
                    _parts = []
                    if _keep_prior and _prior_npz is not None and _fname in _prior_npz:
                        _parts.append(_prior_npz[_fname][_keep_prior].astype(_dtype))
                    if _keep_new and _fname in _new_arrays:
                        _parts.append(_new_arrays[_fname][_keep_new].astype(_dtype))
                    if _parts:
                        _merged_arrays[_fname] = _np_s145.concatenate(_parts)
                    else:
                        _merged_arrays[_fname] = _np_s145.array([], dtype=_dtype)

                # [S163-KARG-NPZ] TB-approved fix: backfill missing fields before sort.
                # Fields absent from older prior NPZ schemas produce size-0 arrays.
                # Backfill to zeros(seed_len) ensures rectangular NPZ — safe for all
                # downstream readers. Tagged ValueError on wrong-length non-empty fields
                # is caught by Patch 2 below and re-raised to prevent silent data loss.
                _seed_len = len(_merged_arrays['seeds'])

                def _dtype_for_field(_fn):
                    if _fn in _FIELDS_UINT32:  return _np_s145.uint32
                    if _fn in _FIELDS_INT32:   return _np_s145.int32
                    if _fn in _FIELDS_UINT8:   return _np_s145.uint8
                    return _np_s145.float32

                for _fn in _FIELDS_UINT32 + _FIELDS_INT32 + _FIELDS_FLOAT32 + _FIELDS_UINT8:
                    if _fn == 'seeds':
                        continue
                    if _fn not in _merged_arrays or len(_merged_arrays[_fn]) == 0:
                        # Missing or empty — backfill with zeros to keep schema rectangular
                        _merged_arrays[_fn] = _np_s145.zeros(
                            _seed_len, dtype=_dtype_for_field(_fn)
                        )
                    elif len(_merged_arrays[_fn]) != _seed_len:
                        raise ValueError(
                            f"[S163-KARG-NPZ] Field {_fn} length "
                            f"{len(_merged_arrays[_fn])} != seeds length {_seed_len}"
                        )
                # [END S163-KARG-NPZ backfill]

                # Sort merged arrays by seed value (all fields now seed_len — safe)
                _sort_idx = _np_s145.argsort(_merged_arrays['seeds'])
                for _fname in _merged_arrays:
                    _merged_arrays[_fname] = _merged_arrays[_fname][_sort_idx]

                _total = len(_merged_arrays['seeds'])
                _net_new = len(_keep_new)
                _superseded_count = len(_superseded_prior_orig)

                # Save accumulator NPZ
                _np_s145.savez_compressed(_accum_npz, **_merged_arrays)

                # Save as canonical bidirectional_survivors_binary.npz (Steps 2-6 input)
                _np_s145.savez_compressed('bidirectional_survivors_binary.npz', **_merged_arrays)

                print(f"\n[S145-R1 v2][NPZ ACCUMULATOR] {_total:,} total survivors across all runs")
                print(f"   Prior kept:   {len(_keep_prior):,}")
                print(f"   Net new:      +{_net_new:,}")
                print(f"   Superseded:   {_superseded_count:,} (prior seeds beaten by new score)")
                print(f"   Accumulator:  {_accum_npz}")
                print(f"✅ bidirectional_survivors_binary.npz written ({_total:,} seeds, 22 fields)")

            except ValueError as _accum_err:
                # [S163-KARG-NPZ] Re-raise only tagged schema-mismatch ValueErrors.
                # Untagged ValueErrors (from numpy/conversion code) still fall through
                # to the fallback path — they may be reasonable fallback candidates.
                # Re-raising tagged errors prevents silent data loss when
                # bidirectional_survivors.json is summary-only (JSON guard active).
                if str(_accum_err).startswith("[S163-KARG-NPZ]"):
                    raise  # schema mismatch — do not silently fall back
                print(f"\n⚠️  [S145-R1 v2][NPZ ACCUMULATOR] Failed: {_accum_err}")
                print(f"   Falling back to per-run convert_survivors_to_binary.py")
                import traceback as _tb_s145
                _tb_s145.print_exc()
                # Fallback: use original conversion path
                from subprocess import run as subprocess_run, CalledProcessError
                try:
                    subprocess_run(
                        ["python3", "convert_survivors_to_binary.py",
                         "bidirectional_survivors.json"],
                        check=True
                    )
                    print(f"✅ Fallback: converted bidirectional_survivors.json to NPZ")
                except CalledProcessError as _e:
                    print(f"❌ NPZ conversion failed: {_e}")
                    raise RuntimeError("Step 1 incomplete - NPZ conversion required for Step 2")
            except Exception as _accum_err:
                print(f"\n⚠️  [S145-R1 v2][NPZ ACCUMULATOR] Failed: {_accum_err}")
                print(f"   Falling back to per-run convert_survivors_to_binary.py")
                import traceback as _tb_s145
                _tb_s145.print_exc()
                # Fallback: use original conversion path
                from subprocess import run as subprocess_run, CalledProcessError
                try:
                    subprocess_run(
                        ["python3", "convert_survivors_to_binary.py",
                         "bidirectional_survivors.json"],
                        check=True
                    )
                    print(f"✅ Fallback: converted bidirectional_survivors.json to NPZ")
                except CalledProcessError as _e:
                    print(f"❌ NPZ conversion failed: {_e}")
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

    _orig_optimize_window = optimize_window
    def optimize_window(self, *args, **kwargs):
        # S159E TB guardrail: single final shutdown in outer finally
        try:
            return _orig_optimize_window(self, *args, **kwargs)
        finally:
            _zmq_sc = getattr(self, '_zmq_session_coord', None)
            if _zmq_sc is not None:
                try:
                    _zmq_sc.shutdown()
                    print('[S159E] ZMQ session coordinator shut down cleanly')
                except Exception as _e:
                    print(f'[S159E] ZMQ session coordinator shutdown error (non-fatal): {_e}')
                finally:
                    self._zmq_session_coord = None

    MultiGPUCoordinator.optimize_window = optimize_window
    print("✅ Window optimizer integrated into MultiGPUCoordinator")
