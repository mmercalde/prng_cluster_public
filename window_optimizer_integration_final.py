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


def _repository_state(repo_root=None) -> Tuple[str, bool]:
    """[S172 Phase-5 D3.5 §7.3] Return (commit_sha, working_tree_clean).

    Lives HERE, in the caller, and deliberately NOT inside `utils/run_finalizer`:
    the finalizer's D3.5 §12 F15 gate asserts at source level that it spawns no
    subprocess at all, so the two provenance facts arrive as explicit arguments
    on its frozen public signature instead.

    A certified generation must record `repository_tree_clean=True` — the first
    certified production baseline must not claim a commit SHA while running
    uncommitted source. Reporting the truth here is what lets the finalizer
    refuse; this helper never "cleans up" the answer.
    """
    import subprocess as _subprocess_d3_5
    root = repo_root or os.path.dirname(os.path.abspath(__file__))
    commit = _subprocess_d3_5.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    porcelain = _subprocess_d3_5.run(
        ["git", "-C", root, "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout
    return commit, (porcelain.strip() == "")


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
    Convert a PWC / ZMQ `step1_trial_populations_v2` result into a TestResult
    and update the accumulator with canonical per-mode candidate records.

    Mirrors the accumulator logic in the original run_bidirectional_test path —
    and as of S172 Phase-5 D3.25 that docstring is finally TRUE. The legacy
    sieve path appends the two modes INDEPENDENTLY at two separate sites
    (:686 constant, :788 variable) with per-mode maps and per-mode aggregates.
    This adapter used to union them (`for seed in bidi_constant | bidi_variable`)
    into ONE record labelled variable, which destroyed the constant candidate of
    any cross-mode seed before the L2 competition boundary, stamped constant-mode
    rates and constant-biased aggregates onto variable records, and emitted a
    string `skip_range` plus a fabricated scalar `"all"` sessions.

    Ordering is trial-major, mode-minor (trial N constant, then trial N
    variable), matching legacy. It is deterministic and decides no winner —
    D3.5's explicit L2 key remains authoritative.

    NOTE the deliberate asymmetry: `TestResult.bidirectional_count` still
    exposes the COMBINED constant+variable total as run telemetry, but that
    value is never copied into a record's mode-specific `bidirectional_count`.
    """
    from window_optimizer import TestResult
    from utils.canonical_records import (
        CanonicalRecordContractError,
        normalize_trial_populations,
        validate_trial_populations,
    )

    # ---- D3.25-A adapter ingress wall [C4] --------------------------------
    # Independent of the producer's own egress validation, and FIRST: nothing
    # below may touch the accumulator until the result is proven well-formed,
    # so a malformed or test-mutated result fails before even one candidate is
    # appended (G4). Two boundaries is what makes either meaningful — deleting
    # one must be caught while the other stays intact (G11).
    validate_trial_populations(pw_result, origin="adapter-ingress")

    # Direct subscript, never `.get(name, {})`: a missing v2 field is a contract
    # violation, not an empty population. (validate_trial_populations has
    # already proven every one of these keys is present.)
    fwd_map_constant = pw_result["forward_map_constant"]
    rev_map_constant = pw_result["reverse_map_constant"]
    fwd_map_variable = pw_result["forward_map_variable"]
    rev_map_variable = pw_result["reverse_map_variable"]
    bidi_constant    = pw_result["bidirectional_constant"]
    bidi_variable    = pw_result["bidirectional_variable"]

    # Telemetry only — these four lists never inform a canonical record, and no
    # map is ever reconstructed from one (REV3 §0.3, binding: PWC builds them
    # from the raw survivor sequence, ZMQ from map keys).
    fwd_records   = pw_result.get("forward_records", [])
    rev_records   = pw_result.get("reverse_records", [])
    fwd_h_records = pw_result.get("forward_records_hybrid", [])
    rev_h_records = pw_result.get("reverse_records_hybrid", [])

    total_bidi = len(bidi_constant) + len(bidi_variable)

    if accumulator is not None:
        # The adapter reads the mandatory WindowConfig attributes DIRECTLY and
        # passes validated values down — `normalize_trial_populations` never
        # receives a config object and performs no attribute lookup [C2].
        # `sessions` is a required WindowConfig field (window_optimizer.py:86),
        # so its absence is a malformed substitute object and fails closed here
        # rather than being fabricated into the legacy scalar "all".
        try:
            _sessions = config.sessions
        except AttributeError as _exc:
            raise CanonicalRecordContractError(
                f"trial {trial_number}: config {type(config).__name__} has no "
                f"'sessions'. WindowConfig declares it as a required field; the "
                f"legacy `getattr(config, 'sessions', 'all')` fallback that "
                f"silently invented a session name is removed (D3.25 §3.4)."
            ) from _exc

        constant_records, variable_records = normalize_trial_populations(
            fwd_map_constant, rev_map_constant,
            fwd_map_variable, rev_map_variable,
            window_size  = config.window_size,
            offset       = config.offset,
            skip_min     = config.skip_min,
            skip_max     = config.skip_max,
            sessions     = _sessions,
            trial_number = trial_number,
            prng_base    = prng_base,
        )
        # Trial-major, mode-minor. A seed present in BOTH modes yields TWO
        # records carrying their own mode's rates and aggregates; D1/D2
        # established that cross-mode duplication is legitimate.
        accumulator['bidirectional'].extend(constant_records)
        accumulator['bidirectional'].extend(variable_records)

        # [S166-ACCUM] Count only — full objects not retained for forward/reverse.
        accumulator['forward_count'] = accumulator.get('forward_count', 0) + len(fwd_records) + len(fwd_h_records)
        accumulator['reverse_count'] = accumulator.get('reverse_count', 0) + len(rev_records) + len(rev_h_records)

        # [S152] Flush survivors to disk as-found (incremental, threshold-gated)
        _flush_npz_incremental(accumulator, label=f"chunk/trial-{trial_number}")

    return TestResult(
        config             = config,
        forward_count      = len(fwd_map_constant),
        reverse_count      = len(rev_map_constant),
        bidirectional_count= total_bidi,
        iteration          = trial_number,
    )


def _build_test_result_from_miner(miner_result: dict, accumulator, config,
                                   prng_base: str, trial_number: int,
                                   optuna_trial=None):
    """
    TestResult for the RANGE-MINER path — candidate ingress is D6's, not D3.25's.

    D3.25 corrects the PWC/ZMQ candidate ingress and gives those two backends a
    versioned four-map contract. The miner is NOT one of those producers: it
    already builds canonical 24-field records inside the Phase-5 assembly
    engine, and D6 will append its `canonical_records_constant` /
    `canonical_records_variable` straight off the stored `MinerTrialAssembly`
    WITHOUT rerunning normalization (D3.25 REV3 §4).

    So this call site is detached from `_build_test_result_from_pw` rather than
    fed a v2-shaped result: routing miner output through the PWC/ZMQ contract is
    exactly what §4 forbids, and pushing a `serve_trial` dict through the new
    ingress wall would fail closed for the wrong reason.

    Behavior is unchanged from what the shared adapter actually did for this
    path before D3.25: `serve_trial` returns run/stripe/manifest state and none
    of the population keys, so the old `.get(..., set()/{})` reads made every
    count zero and appended nothing. That is preserved verbatim — including the
    threshold-gated flush, so flush cadence does not shift — and the miner's
    real candidates arrive with D6.

    Certification status (REV3 §4):
        PWC/ZMQ both-mode canonical candidate output   certified at D3.25
        miner  both-mode run-level candidate output    uncertified until D6
    """
    from window_optimizer import TestResult

    if accumulator is not None:
        # No candidate is appended: D6 owns miner candidate ingress. The two
        # directional counters advance by the miner's own record lists, which
        # serve_trial does not return today (hence +0), exactly as before.
        accumulator['forward_count'] = accumulator.get('forward_count', 0) + len(miner_result.get("forward_records", []))
        accumulator['reverse_count'] = accumulator.get('reverse_count', 0) + len(miner_result.get("reverse_records", []))
        _flush_npz_incremental(accumulator, label=f"chunk/trial-{trial_number}")

    return TestResult(
        config             = config,
        forward_count      = len(miner_result.get("forward_map", {})),
        reverse_count      = len(miner_result.get("reverse_map", {})),
        bidirectional_count= (len(miner_result.get("bidirectional_constant", set()))
                              + len(miner_result.get("bidirectional_variable", set()))),
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
            # Resolved WindowConfig window params. Blocker 2b: DIRECT attribute
            # access, no getattr default — window_size/offset/sessions/skip_min/
            # skip_max are REQUIRED constructor fields of WindowConfig (no defaults;
            # window_optimizer.py:85-89), so a malformed substitute object raises
            # AttributeError loudly instead of silently coercing a missing field to
            # 1/0/None and letting present-but-wrong metadata reach Phase 5.
            window_size            = config.window_size,
            sessions               = config.sessions,
            offset                 = config.offset,
            # D0 seam: the skip bounds were the one WindowConfig pair still dropped
            # here (window_size/offset/sessions were wired, skip_min/skip_max were
            # not) — thread them from the resolved WindowConfig so every published
            # miner manifest carries the real skip range, never a run_trial_miner
            # default. WindowConfig always defines both (window_optimizer.py:88-89).
            skip_min               = config.skip_min,
            skip_max               = config.skip_max,
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
        # D3.25 §4: the miner is NOT a `step1_trial_populations_v2` producer, so
        # it no longer shares the PWC/ZMQ adapter. D6 wires the miner's own
        # already-canonical records in.
        return _build_test_result_from_miner(_miner_result, accumulator, config,
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

        # ====================================================================
        # [S172 Phase-5 D3.5] The inline NPZ-accumulator block that used to sit
        # here is REPLACED by the shared finalizer `utils.run_finalizer`, which
        # every backend (legacy in-process sieve, PWC, ZMQ and — via D6 — the
        # range miner) now goes through.
        #
        # [§10] The legacy `deduplicate_survivors` helper is REMOVED, not merely
        # bypassed. It selected by `lexsort((-scores, seeds))` — seed ascending,
        # score descending, with NO trial_number and NO skip_mode key — and its
        # output fed BOTH `bidirectional_survivors.json` and the NPZ path, so
        # the JSON and the canonical NPZ could disagree about the winner for the
        # same seed. Winner selection is now solely the finalizer's explicit L2
        # key (canonical float32 score -> lowest trial_number -> constant before
        # variable, and only as a tiebreak WITHIN one trial), and the canonical
        # NPZ generation is authoritative.
        # ====================================================================
        from pathlib import Path as _Path_d3_5
        from utils.run_finalizer import (
            ALL_NPZ_NAME as _ALL_NPZ_NAME_d3_5,
            BINARY_NPZ_NAME as _BINARY_NPZ_NAME_d3_5,
            finalize_run as _finalize_run_d3_5,
        )

        # The raw current-run candidates, exactly as the backends appended them.
        # They are NOT pre-deduplicated: the finalizer validates EVERY raw
        # candidate through D3 before L2, so a malformed LOSING candidate fails
        # the run instead of vanishing during selection (D3.5 §3 [B2]).
        _raw_candidates_d3_5 = survivor_accumulator['bidirectional']

        # --------------------------------------------------------------------
        # NON-CANONICAL DIAGNOSTICS — this try/except is allowed to swallow.
        # Nothing canonical happens inside it. That separation is the binding
        # correction of D3.5 §11 [B4]: the previous code ran the whole NPZ
        # accumulator inside a broad `except Exception` that printed a warning,
        # shelled out to `convert_survivors_to_binary.py` and then returned
        # SUCCESS — so a rejected prior or a failed publication still looked
        # like a good run.
        # --------------------------------------------------------------------
        try:
            _fwd_count = survivor_accumulator.get('forward_count', 0)
            _rev_count = survivor_accumulator.get('reverse_count', 0)
            _bid_count = len(_raw_candidates_d3_5)

            print(f"\n[CANDIDATES] fwd={_fwd_count:,} (count only)  "
                  f"rev={_rev_count:,} (count only)  "
                  f"raw bidirectional candidates={_bid_count:,}")

            # [S166-ACCUM] forward/reverse full objects are not retained, so
            # these two files have been summary-only for several sessions.
            for _diag_name, _diag_count in (('forward_survivors.json', _fwd_count),
                                            ('reverse_survivors.json', _rev_count)):
                with open(_diag_name, 'w') as f:
                    json.dump({
                        "survivor_count": _diag_count,
                        "note": (f"Full survivors omitted — objects not retained; "
                                 f"see {_ALL_NPZ_NAME_d3_5}"),
                    }, f, indent=2)
                print(f"⚠️  {_diag_name}: summary only ({_diag_count:,}) — "
                      f"canonical NPZ carries the full data")

            # Telemetry only, over RAW candidates. This prints no winner and
            # decides nothing.
            if _raw_candidates_d3_5:
                sample = _raw_candidates_d3_5[0]
                print(f"\n📊 Sample raw candidate:")
                print(f"   seed: {sample['seed']}")
                print(f"   forward_match_rate: {sample.get('forward_match_rate', 'MISSING')}")
                print(f"   reverse_match_rate: {sample.get('reverse_match_rate', 'MISSING')}")
                print(f"   score: {sample.get('score', 'MISSING')}")
                print(f"   window_size: {sample['window_size']}, trial: {sample['trial_number']}")

            if test_both_modes:
                constant_count = sum(1 for s in _raw_candidates_d3_5
                                     if s.get('skip_mode') == 'constant')
                variable_count = sum(1 for s in _raw_candidates_d3_5
                                     if s.get('skip_mode') == 'variable')
                print(f"\n📈 Raw candidate skip-mode distribution:")
                print(f"   Constant skip: {constant_count} candidates")
                print(f"   Variable skip: {variable_count} candidates")

        except Exception as e:
            print(f"⚠️  Error writing non-canonical survivor diagnostics: {e}")
            import traceback
            traceback.print_exc()

        # --------------------------------------------------------------------
        # CANONICAL FINALIZATION — deliberately OUTSIDE every swallow wrapper.
        #
        # A finalizer rejection (bad prior, broken provenance chain, failed
        # publication) MUST propagate out of `optimize_window`. There is no
        # fallback writer and no subprocess conversion: if canonical
        # finalization fails, the previously certified generation stays current
        # and this function does not return `results` (D3.5 §11, gate F31).
        # --------------------------------------------------------------------
        _repo_commit_d3_5, _repo_clean_d3_5 = _repository_state()

        # From RUN CONFIGURATION, never inferred from survivor rows — an
        # executed mode may legitimately produce zero survivors (D3.5 §8a). The
        # `_hybrid` guard mirrors the variable-skip gate at the sieve call site.
        _skip_modes_d3_5 = (
            ('constant', 'variable')
            if (test_both_modes and not prng_base.endswith('_hybrid'))
            else ('constant',)
        )

        _artifact_d3_5 = _finalize_run_d3_5(
            _raw_candidates_d3_5,
            output_root=_Path_d3_5.cwd(),
            run_id=f"step1_{prng_base}_{int(seed_start)}",   # [S142-C] canonical
            prng_base=prng_base,
            skip_modes_executed=_skip_modes_d3_5,
            seed_start=int(seed_start),
            seed_count=int(seed_count),
            repository_commit=_repo_commit_d3_5,
            repository_tree_clean=_repo_clean_d3_5,
        )

        print(f"\n[S172 D3.5][GENERATION PUBLISHED] "
              f"{_artifact_d3_5.final_row_count:,} rows now current")
        print(f"   Generation:   {_artifact_d3_5.generation_id}")
        print(f"   Directory:    {_artifact_d3_5.generation_dir}")
        print(f"   Raw candidates: {_artifact_d3_5.raw_candidate_count:,}")
        print(f"   L2 winners:     {_artifact_d3_5.l2_winner_count:,}")
        print(f"   Prior rows:     {_artifact_d3_5.prior_row_count:,}")
        print(f"   Artifact sha256: {_artifact_d3_5.artifact_sha256}")
        print(f"   Sidecar  sha256: {_artifact_d3_5.sidecar_sha256}")
        print(f"   Parent generation: {_artifact_d3_5.parent_generation_id}")
        print(f"✅ {_BINARY_NPZ_NAME_d3_5} now resolves to the certified "
              f"generation ({_artifact_d3_5.final_row_count:,} seeds, 22 fields)")

        # [S172 Phase-5 D3.5 §10] `bidirectional_survivors.json` is a
        # POST-SUCCESS SUMMARY of the generation that was just certified. It is
        # NO LONGER the canonical Steps 2-6 input, and it is no longer produced
        # by an independent score-only deduplication that could disagree with
        # the NPZ. Steps 2-6 consume the canonical NPZ.
        with open('bidirectional_survivors.json', 'w') as f:
            json.dump({
                "note": (f"Post-success summary of the certified generation. "
                         f"The canonical Steps 2-6 input is "
                         f"{_BINARY_NPZ_NAME_d3_5}; this file decides no "
                         f"winner and is not independently deduplicated."),
                "generation_id":       _artifact_d3_5.generation_id,
                "generation_dir":      str(_artifact_d3_5.generation_dir),
                "run_id":              _artifact_d3_5.run_id,
                "prng_base":           _artifact_d3_5.prng_base,
                "skip_modes_executed": list(_artifact_d3_5.skip_modes_executed),
                "seed_start":          _artifact_d3_5.seed_start,
                "seed_count":          _artifact_d3_5.seed_count,
                "seed_end_exclusive":  _artifact_d3_5.seed_end_exclusive,
                "raw_candidate_count": _artifact_d3_5.raw_candidate_count,
                "l2_winner_count":     _artifact_d3_5.l2_winner_count,
                "prior_row_count":     _artifact_d3_5.prior_row_count,
                "final_row_count":     _artifact_d3_5.final_row_count,
                "artifact_sha256":     _artifact_d3_5.artifact_sha256,
                "sidecar_sha256":      _artifact_d3_5.sidecar_sha256,
                "created_at":          _artifact_d3_5.created_at,
            }, f, indent=2)
        print(f"✅ bidirectional_survivors.json written as a post-success summary")

        print(f"{'='*80}\n")

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
