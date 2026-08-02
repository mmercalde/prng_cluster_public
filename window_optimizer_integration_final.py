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
    # [S172 §4.3] The admission-timeout DEFAULT is imported from the miner rather
    # than restated here: one authority for the value, so the call site below can
    # honour a coordinator override without ever baking a literal.
    from miner import DEFAULT_WORKER_ADMISSION_TIMEOUT, run_trial_miner
except ImportError:
    run_trial_miner = None
    DEFAULT_WORKER_ADMISSION_TIMEOUT = None

# [S172 Phase-5 D6] The miner candidate-ingress adapter. Imported under the SAME
# guard shape as the runner above (a host without miner/ keeps working); the
# _use_miner gate below raises if the package is missing, so a silent no-ingress
# path is impossible.
try:
    from miner.step1_ingress import (
        MinerIngressError,
        build_assembling_sink,
        certified_paths,
        ingest_assembly,
        require_assembly,
        resolve_assembly_backend,
    )
except ImportError:
    MinerIngressError = None
    build_assembling_sink = None
    certified_paths = None
    ingest_assembly = None
    require_assembly = None
    resolve_assembly_backend = None


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
# [S172 THRESHOLD-REPAIR] Canonical directional-threshold resolution
#
# Beta Ruling 24 items 1-2, Priority-0. Optuna samples forward_threshold /
# reverse_threshold and stores them on the WindowConfig
# (window_optimizer_bayesian.py:445-452 for Route A, this file's :1835-1841 for
# Route B), but BOTH call sites discarded them before run_bidirectional_test:
#
#   Route A  test_config(... ft=bounds.default_forward_threshold ...)  — the
#            sampled value was never read; the signature default won.
#   Route B  _local_test(... forward_threshold=_local_bounds.default_* ...) —
#            the sampled value was in scope on `cfg`, on the adjacent line, and
#            was overwritten by an explicit assignment.
#
# Every trial therefore filtered at the configured default while the study
# recorded the sampled value. Route A was fixed in 3fdf434 (2026-04-30) and
# silently reverted in 2389b61 (2026-07-07) when this file was overwritten from
# a stale copy; Route B was never covered by that fix at all.
#
# The repair puts BOTH routes through this one function, so there is a single
# authority for "what threshold does this trial run at" and no downstream
# reinterpretation — the same shape the miner uses at
# miner/range_miner_coordinator.py:3410-3419. A future stale-copy overwrite of
# either call site changes observable behaviour and reds
# tests/test_s172_threshold_propagation.py, which executes the live source of
# both call sites rather than matching text against them.
# ============================================================================

_THRESHOLD_DIRECTION_ATTR = {
    'forward': 'forward_threshold',
    'reverse': 'reverse_threshold',
}


class ThresholdResolutionError(ValueError):
    """No directional threshold could be resolved for a trial. Fail closed."""


def resolve_directional_threshold(config, direction, explicit=None, default=None):
    """
    Resolve ONE trial's forward/reverse match threshold, once, in the parent.

    Precedence: explicit caller argument > config attribute > supplied default.

    `is None` is the ONLY fallback trigger. 0.0 is a legitimate threshold and
    must never be silently replaced — the `getattr(...) or default` form used by
    s172_threshold_patch.py FIX 2 would replace it, so this does not reuse that
    shape. Raises rather than inventing a value when nothing resolves.
    """
    if direction not in _THRESHOLD_DIRECTION_ATTR:
        raise ThresholdResolutionError(
            f"unknown threshold direction {direction!r} "
            f"(expected one of {sorted(_THRESHOLD_DIRECTION_ATTR)})"
        )
    if explicit is not None:
        return float(explicit)
    value = getattr(config, _THRESHOLD_DIRECTION_ATTR[direction], None)
    if value is not None:
        return float(value)
    if default is None:
        raise ThresholdResolutionError(
            f"no {direction} threshold available: config carries none and no "
            f"default was supplied — refusing to invent one"
        )
    return float(default)


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


def _miner_residues_for_config(config, dataset_path: str):
    """[S172 Phase-5 D6 correction] The RANGE-MINER path's residue derivation —
    SHARED AUTHORITY with the worker (Team Beta §4).

    `_get_residues_for_config` above never passes `sessions`, while the miner
    worker rebuilds its window WITH the session filter applied
    (`miner.range_miner_worker.load_residue_window`). For a both-sessions trial
    the filter is a no-op and the two agreed by luck; for a single-session trial
    (`sessions=['midday']` / `['evening']`) they diverged, the coordinator stamped
    a residue_sha256 the worker could not reproduce, and EVERY stripe failed the
    Blocker-6 residue check.

    The fix is not a second, session-aware copy of the derivation here — it is to
    call the SAME function the worker calls, with the session selection as an
    explicit input. One implementation of the session filter exists in the miner;
    this is a consumer of it, not a peer.

    Deliberately a separate function from `_get_residues_for_config`: the PWC and
    ZMQ call sites are out of scope for this correction pass and keep their
    existing derivation byte-for-byte.
    """
    from miner.range_miner_worker import load_residue_window
    return load_residue_window(dataset_path, config.window_size,
                               config.sessions, config.offset)


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

import sys as _sys_flush
import hashlib as _hashlib_flush
import socket as _socket_flush
import time as _time_flush
import uuid as _uuid_flush

# ═════════════════════════════════════════════════════════════════════════════
# [S172 Phase-5 D6.2] WHAT THIS IS — AND WHAT IT IS NOT
# ═════════════════════════════════════════════════════════════════════════════
# This is the CANONICAL 24-FIELD ACCUMULATOR CHECKPOINT. D6.1's predecessor was
# a NON-AUTHORITATIVE FOUR-FIELD SNAPSHOT under `s172-d6.1-four-field-v1`, and
# that is exactly why the S166 in-memory clear had to stay disabled: the D3.5
# finalizer consumes the in-memory list and requires all 24
# CANONICAL_RECORD_FIELDS, and four cannot make 24.
#
# D6.1 repaired WRITING (it never once succeeded), VISIBILITY (every failure was
# swallowed), PER-FILE ATOMIC REPLACEMENT and ISOLATION (it was aimed at paths
# it does not own). D6.2 keeps all four of those properties and adds the three
# that make the clear safe:
#   * the complete 24-field canonical state, so the state is reconstructible;
#   * two SEPARATE identities — `canonical_state_digest` (shared, content only)
#     and `member_content_digest` (per member, identity included) — so a mixed
#     pair and a tampered identity block are distinguishable failures;
#   * a run-id-only resume path, so a resumed run supplies the finalizer
#     COMPLETE 24-field input rather than the truncated stump left in memory.
#
# THE ASYMMETRIC ARCHITECTURE IS BINDING (REV5 §0, settled):
#   MEMBER A IS A MARKER / COMPATIBILITY STUB. It carries `seed`, `score` and
#   its identity block, nothing more. IT IS NEVER AN ACCUMULATOR BACKUP and no
#   path here describes or consumes it as one.
#   MEMBER B IS THE SOLE RECOVERY PAYLOAD. Loss or corruption of B is
#   unrecoverable and fails closed regardless of A.
#
# The schema, both digests, the CSR `sessions` encoding, the run-id grammar, the
# nine-row recovery matrix and reconciliation all live in `utils/checkpoint_d6_2`
# — a shared module, so the same executable definitions serve the flush, the
# resume path and the gates. `_CHECKPOINT_SCHEMA_VERSION` is imported from there
# and is NOT restated here: one authority for the marker that tells the D6.1
# four-field format apart from this one.
#
# Winner selection is STILL not decided here. Reconciliation ends in the frozen
# `_select_l2_winners`, imported from `utils.run_finalizer` and never forked.
#
# SCOPE, stated so it cannot be over-read: D6.2
# does not restore the optimizer execution cursor.
# A resumed run recovers the accumulated CANONICAL STATE and
# continues optimization under its own trial namespace; where the SEARCH had got
# to remains entirely Optuna's, and nothing here claims otherwise.
#
# PARALLELISM SCOPE (BOUNDED REPAIR §2), stated in both directions so it cannot
# be over-read OR under-read:
#   * D6.2 checkpoint recovery AND the S166 in-memory clear are certified ONLY
#     for the default single-Optuna-trial path, `n_parallel == 1`. A
#     `--resume-checkpoint` request with `n_parallel > 1` is REJECTED as the
#     first executable statement of `optimize_window`, above the NP2 block —
#     before study creation, worker launch, the [NP2-KILL] SSH to every rig, any
#     other fleet action, and any candidate admission.
#   * THAT PATH STILL DISTRIBUTES EACH SIEVE TRIAL ACROSS THE FULL GPU CLUSTER.
#     The limit is on Optuna parallelism, not on fleet use.
#   * NO NP2 CLAIM IS MADE. Not resume, not accumulator clearing. The forked
#     partition workers carry no installed D6.2 run context, and concurrent
#     partition writers cannot safely share the present checkpoint member pair;
#     that needs a separate transaction design.
# ═════════════════════════════════════════════════════════════════════════════

from utils.checkpoint_d6_2 import (                              # [S172 D6.2]
    CHECKPOINT_SCHEMA_VERSION as _CHECKPOINT_SCHEMA_VERSION,
    IDENTITY_KEYS as _CHECKPOINT_IDENTITY_KEYS,
    MEMBER_A_NAME as _CHECKPOINT_ALL_NAME,
    MEMBER_B_NAME as _CHECKPOINT_BINARY_NAME,
    CHECKPOINT_DIRNAME as _CHECKPOINT_DIRNAME,
    CheckpointError as _CheckpointError,
    RunContext as _CheckpointRunContext,
    build_run_context_digest as _build_run_context_digest,
    recover_checkpoint as _recover_checkpoint,
    reconcile as _reconcile_candidates,
    resolve_checkpoint_dir as _resolve_checkpoint_dir,
    run_context_components as _run_context_components,
    validate_new_raw_records as _validate_new_raw_records,
    validate_run_id as _validate_checkpoint_run_id,
    write_transaction as _write_checkpoint_transaction,
)

# ── [S172 Phase-5 D6.1] snapshot namespace and path conditions ───────────────
# The snapshot MUST NOT write to `bidirectional_survivors_all.npz` /
# `bidirectional_survivors_binary.npz`. Since D3.5 those two names are
# COMPATIBILITY SYMLINKS OWNED BY THE FINALIZER
# (`utils.run_finalizer._bootstrap_root_aliases`, run_finalizer.py:1400-1404),
# pointing into `.s172_accumulator/current/`. The finalizer FAILS CLOSED if a
# regular file appears at either path — "the historical root artifacts were
# removed under Ruling F, so something wrote outside the finalizer".
#
# This was invisible while the S152 write was broken (D1 below): the flush
# never actually replaced anything. Repairing the write WITHOUT relocating the
# target would have replaced both symlinks with regular four-field files, and
# the very next `finalize_run` would have raised PublicationError — permanently
# breaking generation publication. Verified by reproduction in D6.1.
#
# Beta's path conditions, each enforced below and gated:
#   1. Git-ignored (`.gitignore`: `.s172_checkpoint/`).
#   2. NOT dependent on the process CWD — resolved from a stable root
#      (`PRNG_CHECKPOINT_ROOT`, else this module's own directory). `os.chdir`
#      during a run must not move or fork the snapshot.
#   3. RUN-ISOLATED — `.s172_checkpoint/<run_id>/`, so consecutive or
#      concurrent runs cannot collide.
#   4. Temp and destination on the SAME FILESYSTEM (same directory), so
#      `os.replace` keeps its atomicity.
#   5. NEVER resolves to either finalizer-owned alias — checked at runtime,
#      fail-closed, not merely by naming convention.
#   6. The schema version above is carried in the artifact itself.
#
# [D6.2] The three names above are IMPORTED from `utils/checkpoint_d6_2` with the
# rest of the schema, so the flush and the resume path cannot drift apart about
# which files a checkpoint consists of.
_CHECKPOINT_TMP_SUFFIX  = ".flush-{pid}.tmp"
_CHECKPOINT_ROOT_ENV    = "PRNG_CHECKPOINT_ROOT"
_CHECKPOINT_RUN_ID_ENV  = "PRNG_CHECKPOINT_RUN_ID"

# The two names this snapshot may never resolve to, at any depth.
_FINALIZER_ALIAS_NAMES = ("bidirectional_survivors_all.npz",
                          "bidirectional_survivors_binary.npz")

# ── [S172 Phase-5 D6.1/D6.2] transaction identity (Beta blocker) ─────────────
# A crash between the two `os.replace` calls leaves a MIXED pair. Comparing
# seed sets does NOT detect that, and D6.1's first report wrongly claimed it
# did. Beta's counterexample: old pair holds seed 42 @ 0.40; the new
# transaction holds seed 42 @ 0.90; a crash after replacing member A leaves
# A=0.90 / B=0.40 with IDENTICAL seed sets {42}. Seed-set comparison reports
# agreement across two different transactions. The same hole exists whenever
# only the match rates change.
#
# Both members therefore carry a TRANSACTION IDENTITY, and both temporary
# artifacts are produced from ONE transaction descriptor built before either is
# written. Detection compares identity, not content shape.
#
# [D6.2] The identity block is now the eleven `IDENTITY_KEYS` imported above.
# D6.1's single `four_field_content_digest` has become TWO SEPARATE IDENTITIES —
# `canonical_state_digest` (shared, content only, no identity field) and
# `member_content_digest` (per member, covering every identity field including
# the state digest, and excluding only itself). Classification of a pair is no
# longer a local helper: `utils.checkpoint_d6_2.recover_checkpoint` implements
# the nine-row mixed-pair matrix, and the flush's own post-write check is
# `validate_installed_pair`.

# ── [S172 Phase-5 D6.2] the S166 in-memory clear is ENABLED ──────────────────
# S166 added `accumulator["bidirectional"] = []` after the flush, justified by a
# comment asserting the survivors were already safely persisted. THE GUARANTEE
# WAS DOUBLY FALSE: the write always failed (D6.1/D1) *and* four fields could
# never have backed it (D6.1/D4), because the D3.5 finalizer consumes the
# in-memory list and requires all 24 CANONICAL_RECORD_FIELDS.
#
# D6.2 makes the guarantee true rather than merely re-asserting it. The
# checkpoint now carries the complete canonical state, the state is validated
# through the three walls `finalize_run` applies before L2, BOTH members are
# read back and validated after installation, and the finalizer is fed the
# reconstructed cumulative state instead of whatever happens to be left in
# memory. Only then does the clear run.
#
# The ORDERING remains the contract and remains gated: the clear may only ever
# run strictly AFTER both replaces have returned AND after the installed pair
# validates, so no candidate is ever dropped on a path where the checkpoint did
# not land. A failure anywhere retains every candidate in memory.
_FLUSH_CLEAR_IN_MEMORY = True

# ── [S172 Phase-5 D6.1] failure observability (D2) ───────────────────────────
# The pre-D6.1 helper funnelled every failure into one indistinguishable stdout
# "Warning:", which is precisely why a total, permanent outage went unnoticed.
_flush_success_count = 0
_flush_failure_count = 0
_flush_last_error    = None
_flush_sequence      = 0

# Stable run identity, fixed at import so every flush in one process agrees.
_FLUSH_RUN_ID_DEFAULT = (f"{_socket_flush.gethostname()}-{_os_flush.getpid()}"
                         f"-{int(_time_flush.time())}")


def _flush_run_id() -> str:
    """Stable run identity — used BOTH in the path and in every member's
    identity block, so a snapshot can always be attributed to its run."""
    return _os_flush.environ.get(_CHECKPOINT_RUN_ID_ENV) or _FLUSH_RUN_ID_DEFAULT


def _flush_checkpoint_root() -> str:
    """The snapshot root — deliberately NOT `os.getcwd()`.

    Beta condition 2: a run that chdirs mid-flight must not move or fork its
    snapshot. `PRNG_CHECKPOINT_ROOT` wins when set (this is also how the gates
    keep their writes inside a temp dir); otherwise the root is THIS MODULE's
    directory, which is fixed for the life of the process.
    """
    _root = _os_flush.environ.get(_CHECKPOINT_ROOT_ENV)
    if _root:
        return _os_flush.path.abspath(_root)
    return _os_flush.path.dirname(_os_flush.path.abspath(__file__))


def _flush_checkpoint_dir() -> str:
    """`<stable root>/.s172_checkpoint/<run_id>/` — run-isolated (condition 3).

    [D6.2] Resolved through the SAME `resolve_checkpoint_dir` the resume path
    uses, so the write side and the read side cannot disagree about where a
    checkpoint lives, and so the run-id grammar (addendum §3: single component,
    no separator, no `.`/`..`) plus the realpath containment check apply to the
    WRITER too — not only to an operator-supplied selector. There is no
    newest-directory discovery here or anywhere below it.
    """
    return _resolve_checkpoint_dir(_flush_checkpoint_root(), _flush_run_id())


def _flush_assert_not_alias(path: str) -> None:
    """Beta condition 5 — fail closed, never merely by naming convention.

    Checks the resolved basename AND the realpath against both finalizer-owned
    aliases, so a symlinked snapshot directory cannot smuggle a write onto
    `.s172_accumulator/current/`.
    """
    if _os_flush.path.basename(path) in _FINALIZER_ALIAS_NAMES:
        raise RuntimeError(
            f"snapshot target {path!r} uses a finalizer-owned alias name; the "
            f"finalizer fails closed on a regular file at those paths")
    _real = _os_flush.path.realpath(path)
    for _alias in _FINALIZER_ALIAS_NAMES:
        _alias_real = _os_flush.path.realpath(
            _os_flush.path.join(_flush_checkpoint_root(), _alias))
        if _real == _alias_real:
            raise RuntimeError(
                f"snapshot target {path!r} resolves to the finalizer-owned "
                f"alias {_alias!r}")


def _flush_assert_same_filesystem(tmp_path: str, final_path: str) -> None:
    """Beta condition 4 — `os.replace` is only atomic within one filesystem."""
    _tdir = _os_flush.path.dirname(tmp_path) or "."
    _fdir = _os_flush.path.dirname(final_path) or "."
    if _os_flush.path.abspath(_tdir) != _os_flush.path.abspath(_fdir):
        raise RuntimeError(
            f"temp {tmp_path!r} and destination {final_path!r} are in "
            f"different directories — os.replace would not be atomic")
    if _os_flush.stat(_tdir).st_dev != _os_flush.stat(_fdir).st_dev:
        raise RuntimeError(
            f"temp {tmp_path!r} and destination {final_path!r} are on "
            f"different filesystems — os.replace would not be atomic")


# ═════════════════════════════════════════════════════════════════════════════
# [S172 Phase-5 D6.2] THE RUN CONTEXT — resolved ONCE at run start, then frozen
# ═════════════════════════════════════════════════════════════════════════════
# The flush can no longer be a pure function of the accumulator. It must:
#   * run the three walls `finalize_run` applies before L2 over every NEWLY
#     observed raw record (REV5 §7.2) — and two of those walls need the run's
#     declared seed interval and run identity;
#   * stamp `run_context_digest` into both members (§4.3);
#   * maintain the CUMULATIVE canonical state across flushes, because the clear
#     is now enabled and memory holds only what arrived since the last flush.
#
# All of that is exactly the information a resumed run must be able to verify it
# is resuming, so it is one object, installed once, and never re-derived
# mid-run. A flush with NO installed context FAILS CLOSED and clears nothing: an
# absent context is not a neutral "unknown", it means nobody established what
# this run is, and clearing memory against an unverifiable checkpoint is the
# precise failure D6.2 exists to prevent.
_flush_run_context = None


def _install_flush_run_context(context) -> None:
    """Install this process's D6.2 run context. Once per run."""
    global _flush_run_context, _flush_sequence
    _flush_run_context = context
    _flush_sequence = int(getattr(context, "sequence", 0))


def _active_flush_run_context():
    """The installed run context, or None. Never discovers one."""
    return _flush_run_context


def _clear_flush_run_context() -> None:
    """Drop the context — for harnesses and for a process that genuinely starts
    a new run in-process. Never called mid-run on the certifying path."""
    global _flush_run_context
    _flush_run_context = None


def _flush_next_checkpoint_id() -> str:
    """A fresh transaction id. ONE descriptor per transaction, built before
    either temp is written and stamped on BOTH members — that identity is what
    makes an interrupted replacement detectable at all."""
    return _uuid_flush.uuid4().hex


def _flush_tmp_name(final_path: str) -> str:
    """The temp target for `final_path`, in the SAME directory (so `os.replace`
    is a same-filesystem atomic rename).

    [D6.1 / D1] The pre-repair code built `<final> + ".flush.tmp"` and passed
    that NAME to `np.savez_compressed`, which appends `.npz` when the name
    lacks one — numpy wrote `...flush.tmp.npz`, and the following
    `os.replace("...flush.tmp", ...)` raised FileNotFoundError into a broad
    `except`. The helper has therefore been a silent no-op since S152.

    The name deliberately still has NO `.npz` tail: the suffix rewrite is
    defeated by the WRITE MECHANISM (an open file handle — see
    `_flush_write_npz`), not by the spelling of the name. Gating it that way is
    strictly stronger, because the property then holds for ANY future temp
    name, including one that reintroduces this exact bug.
    """
    return final_path + _CHECKPOINT_TMP_SUFFIX.format(pid=_os_flush.getpid())


def _flush_write_npz(tmp_path: str, arrays: dict) -> None:
    """Write a COMPLETE, fsynced temp file. Never touches a final name.

    `savez_compressed` is handed an OPEN FILE HANDLE, so numpy writes to
    exactly `tmp_path`: the implicit-`.npz` logic applies only to a string
    filename, never to a file object.

    The fsync is not decoration. `os.replace` is atomic with respect to the
    DIRECTORY ENTRY, but without fsync a power-loss crash can leave the renamed
    file truncated or zero-length — "atomic" without durability is not a
    checkpoint, and durability is the point of D6.1.
    """
    with open(tmp_path, "wb") as _fh:
        _np_flush.savez_compressed(_fh, **arrays)
        _fh.flush()
        _os_flush.fsync(_fh.fileno())


def _flush_fsync_dir(dir_path: str) -> None:
    """Persist the renames themselves, not just the file contents."""
    _fd = _os_flush.open(dir_path, _os_flush.O_RDONLY)
    try:
        _os_flush.fsync(_fd)
    finally:
        _os_flush.close(_fd)


def _flush_remove_temps(*paths) -> None:
    """Requirement 5: temps are removed on EVERY path, success and failure."""
    for _p in paths:
        try:
            _os_flush.unlink(_p)
        except FileNotFoundError:
            pass
        except OSError as _ue:
            print(f"[S152-FLUSH] Warning: could not remove temp {_p}: {_ue}",
                  file=_sys_flush.stderr)


def _flush_purge_stale_temps(dir_path: str) -> int:
    """Remove temp debris left behind by a CRASHED run (crash point (a)).

    A process killed mid-write cannot run its own `finally`, so its orphans are
    collected by the next flush. Only temps whose embedded pid is NO LONGER
    ALIVE are removed: `optimize_window` can run partition workers in parallel
    against one CWD, and a blind `*.tmp` sweep would delete a live sibling's
    in-flight temp. Returns the number removed.
    """
    import glob as _glob
    import re as _re_flush
    _pat = _re_flush.compile(r"\.flush-(\d+)\.tmp$")
    _n = 0
    for _p in _glob.glob(_os_flush.path.join(dir_path, "*.tmp")):
        _m = _pat.search(_p)
        if _m is None:
            continue
        _pid = int(_m.group(1))
        try:
            _os_flush.kill(_pid, 0)
            continue          # owner still alive — not ours to remove
        except ProcessLookupError:
            pass              # owner is gone: genuine orphan
        except OSError:
            continue          # e.g. EPERM — someone else's live process
        try:
            _os_flush.unlink(_p)
            _n += 1
        except OSError:
            pass
    return _n


def _flush_npz_incremental(accumulator: dict, label: str = "") -> None:
    """
    Write the CANONICAL 24-FIELD accumulator checkpoint, then clear what it
    persisted.

    [D6.2] THE BINDING ORDER (§8), and the clear is the LAST step:

        construct cumulative canonical state
        write both temporary artifacts
        fsync/close as required
        validate both temporary artifacts
        replace destination A
        replace destination B
        validate the installed pair
        only then clear the flushed in-memory entries

    A failure at ANY step leaves every candidate in memory. A mutant that clears
    between the two replaces must fail, and does: the clear is textually and
    causally after `_write_checkpoint_transaction` returns, and that function
    only returns after re-reading BOTH installed members from disk.

    [D6.2] THREE PROTECTIONS BEFORE ANY CLEAR (§7.2). Every NEWLY OBSERVED raw
    record passes the walls `finalize_run` applies before L2, in its order:
    `_validate_raw_candidates` -> `_validate_candidate_coverage` ->
    `_validate_candidate_identity`. They are IMPORTED, not duplicated. This must
    happen BEFORE reconciliation, because reconciliation compacts losers away
    and a malformed LOSING candidate must fail the run rather than vanish during
    selection.

    [D6.2] MEMBER ASYMMETRY. Member A is a MARKER / COMPATIBILITY STUB carrying
    `seed`, `score` and its identity block. It is not an accumulator backup, it
    is never merged from, and nothing here reads it as a source of state. Member
    B is the sole recovery payload. The pre-D6.2 "merge the prior member A from
    disk" step is GONE — with the clear enabled the cumulative state lives in the
    run context, seeded from member B on resume, so member A never has to be
    read back as data.

    [D6.1, retained] SEQUENTIAL-ATOMIC WITH SELF-REPAIR, and
    explicitly NOT jointly atomic. Both temps are written to completion first, then the two
    `os.replace` calls run back-to-back. Each file INDIVIDUALLY is always either
    its complete prior content or its complete new content. The PAIR is not
    atomic: a crash between the two replaces leaves member A new and member B
    old. That mixed state is detected by TRANSACTION IDENTITY, never by
    comparing seed sets, and `recover_checkpoint`'s nine-row matrix decides what
    it means. True joint atomicity needs a directory swap or a manifest, which
    is out of scope — so this documents the property the code actually keeps
    instead of claiming one it cannot.

    [D6.1, retained] COMPRESSION IS CORRECT HERE AND IS NOT GOVERNED BY D5
    §6.7.A. That ban applies to the CERTIFIED ARTIFACT written by the miner NPZ
    writer, where D5 enforces it with `_assert_stored_uncompressed` and mutant
    M6a (which reds on `compress_type=8`). The checkpoint is written under a
    different name in a different directory and is never consumed by Steps 2-6.
    Do NOT "harmonize" the two — making this uncompressed buys nothing, and
    making the artifact compressed reds D5's M6a.

    [D6.1, retained] FAILURE CONTRACT (D2) — non-fatal to the trial, never
    silent:
      * WRITE FAILURE (`OSError`) — loud ERROR on stderr with a traceback,
        counted in `_flush_failure_count`, and ALL candidates retained.
      * UNEXPECTED (any other exception, including every `CheckpointError` and
        `AccumulatorConsistencyError`) — loud UNEXPECTED ERROR on stderr with a
        traceback, counted, and ALL candidates retained.
    """
    global _flush_last_count, _flush_success_count, _flush_failure_count
    global _flush_last_error, _flush_sequence

    bidi = accumulator.get("bidirectional", [])
    current_count = len(bidi)

    new_since_last = current_count - _flush_last_count
    if new_since_last < _FLUSH_EVERY:
        return  # not enough new survivors yet

    _ctx = _active_flush_run_context()
    if _ctx is None:
        # FAIL CLOSED, LOUDLY, AND CLEAR NOTHING. An absent context is not a
        # neutral "unknown": it means no run identity, no declared seed
        # interval and no `run_context_digest`, so the three walls cannot run
        # and a written checkpoint could never be verified on resume.
        _flush_failure_count += 1
        _flush_last_error = RuntimeError("no D6.2 run context installed")
        print(f"[S172-D6.2-CHECKPOINT] ERROR: no run context is installed, so "
              f"the canonical checkpoint cannot be written or verified "
              f"(non-fatal to the trial; ALL {current_count:,} candidates "
              f"retained in memory, NOTHING cleared). "
              f"`_install_flush_run_context` must run at run start.",
              file=_sys_flush.stderr)
        return

    _ckpt_dir   = _ctx.checkpoint_dir
    _ACCUM_NPZ, _BINARY_NPZ = _ctx.member_paths()
    _tmp        = _flush_tmp_name(_ACCUM_NPZ)
    _tmp_bin    = _flush_tmp_name(_BINARY_NPZ)

    try:
        # Beta path conditions 4 and 5, checked BEFORE anything is written.
        _flush_assert_not_alias(_ACCUM_NPZ)
        _flush_assert_not_alias(_BINARY_NPZ)

        _os_flush.makedirs(_ckpt_dir, exist_ok=True)
        _flush_assert_same_filesystem(_tmp, _ACCUM_NPZ)
        _flush_assert_same_filesystem(_tmp_bin, _BINARY_NPZ)
        _flush_purge_stale_temps(_ckpt_dir)

        # ── §7.2 THE THREE WALLS, over every NEWLY OBSERVED raw record ───────
        # Before reconciliation, because reconciliation compacts losers away.
        _new_records = list(bidi)
        _validate_new_raw_records(
            _new_records,
            seed_start=_ctx.seed_start,
            seed_end_exclusive=_ctx.seed_end_exclusive,
            prng_base=_ctx.prng_base,
            skip_modes_executed=_ctx.skip_modes_executed)

        # ── §8 step 1: construct the cumulative canonical state ──────────────
        # Replay normalization, then the frozen `_select_l2_winners`. This is
        # NOT a second winner policy: step 2 collapses a bit-identical 24-field
        # replay, step 3 raises on a same-key/different-content collision, and
        # only step 4 selects — through the imported L2 key, never a local one.
        _cumulative = _reconcile_candidates(_ctx.cumulative, _new_records)

        # ── §8 steps 2-7: write, validate, replace A, replace B, validate ────
        _txn = _write_checkpoint_transaction(
            _ctx, _cumulative,
            checkpoint_id=_flush_next_checkpoint_id(),
            write_npz=_flush_write_npz,
            replace=_os_flush.replace,
            fsync_dir=_flush_fsync_dir,
            tmp_name=_flush_tmp_name)
        _flush_sequence = _txn["checkpoint_sequence"]

        # ── §8 step 8: ONLY NOW is the on-disk pair complete and verified ────
        _flush_last_count     = current_count
        _flush_success_count += 1

        if _FLUSH_CLEAR_IN_MEMORY:
            # [S166 / D6.2] The POSITION is the contract and is gated: the clear
            # runs strictly AFTER both replaces have returned AND after the
            # installed pair has been read back and validated, so no candidate
            # is ever dropped on a path where the checkpoint did not land. What
            # is cleared is now recoverable — that is the whole of D6.2.
            accumulator["bidirectional"] = []
            _flush_last_count = 0

        _tag = f" [{label}]" if label else ""
        print(
            f"[S172-D6.2-CHECKPOINT]{_tag} canonical checkpoint written: "
            f"{_txn['logical_candidate_count']:,} canonical records "
            f"(+{new_since_last} raw this flush, threshold={_FLUSH_EVERY}, "
            f"seq={_txn['checkpoint_sequence']}, "
            f"state={_txn['canonical_state_digest'][:12]}…, "
            f"cleared={'yes' if _FLUSH_CLEAR_IN_MEMORY else 'no'})"
        )

    except OSError as _fe:
        # WRITE FAILURE tier — recoverable in kind (ENOSPC, EACCES, EIO), but
        # never silent. ALL candidates stay in memory.
        import traceback as _tb_flush
        _flush_failure_count += 1
        _flush_last_error = _fe
        print(f"[S172-D6.2-CHECKPOINT] ERROR: checkpoint write FAILED "
              f"(non-fatal to the trial; ALL {current_count:,} candidates "
              f"retained in memory, NOTHING cleared): {_fe!r}",
              file=_sys_flush.stderr)
        _tb_flush.print_exc(file=_sys_flush.stderr)

    except Exception as _fe:
        # UNEXPECTED tier — a contract or programming error, not a disk
        # condition. Loudest of the three, still non-fatal to the trial. Every
        # CheckpointError and every AccumulatorConsistencyError lands here.
        import traceback as _tb_flush
        _flush_failure_count += 1
        _flush_last_error = _fe
        print(f"[S172-D6.2-CHECKPOINT] UNEXPECTED ERROR: canonical checkpoint "
              f"raised {type(_fe).__name__} (non-fatal to the trial; ALL "
              f"{current_count:,} candidates retained in memory, NOTHING "
              f"cleared): {_fe!r}", file=_sys_flush.stderr)
        _tb_flush.print_exc(file=_sys_flush.stderr)

    finally:
        # Requirement 5 — temps removed on EVERY path, success and failure.
        _flush_remove_temps(_tmp, _tmp_bin)


def _checkpoint_finalizer_input(accumulator: dict) -> list:
    """The COMPLETE 24-field candidate list `finalize_run` must receive (§8).

    With the clear enabled, `accumulator['bidirectional']` is only the tail that
    has arrived since the last checkpoint — the truncated stump. The finalizer's
    input is the cumulative canonical state reconciled with that tail, which is
    what makes "the finalizer still receives complete 24-field input via the
    resume path" a property of the code rather than a claim about it.

    With NO context installed nothing was ever cleared, so the raw list IS the
    complete input and is returned unchanged.

    §4.5 wording, deliberately: the finalizer's `raw_candidate_count` becomes
    *the records supplied to the finalizer by the resumed execution* — neither
    the original process's raw count nor a cumulative count across all
    pre-compaction observations. NO SIDECAR-FIELD PARITY IS CLAIMED.
    """
    _tail = list(accumulator.get("bidirectional", []))
    _ctx = _active_flush_run_context()
    if _ctx is None:
        return _tail
    # The same three walls, over the records that never reached a checkpoint —
    # otherwise a malformed loser in the tail could vanish in reconciliation
    # instead of failing the run.
    _validate_new_raw_records(
        _tail,
        seed_start=_ctx.seed_start,
        seed_end_exclusive=_ctx.seed_end_exclusive,
        prng_base=_ctx.prng_base,
        skip_modes_executed=_ctx.skip_modes_executed)
    return _reconcile_candidates(_ctx.cumulative, _tail)


class CheckpointResumeError(RuntimeError):
    """[S172 D6.2] A resume request cannot be honoured. Always fail-closed.

    Distinct from `utils.checkpoint_d6_2.CheckpointError`: that family covers
    the artifact (schema, identity, recovery), this one covers the REQUEST —
    an unusable combination of controls, or a trial namespace that would
    manufacture corruption.
    """


def _checkpoint_dataset_identity(dataset_path: str):
    """The run's dataset IDENTITY and DIGEST for `run_context_digest` (§4.3).

    Prefers the P0.5 frozen identity, which is the run's authority: resolved
    once at run start, immutable, and carrying the version id the pointer
    manifest published. Falls back to deriving the digest from the file itself
    only when this process never froze one (a harness, a direct call) — never to
    a default, and never to a value that would make two different datasets look
    like one.

    Returns `(version_id, filename, sha256)`. The ABSOLUTE PATH IS DELIBERATELY
    NOT A COMPONENT: §4.3 excludes a mutable path, and the path is not part of
    what makes two runs the same run — the digest is.
    """
    from miner.dataset_authority import get_frozen_dataset, sha256_file
    _frozen = get_frozen_dataset()
    if _frozen is not None and _os_flush.path.abspath(dataset_path) == _frozen.path:
        return _frozen.version_id, _frozen.filename, _frozen.sha256
    return (None, _os_flush.path.basename(dataset_path),
            sha256_file(dataset_path))


def _checkpoint_execution_set_id():
    """The frozen execution set's `set_id`, or the CANONICAL NULL (§4.3).

    `active_execution_set()` is the CONSUMER api and every call counts as a
    consumer read — including a `None` read, which is the case that matters
    (Beta's freeze-after-read retraction). Both production entry points freeze
    before reaching here, so `None` means "inapplicable", never "not yet".
    """
    try:
        from execution_set import active_execution_set
    except ImportError:                                     # pragma: no cover
        return None
    _set = active_execution_set()
    return None if _set is None else _set.set_id()


def _prepare_checkpoint_run_context(*, dataset_path: str, prng_base: str,
                                    skip_modes_executed, seed_start: int,
                                    seed_count: int, resume_checkpoint: str,
                                    resume_study: bool):
    """Build the D6.2 run context, honouring §4.4's combination matrix.

    Returns `(context, resume_record_ordinal_floor_or_None)` — the second value
    is the maximum PERSISTED RECORD ORDINAL recovered from the checkpoint
    (`trial_number` in the canonical record domain, 1-based). It is NOT an
    Optuna trial number and has no arithmetic relationship to one; its sole
    consumer is the process-local record counter in `optimize_window`
    (BOUNDED REPAIR §1.3.2).

    §4.4 — THE TRIAL-NUMBER COLLISION. `trial_number` is part of the replay key
    `(seed, trial_number, skip_mode)`. A checkpoint-only resume with a FRESH
    Optuna study restarts trial numbering, so a new record can collide with a
    recovered one on the same key with different canonical contents — which
    §6.1 correctly raises as corruption. A RESTART WOULD MANUFACTURE CORRUPTION,
    so that combination is rejected here, BEFORE a single new candidate can be
    admitted:

        resume_checkpoint  resume_study   behaviour
        ------------------------------------------------------------------
        no                 no             normal fresh run
        no                 yes            existing Optuna behaviour, unchanged
        yes                yes            continue, with the persisted record
                                          ordinal continuing above the
                                          recovered maximum (the counter in
                                          `optimize_window`), and the resumed
                                          Optuna study PROVEN to have been
                                          loaded rather than silently created
                                          fresh (BOUNDED REPAIR §1.3.5)
        yes                no             MUST NOT begin new trials -> rejected
                                          with a specific error, because this
                                          codebase has no reconstruct/finalize-
                                          only surface to offer instead

    "Independent controls" means neither argument aliases or implicitly enables
    the other. It does NOT mean every combination may continue optimization —
    and nothing here silently turns one control on because the other was set.
    """
    _version_id, _filename, _sha = _checkpoint_dataset_identity(dataset_path)
    _commit, _clean = _repository_state()
    _components = _run_context_components(
        dataset_version_id=_version_id,
        dataset_filename=_filename,
        dataset_sha256=_sha,
        repository_commit=_commit,
        prng_base=prng_base,
        skip_modes_executed=tuple(skip_modes_executed),
        seed_start=int(seed_start),
        seed_count=int(seed_count),
        execution_set_id=_checkpoint_execution_set_id(),
    )
    _digest = _build_run_context_digest(_components)

    if not resume_checkpoint:
        # Rows 1 and 2 of the matrix: a fresh checkpoint under THIS process's
        # run id. `resume_study` keeps its existing Optuna behaviour, untouched.
        _run_id = _validate_checkpoint_run_id(_flush_run_id())
        _ctx = _CheckpointRunContext(
            run_id=_run_id,
            checkpoint_dir=_resolve_checkpoint_dir(_flush_checkpoint_root(),
                                                   _run_id),
            run_context_digest=_digest, prng_base=prng_base,
            skip_modes_executed=tuple(skip_modes_executed),
            seed_start=int(seed_start), seed_count=int(seed_count),
            components=_components)
        print(f"[S172-D6.2-CHECKPOINT] fresh run context: run_id={_run_id} "
              f"context={_digest[:12]}… dir={_ctx.checkpoint_dir}")
        return _ctx, None

    # ---- resume requested --------------------------------------------------
    if not resume_study:
        # Row 4. Rejected BEFORE optimization, with a specific error.
        raise CheckpointResumeError(
            f"resume_checkpoint={resume_checkpoint!r} was requested with "
            f"resume_study=False. A checkpoint-only resume MUST NOT begin new "
            f"trials: a fresh Optuna study restarts trial numbering, and "
            f"trial_number is part of the replay key (seed, trial_number, "
            f"skip_mode), so a new record would collide with a recovered one "
            f"under the same key with different canonical contents — a restart "
            f"would MANUFACTURE the corruption §6.1 raises. Reconstructing or "
            f"finalizing the recovered accumulator without optimizing is not "
            f"offered here because no such surface exists in this entrypoint; "
            f"pass --resume-study together with --resume-checkpoint, and the "
            f"resumed study must begin above the recovered trial namespace."
        )

    _run_id = _validate_checkpoint_run_id(resume_checkpoint)
    _dir = _resolve_checkpoint_dir(_flush_checkpoint_root(), _run_id)
    _outcome = _recover_checkpoint(_dir, run_id=_run_id,
                                   run_context_digest=_digest)

    _ctx = _CheckpointRunContext(
        run_id=_run_id, checkpoint_dir=_dir, run_context_digest=_digest,
        prng_base=prng_base, skip_modes_executed=tuple(skip_modes_executed),
        seed_start=int(seed_start), seed_count=int(seed_count),
        components=_components,
        # §4.6 — the NEXT sequence exceeds the highest STRUCTURALLY VALID
        # sequence observed in either member, including a discarded newer A
        # marker. `write_transaction` increments, so the context carries
        # `next - 1`.
        sequence=int(_outcome.next_sequence) - 1,
        cumulative=[dict(r) for r in _outcome.records],
        resume_provenance=_outcome.provenance())
    _install_flush_run_context(_ctx)

    # [BOUNDED REPAIR §1.3.1] The maximum recovered RECORD ORDINAL. Named for
    # what it is: `trial_number` is a canonical RECORD field, 1-based, produced
    # by the process-local counter in `optimize_window` — never `trial.number`.
    _record_ordinal_floor = max(
        (int(r["trial_number"]) for r in _outcome.records), default=None)
    print(f"[S172-D6.2-CHECKPOINT] RESUMED run_id={_run_id} "
          f"row={_outcome.row} records={len(_outcome.records):,} "
          f"state={_outcome.canonical_state_digest[:12]}… "
          f"next_sequence={_outcome.next_sequence} "
          f"resume_record_ordinal_floor={_record_ordinal_floor}")
    print(f"[S172-D6.2-CHECKPOINT] the optimizer execution cursor is NOT "
          f"restored — D6.2 does not claim it (REV5 §0).")

    if _outcome.repair_pair:
        # §5 — a fresh pair is installed and validated BEFORE optimization
        # continues. Rows 1, 2, 4 and 5 all reach here; row 6 (a consistent
        # same-transaction pair) does not, because there is nothing to repair.
        _install_repaired_checkpoint_pair(_ctx, {})
    _write_resume_provenance(_ctx)
    return _ctx, _record_ordinal_floor


#: §4.5 — where the durable resumed-run provenance is persisted. A sibling of
#: the checkpoint members inside the run-isolated directory, so it travels with
#: the thing it describes and is removed with it; never a finalizer-owned path.
_RESUME_PROVENANCE_NAME = "resume_provenance.json"


def _write_resume_provenance(context) -> None:
    """§4.5 — record the resumed-run provenance DURABLY, at minimum:
    recovered checkpoint run id · checkpoint id and sequence ·
    `canonical_state_digest` · recovered canonical-record count.

    Written with the same fsync-then-atomic-replace discipline as the members,
    because provenance that can be lost by a crash is not provenance. It is
    ALSO echoed into the finalizer's post-success summary by `optimize_window`,
    so a reader of the certified generation can find it without knowing the
    checkpoint directory exists.
    """
    if context.resume_provenance is None:
        return
    _path = _os_flush.path.join(context.checkpoint_dir,
                                _RESUME_PROVENANCE_NAME)
    _tmp = _flush_tmp_name(_path)
    _payload = dict(context.resume_provenance)
    _payload["run_context_digest"] = context.run_context_digest
    _payload["run_context_components"] = context.components
    try:
        with open(_tmp, "w", encoding="utf-8") as _fh:
            json.dump(_payload, _fh, indent=2, sort_keys=True)
            _fh.flush()
            _os_flush.fsync(_fh.fileno())
        _os_flush.replace(_tmp, _path)
        _flush_fsync_dir(context.checkpoint_dir)
    finally:
        _flush_remove_temps(_tmp)
    print(f"[S172-D6.2-CHECKPOINT] resume provenance written: {_path}")


def _install_repaired_checkpoint_pair(context, accumulator: dict) -> None:
    """§5 — recovery installs and validates a FRESH PAIR before optimization
    continues, sequenced per §4.6.

    Called only on the resume path. It writes the recovered state straight back
    out as a complete, validated transaction, which is what repairs a mixed pair
    (row 2, 4 and 5) rather than leaving the run to trust a half-installed one.
    It does NOT clear anything: the accumulator is empty at this point and there
    is nothing this transaction has persisted on its behalf.
    """
    _os_flush.makedirs(context.checkpoint_dir, exist_ok=True)
    _path_a, _path_b = context.member_paths()
    _flush_assert_not_alias(_path_a)
    _flush_assert_not_alias(_path_b)
    _tmp_a, _tmp_b = _flush_tmp_name(_path_a), _flush_tmp_name(_path_b)
    try:
        _flush_assert_same_filesystem(_tmp_a, _path_a)
        _flush_assert_same_filesystem(_tmp_b, _path_b)
        _txn = _write_checkpoint_transaction(
            context, context.cumulative,
            checkpoint_id=_flush_next_checkpoint_id(),
            write_npz=_flush_write_npz, replace=_os_flush.replace,
            fsync_dir=_flush_fsync_dir, tmp_name=_flush_tmp_name)
    finally:
        _flush_remove_temps(_tmp_a, _tmp_b)
    print(f"[S172-D6.2-CHECKPOINT] repaired pair installed and validated at "
          f"sequence {_txn['checkpoint_sequence']} "
          f"({_txn['logical_candidate_count']:,} canonical records)")


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
                                   optuna_trial=None, phase5_sink=None):
    """
    TestResult for the RANGE-MINER path — candidate ingress is D6's, not D3.25's.

    D3.25 corrects the PWC/ZMQ candidate ingress and gives those two backends a
    versioned four-map contract. The miner is NOT one of those producers: it
    already builds canonical 24-field records inside the Phase-5 assembly
    engine, and D6 appends its `canonical_records_constant` /
    `canonical_records_variable` straight off the stored `MinerTrialAssembly`
    WITHOUT rerunning normalization (D3.25 REV3 §4).

    So this call site is detached from `_build_test_result_from_pw` rather than
    fed a v2-shaped result: routing miner output through the PWC/ZMQ contract is
    exactly what §4 forbids, and pushing a `serve_trial` dict through the new
    ingress wall would fail closed for the wrong reason.

    [S172 Phase-5 D6] THE GAP IS NOW CLOSED. Pre-D6 this function read
    `serve_trial`'s dict with `.get(..., set()/{})` — and `serve_trial` returns
    run/stripe/manifest state and none of the population keys, so every count
    was zero and nothing was accumulated. D6 stops reading the serve dict for
    populations entirely and reads the STORED `MinerTrialAssembly` instead,
    fetched from the Phase-5 sink by `run_id`:

        * candidates come from `canonical_records_constant` / `_variable`
          straight off the assembly, with NO re-normalization and without
          touching the PWC/ZMQ D3.25 ingress (that routing is what REV3 §4
          forbids — the miner is not a `step1_trial_populations_v2` producer);
        * the two directional counters advance by the assembly's REAL
          `directional_counts`, not by `+0`;
        * an absent assembly RAISES (`MinerIngressError`) rather than
          reproducing the old, indistinguishable-from-empty zero;
        * the threshold-gated flush stays exactly where it was, called once per
          trial after the append — so flush cadence does not shift, which is the
          invariant this docstring carried through D3.25.

    The returned `TestResult` shape is unchanged: the same four fields Step 1
    already consumes, computed the same way `_build_test_result_from_pw`
    computes them (constant maps for the two directional counts, constant +
    variable for bidirectional). Only the values become real.

    Certification status (REV3 §4):
        PWC/ZMQ both-mode canonical candidate output   certified at D3.25
        miner  both-mode run-level candidate output    certified at D6
    """
    from window_optimizer import TestResult

    if require_assembly is None or ingest_assembly is None:
        raise ImportError(
            "miner.step1_ingress not found — the RANGE-MINER path cannot "
            "ingest candidates. D6 refuses to fall back to the pre-D6 "
            "no-candidate/+0 behaviour, which is indistinguishable from a real "
            "empty trial."
        )

    # [S172 D6 correction, Beta commit ruling] THRESHOLD-PROVENANCE WALL.
    # Checked BEFORE candidate ingress and BEFORE any accumulator mutation, so a
    # trial whose kernel filter is unproven cannot contribute a single candidate,
    # let alone reach finalize_run. The parent's fail-closed gate normally aborts
    # such a trial inside serve_trial, so reaching here unvalidated means the
    # gate was bypassed entirely — which is exactly the case that must not
    # silently proceed. An ABSENT flag is NOT a neutral "unknown": it means the
    # physical evidence was never checked, so it is refused like a False.
    _prov = miner_result.get("threshold_provenance") if isinstance(
        miner_result, dict) else None
    if not (isinstance(_prov, dict) and _prov.get("validated") is True):
        raise MinerIngressError(
            "RANGE-MINER trial reached candidate ingress without a VALIDATED "
            "threshold provenance record "
            f"(threshold_provenance={_prov!r}). D6 refuses to ingest candidates, "
            "mutate the accumulator or certify a generation for a trial whose "
            "effective sieve threshold was never proven to match the requested "
            "one — that is the whole claim D6 makes."
        )

    # Fail closed NEXT: nothing touches the accumulator until this trial is
    # proven to have a committed assembly (mirrors the PWC/ZMQ adapter's
    # ingress-wall-before-append ordering, without sharing its wall).
    _assembly = require_assembly(phase5_sink, miner_result,
                                 trial_number=trial_number)

    _counts = ingest_assembly(_assembly, accumulator)

    if accumulator is not None:
        # [S152] Same call, same place, same cadence as every other backend.
        _flush_npz_incremental(accumulator, label=f"chunk/trial-{trial_number}")

    return TestResult(
        config             = config,
        forward_count      = _counts.forward_constant,
        reverse_count      = _counts.reverse_constant,
        bidirectional_count= _counts.bidirectional_total,
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
        # ----------------------------------------------------------------
        # [S172 Phase-5 D6] The Phase-5 sink, built around the CONFIGURED
        # assembly backend. Pre-D6 this call passed no `phase5_sink`, so the
        # coordinator's L6 boundary was wired to None and the trial performed
        # no Phase-5 assembly at all — which is why there was never anything
        # to ingest.
        #
        # Backend selection is a coordinator attribute so it follows the same
        # §12.4 precedence as every other knob. `assembly_backend=None` (the
        # normal case) resolves to `serial_reference` inside
        # `resolve_assembly_backend`; `process_sharded` is selectable by
        # explicit name + pool_size (via `assembly_backend_options`) and is
        # NEVER the default — Phase 6 owns its promotion.
        #
        # One sink per trial: the coordinator, ledger and run_id are per-trial
        # too, and the sink's retained-manifest retry contract (§4.0) is
        # scoped to exactly that lifetime.
        # ----------------------------------------------------------------
        if build_assembling_sink is None or resolve_assembly_backend is None:
            raise ImportError(
                "miner.step1_ingress not found — cannot use --use-range-miner. "
                "Without D6's ingress adapter the miner path would run, append "
                "no candidates and report +0, which is indistinguishable from "
                "a real empty trial."
            )
        _miner_backend = resolve_assembly_backend(
            getattr(coordinator, 'assembly_backend', None),
            **(getattr(coordinator, 'assembly_backend_options', None) or {}),
        )
        _miner_sink = build_assembling_sink(_miner_backend)
        _miner_result = run_trial_miner(
            coordinator_cfg        = getattr(coordinator, 'config_file', 'distributed_config.json'),
            config                 = config,
            trial_number           = trial_number,
            prng_base              = prng_base,
            # D6 correction: session-aware, shared with the worker's own
            # derivation (see _miner_residues_for_config). The pre-D6 call to
            # _get_residues_for_config dropped config.sessions and broke every
            # single-session trial on the residue_sha256 check.
            residues               = _miner_residues_for_config(config, dataset_path),
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
            # [S172 §4.3 admission liveness, Beta Ruling 1] The serve timeout above
            # stays UNBOUNDED; this bounds only the pre-assignment wait for the
            # expected worker pool, so a fleet that never comes up (or a loss that
            # crosses the threshold before a stage is assigned) fails explicitly
            # with run/stage/expected/eligible instead of hanging forever. Resolved
            # from the coordinator like every other knob on this call — never baked
            # in here — defaulting to the miner's own 180s constant (the PWC
            # readiness window) rather than a literal repeated at this call site.
            worker_admission_timeout = getattr(
                coordinator, 'worker_admission_timeout',
                DEFAULT_WORKER_ADMISSION_TIMEOUT),
            # [S172 Phase-5 D6] the L6 Phase-5 boundary — assembly happens on
            # the coordinator's commit, and the result is fetched below by
            # run_id. Passing None here is what made this path inert.
            phase5_sink            = _miner_sink,
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
        # already-canonical records in, off the sink's stored assembly.
        return _build_test_result_from_miner(_miner_result, accumulator, config,
                                             prng_base, trial_number, optuna_trial,
                                             phase5_sink=_miner_sink)
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
                                   BidirectionalCountScorer,
                                   require_supported_strategy)   # [S178 P0-2]

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
                        trse_context_file: str = 'trse_context.json',  # S123 TRSE thread
                        resume_checkpoint: str = ''):  # [S172 D6.2] hop 3 of 3
        # ── [S172 Phase-5 D6.2 §4.1] THE SELECTOR IS A RUN ID. ONE API. ──────
        # `resume_checkpoint` is a CHECKPOINT RUN ID, never a path and never a
        # handle. Empty means no resume. It is resolved EXCLUSIVELY beneath
        # `.s172_checkpoint/<run_id>/`, with no absolute path, no `..`, no
        # newest-directory discovery at any layer, and no mutable path in
        # `run_context_digest`. Addendum §3 adds the grammar wall: a single
        # opaque component, because path confinement alone would still let
        # `foo/bar` behave like a handle — the exact two-API ambiguity §4.1
        # exists to close.
        #
        # THIS IS HOP 3 OF 3 (§4.2). Adding the parameter alone leaves the
        # resume path dead, which is the `Advisor -> strategy_recommendation.json
        # -> WATCHER` pattern and the TRSE F1 manifest drift. The other two:
        #   hop 1  agent_manifests/window_optimizer.json -> default_params
        #          (WATCHER's step-scoped filter DROPS an undeclared key —
        #           agents/watcher_agent.py:1312 `if key in allowed_params`)
        #   hop 2  window_optimizer.py -> coordinator.optimize_window(...) kwargs
        #
        # ── [S172 D6.2 BOUNDED REPAIR §2] D6.2 IS SCOPED TO n_parallel == 1 ──
        # THIS REJECTION MUST REMAIN THE FIRST EXECUTABLE STATEMENT OF THIS
        # METHOD. `_prepare_checkpoint_run_context` — where the §4.4 combination
        # matrix, the run-context digest and the checkpoint recovery all live —
        # runs roughly 600 lines BELOW, after the `n_parallel > 1` block has
        # already created the shared Optuna study, SSH'd every AMD rig
        # ([NP2-KILL]), cleaned TCP ports and forked its partition processes.
        # On that path a checkpoint resume would be validated only AFTER the
        # fleet had been driven and the study mutated.
        #
        # The forked partition workers also carry NO installed D6.2 run context,
        # so their flush attempts cannot clear the in-memory accumulator and the
        # S166 OOM protection is not real there; and concurrent partition
        # writers cannot safely share the single checkpoint member pair, which
        # needs a separate transaction design. D6.2 therefore makes NO NP2
        # claim of any kind — not resume, not accumulator clearing.
        #
        # The scope limit is on OPTUNA PARALLELISM ONLY. The certified
        # `n_parallel == 1` path still distributes every sieve trial across the
        # full GPU cluster through the coordinator.
        if resume_checkpoint and int(n_parallel) > 1:
            raise CheckpointResumeError(
                f"resume_checkpoint={resume_checkpoint!r} was requested with "
                f"n_parallel={n_parallel}. S172 D6.2 checkpoint recovery and "
                f"the S166 in-memory clear are certified ONLY for the default "
                f"single-Optuna-trial path (n_parallel == 1); that path still "
                f"distributes each sieve trial across the full GPU cluster. "
                f"Under n_parallel > 1 the resume would be validated only "
                f"AFTER study creation, the [NP2-KILL] SSH to every rig, port "
                f"cleanup and the partition fork; the forked workers would "
                f"carry no installed D6.2 run context, so their flushes could "
                f"not clear memory; and concurrent partition writers cannot "
                f"safely share one checkpoint member pair. Rejected here, "
                f"before study creation, worker launch, any fleet action or "
                f"any candidate admission — no process was started. Re-run "
                f"with --n-parallel 1 to use --resume-checkpoint.")
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
                    # [S172 THRESHOLD-REPAIR R2] same canonical resolver the
                    # single-process route uses — this worker is a separate
                    # process and re-imports the module, so there is exactly one
                    # implementation, not a partition-local copy.
                    from window_optimizer_integration_final import (
                        resolve_directional_threshold as _resolve_dt,
                    )
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
                            # [S172 THRESHOLD-REPAIR R2] Route B (--n-parallel > 1).
                            # These two lines used to read
                            # `_local_bounds.default_forward_threshold` /
                            # `default_reverse_threshold`, explicitly discarding the
                            # sampled values that _worker_obj had just put on `cfg`
                            # (:1835-1841 below, in scope, on the adjacent line).
                            # 3fdf434 never covered this route. Same single authority
                            # as Route A.
                            forward_threshold=_resolve_dt(
                                cfg, 'forward', None,
                                _local_bounds.default_forward_threshold),
                            reverse_threshold=_resolve_dt(
                                cfg, 'reverse', None,
                                _local_bounds.default_reverse_threshold),
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

        # ════════════════════════════════════════════════════════════════════
        # [S172 Phase-5 D6.2] RUN CONTEXT + RESUME — before the first trial
        # ════════════════════════════════════════════════════════════════════
        # Resolved ONCE here and frozen, because everything below depends on it:
        # the three finalizer walls the flush runs, the `run_context_digest`
        # stamped into both members, and the cumulative canonical state the
        # finalizer is ultimately fed. Nothing here is re-derived mid-run.
        _skip_modes_d6_2 = (
            ('constant', 'variable')
            if (test_both_modes and not prng_base.endswith('_hybrid'))
            else ('constant',)
        )
        _d6_2_context, _d6_2_resume_record_ordinal_floor = \
            _prepare_checkpoint_run_context(
                dataset_path=dataset_path,
                prng_base=prng_base,
                skip_modes_executed=_skip_modes_d6_2,
                seed_start=int(seed_start),
                seed_count=int(seed_count),
                resume_checkpoint=resume_checkpoint,
                resume_study=resume_study,
            )
        _install_flush_run_context(_d6_2_context)

        if not _np2_complete:
            optimizer = WindowOptimizer(self, dataset_path)
            bounds = SearchBounds.from_config()
            # [S172 D6.2 §4.4] The record's `trial_number` comes from THIS
            # counter (see `test_config` below), not from `trial.number`. It is
            # a process-local RECORD ORDINAL, 1-based, that restarts at 1 every
            # fresh run, and it is the value that lands in the replay key
            # `(seed, trial_number, skip_mode)`. On a resume it therefore has to
            # CONTINUE above the recovered maximum rather than restart, or the
            # first new trial would collide with recovered trial 1 under a
            # different canonical content — the corruption §4.4 forbids
            # manufacturing.
            #
            # This is NOT "offsetting or rewriting an Optuna trial number": no
            # Optuna number is read, written or shifted here. It is the local
            # record ordinal resuming its own history instead of pretending the
            # recovered trials never happened, and it does not restore the
            # optimizer execution cursor (REV5 §0) — where the SEARCH is remains
            # entirely Optuna's.
            #
            # [BOUNDED REPAIR §1.3.2] THIS IS THE FLOOR'S ONLY CONSUMER. The
            # value is a RECORD ORDINAL, which is why it is named one: the old
            # name (`_resume_trial_floor`) asserted a relationship to Optuna
            # trial numbers that does not exist, and that false name is what
            # produced the off-by-one that rejected every normal resume. It is
            # never forwarded to the study body and never compared against
            # `trial.number`.
            trial_counter = {'count': int(_d6_2_resume_record_ordinal_floor or 0)}
            if _d6_2_resume_record_ordinal_floor:
                print(f"[S172-D6.2-CHECKPOINT] record trial ordinal begins "
                      f"above the recovered namespace: next trial_number = "
                      f"{int(_d6_2_resume_record_ordinal_floor) + 1}")

        def test_config(config,
                        ss=seed_start, sc=seed_count,
                        ft=None, rt=None,
                        optuna_trial=None):  # S115 M2, [S172 THRESHOLD-REPAIR R1]
            # [S172 THRESHOLD-REPAIR R1] Route A (single-process, --n-parallel 1,
            # the default). `ft`/`rt` used to be BOUND AT DEF TIME to
            # bounds.default_forward_threshold / default_reverse_threshold, so the
            # sampled values riding on `config` were never read and every trial ran
            # at 0.30/0.30 while the study recorded the suggestion. Resolution now
            # happens at CALL time, config first — one authority, see
            # resolve_directional_threshold above. The caller
            # (window_optimizer.py:481-482) passes `config` positionally and needs
            # no change: the values are already on the object it hands over, and
            # adding a parallel ft/rt argument there would create a second
            # authority for the same quantity.
            ft = resolve_directional_threshold(config, 'forward', ft, bounds.default_forward_threshold)
            rt = resolve_directional_threshold(config, 'reverse', rt, bounds.default_reverse_threshold)
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

        # [S178 P0-2] Fail closed. This was `strategy_map.get(strategy_name,
        # RandomSearch())`, which made the UNCALLABLE RandomSearch the silent
        # default for any unrecognised strategy name — the same "a request
        # becomes a different algorithm" anti-pattern as the Optuna fallback.
        # require_supported_strategy raises StrategyContractError on both an
        # unknown name and a known-but-uncallable one; the instance still comes
        # from strategy_map so the per-strategy construction args are preserved.
        require_supported_strategy(strategy_name)
        strategy = strategy_map[strategy_name]
        strategy._survivor_accumulator = survivor_accumulator  # [S149]
        # [S172 D6.2 §4.4 / BOUNDED REPAIR §1.3] What crosses the S149 attribute
        # seam is a BOOLEAN — "a checkpoint resume is in force" — and NOT the
        # recovered record-ordinal floor. `f7583bc` forwarded the floor and the
        # study body compared `trial.number` against it; the floor is a 1-based
        # record ordinal and `trial.number` is Optuna's 0-based number, so every
        # normal resume was rejected. The floor's only consumer is
        # `trial_counter` above, in this layer, where the quantity it measures
        # actually lives.
        #
        # Deliberately NOT a new entry in `OPTIMIZE_FORWARDED_KWARGS`: that
        # tuple is AST-gated against the live `strategy.search(...)` call and is
        # also what `strategy_contract_gap` measures the three gated strategies
        # against, so widening it would change an unrelated contract. The flag
        # is read and ENFORCED in the Optuna study body (the loaded-study wall,
        # §1.3.5) — it is not an advisory that dies in an override dict.
        strategy._d6_2_require_loaded_study = bool(resume_checkpoint)

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
        #
        # [S172 D6.2 §8] With the S166 clear ENABLED, this list is only the tail
        # that arrived since the last checkpoint — the TRUNCATED STUMP. The
        # finalizer's input is the cumulative canonical state reconstructed from
        # the checkpoint, reconciled with that tail, which is what makes "the
        # finalizer still receives complete 24-field input via the resume path"
        # a property of the code. `_checkpoint_finalizer_input` also runs the
        # same three walls over the tail, so a malformed LOSING candidate in it
        # fails the run rather than vanishing during reconciliation.
        _raw_candidates_d3_5 = _checkpoint_finalizer_input(survivor_accumulator)

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
                # [S172 D6.2 §4.5] Resumed-run provenance, echoed here so a
                # reader of the certified generation finds it without knowing
                # the checkpoint directory exists. `raw_candidate_count` above
                # is "the records supplied to the finalizer by the resumed
                # execution" — neither the original process's raw count nor a
                # cumulative count across all pre-compaction observations.
                # NO SIDECAR-FIELD PARITY IS CLAIMED.
                "d6_2_checkpoint_run_id":  _d6_2_context.run_id,
                "d6_2_run_context_digest": _d6_2_context.run_context_digest,
                "d6_2_resume_provenance":  _d6_2_context.resume_provenance,
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
