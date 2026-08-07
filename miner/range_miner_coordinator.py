"""
range_miner_coordinator.py — S172 RANGE-MINER stripe coordinator

Phase 4 implementation (staged). This module owns the durable stripe/shard
ledger, the stripe state machine, macro-stripe partitioning, and the L8
completion-reconciliation predicate. Later stages layer identity/fencing
(Stage 2), the staging pipeline (Stage 3), and the retry matrix + trial
lifecycle + Phase5Sink (Stage 4) on top of this core.

`run_trial_miner(...)` stays importable with its existing signature so the
integration gate at window_optimizer_integration_final.py:_use_miner can import
it symmetrically with run_trial_persistent (PWC) and run_trial_zmq_sqlite (ZMQ).
It is wired to drive the coordinator in Stage 5.

INFRASTRUCTURE-NEUTRAL DESIGN (S172_INFRASTRUCTURE_INTERFACE_v1_0):
  Workers are identified by their registered worker_id ('{hostname}:gpu{id}')
  and reached at the NODE CONFIG's SSH address — never a parsed hostname
  (Binding Decision A). Staged shards land in a configurable staging_dir.

Architecture note (v1.4.5 §3.A): a worker partitions ONE assigned macro-stripe
into MANY GPU-safe sub-stripes and emits one SubStripeResultMessage per
sub-stripe THEN one StripeCompleteMessage. The ledger is therefore SHARD-level
(keyed by (run_id, stripe_id, attempt, sub_index)), never one-row-per-stripe.
Phase 4 MUST NOT assemble the 22 arrays, dedup, order, or run the contract wall
(that is Phase 5).
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import inspect
import json
import logging
import math
import os
import re
import select
import socket
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("range_miner_coordinator")

# Family/cap logic + the CANONICAL substripe serialization are shared with the
# Phase-3 worker — reuse rather than re-derive. Importing build_substripe_payload_bytes
# guarantees Phase 4's inline normalization is byte-identical to what the worker
# spools (Blocker 4); expected_substripes uses the SAME select_seed_cap logic the
# worker partitions with (Blocker 7, range_miner_worker.py:467-474).
from miner import dataset_authority
from miner.range_miner_worker import (
    SUBSTRIPE_SCHEMA_VERSION,
    MinerFramedSocket,
    VramCaps,
    build_substripe_payload_bytes,
    is_hybrid_family,
    select_seed_cap,
    sha256_residues,
)
from miner.range_miner_protocol import (
    DEFAULT_MINER_PORT,
    MinerShutdownMessage,
    StripeAssignMessage,
)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def compute_dataset_sha256(path: str) -> str:
    """Streaming sha256 of the dataset file — the coordinator computes the
    mandatory `dataset_sha256` itself (mirrors range_miner_worker._sha256_file so
    the worker's Blocker-6 equality check compares like against like)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_dataset_sha256(dataset_path: str) -> str:
    """The dataset digest for this assignment — RUN-scoped, not trial-scoped.

    [S172 Phase 6-P0.5 §2.1 — the freeze is the point of the freeze]

    Before P0.5 the coordinator derived `dataset_sha256` by hashing the file
    every time `serve_trial` was entered, i.e. once **per Optuna trial**. Nothing
    was wrong with any individual derivation; the defect was the scope. A scrape
    landing between two trials of one study changed the bytes under the run, and
    trial N+1 simply hashed the new file and carried on. Every downstream check
    stayed self-consistent against a different dataset from trial N's, and no
    error was raised anywhere — the study was split across two datasets and the
    only surviving evidence was two different digests in two NPZ files that
    nobody compares.

    So the digest is now taken from the run-start freeze when the path being
    assigned IS the frozen path. A pointer that moves mid-run cannot reach this
    function's answer, which is requirement 7.

    Falls back to hashing when this process has no freeze (a harness driving
    `serve_trial` directly) or when the path is not the frozen one (a genuinely
    different dataset). The fallback is the pre-P0.5 behaviour exactly, so no
    existing caller changes meaning — it simply stops being the only behaviour.
    """
    frozen_sha = dataset_authority.run_frozen_dataset_sha256(dataset_path)
    if frozen_sha is not None:
        return frozen_sha
    return compute_dataset_sha256(dataset_path)


def event_id_for(
    run_id: str, stripe_id: str, attempt: int, sub_index: int, staging_generation: int
) -> str:
    """Immutable unique id for a shard-attempt-generation — the L5 fencing key AND
    the L6 ack key on every ShardReadyManifest. Derived (not random) so it is
    stable across a resume and unique per (run, stripe, attempt, sub, generation)."""
    return f"{run_id}:{stripe_id}:a{attempt}:s{sub_index}:g{staging_generation}"

# ---------------------------------------------------------------------------
# Stripe states (Blocker 5 state machine)
#   pending -> claimed -> staging -> done
#                             \-> failed
#   plus cancelled (whole-trial abort, L3 / Stage 4)
# ---------------------------------------------------------------------------
ST_PENDING = "pending"
ST_CLAIMED = "claimed"
ST_STAGING = "staging"
ST_DONE = "done"
ST_FAILED = "failed"
ST_CANCELLED = "cancelled"

# Shard staging lifecycle (Stage 3 completes the transitions).
SH_PENDING = "pending"     # result recorded; local file not yet materialized
SH_STAGED = "staged"       # local file written, not yet hash-verified
SH_VERIFIED = "verified"   # local size + sha256 verified
SH_FAILED = "failed"

# ---------------------------------------------------------------------------
# [§4.3 ADMISSION LIVENESS — Beta Ruling 1] How long a stage may wait for its
# expected worker pool to be ADMITTED before the trial fails explicitly.
#
# This is NOT a run/serve timeout and must never be confused with one:
#   * ADMISSION (this value)  — bounded. It covers only the interval in which a
#     stage has not yet been assigned and the eligible pool is short. It is armed
#     once per STAGE and is never reset by worker churn.
#   * EXECUTION (serve_timeout) — UNBOUNDED by default (None), by Beta's earlier
#     correction. Once a stage is assigned, dispatch, lease expiry and completion
#     evaluation run regardless of the current eligible count and regardless of
#     how long the scan takes.
#
# 180s matches the PWC readiness window (persistent_worker_coordinator.py:826
# and :864, `_tcp_wait_ready(expected=..., timeout_s=180.0)`) so the miner and PWC
# agree on how long a fleet may take to come up.
# ---------------------------------------------------------------------------
DEFAULT_WORKER_ADMISSION_TIMEOUT = 180.0


# ---------------------------------------------------------------------------
# [RESOLVED EXECUTION SET] consumer seam for the registration path.
# ---------------------------------------------------------------------------
# One named function, so the membership decision has a single site that can be
# observed — and reverted, which is what G-MUTANT does. Lazy and defensive: the
# miner package must stay importable standalone (it deploys to the rigs), and
# with no set frozen every worker is admitted, i.e. exactly the Phase-4
# behaviour every existing loopback gate was written against.
def _execution_set_admission(worker_id):
    try:
        from execution_set import is_admitted_worker
    except ImportError:
        return True, None
    return is_admitted_worker(worker_id)


def _execution_set_expected_workers(context_pool_size):
    """`expected_workers` for THIS run — from the frozen set when there is one.

    [ADMISSION BINDING — Beta, admission-binding brief §1] The set already
    recorded an admission count, but `serve_trial` derived `expected_workers`
    independently from `context["worker_pool_size"]`. Two frozen facts about the
    same run, free to disagree, and they did: a local two-GPU set still waited
    for the default eight workers — six of which the set itself declared could
    never connect, because a worker outside the set is refused admission by
    `_execution_set_admission` above. The trial then spent its whole bounded
    admission window (180s, unchanged) failing to meet a threshold that was
    unmeetable by construction.

    The set is now the authority. `worker_pool_size` keeps its meaning and stays
    the REQUEST the set was resolved from — it just stops being a second answer
    to the same question. With no set frozen (harnesses, direct calls, every
    pre-existing Phase-4 loopback gate) the context value is returned unchanged.

    Returns `(expected_workers, source)`; `source` is recorded in the log so the
    binding is readable off a run rather than assumed from this docstring.
    """
    try:
        from execution_set import admission_expectation
    except ImportError:
        return int(context_pool_size), "context(execution_set unavailable)"
    return admission_expectation(int(context_pool_size), consumer="miner")


# ---------------------------------------------------------------------------
# Coordinator configuration (L4 — injectable, NOT module constants)
# ---------------------------------------------------------------------------
@dataclass
class CoordinatorConfig:
    """All Phase-4 tunables are injectable config, never buried constants (L4).

    The four seed caps mirror the worker's VramCaps tiers. Hybrid workflow
    phases (3/4) use the tighter *_hybrid cap. staging_* bound Zeus-local
    staging capacity (§15); lease/timeouts drive the state machine.
    """
    seed_cap_nvidia: int = 5_000_000
    seed_cap_amd: int = 2_000_000
    seed_cap_nvidia_hybrid: int = 2_500_000
    seed_cap_amd_hybrid: int = 1_000_000
    miner_stripe_size: int = 67_108_864
    staging_high_water_bytes: int = 16 * 1024 ** 3
    staging_high_water_files: int = 512
    staging_dir: Optional[str] = None
    compute_lease_timeout: float = 300.0
    staging_timeout: float = 600.0
    # L7: initial bound for the synchronous Phase5Sink.abort_trial() call. None
    # falls back to staging_timeout (either is acceptable per Beta).
    phase5_abort_timeout: Optional[float] = None
    # serve_trial() framed-TCP server bind target (§8 miner default port 5700).
    # Loopback host for the acceptance gate; set 0.0.0.0 in production.
    miner_host: str = "127.0.0.1"
    miner_port: int = DEFAULT_MINER_PORT
    spool_root: str = "/var/spool/miner"   # default node spool root (Decision A)
    staging_workers: int = 4               # bounded staging executor size (Defect 4)
    # Defect 2 (C3): the staging admission bound is (staging_workers +
    # staging_queue_depth). enqueue_staging acquires one slot BEFORE the payload is
    # retained, so the number of in-flight (queued + active) inline payloads — and
    # thus their RAM — is bounded, not just the number of active threads.
    staging_queue_depth: int = 2
    # Defect 1c (C4): the deferred queue (sub-stripes waiting on capacity/slots) is
    # bounded by COUNT and by retained BYTES (the byte bound reuses
    # staging_high_water_bytes).
    #
    # [S172-BP §2, Beta A — binding] THE CONSTANT 64 IS DELETED. Its meaning was
    # never a statement about memory: 64 inline entries retain ~10 KB against a
    # 16 GiB byte cap, so the count cap fired while the byte cap could not (Alpha's
    # deferred-queue note §2.3). The runtime bound is now DERIVED from the resolved
    # execution set, stripe geometry, phase and per-worker VRAM caps
    # (`staging_burst_bound_conservative` + the resume margin) — never a
    # hand-maintained constant.
    #
    # This field survives ONLY as an OPTIONAL OPERATOR OVERRIDE:
    #   None  -> use the derived bound (the production shape)
    #   int   -> force that bound; if it is BELOW the derived bound a WARNING names
    #            both numbers, because the operator is re-arming the condition the
    #            derivation exists to remove.
    staging_deferred_max: Optional[int] = None
    # [S172-BP §1.5, Beta §1 + gate 11] The ONE permitted trial-terminal path for a
    # coordinator capacity WAIT. Measured from the OLDEST currently-paused
    # connection's pause-entry time; on expiry the coordinator calls fail_trial
    # DIRECTLY with a `coordinator_staging_capacity_timeout:` reason. The event
    # never enters the phase-specific worker retry matrix — a capacity wait is a
    # coordinator/infrastructure condition, not a stripe failure state (§0).
    # Default 600.0 = the same class as staging_timeout; FLAGGED FOR BETA (§1.5).
    staging_capacity_timeout: float = 600.0
    # Defect 4 (C4): a hung fetch whose thread never returns is tracked here; the
    # registry is bounded and, when the cap is exceeded, surfaces a capacity error
    # rather than silently accumulating daemon threads across a soak.
    staging_orphan_fetch_max: int = 8


# ---------------------------------------------------------------------------
# Worker record (Stage 2 fleshes out registration validation; Stage 1 uses a
# plain record the harness constructs directly)
# ---------------------------------------------------------------------------
@dataclass
class WorkerRecord:
    worker_id: str
    backend: str                       # "rocm" | "cuda"
    seed_caps: Dict[str, int]          # advertised {amd,nvidia,amd_hybrid,nvidia_hybrid}
    hostname: str = ""
    spool_root: str = ""
    ssh_address: str = ""
    ssh_user: str = ""


# ---------------------------------------------------------------------------
# Node identity (Binding Decision A) — the coordinator reaches a worker at the
# NODE CONFIG's SSH address/user, NEVER a hostname parsed out of worker_id.
# ---------------------------------------------------------------------------
@dataclass
class NodeConfig:
    """Configured node record for a worker's host. Transfers (Stage 3) use
    ssh_address/ssh_user from HERE; remote spool paths must live under
    spool_root (gate 17)."""
    hostname: str = ""
    spool_root: str = ""
    ssh_address: str = ""
    ssh_user: str = ""


@dataclass
class WorkerConnection:
    """Server-side binding of ONE registered worker's TCP connection
    (Decision A). The connection carries the bound worker identity + node
    record and, per assigned stripe, the ATTEMPT it was handed. L1 pairs that
    recorded attempt with the ledger's authoritative current_attempt so a
    delayed message from a superseded assignment is detectable without any
    per-message `attempt` field (which the protocol does not carry)."""
    worker_id: str
    hostname: str
    backend: str
    seed_caps: Dict[str, int]
    supported_variants: frozenset
    node_config: NodeConfig
    quarantined: bool = False
    quarantine_reason: Optional[str] = None
    # stripe_id -> the attempt this connection was assigned (L1 authority pair)
    assignment_attempts: Dict[str, int] = field(default_factory=dict)

    def record_assignment(self, stripe_id: str, attempt: int) -> None:
        self.assignment_attempts[stripe_id] = attempt


def spool_path_within_root(spool_root: str, spool_path: str) -> bool:
    """True iff spool_path is a normalized absolute path strictly under
    spool_root. normpath collapses `..` so a traversal like
    `<root>/../../etc/x` resolves outside the root and is rejected (gate 17)."""
    if not spool_root or not spool_path:
        return False
    root = os.path.normpath(spool_root)
    p = os.path.normpath(spool_path)
    if not os.path.isabs(root) or not os.path.isabs(p):
        return False
    return p.startswith(root.rstrip(os.sep) + os.sep)


def advertised_effective_cap(
    backend: str, family_name: str, seed_caps: Dict[str, int]
) -> int:
    """The worker's advertised sub-stripe cap for this concrete variant.

    Mirrors range_miner_worker.select_seed_cap() (hybrid phases take the tighter
    cap) but sourced from the worker's REGISTER advertisement rather than local
    argparse — the coordinator must size expected_substripes with the SAME cap
    the worker will use to partition (Blocker 7)."""
    caps = VramCaps(
        amd=seed_caps["amd"],
        nvidia=seed_caps["nvidia"],
        amd_hybrid=seed_caps["amd_hybrid"],
        nvidia_hybrid=seed_caps["nvidia_hybrid"],
    )
    return select_seed_cap(backend, family_name, caps)


# ---------------------------------------------------------------------------
# Macro-stripe partitioner (Blocker 7)
# ---------------------------------------------------------------------------
def partition_macro_stripes(
    total_seeds: int, macro_size: int, base_start: int = 0
) -> List[Tuple[int, int, int]]:
    """Contiguous macro-stripes over [base_start, base_start+total_seeds).

    Returns (stripe_index, seed_start, seed_count) with NO gap/overlap. A
    macro-stripe MAY exceed one GPU cap — the WORKER partitions it into GPU-safe
    sub-stripes at runtime (Blocker 7: macro sizing != sub-stripe sizing)."""
    if macro_size <= 0:
        raise ValueError(f"macro_size must be positive, got {macro_size}")
    if total_seeds < 0:
        raise ValueError(f"total_seeds must be non-negative, got {total_seeds}")
    out: List[Tuple[int, int, int]] = []
    cursor = base_start
    remaining = total_seeds
    idx = 0
    while remaining > 0:
        count = min(macro_size, remaining)
        out.append((idx, cursor, count))
        cursor += count
        remaining -= count
        idx += 1
    return out


def expected_substripes_for(seed_count: int, effective_cap: int) -> int:
    """ceil(seed_count / effective_cap) — the count the worker will report as
    StripeComplete.substripes_done (L8)."""
    if effective_cap <= 0:
        raise ValueError(f"effective_cap must be positive, got {effective_cap}")
    if seed_count <= 0:
        return 0
    return math.ceil(seed_count / effective_cap)


# ---------------------------------------------------------------------------
# [S172-BP §2 — Beta A, binding] THE DERIVED STAGING BURST BOUND
#
# Two PURE functions (no I/O, no coordinator state) so both are unit-testable in
# isolation and neither can drift from the cap logic the worker actually
# partitions with: both resolve caps through `advertised_effective_cap`, which is
# the coordinator's existing single cap path (it wraps the worker's own
# `select_seed_cap`, range_miner_worker.py:472-479), NOT a second phase->cap table.
#
# ⚠ 116 vs 136 — BETA SINGLED THIS DISTINCTION OUT; PRESERVE IT.
#   116 = the EXACT count for the recorded heterogeneous 2026-08-05 assignment
#         (34 + 14 + 34 + 34): three ROCm stripes at cap 2,000,000 and one CUDA
#         stripe at cap 5,000,000, each over stripe_span 67,108,864.
#   136 = the CONSERVATIVE PRE-ASSIGNMENT bound for the same geometry: four
#         simultaneously admitted stripe slots, each sized by the WORST (tightest
#         cap => largest sub-stripe count) worker eligible for that slot, i.e.
#         4 x 34 = 136 when any AMD worker is eligible.
# They are different quantities answering different questions. The RUNTIME uses
# the conservative one, because the assignment is not yet known when capacity must
# already be sized. The exact one is the audit/forensic quantity.
# ---------------------------------------------------------------------------

# Workflow phases 3 and 4 are the hybrid (variable-skip) phases; 1 and 2 are
# constant. Hybrid takes the TIGHTER seed cap (an extra skip_sequences_gpu
# allocation), so it produces MORE sub-stripes per stripe — which is why the
# phase, not just the backend, drives the bound.
_HYBRID_WORKFLOW_PHASES = (3, 4)


def phase_family_probe(phase: int, family_name: Optional[str] = None) -> str:
    """The concrete variant NAME whose hybrid-ness matches `phase`.

    Cap resolution must go through `advertised_effective_cap` (one path, shared
    with the worker). That function keys on a family NAME, not on a phase, so a
    phase-driven caller needs a name whose `is_hybrid_family()` answer matches the
    phase. An explicit `family_name` always wins — a real assignment knows its own
    variant and must not be re-derived from the phase.
    """
    if family_name:
        return family_name
    return "_probe_hybrid" if int(phase) in _HYBRID_WORKFLOW_PHASES else "_probe"


def applicable_seed_cap(
    backend: str, seed_caps: Dict[str, int], phase: int,
    family_name: Optional[str] = None,
) -> int:
    """The seed cap a worker with `seed_caps` will partition with in `phase`.

    Delegates to `advertised_effective_cap` so the bound and the coordinator's own
    `expected_substripes_for` sizing (:364, used by assign_stripes) can never
    diverge — the defect class Beta named in §2.
    """
    return advertised_effective_cap(
        backend, phase_family_probe(phase, family_name), seed_caps)


def _coerce_worker_cap_record(worker: Any, caps: Optional[Dict[str, int]]) -> Tuple[str, Dict[str, int]]:
    """(backend, seed_caps) for one worker record, tolerating the three shapes the
    coordinator actually holds: a WorkerConnection / WorkerRecord (attributes), a
    plain mapping, or a (backend, seed_caps) pair. A worker that advertises no caps
    falls back to the CENTRALLY-resolved `caps` — never to a literal."""
    if isinstance(worker, (tuple, list)) and len(worker) == 2:
        backend, wcaps = worker[0], worker[1]
    elif isinstance(worker, dict):
        backend, wcaps = worker.get("backend"), worker.get("seed_caps")
    else:
        backend = getattr(worker, "backend", None)
        wcaps = getattr(worker, "seed_caps", None)
    if not backend:
        raise ValueError(f"worker record carries no backend: {worker!r}")
    resolved = dict(wcaps) if wcaps else dict(caps or {})
    if not resolved:
        raise ValueError(
            f"worker {backend!r} advertises no seed_caps and no central caps were "
            f"supplied — the bound cannot be derived from nothing")
    return str(backend), resolved


def staging_burst_bound_exact(assignments: Any) -> int:
    """EXACT staging-request burst for a KNOWN assignment (§2, the 116 quantity).

        sum over actual (stripe_span, worker) assignments of
            ceil(stripe_span / applicable_seed_cap(worker, phase))

    `assignments` is an iterable of mappings, each describing ONE assigned stripe:
        stripe_span / seed_count   — the stripe's seed span            (required)
        effective_cap              — the already-resolved cap          (optional)
        backend + seed_caps        — resolve the cap from the worker   (optional)
        phase / workflow_phase     — the workflow phase                (optional)
        family_name                — the concrete variant              (optional)
    `effective_cap`, when present, is used verbatim: `assign_stripes` already
    records it per assignment (:2007) from the very same `advertised_effective_cap`
    call, so consuming it is reuse, not a second derivation.

    Recorded 2026-08-05 assignment => 34 + 14 + 34 + 34 = 116.
    """
    total = 0
    for row in assignments:
        if isinstance(row, dict):
            span = row.get("stripe_span", row.get("seed_count"))
            cap = row.get("effective_cap")
            if cap is None:
                backend, wcaps = _coerce_worker_cap_record(row, row.get("seed_caps"))
                cap = applicable_seed_cap(
                    backend, wcaps,
                    row.get("phase", row.get("workflow_phase", 1)),
                    row.get("family_name"))
        else:
            raise TypeError(
                f"staging_burst_bound_exact expects mappings per assignment, "
                f"got {type(row).__name__}")
        if span is None:
            raise ValueError(f"assignment carries no stripe_span/seed_count: {row!r}")
        total += expected_substripes_for(int(span), int(cap))
    return total


def staging_burst_bound_conservative(
    slots: Any, eligible_workers: Any, phase: int,
    caps: Optional[Dict[str, int]] = None,
    family_name: Optional[str] = None,
) -> int:
    """CONSERVATIVE pre-assignment burst bound (§2, the 136 quantity).

        sum over simultaneously admitted stripe slots of
            max over workers eligible for that slot of
                ceil(stripe_span / applicable_seed_cap(worker, phase))

    This is the bound the RUNTIME uses, because capacity must be sized BEFORE the
    round-robin decides which worker gets which stripe. Taking the max per slot is
    what makes it safe for every assignment the scheduler could still choose.

    `slots` is either
      * a sequence of per-slot stripe SPANS — the honest shape, because the last
        macro-stripe from `partition_macro_stripes` may be shorter than
        `miner_stripe_size`; or
      * an int slot COUNT, which then REQUIRES a uniform span supplied as the
        single-element sequence `[span] * n` by the caller. An int with no spans is
        refused rather than paired with an invented stripe size.
    `eligible_workers` are worker records (WorkerConnection / WorkerRecord / dict /
    (backend, seed_caps) pair). `caps` is the centrally-resolved cap mapping used
    only for a worker that advertises none.

    Four slots of 67,108,864 with any AMD worker (cap 2,000,000) eligible
    => 4 x ceil(67,108,864 / 2,000,000) = 4 x 34 = 136.
    """
    if isinstance(slots, int) and not isinstance(slots, bool):
        raise TypeError(
            "staging_burst_bound_conservative(slots=...) needs the per-slot stripe "
            "SPANS, not a bare count — pass [stripe_span] * n so the bound is "
            "derived from real geometry rather than an assumed stripe size")
    spans = [int(s) for s in slots]
    workers = list(eligible_workers)
    if not workers:
        raise ValueError(
            "staging_burst_bound_conservative requires at least one eligible worker")
    per_worker_caps = [
        applicable_seed_cap(b, c, phase, family_name)
        for b, c in (_coerce_worker_cap_record(w, caps) for w in workers)
    ]
    total = 0
    for span in spans:
        # max over eligible workers == the TIGHTEST cap, which yields the LARGEST
        # sub-stripe count for that slot.
        total += max(expected_substripes_for(span, cap) for cap in per_worker_caps)
    return total


# ---------------------------------------------------------------------------
# L8 completion-reconciliation predicate (pure — no I/O)
# ---------------------------------------------------------------------------
@dataclass
class CompletionCheck:
    """Structured result of the L8 predicate so gates can assert each condition
    independently (gate 35 breaks one at a time)."""
    substripes_match: bool
    seed_sum_match: bool
    survivor_sum_match: bool
    coverage_ok: bool
    all_verified: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def reconciled(self) -> bool:
        """The four L8 accounting invariants (independent of staging status)."""
        return (
            self.substripes_match
            and self.seed_sum_match
            and self.survivor_sum_match
            and self.coverage_ok
        )

    @property
    def is_complete(self) -> bool:
        """A stripe may transition to `done` (reconciled AND every shard staged
        + hash-verified). Publish (Phase5Sink) is layered in Stage 4."""
        return self.reconciled and self.all_verified


def _coverage_exact(stripe_start: int, stripe_count: int, shards: List[Dict[str, Any]]) -> bool:
    """Sub-stripe seed ranges must tile [stripe_start, stripe_start+stripe_count)
    with no gap and no overlap."""
    ordered = sorted(shards, key=lambda s: s["seed_start"])
    cursor = stripe_start
    for s in ordered:
        if s["seed_count"] <= 0:
            return False
        if s["seed_start"] != cursor:   # gap (>) or overlap (<)
            return False
        cursor += s["seed_count"]
    return cursor == stripe_start + stripe_count


def evaluate_stripe_completion(
    stripe: Dict[str, Any], shards: List[Dict[str, Any]]
) -> CompletionCheck:
    """L8: a stripe is complete ONLY when ALL of:
      StripeComplete.substripes_done == expected_substripes == count(distinct sub_index)
      sum(shard.seed_count)     == stripe.seed_count
      sum(shard.survivor_count) == StripeComplete.survivors_total
      exact contiguous coverage (no gap/overlap)
      every shard staged + hash-verified
    Any mismatch → NOT complete (feeds the failure/retry path, Blocker 3)."""
    reasons: List[str] = []

    expected = stripe.get("expected_substripes")
    substripes_done = stripe.get("substripes_done")
    distinct = len({s["sub_index"] for s in shards})

    substripes_match = (
        stripe.get("stripe_complete_seen", False)
        and substripes_done is not None
        and expected is not None
        and substripes_done == expected == distinct
    )
    if not substripes_match:
        reasons.append(
            f"substripe count mismatch: complete_seen="
            f"{stripe.get('stripe_complete_seen', False)} "
            f"substripes_done={substripes_done} expected={expected} distinct={distinct}"
        )

    seed_sum = sum(s["seed_count"] for s in shards)
    seed_sum_match = seed_sum == stripe.get("seed_count")
    if not seed_sum_match:
        reasons.append(
            f"seed_count sum {seed_sum} != stripe.seed_count {stripe.get('seed_count')}"
        )

    survivor_sum = sum(s["survivor_count"] for s in shards)
    survivor_sum_match = survivor_sum == stripe.get("survivors_total")
    if not survivor_sum_match:
        reasons.append(
            f"survivor sum {survivor_sum} != survivors_total {stripe.get('survivors_total')}"
        )

    coverage_ok = _coverage_exact(
        stripe.get("seed_start", 0), stripe.get("seed_count", 0), shards
    )
    if not coverage_ok:
        reasons.append("sub-stripe coverage is not exact (gap or overlap)")

    all_verified = bool(shards) and all(
        s["staging_status"] == SH_VERIFIED for s in shards
    )
    if not all_verified:
        reasons.append("not every shard is staged + hash-verified")

    return CompletionCheck(
        substripes_match=substripes_match,
        seed_sum_match=seed_sum_match,
        survivor_sum_match=survivor_sum_match,
        coverage_ok=coverage_ok,
        all_verified=all_verified,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Durable ledger — SQLite sole-writer (adapted from zmq_sqlite_coordinator.py:
# _write_lock + WAL). Do NOT reuse MAX_ATTEMPTS or the one-row job_results table.
# ---------------------------------------------------------------------------
class MinerLedger:
    """Stripe + shard tables. Zeus is the SOLE writer: every write is taken
    under self._write_lock, WAL mode allows concurrent readers. Result
    cardinality is SHARD-level, keyed (run_id, stripe_id, attempt, sub_index)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._write_lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._write_lock:
            with self._conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS stripes (
                        run_id              TEXT NOT NULL,
                        stripe_id           TEXT NOT NULL,
                        seed_start          INTEGER NOT NULL,
                        seed_count          INTEGER NOT NULL,
                        state               TEXT NOT NULL DEFAULT 'pending',
                        claimed_by          TEXT,
                        current_attempt     INTEGER NOT NULL DEFAULT 0,
                        staging_generation  INTEGER NOT NULL DEFAULT 0,
                        expected_substripes INTEGER,
                        lease_expires_at    REAL,
                        phase               INTEGER NOT NULL DEFAULT 0,
                        family_name         TEXT NOT NULL DEFAULT '',
                        phase_degraded      INTEGER NOT NULL DEFAULT 0,
                        stripe_complete_seen INTEGER NOT NULL DEFAULT 0,
                        substripes_done     INTEGER,
                        survivors_total     INTEGER,
                        created_at          REAL NOT NULL,
                        PRIMARY KEY (run_id, stripe_id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS shards (
                        run_id               TEXT NOT NULL,
                        stripe_id            TEXT NOT NULL,
                        attempt              INTEGER NOT NULL,
                        sub_index            INTEGER NOT NULL,
                        worker_id            TEXT NOT NULL,
                        seed_start           INTEGER NOT NULL,
                        seed_count           INTEGER NOT NULL,
                        survivor_count       INTEGER NOT NULL DEFAULT 0,
                        remote_spool_path    TEXT,
                        local_staged_path    TEXT,
                        size_bytes           INTEGER,
                        sha256               TEXT,
                        staging_status       TEXT NOT NULL DEFAULT 'pending',
                        created_at           REAL NOT NULL,
                        verified_at          REAL,
                        -- SC1: durable remote-deletion status (Decision B)
                        remote_delete_status   TEXT NOT NULL DEFAULT 'none',
                        remote_delete_attempts INTEGER NOT NULL DEFAULT 0,
                        remote_delete_error    TEXT,
                        remote_deleted_at      REAL,
                        -- L2: Phase-5 acknowledgement + local-cleanup seam
                        phase5_status        TEXT NOT NULL DEFAULT 'none',
                        phase5_enqueued_at   REAL,
                        phase5_acked_at      REAL,
                        local_cleanup_status TEXT NOT NULL DEFAULT 'none',
                        local_deleted_at     REAL,
                        PRIMARY KEY (run_id, stripe_id, attempt, sub_index)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workers (
                        worker_id               TEXT PRIMARY KEY,
                        hostname                TEXT NOT NULL DEFAULT '',
                        backend                 TEXT NOT NULL DEFAULT '',
                        seed_caps_json          TEXT NOT NULL DEFAULT '{}',
                        supported_variants_json TEXT NOT NULL DEFAULT '[]',
                        spool_root              TEXT NOT NULL DEFAULT '',
                        ssh_address             TEXT NOT NULL DEFAULT '',
                        ssh_user                TEXT NOT NULL DEFAULT '',
                        status                  TEXT NOT NULL DEFAULT 'eligible',
                        quarantine_reason       TEXT,
                        registered_at           REAL NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reservations (
                        reservation_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id            TEXT NOT NULL UNIQUE,
                        run_id              TEXT NOT NULL,
                        stripe_id           TEXT NOT NULL,
                        attempt             INTEGER NOT NULL,
                        sub_index           INTEGER NOT NULL,
                        staging_generation  INTEGER NOT NULL,
                        size_bytes          INTEGER NOT NULL,
                        status              TEXT NOT NULL DEFAULT 'held',
                        temp_path           TEXT,
                        staged_path         TEXT,
                        created_at          REAL NOT NULL,
                        released_at         REAL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trials (
                        run_id                 TEXT PRIMARY KEY,
                        trial_number           INTEGER,
                        state                  TEXT NOT NULL DEFAULT 'running',
                        abort_event_id         TEXT,
                        abort_cleanup_status   TEXT NOT NULL DEFAULT 'none',
                        commit_event_id        TEXT,
                        commit_delivery_status TEXT NOT NULL DEFAULT 'none',
                        created_at             REAL NOT NULL,
                        finalized_at           REAL
                    )
                """)
                # D0: trial-GLOBAL immutable context, persisted ONCE per run_id
                # before any stripe work. Adjacent to `trials` (not the mutable
                # lifecycle row) precisely because it must NEVER change after trial
                # creation — write-once via INSERT OR IGNORE, no UPDATE path — and
                # must survive a coordinator restart so a manifest can be rebuilt
                # identically from the durable ledger alone (gate D0-4). sessions is
                # stored as JSON.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trial_context (
                        run_id             TEXT PRIMARY KEY,
                        trial_number       INTEGER,
                        window_size        INTEGER,
                        offset_val         INTEGER,
                        sessions_json      TEXT,
                        skip_min           INTEGER,
                        skip_max           INTEGER,
                        prng_base          TEXT,
                        forward_threshold  REAL,
                        reverse_threshold  REAL,
                        dataset_sha256     TEXT,
                        residue_sha256     TEXT,
                        created_at         REAL NOT NULL
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_stripe_state "
                    "ON stripes (run_id, state)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_stripe_lease "
                    "ON stripes (run_id, state, lease_expires_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_res_status "
                    "ON reservations (status)"
                )
                conn.commit()

    # ----- staging reservations (L2 + §15) ----------------------------------
    def reserve(
        self,
        run_id: str,
        stripe_id: str,
        attempt: int,
        sub_index: int,
        staging_generation: int,
        size_bytes: int,
        high_water_bytes: int,
        high_water_files: int,
        now: Optional[float] = None,
    ) -> Optional[int]:
        """Reserve capacity BEFORE transfer. Capacity is a Zeus-local resource
        shared across all runs, so held bytes/files are summed GLOBALLY. Grants
        only if BOTH marks hold; otherwise returns None (back-pressure). A held
        reservation counts until it is explicitly released (ack + local delete,
        or a failure-path cleanup) — never on mere enqueue."""
        now = time.time() if now is None else now
        with self._write_lock:
            with self._conn() as conn:
                agg = conn.execute(
                    "SELECT COALESCE(SUM(size_bytes),0) AS b, COUNT(*) AS n "
                    "FROM reservations WHERE status='held'"
                ).fetchone()
                held_bytes, held_files = agg["b"], agg["n"]
                if held_bytes + size_bytes > high_water_bytes:
                    return None
                if held_files + 1 > high_water_files:
                    return None
                event_id = event_id_for(
                    run_id, stripe_id, attempt, sub_index, staging_generation)
                try:
                    cur = conn.execute(
                        """INSERT INTO reservations
                           (event_id, run_id, stripe_id, attempt, sub_index,
                            staging_generation, size_bytes, status, created_at)
                           VALUES (?,?,?,?,?,?,?, 'held', ?)""",
                        (event_id, run_id, stripe_id, attempt, sub_index,
                         staging_generation, size_bytes, now),
                    )
                except sqlite3.IntegrityError:
                    # Defect 2 defense-in-depth: UNIQUE(event_id) already holds a
                    # reservation for this immutable event (a duplicate result that
                    # slipped past the _serve_dispatch guard). Do NOT create a
                    # second reservation for one logical shard — return None.
                    return None
                conn.commit()
                return cur.lastrowid

    def get_reservation_by_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM reservations WHERE event_id=?", (event_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def release_reservation(self, reservation_id: int, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        with self._write_lock:
            with self._conn() as conn:
                cur = conn.execute(
                    "UPDATE reservations SET status='released', released_at=? "
                    "WHERE reservation_id=? AND status='held'",
                    (now, reservation_id),
                )
                conn.commit()
                return cur.rowcount == 1

    def set_reservation_paths(
        self, reservation_id: int, temp_path: Optional[str] = None,
        staged_path: Optional[str] = None,
    ) -> None:
        sets, vals = [], []
        if temp_path is not None:
            sets.append("temp_path=?"); vals.append(temp_path)
        if staged_path is not None:
            sets.append("staged_path=?"); vals.append(staged_path)
        if not sets:
            return
        vals.append(reservation_id)
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    f"UPDATE reservations SET {', '.join(sets)} WHERE reservation_id=?",
                    vals,
                )
                conn.commit()

    def get_reservation(self, reservation_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM reservations WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def held_bytes(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COALESCE(SUM(size_bytes),0) FROM reservations WHERE status='held'"
            ).fetchone()[0]

    def held_files(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM reservations WHERE status='held'"
            ).fetchone()[0]

    # ----- shard staging-status setters -------------------------------------
    def set_shard_staging_status(
        self, run_id: str, stripe_id: str, attempt: int, sub_index: int, status: str
    ) -> bool:
        with self._write_lock:
            with self._conn() as conn:
                cur = conn.execute(
                    "UPDATE shards SET staging_status=? "
                    "WHERE run_id=? AND stripe_id=? AND attempt=? AND sub_index=?",
                    (status, run_id, stripe_id, attempt, sub_index),
                )
                conn.commit()
                return cur.rowcount == 1

    def mark_shard_enqueued(
        self, run_id: str, stripe_id: str, attempt: int, sub_index: int,
        now: Optional[float] = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE shards SET phase5_status='enqueued', phase5_enqueued_at=? "
                    "WHERE run_id=? AND stripe_id=? AND attempt=? AND sub_index=?",
                    (now, run_id, stripe_id, attempt, sub_index),
                )
                conn.commit()

    def mark_shard_acked(
        self, run_id: str, stripe_id: str, attempt: int, sub_index: int,
        now: Optional[float] = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE shards SET phase5_status='acked', phase5_acked_at=? "
                    "WHERE run_id=? AND stripe_id=? AND attempt=? AND sub_index=?",
                    (now, run_id, stripe_id, attempt, sub_index),
                )
                conn.commit()

    def mark_shard_local_deleted(
        self, run_id: str, stripe_id: str, attempt: int, sub_index: int,
        now: Optional[float] = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE shards SET local_cleanup_status='deleted', local_deleted_at=? "
                    "WHERE run_id=? AND stripe_id=? AND attempt=? AND sub_index=?",
                    (now, run_id, stripe_id, attempt, sub_index),
                )
                conn.commit()

    def set_remote_delete(
        self, run_id: str, stripe_id: str, attempt: int, sub_index: int,
        status: str, error: Optional[str], deleted_at: Optional[float],
    ) -> None:
        """SC1: durable remote-deletion status. attempts increments on every try."""
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    """UPDATE shards
                       SET remote_delete_status=?,
                           remote_delete_attempts=remote_delete_attempts+1,
                           remote_delete_error=?, remote_deleted_at=?
                       WHERE run_id=? AND stripe_id=? AND attempt=? AND sub_index=?""",
                    (status, error, deleted_at, run_id, stripe_id, attempt, sub_index),
                )
                conn.commit()

    def get_shard(
        self, run_id: str, stripe_id: str, attempt: int, sub_index: int
    ) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM shards WHERE run_id=? AND stripe_id=? "
                "AND attempt=? AND sub_index=?",
                (run_id, stripe_id, attempt, sub_index),
            ).fetchone()
        return dict(row) if row is not None else None

    def set_stripe_fields(self, run_id: str, stripe_id: str, **fields: Any) -> None:
        """Small typed UPDATE for stripe columns (phase_degraded, claimed_by, ...)."""
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [run_id, stripe_id]
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    f"UPDATE stripes SET {cols} WHERE run_id=? AND stripe_id=?", vals)
                conn.commit()

    def held_reservations(
        self, run_id: str, stripe_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """All still-held reservations for a run (optionally narrowed to one
        stripe/attempt) — the set to release on attempt cleanup or trial abort."""
        q = "SELECT * FROM reservations WHERE status='held' AND run_id=?"
        params: List[Any] = [run_id]
        if stripe_id is not None:
            q += " AND stripe_id=?"; params.append(stripe_id)
        if attempt is not None:
            q += " AND attempt=?"; params.append(attempt)
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    # ----- trial lifecycle (Blocker 2, L3) ----------------------------------
    def create_trial(
        self, run_id: str, trial_number: int, now: Optional[float] = None
    ) -> None:
        now = time.time() if now is None else now
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO trials
                       (run_id, trial_number, state, created_at)
                       VALUES (?,?, 'running', ?)""",
                    (run_id, trial_number, now),
                )
                conn.commit()

    def get_trial(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trials WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def set_trial_context(
        self, run_id: str, ctx: Dict[str, Any], now: Optional[float] = None
    ) -> None:
        """D0 / Blocker 1: persist the trial-GLOBAL immutable context ONCE per run_id
        via COMPARE-AND-INSERT under the write lock in a single transaction.

          first write           -> insert;
          identical replay      -> idempotent no-op (row unchanged, no raise);
          CONFLICTING re-serve   -> raise MinerMetadataError, ORIGINAL row unchanged.

        `INSERT OR IGNORE` is NO LONGER relied on to enforce immutability (it prevents
        *mutation* but silently accepts a *conflicting* context); it survives only as
        the concurrency-safe insert primitive INSIDE the transaction — a losing
        concurrent (cross-process) insert is then detected by re-reading and comparing,
        so a race resolves to identical->no-op or conflict->raise, never a silent
        divergence where new work runs one config while manifests publish another.

        The read-compare-insert is transactionally protected (self._write_lock AND one
        DB transaction) so two concurrent initializations of the same run_id cannot
        race between the get and the insert. Comparison is SEMANTIC (the same field set
        get_trial_context returns, round-tripped through the same JSON encode/decode)
        so an identical replay compares equal regardless of key spacing or numeric
        string form. Mandatory trial-global/provenance fields must be present and
        non-None, else this fails closed BEFORE any stripe work rather than letting an
        incomplete `{}` reach Phase 5 later. sessions is JSON-encoded (None → [])."""
        now = time.time() if now is None else now
        missing = [k for k in _TRIAL_GLOBAL_FIELDS
                   if k != "sessions" and ctx.get(k) is None]
        missing += [k for k in _PROVENANCE_FIELDS if not ctx.get(k)]
        if missing:
            raise MinerMetadataError(
                f"trial_context for {run_id!r} missing mandatory field(s) {missing!r}; "
                f"refusing to persist an incomplete immutable context (fail-closed)."
            )
        sessions = ctx.get("sessions")
        sessions_json = json.dumps(sessions if sessions is not None else [])
        new_canon = _canonicalize_trial_context(ctx)
        with self._write_lock:
            with self._conn() as conn:
                existing = conn.execute(
                    "SELECT * FROM trial_context WHERE run_id=?", (run_id,)
                ).fetchone()
                if existing is None:
                    # Concurrency-safe insert primitive; a losing concurrent insert is
                    # caught by the re-read + compare below (never assume our row won).
                    conn.execute(
                        """INSERT OR IGNORE INTO trial_context
                           (run_id, trial_number, window_size, offset_val, sessions_json,
                            skip_min, skip_max, prng_base, forward_threshold,
                            reverse_threshold, dataset_sha256, residue_sha256, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (run_id, int(ctx["trial_number"]), int(ctx["window_size"]),
                         int(ctx["offset"]), sessions_json,
                         int(ctx["skip_min"]), int(ctx["skip_max"]), str(ctx["prng_base"]),
                         float(ctx["forward_threshold"]), float(ctx["reverse_threshold"]),
                         str(ctx["dataset_sha256"]), str(ctx["residue_sha256"]), now),
                    )
                    existing = conn.execute(
                        "SELECT * FROM trial_context WHERE run_id=?", (run_id,)
                    ).fetchone()
                existing_canon = _canonicalize_trial_context(
                    _trial_context_row_to_ctx(existing))
                if existing_canon != new_canon:
                    # Conflict: leave the original row untouched (INSERT OR IGNORE is a
                    # no-op when a row exists) and fail closed BEFORE any stripe work.
                    raise MinerMetadataError(
                        f"conflicting immutable trial context for run_id={run_id!r}: a "
                        f"different window_size/offset/skip/prng_base/threshold/provenance "
                        f"was already persisted; refusing to mutate (fail-closed)."
                    )
                conn.commit()

    def get_trial_context(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Reconstruct the trial-global immutable context from the durable ledger.
        Returns None when none was persisted (a bare unit-test path)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trial_context WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return _trial_context_row_to_ctx(row)

    def mark_trial_aborted(
        self, run_id: str, abort_event_id: str, now: Optional[float] = None
    ) -> bool:
        """Persist the terminal abort. Defect 5: terminal transitions happen ONLY
        from state='running', so committed and aborted are mutually exclusive — a
        COMMITTED trial can NEVER be flipped to aborted. Returns True only for the
        FIRST transition to aborted (exactly one abort event); a trial already
        aborted returns False and keeps its original event id."""
        now = time.time() if now is None else now
        with self._write_lock:
            with self._conn() as conn:
                cur = conn.execute(
                    """UPDATE trials
                       SET state='aborted', abort_event_id=?,
                           abort_cleanup_status='pending', finalized_at=?
                       WHERE run_id=? AND state='running'""",
                    (abort_event_id, now, run_id),
                )
                conn.commit()
                return cur.rowcount == 1

    def set_trial_cleanup_status(self, run_id: str, status: str) -> None:
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE trials SET abort_cleanup_status=? WHERE run_id=?",
                    (status, run_id),
                )
                conn.commit()

    def set_trial_commit_status(self, run_id: str, status: str) -> None:
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE trials SET commit_delivery_status=? WHERE run_id=?",
                    (status, run_id),
                )
                conn.commit()

    def mark_trial_committed(
        self, run_id: str, commit_event_id: str, now: Optional[float] = None
    ) -> bool:
        """Defect 5: commit is a terminal transition ONLY from state='running'
        (mutually exclusive with abort — an aborted trial can NEVER become
        committed, and a committed one stays committed). Records the immutable
        commit event id + a durable 'pending' delivery status. Returns True on the
        FIRST commit transition."""
        now = time.time() if now is None else now
        with self._write_lock:
            with self._conn() as conn:
                cur = conn.execute(
                    """UPDATE trials
                       SET state='committed', commit_event_id=?,
                           commit_delivery_status='pending', finalized_at=?
                       WHERE run_id=? AND state='running'""",
                    (commit_event_id, now, run_id),
                )
                conn.commit()
                return cur.rowcount == 1

    def all_stripes(self, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM stripes WHERE run_id=? ORDER BY stripe_id", (run_id,)
            ).fetchall()]

    def expired_claimed_stripes(
        self, run_id: str, now: float
    ) -> List[Dict[str, Any]]:
        """Stripes whose COMPUTE lease has expired (state='claimed'). staging
        stripes are excluded — they have their own timeout (Blocker 5)."""
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                """SELECT * FROM stripes
                   WHERE run_id=? AND state=? AND lease_expires_at IS NOT NULL
                     AND lease_expires_at < ? ORDER BY stripe_id""",
                (run_id, ST_CLAIMED, now),
            ).fetchall()]

    def cancel_active_stripes(self, run_id: str) -> None:
        """Mark every pending/claimed/staging stripe cancelled (whole-trial abort).
        done/failed/cancelled are left as-is."""
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    """UPDATE stripes SET state=?
                       WHERE run_id=? AND state IN (?,?,?)""",
                    (ST_CANCELLED, run_id, ST_PENDING, ST_CLAIMED, ST_STAGING),
                )
                conn.commit()

    # ----- worker registry (Decision A + Blocker 7 quarantine) --------------
    def upsert_worker(
        self,
        worker_id: str,
        hostname: str,
        backend: str,
        seed_caps: Dict[str, int],
        supported_variants: List[str],
        node_config: "NodeConfig",
        status: str,
        quarantine_reason: Optional[str],
        now: Optional[float] = None,
    ) -> None:
        """Durably record a registration + its eligibility. A quarantined worker
        is registered-but-ineligible and remains visible here (never silently
        dropped, never a silently-picked cap)."""
        now = time.time() if now is None else now
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO workers
                       (worker_id, hostname, backend, seed_caps_json,
                        supported_variants_json, spool_root, ssh_address,
                        ssh_user, status, quarantine_reason, registered_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (worker_id, hostname, backend,
                     json.dumps(seed_caps, sort_keys=True),
                     json.dumps(list(supported_variants)),
                     node_config.spool_root, node_config.ssh_address,
                     node_config.ssh_user, status, quarantine_reason, now),
                )
                conn.commit()

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM workers WHERE worker_id=?", (worker_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    # ----- stripe lifecycle -------------------------------------------------
    def add_stripe(
        self,
        run_id: str,
        stripe_id: str,
        seed_start: int,
        seed_count: int,
        family_name: str,
        phase: int,
        now: Optional[float] = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO stripes
                       (run_id, stripe_id, seed_start, seed_count, state,
                        phase, family_name, created_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (run_id, stripe_id, seed_start, seed_count, ST_PENDING,
                     phase, family_name, now),
                )
                conn.commit()

    def claim_stripe(
        self,
        run_id: str,
        stripe_id: str,
        worker_id: str,
        attempt: int,
        expected_substripes: int,
        lease_expires_at: float,
    ) -> bool:
        """pending|failed -> claimed. Records the assignment attempt (L1 authority)
        and the expected sub-stripe count (L8). Returns True on transition."""
        with self._write_lock:
            with self._conn() as conn:
                cur = conn.execute(
                    """UPDATE stripes
                       SET state=?, claimed_by=?, current_attempt=?,
                           expected_substripes=?, lease_expires_at=?,
                           stripe_complete_seen=0, substripes_done=NULL,
                           survivors_total=NULL
                       WHERE run_id=? AND stripe_id=? AND state IN (?,?)""",
                    (ST_CLAIMED, worker_id, attempt, expected_substripes,
                     lease_expires_at, run_id, stripe_id, ST_PENDING, ST_FAILED),
                )
                conn.commit()
                return cur.rowcount == 1

    def renew_lease(
        self, run_id: str, stripe_id: str, worker_id: str, new_expiry: float
    ) -> bool:
        """Heartbeats renew leases ONLY while `claimed` and only for the bound
        worker (Blocker 5). A `staging` stripe's lease is never renewed."""
        with self._write_lock:
            with self._conn() as conn:
                cur = conn.execute(
                    """UPDATE stripes SET lease_expires_at=?
                       WHERE run_id=? AND stripe_id=? AND state=? AND claimed_by=?""",
                    (new_expiry, run_id, stripe_id, ST_CLAIMED, worker_id),
                )
                conn.commit()
                return cur.rowcount == 1

    def reclaim_expired_leases(
        self, run_id: str, now: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Reclaim expired COMPUTE leases. Applies ONLY to state='claimed'
        (Blocker 5: `staging` has its own timeout and is never compute-reclaimed,
        so a stripe whose GPU work finished is not duplicately reassigned).

        Fences the superseded assignment by bumping staging_generation (L5) and
        clearing claim fields back to pending. The phase-specific fail-vs-retry
        policy (Blocker 3) is applied by the coordinator in Stage 4; here we only
        surface + requeue the expired compute claims."""
        now = time.time() if now is None else now
        reclaimed: List[Dict[str, Any]] = []
        with self._write_lock:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT stripe_id, claimed_by, current_attempt,
                              staging_generation
                       FROM stripes
                       WHERE run_id=? AND state=? AND lease_expires_at IS NOT NULL
                         AND lease_expires_at < ?""",
                    (run_id, ST_CLAIMED, now),
                ).fetchall()
                for row in rows:
                    conn.execute(
                        """UPDATE stripes
                           SET state=?, claimed_by=NULL, lease_expires_at=NULL,
                               staging_generation=staging_generation+1
                           WHERE run_id=? AND stripe_id=?""",
                        (ST_PENDING, run_id, row["stripe_id"]),
                    )
                    reclaimed.append(dict(row))
                conn.commit()
        return reclaimed

    def set_stripe_state(
        self, run_id: str, stripe_id: str, state: str
    ) -> None:
        with self._write_lock:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE stripes SET state=? WHERE run_id=? AND stripe_id=?",
                    (state, run_id, stripe_id),
                )
                conn.commit()

    # ----- shard-level results (Blocker 1) ----------------------------------
    def record_substripe_result(
        self,
        run_id: str,
        stripe_id: str,
        attempt: int,
        sub_index: int,
        worker_id: str,
        seed_start: int,
        seed_count: int,
        survivor_count: int,
        remote_spool_path: Optional[str] = None,
        size_bytes: Optional[int] = None,
        sha256: Optional[str] = None,
        now: Optional[float] = None,
    ) -> bool:
        """Insert ONE shard row keyed (run_id, stripe_id, attempt, sub_index).
        A duplicate (attempt, sub_index) is REJECTED (not overwritten) — a worker
        must never emit the same sub_index twice; overwriting would silently mask
        a coverage/accounting corruption. Returns True if inserted, False if a
        duplicate was rejected or the stripe is not in an accepting state."""
        now = time.time() if now is None else now
        with self._write_lock:
            with self._conn() as conn:
                st = conn.execute(
                    "SELECT state FROM stripes WHERE run_id=? AND stripe_id=?",
                    (run_id, stripe_id),
                ).fetchone()
                if st is None or st["state"] not in (ST_CLAIMED, ST_STAGING):
                    return False
                dup = conn.execute(
                    """SELECT 1 FROM shards
                       WHERE run_id=? AND stripe_id=? AND attempt=? AND sub_index=?""",
                    (run_id, stripe_id, attempt, sub_index),
                ).fetchone()
                if dup is not None:
                    return False
                conn.execute(
                    """INSERT INTO shards
                       (run_id, stripe_id, attempt, sub_index, worker_id,
                        seed_start, seed_count, survivor_count,
                        remote_spool_path, size_bytes, sha256,
                        staging_status, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (run_id, stripe_id, attempt, sub_index, worker_id,
                     seed_start, seed_count, survivor_count,
                     remote_spool_path, size_bytes, sha256, SH_PENDING, now),
                )
                conn.commit()
                return True

    def record_stripe_complete(
        self,
        run_id: str,
        stripe_id: str,
        attempt: int,
        worker_id: str,
        substripes_done: int,
        survivors_total: int,
    ) -> bool:
        """StripeComplete arrived: the GPU is free but transfers may still run,
        so claimed -> staging (Blocker 5), NOT straight to done. Stores the two
        authoritative reconciliation inputs (substripes_done, survivors_total).
        Clears the compute lease so the stripe is not compute-reclaimed while
        staging. Returns True on transition."""
        with self._write_lock:
            with self._conn() as conn:
                cur = conn.execute(
                    """UPDATE stripes
                       SET state=?, stripe_complete_seen=1,
                           substripes_done=?, survivors_total=?,
                           lease_expires_at=NULL
                       WHERE run_id=? AND stripe_id=? AND state=?
                         AND claimed_by=? AND current_attempt=?""",
                    (ST_STAGING, substripes_done, survivors_total,
                     run_id, stripe_id, ST_CLAIMED, worker_id, attempt),
                )
                conn.commit()
                return cur.rowcount == 1

    def mark_shard_verified(
        self,
        run_id: str,
        stripe_id: str,
        attempt: int,
        sub_index: int,
        local_staged_path: Optional[str] = None,
        size_bytes: Optional[int] = None,
        sha256: Optional[str] = None,
        now: Optional[float] = None,
    ) -> bool:
        """Mark a shard locally staged + hash-verified. In Stage 1 this is the
        minimal staging stub the harness drives directly; Stage 3 fills the real
        fetch/verify pipeline behind an injectable adapter and calls this."""
        now = time.time() if now is None else now
        sets = ["staging_status=?", "verified_at=?"]
        vals: List[Any] = [SH_VERIFIED, now]
        if local_staged_path is not None:
            sets.append("local_staged_path=?")
            vals.append(local_staged_path)
        if size_bytes is not None:
            sets.append("size_bytes=?")
            vals.append(size_bytes)
        if sha256 is not None:
            sets.append("sha256=?")
            vals.append(sha256)
        vals.extend([run_id, stripe_id, attempt, sub_index])
        with self._write_lock:
            with self._conn() as conn:
                cur = conn.execute(
                    f"""UPDATE shards SET {', '.join(sets)}
                        WHERE run_id=? AND stripe_id=? AND attempt=? AND sub_index=?""",
                    vals,
                )
                conn.commit()
                return cur.rowcount == 1

    # ----- reads ------------------------------------------------------------
    def get_stripe(self, run_id: str, stripe_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM stripes WHERE run_id=? AND stripe_id=?",
                (run_id, stripe_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_shards(
        self, run_id: str, stripe_id: str, attempt: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if attempt is None:
                rows = conn.execute(
                    """SELECT * FROM shards WHERE run_id=? AND stripe_id=?
                       ORDER BY attempt, sub_index""",
                    (run_id, stripe_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM shards
                       WHERE run_id=? AND stripe_id=? AND attempt=?
                       ORDER BY sub_index""",
                    (run_id, stripe_id, attempt),
                ).fetchall()
        return [dict(r) for r in rows]

    def stripes_by_state(self, run_id: str, state: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM stripes WHERE run_id=? AND state=? ORDER BY stripe_id",
                (run_id, state),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Transfer adapter (Binding Decision B) — coordinator-owned; no new protocol
# message. fetch_remote() pulls a remote spool file to a Zeus-local temp path;
# delete_remote() releases the remote spool AFTER local verification. The harness
# injects a stub; the real adapter (scp/ssh via the NODE CONFIG's address/user)
# lands with the multi-rig work.
# ---------------------------------------------------------------------------
class TransferAdapter:
    def fetch_remote(self, node_config: "NodeConfig", remote_path: str,
                     local_temp_path: str) -> None:
        raise NotImplementedError

    def delete_remote(self, node_config: "NodeConfig", remote_path: str) -> None:
        raise NotImplementedError


class Phase5Sink:
    """Injected Phase-5 interface (L6). Phase 4 hands provisional shard manifests
    and trial lifecycle events across this boundary; the harness injects a stub,
    Phase 5 implements it for real.

    abort_trial() is SYNCHRONOUS (L7 — Team Beta binding ruling, Option A): a
    successful return GUARANTEES Phase 5 has stopped/finished all reads for the
    trial, references no trial-owned staged path, has discarded all provisional
    shards + partial assembly, and will harmlessly ignore later stale manifests.
    There is deliberately NO TrialAbortAck / async callback / abort-timeout state
    machine (the async variant is not approved for Phase 4)."""

    def publish_shard(self, manifest: Dict[str, Any]) -> None:
        raise NotImplementedError

    def commit_trial(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError

    def abort_trial(self, event: Dict[str, Any]) -> None:
        raise NotImplementedError


class StagingError(Exception):
    """Staging misconfiguration or precondition failure (no staging_dir/adapter,
    unknown stripe, escaped spool path)."""


class StagingConfigurationError(StagingError):
    """[S172 Staging Part B, Beta binding ruling §2] The coordinator's staging
    CONFIGURATION is invalid: missing, conflicting, non-absolute, unwritable or
    capacity-invalid.

    NON-RETRYABLE, and deliberately NARROW. A configuration defect is permanent —
    retrying it burns a Q3 retry budget and surfaces downstream as a misleading
    `MinerIngressError` naming thresholds, which is what made the 2026-08-04
    diagnosis take an hour.

    ⚠ This subtype exists so the classification change can be bounded to exactly
    these conditions. `StagingError` itself, and every transient transfer,
    filesystem and capacity-PRESSURE condition (StagingBackPressure,
    StagingHashMismatch, StagingTimeout), KEEP their existing retryable
    classification and their existing Blocker-3 matrix rows.
    """


class TrialAborted(Exception):
    """A lifecycle action was attempted on a terminally-aborted trial
    (e.g. TrialCommit after TrialAbort)."""


class StagingBackPressure(Exception):
    """Reservation denied: granting it would exceed a Zeus-local high-water mark."""


class StagingHashMismatch(Exception):
    """Staged/transferred bytes do not match the advertised size+sha256 (§15).
    The sub-stripe fails to the retry path; delete_remote is NOT invoked."""


class StagingTimeout(Exception):
    """A staging task (fetch/verify/rename) exceeded staging_timeout (Defect 4).
    Its file is removed and reservation released, then the stripe is routed
    through the phase-specific matrix."""


class ThresholdProvenanceError(Exception):
    """[S172 D6, Beta commit ruling] The physical evidence that the requested
    sieve threshold reached the kernel is missing, mismatched or internally
    inconsistent for a D6-generated assignment.

    NON-RECOVERABLE for the trial: raised by the parent BEFORE commit, so a run
    whose kernel filter is unproven can never reach Phase-5 assembly, candidate
    ingress, accumulator mutation or `finalize_run`. It is the completion of the
    threshold correctness fix — without it D6 could certify a generation after
    having observed that the effective threshold was absent or wrong.

    Carries the PRIMARY diagnostic. Abort/cleanup failures that follow must never
    replace or obscure it (the D5 REV2 §7 primary-exception discipline)."""


class MinerMetadataError(Exception):
    """D0 fail-closed: a manifest would publish to Phase 5 with an absent or empty
    mandatory metadata field (or an unresolvable workflow phase). The publish is
    ABORTED — no `{}` trial_metadata is ever handed to Phase 5 (§0.5, gate D0-6)."""


# ---------------------------------------------------------------------------
# D0 — trial-metadata seam (Phase-4 correction, [TB-R3, TB-R1 seam])
#
# Every ShardReadyManifest published to Phase 5 must carry a complete, immutable
# trial_metadata projection so Phase 5 can populate the NPZ window/offset/skip/
# trial/prng fields WITHOUT re-deriving identity from spool contents. The projection
# is reconstructed durably from the ledger (trial_context row + the stripe's own
# persisted phase/family_name), so it survives a coordinator restart and is
# identical across every one of a run's manifests where trial-global.
# ---------------------------------------------------------------------------

# Trial-GLOBAL immutable fields (identical across every manifest of one run_id).
_TRIAL_GLOBAL_FIELDS = (
    "trial_number", "window_size", "offset", "sessions", "skip_min", "skip_max",
    "prng_base", "forward_threshold", "reverse_threshold",
)
# Provenance (non-NPZ) fields carried for auditability, also trial-global.
_PROVENANCE_FIELDS = ("dataset_sha256", "residue_sha256")
# Phase/stripe-SPECIFIC fields (correct per stripe, derived from the ledger).
_PHASE_SPECIFIC_FIELDS = (
    "workflow_phase", "family_name", "prng_type", "direction", "skip_mode",
    "threshold_used",
)

# The minimum mandatory manifest metadata (§ "Mandatory manifest metadata"). Every
# published shard MUST carry all of these (non-empty) or publication fails closed.
MANDATORY_MANIFEST_METADATA = (
    "trial_number", "window_size", "offset", "sessions", "skip_min", "skip_max",
    "prng_base", "prng_type", "family_name", "direction", "skip_mode",
    "workflow_phase", "forward_threshold", "reverse_threshold",
)
# Provenance fields are also required on the manifest (retained separately, non-NPZ).
_MANDATORY_PROVENANCE = ("dataset_sha256", "residue_sha256")

# String-identity fields whose EMPTY string is as invalid as absence.
_NON_EMPTY_STRING_FIELDS = frozenset({
    "prng_base", "prng_type", "family_name", "direction", "skip_mode",
    "dataset_sha256", "residue_sha256",
})

# Mandatory keys the raw serve `context` MUST supply to build the immutable trial
# context (Blocker 2: no numeric/family fallback may substitute for a missing field).
# sessions is intentionally excluded — it is optional and normalized None -> [].
_SERVE_CONTEXT_REQUIRED = (
    "trial_number", "window_size", "offset", "skip_min", "skip_max",
    "prng_base", "forward_threshold", "reverse_threshold",
)


def _trial_context_row_to_ctx(row: Any) -> Dict[str, Any]:
    """Map a durable `trial_context` row to the SAME semantic dict get_trial_context
    returns (11 trial-global + provenance), so an existing row and a fresh ctx can be
    canonicalized and compared field-for-field (Blocker 1)."""
    d = dict(row)
    return {
        "trial_number":      d["trial_number"],
        "window_size":       d["window_size"],
        "offset":            d["offset_val"],
        "sessions":          json.loads(d["sessions_json"]),
        "skip_min":          d["skip_min"],
        "skip_max":          d["skip_max"],
        "prng_base":         d["prng_base"],
        "forward_threshold": d["forward_threshold"],
        "reverse_threshold": d["reverse_threshold"],
        "dataset_sha256":    d["dataset_sha256"],
        "residue_sha256":    d["residue_sha256"],
    }


def _canonicalize_trial_context(ctx: Dict[str, Any]) -> str:
    """Canonical SEMANTIC form of a trial context for the immutability comparison
    (Blocker 1). Each field is coerced to the SAME type the durable row stores
    (int / float / str; sessions as a decoded list; None sessions -> []), then the
    whole dict is round-tripped through a sorted-key JSON encode so an identical
    replay compares EQUAL regardless of JSON key spacing or numeric string form.
    Comparison is by VALUE, never raw row bytes."""
    sessions = ctx.get("sessions")
    return json.dumps(
        {
            "trial_number":      int(ctx["trial_number"]),
            "window_size":       int(ctx["window_size"]),
            "offset":            int(ctx["offset"]),
            "sessions":          sessions if sessions is not None else [],
            "skip_min":          int(ctx["skip_min"]),
            "skip_max":          int(ctx["skip_max"]),
            "prng_base":         str(ctx["prng_base"]),
            "forward_threshold": float(ctx["forward_threshold"]),
            "reverse_threshold": float(ctx["reverse_threshold"]),
            "dataset_sha256":    str(ctx["dataset_sha256"]),
            "residue_sha256":    str(ctx["residue_sha256"]),
        },
        sort_keys=True,
    )


def build_trial_context_from_serve(
    context: Dict[str, Any], dataset_sha256: str, residue_sha256: str
) -> Dict[str, Any]:
    """Blocker 2 fail-closed seam: project the durable trial-context dict from the raw
    serve `context` with NO fallback substitution for any mandatory field.

    The pre-fix serve path substituted concrete values (family_name for a missing
    prng_base; 0 / -1 / 0.0 for missing numerics) BEFORE the mandatory-field guard,
    turning a missing field into apparently-present-but-semantically-malformed
    metadata that slipped through. Here every mandatory field is required-key access;
    a missing key (or a None/empty prng_base) raises MinerMetadataError BEFORE any
    stripe assignment/dispatch, so a missing mandatory field can never reach Phase 5
    as present-but-wrong metadata. dataset_sha256/residue_sha256 are coordinator-
    computed and passed in (not defaulted)."""
    missing = [k for k in _SERVE_CONTEXT_REQUIRED if context.get(k) is None]
    if missing:
        raise MinerMetadataError(
            f"serve context missing mandatory field(s) {sorted(missing)!r}; refusing "
            f"to build an immutable trial context with fallback substitutes (fail-closed)."
        )
    prng_base = context["prng_base"]          # required-key access, no family fallback
    if prng_base is None or (isinstance(prng_base, str) and prng_base.strip() == ""):
        raise MinerMetadataError(
            "prng_base missing/empty in trial context (fail-closed; no family_name fallback)."
        )
    return {
        "trial_number":      int(context["trial_number"]),
        "window_size":       int(context["window_size"]),
        "offset":            int(context["offset"]),
        "sessions":          context.get("sessions"),
        "skip_min":          int(context["skip_min"]),
        "skip_max":          int(context["skip_max"]),
        "prng_base":         prng_base,
        "forward_threshold": float(context["forward_threshold"]),
        "reverse_threshold": float(context["reverse_threshold"]),
        "dataset_sha256":    dataset_sha256,
        "residue_sha256":    residue_sha256,
    }


def workflow_phase_semantics(phase: int) -> Tuple[str, str]:
    """§6.8 workflow table → the (direction, skip_mode) identity of a workflow
    phase. EXPLICIT strings, derived from the phase number via the shared table —
    NOT inferred downstream from a numeric phase, and NOT hardcoded to any one base
    family (the same table holds for every prng_base). Hard-fails on an unknown
    phase (fail-closed, gate D0-3).

      1 → forward/constant   2 → reverse/constant
      3 → forward/variable   4 → reverse/variable
    """
    table = {
        1: ("forward", "constant"),
        2: ("reverse", "constant"),
        3: ("forward", "variable"),
        4: ("reverse", "variable"),
    }
    try:
        return table[int(phase)]
    except (KeyError, ValueError, TypeError):
        raise MinerMetadataError(
            f"unknown workflow phase {phase!r}: expected 1..4 (§6.8). Cannot resolve "
            f"direction/skip_mode — refusing to publish a manifest with inferred "
            f"identity."
        )


def derive_trial_metadata(
    trial_ctx: Dict[str, Any], stripe: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the ONE immutable trial_metadata projection for a shard's manifest from
    the durable ledger: trial-GLOBAL fields from the persisted `trial_context` row,
    phase-SPECIFIC fields from the stripe's own persisted `phase`/`family_name`.

    Direction / skip_mode come from workflow_phase_semantics (the §6.8 table via the
    resolved base — never hardcoded to Java LCG). prng_type is the canonical
    encoding key (`prng_base` for constant, `prng_base + '_hybrid'` for variable),
    matching utils/prng_encoding's contract; direction is carried as its own
    explicit field, not folded into prng_type. threshold_used is the forward or
    reverse threshold selected by direction.

    Raises MinerMetadataError (fail-closed) if the phase is unresolvable. Field
    presence/emptiness is enforced separately by validate_trial_metadata so the
    single failure point covers both this path and a hand-built manifest.
    """
    phase = stripe.get("phase")
    family_name = stripe.get("family_name")
    direction, skip_mode = workflow_phase_semantics(phase)
    prng_base = trial_ctx.get("prng_base")
    prng_type = prng_base if skip_mode == "constant" else f"{prng_base}_hybrid"
    threshold_used = (trial_ctx.get("forward_threshold") if direction == "forward"
                      else trial_ctx.get("reverse_threshold"))
    meta: Dict[str, Any] = {
        # trial-global (identical across the run's manifests)
        "trial_number":      trial_ctx.get("trial_number"),
        "window_size":       trial_ctx.get("window_size"),
        "offset":            trial_ctx.get("offset"),
        "sessions":          trial_ctx.get("sessions"),
        "skip_min":          trial_ctx.get("skip_min"),
        "skip_max":          trial_ctx.get("skip_max"),
        "prng_base":         prng_base,
        "forward_threshold": trial_ctx.get("forward_threshold"),
        "reverse_threshold": trial_ctx.get("reverse_threshold"),
        # phase/stripe-specific (correct per stripe)
        "workflow_phase":    int(phase) if phase is not None else None,
        "family_name":       family_name,
        "prng_type":         prng_type,
        "direction":         direction,
        "skip_mode":         skip_mode,
        "threshold_used":    threshold_used,
        # provenance (non-NPZ)
        "dataset_sha256":    trial_ctx.get("dataset_sha256"),
        "residue_sha256":    trial_ctx.get("residue_sha256"),
    }
    return meta


def validate_trial_metadata(meta: Dict[str, Any]) -> None:
    """Fail-closed gate (§0.5, gate D0-6): a manifest may NOT publish to Phase 5
    unless every mandatory field is present and non-empty. A missing key, a None
    value, or an empty string in an identity/provenance field RAISES
    MinerMetadataError before publication — the coordinator never emits a `{}` (or
    partially-populated) trial_metadata to the sink. Numeric 0 / -1 are legitimate
    values (offset 0, skip_min 0, trial_number -1) and pass."""
    required = list(MANDATORY_MANIFEST_METADATA) + list(_MANDATORY_PROVENANCE)
    missing = [k for k in required if k not in meta or meta[k] is None]
    if missing:
        raise MinerMetadataError(
            f"manifest trial_metadata missing mandatory field(s) {missing!r}; "
            f"refusing to publish to Phase 5 (fail-closed)."
        )
    empty = [k for k in _NON_EMPTY_STRING_FIELDS
             if k in meta and isinstance(meta[k], str) and meta[k].strip() == ""]
    if empty:
        raise MinerMetadataError(
            f"manifest trial_metadata has empty identity field(s) {empty!r}; "
            f"refusing to publish to Phase 5 (fail-closed)."
        )


@dataclass
class StagingTask:
    """Immutable identity carried by every async staging task/callback (L5):
    (run_id, stripe_id, attempt, sub_index, staging_generation). The callback
    re-checks this against the live ledger before ANY rename/ledger-update/
    enqueue/delete_remote so a superseded attempt's completion cannot publish."""
    reservation_id: int
    run_id: str
    stripe_id: str
    attempt: int
    sub_index: int
    staging_generation: int
    kind: str                          # 'inline' | 'remote'
    temp_path: str
    staged_path: str
    expected_size: int
    expected_sha256: str
    payload_bytes: Optional[bytes] = None
    remote_spool_path: Optional[str] = None
    node_config: Optional[NodeConfig] = None


# ---------------------------------------------------------------------------
# Coordinator (Stage-1 core + Stage-2 identity + Stage-3 staging)
# ---------------------------------------------------------------------------
class RangeMinerCoordinator:
    """Owns the ledger, config, and the stripe state machine. Stage 2 adds
    connection-bound identity + registration validation + L1 fencing; Stage 3
    the staging pipeline; Stage 4 the retry matrix + trial lifecycle +
    Phase5Sink. This class deliberately performs NO Phase-5 assembly."""

    def __init__(
        self,
        config: CoordinatorConfig,
        ledger: MinerLedger,
        transfer: Optional[TransferAdapter] = None,
        phase5_sink: Optional["Phase5Sink"] = None,
    ):
        self.config = config
        self.ledger = ledger
        self.transfer = transfer
        # Injected Phase-5 interface (L6). None -> the coordinator still tracks the
        # bounded queue + lifecycle; sink calls are simply skipped in that case.
        self.phase5_sink = phase5_sink
        # worker_id -> live WorkerConnection (Decision A binding)
        self.connections: Dict[str, WorkerConnection] = {}
        # Manifests published to Phase 5 this run (attempt-scoped, Blocker 2).
        # Test-inspectable; the real handoff is phase5_sink.publish_shard.
        self.enqueued: List[Dict[str, Any]] = []
        # Injectable final-rename primitive so the atomic-write-failure path is
        # exercisable (gate 36); production default is os.replace.
        self._atomic_replace = os.replace
        # [S172 D6] Threshold provenance (Beta §3) AND the parent-side
        # fail-closed gate over it (Beta's commit ruling). Three legs, recorded
        # separately and never reconciled by assumption:
        #   requested — WindowConfig.forward/reverse_threshold (trial context)
        #   payload   — what build_stripe_assign_payload actually emitted
        #   effective — what the kernel actually filtered at, reported BACK by
        #               the worker off the real executor
        # ONE registry, keyed run_id -> (stripe_id, attempt), feeds BOTH the
        # audit record and the validator, so reported and enforced cannot drift.
        self._assignment_provenance: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
        self._provenance_lock = threading.RLock()
        # Defect 4: a BOUNDED staging executor separate from the abort-cleanup one,
        # so blocking fetch/verify/rename runs OFF the socket dispatch loop.
        self._staging_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._cleanup_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # Defect 4/5: serialize lifecycle mutations (finalize/publish/matrix/commit)
        # so concurrent staging-executor + dispatch threads cannot double-publish or
        # race a terminal transition. RLock: a thread may re-enter its own section.
        self._lifecycle_lock = threading.RLock()
        # Defect 2 (C3): a bounded admission semaphore so the number of retained
        # in-flight staging payloads (queued + active) is capped, not just active
        # threads. Sized to (staging_workers + staging_queue_depth), created lazily.
        self._staging_sem: Optional[threading.BoundedSemaphore] = None
        # Defect 3 (C3): stripe-level admission. `_admitted` maps a live attempt to
        # its committed high-water footprint (files, bytes) so a NEW attempt is only
        # admitted when its WHOLE attempt still fits alongside the committed ones —
        # an admitted attempt is then guaranteed to be able to stage all its
        # sub-stripes (no self-inflicted capacity deadlock). `_deferred` holds
        # sub-stripes whose attempt could not be admitted yet; they resume
        # (nonblocking) when capacity frees, instead of timing out into the matrix.
        self._admission_lock = threading.Lock()
        self._admitted: Dict[Tuple[str, str, int], Dict[str, int]] = {}
        self._deferred: List[tuple] = []
        # [S172-BP §1.2] PER-CONNECTION PAUSE STATE. `_paused_connections` maps a
        # connection key (the raw socket) to its pause record; insertion order IS
        # the FIFO resume order (Beta B.3 fairness). A paused connection holds AT
        # MOST ONE already-decoded envelope, in the reader thread — never here.
        #
        # This is deliberately PER CONNECTION and never global: pausing one reader
        # must not touch another connection's reader, the accept loop, or the serve
        # loop. The serve loop stays single-threaded and reads this state only for
        # the §1.4 lease exemption and the §1.5 capacity timeout.
        self._pause_lock = threading.Lock()
        self._paused_connections: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()
        # [S172-BP AMENDMENT F1] INGRESS RESUME CREDIT. A wake must CONSUME the
        # capacity observation that produced it, or one freed slot satisfies the
        # check repeatedly and wakes the whole paused fleet. At most ONE credit is
        # outstanding at a time; it is granted either by a capacity-release event
        # (`_grant_resume_credit`) or taken by the FIFO-head reader's own defensive
        # poll (`_try_self_resume`). `_resume_credit_holder` is the connection key
        # that owns the outstanding credit, so clearing is attributable and
        # idempotent.
        #
        # [S172-BP AMENDMENT F1-R — WHERE THE RESERVATION ENDS]
        # Round 1 cleared the credit at `inbound.put`. That is INGRESS, not
        # consumption: the freed staging slot is consumed only when the serve loop
        # later dispatches that envelope into `enqueue_staging`, and in the gap the
        # envelope is in `inbound` while the slot is still physically free — so the
        # next FIFO-head reader finds credits == 0 and `staging_can_accept()` true
        # ON THE SAME SLOT. Two wakes, one slot. The credit therefore now rides
        # WITH the envelope and is released at DISPOSITION by the single-threaded
        # serve path (Beta F1-R §4 i-iv): admission acquired, retained in the
        # bounded deferred queue, rejected by the existing identity/attempt/dedup/
        # terminal fence, or connection/trial terminated with the envelope
        # discarded. `_resume_credit_worker` / `_resume_credit_since` exist only so
        # the §4 metrics can report WHO holds an outstanding reservation and for
        # how long — they are never read by the credit arithmetic itself.
        #
        # [S172-BP AMENDMENT F1-R2a — THE CREDIT IS A TOKEN, NOT A SOCKET]
        # Round 2 released the reservation on `rawsock is holder` alone. That
        # identity admits the WRONG envelope: an OLDER, UNCREDITED result from the
        # credit-holder's own connection — one that was already sitting in
        # `inbound` before the pause ever happened — dispatches FIRST, is rejected
        # by the existing fence (so it consumes no capacity at all), and its
        # `finally` clears the credit. The credited envelope is still queued, the
        # slot is still physically free, and the next FIFO head wakes on it: F1's
        # two-wakes-one-slot defect, re-entered through an earlier arrival rather
        # than a later one. "The first result after resume" was never the right
        # identity — it excludes LATER traffic, not EARLIER traffic.
        # `_resume_credit_id` is therefore a monotonically increasing token minted
        # at GRANT, carried ON the envelope through `inbound`, and matched EXACTLY
        # at disposition. `_resume_credit_seq` only ever increases, so a token is
        # unique for the lifetime of the coordinator and can never be confused
        # with a later grant to the same socket.
        self._resume_credits_outstanding: int = 0
        self._resume_credit_holder: Any = None
        self._resume_credit_worker: Optional[str] = None
        self._resume_credit_since: Optional[float] = None
        self._resume_credit_id: Optional[int] = None
        self._resume_credit_seq: int = 0
        # [S172-BP AMENDMENT F2] LEASE-EXEMPTION RESUME GRACE. worker_id -> the
        # absolute time until which `process_lease_expiry` still skips that
        # worker's stripes AFTER its connection has left `_paused_connections`.
        # The window exists because a resumed reader DEREGISTERS FIRST, delivers
        # its held envelope, and only later delivers the heartbeat that was queued
        # behind it: without the grace, a pause of up to
        # `staging_capacity_timeout` (600 s) against a `compute_lease_timeout` of
        # 300 s leaves the stripe expirable in exactly that window. Written by the
        # reader thread under `_pause_lock` — a coordinator dict, NEVER the ledger,
        # so the reader rule ("touches NO ledger state") is preserved.
        self._capacity_resume_grace: Dict[str, float] = {}
        # Latched once the §1.5 bounded capacity timeout is observed, so the reader
        # threads and the serve loop cannot disagree about whether it fired.
        self._capacity_timeout_latched_at: Optional[float] = None
        # [S172-BP AMENDMENT F3] The TRIGGERING evidence, captured in the same
        # critical section that sets the latch. A reader can observe the latch,
        # deregister and exit before the serve loop builds the terminal reason, so
        # reading the LIVE registry then can truthfully report "0 connections
        # paused (none)" about a timeout that paused workers caused.
        self._capacity_timeout_snapshot: Optional[Dict[str, Any]] = None
        # [S172-BP §2] The DERIVED deferred bound for the current stage, set by
        # `derive_staging_deferred_bound` at stage setup. None => nothing derived
        # yet (a bare-API/gate call), in which case the accessor derives on demand
        # from live config + registered workers rather than falling back to a
        # constant.
        self._derived_deferred_bound: Optional[int] = None
        self._derived_bound_detail: Dict[str, Any] = {}
        self._fallback_bound_cache: Dict[Tuple[int, int], int] = {}
        # [S172-BP AMENDMENT F5] which of the three bounds a `_defer_locked`
        # refusal tripped: derived count / operator-override count / retained-bytes
        # high-water. The §1.6 invariant reason must name it — the three are
        # different defects with different owners.
        self._last_defer_refusal: Optional[str] = None
        # [S172-BP §4] Structured, grep-stable metrics. Every emission carries the
        # `[S172-BP]` prefix so gate_s172_prod_shape.py and operators can extract
        # the series from a raw run log.
        self._bp_lock = threading.Lock()
        self._bp: Dict[str, Any] = {
            "inbound_qsize_high_water": 0,
            "deferred_high_water": 0,
            "paused_now": 0,
            "paused_high_water": 0,
            "pause_events": 0,
            "pause_seconds_total": 0.0,
            "pause_seconds_max": 0.0,
            "staging_jobs_completed": 0,
            "capacity_timeout_terminations": 0,
            "capacity_invariant_terminations": 0,
            "trial_started_at": None,
        }
        # Defect 4 (C4): tracked registry of timed-out ('abandoned') fetch daemon
        # threads whose TransferAdapter did not honor cancellation, so permanent
        # network hangs cannot accumulate untracked threads across a 50-trial soak.
        self._orphan_lock = threading.Lock()
        self._orphan_fetch_threads: List[Tuple[str, threading.Thread]] = []

    def _staging_exec(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._staging_executor is None:
            workers = max(2, int(getattr(self.config, "staging_workers", 4)))
            self._staging_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="miner-staging")
        return self._staging_executor

    def _staging_slots(self) -> threading.BoundedSemaphore:
        if self._staging_sem is None:
            workers = max(2, int(getattr(self.config, "staging_workers", 4)))
            depth = max(0, int(getattr(self.config, "staging_queue_depth", 2)))
            self._staging_sem = threading.BoundedSemaphore(workers + depth)
        return self._staging_sem

    # ----- registration + identity (Decision A, Blocker 7, L4) --------------
    def _central_caps(self) -> Dict[str, int]:
        """The centrally-resolved cap config (§7 WATCHER manifest / §6 6-level
        precedence, resolved into CoordinatorConfig) that advertised caps are
        validated against — never a value silently picked from the worker."""
        return {
            "amd": self.config.seed_cap_amd,
            "nvidia": self.config.seed_cap_nvidia,
            "amd_hybrid": self.config.seed_cap_amd_hybrid,
            "nvidia_hybrid": self.config.seed_cap_nvidia_hybrid,
        }

    def _validate_caps(self, seed_caps: Dict[str, Any]) -> Optional[str]:
        """Validate ALL FOUR advertised caps against central config. Missing key,
        non-positive value, or mismatch -> a quarantine reason string; None = OK."""
        if not isinstance(seed_caps, dict):
            return "capabilities.seed_caps missing or not a dict"
        for key, expected in self._central_caps().items():
            if key not in seed_caps:
                return f"missing seed_cap '{key}'"
            val = seed_caps[key]
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                return f"non-positive/invalid seed_cap '{key}': {val!r}"
            if val != expected:
                return f"seed_cap '{key}'={val} != central config {expected}"
        return None

    def register_worker(
        self,
        *,
        worker_id: str,
        hostname: str,
        backend: str,
        capabilities: Optional[Dict[str, Any]],
        node_config: NodeConfig,
        now: Optional[float] = None,
        admission_reason: Optional[str] = None,
    ) -> WorkerConnection:
        """Bind a worker's connection and validate its advertised capabilities.
        A cap inconsistency quarantines the worker (registered-but-ineligible,
        durably visible in the workers table) rather than dropping it or picking
        a value. Returns the bound (possibly quarantined) connection.

        [RESOLVED EXECUTION SET — G-NO-INFERENCE] `admission_reason`, when the
        caller supplies one, is a refusal decided BEFORE this worker said
        anything about itself: it is not in the run's frozen execution set.
        Beta: *unknown miner workers must not become eligible merely because they
        connected.* It is deliberately expressed as a QUARANTINE and not as a
        dropped connection, because quarantine is the mechanism this coordinator
        already has for registered-but-ineligible, and it leaves a durable row
        naming the refusal instead of an unexplained disconnect. It composes with
        the capability check rather than replacing it — a worker can be both
        unlisted and misconfigured, and the record should say so."""
        capabilities = capabilities or {}
        seed_caps = capabilities.get("seed_caps") or {}
        variants = capabilities.get("supported_variants") or []
        cap_reason = self._validate_caps(seed_caps)
        if admission_reason and cap_reason:
            reason = f"{admission_reason} ALSO: {cap_reason}"
        else:
            reason = admission_reason or cap_reason
        status = "quarantined" if reason else "eligible"
        self.ledger.upsert_worker(
            worker_id, hostname, backend, seed_caps, variants, node_config,
            status, reason, now,
        )
        conn = WorkerConnection(
            worker_id=worker_id,
            hostname=hostname,
            backend=backend,
            seed_caps=dict(seed_caps),
            supported_variants=frozenset(variants),
            node_config=node_config,
            quarantined=bool(reason),
            quarantine_reason=reason,
        )
        self.connections[worker_id] = conn
        return conn

    def can_assign_variant(self, worker: Any, family_name: str) -> bool:
        """A stripe of `family_name` may be assigned only to an eligible worker
        that advertises that EXACT concrete variant (Blocker 7). WorkerRecord
        (Stage-1 direct use) carries no advertisement and is treated as eligible."""
        if getattr(worker, "quarantined", False):
            return False
        variants = getattr(worker, "supported_variants", None)
        if variants is None:
            return True
        return family_name in variants

    # ----- message-acceptance gate (Decision A + L1 stale-attempt fencing) --
    def accept_stripe_message(
        self,
        conn: WorkerConnection,
        run_id: str,
        stripe_id: str,
        msg_worker_id: str,
        permitted_states,
    ) -> Tuple[bool, Optional[str]]:
        """Accept a stripe-flow message ONLY when ALL hold (L1):
          connection.worker_id == message.worker_id       (Decision A)
          ledger.claimed_by     == message.worker_id
          ledger.current_attempt == the connection's recorded assignment attempt
          the stripe state permits that message type
        A reject touches NO ledger state — the caller logs and drops it."""
        if conn.worker_id != msg_worker_id:
            return (False, f"worker_id {msg_worker_id!r} != bound connection "
                           f"{conn.worker_id!r}")
        if conn.quarantined:
            return (False, f"worker {conn.worker_id} is quarantined: "
                           f"{conn.quarantine_reason}")
        stripe = self.ledger.get_stripe(run_id, stripe_id)
        if stripe is None:
            return (False, f"unknown stripe {run_id}/{stripe_id}")
        if stripe["claimed_by"] != msg_worker_id:
            return (False, f"stale: stripe claimed_by {stripe['claimed_by']!r} "
                           f"!= {msg_worker_id!r}")
        recorded = conn.assignment_attempts.get(stripe_id)
        if recorded is None or stripe["current_attempt"] != recorded:
            return (False, f"stale: ledger attempt {stripe['current_attempt']} "
                           f"!= connection attempt {recorded}")
        if stripe["state"] not in permitted_states:
            return (False, f"state {stripe['state']!r} does not permit this message")
        return (True, None)

    def validate_spool_path(self, conn: WorkerConnection, spool_path: str) -> bool:
        """Gate 17: a remote spool path must live under the worker's configured
        spool root (path-normalized, `..`-guarded)."""
        return spool_path_within_root(conn.node_config.spool_root, spool_path)

    # ----- assignment (Blocker 7) ------------------------------------------
    def assign_stripes(
        self,
        run_id: str,
        family_name: str,
        phase: int,
        total_seeds: int,
        workers: List[WorkerRecord],
        base_start: int = 0,
        attempt: int = 0,
        now: Optional[float] = None,
        stripe_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Partition total_seeds into contiguous macro-stripes and claim each to
        a worker (round-robin). expected_substripes is recorded from the assigned
        worker's ADVERTISED cap for the concrete variant (Blocker 7) — the macro
        size (config.miner_stripe_size) MAY exceed one GPU cap.

        stripe_prefix defaults to run_id; serve_trial passes a per-STAGE prefix
        (`{run_id}__st{n}`) so a multi-family workflow's stages never collide on
        stripe IDs (Defect 6)."""
        if not workers:
            raise ValueError("assign_stripes requires at least one worker")
        now = time.time() if now is None else now
        prefix = stripe_prefix or run_id
        macro = partition_macro_stripes(
            total_seeds, self.config.miner_stripe_size, base_start
        )
        # Defect 2 (C4): FILTER the scheduling pool to workers that advertise the
        # EXACT concrete variant (and are not quarantined) BEFORE round-robin —
        # round-robin only across COMPATIBLE workers, so a java_lcg stripe is never
        # blindly handed to a pcg32-only worker and stranded `pending` forever.
        compatible = [w for w in workers if self.can_assign_variant(w, family_name)]
        assignments: List[Dict[str, Any]] = []
        for i, (idx, seed_start, seed_count) in enumerate(macro):
            stripe_id = f"{prefix}_s{idx}"
            self.ledger.add_stripe(
                run_id, stripe_id, seed_start, seed_count, family_name, phase, now
            )
            # No compatible worker in the pool -> refuse (pending). serve_trial turns
            # this into an EXPLICIT trial failure (never an indefinite strand).
            if not compatible:
                assignments.append({
                    "stripe_id": stripe_id, "seed_start": seed_start,
                    "seed_count": seed_count, "worker_id": None,
                    "attempt": attempt, "expected_substripes": None,
                    "effective_cap": None, "claimed": False,
                    "refused_reason": f"no eligible worker (cannot serve variant "
                                      f"{family_name!r})",
                })
                continue
            worker = compatible[i % len(compatible)]
            cap = advertised_effective_cap(
                worker.backend, family_name, worker.seed_caps
            )
            expected = expected_substripes_for(seed_count, cap)
            claimed = self.ledger.claim_stripe(
                run_id, stripe_id, worker.worker_id, attempt, expected,
                now + self.config.compute_lease_timeout,
            )
            # L1: the connection records the attempt it was handed, paired against
            # the ledger's authoritative current_attempt on every later message.
            if claimed and hasattr(worker, "record_assignment"):
                worker.record_assignment(stripe_id, attempt)
            assignments.append({
                "stripe_id": stripe_id,
                "seed_start": seed_start,
                "seed_count": seed_count,
                "worker_id": worker.worker_id,
                "attempt": attempt,
                "expected_substripes": expected,
                "effective_cap": cap,
                "claimed": claimed,
            })
        return assignments

    # ----- completion (Blocker 1 + L8) -------------------------------------
    def evaluate_stripe(self, run_id: str, stripe_id: str) -> CompletionCheck:
        """Run the L8 predicate over the CURRENT attempt's shards."""
        stripe = self.ledger.get_stripe(run_id, stripe_id)
        if stripe is None:
            raise KeyError(f"unknown stripe {run_id}/{stripe_id}")
        shards = self.ledger.get_shards(
            run_id, stripe_id, stripe["current_attempt"]
        )
        return evaluate_stripe_completion(stripe, shards)

    def finalize_stripe(self, run_id: str, stripe_id: str, now: Optional[float] = None,
                        eligible_provider=None) -> CompletionCheck:
        """Transition staging -> done ONLY when the L8 predicate fully holds.
        Blocker 2: publish that attempt's manifests to Phase 5 (provisional trial
        input) as part of completion, BEFORE the done transition — so an
        incomplete/failed attempt publishes nothing.

        Defect 4 (C3): distinguish "incomplete, still waiting" from
        "complete-but-invalid." When StripeComplete has arrived and ALL expected
        shards are present (substripes_match) but the accounting does NOT reconcile
        (seed_count sum, survivor sum, or coverage is structurally wrong), that is a
        DEFINITIVE reconciliation failure — route it through the phase-specific
        matrix EXACTLY ONCE rather than leaving the stripe parked in `staging`
        until the global trial timeout. A stripe that is merely missing shards, or
        whose shards are still being staged/verified (reconciled but not all
        verified), is NOT definitive — it keeps waiting."""
        # Lifecycle-locked (Defect 4/5): a staging-executor thread and the dispatch
        # thread may both reach finalize for the same stripe; serialize so the
        # publish + done transition (or the single matrix routing) happen exactly once.
        with self._lifecycle_lock:
            stripe = self.ledger.get_stripe(run_id, stripe_id)
            if stripe is None:
                raise KeyError(f"unknown stripe {run_id}/{stripe_id}")
            check = self.evaluate_stripe(run_id, stripe_id)
            if stripe["state"] != ST_STAGING:
                return check
            if check.is_complete:
                self.publish_attempt(run_id, stripe_id, stripe["current_attempt"], now)
                self.ledger.set_stripe_state(run_id, stripe_id, ST_DONE)
            elif (eligible_provider is not None
                  and check.substripes_match and not check.reconciled):
                # Definitive structural failure: every expected shard is present but
                # the totals/coverage do not reconcile. Feed the retry matrix once.
                # Routing requires the eligible-worker set, so it fires ONLY when the
                # real lifecycle (serve dispatch / staging-job completion) drives
                # finalize — a bare predicate call (eligible_provider=None) still just
                # evaluates + does the done transition, leaving the stripe in staging.
                self._on_staging_failed(
                    run_id, stripe_id, retryable=True,
                    eligible_provider=eligible_provider,
                    reason="StripeComplete reconciliation failed (structural mismatch): "
                           + "; ".join(check.reasons))
            return check

    def publish_attempt(
        self, run_id: str, stripe_id: str, attempt: int, now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Blocker 2: hand this attempt's verified shard manifests to Phase 5 as
        provisional trial input (committed only by a later TrialCommit; discarded
        by TrialAbort). Called only when the whole attempt reconciles. Marks each
        shard enqueued; the reservation stays HELD until Phase 5 acks + local
        delete (L2)."""
        now = time.time() if now is None else now
        stripe = self.ledger.get_stripe(run_id, stripe_id)
        # D0 (Blocker 1, REV3): reconstruct the immutable trial_metadata projection
        # DURABLY from the ledger — the trial-global `trial_context` row (persisted
        # once per run_id, before any stripe work) + this stripe's own persisted
        # phase/family_name. It survives a restart and is identical across the run's
        # manifests where trial-global. The durable row MUST exist before any publish:
        # a completely-absent `trial_context` row now FAILS CLOSED here, instead of the
        # old `... if trial_ctx is not None else None` fallback that let _build_manifest
        # emit `trial_metadata: {}` and leak an empty manifest to Phase 5. (The interim
        # `_finalize_stage` manifest keeps its own no-metadata `{}` shape but is NEVER
        # published to Phase 5 — publish_shard is reached only from this method.)
        trial_ctx = self.ledger.get_trial_context(run_id)
        if trial_ctx is None:
            raise MinerMetadataError(
                f"missing durable trial context for run_id={run_id!r}; "
                "refusing Phase 5 publication"
            )
        trial_metadata = derive_trial_metadata(trial_ctx, stripe)
        manifests: List[Dict[str, Any]] = []
        for sh in self.ledger.get_shards(run_id, stripe_id, attempt):
            if sh["staging_status"] != SH_VERIFIED:
                continue
            res = self.ledger.get_reservation_by_event(
                event_id_for(run_id, stripe_id, attempt, sh["sub_index"],
                             stripe["staging_generation"]))
            event_id = res["event_id"] if res else event_id_for(
                run_id, stripe_id, attempt, sh["sub_index"],
                stripe["staging_generation"])
            manifest = self._build_manifest(
                event_id, run_id, stripe, attempt, sh["sub_index"],
                sh["local_staged_path"], sh["size_bytes"], sh["sha256"],
                trial_metadata=trial_metadata)
            if self.phase5_sink is not None:
                self.phase5_sink.publish_shard(manifest)
            self.ledger.mark_shard_enqueued(run_id, stripe_id, attempt, sh["sub_index"], now)
            self.enqueued.append(manifest)
            manifests.append(manifest)
        return manifests

    # ----- staging pipeline (Blocker 4, Decision B, L2, L5, L8) -------------
    def _staged_path(
        self, run_id: str, stripe_id: str, attempt: int, sub_index: int,
        staging_generation: int, sha256: str,
    ) -> str:
        """Defect 1: the staged (and derived temp) path is ATTEMPT/GENERATION-
        PRIVATE. Identity comes from the immutable task key
        (run_id, stripe_id, attempt, sub_index, staging_generation), NOT the sha —
        so a re-dispatched attempt covering the same seed range (same sha) can
        NEVER collide on one file with the superseded attempt. A stale callback can
        therefore only ever remove a path uniquely owned by its own task key."""
        staging_dir = self.config.staging_dir
        if not staging_dir:
            # [Part B §2] NON-RETRYABLE: a missing staging path is a permanent
            # CONFIGURATION defect, not a transient staging condition. Retrying it
            # burned a Q3 retry and surfaced downstream as MinerIngressError naming
            # thresholds — the wrong subsystem entirely.
            #
            # In production this is now unreachable: run_trial_miner resolves and
            # validates staging BEFORE build_coordinator. It remains as a
            # defence-in-depth backstop for callers that construct a coordinator
            # directly (harnesses via build_coordinator), which is exactly the
            # surface every previously-certified miner run used.
            raise StagingConfigurationError(
                "config.staging_dir is not set (coordinator staging is unconfigured; "
                "see Part B §1.1 — staging_dir is canonical and has no implicit "
                "/dev/shm fallback)")
        os.makedirs(staging_dir, exist_ok=True)
        safe_run = re.sub(r"[^A-Za-z0-9._-]", "_", str(run_id))
        name = (f"{safe_run}__{stripe_id}_a{attempt}_s{sub_index}"
                f"_g{staging_generation}_{sha256[:16]}.json")
        return os.path.join(staging_dir, name)

    def _cleanup_file(self, path: Optional[str]) -> None:
        if not path:
            return
        try:
            os.remove(path)
        except OSError:
            pass

    def _attempt_active(
        self, run_id: str, stripe_id: str, attempt: int, staging_generation: int
    ) -> bool:
        """L5 liveness: the attempt is still the current one AND the staging
        generation has not been fenced AND the trial is not aborted/cancelled."""
        st = self.ledger.get_stripe(run_id, stripe_id)
        if st is None or st["state"] == ST_CANCELLED:
            return False
        return (st["current_attempt"] == attempt
                and st["staging_generation"] == staging_generation)

    def reserved_bytes(self) -> int:
        """Bytes currently reserved-or-staged (held reservations, global)."""
        return self.ledger.held_bytes()

    def reserved_files(self) -> int:
        """Files currently reserved-or-staged (held reservations, global)."""
        return self.ledger.held_files()

    def reserve_capacity(
        self, run_id: str, stripe_id: str, attempt: int, sub_index: int,
        staging_generation: int, size_bytes: int, now: Optional[float] = None,
    ) -> Optional[int]:
        """Public reserve wrapper: returns a reservation_id or None (back-pressure)."""
        return self.ledger.reserve(
            run_id, stripe_id, attempt, sub_index, staging_generation, size_bytes,
            self.config.staging_high_water_bytes,
            self.config.staging_high_water_files, now,
        )

    def _fail_and_release(
        self, reservation_id: int, path_to_remove: Optional[str],
        now: Optional[float] = None,
    ) -> None:
        """L8 order: remove the temp/staged file FIRST, THEN release capacity, so
        the byte/file mark can never account a file that no longer exists and no
        reservation leaks. Idempotent (release only touches a still-held row)."""
        self._cleanup_file(path_to_remove)
        self.ledger.release_reservation(reservation_id, now)

    def _write_bytes_atomic(self, temp_path: str, staged_path: str, payload_bytes: bytes) -> None:
        """Mirror spool_payload_atomic: temp in the SAME dir -> fsync -> replace,
        via the injectable rename primitive (gate 36 atomic-write-failure)."""
        with open(temp_path, "wb") as f:
            f.write(payload_bytes)
            f.flush()
            os.fsync(f.fileno())
        self._atomic_replace(temp_path, staged_path)

    def _build_manifest(
        self, event_id: str, run_id: str, stripe: Dict[str, Any], attempt: int,
        sub_index: int, staged_path: str, size: int, sha256: str,
        trial_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """The ONE uniform ShardReadyManifest shape (Blocker 4 / L6) for inline AND
        remote shards. event_id is the immutable L6 ack key.

        D0: when `trial_metadata` is supplied (the publish path, once a trial_context
        row exists), it is a COMPLETE immutable projection — it is validated
        fail-closed (a missing/empty mandatory field raises MinerMetadataError, never
        a `{}` leak), and its provenance shas are also lifted to top-level manifest
        fields. The interim `_finalize_stage` manifest (not published to Phase 5)
        still calls with no metadata and keeps the legacy `{}` shape."""
        manifest = {
            "event_id": event_id,
            "run_id": run_id,
            "stripe_id": stripe["stripe_id"],
            "workflow_phase": stripe["phase"],
            "attempt": attempt,
            "sub_index": sub_index,
            "local_spool_path": staged_path,
            "expected_size": size,
            "expected_sha256": sha256,
            "trial_metadata": trial_metadata if trial_metadata is not None else {},
        }
        if trial_metadata is not None:
            validate_trial_metadata(trial_metadata)
            # Provenance (non-NPZ) retained as top-level manifest fields too.
            manifest["dataset_sha256"] = trial_metadata["dataset_sha256"]
            manifest["residue_sha256"] = trial_metadata["residue_sha256"]
        return manifest

    def _finalize_stage(
        self, task: StagingTask, actual_bytes: bytes, now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Shared verify → (fence) → atomic rename → publish core for inline and
        remote shards. Returns {"status": "verified"|"stale", ...}."""
        now = time.time() if now is None else now
        size = len(actual_bytes)
        sha = _sha256_bytes(actual_bytes)

        # Verify size + sha256 over the ACTUAL bytes (§15). Mismatch: remove temp
        # FIRST, release capacity; delete_remote is NEVER invoked (gate 15).
        if size != task.expected_size or sha != task.expected_sha256:
            self._fail_and_release(task.reservation_id, task.temp_path, now)
            self.ledger.set_shard_staging_status(
                task.run_id, task.stripe_id, task.attempt, task.sub_index, SH_FAILED)
            raise StagingHashMismatch(
                f"{task.stripe_id}/sub{task.sub_index}: bytes size={size} sha={sha} "
                f"!= advertised size={task.expected_size} sha={task.expected_sha256}")

        # Materialize the staged file atomically (temp -> fsync -> replace).
        try:
            if task.kind == "inline":
                self._write_bytes_atomic(task.temp_path, task.staged_path, actual_bytes)
            else:
                self._atomic_replace(task.temp_path, task.staged_path)
        except Exception:
            self._fail_and_release(task.reservation_id, task.temp_path, now)
            self.ledger.set_shard_staging_status(
                task.run_id, task.stripe_id, task.attempt, task.sub_index, SH_FAILED)
            raise

        # L5 fence: BEFORE any ledger update / enqueue / delete_remote, re-check
        # the attempt is still active. A stale completion deletes its OWN staged
        # file, releases ONLY its own reservation, and publishes nothing.
        # Defect 1 invariant: task.staged_path is attempt/generation-PRIVATE (see
        # _staged_path), so the rename above wrote onto THIS task's own file and
        # this delete removes ONLY that file — it can never clobber or delete a
        # live sibling attempt's staged file (Beta's `attempt1_file_after_stale_finish`).
        if not self._attempt_active(
            task.run_id, task.stripe_id, task.attempt, task.staging_generation
        ):
            self._fail_and_release(task.reservation_id, task.staged_path, now)
            return {"status": "stale", "reservation_id": task.reservation_id}

        # Verified + HELD in the bounded queue. The shard is NOT yet published to
        # Phase 5 — Blocker 2 publishes an attempt's manifests only when the whole
        # attempt is verified + StripeComplete + L8-reconciled (see publish_attempt),
        # so a later same-attempt failure leaves nothing published (gate 7). The
        # remote spool IS released now (Decision B: only after local verify).
        self.ledger.set_reservation_paths(task.reservation_id, staged_path=task.staged_path)
        self.ledger.mark_shard_verified(
            task.run_id, task.stripe_id, task.attempt, task.sub_index,
            local_staged_path=task.staged_path, size_bytes=size, sha256=sha, now=now)
        stripe = self.ledger.get_stripe(task.run_id, task.stripe_id)
        event_id = event_id_for(
            task.run_id, task.stripe_id, task.attempt, task.sub_index,
            task.staging_generation)
        manifest = self._build_manifest(
            event_id, task.run_id, stripe, task.attempt, task.sub_index,
            task.staged_path, size, sha)
        if task.kind == "remote":
            self._delete_remote(task, now)
        return {
            "status": "verified",
            "reservation_id": task.reservation_id,
            "event_id": event_id,
            "staged_path": task.staged_path,
            "manifest": manifest,
            "size": size,
            "sha256": sha,
        }

    def stage_inline_shard(
        self, run_id: str, stripe_id: str, attempt: int, sub_index: int,
        seed_start: int, seed_count: int, survivors: list,
        expected_size: int, expected_sha256: str, now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Blocker 4: normalize an inline result to the SAME canonical
        s172_substripe_v1 bytes the worker would spool, verify, atomically write
        Zeus-local, enqueue the SAME path-manifest as a remote spool."""
        now = time.time() if now is None else now
        stripe = self.ledger.get_stripe(run_id, stripe_id)
        if stripe is None:
            raise StagingError(f"unknown stripe {run_id}/{stripe_id}")
        gen = stripe["staging_generation"]
        _, payload_bytes = build_substripe_payload_bytes(
            stripe_id, sub_index, seed_start, seed_count, survivors)
        size = len(payload_bytes)
        sha = _sha256_bytes(payload_bytes)
        # advertised inline metadata must agree with the canonical bytes
        if size != expected_size or sha != expected_sha256:
            self.ledger.set_shard_staging_status(run_id, stripe_id, attempt, sub_index, SH_FAILED)
            raise StagingHashMismatch(
                f"inline {stripe_id}/sub{sub_index}: canonical size={size} sha={sha} "
                f"!= advertised size={expected_size} sha={expected_sha256}")
        rid = self.reserve_capacity(run_id, stripe_id, attempt, sub_index, gen, size, now)
        if rid is None:
            raise StagingBackPressure(f"cannot reserve {size} bytes for inline shard")
        staged = self._staged_path(run_id, stripe_id, attempt, sub_index, gen, sha)
        temp = f"{staged}.tmp.{os.getpid()}"
        self.ledger.set_reservation_paths(rid, temp_path=temp)
        task = StagingTask(
            reservation_id=rid, run_id=run_id, stripe_id=stripe_id, attempt=attempt,
            sub_index=sub_index, staging_generation=gen, kind="inline",
            temp_path=temp, staged_path=staged, expected_size=size,
            expected_sha256=sha, payload_bytes=payload_bytes)
        return self._finalize_stage(task, payload_bytes, now)

    def _adapter_supports_timeout(self) -> bool:
        """True iff the TransferAdapter's fetch_remote accepts a `timeout` — i.e. the
        transfer can cancel itself natively (ssh ConnectTimeout etc.) so no orphan
        thread is ever left behind."""
        try:
            return "timeout" in inspect.signature(self.transfer.fetch_remote).parameters
        except (TypeError, ValueError):
            return False

    def _call_fetch(self, node_config, remote_path, temp_path, timeout) -> None:
        """Defect 4 (C4): PREFER native TransferAdapter cancellation — pass `timeout`
        when the adapter accepts it so the transfer itself aborts and its thread
        returns. Adapters that don't accept a timeout fall back to the reserve-before-
        launch orphan registry (Defect 2 C5)."""
        if self._adapter_supports_timeout():
            self.transfer.fetch_remote(node_config, remote_path, temp_path, timeout=timeout)
        else:
            self.transfer.fetch_remote(node_config, remote_path, temp_path)

    def _reserve_orphan_slot(self, key: str, th: threading.Thread) -> bool:
        """Defect 2 (C5): RESERVE an orphan-thread slot BEFORE a non-cancellable
        fetch thread is started. Prunes finished threads, then admits only if the
        LIVE count is below cap. Returns False if the budget is exhausted — the
        caller must then NOT start the thread and fail the job with a capacity error,
        so a hung transport never spawns yet another zombie thread. (Old C4 bug: the
        thread was started BEFORE the cap check, so refused fetches still leaked live
        threads.)"""
        cap = max(1, int(getattr(self.config, "staging_orphan_fetch_max", 8)))
        with self._orphan_lock:
            self._orphan_fetch_threads = [
                (k, t) for (k, t) in self._orphan_fetch_threads if t.is_alive()]
            if len(self._orphan_fetch_threads) >= cap:
                return False
            self._orphan_fetch_threads.append((key, th))
            return True

    def _release_orphan_slot(self, th: threading.Thread) -> None:
        """Release a reserved slot for a fetch thread that COMPLETED in time (not an
        orphan); also drop any finished threads."""
        with self._orphan_lock:
            self._orphan_fetch_threads = [
                (k, t) for (k, t) in self._orphan_fetch_threads
                if t is not th and t.is_alive()]

    def account_orphan_fetches(self, join_timeout: float = 0.0) -> int:
        """Defect 4 (C4): shutdown accounting for tracked orphan fetch threads —
        prune finished ones, briefly join, and return the residual live count. Called
        from serve_trial's finally; safe to call directly from a harness."""
        with self._orphan_lock:
            threads = list(self._orphan_fetch_threads)
        for _k, t in threads:
            if t.is_alive() and join_timeout > 0:
                t.join(join_timeout)
        with self._orphan_lock:
            self._orphan_fetch_threads = [
                (k, t) for (k, t) in self._orphan_fetch_threads if t.is_alive()]
            live = len(self._orphan_fetch_threads)
        return live

    def _fetch_with_timeout(self, node_config, remote_path, temp_path, timeout):
        """Bound a (possibly blocking) fetch_remote by staging_timeout, with TRACKED
        cleanup of a late/abandoned write (Defect 1 C3) AND a TRACKED registry of
        never-returning fetch threads (Defect 4 C4).

        The fetch runs on a daemon thread joined with `timeout`. On expiry we mark
        the fetch ABANDONED and raise StagingTimeout — but we do NOT merely rely on
        the daemon disappearing: (1) the fetch thread's own finally removes any temp
        artifact it wrote (temp_path is attempt/generation-PRIVATE, so it can only
        remove THIS task's own file); (2) the abandoned thread is recorded in the
        bounded orphan registry so permanent hangs cannot accumulate threads across a
        soak. A tiny lock serializes the abandon-vs-complete decision."""
        box: Dict[str, Any] = {}
        lock = threading.Lock()
        # phase: 'running' -> 'done' (fetch finished in time) | 'abandoned' (timed out)
        state = {"phase": "running"}

        def _run():
            try:
                self._call_fetch(node_config, remote_path, temp_path, timeout)
                box["ok"] = True
            except Exception as e:  # noqa: BLE001
                box["err"] = e
            finally:
                with lock:
                    if state["phase"] == "abandoned":
                        # The caller already timed out, cleaned up the reservation
                        # and (a possibly-absent) temp file, and moved on. Any bytes
                        # this late fetch just wrote are an orphan — remove them now.
                        self._cleanup_file(temp_path)
                    else:
                        state["phase"] = "done"

        th = threading.Thread(target=_run, name="miner-fetch", daemon=True)
        # Defect 2 (C5): if the adapter cannot cancel, RESERVE an orphan slot BEFORE
        # starting the thread. If the live orphan budget is exhausted, do NOT start
        # the thread — fail the job with a capacity error so a hung transport
        # surfaces as capacity pressure instead of spawning another zombie thread.
        native = self._adapter_supports_timeout()
        reserved = False
        if not native:
            if not self._reserve_orphan_slot(remote_path, th):
                raise StagingError(
                    "orphan fetch-thread budget exhausted (too many hung transfers) "
                    "— refusing to launch another: transport capacity error "
                    "(staging_orphan_fetch_max)")
            reserved = True
        th.start()
        th.join(timeout)
        with lock:
            timed_out = state["phase"] != "done"
            if timed_out:
                state["phase"] = "abandoned"
        if reserved and not timed_out:
            # completed in time -> NOT an orphan; release its reserved slot.
            self._release_orphan_slot(th)
        if timed_out:
            # The reserved slot is KEPT (this thread is now an orphan, still alive);
            # the reserve-before-launch guard bounds the LIVE thread count at cap.
            raise StagingTimeout(
                f"fetch_remote exceeded staging_timeout={timeout}s for {remote_path}")
        if "err" in box:
            raise box["err"]

    def begin_remote_stage(
        self, conn: WorkerConnection, run_id: str, stripe_id: str, attempt: int,
        sub_index: int, remote_spool_path: str, expected_size: int,
        expected_sha256: str, now: Optional[float] = None,
        fetch_timeout: Optional[float] = None,
    ) -> StagingTask:
        """Reserve capacity BEFORE the transfer, then fetch the remote spool to a
        Zeus-local temp path (Decision B). Returns the StagingTask; the caller
        drives finish_remote_stage() (split so the fence race is exercisable).
        `fetch_timeout` (Defect 4) bounds a blocking fetch → StagingTimeout."""
        now = time.time() if now is None else now
        if self.transfer is None:
            raise StagingError("no transfer adapter configured")
        if not self.validate_spool_path(conn, remote_spool_path):
            raise StagingError(
                f"spool path {remote_spool_path!r} escapes worker spool root")
        stripe = self.ledger.get_stripe(run_id, stripe_id)
        if stripe is None:
            raise StagingError(f"unknown stripe {run_id}/{stripe_id}")
        gen = stripe["staging_generation"]
        rid = self.reserve_capacity(
            run_id, stripe_id, attempt, sub_index, gen, expected_size, now)
        if rid is None:
            raise StagingBackPressure(
                f"cannot reserve {expected_size} bytes for {stripe_id}/sub{sub_index}")
        staged = self._staged_path(
            run_id, stripe_id, attempt, sub_index, gen, expected_sha256)
        temp = f"{staged}.tmp.{os.getpid()}"
        self.ledger.set_reservation_paths(rid, temp_path=temp)
        task = StagingTask(
            reservation_id=rid, run_id=run_id, stripe_id=stripe_id, attempt=attempt,
            sub_index=sub_index, staging_generation=gen, kind="remote",
            temp_path=temp, staged_path=staged, expected_size=expected_size,
            expected_sha256=expected_sha256, remote_spool_path=remote_spool_path,
            node_config=conn.node_config)
        try:
            if fetch_timeout is None:
                self.transfer.fetch_remote(conn.node_config, remote_spool_path, temp)
            else:
                self._fetch_with_timeout(
                    conn.node_config, remote_spool_path, temp, fetch_timeout)
        except Exception:
            # fetch failed / timed out: remove any temp FIRST, release capacity
            # (gate 36 / L8 order), then the caller routes through the matrix.
            self._fail_and_release(rid, temp, now)
            self.ledger.set_shard_staging_status(run_id, stripe_id, attempt, sub_index, SH_FAILED)
            raise
        return task

    def finish_remote_stage(
        self, task: StagingTask, now: Optional[float] = None
    ) -> Dict[str, Any]:
        """Read the fetched temp bytes and run the shared verify/fence/publish
        core. A fence that fired between begin and finish yields status 'stale'."""
        with open(task.temp_path, "rb") as f:
            actual = f.read()
        return self._finalize_stage(task, actual, now)

    def stage_remote_shard(
        self, conn: WorkerConnection, run_id: str, stripe_id: str, attempt: int,
        sub_index: int, remote_spool_path: str, expected_size: int,
        expected_sha256: str, now: Optional[float] = None,
        fetch_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convenience: begin + finish with no interleaving (happy path)."""
        task = self.begin_remote_stage(
            conn, run_id, stripe_id, attempt, sub_index, remote_spool_path,
            expected_size, expected_sha256, now, fetch_timeout=fetch_timeout)
        return self.finish_remote_stage(task, now)

    # ----- stripe-level admission (Defect 3 C3) -----------------------------
    def _attempt_live_locked(self, run_id, stripe_id, attempt) -> bool:
        """True iff (run,stripe,attempt) is still the CURRENT, non-terminal attempt
        (so its staging should keep waiting for capacity rather than being routed to
        the matrix or dropped). Read-only; caller may hold _admission_lock."""
        stripe = self.ledger.get_stripe(run_id, stripe_id)
        if stripe is None or stripe["state"] in (ST_DONE, ST_FAILED, ST_CANCELLED):
            return False
        if stripe["current_attempt"] != attempt:
            return False
        trial = self.ledger.get_trial(run_id)
        if trial is not None and trial["state"] in ("committed", "aborted"):
            return False
        return True

    def _prune_admitted_locked(self) -> None:
        """De-commit admission budgets whose attempt is terminal or superseded.
        A DONE attempt is de-committed even though its reservations are still held
        (they are published and WILL be acked → capacity frees), so a newly-admitted
        attempt may proceed and simply wait (nonblocking) for that capacity."""
        dead = []
        for key in self._admitted:
            run_id, stripe_id, attempt = key
            stripe = self.ledger.get_stripe(run_id, stripe_id)
            if (stripe is None
                    or stripe["state"] in (ST_DONE, ST_FAILED, ST_CANCELLED)
                    or stripe["current_attempt"] != attempt):
                dead.append(key)
        for key in dead:
            self._admitted.pop(key, None)

    def _attempt_need_files(self, stripe) -> int:
        """The attempt's FILE footprint: one staging file per expected sub-stripe.
        (Correction 6: there is NO byte estimate here — the BYTE decision is driven
        entirely by ACTUAL advertised sizes, never a static INLINE_BYTE_LIMIT.)"""
        expected = stripe.get("expected_substripes") or 1
        return max(1, int(expected))

    def _attempt_exceeds_highwater(self, stripe) -> Optional[str]:
        """A reason string if the attempt's FILE footprint cannot fit the configured
        file high-water even on an otherwise-empty coordinator — a capacity/config
        error to surface IMMEDIATELY (fail fast). None = it can fit.

        Correction 6: this is a FILES-only sanity guard. The BYTE decision lives
        entirely in enqueue_staging, driven by ACTUAL advertised sizes
        (_attempt_actual_bytes / the per-shard check) — never a static per-file
        estimate, which is simultaneously too large for tiny inline results and too
        small for large remote spools."""
        need_files = self._attempt_need_files(stripe)
        hw_files = self.config.staging_high_water_files
        if need_files > hw_files:
            return (f"attempt needs {need_files} staging files but high-water is "
                    f"{hw_files} (staging_high_water_files)")
        return None

    def _attempt_actual_bytes(self, run_id, stripe_id, attempt) -> int:
        """Defect 1 (C5): the SUM of the attempt's shards' ADVERTISED size_bytes so
        far (inline payload size or remote spool size — the real bytes each shard
        will occupy while staged). The whole attempt's shards are held simultaneously
        until the attempt publishes + is acked, so this sum is the attempt's true
        byte footprint against the byte high-water."""
        return sum(int(sh.get("size_bytes") or 0)
                   for sh in self.ledger.get_shards(run_id, stripe_id, attempt))

    def _try_admit_locked(self, run_id, stripe_id, attempt) -> bool:
        """Correction 6 — ONE coherent admission byte-model via SERIALIZED
        attempt-level staging (Beta's recommended approach A).

        Admit an attempt ONLY if it is already admitted, OR no OTHER attempt is
        currently holding partial staging capacity (`self._admitted` empty). At most
        ONE attempt actively stages at a time; a second attempt DEFERS (bounded) and
        resumes when the first completes + publishes and its capacity is released via
        the ack path. Because only one attempt stages at a time, its shards — each
        reserved by its ACTUAL advertised size_bytes (the C5 per-shard guard) — can
        never be starved by another attempt's partial occupancy, so two attempts can
        never circular-wait. `_admitted` is now a PURE SERIALIZATION GATE (a live
        attempt key → True), never a static-byte-estimate budget. The BYTE decision
        (single shard > high-water, or an admitted attempt's ACCUMULATED actual bytes
        > high-water → fail fast) lives entirely in enqueue_staging on real sizes.
        Caller MUST hold _admission_lock."""
        key = (run_id, stripe_id, attempt)
        if key in self._admitted:
            return True
        if self._admitted:
            # a DIFFERENT attempt is actively staging — serialize: defer this one.
            return False
        if self.ledger.get_stripe(run_id, stripe_id) is None:
            return False
        self._admitted[key] = True
        return True

    def _entry_bytes(self, entry) -> int:
        """Retained coordinator RAM for a deferred entry: an inline result holds its
        full payload; a remote result holds only small metadata (the payload stays
        on the remote spool / the wire, not in coordinator RAM)."""
        kind, msg = entry[0], entry[6]
        if kind == "inline":
            return int(getattr(msg, "size_bytes", 0) or 0)
        return 0

    def _deferred_retained_bytes(self) -> int:
        return sum(self._entry_bytes(e) for e in self._deferred)

    # ==================================================================
    # [S172-BP] Coordinator staging back-pressure: derived bound (§2),
    # per-connection pause/resume (§1), bounded capacity timeout (§1.5),
    # metrics (§4).
    #
    # THE CLASSIFICATION LAW (§0, Beta D — binding):
    #   Coordinator staging back-pressure is a WAITING STATE, not a stripe
    #   failure state. No capacity-wait event may enter the phase-specific
    #   worker retry matrix. The ONE permitted trial-terminal path for
    #   capacity is the bounded timeout of §1.5, which calls fail_trial
    #   DIRECTLY — never handle_stripe_failure.
    # ==================================================================

    def _resume_margin(self) -> int:
        """The documented bounded margin covering the transition window and
        ALREADY-DECODED messages (Beta: *"a documented bounded margin for
        transition and already-decoded messages"*).

        DEFAULT = the live connection count, because that is exactly how many
        envelopes can be in flight past the capacity check at once: each reader
        holds at most one decoded envelope (§1.2), so N connections can contribute
        at most N entries between the check and the enqueue. It doubles as the
        HYSTERESIS gap — pause at the bound, resume at `bound - margin` — which is
        what stops pause/resume thrash from a queue hovering at its limit.
        """
        return max(1, len(self.connections))

    def _derive_bound_from_current_state(self) -> int:
        """Best-effort derivation for a caller that never went through stage
        setup (a bare-API or gate call). Still DERIVED from live config and the
        currently-registered workers — never a constant. One slot of the
        configured macro-stripe size, phase 1.

        ⚠ [S172-BP AMENDMENT F5] THIS IS NOT A PRODUCTION FALLBACK. It survives
        ONLY for bare-API / gate contexts where no stage derivation was ever
        attempted. It answers a DIFFERENT question — one macro-stripe, phase 1 —
        and can therefore be MATERIALLY SMALLER than the stage derivation it used
        to catch (multi-stripe stages, hybrid caps), which is precisely how a
        sizing failure silently re-armed the undersized-queue condition. EVERY
        production stage now installs its bound at setup or FAILS CLOSED via
        `fail_trial(coordinator_staging_sizing: ...)` (serve_trial's stage-setup
        block); no production path reaches this method after a failed derivation.

        Memoized on (connection count, macro-stripe size) because a PAUSED reader
        re-reads the bound every 50 ms; the inputs only change when the fleet or
        the geometry does, and both are in the key."""
        key = (len(self.connections), int(self.config.miner_stripe_size))
        cached = self._fallback_bound_cache.get(key)
        if cached is not None:
            return cached
        workers = [w for w in self.connections.values()
                   if not getattr(w, "quarantined", False)] or \
                  list(self.connections.values())
        if not workers:
            # Nothing registered: the tightest CENTRALLY-CONFIGURED cap is still a
            # real derivation of "how many sub-stripes one stripe can produce".
            caps = self._central_caps()
            tightest = min(v for v in caps.values() if v > 0)
            burst = expected_substripes_for(
                int(self.config.miner_stripe_size), int(tightest))
        else:
            burst = staging_burst_bound_conservative(
                [int(self.config.miner_stripe_size)], workers, 1,
                caps=self._central_caps())
        bound = burst + self._resume_margin()
        self._fallback_bound_cache[key] = bound
        return bound

    def derive_staging_deferred_bound(
        self, stripe_spans, eligible_workers, phase: int,
        family_name: Optional[str] = None,
    ) -> int:
        """[§2] Compute and INSTALL the stage's derived deferred bound from the
        RESOLVED execution set, stripe geometry, phase and per-worker caps:

            staging_deferred_bound = burst_bound_conservative + resume_margin

        Called at stage setup (immediately after assign_stripes), so capacity is
        sized against what the stage will actually produce rather than against a
        number somebody typed."""
        margin = self._resume_margin()
        burst = staging_burst_bound_conservative(
            stripe_spans, eligible_workers, phase, caps=self._central_caps(),
            family_name=family_name)
        bound = burst + margin
        self._derived_deferred_bound = bound
        self._derived_bound_detail = {
            "burst_bound_conservative": burst,
            "resume_margin": margin,
            "bound": bound,
            "slots": len(list(stripe_spans)),
            "phase": int(phase),
            "family_name": family_name,
            "eligible_workers": len(list(eligible_workers)),
        }
        logger.info(
            "[S172-BP] derived_bound stage phase=%s family=%s slots=%d "
            "eligible_workers=%d burst_conservative=%d resume_margin=%d bound=%d",
            phase, family_name, self._derived_bound_detail["slots"],
            self._derived_bound_detail["eligible_workers"], burst, margin, bound)
        return bound

    def staging_deferred_bound(self) -> int:
        """The bound in force RIGHT NOW.

        `config.staging_deferred_max` is an OPTIONAL OPERATOR OVERRIDE of the
        derived value (None => derived). An override BELOW the derived bound logs a
        WARNING naming both numbers — it re-arms the very condition the derivation
        removes, so it must never be silent."""
        derived = self._derived_deferred_bound
        if derived is None:
            derived = self._derive_bound_from_current_state()
        override = getattr(self.config, "staging_deferred_max", None)
        if override is None:
            return max(1, int(derived))
        override = max(1, int(override))
        if override < derived:
            logger.warning(
                "[S172-BP] operator override staging_deferred_max=%d is BELOW the "
                "derived bound %d (burst_conservative + resume_margin) — the "
                "deferred queue can now saturate before back-pressure has absorbed "
                "the burst; detail=%s",
                override, derived, self._derived_bound_detail)
        return override

    # ----- §1.2 the capacity gate the reader consults ----------------------
    def staging_can_accept(self) -> bool:
        """True iff a staging slot is free OR the deferred queue is below
        `bound - resume_margin` (the hysteresis low-water).

        Read WITHOUT `_admission_lock` on purpose: Beta permits an approximate
        lock-free read here precisely because §2's bound carries an explicit
        per-connection margin covering the decode race this read can lose. The
        semaphore probe is acquire-then-immediately-release, so it never holds a
        slot away from a real staging submission for more than the probe itself."""
        sem = self._staging_slots()
        if sem.acquire(blocking=False):
            sem.release()
            return True
        bound = self.staging_deferred_bound()
        return len(self._deferred) < max(0, bound - self._resume_margin())

    # ----- §1.2 pause registry (per connection, FIFO) ----------------------
    def register_paused_connection(
        self, conn_key: Any, worker_id: Optional[str], now: Optional[float] = None,
    ) -> threading.Event:
        """Register `conn_key` as COORDINATOR-INITIATED PAUSED and hand back the
        event its reader waits on. Idempotent: a re-register returns the existing
        event and keeps the ORIGINAL pause-entry time, so §1.5's oldest-pause clock
        cannot be reset by a retry."""
        now = time.time() if now is None else now
        with self._pause_lock:
            rec = self._paused_connections.get(conn_key)
            if rec is None:
                rec = {"event": threading.Event(), "since": now,
                       "worker_id": worker_id}
                self._paused_connections[conn_key] = rec
                paused_now = len(self._paused_connections)
                ids = [r["worker_id"] for r in self._paused_connections.values()]
            else:
                rec["event"].clear()
                paused_now = len(self._paused_connections)
                ids = [r["worker_id"] for r in self._paused_connections.values()]
        with self._bp_lock:
            self._bp["pause_events"] += 1
            self._bp["paused_now"] = paused_now
            self._bp["paused_high_water"] = max(
                self._bp["paused_high_water"], paused_now)
        logger.info(
            "[S172-BP] pause worker=%s deferred=%d bound=%d paused_now=%d "
            "paused_ids=%s", worker_id, len(self._deferred),
            self.staging_deferred_bound(), paused_now, ids)
        return rec["event"]

    def deregister_paused_connection(
        self, conn_key: Any, now: Optional[float] = None, reason: str = "resume",
    ) -> Optional[float]:
        """Leave the paused set; returns this pause's duration in seconds.

        [S172-BP AMENDMENT F2] On a `reason == "resume"` exit the worker is given a
        RESUME GRACE of `compute_lease_timeout` seconds. The exemption of §1.4 only
        covered LIVE membership in `_paused_connections`, and a resuming reader
        deregisters BEFORE it delivers the held envelope and long before it reaches
        the heartbeat queued behind that envelope — so between deregistration and
        the first processed heartbeat the stripe's compute lease (300 s) is
        expirable although the coordinator itself caused every second of the
        silence (a pause may legitimately run to `staging_capacity_timeout`, 600 s).
        The grace is a COORDINATOR DICT written under `_pause_lock`; NO LEDGER
        MUTATION happens on the reader thread, so the reader rule stands. It is
        cleared the moment a real `renew_lease` succeeds (the bridge is no longer
        needed), on connection drop, and at trial-terminal cleanup; if it simply
        expires with no heartbeat, normal expiry handling resumes because the skip
        stops matching — which is what keeps it bounded and narrow."""
        now = time.time() if now is None else now
        grace_until = None
        with self._pause_lock:
            rec = self._paused_connections.pop(conn_key, None)
            paused_now = len(self._paused_connections)
            if (rec is not None and reason == "resume"
                    and rec.get("worker_id") is not None):
                grace_until = now + float(self.config.compute_lease_timeout)
                self._capacity_resume_grace[rec["worker_id"]] = grace_until
        if rec is None:
            return None
        if grace_until is not None:
            logger.info(
                "[S172-BP] resume_grace worker=%s until=%.3f "
                "(compute_lease_timeout=%.1fs) — the heartbeat that renews this "
                "worker's lease is still queued behind the envelope it just "
                "delivered", rec.get("worker_id"), grace_until,
                float(self.config.compute_lease_timeout))
        held = max(0.0, now - rec["since"])
        with self._bp_lock:
            self._bp["paused_now"] = paused_now
            self._bp["pause_seconds_total"] += held
            self._bp["pause_seconds_max"] = max(
                self._bp["pause_seconds_max"], held)
        logger.info(
            "[S172-BP] %s worker=%s pause_seconds=%.3f deferred=%d bound=%d "
            "paused_now=%d", reason, rec.get("worker_id"), held,
            len(self._deferred), self.staging_deferred_bound(), paused_now)
        return held

    def paused_worker_ids(self) -> frozenset:
        """Worker identities whose connection is in COORDINATOR-INITIATED pause.
        Consumed by the §1.4 lease exemption."""
        with self._pause_lock:
            return frozenset(
                r["worker_id"] for r in self._paused_connections.values()
                if r["worker_id"] is not None)

    def paused_connection_count(self) -> int:
        with self._pause_lock:
            return len(self._paused_connections)

    # ----- F1 the ingress resume credit ------------------------------------
    def _grant_resume_credit(self) -> Optional[Any]:
        """[S172-BP AMENDMENT F1] Grant AT MOST ONE wake per invocation, and make
        that wake CONSUME the capacity observation it was granted on.

        The pre-amendment `_resume_paused_connections` set one event, re-checked
        `staging_can_accept()` and looped — but a wake consumed nothing, so ONE
        freed slot satisfied the check on every iteration and released the whole
        paused fleet. Here the grant reserves the observation: while a credit is
        outstanding no further grant is issued, by this method or by a reader's own
        defensive poll (`_try_self_resume`).

        Returns the conn_key that was credited, or None. `_resume_paused_connections`
        is exactly one call to this — the while-loop is DELETED. Liveness is
        PER-EVENT, not per-loop: every capacity-release event (`_on_done` ->
        `_pump_deferred` -> `finally`) grants at most one, and the FIFO head's 50 ms
        self-grant covers the case where no further release event arrives.

        The documented resume margin (§1.2) is UNCHANGED and remains the final
        backstop for the decode race — nothing here is resized.
        """
        with self._pause_lock:
            if self._resume_credits_outstanding != 0:
                return None
            target_key = None
            target = None
            for key, rec in self._paused_connections.items():
                if not rec["event"].is_set():
                    target_key, target = key, rec
                    break
            if target is None:
                return None
            # Checked INSIDE `_pause_lock` so the observation and the credit are
            # taken together. Safe: `staging_can_accept()` never takes this lock,
            # and its semaphore probe is acquire-then-immediately-release.
            if not self.staging_can_accept():
                return None
            self._resume_credits_outstanding += 1
            self._resume_credit_holder = target_key
            worker_id = target["worker_id"]
            self._resume_credit_worker = worker_id
            self._resume_credit_since = time.time()
            # [S172-BP AMENDMENT F1-R2a] MINT THE TOKEN BEFORE THE WAKE. The woken
            # reader reads it back (`resume_credit_id_for`) and attaches it to the
            # envelope it delivers, so minting has to be complete — and recorded in
            # the pause record the reader owns — before `event.set()` releases it.
            self._resume_credit_seq += 1
            cid = self._resume_credit_seq
            self._resume_credit_id = cid
            target["credit_id"] = cid
            target["event"].set()
        logger.info(
            "[S172-BP] resume_signal worker=%s deferred=%d bound=%d "
            "credits_outstanding=1 credit_id=%d", worker_id, len(self._deferred),
            self.staging_deferred_bound(), cid)
        return target_key

    def _try_self_resume(self, conn_key: Any) -> bool:
        """[S172-BP AMENDMENT F1] The paused reader's 50 ms defensive poll, as a
        HEAD-ONLY SELF-GRANT.

        The bare `if self.staging_can_accept()` escape it replaces was a second
        thundering-herd door: EVERY paused reader could self-release on the same
        observation, with no grant involved at all. This preserves the lost-wakeup
        protection the poll exists for — the FIFO-oldest paused connection can
        always escape when capacity truly exists and no grant is in flight — while
        making it impossible for a non-head reader to ride someone else's
        observation. On success this connection TAKES the credit itself, so it is
        indistinguishable downstream from a granted wake.

        The event-wait remains the primary path; the poll cadence is unchanged.
        """
        with self._pause_lock:
            if self._resume_credits_outstanding != 0:
                return False
            head = next(iter(self._paused_connections), None)
            if head is None or head is not conn_key:
                return False
            if not self.staging_can_accept():
                return False
            self._resume_credits_outstanding += 1
            self._resume_credit_holder = conn_key
            rec = self._paused_connections[conn_key]
            worker_id = rec["worker_id"]
            self._resume_credit_worker = worker_id
            self._resume_credit_since = time.time()
            # [S172-BP AMENDMENT F1-R2a] Same mint, same order as the granted path
            # — a self-granted credit is indistinguishable downstream, and that
            # includes carrying a token.
            self._resume_credit_seq += 1
            cid = self._resume_credit_seq
            self._resume_credit_id = cid
            rec["credit_id"] = cid
            rec["event"].set()
        logger.info(
            "[S172-BP] self_resume worker=%s deferred=%d bound=%d "
            "credits_outstanding=1 credit_id=%d — FIFO head took the credit on "
            "its own capacity observation", worker_id, len(self._deferred),
            self.staging_deferred_bound(), cid)
        return True

    def _release_resume_credit(self, conn_key: Any, delivered: bool,
                               disposition: str = "dispatch") -> bool:
        """[S172-BP AMENDMENT F1/F1-R] Clear the outstanding credit held by
        `conn_key`.

        THE RESERVATION ENDS AT DISPOSITION (Beta F1-R §4 i-iv), NEVER AT INGRESS
        — `inbound.put` MOVES the envelope, it does not CONSUME the slot. The four
        dispositions, and the one caller of each:

          (i)   `enqueue_staging` acquired admission          }  the serve loop's
          (ii)  the envelope was retained in `_deferred`       }  post-dispatch
          (iii) the identity/attempt/dedup/terminal fence      }  `finally`
                rejected it                                    }
          (iv)  the connection or the trial terminated and the envelope was
                discarded — the serve loop's `eof` handling, the already-dropped
                socket skip, and the trial-terminal cleanup.

        The reader's own exit clear survives for ONE case only: a wake that
        delivered NOTHING (see `_conn_reader_loop`'s `finally`). A wake that DID
        deliver hands the clear to the serve loop; clearing it at reader exit is
        precisely the round-1 defect, one thread further along.

        Idempotent, and only the holder can clear it. `disposition` is LOG-ONLY —
        it names which of the four ended the reservation and is read by nothing.

        [S172-BP AMENDMENT F1-R2a] THIS IS NOW THE FORCE-CLEAR PATH ONLY, keyed on
        HOLDER STATE. It is correct for the dispositions where no future
        disposition can exist — eof before the envelope was disposed of, the
        reaped-socket discard (holder identity still decides, per Beta 6.1's
        rider), reader-exit-undelivered — because there the credited envelope will
        never be dispatched by anyone, so the slot it reserves is genuinely free
        again. The DISPATCH disposition does NOT come through here: an ordinary
        dispatch of an UNCREDITED envelope from the holder's socket must not clear
        anything (F1-R2a), which is what `_release_resume_credit_exact` enforces."""
        with self._pause_lock:
            if self._resume_credit_holder is not conn_key:
                return False
            cleared_id = self._resume_credit_id
            self._resume_credit_holder = None
            self._resume_credit_worker = None
            self._resume_credit_since = None
            self._resume_credit_id = None
            self._resume_credits_outstanding = max(
                0, self._resume_credits_outstanding - 1)
        logger.info("[S172-BP] resume_credit_cleared delivered=%s disposition=%s "
                    "credit_id=%s credits_outstanding=%d", delivered, disposition,
                    cleared_id, self._resume_credits_outstanding)
        return True

    def _release_resume_credit_exact(self, conn_key: Any, credit_id: Optional[int],
                                     delivered: bool,
                                     disposition: str = "dispatch") -> bool:
        """[S172-BP AMENDMENT F1-R2a] Clear at disposition ONLY for the EXACT
        envelope the credit was granted for.

        Both halves of the identity are required and neither is sufficient:

          * `credit_id is not None` — an envelope that carries NO token was never
            credited. An older, uncredited result of the holder's, queued in
            `inbound` before the pause even began, dispatches first and is fence-
            rejected without consuming any capacity; under round 2's socket-only
            test its `finally` cleared the credit while the credited envelope was
            still queued and the slot still free, and the next FIFO head woke on
            that same slot. It NEVER clears now.
          * `credit_id == self._resume_credit_id` — tokens are minted from a
            monotonic sequence, so a stale token from an earlier grant to the same
            socket cannot release a later one.
          * `conn_key is self._resume_credit_holder` — kept, so a token can only
            ever be redeemed against the connection it was granted to.

        Returns True only when the reservation was actually ended by THIS call."""
        with self._pause_lock:
            if credit_id is None:
                return False
            if credit_id != self._resume_credit_id:
                return False
            if self._resume_credit_holder is not conn_key:
                return False
            self._resume_credit_holder = None
            self._resume_credit_worker = None
            self._resume_credit_since = None
            self._resume_credit_id = None
            self._resume_credits_outstanding = max(
                0, self._resume_credits_outstanding - 1)
        logger.info("[S172-BP] resume_credit_cleared delivered=%s disposition=%s "
                    "credit_id=%s credits_outstanding=%d", delivered, disposition,
                    credit_id, self._resume_credits_outstanding)
        return True

    def clear_any_resume_credit(self, disposition: str = "trial_terminal") -> bool:
        """[S172-BP AMENDMENT F1-R] Disposition (iv), trial-terminal arm: drop ANY
        outstanding reservation regardless of who holds it.

        Unconditional by design — at trial-terminal there is no envelope left to
        dispose of and no reader that could ever clear it, so an outstanding credit
        would only be a leak into the next trial's accounting."""
        with self._pause_lock:
            if self._resume_credit_holder is None and \
                    self._resume_credits_outstanding == 0:
                return False
            worker = self._resume_credit_worker
            cleared_id = self._resume_credit_id
            self._resume_credit_holder = None
            self._resume_credit_worker = None
            self._resume_credit_since = None
            self._resume_credit_id = None
            self._resume_credits_outstanding = 0
        logger.info("[S172-BP] resume_credit_cleared delivered=unknown "
                    "disposition=%s worker=%s credit_id=%s "
                    "credits_outstanding=0", disposition, worker, cleared_id)
        return True

    def resume_credits_outstanding(self) -> int:
        with self._pause_lock:
            return self._resume_credits_outstanding

    def holds_resume_credit(self, conn_key: Any) -> bool:
        """True while `conn_key` still owns an UNDISPOSED reservation."""
        with self._pause_lock:
            return self._resume_credit_holder is conn_key

    def resume_credit_id(self) -> Optional[int]:
        """[S172-BP AMENDMENT F1-R2a] The token of the outstanding reservation, or
        None when none is outstanding. The pre-decode barrier compares against
        THIS: a barrier releases when the coordinator's current token is no longer
        the one the waiting reader delivered — whether that happened by exact
        disposition or by a force-clear."""
        with self._pause_lock:
            return self._resume_credit_id

    def resume_credit_id_for(self, conn_key: Any) -> Optional[int]:
        """[S172-BP AMENDMENT F1-R2a] The token `conn_key` currently holds, or
        None. Read by the woken reader to stamp the envelope it is about to
        deliver, and by the gates to assert on the EXACT reservation rather than on
        a count that cannot distinguish one grant from the next."""
        with self._pause_lock:
            if self._resume_credit_holder is not conn_key:
                return None
            return self._resume_credit_id

    def resume_credit_state(self, now: Optional[float] = None) -> Tuple[
            Optional[str], Optional[float]]:
        """(worker_id, age_seconds) of the outstanding reservation; (None, None)
        when clear. §4 metrics only."""
        now = time.time() if now is None else now
        with self._pause_lock:
            if self._resume_credit_holder is None:
                return None, None
            since = self._resume_credit_since
            return self._resume_credit_worker, (
                None if since is None else max(0.0, now - since))

    def _await_exact_credit_clear(self, credit_id: int, reader_stop) -> bool:
        """[S172-BP AMENDMENT F1-R2b] THE PRE-DECODE BARRIER's wait.

        Block this reader until the reservation identified by `credit_id` — the one
        it delivered — has been disposed of, whether by the dispatch seam's exact
        clear or by any force-clear path. Identity is the TOKEN, not the socket:
        the reader is waiting for ITS OWN envelope to be disposed of, and a socket
        test cannot tell that from some later grant.

        Round 2 ran this wait AFTER `recv_msg` returned. By then the connection
        owned TWO decoded envelopes — the credited one already in `inbound` and the
        next one in the reader's local — which breaks the one-decoded-envelope-per-
        connection bound the §1.2 resume margin is derived from. Called from the
        TOP of the loop instead, the next frame stays ON THE WIRE, where TCP
        back-pressure parks the worker's `_sendall` harmlessly (§1.1).

        Returns True when the reservation has cleared and the loop may decode
        again; False when the reader must exit — on `reader_stop` or the latched
        §1.5 capacity timeout. Exiting here holds NOTHING: the credited envelope
        was already delivered to `inbound`, so nothing is discarded and nothing is
        routed to the matrix. Cadence (50 ms), stop handling and the no-ledger-state
        rule are identical to the pause loop's.

        This cannot wedge: dispatch is single-threaded and unconditional, and every
        terminal path — eof, reaped socket, trial-terminal — clears the reservation
        on the serve side."""
        while not reader_stop.is_set():
            if self.resume_credit_id() != credit_id:
                return True
            if self.staging_capacity_timeout_expired():
                return False
            time.sleep(0.05)
        return False

    def _resume_paused_connections(self) -> None:
        """[§1.2 resume trigger] Exactly ONE grant per capacity-release event
        (F1). Called AFTER `_pump_deferred` at every capacity-release point, so the
        deferred queue gets first claim on a freed slot and a resumed reader does
        not immediately re-pause."""
        self._grant_resume_credit()

    # ----- F2 the lease-exemption resume grace -----------------------------
    def capacity_resume_grace(self, now: Optional[float] = None) -> Dict[str, float]:
        """Worker identities still inside their post-resume grace window, PRUNING
        expired entries in the same pass (F2 item 3)."""
        now = time.time() if now is None else now
        with self._pause_lock:
            for wid in [w for w, until in self._capacity_resume_grace.items()
                        if until <= now]:
                del self._capacity_resume_grace[wid]
            return dict(self._capacity_resume_grace)

    def clear_capacity_resume_grace(self, worker_id: Optional[str]) -> bool:
        """Drop one worker's grace entry. Called when `renew_lease` succeeds (the
        real lease is renewed, so the bridge is no longer needed) and when the
        worker's connection is dropped."""
        if worker_id is None:
            return False
        with self._pause_lock:
            return self._capacity_resume_grace.pop(worker_id, None) is not None

    def clear_all_capacity_resume_grace(self) -> int:
        """Trial-terminal cleanup: no grace outlives its trial (F2 item 5)."""
        with self._pause_lock:
            n = len(self._capacity_resume_grace)
            self._capacity_resume_grace.clear()
        return n

    def _release_capacity(self) -> None:
        """The named capacity-release sequence: pump the deferred queue, which then
        resumes paused readers in its `finally` (§1.2 ordering).

        Deliberately a THIN ALIAS for `_pump_deferred` rather than a second
        release path: putting the resume inside the pump is what let every
        pre-existing capacity-release caller — `_release_admission`,
        `_on_staging_failed`, `_submit_with_slot`'s completion callback, the
        staging-job success tail — become a resume point WITHOUT any out-of-scope
        method being edited (§0)."""
        self._pump_deferred()

    # ----- §1.5 the bounded capacity timeout -------------------------------
    def staging_capacity_timeout_expired(self, now: Optional[float] = None) -> bool:
        """True once the OLDEST currently-paused connection has been paused longer
        than `staging_capacity_timeout`. LATCHED: once observed it stays true, so a
        reader thread and the serve loop can never disagree about whether the
        bounded wait was exceeded."""
        if self._capacity_timeout_latched_at is not None:
            return True
        limit = float(getattr(self.config, "staging_capacity_timeout", 600.0) or 0.0)
        if limit <= 0:
            return False
        now = time.time() if now is None else now
        with self._pause_lock:
            # Re-checked under the lock: two reader threads and the serve loop can
            # all reach here, and exactly one of them must take the snapshot.
            if self._capacity_timeout_latched_at is not None:
                return True
            oldest = min((r["since"] for r in self._paused_connections.values()),
                         default=None)
            if oldest is None or (now - oldest) <= limit:
                return False
            self._capacity_timeout_latched_at = now
            # [S172-BP AMENDMENT F3] TIMEOUT EVIDENCE SNAPSHOT, taken in the SAME
            # critical section as the oldest-pause read that decided the timeout.
            # A reader can observe the latch, deregister and exit before the serve
            # loop builds the terminal reason; reading the live registry then
            # truthfully reports "0 connections paused (none)" about a timeout that
            # paused workers caused. The count and identities in the terminal
            # reason must be the TRIGGERING ones.
            self._capacity_timeout_snapshot = {
                "latched_at": now,
                "oldest_since": oldest,
                "paused_count": len(self._paused_connections),
                "worker_ids": sorted(
                    str(r["worker_id"]) for r in
                    self._paused_connections.values()),
            }
        return True

    def capacity_timeout_snapshot(self) -> Optional[Dict[str, Any]]:
        """The F3 evidence snapshot, or None if the timeout never latched."""
        with self._pause_lock:
            snap = self._capacity_timeout_snapshot
            return dict(snap) if snap is not None else None

    def staging_capacity_timeout_reason(self, now: Optional[float] = None) -> str:
        """The §1.5 terminal reason string. Leads with the ROOT CAUSE (the Part B
        convention) and is explicitly a COORDINATOR/INFRASTRUCTURE condition, so a
        reader of the terminal report is never pointed at a worker.

        [S172-BP AMENDMENT F3] The count and identities come from the SNAPSHOT
        taken when the latch was set. The live registry is consulted ONLY when the
        timeout never latched (there is then nothing to attribute)."""
        now = time.time() if now is None else now
        limit = float(getattr(self.config, "staging_capacity_timeout", 600.0) or 0.0)
        with self._pause_lock:
            snap = self._capacity_timeout_snapshot
            if snap is not None:
                n = int(snap["paused_count"])
                ids = list(snap["worker_ids"])
                held = max(0.0, float(snap["latched_at"])
                           - float(snap["oldest_since"]))
            else:
                n = len(self._paused_connections)
                ids = sorted(str(r["worker_id"]) for r in
                             self._paused_connections.values())
                held = None
        oldest_phrase = ("" if held is None
                         else f"; oldest pause held {held:.1f}s at the latch")
        return (f"coordinator_staging_capacity_timeout: staging did not release "
                f"capacity within {limit:.1f}s; {n} connections paused "
                f"({', '.join(ids) if ids else 'none'}){oldest_phrase}")

    # ----- §4 metrics ------------------------------------------------------
    def note_inbound_occupancy(self, qsize: int) -> None:
        with self._bp_lock:
            self._bp["inbound_qsize_high_water"] = max(
                self._bp["inbound_qsize_high_water"], int(qsize))

    def staging_backpressure_metrics(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        with self._bp_lock:
            out = dict(self._bp)
        started = out.pop("trial_started_at", None)
        elapsed = max(1e-9, now - started) if started else None
        out["elapsed_seconds"] = elapsed
        out["staging_jobs_per_sec"] = (
            out["staging_jobs_completed"] / elapsed if elapsed else None)
        out["derived_bound"] = self._derived_deferred_bound
        # [ALPHA REVIEW FIX, amendment round] THE TERMINAL SUMMARY MUST NEVER
        # RAISE. `staging_deferred_bound()` falls back to the on-demand derivation
        # when no stage bound was installed — and the one production path where
        # that is true at trial-terminal time is precisely an F5 sizing failure,
        # where the SAME malformed cap record that failed stage setup would now
        # raise HERE and mask the honest `coordinator_staging_sizing` termination
        # (the F3 disease, relocated to the reporting layer). Reporting degrades;
        # it never overwrites the terminal truth.
        try:
            out["bound_in_force"] = self.staging_deferred_bound()
        except Exception as _bound_exc:  # noqa: BLE001
            out["bound_in_force"] = None
            out["bound_in_force_error"] = (
                f"{type(_bound_exc).__name__}: {_bound_exc}")
        out["derived_bound_detail"] = dict(self._derived_bound_detail)
        out["deferred_now"] = len(self._deferred)
        out["paused_now"] = self.paused_connection_count()
        # [S172-BP AMENDMENT F3] the TRIGGERING evidence, not the live registry —
        # the same snapshot the terminal reason is built from.
        snap = self.capacity_timeout_snapshot()
        out["capacity_timeout_snapshot"] = snap
        out["paused_at_capacity_timeout"] = (
            int(snap["paused_count"]) if snap else 0)
        out["capacity_timeout_worker_ids"] = (
            list(snap["worker_ids"]) if snap else [])
        # [S172-BP AMENDMENT F1/F2] credit + grace occupancy
        out["resume_credits_outstanding"] = self.resume_credits_outstanding()
        # [S172-BP AMENDMENT F1-R §8.4] WHO holds the undisposed reservation and
        # for how long. The credit now spans reader -> serve-loop disposition, so
        # "outstanding" alone can no longer distinguish a healthy in-flight handoff
        # from a wedged one; both are None while it is clear.
        _credit_worker, _credit_age = self.resume_credit_state(now)
        out["resume_credit_holder_worker"] = _credit_worker
        out["resume_credit_age_s"] = _credit_age
        # [S172-BP AMENDMENT F1-R2a §7.5] WHICH reservation, not just whether one
        # exists — the token that the grant log, the clear log and the delivered
        # envelope all carry, so a wedged handoff is traceable to one grant.
        out["resume_credit_id"] = self.resume_credit_id()
        out["capacity_resume_grace_now"] = len(self.capacity_resume_grace(now))
        return out

    def log_staging_backpressure_summary(
        self, run_id: str, now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """The trial-terminal summary line (§4). One structured, grep-stable
        `[S172-BP] summary` record carrying every required series."""
        m = self.staging_backpressure_metrics(now)
        logger.info(
            "[S172-BP] summary run=%s inbound_qsize_high_water=%d "
            "deferred_high_water=%d derived_bound=%s bound_in_force=%s "
            "paused_high_water=%d pause_events=%d pause_seconds_total=%.3f "
            "pause_seconds_max=%.3f staging_jobs_completed=%d "
            "staging_jobs_per_sec=%s capacity_timeout_terminations=%d "
            "capacity_invariant_terminations=%d",
            run_id, m["inbound_qsize_high_water"], m["deferred_high_water"],
            m["derived_bound"], m["bound_in_force"], m["paused_high_water"],
            m["pause_events"], m["pause_seconds_total"], m["pause_seconds_max"],
            m["staging_jobs_completed"],
            ("%.3f" % m["staging_jobs_per_sec"]) if m["staging_jobs_per_sec"]
            is not None else "n/a",
            m["capacity_timeout_terminations"],
            m["capacity_invariant_terminations"])
        return m

    def _defer_locked(self, entry) -> bool:
        """Defect 1c (C4): bounded add to `_deferred` — both a COUNT cap and a
        retained-BYTES cap (staging_high_water_bytes). Returns False if adding
        would exceed either bound.

        [S172-BP §2] The COUNT cap is now the DERIVED bound
        (`staging_deferred_bound()`), not the deleted constant 64. With a bound of
        (conservative burst + margin) and the §1 reader pause holding traffic on
        the wire, a False return is MATHEMATICALLY UNREACHABLE — §1.6 treats it as
        a sizing INVARIANT violation, never as a matrix event.
        [S172-BP AMENDMENT F5, Beta item-3 ratification detail] A refusal records
        WHICH bound tripped in `_last_defer_refusal`, because the §1.6 invariant
        reason must distinguish the three cases — a derived-count trip is a SIZING
        defect, an operator-override-count trip is an OPERATOR decision that
        re-armed the condition, and a retained-bytes trip is a RAM ceiling that
        the count derivation does not govern at all. Written under the caller's
        `_admission_lock`, read by `enqueue_staging` on the same thread.
        Caller MUST hold _admission_lock."""
        max_count = self.staging_deferred_bound()
        max_bytes = int(self.config.staging_high_water_bytes)
        add_bytes = self._entry_bytes(entry)
        if len(self._deferred) + 1 > max_count:
            self._last_defer_refusal = (
                "operator_override_count_bound"
                if getattr(self.config, "staging_deferred_max", None) is not None
                else "derived_count_bound")
            return False
        if self._deferred_retained_bytes() + add_bytes > max_bytes:
            self._last_defer_refusal = "retained_bytes_high_water"
            return False
        self._last_defer_refusal = None
        self._deferred.append(entry)
        with self._bp_lock:
            self._bp["deferred_high_water"] = max(
                self._bp["deferred_high_water"], len(self._deferred))
        return True

    def _release_admission(self, run_id, stripe_id, attempt) -> None:
        """De-commit one attempt's admission budget (attempt failed/cleaned/
        superseded) and pump any deferred sub-stripes now that capacity may fit.
        [S172-BP §1.2] `_pump_deferred` is now the capacity-release point, so this
        resumes paused readers too — with this method left byte-identical."""
        with self._admission_lock:
            self._admitted.pop((run_id, stripe_id, attempt), None)
        self._pump_deferred()

    # ----- async staging (Defect 4: OFF the dispatch loop) ------------------
    def enqueue_staging(self, kind, wconn, run_id, stripe_id, attempt, sub_index,
                        msg, eligible_provider):
        """Admit + submit a staging job so fetch/verify/rename (and inline
        write+fsync) never run on the socket dispatch loop.

        Defect 1a (C4): this is NONBLOCKING on the dispatch thread. Admission and a
        bounded staging slot are attempted WITHOUT waiting; if no slot is free the
        work is DEFERRED (bounded, D1c) and resumed off the dispatch thread by
        _pump_deferred when a slot/capacity frees. The dispatch thread returns
        immediately in every branch:
          - the attempt cannot fit high-water at all (D1b)  -> fail fast (matrix)
          - admitted AND a slot is free                     -> submit
          - admitted-but-no-slot / not-yet-admittable       -> defer (bounded)
          - deferred bound reached                           -> back-pressure (matrix)"""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        action = None
        reason = None
        with self._admission_lock:
            stripe = self.ledger.get_stripe(run_id, stripe_id)
            # Defect 1 (C5): the BYTE guard uses ACTUAL advertised sizes, not the
            # inline ceiling — a remote spool can far exceed 48 MiB. A single shard
            # bigger than the byte high-water, or an attempt whose accumulated
            # advertised bytes exceed it, can NEVER all stage at once → fail fast
            # (never admit-then-loop on StagingBackPressure).
            hw_bytes = int(self.config.staging_high_water_bytes)
            shard_bytes = int(getattr(msg, "size_bytes", 0) or 0)
            attempt_bytes = self._attempt_actual_bytes(run_id, stripe_id, attempt)
            if shard_bytes > hw_bytes:
                reason = (f"shard advertises {shard_bytes} bytes > byte high-water "
                          f"{hw_bytes} (staging_high_water_bytes) — cannot ever fit")
            elif attempt_bytes > hw_bytes:
                reason = (f"attempt accumulated {attempt_bytes} advertised bytes > "
                          f"byte high-water {hw_bytes} (staging_high_water_bytes)")
            elif stripe is not None:
                reason = self._attempt_exceeds_highwater(stripe)
            if reason is not None:
                action = "failfast"
            else:
                admitted = self._try_admit_locked(run_id, stripe_id, attempt)
                # NONBLOCKING slot acquire (never waits on the dispatch thread).
                if admitted and self._staging_slots().acquire(blocking=False):
                    action = "submit"
                else:
                    entry = (kind, wconn, run_id, stripe_id, attempt, sub_index,
                             msg, eligible_provider, fut)
                    action = "deferred" if self._defer_locked(entry) else "backpressure"
        # Act OUTSIDE the admission lock; still nonblocking on the dispatch thread.
        if action == "submit":
            return self._submit_with_slot(kind, wconn, run_id, stripe_id, attempt,
                                          sub_index, msg, eligible_provider)
        if action == "deferred":
            return fut
        if action == "failfast":
            # D1b: capacity/config error the operator must fix — surfaced now,
            # NOT admitted to wait forever. Non-retryable (retry won't shrink it).
            self._on_staging_failed(run_id, stripe_id, False, eligible_provider,
                                    f"attempt cannot fit staging high-water: {reason}")
            if not fut.done():
                fut.set_result(None)
            return fut
        # [S172-BP §0 + §1.6] THE ONE REMOVED CALL SITE.
        #
        # This branch used to be
        #     self._on_staging_failed(run_id, stripe_id, True, eligible_provider,
        #                             "staging deferred queue full — dispatch back-pressure")
        # which charged a COORDINATOR-SIDE TRANSIENT CAPACITY CONDITION to a
        # worker's stripe as a fault, through the phase-specific retry matrix. On a
        # constant phase (every TFM run) the matrix's very next test is
        # `if phase in (1, 2): fail_trial(...)` (:3059-3066), so `retryable=True`
        # was inert and the trial died — which is exactly what killed the first
        # production-shape run past staging on 2026-08-05.
        #
        # Beta D, binding: coordinator staging back-pressure is a WAITING STATE,
        # not a stripe failure state. The wait now happens at the reader
        # (`_conn_reader_loop`, §1), keeping the payload on the wire / at the
        # worker, and NO capacity-wait event enters the matrix.
        #
        # Reaching HERE after §2's derived bound (>= conservative burst + margin)
        # means the SIZING was wrong, not that a worker failed. Log the full
        # arithmetic and terminate via a DIRECT fail_trial with an infrastructure
        # reason — never `handle_stripe_failure`, never the matrix.
        # ⚠ FLAGGED FOR BETA (§1.6): this disposition is Alpha's reading of "an
        # invariant, not a matrix event"; Beta may amend it.
        detail = self._derived_bound_detail or {}
        # [S172-BP AMENDMENT F5, Beta item-3] WHICH bound tripped, as one of three
        # explicit phrases. They are different defects: the derived-count bound is
        # a SIZING failure (Alpha's), the operator-override count is a decision
        # somebody made against the warning, and the retained-bytes high-water is a
        # RAM ceiling the count derivation does not govern. A single undifferentiated
        # "the deferred queue overflowed" sends all three to the wrong owner.
        _TRIP_PHRASES = {
            "derived_count_bound":
                "the DERIVED COUNT bound (burst_conservative + resume_margin) was "
                "exceeded — the stage sizing was wrong",
            "operator_override_count_bound":
                "the OPERATOR OVERRIDE COUNT bound (staging_deferred_max) was "
                "exceeded — an explicit override below the derived bound re-armed "
                "this condition",
            "retained_bytes_high_water":
                "the RETAINED-BYTES HIGH-WATER (staging_high_water_bytes) was "
                "exceeded — retained coordinator RAM, not the count bound",
        }
        tripped = self._last_defer_refusal
        trip_phrase = _TRIP_PHRASES.get(
            tripped, "the tripped bound could not be determined")
        arithmetic = (
            f"bound_tripped={tripped!r} ({trip_phrase}) "
            f"deferred={len(self._deferred)} bound_in_force="
            f"{self.staging_deferred_bound()} derived_bound="
            f"{self._derived_deferred_bound!r} override="
            f"{getattr(self.config, 'staging_deferred_max', None)!r} "
            f"resume_margin={self._resume_margin()} "
            f"burst_conservative={detail.get('burst_bound_conservative')!r} "
            f"slots={detail.get('slots')!r} phase={detail.get('phase')!r} "
            f"eligible_workers={detail.get('eligible_workers')!r} "
            f"retained_bytes={self._deferred_retained_bytes()} "
            f"byte_high_water={int(self.config.staging_high_water_bytes)} "
            f"staging_workers={getattr(self.config, 'staging_workers', None)!r} "
            f"staging_queue_depth="
            f"{getattr(self.config, 'staging_queue_depth', None)!r}")
        logger.error(
            "[S172-BP] CAPACITY INVARIANT VIOLATED at %s/%s — the deferred queue "
            "overflowed although the reader pause should have made that "
            "unreachable. This is a SIZING defect, not a worker fault: %s",
            run_id, stripe_id, arithmetic)
        with self._bp_lock:
            self._bp["capacity_invariant_terminations"] += 1
        self.fail_trial(
            run_id,
            reason=(f"coordinator_staging_capacity_invariant: deferred staging "
                    f"queue overflowed at {stripe_id}; {arithmetic}"))
        if not fut.done():
            fut.set_result(None)
        return fut

    def _submit_with_slot(self, kind, wconn, run_id, stripe_id, attempt, sub_index,
                          msg, eligible_provider):
        """A bounded admission slot is ALREADY held by the caller. Submit the staging
        job; on completion release the slot AND pump deferred work so a freed slot
        resumes a waiting sub-stripe. NEVER acquires/blocks here (Defect 1a C4)."""
        try:
            fut = self._staging_exec().submit(
                self._run_staging_job, kind, wconn, run_id, stripe_id, attempt,
                sub_index, msg, eligible_provider)
        except BaseException:
            self._staging_slots().release()
            raise

        def _on_done(_f):
            self._staging_slots().release()
            with self._bp_lock:
                self._bp["staging_jobs_completed"] += 1
            # a slot just freed — resume deferred work, and (inside the pump, §1.2)
            # any paused reader, in that order.
            self._pump_deferred()
        fut.add_done_callback(_on_done)
        return fut

    def _pump_deferred(self) -> None:
        """Resume deferred sub-stripes whose attempt can now be admitted AND a slot
        is free. Runs OFF the dispatch thread (staging-completion callback / matrix /
        ack). Dead attempts (terminal/superseded) are dropped. The slot is acquired
        NONBLOCKING under the admission lock; the submit happens outside it.

        [S172-BP §1.2 resume trigger] This IS the capacity-release point. Every
        caller that already pumped (`_release_admission`, `_submit_with_slot`'s
        completion callback, `_on_staging_failed`, the staging-job success tail)
        therefore also resumes paused readers — WITHOUT any of those out-of-scope
        methods being edited. Ordering is what Beta asked for: already-retained
        deferred payloads claim a freed slot FIRST, and only then is a paused
        reader allowed to add another."""
        try:
            ready: List[tuple] = []
            with self._admission_lock:
                self._prune_admitted_locked()
                if not self._deferred:
                    return
                still: List[tuple] = []
                for entry in self._deferred:
                    (_k, _w, run_id, stripe_id, attempt, _s, _m, _e, fut) = entry
                    if not self._attempt_live_locked(run_id, stripe_id, attempt):
                        if not fut.done():
                            fut.set_result(None)
                        continue
                    if (self._try_admit_locked(run_id, stripe_id, attempt)
                            and self._staging_slots().acquire(blocking=False)):
                        ready.append(entry)   # slot held for this entry
                    else:
                        still.append(entry)
                self._deferred = still
            for entry in ready:
                (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig,
                 fut) = entry
                try:
                    real = self._submit_with_slot(kind, wconn, run_id, stripe_id,
                                                  attempt, sub_index, msg, elig)
                except BaseException as e:  # noqa: BLE001
                    if not fut.done():
                        fut.set_exception(e)
                    continue
                self._chain_future(real, fut)
        finally:
            self._resume_paused_connections()

    @staticmethod
    def _chain_future(src: "concurrent.futures.Future",
                      dst: "concurrent.futures.Future") -> None:
        def _done(f):
            if dst.done():
                return
            exc = f.exception()
            if exc is not None:
                dst.set_exception(exc)
            else:
                dst.set_result(f.result())
        src.add_done_callback(_done)

    def _run_staging_job(self, kind, wconn, run_id, stripe_id, attempt, sub_index,
                         msg, eligible_provider):
        """Runs on the staging executor. For an ADMITTED attempt, reserves (with a
        nonblocking back-pressure WAIT + resume — never a timeout/reassign for
        self-inflicted capacity starvation, Defect 3 C3), fetches/writes, verifies,
        renames, marks verified — then completes the stripe. A genuine failure
        (fetch/IO error, fetch StagingTimeout, hash mismatch) is routed through the
        phase-specific matrix (never a bare log)."""
        while True:
            try:
                if kind == "inline":
                    survivors = (getattr(msg, "inline", None) or {}).get("survivors", [])
                    self.stage_inline_shard(
                        run_id, stripe_id, attempt, sub_index, msg.seed_start,
                        msg.seed_count, survivors, msg.size_bytes, msg.sha256)
                else:
                    self.stage_remote_shard(
                        wconn, run_id, stripe_id, attempt, sub_index, msg.spool_path,
                        msg.size_bytes, msg.sha256,
                        fetch_timeout=self.config.staging_timeout)
                break
            except StagingBackPressure:
                # Defect 3 (C3): an admitted attempt's sub-stripe that cannot reserve
                # yet WAITS and resumes (nonblocking, off the dispatch loop) until
                # capacity frees — it is NOT timed out into the retry matrix (that is
                # for genuine worker failure, not self-inflicted starvation). Bail
                # ONLY if the attempt is superseded / the stripe cancelled / the
                # trial terminal, in which case its admission budget is de-committed.
                if not self._attempt_live_locked(run_id, stripe_id, attempt):
                    self._release_admission(run_id, stripe_id, attempt)
                    return None
                time.sleep(0.02)
                continue
            except StagingHashMismatch:
                # Defect 5 (C3): an advertised-bytes hash mismatch is a FAILED
                # sub-stripe that feeds the one-retry path (approved brief), NOT a
                # trial-aborting non-retryable failure. retryable=True lets the
                # phase-specific matrix decide: constant phases (1/2) still fail
                # CLOSED, a hybrid stripe (3/4) gets its single retry to a DIFFERENT
                # worker (phase_degraded). Marking it non-retryable wrongly aborted
                # a hybrid trial on attempt 0.
                self._on_staging_failed(run_id, stripe_id, True, eligible_provider,
                                        "hash mismatch on advertised bytes")
                return None
            except StagingTimeout:
                self._on_staging_failed(run_id, stripe_id, True, eligible_provider,
                                        "staging timeout")
                return None
            except StagingConfigurationError as e:
                # [Part B §2] Caught BEFORE the generic handler below. A staging
                # CONFIGURATION defect (missing / conflicting / non-absolute /
                # unwritable / capacity-invalid) is PERMANENT -> retryable=False,
                # which routes to the matrix's existing non-retryable row and does
                # NOT consume a Q3 retry.
                #
                # ⚠ NARROW BY CONSTRUCTION: only this subtype is reclassified.
                # StagingError itself, StagingBackPressure, StagingHashMismatch and
                # StagingTimeout keep their existing classifications above, and the
                # generic transient handler below is unchanged.
                #
                # The reason string leads with the ROOT CAUSE so the terminal report
                # names staging configuration, not a downstream MinerIngressError.
                self._on_staging_failed(
                    run_id, stripe_id, False, eligible_provider,
                    f"staging configuration error (non-retryable): {e}")
                return None
            except Exception as e:  # noqa: BLE001 — transient fetch/IO -> retryable
                self._on_staging_failed(run_id, stripe_id, True, eligible_provider, str(e))
                return None
        # success -> try to complete the stripe (idempotent, lifecycle-locked).
        # Defect 4 (C3): pass the eligible provider so a definitive reconciliation
        # failure discovered when the final shard verifies is routed to the matrix.
        self.finalize_stripe(run_id, stripe_id, eligible_provider=eligible_provider)
        # Defect 3 (C3): if this shard's attempt just reached a terminal stripe state
        # (done / matrix-failed), its admission budget can be de-committed so any
        # deferred sub-stripes waiting on capacity resume.
        self._pump_deferred()
        return True

    def _on_staging_failed(self, run_id, stripe_id, retryable, eligible_provider, reason):
        logger.warning("staging failed %s/%s (retryable=%s): %s",
                       run_id, stripe_id, retryable, reason)
        eligible = eligible_provider() if callable(eligible_provider) else []
        self.handle_stripe_failure(run_id, stripe_id, retryable=retryable,
                                   eligible_workers=eligible)
        # Defect 3 (C3): the failed attempt is superseded (reassigned) or the trial
        # failed — de-commit its admission budget and resume deferred sub-stripes.
        self._pump_deferred()

    def _delete_remote(self, task: StagingTask, now: Optional[float] = None) -> None:
        """Invoke remote release ONLY after successful local verification
        (Decision B). A failure is recorded durably (SC1) and NEVER invalidates
        the already-verified shard."""
        now = time.time() if now is None else now
        try:
            self.transfer.delete_remote(task.node_config, task.remote_spool_path)
            self.ledger.set_remote_delete(
                task.run_id, task.stripe_id, task.attempt, task.sub_index,
                status="deleted", error=None, deleted_at=now)
        except Exception as e:
            self.ledger.set_remote_delete(
                task.run_id, task.stripe_id, task.attempt, task.sub_index,
                status="failed", error=str(e), deleted_at=None)

    def retry_remote_delete(
        self, node_config: NodeConfig, run_id: str, stripe_id: str, attempt: int,
        sub_index: int, remote_spool_path: str, now: Optional[float] = None,
    ) -> str:
        """Idempotent SC1 retry of a failed remote deletion. Already-deleted is a
        no-op; success/failure updates status durably. Never duplicates the shard."""
        now = time.time() if now is None else now
        shard = self.ledger.get_shard(run_id, stripe_id, attempt, sub_index)
        if shard is not None and shard["remote_delete_status"] == "deleted":
            return "deleted"
        try:
            self.transfer.delete_remote(node_config, remote_spool_path)
            self.ledger.set_remote_delete(
                run_id, stripe_id, attempt, sub_index,
                status="deleted", error=None, deleted_at=now)
            return "deleted"
        except Exception as e:
            self.ledger.set_remote_delete(
                run_id, stripe_id, attempt, sub_index,
                status="failed", error=str(e), deleted_at=None)
            return "failed"

    # ----- L2 acknowledgement + capacity release ---------------------------
    def ack_shard(
        self, run_id: str, stripe_id: str, attempt: int, sub_index: int,
        now: Optional[float] = None,
    ) -> None:
        """Stubbed Phase-5 ack (L6 keys this by event_id in Stage 4). Mere ack
        does NOT release capacity."""
        self.ledger.mark_shard_acked(run_id, stripe_id, attempt, sub_index, now)

    def release_after_ack(
        self, run_id: str, stripe_id: str, attempt: int, sub_index: int,
        reservation_id: int, now: Optional[float] = None,
    ) -> bool:
        """Capacity releases ONLY after Phase 5 has acked AND the local staged
        file is deleted (L2). Returns True if released, False if not yet acked."""
        now = time.time() if now is None else now
        shard = self.ledger.get_shard(run_id, stripe_id, attempt, sub_index)
        if shard is None or shard["phase5_status"] != "acked":
            return False
        self._cleanup_file(shard["local_staged_path"])
        self.ledger.mark_shard_local_deleted(run_id, stripe_id, attempt, sub_index, now)
        released = self.ledger.release_reservation(reservation_id, now)
        # Defect 3 (C3): capacity just freed — resume any deferred sub-stripes.
        self._pump_deferred()
        return released

    # ----- failure-path cleanup primitives (L8, gate 36) -------------------
    def cleanup_reservation(
        self, reservation_id: int, mark_shard_failed: bool = True,
        now: Optional[float] = None,
    ) -> None:
        """Remove a reservation's temp/staged file FIRST, then release capacity
        (staging-timeout / stale-callback / trial-abort). No reservation leak."""
        now = time.time() if now is None else now
        res = self.ledger.get_reservation(reservation_id)
        if res is None:
            return
        self._cleanup_file(res.get("staged_path") or res.get("temp_path"))
        if mark_shard_failed:
            self.ledger.set_shard_staging_status(
                res["run_id"], res["stripe_id"], res["attempt"], res["sub_index"],
                SH_FAILED)
        self.ledger.release_reservation(reservation_id, now)

    # ----- L6 event-id ack (idempotent capacity release) -------------------
    def ack_by_event_id(self, event_id: str, now: Optional[float] = None) -> bool:
        """Phase 5 acks a shard by its immutable event_id (L6). Idempotent: the
        reservation releases EXACTLY once (a duplicate ack finds it already
        released and is a no-op); an ack for event A never touches event B's
        reservation (lookup is keyed solely by event_id)."""
        now = time.time() if now is None else now
        res = self.ledger.get_reservation_by_event(event_id)
        if res is None or res["status"] != "held":
            return False
        self.ledger.mark_shard_acked(
            res["run_id"], res["stripe_id"], res["attempt"], res["sub_index"], now)
        self._cleanup_file(res["staged_path"])
        self.ledger.mark_shard_local_deleted(
            res["run_id"], res["stripe_id"], res["attempt"], res["sub_index"], now)
        released = self.ledger.release_reservation(res["reservation_id"], now)
        # Defect 3 (C3): capacity just freed — resume any deferred sub-stripes.
        self._pump_deferred()
        return released

    # ----- attempt-scoped cleanup (Blocker 2) ------------------------------
    def cleanup_attempt(
        self, run_id: str, stripe_id: str, attempt: int, now: Optional[float] = None,
    ) -> None:
        """Invalidate + remove ALL of a failed attempt's local shards and release
        their reservations (Blocker 2). Publish is attempt-scoped and happens only
        at completion, so a failed attempt has published nothing — no Phase-5
        discard is needed here; the stripe is retried WHOLE."""
        now = time.time() if now is None else now
        for res in self.ledger.held_reservations(run_id, stripe_id, attempt):
            self._cleanup_file(res.get("staged_path") or res.get("temp_path"))
            self.ledger.release_reservation(res["reservation_id"], now)
        for sh in self.ledger.get_shards(run_id, stripe_id, attempt):
            self.ledger.set_shard_staging_status(
                run_id, stripe_id, attempt, sh["sub_index"], SH_FAILED)
        # Defect 3 (C3): the attempt's budget is de-committed + deferred work resumes.
        self._release_admission(run_id, stripe_id, attempt)

    # ----- retry matrix (Blocker 3) ----------------------------------------
    def _pick_other_worker(
        self, workers: List[Any], exclude_worker_id: str, family_name: str
    ) -> Optional[Any]:
        for w in workers:
            if getattr(w, "worker_id", None) == exclude_worker_id:
                continue
            if self.can_assign_variant(w, family_name):
                return w
        return None

    def handle_stripe_failure(
        self, run_id: str, stripe_id: str, retryable: bool,
        eligible_workers: List[Any], now: Optional[float] = None,
        lease_expiry: bool = False,
    ) -> Dict[str, Any]:
        """Blocker 3 retry matrix, implemented EXACTLY (workflow phase 1-4, §6.8):
          - retryable=False              -> fail trial immediately (retry NOT consumed)
          - workflow phase 1/2 (constant)-> fail trial immediately (fail closed)
          - phase 3/4 (hybrid) 1st retryable failure -> reassign ONCE to a DIFFERENT
            eligible worker, phase_degraded=True
          - phase 3/4 2nd retryable failure          -> fail trial
          - lease expiry                              -> same phase-specific policy
        No MAX_ATTEMPTS. Returns a dict describing the action taken."""
        now = time.time() if now is None else now
        # Lifecycle-locked (Defect 4/5): serialize matrix decisions against
        # finalize/commit/other-failure calls from the staging + dispatch threads.
        with self._lifecycle_lock:
            return self._handle_stripe_failure_locked(
                run_id, stripe_id, retryable, eligible_workers, now, lease_expiry)

    def _handle_stripe_failure_locked(
        self, run_id, stripe_id, retryable, eligible_workers, now, lease_expiry,
    ) -> Dict[str, Any]:
        stripe = self.ledger.get_stripe(run_id, stripe_id)
        if stripe is None:
            raise KeyError(f"unknown stripe {run_id}/{stripe_id}")
        # Defect 4/5: a late staging failure can arrive after the trial is already
        # terminal (committed/aborted) or the stripe already settled — do nothing.
        trial = self.ledger.get_trial(run_id)
        if trial is not None and trial["state"] in ("committed", "aborted"):
            return {"action": "noop", "reason": "trial already terminal"}
        if stripe["state"] in (ST_DONE, ST_FAILED, ST_CANCELLED):
            return {"action": "noop", "reason": f"stripe already {stripe['state']}"}
        phase = stripe["phase"]
        attempt = stripe["current_attempt"]

        # ANY explicit non-retryable failure -> fail trial, retry NOT consumed.
        # (Lease expiry is treated as a retryable condition routed through the
        # phase policy below, per the matrix.)
        if not retryable and not lease_expiry:
            self.fail_trial(run_id, reason=f"{stripe_id}: non-retryable failure", now=now)
            return {"action": "fail_trial", "reason": "non_retryable"}

        # Constant workflow phases fail closed.
        if phase in (1, 2):
            self.fail_trial(run_id, reason=f"{stripe_id}: constant-phase failure", now=now)
            return {"action": "fail_trial", "reason": "constant_phase"}

        # Hybrid workflow phases: one retry then fail.
        if attempt == 0:
            # Blocker 2: clean up the failed attempt's local shards BEFORE retry.
            self.cleanup_attempt(run_id, stripe_id, attempt, now)
            other = self._pick_other_worker(
                eligible_workers, stripe["claimed_by"], stripe["family_name"])
            if other is None:
                self.fail_trial(
                    run_id, reason=f"{stripe_id}: no alternate eligible worker", now=now)
                return {"action": "fail_trial", "reason": "no_alternate_worker"}
            # Fence the superseded assignment (staging_generation++ — L5) and
            # requeue the WHOLE stripe as attempt 1 on the different worker.
            self.ledger.set_stripe_fields(
                run_id, stripe_id, phase_degraded=1, state=ST_PENDING,
                claimed_by=None, lease_expires_at=None,
                staging_generation=stripe["staging_generation"] + 1)
            cap = advertised_effective_cap(
                other.backend, stripe["family_name"], other.seed_caps)
            expected = expected_substripes_for(stripe["seed_count"], cap)
            self.ledger.claim_stripe(
                run_id, stripe_id, other.worker_id, 1, expected,
                now + self.config.compute_lease_timeout)
            if hasattr(other, "record_assignment"):
                other.record_assignment(stripe_id, 1)
            return {"action": "reassigned", "worker_id": other.worker_id,
                    "attempt": 1, "phase_degraded": True}

        # Second hybrid failure -> fail trial.
        self.fail_trial(run_id, reason=f"{stripe_id}: hybrid second failure", now=now)
        return {"action": "fail_trial", "reason": "hybrid_second_failure"}

    def process_lease_expiry(
        self, run_id: str, eligible_workers: List[Any], now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Apply the phase-specific matrix to every expired COMPUTE lease
        (state='claimed'). staging leases are never here (Blocker 5).

        [S172-BP §1.4 — LEASE EXEMPTION, REQUIRED BY BETA'S OWN GATES 1-2 AND
        FLAGGED FOR RATIFICATION]
        The back-pressure ruling is SILENT on leases; its gates are not satisfiable
        without this. Heartbeats are the ONLY compute-lease renewal path
        (`_serve_dispatch` :4275-4282, `compute_lease_timeout = 300.0` at :225) and
        they ride the SAME ordered TCP stream as results. A connection paused by
        the coordinator therefore stops delivering renewals, and this scan would
        route the resulting expiry into the matrix with `lease_expiry=True` — which
        SKIPS the non-retryable branch and lands squarely on the constant-phase
        `fail_trial` (:3059-3066). Any pause longer than 300s would red Beta gates
        1-2 through that door.

        So a stripe whose claiming worker's connection is in COORDINATOR-INITIATED
        pause is SKIPPED: the coordinator caused the silence and knows it. The
        exemption is narrow by construction —
          * membership in `_paused_connections` is the only qualifier, and only the
            coordinator ever writes it;
          * the pause is itself bounded by §1.5, so no stripe can be exempt
            forever;
          * an UNPAUSED worker's genuine silence still expires normally.
        On resume, the queued heartbeats process and renewal restarts.

        [S172-BP AMENDMENT F2 — THE RESUME GRACE]
        Live membership in `_paused_connections` is NOT sufficient. A resuming
        reader deregisters FIRST, then delivers its held envelope, and only later
        reaches the heartbeat that was queued behind it on the same ordered TCP
        stream. In that window the worker is unpaused, its lease has been expired
        for as long as the pause ran (up to `staging_capacity_timeout` = 600 s
        against `compute_lease_timeout` = 300 s), and its renewal is still in
        flight — so this scan would route the stripe into the matrix for a silence
        the coordinator itself caused. A worker with a LIVE grace entry is
        therefore skipped too. Expired grace entries are pruned in the same pass,
        so an entry that outlives its bound with no heartbeat simply stops
        matching and normal expiry handling resumes.
        """
        now = time.time() if now is None else now
        paused = self.paused_worker_ids()
        grace = self.capacity_resume_grace(now)
        out = []
        for st in self.ledger.expired_claimed_stripes(run_id, now):
            if st["claimed_by"] in paused:
                logger.info(
                    "[S172-BP] lease_exempt stripe=%s worker=%s — expiry SKIPPED: "
                    "the coordinator paused this connection for staging capacity, "
                    "so the missing heartbeat is coordinator-caused (bounded by "
                    "staging_capacity_timeout=%.1fs)",
                    st["stripe_id"], st["claimed_by"],
                    float(getattr(self.config, "staging_capacity_timeout", 600.0)))
                continue
            if st["claimed_by"] in grace:
                logger.info(
                    "[S172-BP] lease_grace stripe=%s worker=%s — expiry SKIPPED: "
                    "this connection resumed from a coordinator-initiated pause "
                    "and its renewing heartbeat is still queued behind the "
                    "envelope it was holding; grace until=%.3f now=%.3f",
                    st["stripe_id"], st["claimed_by"],
                    grace[st["claimed_by"]], now)
                continue
            out.append(self.handle_stripe_failure(
                run_id, st["stripe_id"], retryable=True,
                eligible_workers=eligible_workers, now=now, lease_expiry=True))
        return out

    # ----- trial lifecycle (Blocker 2, L3, L7) -----------------------------
    def commit_trial(self, run_id: str, now: Optional[float] = None) -> Dict[str, Any]:
        """All stripes done -> TrialCommit. Defect 5: terminal + mutually exclusive
        with abort (refused with TrialAborted once aborted). Immutable event_id
        `{run_id}:commit`, durable delivery status, deliver AFTER the terminal
        decision, idempotent by event_id (a duplicate delivery is a no-op)."""
        now = time.time() if now is None else now
        commit_event_id = f"{run_id}:commit"
        deliver = False
        with self._lifecycle_lock:
            trial = self.ledger.get_trial(run_id)
            if trial is not None and trial["state"] == "aborted":
                raise TrialAborted(f"trial {run_id} is aborted; TrialCommit prohibited")
            if trial is not None and trial["state"] == "committed":
                event_id = trial["commit_event_id"] or commit_event_id
                # already committed: re-deliver ONLY if a prior delivery failed
                deliver = trial["commit_delivery_status"] != "done"
            else:
                if not self.ledger.mark_trial_committed(run_id, commit_event_id, now):
                    raise TrialAborted(
                        f"trial {run_id} is aborted; TrialCommit prohibited")
                event_id = commit_event_id
                deliver = True
        event = {"event_type": "trial_commit", "run_id": run_id, "event_id": event_id}
        if deliver:
            try:
                if self.phase5_sink is not None:
                    self.phase5_sink.commit_trial(event)
                self.ledger.set_trial_commit_status(run_id, "done")
                event["delivery"] = "done"
            except Exception as e:  # noqa: BLE001 — durable + retryable
                self.ledger.set_trial_commit_status(run_id, "failed")
                event["delivery"] = "failed"
                event["error"] = str(e)
        else:
            event["delivery"] = "done"
            event["duplicate"] = True
        return event

    def fail_trial(
        self, run_id: str, reason: str = "", now: Optional[float] = None
    ) -> Dict[str, Any]:
        """Terminal trial failure -> whole-trial abort routed OFF the dispatch loop
        via the cleanup executor (Defect 5), waiting for the synchronous discharge
        to complete. Never runs the abort discharge on the dispatcher."""
        return self.submit_abort(run_id, reason, now).result()

    def abort_trial(
        self, run_id: str, reason: str = "", now: Optional[float] = None
    ) -> Dict[str, Any]:
        """Whole-trial terminal abort (L3) with the L7 synchronous discharge
        (Team Beta binding — Option A). Order (run off the dispatch loop, see
        submit_abort): persist trial=aborted -> fence all active assignments ->
        synchronously call Phase5Sink.abort_trial() -> ONLY on its successful
        return remove staged files + release reservations + complete bookkeeping.

        On raise/timeout: the trial stays terminally aborted, TrialCommit stays
        prohibited, staged files + reservations are RETAINED (never deleted merely
        because delivery was attempted), cleanup_status becomes 'failed', and the
        call is retried idempotently. Idempotent by (event_id, run_id).

        D1.0 terminal-race correction [TB-D1-C1]: the terminal decision is made by
        CAS-RESULT DISAMBIGUATION PLUS A TERMINAL-STATE RE-READ, using the ledger's
        existing atomic `UPDATE ... WHERE state='running'` transitions — NOT a
        lock. The pre-D1.0 code early-returned on `committed` from a possibly STALE
        pre-read and then treated a False from mark_trial_aborted as
        "already aborted, retry the discharge", so a commit that won the atomic race
        in between still got its sink assembly cleared, tombstoned, and its staged
        spools deleted. `False` is now disambiguated: False-because-COMMITTED
        refuses; False-because-already-ABORTED retries the discharge idempotently.

        NO `_lifecycle_lock` is acquired here (Team Beta binding [TB-D1-DL]):
        `fail_trial` = `submit_abort(...).result()` is called from
        `_handle_stripe_failure_locked` while the caller ALREADY holds
        `_lifecycle_lock`, so acquiring it on the cleanup-executor thread would
        deadlock permanently (RLock is reentrant only for the same thread). The
        ledger methods' own internal `_write_lock` is unaffected."""
        now = time.time() if now is None else now
        self.ledger.create_trial(run_id, -1, now)
        trial = self.ledger.get_trial(run_id)
        if trial is not None and trial["state"] == "committed":
            # Defect 5: committed is terminal + mutually exclusive with abort —
            # a committed trial can NEVER be flipped to aborted.
            return {"event": None, "cleanup": "refused", "first": False,
                    "refused": "already_committed"}
        abort_event_id = f"{run_id}:abort"
        if trial is not None and trial["state"] == "aborted":
            first = False
        else:
            first = self.ledger.mark_trial_aborted(run_id, abort_event_id, now)
            if not first:
                # The read above may now be stale. Determine which terminal
                # transition actually won the atomic state='running' race.
                trial = self.ledger.get_trial(run_id)
                if trial is not None and trial["state"] == "committed":
                    return {"event": None, "cleanup": "refused", "first": False,
                            "refused": "already_committed"}
                if trial is None or trial["state"] != "aborted":
                    raise RuntimeError(
                        f"unexpected terminal transition for {run_id!r}")
        if first:
            # fence every still-active assignment (L3): pending/claimed/staging -> cancelled
            self.ledger.cancel_active_stripes(run_id)
        event = {"event_type": "trial_abort", "run_id": run_id,
                 "event_id": abort_event_id, "reason": reason}
        try:
            if self.phase5_sink is not None:
                self.phase5_sink.abort_trial(event)   # SYNCHRONOUS (Option A)
            # Success guaranteed: Phase 5 holds no trial-owned path. ONLY NOW may we
            # delete remaining staged files + release reservations (L7 / gate 34).
            for res in self.ledger.held_reservations(run_id):
                self._cleanup_file(res.get("staged_path") or res.get("temp_path"))
                self.ledger.release_reservation(res["reservation_id"], now)
            self.ledger.set_trial_cleanup_status(run_id, "done")
            # Defect 3 (C3): the trial's stripes are cancelled — de-commit every
            # admission budget and resolve/drop any deferred sub-stripes.
            self._pump_deferred()
            return {"event": event, "cleanup": "done", "first": first}
        except Exception as e:
            # RETAIN files + reservations; cleanup failed; retried idempotently.
            self.ledger.set_trial_cleanup_status(run_id, "failed")
            return {"event": event, "cleanup": "failed", "error": str(e), "first": first}

    def _executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if getattr(self, "_cleanup_executor", None) is None:
            self._cleanup_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="miner-cleanup")
        return self._cleanup_executor

    def submit_abort(
        self, run_id: str, reason: str = "", now: Optional[float] = None
    ) -> "concurrent.futures.Future":
        """Route the synchronous abort discharge onto the cleanup executor so it
        NEVER runs inside the socket receive/dispatch loop (L7 dispatch-thread
        requirement). Returns a Future; the caller waits for successful completion."""
        return self._executor().submit(self.abort_trial, run_id, reason, now)

    # ----- threshold provenance + parent-side enforcement (S172 D6) --------
    #
    # Beta's ruling: the three provenance conditions may NOT be merely detected
    # and recorded. D6 must FAIL CLOSED on them, because otherwise a trial could
    # be certified after the parent observed that the effective threshold was
    # missing, mismatched or inconsistent — which would hollow out the central D6
    # claim that the requested threshold physically reached execution.
    #
    # ONE registry is the single source of truth for BOTH the audit record and
    # the enforcement gate, so what is reported and what is enforced can never
    # drift apart. Keyed run_id -> (stripe_id, attempt).

    def _prov_record(self, run_id: str, stripe_id: str, attempt) -> Dict[str, Any]:
        """Get-or-create the provenance record for one (stripe, attempt).
        Caller MUST hold _provenance_lock."""
        return self._assignment_provenance.setdefault(run_id, {}).setdefault(
            (str(stripe_id), int(attempt)),
            {"stripe_id": str(stripe_id), "attempt": int(attempt), "phase": None,
             "assigned": None, "d6_generated": False,
             "sub_effective": {}, "complete_effective": None,
             "complete_seen": False})

    def record_assignment_threshold(self, run_id: str, stripe_id: str, attempt,
                                    phase, assigned, *,
                                    d6_generated: bool = True) -> None:
        """Record the threshold this assignment CARRIED, and — explicitly, not
        implicitly — whether it is a D6-generated assignment.

        `d6_generated` is the legacy distinction Beta required be made in code:
        a D6-generated assignment MUST come back with provenance, and absent
        provenance on one is a VIOLATION, not a legacy case. Optional schema
        fields make legacy absence *representable*; they do not make provenance
        optional for a D6 run. Every assignment `build_stripe_assign_payload`
        produces is D6-generated by construction (it cannot emit a payload
        without a resolved threshold), so in the current serve path this flag is
        always True; it exists so that any future explicitly-recognised legacy
        work has a defined, visible place to be marked — never a silent default.
        """
        with self._provenance_lock:
            rec = self._prov_record(run_id, stripe_id, attempt)
            rec["assigned"] = None if assigned is None else float(assigned)
            rec["phase"] = None if phase is None else int(phase)
            rec["d6_generated"] = bool(d6_generated)

    def record_substripe_effective(self, run_id: str, stripe_id: str, attempt,
                                   sub_index, value) -> None:
        """Record what the kernel ACTUALLY filtered at for one sub-stripe.

        A `None` is stored AS None — never dropped. Dropping it would turn a
        missing value into a silently shorter list, i.e. into apparent agreement;
        the whole point of this leg is that absence stays visible to the
        validator."""
        with self._provenance_lock:
            rec = self._prov_record(run_id, stripe_id, attempt)
            rec["sub_effective"][int(sub_index)] = (
                None if value is None else float(value))

    def record_stripe_complete_effective(self, run_id: str, stripe_id: str,
                                         attempt, value) -> None:
        """Record the stripe-level roll-up the worker reported on StripeComplete."""
        with self._provenance_lock:
            rec = self._prov_record(run_id, stripe_id, attempt)
            rec["complete_seen"] = True
            rec["complete_effective"] = None if value is None else float(value)

    def _current_assignment_records(self, run_id: str) -> List[Dict[str, Any]]:
        """The provenance records for each stripe's CURRENT attempt only.

        A superseded attempt (one the retry matrix moved past) is deliberately
        excluded: its results were discarded, so holding the trial to them would
        fail a run that actually recovered correctly."""
        with self._provenance_lock:
            records = dict(self._assignment_provenance.get(run_id, {}))
        out: List[Dict[str, Any]] = []
        for (stripe_id, attempt), rec in sorted(records.items()):
            stripe = self.ledger.get_stripe(run_id, stripe_id)
            if stripe is not None and int(stripe["current_attempt"]) != attempt:
                continue
            out.append(dict(rec, sub_effective=dict(rec["sub_effective"])))
        return out

    def threshold_provenance(self, run_id: str,
                             trial_ctx: Optional[Dict[str, Any]] = None,
                             validated: Optional[bool] = None
                             ) -> Dict[str, Any]:
        """The three-leg audit record for `run_id`:

            {"requested": {"forward": f, "reverse": r},
             "payload":   {phase: [values...]},
             "effective": {phase: [values...]},
             "phase_direction": {phase: "forward"|"reverse"},
             "validated": bool}

        This is the mechanism that makes a "threshold adaptation applied" claim
        checkable against physical reality: `requested` is what the tuner asked
        for, `payload` is what the parent transmitted, `effective` is what the
        GPU filtered at. Never assume the three agree — `validate_threshold_
        provenance` compares them and fails the trial when they do not.

        `validated` records whether the parent's fail-closed gate PASSED for this
        run. Downstream consumers (candidate ingress, accumulator mutation,
        finalization) must refuse a run whose provenance is not validated: an
        absent or False flag is not a neutral "unknown", it means the physical
        evidence was never checked.
        """
        ctx = trial_ctx if trial_ctx is not None else (
            self.ledger.get_trial_context(run_id) or {})
        payload: Dict[int, List[float]] = {}
        effective: Dict[int, List[float]] = {}
        for rec in self._current_assignment_records(run_id):
            phase = rec["phase"]
            if phase is None:
                continue
            if rec["assigned"] is not None:
                payload.setdefault(phase, []).append(rec["assigned"])
            for value in rec["sub_effective"].values():
                if value is not None:
                    effective.setdefault(phase, []).append(value)
            if rec["complete_effective"] is not None:
                effective.setdefault(phase, []).append(rec["complete_effective"])
        payload = {p: sorted(set(v)) for p, v in payload.items()}
        effective = {p: sorted(set(v)) for p, v in effective.items()}
        directions = {}
        for p in sorted(set(payload) | set(effective)):
            try:
                directions[p] = workflow_phase_semantics(p)[0]
            except MinerMetadataError:      # pragma: no cover — unreachable
                directions[p] = "unknown"
        requested = {}
        if ctx.get("forward_threshold") is not None:
            requested["forward"] = float(ctx["forward_threshold"])
        if ctx.get("reverse_threshold") is not None:
            requested["reverse"] = float(ctx["reverse_threshold"])
        return {"run_id": run_id, "requested": requested, "payload": payload,
                "effective": effective, "phase_direction": directions,
                "validated": bool(validated)}

    def validate_threshold_provenance(self, run_id: str) -> List[Dict[str, Any]]:
        """FAIL-CLOSED parent-side gate over every D6-generated assignment.

        Beta's five conditions, all enforced here (the fifth — the
        constant/hybrid equality invariant — is enforced earlier and unchanged,
        at payload construction and again on worker receipt):

          1. `effective_threshold` MUST be present;
          2. it MUST equal the threshold assigned in the payload;
          3. all sub-stripes of a stripe MUST report the SAME effective value;
          4. the stripe-complete value MUST agree with the sub-stripe consensus.

        Called BEFORE `commit_trial`, so it precedes Phase-5 assembly, candidate
        ingress, accumulator mutation and `finalize_run`. A violating trial can
        therefore never reach certification.

        Returns the validated records on success. Raises ThresholdProvenanceError
        listing EVERY violation found — the first one does not mask the rest.
        """
        records = self._current_assignment_records(run_id)
        if not records:
            raise ThresholdProvenanceError(
                f"run {run_id!r}: no threshold provenance was recorded for any "
                f"assignment. A D6 trial cannot be certified without physical "
                f"evidence that the requested threshold reached execution.")

        violations: List[str] = []
        for rec in records:
            sid, att = rec["stripe_id"], rec["attempt"]
            tag = f"{sid}#attempt{att}"
            if not rec["d6_generated"]:
                # Explicitly recognised legacy work: absence is permitted HERE and
                # only here. This branch is unreachable from the current serve
                # path (see record_assignment_threshold).
                continue
            assigned = rec["assigned"]
            if assigned is None:
                violations.append(
                    f"{tag}: D6-generated assignment with NO assigned threshold "
                    f"recorded")
                continue

            subs = rec["sub_effective"]
            if not subs:
                violations.append(
                    f"{tag}: NO sub-stripe reported an effective threshold "
                    f"(assigned {assigned}). Absent provenance on a D6-generated "
                    f"assignment is a violation, not a legacy case.")
                continue

            missing = sorted(i for i, v in subs.items() if v is None)
            if missing:
                violations.append(
                    f"{tag}: sub-stripe(s) {missing} reported no effective "
                    f"threshold (assigned {assigned}) — the kernel filter is "
                    f"unverified for those sub-stripes")

            present = [v for v in subs.values() if v is not None]
            distinct = sorted(set(present))
            if len(distinct) > 1:
                violations.append(
                    f"{tag}: sub-stripes DISAGREE on the effective threshold "
                    f"{distinct} — one stripe filtered at more than one value")

            for idx, value in sorted(subs.items()):
                if value is not None and value != assigned:
                    violations.append(
                        f"{tag}: sub-stripe {idx} filtered at {value} but was "
                        f"assigned {assigned} — the kernel did not use the "
                        f"requested threshold")

            if not rec["complete_seen"]:
                violations.append(
                    f"{tag}: no stripe_complete provenance was reported, so the "
                    f"stripe-level effective value cannot be reconciled with the "
                    f"sub-stripe consensus")
            else:
                complete = rec["complete_effective"]
                if complete is None:
                    violations.append(
                        f"{tag}: stripe_complete reported NO effective threshold "
                        f"(sub-stripe consensus {distinct})")
                elif complete != assigned:
                    violations.append(
                        f"{tag}: stripe_complete reports {complete} but the "
                        f"assigned threshold was {assigned}")
                elif len(distinct) == 1 and complete != distinct[0]:
                    violations.append(
                        f"{tag}: stripe_complete reports {complete} but the "
                        f"sub-stripe consensus is {distinct[0]}")

        if violations:
            raise ThresholdProvenanceError(
                f"run {run_id!r}: {len(violations)} threshold-provenance "
                f"violation(s) — refusing to commit a trial whose kernel filter "
                f"is unproven:\n  - " + "\n  - ".join(violations))
        return records

    # ----- assignment payload contract (ties Blocker 6 / Stage 0) ----------
    def build_stripe_assign_payload(
        self, dataset_path: str, window_size: int, sessions, offset: int,
        residues, dataset_sha256: Optional[str] = None,
        *, phase: int, forward_threshold: float, reverse_threshold: float,
    ) -> Dict[str, Any]:
        """Every StripeAssignMessage.payload MUST carry `dataset`, `dataset_sha256`
        (coordinator-computed), `window_size`, `sessions`, `offset`,
        `residue_sha256` (computed via the SAME sha256_residues the worker uses),
        and — since the D6 correction — the RESOLVED directional sieve threshold.
        dataset_sha256 and residue_sha256 are NEVER optional — the worker's
        Blocker-6 check rejects an assignment lacking dataset_sha256.

        THRESHOLD CONTRACT (S172 D6 correction; Beta §1/§2)
        --------------------------------------------------
        `phase`, `forward_threshold` and `reverse_threshold` are REQUIRED
        keyword-only arguments with NO defaults, so it is structurally impossible
        to build a D6 payload that omits the threshold and lets the worker fall
        back to its legacy hardcoded 0.25. The direction comes from the §6.8
        phase table via `workflow_phase_semantics` (phases 1,3 -> forward;
        2,4 -> reverse), which itself fails closed on an unknown phase — the
        worker never chooses a direction and never re-reads the trial config.

        The payload emits TWO threshold keys, and D6 pins them EQUAL. This is not
        a first-stage/second-stage cascade — it is what the executor actually
        needs, read from the live kernel arg builders rather than inferred from
        the field names:

          * constant-skip stripes (phases 1,2) run `_constant_prefix`, whose only
            threshold scalar is `ctx.threshold`, sourced from the payload's
            `min_match_threshold` (range_miner_worker.py `_constant_prefix` ->
            `ScalarArg(ctx.threshold, "float32")`; host post-filter `rate >=
            threshold`).
          * hybrid/variable stripes (phases 3,4) run `_hybrid_prefix`, whose only
            threshold scalar is `ctx.hybrid_threshold`, sourced from the payload's
            `phase2_threshold` when present (`ScalarArg(ctx.hybrid_threshold,
            "float32")`; host post-filter `rate >= hybrid_threshold`).
            `ctx.threshold` is NEVER passed to a hybrid kernel.

        So each stripe launches exactly ONE kernel with exactly ONE threshold
        scalar, and which payload key feeds it is decided by SKIP MODE, not by
        stage. `phase2_threshold` is a legacy PWC name for "the hybrid kernel's
        single threshold", not a second pass. Emitting both keys with the same
        resolved directional value is therefore the only assignment under which
        the Optuna-tuned per-direction threshold reaches the kernel in all four
        §6.8 phases. The equality is asserted here and re-validated by the worker
        (`ThresholdContractError`) so a contradictory pair can never pass
        silently.

        ⚠️ THE `phase2_threshold` NAMING TRAP — READ BEFORE "FIXING" THIS
        ----------------------------------------------------------------
        The name says "phase 2". It does NOT mean a second pass, a second stage,
        or a stricter follow-up filter. **There is no stage-2 filter in the
        miner.** The name is historical: it comes from the legacy PWC / Step-1
        job schema (`phase1_threshold` / `phase2_threshold` in the old
        `job_sieve_*.json` shape), where "phase 2" meant the hybrid/variable-skip
        run — a DIFFERENT KERNEL, not a later stage of the same one. The miner
        inherited only the key name, through
        `sieve_gpu_worker.py`'s hybrid branch.

        The facts above were established by READING the executor
        (`range_miner_worker.py`: `_constant_prefix`, `_hybrid_prefix`,
        `BuildContext`, and the two host post-filters), NOT by inferring meaning
        from the field names. In particular: `ctx.threshold` NEVER reaches a
        hybrid kernel, and `ctx.hybrid_threshold` never reaches a constant one.
        Each is the sole threshold argument of its own kernel ABI.

        So: **do NOT add a second-stage path, and do NOT "relax" or "correct" the
        equality invariant on the assumption that `phase2_threshold` is supposed
        to be a later, stricter stage.** Making the two values differ would not
        create a two-stage filter — it would simply mean constant-skip and
        variable-skip stripes of the SAME trial silently filtered at different
        thresholds, which is the class of defect this whole correction pass
        exists to close. If a genuine multi-stage sieve is ever wanted, it needs
        a new kernel ABI and its own separately governed config field; it must
        not be smuggled in by re-interpreting this key.
        """
        if not dataset_path:
            raise ValueError("dataset is mandatory in a stripe assignment")
        if residues is None:
            raise ValueError("residues are mandatory to compute residue_sha256")
        if dataset_sha256 is None:
            # [S172 P0.5] run-scoped, not per-call — see resolve_dataset_sha256.
            dataset_sha256 = resolve_dataset_sha256(dataset_path)
        if not dataset_sha256:
            raise ValueError("dataset_sha256 is mandatory in a stripe assignment")
        # Direction resolution — the §6.8 shared table, fail-closed on an unknown
        # phase. The parent resolves; the worker consumes.
        direction, _skip_mode = workflow_phase_semantics(phase)
        if direction == "forward":
            resolved_threshold = float(forward_threshold)
        elif direction == "reverse":
            resolved_threshold = float(reverse_threshold)
        else:  # pragma: no cover — workflow_phase_semantics already fails closed
            raise MinerMetadataError(
                f"unresolvable sieve direction {direction!r} for phase {phase!r}")
        payload = {
            "dataset": dataset_path,
            "dataset_sha256": dataset_sha256,
            "window_size": window_size,
            "sessions": sessions,
            "offset": offset,
            "residue_sha256": sha256_residues(residues),
            # ---------------------------------------------------------------
            # SINGLE THRESHOLD CHOKEPOINT (S172 D6). forward_threshold/
            # reverse_threshold reach the kernel ONLY through this payload,
            # direction-resolved per stripe via the §6.8 phase table. Optuna
            # flows through here today. The agent autonomy application path
            # (watcher_policies.json `parameter_application` -> a
            # `reduce_threshold` proposal, TODO_SELFPLAY_AND_LLM_AUTONOMY.md
            # Part B) is DECLARED but NOT BUILT; when it is built it MUST set
            # these same fields here. Do NOT add a second threshold path — a
            # bypass reintroduces the D6 disconnect and lets governance log
            # phantom threshold changes.
            #
            # INVARIANT NOTE (documentation, not enforcement): every threshold
            # source — Optuna today, WATCHER parameter application later, any
            # future tuner — must pass through this one chokepoint. There is no
            # second place in the miner where a sieve threshold may be chosen.
            # ---------------------------------------------------------------
            "min_match_threshold": resolved_threshold,   # constant kernels
            "phase2_threshold": resolved_threshold,      # hybrid kernels
        }
        # The contract asserted at construction (Beta §2): a contradictory pair
        # never leaves the parent. The worker re-validates on receipt.
        if payload["min_match_threshold"] != payload["phase2_threshold"]:
            raise MinerMetadataError(
                "min_match_threshold and phase2_threshold must be identical in a "
                f"D6 stripe payload (got {payload['min_match_threshold']!r} vs "
                f"{payload['phase2_threshold']!r}); the miner runs ONE kernel with "
                "ONE threshold per stripe — see the threshold contract above.")
        return payload

    # ----- real framed-TCP serve loop (Beta-required Phase-4 deliverable) ---
    def _resolve_node_config(self, worker_id: str, hostname: str,
                             node_allowlist) -> NodeConfig:
        """Decision A: the node record comes from CONFIG (the node allowlist),
        never a hostname parsed off the wire. Transfers (Stage 3) use its SSH
        address/user. Falls back to the configured spool_root when a node is not
        separately described (the loopback gate uses inline results, so the SSH
        fields are unused there)."""
        rec = None
        if isinstance(node_allowlist, dict):
            rec = node_allowlist.get(worker_id) or node_allowlist.get(hostname)
        if isinstance(rec, dict):
            return NodeConfig(
                hostname=hostname,
                spool_root=rec.get("spool_root", self.config.spool_root),
                ssh_address=rec.get("ssh_address", ""),
                ssh_user=rec.get("ssh_user", ""))
        return NodeConfig(hostname=hostname, spool_root=self.config.spool_root,
                          ssh_address="", ssh_user="")

    def serve_trial(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Single-trial framed-TCP coordinator server. Binds/listens, accepts
        worker connections over the Phase-2 MinerFramedSocket framing, and wires
        the already-verified handlers together: register_worker (Decision A) ->
        assign_stripes (Blocker 7) -> per-message accept_stripe_message (L1 fence)
        -> record/stage/finalize (Blockers 1/2/4, L8) / handle_stripe_failure
        (Blocker 3 matrix) / process_lease_expiry -> commit_trial or (on terminal
        failure, via the matrix) abort_trial (L3/L7). Runs one trial to a terminal
        state and returns a real result dict. NEVER raises NotImplementedError.

        Testability: pass a pre-bound+listening `context['listen_sock']` (bind
        port 0) so a harness can drive real loopback workers on an ephemeral port;
        otherwise binds config.miner_host:miner_port. The bound address is stashed
        on self.bound_addr."""
        run_id = context["run_id"]
        # [§4.3 ADMISSION LIVENESS — Beta Ruling 1] ADMISSION IS BOUNDED EVEN
        # THOUGH EXECUTION IS NOT. `serve_timeout` stays None by Beta's earlier
        # correction (a multi-billion-seed scan exceeds any wall clock), so before
        # this repair there was NO finite bound on "waiting for the fleet to show
        # up": fewer than expected_workers daemons meant the loop accepted
        # connections forever with no assignment, no dispatch, no error and no
        # timeout. This bounds ONLY the wait for admission, never the work. 180s
        # matches the existing PWC readiness window
        # (persistent_worker_coordinator.py:826 and :864,
        # `_tcp_wait_ready(..., timeout_s=180.0)`), so the two backends agree on
        # how long a fleet is allowed to take to come up.
        #
        # RESOLVED AND VALIDATED FIRST, before the dataset digest, the trial
        # context or the listening socket: a misconfigured admission window is a
        # liveness defect, so it must be refused before the run can start waiting.
        _admission_raw = context.get("worker_admission_timeout",
                                     DEFAULT_WORKER_ADMISSION_TIMEOUT)
        try:
            worker_admission_timeout = float(_admission_raw)
        except (TypeError, ValueError):
            # Notably None, which by analogy with serve_timeout=None would read as
            # "disable the bound" — the one meaning this knob must never carry.
            worker_admission_timeout = float("nan")
        # FAIL CLOSED on a value that would restore the defect. None/0/negative/inf
        # are exactly the shapes that turn the bounded admission wait back into the
        # silent hang this repair exists to remove, so they are refused HERE rather
        # than honoured into an unreachable failure matrix.
        if not (worker_admission_timeout > 0.0
                and math.isfinite(worker_admission_timeout)):
            raise ValueError(
                "worker_admission_timeout must be a POSITIVE FINITE number of "
                f"seconds (got {context.get('worker_admission_timeout')!r}) for "
                f"run {run_id!r}. A non-positive, absent-by-None or infinite "
                "admission window reinstates the §4.3 silent hang: the trial would "
                "wait for a fleet that never arrives with no assignment, no "
                "dispatch and no terminal failure. Bound the ADMISSION wait; "
                "execution stays unbounded via serve_timeout=None."
            )
        family_name = context.get("family_name") or context.get("prng_base") or ""
        phase = int(context.get("phase", context.get("workflow_phase", 1)))
        total_seeds = int(context["total_seeds"])
        residues = context["residues"]
        dataset_path = context["dataset_path"]

        # dataset_sha256 is coordinator-resolved ONCE and reused for every assign.
        # [S172 Phase 6-P0.5 §2.1] "ONCE" used to mean once per TRIAL — serve_trial
        # is entered once per Optuna trial, so this line re-hashed the file between
        # trials and a mid-study scrape silently split the study across two
        # datasets. It now resolves from the RUN-START FREEZE when this path is the
        # frozen one, so the identity is fixed for the whole run and a pointer that
        # moves mid-run cannot change it (requirement 7).
        dataset_sha256 = resolve_dataset_sha256(dataset_path)

        # D0 (Blocker 2 + REV4): persist the trial-GLOBAL immutable context ONCE per
        # run_id — BEFORE any window_size/offset coercion, stripe assignment, or
        # dispatch. build_trial_context_from_serve projects it with NO fallback
        # substitution: a missing mandatory field (prng_base / skip_min / skip_max /
        # window_size / offset / thresholds) fails closed HERE with MinerMetadataError,
        # never as a fabricated 1/0/family_name AND never as a raw int(None) TypeError
        # in the window-param coercions below (REV4: those coercions previously ran
        # first and crashed on an omitted value instead of failing closed cleanly).
        # dataset_sha256 (just computed) and residue_sha256 (the SAME sha256_residues
        # the assign payload uses) are COPIED, not recomputed. Every published manifest
        # is reconstructed from this row + the stripe's persisted phase/family_name, so
        # trial-global metadata is identical across the run and immutable after
        # creation. Compare-and-insert, so a restart's re-serve with an identical
        # context is idempotent and a conflicting one fails closed.
        trial_ctx = build_trial_context_from_serve(
            context, dataset_sha256, sha256_residues(residues))
        self.ledger.set_trial_context(run_id, trial_ctx)

        # Trial-global window params come from the VALIDATED projection (already
        # int-coerced and guaranteed non-None by the guard above) — NEVER re-fabricated
        # from raw context via int(context.get(..., 1/0)).
        window_size = trial_ctx["window_size"]
        sessions = trial_ctx["sessions"]
        offset = trial_ctx["offset"]
        # [ADMISSION BINDING] Still ONE binding of expected_workers, still never
        # reduced dynamically, still derived from the pool size the caller
        # requested — but the frozen execution set, when one exists, is now the
        # authority over that request instead of a second opinion beside it.
        # See `_execution_set_expected_workers` for the defect this closes. The
        # 180s admission window, serve_timeout=None and the Blocker-3 matrix are
        # untouched: what changes is the NUMBER the window waits for, not the
        # window, and not what happens when it expires.
        # NOTE exactly ONE `context.get("worker_pool_size", ...)` read, here.
        # The §4.3 liveness gate counts worker_pool_size CODE sites to prove its
        # unit semantics did not change; a second read for logging would move
        # that count. The requested value is echoed by
        # `execution_set.admission_expectation`, which has it in hand anyway.
        expected_workers, _admission_source = _execution_set_expected_workers(
            int(context.get("worker_pool_size", 1) or 1))
        logger.info("[ADMISSION] run %s: expected_workers=%d (source=%s)",
                    run_id, expected_workers, _admission_source)
        node_allowlist = context.get("node_allowlist")
        poll = float(context.get("serve_poll", 0.1))
        # Production-timeout correction (Beta): the serve/trial timeout is UNBOUNDED
        # by default (a real multi-billion-seed scan far exceeds any fixed 30s), and
        # enforced ONLY when explicitly configured. A None timeout means run until a
        # terminal state; the gates inject their own short timeout.
        _timeout_raw = context.get("serve_timeout", None)
        trial_timeout = None if _timeout_raw is None else float(_timeout_raw)
        # Defect 6 (C3): a per-connection read deadline. A connection that connects
        # but never completes a full frame (silent, or header + partial body) is
        # dropped after this many seconds so it can never wedge registration /
        # dispatch / the timeout loop. Registered workers are exempt (they may idle).
        read_deadline = float(context.get("serve_read_deadline", 15.0))

        listen_sock = context.get("listen_sock")
        own_listen = listen_sock is None
        if own_listen:
            listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_sock.bind((self.config.miner_host, self.config.miner_port))
            listen_sock.listen(max(8, expected_workers + 2))
        listen_sock.settimeout(poll)
        self.bound_addr = listen_sock.getsockname()

        fs_by_sock: Dict[Any, MinerFramedSocket] = {}     # rawsock -> framed
        worker_by_sock: Dict[Any, str] = {}               # rawsock -> worker_id
        wconn_by_worker: Dict[str, WorkerConnection] = {}
        fs_by_worker: Dict[str, MinerFramedSocket] = {}
        registered: List[str] = []
        dispatched: set = set()
        # Defect 6 (C3): each accepted connection gets its OWN reader thread doing
        # blocking frame reads into a bounded inbound queue; the serve loop only ever
        # DRAINS that queue (never a blocking recv on the dispatch thread), so one
        # slow/silent/partial connection cannot stall the others, the assignment
        # logic, or the timeout loop. conn_meta tracks liveness for the read deadline.
        import queue as _queue
        inbound: "_queue.Queue" = _queue.Queue(maxsize=1024)
        reader_stop = threading.Event()
        reader_threads: Dict[Any, threading.Thread] = {}
        conn_meta: Dict[Any, Dict[str, Any]] = {}   # rawsock -> {connect, registered}
        # Defect 6: the workflow's family/phase STAGES. `family_name`/`phase`
        # overrides (used by the gate) collapse to one stage; otherwise the trial
        # runs all stages resolved from prng_base + test_both_modes.
        workflow_stages = context.get("workflow_stages") or [(family_name, phase)]
        stage_idx = 0
        stage_assigned = False
        # [§4.3] Admission-window state. `admission_stage_idx` is the STAGE the
        # current window belongs to, and is the ONLY thing that can re-arm it —
        # which is how "reset only at a genuine new-stage boundary" is enforced
        # structurally rather than by convention. Nothing in the connect/register/
        # drop/quarantine paths touches these two names, so a worker connecting,
        # disconnecting or flapping cannot extend the window.
        admission_stage_idx: Optional[int] = None
        admission_started_at: Optional[float] = None
        # [S172 D6] set ONLY by the fail-closed provenance gate below.
        # Default False: a run that never reached the gate is NOT validated,
        # and downstream refuses it. Absence is never neutral.
        provenance_validated = False
        start = time.time()
        # [S172-BP §4] the trial clock the staging-throughput series is measured
        # against (jobs completed / elapsed).
        with self._bp_lock:
            self._bp["trial_started_at"] = start

        def _eligible():
            return [w for w in wconn_by_worker.values() if not w.quarantined]

        def _terminal() -> bool:
            trial = self.ledger.get_trial(run_id)
            return trial is not None and trial["state"] in ("committed", "aborted")

        def _stage_prefix(i):
            return f"{run_id}__st{i}"

        try:
            while not _terminal():
                now = time.time()
                if trial_timeout is not None and now - start > trial_timeout:
                    # Defect 5: terminal abort routed OFF the dispatch loop.
                    self.fail_trial(run_id, reason="serve_trial timeout")
                    continue

                # [S172-BP §1.5] THE BOUNDED CAPACITY TIMEOUT — the ONE permitted
                # trial-terminal path for a capacity wait. Called DIRECTLY, never
                # through handle_stripe_failure / the matrix, because the terminal
                # reason must be a coordinator/infrastructure condition (Beta §1).
                # Measured from the OLDEST currently-paused connection's entry
                # time, so a long queue of short pauses cannot trip it.
                if self.staging_capacity_timeout_expired(now):
                    _reason = self.staging_capacity_timeout_reason(now)
                    logger.error("[S172-BP] capacity_timeout run=%s %s",
                                 run_id, _reason)
                    with self._bp_lock:
                        self._bp["capacity_timeout_terminations"] += 1
                    self.fail_trial(run_id, reason=_reason, now=now)
                    continue

                # --- accept new connections (no blocking read on the loop) ---
                try:
                    readable, _, _ = select.select([listen_sock], [], [], poll)
                except (OSError, ValueError):
                    readable = []
                if readable:
                    try:
                        csock, _ = listen_sock.accept()
                    except (socket.timeout, OSError):
                        csock = None
                    if csock is not None:
                        cfs = MinerFramedSocket(csock)
                        fs_by_sock[csock] = cfs
                        conn_meta[csock] = {"connect": now, "registered": False}
                        th = threading.Thread(
                            target=self._conn_reader_loop,
                            # [S172-BP §1.2] worker_by_sock lets the reader name
                            # the identity it is pausing, which is what the §1.4
                            # lease exemption keys on. Read-only from the reader.
                            args=(cfs, csock, inbound, reader_stop, worker_by_sock),
                            name="miner-conn-reader", daemon=True)
                        reader_threads[csock] = th
                        th.start()

                # --- drain complete frames from the bounded inbound queue ---
                # [S172-BP §4] inbound-queue occupancy high-water, sampled at the
                # moment of maximum backlog (before the drain).
                self.note_inbound_occupancy(inbound.qsize())
                drained = 0
                while drained < 256:
                    try:
                        # [S172-BP AMENDMENT F1-R2a] FOUR fields: the credit token
                        # rides with the envelope. `None` for every ordinary
                        # message and for every 'eof'.
                        kind, rawsock, msg, credit_id = inbound.get(
                            timeout=poll if drained == 0 else 0)
                    except _queue.Empty:
                        break
                    drained += 1
                    if kind == "eof":
                        # [S172-BP AMENDMENT F1-R] disposition (iv): the connection
                        # terminated. `inbound` is FIFO, so a credited envelope that
                        # was actually delivered has ALREADY been dispatched (and
                        # cleared) above this eof — this therefore only fires for
                        # the undispatched-and-gone case, where nothing else will
                        # ever dispose of the reservation.
                        self._release_resume_credit(rawsock, delivered=False,
                                                    disposition="eof")
                        # Defect 3 (C5): a disconnected worker is evicted from the
                        # eligible pool (wconn_by_worker / connections / registered).
                        self._drop_conn(rawsock, fs_by_sock, worker_by_sock,
                                        fs_by_worker, wconn_by_worker, registered)
                        conn_meta.pop(rawsock, None)
                        reader_threads.pop(rawsock, None)
                        continue
                    if rawsock not in fs_by_sock:
                        # [S172-BP AMENDMENT F1-R] disposition (iv) again: the
                        # socket was already reaped, so this envelope is discarded
                        # here rather than dispatched. Same fact, earlier line.
                        if getattr(msg, "message_type", None) == "sub_stripe_result":
                            self._release_resume_credit(
                                rawsock, delivered=True, disposition="conn_dropped")
                        continue   # connection already dropped (reaped/closed)
                    if msg.message_type == "register":
                        status = self._serve_register(
                            msg, rawsock, node_allowlist, fs_by_sock, worker_by_sock,
                            wconn_by_worker, fs_by_worker, registered)
                        meta = conn_meta.get(rawsock)
                        if status == "ok":
                            if meta is not None:
                                meta["registered"] = True
                        elif status == "reject_dup_worker":
                            # a second socket for an already-live worker_id — drop it.
                            # It was never bound, so the guard leaves the ORIGINAL
                            # worker's identity intact (Defect 3 C4/C5).
                            self._drop_conn(rawsock, fs_by_sock, worker_by_sock,
                                            fs_by_worker, wconn_by_worker, registered)
                            conn_meta.pop(rawsock, None)
                            reader_threads.pop(rawsock, None)
                        # reject_rebind: leave the socket bound to its ORIGINAL id
                        # (already registered); ignore the stray REGISTER frame.
                    else:
                        # Defect 3: the RECEIVING socket's bound identity is
                        # authoritative — pass it, never trust msg.worker_id alone.
                        # [S172-BP AMENDMENT F1-R] through the disposition-bounded
                        # seam: `_serve_dispatch` runs unchanged, and the ingress
                        # reservation is released only AFTER it returns.
                        self.dispatch_inbound_result(
                            msg, rawsock, run_id, worker_by_sock.get(rawsock),
                            wconn_by_worker, _eligible, credit_id)

                # --- read deadline: drop unregistered connections that never
                # completed a frame (silent or partial), so they cannot wedge the
                # loop or hold the timeout hostage (Defect 6 C3) ---
                for rawsock, meta in list(conn_meta.items()):
                    if meta["registered"]:
                        continue
                    if now - meta["connect"] > read_deadline:
                        logger.warning(
                            "dropping connection that never completed a frame "
                            "within %.1fs read deadline (Defect 6)", read_deadline)
                        self._drop_conn(rawsock, fs_by_sock, worker_by_sock,
                                        fs_by_worker, wconn_by_worker, registered)
                        conn_meta.pop(rawsock, None)
                        reader_threads.pop(rawsock, None)

                # --- staged assignment of the workflow (Defect 6 multi-family) ---
                # [§4.3 ADMISSION LIVENESS REPAIR — Beta Ruling 1]
                # The pre-repair guard here was
                #     if len(eligible) >= expected_workers and stage_idx < len(...):
                # which put assign_stripes, _dispatch_pending, process_lease_expiry
                # AND the stage advance behind ONE threshold test. With
                # serve_timeout=None that made the Blocker-3 failure matrix
                # unreachable in exactly the situation it exists for: a worker loss
                # that dropped the pool below expected_workers silently stopped
                # lease expiry from being processed, so the dead worker's stripes
                # stayed `claimed` with an expired lease nobody looked at and the
                # trial neither completed nor failed (FLEET_STATE_REQUIREMENTS_v1
                # §4.3).
                #
                # The threshold test is NOT removed — it is MOVED to where it
                # belongs, and given a bound:
                #   * ADMISSION  — reaching expected_workers is a precondition for
                #     ASSIGNING a stage, and is now bounded by
                #     worker_admission_timeout. Failure to reach it is an explicit
                #     fail_trial, not a hang.
                #   * MAINTENANCE — once a stage IS assigned, dispatch, lease
                #     expiry and completion evaluation run unconditionally, so a
                #     mid-run loss reaches the matrix and the matrix decides
                #     (constant phase -> immediate trial failure; hybrid -> the one
                #     reassignment), and finished work still commits.
                # expected_workers is NOT reduced dynamically, worker_pool_size
                # keeps its current meaning, and the matrix itself is untouched.
                eligible = _eligible()
                if stage_idx < len(workflow_stages):
                    fam, ph = workflow_stages[stage_idx]
                    if not stage_assigned:
                        # ---- ADMISSION (bounded) -------------------------------
                        # Arm the window for THIS stage. The identity test is the
                        # enforcement of "reset only at a genuine new-stage
                        # boundary": the window can only be re-armed when stage_idx
                        # actually changes, so churn below the threshold cannot
                        # extend it.
                        if admission_stage_idx != stage_idx:
                            admission_stage_idx = stage_idx
                            admission_started_at = now
                        if len(eligible) < expected_workers:
                            waited = now - admission_started_at
                            if waited > worker_admission_timeout:
                                # Terminal, explicit, and diagnosable: run id,
                                # stage, expected count and eligible count, so
                                # WATCHER gets something to react to instead of an
                                # indefinitely silent trial. fail_trial routes the
                                # abort off the dispatch loop and marks the trial
                                # aborted, which leaves provenance_validated False
                                # -> the integration adapter refuses ingress and
                                # the failure propagates terminally.
                                self.fail_trial(
                                    run_id,
                                    reason=(
                                        f"worker admission timeout: run {run_id!r} "
                                        f"stage {stage_idx} (family {fam!r}, phase "
                                        f"{ph}) expected {expected_workers} eligible "
                                        f"worker(s), {len(eligible)} admitted after "
                                        f"{waited:.1f}s "
                                        f"(worker_admission_timeout="
                                        f"{worker_admission_timeout:.1f}s)"),
                                    now=now)
                            # Short pool and the window is still open (or the trial
                            # is now terminal): this stage is NOT assigned, so there
                            # is nothing to dispatch and no lease to maintain. Keep
                            # accepting registrations.
                            continue
                        # Defect 2 (C4): if NO eligible worker supports this stage's
                        # variant, FAIL THE TRIAL EXPLICITLY — never strand stripes
                        # `pending` forever (which, with the now-unbounded timeout,
                        # would hang the trial).
                        if not any(self.can_assign_variant(w, fam) for w in eligible):
                            self.fail_trial(
                                run_id,
                                reason=f"no eligible worker supports variant {fam!r}")
                            continue
                        _stage_assignments = self.assign_stripes(
                            run_id, fam, ph, total_seeds, eligible,
                            stripe_prefix=_stage_prefix(stage_idx))
                        # [S172-BP §2] SIZE THE DEFERRED BOUND FROM THE RESOLVED
                        # EXECUTION SET — at stage setup, from real geometry, phase
                        # and per-worker caps. Never a hand-maintained constant.
                        # The CONSERVATIVE bound is used because capacity has to be
                        # sized for any assignment the round-robin could still
                        # produce; the spans come from the assignments themselves,
                        # so a short final macro-stripe is not rounded up to
                        # miner_stripe_size.
                        #
                        # [S172-BP AMENDMENT F5] A SIZING FAILURE FAILS CLOSED.
                        # The pre-amendment handler swallowed the exception and let
                        # `staging_deferred_bound()` fall back to
                        # `_derive_bound_from_current_state` — ONE macro-stripe,
                        # phase 1 — which answers a different question and can be
                        # MATERIALLY SMALLER than the stage derivation that just
                        # failed (multi-stripe stages, hybrid caps). That silently
                        # re-armed the very undersized-queue condition this work
                        # exists to remove. The inputs are materialized ONCE at
                        # entry and any derivation exception terminates the trial
                        # DIRECTLY — never the matrix, never a smaller implicit
                        # bound, and BEFORE any result traffic for this stage
                        # (`_dispatch_pending` is below the `continue`).
                        try:
                            _stripe_spans = [int(a["seed_count"])
                                             for a in _stage_assignments]
                            _eligible_records = list(eligible)
                            _exact_rows = [
                                {"stripe_span": a["seed_count"],
                                 "effective_cap": a["effective_cap"],
                                 "phase": ph, "family_name": fam}
                                for a in _stage_assignments
                                if a.get("effective_cap")]
                            self.derive_staging_deferred_bound(
                                _stripe_spans, _eligible_records, ph,
                                family_name=fam)
                            # The EXACT bound for the assignment that was actually
                            # made — logged beside the conservative one so the
                            # 116-vs-136 distinction is visible in every run log,
                            # not only in the tests.
                            logger.info(
                                "[S172-BP] burst_exact stage=%d family=%s phase=%s "
                                "exact=%d conservative=%d",
                                stage_idx, fam, ph,
                                staging_burst_bound_exact(_exact_rows),
                                self._derived_bound_detail.get(
                                    "burst_bound_conservative"))
                        except Exception as _sizing_exc:      # noqa: BLE001
                            logger.exception(
                                "[S172-BP] STAGING SIZING FAILED CLOSED at stage "
                                "%d — run=%s family=%s phase=%s assignments=%d "
                                "eligible=%d spans=%r caps=%r: the staging "
                                "deferred bound could not be derived, so the "
                                "trial is terminated rather than run on the "
                                "on-demand (one-macro-stripe, phase-1) "
                                "derivation, which can be materially smaller",
                                stage_idx, run_id, fam, ph,
                                len(_stage_assignments), len(eligible),
                                [a.get("seed_count") for a in _stage_assignments],
                                self._central_caps())
                            self.fail_trial(
                                run_id,
                                reason=(
                                    f"coordinator_staging_sizing: could not derive "
                                    f"the staging deferred bound for stage "
                                    f"{stage_idx} — "
                                    f"{type(_sizing_exc).__name__}: "
                                    f"{_sizing_exc}"),
                                now=now)
                            continue
                        stage_assigned = True
                    # ---- MAINTENANCE (unbounded, threshold-free) ---------------
                    # Everything below runs for an ASSIGNED stage regardless of the
                    # current eligible count. `eligible` is still passed to
                    # process_lease_expiry because the matrix needs the CURRENT pool
                    # to pick a reassignment target — a shrunken (even empty) pool is
                    # a legitimate input that the matrix already handles (hybrid with
                    # no alternate -> fail_trial). It is no longer a gate on whether
                    # the matrix runs at all.
                    self._dispatch_pending(
                        run_id, fam, ph, fs_by_worker, dispatched, dataset_path,
                        dataset_sha256, window_size, sessions, offset, residues,
                        context.get("trial_number", -1),
                        trial_ctx["forward_threshold"],
                        trial_ctx["reverse_threshold"])
                    self.process_lease_expiry(run_id, eligible)
                    # advance to the next stage once THIS stage's stripes are done
                    sp = _stage_prefix(stage_idx) + "_s"
                    stage_stripes = [s for s in self.ledger.all_stripes(run_id)
                                     if s["stripe_id"].startswith(sp)]
                    if stage_stripes and all(s["state"] == ST_DONE for s in stage_stripes):
                        stage_idx += 1
                        stage_assigned = False
                        if stage_idx >= len(workflow_stages):
                            # [S172 D6, Beta commit ruling] FAIL-CLOSED THRESHOLD
                            # PROVENANCE GATE. Deliberately placed HERE — before
                            # commit_trial, therefore before Phase-5 assembly,
                            # before candidate ingress, before accumulator
                            # mutation and before finalize_run. A trial whose
                            # kernel filter is unproven must never reach
                            # certification, so this runs while refusing is still
                            # possible.
                            try:
                                self.validate_threshold_provenance(run_id)
                            except ThresholdProvenanceError as primary:
                                # PRIMARY-EXCEPTION DISCIPLINE (D5 REV2 §7): the
                                # violation is the diagnostic that matters. The
                                # abort below may itself fail (cleanup, transport,
                                # a sink error); it must never replace or obscure
                                # the original. Nothing is chained onto `primary`
                                # and nothing is raised in its place.
                                logger.error(
                                    "threshold provenance violation for run %s — "
                                    "aborting the trial, NOT committing: %s",
                                    run_id, primary)
                                try:
                                    # Best-effort audit artifact FIRST, so the
                                    # evidence survives even if the abort fails.
                                    self._write_threshold_provenance(
                                        self.threshold_provenance(
                                            run_id, trial_ctx, validated=False))
                                except Exception:       # noqa: BLE001
                                    logger.exception(
                                        "could not persist the provenance record "
                                        "for the violating run (primary violation "
                                        "is preserved)")
                                try:
                                    self.abort_trial(
                                        run_id,
                                        reason=f"threshold provenance violation: "
                                               f"{primary}")
                                except Exception:       # noqa: BLE001
                                    logger.exception(
                                        "abort/cleanup ALSO failed while handling a "
                                        "threshold provenance violation; the "
                                        "PRIMARY violation below is preserved and "
                                        "re-raised unchanged")
                                raise primary
                            provenance_validated = True
                            try:
                                self.commit_trial(run_id)
                            except TrialAborted:
                                pass
        finally:
            for wid, cfs in list(fs_by_worker.items()):
                try:
                    cfs.send_msg(MinerShutdownMessage(worker_id=wid))
                except Exception:
                    pass
            # Defect 6 (C3): stop every per-connection reader thread and shutdown +
            # close its socket (shutdown wakes a reader blocked in recv AND sends the
            # peer a prompt FIN), then join briefly.
            reader_stop.set()
            for cfs in fs_by_sock.values():
                try:
                    cfs.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    cfs.close()
                except Exception:
                    pass
            for th in list(reader_threads.values()):
                th.join(timeout=1.0)
            if own_listen:
                try:
                    listen_sock.close()
                except Exception:
                    pass
            # Drain in-flight staging jobs FIRST (they may route through the matrix
            # -> submit_abort), then the abort-cleanup executor.
            if self._staging_executor is not None:
                self._staging_executor.shutdown(wait=True)
                self._staging_executor = None
            if self._cleanup_executor is not None:
                self._cleanup_executor.shutdown(wait=True)
                self._cleanup_executor = None
            # Defect 4 (C4): account for any tracked orphan fetch threads on shutdown
            # (prune finished, briefly join, report residual live count).
            residual = self.account_orphan_fetches(join_timeout=0.5)
            if residual:
                logger.warning("%d orphan fetch thread(s) still live at shutdown "
                               "(hung transport)", residual)
            # [S172-BP AMENDMENT F2] trial-terminal cleanup: no resume grace
            # outlives its trial. Anything still here bridges a renewal for a
            # trial that no longer exists.
            _dropped_grace = self.clear_all_capacity_resume_grace()
            if _dropped_grace:
                logger.info("[S172-BP] resume_grace_cleared_at_terminal count=%d "
                            "run=%s", _dropped_grace, run_id)
            # [S172-BP AMENDMENT F1-R] disposition (iv), trial-terminal arm. Any
            # reservation still outstanding here belongs to an envelope that will
            # never be dispatched, on a trial that no longer exists.
            if self.clear_any_resume_credit(disposition="trial_terminal"):
                logger.info("[S172-BP] resume_credit_cleared_at_terminal run=%s",
                            run_id)

        trial = self.ledger.get_trial(run_id)
        stripes = {s["stripe_id"]: {
            "state": s["state"], "phase_degraded": bool(s["phase_degraded"]),
            "claimed_by": s["claimed_by"], "current_attempt": s["current_attempt"],
            "survivors_total": s["survivors_total"],
        } for s in self.ledger.all_stripes(run_id)}
        # [S172 D6] The three-leg threshold audit record. Returned to the caller
        # AND written next to the staged output, so the question "what did the
        # kernel actually filter at?" is answerable from the run's own artifacts
        # rather than from the config that was requested.
        provenance = self.threshold_provenance(
            run_id, trial_ctx, validated=provenance_validated)
        self._write_threshold_provenance(provenance)
        # [S172-BP §4] the trial-terminal metrics summary — emitted for EVERY
        # terminal state (committed or aborted), and returned on the result so it
        # is observable on the run rather than only in the log.
        bp_metrics = self.log_staging_backpressure_summary(run_id)
        return {
            "run_id": run_id,
            "state": trial["state"] if trial else "unknown",
            "committed": bool(trial and trial["state"] == "committed"),
            "workers_registered": list(registered),
            "stripes": stripes,
            "manifests": list(self.enqueued),
            "bound_addr": self.bound_addr,
            "threshold_provenance": provenance,
            "staging_backpressure": bp_metrics,
        }

    def _write_threshold_provenance(self, provenance: Dict[str, Any]) -> Optional[str]:
        """Persist the threshold audit record beside the staged output. Best
        effort: an unwritable staging dir must never fail a completed trial, but
        the failure is logged rather than swallowed silently."""
        staging = getattr(self.config, "staging_dir", None)
        if not staging:
            return None
        path = os.path.join(staging, "threshold_provenance.json")
        try:
            os.makedirs(staging, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(provenance, fh, indent=2, sort_keys=True)
            return path
        except OSError as e:
            logger.warning("could not write threshold provenance to %s: %s", path, e)
            return None

    def _conn_reader_loop(self, cfs, rawsock, inbound, reader_stop,
                          worker_by_sock=None) -> None:
        """Defect 6 (C3): per-connection reader. Does BLOCKING full-frame reads (so
        a legitimately slow but complete frame is never corrupted) and feeds each
        complete message to the bounded coordinator queue. It touches NO ledger
        state — all dispatch stays single-threaded on the serve loop. On EOF / a
        malformed or oversized frame / socket error (including the serve loop
        closing the socket at shutdown or on the read deadline), it enqueues one
        'eof' and exits. A full queue (coordinator overloaded) also drops the
        connection rather than growing memory without bound.

        [S172-BP §1.1 — WHERE THE PAUSE LIVES, AND WHY]
        The capacity gate is HERE, per connection, and not in `enqueue_staging`.
        By the time `enqueue_staging` discovers saturation the payload has ALREADY
        been decoded into coordinator RAM, so the only remaining choices are
        "retain it" or "throw it away" — which is how a coordinator capacity wait
        ended up charged to a worker's stripe as a fault. Gating at the reader
        keeps every SUBSEQUENT payload on the wire / at the worker, which is the
        property C4 §1c named correct: the worker's `_sendall`
        (range_miner_worker.py:1120-1126) is a blocking loop with NO socket
        timeout, so a full TCP buffer simply parks that worker's mining thread
        mid-`_send`, harmlessly, until the coordinator reads again.

        ONLY `sub_stripe_result` is gated. `register` / `heartbeat` /
        `stripe_complete` / `stripe_error` pass through when THEY are the decoded
        frame. TCP is ordered, so frames queued BEHIND a held result stay on the
        wire — that is the point of the design, and it is precisely why the §1.4
        lease exemption has to exist.

        At most ONE already-decoded envelope is held per connection (Beta:
        *"retaining at most one bounded pending envelope per connection is
        acceptable"*), and it is held in this thread's local — never in
        `_deferred`, never in a second queue.
        """
        import queue as _queue
        pending_envelope = None
        # [S172-BP AMENDMENT F1-R] Does the reservation this reader was woken on
        # still belong to this thread at exit? Reset at every PAUSE ENTRY (a new
        # wake cycle, a new reservation) and set once the held envelope reaches
        # `inbound` — from that instant the clear belongs to the serve loop's
        # disposition, never to this thread.
        credit_delivered = False
        # [S172-BP AMENDMENT F1-R2b] The TOKEN of a reservation this thread has
        # already handed to the serve loop, and therefore the thing the PRE-DECODE
        # BARRIER at the top of the loop waits on. None whenever this connection
        # owes the serve loop nothing.
        delivered_credit_id = None
        try:
            while not reader_stop.is_set():
                # ---- PRE-DECODE BARRIER (Beta F1-R2b §4.2) -------------------
                # BEFORE `recv_msg`, never after. While our credited envelope is
                # still undisposed, the next frame must stay ON THE WIRE: decoding
                # it here would give this ONE connection two decoded envelopes at
                # once (the credited one in `inbound`, this one in our local),
                # which is exactly the bound the §1.2 resume margin is derived
                # from. It also re-serves the §4-tail rule — one result per
                # reservation — from the correct side of the decode.
                #
                # Heartbeats and completions queued behind the result are held on
                # the wire too, and Beta §4.2 ACCEPTS that: the interval is short,
                # the RATIFIED resume grace (F2) already covers the lease across
                # exactly this window, and TCP ordering makes selective bypass
                # impossible in any case.
                if delivered_credit_id is not None:
                    if not self._await_exact_credit_clear(delivered_credit_id,
                                                          reader_stop):
                        # shutdown or the latched §1.5 capacity timeout. Nothing is
                        # held here — the credited envelope is already delivered —
                        # so this exit discards nothing and routes nothing.
                        break
                    delivered_credit_id = None

                # Ordinary frames carry NO token. Only the envelope a wake was
                # granted for is stamped, on the resume path below.
                put_credit_id = None
                try:
                    msg = cfs.recv_msg()
                except (ConnectionError, ValueError, OSError):
                    break
                except Exception:  # noqa: BLE001 — any decode error drops the conn
                    break

                # [S172-BP AMENDMENT F4] REGISTERED WORKERS ONLY. The pause
                # condition tested message type + capacity but not IDENTITY, so an
                # unregistered socket sending a well-formed `sub_stripe_result`
                # under saturation acquired pause state (`worker_id=None`),
                # consumed the one-envelope allowance, joined the oldest-pause
                # clock that §1.5 measures, and was held BEFORE the serve loop's
                # identity rejection could ever see it. `worker_by_sock` is written
                # only at registration (`_serve_register`), so it IS the
                # bound-worker predicate. An unbound result under saturation is NOT
                # paused and NOT held: it flows to `inbound` unchanged and dies in
                # the EXISTING serve-loop identity/protocol rejection, exactly as
                # it did pre-amendment. No new rejection logic is added here — the
                # point is to stop intercepting the message before the existing
                # guard.
                bound_worker_id = (worker_by_sock.get(rawsock)
                                   if worker_by_sock is not None else None)
                gated_result = (getattr(msg, "message_type", None)
                                == "sub_stripe_result"
                                and bound_worker_id is not None)

                # [S172-BP AMENDMENT F1-R2b] The round-2 POST-decode §4-tail gate
                # stood here (`holds_resume_credit` + `_await_resume_credit_clear`).
                # It is DELETED, not moved-and-kept: waiting here is what let the
                # connection hold a second decoded envelope. The barrier at the top
                # of the loop enforces the same one-result-per-reservation rule
                # before anything is decoded.

                if gated_result and not self.staging_can_accept():
                    # ---- PAUSE (per connection, never global) ----------------
                    credit_delivered = False
                    delivered_credit_id = None
                    pending_envelope = msg
                    resume_event = self.register_paused_connection(
                        rawsock, bound_worker_id)
                    released = False
                    while not reader_stop.is_set():
                        if resume_event.wait(0.05):
                            released = True
                            break
                        if self.staging_capacity_timeout_expired():
                            # §1.5: the trial is (being) terminated by the serve
                            # loop with a coordinator/infrastructure reason. This
                            # reader observes it, DISCARDS the held envelope and
                            # exits — it never routes anything to the matrix.
                            break
                        # Defensive re-check, [S172-BP AMENDMENT F1] now a
                        # HEAD-ONLY SELF-GRANT. A capacity release that happened
                        # between our gate read and the registration would
                        # otherwise leave this reader waiting for an event nobody
                        # will set again — the decode race the documented resume
                        # margin (§1.2) covers, unchanged and still the final
                        # backstop. The bare `staging_can_accept()` escape this
                        # replaces let EVERY paused reader self-release on ONE
                        # observation; `_try_self_resume` succeeds only for the
                        # FIFO-oldest paused connection, only when no grant is in
                        # flight, and TAKES the credit itself.
                        if self._try_self_resume(rawsock):
                            released = True
                            break
                    self.deregister_paused_connection(
                        rawsock,
                        reason="resume" if released else "pause_aborted")
                    if not released:
                        pending_envelope = None
                        break
                    # [S172-BP AMENDMENT F1-R2a] READ BACK OUR OWN TOKEN. It was
                    # minted into this connection's pause record before the event
                    # was set, so the credited envelope can carry it to the serve
                    # loop and be disposed of by EXACT identity there. None here
                    # means the reservation was already force-cleared (trial
                    # terminal, say) — the envelope then carries no token and
                    # clears nothing, which is correct: there is nothing left to
                    # clear.
                    put_credit_id = self.resume_credit_id_for(rawsock)
                    # ---- RESUME: deliver the held envelope, exactly once ------
                    # It was NEVER dispatched while paused: record_substripe_result
                    # runs only when the serve loop processes it AFTER resume, so
                    # the existing dedup insert and the existing L1
                    # accept_stripe_message fence govern it unchanged. No second
                    # dedup layer is added (§1.3). If the attempt was superseded or
                    # cancelled while paused (staging_generation moved), that fence
                    # drops it here — which is correct, and is Beta gate 7.
                    msg = pending_envelope
                    pending_envelope = None

                try:
                    # [S172-BP AMENDMENT F1-R2a] THE TOKEN RIDES ON THE ENVELOPE.
                    # One producer, one place the stamp can be forgotten: ordinary
                    # frames carry `None` (reset at the top of every iteration),
                    # the credited envelope carries the token read back above.
                    inbound.put(("msg", rawsock, msg, put_credit_id), timeout=1.0)
                except _queue.Full:
                    break
                if put_credit_id is not None:
                    # From here the barrier owes this token a disposition before
                    # this connection may decode anything else.
                    delivered_credit_id = put_credit_id
                # [S172-BP AMENDMENT F1-R] THE RESERVATION RIDES WITH THE ENVELOPE.
                # Round 1 called `_release_resume_credit(..., delivered=True)` HERE.
                # That was ingress, not consumption: the staging slot this wake was
                # granted on is still physically free until the serve loop dispatches
                # this envelope into `enqueue_staging`, so clearing here let the next
                # FIFO head take a second wake on the SAME slot. The clear now
                # belongs to the serve loop's disposition (F1-R §4 i-iv); all this
                # thread records is that the hand-off has happened.
                credit_delivered = True
        finally:
            # A held envelope belongs to a trial that is terminal or a connection
            # that is going away; dropping it is the documented disposition (§1.5).
            pending_envelope = None
            # ORDER IS LOAD-BEARING: deregister FIRST, then clear the credit. A
            # grant can only target a connection that is still in
            # `_paused_connections`, so once this conn has left the registry no
            # grant can land on it — whereas clearing first would leave a window in
            # which a grant lands on a record about to be removed and the credit is
            # never cleared by anyone, wedging the whole paused fleet.
            #
            # [S172-BP AMENDMENT F1-R] AND THE CLEAR IS NOW CONDITIONAL. A wake that
            # delivered NOTHING must still not reserve the observation forever — but
            # a wake that DID deliver has handed its reservation to the serve loop,
            # and clearing it here would reopen the exact window F1-R closes (the
            # envelope is in `inbound`, the slot is still free, and the next FIFO
            # head would wake on it). For a delivered wake the disposition paths own
            # the clear: dispatch, eof, already-dropped socket, or trial-terminal.
            self.deregister_paused_connection(rawsock, reason="reader_exit")
            if not credit_delivered:
                self._release_resume_credit(rawsock, delivered=False,
                                            disposition="reader_exit_undelivered")
        if not reader_stop.is_set():
            try:
                inbound.put(("eof", rawsock, None, None), timeout=0.5)
            except Exception:  # noqa: BLE001
                pass

    def _drop_conn(self, rawsock, fs_by_sock, worker_by_sock, fs_by_worker,
                   wconn_by_worker=None, registered=None) -> None:
        """Drop a connection and EVICT its worker identity from every structure the
        eligible pool is built from (Defect 3 C5): fs_by_worker, wconn_by_worker,
        self.connections, registered — so a worker whose socket is gone is never
        handed NEW stripes. Identity is evicted ONLY if THIS dropped socket is the
        one currently bound to the worker_id, so a fenced replacement that legitimately
        rebound the same worker_id to a DIFFERENT live socket is NOT evicted."""
        fs = fs_by_sock.pop(rawsock, None)
        wid = worker_by_sock.pop(rawsock, None)
        if wid is not None:
            # Only evict the worker_id's identity if the mapping still points at THIS
            # socket (guards the fenced-replacement case from Defect 3 C4).
            if fs is None or fs_by_worker.get(wid) is fs:
                fs_by_worker.pop(wid, None)
                if wconn_by_worker is not None:
                    wconn_by_worker.pop(wid, None)
                self.connections.pop(wid, None)
                if registered is not None and wid in registered:
                    registered.remove(wid)
                # [S172-BP AMENDMENT F2] No grace outlives its connection. The
                # grace bridges a renewal that is in flight ON THIS CONNECTION;
                # once the connection is gone there is nothing in flight, and the
                # worker's silence is genuine again.
                self.clear_capacity_resume_grace(wid)
        # Defect 6 (C3): shutdown BEFORE close so a reader thread blocked in recv on
        # this socket is woken (recv returns EOF) AND the peer promptly sees the FIN
        # — a bare close() on a socket with a concurrent blocked recv may defer both.
        try:
            rawsock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if fs is not None:
            try:
                fs.close()
            except Exception:
                pass

    def _serve_register(self, msg, rawsock, node_allowlist, fs_by_sock,
                        worker_by_sock, wconn_by_worker, fs_by_worker,
                        registered) -> str:
        """Bind ONE worker to this socket. Returns a status:
          "ok"               — registered (or a benign re-send of the SAME id)
          "reject_rebind"    — a REGISTER arrived on an already-bound socket claiming
                               a DIFFERENT id; the socket stays bound to the original
          "reject_dup_worker"— this worker_id is already live on ANOTHER socket

        Defect 3 (C4): a socket registers exactly once, and at most one live socket
        maps to any worker_id — no `[A,B]` rebind, no two sockets sharing an id."""
        already = worker_by_sock.get(rawsock)
        if already is not None:
            if already == msg.worker_id:
                return "ok"   # idempotent re-send of the same identity — harmless
            logger.warning(
                "REGISTER on already-bound socket (bound=%s, new=%s) — rejecting "
                "rebind (Defect 3), socket stays bound to %s",
                already, msg.worker_id, already)
            return "reject_rebind"
        existing_fs = fs_by_worker.get(msg.worker_id)
        if existing_fs is not None and existing_fs is not fs_by_sock.get(rawsock):
            logger.warning(
                "worker_id %s already connected on another live socket — rejecting "
                "the duplicate registration (Defect 3)", msg.worker_id)
            return "reject_dup_worker"
        node = self._resolve_node_config(msg.worker_id, msg.hostname, node_allowlist)
        # [RESOLVED EXECUTION SET — G-NO-INFERENCE] The defect Beta named lived
        # exactly here: `_resolve_node_config` FILTERS NOTHING (it falls back to
        # the configured spool root for an undescribed hostname), there was no
        # expected-membership list anywhere, and the only count in the system was
        # `expected_workers` — so a worker from any host became eligible by the
        # single act of dialling in, and a pool assembled from strangers could
        # satisfy the admission threshold. Membership is now decided against the
        # set that was frozen BEFORE the run started. `expected_workers`,
        # `worker_pool_size`, the bounded-admission window and the Blocker-3
        # matrix are all untouched — an unlisted worker simply never enters
        # `_eligible()`, which is what "must not become eligible" means.
        _admitted, _admission_reason = _execution_set_admission(msg.worker_id)
        if not _admitted:
            logger.warning("[EXEC-SET] %s", _admission_reason)
        wconn = self.register_worker(
            worker_id=msg.worker_id, hostname=msg.hostname, backend=msg.backend,
            capabilities=msg.capabilities, node_config=node,
            admission_reason=_admission_reason)
        worker_by_sock[rawsock] = msg.worker_id
        wconn_by_worker[msg.worker_id] = wconn
        fs_by_worker[msg.worker_id] = fs_by_sock[rawsock]
        if msg.worker_id not in registered:
            registered.append(msg.worker_id)
        if wconn.quarantined:
            logger.warning("worker %s registered but quarantined: %s",
                           msg.worker_id, wconn.quarantine_reason)
        return "ok"

    def dispatch_inbound_result(self, msg, rawsock, run_id, bound_worker_id,
                                wconn_by_worker, eligible_provider,
                                credit_id: Optional[int] = None) -> None:
        """[S172-BP AMENDMENT F1-R] The serve loop's DISPOSITION-BOUNDED dispatch of
        one decoded envelope.

        `_serve_dispatch` is called verbatim and is NOT modified; the only thing
        added is the `finally` that ends the ingress reservation once the envelope
        has been definitively disposed of. The `finally` covers dispositions (i),
        (ii) and (iii) in one place precisely because they are indistinguishable
        from out here — accepted into staging, retained in the deferred queue, or
        dropped by the existing fence, the envelope is in all three cases no longer
        pending, which is the whole content of the invariant.

        [S172-BP AMENDMENT F1-R2a] IT FIRES ON THE EXACT CREDITED ENVELOPE, AND
        NOTHING ELSE. Round 2 argued that the credited envelope is "by construction
        the first `sub_stripe_result` the connection delivers after its resume" and
        cleared on `rawsock is holder`. That reasoning covers LATER traffic only. An
        OLDER result of the same connection's — already sitting in `inbound` before
        the pause began — arrives FIRST, is dropped by the existing identity/
        attempt/dedup/terminal fence (consuming no capacity whatsoever), and under
        the socket-only test its `finally` released the credit while the credited
        envelope was still queued behind it on a still-free slot. The token the
        envelope carries is the identity: an envelope with `credit_id is None` is
        UNCREDITED and clears nothing, no matter which socket it came in on.

        This is a SEAM, not a second dispatch path: the serve loop calls it instead
        of calling `_serve_dispatch` directly, so the clear cannot be forgotten at
        one call site, and a gate can drive the REAL disposition sequence rather
        than modelling it."""
        try:
            self._serve_dispatch(msg, run_id, bound_worker_id, wconn_by_worker,
                                 eligible_provider)
        finally:
            if getattr(msg, "message_type", None) == "sub_stripe_result":
                self._release_resume_credit_exact(
                    rawsock, credit_id, delivered=True, disposition="dispatch")

    def _serve_dispatch(self, msg, run_id, bound_worker_id, wconn_by_worker,
                        eligible_provider) -> None:
        """Route ONE inbound stripe-flow message received on a socket whose bound
        identity is `bound_worker_id`.

        Defect 3: the bound identity — NOT msg.worker_id — is authoritative. A
        message whose worker_id != the receiving socket's bound id is a spoof and
        is dropped BEFORE resolving any connection or touching the ledger.
        Every accepted stripe-flow message then passes accept_stripe_message (L1).
        Defect 4: staging runs OFF this dispatch loop (bounded staging executor)."""
        mt = msg.message_type
        msg_worker_id = getattr(msg, "worker_id", None)
        # Defect 3: connection-bound identity enforced at dispatch.
        if bound_worker_id is None or msg_worker_id != bound_worker_id:
            logger.warning(
                "identity mismatch (Decision A): socket bound=%r but msg.worker_id=%r"
                " — dropping %s, no ledger mutation", bound_worker_id, msg_worker_id, mt)
            return
        wconn = wconn_by_worker.get(bound_worker_id)
        if wconn is None:
            return

        if mt == "heartbeat":
            if msg.current_stripe_id:
                ok, _ = self.accept_stripe_message(
                    wconn, run_id, msg.current_stripe_id, bound_worker_id, (ST_CLAIMED,))
                if ok:
                    renewed = self.ledger.renew_lease(
                        run_id, msg.current_stripe_id, bound_worker_id,
                        time.time() + self.config.compute_lease_timeout)
                    # [S172-BP AMENDMENT F2] The real lease is renewed, so the
                    # resume-grace bridge has done its job and must end HERE —
                    # leaving it in place would widen the exemption past the one
                    # window it exists for. Gated on the LEDGER's own answer: a
                    # renew that did not land (the stripe moved out of `claimed`,
                    # or was re-claimed by another worker) has not restored the
                    # lease, so the bridge must stay up until its own bound.
                    if renewed and self.clear_capacity_resume_grace(
                            bound_worker_id):
                        logger.info(
                            "[S172-BP] resume_grace_cleared worker=%s stripe=%s "
                            "— renew_lease succeeded", bound_worker_id,
                            msg.current_stripe_id)
            return
        if mt not in ("sub_stripe_result", "stripe_complete", "stripe_error"):
            return

        ok, reason = self.accept_stripe_message(
            wconn, run_id, msg.stripe_id, bound_worker_id, (ST_CLAIMED, ST_STAGING))
        if not ok:
            logger.warning("L1 fence dropped %s for %s: %s", mt, msg.stripe_id, reason)
            return
        stripe = self.ledger.get_stripe(run_id, msg.stripe_id)
        attempt = stripe["current_attempt"]

        # [S172 D6] Provenance leg 3 of 3: the EFFECTIVE threshold, reported by
        # the worker off the real executor. Recorded per (stripe, attempt) —
        # sub-stripe values individually, so a disagreement between them stays
        # visible, and the stripe roll-up separately, so it can be reconciled
        # against their consensus. A None is recorded AS None, never dropped:
        # a missing value must not shorten the list into apparent agreement.
        if mt == "sub_stripe_result":
            self.record_substripe_effective(
                run_id, msg.stripe_id, attempt, msg.sub_index,
                getattr(msg, "effective_threshold", None))
        elif mt == "stripe_complete":
            self.record_stripe_complete_effective(
                run_id, msg.stripe_id, attempt,
                getattr(msg, "effective_threshold", None))

        if mt == "sub_stripe_result":
            # Defect 2: a duplicate (attempt, sub_index) is dropped BEFORE staging,
            # so it can never spawn a second reservation for one logical shard.
            inserted = self.ledger.record_substripe_result(
                run_id, msg.stripe_id, attempt, msg.sub_index, bound_worker_id,
                msg.seed_start, msg.seed_count, msg.survivor_count,
                remote_spool_path=(msg.spool_path or None),
                size_bytes=msg.size_bytes, sha256=msg.sha256)
            if not inserted:
                logger.warning("duplicate sub-stripe result %s/%s dropped",
                               msg.stripe_id, msg.sub_index)
                return
            # Defect 4: stage OFF the dispatch loop (fetch/verify/rename/fsync in
            # the bounded staging executor).
            if msg.inline is not None:
                self.enqueue_staging("inline", wconn, run_id, msg.stripe_id,
                                     attempt, msg.sub_index, msg, eligible_provider)
            elif msg.spool_path:
                if self.transfer is None:
                    # Defect 4: a spooled result with no transfer adapter is a
                    # CONFIGURATION error — fail the stripe, do NOT silently ignore.
                    logger.error("spooled result but no transfer adapter configured "
                                 "— failing stripe %s (config error)", msg.stripe_id)
                    self.handle_stripe_failure(
                        run_id, msg.stripe_id, retryable=False,
                        eligible_workers=eligible_provider())
                    return
                self.enqueue_staging("remote", wconn, run_id, msg.stripe_id,
                                     attempt, msg.sub_index, msg, eligible_provider)
            else:
                logger.error("malformed result (neither inline nor spool) %s/%s "
                             "— failing stripe", msg.stripe_id, msg.sub_index)
                self.handle_stripe_failure(
                    run_id, msg.stripe_id, retryable=False,
                    eligible_workers=eligible_provider())
        elif mt == "stripe_complete":
            self.ledger.record_stripe_complete(
                run_id, msg.stripe_id, attempt, bound_worker_id,
                msg.substripes_done, msg.survivors_total)
            # Defect 4 (C3): a StripeComplete whose totals do not reconcile is a
            # definitive failure routed through the matrix, not a park in staging.
            self.finalize_stripe(run_id, msg.stripe_id, eligible_provider=eligible_provider)
        elif mt == "stripe_error":
            self.handle_stripe_failure(
                run_id, msg.stripe_id, retryable=msg.retryable,
                eligible_workers=eligible_provider())

    def _dispatch_pending(self, run_id, family_name, phase, fs_by_worker,
                          dispatched, dataset_path, dataset_sha256, window_size,
                          sessions, offset, residues, trial_number,
                          forward_threshold, reverse_threshold) -> None:
        """Send a StripeAssignMessage for every CLAIMED stripe not yet dispatched
        for its current (worker, attempt) — covers initial assignment AND matrix
        reassignment (which re-claims to a DIFFERENT worker at attempt+1).

        [S172 D6] The trial's requested forward/reverse thresholds are threaded in
        from the persisted trial context and handed to the ONE chokepoint
        (`build_stripe_assign_payload`), which direction-resolves them per stripe.
        They are REQUIRED here — no getattr default, no `or 0.25`."""
        for stripe in self.ledger.all_stripes(run_id):
            if stripe["state"] != ST_CLAIMED:
                continue
            wid = stripe["claimed_by"]
            attempt = stripe["current_attempt"]
            key = (stripe["stripe_id"], wid, attempt)
            if key in dispatched:
                continue
            fs = fs_by_worker.get(wid)
            if fs is None:
                continue
            payload = self.build_stripe_assign_payload(
                dataset_path, window_size, sessions, offset, residues,
                dataset_sha256=dataset_sha256, phase=phase,
                forward_threshold=forward_threshold,
                reverse_threshold=reverse_threshold)
            # Provenance leg 2 of 3: what the payload ACTUALLY carried, recorded
            # per (stripe, attempt) and marked D6-generated EXPLICITLY — this
            # assignment now owes the parent an effective-threshold report, and
            # its absence will fail the trial closed before commit.
            self.record_assignment_threshold(
                run_id, stripe["stripe_id"], attempt, phase,
                payload["min_match_threshold"], d6_generated=True)
            try:
                fs.send_msg(StripeAssignMessage(
                    worker_id=wid, stripe_id=stripe["stripe_id"],
                    trial_number=trial_number, prng_type=family_name,
                    family_name=family_name, seed_start=stripe["seed_start"],
                    seed_count=stripe["seed_count"], phase=phase, attempt=attempt,
                    payload=payload))
                dispatched.add(key)
            except Exception as e:
                logger.warning("dispatch to %s failed: %s", wid, e)


# ===========================================================================
# Integration entry point
# ===========================================================================
def workflow_stages_for(prng_base: str, test_both_modes: bool):
    """§6.8 test-both-modes workflow → the ordered (family, phase) STAGES a trial
    runs (Defect 6). CONSTANT IS ALWAYS BIDIRECTIONAL — phases 1 (forward) and 2
    (reverse) run for every trial, exactly as legacy Step 1 does ("PART 1:
    CONSTANT SKIP TEST (Always runs)", window_optimizer_integration_final.py:561+).
    The HYBRID PAIR (phases 3/4, variable skip) runs ONLY when test_both_modes.

    D1.0 correction [TB-D1-B1]: the pre-D1.0 `test_both_modes=False` branch
    returned the forward-constant stage ALONE, so such a trial executed no P2
    reverse pass and could never produce a constant bidirectional population."""
    if test_both_modes:
        return [(prng_base, 1),
                (f"{prng_base}_reverse", 2),
                (f"{prng_base}_hybrid", 3),
                (f"{prng_base}_hybrid_reverse", 4)]
    return [(prng_base, 1),
            (f"{prng_base}_reverse", 2)]


# ---------------------------------------------------------------------------
# [S172 Staging Part B] Coordinator staging resolution + startup validation
#
# Beta's binding ruling: the production coordinator must NOT auto-detect
# /dev/shm/prng/miner. Worker-local OUTPUT keeps its documented
# `null -> /dev/shm/prng/miner` auto-detect (resolve_miner_output_dir(),
# range_miner_worker.py) — coordinator STAGING is a different thing, with
# different ownership, lifetime and capacity, and `null` is INVALID at the
# production boundary.
#
# Why: coordinator staging is local to the Zeus/VM101 box and receives payloads
# pulled from EVERY worker's spool. On tmpfs those bytes are RAM. The default
# staging_high_water_bytes (16 GiB) exceeds both /dev/shm (7.78 GiB) and total
# RAM (15.9 GiB, swap 0) on VM101 — so the admission control that exists to
# prevent an OOM could not bind before the OOM occurred.
# ---------------------------------------------------------------------------

#: Filesystem types that are RAM-backed, not disk-backed. Coordinator staging on
#: any of these is refused for the approved Phase-7 configuration.
_RAM_BACKED_FSTYPES = frozenset({"tmpfs", "ramfs", "devtmpfs"})

#: Operational headroom demanded ON TOP OF the configured high-water mark:
#: 10% of the high-water, never less than 1 GiB. Staging is not the only writer
#: to the filesystem (ledger DB, provenance records, the run's own artifacts), so
#: a filesystem sized to EXACTLY the high-water is not a safe configuration.
_STAGING_HEADROOM_FLOOR_BYTES = 1024 ** 3


def _staging_headroom_bytes(high_water_bytes: int) -> int:
    """Operational headroom required above the configured high-water mark."""
    return max(_STAGING_HEADROOM_FLOOR_BYTES, int(high_water_bytes) // 10)


def _filesystem_type_for(path: str) -> Optional[str]:
    """Filesystem type backing `path`, via the longest matching /proc/mounts
    mount point. Returns None if it cannot be determined — an UNDETERMINED
    filesystem is never reported as disk-backed (VIR-5: unobservable is not
    clean)."""
    try:
        with open("/proc/mounts", "r") as fh:
            entries = []
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    # /proc/mounts octal-escapes spaces etc. in the mount point.
                    mount_point = parts[1].encode().decode("unicode_escape")
                    entries.append((mount_point, parts[2]))
    except OSError:
        return None
    real = os.path.realpath(path)
    best_mp, best_type = None, None
    for mount_point, fstype in entries:
        if real == mount_point or real.startswith(
                mount_point.rstrip("/") + "/") or mount_point == "/":
            if best_mp is None or len(mount_point) > len(best_mp):
                best_mp, best_type = mount_point, fstype
    return best_type


def resolve_coordinator_staging_dir(
    staging_dir: Optional[str],
    miner_output_dir: Optional[str],
    *,
    warn: bool = True,
) -> str:
    """[Part B §1.1] Resolve the CANONICAL coordinator staging directory.

    `staging_dir` is canonical. `miner_output_dir` is a TEMPORARY
    backward-compatible alias. The five rules, implemented exactly:

      1. only `staging_dir` set                  -> use it
      2. only an explicit `miner_output_dir` set -> populate `staging_dir`,
                                                    with a deprecation warning
      3. both set and they DIFFER                -> FAIL CLOSED
      4. neither set                             -> FAIL CLOSED
      5. any implicit /dev/shm fallback           -> PROHIBITED (never reached:
                                                    rule 4 fails first, and no
                                                    auto-detect is called here)

    Rule 5 is enforced by CONSTRUCTION, not by a check: this function contains no
    fallback candidate at all. `resolve_miner_output_dir()` is deliberately NOT
    called — that resolver remains correct for its own subject (worker-local
    output) and is left untouched.

    Raises StagingConfigurationError (non-retryable) for rules 3 and 4.
    """
    canonical = (staging_dir or "").strip() or None
    alias = (miner_output_dir or "").strip() or None

    if canonical and alias:
        if os.path.abspath(canonical) != os.path.abspath(alias):
            raise StagingConfigurationError(
                "[Part B §1.1 rule 3] conflicting coordinator staging configuration: "
                f"staging_dir={canonical!r} and miner_output_dir={alias!r} differ. "
                "staging_dir is canonical; remove the deprecated miner_output_dir "
                "alias or set both to the same path.")
        return canonical                                    # rule 1 (identical)

    if canonical:
        return canonical                                    # rule 1

    if alias:                                               # rule 2
        if warn:
            logger.warning(
                "[Part B §1.1 rule 2] DEPRECATED: coordinator staging resolved from "
                "miner_output_dir=%r. `staging_dir` is the canonical field; "
                "miner_output_dir is a temporary backward-compatible alias and the "
                "production manifest must populate staging_dir.", alias)
        return alias

    # rule 4 — and, by construction, rule 5.
    raise StagingConfigurationError(
        "[Part B §1.1 rule 4] coordinator staging is not configured: neither "
        "`staging_dir` (canonical) nor `miner_output_dir` (deprecated alias) is "
        "set. There is NO implicit /dev/shm fallback for coordinator staging "
        "(rule 5, PROHIBITED) — worker-local output auto-detect is a different "
        "subject and is unaffected. Declare `staging_dir` in "
        "agent_manifests/window_optimizer.json default_params.")


def validate_coordinator_staging_dir(
    path: str,
    high_water_bytes: int,
    *,
    require_disk_backed: bool = True,
) -> Dict[str, Any]:
    """[Part B §1.3] Validate the resolved coordinator staging location BEFORE
    any dispatch or reservation accounting. Returns a dict of measured evidence.

    Checks, in order — each raising StagingConfigurationError (non-retryable):
      - absolute
      - creatable and writable
      - supports temp-write and ATOMIC RENAME (PROVEN by writing and renaming,
        never inferred from the filesystem type)
      - disk-backed (refused on tmpfs/ramfs/devtmpfs)
      - does not advertise a high-water LARGER than usable capacity
      - has capacity for the configured high-water PLUS operational headroom

    Beta: "Admission control cannot be represented as an OOM safeguard when the
    configured mark exceeds the filesystem that must hold the staged data."

    ⚠ KNOWN LIMITATION — THIN PROVISIONING IS INVISIBLE FROM INSIDE THE GUEST.
    os.statvfs() reports the GUEST filesystem's view. On VM101 the backing store
    is a THIN-PROVISIONED Proxmox local-lvm pool that is OVERSUBSCRIBED
    (measured 2026-08-04: pool 816.21 GiB total, 67.80% used, ~263 GiB actually
    free; vm-101-disk-1 alone is 932 GiB provisioned at 52.20% consumed; total
    provisioned across VMs is 1,188 GiB against an 816 GiB pool). VM101's
    guest-visible ~427 GiB free therefore OVERSTATES the real backing by roughly
    164 GiB.

    This check cannot see that, and must not be read as proving the bytes are
    physically available. Host thin-pool exhaustion would present here as a WRITE
    FAILURE on a filesystem the guest believes has space — not as a capacity
    rejection at startup. The 16 GiB high-water is comfortable against the ~263
    GiB really free, which is why it stands; the limitation is recorded because
    the guarantee is narrower than the number suggests.
    """
    if not path or not os.path.isabs(path):
        raise StagingConfigurationError(
            f"[Part B §1.3] coordinator staging_dir must be ABSOLUTE, got {path!r}")

    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise StagingConfigurationError(
            f"[Part B §1.3] coordinator staging_dir {path!r} is not creatable: {e}") from e
    if not os.path.isdir(path):
        raise StagingConfigurationError(
            f"[Part B §1.3] coordinator staging_dir {path!r} is not a directory")
    if not os.access(path, os.W_OK | os.X_OK):
        raise StagingConfigurationError(
            f"[Part B §1.3] coordinator staging_dir {path!r} is not writable")

    # --- atomic-rename PROOF: write, fsync, rename, verify, unlink ----------
    probe_tmp = os.path.join(path, f".s172_staging_probe_{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp")
    probe_dst = probe_tmp[:-4] + ".committed"
    payload = b"s172-staging-part-b-atomic-rename-probe"
    try:
        with open(probe_tmp, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(probe_tmp, probe_dst)          # same-directory atomic rename
        with open(probe_dst, "rb") as fh:
            if fh.read() != payload:
                raise StagingConfigurationError(
                    f"[Part B §1.3] atomic-rename probe on {path!r} produced wrong bytes")
    except StagingConfigurationError:
        raise
    except OSError as e:
        raise StagingConfigurationError(
            f"[Part B §1.3] coordinator staging_dir {path!r} does not support "
            f"temp-write + atomic rename: {e}") from e
    finally:
        for p in (probe_tmp, probe_dst):
            try:
                os.unlink(p)
            except OSError:
                pass

    # --- disk-backed --------------------------------------------------------
    fstype = _filesystem_type_for(path)
    if require_disk_backed:
        if fstype is None:
            raise StagingConfigurationError(
                f"[Part B §1.3] cannot determine the filesystem backing {path!r}; "
                "an UNDETERMINED filesystem is not reported as disk-backed (VIR-5)")
        if fstype in _RAM_BACKED_FSTYPES:
            raise StagingConfigurationError(
                f"[Part B §1.3] coordinator staging_dir {path!r} is on {fstype!r}, "
                "which is RAM-backed. Coordinator staging must be DISK-BACKED for "
                "the approved Phase-7 configuration: it receives payloads pulled "
                "from every worker's spool, so RAM-backed staging converts a "
                "staging high-water into host memory pressure.")

    # --- capacity -----------------------------------------------------------
    st = os.statvfs(path)
    total_bytes = st.f_blocks * st.f_frsize
    avail_bytes = st.f_bavail * st.f_frsize
    high_water_bytes = int(high_water_bytes)
    headroom = _staging_headroom_bytes(high_water_bytes)

    if high_water_bytes > avail_bytes:
        raise StagingConfigurationError(
            f"[Part B §1.3] capacity-invalid configuration: "
            f"staging_high_water_bytes={high_water_bytes} "
            f"({high_water_bytes / 1024**3:.2f} GiB) EXCEEDS the usable capacity of "
            f"{path!r} ({avail_bytes} B = {avail_bytes / 1024**3:.2f} GiB available, "
            f"fstype={fstype}). Admission control cannot be represented as an OOM "
            f"safeguard when the configured mark exceeds the filesystem that must "
            f"hold the staged data. Lower staging_high_water_bytes or stage elsewhere.")

    if high_water_bytes + headroom > avail_bytes:
        raise StagingConfigurationError(
            f"[Part B §1.3] insufficient operational headroom on {path!r}: "
            f"high_water={high_water_bytes / 1024**3:.2f} GiB + "
            f"headroom={headroom / 1024**3:.2f} GiB exceeds "
            f"{avail_bytes / 1024**3:.2f} GiB available (fstype={fstype})")

    evidence = {
        "staging_dir": path,
        "fstype": fstype,
        "disk_backed": fstype not in _RAM_BACKED_FSTYPES if fstype else False,
        "total_bytes": total_bytes,
        "available_bytes": avail_bytes,
        "high_water_bytes": high_water_bytes,
        "headroom_bytes": headroom,
        "atomic_rename_proven": True,
    }
    logger.info(
        "[S172 Part B] coordinator staging VALIDATED: %s (fstype=%s, disk-backed, "
        "avail=%.2f GiB, high_water=%.2f GiB, headroom=%.2f GiB, atomic-rename proven)",
        path, fstype, avail_bytes / 1024**3, high_water_bytes / 1024**3,
        headroom / 1024**3)
    return evidence


def build_coordinator(
    *,
    staging_dir: Optional[str] = None,
    seed_cap_nvidia: int = 5_000_000,
    seed_cap_amd: int = 2_000_000,
    seed_cap_nvidia_hybrid: int = 2_500_000,
    seed_cap_amd_hybrid: int = 1_000_000,
    miner_stripe_size: int = 67_108_864,
    staging_high_water_bytes: int = 16 * 1024 ** 3,
    staging_high_water_files: int = 512,
    compute_lease_timeout: float = 300.0,
    staging_timeout: float = 600.0,
    # [S172-BP §3, Beta C] The four staging-capacity controls, wired end to end.
    # Before this work they existed ONLY in the dataclass — the identical shape to
    # the `staging_dir` dead read Part B closed, and the reason a badly-sized
    # deferred queue could be changed only by an in-source edit. Values stay at
    # today's defaults: Beta did NOT rule a new number ("tune after measurement").
    staging_workers: int = 4,
    staging_queue_depth: int = 2,
    staging_deferred_max: Optional[int] = None,
    staging_capacity_timeout: float = 600.0,
    miner_host: str = "127.0.0.1",
    miner_port: int = DEFAULT_MINER_PORT,
    db_path: Optional[str] = None,
    transfer: Optional[TransferAdapter] = None,
    phase5_sink: Optional[Phase5Sink] = None,
) -> "RangeMinerCoordinator":
    """Construct a coordinator + its durable ledger from resolved config. Factory
    shared by run_trial_miner() (production) and the harness (plumbing tests)."""
    config = CoordinatorConfig(
        seed_cap_nvidia=seed_cap_nvidia,
        seed_cap_amd=seed_cap_amd,
        seed_cap_nvidia_hybrid=seed_cap_nvidia_hybrid,
        seed_cap_amd_hybrid=seed_cap_amd_hybrid,
        miner_stripe_size=miner_stripe_size,
        staging_high_water_bytes=staging_high_water_bytes,
        staging_high_water_files=staging_high_water_files,
        staging_dir=staging_dir,
        compute_lease_timeout=compute_lease_timeout,
        staging_timeout=staging_timeout,
        staging_workers=staging_workers,
        staging_queue_depth=staging_queue_depth,
        staging_deferred_max=staging_deferred_max,
        staging_capacity_timeout=staging_capacity_timeout,
        miner_host=miner_host,
        miner_port=miner_port,
    )
    if db_path is None:
        base = staging_dir or "."
        db_path = os.path.join(base, "miner_ledger.db")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    ledger = MinerLedger(db_path)
    return RangeMinerCoordinator(config, ledger, transfer=transfer,
                                 phase5_sink=phase5_sink)


def run_trial_miner(
    coordinator_cfg: str,
    config,
    trial_number: int,
    prng_base: str,
    residues,
    total_seeds: int,
    forward_threshold: float,
    reverse_threshold: float,
    test_both_modes: bool,
    dataset_path: str,
    # D0 (Blocker 2, REV3): skip bounds flow from the resolved WindowConfig at the
    # _use_miner call site (they were dropped before) so every published manifest
    # carries the real skip_min/skip_max. FAIL-CLOSED default: None (not 0). An
    # OMITTED skip is no longer fabricated into a valid-looking 0 at the entry point
    # BEFORE build_trial_context_from_serve's missing-field guard sees it — the None
    # flows through the serve `context` unchanged and the guard raises
    # MinerMetadataError. A caller that legitimately needs zero must pass
    # skip_min=0, skip_max=0 EXPLICITLY (the real _use_miner call site already threads
    # config.skip_min/config.skip_max, so production is unaffected).
    skip_min: Optional[int] = None,
    skip_max: Optional[int] = None,
    # D0 (Blocker, REV4): window_size/offset are ALSO mandatory
    # _SERVE_CONTEXT_REQUIRED metadata (REV3 fixed skip_min/skip_max; these two
    # still fabricated 1/0 from kwargs.get). Same fail-closed shape as skip: an
    # Optional=None passthrough so an OMITTED value reaches
    # build_trial_context_from_serve's missing-field guard as None (rejected)
    # instead of a fabricated 1/0. The real _use_miner call site passes
    # config.window_size/config.offset, so production is unchanged. (sessions stays
    # kwargs-optional below: it is NOT in _SERVE_CONTEXT_REQUIRED and normalizes
    # None -> [].)
    window_size: Optional[int] = None,
    offset: Optional[int] = None,
    worker_pool_size: int = 8,
    seed_cap_nvidia: int = 5_000_000,
    seed_cap_amd: int = 2_000_000,
    seed_cap_nvidia_hybrid: int = 2_500_000,
    seed_cap_amd_hybrid: int = 1_000_000,
    miner_stripe_size: int = 67_108_864,
    miner_substripes: int = 8,
    miner_output_dir: str = None,
    staging_high_water_bytes: int = 16 * 1024 ** 3,
    staging_high_water_files: int = 512,
    staging_dir: str = None,
    compute_lease_timeout: float = 300.0,
    staging_timeout: float = 600.0,
    # [S172-BP §3, Beta C] hop 3 of the configuration route for the four staging
    # capacity controls. `staging_deferred_max=None` means USE THE DERIVED BOUND
    # (§2) — it is an operator OVERRIDE, not a value production is expected to set.
    staging_workers: int = 4,
    staging_queue_depth: int = 2,
    staging_deferred_max: Optional[int] = None,
    staging_capacity_timeout: float = 600.0,
    miner_host: str = "127.0.0.1",
    miner_port: int = DEFAULT_MINER_PORT,
    node_allowlist=None,
    transfer: Optional[TransferAdapter] = None,
    phase5_sink: Optional[Phase5Sink] = None,
    listen_sock=None,
    _serve=None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Stripe-based Step 1 backend (S172 RANGE-MINER).

    Signature mirrors run_trial_persistent (persistent_worker_coordinator.py:
    run_trial_persistent) for drop-in integration at
    window_optimizer_integration_final.py:_use_miner gate.

    L4 config wiring: the two hybrid caps and the five staging-resource knobs are
    explicit parameters (NOT buried module constants) and fold into a
    CoordinatorConfig under the 6-level precedence (§12.4/§7:
    CLI > JSON config > WATCHER default_params > WATCHER parameter_bounds.default
    > coordinator constructor defaults > argparse defaults). The values arriving
    here are the RESOLVED result of that precedence; explicit args win.

    This builds the CoordinatorConfig + durable ledger + coordinator, creates the
    trial, and drives it to a terminal state via the REAL default serve path
    (`RangeMinerCoordinator.serve_trial`, a framed-TCP server over the Phase-2
    protocol — Team Beta's binding requirement). `_serve` stays an injectable seam
    for tests. Only the real worker FLEET / CT100 keys remain Phase-6/7; a
    coordinator server binding to loopback workers is in scope and default.
    """
    # [S172 Staging Part B §1.1/§1.3, Beta binding ruling] Resolve the canonical
    # coordinator staging directory and VALIDATE it HERE — before build_coordinator,
    # before the ledger exists, before the trial is created, and therefore before
    # any worker dispatch or reservation accounting.
    #
    # This replaces `staging_dir or miner_output_dir` (Defect 6's half-fix, which
    # was written for the case where miner_output_dir is SET; production supplies
    # null, so it resolved to None and every miner-backed run died at the first
    # sub-stripe with `config.staging_dir is not set`).
    #
    # P0.5's dataset authority is the precedent: FAIL BEFORE FIRST WORKER DISPATCH,
    # naming the resolved path and the reason. A misconfiguration that only surfaces
    # after 25 workers registered and 16 stripes computed is the defect twice over.
    staging_dir_resolved = resolve_coordinator_staging_dir(staging_dir, miner_output_dir)
    staging_evidence = validate_coordinator_staging_dir(
        staging_dir_resolved, staging_high_water_bytes)
    base_dir = staging_dir_resolved
    coordinator = build_coordinator(
        staging_dir=staging_dir_resolved,
        seed_cap_nvidia=seed_cap_nvidia,
        seed_cap_amd=seed_cap_amd,
        seed_cap_nvidia_hybrid=seed_cap_nvidia_hybrid,
        seed_cap_amd_hybrid=seed_cap_amd_hybrid,
        miner_stripe_size=miner_stripe_size,
        staging_high_water_bytes=staging_high_water_bytes,
        staging_high_water_files=staging_high_water_files,
        compute_lease_timeout=compute_lease_timeout,
        staging_timeout=staging_timeout,
        # [S172-BP §3] the four capacity controls reach CoordinatorConfig here.
        staging_workers=staging_workers,
        staging_queue_depth=staging_queue_depth,
        staging_deferred_max=staging_deferred_max,
        staging_capacity_timeout=staging_capacity_timeout,
        miner_host=miner_host,
        miner_port=miner_port,
        transfer=transfer,
        phase5_sink=phase5_sink,
        # base_dir is now the VALIDATED absolute staging path (never "."), so the
        # ledger can no longer land in an arbitrary CWD.
        db_path=os.path.join(base_dir, "miner_ledger.db"),
    )
    # Defect 6: a UNIQUE run_id per trial — NEVER the raw config filename (which
    # would repeat stripe IDs across trials and collide on the PK at trial 2).
    stem = os.path.splitext(os.path.basename(str(coordinator_cfg)))[0] or "trial"
    run_id = f"{stem}_t{trial_number}_{uuid.uuid4().hex[:8]}"
    coordinator.ledger.create_trial(run_id, trial_number)

    # Defect 6: resolve the workflow's family/phase STAGES. An explicit
    # family_name/workflow_phase override (used by the serve-path gate) collapses
    # to a single stage; otherwise test_both_modes drives all four §6.8 families.
    if "family_name" in kwargs or "workflow_phase" in kwargs or "phase" in kwargs:
        fam = kwargs.get("family_name", prng_base)
        ph = int(kwargs.get("phase", kwargs.get("workflow_phase", 1)))
        workflow_stages = [(fam, ph)]
    else:
        workflow_stages = workflow_stages_for(prng_base, test_both_modes)

    context = {
        "run_id": run_id,
        "trial_number": trial_number,
        "prng_base": prng_base,
        "family_name": workflow_stages[0][0],
        "phase": workflow_stages[0][1],
        "workflow_stages": workflow_stages,
        "residues": residues,
        "total_seeds": total_seeds,
        "dataset_path": dataset_path,
        "forward_threshold": forward_threshold,
        "reverse_threshold": reverse_threshold,
        "skip_min": skip_min,
        "skip_max": skip_max,
        "test_both_modes": test_both_modes,
        "worker_pool_size": worker_pool_size,
        "miner_substripes": miner_substripes,
        "node_allowlist": node_allowlist,
        "window_size": window_size,          # REV4: fail-closed passthrough (no `or 1`)
        "sessions": kwargs.get("sessions"),  # intentionally optional (None -> [])
        "offset": offset,                    # REV4: fail-closed passthrough (no `or 0`)
        "staging_dir": staging_dir_resolved,
        # [Part B §1.3] Measured startup-validation evidence, carried on the
        # context so the resolved path, filesystem type, capacity and the
        # atomic-rename proof are OBSERVABLE on the run rather than only logged.
        "staging_validation": staging_evidence,
        "miner_host": miner_host,
        "miner_port": miner_port,
        "listen_sock": listen_sock,
        "serve_poll": kwargs.get("serve_poll", 0.1),
        # Production-timeout correction (Beta): default UNBOUNDED — a real scan far
        # exceeds any fixed 30s. The serve loop enforces a timeout only when one is
        # explicitly configured (the gates inject their own short value).
        "serve_timeout": kwargs.get("serve_timeout", None),
        "serve_read_deadline": kwargs.get("serve_read_deadline", 15.0),
        # [§4.3 admission liveness] BOUNDED admission, UNBOUNDED execution. This is
        # deliberately a separate knob from serve_timeout above and does not weaken
        # it: it bounds only the pre-assignment wait for expected_workers. Beta
        # Ruling 1; default DEFAULT_WORKER_ADMISSION_TIMEOUT (180s, the PWC
        # readiness window).
        "worker_admission_timeout": kwargs.get(
            "worker_admission_timeout", DEFAULT_WORKER_ADMISSION_TIMEOUT),
    }
    # Default: the REAL serve loop (bound method, takes context). Injected test
    # serve keeps the (coordinator, context) shape.
    if _serve is None:
        return coordinator.serve_trial(context)
    return _serve(coordinator, context)
