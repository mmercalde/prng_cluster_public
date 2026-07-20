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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("range_miner_coordinator")

# Family/cap logic + the CANONICAL substripe serialization are shared with the
# Phase-3 worker — reuse rather than re-derive. Importing build_substripe_payload_bytes
# guarantees Phase 4's inline normalization is byte-identical to what the worker
# spools (Blocker 4); expected_substripes uses the SAME select_seed_cap logic the
# worker partitions with (Blocker 7, range_miner_worker.py:467-474).
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
    # staging_high_water_bytes). Beyond the bound, dispatch back-pressures via the
    # retry matrix instead of retaining more payloads.
    staging_deferred_max: int = 64
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
    ) -> WorkerConnection:
        """Bind a worker's connection and validate its advertised capabilities.
        A cap inconsistency quarantines the worker (registered-but-ineligible,
        durably visible in the workers table) rather than dropping it or picking
        a value. Returns the bound (possibly quarantined) connection."""
        capabilities = capabilities or {}
        seed_caps = capabilities.get("seed_caps") or {}
        variants = capabilities.get("supported_variants") or []
        reason = self._validate_caps(seed_caps)
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
                sh["local_staged_path"], sh["size_bytes"], sh["sha256"])
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
            raise StagingError("config.staging_dir is not set")
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
        remote shards. event_id is the immutable L6 ack key."""
        return {
            "event_id": event_id,
            "run_id": run_id,
            "stripe_id": stripe["stripe_id"],
            "workflow_phase": stripe["phase"],
            "attempt": attempt,
            "sub_index": sub_index,
            "local_spool_path": staged_path,
            "expected_size": size,
            "expected_sha256": sha256,
            "trial_metadata": trial_metadata or {},
        }

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

    def _defer_locked(self, entry) -> bool:
        """Defect 1c (C4): bounded add to `_deferred` — both a COUNT cap
        (staging_deferred_max) and a retained-BYTES cap (staging_high_water_bytes).
        Returns False if adding would exceed either bound; the caller then applies
        dispatch-level back-pressure (matrix) instead of retaining the payload.
        Caller MUST hold _admission_lock."""
        max_count = max(1, int(getattr(self.config, "staging_deferred_max", 64)))
        max_bytes = int(self.config.staging_high_water_bytes)
        add_bytes = self._entry_bytes(entry)
        if len(self._deferred) + 1 > max_count:
            return False
        if self._deferred_retained_bytes() + add_bytes > max_bytes:
            return False
        self._deferred.append(entry)
        return True

    def _release_admission(self, run_id, stripe_id, attempt) -> None:
        """De-commit one attempt's admission budget (attempt failed/cleaned/
        superseded) and pump any deferred sub-stripes now that capacity may fit."""
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
        # D1c: the deferred queue is full — reject via the retry matrix (retryable)
        # rather than retain another payload. The matrix reassigns (hybrid) or fails
        # closed (constant); either way the payload is not held in coordinator RAM.
        self._on_staging_failed(run_id, stripe_id, True, eligible_provider,
                                "staging deferred queue full — dispatch back-pressure")
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
            self._pump_deferred()   # a slot just freed — resume deferred work
        fut.add_done_callback(_on_done)
        return fut

    def _pump_deferred(self) -> None:
        """Resume deferred sub-stripes whose attempt can now be admitted AND a slot
        is free. Runs OFF the dispatch thread (staging-completion callback / matrix /
        ack). Dead attempts (terminal/superseded) are dropped. The slot is acquired
        NONBLOCKING under the admission lock; the submit happens outside it."""
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
            (kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig, fut) = entry
            try:
                real = self._submit_with_slot(kind, wconn, run_id, stripe_id, attempt,
                                              sub_index, msg, elig)
            except BaseException as e:  # noqa: BLE001
                if not fut.done():
                    fut.set_exception(e)
                continue
            self._chain_future(real, fut)

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
        (state='claimed'). staging leases are never here (Blocker 5)."""
        now = time.time() if now is None else now
        out = []
        for st in self.ledger.expired_claimed_stripes(run_id, now):
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
        call is retried idempotently. Idempotent by (event_id, run_id)."""
        now = time.time() if now is None else now
        self.ledger.create_trial(run_id, -1, now)
        trial = self.ledger.get_trial(run_id)
        if trial is not None and trial["state"] == "committed":
            # Defect 5: committed is terminal + mutually exclusive with abort —
            # a committed trial can NEVER be flipped to aborted.
            return {"event": None, "cleanup": "refused", "first": False,
                    "refused": "already_committed"}
        abort_event_id = f"{run_id}:abort"
        first = self.ledger.mark_trial_aborted(run_id, abort_event_id, now)
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

    # ----- assignment payload contract (ties Blocker 6 / Stage 0) ----------
    def build_stripe_assign_payload(
        self, dataset_path: str, window_size: int, sessions, offset: int,
        residues, dataset_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Every StripeAssignMessage.payload MUST carry `dataset`, `dataset_sha256`
        (coordinator-computed), `window_size`, `sessions`, `offset`, and
        `residue_sha256` (computed via the SAME sha256_residues the worker uses).
        dataset_sha256 and residue_sha256 are NEVER optional — the worker's
        Blocker-6 check rejects an assignment lacking dataset_sha256."""
        if not dataset_path:
            raise ValueError("dataset is mandatory in a stripe assignment")
        if residues is None:
            raise ValueError("residues are mandatory to compute residue_sha256")
        if dataset_sha256 is None:
            dataset_sha256 = compute_dataset_sha256(dataset_path)
        if not dataset_sha256:
            raise ValueError("dataset_sha256 is mandatory in a stripe assignment")
        return {
            "dataset": dataset_path,
            "dataset_sha256": dataset_sha256,
            "window_size": window_size,
            "sessions": sessions,
            "offset": offset,
            "residue_sha256": sha256_residues(residues),
        }

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
        family_name = context.get("family_name") or context.get("prng_base") or ""
        phase = int(context.get("phase", context.get("workflow_phase", 1)))
        total_seeds = int(context["total_seeds"])
        residues = context["residues"]
        dataset_path = context["dataset_path"]
        window_size = int(context.get("window_size", 1))
        sessions = context.get("sessions")
        offset = int(context.get("offset", 0))
        expected_workers = int(context.get("worker_pool_size", 1) or 1)
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

        # dataset_sha256 is coordinator-computed ONCE and reused for every assign.
        dataset_sha256 = compute_dataset_sha256(dataset_path)

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
        start = time.time()

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
                            args=(cfs, csock, inbound, reader_stop),
                            name="miner-conn-reader", daemon=True)
                        reader_threads[csock] = th
                        th.start()

                # --- drain complete frames from the bounded inbound queue ---
                drained = 0
                while drained < 256:
                    try:
                        kind, rawsock, msg = inbound.get(timeout=poll if drained == 0 else 0)
                    except _queue.Empty:
                        break
                    drained += 1
                    if kind == "eof":
                        # Defect 3 (C5): a disconnected worker is evicted from the
                        # eligible pool (wconn_by_worker / connections / registered).
                        self._drop_conn(rawsock, fs_by_sock, worker_by_sock,
                                        fs_by_worker, wconn_by_worker, registered)
                        conn_meta.pop(rawsock, None)
                        reader_threads.pop(rawsock, None)
                        continue
                    if rawsock not in fs_by_sock:
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
                        self._serve_dispatch(
                            msg, run_id, worker_by_sock.get(rawsock), wconn_by_worker,
                            _eligible)

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
                eligible = _eligible()
                if len(eligible) >= expected_workers and stage_idx < len(workflow_stages):
                    fam, ph = workflow_stages[stage_idx]
                    if not stage_assigned:
                        # Defect 2 (C4): if NO eligible worker supports this stage's
                        # variant, FAIL THE TRIAL EXPLICITLY — never strand stripes
                        # `pending` forever (which, with the now-unbounded timeout,
                        # would hang the trial).
                        if not any(self.can_assign_variant(w, fam) for w in eligible):
                            self.fail_trial(
                                run_id,
                                reason=f"no eligible worker supports variant {fam!r}")
                            continue
                        self.assign_stripes(
                            run_id, fam, ph, total_seeds, eligible,
                            stripe_prefix=_stage_prefix(stage_idx))
                        stage_assigned = True
                    self._dispatch_pending(
                        run_id, fam, ph, fs_by_worker, dispatched, dataset_path,
                        dataset_sha256, window_size, sessions, offset, residues,
                        context.get("trial_number", -1))
                    self.process_lease_expiry(run_id, eligible)
                    # advance to the next stage once THIS stage's stripes are done
                    sp = _stage_prefix(stage_idx) + "_s"
                    stage_stripes = [s for s in self.ledger.all_stripes(run_id)
                                     if s["stripe_id"].startswith(sp)]
                    if stage_stripes and all(s["state"] == ST_DONE for s in stage_stripes):
                        stage_idx += 1
                        stage_assigned = False
                        if stage_idx >= len(workflow_stages):
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

        trial = self.ledger.get_trial(run_id)
        stripes = {s["stripe_id"]: {
            "state": s["state"], "phase_degraded": bool(s["phase_degraded"]),
            "claimed_by": s["claimed_by"], "current_attempt": s["current_attempt"],
            "survivors_total": s["survivors_total"],
        } for s in self.ledger.all_stripes(run_id)}
        return {
            "run_id": run_id,
            "state": trial["state"] if trial else "unknown",
            "committed": bool(trial and trial["state"] == "committed"),
            "workers_registered": list(registered),
            "stripes": stripes,
            "manifests": list(self.enqueued),
            "bound_addr": self.bound_addr,
        }

    def _conn_reader_loop(self, cfs, rawsock, inbound, reader_stop) -> None:
        """Defect 6 (C3): per-connection reader. Does BLOCKING full-frame reads (so
        a legitimately slow but complete frame is never corrupted) and feeds each
        complete message to the bounded coordinator queue. It touches NO ledger
        state — all dispatch stays single-threaded on the serve loop. On EOF / a
        malformed or oversized frame / socket error (including the serve loop
        closing the socket at shutdown or on the read deadline), it enqueues one
        'eof' and exits. A full queue (coordinator overloaded) also drops the
        connection rather than growing memory without bound."""
        import queue as _queue
        while not reader_stop.is_set():
            try:
                msg = cfs.recv_msg()
            except (ConnectionError, ValueError, OSError):
                break
            except Exception:  # noqa: BLE001 — any decode error drops the connection
                break
            try:
                inbound.put(("msg", rawsock, msg), timeout=1.0)
            except _queue.Full:
                break
        if not reader_stop.is_set():
            try:
                inbound.put(("eof", rawsock, None), timeout=0.5)
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
        wconn = self.register_worker(
            worker_id=msg.worker_id, hostname=msg.hostname, backend=msg.backend,
            capabilities=msg.capabilities, node_config=node)
        worker_by_sock[rawsock] = msg.worker_id
        wconn_by_worker[msg.worker_id] = wconn
        fs_by_worker[msg.worker_id] = fs_by_sock[rawsock]
        if msg.worker_id not in registered:
            registered.append(msg.worker_id)
        if wconn.quarantined:
            logger.warning("worker %s registered but quarantined: %s",
                           msg.worker_id, wconn.quarantine_reason)
        return "ok"

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
                    self.ledger.renew_lease(
                        run_id, msg.current_stripe_id, bound_worker_id,
                        time.time() + self.config.compute_lease_timeout)
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
                          sessions, offset, residues, trial_number) -> None:
        """Send a StripeAssignMessage for every CLAIMED stripe not yet dispatched
        for its current (worker, attempt) — covers initial assignment AND matrix
        reassignment (which re-claims to a DIFFERENT worker at attempt+1)."""
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
                dataset_sha256=dataset_sha256)
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
    runs (Defect 6). test_both_modes drives all FOUR families (fwd/rev constant,
    fwd/rev hybrid); otherwise the single forward-constant stage."""
    if test_both_modes:
        return [(prng_base, 1),
                (f"{prng_base}_reverse", 2),
                (f"{prng_base}_hybrid", 3),
                (f"{prng_base}_hybrid_reverse", 4)]
    return [(prng_base, 1)]


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
    # Defect 6: staging_dir defaults from miner_output_dir when not given, so a
    # production caller that only sets --miner-output-dir still stages locally.
    staging_dir_resolved = staging_dir or miner_output_dir
    base_dir = staging_dir_resolved or "."
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
        miner_host=miner_host,
        miner_port=miner_port,
        transfer=transfer,
        phase5_sink=phase5_sink,
        db_path=os.path.join(base_dir, "miner_ledger.db") if base_dir != "." else None,
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
        "test_both_modes": test_both_modes,
        "worker_pool_size": worker_pool_size,
        "miner_substripes": miner_substripes,
        "node_allowlist": node_allowlist,
        "window_size": kwargs.get("window_size", 1),
        "sessions": kwargs.get("sessions"),
        "offset": kwargs.get("offset", 0),
        "staging_dir": staging_dir_resolved,
        "miner_host": miner_host,
        "miner_port": miner_port,
        "listen_sock": listen_sock,
        "serve_poll": kwargs.get("serve_poll", 0.1),
        # Production-timeout correction (Beta): default UNBOUNDED — a real scan far
        # exceeds any fixed 30s. The serve loop enforces a timeout only when one is
        # explicitly configured (the gates inject their own short value).
        "serve_timeout": kwargs.get("serve_timeout", None),
        "serve_read_deadline": kwargs.get("serve_read_deadline", 15.0),
    }
    # Default: the REAL serve loop (bound method, takes context). Injected test
    # serve keeps the (coordinator, context) shape.
    if _serve is None:
        return coordinator.serve_trial(context)
    return _serve(coordinator, context)
