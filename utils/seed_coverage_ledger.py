#!/usr/bin/env python3
"""Seed-domain terminus, Coverage Ledger v1, and the cursor law.

Authority: Team Beta ruling *"S145 / SEED-DOMAIN SWEEP TERMINUS AND COVERAGE
AUTHORITY"* (2026-08-07). This module is the executable form of §§1-7 of that
ruling. It is a SEPARATE amendment from the S172 staging-capacity work and
shares no file with it.

WHAT THIS MODULE REPLACES, AND WHY
----------------------------------
`database_system.get_next_seed_start()` used to mean ``MAX(seed_range_end)``
over `exhaustive_progress`. Beta deauthorized that table wholesale (§§2-4):

  * it advanced past the governed frontier with no terminus (max end
    16,106,127,360, which is ~3.75x the governed domain);
  * its first row was destructively overwritten at least twice by short runs,
    because the write path is `INSERT OR REPLACE` keyed on the STARTING seed;
  * it carries a ~1.07-billion-seed hole at [1,000, 1,073,741,824);
  * `best_seed` is NULL on all 15 rows and `best_score` 0.0 on 13 — it records
    EXTENT, never YIELD;
  * `MAX(seed_range_end)` is invalid in the presence of gaps: it silently
    declares the hole covered.

Beta's disposition is stronger than "clip it to 2^32". The legacy table has
ZERO certified authority — rows 1-4 included — and the new certified coverage
stream STARTS AT ZERO with no provenance migration. The legacy rows are
retained untouched as historical telemetry; this module never reads them.

THE THREE LAWS IMPLEMENTED HERE
-------------------------------
§1  TERMINUS       the governed domain is exactly [0, 2^32). The constant is
                   IMPORTED from `utils.run_finalizer`, never redefined — see
                   the module-level import below and its comment.
§4  CURSOR LAW     the next seed start is the FIRST GAP in the certified union,
                   never the maximum end. Completion is an explicit state, not
                   an out-of-range number.
§5  PRE-DISPATCH   the same domain law the finalizer already applies AFTER the
    WALL           GPU work, applied BEFORE dispatch.

WHY COVERAGE IS BOUND TO PUBLICATION
------------------------------------
Beta, verbatim: *"Starting a run is not coverage. Receiving all GPU results is
not coverage. Writing a provisional DB row is not coverage. The canonical
retained artifact is the evidence wall."* `record_publication()` therefore
takes the `RunArtifactResult` itself and refuses anything that is not one; a
failed publication has no artifact to hand over and so cannot create a row.

APPEND-ONLY, THREE WAYS
-----------------------
The clobber that destroyed the legacy first row twice must be structurally
impossible here, not merely avoided by convention:

  1. the primary key is `coverage_id` — a content hash that is a per-RECORD
     immutable identity. It is NOT the starting seed, so two intervals that
     both start at 0 are two distinct rows and cannot collide;
  2. no UPDATE, DELETE or `INSERT OR REPLACE` path exists in this module —
     `_insert_row` issues a bare `INSERT`;
  3. `BEFORE UPDATE` and `BEFORE DELETE` triggers `RAISE(ABORT, ...)`, so a
     future caller reaching the table through raw SQL is stopped by the
     database itself.

(3) is what closes `INSERT OR REPLACE` specifically: REPLACE satisfies a
constraint by DELETING the conflicting row, which fires the delete trigger.
SQLite only fires those trigger-driven deletes when recursive triggers are
enabled, so `_connect()` sets ``PRAGMA recursive_triggers = ON`` — without it
defense (3) is silently vacuous against exactly the statement it exists to
stop.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# THE SINGLE DOMAIN AUTHORITY (Beta §1, §7: "Do not invent a separate domain
# constant"). These are imported, never restated. A mutation that moves the
# boundary in `utils/run_finalizer.py` must red BOTH the pre-dispatch gate here
# and the finalizer's own parity gate — that shared failure is the proof the
# two walls are the same wall. Importing the finalizer is CPU-only: it pulls
# numpy and the two utils codecs, and no GPU library (verified 2026-08-08).
from utils.run_finalizer import (
    CANONICAL_SKIP_MODES,
    SEED_DOMAIN_CONTRACT,
    SEED_DOMAIN_END_EXCLUSIVE,
    SEED_DOMAIN_EXCLUSIVE_MAX,
    SEED_DOMAIN_START,
    RunArtifactResult,
)

# The canonical identity vocabulary, imported from the frozen contract rather
# than restated (§4 "reuse, never reimplement"). `_DERIVED_IDENTITY_SUFFIXES`
# is private in its home module and has no `__all__` entry; per the standing
# rule the correct move is to import it anyway, never to fork the tuple —
# a second copy of the suffix list is exactly how `_hybrid_reverse` gets
# mis-classified as `_reverse`.
from utils.canonical_arrays import (
    BASE_PRNG_FAMILIES,
    _DERIVED_IDENTITY_SUFFIXES,
)

__all__ = [
    "SEED_DOMAIN_CONTRACT",
    "SEED_DOMAIN_START",
    "SEED_DOMAIN_END_EXCLUSIVE",
    "SEED_DOMAIN_EXCLUSIVE_MAX",
    "CURSOR_STATUS_OPEN",
    "CURSOR_STATUS_COMPLETE",
    "PUBLICATION_STATUS_CERTIFIED",
    "COVERAGE_LEDGER_TABLE",
    "CoverageLedgerError",
    "SeedDomainPreflightError",
    "LedgerIntegrityError",
    "PublicationBindingError",
    "LedgerSchemaError",
    "CursorResult",
    "NormalizedCoverage",
    "CertifiedInterval",
    "assert_seed_domain_preflight",
    "normalize_certified_intervals",
    "first_uncovered_seed",
    "canonical_coverage_identity",
    "CoverageLedger",
]
# NOTE (R1 Blocker A): `_record_certified_interval` is deliberately ABSENT from
# `__all__` and deliberately underscore-prefixed. Beta: "Python underscore
# privacy is not a security boundary; that is not the point. This is an
# authority boundary." The only supported production path that creates
# certified coverage is `CoverageLedger.record_publication`.

CURSOR_STATUS_OPEN = "OPEN"
CURSOR_STATUS_COMPLETE = "COMPLETE"
PUBLICATION_STATUS_CERTIFIED = "CERTIFIED"

COVERAGE_LEDGER_TABLE = "certified_coverage"

_SHA256_HEX_LEN = 64

# R1 Blocker A — the frozen result contract, for witness validation.
# EVERY field of `RunArtifactResult` must be PRESENT on a non-isinstance
# witness; these are the ones whose TYPE is additionally pinned. The Optional
# parent_* fields are deliberately absent from this map (None is legitimate for
# a first generation) — presence alone is required for them.
_RESULT_FIELD_TYPES: Dict[str, Any] = {
    "run_id": str,
    "prng_base": str,
    "skip_modes_executed": tuple,
    "seed_start": int,
    "seed_count": int,
    "seed_end_exclusive": int,
    "artifact_sha256": str,
    "sidecar_sha256": str,
    "generation_id": str,
    "repository_commit": str,
    "repository_tree_clean": bool,
    "artifact_schema_version": str,
    "sidecar_schema_version": str,
    "encoding_contract_version": str,
    "canonical_map_hash": str,
    "raw_candidate_count": int,
    "l2_winner_count": int,
    "prior_row_count": int,
    "final_row_count": int,
    "created_at": str,
    "elapsed_seconds": float,
}


# ---------------------------------------------------------------------------
# Errors.
#
# All derive from RuntimeError, matching `utils.run_finalizer`'s deliberate
# choice: a fail-closed rejection must never be caught by an upstream
# `except ValueError` fallback arm and mistaken for a recoverable condition.
# ---------------------------------------------------------------------------
class CoverageLedgerError(RuntimeError):
    """Base class for every fail-closed rejection raised by this module."""


class SeedDomainPreflightError(CoverageLedgerError):
    """The §5 pre-dispatch wall refused a requested interval.

    Raised BEFORE fleet work assignment, sieve execution, staging, or any
    coverage mutation. Nothing has been dispatched when this is raised.
    """


class LedgerIntegrityError(CoverageLedgerError):
    """The ledger's own contract was violated (schema, append-only, domain)."""


class PublicationBindingError(CoverageLedgerError):
    """An attempt to record coverage without a successful canonical publication.

    R1 Blocker A: this is now also what refuses a witness that is not the
    canonical `RunArtifactResult` and cannot satisfy the complete frozen result
    contract. A stand-in exposing only `artifact_sha256` is not evidence that a
    canonical publication happened.
    """


class LedgerSchemaError(LedgerIntegrityError):
    """The stored table's shape is not the shape this module writes.

    Fails CLOSED rather than adapting. `CREATE TABLE IF NOT EXISTS` is silent
    about drift, so a pre-R1 development table (which carried `prng_type` /
    `mapping_mode` instead of `prng_base` / `skip_modes_executed`) would
    otherwise be written to as though the columns meant the same thing.
    """


# ---------------------------------------------------------------------------
# Typed helpers. `bool` is rejected for every integer field: True == 1 in
# Python, so a bare isinstance(x, int) accepts True as a seed_start.
# ---------------------------------------------------------------------------
def _require_int(value: Any, name: str, error: type = CoverageLedgerError) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(
            f"{name} must be a Python int, got {value!r} "
            f"({type(value).__name__}). All domain arithmetic is performed in "
            f"unbounded Python integers precisely so it cannot wrap."
        )
    return int(value)


def _require_str(value: Any, name: str, error: type = CoverageLedgerError) -> str:
    if not isinstance(value, str) or not value:
        raise error(f"{name} must be a nonempty str, got {value!r}.")
    return value


def _require_sha256(value: Any, name: str, error: type = CoverageLedgerError) -> str:
    text = _require_str(value, name, error)
    if len(text) != _SHA256_HEX_LEN or any(c not in "0123456789abcdef" for c in text):
        raise error(
            f"{name} must be a 64-character lowercase hex sha256 digest, got "
            f"{value!r}. Coverage is bound to artifact identity; a malformed "
            f"digest cannot identify the artifact it claims to certify."
        )
    return text


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# R1 Blocker B — THE CANONICAL COVERAGE IDENTITY.
#
# Beta rejected BOTH options Alpha framed: not raw `prng_type` alone, and not a
# simple `(prng_type, skip_mode)` scalar key either. The coverage identity is
#
#     prng_base  +  the required executed-mode SET
#
# because Step 1 can execute *constant only* or *constant + variable*, and those
# are DISTINCT SEARCHES: `test_both_modes` runs the base PRNG for constant skip
# and the hybrid variant for variable skip. A range searched only under constant
# skip cannot certify that the variable-skip search happened.
#
# This function is also where the `prng_type` / `prng_base` split is
# CANONICALIZED. Beta overruled filing it as backlog — "It is not backlog
# anymore once this table becomes authoritative" — because WATCHER can query
# `java_lcg_hybrid` while the publication hook records `java_lcg`, splitting one
# logical search into two incompatible namespaces inside a brand-new authority
# table. Both sides now come through here.
# ---------------------------------------------------------------------------
def canonical_coverage_identity(
    prng_type: Any,
    *,
    test_both_modes: Any = False,
) -> Tuple[str, frozenset]:
    """Map a caller's `prng_type` + `test_both_modes` onto the canonical identity.

    Returns `(prng_base, required_modes)`.

    The mode set is derived from the SEARCH CONFIGURATION, never inferred from
    survivors — an executed mode may legitimately produce zero survivors, and
    inference would silently shrink the coverage claim (the same rule D3.5
    applies to `skip_modes_executed`).

        java_lcg,        test_both_modes=False  ->  ('java_lcg', {constant})
        java_lcg,        test_both_modes=True   ->  ('java_lcg', {constant, variable})
        java_lcg_hybrid, (either)               ->  ('java_lcg', {variable})

    The third line is the canonicalization Beta required: a hybrid identity is
    the SAME base family searched under variable skip, not an unrelated coverage
    namespace.
    """
    text = _require_str(prng_type, "prng_type")

    base = text
    for suffix in _DERIVED_IDENTITY_SUFFIXES:      # _hybrid_reverse before _hybrid
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    if base not in BASE_PRNG_FAMILIES:
        raise CoverageLedgerError(
            f"prng_type {text!r} does not resolve to a forward, non-hybrid base "
            f"family (got {base!r}). Coverage identity is keyed on the base "
            f"family plus the executed-mode set; a value outside "
            f"BASE_PRNG_FAMILIES cannot be certified against."
        )

    # A `_hybrid` / `_hybrid_reverse` identity IS the variable-skip search.
    # `_reverse` is a DIRECTION, not a skip mode: forward and reverse are two
    # halves of one bidirectional pass, so it carries the constant-skip meaning
    # of its base unless it also carries `_hybrid`.
    if "_hybrid" in text[len(base):]:
        modes = frozenset({"variable"})
    elif test_both_modes:
        modes = frozenset({"constant", "variable"})
    else:
        modes = frozenset({"constant"})

    return base, modes


def _normalize_mode_set(modes: Any, name: str) -> Tuple[str, ...]:
    """Validate a mode set and return it in the frozen canonical order."""
    if isinstance(modes, str):
        raise CoverageLedgerError(
            f"{name} must be a set/sequence of skip modes, not the bare string "
            f"{modes!r} — a string would iterate as characters."
        )
    try:
        values = set(modes)
    except TypeError as exc:
        raise CoverageLedgerError(
            f"{name} must be iterable, got {modes!r}."
        ) from exc
    if not values:
        raise CoverageLedgerError(
            f"{name} must name at least one executed skip mode; an empty set "
            f"would make every certified record vacuously containing."
        )
    unknown = values - set(CANONICAL_SKIP_MODES)
    if unknown:
        raise CoverageLedgerError(
            f"{name} contains non-canonical skip mode(s) {sorted(unknown)}; the "
            f"frozen vocabulary is {list(CANONICAL_SKIP_MODES)}."
        )
    # CANONICAL_SKIP_MODES order, so the stored string is stable and the content
    # hash cannot depend on the caller's set iteration order.
    return tuple(m for m in CANONICAL_SKIP_MODES if m in values)


# ---------------------------------------------------------------------------
# §5 — THE PRE-DISPATCH SEED-DOMAIN WALL.
# ---------------------------------------------------------------------------
def assert_seed_domain_preflight(
    seed_start: Any,
    seed_count: Any,
    *,
    context: str = "",
) -> Tuple[int, int, int]:
    """Apply the governed domain law BEFORE dispatch. Returns (start, count, end).

    The finalizer already applies exactly this law at
    `utils/run_finalizer.py:533` and `:547` — but only AFTER the GPU work has
    run. Beta §7 requires the same law before it. Both read the SAME constant,
    so they cannot drift.

    The four conditions, in Beta's order:

        seed_start >= 0
        seed_count > 0
        seed_start < 2^32
        seed_start + seed_count <= 2^32

    The addition is performed in unbounded Python integers. Doing it in
    np.uint32 would let an escaping interval wrap and silently re-label itself
    as a tiny interval near zero — which would pass every subsequent check.

    Raises:
        SeedDomainPreflightError: nothing has been dispatched. No fleet work
            assignment, no sieve execution, no staging, no coverage mutation.
    """
    where = f" [{context}]" if context else ""

    start = _require_int(seed_start, "seed_start", SeedDomainPreflightError)
    count = _require_int(seed_count, "seed_count", SeedDomainPreflightError)

    if start < 0:
        raise SeedDomainPreflightError(
            f"seed_domain_preflight{where}: seed_start {start} is negative; the "
            f"governed contract {SEED_DOMAIN_CONTRACT} is "
            f"[{SEED_DOMAIN_START}, {SEED_DOMAIN_EXCLUSIVE_MAX})."
        )
    if count <= 0:
        raise SeedDomainPreflightError(
            f"seed_domain_preflight{where}: seed_count {count} must be strictly "
            f"positive; a run declares one contiguous nonempty interval."
        )
    if start >= SEED_DOMAIN_EXCLUSIVE_MAX:
        raise SeedDomainPreflightError(
            f"seed_domain_preflight{where}: requested seed_start {start} is at or "
            f"beyond the terminus; no run may begin at {SEED_DOMAIN_EXCLUSIVE_MAX} "
            f"or above. Governed contract {SEED_DOMAIN_CONTRACT} "
            f"[{SEED_DOMAIN_START}, {SEED_DOMAIN_EXCLUSIVE_MAX})."
        )

    end_exclusive = start + count           # Python ints — cannot wrap
    if end_exclusive > SEED_DOMAIN_EXCLUSIVE_MAX:
        raise SeedDomainPreflightError(
            f"seed_domain_preflight{where}: requested "
            f"[{start},{end_exclusive}) exceeds {SEED_DOMAIN_CONTRACT} "
            f"[{SEED_DOMAIN_START},{SEED_DOMAIN_EXCLUSIVE_MAX}). The java_lcg "
            f"48-bit internal state does NOT authorize sweeping past the "
            f"terminus; a wider domain requires a separately governed schema "
            f"revision."
        )
    return start, count, end_exclusive


# ---------------------------------------------------------------------------
# §4 — THE CURSOR LAW: first gap, never maximum end.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NormalizedCoverage:
    """The certified union, plus an audit trail of what did NOT enter it.

    `dropped` and `clipped` exist so G-OUT-OF-DOMAIN-LEGACY can PROVE that an
    out-of-domain interval was excluded, rather than inferring it from the
    union's shape. An exclusion nobody can observe is not an exclusion.
    """
    intervals: Tuple[Tuple[int, int], ...]
    dropped: Tuple[Tuple[int, int], ...] = ()
    clipped: Tuple[Tuple[int, int], ...] = ()

    @property
    def covered_seed_count(self) -> int:
        return sum(end - start for start, end in self.intervals)


@dataclass(frozen=True)
class CursorResult:
    """The answer to "where does the next certified run start?".

    Beta §6: completion must be representable EXPLICITLY. There is no
    4,294,967,296 next run, so `next_seed_start` is None exactly when
    `status == COMPLETE`. A consumer that ignores `status` and uses the number
    gets a TypeError, not an out-of-domain sweep.
    """
    status: str
    next_seed_start: Optional[int]
    domain_start: int
    domain_end_exclusive: int
    covered_seed_count: int
    certified_interval_count: int
    normalized: NormalizedCoverage = field(
        default_factory=lambda: NormalizedCoverage(intervals=())
    )

    @property
    def is_complete(self) -> bool:
        return self.status == CURSOR_STATUS_COMPLETE

    @property
    def remaining_seed_count(self) -> int:
        return (self.domain_end_exclusive - self.domain_start) - self.covered_seed_count

    def describe(self) -> str:
        if self.is_complete:
            return (
                f"COMPLETE: the certified union covers "
                f"[{self.domain_start},{self.domain_end_exclusive}) exactly; "
                f"there is no next seed start."
            )
        return (
            f"OPEN: next certified run starts at {self.next_seed_start:,}; "
            f"{self.covered_seed_count:,} of "
            f"{self.domain_end_exclusive - self.domain_start:,} seeds certified "
            f"across {self.certified_interval_count} interval(s)."
        )


def normalize_certified_intervals(
    intervals: Iterable[Sequence[int]],
    *,
    domain_start: int = SEED_DOMAIN_START,
    domain_end_exclusive: int = SEED_DOMAIN_EXCLUSIVE_MAX,
) -> NormalizedCoverage:
    """Normalize, clip/reject by the exact domain contract, and merge overlaps.

    Merging is FOR COMPUTATION ONLY (Beta §6) — it never rewrites a stored row.
    The ledger's own rows are domain-validated at write time, so the clip and
    drop arms are unreachable for them; they exist because this function is
    also the thing that proves legacy out-of-domain extents can never enter the
    union (G-OUT-OF-DOMAIN-LEGACY), and that proof must be executable.

    Adjacent intervals ([0,10) and [10,20)) are merged: they leave no gap, so
    the first uncovered seed is 20.
    """
    d0 = _require_int(domain_start, "domain_start")
    d1 = _require_int(domain_end_exclusive, "domain_end_exclusive")
    if d0 >= d1:
        raise LedgerIntegrityError(
            f"domain [{d0}, {d1}) is empty or inverted; a governed domain must "
            f"be a nonempty half-open interval."
        )

    kept: List[Tuple[int, int]] = []
    dropped: List[Tuple[int, int]] = []
    clipped: List[Tuple[int, int]] = []

    for raw in intervals:
        if len(raw) != 2:
            raise LedgerIntegrityError(
                f"interval {raw!r} must be a (start, end_exclusive) pair."
            )
        start = _require_int(raw[0], "interval start")
        end = _require_int(raw[1], "interval end_exclusive")
        if end <= start:
            raise LedgerIntegrityError(
                f"interval [{start}, {end}) is empty or inverted; a certified "
                f"interval is nonempty and half-open."
            )

        # Entirely outside the governed domain -> REJECTED. This is the arm
        # that keeps the 16.1B legacy extent out of the v1.1 union.
        if end <= d0 or start >= d1:
            dropped.append((start, end))
            continue

        new_start, new_end = max(start, d0), min(end, d1)
        if (new_start, new_end) != (start, end):
            clipped.append((start, end))
        kept.append((new_start, new_end))

    kept.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in kept:
        if merged and start <= merged[-1][1]:      # overlap OR adjacency
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return NormalizedCoverage(
        intervals=tuple(merged),
        dropped=tuple(dropped),
        clipped=tuple(clipped),
    )


def first_uncovered_seed(
    intervals: Iterable[Sequence[int]],
    *,
    domain_start: int = SEED_DOMAIN_START,
    domain_end_exclusive: int = SEED_DOMAIN_EXCLUSIVE_MAX,
) -> CursorResult:
    """Beta §6's algorithm, verbatim.

        normalize valid certified intervals
        clip/reject by exact domain contract
        merge overlaps (for computation only)
        start at D0
        return the first uncovered seed

    Beta's worked example: certified [0, 1000) and [2^30, 2^31) => the next
    cursor is 1000, NOT 2^31. `MAX(seed_range_end)` would have answered 2^31
    and silently declared the ~1.07-billion-seed hole covered — which is the
    exact defect that put a hole in the legacy tracker.
    """
    normalized = normalize_certified_intervals(
        intervals,
        domain_start=domain_start,
        domain_end_exclusive=domain_end_exclusive,
    )

    cursor = domain_start
    for start, end in normalized.intervals:
        if start > cursor:
            break                                  # the first gap
        if end > cursor:
            cursor = end

    if cursor >= domain_end_exclusive:
        return CursorResult(
            status=CURSOR_STATUS_COMPLETE,
            next_seed_start=None,
            domain_start=domain_start,
            domain_end_exclusive=domain_end_exclusive,
            covered_seed_count=normalized.covered_seed_count,
            certified_interval_count=len(normalized.intervals),
            normalized=normalized,
        )

    return CursorResult(
        status=CURSOR_STATUS_OPEN,
        next_seed_start=cursor,
        domain_start=domain_start,
        domain_end_exclusive=domain_end_exclusive,
        covered_seed_count=normalized.covered_seed_count,
        certified_interval_count=len(normalized.intervals),
        normalized=normalized,
    )


# ---------------------------------------------------------------------------
# §5 (of the ruling; §3 of the brief) — COVERAGE LEDGER v1.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CertifiedInterval:
    """One immutable certified coverage record, as stored."""
    coverage_id: str
    run_id: str
    study_identity: Optional[str]
    prng_base: str
    skip_modes_executed: str          # canonical order, comma-joined
    seed_domain_contract: str
    seed_start: int
    seed_end_exclusive: int
    dataset_sha256: str
    repository_commit: str
    artifact_sha256: str
    generation_id: Optional[str]
    publication_status: str
    recorded_at: str

    @property
    def seed_count(self) -> int:
        return self.seed_end_exclusive - self.seed_start

    @property
    def executed_modes(self) -> frozenset:
        return frozenset(self.skip_modes_executed.split(","))

    def covers_modes(self, required_modes: frozenset) -> bool:
        """Beta's containment predicate: requested ⊆ this record's executed set.

        `{constant}` certified does NOT satisfy a `{constant, variable}`
        request — the variable-skip search never ran over that range.
        """
        return required_modes <= self.executed_modes

    def as_interval(self) -> Tuple[int, int]:
        return (self.seed_start, self.seed_end_exclusive)


_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {COVERAGE_LEDGER_TABLE} (
    coverage_id            TEXT    PRIMARY KEY,
    run_id                 TEXT    NOT NULL,
    study_identity         TEXT,
    prng_base              TEXT    NOT NULL,
    skip_modes_executed    TEXT    NOT NULL,
    seed_domain_contract   TEXT    NOT NULL,
    seed_start             INTEGER NOT NULL,
    seed_end_exclusive     INTEGER NOT NULL,
    dataset_sha256         TEXT    NOT NULL,
    repository_commit      TEXT    NOT NULL,
    artifact_sha256        TEXT    NOT NULL,
    generation_id          TEXT,
    publication_status     TEXT    NOT NULL,
    recorded_at            TEXT    NOT NULL,
    CHECK (seed_start >= 0),
    CHECK (seed_end_exclusive > seed_start),
    CHECK (seed_end_exclusive <= {SEED_DOMAIN_EXCLUSIVE_MAX}),
    CHECK (publication_status = '{PUBLICATION_STATUS_CERTIFIED}')
)
"""

# Append-only defense (3). See the module docstring: these also close
# `INSERT OR REPLACE`, whose conflict resolution DELETES the losing row —
# but only when PRAGMA recursive_triggers is ON, which `_connect` sets.
_CREATE_NO_UPDATE_TRIGGER_SQL = f"""
CREATE TRIGGER IF NOT EXISTS {COVERAGE_LEDGER_TABLE}_no_update
BEFORE UPDATE ON {COVERAGE_LEDGER_TABLE}
BEGIN
    SELECT RAISE(ABORT, 'certified_coverage is append-only: UPDATE is forbidden - a certified interval is immutable evidence bound to a published artifact');
END
"""

_CREATE_NO_DELETE_TRIGGER_SQL = f"""
CREATE TRIGGER IF NOT EXISTS {COVERAGE_LEDGER_TABLE}_no_delete
BEFORE DELETE ON {COVERAGE_LEDGER_TABLE}
BEGIN
    SELECT RAISE(ABORT, 'certified_coverage is append-only: DELETE is forbidden - this also blocks INSERT OR REPLACE, which satisfies a conflict by deleting');
END
"""

_COLUMNS: Tuple[str, ...] = (
    "coverage_id", "run_id", "study_identity", "prng_base",
    "skip_modes_executed", "seed_domain_contract", "seed_start",
    "seed_end_exclusive", "dataset_sha256", "repository_commit",
    "artifact_sha256", "generation_id", "publication_status", "recorded_at",
)


class CoverageLedger:
    """Append-only certified coverage, and the certified cursor over it.

    The ledger lives in the same SQLite file as the legacy tracker but in its
    own table. It NEVER reads `exhaustive_progress` — that is what makes
    G-LEGACY-NONAUTHORITY provable rather than asserted.
    """

    def __init__(self, db_path: str = "prng_analysis.db") -> None:
        self.db_path = str(db_path)
        self.init_schema()

    # -- connection -------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        # Load-bearing: without recursive triggers, SQLite does NOT fire the
        # BEFORE DELETE trigger for the implicit delete that REPLACE performs,
        # and the append-only guarantee would be vacuous against the exact
        # statement that destroyed the legacy first row twice.
        conn.execute("PRAGMA recursive_triggers = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(_CREATE_TABLE_SQL)
                self._assert_schema_current(conn)
                conn.execute(_CREATE_NO_UPDATE_TRIGGER_SQL)
                conn.execute(_CREATE_NO_DELETE_TRIGGER_SQL)
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_certified_coverage_base "
                    f"ON {COVERAGE_LEDGER_TABLE}(prng_base, seed_start)"
                )

    @staticmethod
    def _assert_schema_current(conn: sqlite3.Connection) -> None:
        """Fail CLOSED on schema drift — `CREATE TABLE IF NOT EXISTS` is silent.

        The pre-R1 development shape carried `prng_type` / `mapping_mode` where
        this module now writes `prng_base` / `skip_modes_executed`. Those are
        NOT renames of the same meaning: Beta's containment law needs the
        executed-mode SET, and a scalar mode column cannot express it. Writing
        the new fields into the old columns would produce a coverage table whose
        rows silently mean something else.
        """
        found = tuple(
            r[1] for r in conn.execute(f"PRAGMA table_info({COVERAGE_LEDGER_TABLE})")
        )
        if found != _COLUMNS:
            raise LedgerSchemaError(
                f"{COVERAGE_LEDGER_TABLE} has an unexpected shape.\n"
                f"  expected: {list(_COLUMNS)}\n"
                f"  found:    {list(found)}\n"
                f"This is a pre-R1 development table (Blocker B changed the "
                f"coverage identity from `prng_type`/`mapping_mode` to "
                f"`prng_base`/`skip_modes_executed`). No certified row has ever "
                f"been published into the old shape — the publication hook did "
                f"not exist when it was created — so the remedy is to DROP it: "
                f"  DROP TABLE {COVERAGE_LEDGER_TABLE};\n"
                f"Refusing to write: a silent column reinterpretation is exactly "
                f"the class of defect this ledger exists to prevent."
            )

    # -- identity ---------------------------------------------------------
    @staticmethod
    def compute_coverage_id(payload: Dict[str, Any]) -> str:
        """Deterministic content identity over the whole bound record.

        Deterministic rather than random so that re-recording the SAME
        publication is idempotent (it collides on the primary key and is
        recognised as already-present) while any DIFFERENT record — including
        one differing only in seed_start — is a different row. Randomness here
        would let a retry silently double-count an interval.
        """
        canonical = json.dumps(
            {k: payload[k] for k in sorted(payload)},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # -- write ------------------------------------------------------------
    #
    # ⚠ R1 BLOCKER A — THERE IS EXACTLY ONE PRODUCTION CERTIFICATION DOOR:
    #
    #     RunArtifactResult -> record_publication() -> certified_coverage
    #
    # `_record_certified_interval` below is an INTERNAL IMPLEMENTATION SEAM, not
    # a supported authority. In the pre-R1 submission it was public, and the
    # amendment's own mutation gate used it to fabricate a never-published
    # billion-seed interval that advanced the authoritative cursor. Beta
    # correctly refused to read that as a hypothetical mutant: it was a live
    # production API bypass of the governing law that canonical publication is
    # the evidence wall.
    # -----------------------------------------------------------------------
    def _record_certified_interval(
        self,
        *,
        run_id: str,
        prng_base: str,
        skip_modes_executed: Any,
        seed_start: int,
        seed_count: int,
        dataset_sha256: str,
        repository_commit: str,
        artifact_sha256: str,
        generation_id: Optional[str] = None,
        study_identity: Optional[str] = None,
    ) -> CertifiedInterval:
        """Append ONE immutable certified interval. INTERNAL — see the note above.

        Every field Beta §5 enumerates is bound here. The interval is
        re-validated against the domain wall on the way in: a certified record
        outside [0, 2^32) must be impossible to create, not merely unlikely.
        """
        start, count, end_exclusive = assert_seed_domain_preflight(
            seed_start, seed_count, context=f"ledger record run_id={run_id!r}"
        )
        base = _require_str(prng_base, "prng_base")
        if base not in BASE_PRNG_FAMILIES:
            raise CoverageLedgerError(
                f"prng_base {base!r} is not a forward, non-hybrid base family; "
                f"coverage identity must use the canonical base family."
            )
        modes = _normalize_mode_set(skip_modes_executed, "skip_modes_executed")

        row: Dict[str, Any] = {
            "run_id": _require_str(run_id, "run_id"),
            "study_identity": (
                None if study_identity is None
                else _require_str(study_identity, "study_identity")
            ),
            "prng_base": base,
            "skip_modes_executed": ",".join(modes),
            "seed_domain_contract": SEED_DOMAIN_CONTRACT,
            "seed_start": start,
            "seed_end_exclusive": end_exclusive,
            "dataset_sha256": _require_sha256(dataset_sha256, "dataset_sha256"),
            "repository_commit": _require_str(repository_commit, "repository_commit"),
            "artifact_sha256": _require_sha256(artifact_sha256, "artifact_sha256"),
            "generation_id": (
                None if generation_id is None
                else _require_str(generation_id, "generation_id")
            ),
            "publication_status": PUBLICATION_STATUS_CERTIFIED,
        }
        coverage_id = self.compute_coverage_id(row)

        existing = self.get(coverage_id)
        if existing is not None:
            # Idempotent: the id is a hash of every bound field, so a colliding
            # row IS this row. Returning it cannot clobber and cannot double
            # count. A DIFFERENT interval hashes differently and inserts.
            return existing

        row["coverage_id"] = coverage_id
        row["recorded_at"] = _utc_now_iso()
        self._insert_row(row)

        stored = self.get(coverage_id)
        if stored is None:                       # pragma: no cover - defensive
            raise LedgerIntegrityError(
                f"coverage row {coverage_id} vanished immediately after insert."
            )
        return stored

    def record_publication(
        self,
        artifact: RunArtifactResult,
        *,
        dataset_sha256: str,
        study_identity: Optional[str] = None,
    ) -> CertifiedInterval:
        """THE certification door. Everything is derived FROM the witness.

        Beta §1 (R1 Blocker A): *"Do not independently accept caller versions of
        fields the witness already possesses."* `RunArtifactResult` is
        constructed ONLY after the canonical publication commit succeeds and it
        already carries `run_id`, `prng_base`, `skip_modes_executed`,
        `seed_start`, `seed_count`, `artifact_sha256`, `generation_id` and
        `repository_commit`. All nine are read off the artifact here. A caller
        therefore CANNOT substitute a different run_id, range, PRNG identity,
        executed-mode set, commit or generation from what the artifact says —
        not because it is discouraged, but because there is no parameter for it.

        Only two things are still accepted from the caller, and both are
        genuinely absent from the frozen result contract:

          * `dataset_sha256` — the run-scoped frozen dataset identity, resolved
            by P0.5 dataset authority. Beta ruled it PROVENANCE, not a v1
            partition key, so it cannot silently split the coverage namespace.
          * `study_identity`  — the Optuna study name; optional, provenance only,
            and never consulted by the cursor.

        Raises:
            PublicationBindingError: the witness is not a canonical
                `RunArtifactResult` and cannot satisfy the complete frozen
                result contract. Nothing is written.
        """
        self._require_publication_witness(artifact)

        return self._record_certified_interval(
            run_id=artifact.run_id,
            prng_base=artifact.prng_base,
            skip_modes_executed=artifact.skip_modes_executed,
            seed_start=artifact.seed_start,
            seed_count=artifact.seed_count,
            dataset_sha256=dataset_sha256,
            repository_commit=artifact.repository_commit,
            artifact_sha256=artifact.artifact_sha256,
            generation_id=artifact.generation_id,
            study_identity=study_identity,
        )

    @staticmethod
    def _require_publication_witness(artifact: Any) -> None:
        """Refuse anything that is not a real publication witness.

        Beta: *"reject an object that is not the canonical result type or cannot
        satisfy the complete frozen result contract."* An `isinstance` pass is
        immediate; anything else must satisfy EVERY field of the frozen
        dataclass with the right type. A stand-in exposing only
        `artifact_sha256` — the pre-R1 `_FakeArtifact` — fails on the first
        missing field, which is the point: possessing a digest is not evidence
        that a canonical publication occurred.
        """
        if isinstance(artifact, RunArtifactResult):
            return

        if artifact is None:
            raise PublicationBindingError(
                "record_publication received no RunArtifactResult. Starting a "
                "run is not coverage; receiving all GPU results is not "
                "coverage; writing a provisional row is not coverage. The "
                "canonical retained artifact is the evidence wall."
            )

        missing, mistyped = [], []
        for field_ in dataclass_fields(RunArtifactResult):
            if not hasattr(artifact, field_.name):
                missing.append(field_.name)
                continue
            expected = _RESULT_FIELD_TYPES.get(field_.name)
            if expected is not None:
                value = getattr(artifact, field_.name)
                if not isinstance(value, expected) or isinstance(value, bool) \
                        and expected is int:
                    mistyped.append(field_.name)

        if missing or mistyped:
            raise PublicationBindingError(
                f"record_publication refused a "
                f"{type(artifact).__name__} witness: it is not a "
                f"RunArtifactResult and does not satisfy the complete frozen "
                f"result contract "
                f"(missing={sorted(missing)}, mistyped={sorted(mistyped)}). "
                f"Coverage may only be certified by the object D3.5 constructs "
                f"AFTER the canonical publication commit succeeds; an object "
                f"carrying a digest is not evidence that a publication happened."
            )

    def _insert_row(self, row: Dict[str, Any]) -> None:
        """The ONLY write statement in this module. A bare INSERT, by design.

        Not `INSERT OR REPLACE` (which clobbered the legacy first row twice),
        not `INSERT OR IGNORE` (which would hide a genuine identity collision).
        """
        placeholders = ", ".join("?" for _ in _COLUMNS)
        columns = ", ".join(_COLUMNS)
        try:
            with closing(self._connect()) as conn:
                with conn:
                    conn.execute(
                        f"INSERT INTO {COVERAGE_LEDGER_TABLE} ({columns}) "
                        f"VALUES ({placeholders})",
                        tuple(row[c] for c in _COLUMNS),
                    )
        except sqlite3.IntegrityError as exc:
            raise LedgerIntegrityError(
                f"refusing to write certified coverage row "
                f"{row.get('coverage_id')}: {exc}"
            ) from exc

    # -- read -------------------------------------------------------------
    @staticmethod
    def _to_interval(row: sqlite3.Row) -> CertifiedInterval:
        return CertifiedInterval(**{c: row[c] for c in _COLUMNS})

    def get(self, coverage_id: str) -> Optional[CertifiedInterval]:
        with closing(self._connect()) as conn:
            cur = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM {COVERAGE_LEDGER_TABLE} "
                f"WHERE coverage_id = ?",
                (coverage_id,),
            )
            row = cur.fetchone()
        return None if row is None else self._to_interval(row)

    def certified_records(
        self,
        prng_base: Optional[str] = None,
        required_modes: Any = None,
    ) -> List[CertifiedInterval]:
        """Certified records, scoped by Beta's canonical coverage identity.

        `prng_base` filters in SQL. `required_modes` applies the CONTAINMENT
        predicate in Python — SQL cannot express subset-of-a-set over a joined
        text column, and doing it here keeps the one predicate in one place
        where `covers_modes` can be tested directly.
        """
        sql = f"SELECT {', '.join(_COLUMNS)} FROM {COVERAGE_LEDGER_TABLE}"
        params: Tuple[Any, ...] = ()
        if prng_base is not None:
            sql += " WHERE prng_base = ?"
            params = (_require_str(prng_base, "prng_base"),)
        sql += " ORDER BY seed_start, seed_end_exclusive, coverage_id"
        with closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()

        records = [self._to_interval(r) for r in rows]
        if required_modes is None:
            return records
        wanted = frozenset(_normalize_mode_set(required_modes, "required_modes"))
        return [r for r in records if r.covers_modes(wanted)]

    def certified_intervals(
        self,
        prng_base: Optional[str] = None,
        required_modes: Any = None,
    ) -> List[Tuple[int, int]]:
        return [r.as_interval()
                for r in self.certified_records(prng_base, required_modes)]

    def certified_cursor(self, prng_base: str, required_modes: Any) -> CursorResult:
        """THE certified cursor. First gap in the certified union, or COMPLETE.

        R1 Blocker B — Beta's coverage identity and containment law:

            record.prng_base == requested.prng_base
            AND requested_modes ⊆ record.skip_modes_executed

        A range certified under `{constant}` therefore does NOT count toward a
        `{constant, variable}` request: `test_both_modes` runs the base PRNG for
        constant skip and the HYBRID variant for variable skip, so those are
        distinct searches and a constant-only sweep never examined the
        variable-skip space. The reverse direction does count — a
        `{constant, variable}` record satisfies a `{constant}` request.

        `required_modes` is mandatory. There is no default, deliberately: a
        caller that omitted it would silently get the most permissive reading
        and over-claim coverage, which is the failure this blocker exists to
        prevent.
        """
        base = _require_str(prng_base, "prng_base")
        wanted = _normalize_mode_set(required_modes, "required_modes")
        return first_uncovered_seed(
            self.certified_intervals(base, wanted),
            domain_start=SEED_DOMAIN_START,
            domain_end_exclusive=SEED_DOMAIN_EXCLUSIVE_MAX,
        )
