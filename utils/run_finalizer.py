#!/usr/bin/env python3
"""
run_finalizer.py — S172 Phase-5 Deliverable D3.5: the shared run finalizer.

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md (REV3.1, Team Beta
approved), frozen against HEAD 70cd6f0.

ONE public entry point, used by EVERY backend (legacy in-process sieve, PWC,
ZMQ and — via D6 — the range miner):

    all raw current-run 24-field candidates
        -> STRICTLY VALIDATE EVERY RAW CANDIDATE           (D3, before anything else)
        -> validate current-run coverage
        -> L2 winner selection                              (RECORD domain)
        -> records_to_arrays(L2 winners)                    (D3 columnization)
        -> load + validate certified prior 22-array bundle
        -> L3 merge                                         (ARRAY domain)
        -> global seed-ascending array ordering
        -> validate_array_bundle(final arrays)
        -> immutable-generation publication
        -> RunArtifactResult

WHY L3 IS ARRAY-DOMAIN [REV2 B1] — the certified prior is a 22-array NPZ. The
canonical 24-field record carries two fields the arrays do not (`sessions` and
`prng_base`), which D3 drops by contract, so reconstructing 24-field prior
records is impossible without inventing data. Merging in the array domain also
makes equal/lower prior retention natural: a retained row is COPIED DIRECTLY
from its existing typed array, never reconstructed and never re-encoded.

THE STRUCTURAL DEFECT BEING DESIGNED OUT (spec §0.2):

    OLD:  one mutable file, edited in place
          merge raises -> fallback writer REPLACES it -> lineage destroyed
    NEW:  immutable generations + parent hash chain + single pointer commit
          any failure -> previous certified generation still current

NO FALLBACK WRITER MAY BE INVOKED FROM THIS MODULE UNDER ANY CIRCUMSTANCE. This
module deliberately imports no `subprocess`, spawns no process, and names no
legacy converter — gate F15 asserts that at source level as well as behaviorally.

REUSE, NEVER REIMPLEMENT (spec §2): `records_to_arrays`, `validate_array_bundle`
and the canonical encoders are imported from D3 / Phase-0. The 24-field
validator is NOT duplicated here; §3's raw-candidate wall IS a call into D3.

WHAT THIS MODULE DELIBERATELY DOES NOT DO:
  * it reads and writes NO coverage database — `prng_analysis.db`,
    `exhaustive_progress` and every other coverage table are outside D3.5
    (spec §6.1); gap detection, overlap handling, consolidation and resume
    policy belong to a separate coverage-ledger deliverable;
  * it decides nothing about GLOBAL coverage continuity — the invariant proven
    here is strictly LOCAL: every candidate seed lies inside the one contiguous
    interval the caller declared;
  * it populates neither `MinerTrialAssembly.binary_npz_path` nor
    `all_npz_path` (both remain deprecated and permanently None, Beta Ruling E);
  * it never imports or migrates a historical pre-D3 artifact. Ruling F is a
    CLEAN START: the archived 20,949-row file is forensic evidence, and gets no
    filename-based exception.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.canonical_arrays import (
    BASE_PRNG_FAMILIES,
    CANONICAL_ARRAY_CONTRACT,
    records_to_arrays,
    validate_array_bundle,
)
from utils.prng_encoding import (
    ENCODING_VERSION,
    PRNG_TYPE_ENCODING,
    SKIP_MODE_ENCODING,
    decode_prng_type,
    decode_skip_mode,
)

__all__ = [
    "ACCUMULATOR_DIRNAME",
    "GENERATIONS_DIRNAME",
    "CURRENT_POINTER_NAME",
    "SIDECAR_NAME",
    "ALL_NPZ_NAME",
    "BINARY_NPZ_NAME",
    "CANONICAL_NPZ_NAME",
    "ARTIFACT_SCHEMA_VERSION",
    "SIDECAR_SCHEMA_VERSION",
    "ENCODING_CONTRACT_VERSION",
    "SIDECAR_REQUIRED_KEYS",
    "CANONICAL_SKIP_MODES",
    "RunFinalizerError",
    "RunParameterError",
    "CandidateValidationError",
    "CoverageValidationError",
    "RunIdentityError",
    "AccumulatorConsistencyError",
    "PriorGenerationError",
    "PublicationError",
    "PublicationDurabilityError",
    "RunArtifactResult",
    "canonical_map_hash",
    "finalize_run",
]


# ---------------------------------------------------------------------------
# Frozen names and versions.
#
# THREE SEPARATE version constants, never one generic `schema_version` (§7.3):
# the on-disk array contract, the sidecar payload shape and the PRNG/skip
# encoding evolve for different reasons and must be able to move independently.
# ---------------------------------------------------------------------------
ACCUMULATOR_DIRNAME = ".s172_accumulator"
GENERATIONS_DIRNAME = "generations"
CURRENT_POINTER_NAME = "current"
SIDECAR_NAME = "provenance.json"

ALL_NPZ_NAME = "bidirectional_survivors_all.npz"
BINARY_NPZ_NAME = "bidirectional_survivors_binary.npz"

# The canonical artifact whose SHA-256 becomes `artifact_sha256`. Both published
# names carry byte-identical payloads (§7.1), so the choice fixes the wording,
# not the value; `bidirectional_survivors_binary.npz` is the Steps 2-6 input and
# is therefore the one named canonical.
CANONICAL_NPZ_NAME = BINARY_NPZ_NAME

ARTIFACT_SCHEMA_VERSION = "s172.d3.arrays.v1"
# D3.5-B bumps ONLY this one: the sidecar gained the nine seed-domain fields.
# The arrays, their order and their dtypes are untouched, and so are the
# PRNG/skip encoding maps — neither of the other two constants may move.
SIDECAR_SCHEMA_VERSION = "s172.d3_5.provenance.v1.1"
ENCODING_CONTRACT_VERSION = "s172.phase0.encoding.v1"

# ---------------------------------------------------------------------------
# Seed-domain v1.1 (D3.5-B) — honest stratum labelling.
#
# The `java_lcg` registry family has a 48-BIT internal state; the canonical
# artifact stores `seeds: uint32`. The sweep therefore covers the `high16 = 0`
# stratum — 1 part in 65,536 of the state space — and those upper 16 bits are
# NOT invisible: they are blind to the mod-8 lane but fully visible to mod-125,
# and at TFM's window all 65,536 high-state classes produce distinct draw
# sequences. This is a LABELLING problem, not a storage problem: TFM does
# functional mimicry, not state reversal, so the artifact stays `uint32` and
# declares honestly which stratum it is.
#
# These nine values distinguish three concepts the artifact previously
# conflated:
#
#     canonical PRNG coordinate : 48-bit internal state
#     stored artifact coordinate: uint32 low-state component
#     certified search stratum  : high16 = 0
#
# EVERY ONE IS A FIXED CONSTANT IN v1.1 — none is caller-supplied, none is read
# from the environment, none is inferred from the candidate maximum and none is
# copied from a supplied prior. That is precisely what stops a run publishing a
# sidecar claiming a stratum other than the one the uint32 domain wall actually
# enforced. A sidecar carrying any other value for any of them FAILS CLOSED.
# ---------------------------------------------------------------------------
SEED_SEMANTICS = "internal_state"
SEED_STORAGE_DTYPE = "uint32"
SEED_EFFECTIVE_BITS = 32
SEED_HIGH16_PREFIX = 0
SEED_DOMAIN_CONTRACT = "v1.1-stratum"
SEED_DOMAIN_START = 0
SEED_DOMAIN_END_EXCLUSIVE = 2 ** 32
EXHAUSTIVE_OVER = "high16=0 stratum only"
EXTERNAL_SEED_TRANSFORM = None

# The frozen (field, value) pairs, used for BOTH payload construction and
# exact-value validation so the two can never drift apart.
SEED_DOMAIN_FIELDS: Tuple[Tuple[str, Any], ...] = (
    ("seed_semantics", SEED_SEMANTICS),
    ("seed_storage_dtype", SEED_STORAGE_DTYPE),
    ("seed_effective_bits", SEED_EFFECTIVE_BITS),
    ("seed_high16_prefix", SEED_HIGH16_PREFIX),
    ("seed_domain_contract", SEED_DOMAIN_CONTRACT),
    ("seed_domain_start", SEED_DOMAIN_START),
    ("seed_domain_end_exclusive", SEED_DOMAIN_END_EXCLUSIVE),
    ("exhaustive_over", EXHAUSTIVE_OVER),
    ("external_seed_transform", EXTERNAL_SEED_TRANSFORM),
)

# Strict type domains. `bool` is REJECTED for every integer field: True == 1 in
# Python, so a bare `isinstance(x, int)` would accept `True` as
# `seed_high16_prefix` — and `False` would sail straight through the `== 0`
# value pin, which is the case a value check alone can never catch.
_SEED_DOMAIN_STR_FIELDS: Tuple[str, ...] = (
    "seed_semantics",
    "seed_storage_dtype",
    "seed_domain_contract",
    "exhaustive_over",
)
_SEED_DOMAIN_INT_FIELDS: Tuple[str, ...] = (
    "seed_effective_bits",
    "seed_high16_prefix",
    "seed_domain_start",
    "seed_domain_end_exclusive",
)
_SEED_DOMAIN_NULL_FIELDS: Tuple[str, ...] = ("external_seed_transform",)

# The exact sidecar key set (§7.3), 23 keys at D3.5 and 32 at D3.5-B, in global
# alphabetical order. `sidecar_sha256` is deliberately ABSENT [REV3 C1]: a file
# cannot contain its own hash — writing the field changes the bytes the field
# describes. It lives in `RunArtifactResult` and, for the next generation, in
# `parent_sidecar_sha256`.
SIDECAR_REQUIRED_KEYS: Tuple[str, ...] = (
    "artifact_schema_version",
    "artifact_sha256",
    "canonical_map_hash",
    "created_at",
    "encoding_contract_version",
    "exhaustive_over",
    "external_seed_transform",
    "final_row_count",
    "generation_id",
    "l2_winner_count",
    "parent_artifact_sha256",
    "parent_generation_id",
    "parent_sidecar_sha256",
    "prior_row_count",
    "prng_base",
    "raw_candidate_count",
    "repository_commit",
    "repository_tree_clean",
    "row_count",
    "run_id",
    "seed_count",
    "seed_domain_contract",
    "seed_domain_end_exclusive",
    "seed_domain_start",
    "seed_effective_bits",
    "seed_end_exclusive",
    "seed_high16_prefix",
    "seed_semantics",
    "seed_start",
    "seed_storage_dtype",
    "sidecar_schema_version",
    "skip_modes_executed",
)

_SIDECAR_REQUIRED_KEY_SET = frozenset(SIDECAR_REQUIRED_KEYS)

# The fourteen fields a certified lineage must agree on AT EVERY LINK (§5).
#
# The first five were already required properties of a homogeneous lineage but
# were never compared per link — `_validate_chain` checked hashes, ids, cycles
# and existence only, and `prng_base` was compared solely on the selected tip in
# `_load_prior_generation`. They are not left unchecked merely because D3.5-B
# exposed the seam. The remaining nine are the seed-domain contract: a
# DIFFERENT seed-domain contract requires a NEW CLEAN ROOT, never a link, which
# is the invariant a future v2 must retain.
_LINEAGE_INVARIANT_KEYS: Tuple[str, ...] = (
    "prng_base",
    "artifact_schema_version",
    "sidecar_schema_version",
    "encoding_contract_version",
    "canonical_map_hash",
) + tuple(name for name, _ in SEED_DOMAIN_FIELDS)

# Canonical stored order of the executed skip modes (§8a).
CANONICAL_SKIP_MODES: Tuple[str, ...] = ("constant", "variable")

# The uint32 seed domain wall (§6). Java LCG's 48-bit internal state does NOT
# silently expand this: the frozen artifact stores `seeds: uint32`, and a 48-bit
# domain needs a separately governed schema revision.
SEED_DOMAIN_EXCLUSIVE_MAX = 2 ** 32

_ARRAY_NAMES: Tuple[str, ...] = tuple(n for n, _ in CANONICAL_ARRAY_CONTRACT)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# ---------------------------------------------------------------------------
# Errors.
#
# Every one derives from RunFinalizerError, and RunFinalizerError derives from
# RuntimeError — NOT from ValueError. That is load-bearing: the live caller's
# legacy accumulator block has a `except ValueError` arm that falls back to the
# `convert_survivors_to_binary.py` subprocess, and a fail-closed finalizer
# rejection must never be mistaken for a fallback candidate (§11 [B4]).
# D3's own ValueErrors are re-labelled into CandidateValidationError /
# PriorGenerationError for the same reason.
# ---------------------------------------------------------------------------
class RunFinalizerError(RuntimeError):
    """Base class for every fail-closed rejection raised by the finalizer."""


class RunParameterError(RunFinalizerError):
    """A `finalize_run` argument violated the frozen public contract."""


class CandidateValidationError(RunFinalizerError):
    """A raw current-run candidate failed D3's strict 24-field wall (§3)."""


class CoverageValidationError(RunFinalizerError):
    """A declared interval or a candidate seed violated local coverage (§6)."""


class RunIdentityError(RunFinalizerError):
    """A candidate or prior row disagreed with the run identity (§8a)."""


class AccumulatorConsistencyError(RunFinalizerError):
    """Two candidates for one seed shared a trial number AND a skip mode."""


class PriorGenerationError(RunFinalizerError):
    """A prior generation, its sidecar, or its provenance chain failed (§9)."""


class PublicationError(RunFinalizerError):
    """Publication failed BEFORE the `current` commit point; nothing published."""


class PublicationDurabilityError(RunFinalizerError):
    """The pointer swap COMMITTED but its durability fsync failed [REV3.1 D4].

    Step 13 is the logical commit, so this cannot honestly be reported as
    "nothing published". No `RunArtifactResult` is returned and no fallback is
    invoked; the next invocation performs §7.1b recovery validation and may
    accept the generation only if directory, artifact, sidecar and hash-bound
    pointer all validate.
    """


# ---------------------------------------------------------------------------
# §8 — the frozen public result object. Constructed ONLY after the pointer swap
# and its durability fsync both succeed. No partially successful write may
# produce one.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunArtifactResult:
    generation_id: str
    generation_dir: Path
    all_npz_path: Path
    binary_npz_path: Path
    sidecar_path: Path
    artifact_sha256: str
    sidecar_sha256: str
    parent_generation_id: Optional[str]
    parent_artifact_sha256: Optional[str]
    parent_sidecar_sha256: Optional[str]
    repository_commit: str
    repository_tree_clean: bool
    artifact_schema_version: str
    sidecar_schema_version: str
    encoding_contract_version: str
    canonical_map_hash: str
    run_id: str
    prng_base: str
    skip_modes_executed: Tuple[str, ...]
    seed_start: int
    seed_count: int
    seed_end_exclusive: int
    raw_candidate_count: int
    l2_winner_count: int
    prior_row_count: int
    final_row_count: int
    created_at: str
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Filesystem primitives.
#
# Routed through named module-level helpers ON PURPOSE: the publication order of
# §7.2 is a behavioral contract, and gates F30/F32/F51 need injectable seams to
# prove the fsyncs precede the pointer swap and that EVERY step 1-11 failure
# leaves `current` untouched. Production behavior is a thin wrapper over os.*.
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    """SHA-256 over the bytes actually STORED at `path`."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bytes(path: Path) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write the 22-array bundle. The open file object keeps numpy from
    appending a second `.npz` suffix to an already-suffixed name."""
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)


def _mkdir(path: Path) -> None:
    os.mkdir(path)


def _write_and_fsync_bytes(path: Path, data: bytes) -> None:
    """Write, flush and fsync in one place, so the sidecar's durability step is
    a named seam a gate can spy on alongside the NPZ and directory fsyncs."""
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_rename(src: Path, dst: Path) -> None:
    """Rename within one filesystem. EXDEV (or any equivalent) fails closed."""
    try:
        os.rename(src, dst)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise PublicationError(
                f"the temporary generation directory {src} and its final "
                f"location {dst} are on different filesystems (EXDEV); the "
                f"publication rename cannot be atomic. Failing closed — the "
                f"previous certified generation remains current."
            ) from exc
        raise


def _replace_symlink(target: str, link_path: Path, tmp_link_path: Path) -> None:
    """Create `tmp_link_path` -> `target`, then ATOMICALLY replace `link_path`.

    This is THE SINGLE COMMIT POINT of the whole finalizer (§7.2 step 13):
    artifact, sidecar and both root aliases become valid together, because they
    are all reached THROUGH this one pointer.
    """
    if os.path.lexists(tmp_link_path):
        os.unlink(tmp_link_path)
    os.symlink(target, tmp_link_path)
    os.replace(tmp_link_path, link_path)


def _remove_tree(path: Path) -> None:
    """Best-effort removal of an UNREFERENCED directory tree.

    Only ever applied to `.tmp-*` staging directories, which `current` can never
    point at: a valid pointer target is `generations/<id>--<64 hex>`, a shape no
    `.tmp-` name can take.
    """
    try:
        shutil.rmtree(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# §7.3 — canonical_map_hash. SHA-256 over canonical UTF-8 JSON of the live
# encoding maps, so a registry shift that would silently renumber `prng_type`
# invalidates every downstream generation instead of quietly relabelling rows.
# ---------------------------------------------------------------------------
def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def canonical_map_hash() -> str:
    return hashlib.sha256(_canonical_json_bytes({
        "encoding_version": ENCODING_VERSION,
        "prng_type_encoding": PRNG_TYPE_ENCODING,
        "skip_mode_encoding": SKIP_MODE_ENCODING,
    })).hexdigest()


# ---------------------------------------------------------------------------
# Small typed guards. `bool` is excluded from every integer check: True == 1 in
# Python, so a Boolean would otherwise sail through a seed or count field.
# ---------------------------------------------------------------------------
def _require_int(value: Any, name: str, error: type) -> int:
    if isinstance(value, bool):
        raise error(
            f"{name} is a bool ({value!r}); a Boolean is not an acceptable "
            f"integer value (True == 1 in Python, so this must be explicit)."
        )
    if not isinstance(value, int):
        raise error(
            f"{name} must be a Python int, got {value!r} "
            f"({type(value).__name__}). The coverage arithmetic is performed in "
            f"unbounded Python integers precisely so it cannot wrap."
        )
    return int(value)


def _require_str(value: Any, name: str, error: type) -> str:
    if not isinstance(value, str) or not value:
        raise error(f"{name} must be a nonempty str, got {value!r}.")
    return value


# ---------------------------------------------------------------------------
# §6 — coverage, and §8a — the run-identity wall.
# ---------------------------------------------------------------------------
def _validate_declared_coverage(seed_start: Any, seed_count: Any) -> Tuple[int, int, int]:
    """Validate the declared interval and return (start, count, end_exclusive).

    ALL ARITHMETIC IN PYTHON INTEGERS. Doing `seed_start + seed_count` in
    np.uint32 would permit a wraparound that silently re-labels the run's
    interval as a tiny one starting near zero — F18 mutates exactly that and
    proves the oracle rejects the wrap.
    """
    start = _require_int(seed_start, "seed_start", CoverageValidationError)
    count = _require_int(seed_count, "seed_count", CoverageValidationError)

    if not (0 <= start < SEED_DOMAIN_EXCLUSIVE_MAX):
        raise CoverageValidationError(
            f"seed_start {start} is outside the frozen uint32 seed domain "
            f"[0, {SEED_DOMAIN_EXCLUSIVE_MAX}). The artifact stores "
            f"`seeds: uint32`; widening the domain requires a separately "
            f"governed schema revision and is out of D3.5 scope."
        )
    if count <= 0:
        raise CoverageValidationError(
            f"seed_count {count} must be strictly positive; a run declares one "
            f"contiguous nonempty interval."
        )

    end_exclusive = start + count           # Python ints — cannot wrap
    if not (start < end_exclusive <= SEED_DOMAIN_EXCLUSIVE_MAX):
        raise CoverageValidationError(
            f"declared interval [{start}, {end_exclusive}) escapes the frozen "
            f"uint32 seed domain [0, {SEED_DOMAIN_EXCLUSIVE_MAX}). "
            f"seed_start + seed_count is computed in Python integers, so this "
            f"is a genuine overflow and not a wrapped artefact of uint32 "
            f"addition."
        )
    return start, count, end_exclusive


def _validate_candidate_coverage(
    candidates: Sequence[Mapping[str, Any]], start: int, end_exclusive: int,
) -> None:
    """Every candidate seed lies inside the ONE declared contiguous interval.

    This is the LOCAL invariant D3.5 owns (§6). Whether global coverage is
    continuous, whether `exhaustive_progress` has gaps, and whether a sweep was
    intentionally non-contiguous are all explicitly outside D3.5 (§6.1) — no
    coverage table is read or written anywhere in this module.
    """
    for index, record in enumerate(candidates):
        seed = _require_int(record["seed"], f"candidate {index}: 'seed'",
                            CoverageValidationError)
        if not (0 <= seed < SEED_DOMAIN_EXCLUSIVE_MAX):
            raise CoverageValidationError(
                f"candidate {index}: seed {seed} is outside the frozen uint32 "
                f"seed domain [0, {SEED_DOMAIN_EXCLUSIVE_MAX})."
            )
        if not (start <= seed < end_exclusive):
            raise CoverageValidationError(
                f"candidate {index}: seed {seed} lies outside the interval "
                f"[{start}, {end_exclusive}) declared for this run. A survivor "
                f"from outside the declared sweep would make the generation's "
                f"coverage claim false."
            )


def _validate_run_identity(
    prng_base: Any, skip_modes_executed: Any,
) -> Tuple[str, Tuple[str, ...]]:
    """Validate the run's own identity declaration (§8a), before any candidate.

    `skip_modes_executed` comes from RUN CONFIGURATION and is never inferred
    from survivor rows: an executed mode may legitimately produce zero
    survivors, so inference would silently shrink the claim.
    """
    base = _require_str(prng_base, "prng_base", RunIdentityError)
    if base not in BASE_PRNG_FAMILIES:
        raise RunIdentityError(
            f"prng_base {base!r} is not a forward, non-hybrid canonical base "
            f"family. Directional and derived registry identities (*_reverse, "
            f"*_hybrid, *_hybrid_reverse) are valid KERNEL_REGISTRY keys but "
            f"invalid prng_base values."
        )

    if isinstance(skip_modes_executed, (str, bytes)) or not isinstance(
            skip_modes_executed, Sequence):
        raise RunIdentityError(
            f"skip_modes_executed must be a sequence of mode names, got "
            f"{skip_modes_executed!r} ({type(skip_modes_executed).__name__})."
        )
    modes = tuple(skip_modes_executed)
    if not modes:
        raise RunIdentityError(
            "skip_modes_executed is empty; a run always executes at least one "
            "skip mode, and the executed set comes from configuration."
        )
    if len(set(modes)) != len(modes):
        raise RunIdentityError(
            f"skip_modes_executed {list(modes)} contains duplicates."
        )
    unknown = [m for m in modes if m not in CANONICAL_SKIP_MODES]
    if unknown:
        raise RunIdentityError(
            f"skip_modes_executed contains non-canonical mode(s) {unknown}; "
            f"the vocabulary is {list(CANONICAL_SKIP_MODES)}."
        )
    canonical_order = tuple(m for m in CANONICAL_SKIP_MODES if m in modes)
    if modes != canonical_order:
        raise RunIdentityError(
            f"skip_modes_executed {list(modes)} is not in the canonical stored "
            f"order {list(canonical_order)} (constant, then variable)."
        )
    return base, modes


def _validate_candidate_identity(
    candidates: Sequence[Mapping[str, Any]], prng_base: str,
    skip_modes_executed: Tuple[str, ...],
) -> None:
    """Bind every candidate to the run identity the sidecar will claim (§8a).

    D3 validates each record INTERNALLY consistent but cannot know which run it
    belongs to. Without this wall a `finalize_run(prng_base="java_lcg", ...)`
    fed an internally-valid `xorshift32` candidate publishes a generation that
    is falsely labelled — both parts individually valid, the whole a lie.
    """
    for index, record in enumerate(candidates):
        candidate_base = record["prng_base"]
        if candidate_base != prng_base:
            raise RunIdentityError(
                f"candidate {index}: prng_base {candidate_base!r} differs from "
                f"the run identity {prng_base!r}. The sidecar would label this "
                f"generation with a family it does not exclusively contain."
            )
        skip_mode = record["skip_mode"]
        if skip_mode not in skip_modes_executed:
            raise RunIdentityError(
                f"candidate {index}: skip_mode {skip_mode!r} is not among the "
                f"executed modes {list(skip_modes_executed)} declared for this "
                f"run."
            )


# ---------------------------------------------------------------------------
# §3 — the raw-candidate wall. EVERY raw candidate, before anything else.
# ---------------------------------------------------------------------------
def _validate_raw_candidates(candidates: Sequence[Mapping[str, Any]]) -> None:
    """Pass the COMPLETE raw candidate list through D3's strict validator.

    A malformed LOSING candidate must fail the run, not vanish during selection
    [REV2 B2]:

        seed X: valid record, score 0.9
        seed X: record MISSING `sessions`, score 0.4   <- loses L2, still fails

    The temporary arrays are discarded; only the validation matters here. The
    24-field validator is NOT reimplemented — this call IS the wall.
    """
    try:
        records_to_arrays(candidates)
    except ValueError as exc:
        raise CandidateValidationError(
            f"a raw current-run candidate failed the canonical 24-field "
            f"contract, so the run is rejected before L2 selection — a "
            f"malformed candidate must never vanish by losing: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# §4 — L2 batch winner selection, in the RECORD domain.
# ---------------------------------------------------------------------------
def _l2_sort_key(record: Mapping[str, Any]) -> Tuple[float, int, int]:
    """The frozen L2 key (Ruling D), highest-wins on every component.

        1. highest canonical float32 score
        2. then lowest trial_number
        3. then constant before variable — ONLY as a tiebreak within one trial

    THE COMPARISON DOMAIN IS float32. Two Python floats differing only beyond
    float32 precision are an EXACT TIE, so they must fall through to the
    trial-number tiebreak rather than being separated by precision the artifact
    cannot store. Comparing pre-rounding Python floats while storing the rounded
    value is the defect this converts away.

    Rule 3 is expressed as the LAST tuple component, which is what makes it a
    within-trial tiebreak only: `-trial_number` is compared first, so a
    lower-trial variable record beats a higher-trial constant record (F2). A
    global mode-first rule is a different, rejected ordering.
    """
    score32 = float(np.float32(record["score"]))
    trial = int(record["trial_number"])
    mode_rank = 1 if record["skip_mode"] == "constant" else 0
    return (score32, -trial, mode_rank)


def _select_l2_winners(
    candidates: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    """Exactly one record per seed; the result is INDEPENDENT of input order.

    Order independence is not an accident of a stable sort: within one seed the
    key is a STRICT total order, because a same-trial/same-mode pair — the only
    way two candidates could tie on all three components — is rejected outright
    as accumulator-state corruption (F7). After D1/D2 that collision is
    impossible, so its appearance means the accumulator was fed twice.
    """
    by_seed: Dict[int, List[Tuple[int, Mapping[str, Any]]]] = {}
    for index, record in enumerate(candidates):
        by_seed.setdefault(int(record["seed"]), []).append((index, record))

    winners: List[Mapping[str, Any]] = []
    for seed in sorted(by_seed):
        group = by_seed[seed]
        seen_trial_mode: Dict[Tuple[int, str], int] = {}
        for index, record in group:
            key = (int(record["trial_number"]), record["skip_mode"])
            if key in seen_trial_mode:
                raise AccumulatorConsistencyError(
                    f"seed {seed}: two candidates share trial_number "
                    f"{key[0]} AND skip_mode {key[1]!r} (raw indices "
                    f"{seen_trial_mode[key]} and {index}). After D1/D2 a "
                    f"same-trial, same-mode collision for one seed is "
                    f"impossible; its presence means the accumulator received "
                    f"the same trial's population more than once."
                )
            seen_trial_mode[key] = index
        winners.append(max((r for _, r in group), key=_l2_sort_key))
    return winners


# ---------------------------------------------------------------------------
# §5 — L3 merge, in the ARRAY domain.
# ---------------------------------------------------------------------------
def _l3_merge(
    new_arrays: Mapping[str, np.ndarray],
    prior_arrays: Optional[Mapping[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Merge L2 winner arrays against the certified prior's arrays.

        new score >  prior score  -> replace with the new row
        new score == prior score  -> RETAIN PRIOR, byte-for-byte, every array
        new score <  prior score  -> RETAIN PRIOR, byte-for-byte, every array

    STRICT GREATER-THAN ONLY. Equal retains the prior, and the L2 tiebreakers
    must never reach across this boundary to displace it — they order a batch,
    they do not unseat a certified row.

    There is NO combined generic max-sort over `prior + raw candidates`: the
    order is fixed at validate-all -> L2 -> columnize winners -> THEN L3. A
    retained prior row is COPIED DIRECTLY from its existing typed array by
    index; nothing is reconstructed, re-encoded, or mutated in place — the prior
    bundle is only ever READ.
    """
    if prior_arrays is None:
        return {name: np.array(new_arrays[name], copy=True) for name in _ARRAY_NAMES}

    prior_seeds = prior_arrays["seeds"].astype(np.int64)
    new_seeds = new_arrays["seeds"].astype(np.int64)
    prior_scores = prior_arrays["score"]
    new_scores = new_arrays["score"]
    prior_count = int(prior_seeds.shape[0])

    if prior_count == 0:
        keep_prior = np.zeros(0, dtype=np.int64)
        keep_new = np.arange(new_seeds.shape[0], dtype=np.int64)
    else:
        # The prior's seeds are proven strictly increasing by §9 validation, so
        # searchsorted is a correct exact-match lookup.
        pos = np.searchsorted(prior_seeds, new_seeds)
        in_range = pos < prior_count
        pos_clipped = np.clip(pos, 0, prior_count - 1)
        matched = in_range & (prior_seeds[pos_clipped] == new_seeds)

        # float32 vs float32 — the artifact's own domain. `>` is strict, so an
        # exact tie leaves `new_replaces` False and the prior row is retained.
        new_replaces = matched & (new_scores > prior_scores[pos_clipped])

        keep_new = np.where((~matched) | new_replaces)[0]
        superseded = pos_clipped[new_replaces]
        prior_mask = np.ones(prior_count, dtype=bool)
        prior_mask[superseded] = False
        keep_prior = np.where(prior_mask)[0]

    merged: Dict[str, np.ndarray] = {}
    for name, dtype in CANONICAL_ARRAY_CONTRACT:
        merged[name] = np.concatenate((
            prior_arrays[name][keep_prior],     # copied directly, never rebuilt
            new_arrays[name][keep_new],
        )).astype(dtype, copy=False)
    return merged


def _sort_by_seed(arrays: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Global seed-ascending ordering of the final bundle (§5).

    D3 preserves the caller's row order exactly and takes no ordering policy;
    imposing the FINAL order is D3.5's job and happens here, once, after L3.
    """
    order = np.argsort(arrays["seeds"], kind="stable")
    ordered = {name: arrays[name][order] for name in _ARRAY_NAMES}
    seeds = ordered["seeds"]
    if seeds.shape[0] > 1 and not bool(np.all(seeds[1:] > seeds[:-1])):
        duplicates = sorted({int(s) for s in seeds[1:][seeds[1:] == seeds[:-1]]})
        raise AccumulatorConsistencyError(
            f"the merged bundle is not strictly seed-ascending — duplicate "
            f"seed(s) {duplicates[:10]}. L2 yields one row per seed and L3 "
            f"either replaces or retains, so a duplicate is an internal "
            f"invariant break, not a caller error."
        )
    return ordered


# ---------------------------------------------------------------------------
# §7.1b / §8b / §9 — the prior generation: pointer, chain, sidecar, arrays.
# ---------------------------------------------------------------------------
def _parse_generation_dir_name(name: str) -> Tuple[str, str]:
    """Split `<generation_id>--<sidecar_sha256>` [REV3.1 D1].

    The hash is bound INTO THE NAME because the newest generation has no child
    to vouch for it: every historical generation is authenticated by its child's
    `parent_sidecar_sha256`, but without this the tip's `provenance.json` could
    be edited and the next run would hash whatever it found, with no
    authoritative expected value. Binding it here makes the atomic `current`
    swap the trust anchor for the live tip.
    """
    generation_id, separator, sidecar_hash = name.rpartition("--")
    if not separator or not generation_id:
        raise PriorGenerationError(
            f"generation directory name {name!r} is not of the required form "
            f"'<generation_id>--<sidecar_sha256>'."
        )
    if not _SHA256_RE.match(sidecar_hash):
        raise PriorGenerationError(
            f"generation directory name {name!r} does not embed a 64-character "
            f"lowercase hex sidecar SHA-256."
        )
    if not _GENERATION_ID_RE.match(generation_id):
        raise PriorGenerationError(
            f"generation directory name {name!r} embeds a malformed "
            f"generation_id {generation_id!r}."
        )
    return generation_id, sidecar_hash


def _read_sidecar(generation_dir: Path) -> Tuple[Dict[str, Any], bytes, str]:
    """Read + structurally validate a sidecar. Returns (payload, bytes, hash)."""
    sidecar_path = generation_dir / SIDECAR_NAME
    if not sidecar_path.is_file():
        raise PriorGenerationError(
            f"generation {generation_dir.name}: {SIDECAR_NAME} is missing. A "
            f"prior without a sidecar FAILS CLOSED — there is no filename-based "
            f"trust, and no historical artifact gets an exception (Ruling F)."
        )
    raw = _read_bytes(sidecar_path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PriorGenerationError(
            f"generation {generation_dir.name}: {SIDECAR_NAME} is not valid "
            f"UTF-8 JSON: {exc}"
        ) from exc
    _validate_sidecar_payload(payload, generation_dir.name)
    return payload, raw, hashlib.sha256(raw).hexdigest()


def _validate_sidecar_payload(payload: Any, label: str) -> None:
    """Exact key set and per-key types (§9)."""
    if not isinstance(payload, dict):
        raise PriorGenerationError(
            f"{label}: sidecar must be a JSON object, got "
            f"{type(payload).__name__}."
        )
    keys = set(payload)
    missing = _SIDECAR_REQUIRED_KEY_SET - keys
    extra = keys - _SIDECAR_REQUIRED_KEY_SET
    if missing:
        raise PriorGenerationError(
            f"{label}: sidecar is missing required key(s) {sorted(missing)}."
        )
    if extra:
        raise PriorGenerationError(
            f"{label}: sidecar carries unexpected key(s) {sorted(extra)}. "
            f"`sidecar_sha256` in particular is NOT a sidecar field — a file "
            f"cannot contain its own hash [REV3 C1]."
        )

    for key in ("artifact_schema_version", "sidecar_schema_version",
                "encoding_contract_version", "generation_id", "run_id",
                "prng_base", "repository_commit", "created_at"):
        _require_str(payload[key], f"{label}: sidecar {key!r}",
                     PriorGenerationError)
    for key in ("artifact_sha256", "canonical_map_hash"):
        value = payload[key]
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            raise PriorGenerationError(
                f"{label}: sidecar {key!r} must be a 64-character lowercase hex "
                f"SHA-256, got {value!r}."
            )
    for key in ("parent_generation_id", "parent_artifact_sha256",
                "parent_sidecar_sha256"):
        value = payload[key]
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise PriorGenerationError(
                f"{label}: sidecar {key!r} must be null or a nonempty str, got "
                f"{value!r}."
            )
        if key != "parent_generation_id" and not _SHA256_RE.match(value):
            raise PriorGenerationError(
                f"{label}: sidecar {key!r} must be null or a 64-character "
                f"lowercase hex SHA-256, got {value!r}."
            )
    if not isinstance(payload["repository_tree_clean"], bool):
        raise PriorGenerationError(
            f"{label}: sidecar 'repository_tree_clean' must be a bool, got "
            f"{payload['repository_tree_clean']!r}."
        )
    for key in ("row_count", "final_row_count", "l2_winner_count",
                "prior_row_count", "raw_candidate_count", "seed_start",
                "seed_count", "seed_end_exclusive"):
        _require_int(payload[key], f"{label}: sidecar {key!r}",
                     PriorGenerationError)
    modes = payload["skip_modes_executed"]
    if not isinstance(modes, list) or not modes or not all(
            isinstance(m, str) for m in modes):
        raise PriorGenerationError(
            f"{label}: sidecar 'skip_modes_executed' must be a nonempty list "
            f"of str, got {modes!r}."
        )

    # --- seed-domain v1.1 (D3.5-B): strict TYPE, then exact VALUE -----------
    # The type pass must run FIRST. `False == 0` in Python, so a Boolean
    # `seed_high16_prefix` or `seed_domain_start` would satisfy the value pin
    # below and only a bool-rejecting integer guard can catch it.
    for key in _SEED_DOMAIN_STR_FIELDS:
        _require_str(payload[key], f"{label}: sidecar {key!r}",
                     PriorGenerationError)
    for key in _SEED_DOMAIN_INT_FIELDS:
        _require_int(payload[key], f"{label}: sidecar {key!r}",
                     PriorGenerationError)
    for key in _SEED_DOMAIN_NULL_FIELDS:
        if payload[key] is not None:
            raise PriorGenerationError(
                f"{label}: sidecar {key!r} must be null, got {payload[key]!r}. "
                f"A v1.1 generation applies NO external transform between the "
                f"stored uint32 coordinate and the PRNG's internal state."
            )
    for key, expected in SEED_DOMAIN_FIELDS:
        if payload[key] != expected:
            raise PriorGenerationError(
                f"{label}: sidecar seed-domain field {key!r} is "
                f"{payload[key]!r}, but the v1.1 contract fixes it at "
                f"{expected!r}. Every seed-domain field is a module-owned "
                f"constant, never caller-supplied; a generation claiming a "
                f"stratum other than the one the uint32 domain wall actually "
                f"enforced FAILS CLOSED and is never silently migrated."
            )


def _load_prior_arrays(generation_dir: Path) -> Dict[str, np.ndarray]:
    """Load the prior 22-array bundle IN ITS STORED KEY ORDER.

    Building the dict from the archive's own order rather than from the frozen
    contract is deliberate: it is what lets `validate_array_bundle` actually
    test the stored order instead of testing a dict this function just sorted.
    """
    npz_path = generation_dir / CANONICAL_NPZ_NAME
    if not npz_path.is_file():
        raise PriorGenerationError(
            f"generation {generation_dir.name}: {CANONICAL_NPZ_NAME} is missing "
            f"— the generation directory is incomplete."
        )
    with np.load(npz_path) as handle:
        arrays = {name: handle[name] for name in handle.files}
    try:
        validate_array_bundle(arrays)
    except ValueError as exc:
        raise PriorGenerationError(
            f"generation {generation_dir.name}: the stored bundle violates the "
            f"frozen 22-array contract: {exc}"
        ) from exc
    return arrays


def _validate_prior_numeric_domains(
    arrays: Mapping[str, np.ndarray], label: str,
) -> None:
    """Full numeric-domain validation of the prior [REV3 C5].

    `validate_array_bundle` confirms keys, order, dimensions and dtypes; it does
    NOT establish semantic domains, so a NaN match rate or a fractional count
    would pass it and then be merged into a certified generation. The bounds
    below are the same ones D3 applies to a record on the way IN — restated here
    for a bundle arriving from disk, WITHOUT modifying D3.

    `bidirectional_selectivity` may legitimately exceed 1 (len(fwd)/max(len(rev),1)),
    so no generic <= 1 ceiling is applied to the ratio/weight group.
    """
    unit_interval = ("forward_matches", "reverse_matches", "score")
    counts = ("forward_count", "reverse_count", "bidirectional_count",
              "intersection_count", "forward_only_count", "reverse_only_count")
    nonnegative = ("intersection_ratio", "survivor_overlap_ratio",
                   "intersection_weight", "bidirectional_selectivity")

    for name in unit_interval:
        values = arrays[name].astype(np.float64)
        if not bool(np.all(np.isfinite(values))):
            raise PriorGenerationError(
                f"{label}: prior array {name!r} contains a non-finite value "
                f"(NaN or +/-inf)."
            )
        if values.size and not bool(np.all((values >= 0.0) & (values <= 1.0))):
            raise PriorGenerationError(
                f"{label}: prior array {name!r} leaves the frozen bound "
                f"[0.0, 1.0]."
            )
    for name in counts:
        values = arrays[name].astype(np.float64)
        if not bool(np.all(np.isfinite(values))):
            raise PriorGenerationError(
                f"{label}: prior count array {name!r} contains a non-finite "
                f"value."
            )
        if values.size and not bool(np.all(values >= 0.0)):
            raise PriorGenerationError(
                f"{label}: prior count array {name!r} contains a negative "
                f"count."
            )
        if values.size and not bool(np.all(values == np.floor(values))):
            raise PriorGenerationError(
                f"{label}: prior count array {name!r} contains a "
                f"non-integer-valued count. The six count columns are float32 "
                f"ONLY because the frozen NPZ schema requires it; they remain "
                f"logical counts."
            )
    for name in nonnegative:
        values = arrays[name].astype(np.float64)
        if not bool(np.all(np.isfinite(values))):
            raise PriorGenerationError(
                f"{label}: prior array {name!r} contains a non-finite value."
            )
        if values.size and not bool(np.all(values >= 0.0)):
            raise PriorGenerationError(
                f"{label}: prior array {name!r} is negative; it is bounded "
                f"below by 0.0 (no <= 1 ceiling applies — "
                f"bidirectional_selectivity may exceed 1 legitimately)."
            )


def _validate_prior_identity(
    arrays: Mapping[str, np.ndarray], prng_base: str, label: str,
) -> None:
    """The prior wall (§8a) — decoding successfully is NOT sufficient.

        skip_mode == constant  -> prng_type ID must encode sidecar.prng_base
        skip_mode == variable  -> prng_type ID must encode prng_base + "_hybrid"

    A row whose IDs are both individually valid but jointly inconsistent (a
    `constant` row carrying `<base>_hybrid`) fails closed: it would otherwise be
    silently relabelled by the generation it is merged into.
    """
    expected = {"constant": prng_base, "variable": prng_base + "_hybrid"}
    for row, (skip_id, type_id) in enumerate(
            zip(arrays["skip_mode"], arrays["prng_type"])):
        try:
            skip_mode = decode_skip_mode(int(skip_id))
        except ValueError as exc:
            raise PriorGenerationError(
                f"{label}: prior row {row} carries an undecodable skip_mode id "
                f"{int(skip_id)}: {exc}"
            ) from exc
        try:
            prng_type = decode_prng_type(int(type_id))
        except ValueError as exc:
            raise PriorGenerationError(
                f"{label}: prior row {row} carries an undecodable prng_type id "
                f"{int(type_id)}: {exc}"
            ) from exc
        if skip_mode not in expected:
            raise PriorGenerationError(
                f"{label}: prior row {row} decodes to skip_mode "
                f"{skip_mode!r}, outside the canonical vocabulary "
                f"{list(CANONICAL_SKIP_MODES)}."
            )
        if prng_type != expected[skip_mode]:
            raise PriorGenerationError(
                f"{label}: prior row {row} is identity-inconsistent — "
                f"skip_mode {skip_mode!r} under prng_base {prng_base!r} "
                f"requires prng_type {expected[skip_mode]!r}, got "
                f"{prng_type!r}. Valid-but-inconsistent IDs fail closed."
            )


def _validate_current_pointer(accumulator_root: Path) -> Path:
    """§7.1b current-generation validation, in the frozen order.

    Older generations stay authenticated through their children's
    `parent_sidecar_sha256`; this covers the TIP, which by definition has no
    child. Any mismatch, an escape outside `generations/`, or a non-directory
    target fails closed.
    """
    pointer = accumulator_root / CURRENT_POINTER_NAME
    if not os.path.islink(pointer):
        raise PriorGenerationError(
            f"{pointer} exists but is not a symlink. The current pointer is the "
            f"trust anchor for the live tip and must be a symlink into "
            f"{GENERATIONS_DIRNAME}/."
        )

    # 1. read the link WITHOUT following an arbitrary external target
    raw_target = os.readlink(pointer)
    if os.path.isabs(raw_target):
        raise PriorGenerationError(
            f"the current pointer targets the absolute path {raw_target!r}; "
            f"only a relative '{GENERATIONS_DIRNAME}/<generation>' target is "
            f"accepted."
        )
    # 2. require a DIRECT CHILD of generations/
    parts = PurePosixPath(raw_target).parts
    if len(parts) != 2 or parts[0] != GENERATIONS_DIRNAME or parts[1] in (
            ".", ".."):
        raise PriorGenerationError(
            f"the current pointer target {raw_target!r} is not a direct child "
            f"of {GENERATIONS_DIRNAME}/. A target that escapes the generations "
            f"directory fails closed."
        )
    # 3. parse <generation_id> and <expected_sidecar_sha256> from the name
    generation_id, expected_sidecar_hash = _parse_generation_dir_name(parts[1])

    generation_dir = accumulator_root / GENERATIONS_DIRNAME / parts[1]
    if os.path.islink(generation_dir) or not generation_dir.is_dir():
        raise PriorGenerationError(
            f"the current pointer target {raw_target!r} is not a real "
            f"directory."
        )

    # 4/5. hash the STORED provenance.json and require it to equal the hash
    #      embedded in the pointer target.
    sidecar, _raw, actual_hash = _read_sidecar(generation_dir)
    if actual_hash != expected_sidecar_hash:
        raise PriorGenerationError(
            f"generation {parts[1]}: the stored {SIDECAR_NAME} hashes to "
            f"{actual_hash}, but the pointer target embeds "
            f"{expected_sidecar_hash}. The live tip's provenance metadata has "
            f"been modified since publication — failing closed."
        )
    # 6. the parsed id must agree with the sidecar's own claim
    if sidecar["generation_id"] != generation_id:
        raise PriorGenerationError(
            f"generation {parts[1]}: the directory name claims generation_id "
            f"{generation_id!r} but the sidecar claims "
            f"{sidecar['generation_id']!r}."
        )
    return generation_dir


def _validate_chain(
    generations_dir: Path, sidecar: Mapping[str, Any],
) -> None:
    """Follow `parent_*` to a clean-start root, verifying EVERY link (§8b).

    Without this the parent hashes are recorded metadata rather than an enforced
    provenance chain — precisely the property whose absence made the historical
    accumulator uncertifiable (Ruling F).
    """
    seen = {sidecar["generation_id"]}
    current = sidecar
    while True:
        parent_id = current["parent_generation_id"]
        parent_artifact = current["parent_artifact_sha256"]
        parent_sidecar_hash = current["parent_sidecar_sha256"]
        present = [v is not None
                   for v in (parent_id, parent_artifact, parent_sidecar_hash)]
        if not any(present):
            return                              # clean-start root reached
        if not all(present):
            raise PriorGenerationError(
                f"generation {current['generation_id']}: the parent reference "
                f"is partially null (parent_generation_id={parent_id!r}, "
                f"parent_artifact_sha256={parent_artifact!r}, "
                f"parent_sidecar_sha256={parent_sidecar_hash!r}). A clean-start "
                f"root has ALL parent_* fields null; anything else must name a "
                f"complete parent."
            )
        if parent_id in seen:
            raise PriorGenerationError(
                f"provenance chain repeats generation id {parent_id!r} — a "
                f"cycle or a duplicated generation. Failing closed."
            )
        parent_dir = generations_dir / f"{parent_id}--{parent_sidecar_hash}"
        if os.path.islink(parent_dir) or not parent_dir.is_dir():
            raise PriorGenerationError(
                f"generation {current['generation_id']}: its declared ancestor "
                f"{parent_dir.name} is missing. The chain must reach a "
                f"clean-start root without a gap."
            )
        parent_sidecar, _raw, actual_sidecar_hash = _read_sidecar(parent_dir)
        if actual_sidecar_hash != parent_sidecar_hash:
            raise PriorGenerationError(
                f"ancestor {parent_dir.name}: its stored {SIDECAR_NAME} hashes "
                f"to {actual_sidecar_hash}, but its child records "
                f"{parent_sidecar_hash}. The ancestor's provenance METADATA has "
                f"been modified — linking only the payload would have missed it."
            )
        if parent_sidecar["generation_id"] != parent_id:
            raise PriorGenerationError(
                f"ancestor {parent_dir.name}: sidecar generation_id "
                f"{parent_sidecar['generation_id']!r} disagrees with the "
                f"reference {parent_id!r}."
            )
        # --- §5 per-link semantic contract (D3.5-B) ------------------------
        # Topology and the publication mechanism are unchanged; this is a
        # comparison loop, not a restructure.
        for key in _LINEAGE_INVARIANT_KEYS:
            if current[key] != parent_sidecar[key]:
                raise PriorGenerationError(
                    f"generation {current['generation_id']}: lineage field "
                    f"{key!r} is {current[key]!r} but its declared parent "
                    f"{parent_id!r} records {parent_sidecar[key]!r}. A "
                    f"certified lineage is homogeneous in prng_base, in all "
                    f"three contract versions, in the canonical map hash and "
                    f"in every seed-domain field. A different seed-domain "
                    f"contract requires a NEW CLEAN ROOT — it is never linked "
                    f"as a certified parent. Failing closed."
                )
        parent_npz = parent_dir / CANONICAL_NPZ_NAME
        if not parent_npz.is_file():
            raise PriorGenerationError(
                f"ancestor {parent_dir.name}: {CANONICAL_NPZ_NAME} is missing."
            )
        actual_artifact_hash = _sha256_file(parent_npz)
        if actual_artifact_hash != parent_artifact:
            raise PriorGenerationError(
                f"ancestor {parent_dir.name}: its artifact hashes to "
                f"{actual_artifact_hash}, but its child records "
                f"{parent_artifact}."
            )
        seen.add(parent_id)
        current = parent_sidecar


@dataclass(frozen=True)
class _PriorGeneration:
    directory: Path
    sidecar: Dict[str, Any]
    sidecar_sha256: str
    arrays: Dict[str, np.ndarray]


def _resolve_prior_directory(
    accumulator_root: Path, prior_generation_dir: Optional[Path],
) -> Optional[Path]:
    """§8b prior selection — all four cases frozen [REV3.1 D2].

        current absent  + prior omitted           -> clean start
        current absent  + prior supplied          -> FAIL CLOSED
        current present + prior omitted           -> AUTOMATICALLY use current
        current present + matching prior supplied -> use it
        current present + nonmatching prior       -> FAIL CLOSED

    Omitting the optional argument must NOT silently start a new lineage: that
    is exactly how a fork gets published while every individual write looks
    correct.
    """
    pointer = accumulator_root / CURRENT_POINTER_NAME
    if not os.path.lexists(pointer):
        if prior_generation_dir is not None:
            raise PriorGenerationError(
                f"a prior generation {prior_generation_dir} was supplied, but "
                f"there is no `current` pointer at {pointer}. Merging against a "
                f"detached generation would fork the lineage — failing closed."
            )
        return None

    target = _validate_current_pointer(accumulator_root)
    if prior_generation_dir is None:
        return target

    supplied = Path(prior_generation_dir)
    try:
        same = supplied.resolve(strict=True) == target.resolve(strict=True)
    except OSError as exc:
        raise PriorGenerationError(
            f"the supplied prior generation {supplied} could not be resolved: "
            f"{exc}"
        ) from exc
    if not same:
        raise PriorGenerationError(
            f"the supplied prior generation {supplied} does not resolve to the "
            f"live `current` target {target}. Production prior selection is "
            f"pinned to the live pointer; a detached or stale generation fails "
            f"closed."
        )
    return target


def _load_prior_generation(
    generations_dir: Path, generation_dir: Path, prng_base: str,
) -> _PriorGeneration:
    """The full §9 checklist for a supplied prior generation."""
    label = generation_dir.name
    sidecar, _raw, sidecar_hash = _read_sidecar(generation_dir)

    npz_path = generation_dir / CANONICAL_NPZ_NAME
    if not npz_path.is_file():
        raise PriorGenerationError(
            f"{label}: {CANONICAL_NPZ_NAME} is missing — the generation "
            f"directory is incomplete."
        )
    actual_artifact_hash = _sha256_file(npz_path)
    if actual_artifact_hash != sidecar["artifact_sha256"]:
        raise PriorGenerationError(
            f"{label}: the stored artifact hashes to {actual_artifact_hash} but "
            f"its sidecar records {sidecar['artifact_sha256']}."
        )
    if not (generation_dir / ALL_NPZ_NAME).is_file():
        raise PriorGenerationError(
            f"{label}: {ALL_NPZ_NAME} is missing — the generation directory is "
            f"incomplete."
        )

    for key, expected in (
            ("artifact_schema_version", ARTIFACT_SCHEMA_VERSION),
            ("sidecar_schema_version", SIDECAR_SCHEMA_VERSION),
            ("encoding_contract_version", ENCODING_CONTRACT_VERSION),
            ("canonical_map_hash", canonical_map_hash())):
        if sidecar[key] != expected:
            raise PriorGenerationError(
                f"{label}: sidecar {key} is {sidecar[key]!r}, this build "
                f"expects {expected!r}. A generation written under a different "
                f"contract is never silently migrated."
            )
    if sidecar["prng_base"] != prng_base:
        raise PriorGenerationError(
            f"{label}: sidecar prng_base {sidecar['prng_base']!r} differs from "
            f"the current accumulator identity {prng_base!r}. Different "
            f"skip_modes_executed sets across generations are allowed; "
            f"different prng_base values are not."
        )

    arrays = _load_prior_arrays(generation_dir)
    seeds = arrays["seeds"]
    row_count = int(seeds.shape[0])
    if row_count > 1:
        widened = seeds.astype(np.int64)
        if not bool(np.all(widened[1:] > widened[:-1])):
            raise PriorGenerationError(
                f"{label}: prior seeds are not strictly increasing and unique. "
                f"A certified generation carries exactly one row per seed in "
                f"global ascending order."
            )
    if row_count != sidecar["row_count"] or row_count != sidecar["final_row_count"]:
        raise PriorGenerationError(
            f"{label}: the stored bundle has {row_count} rows but its sidecar "
            f"records row_count={sidecar['row_count']} / "
            f"final_row_count={sidecar['final_row_count']}."
        )

    _validate_prior_identity(arrays, prng_base, label)
    _validate_prior_numeric_domains(arrays, label)
    _validate_chain(generations_dir, sidecar)
    return _PriorGeneration(generation_dir, sidecar, sidecar_hash, arrays)


# ---------------------------------------------------------------------------
# §7 — publication.
# ---------------------------------------------------------------------------
def _bootstrap_root_aliases(output_root: Path) -> None:
    """§7.1a — both root aliases exist as symlinks BEFORE `current` is committed.

    For F29's single-commit property to hold on the FIRST generation the aliases
    must already be in place (dangling until the swap), otherwise generation 1
    would need separate post-commit filesystem changes and "one commit point"
    would simply be false there.

    NO EXISTING REGULAR FILE MAY BE SILENTLY REPLACED. That matters especially
    because the historical root artifacts were explicitly removed under Ruling
    F: a regular file reappearing at those paths means something wrote outside
    the finalizer, and the run must stop rather than overwrite it.
    """
    for name in (ALL_NPZ_NAME, BINARY_NPZ_NAME):
        alias = output_root / name
        expected = f"{ACCUMULATOR_DIRNAME}/{CURRENT_POINTER_NAME}/{name}"
        if not os.path.lexists(alias):
            os.symlink(expected, alias)
            continue
        if not os.path.islink(alias):
            raise PublicationError(
                f"{alias} exists as a regular file or directory, not the "
                f"expected compatibility symlink. The historical root artifacts "
                f"were removed under Ruling F, so something wrote outside the "
                f"finalizer — failing closed rather than replacing it."
            )
        actual = os.readlink(alias)
        if actual != expected:
            raise PublicationError(
                f"{alias} is a symlink to {actual!r}, expected {expected!r}. "
                f"A wrong-target alias fails closed."
            )
    _fsync_dir(output_root)


def _publish_generation(
    *,
    accumulator_root: Path,
    generations_dir: Path,
    generation_id: str,
    arrays: Mapping[str, np.ndarray],
    sidecar_payload: Dict[str, Any],
) -> Tuple[Path, str, str]:
    """The binding publication order of §7.2. Returns (dir, artifact, sidecar).

    Any failure BEFORE step 13 leaves the previous generation active and nothing
    published. A failure at step 14 — after the swap — is a durability failure,
    not a "nothing happened", and gets its own exception type.
    """
    tmp_dir = generations_dir / f".tmp-{generation_id}"
    if os.path.lexists(tmp_dir):
        # Unreferenced staging debris from a crashed run: `current` can never
        # point at a `.tmp-` name, whose shape lacks the mandatory `--<hash>`.
        _remove_tree(tmp_dir)

    committed = False
    try:
        # 1. staging directory, on the SAME filesystem as its final location
        _mkdir(tmp_dir)

        # 2. write both NPZ names
        all_path = tmp_dir / ALL_NPZ_NAME
        binary_path = tmp_dir / BINARY_NPZ_NAME
        _write_npz(all_path, arrays)
        try:
            # One inode under two names — byte-identity is then structural, not
            # a coincidence of two compressors producing the same bytes.
            os.link(all_path, binary_path)
        except OSError:
            shutil.copyfile(all_path, binary_path)

        # 3. validate both, and verify byte-identical payloads
        for path in (all_path, binary_path):
            with np.load(path) as handle:
                stored = {name: handle[name] for name in handle.files}
            try:
                validate_array_bundle(stored)
            except ValueError as exc:
                raise PublicationError(
                    f"the staged artifact {path.name} does not satisfy the "
                    f"frozen 22-array contract: {exc}"
                ) from exc
        if _read_bytes(all_path) != _read_bytes(binary_path):
            raise PublicationError(
                f"{ALL_NPZ_NAME} and {BINARY_NPZ_NAME} do not carry "
                f"byte-identical payloads."
            )

        # 4. hash the canonical NPZ
        artifact_sha256 = _sha256_file(tmp_dir / CANONICAL_NPZ_NAME)
        sidecar_payload["artifact_sha256"] = artifact_sha256
        _validate_sidecar_payload(sidecar_payload, f".tmp-{generation_id}")

        # 5. serialize the canonical sidecar bytes (no `sidecar_sha256` field)
        sidecar_bytes = _canonical_json_bytes(sidecar_payload)

        # 6. write all sidecar bytes, flush, fsync provenance.json
        sidecar_path = tmp_dir / SIDECAR_NAME
        _write_and_fsync_bytes(sidecar_path, sidecar_bytes)

        # 7. REOPEN and hash the STORED bytes [REV3.1 D3]. The value must
        #    describe the bytes actually on disk, never the pre-write buffer.
        stored_sidecar_bytes = _read_bytes(sidecar_path)
        sidecar_sha256 = hashlib.sha256(stored_sidecar_bytes).hexdigest()

        # 8. fsync NPZ file data
        _fsync_file(all_path)
        _fsync_file(binary_path)

        # 9. fsync the temporary generation directory
        _fsync_dir(tmp_dir)

        # 10. atomic rename into the HASH-BOUND final name. It is only knowable
        #     after step 7, which is why the rename cannot come earlier.
        final_dir = generations_dir / f"{generation_id}--{sidecar_sha256}"
        if os.path.lexists(final_dir):
            raise PublicationError(
                f"generation directory {final_dir.name} already exists; a "
                f"generation is immutable and is never overwritten."
            )
        _atomic_rename(tmp_dir, final_dir)

        # 11. fsync generations/
        _fsync_dir(generations_dir)

        # 12/13. create the temporary pointer, then ATOMICALLY REPLACE `current`
        #        — THE SINGLE COMMIT POINT.
        _replace_symlink(
            f"{GENERATIONS_DIRNAME}/{final_dir.name}",
            accumulator_root / CURRENT_POINTER_NAME,
            accumulator_root / f".{CURRENT_POINTER_NAME}.tmp",
        )
        committed = True

        # 14. fsync the accumulator root. Past this line publication has already
        #     happened logically, so a failure here is a DURABILITY failure.
        try:
            _fsync_dir(accumulator_root)
        except Exception as exc:
            raise PublicationDurabilityError(
                f"the `current` pointer was atomically replaced (the logical "
                f"commit succeeded) but fsync of {accumulator_root} failed: "
                f"{exc}. No RunArtifactResult is returned and no fallback is "
                f"invoked; the next invocation performs recovery validation of "
                f"the pointer, directory, artifact and sidecar before "
                f"proceeding."
            ) from exc

        return final_dir, artifact_sha256, sidecar_sha256
    except BaseException:
        if not committed and os.path.isdir(tmp_dir):
            # Unreferenced staging directory: neither accepted nor returned.
            _remove_tree(tmp_dir)
        raise


# ---------------------------------------------------------------------------
# The single public entry point.
# ---------------------------------------------------------------------------
def finalize_run(
    candidates: Iterable[Mapping[str, object]],
    *,
    output_root: Path,
    run_id: str,
    prng_base: str,
    skip_modes_executed: Sequence[str],
    seed_start: int,
    seed_count: int,
    repository_commit: str,
    repository_tree_clean: bool,
    prior_generation_dir: Optional[Path] = None,
) -> RunArtifactResult:
    """Finalize one run into an immutable, chain-authenticated generation.

    See the module docstring for the binding pipeline order. Every failure mode
    is fail-closed: the previous certified generation stays current and NO
    fallback writer is invoked, ever.

    Raises:
        RunParameterError, CandidateValidationError, CoverageValidationError,
        RunIdentityError, AccumulatorConsistencyError, PriorGenerationError,
        PublicationError: nothing was published; `current` is unchanged.
        PublicationDurabilityError: the pointer swap COMMITTED but its
            durability fsync failed; no result is returned [REV3.1 D4].
    """
    started = time.monotonic()
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

    # --- argument contract --------------------------------------------------
    output_root = Path(output_root)
    if not output_root.is_dir():
        raise RunParameterError(
            f"output_root {output_root} is not an existing directory."
        )
    run_id = _require_str(run_id, "run_id", RunParameterError)
    repository_commit = _require_str(
        repository_commit, "repository_commit", RunParameterError)
    if not isinstance(repository_tree_clean, bool):
        raise RunParameterError(
            f"repository_tree_clean must be a bool, got "
            f"{repository_tree_clean!r}."
        )
    if not repository_tree_clean:
        # §7.3, binding: the first certified production baseline must not claim
        # a commit SHA while running uncommitted source.
        raise RunParameterError(
            f"repository_tree_clean is False: the working tree has uncommitted "
            f"changes, so a certified generation cannot honestly claim commit "
            f"{repository_commit}. Commit the tree, then finalize."
        )

    prng_base, modes = _validate_run_identity(prng_base, skip_modes_executed)
    start, count, end_exclusive = _validate_declared_coverage(
        seed_start, seed_count)

    # --- §3: materialize ONCE, then validate EVERY raw candidate ------------
    # The iterable may be a generator and D3 consumes it in a single pass, so
    # the list is the one materialization; L2 and coverage both read it.
    raw_candidates: List[Mapping[str, Any]] = list(candidates)
    _validate_raw_candidates(raw_candidates)

    # --- coverage + run-identity walls, both BEFORE L2 ----------------------
    _validate_candidate_coverage(raw_candidates, start, end_exclusive)
    _validate_candidate_identity(raw_candidates, prng_base, modes)

    # --- §4 L2 (record domain) -> D3 columnization --------------------------
    winners = _select_l2_winners(raw_candidates)
    try:
        winner_arrays = records_to_arrays(winners)
    except ValueError as exc:                       # pragma: no cover - §3 wall
        raise CandidateValidationError(
            f"an L2 winner failed columnization after passing the raw wall: "
            f"{exc}"
        ) from exc

    # --- prior selection, validation, and §5 L3 (ARRAY domain) --------------
    accumulator_root = output_root / ACCUMULATOR_DIRNAME
    generations_dir = accumulator_root / GENERATIONS_DIRNAME
    prior_dir = None
    if accumulator_root.is_dir():
        prior_dir = _resolve_prior_directory(accumulator_root,
                                             prior_generation_dir)
    elif prior_generation_dir is not None:
        raise PriorGenerationError(
            f"a prior generation {prior_generation_dir} was supplied, but "
            f"{accumulator_root} does not exist. Failing closed rather than "
            f"merging against a detached generation."
        )

    prior = None
    if prior_dir is not None:
        prior = _load_prior_generation(generations_dir, prior_dir, prng_base)

    merged = _l3_merge(winner_arrays,
                       prior.arrays if prior is not None else None)
    final_arrays = _sort_by_seed(merged)
    try:
        validate_array_bundle(final_arrays)
    except ValueError as exc:                       # pragma: no cover
        raise PublicationError(
            f"the final merged bundle violates the frozen 22-array contract "
            f"and will NOT be published: {exc}"
        ) from exc

    # --- §7 publication -----------------------------------------------------
    generation_id = f"gen-{created_at.replace(':', '').replace('.', '').replace('-', '')}-{_sanitize(run_id)}"
    if "--" in generation_id or not _GENERATION_ID_RE.match(generation_id):
        raise RunParameterError(         # pragma: no cover - _sanitize prevents
            f"derived generation_id {generation_id!r} is not a valid, "
            f"hash-bindable directory component."
        )

    prior_row_count = 0 if prior is None else int(prior.arrays["seeds"].shape[0])
    final_row_count = int(final_arrays["seeds"].shape[0])

    sidecar_payload: Dict[str, Any] = {
        "generation_id": generation_id,
        "artifact_sha256": "0" * 64,        # filled at step 4, before serialize
        "parent_generation_id": None if prior is None else prior.sidecar["generation_id"],
        "parent_artifact_sha256": None if prior is None else prior.sidecar["artifact_sha256"],
        "parent_sidecar_sha256": None if prior is None else prior.sidecar_sha256,
        "repository_commit": repository_commit,
        "repository_tree_clean": repository_tree_clean,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "encoding_contract_version": ENCODING_CONTRACT_VERSION,
        "canonical_map_hash": canonical_map_hash(),
        "row_count": final_row_count,
        "run_id": run_id,
        "prng_base": prng_base,
        "skip_modes_executed": list(modes),
        "seed_start": start,
        "seed_count": count,
        "seed_end_exclusive": end_exclusive,
        # Seed-domain v1.1 (D3.5-B). These describe the STRATUM DOMAIN
        # [0, 2**32) and are distinct from the per-run `seed_start` /
        # `seed_count` / `seed_end_exclusive` coverage fields directly above:
        # both sets coexist and neither replaces the other. Every value is
        # inserted INTERNALLY from a module-owned constant — no argument, no
        # environment, no inference from the candidates, no copy from a prior.
        "seed_semantics": SEED_SEMANTICS,
        "seed_storage_dtype": SEED_STORAGE_DTYPE,
        "seed_effective_bits": SEED_EFFECTIVE_BITS,
        "seed_high16_prefix": SEED_HIGH16_PREFIX,
        "seed_domain_contract": SEED_DOMAIN_CONTRACT,
        "seed_domain_start": SEED_DOMAIN_START,
        "seed_domain_end_exclusive": SEED_DOMAIN_END_EXCLUSIVE,
        "exhaustive_over": EXHAUSTIVE_OVER,
        "external_seed_transform": EXTERNAL_SEED_TRANSFORM,
        "raw_candidate_count": len(raw_candidates),
        "l2_winner_count": len(winners),
        "prior_row_count": prior_row_count,
        "final_row_count": final_row_count,
        "created_at": created_at,
    }

    accumulator_root.mkdir(parents=True, exist_ok=True)
    generations_dir.mkdir(parents=True, exist_ok=True)
    _fsync_dir(accumulator_root)

    # §7.1a — the aliases come FIRST, so the step-13 swap makes artifact,
    # sidecar and both root names valid simultaneously.
    _bootstrap_root_aliases(output_root)

    final_dir, artifact_sha256, sidecar_sha256 = _publish_generation(
        accumulator_root=accumulator_root,
        generations_dir=generations_dir,
        generation_id=generation_id,
        arrays=final_arrays,
        sidecar_payload=sidecar_payload,
    )

    # 15. only NOW is a result object constructed.
    return RunArtifactResult(
        generation_id=generation_id,
        generation_dir=final_dir,
        all_npz_path=final_dir / ALL_NPZ_NAME,
        binary_npz_path=final_dir / BINARY_NPZ_NAME,
        sidecar_path=final_dir / SIDECAR_NAME,
        artifact_sha256=artifact_sha256,
        sidecar_sha256=sidecar_sha256,
        parent_generation_id=sidecar_payload["parent_generation_id"],
        parent_artifact_sha256=sidecar_payload["parent_artifact_sha256"],
        parent_sidecar_sha256=sidecar_payload["parent_sidecar_sha256"],
        repository_commit=repository_commit,
        repository_tree_clean=repository_tree_clean,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        sidecar_schema_version=SIDECAR_SCHEMA_VERSION,
        encoding_contract_version=ENCODING_CONTRACT_VERSION,
        canonical_map_hash=sidecar_payload["canonical_map_hash"],
        run_id=run_id,
        prng_base=prng_base,
        skip_modes_executed=modes,
        seed_start=start,
        seed_count=count,
        seed_end_exclusive=end_exclusive,
        raw_candidate_count=len(raw_candidates),
        l2_winner_count=len(winners),
        prior_row_count=prior_row_count,
        final_row_count=final_row_count,
        created_at=created_at,
        elapsed_seconds=time.monotonic() - started,
    )


def _sanitize(value: str) -> str:
    """Reduce a run_id to a safe directory-name component.

    The result can never contain `--`, so `<generation_id>--<sidecar_sha256>`
    stays unambiguously parseable from the right.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._]", "_", value)
    return cleaned or "run"
