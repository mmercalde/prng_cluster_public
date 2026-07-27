#!/usr/bin/env python3
"""
range_miner_npz_writer.py — S172 Phase 5, Deliverable D1.1.

The backend-independent four-population ASSEMBLY ENGINE plus the concrete
`Phase5Sink` the Phase-4 coordinator drives
(docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D1.md REV5 §4-§8).

What this module IS:
  * `assemble_trial(run_id, manifests)` — the ONE derivation D4/D5 will later
    reuse: per-manifest identity validation (§5.1), cross-manifest 11-field
    consistency + phase-set completeness (§5.2), commit-time spool read +
    identity + container + semantic validation (§5.3), the four directional
    `seed -> match_rate` maps with a hard duplicate invariant (§5.4), the two
    intersections and the frozen 24-field canonical records (§5.5/§6).
  * `AssemblingPhase5Sink` — accumulates published manifests, assembles exactly
    once on a successful commit, discharges an abort synchronously (L7).
  * `MinerTrialAssembly` — the stable cross-deliverable result object (§7).

What this module is explicitly NOT (Beta-bound to later deliverables):
  * NO NPZ writing of any kind — no `np.savez*`, no 22-array construction (D3/D4);
  * NO backend parallelism / process pools (D5);
  * NO physical temp-file cleanup or benchmark instrumentation (D7);
  * NO durable cross-process sink reconstruction (§4.0 [TB-D1-B2]) — a failed
    assembly is retryable only while THIS sink instance retains its manifests;
  * NO production wiring / adapter into window_optimizer_integration_final.py (D6).

Records store PRNG/skip identity as STRINGS; the numeric uint8 encoding is D3.
`utils/prng_encoding` is nonetheless the single source of truth used to VALIDATE
every distinct `prng_type` / `skip_mode` (§5.4) — the numeric result is discarded.

Every failure here is a producer/contract defect: fail closed, never resolve
silently, never catch-and-continue inside the engine.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any, Dict, Iterable, Iterator, List, Mapping, NoReturn, Optional, Sequence,
    Tuple, Type, Union,
)

import numpy as np

from miner.range_miner_coordinator import (
    MinerMetadataError,
    Phase5Sink,
    _canonicalize_trial_context,
    validate_trial_metadata,
    workflow_phase_semantics,
    workflow_stages_for,
)
from miner.range_miner_worker import SUBSTRIPE_SCHEMA_VERSION
# D3.25 [C3]: the frozen 24-field record constant and the one shared canonical
# record builder now live beside each other in `utils/canonical_records.py`, so
# PWC/ZMQ can reach the SAME derivation without importing from `miner/` (the
# dependency direction is generic utilities <- {miner, PWC, ZMQ, adapter}).
# Both names are re-exported below, unchanged, for every existing D1 caller.
from utils.canonical_records import CANONICAL_RECORD_FIELDS, build_mode_records
from utils.prng_encoding import encode_prng_type, encode_skip_mode

__all__ = [
    "CANONICAL_RECORD_FIELDS",
    "MinerTrialAssembly",
    "AssemblingPhase5Sink",
    "assemble_trial",
    "prepare_trial_assembly",
    "ValidatedSpoolProjection",
    "SEED_ENCODING_INT64",
    "SEED_ENCODING_SIGNED_BYTES",
    "SEED_ENCODINGS",
    "build_validated_projection",
    "projection_seeds",
    "CapturedSpoolReadError",
    "SpoolReadOutcome",
    "CANONICAL_SPOOL_READ_ERRORS",
    "capture_spool_read_error",
    "raise_captured_spool_error",
    "read_and_validate_spool",
    "merge_validated_spools",
    "ManifestReplayConflict",
    "AssemblyConsistencyError",
    "PhaseIdentityError",
    "SpoolIdentityError",
    "DirectionalDuplicateError",
    "AssemblyStateError",
]


# ---------------------------------------------------------------------------
# §8 — D1 exception types. All are contract defects; none is recoverable inside
# the engine.
# ---------------------------------------------------------------------------
class ManifestReplayConflict(Exception):
    """§4.2.4: the same `event_id` republished with DIFFERENT content, or a
    different `event_id` claiming an already-occupied logical shard slot
    `(run_id, stripe_id, sub_index)` — the latter even when the bytes and SHA are
    byte-identical (two event ids for one logical shard is a producer defect)."""


class AssemblyConsistencyError(Exception):
    """§5.2: the manifests of one run disagree on the 11-field canonical trial
    context, or the set of workflow phases present is not exactly {1,2} (constant
    only) or {1,2,3,4} (both modes) — an incomplete directional pairing."""


class PhaseIdentityError(Exception):
    """§5.1: a per-manifest identity check failed (run_id, manifest-vs-metadata
    workflow_phase, the lifted dataset/residue provenance copies, direction /
    skip_mode vs `workflow_phase_semantics`, family_name vs `workflow_stages_for`,
    prng_type vs skip_mode, or threshold_used vs the directional threshold)."""


class SpoolIdentityError(Exception):
    """§5.3: a staged spool failed size / SHA-256 / JSON-decode / container /
    schema_version / stripe_id / sub_index / tuple-shape / semantic validation.
    A correctly hashed but misassociated, malformed, or semantically invalid
    spool must never enter a directional map."""


class DirectionalDuplicateError(Exception):
    """§5.4: one seed appeared twice inside a SINGLE directional population.

    That is a producer/coverage defect (overlapping sub-stripes, a superseded
    attempt republished, a mis-tiled stripe), never a dedup opportunity — Phase 5
    must NOT resolve it by max match_rate (that rule belongs only to the D3
    cross-trial accumulator boundary, v1_4_4 §4.3).

    The structured attributes below are REAL attributes (not message text) so D2
    can assert on them directly."""

    def __init__(self, message: str, *, run_id: str, workflow_phase: int,
                 direction: str, skip_mode: str, seed: int,
                 first_stripe: str, first_sub_index: int, first_attempt: int,
                 first_match_rate: float, dup_stripe: str, dup_sub_index: int,
                 dup_attempt: int, dup_match_rate: float):
        super().__init__(message)
        self.run_id = run_id
        self.workflow_phase = workflow_phase
        self.direction = direction
        self.skip_mode = skip_mode
        self.seed = seed
        self.first_stripe = first_stripe
        self.first_sub_index = first_sub_index
        self.first_attempt = first_attempt
        self.first_match_rate = first_match_rate
        self.dup_stripe = dup_stripe
        self.dup_sub_index = dup_sub_index
        self.dup_attempt = dup_attempt
        self.dup_match_rate = dup_match_rate


class AssemblyStateError(Exception):
    """Sink lifecycle violation: commit for a tombstoned run; a DIFFERENT commit
    `event_id` for an already-committed run; commit with zero retained manifests;
    a new event / new logical shard published to an already-committed run."""


# ---------------------------------------------------------------------------
# §6 — the frozen 24-field canonical record.
#
# RELOCATED in D3.25 [C3] to `utils/canonical_records.py`, beside the builder
# that produces it, and imported above. Order and membership are unchanged and
# still reproduce the LIVE Step-1 insertion order exactly: seed/rates/score
# followed by `metadata_base`
# (window_optimizer_integration_final.py:683-694 + :652-676 for constant;
# identically :785-796 + :756-780 for the hybrid/variable block).
#
# `threshold_used` is deliberately NOT a 25th field: it is manifest identity /
# validation metadata (§5.1), not a record field.
#
# The re-export is deliberate: `CANONICAL_RECORD_FIELDS` stays in this module's
# `__all__` so every existing D1 importer — including D1.1's G9 harness, which
# compares it against an INDEPENDENTLY hand-transcribed oracle — is untouched
# by the move. Relocating the constant does not authorize importing it into a
# test oracle.
# ---------------------------------------------------------------------------

# The 11-field canonical trial context (9 trial-global + 2 provenance) every
# manifest of one run must agree on (§5.2), canonicalized through the
# coordinator's own `_canonicalize_trial_context` — never a second canonicalizer.
_CONTEXT_FIELDS: Tuple[str, ...] = (
    "trial_number", "window_size", "offset", "sessions", "skip_min", "skip_max",
    "prng_base", "forward_threshold", "reverse_threshold",
    "dataset_sha256", "residue_sha256",
)

# Mandatory keys of an `s172_substripe_v1` spool payload
# (miner/range_miner_worker.py:881-899).
_PAYLOAD_KEYS: Tuple[str, ...] = (
    "schema_version", "stripe_id", "sub_index", "seed_start", "seed_count",
    "survivors",
)

# (direction, skip_mode) -> the directional population it feeds.
_POPULATIONS: Tuple[Tuple[str, str], ...] = (
    ("forward", "constant"), ("reverse", "constant"),
    ("forward", "variable"), ("reverse", "variable"),
)


# ---------------------------------------------------------------------------
# §7 — the stable cross-deliverable result object.
# ---------------------------------------------------------------------------
@dataclass
class MinerTrialAssembly:
    """The complete four-population assembly of ONE trial. Stable across D1-D6 —
    do NOT split it: D3 writes the binary NPZ from it, D4 the all-survivor NPZ,
    D6 adapts it back into the legacy Step-1 accumulator shape."""

    run_id: str
    bidirectional_constant: set          # set[int]
    bidirectional_variable: set          # set[int]
    forward_map_constant: dict           # dict[int, float]
    reverse_map_constant: dict
    forward_map_variable: dict
    reverse_map_variable: dict
    canonical_records_constant: list     # list[dict], ascending seed
    canonical_records_variable: list
    directional_counts: dict             # dict[str, int]
    timing: dict                         # at least {"assembly_s": float}
    # D3/D4 populate these ONLY after the corresponding artifact is successfully
    # written AND validated, via dataclasses.replace() (or an equivalent explicit
    # update) — never by mutating mid-way through a failed write. None means "not
    # produced yet"; an empty string would falsely claim a path exists. The D6
    # adapter fails closed on None where it needs a path.
    binary_npz_path: Optional[str] = None
    all_npz_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Canonical comparison / deep-copy helper (§4.1).
#
# A sorted-key JSON round-trip is BOTH the deep-copy mechanism (the coordinator
# retains and returns the same mutable manifest dicts in `self.enqueued`
# (coordinator:1956-1958), so a caller-side mutation after publication must not
# reach the sink's future assembly input) AND the canonical comparison form
# ("canonically identical" = equal after the sorted-key JSON round-trip).
# ---------------------------------------------------------------------------
def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _canonical_copy(obj: Any) -> Any:
    return json.loads(_canonical_json(obj))


# ---------------------------------------------------------------------------
# Spool read seam. The engine reads staged bytes ONLY through this function, so
# a harness can wrap/count engine file opens (Gate D1.1 instrumentation) without
# monkeypatching anything else, and D5 can later swap the backend.
# ---------------------------------------------------------------------------
def _read_spool_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _is_int(value: Any) -> bool:
    """Integer EXCLUDING bool. `isinstance(True, int)` is True and `True == 1`,
    so equality alone never excludes a Boolean payload identity [TB-D1-PV]."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """int or float, EXCLUDING bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# §5.1 — per-manifest identity validation, BEFORE any grouping.
# ---------------------------------------------------------------------------
def _phase_family_map(prng_base: str) -> Dict[int, str]:
    """The producer's family/phase authority (§3 workflow authority [TB-D1-B5]).

    `workflow_stages_for(prng_base, True)` is imported and inverted — the suffix
    table is NEVER reproduced by hand here. `True` is used deliberately: it is
    the SUPERSET of stages, so the map resolves the family for all four phases
    regardless of the trial's own test_both_modes (phase-set completeness is
    §5.2's job, not this map's)."""
    return {int(phase): family
            for family, phase in workflow_stages_for(prng_base, True)}


def _validate_manifest_identity(run_id: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Every §5.1 identity check for ONE manifest. Returns its trial_metadata.

    `workflow_phase_semantics` and `workflow_stages_for` are consistency ORACLES
    only — the explicit manifest `direction` / `skip_mode` strings remain the
    grouping values, and a disagreement is a hard failure, never a silent
    substitution."""
    def _fail(msg: str) -> None:
        raise PhaseIdentityError(f"{run_id}: {msg}")

    if not isinstance(manifest, dict):
        _fail(f"manifest is {type(manifest).__name__}, expected dict")
    meta = manifest.get("trial_metadata")
    if not isinstance(meta, dict) or not meta:
        _fail("manifest carries no trial_metadata dict")

    if manifest.get("run_id") != run_id:
        _fail(f"manifest run_id {manifest.get('run_id')!r} != assembly run_id "
              f"{run_id!r}")

    # manifest workflow_phase vs its own metadata copy
    m_phase, meta_phase = manifest.get("workflow_phase"), meta.get("workflow_phase")
    if not _is_int(m_phase) or not _is_int(meta_phase) or m_phase != meta_phase:
        _fail(f"manifest workflow_phase {m_phase!r} != trial_metadata "
              f"workflow_phase {meta_phase!r}")
    phase = int(m_phase)

    # the lifted provenance copies (coordinator:2064-2068) must agree
    for prov in ("dataset_sha256", "residue_sha256"):
        if manifest.get(prov) != meta.get(prov):
            _fail(f"lifted {prov} {manifest.get(prov)!r} != trial_metadata copy "
                  f"{meta.get(prov)!r}")

    # explicit (direction, skip_mode) vs the §6.8 oracle
    try:
        expect_dir, expect_mode = workflow_phase_semantics(phase)
    except MinerMetadataError as e:
        _fail(f"unresolvable workflow_phase {phase!r}: {e}")
    direction, skip_mode = meta.get("direction"), meta.get("skip_mode")
    if (direction, skip_mode) != (expect_dir, expect_mode):
        _fail(f"phase {phase} identity ({direction!r}, {skip_mode!r}) != "
              f"workflow_phase_semantics ({expect_dir!r}, {expect_mode!r})")

    # family_name vs the imported workflow authority
    prng_base = meta.get("prng_base")
    expect_family = _phase_family_map(prng_base).get(phase)
    if meta.get("family_name") != expect_family:
        _fail(f"phase {phase} family_name {meta.get('family_name')!r} != "
              f"workflow_stages_for({prng_base!r}, True) family {expect_family!r}")

    # prng_type is base (constant) / base + '_hybrid' (variable)
    expect_prng = prng_base if skip_mode == "constant" else f"{prng_base}_hybrid"
    if meta.get("prng_type") != expect_prng:
        _fail(f"prng_type {meta.get('prng_type')!r} != {expect_prng!r} for "
              f"skip_mode {skip_mode!r}")

    # threshold_used is the DIRECTIONAL threshold
    expect_thresh = (meta.get("forward_threshold") if direction == "forward"
                     else meta.get("reverse_threshold"))
    if not _is_number(meta.get("threshold_used")) or not _is_number(expect_thresh) \
            or float(meta["threshold_used"]) != float(expect_thresh):
        _fail(f"threshold_used {meta.get('threshold_used')!r} != the "
              f"{direction} threshold {expect_thresh!r}")
    return meta


# ---------------------------------------------------------------------------
# §5.3 — the ordered, merge-relevant projection of ONE fully validated spool.
#
# LOCKED DEFINITION (Team Beta D5 ruling, finding F1). This is lossless with
# respect to ALL state observable by canonical assembly — NOT a lossless
# serialization of the source JSON payload. `merge_validated_spools` consumes
# only seed and match_rate per survivor (see the merge loop below); `strategy_id`
# and the ragged `skips` are fully validated inside `read_and_validate_spool` and
# then DISCARDED, exactly as §5.4's numeric encoding is validated-and-discarded.
# They never cross a process boundary because canonical assembly never observes
# them — which is also what lets D5's artifact codec run `allow_pickle=False`
# with no object arrays.
#
# Order and multiplicity are preserved EXACTLY: no sort, no dedup, no
# normalization. Input survivor i is projection row i. An intra-spool duplicate
# seed survives as two rows, so the §5.4 duplicate invariant still fires.
#
# SEED REPRESENTATION IS DUAL, AND THAT IS A CORRECTNESS REQUIREMENT [D5 REV3,
# Team Beta hold]. The first cut of this projection stored seeds as a plain
# `int64` array. That silently NARROWED the engine's accepted input: the §5.3
# validator bounds a seed to the declared window `[seed_start, seed_start +
# seed_count)` and imposes NO signed-64 bound and no non-negativity requirement
# on `seed_start`, and the pre-D5 maps keyed on arbitrary-precision PYTHON ints.
# A spool declaring `seed_start = 2**63` was therefore accepted before D5 and
# raised `OverflowError` after it — a valid-input divergence, which is exactly
# what the "the extraction changed nothing" claim may not contain. ("Unreachable
# for java_lcg" is not a defence: the engine is base-parameterized, not
# contractually one-family.)
#
# So the projection carries ONE of two lossless encodings, chosen per spool:
#
#   * `int64`        — the fast, common path: every seed in the spool is
#                      signed-64 representable, so seeds are one `int64` array.
#   * `signed_bytes` — the fallback, armed the moment ANY seed in the spool
#                      leaves signed-64: every seed of that spool (including its
#                      small ones) becomes a two's-complement big-endian byte run
#                      of DETERMINISTIC SIGNED-BYTE LENGTH in one concatenated
#                      `uint8` array, addressed by a `uint64` offsets array of
#                      length survivor_count + 1. The encoding may be non-minimal
#                      at negative signed-width boundaries, but decoding is exact
#                      and canonical assembly observes the original Python
#                      integer.
#
# Both encodings are plain numeric arrays: NO object array, so D5's artifact
# codec still loads with `allow_pickle=False` (F1). Exactly one representation
# is populated; the other seed fields are None, and `__post_init__` refuses any
# other combination. `projection_seeds` is the ONLY decoder, and it returns
# PYTHON ints in both cases so the merge's map keys are bit-identical to pre-D5.
# ---------------------------------------------------------------------------
SEED_ENCODING_INT64 = "int64"
SEED_ENCODING_SIGNED_BYTES = "signed_bytes"
SEED_ENCODINGS: Tuple[str, ...] = (SEED_ENCODING_INT64,
                                   SEED_ENCODING_SIGNED_BYTES)

_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1


def _encode_seed(seed: int) -> bytes:
    """ONE seed as a two's-complement big-endian byte run whose width comes from
    the DETERMINISTIC SIGNED-BYTE LENGTH FORMULA.

    `(bit_length // 8) + 1` is the deterministic signed-byte length formula, and
    the `+ 1` is not slack: `bit_length()` ignores the sign, so a value needing
    exactly 8k bits of magnitude needs a (k+1)-th byte for the sign bit. 127 ->
    1 byte, 128 -> 2, 255 -> 2, 2**63 -> 9, 0 -> 1. The ±2^(8k-1) boundaries are
    where a `(bit_length + 7) // 8` spelling silently raises OverflowError,
    which is the very failure mode this fallback exists to remove.

    The encoding may be non-minimal at negative signed-width boundaries (-128
    takes 2 bytes here, not 1), but decoding is exact and canonical assembly
    observes the original Python integer: `int.from_bytes` sign-extends the run
    back to the identical value. Round-trip fidelity is the contract; width
    minimality is not."""
    nbytes = (seed.bit_length() // 8) + 1
    return seed.to_bytes(nbytes, "big", signed=True)


def _validate_projection_shape(projection: "ValidatedSpoolProjection") -> None:
    """Refuse any projection that is not exactly one of the two encodings, with
    rectangular, correctly-typed arrays.

    Fail-closed at CONSTRUCTION, so a half-populated or mis-typed projection
    cannot exist to be merged — including one rebuilt by D5's artifact readback
    from a corrupt or foreign artifact."""
    count = projection.survivor_count
    if not _is_int(count) or count < 0:
        raise ValueError(f"survivor_count {count!r} is not a nonnegative int")
    rates = projection.match_rates
    if not isinstance(rates, np.ndarray) or rates.dtype != np.dtype(np.float64) \
            or rates.shape != (count,):
        raise ValueError(
            f"match_rates must be a float64 array of shape ({count},), got "
            f"{getattr(rates, 'dtype', type(rates).__name__)} "
            f"{getattr(rates, 'shape', None)}")

    if projection.seed_encoding == SEED_ENCODING_INT64:
        if projection.seed_bytes is not None or projection.seed_offsets is not None:
            raise ValueError(
                "seed_encoding 'int64' populates seeds_i64 ONLY; the "
                "signed_bytes fields must be None")
        seeds = projection.seeds_i64
        if not isinstance(seeds, np.ndarray) \
                or seeds.dtype != np.dtype(np.int64) or seeds.shape != (count,):
            raise ValueError(
                f"seeds_i64 must be an int64 array of shape ({count},), got "
                f"{getattr(seeds, 'dtype', type(seeds).__name__)} "
                f"{getattr(seeds, 'shape', None)}")
        return

    if projection.seed_encoding == SEED_ENCODING_SIGNED_BYTES:
        if projection.seeds_i64 is not None:
            raise ValueError(
                "seed_encoding 'signed_bytes' populates seed_bytes + "
                "seed_offsets ONLY; seeds_i64 must be None")
        blob, offsets = projection.seed_bytes, projection.seed_offsets
        if not isinstance(blob, np.ndarray) or blob.dtype != np.dtype(np.uint8) \
                or blob.ndim != 1:
            raise ValueError(
                f"seed_bytes must be a 1-D uint8 array, got "
                f"{getattr(blob, 'dtype', type(blob).__name__)}")
        if not isinstance(offsets, np.ndarray) \
                or offsets.dtype != np.dtype(np.uint64) \
                or offsets.shape != (count + 1,):
            raise ValueError(
                f"seed_offsets must be a uint64 array of shape ({count + 1},), "
                f"got {getattr(offsets, 'dtype', type(offsets).__name__)} "
                f"{getattr(offsets, 'shape', None)}")
        # Python ints, not a uint64 diff: an out-of-order pair would WRAP to a
        # huge positive under uint64 subtraction and pass a naive `> 0` check.
        bounds = [int(v) for v in offsets]
        if bounds[0] != 0 or bounds[-1] != int(blob.shape[0]):
            raise ValueError(
                f"seed_offsets {bounds[0]}..{bounds[-1]} do not span the "
                f"{int(blob.shape[0])}-byte seed_bytes run exactly")
        for k in range(count):
            if bounds[k + 1] <= bounds[k]:
                raise ValueError(
                    f"seed_offsets row {k} spans "
                    f"[{bounds[k]}, {bounds[k + 1]}) — every seed occupies at "
                    f"least one byte and offsets must strictly increase")
        return

    raise ValueError(
        f"unknown seed_encoding {projection.seed_encoding!r}; expected one of "
        f"{list(SEED_ENCODINGS)}")


@dataclass(frozen=True)
class ValidatedSpoolProjection:
    seed_encoding: str                   # one of SEED_ENCODINGS
    seeds_i64: Optional[np.ndarray]      # int64, (survivor_count,) — or None
    seed_bytes: Optional[np.ndarray]     # uint8, concatenated runs — or None
    seed_offsets: Optional[np.ndarray]   # uint64, (survivor_count + 1,) — or None
    match_rates: np.ndarray              # float64, (survivor_count,), aligned
    survivor_count: int                  # rows, in survivor order

    def __post_init__(self) -> None:
        _validate_projection_shape(self)


def build_validated_projection(
    seeds: Sequence[int], match_rates: Sequence[float],
) -> ValidatedSpoolProjection:
    """Build the projection of ONE validated spool, choosing the encoding.

    `seeds` are the ALREADY-VALIDATED Python ints, in survivor order — never
    normalized, sorted or deduped here. The encoding is a property of the whole
    spool: if a single seed leaves signed-64, the entire projection switches to
    `signed_bytes` so one projection never mixes representations."""
    count = len(seeds)
    rates = np.asarray(match_rates, dtype=np.float64)
    if all(_INT64_MIN <= seed <= _INT64_MAX for seed in seeds):
        return ValidatedSpoolProjection(
            seed_encoding=SEED_ENCODING_INT64,
            seeds_i64=np.array(seeds, dtype=np.int64),
            seed_bytes=None, seed_offsets=None,
            match_rates=rates, survivor_count=count)
    runs = [_encode_seed(int(seed)) for seed in seeds]
    offsets = [0]
    for run in runs:
        offsets.append(offsets[-1] + len(run))
    return ValidatedSpoolProjection(
        seed_encoding=SEED_ENCODING_SIGNED_BYTES,
        seeds_i64=None,
        seed_bytes=np.frombuffer(b"".join(runs), dtype=np.uint8).copy(),
        seed_offsets=np.array(offsets, dtype=np.uint64),
        match_rates=rates, survivor_count=count)


def projection_seeds(projection: ValidatedSpoolProjection) -> List[int]:
    """Decode a projection's seeds back to PYTHON ints, in survivor order.

    Python ints, not `np.int64`, on BOTH paths: pre-D5 keyed the four
    directional maps on `int(entry[0])`, and an `np.int64` key — though equal
    and equal-hashing — is not the pre-D5 contract, and would leak a numpy
    scalar into every canonical record and every D6 consumer."""
    if projection.seed_encoding == SEED_ENCODING_INT64:
        return [int(seed) for seed in projection.seeds_i64]
    if projection.seed_encoding == SEED_ENCODING_SIGNED_BYTES:
        blob = projection.seed_bytes.tobytes()
        offsets = projection.seed_offsets
        return [int.from_bytes(blob[int(offsets[k]):int(offsets[k + 1])],
                               "big", signed=True)
                for k in range(projection.survivor_count)]
    raise ValueError(
        f"unknown seed_encoding {projection.seed_encoding!r}; expected one of "
        f"{list(SEED_ENCODINGS)}")


# ---------------------------------------------------------------------------
# §5.3 — a canonical spool-read defect carried as DATA [D5 REV2, Team Beta
# ruled option B].
#
# WHY THIS EXISTS. The pre-D5 `assemble_trial` INTERLEAVES read and merge: it
# walks the deterministic order and, per position, reads that spool and
# immediately merges it. So an earlier-position duplicate raises BEFORE a
# later-position spool is ever read. A parallel front end cannot literally do
# that — it must read ahead — so without this type the observable exception
# would depend on how far ahead the readers got. Beta ruled that divergence out:
# a parallel producer captures the canonical read defect as typed data, and the
# parent REPLAYS it at its own position in deterministic order. Precedence is
# then a function of `order` alone, never of completion order.
#
# THESE ARE CANONICAL ASSEMBLY-CONTRACT TYPES, NOT PROCESS MACHINERY. They live
# beside the merge that consumes them, they are frozen after D5 Commit 1, and
# the serial path never produces one — serial reads lazily and raises the
# ORIGINAL exception object, with its original traceback, exactly as before D5.
#
# NEVER A PICKLED EXCEPTION INSTANCE. The descriptor round-trips class identity
# (by allowlisted name), `.args`, the rendered message and any custom
# attribution attributes. Traceback frames are explicitly NOT preserved (REV2
# §3: tracebacks and backend-internal chaining are not contractual).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CapturedSpoolReadError:
    error_code: str                       # canonical class name, allowlisted
    message: str                          # exact rendered message
    args: tuple                           # exc.args, scalar-only
    attributes: Mapping[str, Any]         # custom identity/provenance fields


# A per-position outcome of the spool-read front end: the validated projection,
# or the canonical defect that spool produced.
SpoolReadOutcome = Union[ValidatedSpoolProjection, CapturedSpoolReadError]

# THE ALLOWLIST (REV2 §1). ONLY canonical spool-read defects may become a
# descriptor. `read_and_validate_spool` funnels every I/O, size, SHA, decode,
# container, identity and per-survivor semantic failure into exactly one class
# — `SpoolIdentityError` — so today that hierarchy has exactly one member, and
# this mapping is its complete enumeration.
#
# Fail-closed by construction: a class absent from this mapping cannot be
# captured and cannot be reconstructed. `MemoryError`, `KeyboardInterrupt`,
# `SystemExit`, process-pool failures, artifact-write failures and unexpected
# programming errors are BACKEND failures, not producer defects — descriptorizing
# one would let infrastructure masquerade as a spool contract violation, which
# REV2 §5 forbids. A future semantic payload-validation exception in the
# `SpoolIdentityError` hierarchy must be added HERE, deliberately.
CANONICAL_SPOOL_READ_ERRORS: Dict[str, Type[BaseException]] = {
    "SpoolIdentityError": SpoolIdentityError,
}

# What a descriptor may carry. Scalars (and tuples of scalars) only: the
# descriptor crosses a process boundary, so anything richer would either fail to
# pickle or smuggle live state — and no canonical spool-read defect needs more.
_DESCRIPTOR_SCALARS: Tuple[type, ...] = (str, int, float, bool, type(None))


def _is_descriptor_scalar(value: Any) -> bool:
    if isinstance(value, tuple):
        return all(_is_descriptor_scalar(v) for v in value)
    return isinstance(value, _DESCRIPTOR_SCALARS)


def capture_spool_read_error(exc: BaseException) -> CapturedSpoolReadError:
    """Capture ONE canonical spool-read defect as a descriptor.

    Refuses — with `TypeError`, loudly — anything that is not an allowlisted
    canonical producer defect, and anything whose `.args` or custom attributes
    are not scalar. Refusing is the point: a descriptor is the ONLY thing the
    parent will replay as a producer defect, so a backend failure must never be
    able to enter through this door (REV2 §5).

    The serial path never calls this. It exists so a PARALLEL producer can hand
    a canonical defect back as data instead of letting it surface out of order.
    """
    error_code = type(exc).__name__
    if error_code not in CANONICAL_SPOOL_READ_ERRORS:
        raise TypeError(
            f"{error_code} is not a canonical spool-read defect and must not be "
            f"captured as data: {exc!r}. Only "
            f"{sorted(CANONICAL_SPOOL_READ_ERRORS)} may be replayed as a "
            f"producer defect; everything else is a backend failure.")
    if not _is_descriptor_scalar(tuple(exc.args)):
        raise TypeError(
            f"{error_code}.args {exc.args!r} is not scalar-only — a descriptor "
            f"carries data, never live objects")
    attributes = dict(vars(exc))
    for key, value in attributes.items():
        if not _is_descriptor_scalar(value):
            raise TypeError(
                f"{error_code}.{key} {value!r} is not scalar and cannot be "
                f"round-tripped; dropping it would lose attribution the "
                f"equivalence contract covers")
    return CapturedSpoolReadError(
        error_code=error_code,
        message=str(exc),
        args=tuple(exc.args),
        attributes=attributes,
    )


def raise_captured_spool_error(descriptor: CapturedSpoolReadError) -> NoReturn:
    """Reconstruct and raise the ORIGINAL canonical exception class.

    Class, `.args`, rendered message and custom attribution are preserved; the
    traceback is not (REV2 §3). This is the ONLY place a captured read error
    becomes a live exception in the parent, and it fails closed rather than
    fabricating a canonical exception it cannot reproduce faithfully.
    """
    cls = CANONICAL_SPOOL_READ_ERRORS.get(descriptor.error_code)
    if cls is None:
        raise TypeError(
            f"cannot replay {descriptor.error_code!r}: it is not an allowlisted "
            f"canonical spool-read defect ({sorted(CANONICAL_SPOOL_READ_ERRORS)})")
    exc = cls(*descriptor.args)
    for key, value in descriptor.attributes.items():
        setattr(exc, key, value)
    if str(exc) != descriptor.message:
        raise ValueError(
            f"captured {descriptor.error_code} does not round-trip: rendered "
            f"{str(exc)!r} != captured {descriptor.message!r}")
    raise exc


# ---------------------------------------------------------------------------
# §5.3 — spool read + identity + container + semantic validation.
#
# D5 [ruling item 1, option A]: made PUBLIC and separately callable, and its
# return type narrowed from the parsed payload to `ValidatedSpoolProjection`, so
# `process_sharded` workers reach the IDENTICAL validation by calling this exact
# function rather than a second copy. The validation body below is unchanged;
# D1.1 18/18 staying green with no test edits is the proof.
# ---------------------------------------------------------------------------
def read_and_validate_spool(
    run_id: str, manifest: Dict[str, Any],
) -> ValidatedSpoolProjection:
    """Read this manifest's staged bytes at COMMIT time and return the validated
    projection. Everything that can go wrong — I/O, size, SHA, JSON decode, the
    container shape, the schema/stripe/sub_index identity, and the per-survivor
    semantics — becomes SpoolIdentityError; no raw TypeError/KeyError/quirk may
    escape [TB-D1-PV].

    The parsed payload stays LOCAL and ephemeral: only the projection escapes,
    and it is constructed only after the ENTIRE spool has passed, so a malformed
    survivor near the end of the JSON can never yield a partial result."""
    path = manifest.get("local_spool_path")
    stripe_id = manifest.get("stripe_id")
    sub_index = manifest.get("sub_index")

    def _fail(msg: str) -> None:
        raise SpoolIdentityError(
            f"{run_id} {stripe_id}/sub{sub_index} ({path!r}): {msg}")

    try:
        raw = _read_spool_bytes(path)
    except OSError as e:
        _fail(f"staged spool unreadable: {e}")

    # 1. size + SHA-256 (defense-in-depth: the coordinator already verified these
    #    at staging time, so a mismatch here means post-staging corruption).
    if len(raw) != manifest.get("expected_size"):
        _fail(f"size {len(raw)} != expected_size {manifest.get('expected_size')}")
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != manifest.get("expected_sha256"):
        _fail(f"sha256 {actual_sha} != expected_sha256 "
              f"{manifest.get('expected_sha256')}")

    # 2. parse + CONTAINER validation
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        _fail(f"staged spool is not decodable JSON: {e}")
    if not isinstance(payload, dict):
        _fail(f"payload root is {type(payload).__name__}, expected dict")
    missing = [k for k in _PAYLOAD_KEYS if k not in payload]
    if missing:
        _fail(f"payload missing mandatory key(s) {missing!r}")
    survivors = payload["survivors"]
    if not isinstance(survivors, list):
        _fail(f"payload survivors is {type(survivors).__name__}, expected list")

    # 3. IDENTITY: this spool belongs to THIS manifest's logical shard.
    if payload["schema_version"] != SUBSTRIPE_SCHEMA_VERSION:
        _fail(f"schema_version {payload['schema_version']!r} != "
              f"{SUBSTRIPE_SCHEMA_VERSION!r}")
    if payload["stripe_id"] != stripe_id:
        _fail(f"payload stripe_id {payload['stripe_id']!r} != manifest "
              f"stripe_id {stripe_id!r}")
    # bool is excluded FIRST: True == 1, so equality alone would accept a Boolean
    # payload identity for sub_index 1 [TB-D1-PV].
    if not _is_int(payload["sub_index"]):
        _fail(f"payload sub_index {payload['sub_index']!r} is not an integer "
              f"(bool excluded)")
    if payload["sub_index"] != sub_index:
        _fail(f"payload sub_index {payload['sub_index']!r} != manifest sub_index "
              f"{sub_index!r}")

    # 4. SEMANTICS (bool excluded from every integer check).
    seed_start, seed_count = payload["seed_start"], payload["seed_count"]
    if not _is_int(seed_start):
        _fail(f"seed_start {seed_start!r} is not an integer (bool excluded)")
    if not _is_int(seed_count) or seed_count < 0:
        _fail(f"seed_count {seed_count!r} is not a nonnegative integer "
              f"(bool excluded)")
    lo, hi = seed_start, seed_start + seed_count
    for i, entry in enumerate(survivors):
        if not isinstance(entry, list) or len(entry) != 4:
            _fail(f"survivor[{i}] is not a 4-element list: {entry!r}")
        seed, match_rate, strategy_id, skips = entry
        if not _is_int(seed):
            _fail(f"survivor[{i}] seed {seed!r} is not an integer (bool excluded)")
        if not (lo <= seed < hi):
            _fail(f"survivor[{i}] seed {seed} outside the declared sub-stripe "
                  f"range [{lo}, {hi})")
        if not _is_number(match_rate):
            _fail(f"survivor[{i}] match_rate {match_rate!r} is not numeric "
                  f"(bool excluded)")
        if not math.isfinite(float(match_rate)):
            _fail(f"survivor[{i}] match_rate {match_rate!r} is not finite")
        if not (0.0 <= float(match_rate) <= 1.0):
            _fail(f"survivor[{i}] match_rate {match_rate!r} outside [0.0, 1.0]")
        if strategy_id is not None and not _is_int(strategy_id):
            _fail(f"survivor[{i}] strategy_id {strategy_id!r} is neither None nor "
                  f"an integer (bool excluded)")
        if not isinstance(skips, list):
            _fail(f"survivor[{i}] skip_sequence is {type(skips).__name__}, "
                  f"expected list")
        for j, skip in enumerate(skips):
            if not _is_int(skip):
                _fail(f"survivor[{i}] skip[{j}] {skip!r} is not an integer "
                      f"(bool excluded)")

    # 5. PROJECTION — built only now, after the whole spool has passed.
    #
    #    The seeds handed to the builder are the validated payload ints
    #    THEMSELVES, unconverted: the window check above (`lo <= seed < hi`) is
    #    pure Python arbitrary-precision arithmetic and must stay that way — it
    #    never overflowed, only the projection did [REV3 §2]. The builder picks
    #    `int64` or `signed_bytes` for the whole spool, and `projection_seeds`
    #    reverses either back to the identical Python int, so what reaches a
    #    directional map is bit-identical to the pre-extraction `int(entry[0])`
    #    for EVERY seed the validator accepts — not merely for signed-64 ones.
    rates = np.empty(len(survivors), dtype=np.float64)
    for i, entry in enumerate(survivors):
        rates[i] = entry[1]
    return build_validated_projection([entry[0] for entry in survivors], rates)


# ---------------------------------------------------------------------------
# §5.5/§6 — derived fields + canonical records for ONE mode.
#
# D3.25-B: the body moved VERBATIM to `utils.canonical_records.build_mode_records`
# so PWC/ZMQ reach the identical derivation without a `utils -> miner` import.
# The move is semantics-preserving by requirement, and D1.1 18/18 + D2 7/7
# staying green is its proof. `_mode_records` remains as the private D1 alias so
# no call site, error message or attribute lookup in this module changes.
# ---------------------------------------------------------------------------
_mode_records = build_mode_records


# ---------------------------------------------------------------------------
# §5.4 + §5.5/§6 — GLOBAL ASSEMBLY. Extracted VERBATIM from the inline block
# that used to sit inside `assemble_trial`.
#
# D5 [ruling item 1, option A]: this is the SOLE authority for global assembly
# semantics, and it is unconditionally SERIAL and PARENT-ONLY. `process_sharded`
# parallelizes only the per-spool front end above; the four directional maps,
# within-population duplicate detection with first-vs-dup provenance,
# `prng_type_by_mode` last-writer-wins in loop order [F4], the two
# intersections and the canonical enrichment all happen HERE, once, in the
# parent. A worker must never build a map, sort, dedup, normalize or intersect.
#
# `ctx` and `started` are explicit PARAMETERS rather than re-derived here on
# purpose. Both are trial-global values the caller's metadata gauntlet already
# owns: `ctx` is built from `metas[0]` in ORIGINAL manifest-list order, whereas
# `ordered_outcomes` is in the deterministic SORT order [F2], and `started` is
# stamped before the gauntlet. Re-deriving either from the outcomes would change
# which meta the context is read from and would shorten `assembly_s` — i.e. it
# would break the very byte-equivalence this extraction has to preserve.
#
# THE CANONICAL REPLAY LOOP [D5 REV2 §2]. `ordered_outcomes` is consumed ONE
# POSITION AT A TIME, and the loop body is the whole state machine both backends
# present:
#
#     for position in deterministic_order:
#         outcome = <projection or captured read error at this position>
#         if it is a read error:  re-raise it HERE, at this position
#         merge_insert(projection)   # may raise DirectionalDuplicateError
#
# Because the source is consumed lazily, the SERIAL caller's generator does not
# read position p+1 until position p has been merged — restoring the pre-D5
# interleaving exactly, including which of two independent producer defects
# surfaces first. The PARALLEL caller reads ahead concurrently but yields the
# same per-position outcomes in the same order, so it observes the same defect.
# Completion order is irrelevant by construction; only `order` decides.
# ---------------------------------------------------------------------------
def merge_validated_spools(
    run_id: str,
    ctx: Dict[str, Any],
    ordered_outcomes: Iterable[
        Tuple[Dict[str, Any], Dict[str, Any], SpoolReadOutcome]
    ],
    started: float,
) -> MinerTrialAssembly:
    """Merge per-position spool-read outcomes into the four-population assembly
    of one trial.

    `ordered_outcomes` is `(manifest, meta, outcome)` in the SAME deterministic
    order `assemble_trial` computes — sort key `(workflow_phase, stripe_id,
    sub_index, attempt, event_id)` — where `outcome` is either a
    `ValidatedSpoolProjection` or a `CapturedSpoolReadError` for that position.
    It is an ITERABLE, not a sequence, and it is deliberately NOT materialized:
    a lazy source is what preserves read/merge interleaving.

    The merge reads meta fields (direction, skip_mode, prng_type) AND manifest
    fields (stripe_id, sub_index, attempt), so both must be carried alongside
    the outcome.

    Raises DirectionalDuplicateError; whatever `raise_captured_spool_error`
    replays (a canonical SpoolIdentityError) at its own position; whatever a
    lazy source raises while producing a position (for the serial caller, the
    ORIGINAL SpoolIdentityError object with its original traceback); plus
    ValueError from the canonical record builder. It performs NO I/O itself and
    reads no staged path."""
    maps: Dict[Tuple[str, str], Dict[int, float]] = {p: {} for p in _POPULATIONS}
    prov: Dict[Tuple[str, str], Dict[int, Tuple[str, int, int, float]]] = {
        p: {} for p in _POPULATIONS
    }
    prng_type_by_mode: Dict[str, str] = {}
    for manifest, meta, outcome in ordered_outcomes:
        # A canonical read defect at THIS position pre-empts every later
        # position and is pre-empted by every earlier position's merge — which
        # is exactly the pre-D5 precedence.
        if isinstance(outcome, CapturedSpoolReadError):
            raise_captured_spool_error(outcome)
        projection = outcome
        direction, skip_mode = meta["direction"], meta["skip_mode"]
        prng_type_by_mode[skip_mode] = meta["prng_type"]
        pop_map = maps[(direction, skip_mode)]
        pop_prov = prov[(direction, skip_mode)]
        stripe_id = manifest.get("stripe_id")
        sub_index = int(manifest.get("sub_index", 0))
        attempt = int(manifest.get("attempt", 0))
        # ONE decode per spool, through the single decoder, so both encodings
        # produce the identical Python-int map keys pre-D5 produced [REV3 §3].
        seeds = projection_seeds(projection)
        for k in range(projection.survivor_count):
            seed = seeds[k]
            match_rate = float(projection.match_rates[k])
            if seed in pop_map:
                f_stripe, f_sub, f_attempt, f_rate = pop_prov[seed]
                raise DirectionalDuplicateError(
                    f"{run_id}: seed {seed} appears twice in the "
                    f"{direction}/{skip_mode} population "
                    f"(first {f_stripe}/sub{f_sub}/a{f_attempt}, "
                    f"duplicate {stripe_id}/sub{sub_index}/a{attempt}) — a "
                    f"producer/coverage defect, never a dedup opportunity",
                    run_id=run_id, workflow_phase=int(meta["workflow_phase"]),
                    direction=direction, skip_mode=skip_mode, seed=seed,
                    first_stripe=f_stripe, first_sub_index=f_sub,
                    first_attempt=f_attempt, first_match_rate=f_rate,
                    dup_stripe=stripe_id, dup_sub_index=sub_index,
                    dup_attempt=attempt, dup_match_rate=match_rate)
            pop_map[seed] = match_rate
            pop_prov[seed] = (stripe_id, sub_index, attempt, match_rate)

    fwd_c, rev_c = maps[("forward", "constant")], maps[("reverse", "constant")]
    fwd_v, rev_v = maps[("forward", "variable")], maps[("reverse", "variable")]

    # ---- §5.5/§6 intersections + canonical enrichment ----------------------
    bidi_c, records_c = _mode_records(
        fwd_c, rev_c, ctx, "constant", prng_type_by_mode.get("constant"))
    bidi_v, records_v = _mode_records(
        fwd_v, rev_v, ctx, "variable", prng_type_by_mode.get("variable"))

    return MinerTrialAssembly(
        run_id=run_id,
        bidirectional_constant=bidi_c,
        bidirectional_variable=bidi_v,
        forward_map_constant=fwd_c,
        reverse_map_constant=rev_c,
        forward_map_variable=fwd_v,
        reverse_map_variable=rev_v,
        canonical_records_constant=records_c,
        canonical_records_variable=records_v,
        directional_counts={
            "forward_constant":       len(fwd_c),
            "reverse_constant":       len(rev_c),
            "forward_variable":       len(fwd_v),
            "reverse_variable":       len(rev_v),
            "bidirectional_constant": len(bidi_c),
            "bidirectional_variable": len(bidi_v),
        },
        timing={"assembly_s": time.perf_counter() - started},
    )


# ---------------------------------------------------------------------------
# §5.1 + §5.2 + §5.4 — the METADATA GAUNTLET, extracted VERBATIM, plus the
# deterministic spool order.
#
# D5 [ruling item 1, option A]: extracted so `process_sharded`'s parent can run
# the IDENTICAL gauntlet — in the identical order — BEFORE dispatching any
# worker, rather than growing a second copy of it. That is what makes exception
# PRECEDENCE identical across backends: a PhaseIdentityError or
# AssemblyConsistencyError still pre-empts every SpoolIdentityError, because no
# spool byte is read until this function has returned.
# ---------------------------------------------------------------------------
def prepare_trial_assembly(
    run_id: str, manifests: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[int]]:
    """Run every pre-spool check and compute the deterministic spool order.

    Returns `(metas, ctx, order)`:
      * `metas`  — per-manifest trial_metadata, in ORIGINAL manifest-list order;
      * `ctx`    — the 11-field trial context, read from `metas[0]`;
      * `order`  — indices into `manifests`, sorted by
                   `(workflow_phase, stripe_id, sub_index, attempt, event_id)`.

    Raises AssemblyStateError, PhaseIdentityError, AssemblyConsistencyError, and
    ValueError from `utils/prng_encoding` — every one of them BEFORE any staged
    byte is read. Performs no I/O."""
    if not manifests:
        raise AssemblyStateError(
            f"{run_id}: cannot assemble a trial with zero retained manifests")

    # ---- §5.1 per-manifest identity, BEFORE any grouping -------------------
    metas: List[Dict[str, Any]] = [
        _validate_manifest_identity(run_id, m) for m in manifests
    ]

    # ---- §5.2 cross-manifest 11-field consistency --------------------------
    canon_ctx = None
    for manifest, meta in zip(manifests, metas):
        try:
            ctx_key = _canonicalize_trial_context(meta)
        except (KeyError, TypeError, ValueError) as e:
            raise AssemblyConsistencyError(
                f"{run_id}: manifest {manifest.get('event_id')!r} has an "
                f"uncanonicalizable trial context: {e}")
        if canon_ctx is None:
            canon_ctx = ctx_key
        elif ctx_key != canon_ctx:
            raise AssemblyConsistencyError(
                f"{run_id}: manifest {manifest.get('event_id')!r} disagrees on the "
                f"11-field canonical trial context:\n  {ctx_key}\n!=\n  {canon_ctx}")

    # ---- §5.2 phase-set completeness [TB-D1-B1] ----------------------------
    # Every executed pass yields >= 1 manifest (each completed stripe publishes
    # its verified shards, and a sub-stripe spools a payload even with zero
    # survivors), so an absent phase means the pass DID NOT RUN. D1 never
    # declares an absent reverse population legitimate and never fabricates one.
    phases = {int(meta["workflow_phase"]) for meta in metas}
    if phases not in ({1, 2}, {1, 2, 3, 4}):
        raise AssemblyConsistencyError(
            f"{run_id}: incomplete directional pairing — workflow phases present "
            f"{sorted(phases)}; expected exactly [1, 2] (constant only) or "
            f"[1, 2, 3, 4] (both modes)")

    ctx = {k: metas[0][k] for k in _CONTEXT_FIELDS}
    ctx["sessions"] = ctx["sessions"] if ctx["sessions"] is not None else []

    # ---- §5.4 encoding validation via the single source of truth -----------
    # Strings stay in the records; the numeric encoding is D3 and is DISCARDED
    # here — this is a hard-fail validation of every distinct identity only.
    for prng_type in sorted({meta["prng_type"] for meta in metas}):
        encode_prng_type(prng_type)
    for skip_mode in sorted({meta["skip_mode"] for meta in metas}):
        encode_skip_mode(skip_mode)

    # Deterministic order so a duplicate is always reported against the same
    # "first" insertion regardless of manifest arrival order.
    order = sorted(
        range(len(manifests)),
        key=lambda i: (int(metas[i]["workflow_phase"]),
                       str(manifests[i].get("stripe_id")),
                       int(manifests[i].get("sub_index", 0)),
                       int(manifests[i].get("attempt", 0)),
                       str(manifests[i].get("event_id"))))
    return metas, ctx, order


# ---------------------------------------------------------------------------
# §5.3 — THE SERIAL FRONT END, as a LAZY generator [D5 REV2 §2].
#
# One `read_and_validate_spool` per position, in `order`, pulled by the merge
# ONE AT A TIME. That laziness is load-bearing, not stylistic:
#
#   * position p is read only after position p-1 has been MERGED, so an
#     earlier-position DirectionalDuplicateError still pre-empts a
#     later-position SpoolIdentityError, exactly as the pre-D5 interleaved loop
#     did. A materialized `[read(i) for i in order]` would read everything
#     first and invert that precedence;
#   * the serial path NEVER produces a `CapturedSpoolReadError`. A bad read
#     raises the ORIGINAL exception object, with its original traceback, at its
#     own position — nothing round-trips through a descriptor. That is what
#     makes this extraction a true no-op for the serial backend.
# ---------------------------------------------------------------------------
def _serial_outcomes(
    run_id: str,
    manifests: List[Dict[str, Any]],
    metas: List[Dict[str, Any]],
    order: List[int],
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], ValidatedSpoolProjection]]:
    for i in order:
        yield (manifests[i], metas[i],
               read_and_validate_spool(run_id, manifests[i]))


# ---------------------------------------------------------------------------
# §5 — the assembly engine. ONE module-level entry point so D4/D5 call the SAME
# derivation [TB-R2]; the sink's commit_trial is its only D1 caller.
# ---------------------------------------------------------------------------
def assemble_trial(run_id: str, manifests: List[Dict[str, Any]]) -> MinerTrialAssembly:
    """Derive the complete four-population assembly of one trial from its
    accumulated ShardReadyManifests, reading each staged spool exactly once.

    Backend-independent and side-effect free: it writes nothing, deletes nothing,
    and holds no reference to any staged path after it returns.

    Raises (all fail-closed, §8): PhaseIdentityError, AssemblyConsistencyError,
    SpoolIdentityError, DirectionalDuplicateError, AssemblyStateError; plus
    ValueError from `utils/prng_encoding` on an unknown prng_type / skip_mode
    (the canonical module's own hard-fail is deliberately not re-wrapped).

    This is the SERIAL composition of the three shared units below;
    `process_sharded` composes the SAME three, replacing only how the middle
    one's per-position outcomes are produced — never when they are observed."""
    started = time.perf_counter()

    # ---- §5.1 + §5.2 + §5.4 metadata gauntlet, before any spool read --------
    metas, ctx, order = prepare_trial_assembly(run_id, manifests)

    # ---- §5.3 + §5.4 + §5.5/§6 — read and merge, INTERLEAVED, in `order` ----
    # The generator is passed UNMATERIALIZED on purpose: the merge pulls one
    # position at a time, so each spool is read only after the previous one has
    # been merged. `process_sharded` swaps this generator for one fed by worker
    # outcomes; the merge, and therefore every global-assembly semantic, is the
    # same object in both cases.
    return merge_validated_spools(
        run_id, ctx,
        _serial_outcomes(run_id, manifests, metas, order),
        started)


# ---------------------------------------------------------------------------
# §4 — the concrete Phase5Sink.
# ---------------------------------------------------------------------------
@dataclass
class _RunState:
    """Everything the sink accumulates for ONE run_id."""
    manifests: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # event_id -> manifest
    slots: Dict[Tuple[Any, Any], str] = field(default_factory=dict)      # (stripe, sub) -> event_id
    result: Optional[MinerTrialAssembly] = None
    consumed_commits: set = field(default_factory=set)


class AssemblingPhase5Sink(Phase5Sink):
    """The Phase-5 side of the coordinator's L6 boundary (coordinator:1287-1306).

    RESTART / RETRYABILITY CONTRACT (§4.0 [TB-D1-B2]) — exact wording:

        A failed assembly is retryable only while the same sink instance retains
        its accumulated manifests. Coordinator commit redelivery reuses those
        retained manifests; it does not republish them. Cross-process sink
        reconstruction is not provided by D1 and must not be claimed.

    SYNCHRONIZATION (§4.1 [TB-D1-B3, TB-D1-GC1]): one `threading.RLock`.
    `publish_shard`, `commit_trial`, `abort_trial` and `get_assembly` all
    synchronize through it; `commit_trial` holds it through assembly AND the
    atomic result installation, so `abort_trial` waits for an active assembly to
    finish before clearing state.

    The lock's purpose post-D1.0 is INTERNAL THREAD SAFETY and defense-in-depth
    against direct or malformed callers. Legitimate coordinator flow never
    invokes sink commit and sink abort concurrently for one run — the
    coordinator's ledger CAS + terminal-state re-read (D1.0 §2.2) is what
    provides terminal mutual exclusivity. This lock is NOT that mechanism and
    must not be described as repairing it.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: Dict[str, _RunState] = {}
        self._tombstoned: set = set()

    # ----- §4.2 publish ----------------------------------------------------
    def publish_shard(self, manifest: Dict[str, Any]) -> None:
        """Accumulate one verified shard manifest. NO spool I/O happens here —
        publish stores a canonical DEEP COPY of the manifest and nothing else;
        every staged-spool read happens at commit-time assembly."""
        with self._lock:
            run_id = manifest.get("run_id")
            # Tombstoned run: harmlessly ignore later stale manifests
            # [TB-D1-DEC2] (coordinator:1287-1297). Zero spool opens.
            if run_id in self._tombstoned:
                return
            validate_trial_metadata(manifest.get("trial_metadata") or {})
            # Deep copy BEFORE storing: the coordinator retains and returns the
            # same mutable dicts (`self.enqueued`, coordinator:1956-1958), so a
            # caller-side mutation after publication must not alter the sink's
            # future assembly input (§4.1).
            stored = _canonical_copy(manifest)
            event_id = stored.get("event_id")
            slot = (stored.get("stripe_id"), stored.get("sub_index"))
            state = self._runs.setdefault(run_id, _RunState())

            if event_id in state.manifests:
                if _canonical_json(state.manifests[event_id]) == _canonical_json(stored):
                    return                      # idempotent no-op (§4.2.4, §4.2.5)
                raise ManifestReplayConflict(
                    f"{run_id}: event_id {event_id!r} republished with DIFFERENT "
                    f"content")
            # A NEW event / NEW logical shard for an already-committed run is a
            # state violation, not a replay conflict [TB-D1-API] (§4.2.5).
            if state.result is not None:
                raise AssemblyStateError(
                    f"{run_id}: cannot publish a new shard (event_id "
                    f"{event_id!r}) to an already-committed run")
            if slot in state.slots:
                raise ManifestReplayConflict(
                    f"{run_id}: event_id {event_id!r} claims logical shard slot "
                    f"{slot!r} already held by {state.slots[slot]!r} — two event "
                    f"ids for one logical shard, even with identical bytes+SHA")
            state.manifests[event_id] = stored
            state.slots[slot] = event_id

    # ----- §4.3 commit -----------------------------------------------------
    def commit_trial(self, event: Dict[str, Any]) -> None:
        """Assemble the retained manifests and install the result ATOMICALLY.

        Assembly is stored exactly once SUCCESSFULLY. A successful duplicate
        commit event is an idempotent no-op. If assembly raises, no completed
        result and no consumed-commit marker is stored, the accumulated manifests
        are RETAINED (§4.0), and redelivery of the same commit event attempts
        assembly again — the coordinator converts the raise into
        `delivery: "failed"` and redelivers the SAME event_id."""
        with self._lock:
            run_id = event.get("run_id")
            event_id = event.get("event_id")
            if run_id in self._tombstoned:
                raise AssemblyStateError(
                    f"{run_id}: commit refused — the run is tombstoned (aborted)")
            state = self._runs.get(run_id)
            if state is not None and event_id in state.consumed_commits:
                return          # idempotent: zero spool opens, zero map construction
            if state is not None and state.result is not None:
                raise AssemblyStateError(
                    f"{run_id}: commit event_id {event_id!r} differs from the "
                    f"consumed commit event(s) {sorted(state.consumed_commits)!r} "
                    f"for an already-committed run; no replacement assembly")
            if state is None or not state.manifests:
                raise AssemblyStateError(
                    f"{run_id}: commit with zero retained manifests")
            # 1-2. assemble + validate entirely in LOCAL temporary state.
            try:
                assembly = assemble_trial(run_id, list(state.manifests.values()))
            except Exception:
                # Delete temporary assembly/result state ONLY — the accumulated
                # manifests are retained for redelivery (§4.0/§4.3).
                state.result = None
                state.consumed_commits.discard(event_id)
                raise
            # 3. install the finished assembly; 4. ONLY THEN record the event.
            state.result = assembly
            state.consumed_commits.add(event_id)

    # ----- §4.4 abort ------------------------------------------------------
    def abort_trial(self, event: Dict[str, Any]) -> None:
        """SYNCHRONOUS discharge (L7, Option A). Takes the sink lock — waiting for
        any active assembly to finish — then on return ALL of the following hold:
        the run's accumulated manifests are discarded; any partial or completed
        assembly is discarded; the sink holds NO reference to any trial-owned
        staged path; the run_id is tombstoned so later stale `publish_shard`
        calls are ignored with zero spool opens.

        Idempotent: aborting an unknown or already-aborted run is a successful
        no-op."""
        with self._lock:
            run_id = event.get("run_id")
            self._runs.pop(run_id, None)
            self._tombstoned.add(run_id)

    # ----- §4.5 the frozen D6 accessor -------------------------------------
    def get_assembly(self, run_id: str) -> Optional[MinerTrialAssembly]:
        """The completed assembly, or None before a successful commit and after
        an abort — NEVER a partial result. Signature frozen for D6, which fails
        closed on None."""
        with self._lock:
            state = self._runs.get(run_id)
            return None if state is None else state.result
