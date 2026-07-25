#!/usr/bin/env python3
"""
canonical_arrays.py — S172 Phase-5 Deliverable D3: the shared, backend-neutral
24-field-record -> 22-typed-array columnizer, plus an independently callable
structural validator.

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3.md (REV3, Team Beta
approved), frozen against HEAD 66f0425.

SCOPE — exactly one transformation:

    canonical 24-field record sequence  ->  exact 22 typed NumPy arrays

THE ORDERING DOCTRINE (REV3 §0, correction [C1]) — the single most important
property of this module:

    D3    converts rows.           (preserves the caller's sequence EXACTLY)
    D3.25 orders candidate rows.   (trial-major, mode-minor at ingress)
    D3.5  orders final winner rows. (globally seed-ascending)

`records_to_arrays()` therefore performs NO sorting whatsoever — not by seed,
mode, trial, score, or PRNG identity. Input record *i* becomes output row *i*
in every one of the 22 arrays. Taking ownership of an ordering policy here
would silently undo D3.5's required global seed order.

WHAT THIS MODULE DELIBERATELY DOES NOT DO (REV3 §0):
  * it wires itself into no live path — the existing `_survivors_to_arrays`
    closure in window_optimizer_integration_final.py and the array block in
    convert_survivors_to_binary.py stay in place and in use until D3.5;
  * it performs no L2/L3 selection, loads no prior file, and writes, replaces
    or names no canonical artifact;
  * it populates neither `binary_npz_path` nor `all_npz_path` (both remain
    deprecated and permanently None, Beta Ruling E);
  * it touches neither D6's adapter nor WATCHER.

RULING G (REV3 §3a) — the strict path takes NO directional fallback. Both
`forward_match_rate` and `reverse_match_rate` are required; there are no
aliases, no opposite-direction substitution, and no 0.0 default. The
same-direction alias tolerance belongs ONLY to the explicitly historical legacy
seams, and is corrected there in D3.0-B (a separate commit, not part of D3).

Sibling module: `utils/canonical_records.py` (D3.25) will map
maps + trial context -> 24-field records; this module maps
24-field records -> the typed 22-array bundle.
"""
from __future__ import annotations

import math
import numbers
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np

from prng_registry import KERNEL_REGISTRY
from utils.prng_encoding import encode_prng_type, encode_skip_mode

__all__ = [
    "CANONICAL_ARRAY_CONTRACT",
    "CANONICAL_RECORD_FIELDS",
    "BASE_PRNG_FAMILIES",
    "CanonicalRecordError",
    "ArrayBundleError",
    "records_to_arrays",
    "validate_array_bundle",
]


# ---------------------------------------------------------------------------
# Errors. Both derive from ValueError so a caller that only knows the broad
# contract ("bad input raises ValueError") keeps working, while a caller that
# wants to distinguish a record-level rejection from a bundle-level structural
# violation can. ValueErrors raised by `utils/prng_encoding` propagate
# UNWRAPPED by design (REV3 §4.4) — the canonical registry hard-fail is the
# single source of truth for identity membership and must not be re-labelled.
# ---------------------------------------------------------------------------
class CanonicalRecordError(ValueError):
    """A canonical 24-field input record violated the D3 input contract."""


class ArrayBundleError(ValueError):
    """An array bundle violated the frozen 22-array structural contract."""


# ---------------------------------------------------------------------------
# The frozen 22-array contract (REV3 §2), verified at HEAD 66f0425 against
# convert_survivors_to_binary._EMPTY_NPZ_DTYPES (:50-73) and the identical
# `np.savez_compressed` call order (:200-223).
#
# Every dtype is normalized through np.dtype(...) [C7] so comparisons are
# unambiguous between `np.float32` (a type object) and `np.dtype("float32")`
# (a dtype instance); `validate_array_bundle` compares `array.dtype` against
# the normalized object.
#
# The six `*_count` arrays are float32 DESPITE being logically integral —
# reproduced exactly because the on-disk NPZ schema is frozen. §4.5 restores
# the lost integrality as an input-validation rule instead.
# ---------------------------------------------------------------------------
CANONICAL_ARRAY_CONTRACT: Tuple[Tuple[str, np.dtype], ...] = tuple(
    (_name, np.dtype(_dtype)) for _name, _dtype in (
        ("seeds",                     "uint32"),
        ("forward_matches",           "float32"),
        ("reverse_matches",           "float32"),
        ("window_size",               "int32"),
        ("offset",                    "int32"),
        ("trial_number",              "int32"),
        ("skip_min",                  "int32"),
        ("skip_max",                  "int32"),
        ("skip_range",                "int32"),
        ("forward_count",             "float32"),
        ("reverse_count",             "float32"),
        ("bidirectional_count",       "float32"),
        ("intersection_count",        "float32"),
        ("intersection_ratio",        "float32"),
        ("intersection_weight",       "float32"),
        ("bidirectional_selectivity", "float32"),
        ("forward_only_count",        "float32"),
        ("reverse_only_count",        "float32"),
        ("survivor_overlap_ratio",    "float32"),
        ("score",                     "float32"),
        ("skip_mode",                 "uint8"),
        ("prng_type",                 "uint8"),
    )
)

_ARRAY_NAMES: Tuple[str, ...] = tuple(_n for _n, _d in CANONICAL_ARRAY_CONTRACT)

_DTYPE_BY_ARRAY: Dict[str, np.dtype] = dict(CANONICAL_ARRAY_CONTRACT)


# ---------------------------------------------------------------------------
# The canonical 24-field input record (REV3 §2 / §4.3).
#
# Held here rather than imported from `miner/range_miner_npz_writer.py` on
# purpose: D3 is backend-neutral and lives under `utils/`, so it must not take
# a dependency on the miner package (utils -> miner is the wrong direction, and
# D3.25 will produce these records from a different module entirely). The gate
# harness hand-transcribes the same list a third time, independently.
#
# 24 - 2 = 22: `sessions` and `prng_base` do NOT become arrays (they are
# validated anyway, §4.4); `forward_match_rate` -> `forward_matches` and
# `reverse_match_rate` -> `reverse_matches` are RENAMED; the other 22 map 1:1.
# ---------------------------------------------------------------------------
CANONICAL_RECORD_FIELDS: Tuple[str, ...] = (
    "seed", "forward_match_rate", "reverse_match_rate", "score",
    "window_size", "offset", "skip_min", "skip_max", "skip_range", "sessions",
    "trial_number", "prng_base", "skip_mode", "prng_type",
    "forward_count", "reverse_count", "bidirectional_count",
    "intersection_count", "intersection_ratio",
    "forward_only_count", "reverse_only_count",
    "survivor_overlap_ratio", "bidirectional_selectivity", "intersection_weight",
)

_RECORD_FIELD_SET = frozenset(CANONICAL_RECORD_FIELDS)

# The two renames; every other array reads the identically-named record field.
_RENAMED_SOURCE_FIELDS: Dict[str, str] = {
    "seeds":           "seed",
    "forward_matches": "forward_match_rate",
    "reverse_matches": "reverse_match_rate",
}

_SOURCE_FIELD_BY_ARRAY: Dict[str, str] = {
    _name: _RENAMED_SOURCE_FIELDS.get(_name, _name) for _name in _ARRAY_NAMES
}


# ---------------------------------------------------------------------------
# Base-family restriction [A1] — registry membership alone is INSUFFICIENT for
# `prng_base`.
#
# `java_lcg_reverse`, `java_lcg_hybrid` and `java_lcg_hybrid_reverse` are all
# valid KERNEL_REGISTRY identities, but none is a valid `prng_base`: a record
# like
#
#     prng_base = "java_lcg_reverse"  skip_mode = "constant"  prng_type = "java_lcg_reverse"
#
# satisfies the equality rule of §4.4 and is still semantically invalid, because
# `prng_type` is a MODE label, not a directional identity. So `prng_base` must
# be a FORWARD, NON-HYBRID base family.
#
# Derived from the registry, never hardcoded (CLAUDE.md rule 4). The longest
# suffix is tested first so `_hybrid_reverse` is not mis-classified.
# ---------------------------------------------------------------------------
_DERIVED_IDENTITY_SUFFIXES: Tuple[str, ...] = (
    "_hybrid_reverse", "_hybrid", "_reverse",
)

BASE_PRNG_FAMILIES = frozenset(
    _key for _key in KERNEL_REGISTRY
    if not any(_key.endswith(_sfx) for _sfx in _DERIVED_IDENTITY_SUFFIXES)
)


# ---------------------------------------------------------------------------
# Destination integer ranges (§4.5). Derived from the frozen contract's dtypes
# via np.iinfo rather than written out, so a contract dtype change cannot leave
# a stale literal bound behind.
# ---------------------------------------------------------------------------
_UINT32 = np.iinfo(np.uint32)
_INT32 = np.iinfo(np.int32)

# array name -> (low, high) inclusive, for the seven integer record fields.
_INT_FIELD_RANGE: Dict[str, Tuple[int, int]] = {
    "seeds":        (int(_UINT32.min), int(_UINT32.max)),
    "window_size":  (int(_INT32.min), int(_INT32.max)),
    "offset":       (int(_INT32.min), int(_INT32.max)),
    "trial_number": (int(_INT32.min), int(_INT32.max)),
    "skip_min":     (int(_INT32.min), int(_INT32.max)),
    "skip_max":     (int(_INT32.min), int(_INT32.max)),
    "skip_range":   (int(_INT32.min), int(_INT32.max)),
}

# The six logical counts: float32 columns ONLY because the frozen NPZ schema
# says so. They are nonnegative and integer-valued [A2]; accepting an arbitrary
# nonnegative float would quietly demote a canonical count to a generic
# measurement merely because its destination column happens to be float32.
_COUNT_ARRAYS = frozenset({
    "forward_count", "reverse_count", "bidirectional_count",
    "intersection_count", "forward_only_count", "reverse_only_count",
})

# Fields bounded to the closed unit interval.
_UNIT_INTERVAL_ARRAYS = frozenset({
    "forward_matches", "reverse_matches", "score",
})

# Nonnegative, NOT ceilinged at 1 [A4]: `bidirectional_selectivity` is
# len(fwd)/max(len(rev),1) and may legitimately exceed 1 (100 forward over 10
# reverse -> 10.0). `intersection_weight` is bounded by its own formula. A
# single generic `<= 1` ceiling across every ratio/weight/selectivity field
# would be wrong. Applying only the frozen bounds is a bounds-application rule
# and NOT a licence to re-derive an aggregate — D3 never recomputes one.
_NONNEGATIVE_ARRAYS = frozenset({
    "intersection_ratio", "survivor_overlap_ratio",
    "intersection_weight", "bidirectional_selectivity",
})

_FLOAT_ARRAYS = _UNIT_INTERVAL_ARRAYS | _COUNT_ARRAYS | _NONNEGATIVE_ARRAYS


# ---------------------------------------------------------------------------
# Per-record helpers
# ---------------------------------------------------------------------------
def _field(record: Mapping[str, object], index: int, name: str) -> Any:
    """Strict canonical field access (§4.3): no defaults, no aliases.

    Every read of a canonical field funnels through here so the strictness is
    stated once. `records_to_arrays` has already proven the key set is exactly
    the canonical 24, so a KeyError here would be an internal invariant break.
    """
    return record[name]


def _check_key_set(record: Mapping[str, object], index: int) -> None:
    """Strict, complete 24-field key-set contract [C2].

    Both directions fail closed: a MISSING key is a producer defect, and an
    EXTRA key means an upstream schema extension would otherwise vanish
    silently during the 24 -> 22 conversion.
    """
    if not isinstance(record, Mapping):
        raise CanonicalRecordError(
            f"record {index}: expected a Mapping of the canonical 24 fields, "
            f"got {type(record).__name__}."
        )
    keys = set(record.keys())
    missing = _RECORD_FIELD_SET - keys
    if missing:
        raise CanonicalRecordError(
            f"record {index}: missing canonical field(s) {sorted(missing)}. "
            f"The D3 input contract is the exact 24-field canonical record; "
            f"there is no default and no alias (Ruling G, REV3 §3a)."
        )
    extra = keys - _RECORD_FIELD_SET
    if extra:
        raise CanonicalRecordError(
            f"record {index}: unexpected field(s) {sorted(extra)} outside the "
            f"canonical 24. Failing closed so an upstream schema extension "
            f"cannot silently disappear in the 24 -> 22 conversion."
        )


def _check_sessions(record: Mapping[str, object], index: int) -> None:
    """`sessions` never becomes an array but is validated anyway [C4].

    Must be list[str]. A scalar string, a tuple, None, or a non-string member
    all fail closed. D3.25 normalizes tuple / None inputs BEFORE creating a
    canonical record — that normalization is emphatically not D3's job.
    """
    sessions = _field(record, index, "sessions")
    if not isinstance(sessions, list):
        raise CanonicalRecordError(
            f"record {index}: 'sessions' must be a list of str, got "
            f"{type(sessions).__name__}. A bare string, a tuple and None are "
            f"all rejected; D3.25 normalizes before creating the record."
        )
    for pos, item in enumerate(sessions):
        if not isinstance(item, str):
            raise CanonicalRecordError(
                f"record {index}: 'sessions'[{pos}] must be str, got "
                f"{type(item).__name__}."
            )


def _check_identity(record: Mapping[str, object], index: int) -> Tuple[str, str]:
    """`prng_base` / `skip_mode` / `prng_type` consistency [C3][C4][A1].

    Returns the validated (skip_mode, prng_type) strings.

    The equality rule (§4.4):

        skip_mode == "constant"  ->  prng_type == prng_base
        skip_mode == "variable"  ->  prng_type == prng_base + "_hybrid"

    There is deliberately NO prng_type -> prng_base derivation here (§4.3):
    D1.1 and D3.25 both emit an explicit `prng_type`, so accepting its absence
    would weaken the boundary for no production requirement. If a historical
    conversion ever needs that derivation it belongs in a separately named
    compatibility adapter, never hidden inside `records_to_arrays()`.
    """
    prng_base = _field(record, index, "prng_base")
    skip_mode = _field(record, index, "skip_mode")
    prng_type = _field(record, index, "prng_type")

    for name, value in (("prng_base", prng_base), ("skip_mode", skip_mode),
                        ("prng_type", prng_type)):
        if not isinstance(value, str) or not value:
            raise CanonicalRecordError(
                f"record {index}: {name!r} must be a nonempty str, got "
                f"{value!r} ({type(value).__name__})."
            )

    # [A1] forward, non-hybrid base family — registry membership is NOT enough.
    if prng_base not in BASE_PRNG_FAMILIES:
        raise CanonicalRecordError(
            f"record {index}: 'prng_base' {prng_base!r} is not a forward, "
            f"non-hybrid base family. Directional and derived registry "
            f"identities (*_reverse, *_hybrid, *_hybrid_reverse) are valid "
            f"KERNEL_REGISTRY keys but invalid prng_base values — 'prng_type' "
            f"is a MODE label, not a directional identity [REV3 A1]."
        )

    # Equality rule. `encode_skip_mode` below is the authority on the legal
    # skip_mode vocabulary; this branch only needs the two canonical modes.
    if skip_mode == "constant":
        expected_prng_type = prng_base
    elif skip_mode == "variable":
        expected_prng_type = prng_base + "_hybrid"
    else:
        # Delegate the vocabulary hard-fail to the canonical encoder so the
        # error text stays single-sourced; its ValueError propagates unwrapped.
        encode_skip_mode(skip_mode)
        raise CanonicalRecordError(  # pragma: no cover - encoder raises first
            f"record {index}: unreachable skip_mode {skip_mode!r}."
        )

    if prng_type != expected_prng_type:
        raise CanonicalRecordError(
            f"record {index}: identity inconsistency — skip_mode "
            f"{skip_mode!r} with prng_base {prng_base!r} requires prng_type "
            f"{expected_prng_type!r}, got {prng_type!r}."
        )

    return skip_mode, prng_type


def _check_int(record: Mapping[str, object], index: int, array_name: str) -> int:
    """Integer field validation in PYTHON space, before materialization [C5].

    NumPy would otherwise silently narrow or wrap an out-of-range value into
    the destination dtype, so the range check has to happen here. `bool` is
    excluded explicitly: `isinstance(True, int)` is True in Python, and a
    boolean reaching an integer column is a producer defect, not a 1.
    """
    field = _SOURCE_FIELD_BY_ARRAY[array_name]
    value = _field(record, index, field)
    if isinstance(value, bool):
        raise CanonicalRecordError(
            f"record {index}: {field!r} is a bool ({value!r}); a boolean is "
            f"not an acceptable integer value for the {array_name!r} column."
        )
    if not isinstance(value, numbers.Integral):
        raise CanonicalRecordError(
            f"record {index}: {field!r} must be an integer, got {value!r} "
            f"({type(value).__name__})."
        )
    value = int(value)
    low, high = _INT_FIELD_RANGE[array_name]
    if not (low <= value <= high):
        raise CanonicalRecordError(
            f"record {index}: {field!r} = {value} is outside the destination "
            f"range [{low}, {high}] of the {array_name!r} "
            f"({_DTYPE_BY_ARRAY[array_name].name}) column. "
            f"Failing closed rather than letting NumPy wrap silently."
        )
    return value


def _check_float(record: Mapping[str, object], index: int, array_name: str) -> float:
    """Float field validation in PYTHON space, before materialization [C5][A2].

    Five checks, in the order frozen by REV3 §4.5:
      1. numeric;
      2. bool excluded;
      3. the PYTHON value is finite;
      4. the CONVERTED value is also finite — np.isfinite(np.float32(value)).
         Python-level finiteness does NOT prove float32 representability: a
         large but finite Python float (1e300) becomes inf under np.float32,
         which violates the output contract even though check 3 passed;
      5. the field-specific bounds.
    """
    field = _SOURCE_FIELD_BY_ARRAY[array_name]
    value = _field(record, index, field)

    # 1. numeric
    if not isinstance(value, numbers.Real):
        raise CanonicalRecordError(
            f"record {index}: {field!r} must be a real number, got {value!r} "
            f"({type(value).__name__})."
        )
    # 2. bool excluded. This check cannot be folded into check 1: `bool` IS a
    #    `numbers.Real` in Python, so check 1 admits True/False by construction
    #    and the exclusion has to be stated separately.
    if isinstance(value, bool):
        raise CanonicalRecordError(
            f"record {index}: {field!r} is a bool ({value!r}); a boolean is "
            f"not an acceptable numeric value for the {array_name!r} column."
        )
    value = float(value)
    # 3. the Python value is finite
    if not math.isfinite(value):
        raise CanonicalRecordError(
            f"record {index}: {field!r} = {value!r} is not finite; NaN and "
            f"+/-inf are rejected before materialization."
        )
    # 4. the float32-converted value is ALSO finite
    if not bool(np.isfinite(np.float32(value))):
        raise CanonicalRecordError(
            f"record {index}: {field!r} = {value!r} is finite in Python but "
            f"overflows to infinity under np.float32, which would violate the "
            f"frozen float32 contract of the {array_name!r} column [REV3 A2]."
        )

    # 5. field-specific bounds — ONLY those frozen in §4.5.
    if array_name in _UNIT_INTERVAL_ARRAYS:
        if not (0.0 <= value <= 1.0):
            raise CanonicalRecordError(
                f"record {index}: {field!r} = {value!r} is outside the frozen "
                f"bound [0.0, 1.0] for the {array_name!r} column."
            )
    elif array_name in _COUNT_ARRAYS:
        if value < 0.0:
            raise CanonicalRecordError(
                f"record {index}: {field!r} = {value!r} is a negative count; "
                f"the six logical count columns are nonnegative."
            )
        if value != math.floor(value):
            raise CanonicalRecordError(
                f"record {index}: {field!r} = {value!r} is not integer-valued. "
                f"The six count columns are float32 ONLY because the frozen "
                f"NPZ schema requires it; they remain logical counts [REV3 A2]."
            )
    elif array_name in _NONNEGATIVE_ARRAYS:
        if value < 0.0:
            raise CanonicalRecordError(
                f"record {index}: {field!r} = {value!r} is negative; the "
                f"{array_name!r} column is bounded below by 0.0. No generic "
                f"<= 1 ceiling applies — bidirectional_selectivity may exceed "
                f"1 legitimately [REV3 A4]."
            )
    else:  # pragma: no cover - every float array is in exactly one bound set
        raise AssertionError(f"unbounded float array {array_name!r}")

    return value


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
def records_to_arrays(records: Iterable[Mapping[str, object]]) -> Dict[str, np.ndarray]:
    """Convert canonical 24-field records into the frozen 22-array bundle.

    STRICTLY ORDER-PRESERVING [C1]: input record *i* becomes output row *i* in
    every one of the 22 arrays. No sorting by seed, mode, trial, score or PRNG
    identity is performed. Shuffling the input produces the same corresponding
    shuffle in all 22 arrays. Ordering policy belongs to D3.25 (ingress) and
    D3.5 (final winners), never here.

    REPEATED SEEDS ARE LEGAL (§4.2): a candidate bundle may carry the same seed
    once per mode — cross-mode duplication is legitimate until L2. There is no
    unique-seed and no strictly-increasing-seed wall.

    `records` is an ITERABLE, not a Sequence [A3]: the contract is a single
    forward traversal, requiring neither len() nor indexing. It is consumed
    exactly once, so a generator is a first-class input.

    ONE LOGICAL PASS (§4.6): one accumulation list per output array; the input
    is iterated exactly once, validating and appending during that traversal;
    each array is materialized once afterward.

    Raises:
        CanonicalRecordError: any input-contract violation, naming the record
            index and the offending field.
        ArrayBundleError: the postcondition self-check failed (an internal
            invariant break, not a caller error).
        ValueError: propagated UNWRAPPED from `utils.prng_encoding` when an
            identity is absent from the canonical registry.
    """
    columns: Dict[str, list] = {name: [] for name in _ARRAY_NAMES}

    for index, record in enumerate(records):
        # --- structural + semantic validation, all before any conversion ----
        _check_key_set(record, index)
        _check_sessions(record, index)
        skip_mode, prng_type = _check_identity(record, index)

        # Registry membership stays the canonical encoders' job; their
        # ValueError propagates unwrapped (§4.4).
        skip_mode_id = encode_skip_mode(skip_mode)
        prng_type_id = encode_prng_type(prng_type)

        # --- numeric validation in Python space, then append (§4.5/§4.6) ----
        for array_name in _ARRAY_NAMES:
            if array_name in _INT_FIELD_RANGE:
                columns[array_name].append(_check_int(record, index, array_name))
            elif array_name in _FLOAT_ARRAYS:
                columns[array_name].append(_check_float(record, index, array_name))
            elif array_name == "skip_mode":
                columns[array_name].append(skip_mode_id)
            elif array_name == "prng_type":
                columns[array_name].append(prng_type_id)
            else:  # pragma: no cover - the 22 names are exhaustively covered
                raise AssertionError(f"unclassified array {array_name!r}")

    arrays: Dict[str, np.ndarray] = {
        name: np.array(columns[name], dtype=dtype)
        for name, dtype in CANONICAL_ARRAY_CONTRACT
    }
    validate_array_bundle(arrays)
    return arrays


def validate_array_bundle(arrays: Mapping[str, np.ndarray]) -> None:
    """Independently callable structural validator for a 22-array bundle [C6].

    Enforces: exactly 22 keys; the exact key names; the exact ITERATION order
    (`tuple(arrays.keys())`, not set equality — D3.0's E8 lesson); the exact
    dtype per array, compared against the normalized `np.dtype`; equal lengths;
    and that every array is ONE-DIMENSIONAL. A (N, 1) array has a matching
    outer length but is not contract-compatible with a 1-D NPZ column.

    `records_to_arrays()` calls this before returning as a postcondition
    self-check; the independently callable form stays necessary because D3.5
    will validate bundles assembled through other paths.

    Raises:
        ArrayBundleError: on any violation.
    """
    if not isinstance(arrays, Mapping):
        raise ArrayBundleError(
            f"array bundle must be a Mapping, got {type(arrays).__name__}."
        )

    observed = tuple(arrays.keys())
    if len(observed) != len(_ARRAY_NAMES):
        raise ArrayBundleError(
            f"array bundle has {len(observed)} arrays, expected exactly "
            f"{len(_ARRAY_NAMES)}. Observed: {list(observed)}."
        )
    if observed != _ARRAY_NAMES:
        missing = [n for n in _ARRAY_NAMES if n not in observed]
        extra = [n for n in observed if n not in _ARRAY_NAMES]
        raise ArrayBundleError(
            f"array bundle key order/name mismatch. Expected "
            f"{list(_ARRAY_NAMES)}, got {list(observed)}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else "")
            + "."
        )

    length = None
    for name, dtype in CANONICAL_ARRAY_CONTRACT:
        array = arrays[name]
        if not isinstance(array, np.ndarray):
            raise ArrayBundleError(
                f"array {name!r} must be a numpy.ndarray, got "
                f"{type(array).__name__}."
            )
        if array.ndim != 1:
            raise ArrayBundleError(
                f"array {name!r} has ndim {array.ndim} (shape {array.shape}); "
                f"every contract array must be one-dimensional. A (N, 1) "
                f"array matches on outer length but is not a 1-D NPZ column."
            )
        if array.dtype != dtype:
            raise ArrayBundleError(
                f"array {name!r} has dtype {array.dtype}, expected {dtype}."
            )
        if length is None:
            length = array.shape[0]
        elif array.shape[0] != length:
            raise ArrayBundleError(
                f"array {name!r} has length {array.shape[0]}, expected "
                f"{length} — the bundle is not rectangular."
            )
