#!/usr/bin/env python3
"""
checkpoint_d6_2.py — S172 Phase-5 Deliverable D6.2: the 24-field canonical
accumulator checkpoint, its two digests, the run-id-only resume selector, the
nine-row mixed-pair recovery matrix, and canonical reconciliation.

Spec: `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md`
(REV5) as amended by `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_REV5_BINDING_ADDENDUM.md`
(BINDING; where the two differ the addendum wins).

WHAT CHANGED FROM D6.1
======================
D6.1 wrote a NON-AUTHORITATIVE four-field snapshot (`s172-d6.1-four-field-v1`)
and left `_FLUSH_CLEAR_IN_MEMORY = False`, because four arrays cannot restore
the 24 `CANONICAL_RECORD_FIELDS` the D3.5 finalizer consumes. D6.2 replaces
that payload with the complete canonical state, so the clear becomes safe and
the finalizer receives complete 24-field input **through the resume path, not a
truncated stump**.

THE ASYMMETRIC ARCHITECTURE (REV5 §0, settled — do not reopen)
==============================================================
  * **Member A is a marker / compatibility stub.** It carries `seeds`, `score`
    and its complete identity block, and NOTHING more. It is never an
    accumulator backup, is never described as one, and no path here consumes it
    as one.
  * **Member B is the sole recovery payload.** Loss or corruption of B is
    unrecoverable and fails closed.
  * Pair validation is still required before any in-memory clear.

THE TWO DIGESTS ARE SEPARATE IDENTITIES (REV5 §3)
=================================================
  * `canonical_state_digest` covers ONLY the complete canonical record state,
    in a fixed physical array order over globally seed-sorted rows. It covers
    no identity field, is stored in BOTH members, is recomputed and verified by
    B, and is merely BOUND by A's marker.
  * `member_content_digest` covers every persisted field of that member EXCEPT
    itself — including every identity field, and therefore including
    `canonical_state_digest` (addendum §1). It is computed LAST, over a FIXED
    field order that is never dictionary or NPZ iteration order.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
=========================================
  * it does not restore the optimizer execution cursor (REV5 §0) — a resumed
    run continues optimization under its own trial namespace, and nothing here
    claims otherwise;
  * it does not fork `_l2_sort_key` / `_select_l2_winners` — they are IMPORTED
    from `utils.run_finalizer` (REV5 §6.2, Ruling D, frozen);
  * it does not reimplement the three finalizer walls — `_validate_raw_candidates`,
    `_validate_candidate_coverage` and `_validate_candidate_identity` are
    likewise imported (REV5 §7.2);
  * it does not transcribe a dtype table or a categorical map — every storage
    dtype is derived from `CANONICAL_ARRAY_CONTRACT` and every categorical code
    goes through `utils.prng_encoding` (REV5 §2.1, G-ENCODING-AUTHORITY);
  * it performs NO newest-directory discovery at any layer, and it never
    reconstructs a scalar `sessions` string into `[scalar]`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# --- imported authorities. NONE of these is reimplemented or forked. ---------
from utils.canonical_arrays import (
    CANONICAL_ARRAY_CONTRACT,
    records_to_arrays,
)
from utils.canonical_arrays import _SOURCE_FIELD_BY_ARRAY  # noqa: F401 (private
# by name only — REV5 §2.1 requires importing the frozen contract rather than
# transcribing the record-field -> array-name rename here)
from utils.canonical_records import (
    CANONICAL_RECORD_FIELDS,
    CANONICAL_SKIP_MODES,
    canonical_sessions,
)
from utils.prng_encoding import (
    ENCODING_VERSION,
    decode_prng_type,
    decode_skip_mode,
    encode_prng_type,
    encode_skip_mode,
)
from utils.run_finalizer import (
    AccumulatorConsistencyError,
    canonical_map_hash,
)
from utils.run_finalizer import (
    _l2_sort_key,                    # noqa: F401 — imported, never forked
    _select_l2_winners,
    _validate_candidate_coverage,
    _validate_candidate_identity,
    _validate_raw_candidates,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_DIRNAME",
    "MEMBER_A_NAME",
    "MEMBER_B_NAME",
    "MEMBER_A_ROLE",
    "MEMBER_B_ROLE",
    "MEMBER_A_PAYLOAD_FIELDS",
    "IDENTITY_KEYS",
    "TRANSACTION_INVARIANT_KEYS",
    "STATE_PHYSICAL_ORDER",
    "RUN_CONTEXT_DIGEST_VERSION",
    "CheckpointError",
    "CheckpointSchemaError",
    "CheckpointIdentityError",
    "CheckpointRecoveryError",
    "CheckpointSelectorError",
    "RunContext",
    "RecoveryOutcome",
    "build_run_context_digest",
    "validate_run_id",
    "resolve_checkpoint_dir",
    "canonical_state_arrays",
    "canonical_state_digest",
    "member_content_digest",
    "build_member_arrays",
    "decode_member_b",
    "read_member",
    "recover_checkpoint",
    "reconcile",
    "canonicalize_record",
    "validate_new_raw_records",
    "write_transaction",
    "validate_installed_pair",
]


# ===========================================================================
# Errors
# ===========================================================================
class CheckpointError(RuntimeError):
    """Any D6.2 checkpoint failure. Every one of them is fail-closed."""


class CheckpointSchemaError(CheckpointError):
    """A member violated the D6.2 storage schema (dtype, CSR, field set)."""


class CheckpointIdentityError(CheckpointError):
    """A member's identity block disagrees with the run, the pair, or itself."""


class CheckpointRecoveryError(CheckpointError):
    """The pair cannot be recovered. In-memory state is NEVER cleared here."""


class CheckpointSelectorError(CheckpointError):
    """The `resume_checkpoint` selector is not an acceptable run id."""


# ===========================================================================
# §2.5 / §3.3 — names, versions and the identity block
# ===========================================================================
#: The four-field marker MUST change (REV5 §3.3). A member stamped with the
#: D6.1 version is a different format and is refused before decoding.
CHECKPOINT_SCHEMA_VERSION = "s172-d6.2-canonical-24-field-v1"

#: Unchanged from D6.1: run-isolated, git-ignored, never a finalizer-owned path.
CHECKPOINT_DIRNAME = ".s172_checkpoint"
MEMBER_A_NAME = "incremental_survivors_all.npz"
MEMBER_B_NAME = "incremental_survivors_binary.npz"

MEMBER_A_ROLE = "marker_stub"
MEMBER_B_ROLE = "recovery_payload"

#: Addendum §1 — member A's payload is EXACTLY these two arrays. Nothing more.
MEMBER_A_PAYLOAD_FIELDS: Tuple[str, ...] = ("seed", "score")

#: §3.3 — the identity block, in its FIXED declared order. `member_role` is a
#: D6.2 addition to §3.3's table and is stated as such in the report: the two
#: members are asymmetric by design, so the role belongs inside the digested
#: identity rather than being inferable only from a file name an attacker or a
#: careless copy could swap.
IDENTITY_KEYS: Tuple[str, ...] = (
    "checkpoint_schema_version",
    "checkpoint_id",
    "checkpoint_sequence",
    "run_id",
    "logical_candidate_count",
    "encoding_version",
    "canonical_map_hash",
    "run_context_digest",
    "canonical_state_digest",
    "member_role",
    "member_content_digest",
)

#: §3.3 — what a normal installed pair AGREES on. `member_role` and
#: `member_content_digest` are excluded because they differ BY DESIGN; agreement
#: on them is never required and their difference is explicitly tolerated.
TRANSACTION_INVARIANT_KEYS: Tuple[str, ...] = (
    "checkpoint_schema_version",
    "checkpoint_id",
    "checkpoint_sequence",
    "run_id",
    "logical_candidate_count",
    "encoding_version",
    "canonical_map_hash",
    "run_context_digest",
    "canonical_state_digest",
)

#: The identity fields that must be equal across two DIFFERENT transactions of
#: the same run — i.e. what "all invariant identities agree" means in the
#: recovery matrix (§5 / addendum §2 rows 4 and 5). `checkpoint_id`, sequence,
#: `logical_candidate_count` and `canonical_state_digest` legitimately differ
#: between two transactions and are therefore NOT here.
RUN_INVARIANT_KEYS: Tuple[str, ...] = (
    "checkpoint_schema_version",
    "run_id",
    "encoding_version",
    "canonical_map_hash",
    "run_context_digest",
)

_INT_IDENTITY_KEYS = frozenset({"checkpoint_sequence", "logical_candidate_count"})

#: Domain separators. Two different preimages can never be confused for one
#: another, and neither can be confused with D6.1's digest.
_STATE_DOMAIN = b"s172.d6.2.canonical-state.v1\x00"
_MEMBER_DOMAIN = b"s172.d6.2.member-content.v1\x00"
_ARRAY_DOMAIN = b"s172.d6.2.array\x00"

RUN_CONTEXT_DIGEST_VERSION = "s172-d6.2-run-context-v1"


# ===========================================================================
# §2.2 — the 24 fields and their storage dtypes, DERIVED not transcribed
# ===========================================================================
# `CANONICAL_ARRAY_CONTRACT` is stated in the ARRAY domain, where
# `forward_match_rate`/`reverse_match_rate` are renamed to
# `forward_matches`/`reverse_matches` and `seed` to `seeds`. The checkpoint
# stores RECORD field names (REV5 §2.1: "do not apply that rename here"), so the
# contract is walked back through the frozen rename map rather than re-listed.
_DTYPE_BY_RECORD_FIELD: Dict[str, np.dtype] = {
    _SOURCE_FIELD_BY_ARRAY[_array_name]: _dtype
    for _array_name, _dtype in CANONICAL_ARRAY_CONTRACT
}

#: The two canonical record fields that are NOT stored as a single typed array:
#: `sessions` is CSR (§2.4) and `prng_base` is derived (§2.3).
_CSR_FIELD = "sessions"
_DERIVED_FIELD = "prng_base"

_SESSIONS_VALUES = "sessions_values"
_SESSIONS_OFFSETS = "sessions_offsets"
_SESSIONS_OFFSETS_DTYPE = np.dtype("int64")


def _physical_state_order() -> Tuple[str, ...]:
    """Addendum §1 — the FIXED physical order of the state preimage.

    Derived from `CANONICAL_RECORD_FIELDS` by exactly two rules, so it cannot
    drift from the canonical record: `sessions` expands in place to its two CSR
    arrays, and the derived `prng_base` is dropped (it is reconstructed from
    `prng_type` + `skip_mode` and adds no information, so hashing it would hash
    the same fact twice).

    The result is, literally:

        fields 1-9 · sessions_values · sessions_offsets ·
        trial_number · skip_mode · prng_type · fields 15-24
    """
    order: List[str] = []
    for name in CANONICAL_RECORD_FIELDS:
        if name == _CSR_FIELD:
            order.extend((_SESSIONS_VALUES, _SESSIONS_OFFSETS))
        elif name == _DERIVED_FIELD:
            continue
        else:
            order.append(name)
    return tuple(order)


STATE_PHYSICAL_ORDER: Tuple[str, ...] = _physical_state_order()

#: Everything in the state that is a plain per-record typed column.
_COLUMN_FIELDS: Tuple[str, ...] = tuple(
    n for n in STATE_PHYSICAL_ORDER
    if n not in (_SESSIONS_VALUES, _SESSIONS_OFFSETS)
)

# Structural self-check at import. The 24 canonical fields become 22 typed
# columns (one per array in the frozen contract) plus the 2 CSR arrays: exactly
# one field expands (`sessions` -> 2) and exactly one is dropped (`prng_base`),
# so the total is unchanged at 24 while the COLUMN count is 22.
if len(_COLUMN_FIELDS) != len(CANONICAL_ARRAY_CONTRACT):
    raise RuntimeError(                                     # pragma: no cover
        f"D6.2 stores {len(_COLUMN_FIELDS)} typed columns but the frozen array "
        f"contract declares {len(CANONICAL_ARRAY_CONTRACT)}.")
if len(STATE_PHYSICAL_ORDER) != len(CANONICAL_RECORD_FIELDS):
    raise RuntimeError(                                     # pragma: no cover
        f"D6.2 physical state order has {len(STATE_PHYSICAL_ORDER)} entries; "
        f"{len(CANONICAL_RECORD_FIELDS)} canonical fields expand to the same "
        f"count (sessions -> 2 CSR arrays, prng_base derived and dropped).")
_missing_dtype = [n for n in _COLUMN_FIELDS if n not in _DTYPE_BY_RECORD_FIELD]
if _missing_dtype:
    raise RuntimeError(                                     # pragma: no cover
        f"D6.2 storage dtype could not be derived for {_missing_dtype} — the "
        f"frozen array contract and the canonical record have diverged.")


# ===========================================================================
# Digest primitives — §3, exact preimages
# ===========================================================================
def _hash_array(h, name: str, arr: np.ndarray) -> None:
    """Domain separator · field name · exact dtype · exact shape · bytes.

    The SHAPE is part of the preimage on purpose: D6.1's digest omitted it
    (`window_optimizer_integration_final.py:513-528`), so two differently-shaped
    arrays holding identical bytes would have collided.
    """
    arr = np.ascontiguousarray(arr)
    h.update(_ARRAY_DOMAIN)
    h.update(name.encode("utf-8"))
    h.update(b"\x00")
    h.update(arr.dtype.str.encode("utf-8"))
    h.update(b"\x00")
    h.update(repr(tuple(int(d) for d in arr.shape)).encode("utf-8"))
    h.update(b"\x00")
    h.update(arr.tobytes())


def canonical_state_digest(state_arrays: Mapping[str, np.ndarray]) -> str:
    """§3.1 + addendum §1 — SHA-256 over the complete canonical record state.

    Covers NO identity field. Emitted in `STATE_PHYSICAL_ORDER` over rows that
    `canonical_state_arrays` has already sorted globally by seed, which is what
    makes the digest independent of arrival and flush order: two equivalent
    canonical states assembled through different interleavings produce the same
    bytes here. That is a correctness property, not a nicety — the checkpoint is
    written after arbitrary interleavings.
    """
    h = hashlib.sha256()
    h.update(_STATE_DOMAIN)
    for name in STATE_PHYSICAL_ORDER:
        if name not in state_arrays:
            raise CheckpointSchemaError(
                f"canonical state is missing {name!r}; the preimage is the "
                f"fixed physical order {list(STATE_PHYSICAL_ORDER)} and no "
                f"member of it is optional.")
        _hash_array(h, name, state_arrays[name])
    return h.hexdigest()


def _identity_scalar_array(key: str, value: Any) -> np.ndarray:
    """One identity field as a 0-d array (so `allow_pickle=False` loads it)."""
    if key in _INT_IDENTITY_KEYS:
        return np.array(int(value), dtype=np.int64)
    return np.array(str(value))


def member_content_digest(identity: Mapping[str, Any],
                          payload: Mapping[str, np.ndarray],
                          payload_order: Sequence[str]) -> str:
    """§3.2 + addendum §1 — SHA-256 over EVERY persisted field except itself.

    Included: every identity field (and therefore `canonical_state_digest`) plus
    every payload array of THIS member. Excluded: `member_content_digest` alone —
    a field cannot hash itself.

    The order is FIXED (`IDENTITY_KEYS`, then `payload_order`) and is never
    dictionary or NPZ iteration order, so a caller that hands the same content in
    a differently-ordered mapping gets the identical digest. The caller computes
    this LAST, after every other field is fixed.

    The two members' digests are EXPECTED TO DIFFER: they persist different
    payloads by design (A is a marker stub, B is the recovery payload), and
    agreement between them is never required anywhere in this module.
    """
    h = hashlib.sha256()
    h.update(_MEMBER_DOMAIN)
    for key in IDENTITY_KEYS:
        if key == "member_content_digest":
            continue                    # a field cannot hash itself
        if key not in identity:
            raise CheckpointIdentityError(
                f"identity block is missing {key!r}; the member digest covers "
                f"every persisted identity field and cannot be computed over a "
                f"partial block.")
        _hash_array(h, key, _identity_scalar_array(key, identity[key]))
    for name in payload_order:
        if name not in payload:
            raise CheckpointSchemaError(
                f"member payload is missing {name!r}.")
        _hash_array(h, name, payload[name])
    return h.hexdigest()


# ===========================================================================
# §4.1 + addendum §3 — the selector is an OPAQUE SINGLE-COMPONENT run id
# ===========================================================================
#: Addendum §3: "a conservative alphanumeric / underscore / dot / hyphen grammar
#: is appropriate". The WHOLE string is matched — a partial match would let
#: `foo/bar` through, which is exactly the handle-shaped two-API ambiguity the
#: single-component rule closes.
_RUN_ID_GRAMMAR = re.compile(r"\A[A-Za-z0-9._-]+\Z")

#: Rejected explicitly even though the grammar admits their characters.
_RESERVED_RUN_IDS = frozenset({".", ".."})


def validate_run_id(run_id: Any) -> str:
    """Addendum §3 — accept only a single-component, opaque run id.

    Path confinement ALONE is insufficient: `foo/bar` confined under the
    checkpoint root would still behave like a handle. So the grammar is an
    additional wall, and the realpath / symlink-escape checks in
    `resolve_checkpoint_dir` remain mandatory rather than being replaced by it.
    """
    if not isinstance(run_id, str):
        raise CheckpointSelectorError(
            f"resume_checkpoint must be a str run id, got "
            f"{type(run_id).__name__}.")
    if not run_id:
        raise CheckpointSelectorError(
            "resume_checkpoint is empty; an empty component is not a run id "
            "(the empty string means 'no resume' and is handled by the caller "
            "before this function).")
    if run_id in _RESERVED_RUN_IDS:
        raise CheckpointSelectorError(
            f"resume_checkpoint {run_id!r} is a relative path component, not a "
            f"run id. Bare '.' and '..' are rejected explicitly even though the "
            f"grammar admits the characters.")
    if not _RUN_ID_GRAMMAR.match(run_id):
        raise CheckpointSelectorError(
            f"resume_checkpoint {run_id!r} is not a single-component run id. "
            f"The grammar is [A-Za-z0-9._-]+ over the WHOLE string: no '/', no "
            f"alternate separator, no empty component, no absolute path, no "
            f"traversal. The selector is a run id, never a path handle.")
    return run_id


def resolve_checkpoint_dir(checkpoint_root: str, run_id: str) -> str:
    """§4.1 — resolve `<root>/.s172_checkpoint/<run_id>` and prove containment.

    Required, and all four enforced here:
      * no absolute paths and no `..` traversal — `validate_run_id` first;
      * NO newest-directory discovery, anywhere, at any layer. There is no scan,
        no glob, no mtime sort and no "most recent" fallback in this module;
      * no mutable path in `run_context_digest` (see `build_run_context_digest`);
      * reject any resolved directory escaping the checkpoint root, INCLUDING
        through a symlink — compared on `realpath`, the same way
        `_flush_assert_not_alias` already compares for the finalizer aliases.
    """
    run_id = validate_run_id(run_id)
    base = os.path.join(os.path.abspath(checkpoint_root), CHECKPOINT_DIRNAME)
    target = os.path.join(base, run_id)

    real_base = os.path.realpath(base)
    real_target = os.path.realpath(target)
    if real_target != os.path.join(real_base, run_id):
        # Covers a symlinked run directory pointing anywhere else, and a
        # symlinked checkpoint root whose children resolve outside it.
        if os.path.commonpath([real_base, real_target]) != real_base:
            raise CheckpointSelectorError(
                f"resume_checkpoint {run_id!r} resolves to {real_target!r}, "
                f"which escapes the checkpoint root {real_base!r} (symlink "
                f"escape). The realpath comparison is mandatory and is not "
                f"replaced by the run-id grammar.")
        raise CheckpointSelectorError(
            f"resume_checkpoint {run_id!r} resolves to {real_target!r} rather "
            f"than the single child {os.path.join(real_base, run_id)!r} of the "
            f"checkpoint root — refusing an indirected checkpoint directory.")
    return target


# ===========================================================================
# §4.3 — `run_context_digest`
# ===========================================================================
def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Mirrors `utils.run_finalizer._canonical_json_bytes` EXACTLY (§4.3):
    sorted keys, fixed separators, `ensure_ascii`."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def run_context_components(
    *,
    dataset_version_id: Optional[str],
    dataset_filename: str,
    dataset_sha256: str,
    repository_commit: str,
    prng_base: str,
    skip_modes_executed: Sequence[str],
    seed_start: int,
    seed_count: int,
    execution_set_id: Optional[str],
) -> Dict[str, Any]:
    """The components of `run_context_digest` — exactly these (§4.3).

    EXCLUDED AND GATED AS EXCLUDED: PID, timestamp, mutable path, and any
    newest-directory inference. D6.1's default run id embeds pid and wall time
    (`window_optimizer_integration_final.py:448`); none of that may leak in
    here, or two identical runs would fail to recognise each other's checkpoint
    and a resumed run's context could never be verified.

    The dataset contributes its IDENTITY and its DIGEST — version id, bare
    filename and sha256 — never its absolute path: the path is stable within one
    box but is not part of what makes two runs the same run, and a path is
    exactly the mutable component §4.3 excludes.
    """
    seed_start = int(seed_start)
    seed_count = int(seed_count)
    modes = [str(m) for m in skip_modes_executed]
    if not modes:
        raise CheckpointIdentityError(
            "skip_modes_executed is empty; a run always executes at least one "
            "skip mode and the ORDER is part of the context.")
    unknown = [m for m in modes if m not in CANONICAL_SKIP_MODES]
    if unknown:
        raise CheckpointIdentityError(
            f"skip_modes_executed contains non-canonical mode(s) {unknown}.")
    return {
        "run_context_digest_version": RUN_CONTEXT_DIGEST_VERSION,
        "dataset": {
            "version_id": dataset_version_id,
            "filename": str(dataset_filename),
            "sha256": str(dataset_sha256),
        },
        "repository_commit": str(repository_commit),
        "prng_base": str(prng_base),
        "skip_modes_executed": modes,          # ORDERED — not a set
        "seed_start": seed_start,
        "seed_count": seed_count,
        "seed_end": seed_start + seed_count,
        "execution_set_id": (None if execution_set_id is None
                             else str(execution_set_id)),
    }


def build_run_context_digest(components: Mapping[str, Any]) -> str:
    """Versioned canonical JSON -> SHA-256 (§4.3)."""
    return hashlib.sha256(_canonical_json_bytes(components)).hexdigest()


# ===========================================================================
# §2.3 / §2.4 — derived `prng_base`, CSR `sessions`
# ===========================================================================
_HYBRID_SUFFIX = "_hybrid"


def derive_prng_base(prng_type: str, skip_mode: str) -> str:
    """§2.3 — `constant` -> prng_type == prng_base;
    `variable` -> prng_type == prng_base + "_hybrid".

    The full identity rule is enforced at ingress by
    `utils.canonical_arrays._check_identity`; this is the inverse used when
    rebuilding a record from storage, and it fails closed rather than guessing.
    """
    if skip_mode == "constant":
        if prng_type.endswith(_HYBRID_SUFFIX):
            raise CheckpointSchemaError(
                f"constant-mode record carries hybrid prng_type {prng_type!r}; "
                f"prng_base cannot be derived from an inconsistent pair.")
        return prng_type
    if skip_mode == "variable":
        if not prng_type.endswith(_HYBRID_SUFFIX):
            raise CheckpointSchemaError(
                f"variable-mode record carries non-hybrid prng_type "
                f"{prng_type!r}; prng_base cannot be derived.")
        return prng_type[: -len(_HYBRID_SUFFIX)]
    raise CheckpointSchemaError(                            # pragma: no cover
        f"unknown skip_mode {skip_mode!r}; the vocabulary authority is "
        f"utils.prng_encoding.")


def encode_sessions_csr(records: Sequence[Mapping[str, Any]]
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """§2.4 — `sessions_values` (`<U`, flat, record order) + `sessions_offsets`
    (`int64`, length `records + 1`).

    `[]` is legal and round-trips as `[]`. A SCALAR STRING IS NEVER A SESSION
    LIST: `canonical_sessions` fails closed on one here, because the legacy
    `getattr(config, 'sessions', 'all')` fallback fabricated a session name —
    and the decoder correspondingly never reconstructs a scalar into `[scalar]`,
    because the CSR structure has no way to express one.
    """
    values: List[str] = []
    offsets: List[int] = [0]
    for record in records:
        sessions = canonical_sessions(record[_CSR_FIELD])
        values.extend(sessions)
        offsets.append(len(values))
    return (np.array(values, dtype=np.str_),
            np.array(offsets, dtype=_SESSIONS_OFFSETS_DTYPE))


def decode_sessions_csr(values: np.ndarray, offsets: np.ndarray,
                        row_count: int) -> List[List[str]]:
    """G-CSR-STRICT — every structural property checked, all fail-closed."""
    if offsets.dtype != _SESSIONS_OFFSETS_DTYPE:
        raise CheckpointSchemaError(
            f"{_SESSIONS_OFFSETS!r} dtype is {offsets.dtype}, expected "
            f"{_SESSIONS_OFFSETS_DTYPE}.")
    if offsets.ndim != 1:
        raise CheckpointSchemaError(
            f"{_SESSIONS_OFFSETS!r} must be 1-D, got shape {offsets.shape}.")
    if values.ndim != 1:
        raise CheckpointSchemaError(
            f"{_SESSIONS_VALUES!r} must be 1-D, got shape {values.shape}.")
    if values.dtype.kind != "U":
        raise CheckpointSchemaError(
            f"{_SESSIONS_VALUES!r} dtype kind is {values.dtype.kind!r}, "
            f"expected 'U'. An all-empty session set must stay Unicode; a "
            f"float64 array here means the encoder defaulted instead of "
            f"declaring the string dtype.")
    if offsets.shape[0] != row_count + 1:
        raise CheckpointSchemaError(
            f"{_SESSIONS_OFFSETS!r} has length {offsets.shape[0]}, expected "
            f"records + 1 = {row_count + 1}.")
    if int(offsets[0]) != 0:
        raise CheckpointSchemaError(
            f"{_SESSIONS_OFFSETS!r}[0] is {int(offsets[0])}, must be 0.")
    if int(offsets[-1]) != int(values.shape[0]):
        raise CheckpointSchemaError(
            f"{_SESSIONS_OFFSETS!r}[-1] is {int(offsets[-1])}, must equal "
            f"len({_SESSIONS_VALUES!r}) = {int(values.shape[0])}.")
    out: List[List[str]] = []
    for i in range(row_count):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if hi < lo:
            raise CheckpointSchemaError(
                f"{_SESSIONS_OFFSETS!r} is not monotonic at row {i}: "
                f"{lo} -> {hi}.")
        if lo < 0 or hi > int(values.shape[0]):
            raise CheckpointSchemaError(
                f"{_SESSIONS_OFFSETS!r} row {i} slice [{lo}, {hi}) is out of "
                f"range for {_SESSIONS_VALUES!r} of length "
                f"{int(values.shape[0])}.")
        out.append([str(v) for v in values[lo:hi]])
    return out


# ===========================================================================
# §6.1 step 1 — canonicalization into the checkpoint storage domains
# ===========================================================================
def canonicalize_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert one canonical 24-field record into its STORAGE domain (§6.1.1).

    Every float becomes its exact float32 value, every integer a Python int,
    every categorical a validated code round-tripped through
    `utils.prng_encoding`, and `sessions` a fresh `list[str]`. The result is
    still a canonical 24-field record and still passes D3's validator — the
    point is that a record read back off disk and a record just produced in
    memory become BIT-COMPARABLE, which is what makes the replay collapse in
    `reconcile` a true identity test rather than a fuzzy one.

    Comparing pre-rounding float64 while storing the rounded value is the defect
    this converts away (`_l2_sort_key`, Ruling D).
    """
    missing = [f for f in CANONICAL_RECORD_FIELDS if f not in record]
    if missing:
        raise CheckpointSchemaError(
            f"record is missing canonical field(s) {missing}; the checkpoint "
            f"stores the exact 24-field canonical record and has no default.")
    extra = [k for k in record if k not in CANONICAL_RECORD_FIELDS]
    if extra:
        raise CheckpointSchemaError(
            f"record carries field(s) {sorted(extra)} outside the canonical 24; "
            f"failing closed so an upstream schema extension cannot silently "
            f"disappear into the checkpoint.")

    skip_mode = record["skip_mode"]
    prng_type = record["prng_type"]
    # Round-trip through the shared codec: the vocabulary authority is
    # `utils/prng_encoding`, never a literal map here (G-ENCODING-AUTHORITY).
    skip_mode = decode_skip_mode(encode_skip_mode(skip_mode))
    prng_type = decode_prng_type(encode_prng_type(prng_type))

    out: Dict[str, Any] = {}
    for name in CANONICAL_RECORD_FIELDS:
        if name == _CSR_FIELD:
            out[name] = canonical_sessions(record[name])
            continue
        if name == "skip_mode":
            out[name] = skip_mode
            continue
        if name == "prng_type":
            out[name] = prng_type
            continue
        if name == _DERIVED_FIELD:
            out[name] = derive_prng_base(prng_type, skip_mode)
            continue
        dtype = _DTYPE_BY_RECORD_FIELD[name]
        value = record[name]
        if isinstance(value, bool):
            raise CheckpointSchemaError(
                f"{name!r} is a bool ({value!r}); True == 1 in Python, so a "
                f"Boolean reaching a numeric column is a producer defect.")
        if dtype.kind in "iu":
            if not isinstance(value, (int, np.integer)):
                raise CheckpointSchemaError(
                    f"{name!r} must be an integer for the {dtype} storage "
                    f"column, got {value!r} ({type(value).__name__}).")
            out[name] = int(value)
        else:
            out[name] = float(np.float32(value))
    return out


# ===========================================================================
# §8 step 1 — construct the cumulative canonical state
# ===========================================================================
def canonical_state_arrays(records: Sequence[Mapping[str, Any]]
                           ) -> Dict[str, np.ndarray]:
    """Build the canonical state arrays from ALREADY-CANONICALIZED records.

    Addendum §1: rows are GLOBALLY SEED-SORTED BEFORE these arrays are
    constructed. `reconcile` has already reduced the state to exactly one record
    per seed, so the seed order is a strict total order and the resulting arrays
    — and therefore `canonical_state_digest` — depend on content alone, never on
    arrival or flush order.
    """
    rows = sorted(records, key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in rows]
    if len(set(seeds)) != len(seeds):
        raise CheckpointSchemaError(          # pragma: no cover - reconcile wall
            "canonical state contains duplicate seeds; the state is the L2 "
            "winner set and carries exactly one record per seed.")

    arrays: Dict[str, np.ndarray] = {}
    for name in _COLUMN_FIELDS:
        dtype = _DTYPE_BY_RECORD_FIELD[name]
        if name == "skip_mode":
            column = [encode_skip_mode(r[name]) for r in rows]
        elif name == "prng_type":
            column = [encode_prng_type(r[name]) for r in rows]
        else:
            column = [r[name] for r in rows]
        arrays[name] = np.array(column, dtype=dtype)
    values, offsets = encode_sessions_csr(rows)
    arrays[_SESSIONS_VALUES] = values
    arrays[_SESSIONS_OFFSETS] = offsets
    return arrays


def decode_state_arrays(arrays: Mapping[str, np.ndarray]
                        ) -> List[Dict[str, Any]]:
    """The inverse of `canonical_state_arrays` — full 24-field records back.

    `prng_base` is RECONSTRUCTED from `prng_type` + `skip_mode` (§2.3) rather
    than stored, which is also why it is absent from the state preimage.
    """
    for name in STATE_PHYSICAL_ORDER:
        if name not in arrays:
            raise CheckpointSchemaError(
                f"member B is missing state array {name!r}.")
    row_count = int(arrays["seed"].shape[0])
    for name in _COLUMN_FIELDS:
        arr = arrays[name]
        expected = _DTYPE_BY_RECORD_FIELD[name]
        if arr.dtype != expected:
            raise CheckpointSchemaError(
                f"state array {name!r} has dtype {arr.dtype}, expected "
                f"{expected} — the storage domain is exact.")
        if arr.ndim != 1 or int(arr.shape[0]) != row_count:
            raise CheckpointSchemaError(
                f"state array {name!r} has shape {arr.shape}, expected "
                f"({row_count},).")
    sessions = decode_sessions_csr(arrays[_SESSIONS_VALUES],
                                   arrays[_SESSIONS_OFFSETS], row_count)

    records: List[Dict[str, Any]] = []
    for i in range(row_count):
        skip_mode = decode_skip_mode(int(arrays["skip_mode"][i]))
        prng_type = decode_prng_type(int(arrays["prng_type"][i]))
        record: Dict[str, Any] = {}
        for name in CANONICAL_RECORD_FIELDS:
            if name == _CSR_FIELD:
                record[name] = list(sessions[i])
            elif name == _DERIVED_FIELD:
                record[name] = derive_prng_base(prng_type, skip_mode)
            elif name == "skip_mode":
                record[name] = skip_mode
            elif name == "prng_type":
                record[name] = prng_type
            elif _DTYPE_BY_RECORD_FIELD[name].kind in "iu":
                record[name] = int(arrays[name][i])
            else:
                record[name] = float(arrays[name][i])
        records.append(record)
    return records


# ===========================================================================
# §6 — reconciliation
# ===========================================================================
def _replay_key(record: Mapping[str, Any]) -> Tuple[int, int, str]:
    return (int(record["seed"]), int(record["trial_number"]),
            str(record["skip_mode"]))


def reconcile(recovered: Sequence[Mapping[str, Any]],
              new_records: Sequence[Mapping[str, Any]],
              ) -> List[Dict[str, Any]]:
    """§6 — replay normalization, THEN the canonical authority.

    In order:

      1. **canonicalize both sides into the checkpoint storage domains** (float32
         score, uint8 codes, ...), so a record read back off disk and a record
         just produced in memory are directly comparable;
      2. **collapse a bit-identical 24-field replay** before winner selection.
         THIS IS REPLAY NORMALIZATION, NOT A SECOND WINNER POLICY: it removes an
         exact duplicate of something already recorded — the restart case — and
         decides nothing between two records that differ;
      3. if `(seed, trial_number, skip_mode)` matches but ANY canonical field
         differs, raise `AccumulatorConsistencyError`. That key is the replay
         key; two different contents under one key is corruption, never a
         choice;
      4. pass the remainder to the frozen `_select_l2_winners`.

    Step 4 is the ONLY winner policy. `_select_l2_winners` / `_l2_sort_key` are
    imported from `utils.run_finalizer` and never forked (Ruling D): highest
    float32 score -> lowest `trial_number` -> constant before variable, and only
    as a tiebreak WITHIN one trial. The result is order-independent.
    """
    merged: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for origin, batch in (("recovered", recovered), ("new", new_records)):
        for record in batch:
            canonical = canonicalize_record(record)
            key = _replay_key(canonical)
            previous = merged.get(key)
            if previous is None:
                merged[key] = canonical
                continue
            if previous == canonical:
                continue                # step 2 — bit-identical replay collapses
            differing = sorted(
                name for name in CANONICAL_RECORD_FIELDS
                if previous[name] != canonical[name])
            raise AccumulatorConsistencyError(
                f"seed {key[0]}: two candidates share trial_number {key[1]} AND "
                f"skip_mode {key[2]!r} but disagree on {differing} (second "
                f"observed in the {origin} batch). The replay key is "
                f"(seed, trial_number, skip_mode); one key with two different "
                f"canonical contents is accumulator corruption and is never "
                f"resolved by preferring either side.")
    return [dict(w) for w in _select_l2_winners(list(merged.values()))]


def validate_new_raw_records(records: Sequence[Mapping[str, Any]],
                             *, seed_start: int, seed_end_exclusive: int,
                             prng_base: str,
                             skip_modes_executed: Sequence[str]) -> None:
    """§7.2 — the three walls `finalize_run` applies before L2, in ITS order.

    Applied to EVERY NEWLY OBSERVED RAW RECORD, before that batch is reconciled
    and before anything is cleared, because reconciliation compacts losers away:
    a malformed LOSING candidate must fail the run, not vanish during selection.
    `_validate_raw_candidates`'s own docstring already states that invariant.

    All three are IMPORTED from `utils.run_finalizer`, never duplicated.
    """
    _validate_raw_candidates(records)
    _validate_candidate_coverage(records, int(seed_start),
                                 int(seed_end_exclusive))
    _validate_candidate_identity(records, prng_base,
                                 tuple(skip_modes_executed))


# ===========================================================================
# Members: build, write, read
# ===========================================================================
def build_member_arrays(state_arrays: Mapping[str, np.ndarray],
                        identity: Mapping[str, Any],
                        role: str) -> Tuple[Dict[str, np.ndarray],
                                            Tuple[str, ...]]:
    """Return `(npz_arrays, payload_order)` for one member.

    Addendum §1, member payloads confirmed:
      * **A** — `seed` and `score`, plus its complete identity block. Nothing
        more. It is a MARKER that binds itself to `canonical_state_digest`; it
        does not recompute that digest and it is not an accumulator backup.
      * **B** — the complete reconstructible 24-field state.
    """
    if role == MEMBER_A_ROLE:
        payload_order = MEMBER_A_PAYLOAD_FIELDS
    elif role == MEMBER_B_ROLE:
        payload_order = STATE_PHYSICAL_ORDER
    else:                                                   # pragma: no cover
        raise CheckpointSchemaError(f"unknown member role {role!r}.")

    payload = {name: state_arrays[name] for name in payload_order}
    arrays: Dict[str, np.ndarray] = dict(payload)
    for key in IDENTITY_KEYS:
        arrays[key] = _identity_scalar_array(key, identity[key])
    return arrays, payload_order


def read_member(path: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray],
                                    Tuple[str, ...]]:
    """Return `(identity, payload, payload_order)`, or raise.

    A member with no identity block, or one stamped with a different schema
    version, is REFUSED rather than guessed at — the D6.1 four-field format is a
    different format and its marker is exactly what tells them apart.

    The member's own `member_content_digest` is VERIFIED here, over the same
    fixed field order the writer used.
    """
    with np.load(path, allow_pickle=False) as archive:
        files = set(archive.files)
        missing = [k for k in IDENTITY_KEYS if k not in files]
        if missing:
            raise CheckpointIdentityError(
                f"{path}: missing identity field(s) {missing} — not a "
                f"{CHECKPOINT_SCHEMA_VERSION} member.")
        identity: Dict[str, Any] = {}
        for key in IDENTITY_KEYS:
            value = archive[key]
            identity[key] = (int(value) if value.dtype.kind in "iu"
                             else str(value))
        if identity["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointIdentityError(
                f"{path}: checkpoint_schema_version "
                f"{identity['checkpoint_schema_version']!r} != "
                f"{CHECKPOINT_SCHEMA_VERSION!r}.")
        role = identity["member_role"]
        if role == MEMBER_A_ROLE:
            payload_order = MEMBER_A_PAYLOAD_FIELDS
        elif role == MEMBER_B_ROLE:
            payload_order = STATE_PHYSICAL_ORDER
        else:
            raise CheckpointIdentityError(
                f"{path}: unknown member_role {role!r}.")
        payload_missing = [n for n in payload_order if n not in files]
        if payload_missing:
            raise CheckpointSchemaError(
                f"{path}: missing payload array(s) {payload_missing} for role "
                f"{role!r}.")
        unexpected = files - set(payload_order) - set(IDENTITY_KEYS)
        if unexpected:
            raise CheckpointSchemaError(
                f"{path}: unexpected array(s) {sorted(unexpected)} for role "
                f"{role!r}; the member payload is exactly "
                f"{list(payload_order)}.")
        payload = {name: archive[name] for name in payload_order}

    expected = member_content_digest(identity, payload, payload_order)
    if expected != identity["member_content_digest"]:
        raise CheckpointIdentityError(
            f"{path}: member_content_digest mismatch "
            f"(stored {identity['member_content_digest'][:12]}…, recomputed "
            f"{expected[:12]}…) — this member's own content has been altered.")
    return identity, payload, payload_order


# ===========================================================================
# §5 + addendum §2 — the NINE-row mixed-pair recovery matrix
# ===========================================================================
#: The nine outcomes, named so a gate and a report can both quote them.
ROW_A_ABSENT = "row1_a_missing_or_unreadable"
ROW_A_DIGEST_FAIL = "row2_a_identity_matches_digest_fails"
ROW_A_CONFLICT = "row3_a_conflicts"
ROW_A_NEWER = "row4_a_newer_uncommitted_marker"
ROW_B_NEWER = "row5_b_newer_a_older_invariants_agree"
ROW_CONSISTENT = "row6_consistent_transaction"
ROW_B_INVALID = "row7_b_missing_or_invalid"
ROW_CONTEXT_DISAGREE = "row8_context_schema_encoding_disagreement"
ROW_ID_COLLISION = "row9_equal_sequence_different_checkpoint_id"


@dataclass
class RecoveryOutcome:
    """The result of a recovery attempt. Fail-closed cases RAISE instead."""
    row: str
    records: List[Dict[str, Any]]
    next_sequence: int
    checkpoint_id: str
    checkpoint_sequence: int
    canonical_state_digest: str
    run_id: str
    repair_pair: bool
    discarded_a_sequence: Optional[int] = None

    def provenance(self) -> Dict[str, Any]:
        """§4.5 — the durable resumed-run provenance, at minimum these four."""
        return {
            "resume_recovery_row": self.row,
            "recovered_checkpoint_run_id": self.run_id,
            "recovered_checkpoint_id": self.checkpoint_id,
            "recovered_checkpoint_sequence": self.checkpoint_sequence,
            "recovered_canonical_state_digest": self.canonical_state_digest,
            "recovered_canonical_record_count": len(self.records),
            "next_checkpoint_sequence": self.next_sequence,
            "discarded_newer_member_a_sequence": self.discarded_a_sequence,
            # REV5 §4.5, this exact wording: the finalizer's `raw_candidate_count`
            # is the records supplied to it BY THE RESUMED EXECUTION — neither the
            # original process's raw count nor a cumulative count across all
            # pre-compaction observations. No sidecar-field parity is claimed.
            "raw_candidate_count_semantics": (
                "the records supplied to the finalizer by the resumed execution"),
            # REV5 §0 / G-CURSOR-NOT-CLAIMED.
            "optimizer_execution_cursor_restored": False,
        }


def _require_context(identity: Mapping[str, Any], *, run_id: str,
                     run_context_digest: str, label: str) -> None:
    """Row 8 — any context / schema / encoding disagreement FAILS CLOSED.

    Checked BEFORE any categorical decoding (§4.3): a member whose
    `encoding_version` OR `canonical_map_hash` differs must fail before a single
    `prng_type` code is interpreted, because renaming a registry key preserves
    both `len(PRNG_TYPE_ENCODING)` and `ENCODING_VERSION` while renumbering every
    id after it alphabetically.
    """
    expected = {
        "run_id": run_id,
        "run_context_digest": run_context_digest,
        "encoding_version": ENCODING_VERSION,
        "canonical_map_hash": canonical_map_hash(),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
    }
    for key, want in expected.items():
        got = identity[key]
        if got != want:
            raise CheckpointRecoveryError(
                f"[{ROW_CONTEXT_DISAGREE}] {label}: {key} {got!r} != requested "
                f"{want!r}. Recovery fails closed and the in-memory state is "
                f"NOT cleared.")


def recover_checkpoint(checkpoint_dir: str, *, run_id: str,
                       run_context_digest: str) -> RecoveryOutcome:
    """The nine-row mixed-pair recovery matrix (§5 as amended by addendum §2).

    Replacement is A first, then B, so a legitimate crash leaves A at n+1 and B
    at n — A NEWER and unrecoverable. The blanket "higher valid sequence" rule is
    wrong here, which is why the A cases are disambiguated: agreement with a
    MISSING A is impossible, so "all invariant fields agree" cannot be the test
    for every case.

    Fail-closed means: raise, and do NOT clear in-memory state. This function
    cannot reach the in-memory list at all.
    """
    path_a = os.path.join(checkpoint_dir, MEMBER_A_NAME)
    path_b = os.path.join(checkpoint_dir, MEMBER_B_NAME)

    # ---- member B first: it is the SOLE recovery payload -------------------
    if not os.path.exists(path_b):
        raise CheckpointRecoveryError(
            f"[{ROW_B_INVALID}] member B {path_b} is missing. B is the sole "
            f"recovery payload; member A is a marker stub and is never an "
            f"accumulator backup, so this fails closed REGARDLESS of A.")
    try:
        ident_b, payload_b, order_b = read_member(path_b)
    except Exception as exc:                               # noqa: BLE001
        raise CheckpointRecoveryError(
            f"[{ROW_B_INVALID}] member B {path_b} is invalid ({exc}). B is the "
            f"sole recovery payload; loss or corruption of B is unrecoverable "
            f"and fails closed regardless of A.") from exc
    if ident_b["member_role"] != MEMBER_B_ROLE:
        raise CheckpointRecoveryError(
            f"[{ROW_B_INVALID}] {path_b} declares member_role "
            f"{ident_b['member_role']!r}, not {MEMBER_B_ROLE!r}.")

    _require_context(ident_b, run_id=run_id,
                     run_context_digest=run_context_digest, label="member B")

    # Rejection happens BEFORE categorical decoding — `_require_context` above
    # has already compared encoding_version and canonical_map_hash.
    state_digest = canonical_state_digest(payload_b)
    if state_digest != ident_b["canonical_state_digest"]:
        raise CheckpointRecoveryError(
            f"[{ROW_B_INVALID}] member B {path_b}: canonical_state_digest "
            f"mismatch (stored {ident_b['canonical_state_digest'][:12]}…, "
            f"recomputed {state_digest[:12]}…).")
    records = decode_state_arrays(payload_b)
    if len(records) != int(ident_b["logical_candidate_count"]):
        raise CheckpointRecoveryError(
            f"[{ROW_B_INVALID}] member B {path_b}: decoded {len(records)} "
            f"records but logical_candidate_count is "
            f"{int(ident_b['logical_candidate_count'])}.")

    seq_b = int(ident_b["checkpoint_sequence"])

    def _outcome(row: str, next_sequence: int, repair_pair: bool,
                 discarded_a: Optional[int] = None) -> RecoveryOutcome:
        return RecoveryOutcome(
            row=row, records=records, next_sequence=next_sequence,
            checkpoint_id=str(ident_b["checkpoint_id"]),
            checkpoint_sequence=seq_b,
            canonical_state_digest=state_digest,
            run_id=str(ident_b["run_id"]), repair_pair=repair_pair,
            discarded_a_sequence=discarded_a)

    # ---- now member A, disambiguated ---------------------------------------
    if not os.path.exists(path_a):
        # Row 1 — validate B against the CALLER-SUPPLIED run id and context
        # (already done above); recover B.
        return _outcome(ROW_A_ABSENT, seq_b + 1, repair_pair=True)
    try:
        ident_a, _payload_a, _order_a = read_member(path_a)
    except CheckpointIdentityError as exc:
        # Distinguish "identity block matches but the digest fails" (row 2) from
        # a member that is simply unreadable as a member at all (row 1).
        if "member_content_digest mismatch" in str(exc):
            try:
                probe = _probe_identity(path_a)
            except Exception:                              # noqa: BLE001
                return _outcome(ROW_A_ABSENT, seq_b + 1, repair_pair=True)
            _assert_a_not_conflicting(probe, ident_b, run_id,
                                      run_context_digest)
            # Row 2 — A's identity block matches, A fails its own digest.
            #
            # §4.6: A SEQUENCE EXTRACTED FROM AN OTHERWISE INVALID MEMBER IS NOT
            # A STRUCTURALLY VALID SEQUENCE. A that fails its own
            # `member_content_digest` is exactly such a member, so the sequence
            # it reports is NOT eligible to raise the next one — otherwise a
            # single flipped byte in A's stored sequence could push the run's
            # numbering anywhere it liked. Only B's sequence counts here.
            return _outcome(ROW_A_DIGEST_FAIL, seq_b + 1, repair_pair=True)
        return _outcome(ROW_A_ABSENT, seq_b + 1, repair_pair=True)
    except Exception:                                      # noqa: BLE001
        # Unreadable / structurally broken A — a truncated file, a non-NPZ blob,
        # a pickled payload refused by `allow_pickle=False`, an I/O error. All of
        # them are row 1: A is a MARKER STUB, so nothing about the recovery
        # depends on being able to read it, and its sequence is likewise not a
        # structurally valid sequence (§4.6).
        return _outcome(ROW_A_ABSENT, seq_b + 1, repair_pair=True)

    if ident_a["member_role"] != MEMBER_A_ROLE:
        raise CheckpointRecoveryError(
            f"[{ROW_A_CONFLICT}] {path_a} declares member_role "
            f"{ident_a['member_role']!r}, not {MEMBER_A_ROLE!r}.")

    _assert_a_not_conflicting(ident_a, ident_b, run_id, run_context_digest)
    seq_a = int(ident_a["checkpoint_sequence"])

    if seq_a == seq_b:
        if ident_a["checkpoint_id"] != ident_b["checkpoint_id"]:
            # Row 9 — equal sequence, different checkpoint_id.
            raise CheckpointRecoveryError(
                f"[{ROW_ID_COLLISION}] members share checkpoint_sequence "
                f"{seq_a} but declare different checkpoint_id values "
                f"({ident_a['checkpoint_id']} vs {ident_b['checkpoint_id']}). "
                f"Two transactions cannot occupy one sequence; failing closed.")
        for key in TRANSACTION_INVARIANT_KEYS:
            if ident_a[key] != ident_b[key]:
                raise CheckpointRecoveryError(
                    f"[{ROW_A_CONFLICT}] same-transaction pair disagrees on "
                    f"{key!r} ({ident_a[key]!r} vs {ident_b[key]!r}).")
        # Row 6 — consistent A/B transaction.
        return _outcome(ROW_CONSISTENT, seq_b + 1, repair_pair=False)

    if seq_a > seq_b:
        # Row 4 — A is a valid NEWER uncommitted marker (the A-first crash).
        # Discard it, recover B, and initialize the repaired sequence ABOVE A:
        # A's sequence IS structurally valid, so §4.6 counts it.
        return _outcome(ROW_A_NEWER, seq_a + 1, repair_pair=True,
                        discarded_a=seq_a)

    # Row 5 (addendum §2, restored) — B valid and NEWER, A valid but older, all
    # invariant identities agree. Distinct from an absent/corrupt A and from a
    # consistent same-transaction pair; it need not be reachable from the
    # ordinary A-first crash to remain a valid mixed-pair recovery case.
    return _outcome(ROW_B_NEWER, seq_b + 1, repair_pair=True)


def _probe_identity(path: str) -> Dict[str, Any]:
    """Read a member's identity block WITHOUT verifying its content digest.

    Used only to disambiguate recovery row 2 ("identity block matches, member
    fails its own `member_content_digest`") from row 1. It grants no trust: the
    caller still runs the full conflict check, and the payload is never used.
    """
    with np.load(path, allow_pickle=False) as archive:
        files = set(archive.files)
        missing = [k for k in IDENTITY_KEYS if k not in files]
        if missing:
            raise CheckpointIdentityError(
                f"{path}: missing identity field(s) {missing}.")
        identity: Dict[str, Any] = {}
        for key in IDENTITY_KEYS:
            value = archive[key]
            identity[key] = (int(value) if value.dtype.kind in "iu"
                             else str(value))
    if identity["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointIdentityError(
            f"{path}: checkpoint_schema_version "
            f"{identity['checkpoint_schema_version']!r} != "
            f"{CHECKPOINT_SCHEMA_VERSION!r}.")
    return identity


def _assert_a_not_conflicting(ident_a: Mapping[str, Any],
                              ident_b: Mapping[str, Any],
                              run_id: str, run_context_digest: str) -> None:
    """Row 3 — a structurally valid A that CONFLICTS fails closed.

    "Conflicts" means the RUN-invariant identity disagrees with B or with the
    requested context. `checkpoint_id`, `checkpoint_sequence`,
    `logical_candidate_count` and `canonical_state_digest` are NOT run-invariant:
    they legitimately differ between two transactions of the same run, which is
    exactly the mixed-pair state rows 4 and 5 describe.
    """
    _require_context(ident_a, run_id=run_id,
                     run_context_digest=run_context_digest, label="member A")
    for key in RUN_INVARIANT_KEYS:
        if ident_a[key] != ident_b[key]:
            raise CheckpointRecoveryError(
                f"[{ROW_A_CONFLICT}] member A disagrees with member B on the "
                f"run-invariant field {key!r} ({ident_a[key]!r} vs "
                f"{ident_b[key]!r}). Failing closed; in-memory state is NOT "
                f"cleared.")


# ===========================================================================
# §8 — the write transaction, in the binding order
# ===========================================================================
@dataclass
class RunContext:
    """Everything the flush needs, resolved ONCE at run start and frozen.

    It is deliberately explicit: the flush must be able to run the three
    finalizer walls and build `run_context_digest` without reaching for a
    coordinator, a config object or an environment variable mid-run.
    """
    run_id: str
    checkpoint_dir: str
    run_context_digest: str
    prng_base: str
    skip_modes_executed: Tuple[str, ...]
    seed_start: int
    seed_count: int
    components: Dict[str, Any]
    sequence: int = 0
    cumulative: List[Dict[str, Any]] = field(default_factory=list)
    resume_provenance: Optional[Dict[str, Any]] = None

    @property
    def seed_end_exclusive(self) -> int:
        return int(self.seed_start) + int(self.seed_count)

    def member_paths(self) -> Tuple[str, str]:
        return (os.path.join(self.checkpoint_dir, MEMBER_A_NAME),
                os.path.join(self.checkpoint_dir, MEMBER_B_NAME))


def build_identity(*, checkpoint_id: str, sequence: int, run_id: str,
                   logical_candidate_count: int, run_context_digest: str,
                   state_digest: str, role: str) -> Dict[str, Any]:
    """The identity block for one member, `member_content_digest` still blank.

    `canonical_map_hash` is IMPORTED from `utils.run_finalizer` and never
    reimplemented (§3.3), and it is carried instead of relying on the version
    string alone: `tests/test_prng_encoding.py` pins
    `len(PRNG_TYPE_ENCODING) == 44`, so renaming a registry key preserves both
    the count and `ENCODING_VERSION` while renumbering every id after it
    alphabetically. A member whose `encoding_version` OR `canonical_map_hash`
    differs fails BEFORE decoding.
    """
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": str(checkpoint_id),
        "checkpoint_sequence": int(sequence),
        "run_id": str(run_id),
        "logical_candidate_count": int(logical_candidate_count),
        "encoding_version": ENCODING_VERSION,
        "canonical_map_hash": canonical_map_hash(),
        "run_context_digest": str(run_context_digest),
        "canonical_state_digest": str(state_digest),
        "member_role": role,
        "member_content_digest": "",        # computed LAST, below
    }


def seal_member(state_arrays: Mapping[str, np.ndarray],
                identity: Mapping[str, Any], role: str
                ) -> Tuple[Dict[str, np.ndarray], Tuple[str, ...], str]:
    """Build one member's arrays and compute its digest LAST (addendum §1)."""
    identity = dict(identity)
    identity["member_role"] = role
    arrays, payload_order = build_member_arrays(state_arrays, identity, role)
    payload = {name: arrays[name] for name in payload_order}
    digest = member_content_digest(identity, payload, payload_order)
    identity["member_content_digest"] = digest
    arrays["member_content_digest"] = _identity_scalar_array(
        "member_content_digest", digest)
    return arrays, payload_order, digest


def validate_installed_pair(path_a: str, path_b: str, *,
                            expected_state_digest: str,
                            expected_sequence: int) -> None:
    """Read BOTH installed members back and prove the transaction landed.

    Runs before any clear (§8). It asserts what a normal pair agrees on and
    explicitly TOLERATES the one difference that is by design.
    """
    ident_a, _payload_a, _order_a = read_member(path_a)
    ident_b, payload_b, _order_b = read_member(path_b)

    if ident_a["member_role"] != MEMBER_A_ROLE:
        raise CheckpointIdentityError(
            f"{path_a}: member_role {ident_a['member_role']!r} != "
            f"{MEMBER_A_ROLE!r}.")
    if ident_b["member_role"] != MEMBER_B_ROLE:
        raise CheckpointIdentityError(
            f"{path_b}: member_role {ident_b['member_role']!r} != "
            f"{MEMBER_B_ROLE!r}.")

    for key in TRANSACTION_INVARIANT_KEYS:
        if ident_a[key] != ident_b[key]:
            raise CheckpointIdentityError(
                f"installed pair disagrees on {key!r}: {ident_a[key]!r} vs "
                f"{ident_b[key]!r}.")

    # EXPECTED TO DIFFER (§3.3). Agreement is NOT required; the difference is
    # asserted so a future "harmonization" that made them equal is caught.
    if ident_a["member_content_digest"] == ident_b["member_content_digest"]:
        raise CheckpointIdentityError(
            "the two members report the SAME member_content_digest; they "
            "persist different payloads by design (A is a marker stub carrying "
            f"{list(MEMBER_A_PAYLOAD_FIELDS)}, B carries the complete 24-field "
            "state), so equal digests mean one member is not what it claims.")

    if int(ident_b["checkpoint_sequence"]) != int(expected_sequence):
        raise CheckpointIdentityError(
            f"installed member B is at sequence "
            f"{int(ident_b['checkpoint_sequence'])}, expected "
            f"{int(expected_sequence)}.")

    # B RECOMPUTES AND VERIFIES the state digest. A only BINDS to it: A's marker
    # carries the value in its identity block (and inside its own member digest)
    # and does not claim to recompute it — A does not persist the state.
    recomputed = canonical_state_digest(payload_b)
    if recomputed != expected_state_digest:
        raise CheckpointIdentityError(
            f"installed member B recomputes canonical_state_digest "
            f"{recomputed[:12]}…, expected {expected_state_digest[:12]}….")
    if ident_a["canonical_state_digest"] != expected_state_digest:
        raise CheckpointIdentityError(
            f"installed member A's marker binds canonical_state_digest "
            f"{ident_a['canonical_state_digest'][:12]}…, expected "
            f"{expected_state_digest[:12]}….")


def write_transaction(context: RunContext,
                      records: Sequence[Mapping[str, Any]],
                      *, checkpoint_id: str,
                      write_npz, replace, fsync_dir,
                      tmp_name) -> Dict[str, Any]:
    """§8 — construct, write both temps, validate them, replace A then B,
    validate the installed pair. Returns the transaction descriptor.

    The IO primitives arrive as arguments rather than being called directly so
    the caller keeps its D6.1 durability behaviour (open-handle
    `savez_compressed` + fsync, same-directory temps, `os.replace`) and so a
    gate can inject a fault at any single step without patching this module.

    `savez_compressed` is retained by the caller: the D5 §6.7.A compressed-
    artifact ban is scoped to worker TRANSPORT artifacts, and the D6.1/D6.2
    checkpoint is deliberately separate. Do not harmonize the two.

    THE CLEAR IS NOT PERFORMED HERE. The caller clears only after this function
    returns, which is what makes "a mutant clearing between the two replaces"
    detectable.
    """
    state = canonical_state_arrays(records)
    state_digest = canonical_state_digest(state)
    sequence = int(context.sequence) + 1

    identity = build_identity(
        checkpoint_id=checkpoint_id, sequence=sequence, run_id=context.run_id,
        logical_candidate_count=len(records),
        run_context_digest=context.run_context_digest,
        state_digest=state_digest, role=MEMBER_A_ROLE)

    arrays_a, order_a, digest_a = seal_member(state, identity, MEMBER_A_ROLE)
    arrays_b, order_b, digest_b = seal_member(state, identity, MEMBER_B_ROLE)

    path_a, path_b = context.member_paths()
    tmp_a, tmp_b = tmp_name(path_a), tmp_name(path_b)

    # 1. write BOTH temporary artifacts, fsync/close as required
    write_npz(tmp_a, arrays_a)
    write_npz(tmp_b, arrays_b)

    # 2. validate BOTH temporary artifacts before any destination is touched
    validate_installed_pair(tmp_a, tmp_b,
                            expected_state_digest=state_digest,
                            expected_sequence=sequence)

    # 3. replace destination A, then destination B
    replace(tmp_a, path_a)
    replace(tmp_b, path_b)
    fsync_dir(context.checkpoint_dir)

    # 4. validate the INSTALLED pair — only after this may the caller clear
    validate_installed_pair(path_a, path_b,
                            expected_state_digest=state_digest,
                            expected_sequence=sequence)

    context.sequence = sequence
    context.cumulative = [dict(r) for r in records]
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_sequence": sequence,
        "run_id": context.run_id,
        "logical_candidate_count": len(records),
        "canonical_state_digest": state_digest,
        "member_content_digest_a": digest_a,
        "member_content_digest_b": digest_b,
        "member_a_payload_fields": list(order_a),
        "member_b_payload_fields": list(order_b),
    }
