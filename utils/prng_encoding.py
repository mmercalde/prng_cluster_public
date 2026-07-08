#!/usr/bin/env python3
"""
prng_encoding.py — Shared PRNG_TYPE <-> uint8 encoding (S172 Phase 0, v3.2)

Single source of truth for encoding a prng_type string to a uint8 NPZ id and
decoding it back. Replaces three previously-divergent hardcoded dicts that
silently collapsed unknown / hybrid prng_type values to 0 (java_lcg),
destroying provenance:
  - convert_survivors_to_binary.py        PRNG_TYPE_ENCODING (12 keys)
  - window_optimizer_integration_final.py _PRNG_ENC inline dict (12 keys)
  - utils/survivor_loader.py              PRNG_TYPE_DECODING (12 keys)

DESIGN (TB-approved Option C, 2026-05-xx):
  - The encoding is DERIVED from prng_registry.KERNEL_REGISTRY, not hardcoded.
  - Keys are sorted alphabetically; ids are assigned 0..N-1 in that order.
  - encode() hard-fails on an unknown prng_type (no silent 0 fallback).
  - decode() hard-fails on an unknown id (no silent 'java_lcg' fallback).
  - NPZ files are commit-local artifacts, NOT a durable ABI. If KERNEL_REGISTRY
    changes, ids may shift; regenerate NPZs or bump ENCODING_VERSION explicitly.

GUARDRAIL:
  tests/test_prng_encoding.py pins len(PRNG_TYPE_ENCODING) == 44. Adding or
  removing a registry key breaks that test ON PURPOSE, forcing an explicit
  decision about whether to regenerate NPZs / bump ENCODING_VERSION.

NOTE on skip_mode: skip_mode encoding is a fixed 2-value categorical
({'constant':0,'variable':1}) and is NOT registry-derived, so it stays defined
here as a stable constant for callers that want a single import site.
"""
from prng_registry import KERNEL_REGISTRY

# Bump this string if the derived mapping is ever intentionally changed in a
# way that should invalidate previously written NPZs.
ENCODING_VERSION = "3.2.0"

# --- skip_mode (fixed categorical, not registry-derived) ---
SKIP_MODE_ENCODING = {'constant': 0, 'variable': 1}
SKIP_MODE_DECODING = {v: k for k, v in SKIP_MODE_ENCODING.items()}

# --- prng_type (registry-derived, alphabetical) ---
# Sorted alphabetically so the mapping is reproducible from the key names alone.
_SORTED_PRNG_KEYS = sorted(KERNEL_REGISTRY.keys())
PRNG_TYPE_ENCODING = {name: i for i, name in enumerate(_SORTED_PRNG_KEYS)}
PRNG_TYPE_DECODING = {i: name for name, i in PRNG_TYPE_ENCODING.items()}

# uint8 ceiling check — registry must fit in a uint8 NPZ column.
if len(PRNG_TYPE_ENCODING) > 256:
    raise RuntimeError(
        f"KERNEL_REGISTRY has {len(PRNG_TYPE_ENCODING)} keys; exceeds uint8 (256) "
        f"capacity for the NPZ prng_type column."
    )


def encode_prng_type(prng_type: str) -> int:
    """
    Encode a prng_type string to its uint8 id.

    Raises ValueError on an unknown prng_type. There is intentionally NO
    silent fallback to 0 — silent collapse is the exact bug Phase 0 fixes.

    The caller is responsible for passing a canonical key. If a survivor record
    only carries 'prng_base', the caller should resolve to the full prng_type
    (e.g. prng_base or prng_base + '_hybrid') BEFORE calling encode.
    """
    try:
        return PRNG_TYPE_ENCODING[prng_type]
    except KeyError:
        raise ValueError(
            f"Unknown prng_type {prng_type!r}: not in KERNEL_REGISTRY "
            f"(known keys: {', '.join(_SORTED_PRNG_KEYS[:6])}, ... "
            f"{len(_SORTED_PRNG_KEYS)} total). "
            f"No silent fallback to 0 — fix the producer or add the kernel."
        )


def decode_prng_type(uint8_id: int) -> str:
    """
    Decode a uint8 id back to its prng_type string.

    Raises ValueError on an unknown id. There is intentionally NO silent
    fallback to 'java_lcg' — a id outside the known range means a corrupted
    NPZ or a mapping written by a different code revision, and the caller
    should know rather than silently mislabel provenance.
    """
    try:
        return PRNG_TYPE_DECODING[int(uint8_id)]
    except (KeyError, ValueError, TypeError):
        raise ValueError(
            f"Unknown prng_type id {uint8_id!r}: outside the valid range "
            f"0..{len(PRNG_TYPE_DECODING) - 1}. NPZ may be corrupt or written "
            f"by a different ENCODING_VERSION (current {ENCODING_VERSION})."
        )


def encode_skip_mode(skip_mode: str) -> int:
    """Encode skip_mode string to uint8. Hard-fails on unknown value."""
    try:
        return SKIP_MODE_ENCODING[skip_mode]
    except KeyError:
        raise ValueError(
            f"Unknown skip_mode {skip_mode!r}: expected one of "
            f"{list(SKIP_MODE_ENCODING)}."
        )


def decode_skip_mode(uint8_id: int) -> str:
    """Decode skip_mode uint8 to string. Hard-fails on unknown id."""
    try:
        return SKIP_MODE_DECODING[int(uint8_id)]
    except (KeyError, ValueError, TypeError):
        raise ValueError(
            f"Unknown skip_mode id {uint8_id!r}: expected one of "
            f"{list(SKIP_MODE_DECODING)}."
        )


def resolve_prng_type(record: dict) -> str:
    """
    Resolve the canonical prng_type string from a survivor record.

    Prefers 'prng_type'; falls back to 'prng_base' only if 'prng_type' absent.
    Does NOT default to 'java_lcg' — a record with neither key is a producer
    bug and should surface as a ValueError at encode time.
    """
    val = record.get('prng_type')
    if val is None:
        val = record.get('prng_base')
    if val is None:
        raise ValueError(
            "Survivor record has neither 'prng_type' nor 'prng_base'; "
            "cannot resolve a canonical prng_type for encoding."
        )
    return val
