"""
miner/range_miner_worker.py
===========================
S172 RANGE-MINER Phase 3 — per-GPU worker daemon (rev-4; rev-3 Team Beta APPROVED,
rev-4 adds the Phase-4-driven Stage-0 dataset_sha256 patch to ResidueResolver —
pending Beta re-review).

Spec: docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md (frozen at 1f6c0c5) §5.3, §5.4,
§6.8, §11, §12.4.  Fix brief: docs/S172_PHASE3_FIX_BRIEF.md.
Team Alpha implementation. Do not commit/push from the sandbox.

rev-3 addresses Team Beta's five release-blockers:
  B1  Per-assignment residue window resolution keyed by dataset CONTENT identity
      (not a stale process-lifetime `self.draws`).
  B2  Real atomic spool transport with a byte-exact schema, size-based (not
      count-based) inline/spool selection under the frame cap.
  B3  Exception-safe FULL GPU cleanup (try/finally) after every sub-stripe.
  B4  Real non-Java hybrid builders (Route B) for all 6 covered families, with
      variant-aware handshake capability advertisement + a STOP condition.
  B5  Blocking tests for every dangerous path (in the harness).

AUDITED ABIs — every hybrid kernel signature below was read from the LIVE
`prng_registry.py` (kernel_source strings), NOT extrapolated:

  Forward-hybrid common 13-element prefix (all families):
    seeds, residues, survivors, match_rates, skip_sequences, strategy_ids,
    survivor_count, int32(n_seeds), int32(k), strategy_max_misses,
    strategy_tolerances, int32(n_strategies), float32(threshold)
  Family-specific FORWARD tails (verified):
    java_lcg_hybrid     : uint64 a, uint64 c                       -> 15, NO offset
    lcg32_hybrid        : uint32 a, uint32 c, uint32 m, int32 off  -> 17
    minstd_hybrid       : uint32 a, uint32 m_val                   -> 15, NO offset
    pcg32_hybrid        : uint64 increment, int32 offset           -> 15
    xorshift32_hybrid   : int32 shift_a, shift_b, shift_c          -> 16, NO offset
    xorshift128_hybrid  : int32 dummy1, dummy2, dummy3             -> 16, NO offset
  REVERSE hybrids (ALL families identical — constants hardcoded in-kernel):
    <13-prefix>, int32(offset)                                     -> 14
  seed_type is uint64 ONLY for java_lcg; the other five are uint32 (read from
  each family's registry config — never assumed).

Constant kernels reproduce sieve_gpu_worker.py:208-306 (Beta-approved rev-1).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import signal
import socket
import struct
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from miner.range_miner_protocol import (
    DEFAULT_MINER_PORT,
    HEADER_SIZE,
    MAX_FRAME_BYTES,
    MinerBaseMessage,
    MinerHeartbeatMessage,
    MinerShutdownMessage,
    MinerStatusMessage,
    RegisterMessage,
    StripeAssignMessage,
    StripeCompleteMessage,
    StripeErrorMessage,
    SubStripeResultMessage,
    from_dict,
    message_to_bytes,
)

logger = logging.getLogger(__name__)

# ===========================================================================
# Family coverage (base-family level — §5.3, brief "Counts" box)
# ===========================================================================
COVERED_FAMILIES = frozenset(
    {"java_lcg", "lcg32", "minstd", "pcg32", "xorshift128", "xorshift32"}
)
# UNCOVERED: registered but no kernel-arg builder -> NotImplementedError at dispatch.
UNCOVERED_FAMILIES = frozenset(
    {"mt19937", "philox4x32", "sfc64", "xorshift64", "xoshiro256pp"}
)

_VARIANT_SUFFIXES = ("_hybrid_reverse", "_hybrid", "_reverse")

# B4 (Beta clarification 3): advertise EXACT concrete variants, validated against
# BOTH KERNEL_REGISTRY and a working builder branch — NOT a base x 4-suffix
# cross-product (which re-creates the over-claiming defect).
SUPPORTED_VARIANTS: Dict[str, frozenset] = {
    base: frozenset(
        {base, f"{base}_reverse", f"{base}_hybrid", f"{base}_hybrid_reverse"}
    )
    for base in COVERED_FAMILIES
}


def base_family(family_name: str) -> str:
    """Map a resolved variant name to its base family (longest suffix first)."""
    for suffix in _VARIANT_SUFFIXES:
        if family_name.endswith(suffix):
            return family_name[: -len(suffix)]
    return family_name


def is_hybrid_family(family_name: str) -> bool:
    return "_hybrid" in family_name


def is_reverse_family(family_name: str) -> bool:
    return family_name.endswith("_reverse")


# ===========================================================================
# Declarative kernel-arg elements
# ===========================================================================

@dataclass(frozen=True)
class BufferArg:
    name: str
    dtype: str  # 'uint32' | 'uint64' | 'float32' | 'int32' | 'uint8'


@dataclass(frozen=True)
class ScalarArg:
    value: Union[int, float]
    dtype: str  # 'int32' | 'uint32' | 'uint64' | 'float32'


KernelArg = Union[BufferArg, ScalarArg]


@dataclass
class BuildContext:
    family_name: str
    hybrid: bool
    reverse: bool
    seed_dtype: str           # 'uint32' | 'uint64' (from kernel config seed_type)
    n_seeds: int
    k: int
    skip_min: int
    skip_max: int
    threshold: float
    offset: int
    params: Dict[str, Any]
    n_strategies: int = 0
    hybrid_threshold: float = 0.0


# ---------------------------------------------------------------------------
# Shared arg fragments
# ---------------------------------------------------------------------------

def _constant_prefix(ctx: BuildContext) -> List[KernelArg]:
    """Constant-skip prefix — sieve_gpu_worker.py:210-216 (+ :200-203 dtypes)."""
    return [
        BufferArg("seeds", ctx.seed_dtype),
        BufferArg("residues", "uint32"),
        BufferArg("survivors", ctx.seed_dtype),
        BufferArg("match_rates", "float32"),
        BufferArg("best_skips", "uint8"),
        BufferArg("survivor_count", "uint32"),
        ScalarArg(ctx.n_seeds, "int32"),
        ScalarArg(ctx.k, "int32"),
        ScalarArg(ctx.skip_min, "int32"),
        ScalarArg(ctx.skip_max, "int32"),
        ScalarArg(ctx.threshold, "float32"),
    ]


def _hybrid_prefix(ctx: BuildContext) -> List[KernelArg]:
    """Forward/reverse hybrid common 13-element prefix (AUDITED, all families)."""
    return [
        BufferArg("seeds", ctx.seed_dtype),
        BufferArg("residues", "uint32"),
        BufferArg("survivors", ctx.seed_dtype),
        BufferArg("match_rates", "float32"),
        BufferArg("skip_sequences", "uint32"),
        BufferArg("strategy_ids", "uint32"),
        BufferArg("survivor_count", "uint32"),
        ScalarArg(ctx.n_seeds, "int32"),
        ScalarArg(ctx.k, "int32"),
        BufferArg("strategy_max_misses", "int32"),
        BufferArg("strategy_tolerances", "int32"),
        ScalarArg(ctx.n_strategies, "int32"),
        ScalarArg(ctx.hybrid_threshold, "float32"),
    ]


def _offset_tail(ctx: BuildContext) -> List[KernelArg]:
    return [ScalarArg(ctx.offset, "int32")]


def _reverse_hybrid_tail(ctx: BuildContext) -> List[KernelArg]:
    """ALL reverse hybrids: 13-prefix + int32(offset) = 14 args (constants in-kernel)."""
    return [ScalarArg(ctx.offset, "int32")]


# ---------------------------------------------------------------------------
# Per-family builders — 4 variants each (constant, _reverse, _hybrid,
# _hybrid_reverse). NOTE: forward and reverse CONSTANT kernels do NOT share arg
# layout — every fixed-skip reverse kernel hardcodes its generator params in the
# kernel body, so the reverse-constant ABI is `_constant_prefix + int32(offset)`
# = 12 args with NO family tail (registry-verified, same pattern as the reverse
# hybrids). Only the FORWARD constant branch carries the family-specific tail.
# ---------------------------------------------------------------------------

def build_java_lcg(ctx: BuildContext) -> List[KernelArg]:
    if ctx.hybrid:
        args = _hybrid_prefix(ctx)
        if ctx.reverse:
            return args + _reverse_hybrid_tail(ctx)            # 14
        # forward: uint64 a, c — ABI-critical, NO offset (:1007)
        return args + [
            ScalarArg(ctx.params.get("a", 25214903917), "uint64"),
            ScalarArg(ctx.params.get("c", 11), "uint64"),
        ]                                                       # 15
    if ctx.reverse:
        # java_lcg_reverse_sieve: params hardcoded in-kernel -> 12 args (verified)
        return _constant_prefix(ctx) + _offset_tail(ctx)
    # forward constant: uint64 a, c + offset
    return (
        _constant_prefix(ctx)
        + [
            ScalarArg(ctx.params.get("a", 25214903917), "uint64"),
            ScalarArg(ctx.params.get("c", 11), "uint64"),
        ]
        + _offset_tail(ctx)
    )


def build_lcg32(ctx: BuildContext) -> List[KernelArg]:
    if ctx.hybrid:
        args = _hybrid_prefix(ctx)
        if ctx.reverse:
            return args + _reverse_hybrid_tail(ctx)            # 14
        # forward: uint32 a, c, m AND trailing int32 offset (:2191) -> 17
        return args + [
            ScalarArg(ctx.params.get("a", 1664525), "uint32"),
            ScalarArg(ctx.params.get("c", 1013904223), "uint32"),
            ScalarArg(ctx.params.get("m", 0xFFFFFFFF), "uint32"),
            ScalarArg(ctx.offset, "int32"),
        ]
    if ctx.reverse:
        # lcg32_reverse_sieve: a,c,m hardcoded in-kernel -> 12 args (verified)
        return _constant_prefix(ctx) + _offset_tail(ctx)
    # forward constant: uint32 a, c, m + offset
    return (
        _constant_prefix(ctx)
        + [
            ScalarArg(ctx.params.get("a", 1664525), "uint32"),
            ScalarArg(ctx.params.get("c", 1013904223), "uint32"),
            ScalarArg(ctx.params.get("m", 0xFFFFFFFF), "uint32"),
        ]
        + _offset_tail(ctx)
    )


def build_minstd(ctx: BuildContext) -> List[KernelArg]:
    if ctx.hybrid:
        args = _hybrid_prefix(ctx)
        if ctx.reverse:
            return args + _reverse_hybrid_tail(ctx)            # 14
        # forward: uint32 a, m_val — NO offset (:1138) -> 15
        return args + [
            ScalarArg(ctx.params.get("a", 48271), "uint32"),
            ScalarArg(ctx.params.get("m", 2147483647), "uint32"),
        ]
    if ctx.reverse:
        # minstd_reverse_sieve: a,m hardcoded in-kernel -> 12 args (verified)
        return _constant_prefix(ctx) + _offset_tail(ctx)
    # forward constant: uint32 a, m + offset
    return (
        _constant_prefix(ctx)
        + [
            ScalarArg(ctx.params.get("a", 48271), "uint32"),
            ScalarArg(ctx.params.get("m", 2147483647), "uint32"),
        ]
        + _offset_tail(ctx)
    )


def build_pcg32(ctx: BuildContext) -> List[KernelArg]:
    if ctx.hybrid:
        args = _hybrid_prefix(ctx)
        if ctx.reverse:
            return args + _reverse_hybrid_tail(ctx)            # 14
        # forward: uint64 increment, int32 offset (:2095) -> 15
        return args + [
            ScalarArg(ctx.params.get("increment", 1442695040888963407), "uint64"),
            ScalarArg(ctx.offset, "int32"),
        ]
    if ctx.reverse:
        # pcg32_reverse_sieve: increment hardcoded in-kernel -> 12 args (verified)
        return _constant_prefix(ctx) + _offset_tail(ctx)
    # forward constant: uint64 increment + offset
    return (
        _constant_prefix(ctx)
        + [ScalarArg(ctx.params.get("increment", 1442695040888963407), "uint64")]
        + _offset_tail(ctx)
    )


def build_xorshift32(ctx: BuildContext) -> List[KernelArg]:
    if ctx.hybrid:
        args = _hybrid_prefix(ctx)
        if ctx.reverse:
            return args + _reverse_hybrid_tail(ctx)            # 14
        # forward: int32 shift_a, b, c — NO offset (:864) -> 16
        return args + [
            ScalarArg(ctx.params.get("shift_a", 13), "int32"),
            ScalarArg(ctx.params.get("shift_b", 17), "int32"),
            ScalarArg(ctx.params.get("shift_c", 5), "int32"),
        ]
    if ctx.reverse:
        # xorshift32_reverse_sieve: shifts hardcoded in-kernel -> 12 args (verified)
        return _constant_prefix(ctx) + _offset_tail(ctx)
    # forward constant: int32 shift_a, b, c + offset
    return (
        _constant_prefix(ctx)
        + [
            ScalarArg(ctx.params.get("shift_a", 13), "int32"),
            ScalarArg(ctx.params.get("shift_b", 17), "int32"),
            ScalarArg(ctx.params.get("shift_c", 5), "int32"),
        ]
        + _offset_tail(ctx)
    )


def build_xorshift128(ctx: BuildContext) -> List[KernelArg]:
    if ctx.hybrid:
        args = _hybrid_prefix(ctx)
        if ctx.reverse:
            return args + _reverse_hybrid_tail(ctx)            # 14
        # forward: int32 dummy1, dummy2, dummy3 — NO offset (:1276) -> 16
        return args + [
            ScalarArg(0, "int32"), ScalarArg(0, "int32"), ScalarArg(0, "int32"),
        ]
    if ctx.reverse:
        # xorshift128_reverse_sieve: dummies hardcoded in-kernel -> 12 args (verified)
        return _constant_prefix(ctx) + _offset_tail(ctx)
    # forward constant: int32 dummy1, dummy2, dummy3 + offset
    return (
        _constant_prefix(ctx)
        + [ScalarArg(0, "int32"), ScalarArg(0, "int32"), ScalarArg(0, "int32")]
        + _offset_tail(ctx)
    )


kernel_args_builders: Dict[str, Callable[[BuildContext], List[KernelArg]]] = {
    "java_lcg": build_java_lcg,
    "lcg32": build_lcg32,
    "minstd": build_minstd,
    "pcg32": build_pcg32,
    "xorshift128": build_xorshift128,
    "xorshift32": build_xorshift32,
}


def resolve_builder(family_name: str) -> Callable[[BuildContext], List[KernelArg]]:
    """Look up the builder by base family. Uncovered families raise
    NotImplementedError (naming the family) BEFORE any GPU work (§11.I)."""
    base = base_family(family_name)
    builder = kernel_args_builders.get(base)
    if builder is None:
        raise NotImplementedError(
            f"S172: no kernel-args builder for base family {base!r} "
            f"(requested family_name={family_name!r}). Registered but UNCOVERED — "
            f"no kernel launched. Covered: {sorted(COVERED_FAMILIES)}."
        )
    return builder


# ---------------------------------------------------------------------------
# Materialization — apply EXACTLY the tagged dtype at launch time
# ---------------------------------------------------------------------------

_SCALAR_WRAPPERS: Dict[str, Callable[[Any, Union[int, float]], Any]] = {
    "int32": lambda xp, v: xp.int32(v),
    "uint32": lambda xp, v: xp.uint32(v),
    "uint64": lambda xp, v: xp.uint64(v),
    "float32": lambda xp, v: xp.float32(v),
}


def materialize_kernel_args(
    arg_list: List[KernelArg], buffers: Dict[str, Any], xp: Any
) -> Tuple[Any, ...]:
    out: List[Any] = []
    for a in arg_list:
        if isinstance(a, BufferArg):
            try:
                out.append(buffers[a.name])
            except KeyError:
                raise KeyError(f"missing buffer {a.name!r} for kernel arg list")
        elif isinstance(a, ScalarArg):
            wrap = _SCALAR_WRAPPERS.get(a.dtype)
            if wrap is None:
                raise ValueError(f"unknown scalar dtype {a.dtype!r}")
            out.append(wrap(xp, a.value))
        else:  # pragma: no cover — defensive
            raise TypeError(f"unexpected kernel arg element: {a!r}")
    return tuple(out)


# ===========================================================================
# B4 — variant-aware capability advertisement + STOP condition
# ===========================================================================

class VariantStopCondition(Exception):
    """Beta-mandated STOP: a declared variant is unusable/incomplete. Carries a
    proposed Route-A erratum rather than silently narrowing capability."""


def _validate_variant(variant: str, registry: Dict[str, Any]) -> None:
    """A concrete variant is valid iff it exists in KERNEL_REGISTRY with a
    kernel_name + kernel_source AND its base family has a working builder branch.
    Any failure raises VariantStopCondition (do NOT silently drop the variant)."""
    cfg = registry.get(variant)
    if cfg is None:
        raise VariantStopCondition(
            f"STOP: variant {variant!r} absent from KERNEL_REGISTRY. Propose a "
            f"Route-A erratum for this variant (Beta owns the ruling)."
        )
    if not cfg.get("kernel_name") or not cfg.get("kernel_source"):
        raise VariantStopCondition(
            f"STOP: variant {variant!r} malformed (missing kernel_name/"
            f"kernel_source). Propose a Route-A erratum."
        )
    base = base_family(variant)
    if base not in kernel_args_builders:
        raise VariantStopCondition(
            f"STOP: variant {variant!r} has no builder branch for base {base!r}."
        )


def supported_variants() -> List[str]:
    """Sorted union of successfully VALIDATED concrete variants. Raises
    VariantStopCondition if any declared variant is unusable — it must not
    silently disappear from the handshake (Beta clarification 3)."""
    from prng_registry import KERNEL_REGISTRY  # CPU-safe import (no cupy at top)
    out: set = set()
    for base, variants in SUPPORTED_VARIANTS.items():
        for v in variants:
            _validate_variant(v, KERNEL_REGISTRY)
            out.add(v)
    return sorted(out)


# ===========================================================================
# Per-family VRAM caps (§12.4, TB Q2)
# ===========================================================================

@dataclass(frozen=True)
class VramCaps:
    """Constant + (tighter) hybrid seed caps, per backend. Spec §7 default_params.
    NOTE: the two *_hybrid caps are NOT yet in the WATCHER manifest (v1.8.0);
    Phase 4 (coordinator) / Phase 7 (WATCHER soak) MUST wire them. Until then the
    daemon sources them from argparse defaults."""
    amd: int = 2_000_000
    nvidia: int = 5_000_000
    amd_hybrid: int = 1_000_000
    nvidia_hybrid: int = 2_500_000


def select_seed_cap(backend: str, family_name: str, caps: VramCaps) -> int:
    """Hybrid phase uses the tighter cap (extra skip_sequences_gpu alloc)."""
    hybrid = is_hybrid_family(family_name)
    if backend == "rocm":
        return caps.amd_hybrid if hybrid else caps.amd
    if backend == "cuda":
        return caps.nvidia_hybrid if hybrid else caps.nvidia
    raise ValueError(f"unknown backend {backend!r} (expected 'rocm' | 'cuda')")


# ===========================================================================
# Sub-stripe partitioning
# ===========================================================================

@dataclass(frozen=True)
class SubStripe:
    sub_index: int
    seed_start: int
    seed_count: int


def partition_stripe(seed_start: int, seed_count: int, cap: int) -> List[SubStripe]:
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    if seed_count < 0:
        raise ValueError(f"seed_count must be non-negative, got {seed_count}")
    subs: List[SubStripe] = []
    n = (seed_count + cap - 1) // cap
    for i in range(n):
        start = seed_start + i * cap
        count = min(cap, seed_start + seed_count - start)
        subs.append(SubStripe(sub_index=i, seed_start=start, seed_count=count))
    return subs


# ===========================================================================
# B1 — per-assignment residue window resolution (content-identity keyed)
# ===========================================================================

class ResidueError(Exception):
    """Base for non-retryable residue-window failures -> stripe_error(retryable=False)."""


class ResidueResolutionError(ResidueError):
    """Payload lacks the fields to resolve the window (and no residue reference)."""


class ResidueVerificationError(ResidueError):
    """Loaded residues do not match the coordinator-provided residue_sha256."""


class DatasetProvisioningError(ResidueError):
    """The dataset this stripe needs is absent or unreadable ON THIS NODE.

    [S172 Phase 6-P0.5 — Beta, P0 ruling §3 correction]

    A missing dataset is **not** semantically a residue error, and it must not be
    flattened into an undifferentiated `ResidueError`: that would preserve the
    control flow at the cost of the operational category, leaving "this node was
    never provisioned" indistinguishable from "these residues do not verify".
    The two have different operators, different causes and different fixes.

    It nonetheless subclasses `ResidueError` because the coordinator's failure
    matrix routes the residue hierarchy to `stripe_error(retryable=False)`, and a
    provisioning fault is genuinely non-retryable — retrying a stripe against a
    node that has no dataset produces the same failure more slowly. So: keep the
    control flow, keep the category.

    Two obligations wherever this is raised, both from the ruling:
      * **chain the original exception** (`raise … from exc`) — the underlying
        `FileNotFoundError`/`OSError` is the evidence and must not be discarded;
      * name the **absolute path and the node** — the failure is per-node by
        nature, and a message that omits which node it happened on is not
        actionable on a three-rig fleet.

    Before P0.5 this surfaced as a bare `FileNotFoundError` from `_sha256_file`,
    raised mid-run, unclassified, two layers deep inside a worker. The
    fail-before-dispatch fleet check in `miner/dataset_authority.py` is what
    should normally prevent this from ever being reached; this class is the
    backstop for the case that slips past it.
    """


def sha256_residues(residues: List[int]) -> str:
    """Canonical residue-sequence fingerprint (contract for coordinator-supplied
    `residue_sha256`): sha256 over compact JSON of the int list."""
    body = json.dumps([int(x) for x in residues], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _node_identity() -> str:
    """This worker's node name, for per-node failure messages.

    `socket.gethostname()` deliberately: CT100 is created with the rig's
    canonical hostname so that the hostname IS the coordinator identity
    (`docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md`).
    """
    try:
        return socket.gethostname()
    except Exception:                                     # pragma: no cover
        return "<unknown-host>"


def _dataset_absent(path: str, exc: BaseException, what: str) -> "DatasetProvisioningError":
    """Build the classified, chained, node-naming provisioning error (P0.5 §3)."""
    return DatasetProvisioningError(
        f"dataset unavailable on node {_node_identity()!r}: {what} "
        f"{os.path.abspath(path)!r} ({type(exc).__name__}: {exc}). "
        f"This node was not provisioned with the run's frozen dataset, or the "
        f"file was removed after provisioning. See "
        f"docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md §2-§4."
    )


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        f = open(path, "rb")
    except FileNotFoundError as exc:
        # [S172 Phase 6-P0.5] was a bare FileNotFoundError escaping mid-run,
        # unclassified. Beta's §3 correction: classify, chain, name path + node.
        raise _dataset_absent(path, exc, "cannot hash missing dataset") from exc
    except OSError as exc:
        raise _dataset_absent(path, exc, "cannot read dataset") from exc
    with f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_residue_window(
    path: str, window_size: int, sessions: Optional[list], offset: int
) -> List[int]:
    """THE canonical residue-window derivation for the RANGE-MINER path.

    Loads a residue window from disk with NO pathname cache (mirrors the parse in
    sieve_gpu_worker.load_draws_cached but always reads fresh, so a changed file
    is never served from a stale cache — Beta clarification 1).

    [S172 D6 correction — shared authority, Beta §4] `sessions` is a first-class
    INPUT, applied here and ONLY here. Both sides of the assignment consume this
    one function:

      * the WORKER, through ResidueResolver's default loader, to rebuild the
        window it will sieve;
      * the COORDINATOR side, through
        window_optimizer_integration_final._miner_residues_for_config, to build
        the residues whose sha256 is stamped into the assignment payload.

    Before this fix the parent derived residues WITHOUT the session filter while
    the worker derived them WITH it, so any single-session trial (`sessions=
    ['midday']` or `['evening']`) produced two different ordered windows and every
    stripe died on the residue_sha256 check. Do NOT reintroduce a second
    session-filter implementation on either side — the asymmetry is only
    impossible while exactly one function owns the filter.
    """
    try:
        f = open(path, "r")
    except FileNotFoundError as exc:
        # [S172 Phase 6-P0.5] Same classification as _sha256_file. In the worker
        # the digest check reaches the file first, so this is normally
        # unreachable; the COORDINATOR side calls this function directly
        # (window_optimizer_integration_final._miner_residues_for_config), and
        # there it is the first touch.
        raise _dataset_absent(path, exc, "cannot load residue window from") from exc
    except OSError as exc:
        raise _dataset_absent(path, exc, "cannot read dataset") from exc
    with f:
        data = json.load(f)
    if sessions:
        data = [e for e in data if e.get("session") in sessions]
    n = len(data)
    if n < window_size:
        raise ResidueResolutionError(
            f"dataset {path!r} has only {n} entries, need window_size {window_size}"
        )
    start = max(0, min(int(offset), n - window_size))
    window = data[start:start + window_size]
    return [int(entry.get("full_state", entry["draw"])) for entry in window]


# Pre-D6 private name kept as an alias — same object, so a monkeypatch or a
# reference through either name reaches the one canonical implementation.
_load_window_fresh = load_residue_window


class ResidueResolver:
    """Resolves the residue window per ASSIGNMENT, keyed by dataset CONTENT
    identity — never process-lifetime `self.draws` (B1). Cache key:
      (dataset_reference, dataset_sha256, window_size, canonical_sessions, offset)
    A coordinator-provided `residue_sha256` takes precedence for keying AND is
    verified against the loaded sequence (mismatch -> ResidueVerificationError).

    ASSIGNMENT CONTRACT (v3 clarification — resolution (a), Beta-approved choice):
    Phase 4's `stripe_assign.payload` ALWAYS supplies the window-defining fields
    (`dataset`/`dataset_path`/`dataset_reference` + `window_size`, plus optional
    `sessions`/`offset`/`residue_sha256`). There is intentionally NO bare
    residue-reference path (no `residue_path` / inline residues) — the current
    coordinator direction sends the dataset+window, so an alternate path would be
    dead code. `resolve()` fails CLEARLY with `ResidueResolutionError` if those
    fields are absent, rather than falling back to any stale window (see guard)."""

    def __init__(
        self,
        loader: Optional[Callable[[str, int, Optional[list], int], List[int]]] = None,
        file_hasher: Optional[Callable[[str], str]] = None,
    ) -> None:
        # Shared authority (D6): the worker's default loader IS the canonical
        # session-aware derivation the coordinator side also calls.
        self._loader = loader or load_residue_window
        self._file_hasher = file_hasher or _sha256_file
        self._cache: Dict[tuple, List[int]] = {}

    def resolve(self, payload: Optional[Dict[str, Any]]) -> List[int]:
        payload = payload or {}
        dataset = (
            payload.get("dataset_reference")
            or payload.get("dataset_path")
            or payload.get("dataset")
        )
        window_size = payload.get("window_size")
        sessions = payload.get("sessions")
        offset = payload.get("offset", 0)
        residue_sha = payload.get("residue_sha256")

        if not dataset or window_size is None:
            raise ResidueResolutionError(
                "assignment payload lacks dataset/window_size to resolve the "
                "residue window and no usable residue reference was provided; "
                "refusing to run against stale data (B1)."
            )

        # sessions MUST be canonicalized before entering the key (sorted tuple).
        canonical_sessions = tuple(sorted(sessions)) if sessions else ()
        # CONTENT fingerprint — never key on pathname alone. Reused below for both
        # the Blocker-6 integrity check AND the cache key (hash the file once).
        dataset_sha = self._file_hasher(dataset)

        # Blocker-6 (TB binding ruling — Option C): dataset_sha256 is MANDATORY on
        # every assignment and MUST match the locally computed content hash. Both
        # checks gate the method BEFORE any cache return and BEFORE residue loading,
        # so a cached window can never bypass a later hash mismatch. Both failures
        # are non-retryable (ResidueError -> stripe_error(retryable=False)). Plain
        # `!=` — this is an integrity identifier, not a secret.
        expected_dataset_sha = payload.get("dataset_sha256")
        if not expected_dataset_sha:
            raise ResidueResolutionError(
                "assignment payload missing mandatory dataset_sha256"
            )
        if dataset_sha != expected_dataset_sha:
            raise ResidueVerificationError(
                f"dataset_sha256 mismatch: payload={expected_dataset_sha}, "
                f"computed={dataset_sha}"
            )

        if residue_sha:
            key: tuple = ("residue_sha256", residue_sha, window_size,
                          canonical_sessions, offset)
        else:
            key = (dataset, dataset_sha, window_size, canonical_sessions, offset)

        if key in self._cache:
            return self._cache[key]

        residues = self._loader(
            dataset, window_size, list(canonical_sessions) or None, offset
        )
        if residue_sha is not None:
            got = sha256_residues(residues)
            if got != residue_sha:
                raise ResidueVerificationError(
                    f"residue_sha256 mismatch: payload={residue_sha}, computed={got}"
                )
        self._cache[key] = residues
        return residues


# ===========================================================================
# B3 — exception-safe full GPU cleanup (replicates sieve_gpu_worker:78-94)
# ===========================================================================

def _best_effort_gpu_cleanup() -> None:
    """gc + torch sync/empty_cache + CuPy default & pinned pool free_all_blocks.
    Each step guarded so a missing torch/cupy never crashes cleanup. Called after
    EVERY sub-stripe, success or exception."""
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ===========================================================================
# GPU execution
# ===========================================================================

class ThresholdContractError(Exception):
    """[S172 D6] The assignment payload's threshold fields contradict each other.

    NON-RETRYABLE: a contradictory pair is a coordinator-side contract violation,
    not a transient fault, and retrying it would filter at a value nobody asked
    for. See build_stripe_assign_payload's THRESHOLD CONTRACT."""


@dataclass
class SubStripeOutcome:
    survivors: List[tuple]
    count: int
    # [S172 D6] The threshold the KERNEL was actually launched with and the host
    # post-filter actually applied for this sub-stripe — the `effective` leg of
    # the requested/payload/effective provenance triple. Never re-derived
    # downstream; set from the same local the kernel arg was built from.
    effective_threshold: Optional[float] = None


def _load_strategies(strategies_data: Optional[list]) -> list:
    """Replicate sieve_gpu_worker.py:239-249 strategy resolution."""
    if strategies_data:
        return strategies_data
    try:
        from hybrid_strategy import get_all_strategies
        return [
            s.to_dict() if hasattr(s, "to_dict") else s
            for s in get_all_strategies()
        ]
    except ImportError:
        return [{"max_consecutive_misses": 3, "skip_tolerance": 5}]


class SieveExecutor:
    """Real GPU executor. Resolves the builder (coverage guard) and the residue
    window (B1) BEFORE any allocation; wraps all GPU work in try/finally so the
    FULL cleanup runs on success AND exception (B3)."""

    def __init__(self, resolver: ResidueResolver, device_index: int = 0):
        self.resolver = resolver
        self.device_index = device_index

    # The single mockable "kernel entry" (brief gate: no launch on uncovered).
    def _gpu_launch(self, kernel, blocks: int, threads: int, kernel_args: tuple) -> None:
        kernel((blocks,), (threads,), kernel_args)

    def execute(
        self, assign: StripeAssignMessage, seed_start: int, seed_count: int
    ) -> SubStripeOutcome:
        # (1) Coverage guard FIRST — uncovered families raise here, no GPU/residue.
        builder = resolve_builder(assign.family_name)
        # (2) Per-assignment residue window (B1) — raises ResidueError (non-retryable).
        residues = self.resolver.resolve(assign.payload)

        # (3) Lazy GPU imports (guarded so this module imports on a CPU-only box).
        import cupy as cp
        from sieve_gpu_worker import _get_kernel, coerce_threshold

        family = assign.family_name
        payload = assign.payload or {}
        kernel, config = _get_kernel(family)
        seed_type = config.get("seed_type", "uint32")
        seed_dtype = "uint64" if seed_type == "uint64" else "uint32"
        xp_seed_dtype = cp.uint64 if seed_dtype == "uint64" else cp.uint32

        default_params = dict(config.get("default_params", {}))
        custom_params = payload.get("params") or {}
        if custom_params:
            default_params.update(custom_params)

        skip_min, skip_max = tuple(payload.get("skip_range", [0, 16]))
        # [S172 D6] The worker does NOT choose a threshold and does NOT know
        # about forward/reverse — the parent resolved the directional value from
        # the §6.8 phase table and put it in the payload
        # (build_stripe_assign_payload). The 0.25 default below survives ONLY for
        # legacy pre-D6 payloads that carry no threshold field at all; a
        # newly-generated D6 payload that relies on it is a defect, caught by the
        # D6 threshold gate, not by silent fallback here.
        threshold = coerce_threshold(payload.get("min_match_threshold", None), 0.25)
        # Contract re-validation (Beta §2): if the payload carries BOTH keys they
        # MUST be identical — the miner launches one kernel with one threshold
        # per stripe, so a contradictory pair means one of the two would be
        # silently discarded depending on skip mode. Fail closed instead.
        _mmt_raw = payload.get("min_match_threshold", None)
        _p2t_raw = payload.get("phase2_threshold", None)
        if _mmt_raw is not None and _p2t_raw is not None:
            if coerce_threshold(_p2t_raw, threshold) != threshold:
                raise ThresholdContractError(
                    f"contradictory thresholds in stripe payload: "
                    f"min_match_threshold={_mmt_raw!r} vs phase2_threshold="
                    f"{_p2t_raw!r}. A miner stripe runs ONE kernel with ONE "
                    f"threshold; refusing to filter at a value the parent did "
                    f"not unambiguously request.")
        offset = payload.get("offset", 0)
        hybrid = is_hybrid_family(family)
        reverse = is_reverse_family(family)

        k = len(residues)
        n_seeds = seed_count
        survivors_out: List[tuple] = []

        seeds_gpu = survivors_gpu = match_rates_gpu = best_skips_gpu = None
        survivor_count_gpu = residues_gpu = strategy_ids_gpu = skip_sequences_gpu = None
        strategy_max_misses = strategy_tolerances = None
        try:
            with cp.cuda.Device(self.device_index):
                residues_gpu = cp.array(
                    residues[::-1] if reverse else residues, dtype=cp.uint32
                )
                seeds_gpu = cp.arange(
                    seed_start, seed_start + n_seeds, dtype=xp_seed_dtype
                )
                survivors_gpu = cp.zeros(n_seeds, dtype=xp_seed_dtype)
                match_rates_gpu = cp.zeros(n_seeds, dtype=cp.float32)
                survivor_count_gpu = cp.zeros(1, dtype=cp.uint32)

                buffers: Dict[str, Any] = {
                    "seeds": seeds_gpu,
                    "residues": residues_gpu,
                    "survivors": survivors_gpu,
                    "match_rates": match_rates_gpu,
                    "survivor_count": survivor_count_gpu,
                }

                n_strategies = 0
                hybrid_threshold = threshold
                if hybrid:
                    strategies_data = _load_strategies(payload.get("strategies"))
                    n_strategies = len(strategies_data)
                    strategy_max_misses = cp.array(
                        [s["max_consecutive_misses"] for s in strategies_data],
                        dtype=cp.int32,
                    )
                    strategy_tolerances = cp.array(
                        [s["skip_tolerance"] for s in strategies_data], dtype=cp.int32
                    )
                    strategy_ids_gpu = cp.zeros(n_seeds, dtype=cp.uint32)
                    skip_sequences_gpu = cp.zeros(n_seeds * k, dtype=cp.uint32)
                    buffers.update(
                        {
                            "strategy_max_misses": strategy_max_misses,
                            "strategy_tolerances": strategy_tolerances,
                            "strategy_ids": strategy_ids_gpu,
                            "skip_sequences": skip_sequences_gpu,
                        }
                    )
                    # See build_stripe_assign_payload (range_miner_coordinator.py)
                    # for the constant-vs-hybrid key contract: `phase2_threshold`
                    # is a legacy PWC name for THIS kernel's single threshold, NOT
                    # a second stage — there is no stage-2 filter here. Which key
                    # feeds the kernel is decided by skip mode, not by stage, and
                    # D6 pins the two keys equal at the parent.
                    phase2_raw = payload.get("phase2_threshold", None)
                    hybrid_threshold = (
                        coerce_threshold(phase2_raw, threshold)
                        if phase2_raw is not None
                        else threshold
                    )
                else:
                    best_skips_gpu = cp.zeros(n_seeds, dtype=cp.uint8)
                    buffers["best_skips"] = best_skips_gpu

                ctx = BuildContext(
                    family_name=family, hybrid=hybrid, reverse=reverse,
                    seed_dtype=seed_dtype, n_seeds=n_seeds, k=k,
                    skip_min=skip_min, skip_max=skip_max, threshold=threshold,
                    offset=offset, params=default_params,
                    n_strategies=n_strategies, hybrid_threshold=hybrid_threshold,
                )
                arg_list = builder(ctx)
                kernel_args = materialize_kernel_args(arg_list, buffers, cp)

                threads = 256
                blocks = (n_seeds + threads - 1) // threads
                self._gpu_launch(kernel, blocks, threads, kernel_args)

                count = min(int(survivor_count_gpu[0].get()), n_seeds)
                if count > 0:
                    if hybrid:
                        s_arr = survivors_gpu[:count].get().tolist()
                        r_arr = match_rates_gpu[:count].get().tolist()
                        sid_arr = strategy_ids_gpu[:count].get().tolist()
                        ss_raw = (
                            skip_sequences_gpu[: count * k].get()
                            .reshape(count, k).tolist()
                        )
                        for seed, rate, sid, ss in zip(s_arr, r_arr, sid_arr, ss_raw):
                            if rate >= hybrid_threshold:
                                survivors_out.append(
                                    (int(seed), float(rate), int(sid), list(ss))
                                )
                    else:
                        s_arr = survivors_gpu[:count].get().tolist()
                        r_arr = match_rates_gpu[:count].get().tolist()
                        k_arr = best_skips_gpu[:count].get().tolist()
                        for seed, rate, skip in zip(s_arr, r_arr, k_arr):
                            if rate >= threshold:
                                survivors_out.append(
                                    (int(seed), float(rate), None, [int(skip)])
                                )
            # [S172 D6] The EFFECTIVE threshold is read off the very local the
            # kernel arg was built from (ctx.hybrid_threshold for hybrid kernels,
            # ctx.threshold for constant ones) and the host post-filter compared
            # against — not recomputed from the payload, which would only prove
            # the payload agrees with itself.
            return SubStripeOutcome(
                survivors=survivors_out, count=len(survivors_out),
                effective_threshold=float(
                    hybrid_threshold if hybrid else threshold))
        finally:
            # [S154] explicit per-array del (guarded) + FULL best-effort cleanup —
            # runs after EVERY sub-stripe, success OR exception (B3). Replicates the
            # proven worker's teardown (sieve_gpu_worker.py:331-348): drop references
            # then free the CuPy pools. `del <name>` on a local rebinds-away that
            # name; NameError is swallowed exactly as the live worker does.
            try: del seeds_gpu
            except NameError: pass
            try: del survivors_gpu
            except NameError: pass
            try: del match_rates_gpu
            except NameError: pass
            try: del best_skips_gpu
            except NameError: pass
            try: del survivor_count_gpu
            except NameError: pass
            try: del residues_gpu
            except NameError: pass
            try: del strategy_ids_gpu
            except NameError: pass
            try: del skip_sequences_gpu
            except NameError: pass
            try: del strategy_max_misses
            except NameError: pass
            try: del strategy_tolerances
            except NameError: pass
            try: del buffers, kernel_args, arg_list
            except NameError: pass
            _best_effort_gpu_cleanup()


# ===========================================================================
# B2 — spool transport (byte-exact schema; size-based inline/spool)
# ===========================================================================

SUBSTRIPE_SCHEMA_VERSION = "s172_substripe_v1"

# Inline only if the COMPLETE framed SubStripeResultMessage is <= 48 MiB. This
# leaves a full 16 MiB of headroom under the protocol's 64 MiB hard frame cap
# (MAX_FRAME_BYTES) for the JSON envelope + length prefix, so a message that
# measures under the limit can never overflow the wire (B2 defect B).
INLINE_BYTE_LIMIT = 48 * 1024 * 1024


def build_substripe_payload_bytes(
    stripe_id: str, sub_index: int, seed_start: int, seed_count: int,
    survivors: list,
) -> Tuple[dict, bytes]:
    """Canonical spool/inline payload (Beta clarification 2 — Phase 5 reads these
    exact bytes). Hash/size are ALWAYS taken over `payload_bytes`, never over a
    reconstructed object."""
    payload_obj = {
        "schema_version": SUBSTRIPE_SCHEMA_VERSION,
        "stripe_id": stripe_id,
        "sub_index": sub_index,
        "seed_start": seed_start,
        "seed_count": seed_count,
        "survivors": survivors,
    }
    payload_bytes = json.dumps(
        payload_obj, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return payload_obj, payload_bytes


def resolve_miner_output_dir(cli_value: Optional[str] = None) -> str:
    """--miner-output-dir / auto-detect /dev/shm/prng/miner -> ~/miner_output
    (S172_INFRASTRUCTURE_INTERFACE §5). Creates the dir."""
    if cli_value:
        d = cli_value
    elif os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        d = "/dev/shm/prng/miner"
    else:
        d = os.path.expanduser("~/miner_output")
    os.makedirs(d, exist_ok=True)
    return d


def spool_payload_atomic(
    out_dir: str, stripe_id: str, sub_index: int, sha256: str, payload_bytes: bytes
) -> str:
    """Atomic write: temp file in the SAME dir -> fsync -> os.replace to final.
    The WORKER removes its abandoned temp file on failure; the final spool file
    remains until the coordinator verifies + collects it (Beta clarification 2)."""
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"{stripe_id}_sub{sub_index}_{sha256[:16]}.json")
    tmp = f"{final}.tmp.{os.getpid()}.{sub_index}"
    try:
        with open(tmp, "wb") as f:
            f.write(payload_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return final


# ===========================================================================
# Framed socket — wire-identical to persistent/pwc_transport_tcp.FramedSocket
# ===========================================================================

class MinerFramedSocket:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    def send_msg(self, msg: MinerBaseMessage) -> None:
        data = message_to_bytes(msg)
        with self._send_lock:
            self._sendall(data)

    def recv_msg(self) -> MinerBaseMessage:
        with self._recv_lock:
            header = self._recvall(HEADER_SIZE)
            if len(header) < HEADER_SIZE:
                raise ConnectionError("socket closed while reading header")
            (size,) = struct.unpack(">I", header)
            if size > MAX_FRAME_BYTES:
                raise ValueError(f"oversized message: {size} bytes")
            body = self._recvall(size)
            if len(body) != size:
                raise ConnectionError("socket closed while reading body")
            return from_dict(json.loads(body.decode("utf-8")))

    def _sendall(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            sent = self.sock.send(view)
            if sent <= 0:
                raise ConnectionError("socket send failed")
            view = view[sent:]

    def _recvall(self, n: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < n:
            chunk = self.sock.recv(n - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        return bytes(chunks)

    def close(self) -> None:
        # [DEFECT A] shutdown BEFORE close, for the same reason the coordinator's
        # `_drop_conn` does it (Defect 6 C3): a bare close() on a socket with a
        # concurrent BLOCKED recv may defer both the wake-up and the peer's FIN.
        # The worker's control loop is normally parked in `recv_msg`, so without
        # this a signal-driven `shutdown()` could leave the daemon blocked in a
        # read on an already-closed fd instead of exiting — the SIGNAL leg of §10.
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


# ===========================================================================
# GPU metadata / cold-start warm
# ===========================================================================

@dataclass
class GpuInfo:
    backend: str      # 'rocm' | 'cuda'
    gpu_name: str
    vram_bytes: int


def detect_gpu(device_index: int) -> GpuInfo:
    import cupy as cp
    rt = cp.cuda.runtime
    backend = "rocm" if getattr(rt, "is_hip", False) else "cuda"
    with cp.cuda.Device(device_index):
        try:
            props = rt.getDeviceProperties(device_index)
            name = props["name"]
            gpu_name = name.decode() if isinstance(name, bytes) else str(name)
        except Exception:
            gpu_name = f"{backend}:device{device_index}"
        vram_bytes = int(cp.cuda.Device(device_index).mem_info[1])
    return GpuInfo(backend=backend, gpu_name=gpu_name, vram_bytes=vram_bytes)


def warm_gpu(device_index: int) -> None:
    """Cold-start GPU safety (interface §2.5): warm THIS daemon's single GPU.
    Cross-GPU 'sequential/pairs, never all at once' sequencing is the SPAWNER's
    job — a per-GPU daemon owns exactly one device."""
    import cupy as cp
    with cp.cuda.Device(device_index):
        a = cp.arange(1024, dtype=cp.uint32)
        _ = int((a + 1).sum().get())
        del a
        cp.cuda.Stream.null.synchronize()


# ===========================================================================
# The daemon
# ===========================================================================

ExecutorFn = Callable[[StripeAssignMessage, int, int], SubStripeOutcome]


# ===========================================================================
# [DEFECT A] TRANSPORT-SESSION RECOVERY — Beta Gate-12 attempt-2 ruling §§10-14
# ===========================================================================
# The defect: `serve_forever` exited at ONE point for THREE different causes —
# an explicit `shutdown` frame, a signal, and a bare transport loss — so a
# permitted worker that lost its session could never re-register, and a later
# stage's admission was permanently short (attempt-2's 23/25). The coordinator
# already holds the useful half (`_drop_conn` evicts the dead socket's identity;
# `_serve_register` accepts the same identity on a NEW socket once evicted, and
# rejects a duplicate that races the eviction). The missing seam was worker-side.
#
# §10 three-way session state machine. THE LOAD-BEARING DISCRIMINATOR is
# `self._stop.is_set()` at the moment the transport exception is caught:
#   _stop SET   -> a `shutdown` frame (`_dispatch`) or a signal (`_handle_sig`)
#                  already fired. Exit. NO reconnect. (gate A6)
#   _stop CLEAR -> genuine transport loss. Close the dead session, reconnect
#                  under a bounded backoff, re-register the SAME identity, and
#                  resume an IDLE service loop. (gate A1)
STOP_CAUSE_EXPLICIT_SHUTDOWN = "explicit_shutdown"   # `shutdown` frame
STOP_CAUSE_SIGNAL = "signal"                         # SIGTERM / SIGINT
SESSION_END_TRANSPORT_LOSS = "transport_loss"        # recoverable
SESSION_END_IDENTITY_REFUSED = "identity_refused"    # §13 fail-closed

# The RECEIVE-side transport surface. Identical to the tuple the certified loop
# already caught, so classification does not widen what counts as a transport
# failure on the read path: unexpected EOF and framing failures arrive as
# ConnectionError / ValueError from MinerFramedSocket.recv_msg, and socket faults
# as OSError (ConnectionError is itself an OSError subclass; both are named for
# intent, as before).
TRANSPORT_EXCEPTIONS = (ConnectionError, ValueError, OSError)

# The SEND-side surface, deliberately NARROWER — ValueError is excluded.
# `_dispatch` was previously outside any guard, so widening it needs care: on the
# write path a ValueError is NOT a dead socket, it is `message_to_bytes` refusing
# an oversized frame, i.e. a payload-contract violation. Recovering from that
# would reconnect the worker and swallow a real defect, so it stays uncaught and
# keeps propagating exactly as it does today.
SEND_TRANSPORT_EXCEPTIONS = (ConnectionError, OSError)

# §14 backoff shape. Bounded and DERIVED from the recovery budget rather than
# invented: the per-attempt ceiling is a twelfth of the budget, so at the 180 s
# anchor the ceiling is 15 s and at least a dozen attempts fit inside the bound
# (1+2+4+8 then 15 s steps). The budget itself is the authority; these only
# shape the spacing of attempts underneath it.
RECONNECT_BACKOFF_INITIAL_S = 1.0
RECONNECT_BACKOFF_BUDGET_DIVISOR = 12.0


def default_recovery_budget_s() -> float:
    """§14 RECONNECT BOUND — a POSITIVE FINITE run-scoped bound, DERIVED from the
    existing liveness contract rather than invented.

    The anchor is `DEFAULT_WORKER_ADMISSION_TIMEOUT` (the coordinator's
    `worker_admission_timeout` / `_tcp_wait_ready` authority): a worker whose
    recovery exceeds the window the coordinator would wait for it cannot help
    this trial anyway. The constant is READ, never redefined and never changed —
    it is on Beta's §25 no-touch list.

    Imported lazily and NOT at module scope on purpose: `miner/__init__.py`
    imports the coordinator, and the coordinator imports THIS module at its
    line 55, so a module-level `from miner.range_miner_coordinator import ...`
    here would resolve against a half-initialised coordinator (the constant is
    defined below that import) and raise ImportError. At call time the package
    is fully loaded.
    """
    from miner.range_miner_coordinator import DEFAULT_WORKER_ADMISSION_TIMEOUT
    return float(DEFAULT_WORKER_ADMISSION_TIMEOUT)


@dataclass
class SessionOutcome:
    """Why one transport session ended, and what was in flight when it did."""
    cause: str
    exc_class: Optional[str] = None
    exc_text: Optional[str] = None
    assignment_active: bool = False
    stripe_id: str = ""
    sub_index: int = -1

    @property
    def recoverable(self) -> bool:
        return self.cause == SESSION_END_TRANSPORT_LOSS


class WorkerIdentityChanged(RuntimeError):
    """§13 frozen-cohort wall: a re-registration whose capability identity is not
    byte-identical to the one the cohort was frozen against. Fail closed."""


# §13 IDENTITY = exactly the fields the frozen cohort was admitted on. Enumerated
# rather than taken as "every field of RegisterMessage", because the envelope also
# carries PER-MESSAGE metadata: `timestamp` necessarily differs between the first
# registration and any reconnect, so comparing the whole dataclass would fail
# closed on every recovery and silently re-create the no-reconnect defect wearing
# a §13 costume. `message_type`/`protocol_version` are framing, not identity.
IDENTITY_FIELDS = (
    "worker_id",       # {hostname}:gpu{gpu_id} — the admission key
    "hostname",        # node identity
    "gpu_id",          # device identity within the node
    "gpu_name",
    "backend",         # rocm | cuda
    "vram_bytes",
    "capabilities",    # supported_variants + seed_caps (the capability signature)
)


def identity_projection(msg: RegisterMessage) -> Dict[str, Any]:
    """The §13 comparison surface of a REGISTER frame."""
    return {k: getattr(msg, k) for k in IDENTITY_FIELDS}


class RangeMinerWorker:
    def __init__(
        self,
        host: str,
        port: int,
        gpu_id: int,
        caps: VramCaps,
        *,
        device_index: Optional[int] = None,
        executor: Optional[ExecutorFn] = None,
        gpu_info: Optional[GpuInfo] = None,
        hostname: Optional[str] = None,
        heartbeat_interval: float = 30.0,
        default_stripe_seeds: int = 67_108_864,
        miner_output_dir: Optional[str] = None,
        recovery_budget_s: Optional[float] = None,
        reconnect_enabled: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.gpu_id = gpu_id
        self.device_index = device_index if device_index is not None else gpu_id
        self.caps = caps
        self._executor = executor
        self.gpu_info = gpu_info
        self.hostname = hostname or socket.gethostname()
        self.worker_id = f"{self.hostname}:gpu{gpu_id}"
        self.heartbeat_interval = heartbeat_interval
        self.default_stripe_seeds = default_stripe_seeds
        self.miner_output_dir = miner_output_dir or resolve_miner_output_dir(None)

        self.conn: Optional[MinerFramedSocket] = None
        self._stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self._send_guard = threading.Lock()

        self.state = "idle"
        self.current_stripe_id = ""
        self.current_sub_index = -1
        self.progress = 0.0
        self.stripes_done = 0
        self.stripes_error = 0

        # ----- [DEFECT A] transport-session recovery state (§§10-15) --------
        # `_stop_cause` records WHY the stop flag was raised, first-writer wins,
        # so an explicit shutdown and a signal stay distinguishable (§10) instead
        # of collapsing into the one exit the defect was made of.
        self._stop_cause: Optional[str] = None
        self.reconnect_enabled = reconnect_enabled
        # §14: a POSITIVE FINITE run-scoped bound. Resolution of the default is
        # deferred to first use so the derivation is read at runtime, not baked.
        self._recovery_budget_s: Optional[float] = (
            None if recovery_budget_s is None else float(recovery_budget_s))
        # Cumulative, NOT per-episode: total wall time this worker has ever spent
        # inside recovery. A per-episode budget would reset on every session and
        # re-create the immortal orphan §14 forbids (a duplicate-rejection
        # ping-pong would retry for ever). Productive time is never charged.
        self._recovery_spent_s = 0.0
        self.reconnect_attempts_total = 0
        self.session_generation = 1
        self.session_events: List[Dict[str, Any]] = []
        # §13: the identity the cohort was frozen against, captured at the FIRST
        # registration and re-sent verbatim on every reconnect.
        self._identity_frame: Optional[Dict[str, Any]] = None

    # ----- [DEFECT A] §15 observability ------------------------------------
    def _emit_session_event(self, kind: str, **fields: Any) -> Dict[str, Any]:
        """Record one session-lifecycle fact. Structured fields carry machine
        truth (the harness and any future forensic read off `session_events`),
        the log line carries the diagnostic prose. Emitted ONLY on session
        transitions — never per heartbeat, per §15's no-high-rate-noise bar."""
        event = {
            "event": kind,
            "worker_id": self.worker_id,
            "session_generation": self.session_generation,
            **fields,
        }
        self.session_events.append(event)
        logger.warning("[MINER-SESSION] %s %s", kind, json.dumps(event, default=str,
                                                                 sort_keys=True))
        return event

    def recovery_budget_s(self) -> float:
        """§14 bound, validated POSITIVE FINITE exactly as the coordinator
        validates `worker_admission_timeout` itself."""
        if self._recovery_budget_s is None:
            self._recovery_budget_s = default_recovery_budget_s()
        budget = self._recovery_budget_s
        if not (budget > 0.0 and math.isfinite(budget)):
            raise ValueError(
                "recovery_budget_s must be a POSITIVE FINITE number of seconds "
                f"(got {budget!r})")
        return budget

    # ----- connection / handshake ------------------------------------------
    def connect(self) -> None:
        sock = socket.create_connection((self.host, self.port))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.conn = MinerFramedSocket(sock)

    def _build_register_message(self) -> RegisterMessage:
        info = self.gpu_info
        assert info is not None, "gpu_info must be resolved before register"
        return RegisterMessage(
            worker_id=self.worker_id,
            hostname=self.hostname,
            gpu_id=self.gpu_id,
            gpu_name=info.gpu_name,
            backend=info.backend,
            vram_bytes=info.vram_bytes,
            capabilities={
                # EXACT concrete variants (B4) — validated, not base x suffix.
                "supported_variants": supported_variants(),
                "seed_caps": dataclasses.asdict(self.caps),
            },
        )

    def register(self) -> None:
        if self.gpu_info is None:
            self.gpu_info = detect_gpu(self.device_index)
            warm_gpu(self.device_index)
        msg = self._build_register_message()
        # [DEFECT A §13] FROZEN COHORT REMAINS AUTHORITY. The first registration
        # freezes the identity this worker was admitted under — worker_id, node
        # identity, gpu/device identity, backend, VRAM, supported variants, seed
        # caps. Every reconnect re-sends THAT frame verbatim, and an identity that
        # has drifted fails closed instead of re-registering as something else.
        # No cohort expansion, and no reconnect that quietly changes what the
        # coordinator admitted.
        frame = identity_projection(msg)
        if self._identity_frame is None:
            self._identity_frame = frame
        elif frame != self._identity_frame:
            changed = sorted(
                k for k in set(frame) | set(self._identity_frame)
                if frame.get(k) != self._identity_frame.get(k))
            self._emit_session_event(
                "IDENTITY_REFUSED", changed_fields=changed,
                reason="capability identity differs from the frozen registration")
            raise WorkerIdentityChanged(
                f"reconnect identity for {self.worker_id} differs from the frozen "
                f"cohort registration in {changed}; refusing to re-register "
                f"(Beta §13 — the frozen cohort remains authority)")
        self._send(msg)

    # ----- send helper (serialized) ----------------------------------------
    def _send(self, msg: MinerBaseMessage) -> None:
        assert self.conn is not None, "not connected"
        with self._send_guard:
            self.conn.send_msg(msg)

    # ----- heartbeat -------------------------------------------------------
    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            try:
                self._send(
                    MinerHeartbeatMessage(
                        worker_id=self.worker_id,
                        stripes_done=self.stripes_done,
                        stripes_error=self.stripes_error,
                        current_stripe_id=self.current_stripe_id,
                        busy=(self.state == "mining"),
                    )
                )
            except Exception:
                break

    # ----- stripe handling -------------------------------------------------
    def _executor_for(self) -> ExecutorFn:
        if self._executor is not None:
            return self._executor
        raise RuntimeError("no executor configured")

    def handle_stripe(self, assign: StripeAssignMessage) -> None:
        """Partition a stripe, run each sub-stripe, stream results. A sub-stripe
        failure emits a well-formed stripe_error and stops THIS stripe; the daemon
        stays alive (TB Q3: retry is the coordinator's job)."""
        self.state = "mining"
        self.current_stripe_id = assign.stripe_id
        self.progress = 0.0
        t0 = time.time()

        backend = self.gpu_info.backend if self.gpu_info else "cuda"
        cap = select_seed_cap(backend, assign.family_name, self.caps)
        seed_count = assign.seed_count or self.default_stripe_seeds
        subs = partition_stripe(assign.seed_start, seed_count, cap)
        executor = self._executor_for()

        survivors_total = 0
        # [S172 D6] Effective-threshold provenance for the stripe roll-up. Every
        # sub-stripe of one stripe runs the same kernel with the same threshold,
        # so a disagreement is a real defect — collected as a set and reported,
        # never averaged or silently reduced to the first value seen.
        effective_seen: List[float] = []
        for sub in subs:
            self.current_sub_index = sub.sub_index
            try:
                outcome = executor(assign, sub.seed_start, sub.seed_count)
            except (NotImplementedError, ResidueError, ThresholdContractError) as e:
                # Non-retryable: uncovered family, or unresolved/mismatched window.
                self._fail_stripe(assign, sub, e, retryable=False)
                return
            except Exception as e:
                self._fail_stripe(assign, sub, e, retryable=True)
                return

            survivors_total += outcome.count
            eff = getattr(outcome, "effective_threshold", None)
            if eff is not None:
                effective_seen.append(float(eff))
            self._send(self._build_sub_result(assign, sub, outcome))
            self.progress = (sub.sub_index + 1) / len(subs) if subs else 1.0

        self.stripes_done += 1
        self.state = "idle"
        self.current_sub_index = -1
        distinct_eff = sorted(set(effective_seen))
        if len(distinct_eff) > 1:
            # Not silently reduced: a stripe whose sub-stripes filtered at
            # different thresholds is a defect, and the roll-up must not paper
            # over it by reporting one of them as "the" effective value.
            logger.error(
                "stripe %s sub-stripes filtered at DIFFERENT thresholds %r — "
                "reporting no single effective value (S172 D6 provenance)",
                assign.stripe_id, distinct_eff)
        self._send(
            StripeCompleteMessage(
                worker_id=self.worker_id,
                stripe_id=assign.stripe_id,
                substripes_done=len(subs),
                survivors_total=survivors_total,
                elapsed_s=round(time.time() - t0, 3),
                effective_threshold=(distinct_eff[0]
                                     if len(distinct_eff) == 1 else None),
            )
        )

    def _fail_stripe(self, assign, sub, exc, *, retryable: bool) -> None:
        self.stripes_error += 1
        self.state = "idle"
        self.current_sub_index = -1
        self._send(
            StripeErrorMessage(
                worker_id=self.worker_id,
                stripe_id=assign.stripe_id,
                sub_index=sub.sub_index,
                error=str(exc),
                traceback=traceback.format_exc(),
                retryable=retryable,
            )
        )

    def _build_sub_result(
        self, assign: StripeAssignMessage, sub: SubStripe, outcome: SubStripeOutcome
    ) -> SubStripeResultMessage:
        """Size-based inline/spool (B2). Hash/size are over the canonical
        payload_bytes; inline carries the SAME logical payload_obj."""
        payload_obj, payload_bytes = build_substripe_payload_bytes(
            assign.stripe_id, sub.sub_index, sub.seed_start, sub.seed_count,
            outcome.survivors,
        )
        sha = hashlib.sha256(payload_bytes).hexdigest()
        size = len(payload_bytes)

        candidate = SubStripeResultMessage(
            worker_id=self.worker_id,
            stripe_id=assign.stripe_id,
            sub_index=sub.sub_index,
            seed_start=sub.seed_start,
            seed_count=sub.seed_count,
            survivor_count=outcome.count,
            inline=payload_obj,
            size_bytes=size,
            sha256=sha,
            # [S172 D6] provenance rides the result envelope, NOT the canonical
            # payload_bytes — the spool/inline byte schema and its sha256 are
            # unchanged (D5 contract untouched).
            effective_threshold=getattr(outcome, "effective_threshold", None),
        )
        # Size-based (NOT count-based) inline/spool, WITHOUT framing a known-large
        # candidate (B2 v3): if the payload alone already meets the ceiling, spool
        # straight away — never call message_to_bytes on it (encode_frame raises a
        # ValueError past MAX_FRAME_BYTES, which would abort stripe handling). Only
        # frame to measure when the payload is safely small; a framing ValueError
        # is itself a "must spool" signal, not a failure.
        should_spool = len(payload_bytes) >= INLINE_BYTE_LIMIT
        if not should_spool:
            try:
                should_spool = len(message_to_bytes(candidate)) > INLINE_BYTE_LIMIT
            except ValueError:
                should_spool = True
        if not should_spool:
            return candidate

        # Spool: write the exact payload_bytes atomically, clear inline.
        spool_path = spool_payload_atomic(
            self.miner_output_dir, assign.stripe_id, sub.sub_index, sha, payload_bytes
        )
        candidate.inline = None
        candidate.spool_path = spool_path
        return candidate

    # ----- control loop ----------------------------------------------------
    def serve_forever(self) -> None:
        """[DEFECT A §10] Serve transport SESSIONS, not one socket.

        A session ends for exactly one of three causes, and they no longer share
        an exit: an explicit `shutdown` frame and a signal both end the daemon,
        while a transport loss with `_stop` clear is recovered. See
        `_run_session` for the classification and `_recover_session` for §§11-14.
        """
        assert self.conn is not None, "call connect() + register() first"
        try:
            while True:
                outcome = self._run_session()
                if not outcome.recoverable:
                    # EXPLICIT SHUTDOWN or SIGNAL (or a §13 identity refusal):
                    # exit, NO reconnect. This is the branch A6 gates.
                    self._emit_session_event(
                        "SESSION_END", classification=outcome.cause,
                        exc_class=outcome.exc_class,
                        reconnect_attempted=False,
                        assignment_active_at_loss=outcome.assignment_active)
                    return
                if not self._recover_session(outcome):
                    return
        finally:
            self.shutdown()

    def _run_session(self) -> SessionOutcome:
        """Run ONE transport session to its end and classify why it ended."""
        self._start_heartbeat()
        while not self._stop.is_set():
            conn = self.conn
            if conn is None:
                # shutdown() ran concurrently (signal path closes the socket and
                # clears the handle); that is a stop, never a transport loss.
                break
            try:
                msg = conn.recv_msg()
            except TRANSPORT_EXCEPTIONS as e:
                return self._classify_session_end(e)
            try:
                self._dispatch(msg)
            except SEND_TRANSPORT_EXCEPTIONS as e:
                # The loss surfaced on a SEND — the socket died while this worker
                # was streaming results for an ACTIVE assignment. Before Defect A
                # this propagated straight out of `serve_forever` as an uncaught
                # traceback; it is the same transport loss and is classified as
                # one. What happens to the abandoned assignment is F1/F2's call,
                # never this worker's (§11).
                return self._classify_session_end(e)
        return SessionOutcome(
            cause=self._stop_cause or STOP_CAUSE_EXPLICIT_SHUTDOWN,
            assignment_active=(self.state == "mining"),
            stripe_id=self.current_stripe_id,
            sub_index=self.current_sub_index,
        )

    def _classify_session_end(self, exc: BaseException) -> SessionOutcome:
        """THE §10 DISCRIMINATOR. A transport exception alone does not authorise a
        reconnect — `_stop` must also be clear. If `_stop` is set, a `shutdown`
        frame or a signal already decided this daemon is finished, and the dying
        socket is a CONSEQUENCE of that decision, not a reason to re-register."""
        cause = (self._stop_cause or STOP_CAUSE_EXPLICIT_SHUTDOWN
                 if self._stop.is_set() else SESSION_END_TRANSPORT_LOSS)
        return SessionOutcome(
            cause=cause,
            exc_class=type(exc).__name__,
            exc_text=str(exc),
            assignment_active=(self.state == "mining"),
            stripe_id=self.current_stripe_id,
            sub_index=self.current_sub_index,
        )

    def _start_heartbeat(self) -> None:
        """Bind a heartbeat thread to the CURRENT session. `_heartbeat_loop`
        breaks out when its send fails, so a recovered session needs a live
        thread again; a thread that survived the loss keeps serving the new
        socket and is not duplicated."""
        if self._hb_thread is not None and self._hb_thread.is_alive():
            return
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"miner-heartbeat-g{self.session_generation}", daemon=True)
        self._hb_thread.start()

    # ----- [DEFECT A §§11-14] session recovery ------------------------------
    def _abandon_assignment_no_replay(self, outcome: SessionOutcome) -> None:
        """§11 NO STALE-WORK REPLAY. Reconnect is TRANSPORT recovery, not
        ASSIGNMENT recovery. The in-flight assignment is dropped here and nowhere
        remembered: this worker does not re-send the `SubStripeResult` whose write
        failed, does not resume the stripe from private state, does not declare
        the old assignment complete, and does not create a retry for itself. The
        certified F1/F2 machinery alone decides the abandoned assignment —
        constant-phase loss is terminal, a hybrid first loss earns the one
        certified retry on an alternate worker, a hybrid second loss is terminal.
        The worker comes back IDLE and waits for new dispatch."""
        self.state = "idle"
        self.current_stripe_id = ""
        self.current_sub_index = -1
        self.progress = 0.0
        if outcome.assignment_active:
            self._emit_session_event(
                "ASSIGNMENT_ABANDONED", stripe_id=outcome.stripe_id,
                sub_index=outcome.sub_index, replayed=False,
                authority="F1/F2 decides the abandoned assignment (Beta §11)")

    def _close_dead_session(self) -> None:
        """Drop the dead socket WITHOUT setting `_stop` (which `shutdown()` would
        do) and without opening a second one — §12's one-live-socket rule."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _backoff_delay(self, attempt: int) -> float:
        ceiling = max(RECONNECT_BACKOFF_INITIAL_S,
                      self.recovery_budget_s() / RECONNECT_BACKOFF_BUDGET_DIVISOR)
        return min(RECONNECT_BACKOFF_INITIAL_S * (2 ** max(0, attempt - 1)), ceiling)

    def _recover_session(self, outcome: SessionOutcome) -> bool:
        """Re-establish ONE session for the SAME identity. True if serving may
        resume, False if the daemon must exit (bound exhausted, stop raised, or a
        §13 identity refusal). Gates A1/A3/A4/A5/A7/A8."""
        self._abandon_assignment_no_replay(outcome)
        self._emit_session_event(
            "TRANSPORT_LOSS", classification=SESSION_END_TRANSPORT_LOSS,
            exc_class=outcome.exc_class, exc_text=outcome.exc_text,
            assignment_active_at_loss=outcome.assignment_active,
            stripe_id_at_loss=outcome.stripe_id,
            recovery_spent_s=round(self._recovery_spent_s, 3))
        self._close_dead_session()

        if not self.reconnect_enabled:
            # A2's control: the fix disabled. The worker exits exactly as it did
            # before Defect A, so admission is short and the certified
            # `worker_admission_timeout` decides the trial — untouched.
            self._emit_session_event("RECONNECT_DISABLED", reconnect_attempted=False)
            return False

        budget = self.recovery_budget_s()
        attempt = 0
        while True:
            if self._stop.is_set():
                # A shutdown frame or signal landed mid-recovery: stop wins.
                self._emit_session_event(
                    "RECONNECT_ABANDONED", reason=self._stop_cause or "stop_set",
                    attempt=attempt)
                return False
            remaining = budget - self._recovery_spent_s
            if remaining <= 0.0:
                # §14 exhaustion: exit cleanly. No immortal orphan that could
                # attach to a later, unrelated trial. Gate A8.
                self._emit_session_event(
                    "RECONNECT_EXHAUSTED", attempts=attempt,
                    recovery_budget_s=budget,
                    recovery_spent_s=round(self._recovery_spent_s, 3),
                    reconnect_success=False)
                return False

            delay = min(self._backoff_delay(attempt + 1), remaining)
            waited_from = time.monotonic()
            stopped = self._stop.wait(delay)
            self._recovery_spent_s += time.monotonic() - waited_from
            if stopped:
                self._emit_session_event(
                    "RECONNECT_ABANDONED", reason=self._stop_cause or "stop_set",
                    attempt=attempt)
                return False

            attempt += 1
            self.reconnect_attempts_total += 1
            t0 = time.monotonic()
            try:
                self.connect()
                self.register()
            except WorkerIdentityChanged:
                # §13 fail-closed. Not retryable: the identity, not the socket,
                # is what the coordinator refused. Gate A7.
                self._recovery_spent_s += time.monotonic() - t0
                self._close_dead_session()
                self._emit_session_event(
                    "SESSION_END", classification=SESSION_END_IDENTITY_REFUSED,
                    reconnect_attempted=True, attempt=attempt,
                    reconnect_success=False)
                return False
            except TRANSPORT_EXCEPTIONS as e:
                # §12: a coordinator that rejects the duplicate registration
                # races the eviction of the old socket and simply drops this new
                # one, so the refusal ARRIVES as a transport failure. It is a
                # retryable session-establishment condition: back off and try the
                # SAME identity again. Never force-replace, never hold a second
                # live socket. Gate A5.
                self._recovery_spent_s += time.monotonic() - t0
                self._close_dead_session()
                self._emit_session_event(
                    "RECONNECT_FAILED", attempt=attempt,
                    attempts_total=self.reconnect_attempts_total,
                    exc_class=type(e).__name__, exc_text=str(e),
                    reconnect_success=False,
                    recovery_spent_s=round(self._recovery_spent_s, 3))
                continue

            self._recovery_spent_s += time.monotonic() - t0
            self.session_generation += 1
            self._emit_session_event(
                "RECONNECTED", attempt=attempt,
                attempts_total=self.reconnect_attempts_total,
                reconnect_success=True, resumed_state="idle",
                recovery_spent_s=round(self._recovery_spent_s, 3))
            return True

    def _dispatch(self, msg: MinerBaseMessage) -> None:
        mtype = msg.message_type
        if mtype == "stripe_assign":
            self.handle_stripe(msg)  # type: ignore[arg-type]
        elif mtype == "shutdown":
            # [DEFECT A §10] EXPLICIT SHUTDOWN. Recording the cause is what keeps
            # this distinguishable from a transport loss; the socket dying right
            # after this frame must NOT be recovered. Gate A6.
            self._set_stop_cause(STOP_CAUSE_EXPLICIT_SHUTDOWN)
            self._stop.set()
        elif mtype == "status":
            self._send(
                MinerStatusMessage(
                    worker_id=self.worker_id,
                    state=self.state,
                    current_stripe_id=self.current_stripe_id,
                    sub_index=self.current_sub_index,
                    progress=self.progress,
                    stats={
                        "stripes_done": self.stripes_done,
                        "stripes_error": self.stripes_error,
                    },
                )
            )

    def _set_stop_cause(self, cause: str) -> None:
        """First writer wins — the cause that actually stopped the daemon, not the
        last teardown to run. `serve_forever`'s `finally: shutdown()` must never
        overwrite the real reason with a generic one."""
        if self._stop_cause is None:
            self._stop_cause = cause

    def shutdown(self, cause: Optional[str] = None) -> None:
        if cause is not None:
            self._set_stop_cause(cause)
        self._stop.set()
        if self.conn is not None:
            self.conn.close()
            self.conn = None


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S172 RANGE-MINER Phase 3 — per-GPU worker daemon"
    )
    p.add_argument("--host", required=True, help="coordinator host")
    p.add_argument("--port", type=int, default=DEFAULT_MINER_PORT)
    p.add_argument("--gpu-id", type=int, required=True,
                   help="logical GPU id (worker_id = {hostname}:gpu{gpu_id})")
    p.add_argument("--device-index", type=int, default=None,
                   help="cupy device index to bind (default = gpu-id; on ROCR-"
                        "remapped rigs the spawner sets ROCR_VISIBLE_DEVICES=0)")
    p.add_argument("--miner-output-dir", default=None,
                   help="spool dir; default auto-detect /dev/shm/prng/miner -> "
                        "~/miner_output (S172_INFRASTRUCTURE_INTERFACE §5)")
    p.add_argument("--heartbeat-interval", type=float, default=30.0)
    # Per-family VRAM caps (§12.4, TB Q2). Defaults = spec §7.
    # NOTE: the two *_hybrid caps are NOT yet in the WATCHER manifest (v1.8.0);
    # Phase 4/7 must wire them. Until then they come from these flags.
    p.add_argument("--seed-cap-nvidia", type=int, default=5_000_000)
    p.add_argument("--seed-cap-amd", type=int, default=2_000_000)
    p.add_argument("--seed-cap-nvidia-hybrid", type=int, default=2_500_000)
    p.add_argument("--seed-cap-amd-hybrid", type=int, default=1_000_000)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    caps = VramCaps(
        amd=args.seed_cap_amd,
        nvidia=args.seed_cap_nvidia,
        amd_hybrid=args.seed_cap_amd_hybrid,
        nvidia_hybrid=args.seed_cap_nvidia_hybrid,
    )
    device_index = args.device_index if args.device_index is not None else args.gpu_id

    # B1: NO process-lifetime draws preload. The residue window is resolved per
    # assignment from the payload (content-identity keyed).
    executor = SieveExecutor(resolver=ResidueResolver(), device_index=device_index)

    worker = RangeMinerWorker(
        host=args.host, port=args.port, gpu_id=args.gpu_id, caps=caps,
        device_index=device_index, executor=executor.execute,
        heartbeat_interval=args.heartbeat_interval,
        miner_output_dir=resolve_miner_output_dir(args.miner_output_dir),
    )

    def _handle_sig(signum, frame):  # noqa: ARG001
        # [DEFECT A §10] SIGNAL — a stop, never a transport loss. Naming the cause
        # is what stops the reconnect branch from firing on a real teardown (A6).
        worker.shutdown(cause=STOP_CAUSE_SIGNAL)

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    worker.connect()
    worker.register()
    worker.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
