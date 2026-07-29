#!/usr/bin/env python3
"""
step1_ingress.py — S172 Phase 5, Deliverable D6: the production integration
adapter that turns a committed RANGE-MINER trial into real Step-1 accumulator
candidates and a certified generation.

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D6.md (REV1), frozen against
HEAD 2a6e0f8. Authority: docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md, the D5 chain.

WHAT THIS MODULE IS
    The one seam between a finished miner trial and the legacy Step-1 shapes:

        serve_trial (real coordinator)
          -> AssemblingPhase5Sink  (the L6 boundary; assembly happens at commit)
          -> MinerTrialAssembly    (stored, retrievable by run_id)
          -> THIS MODULE           -> Step-1 accumulator candidates + counts
          -> utils.run_finalizer.finalize_run -> the certified generation

    It resolves the assembly backend (default `serial_reference`), builds the
    sink that uses it, fetches the STORED assembly fail-closed, and adapts that
    assembly back into the legacy accumulator shape.

WHAT THIS MODULE IS EXPLICITLY NOT — every one of these already exists, and D6
reuses it rather than reimplementing it (duplicating any of them is a stop
condition):
    * NO candidate normalization / record building. The miner already emits
      canonical 24-field records inside Phase-5 assembly, so
      `canonical_records_constant` / `_variable` are appended AS THEY STAND
      (D3.25 REV3 §4). This module never calls `normalize_trial_populations`,
      `build_mode_records`, or any four-map derivation.
    * NO PWC/ZMQ ingress. The D3.25 `step1_trial_populations_v2` contract and
      its adapter ingress wall belong to those two producers. The miner is not
      one of them; routing miner output through that wall is what §4 forbids.
      Nothing here imports `utils.canonical_records`.
    * NO assembly. That is D1.1's engine, reached through a D4/D5 backend.
    * NO dedup / winner selection / seed ordering / array merge / publication.
      That is `utils.run_finalizer.finalize_run` (D3.5), which the Step-1 run
      already calls once per RUN over the accumulator this module feeds
      (window_optimizer_integration_final.py:1812). D6 adds no second, per-trial
      finalization: the finalizer is shared, and the D3.5 author wired it that
      way for "every backend (legacy in-process sieve, PWC, ZMQ and — via D6 —
      the range miner)" (:1710).
    * NO NPZ path population on the assembly. Per D3.5 Ruling E,
      `MinerTrialAssembly.binary_npz_path` / `all_npz_path` stay `None` through
      every backend and through this adapter; the certified paths come from
      `RunArtifactResult` and from nowhere else. `certified_paths()` below reads
      the finalizer's result, never the assembly.

FAIL-CLOSED IS THE POINT (§2/§4.4)
    Before D6 this path appended nothing and returned +0 for every count, which
    is indistinguishable from a real empty trial. Every absent publication
    result here therefore RAISES `MinerIngressError`:

        * no sink wired                      -> raise (never a silent +0)
        * no `run_id` in the serve result    -> raise
        * `get_assembly(run_id)` is `None`   -> raise (uncommitted / aborted /
                                                tombstoned run — the accessor
                                                never returns a partial result)
        * a required certified path is `None`/missing on `RunArtifactResult`
                                             -> raise

    A committed trial with genuinely zero survivors is NOT one of these: it
    yields an assembly with empty populations and zero counts, which is a real
    zero and is reported as one.
"""
from typing import Any, Dict, List, Mapping, Optional, Tuple

from miner.assembly_backends import (
    SERIAL_REFERENCE,
    get_assembly_backend,
)
from miner.range_miner_npz_writer import (
    AssemblingPhase5Sink,
    MinerTrialAssembly,
)

__all__ = [
    "DEFAULT_ASSEMBLY_BACKEND",
    "MinerIngressError",
    "IngressCounts",
    "resolve_assembly_backend",
    "build_assembling_sink",
    "require_assembly",
    "ingest_assembly",
    "certified_paths",
]

# §17 / §2.4: `serial_reference` is the production default, applied HERE at the
# configuration layer. `get_assembly_backend` itself must never default — it
# fails closed on an unknown/empty name and never falls back after an error —
# so the default is expressed as the value this module passes IN, not as a
# fallback that module performs.
DEFAULT_ASSEMBLY_BACKEND = SERIAL_REFERENCE

# The four `RunArtifactResult` paths a certified generation must carry for the
# Step-2 loader to reach it. Hand-transcribed from the frozen result object
# (utils/run_finalizer.py:344-373); deliberately NOT derived from the dataclass
# fields, so a field renamed there is caught rather than silently followed.
_REQUIRED_ARTIFACT_PATHS: Tuple[str, ...] = (
    "generation_dir", "all_npz_path", "binary_npz_path", "sidecar_path",
)


class MinerIngressError(Exception):
    """A required D6 publication result was absent, so the miner trial cannot be
    turned into Step-1 candidates. Never downgraded to a zero/empty result."""


class IngressCounts:
    """What one trial's ingress actually appended, for the caller's TestResult.

    A value object, not an accumulator: it is computed once from the stored
    assembly and never mutated afterwards.
    """

    __slots__ = ("appended_constant", "appended_variable",
                 "forward_total", "reverse_total",
                 "forward_constant", "reverse_constant",
                 "bidirectional_total")

    def __init__(self, *, appended_constant: int, appended_variable: int,
                 forward_total: int, reverse_total: int,
                 forward_constant: int, reverse_constant: int,
                 bidirectional_total: int) -> None:
        self.appended_constant = appended_constant
        self.appended_variable = appended_variable
        self.forward_total = forward_total
        self.reverse_total = reverse_total
        self.forward_constant = forward_constant
        self.reverse_constant = reverse_constant
        self.bidirectional_total = bidirectional_total

    @property
    def appended_total(self) -> int:
        return self.appended_constant + self.appended_variable

    def __repr__(self) -> str:                              # pragma: no cover
        return (f"IngressCounts(appended={self.appended_total}, "
                f"fwd={self.forward_total}, rev={self.reverse_total}, "
                f"bidi={self.bidirectional_total})")


def resolve_assembly_backend(name: Optional[str] = None, **options: Any):
    """Resolve the configured assembly backend, defaulting to
    `serial_reference` when nothing is configured.

    `None` / absent configuration is the ONLY thing this function defaults; an
    explicit name is passed through to `get_assembly_backend` verbatim, so an
    unknown name still raises `ValueError` and `process_sharded` selected by
    name alone still raises `NotImplementedError` (it is selectable but
    unpromoted — Phase 6 owns promotion, and it needs an explicit `pool_size`).
    """
    return get_assembly_backend(DEFAULT_ASSEMBLY_BACKEND if name is None
                                else name, **options)


def build_assembling_sink(backend: Any = None) -> AssemblingPhase5Sink:
    """The Phase-5 sink the coordinator publishes into, using `backend`.

    Passing the resolved backend is what makes backend selection real: the sink
    is where assembly happens (at commit), so a backend that never reaches it
    would be configuration theatre.
    """
    return AssemblingPhase5Sink(
        backend=resolve_assembly_backend() if backend is None else backend)


def require_assembly(sink: Any, miner_result: Mapping[str, Any], *,
                     trial_number: int) -> MinerTrialAssembly:
    """Fetch the STORED assembly for this trial, or raise.

    `AssemblingPhase5Sink.get_assembly` returns the completed assembly, or
    `None` before a successful commit and after an abort — never a partial
    result. `None` here therefore means the trial never produced a publication
    result, and D6 refuses to represent that as an empty trial.
    """
    if sink is None:
        raise MinerIngressError(
            f"trial {trial_number}: the miner path ran with NO Phase-5 sink, so "
            f"no assembly exists to ingest. Before D6 this silently appended no "
            f"candidates and returned +0 counts, which is indistinguishable "
            f"from a real empty trial; it is now a hard failure.")
    if not hasattr(sink, "get_assembly"):
        raise MinerIngressError(
            f"trial {trial_number}: Phase-5 sink {type(sink).__name__} has no "
            f"get_assembly(run_id) accessor (frozen for D6).")
    run_id = miner_result.get("run_id") if miner_result is not None else None
    if not isinstance(run_id, str) or not run_id:
        raise MinerIngressError(
            f"trial {trial_number}: the miner result carries no usable run_id "
            f"(got {run_id!r}); the stored assembly cannot be addressed.")
    assembly = sink.get_assembly(run_id)
    if assembly is None:
        state = (miner_result.get("state") if miner_result is not None
                 else None)
        raise MinerIngressError(
            f"trial {trial_number}: no committed assembly for run_id {run_id!r} "
            f"(trial state {state!r}). get_assembly() returns None before a "
            f"successful commit and after an abort, and never a partial "
            f"result — failing closed rather than accumulating a fabricated "
            f"zero-candidate trial.")
    return assembly


def ingest_assembly(assembly: MinerTrialAssembly,
                    accumulator: Optional[Dict[str, Any]]) -> IngressCounts:
    """Adapt ONE stored assembly back into the legacy Step-1 accumulator shape.

    The records are appended EXACTLY as the assembly holds them — the same list
    objects' elements, in the same ascending-seed order, constant before
    variable (trial-major, mode-minor, matching the PWC/ZMQ adapter's ordering).
    Nothing is re-derived, re-sorted, re-keyed or copied through a normalizer:
    the miner's records are already the canonical 24-field shape, and rebuilding
    them would be the D3.25 §4 prohibition.

    `accumulator=None` (the pruned / count-only caller) appends nothing and
    still returns the real counts.
    """
    records_constant: List[Dict[str, Any]] = assembly.canonical_records_constant
    records_variable: List[Dict[str, Any]] = assembly.canonical_records_variable
    counts = assembly.directional_counts or {}

    # Direct subscripts on the four directional keys would be equally defensible
    # (D1.1 always populates all six), but a miner-side omission must not read
    # as a zero, so absence is a contract failure here too.
    missing = [k for k in ("forward_constant", "reverse_constant",
                           "forward_variable", "reverse_variable",
                           "bidirectional_constant", "bidirectional_variable")
               if k not in counts]
    if missing:
        raise MinerIngressError(
            f"run {assembly.run_id}: assembly directional_counts is missing "
            f"{missing} — the six-key shape is D1.1's contract "
            f"(range_miner_npz_writer.py:942); a missing count is a producer "
            f"defect, not a zero.")

    ingress = IngressCounts(
        appended_constant=len(records_constant),
        appended_variable=len(records_variable),
        forward_total=int(counts["forward_constant"]) + int(counts["forward_variable"]),
        reverse_total=int(counts["reverse_constant"]) + int(counts["reverse_variable"]),
        forward_constant=int(counts["forward_constant"]),
        reverse_constant=int(counts["reverse_constant"]),
        bidirectional_total=(int(counts["bidirectional_constant"])
                             + int(counts["bidirectional_variable"])),
    )

    if accumulator is not None:
        # Constant then variable. A seed present in BOTH modes yields TWO
        # records carrying their own mode's rates and aggregates — D1/D2
        # established that cross-mode duplication is legitimate, and the
        # finalizer's L2 key, not this adapter, decides the winner.
        accumulator.setdefault('bidirectional', []).extend(records_constant)
        accumulator['bidirectional'].extend(records_variable)
        # [S166-ACCUM] Count only — full objects are not retained for the two
        # directional populations, exactly as the PWC/ZMQ adapter does.
        accumulator['forward_count'] = (accumulator.get('forward_count', 0)
                                        + ingress.forward_total)
        accumulator['reverse_count'] = (accumulator.get('reverse_count', 0)
                                        + ingress.reverse_total)

    return ingress


def certified_paths(artifact: Any) -> Dict[str, str]:
    """The certified generation's paths, read from the finalizer's
    `RunArtifactResult` and from NOWHERE else.

    Ruling E: `MinerTrialAssembly.binary_npz_path` / `all_npz_path` stay `None`
    through every backend, so an adapter that "carried the certified paths" off
    the assembly would be carrying two Nones. Any missing or `None` path here
    is a failed publication and raises.
    """
    if artifact is None:
        raise MinerIngressError(
            "no RunArtifactResult: finalize_run did not publish a certified "
            "generation, so there are no certified paths to carry.")
    resolved: Dict[str, str] = {}
    for name in _REQUIRED_ARTIFACT_PATHS:
        value = getattr(artifact, name, None)
        if value is None:
            raise MinerIngressError(
                f"certified generation is missing a required path {name!r} "
                f"({type(artifact).__name__}). A generation is published only "
                f"after the pointer swap AND its durability fsync both succeed; "
                f"an absent path means it was not.")
        resolved[name] = str(value)
    return resolved
