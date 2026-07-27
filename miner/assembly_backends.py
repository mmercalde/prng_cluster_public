#!/usr/bin/env python3
"""
assembly_backends.py — S172 Phase 5, Deliverable D4: the two-backend assembly
interface plus the `serial_reference` implementation.

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D4.md (REV3, Team Beta
approved), frozen against HEAD f163199. Authority:
docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md §6.7 / §6.7.B / §17.

WHAT THIS MODULE IS
    * `ASSEMBLY_BACKENDS` — the frozen selector shape
      (`serial_reference`, `process_sharded`), declared in full NOW so D5 plugs
      into an existing seam and changes no interface.
    * `get_assembly_backend(name)` — name -> backend resolution that FAILS
      CLOSED. There is no silent default: per §17 `serial_reference` is the
      production default only as an explicitly configured value, never as a
      fallback after an error.
    * `SerialReferenceBackend` — a THIN wrapper over the existing D1.1 assembly
      path (`miner.range_miner_npz_writer.assemble_trial`), plus the §17
      measurement.
    * `AssemblyMeasurement` / `BackendAssemblyResult` — the frozen return
      contract [REV3 B1].

WHAT THIS MODULE IS EXPLICITLY NOT — every one of these already exists and D4
reuses it rather than reimplementing it (spec §0/§1.2; duplicating any of them
is a stop condition):
    * NO spool reading, manifest validation, directional-map construction or
      canonical record derivation  -> `range_miner_npz_writer.assemble_trial`
      (D1.1) owns all of it.
    * NO record -> 22-array columnization                -> `utils.canonical_arrays`
      (`records_to_arrays` / `validate_array_bundle`, D3).
    * NO canonical record building                       -> `utils.canonical_records`
      (`build_mode_records`, D3.25).
    * NO dedup, winner selection, seed ordering, array merge, contract wall or
      publication                                        -> `utils.run_finalizer`
      (`finalize_run`, D3.5). A backend produces a `MinerTrialAssembly` and
      STOPS. The finalizer is the CALLER's next step, never the backend's — this
      module deliberately does not import it.
    * NO `process_sharded` implementation — that is D5. It is declared here and
      resolving it raises `NotImplementedError`.
    * NO NPZ path population. Per D3.5 Ruling E, `MinerTrialAssembly`'s
      `binary_npz_path` / `all_npz_path` stay `None` through every backend; the
      finalizer publishes generation artifacts under its own layout.

`serial_reference`'s FOUR DOCUMENTED ROLES (§6.7.B)
    1. CORRECTNESS ORACLE  — the reference derivation every other backend's
       output is compared against; it is the definition of a right answer.
    2. FALLBACK            — the configured backend when `process_sharded` is
       not selected. Selecting it is always an explicit configuration act.
    3. BENCHMARK BASELINE  — the denominator of §17's promotion rule:
       `process_sharded` becomes production default only on a >=20% median
       end-to-end improvement over this backend (plus §17's three other
       conditions). D4 defines the measurement so D5 cannot invent one.
    4. DEBUG MODE          — single-process, single-threaded, no pool, no IPC:
       the mode in which an assembly defect is actually diagnosable.

MEASUREMENT ORDER IS BINDING [REV3 B5]
    start timer -> `assemble_trial` (NOT wrapped in try/except) -> stop timer ->
    compute the measurement from the NOW-VALIDATED manifests -> return.

    Computing `spool_bytes_read` BEFORE delegation would let a malformed
    manifest raise a raw `KeyError` out of this wrapper instead of D1.1's
    canonical fail-closed `SpoolIdentityError`. Measurement must never change
    failure behaviour: on any assembly failure the original exception
    propagates UNCHANGED, no result is returned, no partial measurement is
    published, and no backend state is updated. This module holds no mutable
    `last_measurement` state for the same reason.

`peak_rss_bytes` SEMANTICS — FROZEN [REV3 B3]
    Defined as the maximum aggregate resident memory of the backend PROCESS TREE
    during the measured `assemble()` call. For `serial_reference` the tree is
    just the current process, so on Linux:

        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    Three qualifications, recorded here because they bound what the number
    means:
      * on Linux `ru_maxrss` is reported in KiB and MUST be scaled by 1024
        (other platforms differ — macOS reports bytes);
      * it is a PROCESS-LIFETIME high-water mark, not automatically this call's
        peak. A value observed here may predate the measured call entirely;
      * for D5 the parent's RSS alone is NOT compliant. §17 requires the peak
        aggregate host RAM of the parent PLUS its concurrently live workers, and
        `RUSAGE_CHILDREN.ru_maxrss` does not establish that (it reports the
        maximum of any single reaped child, not a concurrent sum).

    BENCHMARK ISOLATION RULE. Because the value is a lifetime high-water mark,
    the authoritative §17 promotion benchmark must run each measured backend in
    a FRESH PROCESS so a previous call cannot contaminate the next. Therefore:

        D4 measurement         : proves the telemetry field and the serial
                                 instrumentation exist.
        Phase 6 isolated bench : produces the authoritative §17 promotion
                                 measurements.

    An in-harness `peak_rss_bytes` is NOT the final §17 comparison number.

DELIBERATELY NOT MEASURED HERE
    * publication time — D3.5's, and backend-independent;
    * GPU time — not Phase 5's.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import time

try:                                    # POSIX only; best-effort by contract
    import resource
except ImportError:                     # pragma: no cover - non-POSIX host
    resource = None                     # type: ignore[assignment]

from miner.assembly_shard_worker import run_sharded_assembly
from miner.range_miner_npz_writer import MinerTrialAssembly, assemble_trial

# ---------------------------------------------------------------------------
# The frozen selector shape (§3). BOTH names are declared in D4 even though only
# the first is implemented, so D5 adds an implementation and changes no
# interface. `process_sharded` must NOT be added later.
# ---------------------------------------------------------------------------
SERIAL_REFERENCE = "serial_reference"
PROCESS_SHARDED = "process_sharded"

ASSEMBLY_BACKENDS = (SERIAL_REFERENCE, PROCESS_SHARDED)


# ---------------------------------------------------------------------------
# §5 — the measurement §17's promotion rule needs, captured identically for
# every backend. Frozen: a measurement is a value, never a mutable accumulator.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AssemblyMeasurement:
    """One `assemble()` call's §17 telemetry.

    `wall_seconds`      perf_counter delta around the delegated assembly ONLY.
    `manifest_count`    number of input manifests.
    `spool_bytes_read`  sum of `expected_size` across the manifests, every one
                        of which D1.1 has by now verified against the actual
                        staged byte length (`range_miner_npz_writer.py:360`).
    `survivor_row_count`
                        constant + variable canonical records, i.e. the rows a
                        finalizer would receive.
    `peak_rss_bytes`    see the module docstring's frozen semantics; `None`
                        when `resource` is unavailable.
    """
    backend_name: str
    wall_seconds: float
    manifest_count: int
    spool_bytes_read: int
    survivor_row_count: int
    peak_rss_bytes: Optional[int]


@dataclass(frozen=True)
class BackendAssemblyResult:
    """The frozen backend return contract [REV3 B1].

    `assembly` is the UNMODIFIED object `assemble_trial` returned. A backend
    does not replace or extend its `timing` dict, does not add backend metrics
    to it, and does not touch either NPZ path field. Measurement travels
    alongside the assembly, never inside it — `MinerTrialAssembly`'s shape is
    frozen (§5).
    """
    assembly: MinerTrialAssembly
    measurement: AssemblyMeasurement


@runtime_checkable
class AssemblyBackend(Protocol):
    """The one interface, two eventual implementations, chosen by name (§3).

    It carries NO publication, dedup or ordering responsibility — those belong
    to D3.5's finalizer.
    """
    backend_name: str

    def assemble(self, run_id: str,
                 manifests: List[Dict[str, Any]]) -> BackendAssemblyResult:
        ...


def _peak_rss_bytes() -> Optional[int]:
    """Process-tree peak RSS in BYTES, best-effort (`None` if unavailable).

    For `serial_reference` the tree is the current process alone, so
    RUSAGE_SELF is the whole tree. `ru_maxrss` is KiB on Linux — hence the
    1024 scaling. See the module docstring for why this is not the §17 number.
    """
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    kib = getattr(usage, "ru_maxrss", None)
    if not isinstance(kib, int) or kib <= 0:
        return None
    return kib * 1024


# ---------------------------------------------------------------------------
# §4 — `serial_reference`: delegation + measurement, and nothing else.
# ---------------------------------------------------------------------------
class SerialReferenceBackend:
    """The reference backend: one process, one call into the shared D1.1 engine.

    Beyond delegating and measuring it does not alter, pre-filter, re-order or
    post-process the assembly in ANY way. See the module docstring for its four
    §6.7.B roles and for why the measurement is computed after delegation.
    """

    backend_name = SERIAL_REFERENCE

    def assemble(self, run_id: str,
                 manifests: List[Dict[str, Any]]) -> BackendAssemblyResult:
        """Assemble one trial through the shared engine and measure the call.

        `manifests` is `List[Dict[str, Any]]` — exactly the input domain
        `assemble_trial` declares (`range_miner_npz_writer.py:450`) and
        enforces (`isinstance(manifest, dict)`, `:280`) [REV3 B4]. Nothing is
        copied, normalized or converted before delegation: a backend exposes
        the shared assembler's domain, it does not widen it.

        Raises: whatever `assemble_trial` raises, UNCHANGED — PhaseIdentityError,
        AssemblyConsistencyError, SpoolIdentityError, DirectionalDuplicateError,
        AssemblyStateError, or ValueError from `utils.prng_encoding`. There is
        no try/except here by design (§4 [B5]).
        """
        # 1. start the timer
        started = time.perf_counter()
        # 2. delegate — deliberately NOT wrapped: the canonical D1.1 exception is
        #    the contract, and translating or annotating it would break it.
        assembly = assemble_trial(run_id, manifests)
        # 3. stop the timer on successful return only
        wall_seconds = time.perf_counter() - started
        # 4. measure from the NOW-VALIDATED manifests and the returned assembly.
        #    `m["expected_size"]` is a direct subscript on purpose: reaching this
        #    line means D1.1 already validated every manifest, so a KeyError is
        #    impossible here — whereas computing it BEFORE step 2 would raise
        #    one instead of D1.1's canonical spool error [B5].
        measurement = AssemblyMeasurement(
            backend_name=self.backend_name,
            wall_seconds=wall_seconds,
            manifest_count=len(manifests),
            spool_bytes_read=sum(m["expected_size"] for m in manifests),
            survivor_row_count=(len(assembly.canonical_records_constant)
                                + len(assembly.canonical_records_variable)),
            peak_rss_bytes=_peak_rss_bytes(),
        )
        # 5. return both, the assembly untouched
        return BackendAssemblyResult(assembly=assembly, measurement=measurement)


# ---------------------------------------------------------------------------
# §6.7 — `process_sharded` (D5): a THIN parent-side wrapper, measured exactly
# like `serial_reference`.
#
# Everything that makes this backend parallel lives in
# `miner/assembly_shard_worker.py`; this class does what
# `SerialReferenceBackend` does — start the timer, delegate, stop the timer,
# measure — and nothing else. Deliberately so: D4's gate G7 freezes THIS module
# as a selector + measurement layer that reads no file, hashes nothing, builds
# no array and sorts nothing, and the whole point of D5 is that both backends
# converge on ONE implementation of validation and of global assembly.
#
# `pool_size` is a REQUIRED constructor argument. There is no default and no
# `os.cpu_count()` anywhere: sizing the pool from the host's thread count is
# §6.7.A's fourth prohibition ("24 processes because Zeus exposes 24 threads").
#
# `peak_rss_bytes` here is the §5 CANONICAL value — the sampled concurrent-tree
# RSS-sum measured across worker creation, validation, artifact loading and the
# parent merge — NOT this process's `RUSAGE_SELF` high-water mark, which would
# ignore the workers entirely. Its definition string and sample interval travel
# in the D5 benchmark evidence rather than in `AssemblyMeasurement`, whose field
# tuple is frozen by D4.
# ---------------------------------------------------------------------------
class ProcessShardedBackend:
    """Spool validation fanned out across CPU-only processes; global assembly
    performed once, serially, in the parent.

    D5 implements it and MEASURES it. It does not promote it: §17's promotion
    rule (a >=20% median end-to-end improvement over `serial_reference`, plus
    §17's three other conditions) is decided by Phase 6's isolated benchmark,
    and `serial_reference` remains the production default until then.
    """

    backend_name = PROCESS_SHARDED

    def __init__(self, pool_size: int, *, start_method: str = "spawn",
                 sample_interval_ms: int = 25):
        self.pool_size = pool_size
        self.start_method = start_method
        self.sample_interval_ms = sample_interval_ms
        self.last_rss_evidence: Optional[Dict[str, Any]] = None

    def assemble(self, run_id: str,
                 manifests: List[Dict[str, Any]]) -> BackendAssemblyResult:
        """Assemble one trial through the sharded path and measure the call.

        Raises whatever the shared engine raises, UNCHANGED and in the same
        precedence as `serial_reference` — the metadata gauntlet runs in the
        parent before any worker is dispatched, and worker failures surface in
        deterministic manifest order. No try/except here, for D4's [B5] reason.
        """
        # NOTE ON WORDING: this method deliberately does NOT reuse
        # `SerialReferenceBackend.assemble`'s step comments or expression
        # spelling. D4's G8 mutates that module by UNIQUE source anchors
        # ("# 2. delegate", "# 3. stop the timer on successful return only",
        # "wall_seconds=wall_seconds,", the inline `spool_bytes_read=sum(...)`),
        # and `_patch` asserts each occurs exactly once. A second verbatim copy
        # here would make every one of those anchors ambiguous and red D4's
        # mutation proof. The ORDER below is nevertheless identical, because
        # [B5] is a behavioural requirement, not a formatting one.

        # (a) start the clock
        started = time.perf_counter()
        # (b) hand off to the sharded orchestration — deliberately NOT wrapped:
        #     the canonical D1.1 exceptions are the contract, and translating or
        #     annotating them would break it.
        outcome = run_sharded_assembly(
            run_id, manifests, self.pool_size,
            start_method=self.start_method,
            sample_interval_ms=self.sample_interval_ms)
        # (c) stop the clock on successful return only
        elapsed = time.perf_counter() - started
        assembly = outcome.assembly
        # (d) measure from the NOW-VALIDATED manifests. The `expected_size`
        #     subscript is safe only because reaching this line means every
        #     manifest already passed the gauntlet; computing it BEFORE the
        #     delegation would raise a raw KeyError out of this wrapper instead
        #     of the canonical fail-closed SpoolIdentityError [B5].
        bytes_read = sum(m["expected_size"] for m in manifests)
        self.last_rss_evidence = outcome.rss_evidence
        measurement = AssemblyMeasurement(
            backend_name=self.backend_name,
            wall_seconds=elapsed,
            manifest_count=len(manifests),
            spool_bytes_read=bytes_read,
            survivor_row_count=(len(assembly.canonical_records_constant)
                                + len(assembly.canonical_records_variable)),
            peak_rss_bytes=outcome.peak_rss,
        )
        # (e) return both, the assembly untouched
        return BackendAssemblyResult(assembly=assembly, measurement=measurement)


# ---------------------------------------------------------------------------
# §3 — resolution. FAILS CLOSED: no path returns `serial_reference` for an
# unknown, empty or missing name.
# ---------------------------------------------------------------------------
def get_assembly_backend(name: str, **options: Any) -> AssemblyBackend:
    """Resolve a backend by name.

    Unknown / empty / non-string name -> `ValueError` (hard fail, NEVER a silent
    default to `serial_reference`; §17 makes `serial_reference` the production
    default only as an explicitly configured value, never a post-error
    fallback).

    `process_sharded` is IMPLEMENTED as of D5, but it has no default
    configuration, so resolving it BY NAME ALONE still raises
    `NotImplementedError` naming D5 — the D4-frozen behaviour of this call
    shape, and the honest one: the backend cannot be constructed without an
    explicit `pool_size`, and §6.7.A forbids inventing one from the host's
    thread count. Select it as `get_assembly_backend("process_sharded",
    pool_size=N)`, or construct `ProcessShardedBackend(pool_size=N)` directly.
    Neither the configured nor the unconfigured path ever degrades to serial.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"assembly_backend must be a non-empty str, one of "
            f"{list(ASSEMBLY_BACKENDS)}; got {name!r} "
            f"({type(name).__name__}). There is no default backend.")
    if name not in ASSEMBLY_BACKENDS:
        raise ValueError(
            f"unknown assembly_backend {name!r}; expected one of "
            f"{list(ASSEMBLY_BACKENDS)}. Resolution fails closed — an unknown "
            f"name never falls back to {SERIAL_REFERENCE!r}.")
    if options:
        if name == SERIAL_REFERENCE:
            # `list`, not `sorted`: D4's G7 forbids this module from calling
            # `sorted` at all, because ordering anything here would be the first
            # step toward reimplementing assembly in the backend layer.
            raise ValueError(
                f"{SERIAL_REFERENCE!r} takes no configuration; got "
                f"{list(options)}. It is single-process by definition (§6.7.B "
                f"role 4: the mode in which a defect is diagnosable).")
        return ProcessShardedBackend(**options)
    if name == PROCESS_SHARDED:
        raise NotImplementedError(
            f"assembly_backend {PROCESS_SHARDED!r} is implemented in D5 but has "
            f"NO default configuration: it requires an explicit pool_size, e.g. "
            f"get_assembly_backend({PROCESS_SHARDED!r}, pool_size=4). Sizing a "
            f"pool from the host thread count is a §6.7.A prohibition, so "
            f"name-only resolution fails closed rather than guessing. It does "
            f"NOT fall back to {SERIAL_REFERENCE!r}; select that backend "
            f"explicitly if you want it.")
    return SerialReferenceBackend()
