#!/usr/bin/env python3
"""
assembly_shard_worker.py — S172 Phase 5, Deliverable D5: the CPU-only per-spool
validation worker, the projection artifact codec, the sampled concurrent-tree
RSS sampler, and the PARENT-side orchestration `process_sharded` drives.

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D5.md (REV1), frozen against
HEAD 3e8580a. Authority: docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md
§6.7.A / §6.7.B / §6.7.C / §17 and the binding Team Beta D5 ruling (option A +
sampled RSS-sum, findings F1-F4 locked).

THE ONE SENTENCE THIS MODULE EXISTS TO ENFORCE
    D5 parallelizes ONLY spool-local validation. D1.1 remains the sole authority
    for validation semantics AND global assembly semantics; workers produce
    ordered, lossless validated-spool artifacts, while the parent alone performs
    deterministic global merge, duplicate attribution, intersection, enrichment
    and final assembly.

WHY BOTH ENDS LIVE IN ONE MODULE
    The parent orchestration cannot live in `miner/assembly_backends.py`: D4's
    gate G7 freezes that module at AST level as a thin selector + measurement
    layer that calls `open`, `hashlib.*`, `numpy.*` and `sorted` NOWHERE and
    imports neither `numpy` nor `hashlib` at all. Artifact readback, digest
    cross-check and ordered result consumption need exactly those. So
    `assembly_backends.ProcessShardedBackend` stays thin and delegates here, and
    both ends of the shard protocol sit beside each other where their
    correspondence is reviewable.

WHAT THIS MODULE IS EXPLICITLY NOT — every one of these already exists and D5
reuses it rather than reimplementing it (brief §1.2; duplicating any is a stop
condition):
    * NO spool reading, identity, container or per-survivor semantic validation
      -> `range_miner_npz_writer.read_and_validate_spool` (D1.1, made public by
      the D5 Commit-1 extraction). The worker CALLS it; there is no second copy,
      which is why the two backends cannot diverge on validation semantics.
    * NO metadata gauntlet -> `range_miner_npz_writer.prepare_trial_assembly`.
    * NO directional-map construction, duplicate detection, provenance
      attribution, intersection or canonical enrichment
      -> `range_miner_npz_writer.merge_validated_spools`, called ONCE, in the
      parent, serially.
    * NO record -> 22-array columnization  -> `utils.canonical_arrays` (D3).
    * NO dedup, winner selection, seed ordering, array merge, contract wall or
      publication -> `utils.run_finalizer.finalize_run` (D3.5). A backend
      produces a `MinerTrialAssembly` and STOPS; this module deliberately does
      not import the finalizer.

THE FOUR §6.7.A IPC PROHIBITIONS, AND HOW EACH IS STRUCTURALLY AVOIDED
    1. survivor dicts through a Queue      — the worker returns a small OUTCOME
       ENVELOPE of scalars only: either the compact result (paths, counts, a
       digest, identity strings) or a `CapturedSpoolReadError` descriptor (a
       class name, a message, scalar args, scalar attribution). No survivor
       structure of any kind is in the return value.
    2. 22 NumPy arrays through pickle      — arrays never enter the IPC channel.
       They travel as an on-disk `.npz` the parent opens itself; only the PATH
       crosses. And the 22-array domain is D3's, reached long after this module.
    3. a giant parsed JSON payload sent parent -> child — the parent sends ONE
       small manifest dict. The child reads the spool bytes itself. The parsed
       payload stays local and ephemeral inside `read_and_validate_spool` and is
       discarded there.
    4. "24 processes because Zeus exposes 24 threads" — `pool_size` is a
       REQUIRED explicit parameter. There is no `os.cpu_count()` default
       anywhere in this module or in the backend.

WORKER LOSSLESSNESS [ruling finding F1, extended by REV3]
    The artifact stores exactly `ValidatedSpoolProjection`: `match_rates`
    (float64) plus whichever seed encoding that projection carries — `seeds_i64`
    (int64) on the fast path, or `seed_bytes` (uint8) + `seed_offsets` (uint64)
    once any seed in the spool leaves signed-64 — aligned, in survivor order,
    with NO sort, NO dedup and NO normalization, so an intra-spool duplicate
    seed survives as two rows and the parent's §5.4 duplicate invariant still
    fires (F3). The byte runs are sized by the DETERMINISTIC SIGNED-BYTE LENGTH
    FORMULA: the encoding may be non-minimal at negative signed-width
    boundaries, but decoding is exact and canonical assembly observes the
    original Python integer. Both encodings are plain numeric arrays, so the
    codec stays `allow_pickle=False` with no object array in either case, and a
    seed the pre-D5 engine accepted (arbitrary-precision Python int) still
    round-trips through a worker process exactly [REV3 §4]. `strategy_id` and
    the ragged `skips` are fully validated in the worker and then DISCARDED,
    exactly as §5.4's numeric encoding is validated-and-discarded, because
    canonical assembly never observes them — which is what keeps the ragged
    dimension, the only thing that would REQUIRE an object array, out of the
    artifact.

EXCEPTION PRECEDENCE IS DRIVEN BY `order`, NOT BY COMPLETION [REV2, Beta ruled B]
    The pre-D5 `assemble_trial` interleaves read and merge, so an
    earlier-position duplicate raises before a later-position spool is read. A
    parallel front end must read ahead, so it cannot reproduce that by raising
    from a worker. Instead:

        worker  -> returns a per-position OUTCOME: a projection artifact, or a
                   `CapturedSpoolReadError` descriptor for an allowlisted
                   canonical producer defect. It raises neither.
        parent  -> fills indexed slots via `as_completed` (observing nothing),
                   then REPLAYS position 0, 1, 2 ... feeding the SAME
                   `merge_validated_spools` the serial path uses. That is the
                   only place an outcome becomes visible.

    Consequence: `process_sharded` raises the same exception class, `.args`,
    rendered message and attribution as `serial_reference`, at the same position
    — reconstructed rather than original (tracebacks are explicitly not
    contractual, REV2 §3). Backend failures are a DIFFERENT type
    (`ProcessShardedAssemblyError`) and can never be mistaken for producer
    defects.

ARTIFACT DURABILITY ORDER — BINDING
    complete semantic validation -> projection construction -> temp write ->
    digest -> local read-back verification -> atomic rename.
    No incremental artifacts: a malformed survivor near the END of the JSON must
    prevent ANY successful artifact result, which it does because
    `read_and_validate_spool` raises before a projection exists at all.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from miner.range_miner_npz_writer import (
    CANONICAL_SPOOL_READ_ERRORS,
    SEED_ENCODING_INT64,
    SEED_ENCODING_SIGNED_BYTES,
    SEED_ENCODINGS,
    CapturedSpoolReadError,
    MinerTrialAssembly,
    ValidatedSpoolProjection,
    capture_spool_read_error,
    merge_validated_spools,
    prepare_trial_assembly,
    projection_seeds,
    read_and_validate_spool,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "PEAK_RSS_DEFINITION",
    "SAMPLE_INTERVAL_MS",
    "OUTCOME_PROJECTION",
    "OUTCOME_READ_ERROR",
    "ProcessShardedAssemblyError",
    "ShardArtifactError",
    "assert_cpu_only",
    "write_projection_artifact",
    "read_projection_artifact",
    "validate_spool_shard",
    "ProcessTreeRssSampler",
    "ShardedAssemblyOutcome",
    "run_sharded_assembly",
]

# The artifact is a D5-internal transport, never a published deliverable — it is
# created and deleted inside one `assemble()` call. The stamp exists so a
# readback can prove it is reading THIS codec's output.
ARTIFACT_SCHEMA_VERSION = "s172_spool_projection_v1"

# §5 canonical peak_rss definition + the binding benchmark sample interval.
PEAK_RSS_DEFINITION = "sampled_sum_of_parent_and_recursive_children_rss"
SAMPLE_INTERVAL_MS = 25

# Neither may be imported by a worker, and neither may have initialized a GPU
# context in it. Assembly is pure CPU work: byte reads, hashing, JSON parsing.
_FORBIDDEN_GPU_MODULES: Tuple[str, ...] = ("torch", "cupy")

# The compact worker return contract (brief §4.1.4). Paths and counts, never
# arrays and never payloads.
_RESULT_KEYS: Tuple[str, ...] = (
    "artifact_path", "survivor_count", "artifact_sha256",
    "stripe_id", "sub_index", "attempt", "workflow_phase",
    "direction", "skip_mode", "prng_type",
)

# Identity scalars carried INSIDE the artifact. These are defense-in-depth
# cross-checks only: the manifest/meta pair the parent already validated remains
# the source of truth, and a disagreement is a hard failure, never a fixup.
_IDENTITY_KEYS: Tuple[str, ...] = (
    "run_id", "stripe_id", "sub_index", "attempt", "workflow_phase",
    "direction", "skip_mode", "prng_type",
)

_STR_IDENTITY: Tuple[str, ...] = (
    "run_id", "stripe_id", "direction", "skip_mode", "prng_type",
)

# The two kinds of per-position outcome a worker can hand back [REV2 §3]. There
# is no third: a worker either fully validated its spool, or the spool carried a
# canonical, allowlisted producer defect. Anything else that goes wrong is a
# BACKEND failure and never travels as an outcome.
OUTCOME_PROJECTION = "projection"
OUTCOME_READ_ERROR = "read_error"

# Exactly the classes `capture_spool_read_error` will accept — imported, never
# re-enumerated here, so the worker's `except` clause and the canonical
# allowlist cannot drift apart.
_CANONICAL_READ_EXCEPTIONS: Tuple[type, ...] = tuple(
    CANONICAL_SPOOL_READ_ERRORS.values())


class ProcessShardedAssemblyError(Exception):
    """A BACKEND failure: a crashed worker, a broken pool, a malformed worker
    outcome, an unreadable or mismatched artifact, a digest failure, a timeout.

    Structurally distinct from every canonical producer defect (REV2 §5). A
    `SpoolReadOutcome` is exactly `projection | captured canonical producer
    defect`; infrastructure failure is NEITHER, and must never be descriptorized
    into a `CapturedSpoolReadError` — a parallel-transport bug masquerading as a
    `SpoolIdentityError` would blame the producer for the backend's own defect
    and would corrupt the equivalence contract it is supposed to prove."""


class ShardArtifactError(ProcessShardedAssemblyError):
    """A shard artifact failed to write, digest, read back, or cross-check
    against its authoritative manifest/meta pair.

    This is a D5 transport defect — one KIND of backend failure — structurally
    distinct from D1.1's `SpoolIdentityError` (a producer/contract defect in the
    spool itself), so a parallel-transport bug can never be mistaken for, or
    silently substituted for, a real spool validation failure."""


# ---------------------------------------------------------------------------
# CPU-only guard (brief §4.1, gated by G-NO-GPU).
# ---------------------------------------------------------------------------
def assert_cpu_only() -> None:
    """Hard-fail if a GPU library reached this interpreter.

    Under `spawn` each worker is a fresh interpreter, so nothing is inherited;
    this asserts that invariant rather than assuming it. `range_miner_worker`
    imports torch/cupy only INSIDE its kernel functions, so importing the D1.1
    engine — which imports the coordinator, which imports that module — pulls in
    no GPU library."""
    leaked = [m for m in _FORBIDDEN_GPU_MODULES if m in sys.modules]
    if leaked:
        raise ShardArtifactError(
            f"assembly shard worker (pid {os.getpid()}) has GPU module(s) "
            f"{leaked} in sys.modules — assembly is CPU-only work and a worker "
            f"must never hold a GPU context (§6.7.A)")


# ---------------------------------------------------------------------------
# Artifact codec — lossless w.r.t. the projection, `allow_pickle=False`.
# ---------------------------------------------------------------------------
def _identity_from(manifest: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """The identity scalars, read from the ALREADY-VALIDATED manifest/meta."""
    meta = manifest["trial_metadata"]
    return {
        "run_id": run_id,
        "stripe_id": str(manifest.get("stripe_id")),
        "sub_index": int(manifest.get("sub_index", 0)),
        "attempt": int(manifest.get("attempt", 0)),
        "workflow_phase": int(meta["workflow_phase"]),
        "direction": str(meta["direction"]),
        "skip_mode": str(meta["skip_mode"]),
        "prng_type": str(meta["prng_type"]),
    }


def write_projection_artifact(path: str, projection: ValidatedSpoolProjection,
                              identity: Dict[str, Any]) -> None:
    """Write ONE projection + its identity scalars to an uncompressed `.npz`.

    UNCOMPRESSED by §6.7.A: this artifact is written once and read once, inside
    a single call, so compression would spend CPU on the exact axis the
    deliverable is trying to parallelize.

    The seed encoding the projection chose travels WITH it as a scalar tag, and
    only that encoding's arrays are stored [REV3 §4] — `seeds_i64` (int64), or
    `seed_bytes` (uint8) + `seed_offsets` (uint64). Every stored value is
    therefore a plain numeric or unicode array — never an object array — so the
    parent can load with `allow_pickle=False` on both paths. Order and
    multiplicity of the seeds and of `match_rates` are stored verbatim."""
    encoding = projection.seed_encoding
    payload: Dict[str, np.ndarray] = {
        "schema_version": np.array(ARTIFACT_SCHEMA_VERSION),
        "seed_encoding": np.array(str(encoding)),
        "match_rates": np.asarray(projection.match_rates, dtype=np.float64),
        "survivor_count": np.array(int(projection.survivor_count),
                                   dtype=np.int64),
    }
    if encoding == SEED_ENCODING_INT64:
        payload["seeds_i64"] = np.asarray(projection.seeds_i64, dtype=np.int64)
    elif encoding == SEED_ENCODING_SIGNED_BYTES:
        payload["seed_bytes"] = np.asarray(projection.seed_bytes,
                                           dtype=np.uint8)
        payload["seed_offsets"] = np.asarray(projection.seed_offsets,
                                             dtype=np.uint64)
    else:
        raise ShardArtifactError(
            f"{path}: projection carries unknown seed_encoding {encoding!r}; "
            f"expected one of {list(SEED_ENCODINGS)}")
    for key in _IDENTITY_KEYS:
        value = identity[key]
        payload[key] = (np.array(str(value)) if key in _STR_IDENTITY
                        else np.array(int(value), dtype=np.int64))
    for key, arr in payload.items():
        if arr.dtype == object:                        # pragma: no cover - guard
            raise ShardArtifactError(
                f"{path}: artifact field {key!r} is an object array — the codec "
                f"must stay `allow_pickle=False`-loadable (§4.2)")
    with open(path, "wb") as fh:
        np.savez(fh, **payload)


def read_projection_artifact(
    path: str,
) -> Tuple[ValidatedSpoolProjection, Dict[str, Any]]:
    """Load an artifact back into `(projection, identity)`.

    `allow_pickle=False` is not a preference: it is the property that makes an
    artifact incapable of carrying executable state between processes.

    Reconstruction goes through `ValidatedSpoolProjection` itself, so the
    canonical shape invariants (exactly one encoding populated, rectangular,
    correctly typed, strictly increasing offsets spanning the byte run) are
    enforced by the ONE definition that owns them. A violation is re-raised as a
    `ShardArtifactError` — a BACKEND failure — because a malformed artifact is
    this transport's defect, never the producer's."""
    with np.load(path, allow_pickle=False) as bundle:
        stamp = str(bundle["schema_version"])
        if stamp != ARTIFACT_SCHEMA_VERSION:
            raise ShardArtifactError(
                f"{path}: artifact schema_version {stamp!r} != "
                f"{ARTIFACT_SCHEMA_VERSION!r}")
        encoding = str(bundle["seed_encoding"])
        if encoding == SEED_ENCODING_INT64:
            required = ("seeds_i64",)
        elif encoding == SEED_ENCODING_SIGNED_BYTES:
            required = ("seed_bytes", "seed_offsets")
        else:
            raise ShardArtifactError(
                f"{path}: artifact seed_encoding {encoding!r} is not one of "
                f"{list(SEED_ENCODINGS)}")
        missing = [k for k in required if k not in bundle.files]
        if missing:
            raise ShardArtifactError(
                f"{path}: artifact declares seed_encoding {encoding!r} but is "
                f"missing {missing!r}")
        seeds_i64 = (np.asarray(bundle["seeds_i64"], dtype=np.int64)
                     if encoding == SEED_ENCODING_INT64 else None)
        seed_bytes = (np.asarray(bundle["seed_bytes"], dtype=np.uint8)
                      if encoding == SEED_ENCODING_SIGNED_BYTES else None)
        seed_offsets = (np.asarray(bundle["seed_offsets"], dtype=np.uint64)
                        if encoding == SEED_ENCODING_SIGNED_BYTES else None)
        rates = np.asarray(bundle["match_rates"], dtype=np.float64)
        count = int(bundle["survivor_count"])
        identity = {
            key: (str(bundle[key]) if key in _STR_IDENTITY
                  else int(bundle[key]))
            for key in _IDENTITY_KEYS
        }
    try:
        projection = ValidatedSpoolProjection(
            seed_encoding=encoding, seeds_i64=seeds_i64,
            seed_bytes=seed_bytes, seed_offsets=seed_offsets,
            match_rates=rates, survivor_count=count)
    except ValueError as exc:
        raise ShardArtifactError(
            f"{path}: artifact does not reconstruct a valid projection: "
            f"{exc}") from exc
    return projection, identity


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The worker task. Top-level and picklable so `spawn` can reach it by name.
# ---------------------------------------------------------------------------
def validate_spool_shard(task: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ONE spool in a fresh CPU-only interpreter and durably stage its
    projection.

    `task` carries ONLY `run_id`, one small `manifest` dict, and the artifact
    directory — never payload bytes, never a parsed JSON object, never arrays.

    RETURNS a per-position OUTCOME envelope, never a raised producer defect
    [REV2 §3]:

        {"outcome_kind": "projection", "result": {<the 10 compact scalars>}}
        {"outcome_kind": "read_error", "descriptor": CapturedSpoolReadError}

    A canonical spool defect is DATA here precisely because raising it would
    make the observed error depend on which worker happened to finish first.
    The parent replays it at this shard's own position instead.

    Raises `ShardArtifactError` (a backend failure — never a producer defect)."""
    assert_cpu_only()
    run_id = task["run_id"]
    manifest = task["manifest"]
    artifact_dir = task["artifact_dir"]

    # 1. FULL semantic validation, through the SAME function the serial backend
    #    calls. A malformed survivor anywhere fails here, so no artifact — not
    #    even a partial one — can exist for a spool that does not fully pass.
    #    ONLY the allowlisted canonical classes are caught: a MemoryError, a
    #    KeyboardInterrupt or an unexpected programming error inside the
    #    validator propagates and becomes a BACKEND failure in the parent, which
    #    is what it is.
    try:
        projection = read_and_validate_spool(run_id, manifest)
    except _CANONICAL_READ_EXCEPTIONS as exc:
        return {"outcome_kind": OUTCOME_READ_ERROR,
                "descriptor": capture_spool_read_error(exc)}
    identity = _identity_from(manifest, run_id)

    stem = f"shard_{identity['workflow_phase']}_{identity['stripe_id']}_" \
           f"sub{identity['sub_index']}_a{identity['attempt']}"
    final_path = os.path.join(artifact_dir, f"{stem}.npz")

    # 2. temp write -> 3. digest -> 4. local read-back verify -> 5. atomic
    #    rename. The temp file is created in the SAME directory so the rename is
    #    a same-filesystem atomic operation, and it is removed on every failure
    #    path so no leaked temp survives an exception.
    fd, temp_path = tempfile.mkstemp(prefix=f".{stem}.", suffix=".npz.tmp",
                                     dir=artifact_dir)
    os.close(fd)
    try:
        write_projection_artifact(temp_path, projection, identity)
        artifact_sha256 = _sha256_file(temp_path)

        # local read-back: prove the bytes on disk decode to the SAME projection
        # before publishing them, so a corrupt artifact is never renamed into
        # place and the parent never has to trust an unverified write.
        echo, echo_identity = read_projection_artifact(temp_path)
        # Seeds are compared as DECODED Python ints, through the canonical
        # decoder, not as raw arrays: that is the statement the merge actually
        # depends on (exact value, order and multiplicity), and it holds
        # identically for both encodings [REV3 §4].
        if echo.survivor_count != projection.survivor_count \
                or echo.seed_encoding != projection.seed_encoding \
                or projection_seeds(echo) != projection_seeds(projection) \
                or not np.array_equal(echo.match_rates, projection.match_rates):
            raise ShardArtifactError(
                f"{temp_path}: read-back does not reproduce the projection "
                f"(order/multiplicity must survive the codec byte-for-byte)")
        if echo_identity != identity:
            raise ShardArtifactError(
                f"{temp_path}: read-back identity {echo_identity} != "
                f"{identity}")
        os.replace(temp_path, final_path)
    except BaseException:
        # G-ATOMIC / G-CLEANUP: nothing at the final path, no leaked temp.
        try:
            os.unlink(temp_path)
        except OSError:                                 # pragma: no cover - race
            pass
        raise

    result = dict(identity)
    result.pop("run_id")
    result["artifact_path"] = final_path
    result["survivor_count"] = int(projection.survivor_count)
    result["artifact_sha256"] = artifact_sha256
    return {"outcome_kind": OUTCOME_PROJECTION,
            "result": {k: result[k] for k in _RESULT_KEYS}}


# ---------------------------------------------------------------------------
# §5 — canonical peak_rss: SAMPLED CONCURRENT-TREE RSS-SUM.
# ---------------------------------------------------------------------------
class ProcessTreeRssSampler:
    """Sample `parent + recursive children` RSS and keep the maximum SUM.

    WHY NOT `RUSAGE_CHILDREN`: it reports the maximum of any single REAPED
    child, never the concurrent sum. With N workers each holding a substantial
    allocation at the same instant, it under-reports by roughly a factor of N —
    which is precisely what D5's RSS mutant demonstrates. It cannot establish
    §17's "peak aggregate host RAM of the parent plus its concurrently live
    workers", so it is ruled out.

    WHAT THE NUMBER MEANS: an RSS sum double-counts pages shared between the
    parent and its children (copy-on-write text, shared libraries). It is
    therefore a CONSERVATIVE process-tree footprint — an upper bound on the
    tree's private footprint — and NOT exact physical RAM. PSS would be exact,
    but it is Linux-only and unreadable for processes owned by another user, so
    it is optional telemetry here and never gating.

    Sampling covers the whole measured region: it starts BEFORE any worker is
    created and stops only AFTER the workers have joined and the parent merge
    has completed, so a peak that occurs during artifact loading or during the
    merge is captured too."""

    def __init__(self, interval_ms: int = SAMPLE_INTERVAL_MS):
        self.interval_s = interval_ms / 1000.0
        self.interval_ms = interval_ms
        self.peak_rss = 0
        self.peak_pss: Optional[int] = None
        self.sample_count = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._psutil = None

    def _tree_sum(self) -> Tuple[int, Optional[int]]:
        psutil = self._psutil
        parent = self._parent
        rss_total = 0
        pss_total: Optional[int] = 0
        seen = set()
        try:
            procs = [parent] + parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.ZombieProcess,
                psutil.AccessDenied):                   # pragma: no cover - race
            return 0, None
        for proc in procs:
            # PID-deduplicated: `children(recursive=True)` can repeat a pid if
            # the tree is re-parented mid-walk, and double-counting one process
            # would inflate the peak.
            if proc.pid in seen:
                continue
            seen.add(proc.pid)
            try:
                rss_total += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.ZombieProcess,
                    psutil.AccessDenied):
                # A process exiting between the walk and the read is normal, not
                # an error: skip it and keep sampling.
                continue
            if pss_total is not None:
                try:                        # optional Linux-only telemetry
                    pss_total += proc.memory_full_info().pss
                except (AttributeError, NotImplementedError,
                        psutil.NoSuchProcess, psutil.ZombieProcess,
                        psutil.AccessDenied):
                    pss_total = None
        return rss_total, pss_total

    def _loop(self) -> None:
        while not self._stop.is_set():
            rss, pss = self._tree_sum()
            self.sample_count += 1
            if rss > self.peak_rss:
                self.peak_rss = rss
            if pss is not None and (self.peak_pss is None or pss > self.peak_pss):
                self.peak_pss = pss
            # time.monotonic()-paced: Event.wait uses the monotonic clock, so a
            # wall-clock adjustment cannot stretch or collapse the interval.
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "ProcessTreeRssSampler":
        import psutil                      # local: keeps it off the import path
        self._psutil = psutil              # of every module that touches D1.1
        self._parent = psutil.Process(os.getpid())
        self._sample_once()                # a peak is never 0 even if the region
        self._thread = threading.Thread(   # is shorter than one interval
            target=self._loop, name="d5-rss-sampler", daemon=True)
        self._thread.start()
        return self

    def _sample_once(self) -> None:
        rss, pss = self._tree_sum()
        self.sample_count += 1
        if rss > self.peak_rss:
            self.peak_rss = rss
        if pss is not None and (self.peak_pss is None or pss > self.peak_pss):
            self.peak_pss = pss

    def __exit__(self, *exc: Any) -> None:
        self._sample_once()                # capture the merge's own high-water
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def evidence(self) -> Dict[str, Any]:
        """The §5 evidence block. `peak_pss_optional` is telemetry ONLY: it never
        replaces `peak_rss` and never participates in a pass/fail."""
        block = {
            "peak_rss": int(self.peak_rss),
            "peak_rss_definition": PEAK_RSS_DEFINITION,
            "sample_interval_ms": self.interval_ms,
            "sample_count": int(self.sample_count),
        }
        if self.peak_pss is not None:
            block["peak_pss_optional"] = int(self.peak_pss)
        return block


# ---------------------------------------------------------------------------
# Parent-side orchestration — the SOLE owner of global state (§6.7.C).
# ---------------------------------------------------------------------------
class ShardedAssemblyOutcome:
    """What `run_sharded_assembly` hands back: the assembly plus D5 telemetry.

    Deliberately NOT folded into `MinerTrialAssembly` (frozen, §7) nor into D4's
    `AssemblyMeasurement` (frozen field tuple, D4 gate). Measurement travels
    alongside the assembly, never inside it."""

    __slots__ = ("assembly", "peak_rss", "rss_evidence", "pool_size",
                 "start_method", "shard_count")

    def __init__(self, assembly: MinerTrialAssembly, peak_rss: int,
                 rss_evidence: Dict[str, Any], pool_size: int,
                 start_method: str, shard_count: int):
        self.assembly = assembly
        self.peak_rss = peak_rss
        self.rss_evidence = rss_evidence
        self.pool_size = pool_size
        self.start_method = start_method
        self.shard_count = shard_count


def _resolve_context(start_method: str):
    """Resolve the multiprocessing context. NEVER falls back to `fork`.

    `spawn` is canonical: a fresh interpreter per worker with no inherited GPU
    library state. `forkserver` is permitted only where a test proves its server
    starts clean. `fork` is refused outright — it would inherit the parent's
    entire address space, including any GPU context, which is the failure mode
    §6.7.A's CPU-only requirement exists to prevent. An unavailable context is a
    hard error, never a silent downgrade."""
    if start_method == "fork":
        raise ValueError(
            "process_sharded refuses the 'fork' start method: a forked worker "
            "inherits the parent's address space including any GPU context. "
            "Use 'spawn' (canonical) or 'forkserver'.")
    if start_method not in ("spawn", "forkserver"):
        raise ValueError(
            f"unknown multiprocessing start method {start_method!r}; "
            f"process_sharded accepts 'spawn' (canonical) or 'forkserver'.")
    available = multiprocessing.get_all_start_methods()
    if start_method not in available:
        raise ValueError(
            f"multiprocessing start method {start_method!r} is unavailable on "
            f"this host (have {available}); process_sharded fails closed rather "
            f"than silently falling back.")
    return multiprocessing.get_context(start_method)


def _crosscheck(index: int, result: Dict[str, Any], identity: Dict[str, Any],
                artifact_identity: Dict[str, Any], run_id: str) -> None:
    """Defense-in-depth: the worker result AND the artifact's own scalars must
    agree with the authoritative manifest/meta pair the parent already
    validated. The manifest/meta remain the source of truth; these are
    cross-checks, and a disagreement is a hard failure."""
    for key in _IDENTITY_KEYS:
        if artifact_identity[key] != identity[key]:
            raise ShardArtifactError(
                f"{run_id}: shard {index} artifact {key} "
                f"{artifact_identity[key]!r} != authoritative {identity[key]!r}")
        if key == "run_id":
            continue
        if result[key] != identity[key]:
            raise ShardArtifactError(
                f"{run_id}: shard {index} worker result {key} {result[key]!r} "
                f"!= authoritative {identity[key]!r}")


def _capture_worker_exception(
    exc: BaseException,
) -> Tuple[Optional[Dict[str, Any]], Optional[BaseException]]:
    """Classify an exception that escaped a worker into (outcome, failure).

    A worker is supposed to hand canonical producer defects back as descriptors
    itself. This is DEFENCE IN DEPTH for the case where one nevertheless
    escapes — for example raised outside the validator's own try block: it is
    still an allowlisted producer defect, so it must still be replayed at its
    own position rather than surfacing in completion order. Everything else is
    a backend failure and stays one."""
    if type(exc).__name__ in CANONICAL_SPOOL_READ_ERRORS:
        try:
            return ({"outcome_kind": OUTCOME_READ_ERROR,
                     "descriptor": capture_spool_read_error(exc)}, None)
        except TypeError:                               # pragma: no cover - guard
            return None, exc
    return None, exc


def _materialize_outcome(run_id: str, position: int, manifest: Dict[str, Any],
                         envelope: Any) -> Any:
    """Turn ONE worker envelope into the per-position `SpoolReadOutcome` the
    canonical merge consumes — reading and cross-checking that shard's artifact
    only if the shard actually produced one.

    Called LAZILY from the replay generator, so an earlier position's duplicate
    pre-empts a later position's artifact readback exactly as it pre-empts a
    later position's read in the serial path.

    Every defect detected here is a BACKEND failure (`ProcessShardedAssemblyError`
    / its `ShardArtifactError` subclass), never a producer defect: the parent
    cannot know a spool is bad — only that its own transport is."""
    if not isinstance(envelope, dict) or "outcome_kind" not in envelope:
        raise ProcessShardedAssemblyError(
            f"{run_id}: shard {position} returned {type(envelope).__name__}, "
            f"not a worker outcome envelope")
    kind = envelope["outcome_kind"]

    if kind == OUTCOME_READ_ERROR:
        descriptor = envelope.get("descriptor")
        if not isinstance(descriptor, CapturedSpoolReadError):
            raise ProcessShardedAssemblyError(
                f"{run_id}: shard {position} read_error carries "
                f"{type(descriptor).__name__}, not a CapturedSpoolReadError")
        if descriptor.error_code not in CANONICAL_SPOOL_READ_ERRORS:
            raise ProcessShardedAssemblyError(
                f"{run_id}: shard {position} claims canonical defect "
                f"{descriptor.error_code!r}, which is not allowlisted — a "
                f"backend failure must never be replayed as a producer defect")
        return descriptor

    if kind != OUTCOME_PROJECTION:
        raise ProcessShardedAssemblyError(
            f"{run_id}: shard {position} returned unknown outcome_kind "
            f"{kind!r}")

    result = envelope.get("result")
    if not isinstance(result, dict) or tuple(result) != _RESULT_KEYS:
        raise ProcessShardedAssemblyError(
            f"{run_id}: shard {position} projection outcome does not carry the "
            f"compact result contract {list(_RESULT_KEYS)}")

    identity = _identity_from(manifest, run_id)
    path = result["artifact_path"]
    actual_sha = _sha256_file(path)
    if actual_sha != result["artifact_sha256"]:
        raise ShardArtifactError(
            f"{run_id}: shard {position} artifact {path} sha256 {actual_sha} "
            f"!= worker-reported {result['artifact_sha256']}")
    projection, artifact_identity = read_projection_artifact(path)
    _crosscheck(position, result, identity, artifact_identity, run_id)
    if projection.survivor_count != result["survivor_count"]:
        raise ShardArtifactError(
            f"{run_id}: shard {position} survivor_count "
            f"{projection.survivor_count} != worker-reported "
            f"{result['survivor_count']}")
    return projection


def _replay_outcomes(run_id: str, manifests: List[Dict[str, Any]],
                     metas: List[Dict[str, Any]], order: List[int],
                     outcomes: List[Any],
                     failures: List[Optional[BaseException]]):
    """THE DETERMINISTIC REPLAY [REV2 §6]. The ONLY place a worker outcome is
    observed.

    Workers filled `outcomes` concurrently, in whatever order they finished.
    This generator walks `order` — position 0, then 1, then 2 — and yields each
    position's outcome to the canonical merge, which either merges it or
    re-raises its captured read error THERE. So the observed exception is a
    function of `order` alone; who finished first cannot change it, and a
    later-position defect can never pre-empt an earlier-position one."""
    for position, i in enumerate(order):
        failure = failures[position]
        if failure is not None:
            manifest = manifests[i]
            raise ProcessShardedAssemblyError(
                f"{run_id}: shard {position} "
                f"({manifest.get('stripe_id')}/sub{manifest.get('sub_index')}) "
                f"failed in the process pool with "
                f"{type(failure).__name__}: {failure}") from failure
        yield (manifests[i], metas[i],
               _materialize_outcome(run_id, position, manifests[i],
                                    outcomes[position]))


def run_sharded_assembly(
    run_id: str,
    manifests: List[Dict[str, Any]],
    pool_size: int,
    *,
    start_method: str = "spawn",
    sample_interval_ms: int = SAMPLE_INTERVAL_MS,
) -> ShardedAssemblyOutcome:
    """Assemble one trial with spool validation fanned out across `pool_size`
    CPU-only processes, and the global merge performed once in this process.

    `pool_size` is REQUIRED and explicit — there is no `os.cpu_count()` default
    (§6.7.A prohibition 4).

    Raises exactly what the serial path raises, at the same positions:
    AssemblyStateError / PhaseIdentityError / AssemblyConsistencyError and the
    encoding ValueError from the gauntlet BEFORE any worker is dispatched;
    SpoolIdentityError replayed from a worker's captured descriptor at that
    shard's own position; DirectionalDuplicateError from the shared merge. The
    exception class, `.args`, rendered message and custom attribution are
    equivalent to serial's; the traceback is not (REV2 §3).

    Plus ProcessShardedAssemblyError (and its ShardArtifactError subclass) for a
    BACKEND failure — a crashed worker, a broken pool, a malformed outcome, an
    unreadable or mismatched artifact. Those are structurally distinct from
    every producer defect and never masquerade as one."""
    if not isinstance(pool_size, int) or isinstance(pool_size, bool) \
            or pool_size < 1:
        raise ValueError(
            f"process_sharded pool_size must be an int >= 1, got "
            f"{pool_size!r}. It is always explicit: sizing the pool from "
            f"os.cpu_count() is a §6.7.A prohibition.")
    context = _resolve_context(start_method)
    started = time.perf_counter()

    # ---- 1. the FULL metadata gauntlet, BEFORE dispatching any worker -------
    # Identical function, identical order, identical exceptions as the serial
    # path — so a PhaseIdentityError / AssemblyConsistencyError still pre-empts
    # every SpoolIdentityError (G-PRECEDENCE). This is also why the gauntlet was
    # extracted rather than copied: there is nothing here to drift.
    metas, ctx, order = prepare_trial_assembly(run_id, manifests)

    artifact_dir = tempfile.mkdtemp(prefix=f"d5_shards_{run_id}_")
    try:
        with ProcessTreeRssSampler(sample_interval_ms) as sampler:
            # ---- 2. dispatch per-spool validation ---------------------------
            # Sampling is already running: worker creation itself is inside the
            # measured region.
            tasks = [{"run_id": run_id, "manifest": manifests[i],
                      "artifact_dir": artifact_dir} for i in order]
            outcomes: List[Any] = [None] * len(tasks)
            failures: List[Optional[BaseException]] = [None] * len(tasks)

            try:
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(pool_size, len(tasks)), mp_context=context)
            except Exception as exc:                    # pragma: no cover - host
                raise ProcessShardedAssemblyError(
                    f"{run_id}: could not start a {start_method!r} pool of "
                    f"{pool_size}: {type(exc).__name__}: {exc}") from exc
            try:
                submitted: Dict[Any, int] = {}
                for position, task in enumerate(tasks):
                    try:
                        submitted[executor.submit(validate_spool_shard, task)] \
                            = position
                    except Exception as exc:            # pragma: no cover - host
                        raise ProcessShardedAssemblyError(
                            f"{run_id}: shard {position} could not be submitted "
                            f"to the pool: {type(exc).__name__}: {exc}") from exc

                # ---- 3. FILL concurrently; OBSERVE nothing ------------------
                # `as_completed` is used ONLY to store each result in its own
                # INDEXED SLOT as it arrives [REV2 §6]. It never raises a
                # canonical error and never merges a projection: a defect
                # discovered here is recorded at its position and stays silent
                # until the replay reaches it. That is what makes completion
                # order unobservable.
                for future in concurrent.futures.as_completed(submitted):
                    position = submitted[future]
                    try:
                        outcomes[position] = future.result()
                    except Exception as exc:
                        outcomes[position], failures[position] = \
                            _capture_worker_exception(exc)
            finally:
                # Every worker has finished or been joined BEFORE any outcome is
                # observed, so the replay below has no pending future to cancel
                # and no live child to terminate [REV2 §7]: cleanup after an
                # early replay failure reduces to removing the artifact
                # directory, which the outer `finally` does unconditionally —
                # without touching the primary exception.
                executor.shutdown(wait=True)

            # ---- 4 + 5. REPLAY in `order`, and merge ------------------------
            # Artifact readback, identity cross-check, captured-read-error
            # replay and the global merge are all driven by ONE lazy walk of
            # `order` (`_replay_outcomes`), feeding the SAME
            # `merge_validated_spools` the serial path calls. Within-population
            # duplicate detection, provenance attribution, prng_type_by_mode
            # last-writer-wins in loop order, intersection and enrichment all
            # happen there — never in a worker, and never through a concurrent
            # dict.
            assembly = merge_validated_spools(
                run_id, ctx,
                _replay_outcomes(run_id, manifests, metas, order, outcomes,
                                 failures),
                started)

        # sampler.__exit__ has run: workers joined AND the merge completed.
        return ShardedAssemblyOutcome(
            assembly=assembly,
            peak_rss=int(sampler.peak_rss),
            rss_evidence=sampler.evidence(),
            pool_size=pool_size,
            start_method=context.get_start_method(),
            shard_count=len(order),
        )
    finally:
        # G-CLEANUP: every artifact is removed after success AND after every
        # failure path. The artifacts are call-scoped transport, never output.
        shutil.rmtree(artifact_dir, ignore_errors=True)
