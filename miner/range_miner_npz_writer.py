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
from typing import Any, Dict, List, Optional, Tuple

from miner.range_miner_coordinator import (
    MinerMetadataError,
    Phase5Sink,
    _canonicalize_trial_context,
    validate_trial_metadata,
    workflow_phase_semantics,
    workflow_stages_for,
)
from miner.range_miner_worker import SUBSTRIPE_SCHEMA_VERSION
from utils.prng_encoding import encode_prng_type, encode_skip_mode

__all__ = [
    "CANONICAL_RECORD_FIELDS",
    "MinerTrialAssembly",
    "AssemblingPhase5Sink",
    "assemble_trial",
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
# Order and membership reproduce the LIVE Step-1 insertion order exactly:
# seed/rates/score followed by `metadata_base`
# (window_optimizer_integration_final.py:683-694 + :652-676 for constant;
# identically :785-796 + :756-780 for the hybrid/variable block).
#
# `threshold_used` is deliberately NOT a 25th field: it is manifest identity /
# validation metadata (§5.1), not a record field.
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
# §5.3 — spool read + identity + container + semantic validation.
# ---------------------------------------------------------------------------
def _read_and_validate_spool(run_id: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Read this manifest's staged bytes at COMMIT time and return the validated
    payload. Everything that can go wrong — I/O, size, SHA, JSON decode, the
    container shape, the schema/stripe/sub_index identity, and the per-survivor
    semantics — becomes SpoolIdentityError; no raw TypeError/KeyError/quirk may
    escape [TB-D1-PV]."""
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
    return payload


# ---------------------------------------------------------------------------
# §5.5/§6 — derived fields + canonical records for ONE mode.
# ---------------------------------------------------------------------------
def _mode_records(
    fwd_map: Dict[int, float], rev_map: Dict[int, float],
    ctx: Dict[str, Any], skip_mode: str, prng_type: Optional[str],
) -> Tuple[set, List[Dict[str, Any]]]:
    """Intersection + the frozen 24-field canonical records for one skip mode.

    Every derived field is frozen from the live constant block
    (window_optimizer_integration_final.py:652-694) and variable block
    (:756-796), including the `max(..., 1)` denominators and the deliberate
    duplication of `bidirectional_count` / `intersection_count`."""
    fwd_set, rev_set = set(fwd_map), set(rev_map)
    both = fwd_set & rev_set
    if prng_type is None:                       # the mode did not run (§5.4)
        return both, []
    union = len(fwd_set | rev_set)
    metadata_base = {
        "window_size":               ctx["window_size"],
        "offset":                    ctx["offset"],
        "skip_min":                  ctx["skip_min"],
        "skip_max":                  ctx["skip_max"],
        "skip_range":                ctx["skip_max"] - ctx["skip_min"],
        "sessions":                  ctx["sessions"],
        "trial_number":              ctx["trial_number"],
        "prng_base":                 ctx["prng_base"],
        "skip_mode":                 skip_mode,
        "prng_type":                 prng_type,
        "forward_count":             len(fwd_map),
        "reverse_count":             len(rev_map),
        "bidirectional_count":       len(both),
        "intersection_count":        len(both),
        "intersection_ratio":        len(both) / max(union, 1),
        "forward_only_count":        len(fwd_set - rev_set),
        "reverse_only_count":        len(rev_set - fwd_set),
        "survivor_overlap_ratio":    len(both) / max(len(fwd_set), 1),
        "bidirectional_selectivity": len(fwd_set) / max(len(rev_set), 1),
        "intersection_weight":       len(both) / max(len(fwd_set) + len(rev_set), 1),
    }
    records = []
    for seed in sorted(both):                   # ascending seed order (§6)
        fwd_rate, rev_rate = fwd_map[seed], rev_map[seed]
        record = {
            "seed":               seed,
            "forward_match_rate": fwd_rate,
            "reverse_match_rate": rev_rate,
            "score":              (fwd_rate + rev_rate) / 2.0,
        }
        record.update(metadata_base)
        # The frozen 24 keys, in the frozen order — gate-enforced (G9). Rebuilt
        # explicitly rather than trusted from insertion order.
        records.append({k: record[k] for k in CANONICAL_RECORD_FIELDS})
    return both, records


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
    (the canonical module's own hard-fail is deliberately not re-wrapped)."""
    started = time.perf_counter()
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

    # ---- §5.3 + §5.4 spool read -> directional maps ------------------------
    maps: Dict[Tuple[str, str], Dict[int, float]] = {p: {} for p in _POPULATIONS}
    prov: Dict[Tuple[str, str], Dict[int, Tuple[str, int, int, float]]] = {
        p: {} for p in _POPULATIONS
    }
    prng_type_by_mode: Dict[str, str] = {}
    # Deterministic order so a duplicate is always reported against the same
    # "first" insertion regardless of manifest arrival order.
    order = sorted(
        range(len(manifests)),
        key=lambda i: (int(metas[i]["workflow_phase"]),
                       str(manifests[i].get("stripe_id")),
                       int(manifests[i].get("sub_index", 0)),
                       int(manifests[i].get("attempt", 0)),
                       str(manifests[i].get("event_id"))))
    for i in order:
        manifest, meta = manifests[i], metas[i]
        direction, skip_mode = meta["direction"], meta["skip_mode"]
        prng_type_by_mode[skip_mode] = meta["prng_type"]
        payload = _read_and_validate_spool(run_id, manifest)
        pop_map = maps[(direction, skip_mode)]
        pop_prov = prov[(direction, skip_mode)]
        stripe_id = manifest.get("stripe_id")
        sub_index = int(manifest.get("sub_index", 0))
        attempt = int(manifest.get("attempt", 0))
        for entry in payload["survivors"]:
            seed, match_rate = int(entry[0]), float(entry[1])
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
