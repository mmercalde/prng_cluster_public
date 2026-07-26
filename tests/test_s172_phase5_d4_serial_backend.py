#!/usr/bin/env python3
"""
test_s172_phase5_d4_serial_backend.py — S172 Phase-5 Deliverable D4 acceptance
harness (docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D4.md REV3 §6, G1-G8).

Subject under test: `miner/assembly_backends.py` — the two-backend assembly
interface (`ASSEMBLY_BACKENDS`, `get_assembly_backend`, `AssemblyBackend`),
the `serial_reference` implementation, and the frozen return contract
(`BackendAssemblyResult` / `AssemblyMeasurement`).

D4 is a NARROW deliverable: it wires a selectable seam over assembly logic that
already exists (D1.1's `assemble_trial`, D3's columnizer, D3.5's finalizer).
These gates therefore test DELEGATION and MEASUREMENT, and G7 proves at AST
level that the backend module reimplements none of it.

The fixture drives the REAL post-D1.0 lifecycle, reusing D1.1's harness pattern
(`tests/test_s172_phase5_d1_engine.py`): a real coordinator + real durable
ledger, real assigned stripes, real staged spool files on disk written through
`stage_inline_shard`, real `publish_attempt` -> `Phase5Sink.publish_shard`. It is
TWO-MODE (workflow phases 1/2/3/4), MULTI-STRIPE and MULTI-SUB-STRIPE.

Every expectation below is HAND-TRANSCRIBED — no oracle is imported from
`miner/assembly_backends.py` (§1.4). The 22 array names, their order, the
encoding integers and the sidecar keys are literals here exactly as D3 / D3.0 /
D3.5 transcribe them.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 tests/test_s172_phase5_d4_serial_backend.py
Exit 0 = all green. Exit 1 = a check failed (DO NOT COMMIT).
"""
import ast
import copy
import dataclasses
import hashlib
import importlib.util
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from miner.range_miner_coordinator import (  # noqa: E402
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    RangeMinerCoordinator,
)
from miner.range_miner_worker import (  # noqa: E402
    build_substripe_payload_bytes,
    supported_variants,
)
from miner.range_miner_npz_writer import (  # noqa: E402
    AssemblingPhase5Sink,
    MinerTrialAssembly,
    SpoolIdentityError,
    assemble_trial,
)
from utils import run_finalizer as RF  # noqa: E402

import miner.assembly_backends as AB  # noqa: E402  (the module under test)

_MODULE_PATH = os.path.join(_ROOT, "miner", "assembly_backends.py")
with open(_MODULE_PATH, "r", encoding="utf-8") as _f:
    _MODULE_SRC = _f.read()

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_results: List[Tuple[str, bool, Any]] = []
_MUTANTS: List[Tuple[str, str, str]] = []      # (mutant, red signature, attribution)


# ═════════════════════════════════════════════════════════════════════════════
# Hand-transcribed oracles — literals, never imported from the module under test
# ═════════════════════════════════════════════════════════════════════════════

# §3's frozen selector shape, written out.
ORACLE_ASSEMBLY_BACKENDS: Tuple[str, ...] = ("serial_reference", "process_sharded")

# [B1] the frozen measurement field set, in declaration order.
ORACLE_MEASUREMENT_FIELDS: Tuple[str, ...] = (
    "backend_name", "wall_seconds", "manifest_count", "spool_bytes_read",
    "survivor_row_count", "peak_rss_bytes",
)
ORACLE_RESULT_FIELDS: Tuple[str, ...] = ("assembly", "measurement")

# The twelve STABLE MinerTrialAssembly fields G3 compares for equality. `timing`
# is deliberately absent: it carries a live perf_counter delta [B2].
ORACLE_STABLE_ASSEMBLY_FIELDS: Tuple[str, ...] = (
    "run_id",
    "bidirectional_constant", "bidirectional_variable",
    "forward_map_constant", "reverse_map_constant",
    "forward_map_variable", "reverse_map_variable",
    "canonical_records_constant", "canonical_records_variable",
    "directional_counts",
    "binary_npz_path", "all_npz_path",
)
# `assemble_trial` records exactly this one timing key (writer:581).
ORACLE_TIMING_KEYS: Tuple[str, ...] = ("assembly_s",)

# The frozen 22 arrays, in the frozen ORDER, with the frozen dtypes — as D3.5's
# harness transcribes them from convert_survivors_to_binary.
ORACLE_ARRAYS: Tuple[Tuple[str, str], ...] = (
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
ORACLE_ARRAY_NAMES = tuple(name for name, _ in ORACLE_ARRAYS)

# Identity encodings — INTEGER LITERALS (D3.0 / D3 / D3.5 transcribe the same).
ORACLE_JAVA_LCG_ID = 0
ORACLE_JAVA_LCG_HYBRID_ID = 1
ORACLE_SKIP_CONSTANT_ID = 0
ORACLE_SKIP_VARIABLE_ID = 1

# D3.5 §7.1 layout + the 32-key sidecar (23 of §7.3 + the nine Seed-Domain v1.1
# stratum fields of D3.5-B). G5 asserts PRESENCE and the nine stratum values; it
# does NOT re-derive what D3.5-B's S1-S9 already gate.
ORACLE_ACCUM_DIR = ".s172_accumulator"
ORACLE_GENERATIONS = "generations"
ORACLE_CURRENT = "current"
ORACLE_SIDECAR_NAME = "provenance.json"
ORACLE_ALL_NPZ = "bidirectional_survivors_all.npz"
ORACLE_BINARY_NPZ = "bidirectional_survivors_binary.npz"

ORACLE_SIDECAR_KEY_ORDER: Tuple[str, ...] = (
    "artifact_schema_version", "artifact_sha256", "canonical_map_hash",
    "created_at", "encoding_contract_version", "exhaustive_over",
    "external_seed_transform", "final_row_count", "generation_id",
    "l2_winner_count", "parent_artifact_sha256", "parent_generation_id",
    "parent_sidecar_sha256", "prior_row_count", "prng_base",
    "raw_candidate_count", "repository_commit", "repository_tree_clean",
    "row_count", "run_id", "seed_count", "seed_domain_contract",
    "seed_domain_end_exclusive", "seed_domain_start", "seed_effective_bits",
    "seed_end_exclusive", "seed_high16_prefix", "seed_semantics", "seed_start",
    "seed_storage_dtype", "sidecar_schema_version", "skip_modes_executed",
)

# The nine frozen Seed-Domain v1.1 stratum fields, hand-transcribed.
ORACLE_SEED_DOMAIN: Tuple[Tuple[str, object], ...] = (
    ("seed_semantics",            "internal_state"),
    ("seed_storage_dtype",        "uint32"),
    ("seed_effective_bits",       32),
    ("seed_high16_prefix",        0),
    ("seed_domain_contract",      "v1.1-stratum"),
    ("seed_domain_start",         0),
    ("seed_domain_end_exclusive", 4294967296),
    ("exhaustive_over",           "high16=0 stratum only"),
    ("external_seed_transform",   None),
)


# ═════════════════════════════════════════════════════════════════════════════
# Real-lifecycle fixture — D1.1's harness pattern (tests/..._d1_engine.py:226+)
# ═════════════════════════════════════════════════════════════════════════════

SPOOL_ROOT = "/var/spool/miner"
PRNG_BASE = "java_lcg"

TOTAL_SEEDS = 40
MACRO_SIZE = 20
SUB_CAP = 10
CAPS = {"amd": SUB_CAP, "nvidia": SUB_CAP,
        "amd_hybrid": SUB_CAP, "nvidia_hybrid": SUB_CAP}

# §6.8 phase -> (family_name, direction, skip_mode, prng_type)
PHASE_TABLE = {
    1: ("java_lcg",                "forward", "constant", "java_lcg"),
    2: ("java_lcg_reverse",        "reverse", "constant", "java_lcg"),
    3: ("java_lcg_hybrid",         "forward", "variable", "java_lcg_hybrid"),
    4: ("java_lcg_hybrid_reverse", "reverse", "variable", "java_lcg_hybrid"),
}

#   CONSTANT  F = {1, 12, 25, 33}   R = {1, 12, 26}   -> F∩R = {1, 12}
#   VARIABLE  F = {2, 15, 27}       R = {2, 7, 15, 38} -> F∩R = {2, 15}
# Deliberately ASYMMETRIC in opposite directions so a forward/reverse swap or a
# constant/variable cross-wire cannot pass the derived-field assertions.
PHASE_POP = {
    1: {1: 0.90, 12: 0.80, 25: 0.70, 33: 0.60},
    2: {1: 0.50, 12: 0.40, 26: 0.30},
    3: {2: 0.95, 15: 0.85, 27: 0.75},
    4: {2: 0.55, 7: 0.65, 15: 0.45, 38: 0.35},
}

CTX = dict(trial_number=7, window_size=5, offset=2,
           sessions=["midday", "evening"], skip_min=1, skip_max=9,
           prng_base=PRNG_BASE, forward_threshold=0.40, reverse_threshold=0.45,
           dataset_sha256="d" * 64, residue_sha256="r" * 64)


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:                                      # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


def _raises(exc, fn, *a, **kw):
    """Assert `fn` raises `exc` (that class or a subclass) and return it."""
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    except Exception as other:                                  # noqa: BLE001
        raise AssertionError(
            f"expected {exc.__name__}, got {type(other).__name__}: {other}")
    raise AssertionError(f"expected {exc.__name__}, nothing was raised")


def _coord(tmp, sink, dbname="l.db"):
    ledger = MinerLedger(os.path.join(tmp, dbname))
    cfg = CoordinatorConfig(staging_dir=os.path.join(tmp, "staging"),
                            miner_stripe_size=MACRO_SIZE,
                            seed_cap_amd=SUB_CAP, seed_cap_nvidia=SUB_CAP,
                            seed_cap_amd_hybrid=SUB_CAP,
                            seed_cap_nvidia_hybrid=SUB_CAP)
    return RangeMinerCoordinator(cfg, ledger, phase5_sink=sink)


def _register(coord, wid="hostA:gpu0"):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend="cuda",
        capabilities={"seed_caps": dict(CAPS),
                      "supported_variants": list(supported_variants())},
        node_config=node, now=100.0)


def _survivor_entries(phase, seed_start, seed_count):
    """The worker's canonical survivor tuples for one sub-stripe range."""
    pop = PHASE_POP[phase]
    variable = PHASE_TABLE[phase][2] == "variable"
    out = []
    for seed in sorted(pop):
        if not (seed_start <= seed < seed_start + seed_count):
            continue
        rate = pop[seed]
        if variable:
            out.append([seed, rate, (seed % 3) + 1, [1, 2, 3]])
        else:
            out.append([seed, rate, None, [(seed % 4) + 1]])
    return out


def _retained(sink, run_id):
    state = sink._runs.get(run_id)
    return [] if state is None else list(state.manifests.values())


def _build_run(tmp, sink, run_id):
    """Drive a REAL trial through the REAL producer surface up to (but not
    including) the terminal commit. Returns the published manifest list."""
    coord = _coord(tmp, sink)
    coord.ledger.create_trial(run_id, CTX["trial_number"], now=100.0)
    coord.ledger.set_trial_context(run_id, dict(CTX))
    conn = _register(coord)
    published: List[Dict[str, Any]] = []
    for phase in (1, 2, 3, 4):
        family = PHASE_TABLE[phase][0]
        recs = coord.assign_stripes(run_id, family, phase, TOTAL_SEEDS, [conn],
                                    stripe_prefix=f"{run_id}__p{phase}", now=100.0)
        assert len(recs) == TOTAL_SEEDS // MACRO_SIZE, recs
        for rec in recs:
            assert rec["claimed"], rec
            sid = rec["stripe_id"]
            survivors_total = 0
            for sub_index in range(rec["expected_substripes"]):
                s_start = rec["seed_start"] + sub_index * SUB_CAP
                entries = _survivor_entries(phase, s_start, SUB_CAP)
                _, pb = build_substripe_payload_bytes(
                    sid, sub_index, s_start, SUB_CAP, entries)
                size, sha = len(pb), hashlib.sha256(pb).hexdigest()
                coord.ledger.record_substripe_result(
                    run_id, sid, 0, sub_index, conn.worker_id, s_start, SUB_CAP,
                    len(entries), remote_spool_path=None, size_bytes=size,
                    sha256=sha, now=100.0)
                res = coord.stage_inline_shard(run_id, sid, 0, sub_index, s_start,
                                               SUB_CAP, entries, size, sha, now=100.0)
                assert res["status"] == "verified", res
                assert os.path.isfile(res["staged_path"]), res
                survivors_total += len(entries)
            assert coord.ledger.record_stripe_complete(
                run_id, sid, 0, conn.worker_id, rec["expected_substripes"],
                survivors_total), "StripeComplete transition failed"
            coord.finalize_stripe(run_id, sid, now=100.0)
            published = _retained(sink, run_id)
    assert len(published) == 4 * (TOTAL_SEEDS // MACRO_SIZE) * (MACRO_SIZE // SUB_CAP)
    return published


# ═════════════════════════════════════════════════════════════════════════════
# Hand-computed G5 expectations — the four L2 winners, seed-ascending
# ═════════════════════════════════════════════════════════════════════════════
#
#   constant  F∩R = {1, 12}:  seed 1  fwd 0.90 rev 0.50 -> score 0.70
#                             seed 12 fwd 0.80 rev 0.40 -> score 0.60
#   variable  F∩R = {2, 15}:  seed 2  fwd 0.95 rev 0.55 -> score 0.75
#                             seed 15 fwd 0.85 rev 0.45 -> score 0.65
#
#   Every seed is distinct, so L2 (one winner per seed) keeps all four; the
#   finalizer's global seed-ascending order is therefore 1, 2, 12, 15 — an
#   INTERLEAVING of the two modes, which is exactly what makes a per-mode
#   concatenation or a mode-major order visible here.
#
#   Derived-field literals (D1.1's hand-computed tables):
#     constant  |F|=4 |R|=3 |F∩R|=2 |F∪R|=5   variable |F|=3 |R|=4 |F∩R|=2 |F∪R|=5
_C = {"forward_count": 4.0, "reverse_count": 3.0, "forward_only_count": 2.0,
      "reverse_only_count": 1.0, "survivor_overlap_ratio": 2 / 4,
      "bidirectional_selectivity": 4 / 3}
_V = {"forward_count": 3.0, "reverse_count": 4.0, "forward_only_count": 1.0,
      "reverse_only_count": 2.0, "survivor_overlap_ratio": 2 / 3,
      "bidirectional_selectivity": 3 / 4}
_ROW_MODES = (_C, _V, _C, _V)                   # seeds 1(c), 2(v), 12(c), 15(v)

ORACLE_G5_ROWS = 4
ORACLE_G5_EXPECTED: Dict[str, List[Any]] = {
    "seeds":                     [1, 2, 12, 15],
    "forward_matches":           [0.90, 0.95, 0.80, 0.85],
    "reverse_matches":           [0.50, 0.55, 0.40, 0.45],
    "score":                     [0.70, 0.75, 0.60, 0.65],
    "window_size":               [5, 5, 5, 5],
    "offset":                    [2, 2, 2, 2],
    "trial_number":              [7, 7, 7, 7],
    "skip_min":                  [1, 1, 1, 1],
    "skip_max":                  [9, 9, 9, 9],
    "skip_range":                [8, 8, 8, 8],
    "bidirectional_count":       [2.0, 2.0, 2.0, 2.0],
    "intersection_count":        [2.0, 2.0, 2.0, 2.0],
    "intersection_ratio":        [2 / 5, 2 / 5, 2 / 5, 2 / 5],
    "intersection_weight":       [2 / 7, 2 / 7, 2 / 7, 2 / 7],
    "forward_count":             [m["forward_count"] for m in _ROW_MODES],
    "reverse_count":             [m["reverse_count"] for m in _ROW_MODES],
    "forward_only_count":        [m["forward_only_count"] for m in _ROW_MODES],
    "reverse_only_count":        [m["reverse_only_count"] for m in _ROW_MODES],
    "survivor_overlap_ratio":    [m["survivor_overlap_ratio"] for m in _ROW_MODES],
    "bidirectional_selectivity": [m["bidirectional_selectivity"] for m in _ROW_MODES],
    "skip_mode":                 [ORACLE_SKIP_CONSTANT_ID, ORACLE_SKIP_VARIABLE_ID,
                                  ORACLE_SKIP_CONSTANT_ID, ORACLE_SKIP_VARIABLE_ID],
    "prng_type":                 [ORACLE_JAVA_LCG_ID, ORACLE_JAVA_LCG_HYBRID_ID,
                                  ORACLE_JAVA_LCG_ID, ORACLE_JAVA_LCG_HYBRID_ID],
}
assert set(ORACLE_G5_EXPECTED) == set(ORACLE_ARRAY_NAMES), "G5 oracle is incomplete"


# ═════════════════════════════════════════════════════════════════════════════
# G1 — the selector shape is frozen
# ═════════════════════════════════════════════════════════════════════════════
def g1_selector_shape():
    assert isinstance(AB.ASSEMBLY_BACKENDS, tuple), type(AB.ASSEMBLY_BACKENDS)
    assert AB.ASSEMBLY_BACKENDS == ORACLE_ASSEMBLY_BACKENDS, AB.ASSEMBLY_BACKENDS
    # exactly two, no extras, no omissions — asserted against the literal
    assert len(AB.ASSEMBLY_BACKENDS) == 2, AB.ASSEMBLY_BACKENDS
    assert "process_sharded" in AB.ASSEMBLY_BACKENDS, (
        "process_sharded must be DECLARED in D4 so D5 changes no interface")
    assert "serial_reference" in AB.ASSEMBLY_BACKENDS

    # the frozen return contract [B1]: two immutable dataclasses, exact fields
    for cls, expect in ((AB.AssemblyMeasurement, ORACLE_MEASUREMENT_FIELDS),
                        (AB.BackendAssemblyResult, ORACLE_RESULT_FIELDS)):
        assert dataclasses.is_dataclass(cls), cls
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} must be frozen"
        got = tuple(f.name for f in dataclasses.fields(cls))
        assert got == expect, (cls.__name__, got, expect)

    # the Protocol declares the one method; the shipped backend structurally
    # conforms to it, which is the seam D5 plugs into
    assert hasattr(AB, "AssemblyBackend")
    assert hasattr(AB.AssemblyBackend, "assemble")
    backend = AB.get_assembly_backend("serial_reference")
    assert isinstance(backend, AB.AssemblyBackend), (
        "SerialReferenceBackend does not satisfy the declared AssemblyBackend "
        "protocol")
    assert backend.backend_name == "serial_reference", backend.backend_name

    # [B4] the declared input domain is List[Dict[str, Any]] — exactly what
    # assemble_trial declares and enforces; a backend must not widen it
    import inspect
    hints = inspect.signature(type(backend).assemble)
    assert list(hints.parameters) == ["self", "run_id", "manifests"], hints
    annotation = str(hints.parameters["manifests"].annotation)
    assert "List" in annotation and "Dict" in annotation, (
        f"manifests must be List[Dict[str, Any]], not a widened mapping "
        f"domain: {annotation}")
    assert "Sequence" not in annotation and "Mapping" not in annotation, annotation


# ═════════════════════════════════════════════════════════════════════════════
# G2 — resolution FAILS CLOSED; nothing silently becomes serial_reference
# ═════════════════════════════════════════════════════════════════════════════
def g2_fail_closed_resolution():
    _g2_probe(AB.get_assembly_backend)


def _g2_probe(resolve):
    """G2's assertions, factored so G8's mutants can be driven through them."""
    for bad in ("", None, "serial", "SERIAL_REFERENCE", "serial_reference ",
                "process_shard", "default", 0, 1, True, [], {},
                b"serial_reference"):
        exc = _raises(ValueError, resolve, bad)
        assert not isinstance(exc, NotImplementedError), (
            f"{bad!r}: unknown name must be ValueError, not the D5 marker")

    # No path may hand back the serial backend for a name that is not exactly
    # "serial_reference": prove it by ATTEMPTING the resolution and requiring a
    # raise, then separately proving the valid name does resolve.
    for bad in ("", None, "serial", "process_shard"):
        try:
            got = resolve(bad)
        except (ValueError, NotImplementedError):
            continue
        raise AssertionError(
            f"{bad!r} resolved to {got!r} instead of failing closed — an "
            f"unknown/empty name must never default to serial_reference")

    backend = resolve("serial_reference")
    assert backend.backend_name == "serial_reference", backend.backend_name
    assert hasattr(backend, "assemble")


# ═════════════════════════════════════════════════════════════════════════════
# G3 — delegation identity: the wrapper returns assemble_trial's own object
# ═════════════════════════════════════════════════════════════════════════════
def g3_delegation_identity():
    with tempfile.TemporaryDirectory() as tmp:
        manifests = _build_run(tmp, AssemblingPhase5Sink(), "g3")
        _g3_probe(AB, AB.get_assembly_backend("serial_reference"), manifests)


def _g3_probe(module, backend, manifests):
    """G3's assertions, factored so G8's mutants can be driven through them.

    `module` is the module the backend came from. A mutant is loaded as its own
    module object and therefore defines its OWN `BackendAssemblyResult` class,
    so the structural isinstance checks must be made against THAT module's
    classes. Checking them against `AB`'s would red every mutant on the type
    check before its injected defect was ever exercised — mutation evidence
    that proves nothing.
    """
    direct = assemble_trial("g3", manifests)
    result = backend.assemble("g3", manifests)

    assert isinstance(result, module.BackendAssemblyResult), type(result)
    assert isinstance(result.assembly, MinerTrialAssembly), type(result.assembly)

    # every declared field is either compared for equality or is `timing`
    declared = tuple(f.name for f in dataclasses.fields(MinerTrialAssembly))
    assert set(declared) == set(ORACLE_STABLE_ASSEMBLY_FIELDS) | {"timing"}, declared

    for field in ORACLE_STABLE_ASSEMBLY_FIELDS:
        mine, theirs = getattr(result.assembly, field), getattr(direct, field)
        assert mine == theirs, f"{field}: backend {mine!r} != direct {theirs!r}"

    # element-wise on the two record lists (order included — a re-order is the
    # exact defect G8's mutant 3 injects)
    for field in ("canonical_records_constant", "canonical_records_variable"):
        mine, theirs = getattr(result.assembly, field), getattr(direct, field)
        assert isinstance(mine, list), type(mine)
        assert len(mine) == len(theirs), (field, len(mine), len(theirs))
        for i, (a, b) in enumerate(zip(mine, theirs)):
            assert a == b, f"{field}[{i}]: {a!r} != {b!r}"
        assert [r["seed"] for r in mine] == [r["seed"] for r in theirs], field

    # D3.5 Ruling E — no backend populates either NPZ path
    assert result.assembly.binary_npz_path is None, result.assembly.binary_npz_path
    assert result.assembly.all_npz_path is None, result.assembly.all_npz_path

    # [B2] timing carries a LIVE perf_counter delta, so equality is impossible.
    # Assert only structure, finiteness, positivity — and that the backend added
    # no key of its own.
    for label, obj in (("backend", result.assembly), ("direct", direct)):
        timing = obj.timing
        assert isinstance(timing, dict), (label, type(timing))
        assert "assembly_s" in timing, (label, timing)
        value = timing["assembly_s"]
        assert isinstance(value, float), (label, type(value))
        assert math.isfinite(value), (label, value)
        assert value > 0, (label, value)
    assert tuple(sorted(result.assembly.timing)) == ORACLE_TIMING_KEYS, (
        f"the backend inserted a backend-specific key into assembly.timing: "
        f"{sorted(result.assembly.timing)} != {list(ORACLE_TIMING_KEYS)}")
    assert set(result.assembly.timing) == set(direct.timing), (
        result.assembly.timing, direct.timing)


# ═════════════════════════════════════════════════════════════════════════════
# G4 — process_sharded is declared, unimplemented, and never degrades to serial
# ═════════════════════════════════════════════════════════════════════════════
def g4_process_sharded_unimplemented():
    _g4_probe(AB.get_assembly_backend)


def _g4_probe(resolve):
    exc = _raises(NotImplementedError, resolve, "process_sharded")
    text = str(exc)
    assert "D5" in text, f"the message must name D5: {text!r}"
    assert "process_sharded" in text, text
    # it must NOT fall back to serial: a returned backend at all is the failure,
    # which `_raises` above already establishes. Belt-and-braces on the type:
    assert not isinstance(exc, ValueError), (
        "process_sharded is declared-but-unimplemented (NotImplementedError), "
        "not an unknown name (ValueError)")


# ═════════════════════════════════════════════════════════════════════════════
# G5 — end to end: backend -> D3.5 finalizer -> a published generation
# ═════════════════════════════════════════════════════════════════════════════
def g5_end_to_end_through_finalizer():
    with tempfile.TemporaryDirectory() as tmp:
        manifests = _build_run(tmp, AssemblingPhase5Sink(), "g5")
        backend = AB.get_assembly_backend("serial_reference")
        result = backend.assemble("g5", manifests)

        candidates = (list(result.assembly.canonical_records_constant)
                      + list(result.assembly.canonical_records_variable))
        assert len(candidates) == ORACLE_G5_ROWS, len(candidates)

        root = Path(tmp) / "accum"
        root.mkdir()
        published = RF.finalize_run(
            candidates,
            output_root=root,
            run_id="g5_java_lcg_0",
            prng_base=PRNG_BASE,
            skip_modes_executed=("constant", "variable"),
            seed_start=0,
            seed_count=TOTAL_SEEDS,
            repository_commit="a" * 40,
            repository_tree_clean=True,
        )

        # --- a generation was published and the pointer commits to it ---------
        gen_dir = Path(published.generation_dir)
        assert gen_dir.is_dir(), gen_dir
        pointer = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
        assert os.path.islink(pointer), f"{pointer} is not a pointer symlink"
        assert Path(os.path.realpath(pointer)) == gen_dir.resolve(), (
            os.path.realpath(pointer), gen_dir)
        assert (gen_dir / ORACLE_ALL_NPZ).is_file(), sorted(os.listdir(gen_dir))
        assert (gen_dir / ORACLE_BINARY_NPZ).is_file(), sorted(os.listdir(gen_dir))
        assert (gen_dir / ORACLE_SIDECAR_NAME).is_file(), sorted(os.listdir(gen_dir))

        # --- the 22 arrays match the HAND-COMPUTED expectations ---------------
        with np.load(gen_dir / ORACLE_ALL_NPZ) as handle:
            order = tuple(handle.files)
            arrays = {name: handle[name] for name in order}
        assert order == ORACLE_ARRAY_NAMES, order
        for name, dtype in ORACLE_ARRAYS:
            got = arrays[name]
            assert got.dtype == np.dtype(dtype), (name, got.dtype, dtype)
            assert got.shape == (ORACLE_G5_ROWS,), (name, got.shape)
            expect = np.asarray(ORACLE_G5_EXPECTED[name], dtype=np.dtype(dtype))
            assert np.array_equal(got, expect), (name, got.tolist(), expect.tolist())

        # --- the sidecar records this run's coverage ---------------------------
        import json
        with open(gen_dir / ORACLE_SIDECAR_NAME, "r", encoding="utf-8") as fh:
            sidecar = json.load(fh)
        assert set(sidecar) == set(ORACLE_SIDECAR_KEY_ORDER), (
            f"sidecar key set drift: extra={sorted(set(sidecar) - set(ORACLE_SIDECAR_KEY_ORDER))} "
            f"missing={sorted(set(ORACLE_SIDECAR_KEY_ORDER) - set(sidecar))}")
        assert len(ORACLE_SIDECAR_KEY_ORDER) == 32, len(ORACLE_SIDECAR_KEY_ORDER)
        assert sidecar["run_id"] == "g5_java_lcg_0", sidecar["run_id"]
        assert sidecar["prng_base"] == PRNG_BASE, sidecar["prng_base"]
        assert sidecar["seed_start"] == 0, sidecar["seed_start"]
        assert sidecar["seed_count"] == TOTAL_SEEDS, sidecar["seed_count"]
        assert sidecar["seed_end_exclusive"] == TOTAL_SEEDS, sidecar["seed_end_exclusive"]
        assert sidecar["skip_modes_executed"] == ["constant", "variable"], \
            sidecar["skip_modes_executed"]
        assert sidecar["raw_candidate_count"] == ORACLE_G5_ROWS, sidecar
        assert sidecar["l2_winner_count"] == ORACLE_G5_ROWS, sidecar
        assert sidecar["prior_row_count"] == 0, sidecar
        assert sidecar["final_row_count"] == ORACLE_G5_ROWS, sidecar
        assert sidecar["row_count"] == ORACLE_G5_ROWS, sidecar
        assert sidecar["parent_generation_id"] is None, sidecar
        # the nine Seed-Domain v1.1 stratum fields (present + correct; D3.5-B's
        # S1-S9 own the contract itself, not re-derived here)
        for key, value in ORACLE_SEED_DOMAIN:
            assert sidecar[key] == value, (key, sidecar[key], value)

        # --- the backend itself never reached the finalizer (G7 owns the proof
        #     at source level; this is the behavioral half) -------------------
        assert result.assembly.binary_npz_path is None
        assert result.assembly.all_npz_path is None
        assert not hasattr(AB, "finalize_run"), (
            "the backend module must not import the finalizer")


# ═════════════════════════════════════════════════════════════════════════════
# G6 — the §17 measurement
# ═════════════════════════════════════════════════════════════════════════════
def g6_measurement():
    with tempfile.TemporaryDirectory() as tmp:
        manifests = _build_run(tmp, AssemblingPhase5Sink(), "g6")
        _g6_probe(AB, AB.get_assembly_backend("serial_reference"), manifests, "g6")


def _g6_probe(module, backend, manifests, run_id):
    """G6's assertions, factored so G8's mutants can be driven through them.

    `module` supplies the dataclass to type-check against — see `_g3_probe` for
    why a mutant must be checked against its own classes.
    """
    result = backend.assemble(run_id, manifests)
    m = result.measurement

    assert isinstance(m, module.AssemblyMeasurement), type(m)
    assert tuple(f.name for f in dataclasses.fields(m)) == ORACLE_MEASUREMENT_FIELDS
    # immutable by contract [B1]
    _raises(dataclasses.FrozenInstanceError, setattr, m, "wall_seconds", 1.0)

    assert m.backend_name == "serial_reference", m.backend_name

    assert isinstance(m.wall_seconds, float), type(m.wall_seconds)
    assert math.isfinite(m.wall_seconds), m.wall_seconds
    assert m.wall_seconds > 0, (
        f"wall_seconds must be a real perf_counter delta, got {m.wall_seconds!r}")

    assert m.manifest_count == len(manifests), (m.manifest_count, len(manifests))

    # INDEPENDENT oracle: the manifests' declared expected_size AND the actual
    # on-disk staged byte lengths must both equal spool_bytes_read.
    declared = sum(mf["expected_size"] for mf in manifests)
    on_disk = sum(os.path.getsize(mf["local_spool_path"]) for mf in manifests)
    assert declared == on_disk, (declared, on_disk)
    assert isinstance(m.spool_bytes_read, int), type(m.spool_bytes_read)
    assert m.spool_bytes_read == declared, (m.spool_bytes_read, declared)
    assert m.spool_bytes_read > 0, m.spool_bytes_read

    expect_rows = (len(result.assembly.canonical_records_constant)
                   + len(result.assembly.canonical_records_variable))
    assert m.survivor_row_count == expect_rows, (m.survivor_row_count, expect_rows)
    # the fixture's hand-computed intersections: 2 constant + 2 variable
    assert m.survivor_row_count == 4, m.survivor_row_count

    # §5's isolation rule: None OR a positive int. NOT the §17 number.
    if m.peak_rss_bytes is not None:
        assert isinstance(m.peak_rss_bytes, int), type(m.peak_rss_bytes)
        assert not isinstance(m.peak_rss_bytes, bool), "bool is not an RSS value"
        assert m.peak_rss_bytes > 0, m.peak_rss_bytes


# ═════════════════════════════════════════════════════════════════════════════
# G7 — no reimplementation, proved over the AST [B6]
# ═════════════════════════════════════════════════════════════════════════════
def _dotted(node: ast.AST) -> str:
    """`a.b.c` for an Attribute/Name chain; "" for anything else."""
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _called_names(tree: ast.AST) -> Tuple[set, set]:
    """(dotted call targets, trailing attribute names of every call)."""
    dotted, tails = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name:
            dotted.add(name)
            tails.add(name.rsplit(".", 1)[-1])
        elif isinstance(node.func, ast.Attribute):
            tails.add(node.func.attr)
    return dotted, tails


def g7_no_reimplementation_ast():
    tree = ast.parse(_MODULE_SRC, filename=_MODULE_PATH)
    dotted, tails = _called_names(tree)

    # 1. the module defines no assembly function of its own
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert node.name != "assemble_trial", (
                f"{_MODULE_PATH}:{node.lineno} defines assemble_trial — D4 must "
                f"CALL D1.1's, never redefine it")

    # 2. assemble_trial IS imported from the D1.1 module
    imported = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom)
                and node.module == "miner.range_miner_npz_writer"):
            for alias in node.names:
                if alias.name == "assemble_trial":
                    imported = True
    assert imported, ("assemble_trial must be imported from "
                      "miner.range_miner_npz_writer")

    # 3. forbidden CALLS — the concrete reimplementation surfaces
    forbidden_dotted = {
        "open",                                       # builtins.open
        "hashlib.sha256", "hashlib.new",
        "numpy.array", "numpy.asarray", "numpy.empty", "numpy.zeros",
        "np.array", "np.asarray", "np.empty", "np.zeros",
        "numpy.savez", "numpy.savez_compressed",
        "np.savez", "np.savez_compressed",
        "sorted",
        "records_to_arrays", "build_mode_records", "finalize_run",
    }
    hits = sorted(forbidden_dotted & dotted)
    assert not hits, f"{_MODULE_PATH} calls forbidden target(s): {hits}"

    # 4. no `.sort()` invocation, and no attribute-spelled reimplementation call
    forbidden_tails = {
        "sort", "sha256", "savez", "savez_compressed",
        "records_to_arrays", "build_mode_records", "finalize_run",
    }
    tail_hits = sorted(forbidden_tails & tails)
    assert not tail_hits, (
        f"{_MODULE_PATH} invokes forbidden attribute call(s): {tail_hits}")

    # 5. the module namespace confirms it: none of the reused helpers, and no
    #    numpy/hashlib, was pulled in at all
    for name in ("records_to_arrays", "build_mode_records", "finalize_run",
                 "np", "numpy", "hashlib", "json", "subprocess"):
        assert not hasattr(AB, name), (
            f"miner.assembly_backends must not import {name} — that layer "
            f"already exists and D4 calls it, or does not need it")
    assert hasattr(AB, "assemble_trial"), "the D1.1 entry point must be imported"


# ═════════════════════════════════════════════════════════════════════════════
# G8 — mutation proof: each mutant must RED a specific gate
# ═════════════════════════════════════════════════════════════════════════════
_MUT_SEQ = 0


def _load_mutant(src: str, label: str):
    """Import a mutated copy of the backend module under a private name."""
    global _MUT_SEQ
    _MUT_SEQ += 1
    directory = tempfile.mkdtemp(prefix="d4_mutant_")
    name = f"_d4_mutant_{_MUT_SEQ}"
    path = os.path.join(directory, f"{name}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _patch(src: str, old: str, new: str, label: str) -> str:
    """Textual mutation that must actually apply (a no-op patch would make the
    mutant vacuously survive, which would be a false green)."""
    assert src.count(old) == 1, (
        f"{label}: anchor is not unique ({src.count(old)} occurrences) — the "
        f"mutation would be unverifiable")
    return src.replace(old, new, 1)


def _record(label, detector, attribution):
    """Run a detector that MUST fail against a mutant, and record its signature."""
    try:
        detector()
    except AssertionError as exc:
        signature = str(exc).splitlines()[0][:150] or type(exc).__name__
        _MUTANTS.append((label, f"AssertionError: {signature}", attribution))
        return
    except Exception as exc:                                    # noqa: BLE001
        signature = f"{type(exc).__name__}: {str(exc).splitlines()[0][:130]}"
        _MUTANTS.append((label, signature, attribution))
        return
    raise AssertionError(f"MUTANT SURVIVED: {label} — {attribution} did not red")


def g8_mutation_proof():
    with tempfile.TemporaryDirectory() as tmp:
        manifests = _build_run(tmp, AssemblingPhase5Sink(), "g3")

        # -- M1: the resolver defaults to serial on an unknown name -----------
        src = _patch(
            _MODULE_SRC,
            "    if name not in ASSEMBLY_BACKENDS:\n"
            "        raise ValueError(",
            "    if name not in ASSEMBLY_BACKENDS:\n"
            "        return SerialReferenceBackend()\n"
            "    if False:\n"
            "        raise ValueError(",
            "M1")
        m1 = _load_mutant(src, "M1")
        _record("M1 resolver defaults to serial on unknown name",
                lambda: _g2_probe(m1.get_assembly_backend),
                "G2 fail-closed resolution")

        # -- M2: process_sharded silently resolves to serial ------------------
        src = _patch(
            _MODULE_SRC,
            "    if name == PROCESS_SHARDED:\n"
            "        raise NotImplementedError(",
            "    if name == PROCESS_SHARDED:\n"
            "        return SerialReferenceBackend()\n"
            "    if False:\n"
            "        raise NotImplementedError(",
            "M2")
        m2 = _load_mutant(src, "M2")
        _record("M2 process_sharded silently resolves to serial",
                lambda: _g4_probe(m2.get_assembly_backend),
                "G4 process_sharded declared but unimplemented")

        # -- M3: the wrapper re-orders records before returning ---------------
        src = _patch(
            _MODULE_SRC,
            "        # 3. stop the timer on successful return only\n",
            "        assembly.canonical_records_constant = list(\n"
            "            reversed(assembly.canonical_records_constant))\n"
            "        # 3. stop the timer on successful return only\n",
            "M3")
        m3 = _load_mutant(src, "M3")
        _record("M3 wrapper re-orders records before returning",
                lambda: _g3_probe(m3, m3.SerialReferenceBackend(), manifests),
                "G3 delegation identity (element-wise record comparison)")

        # -- M3b: the wrapper FILTERS records before returning ----------------
        src = _patch(
            _MODULE_SRC,
            "        # 3. stop the timer on successful return only\n",
            "        assembly.canonical_records_variable = [\n"
            "            r for r in assembly.canonical_records_variable\n"
            "            if r['score'] > 0.70]\n"
            "        # 3. stop the timer on successful return only\n",
            "M3b")
        m3b = _load_mutant(src, "M3b")
        _record("M3b wrapper filters records before returning",
                lambda: _g3_probe(m3b, m3b.SerialReferenceBackend(), manifests),
                "G3 delegation identity (element-wise record comparison)")

        # -- M4: the wrapper populates an NPZ path field ----------------------
        src = _patch(
            _MODULE_SRC,
            "        # 3. stop the timer on successful return only\n",
            "        assembly.all_npz_path = '/tmp/backend_should_not_set.npz'\n"
            "        # 3. stop the timer on successful return only\n",
            "M4")
        m4 = _load_mutant(src, "M4")
        _record("M4 wrapper populates an NPZ path field",
                lambda: _g3_probe(m4, m4.SerialReferenceBackend(), manifests),
                "G3 delegation identity (D3.5 Ruling E: both NPZ paths None)")

        # -- M4b: the wrapper injects a key into assembly.timing --------------
        src = _patch(
            _MODULE_SRC,
            "        # 3. stop the timer on successful return only\n",
            "        assembly.timing['backend_wall_s'] = 0.5\n"
            "        # 3. stop the timer on successful return only\n",
            "M4b")
        m4b = _load_mutant(src, "M4b")
        _record("M4b wrapper injects a backend key into assembly.timing",
                lambda: _g3_probe(m4b, m4b.SerialReferenceBackend(), manifests),
                "G3 delegation identity [B2] (no backend-specific timing key)")

        # -- M5: ASSEMBLY_BACKENDS drops process_sharded ----------------------
        src = _patch(
            _MODULE_SRC,
            "ASSEMBLY_BACKENDS = (SERIAL_REFERENCE, PROCESS_SHARDED)",
            "ASSEMBLY_BACKENDS = (SERIAL_REFERENCE,)",
            "M5")
        m5 = _load_mutant(src, "M5")

        def _detect_m5():
            assert m5.ASSEMBLY_BACKENDS == ORACLE_ASSEMBLY_BACKENDS, \
                m5.ASSEMBLY_BACKENDS
            assert "process_sharded" in m5.ASSEMBLY_BACKENDS, (
                "process_sharded must be DECLARED in D4 so D5 changes no "
                "interface")
        _record("M5 ASSEMBLY_BACKENDS drops process_sharded", _detect_m5,
                "G1 selector shape")

        # -- M6: measurement returns a constant instead of a real timing ------
        src = _patch(
            _MODULE_SRC,
            "            wall_seconds=wall_seconds,",
            "            wall_seconds=0.0,",
            "M6")
        m6 = _load_mutant(src, "M6")
        _record("M6 measurement returns a constant instead of a real timing",
                lambda: _g6_probe(m6, m6.SerialReferenceBackend(), manifests, "g3"),
                "G6 measurement (wall_seconds > 0)")

        # -- M6b: spool_bytes_read is a constant ------------------------------
        src = _patch(
            _MODULE_SRC,
            '            spool_bytes_read=sum(m["expected_size"] for m in manifests),',
            "            spool_bytes_read=1,",
            "M6b")
        m6b = _load_mutant(src, "M6b")
        _record("M6b spool_bytes_read is a constant, not the real byte total",
                lambda: _g6_probe(m6b, m6b.SerialReferenceBackend(), manifests, "g3"),
                "G6 measurement (independent on-disk byte oracle)")

        # -- M7 [B5]: measurement computed BEFORE delegation ------------------
        # The defect: `sum(m["expected_size"] ...)` evaluated before the
        # delegated call, so a malformed manifest raises a raw KeyError out of
        # the WRAPPER instead of D1.1's canonical SpoolIdentityError.
        src = _patch(
            _MODULE_SRC,
            "        # 2. delegate",
            '        _premeasured_bytes = sum(m["expected_size"] for m in manifests)\n'
            "        # 2. delegate",
            "M7")
        src = _patch(
            src,
            '            spool_bytes_read=sum(m["expected_size"] for m in manifests),',
            "            spool_bytes_read=_premeasured_bytes,",
            "M7-use")
        m7 = _load_mutant(src, "M7")
        _record("M7 [B5] measurement computed BEFORE delegation",
                lambda: _b5_probe(m7.SerialReferenceBackend(), manifests),
                "G8/[B5] canonical D1.1 exception survives unchanged")

        # the same probe must PASS against the real module — the ordering in
        # miner/assembly_backends.py is what makes it pass.
        _b5_probe(AB.get_assembly_backend("serial_reference"), manifests)


def _b5_probe(backend, manifests):
    """[B5]: a manifest missing `expected_size` must raise D1.1's canonical
    fail-closed spool error, NOT a KeyError from the wrapper's measurement.

    `expected_size` is read ONLY inside D1.1's spool wall
    (`range_miner_npz_writer.py:360`), so deleting it reaches that wall and
    produces `SpoolIdentityError: ... size N != expected_size None`.
    """
    broken = copy.deepcopy(manifests)
    assert "expected_size" in broken[0], broken[0].keys()
    del broken[0]["expected_size"]

    exc = _raises(SpoolIdentityError, backend.assemble, "g3", broken)
    text = str(exc)
    # D1.1's message, not the wrapper's
    assert "expected_size" in text, text
    assert "!=" in text, f"not D1.1's size-mismatch wording: {text!r}"
    # and the exception really is the assembler's, carrying its run/stripe context
    assert "g3" in text, text
    assert not isinstance(exc, KeyError), type(exc)


# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 74)
    print("S172 Phase-5 D4 gate — serial_reference behind the two-backend "
          "interface")
    print("=" * 74)
    _check("G1: selector shape + frozen return contract",  g1_selector_shape)
    _check("G2: fail-closed resolution (no silent default)",
           g2_fail_closed_resolution)
    _check("G3: delegation identity (12 stable fields + [B2] timing)",
           g3_delegation_identity)
    _check("G4: process_sharded declared, unimplemented, names D5",
           g4_process_sharded_unimplemented)
    _check("G5: end to end backend -> D3.5 finalizer -> published generation",
           g5_end_to_end_through_finalizer)
    _check("G6: §17 measurement fields + semantics",       g6_measurement)
    _check("G7: no reimplementation (AST) [B6]",           g7_no_reimplementation_ast)
    _check("G8: mutation proof (9 mutants)",               g8_mutation_proof)
    print("=" * 74)

    if _MUTANTS:
        print("\nMUTATION EVIDENCE — every mutant RED, with attribution:\n")
        for label, signature, attribution in _MUTANTS:
            print(f"  {label}")
            print(f"      red in : {attribution}")
            print(f"      signature: {signature}")
        print()

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} D4 gate checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All D4 gate checks green — `serial_reference` is a thin, selectable, "
          "measured wrapper over the shared D1.1 assembly path behind a frozen "
          "two-backend interface that fails closed and reimplements nothing "
          "(pending Team Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
