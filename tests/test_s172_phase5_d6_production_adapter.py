#!/usr/bin/env python3
"""
test_s172_phase5_d6_production_adapter.py — S172 Phase 5, Deliverable D6:
the 3.A adapter gates.

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D6.md (REV1) §3.A, frozen
against HEAD 2a6e0f8.

The seven gates, verbatim from §3.A:

  G-INGRESS          a stored MinerTrialAssembly with known canonical_records_*
                     appends exactly those candidates into the accumulator, with
                     real forward / reverse / bidirectional counts — and the
                     appended records are IDENTICAL to the assembly's (no
                     re-normalization, byte-for-byte record equality).
  G-NO-PWZ-INGRESS   the miner path never calls the PWC/ZMQ D3.25 ingress /
                     four-map normalizer (AST + runtime).
  G-FINALIZE         the stored assembly drives finalize_run to a certified
                     generation; RunArtifactResult paths exist and point at the
                     22-array bundle; MinerTrialAssembly.binary_npz_path /
                     all_npz_path remain None.
  G-FAILCLOSED       when a required publication result is absent (assembly
                     None, or a required path None), D6 RAISES — it never
                     returns a zero/empty TestResult.
  G-TESTRESULT       the returned TestResult matches the Step-1 contract fields;
                     the PWC/ZMQ builders are untouched (AST).
  G-BACKEND-DEFAULT  with no backend specified, serial_reference is selected;
                     process_sharded is reachable only by explicit selection.
  G-FLUSH-CADENCE    the threshold-gated flush fires on the same cadence as the
                     pre-D6 path.

House rules honoured here:
  * oracles are HAND-TRANSCRIBED literals, never imported from a module under
    test;
  * fixtures drive the REAL producer surface (coordinator -> ledger -> staging
    -> Phase-5 sink), never a hand-built manifest;
  * every gate is mutation-proved under the four-part rule (applies-once,
    mutated-path-executed, detector-clean-unmutated, injected-defect).
"""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
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
)
from utils import run_finalizer as RF  # noqa: E402
from window_optimizer import TestResult, WindowConfig  # noqa: E402

import miner.assembly_backends as AB            # noqa: E402
import miner.range_miner_npz_writer as RMW      # noqa: E402
import miner.step1_ingress as SI                # noqa: E402
import window_optimizer_integration_final as WOI  # noqa: E402

_INGRESS_PATH = os.path.join(_ROOT, "miner", "step1_ingress.py")
_INTEG_PATH = os.path.join(_ROOT, "window_optimizer_integration_final.py")
_WRITER_PATH = os.path.join(_ROOT, "miner", "range_miner_npz_writer.py")
with open(_INGRESS_PATH, "r", encoding="utf-8") as _f:
    _INGRESS_SRC = _f.read()
with open(_INTEG_PATH, "r", encoding="utf-8") as _f:
    _INTEG_SRC = _f.read()
with open(_WRITER_PATH, "r", encoding="utf-8") as _f:
    _WRITER_SRC = _f.read()

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_results: List[Tuple[str, bool, Any]] = []
_MUTANTS: List[Tuple[str, str, str, str]] = []


# ═════════════════════════════════════════════════════════════════════════════
# Hand-transcribed oracles — literals, never imported from a module under test
# ═════════════════════════════════════════════════════════════════════════════

# The Step-1 TestResult contract fields (window_optimizer.py:232-236), written
# out. D6 may not add, drop or rename one.
ORACLE_TESTRESULT_FIELDS: Tuple[str, ...] = (
    "config", "forward_count", "reverse_count", "bidirectional_count",
    "iteration",
)

# The PWC/ZMQ D3.25 ingress surface the miner path must NEVER touch
# (utils/canonical_records.py). Transcribed by name.
ORACLE_FORBIDDEN_INGRESS: Tuple[str, ...] = (
    "normalize_trial_populations",
    "validate_trial_populations",
    "build_mode_records",
)

# The frozen 22 array names, in the frozen ORDER (D3 / D3.5 / D4 / D5 all
# transcribe the same list).
ORACLE_ARRAY_NAMES: Tuple[str, ...] = (
    "seeds", "forward_matches", "reverse_matches", "window_size", "offset",
    "trial_number", "skip_min", "skip_max", "skip_range", "forward_count",
    "reverse_count", "bidirectional_count", "intersection_count",
    "intersection_ratio", "intersection_weight", "bidirectional_selectivity",
    "forward_only_count", "reverse_only_count", "survivor_overlap_ratio",
    "score", "skip_mode", "prng_type",
)

# The two backend names, and the production default (§17).
ORACLE_BACKENDS: Tuple[str, ...] = ("serial_reference", "process_sharded")
ORACLE_DEFAULT_BACKEND = "serial_reference"

# The six directional_counts keys D1.1 always populates
# (range_miner_npz_writer.py:942-949).
ORACLE_COUNT_KEYS: Tuple[str, ...] = (
    "forward_constant", "reverse_constant", "forward_variable",
    "reverse_variable", "bidirectional_constant", "bidirectional_variable",
)

# The four RunArtifactResult paths a certified generation must carry.
ORACLE_ARTIFACT_PATHS: Tuple[str, ...] = (
    "generation_dir", "all_npz_path", "binary_npz_path", "sidecar_path",
)

# The commit D6 is frozen against. The PWC/ZMQ builders must be BYTE-IDENTICAL
# to their text at this commit.
ORACLE_FROZEN_COMMIT = "2a6e0f8"
ORACLE_PWZ_BUILDERS: Tuple[str, ...] = ("_build_test_result_from_pw",)


# ═════════════════════════════════════════════════════════════════════════════
# Real-lifecycle fixture — the D1.1 / D4 / D5 harness pattern
# ═════════════════════════════════════════════════════════════════════════════
SPOOL_ROOT = "/var/spool/miner"
PRNG_BASE = "java_lcg"

TOTAL_SEEDS = 40
MACRO_SIZE = 20
SUB_CAP = 10

PHASE_TABLE = {
    1: ("java_lcg",                "forward", "constant", "java_lcg"),
    2: ("java_lcg_reverse",        "reverse", "constant", "java_lcg"),
    3: ("java_lcg_hybrid",         "forward", "variable", "java_lcg_hybrid"),
    4: ("java_lcg_hybrid_reverse", "reverse", "variable", "java_lcg_hybrid"),
}

#   CONSTANT  F = {1, 12, 25, 33}   R = {1, 12, 26}    -> F∩R = {1, 12}
#   VARIABLE  F = {2, 15, 27}       R = {2, 7, 15, 38} -> F∩R = {2, 15}
PHASE_POP = {
    1: {1: 0.90, 12: 0.80, 25: 0.70, 33: 0.60},
    2: {1: 0.50, 12: 0.40, 26: 0.30},
    3: {2: 0.95, 15: 0.85, 27: 0.75},
    4: {2: 0.55, 7: 0.65, 15: 0.45, 38: 0.35},
}

# HAND-COMPUTED from PHASE_POP above — the numbers D6 must reproduce.
ORACLE_FORWARD_CONSTANT = 4          # {1, 12, 25, 33}
ORACLE_REVERSE_CONSTANT = 3          # {1, 12, 26}
ORACLE_FORWARD_VARIABLE = 3          # {2, 15, 27}
ORACLE_REVERSE_VARIABLE = 4          # {2, 7, 15, 38}
ORACLE_BIDI_CONSTANT = 2             # {1, 12}
ORACLE_BIDI_VARIABLE = 2             # {2, 15}
ORACLE_FORWARD_TOTAL = 7             # 4 + 3
ORACLE_REVERSE_TOTAL = 7             # 3 + 4
ORACLE_BIDI_TOTAL = 4                # 2 + 2
ORACLE_APPENDED_ROWS = 4             # one canonical record per bidi seed/mode
ORACLE_BIDI_SEEDS_CONSTANT = (1, 12)
ORACLE_BIDI_SEEDS_VARIABLE = (2, 15)

CTX = dict(trial_number=7, window_size=5, offset=2,
           sessions=["midday", "evening"], skip_min=1, skip_max=9,
           prng_base=PRNG_BASE, forward_threshold=0.40, reverse_threshold=0.45,
           dataset_sha256="d" * 64, residue_sha256="r" * 64)

CONFIG = WindowConfig(window_size=CTX["window_size"], offset=CTX["offset"],
                      sessions=list(CTX["sessions"]),
                      skip_min=CTX["skip_min"], skip_max=CTX["skip_max"],
                      forward_threshold=CTX["forward_threshold"],
                      reverse_threshold=CTX["reverse_threshold"])


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:                                      # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


def _raises(exc, fn, *a, **kw):
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
    caps = {"amd": SUB_CAP, "nvidia": SUB_CAP,
            "amd_hybrid": SUB_CAP, "nvidia_hybrid": SUB_CAP}
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend="cuda",
        capabilities={"seed_caps": caps,
                      "supported_variants": list(supported_variants())},
        node_config=node, now=100.0)


def _survivor_entries(phase, seed_start, seed_count):
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


def _build_run(tmp, sink, run_id, phases=(1, 2, 3, 4), dbname="l.db"):
    """Drive a REAL trial through the REAL producer surface up to (but not
    including) the terminal commit. Returns (coordinator, published manifests)."""
    coord = _coord(tmp, sink, dbname)
    coord.ledger.create_trial(run_id, CTX["trial_number"], now=100.0)
    coord.ledger.set_trial_context(run_id, dict(CTX))
    conn = _register(coord)
    for phase in phases:
        family = PHASE_TABLE[phase][0]
        recs = coord.assign_stripes(run_id, family, phase, TOTAL_SEEDS, [conn],
                                    stripe_prefix=f"{run_id}__p{phase}",
                                    now=100.0)
        for rec in recs:
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
                res = coord.stage_inline_shard(run_id, sid, 0, sub_index,
                                               s_start, SUB_CAP, entries, size,
                                               sha, now=100.0)
                assert res["status"] == "verified", res
                survivors_total += len(entries)
            assert coord.ledger.record_stripe_complete(
                run_id, sid, 0, conn.worker_id, rec["expected_substripes"],
                survivors_total), "StripeComplete transition failed"
            coord.finalize_stripe(run_id, sid, now=100.0)
    return coord


def _committed_run(tmp, run_id, sink=None, phases=(1, 2, 3, 4), dbname="l.db"):
    """A REAL trial driven to a REAL sink commit. Returns (sink, assembly,
    miner_result) where `miner_result` carries the serve_trial keys D6 reads."""
    sink = WOI.build_assembling_sink() if sink is None else sink
    _build_run(tmp, sink, run_id, phases=phases, dbname=dbname)
    sink.commit_trial({"run_id": run_id, "event_id": f"{run_id}-commit"})
    assembly = sink.get_assembly(run_id)
    assert assembly is not None, "fixture failed to produce a committed assembly"
    # [S172 D6 correction] `serve_trial` now also returns the threshold-provenance
    # audit record, and only ever reaches its `committed` return after the
    # parent-side fail-closed gate passed — so a committed miner_result ALWAYS
    # carries `validated: True`. The fixture mirrors that shape; omitting it would
    # be modelling a serve result production cannot produce, and the D6 ingress
    # wall correctly refuses it (that refusal is proved by the threshold gate's
    # G13, not weakened here).
    miner_result = {"run_id": run_id, "state": "committed", "committed": True,
                    "threshold_provenance": {
                        "run_id": run_id,
                        "requested": {"forward": CTX["forward_threshold"],
                                      "reverse": CTX["reverse_threshold"]},
                        "payload": {}, "effective": {}, "phase_direction": {},
                        "validated": True}}
    return sink, assembly, miner_result


def _fresh_accumulator():
    return {"forward_count": 0, "reverse_count": 0, "bidirectional": []}


def _canon(obj):
    """The canonical comparison form — a sorted-key JSON round trip, the same
    definition D1 uses for 'canonically identical'."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


# The adapter names `window_optimizer_integration_final` imported BY VALUE. A
# mutant module must be swapped into all of them, or the production call path
# would keep using the real adapter and any red it produced would be
# unattributable (four-part rule, part 4).
_ADAPTER_NAMES = ("MinerIngressError", "build_assembling_sink",
                  "certified_paths", "ingest_assembly", "require_assembly",
                  "resolve_assembly_backend")


class _adapter:
    """Point BOTH this harness and the production module at `module`."""

    def __init__(self, module):
        self.module = module
        self.saved = {}

    def __enter__(self):
        for name in _ADAPTER_NAMES:
            self.saved[name] = getattr(WOI, name)
            setattr(WOI, name, getattr(self.module, name))
        return self.module

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(WOI, name, value)
        return False


# ═════════════════════════════════════════════════════════════════════════════
# G-INGRESS
# ═════════════════════════════════════════════════════════════════════════════
def g_ingress(module=SI):
    """Run the gate with BOTH this harness and the production module
    pointed at `module`, so a mutant really reaches the production path."""
    with _adapter(module):
        return _g_ingress_impl(module)


def _g_ingress_impl(module):
    """A stored assembly's canonical records land in the accumulator EXACTLY as
    the assembly holds them, with the real counts."""
    with tempfile.TemporaryDirectory() as tmp:
        _sink, assembly, _res = _committed_run(tmp, "ing")
        acc = _fresh_accumulator()

        # Snapshot the assembly's records BEFORE ingress, in canonical form, so
        # a mutation that rebuilds/normalizes them is visible afterwards.
        before_c = [_canon(r) for r in assembly.canonical_records_constant]
        before_v = [_canon(r) for r in assembly.canonical_records_variable]

        counts = module.ingest_assembly(assembly, acc)

        appended = acc["bidirectional"]

        # (1) exactly those candidates, constant before variable
        assert len(appended) == ORACLE_APPENDED_ROWS, (
            f"appended {len(appended)} rows, oracle says {ORACLE_APPENDED_ROWS}")
        assert [_canon(r) for r in appended] == before_c + before_v, (
            "the appended rows are not the assembly's rows, in assembly order "
            "(constant then variable)")

        # (2) byte-for-byte record equality — no re-normalization. The canonical
        #     JSON above proves value+key equality; identity proves not even a
        #     copy-through-a-builder happened.
        expected_objs = (list(assembly.canonical_records_constant)
                         + list(assembly.canonical_records_variable))
        for i, (got, want) in enumerate(zip(appended, expected_objs)):
            assert got is want, (
                f"row {i} is a REBUILT dict, not the assembly's own record — "
                f"D6 must append without re-normalizing (D3.25 REV3 §4)")
            assert list(got.keys()) == list(want.keys()), (
                f"row {i} field ORDER differs from the assembly's record")

        # (3) the seeds are the hand-computed intersections, in ascending order
        seeds_c = tuple(int(r["seed"])
                        for r in assembly.canonical_records_constant)
        seeds_v = tuple(int(r["seed"])
                        for r in assembly.canonical_records_variable)
        assert seeds_c == ORACLE_BIDI_SEEDS_CONSTANT, seeds_c
        assert seeds_v == ORACLE_BIDI_SEEDS_VARIABLE, seeds_v

        # (4) REAL counts, not +0
        assert acc["forward_count"] == ORACLE_FORWARD_TOTAL, acc["forward_count"]
        assert acc["reverse_count"] == ORACLE_REVERSE_TOTAL, acc["reverse_count"]
        assert counts.bidirectional_total == ORACLE_BIDI_TOTAL, counts
        assert counts.forward_constant == ORACLE_FORWARD_CONSTANT, counts
        assert counts.reverse_constant == ORACLE_REVERSE_CONSTANT, counts
        assert counts.appended_total == ORACLE_APPENDED_ROWS, counts

        # (5) counts ACCUMULATE across trials rather than being overwritten
        module.ingest_assembly(assembly, acc)
        assert acc["forward_count"] == 2 * ORACLE_FORWARD_TOTAL, acc
        assert acc["reverse_count"] == 2 * ORACLE_REVERSE_TOTAL, acc
        assert len(acc["bidirectional"]) == 2 * ORACLE_APPENDED_ROWS, acc

        # (6) the six-key directional contract is required, never defaulted
        broken = copy.copy(assembly)
        broken.directional_counts = {k: 0 for k in ORACLE_COUNT_KEYS[:-1]}
        _raises(module.MinerIngressError, module.ingest_assembly, broken, None)


def g_ingress_through_builder():
    """The same ingress, driven through the PRODUCTION call
    `_build_test_result_from_miner` rather than the adapter helper."""
    with tempfile.TemporaryDirectory() as tmp:
        sink, assembly, res = _committed_run(tmp, "ingb")
        acc = _fresh_accumulator()
        tr = WOI._build_test_result_from_miner(
            res, acc, CONFIG, PRNG_BASE, CTX["trial_number"], None,
            phase5_sink=sink)
        assert len(acc["bidirectional"]) == ORACLE_APPENDED_ROWS, acc
        assert acc["forward_count"] == ORACLE_FORWARD_TOTAL, acc
        assert acc["reverse_count"] == ORACLE_REVERSE_TOTAL, acc
        assert tr.bidirectional_count == ORACLE_BIDI_TOTAL, tr
        assert tr.forward_count == ORACLE_FORWARD_CONSTANT, tr
        assert tr.reverse_count == ORACLE_REVERSE_CONSTANT, tr
        assert tr.iteration == CTX["trial_number"], tr
        # the pre-D6 behaviour this replaces: everything zero, nothing appended
        assert not (tr.forward_count == 0 and tr.reverse_count == 0
                    and tr.bidirectional_count == 0
                    and not acc["bidirectional"]), (
            "the miner path is still inert (+0, no candidates) — D6 did not "
            "close the gap")


# ═════════════════════════════════════════════════════════════════════════════
# G-NO-PWZ-INGRESS
# ═════════════════════════════════════════════════════════════════════════════
def _imported_names(src: str, path: str) -> set:
    names = set()
    for node in ast.walk(ast.parse(src, filename=path)):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(a.name for a in node.names)
    return names


def _called_names(fn_node) -> set:
    out = set()
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def _func_node(src, path, name):
    tree = ast.parse(src, filename=path)
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def g_no_pwz_ingress(ingress_src=None, integ_src=None):
    """AST: neither the adapter module nor the miner builder reaches the D3.25
    PWC/ZMQ ingress. Runtime: a tripwire over that ingress is never fired by a
    full miner ingress."""
    ingress_src = _INGRESS_SRC if ingress_src is None else ingress_src
    integ_src = _INTEG_SRC if integ_src is None else integ_src

    # --- AST 1: the adapter module imports nothing from the ingress module ---
    imported = _imported_names(ingress_src, _INGRESS_PATH)
    assert "utils.canonical_records" not in imported, (
        "miner/step1_ingress.py imports utils.canonical_records — the miner is "
        "not a step1_trial_populations_v2 producer (D3.25 REV3 §4)")
    # Referenced NAMES, not raw text: the module's own "WHAT THIS MODULE IS
    # NOT" docstring names the forbidden surface on purpose, and a substring
    # check would red on that prose while missing an aliased import.
    referenced = set()
    for node in ast.walk(ast.parse(ingress_src, filename=_INGRESS_PATH)):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
    for name in ORACLE_FORBIDDEN_INGRESS:
        assert name not in imported, f"step1_ingress imports {name}"
        assert name not in referenced, (
            f"step1_ingress references {name} in code — the four-map "
            f"normalizer must be unreachable from the miner path")

    # --- AST 2: the miner builder calls neither the normalizer nor the PWC
    #            builder it is deliberately detached from ---
    fn = _func_node(integ_src, _INTEG_PATH, "_build_test_result_from_miner")
    called = _called_names(fn)
    for name in ORACLE_FORBIDDEN_INGRESS + ORACLE_PWZ_BUILDERS:
        assert name not in called, (
            f"_build_test_result_from_miner calls {name} — the miner path must "
            f"not share the PWC/ZMQ ingress")

    # --- runtime: tripwires over the real ingress surface -------------------
    import utils.canonical_records as CR
    saved = {name: getattr(CR, name) for name in ORACLE_FORBIDDEN_INGRESS
             if hasattr(CR, name)}
    tripped: List[str] = []

    def _tripwire(name):
        def _fn(*a, **kw):
            tripped.append(name)
            raise AssertionError(
                f"the miner path called the PWC/ZMQ D3.25 ingress ({name})")
        return _fn

    try:
        for name in saved:
            setattr(CR, name, _tripwire(name))
        with tempfile.TemporaryDirectory() as tmp:
            sink, _assembly, res = _committed_run(tmp, "nopwz")
            acc = _fresh_accumulator()
            WOI._build_test_result_from_miner(
                res, acc, CONFIG, PRNG_BASE, CTX["trial_number"], None,
                phase5_sink=sink)
        assert not tripped, tripped
        assert len(acc["bidirectional"]) == ORACLE_APPENDED_ROWS, (
            "the tripwire run did not actually ingest, so a clean run proves "
            "nothing")
    finally:
        for name, fn_obj in saved.items():
            setattr(CR, name, fn_obj)


# ═════════════════════════════════════════════════════════════════════════════
# G-FINALIZE
# ═════════════════════════════════════════════════════════════════════════════
def g_finalize(module=SI):
    """Run the gate with BOTH this harness and the production module
    pointed at `module`, so a mutant really reaches the production path."""
    with _adapter(module):
        return _g_finalize_impl(module)


def _g_finalize_impl(module):
    """The stored assembly, ingested into the accumulator, drives the SHARED
    finalizer to a certified 22-array generation — and Ruling E holds."""
    with tempfile.TemporaryDirectory() as tmp:
        sink, assembly, res = _committed_run(tmp, "fin")
        acc = _fresh_accumulator()
        WOI._build_test_result_from_miner(
            res, acc, CONFIG, PRNG_BASE, CTX["trial_number"], None,
            phase5_sink=sink)

        root = Path(tmp) / "gen_root"
        root.mkdir()
        # The SAME call shape the run-level Step-1 finalization uses
        # (window_optimizer_integration_final.py:1812-1822).
        artifact = RF.finalize_run(
            acc["bidirectional"], output_root=root,
            run_id=f"step1_{PRNG_BASE}_0", prng_base=PRNG_BASE,
            skip_modes_executed=("constant", "variable"),
            seed_start=0, seed_count=TOTAL_SEEDS,
            repository_commit="a" * 40, repository_tree_clean=True)

        # (1) every required path exists on disk
        paths = module.certified_paths(artifact)
        assert tuple(paths) == ORACLE_ARTIFACT_PATHS, tuple(paths)
        for name, p in paths.items():
            assert os.path.exists(p), f"{name} does not exist: {p}"

        # (2) the bundle is the frozen 22 arrays, in the frozen order
        with np.load(artifact.binary_npz_path) as npz:
            names = list(npz.files)
            assert tuple(names) == ORACLE_ARRAY_NAMES, names
            assert len(names) == 22, len(names)
            seeds = npz["seeds"]
        assert len(seeds) == artifact.final_row_count, (len(seeds),
                                                        artifact.final_row_count)
        assert artifact.raw_candidate_count == ORACLE_APPENDED_ROWS, artifact

        # (3) Ruling E: the assembly's two NPZ path fields stay None, and the
        #     certified paths come from RunArtifactResult instead.
        assert assembly.binary_npz_path is None, assembly.binary_npz_path
        assert assembly.all_npz_path is None, assembly.all_npz_path
        assert sink.get_assembly("fin").binary_npz_path is None
        assert sink.get_assembly("fin").all_npz_path is None

        # (4) the rows that reached the finalizer are the assembly's rows
        assert artifact.raw_candidate_count == (
            len(assembly.canonical_records_constant)
            + len(assembly.canonical_records_variable)), artifact


# ═════════════════════════════════════════════════════════════════════════════
# G-FAILCLOSED
# ═════════════════════════════════════════════════════════════════════════════
def g_failclosed(module=SI):
    """Run the gate with BOTH this harness and the production module
    pointed at `module`, so a mutant really reaches the production path."""
    with _adapter(module):
        return _g_failclosed_impl(module)


def _g_failclosed_impl(module):
    """Every absent publication result RAISES, and leaves the accumulator
    untouched. A zero/empty TestResult is never returned."""
    err = module.MinerIngressError

    def _untouched(acc):
        assert acc == {"forward_count": 0, "reverse_count": 0,
                       "bidirectional": []}, (
            f"the accumulator was mutated on a fail-closed path: {acc}")

    # (1) no sink wired at all — the pre-D6 shape, which used to return +0
    acc = _fresh_accumulator()
    _raises(err, module.require_assembly, None,
            {"run_id": "x", "state": "committed"}, trial_number=1)
    _raises(err, WOI._build_test_result_from_miner,
            {"run_id": "x"}, acc, CONFIG, PRNG_BASE, 1, None, phase5_sink=None)
    _untouched(acc)

    # (2) a sink with no get_assembly accessor
    class _NoAccessor:
        pass
    _raises(err, module.require_assembly, _NoAccessor(),
            {"run_id": "x"}, trial_number=2)

    # (3) no run_id in the serve result
    sink = module.build_assembling_sink()
    for bad in ({}, {"run_id": None}, {"run_id": ""}, {"run_id": 7}):
        _raises(err, module.require_assembly, sink, bad, trial_number=3)

    # (4) a real, PUBLISHED but UNCOMMITTED run — manifests exist, no assembly
    with tempfile.TemporaryDirectory() as tmp:
        sink = module.build_assembling_sink()
        _build_run(tmp, sink, "uncommitted")
        assert sink.get_assembly("uncommitted") is None
        acc = _fresh_accumulator()
        _raises(err, WOI._build_test_result_from_miner,
                {"run_id": "uncommitted", "state": "assigned"}, acc, CONFIG,
                PRNG_BASE, 4, None, phase5_sink=sink)
        _untouched(acc)

    # (5) a run that was ABORTED after publishing — tombstoned, assembly gone
    with tempfile.TemporaryDirectory() as tmp:
        sink, _assembly, res = _committed_run(tmp, "aborted")
        sink.abort_trial({"run_id": "aborted", "event_id": "a1"})
        assert sink.get_assembly("aborted") is None
        acc = _fresh_accumulator()
        _raises(err, WOI._build_test_result_from_miner,
                res, acc, CONFIG, PRNG_BASE, 5, None, phase5_sink=sink)
        _untouched(acc)

    # (6) a certified generation missing a required path
    class _Artifact:
        def __init__(self, **kw):
            for k in ORACLE_ARTIFACT_PATHS:
                setattr(self, k, kw.get(k, "/tmp/x"))
    _raises(err, module.certified_paths, None)
    for name in ORACLE_ARTIFACT_PATHS:
        _raises(err, module.certified_paths, _Artifact(**{name: None}))
    assert module.certified_paths(_Artifact()) == {
        k: "/tmp/x" for k in ORACLE_ARTIFACT_PATHS}

    # (7) the failure is never a quietly-successful zero TestResult
    with tempfile.TemporaryDirectory() as tmp:
        sink = module.build_assembling_sink()
        _build_run(tmp, sink, "nozero")
        try:
            out = WOI._build_test_result_from_miner(
                {"run_id": "nozero"}, _fresh_accumulator(), CONFIG, PRNG_BASE,
                6, None, phase5_sink=sink)
        except err:
            out = None
        assert out is None, (
            f"a missing assembly produced a TestResult ({out}) instead of "
            f"raising — this is exactly the fabricated zero D6 forbids")


# ═════════════════════════════════════════════════════════════════════════════
# G-TESTRESULT
# ═════════════════════════════════════════════════════════════════════════════
def _git_show(commit: str, path: str) -> str:
    return subprocess.run(["git", "-C", _ROOT, "show", f"{commit}:{path}"],
                          check=True, capture_output=True, text=True).stdout


def _func_src(src: str, path: str, name: str) -> str:
    node = _func_node(src, path, name)
    return ast.get_source_segment(src, node)


def g_testresult(integ_src=None):
    """The returned TestResult matches the Step-1 contract, and the PWC/ZMQ
    builders are byte-identical to their text at the frozen commit."""
    integ_src = _INTEG_SRC if integ_src is None else integ_src

    # (1) the contract fields, hand-transcribed
    import dataclasses
    fields = tuple(f.name for f in dataclasses.fields(TestResult))
    assert fields == ORACLE_TESTRESULT_FIELDS, fields

    with tempfile.TemporaryDirectory() as tmp:
        sink, _assembly, res = _committed_run(tmp, "tr")
        tr = WOI._build_test_result_from_miner(
            res, _fresh_accumulator(), CONFIG, PRNG_BASE, CTX["trial_number"],
            None, phase5_sink=sink)
    assert isinstance(tr, TestResult), type(tr)
    assert tuple(f.name for f in dataclasses.fields(tr)) == \
        ORACLE_TESTRESULT_FIELDS
    assert tr.config is CONFIG
    for name in ("forward_count", "reverse_count", "bidirectional_count",
                 "iteration"):
        assert isinstance(getattr(tr, name), int), (name, getattr(tr, name))
    # the derived properties Step 1 reads must still compute
    assert 0.0 <= tr.precision <= 1.0, tr.precision

    # (2) the PWC/ZMQ builder is BYTE-IDENTICAL to the frozen commit
    frozen_src = _git_show(ORACLE_FROZEN_COMMIT,
                           "window_optimizer_integration_final.py")
    for name in ORACLE_PWZ_BUILDERS:
        live = _func_src(integ_src, _INTEG_PATH, name)
        was = _func_src(frozen_src, _INTEG_PATH, name)
        assert live == was, (
            f"{name} DIFFERS from its text at {ORACLE_FROZEN_COMMIT} — D6 must "
            f"not touch the PWC/ZMQ builders")

    # (3) the two other backends' gates in run_bidirectional_test are unchanged
    live_rbt = _func_src(integ_src, _INTEG_PATH, "run_bidirectional_test")
    was_rbt = _func_src(frozen_src, _INTEG_PATH, "run_bidirectional_test")
    for marker in ("_use_pw = getattr(coordinator, 'use_persistent_workers'",
                   "_use_zmq = getattr(coordinator, 'use_zmq_sqlite'",
                   "return _build_test_result_from_pw(_pw_result, accumulator"):
        assert marker in live_rbt and marker in was_rbt, marker
    # everything after the miner gate must be untouched text
    tail_marker = "# END RANGE-MINER PATH — original path continues unchanged below"
    assert live_rbt.split(tail_marker, 1)[1] == was_rbt.split(tail_marker, 1)[1], (
        "the PWC/ZMQ portion of run_bidirectional_test changed")


# ═════════════════════════════════════════════════════════════════════════════
# G-BACKEND-DEFAULT
# ═════════════════════════════════════════════════════════════════════════════
def g_backend_default(module=SI):
    """Run the gate with BOTH this harness and the production module
    pointed at `module`, so a mutant really reaches the production path."""
    with _adapter(module):
        return _g_backend_default_impl(module)


def _g_backend_default_impl(module):
    """`serial_reference` when nothing is configured; `process_sharded` only by
    explicit selection; and the resolved backend actually reaches assembly."""
    assert module.DEFAULT_ASSEMBLY_BACKEND == ORACLE_DEFAULT_BACKEND, \
        module.DEFAULT_ASSEMBLY_BACKEND
    assert tuple(AB.ASSEMBLY_BACKENDS) == ORACLE_BACKENDS, AB.ASSEMBLY_BACKENDS

    # (1) no configuration -> serial_reference
    for unspecified in (None,):
        backend = module.resolve_assembly_backend(unspecified)
        assert backend.backend_name == ORACLE_DEFAULT_BACKEND, backend
        assert isinstance(backend, AB.SerialReferenceBackend), type(backend)
    assert module.build_assembling_sink()._backend.backend_name == \
        ORACLE_DEFAULT_BACKEND

    # (2) process_sharded is NOT reachable without explicit configuration
    _raises(NotImplementedError, module.resolve_assembly_backend,
            "process_sharded")
    sharded = module.resolve_assembly_backend("process_sharded", pool_size=2)
    assert isinstance(sharded, AB.ProcessShardedBackend), type(sharded)
    assert sharded.backend_name == "process_sharded"
    assert sharded.pool_size == 2

    # (3) resolution still fails closed — never a post-error serial fallback
    for bad in ("", "SERIAL_REFERENCE", "serial", "nonesuch", 7):
        _raises(ValueError, module.resolve_assembly_backend, bad)

    # (4) the backend REACHES assembly — a sink built with it must use it
    class _Recording:
        backend_name = "recording"

        def __init__(self):
            self.calls = 0

        def assemble(self, run_id, manifests):
            self.calls += 1
            return AB.SerialReferenceBackend().assemble(run_id, manifests)

    rec = _Recording()
    with tempfile.TemporaryDirectory() as tmp:
        sink = module.build_assembling_sink(rec)
        _build_run(tmp, sink, "bk")
        sink.commit_trial({"run_id": "bk", "event_id": "bk-c"})
        assert sink.get_assembly("bk") is not None
    assert rec.calls == 1, (
        f"the configured backend was called {rec.calls} times — backend "
        f"selection is not reaching the assembly")

    # (5) the production call site reads the backend from configuration
    fn = _func_node(_INTEG_SRC, _INTEG_PATH, "run_bidirectional_test")
    src = ast.get_source_segment(_INTEG_SRC, fn)
    assert "resolve_assembly_backend(" in src, src[:0]
    assert "'assembly_backend'" in src or '"assembly_backend"' in src, (
        "the miner gate does not read an assembly_backend configuration "
        "attribute")
    assert "phase5_sink            = _miner_sink" in src or \
           "phase5_sink=_miner_sink" in src, (
        "the miner gate does not pass the sink to run_trial_miner — without it "
        "no assembly is ever produced")


# ═════════════════════════════════════════════════════════════════════════════
# G-FLUSH-CADENCE
# ═════════════════════════════════════════════════════════════════════════════
def g_flush_cadence(integ_src=None):
    """The threshold-gated flush fires exactly as it did pre-D6: once per trial,
    after the append, with the same label — and the threshold gate itself is
    untouched."""
    integ_src = _INTEG_SRC if integ_src is None else integ_src

    # (1) the CADENCE RULE is byte-identical to the frozen commit.
    #
    #     [S172 Phase-5 D6.1] This assertion used to demand that the WHOLE of
    #     `_flush_npz_incremental` be byte-identical to its text at
    #     ORACLE_FROZEN_COMMIT. D6.1 repairs the body of that helper (the
    #     never-working atomic write, the swallowing except, the false S166
    #     guarantee), so whole-function identity is no longer the right oracle
    #     — it would red on a mandated repair while proving nothing about
    #     cadence.
    #
    #     What D6 actually owns here is the ENTRY GATE: how `_FLUSH_EVERY` and
    #     `_flush_last_count` decide whether a flush happens at all. That text
    #     is pinned verbatim against the frozen commit, so D6.1 (or anything
    #     later) still cannot shift the cadence while changing the write.
    frozen_src = _git_show(ORACLE_FROZEN_COMMIT,
                           "window_optimizer_integration_final.py")
    _frozen_fn = _func_src(frozen_src, _INTEG_PATH, "_flush_npz_incremental")
    _live_fn = _func_src(integ_src, _INTEG_PATH, "_flush_npz_incremental")

    ORACLE_CADENCE_GATE = (
        '    bidi = accumulator.get("bidirectional", [])\n'
        "    current_count = len(bidi)\n"
        "\n"
        "    new_since_last = current_count - _flush_last_count\n"
        "    if new_since_last < _FLUSH_EVERY:\n"
        "        return  # not enough new survivors yet"
    )
    # the oracle is only trustworthy if it really is the frozen text
    assert ORACLE_CADENCE_GATE in _frozen_fn, (
        "the hand-transcribed cadence-gate oracle is not present at "
        f"{ORACLE_FROZEN_COMMIT} — the oracle itself has drifted")
    assert ORACLE_CADENCE_GATE in _live_fn, (
        "the _flush_npz_incremental ENTRY GATE differs from its text at "
        f"{ORACLE_FROZEN_COMMIT} — the flush cadence rule must not shift")

    # (2) exactly ONE flush call in the miner builder, inside the
    #     `accumulator is not None` guard, with the pre-D6 label
    fn = _func_node(integ_src, _INTEG_PATH, "_build_test_result_from_miner")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "_flush_npz_incremental"]
    assert len(calls) == 1, f"{len(calls)} flush calls in the miner builder"
    pw = _func_node(integ_src, _INTEG_PATH, "_build_test_result_from_pw")
    pw_calls = [n for n in ast.walk(pw) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "_flush_npz_incremental"]
    assert len(pw_calls) == len(calls), (
        "the miner path's flush count differs from the PWC path's")

    # (3) runtime: one call per trial, correct label, and it happens AFTER the
    #     candidates are appended (so the flush sees this trial's rows).
    seen: List[Tuple[str, int]] = []
    real = WOI._flush_npz_incremental

    def _spy(accumulator, label=""):
        seen.append((label, len(accumulator.get("bidirectional", []))))

    try:
        WOI._flush_npz_incremental = _spy
        with tempfile.TemporaryDirectory() as tmp:
            sink, _assembly, res = _committed_run(tmp, "flush")
            acc = _fresh_accumulator()
            WOI._build_test_result_from_miner(
                res, acc, CONFIG, PRNG_BASE, 11, None, phase5_sink=sink)
    finally:
        WOI._flush_npz_incremental = real
    assert seen == [("chunk/trial-11", ORACLE_APPENDED_ROWS)], seen

    # (4) accumulator=None (the count-only caller) flushes NOTHING, as before
    seen.clear()
    try:
        WOI._flush_npz_incremental = _spy
        with tempfile.TemporaryDirectory() as tmp:
            sink, _assembly, res = _committed_run(tmp, "flush2")
            WOI._build_test_result_from_miner(
                res, None, CONFIG, PRNG_BASE, 12, None, phase5_sink=sink)
    finally:
        WOI._flush_npz_incremental = real
    assert seen == [], seen

    # (5) the threshold gate itself still governs: below _FLUSH_EVERY the real
    #     helper returns before doing anything at all; at the threshold it
    #     proceeds past the gate AND THE CHECKPOINT ACTUALLY LANDS.
    #
    #     [S172 Phase-5 D6.1] This assertion previously pinned the FAILED
    #     attempt. Pre-D6.1 the temp name `...all.npz.flush.tmp` did not end in
    #     `.npz`, so `np.savez_compressed` appended one, `os.replace` raised
    #     FileNotFoundError into a broad `except`, and the incremental NPZ was a
    #     permanent no-op. This gate observed that and pinned it. D6.1 repairs
    #     it, so the gate now pins SUCCESSFUL flush behaviour instead: the
    #     checkpoint files exist after the at-threshold call.
    #
    #     The accumulator is STILL not cleared, but for a different and now
    #     deliberate reason: `_FLUSH_CLEAR_IN_MEMORY` is False, because the
    #     4-array checkpoint cannot reconstruct the 24 CANONICAL_RECORD_FIELDS
    #     the D3.5 finalizer consumes from the in-memory list. Candidates
    #     therefore still all reach the finalizer — by design now, not by a bug.
    assert isinstance(WOI._FLUSH_EVERY, int) and WOI._FLUSH_EVERY > 0
    import contextlib as _ctx
    import io as _io
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        # [S172 D6.1] The snapshot root is deliberately NOT the CWD any more
        # (Beta path condition 2), so chdir alone no longer contains the write.
        # Pin the root and the run id, or this gate would write into the repo.
        _prev_root = os.environ.get("PRNG_CHECKPOINT_ROOT")
        _prev_run = os.environ.get("PRNG_CHECKPOINT_RUN_ID")
        os.environ["PRNG_CHECKPOINT_ROOT"] = tmp
        os.environ["PRNG_CHECKPOINT_RUN_ID"] = "d6-cadence"
        try:
            os.chdir(tmp)
            WOI._flush_last_count = 0
            acc = _fresh_accumulator()
            acc["bidirectional"] = [{"seed": i, "score": 0.5}
                                    for i in range(WOI._FLUSH_EVERY - 1)]
            buf = _io.StringIO()
            with _ctx.redirect_stdout(buf):
                real(acc, label="below")
            assert "[S152-FLUSH]" not in buf.getvalue(), (
                f"the flush fired below the threshold: {buf.getvalue()!r}")
            assert not os.listdir(tmp), f"the flush wrote below threshold: {os.listdir(tmp)}"
            assert len(acc["bidirectional"]) == WOI._FLUSH_EVERY - 1

            acc["bidirectional"].append({"seed": 999, "score": 0.5})
            buf = _io.StringIO()
            with _ctx.redirect_stdout(buf):
                real(acc, label="at")
            assert "[S152-FLUSH]" in buf.getvalue(), (
                f"the flush did not fire at the threshold: {buf.getvalue()!r}")
            # [D6.1] the flush now SUCCEEDS — pin the landed snapshot, in its
            # run-isolated directory
            _ck = WOI._flush_checkpoint_dir()
            assert os.path.isfile(os.path.join(_ck, WOI._CHECKPOINT_ALL_NAME)), (
                f"the at-threshold flush did not land a snapshot: "
                f"{os.listdir(tmp)}")
            assert os.path.isfile(os.path.join(_ck, WOI._CHECKPOINT_BINARY_NAME))
            # [D6.1] and it did NOT write the finalizer-owned root aliases
            for _root_name in ("bidirectional_survivors_all.npz",
                               "bidirectional_survivors_binary.npz"):
                assert not os.path.lexists(os.path.join(tmp, _root_name)), (
                    f"the checkpoint wrote {_root_name} in the run root — that "
                    f"path is a finalizer-owned compatibility symlink")
            # the clear stays disabled: candidates still reach the finalizer
            assert len(acc["bidirectional"]) == WOI._FLUSH_EVERY, (
                "the accumulator was cleared — candidates would stop reaching "
                "the finalizer")
        finally:
            os.chdir(cwd)
            WOI._flush_last_count = 0
            for _k, _v in (("PRNG_CHECKPOINT_ROOT", _prev_root),
                           ("PRNG_CHECKPOINT_RUN_ID", _prev_run)):
                if _v is None:
                    os.environ.pop(_k, None)
                else:
                    os.environ[_k] = _v


# ═════════════════════════════════════════════════════════════════════════════
# MUTATION PROOF — the four-part rule
# ═════════════════════════════════════════════════════════════════════════════
_MUT_SEQ = 0
_MUT_DIR = None


def _mut_dir():
    global _MUT_DIR
    if _MUT_DIR is None:
        _MUT_DIR = tempfile.mkdtemp(prefix="d6_mutants_")
        sys.path.insert(0, _MUT_DIR)
    return _MUT_DIR


def _patch(src: str, old: str, new: str, label: str) -> str:
    """Textual mutation that MUST actually apply — part 1 of the four-part rule.
    A no-op patch would make a mutant vacuously survive (a false green)."""
    count = src.count(old)
    assert count == 1, (
        f"{label}: anchor is not unique ({count} occurrences) — the mutation "
        f"would be unverifiable")
    return src.replace(old, new, 1)


def _load_mutant(src: str, label: str):
    global _MUT_SEQ
    _MUT_SEQ += 1
    name = f"_d6_mutant_{_MUT_SEQ}"
    path = os.path.join(_mut_dir(), f"{name}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.__d6_label__ = label
    return module


def _executed(module, marker: str, label: str) -> None:
    """Part 2: the mutated text is present in the module that was loaded."""
    src = inspect.getsource(module)
    assert marker in src, (
        f"{label}: the mutated text is absent from the loaded module — the "
        f"mutant did not take effect")


def _positive_control(name, detector):
    """Part 3: the detector must PASS against the UNMUTATED module."""
    try:
        detector()
    except Exception as exc:                                    # noqa: BLE001
        raise AssertionError(
            f"POSITIVE CONTROL FAILED for {name}: the detector reds against the "
            f"UNMUTATED module ({type(exc).__name__}: {exc}) — any kill it "
            f"records would be unattributable") from exc


def _record(label, detector, credited, marker=None, module=None):
    """Parts 3+4: run `detector` and require it to FAIL."""
    if module is not None and marker is not None:
        _executed(module, marker, label)
    try:
        detector()
    except AssertionError as exc:
        sig = str(exc).splitlines()[0][:150] or type(exc).__name__
        _MUTANTS.append((label, f"AssertionError: {sig}", credited,
                         "applies-once ✓ | mutated-path ✓ | detector-clean ✓ | "
                         "injected-defect ✓"))
        return
    except Exception as exc:                                    # noqa: BLE001
        sig = f"{type(exc).__name__}: {str(exc).splitlines()[0][:130]}"
        _MUTANTS.append((label, sig, credited,
                         "applies-once ✓ | mutated-path ✓ | detector-clean ✓ | "
                         "injected-defect ✓"))
        return
    raise AssertionError(f"MUTANT SURVIVED: {label} — {credited} did not red")


def _ingress_mutant(old, new, label):
    """Load a mutated `miner/step1_ingress.py` under a throwaway module name."""
    return _load_mutant(_patch(_INGRESS_SRC, old, new, label), label)


def _integ_mutant_src(old, new, label):
    """A mutated integration SOURCE STRING for the AST-only detectors (the
    module itself is far too entangled to import a second copy of)."""
    return _patch(_INTEG_SRC, old, new, label)


def run_mutants():
    # ---- G-INGRESS -------------------------------------------------------
    _positive_control("G-INGRESS", lambda: g_ingress(SI))

    m = _ingress_mutant(
        "        accumulator['bidirectional'].extend(records_variable)",
        "        pass  # MUTANT: variable-mode records dropped",
        "M1 drop variable records")
    _record("M1 drop variable records", lambda: g_ingress(m), "G-INGRESS",
            "MUTANT: variable-mode records dropped", m)

    m = _ingress_mutant(
        "        accumulator.setdefault('bidirectional', []).extend(records_constant)\n"
        "        accumulator['bidirectional'].extend(records_variable)",
        "        accumulator.setdefault('bidirectional', []).extend(records_variable)\n"
        "        accumulator['bidirectional'].extend(records_constant)  # MUTANT: mode order swapped",
        "M2 swap mode order")
    _record("M2 swap mode order", lambda: g_ingress(m), "G-INGRESS",
            "MUTANT: mode order swapped", m)

    m = _ingress_mutant(
        "    records_constant: List[Dict[str, Any]] = assembly.canonical_records_constant",
        "    import json as _j  # MUTANT: records rebuilt through a round trip\n"
        "    records_constant: List[Dict[str, Any]] = [\n"
        "        _j.loads(_j.dumps(r)) for r in assembly.canonical_records_constant]",
        "M3 re-normalize records")
    _record("M3 re-normalize records", lambda: g_ingress(m), "G-INGRESS",
            "MUTANT: records rebuilt through a round trip", m)

    m = _ingress_mutant(
        'forward_total=int(counts["forward_constant"]) + int(counts["forward_variable"]),',
        'forward_total=int(counts["forward_constant"]),  # MUTANT: variable direction dropped',
        "M4 forward count drops variable")
    _record("M4 forward count drops variable", lambda: g_ingress(m),
            "G-INGRESS", "MUTANT: variable direction dropped", m)

    m = _ingress_mutant(
        "    missing = [k for k in (",
        "    counts = {k: counts.get(k, 0) for k in (  # MUTANT: missing count reads as zero\n"
        '        \"forward_constant\", \"reverse_constant\", \"forward_variable\",\n'
        '        \"reverse_variable\", \"bidirectional_constant\", \"bidirectional_variable\")}\n'
        "    missing = [k for k in (",
        "M5 missing count defaults to zero")
    _record("M5 missing count defaults to zero", lambda: g_ingress(m),
            "G-INGRESS", "MUTANT: missing count reads as zero", m)

    # ---- G-FAILCLOSED ----------------------------------------------------
    _positive_control("G-FAILCLOSED", lambda: g_failclosed(SI))

    m = _ingress_mutant(
        "    if sink is None:\n"
        "        raise MinerIngressError(",
        "    if sink is None:\n"
        "        return _FABRICATED_EMPTY  # MUTANT: no-sink returns an empty assembly\n"
        "    if False:\n"
        "        raise MinerIngressError(",
        "M6 no-sink fabricates empty")
    # the fabricated object the mutant returns
    m_src = _patch(
        inspect.getsource(m),
        "class MinerIngressError(Exception):",
        "_FABRICATED_EMPTY = MinerTrialAssembly(\n"
        "    run_id='fabricated', bidirectional_constant=set(),\n"
        "    bidirectional_variable=set(), forward_map_constant={},\n"
        "    reverse_map_constant={}, forward_map_variable={},\n"
        "    reverse_map_variable={}, canonical_records_constant=[],\n"
        "    canonical_records_variable=[],\n"
        "    directional_counts={k: 0 for k in ('forward_constant',\n"
        "        'reverse_constant', 'forward_variable', 'reverse_variable',\n"
        "        'bidirectional_constant', 'bidirectional_variable')},\n"
        "    timing={'assembly_s': 0.0})\n\n\n"
        "class MinerIngressError(Exception):",
        "M6 fabricated object")
    # _FABRICATED_EMPTY must be defined after the class it references; rebuild
    # by appending instead.
    m_src = _INGRESS_SRC + (
        "\n\n_ORIG_REQUIRE = require_assembly\n"
        "_FABRICATED_EMPTY = MinerTrialAssembly(\n"
        "    run_id='fabricated', bidirectional_constant=set(),\n"
        "    bidirectional_variable=set(), forward_map_constant={},\n"
        "    reverse_map_constant={}, forward_map_variable={},\n"
        "    reverse_map_variable={}, canonical_records_constant=[],\n"
        "    canonical_records_variable=[],\n"
        "    directional_counts={k: 0 for k in ('forward_constant',\n"
        "        'reverse_constant', 'forward_variable', 'reverse_variable',\n"
        "        'bidirectional_constant', 'bidirectional_variable')},\n"
        "    timing={'assembly_s': 0.0})\n\n\n"
        "def require_assembly(sink, miner_result, *, trial_number):\n"
        "    # MUTANT: an absent publication result becomes an empty assembly\n"
        "    try:\n"
        "        return _ORIG_REQUIRE(sink, miner_result, trial_number=trial_number)\n"
        "    except MinerIngressError:\n"
        "        return _FABRICATED_EMPTY\n")
    m = _load_mutant(m_src, "M6 absent result fabricates empty assembly")
    _record("M6 absent result fabricates empty assembly",
            lambda: g_failclosed(m), "G-FAILCLOSED",
            "MUTANT: an absent publication result becomes an empty assembly", m)

    m_src = _INGRESS_SRC + (
        "\n\ndef certified_paths(artifact):\n"
        "    # MUTANT: a missing certified path is tolerated as an empty string\n"
        "    return {k: str(getattr(artifact, k, '') or '')\n"
        "            for k in _REQUIRED_ARTIFACT_PATHS}\n")
    m = _load_mutant(m_src, "M7 tolerate missing certified path")
    _record("M7 tolerate missing certified path", lambda: g_failclosed(m),
            "G-FAILCLOSED",
            "MUTANT: a missing certified path is tolerated as an empty string",
            m)

    # ---- G-BACKEND-DEFAULT ----------------------------------------------
    _positive_control("G-BACKEND-DEFAULT", lambda: g_backend_default(SI))

    m = _ingress_mutant(
        'DEFAULT_ASSEMBLY_BACKEND = SERIAL_REFERENCE',
        'DEFAULT_ASSEMBLY_BACKEND = "process_sharded"  # MUTANT: unpromoted backend as default',
        "M8 process_sharded as default")
    _record("M8 process_sharded as default", lambda: g_backend_default(m),
            "G-BACKEND-DEFAULT", "MUTANT: unpromoted backend as default", m)

    m_src = _INGRESS_SRC + (
        "\n\ndef build_assembling_sink(backend=None):\n"
        "    # MUTANT: the resolved backend is discarded\n"
        "    return AssemblingPhase5Sink()\n")
    m = _load_mutant(m_src, "M9 sink discards the backend")
    _record("M9 sink discards the backend", lambda: g_backend_default(m),
            "G-BACKEND-DEFAULT", "MUTANT: the resolved backend is discarded", m)

    # the writer-side seam: a sink that ignores its backend entirely
    w_src = _patch(_WRITER_SRC,
                   "        if self._backend is None:\n"
                   "            return assemble_trial(run_id, manifests)\n"
                   "        return self._backend.assemble(run_id, manifests).assembly",
                   "        return assemble_trial(run_id, manifests)  "
                   "# MUTANT: backend seam bypassed",
                   "M10 writer ignores backend")
    w_mod = _load_mutant(w_src, "M10 writer ignores backend")
    i_src = _patch(_INGRESS_SRC,
                   "from miner.range_miner_npz_writer import (\n"
                   "    AssemblingPhase5Sink,\n"
                   "    MinerTrialAssembly,\n"
                   ")",
                   f"from {w_mod.__name__} import (  # MUTANT: backend seam bypassed\n"
                   "    AssemblingPhase5Sink,\n"
                   "    MinerTrialAssembly,\n"
                   ")",
                   "M10 ingress against mutated writer")
    m = _load_mutant(i_src, "M10 writer ignores backend")
    _record("M10 writer ignores backend", lambda: g_backend_default(m),
            "G-BACKEND-DEFAULT", "MUTANT: backend seam bypassed", m)

    # ---- G-NO-PWZ-INGRESS ------------------------------------------------
    _positive_control("G-NO-PWZ-INGRESS", lambda: g_no_pwz_ingress())

    mutated_ingress = _patch(
        _INGRESS_SRC,
        "from miner.range_miner_npz_writer import (",
        "from utils.canonical_records import normalize_trial_populations  "
        "# MUTANT: PWC/ZMQ ingress imported\n"
        "from miner.range_miner_npz_writer import (",
        "M11 adapter imports the four-map normalizer")
    _record("M11 adapter imports the four-map normalizer",
            lambda: g_no_pwz_ingress(ingress_src=mutated_ingress),
            "G-NO-PWZ-INGRESS")

    mutated_integ = _integ_mutant_src(
        "    _assembly = require_assembly(phase5_sink, miner_result,",
        "    validate_trial_populations(miner_result, origin='miner')  "
        "# MUTANT: miner routed through the D3.25 wall\n"
        "    _assembly = require_assembly(phase5_sink, miner_result,",
        "M12 miner builder calls the D3.25 ingress wall")
    _record("M12 miner builder calls the D3.25 ingress wall",
            lambda: g_no_pwz_ingress(integ_src=mutated_integ),
            "G-NO-PWZ-INGRESS")

    # ---- G-TESTRESULT ----------------------------------------------------
    _positive_control("G-TESTRESULT", lambda: g_testresult())

    mutated_integ = _integ_mutant_src(
        "        accumulator['forward_count'] = accumulator.get('forward_count', 0) + len(fwd_records) + len(fwd_h_records)",
        "        accumulator['forward_count'] = accumulator.get('forward_count', 0) + len(fwd_records)  # MUTANT: PWC builder edited",
        "M13 PWC builder edited")
    _record("M13 PWC builder edited",
            lambda: g_testresult(integ_src=mutated_integ), "G-TESTRESULT")

    # ---- G-FLUSH-CADENCE -------------------------------------------------
    _positive_control("G-FLUSH-CADENCE", lambda: g_flush_cadence())

    mutated_integ = _integ_mutant_src(
        "    if accumulator is not None:\n"
        "        # [S152] Same call, same place, same cadence as every other backend.\n"
        "        _flush_npz_incremental(accumulator, label=f\"chunk/trial-{trial_number}\")",
        "    if accumulator is not None:\n"
        "        # MUTANT: flush cadence doubled\n"
        "        _flush_npz_incremental(accumulator, label=f\"chunk/trial-{trial_number}\")\n"
        "        _flush_npz_incremental(accumulator, label=f\"chunk/trial-{trial_number}\")",
        "M14 flush called twice")
    _record("M14 flush called twice",
            lambda: g_flush_cadence(integ_src=mutated_integ),
            "G-FLUSH-CADENCE")

    mutated_integ = _integ_mutant_src(
        "    if new_since_last < _FLUSH_EVERY:\n        return  # not enough new survivors yet",
        "    if False:  # MUTANT: threshold gate removed\n        return",
        "M15 flush threshold gate removed")
    _record("M15 flush threshold gate removed",
            lambda: g_flush_cadence(integ_src=mutated_integ),
            "G-FLUSH-CADENCE")

    # ---- G-FINALIZE ------------------------------------------------------
    _positive_control("G-FINALIZE", lambda: g_finalize(SI))

    m_src = _INGRESS_SRC + (
        "\n\ndef certified_paths(artifact):\n"
        "    # MUTANT: certified paths read off the assembly (Ruling E violation)\n"
        "    return {k: str(getattr(artifact, k, None)) for k in\n"
        "            ('generation_dir', 'binary_npz_path')}\n")
    m = _load_mutant(m_src, "M16 certified paths off the wrong object")
    _record("M16 certified paths off the wrong object",
            lambda: g_finalize(m), "G-FINALIZE",
            "MUTANT: certified paths read off the assembly", m)


# ═════════════════════════════════════════════════════════════════════════════
def main() -> int:
    print("=" * 78)
    print("S172 Phase 5 D6 — 3.A production-adapter gates")
    print("=" * 78)

    _check("G-INGRESS: assembly records -> accumulator, identical + real counts",
           lambda: g_ingress(SI))
    _check("G-INGRESS/builder: the same through _build_test_result_from_miner",
           g_ingress_through_builder)
    _check("G-NO-PWZ-INGRESS: D3.25 four-map ingress never reached (AST+runtime)",
           lambda: g_no_pwz_ingress())
    _check("G-FINALIZE: certified 22-array generation; Ruling E holds",
           lambda: g_finalize(SI))
    _check("G-FAILCLOSED: absent publication result raises, never a zero result",
           lambda: g_failclosed(SI))
    _check("G-TESTRESULT: Step-1 contract kept; PWC/ZMQ builders byte-identical",
           lambda: g_testresult())
    _check("G-BACKEND-DEFAULT: serial_reference default; process_sharded explicit",
           lambda: g_backend_default(SI))
    _check("G-FLUSH-CADENCE: one flush per trial, same label, same threshold",
           lambda: g_flush_cadence())

    print("\n" + "-" * 78)
    print("MUTATION PROOF (four-part rule)")
    print("-" * 78)
    _check("MUTANTS: every injected defect is caught", run_mutants)
    for label, sig, credited, parts in _MUTANTS:
        print(f"  [{_PASS}] {label}\n         killed by {credited} — {sig}\n"
              f"         {parts}")

    passed = sum(1 for _n, ok, _t in _results if ok)
    total = len(_results)
    print("\n" + "=" * 78)
    print(f"{passed}/{total} D6 gate checks green ({len(_MUTANTS)} mutants killed)")
    print("=" * 78)
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All D6 3.A gate checks green — the miner path ingests its own "
          "canonical candidates, reaches a certified generation through the "
          "shared finalizer, and fails closed on an absent publication result "
          "(pending Team Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if _MUT_DIR is not None:
            import shutil
            shutil.rmtree(_MUT_DIR, ignore_errors=True)
