#!/usr/bin/env python3
"""
test_s172_phase5_d5_process_sharded.py — S172 Phase-5 Deliverable D5 acceptance
harness (docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D5.md REV1 §7, the REV2
ADDENDUM's §5 six-row precedence matrix, and the REV3 ADDENDUM's §5 six-row
out-of-range seed domain + §6 oracle durability; G-* + 18 mutants).

Subject under test: `miner/assembly_shard_worker.py` (the CPU-only per-spool
worker, the projection artifact codec, the sampled concurrent-tree RSS sampler,
and the parent-side orchestration) plus `ProcessShardedBackend` in
`miner/assembly_backends.py`.

THE ONE PROPERTY EVERY GATE BELOW SERVES
    `process_sharded` parallelizes ONLY spool-local validation. Its output must
    be field-for-field identical to `serial_reference`'s, its exceptions must be
    equivalent in CLASS, `.args`, RENDERED MESSAGE, ATTRIBUTION and PRECEDENCE
    (tracebacks are explicitly not contractual, REV2 §3), and every
    globally-coupled step (map construction, duplicate detection, intersection,
    enrichment) must still happen exactly once, serially, in the parent.

    REV2 adds the third target: every precedence row is also compared against
    the PRESERVED PRE-D5 REFERENCE, pinned by commit and by digest, because the
    equivalence being proved is with the engine as it stood before D5 touched
    it — not merely between the two D5 backends.

    REV3 adds the third DOMAIN: the accepted-input set. Every gate written
    before it ran seeds that fit in an int64, so all of them stayed green while
    the projection layer silently narrowed what the engine accepts. The
    G-SEED-DOMAIN rows below therefore run seeds at and beyond the signed-64
    boundary — in both directions, and inside negative windows — against the
    same pre-D5 oracle. Two backends that narrow the domain identically agree
    with each other perfectly and are still wrong.

EVERY EXPECTATION IS HAND-TRANSCRIBED. No oracle is imported from a module under
test: the equivalence oracle is `serial_reference` (D4's documented role 1 —
"the definition of a right answer"), and the structural oracles are literals
written out here.

THE FOUR-PART MUTANT RULE (§7.C) — how this harness discharges it, per mutant:
    1. APPLIES EXACTLY ONCE  — `_patch` asserts its anchor occurs exactly once
       in the live source and rewrites exactly one occurrence.
    2. EXECUTES THE MUTATED PATH — `_executed` asserts the mutated text is
       present in the loaded mutant module's source AND that the mutant run
       actually reached it (each mutant's detector drives the mutated function).
    3. REACHES THE CREDITED ASSERTION — every detector is run FIRST against the
       UNMUTATED module as a POSITIVE CONTROL and must PASS there. A detector
       that cannot pass clean cannot be credited with a kill.
    4. FAILS FROM THE INJECTED DEFECT — because of (3), a loader, fixture,
       type-identity or setup failure would have already failed the positive
       control, so the recorded red is attributable to the mutation alone.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 tests/test_s172_phase5_d5_process_sharded.py
Exit 0 = all green. Exit 1 = a check failed (DO NOT COMMIT).
"""
import ast
import copy
import hashlib
import importlib.util
import json
import multiprocessing
import os
import resource
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
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
    AssemblyConsistencyError,
    DirectionalDuplicateError,
    MinerTrialAssembly,
    PhaseIdentityError,
    SpoolIdentityError,
    ValidatedSpoolProjection,
)
from utils import run_finalizer as RF  # noqa: E402

import miner.assembly_backends as AB           # noqa: E402
import miner.assembly_shard_worker as ASW      # noqa: E402
import miner.range_miner_npz_writer as RMW     # noqa: E402

_WORKER_PATH = os.path.join(_ROOT, "miner", "assembly_shard_worker.py")
_BACKEND_PATH = os.path.join(_ROOT, "miner", "assembly_backends.py")
with open(_WORKER_PATH, "r", encoding="utf-8") as _f:
    _WORKER_SRC = _f.read()
with open(_BACKEND_PATH, "r", encoding="utf-8") as _f:
    _BACKEND_SRC = _f.read()

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_results: List[Tuple[str, bool, Any]] = []
_MUTANTS: List[Tuple[str, str, str, str]] = []   # label, signature, credited, four-part
_BENCH: List[Dict[str, Any]] = []


# ═════════════════════════════════════════════════════════════════════════════
# Hand-transcribed oracles — literals, never imported from a module under test
# ═════════════════════════════════════════════════════════════════════════════

# The twelve STABLE MinerTrialAssembly fields equivalence compares. `timing` is
# deliberately absent: it carries a live perf_counter delta (D4 G3 [B2]).
ORACLE_STABLE_ASSEMBLY_FIELDS: Tuple[str, ...] = (
    "run_id",
    "bidirectional_constant", "bidirectional_variable",
    "forward_map_constant", "reverse_map_constant",
    "forward_map_variable", "reverse_map_variable",
    "canonical_records_constant", "canonical_records_variable",
    "directional_counts",
    "binary_npz_path", "all_npz_path",
)
ORACLE_TIMING_KEYS: Tuple[str, ...] = ("assembly_s",)

# The compact worker return contract (brief §4.1.4), written out.
ORACLE_RESULT_KEYS: Tuple[str, ...] = (
    "artifact_path", "survivor_count", "artifact_sha256",
    "stripe_id", "sub_index", "attempt", "workflow_phase",
    "direction", "skip_mode", "prng_type",
)

# The per-position OUTCOME envelope [REV2 §3]: exactly two kinds, and nothing
# else may cross the process boundary.
ORACLE_OUTCOME_PROJECTION = "projection"
ORACLE_OUTCOME_READ_ERROR = "read_error"
ORACLE_OUTCOME_KEYS: Tuple[str, ...] = ("outcome_kind", "result")
ORACLE_READ_ERROR_KEYS: Tuple[str, ...] = ("outcome_kind", "descriptor")

# §5's canonical peak_rss evidence block.
ORACLE_PEAK_RSS_DEFINITION = "sampled_sum_of_parent_and_recursive_children_rss"
ORACLE_SAMPLE_INTERVAL_MS = 25

# The 13 structured DirectionalDuplicateError attributes D2 asserts on.
ORACLE_DUP_ATTRS: Tuple[str, ...] = (
    "run_id", "workflow_phase", "direction", "skip_mode", "seed",
    "first_stripe", "first_sub_index", "first_attempt", "first_match_rate",
    "dup_stripe", "dup_sub_index", "dup_attempt", "dup_match_rate",
)

# The frozen 22 array names, in the frozen ORDER (D3 / D3.5 / D4 transcribe the
# same list) — used by G-FINALIZER.
ORACLE_ARRAY_NAMES: Tuple[str, ...] = (
    "seeds", "forward_matches", "reverse_matches", "window_size", "offset",
    "trial_number", "skip_min", "skip_max", "skip_range", "forward_count",
    "reverse_count", "bidirectional_count", "intersection_count",
    "intersection_ratio", "intersection_weight", "bidirectional_selectivity",
    "forward_only_count", "reverse_only_count", "survivor_overlap_ratio",
    "score", "skip_mode", "prng_type",
)
ORACLE_ACCUM_DIR = ".s172_accumulator"
ORACLE_CURRENT = "current"
ORACLE_ALL_NPZ = "bidirectional_survivors_all.npz"
ORACLE_BINARY_NPZ = "bidirectional_survivors_binary.npz"

# The pool sizes §6.7.A names. NOT 12, NOT 24, and never os.cpu_count().
ORACLE_BENCH_POOL_SIZES: Tuple[int, ...] = (1, 2, 4, 6, 8)


# ═════════════════════════════════════════════════════════════════════════════
# Real-lifecycle fixture — D1.1 / D4 harness pattern
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
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    except Exception as other:                                  # noqa: BLE001
        raise AssertionError(
            f"expected {exc.__name__}, got {type(other).__name__}: {other}")
    raise AssertionError(f"expected {exc.__name__}, nothing was raised")


def _coord(tmp, sink, macro=MACRO_SIZE, cap=SUB_CAP, dbname="l.db"):
    ledger = MinerLedger(os.path.join(tmp, dbname))
    cfg = CoordinatorConfig(staging_dir=os.path.join(tmp, "staging"),
                            miner_stripe_size=macro,
                            seed_cap_amd=cap, seed_cap_nvidia=cap,
                            seed_cap_amd_hybrid=cap,
                            seed_cap_nvidia_hybrid=cap)
    return RangeMinerCoordinator(cfg, ledger, phase5_sink=sink)


def _register(coord, cap=SUB_CAP, wid="hostA:gpu0"):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    caps = {"amd": cap, "nvidia": cap, "amd_hybrid": cap, "nvidia_hybrid": cap}
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend="cuda",
        capabilities={"seed_caps": caps,
                      "supported_variants": list(supported_variants())},
        node_config=node, now=100.0)


def _survivor_entries(phase, seed_start, seed_count, pop=None):
    pop = PHASE_POP[phase] if pop is None else pop
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


def _build_run(tmp, sink, run_id, phases=(1, 2, 3, 4), pops=None,
               total=TOTAL_SEEDS, macro=MACRO_SIZE, cap=SUB_CAP, dbname="l.db"):
    """Drive a REAL trial through the REAL producer surface up to (but not
    including) the terminal commit. Returns the published manifest list."""
    coord = _coord(tmp, sink, macro, cap, dbname)
    coord.ledger.create_trial(run_id, CTX["trial_number"], now=100.0)
    coord.ledger.set_trial_context(run_id, dict(CTX))
    conn = _register(coord, cap)
    published: List[Dict[str, Any]] = []
    for phase in phases:
        family = PHASE_TABLE[phase][0]
        recs = coord.assign_stripes(run_id, family, phase, total, [conn],
                                    stripe_prefix=f"{run_id}__p{phase}",
                                    now=100.0)
        assert len(recs) == total // macro, recs
        for rec in recs:
            assert rec["claimed"], rec
            sid = rec["stripe_id"]
            survivors_total = 0
            for sub_index in range(rec["expected_substripes"]):
                s_start = rec["seed_start"] + sub_index * cap
                entries = _survivor_entries(
                    phase, s_start, cap,
                    None if pops is None else pops.get(phase, {}))
                _, pb = build_substripe_payload_bytes(
                    sid, sub_index, s_start, cap, entries)
                size, sha = len(pb), hashlib.sha256(pb).hexdigest()
                coord.ledger.record_substripe_result(
                    run_id, sid, 0, sub_index, conn.worker_id, s_start, cap,
                    len(entries), remote_spool_path=None, size_bytes=size,
                    sha256=sha, now=100.0)
                res = coord.stage_inline_shard(run_id, sid, 0, sub_index,
                                               s_start, cap, entries, size, sha,
                                               now=100.0)
                assert res["status"] == "verified", res
                survivors_total += len(entries)
            assert coord.ledger.record_stripe_complete(
                run_id, sid, 0, conn.worker_id, rec["expected_substripes"],
                survivors_total), "StripeComplete transition failed"
            coord.finalize_stripe(run_id, sid, now=100.0)
            published = _retained(sink, run_id)
    return published


def _repoint(manifest, tmp, obj_or_bytes, tag):
    """Write a mutated payload to a NEW staged path and repoint a COPY of the
    manifest at it, RECOMPUTING expected_size + expected_sha256 so the gate
    exercises the validator rather than merely the digest check."""
    if isinstance(obj_or_bytes, bytes):
        raw = obj_or_bytes
    else:
        raw = json.dumps(obj_or_bytes, separators=(",", ":"),
                         sort_keys=True).encode("utf-8")
    path = os.path.join(tmp, f"mutated_{tag}.json")
    with open(path, "wb") as f:
        f.write(raw)
    out = copy.deepcopy(manifest)
    out["local_spool_path"] = path
    out["expected_size"] = len(raw)
    out["expected_sha256"] = hashlib.sha256(raw).hexdigest()
    return out


def _load_payload(manifest):
    with open(manifest["local_spool_path"]) as fh:
        return json.load(fh)


def _sharded(manifests, run_id, pool_size=4, module=ASW, **kw):
    """Drive the sharded parent orchestration and return the assembly."""
    return module.run_sharded_assembly(run_id, manifests, pool_size,
                                       **kw).assembly


def _serial(manifests, run_id):
    return AB.get_assembly_backend("serial_reference").assemble(
        run_id, manifests).assembly


# ═════════════════════════════════════════════════════════════════════════════
# THE THIRD TARGET — the PRESERVED PRE-D5 REFERENCE (REV2 §5)
# ═════════════════════════════════════════════════════════════════════════════
# REV2 requires every precedence row to be run against three targets: the
# pre-D5 reference, `serial_reference`, and `process_sharded`. The pre-D5
# reference is the assembly engine as it stood BEFORE the D5 extraction — the
# interleaved read/merge loop whose exception precedence Beta ruled must be
# preserved. It is pinned to a COMMIT, not to HEAD, so it stays frozen once D5
# is committed, and its bytes are pinned by digest so the oracle cannot drift
# under the gate.
#
# It is loaded as an INDEPENDENT module: it therefore defines its own exception
# CLASSES. Cross-target comparison is by class NAME + `.args` + rendered message
# + structured attribution, never by `isinstance` — dying on class identity
# would be a type-identity failure, which proves nothing (§7.C part 4).
#
# DURABILITY [REV3 §6]. The oracle used to be fetched with `git cat-file` at run
# time, which made this gate depend on repository HISTORY: a shallow clone or a
# source archive (`git archive`, a release tarball) has no `3e8580a`, and the
# whole D5 acceptance harness would abort on an environment property rather than
# on a defect. So the oracle is now VENDORED — the exact `3e8580a` blob, byte
# for byte, digest-pinned below — and history is consulted only by G-ORACLE, as
# a FAITHFULNESS check that skips cleanly when history is absent. Durability
# always; faithfulness whenever it is verifiable.
#
# The fixture deliberately carries a `.py.frozen` suffix, not `.py`: it is
# oracle DATA, and a second importable copy of the assembly engine sitting in
# the tree is exactly the kind of stray module that gets imported by accident.
# It is loaded through `_load_mutant`, the same independent-module loader every
# mutant uses. (It also keeps Phase-4 gate 22's changed-`.py` coexistence set —
# and therefore its whitelist — untouched by this fixture.)
_PRE_D5_COMMIT = "3e8580a1e123243428e0d1b8d0ab043032ed11f7"
_PRE_D5_SHA256 = "be3f0a26eefdfe3590623a1af18b3b6d61552fc20beb575ff8771e7969c9f2c1"
_PRE_D5_FIXTURE = os.path.join(_ROOT, "tests", "fixtures",
                               "pre_d5_range_miner_npz_writer.py.frozen")
_PRE_D5_MODULE = None


def _pre_d5_source() -> bytes:
    """The vendored pre-D5 engine source, digest-pinned on every read."""
    assert os.path.isfile(_PRE_D5_FIXTURE), (
        f"the vendored pre-D5 oracle fixture is missing: {_PRE_D5_FIXTURE}")
    with open(_PRE_D5_FIXTURE, "rb") as fh:
        blob = fh.read()
    digest = hashlib.sha256(blob).hexdigest()
    assert digest == _PRE_D5_SHA256, (
        f"pre-D5 oracle fixture digest {digest} != pinned {_PRE_D5_SHA256} — "
        f"the frozen reference has been edited")
    return blob


def _pre_d5_writer():
    """Load (once) the pre-D5 assembly engine from the vendored frozen blob."""
    global _PRE_D5_MODULE
    if _PRE_D5_MODULE is None:
        module = _load_mutant(_pre_d5_source().decode("utf-8"), "PRE-D5")
        # it really is the PRE-extraction shape: the validator is still private
        # and the merge has not been split out yet
        assert hasattr(module, "_read_and_validate_spool"), module
        assert not hasattr(module, "merge_validated_spools"), (
            "the pinned oracle already carries the D5 extraction — it is not a "
            "pre-D5 reference")
        assert not hasattr(module, "CapturedSpoolReadError"), module
        assert not hasattr(module, "ValidatedSpoolProjection"), (
            "the pinned oracle already carries a projection type — it is not a "
            "pre-D5 reference")
        _PRE_D5_MODULE = module
    return _PRE_D5_MODULE


def _reference(manifests, run_id):
    return _pre_d5_writer().assemble_trial(run_id, manifests)


# ═════════════════════════════════════════════════════════════════════════════
# G-ORACLE [REV3 §6] — the vendored oracle is durable AND faithful
# ═════════════════════════════════════════════════════════════════════════════
def g_oracle_durability():
    """Durability is unconditional; faithfulness is checked whenever git history
    can answer, and skipped cleanly — never failed — when it cannot.

    The two properties are deliberately separate. A digest pin makes the oracle
    unforgeable but says nothing about WHICH source it froze; comparing against
    `git cat-file 3e8580a` says exactly that, but only where the commit is
    reachable. Requiring the second would make a shallow clone or a source
    archive red this gate for an environment property, which is precisely the
    dependency REV3 §6 asked to remove."""
    # ---- durability: the fixture exists, is digest-pinned, and LOADS ---------
    blob = _pre_d5_source()
    assert len(blob) > 20000, len(blob)
    module = _pre_d5_writer()
    assert callable(module.assemble_trial), module
    # and it is genuinely independent: its own classes, not the live ones
    assert module.SpoolIdentityError is not SpoolIdentityError, (
        "the oracle resolved to the LIVE engine — it proves nothing")
    assert module.__file__ != RMW.__file__, module.__file__

    # ---- faithfulness: only when full history is present --------------------
    cat = subprocess.run(
        ["git", "cat-file", "-p",
         f"{_PRE_D5_COMMIT}:miner/range_miner_npz_writer.py"],
        cwd=_ROOT, capture_output=True, timeout=120)
    if cat.returncode != 0:
        # shallow clone, source archive, or no repository at all: unverifiable,
        # not wrong. The digest pin above still holds.
        print(f"      (faithfulness skipped — {_PRE_D5_COMMIT[:7]} unreachable: "
              f"{cat.stderr.decode(errors='replace').strip()[:120]})")
        assert os.path.exists(os.path.join(_ROOT, ".git", "shallow")) \
            or not os.path.isdir(os.path.join(_ROOT, ".git")), (
            "git history IS present but the pinned oracle commit is "
            "unreachable — the reference has been rewritten, which is a real "
            "failure, not a shallow-clone skip")
        return
    assert hashlib.sha256(cat.stdout).hexdigest() == _PRE_D5_SHA256, (
        f"the vendored fixture is NOT the {_PRE_D5_COMMIT[:7]} blob: git says "
        f"{hashlib.sha256(cat.stdout).hexdigest()}, fixture is pinned at "
        f"{_PRE_D5_SHA256}")
    assert cat.stdout == blob, (
        "the vendored fixture and the committed blob have the same digest but "
        "differ byte-for-byte — impossible; the comparison itself is broken")


# ═════════════════════════════════════════════════════════════════════════════
# Equivalence comparison — field-for-field, hand-enumerated
# ═════════════════════════════════════════════════════════════════════════════
def _assert_equivalent(sharded: MinerTrialAssembly, serial: MinerTrialAssembly,
                       label: str) -> None:
    """`serial_reference` IS the oracle (D4 §6.7.B role 1). Compare every stable
    field, and compare records ELEMENT-WISE so a re-order or a filter is visible
    rather than being absorbed by a length check."""
    for field in ORACLE_STABLE_ASSEMBLY_FIELDS:
        got, want = getattr(sharded, field), getattr(serial, field)
        if field.startswith("canonical_records_"):
            assert len(got) == len(want), (
                f"{label}: {field} length {len(got)} != serial {len(want)}")
            for i, (g, w) in enumerate(zip(got, want)):
                assert g == w, f"{label}: {field}[{i}] {g} != serial {w}"
            continue
        assert got == want, f"{label}: {field} {got!r} != serial {want!r}"
    # both NPZ path fields stay None through EVERY backend (D3.5 Ruling E)
    assert sharded.binary_npz_path is None and sharded.all_npz_path is None, (
        f"{label}: a backend must not populate an NPZ path field")
    # timing: same shape as D4's G3 — finite, > 0, no backend-specific key
    assert tuple(sharded.timing) == ORACLE_TIMING_KEYS, sharded.timing
    assert sharded.timing["assembly_s"] > 0, sharded.timing
    assert sharded.timing["assembly_s"] == sharded.timing["assembly_s"], "NaN"


# ═════════════════════════════════════════════════════════════════════════════
# G-EQUIV — process_sharded output ≡ serial_reference output, across a matrix
# ═════════════════════════════════════════════════════════════════════════════
def g_equivalence_matrix():
    # (a) BOTH MODES, phases {1,2,3,4}
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "eq_both")
        assert len(ms) == 4 * (TOTAL_SEEDS // MACRO_SIZE) * (MACRO_SIZE // SUB_CAP)
        _assert_equivalent(_sharded(copy.deepcopy(ms), "eq_both"),
                           _serial(copy.deepcopy(ms), "eq_both"), "both-modes")

    # (b) CONSTANT ONLY, phases {1,2} — the other legitimate phase set
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "eq_const", phases=(1, 2))
        _assert_equivalent(_sharded(copy.deepcopy(ms), "eq_const"),
                           _serial(copy.deepcopy(ms), "eq_const"), "constant-only")

    # (c) EMPTY SURVIVORS in every population — the rectangular/empty path
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "eq_empty",
                        pops={p: {} for p in (1, 2, 3, 4)})
        sharded = _sharded(copy.deepcopy(ms), "eq_empty")
        assert sharded.directional_counts == {
            "forward_constant": 0, "reverse_constant": 0,
            "forward_variable": 0, "reverse_variable": 0,
            "bidirectional_constant": 0, "bidirectional_variable": 0,
        }, sharded.directional_counts
        _assert_equivalent(sharded, _serial(copy.deepcopy(ms), "eq_empty"),
                           "empty-survivors")

    # (d) HIGH SURVIVOR — every seed in range survives, so the projection is
    #     dense and the artifact codec carries real volume.
    with tempfile.TemporaryDirectory() as tmp:
        pops = {p: {s: 0.30 + (s % 50) / 100.0 for s in range(400)}
                for p in (1, 2, 3, 4)}
        ms = _build_run(tmp, AssemblingPhase5Sink(), "eq_high", pops=pops,
                        total=400, macro=200, cap=100)
        sharded = _sharded(copy.deepcopy(ms), "eq_high")
        assert sharded.directional_counts["forward_constant"] == 400, \
            sharded.directional_counts
        _assert_equivalent(sharded, _serial(copy.deepcopy(ms), "eq_high"),
                           "high-survivor")

    # (e) pool_size must not change the answer: 1 and 8 agree with serial.
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "eq_pool")
        serial = _serial(copy.deepcopy(ms), "eq_pool")
        for size in (1, 8):
            _assert_equivalent(_sharded(copy.deepcopy(ms), "eq_pool", size),
                               serial, f"pool_size={size}")


# ═════════════════════════════════════════════════════════════════════════════
# G-DUP-CROSS — same seed in one population across TWO spools
# ═════════════════════════════════════════════════════════════════════════════
def _dup_cross_fixture(tmp, run_id):
    ms = _build_run(tmp, AssemblingPhase5Sink(), run_id)
    first = next(m for m in ms if int(m["workflow_phase"]) == 1
                 and m["sub_index"] == 0 and m["stripe_id"].endswith("_s0"))
    second = next(m for m in ms if int(m["workflow_phase"]) == 1
                  and m["sub_index"] == 1 and m["stripe_id"].endswith("_s0"))
    payload = _load_payload(second)
    payload["seed_start"] = 0
    payload["seed_count"] = 20
    payload["survivors"].append([1, 0.11, None, [2]])       # seed 1 is first's
    dup = _repoint(second, tmp, payload, "dupcross")
    out = [dup if m["event_id"] == second["event_id"] else copy.deepcopy(m)
           for m in ms]
    return out, first, second


def _assert_same_duplicate(a, b, label):
    """Identical TYPE and identical first-vs-dup ATTRIBUTION under both
    backends. Attribution is what proves the parent merged in deterministic
    order: whichever spool is 'first' must not depend on who finished first."""
    assert type(a) is type(b), (label, type(a), type(b))
    for attr in ORACLE_DUP_ATTRS:
        av, bv = getattr(a, attr), getattr(b, attr)
        assert av == bv, f"{label}: {attr} sharded {av!r} != serial {bv!r}"
        assert av is not None, f"{label}: {attr} must be populated"
    assert str(a) == str(b), f"{label}: message differs\n{a}\n{b}"


def g_dup_cross():
    with tempfile.TemporaryDirectory() as tmp:
        ms, first, second = _dup_cross_fixture(tmp, "dupx")
        e_par = _raises(DirectionalDuplicateError, _sharded,
                        copy.deepcopy(ms), "dupx")
        e_ser = _raises(DirectionalDuplicateError, _serial,
                        copy.deepcopy(ms), "dupx")
        _assert_same_duplicate(e_par, e_ser, "dup-cross")
        # hand-transcribed attribution: seed 1 is FIRST owned by sub0 (rate
        # 0.90 from PHASE_POP[1]) and duplicated by sub1 at 0.11.
        assert e_par.seed == 1 and e_par.direction == "forward", e_par.seed
        assert e_par.skip_mode == "constant" and e_par.workflow_phase == 1
        assert e_par.first_stripe == first["stripe_id"], e_par.first_stripe
        assert e_par.first_sub_index == 0 and e_par.first_match_rate == 0.90
        assert e_par.dup_stripe == second["stripe_id"], e_par.dup_stripe
        assert e_par.dup_sub_index == 1 and e_par.dup_match_rate == 0.11


# ═════════════════════════════════════════════════════════════════════════════
# G-DUP-INTRA [F3] — same seed TWICE INSIDE ONE SPOOL
# ═════════════════════════════════════════════════════════════════════════════
def _dup_intra_fixture(tmp, run_id):
    """The projection must preserve order AND multiplicity with zero dedup: a
    worker that sorts or uniques its rows makes this duplicate vanish."""
    ms = _build_run(tmp, AssemblingPhase5Sink(), run_id)
    target = next(m for m in ms if int(m["workflow_phase"]) == 1
                  and m["sub_index"] == 0 and m["stripe_id"].endswith("_s0"))
    payload = _load_payload(target)
    assert payload["survivors"][0][0] == 1, payload["survivors"]
    payload["survivors"].append([1, 0.22, None, [3]])     # seed 1 again, later
    bad = _repoint(target, tmp, payload, "dupintra")
    return [bad if m["event_id"] == target["event_id"] else copy.deepcopy(m)
            for m in ms], target


def _detect_dup_intra(module=ASW):
    with tempfile.TemporaryDirectory() as tmp:
        ms, target = _dup_intra_fixture(tmp, "dupi")
        e_par = _raises(DirectionalDuplicateError, _sharded,
                        copy.deepcopy(ms), "dupi", 4, module)
        e_ser = _raises(DirectionalDuplicateError, _serial,
                        copy.deepcopy(ms), "dupi")
        _assert_same_duplicate(e_par, e_ser, "dup-intra")
        # both occurrences are the SAME spool, so first and dup provenance are
        # identical and only the match_rate distinguishes them.
        assert e_par.seed == 1, e_par.seed
        assert e_par.first_stripe == e_par.dup_stripe == target["stripe_id"]
        assert e_par.first_sub_index == e_par.dup_sub_index == 0
        assert e_par.first_match_rate == 0.90, e_par.first_match_rate
        assert e_par.dup_match_rate == 0.22, e_par.dup_match_rate


def g_dup_intra():
    _detect_dup_intra()


# ═════════════════════════════════════════════════════════════════════════════
# G-MALFORMED-DUAL — the EARLIER-in-order defect is the observed one
# ═════════════════════════════════════════════════════════════════════════════
#
# CONSTRUCTION (this is what makes the as_completed mutant die deterministically
# rather than by luck): the EARLIER-in-order malformed spool is made EXPENSIVE —
# ~60k well-formed survivors that must all be hashed, parsed and validated
# before the very last one fails — while the LATER-in-order malformed spool is
# made TRIVIAL: it fails on the size check, before a single byte is parsed. With
# a pool wide enough to run both at once, the cheap one finishes orders of
# magnitude sooner. An `as_completed()` consumer therefore reports the LATER
# spool; an in-order consumer reports the EARLIER one regardless of timing.
def _malformed_dual_fixture(tmp, run_id):
    ms = _build_run(tmp, AssemblingPhase5Sink(), run_id)
    early = next(m for m in ms if int(m["workflow_phase"]) == 1
                 and m["sub_index"] == 0 and m["stripe_id"].endswith("_s0"))
    late = next(m for m in ms if int(m["workflow_phase"]) == 4
                and m["sub_index"] == 1 and m["stripe_id"].endswith("_s1"))

    # EARLY: expensive, fails on the LAST survivor's skip type.
    payload = _load_payload(early)
    payload["seed_start"], payload["seed_count"] = 0, 100000
    payload["survivors"] = [[s, 0.5, None, [1]] for s in range(60000)]
    payload["survivors"].append([60000, 0.5, None, "not-a-list"])
    early_bad = _repoint(early, tmp, payload, "earlybad")

    # LATE: trivial, fails immediately on size (no parse at all).
    late_bad = copy.deepcopy(late)
    late_bad["expected_size"] = 7

    out = []
    for m in ms:
        if m["event_id"] == early["event_id"]:
            out.append(early_bad)
        elif m["event_id"] == late["event_id"]:
            out.append(late_bad)
        else:
            out.append(copy.deepcopy(m))
    return out, early, late


def _detect_malformed_dual(module=ASW):
    with tempfile.TemporaryDirectory() as tmp:
        ms, early, late = _malformed_dual_fixture(tmp, "dual")
        e_par = _raises(SpoolIdentityError, _sharded,
                        copy.deepcopy(ms), "dual", 8, module)
        e_ser = _raises(SpoolIdentityError, _serial, copy.deepcopy(ms), "dual")
        # the EARLIER-in-order defect is the observed one, under both backends
        assert "skip_sequence is str" in str(e_par), (
            f"process_sharded surfaced the LATER-in-order defect — the parent "
            f"is consuming completion order, not manifest order: {e_par}")
        assert str(e_par) == str(e_ser), f"\nsharded: {e_par}\nserial : {e_ser}"
        assert early["stripe_id"] in str(e_par), str(e_par)
        assert late["stripe_id"] not in str(e_par), str(e_par)


def g_malformed_dual():
    _detect_malformed_dual()


# ═════════════════════════════════════════════════════════════════════════════
# G-PRECEDENCE — a metadata defect pre-empts a spool defect
# ═════════════════════════════════════════════════════════════════════════════
def _precedence_fixture(tmp, run_id):
    """BOTH defects present at once: an incomplete phase set (metadata) AND a
    malformed spool. The metadata exception must win, which it can only do if
    the parent ran the FULL gauntlet before dispatching any worker."""
    ms = _build_run(tmp, AssemblingPhase5Sink(), run_id)
    by_phase = {}
    for m in ms:
        by_phase.setdefault(int(m["workflow_phase"]), []).append(m)
    subset = [copy.deepcopy(m) for p in (1, 2, 3) for m in by_phase[p]]  # {1,2,3}
    target = subset[0]
    payload = _load_payload(target)
    payload["stripe_id"] = "not-my-stripe"
    subset[0] = _repoint(target, tmp, payload, "prec")
    return subset


def _detect_precedence(module=ASW):
    with tempfile.TemporaryDirectory() as tmp:
        subset = _precedence_fixture(tmp, "prec")
        e_par = _raises(AssemblyConsistencyError, _sharded,
                        copy.deepcopy(subset), "prec", 4, module)
        assert not isinstance(e_par, SpoolIdentityError), e_par
        assert "phase" in str(e_par).lower(), str(e_par)
        e_ser = _raises(AssemblyConsistencyError, _serial,
                        copy.deepcopy(subset), "prec")
        assert str(e_par) == str(e_ser), f"\nsharded: {e_par}\nserial : {e_ser}"

    # and the per-manifest identity layer likewise pre-empts a spool defect
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "prec2")
        bad = [copy.deepcopy(m) for m in ms]
        payload = _load_payload(bad[0])
        payload["stripe_id"] = "not-my-stripe"
        bad[0] = _repoint(bad[0], tmp, payload, "prec2")
        bad[1]["trial_metadata"]["direction"] = "reverse"     # phase-1 identity
        e_par = _raises(PhaseIdentityError, _sharded,
                        copy.deepcopy(bad), "prec2", 4, module)
        e_ser = _raises(PhaseIdentityError, _serial, copy.deepcopy(bad), "prec2")
        assert str(e_par) == str(e_ser), f"\nsharded: {e_par}\nserial : {e_ser}"


def g_precedence():
    _detect_precedence()


# ═════════════════════════════════════════════════════════════════════════════
# G-MATRIX [REV2 §5] — the SIX-ROW precedence matrix, across THREE targets
# ═════════════════════════════════════════════════════════════════════════════
#
# | earlier position     | later position | required result                     |
# |----------------------|----------------|-------------------------------------|
# | duplicate            | malformed      | identical duplicate + attribution   |
# | malformed            | duplicate      | identical malformed exception       |
# | intra-spool duplicate| malformed      | identical duplicate + attribution   |
# | malformed A          | malformed B    | the error from A                    |
# | valid                | duplicate      | identical duplicate attribution     |
# | valid                | malformed      | identical malformed exception       |
#
# Each row runs against the preserved pre-D5 reference, `serial_reference` and
# `process_sharded`, and each assertion compares MORE than `str(exc)`: class,
# `.args`, rendered message and the structured attribution fields.
#
# THE DETERMINISTIC ORDER, HAND-TRANSCRIBED (never imported): sort key
# `(workflow_phase, stripe_id, sub_index, attempt, event_id)`. With the standard
# fixture that makes ordered position 0 = phase1/_s0/sub0 (seeds 0-9, so seed 1
# @0.90), position 1 = phase1/_s0/sub1, position 2 = phase1/_s1/sub0 (seeds
# 20-29, so seed 25 @0.70), position 3 = phase1/_s1/sub1, and position 15 =
# phase4/_s1/sub1 — the last spool of the whole trial.
def _pos_key(m):
    return (int(m["workflow_phase"]), str(m.get("stripe_id")),
            int(m.get("sub_index", 0)), int(m.get("attempt", 0)),
            str(m.get("event_id")))


def _ordered(ms):
    return sorted(ms, key=_pos_key)


def _swap(ms, target, replacement):
    return [replacement if m["event_id"] == target["event_id"]
            else copy.deepcopy(m) for m in ms]


def _put_malformed(ms, tmp, position, tag):
    """The spool at ordered `position` carries a payload stripe_id that is not
    its own — a §5.3 identity defect, i.e. a SpoolIdentityError."""
    target = _ordered(ms)[position]
    payload = _load_payload(target)
    payload["stripe_id"] = "not-my-stripe"
    return _swap(ms, target, _repoint(target, tmp, payload, tag)), target


def _put_cross_dup(ms, tmp, position, seed, rate, tag, span=(0, 20)):
    """The spool at ordered `position` re-emits a seed an EARLIER position in
    the same population already owns."""
    target = _ordered(ms)[position]
    payload = _load_payload(target)
    payload["seed_start"], payload["seed_count"] = span
    payload["survivors"].append([seed, rate, None, [2]])
    return _swap(ms, target, _repoint(target, tmp, payload, tag)), target


def _put_intra_dup(ms, tmp, position, rate, tag):
    """The spool at ordered `position` emits its OWN first seed twice."""
    target = _ordered(ms)[position]
    payload = _load_payload(target)
    seed = payload["survivors"][0][0]
    payload["survivors"].append([seed, rate, None, [3]])
    return _swap(ms, target, _repoint(target, tmp, payload, tag)), target


def _assert_equivalent_exception(a, b, label):
    """REV2 §4's equivalence contract for exceptions: class, `.args`, rendered
    message and custom attribution. Traceback frames and backend-internal
    chaining are explicitly NOT contractual, so they are not compared."""
    assert type(a).__name__ == type(b).__name__, (
        f"{label}: class {type(a).__name__} != {type(b).__name__}")
    assert a.args == b.args, f"{label}: args {a.args!r} != {b.args!r}"
    assert str(a) == str(b), f"{label}: message\n  {a}\n!=\n  {b}"
    if type(a).__name__ == "DirectionalDuplicateError":
        for attr in ORACLE_DUP_ATTRS:
            av, bv = getattr(a, attr), getattr(b, attr)
            assert av == bv, f"{label}: attribution {attr} {av!r} != {bv!r}"
            assert av is not None, f"{label}: attribution {attr} is unpopulated"


def _observe(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception as exc:                                    # noqa: BLE001
        return exc
    raise AssertionError("the trial assembled successfully — no defect surfaced")


def _run_matrix_row(label, run_id, ms, expect_class, attribution=None):
    """Run ONE row against all three targets and require an equivalent
    exception from each."""
    ref = _observe(_reference, copy.deepcopy(ms), run_id)
    ser = _observe(_serial, copy.deepcopy(ms), run_id)
    par = _observe(_sharded, copy.deepcopy(ms), run_id)
    for name, exc in (("pre-D5 reference", ref), ("serial_reference", ser),
                      ("process_sharded", par)):
        assert type(exc).__name__ == expect_class, (
            f"{label}/{name}: expected {expect_class}, got "
            f"{type(exc).__name__}: {exc}")
    _assert_equivalent_exception(ref, ser, f"{label}: pre-D5 vs serial")
    _assert_equivalent_exception(ref, par, f"{label}: pre-D5 vs process_sharded")
    # serial and sharded come from the SAME live module, so there class
    # identity IS assertable — and required.
    assert type(ser) is type(par), (label, type(ser), type(par))
    if attribution:
        for attr, want in attribution.items():
            for name, exc in (("pre-D5", ref), ("serial", ser),
                              ("sharded", par)):
                got = getattr(exc, attr)
                assert got == want, (
                    f"{label}/{name}: attribution {attr} {got!r} != {want!r}")
    return ref, ser, par


def _row_dup_then_malformed(tmp, run_id):
    """EARLIER duplicate (position 1) + LATER malformed (position 15)."""
    ms = _build_run(tmp, AssemblingPhase5Sink(), run_id)
    ms, dup_target = _put_cross_dup(ms, tmp, 1, 1, 0.11, f"{run_id}_dup")
    ms, _ = _put_malformed(ms, tmp, 15, f"{run_id}_bad")
    return ms, dup_target


def _row_malformed_then_dup(tmp, run_id):
    """EARLIER malformed (position 0) + LATER duplicate (position 3 vs 2)."""
    ms = _build_run(tmp, AssemblingPhase5Sink(), run_id)
    ms, _ = _put_malformed(ms, tmp, 0, f"{run_id}_bad")
    ms, _ = _put_cross_dup(ms, tmp, 3, 25, 0.33, f"{run_id}_dup", span=(20, 20))
    return ms


def _row_intradup_then_malformed(tmp, run_id):
    """EARLIER intra-spool duplicate (position 0) + LATER malformed (15)."""
    ms = _build_run(tmp, AssemblingPhase5Sink(), run_id)
    ms, dup_target = _put_intra_dup(ms, tmp, 0, 0.22, f"{run_id}_dup")
    ms, _ = _put_malformed(ms, tmp, 15, f"{run_id}_bad")
    return ms, dup_target


def g_precedence_matrix():
    # ---- row 1: duplicate BEFORE malformed -> the duplicate wins ------------
    # This is the row the whole REV2 rework exists for: read-all-then-merge
    # reports the malformed spool here, because it reads position 15 before
    # merging position 1. Interleaved (and replayed) assembly reports the
    # duplicate.
    with tempfile.TemporaryDirectory() as tmp:
        ms, dup_target = _row_dup_then_malformed(tmp, "mx1")
        _run_matrix_row("row1 dup->malformed", "mx1", ms,
                        "DirectionalDuplicateError",
                        {"seed": 1, "direction": "forward",
                         "skip_mode": "constant", "workflow_phase": 1,
                         "first_sub_index": 0, "first_match_rate": 0.90,
                         "dup_stripe": dup_target["stripe_id"],
                         "dup_sub_index": 1, "dup_match_rate": 0.11})

    # ---- row 2: malformed BEFORE duplicate -> the malformed wins ------------
    with tempfile.TemporaryDirectory() as tmp:
        ms = _row_malformed_then_dup(tmp, "mx2")
        _, ser, _ = _run_matrix_row("row2 malformed->dup", "mx2", ms,
                                    "SpoolIdentityError")
        assert "not-my-stripe" in str(ser), str(ser)

    # ---- row 3: intra-spool duplicate BEFORE malformed ---------------------
    with tempfile.TemporaryDirectory() as tmp:
        ms, dup_target = _row_intradup_then_malformed(tmp, "mx3")
        _run_matrix_row("row3 intra-dup->malformed", "mx3", ms,
                        "DirectionalDuplicateError",
                        {"seed": 1, "first_stripe": dup_target["stripe_id"],
                         "dup_stripe": dup_target["stripe_id"],
                         "first_sub_index": 0, "dup_sub_index": 0,
                         "first_match_rate": 0.90, "dup_match_rate": 0.22})

    # ---- row 4: malformed A BEFORE malformed B -> A wins -------------------
    # Reuses the expensive-vs-trivial construction: A is ~60k survivors that
    # must all be validated before the last one fails, B fails on the size
    # check before a byte is parsed. B therefore finishes first under any
    # concurrency, so only a position-ordered consumer reports A.
    with tempfile.TemporaryDirectory() as tmp:
        ms, early, late = _malformed_dual_fixture(tmp, "mx4")
        _, ser, par = _run_matrix_row("row4 malformedA->malformedB", "mx4", ms,
                                      "SpoolIdentityError")
        assert "skip_sequence is str" in str(par), str(par)
        assert early["stripe_id"] in str(par) and late["stripe_id"] not in str(par)

    # ---- row 5: valid THEN duplicate -> the duplicate attribution ----------
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "mx5")
        ms, dup_target = _put_cross_dup(ms, tmp, 1, 1, 0.11, "mx5_dup")
        _run_matrix_row("row5 valid->dup", "mx5", ms,
                        "DirectionalDuplicateError",
                        {"seed": 1, "first_sub_index": 0,
                         "first_match_rate": 0.90,
                         "dup_stripe": dup_target["stripe_id"],
                         "dup_sub_index": 1, "dup_match_rate": 0.11})

    # ---- row 6: valid THEN malformed -> the malformed exception ------------
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "mx6")
        ms, bad_target = _put_malformed(ms, tmp, 15, "mx6_bad")
        _, ser, _ = _run_matrix_row("row6 valid->malformed", "mx6", ms,
                                    "SpoolIdentityError")
        assert bad_target["stripe_id"] in str(ser), str(ser)


# ═════════════════════════════════════════════════════════════════════════════
# G-SEED-DOMAIN [REV3 §5] — the OUT-OF-RANGE seed domain, vs the pre-D5 oracle
# ═════════════════════════════════════════════════════════════════════════════
#
# THE GAP THIS CLOSES. Every gate above this one runs seeds that fit in an
# int64, so all of them stayed green while the first D5 projection silently
# narrowed the engine's ACCEPTED INPUT: §5.3 bounds a seed only to the declared
# window `[seed_start, seed_start + seed_count)`, with no signed-64 bound and no
# non-negativity requirement on `seed_start`, and the pre-D5 maps keyed on
# arbitrary-precision Python ints. A spool declaring `seed_start = 2**63` was
# therefore ACCEPTED before D5 and raised `OverflowError` after it. That is a
# valid-input divergence, and it is invisible to a test domain that never leaves
# int64 — the same blind spot that previously hid the precedence divergence.
#
# So these six rows are all run against the PRE-D5 ORACLE, not merely
# backend-vs-backend: two backends that narrow the domain identically would
# agree with each other perfectly and still be wrong.
#
# "Unreachable for java_lcg" is deliberately NOT accepted as a defence: the
# engine is base-parameterized (`prng_base` drives the family map), not
# contractually one-family, and the validator that admits these seeds is the
# generic one.
_SEED_DOMAIN_SPAN = 2 ** 71          # a window wide enough for every row below


def _retarget_spool(ms, tmp, position, seed_start, seed_count, survivors, tag):
    """Give the spool at ordered `position` a new declared window and a new
    survivor list, RECOMPUTING size + sha so the validator (not the digest
    check) is what the row exercises. Returns (manifests, the new manifest)."""
    target = _ordered(ms)[position]
    payload = _load_payload(target)
    payload["seed_start"], payload["seed_count"] = seed_start, seed_count
    payload["survivors"] = survivors
    replacement = _repoint(target, tmp, payload, tag)
    return _swap(ms, target, replacement), replacement


def _armed_encoding(manifest, run_id):
    """WHICH encoding the live projection layer armed for one spool.

    Asserted only as a structural property of the projection — never as an
    output oracle. The output oracle is always the pre-D5 engine."""
    return RMW.read_and_validate_spool(run_id, manifest).seed_encoding


def _three_targets(ms, run_id, label):
    """Assemble one input under all three targets and require field-for-field
    equality with the PRE-D5 oracle. Returns the labelled assemblies."""
    ref = _reference(copy.deepcopy(ms), run_id)
    ser = _serial(copy.deepcopy(ms), run_id)
    par = _sharded(copy.deepcopy(ms), run_id)
    _assert_equivalent(ser, ref, f"{label}: serial_reference vs pre-D5 oracle")
    _assert_equivalent(par, ref, f"{label}: process_sharded vs pre-D5 oracle")
    return (("pre-D5", ref), ("serial", ser), ("sharded", par))


def _assert_seed_key(targets, attr, seed, rate, label):
    """The seed is a key of the named map, with the right rate, and the key is a
    PYTHON int under every target — `np.int64` is equal and equal-hashing but is
    not the pre-D5 contract, and would leak a numpy scalar into every canonical
    record and every D6 consumer."""
    for name, assembly in targets:
        population = getattr(assembly, attr)
        assert seed in population, (
            f"{label}/{name}: seed {seed} is absent from {attr} "
            f"(present: {sorted(population)[:4]})")
        assert population[seed] == rate, (
            f"{label}/{name}: {attr}[{seed}] = {population[seed]!r} != {rate!r}")
        key = next(k for k in population if k == seed)
        assert type(key) is int, (
            f"{label}/{name}: the {attr} key for {seed} is "
            f"{type(key).__name__}, not a Python int")


def g_seed_domain():
    # ---- row 1: seed = 2**63 - 1 (max int64) — stays on the FAST path -------
    with tempfile.TemporaryDirectory() as tmp:
        seed = 2 ** 63 - 1
        ms = _build_run(tmp, AssemblingPhase5Sink(), "dom1")
        ms, repl = _retarget_spool(ms, tmp, 0, seed - 5, 10,
                                   [[seed, 0.90, None, [1]]], "dom1")
        assert _armed_encoding(repl, "dom1") == "int64", (
            "the maximum signed-64 seed must not need the lossless fallback — "
            "the fast path is supposed to cover the whole int64 domain")
        targets = _three_targets(ms, "dom1", "row1 max-int64")
        _assert_seed_key(targets, "forward_map_constant", seed, 0.90, "row1")

    # ---- row 2: seed = 2**63 — the exact value pre-D5 accepted and the -----
    #      first D5 projection rejected. This row IS the blocker.
    with tempfile.TemporaryDirectory() as tmp:
        seed = 2 ** 63
        ms = _build_run(tmp, AssemblingPhase5Sink(), "dom2")
        ms, repl = _retarget_spool(ms, tmp, 0, seed - 5, 10,
                                   [[seed, 0.90, None, [1]]], "dom2")
        assert _armed_encoding(repl, "dom2") == "signed_bytes", (
            "a seed one past int64 must arm the lossless fallback, not raise")
        targets = _three_targets(ms, "dom2", "row2 2**63")
        _assert_seed_key(targets, "forward_map_constant", seed, 0.90, "row2")

    # ---- row 3: NEGATIVE seeds inside a NEGATIVE window (seed_start < 0) ----
    #      The validator imposes no non-negativity on seed_start, so this is
    #      accepted input. The oversized negative arms the fallback; the SMALL
    #      negative alongside it must still decode as a negative number, which
    #      is what an unsigned `int.from_bytes` gets wrong.
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "dom3")
        ms, repl = _retarget_spool(
            ms, tmp, 0, -(2 ** 70), _SEED_DOMAIN_SPAN,
            [[-25, 0.90, None, [1]], [-(2 ** 69), 0.80, None, [2]]], "dom3")
        assert _armed_encoding(repl, "dom3") == "signed_bytes", "fallback"
        targets = _three_targets(ms, "dom3", "row3 negative-window")
        _assert_seed_key(targets, "forward_map_constant", -25, 0.90, "row3")
        _assert_seed_key(targets, "forward_map_constant", -(2 ** 69), 0.80, "row3")

    # ---- row 4: a seed LARGER THAN 64 BITS ---------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        seed = 2 ** 70
        ms = _build_run(tmp, AssemblingPhase5Sink(), "dom4")
        ms, repl = _retarget_spool(ms, tmp, 0, 0, _SEED_DOMAIN_SPAN,
                                   [[seed, 0.90, None, [1]]], "dom4")
        assert _armed_encoding(repl, "dom4") == "signed_bytes", "fallback"
        targets = _three_targets(ms, "dom4", "row4 2**70")
        _assert_seed_key(targets, "forward_map_constant", seed, 0.90, "row4")

    # ---- row 5: MIXED small and oversized in ONE spool ---------------------
    #      The encoding is a property of the whole spool, so the small seeds are
    #      encoded too — minimally — and must come back unchanged.
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "dom5")
        mixed = [[3, 0.91, None, [1]], [2 ** 70, 0.92, None, [2]],
                 [0, 0.93, None, [3]], [9, 0.94, None, [1]],
                 [2 ** 63, 0.95, None, [2]], [127, 0.96, None, [3]]]
        ms, repl = _retarget_spool(ms, tmp, 0, 0, _SEED_DOMAIN_SPAN, mixed,
                                   "dom5")
        assert _armed_encoding(repl, "dom5") == "signed_bytes", (
            "one oversized seed must switch the WHOLE spool to the fallback")
        targets = _three_targets(ms, "dom5", "row5 mixed")
        for seed, rate in ((3, 0.91), (2 ** 70, 0.92), (0, 0.93), (9, 0.94),
                           (2 ** 63, 0.95), (127, 0.96)):
            _assert_seed_key(targets, "forward_map_constant", seed, rate, "row5")

    # ---- row 6: DUPLICATE ATTRIBUTION involving oversized seeds ------------
    #      The §5.4 duplicate invariant, the message and all 13 attribution
    #      fields must be identical to the pre-D5 oracle's with a big-int seed.
    with tempfile.TemporaryDirectory() as tmp:
        seed = 2 ** 70
        ms = _build_run(tmp, AssemblingPhase5Sink(), "dom6")
        ms, first = _retarget_spool(ms, tmp, 0, 0, _SEED_DOMAIN_SPAN,
                                    [[seed, 0.90, None, [1]]], "dom6_first")
        ms, dup_target = _put_cross_dup(ms, tmp, 1, seed, 0.11, "dom6_dup",
                                        span=(0, _SEED_DOMAIN_SPAN))
        _run_matrix_row("row6 oversized-duplicate", "dom6", ms,
                        "DirectionalDuplicateError",
                        {"seed": seed, "direction": "forward",
                         "skip_mode": "constant", "workflow_phase": 1,
                         "first_stripe": first["stripe_id"],
                         "first_sub_index": 0, "first_match_rate": 0.90,
                         "dup_stripe": dup_target["stripe_id"],
                         "dup_sub_index": 1, "dup_match_rate": 0.11})


# ═════════════════════════════════════════════════════════════════════════════
# G-SEED-CODEC [REV3 §2] — the deterministic signed-byte length formula at its boundaries
# ═════════════════════════════════════════════════════════════════════════════
def g_seed_encoding():
    """`(bit_length // 8) + 1` is the minimal SIGNED byte length, and the ±2^(8k-1)
    values are exactly where the plausible-looking `(bit_length + 7) // 8`
    silently raises OverflowError — the off-by-one REV3 called out by name.

    The expectations are hand-written: a list of boundary values and the
    requirement that each survives build -> decode unchanged, with the encoding
    chosen by the signed-64 rule and nothing else."""
    boundaries: List[int] = [0, 1, -1, 127, 128, 129, -127, -128, -129,
                             255, 256, -255, -256, 32767, 32768, -32768, -32769,
                             2 ** 31, -(2 ** 31), 2 ** 55 - 1,
                             2 ** 63 - 1, -(2 ** 63),
                             2 ** 63, -(2 ** 63) - 1, 2 ** 63 + 1,
                             2 ** 64, -(2 ** 64), 2 ** 70, -(2 ** 70),
                             2 ** 71 - 1, 2 ** 200, -(2 ** 200)]
    # every value survives the encode/decode round trip ON ITS OWN spool
    for value in boundaries:
        proj = RMW.build_validated_projection([value], np.array([0.5]))
        assert _oracle_decode_seeds(proj) == [value], (
            f"{value} did not round-trip: {_oracle_decode_seeds(proj)}")
        want = ("int64" if -(2 ** 63) <= value <= 2 ** 63 - 1
                else "signed_bytes")
        assert proj.seed_encoding == want, (
            f"{value}: encoding {proj.seed_encoding!r}, expected {want!r}")

    # ...and all of them together in ONE spool, order and multiplicity intact
    together = boundaries + boundaries[::-1]
    proj = RMW.build_validated_projection(
        together, np.zeros(len(together), dtype=np.float64))
    assert proj.seed_encoding == "signed_bytes", proj.seed_encoding
    assert _oracle_decode_seeds(proj) == together, "order or multiplicity lost"

    # the FAST path is taken only when EVERY seed fits, and its keys are ints
    fast = RMW.build_validated_projection(
        [-(2 ** 63), 0, 2 ** 63 - 1], np.zeros(3))
    assert fast.seed_encoding == "int64", fast.seed_encoding
    assert _oracle_decode_seeds(fast) == [-(2 ** 63), 0, 2 ** 63 - 1]
    assert all(type(s) is int for s in RMW.projection_seeds(fast))
    assert all(type(s) is int for s in RMW.projection_seeds(proj))

    # a projection may never carry BOTH representations, or neither
    for kwargs in (
        dict(seed_encoding="int64", seeds_i64=np.zeros(1, dtype=np.int64),
             seed_bytes=np.zeros(1, dtype=np.uint8),
             seed_offsets=np.array([0, 1], dtype=np.uint64)),
        dict(seed_encoding="int64", seeds_i64=None, seed_bytes=None,
             seed_offsets=None),
        dict(seed_encoding="signed_bytes", seeds_i64=np.zeros(1, dtype=np.int64),
             seed_bytes=np.zeros(1, dtype=np.uint8),
             seed_offsets=np.array([0, 1], dtype=np.uint64)),
        dict(seed_encoding="varint", seeds_i64=np.zeros(1, dtype=np.int64),
             seed_bytes=None, seed_offsets=None),
        # offsets that do not span the byte run, and a zero-length seed row
        dict(seed_encoding="signed_bytes", seeds_i64=None,
             seed_bytes=np.zeros(3, dtype=np.uint8),
             seed_offsets=np.array([0, 1], dtype=np.uint64)),
        dict(seed_encoding="signed_bytes", seeds_i64=None,
             seed_bytes=np.zeros(0, dtype=np.uint8),
             seed_offsets=np.array([0, 0], dtype=np.uint64)),
    ):
        _raises(ValueError, ValidatedSpoolProjection,
                match_rates=np.zeros(1, dtype=np.float64), survivor_count=1,
                **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
# G-SERIAL-ORIGINAL [REV2 §6] — serial raises the ORIGINAL exception object;
# process_sharded raises a reconstructed-but-equivalent one
# ═════════════════════════════════════════════════════════════════════════════
def _deepest_frame(exc):
    tb = exc.__traceback__
    name = None
    while tb is not None:
        name = tb.tb_frame.f_code.co_name
        tb = tb.tb_next
    return name


def g_serial_original():
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "orig")
        ms, _ = _put_malformed(ms, tmp, 15, "orig_bad")

        # record the exact object the SHARED validator raises, and re-raise it
        # unchanged, so the identity of what escapes `assemble_trial` is
        # observable.
        raised = []
        real = RMW.read_and_validate_spool

        def recording(run_id, manifest):
            try:
                return real(run_id, manifest)
            except SpoolIdentityError as exc:
                raised.append(exc)
                raise

        RMW.read_and_validate_spool = recording
        try:
            serial_exc = _observe(_serial, copy.deepcopy(ms), "orig")
        finally:
            RMW.read_and_validate_spool = real

        assert len(raised) == 1, (
            f"the serial front end raised {len(raised)} read defects — it must "
            f"stop at the FIRST one, lazily")
        assert serial_exc is raised[0], (
            "serial_reference did not propagate the ORIGINAL exception object — "
            "it round-tripped through a descriptor, which REV2 §2 forbids for "
            "the serial path")
        # ... and the ORIGINAL traceback: the deepest frame is the validator's
        # own failure helper, not a replay site.
        assert _deepest_frame(serial_exc) == "_fail", _deepest_frame(serial_exc)

        # process_sharded raises an EQUIVALENT but RECONSTRUCTED exception: same
        # class, args, message; raised from the canonical replay site.
        sharded_exc = _observe(_sharded, copy.deepcopy(ms), "orig")
        _assert_equivalent_exception(serial_exc, sharded_exc, "serial vs sharded")
        assert sharded_exc is not serial_exc
        assert _deepest_frame(sharded_exc) == "raise_captured_spool_error", (
            f"process_sharded's exception was not produced by the canonical "
            f"replay site (deepest frame {_deepest_frame(sharded_exc)!r})")

    # the serial path must not even be able to produce a descriptor: nothing in
    # the serial composition captures one.
    src = _WRITER_SRC
    assert "capture_spool_read_error(" in src, "the capture helper is missing"
    tree = ast.parse(src, filename=_WRITER_PATH)
    for fname in ("_serial_outcomes", "assemble_trial"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == fname)
        for node in ast.walk(fn):
            assert not (isinstance(node, ast.Call)
                        and getattr(node.func, "id", None)
                        == "capture_spool_read_error"), (
                f"{fname} captures a read error — serial must raise the "
                f"original object")
            assert not isinstance(node, (ast.Try, ast.ExceptHandler)), (
                f"{fname} swallows or re-wraps an exception")


# ═════════════════════════════════════════════════════════════════════════════
# G-DESCRIPTOR [REV2 §1] — the captured-read-error round trip and its allowlist
# ═════════════════════════════════════════════════════════════════════════════
def _detect_descriptor_roundtrip(module):
    """A canonical spool-read defect must round-trip class, `.args`, rendered
    message AND custom attribution. Attribution is carried generically so a
    future member of the SpoolIdentityError hierarchy round-trips losslessly;
    today's canonical defect carries none of its own, so the mechanism is
    exercised with one set explicitly."""
    exc = module.SpoolIdentityError("run r s0/sub0 ('/p.json'): boom")
    exc.stripe_id = "s0"                 # a custom attribution field
    exc.sub_index = 3
    descriptor = module.capture_spool_read_error(exc)
    assert descriptor.error_code == "SpoolIdentityError", descriptor
    assert descriptor.args == exc.args, descriptor
    assert descriptor.message == str(exc), descriptor

    try:
        module.raise_captured_spool_error(descriptor)
    except BaseException as replayed:                           # noqa: BLE001
        got = replayed
    else:
        raise AssertionError("raise_captured_spool_error returned instead of raising")

    assert type(got).__name__ == "SpoolIdentityError", (
        f"replay produced {type(got).__name__}: {got}")
    assert got.args == exc.args, (got.args, exc.args)
    assert str(got) == str(exc), (str(got), str(exc))
    assert getattr(got, "stripe_id", None) == "s0", (
        "the descriptor dropped a custom attribution field")
    assert getattr(got, "sub_index", None) == 3, (
        "the descriptor dropped a custom attribution field")
    assert got is not exc, "a descriptor must not carry the live exception"


def g_descriptor():
    _detect_descriptor_roundtrip(RMW)

    # ---- the ALLOWLIST is the whole point: only canonical producer defects --
    assert tuple(RMW.CANONICAL_SPOOL_READ_ERRORS) == ("SpoolIdentityError",), \
        RMW.CANONICAL_SPOOL_READ_ERRORS
    dup = DirectionalDuplicateError(
        "d", run_id="r", workflow_phase=1, direction="forward",
        skip_mode="constant", seed=1, first_stripe="a", first_sub_index=0,
        first_attempt=0, first_match_rate=0.1, dup_stripe="b", dup_sub_index=1,
        dup_attempt=0, dup_match_rate=0.2)
    for refused in (MemoryError("oom"), KeyboardInterrupt(), SystemExit(1),
                    RuntimeError("boom"), ASW.ShardArtifactError("artifact"),
                    ASW.ProcessShardedAssemblyError("pool"), dup,
                    AssemblyConsistencyError("meta"), PhaseIdentityError("id")):
        _raises(TypeError, RMW.capture_spool_read_error, refused)
    # and replay refuses a descriptor naming a non-allowlisted class
    forged = RMW.CapturedSpoolReadError(
        error_code="MemoryError", message="oom", args=("oom",), attributes={})
    _raises(TypeError, RMW.raise_captured_spool_error, forged)
    # a descriptor that does not reproduce its own message is refused, not
    # silently raised with a different message
    infidel = RMW.CapturedSpoolReadError(
        error_code="SpoolIdentityError", message="not what it renders",
        args=("boom",), attributes={})
    _raises(ValueError, RMW.raise_captured_spool_error, infidel)

    # ---- it is DATA, not a pickled exception --------------------------------
    import pickle
    exc = RMW.SpoolIdentityError("boom")
    descriptor = RMW.capture_spool_read_error(exc)
    blob = pickle.dumps(descriptor)
    assert len(blob) < 4096, len(blob)
    back = pickle.loads(blob)
    assert back == descriptor, (back, descriptor)
    assert isinstance(back, RMW.CapturedSpoolReadError)
    # non-scalar state can never enter one
    weird = RMW.SpoolIdentityError("boom")
    weird.payload = {"survivors": [[1, 0.5, None, [1]]]}
    _raises(TypeError, RMW.capture_spool_read_error, weird)


# ═════════════════════════════════════════════════════════════════════════════
# G-BACKEND-DISTINCT [REV2 §5] — backend failures are NEVER producer defects
# ═════════════════════════════════════════════════════════════════════════════
# A worker that dies of MemoryError is injected by loading a copy of the worker
# module with an unconditional raise for phase-4 shards. `spawn` children import
# the mutant by name from the mutant dir on sys.path, so the injection really
# runs IN THE CHILD.
_INJECT_ANCHOR = ('    assert_cpu_only()\n'
                  '    run_id = task["run_id"]\n')
_INJECT_BACKEND_FAILURE = (
    '    assert_cpu_only()\n'
    '    if int(task["manifest"].get("workflow_phase", 0)) == 4:\n'
    '        raise MemoryError("injected: worker died in the process pool")\n'
    '    run_id = task["run_id"]\n')


def _injected_worker_module(extra_patches=()):
    src = _patch(_WORKER_SRC, _INJECT_ANCHOR, _INJECT_BACKEND_FAILURE, "INJECT")
    for old, new, label in extra_patches:
        src = _patch(src, old, new, label)
    return _load_mutant(src, "INJECTED")


def _detect_backend_failure_distinct(module):
    """A dead worker must raise a BACKEND error, never a producer defect."""
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "bkfail")
        exc = _observe(_sharded, copy.deepcopy(ms), "bkfail", 4, module)
    assert type(exc).__name__ == "ProcessShardedAssemblyError", (
        f"a crashed worker surfaced as {type(exc).__name__}: {exc} — an "
        f"infrastructure failure must never masquerade as a producer defect")
    assert not isinstance(exc, SpoolIdentityError), exc
    assert "MemoryError" in str(exc), str(exc)
    assert isinstance(exc.__cause__, MemoryError), exc.__cause__


def _detect_canonical_beats_backend(module):
    """REV2 §7: an EARLIER canonical defect stays the primary exception even
    when a LATER position's worker died."""
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "bkprec")
        ms, dup_target = _put_cross_dup(ms, tmp, 1, 1, 0.11, "bkprec_dup")
        exc = _observe(_sharded, copy.deepcopy(ms), "bkprec", 4, module)
    assert type(exc).__name__ == "DirectionalDuplicateError", (
        f"a LATER backend failure pre-empted an EARLIER producer defect: "
        f"{type(exc).__name__}: {exc}")
    assert exc.seed == 1 and exc.dup_sub_index == 1, (exc.seed, exc.dup_sub_index)
    assert exc.dup_stripe == dup_target["stripe_id"], exc.dup_stripe


def g_backend_distinct():
    # the two error families are structurally separate types
    assert issubclass(ASW.ShardArtifactError, ASW.ProcessShardedAssemblyError)
    assert not issubclass(ASW.ProcessShardedAssemblyError, SpoolIdentityError)
    assert not issubclass(SpoolIdentityError, ASW.ProcessShardedAssemblyError)

    injected = _injected_worker_module()
    _detect_backend_failure_distinct(injected)
    _detect_canonical_beats_backend(injected)

    # cleanup still holds when the failure is a BACKEND one
    before = _tempdirs()
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "bkclean")
        _observe(_sharded, ms, "bkclean", 4, injected)
    assert _tempdirs() == before, (
        f"a backend failure leaked {set(_tempdirs()) - set(before)}")

    # a malformed OUTCOME is a backend failure too, and is caught before it can
    # be replayed as a producer defect
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "envelope")
        manifest = ms[0]
        for envelope in (None, {}, {"outcome_kind": "nonsense"},
                         {"outcome_kind": "read_error", "descriptor": "text"},
                         {"outcome_kind": "read_error",
                          "descriptor": ASW.CapturedSpoolReadError(
                              error_code="MemoryError", message="oom",
                              args=("oom",), attributes={})},
                         {"outcome_kind": "projection", "result": {"x": 1}}):
            e = _raises(ASW.ProcessShardedAssemblyError,
                        ASW._materialize_outcome, "envelope", 0, manifest,
                        envelope)
            assert not isinstance(e, SpoolIdentityError), e

    # ...and the classifier keeps producer defects and backend failures apart
    outcome, failure = ASW._capture_worker_exception(SpoolIdentityError("boom"))
    assert failure is None and outcome["outcome_kind"] == ASW.OUTCOME_READ_ERROR
    assert isinstance(outcome["descriptor"], ASW.CapturedSpoolReadError)
    outcome, failure = ASW._capture_worker_exception(MemoryError("oom"))
    assert outcome is None and isinstance(failure, MemoryError)


# ═════════════════════════════════════════════════════════════════════════════
# G-SPAWN — the pool uses spawn; fork is refused outright
# ═════════════════════════════════════════════════════════════════════════════
def _probe_child(_ignored):
    """Runs INSIDE a pool worker: report the child's start method and whether
    any GPU library reached it after the worker module is imported."""
    import miner.assembly_shard_worker as W       # noqa: F401  (import for effect)
    return {
        "start_method": multiprocessing.get_start_method(allow_none=True),
        "gpu_modules": sorted(m for m in ("torch", "cupy") if m in sys.modules),
        "pid": os.getpid(),
    }


def _run_probe(start_method="spawn", n=4):
    import concurrent.futures
    ctx = ASW._resolve_context(start_method)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n,
                                                mp_context=ctx) as ex:
        return list(ex.map(_probe_child, range(n)))


def _detect_spawn(module=ASW):
    # 1. the DEFAULT is spawn, declared in the signature — not merely a habit
    import inspect
    sig = inspect.signature(module.run_sharded_assembly)
    assert sig.parameters["start_method"].default == "spawn", sig
    # 2. fork is refused OUTRIGHT, and an unknown method is too — never a
    #    silent downgrade
    e = _raises(ValueError, module._resolve_context, "fork")
    assert "fork" in str(e), str(e)
    _raises(ValueError, module._resolve_context, "threads")
    # 3. the context actually resolved is spawn
    assert module._resolve_context("spawn").get_start_method() == "spawn"
    # 4. a REAL run reports spawn
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "spawn")
        outcome = module.run_sharded_assembly("spawn", ms, 2)
        assert outcome.start_method == "spawn", outcome.start_method
    # 5. and the CHILD agrees it was spawned
    probes = _run_probe("spawn", 2)
    for p in probes:
        assert p["start_method"] == "spawn", p
        assert p["pid"] != os.getpid(), "a worker ran in the parent process"


def g_spawn():
    _detect_spawn()


# ═════════════════════════════════════════════════════════════════════════════
# G-NO-GPU — no worker process holds a GPU library
# ═════════════════════════════════════════════════════════════════════════════
def g_no_gpu():
    # 1. measured in a REAL spawned child, after importing the worker module
    for probe in _run_probe("spawn", 4):
        assert probe["gpu_modules"] == [], (
            f"a spawned assembly worker imported {probe['gpu_modules']} — "
            f"assembly is CPU-only work (§6.7.A)")
    # 2. the guard is not decorative: it FIRES when a GPU module is present
    sentinel = object()
    injected = "torch" not in sys.modules
    if injected:
        sys.modules["torch"] = sentinel                  # type: ignore[assignment]
    try:
        e = _raises(ASW.ShardArtifactError, ASW.assert_cpu_only)
        assert "torch" in str(e), str(e)
    finally:
        if injected:
            del sys.modules["torch"]
    ASW.assert_cpu_only()          # negative control: clean process passes
    # 3. the guard is the FIRST statement of the worker task, so it cannot be
    #    reached only after a GPU context has already been built
    tree = ast.parse(_WORKER_SRC, filename=_WORKER_PATH)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "validate_spool_shard")
    body = [s for s in fn.body if not (isinstance(s, ast.Expr)
                                       and isinstance(s.value, ast.Constant))]
    first = body[0]
    assert (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
            and getattr(first.value.func, "id", None) == "assert_cpu_only"), \
        ast.dump(first)
    # 4. neither module imports a GPU library at module scope
    for src, path in ((_WORKER_SRC, _WORKER_PATH), (_BACKEND_SRC, _BACKEND_PATH)):
        for node in ast.walk(ast.parse(src, filename=path)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert not ({"torch", "cupy"} & set(names)), (path, names)


# ═════════════════════════════════════════════════════════════════════════════
# G-NO-PAYLOAD-IPC — the four §6.7.A prohibitions, each proven absent
# ═════════════════════════════════════════════════════════════════════════════
def g_no_payload_ipc():
    tree = ast.parse(_WORKER_SRC, filename=_WORKER_PATH)

    # ---- prohibition 4: pool size is EXPLICIT, never os.cpu_count() --------
    # AST, not substring: both modules DOCUMENT the prohibition in prose, and a
    # comment naming `os.cpu_count()` is the opposite of a violation. What must
    # be absent is any actual reference to it in code.
    for src, path in ((_WORKER_SRC, _WORKER_PATH), (_BACKEND_SRC, _BACKEND_PATH)):
        for node in ast.walk(ast.parse(src, filename=path)):
            assert not (isinstance(node, ast.Name) and node.id == "cpu_count"), \
                f"{path}:{node.lineno} references cpu_count"
            assert not (isinstance(node, ast.Attribute)
                        and node.attr == "cpu_count"), (
                f"{path}:{node.lineno} sizes something from cpu_count — '24 "
                f"processes because Zeus exposes 24 threads' is a §6.7.A "
                f"prohibition")
    import inspect
    sig = inspect.signature(ASW.run_sharded_assembly)
    assert sig.parameters["pool_size"].default is inspect.Parameter.empty, (
        "pool_size must be REQUIRED — a default is a guess")
    bsig = inspect.signature(AB.ProcessShardedBackend.__init__)
    assert bsig.parameters["pool_size"].default is inspect.Parameter.empty, bsig
    # and it is validated, not merely accepted
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "poolsz")
        for bad in (0, -1, None, "4", True, 2.0):
            _raises(ValueError, ASW.run_sharded_assembly, "poolsz", ms, bad)

    # ---- prohibitions 1-3, at RUNTIME: what the worker actually returns ----
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "ipc")
        art_dir = os.path.join(tmp, "arts")
        os.makedirs(art_dir)
        envelope = ASW.validate_spool_shard(
            {"run_id": "ipc", "manifest": ms[0], "artifact_dir": art_dir})
        assert tuple(envelope) == ORACLE_OUTCOME_KEYS, tuple(envelope)
        assert envelope["outcome_kind"] == ORACLE_OUTCOME_PROJECTION, envelope
        result = envelope["result"]
        assert tuple(result) == ORACLE_RESULT_KEYS, tuple(result)
        for key, value in result.items():
            assert isinstance(value, (str, int)), (
                f"worker returned {key}={type(value).__name__} — the result is "
                f"paths and counts only")
            assert not isinstance(value, (np.ndarray, np.generic)), (key, value)
            assert not isinstance(value, (list, dict, tuple, set)), (key, value)
        # it is picklable and SMALL: no payload, no arrays, no survivor dicts
        import pickle
        blob = pickle.dumps(envelope)
        assert len(blob) < 4096, (
            f"worker result pickles to {len(blob)} bytes — payload is leaking "
            f"through the IPC channel")
        assert b"survivors" not in blob and b"match_rate" not in blob, blob[:200]

        # the OTHER outcome kind — a captured canonical read defect — is data
        # of the same shape and size class: a class name, a message, scalar
        # args and scalar attribution. No payload rides along with it.
        payload = _load_payload(ms[0])
        payload["stripe_id"] = "not-my-stripe"
        bad = _repoint(ms[0], tmp, payload, "ipcbad")
        err_envelope = ASW.validate_spool_shard(
            {"run_id": "ipc", "manifest": bad, "artifact_dir": art_dir})
        assert tuple(err_envelope) == ORACLE_READ_ERROR_KEYS, tuple(err_envelope)
        assert err_envelope["outcome_kind"] == ORACLE_OUTCOME_READ_ERROR
        descriptor = err_envelope["descriptor"]
        assert type(descriptor).__name__ == "CapturedSpoolReadError", descriptor
        assert descriptor.error_code == "SpoolIdentityError", descriptor
        eblob = pickle.dumps(err_envelope)
        assert len(eblob) < 4096, len(eblob)
        assert b"survivors" not in eblob, eblob[:200]
        # and no worker outcome is ever an exception INSTANCE crossing IPC
        assert not isinstance(descriptor, BaseException), descriptor

        # ---- prohibition 3: what the parent SENDS a child is one small
        #      manifest — never a parsed payload. Proven by pickling the exact
        #      task dict the parent builds.
        task = {"run_id": "ipc", "manifest": ms[0], "artifact_dir": art_dir}
        tblob = pickle.dumps(task)
        assert b"survivors" not in tblob, "a parsed payload is being sent to a worker"
        assert len(tblob) < 8192, len(tblob)

    # ---- prohibition 2: no ndarray crosses the IPC boundary ----------------
    # The arrays travel as an on-disk artifact whose PATH is returned. Prove the
    # worker function's return value is built only from the identity/scalar dict
    # by checking no ndarray survives a pickle round-trip of the result.
    assert not any(isinstance(v, np.ndarray)
                   for v in pickle.loads(blob).values())
    # and structurally: the worker has EXACTLY the two outcome returns, each a
    # dict literal over the frozen key tuple, so nothing can be smuggled out as
    # an extra key.
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "validate_spool_shard")
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert len(returns) == 2, (
        f"expected exactly two returns (projection | read_error), got "
        f"{len(returns)}")
    shapes = []
    for node in returns:
        assert isinstance(node.value, ast.Dict), ast.dump(node)
        keys = tuple(k.value for k in node.value.keys)
        assert keys in (ORACLE_OUTCOME_KEYS, ORACLE_READ_ERROR_KEYS), keys
        shapes.append(keys)
    assert sorted(shapes) == sorted([ORACLE_OUTCOME_KEYS,
                                     ORACLE_READ_ERROR_KEYS]), shapes


# ═════════════════════════════════════════════════════════════════════════════
# G-D4-INTACT — D5's edit to assembly_backends.py must not disturb D4
# ═════════════════════════════════════════════════════════════════════════════
def g_d4_intact():
    """D5 is the first deliverable to ADD code to a module D4 froze, so it is
    the first that can silently break D4's own gates from the inside.

    Two failure modes, both real and both hit during development:
      * D4's G8 mutates `assembly_backends.py` by UNIQUE source anchor and
        asserts each occurs exactly once — a second backend that copies
        `SerialReferenceBackend`'s step comments or expression spelling makes
        every one of those anchors ambiguous;
      * D4's G7 forbids this module from calling `open`, `hashlib.*`, `numpy.*`
        or `sorted`, or importing numpy/hashlib/json/subprocess at all.

    Asserting both HERE means the breakage surfaces in seconds instead of 25
    minutes into the non-regression run. These anchors are hand-transcribed from
    D4's harness, not imported from it."""
    anchors = (
        "    if name not in ASSEMBLY_BACKENDS:\n        raise ValueError(",
        "    if name == PROCESS_SHARDED:\n        raise NotImplementedError(",
        "        # 3. stop the timer on successful return only\n",
        "ASSEMBLY_BACKENDS = (SERIAL_REFERENCE, PROCESS_SHARDED)",
        "            wall_seconds=wall_seconds,",
        '            spool_bytes_read=sum(m["expected_size"] for m in manifests),',
        "        # 2. delegate",
    )
    for anchor in anchors:
        count = _BACKEND_SRC.count(anchor)
        assert count == 1, (
            f"D4 mutation anchor occurs {count}x (must be exactly 1) — D4's G8 "
            f"will red with 'anchor is not unique':\n  {anchor!r}")

    # D4's G7 forbidden CALLS, hand-transcribed
    forbidden_dotted = {
        "open", "hashlib.sha256", "hashlib.new",
        "numpy.array", "numpy.asarray", "numpy.empty", "numpy.zeros",
        "np.array", "np.asarray", "np.empty", "np.zeros",
        "numpy.savez", "numpy.savez_compressed",
        "np.savez", "np.savez_compressed", "sorted",
        "records_to_arrays", "build_mode_records", "finalize_run",
    }
    forbidden_tails = {"sort", "sha256", "savez", "savez_compressed",
                       "records_to_arrays", "build_mode_records", "finalize_run"}

    def _dotted(node):
        bits = []
        while isinstance(node, ast.Attribute):
            bits.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            bits.append(node.id)
            return ".".join(reversed(bits))
        return None

    for node in ast.walk(ast.parse(_BACKEND_SRC, filename=_BACKEND_PATH)):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name:
            assert name not in forbidden_dotted, (
                f"{_BACKEND_PATH}:{node.lineno} calls forbidden target {name!r} "
                f"— D4's G7 will red")
            assert name.rsplit(".", 1)[-1] not in forbidden_tails, (
                f"{_BACKEND_PATH}:{node.lineno} calls {name!r}")
        elif isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_tails, (
                f"{_BACKEND_PATH}:{node.lineno} calls .{node.func.attr}()")

    # D4's G7 forbidden module attributes
    for name in ("records_to_arrays", "build_mode_records", "finalize_run",
                 "np", "numpy", "hashlib", "json", "subprocess"):
        assert not hasattr(AB, name), (
            f"miner.assembly_backends must not import {name} — D4's G7 will red")
    assert hasattr(AB, "assemble_trial"), "the D1.1 entry point must be imported"

    # and D4's G4 contract still holds: name-only resolution refuses to guess
    e = _raises(NotImplementedError, AB.get_assembly_backend, "process_sharded")
    assert "D5" in str(e) and "process_sharded" in str(e), str(e)
    assert not isinstance(e, ValueError), (
        "process_sharded must raise NotImplementedError, not ValueError")
    # while an EXPLICITLY configured selection resolves to the real backend
    backend = AB.get_assembly_backend("process_sharded", pool_size=2)
    assert isinstance(backend, AB.ProcessShardedBackend), type(backend)
    assert backend.backend_name == "process_sharded"
    assert backend.pool_size == 2
    # ...and never degrades to serial
    assert not isinstance(backend, AB.SerialReferenceBackend)
    # serial takes no configuration
    _raises(ValueError, AB.get_assembly_backend, "serial_reference", pool_size=2)


# ═════════════════════════════════════════════════════════════════════════════
# G-CODEC — lossless, allow_pickle=False, no object arrays, uncompressed
# ═════════════════════════════════════════════════════════════════════════════
def _codec_identity():
    return {"run_id": "codec", "stripe_id": "s0", "sub_index": 1, "attempt": 0,
            "workflow_phase": 3, "direction": "forward", "skip_mode": "variable",
            "prng_type": "java_lcg_hybrid"}


def _oracle_decode_seeds(projection) -> List[int]:
    """A HAND-TRANSCRIBED decoder for either encoding [REV3 §1-§3].

    The harness never asks the module under test what its own bytes mean: an
    encoder and decoder that are wrong in the SAME direction round-trip
    perfectly and prove nothing. This one is written from the spec — minimal
    two's-complement big-endian runs, `offsets[i]..offsets[i+1]`, signed — so a
    codec that pads, truncates, reorders or drops the sign reds against it.

    It also asserts the exclusivity invariant: exactly one representation is
    populated, never both."""
    if projection.seed_encoding == "int64":
        assert projection.seed_bytes is None and projection.seed_offsets is None, \
            "int64 projection also carries signed_bytes state"
        return [int(s) for s in projection.seeds_i64]
    assert projection.seed_encoding == "signed_bytes", projection.seed_encoding
    assert projection.seeds_i64 is None, \
        "signed_bytes projection also carries seeds_i64"
    raw = projection.seed_bytes.tobytes()
    offs = [int(v) for v in projection.seed_offsets]
    assert len(offs) == projection.survivor_count + 1, (len(offs),
                                                        projection.survivor_count)
    assert offs[0] == 0 and offs[-1] == len(raw), (offs[0], offs[-1], len(raw))
    return [int.from_bytes(raw[offs[k]:offs[k + 1]], "big", signed=True)
            for k in range(projection.survivor_count)]


def _raw_artifact(path, **arrays):
    """Write an arbitrary `.npz` so the readback's negative cases can be driven
    with artifacts this codec would never produce."""
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)


def _identity_arrays():
    return {k: (np.array(str(v)) if isinstance(v, str)
                else np.array(int(v), dtype=np.int64))
            for k, v in _codec_identity().items()}


def _assert_stored_uncompressed(path):
    """Assert the BAN at the zip level, not via a size delta — a size
    comparison would pass for an incompressible input."""
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, (
                f"{info.filename} is compressed (compress_type="
                f"{info.compress_type}); §6.7.A requires np.savez, not "
                f"savez_compressed")


def _detect_codec(module=ASW):
    # ── (A) the int64 FAST PATH ─────────────────────────────────────────────
    # order + multiplicity, INCLUDING intra-spool duplicate seeds and a
    # deliberately UNSORTED input — a codec that sorts or uniques fails here.
    seeds = [50, 7, 7, 900, 3, 50, 7]
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    proj = RMW.build_validated_projection(seeds, np.array(rates, dtype=np.float64))
    assert proj.seed_encoding == "int64", proj.seed_encoding

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "a.npz")
        module.write_projection_artifact(path, proj, _codec_identity())

        # 1. loadable with allow_pickle=False, every member non-object, and the
        #    fast path stores the int64 array and NEITHER signed_bytes field.
        with np.load(path, allow_pickle=False) as bundle:
            for key in bundle.files:
                assert bundle[key].dtype != object, (key, bundle[key].dtype)
            files = set(bundle.files)
            assert "seeds_i64" in files, files
            assert not ({"seed_bytes", "seed_offsets"} & files), files
            assert str(bundle["seed_encoding"]) == "int64", files
            assert bundle["seeds_i64"].dtype == np.int64
            assert bundle["match_rates"].dtype == np.float64

        # 2. UNCOMPRESSED
        _assert_stored_uncompressed(path)

        # 3. round trip of order, multiplicity and exact values
        back, identity = module.read_projection_artifact(path)
        assert back.survivor_count == 7, back.survivor_count
        assert back.seed_encoding == "int64", back.seed_encoding
        assert back.match_rates.dtype == np.float64
        assert _oracle_decode_seeds(back) == seeds, _oracle_decode_seeds(back)
        assert back.match_rates.tolist() == rates
        assert identity == _codec_identity(), identity

        # ── (B) the signed_bytes FALLBACK [REV3 §4] ─────────────────────────
        # MIXED small and oversized, positive and negative, with a repeat and
        # both signed-64 boundaries — one projection, one encoding, every value
        # reconstructed exactly.
        big = [3, 2 ** 70, -25, 2 ** 70, 0, -(2 ** 69),
               2 ** 63, -(2 ** 63) - 1, 2 ** 63 - 1, -(2 ** 63)]
        brates = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 0.01]
        bproj = RMW.build_validated_projection(
            big, np.array(brates, dtype=np.float64))
        assert bproj.seed_encoding == "signed_bytes", (
            "a seed outside signed-64 did not arm the lossless fallback")
        assert _oracle_decode_seeds(bproj) == big, _oracle_decode_seeds(bproj)

        bpath = os.path.join(tmp, "b.npz")
        module.write_projection_artifact(bpath, bproj, _codec_identity())
        with np.load(bpath, allow_pickle=False) as bundle:
            for key in bundle.files:
                assert bundle[key].dtype != object, (key, bundle[key].dtype)
            files = set(bundle.files)
            assert {"seed_bytes", "seed_offsets"} <= files, files
            assert "seeds_i64" not in files, files
            assert str(bundle["seed_encoding"]) == "signed_bytes", files
            assert bundle["seed_bytes"].dtype == np.uint8, bundle["seed_bytes"].dtype
            assert bundle["seed_offsets"].dtype == np.uint64
        _assert_stored_uncompressed(bpath)
        bback, bidentity = module.read_projection_artifact(bpath)
        assert bback.seed_encoding == "signed_bytes", bback.seed_encoding
        assert bback.survivor_count == len(big), bback.survivor_count
        assert _oracle_decode_seeds(bback) == big, _oracle_decode_seeds(bback)
        assert bback.match_rates.tolist() == brates
        assert bidentity == _codec_identity(), bidentity

        # 4. the empty projection round-trips rectangularly (fast path: an
        #    empty spool has no seed outside signed-64)
        empty = RMW.build_validated_projection([], np.empty(0, dtype=np.float64))
        assert empty.seed_encoding == "int64", empty.seed_encoding
        epath = os.path.join(tmp, "e.npz")
        module.write_projection_artifact(epath, empty, _codec_identity())
        eback, _ = module.read_projection_artifact(epath)
        assert eback.survivor_count == 0 and eback.seeds_i64.shape == (0,)
        assert _oracle_decode_seeds(eback) == []

        # 5. a foreign/absent schema stamp is refused
        bad = os.path.join(tmp, "bad.npz")
        _raw_artifact(bad, schema_version=np.array("nope"),
                      seed_encoding=np.array("int64"),
                      seeds_i64=np.empty(0, dtype=np.int64),
                      match_rates=np.empty(0, dtype=np.float64),
                      survivor_count=np.array(0, dtype=np.int64),
                      **_identity_arrays())
        _raises(module.ShardArtifactError, module.read_projection_artifact, bad)

        # 6. an UNKNOWN seed_encoding is refused — the codec never guesses which
        #    representation a foreign artifact meant.
        unk = os.path.join(tmp, "unk.npz")
        _raw_artifact(unk, schema_version=np.array(ASW.ARTIFACT_SCHEMA_VERSION),
                      seed_encoding=np.array("varint"),
                      seeds_i64=np.empty(0, dtype=np.int64),
                      match_rates=np.empty(0, dtype=np.float64),
                      survivor_count=np.array(0, dtype=np.int64),
                      **_identity_arrays())
        e = _raises(module.ShardArtifactError, module.read_projection_artifact, unk)
        assert "varint" in str(e), str(e)

        # 7. a DECLARED encoding whose arrays are absent, and one whose offsets
        #    are inconsistent, are both backend failures — never a silent
        #    partial projection.
        miss = os.path.join(tmp, "miss.npz")
        _raw_artifact(miss, schema_version=np.array(ASW.ARTIFACT_SCHEMA_VERSION),
                      seed_encoding=np.array("signed_bytes"),
                      match_rates=np.empty(0, dtype=np.float64),
                      survivor_count=np.array(0, dtype=np.int64),
                      **_identity_arrays())
        _raises(module.ShardArtifactError, module.read_projection_artifact, miss)

        torn = os.path.join(tmp, "torn.npz")
        _raw_artifact(torn, schema_version=np.array(ASW.ARTIFACT_SCHEMA_VERSION),
                      seed_encoding=np.array("signed_bytes"),
                      seed_bytes=np.array([1, 2, 3], dtype=np.uint8),
                      seed_offsets=np.array([0, 1], dtype=np.uint64),
                      match_rates=np.array([0.5], dtype=np.float64),
                      survivor_count=np.array(1, dtype=np.int64),
                      **_identity_arrays())
        _raises(module.ShardArtifactError, module.read_projection_artifact, torn)


def g_codec():
    _detect_codec()


# ═════════════════════════════════════════════════════════════════════════════
# G-ATOMIC — a failure after temp-write leaves no artifact and no leaked temp
# ═════════════════════════════════════════════════════════════════════════════
def g_atomic():
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "atomic")
        art_dir = os.path.join(tmp, "arts")
        os.makedirs(art_dir)
        task = {"run_id": "atomic", "manifest": ms[0], "artifact_dir": art_dir}

        # inject a failure BETWEEN the temp write and the rename
        real_replace = os.replace
        boom = RuntimeError("injected: crash after temp-write, before rename")

        def exploding_replace(src, dst):
            raise boom

        os.replace = exploding_replace
        try:
            got = _raises(RuntimeError, ASW.validate_spool_shard, task)
            assert got is boom, got
        finally:
            os.replace = real_replace

        leftovers = sorted(os.listdir(art_dir))
        assert leftovers == [], (
            f"an injected failure after temp-write leaked {leftovers} — no "
            f"artifact may exist at the final path and no temp may survive")

        # negative control: the same task SUCCEEDS and publishes exactly one
        # artifact, so the emptiness above is the injection's doing.
        envelope = ASW.validate_spool_shard(task)
        result = envelope["result"]
        published = sorted(os.listdir(art_dir))
        assert len(published) == 1, published
        assert os.path.basename(result["artifact_path"]) == published[0]
        assert not published[0].startswith("."), "the temp name was published"


# ═════════════════════════════════════════════════════════════════════════════
# G-CLEANUP — zero temporary artifacts after success AND after failure
# ═════════════════════════════════════════════════════════════════════════════
def _tempdirs(prefix="d5_shards_"):
    root = tempfile.gettempdir()
    return sorted(p for p in os.listdir(root) if p.startswith(prefix))


def g_cleanup():
    before = _tempdirs()
    # (a) a SUCCESSFUL run
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "clean_ok")
        _sharded(ms, "clean_ok")
    assert _tempdirs() == before, f"success leaked {set(_tempdirs()) - set(before)}"

    # (b) a run that FAILS in a worker
    with tempfile.TemporaryDirectory() as tmp:
        ms, _e, _l = _malformed_dual_fixture(tmp, "clean_bad")
        _raises(SpoolIdentityError, _sharded, ms, "clean_bad")
    assert _tempdirs() == before, f"failure leaked {set(_tempdirs()) - set(before)}"

    # (c) a run that FAILS in the parent merge (duplicate), i.e. AFTER every
    #     artifact was successfully written — the path most likely to leak.
    #     REV2 §7 also requires that such an EARLY REPLAY FAILURE leaves no live
    #     worker and no running sampler behind, and that the primary exception
    #     is the canonical one, unaltered by cleanup.
    import psutil
    import threading
    with tempfile.TemporaryDirectory() as tmp:
        ms, first, second = _dup_cross_fixture(tmp, "clean_dup")
        exc = _raises(DirectionalDuplicateError, _sharded, ms, "clean_dup")
    assert _tempdirs() == before, f"merge failure leaked {set(_tempdirs()) - set(before)}"
    assert exc.seed == 1 and exc.dup_stripe == second["stripe_id"], exc.seed
    assert exc.__context__ is None and exc.__cause__ is None, (
        "cleanup replaced or chained over the primary canonical exception")
    # `multiprocessing`'s resource_tracker is a per-interpreter helper that
    # legitimately outlives every pool (it is started on first use and reaped at
    # interpreter exit); it is NOT a leaked worker, so it is excluded by name.
    # Anything else still alive would be a pool child that outlived its
    # executor.
    leaked = []
    for proc in psutil.Process(os.getpid()).children(recursive=True):
        try:
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                continue
            cmdline = " ".join(proc.cmdline())
        except (psutil.NoSuchProcess, psutil.ZombieProcess,
                psutil.AccessDenied):                   # pragma: no cover - race
            continue
        if "resource_tracker" in cmdline:
            continue
        leaked.append((proc.pid, cmdline[:120]))
    assert leaked == [], f"an early replay failure left live workers: {leaked}"
    assert not [t for t in threading.enumerate() if t.name == "d5-rss-sampler"], \
        "the RSS sampler thread outlived the failed assembly"


# ═════════════════════════════════════════════════════════════════════════════
# G-FINALIZER — process_sharded -> D3.5 finalize_run -> a published generation
# ═════════════════════════════════════════════════════════════════════════════
def g_finalizer():
    """Reuses D3.5's harness surface. Does NOT re-derive D3.5-B's S1-S9 sidecar
    assertions: it proves the 22 published arrays are IDENTICAL to the ones the
    serial backend's records produce for the same input."""
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "fin")

        def publish(assembly, sub):
            candidates = (list(assembly.canonical_records_constant)
                          + list(assembly.canonical_records_variable))
            root = Path(tmp) / sub
            root.mkdir()
            return RF.finalize_run(
                candidates, output_root=root, run_id="fin_java_lcg_0",
                prng_base=PRNG_BASE, skip_modes_executed=("constant", "variable"),
                seed_start=0, seed_count=TOTAL_SEEDS,
                repository_commit="a" * 40, repository_tree_clean=True,
            ), root

        par_pub, par_root = publish(_sharded(copy.deepcopy(ms), "fin"), "par")
        ser_pub, ser_root = publish(_serial(copy.deepcopy(ms), "fin"), "ser")

        for root in (par_root, ser_root):
            cur = root / ORACLE_ACCUM_DIR / ORACLE_CURRENT
            assert (cur / ORACLE_ALL_NPZ).is_file(), cur
            assert (cur / ORACLE_BINARY_NPZ).is_file(), cur

        for npz_name in (ORACLE_ALL_NPZ, ORACLE_BINARY_NPZ):
            p = np.load(par_root / ORACLE_ACCUM_DIR / ORACLE_CURRENT / npz_name)
            s = np.load(ser_root / ORACLE_ACCUM_DIR / ORACLE_CURRENT / npz_name)
            assert tuple(sorted(p.files)) == tuple(sorted(ORACLE_ARRAY_NAMES)), \
                sorted(p.files)
            for name in ORACLE_ARRAY_NAMES:
                assert p[name].dtype == s[name].dtype, (npz_name, name)
                assert np.array_equal(p[name], s[name]), (
                    f"{npz_name}:{name} differs between backends\n"
                    f"  process_sharded : {p[name]}\n  serial_reference: {s[name]}")

    # a backend produces a MinerTrialAssembly and STOPS: the backend module must
    # not import the finalizer (D4 G7's rule, restated for the D5 module).
    assert not hasattr(AB, "finalize_run"), "assembly_backends imported finalize_run"
    assert not hasattr(ASW, "finalize_run"), "the shard worker imported finalize_run"
    # AST, not substring: the worker module DOCUMENTS that publication belongs
    # to D3.5, and naming `finalize_run` in that prose is not an import of it.
    for src, path in ((_WORKER_SRC, _WORKER_PATH), (_BACKEND_SRC, _BACKEND_PATH)):
        for node in ast.walk(ast.parse(src, filename=path)):
            if isinstance(node, ast.ImportFrom):
                assert "run_finalizer" not in (node.module or ""), (path, node.module)
                assert not any(a.name == "finalize_run" for a in node.names), path
            elif isinstance(node, ast.Import):
                assert not any("run_finalizer" in a.name for a in node.names), path
            elif isinstance(node, ast.Call):
                assert getattr(node.func, "id", None) != "finalize_run", \
                    f"{path}:{node.lineno} calls finalize_run"
                assert getattr(node.func, "attr", None) != "finalize_run", \
                    f"{path}:{node.lineno} calls finalize_run"


# ═════════════════════════════════════════════════════════════════════════════
# G-RSS — canonical peak_rss is a sampled CONCURRENT-TREE SUM
# ═════════════════════════════════════════════════════════════════════════════
_RSS_CHILD_MB = 96
_RSS_HOLD_S = 0.6


def _rss_hog(mb, hold_s):
    """Allocate `mb` of RESIDENT memory and hold it, so two of these overlap
    across many 25 ms samples. Written into (not just reserved) so it is RSS."""
    block = np.ones(mb * 1024 * 1024 // 8, dtype=np.float64)
    block[::4096] = 2.0
    time.sleep(hold_s)
    return int(block[0])


class _RusageChildrenSampler:
    """THE RULED-OUT ALTERNATIVE, implemented here so the gate can show what it
    reports. `RUSAGE_CHILDREN.ru_maxrss` is the maximum of any SINGLE reaped
    child — never the concurrent sum — so with N overlapping children it
    under-reports by roughly a factor of N."""

    def __enter__(self):
        self.peak_rss = 0
        return self

    def __exit__(self, *exc):
        self.peak_rss = resource.getrusage(
            resource.RUSAGE_CHILDREN).ru_maxrss * 1024


def _two_child_workload():
    import concurrent.futures
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,
                                                mp_context=ctx) as ex:
        futures = [ex.submit(_rss_hog, _RSS_CHILD_MB, _RSS_HOLD_S)
                   for _ in range(2)]
        return [f.result() for f in futures]


def g_rss():
    # 1. a REAL assembly produces a positive integer peak_rss and the §5
    #    evidence block, with PSS (if present) non-gating
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "rss")
        outcome = ASW.run_sharded_assembly("rss", ms, 4)
    assert isinstance(outcome.peak_rss, int) and outcome.peak_rss > 0, \
        outcome.peak_rss
    ev = outcome.rss_evidence
    assert ev["peak_rss"] == outcome.peak_rss, ev
    assert ev["peak_rss_definition"] == ORACLE_PEAK_RSS_DEFINITION, ev
    assert ev["sample_interval_ms"] == ORACLE_SAMPLE_INTERVAL_MS, ev
    assert ev["sample_count"] >= 1, ev
    if "peak_pss_optional" in ev:
        assert isinstance(ev["peak_pss_optional"], int), ev
    # PSS never substitutes for peak_rss and is never required
    assert "peak_pss_optional" not in ORACLE_PEAK_RSS_DEFINITION

    # 2. THE RULING CONSTRUCTION: two overlapping children each holding a
    #    substantial allocation across several 25 ms samples. The concurrent
    #    tree-sum must see BOTH; RUSAGE_CHILDREN sees at most one.
    with ASW.ProcessTreeRssSampler(ORACLE_SAMPLE_INTERVAL_MS) as tree:
        with _RusageChildrenSampler() as rusage:
            _two_child_workload()
    assert tree.sample_count >= 4, (
        f"only {tree.sample_count} samples across a {_RSS_HOLD_S}s workload — "
        f"the sampler is not running at {ORACLE_SAMPLE_INTERVAL_MS} ms")
    both_mb = 2 * _RSS_CHILD_MB
    assert tree.peak_rss > both_mb * 1024 * 1024 * 0.9, (
        f"tree-sum peak {tree.peak_rss / 2**20:.0f} MiB did not capture two "
        f"concurrent {_RSS_CHILD_MB} MiB children — it is not a concurrent sum")
    assert tree.peak_rss > rusage.peak_rss, (
        f"tree-sum {tree.peak_rss / 2**20:.0f} MiB <= RUSAGE_CHILDREN "
        f"{rusage.peak_rss / 2**20:.0f} MiB — the sampler is not measuring the "
        f"concurrent tree")


# ═════════════════════════════════════════════════════════════════════════════
# §7.C — MUTATION PROOF
# ═════════════════════════════════════════════════════════════════════════════
_MUT_SEQ = 0
_MUT_DIR = None


def _mut_dir():
    """One temp dir, placed on sys.path so `spawn` children (which inherit
    sys.path through the multiprocessing preparation data) can import a mutant
    module by name and unpickle its worker function."""
    global _MUT_DIR
    if _MUT_DIR is None:
        _MUT_DIR = tempfile.mkdtemp(prefix="d5_mutants_")
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
    name = f"_d5_mutant_{_MUT_SEQ}"
    path = os.path.join(_mut_dir(), f"{name}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.__d5_label__ = label
    return module


def _executed(module, marker: str, label: str) -> None:
    """Part 2 of the four-part rule: the mutated text is present in the module
    that was actually loaded and executed."""
    import inspect
    src = inspect.getsource(module)
    assert marker in src, (
        f"{label}: the mutated text is absent from the loaded module — the "
        f"mutant did not take effect")


def _record(label, detector, credited, marker=None, module=None):
    """Run `detector` and require it to FAIL. Parts 3+4 of the four-part rule
    are discharged by the caller having already run the SAME detector clean as a
    positive control."""
    if module is not None and marker is not None:
        _executed(module, marker, label)
    try:
        detector()
    except AssertionError as exc:
        signature = str(exc).splitlines()[0][:150] or type(exc).__name__
        _MUTANTS.append((label, f"AssertionError: {signature}", credited,
                         "applies-once ✓ | mutated-path ✓ | detector-clean ✓ | "
                         "injected-defect ✓"))
        return
    except Exception as exc:                                    # noqa: BLE001
        signature = f"{type(exc).__name__}: {str(exc).splitlines()[0][:130]}"
        _MUTANTS.append((label, signature, credited,
                         "applies-once ✓ | mutated-path ✓ | detector-clean ✓ | "
                         "injected-defect ✓"))
        return
    raise AssertionError(f"MUTANT SURVIVED: {label} — {credited} did not red")


def _positive_control(name, detector):
    """Part 3 of the four-part rule: the detector must PASS against the
    UNMUTATED module. A detector that cannot pass clean cannot be credited with
    a kill, so no recorded red is attributable to a loader, fixture,
    type-identity or setup failure."""
    try:
        detector()
    except Exception as exc:                                    # noqa: BLE001
        raise AssertionError(
            f"POSITIVE CONTROL FAILED for {name}: the detector reds against the "
            f"UNMUTATED module ({type(exc).__name__}: {exc}) — any kill it "
            f"records would be unattributable") from exc


# ---- writer-module mutant (M1) --------------------------------------------
_WRITER_PATH = os.path.join(_ROOT, "miner", "range_miner_npz_writer.py")
with open(_WRITER_PATH, "r", encoding="utf-8") as _f:
    _WRITER_SRC = _f.read()


def _detect_merge_order(writer_module=None):
    """Duplicate ATTRIBUTION and record ORDER must match the serial oracle. A
    merge that consumes anything other than the deterministic manifest order
    reverses which spool is 'first'."""
    with tempfile.TemporaryDirectory() as tmp:
        ms, first, second = _dup_cross_fixture(tmp, "m1")
        # A separately-loaded writer module defines its OWN exception classes,
        # so this detector must NOT compare class identity — dying on
        # `isinstance` would be a type-identity failure, which proves nothing
        # (§7.C part 4). Match on the class NAME and assert on the structured
        # attribution, so the only thing that can red is the mutation itself.
        try:
            if writer_module is None:
                _serial(copy.deepcopy(ms), "m1")
            else:
                writer_module.assemble_trial("m1", copy.deepcopy(ms))
        except Exception as exc:                                # noqa: BLE001
            e = exc
        else:
            raise AssertionError("no duplicate was raised at all")
        assert type(e).__name__ == "DirectionalDuplicateError", (
            f"expected DirectionalDuplicateError, got {type(e).__name__}: {e}")
        assert e.first_stripe == first["stripe_id"], (
            f"first-insertion attributed to {e.first_stripe} (sub "
            f"{e.first_sub_index}), not the earlier-in-order "
            f"{first['stripe_id']} sub 0")
        assert e.first_sub_index == 0 and e.first_match_rate == 0.90, (
            f"first-insertion attributed to sub{e.first_sub_index} @ "
            f"{e.first_match_rate} — expected sub0 @ 0.90; the merge consumed "
            f"something other than the deterministic manifest order")
        assert e.dup_sub_index == 1 and e.dup_match_rate == 0.11, (
            f"duplicate attributed to sub{e.dup_sub_index} @ "
            f"{e.dup_match_rate} — expected sub1 @ 0.11")


def _detect_interleaved_precedence(writer_module=None):
    """REV2's headline property: with an EARLIER-position duplicate and a
    LATER-position malformed spool, the DUPLICATE must surface. Only an
    interleaved (lazily-consumed) serial front end does that; read-all-then-merge
    reports the malformed spool instead."""
    with tempfile.TemporaryDirectory() as tmp:
        ms, dup_target = _row_dup_then_malformed(tmp, "m10")
        if writer_module is None:
            exc = _observe(_serial, copy.deepcopy(ms), "m10")
        else:
            exc = _observe(writer_module.assemble_trial, "m10",
                           copy.deepcopy(ms))
        assert type(exc).__name__ == "DirectionalDuplicateError", (
            f"expected the EARLIER-position DirectionalDuplicateError, got "
            f"{type(exc).__name__}: {exc} — the front end read ahead of the "
            f"merge instead of interleaving")
        assert exc.seed == 1, exc.seed
        assert exc.first_sub_index == 0 and exc.first_match_rate == 0.90
        assert exc.dup_sub_index == 1 and exc.dup_match_rate == 0.11
        assert exc.dup_stripe == dup_target["stripe_id"], exc.dup_stripe


def mutants_all():
    # ══ M1: merge consumes a different order than the deterministic one ══════
    src = _patch(
        _WRITER_SRC,
        "    for manifest, meta, outcome in ordered_outcomes:",
        "    for manifest, meta, outcome in list(ordered_outcomes)[::-1]:",
        "M1")
    _positive_control("M1 detector", lambda: _detect_merge_order(None))
    m1 = _load_mutant(src, "M1")
    _record("M1 merge_validated_spools consumes reversed (non-deterministic) order",
            lambda: _detect_merge_order(m1),
            "G-DUP-CROSS / G-EQUIV duplicate attribution",
            "list(ordered_outcomes)[::-1]", m1)

    # ══ M2: the worker REIMPLEMENTS validation and drops the skips check ═════
    weak = (
        "        import json as _json\n"
        "        _raw = open(manifest['local_spool_path'], 'rb').read()\n"
        "        _p = _json.loads(_raw.decode('utf-8'))\n"
        "        _s = _p['survivors']\n"
        "        projection = ValidatedSpoolProjection(\n"
        "            seed_encoding=SEED_ENCODING_INT64,\n"
        "            seeds_i64=np.array([e[0] for e in _s], dtype=np.int64),\n"
        "            seed_bytes=None, seed_offsets=None,\n"
        "            match_rates=np.array([e[1] for e in _s], dtype=np.float64),\n"
        "            survivor_count=len(_s))\n")
    src = _patch(
        _WORKER_SRC,
        "        projection = read_and_validate_spool(run_id, manifest)\n",
        weak, "M2")
    _positive_control("M2 detector", lambda: _detect_skip_validation(ASW))
    m2 = _load_mutant(src, "M2")
    _record("M2 worker skips a validation branch (drops the ragged-skips type check)",
            lambda: _detect_skip_validation(m2),
            "G-EQUIV validation parity (a spool serial rejects is accepted)",
            "_json.loads(_raw.decode", m2)

    # ══ M3: the parent REPLAYS in completion order, not manifest order ══════
    # `as_completed` filling indexed slots is CORRECT (REV2 §6); replaying in
    # that order is the defect. The mutant records completion order and walks
    # it, which is precisely what the ruling forbids.
    src = _patch(_WORKER_SRC, 'OUTCOME_PROJECTION = "projection"',
                 'OUTCOME_PROJECTION = "projection"\n_COMPLETION_ORDER = []',
                 "M3a")
    src = _patch(
        src,
        '            tasks = [{"run_id": run_id, "manifest": manifests[i],',
        '            _COMPLETION_ORDER.clear()\n'
        '            tasks = [{"run_id": run_id, "manifest": manifests[i],',
        "M3b")
    src = _patch(
        src,
        "                        outcomes[position] = future.result()\n",
        "                        outcomes[position] = future.result()\n"
        "                        _COMPLETION_ORDER.append(position)\n",
        "M3c")
    src = _patch(
        src,
        "    for position, i in enumerate(order):\n"
        "        failure = failures[position]",
        "    for position in _COMPLETION_ORDER:\n"
        "        i = order[position]\n"
        "        failure = failures[position]",
        "M3d")
    _positive_control("M3 detector", lambda: _detect_malformed_dual(ASW))
    m3 = _load_mutant(src, "M3")
    _record("M3 parent REPLAYS worker outcomes in completion order, not manifest order",
            lambda: _detect_malformed_dual(m3),
            "G-MALFORMED-DUAL / G-MATRIX row 4 (earlier-in-order defect must win)",
            "for position in _COMPLETION_ORDER:", m3)

    # ══ M4: the worker SORTS + DEDUPS the projection ════════════════════════
    sorter = (
        "    _o = np.argsort(projection.seeds_i64, kind='stable')\n"
        "    _s2, _idx = np.unique(projection.seeds_i64[_o], return_index=True)\n"
        "    projection = ValidatedSpoolProjection(\n"
        "        seed_encoding=SEED_ENCODING_INT64, seeds_i64=_s2,\n"
        "        seed_bytes=None, seed_offsets=None,\n"
        "        match_rates=projection.match_rates[_o][_idx],\n"
        "        survivor_count=int(_s2.shape[0]))\n"
        "    identity = _identity_from(manifest, run_id)\n")
    src = _patch(
        _WORKER_SRC,
        '                "descriptor": capture_spool_read_error(exc)}\n'
        "    identity = _identity_from(manifest, run_id)\n",
        '                "descriptor": capture_spool_read_error(exc)}\n' + sorter,
        "M4")
    _positive_control("M4 detector", lambda: _detect_dup_intra(ASW))
    m4 = _load_mutant(src, "M4")
    _record("M4 worker sorts + dedups the projection",
            lambda: _detect_dup_intra(m4),
            "G-DUP-INTRA [F3] (order + multiplicity must survive the worker)",
            "np.unique(projection.seeds_i64", m4)

    # ══ M5: the parent resolves duplicates itself, outside the shared merge ══
    # the mutant DEDUPS in the parent's own dict, so a duplicate never reaches
    # the shared merge that is supposed to raise on it
    concurrent_dict = (
        "            _items = list(_replay_outcomes(run_id, manifests, metas,\n"
        "                                           order, outcomes, failures))\n"
        "            _seen = set()\n"
        "            _dedup = []\n"
        "            for _m, _meta, _pj in _items:\n"
        "                if not isinstance(_pj, ValidatedSpoolProjection):\n"
        "                    _dedup.append((_m, _meta, _pj))\n"
        "                    continue\n"
        "                _keep = []\n"
        "                _decoded = projection_seeds(_pj)\n"
        "                for _k in range(_pj.survivor_count):\n"
        "                    _key = (_meta['direction'], _meta['skip_mode'],\n"
        "                            _decoded[_k])\n"
        "                    if _key in _seen:\n"
        "                        continue\n"
        "                    _seen.add(_key)\n"
        "                    _keep.append(_k)\n"
        "                _idx = np.asarray(_keep, dtype=np.intp)\n"
        "                _dedup.append((_m, _meta, ValidatedSpoolProjection(\n"
        "                    seed_encoding=SEED_ENCODING_INT64,\n"
        "                    seeds_i64=_pj.seeds_i64[_idx],\n"
        "                    seed_bytes=None, seed_offsets=None,\n"
        "                    match_rates=_pj.match_rates[_idx],\n"
        "                    survivor_count=len(_keep))))\n"
        "            assembly = merge_validated_spools(run_id, ctx, _dedup, started)")
    src = _patch(
        _WORKER_SRC,
        "            assembly = merge_validated_spools(\n"
        "                run_id, ctx,\n"
        "                _replay_outcomes(run_id, manifests, metas, order, outcomes,\n"
        "                                 failures),\n"
        "                started)",
        concurrent_dict, "M5")
    _positive_control("M5 detector", lambda: _detect_dup_cross(ASW))
    m5 = _load_mutant(src, "M5")
    _record("M5 parent resolves duplicates in its own dict, outside the shared merge",
            lambda: _detect_dup_cross(m5),
            "G-DUP-CROSS (a duplicate is a producer defect, never a dedup "
            "opportunity)",
            "_seen = set()", m5)

    # ══ M6a: the codec compresses ═══════════════════════════════════════════
    src = _patch(_WORKER_SRC, "        np.savez(fh, **payload)",
                 "        np.savez_compressed(fh, **payload)", "M6a")
    _positive_control("M6a detector", lambda: _detect_codec(ASW))
    m6a = _load_mutant(src, "M6a")
    _record("M6a codec uses savez_compressed",
            lambda: _detect_codec(m6a),
            "G-CODEC (asserts ZIP_STORED — the ban, not a size delta)",
            "np.savez_compressed(fh", m6a)

    # ══ M6b: the codec stores an object array ═══════════════════════════════
    src = _patch(
        _WORKER_SRC,
        '        "schema_version": np.array(ARTIFACT_SCHEMA_VERSION),',
        '        "schema_version": np.array(ARTIFACT_SCHEMA_VERSION),\n'
        '        "skips_ragged": np.array([[1, 2], [3]], dtype=object),',
        "M6b")
    m6b = _load_mutant(src, "M6b")
    _record("M6b codec stores an object array (ragged skips smuggled in)",
            lambda: _detect_codec(m6b),
            "G-CODEC (allow_pickle=False load / object-dtype ban)",
            "dtype=object", m6b)

    # ══ M7: fork substituted for spawn ══════════════════════════════════════
    src = _patch(
        _WORKER_SRC,
        '    if start_method == "fork":',
        '    if start_method == "__never__":', "M7")
    src = _patch(
        src,
        '    if start_method not in ("spawn", "forkserver"):',
        '    start_method = "fork"\n'
        '    if start_method not in ("spawn", "forkserver", "fork"):', "M7b")
    _positive_control("M7 detector", lambda: _detect_spawn(ASW))
    m7 = _load_mutant(src, "M7")
    _record("M7 fork substituted for spawn",
            lambda: _detect_spawn(m7),
            "G-SPAWN (fork refused outright; the resolved method is spawn)",
            'start_method = "fork"', m7)

    # ══ M8: RUSAGE_CHILDREN substituted for the concurrent sampler ══════════
    def _detect_rss_sum(sampler_cls):
        with sampler_cls() as s:
            _two_child_workload()
        both = 2 * _RSS_CHILD_MB * 1024 * 1024
        assert s.peak_rss > both * 0.9, (
            f"peak {s.peak_rss / 2**20:.0f} MiB did not capture two concurrent "
            f"{_RSS_CHILD_MB} MiB children (need > {both * 0.9 / 2**20:.0f} "
            f"MiB) — this is a single-child maximum, not a concurrent sum")
    _positive_control("M8 detector",
                      lambda: _detect_rss_sum(
                          lambda: ASW.ProcessTreeRssSampler(
                              ORACLE_SAMPLE_INTERVAL_MS)))
    _record("M8 RUSAGE_CHILDREN substituted for the sampled concurrent tree-sum",
            lambda: _detect_rss_sum(_RusageChildrenSampler),
            "G-RSS (tree-sum captures both concurrent children)")

    # ══ M9: the metadata gauntlet moved AFTER dispatch ══════════════════════
    src = _patch(
        _WORKER_SRC,
        "    metas, ctx, order = prepare_trial_assembly(run_id, manifests)",
        "    order = list(range(len(manifests)))\n"
        "    _deferred_gauntlet = True", "M9")
    src = _patch(
        src,
        "            assembly = merge_validated_spools(\n"
        "                run_id, ctx,\n",
        "            from miner.range_miner_npz_writer import (\n"
        "                raise_captured_spool_error as _replay_now)\n"
        "            for _o in outcomes:\n"
        "                if isinstance(_o, dict) and _o.get('outcome_kind') == \\\n"
        "                        OUTCOME_READ_ERROR:\n"
        "                    _replay_now(_o['descriptor'])\n"
        "            metas, ctx, order = prepare_trial_assembly(run_id, manifests)\n"
        "            assembly = merge_validated_spools(\n"
        "                run_id, ctx,\n", "M9b")
    _positive_control("M9 detector", lambda: _detect_precedence(ASW))
    m9 = _load_mutant(src, "M9")
    _record("M9 metadata gauntlet moved AFTER dispatch and after outcome observation",
            lambda: _detect_precedence(m9),
            "G-PRECEDENCE (metadata exception must pre-empt SpoolIdentityError)",
            "_deferred_gauntlet = True", m9)

    # ══ M10 [REV2 §2]: the SERIAL path reads all spools, THEN merges ═════════
    # This is the exact structure Team Beta ruled out. It reds only on the
    # earlier-dup + later-malformed corner, which is why the ruling exists.
    src = _patch(
        _WRITER_SRC,
        "        _serial_outcomes(run_id, manifests, metas, order),",
        "        list(_serial_outcomes(run_id, manifests, metas, order)),",
        "M10")
    _positive_control("M10 detector", lambda: _detect_interleaved_precedence(None))
    m10 = _load_mutant(src, "M10")
    _record("M10 serial reads ALL spools then merges (read-all-then-merge)",
            lambda: _detect_interleaved_precedence(m10),
            "G-MATRIX row 1 (earlier duplicate must pre-empt later malformed)",
            "list(_serial_outcomes(", m10)

    # ══ M11: the descriptor drops the RENDERED MESSAGE ══════════════════════
    src = _patch(_WRITER_SRC, "        message=str(exc),",
                 "        message=type(exc).__name__,", "M11")
    _positive_control("M11 detector",
                      lambda: _detect_descriptor_roundtrip(RMW))
    m11 = _load_mutant(src, "M11")
    _record("M11 captured descriptor drops the rendered message",
            lambda: _detect_descriptor_roundtrip(m11),
            "G-DESCRIPTOR (class + args + message must round-trip)",
            "message=type(exc).__name__", m11)

    # ══ M12: the descriptor drops CUSTOM ATTRIBUTION ════════════════════════
    src = _patch(_WRITER_SRC, "    attributes = dict(vars(exc))",
                 "    attributes = {}", "M12")
    m12 = _load_mutant(src, "M12")
    _record("M12 captured descriptor drops custom attribution fields",
            lambda: _detect_descriptor_roundtrip(m12),
            "G-DESCRIPTOR (attribution must round-trip)",
            "    attributes = {}", m12)

    # ══ M13 [REV2 §5]: a BACKEND failure is descriptorized as a producer defect
    # The fault INJECTION (a worker that dies of MemoryError) is the fixture;
    # the mutation is the masquerade. The positive control runs the same
    # detector against the INJECTED-but-unmutated module, so the recorded red is
    # attributable to the masquerade alone.
    masquerade = (
        "    if True:\n"
        "        return ({'outcome_kind': OUTCOME_READ_ERROR,\n"
        "                 'descriptor': CapturedSpoolReadError(\n"
        "                     error_code='SpoolIdentityError', message=str(exc),\n"
        "                     args=(str(exc),), attributes={})}, None)\n"
        "    if type(exc).__name__ in CANONICAL_SPOOL_READ_ERRORS:\n"
        "        try:\n")
    injected_clean = _injected_worker_module()
    _positive_control("M13 detector",
                      lambda: _detect_backend_failure_distinct(injected_clean))
    m13 = _injected_worker_module((
        ("    if type(exc).__name__ in CANONICAL_SPOOL_READ_ERRORS:\n"
         "        try:\n", masquerade, "M13"),))
    _record("M13 backend failure descriptorized as a CapturedSpoolReadError",
            lambda: _detect_backend_failure_distinct(m13),
            "G-BACKEND-DISTINCT (infrastructure must never blame the producer)",
            "'descriptor': CapturedSpoolReadError(", m13)

    # ══ M14 [REV2 §6/§7]: the fill loop RAISES instead of recording ═════════
    _positive_control("M14 detector",
                      lambda: _detect_canonical_beats_backend(injected_clean))
    m14 = _injected_worker_module((
        ("                    except Exception as exc:\n"
         "                        outcomes[position], failures[position] = \\\n"
         "                            _capture_worker_exception(exc)\n",
         "                    except Exception:\n"
         "                        raise\n", "M14"),))
    _record("M14 as_completed fill loop raises immediately instead of recording "
            "the failure at its position",
            lambda: _detect_canonical_beats_backend(m14),
            "G-BACKEND-DISTINCT (an earlier canonical defect stays primary)",
            "                    except Exception:\n                        raise",
            m14)

    # ══ M15 [REV3 §1]: the projection is FORCED onto the int64 fast path ═════
    # This is the BLOCKER itself, re-injected: the fallback never arms, so a
    # seed pre-D5 accepted raises OverflowError out of the projection layer and
    # the engine's accepted input silently narrows.
    src = _patch(
        _WRITER_SRC,
        "    if all(_INT64_MIN <= seed <= _INT64_MAX for seed in seeds):",
        "    if True:          # forced: the lossless fallback never arms",
        "M15")
    _positive_control("M15 detector", lambda: _detect_oversized_domain(None))
    m15 = _load_mutant(src, "M15")
    _record("M15 projection forced onto the int64 fast path (the lossless "
            "fallback never arms)",
            lambda: _detect_oversized_domain(m15),
            "G-SEED-DOMAIN rows 2/4 (a seed pre-D5 accepted must still assemble)",
            "if True:          # forced", m15)

    # ══ M16 [REV3 §3]: the decoder reads the byte runs UNSIGNED ══════════════
    # Lossless storage, lossy interpretation: -25 comes back as 231, so the map
    # keys diverge from the oracle's while every shape invariant still holds.
    src = _patch(
        _WRITER_SRC,
        "        return [int.from_bytes(blob[int(offsets[k]):int(offsets[k + 1])],\n"
        '                               "big", signed=True)',
        "        return [int.from_bytes(blob[int(offsets[k]):int(offsets[k + 1])],\n"
        '                               "big", signed=False)',
        "M16")
    _positive_control("M16 detector", lambda: _detect_negative_fallback(None))
    m16 = _load_mutant(src, "M16")
    _record("M16 signed_bytes decoder reads the runs UNSIGNED",
            lambda: _detect_negative_fallback(m16),
            "G-SEED-DOMAIN row 3 (a negative seed must decode negative)",
            '"big", signed=False)', m16)

    # ══ M17 [REV3 §2]: the classic signed-byte-length boundary off-by-one ══════════
    # `(bit_length + 7) // 8` is the minimal UNSIGNED length. It is correct for
    # most values and raises OverflowError exactly at the ±2^(8k-1) boundaries —
    # which is why only a boundary-valued detector can kill it.
    src = _patch(
        _WRITER_SRC,
        "    nbytes = (seed.bit_length() // 8) + 1",
        "    nbytes = (seed.bit_length() + 7) // 8",
        "M17")
    _positive_control("M17 detector", lambda: _detect_signed_boundary(None))
    m17 = _load_mutant(src, "M17")
    _record("M17 signed-byte length replaced with an insufficient boundary formula",
            lambda: _detect_signed_boundary(m17),
            "G-SEED-CODEC / G-SEED-DOMAIN row 2 (the ±2^(8k-1) boundary)",
            "nbytes = (seed.bit_length() + 7) // 8", m17)


def _domain_assembly(writer_module, run_id, ms):
    """Assemble under the LIVE serial backend, or under a separately-loaded
    writer mutant. Never compares class identity across modules (§7.C part 4)."""
    if writer_module is None:
        return _serial(copy.deepcopy(ms), run_id)
    return writer_module.assemble_trial(run_id, copy.deepcopy(ms))


def _detect_oversized_domain(writer_module=None):
    """A seed outside signed-64 was ACCEPTED by the pre-D5 engine, so it must be
    accepted now, and must produce the identical four maps."""
    with tempfile.TemporaryDirectory() as tmp:
        seed = 2 ** 70
        ms = _build_run(tmp, AssemblingPhase5Sink(), "ovr")
        ms, _ = _retarget_spool(ms, tmp, 0, 0, _SEED_DOMAIN_SPAN,
                                [[seed, 0.90, None, [1]]], "ovr")
        ref = _reference(copy.deepcopy(ms), "ovr")
        got = _domain_assembly(writer_module, "ovr", ms)
        _assert_equivalent(got, ref, "oversized seed vs pre-D5 oracle")
        assert seed in got.forward_map_constant, (
            f"seed {seed} vanished from the population: "
            f"{sorted(got.forward_map_constant)[:4]}")


def _detect_negative_fallback(writer_module=None):
    """A NEGATIVE seed in a spool that armed the fallback must decode back to
    the same negative value — an unsigned read turns -25 into 231."""
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "neg")
        ms, _ = _retarget_spool(
            ms, tmp, 0, -(2 ** 70), _SEED_DOMAIN_SPAN,
            [[-25, 0.90, None, [1]], [-(2 ** 69), 0.80, None, [2]]], "neg")
        ref = _reference(copy.deepcopy(ms), "neg")
        got = _domain_assembly(writer_module, "neg", ms)
        _assert_equivalent(got, ref, "negative fallback vs pre-D5 oracle")
        assert -25 in got.forward_map_constant, (
            f"-25 decoded to something else: "
            f"{sorted(got.forward_map_constant)[:4]}")


def _detect_signed_boundary(writer_module=None):
    """2**63 has bit_length 64: it needs NINE signed bytes, and the unsigned
    formula allocates eight. Both boundary signs are driven."""
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "bnd")
        # the window must contain every boundary value: [-(2**72), 2**72)
        ms, _ = _retarget_spool(
            ms, tmp, 0, -(2 ** 72), 2 ** 73,
            [[2 ** 63, 0.90, None, [1]], [-(2 ** 63) - 1, 0.80, None, [2]],
             [2 ** 71 - 1, 0.70, None, [3]]], "bnd")
        ref = _reference(copy.deepcopy(ms), "bnd")
        got = _domain_assembly(writer_module, "bnd", ms)
        _assert_equivalent(got, ref, "signed boundary vs pre-D5 oracle")
        for seed in (2 ** 63, -(2 ** 63) - 1, 2 ** 71 - 1):
            assert seed in got.forward_map_constant, (
                f"boundary seed {seed} did not survive the projection")


def _detect_dup_cross(module=ASW):
    with tempfile.TemporaryDirectory() as tmp:
        ms, first, second = _dup_cross_fixture(tmp, "dupx2")
        e = _raises(DirectionalDuplicateError, _sharded,
                    copy.deepcopy(ms), "dupx2", 4, module)
        assert e.seed == 1 and e.first_sub_index == 0, (e.seed, e.first_sub_index)
        assert e.dup_sub_index == 1 and e.dup_match_rate == 0.11, e.dup_match_rate


def _detect_skip_validation(module=ASW):
    """A spool whose `skips` is a string is a §5.3 semantic defect. Serial
    rejects it; process_sharded must reject it IDENTICALLY."""
    with tempfile.TemporaryDirectory() as tmp:
        ms = _build_run(tmp, AssemblingPhase5Sink(), "skipval")
        target = next(m for m in ms if int(m["workflow_phase"]) == 1
                      and m["sub_index"] == 0 and m["stripe_id"].endswith("_s0"))
        payload = _load_payload(target)
        payload["survivors"][0][3] = "not-a-list"
        bad = _repoint(target, tmp, payload, "skipval")
        ms2 = [bad if m["event_id"] == target["event_id"] else copy.deepcopy(m)
               for m in ms]
        e_ser = _raises(SpoolIdentityError, _serial, copy.deepcopy(ms2), "skipval")
        assert "skip_sequence is str" in str(e_ser), str(e_ser)
        e_par = _raises(SpoolIdentityError, _sharded,
                        copy.deepcopy(ms2), "skipval", 4, module)
        assert str(e_par) == str(e_ser), f"\nsharded: {e_par}\nserial : {e_ser}"


def g_mutation_proof():
    mutants_all()


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK — 1 / 2 / 4 / 6 / 8 processes, high- and low-survivor (§4.5)
# ═════════════════════════════════════════════════════════════════════════════
def g_benchmark():
    """Produces the numbers §17's promotion rule consumes. D5 does NOT promote:
    §17 promotion is Phase 6's isolated benchmark (a fresh process per measured
    backend), and `serial_reference` stays the production default until then."""
    scenarios = []
    with tempfile.TemporaryDirectory() as tmp:
        # HIGH survivor: 4 phases x 4 stripes x 2 sub-stripes, 4000 survivors
        # each — real hashing + parsing volume, which is what §6.7.A says
        # dominates.
        pops = {p: {s: 0.30 + (s % 60) / 100.0 for s in range(64000)}
                for p in (1, 2, 3, 4)}
        high = _build_run(tmp, AssemblingPhase5Sink(), "bench_hi", pops=pops,
                          total=64000, macro=16000, cap=8000)
        scenarios.append(("high-survivor", "bench_hi", high))
        low = _build_run(tmp, AssemblingPhase5Sink(), "bench_lo", dbname="lo.db")
        scenarios.append(("low-survivor", "bench_lo", low))

        for label, run_id, manifests in scenarios:
            bytes_read = sum(m["expected_size"] for m in manifests)
            baseline = None
            for size in ORACLE_BENCH_POOL_SIZES:
                backend = AB.ProcessShardedBackend(pool_size=size)
                t0 = time.perf_counter()
                result = backend.assemble(run_id, copy.deepcopy(manifests))
                wall = time.perf_counter() - t0
                if baseline is None:
                    baseline = wall
                ev = backend.last_rss_evidence
                _BENCH.append({
                    "scenario": label, "pool_size": size,
                    "manifests": len(manifests), "spool_bytes": bytes_read,
                    "wall_s": wall,
                    "speedup_vs_pool1": baseline / wall if wall else 0.0,
                    "peak_rss": ev["peak_rss"],
                    "peak_rss_definition": ev["peak_rss_definition"],
                    "sample_interval_ms": ev["sample_interval_ms"],
                    "peak_pss_optional": ev.get("peak_pss_optional"),
                    "survivor_rows": result.measurement.survivor_row_count,
                })
                # the measurement the backend itself publishes is coherent
                assert result.measurement.backend_name == "process_sharded"
                assert result.measurement.peak_rss_bytes == ev["peak_rss"]
                assert result.measurement.wall_seconds > 0

            # a serial baseline for the same input, for §17's denominator
            t0 = time.perf_counter()
            AB.get_assembly_backend("serial_reference").assemble(
                run_id, copy.deepcopy(manifests))
            _BENCH.append({"scenario": label, "pool_size": "serial_reference",
                           "manifests": len(manifests), "spool_bytes": bytes_read,
                           "wall_s": time.perf_counter() - t0,
                           "speedup_vs_pool1": None, "peak_rss": None,
                           "peak_rss_definition": None,
                           "sample_interval_ms": None,
                           "peak_pss_optional": None, "survivor_rows": None})

    assert len(_BENCH) == 2 * (len(ORACLE_BENCH_POOL_SIZES) + 1), len(_BENCH)


# ═════════════════════════════════════════════════════════════════════════════
# §7.D — BLOCKING NON-REGRESSION
# ═════════════════════════════════════════════════════════════════════════════
def _run_suite(rel_path, expect_substr, timeout=2400):
    env = dict(os.environ, PYTHONPATH=_ROOT)
    r = subprocess.run([sys.executable, rel_path], cwd=_ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, (
        f"{rel_path} exited {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}")
    assert expect_substr in r.stdout, (
        f"{rel_path}: expected {expect_substr!r} in output\n{r.stdout[-3000:]}")


def g_nonregression():
    """D1.1 is the extraction's proof and itself nests Phase 4 / Phase 3 / D0 /
    D1.0, so this set covers the full §7.D list transitively."""
    _run_suite("tests/test_s172_phase5_d1_engine.py", "18/18 D1.1 gate checks green")
    _run_suite("tests/test_s172_phase5_d2_directional_uniqueness.py",
               "7/7 D2 gate checks green")
    _run_suite("tests/test_s172_phase5_d3_columnizer.py", "10/10 D3 gate checks green")
    _run_suite("tests/test_s172_phase5_d3_0_encoding_contract.py",
               "10/10 D3.0 gate checks green")
    _run_suite("tests/test_s172_phase5_d3_25_candidate_ingress.py",
               "13/13 D3.25 gate checks green")
    _run_suite("tests/test_s172_phase5_d3_5_finalizer.py",
               "60/60 D3.5 gate checks green")
    _run_suite("tests/test_s172_phase5_d4_serial_backend.py",
               "8/8 D4 gate checks green")


# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 78)
    print("S172 Phase-5 D5 — `process_sharded` assembly backend gate")
    print("=" * 78)

    _check("G-EQUIV: process_sharded ≡ serial_reference across the matrix",
           g_equivalence_matrix)
    _check("G-DUP-CROSS: identical duplicate + attribution across two spools",
           g_dup_cross)
    _check("G-DUP-INTRA [F3]: intra-spool duplicate survives the projection",
           g_dup_intra)
    _check("G-MALFORMED-DUAL: earlier-in-order defect wins, not the fastest",
           g_malformed_dual)
    _check("G-PRECEDENCE: metadata gauntlet pre-empts every spool defect",
           g_precedence)
    _check("G-MATRIX [REV2]: the six-row precedence matrix across pre-D5 "
           "reference + serial_reference + process_sharded",
           g_precedence_matrix)
    _check("G-SEED-DOMAIN [REV3]: the out-of-range seed domain (2**63-1, 2**63, "
           "negative windows, >64-bit, mixed, duplicate attribution) matches "
           "the pre-D5 oracle under both backends",
           g_seed_domain)
    _check("G-SEED-CODEC [REV3]: deterministic signed-byte round trip at the "
           "±2^(8k-1) boundaries; exactly one representation per projection",
           g_seed_encoding)
    _check("G-ORACLE [REV3]: the pre-D5 oracle is vendored + digest-pinned, and "
           "faithful to 3e8580a wherever git history can verify it",
           g_oracle_durability)
    _check("G-SERIAL-ORIGINAL [REV2]: serial raises the ORIGINAL exception "
           "object; process_sharded a reconstructed-but-equivalent one",
           g_serial_original)
    _check("G-DESCRIPTOR [REV2]: captured read errors round-trip class, args, "
           "message and attribution; the allowlist refuses everything else",
           g_descriptor)
    _check("G-BACKEND-DISTINCT [REV2]: backend failures are never producer "
           "defects, and never pre-empt an earlier canonical one",
           g_backend_distinct)
    _check("G-SPAWN: spawn is canonical; fork refused outright", g_spawn)
    _check("G-NO-GPU: no worker process holds torch/cupy", g_no_gpu)
    _check("G-NO-PAYLOAD-IPC: the four §6.7.A prohibitions proven absent",
           g_no_payload_ipc)
    _check("G-D4-INTACT: D4's mutation anchors + AST bans survive D5's edit",
           g_d4_intact)
    _check("G-CODEC: lossless, allow_pickle=False, no object arrays, uncompressed",
           g_codec)
    _check("G-ATOMIC: failure after temp-write leaves no artifact, no temp",
           g_atomic)
    _check("G-CLEANUP: zero temp artifacts after success and every failure",
           g_cleanup)
    _check("G-FINALIZER: sharded -> finalize_run -> 22 arrays match serial",
           g_finalizer)
    _check("G-RSS: canonical peak_rss is a sampled concurrent tree-sum", g_rss)
    _check("G-MUTANTS: mutation proof (18 mutants, four-part rule)",
           g_mutation_proof)
    _check("G-BENCH: 1/2/4/6/8-process sweep, high- and low-survivor",
           g_benchmark)
    _check("NR: D1.1 18/18, D2 7/7, D3 10/10, D3.0 10/10, D3.25 13/13, "
           "D3.5 60/60, D4 8/8 (D1.1 nests Phase 4 63/63, Phase 3 17/17, D0, D1.0)",
           g_nonregression)
    print("=" * 78)

    if _MUTANTS:
        print("\nMUTATION EVIDENCE — every mutant RED, with attribution:\n")
        for label, signature, credited, fourpart in _MUTANTS:
            print(f"  {label}")
            print(f"      red in   : {credited}")
            print(f"      four-part: {fourpart}")
            print(f"      signature: {signature}")
        print()

    if _BENCH:
        print("\nBENCHMARK — §4.5 sweep (canonical peak_rss = "
              f"{ORACLE_PEAK_RSS_DEFINITION}, {ORACLE_SAMPLE_INTERVAL_MS} ms):\n")
        print(f"  {'scenario':<15} {'pool':>16} {'manifests':>10} "
              f"{'spool MiB':>10} {'wall s':>9} {'speedup':>8} {'peak RSS MiB':>13}")
        for row in _BENCH:
            rss = ("-" if row["peak_rss"] is None
                   else f"{row['peak_rss'] / 2**20:.1f}")
            sp = ("-" if row["speedup_vs_pool1"] is None
                  else f"{row['speedup_vs_pool1']:.2f}x")
            print(f"  {row['scenario']:<15} {str(row['pool_size']):>16} "
                  f"{row['manifests']:>10} {row['spool_bytes'] / 2**20:>10.1f} "
                  f"{row['wall_s']:>9.3f} {sp:>8} {rss:>13}")
        print("\n  RSS-sum double-counts pages shared between parent and "
              "children, so it is a\n  CONSERVATIVE process-tree footprint, not "
              "exact physical RAM. D5 measures;\n  §17 promotion is Phase 6's "
              "isolated benchmark.\n")

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} D5 gate checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All D5 gate checks green — `process_sharded` parallelizes ONLY "
          "spool-local validation and is field-for-field equivalent to "
          "`serial_reference`, with global merge, duplicate attribution, "
          "intersection and enrichment owned solely by the parent (pending Team "
          "Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if _MUT_DIR is not None:
            import shutil
            shutil.rmtree(_MUT_DIR, ignore_errors=True)
