#!/usr/bin/env python3
"""
test_s172_phase5_d1_engine.py — S172 Phase-5 Deliverable D1.1 acceptance harness
(Gate D1.1, docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D1.md REV5 §9, G1-G16).

Subject under test: `miner/range_miner_npz_writer.py` — the backend-independent
four-population assembly engine (`assemble_trial`), the concrete
`AssemblingPhase5Sink`, `MinerTrialAssembly`, and the D1 exception types.

The fixture drives the REAL post-D1.0 lifecycle (D0 harness pattern): a real
coordinator + real durable ledger, real assigned stripes, real staged spool files
on disk written through `stage_inline_shard`, real `publish_attempt` ->
`Phase5Sink.publish_shard`, and real `coordinator.commit_trial` /
`coordinator.abort_trial` reaching the sink. It is TWO-MODE (workflow phases
1/2/3/4), MULTI-STRIPE (2 macro stripes per phase) and MULTI-SUB-STRIPE (2
sub-stripes per stripe), with hand-computable populations: per mode at least one
seed in F∩R, one in F−R and one in R−F, all with distinct match rates so `score`
averaging is observable.

Direct `sink.*` calls appear ONLY in defense-in-depth probes the post-D1.0
coordinator cannot produce; each is labeled `DIRECT SINK INVARIANT-BREAK PROBE`
[TB-D1-GC1]. Engine-level probes (G11-G14) call `assemble_trial` directly on
mutated copies of REAL published manifests — that is the engine's own module-level
surface, not a bypass of the producer.

Spool opens are instrumented by wrapping the engine's single read seam
(`range_miner_npz_writer._read_spool_bytes`), so "zero spool opens" is measured,
not assumed.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 tests/test_s172_phase5_d1_engine.py
Exit 0 = all green. Exit 1 = a check failed (DO NOT COMMIT).
"""
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import traceback
from contextlib import contextmanager

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from miner.range_miner_coordinator import (  # noqa: E402
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    RangeMinerCoordinator,
    TrialAborted,
    workflow_stages_for,
)
from miner.range_miner_worker import (  # noqa: E402
    build_substripe_payload_bytes,
    supported_variants,
)
from miner import range_miner_npz_writer as npzw  # noqa: E402
from miner.range_miner_npz_writer import (  # noqa: E402
    CANONICAL_RECORD_FIELDS,
    AssemblingPhase5Sink,
    AssemblyConsistencyError,
    AssemblyStateError,
    DirectionalDuplicateError,
    ManifestReplayConflict,
    MinerTrialAssembly,
    PhaseIdentityError,
    SpoolIdentityError,
    assemble_trial,
)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_results = []

# Bounds below are FAILURE DETECTORS, never synchronization.
_EV_TIMEOUT = 30.0
_JOIN_TIMEOUT = 60.0

SPOOL_ROOT = "/var/spool/miner"
PRNG_BASE = "java_lcg"

# Small caps + a small macro-stripe size so the REAL producer path yields a
# multi-stripe, multi-sub-stripe run: 40 seeds / macro 20 = 2 stripes per phase;
# 20 seeds / cap 10 = 2 sub-stripes per stripe.
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

# Hand-chosen populations. Every rate is distinct so `score` averaging is
# observable, and every seed sits inside exactly one sub-stripe range
# ([0,10), [10,20), [20,30), [30,40)).
#
#   CONSTANT  F = {1, 12, 25, 33}   R = {1, 12, 26}
#             F∩R = {1, 12}   F−R = {25, 33}   R−F = {26}
#   VARIABLE  F = {2, 15, 27}       R = {2, 7, 15, 38}
#             F∩R = {2, 15}   F−R = {27}       R−F = {7, 38}
#
# The two modes are deliberately ASYMMETRIC in opposite directions (constant
# |F|>|R|, variable |F|<|R|) so a forward/reverse swap or a constant/variable
# cross-wire cannot pass the derived-field equality assertions.
#
# Phase 2 contributes no survivor in [30,40), phase 3 none in [30,40) and phase 4
# none in [20,30), so the fixture also exercises legitimately EMPTY survivor
# lists (§5.4/G15).
PHASE_POP = {
    1: {1: 0.90, 12: 0.80, 25: 0.70, 33: 0.60},
    2: {1: 0.50, 12: 0.40, 26: 0.30},
    3: {2: 0.95, 15: 0.85, 27: 0.75},
    4: {2: 0.55, 7: 0.65, 15: 0.45, 38: 0.35},
}

# [WINDOW-ANCHOR BRIEF I] Context gains `window_anchor` + `generator_phase`.
# `offset` is retained because the RECORD field keeps that name (frozen canonical
# array 4, TB wire-name ruling) and this dict doubles as the expected-record
# source at :386. Fixture-only: no assertion changed.
CTX = dict(trial_number=7, window_size=5, offset=2, window_anchor=2,
           generator_phase=0,
           sessions=["midday", "evening"], skip_min=1, skip_max=9,
           prng_base=PRNG_BASE, forward_threshold=0.40, reverse_threshold=0.45,
           dataset_sha256="d" * 64, residue_sha256="r" * 64)


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:  # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


def _raises(exc, fn, *a, **kw):
    """Assert `fn` raises `exc` (exactly that class or a subclass) and return it."""
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    except Exception as other:  # noqa: BLE001
        raise AssertionError(
            f"expected {exc.__name__}, got {type(other).__name__}: {other}")
    raise AssertionError(f"expected {exc.__name__}, nothing was raised")


# ---------------------------------------------------------------------------
# Instrumentation — count (and optionally hold open) the engine's spool reads
# through its single read seam. "Zero spool opens" is therefore MEASURED.
# ---------------------------------------------------------------------------
class _ReadCounter:
    def __init__(self):
        self.count = 0
        self.paths = []


@contextmanager
def _count_opens(hook=None):
    """Wrap `npzw._read_spool_bytes`. `hook(path, index)` (optional) runs BEFORE
    the real read so a gate can hold a read open deterministically."""
    counter = _ReadCounter()
    real = npzw._read_spool_bytes

    def wrapped(path):
        idx = counter.count
        counter.count += 1
        counter.paths.append(path)
        if hook is not None:
            hook(path, idx)
        return real(path)

    npzw._read_spool_bytes = wrapped
    try:
        yield counter
    finally:
        npzw._read_spool_bytes = real


class _CountingSink(AssemblingPhase5Sink):
    """The REAL sink under test, with call counters so a gate can assert that a
    coordinator-level duplicate never re-enters the sink. No behavior is
    overridden — each method delegates to the production implementation."""

    def __init__(self):
        super().__init__()
        self.publish_calls = 0
        self.commit_calls = 0
        self.abort_calls = 0
        self.events = []
        self._events_lock = threading.Lock()

    def _log(self, tag):
        with self._events_lock:
            self.events.append(tag)

    def publish_shard(self, manifest):
        self.publish_calls += 1
        return super().publish_shard(manifest)

    def commit_trial(self, event):
        self.commit_calls += 1
        self._log("commit_enter")
        try:
            return super().commit_trial(event)
        finally:
            self._log("commit_return")

    def abort_trial(self, event):
        self.abort_calls += 1
        self._log("abort_enter")
        try:
            return super().abort_trial(event)
        finally:
            self._log("abort_return")


# ---------------------------------------------------------------------------
# Real-lifecycle fixture (D0 harness pattern, post-D1.0 producer)
# ---------------------------------------------------------------------------
def _coord(tmp, sink, dbname="l.db"):
    ledger = MinerLedger(os.path.join(tmp, dbname))
    # The advertised worker caps must EQUAL the coordinator's central config or
    # register_worker quarantines the worker (_validate_caps, coordinator:1680).
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


def _survivor_entries(phase, seed_start, seed_count, pop=None):
    """The worker's canonical survivor tuples for one sub-stripe range
    (miner/range_miner_worker.py:881-899). Constant passes emit
    (seed, rate, null, [best_skip]); hybrid passes emit
    (seed, rate, strategy_id, skip_sequence)."""
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


# The sink under test is the production class; the harness needs to read back
# what it retained. Rather than add a test-only accessor to production code, read
# its documented internal accumulation (§4.1) directly.
def _retained(sink, run_id):
    state = sink._runs.get(run_id)
    return [] if state is None else list(state.manifests.values())


def _build_run(tmp, sink, run_id, phases=(1, 2, 3, 4), pops=None, dbname="l.db"):
    """Drive a REAL trial through the REAL producer surface up to (but not
    including) the terminal commit: durable context, real assigned stripes, real
    staged+verified spool files on disk, real publish_attempt -> publish_shard.

    Returns (coord, published_manifests)."""
    coord = _coord(tmp, sink, dbname)
    coord.ledger.create_trial(run_id, CTX["trial_number"], now=100.0)
    coord.ledger.set_trial_context(run_id, dict(CTX))
    conn = _register(coord)
    published = []
    for phase in phases:
        family = PHASE_TABLE[phase][0]
        recs = coord.assign_stripes(run_id, family, phase, TOTAL_SEEDS, [conn],
                                    stripe_prefix=f"{run_id}__p{phase}", now=100.0)
        assert len(recs) == TOTAL_SEEDS // MACRO_SIZE, recs      # multi-stripe
        for rec in recs:
            assert rec["claimed"], rec
            assert rec["expected_substripes"] == MACRO_SIZE // SUB_CAP, rec
            sid = rec["stripe_id"]
            survivors_total = 0
            for sub_index in range(rec["expected_substripes"]):
                s_start = rec["seed_start"] + sub_index * SUB_CAP
                entries = _survivor_entries(
                    phase, s_start, SUB_CAP,
                    None if pops is None else pops.get(phase, {}))
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
            before = len(_retained(sink, run_id))
            coord.finalize_stripe(run_id, sid, now=100.0)
            after = _retained(sink, run_id)
            assert len(after) == before + rec["expected_substripes"], \
                f"{sid}: expected {rec['expected_substripes']} manifests"
            published = after
    return coord, published


def _expected_constant():
    """HAND-COMPUTED constant-mode expectations — literals, not a second
    implementation of §5.5.

        F = {1, 12, 25, 33}  |F| = 4        R = {1, 12, 26}  |R| = 3
        F∩R = {1, 12} -> 2                  F∪R = {1,12,25,33,26} -> 5
        F−R = {25, 33} -> 2                 R−F = {26} -> 1
    """
    assert set(PHASE_POP[1]) == {1, 12, 25, 33}
    assert set(PHASE_POP[2]) == {1, 12, 26}
    return {
        "forward_map": dict(PHASE_POP[1]), "reverse_map": dict(PHASE_POP[2]),
        "bidirectional": {1, 12},
        "forward_count": 4, "reverse_count": 3, "bidirectional_count": 2,
        "intersection_count": 2,
        "intersection_ratio": 2 / 5,          # |F∩R| / max(|F∪R|, 1)
        "forward_only_count": 2, "reverse_only_count": 1,
        "survivor_overlap_ratio": 2 / 4,      # |F∩R| / max(|F|, 1)
        "bidirectional_selectivity": 4 / 3,   # |F|   / max(|R|, 1)
        "intersection_weight": 2 / 7,         # |F∩R| / max(|F|+|R|, 1)
    }


def _expected_variable():
    """HAND-COMPUTED variable-mode expectations.

        F = {2, 15, 27}  |F| = 3            R = {2, 7, 15, 38}  |R| = 4
        F∩R = {2, 15} -> 2                  F∪R = {2,15,27,7,38} -> 5
        F−R = {27} -> 1                     R−F = {7, 38} -> 2
    """
    assert set(PHASE_POP[3]) == {2, 15, 27}
    assert set(PHASE_POP[4]) == {2, 7, 15, 38}
    return {
        "forward_map": dict(PHASE_POP[3]), "reverse_map": dict(PHASE_POP[4]),
        "bidirectional": {2, 15},
        "forward_count": 3, "reverse_count": 4, "bidirectional_count": 2,
        "intersection_count": 2,
        "intersection_ratio": 2 / 5,
        "forward_only_count": 1, "reverse_only_count": 2,
        "survivor_overlap_ratio": 2 / 3,
        "bidirectional_selectivity": 3 / 4,
        "intersection_weight": 2 / 7,
    }


def _assert_mode(records, fwd_map, rev_map, bidi, expect, skip_mode, prng_type):
    assert fwd_map == expect["forward_map"], (fwd_map, expect["forward_map"])
    assert rev_map == expect["reverse_map"], (rev_map, expect["reverse_map"])
    assert bidi == expect["bidirectional"], (bidi, expect["bidirectional"])
    assert [r["seed"] for r in records] == sorted(expect["bidirectional"]), records
    for rec in records:
        seed = rec["seed"]
        assert rec["forward_match_rate"] == expect["forward_map"][seed]
        assert rec["reverse_match_rate"] == expect["reverse_map"][seed]
        assert rec["score"] == (expect["forward_map"][seed]
                                + expect["reverse_map"][seed]) / 2.0, rec
        # trial-global, straight from the consistency-checked metadata
        assert rec["window_size"] == CTX["window_size"]
        assert rec["offset"] == CTX["offset"]
        assert rec["skip_min"] == CTX["skip_min"]
        assert rec["skip_max"] == CTX["skip_max"]
        assert rec["skip_range"] == CTX["skip_max"] - CTX["skip_min"] == 8
        assert rec["sessions"] == CTX["sessions"]
        assert rec["trial_number"] == CTX["trial_number"]
        assert rec["prng_base"] == PRNG_BASE
        assert rec["skip_mode"] == skip_mode
        assert rec["prng_type"] == prng_type
        # every §5.5 derived field, hand-computed
        for k in ("forward_count", "reverse_count", "bidirectional_count",
                  "intersection_count", "intersection_ratio",
                  "forward_only_count", "reverse_only_count",
                  "survivor_overlap_ratio", "bidirectional_selectivity",
                  "intersection_weight"):
            assert rec[k] == expect[k], (skip_mode, k, rec[k], expect[k])


# ---------------------------------------------------------------------------
# G1 — hand-computed assembly, both modes, + mis-grouping tripwire
# ---------------------------------------------------------------------------
def g1_hand_computed_assembly():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        coord, manifests = _build_run(tmp, sink, "g1")
        # a genuine two-mode, multi-stripe, multi-sub-stripe fixture
        assert len(manifests) == 4 * 2 * 2, len(manifests)
        assert {int(m["workflow_phase"]) for m in manifests} == {1, 2, 3, 4}
        assert len({m["stripe_id"] for m in manifests}) == 8
        assert {m["sub_index"] for m in manifests} == {0, 1}

        with _count_opens() as reads:
            ev = coord.commit_trial("g1", now=200.0)
        assert ev["delivery"] == "done", ev
        assert reads.count == len(manifests), \
            f"every manifest's spool must be read exactly once: {reads.count}"

        a = coord.phase5_sink.get_assembly("g1")
        assert isinstance(a, MinerTrialAssembly), a
        exp_c, exp_v = _expected_constant(), _expected_variable()
        _assert_mode(a.canonical_records_constant, a.forward_map_constant,
                     a.reverse_map_constant, a.bidirectional_constant,
                     exp_c, "constant", "java_lcg")
        _assert_mode(a.canonical_records_variable, a.forward_map_variable,
                     a.reverse_map_variable, a.bidirectional_variable,
                     exp_v, "variable", "java_lcg_hybrid")
        assert a.directional_counts == {
            "forward_constant": 4, "reverse_constant": 3,
            "forward_variable": 3, "reverse_variable": 4,
            "bidirectional_constant": 2, "bidirectional_variable": 2}, \
            a.directional_counts
        assert isinstance(a.timing.get("assembly_s"), float)
        assert a.binary_npz_path is None and a.all_npz_path is None, \
            "D1 must not claim an NPZ path (None = not produced yet)"

        # --- MIS-GROUPING TRIPWIRE ------------------------------------------
        # (a) the perturbed run must RAISE: flip one P2 manifest's explicit
        #     direction, and (separately) its manifest-level workflow_phase.
        bad_dir = copy.deepcopy(manifests)
        p2 = next(m for m in bad_dir if int(m["workflow_phase"]) == 2)
        p2["trial_metadata"]["direction"] = "forward"
        _raises(PhaseIdentityError, assemble_trial, "g1", bad_dir)

        bad_phase = copy.deepcopy(manifests)
        p2b = next(m for m in bad_phase if int(m["workflow_phase"]) == 2)
        p2b["workflow_phase"] = 1
        _raises(PhaseIdentityError, assemble_trial, "g1", bad_phase)

        # (b) the equality gate must FAIL on that mis-grouping: had the engine
        #     grouped ANY reverse-constant shard as forward/constant, the
        #     forward map would gain reverse-only seed 26 and the derived fields
        #     would move. Assert the equality assertions above are discriminating.
        mis_grouped = dict(a.forward_map_constant)
        mis_grouped.update(a.reverse_map_constant)
        assert mis_grouped != a.forward_map_constant, \
            "a mis-grouped forward map must differ from the verified one"
        assert mis_grouped != exp_c["forward_map"], \
            "the G1 equality gate would fail on mis-grouping (it is discriminating)"
        assert 26 in mis_grouped and 26 not in a.forward_map_constant


# ---------------------------------------------------------------------------
# G2 / G3 — get_assembly is None before commit; exactly one complete result after
# ---------------------------------------------------------------------------
def g2_none_before_commit():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        _coord_obj, manifests = _build_run(tmp, sink, "g2")
        assert manifests, "manifests must have been published"
        assert sink.get_assembly("g2") is None, \
            "no assembly may exist before a successful commit"
        assert sink.commit_calls == 0
        assert sink.get_assembly("no-such-run") is None


def g3_commit_installs_one_assembly():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        coord, manifests = _build_run(tmp, sink, "g3")
        ev = coord.commit_trial("g3", now=200.0)
        assert ev["delivery"] == "done", ev
        assert ev["event_id"] == "g3:commit", ev
        assert sink.commit_calls == 1
        a = sink.get_assembly("g3")
        assert isinstance(a, MinerTrialAssembly)
        assert a.run_id == "g3"
        assert sink.get_assembly("g3") is a, "the SAME result object is returned"
        # complete: every population + record list populated for both modes
        assert a.forward_map_constant and a.reverse_map_constant
        assert a.forward_map_variable and a.reverse_map_variable
        assert a.canonical_records_constant and a.canonical_records_variable
        assert coord.ledger.get_trial("g3")["commit_delivery_status"] == "done"


# ---------------------------------------------------------------------------
# G4 — duplicate commit: A real coordinator idempotence; B sink-level contract
# ---------------------------------------------------------------------------
def g4_duplicate_commit():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        coord, _m = _build_run(tmp, sink, "g4")
        ev = coord.commit_trial("g4", now=200.0)
        assert ev["delivery"] == "done", ev
        first = sink.get_assembly("g4")
        assert sink.commit_calls == 1

        # --- G4-A: REAL coordinator duplicate ------------------------------
        with _count_opens() as reads:
            ev2 = coord.commit_trial("g4", now=201.0)
        assert ev2.get("duplicate") is True, ev2
        assert ev2["delivery"] == "done", ev2
        assert ev2["event_id"] == "g4:commit", ev2
        assert sink.commit_calls == 1, \
            "the coordinator must NOT replay a 'done' delivery to the sink"
        assert reads.count == 0, "zero spool opens on a duplicate commit"
        assert sink.get_assembly("g4") is first, "no map reconstruction"

        # --- G4-B: DIRECT SINK INVARIANT-BREAK PROBE [TB-D1-GC1] -----------
        # The real coordinator never replays a consumed commit event to the sink
        # (coordinator:2941-2943); this probes the §4.3 sink-level idempotence
        # contract directly.
        with _count_opens() as reads2:
            sink.commit_trial({"event_type": "trial_commit", "run_id": "g4",
                               "event_id": "g4:commit"})
        assert reads2.count == 0, "consumed event_id replay: zero spool opens"
        assert sink.get_assembly("g4") is first, \
            "the stored assembly must remain the SAME object"
        assert sink.commit_calls == 2   # the probe itself


# ---------------------------------------------------------------------------
# G5 — failed assembly is retryable through the REAL coordinator sequence
# ---------------------------------------------------------------------------
def g5_failed_assembly_retryable():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        coord, manifests = _build_run(tmp, sink, "g5")
        victim = manifests[0]
        path = victim["local_spool_path"]
        with open(path, "rb") as f:
            good = f.read()

        # 1. corrupt one staged spool's bytes AFTER publish
        with open(path, "wb") as f:
            f.write(good + b"  ")

        # 2. the coordinator CATCHES the sink raise; the call does not raise
        ev = coord.commit_trial("g5", now=200.0)
        assert ev["delivery"] == "failed", ev
        assert "error" in ev and ev["error"], ev
        # 3. same immutable event_id
        assert ev["event_id"] == "g5:commit", ev
        # 4. no result, no consumed-commit marker
        assert sink.get_assembly("g5") is None
        assert sink._runs["g5"].consumed_commits == set(), sink._runs["g5"]
        assert sink._runs["g5"].result is None
        # 5. accumulated manifests RETAINED for redelivery (§4.0)
        assert len(_retained(sink, "g5")) == len(manifests), \
            "a failed assembly must retain its manifests"
        assert coord.ledger.get_trial("g5")["commit_delivery_status"] == "failed"

        # 6. repair
        with open(path, "wb") as f:
            f.write(good)

        # 7-8. redeliver the SAME event -> done, one completed result
        ev2 = coord.commit_trial("g5", now=201.0)
        assert ev2["event_id"] == "g5:commit", ev2
        assert ev2["delivery"] == "done", ev2
        assert ev2.get("duplicate") is not True, "this is a redelivery, not a no-op"
        a = sink.get_assembly("g5")
        assert isinstance(a, MinerTrialAssembly)
        assert a.bidirectional_constant == _expected_constant()["bidirectional"]
        assert sink._runs["g5"].consumed_commits == {"g5:commit"}


# ---------------------------------------------------------------------------
# G6 / G7 — replay + slot-conflict rules (§4.2.4/§4.2.5)
# ---------------------------------------------------------------------------
def g6_identical_replay_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        _coord_obj, manifests = _build_run(tmp, sink, "g6")
        before = len(_retained(sink, "g6"))
        original = manifests[0]

        # canonically identical: the SAME content, re-encoded with a different
        # key order (and a rebuilt nested metadata dict).
        replay = json.loads(json.dumps(original))
        replay["trial_metadata"] = {k: replay["trial_metadata"][k]
                                    for k in reversed(list(replay["trial_metadata"]))}
        replay = {k: replay[k] for k in reversed(list(replay))}
        with _count_opens() as reads:
            sink.publish_shard(replay)          # no raise
        assert reads.count == 0, "publish must never touch a spool"
        assert len(_retained(sink, "g6")) == before, "no duplicate accumulation"


def g7_replay_and_slot_conflicts():
    """DIRECT SINK INVARIANT-BREAK PROBES [TB-D1-GC1]: the post-D1.0 coordinator
    publishes each verified shard exactly once and never republishes, so these
    conflicts are unreachable through the real producer. They are the sink's
    defense-in-depth contract (§4.2.4/§4.2.5)."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        coord, manifests = _build_run(tmp, sink, "g7")
        base = manifests[0]

        # (a) same event_id + DIFFERENT content -> ManifestReplayConflict
        diff = copy.deepcopy(base)
        diff["expected_size"] = int(diff["expected_size"]) + 1
        _raises(ManifestReplayConflict, sink.publish_shard, diff)

        # (b) different event_id, same (run_id, stripe_id, sub_index), IDENTICAL
        #     bytes + SHA -> still ManifestReplayConflict
        same_slot = copy.deepcopy(base)
        same_slot["event_id"] = base["event_id"] + ":dup"
        assert same_slot["expected_sha256"] == base["expected_sha256"]
        assert same_slot["expected_size"] == base["expected_size"]
        _raises(ManifestReplayConflict, sink.publish_shard, same_slot)

        # the conflicts changed nothing
        assert len(_retained(sink, "g7")) == len(manifests)
        ev = coord.commit_trial("g7", now=200.0)
        assert ev["delivery"] == "done", ev
        stored = sink.get_assembly("g7")

        # (c) post-commit NEW event / NEW logical shard -> AssemblyStateError
        new_slot = copy.deepcopy(base)
        new_slot["event_id"] = base["event_id"] + ":new"
        new_slot["sub_index"] = 99
        with _count_opens() as reads:
            _raises(AssemblyStateError, sink.publish_shard, new_slot)
        assert reads.count == 0

        # (d) post-commit DIFFERENT commit event_id -> AssemblyStateError, and NO
        #     replacement assembly (stored result unchanged, zero spool opens)
        with _count_opens() as reads2:
            _raises(AssemblyStateError, sink.commit_trial,
                    {"event_type": "trial_commit", "run_id": "g7",
                     "event_id": "g7:commit:other"})
        assert reads2.count == 0, "a refused commit must not read any spool"
        assert sink.get_assembly("g7") is stored, "no replacement assembly"


# ---------------------------------------------------------------------------
# G8 — abort / tombstone (split, [TB-D1-B4])
# ---------------------------------------------------------------------------
def g8_abort_tombstone():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        coord, manifests = _build_run(tmp, sink, "g8")
        assert _retained(sink, "g8")

        # (1) REAL coordinator abort -> sink state cleared + tombstoned
        res = coord.abort_trial("g8", reason="gate", now=200.0)
        assert res["cleanup"] == "done", res
        assert sink.abort_calls == 1
        assert _retained(sink, "g8") == [], "manifests must be discarded"
        assert sink.get_assembly("g8") is None
        assert "g8" in sink._tombstoned

        # a stale publish after abort is IGNORED with zero spool opens
        with _count_opens() as reads:
            sink.publish_shard(copy.deepcopy(manifests[0]))
        assert reads.count == 0
        assert _retained(sink, "g8") == [], "a stale manifest must not accumulate"
        assert sink.get_assembly("g8") is None

        # (2) REAL coordinator commit after abort raises AT THE COORDINATOR
        before = sink.commit_calls
        _raises(TrialAborted, coord.commit_trial, "g8", 201.0)
        assert sink.commit_calls == before, \
            "a post-abort commit must never reach the sink"

        # (3) DIRECT SINK INVARIANT-BREAK PROBE [TB-D1-GC1]: the coordinator
        #     raises TrialAborted first, so only a direct call can reach here.
        _raises(AssemblyStateError, sink.commit_trial,
                {"event_type": "trial_commit", "run_id": "g8",
                 "event_id": "g8:commit"})

        # aborting an unknown / already-aborted run is a successful no-op
        sink.abort_trial({"event_type": "trial_abort", "run_id": "g8",
                          "event_id": "g8:abort"})
        sink.abort_trial({"event_type": "trial_abort", "run_id": "never-seen",
                          "event_id": "never-seen:abort"})


# ---------------------------------------------------------------------------
# G9 — the frozen 24-field canonical record, exact keys, exact order, ascending
#
# INDEPENDENT ORACLE (Team Alpha D1.1 review). The pre-correction G9 imported
# `CANONICAL_RECORD_FIELDS` from the module under test and asserted the records
# against it — circular: a REORDERED or renamed production constant still passed,
# because the records are BUILT from that same constant
# (range_miner_npz_writer._mode_records). The oracle below is transcribed BY HAND
# from docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D1.md REV5 §6 (itself frozen
# from the live insertion order at window_optimizer_integration_final.py:683-694
# + :652-676 for constant, :785-796 + :756-780 for hybrid).
#
# It MUST NEVER be derived from, sorted against, or regenerated out of the module
# under test. If production and this oracle ever disagree, that is the gate doing
# its job — resolve it against REV5 §6, never by editing this tuple to match.
# ---------------------------------------------------------------------------
_G9_RECORD_FIELDS_ORACLE = (
    "seed", "forward_match_rate", "reverse_match_rate", "score",
    "window_size", "offset", "skip_min", "skip_max", "skip_range", "sessions",
    "trial_number", "prng_base", "skip_mode", "prng_type",
    "forward_count", "reverse_count", "bidirectional_count",
    "intersection_count", "intersection_ratio",
    "forward_only_count", "reverse_only_count",
    "survivor_overlap_ratio", "bidirectional_selectivity", "intersection_weight",
)


def g9_frozen_record_shape():
    oracle = _G9_RECORD_FIELDS_ORACLE
    assert len(oracle) == 24, oracle
    assert len(set(oracle)) == 24, "the oracle itself must have no duplicates"
    # threshold_used is validation metadata, NOT a 25th field (§6, G10)
    assert "threshold_used" not in oracle

    # (1) the PRODUCTION constant matches the independent oracle — membership AND
    #     order. This is what a reordered/renamed constant now fails.
    assert CANONICAL_RECORD_FIELDS == oracle, (
        f"production CANONICAL_RECORD_FIELDS drifted from REV5 §6:\n"
        f"  production: {CANONICAL_RECORD_FIELDS}\n"
        f"  oracle:     {oracle}")

    # (2) every emitted record matches the ORACLE directly, so the gate holds
    #     even if the production constant were bypassed entirely.
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        coord, _m = _build_run(tmp, sink, "g9")
        coord.commit_trial("g9", now=200.0)
        a = sink.get_assembly("g9")
        for mode, records in (("constant", a.canonical_records_constant),
                              ("variable", a.canonical_records_variable)):
            assert records, f"{mode} records must be non-empty in this fixture"
            for rec in records:
                assert tuple(rec.keys()) == oracle, \
                    f"{mode}: key set/order drift vs REV5 §6: {tuple(rec.keys())}"
                assert set(rec) == set(oracle), \
                    f"{mode}: key membership drift vs REV5 §6"
                assert len(rec) == 24, f"{mode}: {len(rec)} keys, expected 24"
            seeds = [r["seed"] for r in records]
            assert seeds == sorted(seeds), f"{mode} records must ascend by seed"


# ---------------------------------------------------------------------------
# G10 — threshold_used is validated per direction and is NOT a record field
# ---------------------------------------------------------------------------
def g10_threshold_used():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        coord, manifests = _build_run(tmp, sink, "g10")
        coord.commit_trial("g10", now=200.0)
        a = sink.get_assembly("g10")
        for rec in a.canonical_records_constant + a.canonical_records_variable:
            assert "threshold_used" not in rec, \
                "threshold_used is validation metadata, never a 25th record field"

        # flipping one manifest's threshold_used -> PhaseIdentityError
        for phase in (1, 2, 3, 4):
            bad = copy.deepcopy(manifests)
            m = next(x for x in bad if int(x["workflow_phase"]) == phase)
            meta = m["trial_metadata"]
            other = (meta["reverse_threshold"] if meta["direction"] == "forward"
                     else meta["forward_threshold"])
            assert meta["threshold_used"] != other
            meta["threshold_used"] = other
            _raises(PhaseIdentityError, assemble_trial, "g10", bad)


# ---------------------------------------------------------------------------
# G11 — the identity matrix: each corruption raises PhaseIdentityError
# ---------------------------------------------------------------------------
def g11_identity_matrix():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        _coord_obj, manifests = _build_run(tmp, sink, "g11")

        # negative control: the untouched set assembles cleanly (so the matrix
        # below cannot pass by always raising).
        assert isinstance(assemble_trial("g11", copy.deepcopy(manifests)),
                          MinerTrialAssembly)

        def _meta_mut(key, value):
            def apply(ms):
                ms[0]["trial_metadata"][key] = value
            return apply

        def _top_mut(key, value):
            def apply(ms):
                ms[0][key] = value
            return apply

        def _both_phase(value):
            """Mutate BOTH phase copies so the manifest stays internally
            consistent and the workflow_phase_semantics ORACLE is what fires."""
            def apply(ms):
                ms[0]["workflow_phase"] = value
                ms[0]["trial_metadata"]["workflow_phase"] = value
            return apply

        # (label, mutation, the fragment proving WHICH check fired). The
        # fragments matter: without them a regression that collapsed the matrix
        # onto one shared early check would still show 13 green cases.
        cases = [
            ("direction",                 _meta_mut("direction", "reverse"),
             "workflow_phase_semantics"),
            ("skip_mode",                 _meta_mut("skip_mode", "variable"),
             "workflow_phase_semantics"),
            ("prng_type",                 _meta_mut("prng_type", "java_lcg_hybrid"),
             "prng_type"),
            ("family_name",               _meta_mut("family_name", "java_lcg_reverse"),
             "workflow_stages_for"),
            ("workflow_phase (manifest)", _top_mut("workflow_phase", 3),
             "!= trial_metadata workflow_phase"),
            ("workflow_phase (metadata)", _meta_mut("workflow_phase", 3),
             "!= trial_metadata workflow_phase"),
            ("run_id",                    _top_mut("run_id", "some-other-run"),
             "!= assembly run_id"),
            # single-copy provenance divergence: §5.1's lifted-vs-nested check
            # fires FIRST (the dual-copy case is G13) [TB-D1-G13C]
            ("lifted dataset_sha256",     _top_mut("dataset_sha256", "e" * 64),
             "lifted dataset_sha256"),
            ("lifted residue_sha256",     _top_mut("residue_sha256", "f" * 64),
             "lifted residue_sha256"),
            ("nested dataset_sha256",     _meta_mut("dataset_sha256", "e" * 64),
             "lifted dataset_sha256"),
            ("nested residue_sha256",     _meta_mut("residue_sha256", "f" * 64),
             "lifted residue_sha256"),
            ("threshold_used",            _meta_mut("threshold_used", 0.99),
             "threshold_used"),
            ("unknown workflow_phase",    _both_phase(9),
             "unresolvable workflow_phase"),
        ]
        for label, mutate, fragment in cases:
            bad = copy.deepcopy(manifests)
            # index 0 is a phase-1 manifest; assert that before mutating it.
            assert int(bad[0]["workflow_phase"]) == 1, bad[0]["workflow_phase"]
            mutate(bad)
            e = _raises(PhaseIdentityError, assemble_trial, "g11", bad)
            assert fragment in str(e), \
                f"{label}: raised the WRONG check — {str(e)!r} lacks {fragment!r}"

        # the oracles never silently substitute: the family map is DERIVED from
        # the imported producer authority, not a hand-built suffix table.
        assert npzw._phase_family_map(PRNG_BASE) == \
            {ph: fam for fam, ph in workflow_stages_for(PRNG_BASE, True)}


# ---------------------------------------------------------------------------
# G12 — spool identity + semantics + containers [TB-D1-B5, TB-D1-PV]
# ---------------------------------------------------------------------------
def _repoint(manifest, tmp, obj_or_bytes, tag):
    """Write a mutated payload to a NEW staged path and repoint a COPY of the
    manifest at it, RECOMPUTING expected_size + expected_sha256 so the gate
    exercises the validator rather than merely the digest check (§9 G12)."""
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


def g12_spool_identity_and_semantics():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        _coord_obj, manifests = _build_run(tmp, sink, "g12")

        # a phase-1 manifest whose sub_index is 1 (so `True == 1` would slip
        # through an equality-only sub_index check) and whose range is [10,20).
        target = next(m for m in manifests
                      if int(m["workflow_phase"]) == 1 and m["sub_index"] == 1
                      and m["stripe_id"].endswith("_s0"))
        with open(target["local_spool_path"]) as f:
            payload = json.load(f)
        assert payload["sub_index"] == 1 and payload["seed_start"] == 10
        assert payload["survivors"], "the target shard must carry survivors"

        def mutate(fn, tag):
            obj = copy.deepcopy(payload)
            fn(obj)
            return _repoint(target, tmp, obj, tag)

        def _set_entry(obj, idx, value):
            obj["survivors"][0][idx] = value

        # label -> (manifest repointed at the mutated spool, the fragment proving
        # WHICH check fired — so a collapse onto one shared check cannot pass).
        cases = {
            "stripe_id":        (mutate(lambda o: o.__setitem__("stripe_id", "not-my-stripe"), "stripe"),
                                 "payload stripe_id"),
            "sub_index value":  (mutate(lambda o: o.__setitem__("sub_index", 0), "subidx"),
                                 "!= manifest sub_index"),
            "schema_version":   (mutate(lambda o: o.__setitem__("schema_version", "s172_substripe_v2"), "schema"),
                                 "schema_version"),
            "tuple shape":      (mutate(lambda o: o["survivors"].__setitem__(0, [11, 0.5, None]), "shape"),
                                 "not a 4-element list"),
            "bool seed":        (mutate(lambda o: _set_entry(o, 0, True), "boolseed"),
                                 "seed True is not an integer"),
            "out-of-range seed": (mutate(lambda o: _set_entry(o, 0, 99), "range"),
                                  "outside the declared sub-stripe range"),
            "non-finite rate":  (mutate(lambda o: _set_entry(o, 1, float("inf")), "inf"),
                                 "is not finite"),
            "rate > 1.0":       (mutate(lambda o: _set_entry(o, 1, 1.5), "hirate"),
                                 "outside [0.0, 1.0]"),
            "bool strategy_id": (mutate(lambda o: _set_entry(o, 2, True), "boolstrat"),
                                 "strategy_id True"),
            "bool skip value":  (mutate(lambda o: _set_entry(o, 3, [True]), "boolskip"),
                                 "skip[0] True"),
            "survivors not a list": (mutate(lambda o: o.__setitem__("survivors", {"a": 1}), "survdict"),
                                     "survivors is dict"),
            "sub_index is bool":    (mutate(lambda o: o.__setitem__("sub_index", True), "boolsub"),
                                     "sub_index True is not an integer"),
            "skip_sequence not a list": (mutate(lambda o: _set_entry(o, 3, 7), "skipint"),
                                         "skip_sequence is int"),
            "missing key":      (mutate(lambda o: o.pop("seed_count"), "misskey"),
                                 "missing mandatory key"),
            # root is not a dict / invalid JSON: size+sha recomputed over the
            # replacement bytes, so only the validator can fire.
            "root not a dict":  (_repoint(target, tmp, [1, 2, 3], "rootlist"),
                                 "root is list"),
            "invalid JSON":     (_repoint(target, tmp, b"{not json at all", "badjson"),
                                 "not decodable JSON"),
        }
        for label, (bad_manifest, fragment) in cases.items():
            ms = [bad_manifest if m["event_id"] == target["event_id"]
                  else copy.deepcopy(m) for m in manifests]
            e = _raises(SpoolIdentityError, assemble_trial, "g12", ms)
            assert fragment in str(e), \
                f"{label}: raised the WRONG check — {str(e)!r} lacks {fragment!r}"

        # negative control: repointing at an UNMODIFIED copy still assembles, so
        # the matrix is not passing merely because the path changed.
        ok = _repoint(target, tmp, payload, "ok")
        ms = [ok if m["event_id"] == target["event_id"] else copy.deepcopy(m)
              for m in manifests]
        a = assemble_trial("g12", ms)
        assert a.forward_map_constant == _expected_constant()["forward_map"]


# ---------------------------------------------------------------------------
# G13 — cross-manifest consistency + phase-set completeness [TB-D1-G13C]
# ---------------------------------------------------------------------------
def g13_cross_manifest_consistency():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        _coord_obj, manifests = _build_run(tmp, sink, "g13")

        # (a) DUAL-COPY provenance mutation: the manifest stays internally valid
        #     (§5.1 passes) but its canonical 11-field context diverges.
        for prov, value in (("dataset_sha256", "e" * 64),
                            ("residue_sha256", "f" * 64)):
            bad = copy.deepcopy(manifests)
            bad[0][prov] = value
            bad[0]["trial_metadata"][prov] = value
            _raises(AssemblyConsistencyError, assemble_trial, "g13", bad)

        # (b) a trial-global divergence in the 9 non-provenance fields
        for field_name, value in (("window_size", 99), ("offset", 77),
                                  ("trial_number", 55), ("sessions", ["x"])):
            bad = copy.deepcopy(manifests)
            bad[0]["trial_metadata"][field_name] = value
            _raises(AssemblyConsistencyError, assemble_trial, "g13", bad)

        # (c) INCOMPLETE PHASE SETS [TB-D1-B1] — recording-sink fixtures.
        by_phase = {}
        for m in manifests:
            by_phase.setdefault(int(m["workflow_phase"]), []).append(m)
        for phase_set in ({1}, {1, 2, 3}, {1, 3}, {2}, {3, 4}):
            subset = [copy.deepcopy(m) for p in sorted(phase_set)
                      for m in by_phase[p]]
            e = _raises(AssemblyConsistencyError, assemble_trial, "g13", subset)
            assert "phase" in str(e).lower(), (phase_set, str(e))
        # the two LEGITIMATE sets do not raise
        assert assemble_trial(
            "g13", [copy.deepcopy(m) for p in (1, 2) for m in by_phase[p]])
        assert assemble_trial("g13", copy.deepcopy(manifests))


# ---------------------------------------------------------------------------
# G14 — duplicate seed inside ONE directional population
# ---------------------------------------------------------------------------
def g14_directional_duplicate():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        _coord_obj, manifests = _build_run(tmp, sink, "g14")

        # Engine-level adversarial fixture: make a phase-1 sub-stripe re-emit a
        # seed already owned by ANOTHER phase-1 sub-stripe of the same run (the
        # producer-level fixture is D2). Seed 1 lives in [0,10); repoint the
        # [10,20) shard at a payload that also claims it, with its declared range
        # widened so §5.3's range check passes and the DUPLICATE invariant is what
        # fires.
        first = next(m for m in manifests
                     if int(m["workflow_phase"]) == 1 and m["sub_index"] == 0
                     and m["stripe_id"].endswith("_s0"))
        second = next(m for m in manifests
                      if int(m["workflow_phase"]) == 1 and m["sub_index"] == 1
                      and m["stripe_id"].endswith("_s0"))
        with open(second["local_spool_path"]) as f:
            payload = json.load(f)
        payload["seed_start"] = 0
        payload["seed_count"] = 20
        payload["survivors"].append([1, 0.11, None, [2]])
        dup = _repoint(second, tmp, payload, "dup")

        ms = [dup if m["event_id"] == second["event_id"] else copy.deepcopy(m)
              for m in manifests]
        e = _raises(DirectionalDuplicateError, assemble_trial, "g14", ms)

        # STRUCTURED ATTRIBUTES — asserted directly, never message text.
        assert e.run_id == "g14", e.run_id
        assert e.workflow_phase == 1, e.workflow_phase
        assert e.direction == "forward", e.direction
        assert e.skip_mode == "constant", e.skip_mode
        assert e.seed == 1, e.seed
        assert e.first_stripe == first["stripe_id"], e.first_stripe
        assert e.first_sub_index == 0, e.first_sub_index
        assert e.first_attempt == 0, e.first_attempt
        assert e.first_match_rate == PHASE_POP[1][1], e.first_match_rate
        assert e.dup_stripe == second["stripe_id"], e.dup_stripe
        assert e.dup_sub_index == 1, e.dup_sub_index
        assert e.dup_attempt == 0, e.dup_attempt
        assert e.dup_match_rate == 0.11, e.dup_match_rate
        for attr in ("run_id", "workflow_phase", "direction", "skip_mode", "seed",
                     "first_stripe", "first_sub_index", "first_attempt",
                     "first_match_rate", "dup_stripe", "dup_sub_index",
                     "dup_attempt", "dup_match_rate"):
            assert getattr(e, attr) is not None, f"{attr} must be populated"

        # a duplicate is NEVER resolved by max match_rate at this boundary
        assert not isinstance(e, AssemblyConsistencyError)


# ---------------------------------------------------------------------------
# G15 — post-D1.0 empty-pass legitimacy
# ---------------------------------------------------------------------------
def g15_empty_pass_legitimacy():
    # (a) a REAL test_both_modes=False run: exactly phases {1,2} per the
    #     post-D1.0 producer authority; constant populated, variable empty.
    stages = workflow_stages_for(PRNG_BASE, False)
    assert [ph for _f, ph in stages] == [1, 2], stages
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        coord, manifests = _build_run(tmp, sink, "g15a",
                                      phases=tuple(ph for _f, ph in stages))
        assert {int(m["workflow_phase"]) for m in manifests} == {1, 2}
        ev = coord.commit_trial("g15a", now=200.0)
        assert ev["delivery"] == "done", ev
        a = sink.get_assembly("g15a")
        exp_c = _expected_constant()
        _assert_mode(a.canonical_records_constant, a.forward_map_constant,
                     a.reverse_map_constant, a.bidirectional_constant,
                     exp_c, "constant", "java_lcg")
        assert a.forward_map_variable == {} and a.reverse_map_variable == {}
        assert a.bidirectional_variable == set()
        assert a.canonical_records_variable == []
        assert a.directional_counts["forward_variable"] == 0
        assert a.directional_counts["reverse_variable"] == 0
        assert a.directional_counts["bidirectional_variable"] == 0

    # (b) a pass whose shards ALL carry empty survivor lists -> an empty map with
    #     no error (the max(..., 1) guards absorb the zero denominators; with an
    #     empty intersection there are no records to carry the derived fields).
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        pops = {1: PHASE_POP[1], 2: {}}          # reverse-constant emits nothing
        coord, manifests = _build_run(tmp, sink, "g15b", phases=(1, 2), pops=pops)
        assert len(manifests) == 2 * 2 * 2, len(manifests)
        ev = coord.commit_trial("g15b", now=200.0)
        assert ev["delivery"] == "done", ev
        a = sink.get_assembly("g15b")
        assert a.forward_map_constant == PHASE_POP[1]
        assert a.reverse_map_constant == {}, "an empty pass yields an empty map"
        assert a.bidirectional_constant == set()
        assert a.canonical_records_constant == []
        assert a.directional_counts == {
            "forward_constant": 4, "reverse_constant": 0,
            "forward_variable": 0, "reverse_variable": 0,
            "bidirectional_constant": 0, "bidirectional_variable": 0}, \
            a.directional_counts

    # (c) the fixture's variable mode already contains legitimately EMPTY
    #     survivor lists (phase 3 has none in [30,40), phase 4 none in [20,30)) —
    #     assert those shards really are empty and still assemble.
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        coord, manifests = _build_run(tmp, sink, "g15c")
        # phase 2 has no survivor in [30,40), phase 3 none in [30,40), phase 4
        # none in [20,30) -> exactly three legitimately empty shards.
        empties = 0
        for m in manifests:
            with open(m["local_spool_path"]) as f:
                if not json.load(f)["survivors"]:
                    empties += 1
        assert empties == 3, f"the fixture must contain empty-survivor shards: {empties}"
        assert coord.commit_trial("g15c", now=200.0)["delivery"] == "done"


# ---------------------------------------------------------------------------
# G16 — ownership + concurrency [TB-D1-B3, TB-D1-GC1, TB-D1-G16C]
# ---------------------------------------------------------------------------
def g16a_caller_mutation_does_not_reach_sink():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        coord, manifests = _build_run(tmp, sink, "g16a")
        # The coordinator retains and returns the SAME mutable manifest dicts
        # (coordinator:1956-1958) — mutate them (and the nested trial_metadata)
        # after publication.
        for m in coord.enqueued:
            m["expected_size"] = -1
            m["local_spool_path"] = "/nonexistent/hijacked.json"
            m["trial_metadata"]["window_size"] = 99999
            m["trial_metadata"]["direction"] = "sideways"
            m["trial_metadata"]["sessions"].append("HIJACK")
        assert coord.enqueued and coord.enqueued[0]["expected_size"] == -1

        ev = coord.commit_trial("g16a", now=200.0)
        assert ev["delivery"] == "done", \
            f"caller-side mutation must not affect the sink: {ev}"
        a = sink.get_assembly("g16a")
        _assert_mode(a.canonical_records_constant, a.forward_map_constant,
                     a.reverse_map_constant, a.bidirectional_constant,
                     _expected_constant(), "constant", "java_lcg")
        for rec in a.canonical_records_constant:
            assert rec["window_size"] == CTX["window_size"]
            assert rec["sessions"] == CTX["sessions"], "nested list must be copied"
        # the sink's stored input is untouched
        for stored in _retained(sink, "g16a"):
            assert stored["expected_size"] > 0
            assert stored["trial_metadata"]["window_size"] == CTX["window_size"]
            assert stored["trial_metadata"]["sessions"] == CTX["sessions"]


def g16b_concurrent_direct_commit_abort():
    """DIRECT SINK INVARIANT-BREAK PROBE [TB-D1-GC1]: the post-D1.0 coordinator
    never produces this interleaving (W5 proves that); this probes the sink's own
    lock ownership and the [TB-D1-G16C] abort-last-wins final state."""
    # --- (i) commit takes the lock FIRST: it installs a result, then abort
    #         removes it and tombstones. Abort must not return while a spool read
    #         is active.
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingSink()
        _coord_obj, manifests = _build_run(tmp, sink, "g16b")
        read_active = threading.Event()
        release_read = threading.Event()
        state = {"release_ok": None}

        def hook(_path, idx):
            if idx == 0:                       # hold the FIRST spool read open
                sink._log("read_start")
                read_active.set()
                state["release_ok"] = release_read.wait(timeout=_EV_TIMEOUT)
                sink._log("read_end")

        holder = {}
        with _count_opens(hook) as reads:
            def do_commit():
                try:
                    sink.commit_trial({"event_type": "trial_commit",
                                       "run_id": "g16b", "event_id": "g16b:commit"})
                    holder["commit"] = "done"
                except Exception:  # noqa: BLE001
                    holder["commit_err"] = traceback.format_exc()

            def do_abort():
                try:
                    sink.abort_trial({"event_type": "trial_abort",
                                      "run_id": "g16b", "event_id": "g16b:abort"})
                    holder["abort"] = "done"
                except Exception:  # noqa: BLE001
                    holder["abort_err"] = traceback.format_exc()

            tc = threading.Thread(target=do_commit, daemon=True)
            tc.start()
            assert read_active.wait(timeout=_EV_TIMEOUT), "no spool read started"
            ta = threading.Thread(target=do_abort, daemon=True)
            ta.start()
            # bounded negative check: abort MUST still be blocked on the lock
            ta.join(timeout=1.0)
            assert ta.is_alive(), \
                "abort_trial returned while a spool read was still active"
            release_read.set()
            tc.join(timeout=_JOIN_TIMEOUT)
            ta.join(timeout=_JOIN_TIMEOUT)
            assert not tc.is_alive() and not ta.is_alive(), "threads did not finish"
        assert state["release_ok"] is True
        assert "commit_err" not in holder, holder.get("commit_err")
        assert "abort_err" not in holder, holder.get("abort_err")
        assert holder["commit"] == "done" and holder["abort"] == "done"
        assert reads.count == len(manifests), reads.count

        # ORDERING (deterministic, from the instrumented reader + the sink's own
        # enter/return log): abort returned only AFTER the read completed.
        ev = sink.events
        assert ev.index("abort_return") > ev.index("read_end"), ev
        assert ev.index("abort_enter") < ev.index("read_end"), ev
        assert ev.index("commit_return") < ev.index("abort_return"), ev

        # FINAL STATE [TB-D1-G16C] — abort-last wins, no torn mixed state.
        assert "g16b" in sink._tombstoned, "the run must be tombstoned"
        assert sink.get_assembly("g16b") is None
        assert _retained(sink, "g16b") == [], "no manifests may remain"
        assert "g16b" not in sink._runs, "no temporary run state may remain"
        # NO staged-path reference anywhere in the sink's own state (introspect
        # the whole instance, not just the run entry).
        blob = repr(sink.__dict__)
        for m in manifests:
            assert m["local_spool_path"] not in blob, \
                "no staged-path reference may remain in the sink"

    # --- (ii) abort takes the lock FIRST: it tombstones and the commit raises
    #          AssemblyStateError. Same final state.
    with tempfile.TemporaryDirectory() as tmp:
        sink2 = _CountingSink()
        _c2, manifests2 = _build_run(tmp, sink2, "g16c")
        sink2.abort_trial({"event_type": "trial_abort", "run_id": "g16c",
                           "event_id": "g16c:abort"})
        with _count_opens() as reads2:
            _raises(AssemblyStateError, sink2.commit_trial,
                    {"event_type": "trial_commit", "run_id": "g16c",
                     "event_id": "g16c:commit"})
        assert reads2.count == 0, "a commit onto a tombstone must read no spool"
        assert "g16c" in sink2._tombstoned
        assert sink2.get_assembly("g16c") is None
        assert _retained(sink2, "g16c") == []
        assert "g16c" not in sink2._runs
        assert manifests2, "sanity: the fixture published manifests"


# ---------------------------------------------------------------------------
# Non-regression (blocking): Phase 4 63/63, Phase 3 17/17, D0, D1.0 W1-W6
# ---------------------------------------------------------------------------
def _run_suite(rel_path, expect_substr, timeout=1800):
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, os.path.join(_ROOT, rel_path)],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-25:])
    assert proc.returncode == 0, \
        f"{rel_path} exited {proc.returncode}\n{tail}\n{(proc.stderr or '')[-2000:]}"
    assert expect_substr in (proc.stdout or ""), \
        f"{rel_path}: expected {expect_substr!r} in output\n{tail}"


def nonregression_suites():
    _run_suite("tests/test_s172_phase4_coordinator.py", "63/63 checks green")
    _run_suite("tests/test_s172_phase3_worker.py", "17/17 gates green")
    _run_suite("tests/test_s172_phase5_d0.py", "12/12 D0 gate checks green")
    _run_suite("tests/test_s172_phase5_d1_workflow.py", "8/8 D1.0 gate checks green")


def main():
    print("=" * 74)
    print("S172 Phase-5 D1.1 — assembly engine + AssemblingPhase5Sink gate")
    print("=" * 74)
    _check("G1:  hand-computed assembly, both modes + mis-grouping tripwire",
           g1_hand_computed_assembly)
    _check("G2:  get_assembly is None before commit", g2_none_before_commit)
    _check("G3:  real commit installs exactly one complete assembly",
           g3_commit_installs_one_assembly)
    _check("G4:  duplicate commit — A coordinator no-op, B direct sink replay",
           g4_duplicate_commit)
    _check("G5:  failed assembly is retryable (real coordinator sequence)",
           g5_failed_assembly_retryable)
    _check("G6:  canonically identical manifest replay is a no-op",
           g6_identical_replay_is_noop)
    _check("G7:  replay / slot / post-commit conflicts (direct probes)",
           g7_replay_and_slot_conflicts)
    _check("G8:  abort + tombstone (coordinator abort, commit, direct probe)",
           g8_abort_tombstone)
    _check("G9:  frozen 24-field canonical record, order + ascending seeds",
           g9_frozen_record_shape)
    _check("G10: threshold_used validated per direction, absent from records",
           g10_threshold_used)
    _check("G11: identity matrix — every corruption raises PhaseIdentityError",
           g11_identity_matrix)
    _check("G12: spool identity + semantics + containers", g12_spool_identity_and_semantics)
    _check("G13: cross-manifest consistency + phase-set completeness",
           g13_cross_manifest_consistency)
    _check("G14: directional duplicate — structured attributes", g14_directional_duplicate)
    _check("G15: empty-pass legitimacy (post-D1.0)", g15_empty_pass_legitimacy)
    _check("G16a: caller-owned manifest mutation cannot reach the sink",
           g16a_caller_mutation_does_not_reach_sink)
    _check("G16b: concurrent direct commit/abort — abort-last wins",
           g16b_concurrent_direct_commit_abort)
    _check("NR:  Phase 4 63/63, Phase 3 17/17, D0, D1.0 W1-W6",
           nonregression_suites)

    print("=" * 74)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} D1.1 gate checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All D1.1 gate checks green — assembly engine + Phase-5 sink are "
          "contract-validated (pending Team Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
