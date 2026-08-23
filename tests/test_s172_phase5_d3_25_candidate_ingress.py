#!/usr/bin/env python3
"""
test_s172_phase5_d3_25_candidate_ingress.py — S172 Phase-5 Deliverable D3.25
acceptance harness (mode-preserving backend result contract + canonical
candidate-ingress normalization).

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_25.md (REV3), rebased onto
HEAD c207e3a.

Thirteen gates G1-G13, each constructed to FAIL on the wrong behavior
(REV3 §1.2). The pre-fix adapter was demonstrated RED on G2/G3/G5/G6 before any
edit was made.

G13 was added in the Team Beta correction round: the original twelve never
exercised `_build_test_result_from_miner`, the helper D3.25 detached the
range-miner call site into. It is a TEST-ONLY addition — no production change.

INDEPENDENT ORACLE — the G9 / E8 lesson, binding (REV3 §5). This harness does
NOT import `CANONICAL_RECORD_FIELDS`, `TRIAL_POPULATIONS_SCHEMA_VERSION`,
`TRIAL_POPULATION_MAP_FIELDS` or `TRIAL_POPULATION_SET_FIELDS` from the module
under test and assert against them. The 24 field names, their frozen ORDER, the
v2 schema string, the v2 field names and every derived formula are
HAND-TRANSCRIBED below as literals. Relocating `CANONICAL_RECORD_FIELDS` out of
`miner/range_miner_npz_writer.py` and into `utils/canonical_records.py` in this
deliverable ([C3]) does NOT authorize importing it into an oracle — asserting a
constant against itself is the exact defect corrected in D1.1's G9 and again in
D3.0's E8.

G9 carries the weight deliberately: D1 and PWC/ZMQ now call the SAME extracted
helper, so their direct equality is a REGRESSION check, not independent proof.

G1 drives the REAL `run_trial_persistent` and `run_trial_zmq_sqlite` return
paths end-to-end against a fake sieve backend (no GPU, no rig, no socket), so
the producer-egress assertions being exercised are the live ones — not a
re-implementation in this file.

G11 is the mutation proof. Each mutant is a TEXTUAL edit applied to live source
and exec'd into a fresh module namespace; the affected gates are then re-run
against the mutated module and must go RED. No production file on disk is ever
modified.

Against the corrected tree the expected result is 13/13 green.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import traceback
import types
from typing import List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import utils.canonical_records as PROD                       # noqa: E402
import utils.canonical_arrays as D3                          # noqa: E402
import persistent_worker_coordinator as PWC                  # noqa: E402
import zmq_sqlite_coordinator as ZMQ                         # noqa: E402
import window_optimizer_integration_final as WOIF            # noqa: E402
from window_optimizer import WindowConfig                     # noqa: E402

_RECORDS_PATH = os.path.join(_ROOT, "utils", "canonical_records.py")
_WOIF_PATH = os.path.join(_ROOT, "window_optimizer_integration_final.py")


# ═════════════════════════════════════════════════════════════════════════════
# HAND-TRANSCRIBED ORACLE  (never imported from the code under test)
# ═════════════════════════════════════════════════════════════════════════════

# The frozen 24-field canonical record, in the frozen order.
ORACLE_RECORD_FIELDS: Tuple[str, ...] = (
    "seed", "forward_match_rate", "reverse_match_rate", "score",
    "window_size", "offset", "skip_min", "skip_max", "skip_range", "sessions",
    "trial_number", "prng_base", "skip_mode", "prng_type",
    "forward_count", "reverse_count", "bidirectional_count",
    "intersection_count", "intersection_ratio",
    "forward_only_count", "reverse_only_count",
    "survivor_overlap_ratio", "bidirectional_selectivity", "intersection_weight",
)

# The versioned producer contract.
ORACLE_SCHEMA_VERSION = "step1_trial_populations_v2"
ORACLE_MAP_FIELDS = (
    "forward_map_constant", "reverse_map_constant",
    "forward_map_variable", "reverse_map_variable",
)
ORACLE_SET_FIELDS = ("bidirectional_constant", "bidirectional_variable")
ORACLE_V2_FIELDS = (
    ("schema_version",) + ORACLE_MAP_FIELDS + ORACLE_SET_FIELDS
    + ("pruned", "reason")
)


def oracle_mode_record(seed, fwd, rev, *, window_size, offset, skip_min,
                       skip_max, sessions, trial_number, prng_base, skip_mode):
    """The complete 24-field record for one seed of one mode, computed here from
    first principles — every formula written out, nothing imported.

    `fwd` / `rev` are that MODE's maps only. This is the whole point of D3.25:
    a variable record is a function of the variable maps alone.
    """
    fwd_set, rev_set = set(fwd), set(rev)
    both = fwd_set & rev_set
    union = fwd_set | rev_set
    prng_type = prng_base if skip_mode == "constant" else prng_base + "_hybrid"
    return {
        "seed":                      seed,
        "forward_match_rate":        fwd[seed],
        "reverse_match_rate":        rev[seed],
        "score":                     (fwd[seed] + rev[seed]) / 2.0,
        "window_size":               window_size,
        "offset":                    offset,
        "skip_min":                  skip_min,
        "skip_max":                  skip_max,
        "skip_range":                skip_max - skip_min,
        "sessions":                  list(sessions),
        "trial_number":              trial_number,
        "prng_base":                 prng_base,
        "skip_mode":                 skip_mode,
        "prng_type":                 prng_type,
        "forward_count":             len(fwd),
        "reverse_count":             len(rev),
        "bidirectional_count":       len(both),
        "intersection_count":        len(both),
        "intersection_ratio":        len(both) / max(len(union), 1),
        "forward_only_count":        len(fwd_set - rev_set),
        "reverse_only_count":        len(rev_set - fwd_set),
        "survivor_overlap_ratio":    len(both) / max(len(fwd_set), 1),
        "bidirectional_selectivity": len(fwd_set) / max(len(rev_set), 1),
        "intersection_weight":       len(both) / max(len(fwd_set) + len(rev_set), 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Shared fixture
#
# Seed 42 lives in BOTH modes with DELIBERATELY DIFFERENT rates, and the two
# modes have DELIBERATELY DIFFERENT population sizes, so a constant-mode value
# leaking onto a variable record is arithmetically visible rather than
# coincidentally equal.
#
#   constant: fwd {42:.90, 7:.60, 9:.30, 11:.20}   rev {42:.70, 7:.50}
#   variable: fwd {42:.55}                          rev {42:.95, 13:.40, 17:.10}
#
#   constant intersection {7, 42};  seed 42 score (0.90+0.70)/2 = 0.80
#   variable intersection {42};     seed 42 score (0.55+0.95)/2 = 0.75
# ═════════════════════════════════════════════════════════════════════════════
FMC = {42: 0.90, 7: 0.60, 9: 0.30, 11: 0.20}
RMC = {42: 0.70, 7: 0.50}
FMV = {42: 0.55}
RMV = {42: 0.95, 13: 0.40, 17: 0.10}

WINDOW_SIZE, OFFSET, SKIP_MIN, SKIP_MAX = 30, 2, 5, 56
TRIAL_NUMBER, PRNG_BASE = 3, "java_lcg"
SESSIONS = ["midday", "evening"]


def make_config(sessions=None, **over):
    sess = SESSIONS if sessions is None else sessions
    if isinstance(sess, (list, tuple)):
        sess = list(sess)
    kw = dict(window_size=WINDOW_SIZE, offset=OFFSET, sessions=sess,
              skip_min=SKIP_MIN, skip_max=SKIP_MAX)
    kw.update(over)
    return WindowConfig(**kw)


def v2_result(fmc=None, rmc=None, fmv=None, rmv=None, *, pruned=False,
              reason=None, extra=None, **over):
    """A hand-built, contract-shaped v2 result (no producer involved)."""
    fmc = FMC if fmc is None else fmc
    rmc = RMC if rmc is None else rmc
    fmv = FMV if fmv is None else fmv
    rmv = RMV if rmv is None else rmv
    out = {
        "schema_version":         ORACLE_SCHEMA_VERSION,
        "forward_map_constant":   dict(fmc),
        "reverse_map_constant":   dict(rmc),
        "forward_map_variable":   dict(fmv),
        "reverse_map_variable":   dict(rmv),
        "bidirectional_constant": set(fmc) & set(rmc),
        "bidirectional_variable": set(fmv) & set(rmv),
        "pruned":                 pruned,
        "reason":                 reason,
    }
    out.update(extra or {})
    out.update(over)
    return out


def new_accumulator():
    return {"bidirectional": [], "forward_count": 0, "reverse_count": 0}


@contextlib.contextmanager
def _isolated_checkpoint_root():
    """[S172 Phase-5 D6.1] Contain the live flush inside a throwaway root.

    D6.1 turned `_flush_npz_incremental` from an always-failing attempt into a
    real provisional snapshot write, and it now resolves its directory from a
    STABLE root (`PRNG_CHECKPOINT_ROOT`, else the production module's own
    directory) rather than the CWD. `ingest()` drives the live adapter, so
    without this the suite would deposit `.s172_checkpoint/<run_id>/` in the
    repository — or, worse, in a shared production checkpoint root — on every
    run, and never clean it up.

    Per invocation: fresh temp root, restored env, directory removed.
    """
    import tempfile as _tempfile
    _prev_root = os.environ.get("PRNG_CHECKPOINT_ROOT")
    _prev_run = os.environ.get("PRNG_CHECKPOINT_RUN_ID")
    with _tempfile.TemporaryDirectory(prefix="d3_25_ckpt_") as _tmp:
        os.environ["PRNG_CHECKPOINT_ROOT"] = _tmp
        os.environ["PRNG_CHECKPOINT_RUN_ID"] = "d3-25-ingest"
        try:
            yield _tmp
        finally:
            for _k, _v in (("PRNG_CHECKPOINT_ROOT", _prev_root),
                           ("PRNG_CHECKPOINT_RUN_ID", _prev_run)):
                if _v is None:
                    os.environ.pop(_k, None)
                else:
                    os.environ[_k] = _v


def ingest(result, config=None, accumulator=None, module=None,
           trial_number=TRIAL_NUMBER, prng_base=PRNG_BASE):
    """Drive the live adapter, silencing its progress prints.

    The live adapter calls the live flush (D3.25 asserts exactly that cadence),
    so the snapshot root is isolated per invocation — see
    `_isolated_checkpoint_root`.
    """
    mod = module or WOIF
    acc = new_accumulator() if accumulator is None else accumulator
    cfg = make_config() if config is None else config
    with _isolated_checkpoint_root():
        with contextlib.redirect_stdout(io.StringIO()):
            tr = mod._build_test_result_from_pw(result, acc, cfg, prng_base,
                                                trial_number)
    return acc, tr


def normalize(module=None, **over):
    mod = module or PROD
    kw = dict(window_size=WINDOW_SIZE, offset=OFFSET, skip_min=SKIP_MIN,
              skip_max=SKIP_MAX, sessions=list(SESSIONS),
              trial_number=TRIAL_NUMBER, prng_base=PRNG_BASE)
    maps = [over.pop("fmc", FMC), over.pop("rmc", RMC),
            over.pop("fmv", FMV), over.pop("rmv", RMV)]
    kw.update(over)
    return mod.normalize_trial_populations(*maps, **kw)


def assert_raises(exc_types, fn, what):
    try:
        fn()
    except exc_types as e:
        return e
    except Exception as e:                                        # noqa: BLE001
        raise AssertionError(
            f"{what}: raised {type(e).__name__} ({e}), expected one of "
            f"{exc_types}") from e
    raise AssertionError(f"{what}: did NOT fail closed — it returned normally")


def close(a, b, what=""):
    assert abs(a - b) < 1e-12, f"{what}: {a!r} != {b!r}"


# ═════════════════════════════════════════════════════════════════════════════
# Fake sieve backends — G1 drives the REAL producer return paths
# ═════════════════════════════════════════════════════════════════════════════
class _FakeSieve:
    """Answers `run_sieve_pass` from a prng_type -> (survivors, rates) table.

    A prng_type absent from the table returns zero survivors, which is how the
    constant-only and forward-zero fixtures are expressed.
    """

    def __init__(self, table):
        self.table = table
        self._progress_writer = None
        self.logger = types.SimpleNamespace(
            warning=lambda *a, **k: None, info=lambda *a, **k: None)
        self.calls = []

    def run_sieve_pass(self, prng_type=None, **kw):
        self.calls.append(prng_type)
        survivors, rates = self.table.get(prng_type, ([], []))
        return {"survivors": list(survivors), "match_rates": list(rates)}

    def startup(self):
        pass

    def shutdown(self):
        pass


def _table(fmc=None, rmc=None, fmv=None, rmv=None):
    fmc = FMC if fmc is None else fmc
    rmc = RMC if rmc is None else rmc
    fmv = FMV if fmv is None else fmv
    rmv = RMV if rmv is None else rmv
    out = {}
    for name, m in (("java_lcg", fmc), ("java_lcg_reverse", rmc),
                    ("java_lcg_hybrid", fmv), ("java_lcg_hybrid_reverse", rmv)):
        if m:
            out[name] = (list(m), [m[s] for s in m])
    return out


def drive_pwc(table, *, test_both_modes=True):
    """Run the LIVE `run_trial_persistent` against a fake coordinator."""
    fake = _FakeSieve(table)
    real = PWC.PersistentWorkerCoordinator
    PWC.PersistentWorkerCoordinator = lambda **kw: fake
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return PWC.run_trial_persistent(
                coordinator_cfg="unused.json", config=make_config(),
                trial_number=TRIAL_NUMBER, prng_base=PRNG_BASE, residues=[1, 2],
                total_seeds=1000, forward_threshold=0.4, reverse_threshold=0.45,
                test_both_modes=test_both_modes, dataset_path="unused.csv")
    finally:
        PWC.PersistentWorkerCoordinator = real


def drive_zmq(table, *, test_both_modes=True):
    """Run the LIVE `run_trial_zmq_sqlite` against a fake session coordinator."""
    fake = _FakeSieve(table)
    with contextlib.redirect_stdout(io.StringIO()):
        return ZMQ.run_trial_zmq_sqlite(
            coordinator_cfg="unused.json", config=make_config(),
            trial_number=TRIAL_NUMBER, prng_base=PRNG_BASE, residues=[1, 2],
            total_seeds=1000, forward_threshold=0.4, reverse_threshold=0.45,
            test_both_modes=test_both_modes, dataset_path="unused.csv",
            session_coord=fake)


# ═════════════════════════════════════════════════════════════════════════════
# G1 — producer contract shape, at BOTH boundaries
# ═════════════════════════════════════════════════════════════════════════════
def g1_producer_contract_shape():
    scenarios = {
        "both-mode":     _table(),
        "constant-only": _table(fmv={}, rmv={}),
        # forward-constant zero -> each backend's PRUNED early return
        "pruned":        _table(fmc={}, rmc={}, fmv={}, rmv={}),
    }
    for backend, drive in (("PWC", drive_pwc), ("ZMQ", drive_zmq)):
        for name, table in scenarios.items():
            both_modes = name != "constant-only"
            res = drive(table, test_both_modes=both_modes)

            assert res["schema_version"] == ORACLE_SCHEMA_VERSION, \
                f"{backend}/{name}: schema_version {res.get('schema_version')!r}"
            for field in ORACLE_V2_FIELDS:
                assert field in res, \
                    f"{backend}/{name}: v2 field {field!r} missing from the return"
            for field in ORACLE_MAP_FIELDS:
                assert isinstance(res[field], dict), \
                    f"{backend}/{name}: {field!r} is {type(res[field]).__name__}"
            for field in ORACLE_SET_FIELDS:
                assert isinstance(res[field], set), \
                    f"{backend}/{name}: {field!r} is {type(res[field]).__name__}"

            # Shape NEVER varies: all four maps present even when a mode did
            # not run, and even on the pruned early return.
            if name == "constant-only":
                assert res["forward_map_variable"] == {}, f"{backend}/{name}"
                assert res["reverse_map_variable"] == {}, f"{backend}/{name}"
                assert res["forward_map_constant"] == FMC, f"{backend}/{name}"
                assert res["pruned"] is False and res["reason"] is None
            elif name == "pruned":
                assert res["pruned"] is True, f"{backend}/{name}: not pruned"
                assert res["reason"] == "forward_zero", \
                    f"{backend}/{name}: reason {res['reason']!r}"
                for field in ORACLE_MAP_FIELDS:
                    assert res[field] == {}, f"{backend}/{name}: {field} nonempty"
            else:
                assert res["forward_map_variable"] == FMV, f"{backend}/{name}"
                assert res["reverse_map_variable"] == RMV, f"{backend}/{name}"
                assert res["bidirectional_variable"] == {42}, f"{backend}/{name}"

            # Ingress accepts every one of them.
            ingest(res)

    # ---- missing field fails closed at BOTH boundaries --------------------
    for field in ORACLE_V2_FIELDS:
        broken = v2_result()
        del broken[field]
        assert_raises(ValueError,
                      lambda b=broken: PROD.validate_trial_populations(
                          b, origin="producer-egress"),
                      f"producer egress with {field!r} removed")
        assert_raises(ValueError, lambda b=broken: ingest(b),
                      f"adapter ingress with {field!r} removed")

    # A missing map is NOT an empty map: the same result with the field
    # explicitly present-and-empty is accepted, so the rejection above is about
    # PRESENCE, not about emptiness.
    ok = v2_result(fmv={}, rmv={})
    PROD.validate_trial_populations(ok, origin="producer-egress")
    ingest(ok)

    # Wrong version string is rejected even when every other field is perfect.
    assert_raises(ValueError,
                  lambda: ingest(v2_result(schema_version="step1_trial_populations_v1")),
                  "ingress with a v1 schema stamp")


# ═════════════════════════════════════════════════════════════════════════════
# G2 — same seed, both modes, different rates
# ═════════════════════════════════════════════════════════════════════════════
def g2_same_seed_both_modes():
    acc, _ = ingest(v2_result())
    recs = acc["bidirectional"]

    s42 = [r for r in recs if r["seed"] == 42]
    assert len(s42) == 2, (
        f"expected TWO records for the cross-mode seed 42, got {len(s42)}: "
        f"{[r['skip_mode'] for r in s42]}. The pre-fix adapter unioned the two "
        f"populations and emitted ONE record labelled variable.")

    c42 = [r for r in s42 if r["skip_mode"] == "constant"]
    v42 = [r for r in s42 if r["skip_mode"] == "variable"]
    assert len(c42) == 1 and len(v42) == 1, \
        f"expected one record per mode, got {[r['skip_mode'] for r in s42]}"
    c42, v42 = c42[0], v42[0]

    close(c42["forward_match_rate"], 0.90, "constant fwd rate")
    close(c42["reverse_match_rate"], 0.70, "constant rev rate")
    close(c42["score"], 0.80, "constant score")
    close(v42["forward_match_rate"], 0.55, "variable fwd rate")
    close(v42["reverse_match_rate"], 0.95, "variable rev rate")
    close(v42["score"], 0.75, "variable score")

    assert c42["prng_type"] == "java_lcg", c42["prng_type"]
    assert v42["prng_type"] == "java_lcg_hybrid", v42["prng_type"]
    assert c42["prng_base"] == v42["prng_base"] == "java_lcg"


# ═════════════════════════════════════════════════════════════════════════════
# G3 — mode-specific aggregates
# ═════════════════════════════════════════════════════════════════════════════
def g3_mode_specific_aggregates():
    acc, tr = ingest(v2_result())
    recs = acc["bidirectional"]

    # Independent hand calculations, per mode.
    #   constant: fwd {42,7,9,11}=4, rev {42,7}=2, both {7,42}=2,
    #             union {42,7,9,11}=4
    #     intersection_ratio 2/4 = 0.5 ; forward_only {9,11}=2 ; reverse_only 0
    #     survivor_overlap 2/4 = 0.5 ; selectivity 4/2 = 2.0 ; weight 2/6
    #   variable: fwd {42}=1, rev {42,13,17}=3, both {42}=1, union {42,13,17}=3
    #     intersection_ratio 1/3 ; forward_only 0 ; reverse_only 2
    #     survivor_overlap 1/1 = 1.0 ; selectivity 1/3 ; weight 1/4 = 0.25
    expect = {
        "constant": dict(forward_count=4, reverse_count=2, bidirectional_count=2,
                         intersection_count=2, intersection_ratio=0.5,
                         forward_only_count=2, reverse_only_count=0,
                         survivor_overlap_ratio=0.5,
                         bidirectional_selectivity=2.0,
                         intersection_weight=2 / 6),
        "variable": dict(forward_count=1, reverse_count=3, bidirectional_count=1,
                         intersection_count=1, intersection_ratio=1 / 3,
                         forward_only_count=0, reverse_only_count=2,
                         survivor_overlap_ratio=1.0,
                         bidirectional_selectivity=1 / 3,
                         intersection_weight=0.25),
    }
    seen = set()
    for r in recs:
        mode = r["skip_mode"]
        seen.add(mode)
        for field, want in expect[mode].items():
            got = r[field]
            if isinstance(want, float):
                close(got, want, f"{mode} record seed {r['seed']} {field}")
            else:
                assert got == want, \
                    f"{mode} record seed {r['seed']} {field}: {got} != {want}"
        # Cross-check: no constant-mode aggregate may appear on a variable
        # record. Every field above differs between the two modes, so an
        # accidental match is arithmetically impossible.
        other = "variable" if mode == "constant" else "constant"
        for field, wrong in expect[other].items():
            if isinstance(wrong, float) and abs(expect[mode][field] - wrong) < 1e-12:
                continue
            assert r[field] != wrong, \
                f"{mode} record carries the {other}-mode value for {field}"
    assert seen == {"constant", "variable"}, f"modes emitted: {sorted(seen)}"

    # No combined constant+variable count belongs in a record — but TestResult
    # still exposes it as run telemetry (2 constant + 1 variable = 3).
    assert tr.bidirectional_count == 3, tr.bidirectional_count
    assert all(r["bidirectional_count"] != 3 for r in recs), \
        "a record carried the COMBINED bidirectional total"


# ═════════════════════════════════════════════════════════════════════════════
# G4 — intersection consistency, checked BEFORE accumulator mutation
# ═════════════════════════════════════════════════════════════════════════════
def g4_intersection_consistency():
    for field, corrupt in (
        ("bidirectional_constant", {42}),          # drops 7 from the set
        ("bidirectional_constant", {42, 7, 999}),  # adds a non-member
        ("bidirectional_variable", set()),         # drops 42
        ("bidirectional_variable", {42, 13}),      # 13 is reverse-only
    ):
        bad = v2_result(**{field: corrupt})

        # producer egress
        assert_raises(ValueError,
                      lambda b=bad: PROD.validate_trial_populations(
                          b, origin="producer-egress"),
                      f"egress with {field} = {corrupt}")

        # adapter ingress — and NOTHING is appended
        acc = new_accumulator()
        acc["bidirectional"].append({"pre-existing": True})
        before = len(acc["bidirectional"])
        assert_raises(ValueError, lambda b=bad: ingest(b, accumulator=acc),
                      f"ingress with {field} = {corrupt}")
        after = len(acc["bidirectional"])
        assert before == after, (
            f"accumulator mutated before the consistency wall fired: "
            f"{before} -> {after} (field {field})")
        assert acc["forward_count"] == 0 and acc["reverse_count"] == 0, \
            "directional counters advanced before the consistency wall fired"

    # PRODUCER EGRESS proper: `build_trial_populations` is the single call both
    # backends assemble their return through, so it must refuse to HAND BACK an
    # inconsistent block — not merely be validatable after the fact. Asserting
    # only against `validate_trial_populations` would leave the egress call site
    # itself deletable without any gate noticing.
    assert_raises(
        ValueError,
        lambda: PROD.build_trial_populations(
            forward_map_constant=FMC, reverse_map_constant=RMC,
            forward_map_variable=FMV, reverse_map_variable=RMV,
            bidirectional_constant={42},               # 7 dropped
            bidirectional_variable=set(FMV) & set(RMV),
            pruned=False, reason=None),
        "build_trial_populations with an inconsistent constant set")
    assert_raises(
        ValueError,
        lambda: PROD.build_trial_populations(
            forward_map_constant=FMC, reverse_map_constant=RMC,
            forward_map_variable=FMV, reverse_map_variable=RMV,
            bidirectional_constant=set(FMC) & set(RMC),
            bidirectional_variable={42, 13},           # 13 is reverse-only
            pruned=False, reason=None),
        "build_trial_populations with an inconsistent variable set")

    # The wall is not "repaired" by preferring either side: a result whose set
    # and maps agree passes, and no recomputation is substituted for the set.
    good = v2_result()
    acc, _ = ingest(good)
    assert len(acc["bidirectional"]) == 3, len(acc["bidirectional"])
    consistent = PROD.build_trial_populations(
        forward_map_constant=FMC, reverse_map_constant=RMC,
        forward_map_variable=FMV, reverse_map_variable=RMV,
        bidirectional_constant=set(FMC) & set(RMC),
        bidirectional_variable=set(FMV) & set(RMV),
        pruned=False, reason=None)
    assert consistent["bidirectional_constant"] == {7, 42}
    ingest(consistent)


# ═════════════════════════════════════════════════════════════════════════════
# G5 — skip_range
# ═════════════════════════════════════════════════════════════════════════════
def g5_skip_range():
    acc, _ = ingest(v2_result())
    for r in acc["bidirectional"]:
        assert type(r["skip_range"]) is int, (
            f"skip_range is {type(r['skip_range']).__name__} "
            f"({r['skip_range']!r}); the legacy string form is prohibited")
        assert r["skip_range"] == SKIP_MAX - SKIP_MIN == 51, r["skip_range"]
        assert r["skip_range"] != f"{SKIP_MIN}-{SKIP_MAX}"

    # A string skip bound cannot sneak the string form back in.
    assert_raises(ValueError, lambda: normalize(skip_min="5"),
                  "normalize with a string skip_min")
    assert_raises(ValueError, lambda: normalize(skip_max=None),
                  "normalize with skip_max=None")
    # bool is excluded even though True == 1
    assert_raises(ValueError, lambda: normalize(skip_min=True),
                  "normalize with a Boolean skip_min")

    # Negative-width ranges are arithmetic, not string concatenation.
    c, v = normalize(skip_min=10, skip_max=4)
    assert all(r["skip_range"] == -6 for r in c + v), \
        [r["skip_range"] for r in c + v]


# ═════════════════════════════════════════════════════════════════════════════
# G6 — sessions
# ═════════════════════════════════════════════════════════════════════════════
def g6_sessions():
    # list form is preserved
    acc, _ = ingest(v2_result())
    for r in acc["bidirectional"]:
        assert isinstance(r["sessions"], list), type(r["sessions"]).__name__
        assert r["sessions"] == SESSIONS, r["sessions"]

    # None -> []
    c, v = normalize(sessions=None)
    assert all(r["sessions"] == [] for r in c + v), [r["sessions"] for r in c + v]

    # tuple -> defensive list copy
    c, v = normalize(sessions=("midday",))
    assert all(r["sessions"] == ["midday"] for r in c + v)
    assert all(isinstance(r["sessions"], list) for r in c + v)

    # scalar "all" fails closed — NOT converted to ["all"]
    assert_raises(ValueError, lambda: normalize(sessions="all"),
                  "normalize with the scalar 'all'")
    assert_raises(ValueError, lambda: ingest(v2_result(),
                                             config=make_config(sessions="all")),
                  "ingress with a scalar 'all' sessions on the config")

    # missing attribute fails closed
    class _NoSessions:
        window_size, offset, skip_min, skip_max = 30, 2, 5, 56

    assert_raises((ValueError, AttributeError),
                  lambda: ingest(v2_result(), config=_NoSessions()),
                  "ingress with a config lacking `sessions`")

    # non-str members fail closed
    assert_raises(ValueError, lambda: normalize(sessions=["midday", 7]),
                  "normalize with a non-str session name")

    # [G6, REV2 addition] mutating the CALLER's list afterwards must not reach
    # an already-produced record. (PWC/ZMQ wrapper only — D1's accepted
    # shared-reference behavior is out of scope here.)
    caller_sessions = ["midday", "evening"]
    c, v = normalize(sessions=caller_sessions)
    caller_sessions.append("night")
    caller_sessions[0] = "MUTATED"
    for r in c + v:
        assert r["sessions"] == ["midday", "evening"], (
            f"a caller-side mutation reached a produced record: {r['sessions']}")


# ═════════════════════════════════════════════════════════════════════════════
# G7 — no generic-map authority
# ═════════════════════════════════════════════════════════════════════════════
def g7_no_generic_map_authority():
    # Deliberately MISLEADING legacy aliases: same seeds, wrong rates, wrong
    # sizes. A v2 adapter that reads them produces visibly different records.
    # Sizes are chosen to collide with NEITHER mode's counts (constant 4/2,
    # variable 1/3), so `!= len(alias)` is a real discriminator rather than a
    # coincidence.
    misleading_fwd = {42: 0.01, 7: 0.02, 9: 0.03, 11: 0.04, 555: 0.05, 556: 0.06}
    misleading_rev = {42: 0.07, 7: 0.08, 555: 0.09, 556: 0.10, 557: 0.11}
    res = v2_result(extra={
        "forward_map": misleading_fwd,
        "reverse_map": misleading_rev,
        "bidirectional_count": 999,
        "forward_records": [{"seed": 555, "match_rate": 0.05}],
        "reverse_records": [{"seed": 555, "match_rate": 0.09}],
    })
    acc, tr = ingest(res)

    for r in acc["bidirectional"]:
        assert r["seed"] not in (555, 556), \
            f"seed {r['seed']} could only have come from the legacy alias maps"
        assert r["forward_match_rate"] not in tuple(misleading_fwd.values()), \
            "a record carried a rate from the legacy `forward_map` alias"
        assert r["reverse_match_rate"] not in tuple(misleading_rev.values()), \
            "a record carried a rate from the legacy `reverse_map` alias"
        assert r["forward_count"] != len(misleading_fwd), \
            "forward_count was derived from the legacy alias map"
        assert r["reverse_count"] != len(misleading_rev), \
            "reverse_count was derived from the legacy alias map"
        assert r["bidirectional_count"] != 999

    # Same fixture WITHOUT the aliases must give byte-identical records.
    acc_clean, _ = ingest(v2_result())
    assert acc["bidirectional"] == acc_clean["bidirectional"], \
        "the legacy aliases changed the canonical output"

    # TestResult telemetry comes from the v2 CONSTANT pair, not the aliases.
    assert tr.forward_count == len(FMC) and tr.reverse_count == len(RMC), \
        (tr.forward_count, tr.reverse_count)


# ═════════════════════════════════════════════════════════════════════════════
# G8 — ordering + dual preservation
# ═════════════════════════════════════════════════════════════════════════════
def g8_ordering_and_dual_preservation():
    # Overlapping mode populations, seeds deliberately out of natural order.
    fmc = {90: 0.5, 11: 0.6, 42: 0.7, 7: 0.8}
    rmc = {42: 0.4, 7: 0.3, 90: 0.2}
    fmv = {42: 0.9, 5: 0.1, 90: 0.6}
    rmv = {90: 0.2, 42: 0.8, 5: 0.4}
    acc, _ = ingest(v2_result(fmc, rmc, fmv, rmv))
    recs = acc["bidirectional"]

    modes = [r["skip_mode"] for r in recs]
    seeds = [r["seed"] for r in recs]
    # constant intersection {7,42,90}; variable intersection {5,42,90}
    assert modes == ["constant"] * 3 + ["variable"] * 3, modes
    assert seeds == [7, 42, 90, 5, 42, 90], seeds
    assert seeds[:3] == sorted(seeds[:3]), "constant block not seed-ascending"
    assert seeds[3:] == sorted(seeds[3:]), "variable block not seed-ascending"

    # Both cross-mode candidates survive as DISTINCT records to L2.
    for seed in (42, 90):
        pair = [r for r in recs if r["seed"] == seed]
        assert len(pair) == 2, f"seed {seed}: {len(pair)} record(s)"
        assert {r["skip_mode"] for r in pair} == {"constant", "variable"}
        assert pair[0]["score"] != pair[1]["score"], \
            f"seed {seed}: the two modes collapsed to one score"

    # Trial-major across two trials: trial N constant, trial N variable, then
    # trial N+1 constant, trial N+1 variable.
    acc2 = new_accumulator()
    ingest(v2_result(fmc, rmc, fmv, rmv), accumulator=acc2, trial_number=1)
    ingest(v2_result(fmc, rmc, fmv, rmv), accumulator=acc2, trial_number=2)
    got = [(r["trial_number"], r["skip_mode"]) for r in acc2["bidirectional"]]
    assert got == ([(1, "constant")] * 3 + [(1, "variable")] * 3
                   + [(2, "constant")] * 3 + [(2, "variable")] * 3), got


# ═════════════════════════════════════════════════════════════════════════════
# G9 — canonical oracle (load-bearing)
# ═════════════════════════════════════════════════════════════════════════════
def g9_canonical_oracle():
    acc, _ = ingest(v2_result())
    recs = acc["bidirectional"]

    # Field names AND order, against the hand-transcribed oracle.
    for r in recs:
        assert tuple(r.keys()) == ORACLE_RECORD_FIELDS, (
            f"record key order drifted:\n  got:    {tuple(r.keys())}\n"
            f"  oracle: {ORACLE_RECORD_FIELDS}")
        assert len(r) == 24, len(r)

    # Every value, per mode, from the independent oracle.
    base = dict(window_size=WINDOW_SIZE, offset=OFFSET, skip_min=SKIP_MIN,
                skip_max=SKIP_MAX, sessions=SESSIONS,
                trial_number=TRIAL_NUMBER, prng_base=PRNG_BASE)
    want = []
    for seed in sorted(set(FMC) & set(RMC)):
        want.append(oracle_mode_record(seed, FMC, RMC, skip_mode="constant", **base))
    for seed in sorted(set(FMV) & set(RMV)):
        want.append(oracle_mode_record(seed, FMV, RMV, skip_mode="variable", **base))
    assert recs == want, (
        "production records differ from the hand-computed oracle:\n"
        + "\n".join(f"  got  {g}\n  want {w}" for g, w in zip(recs, want)
                    if g != w))

    # PRNG identity, skip mode, integer skip_range, canonical sessions, and
    # same-seed preservation across modes are all covered by the equality
    # above; assert the load-bearing ones explicitly so a failure names itself.
    c42 = next(r for r in recs if r["seed"] == 42 and r["skip_mode"] == "constant")
    v42 = next(r for r in recs if r["seed"] == 42 and r["skip_mode"] == "variable")
    assert (c42["prng_base"], c42["prng_type"]) == ("java_lcg", "java_lcg")
    assert (v42["prng_base"], v42["prng_type"]) == ("java_lcg", "java_lcg_hybrid")
    assert type(c42["skip_range"]) is int and c42["skip_range"] == 51
    assert c42["sessions"] == ["midday", "evening"]
    assert c42["score"] != v42["score"]

    # The production constant matches the hand-transcribed oracle — and the
    # miner writer's re-export is the SAME object after the [C3] relocation.
    import miner.range_miner_npz_writer as NPZW
    assert PROD.CANONICAL_RECORD_FIELDS == ORACLE_RECORD_FIELDS, \
        f"production constant drifted: {PROD.CANONICAL_RECORD_FIELDS}"
    assert NPZW.CANONICAL_RECORD_FIELDS == ORACLE_RECORD_FIELDS, \
        "the miner writer's re-export drifted from the relocated constant"
    assert D3.CANONICAL_RECORD_FIELDS == ORACLE_RECORD_FIELDS, \
        "D3's deliberately-independent copy drifted"

    # REGRESSION check, explicitly NOT independent proof: D1 and PWC/ZMQ now
    # call the same extracted helper, so this can only catch a rewiring
    # accident. The oracle equality above is what carries the weight.
    # [WINDOW-ANCHOR BRIEF I] `base` is the ORACLE's kwargs and must keep its
    # exact shape. The trial CONTEXT is a separate object: its key is
    # `window_anchor`, while the emitted RECORD field stays `offset` (frozen
    # canonical array 4, TB wire-name ruling). Fixture-only; no assertion changed.
    ctx = dict(base)
    ctx["window_anchor"] = ctx.pop("offset")
    ctx["generator_phase"] = 0
    _, d1_style = NPZW._mode_records(FMC, RMC, ctx, "constant", "java_lcg")
    assert d1_style == [r for r in recs if r["skip_mode"] == "constant"], \
        "D1's helper and the D3.25 wrapper diverged (rewiring regression)"


# ═════════════════════════════════════════════════════════════════════════════
# G10 — a record list is NOT a map
# ═════════════════════════════════════════════════════════════════════════════
def g10_record_list_is_not_a_map():
    # PWC builds `forward_records_hybrid` from the RAW SURVIVOR SEQUENCE, so a
    # repeated raw seed appears twice in the list while the map holds it once.
    repeated = [{"seed": 42, "match_rate": 0.55},
                {"seed": 42, "match_rate": 0.55},
                {"seed": 99, "match_rate": 0.11}]
    res = v2_result(extra={"forward_records_hybrid": repeated,
                           "reverse_records_hybrid":
                               [{"seed": s, "match_rate": r} for s, r in RMV.items()]})
    acc, _ = ingest(res)
    variable = [r for r in acc["bidirectional"] if r["skip_mode"] == "variable"]

    # Derived from the MAP: forward_count 1, and seed 99 never appears.
    assert len(variable) == 1 and variable[0]["seed"] == 42, \
        [(r["seed"]) for r in variable]
    assert variable[0]["forward_count"] == len(FMV) == 1, \
        variable[0]["forward_count"]
    assert all(r["seed"] != 99 for r in acc["bidirectional"]), \
        "seed 99 exists only in the record list — a map was reconstructed from it"

    # A reconstruction-from-records implementation WOULD differ: prove the two
    # disagree rather than asserting they happen to agree.
    reconstructed_fwd = {}
    for entry in repeated:
        reconstructed_fwd[entry["seed"]] = entry["match_rate"]
    reconstructed_fwd[99] = 0.11
    assert reconstructed_fwd != FMV, "fixture is degenerate — make the lists differ"
    recon_c, recon_v = normalize(fmv=reconstructed_fwd)
    assert recon_v != variable, (
        "reconstructing the map from `forward_records_hybrid` produced the SAME "
        "records — the gate cannot distinguish the two implementations")
    # The divergence is concrete and named: the reconstruction sees TWO forward
    # survivors (42 and the list-only 99) where the authoritative map has one,
    # which propagates into every forward-derived aggregate.
    assert recon_v[0]["forward_count"] == 2, recon_v[0]["forward_count"]
    assert variable[0]["forward_count"] == 1, variable[0]["forward_count"]
    assert recon_v[0]["bidirectional_selectivity"] != \
        variable[0]["bidirectional_selectivity"]

    # The v2 maps are the only authority: the record lists are absent entirely
    # and nothing changes.
    acc_noreclists, _ = ingest(v2_result())
    assert acc["bidirectional"] == acc_noreclists["bidirectional"]


# ═════════════════════════════════════════════════════════════════════════════
# G12 — D3 conformance (free gate; D3 already paid for it)
# ═════════════════════════════════════════════════════════════════════════════
def g12_d3_conformance():
    acc, _ = ingest(v2_result())
    recs = acc["bidirectional"]
    assert len(recs) == 3 and any(r["seed"] == 42 for r in recs), \
        "fixture must include the cross-mode seed"

    bundle = D3.records_to_arrays(recs)          # must not raise
    D3.validate_array_bundle(bundle)             # must not raise
    assert len(bundle) == 22, len(bundle)
    assert list(bundle["seeds"]) == [7, 42, 42], list(bundle["seeds"])
    # skip_mode / prng_type columns preserve the two modes distinctly.
    assert len(set(bundle["skip_mode"].tolist())) == 2, bundle["skip_mode"]

    # Negative cases: the two forms D3 rejects outright. A normalizer that
    # emitted either would be caught here even if D3.25's own gates missed it.
    tuple_sessions = [dict(r) for r in recs]
    for r in tuple_sessions:
        r["sessions"] = tuple(r["sessions"])
    assert_raises(ValueError, lambda: D3.records_to_arrays(tuple_sessions),
                  "D3 on tuple sessions")

    str_range = [dict(r) for r in recs]
    for r in str_range:
        r["skip_range"] = f"{SKIP_MIN}-{SKIP_MAX}"
    assert_raises(ValueError, lambda: D3.records_to_arrays(str_range),
                  "D3 on the legacy '5-56' skip_range string")

    scalar_sessions = [dict(r) for r in recs]
    for r in scalar_sessions:
        r["sessions"] = "all"
    assert_raises(ValueError, lambda: D3.records_to_arrays(scalar_sessions),
                  "D3 on scalar 'all' sessions")

    # And the pruned / empty case columnizes rectangularly.
    empty_bundle = D3.records_to_arrays([])
    D3.validate_array_bundle(empty_bundle)
    assert all(len(a) == 0 for a in empty_bundle.values())


# ═════════════════════════════════════════════════════════════════════════════
# G13 — miner-path isolation (Team Beta correction round; UPDATED AT D6)
#
# D3.25 detached the range-miner call site (:426) from the shared PWC/ZMQ
# adapter into `_build_test_result_from_miner`, because REV3 §4 forbids routing
# miner output through the v2 contract and D6 owns miner candidate ingress. The
# original 12 gates never touched that function — G13 closes the hole.
#
# [S172 Phase-5 D6] THE HANDOFF THIS GATE WAS WRITTEN FOR HAS HAPPENED. D3.25's
# own certification note reads "miner both-mode run-level candidate output
# uncertified until D6", and the pre-D6 assertions here — zero deltas, an
# all-zero TestResult, "the miner path consumed canonical records that only D6
# may append" — encoded that INTERIM state, not a permanent invariant. D6
# certifies that output, so those three assertions are updated. What G13 still
# guards, unchanged and now the whole point of it, is:
#
#   * ISOLATION — the miner path still never reaches the PWC/ZMQ D3.25 ingress,
#     the four-map normalizer, or `_build_test_result_from_pw` (AST + runtime);
#   * FLUSH CADENCE — still EXACTLY ONE `_flush_npz_incremental` call per
#     invocation, same label, same accumulator, and none at all for a `None`
#     accumulator;
#   * NEVER A FABRICATED ZERO — the bare `serve_trial` dict this gate has always
#     used carries no population keys, and D6 does not read it for populations.
#     With no Phase-5 sink there is no publication result, so the path now
#     RAISES instead of returning the all-zero TestResult. The accumulator
#     deltas this gate asserted are therefore still zero — by refusal rather
#     than by inertness, which is the stronger property.
#
# The fixture is a REALISTIC `serve_trial` return: exactly the seven keys
# `RangeMinerCoordinator.serve_trial` produces (range_miner_coordinator.py:3386-3393)
# and NOT ONE population key. This is what actually reaches the call site today.
#
# The flush-cadence oracle is hand-transcribed from the PRE-D3.25 shared path at
# c207e3a: given a `serve_trial` dict, that path appended nothing (the union of
# two absent sets is empty), advanced both counters by zero, and called
# `_flush_npz_incremental` EXACTLY ONCE with label f"chunk/trial-{trial_number}".
# Cadence parity is therefore 1 call with that label — asserted, not assumed,
# and D6 must not shift it.
# ═════════════════════════════════════════════════════════════════════════════

# The exact key set serve_trial returns — no population keys, by construction.
# [S172 D6 correction] `threshold_provenance` is the eighth key: serve_trial now
# returns the requested/payload/effective audit record, and `validated` is set
# only by the parent's fail-closed provenance gate — which runs immediately
# before commit_trial, so a `committed: True` serve result ALWAYS carries
# `validated: True`. This oracle is a deliberate drift detector; it flagged the
# new key, and is updated here to track the real contract.
SERVE_TRIAL_KEYS = ("run_id", "state", "committed", "workers_registered",
                    "stripes", "manifests", "bound_addr",
                    "threshold_provenance")


def serve_trial_result():
    return {
        "run_id": "run/rrig6600c-t3",
        "state": "committed",
        "committed": True,
        "workers_registered": ["rrig6600c:gpu0", "rrig6600c:gpu1"],
        "stripes": {
            "s0": {"state": "published", "phase_degraded": False,
                   "claimed_by": "rrig6600c:gpu0", "current_attempt": 0,
                   "survivors_total": 12},
            "s1": {"state": "published", "phase_degraded": False,
                   "claimed_by": "rrig6600c:gpu1", "current_attempt": 1,
                   "survivors_total": 7},
        },
        "manifests": [{"event_id": "e0", "stripe_id": "s0", "sub_index": 0},
                      {"event_id": "e1", "stripe_id": "s1", "sub_index": 0}],
        "bound_addr": ("192.168.3.177", 5700),
        # [S172 D6 correction] the three-leg threshold audit record. `validated`
        # True because this fixture models a COMMITTED trial, and the parent's
        # fail-closed gate is what allows a commit to happen at all.
        "threshold_provenance": {
            "run_id": "run/rrig6600c-t3",
            "requested": {"forward": 0.31, "reverse": 0.47},
            "payload": {1: [0.31], 2: [0.47]},
            "effective": {1: [0.31], 2: [0.47]},
            "phase_direction": {1: "forward", 2: "reverse"},
            "validated": True,
        },
    }


@contextlib.contextmanager
def _flush_spy(module):
    """Replace the module's `_flush_npz_incremental` with a recording stub."""
    calls = []
    real = module._flush_npz_incremental

    def _spy(accumulator, label=None):
        calls.append((id(accumulator), label))

    module._flush_npz_incremental = _spy
    try:
        yield calls
    finally:
        module._flush_npz_incremental = real


class _StoredAssemblySink:
    """The minimum D6 reads across the L6 boundary: `get_assembly(run_id)`.

    Deliberately NOT an `AssemblingPhase5Sink` — this gate is about what the
    ADAPTER does with a stored assembly, and a stub keeps that independent of
    the D1.1 engine (whose own 18 gates cover it)."""

    def __init__(self, run_id, assembly):
        self._run_id, self._assembly = run_id, assembly
        self.calls = []

    def get_assembly(self, run_id):
        self.calls.append(run_id)
        return self._assembly if run_id == self._run_id else None


def _stored_assembly():
    """A fully-formed assembly over this harness's four maps, with the six
    directional counts hand-computed from them."""
    from miner.range_miner_npz_writer import MinerTrialAssembly
    canon_c, canon_v = normalize(module=PROD)
    bidi_c = set(FMC) & set(RMC)
    bidi_v = set(FMV) & set(RMV)
    return MinerTrialAssembly(
        run_id="run/rrig6600c-t3",
        bidirectional_constant=bidi_c,
        bidirectional_variable=bidi_v,
        forward_map_constant=dict(FMC), reverse_map_constant=dict(RMC),
        forward_map_variable=dict(FMV), reverse_map_variable=dict(RMV),
        canonical_records_constant=canon_c,
        canonical_records_variable=canon_v,
        directional_counts={
            "forward_constant": len(FMC), "reverse_constant": len(RMC),
            "forward_variable": len(FMV), "reverse_variable": len(RMV),
            "bidirectional_constant": len(bidi_c),
            "bidirectional_variable": len(bidi_v),
        },
        timing={"assembly_s": 0.0}), canon_c, canon_v


def g13_miner_path_isolation():
    mod = WOIF                      # honours the G11 mutant swap
    from miner.step1_ingress import MinerIngressError
    result = serve_trial_result()
    assert set(result) == set(SERVE_TRIAL_KEYS), \
        f"fixture drifted from the serve_trial key set: {sorted(result)}"

    # ---- no publication result -> RAISE, never a fabricated zero ----------
    # A PRE-POPULATED accumulator: deltas are what matter, and a function that
    # wrongly appends would be invisible against an empty one.
    sentinel = [{"seed": 111, "skip_mode": "constant", "sentinel": True},
                {"seed": 222, "skip_mode": "variable", "sentinel": True}]
    acc = {"bidirectional": list(sentinel),
           "forward_count": 17, "reverse_count": 23}
    before = (len(acc["bidirectional"]), acc["forward_count"], acc["reverse_count"])

    with _flush_spy(mod) as calls0:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                tr0 = mod._build_test_result_from_miner(
                    result, acc, make_config(), PRNG_BASE, TRIAL_NUMBER)
            except MinerIngressError:
                tr0 = None
    assert tr0 is None, (
        f"a serve_trial dict with NO Phase-5 sink produced a TestResult "
        f"({tr0}) — D6 must fail closed, never fabricate a zero result")
    assert calls0 == [], "the flush fired on a fail-closed path"

    after = (len(acc["bidirectional"]), acc["forward_count"], acc["reverse_count"])
    assert after[0] - before[0] == 0, (
        f"accumulator['bidirectional'] delta {after[0] - before[0]} != 0 — the "
        f"fail-closed path touched the accumulator")
    assert after[1] - before[1] == 0, \
        f"accumulator['forward_count'] delta {after[1] - before[1]} != 0"
    assert after[2] - before[2] == 0, \
        f"accumulator['reverse_count'] delta {after[2] - before[2]} != 0"
    assert acc["bidirectional"] == sentinel, \
        "the pre-existing accumulator contents were disturbed"

    # ---- with a stored assembly: real ingress, unchanged flush cadence ----
    assembly, canon_c, canon_v = _stored_assembly()
    sink = _StoredAssemblySink(result["run_id"], assembly)

    acc = {"bidirectional": list(sentinel),
           "forward_count": 17, "reverse_count": 23}
    with _flush_spy(mod) as calls:
        with contextlib.redirect_stdout(io.StringIO()):
            tr = mod._build_test_result_from_miner(
                result, acc, make_config(), PRNG_BASE, TRIAL_NUMBER,
                phase5_sink=sink)

    # the assembly's own records, appended after the sentinel, constant first
    assert acc["bidirectional"] == sentinel + canon_c + canon_v, (
        "the appended candidates are not the stored assembly's records, in "
        "assembly order")
    assert acc["forward_count"] == 17 + len(FMC) + len(FMV), acc["forward_count"]
    assert acc["reverse_count"] == 23 + len(RMC) + len(RMV), acc["reverse_count"]
    assert tr.forward_count == len(FMC), tr.forward_count
    assert tr.reverse_count == len(RMC), tr.reverse_count
    assert tr.bidirectional_count == (len(set(FMC) & set(RMC))
                                      + len(set(FMV) & set(RMV))), tr
    assert tr.iteration == TRIAL_NUMBER, tr.iteration
    assert sink.calls == [result["run_id"]], sink.calls

    # ---- flush cadence + label parity with the pre-D3.25 shared path ------
    assert len(calls) == 1, (
        f"_flush_npz_incremental called {len(calls)} time(s); the pre-D3.25 "
        f"shared path called it EXACTLY ONCE per invocation — flush cadence "
        f"must not shift")
    flushed_id, label = calls[0]
    assert label == f"chunk/trial-{TRIAL_NUMBER}", (
        f"flush label {label!r} != {f'chunk/trial-{TRIAL_NUMBER}'!r}")
    assert flushed_id == id(acc), "the flush received a different accumulator"

    # A different trial number moves the label with it, once per invocation.
    with _flush_spy(mod) as calls2:
        with contextlib.redirect_stdout(io.StringIO()):
            mod._build_test_result_from_miner(result, acc, make_config(),
                                              PRNG_BASE, 41, phase5_sink=sink)
    assert [lbl for _i, lbl in calls2] == ["chunk/trial-41"], calls2

    # accumulator=None -> no flush at all, and no crash.
    with _flush_spy(mod) as calls3:
        with contextlib.redirect_stdout(io.StringIO()):
            tr_none = mod._build_test_result_from_miner(
                result, None, make_config(), PRNG_BASE, TRIAL_NUMBER,
                phase5_sink=sink)
    assert calls3 == [], f"flush fired on a None accumulator: {calls3}"
    assert tr_none.bidirectional_count == tr.bidirectional_count

    # ---- NO PWC/ZMQ ingress: behavioral proof ----------------------------
    # The serve dict is BAITED with population-shaped keys. D6 reads populations
    # ONLY from the stored assembly, so a path that ever fell back to reading
    # the serve result would produce different numbers here.
    baited = dict(result)
    baited["forward_map"] = {9_990: 0.1, 9_991: 0.2, 9_992: 0.3}
    baited["reverse_map"] = {9_990: 0.4}
    baited["bidirectional_constant"] = {9_990, 9_991}
    baited["bidirectional_variable"] = {9_992}
    baited["forward_records"] = [{"seed": 9_990}] * 5
    baited["reverse_records"] = [{"seed": 9_991}] * 6
    acc2 = {"bidirectional": [], "forward_count": 0, "reverse_count": 0}
    with _flush_spy(mod):
        with contextlib.redirect_stdout(io.StringIO()):
            tr2 = mod._build_test_result_from_miner(
                baited, acc2, make_config(), PRNG_BASE, TRIAL_NUMBER,
                phase5_sink=sink)
    assert acc2["bidirectional"] == canon_c + canon_v, (
        "the miner path read candidates from the serve_trial dict instead of "
        "the stored assembly")
    assert acc2["forward_count"] == len(FMC) + len(FMV), acc2["forward_count"]
    assert acc2["reverse_count"] == len(RMC) + len(RMV), acc2["reverse_count"]
    assert tr2.forward_count == len(FMC), tr2.forward_count
    assert tr2.bidirectional_count == tr.bidirectional_count, tr2

    # ---- NO PWC/ZMQ ingress: source proof --------------------------------
    # Nothing in the function body may reach the v2 normalizer, the v2 ingress
    # wall, the PWC builder, a manifest or a spool. The assembly it DOES read is
    # fetched through the D6 adapter, which is itself gated by the D6 harness.
    import inspect
    src = inspect.getsource(mod._build_test_result_from_miner)
    body = src.split('"""', 2)[-1]          # strip the docstring
    for forbidden in ("assemble_trial", "normalize_trial_populations",
                      "validate_trial_populations", "build_mode_records",
                      "_build_test_result_from_pw", "spool", "read_manifest",
                      "_read_and_validate_spool"):
        assert forbidden not in body, (
            f"_build_test_result_from_miner body references {forbidden!r} — "
            f"the miner path must not share the PWC/ZMQ ingress")

    # And the call site really is detached from the v2 adapter, and wires the
    # sink without which there would be no assembly at all.
    with open(_WOIF_PATH) as fh:
        woif_src = fh.read()
    assert "_build_test_result_from_miner(_miner_result" in woif_src, \
        "the range-miner call site no longer routes to the isolated helper"
    assert "phase5_sink" in woif_src, \
        "the range-miner call site no longer wires a Phase-5 sink"


# ═════════════════════════════════════════════════════════════════════════════
# G11 — mutation proof
# ═════════════════════════════════════════════════════════════════════════════
_MUTATION_REPORT: List[str] = []

# (label, target file, old text, new text, gates that must go red)
_MUTANTS: List[Tuple[str, str, str, str, Tuple[str, ...]]] = [
    (
        "restored set union at ingress",
        "woif",
        "        accumulator['bidirectional'].extend(constant_records)\n"
        "        accumulator['bidirectional'].extend(variable_records)",
        "        _by_seed = {}\n"
        "        for _r in constant_records + variable_records:\n"
        "            _by_seed[_r['seed']] = _r\n"
        "        accumulator['bidirectional'].extend(_by_seed.values())",
        ("G2", "G8"),
    ),
    (
        "variable records built from the CONSTANT maps",
        "woif",
        "            fwd_map_variable, rev_map_variable,",
        "            fwd_map_constant, rev_map_constant,",
        ("G2", "G3", "G9"),
    ),
    (
        # The constant-bias defect in its own right: the variable records are
        # built from the correct maps, then their aggregates are overwritten
        # from the constant block — exactly what the pre-fix adapter did by
        # using `len(bidi_constant)` and the generic maps for every record.
        "variable aggregates overwritten with the constant counts",
        "records",
        "    return constant_records, variable_records",
        "    if constant_records and variable_records:\n"
        "        for _r in variable_records:\n"
        "            for _f in ('forward_count', 'reverse_count',\n"
        "                       'intersection_count', 'intersection_ratio',\n"
        "                       'survivor_overlap_ratio', 'intersection_weight'):\n"
        "                _r[_f] = constant_records[0][_f]\n"
        "    return constant_records, variable_records",
        ("G3", "G9"),
    ),
    (
        "string skip_range restored",
        "records",
        '        "skip_range":                ctx["skip_max"] - ctx["skip_min"],',
        '        "skip_range":                f\'{ctx["skip_min"]}-{ctx["skip_max"]}\',',
        ("G5", "G9", "G12"),
    ),
    (
        "'all' sessions fallback restored",
        "records",
        "    if isinstance(sessions, str):\n"
        "        raise CanonicalRecordContractError(",
        "    if isinstance(sessions, str):\n"
        "        return [sessions]\n"
        "    if False:\n"
        "        raise CanonicalRecordContractError(",
        ("G6",),
    ),
    (
        "adapter-ingress validation removed (producer validation intact)",
        "woif",
        '    validate_trial_populations(pw_result, origin="adapter-ingress")',
        "    pass  # ingress wall removed",
        ("G1", "G4"),
    ),
    (
        # REV3 §0.3, binding: no canonical adapter may reconstruct a map from a
        # record list. This mutant prefers the telemetry list whenever one is
        # present — the plausible-looking "the records are richer" regression.
        "map reconstructed from forward_records_hybrid",
        "woif",
        '    fwd_map_variable = pw_result["forward_map_variable"]',
        "    fwd_map_variable = ({_e['seed']: _e['match_rate']\n"
        "                         for _e in pw_result.get('forward_records_hybrid', [])}\n"
        '                        or pw_result["forward_map_variable"])',
        ("G10",),
    ),
    (
        # The `.get(..., {})` default is only OBSERVABLE once the ingress wall
        # is gone — behind the wall it is unreachable code. So this mutant
        # restores the pre-fix pair together: no wall, and a missing field
        # silently becoming an empty population. That is exactly the failure
        # mode the version stamp exists to prevent, and it must be caught.
        "a missing v2 field defaults to {} (wall also removed)",
        "woif",
        ('    validate_trial_populations(pw_result, origin="adapter-ingress")',
         '    fwd_map_constant = pw_result["forward_map_constant"]\n'
         '    rev_map_constant = pw_result["reverse_map_constant"]\n'
         '    fwd_map_variable = pw_result["forward_map_variable"]\n'
         '    rev_map_variable = pw_result["reverse_map_variable"]\n'
         '    bidi_constant    = pw_result["bidirectional_constant"]\n'
         '    bidi_variable    = pw_result["bidirectional_variable"]'),
        ("    pass  # ingress wall removed",
         "    fwd_map_constant = pw_result.get('forward_map_constant', {})\n"
         "    rev_map_constant = pw_result.get('reverse_map_constant', {})\n"
         "    fwd_map_variable = pw_result.get('forward_map_variable', {})\n"
         "    rev_map_variable = pw_result.get('reverse_map_variable', {})\n"
         "    bidi_constant    = pw_result.get('bidirectional_constant', set())\n"
         "    bidi_variable    = pw_result.get('bidirectional_variable', set())"),
        ("G1",),
    ),
    (
        "intersection disagreement silently repaired at ingress",
        "records",
        "        if actual != expected:",
        "        if False and actual != expected:",
        ("G4",),
    ),
    (
        "producer egress validation removed",
        "records",
        '    validate_trial_populations(block, origin=origin)\n    return block',
        "    return block",
        ("G4",),
    ),
    (
        "combined bidirectional total stamped onto records",
        "records",
        '        "bidirectional_count":       len(both),\n'
        '        "intersection_count":        len(both),',
        '        "bidirectional_count":       len(both) + 1,\n'
        '        "intersection_count":        len(both),',
        ("G3", "G9"),
    ),
    (
        "sessions shared by reference with the caller",
        "records",
        "    out = list(sessions)",
        "    out = sessions if isinstance(sessions, list) else list(sessions)",
        ("G6",),
    ),
    (
        "records emitted in mode-major order",
        "woif",
        "        accumulator['bidirectional'].extend(constant_records)\n"
        "        accumulator['bidirectional'].extend(variable_records)",
        "        accumulator['bidirectional'].extend(variable_records)\n"
        "        accumulator['bidirectional'].extend(constant_records)",
        ("G8",),
    ),
    (
        "seed order within a mode no longer ascending",
        "records",
        "    for seed in sorted(both):",
        "    for seed in sorted(both, reverse=True):",
        ("G8", "G9"),
    ),
    (
        # G13 bite proof #1 [D6] — the miner path reverts to appending nothing.
        # Pre-D6 this WAS the required behaviour; post-D6 an inert path is the
        # defect, because a real trial's candidates silently vanish.
        "miner path appends no candidate",
        "woif",
        "    _counts = ingest_assembly(_assembly, accumulator)",
        "    _counts = ingest_assembly(_assembly, None)",
        ("G13",),
    ),
    (
        # G13 bite proof #2 [D6] — the threshold-gated flush is dropped,
        # shifting flush cadence away from the pre-D3.25 shared path.
        "miner path drops the flush call",
        "woif",
        "    if accumulator is not None:\n"
        "        # [S152] Same call, same place, same cadence as every other backend.\n"
        '        _flush_npz_incremental(accumulator, label=f"chunk/trial-{trial_number}")',
        "    if accumulator is not None:\n"
        "        pass",
        ("G13",),
    ),
    (
        # G13 bite proof #3 [D6] — the fail-closed guard is downgraded to a
        # swallowed exception plus the pre-D6 all-zero TestResult, which is
        # exactly the fabricated zero D6 forbids.
        "miner path fabricates a zero result",
        "woif",
        "    _assembly = require_assembly(phase5_sink, miner_result,\n"
        "                                 trial_number=trial_number)",
        "    try:\n"
        "        _assembly = require_assembly(phase5_sink, miner_result,\n"
        "                                     trial_number=trial_number)\n"
        "    except Exception:\n"
        "        return TestResult(config=config, forward_count=0,\n"
        "                          reverse_count=0, bidirectional_count=0,\n"
        "                          iteration=trial_number)",
        ("G13",),
    ),
]

_GATE_FNS = {}          # filled in by main()


def _load_mutant(target: str, old, new):
    """Return (records_module, woif_module) with the textual mutation applied.

    `old`/`new` are either a pair of strings (one edit) or, when a mutation is
    only observable as a combination, tuples of matching length applied in
    order. The mutated source is exec'd into a fresh module object; nothing on
    disk is touched. When `records` is mutated, a fresh `woif` is built on top
    of it so the adapter really calls the mutated normalizer.
    """
    with open(_RECORDS_PATH) as fh:
        rec_src = fh.read()
    with open(_WOIF_PATH) as fh:
        woif_src = fh.read()

    olds = old if isinstance(old, tuple) else (old,)
    news = new if isinstance(new, tuple) else (new,)
    assert len(olds) == len(news), "mutation anchor/replacement count mismatch"

    for o, n in zip(olds, news):
        if target == "records":
            assert rec_src.count(o) == 1, \
                f"mutation anchor not unique in canonical_records.py: {o!r}"
            rec_src = rec_src.replace(o, n)
        else:
            assert woif_src.count(o) == 1, \
                f"mutation anchor not unique in window_optimizer_integration_final.py: {o!r}"
            woif_src = woif_src.replace(o, n)

    rec_mod = types.ModuleType("utils.canonical_records__mutant")
    rec_mod.__file__ = _RECORDS_PATH
    exec(compile(rec_src, _RECORDS_PATH, "exec"), rec_mod.__dict__)

    woif_mod = types.ModuleType("window_optimizer_integration_final__mutant")
    woif_mod.__file__ = _WOIF_PATH
    saved = sys.modules.get("utils.canonical_records")
    sys.modules["utils.canonical_records"] = rec_mod
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(woif_src, _WOIF_PATH, "exec"), woif_mod.__dict__)
    finally:
        if saved is not None:
            sys.modules["utils.canonical_records"] = saved
    return rec_mod, woif_mod


@contextlib.contextmanager
def _swapped(rec_mod, woif_mod):
    """Point this harness's PROD/WOIF handles at the mutant for one run."""
    global PROD, WOIF
    old_prod, old_woif = PROD, WOIF
    old_sys = sys.modules.get("utils.canonical_records")
    PROD, WOIF = rec_mod, woif_mod
    sys.modules["utils.canonical_records"] = rec_mod
    try:
        yield
    finally:
        PROD, WOIF = old_prod, old_woif
        if old_sys is not None:
            sys.modules["utils.canonical_records"] = old_sys


def g11_mutation_proof():
    survivors = []
    for label, target, old, new, expect_red in _MUTANTS:
        rec_mod, woif_mod = _load_mutant(target, old, new)
        reds = {}
        with _swapped(rec_mod, woif_mod):
            for gate in expect_red:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        _GATE_FNS[gate]()
                except Exception as exc:                          # noqa: BLE001
                    sig = str(exc).strip().splitlines()
                    reds[gate] = f"{type(exc).__name__}: {sig[0] if sig else ''}"
        if not reds:
            survivors.append(label)
            _MUTATION_REPORT.append(
                f"  SURVIVED  {label}  (expected red: {', '.join(expect_red)})")
        else:
            for gate, sig in sorted(reds.items()):
                _MUTATION_REPORT.append(f"  killed by {gate:<4} {label}\n"
                                        f"              -> {sig[:150]}")
    assert not survivors, (
        f"{len(survivors)} mutant(s) survived the gate suite — the gates do not "
        f"fail on the wrong behavior: {survivors}")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════
_results: List[Tuple[str, bool, str]] = []


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  ok    {name}")
    except Exception:
        _results.append((name, False, traceback.format_exc()))
        print(f"  FAIL  {name}")


def main():
    _GATE_FNS.update({
        "G1": g1_producer_contract_shape,
        "G2": g2_same_seed_both_modes,
        "G3": g3_mode_specific_aggregates,
        "G4": g4_intersection_consistency,
        "G5": g5_skip_range,
        "G6": g6_sessions,
        "G7": g7_no_generic_map_authority,
        "G8": g8_ordering_and_dual_preservation,
        "G9": g9_canonical_oracle,
        "G10": g10_record_list_is_not_a_map,
        "G12": g12_d3_conformance,
        "G13": g13_miner_path_isolation,
    })

    print("=" * 78)
    print("S172 Phase-5 D3.25 — mode-preserving backend contract + canonical "
          "candidate ingress")
    print("=" * 78)

    _check("G1:  v2 producer contract shape at BOTH boundaries",
           g1_producer_contract_shape)
    _check("G2:  same seed, both modes, different rates -> TWO records",
           g2_same_seed_both_modes)
    _check("G3:  mode-specific aggregates (no constant bias)",
           g3_mode_specific_aggregates)
    _check("G4:  intersection consistency fires BEFORE accumulator mutation",
           g4_intersection_consistency)
    _check("G5:  skip_range is the integer difference", g5_skip_range)
    _check("G6:  sessions canonical form + defensive copy", g6_sessions)
    _check("G7:  legacy generic maps carry no authority",
           g7_no_generic_map_authority)
    _check("G8:  trial-major / mode-minor ordering + dual preservation",
           g8_ordering_and_dual_preservation)
    _check("G9:  hand-written 24-field canonical oracle (load-bearing)",
           g9_canonical_oracle)
    _check("G10: a record list is NOT a map", g10_record_list_is_not_a_map)
    _check("G12: D3 conformance — records_to_arrays + validate_array_bundle",
           g12_d3_conformance)
    _check("G13: miner path is isolated — no ingress, flush cadence preserved",
           g13_miner_path_isolation)
    _check(f"G11: mutation proof ({len(_MUTANTS)} mutants)", g11_mutation_proof)

    if _MUTATION_REPORT:
        print("\n" + "-" * 78)
        print("G11 mutation evidence (red signature per mutant):")
        print("-" * 78)
        for line in _MUTATION_REPORT:
            print(line)

    print("\n" + "=" * 78)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} D3.25 gate checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All D3.25 gate checks green — PWC/ZMQ emit the four-map v2 contract "
          "and the adapter builds canonical per-mode candidates behind an "
          "ingress wall (pending Team Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
