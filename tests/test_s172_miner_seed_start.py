#!/usr/bin/env python3
"""S172 MINER SEED START — Step 1's seed_start reaches the sieve.

Governing brief: docs/CLAUDE_CODE_INSTRUCTIONS_MINER_SEED_START_20261114.md (Part 1).
Finding it fixes: docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md §6 — the
RANGE-MINER route swept [0, seed_count) whatever seed_start was.

CPU only. Real coordinator (run_trial_miner -> serve_trial -> assign_stripes -> scheduler
-> dispatch) against fake workers over real framed loopback sockets; WATCHER with dispatch
intercepted; every database, ledger and staging directory is throwaway. No GPU, no rig, no
live state.

Gates
  G-HOP        run_bidirectional_test hands run_trial_miner the requested seed_start
  G-RUN4-GEOM  start 2^31, total 2^31, stripe 2^26, both modes: 32 contiguous stripes per
               phase in all four phases, [2^31, 2^32)
  G-SHORT      a small non-zero start with a short final stripe
  G-RETRY      a hybrid retry keeps the start
  G-LEASE      a hybrid lease expiry keeps the start
  G-R2-ENDS    the interval wall at the exact endpoints and one past each (integration),
               and the coordinator's fail-closed start check (no default, no bool)
  G-R3-EVID    a real non-zero run credits; gap / wrong start / missing done refuse
  G-R5-ROUTES  PWC and ZMQ refuse a non-zero start before dispatch; at 0 their calls are
               argument-identical to the base commit's
  G-R7-ZERO    seed_start = 0: stripes, StripeAssign payloads and evidence byte-equal to
               the control captured on the unmodified tree
  G-E2E        WATCHER -> Run 4 argv -> the child's route guard -> the real coordinator ->
               evidence [2^31, 2^32) -> credit -> history COMPLETE over [0, 2^32)
Mutants are function-level, compiled against the PRODUCTION module globals (A8-B2).
Every gate terminates PASS | FAIL; only PASS accepts.
"""
import ast
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SCRATCH = os.path.join(ROOT, "data", "miner_seed_start", "suite_tmp")
os.makedirs(SCRATCH, exist_ok=True)

from fixtures import miner_seed_start_driver as D        # noqa: E402

CONTROL = os.path.join(HERE, "fixtures", "miner_seed_start_zero_control_b11b6fd.txt")
T31, T32 = 2 ** 31, 2 ** 32
STRIPE = 2 ** 26
RESULTS, MUTANTS = [], []


class Fail(AssertionError):
    pass


def ok_(cond, msg):
    if not cond:
        raise Fail(msg)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@contextlib.contextmanager
def tmpdir(prefix):
    d = tempfile.mkdtemp(prefix=prefix, dir=SCRATCH)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def expect_raises(exc_name, fn):
    try:
        fn()
    except BaseException as e:   # noqa: BLE001
        if type(e).__name__ != exc_name:
            raise Fail(f"expected {exc_name}, got {type(e).__name__}: {str(e)[:300]}")
        return e
    raise Fail(f"expected {exc_name}, nothing was raised")


def _stage1():
    """The stage-1 suite module (its WATCHER harness and synthetic-witness helpers)."""
    if "s1s_suite" in sys.modules:
        return sys.modules["s1s_suite"]
    spec = importlib.util.spec_from_file_location(
        "s1s_suite", os.path.join(HERE, "test_step1_search_record_stage1.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["s1s_suite"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# function-level mutation against PRODUCTION globals
# ---------------------------------------------------------------------------
def _def_node(tree, qualname):
    parts = qualname.split(".")
    body = tree.body
    node = None
    for part in parts:
        node = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            raise RuntimeError(f"definition {qualname} not found")
        body = node.body
    return node


def def_source(path, qualname, src=None):
    src = src if src is not None else open(path, encoding="utf-8").read()
    node = _def_node(ast.parse(src), qualname)
    start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
    lines = src.splitlines(keepends=True)[start:node.end_lineno]
    return textwrap.dedent("".join(lines))


@contextlib.contextmanager
def patched_def(modname, qualname, old=None, new=None, *, source=None):
    """Replace one live definition by a mutated (or historical) copy compiled in the REAL
    module's globals, so every name it resolves is production's."""
    mod = importlib.import_module(modname)
    if source is None:
        src = def_source(mod.__file__, qualname)
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"MUTANT ANCHOR ERROR {qualname}: found {n} times: {old[:80]!r}")
        source = src.replace(old, new)
    ns = {}
    exec(compile(source, f"{mod.__file__}<{qualname}>", "exec"), mod.__dict__, ns)
    name = qualname.split(".")[-1]
    owner = mod
    for part in qualname.split(".")[:-1]:
        owner = getattr(owner, part)
    saved = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
    setattr(owner, name, ns[name])
    try:
        yield ns[name]
    finally:
        setattr(owner, name, saved)


def run_gate(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail or ""))
    except BaseException as e:   # noqa: BLE001
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"[:900]))


def run_mutant(label, modname, qualname, old, new, gate_fn):
    try:
        with patched_def(modname, qualname, old, new):
            try:
                with quiet():
                    gate_fn()
                MUTANTS.append((label, False, "gate stayed GREEN under the mutant"))
            except BaseException as e:   # noqa: BLE001
                MUTANTS.append((label, True, f"RED: {type(e).__name__}: {str(e)[:220]}"))
    except RuntimeError as e:
        MUTANTS.append((label, False, f"ERROR (not a detection): {e}"))


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _cfg():
    from window_optimizer import WindowConfig
    return WindowConfig(window_size=12, offset=25, sessions=["evening"], skip_min=6,
                        skip_max=99, forward_threshold=0.71, reverse_threshold=0.47)


class _Stop(BaseException):
    pass


def _dataset():
    return _stage1().frozen_dataset().path


def _record_rtm(box):
    def rec(*a, **kw):
        box["calls"] = box.get("calls", 0) + 1
        box["kwargs"] = dict(kw)
        raise _Stop()
    return rec


def _miner_coord(**over):
    c = types.SimpleNamespace(use_range_miner=True, use_persistent_workers=False,
                              use_zmq_sqlite=False, config_file="distributed_config.json",
                              worker_pool_size=2, staging_dir=os.path.join(SCRATCH, "unused"))
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _rbt(coord, seed_start, seed_count, **kw):
    import window_optimizer_integration_final as W
    return W.run_bidirectional_test(coord, _cfg(), _dataset(), seed_start=seed_start,
                                    seed_count=seed_count, prng_base="java_lcg",
                                    test_both_modes=kw.get("both", True),
                                    forward_threshold=0.71, reverse_threshold=0.47,
                                    trial_number=1, accumulator=None)


def _contiguous(rows, phase):
    rs = sorted((r for r in rows if r[3] == phase), key=lambda r: r[1])
    cur = rs[0][1]
    for r in rs:
        if r[1] != cur:
            return None
        cur = r[1] + r[2]
    return rs[0][1], cur, len(rs)


def _evidence(r):
    import window_optimizer_integration_final as W
    coord = types.SimpleNamespace(staging_dir=os.path.dirname(r["ledger_path"]),
                                  miner_output_dir=None)
    return W._step1_miner_sweep_evidence({"run_id": r["run_id"],
                                          "committed": r["committed"]}, coord)


def _credit(ev, req, phases=(1, 2, 3, 4)):
    from utils.step1_search_ledger import credit_decision
    return credit_decision(requested_interval=req, required_phases=phases,
                           evidence_by_trial={0: ev}, trial_states={0: "COMPLETE"})


_RUN4_CACHE = {}


def _run4_run():
    """ONE real run at Run 4's shape, reused by G-RUN4-GEOM and G-R3-EVID. Kept on disk
    (not in a tmpdir) until the battery ends so the ledger can be read and mutated."""
    if "r" not in _RUN4_CACHE:
        d = tempfile.mkdtemp(prefix="run4_", dir=SCRATCH)
        _RUN4_CACHE["dir"] = d
        with quiet():
            _RUN4_CACHE["r"] = D.run_shape(d, seed_start=T31, total_seeds=T31,
                                           stripe_size=STRIPE, test_both_modes=True)
    return _RUN4_CACHE["r"]


# ===========================================================================
# R1
# ===========================================================================
def g_hop():
    """run_bidirectional_test (miner route) -> run_trial_miner: the requested start is
    handed on, by name, from the function's own parameter."""
    import window_optimizer_integration_final as W
    box = {}
    saved = W.run_trial_miner
    W.run_trial_miner = _record_rtm(box)
    try:
        with quiet():
            try:
                _rbt(_miner_coord(), T31, 1000)
            except _Stop:
                pass
    finally:
        W.run_trial_miner = saved
    kw = box.get("kwargs") or {}
    ok_(box.get("calls") == 1, f"run_trial_miner called {box.get('calls')} times")
    ok_(kw.get("seed_start") == T31 and kw.get("total_seeds") == 1000,
        f"seed_start={kw.get('seed_start')!r} total_seeds={kw.get('total_seeds')!r}")
    tree = ast.parse(open(W.__file__, encoding="utf-8").read())
    fn = _def_node(tree, "run_bidirectional_test")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "run_trial_miner"]
    ok_(len(calls) == 1, f"{len(calls)} run_trial_miner calls in run_bidirectional_test")
    kws = {k.arg: k.value for k in calls[0].keywords}
    ok_(isinstance(kws.get("seed_start"), ast.Name) and kws["seed_start"].id == "seed_start",
        f"seed_start= is {ast.unparse(kws['seed_start']) if 'seed_start' in kws else 'absent'}")
    return f"run_trial_miner(seed_start={T31}, total_seeds=1000); AST: seed_start=seed_start"


def g_run4_geom():
    r = _run4_run()
    ok_(r["committed"] and not r["error"], f"run did not commit: {r['state']} {r['error']}")
    rows = r["stripes"]
    det = []
    for ph in (1, 2, 3, 4):
        c = _contiguous(rows, ph)
        ok_(c is not None, f"phase {ph}: stripes are not contiguous")
        ok_(c == (T31, T32, 32), f"phase {ph}: {c}")
        ok_(all(x[5] == "done" for x in rows if x[3] == ph), f"phase {ph}: not all done")
        ok_(all(x[2] == STRIPE for x in rows if x[3] == ph), f"phase {ph}: a stripe != 2^26")
        det.append(f"p{ph}=32")
    # What the WORKERS were told, not only what the ledger holds.
    ok_(len(r["assigns"]) == 128, f"{len(r['assigns'])} StripeAssigns")
    by_row = {(x[0], x[3]): (x[1], x[2]) for x in rows}
    for a in r["assigns"]:
        ok_(by_row.get((a["stripe_id"], a["phase"])) == (a["seed_start"], a["seed_count"]),
            f"assign {a['stripe_id']} [{a['seed_start']}, +{a['seed_count']}) != ledger row")
    ok_(min(a["seed_start"] for a in r["assigns"]) == T31 and
        max(a["seed_start"] + a["seed_count"] for a in r["assigns"]) == T32,
        "assigned seeds escape [2^31, 2^32)")
    return f"{', '.join(det)}; first 2147483648, last ends 4294967296; 128 assigns = ledger"


def g_short():
    with tmpdir("short_") as d:
        with quiet():
            r = D.run_shape(d, seed_start=1000, total_seeds=1050, stripe_size=250,
                            test_both_modes=False, family_name="java_lcg_hybrid", phase=3,
                            behaviours=("complete", "complete"))
    ok_(r["committed"], f"not committed: {r['state']} {r['error']}")
    got = [(x[1], x[2]) for x in r["stripes"]]
    want = [(1000, 250), (1250, 250), (1500, 250), (1750, 250), (2000, 50)]
    ok_(got == want, f"stripes {got}")
    ok_(sorted((a["seed_start"], a["seed_count"]) for a in r["assigns"]) == want,
        "assigns differ from the ledger")
    return "[1000,1250) [1250,1500) [1500,1750) [1750,2000) [2000,2050) — short last = 50"


def _same_start_across_attempts(r):
    rows = {x[0]: x[1] for x in r["stripes"]}
    multi = {}
    for a in r["assigns"]:
        ok_(a["seed_start"] == rows[a["stripe_id"]],
            f"{a['stripe_id']} attempt {a['attempt']} at {a['seed_start']} != row {rows[a['stripe_id']]}")
        multi.setdefault(a["stripe_id"], set()).add(a["attempt"])
    re_ = {k: v for k, v in multi.items() if len(v) > 1}
    ok_(re_, f"no stripe was re-assigned: {multi}")
    return re_


def g_retry():
    with tmpdir("retry_") as d:
        with quiet():
            r = D.run_shape(d, seed_start=T31 + 7, total_seeds=500, stripe_size=250,
                            test_both_modes=False, family_name="java_lcg_hybrid", phase=3,
                            behaviours=("fail", "complete"))
    ok_(r["committed"], f"not committed: {r['state']} {r['error']}")
    re_ = _same_start_across_attempts(r)
    return f"re-assigned {sorted(re_)} at the same start ({T31 + 7} base); committed"


def g_lease():
    with tmpdir("lease_") as d:
        with quiet():
            r = D.run_shape(d, seed_start=T31 + 11, total_seeds=500, stripe_size=250,
                            test_both_modes=False, family_name="java_lcg_hybrid", phase=3,
                            behaviours=("silent", "complete"), compute_lease_timeout=1.5,
                            serve_timeout=90.0)
    ok_(r["committed"], f"not committed: {r['state']} {r['error']}")
    re_ = _same_start_across_attempts(r)
    return f"lease-expired stripe(s) {sorted(re_)} re-placed at the same start; committed"


# ===========================================================================
# R2
# ===========================================================================
def g_r2_ends():
    import window_optimizer_integration_final as W
    from utils.seed_coverage_ledger import SeedDomainPreflightError  # noqa: F401
    out = []
    cases = [((0, 1000), True), ((T32 - 1000, 1000), True), ((-1, 1000), False),
             ((T32 - 999, 1000), False), ((T32, 1), False)]
    saved = W.run_trial_miner
    try:
        for (ss, n), accept in cases:
            box = {}
            W.run_trial_miner = _record_rtm(box)
            raised = None
            with quiet():
                try:
                    _rbt(_miner_coord(), ss, n)
                except _Stop:
                    pass
                except BaseException as e:   # noqa: BLE001
                    raised = e
            if accept:
                ok_(box.get("calls") == 1 and raised is None,
                    f"[{ss}, {ss + n}) refused or not dispatched: {raised!r}")
            else:
                ok_(box.get("calls") is None, f"[{ss}, {ss + n}) reached run_trial_miner")
                ok_(type(raised).__name__ == "SeedDomainPreflightError",
                    f"[{ss}, {ss + n}) raised {type(raised).__name__}: {raised}")
            out.append(f"[{ss},{ss + n}){'ok' if accept else 'refused'}")
    finally:
        W.run_trial_miner = saved

    # The coordinator's own check: an omitted, None, bool, float or negative start is
    # refused before the ledger exists; 0 proceeds.
    from miner.range_miner_coordinator import run_trial_miner
    for bad in ("OMIT", None, True, 1.0, -1):
        with tmpdir("rtm_") as d:
            stg = os.path.join(d, "stg")
            served = {}
            kw = dict(staging_dir=stg, skip_min=0, skip_max=0, window_anchor=0,
                      generator_phase=0, window_size=3,
                      staging_high_water_bytes=256 * 1024 ** 2,
                      _serve=lambda c, ctx: served.setdefault("ctx", ctx) and {})
            if bad != "OMIT":
                kw["seed_start"] = bad
            e = expect_raises("ValueError", lambda: run_trial_miner(
                "x", None, 1, "java_lcg", [1], 1000, 0.3, 0.3, False, "/x.json", **kw))
            ok_("seed_start" in str(e), f"{bad!r}: refusal does not name seed_start: {e}")
            ok_(not served and not os.path.exists(os.path.join(stg, "miner_ledger.db")),
                f"{bad!r}: the trial was created before the refusal")
    with tmpdir("rtm0_") as d:
        served = {}
        run_trial_miner("x", None, 1, "java_lcg", [1], 1000, 0.3, 0.3, False, "/x.json",
                        staging_dir=os.path.join(d, "stg"), skip_min=0, skip_max=0,
                        window_anchor=0, generator_phase=0, window_size=3, seed_start=0,
                        staging_high_water_bytes=256 * 1024 ** 2,
                        _serve=lambda c, ctx: served.update(ctx=ctx) or {})
        ok_(served.get("ctx", {}).get("seed_start") == 0, "seed_start=0 not on the context")

    # serve_trial, reached directly: a context with no start refuses before the socket.
    from miner.range_miner_coordinator import build_coordinator
    with tmpdir("srv_") as d:
        coord = build_coordinator(staging_dir=d, db_path=os.path.join(d, "l.db"))
        e = expect_raises("ValueError", lambda: coord.serve_trial(
            {"run_id": "r", "total_seeds": 10}))
        ok_("seed_start" in str(e), f"serve_trial refusal does not name seed_start: {e}")
        ok_(getattr(coord, "bound_addr", None) is None, "serve_trial bound a socket")
    return ("; ".join(out) + "; coordinator refuses omitted/None/True/1.0/-1 before the "
            "ledger; 0 reaches the context; serve_trial refuses a context with no start")


# ===========================================================================
# R3
# ===========================================================================
def _mut_ledger(src_ledger, sql):
    d = tempfile.mkdtemp(prefix="mut_", dir=SCRATCH)
    dst = os.path.join(d, "miner_ledger.db")
    shutil.copyfile(src_ledger, dst)
    con = sqlite3.connect(dst)
    try:
        n = con.execute(sql[0], sql[1]).rowcount
        con.commit()
    finally:
        con.close()
    ok_(n >= 1, f"mutation touched no row: {sql}")
    return d


def g_r3_evid():
    import window_optimizer_integration_final as W
    r = _run4_run()
    ok_(r["committed"], "run 4 shape did not commit")
    ev = _evidence(r)
    want = {str(p): [T31, T32] for p in (1, 2, 3, 4)}
    ok_(ev and ev["swept_intervals"] == want, f"evidence {ev and ev['swept_intervals']}")
    credited, code, det, _ = _credit(ev, (T31, T32))
    ok_(credited and code == "credited", f"real run not credited: {code} {det}")
    # The same evidence must NOT credit the lower half.
    c0, code0, _, _ = _credit(ev, (0, T31))
    ok_(not c0 and code0 == "interval_mismatch", f"upper-half evidence credited [0,2^31): {code0}")
    run_id = r["run_id"]
    real_id = {x[0] for x in r["stripes"]}
    sid = lambda s: s.replace("<RUN>", run_id)   # noqa: E731
    p2 = sorted((x for x in r["stripes"] if x[3] == 2), key=lambda x: x[1])
    p1 = sorted((x for x in r["stripes"] if x[3] == 1), key=lambda x: x[1])
    p4 = sorted((x for x in r["stripes"] if x[3] == 4), key=lambda x: x[1])
    muts = {
        "gap (phase 2 middle stripe deleted)": (
            "DELETE FROM stripes WHERE run_id=? AND stripe_id=?", (run_id, sid(p2[5][0]))),
        "wrong start (phase 1 shifted down by 2^31)": (
            "UPDATE stripes SET seed_start = seed_start - 2147483648 WHERE run_id=? AND "
            "phase=1", (run_id,)),
        "missing done (one phase 4 stripe claimed)": (
            "UPDATE stripes SET state='claimed' WHERE run_id=? AND stripe_id=?",
            (run_id, sid(p4[-1][0]))),
        "overlap (phase 2 stripe 6 starts one seed early)": (
            "UPDATE stripes SET seed_start = seed_start - 1 WHERE run_id=? AND stripe_id=?",
            (run_id, sid(p2[6][0]))),
    }
    ok_(real_id and p1, "no stripes to mutate")
    out = []
    for name, sql in muts.items():
        d = _mut_ledger(r["ledger_path"], sql)
        try:
            evm = W._step1_miner_sweep_evidence(
                {"run_id": run_id, "committed": True},
                types.SimpleNamespace(staging_dir=d, miner_output_dir=None))
            cm, codem, detm, _ = _credit(evm, (T31, T32))
            ok_(not cm, f"{name}: CREDITED ({detm})")
            out.append(f"{name} -> {codem}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return f"real run credited [2^31,2^32) in 4 phases; " + "; ".join(out)


# ===========================================================================
# R5
# ===========================================================================
BASE = "b11b6fde7ab57321b42c418f87af8cc2c75c814d"


def _base_src(path):
    return subprocess.check_output(["git", "-C", ROOT, "show", f"{BASE}:{path}"], text=True)


def _route_call(route, seed_start, *, base=False):
    """Drive run_bidirectional_test down PWC or ZMQ with every dispatch seam recorded.
    Returns (record, exception)."""
    import window_optimizer_integration_final as W
    rec = {"calls": []}

    def _pwc(**kw):
        rec["calls"].append(("run_trial_persistent", kw))
        raise _Stop()

    def _zmq(**kw):
        rec["calls"].append(("run_trial_zmq_sqlite", kw))
        raise _Stop()

    class _ZMQSC:
        def __init__(self, **kw):
            rec["calls"].append(("ZMQSQLiteCoordinator", kw))

    fake_zmq = types.ModuleType("zmq_sqlite_coordinator")
    fake_zmq.ZMQSQLiteCoordinator = _ZMQSC
    fake_zmq.run_trial_zmq_sqlite = _zmq
    saved = (W.run_trial_persistent, W.run_trial_zmq_sqlite,
             sys.modules.get("zmq_sqlite_coordinator"))
    W.run_trial_persistent, W.run_trial_zmq_sqlite = _pwc, _zmq
    sys.modules["zmq_sqlite_coordinator"] = fake_zmq
    coord = _miner_coord(use_range_miner=False, use_persistent_workers=(route == "pwc"),
                         use_zmq_sqlite=(route == "zmq"))
    exc = None
    ctx = (patched_def("window_optimizer_integration_final", "run_bidirectional_test",
                       source=def_source(None, "run_bidirectional_test",
                                         _base_src("window_optimizer_integration_final.py")))
           if base else contextlib.nullcontext())
    try:
        with ctx, quiet():
            try:
                _rbt(coord, seed_start, 1000)
            except _Stop:
                pass
            except BaseException as e:   # noqa: BLE001
                exc = e
    finally:
        W.run_trial_persistent, W.run_trial_zmq_sqlite = saved[0], saved[1]
        if saved[2] is None:
            sys.modules.pop("zmq_sqlite_coordinator", None)
        else:
            sys.modules["zmq_sqlite_coordinator"] = saved[2]
    return rec, exc


def _canon_calls(rec):
    out = []
    for name, kw in rec["calls"]:
        out.append((name, {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                               else type(v).__name__) for k, v in sorted(kw.items())}))
    return json.dumps(out, sort_keys=True, default=str)


# [PWC-STAGE3] Owner ruling 3 (2026-11-19, docs/PWC_BACKUP_OWNER_RULINGS_20261119.md):
# PWC is no longer refused at a non-zero start; it is handed seed_start. Stage 3 also
# hands it pwc_coordinator_address and min_workers. ZMQ is unchanged.
_PWC_STAGE3_ADDED_KW = ("seed_start", "pwc_coordinator_address", "min_workers")


def _canon_calls_without(rec, drop):
    return _canon_calls({"calls": [(n, {k: v for k, v in kw.items() if k not in drop})
                                   for n, kw in rec["calls"]]})


def g_r5_routes():
    import window_optimizer as WO
    out = []
    # PWC [PWC-STAGE3]: a non-zero start is dispatched, carrying the start
    rec, exc = _route_call("pwc", 1000)
    ok_(exc is None, f"pwc at 1000 raised: {type(exc).__name__}: {exc}")
    ok_(len(rec["calls"]) == 1 and rec["calls"][0][0] == "run_trial_persistent"
        and rec["calls"][0][1].get("seed_start") == 1000,
        f"pwc at 1000 did not dispatch with seed_start=1000: {_canon_calls(rec)}")
    live, e1 = _route_call("pwc", 0)
    base, e2 = _route_call("pwc", 0, base=True)
    ok_(e1 is None and e2 is None, f"pwc at 0 raised: {e1!r} / {e2!r}")
    ok_(live["calls"] and live["calls"][0][1].get("seed_start") == 0
        and _canon_calls_without(live, _PWC_STAGE3_ADDED_KW) == _canon_calls(base),
        f"pwc at 0 differs from {BASE[:7]} beyond the stage-3 keywords:\n live "
        f"{_canon_calls(live)}\n base {_canon_calls(base)}")
    out.append(f"pwc: 1000 dispatched with seed_start=1000; 0 = {BASE[:7]} plus "
               f"{list(_PWC_STAGE3_ADDED_KW)}")
    for route in ("zmq",):
        rec, exc = _route_call(route, 1000)
        ok_(type(exc).__name__ == "Step1RouteStartUnsupportedError",
            f"{route} at 1000: {type(exc).__name__}: {exc}")
        ok_(route in str(exc).lower() and "1000" in str(exc), f"{route}: reason not named: {exc}")
        ok_(not rec["calls"], f"{route} at 1000 dispatched: {[c[0] for c in rec['calls']]}")
        live, e1 = _route_call(route, 0)
        base, e2 = _route_call(route, 0, base=True)
        ok_(e1 is None and e2 is None, f"{route} at 0 raised: {e1!r} / {e2!r}")
        ok_(live["calls"] and _canon_calls(live) == _canon_calls(base),
            f"{route} at 0 differs from {BASE[:7]}:\n live {_canon_calls(live)}\n base "
            f"{_canon_calls(base)}")
        out.append(f"{route}: 1000 refused with no dispatch; 0 = {BASE[:7]} "
                   f"({len(live['calls'])} call(s), argument-identical)")

    # The child's entry (run_bayesian_optimization) and run_with_config, before the
    # coordinator exists.
    import coordinator as C

    class Trip:
        def __init__(self, *a, **k):
            raise RuntimeError("TRIPWIRE: coordinator constructed")

    saved = (C.MultiGPUCoordinator, WO.MultiGPUCoordinator, WO.COORDINATOR_AVAILABLE)
    C.MultiGPUCoordinator = WO.MultiGPUCoordinator = Trip
    WO.COORDINATOR_AVAILABLE = True

    def rbo(ss, **flags):
        return lambda: WO.run_bayesian_optimization(
            lottery_file="/nonexistent.json", trials=1, output_config="/dev/null",
            seed_count=1000, seed_start=ss, **flags)

    try:
        with quiet():
            for flags in ({"use_zmq_sqlite": True},):          # [PWC-STAGE3] PWC not refused
                expect_raises("Step1RouteStartUnsupportedError", rbo(T31, **flags))
            for ss, flags in ((T31, {"use_range_miner": True}), (T31, {}),
                              (T31, {"use_persistent_workers": True}),   # [PWC-STAGE3]
                              (0, {"use_persistent_workers": True}),
                              (0, {"use_zmq_sqlite": True})):
                e = expect_raises("RuntimeError", rbo(ss, **flags))
                ok_("TRIPWIRE" in str(e), f"{ss} {flags}: {e}")
            cfgf = os.path.join(SCRATCH, "rwc_cfg.json")
            with open(cfgf, "w") as f:
                json.dump({"window_size": 12, "offset": 25, "skip_min": 6, "skip_max": 99,
                           "sessions": ["evening"], "prng_type": "java_lcg",
                           "forward_threshold": 0.71, "reverse_threshold": 0.47}, f)
            rwc = lambda it, pwc: (lambda: WO.run_with_config(   # noqa: E731
                config_file=cfgf, lottery_file="/nonexistent.json", max_seeds=1000,
                iterations=it, use_persistent_workers=pwc))
            # [PWC-STAGE3] a PWC plan of two iterations is no longer refused
            for it, pwc in ((1, True), (2, True), (2, False)):
                e = expect_raises("RuntimeError", rwc(it, pwc))
                ok_("TRIPWIRE" in str(e), f"run_with_config it={it} pwc={pwc}: {e}")
    finally:
        C.MultiGPUCoordinator, WO.MultiGPUCoordinator, WO.COORDINATOR_AVAILABLE = saved
    out.append("child entry: ZMQ at 2^31 refused before the coordinator; miner, legacy and "
               "PWC at 2^31, PWC/ZMQ at 0 reach it; run_with_config PWC x1/x2 and legacy x2 "
               "reach it [PWC-STAGE3]")
    return "; ".join(out)


# ===========================================================================
# R7
# ===========================================================================
def g_r7_zero():
    with open(CONTROL) as f:
        ctl = json.load(f)
    ok_(ctl.get("completion_sentinel") == "CAPTURE COMPLETE", "control incomplete")
    ok_(ctl["driver_sha256"] == D.driver_sha256(),
        "the driver changed since the control was captured — comparison invalid")
    base_coord = subprocess.check_output(
        ["git", "-C", ROOT, "show", f"{ctl['head_at_capture']}:miner/range_miner_coordinator.py"])
    ok_(hashlib.sha256(base_coord).hexdigest() == ctl["coordinator_sha256"],
        "control was not captured from the base commit's coordinator")
    live_coord = open(os.path.join(ROOT, "miner/range_miner_coordinator.py"), "rb").read()
    ok_(hashlib.sha256(live_coord).hexdigest() != ctl["coordinator_sha256"],
        "live coordinator equals the base — the comparison would be vacuous")
    import window_optimizer_integration_final as W
    out = []
    for name, shape in ctl["shapes"].items():
        with tmpdir(f"r7{name}_") as d:
            with quiet():
                r = D.run_shape(d, seed_start=0, **shape["params"])
            ok_(r["committed"], f"{name}: not committed: {r['state']} {r['error']}")
            ev = _evidence(r)
            ev = D._normalize(ev, [(r["run_id"], "<RUN>")])
        live = {"stripes": r["stripes"], "assigns": r["assigns"], "evidence": ev}
        dg = hashlib.sha256(D.canonical(live).encode()).hexdigest()
        if dg != shape["digest"]:
            for k in live:
                if D.canonical(live[k]) != D.canonical(shape[k]):
                    raise Fail(f"{name}: {k} differs from the zero-start control")
            raise Fail(f"{name}: digest differs")
        out.append(f"{name} {len(r['stripes'])} stripes / {len(r['assigns'])} assigns "
                   f"byte-equal ({dg[:12]})")
    return "; ".join(out) + f"; control from {ctl['head_at_capture'][:7]}"


# ===========================================================================
# End to end
# ===========================================================================
class _CapsWorker(D.FakeWorker):
    """A driver FakeWorker that advertises the PRODUCTION cap set (the coordinator
    quarantines any other) and splits each stripe into cap-sized sub-stripes, the way
    the real worker partitions (range_miner_worker.partition_stripe)."""

    def __init__(self, host, port, hostname, gpu_id, caps):
        super().__init__(host, port, hostname, gpu_id, "complete", caps["nvidia"])
        self.caps = dict(caps)

    def connect_register(self):
        import socket
        from miner.range_miner_protocol import RegisterMessage
        from miner.range_miner_worker import MinerFramedSocket, supported_variants
        self.fs = MinerFramedSocket(socket.create_connection((self.host, self.port)))
        self.fs.send_msg(RegisterMessage(
            worker_id=self.worker_id, hostname=self.hostname, gpu_id=self.gpu_id,
            gpu_name="fake", backend="cuda", vram_bytes=12 * 1024 ** 3,
            capabilities={"supported_variants": supported_variants(),
                          "seed_caps": self.caps}))

    def _answer(self, a):
        self.cap = int(self.caps["nvidia_hybrid" if "hybrid" in a.family_name else "nvidia"])
        return super()._answer(a)


def g_e2e():
    S = _stage1()
    import window_optimizer as WO
    import window_optimizer_integration_final as W
    from utils import step1_search_ledger as L
    with S.watcher_env() as (WA, cap, dbp, d):
        S.history_row(dbp, S.fake_witness(0, T31, gen="gen-attempt9-like"))
        agent = S.make_agent(WA, d)
        rc, out = S.cli_open(WA, agent, {**S.RUN4_BASE, **S.RUN4_PINS}, start=T31, end=T32,
                             reason="Run 4 (miner seed-start suite)")
        ok_(rc == 0 and "space=pinned" in out, out)
        res = S.run_step1(WA, agent, S.RUN4_BASE, WA.capture_step1_pin_bundle(S.RUN4_PINS))
        ok_(cap["argv"], f"no Run 4 command: {res.get('blocked_by')} {str(res.get('error'))[:300]}")
        argv = cap["argv"][-1]
        av = lambda f: S.argv_value(argv, f)   # noqa: E731
        ss, n = int(av("--seed-start")), int(av("--max-seeds"))
        ok_((ss, n) == (T31, T31), f"argv start/size {ss} {n}")
        ok_("--use-range-miner" in argv and "--use-persistent-workers" not in argv
            and "--use-zmq-sqlite" not in argv, "argv route is not RANGE-MINER alone")

        # The child's entry, with the argv's values: the domain wall and the route guard
        # pass and execution reaches the coordinator (a tripwire).
        import coordinator as C

        class Trip:
            def __init__(self, *a, **k):
                raise RuntimeError("TRIPWIRE: coordinator constructed")

        saved = (C.MultiGPUCoordinator, WO.MultiGPUCoordinator, WO.COORDINATOR_AVAILABLE)
        C.MultiGPUCoordinator = WO.MultiGPUCoordinator = Trip
        WO.COORDINATOR_AVAILABLE = True
        try:
            with quiet():
                e = expect_raises("RuntimeError", lambda: WO.run_bayesian_optimization(
                    lottery_file="/nonexistent.json", trials=1, output_config="/dev/null",
                    seed_count=n, seed_start=ss, use_range_miner=True))
            ok_("TRIPWIRE" in str(e), f"the child's entry refused the Run 4 argv: {e}")
        finally:
            C.MultiGPUCoordinator, WO.MultiGPUCoordinator, WO.COORDINATOR_AVAILABLE = saved

        # The REAL run_bidirectional_test -> REAL run_trial_miner. The wrapper supplies
        # ONLY harness seams (loopback socket, throwaway staging, a probe sink) and passes
        # every argument run_bidirectional_test chose — seed_start, total_seeds, pool —
        # through untouched. It stops after the coordinator returns, before Phase-5
        # assembly, which needs real survivors (labelled intercept).
        real_rtm = W.run_trial_miner
        box = {}
        work = tempfile.mkdtemp(prefix="e2e_", dir=SCRATCH)
        pool = int(av("--worker-pool-size"))
        stripe = int(av("--miner-stripe-size"))

        def wrapped(**kw):
            box["kw"] = {k: v for k, v in kw.items()
                         if isinstance(v, (int, float, str, bool, type(None)))}
            seams = dict(listen_sock=box["lsock"], staging_dir=os.path.join(work, "stg"),
                         miner_output_dir=None, phase5_sink=D.ProbeSink(),
                         serve_timeout=300.0,
                         miner_host="127.0.0.1")
            box["result"] = real_rtm(**{**kw, **seams})
            raise _Stop()

        import socket
        import threading
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(64)
        box["lsock"] = lsock
        port = lsock.getsockname()[1]
        caps = {"nvidia": int(av("--seed-cap-nvidia")), "amd": int(av("--seed-cap-amd")),
                "nvidia_hybrid": 2_500_000, "amd_hybrid": 1_000_000}
        # The hybrid caps are not on the argv; these are the values the integration
        # resolves when the coordinator carries none (its getattr defaults).
        coord = _miner_coord(worker_pool_size=pool, miner_stripe_size=stripe,
                             staging_dir=os.path.join(work, "stg"),
                             seed_cap_nvidia=caps["nvidia"], seed_cap_amd=caps["amd"])
        cfg = _cfg()
        W.run_trial_miner = wrapped

        def _drive():
            try:
                W.run_bidirectional_test(coord, cfg, S.frozen_dataset().path,
                                         seed_start=ss, seed_count=n, prng_base="java_lcg",
                                         test_both_modes=True, forward_threshold=0.71,
                                         reverse_threshold=0.47, trial_number=1)
            except _Stop:
                pass
            except BaseException:   # noqa: BLE001
                box["err"] = traceback.format_exc()

        try:
            t = threading.Thread(target=_drive, daemon=True)
            # [L-4 REPAIR] No pump nudger: the coordinator resumes deferred work itself.
            with quiet():
                t.start()
                import time
                time.sleep(0.2)
                workers = [_CapsWorker("127.0.0.1", port, f"rig{i}", 0, caps)
                           for i in range(pool)]
                for w in workers:
                    w.connect_register()
                    w.start()
                t.join(timeout=400)
                for w in workers:
                    w.stop()
        finally:
            W.run_trial_miner = real_rtm
        try:
            ok_(not box.get("err"), f"run_bidirectional_test failed: {box.get('err')}")
            r = box.get("result") or {}
            ok_(box["kw"].get("seed_start") == T31 and box["kw"].get("total_seeds") == T31,
                f"run_bidirectional_test handed {box['kw'].get('seed_start')} / "
                f"{box['kw'].get('total_seeds')}")
            ok_(r.get("committed"), f"coordinator did not commit: {r.get('state')}")
            ledger = os.path.join(work, "stg", "miner_ledger.db")
            con = sqlite3.connect(f"file:{ledger}?mode=ro&immutable=1", uri=True)
            rows = [tuple(x) for x in con.execute(
                "SELECT stripe_id, seed_start, seed_count, phase, family_name, state FROM "
                "stripes WHERE run_id=?", (r["run_id"],))]
            con.close()
            for ph in (1, 2, 3, 4):
                ok_(_contiguous(rows, ph) == (T31, T32, 32), f"phase {ph}: {_contiguous(rows, ph)}")
            ev = W._step1_miner_sweep_evidence(r, coord)
            ok_(ev and ev["swept_intervals"] == {str(p): [T31, T32] for p in (1, 2, 3, 4)},
                f"evidence {ev}")
            credited, code, det, _ = _credit(ev, (ss, ss + n),
                                             W._step1_required_phases("java_lcg", True))
            ok_(credited, f"not credited: {code} {det}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
            lsock.close()

        # Credit -> history. The publication witness is SYNTHETIC (labelled intercept):
        # D3.5 publication needs a real NPZ, which fake workers do not produce. The link
        # is the production one (record_link_pending, then record_chunk_link).
        led = L.Step1SearchLedger(dbp)
        sel = led.select("java_lcg", ["constant", "variable"])
        ok_(sel.state == L.SEL_GAP and sel.next_seed_start == T31, f"search state {sel.state}")
        S.credit_chunk(led, dbp, sel.search, T31, T32, study="e2e-study")
        import database_system as DS
        cur = DS.DistributedPRNGDatabase().get_certified_cursor("java_lcg", test_both_modes=True)
        ok_(cur.is_complete and cur.covered_seed_count == T32,
            f"history not COMPLETE: covered {cur.covered_seed_count}")
        sel2 = DS.DistributedPRNGDatabase().select_step1_search("java_lcg", test_both_modes=True)
        ok_(sel2.state == L.SEL_SEARCH_COMPLETE, f"search after credit: {sel2.state}")
    return ("WATCHER argv --seed-start 2147483648 --max-seeds 2147483648; child entry "
            "passes; real coordinator 4x32 stripes over [2^31,2^32); evidence = request; "
            "credited; history COMPLETE over [0,2^32); search COMPLETE")


# ===========================================================================
# main
# ===========================================================================
GATES = [
    ("G-HOP run_bidirectional_test hands on seed_start", g_hop),
    ("G-RUN4-GEOM 4 phases x 32 stripes over [2^31, 2^32)", g_run4_geom),
    ("G-SHORT non-zero start, short final stripe", g_short),
    ("G-RETRY hybrid retry keeps the start", g_retry),
    ("G-LEASE hybrid lease expiry keeps the start", g_lease),
    ("G-R2-ENDS interval wall endpoints; coordinator fail-closed start", g_r2_ends),
    ("G-R3-EVID evidence credits a real non-zero run; mutations refuse", g_r3_evid),
    ("G-R5-ROUTES ZMQ refuses non-zero before dispatch; PWC carries the start (stage 3); 0 unchanged", g_r5_routes),
    ("G-R7-ZERO seed_start 0 byte-equal to the base control", g_r7_zero),
    ("G-E2E WATCHER -> coordinator -> credit -> history COMPLETE", g_e2e),
]

I_MOD, C_MOD, WO_MOD = ("window_optimizer_integration_final", "miner.range_miner_coordinator",
                        "window_optimizer")
MUTANT_SPECS = [
    ("M1 integration drops seed_start= (the stage-1 shape)", I_MOD, "run_bidirectional_test",
     "            seed_start             = seed_start,\n", "", g_hop),
    ("M2 serve_trial assigns without base_start", C_MOD, "RangeMinerCoordinator.serve_trial",
     "run_id, fam, ph, total_seeds, eligible, base_start=base_start,",
     "run_id, fam, ph, total_seeds, eligible,", g_short),
    ("M3 run_trial_miner does not carry the start onto the context", C_MOD, "run_trial_miner",
     '        "seed_start": seed_start,            # [MINER SEED-START] the sweep\'s origin\n',
     '        "seed_start": 0,\n', g_short),
    ("M4 coordinator start check disabled", C_MOD, "run_trial_miner",
     "    if (isinstance(seed_start, bool) or not isinstance(seed_start, int)\n"
     "            or seed_start < 0):\n",
     "    if False:\n", g_r2_ends),
    ("M5 integration domain wall removed", I_MOD, "run_bidirectional_test",
     "        _s145_trial_wall(seed_start, seed_count,\n",
     "        (lambda *a, **k: None)(seed_start, seed_count,\n", g_r2_ends),
    # [PWC-STAGE3] the PWC guard call is gone (ruling 3); M6 now drops the start PWC is handed
    ("M6 PWC branch drops seed_start= (stage 3)", I_MOD, "run_bidirectional_test",
     "            seed_start        = seed_start,                                   # [PWC-STAGE3 ruling 3]\n",
     "", g_r5_routes),
    ("M7 ZMQ branch guard removed", I_MOD, "run_bidirectional_test",
     '        _route_start_guard(seed_start, route="zmq",\n',
     '        (lambda *a, **k: None)(seed_start, route="zmq",\n', g_r5_routes),
    ("M8 guard predicate ignores ZMQ", WO_MOD, "assert_step1_route_seed_start_supported",
     "    if seed_start != 0 and use_zmq_sqlite:\n",
     "    if False:\n", g_r5_routes),
    # [PWC-STAGE3] M9 (run_with_config plan guard removed) is retired: run_with_config
    # takes no ZMQ flag, so after ruling 3 its guard can no longer fire on any route and
    # the mutant would be an invalid experiment (VIR-2), not a survivor.
    ("M10 evidence reader accepts non-done stripes", I_MOD, "_step1_miner_sweep_evidence",
     '        if any(st != "done" for _, _, st in items):\n',
     "        if False:\n", g_r3_evid),
    ("M11 zero-start geometry perturbed (R7 is not vacuous)", C_MOD,
     "RangeMinerCoordinator.serve_trial",
     "run_id, fam, ph, total_seeds, eligible, base_start=base_start,",
     "run_id, fam, ph, total_seeds, eligible, base_start=base_start + 1,", g_r7_zero),
]


def main():
    print("=" * 78)
    print("S172 MINER SEED START — gate battery (CPU only, isolated state)")
    print("=" * 78)
    for name, fn in GATES:
        run_gate(name, fn)
        n, ok, det = RESULTS[-1]
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {det}", flush=True)
    print("MUTANTS (each must turn its gate RED)")
    for label, mod, qn, old, new, gate in MUTANT_SPECS:
        run_mutant(label, mod, qn, old, new, gate)
        l, det, txt = MUTANTS[-1]
        print(f"  [{'RED (detected)' if det else 'NOT DETECTED'}] {l}: {txt}", flush=True)
    if "dir" in _RUN4_CACHE:
        shutil.rmtree(_RUN4_CACHE["dir"], ignore_errors=True)
    g_ok = sum(1 for _, ok, _ in RESULTS if ok)
    m_ok = sum(1 for _, det, _ in MUTANTS if det)
    print(f"GATES {g_ok}/{len(RESULTS)}  MUTANTS {m_ok}/{len(MUTANTS)} detected")
    for n, ok, d in RESULTS:
        if not ok:
            print(f"--- FAIL {n}\n{d}")
    ok = g_ok == len(RESULTS) and m_ok == len(MUTANTS)
    print(f"COMPLETION SENTINEL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
