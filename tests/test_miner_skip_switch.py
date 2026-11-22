#!/usr/bin/env python3
"""test_miner_skip_switch.py: the RANGE-MINER constant-skip delivery switch
(`deliver_skip_range`; docs/drafts/miner_skip_switch_v1/PLAN.md).

Run from the repository root:   python3 -u tests/test_miner_skip_switch.py

CPU only: no GPU, no rig, no network, no live ledger. Files go under
<root>/logs/test_miner_skip_switch/ (ignored).

OFF MUST BE THE BASELINE, BYTE FOR BYTE. Every "off" gate compares the changed code with
HEAD's own code, loaded from `git show HEAD:<file>` beside it as a separate module:
  G-OFF-PAYLOAD     build_stripe_assign_payload, phases 1-4: identical canonical bytes
  G-OFF-DISPATCH    _dispatch_pending: the same StripeAssignMessage payloads
  G-OFF-SERVE-CTX   run_trial_miner (through its `_serve` seam): the same serve context,
                    no delivery record on the coordinator
  G-OFF-INTEG-CALL  run_bidirectional_test: run_trial_miner gets no new argument
  G-OFF-SEARCH-ID   presented_configuration: the same search_space_digest (key absent
                    or false)
  G-OFF-ESTIMATOR   estimator geometry without the key: the same request digest
  G-OFF-READERS     no record / NPZ / finalizer reader names the switch (static)
ON:
  G-ON-PAYLOAD      phases 1-2 carry the trial's pair; phases 3-4 byte-identical to off
  G-ON-DISPATCH     _dispatch_pending hands the run's pair; another run gets none
  G-ON-SERVE        run_trial_miner records the pair for the run and on its result;
                    an invalid pair is refused before the staging dir exists
  G-ON-WORKER-ARGS  the worker's own parse of the payload into BuildContext ->
                    _constant_prefix gives the int32 skip scalars = the pair (and [0, 16]
                    off) - the CPU half of the kernel argument path
  G-VALIDATION      0 <= skip_min <= skip_max <= 255, plain ints only
  G-SEARCH          switched-on and baseline runs cannot share a search (child preflight)
  G-ON-INTEG-CALL   run_bidirectional_test passes deliver_skip_range=True when on
  G-ON-ESTIMATOR    the estimator samples the delivered range when the geometry says so
Mutants: one source change each; each must make a named gate fail.
"""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
OUT = os.path.join(ROOT, "logs", "test_miner_skip_switch")
RMC_PATH = os.path.join(ROOT, "miner", "range_miner_coordinator.py")
WOI_PATH = os.path.join(ROOT, "window_optimizer_integration_final.py")
SSL_PATH = os.path.join(ROOT, "utils", "step1_search_ledger.py")
SVE_PATH = os.path.join(ROOT, "utils", "survivor_volume_estimator.py")
WRK_PATH = os.path.join(ROOT, "miner", "range_miner_worker.py")
GATE_PATH = os.path.join(ROOT, "scripts", "survivor_volume_gate.py")
BAY_PATH = os.path.join(ROOT, "window_optimizer_bayesian.py")


def head(rel):
    return subprocess.check_output(["git", "-C", ROOT, "show", f"HEAD:{rel}"], text=True)


def load(src, name, path):
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path
    sys.modules[name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, path, "exec"), mod.__dict__)
    return mod


SRC = {p: Path(p).read_text(encoding="utf-8") for p in (RMC_PATH, WOI_PATH, SSL_PATH, SVE_PATH, GATE_PATH,
                                                          BAY_PATH)}
BASE_RMC = load(head("miner/range_miner_coordinator.py"), "rmc_head", RMC_PATH)
BASE_SSL = load(head("utils/step1_search_ledger.py"), "ssl_head", SSL_PATH)
BASE_SVE = load(head("utils/survivor_volume_estimator.py"), "sve_head", SVE_PATH)
BASE_WOI = None               # loaded lazily (heavy)
MOD = {}                      # the live (or mutant) modules under test


def live_modules(src=None):
    src = src or SRC
    return {"rmc": load(src[RMC_PATH], "rmc_live", RMC_PATH),
            "ssl": load(src[SSL_PATH], "ssl_live", SSL_PATH),
            "sve": load(src[SVE_PATH], "sve_live", SVE_PATH),
            "woi_src": src[WOI_PATH], "wo_src": src.get(WO_PATH), "gate_src": src.get(GATE_PATH),
            "bay_src": src.get(BAY_PATH)}


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


RESIDUES = [101, 202, 303, 404, 505, 606, 707, 808]
COMMON = dict(dataset_path="/nonexistent/daily3-x.json", window_size=8,
              sessions=["evening"], window_anchor=25, residues=RESIDUES)
KW = dict(dataset_sha256="ab" * 32, forward_threshold=0.71, reverse_threshold=0.47,
          generator_phase=0)


def builder(rmc, phase, **extra):
    coord = rmc.RangeMinerCoordinator(rmc.CoordinatorConfig(), rmc.MinerLedger(":memory:"))
    return coord.build_stripe_assign_payload(
        COMMON["dataset_path"], COMMON["window_size"], COMMON["sessions"],
        COMMON["window_anchor"], COMMON["residues"], phase=phase, **KW, **extra)


def refused(fn, needle, exc=Exception):
    try:
        fn()
    except exc as e:
        assert needle in str(e), f"refused, but for {e!s:.300}"
        return
    raise AssertionError(f"not refused (wanted {needle!r})")


# --------------------------------------------------------------------------- OFF
def g_off_payload():
    rmc = MOD["rmc"]
    for ph in (1, 2, 3, 4):
        want = canon(builder(BASE_RMC, ph))
        assert canon(builder(rmc, ph)) == want, f"phase {ph}: off payload differs from HEAD"
        assert canon(builder(rmc, ph, skip_range=None)) == want, f"phase {ph}: explicit None differs"
        assert "skip_range" not in builder(rmc, ph)
    return "phases 1-4, switch off (absent or None): payload bytes identical to HEAD's builder"


class FakeFS:
    def __init__(self):
        self.sent = []

    def send_msg(self, msg):
        self.sent.append(msg)


def dispatch(rmc, run_id, phase, delivery=None):
    coord = rmc.RangeMinerCoordinator(rmc.CoordinatorConfig(), rmc.MinerLedger(":memory:"))
    if delivery is not None:
        coord._constant_skip_delivery = delivery
    stripe = {"stripe_id": f"{run_id}_s0", "state": rmc.ST_CLAIMED, "claimed_by": "w0",
              "current_attempt": 0, "seed_start": 0, "seed_count": 1000}
    coord.ledger = types.SimpleNamespace(all_stripes=lambda rid: [stripe] if rid == run_id else [])
    coord.record_assignment_threshold = lambda *a, **k: None
    fs = FakeFS()
    coord._dispatch_pending(run_id, "java_lcg", phase, {"w0": fs}, set(),
                            COMMON["dataset_path"], KW["dataset_sha256"], COMMON["window_size"],
                            COMMON["sessions"], COMMON["window_anchor"], 0, RESIDUES, 3,
                            KW["forward_threshold"], KW["reverse_threshold"])
    assert len(fs.sent) == 1, fs.sent
    return fs.sent[0].payload


def g_off_dispatch():
    for ph in (1, 2, 3, 4):
        assert canon(dispatch(MOD["rmc"], "r1", ph)) == canon(dispatch(BASE_RMC, "r1", ph)), ph
    return "phases 1-4, no delivery record: dispatched payloads identical to HEAD's"


def rtm(rmc, tag, **over):
    """run_trial_miner through its `_serve` seam: returns (coordinator, context)."""
    staging = os.path.join(OUT, tag, "staging")
    kw = dict(coordinator_cfg="stage.json", config=None, trial_number=3, prng_base="java_lcg",
              residues=RESIDUES, total_seeds=1000, forward_threshold=0.71,
              reverse_threshold=0.47, test_both_modes=True, dataset_path=COMMON["dataset_path"],
              skip_min=6, skip_max=99, window_size=8, window_anchor=25, generator_phase=0,
              seed_start=0, staging_dir=staging, staging_high_water_bytes=1024,
              _serve=lambda coordinator, context: {"coord": coordinator, "ctx": context,
                                                   "run_id": context["run_id"]})
    kw.update(over)
    with contextlib.redirect_stdout(io.StringIO()):
        return rmc.run_trial_miner(**kw), staging


def _ctx_view(ctx):
    """The serve context minus the per-call uuid run id and the one live measurement
    (free bytes on the staging filesystem, which other files change between calls)."""
    c = copy.deepcopy(dict(ctx))
    c.pop("run_id")
    c.get("staging_validation", {}).pop("available_bytes", None)
    c.get("staging_validation", {}).pop("headroom_bytes", None)
    return canon(c)


def g_off_serve_ctx():
    a, _ = rtm(BASE_RMC, "ctx_base")
    b, _ = rtm(MOD["rmc"], "ctx_live")
    assert _ctx_view(a["ctx"]).replace("ctx_base", "X") == _ctx_view(b["ctx"]).replace("ctx_live", "X"), \
        "serve context differs from HEAD's"
    assert not hasattr(b["coord"], "_constant_skip_delivery"), "a delivery record exists while off"
    assert set(b) == set(a) == {"coord", "ctx", "run_id"}, sorted(b)
    return "default off: the same serve context as HEAD, no delivery record, result untouched"


def _miner_kwargs(woi, deliver):
    rec = []
    saved = {k: getattr(woi, k) for k in ("run_trial_miner", "build_assembling_sink",
                                          "resolve_assembly_backend", "_miner_residues_for_config")}
    woi.run_trial_miner = lambda **kw: rec.append(kw) or {"pruned": True}
    woi.build_assembling_sink = lambda backend: types.SimpleNamespace(backend=backend)
    woi.resolve_assembly_backend = lambda name, **kw: f"backend:{name}"
    woi._miner_residues_for_config = lambda config, path: list(RESIDUES)
    coord = types.SimpleNamespace(use_range_miner=True, staging_dir="/nonexistent/staging")
    if deliver:
        coord.deliver_skip_range = True
    from window_optimizer import WindowConfig
    cfg = WindowConfig(window_size=8, offset=25, sessions=["evening"], skip_min=6, skip_max=99,
                       forward_threshold=0.71, reverse_threshold=0.47)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            woi.run_bidirectional_test(coord, cfg, "/nonexistent/daily3.json", seed_start=0,
                                       seed_count=1000, prng_base="java_lcg", test_both_modes=True,
                                       forward_threshold=0.71, reverse_threshold=0.47,
                                       trial_number=3, accumulator=None)
    finally:
        for k, v in saved.items():
            setattr(woi, k, v)
    assert len(rec) == 1, rec
    return rec[0]


def woi_mod(src=None):
    return load(src or MOD["woi_src"], "woi_live_sw", WOI_PATH)


def g_off_integ_call():
    global BASE_WOI
    if BASE_WOI is None:
        BASE_WOI = load(head("window_optimizer_integration_final.py"), "woi_head_sw", WOI_PATH)
    a, b = _miner_kwargs(BASE_WOI, False), _miner_kwargs(woi_mod(), False)
    assert canon(a) == canon(b), sorted(k for k in set(a) | set(b) if canon(a.get(k)) != canon(b.get(k)))
    assert "deliver_skip_range" not in b
    return f"switch off: run_trial_miner gets the same {len(b)} arguments as HEAD's integration"


BOUNDS = {"window_size": {"min": 2, "max": 50}}
SESS = [["midday"], ["evening"], ["midday", "evening"]]
PARAMS = {"prng_type": "java_lcg", "test_both_modes": True, "window_trials": 1, "max_seeds": 1000}


def g_off_search_id():
    ssl = MOD["ssl"]
    want = BASE_SSL.presented_configuration(dict(PARAMS), None, BOUNDS, SESS)
    for p in (dict(PARAMS), dict(PARAMS, deliver_skip_range=False)):
        got = ssl.presented_configuration(p, None, BOUNDS, SESS)
        assert canon(got) == canon(want), "presented configuration differs from HEAD's"
    return f"key absent or false: search_space_digest {want['search_space_digest'][:12]} = HEAD's"


GEOM = {"prng_type": "java_lcg", "window_size": 8, "window_anchor": 25, "generator_phase": 0,
        "sessions": ["evening"], "forward_threshold": 0.71, "reverse_threshold": 0.47,
        "skip_min": 6, "skip_max": 99, "dataset": "/nonexistent/daily3.json", "test_both_modes": True}


def g_off_estimator():
    sve = MOD["sve"]
    a = BASE_SVE.request_digest(BASE_SVE.validate_geometry(dict(GEOM)), 0, 1000)
    b = sve.request_digest(sve.validate_geometry(dict(GEOM)), 0, 1000)
    assert a == b
    pa, pb = estimator_payloads(BASE_SVE, dict(GEOM)), estimator_payloads(sve, dict(GEOM))
    assert canon(pa) == canon(pb), "estimator payloads differ from HEAD's"
    return "geometry without the key: the same request digest and sampled payloads as HEAD's"


def estimator_payloads(sve, g):
    """`_phase_payloads` imports its loader and digest from the miner modules at call
    time, so those two are replaced there (no dataset file is read)."""
    import miner.range_miner_coordinator as C
    import miner.range_miner_worker as W
    saved = (W.load_residue_window, C.resolve_dataset_sha256)
    W.load_residue_window = lambda *a, **k: list(RESIDUES)
    C.resolve_dataset_sha256 = lambda p: "ab" * 32
    try:
        out, _residues, _sha = sve._phase_payloads(sve.validate_geometry(g))
        return [(f, ph, p) for f, ph, p in out]
    finally:
        W.load_residue_window, C.resolve_dataset_sha256 = saved


def g_off_readers():
    readers = ["utils/canonical_records.py", "utils/run_finalizer.py", "miner/range_miner_npz_writer.py",
               "miner/step1_ingress.py", "convert_survivors_to_binary.py", "utils/canonical_arrays.py",
               "miner/assembly_backends.py", "miner/range_miner_worker.py"]
    for r in readers:
        txt = Path(ROOT, r).read_text(encoding="utf-8")
        assert "deliver_skip_range" not in txt and "_constant_skip_delivery" not in txt, r
        assert Path(ROOT, r).read_bytes() == subprocess.check_output(
            ["git", "-C", ROOT, "show", f"HEAD:{r}"]), f"{r} changed"
    return (f"{len(readers)} record/NPZ/finalizer/worker files: unchanged from HEAD and never "
            f"read the switch; the payload (identical off) is the only GPU input")


# --------------------------------------------------------------------------- ON
def g_on_payload():
    rmc = MOD["rmc"]
    for ph in (1, 2):
        off, on = builder(rmc, ph), builder(rmc, ph, skip_range=(6, 99))
        assert on.pop("skip_range") == [6, 99], ph
        assert canon(on) == canon(off), f"phase {ph}: anything but skip_range changed"
    for ph in (3, 4):
        assert canon(builder(rmc, ph, skip_range=(6, 99))) == canon(builder(BASE_RMC, ph)), \
            f"phase {ph}: the hybrid payload changed"
    for pair in ((0, 16), (3, 3), (0, 255)):
        assert builder(rmc, 1, skip_range=pair)["skip_range"] == list(pair)
    return "on: phases 1-2 add exactly skip_range = the pair; phases 3-4 identical to HEAD's"


def g_on_dispatch():
    rmc = MOD["rmc"]
    p = dispatch(rmc, "r1", 1, delivery={"r1": (6, 99)})
    assert p["skip_range"] == [6, 99], p.get("skip_range")
    assert "skip_range" not in dispatch(rmc, "r1", 1, delivery={"other": (6, 99)})
    assert "skip_range" not in dispatch(rmc, "r1", 3, delivery={"r1": (6, 99)})
    return "the run's pair reaches the phase 1 payload; another run's record, or phase 3, gets none"


def g_on_serve():
    rmc = MOD["rmc"]
    res, _ = rtm(rmc, "on_ok", deliver_skip_range=True)
    assert res["coord"]._constant_skip_delivery == {res["run_id"]: (6, 99)}
    assert res["constant_skip_delivery"] == {"requested": [6, 99], "effective": [6, 99], "phases": [1, 2]}
    for tag, over, needle in (("bad_hi", dict(skip_max=256), "outside"),
                              ("bad_order", dict(skip_min=50, skip_max=10), "outside"),
                              ("bad_none", dict(skip_min=None), "plain int"),
                              ("bad_flag", dict(deliver_skip_range=1), "must be a bool")):
        staging = os.path.join(OUT, f"on_{tag}", "staging")
        refused(lambda: rtm(rmc, f"on_{tag}", deliver_skip_range=over.pop("deliver_skip_range", True), **over),
                needle)
        assert not os.path.exists(staging), f"{tag}: something was created before the refusal"
    return ("on: the pair recorded for this run and on its result; out of range, wrong order, "
            "None, or a non-bool switch refused before the staging dir exists")


def worker_scalars(payload):
    import miner.range_miner_worker as W
    tree = ast.parse(Path(WRK_PATH).read_text(encoding="utf-8"))
    default = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
                and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "skip_range"):
            default = ast.literal_eval(n.args[1])
    assert default == [0, 16], default
    skip_min, skip_max = tuple(payload.get("skip_range", default))     # the worker's own line
    ctx = W.BuildContext(family_name="java_lcg", hybrid=False, reverse=False, seed_dtype="uint32",
                         n_seeds=1000, k=8, skip_min=skip_min, skip_max=skip_max, threshold=0.71,
                         generator_phase=0, params={})
    sc = [a for a in W._constant_prefix(ctx) if isinstance(a, W.ScalarArg)]
    return [(a.value, a.dtype) for a in sc[2:4]]


def g_on_worker_args():
    rmc = MOD["rmc"]
    assert worker_scalars(builder(rmc, 1)) == [(0, "int32"), (16, "int32")]
    assert worker_scalars(builder(rmc, 1, skip_range=(6, 99))) == [(6, "int32"), (99, "int32")]
    return "the worker's parse -> _constant_prefix: int32 skip scalars [0, 16] off, [6, 99] on"


def g_validation():
    rmc = MOD["rmc"]
    for pair in ((0, 0), (0, 255), (255, 255), (6, 99)):
        assert builder(rmc, 1, skip_range=pair)["skip_range"] == list(pair)
    bad = (((-1, 5), "outside"), ((5, 4), "outside"), ((0, 256), "outside"),
           ((True, 5), "plain int"), ((1.0, 5), "plain int"), (("1", 5), "plain int"))
    for i, (pair, needle) in enumerate(bad):
        refused(lambda: builder(rmc, 1, skip_range=pair), needle)          # the builder
        staging = os.path.join(OUT, f"val_{i}", "staging")                 # run_trial_miner
        refused(lambda: rtm(rmc, f"val_{i}", deliver_skip_range=True, skip_min=pair[0],
                            skip_max=pair[1]), needle)
        assert not os.path.exists(staging)
    return ("0..255, min <= max, plain ints: accepted at the bounds; every violation refused "
            "at both sites (builder and run_trial_miner, before anything exists)")


def g_search():
    woi = woi_mod()
    E = woi.Step1SearchPreflightError
    rec = lambda space: {"record": types.SimpleNamespace(search_id="s1", search_space=space)}
    base, dlv = rec({"kind": "pinned"}), rec({"kind": "pinned", "constant_skip_delivery": "trial_range"})
    woi._step1_skip_delivery_preflight(None, False)
    woi._step1_skip_delivery_preflight(base, False)
    woi._step1_skip_delivery_preflight(dlv, True)
    refused(lambda: woi._step1_skip_delivery_preflight(None, True), "needs a Step 1 search", E)
    refused(lambda: woi._step1_skip_delivery_preflight(base, True), "was opened without it", E)
    refused(lambda: woi._step1_skip_delivery_preflight(dlv, False), "may not add trials", E)
    ssl = MOD["ssl"]
    on = ssl.presented_configuration(dict(PARAMS, deliver_skip_range=True), None, BOUNDS_FULL, SESS)
    off = ssl.presented_configuration(dict(PARAMS), None, BOUNDS_FULL, SESS)
    refused(lambda: ssl.presented_configuration(dict(PARAMS, deliver_skip_range=True), None, BOUNDS,
                                                SESS), "plain int", ssl.Step1SearchRefused)
    assert on["search_space"]["constant_skip_delivery"] == "trial_range"
    assert on["search_space_digest"] != off["search_space_digest"]
    return ("on needs its own search (digest differs); on outside a search, on with a baseline "
            "search, or off with a switched-on search: refused before any study")


def g_on_integ_call():
    kw = _miner_kwargs(woi_mod(), True)
    assert kw.get("deliver_skip_range") is True, kw.get("deliver_skip_range")
    return "coordinator.deliver_skip_range=True: run_trial_miner gets deliver_skip_range=True"


def g_on_estimator():
    sve = MOD["sve"]
    pays = estimator_payloads(sve, dict(GEOM, deliver_skip_range=True))
    by_phase = {ph: p for _, ph, p in pays}
    assert by_phase[1]["skip_range"] == [6, 99] and by_phase[2]["skip_range"] == [6, 99]
    assert "skip_range" not in by_phase[3] and "skip_range" not in by_phase[4]
    refused(lambda: sve.validate_geometry(dict(GEOM, deliver_skip_range="yes")), "must be a bool")
    off = estimator_payloads(sve, dict(GEOM, deliver_skip_range=False))
    assert canon(off) == canon(estimator_payloads(BASE_SVE, dict(GEOM)))
    return "geometry deliver_skip_range=true: phases 1-2 sampled at [6, 99]; false = HEAD's"


WO_PATH = os.path.join(ROOT, "window_optimizer.py")
SRC[WO_PATH] = Path(WO_PATH).read_text(encoding="utf-8")


class _Captured(Exception):
    pass


def wo_settings(wo, **kw):
    """run_bayesian_optimization up to optimize_window (the Stage 3 suite's harness):
    the coordinator's attributes at hand-off."""
    seen = {}

    class Coord:
        def __init__(self, *a, **k):
            pass

        def optimize_window(self, **k):
            seen["attrs"] = dict(vars(self))
            raise _Captured()
    import window_optimizer_integration_final as WOI
    saved = (wo.MultiGPUCoordinator, wo.COORDINATOR_AVAILABLE, WOI.add_window_optimizer_to_coordinator)
    wo.MultiGPUCoordinator, wo.COORDINATOR_AVAILABLE = Coord, True
    WOI.add_window_optimizer_to_coordinator = lambda: None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                wo.run_bayesian_optimization(lottery_file="/nonexistent.json", trials=1,
                                             output_config="/dev/null", seed_count=1000,
                                             prng_type="java_lcg", test_both_modes=True,
                                             use_range_miner=True, n_parallel=1,
                                             staging_dir="/nonexistent/staging", **kw)
            except _Captured:
                pass
    finally:
        wo.MultiGPUCoordinator, wo.COORDINATOR_AVAILABLE, WOI.add_window_optimizer_to_coordinator = saved
    assert "attrs" in seen, "optimize_window was never reached"
    return seen["attrs"]


def g_wo_on():
    wo = load(MOD.get("wo_src") or SRC[WO_PATH], "wo_live_sw", WO_PATH)
    base = load(head("window_optimizer.py"), "wo_head_sw", WO_PATH)
    off, on, b = wo_settings(wo), wo_settings(wo, deliver_skip_range=True), wo_settings(base)
    assert off == b, "switch off: coordinator attributes differ from HEAD's"
    assert set(on) - set(off) == {"deliver_skip_range"} and on["deliver_skip_range"] is True, \
        sorted(set(on) ^ set(off))
    assert all(on[k] == off[k] for k in off)
    return "run_bayesian_optimization: off = HEAD's coordinator; on adds exactly deliver_skip_range=True"


def g_wo_cli():
    r = subprocess.run([sys.executable, WO_PATH, "--deliver-skip-range", "--lottery-file",
                        "/nonexistent.json", "--trials", "1"], capture_output=True, text=True,
                       timeout=120, cwd=ROOT)
    assert r.returncode == 2 and "applies to the RANGE-MINER route only" in r.stderr, \
        (r.returncode, r.stderr[-300:])
    r = subprocess.run([sys.executable, WO_PATH, "--help"], capture_output=True, text=True,
                       timeout=120, cwd=ROOT)
    assert "--deliver-skip-range" in r.stdout
    return "CLI: --deliver-skip-range without --use-range-miner exits 2 before any work; flag listed"




# --------------------------------------------------------------------------- R2 (Beta R1)
def g_q1_absent_false():
    """Beta R1 Q1: absent and explicit false must be the same bytes everywhere, the
    volume estimator's sample and the gate's evidence record included."""
    sve = MOD["sve"]
    ga = sve.validate_geometry(dict(GEOM))
    gf = sve.validate_geometry(dict(GEOM, deliver_skip_range=False))
    assert canon(ga) == canon(gf), "validated geometry differs (absent vs false)"
    da, df = sve.request_digest(ga, 0, 1000000), sve.request_digest(gf, 0, 1000000)
    assert da == df, "request digest differs"
    sa = sve.sample_intervals(0, 1000000, 1000, da)
    sf = sve.sample_intervals(0, 1000000, 1000, df)
    assert sa == sf, (sa[:1], sf[:1])
    assert canon(estimator_payloads(sve, dict(GEOM))) == canon(
        estimator_payloads(sve, dict(GEOM, deliver_skip_range=False)))
    gate = load(MOD.get("gate_src") or SRC[GATE_PATH], "svg_live_sw", GATE_PATH)
    gate.SVE = sve
    probe = lambda: {"status": "ABSENT", "detail": "test: no GPU"}       # stops before sampling
    clock = lambda: __import__("datetime").datetime(2026, 11, 21, tzinfo=__import__("datetime").timezone.utc)
    base_p = dict(GEOM, seed_start=0, seed_end_exclusive=1000000)
    ra = gate.evaluate(json.dumps(base_p), gpu_probe=probe, clock=clock, sample_seeds=10 ** 8)
    rf = gate.evaluate(json.dumps(dict(base_p, deliver_skip_range=False)), gpu_probe=probe,
                       clock=clock, sample_seeds=10 ** 8)
    assert canon(ra) == canon(rf), "gate evidence records differ (absent vs false)"
    return (f"absent vs false: same validated geometry, digest {da[:12]}, sample "
            f"{sa[0]}, payloads, and gate evidence record ({ra['verdict']})")


def g_q2_checkpoint():
    """Beta R1 Q2: the switch is part of the D6.2 run-context identity, so recovery
    refuses the other arm's checkpoint; off, the digest equals HEAD's."""
    global BASE_WOI
    if BASE_WOI is None:
        BASE_WOI = load(head("window_optimizer_integration_final.py"), "woi_head_sw", WOI_PATH)
    woi = woi_mod()

    def digest(mod, **kw):
        saved = (mod._checkpoint_dataset_identity, mod._repository_state,
                 mod._checkpoint_execution_set_id)
        mod._checkpoint_dataset_identity = lambda p: ("daily3-v1", "daily3-v1.json", "ab" * 32)
        mod._repository_state = lambda: ("0" * 40, True)
        mod._checkpoint_execution_set_id = lambda: None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ctx, _ = mod._prepare_checkpoint_run_context(
                    dataset_path="/nonexistent.json", prng_base="java_lcg",
                    skip_modes_executed=("constant", "variable"), seed_start=0, seed_count=1000,
                    resume_checkpoint="", resume_study=False, **kw)
            return ctx.run_context_digest
        finally:
            (mod._checkpoint_dataset_identity, mod._repository_state,
             mod._checkpoint_execution_set_id) = saved
    d_head, d_off, d_on = digest(BASE_WOI), digest(woi), digest(woi, skip_delivery_on=True)
    assert d_off == d_head, "switch off: run-context digest differs from HEAD's"
    assert d_on != d_off, "switch on: run-context digest does not bind the arm"
    import utils.checkpoint_d6_2 as CK
    ident = lambda d: {"run_id": "run-sw-1", "run_context_digest": d,
                       "encoding_version": CK.ENCODING_VERSION,
                       "canonical_map_hash": CK.canonical_map_hash(),
                       "checkpoint_schema_version": CK.CHECKPOINT_SCHEMA_VERSION}
    for have, want in ((d_off, d_on), (d_on, d_off)):
        refused(lambda: CK._require_context(ident(have), run_id="run-sw-1",
                                            run_context_digest=want, label="member A"),
                "run_context_digest", CK.CheckpointRecoveryError)
    CK._require_context(ident(d_on), run_id="run-sw-1", run_context_digest=d_on, label="A")
    return ("checkpoint identity: off digest = HEAD's; on differs; recovery's context check "
            "refuses the other arm's member in both directions, accepts its own")


def g_q2_study():
    """Beta R1 Q2: a switched search never adopts a legacy or baseline study, and a baseline
    run never resumes a switched one; refused before any study is opened for the run."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    woi = woi_mod()
    E = woi.Step1SearchPreflightError
    root = os.path.join(OUT, "studies")
    shutil.rmtree(root, ignore_errors=True)          # a gate run again (mutants) starts fresh
    os.makedirs(os.path.join(root, "optuna_studies"))
    url = lambda name: f"sqlite:///{root}/optuna_studies/{name}.db"
    woi._step1_storage_url = url
    legacy = optuna.create_study(study_name="window_opt_legacy", storage=url("window_opt_legacy"))
    legacy.add_trial(optuna.trial.create_trial(value=1.0, params={}, distributions={}))
    sw = optuna.create_study(study_name="window_opt_sw", storage=url("window_opt_sw"))
    sw.set_user_attr("constant_skip_delivery", "trial_range")
    dlv = {"record": types.SimpleNamespace(search_id="s1", search_space={"constant_skip_delivery": "trial_range"})}
    base = {"record": types.SimpleNamespace(search_id="s0", search_space={"kind": "pinned"})}
    cwd = os.getcwd()
    os.chdir(root)
    try:
        refused(lambda: woi._step1_skip_delivery_preflight(dlv, True, True, "window_opt_legacy"),
                "never adopts", E)
        refused(lambda: woi._step1_skip_delivery_preflight(base, False, True, "window_opt_sw"),
                "may not resume it", E)
        refused(lambda: woi._step1_skip_delivery_preflight(dlv, True, True, ""),
                "automatic selection", E)
        woi._step1_skip_delivery_preflight(dlv, True, True, "window_opt_sw")
        woi._step1_skip_delivery_preflight(base, False, True, "window_opt_legacy")
        woi._step1_skip_delivery_preflight(dlv, True, True, "window_opt_absent")   # fresh follows
    finally:
        os.chdir(cwd)
    # the first-trial backstop (covers automatic resume selection)
    fresh = optuna.create_study()
    woi._skip_delivery_study_check(fresh, True, 0)
    assert fresh.user_attrs.get("constant_skip_delivery") == "trial_range"
    old = optuna.create_study()
    old.add_trial(optuna.trial.create_trial(value=1.0, params={}, distributions={}))
    refused(lambda: woi._skip_delivery_study_check(old, True, 1), "another arm", E)
    refused(lambda: woi._skip_delivery_study_check(fresh, False, 1), "may not add trials", E)
    plain = optuna.create_study()
    woi._skip_delivery_study_check(plain, False, 0)
    assert plain.user_attrs == {}, "the baseline path wrote to the study"
    return ("study: legacy adopt, baseline resume of a switched study, and automatic selection "
            "refused before any study opens; backstop tags a fresh switched study, refuses an "
            "untagged one with trials and a baseline add to a switched one, writes nothing off")


def _fin(root, run):
    from utils.run_finalizer import finalize_run
    return finalize_run([], output_root=Path(root), run_id=run, prng_base="java_lcg",
                        skip_modes_executed=("constant",), seed_start=0, seed_count=1000,
                        repository_commit="0" * 40, repository_tree_clean=True)


def g_r3_q2_output_root():
    """Beta R3 Q2, reproduced with the REAL finalizer: a switched run publishes a
    generation and dies before its search link is durable (no link at all, no ledger).
    The root itself carries the arm, so a baseline run is refused at preflight AND at the
    pre-publication recheck; no SQLite is read, so no WAL can make the answer stale."""
    woi = woi_mod()
    E = woi.Step1SearchPreflightError
    arm = woi._step1_skip_delivery_root_arm
    sw, bl, em = (os.path.join(OUT, f"r3root_{k}") for k in ("sw", "bl", "em"))
    for d in (sw, bl, em):
        shutil.rmtree(d, ignore_errors=True)          # a gate run again (mutants) starts fresh
        os.makedirs(d)
    arm(True, sw, create=True)                        # the switched run's preflight marks its root
    _fin(sw, "step1_sw")                              # ... publishes; its link is never written
    assert not os.path.exists(os.path.join(sw, "prng_analysis.db"))
    _fin(bl, "step1_bl")                              # a baseline root with a current generation
    before = {d: sorted(os.listdir(os.path.join(d, ".s172_accumulator"))) for d in (sw, bl)}
    refused(lambda: arm(False, sw, create=True), "belongs to the switched-on arm", E)
    refused(lambda: arm(False, sw), "belongs to the switched-on arm", E)
    refused(lambda: arm(True, bl, create=True), "already holds a generation", E)
    refused(lambda: arm(True, bl), "already holds a generation", E)
    refused(lambda: arm(True, em), "is missing at publication", E)   # recheck never marks
    arm(True, sw)
    arm(True, sw, create=True)
    arm(False, bl)
    arm(False, bl, create=True)
    arm(False, em, create=True)
    assert os.listdir(em) == [], "the off path wrote into an empty root"
    assert before == {d: sorted(os.listdir(os.path.join(d, ".s172_accumulator"))) for d in (sw, bl)}
    # the marked root still finalizes (the finalizer ignores the marker) and keeps it
    _fin(sw, "step1_sw2")
    assert Path(sw, woi.SKIP_DELIVERY_ROOT_MARKER).read_text().strip() == woi.SKIP_DELIVERY_ON
    bad = os.path.join(OUT, "r3root_bad")
    shutil.rmtree(bad, ignore_errors=True)
    os.makedirs(os.path.join(bad, ".s172_accumulator"))
    Path(bad, woi.SKIP_DELIVERY_ROOT_MARKER).write_text("other\n")
    refused(lambda: arm(True, bad), "not 'trial_range'", E)
    # the recheck is the statement immediately before the finalizer call (an intervening
    # publication between preflight and finalize is caught there)
    src = MOD["woi_src"]
    k = src.index("        _artifact_d3_5 = _finalize_run_d3_5(")
    prev = src[:k].rstrip("\n").rsplit("\n", 1)[-1]
    assert prev == "        _step1_skip_delivery_root_arm(_s1_skip_delivery_on)", prev
    return ("output root (real finalizer, link never written): off refused on a switched root "
            "at preflight and recheck; on refused over a baseline generation and on an unmarked "
            "root at recheck; same-arm and empty roots allowed; off writes nothing; marked root "
            "still finalizes; recheck sits immediately before finalize_run")


def g_r3_q2_autoselect():
    """Beta R3 Q2, reproduced: an off run with unnamed --resume-study outside WATCHER.
    Automatic selection must skip a switched study BEFORE optimization allocates a trial,
    and still resume a baseline one exactly as before."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    bay = load(MOD.get("bay_src") or SRC[BAY_PATH], "bay_live_sw", BAY_PATH)
    root = os.path.join(OUT, "autoselect")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "optuna_studies"))
    bay._OPTUNA_STORAGE_BASE = root                  # never the live optuna_studies/
    url = lambda n: f"sqlite:///{root}/optuna_studies/{n}.db"
    import time as _t
    b = optuna.create_study(study_name="window_opt_base", storage=url("window_opt_base"))
    b.add_trial(optuna.trial.create_trial(value=1.0, params={}, distributions={}))
    _t.sleep(0.05)
    s = optuna.create_study(study_name="window_opt_sw", storage=url("window_opt_sw"))
    s.set_user_attr("constant_skip_delivery", "trial_range")
    s.add_trial(optuna.trial.create_trial(value=1.0, params={}, distributions={}))
    os.utime(url("window_opt_sw")[len("sqlite:///"):], None)   # the newest: picked first today

    class Stop(Exception):
        pass
    picked = []

    def _create(study_name=None, storage=None, load_if_exists=False, **kw):
        picked.append((study_name, load_if_exists))
        raise Stop()
    real = bay.optuna
    bay.optuna = types.SimpleNamespace(load_study=real.load_study, create_study=_create,
                                       pruners=real.pruners, samplers=real.samplers,
                                       trial=real.trial, logging=real.logging)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        srch = bay.OptunaBayesianSearch(n_startup_trials=1, seed=0)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                srch.run_optimization(lambda c: None, types.SimpleNamespace(), 5, None,
                            sampler=real.samplers.RandomSampler(seed=0),
                            sampler_metadata={"strategy": "t"}, resume_study=True, study_name="")
            except Stop:
                pass
    finally:
        os.chdir(cwd)
        bay.optuna = real
    assert picked == [("window_opt_base", True)], picked
    n_sw = len(optuna.load_study(study_name="window_opt_sw", storage=url("window_opt_sw")).trials)
    assert n_sw == 1, f"a trial was allocated in the switched study ({n_sw})"
    return ("automatic selection: the newer switched study is skipped and the baseline one "
            "resumed; no trial is allocated in the switched study")


def g_q4_validator():
    """Beta R1 Q4: ONE validator; every non-bool switch refused loudly at every entry
    point; out-of-range pins refused before a search is opened."""
    rmc, ssl, sve = MOD["rmc"], MOD["ssl"], MOD["sve"]
    v = rmc.validate_skip_delivery
    assert v(False, where="t") is False and v(True, where="t") is True
    assert v(False, -5, 999, where="t") is False                  # off: the pair is not examined
    for bad in (None, "", "true", "false", 1, 0, [], {}):
        refused(lambda: v(bad, where="t"), "must be a bool", rmc.MinerMetadataError)
        refused(lambda: ssl.presented_configuration(dict(PARAMS, deliver_skip_range=bad), None,
                                                    BOUNDS_FULL, SESS), "must be a bool",
                ssl.Step1SearchRefused)
        refused(lambda: sve.validate_geometry(dict(GEOM, deliver_skip_range=bad)), "must be a bool",
                sve.GeometryError)
        refused(lambda: rtm(rmc, "q4_flag", deliver_skip_range=bad), "must be a bool")
    pins = {"warm_start_window": 8, "warm_start_offset": 25, "warm_start_session_idx": 1,
            "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47,
            "warm_start_skip_min": 6, "warm_start_skip_max": 256}
    refused(lambda: ssl.presented_configuration(dict(PARAMS, deliver_skip_range=True), pins,
                                                BOUNDS_FULL, SESS), "outside", ssl.Step1SearchRefused)
    ssl.presented_configuration(dict(PARAMS), pins, BOUNDS_FULL, SESS)  # off: unchanged
    wide = dict(BOUNDS_FULL, skip_max={"min": 10, "max": 300})
    refused(lambda: ssl.presented_configuration(dict(PARAMS, deliver_skip_range=True), None,
                                                wide, SESS), "outside", ssl.Step1SearchRefused)
    refused(lambda: sve.validate_geometry(dict(GEOM, skip_max=256, deliver_skip_range=True)),
            "outside", sve.GeometryError)
    calls = []
    gate = load(MOD.get("gate_src") or SRC[GATE_PATH], "svg_live_sw2", GATE_PATH)
    gate.SVE = sve
    rec = gate.evaluate(json.dumps(dict(GEOM, skip_max=256, deliver_skip_range=True, seed_start=0,
                                        seed_end_exclusive=1000000)),
                        gpu_probe=lambda: calls.append(1) or {}, sample_seeds=10 ** 8)
    assert rec["verdict"] == "REFUSE" and not calls, (rec["verdict"], calls)
    woi = woi_mod()
    coord = types.SimpleNamespace(deliver_skip_range="true")
    refused(lambda: woi._skip_delivery_on(coord), "must be a bool", woi.Step1SearchPreflightError)
    return ("validator: None, '', 'true', 'false', 1, 0, [], {} refused at the validator, search "
            "open/S4, estimator and run_trial_miner; on pins [6, 256] and sampled bound 300 "
            "refused at search open; the gate refuses [6, 256] before its GPU probe")


def g_q4_cli():
    # in process, on the module under test (so a mutant is exercised), then as a program
    wo = load(MOD.get("wo_src") or SRC[WO_PATH], "wo_live_cli", WO_PATH)
    for args, needle in ((["--use-range-miner", "--deliver-skip-range", "--warm-start-skip-min", "6",
                           "--warm-start-skip-max", "256"], "outside"),
                         (["--use-range-miner", "--deliver-skip-range", "--config-file", "/nonexistent.json"],
                          "not carried by the --config-file route")):
        argv, err = sys.argv, io.StringIO()
        sys.argv = [WO_PATH, "--lottery-file", "/nonexistent.json", "--trials", "1"] + args
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                try:
                    wo.main()
                    code = 0
                except SystemExit as e:
                    code = e.code
        finally:
            sys.argv = argv
        assert code == 2 and needle in err.getvalue(), (args, code, err.getvalue()[-300:])
    for args, needle in ((["--use-range-miner", "--deliver-skip-range", "--warm-start-skip-min", "6",
                           "--warm-start-skip-max", "256"], "outside"),
                         (["--use-range-miner", "--deliver-skip-range", "--config-file", "/nonexistent.json"],
                          "not carried by the --config-file route")):
        r = subprocess.run([sys.executable, WO_PATH, "--lottery-file", "/nonexistent.json",
                            "--trials", "1"] + args, capture_output=True, text=True, timeout=120, cwd=ROOT)
        assert r.returncode == 2 and needle in r.stderr, (args, r.returncode, r.stderr[-300:])
    return "CLI: on with pin [6, 256], or on with --config-file: exit 2 before any work"


def g_r3_q4_sampling():
    """Beta R3 Q4, reproduced: a valid pin [6, 99], a three-trial budget, sampled skip_max
    256. Search open must refuse it; a one-trial pinned search does not sample and is
    accepted; off is unchanged. The child checks the sampling bounds it will use."""
    ssl = MOD["ssl"]
    pins = {"warm_start_window": 8, "warm_start_offset": 25, "warm_start_session_idx": 1,
            "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47,
            "warm_start_skip_min": 6, "warm_start_skip_max": 99}
    wide = dict(BOUNDS_FULL, skip_max={"min": 10, "max": 256})
    for n in (3, None):
        prm = dict(PARAMS, deliver_skip_range=True, window_trials=n)
        if n is None:
            del prm["window_trials"]
        refused(lambda: ssl.presented_configuration(prm, pins, wide, SESS), "sampling bounds",
                ssl.Step1SearchRefused)
    ssl.presented_configuration(dict(PARAMS, deliver_skip_range=True, window_trials=1), pins, wide, SESS)
    ssl.presented_configuration(dict(PARAMS, deliver_skip_range=True, window_trials=3), pins,
                                BOUNDS_FULL, SESS)                              # tracked 250
    for prm in (dict(PARAMS, window_trials=3), dict(PARAMS, window_trials=3, deliver_skip_range=False)):
        assert canon(ssl.presented_configuration(prm, pins, wide, SESS)) == \
            canon(BASE_SSL.presented_configuration(dict(PARAMS, window_trials=3), pins, wide, SESS))
    woi = woi_mod()
    import window_optimizer as _wo
    orig = _wo.SearchBounds.from_config
    try:
        _wo.SearchBounds.from_config = classmethod(lambda cls: types.SimpleNamespace(
            min_skip_min=0, max_skip_min=10, min_skip_max=10, max_skip_max=256))
        refused(lambda: woi._step1_skip_delivery_bounds_check(True), "outside",
                woi.Step1SearchPreflightError)
        woi._step1_skip_delivery_bounds_check(False)                  # off reads nothing
    finally:
        _wo.SearchBounds.from_config = orig
    woi._step1_skip_delivery_bounds_check(True)                        # the tracked config passes
    return ("pin [6, 99] with 3 trials (or no budget) and sampled max 256 refused at search "
            "open; 1 trial accepted; tracked 250 accepted; off digest = HEAD's; the child "
            "refuses sampled 256 and accepts the tracked bounds")


SKIP_BOUND_CASES = {     # (skip_min.min, skip_min.max, skip_max.min, skip_max.max) -> legal?
    "tracked": ((0, 10, 10, 250), True),
    "beta_r3_256": ((0, 10, 10, 256), False),
    "beta_sw2_order": ((0, 256, 10, 250), False),     # Beta X-MINER-SKIP-SWITCH-2 R1 Q4
    "order_in_range": ((0, 200, 10, 150), False),
    "min_range_inverted": ((5, 3, 10, 250), False),
    "edge_255": ((0, 255, 0, 255), True),
}


def _child_verdict(woi, case, pinned, trials, resume):
    import window_optimizer as _wo
    a, b, c, d = case
    orig = _wo.SearchBounds.from_config
    obj = types.SimpleNamespace(min_skip_min=a, max_skip_min=b, min_skip_max=c, max_skip_max=d)
    try:
        _wo.SearchBounds.from_config = classmethod(lambda cls: obj)
        got = woi._step1_skip_delivery_bounds_check(True, pinned=pinned, trials=trials,
                                                    resume_study=resume)
        assert got is obj, "the child did not return the bounds object it checked"
        return True
    except woi.Step1SearchPreflightError:
        return False
    finally:
        _wo.SearchBounds.from_config = orig


def _open_verdict(ssl, case, pinned, trials, resume):
    a, b, c, d = case
    bnd = dict(BOUNDS_FULL, skip_min={"min": a, "max": b}, skip_max={"min": c, "max": d})
    pins = {"warm_start_window": 8, "warm_start_offset": 25, "warm_start_session_idx": 1,
            "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47,
            "warm_start_skip_min": 6, "warm_start_skip_max": 99} if pinned else None
    # under WATCHER the effective resume flag is bool(study_name) (G-SW3-RESUME-RULE)
    prm = dict(PARAMS, deliver_skip_range=True, study_name="window_opt_named" if resume else "")
    if trials is None:
        prm.pop("window_trials", None)
    else:
        prm["window_trials"] = trials
    try:
        ssl.presented_configuration(prm, pins, bnd, SESS)
        return True
    except ssl.Step1SearchRefused:
        return False


def g_sw2_q4_domains():
    """Beta X-MINER-SKIP-SWITCH-2 R1 Q4, reproduced: skip_min.max=256 with skip_max.max=250
    passed both preflights and failed inside Optuna. Now the FULL domains and
    skip_min.max <= skip_max.max are checked, by ONE rule at search open and in the child,
    and the two agree on every case: bounds x pinned x trials x resume."""
    ssl, woi = MOD["ssl"], woi_mod()
    n = 0
    for name, (case, legal) in SKIP_BOUND_CASES.items():
        for pinned in (False, True):
            for trials in (1, 3, None):
                for resume in (False, True):
                    samples = not (pinned and trials == 1 and not resume)
                    want = legal or not samples
                    o = _open_verdict(ssl, case, pinned, trials, resume)
                    c = _child_verdict(woi, case, pinned, trials, resume)
                    assert o == c == want, (name, pinned, trials, resume, "open", o, "child", c,
                                            "want", want)
                    n += 1
    # the child runs BEFORE the named-study check, and the sampler uses the checked object
    src = MOD["woi_src"]
    assert src.index("_s1_skip_bounds = _step1_skip_delivery_bounds_check(") < \
        src.index("        _step1_skip_delivery_preflight(_s1_search_ctx, _s1_skip_delivery_on,")
    assert "bounds = _s1_skip_bounds if _s1_skip_bounds is not None else SearchBounds.from_config()" in src
    assert woi._step1_skip_delivery_bounds_check(False) is None          # off: nothing loaded
    return (f"{n} cases: search open and child agree; Beta's skip_min.max=256 / skip_max.max=250 "
            f"refused wherever the run can sample; a fresh 1-trial pinned run exempt, a resumed one "
            f"not; the checked object reaches the sampler; check precedes the study load")


def g_sw2_q2_scope():
    """Beta X-MINER-SKIP-SWITCH-2 R1 Q2, reproduced: two publishers overlapping in one
    unmarked root. Owner path: the isolation claim is STATED as separate roots and one
    publisher at a time (no lock). The overlap still passes the guard (that is the known
    limit); the switched run's banner and every refusal say the limit; the serial orders
    are refused."""
    woi = woi_mod()
    E = woi.Step1SearchPreflightError
    arm = woi._step1_skip_delivery_root_arm
    root = os.path.join(OUT, "sw2_overlap")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    arm(False, root, create=True)                    # baseline preflight in an empty root
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        arm(True, root, create=True)                 # switched preflight marks it
    banner = out.getvalue()
    for need in ("ONE publisher at a time", "separate output root per arm",
                 "backup or restore", woi.SKIP_DELIVERY_ROOT_MARKER):
        assert need in banner, (need, banner)
    arm(True, root)                                  # switched recheck passes
    # the known limit: the baseline already passed its checks before the marker existed;
    # its LATER recheck is refused, but a recheck taken before the mark is not repeated
    refused(lambda: arm(False, root), "one publisher at a time", E)
    _fin(root, "step1_sw2_on")
    bl = os.path.join(OUT, "sw2_serial_bl")
    shutil.rmtree(bl, ignore_errors=True)
    os.makedirs(bl)
    _fin(bl, "step1_sw2_bl")
    refused(lambda: arm(True, bl, create=True), "one publisher at a time", E)
    doc = " ".join(woi._step1_skip_delivery_root_arm.__doc__.split())
    for need in ("ONE publisher at a time", "NOT isolated", "restore of the root must carry"):
        assert need in doc, ("docstring", need)
    off = io.StringIO()
    with contextlib.redirect_stdout(off):
        arm(False, bl)
    assert off.getvalue() == "", "the off path printed"
    return ("overlap: off passed before the mark (the stated limit); the banner, docstring and "
            "both refusals state separate roots, one publisher at a time and the marker in "
            "backup/restore; the serial orders are refused; off prints nothing")


def _cand_row(seed):
    """One canonical 24-field constant record (tests/test_s172_phase5_d3_5_finalizer.py cand)."""
    return {"seed": seed, "forward_match_rate": 0.9, "reverse_match_rate": 0.9, "score": 0.9,
            "window_size": 12, "offset": 3, "skip_min": 1, "skip_max": 4, "skip_range": 3,
            "sessions": ["midday", "evening"], "trial_number": 1, "prng_base": "java_lcg",
            "skip_mode": "constant", "prng_type": "java_lcg", "forward_count": 9.0,
            "reverse_count": 7.0, "bidirectional_count": 4.0, "intersection_count": 4.0,
            "intersection_ratio": 0.25, "forward_only_count": 5.0, "reverse_only_count": 3.0,
            "survivor_overlap_ratio": 0.44, "bidirectional_selectivity": 1.2857,
            "intersection_weight": 0.25}


def _fin_rows(root, run, rows):
    from utils.run_finalizer import finalize_run
    return finalize_run(rows, output_root=Path(root), run_id=run, prng_base="java_lcg",
                        skip_modes_executed=("constant",), seed_start=0, seed_count=1000,
                        repository_commit="0" * 40, repository_tree_clean=True)


def g_sw3_q2_overlap():
    """Beta X-MINER-SKIP-SWITCH-2 R2 Tests suggestion: the same-root overlap, in BOTH
    publication orders, with both finalizer calls; the second publisher names the first
    as its parent and carries its row. This is the STATED limit (one publisher at a time),
    asserted so that the limit cannot silently change."""
    woi = woi_mod()
    arm = woi._step1_skip_delivery_root_arm
    out = []
    for order in ("switched_first", "baseline_first"):
        root = os.path.join(OUT, f"sw3_overlap_{order}")
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root)
        arm(False, root, create=True)                   # baseline preflight, empty root
        arm(False, root)                                # baseline publication recheck
        with contextlib.redirect_stdout(io.StringIO()):
            arm(True, root, create=True)                # switched preflight marks the root
        if order == "switched_first":
            arm(True, root)
            first = _fin_rows(root, "step1_on", [_cand_row(5)])
            second = _fin_rows(root, "step1_off", [])  # baseline already passed its recheck
        else:
            first = _fin_rows(root, "step1_off", [_cand_row(7)])
            arm(True, root)                             # marker present: passes
            second = _fin_rows(root, "step1_on", [])
        assert second.parent_generation_id == first.generation_id, (order, second.parent_generation_id)
        assert first.final_row_count == 1 and second.final_row_count == 1, \
            (order, first.final_row_count, second.final_row_count)
        # the second generation's arrays ARE the first's row (Beta R3 Tests suggestion)
        import numpy as np
        for npz in ("all_npz_path", "binary_npz_path"):
            with np.load(getattr(first, npz), allow_pickle=False) as a, \
                    np.load(getattr(second, npz), allow_pickle=False) as b:
                assert sorted(a.files) == sorted(b.files), (order, npz)
                for k in a.files:
                    assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), (order, npz, k)
        out.append(order)
    return ("same-root overlap, both orders: the second generation's parent is the first and it "
            "carries the first's one row with zero fresh rows (the stated limit)")


class _Res:
    def __init__(self, config):
        self.config, self.iteration = config, 0
        self.bidirectional_survivors, self.bidirectional_count = [], 0
        self.forward_count = self.reverse_count = 0

    def to_dict(self):
        return {"skip_min": self.config.skip_min, "skip_max": self.config.skip_max}


def _run_study(bay, ctx, trials, tag, bounds=None):
    """The REAL serial Optuna study body (run_optimization) with a stub sieve objective.
    Storage is redirected to scratch (never the live optuna_studies/)."""
    import optuna
    import window_optimizer as _wo
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    root = os.path.join(OUT, f"study_{tag}")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(os.path.join(root, "optuna_studies"))
    bay._OPTUNA_STORAGE_BASE = root
    seen = []
    # the TRACKED config, by explicit path, before any chdir (Beta X-MINER-SKIP-SWITCH-2 R3 Tests)
    bounds = bounds or _wo.SearchBounds.from_config(os.path.join(ROOT, "distributed_config.json"))

    def objective(config, optuna_trial=None):
        seen.append((config.skip_min, config.skip_max))
        return _Res(config)
    scorer = types.SimpleNamespace(score=lambda r: 1.0)
    cwd = os.getcwd()
    os.chdir(root)
    err = None
    try:
        srch = bay.OptunaBayesianSearch(n_startup_trials=1, seed=0)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                srch.run_optimization(objective, bounds, trials, scorer,
                                      sampler=optuna.samplers.RandomSampler(seed=0),
                                      sampler_metadata={"strategy": "t"},
                                      trse_context_file=os.path.join(root, "no_trse.json"),
                                      trial_history_context=ctx)
            except Exception as e:                                    # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
    finally:
        os.chdir(cwd)
    trials_seen = []
    store = f"sqlite:///{root}/optuna_studies/"
    for db in sorted(Path(root, "optuna_studies").glob("*.db")):
        for summ in optuna.get_all_study_summaries(storage=f"{store}{db.name}"):
            st = optuna.load_study(study_name=summ.study_name, storage=f"{store}{db.name}")
            for t in st.trials:
                d = {k: (t.distributions[k].low, t.distributions[k].high)
                     for k in ("skip_min", "skip_max") if k in t.distributions}
                trials_seen.append((t.number, t.state.name,
                                    {k: t.params.get(k) for k in ("skip_min", "skip_max")}, d))
    return seen, err, trials_seen


def _pin_ctx(lo, hi, exact=False):
    ctx = {"warm_start_window": 8, "warm_start_offset": 25, "warm_start_skip_min": lo,
           "warm_start_skip_max": hi, "warm_start_fwd_thresh": 0.71,
           "warm_start_rev_thresh": 0.47, "warm_start_session_idx": 1}
    if exact:
        ctx["skip_delivery_exact_pin"] = [lo, hi]
    return ctx


def g_sw3_q4_exact_pin():
    """Beta X-MINER-SKIP-SWITCH-2 R2 Q4 LIVE, reproduced through a REAL Optuna trial: a legal
    switched one-trial pin [251, 255] under the tracked bounds made suggest_int('skip_max',
    251, 250) raise before the objective. The child now hands the exact pin to the study
    body, which draws single-point distributions: [251, 255] and the control [6, 99] both
    reach the objective exactly. Off (no key) is HEAD's study body, byte for byte."""
    bay = load(MOD.get("bay_src") or SRC[BAY_PATH], "bay_live_pin", BAY_PATH)
    bay_h = load(head("window_optimizer_bayesian.py"), "bay_head_pin", BAY_PATH)
    for lo, hi in ((251, 255), (6, 99), (0, 255), (0, 0), (255, 255)):
        seen, err, tr = _run_study(bay, _pin_ctx(lo, hi, exact=True), 1, f"on_{lo}_{hi}")
        assert seen == [(lo, hi)] and err is None, (lo, hi, seen, err)
        assert tr == [(0, "COMPLETE", {"skip_min": lo, "skip_max": hi},
                       {"skip_min": (lo, lo), "skip_max": (hi, hi)})], (lo, hi, tr)
    # R2's inverted skip_min domain [5, 3]: the exact pin completes; without it, empty range
    import dataclasses
    import window_optimizer as _wo
    inv = dataclasses.replace(_wo.SearchBounds.from_config(os.path.join(ROOT, "distributed_config.json")),
                              min_skip_min=5, max_skip_min=3)
    seen, err, tr = _run_study(bay, _pin_ctx(6, 99, exact=True), 1, "on_inverted", bounds=inv)
    assert seen == [(6, 99)] and err is None and tr[0][1] == "COMPLETE", (seen, err, tr)
    seen, err, _ = _run_study(bay, _pin_ctx(6, 99), 1, "off_inverted",
                              bounds=dataclasses.replace(inv))
    assert seen == [] and err and "ValueError" in err, (seen, err)
    # the reproduction itself: without the exact pin the same legal pin fails, live AND HEAD
    for b, tag in ((bay, "live"), (bay_h, "head")):
        seen, err, _ = _run_study(b, _pin_ctx(251, 255), 1, f"off251_{tag}")
        assert seen == [] and err and "ValueError" in err, (tag, seen, err)
    # off: same trials as HEAD's study body (pinned control, and an unpinned seeded run)
    for ctx, n, tag in ((_pin_ctx(6, 99), 1, "pin"), (None, 3, "free")):
        a = _run_study(bay, ctx, n, f"offcmp_live_{tag}")
        b = _run_study(bay_h, ctx, n, f"offcmp_head_{tag}")
        assert a == b and a[1] is None, (tag, a, b)
    # trial 0 only: an exact pin on a multi-trial study fails closed at trial 1
    seen, err, _ = _run_study(bay, _pin_ctx(6, 99, exact=True), 2, "on_multi")
    assert seen == [(6, 99)] and err and "trial 0 only" in err, (seen, err)
    # the child's rule: only a switched, fresh, one-trial, fully pinned run gets the key
    woi = woi_mod()
    xp = woi._step1_skip_delivery_exact_pin
    assert xp(False, True, 1, False, 251, 255) is None
    assert xp(True, True, 1, False, 251, 255) == [251, 255]
    for args in ((True, False, 1, False), (True, True, 3, False), (True, True, None, False),
                 (True, True, 1, True)):
        assert xp(*args, 251, 255) is None, args
    refused(lambda: xp(True, True, 1, False, 251, 256), "outside", woi.Step1SearchPreflightError)
    src = MOD["woi_src"]
    assert ("        if _s1_skip_exact_pin is not None:\n"
            "            # [SKIP-DELIVERY SWITCH] only a switched run that cannot sample carries it\n"
            "            _trial_history_ctx['skip_delivery_exact_pin'] = _s1_skip_exact_pin\n") in src
    return ("real Optuna: switched exact pins [251, 255], [6, 99], [0, 255] reach the objective "
            "exactly; without the key [251, 255] fails live and at HEAD (the reproduction); off "
            "runs (pinned and unpinned) equal HEAD's; an exact pin past trial 0 fails closed; the "
            "child sets the key only for a switched fresh one-trial full pin")


def g_sw3_resume_rule():
    """Beta X-MINER-SKIP-SWITCH-2 R2 Q4 THEORETICAL, reproduced: WATCHER sets resume_study
    from study_name AFTER search open, so search open must read the effective flag. Both of
    Beta's directions now agree with the child; WATCHER's rule is pinned so drift reds here
    (agents/watcher_agent.py is owner-landed and not edited)."""
    w = Path(ROOT, "agents", "watcher_agent.py").read_text(encoding="utf-8")
    rule = ("            _explicit_study = final_params.get('study_name', '')\n"
            "            if not _explicit_study:\n"
            "                final_params['resume_study'] = False\n"
            "                final_params['study_name'] = ''\n"
            "            else:\n"
            "                final_params['resume_study'] = True\n")
    assert w.count(rule) == 1, "WATCHER's study-continuity rule changed; re-derive search open's flag"
    ssl, woi = MOD["ssl"], woi_mod()
    pins = {"warm_start_window": 8, "warm_start_offset": 25, "warm_start_session_idx": 1,
            "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47,
            "warm_start_skip_min": 6, "warm_start_skip_max": 99}
    bnd = dict(BOUNDS_FULL, skip_max={"min": 10, "max": 256})
    case = (0, 10, 10, 256)
    for name, resume_param, watcher_resume in (("window_opt_named", False, True), ("", True, False)):
        prm = dict(PARAMS, deliver_skip_range=True, window_trials=1, study_name=name,
                   resume_study=resume_param)
        try:
            ssl.presented_configuration(prm, pins, bnd, SESS)
            o = True
        except ssl.Step1SearchRefused:
            o = False
        c = _child_verdict(woi, case, True, 1, watcher_resume)
        assert o == c == (not watcher_resume), (name, resume_param, o, c)
    return ("WATCHER's rule pinned; named study + resume_study=false and empty name + "
            "resume_study=true: search open and the child give the same verdict")


def g_r3_q4_launcher():
    """Beta R3 Q4, reproduced: None, "true" and 1 passed to run_bayesian_optimization were
    silently off. They are refused at entry, before any coordinator exists; False and the
    default take exactly HEAD's path (same outcome as HEAD on the same arguments)."""
    rmc = MOD["rmc"]
    wo = load(MOD.get("wo_src") or SRC[WO_PATH], "wo_live_entry", WO_PATH)
    wo_h = load(head("window_optimizer.py"), "wo_head_entry", WO_PATH)
    made = []

    class NoCoordinator(Exception):
        pass

    def _no_coord(*a, **k):                       # never a real coordinator, in either module
        made.append(1)
        raise NoCoordinator()
    wo.MultiGPUCoordinator = _no_coord
    wo_h.MultiGPUCoordinator = _no_coord
    args = dict(lottery_file="/nonexistent-r3.json", trials=1, output_config=os.path.join(OUT, "r3.json"))

    def outcome(f, **kw):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                f(**args, **kw)
            return "returned"
        except BaseException as e:                                      # noqa: BLE001
            return f"{type(e).__name__}: {str(e)[:200]}"
    for bad in (None, "true", 1, 0, ""):
        refused(lambda: wo.run_bayesian_optimization(**args, deliver_skip_range=bad),
                "must be a bool")
    assert not made, "a coordinator was created before the keyword was validated"
    for kw in ({}, {"deliver_skip_range": False}):
        a, b = outcome(wo.run_bayesian_optimization, **kw), outcome(wo_h.run_bayesian_optimization)
        assert a == b, (kw, a, b)
    assert made, "the comparison never reached coordinator creation"
    return "None, 'true', 1, 0, '' refused at entry; default and False reach HEAD's outcome"


def g_finding4_kwonly():
    import inspect
    rmc = MOD["rmc"]
    wo = load(MOD.get("wo_src") or SRC[WO_PATH], "wo_live_kw", WO_PATH)
    wo_h = load(head("window_optimizer.py"), "wo_head_kw", WO_PATH)
    for live_f, head_f in ((rmc.run_trial_miner, BASE_RMC.run_trial_miner),
                           (wo.run_bayesian_optimization, wo_h.run_bayesian_optimization)):
        sig = inspect.signature(live_f)
        assert sig.parameters["deliver_skip_range"].kind is inspect.Parameter.KEYWORD_ONLY
        pos = lambda f: [n for n, p in inspect.signature(f).parameters.items()
                         if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        assert pos(live_f) == pos(head_f), f"{live_f.__name__}: positional slots moved"
    return "deliver_skip_range is keyword-only; both functions' positional slots equal HEAD's"


BOUNDS_FULL = {"skip_min": {"min": 0, "max": 10}, "skip_max": {"min": 10, "max": 250},
               "window_size": {"min": 2, "max": 50}}


GATES = [("G-OFF-PAYLOAD", g_off_payload), ("G-OFF-DISPATCH", g_off_dispatch),
         ("G-OFF-SERVE-CTX", g_off_serve_ctx), ("G-OFF-INTEG-CALL", g_off_integ_call),
         ("G-OFF-SEARCH-ID", g_off_search_id), ("G-OFF-ESTIMATOR", g_off_estimator),
         ("G-OFF-READERS", g_off_readers), ("G-ON-PAYLOAD", g_on_payload),
         ("G-ON-DISPATCH", g_on_dispatch), ("G-ON-SERVE", g_on_serve),
         ("G-ON-WORKER-ARGS", g_on_worker_args), ("G-VALIDATION", g_validation),
         ("G-SEARCH", g_search), ("G-ON-INTEG-CALL", g_on_integ_call),
         ("G-ON-ESTIMATOR", g_on_estimator), ("G-WO-ON", g_wo_on), ("G-WO-CLI", g_wo_cli),
         ("G-Q1-ABSENT-FALSE", g_q1_absent_false), ("G-Q2-CHECKPOINT", g_q2_checkpoint),
         ("G-Q2-STUDY", g_q2_study), ("G-R3-Q2-OUTPUT-ROOT", g_r3_q2_output_root),
         ("G-R3-Q2-AUTOSELECT", g_r3_q2_autoselect), ("G-R3-Q4-SAMPLING", g_r3_q4_sampling),
         ("G-R3-Q4-LAUNCHER", g_r3_q4_launcher), ("G-SW2-Q4-DOMAINS", g_sw2_q4_domains),
         ("G-SW2-Q2-SCOPE", g_sw2_q2_scope), ("G-SW3-Q2-OVERLAP", g_sw3_q2_overlap),
         ("G-SW3-Q4-EXACT-PIN", g_sw3_q4_exact_pin), ("G-SW3-RESUME-RULE", g_sw3_resume_rule),
         ("G-Q4-VALIDATOR", g_q4_validator), ("G-Q4-CLI", g_q4_cli),
         ("G-F4-KWONLY", g_finding4_kwonly)]

R, W_, L, V, O = RMC_PATH, WOI_PATH, SSL_PATH, SVE_PATH, WO_PATH
MUTANTS = [
    ("MS1 key added while off", R, 'if skip_range is not None and _skip_mode == "constant":',
     'if _skip_mode == "constant" and skip_range is not False:', g_off_payload),
    ("MS2 key added on hybrid phases", R, 'if skip_range is not None and _skip_mode == "constant":',
     "if skip_range is not None:", g_on_payload),
    ("MS3 bounds swapped", R, 'payload["skip_range"] = [int(lo), int(hi)]', 'payload["skip_range"] = [int(hi), int(lo)]',
     g_on_payload),
    ("MS4 the 255 bound dropped", R, "if not 0 <= skip_min <= skip_max <= CONSTANT_SKIP_DELIVERY_MAX:",
     "if not 0 <= skip_min <= skip_max:", g_validation),
    ("MS5 bool accepted as int", R, "        if isinstance(v, bool) or not isinstance(v, int):",
     "        if not isinstance(v, int):", g_validation),
    ("MS5b a non-bool switch accepted", R, "    if value is not True and value is not False:",
     "    if value is None:", g_q4_validator),
    ("MS6 dispatch does not hand the pair", R,
     'skip_range=(getattr(self, "_constant_skip_delivery", None) or {}).get(run_id))',
     "skip_range=None)", g_on_dispatch),
    ("MS7 run_trial_miner records nothing", R,
     "coordinator._constant_skip_delivery = {run_id: _skip_delivery}", "pass", g_on_serve),
    ("MS8 run_trial_miner bypasses the validator", R,
     "    if validate_skip_delivery(deliver_skip_range, skip_min, skip_max,\n                              where=\"run_trial_miner\"):",
     "    if deliver_skip_range:", g_on_serve),
    ("MS9 a delivery record exists while off", R,
     "    if _skip_delivery is not None:\n        # read by _dispatch_pending",
     "    if True:\n        # read by _dispatch_pending", g_off_serve_ctx),
    ("MS10 search key dropped", L, '        space["constant_skip_delivery"] = "trial_range"',
     "        pass", g_search),
    ("MS11 search key always present", L, "    if _sd_on:\n        # Only when on",
     "    if True:\n        # Only when on", g_off_search_id),
    ("MS12 baseline may use a switched-on search", W_, "    if not deliver_on and recorded is not None:",
     "    if False:", g_search),
    ("MS13 switched-on run outside a search allowed", W_, "    if deliver_on and search_ctx is None:",
     "    if False:", g_search),
    ("MS14 integration always passes the switch", W_,
     '**({"deliver_skip_range": True} if _skip_delivery_on(coordinator) else {}),',
     '**({"deliver_skip_range": True} if True else {}),', g_off_integ_call),
    ("MS15 integration never passes the switch", W_,
     '**({"deliver_skip_range": True} if _skip_delivery_on(coordinator) else {}),',
     '**({"deliver_skip_range": True} if False else {}),', g_on_integ_call),
    ("MS16 estimator ignores the switch", V,
     'if g.get("deliver_skip_range") is True else {}))', "if False else {}))", g_on_estimator),
    ("MS17 estimator rejects a baseline-free key check", V,
     '    extra = sorted(set(geometry) - set(GEOMETRY_KEYS) - set(OPTIONAL_GEOMETRY_KEYS))',
     '    extra = sorted(set(geometry) - set(GEOMETRY_KEYS))', g_on_estimator),
    ("MS18 coordinator attribute set while off", O,
     "    if deliver_skip_range is True:\n        # [SKIP-DELIVERY SWITCH] set ONLY when on",
     "    coordinator.deliver_skip_range = deliver_skip_range\n    if deliver_skip_range is True:\n        # [SKIP-DELIVERY SWITCH] set ONLY when on",
     g_wo_on),
    ("MS20 explicit false kept in the estimator geometry", V,
     "        if not on:\n            del g[\"deliver_skip_range\"]", "        pass", g_q1_absent_false),
    ("MS21 explicit false kept in the gate record", GATE_PATH,
     "        params = {k: v for k, v in params.items() if k != \"deliver_skip_range\"}",
     "        params = dict(params)", g_q1_absent_false),
    ("MS22 checkpoint identity ignores the switch", W_,
     "        _components = dict(_components, **{SKIP_DELIVERY_CHECKPOINT_KEY: SKIP_DELIVERY_ON})",
     "        pass", g_q2_checkpoint),
    ("MS23 a switched search adopts a legacy study", W_,
     "        if deliver_on and tag != SKIP_DELIVERY_ON:\n            raise Step1SearchPreflightError(\n                f\"--deliver-skip-range: study",
     "        if False:\n            raise Step1SearchPreflightError(\n                f\"--deliver-skip-range: study", g_q2_study),
    ("MS24 baseline resumes a switched study", W_,
     "        if not deliver_on and tag is not None:\n            raise Step1SearchPreflightError(\n                f\"study {study_name!r}",
     "        if False:\n            raise Step1SearchPreflightError(\n                f\"study {study_name!r}", g_q2_study),
    ("MS25 backstop tags a study with earlier trials", W_,
     "        if tag is None and not earlier:", "        if tag is None:", g_q2_study),
    ("MS26 a baseline run accepted on a switched root", W_,
     "    if not deliver_on:\n        if os.path.lexists(marker):", "    if not deliver_on:\n        if False:",
     g_r3_q2_output_root),
    ("MS30 a switched run accepted over a baseline generation", W_,
     "    if os.path.lexists(current):\n        raise", "    if False:\n        raise", g_r3_q2_output_root),
    ("MS31 the pre-publication recheck marks an unmarked root", W_,
     "    if not create:\n        raise", "    if False:\n        raise", g_r3_q2_output_root),
    ("MS32 the recheck before finalize_run removed", W_,
     "        _step1_skip_delivery_root_arm(_s1_skip_delivery_on)\n        _artifact_d3_5",
     "        _artifact_d3_5", g_r3_q2_output_root),
    ("MS33 automatic selection picks a switched study", BAY_PATH,
     '                    if not study_name and (_candidate_study.user_attrs or {}).get(',
     '                    if False and (_candidate_study.user_attrs or {}).get(', g_r3_q2_autoselect),
    ("MS34 search open skips the sampling domains", L,
     "        if _sd_on and skip_delivery_can_sample(", "        if False and skip_delivery_can_sample(",
     g_r3_q4_sampling),
    ("MS35 the child skips the sampling domains", W_,
     "    if skip_delivery_can_sample(pinned, trials, resume_study):\n        try:",
     "    if False:\n        try:", g_r3_q4_sampling),
    ("MS37 skip_min.max <= skip_max.max dropped", L,
     "    validate_skip_delivery(True, max_skip_min, max_skip_max,", "    (lambda *a, **k: None)(True, max_skip_min, max_skip_max,",
     g_sw2_q4_domains),
    ("MS38 a resumed pinned run treated as unable to sample", L,
     "int(trials) == 1 and not resume_study)", "int(trials) == 1)", g_sw2_q4_domains),
    ("MS39 search open checks only the extremes again", L,
     "            validate_skip_sampling_domains(_smin.get(\"min\"), _smin.get(\"max\"), _smax.get(\"min\"),\n                                           _smax.get(\"max\"),",
     "            validate_skip_sampling_domains(_smin.get(\"min\"), 0, 0,\n                                           _smax.get(\"max\"),",
     g_sw2_q4_domains),
    ("MS40 the sampler reloads unchecked bounds", W_,
     "            bounds = _s1_skip_bounds if _s1_skip_bounds is not None else SearchBounds.from_config()",
     "            bounds = SearchBounds.from_config()", g_sw2_q4_domains),
    ("MS41 the child's check after the study load", W_,
     '        # the bounds first: before the named-study check loads any store\n        _s1_pinned = None not in (warm_start_window, warm_start_offset, warm_start_skip_min,\n                                  warm_start_skip_max, warm_start_fwd_thresh,\n                                  warm_start_rev_thresh, warm_start_session_idx)\n        _s1_skip_bounds = _step1_skip_delivery_bounds_check(\n            _s1_skip_delivery_on, pinned=_s1_pinned,\n            trials=max_iterations, resume_study=resume_study)\n        _s1_skip_exact_pin = _step1_skip_delivery_exact_pin(\n            _s1_skip_delivery_on, _s1_pinned, max_iterations, resume_study,\n            warm_start_skip_min, warm_start_skip_max)\n        _step1_skip_delivery_preflight(_s1_search_ctx, _s1_skip_delivery_on,\n                                       resume_study=resume_study, study_name=study_name)\n',
     '        _step1_skip_delivery_preflight(_s1_search_ctx, _s1_skip_delivery_on,\n                                       resume_study=resume_study, study_name=study_name)\n        # the bounds first: before the named-study check loads any store\n        _s1_pinned = None not in (warm_start_window, warm_start_offset, warm_start_skip_min,\n                                  warm_start_skip_max, warm_start_fwd_thresh,\n                                  warm_start_rev_thresh, warm_start_session_idx)\n        _s1_skip_bounds = _step1_skip_delivery_bounds_check(\n            _s1_skip_delivery_on, pinned=_s1_pinned,\n            trials=max_iterations, resume_study=resume_study)\n        _s1_skip_exact_pin = _step1_skip_delivery_exact_pin(\n            _s1_skip_delivery_on, _s1_pinned, max_iterations, resume_study,\n            warm_start_skip_min, warm_start_skip_max)\n',
     g_sw2_q4_domains),
    ("MS43 the child never hands the exact pin", W_,
     "    return [int(skip_min), int(skip_max)]", "    return None", g_sw3_q4_exact_pin),
    ("MS44 the study body ignores the exact pin", BAY_PATH,
     "            if _exact_pin is not None:\n                if trial.number != 0:",
     "            if False:\n                if trial.number != 0:", g_sw3_q4_exact_pin),
    ("MS45 the exact pin carried on every run", W_,
     "        if _s1_skip_exact_pin is not None:\n            # [SKIP-DELIVERY SWITCH] only",
     "        if True:\n            # [SKIP-DELIVERY SWITCH] only", g_sw3_q4_exact_pin),
    ("MS46 the exact pin applied past trial 0", BAY_PATH,
     "                if trial.number != 0:\n                    raise RuntimeError(",
     "                if False:\n                    raise RuntimeError(", g_sw3_q4_exact_pin),
    ("MS47 search open reads the raw resume_study again", L,
     '                                               bool(final_params.get("study_name"))):',
     '                                               bool(final_params.get("resume_study", False))):',
     g_sw3_resume_rule),
    ("MS42 the switched run's scope banner removed", W_,
     "f\"holds only with a separate output root per arm and ONE publisher at a time: no \"",
     "f\"holds: no \"", g_sw2_q2_scope),
    ("MS36 the launcher reads a non-bool as off", O,
     '    _vsd(deliver_skip_range, where="run_bayesian_optimization")\n',
     "    pass\n", g_r3_q4_launcher),
    ("MS27 search open skips the on-pin range check", L,
     "                _sd_value, pin_bundle[\"warm_start_skip_min\"], pin_bundle[\"warm_start_skip_max\"],\n                where=\"presented_configuration\")",
     "                _sd_value, where=\"presented_configuration\")", g_q4_validator),
    ("MS28 estimator range check removed", V,
     "            on = validate_skip_delivery(g[\"deliver_skip_range\"], g[\"skip_min\"], g[\"skip_max\"],\n                                        where=\"volume geometry\")",
     "            on = validate_skip_delivery(g[\"deliver_skip_range\"], where=\"volume geometry\")",
     g_q4_validator),
    ("MS29 CLI config-file route drops the switch", O,
     "        if getattr(args, 'config_file', None):\n            parser.error(\"--deliver-skip-range is not carried",
     "        if False:\n            parser.error(\"--deliver-skip-range is not carried", g_q4_cli),
    ("MS19 coordinator attribute never set", O,
     "        coordinator.deliver_skip_range = True\n", "        pass\n", g_wo_on),
]


def main():
    global MOD
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    print("VERIFICATION test_miner_skip_switch START", flush=True)
    for p in (RMC_PATH, WOI_PATH, SSL_PATH, SVE_PATH):
        print(f"  {os.path.relpath(p, ROOT)} sha256 {hashlib.sha256(SRC[p].encode()).hexdigest()[:16]}")
    MOD = live_modules()
    passed = 0
    for name, fn in GATES:
        try:
            print(f"[PASS] {name}: {fn()}", flush=True)
            passed += 1
        except Exception as exc:                                       # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:400]}", flush=True)
    killed = 0
    for name, path, old, new, gate in MUTANTS:
        n = SRC[path].count(old)
        if n != 1:
            print(f"[FAIL] {name}: anchor found {n} times", flush=True)
            continue
        src = dict(SRC)
        src[path] = SRC[path].replace(old, new)
        MOD = live_modules(src)
        why = ""
        try:
            gate()
            caught = False
        except Exception as exc:                                       # noqa: BLE001
            caught, why = True, f"{type(exc).__name__}: {str(exc)[:140]}"
        finally:
            MOD = live_modules()
        print(f"[{'KILLED' if caught else 'SURVIVED'}] {name} (by {gate.__name__}) {why}", flush=True)
        killed += caught
    total = len(GATES) + len(MUTANTS)
    print(f"\n{passed}/{len(GATES)} gates PASS; {killed}/{len(MUTANTS)} mutants KILLED")
    print("RESULT: PASS" if passed + killed == total else "RESULT: FAIL")
    print("VERIFICATION test_miner_skip_switch COMPLETE", flush=True)
    return 0 if passed + killed == total else 1


if __name__ == "__main__":
    sys.exit(main())
