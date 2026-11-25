#!/usr/bin/env python3
"""Window-anchor Brief II (flag-selectable derived anchor domain): gates and mutants.

Run: python3 -u tests/test_window_anchor_brief2.py   (VM 101, ~/venvs/torch, CPU only)

Fixtures only. No GPU, no rig, no Step-1 run, no pipeline step. All scratch output goes to
<worktree>/data/test_window_anchor_brief2/ and is removed at the end.

Gates
  G-AC3         anchor 149 with window 50 is NOT in control_era; the control ceiling is the
                live bounded anchor max (100), never the envelope (149)
  G-DOMAIN-<s>  per session filter: effective range == [0, N_filtered - w]; the edge agrees
                with the worker's own load_residue_window (loads at the max, refuses max+1)
  G-REFUSE      out-of-domain refusals name anchor, domain, sessions and dataset sha256
  G-CAP         max_cap narrows, never widens; empty domain refused; malformed config refused
  G-RESOLVE     the resolver proves its count against the worker; a wrong frozen digest refused
  G-OBJ-DERIVED the real BayesianOptimization.search, setting on: every trial's anchor
                distribution is [0, N - w] for its own window and sessions; record attached
  G-PIN         a hand pin anywhere in the derived domain runs; one past it, or a window size
                outside search_bounds.window_size, is refused before the objective is called
  G-DEFAULT-OBJ setting off: trial params of a seeded study are identical to HEAD's code
  G-DEFAULT-SEARCH setting off/absent: presented_configuration identical to HEAD's code
  G-ARM         a study or search made by one mode refuses the other; untagged studies are
                unaffected in the bounded mode
  G-ROUTE       three hops: manifest default_params + args_map, CLI flag, signatures
  G-CLI         CLI refuses the flag with --config-file and without --use-range-miner
Mutants (each: APPLIED, EXECUTED, DETECTED at its credited assertion)
  M8  control-era ceiling 100 -> envelope (100 + w - 1)        credited: G-AC3
  M1  clamp restored in check_window_choice                    credited: G-REFUSE
  M2  max_cap widens (max instead of min)                      credited: G-CAP
  M3  session filter dropped from the count                    credited: G-RESOLVE
  M4  derived branch samples the bounded range                 credited: G-OBJ-DERIVED
  M5  seam forwarding removed in BayesianOptimization.search   credited: G-OBJ-DERIVED
  M6  pinned choice not validated in the objective             credited: G-PIN
  M7  search key added when off                                credited: G-DEFAULT-SEARCH
"""
import ast
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(ROOT, "data", "test_window_anchor_brief2")
sys.path.insert(0, ROOT)


def out(msg):
    # importing sieve_gpu_worker (via miner.range_miner_worker) replaces sys.stdout
    os.write(1, (msg + "\n").encode())


# ---------------------------------------------------------------- fixtures --------
def make_fixture(path):
    """60 evening-only days, then 200 days with evening before midday (canonical order).
    N: evening 260, midday 200, both 460."""
    recs = []
    import datetime as _dt
    d0 = _dt.date(2000, 1, 1)
    k = 0
    for i in range(260):
        day = (d0 + _dt.timedelta(days=i)).isoformat()
        recs.append({"date": day, "session": "evening", "draw": (k * 37) % 1000}); k += 1
        if i >= 60:
            recs.append({"date": day, "session": "midday", "draw": (k * 53) % 1000}); k += 1
    with open(path, "w") as f:
        json.dump(recs, f)
    return {("evening",): 260, ("midday",): 200, ("evening", "midday"): 460}


SESSION_OPTIONS = [["midday", "evening"], ["midday"], ["evening"]]


def fixture_domain(path, cap=None, mn=0, ws_min=6, ws_max=50, bounded_max=100, guard=False):
    from utils.window_anchor_domain import resolve_anchor_domain
    from window_optimizer import step1_holdout_split_point
    return resolve_anchor_domain(path, SESSION_OPTIONS, window_size_min=ws_min,
                                 window_size_max=ws_max, bounded_anchor_max=bounded_max,
                                 anchor_bounds={"min": mn, "max_cap": cap},
                                 holdout_guard=guard,
                                 holdout_split_point_fn=step1_holdout_split_point if guard else None)


class GateFail(AssertionError):
    pass


def check(cond, aid, detail=""):
    if not cond:
        raise GateFail(f"[{aid}] {detail}")


# ---------------------------------------------------------------- gates -----------
def gate_ac3(fx):
    dom = fixture_domain(fx)
    lo, hi = dom.control_era_range(50, ["midday"])
    eff = dom.effective_range(50, ["midday"])
    check(not (lo <= 149 <= hi), "AC3-149-not-in-control", f"149 in control_era {(lo, hi)}")
    check(eff == (0, 150), "AC3-eff", f"effective {eff}")
    check((lo, hi) == (0, 100), "AC3-ceiling", f"control_era {(lo, hi)}")
    check(eff[0] <= 149 <= eff[1], "AC3-149-in-derived", "149 should be a legal derived anchor")
    small = fixture_domain(fx, cap=60)
    check(small.control_era_range(50, ["midday"]) == (0, 60), "AC3-capped", "")


def gate_domain(fx, counts):
    from miner.range_miner_worker import load_residue_window, ResidueResolutionError
    dom = fixture_domain(fx)
    for opt in SESSION_OPTIONS:
        key = tuple(sorted(opt))
        n = counts[key]
        for w in (6, 12, 50):
            lo, hi = dom.effective_range(w, opt)
            check((lo, hi) == (0, n - w), f"DOM-{'+'.join(key)}-{w}", f"{(lo, hi)} vs N={n}")
            dom.check_window_choice(w, hi, opt)
            load_residue_window(fx, w, opt, hi)            # worker agrees at the max
            try:
                load_residue_window(fx, w, opt, hi + 1)
                raise GateFail(f"[DOM-worker-edge] worker loaded {hi + 1}")
            except ResidueResolutionError:
                pass
            try:
                dom.check_window_choice(w, hi + 1, opt)
                raise GateFail(f"[DOM-edge-{'+'.join(key)}-{w}] {hi + 1} accepted")
            except GateFail:
                raise
            except Exception as e:
                check(type(e).__name__ == "WindowAnchorDomainError", "DOM-edge-type", repr(e))


def gate_refuse(fx):
    from utils.window_anchor_domain import WindowAnchorDomainError
    dom = fixture_domain(fx)
    sha = hashlib.sha256(open(fx, "rb").read()).hexdigest()
    for bad in (151, -1, 10_000):
        try:
            got = dom.check_window_choice(50, bad, ["midday"])
            raise GateFail(f"[REFUSE-raised] anchor {bad} returned {got}")
        except WindowAnchorDomainError as e:
            m = str(e)
            check(f"window_anchor {bad}" in m, "REFUSE-anchor", m)
            check("[0, 150]" in m, "REFUSE-domain", m)
            check("['midday']" in m, "REFUSE-sessions", m)
            check(sha in m, "REFUSE-sha256", m)
    for bad in (True, 3.0, "7"):
        try:
            dom.check_window_choice(12, bad, ["midday"])
            raise GateFail(f"[REFUSE-type] {bad!r} accepted")
        except WindowAnchorDomainError:
            pass


def gate_cap(fx):
    from utils.window_anchor_domain import WindowAnchorDomainError, load_window_anchor_bounds
    check(fixture_domain(fx, cap=10_000).effective_range(12, ["midday"]) == (0, 188),
          "CAP-never-widens", str(fixture_domain(fx, cap=10_000).effective_range(12, ["midday"])))
    check(fixture_domain(fx, cap=40).effective_range(12, ["midday"]) == (0, 40), "CAP-narrows", "")
    try:
        fixture_domain(fx, mn=190).effective_range(12, ["midday"])
        raise GateFail("[CAP-empty] empty domain accepted")
    except WindowAnchorDomainError:
        pass
    cfgdir = os.path.join(SCRATCH, "cfg"); os.makedirs(cfgdir, exist_ok=True)
    good = os.path.join(cfgdir, "good.json")
    json.dump({"search_bounds": {"window_anchor": {"min": 3, "max_cap": None}}}, open(good, "w"))
    check(load_window_anchor_bounds(good) == {"min": 3, "max_cap": None}, "CAP-load", "")
    check(load_window_anchor_bounds(os.path.join(cfgdir, "absent.json")) ==
          {"min": 0, "max_cap": None}, "CAP-default", "")
    for bad in ({"min": -1}, {"min": True}, {"max_cap": "9"}, {"mx": 3}, 5):
        p = os.path.join(cfgdir, "bad.json")
        json.dump({"search_bounds": {"window_anchor": bad}}, open(p, "w"))
        try:
            load_window_anchor_bounds(p)
            raise GateFail(f"[CAP-malformed] {bad!r} accepted")
        except WindowAnchorDomainError:
            pass
    # the live config today has no window_anchor key: declared defaults apply
    check(load_window_anchor_bounds(os.path.join(ROOT, "distributed_config.json")) ==
          {"min": 0, "max_cap": None}, "CAP-live-default", "")


def gate_resolve(fx, counts):
    from utils.window_anchor_domain import WindowAnchorDomainError
    dom = fixture_domain(fx)
    check(dict(dom.n_filtered) == counts, "RESOLVE-counts", str(dict(dom.n_filtered)))
    check(dom.dataset_sha256 == hashlib.sha256(open(fx, "rb").read()).hexdigest(), "RESOLVE-sha", "")
    from utils.window_anchor_domain import resolve_anchor_domain
    try:
        resolve_anchor_domain(fx, SESSION_OPTIONS, window_size_min=6, window_size_max=50,
                              bounded_anchor_max=100, expected_sha256="0" * 64)
        raise GateFail("[RESOLVE-frozen] wrong frozen digest accepted")
    except WindowAnchorDomainError:
        pass


def _stub_objective(calls):
    from window_optimizer_bayesian import OptimizationResult

    def f(config, optuna_trial=None):
        calls.append((config.window_size, config.offset, list(config.sessions)))
        return OptimizationResult(config=config, forward_count=1, reverse_count=1,
                                  bidirectional_count=(config.offset % 7) + 1)
    return f


CREATED = []          # every study file a search in this process created, for cleanup


def _run_search(strategy, bounds, trials, ctx=None):
    from window_optimizer_bayesian import ResultScorer
    calls = []
    os.makedirs(SCRATCH, exist_ok=True)
    # study names are window_opt_<unix seconds> (window_optimizer_bayesian.py): space the calls
    time.sleep(1.1)
    before = _snap()
    cwd = os.getcwd(); os.chdir(SCRATCH)
    try:
        res = strategy.search(_stub_objective(calls), bounds, trials, ResultScorer(),
                              study_name="", trse_context_file=os.path.join(SCRATCH, "no_trse.json"),
                              trial_history_context=ctx)
    finally:
        os.chdir(cwd)
        CREATED.extend(_studies_created_since(before))
    return res, calls


def _snap():
    d = os.path.join(ROOT, "optuna_studies")
    return set(os.listdir(d)) if os.path.isdir(d) else set()


def _studies_created_since(before):
    return sorted(_snap() - before)


def _load_study(name):
    import optuna
    return optuna.load_study(study_name=name[:-3],
                             storage=f"sqlite:///{ROOT}/optuna_studies/{name}")


def gate_obj_derived(fx, counts):
    from window_optimizer import BayesianOptimization, SearchBounds
    bounds = SearchBounds()       # code defaults: window [2,50], offset [0,100]
    bounds.min_window_size, bounds.max_window_size = 6, 50
    dom = fixture_domain(fx)
    strat = BayesianOptimization(n_initial=3)
    strat._window_anchor_domain = dom
    before = _snap()
    _, calls = _run_search(strat, bounds, 10)
    new = _studies_created_since(before)
    check(len(new) == 1, "OBJ-one-study", str(new))
    st = _load_study(new[0])
    check(len(calls) == 10, "OBJ-calls", str(len(calls)))
    for t in st.trials:
        ws, si = t.params["window_size"], t.params["session_idx"]
        sess = bounds.session_options[si]
        n = counts[tuple(sorted(sess))]
        dist = t.distributions["offset"]
        check((dist.low, dist.high) == (0, n - ws), "OBJ-dist-is-derived",
              f"trial {t.number}: {(dist.low, dist.high)} vs N-w={n - ws}")
        rec = t.user_attrs.get("window_anchor_domain")
        check(rec is not None and rec["anchor_range"] == [0, n - ws]
              and rec["dataset_sha256"] == dom.dataset_sha256, "OBJ-record", str(rec))
    return new


def gate_pin(fx, counts):
    from window_optimizer import BayesianOptimization, SearchBounds
    bounds = SearchBounds(); bounds.min_window_size, bounds.max_window_size = 6, 50
    dom = fixture_domain(fx)
    created = []
    base = {"warm_start_skip_min": 0, "warm_start_skip_max": 16,
            "warm_start_fwd_thresh": 0.5, "warm_start_rev_thresh": 0.5}
    # in-domain pins: far above the bounded 100, at the derived max, for each session option
    for si, ws, anchor in ((0, 12, 448), (1, 50, 150), (2, 6, 254), (1, 12, 101)):
        before = _snap()
        strat = BayesianOptimization(n_initial=3); strat._window_anchor_domain = dom
        ctx = dict(base, warm_start_window=ws, warm_start_offset=anchor, warm_start_session_idx=si)
        _, calls = _run_search(strat, bounds, 1, ctx)
        created += _studies_created_since(before)
        check(calls == [(ws, anchor, bounds.session_options[si])], "PIN-runs", str(calls))
    # out of domain: one past the derived max; a window size outside [6, 50]
    for si, ws, anchor, aid in ((1, 50, 151, "PIN-past-max"), (2, 60, 10, "PIN-window-size")):
        before = _snap()
        strat = BayesianOptimization(n_initial=3); strat._window_anchor_domain = dom
        ctx = dict(base, warm_start_window=ws, warm_start_offset=anchor, warm_start_session_idx=si)
        calls = []
        try:
            _, calls = _run_search(strat, bounds, 1, ctx)
            refused = False
        except Exception as e:
            refused = "WindowAnchorDomainError" in type(e).__name__ or \
                "WindowAnchorDomainError" in repr(e)
        created += _studies_created_since(before)
        if not refused:
            # Optuna may catch the objective's exception and mark the trial FAIL
            st = _load_study(created[-1]) if created else None
            failed = st is not None and all(t.state.name == "FAIL" for t in st.trials)
            refused = failed and not calls
        check(refused and not calls, aid, f"calls={calls}")
    return created


def _head_source(rel):
    # base, not HEAD: HEAD may be a WIP snapshot of this change (b78a15c, 2026-11-23 20:44)
    base = os.environ.get("TFM_IDENTITY_BASE") or subprocess.check_output(
        ["git", "-C", ROOT, "merge-base", "HEAD", "main"]).decode().strip()
    return subprocess.check_output(["git", "-C", ROOT, "show", f"{base}:{rel}"]).decode()


_DEFAULT_OBJ_CHILD = r'''
import json, os, sys
root, mode, scratch = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "head":
    sys.path.insert(0, os.path.join(scratch, "head_src"))
sys.path.insert(1, root)
import window_optimizer_bayesian as wob
from window_optimizer import SearchBounds
b = SearchBounds(); b.min_window_size, b.max_window_size = 6, 50
s = wob.OptunaBayesianSearch(n_startup_trials=3, seed=11)
seen = []
def f(config, optuna_trial=None):
    seen.append([config.window_size, config.offset, list(config.sessions)])
    return wob.OptimizationResult(config=config, forward_count=1, reverse_count=1,
                                  bidirectional_count=(config.offset % 7) + 1)
os.chdir(scratch)
s.search(f, b, 8, wob.ResultScorer(), trse_context_file=os.path.join(scratch, "none.json"))
os.write(1, ("RESULT " + json.dumps({"seen": seen, "file": wob.__file__}) + "\n").encode())
'''


def gate_default_obj(created):
    hs = os.path.join(SCRATCH, "head_src"); os.makedirs(os.path.join(hs, "optuna_studies"),
                                                         exist_ok=True)
    for rel in ("window_optimizer.py", "window_optimizer_bayesian.py"):
        open(os.path.join(hs, rel), "w").write(_head_source(rel))
    child = os.path.join(SCRATCH, "default_obj_child.py")
    open(child, "w").write(_DEFAULT_OBJ_CHILD)
    res = {}
    before = _snap()
    for mode in ("head", "live"):
        p = subprocess.run([sys.executable, "-u", child, ROOT, mode, SCRATCH],
                           capture_output=True, text=True, timeout=600)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        check(p.returncode == 0 and line, f"DEFOBJ-{mode}-ran", p.stderr[-800:])
        res[mode] = json.loads(line[-1][7:])
    created.extend(_studies_created_since(before))
    check(res["head"]["file"].startswith(hs) and res["live"]["file"].startswith(ROOT)
          and not res["live"]["file"].startswith(hs), "DEFOBJ-which-code", str(res))
    check(res["head"]["seen"] == res["live"]["seen"], "DEFOBJ-identical",
          f"head={res['head']['seen']} live={res['live']['seen']}")
    check(len(res["live"]["seen"]) == 8 and all(o <= 100 for _, o, _ in res["live"]["seen"]),
          "DEFOBJ-bounded", str(res["live"]["seen"]))


def _presented(mod, final_params, pin=None):
    from window_optimizer import load_search_bounds_from_config, SearchBounds
    bounds = load_search_bounds_from_config(os.path.join(ROOT, "distributed_config.json"))
    return mod.presented_configuration(final_params, pin, bounds, SearchBounds().session_options)


def gate_default_search():
    import importlib.util
    hs = os.path.join(SCRATCH, "head_src"); os.makedirs(hs, exist_ok=True)
    hp = os.path.join(hs, "step1_search_ledger_head.py")
    open(hp, "w").write(_head_source("utils/step1_search_ledger.py"))
    spec = importlib.util.spec_from_file_location("step1_search_ledger_head", hp)
    head = importlib.util.module_from_spec(spec)
    sys.modules["step1_search_ledger_head"] = head
    spec.loader.exec_module(head)
    import utils.step1_search_ledger as live
    base = {"prng_type": "java_lcg", "test_both_modes": True, "window_trials": 1,
            "max_seeds": 1 << 26, "strategy": "bayesian", "n_parallel": 1}
    pin = {"warm_start_window": 12, "warm_start_offset": 25, "warm_start_skip_min": 6,
           "warm_start_skip_max": 99, "warm_start_fwd_thresh": 0.71,
           "warm_start_rev_thresh": 0.47, "warm_start_session_idx": 1}
    for fp, pb in ((base, None), (dict(base, derived_anchor_domain=False), None),
                   (base, pin), (dict(base, derived_anchor_domain=False), pin)):
        h = _presented(head, {k: v for k, v in fp.items() if k != "derived_anchor_domain"}, pb)
        l = _presented(live, fp, pb)
        check(h == l, "DEFSEARCH-identical", f"\nhead={h}\nlive={l}")
    on = _presented(live, dict(base, derived_anchor_domain=True))
    check(on["search_space"].get("window_anchor_domain") == "derived"
          and on["search_space_digest"] != _presented(live, base)["search_space_digest"],
          "DEFSEARCH-on-separate", str(on["search_space"]))
    for bad in (None, "true", 1):
        try:
            _presented(live, dict(base, derived_anchor_domain=bad))
            raise GateFail(f"[DEFSEARCH-validator] {bad!r} accepted")
        except live.Step1SearchRefused:
            pass


def gate_arm(created_names):
    from utils.window_anchor_domain import (anchor_domain_study_check, anchor_domain_search_check,
                                            WindowAnchorDomainError)
    import optuna
    st = optuna.create_study()
    anchor_domain_study_check(st, None, 0)                # untagged, bounded: no effect
    check("window_anchor_domain" not in st.user_attrs, "ARM-off-writes-nothing", "")
    st.enqueue_trial({"x": 1}); st.optimize(lambda t: t.suggest_int("x", 0, 2), n_trials=1)
    try:
        anchor_domain_study_check(st, "derived", 1)       # derived on a bounded study
        raise GateFail("[ARM-derived-adopts-bounded] accepted")
    except WindowAnchorDomainError:
        pass
    fresh = optuna.create_study()
    anchor_domain_study_check(fresh, "derived", 0)
    check(fresh.user_attrs.get("window_anchor_domain") == "derived", "ARM-tag", "")
    for other in (None, "derived_holdout_guard"):
        try:
            anchor_domain_study_check(fresh, other, 0)
            raise GateFail(f"[ARM-other-on-derived] {other!r} accepted")
        except WindowAnchorDomainError:
            pass
    anchor_domain_search_check({}, None, "s0")
    anchor_domain_search_check({"window_anchor_domain": "derived"}, "derived", "s0")
    for space, tag in (({}, "derived"), ({"window_anchor_domain": "derived"}, None),
                       ({"window_anchor_domain": "derived"}, "derived_holdout_guard"),
                       ({"window_anchor_domain": "derived_holdout_guard"}, "derived")):
        try:
            anchor_domain_search_check(space, tag, "s1")
            raise GateFail(f"[ARM-search] {space} run={tag} accepted")
        except WindowAnchorDomainError:
            pass


def gate_route():
    m = json.load(open(os.path.join(ROOT, "agent_manifests", "window_optimizer.json")))
    check(m["default_params"].get("derived_anchor_domain") is False, "ROUTE-hop1-default", "")
    amap = {}
    for a in m["actions"]:
        amap.update(a.get("args_map", {}))
    check(amap.get("derived-anchor-domain") == "derived_anchor_domain", "ROUTE-hop1-args_map", "")
    check(m["param_docs"]["derived_anchor_domain"]["default"] is False, "ROUTE-hop1-doc", "")
    check(m["default_params"].get("anchor_holdout_guard") is False
          and amap.get("anchor-holdout-guard") == "anchor_holdout_guard"
          and m["param_docs"]["anchor_holdout_guard"]["default"] is False, "ROUTE-guard-hop1", "")
    import window_optimizer as wo
    sig = inspect.signature(wo.run_bayesian_optimization)
    p = sig.parameters.get("derived_anchor_domain")
    check(p is not None and p.default is False and p.kind == p.KEYWORD_ONLY, "ROUTE-hop2-sig", str(p))
    pg = sig.parameters.get("anchor_holdout_guard")
    check(pg is not None and pg.default is False and pg.kind == pg.KEYWORD_ONLY,
          "ROUTE-guard-hop2-sig", str(pg))
    src = open(os.path.join(ROOT, "window_optimizer.py")).read()
    tree = ast.parse(src)
    flags = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "add_argument" and n.args
             and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--derived-anchor-domain"]
    check(len(flags) == 1, "ROUTE-hop2-flag", str(len(flags)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "run_bayesian_optimization"]
    check(any(any(k.arg == "derived_anchor_domain" for k in c.keywords) for c in calls),
          "ROUTE-hop2-call", "")
    check(any(any(k.arg == "anchor_holdout_guard" for k in c.keywords) for c in calls),
          "ROUTE-guard-hop2-call", "")
    gflags = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and getattr(n.func, "attr", "") == "add_argument" and n.args
              and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--anchor-holdout-guard"]
    check(len(gflags) == 1, "ROUTE-guard-flag", str(len(gflags)))
    ow = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
          and getattr(n.func, "attr", "") == "optimize_window"]
    check(any(any(k.arg is None and "derived_anchor_domain" in ast.unparse(k.value)
                  for k in c.keywords) for c in ow), "ROUTE-hop2-forward", "")
    check(any(any(k.arg is None and "anchor_holdout_guard" in ast.unparse(k.value)
                  for k in c.keywords) for c in ow), "ROUTE-guard-hop2-forward", "")
    fn3 = _optimize_window_ast()
    a3 = [a.arg for a in fn3.args.args + fn3.args.kwonlyargs]
    check("anchor_holdout_guard" in a3, "ROUTE-guard-hop3-sig", str(a3[-3:]))
    import window_optimizer_integration_final as wif
    fn = None
    for name in dir(wif):
        obj = getattr(wif, name)
        if inspect.isclass(obj) and hasattr(obj, "optimize_window"):
            fn = obj.optimize_window
    if fn is None:
        isrc = open(os.path.join(ROOT, "window_optimizer_integration_final.py")).read()
        for n in ast.walk(ast.parse(isrc)):
            if isinstance(n, ast.FunctionDef) and n.name == "optimize_window":
                args = [a.arg for a in n.args.args + n.args.kwonlyargs]
                check("derived_anchor_domain" in args, "ROUTE-hop3-sig", str(args))
                return
        raise GateFail("[ROUTE-hop3-sig] optimize_window not found")
    check("derived_anchor_domain" in inspect.signature(fn).parameters, "ROUTE-hop3-sig", "")


def gate_cli():
    for flags, extra, aid, msg in (
            (["--derived-anchor-domain"], ["--config-file", "x.json", "--use-range-miner"],
             "CLI-config-file", "--derived-anchor-domain is refused"),
            (["--derived-anchor-domain"], [], "CLI-no-miner", "--derived-anchor-domain is refused"),
            (["--anchor-holdout-guard"], ["--use-range-miner"], "CLI-guard-needs-derived",
             "--anchor-holdout-guard caps the derived domain only")):
        p = subprocess.run([sys.executable, "-u", os.path.join(ROOT, "window_optimizer.py")]
                           + flags + ["--lottery-file", "nope.json"] + extra,
                           capture_output=True, text=True, timeout=300, cwd=SCRATCH)
        check(p.returncode == 2 and msg in p.stderr, aid, f"rc={p.returncode} {p.stderr[-600:]}")


# ---------------------------------------------------------------- lead round 1 ----
def _optimize_window_ast():
    # A structural mutant installs its mutated source on the module; the gate then reads
    # THAT text, and calling _MUT_HIT is the proof the gate consumed the mutation.
    import window_optimizer_integration_final as _wif
    src = getattr(_wif, "__mutated_source__", None)
    if src is not None:
        import builtins
        getattr(builtins, "_MUT_HIT", lambda: None)()
    else:
        src = open(os.path.join(ROOT, "window_optimizer_integration_final.py")).read()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == "optimize_window":
            return n
    raise GateFail("[NP2-BIND-fn] optimize_window not found")


def gate_np2_bind():
    """Point 1: `_wad_domain` is bound by a statement of optimize_window's own body (not
    under any branch) before every read; every read sits where None is handled."""
    fn = _optimize_window_ast()
    inits = [st for st in fn.body if isinstance(st, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_wad_domain" for t in st.targets)
             and isinstance(st.value, ast.Constant) and st.value.value is None]
    check(len(inits) == 1, "NP2-BIND-init", f"{len(inits)} function-level `_wad_domain = None`")
    init_line = inits[0].lineno
    loads = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "_wad_domain"
             and isinstance(n.ctx, ast.Load)]
    check(loads and all(n.lineno > init_line for n in loads), "NP2-BIND-order",
          f"init {init_line}, loads {[n.lineno for n in loads]}")
    # the only unguarded use is passing it to the None-safe helper; method calls on it sit
    # under an `if` that binds it in the same block
    attach = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
              and getattr(n.func, "id", "") == "_attach_window_anchor_domain_record"]
    check(len(attach) == 1, "NP2-BIND-attach", str(len(attach)))
    import window_optimizer_integration_final as wif
    r = {"x": 1}
    check(wif._attach_window_anchor_domain_record(r, None) == {"x": 1}, "NP2-BIND-none-safe", str(r))


def gate_results_file(fx):
    """Point 2: the record is attached before the save; the saved file carries it."""
    fn = _optimize_window_ast()
    found = False
    for blk in ast.walk(fn):
        body = getattr(blk, "body", None)
        if not isinstance(body, list):
            continue
        idx_a = [i for i, st in enumerate(body) if isinstance(st, ast.Expr)
                 and isinstance(st.value, ast.Call)
                 and getattr(st.value.func, "id", "") == "_attach_window_anchor_domain_record"]
        idx_s = [i for i, st in enumerate(body) if isinstance(st, ast.Expr)
                 and isinstance(st.value, ast.Call)
                 and getattr(st.value.func, "attr", "") == "save_results"]
        if idx_a and idx_s:
            check(idx_a[0] < idx_s[0], "RESFILE-order",
                  f"attach at stmt {idx_a[0]}, save_results at stmt {idx_s[0]}")
            found = True
    check(found, "RESFILE-same-block", "attach and save_results not in one block")
    import window_optimizer_integration_final as wif
    from window_optimizer import WindowOptimizer
    dom = fixture_domain(fx, guard=True)
    out_path = os.path.join(SCRATCH, "window_optimization_results.json")
    res = wif._attach_window_anchor_domain_record({"best_config": {"offset": 3}}, dom)
    WindowOptimizer.save_results(None, res, out_path)
    saved = json.load(open(out_path))
    check(saved.get("window_anchor_domain") == json.loads(json.dumps(dom.run_record())),
          "RESFILE-saved", str(saved.get("window_anchor_domain")))
    check(saved["window_anchor_domain"]["holdout_guard"] is True, "RESFILE-guard", "")


def _filtered_combined_index(records, opt):
    return [i for i, e in enumerate(records) if e.get("session") in opt]


def gate_holdout(fx):
    """Point 4: with the guard on, the whole window ends before the first holdout record of
    the Step 1 split for that session filter; one past it is refused, never clamped."""
    from utils.window_anchor_domain import (WindowAnchorDomainError, resolve_anchor_domain,
                                            validate_anchor_holdout_guard)
    from window_optimizer import step1_holdout_split_point
    records = json.load(open(fx))
    split = step1_holdout_split_point(len(records))
    check(split == int(len(records) * 0.8), "HOLD-split-formula", str(split))
    dom = fixture_domain(fx, guard=True)
    sha = dom.dataset_sha256
    for opt in SESSION_OPTIONS:
        comb = _filtered_combined_index(records, opt)
        n_train = sum(1 for i in comb if i < split)
        for w in (6, 12, 50):
            lo, hi = dom.effective_range(w, opt)
            check(hi == min(len(comb) - w, n_train - w), "HOLD-max", f"{opt} w{w}: {hi}")
            # the window's last record is train; the record after it is the first holdout one
            check(comb[hi + w - 1] < split, "HOLD-ends-before",
                  f"{opt} w{w}: last record {comb[hi + w - 1]} >= split {split}")
            check(comb[hi + w] >= split, "HOLD-is-max", f"{opt} w{w}: {comb[hi + w]} < split")
            try:
                dom.check_window_choice(w, hi + 1, opt)
                raise GateFail(f"[HOLD-refuse] {hi + 1} accepted for {opt} w{w}")
            except WindowAnchorDomainError as e:
                m = str(e)
                check("holdout_guard=True" in m and sha in m and f"window_anchor {hi + 1}" in m,
                      "HOLD-refuse-names", m)
            rec = dom.trial_record(w, opt)
            check(rec["holdout_guard"] is True and rec["n_train"] == n_train
                  and rec["holdout_split_point"] == split and rec["anchor_range"] == [lo, hi],
                  "HOLD-trial-record", str(rec))
    rr = dom.run_record()
    check(rr["holdout_guard"] is True and rr["holdout_split_point"] == split
          and rr["tag"] == "derived_holdout_guard", "HOLD-run-record", str(rr))
    off = fixture_domain(fx, guard=False)
    check(off.effective_range(12, ["midday"]) == (0, 188) and off.run_record()["holdout_guard"] is False,
          "HOLD-off-unchanged", str(off.effective_range(12, ["midday"])))
    for bad in (("x", True), (None, True), (1, True)):
        try:
            validate_anchor_holdout_guard(bad[0], bad[1], where="t")
            raise GateFail(f"[HOLD-validator] {bad!r} accepted")
        except WindowAnchorDomainError:
            pass
    try:
        validate_anchor_holdout_guard(True, False, where="t")
        raise GateFail("[HOLD-needs-derived] guard without derived accepted")
    except WindowAnchorDomainError:
        pass
    try:
        resolve_anchor_domain(fx, SESSION_OPTIONS, window_size_min=6, window_size_max=50,
                              bounded_anchor_max=100, holdout_guard=True)
        raise GateFail("[HOLD-needs-split] guard without the split helper accepted")
    except WindowAnchorDomainError:
        pass


def gate_holdout_obj(fx):
    """Point 4 through the real search: sampled ranges are the guarded ones; a pin inside
    runs; a pin one past the guard is refused before the objective."""
    from window_optimizer import BayesianOptimization, SearchBounds, step1_holdout_split_point
    records = json.load(open(fx))
    split = step1_holdout_split_point(len(records))
    bounds = SearchBounds(); bounds.min_window_size, bounds.max_window_size = 6, 50
    dom = fixture_domain(fx, guard=True)
    strat = BayesianOptimization(n_initial=3); strat._window_anchor_domain = dom
    before = _snap()
    _, calls = _run_search(strat, bounds, 8)
    new = _studies_created_since(before)
    st = _load_study(new[0])
    for t in st.trials:
        ws, si = t.params["window_size"], t.params["session_idx"]
        opt = bounds.session_options[si]
        n_train = sum(1 for i in _filtered_combined_index(records, opt) if i < split)
        d = t.distributions["offset"]
        check((d.low, d.high) == (0, n_train - ws), "HOLDOBJ-dist", f"{(d.low, d.high)}")
        check(t.user_attrs["window_anchor_domain"]["holdout_guard"] is True, "HOLDOBJ-record", "")
    base = {"warm_start_skip_min": 0, "warm_start_skip_max": 16,
            "warm_start_fwd_thresh": 0.5, "warm_start_rev_thresh": 0.5}
    hi = dom.effective_range(12, ["midday"])[1]
    strat = BayesianOptimization(n_initial=3); strat._window_anchor_domain = dom
    _, calls = _run_search(strat, bounds, 1, dict(base, warm_start_window=12,
                                                  warm_start_offset=hi, warm_start_session_idx=1))
    check(calls == [(12, hi, ["midday"])], "HOLDOBJ-pin-in", str(calls))
    strat = BayesianOptimization(n_initial=3); strat._window_anchor_domain = dom
    calls = []
    try:
        _, calls = _run_search(strat, bounds, 1, dict(base, warm_start_window=12,
                                                      warm_start_offset=hi + 1,
                                                      warm_start_session_idx=1))
        raise GateFail(f"[HOLDOBJ-pin-out] anchor {hi + 1} ran: {calls}")
    except GateFail:
        raise
    except Exception as e:
        check("WindowAnchorDomainError" in type(e).__name__ and not calls, "HOLDOBJ-pin-out", repr(e))
    return new


TPE_MEASUREMENT = {}


def _instrumented_run(strategy, bounds, trials):
    """Run the real search with TPESampler's three entry points wrapped, and classify, per
    trial, where `offset` came from: relative (joint multivariate TPE), independent during
    the random startup, or independent TPE after startup."""
    from optuna.samplers import TPESampler
    import logging as _lg
    rec = {"relative_space": {}, "relative_keys": {}, "independent": {}, "sampler": {}}
    orig = (TPESampler.infer_relative_search_space, TPESampler.sample_relative,
            TPESampler.sample_independent)
    warn = []

    class _H(_lg.Handler):
        def emit(self, r):
            if "sampled independently" in r.getMessage():
                warn.append(r.getMessage())

    h = _H(); _lg.getLogger("optuna").addHandler(h)

    def irs(self, study, trial):
        rec["sampler"] = {"multivariate": bool(self._multivariate),
                          "n_startup_trials": int(self._n_startup_trials),
                          "group": bool(getattr(self, "_group", False))}
        sp = orig[0](self, study, trial)
        rec["relative_space"][trial.number] = sorted(sp)
        return sp

    def sr(self, study, trial, space):
        out_ = orig[1](self, study, trial, space)
        rec["relative_keys"][trial.number] = sorted(out_)
        return out_

    def si(self, study, trial, name, dist):
        from optuna.trial import TrialState
        n_done = len(study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,
                                                                TrialState.PRUNED)))
        rec["independent"].setdefault(trial.number, {})[name] = (
            "random_startup" if n_done < self._n_startup_trials else "tpe_independent")
        return orig[2](self, study, trial, name, dist)

    TPESampler.infer_relative_search_space, TPESampler.sample_relative, \
        TPESampler.sample_independent = irs, sr, si
    try:
        before = _snap()
        _run_search(strategy, bounds, trials)
        new = _studies_created_since(before)
    finally:
        TPESampler.infer_relative_search_space, TPESampler.sample_relative, \
            TPESampler.sample_independent = orig
        _lg.getLogger("optuna").removeHandler(h)
    return new, rec, warn


def gate_tpe_storage(fx):
    """Point 3: on the sqlite study storage Step 1 uses (window_optimizer_bayesian.py:709,
    checkout-relative), several trials with different `offset` ranges are accepted and
    persisted; and a measurement, not a verdict, of how TPE samples `offset` then."""
    import optuna
    from window_optimizer import BayesianOptimization, SearchBounds
    bounds = SearchBounds(); bounds.min_window_size, bounds.max_window_size = 6, 50
    created = []
    for arm in ("derived", "bounded"):
        strat = BayesianOptimization(n_initial=3)
        if arm == "derived":
            strat._window_anchor_domain = fixture_domain(fx)
        new, rec, warn = _instrumented_run(strat, bounds, 24)
        created += new
        check(len(new) == 1, f"TPE-{arm}-one-study", str(new))
        path = os.path.join(ROOT, "optuna_studies", new[0])
        st = optuna.load_study(study_name=new[0][:-3], storage=f"sqlite:///{path}")
        check(len(st.trials) == 24 and all(t.state.name == "COMPLETE" for t in st.trials),
              f"TPE-{arm}-complete", str([t.state.name for t in st.trials]))
        ranges = sorted({(t.distributions["offset"].low, t.distributions["offset"].high)
                         for t in st.trials})
        if arm == "derived":
            check(len(ranges) >= 2, "TPE-derived-ranges-differ", str(ranges))
        cls = {}
        for t in st.trials:
            if "offset" in rec["relative_keys"].get(t.number, []):
                c = "relative_joint"
            else:
                c = rec["independent"].get(t.number, {}).get("offset", "UNCLASSIFIED")
            cls[c] = cls.get(c, 0) + 1
        check("UNCLASSIFIED" not in cls, f"TPE-{arm}-classified", str(cls))
        sampler_desc = dict(rec["sampler"], **{"class": "TPESampler", "optuna": optuna.__version__})
        check(rec["sampler"], f"TPE-{arm}-sampler-read", "sampler attributes not observed")
        rel_has_offset = sum(1 for v in rec["relative_space"].values() if "offset" in v)
        TPE_MEASUREMENT[arm] = {
            "storage": f"sqlite:///<worktree>/optuna_studies/{new[0]}",
            "trials": 24, "distinct_offset_ranges": len(ranges),
            "offset_source_counts": cls,
            "trials_with_offset_in_relative_search_space": rel_has_offset,
            "independent_sampling_warnings": len(warn), "sampler": sampler_desc}
        out(f"   MEASURED {arm}: {json.dumps(TPE_MEASUREMENT[arm], sort_keys=True)}")
    return created


def gate_split_same():
    """The guard and the files read one split: both split assignments call the helper."""
    tree = ast.parse(open(os.path.join(ROOT, "window_optimizer.py")).read())
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "split_point" for t in n.targets)]
    check(len(assigns) == 2 and all(isinstance(a.value, ast.Call) and
                                    getattr(a.value.func, "id", "") == "step1_holdout_split_point"
                                    for a in assigns), "SPLIT-helper", ast.dump(assigns[0].value)[:200])
    from window_optimizer import step1_holdout_split_point
    check(all(step1_holdout_split_point(n) == int(n * 0.8) for n in range(0, 20000, 7)),
          "SPLIT-value", "")


# ---------------------------------------------------------------- mutants ---------
MUTANTS = {
    "M8": ("utils/window_anchor_domain.py", "G-AC3", "AC3-149-not-in-control",
           "return lo, min(int(self.bounded_anchor_max), emax)",
           "_MUT_HIT(); return lo, min(int(self.bounded_anchor_max) + int(window_size) - 1, emax)"),
    "M1": ("utils/window_anchor_domain.py", "G-REFUSE", "REFUSE-raised",
           "        if not lo <= anchor <= hi:\n            raise WindowAnchorDomainError(\n"
           "                f\"window_anchor {anchor} is outside",
           "        if not lo <= anchor <= hi:\n            _MUT_HIT(); return ws, max(lo, min(anchor, hi))\n"
           "            raise WindowAnchorDomainError(\n"
           "                f\"window_anchor {anchor} is outside"),
    "M2": ("utils/window_anchor_domain.py", "G-CAP", "CAP-never-widens",
           "emax = dmax if self.anchor_max_cap is None else min(int(self.anchor_max_cap), dmax)",
           "_MUT_HIT(); emax = dmax if self.anchor_max_cap is None else max(int(self.anchor_max_cap), dmax)"),
    "M3": ("utils/window_anchor_domain.py", "G-RESOLVE", "RESOLVE-proof",
           'n = sum(1 for e in data if e.get("session") in list(opt))',
           '_MUT_HIT(); n = sum(1 for e in data)'),
    "M4": ("window_optimizer_bayesian.py", "G-OBJ-DERIVED", "OBJ-dist-is-derived",
           "offset = trial.suggest_int('offset', _ad_lo, _ad_hi)",
           "_MUT_HIT(); offset = trial.suggest_int('offset', bounds.min_offset, min(bounds.max_offset, _ad_hi))"),
    "M5": ("window_optimizer.py", "G-OBJ-DERIVED", "OBJ-dist-is-derived",
           "                self.optuna_search._window_anchor_domain = self._window_anchor_domain",
           "                _MUT_HIT(); pass"),
    "M6": ("window_optimizer_bayesian.py", "G-PIN", "PIN-past-max",
           "                _anchor_domain.check_window_choice(window_size, offset, _ad_sessions)",
           "                _MUT_HIT()"),
    "M9": ("utils/window_anchor_domain.py", "G-HOLDOUT", "HOLD-max",
           "        if hmax is not None:\n            emax = min(emax, hmax)",
           "        if hmax is not None:\n            _MUT_HIT(); emax = emax"),
    "M10": ("utils/window_anchor_domain.py", "G-HOLDOUT", "HOLD-max",
            "return int(self.n_train[_session_key(sessions)]) - int(window_size)",
            "_MUT_HIT(); return int(self.n_train[_session_key(sessions)]) - int(window_size) + 1"),
    "M11": ("window_optimizer_integration_final.py", "G-NP2-BIND", "NP2-BIND-init",
            "        _wad_domain = None\n",
            "        _MUT_HIT()\n"),
    "M12": ("window_optimizer_integration_final.py", "G-RESULTS-FILE", "RESFILE-order",
            "            _attach_window_anchor_domain_record(results, _wad_domain)\n"
            "            optimizer.save_results(results, output_file)\n",
            "            _MUT_HIT()\n            optimizer.save_results(results, output_file)\n"
            "            _attach_window_anchor_domain_record(results, _wad_domain)\n"),
    "M7": ("utils/step1_search_ledger.py", "G-DEFAULT-SEARCH", "DEFSEARCH-identical",
           "        if _ad_on:\n            space[ANCHOR_DOMAIN_SEARCH_KEY]",
           "        _MUT_HIT()\n        if True:\n            space[ANCHOR_DOMAIN_SEARCH_KEY]"),
}

_MUTANT_CHILD = r'''
import os, sys, importlib, json
root, mid, rel, old, new, gate, marker = sys.argv[1:8]
sys.path.insert(0, root)
modname = rel[:-3].replace("/", ".")
src = open(os.path.join(root, rel)).read()
cnt = src.count(old)
if cnt != 1:
    os.write(1, f"MUTANT NOT APPLIED count={cnt}\n".encode()); sys.exit(3)
import builtins
def _hit():
    open(marker, "a").write("hit\n")
builtins._MUT_HIT = _hit
mutated = src.replace(old, new)
import types
mod = types.ModuleType(modname); mod.__file__ = os.path.join(root, rel)
sys.modules[modname] = mod
if "." in modname:
    pkg = importlib.import_module(modname.rsplit(".", 1)[0]); setattr(pkg, modname.rsplit(".", 1)[1], mod)
mod.__dict__["__mutated_source__"] = mutated
exec(compile(mutated, mod.__file__, "exec"), mod.__dict__)
os.write(1, b"MUTANT APPLIED\n")
sys.argv = [sys.argv[0], "--only", gate]
import runpy
runpy.run_path(os.path.join(root, "tests", "test_window_anchor_brief2.py"), run_name="__main__")
'''


def run_mutants():
    child = os.path.join(SCRATCH, "mutant_child.py")
    open(child, "w").write(_MUTANT_CHILD)
    results = {}
    for mid, (rel, gate, aid, old, new) in MUTANTS.items():
        marker = os.path.join(SCRATCH, f"{mid}.hit")
        if os.path.exists(marker):
            os.remove(marker)
        p = subprocess.run([sys.executable, "-u", child, ROOT, mid, rel, old, new, gate, marker],
                           capture_output=True, text=True, timeout=900)
        applied = "MUTANT APPLIED" in p.stdout
        executed = os.path.exists(marker)
        detected = f"FAIL {gate} [{aid}]" in p.stdout
        status = "DETECTED" if (applied and executed and detected) else \
            ("NOT APPLIED" if not applied else ("NOT EXECUTED" if not executed else "SURVIVED"))
        results[mid] = status
        out(f"  {mid:3s} {status:12s} applied={applied} executed={executed} "
            f"credited={gate}[{aid}] detected={detected}")
        if status != "DETECTED":
            out("     stdout tail: " + p.stdout[-600:].replace("\n", " | "))
            out("     stderr tail: " + p.stderr[-600:].replace("\n", " | "))
    return results


# ---------------------------------------------------------------- driver ----------
def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    os.makedirs(SCRATCH, exist_ok=True)
    # the worktree has no optuna_studies/ (gitignored); storage is checkout-relative
    # (window_optimizer_bayesian.py:23). Created here if absent and removed if this run made it.
    _studies_dir = os.path.join(ROOT, "optuna_studies")
    _made_studies_dir = not os.path.isdir(_studies_dir)
    os.makedirs(_studies_dir, exist_ok=True)
    fx = os.path.join(SCRATCH, "fixture_daily3.json")
    counts = make_fixture(fx)
    created = []
    gates = [
        ("G-AC3", lambda: gate_ac3(fx)),
        ("G-DOMAIN", lambda: gate_domain(fx, counts)),
        ("G-REFUSE", lambda: gate_refuse(fx)),
        ("G-CAP", lambda: gate_cap(fx)),
        ("G-RESOLVE", lambda: gate_resolve(fx, counts)),
        ("G-OBJ-DERIVED", lambda: created.extend(gate_obj_derived(fx, counts))),
        ("G-PIN", lambda: created.extend(gate_pin(fx, counts))),
        ("G-DEFAULT-OBJ", lambda: gate_default_obj(created)),
        ("G-DEFAULT-SEARCH", gate_default_search),
        ("G-ARM", lambda: gate_arm(created)),
        ("G-NP2-BIND", gate_np2_bind),
        ("G-RESULTS-FILE", lambda: gate_results_file(fx)),
        ("G-HOLDOUT", lambda: gate_holdout(fx)),
        ("G-HOLDOUT-OBJ", lambda: created.extend(gate_holdout_obj(fx))),
        ("G-TPE-STORAGE", lambda: created.extend(gate_tpe_storage(fx))),
        ("G-SPLIT-SAME", gate_split_same),
        ("G-ROUTE", gate_route),
        ("G-CLI", gate_cli),
    ]
    passed = failed = 0
    for name, fn in gates:
        if only and name != only:
            continue
        t0 = time.time()
        try:
            if name == "G-RESOLVE":
                # M3's credited assertion: the resolver must not raise on a good fixture
                try:
                    fn()
                except GateFail:
                    raise
                except Exception as e:
                    raise GateFail(f"[RESOLVE-proof] {type(e).__name__}: {e}")
            else:
                fn()
            passed += 1
            out(f"PASS {name} ({time.time() - t0:.1f}s)")
        except GateFail as e:
            failed += 1
            out(f"FAIL {name} {e}")
        except Exception as e:
            failed += 1
            out(f"FAIL {name} [UNEXPECTED] {type(e).__name__}: {e}")
            out(traceback.format_exc()[-1500:])
    # studies this suite created live under the worktree's optuna_studies/: remove them
    for n in set(created) | set(CREATED):
        try:
            os.remove(os.path.join(ROOT, "optuna_studies", n))
        except FileNotFoundError:
            pass
    if _made_studies_dir and not os.listdir(_studies_dir):
        os.rmdir(_studies_dir)
    if only:
        return 1 if failed else 0
    out(f"GATES {passed}/{passed + failed}")
    out("MUTANTS")
    mres = run_mutants()
    det = sum(1 for v in mres.values() if v == "DETECTED")
    out(f"MUTANTS {det}/{len(mres)} DETECTED")
    ok = failed == 0 and det == len(mres)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    out("COMPLETION SENTINEL: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
