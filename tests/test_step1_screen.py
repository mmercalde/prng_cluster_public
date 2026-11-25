#!/usr/bin/env python3
"""Step 1 cheap screen (--step1-objective screen_hits | screen_rank): gates and mutants.

Run: taskset -c 4,5 nice -n 15 python3 -u tests/test_step1_screen.py   (VM 101, ~/venvs/torch, CPU)

Fixtures only. No GPU, no rig, no Step 1 run. Scratch output under
<worktree>/data/test_step1_screen/, removed at the end. Studies the suite creates live in the
worktree's own optuna_studies/ (window_optimizer_bayesian.py:23) and are removed.

Fixture: 2,000 combined rows, evening/midday alternating. Rows [500, 1100) are planted: the draw
at combined index t is seq[t] of seed S_STAR (java_lcg, skip 0) - the Step 6 vote of a survivor
with that seed. Elsewhere the draws are a fixed pattern. Split 1,600 (int(0.8 N)); confirmation
block = [1600, 2000).

Gates
  G-PLANTED        a trial whose survivors' pools hit its selection block scores higher than
                   one whose pools do not, under screen_hits and screen_rank; empty trial = 0
  G-PLANTED-OPT    the real OptunaBayesianSearch with ScreenScorer: the best trial's selection
                   block lies in the planted rows and beats every trial whose block does not
  G-DEFAULT-OBJ    default objective: trial params AND trial user-attr keys of a seeded study
                   identical to the base commit's code (merge-base HEAD main = 41ad509)
  G-DEFAULT-WIRING default: optimize_window keeps BidirectionalCountScorer(), keeps records off
  G-OVERLAP        a confirmation block that a selection block could overlap is refused at
                   load; an overlapping or later selection block is refused per trial
  G-PHASEC         no draw value outside the selection block is read; the block starts after
                   the window's last draw
  G-HOOK           the anchor domain is narrowed so every anchor's block fits; one past is
                   refused (validated, never clamped)
  G-DISTINCT       k_t counts distinct pool values (grader R4)
  G-POOL-IDENTITY  the screen's pool == PredictionGenerator._build_prediction_pool's, with ties
  G-ORDERING       value_scores_for == e2e_vote_harness.py value_scores_for
  G-GRADER-PIN     a grader with another sha256 is refused
  G-STUDY-ARM      a study tagged by one objective or screen setting refuses another
  G-HYBRID         variable-mode records (<base>_hybrid) vote with the base type; others refused
  G-CLI            CLI refusals; G-ROUTE  hops 2 and 3, manifest untouched (hand-landed)
"""
import ast
import inspect
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(ROOT, "data", "test_step1_screen")
sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

GRADER = os.path.expanduser("~/tfm_worktrees/e2e-grader/scripts/diagnostics/e2e_grader.py")
GRADER_SHA = "c7fe48bf102daa42b284713f4b2c2508d7f21c9613a01f063be443c881fd9ace"
HARNESS = os.path.expanduser("~/tfm_worktrees/e2e-vote/scripts/diagnostics/e2e_vote_harness.py")
S_STAR = 987654321
N_ROWS, PLANT = 2000, (500, 1100)
OPTS = [["midday", "evening"], ["midday"], ["evening"]]
CREATED = []


def out(msg):
    os.write(1, (msg + "\n").encode())


class GateFail(AssertionError):
    pass


def check(cond, aid, detail=""):
    if not cond:
        raise GateFail(f"[{aid}] {detail}")


def hit():
    import builtins
    getattr(builtins, "_MUT_HIT", lambda: None)()


def base_commit():
    """The commit the off-path identity is proved against: merge-base of HEAD and main
    (41ad509 for this worktree), NOT HEAD, which may be a WIP snapshot holding this very
    change (b78a15c, 2026-11-23 20:44). Override: TFM_IDENTITY_BASE."""
    env = os.environ.get("TFM_IDENTITY_BASE")
    if env:
        return env
    return subprocess.check_output(["git", "-C", ROOT, "merge-base", "HEAD", "main"]).decode().strip()


# ---------------------------------------------------------------- fixtures --------
def make_fixture(path):
    from survivor_scorer import SurvivorScorer
    import datetime as dt
    seq = SurvivorScorer(prng_type="java_lcg", mod=1000)._generate_sequence(S_STAR, N_ROWS, skip=0)
    d0 = dt.date(2000, 1, 1)
    recs = []
    for i in range(N_ROWS):
        recs.append({"date": (d0 + dt.timedelta(days=i // 2)).isoformat(),
                     "session": "evening" if i % 2 == 0 else "midday",
                     "draw": int(seq[i]) if PLANT[0] <= i < PLANT[1] else (i * 37 + 11) % 1000})
    with open(path, "w") as f:
        json.dump(recs, f)


def cfg(**kw):
    c = {"grader_path": GRADER, "grader_sha256": GRADER_SHA, "selection_draws": 100,
         "rank_test_reps": 500}
    c.update(kw)
    return c


def plan_for(fx, objective="screen_hits", **kw):
    from utils.step1_screen import build_screen_plan
    from window_optimizer import step1_holdout_split_point
    return build_screen_plan(objective, cfg(**kw), fx, OPTS, split_point_fn=step1_holdout_split_point)


def rec(seed, fm=0.5, rm=0.5):
    return {"seed": int(seed), "forward_match_rate": fm, "reverse_match_rate": rm,
            "prng_type": "java_lcg"}


def bad_records(n=40, seed=1):
    r = random.Random(seed)
    return [rec(r.randrange(2 ** 32)) for _ in range(n)]


def good_records(n=40):
    return bad_records(n - 1) + [rec(S_STAR, 0.9, 0.9)]


def domain_with_hook(fx, plan):
    import dataclasses
    from utils.window_anchor_domain import resolve_anchor_domain
    from window_optimizer import step1_holdout_split_point
    d = resolve_anchor_domain(fx, OPTS, window_size_min=6, window_size_max=50,
                              bounded_anchor_max=100, anchor_bounds={"min": 0, "max_cap": None},
                              holdout_guard=True, holdout_split_point_fn=step1_holdout_split_point)
    return d, dataclasses.replace(d, anchor_max_hook=plan.anchor_max)


# ---------------------------------------------------------------- gates -----------
def gate_planted(fx):
    for obj in ("screen_hits", "screen_rank"):
        p = plan_for(fx, obj)
        # all-sessions window of 10 ending at 599: block [600, 700), inside the planted rows
        vg, rg = p.score(good_records(), 590, 10, ["midday", "evening"])
        vb, rb = p.score(bad_records(), 590, 10, ["midday", "evening"])
        ve, re_ = p.score([], 590, 10, ["midday", "evening"])
        check(rg["selection_block"] == [600, 700], "PL-block", str(rg["selection_block"]))
        check(vg > vb, "PL-good-beats-bad", f"{obj}: good {vg} bad {vb}")
        check(ve == 0.0 and re_["hits"] == 0 and re_["expected_hits"] == 0.0, "PL-empty-zero",
              f"{obj}: {ve} {re_}")
        check(rg["hits"] == 100 and rg["mode"] == "development", "PL-hits", str(rg))


def gate_hybrid(fx):
    """Variable-mode records (<base>_hybrid) are voted with the base type, as Step 6 does;
    another generator is refused."""
    from utils.step1_screen import Step1ScreenError
    p = plan_for(fx)
    g = [dict(r, prng_type="java_lcg_hybrid") if i % 2 else r for i, r in enumerate(good_records())]
    v, info = p.score(g, 590, 10, ["midday", "evening"])
    check(info["record_prng_types"] == {"java_lcg": 20, "java_lcg_hybrid": 20}
          and info["hits"] == 100, "HYB-accepted", str(info))
    try:
        p.score([dict(good_records()[0], prng_type="xorshift32")], 590, 10, ["midday", "evening"])
        raise GateFail("[HYB-other-refused] xorshift32 record accepted")
    except Step1ScreenError:
        pass


def _stub_objective(calls, fx_plan):
    """A fake sieve: a window whose selection block lies in the planted rows 'finds' S_STAR.
    Every trial reports the same survivor count, so survivor count cannot tell them apart."""
    from window_optimizer_bayesian import OptimizationResult

    def f(config, optuna_trial=None):
        s0, s1 = fx_plan.selection_block(config.offset, config.window_size, config.sessions)
        planted = PLANT[0] <= s0 and s1 <= PLANT[1]
        r = OptimizationResult(config=config, forward_count=40, reverse_count=40,
                               bidirectional_count=40)
        r.screen_records = good_records() if planted else bad_records(seed=config.offset)
        calls.append((config.window_size, config.offset, list(config.sessions), planted))
        return r
    return f


def _snap():
    d = os.path.join(ROOT, "optuna_studies")
    return set(os.listdir(d)) if os.path.isdir(d) else set()


def gate_planted_opt(fx):
    import optuna
    from window_optimizer import SearchBounds, ScreenScorer
    import window_optimizer_bayesian as wob
    plan = plan_for(fx, "screen_hits")
    _, dom = domain_with_hook(fx, plan)
    b = SearchBounds(); b.min_window_size, b.max_window_size = 6, 12
    b.session_options = [["midday", "evening"]]
    s = wob.OptunaBayesianSearch(n_startup_trials=6, seed=7)
    s._window_anchor_domain = dom
    calls = []
    time.sleep(1.1)
    before = _snap()
    cwd = os.getcwd(); os.chdir(SCRATCH)
    try:
        s.search(_stub_objective(calls, plan), b, 24, ScreenScorer(plan),
                 trse_context_file=os.path.join(SCRATCH, "none.json"))
    finally:
        os.chdir(cwd)
        new = sorted(_snap() - before)
        CREATED.extend(new)
    check(len(new) == 1, "OPT-one-study", str(new))
    st = optuna.load_study(study_name=new[0][:-3], storage=f"sqlite:///{ROOT}/optuna_studies/{new[0]}")
    done = [t for t in st.trials if t.state.name == "COMPLETE"]
    check(all("step1_screen" in t.user_attrs for t in done), "OPT-attr", "")
    pl = [t for t in done if PLANT[0] <= t.user_attrs["step1_screen"]["selection_block"][0]
          and t.user_attrs["step1_screen"]["selection_block"][1] <= PLANT[1]]
    un = [t for t in done if t not in pl]
    check(pl and un, "OPT-both-kinds", f"planted {len(pl)} unplanted {len(un)} of {len(done)}")
    best = max(done, key=lambda t: t.value)
    check(best in pl and best.value > max(t.value for t in un), "OPT-best-in-planted",
          f"best {best.params} {best.value}; max unplanted {max(t.value for t in un)}")


_DEFAULT_OBJ_CHILD = r'''
import json, os, sys
root, scratch = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(scratch, "head_src"))
sys.path.insert(1, root)
os.chdir(scratch)
import window_optimizer_bayesian as wob
from window_optimizer import SearchBounds
b = SearchBounds(); b.min_window_size, b.max_window_size = 6, 50
s = wob.OptunaBayesianSearch(n_startup_trials=3, seed=11)
seen = []
def f(config, optuna_trial=None):
    seen.append([config.window_size, config.offset, list(config.sessions)])
    return wob.OptimizationResult(config=config, forward_count=1, reverse_count=1,
                                  bidirectional_count=(config.offset % 7) + 1)
r = s.search(f, b, 8, wob.ResultScorer(), trse_context_file=os.path.join(scratch, "none.json"))
os.write(1, ("RESULT " + json.dumps({"seen": seen, "file": wob.__file__}) + "\n").encode())
'''


def _default_obj_live(scratch):
    import window_optimizer_bayesian as wob
    from window_optimizer import SearchBounds
    hit()
    b = SearchBounds(); b.min_window_size, b.max_window_size = 6, 50
    s = wob.OptunaBayesianSearch(n_startup_trials=3, seed=11)
    seen = []

    def f(config, optuna_trial=None):
        seen.append([config.window_size, config.offset, list(config.sessions)])
        return wob.OptimizationResult(config=config, forward_count=1, reverse_count=1,
                                      bidirectional_count=(config.offset % 7) + 1)
    cwd = os.getcwd(); os.chdir(scratch)
    try:
        s.search(f, b, 8, wob.ResultScorer(), trse_context_file=os.path.join(scratch, "none.json"))
    finally:
        os.chdir(cwd)
    return seen


def _attr_keys(name):
    import optuna
    st = optuna.load_study(study_name=os.path.basename(name)[:-3], storage=f"sqlite:///{name}")
    return [sorted(t.user_attrs) for t in st.trials]


def gate_default_obj():
    hs = os.path.join(SCRATCH, "head_src")
    os.makedirs(os.path.join(hs, "optuna_studies"), exist_ok=True)
    for rel in ("window_optimizer.py", "window_optimizer_bayesian.py"):
        src = subprocess.check_output(["git", "-C", ROOT, "show", f"{base_commit()}:{rel}"]).decode()
        open(os.path.join(hs, rel), "w").write(src)
        check(src != open(os.path.join(ROOT, rel)).read(), "DEFOBJ-nonvacuous",
              f"{rel}: base {base_commit()} equals the live file, so identity would prove nothing")
    child = os.path.join(SCRATCH, "default_obj_child.py")
    open(child, "w").write(_DEFAULT_OBJ_CHILD)
    hb = set(os.listdir(os.path.join(hs, "optuna_studies")))
    p = subprocess.run([sys.executable, "-u", child, ROOT, SCRATCH], capture_output=True,
                       text=True, timeout=600)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    check(p.returncode == 0 and line, "DEFOBJ-head-ran", p.stderr[-800:])
    head = json.loads(line[-1][7:])
    check(head["file"].startswith(hs), "DEFOBJ-which-code", head["file"])
    hnew = sorted(set(os.listdir(os.path.join(hs, "optuna_studies"))) - hb)
    time.sleep(1.1)
    before = _snap()
    live_seen = _default_obj_live(SCRATCH)
    lnew = sorted(_snap() - before)
    CREATED.extend(lnew)
    check(len(hnew) == 1 and len(lnew) == 1, "DEFOBJ-studies", f"{hnew} {lnew}")
    check(head["seen"] == live_seen, "DEFOBJ-identical", f"head={head['seen']} live={live_seen}")
    hk = _attr_keys(os.path.join(hs, "optuna_studies", hnew[0]))
    lk = _attr_keys(os.path.join(ROOT, "optuna_studies", lnew[0]))
    check(hk == lk, "DEFOBJ-attrs-identical", f"head={hk[:2]} live={lk[:2]}")


def _optimize_window_src():
    import window_optimizer_integration_final as wif
    src = getattr(wif, "__mutated_source__", None)
    if src is not None:
        hit()
    else:
        src = open(os.path.join(ROOT, "window_optimizer_integration_final.py")).read()
    return src


def gate_default_wiring():
    src = _optimize_window_src()
    tree = ast.parse(src)
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "optimize_window"][0]
    inits = [st for st in fn.body if isinstance(st, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_s1_screen_plan" for t in st.targets)
             and isinstance(st.value, ast.Constant) and st.value.value is None]
    check(len(inits) == 1, "WIRE-init", f"{len(inits)} function-level `_s1_screen_plan = None`")
    scorers = [k.value for n in ast.walk(fn) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", "") == "optimize" for k in n.keywords if k.arg == "scorer"]
    check(len(scorers) == 1, "WIRE-scorer-one", str(len(scorers)))
    sc = scorers[0]
    ok = (isinstance(sc, ast.IfExp) and isinstance(sc.test, ast.Compare)
          and ast.unparse(sc.test) == "_s1_screen_plan is None"
          and ast.unparse(sc.body) == "BidirectionalCountScorer()")
    check(ok, "WIRE-scorer-default", ast.unparse(sc))
    f2 = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
          and n.name == "_build_test_result_from_miner"][0]
    kw = {a.arg: d for a, d in zip(f2.args.args[-len(f2.args.defaults):], f2.args.defaults)}
    check("keep_screen_records" in kw and ast.unparse(kw["keep_screen_records"]) == "False",
          "WIRE-records-default-off", str(list(kw)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_build_test_result_from_miner"]
    check(len(calls) == 1 and all(k.arg != "keep_screen_records" for k in calls[0].keywords)
          and "_step1_screen_on" in ast.unparse(calls[0]), "WIRE-records-only-when-on",
          ast.unparse(calls[0])[-200:] if calls else "none")


def gate_overlap(fx):
    from utils.step1_screen import Step1ScreenError
    try:
        plan_for(fx, confirmation_blocks=[[1500, 2000]])
        raise GateFail("[OVL-load] confirmation block [1500, 2000) before end_max 1600 accepted")
    except Step1ScreenError as e:
        check("starts before selection_end_max" in str(e), "OVL-load", str(e))
    p = plan_for(fx)
    p.selection_end_max = 2000            # simulate a block the load rule would not allow
    for blk, msg in (((1590, 1690), "overlaps confirmation block"),):
        try:
            p.check_block(blk[0], blk[1], blk[0] - 1)
            raise GateFail(f"[OVL-check-block] {blk} accepted")
        except Step1ScreenError as e:
            check(msg in str(e), "OVL-check-block", str(e))
    p2 = plan_for(fx, confirmation_blocks=[[1700, 2000]])
    p2.selection_end_max = 2000
    try:
        p2.check_block(1750 - 100, 1750, 1500)
        raise GateFail("[OVL-check-block] block ending after a confirmation start accepted")
    except Step1ScreenError as e:
        check("overlaps" in str(e) or "ends after the start" in str(e), "OVL-check-block", str(e))


class _SealedDraws(list):
    def __init__(self, data, lo, hi):
        super().__init__(data); self.lo, self.hi, self.reads = lo, hi, []

    def __getitem__(self, i):
        self.reads.append(i)
        if not (self.lo <= i < self.hi):
            raise GateFail(f"[PC-only-block] draw {i} read outside [{self.lo}, {self.hi})")
        return list.__getitem__(self, i)


def gate_phasec(fx):
    from utils.step1_screen import Step1ScreenError
    for obj in ("screen_hits", "screen_rank"):
        p = plan_for(fx, obj)
        c_first, c_last = p.window_combined(590, 10, ["midday", "evening"])
        try:
            s0, s1 = p.selection_block(590, 10, ["midday", "evening"])
        except Step1ScreenError as e:
            raise GateFail(f"[PC-after-window] selection_block refused its own block: {e}")
        check(s0 == c_last + 1, "PC-after-window", f"s0 {s0} window last {c_last}")
        for t in (s0 - 1, s1, c_last):          # the guard itself, on the unsealed list
            try:
                p._open_block = (s0, s1)
                p.draw_value(t)
                raise GateFail(f"[PC-draw-guard] draw_value({t}) outside [{s0}, {s1}) accepted")
            except Step1ScreenError:
                pass
            finally:
                p._open_block = None
        p._draws = _SealedDraws(p._draws, s0, s1)
        p.score(good_records(), 590, 10, ["midday", "evening"])
        check(len(p._draws.reads) == s1 - s0, "PC-only-block", f"{len(p._draws.reads)} reads")
    p = plan_for(fx)
    try:
        p.draw_value(650)
        raise GateFail("[PC-draw-guard] draw_value with no open block accepted")
    except Step1ScreenError:
        pass


def gate_hook(fx):
    from utils.window_anchor_domain import WindowAnchorDomainError
    plan = plan_for(fx)
    base, dom = domain_with_hook(fx, plan)
    for opt in OPTS:
        for w in (6, 10, 20):
            lo, hi = dom.effective_range(w, opt)
            _, bhi = base.effective_range(w, opt)
            check(hi == min(bhi, plan.anchor_max(w, opt)) and hi < bhi, "HOOK-max",
                  f"{opt} w{w}: hooked {hi} guard-only {bhi} plan {plan.anchor_max(w, opt)}")
            plan.selection_block(hi, w, opt)          # fits
            try:
                plan.selection_block(hi + 1, w, opt)
                raise GateFail(f"[HOOK-edge] {opt} w{w} anchor {hi + 1} block accepted")
            except Exception as e:
                check(not isinstance(e, GateFail), "HOOK-edge", str(e))
            try:
                dom.check_window_choice(w, hi + 1, opt)
                raise GateFail(f"[HOOK-max] pinned anchor {hi + 1} accepted")
            except WindowAnchorDomainError as e:
                check("never clamped" in str(e), "HOOK-max", str(e))
    check(base.anchor_max_hook is None and base.trial_record(10, OPTS[0]).get("screen_anchor_max")
          is None, "HOOK-off-unchanged", "")


def gate_distinct(fx):
    p = plan_for(fx)
    r = [rec(S_STAR + 1, 0.9, 0.9)] * 20 + bad_records(20)      # top 20 all the same seed
    v, info = p.score(r, 590, 10, ["midday", "evening"])
    check(abs(info["expected_hits"] - info["graded_draws"] * 1 / 1000) < 1e-9, "DIST-expected",
          f"expected {info['expected_hits']} for {info['graded_draws']} draws with k_t = 1")


def gate_pool_identity(fx):
    import types
    import numpy as np
    from prediction_generator import PredictionGenerator
    p = plan_for(fx, pool_size=10)
    rnd = random.Random(5)
    rows = [rec(rnd.randrange(2 ** 32), fm, fm) for fm in (0.5, 0.5, 0.5, 0.7, 0.7, 0.3) * 4]
    order = p.rank_order(rows)
    from survivor_scorer import SurvivorScorer
    sc = SurvivorScorer(prng_type="java_lcg", mod=1000)
    for t in (650, 777):
        mine = [int(sc._generate_sequence(rows[i]["seed"], t + 1, skip=0)[t]) for i in order[:10]]
        pos = np.empty(len(rows)); pos[order] = np.arange(len(rows), 0, -1)   # distinct positions

        class M:
            def predict(self, X):
                return pos.copy()
        stub = types.SimpleNamespace(
            config=types.SimpleNamespace(require_feature_schema=False, default_skip=0),
            feature_names=["forward_match_rate"], model=M(), model_meta={"model_type": "stub"},
            scorer=sc, logger=types.SimpleNamespace(debug=lambda *a: None, warning=lambda *a: None),
            _normalize_features=False, _scaler_mean=None)
        stub._get_survivor_skip = types.MethodType(PredictionGenerator._get_survivor_skip, stub)
        stub._empty_pool_result = lambda: {"predictions": []}
        res = PredictionGenerator._build_prediction_pool(
            stub, [dict(r, features={"forward_match_rate": r["forward_match_rate"]}) for r in rows],
            [0] * t, 10)
        check(res["predictions"] == mine, "POOL-equal", f"t={t} module {res['predictions']} screen {mine}")
        # tie order: equal keys ordered by ascending seed (run H A0 tie_break seed_asc)
        keys = [rows[i]["forward_match_rate"] for i in order]
        seeds = [rows[i]["seed"] for i in order]
        check(all(keys[i] > keys[i + 1] or (keys[i] == keys[i + 1] and seeds[i] < seeds[i + 1])
                  for i in range(len(order) - 1)), "POOL-equal", "tie order not seed_asc")


def gate_ordering():
    import importlib.util
    import numpy as np
    from utils.step1_screen import value_scores_for, sha256_file
    check(os.path.isfile(HARNESS), "ORD-harness-present", HARNESS)
    spec = importlib.util.spec_from_file_location("_h6_for_test", HARNESS)
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    rng = np.random.default_rng(3)
    for n in (0, 1, 7, 300):
        col = rng.integers(0, 1000, size=n).astype(np.int16)
        if n == 0:
            continue
        a = value_scores_for(col, 1000); b = h.value_scores_for(col, 1000)
        check(a.tolist() == b.tolist(), "ORD-equal", f"n={n}")
    out(f"     (harness {HARNESS} sha256 {sha256_file(HARNESS)})")


def gate_grader_pin(fx):
    from utils.step1_screen import Step1ScreenError
    try:
        plan_for(fx, grader_sha256="0" * 64)
        raise GateFail("[PIN-refused] wrong grader sha256 accepted")
    except Step1ScreenError as e:
        check("!= pinned" in str(e), "PIN-refused", str(e))


def gate_study_arm():
    import optuna
    from utils.step1_screen import screen_study_check, Step1ScreenError
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    st = optuna.create_study(direction="maximize")
    screen_study_check(st, "screen_hits:aaa", 0)
    check(st.user_attrs.get("step1_objective") == "screen_hits:aaa", "ARM-tag", str(st.user_attrs))
    for other in (None, "screen_hits:bbb", "screen_rank:aaa"):
        try:
            screen_study_check(st, other, 1)
            raise GateFail(f"[ARM-mismatch] tagged study accepted run tag {other!r}")
        except Step1ScreenError:
            pass
    st2 = optuna.create_study(direction="maximize")
    st2.enqueue_trial({"x": 1}); st2.optimize(lambda t: t.suggest_int("x", 0, 2), n_trials=1)
    screen_study_check(st2, None, 1)              # default run on an untagged study: unchanged
    check("step1_objective" not in st2.user_attrs, "ARM-default-untouched", str(st2.user_attrs))
    try:
        screen_study_check(st2, "screen_hits:aaa", 1)
        raise GateFail("[ARM-mismatch] screen run on an untagged study with trials accepted")
    except Step1ScreenError:
        pass


def gate_cli():
    for flags, aid, msg in (
            (["--step1-objective", "screen_hits", "--use-range-miner"], "CLI-needs-derived",
             "needs --derived-anchor-domain and --anchor-holdout-guard"),
            (["--step1-objective", "screen_rank", "--use-range-miner", "--derived-anchor-domain",
              "--anchor-holdout-guard", "--strategy", "bayesian"], "CLI-needs-config",
             "needs --screen-config"),
            (["--screen-config", "x.json", "--use-range-miner"], "CLI-config-alone",
             "--screen-config is read only with a screen")):
        p = subprocess.run([sys.executable, "-u", os.path.join(ROOT, "window_optimizer.py")]
                           + flags + ["--lottery-file", "nope.json"],
                           capture_output=True, text=True, timeout=300, cwd=SCRATCH)
        check(p.returncode == 2 and msg in p.stderr, aid, f"rc={p.returncode} {p.stderr[-500:]}")


def gate_route():
    import window_optimizer as wo
    sig = inspect.signature(wo.run_bayesian_optimization)
    for name, default in (("step1_objective", "bidirectional_count"), ("screen_config", None)):
        p = sig.parameters.get(name)
        check(p is not None and p.default == default and p.kind == p.KEYWORD_ONLY,
              "ROUTE-hop2-sig", f"{name} {p}")
    src = open(os.path.join(ROOT, "window_optimizer_integration_final.py")).read()
    fn = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)
          and n.name == "optimize_window"][0]
    args = {a.arg: d for a, d in zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults)}
    check(ast.unparse(args.get("step1_objective", ast.Constant(None))) == "'bidirectional_count'",
          "ROUTE-hop3-sig", str(list(args)[-3:]))
    m = json.load(open(os.path.join(ROOT, "agent_manifests", "window_optimizer.json")))
    check("step1_objective" not in m.get("default_params", {}), "ROUTE-manifest-untouched",
          "the manifest is hand-landed; its entries are a draft diff only")


# ---------------------------------------------------------------- mutants ---------
MUTANTS = {
    "S1": ("utils/step1_screen.py", "G-OVERLAP", "OVL-check-block",
           "            if s0 < b1 and b0 < s1:\n                raise",
           "            if s0 < b1 and b0 < s1 and _MUT_HIT():\n                raise"),
    "S2": ("utils/step1_screen.py", "G-OVERLAP", "OVL-load",
           "        if b[0] < end_max:\n", "        if b[0] < end_max and _MUT_HIT():\n"),
    "S3": ("utils/step1_screen.py", "G-PHASEC", "PC-after-window",
           'self.settings["selection_gap"]\n        s1 = s0',
           'self.settings["selection_gap"] - 1 - int(bool(_MUT_HIT()))\n        s1 = s0'),
    "S4": ("utils/step1_screen.py", "G-PHASEC", "PC-draw-guard",
           "        if blk is None or not (blk[0] <= t < blk[1]):\n",
           "        if _MUT_HIT() and (blk is None or not (blk[0] <= t < blk[1])):\n"),
    "S5": ("utils/window_anchor_domain.py", "G-HOOK", "HOOK-max",
           "            emax = min(emax, smax)\n", "            _MUT_HIT(); emax = emax\n"),
    "S6": ("utils/step1_screen.py", "G-DISTINCT", "DIST-expected",
           "                kt = len(distinct)\n", "                _MUT_HIT(); kt = len(raw_pool)\n"),
    "S7": ("utils/step1_screen.py", "G-POOL-IDENTITY", "POOL-equal",
           "key=lambda i: (-keys[i], seeds[i]))", "key=lambda i: (_MUT_HIT() or -keys[i], -seeds[i]))"),
    "S8": ("window_optimizer.py", "G-PLANTED-OPT", "OPT-best-in-planted",
           "        result.screen_record = rec\n        return value\n",
           "        result.screen_record = rec\n        _MUT_HIT(); return float(result.bidirectional_count)\n"),
    "S9": ("utils/step1_screen.py", "G-GRADER-PIN", "PIN-refused",
           "    if got != str(expected_sha256).lower():\n",
           "    if got != str(expected_sha256).lower() and _MUT_HIT():\n"),
    "S10": ("window_optimizer_bayesian.py", "G-DEFAULT-OBJ", "DEFOBJ-attrs-identical",
            "            if getattr(result, 'screen_record', None) is not None:\n"
            "                trial.set_user_attr(\"step1_screen\", result.screen_record)\n",
            "            _MUT_HIT()\n"
            "            trial.set_user_attr(\"step1_screen\", getattr(result, 'screen_record', None))\n"),
    "S11": ("window_optimizer_integration_final.py", "G-DEFAULT-WIRING", "WIRE-scorer-default",
            "scorer=(BidirectionalCountScorer() if _s1_screen_plan is None",
            "scorer=(BidirectionalCountScorer() if _s1_screen_plan is not None"),
    "S13": ("utils/step1_screen.py", "G-HYBRID", "HYB-other-refused",
            "            if pt is not None and pt not in allowed:\n",
            "            if pt is not None and pt not in allowed and _MUT_HIT():\n"),
    "S12": ("utils/step1_screen.py", "G-STUDY-ARM", "ARM-mismatch",
            "    if existing != run_tag:\n", "    if existing != run_tag and _MUT_HIT():\n"),
}

_MUTANT_CHILD = r'''
import os, sys, importlib, types, builtins, runpy
root, mid, rel, old, new, gate, marker = sys.argv[1:8]
sys.path.insert(0, root)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
modname = rel[:-3].replace("/", ".")
src = open(os.path.join(root, rel)).read()
cnt = src.count(old)
if cnt != 1:
    os.write(1, f"MUTANT NOT APPLIED count={cnt}\n".encode()); sys.exit(3)
def _hit():
    open(marker, "a").write("hit\n")
    return False
builtins._MUT_HIT = _hit
mutated = src.replace(old, new)
mod = types.ModuleType(modname); mod.__file__ = os.path.join(root, rel)
mod.__dict__["__mutated_source__"] = mutated
sys.modules[modname] = mod
if "." in modname:
    pkg = importlib.import_module(modname.rsplit(".", 1)[0]); setattr(pkg, modname.rsplit(".", 1)[1], mod)
if rel == "window_optimizer_integration_final.py":
    # structural gate: the gate reads the mutated text (and calls _MUT_HIT itself)
    mod.__dict__["__name__"] = modname
else:
    exec(compile(mutated, mod.__file__, "exec"), mod.__dict__)
os.write(1, b"MUTANT APPLIED\n")
sys.argv = [sys.argv[0], "--only", gate]
runpy.run_path(os.path.join(root, "tests", "test_step1_screen.py"), run_name="__main__")
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
            out("     stdout tail: " + p.stdout[-700:].replace("\n", " | "))
            out("     stderr tail: " + p.stderr[-700:].replace("\n", " | "))
    return results


# ---------------------------------------------------------------- driver ----------
def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    os.makedirs(SCRATCH, exist_ok=True)
    studies = os.path.join(ROOT, "optuna_studies")
    made_studies = not os.path.isdir(studies)
    os.makedirs(studies, exist_ok=True)
    fx = os.path.join(SCRATCH, "fixture_daily3.json")
    if not os.path.exists(fx):
        make_fixture(fx)
    gates = [
        ("G-PLANTED", lambda: gate_planted(fx)),
        ("G-HYBRID", lambda: gate_hybrid(fx)),
        ("G-PLANTED-OPT", lambda: gate_planted_opt(fx)),
        ("G-DEFAULT-OBJ", gate_default_obj),
        ("G-DEFAULT-WIRING", gate_default_wiring),
        ("G-OVERLAP", lambda: gate_overlap(fx)),
        ("G-PHASEC", lambda: gate_phasec(fx)),
        ("G-HOOK", lambda: gate_hook(fx)),
        ("G-DISTINCT", lambda: gate_distinct(fx)),
        ("G-POOL-IDENTITY", lambda: gate_pool_identity(fx)),
        ("G-ORDERING", gate_ordering),
        ("G-GRADER-PIN", lambda: gate_grader_pin(fx)),
        ("G-STUDY-ARM", gate_study_arm),
        ("G-CLI", gate_cli),
        ("G-ROUTE", gate_route),
    ]
    passed = failed = 0
    for name, fn in gates:
        if only and name != only:
            continue
        t0 = time.time()
        try:
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
    for n in set(CREATED):
        try:
            os.remove(os.path.join(studies, n))
        except FileNotFoundError:
            pass
    if made_studies and not os.listdir(studies):
        os.rmdir(studies)
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
