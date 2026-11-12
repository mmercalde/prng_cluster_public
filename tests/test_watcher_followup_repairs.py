#!/usr/bin/env python3
"""WATCHER follow-up repairs F-1 / F-2 — gate battery (2026-11-11).

Brief: docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_FOLLOWUP_REPAIRS_20261111.md

  F-1  run_pipeline stops cleanly (no evaluate/escalate, no halt file, no
       advance) when run_step returns seed_domain_complete is True.
  F-2  dispatch_learning_loop stops on results["success"] is False, and its
       heuristic path reads AgentDecision.recommended_action (not the
       non-existent `.action`) and records STEP_HALTED on a non-proceed verdict.

Safety: no fleet, no pipeline, no subprocess. run_step is stubbed on every
path; check_safety, notify_telegram, the dispatch decision log, the LLM bundle
builder and the LLM evaluator are replaced by recorders. Halt files live in a
temp directory; /tmp/agent_halt is never created (HYG gate checks).

RED arms execute pre-repair source from BASE (git show); MUT arms execute a
mutated copy of the LIVE source. Both are compiled into a copy of the
production module's globals. A mutant whose marker is not found is an ERROR,
never a detection; every mutant carries a hit sentinel proving it ran.

Prints one line per gate and `COMPLETION SENTINEL: PASS` only if every gate ran
and passed.
"""

import ast
import contextlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import agents.watcher_agent as W      # noqa: E402
import agents.watcher_dispatch as D   # noqa: E402
from agents.agent_decision import AgentDecision  # noqa: E402

BASE = "b6cb789"
PROD_HALT = "/tmp/agent_halt"   # observed only, never created
WA_PATH = "agents/watcher_agent.py"
WD_PATH = "agents/watcher_dispatch.py"

RESULTS = []


class Unavailable(Exception):
    pass


class MutantNotApplied(Exception):
    pass


def record(name, verdict, detail=""):
    RESULTS.append((name, verdict, detail))
    print(f"  [{verdict:<11}] {name}  — {detail}", flush=True)


def run_gate(name, fn):
    try:
        ok, detail = fn()
        record(name, "PASS" if ok else "FAIL", detail)
    except Unavailable as e:
        record(name, "UNAVAILABLE", str(e))
    except MutantNotApplied as e:
        record(name, "ERROR", f"mutant not applied: {e}")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        record(name, "ERROR", f"{type(e).__name__}: {e} [{tb}]")


# ─────────────────────────────────────────────────────────────────────────────
# Source helpers
# ─────────────────────────────────────────────────────────────────────────────

_GIT_CACHE = {}


def git_show(path):
    if path not in _GIT_CACHE:
        r = subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise Unavailable(f"git show {BASE}:{path} failed: {r.stderr.strip()[:200]}")
        _GIT_CACHE[path] = r.stdout
    return _GIT_CACHE[path]


def live_source(path):
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


def method_source(module_src, name, cls="WatcherAgent"):
    tree = ast.parse(module_src)
    c = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
    if len(c) != 1:
        raise Unavailable(f"class {cls} not found exactly once")
    f = [n for n in c[0].body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(f) != 1:
        raise Unavailable(f"{cls}.{name} not found exactly once")
    return textwrap.dedent(ast.get_source_segment(module_src, f[0]))


def exec_in_copy(src, base_globals, filename):
    ns = dict(base_globals)
    ns["_MUT_HITS"] = []
    exec(compile(src, filename, "exec"), ns)
    return ns


def mutate_block(src, begin, end, stmt):
    lines = src.splitlines(keepends=True)
    b = [i for i, ln in enumerate(lines) if begin in ln]
    e = [i for i, ln in enumerate(lines) if end in ln]
    if len(b) != 1 or len(e) != 1 or e[0] <= b[0]:
        raise MutantNotApplied(f"markers {begin}/{end}: begin={b} end={e}")
    ind = lines[b[0]][:len(lines[b[0]]) - len(lines[b[0]].lstrip())]
    return "".join(lines[:b[0]] + [f"{ind}{stmt}\n"] + lines[e[0] + 1:])


def mutate_line(src, marker, new_lines):
    lines = src.splitlines(keepends=True)
    hit = [i for i, ln in enumerate(lines) if marker in ln]
    if len(hit) != 1:
        raise MutantNotApplied(f"marker {marker}: found on {len(hit)} lines")
    ind = lines[hit[0]][:len(lines[hit[0]]) - len(lines[hit[0]].lstrip())]
    repl = [f"{ind}{ln}\n" for ln in new_lines]
    return "".join(lines[:hit[0]] + repl + lines[hit[0] + 1:])


@contextlib.contextmanager
def patched(ns, **repl):
    missing = object()
    saved = {k: ns.get(k, missing) for k in repl}
    ns.update(repl)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is missing:
                ns.pop(k, None)
            else:
                ns[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

TMP = tempfile.mkdtemp(prefix="watcher_followup_repairs_")
TMP_HALT = os.path.join(TMP, "agent_halt_fixture")
MAN_DIR = os.path.join(TMP, "manifests")
OUT_DIR = os.path.join(TMP, "outputs")
os.makedirs(MAN_DIR)
os.makedirs(OUT_DIR)

VALID = json.dumps({"fixture": "watcher_followup_repairs", "pad": "x" * 128})

for _step in (1, 2, 3, 5):
    _name = W.STEP_MANIFESTS[_step]
    with open(os.path.join(ROOT, "agent_manifests", _name)) as _f:
        _m = json.load(_f)
    _out = os.path.join(OUT_DIR, f"step{_step}_out.json")
    _m["success_condition"] = [_out]
    with open(os.path.join(MAN_DIR, _name), "w") as _f:
        json.dump(_m, _f, indent=2)
    with open(_out, "w") as _f:
        _f.write(VALID)


def sdc_results():
    """The §6 COMPLETE dict, keys exactly as run_step builds them at
    agents/watcher_agent.py (S145 block, `if _cursor.is_complete:`)."""
    msg = ("SEED_DOMAIN_COMPLETE: the certified coverage union for java_lcg covers the "
           "governed domain [0, 4294967296) exactly (4,294,967,296 seeds across 3 "
           "certified interval(s)). The sweep is finished; there is no next seed_start. "
           "No Step 1 run was generated.")
    return {"success": False, "error": msg, "blocked_by": "seed_domain_complete",
            "seed_domain_complete": True, "next_seed_start": None}


def make_agent(use_llm=False):
    cfg = W.WatcherConfig(
        use_llm=False,
        manifests_dir=MAN_DIR,
        history_file=os.path.join(TMP, "watcher_history.json"),
        log_file=os.path.join(TMP, "watcher_decisions.jsonl"),
        pid_file=os.path.join(TMP, "watcher_daemon.pid"),
        daemon_state_file=os.path.join(TMP, "daemon_state.json"),
        pending_approval_file=os.path.join(TMP, "pending_approval.json"),
        halt_file=TMP_HALT,
    )
    agent = W.WatcherAgent(cfg)
    agent.llm_lifecycle = None
    agent.llm_router = None
    agent.config.use_llm = use_llm
    return agent


def clear_halt():
    if os.path.exists(TMP_HALT):
        os.remove(TMP_HALT)


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, r):
        self.records.append((r.levelname, r.getMessage()))


# ─────────────────────────────────────────────────────────────────────────────
# F-1
# ─────────────────────────────────────────────────────────────────────────────

def drive_pipeline(rp_fn, stub_results, start=1, end=1):
    """Run `rp_fn(agent, start, end)` with run_step stubbed; observe everything."""
    clear_halt()
    agent = make_agent()
    obs = {"run_step": [], "evaluate": 0, "execute": 0, "escalate": 0, "telegram": []}

    def run_step(step, params=None, **kw):
        obs["run_step"].append(step)
        r = stub_results(step)
        return dict(r) if isinstance(r, dict) else r

    orig_eval, orig_exec, orig_esc = agent.evaluate_results, agent.execute_decision, agent._handle_escalate

    def ev(*a, **k):
        obs["evaluate"] += 1
        return orig_eval(*a, **k)

    def ex(*a, **k):
        obs["execute"] += 1
        return orig_exec(*a, **k)

    def es(*a, **k):
        obs["escalate"] += 1
        return orig_esc(*a, **k)

    agent.run_step, agent.evaluate_results = run_step, ev
    agent.execute_decision, agent._handle_escalate = ex, es
    tg = lambda msg, *a, **k: obs["telegram"].append(msg)  # noqa: E731
    cap = LogCapture()
    W.logger.addHandler(cap)
    try:
        with patched(W.__dict__, check_safety=lambda: True, notify_telegram=tg,
                     PROGRESS_DISPLAY_AVAILABLE=False), \
                contextlib.redirect_stdout(io.StringIO()):
            ret = rp_fn(agent, start, end)
    finally:
        W.logger.removeHandler(cap)
    obs["halt_file"] = os.path.exists(TMP_HALT)
    obs["current_step"] = agent.current_step
    obs["pipeline_running"] = agent._pipeline_running
    obs["returned"] = ret
    obs["critical"] = sum(1 for m in obs["telegram"] if "CRITICAL" in m)
    obs["sdc_info"] = [m for lvl, m in cap.records
                       if lvl == "INFO" and m.startswith("SEED_DOMAIN_COMPLETE")]
    clear_halt()
    return obs


def live_rp():
    return lambda agent, s, e: agent.run_pipeline(start_step=s, end_step=e)


def base_rp():
    src = method_source(git_show(WA_PATH), "run_pipeline")
    if "F1_SEED_DOMAIN_COMPLETE" in src:
        raise Unavailable("BASE run_pipeline already contains F-1")
    ns = exec_in_copy(src, W.__dict__, f"<{BASE}:run_pipeline>")
    fn = ns["run_pipeline"]
    return lambda agent, s, e: fn(agent, start_step=s, end_step=e), ns


def _summ(o):
    keys = ("run_step", "evaluate", "execute", "escalate", "halt_file", "critical",
            "current_step", "pipeline_running", "returned")
    return {k: o[k] for k in keys}


def f1_red():
    fn, ns = base_rp()
    with patched(ns, check_safety=lambda: True, PROGRESS_DISPLAY_AVAILABLE=False):
        o = drive_pipeline(fn, lambda s: sdc_results())
    ok = o["escalate"] == 1 and o["halt_file"] and o["critical"] == 1
    return ok, f"pre-repair: {_summ(o)} (fault reproduced)"


def f1_a():
    o = drive_pipeline(live_rp(), lambda s: sdc_results())
    ok = (not o["halt_file"] and o["escalate"] == 0 and o["evaluate"] == 0
          and o["execute"] == 0 and o["critical"] == 0 and not o["telegram"]
          and o["current_step"] == 1 and o["pipeline_running"] is False
          and len(o["sdc_info"]) == 1 and "The sweep is finished" in o["sdc_info"][0])
    return ok, f"{_summ(o)}; INFO lines={len(o['sdc_info'])}"


def f1_b():
    o = drive_pipeline(live_rp(), lambda s: sdc_results(), start=1, end=2)
    ok = o["run_step"] == [1] and o["current_step"] == 1 and not o["halt_file"]
    return ok, f"run_step calls={o['run_step']}, current_step={o['current_step']}, halt={o['halt_file']}"


def _same_as_base(stub, start, end, expect):
    live = _summ(drive_pipeline(live_rp(), stub, start, end))
    fn, ns = base_rp()
    with patched(ns, check_safety=lambda: True, PROGRESS_DISPLAY_AVAILABLE=False):
        base = _summ(drive_pipeline(fn, stub, start, end))
    return live == base and expect(live), f"live={live} base={'same' if live == base else base}"


def f1_c():
    return _same_as_base(lambda s: {"success": False, "return_code": 1}, 1, 2,
                         lambda o: o["escalate"] == 1 and o["halt_file"] and o["run_step"] == [1])


def f1_d():
    stub = lambda s: {"success": False, "seed_domain_complete": False}  # noqa: E731
    o = drive_pipeline(live_rp(), stub)
    ok = o["escalate"] == 1 and o["halt_file"] and o["critical"] == 1 and not o["sdc_info"]
    return ok, f"{_summ(o)}"


def f1_e():
    return _same_as_base(lambda s: {"success": True}, 1, 2,
                         lambda o: o["escalate"] == 0 and not o["halt_file"]
                         and o["run_step"] == [1, 2] and o["current_step"] == 3)


def f1_mut():
    live = method_source(live_source(WA_PATH), "run_pipeline")
    mut = mutate_block(live, ">>> F1_SEED_DOMAIN_COMPLETE_BEGIN",
                       "<<< F1_SEED_DOMAIN_COMPLETE_END", '_MUT_HITS.append("F1")')
    ns = exec_in_copy(mut, W.__dict__, "<F1-MUT:run_pipeline>")
    fn = ns["run_pipeline"]
    with patched(ns, check_safety=lambda: True, PROGRESS_DISPLAY_AVAILABLE=False):
        o = drive_pipeline(lambda a, s, e: fn(a, start_step=s, end_step=e),
                           lambda s: sdc_results())
    red = o["halt_file"] or o["escalate"] > 0 or o["evaluate"] > 0
    hits = ns["_MUT_HITS"]
    return (red and hits == ["F1"],
            f"check removed, sentinel hits={hits}; F1-A under mutant: {_summ(o)} "
            f"({'RED' if red else 'still green'})")


# ─────────────────────────────────────────────────────────────────────────────
# F-2
# ─────────────────────────────────────────────────────────────────────────────

def decision(action):
    return AgentDecision(success_condition_met=(action == "proceed"), confidence=0.9,
                         reasoning=f"fixture verdict {action}", recommended_action=action,
                         parse_method="fixture")


def drive_loop(ns, step_results, use_llm=False, verdict=None):
    """Run ns['dispatch_learning_loop'] over steps [3, 5] on a real WatcherAgent
    (temp config) with run_step stubbed. `verdict` None -> real evaluate_results."""
    agent = make_agent(use_llm=use_llm)
    obs = {"run_step": [], "logged": [], "llm_calls": 0, "evaluate": 0}

    def run_step(step, *a, **k):
        obs["run_step"].append(step)
        r = step_results(step)
        return dict(r) if isinstance(r, dict) else r

    agent.run_step = run_step
    orig_eval = agent.evaluate_results

    def ev(step, results, *a, **k):
        obs["evaluate"] += 1
        if verdict is not None:
            return decision(verdict), None
        return orig_eval(step, results, *a, **k)

    agent.evaluate_results = ev

    def llm(self_, prompt, grammar):
        obs["llm_calls"] += 1
        return {"decision": "proceed", "confidence": 0.9}

    def log(self_, run_id, kind, status, details):
        obs["logged"].append((status, details))

    with patched(ns, _log_dispatch_decision=log, _evaluate_step_via_bundle=llm,
                 _get_build_llm_context=lambda: (lambda **k: ("prompt", "grammar", {}))), \
            patched(W.__dict__, notify_telegram=lambda *a, **k: None), \
            contextlib.redirect_stdout(io.StringIO()):
        obs["returned"] = ns["dispatch_learning_loop"](agent, scope="steps_3_5", dry_run=False)
    obs["statuses"] = [s for s, _ in obs["logged"]]
    clear_halt()
    return obs


def base_dispatch_ns():
    src = git_show(WD_PATH)
    if "F2_STEP_FAILED" in src:
        raise Unavailable("BASE watcher_dispatch already contains F-2")
    return exec_in_copy(src, D.__dict__, f"<{BASE}:watcher_dispatch>")


FAIL_FIRST = lambda s: {"success": False, "return_code": 1} if s == 3 else {"success": True}  # noqa: E731
ALL_OK = lambda s: {"success": True}  # noqa: E731


def _s(o):
    return {k: o[k] for k in ("returned", "run_step", "statuses", "llm_calls", "evaluate")}


def f2_red_1():
    o = drive_loop(base_dispatch_ns(), FAIL_FIRST)
    return 5 in o["run_step"], f"pre-repair: {_s(o)} (loop went on to step 5: fault reproduced)"


def f2_red_2():
    o = drive_loop(base_dispatch_ns(), ALL_OK, verdict="escalate")
    return 5 in o["run_step"], f"pre-repair, verdict=escalate: {_s(o)} (fault reproduced)"


def _f2a(o):
    sf = [d for s, d in o["logged"] if s == "STEP_FAILED"]
    return (o["returned"] is False and o["run_step"] == [3] and len(sf) == 1
            and sf[0].get("failed_step") == 3 and sf[0].get("completed") == [])


def _f2b(o):
    return o["returned"] is False and o["llm_calls"] == 0 and o["run_step"] == [3]


def _f2c(o):
    return (o["returned"] is False and o["run_step"] == [3]
            and any(s == "STEP_HALTED" and d.get("action") == "escalate"
                    and d.get("halted_step") == 3 for s, d in o["logged"]))


def f2_a():
    o = drive_loop(D.__dict__, FAIL_FIRST)
    return _f2a(o), f"{_s(o)}"


def f2_b():
    o = drive_loop(D.__dict__, FAIL_FIRST, use_llm=True)
    return _f2b(o), f"use_llm=True: {_s(o)}"


def f2_c():
    o = drive_loop(D.__dict__, ALL_OK, verdict="escalate")
    return _f2c(o), f"{_s(o)}"


def f2_d():
    live = _s(drive_loop(D.__dict__, ALL_OK))
    base = _s(drive_loop(base_dispatch_ns(), ALL_OK))
    ok = live == base and live["returned"] is True and live["run_step"] == [3, 5] \
        and live["statuses"] == ["COMPLETED"]
    return ok, f"live={live} base={'same' if live == base else base}"


def f2_e():
    stub = lambda s: None  # noqa: E731
    live = _s(drive_loop(D.__dict__, stub))
    base = _s(drive_loop(base_dispatch_ns(), stub))
    ok = live == base and live["returned"] is False and live["statuses"] == ["STEP_FAILED"]
    return ok, f"live={live} base={'same' if live == base else base}"


def f2_mut_1():
    mut = mutate_block(live_source(WD_PATH), ">>> F2_STEP_FAILED_BEGIN",
                       "<<< F2_STEP_FAILED_END", '_MUT_HITS.append("F2-1")')
    ns = exec_in_copy(mut, D.__dict__, "<F2-MUT-1:watcher_dispatch>")
    a = drive_loop(ns, FAIL_FIRST)
    b = drive_loop(ns, FAIL_FIRST, use_llm=True)
    a_red, b_red = not _f2a(a), not _f2b(b)
    hits = ns["_MUT_HITS"]
    return (a_red and b_red and len(hits) >= 2,
            f"check removed, sentinel hits={len(hits)}; F2-A {'RED' if a_red else 'still green'} "
            f"({_s(a)}); F2-B {'RED' if b_red else 'still green'} ({_s(b)})")


def f2_mut_2():
    mut = mutate_line(live_source(WD_PATH), "F2_HEURISTIC_ACTION_READ",
                      ['_MUT_HITS.append("F2-2")',
                       "action = getattr(heuristic_decision, 'action', 'proceed')"])
    ns = exec_in_copy(mut, D.__dict__, "<F2-MUT-2:watcher_dispatch>")
    c = drive_loop(ns, ALL_OK, verdict="escalate")
    red = not _f2c(c)
    hits = ns["_MUT_HITS"]
    return (red and len(hits) >= 1,
            f"getattr(...'action'...) restored, sentinel hits={len(hits)}; F2-C "
            f"{'RED' if red else 'still green'} ({_s(c)})")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    before = os.path.exists(PROD_HALT)
    print("WATCHER FOLLOW-UP REPAIRS — F-1 / F-2", flush=True)
    print(f"  base={BASE}  temp={TMP}  /tmp/agent_halt present at start={before}", flush=True)

    print("F-1 a finished sweep is a clean stop", flush=True)
    for n, f in (("F1-RED", f1_red), ("F1-A", f1_a), ("F1-B", f1_b), ("F1-C", f1_c),
                 ("F1-D", f1_d), ("F1-E", f1_e), ("F1-MUT", f1_mut)):
        run_gate(n, f)

    print("F-2 the learning loop stops on a failed step", flush=True)
    for n, f in (("F2-RED-1", f2_red_1), ("F2-RED-2", f2_red_2), ("F2-A", f2_a),
                 ("F2-B", f2_b), ("F2-C", f2_c), ("F2-D", f2_d), ("F2-E", f2_e),
                 ("F2-MUT-1", f2_mut_1), ("F2-MUT-2", f2_mut_2)):
        run_gate(n, f)

    after = os.path.exists(PROD_HALT)
    run_gate("HYG-NO-PROD-HALT", lambda: (before == after,
             f"/tmp/agent_halt before={before} after={after}"))
    shutil.rmtree(TMP, ignore_errors=True)

    n_pass = sum(1 for _, v, _ in RESULTS if v == "PASS")
    print(f"  {n_pass}/{len(RESULTS)} PASS", flush=True)
    if n_pass == len(RESULTS):
        print("COMPLETION SENTINEL: PASS", flush=True)
        return 0
    print("COMPLETION SENTINEL: NOT REACHED — "
          + ", ".join(f"{n}={v}" for n, v, _ in RESULTS if v != "PASS"), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
