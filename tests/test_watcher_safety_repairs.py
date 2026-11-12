#!/usr/bin/env python3
"""WATCHER safety repairs R-1 / R-2 / R-3 — gate battery (2026-11-11).

Brief: docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_SAFETY_REPAIRS_20261111.md

  R-1  an explicit step failure (results["success"] is False) beats "the
       output file exists" in WatcherAgent.evaluate_results.
  R-2  `watcher_agent.py --dry-run` combined with a launching mode that ignores
       it (--run-pipeline / --approve / --daemon) exits 2 before any
       WatcherAgent is constructed.
  R-3  the three dispatch halt sites share one helper that honours the agent's
       kill-switch halt file AND the legacy working-directory flag.

Safety: no fleet, no pipeline, no subprocess launch. Every step runner, the
selfplay subprocess, the dispatch decision log and WatcherAgent itself (for
main()) are stubbed. All halt files live in a temp directory; this suite never
creates /tmp/agent_halt and checks that it did not.

RED arms execute the pre-repair source from BASE (git show), MUT arms execute
a mutated copy of the LIVE source; both are compiled into a copy of the
production module's globals. A mutant whose markers cannot be found is an
ERROR, never a detection; each mutant carries a hit sentinel that proves the
mutated code actually ran.

Prints one line per gate, then `COMPLETION SENTINEL: PASS` only if every gate
ran and passed.
"""

import ast
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import agents.watcher_agent as W      # noqa: E402
import agents.watcher_dispatch as D   # noqa: E402

BASE = "741822b"
PROD_HALT = "/tmp/agent_halt"   # observed only, never created
WA_PATH = "agents/watcher_agent.py"
WD_PATH = "agents/watcher_dispatch.py"

RESULTS = []


class Unavailable(Exception):
    """The gate's observer/input could not be obtained."""


class MutantNotApplied(Exception):
    """A mutant's markers were not found — an error, never a detection."""


def record(name, verdict, detail=""):
    RESULTS.append((name, verdict, detail))
    print(f"  [{verdict:<11}] {name}  — {detail}", flush=True)
    return verdict


def run_gate(name, fn):
    try:
        ok, detail = fn()
        return record(name, "PASS" if ok else "FAIL", detail)
    except Unavailable as e:
        return record(name, "UNAVAILABLE", str(e))
    except MutantNotApplied as e:
        return record(name, "ERROR", f"mutant not applied: {e}")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        return record(name, "ERROR", f"{type(e).__name__}: {e} [{tb}]")


# ─────────────────────────────────────────────────────────────────────────────
# Source helpers
# ─────────────────────────────────────────────────────────────────────────────

def git_show(path):
    r = subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise Unavailable(f"git show {BASE}:{path} failed: {r.stderr.strip()[:200]}")
    return r.stdout


def live_source(path):
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


def function_source(module_src, name, cls=None):
    tree = ast.parse(module_src)
    scope = tree.body
    if cls:
        hits = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
        if len(hits) != 1:
            raise Unavailable(f"class {cls} not found exactly once")
        scope = hits[0].body
    hits = [n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(hits) != 1:
        raise Unavailable(f"function {name} not found exactly once")
    return textwrap.dedent(ast.get_source_segment(module_src, hits[0]))


def exec_in_copy(src, base_globals, filename):
    """Compile `src` into a COPY of a production module's globals."""
    ns = dict(base_globals)
    ns["_MUT_HITS"] = []
    exec(compile(src, filename, "exec"), ns)
    return ns


def apply_mutant(src, begin, end, replacement_stmt):
    """Replace the marker-delimited block (inclusive) with one statement at the
    marker's indentation. Raises MutantNotApplied if markers are not found
    exactly once each and in order."""
    lines = src.splitlines(keepends=True)
    b = [i for i, ln in enumerate(lines) if begin in ln]
    e = [i for i, ln in enumerate(lines) if end in ln]
    if len(b) != 1 or len(e) != 1 or e[0] <= b[0]:
        raise MutantNotApplied(f"markers {begin}/{end}: begin={b} end={e}")
    indent = lines[b[0]][:len(lines[b[0]]) - len(lines[b[0]].lstrip())]
    new = lines[:b[0]] + [f"{indent}{replacement_stmt}\n"] + lines[e[0] + 1:]
    out = "".join(new)
    if out == src:
        raise MutantNotApplied("mutation produced identical source")
    return out


@contextlib.contextmanager
def patched(ns, **repl):
    """Temporarily replace names in a globals dict (module.__dict__ or a copy)."""
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

TMP = tempfile.mkdtemp(prefix="watcher_safety_repairs_")
TMP_HALT = os.path.join(TMP, "agent_halt_fixture")
MAN_DIR = os.path.join(TMP, "manifests")
OUT_DIR = os.path.join(TMP, "outputs")
os.makedirs(MAN_DIR)
os.makedirs(OUT_DIR)

VALID_JSON_BODY = json.dumps({"fixture": "watcher_safety_repairs",
                              "note": "valid output file present", "pad": "x" * 128})


def make_manifest(step, out_name):
    """Copy the REAL manifest for `step` (so AgentManifest.load accepts it) and
    point its success_condition at a temp output path."""
    name = W.STEP_MANIFESTS[step]
    with open(os.path.join(ROOT, "agent_manifests", name)) as f:
        m = json.load(f)
    out = os.path.join(OUT_DIR, out_name)
    m["success_condition"] = [out]
    with open(os.path.join(MAN_DIR, name), "w") as f:
        json.dump(m, f, indent=2)
    return out, m


def write_valid(path):
    with open(path, "w") as f:
        f.write(VALID_JSON_BODY)


OUT1, M1 = make_manifest(1, "step1_out.json")
OUT2, M2 = make_manifest(2, "step2_out.json")
OUT0, M0 = make_manifest(0, "step0_trse_context.json")
write_valid(OUT1)
write_valid(OUT2)
write_valid(OUT0)


def make_agent():
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
    return agent


def clear_tmp_halt():
    if os.path.exists(TMP_HALT):
        os.remove(TMP_HALT)


# ─────────────────────────────────────────────────────────────────────────────
# R-1
# ─────────────────────────────────────────────────────────────────────────────

FAIL_RESULTS = {"success": False, "return_code": 1, "error": "fixture: non-zero exit"}


def r1_red():
    old_src = function_source(git_show(WA_PATH), "evaluate_results", cls="WatcherAgent")
    if "step_failed" in old_src:
        raise Unavailable("BASE evaluate_results already contains the repair")
    ns = exec_in_copy(old_src, W.__dict__, f"<{BASE}:evaluate_results>")
    agent = make_agent()
    write_valid(OUT1)
    d, _ = ns["evaluate_results"](agent, 1, dict(FAIL_RESULTS))
    return (d.recommended_action == "proceed",
            f"pre-repair {BASE} evaluate_results on success=False + valid file -> "
            f"{d.recommended_action}/{d.parse_method}/conf={d.confidence:.2f} (fault reproduced)")


def r1_a():
    agent = make_agent()
    write_valid(OUT1)
    d, _ = agent.evaluate_results(1, dict(FAIL_RESULTS))
    ok = (d.recommended_action == "escalate" and d.parse_method == "step_failed"
          and d.success_condition_met is False and d.confidence == 0.0
          and "return_code=1" in d.reasoning and "non-zero exit" in d.reasoning
          and "Step 1" in d.reasoning)
    return ok, f"{d.recommended_action}/{d.parse_method}/conf={d.confidence}; reasoning={d.reasoning!r}"


def r1_b():
    agent = make_agent()
    write_valid(OUT1)
    write_valid(OUT2)
    clear_tmp_halt()
    calls = []

    def stub_run_step(step, params=None, **kw):
        calls.append(step)
        return dict(FAIL_RESULTS) if step == 1 else {"success": True}

    agent.run_step = stub_run_step
    with patched(W.__dict__, check_safety=lambda: True,
                 notify_telegram=lambda *a, **k: None,
                 PROGRESS_DISPLAY_AVAILABLE=False):
        with contextlib.redirect_stdout(io.StringIO()):
            agent.run_pipeline(start_step=1, end_step=2)
    halted = os.path.exists(TMP_HALT)
    clear_tmp_halt()
    ok = calls == [1] and agent.current_step == 1 and halted
    return ok, (f"run_step calls={calls}, current_step={agent.current_step}, "
                f"escalation halt written to temp halt path={halted}")


def r1_c():
    agent = make_agent()
    write_valid(OUT1)
    d, _ = agent.evaluate_results(1, {"success": True})
    return (d.recommended_action == "proceed" and d.parse_method == "file_exists",
            f"{d.recommended_action}/{d.parse_method}")


def r1_d():
    agent = make_agent()
    write_valid(OUT1)
    d, _ = agent.evaluate_results(1, {"pipeline_step": 1})
    return (d.recommended_action == "proceed" and d.parse_method == "file_exists",
            f"{d.recommended_action}/{d.parse_method}")


def r1_e():
    if M0.get("skip_on_fail") is not True:
        raise Unavailable("real trse.json no longer declares skip_on_fail: true")
    agent = make_agent()
    out = []
    for label, present in (("file valid", True), ("file absent", False)):
        if present:
            write_valid(OUT0)
        elif os.path.exists(OUT0):
            os.remove(OUT0)
        with contextlib.redirect_stdout(io.StringIO()):
            d, _ = agent.evaluate_results(0, {"success": False})
        warned = any("exited non-zero" in w for w in d.warnings)
        out.append((label, d.recommended_action, d.parse_method, warned))
    write_valid(OUT0)
    ok = all(a == "proceed" and p == "skip_on_fail" and w for _, a, p, w in out)
    return ok, "; ".join(f"{l}: {a}/{p} warning={w}" for l, a, p, w in out)


def r1_mut():
    live = function_source(live_source(WA_PATH), "evaluate_results", cls="WatcherAgent")
    mut = apply_mutant(live, ">>> R1_STEP_FAILED_BRANCH_BEGIN",
                       "<<< R1_STEP_FAILED_BRANCH_END", '_MUT_HITS.append("R1")')
    if "parse_method=\"step_failed\"" in mut:
        raise MutantNotApplied("step_failed branch still present after mutation")
    ns = exec_in_copy(mut, W.__dict__, "<R1-MUT:evaluate_results>")
    agent = make_agent()
    write_valid(OUT1)
    d, _ = ns["evaluate_results"](agent, 1, dict(FAIL_RESULTS))
    r1a_red = not (d.recommended_action == "escalate" and d.parse_method == "step_failed")
    hits = ns["_MUT_HITS"]
    return (r1a_red and hits == ["R1"],
            f"branch deleted between R1 markers, sentinel hits={hits}; R1-A under mutant -> "
            f"{d.recommended_action}/{d.parse_method} ({'RED' if r1a_red else 'still green'})")


# ─────────────────────────────────────────────────────────────────────────────
# R-2
# ─────────────────────────────────────────────────────────────────────────────

class FakeWatcher:
    """Stands in for WatcherAgent inside main(): records, launches nothing."""
    calls = None

    @classmethod
    def reset(cls):
        cls.calls = {"init": 0, "run_pipeline": [], "run_daemon": 0,
                     "dispatch_selfplay": []}

    @staticmethod
    def _read_pid(_p):
        return None

    def __init__(self, config=None):
        FakeWatcher.calls["init"] += 1

    def run_pipeline(self, *a, **k):
        FakeWatcher.calls["run_pipeline"].append((a, k))

    def run_daemon(self, *a, **k):
        FakeWatcher.calls["run_daemon"] += 1

    def dispatch_selfplay(self, request, dry_run=False):
        FakeWatcher.calls["dispatch_selfplay"].append((request, dry_run))
        return True

    def dispatch_learning_loop(self, *a, **k):
        return True

    def _scan_watcher_requests(self, dry_run=False):
        return 0


def run_main(main_fn, ns, argv):
    FakeWatcher.reset()
    saved_argv = sys.argv
    buf = io.StringIO()
    code = None
    sys.argv = ["watcher_agent.py"] + list(argv)
    try:
        with patched(ns, WatcherAgent=FakeWatcher), contextlib.redirect_stdout(buf):
            try:
                main_fn()
            except SystemExit as e:
                code = e.code
    finally:
        sys.argv = saved_argv
    return code, buf.getvalue(), FakeWatcher.calls


R2_ARGV = ["--run-pipeline", "--dry-run", "--start-step", "1", "--end-step", "1"]


def base_main():
    src = function_source(git_show(WA_PATH), "main")
    if "R2_DRY_RUN_REFUSAL" in src:
        raise Unavailable("BASE main already contains the refusal")
    ns = exec_in_copy(src, W.__dict__, f"<{BASE}:main>")
    return ns["main"], ns


def r2_red():
    fn, ns = base_main()
    code, _, calls = run_main(fn, ns, R2_ARGV)
    return (len(calls["run_pipeline"]) == 1,
            f"pre-repair main: exit={code}, WatcherAgent constructed {calls['init']}x, "
            f"run_pipeline calls={calls['run_pipeline']} (fault reproduced)")


def _refused(argv, mode):
    code, out, calls = run_main(W.main, W.__dict__, argv)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    ok = (code == 2 and calls["init"] == 0 and not calls["run_pipeline"]
          and calls["run_daemon"] == 0 and len(lines) == 1 and mode in lines[0]
          and "dry-run is not supported" in lines[0])
    return ok, f"exit={code}, constructed={calls['init']}, run_pipeline={len(calls['run_pipeline'])}, out={lines}"


def r2_a():
    return _refused(R2_ARGV, "--run-pipeline")


def r2_b_approve():
    return _refused(["--approve", "--dry-run"], "--approve")


def r2_b_daemon():
    return _refused(["--daemon", "--dry-run"], "--daemon")


def r2_c():
    code, _, calls = run_main(W.main, W.__dict__, ["--dispatch-selfplay", "--dry-run"])
    sp = calls["dispatch_selfplay"]
    ok = code == 0 and sp == [({"episodes": 5}, True)]
    return ok, f"exit={code}, dispatch_selfplay calls={sp}"


def r2_d():
    argv = ["--run-pipeline", "--start-step", "1", "--end-step", "1"]
    _, _, live = run_main(W.main, W.__dict__, argv)
    live_calls = list(live["run_pipeline"])
    fn, ns = base_main()
    _, _, base = run_main(fn, ns, argv)
    base_calls = list(base["run_pipeline"])
    ok = len(live_calls) == 1 and live_calls == base_calls
    return ok, f"live={live_calls} base={base_calls}"


def r2_mut():
    live = function_source(live_source(WA_PATH), "main")
    mut = apply_mutant(live, ">>> R2_DRY_RUN_REFUSAL_BEGIN",
                       "<<< R2_DRY_RUN_REFUSAL_END", '_MUT_HITS.append("R2")')
    ns = exec_in_copy(mut, W.__dict__, "<R2-MUT:main>")
    code, _, calls = run_main(ns["main"], ns, R2_ARGV)
    hits = ns["_MUT_HITS"]
    r2a_red = code != 2 or bool(calls["run_pipeline"]) or calls["init"] > 0
    return (r2a_red and hits == ["R2"],
            f"refusal removed, sentinel hits={hits}; R2-A under mutant: exit={code}, "
            f"run_pipeline={len(calls['run_pipeline'])} ({'RED' if r2a_red else 'still green'})")


# ─────────────────────────────────────────────────────────────────────────────
# R-3
# ─────────────────────────────────────────────────────────────────────────────

class FakeDispatchAgent:
    def __init__(self, halt_path, after_step=None):
        self.kill_switch = W.KillSwitch(halt_file_path=halt_path)
        self.config = SimpleNamespace(halt_file=halt_path, use_llm=False,
                                      manifests_dir=MAN_DIR)
        self.llm_lifecycle = None
        self.steps_run = []
        self._after_step = after_step

    def run_step(self, step, *a, **k):
        self.steps_run.append(step)
        if self._after_step:
            self._after_step(step)
        return {"success": True}

    def evaluate_results(self, step, results, *a, **k):
        return SimpleNamespace(recommended_action="proceed"), None


class FakeSubprocess:
    TimeoutExpired = subprocess.TimeoutExpired
    CompletedProcess = subprocess.CompletedProcess
    PIPE = subprocess.PIPE
    DEVNULL = subprocess.DEVNULL

    def __init__(self):
        self.launched = []

    def run(self, cmd, *a, **k):
        self.launched.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def Popen(self, cmd, *a, **k):  # noqa: N802
        raise AssertionError(f"unexpected Popen in dispatch test: {cmd}")


def dispatch_ns(ns):
    """Stubs that must be in force for any dispatch call against `ns`."""
    fake = FakeSubprocess()
    stubs = dict(subprocess=fake,
                 _log_dispatch_decision=lambda *a, **k: None,
                 _get_build_llm_context=lambda: (lambda **k: ("prompt", "grammar", {})))
    return fake, stubs


def call_selfplay(ns, agent):
    fake, stubs = dispatch_ns(ns)
    with patched(ns, **stubs):
        ok = ns["dispatch_selfplay"](agent, {"episodes": 1}, dry_run=False)
    return ok, fake.launched


def call_learning(ns, agent):
    fake, stubs = dispatch_ns(ns)
    with patched(ns, **stubs):
        ok = ns["dispatch_learning_loop"](agent, scope="steps_3_5", dry_run=False)
    return ok, agent.steps_run


def touch(path):
    with open(path, "w") as f:
        f.write("fixture halt — watcher_safety_repairs suite")


def base_dispatch_ns():
    src = git_show(WD_PATH)
    if "_halt_requested" in src:
        raise Unavailable("BASE watcher_dispatch already contains the helper")
    return exec_in_copy(src, D.__dict__, f"<{BASE}:watcher_dispatch>")


def r3_red():
    ns = base_dispatch_ns()
    clear_tmp_halt()
    touch(TMP_HALT)
    try:
        ok_sp, launched = call_selfplay(ns, FakeDispatchAgent(TMP_HALT))
        _, steps = call_learning(ns, FakeDispatchAgent(TMP_HALT))
    finally:
        clear_tmp_halt()
    ok = bool(launched) and steps == [3, 5]
    return ok, (f"kill-switch halt present at temp path; pre-repair selfplay launched "
                f"{len(launched)} subprocess(es), learning loop ran steps {steps} (fault reproduced)")


def _site_results(ns):
    """Run all three halt sites with only the kill-switch halt present."""
    res = {}
    clear_tmp_halt()
    touch(TMP_HALT)
    try:
        ok, launched = call_selfplay(ns, FakeDispatchAgent(TMP_HALT))
        res["site1_selfplay"] = (ok is False and not launched, f"ok={ok} launched={len(launched)}")
        ok, steps = call_learning(ns, FakeDispatchAgent(TMP_HALT))
        res["site2_loop_start"] = (ok is False and steps == [], f"ok={ok} steps={steps}")
    finally:
        clear_tmp_halt()
    agent = FakeDispatchAgent(TMP_HALT, after_step=lambda s: touch(TMP_HALT) if s == 3 else None)
    try:
        ok, steps = call_learning(ns, agent)
        res["site3_between_steps"] = (ok is False and steps == [3], f"ok={ok} steps={steps}")
    finally:
        clear_tmp_halt()
    return res


def _legacy_results(ns):
    res = {}
    cwd = os.getcwd()
    wd = tempfile.mkdtemp(prefix="legacy_cwd_", dir=TMP)
    os.chdir(wd)
    try:
        touch(os.path.join(wd, "watcher_halt" + ".flag"))
        nohalt = os.path.join(TMP, "no_such_halt")
        ok, launched = call_selfplay(ns, FakeDispatchAgent(nohalt))
        res["selfplay"] = (ok is False and not launched, f"ok={ok} launched={len(launched)}")
        ok, steps = call_learning(ns, FakeDispatchAgent(nohalt))
        res["learning"] = (ok is False and steps == [], f"ok={ok} steps={steps}")
    finally:
        os.chdir(cwd)
    return res


def _fmt(res):
    return "; ".join(f"{k}: {'ok' if v[0] else 'NOT HALTED'} ({v[1]})" for k, v in res.items())


def r3_a():
    res = _site_results(D.__dict__)
    return all(v[0] for v in res.values()), _fmt(res)


def r3_b():
    res = _legacy_results(D.__dict__)
    return all(v[0] for v in res.values()), _fmt(res)


def r3_c():
    clear_tmp_halt()
    nohalt = os.path.join(TMP, "no_such_halt")
    ok_sp, launched = call_selfplay(D.__dict__, FakeDispatchAgent(nohalt))
    ok_ll, steps = call_learning(D.__dict__, FakeDispatchAgent(nohalt))
    ok = ok_sp is True and len(launched) == 1 and ok_ll is True and steps == [3, 5]
    return ok, f"selfplay ok={ok_sp} launched={len(launched)}; learning ok={ok_ll} steps={steps}"


def r3_scan():
    src = live_source(WD_PATH)
    tree = ast.parse(src)
    helper = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_halt_requested"]
    if len(helper) != 1:
        return False, "helper _halt_requested not found exactly once"
    lo, hi = helper[0].lineno, helper[0].end_lineno
    occ = [i for i, ln in enumerate(src.splitlines(), 1) if "watcher_halt.flag" in ln]
    outside = [i for i in occ if not lo <= i <= hi]
    other_literals = sorted({
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and ("agent_halt" in n.value or n.value.endswith("halt.flag"))
        and not (lo <= n.lineno <= hi)
    })
    calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_halt_requested"]
    ok = bool(occ) and not outside and not other_literals and len(calls) == 3
    return ok, (f"'watcher_halt.flag' on lines {occ}, helper spans {lo}-{hi}, outside={outside}; "
                f"other halt literals={other_literals}; helper call sites={calls}")


def r3_mut():
    mut = apply_mutant(live_source(WD_PATH), ">>> R3_HALT_HELPER_BODY_BEGIN",
                       "<<< R3_HALT_HELPER_BODY_END",
                       'return _MUT_HITS.append("R3")  # always None')
    ns = exec_in_copy(mut, D.__dict__, "<R3-MUT:watcher_dispatch>")
    a = _site_results(ns)
    b = _legacy_results(ns)
    hits = ns["_MUT_HITS"]
    a_red = not all(v[0] for v in a.values())
    b_red = not all(v[0] for v in b.values())
    return (a_red and b_red and len(hits) > 0,
            f"helper forced to None, sentinel hits={len(hits)}; R3-A {'RED' if a_red else 'still green'} "
            f"({_fmt(a)}); R3-B {'RED' if b_red else 'still green'} ({_fmt(b)})")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    prod_halt_before = os.path.exists(PROD_HALT)
    print("WATCHER SAFETY REPAIRS — R-1 / R-2 / R-3", flush=True)
    print(f"  base={BASE}  temp={TMP}  /tmp/agent_halt present at start={prod_halt_before}", flush=True)

    print("R-1 explicit step failure beats file-exists", flush=True)
    for name, fn in (("R1-RED", r1_red), ("R1-A", r1_a), ("R1-B", r1_b), ("R1-C", r1_c),
                     ("R1-D", r1_d), ("R1-E", r1_e), ("R1-MUT", r1_mut)):
        run_gate(name, fn)

    print("R-2 --dry-run must never launch", flush=True)
    for name, fn in (("R2-RED", r2_red), ("R2-A", r2_a), ("R2-B-approve", r2_b_approve),
                     ("R2-B-daemon", r2_b_daemon), ("R2-C", r2_c), ("R2-D", r2_d),
                     ("R2-MUT", r2_mut)):
        run_gate(name, fn)

    print("R-3 one halt authority", flush=True)
    for name, fn in (("R3-RED", r3_red), ("R3-A", r3_a), ("R3-B", r3_b), ("R3-C", r3_c),
                     ("R3-SCAN", r3_scan), ("R3-MUT", r3_mut)):
        run_gate(name, fn)

    prod_halt_after = os.path.exists(PROD_HALT)
    run_gate("HYG-NO-PROD-HALT", lambda: (
        prod_halt_after == prod_halt_before and not (prod_halt_after and not prod_halt_before),
        f"/tmp/agent_halt before={prod_halt_before} after={prod_halt_after}"))
    shutil.rmtree(TMP, ignore_errors=True)

    n_pass = sum(1 for _, v, _ in RESULTS if v == "PASS")
    print(f"  {n_pass}/{len(RESULTS)} PASS", flush=True)
    if n_pass == len(RESULTS):
        print("COMPLETION SENTINEL: PASS", flush=True)
        return 0
    bad = [f"{n}={v}" for n, v, _ in RESULTS if v != "PASS"]
    print(f"COMPLETION SENTINEL: NOT REACHED — {', '.join(bad)}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
