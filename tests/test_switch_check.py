#!/usr/bin/env python3
"""
Self-test for scripts/diagnostics/switch_check.py on a FIXTURE: a small git repository with a
WATCHER step map, manifests, configs, governance lists, a frozen list and a registry in the real
agents/registry/parameter_registry.py format (its classes are taken from the main tree at 41ad509;
only PARAMETER_REGISTRIES is replaced), plus two worktrees. CPU, one process; everything under
<repo>/data/e2e_grader_selftest/ (never /tmp).

Run:  python3 tests/test_switch_check.py [--only NAME]
Env:  E2E_SWITCH_CHECK_PATH (checker under test), E2E_SELFTEST_DIR (work dir).
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True     # lead note 2026-11-23: never leave a __pycache__ beside loaded code

REPO = Path(__file__).resolve().parents[1]
SC_PATH = Path(os.environ.get("E2E_SWITCH_CHECK_PATH", REPO / "scripts" / "diagnostics" / "switch_check.py"))
WORK = Path(os.environ.get("E2E_SELFTEST_DIR", REPO / "data" / "e2e_grader_selftest" /
                           time.strftime("sc_run_%Y%m%dT%H%M%S")))
MAIN_TREE = Path("~/distributed_prng_analysis").expanduser()


def _load(path, name):
    ld = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, ld)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    ld.exec_module(mod)
    return mod


SC = _load(SC_PATH, "switch_check_under_test")
TESTS = []


def gate(fn):
    TESTS.append(fn)
    return fn


def check(cond, tag, msg=""):
    if not cond:
        raise AssertionError(f"[{tag}] {msg}")


def sh(cwd, *args):
    p = subprocess.run(list(args), cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{args}: {p.stderr}")
    return p.stdout


STEP1 = '''import argparse, os
MODES = ("a", "b")
DEFAULTS = {"w": 12}
ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=5, help="number of trials")
ap.add_argument("--window-size", type=int, default=DEFAULTS["w"], help="window")
ap.add_argument("--mode", choices=list(MODES), default="a", help="the mode")
ap.add_argument("--flag", action="store_true", help="a flag")
ap.add_argument("--hidden", type=int, default=1, help="registered, not in the manifest")
ap.add_argument("--warm-x", type=int, default=0, help="internal-only")
ap.add_argument("--cfg-path", default=os.getcwd(), help="expression default")
'''
WATCHER = '''STEP_SCRIPTS = {1: "step1.py", 2: "run_step2.sh", 3: "run_step3.sh"}
STEP_MANIFESTS = {1: "s1.json", 2: "s2.json", 3: "s3.json"}
def build(final_params):
    final_params.pop("output_file", None)
    _INTERNAL_ONLY_PARAMS = {"warm_x"}
    return _INTERNAL_ONLY_PARAMS
'''
WRAPPER2 = '''#!/bin/bash
TRIALS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --legacy) LEGACY=1; shift;;
    *) TRIALS=$1; shift;;
  esac
done
'''
WRAPPER = '''#!/bin/bash
while [ $# -gt 0 ]; do
  case "$1" in
    --chunk) CHUNK="$2"; shift 2;;
    *) shift;;
  esac
done
python3 worker3.py --chunk "$CHUNK"
'''
FIXTURE_REGISTRY = '''PARAMETER_REGISTRIES: Dict[str, ScriptParameterRegistry] = {
    "step1.py": ScriptParameterRegistry(
        script_name="step1.py",
        description="fixture step 1",
        parameters=[
            ParameterSpec(
                name="trials",
                cli_flag="--trials",
                param_type=ParamType.INT,
                min_value=1,
                max_value=9,
                default=5
            ),
            ParameterSpec(
                name="hidden",
                cli_flag="--hidden",
                param_type=ParamType.INT,
                default=1
            ),
            ParameterSpec(
                name="gone",
                cli_flag="--gone",
                param_type=ParamType.INT,
                default=0
            ),
        ]
    ),
    "nofile.py": ScriptParameterRegistry(
        script_name="nofile.py",
        description="a script that does not exist",
        parameters=[
            ParameterSpec(
                name="z",
                cli_flag="--z",
                param_type=ParamType.INT,
                default=0
            ),
        ]
    ),
}
'''


def real_registry_with(fixture_dict):
    """The real 41ad509 registry module with only PARAMETER_REGISTRIES replaced."""
    src = sh(MAIN_TREE, "git", "show", "41ad509:agents/registry/parameter_registry.py")
    import ast
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign)) and
                getattr(n.target if isinstance(n, ast.AnnAssign) else n.targets[0], "id", None) == "PARAMETER_REGISTRIES")
    lines = src.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + fixture_dict + "".join(lines[node.end_lineno:])


def build_fixture(name):
    root = WORK / name
    main = root / "main"
    wts = root / "wts"
    (main / "agents" / "registry").mkdir(parents=True)
    (main / "agent_manifests").mkdir()
    (main / "scripts").mkdir()
    (main / "miner").mkdir()
    (main / "backups").mkdir()
    wts.mkdir()
    files = {
        "step1.py": STEP1, "agents/watcher_agent.py": WATCHER, "run_step3.sh": WRAPPER, "run_step2.sh": WRAPPER2,
        "worker3.py": 'import argparse\nap = argparse.ArgumentParser()\nap.add_argument("--chunk", type=int, default=10)\n'
                      'ap.add_argument("--other", default="x")\n',
        "gov.py": 'import argparse\nap = argparse.ArgumentParser()\nap.add_argument("--g", default=1)\n',
        "miner/m.py": 'import argparse\nap = argparse.ArgumentParser()\nap.add_argument("--m", default=1)\n',
        "backups/step1.py": 'import argparse\nap = argparse.ArgumentParser()\nap.add_argument("--trials", type=int, default=5)\n',
        "scripts/owner_land.py": 'PARITY_GATE_REL = "scripts/gate.py"\nHAND_LANDED_FILES = ("agents/watcher_agent.py", "gate.sh", PARITY_GATE_REL)\n'
                                 'HAND_LANDED_DIRS = ("miner/", "agent_manifests/")\n',
        "scripts/gate.py": 'GOVERNED_FILES = ("gov.py",)\n',
        "chapter_13_acceptance.py": 'FROZEN_PARAMETERS = frozenset([\n    "window_size",\n])\n',
        "agents/registry/parameter_registry.py": real_registry_with(FIXTURE_REGISTRY),
        "agent_manifests/s1.json": json.dumps({
            "agent_name": "s1", "pipeline_step": 1, "outputs": ["out1.json"],
            "default_params": {"trials": 5, "win": 12, "mode": "a", "warm_x": 1, "ghost": 3, "output_file": "o.json"},
            "actions": [{"type": "run_script", "script": "step1.py", "args_map": {"window-size": "win", "trials": "trials"}}],
            "parameter_bounds": {"window_size": {"min": 2, "max": 40}}}, indent=1),
        "agent_manifests/s2.json": json.dumps({
            "agent_name": "s2", "pipeline_step": 2, "arg_style": "positional", "positional_args": ["trials2"],
            "flag_args": ["legacy"], "default_params": {"trials2": 5, "legacy": False, "unused": 1},
            "actions": [{"type": "run_script", "script": "run_step2.sh", "args_map": {}}]}, indent=1),
        "agent_manifests/s3.json": json.dumps({
            "agent_name": "s3", "pipeline_step": 3, "default_params": {"chunk": 10},
            "actions": [{"type": "run_script", "script": "run_step3.sh", "args_map": {}}]}, indent=1),
        "distributed_config.json": json.dumps({"search_bounds": {"window_size": {"min": 6, "max": 50, "default": 12}},
                                               "sieve_defaults": {"x": 1}}, indent=1),
        "watcher_policies.json": json.dumps({"_note": "x", "a": {"b": 1},
                                             "parameter_bounds": {"mode": {"choices": ["a", "b"]}}}, indent=1),
    }
    for p, t in files.items():
        (main / p).write_text(t)
    sh(main, "git", "init", "-q")
    sh(main, "git", "add", "-A")
    sh(main, "git", "-c", "user.name=fixture", "-c", "user.email=fixture@localhost", "commit", "-q", "-m", "fixture")
    commit = sh(main, "git", "rev-parse", "HEAD").strip()
    # run record on disk (not committed) and an uncommitted change that must NOT be read
    (main / "out1.json").write_text(json.dumps({"trials": 5, "argv": ["--mode", "a"]}))
    (main / "step1.py").write_text(STEP1 + 'ap.add_argument("--dirty", default=1)\n')
    for wt, default in (("wt1", "off"), ("wt2", "on")):
        sh(main, "git", "worktree", "add", "-q", "--detach", str(wts / wt), commit)
        (wts / wt / "step1.py").write_text(
            STEP1 + f'ap.add_argument("--new-flag", choices=["off", "on"], default="{default}", help="new")\n')
    (wts / "wt1" / "scripts" / "diagnostics").mkdir(parents=True)
    (wts / "wt1" / "scripts" / "diagnostics" / "tool.py").write_text(
        'import argparse\nap = argparse.ArgumentParser()\nap.add_argument("--x", type=int, default=3, help="tool x")\n')
    (wts / "wt1" / "tests").mkdir()
    (wts / "wt1" / "tests" / "test_tool.py").write_text(
        'import argparse\nap = argparse.ArgumentParser()\nap.add_argument("--only", default=None)\n')
    return root, main, wts, commit


DECISIONS = {"record": "FIXTURE_RECORD.md", "drop_empty_registries": True, "entries": [
    {"row": 1, "script": "step1.py", "name": "gone", "action": "DROPPED", "why": "no --gone flag in step1.py"},
    {"row": 2, "script": "nofile.py", "name": "z", "action": "DROPPED", "why": "nofile.py never existed"}]}

_CACHE = {}


def result():
    if "r" in _CACHE:
        return _CACHE["r"]
    root, main, wts, commit = build_fixture(f"fixture_{os.getpid()}")   # one per process (gates may share WORK)
    out = root / "out"
    out.mkdir()
    (root / "decisions.json").write_text(json.dumps(DECISIONS))
    argv = ["--main-repo", str(main), "--commit", commit, "--worktrees-root", str(wts),
            "--out-json", str(out / "c.json"), "--out-md", str(out / "c.md"), "--patch-dir", str(out / "patch"),
            "--new-since", "2000-01-01", "--repair-decisions", str(root / "decisions.json"),
            "--repair-dir", str(out / "repair")]
    err, so = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(so):
        rc = SC.main(argv)
    reg = json.loads((out / "c.json").read_text()) if (out / "c.json").is_file() else None
    _CACHE["r"] = (rc, reg, err.getvalue(), root, out, argv)
    return _CACHE["r"]


def row(reg, file, name, tree="main"):
    rs = [r for r in reg["rows"] if r["file"] == file and r["name"] == name and r["tree"] == tree]
    return rs[0] if len(rs) == 1 else None


# ---------------------------------------------------------------------------- gates ------
@gate
def test_sc_argparse():
    rc, reg, err, *_ = result()
    check(rc == 0, "A-SC-RC", f"rc={rc} {err}")
    t = row(reg, "step1.py", "--trials")
    check(t and t["default"] == 5 and t["type"] == "int" and t["plain_words"] == "number of trials", "A-SC-ARGPARSE", str(t))
    w = row(reg, "step1.py", "--window-size")
    check(w and w["default"] == 12, "A-SC-ARGPARSE", f"DEFAULTS['w'] not resolved: {w and w['default']}")
    m = row(reg, "step1.py", "--mode")
    check(m and m["choices"] == ["a", "b"] and m["default"] == "a", "A-SC-ARGPARSE", str(m and m["choices"]))
    f = row(reg, "step1.py", "--flag")
    check(f and f["default"] is False and f["choices"] == [True, False], "A-SC-ARGPARSE", str(f))
    c = row(reg, "step1.py", "--cfg-path")
    check(c and str(c["default"]).startswith("expr:"), "A-SC-ARGPARSE", str(c and c["default"]))


@gate
def test_sc_commit_not_working_copy():
    rc, reg, *_ = result()
    check(row(reg, "step1.py", "--dirty") is None, "A-SC-COMMIT", "an uncommitted main-tree change was read")


@gate
def test_sc_reach():
    rc, reg, *_ = result()
    check(row(reg, "step1.py", "--trials")["watcher"]["reachable"] is True, "A-SC-REACH", "trials")
    check(row(reg, "step1.py", "--window-size")["watcher"]["reachable"] is True, "A-SC-REACH",
          "window-size must be reachable through args_map window-size -> win")
    check(row(reg, "step1.py", "--hidden")["watcher"]["reachable"] is False, "A-SC-REACH", "hidden")
    check(row(reg, "step1.py", "--warm-x")["watcher"]["reachable"] is False, "A-SC-REACH",
          "an _INTERNAL_ONLY_PARAMS key is stripped by WATCHER")
    g = row(reg, "agent_manifests/s1.json", "ghost")
    check(g and "MANIFEST_KEY_NOT_ACCEPTED" in g["flags_raised"], "A-SC-NOTACCEPTED", str(g and g["flags_raised"]))
    ch = row(reg, "worker3.py", "--chunk")
    ot = row(reg, "worker3.py", "--other")
    check(ch["step"] == 3 and ch["watcher"]["reachable"] is True and ot["watcher"]["reachable"] is False,
          "A-SC-WRAPPER", f"{ch['watcher']} {ot['watcher']}")


@gate
def test_sc_positional_and_popped():
    rc, reg, *_ = result()
    lg = row(reg, "run_step2.sh", "--legacy")
    check(lg and lg["step"] == 2 and lg["watcher"]["reachable"] is True, "A-SC-POSITIONAL", str(lg and lg["watcher"]))
    un = row(reg, "agent_manifests/s2.json", "unused")
    t2 = row(reg, "agent_manifests/s2.json", "trials2")
    check("MANIFEST_KEY_NOT_PASSED" in un["flags_raised"] and "MANIFEST_KEY_NOT_ACCEPTED" not in t2["flags_raised"]
          and "MANIFEST_KEY_NOT_PASSED" not in t2["flags_raised"], "A-SC-POSITIONAL",
          f"unused {un['flags_raised']} trials2 {t2['flags_raised']}")
    of = row(reg, "agent_manifests/s1.json", "output_file")
    check(of and "MANIFEST_KEY_NOT_PASSED" in of["flags_raised"] and "MANIFEST_KEY_NOT_ACCEPTED" not in of["flags_raised"],
          "A-SC-POPPED", str(of and of["flags_raised"]))


@gate
def test_sc_registry_class():
    rc, reg, err, *_ = result()
    if reg is None:     # the repair refuses when the stale set is wrong: that is a stale-reporting failure
        check("must cover exactly the stale specs" not in err, "A-SC-STALE", err[-300:])
        check(False, "A-SC-RC", err[-300:])
    cls = lambda f, n: row(reg, f, n)["registry"]["class"]
    check(cls("step1.py", "--trials") == "registered and reachable", "A-SC-CLASS", cls("step1.py", "--trials"))
    check(cls("step1.py", "--hidden") == "registered but unreachable", "A-SC-CLASS", cls("step1.py", "--hidden"))
    check(cls("step1.py", "--mode") == "not registered", "A-SC-CLASS", cls("step1.py", "--mode"))
    check(cls("backups/step1.py", "--trials").startswith("not registered (same basename"), "A-SC-CLASS",
          cls("backups/step1.py", "--trials"))
    check(row(reg, "step1.py", "--trials")["registry"]["llm_visible"] is True
          and row(reg, "step1.py", "--hidden")["registry"]["llm_visible"] is False, "A-SC-LLM", "llm visibility")
    stale = {(d["script"], d["spec"]) for d in reg["pydantic_registry"]["stale_specs"]}
    check(stale == {("step1.py", "gone"), ("nofile.py", "z")}, "A-SC-STALE", str(stale))


@gate
def test_sc_would_register():
    rc, reg, *_ = result()
    wr = row(reg, "step1.py", "--mode")["registry"]["would_register"]
    sp, me = wr["parameter_spec"], wr["manifest_entry"]
    check(sp["param_type"] == "CHOICE" and sp["choices"] == ["a", "b"] and sp["default"] == "a"
          and sp["cli_flag"] == "--mode" and sp["description"] == "the mode", "A-SC-WOULD", str(sp))
    check(me and me["manifest"] == "agent_manifests/s1.json" and me["default_params"] == {"mode": "a"}, "A-SC-WOULD", str(me))
    w = row(reg, "step1.py", "--window-size")
    check("BOUNDS_DIFFER" in w["flags_raised"], "A-SC-BOUNDS", str(w["bounds"]))


@gate
def test_sc_who_and_record():
    rc, reg, *_ = result()
    check(row(reg, "step1.py", "--window-size")["who"] == "frozen", "A-SC-WHO", row(reg, "step1.py", "--window-size")["who"])
    check(row(reg, "gov.py", "--g")["who"] == "owner only" and row(reg, "miner/m.py", "--m")["who"] == "owner only",
          "A-SC-WHO", "governed and hand-landed files are owner only")
    check(row(reg, "step1.py", "--mode")["who"] == "learner within bounds"
          and row(reg, "step1.py", "--hidden")["who"] == "lead", "A-SC-WHO", "bounds / lead")
    check(row(reg, "step1.py", "--trials")["run_record"]["stored"] is True, "A-SC-RUNREC", "key in out1.json")
    check(row(reg, "step1.py", "--mode")["run_record"]["stored"] is True, "A-SC-RUNREC", "argv token in out1.json")
    h = row(reg, "step1.py", "--hidden")
    check(h["run_record"]["stored"] is False and "NO_RUN_RECORD" in h["flags_raised"], "A-SC-RUNREC", "hidden")


@gate
def test_sc_worktrees():
    rc, reg, *_ = result()
    n = row(reg, "step1.py", "--new-flag", "wt1")
    x = row(reg, "scripts/diagnostics/tool.py", "--x", "wt1")
    check(n and n["worktree_status"] == "added" and x and x["worktree_status"] == "added", "A-SC-WT", f"{n} {x}")
    dup = [r for r in reg["rows"] if r["file"] == "step1.py" and r["name"] == "--trials" and r["tree"] != "main"]
    check(not dup, "A-SC-WT-DEDUP", f"unchanged switches repeated for worktrees: {len(dup)}")


@gate
def test_sc_patch():
    rc, reg, err, root, out, argv = result()
    ins = {(d["script"], d["name"]) for d in reg["patch"]["inserted"]}
    check(("step1.py", "new_flag") in ins and ("tool.py", "x") in ins, "A-SC-PATCH", str(ins))
    check(("test_tool.py", "only") not in ins, "A-SC-PATCH-EXCL", "a test-harness flag entered the patch")
    conf = reg["patch"]["conflicts"]
    check(len(conf) == 1 and conf[0]["name"] == "new_flag" and {conf[0]["kept_tree"], conf[0]["other_tree"]} == {"wt1", "wt2"},
          "A-SC-CONFLICT", str(conf))
    frag = json.loads((out / "patch" / "manifest_fragments.json").read_text())["fragments"]
    check(frag.get("agent_manifests/s1.json", {}).get("default_params") == {"new_flag": "off"}, "A-SC-FRAG", str(frag))
    # the patched registry imports with pydantic; existing specs unchanged; new defaults validate
    orig_path = root / "orig_registry.py.txt"
    orig_path.write_text(sh(root / "main", "git", "show", "HEAD:agents/registry/parameter_registry.py"))
    A = _load(orig_path, "fixture_reg_orig")
    B = _load(out / "patch" / "parameter_registry.py.patched", "fixture_reg_new")
    old = {(k, p.name): p.model_dump() for k, g in A.PARAMETER_REGISTRIES.items() for p in g.parameters}
    new = {(k, p.name): p.model_dump() for k, g in B.PARAMETER_REGISTRIES.items() for p in g.parameters}
    check(all(new.get(k) == v for k, v in old.items()) and len(new) == len(old) + 2, "A-SC-PATCH-IMPORT",
          f"old {len(old)} new {len(new)}")
    nf = B.get_registry("step1.py").get_param("new-flag")
    check(nf is not None and nf.validate_value(nf.default) == (True, None) and nf.choices == ["off", "on"],
          "A-SC-PATCH-IMPORT", str(nf))
    check(B.get_registry("tool.py").get_param("x").default == 3, "A-SC-PATCH-IMPORT", "tool.py x")
    sums = (out / "patch" / "SHA256SUMS").read_text()
    check("parameter_registry.py.patch" in sums, "A-SC-PATCH-SUMS", sums)


def _run(argv):
    e, o = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(e), contextlib.redirect_stdout(o):
        rc = SC.main(argv)
    return rc, e.getvalue()


@gate
def test_sc_repair():
    rc, reg, err, root, out, argv = result()
    check(rc == 0, "A-SC-REPAIR", f"rc={rc} {err}")
    rd = out / "repair"
    A = _load(root / "orig_registry2.py.txt", "fx_orig2") if (root / "orig_registry2.py.txt").write_text(
        sh(root / "main", "git", "show", "HEAD:agents/registry/parameter_registry.py")) else None
    B = _load(rd / "parameter_registry.py.repaired", "fx_repaired")
    check(set(B.PARAMETER_REGISTRIES) == {"step1.py"}, "A-SC-REPAIR-EMPTY",
          f"a registry whose every spec was dropped must go: {sorted(B.PARAMETER_REGISTRIES)}")
    names = [p.name for p in B.get_registry("step1.py").parameters]
    check(names == ["trials", "hidden"], "A-SC-REPAIR", str(names))
    old = {p.name: p.model_dump() for p in A.get_registry("step1.py").parameters}
    check(all(old[p.name] == p.model_dump() for p in B.get_registry("step1.py").parameters), "A-SC-REPAIR",
          "a kept spec changed")
    txt = (rd / "parameter_registry.py.repaired").read_text()
    check("# DROPPED: gone --gone (row 1 of FIXTURE_RECORD.md)" in txt
          and '# DROPPED: registry "nofile.py" (z): every spec stale' in txt, "A-SC-REPAIR-DOC", "drop comments missing")
    rr = json.loads((rd / "repair_record.json").read_text())
    check([d["name"] for d in rr["dropped"]] == ["z", "gone"] or sorted(d["name"] for d in rr["dropped"]) == ["gone", "z"],
          "A-SC-REPAIR-DOC", str(rr["dropped"]))
    check(rr["n_specs_before"] == 4 and rr["n_specs_after"] == 2 and rr["registries_removed"] == ["nofile.py"],
          "A-SC-REPAIR-DOC", str({k: rr[k] for k in ("n_specs_before", "n_specs_after", "registries_removed")}))
    sums = (rd / "SHA256SUMS").read_text()
    check("parameter_registry.py.repair.patch" in sums and not (rd / "__pycache__").exists(), "A-SC-REPAIR-DOC", sums)


@gate
def test_sc_repair_refusals():
    rc, reg, err, root, out, argv = result()
    def with_decisions(name, dec):
        path = root / f"{name}.json"
        path.write_text(json.dumps(dec))
        a = list(argv)
        a[a.index("--repair-decisions") + 1] = str(path)
        for flag, sub in (("--out-json", "c.json"), ("--out-md", "c.md"), ("--patch-dir", "patch"), ("--repair-dir", "repair")):
            a[a.index(flag) + 1] = str(root / f"out_{name}" / sub)
        (root / f"out_{name}").mkdir()
        return _run(a)
    rc, e = with_decisions("missing", dict(DECISIONS, entries=DECISIONS["entries"][:1]))
    check(rc == 2 and "must cover exactly the stale specs" in e, "A-SC-REPAIR-COVER", f"rc={rc} {e}")
    rc, e = with_decisions("corrected", dict(DECISIONS, entries=[dict(DECISIONS["entries"][0], action="CORRECTED"),
                                                                 DECISIONS["entries"][1]]))
    check(rc == 2 and "only action DROPPED is implemented" in e, "A-SC-REPAIR-ACTION", f"rc={rc} {e}")
    rc, e = with_decisions("nowhy", dict(DECISIONS, entries=[dict(DECISIONS["entries"][0], why=" "),
                                                             DECISIONS["entries"][1]]))
    check(rc == 2 and "a 'why' is required" in e, "A-SC-REPAIR-WHY", f"rc={rc} {e}")
    # defence in depth: a spec left behind whose flag the script lacks is refused
    src = sh(root / "main", "git", "show", "HEAD:agents/registry/parameter_registry.py")
    preg, _ = SC.parse_param_registry(src)
    stale = [{"script": "step1.py", "spec": "gone"}]
    try:
        SC.repair_registry(src, preg, stale, dict(DECISIONS, entries=DECISIONS["entries"][:1]),
                           {"step1.py": {"trials", "hidden"}, "nofile.py": set()})
        ok = False
    except SC.RegisterError as ex:
        ok = "name a flag their script does not define" in str(ex)
    check(ok, "A-SC-REPAIR-FLAGCHECK", "a kept spec with no flag in its script must be refused")


@gate
def test_sc_refuse_existing():
    rc, reg, err, root, out, argv = result()
    e, o = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(e), contextlib.redirect_stdout(o):
        rc2 = SC.main(argv)
    check(rc2 == 2 and "exists" in e.getvalue(), "A-SC-REFUSE", f"rc={rc2} {e.getvalue()}")


# ---------------------------------------------------------------------------- runner -----
def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    WORK.mkdir(parents=True, exist_ok=True)
    sel = [t for t in TESTS if only is None or t.__name__ == only]
    if not sel:
        print(f"no test named {only}")
        return 2
    fails = 0
    t0 = time.time()
    for t in sel:
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"FAIL {t.__name__}: AssertionError: {e}", flush=True)
        except Exception as e:
            fails += 1
            print(f"FAIL {t.__name__}: CRASH {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"{len(sel) - fails}/{len(sel)} passed in {time.time() - t0:.1f}s; checker={SC_PATH}; work={WORK}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
