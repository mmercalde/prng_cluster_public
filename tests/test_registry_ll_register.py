#!/usr/bin/env python3
"""
Test of the LL-1/2/3 switch registrations (TASK_REGISTRY_REPAIR_LAND_20261125 step 2; owner ruling
2026-11-25 17:45, "registry repair, then switch registrations, then manifest entries").

File under test: the registered registry, default
docs/drafts/loop_v1/registry_repair_land_20261125/register/parameter_registry.py.patched (override
with E2E_REGISTERED_REGISTRY, e.g. agents/registry/parameter_registry.py after landing). It is
executed in memory (no bytecode written); nothing in the project is imported.

Gates:
  A-LL-IMPORT     the file loads under Pydantic and the lookups work
  A-LL-REPAIR     the 20 repaired-away specs and the two never-existing registries stay gone; the
                  9 kept specs equal the originals at 41ad509 field by field
  A-LL-ADDONLY    against the repaired file (part 1) the registration only adds: every repaired
                  spec unchanged, exactly the 7 LL specs new, nothing outside PARAMETER_REGISTRIES
  A-LL-CODE       each LL spec's type, choices and default equal its argparse call at the code
                  commit (E2E_CODE_COMMIT, default HEAD of this repository)
  A-LL-ACCEPT     validate_value accepts every legal LL value (each choice, the default, None)
  A-LL-REFUSE     validate_value refuses values the script would refuse (choices; bool words)
  A-LL-UNIQUE     every script's registry lists each normalised name once (Beta X-REGISTRY-REPAIR
                  R1 Q5: a duplicate collapses in every dict-based gate and get_param returns the
                  first, so it is checked on the source's own list, by ast and by Pydantic)
  A-LL-MANIFEST   the manifest fragments declare exactly the code defaults, only values WATCHER's
                  command builder leaves off argv (None / False), and no LL-2 entry (no route)
  A-LL-ARGV       BEHAVIOUR (Beta R1 Q5): the exact owner commands (OWNER_MANIFEST_COMMANDS.txt) are
                  applied to copies of prediction.json and reinforcement.json at the code commit;
                  WATCHER's real run_step builds the Step 5 and Step 6 argv (stub boundary of
                  tests/fixtures/run4_routing_clean_control.py; the command builder is not stubbed).
                  With no params the argv is identical before and after; one non-default value per
                  declared key is dropped before and reaches argv after.

Run:  python3 -B tests/test_registry_ll_register.py [--only NAME]
Env:  E2E_REGISTERED_REGISTRY, E2E_REPAIRED_REGISTRY (part 1), E2E_SWITCH_CHECK_PATH, E2E_CODE_COMMIT,
      E2E_MANIFEST_FRAGMENTS, E2E_OWNER_COMMANDS, E2E_WATCHER_PATH (a copy of agents/watcher_agent.py
      to load instead, for mutants), E2E_SELFTEST_DIR (scratch, under the repository's data/).
No git commit is made: git is used only for `git show`.
"""
import ast
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[1]
LAND = REPO / "docs" / "drafts" / "loop_v1" / "registry_repair_land_20261125"
REGISTERED = Path(os.environ.get("E2E_REGISTERED_REGISTRY", LAND / "register" / "parameter_registry.py.patched"))
REPAIRED = Path(os.environ.get("E2E_REPAIRED_REGISTRY", LAND / "repair" / "parameter_registry.py.repaired"))
FRAGMENTS = Path(os.environ.get("E2E_MANIFEST_FRAGMENTS", LAND / "register" / "manifest_fragments.json"))
OWNER_COMMANDS = Path(os.environ.get("E2E_OWNER_COMMANDS", LAND / "OWNER_MANIFEST_COMMANDS.txt"))
WATCHER_PATH = os.environ.get("E2E_WATCHER_PATH")
WORK = Path(os.environ.get("E2E_SELFTEST_DIR", REPO / "data" / "e2e_grader_selftest" /
                           time.strftime("ll_run_%Y%m%dT%H%M%S")))
# One non-default value per declared key, to prove reach (values only; types come from the code).
REACH_VALUES = {"schema_check": "warn", "exclude_features": "score,confidence,holdout_hits",
                "include_feature_groups": "global", "exclude_feature_groups": "dead",
                "allow_target_columns_as_features": True}
DECISIONS = REPO / "docs" / "drafts" / "loop_v1" / "REGISTRY_REPAIR_DECISIONS_20261124.json"
SC_PATH = Path(os.environ.get("E2E_SWITCH_CHECK_PATH", REPO / "scripts" / "diagnostics" / "switch_check.py"))
CODE_COMMIT = os.environ.get("E2E_CODE_COMMIT", "HEAD")
ORIG_COMMIT = "41ad509"           # the registry before the repair (unchanged through 7f4505a)
REG = "agents/registry/parameter_registry.py"
WATCHER = "agents/watcher_agent.py"

# The LL options the task names: (script, flag, LL item). Types, choices and defaults are NOT
# written here; they are read from the code (A-LL-CODE).
LL = [("prediction_generator.py", "--schema-check", "LL-1"),
      ("full_scoring_worker.py", "--per-seed-sieve-features", "LL-2"),
      ("full_scoring_worker.py", "--fallback-metadata-merge", "LL-2"),
      ("meta_prediction_optimizer_anti_overfit.py", "--exclude-features", "LL-3"),
      ("meta_prediction_optimizer_anti_overfit.py", "--include-feature-groups", "LL-3"),
      ("meta_prediction_optimizer_anti_overfit.py", "--exclude-feature-groups", "LL-3"),
      ("meta_prediction_optimizer_anti_overfit.py", "--allow-target-columns-as-features", "LL-3")]
TESTS = []


def gate(fn):
    TESTS.append(fn)
    return fn


def check(cond, tag, msg=""):
    if not cond:
        raise AssertionError(f"[{tag}] {msg}")


def show(commit, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, text=True,
                          check=True).stdout


def _load_sc():
    ld = importlib.machinery.SourceFileLoader("sc_for_ll_test", str(SC_PATH))
    spec = importlib.util.spec_from_loader(ld.name, ld)
    m = importlib.util.module_from_spec(spec)
    sys.modules[ld.name] = m
    ld.exec_module(m)
    return m


def _exec(src, name):
    mod = types.ModuleType(name)
    exec(compile(src, name, "exec"), mod.__dict__)        # in memory: no bytecode file
    return mod


SC = _load_sc()
_C = {}


def ctx():
    if _C:
        return _C
    reg_src, rep_src, orig_src = REGISTERED.read_text(), REPAIRED.read_text(), show(ORIG_COMMIT, REG)
    _C.update(reg_src=reg_src, rep_src=rep_src, orig_src=orig_src, mod=_exec(reg_src, str(REGISTERED)),
              rep=_exec(rep_src, str(REPAIRED)), orig=_exec(orig_src, f"{ORIG_COMMIT}:{REG}"),
              dec=json.loads(DECISIONS.read_text()))
    return _C


def spec_of(mod, script, flag):
    reg = mod.get_registry(script)
    return reg.get_param(flag) if reg else None


def code_row(script, flag):
    rows, err = SC.parse_argparse(show(CODE_COMMIT, script), script)
    check(err is None, "A-LL-CODE", f"{script}: {err}")
    hits = [r for r in rows if flag in r["flags"]]
    check(len(hits) == 1, "A-LL-CODE", f"{script} {flag}: {len(hits)} argparse calls at {CODE_COMMIT}")
    return hits[0]


def dump(mod):
    return {(k, p.name): p.model_dump() for k, g in mod.PARAMETER_REGISTRIES.items() for p in g.parameters}


@gate
def test_ll_imports():
    m = ctx()["mod"]
    check(isinstance(m.PARAMETER_REGISTRIES, dict) and m.get_registry("window_optimizer.py") is not None,
          "A-LL-IMPORT", "registered registry did not load")
    check(m.get_registry_for_agent("window_optimizer_agent") is not None, "A-LL-IMPORT", "agent lookup")
    for script, flag, _ in LL:
        check(spec_of(m, script, flag) is not None, "A-LL-IMPORT", f"{script} {flag} not registered")


@gate
def test_ll_repair_holds():
    c = ctx()
    m = c["mod"]
    for e in c["dec"]["entries"]:
        reg = m.PARAMETER_REGISTRIES.get(e["script"])
        check(reg is None or reg.get_param(e["name"]) is None, "A-LL-REPAIR", f"{e['script']}:{e['name']} is back")
    for script in ("scorer_meta_optimizer.py", "full_scoring.py"):
        check(script not in m.PARAMETER_REGISTRIES, "A-LL-REPAIR", f"registry {script} is back")
    dropped = {(e["script"], e["name"]) for e in c["dec"]["entries"]}
    want = {k: v for k, v in dump(c["orig"]).items() if k not in dropped}
    got = dump(m)
    check(len(want) == 9 and all(got.get(k) == v for k, v in want.items()), "A-LL-REPAIR",
          f"kept specs changed: {[k for k in want if got.get(k) != want[k]]}")


@gate
def test_ll_add_only():
    c = ctx()
    rep, got = dump(c["rep"]), dump(c["mod"])
    new = set(got) - set(rep)
    want_new = set()
    for script, flag, _ in LL:
        want_new.add((script, SC._clean(flag)))
    check(all(got.get(k) == v for k, v in rep.items()), "A-LL-ADDONLY",
          f"a repaired spec changed: {[k for k in rep if got.get(k) != rep[k]]}")
    check(new == want_new, "A-LL-ADDONLY", f"new specs {sorted(new)} != the 7 LL specs {sorted(want_new)}")
    for k in c["rep"].PARAMETER_REGISTRIES:
        a, b = c["rep"].PARAMETER_REGISTRIES[k], c["mod"].PARAMETER_REGISTRIES[k]
        check(a.script_name == b.script_name and a.description == b.description, "A-LL-ADDONLY", f"header {k} changed")

    def outside(src):
        t = ast.parse(src)
        n = next(x for x in t.body if isinstance(x, (ast.Assign, ast.AnnAssign)) and
                 getattr(x.target if isinstance(x, ast.AnnAssign) else x.targets[0], "id", None) == "PARAMETER_REGISTRIES")
        L = src.splitlines()
        return L[:n.lineno - 1], L[n.end_lineno:]
    check(outside(c["rep_src"]) == outside(c["reg_src"]), "A-LL-ADDONLY", "a line outside PARAMETER_REGISTRIES changed")


@gate
def test_ll_matches_code():
    m = ctx()["mod"]
    for script, flag, _ in LL:
        sp, r = spec_of(m, script, flag), code_row(script, flag)
        check(sp.cli_flag == flag, "A-LL-CODE", f"{script} {flag}: cli_flag {sp.cli_flag}")
        if r["action"] in ("store_true", "store_false"):
            check(sp.param_type.value == "bool" and sp.default is r["default"], "A-LL-CODE",
                  f"{flag}: {sp.param_type.value}/{sp.default!r} vs code {r['action']}/{r['default']!r}")
        elif r["choices"] is not None:
            check(sp.param_type.value == "choice" and sp.choices == list(r["choices"]) and sp.default == r["default"],
                  "A-LL-CODE", f"{flag}: {sp.param_type.value}/{sp.choices}/{sp.default!r} vs code {r['choices']}/{r['default']!r}")
        else:
            check(sp.param_type.value == "string" and r["type"] in (None, "str") and sp.default == r["default"],
                  "A-LL-CODE", f"{flag}: {sp.param_type.value}/{sp.default!r} vs code type {r['type']}/{r['default']!r}")
        check(not sp.required and sp.min_value is None and sp.max_value is None, "A-LL-CODE", f"{flag}: extra bounds")


@gate
def test_ll_validate_accepts():
    m = ctx()["mod"]
    for script, flag, _ in LL:
        sp, r = spec_of(m, script, flag), code_row(script, flag)
        legal = [None, r["default"]]
        if sp.param_type.value == "choice":
            legal += list(r["choices"])
        elif sp.param_type.value == "bool":
            legal += [True, False, "true", "false"]
        else:
            legal += ["global,dead", "score"]
        for v in legal:
            check(sp.validate_value(v) == (True, None), "A-LL-ACCEPT", f"{flag}={v!r} refused: {sp.validate_value(v)}")


@gate
def test_ll_validate_refuses():
    m = ctx()["mod"]
    bad = {"--schema-check": ["on", "FAIL", "", True, 1],
           "--per-seed-sieve-features": ["yes", "ON", True, 1, ""],
           "--fallback-metadata-merge": ["legacy", "batch", "", True],
           "--allow-target-columns-as-features": ["maybe", "2", "on"]}
    for script, flag, _ in LL:
        sp = spec_of(m, script, flag)
        for v in bad.get(flag, []):
            ok, msg = sp.validate_value(v)
            check(ok is False and isinstance(msg, str) and sp.name in msg, "A-LL-REFUSE", f"{flag}={v!r} accepted")
        adj = m.get_registry(script).validate_adjustments({flag: (bad.get(flag) or ["x"])[0]})
        if flag in bad:
            check(len(adj) == 1, "A-LL-REFUSE", f"validate_adjustments {flag}: {adj}")


@gate
def test_ll_unique():
    """Beta X-REGISTRY-REPAIR R1 Q5: one spec per normalised name per script, on the source's own
    lists (ast) and on the loaded Pydantic lists; never through a dict, which would collapse them."""
    c = ctx()
    preg, _ = SC.parse_param_registry(c["reg_src"])
    by_ast = []
    for script, reg in preg.items():
        names = [SC._clean(sp.get("name", "")) for sp in reg["specs"]]
        by_ast += [(script, n) for n in sorted(set(names)) if names.count(n) > 1]
    check(not by_ast, "A-LL-UNIQUE", f"names listed twice (ast): {by_ast}")
    by_model = []
    for script, reg in c["mod"].PARAMETER_REGISTRIES.items():
        names = [p.name.lstrip("-").replace("-", "_") for p in reg.parameters]
        by_model += [(script, n) for n in sorted(set(names)) if names.count(n) > 1]
    check(not by_model, "A-LL-UNIQUE", f"names listed twice (Pydantic): {by_model}")
    n_rep = sum(len(r.parameters) for r in c["rep"].PARAMETER_REGISTRIES.values())
    n_reg = sum(len(r.parameters) for r in c["mod"].PARAMETER_REGISTRIES.values())
    check(n_reg == n_rep + len(LL), "A-LL-UNIQUE", f"{n_reg} specs listed; want {n_rep} repaired + {len(LL)} LL")


# The manifests BEFORE part 3: the base commit the request was reviewed on (main before the owner
# commands). Pinned, so the gate stays meaningful after part 3 lands (then HEAD must equal "after").
MANIFEST_BASE = os.environ.get("E2E_MANIFEST_BASE", "579dbb7")
STEP_MANIFEST = {5: "agent_manifests/reinforcement.json", 6: "agent_manifests/prediction.json"}


def _owner_commands():
    cmds = [ln for ln in OWNER_COMMANDS.read_text().splitlines() if ln.startswith("python3 -B -c ")]
    check(len(cmds) == 2, "A-LL-ARGV", f"{OWNER_COMMANDS}: want 2 owner commands, found {len(cmds)}")
    return cmds


def _watcher():
    """agents.watcher_agent from this tree, or the copy E2E_WATCHER_PATH names (mutants)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import agents.watcher_agent as W                        # noqa: E402  (also imports the agents package)
    if not WATCHER_PATH:
        return W
    ld = importlib.machinery.SourceFileLoader("watcher_agent_under_test", WATCHER_PATH)
    spec = importlib.util.spec_from_loader(ld.name, ld)
    m = importlib.util.module_from_spec(spec)
    sys.modules[ld.name] = m
    ld.exec_module(m)
    return m


class _Done:
    returncode = 0
    stdout = ""
    stderr = ""


def _build_argv(W, step, params, manifests_dir):
    """run_step(step, params) on W up to dispatch; returns the argv. Stub boundary as in
    tests/fixtures/run4_routing_clean_control.py build_argv; the command builder is not stubbed."""
    cfg = types.SimpleNamespace(manifests_dir=str(manifests_dir), force_steps=set(),
                                get_step_timeout_minutes=lambda s: 600)
    agent = W.WatcherAgent.__new__(W.WatcherAgent)
    agent.config = cfg
    agent.retry_counts = {}
    got = {}

    def _stream(cmd, step, timeout):
        got["argv"] = list(cmd)
        return _Done()
    agent._ensure_execution_set = lambda p=None: object()
    agent._run_preflight_check = lambda s: (True, "A-LL-ARGV")
    agent._run_step_streaming = _stream
    agent._run_post_step_cleanup = lambda *a, **kw: None
    agent._find_results = lambda *a, **kw: {}
    saved = {}
    for name, val in (("check_output_freshness", lambda s: (False, "A-LL-ARGV: stale", False)),
                      ("p05_freeze_dataset", lambda run_label="watcher": types.SimpleNamespace(
                          path="daily3.json", sha256="0" * 64, size=0, record_count=0, manifest_id="A-LL-ARGV")),
                      ("p05_resolve_dataset_path", lambda p: str(p))):
        saved[name] = getattr(W, name)
        setattr(W, name, val)
    import miner.dataset_authority as _ds
    ds_saved = {}
    for n, v in (("_active_execution_set", lambda: None), ("load_provisioning_nodes", lambda *a, **kw: []),
                 ("fleet_preflight", lambda *a, **kw: []),
                 ("resolve_absent_fleet_status", lambda *a, **kw: "NOT_APPLICABLE"),
                 ("default_provisioning_manifest_path", lambda *a, **kw: "<A-LL-ARGV>"),
                 ("write_run_provenance", lambda *a, **kw: None)):
        ds_saved[n] = getattr(_ds, n, None)
        setattr(_ds, n, v)
    cwd = os.getcwd()
    try:
        os.chdir(REPO)
        W.WatcherAgent.run_step(agent, step, dict(params))
    finally:
        os.chdir(cwd)
        for n, v in saved.items():
            setattr(W, n, v)
        for n, v in ds_saved.items():
            if v is not None:
                setattr(_ds, n, v)
    check("argv" in got, "A-LL-ARGV", f"step {step}: run_step returned before dispatch (no argv built)")
    return got["argv"]


@gate
def test_ll_argv_behaviour():
    import shutil
    W = _watcher()
    WORK.mkdir(parents=True, exist_ok=True)
    root = WORK / f"ll_argv_{os.getpid()}"
    before, after = root / "before", root / "after"
    try:
        for d in (before, after):
            (d / "agent_manifests").mkdir(parents=True)
            for man in STEP_MANIFEST.values():
                (d / man).write_text(show(MANIFEST_BASE, man))
        for cmd in _owner_commands():
            p = subprocess.run(["bash", "-c", cmd], cwd=after, capture_output=True, text=True)
            check(p.returncode == 0 and p.stdout.startswith("ok "), "A-LL-ARGV",
                  f"owner command failed: rc={p.returncode} {p.stdout} {p.stderr[-300:]}")
        frag = json.loads(FRAGMENTS.read_text())["fragments"]
        for man in STEP_MANIFEST.values():
            b, a = json.loads((before / man).read_text()), json.loads((after / man).read_text())
            added = {k: v for k, v in a["default_params"].items() if k not in b["default_params"]}
            check(added == frag[man]["default_params"] and {k: a[k] for k in b if k != "default_params"}
                  == {k: b[k] for k in b if k != "default_params"}, "A-LL-ARGV",
                  f"{man}: the owner command added {added}, the fragment says {frag[man]['default_params']}")
            # after part 3 lands, HEAD's manifest must be exactly the "after" file
            head = json.loads(show(CODE_COMMIT, man))
            if any(k in head.get("default_params", {}) for k in added):
                check((after / man).read_text() == show(CODE_COMMIT, man), "A-LL-ARGV",
                      f"{man} at {CODE_COMMIT} declares the keys but differs from the owner command's output")
        for step, man in STEP_MANIFEST.items():
            a0, a1 = _build_argv(W, step, {}, before / "agent_manifests"), _build_argv(W, step, {}, after / "agent_manifests")
            check(a0 == a1, "A-LL-ARGV", f"step {step} default argv changed: {a0} -> {a1}")
            for key in frag[man]["default_params"]:
                flag = "--" + key.replace("_", "-")
                val = REACH_VALUES[key]
                b1 = _build_argv(W, step, {key: val}, before / "agent_manifests")
                c1 = _build_argv(W, step, {key: val}, after / "agent_manifests")
                check(flag not in b1, "A-LL-ARGV", f"step {step} {key}: reached argv BEFORE the declaration: {b1}")
                want = [flag] if val is True else [flag, str(val)]
                i = c1.index(flag) if flag in c1 else -1
                check(i >= 0 and c1[i:i + len(want)] == want, "A-LL-ARGV",
                      f"step {step} {key}={val!r}: not passed after the declaration: {c1}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


@gate
def test_ll_manifest_fragments():
    frag = json.loads(FRAGMENTS.read_text())["fragments"]
    m = ctx()["mod"]
    want = {"agent_manifests/prediction.json": {"prediction_generator.py"},
            "agent_manifests/reinforcement.json": {"meta_prediction_optimizer_anti_overfit.py"}}
    check(set(frag) == set(want), "A-LL-MANIFEST", f"fragments for {sorted(frag)}; LL-2 must have none (no route)")
    seen = set()
    for man, f in frag.items():
        check(f.get("args_map_primary_action") == {}, "A-LL-MANIFEST", f"{man}: an args_map change")
        for key, val in f["default_params"].items():
            script, flag = next((s, fl) for s, fl, _ in LL if SC._clean(fl) == key)
            check(script in want[man], "A-LL-MANIFEST", f"{key} in {man}")
            r = code_row(script, flag)
            check(val == r["default"] and (val is None or val is False), "A-LL-MANIFEST",
                  f"{man} {key}={val!r}: must equal the code default {r['default']!r} and be a value the builder omits")
            check(spec_of(m, script, flag).default == val, "A-LL-MANIFEST", f"{key}: registry default differs")
            mj = json.loads(show(MANIFEST_BASE, man))
            check(key not in mj.get("default_params", {}), "A-LL-MANIFEST",
                  f"{man} already declares {key} at the base {MANIFEST_BASE}")
            seen.add((script, flag))
    want_seen = {(s, fl) for s, fl, ll in LL if ll != "LL-2"}
    check(seen == want_seen, "A-LL-MANIFEST", f"declared {sorted(seen)} != LL-1 and LL-3 {sorted(want_seen)}")


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
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
    print(f"{len(sel) - fails}/{len(sel)} passed in {time.time() - t0:.1f}s; registered={REGISTERED}; "
          f"code={CODE_COMMIT}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
