#!/usr/bin/env python3
"""
Test of the REAL registry repair draft (TASK_REGISTRY_REPAIR_20261124; owner ruling #7, 2026-11-24):
docs/drafts/loop_v1/registry_repair_20261124/parameter_registry.py.repaired against the main tree
at 41ad509 (read with `git show`; nothing in the main tree is imported or written).

Gates: the repaired file imports under Pydantic (executed in memory, no bytecode written); every
dropped spec is gone and has its dated comment; the 9 kept specs equal the originals field by
field; every kept spec's flag is defined by its script's argparse at 41ad509; nothing outside
PARAMETER_REGISTRIES changed.

Run:  python3 -B tests/test_registry_repair.py [--only NAME]
Env:  E2E_REPAIRED_REGISTRY (file under test), E2E_SWITCH_CHECK_PATH (checker used for its ast parsers).
"""
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
DRAFT = REPO / "docs" / "drafts" / "loop_v1"
REPAIRED = Path(os.environ.get("E2E_REPAIRED_REGISTRY", DRAFT / "registry_repair_20261124" / "parameter_registry.py.repaired"))
DECISIONS = DRAFT / "REGISTRY_REPAIR_DECISIONS_20261124.json"
SC_PATH = Path(os.environ.get("E2E_SWITCH_CHECK_PATH", REPO / "scripts" / "diagnostics" / "switch_check.py"))
MAIN = Path("~/distributed_prng_analysis").expanduser()
COMMIT = "41ad509"
REG = "agents/registry/parameter_registry.py"
TESTS = []


def gate(fn):
    TESTS.append(fn)
    return fn


def check(cond, tag, msg=""):
    if not cond:
        raise AssertionError(f"[{tag}] {msg}")


def show(path):
    return subprocess.run(["git", "-C", str(MAIN), "show", f"{COMMIT}:{path}"], capture_output=True, text=True,
                          check=True).stdout


def _load_sc():
    ld = importlib.machinery.SourceFileLoader("sc_for_repair_test", str(SC_PATH))
    spec = importlib.util.spec_from_loader(ld.name, ld)
    m = importlib.util.module_from_spec(spec)
    sys.modules[ld.name] = m
    ld.exec_module(m)
    return m


SC = _load_sc()
_C = {}


def ctx():
    if _C:
        return _C
    orig = show(REG)
    rep = REPAIRED.read_text()
    mod = types.ModuleType("repaired_registry")
    exec(compile(rep, str(REPAIRED), "exec"), mod.__dict__)        # in memory: no bytecode file
    omod = types.ModuleType("original_registry")
    exec(compile(orig, "41ad509:" + REG, "exec"), omod.__dict__)
    _C.update(orig=orig, rep=rep, mod=mod, omod=omod, dec=json.loads(DECISIONS.read_text()),
              preg_o=SC.parse_param_registry(orig), preg_r=SC.parse_param_registry(rep))
    return _C


@gate
def test_rr_imports():
    c = ctx()
    m = c["mod"]
    check(isinstance(m.PARAMETER_REGISTRIES, dict) and m.get_registry("window_optimizer.py") is not None,
          "A-RR-IMPORT", "repaired registry did not load")
    check(m.get_registry_for_agent("window_optimizer_agent") is not None, "A-RR-IMPORT", "agent lookup")


@gate
def test_rr_dropped():
    c = ctx()
    m = c["mod"]
    for e in c["dec"]["entries"]:
        reg = m.PARAMETER_REGISTRIES.get(e["script"])
        check(reg is None or reg.get_param(e["name"]) is None, "A-RR-DROPPED",
              f"{e['script']}:{e['name']} is still registered")
    for script in ("scorer_meta_optimizer.py", "full_scoring.py"):
        check(script not in m.PARAMETER_REGISTRIES, "A-RR-DROPPED", f"registry {script} for a script that never existed remains")
    tag = c["dec"]["comment_tag"]
    for e in c["dec"]["entries"]:
        if e["script"] in ("scorer_meta_optimizer.py", "full_scoring.py"):
            continue
        check(f"# {tag}: {e['name']} {e['old_flag']} (row {e['row']} of " in c["rep"], "A-RR-DOC",
              f"no drop comment for row {e['row']}")
    for script in ("scorer_meta_optimizer.py", "full_scoring.py"):
        check(f'# {tag}: registry "{script}"' in c["rep"], "A-RR-DOC", f"no drop comment for registry {script}")


@gate
def test_rr_kept_unchanged():
    c = ctx()
    o, r = c["omod"].PARAMETER_REGISTRIES, c["mod"].PARAMETER_REGISTRIES
    dropped = {(e["script"], e["name"]) for e in c["dec"]["entries"]}
    want = {(k, p.name): p.model_dump() for k, g in o.items() for p in g.parameters if (k, p.name) not in dropped}
    got = {(k, p.name): p.model_dump() for k, g in r.items() for p in g.parameters}
    check(len(want) == 9 and set(got) == set(want), "A-RR-KEPT", f"kept set {sorted(got)} != {sorted(want)}")
    check(all(got[k] == want[k] for k in want), "A-RR-KEPT",
          f"changed: {[k for k in want if got.get(k) != want[k]]}")
    for k in r:
        check(r[k].script_name == o[k].script_name and r[k].description == o[k].description, "A-RR-KEPT",
              f"registry header changed for {k}")


@gate
def test_rr_flags_exist():
    c = ctx()
    for (script, name) in [(k, p.name) for k, g in c["mod"].PARAMETER_REGISTRIES.items() for p in g.parameters]:
        spec = c["mod"].get_registry(script).get_param(name)
        rows, err = SC.parse_argparse(show(script), script)
        check(err is None, "A-RR-FLAG", f"{script}: {err}")
        flags = {f for r in rows for f in r["flags"]}
        check(spec.cli_flag in flags, "A-RR-FLAG", f"{script}:{name} cli_flag {spec.cli_flag} not in its argparse")


@gate
def test_rr_outside_unchanged():
    c = ctx()
    import ast
    def outside(src):
        t = ast.parse(src)
        n = next(x for x in t.body if isinstance(x, (ast.Assign, ast.AnnAssign)) and
                 getattr(x.target if isinstance(x, ast.AnnAssign) else x.targets[0], "id", None) == "PARAMETER_REGISTRIES")
        L = src.splitlines()
        return L[:n.lineno - 1], L[n.end_lineno:]
    check(outside(c["orig"]) == outside(c["rep"]), "A-RR-OUTSIDE", "a line outside PARAMETER_REGISTRIES changed")


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
    print(f"{len(sel) - fails}/{len(sel)} passed in {time.time() - t0:.1f}s; repaired={REPAIRED}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
