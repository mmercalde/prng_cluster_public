#!/usr/bin/env python3
"""
Unit gates for scripts/diagnostics/switch_check.py that need NO git repository.

Written after Beta X-REGISTRY-REPAIR R1 Q5 (TASK_REG_BETA_Q5_FIXES_20261125). The fixture suite
tests/test_switch_check.py builds a repository and makes a git commit. Until the owner rules on
fixture commits, that suite is not run, so these gates call the checker's own functions directly
on in-memory sources and rows. Nothing is written; no subprocess is started.

Gates:
  A-SCU-DUP        duplicate_specs finds a name listed twice in one script, and finds none in the
                   registered draft
  A-SCU-PATCHDUP   a patch built on a registry that already lists a name twice is caught
  A-SCU-NARGS      parse_argparse keeps nargs; R10 (select_register_rows) refuses an nargs option
  A-SCU-ACTION     R10 refuses append/count/store_const actions; accepts store/store_true
  A-SCU-SELECT     R10 still refuses unknown, duplicate and already-registered entries

Run:  python3 -B tests/test_switch_check_units.py [--only NAME]
Env:  E2E_SWITCH_CHECK_PATH (checker under test), E2E_REGISTERED_REGISTRY (registered draft).
"""
import importlib.machinery
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[1]
SC_PATH = Path(os.environ.get("E2E_SWITCH_CHECK_PATH", REPO / "scripts" / "diagnostics" / "switch_check.py"))
REGISTERED = Path(os.environ.get("E2E_REGISTERED_REGISTRY", REPO / "docs" / "drafts" / "loop_v1" /
                                 "registry_repair_land_20261125" / "register" / "parameter_registry.py.patched"))
TESTS = []


def gate(fn):
    TESTS.append(fn)
    return fn


def check(cond, tag, msg=""):
    if not cond:
        raise AssertionError(f"[{tag}] {msg}")


def _load(path, name):
    ld = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, ld)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    ld.exec_module(m)
    return m


SC = _load(SC_PATH, "switch_check_units_under_test")

REG_SRC = '''PARAMETER_REGISTRIES: Dict[str, ScriptParameterRegistry] = {
    "s.py": ScriptParameterRegistry(
        script_name="s.py",
        description="fixture",
        parameters=[
            ParameterSpec(
                name="mode",
                cli_flag="--mode",
                param_type=ParamType.CHOICE,
                choices=["a", "b"],
                default="a"
            ),
        ]
    ),
}
'''
SCRIPT_SRC = '''import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["a", "b"], default="a", help="registered")
ap.add_argument("--level", type=int, default=3, help="scalar")
ap.add_argument("--flag", action="store_true", help="flag")
ap.add_argument("--steps", type=int, nargs="+", default=[3, 5, 6], help="a list")
ap.add_argument("--maybe", nargs="?", const="x", default=None, help="optional value")
ap.add_argument("--tag", action="append", default=None, help="append")
ap.add_argument("-v", "--verbose", action="count", default=0, help="count")
ap.add_argument("--const", action="store_const", const=1, default=0, help="store_const")
'''


def rows():
    rs, err = SC.parse_argparse(SCRIPT_SRC, "s.py")
    check(err is None, "A-SCU-NARGS", err)
    for r in rs:
        r["tree"] = "main"
        r["step"] = "-"                      # not a WATCHER step (R1), as the walk would set it
        r["registry"] = {"class": "registered and reachable" if r["name"] == "--mode" else "not registered"}
    return rs


def refused(entries, want):
    try:
        SC.select_register_rows(rows(), entries)
    except SC.RegisterError as e:
        return want in str(e), str(e)
    return False, "accepted"


@gate
def test_scu_duplicates():
    preg, _ = SC.parse_param_registry(REG_SRC)
    check(SC.duplicate_specs(preg) == [], "A-SCU-DUP", "a clean registry reported a duplicate")
    dup_src = REG_SRC.replace("        ]\n", "            ParameterSpec(name=\"mode\", cli_flag=\"--mode\"),\n        ]\n", 1)
    dpreg, _ = SC.parse_param_registry(dup_src)
    d = SC.duplicate_specs(dpreg)
    check(len(d) == 1 and d[0][0] == "s.py" and d[0][1] == "mode" and len(d[0][2]) == 2, "A-SCU-DUP", str(d))
    # the hyphen and underscore forms are one name, as get_param treats them
    hy_src = REG_SRC.replace("        ]\n", "            ParameterSpec(name=\"--mode\", cli_flag=\"--mode\"),\n        ]\n", 1)
    check(len(SC.duplicate_specs(SC.parse_param_registry(hy_src)[0])) == 1, "A-SCU-DUP", "'--mode' vs 'mode' not one name")
    real, _ = SC.parse_param_registry(REGISTERED.read_text())
    check(SC.duplicate_specs(real) == [], "A-SCU-DUP", f"the registered draft has duplicates: {SC.duplicate_specs(real)}")


def _patch_and_check(base, flags):
    preg, close = SC.parse_param_registry(base)
    new = [r for r in rows() if r["name"] in flags]
    ctx = {"step_scripts": {}, "step_manifests": {}}
    patched, inserted, conflicts = SC.build_patch(base, preg, close, new, ctx)
    try:
        SC.check_patched(preg, patched, inserted, conflicts, True)
    except SC.RegisterError as e:
        return str(e)
    return None


@gate
def test_scu_patch_duplicate():
    """check_patched (what walk runs before writing any patch) refuses a patched registry that
    lists a name twice in one script."""
    # clean control: a scalar added to a clean registry passes
    check(_patch_and_check(REG_SRC, {"--level"}) is None, "A-SCU-PATCHDUP", "clean patch refused")
    # 1) the insert itself duplicates a name the base already lists
    base1 = REG_SRC.replace("        ]\n", "            ParameterSpec(name=\"level\", cli_flag=\"--level\"),\n        ]\n", 1)
    e1 = _patch_and_check(base1, {"--level"})
    check(e1 is not None and "registered twice" in e1, "A-SCU-PATCHDUP", f"insert duplicate: {e1}")
    # 2) the base already lists `mode` twice; an unrelated insert must not hide it (every dict-based
    #    check collapses the two; only duplicate_specs sees it)
    base2 = REG_SRC.replace("        ]\n", "            ParameterSpec(name=\"mode\", cli_flag=\"--mode\", param_type=ParamType.CHOICE, "
                            "choices=[\"a\", \"b\"], default=\"a\"),\n        ]\n", 1)
    e2 = _patch_and_check(base2, {"--level"})
    check(e2 is not None and "registered twice" in e2 and "mode" in e2, "A-SCU-PATCHDUP", f"base duplicate: {e2}")


@gate
def test_scu_nargs():
    by = {r["name"]: r for r in rows()}
    check(by["--steps"]["nargs"] == "+" and by["--maybe"]["nargs"] == "?" and by["--level"]["nargs"] is None,
          "A-SCU-NARGS", f"nargs not kept: {[(k, v.get('nargs')) for k, v in by.items()]}")
    for flag, nv in (("--steps", "'+'"), ("--maybe", "'?'")):
        ok, msg = refused([{"file": "s.py", "flag": flag}], f"nargs={nv}")
        check(ok, "A-SCU-NARGS", f"{flag}: {msg}")


@gate
def test_scu_actions():
    for flag, act in (("--tag", "append"), ("--verbose", "count"), ("--const", "store_const")):
        ok, msg = refused([{"file": "s.py", "flag": flag}], f"action='{act}'")
        check(ok, "A-SCU-ACTION", f"{flag}: {msg}")
    got = SC.select_register_rows(rows(), [{"file": "s.py", "flag": "--level"}, {"file": "s.py", "flag": "--flag"}])
    check([r["name"] for r in got] == ["--level", "--flag"], "A-SCU-ACTION", f"scalar/flag refused: {got}")


@gate
def test_scu_select():
    for ents, want in (([{"file": "s.py", "flag": "--nope"}], "matches 0 argparse options"),
                       ([{"file": "s.py", "flag": "--level"}] * 2, "duplicate entry"),
                       ([{"file": "s.py", "flag": "--mode"}], "is already registered"),
                       ([], "'entries' must be a non-empty list")):
        ok, msg = refused(ents, want)
        check(ok, "A-SCU-SELECT", f"{ents}: {msg}")


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
    print(f"{len(sel) - fails}/{len(sel)} passed in {time.time() - t0:.1f}s; checker={SC_PATH}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
