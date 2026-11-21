#!/usr/bin/env python3
"""Tests for scripts/prove_unchanged_outside.py (PWC-BACKUP Stage 3 proof tool).

CPU only, read-only on the repository. Fixture files go under
logs/test_prove_unchanged_outside/ (never /tmp). Runs as a script
(`python3 tests/test_prove_unchanged_outside.py`) or under pytest.

PASS cases: the tool must report no unexpected difference.
CATCH cases: deliberate breakages the tool must report (exit 1), each naming
what changed.
"""
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "scripts", "prove_unchanged_outside.py")
FIX = os.path.join(ROOT, "logs", "test_prove_unchanged_outside")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import prove_unchanged_outside as P  # noqa: E402

BASE = textwrap.dedent('''\
    """module docstring"""
    import os
    import sys

    LIMIT = 268435456
    os.environ.setdefault("X_FLAG", "1")

    def deco(f):
        return f

    @deco
    def checked(a, b=1):
        def inner(x):
            return x + 1
        return inner(a) + b

    def allowed(job):
        return job.get("offset", 0)

    class Sieve:
        def run(self, offset=0):
            return offset

        def other(self):
            return 2

    try:
        import cupy
    except ImportError:
        cupy = None

        def fallback():
            return None
    ''')


def _cmp(new, allow=("allowed",), base=BASE):
    return P.compare(base, new, allow)


def _kinds(rep):
    return {p["kind"] for p in rep["problems"]}


def _defs(rep):
    for p in rep["problems"]:
        if p["kind"] == "definitions-differ":
            return {d["name"]: d["change"] for d in p["definitions"]}
    return {}


def _cli(new_src, *args, base_src=BASE):
    os.makedirs(FIX, exist_ok=True)
    b, n = os.path.join(FIX, "base.py"), os.path.join(FIX, "new.py")
    with open(b, "w") as f:
        f.write(base_src)
    with open(n, "w") as f:
        f.write(new_src)
    return subprocess.run([sys.executable, TOOL, "--base-file", b, "--new-file", n, *args],
                          capture_output=True, text=True)


# ------------------------------------------------------------------ PASS cases

def test_pass_identical():
    rep = _cmp(BASE)
    assert rep["ok"], rep["problems"]


def test_pass_change_inside_allowed_function():
    # body, signature, default and a new decorator: all inside the allowed def
    # body, signature and a literal default: inside the allowed def, inert at import
    new = BASE.replace('def allowed(job):\n    return job.get("offset", 0)',
                       'def allowed(job, phase=0):\n    if phase:\n        raise ValueError\n'
                       '    return job["window_anchor"]')
    assert new != BASE
    rep = _cmp(new)
    assert rep["ok"], rep["problems"]
    assert rep["allowed"]["allowed"]["changed"] is True
    assert rep["allowed_header_changes"] and not rep["allowed_header_changes"][0]["issues"]


def test_pass_change_inside_allowed_method():
    new = BASE.replace("    def run(self, offset=0):\n        return offset",
                       "    def run(self, offset=0, phase=0):\n        return offset + phase")
    assert new != BASE
    assert _cmp(new, allow=("Sieve.run",))["ok"]


def test_pass_comment_only_change_is_reported_not_failed():
    new = BASE.replace("LIMIT = 268435456", "LIMIT = 268435456  # 256 MiB")
    assert _cmp(new)["ok"]
    r = _cli(new, "--allow", "allowed")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "changed lines outside allowed definitions" in r.stdout
    r = _cli(new, "--allow", "allowed", "--strict-bytes")
    assert r.returncode == 1, r.stdout


# ------------------------------------------------------------------ CATCH cases

def test_catch_changed_default_argument():
    new = BASE.replace("def checked(a, b=1):", "def checked(a, b=2):")
    rep = _cmp(new)
    assert not rep["ok"]
    assert _defs(rep).get("checked") == "changed"


def test_catch_changed_default_in_non_allowed_method():
    new = BASE.replace("    def run(self, offset=0):", "    def run(self, offset=1):")
    rep = _cmp(new)
    assert not rep["ok"]
    assert _defs(rep).get("Sieve.run") == "changed"
    assert _defs(rep).get("Sieve") == "changed"


def test_catch_changed_decorator():
    new = BASE.replace("@deco\ndef checked", "@staticmethod\ndef checked")
    assert new != BASE
    rep = _cmp(new)
    assert not rep["ok"] and _defs(rep).get("checked") == "changed"
    # decorator removed entirely
    rep = _cmp(BASE.replace("@deco\ndef checked", "def checked"))
    assert not rep["ok"] and _defs(rep).get("checked") == "changed"


def test_catch_moved_import_reordered():
    new = BASE.replace("import os\nimport sys\n", "import sys\nimport os\n")
    assert new != BASE
    rep = _cmp(new)
    assert not rep["ok"] and "top-level-statements-differ" in _kinds(rep)


def test_catch_import_moved_below_side_effect():
    new = BASE.replace('import sys\n', '').replace(
        'os.environ.setdefault("X_FLAG", "1")\n', 'os.environ.setdefault("X_FLAG", "1")\nimport sys\n')
    assert new != BASE
    rep = _cmp(new)
    assert not rep["ok"] and "top-level-statements-differ" in _kinds(rep)


def test_catch_import_moved_into_allowed_function_is_still_caught():
    # removing a module-level import is a top-level change even if the allowed
    # function now imports it itself
    new = BASE.replace("import sys\n", "").replace(
        'def allowed(job):\n    return', 'def allowed(job):\n    import sys\n    return')
    rep = _cmp(new)
    assert not rep["ok"] and "top-level-statements-differ" in _kinds(rep)


def test_catch_changed_module_constant():
    rep = _cmp(BASE.replace("LIMIT = 268435456", "LIMIT = 536870912"))
    assert not rep["ok"] and "top-level-statements-differ" in _kinds(rep)
    hunk = [p for p in rep["problems"] if p["kind"] == "top-level-statements-differ"][0]["hunks"]
    assert any("LIMIT" in s["stmt"] for h in hunk for s in h["new"])


def test_catch_changed_module_side_effect():
    rep = _cmp(BASE.replace('setdefault("X_FLAG", "1")', 'setdefault("X_FLAG", "0")'))
    assert not rep["ok"] and "top-level-statements-differ" in _kinds(rep)


def test_catch_change_inside_nested_function_of_non_allowed_function():
    new = BASE.replace("        return x + 1", "        return x + 2")
    assert new != BASE
    rep = _cmp(new)
    d = _defs(rep)
    assert not rep["ok"] and d.get("checked.inner") == "changed" and d.get("checked") == "changed"
    marked = [x for p in rep["problems"] if p["kind"] == "definitions-differ"
              for x in p["definitions"] if x["name"] == "checked"][0]
    assert marked["has_changed_child"] is True


def test_catch_parent_change_when_only_nested_is_allowed():
    new = BASE.replace("    return inner(a) + b", "    return inner(a) - b")
    rep = _cmp(new, allow=("checked.inner",))
    assert not rep["ok"] and _defs(rep).get("checked") == "changed"


def test_catch_other_method_when_one_method_allowed():
    new = BASE.replace("        return 2", "        return 3")
    rep = _cmp(new, allow=("Sieve.run",))
    assert not rep["ok"] and _defs(rep).get("Sieve.other") == "changed"


def test_catch_reordered_methods():
    a = "    def run(self, offset=0):\n        return offset\n"
    b = "    def other(self):\n        return 2\n"
    new = BASE.replace(a + "\n" + b, b + "\n" + a)
    assert new != BASE
    rep = _cmp(new)
    assert not rep["ok"] and _defs(rep).get("Sieve") == "changed"


def test_catch_function_inside_module_try_block():
    rep = _cmp(BASE.replace("        return None", "        return 0"))
    assert not rep["ok"] and _defs(rep).get("fallback") == "changed"


def test_catch_added_and_removed_definitions():
    rep = _cmp(BASE + "\ndef extra():\n    return 1\n")
    assert not rep["ok"] and _defs(rep).get("extra") == "added"
    rep = _cmp(BASE.replace("def deco(f):\n    return f\n", ""))
    assert not rep["ok"] and _defs(rep).get("deco") == "removed"


def test_catch_allowed_prefix_does_not_cover_longer_name():
    base = BASE + "\ndef allowedness():\n    return 1\n"
    new = BASE + "\ndef allowedness():\n    return 2\n"
    rep = _cmp(new, base=base)
    assert not rep["ok"] and _defs(rep).get("allowedness") == "changed"


def test_catch_allowed_function_moved():
    blk = 'def allowed(job):\n    return job.get("offset", 0)\n\n'
    assert blk in BASE
    new = BASE.replace(blk, "").replace("def deco(f):", blk + "def deco(f):")
    rep = _cmp(new)
    assert not rep["ok"] and "top-level-statements-differ" in _kinds(rep)


def test_catch_missing_allowed_name():
    rep = _cmp(BASE, allow=("allowed", "no_such_function"))
    assert not rep["ok"] and "allowed-name-not-unique-or-missing" in _kinds(rep)


def test_catch_docstring_change_outside():
    rep = _cmp(BASE.replace('"""module docstring"""', '"""module docstring v2"""'))
    assert not rep["ok"]


# ------------------------------------------------------------------ CLI + real file

def test_cli_exit_codes():
    assert _cli(BASE, "--allow", "allowed").returncode == 0
    r = _cli(BASE.replace("def checked(a, b=1):", "def checked(a, b=2):"), "--allow", "allowed")
    assert r.returncode == 1 and "changed: checked" in r.stdout
    r = _cli("def broken(:\n", "--allow", "allowed")
    assert r.returncode == 2
    r = subprocess.run([sys.executable, TOOL, "--base", "no_such_rev_zz", "--file", "sieve_filter.py"],
                       capture_output=True, text=True)
    assert r.returncode == 2


def test_real_sieve_filter_against_stage2b_base():
    """The Stage 2b edit to sieve_filter.py, against 0caeaf9 (Stage 2 landed):
    unchanged outside execute_sieve_job and GPUSieve.run_hybrid_sieve; and the
    same comparison allowing only execute_sieve_job must fail and name
    GPUSieve.run_hybrid_sieve. UNAVAILABLE (not passed) without 0caeaf9."""
    if subprocess.run(["git", "-C", ROOT, "cat-file", "-e", "0caeaf9^{commit}"]).returncode != 0:
        print("  [UNAVAILABLE] 0caeaf9 not in this repository")
        return
    ok = subprocess.run([sys.executable, TOOL, "--base", "0caeaf9", "--file", "sieve_filter.py",
                         "--allow", "execute_sieve_job", "GPUSieve.run_hybrid_sieve"],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = subprocess.run([sys.executable, TOOL, "--base", "0caeaf9", "--file", "sieve_filter.py",
                          "--allow", "execute_sieve_job"], capture_output=True, text=True)
    if "identical" in bad.stdout.split("bytes", 1)[-1].splitlines()[0]:
        return  # working tree equals base: nothing to catch
    assert bad.returncode == 1 and "GPUSieve.run_hybrid_sieve" in bad.stdout, bad.stdout


# ------------------------------------------------------------------ --allow-added

def test_pass_allowed_added_function_and_constant():
    new = BASE.replace("def deco(f):", "NEW_PIN = 0\n\n\ndef new_helper(x):\n    return x\n\n\ndef deco(f):")
    assert new != BASE
    rep = P.compare(BASE, new, ["allowed"], ["NEW_PIN", "new_helper"])
    assert rep["ok"], rep["problems"]


def test_catch_added_function_not_listed():
    new = BASE.replace("def deco(f):", "def new_helper(x):\n    return x\n\n\ndef deco(f):")
    rep = P.compare(BASE, new, ["allowed"], [])
    assert not rep["ok"]


def test_catch_allowed_added_name_already_in_base():
    # "LIMIT" exists in the base, so it cannot be declared as added
    new = BASE.replace("LIMIT = 268435456", "LIMIT = 1")
    rep = P.compare(BASE, new, ["allowed"], ["LIMIT"])
    assert not rep["ok"] and "allowed-added-name-invalid" in _kinds(rep)


def test_catch_allowed_added_missing_from_new():
    rep = P.compare(BASE, BASE, ["allowed"], ["NEVER_ADDED"])
    assert not rep["ok"] and "allowed-added-name-invalid" in _kinds(rep)


def test_catch_side_effect_hidden_next_to_allowed_added():
    # an added constant is allowed; an added call statement beside it is not
    new = BASE.replace("def deco(f):", "NEW_PIN = 0\nos.environ['Y'] = '1'\n\n\ndef deco(f):")
    rep = P.compare(BASE, new, ["allowed"], ["NEW_PIN"])
    assert not rep["ok"] and "top-level-statements-differ" in _kinds(rep)



# ------------------------------------------------------------------ Stage 3a: Beta X-PWC-STAGE2B R1 Q8

def test_catch_decorator_added_to_allowed_function():
    new = BASE.replace('def allowed(job):', '@deco\ndef allowed(job):')
    rep = _cmp(new)
    assert not rep["ok"] and "allowed-header-import-time-evaluation" in _kinds(rep)


def test_catch_call_default_added_to_allowed_function():
    new = BASE.replace('def allowed(job):', 'def allowed(job, t=os.getpid()):')
    rep = _cmp(new)
    assert not rep["ok"] and "allowed-header-import-time-evaluation" in _kinds(rep)


def test_pass_unchanged_decorator_on_allowed_function():
    # the checked function already has @deco in the base: allowing it and changing
    # only its body must not be flagged for the unchanged decorator
    new = BASE.replace("    return inner(a) + b", "    return inner(a) * b")
    rep = P.compare(BASE, new, ["checked"])
    assert rep["ok"], rep["problems"]


# ------------------------------------------------------------------ Stage 3a, narrowed header check (owner 2026-11-21)
# A non-inert default/annotation in an allowed definition's header is flagged only
# when its AST differs from the same-named parameter's in the base, or the parameter
# is new. Reordering or re-kinding existing parameters is always caught.

HDR_BASE = textwrap.dedent('''\
    import os

    def deco(f):
        return f

    def run_opt(trials, seed=0, staging_high_water_bytes=16 * 1024 ** 3, *, tag=os.sep):
        return trials
    ''')


def _hdr(new, allow=("run_opt",)):
    return P.compare(HDR_BASE, new, allow)


def test_pass_unchanged_noninert_default_with_new_trailing_inert_parameter():
    # (a) the run_bayesian_optimization shape: the 16 * 1024 ** 3 default is unchanged,
    # only an inert keyword-only parameter is appended
    new = HDR_BASE.replace("tag=os.sep):", "tag=os.sep, miner_abi=None):")
    rep = _hdr(new)
    assert rep["ok"], rep["problems"]
    note = rep["allowed_header_changes"][0]
    assert note["issues"] == [] and note["order_issues"] == [] and "parameters" in note["changed"]


def test_pass_unchanged_noninert_default_with_new_trailing_positional_parameter():
    new = HDR_BASE.replace("staging_high_water_bytes=16 * 1024 ** 3, *",
                           "staging_high_water_bytes=16 * 1024 ** 3, extra=None, *")
    rep = _hdr(new)
    assert rep["ok"], rep["problems"]


def test_catch_changed_noninert_default():
    # (b)
    new = HDR_BASE.replace("16 * 1024 ** 3", "8 * 1024 ** 3")
    rep = _hdr(new)
    assert not rep["ok"] and "allowed-header-import-time-evaluation" in _kinds(rep)
    assert any("staging_high_water_bytes" in i for p in rep["problems"] for i in p.get("issues", []))


def test_catch_new_parameter_with_noninert_default():
    # (c) for keyword-only and positional additions, and a non-inert annotation
    for new in (HDR_BASE.replace("tag=os.sep):", "tag=os.sep, pid=os.getpid()):"),
                HDR_BASE.replace("seed=0,", "seed=0, cap=2 ** 30,"),
                HDR_BASE.replace("tag=os.sep):", "tag=os.sep, cb: deco(int) = None):")):
        rep = _hdr(new)
        assert not rep["ok"] and "allowed-header-import-time-evaluation" in _kinds(rep), new


def test_catch_reordered_parameters():
    # (d) swapping two defaulted parameters; also a pure swap of default-less ones,
    # which leaves decorators/defaults/annotations unchanged
    new = HDR_BASE.replace("seed=0, staging_high_water_bytes=16 * 1024 ** 3",
                           "staging_high_water_bytes=16 * 1024 ** 3, seed=0")
    rep = _hdr(new)
    assert not rep["ok"] and "allowed-header-parameter-order-changed" in _kinds(rep)
    base2 = HDR_BASE.replace("def run_opt(trials, seed=0,", "def run_opt(trials, study, seed=0,")
    new2 = base2.replace("def run_opt(trials, study,", "def run_opt(study, trials,")
    rep2 = P.compare(base2, new2, ["run_opt"])
    assert not rep2["ok"] and "allowed-header-parameter-order-changed" in _kinds(rep2)


def test_catch_parameter_kind_change():
    # moving an existing parameter behind * (positional -> keyword-only), order kept
    new = HDR_BASE.replace("seed=0, staging_high_water_bytes=16 * 1024 ** 3, *,",
                           "seed=0, *, staging_high_water_bytes=16 * 1024 ** 3,")
    rep = _hdr(new)
    assert not rep["ok"] and "allowed-header-parameter-order-changed" in _kinds(rep)


def test_catch_decorator_added_next_to_unchanged_noninert_default():
    new = HDR_BASE.replace("def run_opt(", "@deco\ndef run_opt(")
    rep = _hdr(new)
    assert not rep["ok"] and "allowed-header-import-time-evaluation" in _kinds(rep)


def test_catch_allowed_added_with_side_effect_value():
    # Beta's example: PIN = os.environ.setdefault('FLAG', '1')
    new = BASE.replace("def deco(f):", "PIN = os.environ.setdefault('FLAG', '1')\n\n\ndef deco(f):")
    rep = P.compare(BASE, new, ["allowed"], ["PIN"])
    assert not rep["ok"] and "allowed-added-import-time-evaluation" in _kinds(rep)


def test_catch_allowed_added_shadowing_an_import():
    # Beta's example: time = 0 after a base `import time`
    base = BASE.replace("import sys\n", "import sys\nimport time\n")
    new = base.replace("def deco(f):", "time = 0\n\n\ndef deco(f):")
    rep = P.compare(base, new, ["allowed"], ["time"])
    assert not rep["ok"] and "allowed-added-name-invalid" in _kinds(rep)
    # `from x import y as time` and `import a.b` (binds a) count too
    base2 = BASE.replace("import sys\n", "import sys\nfrom os import path as time\nimport xml.dom\n")
    for nm in ("time", "xml"):
        new2 = base2.replace("def deco(f):", f"{nm} = 0\n\n\ndef deco(f):")
        assert "allowed-added-name-invalid" in _kinds(P.compare(base2, new2, ["allowed"], [nm])), nm


def test_catch_allowed_added_rebound_later():
    new = BASE.replace("def deco(f):", "NEW_PIN = 0\n\n\ndef deco(f):") + "\nNEW_PIN = 1\n"
    rep = P.compare(BASE, new, ["allowed"], ["NEW_PIN"])
    assert not rep["ok"]


def test_catch_allowed_added_function_with_decorator_or_call_default():
    for hdr in ("@deco\ndef new_helper(x):", "def new_helper(x=os.getcwd()):",
                "def new_helper(x: make_type()):"):
        new = BASE.replace("def deco(f):", f"{hdr}\n    return x\n\n\ndef deco(f):")
        rep = P.compare(BASE, new, ["allowed"], ["new_helper"])
        assert [p["kind"] for p in rep["problems"]] == ["allowed-added-import-time-evaluation"], \
            (hdr, rep["problems"])
    # the same helper with an inert header passes
    ok = BASE.replace("def deco(f):", "def new_helper(x: int = 0) -> int:\n    return x\n\n\ndef deco(f):")
    assert P.compare(BASE, ok, ["allowed"], ["new_helper"])["ok"]


def test_catch_allowed_added_class_with_body_side_effect():
    new = BASE.replace("def deco(f):", "class NewThing:\n    X = 1\n    print('hi')\n\n\ndef deco(f):")
    rep = P.compare(BASE, new, ["allowed"], ["NewThing"])
    assert not rep["ok"] and "allowed-added-import-time-evaluation" in _kinds(rep)
    ok = BASE.replace("def deco(f):", 'class NewThing:\n    """doc"""\n    X = 1\n\n    def m(self, a: int = 0) -> int:\n        return a\n\n\ndef deco(f):')
    assert P.compare(BASE, ok, ["allowed"], ["NewThing"])["ok"]


ROUTED = textwrap.dedent('''\
    def route(coordinator, x):
        _use_miner = getattr(coordinator, "use_range_miner", False)
        if _use_miner:
            r = x + 1
            return r
        _use_pw = getattr(coordinator, "use_persistent_workers", False)
        if _use_pw:
            return x + 2
        return x
    ''')


def test_pass_keep_block_when_only_other_branch_changes():
    new = ROUTED.replace("return x + 2", "return x + 3")
    rep = P.compare(ROUTED, new, ["route"], (), ["route::if _use_miner:"])
    assert rep["ok"], rep["problems"]
    assert rep["kept_blocks"] == [{"spec": "route::if _use_miner:", "ok": True}]


def test_catch_keep_block_change_inside_kept_branch():
    new = ROUTED.replace("r = x + 1", "r = x + 9")
    assert P.compare(ROUTED, new, ["route"])["ok"]                       # hidden without --keep-block
    rep = P.compare(ROUTED, new, ["route"], (), ["route::if _use_miner:"])
    assert not rep["ok"] and "kept-block-differs" in _kinds(rep)


def test_catch_keep_block_preamble_change():
    new = ROUTED.replace('getattr(coordinator, "use_range_miner", False)',
                         'getattr(coordinator, "use_range_miner", True)')
    rep = P.compare(ROUTED, new, ["route"], (), ["route::if _use_miner:"])
    assert not rep["ok"] and "kept-block-preamble-differs" in _kinds(rep)
    # a statement inserted before the block counts as a preamble change
    new2 = ROUTED.replace("    _use_miner =", "    x = x * 2\n    _use_miner =")
    assert "kept-block-preamble-differs" in _kinds(P.compare(ROUTED, new2, ["route"], (), ["route::if _use_miner:"]))
    # and so does a signature change
    new3 = ROUTED.replace("def route(coordinator, x):", "def route(coordinator, x, y=0):")
    assert "kept-block-preamble-differs" in _kinds(P.compare(ROUTED, new3, ["route"], (), ["route::if _use_miner:"]))


def test_catch_keep_block_missing_or_duplicated():
    gone = ROUTED.replace("if _use_miner:", "if _use_miner and x:")
    assert "kept-block-not-unique" in _kinds(P.compare(ROUTED, gone, ["route"], (), ["route::if _use_miner:"]))
    assert "kept-block-function-not-unique" in _kinds(P.compare(ROUTED, ROUTED, ["route"], (), ["nope::if _use_miner:"]))


def test_cli_keep_block_and_bad_spec():
    new = ROUTED.replace("r = x + 1", "r = x + 9")
    r = _cli(new, "--allow", "route", "--keep-block", "route::if _use_miner:", base_src=ROUTED)
    assert r.returncode == 1 and "kept-block-differs" in r.stdout, r.stdout
    r = _cli(ROUTED, "--allow", "route", "--keep-block", "no-separator", base_src=ROUTED)
    assert r.returncode == 2

def _main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS {n}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {n}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
