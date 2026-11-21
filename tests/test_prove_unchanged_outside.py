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
    new = BASE.replace('def allowed(job):\n    return job.get("offset", 0)',
                       '@deco\ndef allowed(job, phase=0):\n    if phase:\n        raise ValueError\n'
                       '    return job["window_anchor"]')
    assert new != BASE
    rep = _cmp(new)
    assert rep["ok"], rep["problems"]
    assert rep["allowed"]["allowed"]["changed"] is True


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
