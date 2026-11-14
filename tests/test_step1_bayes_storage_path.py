#!/usr/bin/env python3
"""tests/test_step1_bayes_storage_path.py — the Optuna storage path fix (stage 1).

Owner decision 2026-11-14: the two storage strings in window_optimizer_bayesian.py
take their base from the checkout the file runs from, not from the literal
/home/michael/distributed_prng_analysis. From the main tree the URL must be
BYTE-IDENTICAL to the base commit's.

Gates (CPU only; nothing is opened, created or written under any optuna_studies/):

  P1 MAIN-TREE BYTE IDENTITY. Both storage expressions are taken out of the BASE
     source (git show <base>:window_optimizer_bayesian.py) and out of the LIVE
     source by AST, never retyped, and evaluated over the same inputs. The live
     base constant is produced by executing the LIVE module-level assignment with
     __file__ = <main tree>/window_optimizer_bayesian.py (absolute) and again with a
     RELATIVE __file__ under cwd = main tree (the PYTHONPATH=. case). Each pair
     must be equal as bytes.
  P2 CHECKOUT-RELATIVE. The real module imported from this checkout has its base at
     this checkout, and a later chdir does not move it.
  P3 THE STUDY BODY USES IT. The live run_optimization is driven with
     optuna.create_study / optuna.load_study replaced by recorders that raise before
     any storage is touched. Fresh path and resume path: the storage URL handed to
     Optuna lies under THIS checkout. From a worktree it is therefore NOT the main
     tree's live folder.
  Mutants (each must turn its gate red): M1 wrong base (parent dir); M2 the fresh
     literal restored; M3 the candidate literal restored; M4 the base computed at
     call time from the cwd.

Run:  source ~/venvs/torch/bin/activate && PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
      python3 -u tests/test_step1_bayes_storage_path.py
"""
import ast
import contextlib
import io
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_TREE = "/home/michael/distributed_prng_analysis"
BASE_COMMIT = "76739247ef46a0f921ed89d684a209800923b2d7"
REL = "window_optimizer_bayesian.py"
TARGETS = ("_fresh_storage_path", "_candidate_storage")
INPUTS = {"_fresh_study_name": "window_opt_1794600000",
          "_candidate_db": "optuna_studies/window_opt_1794600000.db"}

RESULTS = []


def _check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail or ""))
    except Exception as e:  # noqa: BLE001 — a gate reports, never hides
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


def _base_source():
    return subprocess.run(["git", "-C", ROOT, "show", f"{BASE_COMMIT}:{REL}"],
                          check=True, capture_output=True, text=True).stdout


def _live_source():
    with open(os.path.join(ROOT, REL), encoding="utf-8") as f:
        return f.read()


def _assign_values(src):
    """{target: expr-source} for the two storage assignments, exactly once each."""
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = getattr(node.targets[0], "id", None)
            if t in TARGETS:
                if t in out:
                    raise AssertionError(f"{t} assigned more than once")
                out[t] = ast.unparse(node.value)
    missing = set(TARGETS) - set(out)
    if missing:
        raise AssertionError(f"storage assignment(s) not found: {sorted(missing)}")
    return out


def _module_base_stmt(src):
    """The live module-level statements that define _OPTUNA_STORAGE_BASE (+ its import)."""
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.asname == "_storage_os" for a in node.names):
            keep.append(node)
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "_OPTUNA_STORAGE_BASE" for t in node.targets):
            keep.append(node)
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def _eval_storage(src, base_ns):
    vals = _assign_values(src)
    ns = dict(INPUTS)
    ns.update(base_ns)
    return {t: eval(compile(e, f"<{t}>", "eval"), ns) for t, e in vals.items()}


def _live_base(src, file_value, cwd):
    stmts = _module_base_stmt(src)
    if "_OPTUNA_STORAGE_BASE" not in stmts:
        return {}
    prev = os.getcwd()
    os.chdir(cwd)
    try:
        ns = {"__file__": file_value}
        exec(compile(stmts, "<base>", "exec"), ns)
    finally:
        os.chdir(prev)
    return {"_OPTUNA_STORAGE_BASE": ns["_OPTUNA_STORAGE_BASE"]}


# ---------------------------------------------------------------------------
def gate_p1(live_src=None):
    live_src = live_src if live_src is not None else _live_source()
    old = _eval_storage(_base_source(), {})
    lines = []
    for label, fv in (("absolute __file__", os.path.join(MAIN_TREE, REL)),
                      ("relative __file__", REL),
                      ("dot-relative __file__", "./" + REL)):
        new = _eval_storage(live_src, _live_base(live_src, fv, MAIN_TREE))
        for t in TARGETS:
            if new[t].encode() != old[t].encode():
                raise AssertionError(f"{label}: {t} differs: base={old[t]!r} live={new[t]!r}")
        lines.append(label)
    return f"byte-identical for {', '.join(lines)}: {old['_fresh_storage_path']} | {old['_candidate_storage']}"


def gate_p2():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import importlib
    import window_optimizer_bayesian as wob
    importlib.reload(wob)
    if os.path.dirname(os.path.abspath(wob.__file__)) != ROOT:
        raise AssertionError(f"imported {wob.__file__}, not this checkout {ROOT}")
    b0 = wob._OPTUNA_STORAGE_BASE
    if b0 != ROOT:
        raise AssertionError(f"base {b0!r} != checkout {ROOT!r}")
    prev = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="storage_p2_")
    os.chdir(tmp)
    try:
        if wob._OPTUNA_STORAGE_BASE != b0:
            raise AssertionError("base moved on chdir")
    finally:
        os.chdir(prev)
        os.rmdir(tmp)
    return f"base={b0}"


class _Stop(BaseException):
    pass


def _drive_study_body(module, *, resume):
    """Run the live run_optimization until Optuna is first asked for storage."""
    import optuna
    seen = []
    real = (optuna.create_study, optuna.load_study)

    def rec_create(*a, **k):
        seen.append(("create_study", k.get("storage")))
        raise _Stop()

    def rec_load(*a, **k):
        seen.append(("load_study", k.get("storage")))
        raise RuntimeError("recorder: load refused")  # caught by the resume loop

    optuna.create_study, optuna.load_study = rec_create, rec_load
    prev = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="storage_p3_")
    try:
        os.chdir(tmp)
        if resume:
            # The resume loop checks a RELATIVE optuna_studies/<name>.db exists.
            # An EMPTY placeholder in the scratch cwd; it is never opened.
            os.makedirs("optuna_studies")
            open(os.path.join("optuna_studies", "window_opt_1794600000.db"), "w").close()
        from window_optimizer import SearchBounds
        search = module.OptunaBayesianSearch(n_startup_trials=1, seed=3)
        sampler = optuna.samplers.RandomSampler(seed=3)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                search.run_optimization(
                    lambda c, optuna_trial=None: None, SearchBounds(), 1, None,
                    sampler=sampler,
                    sampler_metadata=module.describe_sampler(sampler, strategy="storage_gate", seed=3),
                    resume_study=resume,
                    study_name="window_opt_1794600000" if resume else "")
        except _Stop:
            pass
    finally:
        optuna.create_study, optuna.load_study = real
        os.chdir(prev)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return seen


def gate_p3(module=None):
    if module is None:
        import window_optimizer_bayesian as module
    out = []
    for resume in (False, True):
        seen = _drive_study_body(module, resume=resume)
        if not seen:
            raise AssertionError(f"resume={resume}: Optuna was never asked for storage")
        for fn, url in seen:
            want = f"sqlite:///{ROOT}/optuna_studies/"
            if not str(url).startswith(want):
                raise AssertionError(f"resume={resume}: {fn} storage {url!r} is not under {want!r}")
            if ROOT != MAIN_TREE and str(url).startswith(f"sqlite:///{MAIN_TREE}/"):
                raise AssertionError(f"resume={resume}: {fn} storage {url!r} is the LIVE main tree")
            out.append(f"{fn}={url}")
    return "; ".join(out)


# ---------------------------------------------------------------------------
def _mutate(src, old, new):
    if src.count(old) != 1:
        raise AssertionError(f"mutant anchor found {src.count(old)} times: {old!r}")
    return src.replace(old, new)


def _load_mutant_module(src, name):
    import types
    m = types.ModuleType(name)
    m.__file__ = os.path.join(ROOT, REL)
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


def run_mutants():
    live = _live_source()
    base_line = "_OPTUNA_STORAGE_BASE = _storage_os.path.dirname(_storage_os.path.abspath(__file__))"
    fresh_new = 'f"sqlite:///{_OPTUNA_STORAGE_BASE}/optuna_studies/{_fresh_study_name}.db"'
    cand_new = 'f"sqlite:///{_OPTUNA_STORAGE_BASE}/{_candidate_db}"'
    mutants = [
        ("M1 wrong base (parent dir)", "p1",
         _mutate(live, base_line,
                 "_OPTUNA_STORAGE_BASE = _storage_os.path.dirname(_storage_os.path.dirname(_storage_os.path.abspath(__file__)))")),
        ("M2 fresh literal restored", "p3",
         _mutate(live, fresh_new,
                 'f"sqlite:////home/michael/distributed_prng_analysis/optuna_studies/{_fresh_study_name}.db"')),
        ("M3 candidate literal restored", "p3",
         _mutate(live, cand_new, 'f"sqlite:////home/michael/distributed_prng_analysis/{_candidate_db}"')),
        ("M4 base from call-time cwd", "p3",
         _mutate(live, fresh_new, 'f"sqlite:///{_storage_os.getcwd()}/optuna_studies/{_fresh_study_name}.db"')),
    ]
    rows = []
    for label, gate, src in mutants:
        try:
            if gate == "p1":
                gate_p1(src)
            else:
                if ROOT == MAIN_TREE and label.startswith(("M2", "M3")):
                    rows.append((label, None, "not discriminating from the main tree (expected)"))
                    continue
                gate_p3(_load_mutant_module(src, "_mut_" + label[:2]))
            rows.append((label, False, "gate stayed GREEN"))
        except AssertionError as e:
            rows.append((label, True, f"RED: {e}"))
    return rows


def main():
    print(f"checkout: {ROOT}")
    _check("P1 MAIN-TREE BYTE IDENTITY", gate_p1)
    _check("P2 CHECKOUT-RELATIVE BASE", gate_p2)
    _check("P3 STUDY BODY USES THE CHECKOUT", gate_p3)
    ok = True
    for name, passed, detail in RESULTS:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print("mutants:")
    for label, red, detail in run_mutants():
        if red is None:
            print(f"  [SKIP] {label}: {detail}")
            continue
        ok &= red
        print(f"  [{'RED (detected)' if red else 'NOT DETECTED'}] {label}: {detail}")
    print(f"COMPLETION SENTINEL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
