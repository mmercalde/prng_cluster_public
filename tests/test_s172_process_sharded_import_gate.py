#!/usr/bin/env python3
"""
test_s172_process_sharded_import_gate.py — S172 `process_sharded` CPU-only
IMPORT gate (Beta-REQUIRED hardening, brief
docs/CLAUDE_CODE_INSTRUCTIONS_S172_PROCESS_SHARDED_IMPORT_GATE.md REV1).

Subject under test: the CPU-only invariant that
`miner/assembly_shard_worker.assert_cpu_only()` asserts, exercised END TO END
against a FRESH interpreter that imports the real Step-1 module surface.

WHY THIS FILE EXISTS SEPARATELY FROM THE D5 SUITE
    Gap 2.1 of the brief requires an AST assertion that NO GPU-module-name
    literal appears anywhere in the new gate — the probe must read the
    forbidden list from production instead of restating it. The D5 suite's
    existing `g_no_gpu` arms legitimately contain those literals (their AST
    arms are written against them), so a whole-file assertion cannot be made
    there without either weakening it to a line range or editing arms the
    brief freezes. A separate file makes the assertion total and unambiguous:
    every name this gate checks is READ FROM `_FORBIDDEN_GPU_MODULES`, and
    G-NO-DUP-LIST proves it by scanning this file's own source.

    This file is reached from the D5 suite's `main()` as one added `_check`
    row, so it cannot silently never run.

THE FOUR GAPS THIS CLOSES (brief §2), none of which the existing arms cover
    2.1 the existing probe hardcodes the forbidden pair at :1436 instead of
        reading production's authority at assembly_shard_worker.py:170;
    2.2 the existing probe inspects `sys.modules` itself rather than INVOKING
        the production guard — delete the guard and that arm still passes;
    2.3 a multiprocessing child reaches the worker module but never the real
        Step-1 module surface, so the import-graph claim in the guard's own
        docstring is untested end to end;
    2.4 runtime fault injection covers ONE of the two forbidden modules.

WHAT "THE REAL STEP-1 MODULE SURFACE" MEANS HERE — and why
    It is DERIVED, never named: `_step1_surface()` AST-extracts the
    module-scope imports of the miner package made by Step-1's host module
    (`window_optimizer_integration_final.py`), which today are `miner` and
    `miner.step1_ingress`. Those two transitively pull the whole graph the
    guard's docstring makes its claim about — the D1.1 engine
    (`range_miner_npz_writer`) -> the coordinator -> `range_miner_worker` —
    plus the assembly backends and the shard worker itself.

    It is deliberately NOT the host module itself. Step-1's host imports
    `sieve_filter` at module scope, and that module imports a GPU library at
    module scope, so the Step-1 PROCESS legitimately holds a GPU context — it
    is the sieve host. The guard's claim is about the MINER SUBGRAPH Step-1
    pulls in, not about Step-1's own interpreter. G-HOST-BOUNDARY pins exactly
    that distinction, so the exclusion stays a measured fact rather than a
    convenience: it reds if the miner package ever becomes the culprit.

VERIFICATION-INTEGRITY CONTROLS (brief VIR-1..6)
    execution proof   every child prints the forbidden tuple it RESOLVED from
                      production, the surface modules it was asked to import,
                      the miner modules that actually landed in `sys.modules`
                      and the file each resolved from — a child that silently
                      did nothing is distinguishable from one that passed.
    clean control     the unmutated tree runs through the SAME runner and must
                      pass (G-SURFACE-GUARD), and is the positive control that
                      makes every mutant red attributable to its mutation.
    fault injection   §3's module-scope-import mutants, plus §2.4's per-module
                      runtime injection driven off the production tuple.
    sentinel          PASS | FAIL | UNAVAILABLE | INCOMPLETE. Only PASS
                      accepts; a child that dies without printing its sentinel
                      is INCOMPLETE, never PASS.
    no fleet need     nothing here touches a rig. UNAVAILABLE is a finding.

Run:
    cd ~/distributed_prng_analysis
    source the project virtualenv
    PYTHONPATH=. python3 tests/test_s172_process_sharded_import_gate.py
Exit 0 = all green. Exit 1 = a check failed (DO NOT COMMIT).
"""
import ast
import io
import os
import shutil
import subprocess
import sys
import tempfile
import tokenize
import traceback
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import miner.assembly_shard_worker as ASW      # noqa: E402

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_results: List[Tuple[str, bool, Any]] = []
_MUTANTS: List[Tuple[str, str, str, str]] = []
_EVIDENCE: List[Tuple[str, str]] = []

_SELF_PATH = os.path.abspath(__file__)
_STEP1_HOST_PATH = os.path.join(_ROOT, "window_optimizer_integration_final.py")
_WORKER_PATH = os.path.join(_ROOT, "miner", "assembly_shard_worker.py")
_MINER_PKG = "miner"

# The production authority. Read, never restated — this single reference is the
# ONLY place a forbidden module name enters this file, and G-NO-DUP-LIST proves
# no literal one ever does.
_FORBIDDEN: Tuple[str, ...] = tuple(ASW._FORBIDDEN_GPU_MODULES)

# The transitive miner graph the guard's docstring makes its claim about. These
# are miner module names, not GPU module names, so writing them is not a
# duplication of the forbidden list.
_EXPECTED_TRANSITIVE: Tuple[str, ...] = (
    "miner.range_miner_npz_writer",      # the D1.1 engine
    "miner.range_miner_coordinator",     # ...which imports the coordinator
    "miner.range_miner_worker",          # ...which imports that module
    "miner.assembly_backends",
    "miner.assembly_shard_worker",
    "miner.step1_ingress",
)

_SENTINELS = ("PASS", "FAIL", "UNAVAILABLE", "INCOMPLETE")
_TIMEOUT = 300
_CENSUS_BEFORE: List[str] = []


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:                                      # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    except Exception as other:                                  # noqa: BLE001
        raise AssertionError(
            f"expected {exc.__name__}, got {type(other).__name__}: {other}")
    raise AssertionError(f"expected {exc.__name__}, nothing was raised")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ═════════════════════════════════════════════════════════════════════════════
# The Step-1 surface — DERIVED from Step-1's own source, never named here
# ═════════════════════════════════════════════════════════════════════════════
def _module_scope_imports(tree: ast.Module) -> List[str]:
    """Every absolute import made at MODULE scope — descending into `try`,
    `if` and `with` (Step-1 guards its optional imports in `try/except
    ImportError`) but never into a function or class body, because an import
    inside a function does not run at import time and therefore cannot put a
    library into a fresh interpreter's `sys.modules`."""
    found: List[str] = []

    def walk(nodes):
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
                continue
            if isinstance(n, ast.Import):
                found.extend(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                found.append(n.module)
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(n, field, None)
                if isinstance(sub, list):
                    walk(sub)
            for handler in getattr(n, "handlers", []) or []:
                walk(handler.body)

    walk(tree.body)
    return found


def _step1_constant(name: str) -> str:
    """Read a module-scope string constant out of Step-1's SOURCE.

    Deliberately AST, not import: importing Step-1's host module into THIS
    process would pull a GPU library into the parent interpreter and break
    G-RUNTIME-INJECTION's negative controls, which require a clean process."""
    tree = ast.parse(_read(_STEP1_HOST_PATH), filename=_STEP1_HOST_PATH)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return node.value.value
    raise AssertionError(
        f"Step-1's host module no longer defines {name} as a module-scope "
        f"string constant — the contamination guard cannot be derived")


# Where Step-1 would scatter checkpoint directories if a flush ever ran, and the
# env var that redirects it. Both READ FROM Step-1's source, never restated.
_CHECKPOINT_DIRNAME = _step1_constant("_CHECKPOINT_DIRNAME")
_CHECKPOINT_ROOT_ENV = _step1_constant("_CHECKPOINT_ROOT_ENV")
# `_flush_checkpoint_root()` falls back to the directory holding Step-1's host
# module, which is the repo root — the tree a retention census runs against.
_REAL_CHECKPOINT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(_STEP1_HOST_PATH)), _CHECKPOINT_DIRNAME)


def _checkpoint_census() -> List[str]:
    """Every run-id directory currently under the REAL checkpoint root."""
    if not os.path.isdir(_REAL_CHECKPOINT_ROOT):
        return []
    return sorted(os.listdir(_REAL_CHECKPOINT_ROOT))


def _step1_surface() -> List[str]:
    """The miner-package modules Step-1 imports at module scope, in source
    order. This is the gate's definition of 'the real Step-1 module surface'
    and it is machine-derived: if Step-1 gains a third miner import, the
    surface widens automatically and the gate keeps covering all of it."""
    tree = ast.parse(_read(_STEP1_HOST_PATH), filename=_STEP1_HOST_PATH)
    seen, out = set(), []
    for name in _module_scope_imports(tree):
        if name != _MINER_PKG and not name.startswith(_MINER_PKG + "."):
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# The fresh-interpreter child — a real file, run via `sys.executable`
# ═════════════════════════════════════════════════════════════════════════════
# Not a multiprocessing child: a spawn child is prepared from the parent's
# state and re-imports the parent's __main__, which muddies any claim about
# what a clean interpreter pulls in. This is a bare interpreter that imports
# only what it is told to on argv, then calls the PRODUCTION guard.
_CHILD_SOURCE = '''\
import importlib
import os
import sys

def emit(key, value):
    print(f"{key}={value}", flush=True)

surface = sys.argv[1:]
emit("CHILD_PID", os.getpid())
emit("CHILD_EXE", sys.executable)
emit("SURFACE_REQUESTED", ",".join(surface))
if not surface:
    emit("SENTINEL", "INCOMPLETE")
    sys.exit(6)

# The guard and its authority come from production. The forbidden list is
# RESOLVED here and echoed, so the parent can prove the child read the real
# tuple rather than a stale copy of it.
try:
    worker = importlib.import_module("miner.assembly_shard_worker")
except BaseException as exc:
    emit("GUARD", f"UNIMPORTABLE {type(exc).__name__}: {exc}")
    emit("SENTINEL", "INCOMPLETE")
    sys.exit(4)
emit("FORBIDDEN_RESOLVED", ",".join(worker._FORBIDDEN_GPU_MODULES))
emit("WORKER_FILE", getattr(worker, "__file__", "?"))

# Import the real Step-1 module surface. A failure here is INCOMPLETE, never
# FAIL: the gate would not have reached its assertion.
for name in surface:
    try:
        mod = importlib.import_module(name)
    except BaseException as exc:
        emit("IMPORT_ERROR", f"{name} {type(exc).__name__}: {exc}")
        emit("SENTINEL", "INCOMPLETE")
        sys.exit(5)
    emit("IMPORTED", f"{name} {getattr(mod, '__file__', '?')}")

loaded = sorted(m for m in sys.modules if m.split(".")[0] == "miner")
emit("MINER_MODULES", ",".join(loaded))

# Contamination proof: if anything imported resolves a checkpoint root, report
# where it actually points, so the redirect is shown to have TAKEN EFFECT
# rather than merely having been requested. Discovered by attribute, so no
# module name is baked in here.
for _name in surface:
    _resolver = getattr(sys.modules.get(_name), "_flush_checkpoint_root", None)
    if callable(_resolver):
        try:
            emit("CHECKPOINT_ROOT", f"{_name} {_resolver()}")
        except BaseException as exc:
            emit("CHECKPOINT_ROOT", f"{_name} UNRESOLVED {type(exc).__name__}")
present = [m for m in worker._FORBIDDEN_GPU_MODULES if m in sys.modules]
emit("PRESENT", ",".join(present))
for m in present:
    src = getattr(sys.modules[m], "__file__", "?")
    emit("PRESENT_FILE", f"{m} {src}")

# The assertion itself: the PRODUCTION guard is INVOKED, not restated.
emit("REACHED_ASSERTION", "1")
try:
    worker.assert_cpu_only()
except BaseException as exc:
    emit("GUARD", "FAIL")
    emit("GUARD_TYPE", type(exc).__name__)
    emit("GUARD_MRO", ",".join(c.__name__ for c in type(exc).__mro__))
    emit("GUARD_MESSAGE", str(exc).replace(chr(10), " "))
    emit("SENTINEL", "FAIL")
    sys.exit(3)
emit("GUARD", "PASS")
emit("SENTINEL", "PASS")
sys.exit(0)
'''


class ChildResult:
    def __init__(self, proc, fields, raw):
        self.returncode = proc.returncode
        self.fields: Dict[str, str] = fields
        self.multi: Dict[str, List[str]] = {}
        self.raw = raw

    @property
    def sentinel(self) -> str:
        return self.fields.get("SENTINEL", "INCOMPLETE")

    def __str__(self):
        return f"exit={self.returncode}\n{self.raw}"


def _runner_dir() -> str:
    """The child script lives in its OWN directory, never inside a candidate
    import tree — so `sys.path[0]` (the script's directory) can never shadow
    the tree under test and resolution is decided purely by PYTHONPATH."""
    global _RUNNER_DIR
    if _RUNNER_DIR is None:
        _RUNNER_DIR = tempfile.mkdtemp(prefix="s172_import_gate_runner_")
        path = os.path.join(_RUNNER_DIR, "cpu_only_child.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_CHILD_SOURCE)
    return _RUNNER_DIR


_RUNNER_DIR = None
_SANDBOX_ROOT = None
_TMPDIRS: List[str] = []


def _sandbox_root() -> str:
    """One throwaway checkpoint root shared by every child of this run, so an
    escaped write is still contained AND still observable afterwards."""
    global _SANDBOX_ROOT
    if _SANDBOX_ROOT is None:
        _SANDBOX_ROOT = tempfile.mkdtemp(prefix="s172_import_gate_ckpt_")
        _TMPDIRS.append(_SANDBOX_ROOT)
    return _SANDBOX_ROOT


def _run_child(surface: List[str], tree_first: str = None) -> ChildResult:
    """One FRESH interpreter via `sys.executable`. `tree_first`, when given, is
    prepended to PYTHONPATH so a mutated copy of the miner package wins over
    the live tree — the live file is never touched."""
    script = os.path.join(_runner_dir(), "cpu_only_child.py")
    pythonpath = _ROOT if tree_first is None else os.pathsep.join(
        (tree_first, _ROOT))
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # CONTAMINATION GUARD. Step-1's host module fixes a run identity at IMPORT
    # time from (hostname, pid, epoch), and every fresh interpreter here is a
    # new pid. A stray flush would therefore scatter run-id directories into
    # the real checkpoint root, where they would be indistinguishable from
    # production ones and would corrupt a retention census. Every child writes
    # into a throwaway root instead; G-NO-CONTAMINATION proves none escaped.
    env[_CHECKPOINT_ROOT_ENV] = _sandbox_root()
    try:
        proc = subprocess.run([sys.executable, script, *surface],
                              cwd=_ROOT, env=env, capture_output=True,
                              text=True, timeout=_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"the fresh interpreter did not finish in {_TIMEOUT}s — a gate "
            f"that times out is INCOMPLETE, never PASS: {exc}")
    fields: Dict[str, str] = {}
    multi: Dict[str, List[str]] = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        fields.setdefault(key, value)
        multi.setdefault(key, []).append(value)
    raw = f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-2000:]}"
    result = ChildResult(proc, fields, raw)
    result.multi = multi
    return result


def _assert_executed(result: ChildResult, surface: List[str]) -> None:
    """VIR-1 execution proof, applied to EVERY child: it really started, it
    really read production's tuple, and it really imported what it was told.
    A child that silently did nothing cannot be mistaken for one that passed."""
    assert result.sentinel in _SENTINELS, (
        f"child printed no recognisable sentinel — INCOMPLETE\n{result}")
    assert result.sentinel != "UNAVAILABLE", (
        f"this gate has no fleet dependency; UNAVAILABLE is a finding\n{result}")
    assert result.fields.get("CHILD_PID"), f"child never started\n{result}"
    assert int(result.fields["CHILD_PID"]) != os.getpid(), (
        f"the 'fresh interpreter' ran in the parent process\n{result}")
    assert result.fields.get("SURFACE_REQUESTED") == ",".join(surface), (
        f"child was asked for a different surface\n{result}")
    resolved = result.fields.get("FORBIDDEN_RESOLVED", "")
    assert resolved == ",".join(_FORBIDDEN), (
        f"the child resolved a forbidden tuple that differs from production's "
        f"{_FORBIDDEN} — the gate and the authority have drifted\n{result}")


# ═════════════════════════════════════════════════════════════════════════════
# G-SURFACE — the surface is derived from Step-1's source, and is not empty
# ═════════════════════════════════════════════════════════════════════════════
def g_surface():
    surface = _step1_surface()
    assert surface, (
        "no module-scope miner import was found in Step-1's host module — the "
        "gate would be vacuous, importing nothing and asserting on nothing")
    # every derived name must be a real, importable module of the package, or
    # the derivation has silently picked up prose
    for name in surface:
        assert name == _MINER_PKG or name.startswith(_MINER_PKG + "."), name
        rel = os.path.join(_ROOT, *name.split("."))
        assert os.path.isdir(rel) or os.path.isfile(rel + ".py"), (
            f"derived surface member {name!r} has no module on disk")
    # the derivation must reach the package root: that is the edge that pulls
    # the coordinator -> range_miner_worker chain the guard's docstring names
    assert _MINER_PKG in surface, (
        f"Step-1 no longer imports the miner package root at module scope; the "
        f"derived surface {surface} would miss the coordinator chain")
    _EVIDENCE.append(("step1 surface (derived from Step-1 source)",
                      " + ".join(surface)))


# ═════════════════════════════════════════════════════════════════════════════
# G-SURFACE-GUARD [gaps 2.2 + 2.3] — the CLEAN CONTROL, and the whole point:
# a fresh interpreter imports the real Step-1 surface, then INVOKES the
# production guard
# ═════════════════════════════════════════════════════════════════════════════
def g_surface_guard():
    surface = _step1_surface()
    result = _run_child(surface)
    _assert_executed(result, surface)
    assert result.fields.get("REACHED_ASSERTION") == "1", (
        f"the child never reached the guard call — INCOMPLETE\n{result}")
    assert result.sentinel == "PASS", (
        f"importing the real Step-1 module surface {surface} left "
        f"{result.fields.get('PRESENT', '?')!r} in sys.modules and the "
        f"production guard fired: "
        f"{result.fields.get('GUARD_MESSAGE', '')}\n{result}")
    assert result.returncode == 0, f"clean control exited nonzero\n{result}"
    assert result.fields.get("GUARD") == "PASS", result.raw
    assert result.fields.get("PRESENT", "") == "", result.raw

    # ...and it really did pull the transitive graph the docstring claims for
    # it, so 'nothing leaked' is not merely 'nothing was imported'
    loaded = set(result.fields.get("MINER_MODULES", "").split(","))
    missing = [m for m in _EXPECTED_TRANSITIVE if m not in loaded]
    assert not missing, (
        f"the surface did not transitively reach {missing} — the import-graph "
        f"claim is not being exercised end to end\n{result}")
    _EVIDENCE.append(("clean control (unmutated tree)",
                      f"exit=0 sentinel=PASS "
                      f"miner modules loaded={len(loaded)} present=none"))


# ═════════════════════════════════════════════════════════════════════════════
# G-NO-DUP-LIST [gap 2.1] — this gate never restates production's list
# ═════════════════════════════════════════════════════════════════════════════
def g_no_dup_list():
    src = _read(_SELF_PATH)
    lowered = tuple(name.lower() for name in _FORBIDDEN)

    def offending(text: str) -> List[str]:
        low = text.lower()
        return [n for n in lowered if n in low]

    # 1. no string literal anywhere in this file names a forbidden module
    tree = ast.parse(src, filename=_SELF_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            hits = offending(node.value)
            assert not hits, (
                f"{_SELF_PATH}:{node.lineno} restates {hits} as a literal — "
                f"the probe must READ the forbidden list from production, so "
                f"widening it there cannot leave this gate checking fewer")
        # ...nor as an identifier, attribute or import alias
        if isinstance(node, ast.Name):
            assert not offending(node.id), f"{_SELF_PATH}:{node.lineno} {node.id}"
        if isinstance(node, ast.Attribute):
            assert not offending(node.attr), f"{_SELF_PATH}:{node.lineno}"
        if isinstance(node, ast.alias):
            assert not offending(node.name), f"{_SELF_PATH}: import {node.name}"

    # 2. comments are not in the AST, so they are swept separately — a
    #    forbidden name surviving in prose here would be a latent copy of the
    #    authority that the AST pass alone would never see
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            hits = offending(tok.string)
            assert not hits, (
                f"{_SELF_PATH}:{tok.start[0]} comment names {hits}")

    # 3. the child this gate runs is held to the same rule
    hits = offending(_CHILD_SOURCE)
    assert not hits, f"the fresh-interpreter child restates {hits}"

    # 4. the rule is not vacuous: the scanner must actually detect a name
    assert offending(f"import {_FORBIDDEN[0]}") == [lowered[0]], (
        "the duplicate-list scanner does not detect a forbidden name — it "
        "would pass on any file, including one that duplicates the list")
    # 5. and production is still the authority it is read from
    assert _FORBIDDEN and all(isinstance(n, str) for n in _FORBIDDEN), _FORBIDDEN
    assert "_FORBIDDEN_GPU_MODULES" in _read(_WORKER_PATH), (
        "production no longer declares the authority this gate reads")
    _EVIDENCE.append(("forbidden list resolved from production",
                      f"{len(_FORBIDDEN)} modules, read from "
                      f"assembly_shard_worker._FORBIDDEN_GPU_MODULES"))


# ═════════════════════════════════════════════════════════════════════════════
# G-RUNTIME-INJECTION [gap 2.4] — EVERY forbidden module injected in turn
# ═════════════════════════════════════════════════════════════════════════════
def g_runtime_injection():
    assert len(_FORBIDDEN) >= 2, (
        f"production declares {len(_FORBIDDEN)} forbidden module(s); this arm "
        f"exists because covering only the first is what it replaces")
    ASW.assert_cpu_only()          # negative control BEFORE: the process is clean

    covered = []
    for name in _FORBIDDEN:
        sentinel = object()
        injected = name not in sys.modules
        if injected:
            sys.modules[name] = sentinel        # type: ignore[assignment]
        try:
            e = _raises(ASW.ShardArtifactError, ASW.assert_cpu_only)
            assert name in str(e), (
                f"the guard fired but did not NAME the injected module "
                f"{name!r}: {e}")
            # the diagnosis must be usable: it identifies the module AND the
            # process holding it
            assert str(os.getpid()) in str(e), str(e)
        finally:
            # restored in EVERY case — a leaked sentinel would poison every
            # later test in this interpreter, including the negative control
            if injected:
                del sys.modules[name]
        assert name not in sys.modules, (
            f"the injected sentinel for {name!r} leaked out of its arm")
        covered.append(name)

    assert covered == list(_FORBIDDEN), (covered, _FORBIDDEN)
    ASW.assert_cpu_only()          # negative control AFTER: nothing leaked
    _EVIDENCE.append(("runtime injection",
                      f"{len(covered)}/{len(_FORBIDDEN)} forbidden modules "
                      f"injected in turn, guard fired and named each"))


# ═════════════════════════════════════════════════════════════════════════════
# G-MUTANT [§3] — a module-scope GPU import in the Step-1 chain must red the
# fresh-interpreter gate, and red for the RIGHT reason
# ═════════════════════════════════════════════════════════════════════════════
def _mutant_tree(target_rel: str, module_name: str, label: str) -> Tuple[str, str]:
    """Copy the miner package into a temp tree and inject a MODULE-SCOPE import
    of `module_name` into `target_rel`. The live file is never touched: the
    copy wins only because its directory is prepended to the child's
    PYTHONPATH."""
    root = tempfile.mkdtemp(prefix="s172_import_gate_mutant_")
    _TMPDIRS.append(root)
    shutil.copytree(os.path.join(_ROOT, _MINER_PKG),
                    os.path.join(root, _MINER_PKG),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = os.path.join(root, target_rel)
    original = _read(path)

    # part 1 of the four-part rule: the mutation APPLIES, exactly once, and was
    # not already present AT MODULE SCOPE. A function-scope import of the same
    # library is not a pre-existing mutation — it is the very arrangement the
    # guard's docstring blesses, and the defect being injected is the LIFT of
    # such an import to module scope.
    tree = ast.parse(original, filename=path)
    assert module_name not in _module_scope_imports(tree), (
        f"{label}: {target_rel} already imports {module_name} at module scope "
        f"— the mutant would be vacuous")
    injected = f"import {module_name}  # S172 import-gate mutant\n"
    assert original.count(injected) == 0, label
    # Placed after the module docstring AND after any `__future__` import, so
    # it is unambiguously module scope and still legal Python: a `__future__`
    # import must be the first statement, so injecting above one yields a
    # SyntaxError — a mutant that reds for the wrong reason, which the
    # four-part rule must refuse to credit rather than bank as a kill.
    body = ast.parse(original, filename=path).body
    insert_line = 0
    for node in body:
        if isinstance(node, ast.Expr) and isinstance(
                getattr(node, "value", None), ast.Constant):
            insert_line = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_line = node.end_lineno
        else:
            break
    lines = original.splitlines(keepends=True)
    mutated = "".join(lines[:insert_line]) + injected + "".join(lines[insert_line:])
    assert mutated.count(injected) == 1, f"{label}: mutation did not apply once"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(mutated)

    # ...and it really is module scope in the file that will be imported
    mtree = ast.parse(_read(path), filename=path)
    top = [n for n in mtree.body if isinstance(n, ast.Import)
           and module_name in [a.name for a in n.names]]
    assert len(top) == 1, (
        f"{label}: the injected import is not a single module-scope statement")
    return root, path


def _one_mutant(target_rel: str, module_name: str, label: str,
                surface: List[str]) -> None:
    root, path = _mutant_tree(target_rel, module_name, label)
    result = _run_child(surface, tree_first=root)
    _assert_executed(result, surface)

    # part 2: the MUTATED path executed — the child loaded the mutated copy,
    # not the live tree, and the injected import actually ran
    imported = {v.split(" ", 1)[0]: v.split(" ", 1)[1]
                for v in result.multi.get("IMPORTED", []) if " " in v}
    resolved_from_mutant = [n for n, f in imported.items()
                            if f.startswith(root)]
    assert resolved_from_mutant, (
        f"{label}: no surface module resolved from the mutated tree {root} — "
        f"the child imported the live tree and the mutant proves nothing\n"
        f"{result}")
    assert result.fields.get("WORKER_FILE", "").startswith(root), (
        f"{label}: the guard was read from the live tree, not the mutant\n"
        f"{result}")
    present = [p for p in result.fields.get("PRESENT", "").split(",") if p]
    assert present == [module_name], (
        f"{label}: expected exactly the injected module to be present, got "
        f"{present}\n{result}")

    # part 3: the gate REACHED its credited assertion (and the same runner
    # passed clean in g_surface_guard — the positive control)
    assert result.fields.get("REACHED_ASSERTION") == "1", (
        f"{label}: the child died before the guard call — this is INCOMPLETE, "
        f"not a kill\n{result}")

    # part 4: the red is the INJECTED DEFECT, not a loader/timeout/collection
    # failure. The brief is explicit: ShardArtifactError naming the module.
    assert result.sentinel == "FAIL", (
        f"{label}: MUTANT SURVIVED — sentinel {result.sentinel}\n{result}")
    assert result.returncode != 0, f"{label}: MUTANT SURVIVED (exit 0)\n{result}"
    assert result.fields.get("GUARD") == "FAIL", f"{label}\n{result}"
    guard_type = result.fields.get("GUARD_TYPE", "")
    assert guard_type == ASW.ShardArtifactError.__name__, (
        f"{label}: red for the WRONG REASON — {guard_type}, not "
        f"{ASW.ShardArtifactError.__name__}\n{result}")
    mro = result.fields.get("GUARD_MRO", "").split(",")
    assert ASW.ProcessShardedAssemblyError.__name__ in mro, (
        f"{label}: the failure is not a backend failure\n{result}")
    assert "ImportError" not in mro and "ModuleNotFoundError" not in mro, (
        f"{label}: red is an import failure, not the guard\n{result}")
    message = result.fields.get("GUARD_MESSAGE", "")
    assert module_name in message, (
        f"{label}: the guard did not NAME the injected module\n{result}")

    _MUTANTS.append((
        label,
        f"exit={result.returncode} {guard_type}: {message[:110]}",
        "G-SURFACE-GUARD (fresh interpreter, production assert_cpu_only)",
        "applies-once ✓ | mutated-path ✓ | reached-assertion ✓ | "
        "injected-defect ✓"))


def g_mutants():
    surface = _step1_surface()
    worker_rel = os.path.join(_MINER_PKG, "range_miner_worker.py")
    coord_rel = os.path.join(_MINER_PKG, "range_miner_coordinator.py")

    # `range_miner_worker` is the module the guard's docstring names by hand:
    # "imports [the GPU libraries] only INSIDE its kernel functions". Lifting
    # one to module scope is exactly the refactor the gate exists to catch, so
    # it is mutated once per forbidden module.
    for name in _FORBIDDEN:
        _one_mutant(worker_rel, name,
                    f"M-WORKER-{name.upper()}: module-scope GPU import in "
                    f"miner/range_miner_worker.py", surface)

    # ...and once a link DEEPER in the chain, so the gate is proven to cover
    # the graph rather than one file: the coordinator is the hop between the
    # D1.1 engine and the worker.
    _one_mutant(coord_rel, _FORBIDDEN[0],
                f"M-COORD-{_FORBIDDEN[0].upper()}: module-scope GPU import in "
                f"miner/range_miner_coordinator.py", surface)

    # the live tree is untouched by all of the above — same runner, still clean
    control = _run_child(surface)
    _assert_executed(control, surface)
    assert control.sentinel == "PASS", (
        f"the live tree no longer passes after mutation — a mutant escaped its "
        f"temp copy\n{control}")
    _EVIDENCE.append(("post-mutation control", "live tree still exit=0 PASS"))


# ═════════════════════════════════════════════════════════════════════════════
# G-HOST-BOUNDARY — why the Step-1 HOST module is not the surface, measured
# ═════════════════════════════════════════════════════════════════════════════
def g_host_boundary():
    """Step-1's host module holds a GPU library at import time. That is not a
    defect — Step-1 IS the sieve host — but it is the reason the surface is the
    miner subgraph rather than the host itself. This arm measures it instead of
    assuming it, and reds if the miner package ever becomes the culprit."""
    host = os.path.splitext(os.path.basename(_STEP1_HOST_PATH))[0]
    result = _run_child([host])
    assert result.sentinel in _SENTINELS, f"no sentinel\n{result}"
    assert result.fields.get("REACHED_ASSERTION") == "1", (
        f"the host module could not be imported at all\n{result}")
    present = [p for p in result.fields.get("PRESENT", "").split(",") if p]
    files = {v.split(" ", 1)[0]: v.split(" ", 1)[1]
             for v in result.multi.get("PRESENT_FILE", []) if " " in v}

    # whatever the host pulls in, it must NOT be the miner package that pulls
    # it: that is the whole distinction the surface choice rests on
    for name in present:
        origin = files.get(name, "?")
        assert os.path.join(_ROOT, _MINER_PKG) not in origin, (
            f"a GPU library reached the interpreter FROM the miner package "
            f"({name} <- {origin}) — the Step-1 surface is no longer CPU-only "
            f"and G-SURFACE-GUARD's clean control is the real finding")

    # the host is the ONE surface that resolves a checkpoint root, and it must
    # have resolved it INSIDE the sandbox — proof the redirect took effect in
    # the very child that could otherwise have written into the repo
    roots = [v.split(" ", 1)[1] for v in result.multi.get("CHECKPOINT_ROOT", [])
             if " " in v]
    assert roots, (
        f"the host module exposed no checkpoint-root resolver, so the "
        f"contamination redirect could not be proven to take effect\n{result}")
    for root in roots:
        assert os.path.abspath(root) == os.path.abspath(_sandbox_root()), (
            f"a fresh interpreter resolved its checkpoint root to {root!r}, "
            f"OUTSIDE the sandbox — a flush would have scattered a run-id "
            f"directory into the real tree\n{result}")

    _EVIDENCE.append((
        "step1 host boundary",
        f"{host} sentinel={result.sentinel}; "
        f"{len(present)} forbidden module(s) present, none from miner/ "
        f"(host holds a GPU context legitimately; the miner surface does not)"))
    _EVIDENCE.append(("checkpoint redirect honoured in-child",
                      f"resolved root == sandbox for {len(roots)} resolver(s)"))


# ═════════════════════════════════════════════════════════════════════════════
# G-NO-CONTAMINATION — this gate scattered nothing into the real tree
# ═════════════════════════════════════════════════════════════════════════════
def g_no_contamination():
    """Every fresh interpreter is a new pid, and Step-1 fixes a run identity
    from (hostname, pid, epoch) at import time. Had a flush run unredirected,
    each child would have left its own run-id directory in the real checkpoint
    root, indistinguishable from a production one and corrupting the retention
    census that runs against that directory."""
    after = _checkpoint_census()
    created = sorted(set(after) - set(_CENSUS_BEFORE))
    removed = sorted(set(_CENSUS_BEFORE) - set(after))
    assert not created, (
        f"this gate created {len(created)} directory(ies) under "
        f"{_REAL_CHECKPOINT_ROOT}: {created} — they would be "
        f"indistinguishable from production run directories")
    assert not removed, (
        f"this gate REMOVED {removed} from {_REAL_CHECKPOINT_ROOT}; it must "
        f"neither add to nor prune the real tree")
    assert len(after) == len(_CENSUS_BEFORE), (len(_CENSUS_BEFORE), len(after))

    # the guard is not decorative: the sandbox is where writes were pointed,
    # and it is a real directory outside the repo
    sandbox = _sandbox_root()
    assert os.path.isdir(sandbox), sandbox
    assert not os.path.abspath(sandbox).startswith(os.path.abspath(_ROOT)), (
        f"the sandbox {sandbox} is INSIDE the repo tree")
    _EVIDENCE.append((
        "checkpoint census (real root)",
        f"before={len(_CENSUS_BEFORE)} after={len(after)} created=0 "
        f"root={_REAL_CHECKPOINT_ROOT}"))


# ═════════════════════════════════════════════════════════════════════════════
def main():
    global _CENSUS_BEFORE
    # taken BEFORE any child runs, so the census brackets the entire gate
    _CENSUS_BEFORE = _checkpoint_census()
    print("=" * 78)
    print("S172 — `process_sharded` CPU-only IMPORT gate (Beta hardening)")
    print("=" * 78)
    print(f"  checkpoint census before: {len(_CENSUS_BEFORE)} directory(ies) "
          f"under {_REAL_CHECKPOINT_ROOT}")

    _check("G-SURFACE: the Step-1 module surface is DERIVED from Step-1's "
           "source, not named here, and is not empty",
           g_surface)
    _check("G-SURFACE-GUARD [2.2+2.3]: a FRESH interpreter imports the real "
           "Step-1 surface and the PRODUCTION assert_cpu_only() passes",
           g_surface_guard)
    _check("G-NO-DUP-LIST [2.1]: no GPU-module-name literal anywhere in this "
           "gate — the forbidden list is read from production",
           g_no_dup_list)
    _check("G-RUNTIME-INJECTION [2.4]: every forbidden module injected in "
           "turn; the guard fires and names each",
           g_runtime_injection)
    _check("G-MUTANTS [§3]: module-scope GPU imports in the Step-1 chain red "
           "the gate, four-part rule, with clean control",
           g_mutants)
    _check("G-HOST-BOUNDARY: the Step-1 host's own GPU context is measured, "
           "and provably does not come from the miner package",
           g_host_boundary)
    _check("G-NO-CONTAMINATION: no run-id directory was scattered into the "
           "real checkpoint root by any fresh interpreter",
           g_no_contamination)
    print("=" * 78)

    if _MUTANTS:
        print("\nMUTATION EVIDENCE — every mutant RED, with attribution:\n")
        for label, signature, credited, fourpart in _MUTANTS:
            print(f"  {label}")
            print(f"      red in   : {credited}")
            print(f"      four-part: {fourpart}")
            print(f"      signature: {signature}")
        print()

    if _EVIDENCE:
        print("EXECUTION EVIDENCE (VIR-1):\n")
        for key, value in _EVIDENCE:
            print(f"  {key:<46} {value}")
        print()

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} import-gate checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All import-gate checks green — a fresh interpreter that imports the "
          "real Step-1 module surface holds no GPU library, the production "
          "guard is INVOKED rather than restated, every forbidden module is "
          "covered at runtime, and a module-scope GPU import anywhere in the "
          "chain reds this gate (pending Team Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        for _d in [_RUNNER_DIR, *_TMPDIRS]:
            if _d:
                shutil.rmtree(_d, ignore_errors=True)
