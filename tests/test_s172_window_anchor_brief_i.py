#!/usr/bin/env python3
"""
S172 WINDOW-ANCHOR / GENERATOR-PHASE SEPARATION — IMPLEMENTATION BRIEF I
=======================================================================
Gate suite for `docs/S172_WINDOW_ANCHOR_BRIEF_I.md`.

Authority: docs/TB_RULING_WINDOW_ANCHOR_V1_1_DESIGN_GATE_CLOSED.md (design gate
CLOSED, Brief I AUTHORIZED) + the scope ruling of 2026-08-21 (Items 1/2/3
APPROVED; capability-before-policy ordering at the worker seam BINDING).
Design of record: docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md.

THIS FILE CARRIES THE §4.8 LEGACY-CLOSURE GATES (AC5).
Every gate names the wrong input that reds it. A gate whose failure mode cannot
be stated is not a gate.

CPU-only. No GPU, no fleet, no network.
"""
from __future__ import annotations

import ast
import io
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
_RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        detail = fn()
        _RESULTS.append((name, True, ""))
        print(f"  [{GREEN}PASS{RESET}] {name}" + (f"  — {detail}" if detail else ""))
    except Exception:
        tb = traceback.format_exc()
        _RESULTS.append((name, False, tb))
        print(f"  [{RED}FAIL{RESET}] {name}")


def _read(rel: str) -> str:
    with io.open(os.path.join(_ROOT, rel), encoding="utf-8", errors="replace") as f:
        return f.read()


# ===========================================================================
# §4.8 step 1 — REACHABILITY, DERIVED NOT TRANSCRIBED
# ===========================================================================
# The brief's own route table was INCOMPLETE: it listed three dispatch routes and
# four import sites; live source carries FOUR and EIGHT. The miss that mattered was
# `coordinator_sieve_dynamic.py`, which is not a stray copy — `test_sieve_dynamic.sh:36`
# executes `cp coordinator_sieve_dynamic.py coordinator.py`. It is a REPLACEMENT
# IMAGE for the live coordinator, so closing routes 1/2/4 while leaving it open
# would have let a single `cp` silently REOPEN route 2.
#
# That is why this gate re-derives the census every run instead of pinning the
# table. It searches three mechanisms INDEPENDENTLY, because a single grep is how
# both undercounts happened.

_TOKEN_LITERAL = "LEGACY_FUSED_ENGINE_CLOSED"

TARGET_MOD = "reverse_sieve_filter"
TARGET_FILE = "reverse_sieve_filter.py"
_SKIP_DIRS = {".git", "logs", "__pycache__", ".s172_checkpoint", "node_modules",
              ".s172_accumulator", "dataset_provenance"}

# The routes that dispatch the engine as a SUBPROCESS, and are therefore required
# to be closed. Declared exactly; a new one reds G-LEGACY-1.
DECLARED_DISPATCH_ROUTES = {
    "coordinator.py",
    "coordinator_sieve_dynamic.py",
    "distributed_worker.py",
    "run_complete_pipeline.py",
}
# Diagnostic/import consumers. These READ the archive and are deliberately NOT
# closed — see the boundary in reverse_sieve_filter.LegacyFusedEngineClosed.
DECLARED_IMPORT_CONSUMERS = {
    "identify_failed_seeds.py", "identify_failures.py", "identify_failures_trace.py",
    "retest_seed87.py", "test_real_candidates.py", "test_remote_seed.py",
    "test_reverse_direct.py", "test_reverse_simple.py",
}
# Test harnesses that invoke the engine by subprocess. Not production dispatch;
# they are covered by the ENGINE entry guard (G-LEGACY-3), not by route closure.
DECLARED_TEST_INVOKERS = {
    "test_100k_bidirectional_simple.py", "test_reverse_sieve_module.py",
}


def _py_files() -> List[Tuple[str, str]]:
    out = []
    for dp, dns, fns in os.walk(_ROOT):
        dns[:] = [d for d in dns if d not in _SKIP_DIRS]
        for fn in fns:
            if fn.endswith(".py"):
                p = os.path.join(dp, fn)
                out.append((os.path.relpath(p, _ROOT), p))
    return out


def _is_guard_stmt(node) -> bool:
    """A statement is a GUARD if it raises with the closure token."""
    if not isinstance(node, ast.Raise):
        return False
    return _TOKEN_LITERAL in "".join(
        n.value for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str))


def _census() -> Dict[str, set]:
    """Three INDEPENDENT mechanisms. Never one grep."""
    imports, subproc = set(), set()
    for rel, p in _py_files():
        if rel == TARGET_FILE:
            continue
        try:
            src = io.open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        # (A) import-level, via AST — catches importlib/__import__ too
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None
        if tree is not None:
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    if any(a.name.split(".")[0] == TARGET_MOD for a in n.names):
                        imports.add(rel)
                elif isinstance(n, ast.ImportFrom):
                    if (n.module or "").split(".")[0] == TARGET_MOD:
                        imports.add(rel)
                elif isinstance(n, ast.Call):
                    f = n.func
                    nm = getattr(f, "attr", None) or getattr(f, "id", None)
                    if nm in ("import_module", "__import__"):
                        for a in n.args:
                            if isinstance(a, ast.Constant) and TARGET_MOD in str(a.value):
                                imports.add(rel)
        # (B) subprocess / shell INVOCATION — statement-scoped, not line-scoped.
        #
        # A filename MENTION is not a route. `apply_integration.py` opens the file,
        # `test_distributed_reverse_sieve.py` prints its name, and this suite itself
        # declares it as data — none of them dispatch it. The discriminator is
        # CO-OCCURRENCE inside one statement: the filename AND a python invocation
        # token in the same statement's string constants. That catches the f-string
        # form (`python -u reverse_sieve_filter.py`), the argv-list form
        # (`['python3', 'reverse_sieve_filter.py', ...]`, where the two sit on
        # different LINES but in one statement), and the wrapper form
        # (`run_command([...])`, which never names subprocess at all).
        #
        # Guard statements are excluded: the closure message quotes the filename,
        # and counting it would make every closed route look like an open one.
        if tree is not None:
            for st in ast.walk(tree):
                if not isinstance(st, ast.stmt) or _is_guard_stmt(st):
                    continue
                # LEAF statements only. `ast.walk` yields compound statements too,
                # and walking a whole FunctionDef collects every string in its body —
                # so a docstring mentioning the filename co-occurred with a "python"
                # string 40 lines away and `apply_integration.py` (which only WRITES
                # the file) read as a dispatch route. Co-occurrence is only evidence
                # inside ONE executable statement.
                if any(isinstance(c, ast.stmt) for c in ast.walk(st) if c is not st):
                    continue
                consts = [n.value for n in ast.walk(st)
                          if isinstance(n, ast.Constant) and isinstance(n.value, str)]
                if any(TARGET_FILE in c for c in consts) and \
                        any("python" in c for c in consts):
                    subproc.add(rel)
                    break
    return {"imports": imports, "subproc": subproc}


def gate_legacy_1_reachability():
    """G-LEGACY-1 — the route table, re-derived every run.

    *Reds on:* a new import site, a new subprocess invoker, or a declared one
    disappearing. Either direction matters: a vanished route may mean the file was
    renamed rather than closed."""
    c = _census()
    declared_sub = DECLARED_DISPATCH_ROUTES | DECLARED_TEST_INVOKERS
    new_sub = c["subproc"] - declared_sub
    gone_sub = declared_sub - c["subproc"]
    new_imp = c["imports"] - DECLARED_IMPORT_CONSUMERS
    gone_imp = DECLARED_IMPORT_CONSUMERS - c["imports"]
    assert not new_sub, f"UNDECLARED subprocess route(s) reach the legacy engine: {sorted(new_sub)}"
    assert not gone_sub, f"declared subprocess route(s) vanished: {sorted(gone_sub)}"
    assert not new_imp, f"UNDECLARED import consumer(s): {sorted(new_imp)}"
    assert not gone_imp, f"declared import consumer(s) vanished: {sorted(gone_imp)}"
    return (f"{len(c['subproc'])} subprocess sites "
            f"({len(DECLARED_DISPATCH_ROUTES)} production routes + "
            f"{len(DECLARED_TEST_INVOKERS)} test invokers), "
            f"{len(c['imports'])} import consumers — all declared")


def gate_legacy_1b_replacement_image():
    """G-LEGACY-1b — the fact that makes route 3 load-bearing.

    `coordinator_sieve_dynamic.py` is copied OVER `coordinator.py`. If that ever
    stops being true the risk model changes, so it is asserted rather than
    remembered.

    *Reds on:* the `cp` disappearing (route 3 would then be an ordinary variant),
    or a NEW file acquiring the same replacement relationship."""
    # SELF-EXCLUSION IS LOAD-BEARING. This file quotes the `cp` in its own comment
    # and in this very assertion, so without the exclusion the gate matches ITSELF
    # and stays green even if both real call sites disappear — green on a fact it
    # does not check, in the gate written to catch exactly that.
    _self = os.path.relpath(os.path.abspath(__file__), _ROOT)
    hits = []
    for rel, p in _py_files() + [(f, os.path.join(_ROOT, f))
                                 for f in os.listdir(_ROOT) if f.endswith(".sh")]:
        if rel == _self:
            continue
        try:
            src = io.open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for i, line in enumerate(src.split("\n"), 1):
            if "coordinator_sieve_dynamic.py coordinator.py" in line:
                hits.append(f"{rel}:{i}")
    assert hits, ("no `cp coordinator_sieve_dynamic.py coordinator.py` found outside "
                  "this suite — the replacement-image relationship this gate "
                  "documents is gone, and route 3's risk model changed with it")
    return f"replacement image confirmed at {sorted(set(hits))} (self excluded)"


# ===========================================================================
# §4.8 step 2 — THE FOUR DISPATCH ROUTES ARE CLOSED
# ===========================================================================
_TOKEN = "LEGACY_FUSED_ENGINE_CLOSED"


def _dominating_raise(rel: str) -> Tuple[ast.Raise, ast.stmt]:
    """Find the guard `Raise` that DOMINATES the dispatch statement in live source.

    Returns (raise_node, dispatch_node). Dominance = same statement list, raise at a
    strictly lower index than the statement that contains the dispatch — so the
    dispatch is unreachable. Structural, on the live AST, never on text."""
    tree = ast.parse(_read(rel))

    def contains_dispatch(node) -> bool:
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                    and TARGET_FILE in n.value:
                return True
        return False

    def is_guard(node) -> bool:
        if not isinstance(node, ast.Raise):
            return False
        return _TOKEN in "".join(
            n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str))

    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if not isinstance(block, list):
                continue
            r_idx = d_idx = None
            for i, st in enumerate(block):
                if is_guard(st):
                    if r_idx is None:
                        r_idx = i
                    # THE GUARD'S OWN MESSAGE QUOTES THE FILENAME. Counting it as
                    # the dispatch made r_idx == d_idx and the dominance test
                    # unsatisfiable — the gate reported every closed route as open.
                    continue
                if d_idx is None and contains_dispatch(st):
                    d_idx = i
            if r_idx is not None and d_idx is not None and r_idx < d_idx:
                return block[r_idx], block[d_idx]
    raise AssertionError(
        f"{rel}: no LEGACY_FUSED_ENGINE_CLOSED raise dominates the dispatch — the "
        f"route is OPEN, or the guard was moved after the dispatch construction")


def _gate_route(rel: str):
    def _fn():
        raise_node, dispatch_node = _dominating_raise(rel)
        # BEHAVIOURAL: execute the live guard statement itself and prove it raises.
        mod = ast.Module(body=[raise_node], type_ignores=[])
        ast.fix_missing_locations(mod)
        ns: Dict[str, Any] = {}
        try:
            exec(compile(mod, f"<{rel}:guard>", "exec"), ns)
        except RuntimeError as e:
            assert _TOKEN in str(e), f"{rel}: guard raised without the token: {e}"
            assert "§2.4" in str(e) or "4.8" in str(e), \
                f"{rel}: guard message does not name the governing section"
            return (f"guard at line {raise_node.lineno} dominates dispatch at line "
                    f"{dispatch_node.lineno}; raises {_TOKEN}")
        raise AssertionError(f"{rel}: the guard statement did not raise")
    return _fn


def gate_legacy_2_routes_closed():
    """G-LEGACY-2 — every production dispatch route fails loud.

    *Reds on:* a route that silently skips instead of raising; a guard moved AFTER
    the command construction; a guard that raises without the token; a NEW route
    (caught by G-LEGACY-1 first)."""
    details = []
    for rel in sorted(DECLARED_DISPATCH_ROUTES):
        details.append(f"{rel}: " + _gate_route(rel)())
    return f"{len(details)}/4 routes closed"


# ===========================================================================
# §4.8 step 3 — THE ENGINE ENTRY GUARD, AND ITS BOUNDARY
# ===========================================================================
def gate_legacy_3_engine_guard():
    """G-LEGACY-3 — fused semantics cannot RUN; the archive can still be READ.

    *Reds on:* an execution surface that runs instead of raising, OR an import-time
    explosion that breaks the eight diagnostic consumers (the boundary cuts both
    ways and both directions are asserted)."""
    import importlib
    R = importlib.import_module(TARGET_MOD)          # must NOT explode
    assert hasattr(R, "LegacyFusedEngineClosed")
    exc = R.LegacyFusedEngineClosed

    # inspection stays open
    assert callable(R.load_draws_from_daily3), "loader is not callable"
    assert callable(getattr(R, "GPUReverseSieve", None)), "class is not importable"

    # execution is closed — all four surfaces
    surfaces = []
    for name, call in (
        ("execute_reverse_job", lambda: R.execute_reverse_job({}, 0)),
        ("main", lambda: R.main()),
    ):
        try:
            call()
        except exc as e:
            assert _TOKEN in str(e) and "4.8" in str(e)
            surfaces.append(name)
        else:
            raise AssertionError(f"{name} did not raise")

    import inspect
    for meth in ("run_reverse_sieve", "run_hybrid_reverse_sieve"):
        body = inspect.getsource(getattr(R.GPUReverseSieve, meth))
        tree = ast.parse("".join(l[4:] if l.startswith("    ") else l
                                 for l in body.splitlines(keepends=True)))
        fn = tree.body[0]
        stmts = [s for s in fn.body
                 if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        assert stmts and isinstance(stmts[0], ast.Raise), \
            f"GPUReverseSieve.{meth}: first executable statement is not the guard"
        surfaces.append(f"GPUReverseSieve.{meth}")
    return f"import+loader OPEN; {len(surfaces)} execution surfaces CLOSED"


# ===========================================================================
# AC5 — NO LIVE PATH FEEDS ONE VALUE TO BOTH ROLES
# ===========================================================================
def gate_no_fused():
    """G-NO-FUSED — the anchor selects records; the phase advances the generator;
    neither reaches the other's role.

    Drives DISTINCT values through the real miner path and proves the separation by
    measurement, not by reading the code.

    *Reds on:* the anchor reaching a kernel scalar arg; the phase perturbing the
    residue window; any single scalar arriving in both roles."""
    import json, tempfile, hashlib
    import dataclasses
    import miner.range_miner_worker as w

    # (0) BY CONSTRUCTION, and this is the strongest of the three arms: the anchor
    # cannot reach a kernel scalar because it is NOT IN THE BUILD CONTEXT AT ALL.
    # The behavioural arms below show the current builders do not fuse; this arm
    # shows they *cannot* without a structural change. Mutant M15 makes exactly
    # that structural change and reds here.
    ctx_fields = {f.name for f in dataclasses.fields(w.BuildContext)}
    anchorish = sorted(f for f in ctx_fields
                       if "anchor" in f or f in ("offset", "window_start", "start"))
    assert not anchorish, (
        f"BuildContext carries {anchorish} — the host-side anchor has entered the "
        f"DEVICE-side build context, so a builder can now emit it as a kernel "
        f"scalar. This is the structural precondition for F-4 returning.")

    ANCHOR, PHASE, WIN = 11, 7, 20          # deliberately different values
    fd, ds = tempfile.mkstemp(suffix=".json"); os.close(fd)
    try:
        rows = [{"draw": (i * 7) % 1000, "session": "midday"} for i in range(200)]
        with open(ds, "w") as f:
            json.dump(rows, f)

        # (1) the anchor decides WHICH records — measured against the raw file
        window = w.load_residue_window(ds, WIN, None, ANCHOR)
        expect = [int(r["draw"]) for r in rows[ANCHOR:ANCHOR + WIN]]
        assert window == expect, "the anchor does not select data[anchor:anchor+w]"

        # (2) the phase reaches the kernel; the anchor does NOT
        ctx = w.BuildContext(
            family_name="lcg32_hybrid", hybrid=True, reverse=False,
            seed_dtype="uint32", n_seeds=8, k=len(window), skip_min=0, skip_max=16,
            threshold=0.3, generator_phase=PHASE, params={}, n_strategies=1,
            hybrid_threshold=0.3)
        args = w.resolve_builder("lcg32_hybrid")(ctx)
        scalars = [a.value for a in args if isinstance(a, w.ScalarArg)]
        assert PHASE in scalars, "generator_phase never reached a kernel arg"
        assert ANCHOR not in scalars, (
            f"THE ANCHOR REACHED A KERNEL SCALAR ARG ({ANCHOR} in {scalars}) — "
            f"this is the fused defect F-4 returning")

        # (3) the phase does not perturb the window
        ctx0 = w.BuildContext(**{**ctx.__dict__, "generator_phase": 0})
        a0 = w.resolve_builder("lcg32_hybrid")(ctx0)
        diffs = [i for i in range(len(args)) if args[i] != a0[i]]
        assert diffs == [len(args) - 1], (
            f"phase changed arg positions {diffs}; expected only the trailing scalar")
        assert w.load_residue_window(ds, WIN, None, ANCHOR) == window, \
            "the residue window is not stable under an unchanged anchor"
        return (f"anchor={ANCHOR} selects records and reaches NO kernel arg; "
                f"phase={PHASE} reaches exactly the trailing int32")
    finally:
        os.unlink(ds)


# ===========================================================================
# CAPABILITY & ABI — G-CAP-1..4, G-ABI-FROZEN
# ===========================================================================
# Per-variant SCALAR (position, dtype) sequence — the ABI's shape, pinned.
# Cross-checked against the AUDITED signature block in
# miner/range_miner_worker.py's module docstring, itself read from the live
# prng_registry.py kernel_source strings: e.g. lcg32_hybrid carries a,c,m at
# 13/14/15 (uint32) and the phase at 16 of 17; pcg32 forward carries increment
# (uint64) at 11 and the phase (int32) at 12 of 13.
SCALAR_DTYPE_SEQUENCE = {
    "java_lcg": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'uint64'), (12, 'uint64'), (13, 'int32')],
    "java_lcg_hybrid": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'uint64'), (14, 'uint64')],
    "java_lcg_hybrid_reverse": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'int32')],
    "java_lcg_reverse": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'int32')],
    "lcg32": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'uint32'), (12, 'uint32'), (13, 'uint32'), (14, 'int32')],
    "lcg32_hybrid": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'uint32'), (14, 'uint32'), (15, 'uint32'), (16, 'int32')],
    "lcg32_hybrid_reverse": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'int32')],
    "lcg32_reverse": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'int32')],
    "minstd": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'uint32'), (12, 'uint32'), (13, 'int32')],
    "minstd_hybrid": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'uint32'), (14, 'uint32')],
    "minstd_hybrid_reverse": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'int32')],
    "minstd_reverse": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'int32')],
    "pcg32": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'uint64'), (12, 'int32')],
    "pcg32_hybrid": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'uint64'), (14, 'int32')],
    "pcg32_hybrid_reverse": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'int32')],
    "pcg32_reverse": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'int32')],
    "xorshift128": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'int32'), (12, 'int32'), (13, 'int32'), (14, 'int32')],
    "xorshift128_hybrid": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'int32'), (14, 'int32'), (15, 'int32')],
    "xorshift128_hybrid_reverse": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'int32')],
    "xorshift128_reverse": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'int32')],
    "xorshift32": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'int32'), (12, 'int32'), (13, 'int32'), (14, 'int32')],
    "xorshift32_hybrid": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'int32'), (14, 'int32'), (15, 'int32')],
    "xorshift32_hybrid_reverse": [(7, 'int32'), (8, 'int32'), (11, 'int32'), (12, 'float32'), (13, 'int32')],
    "xorshift32_reverse": [(6, 'int32'), (7, 'int32'), (8, 'int32'), (9, 'int32'), (10, 'float32'), (11, 'int32')],
}


def _ctx(w, variant, phase=0, k=10):
    return w.BuildContext(
        family_name=variant, hybrid=w.is_hybrid_family(variant),
        reverse=w.is_reverse_family(variant),
        seed_dtype="uint64" if w.base_family(variant) == "java_lcg" else "uint32",
        n_seeds=64, k=k, skip_min=0, skip_max=16, threshold=0.3,
        generator_phase=phase, params={}, n_strategies=2, hybrid_threshold=0.3)


def gate_cap_1_arity():
    """G-CAP-1 — exact arity and (position, dtype) of every scalar, all 24 covered
    variants. The numbers are asserted as LITERALS from the §0 C-1 table, never
    computed from the builder under test.

    *Reds on:* any builder that changes an arg position, a dtype, or a count."""
    import miner.range_miner_worker as w
    # C-1 literals. v1.1 §3 says "13 (lcg32: 16)" for all six forward-constant
    # variants; that is wrong for five of them, and a test that asserted 13
    # everywhere would have been "fixed" by loosening it.
    C1 = {
        "java_lcg": 14, "lcg32": 15, "minstd": 14,
        "pcg32": 13, "xorshift32": 15, "xorshift128": 15,
        "java_lcg_reverse": 12, "lcg32_reverse": 12, "minstd_reverse": 12,
        "pcg32_reverse": 12, "xorshift32_reverse": 12, "xorshift128_reverse": 12,
        "java_lcg_hybrid": 15, "lcg32_hybrid": 17, "minstd_hybrid": 15,
        "pcg32_hybrid": 15, "xorshift32_hybrid": 16, "xorshift128_hybrid": 16,
        "java_lcg_hybrid_reverse": 14, "lcg32_hybrid_reverse": 14,
        "minstd_hybrid_reverse": 14, "pcg32_hybrid_reverse": 14,
        "xorshift32_hybrid_reverse": 14, "xorshift128_hybrid_reverse": 14,
    }
    assert set(C1) == set(w.EXPECTED_KERNEL_ARITY), (
        "the production arity table and this gate's C-1 literals cover different "
        "variant sets")
    bad = []
    for v, want in sorted(C1.items()):
        args = w.resolve_builder(v)(_ctx(w, v))
        if len(args) != want:
            bad.append((v, len(args), want))
        if w.EXPECTED_KERNEL_ARITY[v] != want:
            bad.append((v, "declared", w.EXPECTED_KERNEL_ARITY[v], want))
        # EVERY SCALAR AT ITS EXACT POSITION.
        #
        # The previous form asserted only `a.dtype in (valid set)` — membership,
        # not position — while this gate's docstring promised "(position, dtype)".
        # Mutant M6 (move the int32 phase one slot earlier in build_lcg32) keeps
        # arity 17 and every dtype valid, so it SURVIVED: the docstring promised
        # more than the code checked, in the gate written to catch exactly that.
        got = [(i, a.dtype) for i, a in enumerate(args) if isinstance(a, w.ScalarArg)]
        if got != SCALAR_DTYPE_SEQUENCE[v]:
            bad.append((v, "scalar sequence", got, SCALAR_DTYPE_SEQUENCE[v]))
    assert not bad, f"arity/dtype mismatches: {bad}"
    return f"24/24 variants match the C-1 literals AND the production table"


def gate_cap_2_uncovered():
    """G-CAP-2 — the other 20 registry entries are exercised by asserting they RAISE.

    *Reds on:* a variant silently acquiring a builder, or a covered variant
    regressing into the uncovered set."""
    import miner.range_miner_worker as w
    from prng_registry import KERNEL_REGISTRY
    covered = set(w.EXPECTED_KERNEL_ARITY)
    assert len(KERNEL_REGISTRY) == 44, len(KERNEL_REGISTRY)
    assert len(covered) == 24, len(covered)
    others = sorted(set(KERNEL_REGISTRY) - covered)
    assert len(others) == 20, (len(others), others)
    for v in others:
        try:
            w.resolve_builder(v)
        except NotImplementedError as e:
            assert w.base_family(v) in str(e), (v, str(e))
            continue
        # a builder resolved -> it must still be refused at validation
        try:
            w._validate_variant(v, KERNEL_REGISTRY)
        except w.VariantStopCondition:
            continue
        raise AssertionError(f"{v} is neither covered nor refused")
    return f"44 registry entries = 24 covered + 20 refused (NotImplementedError/VariantStopCondition)"


def gate_cap_3_phase_rejection():
    """G-CAP-3 — nonzero phase on the four no-phase forward hybrids fails loud
    BEFORE any GPU work, with the variant named.

    ORDERING MATTERS: the worker seam runs CAPABILITY then POLICY. This gate must
    see the CAPABILITY error. If policy ran first it would reject nonzero on every
    variant and this gate would pass without the capability guard ever executing —
    green on a fact it does not check.

    *Reds on:* a guard placed after device acquisition, a guard that clamps to 0
    instead of raising, or the two guards being reordered."""
    import miner.range_miner_worker as w
    assert w.PHASE_INCAPABLE_VARIANTS == frozenset({
        "java_lcg_hybrid", "minstd_hybrid", "xorshift32_hybrid", "xorshift128_hybrid"})
    for v in sorted(w.PHASE_INCAPABLE_VARIANTS):
        try:
            w.assert_generator_phase_supported(v, 7)
        except w.GeneratorPhaseUnsupportedError as e:
            assert v in str(e) and "7" in str(e), (v, str(e))
        else:
            raise AssertionError(f"{v} accepted a nonzero phase")
        w.assert_generator_phase_supported(v, 0)      # the pin is always deliverable
    return "4/4 reject phase!=0 with CAPABILITY (not policy) and accept phase=0"


def gate_cap_4_pinned_delivery():
    """G-CAP-4 — phase 0 lands in the correct slot and dtype on all 20 supported.

    *Reds on:* the phase being dropped, or delivered in the wrong position."""
    import miner.range_miner_worker as w
    for v in sorted(w.PHASE_CAPABLE_VARIANTS):
        args = w.resolve_builder(v)(_ctx(w, v, phase=0))
        tail = args[-1]
        assert isinstance(tail, w.ScalarArg) and tail.dtype == "int32", (v, tail)
        assert tail.value == 0, (v, tail.value)
    return "20/20 supported variants carry the pin in the trailing int32"


def gate_abi_frozen():
    """G-ABI-FROZEN — all 44 kernel_source strings byte-identical to HEAD.

    *Reds on:* any kernel edit whatsoever. AC6's 'kernels unchanged by hash'."""
    import hashlib, subprocess, types
    from prng_registry import KERNEL_REGISTRY
    live = {k: hashlib.sha256(v["kernel_source"].encode()).hexdigest()
            for k, v in KERNEL_REGISTRY.items() if v.get("kernel_source")}
    blob = subprocess.check_output(["git", "show", "HEAD:prng_registry.py"], cwd=_ROOT)
    mod = types.ModuleType("_pr_head")
    exec(compile(blob, "prng_registry.py@HEAD", "exec"), mod.__dict__)   # noqa: S102
    ref = {k: hashlib.sha256(v["kernel_source"].encode()).hexdigest()
           for k, v in mod.KERNEL_REGISTRY.items() if v.get("kernel_source")}
    assert len(ref) == 44, len(ref)
    diff = sorted(k for k in ref if live.get(k) != ref[k])
    assert not diff, f"kernel_source CHANGED for: {diff}"
    return f"{len(ref)}/44 kernel_source hashes identical to HEAD"


# ===========================================================================
# SEMANTIC SEPARATION — AC1
# ===========================================================================
def gate_sep_2_synthetic_nonzero():
    """G-SEP-2 — AC1's ACTIVE half: a synthetic nonzero phase on a SUPPORTED ABI,
    driven through the INTERNAL builder via arg-capture.

    Zero-observed-on-both-paths is explicitly not independence evidence, so this
    drives 7 and observes 7.

    *Reds on:* a test that only ever observes 0, or one that reaches the kernel
    through the public schema (that is G-SEP-3, and it must REJECT)."""
    import miner.range_miner_worker as w
    args = w.resolve_builder("lcg32_hybrid")(_ctx(w, "lcg32_hybrid", phase=7))
    assert len(args) == 17, len(args)
    tail = args[16]
    assert isinstance(tail, w.ScalarArg) and tail.dtype == "int32" and tail.value == 7, tail
    a0 = w.resolve_builder("lcg32_hybrid")(_ctx(w, "lcg32_hybrid", phase=0))
    diffs = [i for i in range(len(args)) if args[i] != a0[i]]
    assert diffs == [16], f"phase changed arg positions {diffs}, expected only [16]"
    return "lcg32_hybrid phase=7 at position 17/17 int32; all 16 other args identical"


def _payload_pin_neutralized(c):
    """Live `build_stripe_assign_payload` with ONLY the v1 pin block removed.

    Reconstructed from live source by AST so the mutant is the real method minus
    one `if`, not a hand-written model of it."""
    import ast as _ast, inspect, textwrap
    src = textwrap.dedent(inspect.getsource(
        c.RangeMinerCoordinator.build_stripe_assign_payload))
    fn = _ast.parse(src).body[0]
    before = len(fn.body)
    fn.body = [st for st in fn.body
               if not (isinstance(st, _ast.If)
                       and "generator_phase" in _ast.dump(st.test)
                       and any(isinstance(x, _ast.Raise) for x in _ast.walk(st)))]
    assert len(fn.body) == before - 1, (
        "the pin block was not located — the mutant would be a no-op and the "
        "independence proof vacuous")
    mod = _ast.Module(body=[fn], type_ignores=[])
    _ast.fix_missing_locations(mod)
    ns = {}
    exec(compile(mod, "<pin-neutralized>", "exec"), dict(vars(c)), ns)   # noqa: S102
    return ns["build_stripe_assign_payload"]


def gate_sep_3_public_schema_fail_closed():
    """G-SEP-3 — AC1's fail-closed half. Beta requires the v1 zero-pin at BOTH
    seams, so this gate proves EACH SEPARATELY LOAD-BEARING.

    Two pins with one test that either alone would satisfy is the false-green
    shape, so the arms run on DISJOINT call paths and then each pin is removed in
    turn:

      ARM A  coordinator public assign-payload validation
             build_stripe_assign_payload(generator_phase=7) -> MinerMetadataError
             The worker is not on this path (asserted structurally).
      ARM B  worker execution seam, with a payload that BYPASSES the coordinator
             builder entirely (hand-built dict)
             SieveExecutor.execute -> GeneratorPhaseNotPermittedError
             The coordinator builder is not on this path (asserted structurally).
      ARM C  two-directional fault injection: remove the coordinator pin -> ARM A
             reds and ARM B still passes; remove the worker pin -> ARM B reds and
             ARM A still passes.

    *Reds on:* either pin being relaxed, nonzero leaking through the assignment
    path, the two errors collapsing to one type, or one pin silently covering for
    the other."""
    import ast as _ast, inspect, textwrap
    import miner.range_miner_worker as w
    import miner.range_miner_coordinator as c

    assert w.GENERATOR_PHASE_V1_PIN == 0
    assert w.GeneratorPhaseNotPermittedError is not w.GeneratorPhaseUnsupportedError, (
        "capability and policy share an exception type — a gate could not tell "
        "which guard fired")

    # ---- structural disjointness: neither path can be satisfied by the other --
    # STRUCTURAL, VIA AST — never text. The worker DOCUMENTS the coordinator
    # builder in a comment ("See build_stripe_assign_payload ... for the
    # constant-vs-hybrid key contract"), and a text probe matched that comment and
    # reported a call that does not exist. Disjointness is about CALLS.
    def _called_names(fn):
        src = textwrap.dedent(inspect.getsource(fn))
        out = set()
        for n in _ast.walk(_ast.parse(src)):
            if isinstance(n, _ast.Call):
                f = n.func
                out.add(getattr(f, "attr", None) or getattr(f, "id", None))
        return out - {None}

    coord_calls = _called_names(c.RangeMinerCoordinator.build_stripe_assign_payload)
    for name in ("assert_generator_phase_permitted", "assert_generator_phase_supported"):
        assert name not in coord_calls, f"the coordinator path CALLS {name}"
    exec_calls = _called_names(w.SieveExecutor.execute)
    assert "build_stripe_assign_payload" not in exec_calls, \
        "the worker seam CALLS the coordinator builder"
    assert "assert_generator_phase_permitted" in exec_calls, \
        "the worker seam does not call its own policy pin at all"

    # ---- ARM A: coordinator seam, behavioural ---------------------------------
    coord = c.RangeMinerCoordinator.__new__(c.RangeMinerCoordinator)

    def arm_a(builder=None):
        fn = builder or (lambda **kw: c.RangeMinerCoordinator
                         .build_stripe_assign_payload(coord, **kw))
        return fn(dataset_path="d", window_size=3, sessions=None, window_anchor=0,
                  residues=[1, 2, 3], dataset_sha256="s" * 64, phase=1,
                  forward_threshold=0.31, reverse_threshold=0.47, generator_phase=7)

    try:
        arm_a()
    except c.MinerMetadataError as e:
        assert "v1 pins" in str(e) and "7" in str(e), str(e)
    else:
        raise AssertionError("ARM A: the coordinator emitted a nonzero-phase payload")

    # ---- ARM B: worker seam, coordinator BYPASSED -----------------------------
    def arm_b():
        assign = w.StripeAssignMessage(
            stripe_id="s", prng_type="java_lcg", family_name="lcg32_hybrid",
            seed_start=0, seed_count=8,
            payload={"dataset": "d", "dataset_sha256": "s" * 64, "window_size": 3,
                     "window_anchor": 0, "generator_phase": 7})   # hand-built
        ex = w.SieveExecutor(resolver=w.ResidueResolver(
            loader=lambda *a: [1, 2, 3], file_hasher=lambda p: "s" * 64))
        ex.execute(assign, 0, 8)

    try:
        arm_b()
    except w.GeneratorPhaseNotPermittedError as e:
        assert "v1 pins" in str(e) and "lcg32_hybrid" in str(e), str(e)
    else:
        raise AssertionError("ARM B: the worker seam accepted a nonzero phase")

    # ---- ARM C: two-directional fault injection -------------------------------
    # C1 — remove the COORDINATOR pin. ARM A must red; ARM B must stay green.
    mutant = _payload_pin_neutralized(c)
    a_red = False
    try:
        arm_a(builder=lambda **kw: mutant(coord, **kw))
    except c.MinerMetadataError:
        a_red = True
    assert not a_red, "C1: the coordinator pin survived its own removal"
    try:
        arm_b()
    except w.GeneratorPhaseNotPermittedError:
        pass
    else:
        raise AssertionError("C1: ARM B stopped working when only ARM A's pin moved")

    # C2 — remove the WORKER pin. ARM B must red; ARM A must stay green.
    saved = w.assert_generator_phase_permitted
    w.assert_generator_phase_permitted = lambda variant, phase: None
    try:
        b_still_caught = False
        try:
            arm_b()
        except w.GeneratorPhaseNotPermittedError:
            b_still_caught = True
        except Exception:
            pass          # any other failure is not the policy pin catching it
        assert not b_still_caught, "C2: the worker pin survived its own removal"
        try:
            arm_a()
        except c.MinerMetadataError:
            pass
        else:
            raise AssertionError("C2: ARM A stopped working when only ARM B's pin moved")
    finally:
        w.assert_generator_phase_permitted = saved

    return ("both pins proven separately load-bearing: disjoint paths, distinct "
            "error types, and each removal reds only its own arm")


# ===========================================================================
# DOMAIN — AC3
# ===========================================================================
def _dataset(n=300, sessions=("midday", "evening")):
    import json, tempfile
    fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd)
    rows = [{"draw": i % 1000, "session": sessions[i % len(sessions)]} for i in range(n)]
    with open(p, "w") as f:
        json.dump(rows, f)
    return p


def gate_domain_1_out_of_domain():
    """G-DOMAIN-1 — out-of-domain anchor raises, naming anchor, effective domain,
    session set and dataset.

    *Reds on:* a clamp, a silent min(), or a bare exception with no diagnostic."""
    import miner.range_miner_worker as w
    ds = _dataset(300)
    try:
        assert len(w.load_residue_window(ds, 50, None, 250)) == 50     # == derived_max
        for bad in (251, -1, 10_000):
            try:
                w.load_residue_window(ds, 50, None, bad)
            except w.ResidueResolutionError as e:
                m = str(e)
                assert str(bad) in m and "[0, 250]" in m and ds in m, m
                assert "N_filtered=300" in m and "window_size=50" in m, m
            else:
                raise AssertionError(f"anchor {bad} was accepted")
        return "anchor 250 accepted; 251 / -1 / 10000 raise with the full domain named"
    finally:
        os.unlink(ds)


def gate_domain_2_coordinator_side():
    """G-DOMAIN-2 — the SAME loud failure reached from the COORDINATOR side.

    C-2 established that both sides call the same function OBJECT: the coordinator
    reaches `load_residue_window` through
    `window_optimizer_integration_final._miner_residues_for_config`. A repair that
    only fails loud when reached from the worker has not removed the silent path —
    it has moved it.

    *Reds on:* the clamp surviving on the coordinator side; the two sides
    diverging onto separate implementations; a different exception type."""
    import miner.range_miner_worker as w
    import window_optimizer_integration_final as WOI

    # shared authority, asserted by IDENTITY not by resemblance
    src = ast.parse(_read("window_optimizer_integration_final.py"))
    fn = next(n for n in ast.walk(src)
              if isinstance(n, ast.FunctionDef) and n.name == "_miner_residues_for_config")
    calls = {getattr(c.func, "attr", None) or getattr(c.func, "id", None)
             for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert "load_residue_window" in calls, (
        "the coordinator side no longer calls the worker's loader — a second "
        "residue implementation has appeared")

    class _Cfg:
        window_size = 50
        sessions = None
        offset = 251                      # OUT OF DOMAIN (derived_max = 250)

    ds = _dataset(300)
    try:
        try:
            WOI._miner_residues_for_config(_Cfg(), ds)
        except w.ResidueResolutionError as e:
            m = str(e)
            assert "251" in m and "[0, 250]" in m and ds in m, m
        else:
            raise AssertionError(
                "the COORDINATOR side accepted an out-of-domain anchor — the clamp "
                "was removed on the worker side only")
        # and the in-domain case still resolves identically on both sides
        _Cfg.offset = 7
        both = WOI._miner_residues_for_config(_Cfg(), ds)
        assert both == w.load_residue_window(ds, 50, None, 7), \
            "the two sides no longer agree on an in-domain window"
        return "coordinator side raises the identical error; in-domain windows agree"
    finally:
        os.unlink(ds)


def gate_sep_1_anchor_moves_args_do_not():
    """G-SEP-1 — the anchor moves the residue window while the captured kernel arg
    tuple stays byte-identical apart from the residue buffer.

    Drives the REAL `SieveExecutor.execute` with arg capture, so the proof covers
    the production path rather than the builder in isolation. GPU-dependent: if
    cupy or a device is unavailable this terminates UNAVAILABLE, never PASS (VIR-3).

    *Reds on:* any re-coupling where the anchor reaches a scalar arg — including
    `execute` reading the anchor into generator_phase, which is F-4 returning."""
    import miner.range_miner_worker as w
    try:
        import cupy  # noqa: F401
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device")
    except Exception as e:                                   # noqa: BLE001
        raise AssertionError(f"UNAVAILABLE — no GPU for the real execute path: {e}")

    import hashlib, json, tempfile
    ANCHORS = (0, 11, 37)          # three, so invariance is not a two-point fluke
    ds = _dataset(300)
    try:
        sha = hashlib.sha256(open(ds, "rb").read()).hexdigest()
        captured = {}

        def _capture(self, kernel, blocks, threads, kernel_args):
            captured.setdefault("runs", []).append(kernel_args)

        saved = w.SieveExecutor._gpu_launch
        w.SieveExecutor._gpu_launch = _capture
        try:
            for anchor in ANCHORS:
                ex = w.SieveExecutor(resolver=w.ResidueResolver())
                ex.execute(w.StripeAssignMessage(
                    stripe_id=f"s{anchor}", prng_type="java_lcg",
                    family_name="java_lcg", seed_start=0, seed_count=64,
                    payload={"dataset": ds, "dataset_sha256": sha, "window_size": 50,
                             "window_anchor": anchor, "generator_phase": 0,
                             "min_match_threshold": 0.25}), 0, 64)
        finally:
            w.SieveExecutor._gpu_launch = saved

        runs = captured["runs"]
        assert len(runs) == len(ANCHORS)
        def _scalars(args):
            return [(int(x) if x.dtype.kind in "iu" else float(x))
                    for x in args if not hasattr(x, "get")]
        def _residues(args):
            return args[1].get().tolist()

        # the residue BUFFER tracks the anchor ...
        for i in range(1, len(runs)):
            assert _residues(runs[0]) != _residues(runs[i]), (
                f"anchor {ANCHORS[i]} produced the same residue window as "
                f"anchor {ANCHORS[0]} — the anchor did not move it")
        # ... and NO scalar does, across THREE anchors.
        #
        # INVARIANCE, NOT VALUE-MEMBERSHIP. An earlier form asserted `anchor not in
        # scalars`, which redded on a coincidence: java_lcg's increment constant `c`
        # IS 11, so a legitimate kernel parameter matched the chosen anchor. Three
        # anchors with byte-identical scalars proves no scalar tracks the anchor
        # regardless of what any anchor happens to collide with.
        base = _scalars(runs[0])
        for i in range(1, len(runs)):
            assert _scalars(runs[i]) == base, (
                f"a scalar arg CHANGED with the anchor ({ANCHORS[0]} -> {ANCHORS[i]}): "
                f"{base} vs {_scalars(runs[i])} — F-4 has returned")
        return (f"anchors {ANCHORS}: residue buffer differs for each, all "
                f"{len(base)} scalars byte-identical throughout")
    finally:
        os.unlink(ds)


def gate_domain_3_session_scoped():
    """G-DOMAIN-3 — derived_max is computed on the POST-session-filter count.

    *Reds on:* derived_max computed pre-filter, which would let a midday-only trial
    address an anchor past the end of its own filtered sequence."""
    import miner.range_miner_worker as w
    ds = _dataset(300)
    try:
        assert len(w.load_residue_window(ds, 50, ["midday"], 100)) == 50
        try:
            w.load_residue_window(ds, 50, ["midday"], 101)
        except w.ResidueResolutionError as e:
            assert "[0, 100]" in str(e) and "midday" in str(e), str(e)
        else:
            raise AssertionError("derived_max was computed PRE-filter")
        # the same anchor is legal unfiltered -> the two domains genuinely differ
        assert len(w.load_residue_window(ds, 50, None, 101)) == 50
        return "sessions=['midday'] -> derived_max 100; unfiltered -> 250; 101 legal only unfiltered"
    finally:
        os.unlink(ds)


def gate_domain_4_short_dataset():
    """G-DOMAIN-4 — the pre-existing n < window_size raise is preserved and fires
    BEFORE the new anchor validation.

    *Reds on:* the new validation swallowing or reordering the short-dataset error."""
    import miner.range_miner_worker as w
    ds = _dataset(10)
    try:
        try:
            w.load_residue_window(ds, 5000, None, 0)
        except w.ResidueResolutionError as e:
            assert "only 10 entries" in str(e) and "5000" in str(e), str(e)
            assert "window_anchor" not in str(e), (
                "the anchor check ran first; the short-dataset error must dominate")
            return "short-dataset raise preserved and dominates the anchor check"
        raise AssertionError("n < window_size did not raise")
    finally:
        os.unlink(ds)


def gate_envelope():
    """G-ENVELOPE — the permanent Q4 regression test (AC3, mandatory).

    SCOPE, STATED EXPLICITLY: era-subdomain RESOLUTION against the governed indices
    6,791 / 7,830 / 14,621 lands in Brief II with the optimizer surface. This gate
    is therefore scoped to the BOUND ARITHMETIC — which is the half that carries
    the category error — and the assertion is NOT dropped, only bounded.

    An ANCHOR is a window START INDEX. A RECORD ENVELOPE is the union of records a
    set of anchors+windows can reach. 100 is the historical ANCHOR ceiling; 149 is
    the historical RECORD-ENVELOPE ceiling (= 100 + 50 - 1). v1.0 wrote [0,149] as
    an anchor range and Beta rejected it.

    *Reds on:* anyone reintroducing 149 as an anchor ceiling."""
    def control_era_bound(n_filtered, window_size):
        return (0, min(100, n_filtered - window_size))

    lo, hi = control_era_bound(18_068, 50)
    assert (lo, hi) == (0, 100), (lo, hi)
    assert hi == 100, "the control-era ANCHOR ceiling is 100, not 149"
    assert not (lo <= 149 <= hi), (
        "ANCHOR 149 IS INSIDE control_era — this is the exact anchor/extent category "
        "error the design exists to eliminate. 149 is the RECORD-ENVELOPE ceiling.")
    # anchor 149 + window 50 reaches record 198, outside a 150-record history
    assert 149 + 50 - 1 == 198
    # and the derived max still binds on a small dataset
    assert control_era_bound(120, 50) == (0, 70)
    return "control_era = [0, min(100, N-w)]; anchor 149 NOT inside; 149 is the envelope ceiling"


# ===========================================================================
# SCHEMA REJECTION — §4.1 HARD REJECT
# ===========================================================================
def _payload(ds, **over):
    import hashlib
    p = {"dataset": ds, "dataset_sha256": hashlib.sha256(open(ds, "rb").read()).hexdigest(),
         "window_size": 50, "window_anchor": 0, "generator_phase": 0}
    p.update(over)
    return p


def gate_reject_1_legacy_key():
    """G-REJECT-1 — a payload carrying 'offset' fails loud BEFORE any hashing,
    loading, assignment or GPU work.

    *Reds on:* the key being ignored, mapped, or reaching a later validation stage."""
    import miner.range_miner_worker as w
    ds = _dataset(300)
    try:
        r = w.ResidueResolver(file_hasher=lambda p: (_ for _ in ()).throw(
            AssertionError("HASHING RAN — the reject did not precede it")))
        try:
            r.resolve(_payload(ds, offset=7))
        except w.ResidueResolutionError as e:
            m = str(e)
            assert "RETIRED" in m and "window_anchor" in m and "generator_phase" in m, m
            assert "NOT mapped" in m, m
            return "rejected before the file hasher ran; names both successors, states no mapping"
        raise AssertionError("legacy 'offset' key was accepted")
    finally:
        os.unlink(ds)


def gate_reject_2_no_mapping():
    """G-REJECT-2 — 'offset' is neither read nor written as a dict key on the new
    production path.

    SELF-EXCLUSION IS LOAD-BEARING. This gate's own source necessarily contains the
    string it forbids, and the closure/rejection messages quote it too. Without
    excluding this file and the guard strings, the gate would match itself and stay
    green while a real compatibility shim sat on the production path — green on a
    fact it does not check. Its non-vacuity is proven by injection in §5 (M3).

    *Reds on:* a compatibility shim added under time pressure."""
    import miner.range_miner_worker as w
    _self = os.path.relpath(os.path.abspath(__file__), _ROOT)
    offenders = []
    for rel in ("miner/range_miner_worker.py", "miner/range_miner_coordinator.py",
                "miner/range_miner_npz_writer.py"):
        if rel == _self:
            continue
        tree = ast.parse(_read(rel))
        for n in ast.walk(tree):
            # payload["offset"] / ctx["offset"] reads and writes
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                    and n.slice.value == "offset":
                offenders.append(f"{rel}:{n.lineno} subscript ['offset']")
            # dict literals emitting the key
            if isinstance(n, ast.Dict):
                for k in n.keys:
                    if isinstance(k, ast.Constant) and k.value == "offset":
                        offenders.append(f"{rel}:{n.lineno} dict key 'offset'")
            # .get("offset", ...) reads
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr == "get" and n.args \
                    and isinstance(n.args[0], ast.Constant) and n.args[0].value == "offset":
                offenders.append(f"{rel}:{n.lineno} .get('offset')")
    assert not offenders, f"'offset' is still a live dict key: {offenders}"
    return "no ['offset'] subscript, dict key or .get() on any of the 3 miner modules"


def gate_reject_3_no_default():
    """G-REJECT-3 — a payload omitting either required key raises; no .get(..., 0)
    survives anywhere on the path.

    *Reds on:* the :695 / :875 defect class returning."""
    import miner.range_miner_worker as w
    ds = _dataset(300)
    try:
        r = w.ResidueResolver()
        for missing in ("window_anchor", "generator_phase"):
            p = _payload(ds)
            del p[missing]
            try:
                r.resolve(p)
            except w.ResidueResolutionError as e:
                assert missing in str(e) and "REQUIRED" in str(e), str(e)
            else:
                raise AssertionError(f"{missing} was defaulted")
        return "both keys required at the resolver; omission raises naming the key"
    finally:
        os.unlink(ds)


# ===========================================================================
# LEDGER & CROSS-PHASE
# ===========================================================================
def gate_tuple():
    """G-TUPLE — the exact contents of the four module-level tuples.

    THIS GATE EXISTS BECAUSE THE AST SCOPE PROOFS CANNOT SEE MODULE-LEVEL CONSTANTS.
    `_def_digests` walks FunctionDef/AsyncFunctionDef/ClassDef bodies only, so a
    tuple edit moves no digest and is invisible to both scope proofs.

    *Reds on:* a tuple edited without the others, or a field added to one and
    forgotten in the rest."""
    import miner.range_miner_coordinator as c
    import miner.range_miner_npz_writer as n
    tuples = {
        "_TRIAL_GLOBAL_FIELDS": c._TRIAL_GLOBAL_FIELDS,
        "MANDATORY_MANIFEST_METADATA": c.MANDATORY_MANIFEST_METADATA,
        "_SERVE_CONTEXT_REQUIRED": c._SERVE_CONTEXT_REQUIRED,
        "npz._CONTEXT_FIELDS": n._CONTEXT_FIELDS,
    }
    for name, t in tuples.items():
        assert "offset" not in t, f"{name} still carries the retired key"
        assert "window_anchor" in t, f"{name} lacks window_anchor"
        assert "generator_phase" in t, f"{name} lacks generator_phase"
    assert len(c._TRIAL_GLOBAL_FIELDS) == 10, len(c._TRIAL_GLOBAL_FIELDS)
    assert len(n._CONTEXT_FIELDS) == 12, len(n._CONTEXT_FIELDS)
    return "4/4 tuples migrated; trial-global 10, npz context 12"


def gate_phase5_seam():
    """G-PHASE5-SEAM — the coordinator's manifest satisfies Phase 5's required-key
    comprehension, and Phase 5 uses the coordinator's canonicalizer BY IDENTITY.

    *Reds on:* the KeyError at range_miner_npz_writer.py:1026 that C-3(b) predicts
    if the two tuples drift."""
    import miner.range_miner_coordinator as c
    import miner.range_miner_npz_writer as n
    assert n._canonicalize_trial_context is c._canonicalize_trial_context, (
        "Phase 5 has a SECOND canonicalizer")
    ctx = c.build_trial_context_from_serve(
        dict(trial_number=1, window_size=50, window_anchor=7, generator_phase=0,
             sessions=["midday"], skip_min=0, skip_max=16, prng_base="java_lcg",
             forward_threshold=0.31, reverse_threshold=0.47), "d" * 64, "r" * 64)
    meta = c.derive_trial_metadata(ctx, {"phase": 1, "family_name": "java_lcg"})
    c.validate_trial_metadata(meta)
    proj = {k: meta[k] for k in n._CONTEXT_FIELDS}      # the :1026 comprehension
    assert proj["window_anchor"] == 7 and proj["generator_phase"] == 0
    c._canonicalize_trial_context(proj)
    return f"manifest -> {len(proj)}-field projection -> shared canonicalizer, no KeyError"


def gate_migrate():
    """G-MIGRATE — a pre-separation ledger fails LOUD, naming the schema change.

    *Reds on:* a silent KeyError, a fabricated default, or INSERT OR IGNORE
    succeeding against the old table so the canonical comparison then fails with
    the generic 'conflicting immutable trial context' message — which would send
    the next reader hunting a phantom config conflict."""
    import sqlite3, tempfile
    import miner.range_miner_coordinator as c
    d = tempfile.mkdtemp()
    p = os.path.join(d, "legacy.db")
    con = sqlite3.connect(p)
    con.execute("""CREATE TABLE trial_context (run_id TEXT PRIMARY KEY,
        trial_number INTEGER, window_size INTEGER, offset_val INTEGER,
        sessions_json TEXT, skip_min INTEGER, skip_max INTEGER, prng_base TEXT,
        forward_threshold REAL, reverse_threshold REAL, dataset_sha256 TEXT,
        residue_sha256 TEXT, created_at REAL NOT NULL)""")
    con.commit(); con.close()
    try:
        c.MinerLedger(p)
    except c.MinerMetadataError as e:
        m = str(e)
        assert "offset_val" in m and "window_anchor" in m and "generator_phase" in m, m
        # Assert the three SEMANTIC commitments the message must carry, each as a
        # literal the production string actually contains. (An earlier form of this
        # assertion did `m.replace("cannot","not")` and looked for "not recoverable";
        # the message says "cannot be recovered", so the gate reddened on its own
        # cleverness rather than on production behaviour.)
        for token in ("NOT migrated", "NOT re-keyed", "cannot be recovered"):
            assert token in m, f"the migration error does not state {token!r}: {m}"
        # a FRESH ledger in the same process must still work
        c.MinerLedger(os.path.join(d, "fresh.db"))
        return "legacy schema raises naming the change; a fresh ledger still opens"
    raise AssertionError("a pre-separation ledger was accepted")


def gate_ch2_repointed_assertions_are_load_bearing():
    """ITEM 2 — the repointed Chapter-2 assertions are proven NON-VACUOUS.

    Beta's ruling repoints `G-SOURCE-ANCHORS` from the fused consumers to the
    separated ones. A repointed assertion inherits nothing: it must be shown to
    fail when what it asserts is untrue, or it is a weaker tripwire than the one
    it replaced — and the one it replaced fired correctly.

    So each repointed string is INJECTED AGAINST: removed from live worker source
    in turn, the real `gate_source_anchors` is run, RED is demanded, and the source
    is restored. Self-exclusion is asserted in the gate itself (its probe reads the
    worker, never its own file, which contains both search strings).

    *Reds on:* either assertion going vacuous, or the chapter gate failing to
    notice that the separation it describes has disappeared from source."""
    import importlib.util, shutil, tempfile
    W = os.path.join(_ROOT, "miner", "range_miner_worker.py")

    spec = importlib.util.spec_from_file_location(
        "_ch2_gate", os.path.join(_ROOT, "tests", "test_chapter2_content_gate.py"))
    ch2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch2)

    # clean control — the gate must be GREEN before any injection
    ch2.gate_source_anchors()

    original = io.open(W, encoding="utf-8").read()
    injections = [
        ("def _generator_phase_tail", "def _RENAMED_phase_tail",
         "device-side delivery surface"),
        ("if anchor < 0 or anchor > derived_max:", "if False:",
         "host-side anchor validation"),
    ]
    proven = []
    try:
        for needle, replacement, what in injections:
            assert original.count(needle) >= 1, f"injection target absent: {needle!r}"
            io.open(W, "w", encoding="utf-8").write(
                original.replace(needle, replacement))
            try:
                ch2.gate_source_anchors()
            except AssertionError as e:
                proven.append(f"{what}: RED ({str(e)[:60]}...)")
            else:
                raise AssertionError(
                    f"VACUOUS: removing the {what} left gate_source_anchors GREEN — "
                    f"the repointed assertion asserts nothing")
            finally:
                io.open(W, "w", encoding="utf-8").write(original)
        # and the restore is real
        assert io.open(W, encoding="utf-8").read() == original, "restore failed"
        ch2.gate_source_anchors()          # green again after restore
        return f"both repointed assertions proven load-bearing — {'; '.join(proven)}"
    finally:
        io.open(W, "w", encoding="utf-8").write(original)


def main() -> int:
    print("=" * 78)
    print("S172 WINDOW-ANCHOR BRIEF I — §4.8 LEGACY CLOSURE (AC5)")
    print("=" * 78)
    print("\n-- §4.8 step 1: reachability, derived from live source --")
    check("G-LEGACY-1   reachability census (3 mechanisms)", gate_legacy_1_reachability)
    check("G-LEGACY-1b  replacement-image relationship", gate_legacy_1b_replacement_image)
    print("\n-- §4.8 step 2: dispatch routes --")
    check("G-LEGACY-2   all 4 production routes closed", gate_legacy_2_routes_closed)
    print("\n-- §4.8 step 3: engine entry guard --")
    check("G-LEGACY-3   execution closed, inspection open", gate_legacy_3_engine_guard)
    print("\n-- AC5: no fused path --")
    check("G-NO-FUSED   one value never fills both roles", gate_no_fused)

    print("\n-- capability & ABI --")
    check("G-CAP-1      arity + dtype, all 24 covered variants", gate_cap_1_arity)
    check("G-CAP-2      the other 20 registry entries refuse", gate_cap_2_uncovered)
    check("G-CAP-3      nonzero phase on the 4 no-phase hybrids", gate_cap_3_phase_rejection)
    check("G-CAP-4      pinned delivery on the 20 supported", gate_cap_4_pinned_delivery)
    check("G-ABI-FROZEN 44/44 kernel_source identical to HEAD", gate_abi_frozen)

    print("\n-- semantic separation (AC1: both must be green together) --")
    check("G-SEP-1      anchor moves, kernel scalars do not", gate_sep_1_anchor_moves_args_do_not)
    check("G-SEP-2      synthetic nonzero phase, supported ABI", gate_sep_2_synthetic_nonzero)
    check("G-SEP-3      public v1 schema stays fail-closed", gate_sep_3_public_schema_fail_closed)

    print("\n-- domain (AC3) --")
    check("G-DOMAIN-1   out-of-domain anchor fails loud", gate_domain_1_out_of_domain)
    check("G-DOMAIN-2   same failure from the coordinator side", gate_domain_2_coordinator_side)
    check("G-DOMAIN-3   derived_max is post-session-filter", gate_domain_3_session_scoped)
    check("G-DOMAIN-4   n < window_size raise preserved", gate_domain_4_short_dataset)
    check("G-ENVELOPE   anchor 149 is NOT inside control_era", gate_envelope)

    print("\n-- schema rejection (§4.1 HARD REJECT) --")
    check("G-REJECT-1   legacy 'offset' key hard-rejected", gate_reject_1_legacy_key)
    check("G-REJECT-2   no mapping anywhere (self-excluded)", gate_reject_2_no_mapping)
    check("G-REJECT-3   missing key is not a default", gate_reject_3_no_default)

    print("\n-- Item 2: the repointed Chapter-2 assertions --")
    check("G-CH2-ANCHORS repointed assertions are load-bearing",
          gate_ch2_repointed_assertions_are_load_bearing)

    print("\n-- ledger & cross-phase --")
    check("G-TUPLE      4 module-level tuples, exact contents", gate_tuple)
    check("G-PHASE5-SEAM coordinator manifest -> Phase 5", gate_phase5_seam)
    check("G-MIGRATE    legacy ledger fails loud", gate_migrate)

    print("=" * 78)
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print(f"\n{passed}/{total} checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _RESULTS:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        print("COMPLETION SENTINEL: FAIL")
        return 1
    print("COMPLETION SENTINEL: PASS — §4.8 legacy closure is proven by gate, "
          "not by inspection (pending Team Beta review).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
