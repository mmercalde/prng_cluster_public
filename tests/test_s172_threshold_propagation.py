#!/usr/bin/env python3
"""
test_s172_threshold_propagation.py — S172 optimizer threshold-propagation gates.

Spec: docs/CLAUDE_CODE_INSTRUCTIONS_S172_THRESHOLD_REPAIR.md (REV2) §4.
Audit: docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md.

THE DEFECT THESE GATES CLOSE
----------------------------
Optuna samples `forward_threshold` / `reverse_threshold` and puts them on the
WindowConfig. Two call sites then dropped them before `run_bidirectional_test`:

  Route A (single-process, --n-parallel 1, the default)
      window_optimizer_integration_final.py `test_config` bound
      ft=bounds.default_forward_threshold / rt=bounds.default_reverse_threshold
      IN THE SIGNATURE, so `config.forward_threshold` was never read.

  Route B (--n-parallel > 1)
      window_optimizer_integration_final.py `_local_test` passed
      forward_threshold=_local_bounds.default_forward_threshold — an EXPLICIT
      overwrite of the sampled values that `_worker_obj` had just put on `cfg`.

Every trial therefore filtered at the configured default (live: 0.30/0.30) while
the study recorded the suggested value. Route A was fixed in 3fdf434
(2026-04-30) and silently reverted in 2389b61 (2026-07-07) by a stale-copy
overwrite of the whole file; Route B was never covered by that fix.

WHY THIS HARNESS IS BUILT THE WAY IT IS
---------------------------------------
2389b61 replaced the entire block. A text-anchor check would not have caught it,
because the anchor it was matching disappeared along with the fix. So every gate
here EXTRACTS THE LIVE SOURCE of the real call site (by AST, from the file on
disk, at run time) and EXECUTES it. If someone overwrites either file from a
stale copy again, the extracted source changes behaviour and these gates red.

Nothing here is a hand-written replica of a call site. The only hand-written
values are the ORACLES, which are transcribed literals and are never imported
from a module under test.

THE GATES (brief §4)

  G-ROUTE-A          n_parallel=1: the sampled 0.73/0.31 reach run_bidirectional_test
  G-ROUTE-B          n_parallel>1: the sampled values survive; the explicit
                     _local_bounds.default_* override is gone, and the partition
                     worker really binds the resolver it now calls
  G-KERNEL           the value reaching the kernel is float32(0.73)/float32(0.31),
                     READ AT THE EXECUTOR (real cupy, real RawKernel launch args),
                     not recomputed from config — and chained hop to hop, each hop
                     fed the value OBSERVED at the previous hop
  G-MINER-UNCHANGED  miner/, sieve_gpu_worker.py and prng_registry.py are
                     byte-identical to HEAD, and the D6 threshold harness still
                     passes 17/17
  G-PWC-HYBRID       Option B: PWC variable-skip fails closed at the execution
                     boundary with PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED, a
                     real both-mode trial dies on the first hybrid pass, and the
                     D3.25 v2 return-shape path is NOT collateral damage

MUTANTS (four-part kill rule, VIR-2)

  M1  restore the bounds.default_* signature default in test_config  -> G-ROUTE-A reds
  M2  restore the _local_bounds.default_* override in _local_test    -> G-ROUTE-B reds
  M3  delete the quarantine guard / neuter its body                  -> G-PWC-HYBRID reds

Each mutant proves: applies-exactly-once, mutated-path-executed,
detector-clean-when-unmutated, detector-fires-on-injected-defect.

G-KERNEL executes the REAL GPU executor and therefore requires cupy + a visible
device. It reports UNAVAILABLE and FAILS the run rather than skipping quietly: a
threshold gate that cannot reach the kernel has not tested the thing that broke.

Run:  python tests/test_s172_threshold_propagation.py
"""
import ast
import os
import subprocess
import sys
import textwrap
import traceback
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_UNAV = "\033[93mUNAVAILABLE\033[0m"

_results: List[Tuple[str, str, Any]] = []          # (gate, PASS|FAIL|UNAVAILABLE, detail)
_MUTANTS: List[Tuple[str, str, str, str]] = []     # (id, applies_once, clean, injected)

_INTEG_PATH = os.path.join(_ROOT, "window_optimizer_integration_final.py")
_PWC_PATH = os.path.join(_ROOT, "persistent_worker_coordinator.py")
_DATASET = os.path.join(_ROOT, "daily3.json")


# ═════════════════════════════════════════════════════════════════════════════
# ORACLES — hand-transcribed literals. Never imported from a module under test.
# ═════════════════════════════════════════════════════════════════════════════

# Brief §4: distinctive asymmetric values. This is the exact pair the audit found
# stranded in optuna_studies/window_opt_1778552567.db, recorded by the study and
# executed by no kernel. 0.30 is deliberately NOT used anywhere as an expected
# value: it is the live configured default and is indistinguishable from the
# defect.
ORACLE_FORWARD = 0.73
ORACLE_REVERSE = 0.31

# The live configured default (distributed_config.json search_bounds.*.default),
# transcribed. Used ONLY as the value a defective path collapses to, so that a
# mutant's failure is legible.
DEFECT_COLLAPSE_VALUE = 0.30

# The quarantine code Beta's comparator ruling §5 Option B requires.
ORACLE_QUARANTINE_CODE = "PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED"


# ═════════════════════════════════════════════════════════════════════════════
# Live-source extraction
# ═════════════════════════════════════════════════════════════════════════════

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _find_funcdef(source: str, name: str, path: str) -> ast.AST:
    """The ONE FunctionDef called `name` anywhere in `source`. Ambiguity is fatal:
    if a second definition appears, this harness would silently test the wrong
    one."""
    tree = ast.parse(source)
    found = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(found) != 1:
        raise AssertionError(
            f"expected exactly 1 def {name}() in {os.path.basename(path)}, found {len(found)}"
        )
    return found[0]


def _extract_source(source: str, name: str, path: str) -> str:
    """Dedented source text of the live function, straight off disk."""
    node = _find_funcdef(source, name, path)
    seg = ast.get_source_segment(source, node)
    if not seg:
        raise AssertionError(f"could not extract source for {name}() in {path}")
    return textwrap.dedent(seg)


def _compile_func(func_src: str, module, extra_ns: Dict[str, Any], name: str):
    """Execute the extracted source in a namespace seeded from the REAL module
    globals, so imports/helpers resolve exactly as they do in production, then
    overlay the closure variables the function normally reads from its enclosing
    scope."""
    ns: Dict[str, Any] = dict(vars(module))
    ns.update(extra_ns)
    exec(compile(func_src, f"<live:{name}>", "exec"), ns)
    return ns[name], ns


def _mutate_once(src: str, old: str, new: str, mutant_id: str) -> str:
    """Textual mutation with an applies-exactly-once proof (VIR-2)."""
    n = src.count(old)
    if n != 1:
        raise AssertionError(
            f"{mutant_id}: mutation anchor must appear exactly once, found {n}: {old!r}"
        )
    return src.replace(old, new, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Recorders / stubs
# ═════════════════════════════════════════════════════════════════════════════

class _Bounds:
    """Stands in for SearchBounds. Carries the LIVE configured defaults so a
    defective path collapses to a value we can name."""
    default_forward_threshold = DEFECT_COLLAPSE_VALUE
    default_reverse_threshold = DEFECT_COLLAPSE_VALUE
    session_options = [["midday", "evening"]]


class _Recorder:
    """Captures every kwarg handed to run_bidirectional_test."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        return {"_recorder": True}

    @property
    def last(self) -> Dict[str, Any]:
        assert self.calls, "run_bidirectional_test was never reached"
        return self.calls[-1]


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


class _DispatchReached(Exception):
    """Raised by the PWC stub when execution reaches worker dispatch — i.e. the
    quarantine guard did NOT fire."""


def _window_config(ft: float, rt: float):
    from window_optimizer import WindowConfig
    return WindowConfig(window_size=20, offset=0, sessions=["midday", "evening"],
                        skip_min=0, skip_max=10,
                        forward_threshold=ft, reverse_threshold=rt)


# ═════════════════════════════════════════════════════════════════════════════
# Route A / Route B drivers — each executes the LIVE call-site source
# ═════════════════════════════════════════════════════════════════════════════

def _run_route_a(integ_src: str) -> Dict[str, Any]:
    """Execute the live `test_config` closure and return the kwargs it hands to
    run_bidirectional_test."""
    import window_optimizer_integration_final as integ

    src = _extract_source(integ_src, "test_config", _INTEG_PATH)
    rec = _Recorder()
    ns = {
        "bounds": _Bounds(),
        "seed_start": 0,
        "seed_count": 1000,
        "trial_counter": {"count": 0},
        "n_parallel": 1,
        "self": object(),
        "dataset_path": _DATASET,
        "prng_base": "java_lcg",
        "test_both_modes": False,
        "survivor_accumulator": {"forward": [], "reverse": [], "bidirectional": []},
        "enable_pruning": False,
        "run_bidirectional_test": rec,
        "_get_partition_coordinator": lambda p: object(),
        "_PARALLEL_PARTITIONS": [["node"]],
    }
    fn, _ = _compile_func(src, integ, ns, "test_config")
    fn(_window_config(ORACLE_FORWARD, ORACLE_REVERSE))
    return rec.last


def _route_b_resolver(integ_src: str):
    """The Route-B partition worker runs in a SEPARATE PROCESS and re-imports the
    module, so the resolver it calls must actually be bound by an import inside
    `_partition_worker`. Injecting it here would mask a NameError that only
    appears under --n-parallel > 1. So: find the binding in the live source, and
    execute exactly that import to obtain the object."""
    node = _find_funcdef(integ_src, "_partition_worker", _INTEG_PATH)
    stmts = [n for n in ast.walk(node)
             if isinstance(n, (ast.Import, ast.ImportFrom))
             and any(a.asname == "_resolve_dt" or a.name == "_resolve_dt" for a in n.names)]
    if len(stmts) != 1:
        raise AssertionError(
            "_partition_worker must bind the shared threshold resolver exactly once "
            f"(found {len(stmts)} import statements binding `_resolve_dt`) — without it "
            "the Route-B call site raises NameError in the partition subprocess"
        )
    ns: Dict[str, Any] = {}
    exec(compile(ast.unparse(stmts[0]), "<live:_partition_worker import>", "exec"), ns)
    return ns["_resolve_dt"]


def _run_route_b(integ_src: str, resolver=None) -> Dict[str, Any]:
    """Execute the live `_local_test` closure and return the kwargs it hands to
    run_bidirectional_test (bound as `_wbt` in the partition worker)."""
    import window_optimizer_integration_final as integ

    src = _extract_source(integ_src, "_local_test", _INTEG_PATH)
    rec = _Recorder()
    ns = {
        "_wbt": rec,
        "_wcoord": object(),
        "dataset_path_w": _DATASET,
        "seed_start_w": 0,
        "seed_count_w": 1000,
        "prng_base_w": "java_lcg",
        "test_both_modes_w": False,
        "_local_bounds": _Bounds(),
        "_local_acc": {"forward": [], "reverse": [], "bidirectional": []},
        "_tctr": {"n": 0},
        "_resolve_dt": resolver if resolver is not None else _route_b_resolver(integ_src),
    }
    fn, _ = _compile_func(src, integ, ns, "_local_test")
    fn(_window_config(ORACLE_FORWARD, ORACLE_REVERSE))
    return rec.last


# ═════════════════════════════════════════════════════════════════════════════
# G-ROUTE-A
# ═════════════════════════════════════════════════════════════════════════════

def gate_route_a() -> str:
    integ_src = _read(_INTEG_PATH)
    kw = _run_route_a(integ_src)

    ft, rt = kw.get("forward_threshold"), kw.get("reverse_threshold")
    assert ft == ORACLE_FORWARD, (
        f"Route A dropped the sampled forward threshold: run_bidirectional_test got "
        f"{ft!r}, expected {ORACLE_FORWARD!r}"
        + (f" — collapsed to the configured default, the exact defect"
           if ft == DEFECT_COLLAPSE_VALUE else "")
    )
    assert rt == ORACLE_REVERSE, (
        f"Route A dropped the sampled reverse threshold: run_bidirectional_test got "
        f"{rt!r}, expected {ORACLE_REVERSE!r}"
        + (f" — collapsed to the configured default, the exact defect"
           if rt == DEFECT_COLLAPSE_VALUE else "")
    )
    # not collapsed, not swapped
    assert ft != rt, "forward and reverse thresholds collapsed to one value"

    # An explicit caller argument must still win over the config (documented
    # precedence: explicit > config > bounds default).
    import window_optimizer_integration_final as integ
    src = _extract_source(integ_src, "test_config", _INTEG_PATH)
    rec = _Recorder()
    ns = {
        "bounds": _Bounds(), "seed_start": 0, "seed_count": 1000,
        "trial_counter": {"count": 0}, "n_parallel": 1, "self": object(),
        "dataset_path": _DATASET, "prng_base": "java_lcg", "test_both_modes": False,
        "survivor_accumulator": {"forward": [], "reverse": [], "bidirectional": []},
        "enable_pruning": False, "run_bidirectional_test": rec,
        "_get_partition_coordinator": lambda p: object(), "_PARALLEL_PARTITIONS": [["n"]],
    }
    fn, _ = _compile_func(src, integ, ns, "test_config")
    fn(_window_config(ORACLE_FORWARD, ORACLE_REVERSE), ft=0.61, rt=0.62)
    assert rec.last["forward_threshold"] == 0.61 and rec.last["reverse_threshold"] == 0.62, (
        "an explicit ft/rt argument must override the config value"
    )
    return f"sampled {ORACLE_FORWARD}/{ORACLE_REVERSE} reached run_bidirectional_test intact"


# ═════════════════════════════════════════════════════════════════════════════
# G-ROUTE-B
# ═════════════════════════════════════════════════════════════════════════════

def gate_route_b() -> str:
    integ_src = _read(_INTEG_PATH)

    # (a) the partition subprocess really binds the resolver it calls
    resolver = _route_b_resolver(integ_src)

    # (b) behavioural: the sampled values survive the hop
    kw = _run_route_b(integ_src, resolver)
    ft, rt = kw.get("forward_threshold"), kw.get("reverse_threshold")
    assert ft == ORACLE_FORWARD, (
        f"Route B dropped the sampled forward threshold: _wbt got {ft!r}, "
        f"expected {ORACLE_FORWARD!r}"
        + (" — collapsed to the configured default, the exact defect"
           if ft == DEFECT_COLLAPSE_VALUE else "")
    )
    assert rt == ORACLE_REVERSE, (
        f"Route B dropped the sampled reverse threshold: _wbt got {rt!r}, "
        f"expected {ORACLE_REVERSE!r}"
        + (" — collapsed to the configured default, the exact defect"
           if rt == DEFECT_COLLAPSE_VALUE else "")
    )
    assert ft != rt, "forward and reverse thresholds collapsed to one value"

    # (c) structural: the explicit override is gone from the live call site.
    #     Secondary to (b) — a text check alone would not have caught 2389b61 —
    #     but it names the exact regression if it ever returns.
    local_src = _extract_source(integ_src, "_local_test", _INTEG_PATH)
    for banned in ("forward_threshold=_local_bounds.default_forward_threshold",
                   "reverse_threshold=_local_bounds.default_reverse_threshold"):
        assert banned not in local_src, (
            f"the Route-B explicit override is back in _local_test: {banned}"
        )

    # (d) _worker_obj still puts the sampled values on cfg — the producer half of
    #     the hop. If this ever stops, (b) would pass vacuously.
    wo_src = _extract_source(integ_src, "_worker_obj", _INTEG_PATH)
    for needed in ("trial.suggest_float('forward_threshold'",
                   "trial.suggest_float('reverse_threshold'",
                   "forward_threshold=round(ft, 2)",
                   "reverse_threshold=round(rt, 2)"):
        assert needed in wo_src, f"_worker_obj no longer produces the sampled value: {needed}"

    return f"sampled {ORACLE_FORWARD}/{ORACLE_REVERSE} survived Route B; override removed"


# ═════════════════════════════════════════════════════════════════════════════
# G-KERNEL — chained, read at the executor
# ═════════════════════════════════════════════════════════════════════════════

def _pwc_job_dict(prng_type: str, threshold: float) -> Dict[str, Any]:
    """Build a job dict by executing the LIVE `dispatch_chunk` closure out of
    persistent_worker_coordinator.run_sieve_pass — the real producer of the job
    the worker executes. Nothing here hand-writes a job."""
    import persistent_worker_coordinator as pwc

    src = _extract_source(_read(_PWC_PATH), "dispatch_chunk", _PWC_PATH)
    captured: Dict[str, Any] = {}

    class _Self:
        _tcp_transport = object()          # force the TCP branch
        _progress_writer = None
        logger = _Logger()

        @staticmethod
        def _dispatch_to_tcp(job):
            captured["job"] = job
            return {"status": "ok", "survivors": []}

    ns = {
        "self": _Self(),
        "sessions": ["midday", "evening"],
        "skip_range": [0, 10],
        "prng_type": prng_type,
        "dataset_path": _DATASET,
        "target_file": _DATASET,
        "window_size": 20,
        "threshold": threshold,
        "offset": 0,
        "strategies": None,
        "is_hybrid": "_hybrid" in prng_type,
        "phase2_threshold": 0.5,
        "lock": __import__("threading").Lock(),
        "results_by_chunk": {},
    }
    fn, _ = _compile_func(src, pwc, ns, "dispatch_chunk")
    fn(0, 0, 200_000, None)
    assert "job" in captured, "dispatch_chunk never reached worker dispatch"
    return captured["job"]


def gate_kernel() -> str:
    """Chain the value hop to hop, each hop fed the value OBSERVED at the
    previous hop, and read the final scalar off the real kernel launch."""
    try:
        import cupy as cp
        import numpy as np
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no visible CUDA/ROCm device")
    except Exception as exc:      # VIR-5: unobservable is not clean
        raise AssertionError(
            f"UNAVAILABLE: G-KERNEL needs cupy + a visible device to read the "
            f"effective threshold AT THE EXECUTOR ({exc}). A threshold gate that "
            f"cannot reach the kernel has not tested the thing that broke."
        )

    import sieve_gpu_worker as sgw
    if not getattr(sgw, "GPU_AVAILABLE", False):
        raise AssertionError("UNAVAILABLE: sieve_gpu_worker reports GPU_AVAILABLE=False")
    if not os.path.exists(_DATASET):
        raise AssertionError(f"UNAVAILABLE: dataset not present at {_DATASET}")

    integ_src = _read(_INTEG_PATH)

    # hop 1-3: Optuna sample -> WindowConfig -> the live Route-A call site
    route_a = _run_route_a(integ_src)
    observed_ft = route_a["forward_threshold"]
    observed_rt = route_a["reverse_threshold"]

    results = {}
    for direction, prng_type, observed in (
        ("forward", "java_lcg", observed_ft),
        ("reverse", "java_lcg_reverse", observed_rt),
    ):
        # hop 6-8: the observed value -> the live PWC job producer
        job = _pwc_job_dict(prng_type, observed)
        assert job["min_match_threshold"] == observed, (
            f"{direction}: PWC job dict carries {job['min_match_threshold']!r}, "
            f"not the {observed!r} it was handed"
        )
        job["seed_start"] = 0
        job["seed_end"] = 200_000

        # hop 9: the real executor. Capture the scalar at the kernel launch —
        # this is a READ AT THE EXECUTOR, not a recomputation from config.
        launched: List[Any] = []
        real_get_kernel = sgw._get_kernel

        def _wrapped_get_kernel(family, _real=real_get_kernel, _sink=launched):
            kernel, config = _real(family)

            class _Recording:
                def __call__(self, grid, block, args):
                    _sink.append(args)
                    return kernel(grid, block, args)

            return _Recording(), config

        sgw._get_kernel = _wrapped_get_kernel
        try:
            sgw.run_sieve_job(job, gpu_id=0)
        finally:
            sgw._get_kernel = real_get_kernel

        assert launched, f"{direction}: no kernel was launched — nothing to read"
        # Constant-skip kernel arg layout (sieve_gpu_worker.py kernel_args):
        #   seeds, residues, survivors, match_rates, best_skips, survivor_count,
        #   n_seeds, k, skip_min, skip_max, threshold, ...
        scalar = launched[0][10]
        assert isinstance(scalar, np.float32), (
            f"{direction}: kernel arg 10 is {type(scalar)}, expected float32 threshold"
        )
        effective = float(scalar)
        results[direction] = effective

    expect_f = float(np.float32(ORACLE_FORWARD))
    expect_r = float(np.float32(ORACLE_REVERSE))
    assert results["forward"] == expect_f, (
        f"the kernel received forward threshold {results['forward']!r}, expected "
        f"float32({ORACLE_FORWARD}) = {expect_f!r}"
        + (" — the configured default, i.e. the defect is live"
           if abs(results["forward"] - DEFECT_COLLAPSE_VALUE) < 1e-6 else "")
    )
    assert results["reverse"] == expect_r, (
        f"the kernel received reverse threshold {results['reverse']!r}, expected "
        f"float32({ORACLE_REVERSE}) = {expect_r!r}"
        + (" — the configured default, i.e. the defect is live"
           if abs(results["reverse"] - DEFECT_COLLAPSE_VALUE) < 1e-6 else "")
    )
    assert results["forward"] != results["reverse"], "directional thresholds collapsed"
    return (f"executor read: forward={results['forward']!r} reverse={results['reverse']!r} "
            f"(real cupy RawKernel launch args)")


# ═════════════════════════════════════════════════════════════════════════════
# G-MINER-UNCHANGED
# ═════════════════════════════════════════════════════════════════════════════

def gate_miner_unchanged() -> str:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=_ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"git status failed: {r.stderr}"
    touched = {ln[3:].strip() for ln in r.stdout.splitlines() if ln[3:].strip()}
    frozen_prefixes = ("miner/",)
    frozen_files = {"sieve_gpu_worker.py", "prng_registry.py",
                    "persistent/pwc_protocol.py"}

    # [S172 Phase 6-P0.5] Registered exemptions, same standing rule gate 22 uses
    # (tests/test_s172_phase4_coordinator.py:1602): a deliverable-scoped tripwire
    # is extended by REGISTERING the later deliverable's files with a rationale,
    # never by loosening the predicate.
    #
    # This gate's subject is the THRESHOLD REPAIR (8a55a68): that repair had to
    # leave the miner alone because the miner was already correct and PWC held
    # the defect. P0.5 is a different deliverable with different authority, and
    # it necessarily changes the miner — the run-scoped dataset freeze and the
    # FileNotFoundError classification both live there. Three files, all confined
    # to the dataset-authority seam:
    #   * miner/dataset_authority.py            — NEW; pointer resolution, the
    #     run-start freeze, per-node provisioning/verification, run provenance.
    #   * miner/range_miner_coordinator.py      — serve_trial's dataset_sha256
    #     moves from per-TRIAL derivation to the run-start freeze.
    #   * miner/range_miner_worker.py           — DatasetProvisioningError plus
    #     chained, path-and-node-naming classification.
    #
    # What this gate still protects is UNCHANGED and is the part that matters
    # here: the kernel/executor surface (sieve_gpu_worker.py, prng_registry.py)
    # and pwc_protocol.py stay byte-identical, no threshold logic is touched by
    # any of the three, and the behavioural half below still requires
    # D6-threshold 17/17. Any OTHER miner/ file appearing here is still a red.
    p05_registered = {
        "miner/dataset_authority.py",
        "miner/range_miner_coordinator.py",
        "miner/range_miner_worker.py",
    }

    # [S172 §4.3 ADMISSION LIVENESS REPAIR — Beta Ruling 1] A THIRD deliverable,
    # registered by the same standing rule and APPENDED to the P0.5 set above
    # rather than replacing it (the P0.5 registration and its verification below
    # stay exactly as they were).
    #
    # This repair separates admission liveness from execution maintenance in
    # serve_trial: the pre-assignment wait for expected_workers becomes bounded
    # (worker_admission_timeout, default 180s) and fails the trial explicitly,
    # while dispatch / lease expiry / completion evaluation run unconditionally
    # once a stage is assigned — which is what makes the Blocker-3 failure matrix
    # reachable after a mid-run worker loss instead of hanging silently
    # (docs/FLEET_STATE_REQUIREMENTS_v1.md §4.3).
    #   * miner/range_miner_coordinator.py — already in the P0.5 set above; named
    #     again here because it changes for a reason that belongs to THIS
    #     deliverable, and that reason must be legible in this list.
    #   * miner/__init__.py — export-only (DEFAULT_WORKER_ADMISSION_TIMEOUT
    #     alongside run_trial_miner) so the integration call site imports the
    #     default instead of restating it.
    # The kernel/executor surface this gate exists to protect is untouched, and
    # the threshold-token bleed check below applies to these files unchanged: the
    # repair moves a WORKER-COUNT guard, never a threshold.
    admission_registered = {
        "miner/range_miner_coordinator.py",
        "miner/__init__.py",
    }
    registered = p05_registered | admission_registered
    offenders = sorted(p for p in touched
                       if (p.startswith(frozen_prefixes) or p in frozen_files)
                       and p not in registered)
    assert not offenders, (
        "this repair must not touch the miner or the kernel/executor surface "
        f"(the miner is correct; PWC has the defect): {offenders}"
    )

    # A registration is a claim, so verify it rather than trust it: the exempted
    # P0.5 files must not touch threshold logic. This keeps THIS gate's actual
    # subject enforced even on the files it now permits to change.
    if registered & touched:
        # encoding pinned explicitly, as the D6 sub-harness call below does: the
        # diff carries non-ASCII prose and the ambient default codec is ascii.
        d = subprocess.run(["git", "diff", "--unified=0", "--",
                            *sorted(registered & touched)],
                           cwd=_ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        assert d.returncode == 0, f"git diff failed: {d.stderr}"
        changed = [ln for ln in d.stdout.splitlines()
                   if (ln.startswith("+") or ln.startswith("-"))
                   and not ln.startswith(("+++", "---"))]
        threshold_tokens = ("forward_threshold", "reverse_threshold",
                            "resolve_directional_threshold", "min_match_threshold",
                            "effective_threshold", "threshold_provenance")
        bleed = [ln.strip() for ln in changed
                 if any(t in ln for t in threshold_tokens)]
        assert not bleed, (
            "a REGISTERED miner file changed THRESHOLD logic — that is outside "
            "the registered dataset-authority / admission-liveness scope and is "
            f"exactly what this gate exists to catch:\n  " + "\n  ".join(bleed[:10])
        )

    # Behavioural: the D6 threshold contract still holds end to end.
    suite = os.path.join(_ROOT, "tests", "test_s172_phase5_d6_threshold_path.py")
    assert os.path.exists(suite), f"D6 threshold harness missing: {suite}"
    p = subprocess.run([sys.executable, suite], cwd=_ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=3600)
    tail = (p.stdout or "")[-4000:] + (p.stderr or "")[-2000:]
    assert p.returncode == 0, (
        f"D6-threshold harness regressed (rc={p.returncode}). Tail:\n{tail}"
    )
    # Parse its own tally line rather than trusting the exit code alone.
    import re
    m = re.search(r"(\d+)/(\d+) D6 threshold-path checks green", p.stdout or "")
    assert m, ("D6-threshold printed no tally line — a missing completion sentinel "
               "is failure, never success (VIR-3). Tail:\n" + tail)
    got, total = int(m.group(1)), int(m.group(2))
    assert got == total == 17, (
        f"D6-threshold must stay 17/17, got {got}/{total}"
    )
    if registered & touched:
        return ("kernel/executor surface (sieve_gpu_worker, prng_registry, "
                f"pwc_protocol) byte-identical to HEAD; the "
                f"{len(registered & touched)} registered miner file(s) changed "
                "only the dataset-authority / admission-liveness seams (no "
                "threshold-token bleed); D6-threshold 17/17")
    return "miner/ + executor surface byte-identical to HEAD; D6-threshold 17/17"


# ═════════════════════════════════════════════════════════════════════════════
# G-PWC-HYBRID (Option B — quarantine)
# ═════════════════════════════════════════════════════════════════════════════

def _drive_pwc_run_sieve_pass(src_override: Optional[str] = None):
    """Execute the LIVE `run_sieve_pass` for a hybrid prng_type. The stub `self`
    raises _DispatchReached at the first thing the guard is supposed to prevent,
    so 'the guard did not fire' is observable and cannot be confused with a pass."""
    import persistent_worker_coordinator as pwc

    src = src_override if src_override is not None else _extract_source(
        _read(_PWC_PATH), "run_sieve_pass", _PWC_PATH)

    class _Self:
        _tcp_transport = None
        _progress_writer = None
        seed_cap_amd = 1_000_000
        min_workers = 1
        logger = _Logger()

        @staticmethod
        def _get_available_workers():
            raise _DispatchReached("run_sieve_pass reached worker acquisition")

    fn, _ = _compile_func(src, pwc, {}, "run_sieve_pass")
    return fn(_Self(), prng_type="java_lcg_hybrid", residues=[1, 2, 3],
              total_seeds=1000, threshold=ORACLE_FORWARD, window_size=20,
              output_file="/dev/null", dataset_path=_DATASET)


def _drive_pwc_both_modes_trial(hybrid_raises: bool):
    """Execute the LIVE `run_trial_persistent` both-mode against a FAKE sieve —
    the same shape D3.25's G1 uses to assert the v2 four-map return contract.

    `hybrid_raises=True` models the real coordinator (its run_sieve_pass carries
    the quarantine guard); `hybrid_raises=False` models D3.25's fake, which never
    reaches the guarded code at all. Both must behave as designed:
      * real  -> the trial fails closed on the first hybrid pass,
      * fake  -> the return-shape contract still runs to completion.
    """
    import persistent_worker_coordinator as pwc

    class _FakeSieve:
        _progress_writer = None
        logger = _Logger()

        def __init__(self):
            self.calls: List[str] = []

        def run_sieve_pass(self, prng_type=None, **kw):
            self.calls.append(prng_type)
            if hybrid_raises and "_hybrid" in (prng_type or ""):
                pwc.assert_pwc_hybrid_not_quarantined(prng_type, call_site="fake dispatch")
            return {"survivors": [7, 42], "match_rates": [0.9, 0.8]}

        def startup(self):
            pass

        def shutdown(self):
            pass

    fake = _FakeSieve()
    real = pwc.PersistentWorkerCoordinator
    pwc.PersistentWorkerCoordinator = lambda **kw: fake
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            out = pwc.run_trial_persistent(
                coordinator_cfg="unused.json",
                config=_window_config(ORACLE_FORWARD, ORACLE_REVERSE),
                trial_number=1, prng_base="java_lcg", residues=[1, 2],
                total_seeds=1000, forward_threshold=ORACLE_FORWARD,
                reverse_threshold=ORACLE_REVERSE, test_both_modes=True,
                dataset_path="unused.csv")
        return out, fake.calls
    finally:
        pwc.PersistentWorkerCoordinator = real


def _detect_quarantine(driver, src_override: Optional[str] = None) -> None:
    """THE detector. Shared verbatim by G-PWC-HYBRID and mutant M3, so the mutant
    proves this exact detector fires — not a lookalike written for the mutant."""
    import persistent_worker_coordinator as pwc

    exc_type = getattr(pwc, "PwcHybridThresholdContractUncertified", None)
    assert exc_type is not None, "quarantine exception type missing"
    try:
        driver(src_override)
    except exc_type as exc:
        assert ORACLE_QUARANTINE_CODE in str(exc), (
            f"quarantine error must name {ORACLE_QUARANTINE_CODE}: {exc}"
        )
        return
    except _DispatchReached as exc:
        raise AssertionError(f"quarantine did not fire — reached dispatch: {exc}")
    raise AssertionError("a PWC hybrid pass completed instead of failing closed")


def gate_pwc_hybrid() -> str:
    import persistent_worker_coordinator as pwc

    assert getattr(pwc, "PWC_HYBRID_QUARANTINE_CODE", None) == ORACLE_QUARANTINE_CODE, (
        f"PWC hybrid quarantine code must be {ORACLE_QUARANTINE_CODE!r}"
    )

    # (1) the execution boundary itself — run_sieve_pass fails closed on hybrid
    _detect_quarantine(_drive_pwc_run_sieve_pass)

    # (2) a real both-mode trial therefore fails closed too, and does so on the
    #     FIRST hybrid pass — after the two constant passes, before any hybrid
    #     survivor exists. Proven by the call log, not assumed.
    exc_type = pwc.PwcHybridThresholdContractUncertified
    try:
        _drive_pwc_both_modes_trial(hybrid_raises=True)
        raise AssertionError("a both-mode PWC trial completed instead of failing closed")
    except exc_type as exc:
        assert ORACLE_QUARANTINE_CODE in str(exc), f"wrong error: {exc}"

    # (3) the quarantine must be scoped to EXECUTION, not to the return-shape
    #     contract: D3.25's G1 drives this same both-mode path against a fake
    #     sieve that never reaches the guard, and must still complete. Blocking
    #     that would quarantine a contract check, not a known-wrong execution.
    out, calls = _drive_pwc_both_modes_trial(hybrid_raises=False)
    assert isinstance(out, dict) and out.get("schema_version"), (
        "the both-mode v2 return-shape path must still complete against a fake "
        "sieve — the quarantine is scoped to execution, not to the contract"
    )
    assert any("_hybrid" in c for c in calls), (
        "the fake-sieve both-mode path did not reach the variable-skip passes, so "
        "check (3) would be vacuous"
    )

    # Constant-skip is explicitly NOT quarantined — the guard must be scoped to
    # hybrid only, or it would take out the diagnostic comparator Beta kept.
    # Reaching worker acquisition is the CORRECT outcome here.
    for constant_family in ("java_lcg", "java_lcg_reverse"):
        pwc.assert_pwc_hybrid_not_quarantined(constant_family)   # must not raise
    return ("variable-skip fails closed at the execution boundary and kills a real "
            "both-mode trial; constant-skip and the D3.25 v2 shape path unaffected")


# ═════════════════════════════════════════════════════════════════════════════
# MUTANTS — four-part kill rule
# ═════════════════════════════════════════════════════════════════════════════

def _detector_fires(fn, *args, **kwargs) -> Tuple[bool, str]:
    try:
        fn(*args, **kwargs)
        return False, "detector did not fire"
    except AssertionError as exc:
        return True, str(exc).splitlines()[0][:200]
    except Exception as exc:  # a crash is a fire, but name it
        return True, f"{type(exc).__name__}: {str(exc).splitlines()[0][:180]}"


def mutant_m1_route_a() -> Tuple[str, str, str]:
    """Restore the bounds.default_* signature default in test_config."""
    integ_src = _read(_INTEG_PATH)
    src = _extract_source(integ_src, "test_config", _INTEG_PATH)

    src = _mutate_once(
        src, "ft=None, rt=None,",
        "ft=bounds.default_forward_threshold, rt=bounds.default_reverse_threshold,", "M1")
    src = _mutate_once(
        src,
        "ft = resolve_directional_threshold(config, 'forward', ft, bounds.default_forward_threshold)",
        "pass", "M1")
    src = _mutate_once(
        src,
        "rt = resolve_directional_threshold(config, 'reverse', rt, bounds.default_reverse_threshold)",
        "pass", "M1")

    import window_optimizer_integration_final as integ
    rec = _Recorder()
    ns = {
        "bounds": _Bounds(), "seed_start": 0, "seed_count": 1000,
        "trial_counter": {"count": 0}, "n_parallel": 1, "self": object(),
        "dataset_path": _DATASET, "prng_base": "java_lcg", "test_both_modes": False,
        "survivor_accumulator": {"forward": [], "reverse": [], "bidirectional": []},
        "enable_pruning": False, "run_bidirectional_test": rec,
        "_get_partition_coordinator": lambda p: object(), "_PARALLEL_PARTITIONS": [["n"]],
    }
    fn, _ = _compile_func(src, integ, ns, "test_config")
    fn(_window_config(ORACLE_FORWARD, ORACLE_REVERSE))
    executed = f"mutated test_config executed; run_bidirectional_test got " \
               f"{rec.last['forward_threshold']!r}/{rec.last['reverse_threshold']!r}"

    def _detect():
        kw = rec.last
        assert kw["forward_threshold"] == ORACLE_FORWARD, "forward dropped"
        assert kw["reverse_threshold"] == ORACLE_REVERSE, "reverse dropped"

    fired, why = _detector_fires(_detect)
    assert fired, "M1: G-ROUTE-A's detector did not fire on the injected defect"
    return executed, why, "applies-once verified on 3 anchors"


def mutant_m2_route_b() -> Tuple[str, str, str]:
    """Restore the explicit _local_bounds.default_* override in _local_test."""
    integ_src = _read(_INTEG_PATH)
    src = _extract_source(integ_src, "_local_test", _INTEG_PATH)

    # Rebuild the two keyword arguments as the pre-repair explicit override.
    # Line-wrapping-tolerant, so the mutant survives a reformat of the call site.
    import re
    new_src, n_f = re.subn(
        r"forward_threshold=_resolve_dt\((?:[^()]|\([^()]*\))*\),",
        "forward_threshold=_local_bounds.default_forward_threshold,", src)
    new_src, n_r = re.subn(
        r"reverse_threshold=_resolve_dt\((?:[^()]|\([^()]*\))*\),",
        "reverse_threshold=_local_bounds.default_reverse_threshold,", new_src)
    assert n_f == 1 and n_r == 1, (
        f"M2: mutation must apply exactly once per direction, applied {n_f}/{n_r}"
    )

    import window_optimizer_integration_final as integ
    rec = _Recorder()
    ns = {
        "_wbt": rec, "_wcoord": object(), "dataset_path_w": _DATASET,
        "seed_start_w": 0, "seed_count_w": 1000, "prng_base_w": "java_lcg",
        "test_both_modes_w": False, "_local_bounds": _Bounds(),
        "_local_acc": {"forward": [], "reverse": [], "bidirectional": []},
        "_tctr": {"n": 0}, "_resolve_dt": _route_b_resolver(integ_src),
    }
    fn, _ = _compile_func(new_src, integ, ns, "_local_test")
    fn(_window_config(ORACLE_FORWARD, ORACLE_REVERSE))
    executed = f"mutated _local_test executed; _wbt got " \
               f"{rec.last['forward_threshold']!r}/{rec.last['reverse_threshold']!r}"

    def _detect():
        kw = rec.last
        assert kw["forward_threshold"] == ORACLE_FORWARD, "forward dropped"
        assert kw["reverse_threshold"] == ORACLE_REVERSE, "reverse dropped"

    fired, why = _detector_fires(_detect)
    assert fired, "M2: G-ROUTE-B's detector did not fire on the injected defect"
    return executed, why, f"applies-once verified ({n_f}+{n_r} substitutions)"


def mutant_m3_pwc_quarantine() -> Tuple[str, str, str]:
    """Delete the quarantine guard from each PWC call site."""
    pwc_src = _read(_PWC_PATH)
    notes = []

    # call site 1 — run_sieve_pass
    s1 = _extract_source(pwc_src, "run_sieve_pass", _PWC_PATH)
    s1 = _mutate_once(
        s1,
        'assert_pwc_hybrid_not_quarantined(prng_type, call_site="run_sieve_pass")',
        "pass", "M3/run_sieve_pass")
    fired1, why1 = _detector_fires(_detect_quarantine, _drive_pwc_run_sieve_pass, s1)
    assert fired1, "M3: removing the run_sieve_pass guard was not detected"
    notes.append(f"run_sieve_pass -> {why1}")

    # The guard function itself: neutering it must also be caught, so the gate
    # cannot pass on the mere presence of a call that does nothing.
    def _detect_via_guard():
        pwc_mod = sys.modules["persistent_worker_coordinator"]
        real = pwc_mod.assert_pwc_hybrid_not_quarantined
        pwc_mod.assert_pwc_hybrid_not_quarantined = lambda *a, **k: None
        try:
            _drive_pwc_both_modes_trial(hybrid_raises=True)
        finally:
            pwc_mod.assert_pwc_hybrid_not_quarantined = real
        raise AssertionError("a both-mode PWC trial completed with the guard neutered")

    fired2, why2 = _detector_fires(_detect_via_guard)
    assert fired2, "M3: neutering the guard body was not detected"
    notes.append(f"guard body neutered -> {why2}")

    return ("mutated run_sieve_pass executed and reached dispatch; neutered-guard "
            "both-mode trial ran to completion",
            " | ".join(notes),
            "applies-once verified at the call site; guard body swapped once")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

GATES = [
    ("G-ROUTE-A", gate_route_a),
    ("G-ROUTE-B", gate_route_b),
    ("G-KERNEL", gate_kernel),
    ("G-MINER-UNCHANGED", gate_miner_unchanged),
    ("G-PWC-HYBRID", gate_pwc_hybrid),
]

MUTANTS = [
    ("M1 restore bounds.default_* in test_config -> G-ROUTE-A", mutant_m1_route_a),
    ("M2 restore _local_bounds override in _local_test -> G-ROUTE-B", mutant_m2_route_b),
    ("M3 remove the PWC hybrid quarantine guard -> G-PWC-HYBRID", mutant_m3_pwc_quarantine),
]


def _p(*args) -> None:
    """print + flush. sieve_gpu_worker.py:44 REPLACES sys.stdout with a fresh
    file object on the same fd at import time, which discards whatever the
    original stdout still had buffered. G-KERNEL imports it mid-run, so anything
    printed before that and not yet flushed would vanish from the transcript —
    a verification report losing its own earlier lines (VIR-1)."""
    print(*args, flush=True)


def main() -> int:
    _p("=" * 78)
    _p("S172 THRESHOLD PROPAGATION — acceptance gates")
    _p(f"repo: {_ROOT}")
    _p(f"oracles (hand-transcribed): forward={ORACLE_FORWARD} reverse={ORACLE_REVERSE}")
    _p("=" * 78)

    unavailable = 0
    for name, fn in GATES:
        try:
            detail = fn()
            _results.append((name, "PASS", detail))
            _p(f"[{_PASS}] {name}: {detail}")
        except AssertionError as exc:
            msg = str(exc)
            if msg.startswith("UNAVAILABLE:"):
                unavailable += 1
                _results.append((name, "UNAVAILABLE", msg))
                _p(f"[{_UNAV}] {name}: {msg}")
            else:
                _results.append((name, "FAIL", msg))
                _p(f"[{_FAIL}] {name}: {msg}")
        except Exception:
            _results.append((name, "FAIL", traceback.format_exc(limit=4)))
            _p(f"[{_FAIL}] {name}:\n{traceback.format_exc(limit=4)}")

    _p("\n" + "-" * 78)
    _p("MUTANTS (four-part kill rule)")
    _p("-" * 78)
    mutant_ok = True
    for name, fn in MUTANTS:
        try:
            executed, fired, once = fn()
            _MUTANTS.append((name, once, "clean run passed above", fired))
            _p(f"[{_PASS}] {name}")
            _p(f"        applies-once : {once}")
            _p(f"        executed     : {executed}")
            _p(f"        detector     : FIRED — {fired}")
        except Exception as exc:
            mutant_ok = False
            _MUTANTS.append((name, "?", "?", f"MUTANT FAILED: {exc}"))
            _p(f"[{_FAIL}] {name}: {exc}")

    passed = sum(1 for _, s, _ in _results if s == "PASS")
    failed = sum(1 for _, s, _ in _results if s == "FAIL")
    _p("\n" + "=" * 78)
    _p(f"GATES {passed}/{len(GATES)} PASS · {failed} FAIL · {unavailable} UNAVAILABLE")
    _p(f"MUTANTS {len(MUTANTS) - sum(1 for m in _MUTANTS if m[3].startswith('MUTANT FAILED'))}"
          f"/{len(MUTANTS)} killed")

    if failed:
        verdict = "FAIL"
    elif unavailable:
        verdict = "INCOMPLETE"          # VIR-3/VIR-5: unobservable is not clean
    elif not mutant_ok:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    _p(f"COMPLETION SENTINEL: {verdict}")
    _p("=" * 78)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
