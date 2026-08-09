#!/usr/bin/env python3
"""
test_seed_domain_cursor_amendment.py — the seven gates of Beta §10.

Authority: Team Beta ruling *"S145 / SEED-DOMAIN SWEEP TERMINUS AND COVERAGE
AUTHORITY"* (2026-08-07), implemented per
`docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_AMENDMENT.md`.

WHY THIS FILE IS SEPARATE FROM THE STAGING SUITES
-------------------------------------------------
Beta: *"Do not merge their production code into one mega-patch… They have
different authorities, different test suites and different rollback surfaces."*
This module imports nothing from `test_s172_staging_backpressure.py`,
`test_s172_staging_partb.py` or `test_s172_phase4_coordinator.py`, constructs
its own temporary databases, and touches no staging-capacity surface.

WHAT IS UNDER TEST — AND WHAT THAT MEANS FOR VACUITY
----------------------------------------------------
Four gates drive PRODUCTION surfaces that already existed at the pre-amendment
base (`database_system.DistributedPRNGDatabase.get_next_seed_start`,
`agents.watcher_agent.WatcherAgent.run_step`). Those gates therefore go red at
base for a BEHAVIOURAL reason — the old `MAX(seed_range_end)` rule returns the
wrong number — not merely because a new module is missing. The remaining arms
exercise capability that does not exist at base and red on its absence; each
says so explicitly rather than reporting a bare ImportError.

Every gate that drives `run_step` carries a CLEAN CONTROL (an in-domain request
that must reach dispatch) alongside its FAULT INJECTION (an out-of-domain
request that must not). Without the clean control, "no dispatch happened" is
equally consistent with "the whole path is dead", which is the VIR-2 vacuous
class.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 -u tests/test_seed_domain_cursor_amendment.py \
        | tee /tmp/seed_domain_cursor.log
"""
import ast
import contextlib
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"

_results = []

# The governed terminus, restated ONLY here in the test as an independent
# oracle. Production must never restate it (Beta §1, §7); an arm below proves
# the production module imports it rather than defining its own.
_TERMINUS = 2 ** 32

# The legacy tracker's real recorded history (Beta §2): 15 rows, java_lcg /
# bidirectional, max end 16,106,127,360, with the ~1.07-billion-seed hole at
# [1,000, 1,073,741,824).
_LEGACY_MAX_END = 16_106_127_360
_LEGACY_ROWS = [
    (0, 1_000),
    (1_073_741_824, 2_147_483_648),
    (2_147_483_648, 3_221_225_472),
    (3_221_225_472, 4_294_967_296),
] + [
    (4_294_967_296 + i * 1_073_741_824, 4_294_967_296 + (i + 1) * 1_073_741_824)
    for i in range(11)
]

try:
    from utils.seed_coverage_ledger import (
        COVERAGE_LEDGER_TABLE,
        CURSOR_STATUS_COMPLETE,
        CURSOR_STATUS_OPEN,
        canonical_coverage_identity,
        CoverageLedger,
        CoverageLedgerError,
        LedgerIntegrityError,
        PublicationBindingError,
        SeedDomainPreflightError,
        assert_seed_domain_preflight,
        first_uncovered_seed,
        normalize_certified_intervals,
    )
    _AMENDMENT_IMPORT_ERROR = None
except Exception as _exc:                                     # noqa: BLE001
    _AMENDMENT_IMPORT_ERROR = _exc


def _require_amendment():
    """Red with a diagnosis, never a bare ImportError."""
    if _AMENDMENT_IMPORT_ERROR is not None:
        raise AssertionError(
            f"the seed-domain amendment is not present on this tree: "
            f"`utils.seed_coverage_ledger` did not import "
            f"({type(_AMENDMENT_IMPORT_ERROR).__name__}: "
            f"{_AMENDMENT_IMPORT_ERROR}). The certified cursor, the coverage "
            f"ledger and the pre-dispatch domain wall therefore do not exist."
        )


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception:                                          # noqa: BLE001
        tb = traceback.format_exc()
        _results.append((name, False, tb))
        print(f"  [{_FAIL}] {name}")


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


@contextlib.contextmanager
def _tmpdb():
    d = tempfile.mkdtemp(prefix="s145_gate_")
    try:
        yield os.path.join(d, "prng_analysis.db")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _seed_legacy_tracker(db_path, rows=None, prng_type="java_lcg"):
    """Populate `exhaustive_progress` exactly as the legacy writer does.

    Uses `DistributedPRNGDatabase.update_exhaustive_progress` — the real
    production writer, INSERT OR REPLACE and all — so the gate measures the
    legacy table as it actually behaves rather than a reconstruction.
    """
    from database_system import DistributedPRNGDatabase
    db = DistributedPRNGDatabase(db_path)
    for start, end in (rows if rows is not None else _LEGACY_ROWS):
        db.update_exhaustive_progress(
            search_id=f"step1_{prng_type}_{start}",
            prng_type=prng_type,
            mapping_type="bidirectional",
            seed_range_start=start,
            seed_range_end=end,
            seeds_completed=end - start,
            best_score=0.0,
            best_seed=None,
        )
    return db


class _FakeArtifact:
    """The pre-R1 stand-in: a bare object exposing `artifact_sha256`.

    R1 Blocker A made this a NEGATIVE fixture. It used to be accepted as a
    publication witness, which is the production API bypass Beta found: holding
    a digest is not evidence that a canonical publication occurred. Every arm
    below now requires it to be REFUSED.
    """

    def __init__(self, artifact_sha256, generation_id="gen-test"):
        self.artifact_sha256 = artifact_sha256
        self.generation_id = generation_id


def _real_artifact(*, start, count, digest, run_id="run-1",
                   prng_base="java_lcg", modes=("constant",),
                   generation_id="gen-test", repository_commit="4dd5535"):
    """A genuine `RunArtifactResult` — the only thing that may certify coverage.

    Constructed positionally-complete from the frozen dataclass so that a field
    added to the contract breaks this helper loudly rather than silently
    producing a witness that no longer matches.
    """
    from pathlib import Path

    from utils.run_finalizer import RunArtifactResult
    return RunArtifactResult(
        generation_id=generation_id,
        generation_dir=Path("/tmp/s145/gen"),
        all_npz_path=Path("/tmp/s145/all.npz"),
        binary_npz_path=Path("/tmp/s145/bin.npz"),
        sidecar_path=Path("/tmp/s145/sidecar.json"),
        artifact_sha256=digest,
        sidecar_sha256="e" * 64,
        parent_generation_id=None,
        parent_artifact_sha256=None,
        parent_sidecar_sha256=None,
        repository_commit=repository_commit,
        repository_tree_clean=True,
        artifact_schema_version="1.0",
        sidecar_schema_version="1.0",
        encoding_contract_version="3.2",
        canonical_map_hash="c" * 64,
        run_id=run_id,
        prng_base=prng_base,
        skip_modes_executed=tuple(modes),
        seed_start=start,
        seed_count=count,
        seed_end_exclusive=start + count,
        raw_candidate_count=1,
        l2_winner_count=1,
        prior_row_count=0,
        final_row_count=1,
        created_at="2026-08-09T00:00:00+00:00",
        elapsed_seconds=1.0,
    )


def _record(ledger, start, count, *, run_id="run-1", digest=None,
            prng_base="java_lcg", modes=("constant",)):
    """Record coverage through THE ONE certification door."""
    return ledger.record_publication(
        _real_artifact(start=start, count=count, run_id=run_id,
                       digest=digest or f"{start:064x}",
                       prng_base=prng_base, modes=modes),
        dataset_sha256="d" * 64,
    )


# ---------------------------------------------------------------------------
# WATCHER harness. Stubs ONLY the upstream environmental gates that stand
# between `run_step`'s entry and the S145 block, and tripwires dispatch.
#
# Each stub is an unrelated gate that fails in this environment for reasons
# that have nothing to do with the seed domain (the rigs are unreachable from
# VM101 right now). Stubbing them is what makes the measurement possible; the
# tripwire is the measurement itself, and it is never stubbed.
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _watcher_harness():
    import agents.watcher_agent as wa
    import database_system as ds
    from miner import dataset_authority as dsauth

    fired = []

    # DB ISOLATION. The WATCHER block constructs `DistributedPRNGDatabase()`
    # with no arguments, which resolves cwd-relative to the LIVE
    # `prng_analysis.db`. The pre-R1 suite therefore created a real
    # `certified_coverage` table in the production database as a side effect of
    # running its gates. A test that mutates the artifact it is auditing is not
    # a test; every instance the harness produces now points at a temp file.
    _db_dir = tempfile.mkdtemp(prefix="s145_watcher_db_")
    _RealDB = ds.DistributedPRNGDatabase

    class _IsolatedDB(_RealDB):
        def __init__(self, db_path=None):
            super().__init__(os.path.join(_db_dir, "prng_analysis.db"))

    class _Tripwire:
        TimeoutExpired = wa.subprocess.TimeoutExpired

        def __getattr__(self, name):
            def boom(*a, **k):
                fired.append(name)
                raise _DispatchAttempted(f"subprocess.{name} was called")
            return boom

    saved = (
        wa.subprocess, wa.check_output_freshness,
        wa.WatcherAgent._run_preflight_check,
        dsauth.fleet_preflight, dsauth.write_run_provenance,
        ds.DistributedPRNGDatabase,
    )
    wa.subprocess = _Tripwire()
    wa.check_output_freshness = lambda step: (False, "harness: forced stale", False)
    wa.WatcherAgent._run_preflight_check = lambda self, step: (True, "harness: stubbed")
    dsauth.fleet_preflight = lambda *a, **k: []
    dsauth.write_run_provenance = lambda *a, **k: None
    ds.DistributedPRNGDatabase = _IsolatedDB
    try:
        yield wa, fired
    finally:
        (wa.subprocess, wa.check_output_freshness,
         wa.WatcherAgent._run_preflight_check,
         dsauth.fleet_preflight, dsauth.write_run_provenance,
         ds.DistributedPRNGDatabase) = saved
        shutil.rmtree(_db_dir, ignore_errors=True)


class _DispatchAttempted(BaseException):
    """Raised by the tripwire the instant anything tries to spawn work.

    Derives from BaseException, NOT Exception, and that is load-bearing:
    `run_step` wraps its dispatch in `except Exception` and converts anything
    it catches into `{'success': False, 'error': ...}`. A tripwire deriving
    from Exception is swallowed there and the gate reads "no dispatch" for a
    run that dispatched — the measurement would invert.
    """


def _run_step1(wa, params):
    """Drive the real `run_step(1, ...)`. Returns (result_dict, dispatched)."""
    agent = wa.WatcherAgent()
    try:
        return agent.run_step(1, dict(params)), False
    except _DispatchAttempted:
        return None, True


# ===========================================================================
# G-DOMAIN-PREFLIGHT
# ===========================================================================
def arm_preflight_matrix():
    """Beta §10's five cases, against the production wall."""
    _require_amendment()
    _assert(assert_seed_domain_preflight(0, 1) == (0, 1, 1),
            "start=0,count=1 must PASS")
    _assert(assert_seed_domain_preflight(0, _TERMINUS)[2] == _TERMINUS,
            "an interval ending exactly at 2^32 must PASS")
    _assert(assert_seed_domain_preflight(_TERMINUS - 1, 1)[2] == _TERMINUS,
            "the final seed must PASS")

    for start, count, why in [
        (_TERMINUS, 1, "start exactly at the terminus"),
        (_TERMINUS + 1, 1, "start beyond the terminus"),
        (0, _TERMINUS + 1, "end at 2^32+1"),
        (_TERMINUS - 1, 2, "an interval straddling the terminus"),
        (-1, 10, "negative start"),
        (0, 0, "zero count"),
        (0, -5, "negative count"),
    ]:
        try:
            assert_seed_domain_preflight(start, count)
            raise AssertionError(f"{why}: [{start},{start+count}) was ACCEPTED")
        except SeedDomainPreflightError as e:
            _assert("seed_domain_preflight" in str(e),
                    f"{why}: reason string must identify the governed contract, got {e}")


def arm_preflight_rejects_non_int():
    """True is an int in Python; a bool seed_start must not sail through."""
    _require_amendment()
    for bad in (True, False, 1.0, "0", None):
        try:
            assert_seed_domain_preflight(bad, 10)
            raise AssertionError(f"seed_start={bad!r} was accepted")
        except SeedDomainPreflightError:
            pass


def arm_preflight_precedes_dispatch_in_live_source():
    """AST, not text: the refusal returns strictly before any Popen in run_step.

    `2389b61` reverted a fix by whole-block replacement, so a text anchor can go
    green against source that no longer does the thing. This extracts the live
    AST of `run_step` and compares line numbers.
    """
    _require_amendment()
    src = open(os.path.join(_ROOT, "agents", "watcher_agent.py"),
               encoding="utf-8").read()
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    _assert("run_step" in funcs, "run_step not found in agents/watcher_agent.py")

    # `run_step` does not call Popen itself — it delegates to
    # `_run_step_streaming`, which owns the spawn. Locate the spawn owner from
    # the live AST rather than assuming either shape.
    spawn_owners = [
        name for name, fn in funcs.items()
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "Popen" for n in ast.walk(fn))
    ]
    _assert(spawn_owners,
            "no subprocess.Popen call found anywhere in watcher_agent.py — the "
            "dispatch point this gate orders against has moved")

    run_step = funcs["run_step"]
    dispatch_lines, refusal_lines = [], []
    for node in ast.walk(run_step):
        if isinstance(node, ast.Call):
            fn = node.func
            # a direct spawn, or the call that delegates to the spawn owner
            if isinstance(fn, ast.Attribute) and (
                    fn.attr == "Popen" or fn.attr in spawn_owners):
                dispatch_lines.append(node.lineno)
        if isinstance(node, ast.Constant) and node.value == "seed_domain_preflight":
            refusal_lines.append(node.lineno)

    _assert(dispatch_lines,
            f"run_step contains no call to a spawn owner ({spawn_owners}) — the "
            f"dispatch point this gate orders against has moved")
    _assert(refusal_lines, "no 'seed_domain_preflight' refusal found inside "
                           "run_step — the pre-dispatch wall is absent")
    _assert(max(refusal_lines) < min(dispatch_lines),
            f"the domain refusal at line {max(refusal_lines)} does not precede "
            f"dispatch at line {min(dispatch_lines)}")


def arm_preflight_zero_dispatch_executed():
    """Fault injection AND clean control, both executed against run_step."""
    _require_amendment()
    with _watcher_harness() as (wa, fired):
        # FAULT INJECTION: an out-of-domain request must be refused, and
        # nothing may be dispatched.
        result, dispatched = _run_step1(
            wa, {"seed_start": _TERMINUS, "max_seeds": 1000, "prng_type": "java_lcg"})
        _assert(not dispatched, "an out-of-domain request REACHED DISPATCH")
        _assert(result is not None and result.get("blocked_by") == "seed_domain_preflight",
                f"expected blocked_by='seed_domain_preflight', got "
                f"{result.get('blocked_by')!r} (error={str(result.get('error'))[:160]!r})")
        _assert(fired == [], f"dispatch was attempted: {fired}")

        # CLEAN CONTROL: an in-domain request must get PAST the wall and reach
        # dispatch. Without this, "nothing dispatched" would also be satisfied
        # by a dead path or a blanket block.
        fired.clear()
        result, dispatched = _run_step1(
            wa, {"seed_start": 0, "max_seeds": 1000, "prng_type": "java_lcg"})
        _assert(dispatched,
                f"clean control did NOT reach dispatch — the measurement is "
                f"vacuous. blocked_by={None if result is None else result.get('blocked_by')!r}")
        _assert(fired == ["Popen"], f"unexpected dispatch surface: {fired}")


def arm_cli_entry_point_refuses_before_dispatch():
    """The OTHER entry point: `run_bayesian_optimization` refuses out-of-domain.

    The lottery file passed here does not exist. That is the discriminator: if
    the wall did NOT fire, the function would proceed and fail on the missing
    file (or on the integration/fleet path) with some other exception. Getting
    SeedDomainPreflightError proves the refusal happened FIRST.

    There is deliberately no executed clean control on this arm — an in-domain
    call to this function proceeds toward real fleet work, and this suite is
    forbidden from launching anything. The ordering proof below is structural
    instead.
    """
    _require_amendment()
    import window_optimizer as wo
    try:
        wo.run_bayesian_optimization(
            lottery_file="/nonexistent/s145_gate_no_such_file.json",
            trials=1, output_config="/dev/null",
            seed_count=1000, seed_start=_TERMINUS, prng_type="java_lcg")
        raise AssertionError(
            "the CLI entry point ACCEPTED an out-of-domain seed_start")
    except SeedDomainPreflightError:
        pass


def arm_cli_wall_precedes_everything_in_its_function():
    """Structural: no call of any kind precedes the wall in that function."""
    _require_amendment()
    src = open(os.path.join(_ROOT, "window_optimizer.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    func = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)
                 and n.name == "run_bayesian_optimization"), None)
    _assert(func is not None, "run_bayesian_optimization not found")

    wall_lines = [n.lineno for n in ast.walk(func)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "_s145_domain_wall"]
    _assert(wall_lines, "the CLI pre-dispatch wall is absent from "
                        "run_bayesian_optimization")
    wall = min(wall_lines)
    earlier = sorted({n.lineno for n in ast.walk(func)
                      if isinstance(n, ast.Call) and n.lineno < wall})
    _assert(not earlier,
            f"calls execute before the seed-domain wall at line {wall}: {earlier}")


def arm_publication_binding_is_wired_after_finalize():
    """The ledger has a live producer, and it sits AFTER publication."""
    _require_amendment()
    path = os.path.join(_ROOT, "window_optimizer_integration_final.py")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    finalize_lines, record_lines = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "_finalize_run_d3_5":
                finalize_lines.append(node.lineno)
            if isinstance(fn, ast.Attribute) and fn.attr == "record_certified_coverage":
                record_lines.append(node.lineno)

    _assert(finalize_lines, "no call to _finalize_run_d3_5 found — the "
                            "publication site this gate orders against has moved")
    _assert(record_lines,
            "the coverage ledger has NO live producer: nothing calls "
            "record_certified_coverage. The certified cursor would stay at 0 "
            "forever and every run would restart at seed 0.")
    _assert(len(record_lines) == 1,
            f"certified coverage must have exactly ONE producer, found "
            f"{len(record_lines)} at lines {record_lines}")
    _assert(record_lines[0] > finalize_lines[0],
            f"the coverage record at line {record_lines[0]} does not follow the "
            f"publication at line {finalize_lines[0]} — coverage would be "
            f"claimed before the artifact exists")


def arm_domain_constant_is_shared_not_restated():
    """The wall and the finalizer read ONE constant (Beta §1, §7)."""
    _require_amendment()
    import utils.run_finalizer as rf
    import utils.seed_coverage_ledger as scl
    _assert(scl.SEED_DOMAIN_EXCLUSIVE_MAX == rf.SEED_DOMAIN_EXCLUSIVE_MAX == _TERMINUS,
            "the ledger and the finalizer disagree on the terminus")

    # Structural: the module must IMPORT the name, never assign it.
    src = open(os.path.join(_ROOT, "utils", "seed_coverage_ledger.py"),
               encoding="utf-8").read()
    tree = ast.parse(src)
    imported_from = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "SEED_DOMAIN_EXCLUSIVE_MAX":
                    imported_from = node.module
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "SEED_DOMAIN_EXCLUSIVE_MAX":
                    raise AssertionError(
                        "utils/seed_coverage_ledger.py ASSIGNS "
                        "SEED_DOMAIN_EXCLUSIVE_MAX — that is a second domain "
                        "constant, which Beta §7 forbids")
    _assert(imported_from == "utils.run_finalizer",
            f"SEED_DOMAIN_EXCLUSIVE_MAX must be imported from "
            f"utils.run_finalizer, got {imported_from!r}")


def arm_boundary_mutation_reds_both_walls():
    """§1: moving the boundary must red the pre-dispatch gate AND the finalizer.

    Executed in a SUBPROCESS that mutates `run_finalizer`'s constant BEFORE the
    ledger imports it, which is the only way to prove the two walls move
    together from a single edit — an in-process monkeypatch cannot, because the
    ledger's `from ... import` binds the value at import time.
    """
    _require_amendment()
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {_ROOT!r})
        import utils.run_finalizer as rf
        rf.SEED_DOMAIN_EXCLUSIVE_MAX = 2 ** 31          # THE MUTATION
        import utils.seed_coverage_ledger as scl

        # 1. the pre-dispatch wall must now refuse what it used to accept
        try:
            scl.assert_seed_domain_preflight(0, 2 ** 32)
            print("PREFLIGHT_SURVIVED")
        except scl.SeedDomainPreflightError:
            print("PREFLIGHT_RED")

        # 2. the finalizer's own parity gate must refuse it too
        try:
            rf._validate_declared_coverage(0, 2 ** 32)
            print("FINALIZER_SURVIVED")
        except rf.CoverageValidationError:
            print("FINALIZER_RED")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=180, cwd=_ROOT)
    _assert("PREFLIGHT_RED" in out.stdout,
            f"the pre-dispatch wall SURVIVED the boundary mutation: {out.stdout}{out.stderr}")
    _assert("FINALIZER_RED" in out.stdout,
            f"the finalizer parity gate SURVIVED the boundary mutation: {out.stdout}{out.stderr}")


# ===========================================================================
# G-CURSOR-FIRST-GAP
# ===========================================================================
def arm_first_gap_beta_worked_example():
    """Certified [0,1000) and [2^30,2^31) => 1000, NOT 2^31."""
    _require_amendment()
    r = first_uncovered_seed([(0, 1000), (2 ** 30, 2 ** 31)])
    _assert(r.status == CURSOR_STATUS_OPEN, f"expected OPEN, got {r.status}")
    _assert(r.next_seed_start == 1000,
            f"Beta's worked example requires 1000, got {r.next_seed_start}")


def arm_first_gap_through_production_db():
    """The same law through `DistributedPRNGDatabase.get_next_seed_start`.

    The FIRST assertion is deliberately independent of the new module, so this
    gate reds at the pre-amendment base for a BEHAVIOURAL reason — the old
    `MAX(seed_range_end)` rule answers 2^31 and declares the ~1.07-billion-seed
    hole covered — rather than merely because a module is missing.
    """
    from database_system import DistributedPRNGDatabase
    with _tmpdb() as path:
        legacy = _seed_legacy_tracker(path, rows=[(0, 1000), (2 ** 30, 2 ** 31)])
        got = legacy.get_next_seed_start("java_lcg", 5_000_000, test_both_modes=False)
        _assert(got != 2 ** 31,
                f"get_next_seed_start returned {got:,} — that is "
                f"MAX(seed_range_end) over the LEGACY tracker, the rule Beta §6 "
                f"invalidated. It skips the ~1.07-billion-seed hole at "
                f"[1,000, 1,073,741,824) and declares it covered.")

    _require_amendment()
    with _tmpdb() as path:
        db = DistributedPRNGDatabase(path)
        ledger = CoverageLedger(path)
        _record(ledger, 0, 1000, run_id="r-a", digest="a" * 64)
        _record(ledger, 2 ** 30, 2 ** 30, run_id="r-b", digest="b" * 64)
        got = db.get_next_seed_start("java_lcg", 5_000_000, test_both_modes=False)
        _assert(got == 1000, f"production cursor returned {got}, expected 1000")


def arm_first_gap_interior_hole_and_ordering():
    """Insertion order must not matter; adjacency must not fake a gap."""
    _require_amendment()
    a = first_uncovered_seed([(2 ** 30, 2 ** 31), (0, 1000)]).next_seed_start
    b = first_uncovered_seed([(0, 1000), (2 ** 30, 2 ** 31)]).next_seed_start
    _assert(a == b == 1000, f"order-dependent cursor: {a} vs {b}")
    _assert(first_uncovered_seed([(0, 10), (10, 20)]).next_seed_start == 20,
            "adjacent intervals must merge — [0,10) and [10,20) leave no gap")
    _assert(first_uncovered_seed([(0, 100), (50, 200)]).next_seed_start == 200,
            "overlapping intervals must merge")
    _assert(first_uncovered_seed([]).next_seed_start == 0,
            "no certified coverage means the cursor is 0")


def arm_first_gap_mutation_max_end_is_caught():
    """MUTANT: reinstate `MAX(seed_range_end)` and prove this gate reds.

    The mutant is the exact rule Beta invalidated. If the oracle above could
    not tell the two apart, the gate would be decorative.
    """
    _require_amendment()
    intervals = [(0, 1000), (2 ** 30, 2 ** 31)]
    mutant_answer = max(end for _, end in intervals)      # the old rule
    correct_answer = first_uncovered_seed(intervals).next_seed_start
    _assert(mutant_answer == 2 ** 31,
            "the mutant did not reproduce the old rule")
    _assert(correct_answer != mutant_answer,
            "the gate cannot distinguish first-gap from MAX(seed_range_end)")
    _assert(correct_answer == 1000, f"got {correct_answer}")


# ===========================================================================
# G-CURSOR-COMPLETE
# ===========================================================================
def arm_complete_returns_no_number():
    _require_amendment()
    r = first_uncovered_seed([(0, _TERMINUS)])
    _assert(r.status == CURSOR_STATUS_COMPLETE, f"expected COMPLETE, got {r.status}")
    _assert(r.next_seed_start is None,
            f"COMPLETE must carry NO numeric next seed, got {r.next_seed_start}")
    _assert(r.is_complete and r.remaining_seed_count == 0, "remaining must be 0")

    pieced = first_uncovered_seed([(0, 2 ** 31), (2 ** 31, _TERMINUS)])
    _assert(pieced.status == CURSOR_STATUS_COMPLETE,
            "a union assembled from two halves must also be COMPLETE")

    almost = first_uncovered_seed([(0, _TERMINUS - 1)])
    _assert(almost.status == CURSOR_STATUS_OPEN
            and almost.next_seed_start == _TERMINUS - 1,
            "one seed short of the terminus is OPEN, not COMPLETE")


def arm_complete_production_db_returns_none():
    _require_amendment()
    from database_system import DistributedPRNGDatabase
    with _tmpdb() as path:
        db = DistributedPRNGDatabase(path)
        ledger = CoverageLedger(path)
        _record(ledger, 0, 2 ** 31, run_id="h1", digest="a" * 64)
        _record(ledger, 2 ** 31, 2 ** 31, run_id="h2", digest="b" * 64)
        got = db.get_next_seed_start("java_lcg", 5_000_000, test_both_modes=False)
        _assert(got is None,
                f"a COMPLETE domain must return None, got {got!r} — Beta §6: "
                f"'There is no 4,294,967,296 next run.'")


def arm_complete_generates_no_watcher_run():
    """COMPLETE => no run generated, and nothing dispatched."""
    _require_amendment()
    import database_system as ds
    from utils.seed_coverage_ledger import CursorResult

    complete = CursorResult(
        status=CURSOR_STATUS_COMPLETE, next_seed_start=None,
        domain_start=0, domain_end_exclusive=_TERMINUS,
        covered_seed_count=_TERMINUS, certified_interval_count=1,
    )
    saved = ds.DistributedPRNGDatabase.get_certified_cursor
    ds.DistributedPRNGDatabase.get_certified_cursor = (
        lambda self, p, **kw: complete)
    try:
        with _watcher_harness() as (wa, fired):
            result, dispatched = _run_step1(
                wa, {"seed_start": 0, "max_seeds": 1000, "prng_type": "java_lcg"})
            _assert(not dispatched, "a COMPLETE domain REACHED DISPATCH")
            _assert(result is not None
                    and result.get("blocked_by") == "seed_domain_complete",
                    f"expected blocked_by='seed_domain_complete', got "
                    f"{None if result is None else result.get('blocked_by')!r}")
            _assert(result.get("seed_domain_complete") is True,
                    "the COMPLETE state must be explicit in the result")
            _assert(result.get("next_seed_start") is None,
                    "COMPLETE must not hand the operator a number")
            _assert(fired == [], f"dispatch was attempted: {fired}")
    finally:
        ds.DistributedPRNGDatabase.get_certified_cursor = saved


# ===========================================================================
# G-LEGACY-NONAUTHORITY
# ===========================================================================
def arm_legacy_history_is_ignored_by_certified_cursor():
    """Populate the old tracker with the real 16.1B history; cursor ignores it."""
    from database_system import DistributedPRNGDatabase
    with _tmpdb() as path:
        db = _seed_legacy_tracker(path)

        # The legacy table really is populated, and really does carry the
        # out-of-domain extent — this is the positive control for the gate.
        rows = db.get_exhaustive_progress("step1_java_lcg_0")
        _assert(rows, "legacy seeding did not write anything")
        with sqlite3.connect(path) as conn:
            legacy_max = conn.execute(
                "SELECT MAX(seed_range_end) FROM exhaustive_progress "
                "WHERE prng_type='java_lcg'").fetchone()[0]
        _assert(legacy_max == _LEGACY_MAX_END,
                f"legacy fixture max end {legacy_max} != {_LEGACY_MAX_END}")

        # BEHAVIOURAL, and independent of the new module: whatever the cursor
        # answers, it may never be the legacy extent. At the pre-amendment base
        # this reds with 16,106,127,360 — 11,811,160,064 seeds BEYOND the
        # terminus Beta forbade any run from beginning at.
        got = DistributedPRNGDatabase(path).get_next_seed_start("java_lcg", 5_000_000, test_both_modes=False)
        _assert(got is None or got < _TERMINUS,
                f"the cursor returned {got:,}, which is beyond the terminus "
                f"{_TERMINUS:,} by {got - _TERMINUS:,} seeds. Beta §1: 'No run "
                f"may begin at 2^32, cross 2^32, or publish a candidate outside "
                f"that interval.'")
        _assert(got != _LEGACY_MAX_END,
                f"the cursor returned the legacy MAX(seed_range_end) {got:,}")

        _require_amendment()
        _assert(got == 0,
                f"the certified cursor returned {got!r}; it must ignore the "
                f"legacy tracker COMPLETELY and start the certified stream at "
                f"zero (Beta §§2-4: rows 1-4 are not certified coverage either)")


def arm_legacy_rows_are_not_deleted_or_rewritten():
    """Beta: retain and display; never fold, renumber or delete."""
    _require_amendment()
    from database_system import DistributedPRNGDatabase
    with _tmpdb() as path:
        db = _seed_legacy_tracker(path)
        ledger = CoverageLedger(path)
        _record(ledger, 0, 1_000_000, run_id="new-1", digest="a" * 64)
        db.get_next_seed_start("java_lcg", 5_000_000, test_both_modes=False)      # exercise the read path

        with sqlite3.connect(path) as conn:
            after = conn.execute(
                "SELECT seed_range_start, seed_range_end FROM exhaustive_progress "
                "ORDER BY seed_range_start").fetchall()
        _assert(len(after) == len(_LEGACY_ROWS),
                f"legacy row count changed: {len(after)} != {len(_LEGACY_ROWS)}")
        _assert([tuple(r) for r in after] == sorted(_LEGACY_ROWS),
                "legacy row VALUES were rewritten")
        _assert(any(e > _TERMINUS for _, e in after),
                "the out-of-domain rows were folded back into [0,2^32)")


def arm_legacy_table_is_never_read_by_the_ledger():
    """Structural: the ledger module never names the legacy table."""
    _require_amendment()
    src = open(os.path.join(_ROOT, "utils", "seed_coverage_ledger.py"),
               encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            _assert("exhaustive_progress" not in node.value.upper().lower()
                    or "FROM exhaustive_progress" not in node.value,
                    "the ledger module contains SQL against exhaustive_progress")


# ===========================================================================
# G-NO-REPLACE-CLOBBER
# ===========================================================================
def arm_smoke_run_cannot_replace_a_certified_interval():
    """The exact incident: a 1,000-seed run at zero must not erase a big one."""
    _require_amendment()
    with _tmpdb() as path:
        ledger = CoverageLedger(path)
        big = _record(ledger, 0, 1_000_000_000, run_id="production", digest="a" * 64)
        small = _record(ledger, 0, 1_000, run_id="smoke-test", digest="b" * 64)

        _assert(big.coverage_id != small.coverage_id,
                "two intervals starting at 0 collapsed to one identity")
        rows = ledger.certified_records("java_lcg", {"constant"})
        _assert(len(rows) == 2, f"expected both records, found {len(rows)}")
        _assert(ledger.get(big.coverage_id) is not None,
                "the billion-seed production interval was erased by a smoke test")
        _assert(ledger.certified_cursor("java_lcg", {"constant"}).next_seed_start == 1_000_000_000,
                "coverage regressed after the smoke test")


def arm_legacy_writer_really_does_clobber():
    """Fault-injection control: prove the defect EXISTS in the legacy table.

    Without this, "the ledger did not clobber" is unfalsifiable — it could just
    mean the scenario never clobbers anything anywhere.
    """
    from database_system import DistributedPRNGDatabase
    with _tmpdb() as path:
        db = DistributedPRNGDatabase(path)
        db.update_exhaustive_progress(
            search_id="step1_java_lcg_0", prng_type="java_lcg",
            mapping_type="bidirectional", seed_range_start=0,
            seed_range_end=1_000_000_000, seeds_completed=1_000_000_000)
        db.update_exhaustive_progress(
            search_id="step1_java_lcg_0", prng_type="java_lcg",
            mapping_type="bidirectional", seed_range_start=0,
            seed_range_end=1_000, seeds_completed=1_000)
        with sqlite3.connect(path) as conn:
            ends = [r[0] for r in conn.execute(
                "SELECT seed_range_end FROM exhaustive_progress").fetchall()]
        _assert(ends == [1_000],
                f"expected the legacy INSERT OR REPLACE to destroy the "
                f"billion-seed row, got {ends} — the control is not reproducing "
                f"the incident this gate exists for")


# --- R2: the append-only fixture --------------------------------------------
#
# The pre-R2 version of the arm below hardcoded the PRE-R1 column list
# (`prng_type` / `mapping_mode`) and caught any `sqlite3.Error` as success. The
# path it actually observed was
#
#     INSERT OR REPLACE -> "no column named prng_type" -> sqlite3.Error -> PASS
#
# not the path it claimed to prove. Two structural changes stop that class of
# false green recurring:
#
#   1. the row is built as a {column: value} MAPPING and reconciled against
#      `PRAGMA table_info` at run time, so a schema change reds this arm with
#      "fixture no longer matches the live schema" instead of silently passing
#      on a bad-column error;
#   2. every arm asserts the REASON, never merely "some SQLite error".
# ----------------------------------------------------------------------------
_APPEND_ONLY_REASON = "certified_coverage is append-only"


def _valid_replacement_row(coverage_id):
    """A COMPLETELY VALID row that collides on `coverage_id`.

    Every value satisfies its CHECK constraint and column type, so the ONLY
    thing that can stop the statement is the append-only enforcement itself —
    which is exactly what Beta requires the REPLACE arm to isolate.
    """
    return {
        "coverage_id": coverage_id,          # the genuine uniqueness conflict
        "run_id": "smoke-test",
        "study_identity": None,
        "prng_base": "java_lcg",
        "skip_modes_executed": "constant",
        "seed_domain_contract": "v1.1-stratum",
        "seed_start": 0,
        "seed_end_exclusive": 1000,          # the 1,000-seed smoke interval
        "dataset_sha256": "d" * 64,
        "repository_commit": "4dd5535",
        "artifact_sha256": "b" * 64,
        "generation_id": "gen-smoke",
        "publication_status": "CERTIFIED",
        "recorded_at": "2026-08-09T00:00:00+00:00",
    }


def _replace_statement(conn, coverage_id):
    """Build `INSERT OR REPLACE` from the LIVE schema. Returns (sql, params)."""
    live = [r[1] for r in conn.execute("PRAGMA table_info(certified_coverage)")]
    _assert(live, "certified_coverage does not exist")
    row = _valid_replacement_row(coverage_id)
    missing, extra = set(live) - set(row), set(row) - set(live)
    _assert(not missing and not extra,
            f"the append-only fixture no longer matches the live schema "
            f"(missing={sorted(missing)}, extra={sorted(extra)}). Fix the "
            f"fixture — do NOT let a column error stand in for append-only "
            f"enforcement; that is the R2 false green.")
    sql = (f"INSERT OR REPLACE INTO certified_coverage ({', '.join(live)}) "
           f"VALUES ({', '.join('?' for _ in live)})")
    return sql, tuple(row[c] for c in live)


def _assert_append_only_refusal(conn, label, run, expect_verb):
    """Run a mutating statement and require the APPEND-ONLY reason, specifically."""
    try:
        run()
        conn.commit()
        raise AssertionError(f"{label} was ALLOWED against certified_coverage")
    except sqlite3.Error as exc:
        conn.rollback()
        message = str(exc)
        _assert(_APPEND_ONLY_REASON in message,
                f"{label} failed, but NOT for the append-only reason: "
                f"{type(exc).__name__}: {message}. A schema, CHECK-constraint, "
                f"malformed-value or syntax error must never masquerade as "
                f"append-only enforcement.")
        _assert(expect_verb in message,
                f"{label} was refused by the wrong trigger: expected "
                f"{expect_verb!r}, got {message!r}")
        for masquerade in ("no column named", "no such column", "syntax error",
                           "has no column"):
            _assert(masquerade not in message.lower(),
                    f"{label} failed on a {masquerade!r} error: {message}")
        return message


def arm_append_only_is_enforced_by_the_database():
    """UPDATE, DELETE and INSERT OR REPLACE must all ABORT — for the right reason."""
    _require_amendment()
    with _tmpdb() as path:
        ledger = CoverageLedger(path)
        kept = _record(ledger, 0, 1_000_000_000, run_id="production", digest="a" * 64)

        conn = sqlite3.connect(path)
        conn.execute("PRAGMA recursive_triggers = ON")
        try:
            replace_sql, replace_params = _replace_statement(conn, kept.coverage_id)

            reasons = {}
            reasons["DELETE"] = _assert_append_only_refusal(
                conn, "DELETE",
                lambda: conn.execute("DELETE FROM certified_coverage"),
                "DELETE is forbidden")
            reasons["UPDATE"] = _assert_append_only_refusal(
                conn, "UPDATE",
                lambda: conn.execute(
                    "UPDATE certified_coverage SET seed_end_exclusive = 1000"),
                "UPDATE is forbidden")
            # REPLACE satisfies the coverage_id conflict by DELETING the losing
            # row, so it must be refused by the DELETE trigger — that is the
            # specific claim the production `recursive_triggers` pragma exists
            # to make true.
            reasons["REPLACE"] = _assert_append_only_refusal(
                conn, "REPLACE",
                lambda: conn.execute(replace_sql, replace_params),
                "DELETE is forbidden")
        finally:
            conn.close()

        # The record survived UNCHANGED — not merely present.
        survivor = ledger.get(kept.coverage_id)
        _assert(survivor is not None, "the record did not survive")
        _assert(survivor.run_id == "production"
                and survivor.seed_end_exclusive == 1_000_000_000,
                f"the certified row was mutated: run_id={survivor.run_id!r} "
                f"end={survivor.seed_end_exclusive}")
        _assert(len(ledger.certified_records()) == 1, "row count changed")


def arm_recursive_triggers_is_load_bearing():
    """R2 §C — prove `PRAGMA recursive_triggers = ON` is NECESSARY, not decorative.

    The production ledger states that the BEFORE DELETE trigger closes
    `INSERT OR REPLACE` *only* because REPLACE satisfies a conflict by deleting,
    and that SQLite fires trigger-driven deletes only under recursive triggers.
    That is a claim about a pragma, and until this arm existed it was uncertified.

    Same table, same existing `coverage_id`, same fully valid REPLACE — the ONLY
    difference between the two halves is the pragma.
    """
    _require_amendment()

    # --- OFF: the mutant. REPLACE must SUCCEED and actually clobber. ---
    with _tmpdb() as path:
        ledger = CoverageLedger(path)
        kept = _record(ledger, 0, 1_000_000_000, run_id="production", digest="a" * 64)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA recursive_triggers = OFF")
        try:
            sql, params = _replace_statement(conn, kept.coverage_id)
            try:
                conn.execute(sql, params)
                conn.commit()
            except sqlite3.Error as exc:
                raise AssertionError(
                    f"with recursive_triggers OFF the REPLACE was still refused "
                    f"({exc}). The mutant is not live, so the ON half proves "
                    f"nothing about the pragma.") from exc
        finally:
            conn.close()

        clobbered = ledger.get(kept.coverage_id)
        _assert(clobbered is not None, "the row vanished entirely")
        _assert(clobbered.run_id == "smoke-test"
                and clobbered.seed_end_exclusive == 1000,
                f"REPLACE succeeded but did not actually replace the record: "
                f"run_id={clobbered.run_id!r} end={clobbered.seed_end_exclusive}. "
                f"Without a real replacement the mutant does not demonstrate the "
                f"clobber the pragma prevents.")

    # --- ON: production. The identical statement must ABORT. ---
    with _tmpdb() as path:
        ledger = CoverageLedger(path)
        kept = _record(ledger, 0, 1_000_000_000, run_id="production", digest="a" * 64)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA recursive_triggers = ON")
        try:
            sql, params = _replace_statement(conn, kept.coverage_id)
            _assert_append_only_refusal(
                conn, "REPLACE (recursive_triggers=ON)",
                lambda: conn.execute(sql, params), "DELETE is forbidden")
        finally:
            conn.close()

        survivor = ledger.get(kept.coverage_id)
        _assert(survivor.run_id == "production"
                and survivor.seed_end_exclusive == 1_000_000_000,
                "the certified row was replaced despite recursive_triggers=ON")

    # And production really does set it — the pragma is not left to the caller.
    src = open(os.path.join(_ROOT, "utils", "seed_coverage_ledger.py"),
               encoding="utf-8").read()
    tree = ast.parse(src)
    connect_fn = next((n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == "_connect"),
                      None)
    _assert(connect_fn is not None, "CoverageLedger._connect not found")
    pragmas = [n.value for n in ast.walk(connect_fn)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    _assert(any("recursive_triggers" in p and "ON" in p.upper() for p in pragmas),
            f"_connect does not set recursive_triggers ON; the append-only "
            f"guarantee against REPLACE would be vacuous. Found: {pragmas}")


# ===========================================================================
# G-PUBLICATION-BINDS-COVERAGE
# ===========================================================================
def arm_failed_publication_creates_no_interval():
    _require_amendment()
    with _tmpdb() as path:
        ledger = CoverageLedger(path)
        for bad in (None, object(), _FakeArtifact("not-a-digest"),
                    _FakeArtifact("f" * 64)):
            try:
                ledger.record_publication(bad, dataset_sha256="d" * 64)
                raise AssertionError(
                    f"a failed publication ({type(bad).__name__}) created coverage")
            except (PublicationBindingError, CoverageLedgerError):
                pass
        _assert(ledger.certified_records() == [],
                "a failed publication left rows behind")
        _assert(ledger.certified_cursor("java_lcg", {"constant"}).next_seed_start == 0,
                "a failed publication advanced the cursor")


def arm_successful_publication_creates_exactly_one():
    _require_amendment()
    with _tmpdb() as path:
        ledger = CoverageLedger(path)
        art = _real_artifact(start=0, count=1_000_000, digest="f" * 64,
                             run_id="run-ok",
                             generation_id="gen-20260808T000000Z-step1")
        rec = ledger.record_publication(art, dataset_sha256="d" * 64)

        rows = ledger.certified_records()
        _assert(len(rows) == 1, f"expected exactly one interval, got {len(rows)}")
        _assert(rows[0].artifact_sha256 == "f" * 64,
                "the interval is not bound to the artifact digest")
        _assert(rows[0].generation_id == art.generation_id,
                "the interval is not bound to the generation")
        _assert(rows[0].publication_status == "CERTIFIED", "status not CERTIFIED")
        _assert(rows[0].seed_domain_contract == "v1.1-stratum",
                "the interval does not declare the governed contract")

        _assert(rows[0].prng_base == "java_lcg" and
                rows[0].skip_modes_executed == "constant",
                "the interval does not carry the canonical coverage identity")

        # Re-recording the SAME publication is idempotent, not additive.
        again = ledger.record_publication(art, dataset_sha256="d" * 64)
        _assert(again.coverage_id == rec.coverage_id, "identity is not deterministic")
        _assert(len(ledger.certified_records()) == 1,
                "re-recording one publication double-counted the interval")


def arm_publication_binding_mutation_is_caught():
    """MUTANT: the raw writer, now demoted to a test-only internal bypass.

    In the pre-R1 submission this arm called the PUBLIC
    `record_certified_interval` — which is exactly why Beta refused to read it
    as a hypothetical mutant. It was a live production API bypass.

    The mutation is retained because it still proves the DETECTOR works: a
    fabricated interval really does move the authoritative cursor, so a gate
    that missed it would be measuring nothing. What changed is its STATUS —
    reaching it now requires the underscore-prefixed internal seam, which no
    production path calls (proved by `arm_no_production_certification_bypass`).
    """
    _require_amendment()
    with _tmpdb() as path:
        ledger = CoverageLedger(path)
        before = len(ledger.certified_records())

        # The public name must be GONE — this is the authority boundary.
        _assert(not hasattr(ledger, "record_certified_interval"),
                "record_certified_interval is still a public method; the raw "
                "writer must be an internal seam only (Beta §1)")

        # THE MUTATION, via the internal seam.
        ledger._record_certified_interval(
            run_id="never-published", prng_base="java_lcg",
            skip_modes_executed=("constant",), seed_start=0,
            seed_count=1_000_000_000, dataset_sha256="d" * 64,
            repository_commit="4dd5535", artifact_sha256="0" * 64)

        after = len(ledger.certified_records())
        _assert(after == before + 1,
                "the mutant did not write — the mutation is not live")
        _assert(ledger.certified_cursor("java_lcg", {"constant"}).next_seed_start == 1_000_000_000,
                "the fabricated interval did not move the cursor, so this gate "
                "would not have detected it")
        # The oracle: only record_publication is a legitimate producer, and it
        # refuses every non-witness outright.
        try:
            ledger.record_publication(None, dataset_sha256="d" * 64)
            raise AssertionError("the publication-bound path accepted a non-publication")
        except PublicationBindingError:
            pass


# ===========================================================================
# G-OUT-OF-DOMAIN-LEGACY
# ===========================================================================
def arm_out_of_domain_never_enters_the_union():
    _require_amendment()
    n = normalize_certified_intervals(
        [(0, 1000), (_TERMINUS, _LEGACY_MAX_END), (2 ** 33, 2 ** 34)])
    _assert(n.intervals == ((0, 1000),),
            f"out-of-domain extents entered the union: {n.intervals}")
    _assert(len(n.dropped) == 2,
            f"the rejection is not observable: dropped={n.dropped}")
    _assert(n.covered_seed_count == 1000,
            f"covered count absorbed out-of-domain seeds: {n.covered_seed_count}")


def arm_out_of_domain_cursor_never_exceeds_terminus():
    _require_amendment()
    for intervals in ([(0, 1000), (_TERMINUS, _LEGACY_MAX_END)],
                      [(_TERMINUS, _LEGACY_MAX_END)],
                      [(0, _TERMINUS), (_TERMINUS, _LEGACY_MAX_END)]):
        r = first_uncovered_seed(intervals)
        if r.next_seed_start is not None:
            _assert(0 <= r.next_seed_start < _TERMINUS,
                    f"cursor {r.next_seed_start} escaped [0,{_TERMINUS})")


def arm_out_of_domain_cannot_be_recorded_as_certified():
    _require_amendment()
    with _tmpdb() as path:
        ledger = CoverageLedger(path)
        for start, count, why in [
            (_TERMINUS, 1000, "starting at the terminus"),
            (_TERMINUS - 500, 1000, "straddling the terminus"),
            (2 ** 33, 1000, "wholly beyond the terminus"),
        ]:
            try:
                _record(ledger, start, count, run_id="oob", digest="e" * 64)
                raise AssertionError(f"{why}: an out-of-domain interval was CERTIFIED")
            except SeedDomainPreflightError:
                pass
        _assert(ledger.certified_records() == [], "an out-of-domain row was stored")


def arm_legacy_rows_remain_auditable():
    """Displayable and auditable — the retention half of Beta's disposition."""
    _require_amendment()
    with _tmpdb() as path:
        db = _seed_legacy_tracker(path)
        rows = db.get_exhaustive_progress("step1_java_lcg_4294967296")
        _assert(rows, "an out-of-domain legacy row is no longer readable/auditable")
        _assert(rows[0]["seed_range_start"] == _TERMINUS,
                "the retained row's values were altered")


# ===========================================================================
# R1 BLOCKER A — ONE CERTIFICATION DOOR
# ===========================================================================
def arm_witness_requires_complete_frozen_contract():
    """`_FakeArtifact` refused; a real `RunArtifactResult` succeeds.

    ⚠ R2 §D — WHAT THIS DOES AND DOES NOT GUARANTEE. The pre-R2 title claimed
    "only a canonical RunArtifactResult may certify", which is STRONGER than
    what is enforced. `_require_publication_witness` accepts an `isinstance`
    match OR an object satisfying the COMPLETE frozen dataclass contract — which
    is exactly Beta's own wording ("not the canonical result type **or** cannot
    satisfy the complete frozen result contract"), so the implementation is
    correct and unchanged.

    The real guarantee is therefore: **the production call path, plus complete
    frozen-contract validation.** It is not an unforgeable Python object, and
    nothing here claims one. A stand-in that genuinely reproduces all 28 fields
    with correct types IS the contract; what is refused is the pre-R1 bypass —
    an object whose entire claim to be a publication is a well-formed digest.
    """
    _require_amendment()
    with _tmpdb() as path:
        ledger = CoverageLedger(path)

        # The pre-R1 bypass: an object whose whole claim is a well-formed digest.
        try:
            ledger.record_publication(_FakeArtifact("f" * 64),
                                      dataset_sha256="d" * 64)
            raise AssertionError(
                "_FakeArtifact('f'*64) was accepted as a publication witness — "
                "possessing a digest is not evidence that a canonical "
                "publication occurred (Beta §1)")
        except PublicationBindingError:
            pass
        _assert(ledger.certified_records() == [],
                "the refused witness still left a row")

        # The real thing certifies.
        rec = ledger.record_publication(
            _real_artifact(start=0, count=1000, digest="a" * 64),
            dataset_sha256="d" * 64)
        _assert(rec.artifact_sha256 == "a" * 64, "witness digest not bound")
        _assert(len(ledger.certified_records()) == 1, "real witness did not certify")


def arm_caller_cannot_substitute_artifact_fields():
    """There is NO PARAMETER by which a caller can contradict the witness."""
    _require_amendment()
    import inspect

    sig = inspect.signature(CoverageLedger.record_publication)
    params = set(sig.parameters) - {"self"}
    _assert(params == {"artifact", "dataset_sha256", "study_identity"},
            f"record_publication accepts {sorted(params)}; Beta §1 requires the "
            f"shape (artifact, *, dataset_sha256, study_identity=None) so that "
            f"no caller value can contradict the witness")

    forbidden = {"run_id", "prng_type", "prng_base", "seed_start", "seed_count",
                 "artifact_sha256", "generation_id", "repository_commit",
                 "skip_modes_executed", "mapping_mode"}
    _assert(not (params & forbidden),
            f"record_publication still accepts caller versions of "
            f"{sorted(params & forbidden)} — fields the witness already possesses")

    # And behaviourally: what is stored is what the artifact said.
    with _tmpdb() as path:
        ledger = CoverageLedger(path)
        art = _real_artifact(start=4096, count=2048, digest="b" * 64,
                             run_id="step1_java_lcg_4096",
                             modes=("constant", "variable"),
                             generation_id="gen-real",
                             repository_commit="cafebabe")
        rec = ledger.record_publication(art, dataset_sha256="d" * 64)
        for field_, expected in [
            ("run_id", art.run_id), ("prng_base", art.prng_base),
            ("seed_start", art.seed_start),
            ("seed_end_exclusive", art.seed_end_exclusive),
            ("artifact_sha256", art.artifact_sha256),
            ("generation_id", art.generation_id),
            ("repository_commit", art.repository_commit),
        ]:
            _assert(getattr(rec, field_) == expected,
                    f"stored {field_}={getattr(rec, field_)!r} != artifact "
                    f"{expected!r}")
        _assert(rec.skip_modes_executed == "constant,variable",
                f"executed modes not taken from the artifact: "
                f"{rec.skip_modes_executed!r}")


def arm_no_production_certification_bypass():
    """Repo scan: the ONLY production path creating coverage is record_publication."""
    _require_amendment()
    ledger_rel = os.path.join("utils", "seed_coverage_ledger.py")
    offenders, producers = [], []

    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "__pycache__", "node_modules"}]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, _ROOT)
            # tests/ may reach the internal seam (that is what makes it a
            # test-only bypass); the ledger itself defines it; the apply_s*/
            # verify_s* corpus is forensic and never re-executed.
            if rel.startswith("tests" + os.sep) or rel == ledger_rel:
                continue
            if name.startswith(("apply_s", "verify_s", "fix_s")):
                continue
            try:
                tree = ast.parse(open(full, encoding="utf-8", errors="replace").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("_record_certified_interval",
                                          "record_certified_interval"):
                        offenders.append(f"{rel}:{node.lineno} {node.func.attr}")
                    if node.func.attr in ("record_publication",
                                          "record_certified_coverage"):
                        producers.append(f"{rel}:{node.lineno}")
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    text = node.value.upper()
                    if "INSERT" in text and COVERAGE_LEDGER_TABLE.upper() in text:
                        offenders.append(f"{rel}:{node.lineno} raw INSERT")

    _assert(not offenders,
            f"PRODUCTION CERTIFICATION BYPASS: {offenders}. The only supported "
            f"path that creates certified coverage is record_publication.")
    _assert(producers,
            "no production caller of record_publication/record_certified_coverage "
            "was found — the ledger would have no producer at all")


# ===========================================================================
# R1 BLOCKER B — CANONICAL COVERAGE SCOPE
# ===========================================================================
def arm_containment_matrix():
    """Beta's table, exactly."""
    _require_amendment()
    C, V = "constant", "variable"
    matrix = [
        (( C,   ), {C},    True),
        (( C,   ), {C, V}, False),
        ((C, V  ), {C},    True),
        ((C, V  ), {C, V}, True),
        (( V,   ), {C},    False),
    ]
    for certified, requested, should_count in matrix:
        with _tmpdb() as path:
            ledger = CoverageLedger(path)
            _record(ledger, 0, 1000, run_id="m", digest="a" * 64, modes=certified)
            cursor = ledger.certified_cursor("java_lcg", requested)
            counted = cursor.next_seed_start == 1000
            _assert(counted == should_count,
                    f"certified={set(certified)} requested={requested}: "
                    f"counts={counted}, Beta requires {should_count} "
                    f"(cursor={cursor.next_seed_start})")


def arm_canonical_identity_unifies_hybrid_and_base():
    """The `prng_type` / `prng_base` split is canonicalized on BOTH sides."""
    _require_amendment()
    base, modes = canonical_coverage_identity("java_lcg", test_both_modes=False)
    _assert((base, modes) == ("java_lcg", frozenset({"constant"})), f"{base} {modes}")
    base, modes = canonical_coverage_identity("java_lcg", test_both_modes=True)
    _assert((base, modes) == ("java_lcg", frozenset({"constant", "variable"})),
            f"{base} {modes}")
    for hybrid in ("java_lcg_hybrid", "java_lcg_hybrid_reverse"):
        base, modes = canonical_coverage_identity(hybrid)
        _assert((base, modes) == ("java_lcg", frozenset({"variable"})),
                f"{hybrid} -> {base} {modes}; a hybrid identity is the SAME base "
                f"family under variable skip, not a separate namespace")

    # The end-to-end mismatch Beta refused to leave as backlog: the publication
    # hook records `prng_base`, WATCHER queries `prng_type`. They must meet.
    from database_system import DistributedPRNGDatabase
    with _tmpdb() as path:
        db = DistributedPRNGDatabase(path)
        ledger = CoverageLedger(path)
        # A publication of a both-modes run, recorded as the base family.
        _record(ledger, 0, 1_000_000, run_id="p", digest="a" * 64,
                prng_base="java_lcg", modes=("constant", "variable"))
        # WATCHER querying the HYBRID identity must see it.
        got = db.get_next_seed_start("java_lcg_hybrid", 5_000_000,
                                     test_both_modes=False)
        _assert(got == 1_000_000,
                f"a hybrid query saw {got!r} instead of the base family's "
                f"certified coverage — the namespaces are still split")


def arm_required_modes_has_no_default():
    """Omitting the mode set must be an error, never a permissive guess."""
    _require_amendment()
    import inspect
    from database_system import DistributedPRNGDatabase

    for fn, name in [(CoverageLedger.certified_cursor, "certified_cursor"),
                     (DistributedPRNGDatabase.get_certified_cursor,
                      "get_certified_cursor"),
                     (DistributedPRNGDatabase.get_next_seed_start,
                      "get_next_seed_start")]:
        sig = inspect.signature(fn)
        target = ("required_modes" if name == "certified_cursor"
                  else "test_both_modes")
        _assert(target in sig.parameters,
                f"{name} does not take {target}")
        _assert(sig.parameters[target].default is inspect.Parameter.empty,
                f"{name}'s {target} has a default; a defaulted mode set silently "
                f"picks the weakest request and over-claims coverage")

    with _tmpdb() as path:
        ledger = CoverageLedger(path)
        for bad in (set(), "constant", ("nonsense",)):
            try:
                ledger.certified_cursor("java_lcg", bad)
                raise AssertionError(f"required_modes={bad!r} was accepted")
            except CoverageLedgerError:
                pass


def arm_schema_drift_fails_closed():
    """A pre-R1 table shape must be refused, not silently written into."""
    _require_amendment()
    from utils.seed_coverage_ledger import LedgerSchemaError
    with _tmpdb() as path:
        conn = sqlite3.connect(path)
        conn.execute(
            f"CREATE TABLE {COVERAGE_LEDGER_TABLE} ("
            f"coverage_id TEXT PRIMARY KEY, run_id TEXT, study_identity TEXT, "
            f"prng_type TEXT, mapping_mode TEXT, seed_domain_contract TEXT, "
            f"seed_start INTEGER, seed_end_exclusive INTEGER, "
            f"dataset_sha256 TEXT, repository_commit TEXT, artifact_sha256 TEXT, "
            f"generation_id TEXT, publication_status TEXT, recorded_at TEXT)")
        conn.commit()
        conn.close()
        try:
            CoverageLedger(path)
            raise AssertionError(
                "the pre-R1 `prng_type`/`mapping_mode` table was accepted; "
                "CREATE TABLE IF NOT EXISTS is silent about drift and the new "
                "fields would have been written into columns meaning something "
                "else")
        except LedgerSchemaError:
            pass


# ===========================================================================
# R1 BLOCKER C — THE THIRD EXECUTION PATH
# ===========================================================================
def arm_config_mode_plan_matrix():
    """Beta's four config-mode cases, plus the clean control that isolates the wall.

    `COORDINATOR_AVAILABLE` is forced False so that anything surviving the wall
    dies immediately at the next statement with `SystemExit`. That is the
    discriminator: `SeedDomainPreflightError` means the wall refused the plan;
    `SystemExit` means the wall passed it and nothing sieve-related ran.
    """
    _require_amendment()
    import window_optimizer as wo

    saved = wo.COORDINATOR_AVAILABLE
    wo.COORDINATOR_AVAILABLE = False
    try:
        cases = [
            ("exact bound 2^32 x 1", _TERMINUS, 1, True),
            ("final seed 2^31 x 2", 2 ** 31, 2, True),
            ("2^30 x 4", 2 ** 30, 4, True),
            ("2^30 x 5 (fifth escapes)", 2 ** 30, 5, False),
        ]
        for label, max_seeds, iterations, should_pass in cases:
            try:
                wo.run_with_config(config_file="/nonexistent/s145.json",
                                   lottery_file="/nonexistent/s145.json",
                                   max_seeds=max_seeds, iterations=iterations)
                raise AssertionError(f"{label}: returned without exiting")
            except SeedDomainPreflightError:
                _assert(not should_pass,
                        f"{label}: the wall REFUSED a legal plan")
            except SystemExit:
                _assert(should_pass,
                        f"{label}: the wall PASSED an illegal plan — the fifth "
                        f"interval starts at {4 * 2 ** 30 + 2 ** 30 - 2 ** 30:,} "
                        f"and escapes [0, {_TERMINUS:,})")
    finally:
        wo.COORDINATOR_AVAILABLE = saved


def arm_config_mode_no_sieve_before_plan_validation():
    """Structural + executed: no run_bidirectional_test before the plan validates."""
    _require_amendment()
    src = open(os.path.join(_ROOT, "window_optimizer.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    func = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "run_with_config"),
                None)
    _assert(func is not None, "run_with_config not found")

    wall_lines = [n.lineno for n in ast.walk(func)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "_s145_domain_wall"]
    sieve_lines = [n.lineno for n in ast.walk(func)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "run_bidirectional_test"]
    _assert(wall_lines, "run_with_config has no S145 plan wall")
    _assert(sieve_lines, "no run_bidirectional_test call found in run_with_config "
                         "— the dispatch point this gate orders against has moved")
    _assert(max(wall_lines) < min(sieve_lines),
            f"plan validation at line {max(wall_lines)} does not precede sieve "
            f"execution at line {min(sieve_lines)}")
    earlier = sorted({n.lineno for n in ast.walk(func)
                      if isinstance(n, ast.Call) and n.lineno < min(wall_lines)})
    _assert(not earlier,
            f"calls execute before the plan wall: {earlier}")

    # Executed: an illegal plan must not reach the sieve.
    import window_optimizer_integration_final as woif
    import window_optimizer as wo
    fired = []
    saved_sieve = woif.run_bidirectional_test
    woif.run_bidirectional_test = (
        lambda *a, **k: fired.append("run_bidirectional_test"))
    try:
        try:
            wo.run_with_config(config_file="/nonexistent/s145.json",
                               lottery_file="/nonexistent/s145.json",
                               max_seeds=2 ** 30, iterations=5)
            raise AssertionError("the illegal plan was not refused")
        except SeedDomainPreflightError:
            pass
        _assert(fired == [],
                f"the sieve ran despite an illegal plan: {fired}")
    finally:
        woif.run_bidirectional_test = saved_sieve


def arm_wrappers_do_not_coerce_types():
    """R1 C.2 — malformed values driven through the REAL wrappers, not the helper."""
    _require_amendment()
    import window_optimizer as wo

    # --- CLI wrapper ---
    saved = wo.COORDINATOR_AVAILABLE
    wo.COORDINATOR_AVAILABLE = False
    try:
        for bad in (True, 1.5, "0"):
            try:
                wo.run_bayesian_optimization(
                    lottery_file="/nonexistent/s145.json", trials=1,
                    output_config="/dev/null", seed_count=1000,
                    seed_start=bad, prng_type="java_lcg")
                raise AssertionError(
                    f"CLI wrapper accepted seed_start={bad!r}")
            except SeedDomainPreflightError:
                pass
            except SystemExit:
                raise AssertionError(
                    f"CLI wrapper COERCED seed_start={bad!r} past the wall — "
                    f"int({bad!r}) defeats _require_int's contract")
    finally:
        wo.COORDINATOR_AVAILABLE = saved

    # --- config-mode wrapper ---
    wo.COORDINATOR_AVAILABLE = False
    try:
        for bad in (True, 1.5, "0"):
            try:
                wo.run_with_config(config_file="/nonexistent/s145.json",
                                   lottery_file="/nonexistent/s145.json",
                                   max_seeds=bad, iterations=2)
                raise AssertionError(f"config wrapper accepted max_seeds={bad!r}")
            except SeedDomainPreflightError:
                pass
            except SystemExit:
                raise AssertionError(
                    f"config wrapper COERCED max_seeds={bad!r} past the wall")
    finally:
        wo.COORDINATOR_AVAILABLE = saved

    # --- WATCHER wrapper ---
    with _watcher_harness() as (wa, fired):
        for bad in (True, 1.5, "0"):
            fired.clear()
            result, dispatched = _run_step1(
                wa, {"seed_start": bad, "max_seeds": 1000,
                     "prng_type": "java_lcg"})
            _assert(not dispatched,
                    f"WATCHER dispatched with seed_start={bad!r}")
            _assert(result is not None
                    and result.get("blocked_by") == "seed_domain_preflight",
                    f"WATCHER accepted seed_start={bad!r}: blocked_by="
                    f"{None if result is None else result.get('blocked_by')!r} "
                    f"— int({bad!r}) coercion would produce exactly this")


def main():
    print("=" * 78)
    print("SEED-DOMAIN / COVERAGE-CURSOR AMENDMENT — Beta §10 gates + R1 blockers")
    print("=" * 78)

    print("\nG-DOMAIN-PREFLIGHT")
    _check("G-DOMAIN-PREFLIGHT: Beta's five-case boundary matrix", arm_preflight_matrix)
    _check("G-DOMAIN-PREFLIGHT: bool/float/str seed_start rejected", arm_preflight_rejects_non_int)
    _check("G-DOMAIN-PREFLIGHT: refusal precedes dispatch in live AST", arm_preflight_precedes_dispatch_in_live_source)
    _check("G-DOMAIN-PREFLIGHT: zero dispatch executed (+ clean control)", arm_preflight_zero_dispatch_executed)
    _check("G-DOMAIN-PREFLIGHT: CLI entry point refuses out-of-domain", arm_cli_entry_point_refuses_before_dispatch)
    _check("G-DOMAIN-PREFLIGHT: nothing executes before the CLI wall", arm_cli_wall_precedes_everything_in_its_function)
    _check("G-PUBLICATION-BINDS-COVERAGE: single live producer, after publication", arm_publication_binding_is_wired_after_finalize)
    _check("G-DOMAIN-PREFLIGHT: one shared domain constant, not restated", arm_domain_constant_is_shared_not_restated)
    _check("G-DOMAIN-PREFLIGHT: boundary mutation reds BOTH walls", arm_boundary_mutation_reds_both_walls)

    print("\nG-CURSOR-FIRST-GAP")
    _check("G-CURSOR-FIRST-GAP: Beta's worked example returns 1000", arm_first_gap_beta_worked_example)
    _check("G-CURSOR-FIRST-GAP: through production get_next_seed_start", arm_first_gap_through_production_db)
    _check("G-CURSOR-FIRST-GAP: order-invariant, adjacency and overlap", arm_first_gap_interior_hole_and_ordering)
    _check("G-CURSOR-FIRST-GAP: MUTANT MAX(seed_range_end) is caught", arm_first_gap_mutation_max_end_is_caught)

    print("\nG-CURSOR-COMPLETE")
    _check("G-CURSOR-COMPLETE: COMPLETE carries no numeric next seed", arm_complete_returns_no_number)
    _check("G-CURSOR-COMPLETE: production cursor returns None", arm_complete_production_db_returns_none)
    _check("G-CURSOR-COMPLETE: no WATCHER run generated, no dispatch", arm_complete_generates_no_watcher_run)

    print("\nG-LEGACY-NONAUTHORITY")
    _check("G-LEGACY-NONAUTHORITY: 16.1B history ignored completely", arm_legacy_history_is_ignored_by_certified_cursor)
    _check("G-LEGACY-NONAUTHORITY: rows retained, not deleted or folded", arm_legacy_rows_are_not_deleted_or_rewritten)
    _check("G-LEGACY-NONAUTHORITY: ledger never queries the legacy table", arm_legacy_table_is_never_read_by_the_ledger)

    print("\nG-NO-REPLACE-CLOBBER")
    _check("G-NO-REPLACE-CLOBBER: 1,000-seed smoke cannot erase a big interval", arm_smoke_run_cannot_replace_a_certified_interval)
    _check("G-NO-REPLACE-CLOBBER: control — the legacy writer really clobbers", arm_legacy_writer_really_does_clobber)
    _check("G-NO-REPLACE-CLOBBER: UPDATE/DELETE/REPLACE abort for the append-only reason", arm_append_only_is_enforced_by_the_database)
    _check("G-NO-REPLACE-CLOBBER: recursive_triggers=ON is load-bearing (ON/OFF)", arm_recursive_triggers_is_load_bearing)

    print("\nG-PUBLICATION-BINDS-COVERAGE")
    _check("G-PUBLICATION-BINDS-COVERAGE: failed publication creates none", arm_failed_publication_creates_no_interval)
    _check("G-PUBLICATION-BINDS-COVERAGE: success creates exactly one", arm_successful_publication_creates_exactly_one)
    _check("G-PUBLICATION-BINDS-COVERAGE: MUTANT unbound write is caught", arm_publication_binding_mutation_is_caught)

    print("\nR1 BLOCKER A — ONE CERTIFICATION DOOR")
    _check("R1-A: canonical publication witness / complete frozen contract required", arm_witness_requires_complete_frozen_contract)
    _check("R1-A: caller cannot substitute any artifact field", arm_caller_cannot_substitute_artifact_fields)
    _check("R1-A: repo scan — no production certification bypass", arm_no_production_certification_bypass)

    print("\nR1 BLOCKER B — CANONICAL COVERAGE SCOPE")
    _check("R1-B: Beta's containment matrix (5 rows)", arm_containment_matrix)
    _check("R1-B: hybrid and base canonicalize to one namespace", arm_canonical_identity_unifies_hybrid_and_base)
    _check("R1-B: the mode set has no default, anywhere", arm_required_modes_has_no_default)
    _check("R1-B: pre-R1 schema shape fails closed", arm_schema_drift_fails_closed)

    print("\nR1 BLOCKER C — THE THIRD EXECUTION PATH")
    _check("R1-C: config-mode whole-plan matrix (+ clean control)", arm_config_mode_plan_matrix)
    _check("R1-C: no sieve executes before plan validation", arm_config_mode_no_sieve_before_plan_validation)
    _check("R1-C.2: wrappers do not coerce types", arm_wrappers_do_not_coerce_types)

    print("\nG-OUT-OF-DOMAIN-LEGACY")
    _check("G-OUT-OF-DOMAIN-LEGACY: never enters the normalized union", arm_out_of_domain_never_enters_the_union)
    _check("G-OUT-OF-DOMAIN-LEGACY: cursor never escapes the terminus", arm_out_of_domain_cursor_never_exceeds_terminus)
    _check("G-OUT-OF-DOMAIN-LEGACY: cannot be recorded as certified", arm_out_of_domain_cannot_be_recorded_as_certified)
    _check("G-OUT-OF-DOMAIN-LEGACY: retained rows remain auditable", arm_legacy_rows_remain_auditable)

    print()
    print("=" * 78)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    for name, ok, tb in _results:
        if not ok:
            print(f"\n--- {name} ---\n{tb}")
    print(f"\n{passed}/{total} checks green")
    if passed == total:
        print("COMPLETION SENTINEL: PASS — seed-domain cursor amendment green")
        return 0
    print("COMPLETION SENTINEL: FAIL — seed-domain cursor amendment red")
    return 1


if __name__ == "__main__":
    sys.exit(main())
