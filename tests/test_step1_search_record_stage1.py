#!/usr/bin/env python3
"""tests/test_step1_search_record_stage1.py — Step 1 search record, STAGE 1.

Brief:  docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_SEARCH_RECORD_STAGE1_20261114.md §3
Design: docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md §2.1-§2.7
Owner rulings 2026-11-14: G-MINER-START and the S7 interval guard (report §6).

CPU only. No fleet, no GPU, no rig, no SSH, no sieve. Every database, ledger, Optuna
study, checkpoint root and output is a throwaway under
<worktree>/data/step1_search_runs/suite_tmp/. Dispatch is intercepted on every WATCHER
path (`_run_step_streaming` captures the argv and returns); every integration path runs
the REAL optimize_window with `run_bidirectional_test` replaced by a stub that runs no
sieve. The Optuna storage base is pointed at the throwaway directory.

One labelled intercept: `window_optimizer_integration_final._repository_state` reports
the base commit as CLEAN, because the finalizer refuses an uncommitted tree and this
worktree is uncommitted by rule. It is what running from a clean checkout does.

Groups (Beta R1 Q11) and the brief's extra cases:
  G1 credit · G2 budget/resume · G3 identity · G4 publication/link · G5 effective input
  G6 authority/freshness · F5a · RUN4-SHAPE · G-MINER-START
Every gate that guards a refusal has a mutant that must turn it RED; a mutant whose
anchor is not found exactly once is an ERROR, never a detection.

Run:  source ~/venvs/torch/bin/activate && PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
      CUDA_VISIBLE_DEVICES="" python3 -u tests/test_step1_search_record_stage1.py
Prints one line per gate and per mutant, then `COMPLETION SENTINEL: PASS` only if every
gate passed and every mutant was detected.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import traceback
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SCRATCH = os.path.join(ROOT, "data", "step1_search_runs", "suite_tmp")
os.makedirs(SCRATCH, exist_ok=True)

T31, T32 = 2 ** 31, 2 ** 32
RUN4_PINS = {"warm_start_window": 12, "warm_start_offset": 25,
             "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71,
             "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 6,
             "warm_start_skip_max": 99}
# Run 4's non-pin params (tests/fixtures/run4_routing_clean_control.CONTROL_PARAMS).
RUN4_BASE = {"prng_type": "java_lcg", "seed_start": 0, "max_seeds": T31,
             "miner_stripe_size": 67108864, "worker_pool_size": 25,
             "test_both_modes": True, "use_range_miner": True,
             "use_persistent_workers": False, "window_trials": 1, "n_parallel": 1}

RESULTS = []          # (name, ok, detail)
MUTANTS = []          # (label, detected, detail)


class Fail(AssertionError):
    pass


def ok_(cond, msg):
    if not cond:
        raise Fail(msg)


def run_gate(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail or ""))
    except BaseException as e:      # noqa: BLE001 — a gate reports
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"[:600]))


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ===========================================================================
# mutants
# ===========================================================================
_MOD_PATHS = {
    "utils.step1_search_ledger": os.path.join(ROOT, "utils", "step1_search_ledger.py"),
    "window_optimizer_integration_final": os.path.join(ROOT, "window_optimizer_integration_final.py"),
    "agents.watcher_agent": os.path.join(ROOT, "agents", "watcher_agent.py"),
    "window_optimizer": os.path.join(ROOT, "window_optimizer.py"),
}


@contextlib.contextmanager
def mutated(modname, old, new):
    """Load a mutated copy of a live module and install it under its real name."""
    import importlib
    path = _MOD_PATHS[modname]
    src = open(path, encoding="utf-8").read()
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"MUTANT ANCHOR ERROR in {modname}: found {n} times: {old[:80]!r}")
    real = importlib.import_module(modname)
    mod = types.ModuleType(modname)
    mod.__file__ = path
    mod.__dict__["__name__"] = modname
    if "." in modname:
        mod.__package__ = modname.rsplit(".", 1)[0]
    exec(compile(src.replace(old, new), path + "<mutant>", "exec"), mod.__dict__)
    parent = None
    if "." in modname:
        parent = importlib.import_module(modname.rsplit(".", 1)[0])
    sys.modules[modname] = mod
    if parent is not None:
        setattr(parent, modname.rsplit(".", 1)[1], mod)
    try:
        yield mod
    finally:
        sys.modules[modname] = real
        if parent is not None:
            setattr(parent, modname.rsplit(".", 1)[1], real)


def run_mutant(label, modname, old, new, gate_fn):
    try:
        with mutated(modname, old, new):
            try:
                gate_fn()
                MUTANTS.append((label, False, "gate stayed GREEN under the mutant"))
            except BaseException as e:   # noqa: BLE001
                MUTANTS.append((label, True, f"RED: {type(e).__name__}: {str(e)[:220]}"))
    except RuntimeError as e:
        MUTANTS.append((label, False, f"ERROR (not a detection): {e}"))


# ===========================================================================
# shared fixtures
# ===========================================================================
def frozen_dataset():
    from miner import dataset_authority as DA
    fz = DA.get_frozen_dataset()
    if fz is None:
        with quiet():
            fz = DA.freeze_for_run(os.path.join(ROOT, "daily3.json"), run_label="s1s_suite")
    return fz


def dataset_fields(fz=None):
    fz = fz or frozen_dataset()
    return {"dataset_version_id": fz.version_id or "", "dataset_sha256": fz.sha256,
            "input_manifest_sha256": fz.manifest_sha256 or "",
            "dataset_lineage_id": fz.dataset_lineage_id or ""}


def ledger_at(path):
    from utils.step1_search_ledger import Step1SearchLedger
    return Step1SearchLedger(path)


def cfg_for(final_params, pin=None):
    """The configuration a run presents, computed the way WATCHER computes it."""
    import agents.watcher_agent as W
    return W.step1_presented_configuration(final_params, pin)


def simple_final(trials=1, chunk=1000, both=True, pruning=False):
    return {"prng_type": "java_lcg", "test_both_modes": both, "window_trials": trials,
            "max_seeds": chunk, "strategy": "bayesian", "n_parallel": 1,
            "enable_pruning": pruning}


def open_simple(led, *, start=0, end=T32, trials=1, chunk=1000, both=True, repeat_of=None,
                reason="operator", dataset=None):
    rec, created = led.open_search(
        configuration=cfg_for(simple_final(trials, chunk, both)), range_start=start,
        range_end_exclusive=end, dataset=dataset or dataset_fields(),
        opened_reason=reason, opened_by_ref="suite", repository_commit="suite",
        repeat_of=repeat_of)
    return rec, created


@contextlib.contextmanager
def tmpdir(prefix):
    d = tempfile.mkdtemp(prefix=prefix, dir=SCRATCH)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def table_rows(path):
    """{table: sorted list of row tuples} for every table in a db."""
    out = {}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            out[t] = sorted(tuple(r) for r in con.execute(f"SELECT * FROM {t}"))
    finally:
        con.close()
    return out


def rows_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


# --- synthetic publication witnesses (for ledger-level link tests only) -----
def fake_witness(start, end, *, modes=("constant", "variable"), gen="gen-x", prng="java_lcg",
                 sha="a" * 64):
    """A RunArtifactResult VALUE for ledger-level tests. It passes the type check because it
    IS the frozen dataclass; it is never published anywhere."""
    from pathlib import Path
    from utils.run_finalizer import RunArtifactResult
    import dataclasses
    vals = {}
    for f in dataclasses.fields(RunArtifactResult):
        t = str(f.type)
        vals[f.name] = (Path("/nonexistent") if "Path" in t else
                        0 if t in ("int", "<class 'int'>") else
                        False if "bool" in t else None if "Optional" in t else "x")
    vals.update(run_id=f"step1_{prng}_{start}", prng_base=prng,
                skip_modes_executed=tuple(modes), seed_start=start,
                seed_count=end - start, seed_end_exclusive=end, generation_id=gen,
                artifact_sha256=sha, repository_commit="suite", repository_tree_clean=True)
    return RunArtifactResult(**vals)


def history_row(db_path, witness, dataset_sha="b" * 64, study=None):
    from utils.seed_coverage_ledger import CoverageLedger
    return CoverageLedger(db_path).record_publication(witness, dataset_sha256=dataset_sha,
                                                      study_identity=study)


def credit_chunk(led, db_path, rec, start, end, *, study="study-x", gen=None):
    ch = led.chunk_open(search_id=rec.search_id, seed_start=start, seed_end_exclusive=end,
                        study_name=study, study_trials_at_open=0,
                        budget_target=rec.trials_per_chunk, checkpoint_run_id="ck")
    w = fake_witness(start, end, modes=rec.required_modes, gen=gen or f"gen-{start}")
    cov = history_row(db_path, w)
    args = dict(search_id=rec.search_id, chunk_id=ch["chunk_id"], artifact=w, coverage=cov,
                trial_counts={"complete": 1})
    led.record_link_pending(**args)            # production order: pending, then link
    return led.record_chunk_link(**args)


# ===========================================================================
# integration harness — the REAL optimize_window, no sieve
# ===========================================================================
class Behaviour:
    """Per-call stub behaviour for run_bidirectional_test."""

    def __init__(self, kinds, *, interval=None, phases=None):
        self.kinds = list(kinds)     # "ok" | "prune" | "fail" | "noev"
        self.interval = interval     # override the swept interval
        self.phases = phases         # override the phase list
        self.calls = []


@contextlib.contextmanager
def integ_env():
    """cwd, checkpoint root, Optuna storage and the history/search db, all throwaway."""
    import window_optimizer_bayesian as WOB
    import window_optimizer_integration_final as W
    d = tempfile.mkdtemp(prefix="integ_", dir=SCRATCH)
    os.makedirs(os.path.join(d, "optuna_studies"))
    prev = (os.getcwd(), os.environ.get("PRNG_CHECKPOINT_ROOT"), WOB._OPTUNA_STORAGE_BASE,
            W._repository_state)
    os.chdir(d)
    os.environ["PRNG_CHECKPOINT_ROOT"] = os.path.join(d, "ckpt")
    WOB._OPTUNA_STORAGE_BASE = d
    _real_rs = W._repository_state
    W._repository_state = lambda *a, **k: (_real_rs(*a, **k)[0], True)   # LABELLED INTERCEPT
    frozen_dataset()
    try:
        yield d
    finally:
        os.chdir(prev[0])
        if prev[1] is None:
            os.environ.pop("PRNG_CHECKPOINT_ROOT", None)
        else:
            os.environ["PRNG_CHECKPOINT_ROOT"] = prev[1]
        WOB._OPTUNA_STORAGE_BASE = prev[2]
        W._repository_state = prev[3]
        shutil.rmtree(d, ignore_errors=True)


def run_chunk(beh, *, seed_start, seed_count, trials=1, resume=False, study="",
              search_id="", both=True, n_parallel=1, legacy_result=None):
    """Drive the REAL optimize_window (in the module currently in sys.modules)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    import importlib
    W = importlib.import_module("window_optimizer_integration_final")
    import coordinator as C
    from window_optimizer import TestResult

    class FakeCoord:
        use_range_miner = False
        use_persistent_workers = False
        use_zmq_sqlite = False

        def __init__(self, *a, **k):
            self.config_file = "distributed_config.json"

        def execute_distributed_analysis(self, target_file, output_file, args, seeds,
                                         samples, lmax, grid):
            # The coordinator's own result SHAPE (coordinator.py:2203-2222); no worker runs.
            beh.calls.append({"kind": "legacy_call", "prng": getattr(args, "prng_type", None)})
            return legacy_result(args)

    def stub(coordinator, config, dataset_path, seed_start, seed_count, **kw):
        kind = beh.kinds.pop(0) if beh.kinds else "ok"
        beh.calls.append({"kind": kind, "optuna": getattr(kw.get("optuna_trial"), "number", None)})
        if kind == "prune":
            raise optuna.exceptions.TrialPruned()
        if kind == "fail":
            raise RuntimeError("suite: injected trial failure")
        tr = TestResult(config=config, forward_count=0, reverse_count=0,
                        bidirectional_count=0, iteration=kw.get("trial_number", 0))
        if kind == "ok":
            iv = beh.interval or [int(seed_start), int(seed_start) + int(seed_count)]
            ph = beh.phases or W._step1_required_phases(kw.get("prng_base", "java_lcg"),
                                                        kw.get("test_both_modes", both))
            tr.step1_sweep_evidence = {"route": "suite_stub", "source": "suite",
                                       "swept_intervals": {str(p): list(iv) for p in ph}}
        return tr                                   # "noev": no evidence at all

    saved = (C.MultiGPUCoordinator, W.run_bidirectional_test)
    C.MultiGPUCoordinator = FakeCoord
    if legacy_result is None:
        W.run_bidirectional_test = stub          # else: the REAL evidence producer runs
    try:
        with quiet():
            W.add_window_optimizer_to_coordinator()
            return FakeCoord().optimize_window(
                dataset_path=frozen_dataset().path, seed_start=seed_start,
                seed_count=seed_count, prng_base="java_lcg", test_both_modes=both,
                strategy_name="bayesian", max_iterations=trials,
                output_file="window_optimization_results.json", resume_study=resume,
                study_name=study, n_parallel=n_parallel, enable_pruning=False,
                step1_search_id=search_id)
    finally:
        C.MultiGPUCoordinator, W.run_bidirectional_test = saved


def coverage_rows(d):
    p = os.path.join(d, "prng_analysis.db")
    if not os.path.exists(p):
        return []
    con = sqlite3.connect(p)
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name='certified_coverage'").fetchone():
            return []
        return con.execute("SELECT seed_start, seed_end_exclusive, generation_id "
                           "FROM certified_coverage ORDER BY seed_start").fetchall()
    finally:
        con.close()


def search_rows(d, table):
    p = os.path.join(d, "prng_analysis.db")
    if not os.path.exists(p):
        return []
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone():
            return []
        return [dict(r) for r in con.execute(f"SELECT * FROM {table}")]
    finally:
        con.close()


def latest_study(d):
    names = sorted(n[:-3] for n in os.listdir(os.path.join(d, "optuna_studies"))
                   if n.endswith(".db"))
    return names[-1] if names else None


def expect_raises(exc_name, fn):
    try:
        fn()
    except BaseException as e:   # noqa: BLE001
        if type(e).__name__ != exc_name:
            raise Fail(f"expected {exc_name}, got {type(e).__name__}: {e}")
        return e
    raise Fail(f"expected {exc_name}, nothing was raised")


# ===========================================================================
# WATCHER harness — real run_step / run_pipeline, dispatch intercepted
# ===========================================================================
class _Done:
    returncode, stdout, stderr = 0, "", ""


@contextlib.contextmanager
def watcher_env(*, fresh=False, frozen_override=None, db_dir=None):
    """Isolated db, real manifests, real dataset freeze (worktree copy), argv captured."""
    import importlib
    W = importlib.import_module("agents.watcher_agent")
    import database_system as DS
    from miner import dataset_authority as DA
    d = db_dir or tempfile.mkdtemp(prefix="watch_", dir=SCRATCH)
    dbp = os.path.join(d, "prng_analysis.db")
    Real = DS.DistributedPRNGDatabase

    class IsolatedDB(Real):
        def __init__(self, db_path=None):
            super().__init__(dbp)

    cap = {"argv": [], "telegram": []}
    saved_w = {n: getattr(W, n) for n in ("check_output_freshness", "check_safety",
                                         "notify_telegram", "PROGRESS_DISPLAY_AVAILABLE",
                                         "p05_freeze_dataset")}
    _da_stubs = {"fleet_preflight": lambda *a, **k: [],
                 "write_run_provenance": lambda *a, **k: None,
                 "_active_execution_set": lambda: None,
                 "load_provisioning_nodes": lambda *a, **k: [],
                 "resolve_absent_fleet_status": lambda *a, **k: "NOT_APPLICABLE",
                 "default_provisioning_manifest_path": lambda *a, **k: "<suite-stub>"}
    saved_da = {n: getattr(DA, n) for n in _da_stubs}
    saved_cls = (W.WatcherAgent._run_step_streaming, W.WatcherAgent._run_preflight_check)
    W.check_output_freshness = (lambda step: (True, "suite: forced FRESH", False)) if fresh \
        else (lambda step: (False, "suite: forced stale", False))
    W.check_safety = lambda: True
    W.notify_telegram = lambda msg, *a, **k: cap["telegram"].append(msg)
    W.PROGRESS_DISPLAY_AVAILABLE = False
    if frozen_override is not None:
        W.p05_freeze_dataset = lambda run_label="watcher": frozen_override
    for _n, _v in _da_stubs.items():
        setattr(DA, _n, _v)

    def _stream(self, cmd, step, timeout):
        cap["argv"].append(list(cmd))
        return _Done()

    W.WatcherAgent._run_step_streaming = _stream
    W.WatcherAgent._run_preflight_check = lambda self, step: (True, "suite")
    DS.DistributedPRNGDatabase = IsolatedDB
    try:
        yield W, cap, dbp, d
    finally:
        for n, v in saved_w.items():
            setattr(W, n, v)
        for n, v in saved_da.items():
            setattr(DA, n, v)
        W.WatcherAgent._run_step_streaming, W.WatcherAgent._run_preflight_check = saved_cls
        DS.DistributedPRNGDatabase = Real
        if db_dir is None:
            shutil.rmtree(d, ignore_errors=True)


def make_agent(W, d):
    cfg = W.WatcherConfig(
        use_llm=False, manifests_dir=os.path.join(ROOT, "agent_manifests"),
        history_file=os.path.join(d, "watcher_history.json"),
        log_file=os.path.join(d, "watcher_decisions.jsonl"),
        pid_file=os.path.join(d, "watcher_daemon.pid"),
        daemon_state_file=os.path.join(d, "daemon_state.json"),
        pending_approval_file=os.path.join(d, "pending_approval.json"),
        halt_file=os.path.join(d, "halt"))
    a = W.WatcherAgent(cfg)
    a.llm_lifecycle = None
    a.llm_router = None
    a._run_post_step_cleanup = lambda *x, **k: None
    a._find_results = lambda *x, **k: {}
    # The execution set is frozen per process and the integration cases already read
    # it; the Route-A fixture stubs this seam the same way (non-dict => proceed).
    a._ensure_execution_set = lambda p=None: object()
    return a


def cli_open(W, agent, params, *, start, end, reason="suite opening", repeat_of=None):
    """The operator's open command, through the production CLI function."""
    args = types.SimpleNamespace(
        params=json.dumps(params), step1_search_open=True, step1_search_close=None,
        step1_search_status=False, search_range_start=start, search_range_end=end,
        search_reason=reason, search_repeat_of=repeat_of)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = W.step1_search_cli(agent, args)
    return rc, buf.getvalue()


def run_step1(W, agent, params, pin=None):
    with quiet():
        return agent.run_step(1, dict(params), _pin_bundle=pin)


def argv_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


# ===========================================================================
# G1 — CREDIT
# ===========================================================================
def g1_no_complete_trial():
    """PRUNED-only chunk: no certified publication, no coverage row, no link, one
    non-certifying record."""
    with integ_env() as d:
        led = ledger_at("prng_analysis.db")
        rec, _ = open_simple(led, trials=2, chunk=1000)
        beh = Behaviour(["prune", "prune"])
        expect_raises("Step1NoCreditError",
                      lambda: run_chunk(beh, seed_start=0, seed_count=1000, trials=2,
                                        search_id=rec.search_id))
        ok_(coverage_rows(d) == [], f"a coverage row was written: {coverage_rows(d)}")
        ok_(search_rows(d, "step1_chunk_link") == [], "a link was written")
        unc = search_rows(d, "step1_uncredited_attempt")
        ok_(len(unc) == 1 and unc[0]["reason_code"] == "no_complete_trial",
            f"non-certifying record: {unc}")
        ok_(not os.path.exists(os.path.join(d, "survivors")) or not any(
            "gen-" in n for n in os.listdir(os.path.join(d, "survivors"))),
            "a generation was published")
        return f"2 pruned trials -> no credit; uncredited={unc[0]['reason_code']}"


def g1_zero_survivors_credited():
    with integ_env() as d:
        led = ledger_at("prng_analysis.db")
        rec, _ = open_simple(led, trials=1, chunk=1000)
        run_chunk(Behaviour(["ok"]), seed_start=0, seed_count=1000, trials=1,
                  search_id=rec.search_id)
        cov = coverage_rows(d)
        links = search_rows(d, "step1_chunk_link")
        ok_(cov == [(0, 1000, cov[0][2])] if cov else False, f"coverage: {cov}")
        ok_(len(links) == 1 and links[0]["trials_complete"] == 1
            and (links[0]["seed_start"], links[0]["seed_end_exclusive"]) == (0, 1000),
            f"link: {links}")
        ok_(ledger_at("prng_analysis.db").cursor(rec).next_seed_start == 1000,
            "the search did not advance to 1000")
        return f"0 survivors, 1 COMPLETE trial -> credited; coverage {cov[0][:2]}; cursor 1000"


def g1_complete_trial_of_other_chunk():
    """The study holds a COMPLETE trial from an earlier chunk; this chunk's only trial is
    COMPLETE without sweep evidence (the miner 'pruned' return shape). No credit."""
    with integ_env() as d:
        run_chunk(Behaviour(["ok"]), seed_start=0, seed_count=1000, trials=1)
        study = latest_study(d)
        n_before = len(coverage_rows(d))
        expect_raises("Step1NoCreditError",
                      lambda: run_chunk(Behaviour(["noev"]), seed_start=1000, seed_count=1000,
                                        trials=2, resume=True, study=study))
        import optuna
        st = optuna.load_study(study_name=study, storage=f"sqlite:///{d}/optuna_studies/{study}.db")
        nc = sum(1 for t in st.trials if t.state.name == "COMPLETE")
        ok_(nc == 2, f"study COMPLETE count {nc}, expected 2")
        ok_(len(coverage_rows(d)) == n_before, "chunk 2 earned a coverage row")
        unc = search_rows(d, "step1_uncredited_attempt")
        ok_(unc and unc[-1]["reason_code"] == "no_sweep_evidence", f"uncredited: {unc}")
        return f"study COMPLETE=2, this chunk's trial has no evidence -> no credit"


def g1_interval_mismatch():
    """OWNER GUARD 2: the swept interval must EQUAL the requested one. The stub reports
    the miner's behaviour: requested [N, 2N), swept [0, N)."""
    with integ_env() as d:
        N = 1000
        expect_raises("Step1NoCreditError",
                      lambda: run_chunk(Behaviour(["ok"], interval=[0, N]), seed_start=N,
                                        seed_count=N, trials=1))
        ok_(coverage_rows(d) == [], f"coverage written: {coverage_rows(d)}")
        unc = search_rows(d, "step1_uncredited_attempt")
        ok_(unc and unc[-1]["reason_code"] == "interval_mismatch", f"uncredited: {unc}")
        return "requested [1000,2000), swept [0,1000) -> interval_mismatch, nothing published"


def g1_missing_phase():
    with integ_env() as d:
        expect_raises("Step1NoCreditError",
                      lambda: run_chunk(Behaviour(["ok"], phases=[1, 2, 3]), seed_start=0,
                                        seed_count=1000, trials=1))
        ok_(coverage_rows(d) == [], "coverage written")
        unc = search_rows(d, "step1_uncredited_attempt")
        ok_(unc and unc[-1]["reason_code"] == "no_sweep_evidence"
            and "not finished" in unc[-1]["reason_detail"], f"reason: {unc[-1:]}")
        return "phase 4 not finished -> no credit (no_sweep_evidence: phases [4] not finished)"


def g1_miner_evidence_reads_ledger():
    """The miner evidence builder reads the run's stripe rows; a phase with a non-done
    stripe or a gap is not evidence."""
    import window_optimizer_integration_final as W
    with tmpdir("mev_") as d:
        con = sqlite3.connect(os.path.join(d, "miner_ledger.db"))
        con.execute("CREATE TABLE stripes (run_id, phase, seed_start, seed_count, state)")
        rows = [("r", 1, 0, 500, "done"), ("r", 1, 500, 500, "done"),
                ("r", 2, 0, 500, "done"), ("r", 2, 500, 500, "cancelled"),
                ("r", 3, 0, 400, "done"), ("r", 3, 500, 500, "done")]
        con.executemany("INSERT INTO stripes VALUES (?,?,?,?,?)", rows)
        con.commit()
        con.close()
        coord = types.SimpleNamespace(staging_dir=d, miner_output_dir=None)
        ev = W._step1_miner_sweep_evidence({"run_id": "r", "committed": True}, coord)
        ok_(ev["swept_intervals"] == {"1": [0, 1000]}, f"evidence: {ev}")
        ok_(W._step1_miner_sweep_evidence({"run_id": "r", "committed": False}, coord) is None,
            "an uncommitted trial produced evidence")
        return f"phase1 [0,1000) only; phase2 cancelled and phase3 gap excluded"


# ===========================================================================
# G2 — BUDGET AND RESUME
# ===========================================================================
def g2_m1_case_refused():
    """M1 run B after the change: a resumed study that already met its budget refuses
    before dispatch, naming the counts — never a silent certification."""
    with integ_env() as d:
        run_chunk(Behaviour(["ok"]), seed_start=0, seed_count=1000, trials=1)
        study = latest_study(d)
        beh = Behaviour(["ok"])
        e = expect_raises("Step1ChunkBudgetError",
                          lambda: run_chunk(beh, seed_start=1000, seed_count=1000, trials=1,
                                            resume=True, study=study))
        ok_(beh.calls == [], f"the objective ran: {beh.calls}")
        ok_([r[:2] for r in coverage_rows(d)] == [(0, 1000)], f"coverage: {coverage_rows(d)}")
        ok_("completed=1" in str(e) and "budget=1" in str(e), f"counts not named: {e}")
        return "zero-trial chunk refused before dispatch: " + str(e)[-40:]


def g2_fail_only():
    with integ_env() as d:
        expect_raises("RuntimeError",
                      lambda: run_chunk(Behaviour(["fail"]), seed_start=0, seed_count=1000,
                                        trials=1))
        ok_(coverage_rows(d) == [], "coverage written")
        unc = search_rows(d, "step1_uncredited_attempt")
        ok_(unc and unc[-1]["reason_code"] == "search_raised", f"uncredited: {unc}")
        return "FAIL-only chunk -> no credit, recorded search_raised"


def g2_pruned_only_no_search():
    with integ_env() as d:
        expect_raises("Step1NoCreditError",
                      lambda: run_chunk(Behaviour(["prune"]), seed_start=0, seed_count=1000,
                                        trials=1))
        ok_(coverage_rows(d) == [], "coverage written")
        return "PRUNED-only chunk outside any search -> no credit"


# ===========================================================================
# G3 — IDENTITY
# ===========================================================================
def g3_retry_same_id():
    with tmpdir("id_") as d:
        led = ledger_at(os.path.join(d, "db"))
        a, c1 = open_simple(led)
        b, c2 = open_simple(led)
        ok_(a.search_id == b.search_id and c1 and not c2, f"{a.search_id} {b.search_id} {c1} {c2}")
        return f"retry -> same id {a.search_id[:20]}, created only once"


def g3_explicit_repeat():
    with tmpdir("id_") as d:
        led = ledger_at(os.path.join(d, "db"))
        a, _ = open_simple(led)
        led.close_search(a.search_id, "suite: done")
        expect_raises("Step1SearchRefused", lambda: open_simple(led))
        b, created = open_simple(led, repeat_of=a.search_id)
        ok_(created and b.search_id != a.search_id and b.repeat_of == a.search_id, "repeat")
        return f"same identity after close refused; explicit repeat -> new id, repeat_of set"


def g3_second_open_refuses():
    with tmpdir("id_") as d:
        led = ledger_at(os.path.join(d, "db"))
        open_simple(led, chunk=1000)
        e = expect_raises("Step1SearchRefused", lambda: open_simple(led, chunk=2000))
        return f"second open for the same selector refused: {str(e)[:60]}"


def g3_one_open_in_one_transaction():
    """Eight concurrent openers of DIFFERENT searches on one selector: exactly one wins."""
    with tmpdir("race_") as d:
        path = os.path.join(d, "db")
        ledger_at(path)
        fz = dataset_fields()
        wins, errs = [], []
        barrier = threading.Barrier(8)

        def opener(i):
            led = ledger_at(path)
            barrier.wait()
            try:
                rec, created = led.open_search(
                    configuration=cfg_for(simple_final(chunk=1000 + i)), range_start=0,
                    range_end_exclusive=T32, dataset=fz, opened_reason="operator",
                    opened_by_ref=f"t{i}", repository_commit="suite")
                if created:
                    wins.append(rec.search_id)
            except Exception as e:      # noqa: BLE001
                errs.append(type(e).__name__)

        ts = [threading.Thread(target=opener, args=(i,)) for i in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        ok_(len(wins) == 1 and errs.count("Step1SearchRefused") == 7, f"wins={wins} errs={errs}")
        return "8 racing openers -> 1 opened, 7 refused"


def g3_study_reuse_refuses():
    with tmpdir("own_") as d:
        dbp = os.path.join(d, "db")
        led = ledger_at(dbp)
        a, _ = open_simple(led, chunk=1000)
        led.chunk_open(search_id=a.search_id, seed_start=0, seed_end_exclusive=1000,
                       study_name="study-A", study_trials_at_open=0, budget_target=1,
                       checkpoint_run_id="ck")
        led.close_search(a.search_id, "suite")
        b, _ = open_simple(led, chunk=2000)
        expect_raises("Step1StudyOwnershipError", lambda: led.chunk_open(
            search_id=b.search_id, seed_start=0, seed_end_exclusive=2000, study_name="study-A",
            study_trials_at_open=0, budget_target=1, checkpoint_run_id="ck"))
        led.chunk_open(search_id=b.search_id, seed_start=0, seed_end_exclusive=2000,
                       study_name="study-B", study_trials_at_open=0, budget_target=1,
                       checkpoint_run_id="ck")
        expect_raises("Step1StudyOwnershipError",
                      lambda: led.assert_study_usable(b.search_id, "study-C"))
        return "study owned by another search refused; a name other than the bound study refused"


def g3_close_then_open():
    with tmpdir("id_") as d:
        led = ledger_at(os.path.join(d, "db"))
        a, _ = open_simple(led, chunk=1000)
        led.close_search(a.search_id, "suite")
        b, created = open_simple(led, chunk=2000)
        ok_(created and led.search_state(a.search_id) == "closed"
            and led.search_state(b.search_id) == "open", "close/open")
        return "close then open works"


def g3_append_only():
    from utils import step1_search_ledger as L
    with tmpdir("ao_") as d:
        p = os.path.join(d, "db")
        led = ledger_at(p)
        a, _ = open_simple(led, chunk=T31)
        led.record_uncredited(search_id=a.search_id, chunk_id=None, prng_base="java_lcg",
                              seed_start=0, seed_end_exclusive=1, reason_code="x",
                              reason_detail="x")
        credit_chunk(led, p, a, 0, T31)          # chunk-open + link rows
        con0 = sqlite3.connect(p)
        empty = [t for t in L.ALL_TABLES
                 if con0.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0]
        con0.close()
        ok_(not empty, f"fixture: tables still empty {empty}")
        blocked = []
        con = sqlite3.connect(p)
        con.execute("PRAGMA recursive_triggers = ON")
        for t in L.ALL_TABLES:
            for sql in (f"UPDATE {t} SET rowid = rowid", f"DELETE FROM {t}"):
                try:
                    con.execute(sql)
                    con.commit()
                    n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    if sql.startswith("DELETE") and n == 0 and t in (L.T_META, L.T_SEARCH):
                        raise Fail(f"{sql} succeeded")
                except sqlite3.DatabaseError as e:
                    if "append-only" in str(e):
                        blocked.append(f"{t}:{sql.split()[0]}")
        try:
            con.execute(f"INSERT OR REPLACE INTO {L.T_META}(key, value) VALUES ('schema_version','1')")
            con.commit()
            raise Fail("INSERT OR REPLACE succeeded on meta")
        except sqlite3.DatabaseError as e:
            ok_("append-only" in str(e), f"REPLACE refused for another reason: {e}")
        con.close()
        ok_(len(blocked) == 2 * len(L.ALL_TABLES), f"blocked only {blocked}")
        src = open(_MOD_PATHS["utils.step1_search_ledger"], encoding="utf-8").read()
        import ast
        import re
        tree = ast.parse(src)
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body \
                    and isinstance(node.body[0], ast.Expr) \
                    and isinstance(getattr(node.body[0], "value", None), ast.Constant):
                docs.add(id(node.body[0].value))
        sql = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]
        stmts = [m for s in sql for m in re.findall(
            r'(UPDATE\s+\{?\w+\}?\s+SET|DELETE\s+FROM\s+\{?\w+|INSERT\s+OR\s+REPLACE)', s)]
        ok_(not stmts, f"write statements other than INSERT in the module: {stmts}")
        return f"UPDATE/DELETE/REPLACE aborted on all {len(L.ALL_TABLES)} tables; no such statement in source"


def g3_exact_schema():
    with tmpdir("sch_") as d:
        p = os.path.join(d, "db")
        ledger_at(p)
        con = sqlite3.connect(p)
        con.execute("ALTER TABLE step1_search_event ADD COLUMN extra TEXT")
        con.commit()
        con.close()
        expect_raises("Step1SearchSchemaError", lambda: ledger_at(p))
        p2 = os.path.join(d, "db2")
        con = sqlite3.connect(p2)
        con.execute("CREATE TABLE step1_search_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        con.execute("INSERT INTO step1_search_schema_meta VALUES ('schema_version','99')")
        con.commit()
        con.close()
        expect_raises("Step1SearchSchemaError", lambda: ledger_at(p2))
        # [Beta R1 Q4] same NAME, wrong DEFINITION: an ineffective trigger body ...
        p3 = os.path.join(d, "db3")
        ledger_at(p3)
        con = sqlite3.connect(p3)
        con.execute("DROP TRIGGER step1_chunk_link_no_delete")
        con.execute("CREATE TRIGGER step1_chunk_link_no_delete BEFORE DELETE ON step1_chunk_link "
                    "BEGIN SELECT 1; END")
        con.commit()
        con.close()
        e = expect_raises("Step1SearchSchemaError", lambda: ledger_at(p3))
        ok_("step1_chunk_link_no_delete" in str(e), f"{e}")
        # ... and a same-named column with a different type.
        p4 = os.path.join(d, "db4")
        con = sqlite3.connect(p4)
        con.execute("CREATE TABLE step1_search_event (event_id TEXT PRIMARY KEY, search_id TEXT "
                    "NOT NULL, selector_key TEXT NOT NULL, event_type TEXT NOT NULL, reason TEXT "
                    "NOT NULL, recorded_utc TEXT NOT NULL, seq TEXT NOT NULL)")
        con.commit()
        con.close()
        e = expect_raises("Step1SearchSchemaError", lambda: ledger_at(p4))
        ok_("step1_search_event" in str(e), f"{e}")
        return ("extra column refused; wrong version refused; same-name ineffective trigger "
                "refused; same-name column of another type refused")


def _legacy(ok):
    """The legacy coordinator's result for one phase: all jobs successful (zero survivors),
    or the permanent-failure shape (coordinator.py:2193-2222) Beta used in R1 Q1."""
    def f(args):
        if ok:
            return {"metadata": {"total_jobs": 1, "successful_jobs": 1, "failed_jobs": 0},
                    "results": [], "failed_jobs": []}
        return {"metadata": {"total_jobs": 1, "successful_jobs": 0, "failed_jobs": 1},
                "results": [], "failed_jobs": [("j", "worker timeout")]}
    return f


def g1_legacy_failed_jobs_not_credited():
    """[Beta R1 Q1, LIVE] Through the REAL legacy evidence producer (run_bidirectional_test
    is not replaced) and the real publication caller: permanently failed jobs on every call
    earn NO credit; the paired control (all jobs successful, zero survivors) IS credited."""
    out = []
    with integ_env() as d:
        led = ledger_at("prng_analysis.db")
        rec, _ = open_simple(led, trials=1, chunk=1000, both=False)
        beh = Behaviour([])
        expect_raises("Step1NoCreditError", lambda: run_chunk(
            beh, seed_start=0, seed_count=1000, trials=1, search_id=rec.search_id, both=False,
            legacy_result=_legacy(False)))
        ok_(len([c for c in beh.calls if c["kind"] == "legacy_call"]) >= 2,
            f"the real legacy route did not run both phases: {beh.calls}")
        ok_(coverage_rows(d) == [] and search_rows(d, "step1_chunk_link") == [],
            "failed jobs certified")
        unc = search_rows(d, "step1_uncredited_attempt")
        ok_(unc and unc[-1]["reason_code"] == "no_sweep_evidence", f"uncredited: {unc[-1:]}")
        ok_(led.select("java_lcg", ["constant"]).state == "gap", "search not still open at its gap")
        out.append("failed jobs -> no credit, no row, no link, search still at its gap")
    with integ_env() as d:
        led = ledger_at("prng_analysis.db")
        rec, _ = open_simple(led, trials=1, chunk=1000, both=False)
        run_chunk(Behaviour([]), seed_start=0, seed_count=1000, trials=1,
                  search_id=rec.search_id, both=False, legacy_result=_legacy(True))
        ok_(len(coverage_rows(d)) == 1 and len(search_rows(d, "step1_chunk_link")) == 1,
            "control not credited")
        out.append("control: all jobs successful, 0 survivors -> credited")
    return "; ".join(out)


def g1_pwc_zmq_refused():
    """[Beta R1 Q1, LIVE] PWC/ZMQ carry no completion accounting to the trial: their
    evidence has no swept interval, so S7 never credits them in stage 1. Unit-level on the
    real evidence builder, plus the live source shows both branches use it."""
    import ast
    import window_optimizer_integration_final as W
    from utils.step1_search_ledger import credit_decision
    for route in ("pwc", "zmq"):
        ev = W._step1_unaccounted_route_evidence(route, "suite")
        okc, code, det, _ = credit_decision(requested_interval=(0, 1000),
                                            required_phases=(1, 2, 3, 4),
                                            evidence_by_trial={0: ev},
                                            trial_states={0: "COMPLETE"})
        ok_(not okc and code == "no_sweep_evidence", f"{route}: {code} {det}")
    src = open(_MOD_PATHS["window_optimizer_integration_final"], encoding="utf-8").read()
    for var in ("_pw_tr", "_zmq_tr"):
        ok_(f"{var}.step1_sweep_evidence = _step1_unaccounted_route_evidence(" in src,
            f"{var} does not use the refusal evidence")
    return "PWC and ZMQ evidence has no swept interval -> no credit; both branches wired"


def _gen_dirs(d):
    return sorted(dp for dp, dn, fn in os.walk(d) if os.path.basename(dp).startswith("gen-"))


def q5_child(db_dir, mutant=None):
    """Runs in a FRESH PROCESS: WATCHER's real run_step(1) against db_dir, dispatch
    intercepted. Prints one JSON line."""
    ctx = contextlib.nullcontext()
    if mutant == "select_repair_off":
        ctx = mutated(L_MOD,
                      "                    rep = tuple(r[\"coverage_id\"] for r in self.repair_pending_links(rec.search_id))\n",
                      "                    rep = ()\n")
    elif mutant == "reconstruct_off":
        ctx = mutated(L_MOD, "            return out + self._reconstruct_links(search_id)\n",
                      "            return out\n")
    with ctx:
        with watcher_env(db_dir=db_dir) as (W, cap, dbp, d):
            agent = make_agent(W, d)
            res = run_step1(W, agent, BASE_PARAMS)
            print("Q5CHILD " + json.dumps({"result": res.get(W.STEP1_SEARCH_RESULT_KEY),
                                           "blocked_by": res.get("blocked_by"),
                                           "argv": len(cap["argv"])}), flush=True)


def g4_link_repair_fresh_process(child_mutant=None, failure="link"):
    """[Beta R1 Q5, LIVE] Publication succeeds, the link write fails (transient). A FRESH
    PROCESS through WATCHER's supported entry (run_step) repairs exactly the missing link:
    zero objective calls, no dispatch, no replacement generation, exactly one link, and the
    one-chunk search is complete. The child's own preflight repair is shown too."""
    import subprocess
    from utils import step1_search_ledger as L
    with integ_env() as d:
        led = ledger_at("prng_analysis.db")
        rec, _ = open_simple(led, start=0, end=1000, trials=1, chunk=1000)
        real_link = L.Step1SearchLedger.record_chunk_link

        def transient(self, **k):
            raise L.Step1LinkError("suite: injected transient link failure")

        real_ins = L.Step1SearchLedger._insert_idempotent

        def pending_fails(self, table, key_col, row):
            # Beta R2's injection: the pending INSERT fails AFTER its real validation
            if table == L.T_PENDING:
                raise sqlite3.OperationalError("suite: injected transient pending-write failure")
            return real_ins(self, table, key_col, row)

        if failure == "link":
            L.Step1SearchLedger.record_chunk_link = transient
            expected_exc, expected_state = "Step1LinkError", (1, 1, 0)
        else:
            L.Step1SearchLedger._insert_idempotent = pending_fails
            expected_exc, expected_state = "OperationalError", (1, 0, 0)
        try:
            beh = Behaviour(["ok"])
            expect_raises(expected_exc, lambda: run_chunk(
                beh, seed_start=0, seed_count=1000, trials=1, search_id=rec.search_id))
        finally:
            L.Step1SearchLedger.record_chunk_link = real_link
            L.Step1SearchLedger._insert_idempotent = real_ins
        ok_((len(coverage_rows(d)), len(search_rows(d, "step1_link_pending")),
             len(search_rows(d, "step1_chunk_link"))) == expected_state,
            f"precondition: cov/pending/link = {len(coverage_rows(d))}/"
            f"{len(search_rows(d, 'step1_link_pending'))}/{len(search_rows(d, 'step1_chunk_link'))}")
        gens = _gen_dirs(d)
        argv = [sys.executable, os.path.abspath(__file__), "--q5-child", d]
        if child_mutant:
            argv.append(child_mutant)
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1",
                   CUDA_VISIBLE_DEVICES="")
        pr = subprocess.run(argv, capture_output=True, text=True, timeout=300, env=env, cwd=ROOT)
        line = [l for l in pr.stdout.splitlines() if l.startswith("Q5CHILD ")]
        ok_(line, f"child produced no result: rc={pr.returncode} {pr.stderr[-500:]}")
        got = json.loads(line[-1][8:])
        ok_(got["result"] == "search_complete" and got["argv"] == 0,
            f"fresh process did not repair to completion: {got}")
        ok_(len(search_rows(d, "step1_chunk_link")) == 1 and len(coverage_rows(d)) == 1,
            "not exactly one link / one coverage row")
        ok_(_gen_dirs(d) == gens, "a replacement generation appeared")
        lk = search_rows(d, "step1_chunk_link")[0]
        want_tc = 1 if failure == "link" else -1      # -1: counts not recorded (reconstructed)
        ok_(lk["trials_complete"] == want_tc and (lk["seed_start"], lk["seed_end_exclusive"]) == (0, 1000),
            f"repaired link content: {lk}")
        # the child-preflight route: a stale re-dispatch of the published chunk is refused
        beh2 = Behaviour(["ok"])
        expect_raises("Step1SearchPreflightError", lambda: run_chunk(
            beh2, seed_start=0, seed_count=1000, trials=1, search_id=rec.search_id))
        ok_(beh2.calls == [] and len(search_rows(d, "step1_chunk_link")) == 1,
            f"stale re-dispatch ran trials or relinked: {beh2.calls}")
    return ("fresh process: search_complete, 0 dispatch, 1 link, no new generation; stale "
            "re-dispatch refused before any trial")


def g4_link_repair_after_failed_pending_write():
    """[Beta R2 Q2, LIVE] The pending INSERT itself fails after the history commit
    (coverage 1, pending 0, link 0). A fresh process via WATCHER reconstructs the link from
    the chunk-open + the history row (study_identity): search complete, 0 dispatch, exactly
    one link, no replacement generation."""
    return g4_link_repair_fresh_process(failure="pending")


def g4_reconstruction_binds_only_this_chunk():
    """The reconstruction binds a history row only by interval + family + modes + dataset
    + study_identity == the chunk's study + recorded after the chunk-open; a foreign row
    (another study) is never linked; two matching rows refuse."""
    from utils.seed_coverage_ledger import CoverageLedger
    with tmpdir("rec_") as d:
        dbp = os.path.join(d, "db")
        led = ledger_at(dbp)
        rec, _ = open_simple(led, start=0, end=1000, chunk=1000,
                             dataset=dict(dataset_fields(), dataset_sha256="b" * 64))
        led.chunk_open(search_id=rec.search_id, seed_start=0, seed_end_exclusive=1000,
                       study_name="study-own", study_trials_at_open=0, budget_target=1,
                       checkpoint_run_id="ck")
        history_row(dbp, fake_witness(0, 1000, gen="g-foreign", sha="1" * 64), study="study-other")
        history_row(dbp, fake_witness(0, 1000, gen="g-nostudy", sha="2" * 64), study=None)
        ok_(led.repair_pending_links(rec.search_id) == [], "a foreign row was linked")
        ok_(led.cursor(rec).next_seed_start == 0, "the cursor moved on a foreign row")
        history_row(dbp, fake_witness(0, 1000, gen="g-own", sha="3" * 64), study="study-own")
        got = led.repair_pending_links(rec.search_id)
        ok_(len(got) == 1 and got[0]["generation_id"] == "g-own", f"own row not linked: {got}")
        # ambiguity: a second search chunk with two matching rows refuses
        led.close_search(rec.search_id, "suite")
        r2, _ = open_simple(led, start=0, end=1000, chunk=1000, trials=2,
                            dataset=dict(dataset_fields(), dataset_sha256="b" * 64))
        led.chunk_open(search_id=r2.search_id, seed_start=0, seed_end_exclusive=1000,
                       study_name="study-two", study_trials_at_open=0, budget_target=2,
                       checkpoint_run_id="ck")
        history_row(dbp, fake_witness(0, 1000, gen="g-a", sha="4" * 64), study="study-two")
        history_row(dbp, fake_witness(0, 1000, gen="g-b", sha="5" * 64), study="study-two")
        expect_raises("Step1LinkError", lambda: led.repair_pending_links(r2.search_id))
    return "foreign-study and study-less rows not linked; own row linked; two matches refused"


def g4_link_repair_in_child_preflight():
    """[Beta R1 Q5] With no WATCHER lookup in between, the child's own preflight repairs
    the stranded link and then refuses the stale chunk — zero objective calls."""
    from utils import step1_search_ledger as L
    with integ_env() as d:
        led = ledger_at("prng_analysis.db")
        rec, _ = open_simple(led, start=0, end=2000, trials=1, chunk=1000)
        real_link = L.Step1SearchLedger.record_chunk_link
        L.Step1SearchLedger.record_chunk_link = lambda self, **k: (_ for _ in ()).throw(
            L.Step1LinkError("suite: transient"))
        try:
            expect_raises("Step1LinkError", lambda: run_chunk(
                Behaviour(["ok"]), seed_start=0, seed_count=1000, trials=1,
                search_id=rec.search_id))
        finally:
            L.Step1SearchLedger.record_chunk_link = real_link
        beh = Behaviour(["ok"])
        e = expect_raises("Step1SearchPreflightError", lambda: run_chunk(
            beh, seed_start=0, seed_count=1000, trials=1, search_id=rec.search_id))
        ok_(beh.calls == [] and len(search_rows(d, "step1_chunk_link")) == 1
            and "first gap=1000" in str(e), f"{beh.calls} {e}")
    return "child preflight repaired the link, then refused the stale chunk (gap now 1000)"


# ===========================================================================
# G4 — PUBLICATION AND LINK
# ===========================================================================
def g4_link_failure_surfaces():
    """A link write that fails after publication surfaces, does not advance the search,
    and a retry with the same witness links exactly once."""
    from utils import step1_search_ledger as L
    import utils.run_finalizer as RF
    import database_system as DS
    with integ_env() as d:
        led = ledger_at("prng_analysis.db")
        rec, _ = open_simple(led, trials=1, chunk=1000)
        held = {}
        real_fr, real_rcc, real_link = (RF.finalize_run, DS.DistributedPRNGDatabase.record_certified_coverage,
                                        L.Step1SearchLedger.record_chunk_link)

        def fr(*a, **k):
            held["artifact"] = real_fr(*a, **k)
            return held["artifact"]

        def rcc(self, *a, **k):
            held["coverage"] = real_rcc(self, *a, **k)
            return held["coverage"]

        def boom(self, **k):
            held["chunk_id"] = k["chunk_id"]
            raise L.Step1LinkError("suite: injected link failure")

        RF.finalize_run = fr
        DS.DistributedPRNGDatabase.record_certified_coverage = rcc
        L.Step1SearchLedger.record_chunk_link = boom
        try:
            expect_raises("Step1LinkError", lambda: run_chunk(
                Behaviour(["ok"]), seed_start=0, seed_count=1000, trials=1,
                search_id=rec.search_id))
        finally:
            RF.finalize_run = real_fr
            DS.DistributedPRNGDatabase.record_certified_coverage = real_rcc
            L.Step1SearchLedger.record_chunk_link = real_link
        ok_(len(coverage_rows(d)) == 1, "the publication did not happen (precondition)")
        led2 = ledger_at("prng_analysis.db")
        ok_(led2.cursor(rec).next_seed_start == 0, "the search advanced without a link")
        for _ in range(2):
            led2.record_chunk_link(search_id=rec.search_id, chunk_id=held["chunk_id"],
                                   artifact=held["artifact"], coverage=held["coverage"],
                                   trial_counts={"complete": 1})
        ok_(len(search_rows(d, "step1_chunk_link")) == 1, "retry did not link exactly once")
        ok_(led2.cursor(rec).next_seed_start == 1000, "the retry did not advance the search")
        return "failure surfaced, cursor stayed 0; retry x2 with same witness -> 1 link, cursor 1000"


def g4_wrong_witness_refuses():
    from utils.seed_coverage_ledger import CertifiedInterval
    import dataclasses
    with tmpdir("lnk_") as d:
        dbp = os.path.join(d, "db")
        led = ledger_at(dbp)
        rec, _ = open_simple(led, chunk=1000)
        ch = led.chunk_open(search_id=rec.search_id, seed_start=0, seed_end_exclusive=1000,
                            study_name="s", study_trials_at_open=0, budget_target=1,
                            checkpoint_run_id="ck")
        w = fake_witness(0, 1000)
        cov = history_row(dbp, w)
        out = []
        cases = {
            "generation": (w, dataclasses.replace(cov, generation_id="gen-other")),
            "mode": (fake_witness(0, 1000, modes=("constant",), gen="g2", sha="c" * 64), None),
            "interval": (fake_witness(0, 999, gen="g3", sha="d" * 64), None),
            "family": (fake_witness(0, 1000, prng="xorshift32", gen="g4", sha="e" * 64), None),
        }
        for name, (wit, covr) in cases.items():
            covr = covr or history_row(dbp, wit)
            try:
                led.record_chunk_link(search_id=rec.search_id, chunk_id=ch["chunk_id"],
                                      artifact=wit, coverage=covr, trial_counts={})
                raise Fail(f"{name}: wrong witness LINKED")
            except Exception as e:      # noqa: BLE001
                if isinstance(e, Fail):
                    raise
                ok_(type(e).__name__ == "Step1LinkError", f"{name}: {type(e).__name__}: {e}")
                out.append(name)
        expect_raises("PublicationBindingError", lambda: led.record_chunk_link(
            search_id=rec.search_id, chunk_id=ch["chunk_id"],
            artifact=types.SimpleNamespace(artifact_sha256="a" * 64), coverage=cov,
            trial_counts={}))
        return f"refused: {out} + a non-witness"


def g4_dataset_mismatch_refuses_in_child():
    """The search was opened on another dataset version: the child refuses before any
    study or dispatch, naming both versions."""
    with integ_env() as d:
        led = ledger_at("prng_analysis.db")
        other = dict(dataset_fields(), dataset_version_id="daily3-OTHER", dataset_sha256="f" * 64)
        rec, _ = open_simple(led, trials=1, chunk=1000, dataset=other)
        beh = Behaviour(["ok"])
        e = expect_raises("Step1SearchPreflightError", lambda: run_chunk(
            beh, seed_start=0, seed_count=1000, trials=1, search_id=rec.search_id))
        ok_(beh.calls == [] and "daily3-OTHER" in str(e) and frozen_dataset().version_id in str(e),
            f"{e}")
        ok_(coverage_rows(d) == [], "coverage written")
        return "dataset mismatch refused before dispatch, both versions named"


def g4_endpoints_gaps_overlaps():
    from utils.step1_search_ledger import first_uncovered_seed
    with tmpdir("ep_") as d:
        dbp = os.path.join(d, "db")
        led = ledger_at(dbp)
        r1, _ = open_simple(led, start=0, end=T32, chunk=T31)
        ok_(led.cursor(r1).next_seed_start == 0, "fresh search cursor is not 0")
        credit_chunk(led, dbp, r1, 0, T31)
        ok_(led.cursor(r1).next_seed_start == T31, "after [0,2^31) cursor is not 2^31")
        credit_chunk(led, dbp, r1, T31, T32)
        ok_(led.cursor(r1).is_complete, "[0,2^32) not complete")
        ok_(led.select("java_lcg", ["constant", "variable"]).state == "search_complete", "state")
        led.close_search(r1.search_id, "suite")
        r2, _ = open_simple(led, start=T31, end=T32, chunk=T31)
        ok_(led.cursor(r2).next_seed_start == T31, "upper-half search does not start at 2^31")
        expect_raises("Step1SearchRefused", lambda: led.open_search(
            configuration=cfg_for(simple_final()), range_start=0, range_end_exclusive=T32 + 1,
            dataset=dataset_fields(), opened_reason="operator", opened_by_ref="x",
            repository_commit="x"))
        c = first_uncovered_seed([(0, 10), (20, 30), (5, 12)], domain_start=0,
                                 domain_end_exclusive=40)
        ok_(c.next_seed_start == 12, f"gap/overlap cursor {c.next_seed_start}")
        return "cursor 0 -> 2^31 -> COMPLETE at 2^32; upper search starts 2^31; end 2^32+1 refused; gap after overlap = 12"


# ===========================================================================
# G5 — EFFECTIVE INPUT
# ===========================================================================
BASE_PARAMS = {"prng_type": "java_lcg", "max_seeds": 1000, "window_trials": 1,
               "test_both_modes": True, "use_persistent_workers": False}


def _open_and_run(W, cap, d, open_params, run_params, *, pin_open=None, pin_run=None,
                  start=0, end=T32):
    agent = make_agent(W, d)
    rc, out = cli_open(W, agent, {**open_params, **(pin_open or {})}, start=start, end=end)
    ok_(rc == 0 and "STEP1_SEARCH_OPENED" in out, f"open failed rc={rc}: {out}")
    pin = W.capture_step1_pin_bundle(pin_run) if pin_run else None
    return run_step1(W, agent, run_params, pin), out


def g5_each_field_refuses():
    # The mode set SELECTS the search (one open search per family and mode set), so a
    # different mode set finds no search: no_open_search, still nothing dispatched.
    with watcher_env() as (W, cap, dbp, d):
        res, _ = _open_and_run(W, cap, d, BASE_PARAMS, {**BASE_PARAMS, "test_both_modes": False})
        ok_(res.get("blocked_by") == "step1_no_open_search" and not cap["argv"],
            f"test_both_modes changed: {res.get('blocked_by')}")
    mutations = [("window_trials", 2), ("max_seeds", 999),
                 ("enable_pruning", False), ("n_parallel", 2), ("strategy", "random")]
    refused = []
    for key, val in mutations:
        with watcher_env() as (W, cap, dbp, d):
            res, _ = _open_and_run(W, cap, d, BASE_PARAMS, {**BASE_PARAMS, key: val})
            ok_(res.get("blocked_by") == "step1_search_mismatch" and not cap["argv"],
                f"{key}={val!r} was not refused: {res.get('blocked_by')} {res.get('error')}")
            ok_("presented=" in res["error"] and "recorded=" in res["error"],
                f"{key}: both values not named: {res['error'][:200]}")
            refused.append(key)
    for k in RUN4_PINS:
        with watcher_env() as (W, cap, dbp, d):
            bad = dict(RUN4_PINS)
            bad[k] = bad[k] + (1 if isinstance(bad[k], int) else 0.01)
            res, _ = _open_and_run(W, cap, d, BASE_PARAMS, BASE_PARAMS,
                                   pin_open=RUN4_PINS, pin_run=bad)
            ok_(res.get("blocked_by") == "step1_search_mismatch" and not cap["argv"],
                f"pin {k} changed was not refused: {res.get('blocked_by')}")
            refused.append(k)
    import window_optimizer as WO
    for label, patch in (("search bounds", "bounds"), ("session options", "sessions")):
        with watcher_env() as (W, cap, dbp, d):
            agent = make_agent(W, d)
            rc, _ = cli_open(W, agent, BASE_PARAMS, start=0, end=T32)
            saved = (WO.load_search_bounds_from_config, WO.SearchBounds.__post_init__)
            if patch == "bounds":
                real = saved[0]
                WO.load_search_bounds_from_config = lambda p="": {
                    **real(p), "window_size": {"min": 2, "max": 49}}
            else:
                def pi(self):
                    saved[1](self)
                    self.session_options = [["midday"], ["midday", "evening"], ["evening"]]
                WO.SearchBounds.__post_init__ = pi
            try:
                res = run_step1(W, agent, BASE_PARAMS)
            finally:
                WO.load_search_bounds_from_config, WO.SearchBounds.__post_init__ = saved
            ok_(rc == 0 and res.get("blocked_by") == "step1_search_mismatch" and not cap["argv"],
                f"{label} change not refused: {res.get('blocked_by')}")
            refused.append(label)
    return f"{len(refused)} single-field mutations each refused: {refused}"


def g5_dataset_pointer_moved():
    from miner import dataset_authority as DA
    real = frozen_dataset()
    moved = types.SimpleNamespace(**{**real.__dict__, "version_id": "daily3-MOVED",
                                     "sha256": "9" * 64}) if hasattr(real, "__dict__") else None
    import dataclasses
    moved = dataclasses.replace(real, version_id="daily3-MOVED", sha256="9" * 64)
    with watcher_env() as (W, cap, dbp, d):
        agent = make_agent(W, d)
        rc, _ = cli_open(W, agent, BASE_PARAMS, start=0, end=T32)
        saved = W.p05_freeze_dataset
        W.p05_freeze_dataset = lambda run_label="watcher": moved
        try:
            res = run_step1(W, agent, BASE_PARAMS)
        finally:
            W.p05_freeze_dataset = saved
        e = res.get("error", "")
        ok_(rc == 0 and res.get("blocked_by") == "step1_search_mismatch" and not cap["argv"],
            f"moved pointer not refused: {res.get('blocked_by')}")
        ok_("daily3-MOVED" in e and real.version_id in e, f"both versions not named: {e[:300]}")
        return "pointer moved after opening -> refused, naming both versions"


def g5_search_id_three_hops():
    """Hop 1: the manifest declares it. WATCHER-shaped params dict -> built argv carries
    it. Hop 2: window_optimizer.main parses that argv. Hop 3: optimize_window receives it."""
    m = json.load(open(os.path.join(ROOT, "agent_manifests", "window_optimizer.json")))
    ok_(m["default_params"].get("step1_search_id") == "", "hop 1: default_params")
    ok_(m["actions"][0]["args_map"].get("step1-search-id") == "step1_search_id", "hop 1: args_map")
    with watcher_env() as (W, cap, dbp, d):
        agent = make_agent(W, d)
        rc, out = cli_open(W, agent, BASE_PARAMS, start=0, end=T32)
        sid = out.split()[1]
        run_step1(W, agent, BASE_PARAMS)
        ok_(cap["argv"], "no argv built")
        argv = cap["argv"][-1]
        ok_(argv_value(argv, "--step1-search-id") == sid, f"argv: {argv}")
    import window_optimizer as WO
    import window_optimizer_integration_final as INT
    import coordinator as C
    from miner import dataset_authority as DA
    got = {}

    class FakeCoord:
        def __init__(self, *a, **k):
            pass

        def optimize_window(self, **kw):
            got.update(kw)
            raise SystemExit(0)

    import execution_set as XS
    xstub = types.SimpleNamespace(remote_execution=False, describe=lambda: "suite stub set")
    saved_x = (XS.resolve_execution_set, XS.freeze_execution_set)
    XS.resolve_execution_set = lambda **k: xstub
    XS.freeze_execution_set = lambda s: s
    saved = (C.MultiGPUCoordinator, WO.MultiGPUCoordinator, INT.add_window_optimizer_to_coordinator,
             DA.run_start_dataset_gate, sys.argv)
    C.MultiGPUCoordinator = WO.MultiGPUCoordinator = FakeCoord
    INT.add_window_optimizer_to_coordinator = lambda: None
    DA.run_start_dataset_gate = lambda *a, **k: frozen_dataset()
    child_argv = [a for a in argv[2:]]
    sys.argv = ["window_optimizer.py"] + child_argv
    buf, ebuf = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(ebuf):
            try:
                WO.main()
            except SystemExit:
                pass
    finally:
        (C.MultiGPUCoordinator, WO.MultiGPUCoordinator, INT.add_window_optimizer_to_coordinator,
         DA.run_start_dataset_gate, sys.argv) = saved
        XS.resolve_execution_set, XS.freeze_execution_set = saved_x
    ok_(got.get("step1_search_id") == sid,
        f"hop 3 got {got.get('step1_search_id')!r}; stderr={ebuf.getvalue()[-300:]!r} "
        f"stdout={buf.getvalue()[-300:]!r}")
    return f"manifest -> argv -> window_optimizer.main -> optimize_window(step1_search_id={sid[:16]}...)"


def g5_search_id_missing_refuses():
    """S6: a Step 1 command built WITHOUT the id refuses. Shown with WATCHER's own
    assignment removed — the refusal on the built command must fire."""
    from contextlib import ExitStack
    with mutated("agents.watcher_agent",
                 "            final_params['step1_search_id'] = _s1_rec.search_id\n",
                 "            final_params['step1_search_id'] = ''\n"):
        with watcher_env() as (W, cap, dbp, d):
            agent = make_agent(W, d)
            rc, _ = cli_open(W, agent, BASE_PARAMS, start=0, end=T32)
            res = run_step1(W, agent, BASE_PARAMS)
            ok_(res.get("blocked_by") == "step1_search_id_not_routed" and not cap["argv"],
                f"a command without the id was not refused: {res.get('blocked_by')} argv={cap['argv']}")
    return "id dropped before the builder -> step1_search_id_not_routed, nothing dispatched"


# ===========================================================================
# G6 — AUTHORITY AND FRESHNESS
# ===========================================================================
def _pipeline(W, agent, params, pin=None):
    obs = {"escalate": 0}
    real = agent._handle_escalate

    def esc(*a, **k):
        obs["escalate"] += 1
        return real(*a, **k)

    agent._handle_escalate = esc
    with quiet():
        agent.run_pipeline(start_step=1, end_step=2, params=dict(params),
                           _operator_pin_params=pin)
    obs["halt"] = os.path.exists(agent.config.halt_file)
    obs["step"] = agent.current_step
    return obs


def g6_three_results():
    out = []
    with watcher_env() as (W, cap, dbp, d):
        agent = make_agent(W, d)
        res = run_step1(W, agent, BASE_PARAMS)
        ok_(res.get(W.STEP1_SEARCH_RESULT_KEY) == "no_open_search" and not cap["argv"], f"{res}")
        o = _pipeline(W, agent, BASE_PARAMS)
        ok_(o["escalate"] == 0 and not o["halt"] and o["step"] == 1 and not cap["argv"], f"{o}")
        ok_(run_step1(W, agent, BASE_PARAMS).get(W.STEP1_SEARCH_RESULT_KEY) == "no_open_search",
            "not durable")
        out.append("no_open_search: clean stop, durable")
    with watcher_env() as (W, cap, dbp, d):
        agent = make_agent(W, d)
        cli_open(W, agent, {**BASE_PARAMS, "max_seeds": T31}, start=T31, end=T32)
        import database_system as DS
        led = DS.DistributedPRNGDatabase().step1_search_ledger()
        rec = led.select("java_lcg", ["constant", "variable"]).search
        credit_chunk(led, dbp, rec, T31, T32)
        for _ in range(2):
            res = run_step1(W, agent, {**BASE_PARAMS, "max_seeds": T31})
            ok_(res.get(W.STEP1_SEARCH_RESULT_KEY) == "search_complete" and not cap["argv"], f"{res}")
        o = _pipeline(W, agent, {**BASE_PARAMS, "max_seeds": T31})
        ok_(o["escalate"] == 0 and not o["halt"] and o["step"] == 1, f"{o}")
        out.append("search_complete: clean stop, durable")
    with watcher_env() as (W, cap, dbp, d):
        agent = make_agent(W, d)
        cli_open(W, agent, BASE_PARAMS, start=0, end=T32)
        con = sqlite3.connect(dbp)
        con.execute("ALTER TABLE step1_search_event ADD COLUMN corrupt TEXT")
        con.commit()
        con.close()
        res = run_step1(W, agent, BASE_PARAMS)
        ok_(res.get("blocked_by") == "step1_search_lookup_failed" and not cap["argv"], f"{res}")
        o = _pipeline(W, agent, BASE_PARAMS)
        ok_(o["escalate"] >= 1 and o["halt"], f"a lookup failure did not block/escalate: {o}")
        out.append("lookup failure: blocked, escalated, no command")
    return "; ".join(out)


def g6_no_default_search_on_error():
    import database_system as DS
    with watcher_env() as (W, cap, dbp, d):
        agent = make_agent(W, d)
        saved = DS.DistributedPRNGDatabase.select_step1_search

        def broken(self, *a, **k):
            raise sqlite3.OperationalError("suite: disk I/O error")

        DS.DistributedPRNGDatabase.select_step1_search = broken
        try:
            res = run_step1(W, agent, BASE_PARAMS)
        finally:
            DS.DistributedPRNGDatabase.select_step1_search = saved
        ok_(res.get("blocked_by") == "step1_search_lookup_failed" and not cap["argv"], f"{res}")
        led = DS.DistributedPRNGDatabase().step1_search_ledger()
        n = len(search_rows(d, "step1_search"))
        ok_(n == 0, "a search was opened by default")
        return "lookup error -> blocked, no command, no seed 0, no default search"


def g6_no_opening_from_params_record_env():
    with watcher_env() as (W, cap, dbp, d):
        agent = make_agent(W, d)
        os.environ["STEP1_SEARCH_OPEN"] = "1"
        os.environ["STEP1_SEARCH_ID"] = "s1s-forged"
        try:
            params = {**BASE_PARAMS, "step1_search_open": True, "search_range_start": 0,
                      "search_range_end": T32, "step1_search_id": "s1s-forged"}
            res = run_step1(W, agent, params)
            _pipeline(W, agent, params)
        finally:
            os.environ.pop("STEP1_SEARCH_OPEN", None)
            os.environ.pop("STEP1_SEARCH_ID", None)
        n = len(search_rows(d, "step1_search"))
        ok_(n == 0 and res.get(W.STEP1_SEARCH_RESULT_KEY) == "no_open_search" and not cap["argv"],
            f"opened={n} res={res.get('blocked_by')}")
        return "params/env carrying an opening or a forged id opened nothing; no command"


def g6_no_pins_inherited():
    """A pinned search; a run WITHOUT operator pins must refuse — the record's pins are
    never reconstructed into the bundle."""
    with watcher_env() as (W, cap, dbp, d):
        res, _ = _open_and_run(W, cap, d, BASE_PARAMS, BASE_PARAMS, pin_open=RUN4_PINS,
                               pin_run=None)
        ok_(res.get("blocked_by") == "step1_search_mismatch" and not cap["argv"], f"{res}")
        ok_("search_space" in res["error"], "the mismatch is not on the search space")
        return "pinned record + unpinned run -> refused; no pin reconstructed from the record"


def g6_step1_due_when_fresh():
    with watcher_env(fresh=True) as (W, cap, dbp, d):
        agent = make_agent(W, d)
        cli_open(W, agent, BASE_PARAMS, start=0, end=T32)
        res = run_step1(W, agent, BASE_PARAMS)
        ok_(cap["argv"] and not res.get("skipped"), f"Step 1 was skipped on fresh output: {res}")
        with quiet():
            r2 = agent.run_step(2, {})
        ok_(r2.get("skipped") is True, f"step 2 lost its freshness check: {r2}")
        return "Step 1 with a gap ran despite fresh output; Step 2 still freshness-skipped"


# ===========================================================================
# F5a, RUN4 SHAPE, G-MINER-START
# ===========================================================================
def f5a():
    """History COMPLETE over [0, 2^32): no open search -> clean stop; operator opens a
    search -> a Step 1 command at its first gap; no existing row changed or deleted."""
    with watcher_env() as (W, cap, dbp, d):
        for s, e in ((0, T31), (T31, T32)):
            history_row(dbp, fake_witness(s, e, gen=f"hist-{s}", sha=hashlib.sha256(
                str(s).encode()).hexdigest()))
        import database_system as DS
        cur = DS.DistributedPRNGDatabase().get_certified_cursor("java_lcg", test_both_modes=True)
        ok_(cur.is_complete, "fixture: history is not COMPLETE")
        before = table_rows(dbp)
        agent = make_agent(W, d)
        o = _pipeline(W, agent, BASE_PARAMS)
        ok_(not cap["argv"] and o["escalate"] == 0 and not o["halt"], f"no-search stop: {o}")
        rc, out = cli_open(W, agent, BASE_PARAMS, start=0, end=T32, reason="regime rerun (suite)")
        ok_(rc == 0, out)
        run_step1(W, agent, BASE_PARAMS)
        ok_(cap["argv"], "no command after opening")
        argv = cap["argv"][-1]
        ok_(argv_value(argv, "--seed-start") == "0" and argv_value(argv, "--step1-search-id") == out.split()[1],
            f"argv start {argv_value(argv, '--seed-start')}")
        after = table_rows(dbp)
        for t, rows in before.items():
            ok_(all(r in after.get(t, []) for r in rows), f"a row of {t} changed or vanished")
        ok_(rows_hash({t: before[t] for t in before}) ==
            rows_hash({t: [r for r in after[t] if r in before[t]] for t in before}), "content hash")
        counts = {t: (len(before[t]), len(after.get(t, []))) for t in before}
        return f"history COMPLETE; stop, then command at 0; pre-existing rows intact {counts}"


def run4_shape():
    with watcher_env() as (W, cap, dbp, d):
        history_row(dbp, fake_witness(0, T31, gen="gen-attempt9-like"))
        agent = make_agent(W, d)
        rc, out = cli_open(W, agent, {**RUN4_BASE, **RUN4_PINS}, start=T31, end=T32,
                           reason="Run 4 (suite)")
        ok_(rc == 0 and "space=pinned" in out, out)
        r4 = run_step1(W, agent, RUN4_BASE, W.capture_step1_pin_bundle(RUN4_PINS))
        ok_(cap["argv"], f"no Run 4 command: {r4.get('blocked_by')} {str(r4.get('error'))[:300]}")
        argv = cap["argv"][-1]
        ok_(argv_value(argv, "--seed-start") == str(T31) and argv_value(argv, "--max-seeds") == str(T31),
            f"start/size: {argv_value(argv, '--seed-start')} {argv_value(argv, '--max-seeds')}")
        for k, v in RUN4_PINS.items():
            f = "--" + k.replace("_", "-")
            ok_(argv_value(argv, f) == str(v), f"pin {k}: {argv_value(argv, f)} != {v}")
        bad = {**RUN4_PINS, "warm_start_offset": 0}
        cap["argv"].clear()
        res = run_step1(W, agent, RUN4_BASE, W.capture_step1_pin_bundle(bad))
        ok_(res.get("blocked_by") == "step1_search_mismatch" and not cap["argv"], "different pins")
        import window_optimizer as WO
        e = expect_raises("Step1MinerStartUnsupportedError",
                          lambda: WO.assert_miner_seed_start_supported(
                              int(argv_value(argv, "--seed-start")), "--use-range-miner" in argv,
                              context="run4 argv"))
        return ("--seed-start 2147483648 --max-seeds 2147483648 + 7 pins; different pins refused; "
                "the child's G-MINER-START refuses this argv until the base_start brief lands")


def g_miner_start():
    import window_optimizer as WO
    import coordinator as C

    class Trip:
        def __init__(self, *a, **k):
            raise RuntimeError("TRIPWIRE: coordinator constructed")

    saved = (C.MultiGPUCoordinator, WO.MultiGPUCoordinator, WO.COORDINATOR_AVAILABLE)
    C.MultiGPUCoordinator = WO.MultiGPUCoordinator = Trip
    WO.COORDINATOR_AVAILABLE = True
    try:
        with quiet():
            e = expect_raises("Step1MinerStartUnsupportedError", lambda: WO.run_bayesian_optimization(
                lottery_file="/nonexistent.json", trials=1, output_config="/dev/null",
                seed_count=1000, seed_start=T31, use_range_miner=True))
            for ss, miner in ((0, True), (T31, False)):
                try:
                    WO.run_bayesian_optimization(
                        lottery_file="/nonexistent.json", trials=1, output_config="/dev/null",
                        seed_count=1000, seed_start=ss, use_range_miner=miner)
                except RuntimeError as x:
                    ok_("TRIPWIRE" in str(x), f"seed_start={ss} miner={miner}: {x}")
    finally:
        C.MultiGPUCoordinator, WO.MultiGPUCoordinator, WO.COORDINATOR_AVAILABLE = saved
    return ("miner + seed_start 2^31 refused before the coordinator exists; miner at 0 and "
            "non-miner at 2^31 pass the guard")


# ===========================================================================
# main
# ===========================================================================
GATES = [
    ("G1 no COMPLETE trial -> no publication, no row, no link, 1 non-certifying", g1_no_complete_trial),
    ("G1 zero survivors from a completed sweep -> credited", g1_zero_survivors_credited),
    ("G1 COMPLETE trial of another chunk -> no credit", g1_complete_trial_of_other_chunk),
    ("G1/owner guard 2: swept interval != requested -> no credit", g1_interval_mismatch),
    ("G1 a required phase not finished -> no credit", g1_missing_phase),
    ("G1 miner evidence read from the run's stripe rows", g1_miner_evidence_reads_ledger),
    ("G1/Beta Q1 real legacy route: failed jobs -> no credit; success control credited", g1_legacy_failed_jobs_not_credited),
    ("G1/Beta Q1 PWC and ZMQ refused certification", g1_pwc_zmq_refused),
    ("G2 M1 case: met budget -> refused before dispatch, counts named", g2_m1_case_refused),
    ("G2 FAIL-only chunk -> no credit", g2_fail_only),
    ("G2 PRUNED-only chunk (no search) -> no credit", g2_pruned_only_no_search),
    ("G3 retry of an opening -> same id", g3_retry_same_id),
    ("G3 explicit repeat -> new id with repeat_of", g3_explicit_repeat),
    ("G3 second open search refuses", g3_second_open_refuses),
    ("G3 one open search, enforced in one transaction (race)", g3_one_open_in_one_transaction),
    ("G3 study reuse / wrong study refuses", g3_study_reuse_refuses),
    ("G3 close, then open, works", g3_close_then_open),
    ("G3 append-only in fact", g3_append_only),
    ("G3 exact-schema validation", g3_exact_schema),
    ("G4 link failure after publication surfaces, no advance, retry once", g4_link_failure_surfaces),
    ("G4/Beta Q5 stranded link repaired by a FRESH process via WATCHER", g4_link_repair_fresh_process),
    ("G4/Beta Q5 stranded link repaired by the child preflight", g4_link_repair_in_child_preflight),
    ("G4/Beta R2 pending write fails -> fresh process reconstructs the link", g4_link_repair_after_failed_pending_write),
    ("G4/Beta R2 reconstruction binds only this chunk's own history row", g4_reconstruction_binds_only_this_chunk),
    ("G4 wrong generation / mode / interval / family / non-witness refuses", g4_wrong_witness_refuses),
    ("G4 wrong dataset refuses in the child before dispatch", g4_dataset_mismatch_refuses_in_child),
    ("G4 gaps, overlaps, endpoints 0 / 2^31 / 2^32", g4_endpoints_gaps_overlaps),
    ("G5 each configuration field mutated alone refuses", g5_each_field_refuses),
    ("G5 dataset pointer moved after opening refuses, naming both", g5_dataset_pointer_moved),
    ("G5 step1_search_id arrives on all three hops", g5_search_id_three_hops),
    ("G5 a Step 1 command without the id refuses", g5_search_id_missing_refuses),
    ("G6 the three WATCHER results (distinct, durable)", g6_three_results),
    ("G6 no default search on error", g6_no_default_search_on_error),
    ("G6 no opening from params / record / env", g6_no_opening_from_params_record_env),
    ("G6 no pins inherited from a record", g6_no_pins_inherited),
    ("G6 Step 1 due with a gap even when output is fresh; others unchanged", g6_step1_due_when_fresh),
    ("F5a history COMPLETE: stop; open -> command; no row changed", f5a),
    ("RUN4-SHAPE pinned upper-half search", run4_shape),
    ("G-MINER-START owner guard 1", g_miner_start),
]

L_MOD, I_MOD, W_MOD, WO_MOD = ("utils.step1_search_ledger", "window_optimizer_integration_final",
                               "agents.watcher_agent", "window_optimizer")
MUTANT_SPECS = [
    ("M-CREDIT-GATE removed", I_MOD, "        if not _s1_ok:\n", "        if False:\n",
     g1_no_complete_trial),
    ("M-CREDIT always true", L_MOD, "    req = (int(requested_interval[0]), int(requested_interval[1]))\n",
     "    return (True, 'credited', 'MUTANT', [])\n", g1_no_complete_trial),
    ("M-EVIDENCE not required", L_MOD, "        ev = evidence_by_trial.get(n)\n",
     "        ev = evidence_by_trial.get(n) or {'swept_intervals': {str(p): list(req) for p in need}}\n",
     g1_complete_trial_of_other_chunk),
    ("M-INTERVAL check removed (owner guard 2)", L_MOD,
     "            if iv is None or (int(iv[0]), int(iv[1])) != req:\n",
     "            if iv is None:\n", g1_interval_mismatch),
    ("M-PHASE check removed", L_MOD,
     "        missing = [p for p in need if str(p) not in phases and p not in phases]\n",
     "        missing = []\n", g1_missing_phase),
    ("M-BUDGET refusal removed (S8)", I_MOD, "    if done is not None and done >= int(max_iterations):\n",
     "    if False:\n", g2_m1_case_refused),
    ("M-RETRY id not deterministic", L_MOD, "    return \"s1s-\" + sha256_text(canonical_json(payload))\n",
     "    return \"s1s-\" + uuid.uuid4().hex\n", g3_retry_same_id),
    ("M-REPEAT reopens a closed identity", L_MOD, "                    if state == \"open\":\n",
     "                    if True:\n", g3_explicit_repeat),
    ("M-ONE-OPEN check removed", L_MOD, "                if open_ids:\n                    raise Step1SearchRefused(",
     "                if False:\n                    raise Step1SearchRefused(",
     g3_second_open_refuses),
    ("M-ONE-OPEN check and insert split across transactions", L_MOD,
     "                open_ids = self._open_ids(conn, sel)\n                if existing is not None:\n",
     "                open_ids = self._open_ids(conn, sel)\n"
     "                conn.execute('COMMIT'); import time as _t; _t.sleep(0.3); conn.execute('BEGIN IMMEDIATE')\n"
     "                if existing is not None:\n",
     g3_one_open_in_one_transaction),
    ("M-STUDY owner check removed (S9)", L_MOD, "        if owner is not None and owner != search_id:\n",
     "        if False:\n", g3_study_reuse_refuses),
    ("M-APPEND-ONLY triggers removed", L_MOD,
     "    return (f\"CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()} BEFORE {op} ON {table} \"\n",
     "    return (f\"CREATE TABLE IF NOT EXISTS _noop_{table}_{op.lower()} (x)\" if True else f\"\"\n",
     g3_append_only),
    ("M-SCHEMA definition check removed (Beta Q4)", L_MOD, "        if drift:\n",
     "        if False:\n", g3_exact_schema),
    ("M-LINK idempotency removed", L_MOD, "                if ex is not None:\n",
     "                if False:\n", g4_link_failure_surfaces),
    ("M-LINK generation check removed", L_MOD,
     "        if coverage.generation_id != artifact.generation_id:\n", "        if False:\n",
     g4_wrong_witness_refuses),
    ("M-LINK chunk interval check removed", L_MOD,
     "        if iv != (chunk[\"seed_start\"], chunk[\"seed_end_exclusive\"]):\n", "        if False:\n",
     g4_wrong_witness_refuses),
    ("M-CHILD preflight removed (S4 child)", I_MOD, "    if probs:\n        raise Step1SearchPreflightError(",
     "    if False:\n        raise Step1SearchPreflightError(", g4_dataset_mismatch_refuses_in_child),
    ("M-CONFIG compare removed (S4)", W_MOD,
     "                _s1_refusals.extend(_s1l.compare_configuration(_s1_presented, _s1_rec))\n",
     "                pass\n", g5_each_field_refuses),
    ("M-DATASET check removed (S4)", W_MOD,
     "            elif (_s1_fz_sha != _s1_rec.dataset_sha256\n                  or _s1_fz_ver != _s1_rec.dataset_version_id):\n",
     "            elif False:\n", g5_dataset_pointer_moved),
    ("M-NOT-ROUTED refusal removed (S6)", W_MOD,
     "            if not _s1_want or _s1_got != _s1_want:\n", "            if False:\n",
     None),   # composed below with the assignment removal
    ("M-LOOKUP failure becomes a clean stop", W_MOD,
     "                _sel = _db.select_step1_search(_prng_type, test_both_modes=_test_both)\n",
     "                try:\n                    _sel = _db.select_step1_search(_prng_type, test_both_modes=_test_both)\n"
     "                except Exception:\n                    _sel = _s1l.Selection(_s1l.SEL_NO_OPEN_SEARCH, None, None)\n",
     g6_three_results),
    ("M-OPEN from params (bypass injected)", W_MOD,
     "                _sel = _db.select_step1_search(_prng_type, test_both_modes=_test_both)\n",
     "                if (params or {}).get('step1_search_open'):\n"
     "                    _rec_m, _ = _db.step1_search_ledger().open_search(configuration=step1_presented_configuration(final_params, None), range_start=0, range_end_exclusive=2**32, dataset={'dataset_version_id': 'x', 'dataset_sha256': 'x', 'input_manifest_sha256': 'x', 'dataset_lineage_id': 'x'}, opened_reason='operator', opened_by_ref='mut', repository_commit='mut')\n"
     "                _sel = _db.select_step1_search(_prng_type, test_both_modes=_test_both)\n",
     g6_no_opening_from_params_record_env),
    ("M-PINS inherited from the record", W_MOD,
     "                    final_params, _step1_explicit_pin(step, _pin_bundle))\n",
     "                    final_params, _step1_explicit_pin(step, _pin_bundle) or (_s1_rec.search_space.get('pins') if isinstance(_s1_rec.search_space, dict) else None))\n",
     g6_no_pins_inherited),
    ("M-FRESHNESS Step 1 skipped again (S5)", W_MOD, "        if is_fresh and step != 1:\n",
     "        if is_fresh:\n", g6_step1_due_when_fresh),
    ("M-LEGACY job accounting ignored (Beta Q1)", I_MOD,
     "        _ok, _why = _step1_legacy_jobs_complete(_res)\n",
     "        _ok, _why = True, ''\n", g1_legacy_failed_jobs_not_credited),
    ("M-PWC/ZMQ handed interval restored (Beta Q1)", I_MOD,
     "            \"swept_intervals\": {}, \"certification_refused\": True}\n",
     "            \"swept_intervals\": {str(p): [0, 1000] for p in (1, 2, 3, 4)}, \"certification_refused\": True}\n",
     g1_pwc_zmq_refused),
    ("M-PENDING record not written (Beta Q5)", I_MOD,
     "            _s1_search_ctx[\"ledger\"].record_link_pending(**_s1_link_args)\n",
     "            pass\n", g4_link_repair_fresh_process),
    ("M-REPAIR removed from the child preflight (Beta Q5)", I_MOD,
     "    _repaired = led.repair_pending_links(step1_search_id)\n",
     "    _repaired = []\n", g4_link_repair_in_child_preflight),
    ("M-REPAIR removed from WATCHER's lookup, in the fresh process (Beta Q5)", L_MOD, None, None,
     "child:select_repair_off:link"),
    ("M-RECONSTRUCTION removed, in the fresh process (Beta R2)", L_MOD, None, None,
     "child:reconstruct_off:pending"),
    ("M-STUDY_IDENTITY not passed for a search chunk (Beta R2)", I_MOD,
     "            study_identity=(_s1_chunk.study_name if _s1_search_ctx is not None\n",
     "            study_identity=(None if _s1_search_ctx is not None\n",
     g4_link_repair_after_failed_pending_write),
    ("M-RECONSTRUCTION study filter removed (Beta R2)", L_MOD,
     "                     and r.study_identity == ch[\"study_name\"]\n",
     "                     and True\n", g4_reconstruction_binds_only_this_chunk),
    ("M-MINER-START guard removed (owner guard 1)", WO_MOD,
     "    if use_range_miner and seed_start != 0:\n", "    if False:\n", g_miner_start),
]


def mutant_not_routed():
    """Both the assignment AND the refusal removed: the command dispatches without the id
    — the gate g5_search_id_missing_refuses must go red."""
    path = _MOD_PATHS[W_MOD]
    src = open(path, encoding="utf-8").read()
    a = "            final_params['step1_search_id'] = _s1_rec.search_id\n"
    b = "            if not _s1_want or _s1_got != _s1_want:\n"
    if src.count(a) != 1 or src.count(b) != 1:
        MUTANTS.append(("M-NOT-ROUTED refusal removed (S6)", False, "ERROR: anchor"))
        return
    # Compose both edits in one mutant module.
    import importlib
    real = importlib.import_module(W_MOD)
    import agents
    mod = types.ModuleType(W_MOD)
    mod.__file__ = path
    mod.__package__ = "agents"
    exec(compile(src.replace(a, "            final_params['step1_search_id'] = ''\n")
                 .replace(b, "            if False:\n"), path + "<mutant>", "exec"), mod.__dict__)
    sys.modules[W_MOD] = mod
    agents.watcher_agent = mod
    try:
        with watcher_env() as (W, cap, dbp, d):
            agent = make_agent(W, d)
            cli_open(W, agent, BASE_PARAMS, start=0, end=T32)
            res = run_step1(W, agent, BASE_PARAMS)
            leaked = bool(cap["argv"]) and "--step1-search-id" not in cap["argv"][-1]
        MUTANTS.append(("M-NOT-ROUTED refusal removed (S6)", leaked,
                        "RED: a Step 1 command dispatched WITHOUT --step1-search-id"
                        if leaked else f"NOT DETECTED: {res.get('blocked_by')}"))
    finally:
        sys.modules[W_MOD] = real
        agents.watcher_agent = real


def main():
    print("=" * 78)
    print("STEP 1 SEARCH RECORD — STAGE 1 — gate battery (CPU only, isolated state)")
    print("=" * 78)
    for name, fn in GATES:
        run_gate(name, fn)
        n, ok, det = RESULTS[-1]
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {det}", flush=True)
    print("MUTANTS (each must turn its gate RED)")
    for label, mod, old, new, gate in MUTANT_SPECS:
        if gate is None:
            mutant_not_routed()
        elif isinstance(gate, str) and gate.startswith("child:"):
            _, cm, fm = gate.split(":")
            try:
                g4_link_repair_fresh_process(child_mutant=cm, failure=fm)
                MUTANTS.append((label, False, "gate stayed GREEN under the mutant"))
            except BaseException as e:   # noqa: BLE001
                MUTANTS.append((label, True, f"RED: {type(e).__name__}: {str(e)[:220]}"))
        else:
            run_mutant(label, mod, old, new, gate)
        l, det, txt = MUTANTS[-1]
        print(f"  [{'RED (detected)' if det else 'NOT DETECTED'}] {l}: {txt}", flush=True)
    g_ok = sum(1 for _, ok, _ in RESULTS if ok)
    m_ok = sum(1 for _, det, _ in MUTANTS if det)
    print(f"GATES {g_ok}/{len(RESULTS)}  MUTANTS {m_ok}/{len(MUTANTS)} detected")
    fails = [(n, d) for n, ok, d in RESULTS if not ok]
    for n, d in fails:
        print(f"--- FAIL {n}\n{d}")
    ok = g_ok == len(RESULTS) and m_ok == len(MUTANTS)
    print(f"COMPLETION SENTINEL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--q5-child":
        q5_child(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        sys.exit(0)
    sys.exit(main())
