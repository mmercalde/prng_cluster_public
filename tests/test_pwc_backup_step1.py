#!/usr/bin/env python3
"""test_pwc_backup_step1.py — acceptance harness for PWC backup step 1 (S1-S8).

Brief: docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md.

CPU only. No GPU, no fleet, no SSH, no engine run, no socket bound:
  * the TCP transport is a fake object; `_dispatch_to_tcp` is a fake worker that
    echoes what sieve_filter.execute_sieve_job echoes (seed_range,
    min_match_threshold);
  * `subprocess.run` / `subprocess.Popen` are INTERCEPTED for every gate that calls
    startup(), so the S156 reap, the live-GPU probe, the Zeus `pkill` and the
    launch-script writes are recorded, never executed;
  * `progress_display` is replaced by a stub so no dashboard file is written;
  * the only real subprocesses are `python3 -m persistent.pwc_worker_service`
    with PWC_HOST/PWC_PORT UNSET (it must exit before any connection), an import
    of persistent/pwc_protocol.py, and `git show HEAD:persistent_worker_coordinator.py`
    for the S2 byte-equality baseline;
  * the suite RUNS from a fresh temporary working directory created inside
    <worktree>/data/pwc_step1/test_out (Beta X-PWC-BACKUP R1 Finding 2: the
    trial builder writes `results/window_opt_*.json` relative to the cwd, and
    the first version ran from the repository root). Every file it writes is
    therefore under test_out; a final containment check asserts the
    repository's own `results/` listing is unchanged.

Each refusal has a MUTANT: a textual edit of the LIVE source that removes it,
loaded as a separate module. A mutant is killed only if (1) the edit applies
exactly once, (2) the same detector passes on the unmutated module (positive
control) and (3) the detector fails on the mutant.
"""
import ast
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

OUT = os.path.join(ROOT, "data", "pwc_step1", "test_out")
PWC_PATH = os.path.join(ROOT, "persistent_worker_coordinator.py")
WORKER_PATH = os.path.join(ROOT, "persistent", "pwc_worker_service.py")
PROTO_PATH = os.path.join(ROOT, "persistent", "pwc_protocol.py")
GATE_PATH = os.path.join(ROOT, "scripts", "pwc_parity_gate.py")
TB_PATH = os.path.join(ROOT, "persistent", "pwc_transport_base.py")

import persistent_worker_coordinator as PWC                       # noqa: E402
import pwc_parity_gate as PG                                       # noqa: E402
from window_optimizer import WindowConfig                          # noqa: E402

RIG = "10.9.9.1"          # a fake remote ROCm node; nothing ever dials it
DIAL = "192.168.3.177"    # VM101 (CLAUDE.md §3); only written into captured text
TWO32 = 1 << 32


# ═════════════════════════════════════════════════════════════════════════════
# fixtures
# ═════════════════════════════════════════════════════════════════════════════
class FakeTransport:
    def __init__(self, n):
        self.n = n
    def ready_count(self):
        return self.n
    def worker_count(self):
        return self.n
    def start(self):
        pass
    def stop(self):
        pass
    def broadcast_init(self):
        return self.n
    # [PWC-STAGE1 A3 level 1] the launch-set admission API startup() now calls;
    # recorded so a gate can read what was admitted.
    def set_allowed_worker_ids(self, ids):
        self.allowed = None if ids is None else set(ids)
    def allow_worker_id(self, wid):
        self.allowed.add(wid)
    def revoke_worker_id(self, wid):
        self.allowed.discard(wid)


class Worker:
    """A fake TCP worker. `plan(idx_job) -> dict|None` overrides the reply for a
    job; default is a clean reply echoing the job's range and threshold."""
    def __init__(self, plan=None, survivors_per_chunk=2):
        self.plan = plan or (lambda job: None)
        self.jobs = []
        self.lock = threading.Lock()
        self.k = survivors_per_chunk

    def __call__(self, job):
        with self.lock:
            self.jobs.append(dict(job))
        over = self.plan(job)
        if over is not None:
            return over
        s = job["seed_start"]
        return {"status": "ok", "job_id": job["job_id"],
                "survivors": [s + i for i in range(self.k)],
                "match_rates": [0.9] * self.k,
                "skip_sequences": [[] for _ in range(self.k)],
                "strategy_ids": [0] * self.k,
                "worker_seed_range": {"start": job["seed_start"], "end": job["seed_end"]},
                "worker_min_match_threshold": job["min_match_threshold"]}


def write_config(gpu_count=8):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"pwc_cfg_{gpu_count}.json")
    with open(path, "w") as fh:
        json.dump({"nodes": [{"hostname": RIG, "gpu_type": "RX 6600 XT",
                              "gpu_count": gpu_count,
                              "python_env": "/nowhere/venv/bin/python",
                              "script_path": "/nowhere/repo",
                              "username": "michael"}]}, fh)
    return path


def make_pwc(M, *, transport="tcp", pool=4, cap=25_000, gpu_count=8,
             address=DIAL, planned=None, ready=4, worker=None):
    with contextlib.redirect_stderr(io.StringIO()):
        p = M.PersistentWorkerCoordinator(
            config_file=write_config(gpu_count), worker_pool_size=pool,
            seed_cap_amd=cap, seed_cap_nvidia=cap, pwc_transport=transport,
            pwc_coordinator_address=address, planned_total_seeds=planned)
    p._tcp_transport = FakeTransport(ready)
    p._effective_pool_by_host = {RIG: min(pool, gpu_count)}
    p._dispatch_to_tcp = worker or Worker()
    return p


def sieve(p, *, prng="java_lcg", total=100_000, start=0, thr=0.4):
    with contextlib.redirect_stderr(io.StringIO()):
        return p.run_sieve_pass(
            prng_type=prng, residues=[1, 2, 3], total_seeds=total, threshold=thr,
            window_size=8, output_file=os.path.join(OUT, f"pass_{prng}.json"),
            dataset_path="unused.json", sessions=["midday", "evening"],
            skip_range=[0, 10], seed_start=start)


class Intercept:
    """Replace subprocess.run/Popen, progress_display and the TCP transport
    factory for the duration of a startup() test. Records every call."""
    def __init__(self):
        self.calls = []

    def _run(self, cmd, *a, **kw):
        self.calls.append(("run", cmd, kw.get("input")))
        text = kw.get("text") or kw.get("universal_newlines")
        # [PWC-STAGE1] A fake rig that answers each remote command the way a
        # healthy rig does. Before, every command got "8\n" (the live-GPU-probe
        # answer), so the cache reset (on main since 7444aea) and the stale-worker
        # kill (stage 1 B5) saw no marker and refused every rig.
        flat = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
        if "cupy_cache_gpu_" in flat and PWC.PWC_CACHE_RESET_OK in flat:
            line = PWC.PWC_CACHE_RESET_OK
        elif PWC.PWC_STALE_KILL_OK in flat:
            line = PWC.PWC_STALE_KILL_OK
        elif flat.endswith(".sh") and "bash /tmp/pwc_tcp_launch_gpu" in flat:
            line = "PID=4242"
        else:
            line = "8"
        out = line + "\n" if text else (line + "\n").encode()
        return subprocess.CompletedProcess(cmd, 0, out, "" if text else b"")

    def _popen(self, cmd, *a, **kw):
        self.calls.append(("popen", cmd, None))
        raise RuntimeError("Popen intercepted")

    def __enter__(self):
        import persistent.pwc_transport_base as TB
        self._saved = (subprocess.run, subprocess.Popen, TB.build_transport,
                       sys.modules.get("progress_display"), time.sleep)
        subprocess.run = self._run
        subprocess.Popen = self._popen
        TB.build_transport = lambda **kw: FakeTransport(64)
        stub = types.ModuleType("progress_display")
        class _PW:
            def __init__(self, *a, **k): pass
            def __getattr__(self, n): return lambda *a, **k: None
        stub.ProgressWriter = _PW
        sys.modules["progress_display"] = stub
        time.sleep = lambda s: None
        return self

    def __exit__(self, *exc):
        import persistent.pwc_transport_base as TB
        (subprocess.run, subprocess.Popen, TB.build_transport, pd, time.sleep) = self._saved
        if pd is None:
            sys.modules.pop("progress_display", None)
        else:
            sys.modules["progress_display"] = pd
        return False

    def launch_scripts(self):
        return [inp.decode() for kind, cmd, inp in self.calls
                if kind == "run" and inp is not None]


def startup_for_real(p):
    """Run the REAL startup() with every outward call intercepted."""
    with Intercept() as ic, contextlib.redirect_stderr(io.StringIO()):
        try:
            p.startup()
            return ic, None
        except Exception as exc:                                    # noqa: BLE001
            return ic, exc


def trial(M, prepared, *, both=False, start=0, total=100_000):
    real = M.PersistentWorkerCoordinator
    prepared.startup = lambda: None
    M.PersistentWorkerCoordinator = lambda **kw: prepared
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return M.run_trial_persistent(
                coordinator_cfg="unused.json",
                config=WindowConfig(window_size=8, offset=0,
                                    sessions=["midday", "evening"], skip_min=0,
                                    skip_max=10),
                trial_number=1, prng_base="java_lcg", residues=[1, 2],
                total_seeds=total, forward_threshold=0.4, reverse_threshold=0.45,
                test_both_modes=both, dataset_path="unused.csv",
                pwc_transport="tcp", worker_pool_size=4, seed_cap_amd=25_000,
                seed_start=start, pwc_coordinator_address=DIAL)
    finally:
        M.PersistentWorkerCoordinator = real


def raises(exc_type, fn):
    try:
        fn()
    except exc_type as e:
        return e
    except Exception as e:                                          # noqa: BLE001
        raise AssertionError(f"expected {exc_type.__name__}, got "
                             f"{type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc_type.__name__}; nothing was raised")


# ═════════════════════════════════════════════════════════════════════════════
# detectors — each takes the module under test, so mutants reuse them verbatim
# ═════════════════════════════════════════════════════════════════════════════
# ── S1 ───────────────────────────────────────────────────────────────────────
def d_s1_clean(M):
    r = sieve(make_pwc(M))
    a = r["pwc_pass_accounting"]
    assert a["schema"] == M.PWC_PASS_ACCOUNTING_SCHEMA
    assert r["pass_complete"] is True and a["complete"] is True, a
    assert (a["planned"], a["dispatched"], a["completed"]) == (4, 4, 4), a
    assert a["failed"] == a["timed_out"] == a["not_dispatched"] == a["missing"] == 0
    assert a["completed_intervals"] == [[0, 100_000]], a["completed_intervals"]
    assert r["failed_chunks"] == 0 and r["total_chunks"] == 4      # old fields kept
    return "clean pass: 4/4 completed, tiles [0, 100000), complete"


def d_s1_failed_chunk(M):
    w = Worker(lambda j: {"status": "error", "message": "boom"}
               if j["seed_start"] == 50_000 else None)
    r = sieve(make_pwc(M, worker=w))
    a = r["pwc_pass_accounting"]
    assert a["failed"] == 1 and a["completed"] == 3, a
    assert r["pass_complete"] is False and a["complete"] is False, a
    assert a["completed_intervals"] == [[0, 50_000], [75_000, 100_000]], a
    assert a["incomplete_chunks"][0]["outcome"] == "failed"
    return "failed chunk: 3/4 completed, gap [50000, 75000), marked incomplete"


def d_s1_timed_out_chunk(M):
    w = Worker(lambda j: {"status": "error", "message": "TCP worker timeout"}
               if j["seed_start"] == 25_000 else None)
    r = sieve(make_pwc(M, worker=w))
    a = r["pwc_pass_accounting"]
    assert a["timed_out"] == 1 and a["completed"] == 3 and a["complete"] is False, a
    return "timed-out chunk (after its one retry): counted timed_out, incomplete"


def d_s1_pass_abort(M):
    """The pass-progress abort: chunks still running are timed_out, chunks never
    started are not_dispatched, and the late threads cannot change the record."""
    saved = M.JOB_TIMEOUT_S
    M.JOB_TIMEOUT_S = 0.25                       # progress window 0.5 s
    try:
        def plan(j):
            if j["seed_start"] != 0:
                time.sleep(2.0)
            return None
        r = sieve(make_pwc(M, worker=Worker(plan)), total=200_000)
    finally:
        M.JOB_TIMEOUT_S = saved
    a = r["pwc_pass_accounting"]
    assert a["planned"] == 8 and a["pass_progress_timeout"] is True, a
    assert a["completed"] == 1, a
    assert a["timed_out"] + a["not_dispatched"] == 7 and a["not_dispatched"] >= 1, a
    assert a["dispatched"] == 8 - a["not_dispatched"], a
    assert a["complete"] is False
    return (f"pass abort: completed={a['completed']} timed_out={a['timed_out']} "
            f"not_dispatched={a['not_dispatched']}, incomplete")


def d_s1_range_mismatch(M):
    def plan(j):
        if j["seed_start"] == 0:
            return {"status": "ok", "survivors": [999_999], "match_rates": [0.9],
                    "skip_sequences": [[]], "strategy_ids": [0],
                    "worker_seed_range": {"start": 900_000, "end": 925_000},
                    "worker_min_match_threshold": j["min_match_threshold"]}
    r = sieve(make_pwc(M, worker=Worker(plan)))
    a = r["pwc_pass_accounting"]
    assert a["failed"] == 1 and a["complete"] is False, a
    assert 999_999 not in r["survivors"], "a wrong-range chunk's survivors were kept"
    return "worker reports a different seed_range: failed, survivors excluded"


def d_s1_trial(M):
    clean = trial(M, make_pwc(M))
    c = clean["pwc_completion"]
    assert clean["pwc_trial_complete"] is True and c["trial_complete"] is True, c
    assert c["swept_intervals"] == {"1": [0, 100_000], "2": [0, 100_000]}, c
    assert c["passes"]["3"]["status"] == c["passes"]["4"]["status"] == "not_run"
    for k in ("forward_map_constant", "reverse_map_constant", "bidirectional_constant",
              "schema_version", "forward_map", "bidirectional_count"):
        assert k in clean, f"v2/legacy field {k} missing"         # nothing removed

    w = Worker(lambda j: {"status": "error", "message": "boom"}
               if "reverse" in j["prng_families"][0] and j["seed_start"] == 0 else None)
    bad = trial(M, make_pwc(M, worker=w))
    b = bad["pwc_completion"]
    assert bad["pwc_trial_complete"] is False and b["incomplete_passes"] == ["2"], b
    assert b["swept_intervals"] == {"1": [0, 100_000]}, b
    assert b["passes"]["2"]["accounting"]["failed"] == 1
    return "trial: clean -> complete, swept {1,2}; reverse chunk fails -> incomplete [2]"


def d_s1_unaccounted(M):
    comp = M.build_pwc_trial_completion((0, 10), {"1": {"survivors": [1]}})
    assert comp["passes"]["1"]["status"] == "unaccounted", comp
    assert comp["trial_complete"] is False and comp["swept_intervals"] == {}, comp
    return "a pass result with no accounting is 'unaccounted', never complete"


# ── S2 ───────────────────────────────────────────────────────────────────────
# [PWC-STAGE2] The base is the commit BEFORE step 1 / stage 1, pinned. This used
# to read HEAD, which was that base until stage 1 landed (ee39404); since then
# HEAD carries the change and the gate could only fail.
PRE_STAGE1_BASE = "27eb991"


def _baseline_module():
    src = subprocess.run(["git", "show", PRE_STAGE1_BASE + ":persistent_worker_coordinator.py"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert "[PWC-BACKUP" not in src, f"{PRE_STAGE1_BASE} already carries this change"
    return _load_source(src, "pwc_baseline_head")


# [PWC-STAGE2 ruling 1] Stage 2 changes ONE job key on purpose: a constant job's
# phase2_threshold was always the unused 0.5 default; it now carries the trial's
# threshold, as RANGE-MINER writes one value as both keys. [PWC-STAGE2 R2] job_id
# is now unique per pass (`sieve_<pass id>_NNN`, Beta X-PWC-STAGE2 R1 finding 2).
# Every other byte of the job dicts must still match the base.
_STAGE2_CHANGED_JOB_KEYS = ("phase2_threshold", "job_id")


def _jobs_without_stage2_keys(jobs):
    return [{k: v for k, v in j.items() if k not in _STAGE2_CHANGED_JOB_KEYS} for j in jobs]


def d_s2_zero_byte_equal(M):
    base = _baseline_module()
    for total, ready, cap in ((100_000, 4, 25_000), (1_000_003, 6, 50_000),
                              (7, 3, 25_000)):
        jobs = {}
        for tag, mod in (("new", M), ("base", base)):
            p = make_pwc(M, cap=cap, ready=ready, pool=min(ready, 6))
            w = p._dispatch_to_tcp
            with contextlib.redirect_stderr(io.StringIO()):
                mod.PersistentWorkerCoordinator.run_sieve_pass(
                    p, prng_type="java_lcg", residues=[1], total_seeds=total,
                    threshold=0.4, window_size=8,
                    output_file=os.path.join(OUT, f"s2_{tag}.json"),
                    dataset_path="unused.json", sessions=["midday", "evening"],
                    skip_range=[0, 10])
            jobs[tag] = json.dumps(sorted(_jobs_without_stage2_keys(w.jobs),
                                          key=lambda j: (j["seed_start"], j["seed_end"])),
                                   sort_keys=True)
        assert jobs["new"] == jobs["base"], (
            f"seed_start=0 job dicts differ from {PRE_STAGE1_BASE} (total={total})")
    return (f"seed_start=0: job dicts byte-equal to {PRE_STAGE1_BASE}'s for 3 shapes "
            f"(except {list(_STAGE2_CHANGED_JOB_KEYS)}, changed by stage 2)")


def d_s2_two31(M):
    w = Worker()
    r = sieve(make_pwc(M, worker=w), start=1 << 31)
    got = sorted((j["seed_start"], j["seed_end"]) for j in w.jobs)
    s = 1 << 31
    assert got == [(s + i * 25_000, s + (i + 1) * 25_000) for i in range(4)], got
    a = r["pwc_pass_accounting"]
    assert a["requested_interval"] == [s, s + 100_000] and a["complete"], a
    assert (r["seed_start"], r["seed_end"]) == (s, s + 100_000)
    return "seed_start=2^31: chunks partition [2^31, 2^31+100000)"


def d_s2_endpoints(M):
    w = Worker()
    sieve(make_pwc(M, worker=w), start=TWO32 - 100_000)
    assert max(j["seed_end"] for j in w.jobs) == TWO32
    raises(M.PwcSeedIntervalRefused,
           lambda: sieve(make_pwc(M), start=TWO32 - 100_000 + 1))
    raises(M.PwcSeedIntervalRefused, lambda: sieve(make_pwc(M), start=-1))
    raises(M.PwcSeedIntervalRefused, lambda: sieve(make_pwc(M), start=True))
    assert M.validate_pwc_seed_interval(0, 0) == (0, 0)
    # the trial refuses before the fleet is built
    built = []
    real = M.PersistentWorkerCoordinator
    M.PersistentWorkerCoordinator = lambda **kw: built.append(kw)
    try:
        raises(M.PwcSeedIntervalRefused, lambda: M.run_trial_persistent(
            "x", WindowConfig(8, 0, ["midday"], 0, 10), 1, "java_lcg", [1], 10,
            0.4, 0.4, False, seed_start=TWO32 - 9))
    finally:
        M.PersistentWorkerCoordinator = real
    assert not built, "coordinator built before the interval check"
    return "end = 2^32 accepted; 2^32+1, -1, bool refused; trial refuses pre-fleet"


# ── S3 ───────────────────────────────────────────────────────────────────────
def d_s3_unset(M):
    for bad in (None, "", "0.0.0.0", "127.0.0.1", "localhost"):
        p = make_pwc(M, address=bad, planned=100_000)
        ic, exc = startup_for_real(p)
        assert isinstance(exc, M.PwcCoordinatorAddressUnset), (bad, exc)
        assert "pwc_coordinator_address" in str(exc)
        assert not ic.calls, f"address={bad!r}: subprocess called before refusal"
    return "unset/empty/wildcard/loopback dial address refuses before any SSH"


def d_s3_set(M):
    p = make_pwc(M, address=DIAL, planned=100_000)
    ic, exc = startup_for_real(p)
    assert exc is None, f"startup failed: {exc!r}"
    scripts = ic.launch_scripts()
    assert len(scripts) == 4, f"expected 4 launch scripts, got {len(scripts)}"
    for s in scripts:
        assert f"export PWC_HOST={DIAL}" in s and "export PWC_PORT=5600" in s, s
        assert "192.168.3.127" not in s, s
    assert p._effective_pool_by_host == {RIG: 4}
    return "set: every launch script exports PWC_HOST=192.168.3.177, PWC_PORT; no .127"


def d_s3_worker(W_src=None):
    """The rig worker: PWC_HOST / PWC_PORT come only from _required_env (AST of
    the LIVE __main__), and unset exits 2 before connecting (executed)."""
    src = W_src if W_src is not None else open(WORKER_PATH, encoding="utf-8").read()
    tree = ast.parse(src)
    main = [n for n in tree.body if isinstance(n, ast.If)
            and "__main__" in ast.unparse(n.test)]
    assert main, "no __main__ block"
    assigns = {ast.unparse(t): ast.unparse(n.value) for n in ast.walk(main[0])
               if isinstance(n, ast.Assign) for t in n.targets}
    assert assigns.get("host") == "_required_env('PWC_HOST')", assigns.get("host")
    assert "_required_env('PWC_PORT')" in assigns.get("port", ""), assigns.get("port")
    # no os.environ.get("PWC_HOST"/"PWC_PORT", <fallback>) anywhere in the module
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and ast.unparse(n.func).endswith("environ.get")
                and n.args and isinstance(n.args[0], ast.Constant)
                and n.args[0].value in ("PWC_HOST", "PWC_PORT")):
            raise AssertionError(f"fallback read of {n.args[0].value}: {ast.unparse(n)}")
    if W_src is None:
        env = {k: v for k, v in os.environ.items() if k not in ("PWC_HOST", "PWC_PORT")}
        r = subprocess.run([sys.executable, "-m", "persistent.pwc_worker_service"],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and "PWC_HOST is not set" in r.stderr, (r.returncode, r.stderr[-400:])
        env["PWC_HOST"] = DIAL
        r = subprocess.run([sys.executable, "-m", "persistent.pwc_worker_service"],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and "PWC_PORT is not set" in r.stderr, (r.returncode, r.stderr[-400:])
    return "worker: dial target from env only; unset PWC_HOST / PWC_PORT exits 2"


# ── S4 ───────────────────────────────────────────────────────────────────────
def d_s4_transport(M):
    for bad in (None, "TCP", "zmq"):
        p = make_pwc(M, transport=bad, planned=100_000)
        ic, exc = startup_for_real(p)
        assert isinstance(exc, M.PwcSettingRefused) and "pwc_transport" in str(exc), (bad, exc)
        assert not ic.calls, f"transport={bad!r}: subprocess before refusal"
    import persistent.pwc_transport_base as TB
    raises(ValueError, lambda: TB.build_transport("TCP"))
    assert TB.build_transport("ssh") is None
    return "unset/unknown transport refuses at startup and in build_transport"


def d_s4_factory(TB=None):
    """[Beta R1 Q3] build_transport with the transport OMITTED must not return the
    SSH sentinel; it is a TypeError."""
    if TB is None:
        import persistent.pwc_transport_base as TB
    raises(TypeError, lambda: TB.build_transport())
    raises(ValueError, lambda: TB.build_transport("TCP"))
    assert TB.build_transport("ssh") is None
    return "build_transport() with no transport raises TypeError (no silent SSH)"


def d_s4_config_and_fields(M):
    raises(M.PwcSettingRefused, lambda: M.PersistentWorkerCoordinator(
        config_file=os.path.join(OUT, "does_not_exist.json"), pwc_transport="tcp"))
    p = make_pwc(M)
    raises(M.PwcSettingRefused, lambda: p.run_sieve_pass(
        prng_type="java_lcg", residues=[1], total_seeds=10, threshold=0.4,
        window_size=8, output_file=os.path.join(OUT, "x.json"), sessions=None,
        skip_range=[0, 10]))
    raises(M.PwcSettingRefused, lambda: M.run_trial_persistent(
        "x", types.SimpleNamespace(window_size=8, offset=0), 1, "java_lcg", [1], 10,
        0.4, 0.4, False))
    return "missing config file, missing sessions/skip_range, config lacking fields refuse"


# ── S5 ───────────────────────────────────────────────────────────────────────
def d_s5_envelope(M):
    inside = [(6, 50_000), (8, 25_000), (4, 50_000), (7, 25_000), (1, 1)]
    outside = [(6, 50_001), (8, 25_001), (7, 25_001), (9, 1), (8, 50_000), (6, 100_000)]
    for w, c in inside:
        assert M.pwc_envelope_violation(w, c) is None, (w, c)
    for w, c in outside:
        assert M.pwc_envelope_violation(w, c) is not None, (w, c)
    # through the REAL startup: refused before any subprocess; inside proceeds
    for pool, cap in ((8, 25_001), (6, 50_001), (8, 2_000_000)):
        p = make_pwc(M, pool=pool, cap=cap, planned=10_000_000)
        ic, exc = startup_for_real(p)
        assert isinstance(exc, M.PwcStableEnvelopeRefused), (pool, cap, exc)
        assert f"workers_per_rig={pool}" in str(exc) and "PWC_STABLE_ENVELOPE" in str(exc)
        assert not ic.calls, f"({pool},{cap}): subprocess before refusal"
    for pool, cap in ((8, 25_000), (6, 50_000)):
        p = make_pwc(M, pool=pool, cap=cap, planned=10_000_000)
        ic, exc = startup_for_real(p)
        assert exc is None, (pool, cap, exc)
    # planned_total_seeds bounds the chunk: a 2M cap with 40,000 seeds is inside
    p = make_pwc(M, pool=6, cap=2_000_000, planned=40_000)
    ic, exc = startup_for_real(p)
    assert exc is None, exc
    # dispatch-site recheck with exact values, before any job
    w = Worker()
    p = make_pwc(M, pool=8, cap=2_000_000, ready=8, worker=w)
    raises(M.PwcStableEnvelopeRefused, lambda: sieve(p, total=10_000_000))
    assert not w.jobs, "a job was dispatched outside the envelope"
    return "inside/edge pass, outside refuses pre-spawn (no SSH) and pre-dispatch"


# ── S6 ───────────────────────────────────────────────────────────────────────
def d_s6_agree(M):
    for prng, thr in (("java_lcg", 0.4), ("java_lcg_reverse", 0.45)):
        r = sieve(make_pwc(M), prng=prng, thr=thr)
        t = r["pwc_pass_accounting"]["threshold"]
        assert t["requested"] == thr and t["payload"] == [thr] and t["effective"] == [thr], t
        assert t["agree"] and t["verified"], t
    r = sieve(make_pwc(M, worker=Worker(lambda j: {
        "status": "ok", "survivors": [], "match_rates": [], "skip_sequences": [],
        "strategy_ids": [], "worker_seed_range": None,
        "worker_min_match_threshold": None})))
    t = r["pwc_pass_accounting"]["threshold"]
    assert t["agree"] and not t["verified"] and t["effective_unreported_chunks"] == 4, t
    return "fwd 0.4 / rev 0.45 agree and verified; unreported effective -> unverified"


def d_s6_disagree(M):
    def plan(j):
        if j["seed_start"] == 25_000:
            return {"status": "ok", "survivors": [], "match_rates": [],
                    "skip_sequences": [], "strategy_ids": [],
                    "worker_seed_range": {"start": j["seed_start"], "end": j["seed_end"]},
                    "worker_min_match_threshold": 0.5}
    e = raises(M.PwcThresholdContractMismatch,
               lambda: sieve(make_pwc(M, worker=Worker(plan)), thr=0.4))
    assert "requested=0.4" in str(e) and "0.5" in str(e), e
    return "one chunk filtered at 0.5 for a 0.4 request -> the pass refuses"


def d_s6_hybrid_needs_threshold(M):
    # [PWC-STAGE2 ruling 1] the quarantine is lifted; in its place a hybrid pass
    # with no phase2_threshold refuses before any dispatch (tests/test_pwc_backup_stage2.py
    # covers the whole contract)
    w = Worker()
    e = raises(M.PwcSettingRefused,
               lambda: sieve(make_pwc(M, worker=w), prng="java_lcg_hybrid"))
    assert "phase2_threshold" in str(e) and not w.jobs, (e, w.jobs)
    return "hybrid run_sieve_pass without phase2_threshold refuses before any dispatch"


# ── S7 ───────────────────────────────────────────────────────────────────────
def d_s7_import(src=None):
    if src is None:
        r = subprocess.run([sys.executable, "-c",
                            "import persistent.pwc_protocol as P;"
                            "m=P.from_dict({'message_type':'hello','worker_id':'w','gpu_id':1});"
                            "assert m.worker_id=='w' and m.gpu_id==1;"
                            "assert P.BaseMessage.__dataclass_fields__['worker_id'].default==''"],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr[-400:]
        return "persistent/pwc_protocol.py imports; from_dict round-trips"
    _load_source(src, "pwc_protocol_mutant")
    return "imported"


# ── S8 ───────────────────────────────────────────────────────────────────────
TARGETS = [{"node_id": f"rig{i}", "endpoint": f"10.9.9.{i}", "ssh_user": "michael",
            "worker_hostname": f"rig{i}", "script_path": "/nowhere/repo"}
           for i in (1, 2, 3)]


def _fake_gate_runner(G, transport, *, differ=None, unreachable=None, hostname_of=None,
                      truncated=None, dirty=False):
    gov = G.GOVERNED_FILES[transport]
    exp = G.expected_digests(gov)
    calls = []

    def runner(cmd, timeout):
        calls.append(cmd)
        if cmd[0] == "git":
            out = b""
            if "status" in cmd and dirty:
                out = f" M {gov[0]}\n".encode()
            if "rev-parse" in cmd:
                out = b"deadbeef\n"
            return subprocess.CompletedProcess(cmd, 0, out, b"")
        assert cmd[0] == "ssh", cmd
        ep = cmd[-2].split("@", 1)[1]
        if ep == unreachable:
            return subprocess.CompletedProcess(cmd, 255, b"", b"ssh: connect timed out")
        host = (hostname_of or {}).get(ep, "rig" + ep.rsplit(".", 1)[1])
        lines = [G.BEGIN_SENTINEL, f"HOST\t{host}"]
        for rel in gov:
            d = exp[rel]["sha256"]
            if differ and ep == differ[0] and rel == differ[1]:
                d = "0" * 64
            lines.append(f"FILE\t{rel}\t{d}\t{exp[rel]['size']}")
        if ep != truncated:
            lines.append(G.END_SENTINEL)
        return subprocess.CompletedProcess(cmd, 0, ("\n".join(lines) + "\n").encode(), b"")
    return runner, calls


def _gate(G, transport="tcp", targets=TARGETS, **kw):
    runner, calls = _fake_gate_runner(G, transport, **kw)
    sink = []
    rc = G.run_gate(transport, runner=runner, resolve_targets=lambda prof: list(targets),
                    out=sink.append)
    return rc, "\n".join(sink), calls


def d_s8_gate(G):
    for t in ("tcp", "ssh"):
        rc, out, _ = _gate(G, t)
        assert rc == G.EXIT_PROCEED, f"{t} equal: {out}"
    cases = {
        "one file differing": dict(differ=("10.9.9.2", "sieve_filter.py")),
        "rig unreachable": dict(unreachable="10.9.9.3"),
        "truncated probe": dict(truncated="10.9.9.1"),
        "wrong hostname": dict(hostname_of={"10.9.9.1": "rig2"}),
        "dirty local tree": dict(dirty=True),
    }
    for name, kw in cases.items():
        rc, out, _ = _gate(G, **kw)
        assert rc == G.EXIT_REFUSE and "REFUSED" in out, f"{name}: rc={rc}\n{out}"
    rc, out, _ = _gate(G, targets=[])
    assert rc == G.EXIT_REFUSE and "ZERO rigs" in out, out
    rc, out = G.run_gate("tcp", runner=_fake_gate_runner(G, "tcp")[0],
                         resolve_targets=lambda p: (_ for _ in ()).throw(KeyError("x")),
                         out=lambda s: None), ""
    assert rc == G.EXIT_REFUSE
    assert G.run_gate("zmq", out=lambda s: None) == G.EXIT_REFUSE
    return ("equal (tcp, ssh) PASS; differing, unreachable, truncated, hostname, dirty, "
            "zero rigs, resolver error, unknown transport all REFUSE")


def d_s8_closure_and_fleet(G):
    for t in ("tcp", "ssh"):
        ok, unc, miss = G.closure_coverage(t)
        assert ok, (t, unc, miss)
    ok, unc, _ = G.closure_coverage("tcp", governed=G.GOVERNED_FILES["tcp"][1:])
    assert not ok and unc, "a narrowed pin was not caught"
    targets = G.gate_targets()                 # committed files only; no SSH
    assert len(targets) == 3 and all(t["script_path"] for t in targets), targets
    two = G.gate_targets(declared_nodes=["rrig6600", "rrig6600b"])
    assert [t["node_id"] for t in two] == ["rrig6600", "rrig6600b"], two
    raises(Exception, lambda: G.gate_targets(declared_nodes=["no_such_rig"]))
    return (f"pins cover derived closures; narrowed pin refuses; committed fleet "
            f"resolves to {[t['endpoint'] for t in targets]} (not probed)")


# ═════════════════════════════════════════════════════════════════════════════
# mutants
# ═════════════════════════════════════════════════════════════════════════════
_MUT_N = [0]


def _load_source(src, name, file=None):
    _MUT_N[0] += 1
    name = f"{name}_{_MUT_N[0]}"
    mod = types.ModuleType(name)
    # the REAL path, so path-derived constants (REPO_ROOT) are those of the
    # unmutated module and a kill cannot come from a broken __file__
    mod.__file__ = file or f"<{name}>"
    sys.modules[name] = mod                       # dataclasses need it
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    return mod


def _mutate(path, old, new):
    src = open(path, encoding="utf-8").read()
    n = src.count(old)
    assert n == 1, f"mutant anchor occurs {n} times in {os.path.basename(path)}: {old!r}"
    return src.replace(old, new)


MUTANTS = [
    ("M-S1a pass completeness forced true", PWC_PATH,
     '        _pass_complete = (\n'
     '            _outcomes["completed"] == len(chunks)\n'
     '            and (_merged == [[_req_start, _req_end]] or _req_start == _req_end))\n',
     '        _pass_complete = True\n', d_s1_failed_chunk, "pwc"),
    ("M-S1b timed-out chunk not counted (outcome = completed)", PWC_PATH,
     '        return "timed_out", msg[:200]\n', '        return "completed", None\n',
     d_s1_timed_out_chunk, "pwc"),
    ("M-S1c unaccounted pass treated as complete", PWC_PATH,
     '            status = "unaccounted"\n', '            status = "complete"\n',
     d_s1_unaccounted, "pwc"),
    ("M-S1d trial ignores pass accounting", PWC_PATH,
     '        elif acc.get("complete") and list(acc.get("requested_interval", [])) == req:\n',
     '        elif True:\n', d_s1_trial, "pwc"),
    ("M-S2a partition restarts at seed 0", PWC_PATH,
     '        seed   = _req_start\n', '        seed   = 0\n', d_s2_two31, "pwc"),
    ("M-S2b end bound check removed", PWC_PATH,
     '    if end > PWC_SEED_SPACE_END:\n', '    if False:\n', d_s2_endpoints, "pwc"),
    ("M-S3a dial address not checked at startup", PWC_PATH,
     '        if self.pwc_transport == "tcp" and rigs:\n', '        if False:\n',
     d_s3_unset, "pwc"),
    ("M-S3b worker falls back to 192.168.3.127", WORKER_PATH,
     '    host      = _required_env("PWC_HOST")\n',
     '    host      = os.environ.get("PWC_HOST", "192.168.3.127")\n',
     d_s3_worker, "src"),
    ("M-S4a unknown transport accepted", PWC_PATH,
     '    if pwc_transport not in PWC_TRANSPORTS:\n', '    if False:\n',
     d_s4_transport, "pwc"),
    ("M-S4c factory default transport restored", TB_PATH,
     'def build_transport(pwc_transport: str, **kwargs)',
     'def build_transport(pwc_transport: str = "ssh", **kwargs)',
     d_s4_factory, "tb"),
    ("M-S4b sessions/skip_range fallback restored", PWC_PATH,
     '        if sessions is None or skip_range is None:\n', '        if False:\n',
     d_s4_config_and_fields, "pwc"),
    ("M-S5a envelope admits everything", PWC_PATH,
     '        if w <= max_w and c <= max_c:\n', '        if True:\n',
     d_s5_envelope, "pwc"),
    ("M-S5b startup envelope check skipped", PWC_PATH,
     '        # [PWC-BACKUP] refuse before anything touches a rig\n        self._startup_refusals()\n',
     '', d_s5_envelope, "pwc"),
    ("M-S6 threshold disagreement not refused", PWC_PATH,
     '            assert_pwc_threshold_contract(_thr_rec)\n', '', d_s6_disagree, "pwc"),
    ("M-S7 worker_id default removed", PROTO_PATH,
     '    worker_id: str = ""\n', '    worker_id: str\n', d_s7_import, "import"),
    ("M-S8a zero rigs pass", GATE_PATH,
     '    if not results:\n', '    if False:\n', d_s8_gate, "gate"),
    ("M-S8b UNAVAILABLE not refused", GATE_PATH,
     '        if r["status"] == PROBE_UNAVAILABLE:\n            refusals.append(',
     '        if r["status"] == PROBE_UNAVAILABLE:\n            continue\n            refusals.append(',
     d_s8_gate, "gate"),
]

GATES = [
    ("G-S1-CLEAN", lambda: d_s1_clean(PWC)),
    ("G-S1-FAILED", lambda: d_s1_failed_chunk(PWC)),
    ("G-S1-TIMEOUT", lambda: d_s1_timed_out_chunk(PWC)),
    ("G-S1-ABORT", lambda: d_s1_pass_abort(PWC)),
    ("G-S1-RANGE", lambda: d_s1_range_mismatch(PWC)),
    ("G-S1-TRIAL", lambda: d_s1_trial(PWC)),
    ("G-S1-UNACCOUNTED", lambda: d_s1_unaccounted(PWC)),
    ("G-S2-ZERO-BYTE-EQUAL", lambda: d_s2_zero_byte_equal(PWC)),
    ("G-S2-2^31", lambda: d_s2_two31(PWC)),
    ("G-S2-ENDPOINTS", lambda: d_s2_endpoints(PWC)),
    ("G-S3-UNSET", lambda: d_s3_unset(PWC)),
    ("G-S3-SET", lambda: d_s3_set(PWC)),
    ("G-S3-WORKER", lambda: d_s3_worker()),
    ("G-S4-TRANSPORT", lambda: d_s4_transport(PWC)),
    ("G-S4-FACTORY", lambda: d_s4_factory()),
    ("G-S4-FIELDS", lambda: d_s4_config_and_fields(PWC)),
    ("G-S5-ENVELOPE", lambda: d_s5_envelope(PWC)),
    ("G-S6-AGREE", lambda: d_s6_agree(PWC)),
    ("G-S6-DISAGREE", lambda: d_s6_disagree(PWC)),
    ("G-S6-HYBRID-NEEDS-THRESHOLD", lambda: d_s6_hybrid_needs_threshold(PWC)),
    ("G-S7-IMPORT", lambda: d_s7_import()),
    ("G-S8-GATE", lambda: d_s8_gate(PG)),
    ("G-S8-CLOSURE-FLEET", lambda: d_s8_closure_and_fleet(PG)),
]


# [PWC-STAGE1] HELD, not run. Both check persistent/pwc_worker_service.py's
# `_required_env` (candidate S3, worker side). That hunk is NOT applied in stage 1:
# the stage-1 order forbids editing "the worker", and this is the PWC worker. The
# hunk is kept as docs/drafts/pwc_backup_stage1_v1/HELD_pwc_worker_service.diff for
# the owner. HELD is printed as HELD, never as PASS, and the RESULT line says so.
HELD = {
    "G-S3-WORKER": "needs the held pwc_worker_service.py hunk (owner decision)",
    "M-S3b worker falls back to 192.168.3.127": "needs the held pwc_worker_service.py hunk (owner decision)",
}


def run_mutant(label, path, old, new, detector, kind):
    src = _mutate(path, old, new)                                   # (1) applies once
    if kind == "pwc":
        detector(PWC)                                               # (2) control
        target = _load_source(src, "pwc_mutant", PWC_PATH)
    elif kind == "gate":
        detector(PG)
        target = _load_source(src, "gate_mutant", GATE_PATH)
    elif kind == "tb":
        detector()
        target = _load_source(src, "tb_mutant", TB_PATH)
    else:                                                           # source-level
        detector()
        target = src
    # (3) the detector must fail with its OWN assertion — any other exception is
    # unattributable. The S7 mutant is the exception: the defect IS the import
    # TypeError, so that is the expected kill.
    kill = (TypeError,) if kind == "import" else (AssertionError,)
    try:
        detector(target)
    except kill as exc:
        return f"killed: {type(exc).__name__}: {str(exc)[:110]}"
    except Exception as exc:                                        # noqa: BLE001
        raise AssertionError(f"UNATTRIBUTABLE kill of {label}: {type(exc).__name__}: {exc}")
    raise AssertionError(f"SURVIVED: {label}")


def _results_listing():
    d = os.path.join(ROOT, "results")
    if not os.path.isdir(d):
        return None
    return sorted((n, os.stat(os.path.join(d, n)).st_mtime_ns) for n in os.listdir(d))


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)
    repo_results_before = _results_listing()
    run_cwd = tempfile.mkdtemp(prefix="cwd_", dir=OUT)
    os.chdir(run_cwd)             # Finding 2: relative writes land here, not in ROOT
    logging_null = io.StringIO()
    import logging
    logging.getLogger("PersistentWorkerCoordinator").handlers[:] = [
        logging.StreamHandler(logging_null)]
    passed = 0
    held = []
    for name, fn in GATES:
        if name in HELD:
            held.append(name)
            print(f"[HELD] {name}: {HELD[name]}")
            continue
        try:
            msg = fn()
            passed += 1
            print(f"[PASS] {name}: {msg}")
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    killed = 0
    for label, path, old, new, det, kind in MUTANTS:
        if label in HELD:
            held.append(label)
            print(f"[HELD] {label}: {HELD[label]}")
            continue
        try:
            msg = run_mutant(label, path, old, new, det, kind)
            killed += 1
            print(f"[PASS] {label} -> {msg}")
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
    os.chdir(ROOT)
    contained = _results_listing() == repo_results_before
    stray = sorted(os.listdir(os.path.join(run_cwd, "results"))) \
        if os.path.isdir(os.path.join(run_cwd, "results")) else []
    print(f"[{'PASS' if contained else 'FAIL'}] CONTAINMENT: repository results/ "
          f"unchanged; the suite's own relative writes went to {run_cwd}: {stray}")
    print("")
    n_g = len([g for g, _ in GATES if g not in HELD])
    n_m = len([m for m, *_ in MUTANTS if m not in HELD])
    print(f"GATES {passed}/{n_g} PASS ({len(GATES) - n_g} HELD)")
    print(f"MUTANTS {killed}/{n_m} killed ({len(MUTANTS) - n_m} HELD)")
    ok = passed == n_g and killed == n_m and contained
    print(("RESULT: PASS" if ok else "RESULT: FAIL")
          + (f" — {len(held)} HELD, not run: {held}" if held else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
