#!/usr/bin/env python3
"""test_pwc_backup_stage2.py — acceptance harness for PWC full-backup STAGE 2.

What Stage 2 changes (docs/drafts/pwc_backup_stage2_v1/README.md has file:line):
  ruling 1  the variable-skip quarantine is lifted; the threshold contract replaces it
            (no 0.5 default, one value written as both keys, the worker's report
            checked, SSH refused for variable-skip passes);
  ruling 4  no early stop: phases 1-2 always, 3-4 whenever test_both_modes;
  default   all five hybrid strategies (S147 Q2 single strategy removed);
  owner     window offset != 0 refused (2026-11-20), until the anchor/phase split lands;
            [PWC-BACKUP STAGE 2b] it has landed: TCP carries window_anchor + generator_phase 0
            and the echo is checked; the refusal stays on the SSH route only (G-ANCHOR);
  Beta-2    a process check must prove it can see processes (positive control) and
            read pgrep's exit status exactly (rig kill, S156 find/reap, VM101 kill);
  Beta-3    revocation drops a connection the id already holds; an unconfirmed launch
            is not counted; only a connection that still owns an id cleans up under it;
  Beta-4    a worker exits on an explicit hello refusal instead of retrying forever.

Run from the repository root:   python3 -u tests/test_pwc_backup_stage2.py

CPU only. No GPU, no rig, no ssh, no network beyond 127.0.0.1:
  * every pkill/pgrep this suite EXECUTES runs inside a fresh unprivileged user + PID
    namespace (unshare --user --map-root-user --pid --fork --mount-proc); without
    unshare those gates report UNAVAILABLE and the suite does not pass;
  * transport gates bind a real TCPWorkerTransport on 127.0.0.1, port 0;
  * the worker-exit gate runs the real worker service in a child process against that
    loopback transport, with its heartbeat directory pointed under OUT;
  * the echo gate runs the REAL sieve_filter.execute_sieve_job in a child process with
    the GPU sieve methods replaced (no device is touched);
  * everything written goes under <root>/logs/pwc_stage2_test/ (an ignored path).

Every repair has a MUTANT: a textual edit of the LIVE source that undoes it, loaded as a
separate module (or run from a temporary copy). A mutant counts as killed only if the
edit applies exactly once, the detector passes on the live source (positive control),
and the detector fails on the mutant with its own AssertionError (VIR-2).
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = os.path.join(ROOT, "logs", "pwc_stage2_test")
PWC_PATH = os.path.join(ROOT, "persistent_worker_coordinator.py")
TCP_PATH = os.path.join(ROOT, "persistent", "pwc_transport_tcp.py")
SVC_PATH = os.path.join(ROOT, "persistent", "pwc_worker_service.py")

import persistent_worker_coordinator as PWC                 # noqa: E402
import persistent.pwc_transport_tcp as TCP                  # noqa: E402

RIGS = ("192.168.3.122", "192.168.3.156")
DIAL = "192.168.3.177"
FWD, REV = 0.73125, 0.31


class Unavailable(Exception):
    """The check could not run here (not a pass, not a fail)."""


def _load_source(src, name, path):
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def _mutate(path, old, new):
    src = open(path, encoding="utf-8").read()
    n = src.count(old)
    assert n == 1, f"mutant anchor occurs {n} times in {os.path.basename(path)}: {old[:80]!r}"
    return src.replace(old, new)


def raises(exc_type, fn):
    try:
        fn()
    except exc_type as e:
        return e
    except Exception as e:                                          # noqa: BLE001
        raise AssertionError(f"expected {exc_type.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc_type.__name__}; nothing was raised")


# ═════════════════════════════════════════════════════════════════════════════
# A coordinator on the TCP route with a fake transport and a fake worker
# ═════════════════════════════════════════════════════════════════════════════
def _write_config(rigs=RIGS, gpu_count=8, localhost=False):
    os.makedirs(OUT, exist_ok=True)
    nodes = [{"hostname": h, "gpu_type": "RX 6600 XT", "gpu_count": gpu_count,
              "python_env": "/home/michael/rocm_env/bin/python",
              "script_path": "/home/michael/distributed_prng_analysis",
              "username": "michael"} for h in rigs]
    if localhost:
        nodes.append({"hostname": "localhost", "gpu_type": "RTX 3080 Ti", "gpu_count": 2,
                      "python_env": sys.executable, "script_path": ROOT})
    path = tempfile.mktemp(suffix=".json", dir=OUT)
    with open(path, "w") as f:
        json.dump({"nodes": nodes}, f)
    return path


def sieve_like_payload(job, *, echo="exact", ok=True):
    """What sieve_filter.execute_sieve_job returns for one TCP job (final_result),
    reduced to the keys PWC reads. `echo`: 'exact' (the 4-place report sieve_filter
    makes), a float to report instead, or None to report nothing."""
    fam = job["prng_families"][0]
    per = {"survivors": [], "tested": job["seed_end"] - job["seed_start"]}
    if job.get("hybrid"):
        if echo == "exact":
            per["single_phase"] = {"threshold": round(job["phase2_threshold"], 4),
                                   "strategies_tested": len(job["strategies"] or [])}
        elif echo is not None:
            per["single_phase"] = {"threshold": echo, "strategies_tested": 5}
    out = {"success": ok, "seed_range": {"start": job["seed_start"], "end": job["seed_end"]},
           "min_match_threshold": job["min_match_threshold"], "survivors": [],
           "per_family": {fam: per}}
    return _anchor_echo(out, job)


def _anchor_echo(out, job):
    """[PWC-BACKUP STAGE 2b] sieve_filter.execute_sieve_job records window_anchor and
    generator_phase in its result for a new-key job (and PWC checks the echo)."""
    for k in ("window_anchor", "generator_phase"):
        if k in job:
            out[k] = job[k]
    return out


class FakeTransport:
    """Stands in for TCPWorkerTransport: records jobs, answers with provider(job)."""
    def __init__(self, provider, ready=2):
        self.provider, self.ready, self.jobs = provider, ready, []
        self._results = {}

    def ready_count(self):
        return self.ready

    def worker_count(self):
        return self.ready

    def submit_job(self, job):
        self.jobs.append(json.loads(json.dumps(job)))
        payload = self.provider(job)
        self._results[job["job_id"]] = {"status": "ok", "result": payload,
                                        "worker_id": "192_168_3_122_gpu0"}

    def recv_result(self, timeout_s=None, job_id=None):
        return self._results.pop(job_id, None)


def make_tcp_pwc(M, provider, *, ready=2):
    p = M.PersistentWorkerCoordinator(config_file=_write_config(), worker_pool_size=6,
                                      seed_cap_amd=25_000, pwc_transport="tcp",
                                      pwc_coordinator_address=DIAL)
    p._tcp_transport = FakeTransport(provider, ready=ready)
    p._effective_pool_by_host = {RIGS[0]: 6}
    return p


def run_pass(p, prng, *, thr=FWD, phase2="same", offset=0, total=40_000, strategies=None):
    kw = {}
    if phase2 == "same":
        kw["phase2_threshold"] = thr
    elif phase2 is not None:
        kw["phase2_threshold"] = phase2
    with contextlib.redirect_stderr(io.StringIO()):
        return p.run_sieve_pass(
            prng_type=prng, residues=[1, 2, 3], total_seeds=total, threshold=thr,
            window_size=8, output_file=os.path.join(OUT, f"pass_{prng}.json"),
            dataset_path="unused.json", sessions=["midday", "evening"],
            skip_range=[0, 10], offset=offset, strategies=strategies, **kw)


# ═════════════════════════════════════════════════════════════════════════════
# Ruling 1 — the threshold contract
# ═════════════════════════════════════════════════════════════════════════════
def d_thr_jobs(M=PWC):
    """Every job carries the one requested value as BOTH keys (constant and hybrid)."""
    for prng, thr in (("java_lcg", FWD), ("java_lcg_reverse", REV),
                      ("java_lcg_hybrid", FWD), ("java_lcg_hybrid_reverse", REV)):
        p = make_tcp_pwc(M, sieve_like_payload)
        res = run_pass(p, prng, thr=thr)
        jobs = p._tcp_transport.jobs
        assert jobs, f"{prng}: nothing dispatched"
        for j in jobs:
            assert j["min_match_threshold"] == thr and j["phase2_threshold"] == thr, \
                f"{prng}: job carries min={j['min_match_threshold']} phase2={j['phase2_threshold']}, want {thr}"
        rec = res["pwc_pass_accounting"]["threshold"]
        assert rec["agree"] and rec["verified"], f"{prng}: {rec}"
        if "_hybrid" in prng:
            assert rec["kind"] == "hybrid" and rec["effective"] == [round(thr, 4)], rec
        assert res["pwc_pass_accounting"]["transport"] == "tcp"
    # a constant pass may omit phase2_threshold; the job then carries `threshold`
    p = make_tcp_pwc(M, sieve_like_payload)
    run_pass(p, "java_lcg", thr=FWD, phase2=None)
    assert all(j["phase2_threshold"] == FWD for j in p._tcp_transport.jobs)
    return "constant and hybrid jobs carry the requested value as both keys; records verified"


def d_thr_refusals(M=PWC):
    # no phase2_threshold on a hybrid pass
    p = make_tcp_pwc(M, sieve_like_payload)
    e = raises(M.PwcSettingRefused, lambda: run_pass(p, "java_lcg_hybrid", phase2=None))
    assert "needs phase2_threshold" in str(e) and not p._tcp_transport.jobs, e
    # phase2 != threshold (the old 0.5)
    p = make_tcp_pwc(M, sieve_like_payload)
    e = raises(M.PwcSettingRefused, lambda: run_pass(p, "java_lcg_hybrid", phase2=0.5))
    assert "!= threshold" in str(e) and not p._tcp_transport.jobs, e
    # SSH route (no TCP transport): a hybrid pass refuses before any dispatch
    p = M.PersistentWorkerCoordinator(config_file=_write_config(), worker_pool_size=6,
                                      seed_cap_amd=25_000, pwc_transport="ssh")
    sent = []
    p._dispatch_to_worker = lambda h, j: sent.append(j) or {}
    p._dispatch_local_sieve = lambda j, n: sent.append(j) or {}
    e = raises(M.PwcSettingRefused, lambda: run_pass(p, "java_lcg_hybrid_reverse", thr=REV))
    assert "need the TCP transport" in str(e) and not sent, e
    return "no threshold / threshold != requested / SSH route: hybrid pass refused, nothing dispatched"


def d_thr_echo(M=PWC):
    """The worker's report is checked: wrong value, or none, fails the pass."""
    p = make_tcp_pwc(M, lambda j: sieve_like_payload(j, echo=0.5))
    e = raises(M.PwcThresholdContractMismatch, lambda: run_pass(p, "java_lcg_hybrid"))
    assert "0.5" in str(e), e
    p = make_tcp_pwc(M, lambda j: sieve_like_payload(j, echo=None))
    e = raises(M.PwcThresholdContractMismatch, lambda: run_pass(p, "java_lcg_hybrid"))
    assert "no reported threshold=" in str(e), e
    # one chunk of several reports the wrong value
    calls = {"n": 0}

    def one_bad(j):
        calls["n"] += 1
        return sieve_like_payload(j, echo=0.5 if calls["n"] == 2 else "exact")
    p = make_tcp_pwc(M, one_bad)
    raises(M.PwcThresholdContractMismatch, lambda: run_pass(p, "java_lcg_hybrid_reverse", thr=REV))
    return "hybrid report 0.5 for 0.73125 / no report / one bad chunk: the pass refuses"


_ECHO_CHILD = r'''
import json, sys, types
sys.path.insert(0, sys.argv[1])
jobs = json.load(open(sys.argv[2]))
import sieve_filter as SF
SF.load_draws_from_daily3 = lambda path, w, sessions, offset: list(range(w))
class _Sieve:
    def __init__(self, gpu_id=0): pass
    def run_hybrid_sieve(self, prng_family, seed_start, seed_end, residues, strategies,
                         min_match_threshold=0.25, chunk_size=100_000, offset=0,
                         miner_abi=False):
        return {"family": prng_family, "seed_range": {"start": seed_start, "end": seed_end},
                "survivors": [], "strategies_tested": len(strategies),
                "stats": {"seeds_tested": seed_end - seed_start, "survivors_found": 0,
                          "duration_ms": 1.0, "seeds_per_sec": 1.0},
                "_threshold_seen": min_match_threshold, "_offset_seen": offset,
                "_miner_abi_seen": miner_abi}
    def run_sieve(self, **kw):
        raise AssertionError("constant sieve called for a hybrid job")
SF.GPUSieve = _Sieve
out = []
for j in jobs:
    out.append(SF.execute_sieve_job(j, 0))
json.dump(out, open(sys.argv[3], "w"))
'''


def g_echo_real_sieve_filter():
    """The REAL sieve_filter.execute_sieve_job (child process, GPU methods replaced):
    the report PWC reads is where PWC reads it, and equals the value the job carried
    to the sieve call, to 4 places. Not a mutant gate: it checks the shape PWC relies on."""
    p = make_tcp_pwc(PWC, sieve_like_payload)
    run_pass(p, "java_lcg_hybrid", thr=FWD, total=20_000)
    jobs = p._tcp_transport.jobs
    jpath, opath = os.path.join(OUT, "echo_jobs.json"), os.path.join(OUT, "echo_out.json")
    # [PWC-BACKUP STAGE 2b] a new-key job's anchor is checked against the dataset's
    # length before the (stubbed) loader runs, so the child needs a real file
    dpath = os.path.join(OUT, "echo_dataset.json")
    json.dump([{"draw": i, "session": "midday" if i % 2 else "evening"} for i in range(40)],
              open(dpath, "w"))
    jobs = [dict(j, dataset_path=dpath) for j in jobs]
    json.dump(jobs, open(jpath, "w"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", _ECHO_CHILD, ROOT, jpath, opath],
                       capture_output=True, text=True, timeout=120, env=env)
    if r.returncode != 0 or not os.path.exists(opath):
        raise Unavailable(f"sieve_filter child rc={r.returncode}: {r.stderr.strip()[-200:]!r}")
    results = json.load(open(opath))
    for job, res in zip(jobs, results):
        assert res.get("success"), res.get("error")
        fam = job["prng_families"][0]
        seen = res["per_family"][fam]["_threshold_seen"]
        assert seen == job["phase2_threshold"] == FWD, (seen, job["phase2_threshold"])
        assert res["per_family"][fam]["_offset_seen"] == 0
        # [PWC-BACKUP STAGE 2b] a PWC job is a new-key job: the miner's hybrid ABI
        assert res["per_family"][fam]["_miner_abi_seen"] is True
        assert res.get("window_anchor") == job["window_anchor"] == 0, res.get("window_anchor")
        assert res.get("generator_phase") == job["generator_phase"] == 0
        echo = PWC._pwc_hybrid_echo(res, job)
        assert echo == round(FWD, 4), (echo, FWD)
    # and the whole record built from those real reports verifies
    by_range = {(r["seed_range"]["start"], r["seed_range"]["end"]): r for r in results}
    p2 = make_tcp_pwc(PWC, lambda j: by_range[(j["seed_start"], j["seed_end"])])
    res = run_pass(p2, "java_lcg_hybrid", thr=FWD, total=20_000)
    rec = res["pwc_pass_accounting"]["threshold"]
    assert rec["verified"] and rec["effective"] == [round(FWD, 4)], rec
    return (f"real execute_sieve_job: sieve got {FWD} (the job's phase2_threshold), reported "
            f"{round(FWD, 4)} at per_family[..].single_phase.threshold; record verified")


# ═════════════════════════════════════════════════════════════════════════════
# Ruling 4 + strategies + offset + SSH: run_trial_persistent
# ═════════════════════════════════════════════════════════════════════════════
class _Cfg:
    def __init__(self, offset=0):
        self.window_size, self.offset, self.sessions = 8, offset, ["midday", "evening"]
        self.skip_min, self.skip_max = 0, 10


def drive_trial(M, table, *, both=True, offset=0, transport="tcp", base="java_lcg"):
    calls, made = [], []

    class Fake:
        _progress_writer = None
        logger = types.SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None,
                                       error=lambda *a, **k: None)

        def startup(self):
            pass

        def shutdown(self):
            pass

        def run_sieve_pass(self, prng_type=None, **kw):
            calls.append((prng_type, kw))
            s = table.get(prng_type, [])
            return {"survivors": list(s), "match_rates": [0.9] * len(s)}

    real = M.PersistentWorkerCoordinator
    M.PersistentWorkerCoordinator = lambda **kw: made.append(kw) or Fake()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = M.run_trial_persistent(
                coordinator_cfg="unused.json", config=_Cfg(offset), trial_number=3,
                prng_base=base, residues=[1, 2], total_seeds=1000, forward_threshold=FWD,
                reverse_threshold=REV, test_both_modes=both, dataset_path="unused.json",
                pwc_transport=transport)
    finally:
        M.PersistentWorkerCoordinator = real
    return out, calls, made


ALL4 = ["java_lcg", "java_lcg_reverse", "java_lcg_hybrid", "java_lcg_hybrid_reverse"]


def d_phases(M=PWC):
    out, calls, _ = drive_trial(M, {})                       # zero survivors everywhere
    assert [c[0] for c in calls] == ALL4, f"forward zero: {[c[0] for c in calls]}"
    assert out["pruned"] is False and out["reason"] is None, out.get("reason")
    out, calls, _ = drive_trial(M, {"java_lcg": [7], "java_lcg_reverse": [7]})
    assert [c[0] for c in calls] == ALL4, f"variable forward zero: {[c[0] for c in calls]}"
    out, calls, _ = drive_trial(M, {"java_lcg": [7]}, both=False)
    assert [c[0] for c in calls] == ALL4[:2], [c[0] for c in calls]
    return "zero forward survivors: all 4 phases; zero variable-forward: phase 4 still runs; both=False: 1-2"


def d_trial_thresholds_and_strategies(M=PWC):
    _, calls, _ = drive_trial(M, {"java_lcg": [7]})
    want = {"java_lcg": FWD, "java_lcg_reverse": REV, "java_lcg_hybrid": FWD,
            "java_lcg_hybrid_reverse": REV}
    for prng, kw in calls:
        assert kw.get("threshold") == want[prng] and kw.get("phase2_threshold") == want[prng], \
            f"{prng}: threshold={kw.get('threshold')!r} phase2_threshold={kw.get('phase2_threshold')!r}"
        if "_hybrid" in prng:
            assert "strategies" in kw and kw["strategies"] is None, \
                f"{prng}: strategies={kw.get('strategies', 'MISSING')!r}"
    return "each pass gets its directional threshold as both keys; hybrid passes strategies=None"


def d_strategies_all_five(M=PWC):
    from hybrid_strategy import get_all_strategies
    want = [s.to_dict() if hasattr(s, "to_dict") else s for s in get_all_strategies()]
    p = make_tcp_pwc(M, sieve_like_payload)
    run_pass(p, "java_lcg_hybrid", strategies=None)
    got = p._tcp_transport.jobs[0]["strategies"]
    assert got == json.loads(json.dumps(want)) and len(got) == len(want) == 5, \
        f"{len(got)} strategies sent: {[s.get('name') for s in got]}"
    return f"strategies=None sends all {len(want)}: {[s.get('name') for s in got]}"


def _ssh_pass(M, offset):
    """A run_sieve_pass on the SSH route (no TCP transport). Dispatch is stubbed so
    that a pass which should have been refused finishes instead of blocking."""
    import queue as _q
    p = make_tcp_pwc(M, sieve_like_payload)
    p._tcp_transport = None
    _node = types.SimpleNamespace(hostname="localhost", gpu_type="stub", gpu_count=1)
    p._get_available_workers = lambda *a, **k: [_node]
    p._localhost_gpu_slots = _q.Queue()
    p._localhost_gpu_slots.put(0)
    p._dispatch_local_sieve = lambda job, wh: {"status": "error", "message": "stub"}
    p._dispatch_to_worker = lambda wh, job: {"status": "error", "message": "stub"}
    return run_pass(p, "java_lcg", offset=offset)


def d_anchor(M=PWC):
    """[PWC-BACKUP STAGE 2b, owner ruling 2026-11-20 08:30 PST] The trial's offset picks
    the draw window only; the generator phase is 0; RANGE-MINER's method. On TCP the
    anchor travels as `window_anchor` with `generator_phase` 0 and NO `offset`, and the
    worker's echo of both is checked. The SSH route ends in sieve_gpu_worker.py (pinned),
    so there offset != 0 is still refused (Stage 2's refusal)."""
    # TCP: anchor 25 runs, every job carries the split, the pass is complete
    p = make_tcp_pwc(M, sieve_like_payload)
    res = run_pass(p, "java_lcg", offset=25)
    jobs = p._tcp_transport.jobs
    assert jobs and all(j.get("window_anchor") == 25 and j.get("generator_phase") == 0
                        and "offset" not in j for j in jobs), jobs[:1]
    assert res["pwc_pass_accounting"]["complete"], res["pwc_pass_accounting"]
    # a worker without the split echoes nothing: no chunk counts
    p = make_tcp_pwc(M, lambda j: {k: v for k, v in sieve_like_payload(j).items()
                                   if k not in ("window_anchor", "generator_phase")})
    acc = run_pass(p, "java_lcg", offset=25)["pwc_pass_accounting"]
    assert not acc["complete"] and acc["completed"] == 0, acc
    # a worker that echoes another anchor (or phase): no chunk counts
    for bad in ({"window_anchor": 0}, {"generator_phase": 25}):
        p = make_tcp_pwc(M, lambda j, b=bad: dict(sieve_like_payload(j), **b))
        acc = run_pass(p, "java_lcg", offset=25)["pwc_pass_accounting"]
        assert not acc["complete"] and acc["completed"] == 0, (bad, acc)
    # non-integer or negative anchor: refused before anything is dispatched
    for bad in (-1, "25", 2.0, True):
        p = make_tcp_pwc(M, sieve_like_payload)
        raises(M.PwcWindowAnchorRefused, lambda: run_pass(p, "java_lcg", offset=bad))
        assert not p._tcp_transport.jobs, bad
    # SSH trial with offset 25: refused before the fleet is built
    real = M.PersistentWorkerCoordinator
    built = []
    M.PersistentWorkerCoordinator = lambda **kw: built.append(kw)
    try:
        e = raises(M.PwcWindowAnchorRefused, lambda: M.run_trial_persistent(
            coordinator_cfg="unused.json", config=_Cfg(25), trial_number=1,
            prng_base="java_lcg", residues=[1], total_seeds=10, forward_threshold=FWD,
            reverse_threshold=REV, test_both_modes=False, pwc_transport="ssh"))
    finally:
        M.PersistentWorkerCoordinator = real
    assert not built and "25" in str(e), (built, e)
    # SSH pass with offset 3: refused; SSH pass with offset 0: not refused
    raises(M.PwcWindowAnchorRefused, lambda: _ssh_pass(M, 3))
    _ssh_pass(M, 0)
    # TCP trial with offset 25 runs all four passes, each given offset 25
    out, calls, _ = drive_trial(M, {}, offset=25)
    assert len(calls) == 4 and all(kw["offset"] == 25 for _, kw in calls), calls
    return ("TCP anchor 25: jobs carry window_anchor 25 / generator_phase 0 / no offset, "
            "echo checked (none, wrong anchor, wrong phase: no chunk counts); -1/'25'/2.0/True "
            "refused before dispatch; SSH: 25 refused before the fleet, pass 3 refused, 0 runs")


def d_ssh_variable(M=PWC):
    real = M.PersistentWorkerCoordinator
    built = []
    M.PersistentWorkerCoordinator = lambda **kw: built.append(kw)
    try:
        e = raises(M.PwcSettingRefused, lambda: M.run_trial_persistent(
            coordinator_cfg="unused.json", config=_Cfg(0), trial_number=1,
            prng_base="java_lcg", residues=[1], total_seeds=10, forward_threshold=FWD,
            reverse_threshold=REV, test_both_modes=True, pwc_transport="ssh"))
    finally:
        M.PersistentWorkerCoordinator = real
    assert "pwc_transport='tcp'" in str(e) and not built, e
    out, calls, _ = drive_trial(M, {}, both=False, transport="ssh")
    assert [c[0] for c in calls] == ALL4[:2]
    return "both-mode on SSH refused before the fleet; constant-only on SSH still runs"


# ═════════════════════════════════════════════════════════════════════════════
# Beta-2 — process checks prove they can see (isolated PID namespace)
# ═════════════════════════════════════════════════════════════════════════════
_HARNESS = r'''#!/bin/bash
a=pwc_worker; b=_service; c=sieve_gpu; d=_worker.py
DUMMY_PWC="${a}${b}"; DUMMY_SIEVE="/opt/x/${c}${d}"
pids=""; p_pwc=""; p_sieve=""
if [ "$WITH_DUMMIES" = 1 ]; then
  python3 -c 'import time; time.sleep(120)' "$DUMMY_PWC" --gpu-id 0 & p_pwc=$!
  python3 -c 'import time; time.sleep(120)' "$DUMMY_SIEVE" --persistent & p_sieve=$!
  pids="$p_pwc $p_sieve"
  sleep 0.4
fi
[ -n "$FAKE_DIR" ] && export PATH="$FAKE_DIR:$PATH"
bash -c "$CMD" > "$OUTDIR/stdout" 2> "$OUTDIR/stderr"
echo $? > "$OUTDIR/rc"
a_pwc=0; a_sieve=0
[ -n "$p_pwc" ] && kill -0 "$p_pwc" 2>/dev/null && a_pwc=1
[ -n "$p_sieve" ] && kill -0 "$p_sieve" 2>/dev/null && a_sieve=1
echo "$a_pwc $a_sieve" > "$OUTDIR/alive_each"
for p in $pids; do kill -9 "$p" 2>/dev/null; done
exit 0
'''

REAL_PGREP = shutil.which("pgrep") or "/usr/bin/pgrep"
FAKE_PGREP = {
    # pgrep that cannot run at all (Beta's reproduction: exit 2)
    "pgrep_exit2": "#!/bin/sh\nexit 2\n",
    # pgrep that runs but sees no process (e.g. hidden /proc): always "no match"
    "pgrep_blind": "#!/bin/sh\n[ \"$1\" = \"-c\" ] || [ \"$2\" = \"-c\" ] && echo 0\nexit 1\n",
    # pgrep that works for the control token and errors on the worker patterns
    "pgrep_err_on_target": "#!/bin/sh\ncase \"$*\" in *%s*) exec %s \"$@\";; esac\nexit 2\n"
                           % (PWC.PWC_PROBE_CONTROL_TOKEN, REAL_PGREP),
}


def run_isolated(cmd, *, dummies=True, fake=None):
    if shutil.which("unshare") is None:
        raise Unavailable("unshare not installed")
    d = tempfile.mkdtemp(prefix="iso_", dir=OUT)
    h = os.path.join(d, "harness.sh")
    with open(h, "w") as f:
        f.write(_HARNESS)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "CMD": cmd, "OUTDIR": d,
           "WITH_DUMMIES": "1" if dummies else "0", "FAKE_DIR": ""}
    if fake:
        fp = os.path.join(d, "fakebin")
        os.makedirs(fp)
        with open(os.path.join(fp, "pgrep"), "w") as f:
            f.write(FAKE_PGREP[fake])
        os.chmod(os.path.join(fp, "pgrep"), 0o755)
        env["FAKE_DIR"] = fp
    r = subprocess.run(["unshare", "--user", "--map-root-user", "--pid", "--fork",
                        "--mount-proc", "bash", h], env=env, capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0 or not os.path.exists(os.path.join(d, "rc")):
        raise Unavailable(f"unshare rc={r.returncode} stderr={r.stderr.strip()[:160]!r}")
    read = lambda n: open(os.path.join(d, n)).read()
    a_pwc, a_sieve = (int(x) for x in read("alive_each").split())
    return {"rc": int(read("rc")), "stdout": read("stdout").strip().splitlines(),
            "alive_pwc": a_pwc, "alive_sieve": a_sieve}


def d_probe_kill(M=PWC):
    cmd = M.pwc_stale_kill_command()
    ok = M.PWC_STALE_KILL_OK
    r = run_isolated(cmd)                                   # clean control
    assert r["rc"] == 0 and r["stdout"][-1:] == [ok] and not (r["alive_pwc"] or r["alive_sieve"]), r
    r = run_isolated(cmd, dummies=False)
    assert r["rc"] == 0 and r["stdout"][-1:] == [ok], r
    for fake in ("pgrep_exit2", "pgrep_blind", "pgrep_err_on_target"):
        r = run_isolated(cmd, dummies=False, fake=fake)
        assert r["rc"] == M.PWC_KILL_EXIT_UNAVAILABLE and ok not in r["stdout"], f"{fake}: {r}"
    loc = M.pwc_stale_kill_command(M.PWC_LOCAL_STALE_KILL_PATTERNS)
    r = run_isolated(loc)
    assert r["rc"] == 0 and r["stdout"][-1:] == [ok], r
    assert r["alive_pwc"] == 0 and r["alive_sieve"] == 1, f"local kill must touch only pwc: {r}"
    r = run_isolated(loc, dummies=False, fake="pgrep_blind")
    assert r["rc"] == M.PWC_KILL_EXIT_UNAVAILABLE, r
    return ("kill: real pgrep -> marker; pgrep exit 2 / blind / erroring on the target -> "
            "rc 7, no marker; VM101-local string kills only the pwc dummy")


def d_probe_s156(M=PWC):
    find, reap = M.pwc_s156_find_command(), M.pwc_s156_reap_command()
    U = M.PWC_S156_UNAVAILABLE
    r = run_isolated(find, dummies=False)
    assert r["stdout"] == ["none"], r
    r = run_isolated(find)
    assert len(r["stdout"]) == 1 and "--persistent" in r["stdout"][0], r
    r = run_isolated(reap)
    assert r["stdout"] == ["reaped:0"] and r["alive_sieve"] == 0 and r["alive_pwc"] == 1, r
    for fake in ("pgrep_exit2", "pgrep_blind", "pgrep_err_on_target"):
        r = run_isolated(find, dummies=False, fake=fake)
        assert r["stdout"] == [U], f"find/{fake}: {r}"
        r = run_isolated(reap, dummies=False, fake=fake)
        assert r["stdout"] == ["reaped:unavailable"], f"reap/{fake}: {r}"
    return ("S156 find: none / the worker / probe_unavailable; reap: 'reaped:0' exactly "
            "(no double zero) / 'reaped:unavailable' for all three broken probes")


# ═════════════════════════════════════════════════════════════════════════════
# Beta-2/3 wiring in _tcp_launch_workers (subprocess recorded, nothing sent)
# ═════════════════════════════════════════════════════════════════════════════
class Admission:
    def __init__(self):
        self.allowed, self.revoked = set(), []

    def set_allowed_worker_ids(self, ids):
        self.allowed = set(ids)

    def allow_worker_id(self, wid):
        self.allowed.add(wid)

    def revoke_worker_id(self, wid):
        self.allowed.discard(wid)
        self.revoked.append(wid)


def launch(M, *, local_kill, no_pid=(), localhost=True):
    cfg = _write_config(rigs=RIGS[:1], localhost=localhost)
    p = M.PersistentWorkerCoordinator(config_file=cfg, worker_pool_size=2, seed_cap_amd=25_000,
                                      pwc_transport="tcp", pwc_coordinator_address=DIAL)
    p._tcp_transport = Admission()
    p._probe_live_rocm_gpu_count = lambda node: 2
    p._reset_rig_cupy_cache = lambda *a: True
    waits, popens = [], []
    p._tcp_wait_online = lambda expected, timeout_s=0: waits.append(expected) or 0
    p._tcp_broadcast_init = lambda: 0
    p._tcp_wait_ready = lambda expected, timeout_s=0: 0
    ok = M.PWC_STALE_KILL_OK

    def fake_run(cmd, *a, **k):
        flat = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
        if isinstance(cmd, list) and cmd[:2] == ["bash", "-c"]:      # VM101-local kill
            return local_kill(cmd)
        if ok in flat:
            return subprocess.CompletedProcess(cmd, 0, ok + "\n", "")
        if "cat > " in flat:
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        if "bash /tmp/pwc_tcp_launch_gpu" in flat:
            gpu = int(flat.rsplit("gpu", 1)[1].split(".")[0])
            return subprocess.CompletedProcess(cmd, 0, "" if gpu in no_pid else "PID=4242\n", "")
        raise AssertionError(f"unexpected subprocess: {flat[:120]}")

    class FakePopen:
        def __init__(self, cmd, **kw):
            popens.append(kw.get("env", {}).get("PWC_WORKER_ID"))
            self.pid = 999

    import builtins

    def redirected_open(path, *a, **k):
        # the local launch appends to /tmp/pwc_tcp_worker_zeus_gpuN.log; keep it under OUT
        if isinstance(path, str) and path.startswith("/tmp/"):
            path = os.path.join(OUT, "tmp_redirect_" + os.path.basename(path))
        return builtins.open(path, *a, **k)

    saved = (subprocess.run, subprocess.Popen, time.sleep)
    subprocess.run, subprocess.Popen, time.sleep = fake_run, FakePopen, (lambda s: None)
    M.open = redirected_open
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            p._tcp_launch_workers()
    finally:
        subprocess.run, subprocess.Popen, time.sleep = saved
        del M.open
    return p, waits, popens


def d_local_kill_read(M=PWC):
    good = lambda c: subprocess.CompletedProcess(c, 0, M.PWC_STALE_KILL_OK + "\n", "")
    p, waits, popens = launch(M, local_kill=good)
    assert popens == ["zeus_gpu0", "zeus_gpu1"] and {"zeus_gpu0", "zeus_gpu1"} <= p._tcp_transport.allowed, popens
    for name, reply in (("rc 7", lambda c: subprocess.CompletedProcess(c, 7, "", "")),
                        ("rc 6", lambda c: subprocess.CompletedProcess(c, 6, "", "")),
                        ("no marker", lambda c: subprocess.CompletedProcess(c, 0, "", "")),
                        ("raises", lambda c: (_ for _ in ()).throw(OSError("no bash")))):
        p, waits, popens = launch(M, local_kill=reply)
        assert popens == [], f"{name}: local workers launched: {popens}"
        assert not any(w.startswith("zeus_") for w in p._tcp_transport.allowed), name
        assert waits == [2], f"{name}: online wait sized {waits}, want the 2 rig workers"
    return "VM101 kill confirmed -> 2 local workers; rc 7 / rc 6 / no marker / error -> none launched or admitted"


def d_launch_counted(M=PWC):
    good = lambda c: subprocess.CompletedProcess(c, 0, M.PWC_STALE_KILL_OK + "\n", "")
    p, waits, _ = launch(M, local_kill=good, no_pid=(1,), localhost=False)
    assert waits == [1], f"online wait sized {waits}; the no-PID launch must not count"
    assert p._effective_pool_by_host == {RIGS[0]: 1}, p._effective_pool_by_host
    assert p._tcp_transport.revoked == [f"{RIGS[0].replace('.', '_')}_gpu1"], p._tcp_transport.revoked
    return "a launch with no PID is revoked and not counted (online wait 1, effective pool 1)"


# ═════════════════════════════════════════════════════════════════════════════
# Beta-3 — the real transport on loopback
# ═════════════════════════════════════════════════════════════════════════════
def _send(sock, obj):
    b = json.dumps(obj, separators=(",", ":")).encode()
    sock.sendall(struct.pack(">I", len(b)) + b)


def _recv(sock):
    def n(k):
        buf = b""
        while len(buf) < k:
            c = sock.recv(k - len(buf))
            if not c:
                raise ConnectionError("closed")
            buf += c
        return buf
    return json.loads(n(struct.unpack(">I", n(4))[0]))


def _hello(port, wid):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    _send(s, {"message_type": "hello", "protocol_version": 1, "worker_id": wid,
              "timestamp": time.time()})
    return s, _recv(s)


def _wait(pred, t=3.0):
    end = time.time() + t
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _ready_worker(port, wid):
    s, ack = _hello(port, wid)
    assert ack.get("accepted") is True, ack
    _send(s, {"message_type": "online", "worker_id": wid, "timestamp": time.time()})
    _send(s, {"message_type": "ready", "worker_id": wid, "timestamp": time.time()})
    return s


def _transport(T, ids):
    tr = T.TCPWorkerTransport(host="127.0.0.1", port=0)
    tr.set_allowed_worker_ids(set(ids))
    try:
        tr.start()
    except OSError as e:
        raise Unavailable(f"cannot bind 127.0.0.1: {e}")
    return tr, tr._server.getsockname()[1]


def d_revoke_drops_connection(T=TCP):
    tr, port = _transport(T, {"rig_gpu0"})
    try:
        s = _ready_worker(port, "rig_gpu0")
        assert _wait(lambda: tr.ready_count() == 1), tr.ready_count()
        tr.revoke_worker_id("rig_gpu0")
        assert _wait(lambda: tr.ready_count() == 0 and tr.worker_count() == 0), \
            f"revoked worker still held: ready={tr.ready_count()} workers={tr.worker_count()}"
        s.settimeout(3)
        try:
            data = s.recv(1)
        except OSError:
            data = b""
        assert data == b"", "revoked worker's socket still open"
        s.close()
    finally:
        tr.stop()
    return "revoke: the id's open connection is closed, it leaves ready/online at once"


def d_request_job_admission(T=TCP):
    """An id no longer on the list (list replaced, no revoke call) gets no job."""
    tr, port = _transport(T, {"rig_gpu0"})
    try:
        s = _ready_worker(port, "rig_gpu0")
        assert _wait(lambda: tr.ready_count() == 1)
        tr.submit_job({"job_id": "j1", "seed_start": 0, "seed_end": 10})
        tr.set_allowed_worker_ids({"other"})
        _send(s, {"message_type": "request_job", "worker_id": "rig_gpu0", "timestamp": time.time()})
        s.settimeout(3)
        try:
            msg = _recv(s)
        except (ConnectionError, OSError):
            msg = None
        assert msg is None, f"a worker off the list was sent {msg}"
        assert _wait(lambda: tr.worker_count() == 0), tr.worker_count()
        assert tr._pending_jobs.qsize() == 1, "the job was taken by a worker off the list"
        s.close()
    finally:
        tr.stop()
    return "a worker no longer on the list asks for a job: no job, connection ended, job still queued"


def d_owner_cleanup(T=TCP):
    """A reconnect replaces the id's connection; when the OLD connection ends, the new
    one stays registered and ready."""
    tr, port = _transport(T, {"rig_gpu0"})
    try:
        old = _ready_worker(port, "rig_gpu0")
        assert _wait(lambda: tr.ready_count() == 1)
        new = _ready_worker(port, "rig_gpu0")
        assert _wait(lambda: tr._workers.get("rig_gpu0") is not None)
        time.sleep(0.2)
        old.close()
        time.sleep(0.5)
        assert tr.worker_count() == 1 and tr.ready_count() == 1, \
            f"old connection's end evicted the new one: workers={tr.worker_count()} ready={tr.ready_count()}"
        new.close()
        assert _wait(lambda: tr.worker_count() == 0 and tr.ready_count() == 0)
    finally:
        tr.stop()
    return "reconnect: the old connection ending leaves the new one registered and ready"


# ═════════════════════════════════════════════════════════════════════════════
# Beta-4 — a refused worker exits (real worker service, child process)
# ═════════════════════════════════════════════════════════════════════════════
_WORKER_CHILD = r'''
import importlib.util, sys
root, src_path, port, hb = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("svc_under_test", src_path)
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
W.HEARTBEAT_DIR = hb
W.RECONNECT_DELAY_S = 0.2
W.PWCWorkerService(worker_id="stale_gpu3", host="127.0.0.1", port=port, gpu_id=0,
                   use_rocm=False).run_forever()
'''


def d_worker_exits(src_path=SVC_PATH):
    tr, port = _transport(TCP, {"someone_else"})
    hb = os.path.join(OUT, "hb")
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        try:
            r = subprocess.run([sys.executable, "-c", _WORKER_CHILD, ROOT, src_path,
                                str(port), hb], capture_output=True, text=True,
                               timeout=8, env=env)
        except subprocess.TimeoutExpired:
            raise AssertionError(f"refused worker still running after 8 s "
                                 f"({tr.refused_hellos.count('stale_gpu3')} hellos refused)")
    finally:
        tr.stop()
    assert r.returncode == 43, f"rc={r.returncode} stderr={r.stderr.strip()[-200:]!r}"
    assert tr.refused_hellos.count("stale_gpu3") == 1, tr.refused_hellos
    return "real worker service, refused hello: exits rc 43 after ONE attempt"


# ═════════════════════════════════════════════════════════════════════════════
# [R2] Beta X-PWC-STAGE2 R1 Q6 — reconnect overlapping the old handler's cleanup
# ═════════════════════════════════════════════════════════════════════════════
def d_reconnect_race(T=TCP):
    """Beta's interleaving, forced: the OLD connection's handler is paused inside its
    cleanup, right after it has dealt with the registry; meanwhile the SAME worker
    reconnects, registers, reports ready and leases a job. When the old handler
    resumes, the replacement must still be registered and ready, its lease must still
    be active (not requeued, not failed), and its result must still be accepted."""
    tr, port = _transport(T, {"rig_gpu0"})
    gate, entered, calls = threading.Event(), threading.Event(), []
    real = tr._reclaim_worker_jobs

    def paused(worker_id, conn=None):
        if not calls:
            calls.append(worker_id)
            entered.set()
            gate.wait(10)
        return real(worker_id, conn)
    tr._reclaim_worker_jobs = paused
    new = None
    try:
        old = _ready_worker(port, "rig_gpu0")
        assert _wait(lambda: tr.ready_count() == 1)
        old.close()
        assert entered.wait(5), "the old handler never reached its cleanup"
        new = _ready_worker(port, "rig_gpu0")
        assert _wait(lambda: tr.ready_count() == 1 and tr.worker_count() == 1), \
            f"replacement not ready: ready={tr.ready_count()} workers={tr.worker_count()}"
        tr.submit_job({"job_id": "jR", "seed_start": 0, "seed_end": 10})
        _send(new, {"message_type": "request_job", "worker_id": "rig_gpu0", "timestamp": time.time()})
        new.settimeout(3)
        msg = _recv(new)
        assert msg.get("job_id") == "jR", msg
        lease = msg["lease_id"]
        gate.set()
        time.sleep(0.5)
        assert tr.worker_count() == 1 and tr.ready_count() == 1, \
            f"old cleanup hit the replacement: workers={tr.worker_count()} ready={tr.ready_count()}"
        info = tr._inflight.get("jR")
        assert info is not None and info["lease_id"] == lease, \
            f"the replacement's active lease was reclaimed (pending={tr._pending_jobs.qsize()})"
        assert tr._pending_jobs.qsize() == 0, "the replacement's job was requeued"
        _send(new, {"message_type": "result_inline", "job_id": "jR", "lease_id": lease,
                    "worker_id": "rig_gpu0", "result": {"success": True, "payload": {}}})
        r = tr.recv_result(timeout_s=3, job_id="jR")
        assert r is not None and r.get("status") == "ok", f"replacement's result not delivered: {r}"
    finally:
        gate.set()
        if new is not None:
            new.close()
        tr.stop()
    return ("old handler paused mid-cleanup while the worker reconnected, went ready and "
            "leased a job: replacement stays registered + ready, lease active, result delivered")


def d_reaper_ownership(T=TCP):
    """Timeout eviction under the same rule: the silent connection's id state is
    cleared at once, only its own lease is reclaimed, and a fresh one is left alone."""
    tr, port = _transport(T, {"rig_gpu0", "rig_gpu1"})
    try:
        s = _ready_worker(port, "rig_gpu0")
        f = _ready_worker(port, "rig_gpu1")
        assert _wait(lambda: tr.ready_count() == 2)
        tr.submit_job({"job_id": "jP", "seed_start": 0, "seed_end": 10})
        _send(s, {"message_type": "request_job", "worker_id": "rig_gpu0", "timestamp": time.time()})
        s.settimeout(3)
        assert _recv(s).get("job_id") == "jP"
        with tr._last_seen_lock:
            tr._last_seen["rig_gpu0"] = time.time() - (T.HEARTBEAT_TIMEOUT_S + 10)
        tr._reap_silent_workers(time.time())
        assert "rig_gpu0" not in tr._workers and tr.ready_count() == 1, \
            f"silent worker not fully evicted: workers={sorted(tr._workers)} ready={tr.ready_count()}"
        assert "jP" not in tr._inflight and tr._pending_jobs.qsize() == 1, "its lease was not requeued"
        assert "rig_gpu1" in tr._workers, "the fresh worker was evicted"
        try:
            data = s.recv(1)
        except OSError:
            data = b""
        assert data == b"", "evicted worker's socket still open"
        s.close(); f.close()
    finally:
        tr.stop()
    return "silent worker: registry + ready cleared at once, its lease requeued, socket shut; fresh worker kept"


# ═════════════════════════════════════════════════════════════════════════════
# [R2] Beta X-PWC-STAGE2 R1 finding 2 — late results
# ═════════════════════════════════════════════════════════════════════════════
def d_job_ids_unique_per_pass(M=PWC):
    ids = []
    for prng in ("java_lcg", "java_lcg_reverse"):
        p = make_tcp_pwc(M, sieve_like_payload)
        run_pass(p, prng, thr=0.5)
        ids.append([j["job_id"] for j in p._tcp_transport.jobs])
    a, b = ids
    assert len(set(a)) == len(a) and len(set(b)) == len(b), ids
    assert not set(a) & set(b), f"two passes share job ids: {sorted(set(a) & set(b))}"
    return f"P1 {a[0]}.. / P2 {b[0]}..: no job id shared between passes (equal thresholds 0.5)"


def d_active_lease_only(T=TCP):
    """Within one job id: after the job is re-leased to another worker, the first
    worker's late result neither retires the active lease nor is delivered."""
    tr, port = _transport(T, {"wA", "wB"})
    try:
        a = _ready_worker(port, "wA")
        b = _ready_worker(port, "wB")
        assert _wait(lambda: tr.ready_count() == 2)
        job = {"job_id": "jX", "seed_start": 0, "seed_end": 10}
        tr.submit_job(dict(job))
        _send(a, {"message_type": "request_job", "worker_id": "wA", "timestamp": time.time()})
        a.settimeout(3); b.settimeout(3)
        la = _recv(a)["lease_id"]
        tr.submit_job(dict(job))                      # the coordinator's retry after a timeout
        _send(b, {"message_type": "request_job", "worker_id": "wB", "timestamp": time.time()})
        lb = _recv(b)["lease_id"]
        assert la != lb
        res = lambda lease, wid: {"message_type": "result_inline", "job_id": "jX", "lease_id": lease,
                                  "worker_id": wid, "result": {"success": True, "payload": {"who": wid}}}
        _send(a, res(la, "wA"))                       # late, superseded lease
        _send(a, res(lb, "wA"))                       # right lease id, wrong connection
        assert tr.recv_result(timeout_s=0.7, job_id="jX") is None, "a superseded/foreign result was delivered"
        info = tr._inflight.get("jX")
        assert info is not None and info["lease_id"] == lb, "the active lease was retired by a late result"
        assert ("wA", "jX") in tr.stale_results, tr.stale_results
        _send(b, res(lb, "wB"))
        r = tr.recv_result(timeout_s=3, job_id="jX")
        assert r is not None and r["result"].get("who") == "wB", r
        a.close(); b.close()
    finally:
        tr.stop()
    return "superseded lease and wrong-connection results dropped and counted; the active lease's result delivered"


def d_late_result_across_phases(M=PWC, T=TCP):
    """Beta's sequence end to end, through the real coordinator and the real transport,
    with EQUAL forward and reverse thresholds: worker A stalls on a P1 chunk, the chunk
    times out and is retried on worker B, P1 completes, P2 is submitted, and only then
    does A send its late P1 result. P2 must report B's survivors, never A's."""
    tr, port = _transport(T, {"wA", "wB"})
    saved_timeout = M.JOB_TIMEOUT_S
    M.JOB_TIMEOUT_S = 2.0
    p2_jobs, p2_ready, late_sent, stop = [], threading.Event(), threading.Event(), threading.Event()
    real_submit = tr.submit_job

    def submit(job):
        real_submit(job)
        if job["prng_families"][0].endswith("_reverse"):
            p2_jobs.append(job["job_id"])
            if len(p2_jobs) == 2:
                p2_ready.set()
    tr.submit_job = submit

    def payload(job, seed):
        return {"success": True, "payload": _anchor_echo({
            "success": True, "seed_range": {"start": job["seed_start"], "end": job["seed_end"]},
            "min_match_threshold": job["min_match_threshold"],
            "survivors": [{"seed": seed, "match_rate": 0.9}]}, job)}

    # [PWC-STAGE2 R3, Beta X-PWC-STAGE2 R2 finding 1] Deterministic assignment: A
    # takes the FIRST job of the run (a P1 chunk) before B asks for anything; after
    # its late result A asks for nothing more, so B serves the P1 retry and every P2
    # chunk. The check then measures late-result handling, not scheduling.
    a_has_job = threading.Event()

    def worker(wid, role):
        s = _ready_worker(port, wid)
        s.settimeout(30)
        stalled = None
        try:
            if role == "B":
                a_has_job.wait(30)
            while not stop.is_set():
                if role == "A" and stalled is not None:
                    stop.wait(0.1)                    # A is done: it never takes a P2 job
                    continue
                _send(s, {"message_type": "request_job", "worker_id": wid, "timestamp": time.time()})
                m = _recv(s)
                if m.get("message_type") != "job_assign" or not m.get("payload"):
                    time.sleep(0.05)
                    continue
                job = m["payload"]
                if role == "A" and stalled is None:
                    stalled = (job, m["lease_id"])
                    assert not job["prng_families"][0].endswith("_reverse"), "A's first job is not a P1 job"
                    a_has_job.set()
                    p2_ready.wait(30)
                    _send(s, {"message_type": "result_inline", "job_id": job["job_id"],
                              "lease_id": m["lease_id"], "worker_id": wid, "result": payload(job, 111)})
                    late_sent.set()
                    continue
                if role == "B" and job["prng_families"][0].endswith("_reverse"):
                    late_sent.wait(30)
                    time.sleep(0.4)
                _send(s, {"message_type": "result_inline", "job_id": job["job_id"],
                          "lease_id": m["lease_id"], "worker_id": wid,
                          "result": payload(job, 222 if role == "B" else 333)})
        except (ConnectionError, OSError, ValueError):
            pass
    threads = [threading.Thread(target=worker, args=(w, r), daemon=True) for w, r in (("wA", "A"), ("wB", "B"))]
    try:
        for t in threads:
            t.start()
        assert _wait(lambda: tr.ready_count() == 2, 5)
        p = M.PersistentWorkerCoordinator(config_file=_write_config(), worker_pool_size=6,
                                          seed_cap_amd=25_000, pwc_transport="tcp",
                                          pwc_coordinator_address=DIAL)
        p._tcp_transport = tr
        p._effective_pool_by_host = {RIGS[0]: 6}
        r1 = run_pass(p, "java_lcg", thr=0.5)
        assert r1["pass_complete"], r1["pwc_pass_accounting"]
        assert 111 not in r1["survivors"], "P1 took A's stalled result after its retry"
        r2 = run_pass(p, "java_lcg_reverse", thr=0.5)
        assert late_sent.is_set(), "the late result was never sent (test did not exercise the path)"
        assert 111 not in r2["survivors"], f"P2 reported worker A's late P1 result: {r2['survivors']}"
        assert r2["pass_complete"] and r2["survivors"] and set(r2["survivors"]) == {222}, \
            f"P2 must be served by B alone: {r2['survivors']} {r2['pwc_pass_accounting']}"
    finally:
        stop.set()
        M.JOB_TIMEOUT_S = saved_timeout
        tr.stop()
    return ("A stalls on a P1 chunk, retry on B completes P1, P2 submitted, A's late P1 result "
            "arrives (thresholds equal 0.5/0.5): P2 reports B's survivors only")


# ═════════════════════════════════════════════════════════════════════════════
# [R2] Beta X-PWC-STAGE2 R1 Q7 — the refusal exit writes no GPU heartbeat
# ═════════════════════════════════════════════════════════════════════════════
_WORKER_CHILD_CUPY = _WORKER_CHILD.replace(
    "W.HEARTBEAT_DIR = hb\n",
    "W.HEARTBEAT_DIR = hb\nimport atexit\n"
    "atexit.register(lambda: print('CUPY_LOADED=%s' % ('cupy' in sys.modules), flush=True))\n")


def d_refusal_no_gpu(src_path=SVC_PATH):
    if importlib.util.find_spec("cupy") is None:
        raise Unavailable("cupy is not installed here, so a CuPy import could not be observed")
    tr, port = _transport(TCP, {"someone_else"})
    hb = os.path.join(OUT, "hb_q7")
    try:
        r = subprocess.run([sys.executable, "-c", _WORKER_CHILD_CUPY, ROOT, src_path, str(port), hb],
                           capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    finally:
        tr.stop()
    assert r.returncode == 43, f"rc={r.returncode} {r.stderr.strip()[-200:]!r}"
    assert "CUPY_LOADED=False" in r.stdout, f"the refusal exit imported CuPy: {r.stdout.strip()!r}"
    written = os.listdir(hb) if os.path.isdir(hb) else []
    assert not any(n.endswith(".json") for n in written), f"heartbeat written on refusal: {written}"
    return "refused worker exits 43 without importing CuPy or writing a heartbeat"


def _q6_old():
    src = open(TCP_PATH, encoding="utf-8").read()
    a = "                with self._workers_lock:\n                    owner = self._workers.get(worker_id) is conn\n"
    b = "                self._reclaim_worker_jobs(worker_id, conn)\n"
    i = src.index(a)
    return src[i:src.index(b, i) + len(b)]


_Q6_R1 = ("                with self._workers_lock:\n"
          "                    owner = self._workers.get(worker_id) is conn\n"
          "                    if owner:\n"
          "                        self._workers.pop(worker_id, None)\n"
          "                self._reclaim_worker_jobs(worker_id, None if owner else conn)\n"
          "                if owner:\n"
          "                    with self._last_seen_lock:\n"
          "                        self._last_seen.pop(worker_id, None)\n"
          "                    with self._state_lock:\n"
          "                        self._online_workers.discard(worker_id)\n"
          "                        self._ready_workers.discard(worker_id)\n")
_LEASE_CHECK = ("            info = self._inflight.get(job_id)\n"
                "            if info is None or info.get(\"lease_id\") != lease_id or info.get(\"conn\") is not conn:\n"
                "                return False\n"
                "            del self._inflight[job_id]\n"
                "            return True\n")
_LEASE_OLD = "            self._inflight.pop(job_id, None)\n            return True\n"
_JOBID_NEW = '"job_id":               f"sieve_{_pass_id}_{idx:03d}",   # [PWC-STAGE2 R2]'
_JOBID_OLD = '"job_id":               f"sieve_{idx:03d}",'


# ═════════════════════════════════════════════════════════════════════════════
# [R3] Beta X-PWC-STAGE2 R2 Q1 — same-id reconnect at the timeout boundary
# ═════════════════════════════════════════════════════════════════════════════
class _PausingLock:
    """Stands in for `_last_seen_lock`. When armed, the first acquisition by a
    transport handler thread (`pwc-tcp-worker-*`) pauses until released, which
    holds that handler exactly at its last-seen refresh."""
    def __init__(self, real):
        self.real, self.armed = real, False
        self.entered, self.gate = threading.Event(), threading.Event()

    def __enter__(self):
        if self.armed and threading.current_thread().name.startswith("pwc-tcp-worker-"):
            self.armed = False
            self.entered.set()
            self.gate.wait(10)
        return self.real.__enter__()

    def __exit__(self, *a):
        return self.real.__exit__(*a)


def d_reaper_same_id_reconnect(T=TCP):
    """Beta's interleaving: the old connection is still registered with a timestamp
    past the timeout; the SAME worker reconnects; the replacement is held at its
    last-seen refresh while the reaper runs. The reaper must not evict the
    replacement, which must then go ready, take a job and have its result delivered."""
    tr, port = _transport(T, {"rig_gpu0"})
    pl = _PausingLock(tr._last_seen_lock)
    new, reaper = None, None
    try:
        old = _ready_worker(port, "rig_gpu0")
        assert _wait(lambda: tr.ready_count() == 1)
        with tr._last_seen_lock:
            tr._last_seen["rig_gpu0"] = time.time() - (T.HEARTBEAT_TIMEOUT_S + 10)
        tr._last_seen_lock = pl
        pl.armed = True
        new, ack = _hello(port, "rig_gpu0")
        assert ack.get("accepted") is True, ack
        assert pl.entered.wait(5), "the replacement's handler never reached its last-seen refresh"
        reaper = threading.Thread(target=tr._reap_silent_workers, args=(time.time(),), daemon=True)
        reaper.start()
        reaper.join(0.4)                       # the fix: the reaper waits for the registry lock
        pl.gate.set()
        reaper.join(5)
        assert not reaper.is_alive(), "reaper did not finish"
        assert tr._workers.get("rig_gpu0") is not None, "the reaper evicted the fresh replacement"
        _send(new, {"message_type": "online", "worker_id": "rig_gpu0", "timestamp": time.time()})
        _send(new, {"message_type": "ready", "worker_id": "rig_gpu0", "timestamp": time.time()})
        assert _wait(lambda: tr.ready_count() == 1 and tr.worker_count() == 1), \
            f"replacement not ready after the reaper ran: ready={tr.ready_count()} workers={tr.worker_count()}"
        tr.submit_job({"job_id": "jT", "seed_start": 0, "seed_end": 10})
        _send(new, {"message_type": "request_job", "worker_id": "rig_gpu0", "timestamp": time.time()})
        new.settimeout(3)
        m = _recv(new)
        assert m.get("job_id") == "jT", m
        _send(new, {"message_type": "result_inline", "job_id": "jT", "lease_id": m["lease_id"],
                    "worker_id": "rig_gpu0", "result": {"success": True, "payload": {}}})
        r = tr.recv_result(timeout_s=3, job_id="jT")
        assert r is not None and r.get("status") == "ok", f"replacement's result not delivered: {r}"
        old.close()
    finally:
        pl.gate.set()
        if new is not None:
            new.close()
        tr.stop()
    return ("old connection 310 s silent, same id reconnects, held at its last-seen refresh while "
            "the reaper runs: replacement kept, goes ready, its job's result delivered")


def d_replaced_conn_no_refresh(T=TCP):
    """A replaced connection's message never refreshes the id's last-seen; it ends."""
    tr, port = _transport(T, {"rig_gpu0"})
    try:
        old = _ready_worker(port, "rig_gpu0")
        # the old connection must be fully up (its online/ready handled) BEFORE the
        # replacement registers, or the check would end it before its heartbeat
        assert _wait(lambda: tr.ready_count() == 1)
        old_conn = tr._workers.get("rig_gpu0")
        new = _ready_worker(port, "rig_gpu0")
        assert _wait(lambda: tr._workers.get("rig_gpu0") not in (None, old_conn) and tr.ready_count() == 1)
        time.sleep(0.2)
        with tr._last_seen_lock:
            tr._last_seen["rig_gpu0"] = 1000.0
        _send(old, {"message_type": "heartbeat", "worker_id": "rig_gpu0", "timestamp": time.time()})
        time.sleep(0.5)
        with tr._last_seen_lock:
            ts = tr._last_seen.get("rig_gpu0")
        assert ts == 1000.0, f"a replaced connection refreshed the id's last-seen: {ts}"
        assert tr.worker_count() == 1 and tr.ready_count() == 1, "the replaced connection's end hit the new one"
        old.settimeout(3)
        try:
            data = old.recv(1)
        except OSError:
            data = b""
        assert data == b"", "the replaced connection was not ended"
        new.close()
    finally:
        tr.stop()
    return "a replaced connection's heartbeat leaves last-seen untouched and ends that connection"


_R3_REG_NEW = ("            with self._workers_lock:\n"
               "                self._workers[worker_id] = conn\n"
               "                with self._last_seen_lock:\n"
               "                    self._last_seen[worker_id] = time.time()\n")
_R3_REG_OLD = ("            with self._workers_lock:\n"
               "                self._workers[worker_id] = conn\n"
               "            with self._last_seen_lock:\n"
               "                self._last_seen[worker_id] = time.time()\n")
_R3_MSG_NEW = ("                with self._workers_lock:\n"
               "                    if self._workers.get(worker_id) is not conn:\n"
               "                        break\n"
               "                    with self._last_seen_lock:\n"
               "                        self._last_seen[worker_id] = time.time()\n")
_R3_MSG_OLD = ("                with self._last_seen_lock:\n"
               "                    self._last_seen[worker_id] = time.time()\n")


# ═════════════════════════════════════════════════════════════════════════════
# Nothing outside PWC moved
# ═════════════════════════════════════════════════════════════════════════════
def d_import():
    code = ("import window_optimizer_integration_final as W; import miner; "
            "import persistent_worker_coordinator as P; "
            "assert W.run_trial_persistent is P.run_trial_persistent; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                       text=True, timeout=120, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-300:]
    return "integration module (Run 4's launch import) + miner package import cleanly, fresh process"


GATES = [
    ("G-THR-JOBS", d_thr_jobs),
    ("G-THR-REFUSALS", d_thr_refusals),
    ("G-THR-ECHO", d_thr_echo),
    ("G-THR-ECHO-REAL-SIEVE-FILTER", g_echo_real_sieve_filter),
    ("G-PHASES", d_phases),
    ("G-TRIAL-THRESHOLDS-STRATEGIES", d_trial_thresholds_and_strategies),
    ("G-STRATEGIES-ALL-FIVE", d_strategies_all_five),
    ("G-ANCHOR", d_anchor),
    ("G-SSH-VARIABLE", d_ssh_variable),
    ("G-PROBE-KILL", d_probe_kill),
    ("G-PROBE-S156", d_probe_s156),
    ("G-LOCAL-KILL-READ", d_local_kill_read),
    ("G-LAUNCH-COUNTED", d_launch_counted),
    ("G-REVOKE-DROPS-CONNECTION", d_revoke_drops_connection),
    ("G-REQUEST-JOB-ADMISSION", d_request_job_admission),
    ("G-OWNER-CLEANUP", d_owner_cleanup),
    ("G-WORKER-EXITS", d_worker_exits),
    ("G-R2-RECONNECT-RACE", d_reconnect_race),
    ("G-R2-REAPER-OWNERSHIP", d_reaper_ownership),
    ("G-R2-JOB-IDS-UNIQUE-PER-PASS", d_job_ids_unique_per_pass),
    ("G-R2-ACTIVE-LEASE-ONLY", d_active_lease_only),
    ("G-R2-LATE-RESULT-ACROSS-PHASES", d_late_result_across_phases),
    ("G-R2-REFUSAL-NO-GPU", d_refusal_no_gpu),
    ("G-R3-REAPER-SAME-ID-RECONNECT", d_reaper_same_id_reconnect),
    ("G-R3-REPLACED-CONN-NO-REFRESH", d_replaced_conn_no_refresh),
    ("G-IMPORT", d_import),
]

MUTANTS = [
    ("M-THR-1 hybrid phase2 default restored + equality off", "pwc", [
        ("                       phase2_threshold: Optional[float] = None,   # [PWC-STAGE2] no 0.5 default\n",
         "                       phase2_threshold: Optional[float] = 0.5,\n"),
        ("            if not _thr_equal(phase2_threshold, threshold):\n", "            if False:\n")],
     d_thr_refusals),
    ("M-THR-2 SSH hybrid allowed", "pwc", [
        ("            if self._tcp_transport is None:\n                raise PwcSettingRefused(\n"
         "                    f\"[{prng_type}] variable-skip passes need the TCP transport",
         "            if False:\n                raise PwcSettingRefused(\n"
         "                    f\"[{prng_type}] variable-skip passes need the TCP transport")],
     d_thr_refusals),
    ("M-THR-3 hybrid report not checked", "pwc", [
        ("            assert_pwc_hybrid_threshold_contract(_thr_rec)\n", "")], d_thr_echo),
    ("M-THR-4 missing report tolerated", "pwc", [
        ("        \"agree\": not disagree and int(effective_unreported) == 0,\n",
         "        \"agree\": not disagree,\n")], d_thr_echo),
    ("M-PH-1 forward-zero prune restored", "pwc", [
        ("        # [PWC-STAGE2 ruling 4] The S147 B1 forward-zero prune that stood here is\n",
         "        if not fwd_survivors:\n"
         "            return build_trial_populations(forward_map_constant={}, reverse_map_constant={},"
         " forward_map_variable={}, reverse_map_variable={}, bidirectional_constant=set(),"
         " bidirectional_variable=set(), pruned=True, reason='forward_zero',"
         " extra={'bidirectional_count': 0, **_completion_extra()}, origin='mutant')\n"
         "        # [PWC-STAGE2 ruling 4] The S147 B1 forward-zero prune that stood here is\n")],
     d_phases),
    ("M-PH-2 phase 3 threshold dropped", "pwc", [
        ("                strategies   = None,           # [PWC-STAGE2] all five\n"
         "                phase2_threshold = forward_threshold,   # [PWC-STAGE2 ruling 1]\n",
         "                strategies   = None,           # [PWC-STAGE2] all five\n")],
     d_trial_thresholds_and_strategies),
    ("M-ST-1 one strategy only", "pwc", [
        ("                raw = get_all_strategies()\n", "                raw = get_all_strategies()[:1]\n")],
     d_strategies_all_five),
    ("M-OF-1 trial anchor check removed", "pwc", [
        ("    validate_pwc_window_anchor(config.offset, ssh_route=(pwc_transport != \"tcp\"))\n",
         "")], d_anchor),
    ("M-OF-2 pass anchor check treats SSH as TCP", "pwc", [
        ("            offset, ssh_route=self._tcp_transport is None)\n",
         "            offset, ssh_route=False)\n")], d_anchor),
    ("M-AN-1 TCP job keeps the old offset (no split)", "pwc", [
        ("                    return self._dispatch_to_tcp(pwc_new_key_job(job, _window_anchor))  # [STAGE 2b]\n",
         "                    return self._dispatch_to_tcp(job)\n")], d_anchor),
    ("M-AN-2 worker echo not checked", "pwc", [
        ("    if \"window_anchor\" not in job:\n        return None\n    if not isinstance(payload, dict):\n",
         "    return None\n    if not isinstance(payload, dict):\n")], d_anchor),
    ("M-AN-3 generator phase follows the anchor", "pwc", [
        ("            out[\"generator_phase\"] = PWC_GENERATOR_PHASE_V1_PIN\n",
         "            out[\"generator_phase\"] = int(window_anchor)\n")], d_anchor),
    ("M-AN-4 negative anchor accepted", "pwc", [
        ("    if int(anchor) < 0:\n", "    if False:\n")], d_anchor),
    ("M-SSH-1 both-mode SSH not refused up front", "pwc", [
        ("    if (_run_variable or prng_base.endswith(\"_hybrid\")) and pwc_transport != \"tcp\":\n",
         "    if False:\n")], d_ssh_variable),
    ("M-PR-1 kill probe control removed", "pwc", [
        ("    return (\"pgrep -f '%s' >/dev/null 2>&1 || exit %d; \"\n"
         "            % (PWC_PROBE_CONTROL_TOKEN, PWC_KILL_EXIT_UNAVAILABLE))\n",
         "    return \"\"\n")], d_probe_kill),
    ("M-PR-2 kill: any non-match status counts as clean", "pwc", [
        ("            \"if [ $r -eq 1 ]; then echo %s; elif [ $r -eq 0 ]; then exit %d; \"\n",
         "            \"if [ $r -ne 0 ]; then echo %s; elif [ $r -eq 0 ]; then exit %d; \"\n")],
     d_probe_kill),
    ("M-PR-3 S156 reap: double-zero construct restored", "pwc", [
        ("            \"remaining=$(pgrep -c -f \\\"%s\\\" 2>/dev/null); r=$?; \"\n"
         "            \"if [ $r -le 1 ]; then echo \\\"reaped:$remaining\\\"; else echo reaped:unavailable; fi\"\n",
         "            \"remaining=$(pgrep -c -f \\\"%s\\\" 2>/dev/null || echo 0); \"\n"
         "            \"echo \\\"reaped:$remaining\\\"\"\n")], d_probe_s156),
    ("M-PR-4 S156 find: probe control removed", "pwc", [
        ("    return (\"pgrep -f '%s' >/dev/null 2>&1 || { echo %s; exit 0; }; \"\n"
         "            \"out=$(pgrep -af",
         "    return (\"true '%s' %s; \"\n"
         "            \"out=$(pgrep -af")], d_probe_s156),
    ("M-LK-1 VM101 kill result ignored", "pwc", [
        ("            if not _zeus_kill_ok:                # [PWC-STAGE2 BETA-2]\n                continue\n", "")],
     d_local_kill_read),
    ("M-LC-1 unconfirmed launch counted", "pwc", [
        ("                            \"id revoked, not counted\")\n                        continue\n",
         "                            \"id revoked, not counted\")\n")], d_launch_counted),
    ("M-RV-1 revoke leaves the connection", "tcp", [
        ("            try:\n                conn.sock.shutdown(socket.SHUT_RDWR)\n            except Exception:\n                pass\n",
         "            pass\n")], d_revoke_drops_connection),
    ("M-RV-2 revoke only edits the list (stage 1 behaviour)", "tcp", [
        ("        with self._state_lock:\n            self._online_workers.discard(wid)\n"
         "            self._ready_workers.discard(wid)\n        with self._workers_lock:\n"
         "            conn = self._workers.get(wid)\n        if conn is not None:\n",
         "        with self._workers_lock:\n            conn = self._workers.get(wid)\n"
         "        if False:\n")], d_revoke_drops_connection),
    ("M-RQ-1 request_job admission removed", "tcp", [
        ("                    # [PWC-STAGE2 BETA-3] an id revoked after its hello gets no job\n"
         "                    if not self._admits(worker_id):\n                        break\n", "")],
     d_request_job_admission),
    ("M-OW-1 owner check removed", "tcp", [
        ("                    owner = self._workers.get(worker_id) is conn\n",
         "                    owner = True\n")], d_owner_cleanup),
    ("M-R2-Q6a R1 cleanup restored (state cleared after the lock, reclaim by id)", "tcp", [
        (_q6_old(), _Q6_R1)], d_reconnect_race),
    ("M-R2-Q6b reaper leaves ready set", "tcp", [
        ("                conn = self._workers.pop(worker_id, None)\n"
         "                with self._state_lock:\n"
         "                    self._online_workers.discard(worker_id)\n"
         "                    self._ready_workers.discard(worker_id)\n"
         "            if conn is not None:\n",
         "                conn = self._workers.pop(worker_id, None)\n"
         "            if conn is not None:\n")], d_reaper_ownership),
    ("M-R2-F2a job ids restarted at sieve_000 each pass", "pwc", [(_JOBID_NEW, _JOBID_OLD)],
     d_job_ids_unique_per_pass),
    ("M-R2-F2b any result retires the lease", "tcp", [(_LEASE_CHECK, _LEASE_OLD)], d_active_lease_only),
    ("M-R2-F2c both removed (the pre-R2 path Beta reproduced)", "pwc+tcp", None,
     d_late_result_across_phases),
    ("M-R2-Q7 heartbeat written on refusal", "svc", [
        ("            self._close()\n            sys.exit(PWC_EXIT_HELLO_REJECTED)\n",
         "            self._emit_heartbeat(\"shutdown\")\n            self._close()\n"
         "            sys.exit(PWC_EXIT_HELLO_REJECTED)\n")], d_refusal_no_gpu),
    ("M-R3-Q1a register and last-seen refresh in separate sections (R2)", "tcp",
     [(_R3_REG_NEW, _R3_REG_OLD)], d_reaper_same_id_reconnect),
    ("M-R3-Q1b any connection refreshes last-seen", "tcp",
     [(_R3_MSG_NEW, _R3_MSG_OLD)], d_replaced_conn_no_refresh),
    ("M-WK-1 worker retries after refusal", "svc", [
        ("            sys.exit(PWC_EXIT_HELLO_REJECTED)\n",
         "            raise ConnectionError('rejected')\n")], d_worker_exits),
]


def _mutant_pair():
    """[R2] both late-result guards removed at once: a mutated coordinator module and a
    mutated transport module. The detector receives both."""
    psrc = _mutate(PWC_PATH, _JOBID_NEW, _JOBID_OLD)
    tsrc = _mutate(TCP_PATH, _LEASE_CHECK, _LEASE_OLD)
    return (_load_source(psrc, "stage2_mutant_pwc_pair", PWC_PATH),
            _load_source(tsrc, "stage2_mutant_tcp_pair", TCP_PATH))


def run_mutant(label, kind, edits, detector):
    if kind == "pwc+tcp":
        detector()                                                  # positive control
        m, t = _mutant_pair()
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                detector(m, t)
        except AssertionError as exc:
            return f"killed: {str(exc)[:110]}"
        except Exception as exc:                                    # noqa: BLE001
            raise AssertionError(f"UNATTRIBUTABLE kill: {type(exc).__name__}: {exc}")
        raise AssertionError(f"SURVIVED: {label}")
    path = {"pwc": PWC_PATH, "tcp": TCP_PATH, "svc": SVC_PATH}[kind]
    src = open(path, encoding="utf-8").read()
    for old, new in edits:
        n = src.count(old)
        assert n == 1, f"mutant anchor occurs {n} times in {os.path.basename(path)}: {old[:80]!r}"
        src = src.replace(old, new)
    detector()                                                      # positive control
    if kind == "svc":
        mpath = os.path.join(OUT, "mutant_pwc_worker_service.py")
        with open(mpath, "w") as f:
            f.write(src)
        target = mpath
    else:
        target = _load_source(src, f"stage2_mutant_{kind}", path)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            detector(target)
    except AssertionError as exc:
        return f"killed: {str(exc)[:110]}"
    except Exception as exc:                                        # noqa: BLE001
        raise AssertionError(f"UNATTRIBUTABLE kill: {type(exc).__name__}: {exc}")
    raise AssertionError(f"SURVIVED: {label}")


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    import logging
    logging.getLogger("PersistentWorkerCoordinator").handlers[:] = [
        logging.StreamHandler(io.StringIO())]
    logging.getLogger("PersistentWorkerCoordinator").propagate = False
    print("VERIFICATION test_pwc_backup_stage2 START", flush=True)
    passed = unavailable = 0
    for name, fn in GATES:
        try:
            print(f"[PASS] {name}: {fn()}", flush=True)
            passed += 1
        except Unavailable as exc:
            unavailable += 1
            print(f"[UNAVAILABLE] {name}: {exc}", flush=True)
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    killed = 0
    for label, kind, edits, det in MUTANTS:
        try:
            print(f"[PASS] {label} -> {run_mutant(label, kind, edits, det)}", flush=True)
            killed += 1
        except Unavailable as exc:
            print(f"[UNAVAILABLE] {label}: {exc}", flush=True)
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}", flush=True)
    print("")
    print(f"GATES {passed}/{len(GATES)} PASS ({unavailable} UNAVAILABLE)")
    print(f"MUTANTS {killed}/{len(MUTANTS)} killed")
    ok = passed == len(GATES) and killed == len(MUTANTS)
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    print("VERIFICATION test_pwc_backup_stage2 COMPLETE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
