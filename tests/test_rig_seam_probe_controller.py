#!/usr/bin/env python3
"""Off-GPU gates and mutants for scripts/rig_seam_probe.py + scripts/rig_seam_controller.py.

Thread X-RIG-GPU-PROBE (R2). NO GPU, NO cupy, NO torch:
  * controller gates run fake-backend probes on localhost (stub cupy pools, stub torch);
  * real-backend gates run the REAL backend code against a STUB repository (stub `cupy`,
    stub `miner.range_miner_worker`, stub `sieve_gpu_worker`) placed first on sys.path, so the
    BDF guard, the pin check and the cleanup replacement are exercised without any GPU library.

Each gate runs the real controller in a FRESH interpreter. Mutants follow the repository
convention (tests/test_s172_window_anchor_brief_i_mutants.py): applied to a temporary COPY of
the scripts, the credited gate runs on the copy in a fresh interpreter and is classed DETECTED /
SURVIVED / INVALID, each with a CLEAN CONTROL run of the same gate on unmutated source.

Run:  python3 -B -m pytest -q -p no:cacheprovider tests/test_rig_seam_probe_controller.py
  or: python3 -B tests/test_rig_seam_probe_controller.py gate <NAME> --root <dir> [--work <dir>]
      python3 -B tests/test_rig_seam_probe_controller.py mutants
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BDF1, BDF2 = "0000:16:00.0", "0000:06:00.0"
HOST_IP, HOST_IP2 = "10.0.0.1", "10.0.0.2"
LOCAL_ALIVE = ("if [ -d /proc/{pid} ] && grep -q rig_seam_probe /proc/{pid}/cmdline 2>/dev/null; "
               "then echo ALIVE; else echo GONE; fi")


# ------------------------------------------------------------------------------------------
# harness
# ------------------------------------------------------------------------------------------
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _stamp(t=None):
    t = time.time() if t is None else t
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1e6):06d}"


def _fake_plan(path):
    plan = {"v": 1, "family3": "java_lcg_hybrid", "family4": "java_lcg_hybrid_reverse",
            "assign3": {"family_name": "java_lcg_hybrid", "phase": 3},
            "assign4": {"family_name": "java_lcg_hybrid_reverse", "phase": 4},
            "ranges3": [[0, 10], [10, 10], [20, 5]], "ranges4": [[0, 10], [10, 10], [20, 5]]}
    Path(path).write_text(json.dumps(plan))


class Recorder(threading.Thread):
    """Writes recorder-format OBSERVED rows for `hosts` every 0.3 s, starting IMMEDIATELY.
    unobserved_window=(t0,t1): UNOBSERVED rows inside that window (s from start).
    die_after: stop writing after this many seconds (recorder death)."""
    def __init__(self, path, hosts=(HOST_IP,), unobserved_window=None, die_after=None):
        super().__init__(daemon=True)
        self.path, self.hosts, self.win, self.die = path, hosts, unobserved_window, die_after
        self.stop = threading.Event()
        self._write(0.0)

    def _write(self, rel):
        st = "UNOBSERVED" if self.win and self.win[0] <= rel <= self.win[1] else "OBSERVED"
        with open(self.path, "a") as f:
            for h in self.hosts:
                f.write(json.dumps({"kind": "sample", "status": st, "host_ip": h,
                                    "recv_ts": time.time()}) + "\n")

    def run(self):
        t0 = time.time()
        while not self.stop.wait(0.3):
            rel = time.time() - t0
            if self.die is not None and rel > self.die:
                return
            self._write(rel)


def _controller_cmd(root, work, arm, tf, series, nc, rec, port, extra):
    launch = (f"{sys.executable} -B {root}/scripts/rig_seam_probe.py --controller {{ctl}} "
              f"--bdf {{bdf}} --torch-mode {{arm}} --attempt-id {{aid}} --backend fake "
              f"--local-trace {work}/probe_{{aid}}_{{rocr}}.jsonl "
              f"> {work}/probe_{{aid}}_{{rocr}}.out 2>&1 &")
    return [sys.executable, "-B", f"{root}/scripts/rig_seam_controller.py", "run",
            "--targets", str(tf), "--arm", arm, "--series", str(series),
            "--netconsole", str(nc), "--recorder", str(rec), "--bind", "127.0.0.1",
            "--advertise", "127.0.0.1", "--port", str(port), "--launch-template", launch,
            "--kill-template", "kill -{sig} {pid}", "--alive-template", LOCAL_ALIVE,
            "--launch-stagger-s", "0", "--hello-timeout-s", "20", "--step-timeout-s", "20",
            "--idle-s", "0.5", "--window-after-s", "1", "--deadline-s", "25",
            "--silence-s", "3", "--kill-wait-s", "2", "--post-proof-wait-s", "4",
            "--pre-proof-max-age-s", "600", "--rec-stale-s", "2", "--remote-timeout-s", "5", *extra]


def setup(work, targets=None, pre_proof=True, proof_hosts=(HOST_IP,)):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    nc, rec = work / "netconsole.log", work / "recorder.jsonl"
    nc.touch()
    rec.touch()
    if pre_proof:
        with open(nc, "a") as f:
            for h in proof_hosts:
                f.write(f"{_stamp()} {h}: [ 1.0] NCPROOF-pre\n")
    plan = work / "plan.json"
    _fake_plan(plan)
    targets = targets or [{"ct": "local", "ct_hostname": socket.gethostname(), "rocr": 0,
                           "bdf": BDF1, "host_ip": HOST_IP}]
    for t in targets:
        t.setdefault("plan", str(plan))
    tf = work / "targets.json"
    tf.write_text(json.dumps(targets))
    return work, nc, rec, tf


class ProofTicker(threading.Thread):
    """Appends an NCPROOF line for HOST_IP every 0.5 s (stands in for the owner's post-attempt
    proof when the probe is the REAL backend, which has no test hooks)."""
    def __init__(self, nc):
        super().__init__(daemon=True)
        self.nc, self.stop = nc, threading.Event()

    def run(self):
        while not self.stop.wait(0.5):
            with open(self.nc, "a") as f:
                f.write(f"{_stamp()} {HOST_IP}: [ 1.0] NCPROOF-ticker\n")


def run_controller(root, work, arm="P", targets=None, env=None, extra=(), pre_proof=True,
                   recorder=True, rec_kw=None, series=None, proof_hosts=(HOST_IP,), background=False,
                   proof_ticker=False):
    work, nc, rec, tf = setup(work, targets, pre_proof, proof_hosts)
    tick = ProofTicker(nc) if proof_ticker else None
    if tick:
        tick.start()
    penv = dict(os.environ, RSP_FAKE_NC_FILE=str(nc), RSP_FAKE_NC_HOST=HOST_IP,
                RSP_FAKE_NC_POSTPROOF="1", RSP_FAKE_SUBSTRIPE_S="0.02",
                # belt and braces: no device is visible to any test process
                CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    penv.update(env or {})
    r = Recorder(rec, **(rec_kw or {})) if recorder else None
    if r:
        r.start()
    series = Path(series or (work / "series"))
    cmd = _controller_cmd(root, work, arm, tf, series, nc, rec, _free_port(), extra)
    if background:
        return subprocess.Popen(cmd, env=penv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True), series, r, work
    p = subprocess.run(cmd, env=penv, capture_output=True, text=True, timeout=180)
    if r:
        r.stop.set()
    if tick:
        tick.stop.set()
    return p, _rows(series), _events(series)


def _rows(series):
    f = Path(series) / "attempts.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def _final(rows):
    fin = [r for r in rows if r.get("status") == "final" or "class" in r]
    return fin[-1] if fin else None


def _events(series):
    rows = _rows(series)
    fin = _final(rows)
    if not fin:
        return {}
    f = Path(series) / f"{fin['attempt_id']}_events.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _ops(ev, key=None):
    out = []
    for k, recs in ev.get("trace", {}).items():
        if key and k != key:
            continue
        out += [(r.get("kind"), r.get("op"), r.get("msg")) for r in recs]
    return out


# ------------------------------------------------------------------------------------------
# controller gates
# ------------------------------------------------------------------------------------------
def gate_clean_P(root, work):
    p, rows, ev = run_controller(root, work, "P")
    a = _final(rows)
    assert a, f"G-CLEAN-P: no attempt recorded: {p.stdout} {p.stderr}"
    assert a["class"] == "C", f"G-CLEAN-P: expected C, got {a['class']} trig={a['trigger']} unres={a['unresolved']}"
    assert not a["needs_owner_recovery"], "G-CLEAN-P: a clean attempt must not hold the barrier"
    msgs = [m for _, _, m in _ops(ev)]
    for want in ("stub torch.cuda.is_available", "stub torch.cuda.synchronize", "stub torch.cuda.empty_cache"):
        assert want in msgs, f"G-CLEAN-P: arm P must run production torch cleanup; missing {want}"


def gate_arms_MN(root, work):
    _, rM, evM = run_controller(root, Path(work) / "M", "M")
    _, rN, evN = run_controller(root, Path(work) / "N", "N")
    assert _final(rM)["class"] == "C" and _final(rN)["class"] == "C", "G-ARMS: M/N should be clean"
    opsM, opsN = _ops(evM), _ops(evN)
    assert ("entry", "import_torch", None) in opsM, "G-ARMS: M must import torch"
    assert not any(op == "torch_is_available" for _, op, _ in opsM), "G-ARMS: M must never call torch"
    assert not any(m == "stub torch imported" for _, _, m in opsN), "G-ARMS: N imported torch"
    guards = [r for recs in evN["trace"].values() for r in recs if r.get("kind") == "torch_guard"]
    assert guards and guards[0]["torch_blocked"] is True, "G-ARMS: arm N ran without the torch import blocker"


def gate_arm_P1(root, work):
    _, rows, ev = run_controller(root, work, "P1")
    assert _final(rows)["class"] == "C", f"G-P1: expected C, got {_final(rows)['class']}"
    msgs = [m for _, _, m in _ops(ev)]
    n_sync = msgs.count("stub torch.cuda.synchronize")
    n_phase = sum(1 for k, _, _ in _ops(ev) if k == "phase_end")
    assert n_phase == 2 and n_sync == n_phase, f"G-P1: torch cleanup must run once per phase ({n_phase}), ran {n_sync}"


def gate_fault_withhold(root, work):
    targets = [{"ct": "local", "ct_hostname": socket.gethostname(), "rocr": 0, "bdf": BDF1,
                "host_ip": HOST_IP, "seam_offset_s": 0.0},
               {"ct": "local", "ct_hostname": socket.gethostname(), "rocr": 1, "bdf": BDF2,
                "host_ip": HOST_IP, "seam_offset_s": 4.0}]
    p, rows, ev = run_controller(root, work, "P", targets=targets, env={"RSP_FAKE_NC_E1_AT": "seam:launch"})
    a = _final(rows)
    assert a["class"] == "F", f"G-FAULT: expected F, got {a['class']} trig={a['trigger']}"
    assert a["trigger"]["why"] == "E1", f"G-FAULT: trigger should be E1, got {a['trigger']}"
    assert [w for w in a["withheld"] if w.get("cmd") == "seam"], "G-FAULT: later seam slot must be WITHHELD"
    assert a["needs_owner_recovery"], "G-FAULT: a fault must hold the barrier"
    assert not a["unresolved"], f"G-FAULT: probes not confirmed gone: {a['unresolved']}"
    p2, _, _ = run_controller(root, Path(work) / "again", "P", series=Path(work) / "series")
    assert p2.returncode == 2 and "REFUSED" in p2.stdout, f"G-FAULT: next attempt not refused: {p2.stdout}"


def gate_warmup_fault(root, work):
    _, rows, _ = run_controller(root, work, "P", env={"RSP_FAKE_NC_E1_AT": "warm:launch"})
    a = _final(rows)
    assert a["class"] == "FW", f"G-WARM: E1 during warm-up must be FW, got {a['class']}"
    assert not a["seam_sent_at"], "G-WARM: seam must not be sent after a warm-up fault"


def gate_hang_deadline(root, work):
    _, rows, _ = run_controller(root, work, "P", env={"RSP_FAKE_HANG_AT": "seam:launch"},
                                extra=("--deadline-s", "10", "--step-timeout-s", "60"))
    a = _final(rows)
    assert a["trigger"] and a["trigger"]["why"] == "deadline", f"G-HANG: expected deadline, got {a['trigger']}"
    assert a["class"] == "I", f"G-HANG: a hung attempt without E1 is I, got {a['class']}"
    assert not a["unresolved"], f"G-HANG: escalation did not confirm the hung probe gone: {a['unresolved']}"


def gate_trace_gap(root, work):
    _, rows, _ = run_controller(root, work, "P", env={"RSP_FAKE_DROP_SEQ": "7"})
    a = _final(rows)
    assert a["class"] == "I" and not a["trace_complete"], f"G-GAP: lost record must make I, got {a['class']}"


def gate_recorder_unobserved(root, work):
    _, rows, _ = run_controller(root, work, "P", rec_kw={"unobserved_window": (1.0, 2.0)})
    assert _final(rows)["class"] == "I", f"G-REC: recorder UNOBSERVED must make I, got {_final(rows)['class']}"


def gate_recorder_death(root, work):
    _, rows, _ = run_controller(root, work, "P", rec_kw={"die_after": 0.5}, extra=("--idle-s", "4"))
    a = _final(rows)
    assert a["class"] == "I", f"G-RECDEATH: recorder death must make I, got {a['class']}"
    assert a["trigger"] and a["trigger"]["why"] == "recorder_stale", f"G-RECDEATH: expected recorder_stale, got {a['trigger']}"
    assert a["needs_owner_recovery"], "G-RECDEATH: recorder loss must hold the barrier"


def gate_recorder_absent(root, work):
    p, rows, _ = run_controller(root, work, "P", recorder=False)
    assert p.returncode == 2 and "REFUSED" in p.stdout, f"G-RECABSENT: must refuse before dispatch: {p.stdout}"
    assert not rows, "G-RECABSENT: no attempt may be started without a live recorder"
    assert not list(Path(work).glob("probe_*.out")), "G-RECABSENT: a probe was launched"


def gate_recorder_per_host(root, work):
    targets = [{"ct": "local", "ct_hostname": socket.gethostname(), "rocr": 0, "bdf": BDF1, "host_ip": HOST_IP},
               {"ct": "local", "ct_hostname": socket.gethostname(), "rocr": 1, "bdf": BDF2, "host_ip": HOST_IP2}]
    p, rows, _ = run_controller(root, work, "P", targets=targets, proof_hosts=(HOST_IP, HOST_IP2),
                                rec_kw={"hosts": (HOST_IP,)})
    assert p.returncode == 2 and "REFUSED" in p.stdout and not rows, \
        f"G-RECHOST: a host without recorder coverage must refuse dispatch: {p.stdout}"


def gate_no_pre_proof(root, work):
    p, rows, _ = run_controller(root, work, "P", pre_proof=False)
    assert p.returncode == 2 and "REFUSED" in p.stdout and not rows, \
        f"G-PROOF: no pre-proof must refuse before dispatch: {p.stdout}"


def gate_shutdown_unreachable(root, work):
    _, rows, _ = run_controller(root, work, "P", extra=("--alive-template", "true"))
    a = _final(rows)
    assert a["unresolved"], "G-UNREACH: a transport failure must leave the target UNRESOLVED"
    assert a["class"] != "C" and a["needs_owner_recovery"], \
        f"G-UNREACH: unresolved cleanup must never be clean: {a['class']}"


def gate_shutdown_timeout(root, work):
    _, rows, _ = run_controller(root, work, "P", extra=("--alive-template", "sleep 4; echo GONE",
                                                        "--remote-timeout-s", "1"))
    a = _final(rows)
    assert a, "G-TIMEOUT: an alive-check timeout must not lose the attempt record"
    assert a["unresolved"] and "timeout" in json.dumps(a["unresolved"]), \
        f"G-TIMEOUT: timeout must be reported unresolved: {a['unresolved']}"
    assert a["needs_owner_recovery"], "G-TIMEOUT: must hold the barrier"


def gate_interrupt_barrier(root, work):
    proc, series, r, w = run_controller(root, work, "P", background=True, extra=("--idle-s", "6"))
    deadline = time.time() + 20
    while time.time() < deadline and not _rows(series):
        time.sleep(0.2)
    time.sleep(2.0)
    proc.send_signal(signal.SIGKILL)          # harshest case: no finally runs at all
    proc.wait(timeout=10)
    if r:
        r.stop.set()
    rows = _rows(series)
    assert rows and rows[0].get("status") == "in_progress", "G-INTERRUPT: no durable in-progress record"
    p2, _, _ = run_controller(root, Path(work) / "again", "P", series=series)
    assert p2.returncode == 2 and "REFUSED" in p2.stdout, \
        f"G-INTERRUPT: restart after an interrupted attempt must be refused: {p2.stdout}"


def gate_clear(root, work):
    _, rows, _ = run_controller(root, work, "P", env={"RSP_FAKE_NC_E1_AT": "seam:launch"})
    a = _final(rows)
    assert a["needs_owner_recovery"], "G-CLEAR: setup needs a barrier"
    series = Path(work) / "series"
    nc, tf = Path(work) / "netconsole.log", Path(work) / "targets.json"
    base = [sys.executable, "-B", f"{root}/scripts/rig_seam_controller.py", "clear",
            "--series", str(series), "--attempt-id", a["attempt_id"], "--targets", str(tf),
            "--netconsole", str(nc), "--alive-template", LOCAL_ALIVE,
            "--host-check-template", "echo 01234567-89ab-cdef-0123-456789abcdef"]
    bad = subprocess.run(base + ["--map-template", "printf '0\\n1536\\n'"], capture_output=True, text=True)
    assert bad.returncode == 2 and "REFUSED" in bad.stdout, f"G-CLEAR: wrong card map must refuse: {bad.stdout}"
    with open(nc, "a") as f:
        f.write(f"{_stamp()} {HOST_IP}: [ 1.0] NCPROOF-after-recovery\n")
    good = subprocess.run(base + ["--map-template", "printf '0\\n5632\\n'"], capture_output=True, text=True)
    assert good.returncode == 0, f"G-CLEAR: checks pass but clear refused: {good.stdout} {good.stderr}"
    p3, _, _ = run_controller(root, Path(work) / "next", "P", series=series)
    assert "REFUSED: recovery barrier" not in p3.stdout, f"G-CLEAR: barrier not lifted: {p3.stdout}"


def gate_no_first_heartbeat(root, work):
    _, rows, _ = run_controller(root, work, "P", env={"RSP_FAKE_NO_UDP": "1"},
                                extra=("--silence-s", "2", "--deadline-s", "60", "--step-timeout-s", "60"))
    a = _final(rows)
    assert a["trigger"] and a["trigger"]["why"] == "probe_silence", \
        f"G-NOHB: a probe that never sends a datagram must trip silence, got {a['trigger']}"


def gate_resume_mode(root, work):
    _, rows, ev = run_controller(root, work, "P", extra=("--seam-mode", "resume"))
    fams = [r.get("family") for recs in ev["trace"].values() for r in recs if r.get("op") == "module_load"]
    assert _final(rows)["class"] == "C", f"G-RESUME: expected C, got {_final(rows)['class']}"
    assert "java_lcg_hybrid_reverse" not in fams, "G-RESUME: resume must not load the reverse module"


def gate_stats(root, work):
    sys.path.insert(0, str(Path(root) / "scripts"))
    import importlib
    ctl = importlib.import_module("rig_seam_controller")
    assert abs(ctl.upper_bound_one_sided(0, 8) - 0.3123) < 1e-3, "G-STATS: 0/8 bound"
    assert abs(ctl.upper_bound_one_sided(0, 4) - 0.5271) < 1e-3, "G-STATS: 0/4 bound"
    assert abs(ctl.fisher_one_sided(4, 8, 0, 8) - 0.0385) < 1e-3, "G-STATS: Fisher 4/8 vs 0/8"
    Path(work).mkdir(parents=True, exist_ok=True)
    f = Path(work) / "att.jsonl"
    rows = [{"attempt_id": f"p{i}", "status": "final", "arm": "P", "class": c}
            for i, c in enumerate(("F", "C", "I", "I"))] + \
           [{"attempt_id": f"n{i}", "status": "final", "arm": "N", "class": c}
            for i, c in enumerate(("C", "C", "C", "I"))]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    s = ctl.summarize(f)["arms"]
    assert s["P"]["e1_range"] == [0.25, 0.75], f"G-STATS: P range {s['P']['e1_range']}"
    assert s["P"]["observed_n"] == 2, "G-STATS: bound must be over F+C only"
    assert abs(s["P"]["upper95_over_observed"] - ctl.upper_bound_one_sided(1, 2)) < 1e-9, "G-STATS: bound over B"


def gate_fw_range(root, work):
    sys.path.insert(0, str(Path(root) / "scripts"))
    import importlib
    ctl = importlib.import_module("rig_seam_controller")
    Path(work).mkdir(parents=True, exist_ok=True)
    f = Path(work) / "att.jsonl"
    f.write_text(json.dumps({"attempt_id": "a1", "status": "final", "arm": "P", "class": "FW"}))
    s = ctl.summarize(f)["arms"]["P"]
    assert s["e1_range"] == [0.0, 1.0] and s["reached_seam"] == 0, \
        f"G-FW: an FW attempt's seam outcome is unobserved, not negative: {s}"


def gate_cleanup_pin(root, work):
    src = (REPO / "miner" / "range_miner_worker.py").read_text()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_best_effort_gpu_cleanup"][0]
    seg = "".join(src.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])
    got = hashlib.sha256(seg.encode()).hexdigest()
    want = (Path(root) / "scripts" / "rig_seam_probe.cleanup.sha256").read_text().split()[0]
    assert got == want, f"G-PIN: production cleanup changed ({got} != {want}); arm P would be stale"


# ------------------------------------------------------------------------------------------
# real-backend gates (stub repository; no GPU library is importable from it)
# ------------------------------------------------------------------------------------------
STUB_CUPY = '''
import os
class _Pool:
    def free_all_blocks(self): return None
def get_default_memory_pool(): return _Pool()
def get_default_pinned_memory_pool(): return _Pool()
class _RT:
    is_hip = True
    @staticmethod
    def getDeviceProperties(i):
        b = os.environ.get("STUB_BDF", "0000:16:00.0")
        dom, bus, rest = b.split(":"); dev = rest.split(".")[0]
        return {"name": b"stub", "pciDomainID": int(dom, 16), "pciBusID": int(bus, 16), "pciDeviceID": int(dev, 16)}
class _Cuda:
    runtime = _RT()
cuda = _Cuda()
'''
STUB_RMW = '''
import sieve_gpu_worker as sgw
class GpuInfo:
    gpu_name = "stub"
def detect_gpu(i): return GpuInfo()
def warm_gpu(i): return None
def _best_effort_gpu_cleanup():
    return None
class ResidueResolver:
    pass
class _Out:
    count = 0
class SieveExecutor:
    def __init__(self, resolver=None, device_index=0): pass
    def _gpu_launch(self, kernel, blocks, threads, args): return None
    def execute(self, assign, s, c):
        k = sgw._get_kernel(assign.get("family_name") if isinstance(assign, dict) else "x")
        self._gpu_launch(k, 1, 1, (1, 2))
        import miner.range_miner_worker as me
        me._best_effort_gpu_cleanup()      # production calls the MODULE GLOBAL (:1331)
        return _Out()
'''
STUB_SGW = '''
def _get_kernel(family, *a, **k): return ("kernel", family)
'''
STUB_PROTO = '''
def from_dict(d): return d
'''


def _stub_repo(dst):
    d = Path(dst)
    (d / "cupy").mkdir(parents=True)
    (d / "cupy" / "__init__.py").write_text(STUB_CUPY)
    (d / "miner").mkdir()
    (d / "miner" / "__init__.py").write_text("")
    (d / "miner" / "range_miner_worker.py").write_text(STUB_RMW)
    (d / "miner" / "range_miner_protocol.py").write_text(STUB_PROTO)
    (d / "sieve_gpu_worker.py").write_text(STUB_SGW)
    # The stub repository shadows the REAL torch: any `import torch` in a stub-backed probe fails,
    # so no test can ever reach a CUDA/HIP device through torch (found in R2: the M11 mutant
    # reached the real torch on VM101 before this stub existed).
    (d / "torch").mkdir()
    (d / "torch" / "__init__.py").write_text(
        'raise ImportError("stub repository: the real torch is forbidden in off-GPU tests")\n')
    return d


def _real_extra(work, root):
    work, root = Path(work).resolve(), Path(root).resolve()
    stub = _stub_repo(work / "stubrepo")
    launch = (f"cd {stub} && {sys.executable} -B {root}/scripts/rig_seam_probe.py --controller {{ctl}} "
              f"--bdf {{bdf}} --torch-mode {{arm}} --attempt-id {{aid}} --backend real --repo {stub} "
              f"--local-trace {work}/probe_{{aid}}_{{rocr}}.jsonl > {work}/probe_{{aid}}_{{rocr}}.out 2>&1 &")
    return ("--launch-template", launch)


def gate_real_backend_bdf(root, work):
    good = Path(work) / "good"
    good.mkdir(parents=True, exist_ok=True)
    _, rows, ev = run_controller(root, good, "N", extra=_real_extra(good, root), env={"STUB_BDF": BDF1},
                                 proof_ticker=True)
    a = _final(rows)
    assert a["class"] == "C", f"G-REALBDF: matching BDF should run clean on stubs, got {a['class']} {a['trigger']}"
    ops = [op for _, op, _ in _ops(ev)]
    assert "cupy_free_default_pool" in ops and "cleanup_gc" in ops, \
        "G-REALBDF: the arm cleanup must replace the module global the executor calls"
    bad = Path(work) / "bad"
    bad.mkdir(parents=True, exist_ok=True)
    _, rows2, _ = run_controller(root, bad, "N", extra=_real_extra(bad, root), env={"STUB_BDF": "0000:19:00.0"})
    b = _final(rows2)
    assert b["class"] == "I" and not b["seam_sent_at"], \
        f"G-REALBDF: a BDF mismatch must stop before any stripe; got {b['class']} seam={b['seam_sent_at']}"
    assert b["trigger"] and b["trigger"]["why"] == "probe_refused", f"G-REALBDF: expected probe_refused, got {b['trigger']}"


def gate_real_pin(root, work):
    w = Path(work) / "mismatch"
    w.mkdir(parents=True, exist_ok=True)
    _, rows, _ = run_controller(root, w, "P", extra=_real_extra(w, root), env={"STUB_BDF": BDF1},
                                proof_ticker=True)
    a = _final(rows)
    assert a["trigger"] and a["trigger"]["why"] == "probe_refused", f"G-REALPIN: P with a changed cleanup must refuse, got {a['trigger']}"
    root2 = Path(work) / "nopin"
    _copy_root(root2, src=Path(root))      # copy the ROOT UNDER TEST (mutant or clean), not REPO
    (root2 / "scripts" / "rig_seam_probe.cleanup.sha256").unlink()
    w2 = Path(work) / "absent"
    w2.mkdir(parents=True, exist_ok=True)
    _, rows2, _ = run_controller(root2, w2, "P", extra=_real_extra(w2, root2), env={"STUB_BDF": BDF1})
    b = _final(rows2)
    assert b["trigger"] and b["trigger"]["why"] == "probe_refused", f"G-REALPIN: an absent pin must refuse P, got {b['trigger']}"


GATES = {f.__name__[5:]: f for f in (
    gate_clean_P, gate_arms_MN, gate_arm_P1, gate_fault_withhold, gate_warmup_fault,
    gate_hang_deadline, gate_trace_gap, gate_recorder_unobserved, gate_recorder_death,
    gate_recorder_absent, gate_recorder_per_host, gate_no_pre_proof, gate_shutdown_unreachable,
    gate_shutdown_timeout, gate_interrupt_barrier, gate_clear, gate_no_first_heartbeat,
    gate_resume_mode, gate_stats, gate_fw_range, gate_cleanup_pin, gate_real_backend_bdf,
    gate_real_pin)}

# (id, file, [(old, new), ...], credited gate)
MUTANTS = [
    ("M1-invalid-counted-clean", "rig_seam_controller.py",
     [("    elif complete and at.trigger is None:", "    elif True:")], "trace_gap"),
    ("M2-no-withholding", "rig_seam_controller.py",
     [('if self.trigger is not None and obj.get("cmd") not in ("stop", "exit"):', "if False:")], "fault_withhold"),
    ("M3-window-ignored", "rig_seam_controller.py",
     [('    in_win = [e for e in e1 if seam_start is not None and seam_start - a.window_before_s <= e["nc_ts"] <= win_end]',
       "    in_win = list(e1)")], "warmup_fault"),
    ("M4-deadline-ignored", "rig_seam_controller.py",
     [("        self.deadline = self.t0 + a.deadline_s\n", "        self.deadline = self.t0 + a.deadline_s * 100\n")],
     "hang_deadline"),
    ("M5-bound-over-B", "rig_seam_controller.py",
     [('"upper95_over_observed": upper_bound_one_sided(F, obs) if obs else None,',
       '"upper95_over_observed": upper_bound_one_sided(F, B) if obs else None,')], "stats"),
    ("M6-N-without-blocker", "rig_seam_probe.py",
     [('    if a.torch_mode == "N":\n        install_torch_blocker()', '    if False:\n        install_torch_blocker()')],
     "arms_MN"),
    ("M7-P-without-torch-sync", "rig_seam_probe.py",
     [('snap(tr, "after_torch_is_available")\n            if avail:', 'snap(tr, "after_torch_is_available")\n            if False:')],
     "clean_P"),
    ("M8-barrier-ignored", "rig_seam_controller.py",
     [("    if blocked:\n", "    if False:\n")], "fault_withhold"),
    ("M9-P1-torch-per-substripe", "rig_seam_probe.py", [('"P1": cleanup_M,', '"P1": cleanup_P,')], "arm_P1"),
    ("M10-real-BDF-guard-off", "rig_seam_probe.py",
     [("            if observed != expected_bdf:", "            if False:")], "real_backend_bdf"),
    ("M11-pin-lenient (restores the R1 code)", "rig_seam_probe.py",
     [('            if not want or len(want) != 64 or any(ch not in "0123456789abcdef" for ch in want):',
       "            if False:"),
      ("            if got != want:\n", "            if want and got != want:\n")], "real_pin"),
    ("M12-recorder-freshness-off", "rig_seam_controller.py",
     [("                if now - last >= self.a.rec_stale_s:", "                if False:")], "recorder_death"),
    ("M13-transport-failure-as-gone", "rig_seam_controller.py",
     [('    return f"UNKNOWN:rc={rc} out={out[-80:]!r}"', '    return "GONE"')], "shutdown_unreachable"),
    ("M14-no-durable-in-progress", "rig_seam_controller.py",
     [('            st["barrier"] = True          # durable: holds until a final record or a clear',
       '            st["barrier"] = False')], "interrupt_barrier"),
    ("M15-first-heartbeat-reset", "rig_seam_controller.py",
     [("                        self.last_rx.setdefault(key, now)", "                        pass"),
      ("self.last_rx.get(k, self.hello_at.get(k, now))", "self.last_rx.get(k, now)")], "no_first_heartbeat"),
    ("M16-FW-counted-negative", "rig_seam_controller.py",
     [('"e1_range": [F / B, (F + I + FW) / B] if B else None,', '"e1_range": [F / B, (F + I) / B] if B else None,')],
     "fw_range"),
    ("M17-predispatch-recorder-check-off", "rig_seam_controller.py",
     [("    if not fresh:\n", "    if False:\n")], "recorder_absent"),
    ("M18-timeout-raises", "rig_seam_controller.py",
     [('    except subprocess.TimeoutExpired:\n        return None, "", f"timeout after {timeout}s"',
       '    except subprocess.TimeoutExpired:\n        raise')], "shutdown_timeout"),
    ("M19-clear-without-map-check", "rig_seam_controller.py",
     [('        good = len(bdfs) > t["rocr"] and bdfs[t["rocr"]] == t["bdf"]', "        good = True")], "clear"),
]


def _copy_root(dst, src=REPO):
    (Path(dst) / "scripts").mkdir(parents=True)
    for f in ("rig_seam_probe.py", "rig_seam_controller.py", "rig_seam_probe.cleanup.sha256"):
        shutil.copy(Path(src) / "scripts" / f, Path(dst) / "scripts" / f)
    return Path(dst)


def run_gate_subprocess(name, root, work):
    return subprocess.run([sys.executable, "-B", __file__, "gate", name, "--root", str(root),
                           "--work", str(work)], capture_output=True, text=True, timeout=400)


def run_mutant(mid, fname, pairs, gate):
    with tempfile.TemporaryDirectory(dir=str(REPO / "logs")) as td:
        clean = _copy_root(Path(td) / "clean")
        cp = run_gate_subprocess(gate, clean, Path(td) / "w_clean")
        if cp.returncode != 0:
            return {"mutant": mid, "result": "INVALID", "why": "clean control failed",
                    "out": (cp.stdout + cp.stderr)[-800:]}
        mut = _copy_root(Path(td) / "mut")
        f = mut / "scripts" / fname
        src = f.read_text()
        for old, new in pairs:
            if src.count(old) != 1:
                return {"mutant": mid, "result": "INVALID", "why": f"pattern count {src.count(old)}: {old[:50]!r}"}
            src = src.replace(old, new)
        f.write_text(src)
        mp = run_gate_subprocess(gate, mut, Path(td) / "w_mut")
        out = mp.stdout + mp.stderr
        if mp.returncode == 0:
            return {"mutant": mid, "result": "SURVIVED", "gate": gate}
        if "AssertionError" in out and "G-" in out:
            line = [l for l in out.splitlines() if "AssertionError" in l][-1]
            return {"mutant": mid, "result": "DETECTED", "gate": gate, "assertion": line[-300:]}
        return {"mutant": mid, "result": "INVALID", "why": "died before the credited assertion", "out": out[-800:]}


# ------------------------------------------------------------------------------------------
# pytest entry points: one test per gate + one aggregate mutant test
# ------------------------------------------------------------------------------------------
def _pytest_gate(name):
    os.makedirs(REPO / "logs", exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(REPO / "logs")) as td:
        p = run_gate_subprocess(name, REPO, Path(td) / "w")
        assert p.returncode == 0, (p.stdout + p.stderr)[-2000:]


for _n in GATES:
    globals()[f"test_gate_{_n}"] = (lambda n: (lambda: _pytest_gate(n)))(_n)


def test_mutants_all_detected():
    os.makedirs(REPO / "logs", exist_ok=True)
    res = [run_mutant(*m) for m in MUTANTS]
    print(json.dumps(res, indent=1))
    bad = [r for r in res if r["result"] != "DETECTED"]
    assert not bad, json.dumps(bad, indent=1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "gate":
        name = sys.argv[2]
        root = sys.argv[sys.argv.index("--root") + 1]
        work = sys.argv[sys.argv.index("--work") + 1] if "--work" in sys.argv else tempfile.mkdtemp()
        GATES[name](root, work)
        print(f"GATE {name}: PASS")
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "mutants":
        print(json.dumps([run_mutant(*m) for m in MUTANTS], indent=1))
        sys.exit(0)
    print(__doc__)
