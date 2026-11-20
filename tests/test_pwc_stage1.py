#!/usr/bin/env python3
"""test_pwc_stage1.py — acceptance harness for PWC full-backup STAGE 1 additions.

Scope: docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md, Stage 1 row: B5 (remote
pkill self-match), A3 level 1 (TCP launch-set admission), B6 (legacy survivor
field name). The step-1 candidate (S1-S8) has its own suite,
tests/test_pwc_backup_step1.py; the four repaired old harnesses run on their own.

Run from the repository root:   python3 -u tests/test_pwc_stage1.py

CPU only. No GPU, no rig, no ssh, no network beyond 127.0.0.1:
  * every pkill/pgrep this suite EXECUTES runs inside a fresh unprivileged user +
    PID namespace (`unshare --user --map-root-user --pid --fork --mount-proc`),
    so it can see and signal only the dummy processes the suite starts there —
    never a process on this box (CLAUDE.md hard rule 8). If unshare is not
    available those gates report UNAVAILABLE and the suite does not pass;
  * the A3 transport gates bind a real TCPWorkerTransport on 127.0.0.1, port 0
    (the OS picks a free port), and talk to it with fake workers;
  * startup()/_tcp_launch_workers gates replace subprocess.run/Popen with
    recorders; nothing is sent anywhere;
  * every file written goes under <root>/logs/pwc_stage1_test/ (ignored path).

Each repair has a MUTANT: a textual edit of the LIVE source that undoes it,
loaded as a separate module. A mutant is killed only if the edit applies exactly
once, the same detector passes on the live module, and the detector fails on the
mutant with its own AssertionError.
"""
import importlib.util
import io
import contextlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = os.path.join(ROOT, "logs", "pwc_stage1_test")
PWC_PATH = os.path.join(ROOT, "persistent_worker_coordinator.py")
TCP_PATH = os.path.join(ROOT, "persistent", "pwc_transport_tcp.py")

import persistent_worker_coordinator as PWC                 # noqa: E402
import persistent.pwc_transport_tcp as TCP                  # noqa: E402

RIGS = ("192.168.3.122", "192.168.3.156")
DIAL = "192.168.3.177"


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


def _head_source():
    r = subprocess.run(["git", "show", "HEAD:persistent_worker_coordinator.py"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise Unavailable(f"git show HEAD failed: {r.stderr.strip()[:120]}")
    return r.stdout


# ═════════════════════════════════════════════════════════════════════════════
# Isolated execution of a remote shell string (B5)
# ═════════════════════════════════════════════════════════════════════════════
_HARNESS = r'''#!/bin/bash
# Runs inside a private PID namespace. The dummy names are assembled from parts so
# that THIS script's command line ("bash <file>") never contains them.
a=pwc_worker; b=_service; c=sieve_gpu; d=_worker.py
DUMMY_PWC="${a}${b}"; DUMMY_SIEVE="/opt/x/${c}${d}"
pids=""; p_pwc=""; p_sieve=""
if [ "$WITH_DUMMIES" = 1 ]; then
  python3 -c 'import time; time.sleep(120)' "$DUMMY_PWC" --gpu-id 0 & p_pwc=$!
  python3 -c 'import time; time.sleep(120)' "$DUMMY_SIEVE" --persistent & p_sieve=$!
  pids="$p_pwc $p_sieve"
  sleep 0.4
fi
[ -n "$FAKE_PKILL_DIR" ] && export PATH="$FAKE_PKILL_DIR:$PATH"
bash -c "$CMD" > "$OUTDIR/stdout" 2> "$OUTDIR/stderr"
echo $? > "$OUTDIR/rc"
alive=0
for p in $pids; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
echo "$alive" > "$OUTDIR/alive"
a_pwc=0; a_sieve=0
[ -n "$p_pwc" ] && kill -0 "$p_pwc" 2>/dev/null && a_pwc=1
[ -n "$p_sieve" ] && kill -0 "$p_sieve" 2>/dev/null && a_sieve=1
echo "$a_pwc $a_sieve" > "$OUTDIR/alive_each"
for p in $pids; do kill -9 "$p" 2>/dev/null; done
exit 0
'''


def run_isolated(cmd, *, dummies=True, fake_pkill=False):
    """Run `bash -c cmd` (what sshd runs on a rig) in a private PID namespace
    with two dummy worker processes. Returns rc, stdout lines, dummies alive."""
    if shutil.which("unshare") is None:
        raise Unavailable("unshare not installed")
    d = tempfile.mkdtemp(prefix="iso_", dir=OUT)
    h = os.path.join(d, "harness.sh")
    with open(h, "w") as f:
        f.write(_HARNESS)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "CMD": cmd, "OUTDIR": d,
           "WITH_DUMMIES": "1" if dummies else "0", "FAKE_PKILL_DIR": ""}
    if fake_pkill:
        fp = os.path.join(d, "fakebin")
        os.makedirs(fp)
        with open(os.path.join(fp, "pkill"), "w") as f:
            f.write("#!/bin/sh\nexit 0\n")           # a pkill that kills nothing
        os.chmod(os.path.join(fp, "pkill"), 0o755)
        env["FAKE_PKILL_DIR"] = fp
    r = subprocess.run(["unshare", "--user", "--map-root-user", "--pid", "--fork",
                        "--mount-proc", "bash", h], env=env, capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0 or not os.path.exists(os.path.join(d, "rc")):
        raise Unavailable(f"unshare rc={r.returncode} stderr={r.stderr.strip()[:160]!r}")
    read = lambda n: open(os.path.join(d, n)).read()
    a_pwc, a_sieve = (int(x) for x in read("alive_each").split())
    return {"rc": int(read("rc")), "stdout": read("stdout").strip().splitlines(),
            "alive": int(read("alive")), "alive_pwc": a_pwc, "alive_sieve": a_sieve,
            "stderr": read("stderr")}


# ── B5 evidence: the strings on main ────────────────────────────────────────
_OLD_TCP_KILL_LINES = ('"pkill -9 -f pwc_worker_service 2>/dev/null; "',
                       '"pkill -9 -f sieve_gpu_worker 2>/dev/null; "',
                       '"sleep 2; echo killed"')
_OLD_S156_FIND_LINE = '''"'pgrep -af \\"sieve_gpu_worker.*--persistent\\" 2>/dev/null "'''


def d_b5_old_self_kill():
    """EVIDENCE (no mutant): the TCP kill string on HEAD kills its own shell.
    The three source lines are read from `git show HEAD:` so the string is the
    committed one, not one typed from memory."""
    head = _head_source()
    for ln in _OLD_TCP_KILL_LINES:
        assert ln in head, f"HEAD no longer carries {ln}"
    old = ("pkill -9 -f pwc_worker_service 2>/dev/null; "
           "pkill -9 -f sieve_gpu_worker 2>/dev/null; sleep 2; echo killed")
    r = run_isolated(old)
    assert "killed" not in r["stdout"], f"old string printed its marker: {r}"
    assert r["rc"] in (137, -9), f"expected the shell SIGKILLed (137), got {r}"
    # The sieve dummy can only die from the SECOND pkill, so it surviving shows that
    # pkill never ran. Whether the pwc dummy dies is a race between the first
    # pkill's signals and its own parent shell's death (seen both ways on VM101),
    # so it is reported, not asserted.
    assert r["alive_sieve"] == 1, f"sieve dummy was killed, so the second pkill ran: {r}"
    return (f"HEAD's string: shell rc={r['rc']} (SIGKILL), no 'killed', sieve dummy alive "
            f"(second pkill never ran); pwc dummy {'alive' if r['alive_pwc'] else 'killed'} this time")


def d_b5_new_kill(M=PWC):
    cmd = M.pwc_stale_kill_command()
    r = run_isolated(cmd)
    assert r["rc"] == 0 and r["stdout"][-1:] == [M.PWC_STALE_KILL_OK], r
    assert r["alive_pwc"] == 0 and r["alive_sieve"] == 0, f"dummies left alive: {r}"
    r0 = run_isolated(cmd, dummies=False)
    assert r0["rc"] == 0 and r0["stdout"][-1:] == [M.PWC_STALE_KILL_OK], r0
    rs = run_isolated(cmd, fake_pkill=True)
    assert rs["rc"] == 6 and M.PWC_STALE_KILL_OK not in rs["stdout"], rs
    return ("new string: both dummies killed, shell survives, marker printed; no workers: "
            "marker; a worker that survives the kill: rc 6, no marker")


def _capture_s156_strings(M):
    """Run the REAL startup() (SSH route) with subprocess.run recorded; the find
    reports a stale worker so the reap string is built. Return (find, reap)."""
    cfg = _write_config(gpu_count=8)
    p = M.PersistentWorkerCoordinator(config_file=cfg, worker_pool_size=6, seed_cap_amd=25_000,
                                      pwc_transport="ssh", pwc_coordinator_address=DIAL)
    p._reset_rig_cupy_cache = lambda *a, **k: False       # stop after the reap
    seen = []

    def fake_run(cmd, *a, **k):
        seen.append(cmd)
        out = "4242 python3 sieve_gpu_worker.py --persistent" if "pgrep -af" in str(cmd) else ""
        return subprocess.CompletedProcess(cmd, 0, out, "")
    saved = (subprocess.run, time.sleep, sys.modules.get("progress_display"))
    subprocess.run, time.sleep = fake_run, (lambda s: None)
    stub = types.ModuleType("progress_display")
    stub.ProgressWriter = lambda *a, **k: types.SimpleNamespace(
        register_node=lambda *a, **k: None)
    sys.modules["progress_display"] = stub
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            p.startup()
    finally:
        subprocess.run, time.sleep = saved[0], saved[1]
        if saved[2] is None:
            sys.modules.pop("progress_display", None)
        else:
            sys.modules["progress_display"] = saved[2]
    strs = [c for c in seen if isinstance(c, str) and c.startswith("ssh ")]
    find = next(c for c in strs if "pgrep -af" in c)
    reap = next(c for c in strs if "pkill -15" in c)
    remote = lambda c: c[c.index("'") + 1:c.rindex("'")]
    return remote(find), remote(reap)


def d_b5_s156(M=PWC):
    find, reap = _capture_s156_strings(M)
    r = run_isolated(find, dummies=False)
    assert r["stdout"] == ["none"], f"find matched something with no worker running: {r}"
    r = run_isolated(reap)
    assert r["rc"] == 0 and "reaped:0" in " ".join(r["stdout"]).split(), r
    assert r["alive_sieve"] == 0 and r["alive_pwc"] == 1, \
        f"reap must kill the --persistent dummy and only it: {r}"
    head = _head_source()
    assert _OLD_S156_FIND_LINE in head, "HEAD no longer carries the old S156 find"
    old_find = 'pgrep -af "sieve_gpu_worker.*--persistent" 2>/dev/null || echo none'
    ro = run_isolated(old_find, dummies=False)
    return (f"live S156 find with no worker: 'none'; reap kills the --persistent dummy, "
            f"shell survives, reaped:0. HEAD's find with no worker printed "
            f"{ro['stdout'][:1]!r} (its own shell) instead of 'none'")


# ═════════════════════════════════════════════════════════════════════════════
# _tcp_launch_workers wiring (B5 result read, A3 admission)
# ═════════════════════════════════════════════════════════════════════════════
def _write_config(gpu_count=8, rigs=RIGS):
    os.makedirs(OUT, exist_ok=True)
    cfg = {"nodes": [{"hostname": h, "gpu_type": "RX 6600 XT", "gpu_count": gpu_count,
                      "python_env": "/home/michael/rocm_env/bin/python3",
                      "script_path": "/home/michael/distributed_prng_analysis",
                      "username": "michael"} for h in rigs]}
    fd, path = tempfile.mkstemp(prefix="cfg_", suffix=".json", dir=OUT)
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f)
    return path


class Admission:
    """Recorder for the launch-set API. `events` keeps the call order."""
    def __init__(self):
        self.allowed = None
        self.events = []
    def set_allowed_worker_ids(self, ids):
        self.events.append(("set", sorted(ids)))
        self.allowed = set(ids)
    def allow_worker_id(self, w):
        if self.allowed is None:
            raise RuntimeError("allow before set")
        self.events.append(("allow", w))
        self.allowed.add(w)
    def revoke_worker_id(self, w):
        self.events.append(("revoke", w))
        if self.allowed is not None:
            self.allowed.discard(w)
    def start(self):
        self.events.append(("start",))
    def stop(self):
        pass
    def ready_count(self):
        return 0
    def worker_count(self):
        return 0
    def online_count(self):
        return 0
    def broadcast_init(self):
        return 0


def _launch(M, kill_reply, *, no_pid=(), pool=2):
    """Drive the REAL _tcp_launch_workers. kill_reply(host) -> CompletedProcess or
    raise. `no_pid` = {(host, gpu)} whose launch prints no PID line."""
    p = M.PersistentWorkerCoordinator(config_file=_write_config(), worker_pool_size=pool,
                                      seed_cap_amd=25_000, pwc_transport="tcp",
                                      pwc_coordinator_address=DIAL)
    p._tcp_transport = Admission()
    p._tcp_transport.set_allowed_worker_ids(set())
    p._probe_live_rocm_gpu_count = lambda node: 8
    p._tcp_wait_online = lambda expected, timeout_s=0: 0
    p._tcp_broadcast_init = lambda: 0
    p._tcp_wait_ready = lambda expected, timeout_s=0: 0
    resets, scripts = [], []
    p._reset_rig_cupy_cache = lambda ssh_base, host, pool: resets.append(host) or True

    def fake_run(cmd, *a, **k):
        flat = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
        host = next((h for h in RIGS if h in flat), None)
        if flat.startswith("pkill"):                      # VM101-local kill: recorded only
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        if M.PWC_STALE_KILL_OK in flat:
            return kill_reply(host)
        if "cat > " in flat:
            scripts.append(host)
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        if "bash /tmp/pwc_tcp_launch_gpu" in flat:
            gpu = int(flat.rsplit("gpu", 1)[1].split(".")[0])
            out = "" if (host, gpu) in no_pid else "PID=4242"
            return subprocess.CompletedProcess(cmd, 0, out, "")
        raise AssertionError(f"unexpected subprocess: {flat[:120]}")
    saved = (subprocess.run, time.sleep)
    subprocess.run, time.sleep = fake_run, (lambda s: None)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            p._tcp_launch_workers()
    finally:
        subprocess.run, time.sleep = saved
    return p, resets, scripts


def _ok(host):
    return subprocess.CompletedProcess([], 0, "PWC_STALE_KILL_OK\n", "")


def d_b5_wiring(M=PWC):
    good, bad = RIGS
    cases = {
        "no marker": lambda h: _ok(h) if h == good else subprocess.CompletedProcess([], 0, "", ""),
        "marker but rc 6": lambda h: _ok(h) if h == good else
            subprocess.CompletedProcess([], 6, "PWC_STALE_KILL_OK\n", ""),
        "marker not last": lambda h: _ok(h) if h == good else
            subprocess.CompletedProcess([], 0, "PWC_STALE_KILL_OK\nx\n", ""),
        "ssh raises": lambda h: _ok(h) if h == good else (_ for _ in ()).throw(
            subprocess.TimeoutExpired("ssh", 15)),
    }
    for name, reply in cases.items():
        p, resets, scripts = _launch(M, reply)
        assert bad not in resets and bad not in scripts, f"{name}: refused rig went on: {resets} {scripts}"
        assert bad not in p._effective_pool_by_host, f"{name}: {p._effective_pool_by_host}"
        assert good in resets and scripts.count(good) == 2, f"{name}: healthy rig blocked"
        assert not any(w.startswith(bad.replace(".", "_")) for w in p._tcp_transport.allowed), name
    return "kill unconfirmed (no marker / rc 6 / marker not last / ssh error): rig refused, no reset, no launch, not admitted; healthy rig unaffected"


def d_a3_launch_set(M=PWC):
    good, other = RIGS
    p, resets, scripts = _launch(M, _ok, no_pid={(other, 1)})
    want = {f"{good.replace('.', '_')}_gpu0", f"{good.replace('.', '_')}_gpu1",
            f"{other.replace('.', '_')}_gpu0"}
    assert p._tcp_transport.allowed == want, p._tcp_transport.allowed
    ev = p._tcp_transport.events
    i_allow = ev.index(("allow", f"{other.replace('.', '_')}_gpu1"))
    assert ("revoke", f"{other.replace('.', '_')}_gpu1") in ev[i_allow:], ev
    return f"admitted = ids whose launch printed a PID: {sorted(want)}; the no-PID launch was revoked"


def d_a3_startup_order(M=PWC):
    """startup() (TCP) opens the EMPTY set before the listener starts."""
    import persistent.pwc_transport_base as TB
    rec = Admission()
    p = M.PersistentWorkerCoordinator(config_file=_write_config(), worker_pool_size=2,
                                      seed_cap_amd=25_000, pwc_transport="tcp",
                                      pwc_coordinator_address=DIAL)
    p._tcp_launch_workers = lambda: None
    saved = (TB.build_transport, subprocess.run, time.sleep, sys.modules.get("progress_display"))
    TB.build_transport = lambda **kw: rec
    subprocess.run = lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "none", "")
    time.sleep = lambda s: None
    stub = types.ModuleType("progress_display")
    stub.ProgressWriter = lambda *a, **k: types.SimpleNamespace(register_node=lambda *a, **k: None)
    sys.modules["progress_display"] = stub
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            p.startup()
    finally:
        TB.build_transport, subprocess.run, time.sleep = saved[0], saved[1], saved[2]
        if saved[3] is None:
            sys.modules.pop("progress_display", None)
        else:
            sys.modules["progress_display"] = saved[3]
    assert rec.events[:2] == [("set", []), ("start",)], rec.events
    return "set_allowed_worker_ids(set()) precedes start(): nothing is admitted before a launch"


# ═════════════════════════════════════════════════════════════════════════════
# A3 on a real transport, 127.0.0.1 only
# ═════════════════════════════════════════════════════════════════════════════
def _send(sock, obj):
    b = json.dumps(obj, separators=(",", ":")).encode()
    sock.sendall(struct.pack(">I", len(b)) + b)


def _recv(sock):
    hdr = b""
    while len(hdr) < 4:
        c = sock.recv(4 - len(hdr))
        if not c:
            raise ConnectionError("closed")
        hdr += c
    n = struct.unpack(">I", hdr)[0]
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise ConnectionError("closed")
        buf += c
    return json.loads(buf)


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


def d_a3_transport(T=TCP):
    tr = T.TCPWorkerTransport(host="127.0.0.1", port=0)
    tr.set_allowed_worker_ids({"rig_gpu0"})
    tr.start()
    port = tr._server.getsockname()[1]
    socks = []
    try:
        s, ack = _hello(port, "stale_gpu7"); socks.append(s)
        assert ack.get("accepted") is False, f"unlisted id accepted: {ack}"
        assert _wait(lambda: "stale_gpu7" in tr.refused_hellos)
        assert tr.worker_count() == 0, "refused worker was registered"
        s, ack = _hello(port, "rig_gpu0"); socks.append(s)
        assert ack.get("accepted") is True, ack
        assert _wait(lambda: tr.worker_count() == 1), tr.worker_count()
        _send(s, {"message_type": "online", "worker_id": "rig_gpu0", "timestamp": time.time()})
        assert _wait(lambda: tr.online_count() == 1), tr.online_count()
        # a refused hello under a registered id must not evict the registered one
        tr.revoke_worker_id("rig_gpu0")
        s2, ack2 = _hello(port, "rig_gpu0"); socks.append(s2)
        assert ack2.get("accepted") is False, ack2
        time.sleep(0.3)
        assert tr.worker_count() == 1 and tr.online_count() == 1, \
            f"refused duplicate evicted the live worker: {tr.worker_count()}/{tr.online_count()}"
    finally:
        for s in socks:
            s.close()
        tr.stop()
    # no list at all = the old behaviour (any id accepted)
    tr = T.TCPWorkerTransport(host="127.0.0.1", port=0)
    tr.start()
    port = tr._server.getsockname()[1]
    try:
        s, ack = _hello(port, "anyone")
        assert ack.get("accepted") is True, ack
        s.close()
    finally:
        tr.stop()
    return ("unlisted id refused (accepted=False, not registered); listed id served; "
            "refused duplicate leaves the live worker in place; no list = accept all")


# ═════════════════════════════════════════════════════════════════════════════
# B6
# ═════════════════════════════════════════════════════════════════════════════
def d_b6(M=PWC):
    p = M.PersistentWorkerCoordinator.__new__(M.PersistentWorkerCoordinator)
    reply = {}
    p._tcp_transport = types.SimpleNamespace(
        worker_count=lambda: 1, submit_job=lambda job: None,
        recv_result=lambda timeout_s=None, job_id=None: reply["r"])
    def call(surv):
        reply["r"] = {"status": "ok", "result": {"payload": {"survivors": surv}}}
        return p._dispatch_to_tcp({"job_id": "j1", "hybrid": True})
    r = call([{"seed": 5, "match_rate": 0.6, "skip_pattern": [1, 2], "strategy_id": 3}])
    assert r["skip_sequences"] == [[1, 2]] and r["survivors"] == [5], r
    r = call([{"seed": 6, "match_rate": 0.6, "skip_sequence": [4], "strategy_id": 0}])
    assert r["skip_sequences"] == [[4]], r
    return "legacy record: skip_pattern (sieve_filter's name) read; skip_sequence still read"


# ═════════════════════════════════════════════════════════════════════════════
# The miner launch process still imports the edited files
# ═════════════════════════════════════════════════════════════════════════════
def d_import():
    code = ("import window_optimizer_integration_final as W, persistent_worker_coordinator as P, "
            "persistent.pwc_transport_tcp as T, persistent.pwc_transport_base as B, "
            "persistent.pwc_protocol as R, miner\n"
            "from miner import run_trial_miner\n"
            "assert W.run_trial_persistent is P.run_trial_persistent\n"
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       timeout=180, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-400:]
    return "integration module (Run 4's launch import) + miner package import cleanly, fresh process"


GATES = [
    ("G-B5-OLD-SELF-KILL (evidence)", d_b5_old_self_kill),
    ("G-B5-NEW-KILL", d_b5_new_kill),
    ("G-B5-S156", d_b5_s156),
    ("G-B5-WIRING", d_b5_wiring),
    ("G-A3-STARTUP-ORDER", d_a3_startup_order),
    ("G-A3-LAUNCH-SET", d_a3_launch_set),
    ("G-A3-TRANSPORT", d_a3_transport),
    ("G-B6", d_b6),
    ("G-IMPORT", d_import),
]

MUTANTS = [
    ("M-B5a kill marker not read", PWC_PATH,
     '            if (_kill_r.returncode != 0\n'
     '                    or (_kill_r.stdout or "").strip().splitlines()[-1:] != [PWC_STALE_KILL_OK]):\n',
     '            if False:\n', d_b5_wiring, "pwc"),
    ("M-B5b unbracketed kill patterns restored", PWC_PATH,
     'PWC_STALE_KILL_PATTERNS = ("[p]wc_worker_service", "[s]ieve_gpu_worker")\n',
     'PWC_STALE_KILL_PATTERNS = ("pwc_worker_service", "sieve_gpu_worker")\n',
     d_b5_new_kill, "pwc"),
    ("M-B5c unbracketed S156 pattern restored", PWC_PATH,
     'PWC_S156_PERSISTENT_PATTERN = "[s]ieve_gpu_worker.*--persistent"\n',
     'PWC_S156_PERSISTENT_PATTERN = "sieve_gpu_worker.*--persistent"\n',
     d_b5_s156, "pwc"),
    ("M-A3a startup does not open an empty set", PWC_PATH,
     '                self._tcp_transport.set_allowed_worker_ids(set())\n', '',
     d_a3_startup_order, "pwc"),
    ("M-A3b no-PID launch not revoked", PWC_PATH,
     '                        # [PWC-STAGE1 A3 level 1] no PID: not in the launch set\n'
     '                        self._tcp_transport.revoke_worker_id(worker_id)\n',
     '                        pass\n', d_a3_launch_set, "pwc"),
    ("M-A3c transport admits every id", TCP_PATH,
     '            return self._allowed_worker_ids is None or str(worker_id) in self._allowed_worker_ids\n',
     '            return True\n', d_a3_transport, "tcp"),
    ("M-A3d refused hello cleans up a same-named live worker", TCP_PATH,
     '            if not refused:\n', '            if True:\n', d_a3_transport, "tcp"),
    ("M-B6 skip_pattern not read", PWC_PATH,
     '            skip_seqs   = [(s.get("skip_pattern", s.get("skip_sequence", []))'
     ' if isinstance(s, dict) else [])\n',
     '            skip_seqs   = [(s.get("skip_sequence", []) if isinstance(s, dict) else [])\n',
     d_b6, "pwc"),
]


def run_mutant(label, path, old, new, detector, kind):
    src = _mutate(path, old, new)
    detector()                                                      # positive control
    if kind == "pwc":
        target = _load_source(src, "pwc_stage1_mutant", PWC_PATH)
    else:
        target = _load_source(src, "tcp_stage1_mutant", TCP_PATH)
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
    for label, path, old, new, det, kind in MUTANTS:
        try:
            print(f"[PASS] {label} -> {run_mutant(label, path, old, new, det, kind)}", flush=True)
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
    shutil.rmtree(OUT, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
