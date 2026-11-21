#!/usr/bin/env python3
"""test_pwc_shutdown_reap.py — PWC TCP rig workers end on a PLANNED shutdown, and the
coordinator's shutdown reap touches only its OWN workers (Stage 4, 2026-11-21: after
`[PWC-TCP] transport stopped` all 24 rig pwc_worker_service processes were still alive 2
minutes later). CPU only: no GPU, no rig, no network; SSH is stubbed; the worker's
connect/init/ready steps are replaced by fakes.

Run from the repository root:   python3 -u tests/test_pwc_shutdown_reap.py

A gate that cannot run prints [UNAVAILABLE] (never [PASS]) and the RESULT is INCOMPLETE
(Beta X-PWC-SHUTDOWN R1 Q8; docs/VERIFICATION_INTEGRITY_STANDARD.md).

Worker (persistent/pwc_worker_service.py — a rig file):
  W1 shutdown received in the main loop      -> run_forever returns after ONE connect
  W2 shutdown received while waiting for init -> returns, no reconnect
  W3 unplanned loss AFTER READY -> exits after exactly _MAX_REFUSED refusals (~5 min,
     S163-KARG Fix 8); before the fix a dropped session never counted and retried forever
  W4 unplanned loss BEFORE any session -> keeps retrying (the recorded behaviour)
  W-MUTANT-A / -B controlled mutations of the CURRENT source (not git HEAD, so a clean
     checkout after landing still exercises them): A removes the planned-shutdown exit
     (W1 must fail), B moves `_had_session = True` back after `_main_loop()` (W3 must fail)
Coordinator (persistent_worker_coordinator.py — VM101 only):
  C1 classify_shutdown_reap: exited / killed / survivor / unavailable / error
  C2 shutdown(): transport stop, THEN the token-bound reap of every launch-attempted rig and
     the local workers; an ssh exception, a failed local wait+kill and a THROWING transport
     stop are all recorded, never raised; an incomplete reap logs "shutdown INCOMPLETE",
     never "All workers shut down"
  C3 the reap shell in real bash against dummy processes carrying the token:
     exits by itself -> exited; lingers -> killed; pkill stubbed -> survivor (rc 6);
     pgrep stubbed to exit 2 -> unavailable (rc 7)
  C3-TOKEN a dummy carrying ANOTHER coordinator's token survives this coordinator's reap
  C4 the reap pattern is PWC-only, bracketed and token-bound; a malformed token is refused
  C5 _tcp_launch_workers bookkeeping: every worker's launch line carries this coordinator's
     token; a rig is a reap target from the launch ATTEMPT, even when every acknowledgment
     is lost (no PID line), and is not one when the stale kill refused it
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
OUT = os.path.join(ROOT, "logs", "test_pwc_shutdown_reap")
WORKER_SRC = os.path.join(ROOT, "persistent", "pwc_worker_service.py")

import persistent.pwc_worker_service as WS                                   # noqa: E402
import persistent_worker_coordinator as PWC                                   # noqa: E402

UNAVAILABLE = "UNAVAILABLE"


class Stop(Exception):
    """Raised by the patched sleep to end a loop that would otherwise run forever."""


def fake_worker(mod, script, max_sleeps=200):
    """A PWCWorkerService whose network steps follow `script`, a list of per-connect
    plans: 'refused', 'init_shutdown', 'main_shutdown', 'main_drop'."""
    w = mod.PWCWorkerService(worker_id="t:gpu0", host="127.0.0.1", port=1, gpu_id=0)
    state = {"connects": 0, "sleeps": 0, "plan": None}
    it = iter(script)

    def _connect():
        state["connects"] += 1
        state["plan"] = next(it, "refused")
        if state["plan"] == "refused":
            raise ConnectionRefusedError("[Errno 111] Connection refused")
        w._conn = types.SimpleNamespace(close=lambda: None)

    def _wait_for_init():
        if state["plan"] == "init_shutdown":
            msg = {"message_type": "shutdown", "reason": "coordinator_request"}
            w._conn = types.SimpleNamespace(recv_obj=lambda: msg, close=lambda: None)
            return mod.PWCWorkerService._wait_for_init(w)

    def _main_loop():
        if state["plan"] == "main_shutdown":
            msgs = iter([{"message_type": "shutdown", "reason": "coordinator_request"}])
            w._conn = types.SimpleNamespace(send_obj=lambda o: None, recv_obj=lambda: next(msgs),
                                            close=lambda: None)
            return mod.PWCWorkerService._main_loop(w)
        if state["plan"] == "main_drop":
            raise ConnectionResetError("[Errno 104] Connection reset by peer")

    for name, fn in (("_setup_env", lambda: None), ("_connect", _connect),
                     ("_send_online", lambda: None), ("_wait_for_init", _wait_for_init),
                     ("_import_sieve", lambda: None), ("_validate_compute_device", lambda: None),
                     ("_send_ready", lambda: None), ("_main_loop", _main_loop),
                     ("_emit_heartbeat", lambda *a, **k: None), ("_close", lambda: None)):
        setattr(w, name, fn)

    def _sleep(_s):
        state["sleeps"] += 1
        if state["sleeps"] > max_sleeps:
            raise Stop()
    return w, state, _sleep


def run_worker(mod, script, max_sleeps=200):
    w, state, sl = fake_worker(mod, script, max_sleeps)
    real = mod.time.sleep
    mod.time.sleep = sl
    try:
        w.run_forever()
        state["returned"] = True
    except Stop:
        state["returned"] = False
    finally:
        mod.time.sleep = real
    return state


def w1(mod=WS):
    st = run_worker(mod, ["main_shutdown"])
    assert st["returned"] and st["connects"] == 1 and st["sleeps"] == 0, st
    return f"returned after {st['connects']} connect, {st['sleeps']} reconnect sleeps"


def w2(mod=WS):
    st = run_worker(mod, ["init_shutdown"])
    assert st["returned"] and st["connects"] == 1, st
    return f"returned after {st['connects']} connect"


def w3(mod=WS):
    st = run_worker(mod, ["main_drop"], max_sleeps=500)
    assert st["returned"], f"never exited: {st}"
    assert st["connects"] == 1 + 60, st
    return f"exited after {st['connects'] - 1} refused reconnects (_MAX_REFUSED = 60, ~5 min)"


def w4(mod=WS):
    st = run_worker(mod, [], max_sleeps=150)
    assert not st["returned"] and st["connects"] > 100, st
    return f"no session yet: still retrying after {st['connects']} refused connects (recorded behaviour)"


# Controlled mutations of the CURRENT worker source (each must apply exactly once).
MUT_A = ('''            if self._shutdown_requested:
                log.info(f"[{self.worker_id}] planned shutdown "
                         f"({self._shutdown_requested}) — exiting, no reconnect")
                self._emit_heartbeat("shutdown")
                self._close()
                break''', '''            pass''')
MUT_A2 = ('''                if self._shutdown_requested:
                    continue                     # the top of the loop exits''', '''                pass''')
MUT_B = ('''                _had_session = True

                self._main_loop()''', '''                self._main_loop()
                _had_session = True''')


def mutated(*edits):
    src = open(WORKER_SRC).read()
    for old, new in edits:
        assert src.count(old) == 1, f"mutation anchor not found exactly once: {old[:60]!r}"
        src = src.replace(old, new)
    mod = types.ModuleType("pwc_worker_service_mutant")
    mod.__file__ = WORKER_SRC
    exec(compile(src, WORKER_SRC, "exec"), mod.__dict__)
    return mod


def w_mutant_a():
    try:
        w1(mutated(MUT_A, MUT_A2))
    except AssertionError:
        return "without the planned-shutdown exit, W1 fails (the worker reconnects)"
    raise AssertionError("W1 passed against the mutant: it does not test the exit")


def w_mutant_b():
    try:
        w3(mutated(MUT_B))
    except AssertionError:
        return "with _had_session set only after _main_loop returns, W3 fails (retries forever)"
    raise AssertionError("W3 passed against the mutant: it does not test the session timing")


def c1():
    t = PWC.PWC_SHUTDOWN_WAIT_TAG
    ok = PWC.PWC_STALE_KILL_OK
    cases = [((0, f"{t}:0:1\n{ok}\n"), "exited"), ((0, f"{t}:15:0\n{ok}\n"), "killed"),
             ((6, f"{t}:15:0\n"), "survivor"), ((7, f"{t}:0:1\n"), "unavailable"),
             ((0, f"{t}:0:2\n{ok}\n"), "unavailable"), ((0, "garbage"), "error"),
             ((255, ""), "error"), ((0, f"{ok}\n"), "error")]
    for (rc, out), want in cases:
        got = PWC.classify_shutdown_reap(rc, out)
        assert got == want, (rc, out, got, want)
    return f"{len(cases)} result shapes classified"


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(("INFO", m))

    def warning(self, m):
        self.lines.append(("WARN", m))

    def error(self, m):
        self.lines.append(("ERROR", m))


def bare_coordinator(order, stop_raises=False):
    c = PWC.PersistentWorkerCoordinator.__new__(PWC.PersistentWorkerCoordinator)
    c.logger = _Log()
    c._progress_writer = None
    c.workers = []

    def _stop():
        order.append("stop")
        if stop_raises:
            raise OSError("listener already closed")
    c._tcp_transport = types.SimpleNamespace(stop=_stop)
    c._tcp_reap_targets = {"rigA": ["ssh", "u@rigA"], "rigB": ["ssh", "u@rigB"],
                           "rigC": ["ssh", "u@rigC"]}
    c._tcp_local_procs = []
    c.shutdown_reap = {}
    c._launch_token = uuid.uuid4().hex
    return c


class LocalProc:
    def __init__(self, pid, exits, kill_fails=False):
        self.pid, self.exits, self.kill_fails, self.killed = pid, exits, kill_fails, False

    def wait(self, timeout=None):
        if self.exits or self.killed:
            return 0
        raise subprocess.TimeoutExpired("x", timeout)

    def kill(self):
        if self.kill_fails:
            raise PermissionError("operation not permitted")
        self.killed = True


def _run_shutdown(c, outputs):
    calls, order = [], c._order

    def fake_run(argv, **kw):
        host = argv[1].split("@")[1]
        order.append(f"reap:{host}")
        calls.append(argv)
        out = outputs[host]
        if isinstance(out, Exception):
            raise out
        return types.SimpleNamespace(returncode=out[0], stdout=out[1], stderr="")
    real = subprocess.run
    subprocess.run = fake_run
    try:
        c.shutdown()                                   # must not raise
    finally:
        subprocess.run = real
    return calls


def c2():
    t, ok = PWC.PWC_SHUTDOWN_WAIT_TAG, PWC.PWC_STALE_KILL_OK
    res = []
    for stop_raises in (False, True):
        order = []
        c = bare_coordinator(order, stop_raises=stop_raises)
        c._order = order
        lp = [LocalProc(11, True), LocalProc(12, False), LocalProc(13, False, kill_fails=True)]
        c._tcp_local_procs = list(lp)
        calls = _run_shutdown(c, {"rigA": (0, f"{t}:0:1\n{ok}\n"), "rigB": (0, f"{t}:15:0\n{ok}\n"),
                                  "rigC": OSError("ssh: No route to host")})
        assert order[0] == "stop" and sorted(order[1:]) == ["reap:rigA", "reap:rigB", "reap:rigC"], order
        want_cmd = PWC.pwc_shutdown_reap_command(token=c._launch_token)
        assert all(a[-1] == want_cmd for a in calls), "reap is not token-bound"
        r = c.shutdown_reap
        assert (r["rigA"], r["rigB"], r["rigC"]) == ("exited", "killed", "error"), r
        assert (r["localhost:pid11"], r["localhost:pid12"], r["localhost:pid13"]) == \
            ("exited", "killed", "error"), r
        msgs = [m for _, m in c.logger.lines]
        assert any("shutdown INCOMPLETE" in m for m in msgs), msgs
        assert not any(m == "All workers shut down" for m in msgs), "all-clear logged after an error"
        assert c._tcp_reap_targets == {} and c._tcp_local_procs == []
        res.append("stop raising" if stop_raises else "stop ok")
    # a fully clean reap logs the all-clear
    order = []
    c = bare_coordinator(order)
    c._order = order
    c._tcp_reap_targets = {"rigA": ["ssh", "u@rigA"]}
    _run_shutdown(c, {"rigA": (0, f"{t}:0:1\n{ok}\n")})
    assert ("INFO", "All workers shut down") in c.logger.lines, c.logger.lines
    return ("stop first, then token-bound reap of 3 rigs and 3 local procs (ssh error, failed kill "
            f"recorded, no raise) with {' and '.join(res)}; INCOMPLETE logged; clean reap -> all-clear")


def _stub_bin(d, name, body):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write("#!/bin/bash\n" + body + "\n")
    os.chmod(p, 0o755)


def _dummy(token, life):
    return subprocess.Popen(["bash", "-c", f'exec -a "pwc_worker_service {PWC.PWC_LAUNCH_TOKEN_FLAG} '
                                           f'{token}" sleep {life}'], start_new_session=True)


def _reap_local(token, grace=3, path_prefix=None):
    env = dict(os.environ)
    if path_prefix:
        env["PATH"] = path_prefix + os.pathsep + env["PATH"]
    r = subprocess.run(["bash", "-c", PWC.pwc_shutdown_reap_command(grace_s=grace, token=token)],
                       capture_output=True, text=True, timeout=60, env=env)
    return PWC.classify_shutdown_reap(r.returncode, r.stdout), r


def c3():
    if shutil.which("bash") is None or shutil.which("pgrep") is None or shutil.which("pkill") is None:
        return f"{UNAVAILABLE}: bash/pgrep/pkill not available on this host"
    res = {}
    stubs = os.path.join(OUT, "stubs")
    _stub_bin(os.path.join(stubs, "nokill"), "pkill", "exit 0")          # pkill does nothing
    _stub_bin(os.path.join(stubs, "noprobe"), "pgrep", "exit 2")         # pgrep cannot run
    for tag, life, prefix, want in (("self_exit", "1", None, "exited"),
                                    ("stays", "60", None, "killed"),
                                    ("survivor", "60", os.path.join(stubs, "nokill"), "survivor"),
                                    ("noprobe", "60", os.path.join(stubs, "noprobe"), "unavailable")):
        tok = uuid.uuid4().hex                     # fresh: matches nothing but this dummy
        d = _dummy(tok, life)
        time.sleep(0.3)
        try:
            got, r = _reap_local(tok, path_prefix=prefix)
            res[tag] = got
            assert got == want, (tag, got, want, r.returncode, r.stdout, r.stderr)
        finally:
            if d.poll() is None:
                d.kill()
            d.wait()
    return f"real bash: {res}"


def c3_token():
    mine, other = uuid.uuid4().hex, uuid.uuid4().hex
    d_other = _dummy(other, "60")
    time.sleep(0.3)
    try:
        got, r = _reap_local(mine, grace=1)
        assert got == "exited", (got, r.stdout)              # nothing of MINE was running
        assert d_other.poll() is None, "another coordinator's worker was killed"
    finally:
        d_other.kill()
        d_other.wait()
    return "a worker with another coordinator's token survives this coordinator's reap"


def c4():
    tok = uuid.uuid4().hex
    cmd = PWC.pwc_shutdown_reap_command(token=tok)
    assert "sieve_gpu_worker" not in cmd and "range_miner" not in cmd, cmd
    assert f"[p]wc_worker_service.*{PWC.PWC_LAUNCH_TOKEN_FLAG} {tok}" in cmd, cmd
    for bad in ("", "abc", tok.upper(), tok + "'; rm -rf /", "g" * 32):
        try:
            PWC.pwc_token_pattern(bad)
        except ValueError:
            continue
        raise AssertionError(f"token {bad!r} accepted")
    return "token-bound, bracketed, PWC-only pattern; malformed tokens refused"


def c5():
    """_tcp_launch_workers with every external step stubbed: two rigs; rigA's launches all
    lose their acknowledgment (exit 0, no PID line), rigB's stale kill is refused."""
    c = PWC.PersistentWorkerCoordinator.__new__(PWC.PersistentWorkerCoordinator)
    c.logger = _Log()
    c._launch_token = uuid.uuid4().hex
    c._tcp_reap_targets, c._tcp_local_procs, c._effective_pool_by_host = {}, [], {}
    c.worker_pool_size, c.pwc_port, c.pwc_coordinator_address = 2, 5600, "192.168.3.177"
    node = lambda h: types.SimpleNamespace(hostname=h, username="michael", gpu_count=2,   # noqa: E731
                                           python_env="/r/bin/python", script_path="/r/tree")
    c.nodes = [node("rigA"), node("rigB")]
    c._is_localhost = lambda h: False
    c._is_rocm = lambda n: True
    c._probe_live_rocm_gpu_count = lambda n: 2
    c._reset_rig_cupy_cache = lambda ssh, host, pool: True
    c._tcp_wait_online = lambda expected, timeout_s=0: 0
    c._tcp_broadcast_init = lambda: 0
    c._tcp_wait_ready = lambda expected, timeout_s=0: 0
    c._tcp_transport = types.SimpleNamespace(allow_worker_id=lambda i: None,
                                             revoke_worker_id=lambda i: None)
    scripts = []

    def fake_run(argv, **kw):
        cmd = argv[-1]
        host = next((a.split("@")[1] for a in argv if "@" in a), "local")
        if argv[0] == "bash":                                   # VM101 local stale kill
            return types.SimpleNamespace(returncode=0, stdout=PWC.PWC_STALE_KILL_OK + "\n", stderr="")
        if cmd == PWC.pwc_stale_kill_command():
            if host == "rigB":
                return types.SimpleNamespace(returncode=PWC.PWC_KILL_EXIT_SURVIVOR, stdout="", stderr="")
            return types.SimpleNamespace(returncode=0, stdout=PWC.PWC_STALE_KILL_OK + "\n", stderr="")
        if cmd.startswith("cat > "):
            scripts.append((host, kw["input"].decode()))
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if cmd.startswith("bash "):
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")    # ack lost
        raise AssertionError(f"unexpected command {argv}")
    real = subprocess.run
    subprocess.run = fake_run
    try:
        c._tcp_launch_workers()
    finally:
        subprocess.run = real
    assert set(c._tcp_reap_targets) == {"rigA"}, c._tcp_reap_targets
    assert c._effective_pool_by_host.get("rigA") == 0, c._effective_pool_by_host
    assert len(scripts) == 2 and all(h == "rigA" for h, _ in scripts), scripts
    flag = f"{PWC.PWC_LAUNCH_TOKEN_FLAG} {c._launch_token}"
    assert all(flag in sc for _, sc in scripts), "a launch line lacks the token"
    assert any(f"launch token {c._launch_token}" in m for _, m in c.logger.lines)
    return ("rigA: 0 acknowledged launches but still a reap target; both launch lines carry the "
            "token; rigB (stale kill refused) launched nothing and is not a target")


GATES = [
    ("W1 shutdown in main loop -> process exits", w1),
    ("W2 shutdown before init -> process exits", w2),
    ("W3 unplanned loss after READY -> exits after the 5-min window", w3),
    ("W4 no session yet -> keeps retrying (unchanged)", w4),
    ("W-MUTANT-A no planned-shutdown exit -> W1 fails", w_mutant_a),
    ("W-MUTANT-B session counted after main loop -> W3 fails", w_mutant_b),
    ("C1 reap result classification", c1),
    ("C2 shutdown(): stop, token-bound reap, faults recorded", c2),
    ("C3 reap shell on real bash: exited/killed/survivor/unavailable", c3),
    ("C3-TOKEN another coordinator's worker survives", c3_token),
    ("C4 reap pattern PWC-only, token-bound; bad tokens refused", c4),
    ("C5 launch bookkeeping: token on every launch, target from the attempt", c5),
]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    print("VERIFICATION test_pwc_shutdown_reap START", flush=True)
    passed = unavailable = 0
    for name, fn in GATES:
        try:
            out = fn()
            if isinstance(out, str) and out.startswith(UNAVAILABLE):
                print(f"[UNAVAILABLE] {name}: {out}", flush=True)
                unavailable += 1
            else:
                print(f"[PASS] {name}: {out}", flush=True)
                passed += 1
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
    failed = len(GATES) - passed - unavailable
    print(f"\n{passed}/{len(GATES)} PASS, {unavailable} UNAVAILABLE, {failed} FAIL")
    print("RESULT: " + ("FAIL" if failed else "INCOMPLETE" if unavailable else "PASS"))
    print("VERIFICATION test_pwc_shutdown_reap COMPLETE", flush=True)
    return 0 if passed == len(GATES) else 1


if __name__ == "__main__":
    sys.exit(main())
