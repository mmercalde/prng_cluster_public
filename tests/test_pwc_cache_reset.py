#!/usr/bin/env python3
"""test_pwc_cache_reset.py: PWC fresh CuPy cache per run on each rig ([PWC-CACHE-RESET]).

Run from the repository root:  python3 -u tests/test_pwc_cache_reset.py
CPU only. No ssh, no rig, no real /tmp cache, no pkill: the remote command runs under `bash -c`
against a scratch copy of its paths in <root>/logs/, and every subprocess call of the two start
paths is replaced by a recorder.
"""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import persistent_worker_coordinator as PWC  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


ORIG_CMD = PWC.pwc_cache_reset_command
PREFIX = "/tmp/cupy_cache_gpu_"


def rooted(scratch):
    return lambda pool: ORIG_CMD(pool).replace(PREFIX, scratch + "/cupy_cache_gpu_")


def run_cmd(scratch, pool):
    return subprocess.run(["bash", "-c", rooted(scratch)(pool)], capture_output=True, text=True)


def entries(p):
    return sorted(os.listdir(p)) if os.path.isdir(p) else None


def main():
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="pwc_cache_reset_", dir=os.path.join(ROOT, "logs"))
    print(f"scratch: {scratch}", flush=True)
    try:
        # ── C: the remote command itself ───────────────────────────────────────────────────
        check("C0 command names exactly the worker cache path, once per loop",
              ORIG_CMD(8).count(PREFIX) == 1 and "for g in 0 1 2 3 4 5 6 7;" in ORIG_CMD(8))
        for g in range(4):
            d = os.path.join(scratch, f"cupy_cache_gpu_{g}")
            os.makedirs(os.path.join(d, "sub"))
            open(os.path.join(d, "kern.cubin"), "w").close()
            open(os.path.join(d, "sub", "x"), "w").close()
        r = run_cmd(scratch, 3)
        check("C1 warm caches 0..2 emptied, rc 0 and OK marker",
              r.returncode == 0 and r.stdout.split() == [PWC.PWC_CACHE_RESET_OK]
              and all(entries(os.path.join(scratch, f"cupy_cache_gpu_{g}")) == [] for g in range(3)),
              f"rc={r.returncode} out={r.stdout.strip()!r}")
        check("C2 folder outside the pool (gpu_3) untouched",
              entries(os.path.join(scratch, "cupy_cache_gpu_3")) == ["kern.cubin", "sub"])
        shutil.rmtree(os.path.join(scratch, "cupy_cache_gpu_0"))
        r = run_cmd(scratch, 3)
        check("C3 missing folder (fresh boot) is created empty, rc 0",
              r.returncode == 0 and entries(os.path.join(scratch, "cupy_cache_gpu_0")) == [])
        locked = os.path.join(scratch, "cupy_cache_gpu_1", "locked")
        os.makedirs(locked)
        open(os.path.join(locked, "f"), "w").close()
        os.chmod(locked, stat.S_IRUSR | stat.S_IXUSR)
        r = run_cmd(scratch, 3)
        os.chmod(locked, stat.S_IRWXU)
        check("C4 wipe that cannot delete: rc 3, no OK marker",
              r.returncode == 3 and PWC.PWC_CACHE_RESET_OK not in r.stdout, f"rc={r.returncode}")
        shutil.rmtree(os.path.join(scratch, "cupy_cache_gpu_1"))
        ro = os.path.join(scratch, "ro")
        os.makedirs(ro)
        os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
        r = subprocess.run(["bash", "-c", ORIG_CMD(1).replace(PREFIX, ro + "/cupy_cache_gpu_")],
                           capture_output=True, text=True)
        os.chmod(ro, stat.S_IRWXU)
        check("C5 folder cannot be recreated: rc 4, no OK marker",
              r.returncode == 4 and PWC.PWC_CACHE_RESET_OK not in r.stdout, f"rc={r.returncode}")
        cmd = rooted(scratch)(2)
        r = subprocess.run(["bash", "-c", cmd.replace("ls -A --", "false")], capture_output=True, text=True)
        check("C6 emptiness listing fails (injected): rc 5, no OK marker",
              "ls -A --" in cmd and r.returncode == 5 and PWC.PWC_CACHE_RESET_OK not in r.stdout,
              f"rc={r.returncode}")
        r = subprocess.run(["bash", "-c", cmd.replace('n=$(ls -A -- "$d")', 'n=$(echo leftover)')],
                           capture_output=True, text=True)
        check("C7 folder not empty after the wipe (injected): rc 5, no OK marker",
              r.returncode == 5 and PWC.PWC_CACHE_RESET_OK not in r.stdout, f"rc={r.returncode}")

        # ── M: _reset_rig_cupy_cache reads rc AND marker ────────────────────────────────────
        pwc = PWC.PersistentWorkerCoordinator.__new__(PWC.PersistentWorkerCoordinator)
        pwc.logger = PWC.logging.getLogger("test_pwc_cache_reset")
        with mock.patch.object(PWC, "pwc_cache_reset_command", rooted(scratch)):
            ok = pwc._reset_rig_cupy_cache(["bash", "-c"], "stub", 3)
        check("M1 real command through the method: True", ok is True)
        for name, script, want in [
                ("M2 rc 0 but no marker -> refused", "exit 0", False),
                ("M3 marker but rc 5 -> refused", f"echo {PWC.PWC_CACHE_RESET_OK}; exit 5", False),
                ("M4 marker not last line -> refused", f"echo {PWC.PWC_CACHE_RESET_OK}; echo x", False),
                ("M5 marker and rc 0 -> accepted", f"echo {PWC.PWC_CACHE_RESET_OK}", True)]:
            with mock.patch.object(PWC, "pwc_cache_reset_command", lambda pool, s=script: s):
                got = pwc._reset_rig_cupy_cache(["bash", "-c"], "stub", 3)
            check(name, got is want, f"got {got}")
        check("M6 transport cannot start (no such binary) -> refused",
              pwc._reset_rig_cupy_cache([os.path.join(scratch, "no_ssh")], "stub", 3) is False)

        # ── W: wiring in both start paths; one rig refused ──────────────────────────────────
        cfg = {"nodes": [{"hostname": h, "gpu_type": "RX 6600 XT", "gpu_count": 8,
                          "python_env": "/home/michael/rocm_env/bin/python3",
                          "script_path": "/home/michael/distributed_prng_analysis",
                          "username": "michael"}
                         for h in ("192.168.3.122", "192.168.3.156", "192.168.3.164")]}
        cfg_path = os.path.join(scratch, "cfg.json")
        with open(cfg_path, "w") as f:
            PWC.json.dump(cfg, f)
        refused = "192.168.3.156"

        class FakeAdmission:
            """[PWC-STAGE1 A3 level 1] the launch-set API _tcp_launch_workers calls."""
            def __init__(self):
                self.allowed = set()
            def allow_worker_id(self, wid):
                self.allowed.add(wid)
            def revoke_worker_id(self, wid):
                self.allowed.discard(wid)

        def make(transport):
            # [PWC-STAGE1] stage 1 lands the step-1 refusals (PWC-BACKUP S3/S5): a TCP run
            # with rigs needs a dial address, and pool 8 is inside the stable envelope only
            # with chunks <= 25,000. Both are passed explicitly; the transport already was.
            p = PWC.PersistentWorkerCoordinator(config_file=cfg_path, worker_pool_size=8,
                                                seed_cap_amd=25_000, pwc_transport=transport,
                                                pwc_coordinator_address="192.168.3.177")
            p._tcp_transport = FakeAdmission()
            log = []

            def fake_run(args, *a, **k):
                flat = args if isinstance(args, str) else " ".join(args)
                log.append(("run", flat))
                # [PWC-STAGE1 B5] the stale-worker kill is read now: answer it as a
                # healthy rig does; a launch answers with a PID (A3 level 1).
                if PWC.PWC_STALE_KILL_OK in flat:
                    out = PWC.PWC_STALE_KILL_OK + "\n"
                elif "bash /tmp/pwc_tcp_launch_gpu" in flat:
                    out = "PID=4242\n"
                else:
                    out = "none"
                return subprocess.CompletedProcess(args, 0, stdout=out if k.get("text") else out.encode(),
                                                   stderr="" if k.get("text") else b"")

            def fake_reset(ssh_base, host, pool):
                log.append(("reset", host, pool, ssh_base[-1]))
                return host != refused
            p._reset_rig_cupy_cache = fake_reset
            return p, log, fake_run

        # TCP path
        p, log, fake_run = make("tcp")
        waits = []
        p._probe_live_rocm_gpu_count = lambda node: 8
        p._tcp_wait_online = lambda expected, timeout_s=0: waits.append(expected) or expected
        p._tcp_broadcast_init = lambda: 0
        p._tcp_wait_ready = lambda expected, timeout_s=0: expected
        with mock.patch("subprocess.run", fake_run), mock.patch("subprocess.Popen") as popen, \
                mock.patch("time.sleep", lambda s: None):
            p._tcp_launch_workers()
        def first(pred):
            return next((i for i, e in enumerate(log) if pred(e)), None)
        ok_order = True
        for h in ("192.168.3.122", "192.168.3.164"):
            i_kill = first(lambda e: e[0] == "run" and h in e[1] and "pkill" in e[1])
            i_reset = first(lambda e: e[0] == "reset" and e[1] == h)
            i_write = first(lambda e: e[0] == "run" and h in e[1] and "cat >" in e[1])
            ok_order &= None not in (i_kill, i_reset, i_write) and i_kill < i_reset < i_write
        check("W1 TCP: reset runs after the stale-worker kill and before any launch", ok_order)
        check("W2 TCP: refused rig gets no launch script and no launch",
              not any(e[0] == "run" and refused in e[1] and ("cat >" in e[1] or "bash /tmp" in e[1])
                      for e in log))
        check("W3 TCP: online barrier expects only the two admitted rigs (16)", waits == [16], f"{waits}")
        check("W4 TCP: reset asked for pool 8 on all three rigs",
              sorted((e[1], e[2]) for e in log if e[0] == "reset")
              == [(h, 8) for h in ("192.168.3.122", "192.168.3.156", "192.168.3.164")])

        # SSH-PWC path
        p, log, fake_run = make("ssh")
        spawned = []

        def fake_spawn(handle):
            spawned.append((handle.node.hostname, handle.gpu_id))
            handle.alive = True
            return True
        p._spawn_worker = fake_spawn
        with mock.patch("subprocess.run", fake_run), mock.patch("time.sleep", lambda s: None), \
                mock.patch("progress_display.ProgressWriter", create=True):
            p.startup()
        check("W5 SSH-PWC: refused rig spawns nothing; admitted rigs spawn 8 each",
              sorted(set(h for h, _ in spawned)) == ["192.168.3.122", "192.168.3.164"]
              and len(spawned) == 16, f"{len(spawned)} spawns")
        q = [w for w in p.workers if w.node.hostname == refused]
        check("W6 SSH-PWC: refused rig carries 8 quarantined handles (same as a failed spawn)",
              len(q) == 8 and all(w.quarantined for w in q))
        check("W7 SSH-PWC: reset targets michael@<rig> before its first spawn",
              all(e[3] == "michael@" + e[1] for e in log if e[0] == "reset") and len(
                  [e for e in log if e[0] == "reset"]) == 3)
    finally:
        for dp, dn, _ in os.walk(scratch):
            for d in dn:
                os.chmod(os.path.join(dp, d), stat.S_IRWXU)
        shutil.rmtree(scratch, ignore_errors=True)
    n, ok = len(RESULTS), sum(RESULTS)
    print(f"{ok}/{n} PWC cache-reset checks green" if ok == n else f"{n - ok} of {n} FAILED", flush=True)
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
