#!/usr/bin/env python3
"""rig_gpu_recorder.py — 1 Hz per-card GPU state recorder for the Proxmox rig hosts.

Runs on VM101. Brief: docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md (T3).

What it does
------------
* Opens ONE persistent SSH session per host (root@<host>) and streams a small, read-only Python
  sampler into it on stdin. The sampler reads sysfs only — no file is written on the host, no
  sysfs value is written, nothing is installed. It never opens /dev/dri or /dev/kfd, so it puts
  no GPU context on any card; the hwmon/sysfs reads are the same class the hosts' own
  rig-power-telemetry.service already performs once a second.
* Each host prints one JSON line per sample (about once a second). VM101 stamps the receive time
  and appends it to a JSONL file under logs/ (ignored by git).
* A host that stops answering is recorded as status "UNOBSERVED" once per second, with the reason
  and the seconds since its last sample. It is NEVER recorded as zeros. Fields that cannot be read
  on a live host are null with the field named in "missing", never 0.
* "OBSERVED" means a sample was RECEIVED — possibly late (see recv_lag_s). It is not a statement
  that the host is alive now; host_ts is the time the sample was taken.
* Also binds UDP 6668 (the destination of the hosts' rig-power-telemetry.service, which had no
  listener) and records every datagram verbatim, with source IP and receive time. That producer
  (/usr/local/sbin/rig-power-telemetry.sh:20-21) substitutes 0 when a read fails, so its values
  are a separate raw channel and must not be read as independent healthy measurements; every
  udp6668 record carries "producer_zero_fill_possible": true.

Output
------
logs/rig_gpu_recorder/rig_gpu_<YYYYmmdd_HHMMSS>.jsonl, one JSON object per line:
  {"kind":"header", ...}                         once, at start
  {"kind":"sample","status":"OBSERVED", ...}     one per host per sample
  {"kind":"sample","status":"UNOBSERVED", ...}   one per host per second while it is silent
  {"kind":"udp6668", ...}                        one per telemetry datagram
  {"kind":"event", ...}                          ssh start/exit, signals, stop
Memory is bounded: every record is written and flushed as it arrives; nothing accumulates.

Stopping
--------
SIGINT/SIGTERM stop it cleanly. The PID is written to logs/rig_gpu_recorder/recorder.pid so it
can be stopped with `kill $(cat logs/rig_gpu_recorder/recorder.pid)` — never with pkill.
A second recorder refuses to start (exit 2): it cannot take the exclusive flock on
recorder.lock, which the running recorder holds for its whole lifetime (the kernel releases it on
any exit, so there is no stale-file takeover). recorder.pid is metadata only.
On stop it checks that every recorder thread (host, stderr-drain, watchdog, UDP) and every local
ssh process actually ended; anything still alive is listed in the final "stopped" event as
"unresolved" and the exit code is 3. This certifies the LOCAL side only; that the remote sampler
exited is a separate check (e.g. `pgrep -af "[p]ython3 -B -I"` on each host).
--duration N stops it after N seconds (used for tests).

Hosts
-----
Default: derived from rig_profiles_config.json — for each non-local node, the Proxmox host is the
CT100 'proxmox' endpoint minus one (CLAUDE.md §3: host = rig + 1, CT100 = host + 1). Each
host's own `hostname` is recorded in every sample so a wrong derivation is visible, not silent.
Override with --host IP (repeatable).
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_LINE = 1 << 20          # 1 MiB cap on one sampler line (a real line is ~5 KB)
DEFAULT_OUT_DIR = REPO_ROOT / "logs" / "rig_gpu_recorder"
PROFILES = REPO_ROOT / "rig_profiles_config.json"

# --------------------------------------------------------------------------------------------
# The remote sampler. Read-only: it only opens sysfs files for reading and writes to stdout.
# Python is run as `python3 -B -I -u -`: it reads this program from stdin; -B disables bytecode
# writes for anything it imports (-I alone does NOT, Beta X-RIG-GPU-FAULT R1 Q5); -I ignores the
# host's cwd, user site and PYTHON* environment.
# --------------------------------------------------------------------------------------------
REMOTE_SAMPLER = r'''
import glob, json, os, socket, sys, time
PERIOD = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

def rd(path):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception as e:
        return None

def cards():
    out = []
    for c in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        if "-" in os.path.basename(c):
            continue
        dev = os.path.realpath(c + "/device")
        if rd(dev + "/vendor") != "0x1002":
            continue
        hw = sorted(glob.glob(dev + "/hwmon/hwmon*"))
        out.append({"drm": os.path.basename(c), "bdf": os.path.basename(dev),
                    "dev": dev, "hw": hw[0] if hw else None})
    out.sort(key=lambda d: d["bdf"])
    for i, d in enumerate(out):
        d["gpu_index"] = i        # bus order == CT KFD node order (verified 2026-11-15)
    return out

def cur_level(txt):
    if not txt:
        return None
    for line in txt.splitlines():
        if line.rstrip().endswith("*"):
            return line.split(":", 1)[1].replace("*", "").strip()
    return None

def num(v, scale=1.0):
    try:
        return float(v) / scale
    except Exception:
        return None

HW = [  # (output field, hwmon file, divisor)
    ("power_w", "power1_average", 1e6), ("power_cap_w", "power1_cap", 1e6),
    ("vddgfx_mv", "in0_input", 1.0), ("sclk_mhz_hwmon", "freq1_input", 1e6),
    ("mclk_mhz_hwmon", "freq2_input", 1e6), ("temp_edge_c", "temp1_input", 1e3),
    ("temp_junction_c", "temp2_input", 1e3), ("temp_mem_c", "temp3_input", 1e3),
    ("fan_rpm", "fan1_input", 1.0), ("pwm", "pwm1", 1.0),
]
DEV = [  # (output field, device file)
    ("busy_pct", "gpu_busy_percent"), ("mem_busy_pct", "mem_busy_percent"),
]

host = socket.gethostname()
cl = cards()
seq = 0
nxt = time.monotonic()
while True:
    seq += 1
    rec = {"host": host, "seq": seq, "host_ts": time.time(), "cards": []}
    for d in cl:
        c = {"gpu_index": d["gpu_index"], "bdf": d["bdf"], "drm": d["drm"]}
        missing = []
        for k, f in DEV:
            v = num(rd(d["dev"] + "/" + f))
            c[k] = v
            if v is None: missing.append(k)
        c["perf_level"] = rd(d["dev"] + "/power_dpm_force_performance_level")
        if c["perf_level"] is None: missing.append("perf_level")
        c["sclk_level"] = cur_level(rd(d["dev"] + "/pp_dpm_sclk"))
        if c["sclk_level"] is None: missing.append("sclk_level")
        c["mclk_level"] = cur_level(rd(d["dev"] + "/pp_dpm_mclk"))
        if c["mclk_level"] is None: missing.append("mclk_level")
        for k, f, s in HW:
            v = num(rd(d["hw"] + "/" + f), s) if d["hw"] else None
            c[k] = v
            if v is None: missing.append(k)
        c["device_present"] = os.path.exists(d["dev"] + "/vendor") and rd(d["dev"] + "/vendor") == "0x1002"
        c["missing"] = missing
        rec["cards"].append(c)
    rec["sample_s"] = round(time.time() - rec["host_ts"], 4)
    sys.stdout.write(json.dumps(rec, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    nxt += PERIOD
    dt = nxt - time.monotonic()
    if dt > 0:
        time.sleep(dt)
    else:
        nxt = time.monotonic()
'''


# Passive sampler (--mode passive): NO SMU-backed or driver-computed GPU reads at all — no hwmon,
# no gpu/mem busy percent, no pp_dpm_* — because a 1 Hz metrics read is itself a recorded SMU
# event source: rrig6600b 0000:19:00.0 logged "SMU: response:0xFFFFFFFF ... TransferTableSmu2Dram"
# x2 at 13:47:28 on 2026-11-15, coinciding with its own telemetry's zero-filled read
# (docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md T4.11 (2)). It reads only /proc/uptime, /proc/loadavg
# and each card's PCI `vendor` attribute (served from the kernel's cached config header) as a
# host-alive heartbeat. Every GPU sensor field is explicitly UNMEASURED, never 0.
REMOTE_SAMPLER_PASSIVE = r'''
import glob, json, os, socket, sys, time
PERIOD = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
def rd(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except Exception:
        return None
devs = []
for c in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
    if "-" in os.path.basename(c):
        continue
    d = os.path.realpath(c + "/device")
    if rd(d + "/vendor") == "0x1002":
        devs.append((os.path.basename(d), os.path.basename(c), d))
devs.sort()
host = socket.gethostname(); seq = 0; nxt = time.monotonic()
while True:
    seq += 1
    rec = {"host": host, "seq": seq, "host_ts": time.time(), "mode": "passive",
           "uptime": rd("/proc/uptime"), "loadavg": rd("/proc/loadavg"),
           "gpu_sensor_fields": "UNMEASURED (passive mode)",
           "cards": [{"gpu_index": i, "bdf": b, "drm": n, "pci_vendor": rd(d + "/vendor")}
                     for i, (b, n, d) in enumerate(devs)]}
    sys.stdout.write(json.dumps(rec, separators=(",", ":")) + "\n"); sys.stdout.flush()
    nxt += PERIOD
    dt = nxt - time.monotonic()
    time.sleep(dt) if dt > 0 else None
'''


class Writer:
    """Single locked append-and-flush JSONL writer. Holds nothing in memory."""

    def __init__(self, path: Path):
        self.path = path
        self._f = open(path, "a", buffering=1)
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def write(self, rec: dict) -> None:
        rec.setdefault("recv_ts", time.time())
        line = json.dumps(rec, separators=(",", ":"))
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()
            key = rec.get("kind", "?") + ":" + str(rec.get("status", ""))
            self.counts[key] = self.counts.get(key, 0) + 1

    def close(self) -> None:
        with self._lock:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()


class HostStream(threading.Thread):
    """One persistent SSH session to one host; reconnects with backoff after it exits."""

    def __init__(self, ip: str, label: str, period: float, w: Writer, stop: threading.Event,
                 ssh_user: str = "root", sampler: str = None):
        super().__init__(daemon=True, name=f"host-{ip}")
        self.ip, self.label, self.period, self.w, self.stop = ip, label, period, w, stop
        self.user = ssh_user
        self.sampler = sampler or REMOTE_SAMPLER
        self.last_sample = None          # monotonic time of last OBSERVED line
        self.state = "connecting"        # connecting | streaming | ssh_exited:<rc> | bad_line
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _ssh(self) -> subprocess.Popen:
        cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
               "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=3",
               f"{self.user}@{self.ip}", f"python3 -B -I -u - {self.period}"]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, bufsize=1)
        p.stdin.write(self.sampler)
        p.stdin.close()
        # Drain stderr continuously, keeping only the last 4 KiB, so a chatty remote can
        # neither fill the pipe (stalling the stream) nor grow memory.
        self._err_tail = ""

        def drain(f=p.stderr):
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                self._err_tail = (self._err_tail + chunk)[-4096:]
        self._err_thread = threading.Thread(target=drain, daemon=True, name=f"err-{self.ip}")
        self._err_thread.start()
        return p

    def run(self) -> None:
        backoff = 1.0
        while not self.stop.is_set():
            with self._lock:
                self.state = "connecting"
            try:
                self.proc = self._ssh()
            except Exception as e:  # pragma: no cover - ssh binary missing
                self.w.write({"kind": "event", "host_ip": self.ip, "event": "ssh_spawn_failed",
                              "error": repr(e)})
                self.stop.wait(backoff)
                continue
            self.w.write({"kind": "event", "host_ip": self.ip, "event": "ssh_started",
                          "pid": self.proc.pid})
            while True:
                line = self.proc.stdout.readline(MAX_LINE)   # bounded: never an unbounded line
                if not line or self.stop.is_set():
                    break
                if not line.endswith("\n"):                  # over-long line: skip remainder
                    while line and not line.endswith("\n"):
                        line = self.proc.stdout.readline(MAX_LINE)
                    with self._lock:
                        self.state = "bad_line"
                    self.w.write({"kind": "event", "host_ip": self.ip, "event": "line_too_long",
                                  "limit": MAX_LINE})
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    with self._lock:
                        self.state = "bad_line"
                    self.w.write({"kind": "event", "host_ip": self.ip, "event": "bad_line",
                                  "raw": line[:500]})
                    continue
                with self._lock:
                    self.last_sample = time.monotonic()
                    self.state = "streaming"
                backoff = 1.0
                now = time.time()
                rec.update({"kind": "sample", "status": "OBSERVED", "host_ip": self.ip,
                            "rig": self.label, "recv_ts": now,
                            # > ~1 s means the line was delivered late (backed-up stream);
                            # host_ts is the true sample time.
                            "recv_lag_s": round(now - rec.get("host_ts", now), 3)})
                self.w.write(rec)
            rc = self.proc.poll()
            if rc is None:
                try:
                    self.proc.terminate()
                    rc = self.proc.wait(timeout=3)
                except Exception:
                    self.proc.kill()
                    rc = self.proc.wait()
            self._err_thread.join(timeout=2)
            err = self._err_tail[-500:]
            with self._lock:
                self.state = f"ssh_exited:{rc}"
            self.w.write({"kind": "event", "host_ip": self.ip, "event": "ssh_exited", "rc": rc,
                          "stderr_tail": err})
            if self.stop.is_set():
                break
            self.stop.wait(backoff)
            backoff = min(backoff * 2, 10.0)

    def status(self):
        with self._lock:
            return self.last_sample, self.state

    def kill(self) -> None:
        p = self.proc
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass

    def unresolved(self) -> list:
        """What is still alive after stop: this thread and/or its ssh process."""
        left = []
        if self.is_alive():
            left.append(f"thread:{self.name}")
        et = getattr(self, "_err_thread", None)
        if et is not None and et.is_alive():
            left.append(f"thread:{et.name}")
        p = self.proc
        if p is not None and p.poll() is None:
            left.append(f"ssh_pid:{p.pid}")
        return left


def watchdog(streams, w: Writer, stop: threading.Event, grace: float) -> None:
    """Once a second, record UNOBSERVED for every host with no sample inside `grace` seconds."""
    t0 = time.monotonic()
    while not stop.wait(1.0):
        now = time.monotonic()
        for s in streams:
            last, state = s.status()
            ref = last if last is not None else t0
            if now - ref > grace:
                w.write({"kind": "sample", "status": "UNOBSERVED", "host_ip": s.ip,
                         "rig": s.label, "reason": f"no_sample_within_{grace}s",
                         "ssh_state": state,
                         "since_last_sample_s": round(now - last, 3) if last is not None else None,
                         "cards": None})


def udp_listener(port: int, w: Writer, stop: threading.Event) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as e:
        w.write({"kind": "event", "event": "udp_bind_failed", "port": port, "error": repr(e)})
        return
    sock.settimeout(0.5)
    w.write({"kind": "event", "event": "udp_listening", "port": port})
    while not stop.is_set():
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        txt = data.decode("utf-8", "replace").rstrip("\n")
        rec = {"kind": "udp6668", "src_ip": addr[0], "raw": txt,
               "producer_zero_fill_possible": True}
        # Parse the rig-power-telemetry.sh line shape; keep raw regardless.
        try:
            parts = txt.split()
            rec["host"] = parts[0]
            for p in parts[1:]:
                k, v = p.split("=", 1)
                if k in ("per_gpu_W", "per_gpu_C"):
                    rec[k] = [int(x) for x in v.split(",") if x != ""]
                elif k == "total_pkg_W":
                    rec[k] = int(v)
                else:
                    rec[k] = v
        except Exception:
            rec["parse_error"] = True
        w.write(rec)
    sock.close()


def live_recorder_pid(pidf: Path):
    """PID of another live recorder named by pidf, else None (a stale file is ignored)."""
    try:
        pid = int(pidf.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    if pid == os.getpid():
        return None
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read()
    except OSError:
        return None
    return pid if b"rig_gpu_recorder" in cmd else None


def derive_hosts() -> list[tuple[str, str]]:
    cfg = json.loads(PROFILES.read_text())
    out = []
    for n in cfg["nodes"]:
        if n.get("local"):
            continue
        ct = ipaddress.ip_address(n["endpoints"]["proxmox"])
        out.append((str(ct - 1), n["node_id"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", action="append", help="Proxmox host IP (repeatable); default derived")
    ap.add_argument("--period", type=float, default=1.0, help="sample period, seconds (default 1)")
    ap.add_argument("--grace", type=float, default=2.5,
                    help="seconds without a sample before a host is UNOBSERVED (default 2.5)")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = run)")
    ap.add_argument("--udp-port", type=int, default=6668, help="telemetry port (0 disables)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--ssh-user", default="root")
    ap.add_argument("--mode", choices=("full", "passive"), default="full",
                    help="passive = no SMU-backed GPU reads (heartbeat + UDP 6668 only); use for g2")
    a = ap.parse_args(argv)

    hosts = [(h, h) for h in a.host] if a.host else derive_hosts()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pidf = a.out_dir / "recorder.pid"
    # Single instance (Beta X-RIG-GPU-FAULT R3 Q5): an exclusive flock on recorder.lock, held
    # by the kernel for this process's whole lifetime (released automatically on any exit).
    # The PID file is metadata only and is never used to decide ownership.
    import fcntl
    lockf = open(a.out_dir / "recorder.lock", "a+")
    try:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            other = pidf.read_text().strip()
        except OSError:
            other = "?"
        print(f"[recorder] REFUSED: another recorder holds {a.out_dir / 'recorder.lock'} (pid {other})",
              flush=True)
        return 2
    pidf.write_text(f"{os.getpid()}\n")
    out = a.out_dir / time.strftime("rig_gpu_%Y%m%d_%H%M%S.jsonl")
    w = Writer(out)
    stop = threading.Event()

    def on_signal(signum, _frame):
        w.write({"kind": "event", "event": "signal", "signum": signum})
        stop.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    w.write({"kind": "header", "version": 1, "pid": os.getpid(), "out": str(out),
             "hosts": [{"ip": h, "rig": l} for h, l in hosts], "period_s": a.period,
             "grace_s": a.grace, "udp_port": a.udp_port, "vm_host": socket.gethostname(),
             "mode": a.mode,
             "host_source": "--host" if a.host else str(PROFILES.relative_to(REPO_ROOT))})
    print(f"[recorder] pid={os.getpid()} out={out}", flush=True)

    sampler = REMOTE_SAMPLER_PASSIVE if a.mode == "passive" else REMOTE_SAMPLER
    streams = [HostStream(h, l, a.period, w, stop, a.ssh_user, sampler) for h, l in hosts]
    for s in streams:
        s.start()
    threads = [threading.Thread(target=watchdog, args=(streams, w, stop, a.grace), daemon=True)]
    if a.udp_port:
        threads.append(threading.Thread(target=udp_listener, args=(a.udp_port, w, stop),
                                        daemon=True))
    for t in threads:
        t.start()
    t_end = time.monotonic() + a.duration if a.duration > 0 else None
    try:
        while not stop.is_set():
            if t_end and time.monotonic() >= t_end:
                stop.set()
                break
            stop.wait(0.5)
    finally:
        for s in streams:
            s.kill()
        for s in streams:
            s.join(timeout=5)
        for t in threads:
            t.join(timeout=2)
        unresolved = [u for s in streams for u in s.unresolved()]
        unresolved += [f"thread:{t.name}" for t in threads if t.is_alive()]
        w.write({"kind": "event", "event": "stopped", "counts": dict(w.counts),
                 "unresolved": unresolved})
        print(f"[recorder] stopped counts={w.counts} unresolved={unresolved}", flush=True)
        w.close()
        try:
            if pidf.read_text().strip() == str(os.getpid()):
                pidf.unlink()
        except FileNotFoundError:
            pass
    return 3 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
