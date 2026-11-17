#!/usr/bin/env python3
"""rig_fault_recorder.py — OUTSIDE recorder for the rig GPU fault (no worker code change).

Question it serves (Alpha, 2026-11-16): at a fault, did the HOST hand the card a bad memory
picture, or is it the CARD?  Design: two runs on one boot (Run A = first after boot, Run B =
second, no reset), compared point by point and rig by rig (rrig6600b = non-faulting reference).

Runs on VM101 only. Every host access is `ssh root@<host>`.

  plan                       print the host settings `arm` would switch on, and their revert. No ssh.
  arm    --hosts ...         switch the tracefs settings on (vanish on reboot). NEEDS --i-have-go.
  disarm --hosts ...         switch them off and restore the saved clock/buffer size.
  stream --hosts ... --out D one ssh `cat trace_pipe` per host; every line gets a VM101 receipt
                             stamp and is flushed at once (survives a host freeze up to the last
                             line that left the host). nohup it.
  snap   --hosts ... --out D --point TX [--marker]
                             one read-only snapshot per host (allow-listed files only; see
                             SNAP_SCRIPT). --marker also writes one line to trace_marker (only
                             meaningful when armed) so stream and snapshots share a clock anchor.
  auto   --hosts ... --out D --console F --coord-glob G [--netconsole F]
                             tails the run's console + coordinator log and takes T1..T7 (and a
                             FAULT snapshot of every host) at the trigger lines. T0 is manual.
  compare --a DA --b DB      Run A vs Run B, point by point and host by host.

READ-ONLY CONTRACT for `snap`: it reads only the files in ALLOW_* below. It never opens an
amdgpu debugfs file that acts on read (amdgpu_evict_*, amdgpu_gpu_recover, amdgpu_test_ib,
amdgpu_benchmark, amdgpu_preempt_ib), touches registers/SMU (amdgpu_regs*, amdgpu_pm_info,
amdgpu_sensors, amdgpu_smu_*, amdgpu_gfxoff*, amdgpu_vbios, amdgpu_firmware_info, amdgpu_wave,
amdgpu_gpr*, amdgpu_iomem, amdgpu_vram, hwmon) or KFD hang_hws / hqds / mqds, nor KFD
cu_occupancy. G-ALLOW in tests/test_rig_fault_recorder.py pins this.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

SSH = ["ssh", "-o", "BatchMode=yes",            # no -n: `bash -s` reads the script on stdin "-o", "ConnectTimeout=8",
       "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=5"]

# ---------------------------------------------------------------- what `snap` reads
ALLOW_DRI = ("name", "clients", "amdgpu_vm_info", "amdgpu_gem_info", "amdgpu_sa_info",
             "amdgpu_vram_mm", "amdgpu_gtt_mm", "ttm_page_pool")
ALLOW_SYSFS_CARD = ("mem_info_vram_used", "mem_info_vis_vram_used", "mem_info_gtt_used",
                    "mem_info_vram_total", "mem_info_gtt_total")
ALLOW_KFD_DEBUG = ("mem_limit", "rls")
ALLOW_HOST = ("/proc/meminfo", "/proc/vmstat", "/proc/buddyinfo", "/proc/loadavg",
              "/proc/uptime", "/proc/sys/kernel/random/boot_id",
              "/sys/kernel/mm/transparent_hugepage/enabled",
              "/sys/kernel/mm/transparent_hugepage/defrag",
              "/sys/kernel/mm/transparent_hugepage/khugepaged/defrag",
              "/proc/sys/vm/compaction_proactiveness", "/proc/sys/vm/swappiness",
              "/proc/sys/vm/min_free_kbytes", "/proc/sys/vm/overcommit_memory")
WORKER_ENV_RE = r"^(HSA_|ROCR_|HIP_|CUPY_|AMD_|GPU_|HCC_|ROCM_|PYTHONPATH=|LD_LIBRARY_PATH=)"
WORKER_MATCH = "range_miner_worker.py"

SNAP_SCRIPT = r'''
set -u
sec(){ printf '\n@@@ %s\n' "$1"; }
rd(){ sec "FILE $1"; timeout 10 cat "$1" 2>&1 || echo "@@ERR rc=$?"; }
sec "META host=$(hostname) boot_id=$(cat /proc/sys/kernel/random/boot_id) mono_ns=$(awk '{printf "%d", $1*1e9}' /proc/uptime)"
for f in __HOST__; do rd "$f"; done
for d in /sys/kernel/debug/dri/0000:*; do
  b=$(basename "$d"); [ -e "/sys/bus/pci/drivers/amdgpu/$b" ] || continue
  for f in __DRI__; do rd "$d/$f"; done
  for f in __SYSFS__; do rd "/sys/bus/pci/drivers/amdgpu/$b/$f"; done
done
for f in __KFD__; do rd "/sys/kernel/debug/kfd/$f"; done
for p in /sys/class/kfd/kfd/proc/*; do
  [ -d "$p" ] || continue
  find "$p" -type f ! -name 'cu_occupancy*' 2>/dev/null | sort | while read -r f; do rd "$f"; done
done
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null) || continue
  case "$c" in *__MATCH__*) ;; *) continue;; esac
  sec "WORKER pid=$p cmdline=$c"
  rd /proc/$p/status; rd /proc/$p/maps; rd /proc/$p/smaps_rollup
  sec "FILE /proc/$p/environ(filtered)"; tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -E '__ENV__'
done
sec "END mono_ns=$(awk '{printf "%d", $1*1e9}' /proc/uptime)"
'''


def build_snap_script(marker: str | None) -> str:
    s = (SNAP_SCRIPT.replace("__HOST__", " ".join(ALLOW_HOST))
         .replace("__DRI__", " ".join(ALLOW_DRI))
         .replace("__SYSFS__", " ".join(ALLOW_SYSFS_CARD))
         .replace("__KFD__", " ".join(ALLOW_KFD_DEBUG))
         .replace("__MATCH__", WORKER_MATCH)
         .replace("__ENV__", WORKER_ENV_RE))
    if marker:
        safe = re.sub(r"[^A-Za-z0-9_=.:-]", "_", marker)
        s = f"[ -w /sys/kernel/tracing/trace_marker ] && echo 'TFMREC {safe}' > /sys/kernel/tracing/trace_marker\n" + s
    return s


# ---------------------------------------------------------------- tracefs settings (`arm`)
EVENTS_BASE = (
    "amdkfd/kfd_evict_process_worker_start", "amdkfd/kfd_evict_process_worker_end",
    "amdkfd/kfd_restore_process_worker_start", "amdkfd/kfd_restore_process_worker_end",
    "amdkfd/kfd_kgd2kfd_schedule_evict_and_restore_process",
    "amdkfd/kfd_map_memory_to_gpu_start", "amdkfd/kfd_map_memory_to_gpu_end",
    "amdgpu/amdgpu_bo_create", "amdgpu/amdgpu_bo_move",
    "amdgpu/amdgpu_vm_bo_map", "amdgpu/amdgpu_vm_bo_unmap",
    "amdgpu/amdgpu_pasid_allocated", "amdgpu/amdgpu_pasid_freed",
    "amdgpu/amdgpu_iv",
)
EVENTS_PTES = ("amdgpu/amdgpu_vm_update_ptes", "amdgpu/amdgpu_vm_flush")
NEVER_EVENTS = ("amdgpu/amdgpu_device_rreg", "amdgpu/amdgpu_device_wreg",
                "amdgpu/amdgpu_vm_set_ptes", "amdgpu/amdgpu_vm_copy_ptes")
TRACE_CLOCK = "mono"
BUFFER_KB = 8192
STATE = "/run/tfmrec_trace_saved"        # /run is tmpfs: vanishes on reboot


def arm_script(level: str) -> str:
    ev = EVENTS_BASE + (EVENTS_PTES if level == "ptes" else ())
    lines = ["set -e", "T=/sys/kernel/tracing",
             f'[ -e {STATE} ] || {{ echo "clock=$(sed -n \'s/.*\\[\\(.*\\)\\].*/\\1/p\' $T/trace_clock) '
             f'buffer_kb=$(cat $T/buffer_size_kb)" > {STATE}; }}',
             "echo 0 > $T/tracing_on", f"echo {TRACE_CLOCK} > $T/trace_clock",
             f"echo {BUFFER_KB} > $T/buffer_size_kb", "echo > $T/trace"]
    lines += [f"echo 1 > $T/events/{e}/enable" for e in ev]
    lines += ["echo 1 > $T/tracing_on", "echo 'TFMREC ARMED' > $T/trace_marker",
              "echo ARMED: $(grep -c . $T/set_event) events; cat $T/set_event"]
    return "\n".join(lines)


def disarm_script() -> str:
    ev = EVENTS_BASE + EVENTS_PTES
    lines = ["T=/sys/kernel/tracing", "echo 0 > $T/tracing_on"]
    lines += [f"echo 0 > $T/events/{e}/enable 2>/dev/null" for e in ev]
    # The saved file is parsed, never sourced: an unexpanded buffer reads "7 (expanded: 1408)",
    # which is not shell syntax. Restore the expanded size (what the buffer becomes on first use).
    lines += [f'if [ -e {STATE} ]; then '
              f'clock=$(sed -n "s/.*clock=\\([a-z_0-9-]*\\).*/\\1/p" {STATE}); '
              f'kb=$(sed -n "s/.*expanded: \\([0-9]*\\).*/\\1/p" {STATE}); '
              f'[ -n "$kb" ] || kb=$(sed -n "s/.*buffer_kb=\\([0-9]*\\).*/\\1/p" {STATE}); '
              f'[ -n "$clock" ] && echo "$clock" > $T/trace_clock; '
              f'[ -n "$kb" ] && echo "$kb" > $T/buffer_size_kb; rm -f {STATE}; fi',
              "echo > $T/trace", "echo DISARMED: $(grep -c . $T/set_event) events on, "
              "clock=$(cat $T/trace_clock) buffer_kb=$(cat $T/buffer_size_kb) tracing_on=$(cat $T/tracing_on)"]
    return "\n".join(lines)


def plan_text(level: str) -> str:
    ev = EVENTS_BASE + (EVENTS_PTES if level == "ptes" else ())
    return "\n".join([
        "HOST SETTINGS `arm` SWITCHES ON (tracefs only; every one vanishes on reboot):",
        f"  /sys/kernel/tracing/trace_clock   -> {TRACE_CLOCK}   (old value saved in {STATE}, tmpfs)",
        f"  /sys/kernel/tracing/buffer_size_kb -> {BUFFER_KB} per CPU",
        "  /sys/kernel/tracing/trace           cleared",
        *[f"  events/{e}/enable -> 1" for e in ev],
        "  /sys/kernel/tracing/tracing_on     -> 1",
        "  /sys/kernel/tracing/trace_marker   one line per snapshot (`snap --marker`)",
        "NEVER enabled: " + ", ".join(NEVER_EVENTS),
        "Not touched: module parameters, dynamic debug, amdgpu debugfs writes, kernel cmdline,",
        "  rig-power-telemetry (paused separately by the owner), amdgpu-powercap-min.",
        "Revert: `disarm` (events off, tracing_on 0, clock + buffer restored, buffer cleared).",
    ])


# ---------------------------------------------------------------- runners
def ssh_bash(host: str, script: str, timeout: float = 60.0) -> tuple[int, str, float, float]:
    t0 = time.time()
    try:
        p = subprocess.run(SSH + [f"root@{host}", "bash -s"], input=script, capture_output=True,
                           text=True, timeout=timeout)
        out, rc = p.stdout + (("\n@@@ STDERR\n" + p.stderr) if p.stderr else ""), p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "@@@ TIMEOUT\n", 124
    return rc, out, t0, time.time()


def do_snap(hosts, out: Path, point: str, marker: bool, label: str = "") -> dict:
    d = out / "snaps" / point
    d.mkdir(parents=True, exist_ok=True)
    res = {}

    def one(h):
        rc, txt, t0, t1 = ssh_bash(h, build_snap_script(f"SNAP {point} vm101_ns={time.time_ns()} {label}"
                                                        if marker else None), timeout=60)
        (d / f"{h}.txt").write_text(txt)
        res[h] = {"rc": rc, "vm101_start": t0, "vm101_end": t1, "bytes": len(txt)}

    th = [threading.Thread(target=one, args=(h,)) for h in hosts]
    [t.start() for t in th]
    [t.join() for t in th]
    meta = {"point": point, "label": label, "marker": marker, "hosts": res}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[snap] {point}: " + ", ".join(f"{h} rc={r['rc']} {r['vm101_end'] - r['vm101_start']:.1f}s "
                                          f"{r['bytes']}B" for h, r in res.items()), flush=True)
    return meta


# The remote reader must not outlive its SSH session. A bare `cat trace_pipe` did (11-16: an orphan held
# .155's trace_pipe for ~30 min, so new readers got EBUSY). The watchdog ends cat when the session's
# shell is re-parented (sshd gone) or stdout breaks.
REMOTE_READER = ("bash -c 'cat /sys/kernel/tracing/trace_pipe & c=$!; cat >/dev/null; kill $c 2>/dev/null'")
# stdin of the remote shell is the SSH channel: when the local reader dies or the connection drops, the
# inner `cat >/dev/null` sees EOF and the trace_pipe reader is killed. Needs ssh WITHOUT -n and a local
# stdin pipe that is kept open.


def do_stream(hosts, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def one(h):
        f = open(out / f"trace_{h}.log", "a", buffering=1)
        while True:
            p = subprocess.Popen(SSH + [f"root@{h}", REMOTE_READER], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            f.write(f"{time.time_ns()}\tTFMREC_STREAM_CONNECTED pid={p.pid}\n")
            for line in p.stdout:
                f.write(f"{time.time_ns()}\t{line.rstrip()}\n")
            rc = p.wait()
            f.write(f"{time.time_ns()}\tTFMREC_STREAM_LOST rc={rc}\n")
            time.sleep(5)          # host frozen or rebooting: retry, never give up silently

    th = [threading.Thread(target=one, args=(h,), daemon=True) for h in hosts]
    [t.start() for t in th]
    while True:
        time.sleep(3600)


# ---------------------------------------------------------------- `auto` trigger machine
RE_T1 = re.compile(r"GATE-12 SENTINEL\s*:\s*PASS")
RE_DISPATCH = re.compile(r"derived_bound stage phase=(\d)\b")
RE_ACTIVE = re.compile(r'"active_count": (\d+).*?"stage_idx": (\d+)')
RE_FAULT = re.compile(r"(\d+\.\d+\.\d+\.\d+): .*amdgpu (\S+): amdgpu: \[gfxhub\] page fault")


class Triggers:
    """Pure state machine (unit-tested): feed lines, get the points to take, in order."""

    def __init__(self, t5_delay: float):
        self.t5_delay = t5_delay
        self.taken: set = set()
        self.active_seen: dict = {}
        self.dispatched: set = set()
        self.t4_time = None
        self.faults: list = []

    def _take(self, p):
        if p in self.taken:
            return []
        self.taken.add(p)
        return [p]

    def feed(self, line: str, now: float) -> list:
        out = []
        if RE_T1.search(line):
            out += self._take("T1")
        m = RE_DISPATCH.search(line)
        if m:
            self.dispatched.add(int(m.group(1)))
        if "ACTIVE_STRIPES" in line:
            m = RE_ACTIVE.search(line)
            if m:
                n, st = int(m.group(1)), int(m.group(2))
                if n > 0:
                    self.active_seen[st] = True
                elif self.active_seen.get(st) and (st + 1) in self.dispatched:
                    p = {0: "T2", 1: "T3", 2: "T4", 3: "T6"}.get(st)
                    if p:
                        got = self._take(p)
                        if got and p == "T4":
                            self.t4_time = now
                        out += got
        m = RE_FAULT.search(line)
        if m:
            self.faults.append((m.group(1), m.group(2)))
            out += self._take("FAULT")
        return out

    def tick(self, now: float) -> list:
        if self.t4_time and "T5" not in self.taken and now - self.t4_time >= self.t5_delay:
            if 4 in self.dispatched:            # never snapshot into the phase-4 dispatch
                self.taken.add("T5")
                return ["T5_SKIPPED_PHASE4_ALREADY_DISPATCHED"]
            return self._take("T5")
        return []


def follow(path_fn, stop):
    """Yield new lines of the file returned by path_fn(). Re-resolved while idle: if a NEWER
    file appears (e.g. an aborted launch left the first evidence file), switch to it."""
    f, cur, idle = None, None, 0
    while not stop.is_set():
        if f is None or idle >= 5:
            idle = 0
            p = path_fn()
            if p and p != cur and os.path.exists(p):
                f, cur = open(p, errors="replace"), p
                yield f"TFMREC_FOLLOW {p}\n"
            elif f is None:
                time.sleep(1)
                continue
        line = f.readline()
        if line:
            idle = 0
            yield line
        else:
            idle += 1
            time.sleep(0.2)


def do_auto(hosts, out: Path, console: str, coord_glob: str, netconsole: str | None,
            t5_delay: float, marker: bool, t7_timeout: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    trig = Triggers(t5_delay)
    log = open(out / "auto_events.log", "a", buffering=1)
    stop = threading.Event()
    lock = threading.Lock()
    nc_start = os.path.getsize(netconsole) if netconsole and os.path.exists(netconsole) else 0

    def fire(points, src):
        for p in points:
            log.write(f"{time.time_ns()}\t{p}\tfrom={src}\n")
            if p.startswith("T5_SKIPPED"):
                continue
            snap_hosts = hosts
            if p == "FAULT":
                log.write(f"{time.time_ns()}\tFAULT_LINES {trig.faults}\n")
            do_snap(snap_hosts, out, p, marker, label=src)
            if p == "T6":
                threading.Thread(target=t7_wait, daemon=True).start()

    def t7_wait():
        t0 = time.time()
        while time.time() - t0 < t7_timeout:
            n = 0
            for h in hosts:
                rc, txt, _, _ = ssh_bash(h, f"ls /proc | grep -E '^[0-9]+$' | while read p; do "
                                             f"tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q '{WORKER_MATCH}' && echo W; done | wc -l",
                                         timeout=20)
                n += int(txt.split()[0]) if rc == 0 and txt.split() else 1
            if n == 0:
                with lock:
                    fire(["T7"], "no-workers")    # snapshot FIRST (11-16 Run D: marking T7 taken before
                    trig.taken.add("T7")          # the snap let the main loop exit and kill it)
                return
            time.sleep(10)
        log.write(f"{time.time_ns()}\tT7_TIMEOUT\n")

    def watch(path_fn, src, start_at_end=False):
        first = True
        for line in follow(path_fn, stop):
            if line.startswith("TFMREC_FOLLOW "):
                log.write(f"{time.time_ns()}\t{src} {line.strip()}\n")
                continue
            with lock:
                pts = trig.feed(line, time.time())
                if pts:
                    fire(pts, src)

    auto_t0 = time.time()

    def coord_path():
        # only a coordinator log CREATED after `auto` started: never a previous run's log
        c = [p for p in glob.glob(coord_glob) if os.stat(p).st_ctime >= auto_t0 - 5]
        c.sort(key=os.path.getmtime)
        return c[-1] if c else None

    def nc_lines():
        f = open(netconsole, errors="replace")
        f.seek(nc_start)
        while not stop.is_set():
            line = f.readline()
            if line:
                with lock:
                    pts = trig.feed(line, time.time())
                    if pts:
                        fire(pts, "netconsole")
            else:
                time.sleep(0.2)

    def console_path():
        # --console is a glob too (e.g. the launch's logs/gate12_*_evidence.txt, where the
        # sentinel gate's PASS line is tee'd); only a file created after `auto` started counts
        if not any(ch in console for ch in "*?["):
            return console                      # an explicit file: taken as given
        c = [p for p in glob.glob(console) if os.stat(p).st_ctime >= auto_t0 - 5]
        c.sort(key=os.path.getmtime)
        return c[-1] if c else None

    ths = [threading.Thread(target=watch, args=(console_path, "console"), daemon=True),
           threading.Thread(target=watch, args=(coord_path, "coordinator"), daemon=True)]
    if netconsole:
        ths.append(threading.Thread(target=nc_lines, daemon=True))
    [t.start() for t in ths]
    print(f"[auto] watching console={console} coord={coord_glob} netconsole={netconsole}", flush=True)
    try:
        while "T7" not in trig.taken:
            with lock:
                pts = trig.tick(time.time())
                if pts:
                    fire(pts, "timer")
            time.sleep(1)
    finally:
        stop.set()
        log.write(f"{time.time_ns()}\tAUTO_END taken={sorted(trig.taken)}\n")


# ---------------------------------------------------------------- parse + compare
def parse_snap(txt: str) -> dict:
    sec, cur, buf = {}, None, []
    for line in txt.splitlines():
        if line.startswith("@@@ "):
            if cur:
                sec[cur] = "\n".join(buf)
            cur, buf = line[4:], []
        else:
            buf.append(line)
    if cur:
        sec[cur] = "\n".join(buf)
    return sec


def summarize(txt: str) -> dict:
    s = parse_snap(txt)
    o: dict = {"cards": {}, "workers": {}}
    for k, v in s.items():
        if not k.startswith("FILE "):
            if k.startswith("WORKER "):
                pid = re.search(r"pid=(\d+)", k).group(1)
                o["workers"].setdefault(pid, {"cmd": k.split("cmdline=", 1)[-1][-120:]})
            continue
        f = k[5:]
        m = re.match(r"/sys/kernel/debug/dri/(0000:[0-9a-f:.]+)/(\S+)", f) or \
            re.match(r"/sys/bus/pci/drivers/amdgpu/(0000:[0-9a-f:.]+)/(\S+)", f)
        if m:
            c = o["cards"].setdefault(m.group(1), {})
            name = m.group(2)
            if name.startswith("mem_info"):
                c[name] = int(v.strip() or 0) if v.strip().isdigit() else v.strip()
            elif name == "amdgpu_vm_info":
                c["vm_pids"] = len(re.findall(r"^pid:", v, re.M))
                c["vm_bo_lines"] = len(re.findall(r"^\s*0x[0-9a-f]+", v, re.M))
                c["vm_info_bytes"] = len(v)
                # per BO-state counts and bytes, summed over the card's VMs
                state = None
                for ln in v.splitlines():
                    mm = re.match(r"^\t(\w+) BOs:", ln)
                    if mm:
                        state = mm.group(1).lower()
                        continue
                    mm = re.match(r"^\t\t0x[0-9a-f]+:\s+(\d+) byte", ln)
                    if mm and state:
                        c[f"vm_{state}_n"] = c.get(f"vm_{state}_n", 0) + 1
                        c[f"vm_{state}_bytes"] = c.get(f"vm_{state}_bytes", 0) + int(mm.group(1))
            elif name == "amdgpu_gem_info":
                c["gem_pids"] = len(re.findall(r"^pid", v, re.M))
                c["gem_bytes"] = len(v)
            elif name == "clients":
                c["clients"] = max(0, len(v.strip().splitlines()) - 1)
            elif name == "amdgpu_gtt_mm":
                mm = re.search(r"total: (\d+), used (\d+) free (\d+)", v)
                if mm:
                    c["gtt_mm_used_pages"] = int(mm.group(2))
            continue
        if f == "/sys/kernel/debug/kfd/mem_limit":
            mm = re.search(r"System mem used (\d+)M out of (\d+)M.*?TTM mem used (\d+)M", v, re.S)
            if mm:
                o["kfd_sys_used_M"], o["kfd_sys_limit_M"], o["kfd_ttm_used_M"] = map(int, mm.groups())
        elif f == "/proc/meminfo":
            for key in ("MemFree", "MemAvailable", "Cached", "SwapFree", "AnonHugePages",
                        "Unevictable", "Mlocked", "Shmem"):
                mm = re.search(rf"^{key}:\s+(\d+)", v, re.M)
                if mm:
                    o[f"mem_{key}_kB"] = int(mm.group(1))
        elif f == "/proc/vmstat":
            for key in ("compact_stall", "compact_fail", "compact_success", "allocstall_normal",
                        "allocstall_movable", "pgscan_direct", "pgscan_kswapd", "pgsteal_direct",
                        "thp_fault_fallback", "thp_collapse_alloc", "pswpout", "oom_kill"):
                mm = re.search(rf"^{key} (\d+)", v, re.M)
                if mm:
                    o[f"vm_{key}"] = int(mm.group(1))
        elif f.startswith("/sys/kernel/mm/transparent_hugepage/"):
            o["thp_" + f[len("/sys/kernel/mm/transparent_hugepage/"):].replace("/", "_")] = v.strip()
        elif f.startswith("/sys/class/kfd/kfd/proc/"):
            rel = f[len("/sys/class/kfd/kfd/proc/"):]
            o.setdefault("kfd_proc_files", {})[rel] = v.strip()[:80]
            leaf = rel.rsplit("/", 1)[-1]
            val = v.strip().split()[0] if v.strip() else ""
            if val.lstrip("-").isdigit() and leaf in ("evicted_ms", "faults", "page_in", "page_out"):
                o[f"kfdproc_sum_{leaf}"] = o.get(f"kfdproc_sum_{leaf}", 0) + int(val)
            if leaf.startswith("vram_") and val.isdigit():
                o["kfdproc_sum_vram"] = o.get("kfdproc_sum_vram", 0) + int(val)
        else:
            m = re.match(r"/proc/(\d+)/(maps|status|smaps_rollup)", f)
            if m:
                w = o["workers"].setdefault(m.group(1), {})
                if m.group(2) == "maps":
                    w["maps_lines"] = len(v.splitlines())
                    w["maps_kfd"] = v.count("/dev/kfd")
                    w["maps_render"] = len(re.findall(r"/dev/dri/renderD\d+", v))
                elif m.group(2) == "status":
                    mm = re.search(r"VmRSS:\s+(\d+)", v)
                    w["VmRSS_kB"] = int(mm.group(1)) if mm else None
                    mm = re.search(r"NSpid:\s+(.*)", v)
                    w["NSpid"] = mm.group(1).strip() if mm else None
    return o


def do_compare(a: Path, b: Path) -> None:
    pts = sorted({p.name for p in (a / "snaps").iterdir()} | {p.name for p in (b / "snaps").iterdir()})
    for p in pts:
        hosts = sorted({f.stem for d in (a, b) for f in (d / "snaps" / p).glob("*.txt")})
        for h in hosts:
            sa = summarize((a / "snaps" / p / f"{h}.txt").read_text()) if (a / "snaps" / p / f"{h}.txt").exists() else {}
            sb = summarize((b / "snaps" / p / f"{h}.txt").read_text()) if (b / "snaps" / p / f"{h}.txt").exists() else {}
            keys = sorted(k for k in set(sa) | set(sb) if k not in ("cards", "workers", "kfd_proc_files"))
            print(f"\n=== {p} {h}")
            for k in keys:
                if sa.get(k) != sb.get(k):
                    print(f"  {k:28s} A={sa.get(k)!s:>14}  B={sb.get(k)!s:>14}")
            for c in sorted(set(sa.get("cards", {})) | set(sb.get("cards", {}))):
                ca, cb = sa.get("cards", {}).get(c, {}), sb.get("cards", {}).get(c, {})
                d = {k: (ca.get(k), cb.get(k)) for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
                if d:
                    print(f"  card {c}: " + "; ".join(f"{k} A={x} B={y}" for k, (x, y) in sorted(d.items())))
            print(f"  workers A={len(sa.get('workers', {}))} B={len(sb.get('workers', {}))}")


def trace_counts(path: Path) -> dict:
    """Event counts per type between consecutive TFMREC markers (for A/B by interval)."""
    out, cur = {}, "start"
    for line in open(path, errors="replace"):
        if "TFMREC SNAP" in line:
            # the marker is sanitised on write: "TFMREC SNAP_T0_vm101_ns=..._"
            m = re.search(r"TFMREC SNAP[ _]([A-Za-z0-9]+)", line)
            cur = m.group(1) if m else cur
            continue
        m = re.search(r" (amdgpu_\w+|kfd_\w+):", line)
        if m:
            out.setdefault(cur, {}).setdefault(m.group(1), 0)
            out[cur][m.group(1)] += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("plan", "arm", "disarm", "stream", "snap", "auto", "compare", "counts"):
        s = sub.add_parser(n)
        if n not in ("plan", "compare", "counts"):
            s.add_argument("--hosts", nargs="+", required=True)
        if n in ("plan", "arm"):
            s.add_argument("--level", choices=("base", "ptes"), default="ptes")
        if n == "arm":
            s.add_argument("--i-have-go", action="store_true")
        if n in ("stream", "snap", "auto"):
            s.add_argument("--out", required=True)
        if n == "snap":
            s.add_argument("--point", required=True)
            s.add_argument("--marker", action="store_true")
        if n == "auto":
            s.add_argument("--console", required=True)
            s.add_argument("--coord-glob", required=True)
            s.add_argument("--netconsole", default=None)
            s.add_argument("--t5-delay", type=float, default=45.0)
            s.add_argument("--t7-timeout", type=float, default=900.0)
            s.add_argument("--marker", action="store_true")
        if n == "compare":
            s.add_argument("--a", required=True)
            s.add_argument("--b", required=True)
        if n == "counts":
            s.add_argument("trace")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        print(plan_text(a.level))
    elif a.cmd == "arm":
        if not a.i_have_go:
            print(plan_text(a.level))
            print("\nREFUSED: pass --i-have-go only after Alpha's go for these settings.", file=sys.stderr)
            return 2
        for h in a.hosts:
            rc, out, _, _ = ssh_bash(h, arm_script(a.level))
            print(f"== {h} rc={rc}\n{out}")
    elif a.cmd == "disarm":
        for h in a.hosts:
            rc, out, _, _ = ssh_bash(h, disarm_script())
            print(f"== {h} rc={rc}\n{out}")
    elif a.cmd == "stream":
        do_stream(a.hosts, Path(a.out))
    elif a.cmd == "snap":
        do_snap(a.hosts, Path(a.out), a.point, a.marker)
    elif a.cmd == "auto":
        do_auto(a.hosts, Path(a.out), a.console, a.coord_glob, a.netconsole, a.t5_delay,
                a.marker, a.t7_timeout)
    elif a.cmd == "compare":
        do_compare(Path(a.a), Path(a.b))
    elif a.cmd == "counts":
        print(json.dumps(trace_counts(Path(a.trace)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
