#!/usr/bin/env python3
"""rig_fault_map.py — map each GCVM fault address against the recorder's trace and snapshots.

Offline, VM101 only, reads files only.

  python3 scripts/rig_fault_map.py --netconsole N --since "2026-11-16 16:46" \
      --stream DIR/stream --run DIR/runB [--last 50]

For the FIRST printed fault per card after --since:
  (a) amdgpu_vm_bo_map / _unmap / vm_update_ptes / bo_create events of the faulting process
      (trace task pid == the 'in process python pid N' of the fault line): every map whose
      [start,last] GPU-page range covers the fault page, and every unmap of such a range;
  (b) the process's last /proc/PID/maps (FAULT snapshot if present, else the latest point):
      the CPU mapping that contains the fault address;
  (c) the T5 tables for that process/card (vm_info carries BO sizes and states, not VAs);
  verdict: NEVER MAPPED / MAPPED THEN UNMAPPED BEFORE THE FAULT / MAPPED AND STILL VALID
  (by VM101 receipt order of trace lines vs the fault line), plus the last N trace events of
  that process and the IVs of its PASID before the fault.
Units: vm_bo_map start/last are GPU page numbers (4 KiB) [driver: mapping->start/last].
"""
import argparse
import glob
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_fault_recorder as rec  # noqa: E402

RE_NC = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) (\d+\.\d+\.\d+\.\d+): \[\s*([\d.]+)\] "
                   r"amdgpu (\S+): amdgpu: (.*)$")
RE_TASK = re.compile(r"^\s*(\S+)-(\d+)\s+\[\d+\]\s+\S+\s+([\d.]+): (\w+): (.*)$")


def faults(netconsole, since):
    """First fault record per (host, card) after `since`: pid, pasid, page, status, vm101 ts."""
    out, cur = {}, {}
    for line in open(netconsole, errors="replace"):
        m = RE_NC.match(line.rstrip())
        if not m or m.group(1) < since:
            continue
        ts, host, kt, card, msg = m.groups()
        key = (host, card)
        if "[gfxhub] page fault" in msg:
            cur[key] = {"host": host, "card": card, "ts": ts, "ktime": kt,
                        "pasid": int(re.search(r"pasid:(\d+)", msg).group(1))}
        elif key in cur and "in process" in msg:
            # "in process <comm> pid <TGID> thread <comm> pid <TID>": keep both (Beta X-GCVM-LOCATED R1 Q1)
            m2 = re.search(r"in process \S+ pid (\d+) thread \S+ pid (\d+)", msg)
            cur[key]["pid"], cur[key]["tid"] = int(m2.group(1)), int(m2.group(2))
        elif key in cur and "in page starting at address" in msg:
            cur[key]["addr"] = int(re.search(r"address (0x[0-9a-f]+)", msg).group(1), 16)
        elif key in cur and "PROTECTION_FAULT_STATUS" in msg:
            cur[key]["status"] = msg.split(":")[-1].strip()
            if key not in out:
                out[key] = dict(cur[key])
    return list(out.values())


def all_faults(netconsole, since):
    """Every printed fault record with a decodable status after `since` (first-per-card is faults())."""
    out, cur = [], {}
    for line in open(netconsole, errors="replace"):
        m = RE_NC.match(line.rstrip())
        if not m or m.group(1) < since:
            continue
        ts, host, kt, card, msg = m.groups()
        key = (host, card)
        if "[gfxhub] page fault" in msg:
            cur[key] = {"host": host, "card": card, "ts": ts, "ktime": kt,
                        "pasid": int(re.search(r"pasid:(\d+)", msg).group(1))}
        elif key in cur and "in process" in msg:
            m2 = re.search(r"in process \S+ pid (\d+) thread \S+ pid (\d+)", msg)
            cur[key]["pid"], cur[key]["tid"] = int(m2.group(1)), int(m2.group(2))
        elif key in cur and "in page starting at address" in msg:
            cur[key]["addr"] = int(re.search(r"address (0x[0-9a-f]+)", msg).group(1), 16)
        elif key in cur and "PROTECTION_FAULT_STATUS" in msg:
            cur[key]["status"] = msg.split(":")[-1].strip()
            if cur[key]["status"] != "0xFFFFFFFF" and "addr" in cur[key]:
                out.append(dict(cur[key]))
    return out


def ts_ns(ts):
    return int(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1e9)


def trace_for(stream, host):
    """Rows (rx_ns, comm, task_id, ktime_s, event, text, dst). `dst` is the continuation line that
    follows an amdgpu_vm_update_ptes record: the card-local physical addresses written (Beta R1 Q6)."""
    rows = []
    for line in open(os.path.join(stream, f"trace_{host}.log"), errors="replace"):
        rx, _, body = line.rstrip("\n").partition("\t")
        m = RE_TASK.match(body)
        if m:
            rows.append([int(rx), m.group(1), int(m.group(2)), float(m.group(3)), m.group(4), m.group(5), None])
        elif body.startswith("{") and rows and rows[-1][4] == "amdgpu_vm_update_ptes" and rows[-1][6] is None:
            rows[-1][6] = [int(x, 16) for x in body.strip("{}").split(",") if x]
    return [tuple(r) for r in rows]


def task_process(rows):
    """task id -> process (TGID), from the explicit `pid:` field of vm_update_ptes and the
    'Process pid =' field of KFD events. A task whose events name more than one non-zero
    process, or none, is left unassigned (never assigned by task name)."""
    seen = {}
    for r in rows:
        m = re.search(r"\bpid:(\d+) ", r[5]) if r[4] == "amdgpu_vm_update_ptes" else \
            re.search(r"Process pid\s*=\s*(\d+)", r[5]) if r[4].startswith("kfd_") else None
        if m and int(m.group(1)) != 0:
            seen.setdefault(r[2], set()).add(int(m.group(1)))
    return {t: next(iter(p)) for t, p in seen.items() if len(p) == 1}


def owner(r, tp):
    """Process owning an event: its explicit pid field when present, else its task's process."""
    if r[4] == "amdgpu_vm_update_ptes":
        m = re.search(r"\bpid:(\d+) ", r[5])
        if m and int(m.group(1)):
            return int(m.group(1))
    return tp.get(r[2])


def maps_hit(run, host, pid, addr):
    pts = ["FAULT", "T6", "T5", "T4", "T3", "T2", "T1"]
    for p in pts:
        f = os.path.join(run, "snaps", p, f"{host}.txt")
        if not os.path.exists(f):
            continue
        s = rec.parse_snap(open(f).read()).get(f"FILE /proc/{pid}/maps")
        if s is None:
            continue
        for ln in s.splitlines():
            m = re.match(r"([0-9a-f]+)-([0-9a-f]+) (\S+) \S+ \S+ \S+\s*(.*)", ln)
            if m and int(m.group(1), 16) <= addr < int(m.group(2), 16):
                return p, ln
        return p, None
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--netconsole", required=True)
    ap.add_argument("--since", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--last", type=int, default=50)
    ap.add_argument("--all-lines", action="store_true", help="every printed fault record, not only the first per card")
    a = ap.parse_args(argv)
    F = faults(a.netconsole, a.since)
    if a.all_lines:
        F = all_faults(a.netconsole, a.since)
    if not F:
        print("no fault lines after", a.since)
        return 0
    cache = {}
    for f in F:
        host, pid, addr = f["host"], f.get("pid"), f.get("addr")
        print(f"\n######## {host} {f['card']} pid={pid} pasid={f['pasid']} addr={addr:#x} "
              f"status={f.get('status')} rx={f['ts']}")
        rows = cache.setdefault(host, trace_for(a.stream, host))
        tp = cache.setdefault(host + "#tp", task_process(rows))
        t_fault = ts_ns(f["ts"])
        page = addr >> 12
        # Process attribution (thread ownership joined with explicit VM pid). Coverage is searched over
        # the WHOLE retained stream, not cut at the VM101 receipt time; order is reported, not assumed.
        mine = [r for r in rows if owner(r, tp) == pid]
        maps, unmaps, ptes = [], [], []
        for r in mine:
            if r[4] in ("amdgpu_vm_bo_map", "amdgpu_vm_bo_unmap"):
                m = re.search(r"start=([0-9a-f]+), last=([0-9a-f]+)", r[5])
                if m and int(m.group(1), 16) <= page <= int(m.group(2), 16):
                    (maps if r[4].endswith("_map") else unmaps).append(r)
            elif r[4] == "amdgpu_vm_update_ptes":
                m = re.search(r"start:0x([0-9a-f]+) end:0x([0-9a-f]+)", r[5])
                if m and int(m.group(1), 16) <= page < int(m.group(2), 16):
                    ptes.append(r)
        print(f"(a) process trace events before fault: {len(mine)}; maps covering page: {len(maps)}; "
              f"unmaps covering page: {len(unmaps)}; PTE updates covering page: {len(ptes)}")
        for r in (maps[-3:] + unmaps[-3:] + ptes[-3:]):
            print("    ", r[4], r[5][:160], "rx_ms_before_fault=%.1f" % ((t_fault - r[0]) / 1e6),
                  ("dst=" + ",".join(hex(x) for x in r[6])) if r[6] else "")
        # Wording per Beta X-GCVM-LOCATED R1 Q2: scoped to the retained capture; no PTE read-back.
        if not maps and not ptes:
            verdict = "NO COVERING MAPPING/PTE EVENT IN THE RETAINED CAPTURE"
        elif unmaps and maps and unmaps[-1][3] > maps[-1][3]:
            verdict = "RECORDED MAPPING, THEN A RECORDED UNMAP OF IT"
        else:
            fl = re.search(r"flags=(\w+)", maps[-1][5]).group(1) if maps else "?"
            verdict = (f"RECORDED MAPPING (flags={fl}), NO RECORDED SUBSEQUENT REVOCATION"
                       + (" — not executable" if fl != "?" and not int(fl, 16) & 0x10 else ""))
        p, ln = maps_hit(a.run, host, pid, addr)
        print(f"(b) /proc/{pid}/maps at {p}: {ln or 'NO CPU MAPPING CONTAINS THE ADDRESS'}")
        t5 = os.path.join(a.run, "snaps", "T5", f"{host}.txt")
        if os.path.exists(t5):
            s = rec.parse_snap(open(t5).read())
            vm = s.get(f"FILE /sys/kernel/debug/dri/{f['card']}/amdgpu_vm_info", "")
            blk = re.search(rf"pid:{pid}\t.*?(?=^pid:|\Z)", vm, re.S | re.M)
            st = re.findall(r"^\t(\w+) BOs:", blk.group(0), re.M) if blk else []
            cnt = {k: len(re.findall(r"^\t\t0x", x, re.M)) for k, x in
                   re.findall(r"^\t(\w+) BOs:\n((?:\t\t.*\n?)*)", blk.group(0), re.M)} if blk else {}
            print(f"(c) T5 vm_info for pid {pid} on {f['card']}: {'present ' + str(cnt) if blk else 'ABSENT'}")
        print(f"VERDICT {f['card']}: {verdict}")
        last = [r for r in rows if (owner(r, tp) == pid or (r[4] == "amdgpu_iv" and f"pasid:{f['pasid']} " in r[5]))
                and r[0] <= t_fault][-a.last:]
        print(f"last {len(last)} trace events of pid {pid} / pasid {f['pasid']} before the fault:")
        for r in last:
            print(f"    -{(t_fault - r[0]) / 1e6:9.1f} ms  {r[4]}: {r[5][:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
