#!/usr/bin/env python3
"""rig_seam_controller.py — VM101 stop controller for ONE seam-probe attempt (any rung).

Brief: docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md (T4.2, revised). Thread X-RIG-GPU-PROBE.

ONE INVOCATION = ONE ATTEMPT. There is no automatic retry, GPU reset, reboot or next attempt.
The owner starts every attempt; after any trigger the controller prints OWNER ACTION REQUIRED and
exits. Rungs: 1 card, 8 cards on one CT, or the fleet — the same controller drives N probes.

Sub-commands
  plan     build a plan file (CPU only): production load_residue_window + the production payload
           builder RangeMinerCoordinator.build_stripe_assign_payload (it reads no instance state,
           range_miner_coordinator.py:9216-9337) + production partition_stripe/select_seed_cap.
  run      run one attempt from a targets file; append one record to <series>/attempts.jsonl.
  summary  F/C/I per arm, E1-rate range F/B..(F+I)/B, exact one-sided 95 % bound over fully
           observed attempts, interleaved-arm Fisher exact p.

One clock: windows and E1/E2 are judged on VM101 receipt time (the netconsole listener's
timestamp column and this controller's own receipt stamps), never on CT/host clocks.

Triggers (any one, first wins):  E1 or E2 netconsole line from a target host; recorder
UNOBSERVED for a target host for >= --e3-grace s (E3 candidate); probe error/refusal;
probe heartbeat silence >= --silence-s; the attempt wall deadline.
On a trigger: (1) WITHHOLD every not-yet-sent command fleet-wide, (2) send stop to all probes,
(3) after --kill-wait s, SIGTERM then SIGKILL the recorded probe PIDs via --kill-template,
(4) report every PID not confirmed gone as unresolved.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

E1_RE = re.compile(r"GCVM_L2_PROTECTION_FAULT|\[gfxhub\] page fault")
E2_RE = re.compile(r"STATUS:0xFFFFFFFF|device inaccessible|device lost from bus|"
                   r"Unable to change power state")
PROOF_RE = re.compile(r"NCPROOF")
NC_LINE_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) (\d+\.\d+\.\d+\.\d+): (.*)$")
BDF_RE = re.compile(r"amdgpu (\d{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d)")

DEFAULT_LAUNCH = ("ssh -n -o BatchMode=yes michael@{ct} "
                  "'cd {repo} && ROCR_VISIBLE_DEVICES={rocr} CUPY_CACHE_DIR=/tmp/cupy_cache_gpu{rocr} "
                  "{extra_env} nohup {python} -B {repo}/scripts/rig_seam_probe.py "
                  "--controller {ctl} --bdf {bdf} --torch-mode {arm} --attempt-id {aid} "
                  "> /tmp/rig_seam_probe_{aid}_{rocr}.log 2>&1 &'")
DEFAULT_KILL = "ssh -n -o BatchMode=yes michael@{ct} 'kill -{sig} {pid}'"
# ALIVE only if the PID exists AND is a rig_seam_probe process (guards PID reuse and PIDs we may
# not signal); GONE otherwise. Anything else (no token, ssh 255, timeout) is UNKNOWN = unresolved.
DEFAULT_ALIVE = ("ssh -n -o BatchMode=yes -o ConnectTimeout=10 michael@{ct} "
                 "'if [ -d /proc/{pid} ] && grep -q rig_seam_probe /proc/{pid}/cmdline 2>/dev/null; "
                 "then echo ALIVE; else echo GONE; fi'")
DEFAULT_HOST_CHECK = ("ssh -n -o BatchMode=yes -o ConnectTimeout=10 root@{host_ip} "
                      "cat /proc/sys/kernel/random/boot_id")
DEFAULT_MAP = ("ssh -n -o BatchMode=yes -o ConnectTimeout=10 michael@{ct} "
               "'for n in $(ls -d /sys/class/kfd/kfd/topology/nodes/* | sort -t/ -k8 -n); do "
               "awk \"/^location_id/{{print \\$2}}\" $n/properties; done'")


# ==========================================================================================
# statistics (no scipy)
# ==========================================================================================
def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def upper_bound_one_sided(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) one-sided upper bound on a rate after k events in n trials."""
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def fisher_one_sided(k_a, n_a, k_b, n_b):
    """P(X >= k_a) under the hypergeometric null: arm A has MORE events than arm B."""
    K, N = k_a + k_b, n_a + n_b
    if N == 0:
        return None
    tot = math.comb(N, K)
    return sum(math.comb(n_a, x) * math.comb(n_b, K - x)
               for x in range(k_a, min(K, n_a) + 1)) / tot


# ==========================================================================================
# plan
# ==========================================================================================
def build_plan(args):
    sys.path.insert(0, str(REPO_ROOT))
    from miner.range_miner_worker import (load_residue_window, partition_stripe,
                                          select_seed_cap, VramCaps)
    from miner.range_miner_coordinator import RangeMinerCoordinator
    import sqlite3
    c = sqlite3.connect(f"file:{args.ledger}?mode=ro&immutable=1", uri=True)
    tc = c.execute("select window_size, window_anchor_val, generator_phase, sessions_json, "
                   "forward_threshold, reverse_threshold, dataset_sha256, prng_base, trial_number "
                   "from trial_context where run_id=?", (args.run_id,)).fetchone()
    if tc is None:
        raise SystemExit(f"no trial_context for {args.run_id}")
    ws, anchor, gphase, sess_json, ft, rt, dsha, base, trial = tc
    sessions = json.loads(sess_json) if sess_json else None
    import hashlib
    got = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    if got != dsha:
        raise SystemExit(f"dataset sha256 {got} != trial's {dsha}: refusing to build a plan")
    residues = load_residue_window(args.dataset, ws, sessions, anchor)
    out = {"v": 1, "run_id": args.run_id, "worker": args.worker, "dataset": args.dataset}
    caps = VramCaps()
    for phase, key in ((3, "3"), (4, "4")):
        row = c.execute("select stripe_id, family_name, seed_start, seed_count from stripes "
                        "where run_id=? and stripe_id=?",
                        (args.run_id, args.stripe3 if phase == 3 else args.stripe4)).fetchone()
        if row is None:
            raise SystemExit(f"stripe for phase {phase} not found")
        sid, fam, s0, n0 = row
        payload = RangeMinerCoordinator.build_stripe_assign_payload(
            None, args.dataset, ws, sessions, anchor, residues, dataset_sha256=dsha,
            phase=phase, forward_threshold=ft, reverse_threshold=rt, generator_phase=gphase)
        cap = select_seed_cap(args.backend, fam, caps)
        subs = partition_stripe(s0, n0, cap)
        out["family" + key] = fam
        out["assign" + key] = {"message_type": "stripe_assign", "stripe_id": sid,
                               "trial_number": trial, "seed_start": s0, "seed_count": n0,
                               "substripes": len(subs), "prng_type": base, "family_name": fam,
                               "phase": phase, "attempt": 0, "payload": payload}
        out["ranges" + key] = [[getattr(x, "seed_start", x[0] if isinstance(x, tuple) else None),
                                getattr(x, "seed_count", x[1] if isinstance(x, tuple) else None)]
                               for x in subs]
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[plan] wrote {args.out}: phase3 {out['family3']} x{len(out['ranges3'])}, "
          f"phase4 {out['family4']} x{len(out['ranges4'])}")
    return 0


# ==========================================================================================
# run
# ==========================================================================================
class Tail:
    """Polls a growing text file from a start offset; never holds more than one line."""
    def __init__(self, path):
        self.path = path
        try:
            self.pos = os.path.getsize(path)
        except OSError:
            self.pos = 0

    def read_new(self):
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
        except OSError:
            return []
        return [ln.decode("utf-8", "replace").replace("\x00", "") for ln in data.splitlines()]


class Attempt:
    def __init__(self, a, targets, plans):
        self.a, self.targets, self.plans = a, targets, plans
        self.t0 = time.time()
        self.deadline = self.t0 + a.deadline_s
        self.events = []                      # (recv_ts, kind, detail)
        self.lock = threading.Lock()
        self.conns = {}                       # key -> socket
        self.hello = {}                       # key -> hello msg
        self.done = {}                        # (key, cmd) -> msg
        self.trace = {}                       # key -> list of (recv_ts, rec)
        self.last_rx = {}                     # key -> recv_ts
        self.trigger = None
        self.withheld = []
        self.sent = []
        self.seam_sent_at = {}
        self.first_launch_exit_seam = {}
        self.ended = {}
        self.finished = False
        self.hello_at = {}                    # key -> recv_ts of hello (heartbeat baseline)
        self.rec_obs = {t["host_ip"]: [] for t in targets}   # host -> OBSERVED recv times
        self.exit_sent_at = None

    # -- bookkeeping ----------------------------------------------------------------------
    def ev(self, kind, **d):
        with self.lock:
            self.events.append({"recv_ts": time.time(), "kind": kind, **d})

    def set_trigger(self, why, **d):
        with self.lock:
            if self.trigger is None:
                self.trigger = {"why": why, "recv_ts": time.time(), **d}
        self.ev("trigger", why=why, **d)

    # -- channels -------------------------------------------------------------------------
    def serve_tcp(self, srv):
        srv.settimeout(0.5)
        while time.time() < self.deadline + 30 and len(self.ended) < len(self.targets):
            try:
                cs, _ = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self._conn, args=(cs,), daemon=True).start()

    def _conn(self, cs):
        buf = b""
        key = None
        cs.settimeout(1.0)
        while True:
            try:
                chunk = cs.recv(65536)
            except socket.timeout:
                if key and key in self.ended:
                    return
                continue
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                ln, buf = buf.split(b"\n", 1)
                msg = json.loads(ln)
                if msg.get("t") == "hello":
                    key = self._key_for(msg)
                    now = time.time()
                    with self.lock:
                        self.conns[key] = cs
                        self.hello[key] = msg
                        self.hello_at[key] = now
                        # The first UDP heartbeat must arrive within --silence-s of hello
                        # (Beta X-RIG-GPU-PROBE R1 Q7.3): the baseline is the hello itself.
                        self.last_rx.setdefault(key, now)
                elif key is not None:
                    with self.lock:
                        self.done[(key, msg.get("cmd") or msg.get("t"))] = msg

    def _key_for(self, rec):
        """A probe is identified by (CT hostname, BDF): BDFs repeat across rigs."""
        for t in self.targets:
            if t["bdf"] == rec.get("bdf") and t["ct_hostname"] == rec.get("host"):
                return t["key"]
        return f"unknown:{rec.get('host')}:{rec.get('bdf')}"

    def serve_udp(self, us):
        us.settimeout(0.5)
        while time.time() < self.deadline + 30 and len(self.ended) < len(self.targets):
            try:
                data, _ = us.recvfrom(65535)
            except socket.timeout:
                continue
            now = time.time()
            try:
                rec = json.loads(data)
            except ValueError:
                continue
            key = self._key_for(rec)
            with self.lock:
                self.trace.setdefault(key, []).append((now, rec))
                self.last_rx[key] = now
                if rec.get("kind") == "end":
                    self.ended[key] = rec
                if (rec.get("kind") == "exit" and rec.get("op") in ("launch", "execute")
                        and key in self.seam_sent_at and key not in self.first_launch_exit_seam):
                    self.first_launch_exit_seam[key] = now
            if rec.get("kind") in ("error", "refused"):
                self.set_trigger("probe_" + rec["kind"], target=key, detail=rec.get("error") or rec.get("reason"))

    def watch(self):
        nc = Tail(self.a.netconsole)
        rec_tail = Tail(self.a.recorder) if self.a.recorder else None
        hosts = {t["host_ip"] for t in self.targets}
        unobs_since = {}
        while not self.finished and time.time() < self.deadline + self.a.post_proof_wait_s + 120:
            for ln in nc.read_new():
                m = NC_LINE_RE.match(ln)
                if not m:
                    continue
                ts = _parse_ts(m.group(1))
                ip, body = m.group(2), m.group(3)
                if ip not in hosts:
                    continue
                bdfm = BDF_RE.search(body)
                bdf = bdfm.group(1) if bdfm else None
                if E1_RE.search(body):
                    self.ev("E1", host_ip=ip, bdf=bdf, nc_ts=ts, line=body[:300])
                    self.set_trigger("E1", host_ip=ip, bdf=bdf, nc_ts=ts)
                if E2_RE.search(body):
                    self.ev("E2", host_ip=ip, bdf=bdf, nc_ts=ts, line=body[:300])
                    self.set_trigger("E2", host_ip=ip, bdf=bdf, nc_ts=ts)
                if PROOF_RE.search(body):
                    self.ev("nc_proof", host_ip=ip, nc_ts=ts)
            if rec_tail:
                for ln in rec_tail.read_new():
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    ip = r.get("host_ip")
                    if ip not in hosts or r.get("kind") != "sample":
                        continue
                    if r.get("status") == "UNOBSERVED":
                        unobs_since.setdefault(ip, time.time())
                        self.ev("recorder_unobserved", host_ip=ip)
                    else:
                        unobs_since.pop(ip, None)
                        with self.lock:
                            self.rec_obs[ip].append(time.time())
                        self.ev("recorder_observed", host_ip=ip)
            now = time.time()
            # Recorder coverage is checked on EVERY pass, including complete file silence
            # (Beta R1 Q7.1): a host whose last OBSERVED sample is older than --rec-stale-s is a
            # lost observation channel; a host UNOBSERVED for --e3-grace is an E3 candidate.
            for ip in hosts:
                last = self.rec_obs[ip][-1] if self.rec_obs[ip] else self.t0
                if now - last >= self.a.rec_stale_s:
                    self.set_trigger("recorder_stale", host_ip=ip, since_s=round(now - last, 2))
                if ip in unobs_since and now - unobs_since[ip] >= self.a.e3_grace:
                    self.set_trigger("E3_candidate", host_ip=ip)
            for t in self.targets:
                k = t["key"]
                if k in self.hello and k not in self.ended:
                    if now - self.last_rx.get(k, self.hello_at.get(k, now)) >= self.a.silence_s:
                        self.set_trigger("probe_silence", target=k)
            if now >= self.deadline and len(self.ended) < len(self.targets):
                self.set_trigger("deadline")
            time.sleep(0.2)

    # -- commands -------------------------------------------------------------------------
    def send(self, key, obj):
        # The withhold decision and the dispatch happen under ONE lock, so a trigger cannot land
        # between the check and the send (Beta R1 Q7.4); set_trigger takes the same lock.
        with self.lock:
            if self.trigger is not None and obj.get("cmd") not in ("stop", "exit"):
                self.withheld.append({"target": key, **obj})
                return False
            cs = self.conns.get(key)
            if cs is None:
                return False
            try:
                cs.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode())
            except OSError:
                return False
            self.sent.append({"recv_ts": time.time(), "target": key, "cmd": obj.get("cmd")})
            if obj.get("cmd") == "seam":
                self.seam_sent_at[key] = time.time()
        return True

    def wait_done(self, cmd, keys, timeout):
        t_end = min(time.time() + timeout, self.deadline)
        while time.time() < t_end:
            if self.trigger is not None:
                return False
            if all((k, cmd) in self.done for k in keys):
                return True
            time.sleep(0.1)
        if self.trigger is None:
            if time.time() >= self.deadline:
                self.set_trigger("deadline", step=cmd)
            else:
                self.set_trigger("step_timeout", step=cmd)
        return False


def _parse_ts(s):
    return time.mktime(time.strptime(s.split(".")[0], "%Y-%m-%d %H:%M:%S")) + float("0." + s.split(".")[1])


def _series_state(att_file):
    """Latest state per attempt id from the append-only attempts.jsonl.
    Returns (rows, blocked) where blocked lists attempts that hold the recovery barrier."""
    rows = []
    if att_file.exists():
        for ln in att_file.read_text().splitlines():
            if ln.strip():
                rows.append(json.loads(ln))
    state = {}
    for r in rows:
        aid = r.get("attempt_id")
        st = state.setdefault(aid, {"barrier": False, "cleared": False, "final": None})
        if r.get("owner_cleared"):
            st["cleared"] = True
            st["barrier"] = False
            continue
        if r.get("status") == "in_progress":
            st["barrier"] = True          # durable: holds until a final record or a clear
            st["cleared"] = False
            continue
        st["final"] = r
        st["barrier"] = bool(r.get("needs_owner_recovery"))
        st["cleared"] = False
    blocked = [aid for aid, st in state.items() if st["barrier"]]
    return rows, state, blocked


def _append(att_file, rec):
    with open(att_file, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _remote(cmd, timeout):
    """Run a check/kill command. Returns (rc, stdout, error) and never raises."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), None
    except subprocess.TimeoutExpired:
        return None, "", f"timeout after {timeout}s"
    except Exception as e:  # pragma: no cover
        return None, "", repr(e)


def _alive(a, t, pid):
    """'ALIVE' | 'GONE' | 'UNKNOWN:<why>' — only an explicit remote token counts (Beta R1 Q7.2)."""
    rc, out, err = _remote(a.alive_template.format(ct=t["ct"], pid=pid), a.remote_timeout_s)
    if err:
        return f"UNKNOWN:{err}"
    tok = out.split()[-1] if out.split() else ""
    if tok in ("ALIVE", "GONE"):
        return tok
    return f"UNKNOWN:rc={rc} out={out[-80:]!r}"


def _shutdown(a, at, targets, launched):
    """Best-effort stop of every launched target; returns the unresolved list. A target is resolved
    only by an explicit GONE. Transport failure, timeout or a missing PID is unresolved."""
    unresolved = []
    for t in targets:
        k = t["key"]
        if k not in launched:
            continue
        pid = at.hello.get(k, {}).get("pid")
        if pid is None:
            unresolved.append({"target": k, "reason": "launched but no hello/PID received"})
            continue
        state = _alive(a, t, pid)
        for sig in ("TERM", "KILL"):
            if state != "ALIVE":
                break
            rc, _, err = _remote(a.kill_template.format(ct=t["ct"], sig=sig, pid=pid), a.remote_timeout_s)
            at.ev("kill", target=k, sig=sig, rc=rc, error=err)
            time.sleep(1.0)
            state = _alive(a, t, pid)
        at.ev("alive_check", target=k, pid=pid, state=state)
        if state != "GONE":
            unresolved.append({"target": k, "pid": pid, "state": state})
    return unresolved


def _recorder_fresh(a, targets):
    """Every target host has an OBSERVED recorder sample within --rec-stale-s (pre-dispatch)."""
    need = {t["host_ip"] for t in targets}
    seen = {}
    try:
        with open(a.recorder, "rb") as f:
            f.seek(max(0, os.path.getsize(a.recorder) - 2_000_000))
            for ln in f.read().decode("utf-8", "replace").splitlines():
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get("kind") == "sample" and r.get("status") == "OBSERVED" and r.get("host_ip") in need:
                    seen[r["host_ip"]] = float(r.get("recv_ts", 0))
    except OSError:
        return False, {}
    now = time.time()
    ok = all(ip in seen and now - seen[ip] <= a.rec_stale_s for ip in need)
    return ok, seen


def run_attempt(a):
    targets = json.loads(Path(a.targets).read_text())
    for t in targets:
        t.setdefault("key", f"{t['ct']}:{t['rocr']}")
        t.setdefault("seam_offset_s", 0.0)
    plans = {t["key"]: json.loads(Path(t["plan"]).read_text()) for t in targets}
    series = Path(a.series)
    series.mkdir(parents=True, exist_ok=True)
    att_file = series / "attempts.jsonl"
    rows, state, blocked = _series_state(att_file)
    if blocked:
        print(f"[ctl] REFUSED: recovery barrier held by attempt(s) {blocked}. Recover the host(s), "
              f"then run `rig_seam_controller.py clear --series {series} --attempt-id <id> ...`.")
        return 2
    aid = a.attempt_id or f"a{len(state) + 1:03d}"
    if aid in state:
        print(f"[ctl] REFUSED: attempt id {aid} already recorded")
        return 2
    # Pre-dispatch prerequisites (Beta R1 Q7.5, Q9): no GPU work is dispatched without them.
    if not _pre_proof_ok(a, targets):
        print(f"[ctl] REFUSED: no NCPROOF from every target host within {a.pre_proof_max_age_s:.0f}s")
        return 2
    fresh, seen = _recorder_fresh(a, targets)
    if not fresh:
        print(f"[ctl] REFUSED: recorder {a.recorder} has no OBSERVED sample within {a.rec_stale_s}s "
              f"for every target host (seen: {seen})")
        return 2

    at = Attempt(a, targets, plans)
    # DURABLE barrier first: an in-progress record that only a final record or `clear` lifts.
    _append(att_file, {"attempt_id": aid, "status": "in_progress", "needs_owner_recovery": True,
                       "arm": a.arm, "rung": a.rung, "t0": at.t0, "targets": [t["key"] for t in targets]})
    launched = set()
    unresolved = []
    aborted = None
    srv = us = None

    def _sig(signum, _f):
        raise KeyboardInterrupt(f"signal {signum}")
    old = {s_: signal.signal(s_, _sig) for s_ in (signal.SIGTERM, signal.SIGINT)}
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((a.bind, a.port))
        srv.listen(64)
        us = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        us.bind((a.bind, a.port))
        for fn, arg in ((at.serve_tcp, srv), (at.serve_udp, us), (at.watch, None)):
            threading.Thread(target=fn, args=(() if arg is None else (arg,)), daemon=True).start()
        ctl = f"{a.advertise}:{a.port}"
        for t in targets:
            if at.trigger is not None:          # no further launches after a trigger
                at.withheld.append({"target": t["key"], "cmd": "launch"})
                continue
            cmd = a.launch_template.format(ct=t["ct"], rocr=t["rocr"], bdf=t["bdf"], arm=a.arm,
                                           aid=aid, ctl=ctl, repo=a.repo, python=a.python,
                                           extra_env=a.extra_env)
            at.ev("launch", target=t["key"], cmd=cmd)
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            launched.add(t["key"])
            time.sleep(a.launch_stagger_s)
        keys = [t["key"] for t in targets]
        t_end = time.time() + a.hello_timeout_s
        while time.time() < t_end and not all(k in at.hello for k in keys) and at.trigger is None:
            time.sleep(0.1)
        if not all(k in at.hello for k in keys):
            at.set_trigger("hello_timeout")
        ok = at.trigger is None
        for step, timeout in (("plan", 30), ("bring_up", a.step_timeout_s), ("warm", a.step_timeout_s)):
            if not ok:
                break
            for k in keys:
                msg = {"cmd": step}
                if step == "plan":
                    msg.update(plans[k])
                at.send(k, msg)
            ok = at.wait_done(step, keys, timeout)
        if ok:
            for k in keys:
                at.send(k, {"cmd": "idle", "seconds": a.idle_s})
            ok = at.wait_done("idle", keys, a.idle_s + 30)
        if ok:
            base = time.time()
            for t in sorted(targets, key=lambda x: x["seam_offset_s"]):
                while time.time() < base + t["seam_offset_s"] and at.trigger is None:
                    time.sleep(0.01)
                at.send(t["key"], {"cmd": "seam", "mode": a.seam_mode})
            ok = at.wait_done("seam", keys, a.step_timeout_s)
            win_end = max([at.first_launch_exit_seam.get(k, time.time()) for k in keys]) + a.window_after_s
            while time.time() < min(win_end, at.deadline) and at.trigger is None:
                time.sleep(0.2)
        at.exit_sent_at = time.time()
        for k in keys:
            at.send(k, {"cmd": "stop" if at.trigger else "exit"})
        t_wait = time.time() + a.kill_wait_s
        while time.time() < t_wait and len(at.ended) < len(launched):
            time.sleep(0.2)
    except KeyboardInterrupt as e:
        aborted = f"interrupted: {e}"
        at.set_trigger("controller_interrupted", detail=str(e))
    except Exception as e:
        aborted = f"{type(e).__name__}: {e}"
        at.set_trigger("controller_error", detail=aborted[:300])
    finally:
        for s_, h in old.items():
            signal.signal(s_, h)
        try:
            unresolved = _shutdown(a, at, targets, launched)
        except BaseException as e:   # never lose the record
            unresolved = [{"target": "*", "reason": f"shutdown failed: {e!r}"}]
        post_proof = False if aborted else _wait_post_proof(a, at, targets)
        at.finished = True
        rec = classify(a, at, targets, aid, True, post_proof, unresolved)
        if aborted:
            rec["class"] = "I"
            rec["aborted"] = aborted
            rec["needs_owner_recovery"] = True
        _append(att_file, rec)
        (series / f"{aid}_events.json").write_text(json.dumps(
            {"events": at.events, "sent": at.sent, "withheld": at.withheld,
             "hello": at.hello, "trace": {k: [r for _, r in v] for k, v in at.trace.items()}}, indent=0))
        for sock_ in (srv, us):
            try:
                sock_ and sock_.close()
            except OSError:
                pass
    print(f"[ctl] attempt {aid} arm={a.arm} class={rec['class']} trigger={rec['trigger']} "
          f"withheld={len(at.withheld)} unresolved={unresolved}")
    if rec["needs_owner_recovery"]:
        print("[ctl] OWNER ACTION REQUIRED: recovery barrier set. Recover the host(s), then "
              f"`rig_seam_controller.py clear --series {series} --attempt-id {aid} --targets {a.targets} "
              f"--netconsole {a.netconsole}` (it re-checks host, CT card map, NCPROOF and probe PIDs).")
    return 0 if rec["class"] == "C" else 1


def _pre_proof_ok(a, targets):
    """An NCPROOF line from every target host within --pre-proof-max-age-s before now."""
    if a.pre_proof_max_age_s <= 0:
        return True
    need = {t["host_ip"] for t in targets}
    seen = set()
    try:
        with open(a.netconsole, "rb") as f:
            f.seek(max(0, os.path.getsize(a.netconsole) - 4_000_000))
            for ln in f.read().decode("utf-8", "replace").splitlines():
                m = NC_LINE_RE.match(ln.replace("\x00", ""))
                if m and PROOF_RE.search(m.group(3)) and m.group(2) in need:
                    if time.time() - _parse_ts(m.group(1)) <= a.pre_proof_max_age_s:
                        seen.add(m.group(2))
    except OSError:
        return False
    return seen == need


def _wait_post_proof(a, at, targets):
    """An NCPROOF from every target host received AFTER the stop/exit was sent. The wait limit is
    measured from the start of this wait (Beta R1 Q7.5), not from the exit dispatch."""
    need = {t["host_ip"] for t in targets}
    accept_from = at.exit_sent_at or time.time()
    wait_start = time.time()
    if a.post_proof_wait_s > 0:
        print(f"[ctl] waiting up to {a.post_proof_wait_s:.0f}s for NCPROOF from {sorted(need)}", flush=True)
    while True:
        got = {e["host_ip"] for e in list(at.events) if e["kind"] == "nc_proof" and e["recv_ts"] >= accept_from}
        if got >= need:
            return True
        if time.time() - wait_start >= a.post_proof_wait_s:
            return False
        time.sleep(0.5)


def _recorder_coverage(a, at, targets, t_end):
    """Per host: max gap between consecutive OBSERVED receipts across [t0, t_end], incl. both ends."""
    out = {}
    for ip in {t["host_ip"] for t in targets}:
        ts = [x for x in at.rec_obs.get(ip, []) if at.t0 <= x <= t_end]
        pts = [at.t0] + ts + [t_end]
        gap = max(b - a_ for a_, b in zip(pts, pts[1:]))
        # Coverage must extend THROUGH the attempt: a sample after the stop/exit was sent proves the
        # channel outlived the declared window (Beta R1 Q7.1).
        after = at.exit_sent_at is not None and any(x >= at.exit_sent_at for x in ts)
        out[ip] = {"samples": len(ts), "max_gap_s": round(gap, 2), "sample_after_exit": after,
                   "ok": bool(ts) and gap <= a.rec_stale_s and after}
    return out


def classify(a, at, targets, aid, pre_proof, post_proof, unresolved):
    keys = [t["key"] for t in targets]
    t_end = time.time()
    e1 = [e for e in at.events if e["kind"] == "E1"]
    e2 = [e for e in at.events if e["kind"] == "E2"]
    e3 = [e for e in at.events if e["kind"] == "trigger" and e.get("why") == "E3_candidate"]
    seam_start = min(at.seam_sent_at.values()) if at.seam_sent_at else None
    fle = at.first_launch_exit_seam.values()
    win_end = (max(fle) + a.window_after_s) if fle else (at.deadline if seam_start else None)
    in_win = [e for e in e1 if seam_start is not None and seam_start - a.window_before_s <= e["nc_ts"] <= win_end]
    before = [e for e in e1 if seam_start is None or e["nc_ts"] < seam_start - a.window_before_s]
    trace_ok = True
    gaps = {}
    for k in keys:
        seqs = sorted(r["seq"] for _, r in at.trace.get(k, []))
        if not seqs or seqs[0] != 1 or seqs != list(range(1, seqs[-1] + 1)):
            trace_ok = False
            gaps[k] = _gaps(seqs)
        end = at.ended.get(k)
        if not end or end.get("rc") != 0:
            trace_ok = False
    cov = _recorder_coverage(a, at, targets, t_end)
    rec_ok = all(v["ok"] for v in cov.values()) and \
        not any(e["kind"] == "recorder_unobserved" for e in at.events)
    complete = (pre_proof and post_proof and trace_ok and rec_ok
                and at.seam_sent_at.keys() >= set(keys) and not unresolved)
    if in_win:
        cls = "F"
    elif before:
        cls = "FW"                       # warm-up / idle reproduction: the seam was NOT observed
    elif complete and at.trigger is None:
        cls = "C"
    else:
        cls = "I"
    return {"attempt_id": aid, "status": "final", "arm": a.arm, "seam_mode": a.seam_mode,
            "rung": a.rung, "class": cls, "trigger": at.trigger, "E1_in_window": in_win,
            "E1_before_seam": before, "E2": bool(e2), "E3_candidate": bool(e3),
            "pre_proof": pre_proof, "post_proof": post_proof, "trace_complete": trace_ok,
            "trace_gaps": gaps, "recorder_ok": rec_ok, "recorder_coverage": cov,
            "withheld": at.withheld, "unresolved": unresolved, "seam_sent_at": at.seam_sent_at,
            "boot_stratum": a.boot_stratum, "hello": {k: {"pid": v.get("pid"), "host": v.get("host")}
                                                       for k, v in at.hello.items()},
            # Every non-clean attempt latches the recovery barrier (Beta R1 Q7.4; Alpha 16:39).
            "needs_owner_recovery": cls != "C",
            "t0": at.t0, "t_end": t_end}


def _gaps(seqs):
    if not seqs:
        return "no records"
    s = set(seqs)
    return [i for i in range(1, seqs[-1] + 1) if i not in s][:50]


# ==========================================================================================
# clear — lift a recovery barrier only after re-checking the rig
# ==========================================================================================
def _kfd_bdfs(text):
    out = []
    for ln in text.splitlines():
        try:
            loc = int(ln.strip())
        except ValueError:
            continue
        if loc == 0:
            continue                      # CPU node
        out.append(f"0000:{loc >> 8:02x}:{(loc >> 3) & 31:02x}.{loc & 7:x}")
    return out


def clear_barrier(a):
    series = Path(a.series)
    att_file = series / "attempts.jsonl"
    rows, state, blocked = _series_state(att_file)
    if a.attempt_id not in blocked:
        print(f"[clear] REFUSED: {a.attempt_id} does not hold the barrier (blocked: {blocked})")
        return 2
    targets = json.loads(Path(a.targets).read_text())
    for t in targets:
        t.setdefault("key", f"{t['ct']}:{t['rocr']}")
    checks = {}
    ok = True
    for ip in sorted({t["host_ip"] for t in targets}):
        rc, out, err = _remote(a.host_check_template.format(host_ip=ip), a.remote_timeout_s)
        good = rc == 0 and not err and len(out.strip()) >= 8
        checks[f"host:{ip}"] = {"ok": good, "boot_id": out.strip()[:64], "error": err, "rc": rc}
        ok &= good
    final = state[a.attempt_id]["final"] or {}
    pids = {k: v.get("pid") for k, v in (final.get("hello") or {}).items()}
    for t in targets:
        rc, out, err = _remote(a.map_template.format(ct=t["ct"]), a.remote_timeout_s)
        bdfs = _kfd_bdfs(out) if rc == 0 and not err else []
        good = len(bdfs) > t["rocr"] and bdfs[t["rocr"]] == t["bdf"]
        checks[f"map:{t['key']}"] = {"ok": good, "bdfs": bdfs, "expected": t["bdf"], "error": err}
        ok &= good
        pid = pids.get(t["key"])
        if pid is not None:
            st = _alive(a, t, pid)
            checks[f"pid:{t['key']}"] = {"ok": st == "GONE", "state": st, "pid": pid}
            ok &= st == "GONE"
    proof = _pre_proof_ok(a, targets)
    checks["ncproof_fresh"] = {"ok": proof, "max_age_s": a.pre_proof_max_age_s}
    ok &= proof
    print(json.dumps(checks, indent=1))
    if not ok:
        print(f"[clear] REFUSED: recovery not confirmed for {a.attempt_id}; barrier stays.")
        return 2
    _append(att_file, {"attempt_id": a.attempt_id, "owner_cleared": True, "cleared_at": time.time(),
                       "checks": checks, "note": a.note})
    print(f"[clear] barrier lifted for {a.attempt_id}")
    return 0


# ==========================================================================================
# summary
# ==========================================================================================
def summarize(path):
    rows, state, blocked = _series_state(Path(path))
    finals = [st["final"] for st in state.values() if st["final"]]
    unfinished = [aid for aid, st in state.items() if st["final"] is None]
    def tally(rs):
        F = sum(r["class"] == "F" for r in rs)
        C = sum(r["class"] == "C" for r in rs)
        I = sum(r["class"] == "I" for r in rs)
        FW = sum(r["class"] == "FW" for r in rs)
        B = len(rs)
        obs = F + C
        # FW stopped BEFORE the seam: its seam outcome is unobserved, so it widens the range
        # like I does (Beta R1 Q7.5); reached_seam excludes it.
        return {"B": B, "F": F, "C": C, "I": I, "FW": FW, "reached_seam": F + C + I,
                "e1_range": [F / B, (F + I + FW) / B] if B else None,
                "upper95_over_observed": upper_bound_one_sided(F, obs) if obs else None,
                "observed_n": obs}
    out = {"unfinished_attempts": unfinished, "barrier_held_by": blocked, "arms": {}, "strata": {}}
    arms = {}
    for r in finals:
        arms.setdefault(r["arm"], []).append(r)
        out["strata"].setdefault(str(r.get("boot_stratum")), {}).setdefault(r["arm"], []).append(r["class"])
    for arm, rs in arms.items():
        out["arms"][arm] = tally(rs)
    if "P" in out["arms"] and "N" in out["arms"]:
        p, n = out["arms"]["P"], out["arms"]["N"]
        out["fisher_P_gt_N_pooled"] = {
            "p": fisher_one_sided(p["F"], p["observed_n"], n["F"], n["observed_n"]),
            "assumes": "independent attempts with a common rate per arm across all strata; "
                       "fully observed attempts only"}
    return out


# ==========================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="sub", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("--ledger", required=True)
    pp.add_argument("--run-id", required=True)
    pp.add_argument("--worker", required=True)
    pp.add_argument("--stripe3", required=True)
    pp.add_argument("--stripe4", required=True)
    pp.add_argument("--dataset", required=True)
    pp.add_argument("--backend", default="rocm")
    pp.add_argument("--out", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("--targets", required=True,
                    help="JSON list of {ct, ct_hostname, rocr, bdf, host_ip, plan[, seam_offset_s]}")
    rp.add_argument("--arm", choices=("P", "P1", "M", "N"), required=True)
    rp.add_argument("--seam-mode", choices=("handover", "resume"), default="handover")
    rp.add_argument("--rung", default="1card")
    rp.add_argument("--series", required=True)
    rp.add_argument("--attempt-id", default=None)
    rp.add_argument("--boot-stratum", default=None, help="host boot id(s) for this attempt")
    rp.add_argument("--netconsole", default=str(REPO_ROOT.parent.parent / "distributed_prng_analysis/logs/netconsole_all_rigs.log"))
    rp.add_argument("--recorder", required=True, help="recorder JSONL (passive mode) to watch; required")
    rp.add_argument("--bind", default="0.0.0.0")
    rp.add_argument("--advertise", default="192.168.3.177")
    rp.add_argument("--port", type=int, default=6670)
    rp.add_argument("--repo", default="/home/michael/distributed_prng_analysis")
    rp.add_argument("--python", default="/home/michael/rocm_env/bin/python")
    rp.add_argument("--extra-env", default="", help="e.g. AMD_LOG_LEVEL=3 (diagnostic variants only)")
    rp.add_argument("--launch-template", default=DEFAULT_LAUNCH)
    rp.add_argument("--kill-template", default=DEFAULT_KILL)
    rp.add_argument("--alive-template", default=DEFAULT_ALIVE)
    rp.add_argument("--launch-stagger-s", type=float, default=3.0)
    rp.add_argument("--hello-timeout-s", type=float, default=180)
    rp.add_argument("--step-timeout-s", type=float, default=600)
    rp.add_argument("--idle-s", type=float, default=200)
    rp.add_argument("--window-before-s", type=float, default=1.0)
    rp.add_argument("--window-after-s", type=float, default=30.0)
    rp.add_argument("--deadline-s", type=float, default=1500)
    rp.add_argument("--silence-s", type=float, default=5.0)
    rp.add_argument("--e3-grace", type=float, default=10.0)
    rp.add_argument("--rec-stale-s", type=float, default=5.0,
                    help="max age of a host's last OBSERVED recorder sample before the channel is lost")
    rp.add_argument("--remote-timeout-s", type=float, default=30.0)
    rp.add_argument("--kill-wait-s", type=float, default=10.0)
    rp.add_argument("--pre-proof-max-age-s", type=float, default=1800)
    rp.add_argument("--post-proof-wait-s", type=float, default=300)
    sp = sub.add_parser("summary")
    sp.add_argument("attempts")
    cp_ = sub.add_parser("clear", help="lift a recovery barrier after re-checking the rig")
    cp_.add_argument("--series", required=True)
    cp_.add_argument("--attempt-id", required=True)
    cp_.add_argument("--targets", required=True)
    cp_.add_argument("--netconsole", default=str(REPO_ROOT.parent.parent / "distributed_prng_analysis/logs/netconsole_all_rigs.log"))
    cp_.add_argument("--pre-proof-max-age-s", type=float, default=600)
    cp_.add_argument("--remote-timeout-s", type=float, default=30.0)
    cp_.add_argument("--host-check-template", default=DEFAULT_HOST_CHECK)
    cp_.add_argument("--map-template", default=DEFAULT_MAP)
    cp_.add_argument("--alive-template", default=DEFAULT_ALIVE)
    cp_.add_argument("--note", default="")
    a = ap.parse_args(argv)
    if a.sub == "plan":
        return build_plan(a)
    if a.sub == "run":
        return run_attempt(a)
    if a.sub == "clear":
        return clear_barrier(a)
    print(json.dumps(summarize(a.attempts), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
