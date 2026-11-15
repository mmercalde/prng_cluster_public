"""CPU driver for the miner seed-start work: the REAL run_trial_miner -> serve_trial ->
assign_stripes -> scheduler -> dispatch chain, against fake workers over real framed
loopback sockets. No GPU, no rig, no live state: staging and the miner ledger live in a
caller-supplied directory.

This one module produces BOTH sides of the R7 zero-start control: the capture taken on
the unmodified tree, and the gate run on the patched tree. Its sha256 is recorded in the
captured control, and the gate refuses to compare if it changed (an edited driver would
make the comparison meaningless).

`OMIT` means "do not pass seed_start at all" — the only form the pre-change tree accepts.
"""
import hashlib
import json
import os
import socket
import sqlite3
import threading
import time
import traceback

from miner.range_miner_coordinator import Phase5Sink
from miner.range_miner_protocol import (
    RegisterMessage, StripeCompleteMessage, StripeErrorMessage,
    SubStripeResultMessage,
)
from miner.range_miner_worker import (
    MinerFramedSocket, build_substripe_payload_bytes, supported_variants,
)

OMIT = object()

# [L-4 REPAIR, docs/CLAUDE_CODE_INSTRUCTIONS_MINER_ADMISSION_STALL_20261114.md] The labelled
# PUMP NUDGER that used to live here (a timer calling the coordinator's `_pump_deferred`
# every 0.5 s, needed while L-4 could strand staging admission) is REMOVED. Nothing in
# this driver calls into the coordinator's staging machinery.


def driver_sha256():
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class FakeWorker:
    """Register, receive StripeAssign, answer. Speaks the real MinerFramedSocket wire.

    It advertises one seed cap for all four variants (`cap`), so the coordinator's
    expected_substripes is ceil(seed_count / cap); the worker answers with exactly that
    many contiguous sub-stripe results, one survivor at each sub-stripe's first seed.

    behaviour:
      "complete" — answer every assignment;
      "fail"     — a retryable stripe_error for every assignment;
      "silent"   — record the assignment and never answer (lease expiry).
    """

    def __init__(self, host, port, hostname, gpu_id, behaviour, cap):
        self.host, self.port = host, port
        self.hostname, self.gpu_id = hostname, gpu_id
        self.worker_id = f"{hostname}:gpu{gpu_id}"
        self.behaviour, self.cap = behaviour, int(cap)
        self.assigns = []            # every StripeAssign received, as a dict
        self.err = None
        self._stop = threading.Event()
        self._t = None
        self.fs = None

    def connect_register(self):
        sock = socket.create_connection((self.host, self.port))
        self.fs = MinerFramedSocket(sock)
        caps = {"amd": self.cap, "nvidia": self.cap,
                "amd_hybrid": self.cap, "nvidia_hybrid": self.cap}
        self.fs.send_msg(RegisterMessage(
            worker_id=self.worker_id, hostname=self.hostname, gpu_id=self.gpu_id,
            gpu_name="fake", backend="cuda", vram_bytes=12 * 1024 ** 3,
            capabilities={"supported_variants": supported_variants(), "seed_caps": caps}))

    def start(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        try:
            self.fs.sock.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    msg = self.fs.recv_msg()
                except socket.timeout:
                    continue
                except (ConnectionError, ValueError, OSError):
                    break
                if msg.message_type == "stripe_assign":
                    self.assigns.append({
                        "stripe_id": msg.stripe_id, "seed_start": int(msg.seed_start),
                        "seed_count": int(msg.seed_count), "phase": int(msg.phase),
                        "attempt": int(msg.attempt), "family_name": msg.family_name,
                        "prng_type": msg.prng_type, "trial_number": msg.trial_number,
                        "payload": msg.payload, "worker_id": self.worker_id})
                    self._respond(msg)
                elif msg.message_type == "shutdown":
                    break
        except Exception:
            self.err = traceback.format_exc()

    def _respond(self, a):
        if self.behaviour == "silent":
            return
        if self.behaviour == "fail":
            self.fs.send_msg(StripeErrorMessage(
                worker_id=self.worker_id, stripe_id=a.stripe_id, sub_index=0,
                error="driver: synthetic retryable failure", retryable=True))
            return
        self._answer(a)

    def _answer(self, a):
        eff = (a.payload or {}).get("min_match_threshold")
        start, count = int(a.seed_start), int(a.seed_count)
        n_sub = -(-count // self.cap)
        for sub in range(n_sub):
            ss = start + sub * self.cap
            sc = min(self.cap, start + count - ss)
            obj, pb = build_substripe_payload_bytes(a.stripe_id, sub, ss, sc,
                                                    [[ss, 0.9, None, [1]]])
            self.fs.send_msg(SubStripeResultMessage(
                worker_id=self.worker_id, stripe_id=a.stripe_id, sub_index=sub,
                seed_start=ss, seed_count=sc, survivor_count=1, inline=obj,
                size_bytes=len(pb), sha256=hashlib.sha256(pb).hexdigest(),
                effective_threshold=eff))
        self.fs.send_msg(StripeCompleteMessage(
            worker_id=self.worker_id, stripe_id=a.stripe_id, substripes_done=n_sub,
            survivors_total=n_sub, effective_threshold=eff))

    def stop(self):
        self._stop.set()
        try:
            self.fs.close()
        except Exception:
            pass


class ProbeSink(Phase5Sink):
    """A Phase5Sink that accepts everything (no assembly). Records what it was given."""

    def __init__(self):
        self.published, self.commits, self.aborts = [], [], []

    def publish_shard(self, manifest):
        self.published.append(manifest)

    def commit_trial(self, event):
        self.commits.append(event)

    def abort_trial(self, event):
        self.aborts.append(event)


def _normalize(obj, subs):
    if isinstance(obj, str):
        for old, new in subs:
            obj = obj.replace(old, new)
        return obj
    if isinstance(obj, dict):
        return {_normalize(k, subs): _normalize(v, subs) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v, subs) for v in obj]
    return obj


def run_shape(work_dir, *, seed_start=OMIT, total_seeds, stripe_size, test_both_modes,
              behaviours=("complete",) * 4, cap=None, family_name=None, phase=None,
              compute_lease_timeout=300.0, serve_timeout=180.0, extra=None):
    """Drive one trial. Returns a dict of what happened, normalized for comparison."""
    from miner.range_miner_coordinator import run_trial_miner
    os.makedirs(work_dir, exist_ok=True)
    ds = os.path.join(work_dir, "dataset.json")
    with open(ds, "w") as f:
        f.write('[{"draw":1},{"draw":2},{"draw":3}]')
    stg = os.path.join(work_dir, "stg")
    sink = ProbeSink()
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(16)
    port = lsock.getsockname()[1]
    holder = {}
    cap = int(cap or stripe_size)
    # The coordinator QUARANTINES a worker whose advertised caps differ from its central
    # config, so the fake workers' cap and the coordinator's caps are the same value.
    kw = dict(worker_pool_size=len(behaviours), staging_dir=stg,
              seed_cap_nvidia=cap, seed_cap_amd=cap, seed_cap_nvidia_hybrid=cap,
              seed_cap_amd_hybrid=cap,
              phase5_sink=sink,
              listen_sock=lsock, skip_min=0, skip_max=0, window_anchor=0,
              generator_phase=0, window_size=3, serve_timeout=serve_timeout,
              miner_stripe_size=stripe_size,
              compute_lease_timeout=compute_lease_timeout)
    if family_name is not None:
        kw["family_name"] = family_name
        kw["workflow_phase"] = phase
    if seed_start is not OMIT:
        kw["seed_start"] = seed_start
    kw.update(extra or {})

    def _run():
        try:
            holder["result"] = run_trial_miner(
                "driver", None, 1, "java_lcg", [1, 2, 3], total_seeds, 0.25, 0.25,
                test_both_modes, ds, **kw)
        except BaseException as e:   # noqa: BLE001
            holder["exc"] = e
            holder["err"] = traceback.format_exc()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    workers = [FakeWorker("127.0.0.1", port, f"host{i}", 0, b, cap)
               for i, b in enumerate(behaviours)]
    time.sleep(0.05)
    for w in workers:
        try:
            w.connect_register()
            w.start()
        except OSError as e:
            holder.setdefault("connect_errors", []).append(repr(e))
    t.join(timeout=serve_timeout + 60)
    for w in workers:
        w.stop()
    out = {"serve_alive": t.is_alive(), "error": holder.get("err"),
           "exc_type": type(holder["exc"]).__name__ if "exc" in holder else None,
           "connect_errors": holder.get("connect_errors")}
    res = holder.get("result")
    run_id = res.get("run_id") if isinstance(res, dict) else None
    out["run_id"] = run_id
    out["committed"] = res.get("committed") if isinstance(res, dict) else None
    out["state"] = res.get("state") if isinstance(res, dict) else None
    ledger = os.path.join(stg, "miner_ledger.db")
    out["ledger_path"] = ledger if os.path.exists(ledger) else None
    rows = []
    if out["ledger_path"]:
        c = sqlite3.connect(f"file:{ledger}?mode=ro&immutable=1", uri=True)
        try:
            rows = [list(r) for r in c.execute(
                "SELECT stripe_id, seed_start, seed_count, phase, family_name, state "
                "FROM stripes ORDER BY phase, seed_start, stripe_id")]
        finally:
            c.close()
    assigns = [a for w in workers for a in w.assigns]
    subs = [(run_id, "<RUN>")] if run_id else []
    subs.append((work_dir, "<WORK>"))
    out["stripes"] = _normalize(rows, subs)
    out["assigns"] = sorted(
        (_normalize({k: v for k, v in a.items() if k != "worker_id"}, subs) for a in assigns),
        key=lambda a: json.dumps(a, sort_keys=True))
    out["assigns_by_worker"] = {w.worker_id: [(_normalize(a["stripe_id"], subs),
                                              a["seed_start"], a["attempt"])
                                             for a in w.assigns] for w in workers}
    out["worker_errors"] = [w.err for w in workers if w.err]
    out["sink"] = {"published": len(sink.published), "commits": len(sink.commits),
                   "aborts": len(sink.aborts)}
    return out


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
