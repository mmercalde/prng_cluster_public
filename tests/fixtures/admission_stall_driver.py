"""CPU driver for the L-4 staging-admission stall work: the REAL run_trial_miner ->
serve_trial -> staging chain against fake workers over real framed loopback sockets, with
the relative order of "a stripe's last staging job finishes" and "its StripeComplete is
processed" FORCED, in either direction, with no pump nudger anywhere.

HOW THE ORDER IS FORCED, AND WHY IT IS NOT A NUDGER. Four OBSERVERS are bound on the
coordinator instance as soon as `build_coordinator` returns (before any traffic):
`_submit_with_slot`, `_run_staging_job`, `_pump_deferred` and `finalize_stripe`. Each calls the real method
exactly once, with the same arguments, whenever the coordinator itself calls it, and only
records that it happened (thread and sequence number). No observer ever initiates a call:
the number of `_pump_deferred` executions is exactly the number the coordinator makes.
The ordering is imposed by WAITING, never by calling:

  staging_first   the worker sends a stripe's sub-stripe results, then holds its
                  StripeComplete until every one of those staging jobs has returned AND
                  the pump in that job's completion callback has returned on the same
                  thread. The StripeComplete is therefore processed after the stripe's
                  last staging job, and after every pump that job's path makes — the
                  exact ordering L-4 names.
  complete_first  the staging job of a stripe's last sub-stripe is held until the serve
                  thread's `finalize_stripe` for that stripe has returned (StripeComplete
                  processed, stripe still waiting for its shard). The last staging job
                  therefore finishes after the StripeComplete is processed.
  free            no forcing.

Every wait is bounded (`force_wait_s`). A wait that times out is recorded as a forcing
FAILURE for that stripe and the run is not credited as forced.

Every database, ledger and staging directory lives under the caller's work_dir.
"""
import contextlib
import hashlib
import json
import os
import socket
import sqlite3
import threading
import time
import traceback

from miner.range_miner_coordinator import Phase5Sink
from miner.range_miner_protocol import (RegisterMessage, StripeCompleteMessage,
                                        SubStripeResultMessage)
from miner.range_miner_worker import (MinerFramedSocket, build_substripe_payload_bytes,
                                      supported_variants)

# SELF-CONTAINED ON PURPOSE: the one-worker control (G-ONE) records this module's sha256
# and nothing else from tests/, so no other fixture's edit can silently change what the
# control compares.


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


def driver_sha256():
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class Orchestrator:
    def __init__(self, order, force_wait_s):
        self.order = order
        self.force_wait_s = float(force_wait_s)
        self.cond = threading.Condition()
        self.seq = 0
        self.coord = None
        self.serve_ident = None
        self.returned = {}            # (stripe_id, attempt, sub) -> (seq, thread ident)
        self.pump_returns = {}        # thread ident -> [seq, ...]
        self.pump_calls = {"serve": 0, "staging": 0, "other": 0}
        self.pump_callers = {}        # immediate caller's function name -> count
        self.submitter = {}           # (stripe_id, attempt, sub) -> submitting thread
        self.serve_finalized = set()  # stripe ids finalized on the serve thread
        self.forced_ok = 0
        self.forced_failed = []

    def _next(self):
        self.seq += 1
        return self.seq

    # ---- observers (bound on the coordinator instance) --------------------
    def install(self, c):
        self.coord = c
        self.serve_ident = threading.get_ident()
        real_job, real_pump, real_fin = (c._run_staging_job, c._pump_deferred,
                                         c.finalize_stripe)
        real_submit = c._submit_with_slot
        O = self

        def submit(kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig):
            with O.cond:
                O.submitter[(stripe_id, attempt, sub_index)] = threading.get_ident()
            return real_submit(kind, wconn, run_id, stripe_id, attempt, sub_index,
                               msg, elig)

        def job(kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig):
            O._before_job(run_id, stripe_id, attempt, sub_index)
            try:
                return real_job(kind, wconn, run_id, stripe_id, attempt, sub_index,
                                msg, elig)
            finally:
                with O.cond:
                    O.returned[(stripe_id, attempt, sub_index)] = (
                        O._next(), threading.get_ident())
                    O.cond.notify_all()

        def pump():
            import sys as _sys
            caller = _sys._getframe(1).f_code.co_name
            try:
                return real_pump()
            finally:
                ident = threading.get_ident()
                name = threading.current_thread().name
                with O.cond:
                    O.pump_returns.setdefault(ident, []).append(O._next())
                    O.pump_callers[caller] = O.pump_callers.get(caller, 0) + 1
                    if ident == O.serve_ident:
                        O.pump_calls["serve"] += 1
                    elif name.startswith("miner-staging"):
                        O.pump_calls["staging"] += 1
                    else:
                        O.pump_calls["other"] += 1
                    O.cond.notify_all()

        def fin(run_id, stripe_id, now=None, eligible_provider=None):
            r = real_fin(run_id, stripe_id, now=now, eligible_provider=eligible_provider)
            if threading.get_ident() == O.serve_ident:
                with O.cond:
                    O.serve_finalized.add(stripe_id)
                    O.cond.notify_all()
            return r

        c._submit_with_slot = submit
        c._run_staging_job = job
        c._pump_deferred = pump
        c.finalize_stripe = fin

    def _before_job(self, run_id, stripe_id, attempt, sub_index):
        if self.order != "complete_first":
            return
        st = self.coord.ledger.get_stripe(run_id, stripe_id) or {}
        n = int(st.get("expected_substripes") or 1)
        if sub_index != n - 1:
            return
        deadline = time.time() + self.force_wait_s
        with self.cond:
            while stripe_id not in self.serve_finalized:
                left = deadline - time.time()
                if left <= 0:
                    self.forced_failed.append((stripe_id, attempt, "hold timed out"))
                    return
                self.cond.wait(left)
            self.forced_ok += 1

    # ---- the staging_first wait, called by the worker ---------------------
    def wait_staged(self, stripe_id, attempt, n_sub):
        deadline = time.time() + self.force_wait_s
        keys = [(stripe_id, attempt, s) for s in range(n_sub)]
        with self.cond:
            while True:
                done = all(k in self.returned for k in keys)
                if done:
                    ok = True
                    for k in keys:
                        # The job's completion callback pumps on the job's thread, or
                        # INLINE on the submitting thread when the job finished before
                        # the callback was attached; either counts.
                        r_seq, ident = self.returned[k]
                        cands = (self.pump_returns.get(ident, [])
                                 + self.pump_returns.get(self.submitter.get(k), []))
                        if not any(p > r_seq for p in cands):
                            ok = False
                            break
                    if ok:
                        self.forced_ok += 1
                        return True
                left = deadline - time.time()
                if left <= 0:
                    self.forced_failed.append((stripe_id, attempt, "staging wait timed out"))
                    return False
                self.cond.wait(min(left, 0.5))


class OrderedWorker:
    """Register, receive StripeAssign, answer over the real MinerFramedSocket wire. Splits
    each stripe by the cap for its family (hybrid or not) and, under staging_first, holds
    StripeComplete until its staging has finished."""

    def __init__(self, host, port, hostname, gpu_id, caps, orch):
        self.host, self.port = host, port
        self.hostname, self.gpu_id = hostname, gpu_id
        self.worker_id = f"{hostname}:gpu{gpu_id}"
        self.caps = dict(caps)
        self.orch = orch
        self.assigns = []
        self.err = None
        self._stop = threading.Event()
        self._t = None
        self.fs = None

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
                    self._answer(msg)
                elif msg.message_type == "shutdown":
                    break
        except Exception:
            self.err = traceback.format_exc()

    def stop(self):
        self._stop.set()
        try:
            self.fs.close()
        except Exception:
            pass

    def connect_register(self):
        self.fs = MinerFramedSocket(socket.create_connection((self.host, self.port)))
        self.fs.send_msg(RegisterMessage(
            worker_id=self.worker_id, hostname=self.hostname, gpu_id=self.gpu_id,
            gpu_name="fake", backend="cuda", vram_bytes=12 * 1024 ** 3,
            capabilities={"supported_variants": supported_variants(),
                          "seed_caps": self.caps}))

    def _answer(self, a):
        cap = int(self.caps["nvidia_hybrid" if "hybrid" in a.family_name else "nvidia"])
        eff = (a.payload or {}).get("min_match_threshold")
        start, count = int(a.seed_start), int(a.seed_count)
        n_sub = -(-count // cap)
        for sub in range(n_sub):
            ss = start + sub * cap
            sc = min(cap, start + count - ss)
            obj, pb = build_substripe_payload_bytes(a.stripe_id, sub, ss, sc,
                                                    [[ss, 0.9, None, [1]]])
            self.fs.send_msg(SubStripeResultMessage(
                worker_id=self.worker_id, stripe_id=a.stripe_id, sub_index=sub,
                seed_start=ss, seed_count=sc, survivor_count=1, inline=obj,
                size_bytes=len(pb), sha256=hashlib.sha256(pb).hexdigest(),
                effective_threshold=eff))
        if self.orch.order == "staging_first":
            self.orch.wait_staged(a.stripe_id, int(a.attempt), n_sub)
        self.fs.send_msg(StripeCompleteMessage(
            worker_id=self.worker_id, stripe_id=a.stripe_id, substripes_done=n_sub,
            survivors_total=n_sub, effective_threshold=eff))


_VOLATILE = ("_at", "path", "elapsed", "lease", "reservation_id", "created",
             "time", "finalized", "registered", "pid")


def ledger_dump(ledger, subs):
    """Every table, every row, with volatile columns (clocks, paths, autoincrement ids)
    dropped and the run id / work dir substituted; rows sorted. Read through an
    immutable connection after the coordinator has returned (no writer is live)."""
    out = {}
    con = sqlite3.connect(f"file:{ledger}?mode=ro&immutable=1", uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        for t in tables:
            if t.startswith("sqlite_"):
                continue
            cur = con.execute(f"SELECT * FROM {t}")
            cols = [d[0] for d in cur.description]
            keep = [i for i, c in enumerate(cols)
                    if not any(v in c for v in _VOLATILE)]
            raw = cur.fetchall()
            if t == "shards":
                # A shard's sha256 covers its payload, which embeds the stripe id, which
                # embeds the random run id — so it differs between otherwise identical
                # runs. It is CHECKED, not dropped: recomputed from the payload this
                # driver sends for that sub-stripe, and replaced by a marker only when
                # it matches (a mismatch keeps the raw value and fails the comparison).
                ix = {c: i for i, c in enumerate(cols)}
                fixed = []
                for r in raw:
                    r = list(r)
                    _o, pb = build_substripe_payload_bytes(
                        r[ix["stripe_id"]], r[ix["sub_index"]], r[ix["seed_start"]],
                        r[ix["seed_count"]], [[r[ix["seed_start"]], 0.9, None, [1]]])
                    if r[ix["sha256"]] == hashlib.sha256(pb).hexdigest():
                        r[ix["sha256"]] = "<sha256:payload-match>"
                    fixed.append(r)
                raw = fixed
            rows = [[_normalize(r[i], subs) for i in keep] for r in raw]
            rows.sort(key=lambda r: json.dumps(r, sort_keys=True, default=str))
            out[t] = {"columns": [cols[i] for i in keep], "rows": rows}
    finally:
        con.close()
    return out


def run_ordered(work_dir, *, n_workers, order, total_seeds, stripe_size,
                test_both_modes, caps, seed_start=0, serve_timeout=120.0,
                force_wait_s=20.0, dump_ledger=False):
    """Drive one trial. Returns a dict of what happened."""
    import miner.range_miner_coordinator as C
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
    lsock.listen(64)
    port = lsock.getsockname()[1]
    orch = Orchestrator(order, force_wait_s)
    kw = dict(worker_pool_size=n_workers, staging_dir=stg,
              seed_cap_nvidia=int(caps["nvidia"]), seed_cap_amd=int(caps["amd"]),
              seed_cap_nvidia_hybrid=int(caps["nvidia_hybrid"]),
              seed_cap_amd_hybrid=int(caps["amd_hybrid"]),
              phase5_sink=sink, listen_sock=lsock, skip_min=0, skip_max=0,
              window_anchor=0, generator_phase=0, window_size=3,
              serve_timeout=serve_timeout, miner_stripe_size=stripe_size,
              compute_lease_timeout=300.0, seed_start=seed_start)
    holder = {}
    real_build = C.build_coordinator

    def capture(**bkw):
        c = real_build(**bkw)
        orch.install(c)
        return c

    def _run():
        try:
            holder["result"] = run_trial_miner(
                "stall", None, 1, "java_lcg", [1, 2, 3], total_seeds, 0.25, 0.25,
                test_both_modes, ds, **kw)
        except BaseException as e:   # noqa: BLE001
            holder["exc"] = e
            holder["err"] = traceback.format_exc()

    C.build_coordinator = capture
    t0 = time.time()
    try:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        workers = [OrderedWorker("127.0.0.1", port, f"host{i}", 0, caps, orch)
                   for i in range(n_workers)]
        time.sleep(0.05)
        for w in workers:
            try:
                w.connect_register()
                w.start()
            except OSError as e:
                holder.setdefault("connect_errors", []).append(repr(e))
        t.join(timeout=serve_timeout + 60)
    finally:
        C.build_coordinator = real_build
    dt = time.time() - t0
    for w in workers:
        w.stop()
    res = holder.get("result") if isinstance(holder.get("result"), dict) else {}
    run_id = res.get("run_id")
    out = {"order": order, "n_workers": n_workers, "seconds": round(dt, 2),
           "serve_alive": t.is_alive(), "error": holder.get("err"),
           "committed": res.get("committed"), "state": res.get("state"),
           "run_id": run_id, "forced_ok": orch.forced_ok,
           "forced_failed": list(orch.forced_failed),
           "pump_calls": dict(orch.pump_calls),
           "pump_callers": dict(orch.pump_callers),
           "worker_errors": [w.err for w in workers if w.err],
           "sink": {"published": len(sink.published), "commits": len(sink.commits),
                    "aborts": len(sink.aborts)}}
    ledger = os.path.join(stg, "miner_ledger.db")
    subs = [(run_id, "<RUN>")] if run_id else []
    subs.append((work_dir, "<WORK>"))
    rows = []
    if os.path.exists(ledger):
        con = sqlite3.connect(f"file:{ledger}?mode=ro&immutable=1", uri=True)
        try:
            rows = [list(r) for r in con.execute(
                "SELECT stripe_id, seed_start, seed_count, phase, family_name, state "
                "FROM stripes ORDER BY phase, seed_start, stripe_id")]
        finally:
            con.close()
        if dump_ledger:
            out["ledger"] = ledger_dump(ledger, subs)
    out["stripes"] = _normalize(rows, subs)
    out["stripe_states"] = {}
    for r in rows:
        out["stripe_states"][r[5]] = out["stripe_states"].get(r[5], 0) + 1
    assigns = [a for w in workers for a in w.assigns]
    out["assigns"] = sorted(
        (_normalize({k: v for k, v in a.items() if k != "worker_id"}, subs)
         for a in assigns),
        key=lambda a: json.dumps(a, sort_keys=True))
    out["n_stripes_expected_forced"] = len(rows)
    return out


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
