#!/usr/bin/env python3
"""S172 DEFECT A — RANGE-MINER TRANSPORT-SESSION RECOVERY (arms A1-A8).

Beta ruling "GATE-12 ATTEMPT-2 FORENSIC RULING" (2026-08-10) §§10-16. Defect A
only; Defect B (sampler turnover aggregation) is a separate, already-committed
amendment and nothing here touches the sampler.

THE DEFECT THIS SUITE EXISTS FOR (`distributed_config_t1_abc63f71`, 2026-08-10)
-----------------------------------------------------------------------------
`serve_forever` exited at ONE point for THREE causes — an explicit `shutdown`
frame, a signal, and a bare transport loss:

    while not self._stop.is_set():
        try:    msg = self.conn.recv_msg()
        except (ConnectionError, ValueError, OSError):
            break                    # <- all three collapse here

and `main()` then returned 0, so a transport blip terminated a permitted worker
SILENTLY and SUCCESSFULLY. With `use_persistent_workers=false` nothing ever
re-established that session, so stage 4's admission was permanently short:
23 of 25 after 180.1 s, and the trial died at the stage-3->4 boundary having
already spent 6.44 B seed-evals.

The coordinator already held the useful half and is NOT rebuilt here:
`_drop_conn` evicts the dead socket's identity from the structures `_eligible()`
is built from, and `_serve_register` accepts that same identity on a NEW socket
once evicted while rejecting a duplicate that races the eviction. The missing
seam was worker-side, and that is what these arms gate.

WHAT EACH ARM PROVES
--------------------
  A1  the reproduction: 25 admitted, stages 1-2-3 complete, two IDLE workers lose
      transport at the stage-3->4 boundary, the SAME two processes reconnect with
      the SAME identities, stage 4 admits 25 and the trial completes.
  A2  the CONTROL for A1: identical fixture with reconnection disabled -> 23/25,
      admission refused, stage 4 never assigned. Proves the fix is what closes
      A1 and that §4.3 admission is untouched.
  A3  a constant-phase ACTIVE loss still lands on the certified terminal policy;
      reconnect does not erase it.
  A4  a hybrid ACTIVE loss consumes the certified retry EXACTLY ONCE on an
      alternate worker; the failed worker cannot self-reclaim, and its reconnect
      restores FUTURE eligibility only.
  A5  §12 duplicate-socket race: reconnect before eviction is rejected, the
      identity stays singular, the worker retries, and the retry after eviction
      succeeds.
  A6  the CONTROL for the reconnect branch: an explicit `shutdown` frame exits
      with ZERO reconnect attempts.
  A7  §13 frozen-cohort wall: same identity reconnects; a changed capability
      identity and a non-frozen identity both fail closed.
  A8  §14 bound: a coordinator unavailable past the recovery bound exits cleanly
      and does not haunt a future run.

FIXTURE SHAPE (Beta §16 + §27). The fleet arms run a REAL 25-worker, four-stage
trial shape on loopback with stub executors — no GPUs — because small-count
fixtures are exactly what missed this defect. The 25 identities are DERIVED from
the frozen execution set (`resolve_execution_set`), which reads the committed
`distributed_config.json`; they are never a hardcoded list, and they are the same
25 the attempt-2 log names in its `[S172-CAP] cohort frozen` line.

The identity-binding, eviction and duplicate-rejection decisions are made by the
REAL coordinator methods (`_serve_register`, `_drop_conn`) called with the real
serve-loop structures, and the F1/F2 arms drive the REAL certified retry machinery
(`assign_stripes`, `process_lease_expiry`, `schedule_pending_stripes`). What the
harness supplies is the accept loop and the stub executor, not the decisions.

Red-first + mutation evidence per arm (`_mutant_red`): an arm that cannot detect
the removal of the behaviour it claims to check is vacuous and proves nothing.

Run:  source ~/venvs/torch/bin/activate
      python3 -u tests/test_s172_defect_a_transport_recovery.py | tee /tmp/defect_a.log
"""
import json
import logging
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_results = []


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:                                   # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


def _mutant_red(fn, label):
    """VIR-2 positive control: `fn` MUST raise. A mutation an arm cannot detect
    means the arm is vacuous."""
    try:
        fn()
    except Exception:                                        # noqa: BLE001
        return
    raise AssertionError(
        f"MUTANT SURVIVED ({label}) — the arm did not detect it, so it is "
        f"vacuous and proves nothing")


from miner.range_miner_coordinator import (  # noqa: E402
    DEFAULT_WORKER_ADMISSION_TIMEOUT,
    ST_CLAIMED,
    ST_PENDING,
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    RangeMinerCoordinator,
)
import miner.range_miner_coordinator as COORD  # noqa: E402
import miner.range_miner_worker as WORKER  # noqa: E402
from miner.range_miner_worker import (  # noqa: E402
    SESSION_END_IDENTITY_REFUSED,
    SESSION_END_TRANSPORT_LOSS,
    STOP_CAUSE_EXPLICIT_SHUTDOWN,
    STOP_CAUSE_SIGNAL,
    GpuInfo,
    MinerFramedSocket,
    RangeMinerWorker,
    SubStripeOutcome,
    VramCaps,
    WorkerIdentityChanged,
    default_recovery_budget_s,
)
from miner.range_miner_protocol import (  # noqa: E402
    MinerShutdownMessage,
    StripeAssignMessage,
)
from execution_set import (  # noqa: E402
    active_execution_set,
    clear_execution_set,
    freeze_execution_set,
    resolve_execution_set,
)

LEASE = 300.0
MACRO = 1000
CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
VARIANTS = ["java_lcg", "java_lcg_reverse", "java_lcg_hybrid",
            "java_lcg_hybrid_reverse"]
# The four-stage workflow shape, exactly as the attempt-2 run logged it
# ([S172-CAP] cohort frozen stages java_lcg/1, java_lcg_reverse/2,
# java_lcg_hybrid/3, java_lcg_hybrid_reverse/4).
STAGES = [("java_lcg", 1), ("java_lcg_reverse", 2),
          ("java_lcg_hybrid", 3), ("java_lcg_hybrid_reverse", 4)]
# Test-scoped recovery bound. The PRODUCTION bound is the derived 180 s anchor
# (gated in g_a8_bound_derivation); a suite cannot wait three minutes per arm.
TEST_BUDGET = 6.0
DEADLINE = 30.0            # per-wait ceiling so a wedged arm fails, not hangs


# ===========================================================================
# Frozen 25-worker cohort, DERIVED (never a hardcoded identity list)
# ===========================================================================
def frozen_cohort():
    """The 25 production identities, read from the committed config through the
    certified `resolve_execution_set`. Returns (set, worker_ids)."""
    s = resolve_execution_set(backend="miner", invoked_by=__file__,
                              admission_count=25, rig_profile="proxmox")
    wids = list(s.worker_ids())
    assert len(wids) == 25, f"expected the 25-GPU production shape, got {len(wids)}"
    assert s.admission_count == 25, s.admission_count
    return s, wids


# ===========================================================================
# §15 observability capture — read off what the code EMITTED while running
# ===========================================================================
class _SessionLog:
    """Capture `[MINER-SESSION]` records from the coordinator's and the worker's
    OWN logger objects. Attaching by dotted name would be a guess (the production
    log shows these emitting as `range_miner_coordinator`), and a handler that
    captures nothing would let every §15 gate pass vacuously."""

    def __init__(self):
        self.events = []
        self._h = None
        self._lock = threading.Lock()

    def __enter__(self):
        outer = self

        class _H(logging.Handler):
            def emit(self, record):
                try:
                    msg = record.getMessage()
                except Exception:                            # noqa: BLE001
                    return
                if "[MINER-SESSION]" not in msg:
                    return
                start = msg.find("{")
                if start < 0:
                    return
                try:
                    payload = json.loads(msg[start:])
                except json.JSONDecodeError:
                    return
                with outer._lock:
                    outer.events.append(payload)

        self._h = _H()
        self._h.setLevel(logging.DEBUG)
        # The suite silences the ROOT logger, and these loggers inherit their
        # effective level from it — so without raising them here every record is
        # dropped BEFORE reaching the handler and every §15 gate would pass
        # vacuously on an empty capture.
        self._levels = [(lg, lg.level, lg.propagate)
                        for lg in (COORD.logger, WORKER.logger)]
        for lg, _lvl, _prop in self._levels:
            lg.setLevel(logging.DEBUG)
            lg.propagate = False        # capture without spraying the suite output
            lg.addHandler(self._h)
        return self

    def __exit__(self, *a):
        for lg, lvl, prop in self._levels:
            lg.removeHandler(self._h)
            lg.setLevel(lvl)
            lg.propagate = prop
        return False

    def of(self, kind, worker_id=None):
        with self._lock:
            out = [e for e in self.events if e.get("event") == kind]
        if worker_id is not None:
            out = [e for e in out if e.get("worker_id") == worker_id]
        return out


# ===========================================================================
# The controlled coordinator-side harness
# ===========================================================================
class _HarnessCoordinator:
    """A real framed-TCP acceptor that delegates every IDENTITY decision to the
    REAL coordinator methods, driving them with the same structures `serve_trial`
    builds. It supplies only the accept loop and the test's dispatch.

    Deliberately NOT `serve_trial` itself: these arms need to drop one named
    worker's transport at a chosen instant and to hold the registration of a
    racing duplicate, neither of which is reachable from outside that loop. The
    admission POLICY is not re-decided here — `admit()` applies the same rule
    `_eligible()` feeds (`len(eligible) >= expected_workers`) and the production
    policy remains certified by test_s172_admission_liveness / phase-4.
    """

    def __init__(self, coord, expected_workers):
        self.coord = coord
        self.expected_workers = expected_workers
        self.fs_by_sock = {}
        self.worker_by_sock = {}
        self.wconn_by_worker = {}
        self.fs_by_worker = {}
        self.registered = []
        self.received = []              # every frame, in arrival order
        self.register_status = []       # (worker_id, status) per REGISTER
        self.stage_idx = 0
        self.stage_assigned = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._hold_register = None      # worker_id whose REGISTER is held (A5)
        self._held = []
        # A5: in production the reader's `eof` goes onto the serve loop's inbound
        # QUEUE and `_drop_conn` runs later in the dispatch loop — so a reconnect
        # can genuinely beat the eviction. Suppressing the reader-thread eviction
        # models exactly that window; it is not a weakening of the eviction path,
        # which A1 exercises in full.
        self._suppress_eviction = False
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(64)
        self.srv.settimeout(0.25)
        self.port = self.srv.getsockname()[1]
        self._acceptor = threading.Thread(target=self._accept_loop, daemon=True)
        self._acceptor.start()

    # ----- the same expression serve_trial's _eligible() is ----------------
    def eligible(self):
        with self._lock:
            return [w for w in self.wconn_by_worker.values() if not w.quarantined]

    def eligible_ids(self):
        return sorted(w.worker_id for w in self.eligible())

    def admit(self, timeout=5.0):
        """The admission rule, applied to the live pool. True iff the frozen
        expectation is met inside the window; the count is returned either way."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            n = len(self.eligible())
            if n >= self.expected_workers:
                return True, n
            time.sleep(0.05)
        return False, len(self.eligible())

    # ----- accept / read --------------------------------------------------
    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                sock, _ = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._reader, args=(sock,), daemon=True).start()

    def _reader(self, sock):
        fs = MinerFramedSocket(sock)
        with self._lock:
            self.fs_by_sock[sock] = fs
        while not self._stop.is_set():
            try:
                msg = fs.recv_msg()
            except Exception:                                # noqa: BLE001
                break
            if msg.message_type == "register":
                self._on_register(msg, sock)
                continue
            with self._lock:
                self.received.append(msg)
        # The socket died or the peer went away: evict through the REAL path.
        with self._lock:
            if self._suppress_eviction:
                return
            if sock in self.fs_by_sock or sock in self.worker_by_sock:
                self._drop(sock)

    def _on_register(self, msg, sock):
        with self._lock:
            if self._hold_register == msg.worker_id:
                # A5: hold this REGISTER so the reconnect RACES the eviction that
                # has not happened yet — the duplicate arrives while the old
                # socket is still live and bound.
                self._held.append((msg, sock))
                return
            status = self.coord._serve_register(
                msg, sock, None, self.fs_by_sock, self.worker_by_sock,
                self.wconn_by_worker, self.fs_by_worker, self.registered,
                eligible_fn=self.eligible)
            self.register_status.append((msg.worker_id, status))
            if status == "reject_dup_worker":
                # Verbatim the serve-loop disposition: the duplicate was never
                # bound, so the ORIGINAL worker's identity survives the drop.
                self._drop(sock)

    def release_held_registers(self):
        with self._lock:
            held, self._held = self._held, []
            self._hold_register = None
        for msg, sock in held:
            self._on_register(msg, sock)
        return len(held)

    # ----- the real eviction ---------------------------------------------
    def _drop(self, sock):
        self.coord._drop_conn(
            sock, self.fs_by_sock, self.worker_by_sock, self.fs_by_worker,
            self.wconn_by_worker, self.registered,
            stage_idx=self.stage_idx, stage_assigned=self.stage_assigned,
            eligible_fn=self.eligible)

    def drop_transport(self, worker_id, reset=False):
        """Kill ONE worker's session with no `shutdown` frame — the production
        shape: the worker learns of it as a transport exception, not as a stop.

        `reset=True` sends an RST instead of a FIN (SO_LINGER 0, close before the
        eviction's own shutdown can emit a FIN). That matters for the ACTIVE-loss
        arms: after a FIN, the worker's next `send` may still SUCCEED locally into
        the kernel buffer, so the loss would only surface later at the next read
        and the stripe would look like it completed. An RST makes the very next
        I/O fail, which is what "the transport died while a stripe was active"
        has to mean for A3/A4 to be testing that path at all.
        """
        with self._lock:
            sock = next((s for s, w in self.worker_by_sock.items()
                         if w == worker_id), None)
            assert sock is not None, f"{worker_id} is not bound to any socket"
            if reset:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                    struct.pack("ii", 1, 0))
                    sock.close()
                except OSError:
                    pass
            self._drop(sock)

    def send(self, worker_id, msg):
        with self._lock:
            fs = self.fs_by_worker.get(worker_id)
            assert fs is not None, f"{worker_id} has no live socket"
        fs.send_msg(msg)

    def send_shutdown(self, worker_id):
        self.send(worker_id, MinerShutdownMessage())

    def frames_for(self, stripe_id, types=None):
        with self._lock:
            out = [m for m in self.received
                   if getattr(m, "stripe_id", None) == stripe_id]
        if types is not None:
            out = [m for m in out if m.message_type in types]
        return out

    def wait_registered(self, n, timeout=DEADLINE):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if len(self.eligible()) >= n:
                return True
            time.sleep(0.02)
        return False

    def close(self):
        self._stop.set()
        try:
            self.srv.close()
        except Exception:                                    # noqa: BLE001
            pass


# ===========================================================================
# Controlled workers
# ===========================================================================
def _fail_worker_writes(worker):
    """Make the ACTIVE-loss path deterministic.

    A worker computing a stripe is neither reading nor writing, so it cannot
    notice a peer's disappearance until its next I/O — and a peer FIN does NOT
    reliably fail that next `send` (the bytes land in the local kernel buffer and
    the loss only surfaces at a later read, by which time the stripe has finished
    and the loss looks IDLE). An RST is not reliable either while the peer's
    reader is still blocked on the socket. So the fault is injected at the
    worker's own socket: after SHUT_WR the very next `send` raises, which is
    exactly the observable the code under test reacts to — a transport exception
    raised while `state == "mining"`. The peer-side eviction still runs through
    the real `_drop_conn`; this only fixes WHEN the worker finds out.
    """
    assert worker.conn is not None
    worker.conn.sock.shutdown(socket.SHUT_WR)


class _FakeSock:
    """A socket stand-in for the two arms that drive `_serve_register` /
    `_drop_conn` directly, with no real connection. It must answer the same
    teardown calls the real path makes (`shutdown`/`close`), because `_drop_conn`
    deliberately shuts down before closing."""

    def __init__(self, name="fake"):
        self.name = name
        self.closed = False
        self.shutdown_called = False

    def shutdown(self, _how):
        self.shutdown_called = True

    def close(self):
        self.closed = True


def _stub_executor(delay=0.0, block=None):
    """A stub sieve. `block` (an Event) lets an arm hold a stripe ACTIVE while its
    transport is dropped — that is how A3/A4 reach the active-loss path without a
    GPU."""
    def _exec(assign, seed_start, seed_count):
        if block is not None:
            block.wait(DEADLINE)
        if delay:
            time.sleep(delay)
        return SubStripeOutcome(survivors=[(seed_start, 0.9, None, [1])], count=1)
    return _exec


def _mk_worker(port, worker_id, spool, *, executor=None, reconnect_enabled=True,
               budget=TEST_BUDGET, gpu_info=None):
    """A controlled worker bound to a chosen identity. `hostname`/`gpu_id` are set
    so `worker_id` is exactly the frozen-cohort identity under test."""
    host, gpu = worker_id.rsplit(":gpu", 1)
    w = RangeMinerWorker(
        host="127.0.0.1", port=port, gpu_id=int(gpu), caps=VramCaps(**CAPS),
        executor=executor or _stub_executor(),
        gpu_info=gpu_info or GpuInfo("cuda", "stub", 12 * 1024 ** 3),
        hostname=host, heartbeat_interval=999, miner_output_dir=spool,
        recovery_budget_s=budget, reconnect_enabled=reconnect_enabled)
    assert w.worker_id == worker_id, (w.worker_id, worker_id)
    return w


def _spin(worker):
    """Run one worker's whole session lifecycle in a thread, as `main()` does."""
    box = {}

    def _run():
        try:
            worker.connect()
            worker.register()
            worker.serve_forever()
            box["returned"] = True
        except BaseException:                                # noqa: BLE001
            box["err"] = traceback.format_exc()

    t = threading.Thread(target=_run, name=f"w-{worker.worker_id}", daemon=True)
    t.start()
    return t, box


def _assign(worker_id, stripe_id, family, phase, seed_start=0, seed_count=10):
    return StripeAssignMessage(
        worker_id=worker_id, stripe_id=stripe_id, family_name=family,
        seed_start=seed_start, seed_count=seed_count, phase=phase,
        payload={"min_match_threshold": 0.4, "k": 3, "residues": [1, 2, 3],
                 "strategies": None})


def _coord(tmp, **cfg):
    cfg.setdefault("miner_stripe_size", MACRO)
    cfg.setdefault("compute_lease_timeout", LEASE)
    cfg.setdefault("staging_dir", os.path.join(tmp, "staging"))
    ledger = MinerLedger(os.path.join(tmp, "l.db"))
    return RangeMinerCoordinator(CoordinatorConfig(**cfg), ledger)


def _register_direct(coord, wid, now=100.0):
    """Direct (non-socket) registration, for the F1/F2 arms that need certified
    lease arithmetic rather than a live session."""
    node = NodeConfig(hostname=wid.split(":")[0], spool_root="/var/spool/miner",
                      ssh_address="10.0.0.9", ssh_user="michael")
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend="cuda",
        capabilities={"seed_caps": dict(CAPS), "supported_variants": list(VARIANTS)},
        node_config=node, now=now)


# ===========================================================================
# The 25-worker x 4-stage fleet fixture (A1 / A2)
# ===========================================================================
def _run_fleet(reconnect_enabled, drop_count=2, log=None):
    """Drive the real four-stage shape over 25 controlled workers.

    Returns a dict of everything the arms assert on. Stages 1-3 complete on all
    25; at the stage-3->4 boundary `drop_count` IDLE workers lose transport; then
    stage 4's admission is evaluated.
    """
    _set, wids = frozen_cohort()
    out = {"worker_ids": wids, "stage_admitted": [], "stage_completed": [],
           "dropped": [], "eligible_at_stage4": None, "stage4_assigned": False}
    with tempfile.TemporaryDirectory() as tmp:
        spool = os.path.join(tmp, "spool")
        os.makedirs(spool, exist_ok=True)
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=len(wids))
        workers, threads = [], []
        try:
            for wid in wids:
                w = _mk_worker(h.port, wid, spool,
                               reconnect_enabled=reconnect_enabled)
                t, box = _spin(w)
                workers.append((w, box))
                threads.append(t)
            assert h.wait_registered(len(wids)), \
                f"only {len(h.eligible())}/{len(wids)} registered"

            for si, (family, phase) in enumerate(STAGES):
                h.stage_idx = si
                h.stage_assigned = False
                ok, n = h.admit(timeout=5.0)
                out["stage_admitted"].append((si, ok, n))
                if not ok:
                    # Admission refused: the stage is NOT assigned at a short
                    # pool. This is the A2 outcome and it stops the trial here.
                    out["eligible_at_stage4"] = n
                    break
                # assign one stripe per eligible worker for this stage
                live = h.eligible_ids()
                h.stage_assigned = True
                if si == 3:
                    out["stage4_assigned"] = True
                    out["eligible_at_stage4"] = n
                expect = set()
                for k, wid in enumerate(live):
                    sid = f"run__st{si}_s{k}"
                    expect.add(sid)
                    h.send(wid, _assign(wid, sid, family, phase))
                # every stripe of this stage must report complete
                t0 = time.monotonic()
                while time.monotonic() - t0 < DEADLINE:
                    done = {m.stripe_id for m in h.received
                            if m.message_type == "stripe_complete"
                            and m.stripe_id in expect}
                    if done == expect:
                        break
                    time.sleep(0.02)
                else:
                    raise AssertionError(
                        f"stage {si} did not complete: "
                        f"{len(done)}/{len(expect)} stripes")
                out["stage_completed"].append(si)

                if si == 2:
                    # THE STAGE-3->4 BOUNDARY. Every worker is idle (its stage-3
                    # stripe reported complete), which is the exact production
                    # blind spot: an IDLE session lost with nothing in flight.
                    victims = h.eligible_ids()[:drop_count]
                    out["dropped"] = list(victims)
                    assert len(h.eligible()) == len(wids), len(h.eligible())
                    for v in victims:
                        h.drop_transport(v)
                    # the pool must actually shrink first, or the arm is vacuous
                    t0 = time.monotonic()
                    while (time.monotonic() - t0 < DEADLINE
                           and len(h.eligible()) > len(wids) - drop_count):
                        time.sleep(0.02)
                    out["eligible_after_drop"] = len(h.eligible())
                    assert out["eligible_after_drop"] == len(wids) - drop_count, \
                        out["eligible_after_drop"]
                    if reconnect_enabled:
                        # give recovery its bounded window to re-register
                        h.wait_registered(len(wids), timeout=TEST_BUDGET + 10.0)
            out["final_eligible"] = len(h.eligible())
            out["errors"] = {w.worker_id: b["err"] for w, b in workers if "err" in b}
            out["returned"] = {w.worker_id for w, b in workers if b.get("returned")}
            out["session_events"] = {w.worker_id: list(w.session_events)
                                     for w, _b in workers}
            out["register_status"] = list(h.register_status)
            return out
        finally:
            for w, _b in workers:
                try:
                    w.shutdown(cause=STOP_CAUSE_EXPLICIT_SHUTDOWN)
                except Exception:                            # noqa: BLE001
                    pass
            h.close()


# ===========================================================================
# A1 — the exact production blind spot (the reproduction)
# ===========================================================================
def g_a1_reproduction():
    """25 admitted; stages 1-2-3 complete; two IDLE workers lose transport at the
    stage-3->4 boundary; the SAME two processes reconnect with the SAME
    identities; stage 4 admits 25 and completes."""
    with _SessionLog() as log:
        r = _run_fleet(reconnect_enabled=True, log=log)

    assert not r["errors"], r["errors"]
    assert r["stage_completed"] == [0, 1, 2, 3], r["stage_completed"]
    for si, ok, n in r["stage_admitted"]:
        assert ok and n == 25, (si, ok, n)
    assert r["eligible_after_drop"] == 23, r["eligible_after_drop"]
    assert r["stage4_assigned"] is True
    assert r["eligible_at_stage4"] == 25, r["eligible_at_stage4"]
    assert r["final_eligible"] == 25, r["final_eligible"]

    # the SAME two identities came back — not replacements, not a widened cohort
    for wid in r["dropped"]:
        evs = r["session_events"][wid]
        kinds = [e["event"] for e in evs]
        assert "TRANSPORT_LOSS" in kinds, kinds
        assert "RECONNECTED" in kinds, kinds
        loss = [e for e in evs if e["event"] == "TRANSPORT_LOSS"][0]
        assert loss["classification"] == SESSION_END_TRANSPORT_LOSS, loss
        assert loss["assignment_active_at_loss"] is False, \
            "the production blind spot is an IDLE loss"
        rec = [e for e in evs if e["event"] == "RECONNECTED"][0]
        assert rec["reconnect_success"] is True, rec
        assert rec["resumed_state"] == "idle", rec
        assert rec["session_generation"] == 2, rec
    # §15 coordinator bracket: the drop and the re-registration both NAME the id
    dis = {e["worker_id"] for e in log.of("WORKER_DISCONNECTED")}
    rec = {e["worker_id"] for e in log.of("WORKER_RECONNECTED")}
    assert set(r["dropped"]) <= dis, (r["dropped"], dis)
    assert set(r["dropped"]) <= rec, (r["dropped"], rec)
    # The two drops are sequential, so the bracket reads the pool shrinking
    # 25 -> 24 -> 23. That descent IS the record attempt 2 did not have: a future
    # 23/25 can name both IDs and the transition, with no reconstruction.
    # First two records only: the fixture's own teardown legitimately drops every
    # worker afterwards, and those records are real, not noise to be filtered out
    # of the emitter.
    counts = [e["eligible_count_after_drop"] for e in log.of("WORKER_DISCONNECTED")
              if e["worker_id"] in r["dropped"]]
    assert counts[:2] == [24, 23], counts
    for wid in r["dropped"]:
        first = log.of("WORKER_DISCONNECTED", worker_id=wid)[0]
        assert first["obs_status"] == "OBSERVED", first
        assert first["stage_idx"] == 2, first          # the stage-3->4 boundary
        assert first["stage_assigned"] is True, first
        assert first["identity_evicted"] is True, first
    for e in log.of("WORKER_RECONNECTED"):
        if e["worker_id"] in r["dropped"]:
            assert e["registration_generation"] == 2, e
            assert e["quarantined"] is False, e


def _PREFIX_serve_forever(self):
    """The CERTIFIED PRE-FIX body of `serve_forever`, copied verbatim from
    `f216475:miner/range_miner_worker.py` (the defect site Beta named). Restored
    only inside the red-first gate below, to prove this suite has power against the
    real defect and not merely against a feature flag."""
    assert self.conn is not None, "call connect() + register() first"
    self._hb_thread = threading.Thread(
        target=self._heartbeat_loop, name="miner-heartbeat", daemon=True
    )
    self._hb_thread.start()
    try:
        while not self._stop.is_set():
            try:
                msg = self.conn.recv_msg()
            except (ConnectionError, ValueError, OSError):
                break
            self._dispatch(msg)
    finally:
        self.shutdown()


def g_a1_red_first():
    """RED-FIRST. With the pre-fix `serve_forever` restored verbatim, a single idle
    transport loss ends the daemon silently and the identity never returns — the
    exact attempt-2 mechanism. The gate must detect it."""
    _set, wids = frozen_cohort()
    wid = wids[0]

    def _prefix_behaviour():
        with tempfile.TemporaryDirectory() as tmp:
            spool = os.path.join(tmp, "spool")
            os.makedirs(spool, exist_ok=True)
            coord = _coord(tmp)
            h = _HarnessCoordinator(coord, expected_workers=1)
            w = _mk_worker(h.port, wid, spool)
            orig = WORKER.RangeMinerWorker.serve_forever
            WORKER.RangeMinerWorker.serve_forever = _PREFIX_serve_forever
            try:
                t, box = _spin(w)
                assert h.wait_registered(1)
                h.drop_transport(wid)
                t.join(timeout=TEST_BUDGET + 10.0)
                # the pre-fix daemon exits silently and SUCCESSFULLY
                assert not t.is_alive(), "pre-fix worker did not exit"
                assert box.get("returned") is True, box
                assert w.reconnect_attempts_total == 0, w.reconnect_attempts_total
                # THE ASSERTION THE FIX MAKES TRUE — and that must fail here
                assert h.eligible_ids() == [wid], \
                    f"the identity never came back: {h.eligible_ids()}"
            finally:
                WORKER.RangeMinerWorker.serve_forever = orig
                w.shutdown()
                h.close()
    _mutant_red(_prefix_behaviour,
                "pre-fix serve_forever must lose the identity for good")


def g_a1_mutant():
    """Red-first: with the §10 discriminator removed the arm must RED. A worker
    that cannot recover leaves the pool at 23 and stage 4 is never assigned."""
    def _no_recovery():
        r = _run_fleet(reconnect_enabled=False)
        assert r["final_eligible"] == 25, \
            f"pool never recovered: {r['final_eligible']}"
    _mutant_red(_no_recovery, "reconnect disabled -> pool stays at 23")


# ===========================================================================
# A2 — the CONTROL: no dynamic downsizing
# ===========================================================================
def g_a2_no_downsizing():
    """Identical fixture, reconnection DISABLED -> 23/25 -> admission refused and
    stage 4 NOT assigned. §4.3 admission is untouched: the pool shrinks, the
    expectation does not, and the stage does not run at a short pool."""
    with _SessionLog() as log:
        r = _run_fleet(reconnect_enabled=False)

    assert not r["errors"], r["errors"]
    assert r["stage_completed"] == [0, 1, 2], r["stage_completed"]
    assert r["eligible_after_drop"] == 23, r["eligible_after_drop"]
    assert r["stage4_assigned"] is False, "stage 4 must NOT be assigned at 23"
    assert r["eligible_at_stage4"] == 23, r["eligible_at_stage4"]
    si, ok, n = r["stage_admitted"][-1]
    assert si == 3 and ok is False and n == 23, r["stage_admitted"]
    # the expectation itself was never renegotiated downward
    assert len(r["worker_ids"]) == 25
    # the dropped workers exited; they did NOT reconnect
    for wid in r["dropped"]:
        kinds = [e["event"] for e in r["session_events"][wid]]
        assert "RECONNECT_DISABLED" in kinds, kinds
        assert "RECONNECTED" not in kinds, kinds
        assert wid in r["returned"], f"{wid} did not exit cleanly"
    # NO-TOUCH guard: the production admission authority is unchanged
    assert DEFAULT_WORKER_ADMISSION_TIMEOUT == 180.0, DEFAULT_WORKER_ADMISSION_TIMEOUT


def g_a2_mutant():
    """A2 is only a control if a SHORT pool would really be refused. Assert the
    rule reds when the expectation is lowered to the degraded count — i.e. the
    arm is detecting admission, not just counting sockets."""
    def _downsized():
        _set, wids = frozen_cohort()
        with tempfile.TemporaryDirectory() as tmp:
            coord = _coord(tmp)
            h = _HarnessCoordinator(coord, expected_workers=len(wids))
            try:
                for wid in wids[:23]:
                    _register_direct(coord, wid)
                    h.wconn_by_worker[wid] = coord.connections[wid]
                ok, n = h.admit(timeout=0.3)
                assert ok, f"admission must refuse at {n}/25, but it admitted"
            finally:
                h.close()
    _mutant_red(_downsized, "23 of 25 must not satisfy a 25-worker expectation")


# ===========================================================================
# A3 — constant-phase active loss unchanged
# ===========================================================================
def g_a3_constant_phase_terminal():
    """A transport loss while a CONSTANT-phase stripe is active: the worker
    abandons it with no replay, and the certified F1/F2 constant-phase policy
    still fails the trial. Reconnect restores the session, NOT the assignment."""
    _set, wids = frozen_cohort()
    victim, alt = wids[0], wids[1]
    with tempfile.TemporaryDirectory() as tmp:
        spool = os.path.join(tmp, "spool")
        os.makedirs(spool, exist_ok=True)
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=1)
        block = threading.Event()
        w = _mk_worker(h.port, victim, spool, executor=_stub_executor(block=block))
        t, box = _spin(w)
        try:
            assert h.wait_registered(1), "worker never registered"
            # the ledger-side stripe the certified machinery will rule on
            coord.ledger.create_trial("run", 1, now=1000.0)
            A = _register_direct(coord, victim, now=1000.0)
            B = _register_direct(coord, alt, now=1000.0)
            assigns = coord.assign_stripes("run", "java_lcg", 1, MACRO * 2, [A, B],
                                           stripe_prefix="run__st0", now=1000.0)
            sid = next(a["stripe_id"] for a in assigns
                       if a["claimed"] and a["worker_id"] == victim)
            # hand the SAME stripe to the live worker and let it go active
            h.send(victim, _assign(victim, sid, "java_lcg", 1))
            t0 = time.monotonic()
            while w.state != "mining" and time.monotonic() - t0 < DEADLINE:
                time.sleep(0.01)
            assert w.state == "mining", w.state

            _fail_worker_writes(w)            # the next send WILL fail (see helper)
            h.drop_transport(victim)          # loss WHILE the stripe is active
            block.set()                       # executor finishes into a dead socket

            # certified constant-phase policy, driven by the real machinery
            tt = 1000.0 + LEASE + 1.0
            out = coord.process_lease_expiry("run", [A, B], now=tt)
            assert out, out
            assert any(o["action"] == "fail_trial" for o in out), out
            assert coord.ledger.get_trial("run")["state"] == "aborted", \
                "constant-phase loss must remain TERMINAL"
            st = coord.ledger.get_stripe("run", sid)
            assert st["current_attempt"] == 0, \
                f"constant phase must not consume a retry: {st['current_attempt']}"

            # §11: the worker came back IDLE and replayed NOTHING about that stripe
            assert h.wait_registered(1, timeout=TEST_BUDGET + 10.0), \
                "worker did not recover its session"
            assert w.state == "idle", w.state
            assert w.current_stripe_id == "", w.current_stripe_id
            kinds = [e["event"] for e in w.session_events]
            assert "ASSIGNMENT_ABANDONED" in kinds, kinds
            ab = [e for e in w.session_events
                  if e["event"] == "ASSIGNMENT_ABANDONED"][0]
            assert ab["replayed"] is False, ab
            assert ab["stripe_id"] == sid, ab
            loss = [e for e in w.session_events
                    if e["event"] == "TRANSPORT_LOSS"][0]
            assert loss["assignment_active_at_loss"] is True, loss
            # nothing about the abandoned stripe arrived after the reconnect
            time.sleep(0.4)
            late = h.frames_for(sid, {"sub_stripe_result", "stripe_complete",
                                      "stripe_error"})
            assert not late, f"stale-work replay: {[m.message_type for m in late]}"
            # and the trial's terminal state was NOT reopened by the reconnect
            assert coord.ledger.get_trial("run")["state"] == "aborted"
        finally:
            block.set()
            w.shutdown(cause=STOP_CAUSE_EXPLICIT_SHUTDOWN)
            h.close()


def g_a3_mutant():
    """Red-first for the no-replay rule: a worker that DID re-send the abandoned
    stripe's completion would be detected."""
    def _replay():
        _set, wids = frozen_cohort()
        with tempfile.TemporaryDirectory() as tmp:
            spool = os.path.join(tmp, "spool")
            os.makedirs(spool, exist_ok=True)
            coord = _coord(tmp)
            h = _HarnessCoordinator(coord, expected_workers=1)
            w = _mk_worker(h.port, wids[0], spool)
            t, box = _spin(w)
            try:
                assert h.wait_registered(1)
                sid = "run__st0_s0"
                # simulate the defect: replay a completion for a stripe the
                # worker no longer owns
                h.send(wids[0], _assign(wids[0], sid, "java_lcg", 1))
                t0 = time.monotonic()
                while time.monotonic() - t0 < DEADLINE:
                    if h.frames_for(sid, {"stripe_complete"}):
                        break
                    time.sleep(0.02)
                late = h.frames_for(sid, {"stripe_complete"})
                assert not late, f"stale-work replay: {len(late)}"
            finally:
                w.shutdown(cause=STOP_CAUSE_EXPLICIT_SHUTDOWN)
                h.close()
    _mutant_red(_replay, "a completion for an abandoned stripe must be detected")


# ===========================================================================
# A4 — hybrid retry unchanged
# ===========================================================================
def g_a4_hybrid_retry_once():
    """A hybrid ACTIVE loss consumes the certified retry EXACTLY once, an
    alternate executes it, the failed worker cannot self-reclaim its failed
    attempt, and its reconnect restores FUTURE eligibility only."""
    _set, wids = frozen_cohort()
    victim, alt = wids[0], wids[1]
    with tempfile.TemporaryDirectory() as tmp:
        spool = os.path.join(tmp, "spool")
        os.makedirs(spool, exist_ok=True)
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=1)
        block = threading.Event()
        w = _mk_worker(h.port, victim, spool, executor=_stub_executor(block=block))
        t, box = _spin(w)
        try:
            assert h.wait_registered(1)
            coord.ledger.create_trial("run", 1, now=1000.0)
            A = _register_direct(coord, victim, now=1000.0)
            B = _register_direct(coord, alt, now=1000.0)
            assigns = coord.assign_stripes("run", "java_lcg_hybrid", 3, MACRO * 2,
                                           [A, B], stripe_prefix="run__st0",
                                           now=1000.0)
            sid = next(a["stripe_id"] for a in assigns
                       if a["claimed"] and a["worker_id"] == victim)
            other = next(a["stripe_id"] for a in assigns
                         if a["claimed"] and a["worker_id"] == alt)
            h.send(victim, _assign(victim, sid, "java_lcg_hybrid", 3))
            t0 = time.monotonic()
            while w.state != "mining" and time.monotonic() - t0 < DEADLINE:
                time.sleep(0.01)
            assert w.state == "mining", w.state

            _fail_worker_writes(w)                 # see A3: deterministic active loss
            h.drop_transport(victim)
            block.set()

            tt = 1000.0 + LEASE + 1.0
            out = coord.process_lease_expiry("run", [A, B], now=tt)
            assert out, out
            st = coord.ledger.get_stripe("run", sid)
            assert coord.ledger.get_trial("run")["state"] == "running", \
                "a hybrid retryable expiry must NOT fail the trial"
            assert st["phase_degraded"] == 1, st
            assert st["current_attempt"] == 1, \
                f"retry consumed exactly once, got {st['current_attempt']}"
            if out[0]["action"] == "requeued":
                assert st["state"] == ST_PENDING, st
                coord.ledger.record_stripe_complete("run", other, 0, alt, 1, 0)
                placed = coord.schedule_pending_stripes(
                    "run", "java_lcg_hybrid", 3, [A, B],
                    stage_prefix="run__st0", now=tt + 10.0)
                assert placed, placed
                st = coord.ledger.get_stripe("run", sid)
            # THE ALTERNATE executes the certified retry — not the failed worker
            assert st["state"] == ST_CLAIMED, st
            assert st["claimed_by"] == alt, \
                f"the failed worker self-reclaimed its attempt: {st['claimed_by']}"

            # reconnect restores FUTURE eligibility only
            assert h.wait_registered(1, timeout=TEST_BUDGET + 10.0)
            assert w.state == "idle", w.state
            loss = [e for e in w.session_events
                    if e["event"] == "TRANSPORT_LOSS"][0]
            assert loss["assignment_active_at_loss"] is True, loss
            ab = [e for e in w.session_events
                  if e["event"] == "ASSIGNMENT_ABANDONED"]
            assert ab and ab[0]["replayed"] is False, w.session_events
            assert victim in h.eligible_ids(), h.eligible_ids()
            st2 = coord.ledger.get_stripe("run", sid)
            assert st2["claimed_by"] == alt, st2
            assert st2["current_attempt"] == 1, \
                "the reconnect must not consume a second retry"
            late = h.frames_for(sid, {"sub_stripe_result", "stripe_complete",
                                      "stripe_error"})
            assert not late, f"stale-work replay: {[m.message_type for m in late]}"
            # a NEW stripe still reaches the recovered worker (future eligibility)
            h.send(victim, _assign(victim, "run__st1_s0", "java_lcg_hybrid", 3))
            t0 = time.monotonic()
            while time.monotonic() - t0 < DEADLINE:
                if h.frames_for("run__st1_s0", {"stripe_complete"}):
                    break
                time.sleep(0.02)
            assert h.frames_for("run__st1_s0", {"stripe_complete"}), \
                "the recovered worker never took new work"
        finally:
            block.set()
            w.shutdown(cause=STOP_CAUSE_EXPLICIT_SHUTDOWN)
            h.close()


def g_a4_mutant():
    """Red-first on the two claims A4 actually makes about a hybrid active loss:
    the retry is consumed EXACTLY once, and the failed worker does not execute it.

    The second-loss-is-terminal half of the matrix is NOT re-derived here — it is
    certified by G-F1-HYBRID-MATRIX in test_s172_f1_f2_active_lease.py, and
    duplicating it would add a second, weaker authority for the same fact.
    """
    _set, wids = frozen_cohort()
    victim, alt = wids[0], wids[1]
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        A = _register_direct(coord, victim, now=1000.0)
        B = _register_direct(coord, alt, now=1000.0)
        coord.ledger.create_trial("run", 1, now=1000.0)
        assigns = coord.assign_stripes("run", "java_lcg_hybrid", 3, MACRO * 2,
                                       [A, B], stripe_prefix="run__st0", now=1000.0)
        sid = next(a["stripe_id"] for a in assigns
                   if a["claimed"] and a["worker_id"] == victim)
        other = next(a["stripe_id"] for a in assigns
                     if a["claimed"] and a["worker_id"] == alt)
        tt = 1000.0 + LEASE + 1.0
        out = coord.process_lease_expiry("run", [A, B], now=tt)
        assert out, out
        st = coord.ledger.get_stripe("run", sid)
        if st["state"] == ST_PENDING:
            coord.ledger.record_stripe_complete("run", other, 0, alt, 1, 0)
            coord.schedule_pending_stripes("run", "java_lcg_hybrid", 3, [A, B],
                                           stage_prefix="run__st0", now=tt + 10.0)
            st = coord.ledger.get_stripe("run", sid)

        def _retry_twice():
            assert st["current_attempt"] == 2, \
                f"the retry is consumed ONCE, not twice (got {st['current_attempt']})"
        _mutant_red(_retry_twice, "retry budget is one, not two")

        def _self_reclaim():
            assert st["claimed_by"] == victim, \
                f"the failed worker must NOT execute its own retry ({st['claimed_by']})"
        _mutant_red(_self_reclaim, "the failed worker must not self-reclaim")


# ===========================================================================
# A5 — duplicate-socket race (§12 one active connection per identity)
# ===========================================================================
def g_a5_duplicate_socket_race():
    """The reconnect RACES the eviction: the duplicate registration is rejected,
    the identity stays singular, the worker treats it as a retryable
    session-establishment condition, and the retry after eviction succeeds. Never
    a force-replace, never two live sockets for one identity."""
    _set, wids = frozen_cohort()
    victim = wids[0]
    with tempfile.TemporaryDirectory() as tmp:
        spool = os.path.join(tmp, "spool")
        os.makedirs(spool, exist_ok=True)
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=1)
        w = _mk_worker(h.port, victim, spool, budget=25.0)
        t, box = _spin(w)
        try:
            assert h.wait_registered(1)
            first_sock = next(s for s, x in h.worker_by_sock.items() if x == victim)

            # Hold the NEXT registration so the reconnect arrives while the old
            # socket is STILL bound — the §12 race, deliberately induced.
            h._hold_register = victim
            # kill the worker's socket WITHOUT evicting the identity: the peer
            # sees a transport loss while the coordinator still holds the binding
            h._suppress_eviction = True
            h.fs_by_sock[first_sock].close()

            # the reconnect is held -> the worker is left waiting on a socket the
            # coordinator has not accepted as a registration
            t0 = time.monotonic()
            while time.monotonic() - t0 < DEADLINE and not h._held:
                time.sleep(0.02)
            assert h._held, "the reconnect never arrived to race the eviction"
            assert h.worker_by_sock.get(first_sock) == victim, \
                "the ORIGINAL identity binding must still be live for the race"

            # release it INTO the still-live binding -> duplicate rejected
            assert h.release_held_registers() >= 1
            assert ("reject_dup_worker" in
                    [s for wid, s in h.register_status if wid == victim]), \
                h.register_status
            # the identity remained singular throughout
            bound = [s for s, x in h.worker_by_sock.items() if x == victim]
            assert len(bound) <= 1, f"two live sockets for one identity: {bound}"

            # now evict the stale socket, as the real serve loop eventually does
            h._suppress_eviction = False
            h._drop(first_sock)
            assert victim not in h.eligible_ids(), h.eligible_ids()

            # the worker keeps retrying the SAME identity and now succeeds
            assert h.wait_registered(1, timeout=30.0), \
                "the worker did not re-establish after the eviction"
            assert h.eligible_ids() == [victim], h.eligible_ids()
            assert w.reconnect_attempts_total >= 2, \
                f"the rejection was not retried: {w.reconnect_attempts_total}"
            # HOW the refusal reaches the worker: `_serve_register` returns
            # `reject_dup_worker` and the serve loop DROPS that socket — it does not
            # send a refusal frame. So the worker experiences the rejection as the
            # next session's transport loss and recovers again, which is precisely
            # §12's "retryable session-establishment condition, retried after
            # backoff". What must NOT happen is a force-replace or a second live
            # socket, and neither did.
            kinds = [e["event"] for e in w.session_events]
            assert kinds.count("TRANSPORT_LOSS") >= 2, kinds
            assert kinds.count("RECONNECTED") >= 2, kinds
            assert "RECONNECT_ABANDONED" not in kinds, \
                "the worker gave up instead of retrying the rejection"
        finally:
            w.shutdown(cause=STOP_CAUSE_EXPLICIT_SHUTDOWN)
            h.close()


def g_a5_mutant():
    """Red-first: `_serve_register` must REJECT a second live socket for one
    identity. If it bound both, the singularity assertion would survive — so
    assert the rejection is real."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=1)
        try:
            class _Msg:
                message_type = "register"
                worker_id = "hostA:gpu0"
                hostname = "hostA"
                gpu_id = 0
                gpu_name = "stub"
                backend = "cuda"
                vram_bytes = 12 * 1024 ** 3
                capabilities = {"seed_caps": dict(CAPS),
                                "supported_variants": list(VARIANTS)}

            s1, s2 = _FakeSock("s1"), _FakeSock("s2")
            h.fs_by_sock[s1] = _FakeSock("fs1")
            h.fs_by_sock[s2] = _FakeSock("fs2")
            st1 = coord._serve_register(_Msg(), s1, None, h.fs_by_sock,
                                        h.worker_by_sock, h.wconn_by_worker,
                                        h.fs_by_worker, h.registered,
                                        eligible_fn=h.eligible)
            assert st1 == "ok", st1
            st2 = coord._serve_register(_Msg(), s2, None, h.fs_by_sock,
                                        h.worker_by_sock, h.wconn_by_worker,
                                        h.fs_by_worker, h.registered,
                                        eligible_fn=h.eligible)
            assert st2 == "reject_dup_worker", st2

            def _accepts_two():
                assert st2 == "ok", \
                    "a second live socket for one identity must be REJECTED"
            _mutant_red(_accepts_two, "duplicate identity binding must be refused")
        finally:
            h.close()


# ===========================================================================
# A6 — explicit shutdown: the CONTROL for the reconnect branch
# ===========================================================================
def g_a6_explicit_shutdown_no_reconnect():
    """A `shutdown` frame ends the daemon with ZERO reconnect attempts. This is
    the load-bearing discriminator's negative control: the socket dies here too,
    but `_stop` is set, so the recovery branch must NOT fire."""
    _set, wids = frozen_cohort()
    wid = wids[0]
    with tempfile.TemporaryDirectory() as tmp:
        spool = os.path.join(tmp, "spool")
        os.makedirs(spool, exist_ok=True)
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=1)
        w = _mk_worker(h.port, wid, spool)
        t, box = _spin(w)
        try:
            assert h.wait_registered(1)
            h.send_shutdown(wid)
            t.join(timeout=10.0)
            assert not t.is_alive(), "the worker did not exit on a shutdown frame"
            assert box.get("returned") is True, box
            assert w.reconnect_attempts_total == 0, \
                f"reconnected after an EXPLICIT shutdown: {w.reconnect_attempts_total}"
            assert w._stop_cause == STOP_CAUSE_EXPLICIT_SHUTDOWN, w._stop_cause
            kinds = [e["event"] for e in w.session_events]
            assert "RECONNECTED" not in kinds, kinds
            assert "TRANSPORT_LOSS" not in kinds, kinds
            end = [e for e in w.session_events if e["event"] == "SESSION_END"]
            assert end and end[0]["classification"] == STOP_CAUSE_EXPLICIT_SHUTDOWN, end
            assert end[0]["reconnect_attempted"] is False, end
        finally:
            w.shutdown()
            h.close()


def g_a6_signal_no_reconnect():
    """The SIGNAL leg of the same rule: `_handle_sig`'s `shutdown(cause=signal)`
    sets `_stop`, so the dying socket is a consequence of the stop and not a
    transport loss to recover."""
    _set, wids = frozen_cohort()
    wid = wids[0]
    with tempfile.TemporaryDirectory() as tmp:
        spool = os.path.join(tmp, "spool")
        os.makedirs(spool, exist_ok=True)
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=1)
        w = _mk_worker(h.port, wid, spool)
        t, box = _spin(w)
        try:
            assert h.wait_registered(1)
            w.shutdown(cause=STOP_CAUSE_SIGNAL)      # exactly what _handle_sig does
            t.join(timeout=10.0)
            assert not t.is_alive(), "the worker did not exit on a signal"
            assert w.reconnect_attempts_total == 0, w.reconnect_attempts_total
            assert w._stop_cause == STOP_CAUSE_SIGNAL, w._stop_cause
            kinds = [e["event"] for e in w.session_events]
            assert "RECONNECTED" not in kinds, kinds
        finally:
            h.close()


def g_a6_mutant():
    """Red-first on the DISCRIMINATOR itself.

    The stop is defended twice — `_classify_session_end` refuses to call a stop a
    transport loss, and `_recover_session` re-checks `_stop` before its first
    attempt — so a mutation of the classifier alone still cannot make the worker
    reconnect. That defence-in-depth is real and worth keeping, but it means an
    arm that only counts reconnect attempts cannot see the classifier break. This
    mutant therefore targets what the classifier actually decides: with the stop
    flag ignored, a SIGNAL is misclassified as a recoverable transport loss and
    the worker enters the recovery path it should never have entered.
    """
    _set, wids = frozen_cohort()
    wid = wids[0]

    def _ignore_stop():
        with tempfile.TemporaryDirectory() as tmp:
            spool = os.path.join(tmp, "spool")
            os.makedirs(spool, exist_ok=True)
            coord = _coord(tmp)
            h = _HarnessCoordinator(coord, expected_workers=1)
            w = _mk_worker(h.port, wid, spool)
            orig = WORKER.RangeMinerWorker._classify_session_end

            def _mutant(self_, exc):
                # THE MUTATION: classify on the exception alone, ignoring _stop.
                return WORKER.SessionOutcome(
                    cause=SESSION_END_TRANSPORT_LOSS,
                    exc_class=type(exc).__name__, exc_text=str(exc),
                    assignment_active=(self_.state == "mining"),
                    stripe_id=self_.current_stripe_id,
                    sub_index=self_.current_sub_index)

            WORKER.RangeMinerWorker._classify_session_end = _mutant
            try:
                t, box = _spin(w)
                assert h.wait_registered(1)
                w.shutdown(cause=STOP_CAUSE_SIGNAL)
                t.join(timeout=10.0)
                kinds = [e["event"] for e in w.session_events]
                assert "TRANSPORT_LOSS" not in kinds, \
                    f"a STOP was classified as a transport loss: {kinds}"
            finally:
                WORKER.RangeMinerWorker._classify_session_end = orig
                w.shutdown()
                h.close()
    _mutant_red(_ignore_stop, "_stop-blind classification must misread a signal")


def g_a6_stop_defended_twice():
    """The second guard, asserted directly: even with a recoverable outcome handed
    to it, `_recover_session` refuses to attempt anything once `_stop` is set."""
    w = RangeMinerWorker(host="127.0.0.1", port=1, gpu_id=0, caps=VramCaps(**CAPS),
                         gpu_info=GpuInfo("cuda", "stub", 1024), hostname="h",
                         executor=_stub_executor(), recovery_budget_s=TEST_BUDGET)
    w.shutdown(cause=STOP_CAUSE_SIGNAL)
    outcome = WORKER.SessionOutcome(cause=SESSION_END_TRANSPORT_LOSS,
                                    exc_class="ConnectionError")
    assert outcome.recoverable
    assert w._recover_session(outcome) is False, \
        "recovery ran despite _stop being set"
    assert w.reconnect_attempts_total == 0, w.reconnect_attempts_total
    kinds = [e["event"] for e in w.session_events]
    assert "RECONNECT_ABANDONED" in kinds, kinds
    ab = [e for e in w.session_events if e["event"] == "RECONNECT_ABANDONED"][0]
    assert ab["reason"] == STOP_CAUSE_SIGNAL, ab


# ===========================================================================
# A7 — frozen-cohort wall (§13)
# ===========================================================================
def g_a7_same_identity_reconnects():
    """Same identity: the reconnect re-sends the frame the cohort was frozen
    against, byte-for-byte, and is admitted."""
    _set, wids = frozen_cohort()
    wid = wids[0]
    with tempfile.TemporaryDirectory() as tmp:
        spool = os.path.join(tmp, "spool")
        os.makedirs(spool, exist_ok=True)
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=1)
        w = _mk_worker(h.port, wid, spool)
        t, box = _spin(w)
        try:
            assert h.wait_registered(1)
            frozen_frame = dict(w._identity_frame)
            h.drop_transport(wid)
            assert h.wait_registered(1, timeout=TEST_BUDGET + 10.0)
            assert w._identity_frame == frozen_frame, "the identity frame drifted"
            assert h.eligible_ids() == [wid], h.eligible_ids()
            assert w.session_generation == 2, w.session_generation
        finally:
            w.shutdown(cause=STOP_CAUSE_EXPLICIT_SHUTDOWN)
            h.close()


def g_a7_changed_identity_fails_closed():
    """A reconnect whose CAPABILITY identity has drifted fails closed: it is not
    re-registered as something else, and the worker exits rather than entering the
    cohort under an identity it was not admitted under."""
    _set, wids = frozen_cohort()
    wid = wids[0]
    with tempfile.TemporaryDirectory() as tmp:
        spool = os.path.join(tmp, "spool")
        os.makedirs(spool, exist_ok=True)
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=1)
        w = _mk_worker(h.port, wid, spool)
        t, box = _spin(w)
        try:
            assert h.wait_registered(1)
            # the device identity changes under the worker between sessions
            w.gpu_info = GpuInfo("rocm", "different-gpu", 8 * 1024 ** 3)
            h.drop_transport(wid)
            t.join(timeout=TEST_BUDGET + 15.0)
            assert not t.is_alive(), "the worker did not fail closed"
            assert wid not in h.eligible_ids(), \
                "a drifted identity re-entered the cohort"
            end = [e for e in w.session_events
                   if e.get("classification") == SESSION_END_IDENTITY_REFUSED]
            assert end, [e["event"] for e in w.session_events]
            refused = [e for e in w.session_events
                       if e["event"] == "IDENTITY_REFUSED"]
            assert refused, w.session_events
            changed = refused[0]["changed_fields"]
            assert "backend" in changed and "gpu_name" in changed, changed
        finally:
            w.shutdown()
            h.close()


def g_a7_non_frozen_identity_fails_closed():
    """A worker the FROZEN SET does not name is refused admission — registered but
    quarantined, never eligible. Certified exec-set behaviour, gated here because
    a reconnect must not become a way in."""
    clear_execution_set()
    s, wids = frozen_cohort()
    try:
        freeze_execution_set(s)
        assert active_execution_set() is not None
        with tempfile.TemporaryDirectory() as tmp:
            spool = os.path.join(tmp, "spool")
            os.makedirs(spool, exist_ok=True)
            coord = _coord(tmp)
            h = _HarnessCoordinator(coord, expected_workers=1)
            stranger = "stranger-host:gpu0"
            assert stranger not in wids
            w = _mk_worker(h.port, stranger, spool)
            t, box = _spin(w)
            try:
                t0 = time.monotonic()
                while (time.monotonic() - t0 < DEADLINE
                       and stranger not in h.wconn_by_worker):
                    time.sleep(0.02)
                assert stranger in h.wconn_by_worker, "never reached registration"
                conn = h.wconn_by_worker[stranger]
                assert conn.quarantined is True, "a non-frozen worker was admitted"
                assert stranger not in h.eligible_ids(), h.eligible_ids()
                assert "NOT in the resolved execution set" in \
                    (conn.quarantine_reason or ""), conn.quarantine_reason
                # a member of the frozen set, by contrast, IS eligible
                member = _mk_worker(h.port, wids[0], spool)
                tm, _bm = _spin(member)
                try:
                    t0 = time.monotonic()
                    while (time.monotonic() - t0 < DEADLINE
                           and wids[0] not in h.eligible_ids()):
                        time.sleep(0.02)
                    assert wids[0] in h.eligible_ids(), h.eligible_ids()
                finally:
                    member.shutdown(cause=STOP_CAUSE_EXPLICIT_SHUTDOWN)
            finally:
                w.shutdown(cause=STOP_CAUSE_EXPLICIT_SHUTDOWN)
                h.close()
    finally:
        clear_execution_set()


def g_a7_mutant():
    """Red-first: the §13 wall is an EQUALITY on the frozen frame. If re-register
    rebuilt the frame instead of comparing it, drift would pass silently."""
    def _rebuild_instead_of_compare():
        _set, wids = frozen_cohort()
        with tempfile.TemporaryDirectory() as tmp:
            spool = os.path.join(tmp, "spool")
            os.makedirs(spool, exist_ok=True)
            coord = _coord(tmp)
            h = _HarnessCoordinator(coord, expected_workers=1)
            w = _mk_worker(h.port, wids[0], spool)
            try:
                w.connect()
                w.register()
                w.gpu_info = GpuInfo("rocm", "drifted", 1024)
                w._identity_frame = None      # THE MUTATION: forget the freeze
                w.register()                  # would now register as anything
                raise AssertionError("drifted identity re-registered")
            except WorkerIdentityChanged:
                pass
            finally:
                w.shutdown()
                h.close()
    _mutant_red(_rebuild_instead_of_compare,
                "forgetting the frozen frame must let drift through")


# ===========================================================================
# A8 — retry exhaustion (§14 bound)
# ===========================================================================
def g_a8_exhaustion_exits_cleanly():
    """A coordinator unavailable past the recovery bound: the worker exits
    CLEANLY, having made at least one real attempt, and does not haunt a later
    run."""
    _set, wids = frozen_cohort()
    wid = wids[0]
    with tempfile.TemporaryDirectory() as tmp:
        spool = os.path.join(tmp, "spool")
        os.makedirs(spool, exist_ok=True)
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=1)
        w = _mk_worker(h.port, wid, spool, budget=3.0)
        t, box = _spin(w)
        try:
            assert h.wait_registered(1)
            h.drop_transport(wid)
            h.close()                 # the coordinator is GONE from here on
            t0 = time.monotonic()
            t.join(timeout=60.0)
            elapsed = time.monotonic() - t0
            assert not t.is_alive(), "the worker outlived its recovery bound"
            assert box.get("returned") is True, box
            assert w.reconnect_attempts_total >= 1, w.reconnect_attempts_total
            kinds = [e["event"] for e in w.session_events]
            assert "RECONNECT_EXHAUSTED" in kinds, kinds
            ex = [e for e in w.session_events
                  if e["event"] == "RECONNECT_EXHAUSTED"][0]
            assert ex["reconnect_success"] is False, ex
            assert ex["recovery_budget_s"] == 3.0, ex
            assert ex["recovery_spent_s"] >= 3.0, ex
            # bounded: it exited near its bound, it did not linger
            assert elapsed < 40.0, elapsed
            assert w.conn is None, "a dead session was left open"
        finally:
            w.shutdown()
            h.close()


def g_a8_bound_derivation():
    """§14: the bound is POSITIVE FINITE and DERIVED from the existing liveness
    contract — the `worker_admission_timeout` 180 s authority — not invented, and
    the authority itself is unchanged (NO-TOUCH)."""
    assert DEFAULT_WORKER_ADMISSION_TIMEOUT == 180.0, DEFAULT_WORKER_ADMISSION_TIMEOUT
    assert default_recovery_budget_s() == DEFAULT_WORKER_ADMISSION_TIMEOUT, \
        default_recovery_budget_s()
    # the derivation is READ at runtime from the coordinator, not copied
    src = open(os.path.join(_ROOT, "miner", "range_miner_worker.py")).read()
    assert "from miner.range_miner_coordinator import DEFAULT_WORKER_ADMISSION_TIMEOUT" \
        in src, "the bound is no longer derived from the admission authority"
    w = RangeMinerWorker(host="127.0.0.1", port=1, gpu_id=0, caps=VramCaps(**CAPS),
                         gpu_info=GpuInfo("cuda", "stub", 1024),
                         hostname="h", executor=_stub_executor())
    assert w.recovery_budget_s() == 180.0, w.recovery_budget_s()
    # a non-positive or non-finite bound is refused, exactly as the coordinator
    # refuses a bad worker_admission_timeout
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        bw = RangeMinerWorker(host="127.0.0.1", port=1, gpu_id=0,
                              caps=VramCaps(**CAPS),
                              gpu_info=GpuInfo("cuda", "stub", 1024), hostname="h",
                              executor=_stub_executor(), recovery_budget_s=bad)
        try:
            bw.recovery_budget_s()
        except ValueError:
            continue
        raise AssertionError(f"a bound of {bad!r} must be refused")
    # backoff is bounded and fits many attempts inside the bound
    delays = [w._backoff_delay(i) for i in range(1, 12)]
    assert delays[0] == 1.0, delays
    assert max(delays) <= 180.0 / 12.0 + 1e-9, delays
    assert delays == sorted(delays), "backoff must be monotonic non-decreasing"


def g_a8_mutant():
    """Red-first: an UNBOUNDED recovery is the orphan §14 forbids. A budget that
    is not positive-finite must be refused, not silently accepted."""
    def _infinite_budget():
        w = RangeMinerWorker(host="127.0.0.1", port=1, gpu_id=0,
                             caps=VramCaps(**CAPS),
                             gpu_info=GpuInfo("cuda", "stub", 1024), hostname="h",
                             executor=_stub_executor(),
                             recovery_budget_s=float("inf"))
        assert w.recovery_budget_s() == float("inf"), "an infinite bound was refused"
    _mutant_red(_infinite_budget, "an infinite recovery bound must be refused")


# ===========================================================================
# §10/§15 structural gates
# ===========================================================================
def g_state_machine_three_way():
    """The §10 table, as implemented: three causes, three outcomes, one
    discriminator. Asserted on the classifier itself so the branch cannot quietly
    become two-way again."""
    w = RangeMinerWorker(host="127.0.0.1", port=1, gpu_id=0, caps=VramCaps(**CAPS),
                         gpu_info=GpuInfo("cuda", "stub", 1024), hostname="h",
                         executor=_stub_executor())
    exc = ConnectionError("peer went away")
    # _stop CLEAR -> transport loss, recoverable
    out = w._classify_session_end(exc)
    assert out.cause == SESSION_END_TRANSPORT_LOSS and out.recoverable, out
    # _stop SET by a shutdown frame -> NOT recoverable
    w._set_stop_cause(STOP_CAUSE_EXPLICIT_SHUTDOWN)
    w._stop.set()
    out = w._classify_session_end(exc)
    assert out.cause == STOP_CAUSE_EXPLICIT_SHUTDOWN and not out.recoverable, out
    # _stop SET by a signal -> NOT recoverable, and distinguishable
    w2 = RangeMinerWorker(host="127.0.0.1", port=1, gpu_id=0, caps=VramCaps(**CAPS),
                          gpu_info=GpuInfo("cuda", "stub", 1024), hostname="h",
                          executor=_stub_executor())
    w2.shutdown(cause=STOP_CAUSE_SIGNAL)
    out = w2._classify_session_end(exc)
    assert out.cause == STOP_CAUSE_SIGNAL and not out.recoverable, out
    # first-writer-wins: a later generic teardown cannot rewrite the real cause
    w2.shutdown()
    assert w2._stop_cause == STOP_CAUSE_SIGNAL, w2._stop_cause
    # every exception the certified READ loop caught is still classified as
    # transport, so classification did not narrow what recovery covers
    for e in (ConnectionError("x"), OSError("y"), ValueError("bad frame")):
        w3 = RangeMinerWorker(host="127.0.0.1", port=1, gpu_id=0,
                             caps=VramCaps(**CAPS),
                             gpu_info=GpuInfo("cuda", "stub", 1024), hostname="h",
                             executor=_stub_executor())
        assert w3._classify_session_end(e).recoverable, e
    # ...and the SEND surface is deliberately narrower: an oversized-frame
    # ValueError is a payload-contract violation, not a dead socket, so it must
    # NOT be swallowed into a reconnect.
    assert ValueError not in WORKER.SEND_TRANSPORT_EXCEPTIONS, \
        WORKER.SEND_TRANSPORT_EXCEPTIONS
    assert set(WORKER.SEND_TRANSPORT_EXCEPTIONS) <= set(WORKER.TRANSPORT_EXCEPTIONS)


def g_observability_fields():
    """§15: the required worker-side fields are present and NO high-rate heartbeat
    noise is emitted. A heartbeat must produce no session record."""
    _set, wids = frozen_cohort()
    wid = wids[0]
    with tempfile.TemporaryDirectory() as tmp:
        spool = os.path.join(tmp, "spool")
        os.makedirs(spool, exist_ok=True)
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=1)
        # a REAL heartbeat interval, so several heartbeats fly during the arm
        host, gpu = wid.rsplit(":gpu", 1)
        w = RangeMinerWorker(host="127.0.0.1", port=h.port, gpu_id=int(gpu),
                             caps=VramCaps(**CAPS), executor=_stub_executor(),
                             gpu_info=GpuInfo("cuda", "stub", 12 * 1024 ** 3),
                             hostname=host, heartbeat_interval=0.05,
                             miner_output_dir=spool, recovery_budget_s=TEST_BUDGET)
        t, box = _spin(w)
        try:
            assert h.wait_registered(1)
            time.sleep(0.5)                      # ~10 heartbeats
            assert w.session_events == [], \
                f"heartbeats emitted session noise: {w.session_events}"
            h.drop_transport(wid)
            assert h.wait_registered(1, timeout=TEST_BUDGET + 10.0)
            loss = [e for e in w.session_events if e["event"] == "TRANSPORT_LOSS"][0]
            for f in ("worker_id", "session_generation", "classification",
                      "exc_class", "assignment_active_at_loss", "recovery_spent_s"):
                assert f in loss, (f, loss)
            rec = [e for e in w.session_events if e["event"] == "RECONNECTED"][0]
            for f in ("attempt", "attempts_total", "reconnect_success",
                      "session_generation"):
                assert f in rec, (f, rec)
            assert loss["exc_class"] in ("ConnectionError", "OSError"), loss
        finally:
            w.shutdown(cause=STOP_CAUSE_EXPLICIT_SHUTDOWN)
            h.close()


def g_observability_unobserved_not_zero():
    """§15 + the S4 lesson: an UNMEASURED eligible count is reported UNOBSERVED,
    never as 0 — a genuinely empty pool is a different and real fact."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        h = _HarnessCoordinator(coord, expected_workers=1)
        try:
            with _SessionLog() as log:
                class _Msg:
                    message_type = "register"
                    worker_id = "hostA:gpu0"
                    hostname = "hostA"
                    gpu_id = 0
                    gpu_name = "stub"
                    backend = "cuda"
                    vram_bytes = 1024
                    capabilities = {"seed_caps": dict(CAPS),
                                    "supported_variants": list(VARIANTS)}
                sock = _FakeSock("unobserved")
                h.fs_by_sock[sock] = _FakeSock("fs")
                # no eligible_fn -> UNOBSERVED, not zero
                coord._serve_register(_Msg(), sock, None, h.fs_by_sock,
                                      h.worker_by_sock, h.wconn_by_worker,
                                      h.fs_by_worker, h.registered)
                reg = log.of("WORKER_REGISTERED")
                assert reg, log.events
                assert reg[-1]["obs_status"] == "UNOBSERVED", reg[-1]
                assert reg[-1]["eligible_count_after_register"] is None, reg[-1]
                # and an OBSERVED genuine zero still reports 0
                h.wconn_by_worker.clear()
                coord._drop_conn(sock, h.fs_by_sock, h.worker_by_sock,
                                 h.fs_by_worker, h.wconn_by_worker, h.registered,
                                 stage_idx=1, stage_assigned=True,
                                 eligible_fn=h.eligible)
                dis = log.of("WORKER_DISCONNECTED")
                assert dis, log.events
                assert dis[-1]["obs_status"] == "OBSERVED", dis[-1]
                assert dis[-1]["eligible_count_after_drop"] == 0, dis[-1]
                assert dis[-1]["stage_idx"] == 1, dis[-1]
                assert dis[-1]["stage_assigned"] is True, dis[-1]
        finally:
            h.close()


def g_no_touch_admission_surfaces():
    """NO-TOUCH (Beta §25) spot-check on the surfaces these arms border: the
    admission authority, the frozen-cohort authority and the lease authority are
    read, never redefined, by this amendment."""
    src = open(os.path.join(_ROOT, "miner", "range_miner_worker.py")).read()
    assert "DEFAULT_WORKER_ADMISSION_TIMEOUT =" not in src, \
        "the worker must not define its own admission timeout"
    assert "expected_workers" not in src, \
        "the worker must not touch expected_workers semantics"
    coord_src = open(os.path.join(_ROOT, "miner",
                                  "range_miner_coordinator.py")).read()
    assert "DEFAULT_WORKER_ADMISSION_TIMEOUT = 180.0" in coord_src, \
        "the admission authority moved or changed"
    # the eligible-pool rule is still LIVE registered connections, unchanged
    assert "return [w for w in wconn_by_worker.values() if not w.quarantined]" \
        in coord_src, "the live-eligibility rule changed"


# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    print("=" * 70)
    print("S172 DEFECT A — TRANSPORT-SESSION RECOVERY (Beta §§10-16, arms A1-A8)")
    print("=" * 70)

    print("\n-- §10 state machine + §15 observability --")
    _check("G-DA-STATE-MACHINE       three causes, one discriminator",
           g_state_machine_three_way)
    _check("G-DA-OBS-FIELDS          §15 fields, no heartbeat noise",
           g_observability_fields)
    _check("G-DA-OBS-UNOBSERVED      unmeasured is UNOBSERVED, never 0",
           g_observability_unobserved_not_zero)
    _check("G-DA-NO-TOUCH            admission/cohort/lease read, not redefined",
           g_no_touch_admission_surfaces)

    print("\n-- A1 / A2: the reproduction and its control (25 workers x 4 stages) --")
    _check("A1/RED red-first         pre-fix serve_forever loses the identity",
           g_a1_red_first)
    _check("A1  reproduction         2 idle losses recover, stage 4 admits 25",
           g_a1_reproduction)
    _check("A1/M mutation            no recovery -> pool stays 23", g_a1_mutant)
    _check("A2  control              23/25 refused, stage 4 NOT assigned",
           g_a2_no_downsizing)
    _check("A2/M mutation            23 must not satisfy a 25 expectation",
           g_a2_mutant)

    print("\n-- A3 / A4: F1/F2 semantics are not erased by reconnect --")
    _check("A3  constant-phase       active loss stays TERMINAL, no replay",
           g_a3_constant_phase_terminal)
    _check("A3/M mutation            a replayed completion is detected", g_a3_mutant)
    _check("A4  hybrid retry         consumed once, alternate executes it",
           g_a4_hybrid_retry_once)
    _check("A4/M mutation            retry budget stays exactly one", g_a4_mutant)

    print("\n-- A5 / A6: the §12 race and the reconnect-branch control --")
    _check("A5  duplicate race       rejected, singular, retried, then admitted",
           g_a5_duplicate_socket_race)
    _check("A5/M mutation            a duplicate binding must be refused",
           g_a5_mutant)
    _check("A6  explicit shutdown    exits with ZERO reconnect attempts",
           g_a6_explicit_shutdown_no_reconnect)
    _check("A6  signal               exits with ZERO reconnect attempts",
           g_a6_signal_no_reconnect)
    _check("A6  stop defended twice   _recover_session re-checks _stop",
           g_a6_stop_defended_twice)
    _check("A6/M mutation            _stop-blind classification misreads a signal",
           g_a6_mutant)

    print("\n-- A7 / A8: the frozen-cohort wall and the recovery bound --")
    _check("A7  same identity        reconnects on the frozen frame",
           g_a7_same_identity_reconnects)
    _check("A7  changed identity     fails closed", g_a7_changed_identity_fails_closed)
    _check("A7  non-frozen identity  registered but never eligible",
           g_a7_non_frozen_identity_fails_closed)
    _check("A7/M mutation            forgetting the freeze lets drift through",
           g_a7_mutant)
    _check("A8  exhaustion           exits cleanly at the bound",
           g_a8_exhaustion_exits_cleanly)
    _check("A8  bound derivation     positive finite, derived from 180 s anchor",
           g_a8_bound_derivation)
    _check("A8/M mutation            an infinite bound must be refused", g_a8_mutant)

    print("=" * 70)
    ok = sum(1 for _n, p, _t in _results if p)
    total = len(_results)
    for name, passed, tb in _results:
        if not passed:
            print(f"\n--- {name} ---\n{tb}")
    print(f"\n{ok}/{total} checks green")
    if ok == total:
        print("All checks green — S172 Defect A transport-session recovery "
              "(pending Team Beta review).")
    sys.exit(0 if ok == total else 1)
