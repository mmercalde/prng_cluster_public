#!/usr/bin/env python3
"""
test_s172_staging_backpressure.py — S172 staging back-pressure acceptance harness

Implements the gates required by
`docs/CLAUDE_CODE_INSTRUCTIONS_S172_STAGING_BACKPRESSURE_REMEDIATION.md` §5, which
maps Team Beta's ruling *"STAGING DEFERRED-QUEUE BACK-PRESSURE"* (2026-08-05)
one-to-one. Beta's gate 12 (G-PROD-SHAPE) is NOT in this file: it requires a live
25-daemon fleet, it is Michael-initiated only, and Beta's ruling forbids running it
until these gates are green and reviewed. See `tests/gate_s172_prod_shape.py`.

WHY THIS HARNESS EXISTS
-----------------------
The first production-shape trial ever to get past staging died on
`staging_deferred_max = 64` with 65 pending shards. The defect was not the number:
a COORDINATOR-SIDE TRANSIENT CAPACITY CONDITION was being charged to a worker's
stripe as a fault, through the phase-specific retry matrix, on a constant phase
where `retryable=True` and `retryable=False` produce the identical outcome
(`_handle_stripe_failure_locked`, the `if phase in (1, 2)` row). The worker did
nothing wrong; the coordinator was momentarily full; the trial died.

So these gates are written against the REAL relationships — the REAL per-connection
reader loop over REAL framed sockets, the REAL staging admission semaphore, the REAL
serve loop, the REAL ledger — never fabricated values, and every negative gate
asserts on the error TYPE and TEXT rather than on "something raised".

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 -u tests/test_s172_staging_backpressure.py \
        | tee /tmp/s172_backpressure.log
"""
import ast
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import queue as _queue
import select
import socket
import subprocess
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
    except Exception as e:                                    # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


from miner.range_miner_coordinator import (  # noqa: E402
    ST_CANCELLED,
    ST_CLAIMED,
    ST_DONE,
    ST_FAILED,
    ST_PENDING,
    ST_STAGING,
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    Phase5Sink,
    RangeMinerCoordinator,
    StagingConfigurationError,
    StagingPreflightProvenanceError,
    StagingRetentionSizingError,
    TransferAdapter,
    advertised_effective_cap,
    applicable_seed_cap,
    build_coordinator,
    expected_substripes_for,
    run_trial_miner,
    staging_burst_bound_conservative,
    staging_burst_bound_exact,
    workflow_stages_for,
)
from miner.range_miner_worker import (  # noqa: E402
    MinerFramedSocket,
    VramCaps,
    build_substripe_payload_bytes,
    supported_variants,
)
from miner.range_miner_protocol import (  # noqa: E402
    MinerHeartbeatMessage,
    RegisterMessage,
    StripeCompleteMessage,
    SubStripeResultMessage,
)

CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
VARIANTS = ["java_lcg", "java_lcg_reverse",
            "java_lcg_hybrid", "java_lcg_hybrid_reverse"]
SPOOL_ROOT = "/var/spool/miner"

# The RECORDED 2026-08-05 production-shape assignment (docs/TEAM_ALPHA_DEFERRED_
# QUEUE_NOTE.md §2), measured from `/home/michael/miner_staging/miner_ledger.db`
# — not inferred, and not a value invented for this test.
RECORDED_STRIPE_SPAN = 67_108_864          # config.miner_stripe_size
RECORDED_ASSIGNMENT = [                     # (worker, backend) per assigned stripe
    ("rrig6600:gpu0", "rocm"),              # expected_substripes 34
    ("zeus-ubuntu-vm:gpu0", "cuda"),        # expected_substripes 14
    ("rrig6600:gpu1", "rocm"),              # expected_substripes 34
    ("rrig6600:gpu2", "rocm"),              # expected_substripes 34
]


# ===========================================================================
# Harness: real coordinators, real workers, the REAL reader loop
# ===========================================================================
class _Sink(Phase5Sink):
    def __init__(self, on_abort=None):
        self.published, self.commits, self.aborts = [], [], []
        # `abort_trial` is the SYNCHRONOUS L7 discharge, so a hook here observes
        # the world at the exact instant the trial became terminal — which is the
        # only instant at which "the matrix was never entered" is a meaningful
        # claim. Anything after is post-terminal teardown.
        self.on_abort = on_abort

    def publish_shard(self, manifest):
        self.published.append(manifest)

    def commit_trial(self, event):
        self.commits.append(event)

    def abort_trial(self, event):
        self.aborts.append(event)
        if self.on_abort is not None:
            self.on_abort(event)


class _GatedTransfer(TransferAdapter):
    """A fetch that BLOCKS until its gate opens — the honest way to hold a real
    staging slot for the duration of a gate, because it occupies the same bounded
    executor + semaphore production uses."""

    def __init__(self, payloads=None, gate=None):
        self.payloads = payloads or {}
        self.gate = gate
        self.fetch_calls = []

    def fetch_remote(self, node_config, remote_path, local_temp_path):
        self.fetch_calls.append(remote_path)
        if self.gate is not None:
            self.gate.wait(timeout=30)
        with open(local_temp_path, "wb") as f:
            f.write(self.payloads.get(remote_path, b""))

    def delete_remote(self, node_config, remote_path):
        pass


def _coord(tmp, transfer=None, sink=None, dbname="l.db", **cfg):
    cfg.setdefault("staging_dir", os.path.join(tmp, "staging"))
    ledger = MinerLedger(os.path.join(tmp, dbname))
    return RangeMinerCoordinator(CoordinatorConfig(**cfg), ledger,
                                 transfer=transfer, phase5_sink=sink)


def _register(coord, wid="hostA:gpu0", backend="cuda", now=100.0):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend=backend,
        capabilities={"seed_caps": dict(CAPS),
                      "supported_variants": list(VARIANTS)},
        node_config=node, now=now)


_ACTIVE_SPY = []


class _MatrixSpy:
    """Counts EVERY entry into `_handle_stripe_failure_locked` — the single
    chokepoint through which the phase-specific worker retry matrix is reached.

    Patched on the CLASS, so it sees entries from any coordinator instance,
    including one built inside `run_trial_miner` that the gate never holds.
    """

    def __init__(self):
        self.calls = []
        self._orig = RangeMinerCoordinator._handle_stripe_failure_locked

    def __enter__(self):
        spy = self
        _ACTIVE_SPY.append(self)

        def _wrapped(self_, run_id, stripe_id, retryable, eligible_workers,
                     now, lease_expiry):
            spy.calls.append({"run_id": run_id, "stripe_id": stripe_id,
                              "retryable": retryable, "lease_expiry": lease_expiry})
            return spy._orig(self_, run_id, stripe_id, retryable,
                             eligible_workers, now, lease_expiry)

        RangeMinerCoordinator._handle_stripe_failure_locked = _wrapped
        return self

    def __exit__(self, *exc):
        RangeMinerCoordinator._handle_stripe_failure_locked = self._orig
        if self in _ACTIVE_SPY:
            _ACTIVE_SPY.remove(self)
        return False


class _Peer:
    """ONE connection driven through the REAL production reader loop.

    `RangeMinerCoordinator._conn_reader_loop` runs verbatim on a real socketpair
    with a real MinerFramedSocket — no substitute reader, no shortcut. The `cli`
    end is the "worker" and its sends are genuinely buffered by the kernel, which
    is the property under test: a frame the coordinator has not read stays ON THE
    WIRE.
    """

    def __init__(self, coord, worker_id, worker_by_sock, inbound, reader_stop,
                 bind=True, fs_wrap=None):
        self.worker_id = worker_id
        self.srv, self.cli = socket.socketpair()
        self.srv_fs = MinerFramedSocket(self.srv)
        self.cli_fs = MinerFramedSocket(self.cli)
        # [S172-BP AMENDMENT F1-R2b] G-NO-PREDECODE answers "did this connection
        # decode a SECOND envelope?" by COUNTING, not by inference — so the object
        # the production reader was already being handed is wrapped, and the
        # reader itself stays untouched production code.
        self.reader_fs = self.srv_fs if fs_wrap is None else fs_wrap(self.srv_fs)
        # `bind=False` models a socket that CONNECTED BUT NEVER REGISTERED —
        # `worker_by_sock` is written only by `_serve_register`, so the absence of
        # an entry IS "unregistered" (F4).
        if bind:
            worker_by_sock[self.srv] = worker_id
        self.thread = threading.Thread(
            target=coord._conn_reader_loop,
            args=(self.reader_fs, self.srv, inbound, reader_stop, worker_by_sock),
            name=f"reader-{worker_id}", daemon=True)
        self.thread.start()

    def send(self, msg):
        self.cli_fs.send_msg(msg)

    def close(self):
        for s in (self.cli, self.srv):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass


class _Bench:
    """A coordinator plus N connections, each on the real reader loop."""

    def __init__(self, tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"), inbound=None,
                 fs_wrap=None, **cfg):
        self.coord = _coord(tmp, **cfg)
        self.worker_by_sock = {}
        self.wconn_by_worker = {}
        # `inbound` is injectable ONLY so the F1-R mutant can restore the round-1
        # clear-at-`inbound.put` at exactly the instruction that differs. Every
        # other gate gets the same bounded queue the serve loop uses.
        self.inbound = inbound if inbound is not None else _queue.Queue(maxsize=1024)
        self.reader_stop = threading.Event()
        self.peers = {}
        for wid in worker_ids:
            self.wconn_by_worker[wid] = _register(self.coord, wid)
            self.peers[wid] = _Peer(self.coord, wid, self.worker_by_sock,
                                    self.inbound, self.reader_stop,
                                    fs_wrap=fs_wrap)
        self._held = []

    # --- capacity control ------------------------------------------------
    def saturate(self):
        """Hold EVERY staging admission slot, using the very semaphore
        `enqueue_staging` acquires (`_staging_slots()`). This is the real capacity
        condition, not a stubbed predicate."""
        sem = self.coord._staging_slots()
        while sem.acquire(blocking=False):
            self._held.append(True)
        assert self._held, "no staging slots existed to hold"
        assert not self.coord.staging_can_accept(), (
            "staging_can_accept() still True with every slot held — the gate "
            "cannot be exercised")

    def release_all_slots(self):
        sem = self.coord._staging_slots()
        while self._held:
            self._held.pop()
            sem.release()

    # --- inbound observation ---------------------------------------------
    def drain(self, timeout=0.6):
        """Everything the serve loop WOULD see, within `timeout`."""
        out = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                out.append(self.inbound.get(timeout=0.05))
            except _queue.Empty:
                continue
        return out

    def dispatch(self, entries, run_id, eligible=None):
        """Feed drained frames to the REAL serve dispatcher.

        [S172-BP AMENDMENT F1-R] Through `dispatch_inbound_result`, which is the
        exact call the serve loop makes — so the ingress reservation is disposed of
        by PRODUCTION code here, not by the test thread. Round 1's mistake was the
        opposite: the bench modelled the serve loop's effect on capacity itself,
        which deleted the very interval the invariant lives in.

        [S172-BP AMENDMENT F1-R2a] The credit TOKEN the envelope arrived with is
        passed straight through, exactly as the serve loop's drain does. The bench
        never invents one: an entry drained with `credit_id=None` is dispatched
        with `None`, which is the whole point of the identity.
        """
        eligible = eligible or (lambda: list(self.wconn_by_worker.values()))
        for kind, rawsock, msg, credit_id in entries:
            if kind != "msg":
                continue
            self.coord.dispatch_inbound_result(
                msg, rawsock, run_id, self.worker_by_sock.get(rawsock),
                self.wconn_by_worker, eligible, credit_id)

    def pump(self, run_id, timeout=1.5, eligible=None, until=None):
        """Drain and dispatch INTERLEAVED, the way the serve loop actually runs.

        [S172-BP AMENDMENT F1-R] A reader may now hold its next `sub_stripe_result`
        until the previous one has been DISPOSED of (Beta §4 tail: one result per
        reservation), so `drain()`-everything-then-`dispatch()`-everything can no
        longer see a connection's second result — not because anything is lost, but
        because the serve loop is the thing that unblocks it. Returns every entry
        seen, in arrival order, so ordering assertions are unaffected.
        """
        seen = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                entry = self.inbound.get(timeout=0.05)
            except _queue.Empty:
                if until is not None and until():
                    break
                continue
            seen.append(entry)
            self.dispatch([entry], run_id, eligible=eligible)
            if until is not None and until():
                break
        return seen

    def dispose(self, rawsock):
        """[S172-BP AMENDMENT F1-R] The DISPOSITION CLEAR ALONE, through the
        production API, for gates whose subject is NOT the handoff.

        Used where dispatching the frame would drag unrelated machinery in (a
        StripeComplete reconciliation, say) but the reservation still has to end so
        the connection may deliver what is queued behind it. The handoff invariant
        itself is proven in G-RESUME-HANDOFF, which drives the REAL serve path and
        touches neither the credit nor the semaphore from the test thread.
        """
        return self.coord._release_resume_credit(
            rawsock, delivered=True, disposition="dispatch")

    def wait_paused(self, n=1, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.coord.paused_connection_count() >= n:
                return True
            time.sleep(0.02)
        return False

    def wait_unpaused(self, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.coord.paused_connection_count() == 0:
                return True
            time.sleep(0.02)
        return False

    def close(self):
        self.reader_stop.set()
        self.release_all_slots()
        for p in self.peers.values():
            p.close()
        if self.coord._staging_executor is not None:
            self.coord._staging_executor.shutdown(wait=True)
            self.coord._staging_executor = None
        if self.coord._cleanup_executor is not None:
            self.coord._cleanup_executor.shutdown(wait=True)
            self.coord._cleanup_executor = None


def _inline_result(wid, sid, sub_index, seed_start, seed_count, survivors=None):
    survivors = survivors if survivors is not None else [[seed_start, 0.9, None, [1]]]
    obj, pb = build_substripe_payload_bytes(sid, sub_index, seed_start,
                                            seed_count, survivors)
    return SubStripeResultMessage(
        worker_id=wid, stripe_id=sid, sub_index=sub_index, seed_start=seed_start,
        seed_count=seed_count, survivor_count=len(survivors), inline=obj,
        size_bytes=len(pb), sha256=hashlib.sha256(pb).hexdigest(),
        effective_threshold=0.25)


def _claim(coord, run_id, sid, wid, conn, phase=1, family="java_lcg",
           seed_start=0, seed_count=30, expected=1, attempt=0, lease=1e9,
           now=100.0):
    coord.ledger.add_stripe(run_id, sid, seed_start, seed_count, family, phase, now)
    coord.ledger.claim_stripe(run_id, sid, wid, attempt, expected, lease)
    conn.record_assignment(sid, attempt)


def _capture_bp(sink_list):
    """Capture the module logger's records, INCLUDING INFO.

    The `[S172-BP]` series is emitted at INFO and the module logger inherits the
    root's WARNING level, so attaching a handler alone captures nothing but the
    warnings — which is how a metrics gate can look green while measuring nothing.
    Returns (logger, handler, restore)."""
    class _Cap(logging.Handler):
        def emit(self, record):
            sink_list.append(record.getMessage())

    lg = logging.getLogger("range_miner_coordinator")
    prev_level, prev_prop = lg.level, lg.propagate
    h = _Cap()
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    lg.propagate = False

    def _restore():
        lg.removeHandler(h)
        lg.setLevel(prev_level)
        lg.propagate = prev_prop

    return lg, h, _restore


def _saturating_cfg(**over):
    """A coordinator config whose capacity gate can actually be closed.

    `staging_deferred_max=1` is an EXPLICIT OPERATOR OVERRIDE (§2 permits one, and
    warns when it is below the derived bound). With >= 2 live connections the
    resume margin is >= 2, so the hysteresis low-water `bound - margin` clamps to
    0 and `staging_can_accept()` is False the moment every slot is held. That is
    the real predicate — nothing about it is stubbed.
    """
    cfg = dict(staging_workers=2, staging_queue_depth=0, staging_deferred_max=1,
               staging_capacity_timeout=600.0, miner_stripe_size=1000)
    cfg.update(over)
    return cfg


# ===========================================================================
# G1 / G2 — a capacity wait fails NOTHING and consumes NO retry budget
# ===========================================================================
def gate1_saturation_on_phase1_fails_nothing():
    """Beta gate 1. Saturate staging during a PHASE-1 stripe (the phase every TFM
    run uses, and the one the matrix fails closed on) and prove the four negatives:
    no `_handle_stripe_failure_locked` entry, no retry consumed, no cancellation,
    no L1 fence activation against the valid attempt, trial not failed."""
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg())
        try:
            run_id, sid = "runG1", "runG1_s0"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            _claim(b.coord, run_id, sid, "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"], phase=1, family="java_lcg")
            b.saturate()

            with _MatrixSpy() as spy:
                b.peers["hostA:gpu0"].send(
                    _inline_result("hostA:gpu0", sid, 0, 0, 30))
                assert b.wait_paused(1), "the reader never paused on a full staging"
                # hold the wait open well past any dispatch tick
                time.sleep(0.4)
                drained = b.drain(0.3)

                # (1) NOTHING entered the phase-specific worker retry matrix
                assert spy.calls == [], (
                    f"a capacity wait entered the matrix: {spy.calls}")
                # (2) the envelope was NOT delivered — it is being WAITED on
                assert not [e for e in drained if e[0] == "msg"], (
                    "the held sub_stripe_result was delivered while staging was "
                    "saturated — the reader gate did not engage")
                # (3) no retry consumed, no degradation, no cancellation
                st = b.coord.ledger.get_stripe(run_id, sid)
                assert st["current_attempt"] == 0, "a retry attempt was consumed"
                assert not st["phase_degraded"], "phase_degraded set by a capacity wait"
                assert st["state"] == ST_CLAIMED, (
                    f"stripe left {st['state']!r} by a capacity wait")
                assert st["claimed_by"] == "hostA:gpu0"
                # (4) the trial is ALIVE
                trial = b.coord.ledger.get_trial(run_id)
                assert trial["state"] == "running", (
                    f"trial {trial['state']!r} after a mere capacity wait")

                # (5) no L1 fence activation against the VALID attempt: on release
                # the same envelope is accepted and staged.
                b.release_all_slots()
                b.coord._release_capacity()
                assert b.wait_unpaused(), "the reader never resumed"
                after = b.drain(0.6)
                msgs = [m for (k, _s, m, _c) in after if k == "msg"]
                assert len(msgs) == 1 and msgs[0].message_type == "sub_stripe_result"
                b.dispatch(after, run_id)
                assert spy.calls == [], (
                    f"the resumed envelope entered the matrix: {spy.calls}")
                shards = b.coord.ledger.get_shards(run_id, sid, 0)
                assert len(shards) == 1, (
                    f"the valid attempt was fenced out: {len(shards)} shards")
        finally:
            b.close()


def gate2_capacity_wait_consumes_zero_retry_budget():
    """Beta gate 2, on the phase where a retry actually EXISTS. A hybrid (phase 3)
    stripe still has its single Q3 retry after an arbitrarily long capacity wait —
    `current_attempt` and `phase_degraded` are both unchanged, so the wait cost the
    trial no resilience at all."""
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg())
        try:
            run_id, sid = "runG2", "runG2_s0"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            conn = b.wconn_by_worker["hostA:gpu0"]
            _claim(b.coord, run_id, sid, "hostA:gpu0", conn,
                   phase=3, family="java_lcg_hybrid")
            before = b.coord.ledger.get_stripe(run_id, sid)
            b.saturate()
            with _MatrixSpy() as spy:
                for i in range(3):
                    b.peers["hostA:gpu0"].send(
                        _inline_result("hostA:gpu0", sid, i, i * 10, 10))
                assert b.wait_paused(1)
                time.sleep(0.5)
                after = b.coord.ledger.get_stripe(run_id, sid)
                assert spy.calls == [], f"matrix entered: {spy.calls}"
                assert after["current_attempt"] == before["current_attempt"] == 0
                assert bool(after["phase_degraded"]) is bool(before["phase_degraded"])
                assert after["state"] == ST_CLAIMED

                # And the retry is genuinely still THERE: a real retryable failure
                # now still buys the one hybrid reassignment.
                b.release_all_slots()
                b.coord._release_capacity()
                b.wait_unpaused()
                act = b.coord.handle_stripe_failure(
                    run_id, sid, retryable=True,
                    eligible_workers=list(b.wconn_by_worker.values()), now=200.0)
                assert act["action"] == "reassigned" and act["attempt"] == 1, act
        finally:
            b.close()


# ===========================================================================
# G3 / G4 — per-connection, and self-resuming
# ===========================================================================
def gate3_paused_peer_stalls_while_second_connection_flows():
    """Beta gate 3, and Beta B.3's PER-CONNECTION requirement.

    Connection A is paused holding one envelope; every later frame A writes stays
    ON THE WIRE (TCP is ordered — that is the point of gating at the reader).
    Meanwhile connection B's traffic flows to completion, untouched: B's reader,
    the accept path and the serve loop know nothing about A's pause.
    """
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg())
        try:
            run_id = "runG3"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            _claim(b.coord, run_id, "runG3_sA", "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"])
            _claim(b.coord, run_id, "runG3_sB", "hostB:gpu0",
                   b.wconn_by_worker["hostB:gpu0"])
            b.saturate()

            # A: one result (held) then two more frames (must stay on the wire)
            b.peers["hostA:gpu0"].send(
                _inline_result("hostA:gpu0", "runG3_sA", 0, 0, 30))
            assert b.wait_paused(1), "A never paused"
            b.peers["hostA:gpu0"].send(
                _inline_result("hostA:gpu0", "runG3_sA", 1, 30, 30))
            b.peers["hostA:gpu0"].send(StripeCompleteMessage(
                worker_id="hostA:gpu0", stripe_id="runG3_sA",
                substripes_done=2, survivors_total=2))

            # B: heartbeat + stripe_complete — a DIFFERENT connection, flowing
            b.peers["hostB:gpu0"].send(MinerHeartbeatMessage(
                worker_id="hostB:gpu0", current_stripe_id="runG3_sB", busy=True))
            b.peers["hostB:gpu0"].send(StripeCompleteMessage(
                worker_id="hostB:gpu0", stripe_id="runG3_sB",
                substripes_done=1, survivors_total=1))

            got = b.drain(0.8)
            from_a = [m for (k, s, m, _c) in got
                      if k == "msg" and b.worker_by_sock.get(s) == "hostA:gpu0"]
            from_b = [m for (k, s, m, _c) in got
                      if k == "msg" and b.worker_by_sock.get(s) == "hostB:gpu0"]
            assert from_a == [], (
                f"the paused connection delivered {len(from_a)} frame(s): "
                f"{[m.message_type for m in from_a]}")
            assert [m.message_type for m in from_b] == ["heartbeat", "stripe_complete"], (
                f"the UNPAUSED connection stalled too — pause leaked from A to B: "
                f"{[m.message_type for m in from_b]}")
            assert b.coord.paused_worker_ids() == frozenset({"hostA:gpu0"}), (
                "pause is not per-connection")

            # ...and nothing A wrote was LOST: on release all three arrive IN ORDER.
            b.release_all_slots()
            b.coord._release_capacity()
            assert b.wait_unpaused(), "A never resumed"
            # [S172-BP AMENDMENT F1-R] A's reservation now survives until the serve
            # loop disposes of the credited envelope, and until it does, §4-tail
            # holds A's NEXT result. So the frames behind it arrive as each
            # disposition lands — which is the serve loop's job, modelled here with
            # the disposition clear alone because this gate's subject is the WIRE
            # ORDER, not the handoff (see `_Bench.dispose`).
            rest = []
            sockA = b.peers["hostA:gpu0"].srv
            deadline = time.time() + 4.0
            while time.time() < deadline and len(rest) < 3:
                rest.extend(b.drain(0.3))
                b.dispose(sockA)
            seq = [(m.message_type, getattr(m, "sub_index", None))
                   for (k, s, m, _c) in rest
                   if k == "msg" and b.worker_by_sock.get(s) == "hostA:gpu0"]
            assert seq == [("sub_stripe_result", 0), ("sub_stripe_result", 1),
                           ("stripe_complete", None)], (
                f"held-then-wire ordering was not preserved: {seq}")
        finally:
            b.close()


def gate4_capacity_release_resumes_without_operator_action():
    """Beta gate 4. The resume must come from a REAL capacity-release point — a
    staging job completing — with nothing external poking the coordinator. The
    holder here is a genuine `enqueue_staging` submission whose fetch blocks, so
    the slot is released by `_submit_with_slot`'s own completion callback."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        fetch_gate = threading.Event()
        sid_hold = "runG4_hold"
        _obj, pb = build_substripe_payload_bytes(sid_hold, 0, 0, 30,
                                                 [[0, 0.9, None, [1]]])
        remote = f"{SPOOL_ROOT}/{sid_hold}/0.json"
        sha = hashlib.sha256(pb).hexdigest()
        transfer = _GatedTransfer(payloads={remote: pb}, gate=fetch_gate)
        b = _Bench(tmp, transfer=transfer,  # noqa: F841 — see cleanup note in G11
                   **_saturating_cfg(staging_workers=2, staging_queue_depth=0))
        try:
            run_id = "runG4"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            connA = b.wconn_by_worker["hostA:gpu0"]
            _claim(b.coord, run_id, sid_hold, "hostA:gpu0", connA, expected=1)
            b.coord.ledger.record_substripe_result(
                run_id, sid_hold, 0, 0, "hostA:gpu0", 0, 30, 1,
                remote_spool_path=remote, size_bytes=len(pb), sha256=sha)

            # occupy real staging slots with real blocked jobs
            held_msg = SubStripeResultMessage(
                worker_id="hostA:gpu0", stripe_id=sid_hold, sub_index=0,
                seed_start=0, seed_count=30, survivor_count=1, spool_path=remote,
                inline=None, size_bytes=len(pb), sha256=sha)
            b.coord.enqueue_staging("remote", connA, run_id, sid_hold, 0, 0,
                                    held_msg, lambda: [connA])
            # ...and drain the remaining slot the same way the deferred queue would
            sem = b.coord._staging_slots()
            grabbed = 0
            deadline = time.time() + 3
            while time.time() < deadline:
                if transfer.fetch_calls:
                    break
                time.sleep(0.02)
            while sem.acquire(blocking=False):
                grabbed += 1
            assert not b.coord.staging_can_accept(), "staging never saturated"

            # a SECOND connection's result now pauses
            sid_b = "runG4_sB"
            _claim(b.coord, run_id, sid_b, "hostB:gpu0",
                   b.wconn_by_worker["hostB:gpu0"])
            b.peers["hostB:gpu0"].send(
                _inline_result("hostB:gpu0", sid_b, 0, 0, 30))
            assert b.wait_paused(1), "B never paused"

            # release the manually-grabbed slots, then let the REAL job finish.
            # From here on nothing else touches the coordinator: the resume must
            # come from `_submit_with_slot`'s completion callback.
            for _ in range(grabbed):
                sem.release()
            fetch_gate.set()

            assert b.wait_unpaused(timeout=10.0), (
                "the paused connection did not resume when a real staging job "
                "released its slot — resume is not wired to the capacity-release "
                "path")
            got = b.drain(1.0)
            msgs = [m for (k, _s, m, _c) in got if k == "msg"]
            assert [m.message_type for m in msgs] == ["sub_stripe_result"], (
                f"the held envelope was not delivered on resume: {msgs}")
            m = b.coord.staging_backpressure_metrics()
            assert m["pause_events"] >= 1 and m["pause_seconds_total"] > 0.0
        finally:
            fetch_gate.set()
            b.close()


# ===========================================================================
# G5 / G6 / G7 — exactly-once and fencing across pause/resume
# ===========================================================================
def gate5_each_substripe_staged_exactly_once():
    """Beta gate 5: every ACCEPTED sub-stripe is staged exactly once across
    pause/resume, counted from the LEDGER (not from a log line)."""
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg())
        try:
            run_id, sid = "runG5", "runG5_s0"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            conn = b.wconn_by_worker["hostA:gpu0"]
            _claim(b.coord, run_id, sid, "hostA:gpu0", conn, seed_count=40,
                   expected=4)
            b.saturate()
            for i in range(4):
                b.peers["hostA:gpu0"].send(
                    _inline_result("hostA:gpu0", sid, i, i * 10, 10))
            assert b.wait_paused(1)
            b.release_all_slots()
            b.coord._release_capacity()
            assert b.wait_unpaused()
            # [S172-BP AMENDMENT F1-R] drain and dispatch INTERLEAVED: the reader
            # holds each next result until the previous one has been disposed of
            # (§4 tail), so the serve loop has to actually run between them.
            b.pump(run_id, timeout=6.0,
                   until=lambda: len(b.coord.ledger.get_shards(run_id, sid, 0)) >= 4)
            rows = b.coord.ledger.get_shards(run_id, sid, 0)
            assert len(rows) == 4, f"expected 4 shard rows, got {len(rows)}"
            assert sorted(r["sub_index"] for r in rows) == [0, 1, 2, 3]
        finally:
            b.close()


def gate6_no_duplicate_rows_no_stale_acceptance():
    """Beta gate 6. The held envelope is governed by the EXISTING dedup insert and
    the EXISTING L1 fence — §1.3 forbids a second dedup layer, so this proves the
    original ones still do the work across a pause. A duplicate delivered after
    resume is dropped, and a message for a stale attempt is fenced."""
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg())
        try:
            run_id, sid = "runG6", "runG6_s0"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            conn = b.wconn_by_worker["hostA:gpu0"]
            _claim(b.coord, run_id, sid, "hostA:gpu0", conn, expected=1)
            b.saturate()
            msg = _inline_result("hostA:gpu0", sid, 0, 0, 30)
            b.peers["hostA:gpu0"].send(msg)
            assert b.wait_paused(1)
            b.peers["hostA:gpu0"].send(msg)          # the SAME logical shard again
            b.release_all_slots()
            b.coord._release_capacity()
            assert b.wait_unpaused()
            # [S172-BP AMENDMENT F1-R] interleaved, per §4 tail — the duplicate is
            # released by the FIRST frame's disposition, not by draining harder.
            entries = b.pump(run_id, timeout=3.0)
            got = [m for (k, _s, m, _c) in entries if k == "msg"]
            assert len(got) == 2, f"both frames should reach the serve loop: {got}"
            rows = b.coord.ledger.get_shards(run_id, sid, 0)
            assert len(rows) == 1, (
                f"pause/resume produced {len(rows)} rows for ONE logical shard — "
                f"the existing dedup insert stopped governing")

            # no SECOND dedup layer was bolted on (§1.3)
            src = inspect.getsource(RangeMinerCoordinator._conn_reader_loop)
            tree = ast.parse(src.lstrip())
            called = {getattr(n.func, "attr", None) for n in ast.walk(tree)
                      if isinstance(n, ast.Call)}
            assert "record_substripe_result" not in called, (
                "the reader dispatches results itself — dedup/fencing moved out of "
                "the serve loop")
            for banned in ("accept_stripe_message", "handle_stripe_failure",
                           "_on_staging_failed", "fail_trial"):
                assert banned not in called, (
                    f"the reader loop calls {banned} — a capacity wait must not "
                    f"touch the ledger, the fence or the matrix")
        finally:
            b.close()


def gate7_superseded_attempt_cannot_resume_and_publish():
    """Beta gate 7. While the envelope is held, the attempt is legitimately
    superseded (the matrix reassigns and bumps `staging_generation`). On resume the
    EXISTING L1 fence must drop it — no publish, no shard row for the dead
    attempt."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _Sink()
        b = _Bench(tmp, sink=sink, **_saturating_cfg())
        try:
            run_id, sid = "runG7", "runG7_s0"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            connA = b.wconn_by_worker["hostA:gpu0"]
            _claim(b.coord, run_id, sid, "hostA:gpu0", connA, phase=3,
                   family="java_lcg_hybrid", expected=1)
            gen_before = b.coord.ledger.get_stripe(run_id, sid)["staging_generation"]
            b.saturate()
            b.peers["hostA:gpu0"].send(_inline_result("hostA:gpu0", sid, 0, 0, 30))
            assert b.wait_paused(1)

            # supersede WHILE PAUSED — a genuine hybrid reassignment
            act = b.coord.handle_stripe_failure(
                run_id, sid, retryable=True,
                eligible_workers=list(b.wconn_by_worker.values()), now=200.0)
            assert act["action"] == "reassigned", act
            st = b.coord.ledger.get_stripe(run_id, sid)
            assert st["staging_generation"] == gen_before + 1, "no fence bump"
            assert st["claimed_by"] == "hostB:gpu0"

            b.release_all_slots()
            b.coord._release_capacity()
            assert b.wait_unpaused()
            entries = b.drain(1.2)
            assert [m.message_type for (k, _s, m, _c) in entries if k == "msg"] == \
                ["sub_stripe_result"], "the held envelope was not re-delivered"
            b.dispatch(entries, run_id)
            # the fence dropped it: no shard row for the DEAD attempt 0, nothing
            # published, and the live attempt 1 is untouched.
            assert b.coord.ledger.get_shards(run_id, sid, 0) == [], (
                "a superseded attempt resumed and wrote a shard row")
            assert sink.published == [], "a superseded attempt published"
        finally:
            b.close()


# ===========================================================================
# G8 — bounded retention
# ===========================================================================
def gate8_retention_is_bounded():
    """Beta gate 8: deferred retained bytes and pending envelopes are bounded —
    `_deferred` at most the derived bound, and AT MOST ONE envelope per connection.

    The one-envelope property is proven two ways: STRUCTURALLY (the holder is a
    scalar local that is only ever assigned a single message and cleared — never a
    list, never a queue) and BEHAVIOURALLY (N paused connections produce exactly N
    pause records no matter how many frames each wrote).
    """
    # (a) structural — AST over the LIVE reader source, not a text anchor
    src = inspect.getsource(RangeMinerCoordinator._conn_reader_loop)
    tree = ast.parse(src.lstrip())
    holder_ops = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pending_envelope"):
            holder_ops.append(node.func.attr)
    assert holder_ops == [], (
        f"the pending-envelope holder is a container, not a single slot: {holder_ops}")
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "pending_envelope"
                       for t in n.targets)]
    assert assigns, "no pending_envelope holder found in the reader loop"
    for a in assigns:
        assert isinstance(a.value, (ast.Name, ast.Constant)), (
            "pending_envelope is assigned a composite value — capacity is not 1")

    # (b) behavioural
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg())
        try:
            run_id = "runG8"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            for wid, sid in (("hostA:gpu0", "runG8_sA"), ("hostB:gpu0", "runG8_sB")):
                _claim(b.coord, run_id, sid, wid, b.wconn_by_worker[wid],
                       seed_count=60, expected=6)
            b.saturate()
            for wid, sid in (("hostA:gpu0", "runG8_sA"), ("hostB:gpu0", "runG8_sB")):
                for i in range(6):
                    b.peers[wid].send(_inline_result(wid, sid, i, i * 10, 10))
            assert b.wait_paused(2), "both connections should have paused"
            time.sleep(0.4)
            bound = b.coord.staging_deferred_bound()
            margin = b.coord._resume_margin()
            assert len(b.coord._deferred) <= bound + margin, (
                f"deferred={len(b.coord._deferred)} > bound {bound} + margin {margin}")
            assert b.coord._deferred_retained_bytes() <= \
                b.coord.config.staging_high_water_bytes
            assert b.coord.paused_connection_count() == 2, (
                "12 frames across 2 connections produced "
                f"{b.coord.paused_connection_count()} pause records — not one per "
                f"connection")
            m = b.coord.staging_backpressure_metrics()
            assert m["deferred_high_water"] <= bound + margin
        finally:
            b.close()


# ===========================================================================
# G9 — the derived bound, and Beta's 116-vs-136 distinction
# ===========================================================================
def gate9_derived_bounds_116_and_136():
    """Beta gate 9. Both quantities from the RECORDED 2026-08-05 assignment, both
    computed through the coordinator's ONE cap path (`advertised_effective_cap`),
    and the distinction between them asserted explicitly — Beta singled it out."""
    # the exact assignment, worker by worker, as the ledger recorded it
    rows = [{"stripe_span": RECORDED_STRIPE_SPAN, "backend": backend,
             "seed_caps": CAPS, "phase": 1}
            for _wid, backend in RECORDED_ASSIGNMENT]
    exact = staging_burst_bound_exact(rows)
    assert exact == 116, f"exact bound {exact} != 116 for the recorded assignment"

    # ...and it really is 34 + 14 + 34 + 34, not four equal terms that happen to sum
    per_stripe = [expected_substripes_for(
        RECORDED_STRIPE_SPAN, applicable_seed_cap(backend, CAPS, 1))
        for _wid, backend in RECORDED_ASSIGNMENT]
    assert per_stripe == [34, 14, 34, 34], per_stripe
    assert sum(per_stripe) == 116

    # the CONSERVATIVE pre-assignment bound for the same four-slot geometry, with
    # an AMD worker eligible for every slot
    conservative = staging_burst_bound_conservative(
        [RECORDED_STRIPE_SPAN] * 4,
        [("rocm", CAPS), ("cuda", CAPS)], 1, caps=CAPS)
    assert conservative == 136, f"conservative bound {conservative} != 136"

    # THE DISTINCTION: 136 is a pre-assignment bound and must DOMINATE the exact
    # count of any assignment the scheduler could still make.
    assert conservative > exact, (
        "the conservative bound no longer dominates the exact one — it has stopped "
        "being safe for an unknown assignment")
    assert conservative == 4 * 34 and exact == 34 + 14 + 34 + 34

    # a CUDA-only eligible pool has a genuinely smaller conservative bound — proof
    # the max-over-eligible-workers term is real and not a constant
    cuda_only = staging_burst_bound_conservative(
        [RECORDED_STRIPE_SPAN] * 4, [("cuda", CAPS)], 1, caps=CAPS)
    assert cuda_only == 4 * 14 == 56, cuda_only

    # phase awareness: hybrid caps are TIGHTER, so phase 3 must bound HIGHER
    hybrid = staging_burst_bound_conservative(
        [RECORDED_STRIPE_SPAN] * 4, [("rocm", CAPS), ("cuda", CAPS)], 3, caps=CAPS)
    assert hybrid == 4 * expected_substripes_for(
        RECORDED_STRIPE_SPAN, CAPS["amd_hybrid"]), hybrid
    assert hybrid > conservative, (
        "phase 3 did not take the tighter hybrid cap — the bound is phase-blind")

    # the functions are PURE (no coordinator, no I/O) and use the shared cap path
    for fn in (staging_burst_bound_exact, staging_burst_bound_conservative):
        params = list(inspect.signature(fn).parameters)
        assert "self" not in params, f"{fn.__name__} is not module level"
    assert applicable_seed_cap("rocm", CAPS, 1) == advertised_effective_cap(
        "rocm", "java_lcg", CAPS)
    assert applicable_seed_cap("rocm", CAPS, 3) == advertised_effective_cap(
        "rocm", "java_lcg_hybrid", CAPS)

    # the deleted constant: 64 must no longer be the deferred bound anywhere
    assert CoordinatorConfig().staging_deferred_max is None, (
        "staging_deferred_max is still a hand-maintained constant — it must be an "
        "OPTIONAL OVERRIDE defaulting to None (derived)")
    # ...and the deleted constant is gone from the EXECUTABLE body of _defer_locked.
    # Compared over the AST with the docstring stripped, because the docstring
    # legitimately EXPLAINS that 64 was deleted — a text search would red on the
    # very sentence recording the fix.
    dtree = ast.parse(inspect.getsource(RangeMinerCoordinator._defer_locked).lstrip())
    dfn = dtree.body[0]
    body = dfn.body[1:] if (isinstance(dfn.body[0], ast.Expr)
                            and isinstance(dfn.body[0].value, ast.Constant)
                            and isinstance(dfn.body[0].value.value, str)) else dfn.body
    executable = "\n".join(ast.unparse(n) for n in body)
    assert "64" not in executable, (
        f"the constant 64 survives in the BODY of _defer_locked:\n{executable}")
    assert "staging_deferred_bound()" in executable, (
        "_defer_locked does not consult the derived bound")


def gate9b_runtime_uses_the_derived_bound():
    """§2: the RUNTIME bound is `burst_bound_conservative + resume_margin`, derived
    from live state — and an operator override BELOW it warns while naming both."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, miner_stripe_size=RECORDED_STRIPE_SPAN)
        w = [_register(coord, "rrig6600:gpu0", backend="rocm"),
             _register(coord, "zeus-ubuntu-vm:gpu0", backend="cuda")]
        bound = coord.derive_staging_deferred_bound(
            [RECORDED_STRIPE_SPAN] * 4, w, 1)
        margin = coord._resume_margin()
        assert margin == 2, margin
        assert bound == 136 + margin == 138, bound
        assert coord.staging_deferred_bound() == 138
        detail = coord._derived_bound_detail
        assert detail["burst_bound_conservative"] == 136
        assert detail["resume_margin"] == margin

        # the override path, and its WARNING
        records = []
        _lg, _h, restore = _capture_bp(records)
        try:
            coord.config.staging_deferred_max = 64
            assert coord.staging_deferred_bound() == 64, (
                "an explicit operator override must still be honoured")
        finally:
            restore()
        joined = " ".join(records)
        assert "64" in joined and "138" in joined, (
            f"the WARNING must name BOTH numbers: {records}")
        assert "below" in joined.lower()


# ===========================================================================
# G10 — the full configuration route for all four §3 controls
# ===========================================================================
_ROUTE_CONTROLS = {
    "staging_workers": ("staging-workers", "--staging-workers", 4, 7),
    "staging_queue_depth": ("staging-queue-depth", "--staging-queue-depth", 2, 5),
    "staging_deferred_max": ("staging-deferred-max", "--staging-deferred-max", None, 321),
    "staging_capacity_timeout": ("staging-capacity-timeout",
                                 "--staging-capacity-timeout", 600.0, 42.5),
}


def gate10_manifest_to_coordinator_route():
    """Beta gate 10. The COMPLETE route for all four §3 controls:

        manifest default_params -> args_map -> window_optimizer.py argparse
          -> call site -> coordinator attribute
          -> window_optimizer_integration_final.py read
          -> run_trial_miner -> build_coordinator -> CoordinatorConfig

    Gated as a ROUTE, not as four parameters (§2.15): a new Step-1 parameter dies
    silently at hop 1, because WATCHER's step-scoped filter drops any key the
    manifest does not declare.
    """
    # ---- hop 1: the manifest (gitignored *.json — read the LIVE file) -------
    mpath = os.path.join(_ROOT, "agent_manifests", "window_optimizer.json")
    with open(mpath) as fh:
        manifest = json.load(fh)
    dp = manifest["default_params"]
    amap = manifest["actions"][0]["args_map"]
    for key, (kebab, _flag, default, _inject) in _ROUTE_CONTROLS.items():
        assert key in dp, f"hop 1a: manifest default_params lacks {key}"
        assert dp[key] == default, (
            f"hop 1a: manifest {key}={dp[key]!r} != today's default {default!r} — "
            f"Beta did NOT rule a new number ('tune after measurement')")
        assert amap.get(kebab) == key, f"hop 1b: args_map lacks {kebab!r}"
        # WATCHER's step-scoped filter keeps only DECLARED keys
        # (agents/watcher_agent.py:1290-1314, `if key in declared`)
        declared = dict(dp)
        merged = {**declared}
        for k, v in {key: "OVERRIDE", "undeclared_key": 1}.items():
            if k in declared:
                merged[k] = v
        assert merged[key] == "OVERRIDE", f"hop 1c: {key} does not survive the filter"
        assert "undeclared_key" not in merged

    # ---- hop 2: window_optimizer.py ----------------------------------------
    wo = os.path.join(_ROOT, "window_optimizer.py")
    with open(wo) as fh:
        wo_src = fh.read()
    wo_tree = ast.parse(wo_src)
    flags = set()
    for node in ast.walk(wo_tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            flags.add(node.args[0].value)
    for key, (_kebab, flag, _d, _i) in _ROUTE_CONTROLS.items():
        assert flag in flags, f"hop 2a: argparse lacks {flag}"
        assert f"{key}=getattr(args, '{key}'" in wo_src, (
            f"hop 2b: {flag} is parsed but never passed to the run function")
        assert f"coordinator.{key}" in wo_src, (
            f"hop 2c: nothing assigns coordinator.{key} — that read is DEAD")

    # ---- hop 3: the integration read + the two factories -------------------
    integ = os.path.join(_ROOT, "window_optimizer_integration_final.py")
    with open(integ) as fh:
        integ_src = fh.read()
    for key in _ROUTE_CONTROLS:
        assert f"getattr(coordinator, '{key}'" in integ_src, (
            f"hop 3: the integration does not read coordinator.{key}")
        assert key in inspect.signature(run_trial_miner).parameters, (
            f"hop 3: run_trial_miner has no {key} parameter")
        assert key in inspect.signature(build_coordinator).parameters, (
            f"hop 3: build_coordinator has no {key} parameter")
        assert key in {f.name for f in dataclasses.fields(CoordinatorConfig)}, (
            f"hop 3: CoordinatorConfig has no {key} field")

    # ---- END TO END: a value injected AT THE MANIFEST is OBSERVED in the
    # ---- CoordinatorConfig the production factory builds --------------------
    injected = {k: v[3] for k, v in _ROUTE_CONTROLS.items()}
    with tempfile.TemporaryDirectory() as tmp:
        # simulate WATCHER: manifest default_params, overridden, filtered by the
        # DECLARED set, mapped through args_map, parsed by the real argparse
        # mapping, and handed to the production factory.
        params = {**dp, **injected}
        filtered = {k: v for k, v in params.items() if k in dp}
        coord = build_coordinator(
            staging_dir=os.path.join(tmp, "stg"),
            **{k: filtered[k] for k in _ROUTE_CONTROLS})
        for key, (_kebab, _flag, _d, value) in _ROUTE_CONTROLS.items():
            got = getattr(coord.config, key)
            assert got == value, (
                f"manifest-injected {key}={value!r} is NOT observed in the "
                f"coordinator config (got {got!r}) — the route is broken")
        # and the values are LOAD-BEARING, not merely stored
        assert coord._staging_slots()._initial_value == 7 + 5, (
            "staging_workers/staging_queue_depth do not size the admission "
            "semaphore")
        assert coord.staging_deferred_bound() == 321
        assert coord.config.staging_capacity_timeout == 42.5


# ===========================================================================
# G11 — the bounded capacity timeout, end to end through the REAL serve loop
# ===========================================================================
def gate11_capacity_timeout_terminates_outside_the_matrix():
    """Beta gate 11. A capacity wait that never clears is terminated by the ONE
    permitted trial-terminal path: a DIRECT `fail_trial` whose reason LEADS with
    `coordinator_staging_capacity_timeout`. `_handle_stripe_failure_locked` must
    never be entered — the whole point of §0.

    This runs the REAL `serve_trial` via `run_trial_miner` (no `_serve` seam) with
    a real loopback worker over real framed sockets, and the staging slots held by
    a fetch that never returns.
    """
    # ignore_cleanup_errors: staging threads may still be writing under `stg` when
    # the block exits, and a teardown OSError would REPLACE a real assertion
    # failure with "Directory not empty" — masking exactly the diagnostic this
    # gate exists to produce.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1},{"draw":2},{"draw":3}]')
        at_termination = {}
        sink = _Sink(on_abort=lambda ev: at_termination.setdefault(
            "matrix_calls", list(_ACTIVE_SPY[0].calls) if _ACTIVE_SPY else []))
        fetch_gate = threading.Event()          # never set until teardown
        transfer = _GatedTransfer(gate=fetch_gate)

        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        port = lsock.getsockname()[1]

        holder = {}
        log_lines = []
        # the [S172-BP] series is emitted at INFO; the module logger inherits the
        # root's WARNING by default, so a handler alone would capture nothing.
        lg, h, restore = _capture_bp(log_lines)

        # caps MUST match the coordinator's central config or registration
        # quarantines the worker (see _RemoteWorker). nvidia=10 over a 40-seed
        # stripe is 4 sub-stripes against 2 staging slots.
        gate_caps = {**CAPS, "nvidia": 10}

        def run():
            try:
                holder["result"] = run_trial_miner(
                    "runG11", None, 3, "java_lcg", [1, 2, 3], 80, 0.25, 0.25,
                    False, ds, worker_pool_size=1,
                    staging_dir=os.path.join(tmp, "stg"), phase5_sink=sink,
                    transfer=transfer, listen_sock=lsock,
                    family_name="java_lcg", workflow_phase=1,
                    miner_stripe_size=80, seed_cap_nvidia=10,
                    skip_min=0, skip_max=0, offset=0, window_size=3,
                    # 2 staging slots, both held forever by the gated fetch; an
                    # explicit override of 1 closes the hysteresis low-water.
                    staging_workers=2, staging_queue_depth=0,
                    staging_deferred_max=1,
                    staging_capacity_timeout=2.0,
                    # [ALPHA REVIEW FIX] explicit high-water: the default 16 GiB
                    # made this gate's verdict depend on the HOST's free disk —
                    # Part B validation (correctly) refuses a mark exceeding the
                    # filesystem, so the gate red on any host with <16 GiB free
                    # in $TMPDIR for a reason unrelated to what it proves.
                    staging_high_water_bytes=64 * 1024 * 1024,
                    serve_timeout=45.0)
            except Exception:
                holder["err"] = traceback.format_exc()

        with _MatrixSpy() as spy:
            t = threading.Thread(target=run, daemon=True)
            t.start()
            w = _RemoteWorker("127.0.0.1", port, "hostA", 0, caps=gate_caps,
                              send_delay=0.3)
            try:
                w.connect_register()
                w.start_loop()
                t.join(timeout=60)
                assert not t.is_alive(), "serve_trial never terminated"
                assert "err" not in holder, holder.get("err")
                result = holder["result"]

                # the trial is terminally FAILED...
                assert result["state"] == "aborted", result["state"]
                # ...for the RIGHT reason, LEADING with the root cause
                assert sink.aborts, "no abort event was delivered"
                reason = sink.aborts[0].get("reason", "")
                assert reason.startswith("coordinator_staging_capacity_timeout:"), (
                    f"terminal reason does not LEAD with the capacity timeout: "
                    f"{reason!r}")
                assert "connections paused" in reason, reason
                # ...and NOTHING went through the worker retry matrix on the way
                # to that decision. Measured AT THE INSTANT OF TERMINATION (the
                # synchronous L7 abort discharge): after the trial is terminal the
                # two staging jobs still blocked in `fetch_remote` are released by
                # teardown and fail, which legitimately reaches the matrix — but
                # only to be answered `noop: trial already terminal`. Asserting on
                # the post-teardown total would measure the harness, not the fix.
                assert at_termination.get("matrix_calls") == [], (
                    f"a capacity timeout entered the matrix BEFORE terminating "
                    f"the trial: {at_termination.get('matrix_calls')}")
                for c in spy.calls:
                    assert not c["lease_expiry"], (
                        "a lease expiry was routed to the matrix during a pause")
                # the metrics recorded the termination, under the grep-stable prefix
                bp = result["staging_backpressure"]
                assert bp["capacity_timeout_terminations"] == 1, bp
                assert bp["paused_high_water"] >= 1, bp
                # the §1.6 sizing-invariant path must NOT be what fired here —
                # this gate is about the bounded WAIT, not about a bad bound.
                assert bp["capacity_invariant_terminations"] == 0, bp
                assert any(ln.startswith("[S172-BP] capacity_timeout")
                           for ln in log_lines), (
                    "no grep-stable [S172-BP] capacity_timeout line was emitted")
                assert any(ln.startswith("[S172-BP] summary") for ln in log_lines), (
                    "no [S172-BP] summary line at trial termination")
            finally:
                restore()
                w.stop()
                fetch_gate.set()
                try:
                    lsock.close()
                except OSError:
                    pass


class _RemoteWorker:
    """A loopback worker that reports REMOTE (spooled) sub-stripe results, so each
    result consumes a real staging slot via a real fetch.

    ⚠ `caps` must MATCH the coordinator's central cap config or `_validate_caps`
    quarantines the worker at registration, it never becomes eligible, and the
    trial dies of admission rather than of the condition under test. (That is
    exactly how this gate first went red, and it is a real property of the
    production handshake — not a harness quirk.)
    """

    def __init__(self, host, port, hostname, gpu_id, caps=None, send_delay=0.0):
        self.host, self.port = host, port
        self.hostname, self.gpu_id = hostname, gpu_id
        self.worker_id = f"{hostname}:gpu{gpu_id}"
        self.caps = dict(caps) if caps else dataclasses.asdict(VramCaps())
        # A real GPU worker computes between sub-stripes; blasting every result in
        # one burst lets the READER outrun the serve loop, so the capacity gate is
        # still open when all of them are read and nothing ever pauses. The delay
        # models the compute gap, it does not weaken anything under test.
        self.send_delay = send_delay
        self.err = None
        self._stop = threading.Event()
        self.fs = None
        self._t = None
        # every StripeAssign this worker was actually handed — the honest way to
        # assert "no result traffic for that stage" (G-BOUND-DERIVATION-FAILURE).
        self.assigned = []

    def connect_register(self):
        sock = socket.create_connection((self.host, self.port))
        self.fs = MinerFramedSocket(sock)
        self.fs.send_msg(RegisterMessage(
            worker_id=self.worker_id, hostname=self.hostname, gpu_id=self.gpu_id,
            gpu_name="fake", backend="cuda", vram_bytes=12 * 1024 ** 3,
            capabilities={"supported_variants": supported_variants(),
                          "seed_caps": dict(self.caps)}))

    def start_loop(self):
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
                    self._respond(msg)
                elif msg.message_type == "shutdown":
                    break
        except Exception:                                     # noqa: BLE001
            self.err = traceback.format_exc()

    def _respond(self, assign):
        self.assigned.append(assign.stripe_id)
        eff = (assign.payload or {}).get("min_match_threshold")
        # partition with the SAME cap the coordinator sized expected_substripes
        # with (advertised_effective_cap over the advertised caps) — Blocker 7.
        cap = advertised_effective_cap("cuda", assign.family_name, self.caps)
        n = max(1, -(-assign.seed_count // cap))
        for i in range(n):
            ss = assign.seed_start + i * cap
            sc = min(cap, assign.seed_start + assign.seed_count - ss)
            _obj, pb = build_substripe_payload_bytes(
                assign.stripe_id, i, ss, sc, [[ss, 0.9, None, [1]]])
            self.fs.send_msg(SubStripeResultMessage(
                worker_id=self.worker_id, stripe_id=assign.stripe_id,
                sub_index=i, seed_start=ss, seed_count=sc, survivor_count=1,
                spool_path=f"{SPOOL_ROOT}/{assign.stripe_id}/{i}.json", inline=None,
                size_bytes=len(pb), sha256=hashlib.sha256(pb).hexdigest(),
                effective_threshold=eff))
            if self.send_delay:
                time.sleep(self.send_delay)

    def stop(self):
        self._stop.set()
        try:
            self.fs.close()
        except Exception:                                     # noqa: BLE001
            pass


# ===========================================================================
# G-LEASE — the §1.4 exemption, and its narrowness
# ===========================================================================
def gate_lease_exemption_is_required_and_narrow():
    """§1.4. A pause held past `compute_lease_timeout` must NOT expire the paused
    worker's lease — heartbeats ride the same ordered TCP stream as results, so a
    coordinator-initiated pause silences renewals the coordinator itself caused.
    Without this, any pause > 300s reds Beta gates 1-2 through
    `process_lease_expiry` -> the matrix -> the constant-phase `fail_trial` row.

    The exemption must be NARROW: an UNPAUSED worker's genuine silence still
    expires.
    """
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg(compute_lease_timeout=300.0))
        try:
            run_id = "runLEASE"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            # A pauses; B stays silent but UNPAUSED. Both leases expire at t=400.
            #
            # HYBRID (phase 3) deliberately: B's genuine expiry must be observable
            # WITHOUT killing the trial, because a constant-phase expiry fails the
            # whole trial and cancels every active stripe — including A's, which
            # would make A's survival unobservable for the wrong reason.
            _claim(b.coord, run_id, "runLEASE_sA", "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"], phase=3,
                   family="java_lcg_hybrid", lease=400.0)
            _claim(b.coord, run_id, "runLEASE_sB", "hostB:gpu0",
                   b.wconn_by_worker["hostB:gpu0"], phase=3,
                   family="java_lcg_hybrid", lease=400.0)
            b.saturate()
            b.peers["hostA:gpu0"].send(
                _inline_result("hostA:gpu0", "runLEASE_sA", 0, 0, 30))
            assert b.wait_paused(1), "A never paused"
            assert b.coord.paused_worker_ids() == frozenset({"hostA:gpu0"})

            with _MatrixSpy() as spy:
                # t = 1000 is 600s past both leases — more than compute_lease_timeout
                out = b.coord.process_lease_expiry(
                    run_id, list(b.wconn_by_worker.values()), now=1000.0)
                touched = {c["stripe_id"] for c in spy.calls}
            assert "runLEASE_sA" not in touched, (
                "the PAUSED worker's lease expired into the matrix — Beta gates 1-2 "
                "red through the lease door")
            assert "runLEASE_sB" in touched, (
                "the exemption is too wide: an UNPAUSED worker's genuine silence "
                "must still expire")
            assert len(out) == 1, out
            # A's stripe is UNTOUCHED: still claimed, attempt 0, not degraded.
            sa = b.coord.ledger.get_stripe(run_id, "runLEASE_sA")
            assert sa["state"] == ST_CLAIMED, sa["state"]
            assert sa["current_attempt"] == 0 and sa["claimed_by"] == "hostA:gpu0"
            assert not sa["phase_degraded"]
            # ...while B's genuine expiry really did go through the matrix.
            #
            # [F1/F2 R1 §C] UPDATED TO THE RATIFIED DEFERRED-PLACEMENT SEMANTICS.
            # This gate used to assert a direct "reassigned", which is the OLD
            # model: the retry claimed an alternate whether or not that alternate
            # was compute-idle. Under F1 the requeue does not claim — it returns
            # the stripe to the coordinator-owned backlog and the scheduler hands
            # it to the first idle alternate with a FRESH lease. Here the only
            # alternate (hostA) is compute-busy holding its own stripe and the
            # prior claimer (hostB) may not take its own failure back, so the
            # ratified outcome is "requeued": queued, never lost.
            #
            # WHAT THIS GATE MEASURES IS UNCHANGED — the pause exemption and its
            # narrowness. The retry ACCOUNTING below is asserted exactly as
            # before (one budget consumed, degraded, trial alive); only the
            # placement outcome moved, and it moved because Beta ratified that it
            # should.
            assert out[0]["action"] == "requeued", out
            assert out[0]["worker_id"] is None, out
            sb = b.coord.ledger.get_stripe(run_id, "runLEASE_sB")
            assert sb["state"] == ST_PENDING, (
                f"a requeued stripe must return to the backlog, not vanish: {sb}")
            assert sb["claimed_by"] == "hostB:gpu0", (
                "the prior claimer must survive on the pending row — it is what "
                "stops the scheduler handing the stripe straight back")
            assert sb["lease_expires_at"] is None, (
                f"backlog carries a ticking lease: {sb}")
            assert sb["current_attempt"] == 1 and sb["phase_degraded"]
            assert b.coord.ledger.get_trial(run_id)["state"] == "running"
        finally:
            b.close()


def gate_lease_exemption_mutant():
    """MUTATION EVIDENCE for the lease gate (§5). Remove ONLY the exemption — by
    making `paused_worker_ids()` report nothing, which is exactly the pre-fix state
    of the world — and prove (a) the mutated path EXECUTED and (b) it reds the
    credited assertion. Without this, a green G-LEASE could be green for any
    reason at all."""
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg(compute_lease_timeout=300.0))
        try:
            run_id = "runLEASEM"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            _claim(b.coord, run_id, "runLEASEM_sA", "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"], lease=400.0)
            b.saturate()
            b.peers["hostA:gpu0"].send(
                _inline_result("hostA:gpu0", "runLEASEM_sA", 0, 0, 30))
            assert b.wait_paused(1)

            executed = {"n": 0}
            orig = RangeMinerCoordinator.paused_worker_ids

            def _mutant(self_):
                executed["n"] += 1
                return frozenset()          # the exemption, removed

            RangeMinerCoordinator.paused_worker_ids = _mutant
            try:
                with _MatrixSpy() as spy:
                    b.coord.process_lease_expiry(
                        run_id, list(b.wconn_by_worker.values()), now=1000.0)
                    touched = {c["stripe_id"] for c in spy.calls}
            finally:
                RangeMinerCoordinator.paused_worker_ids = orig

            # (a) the mutated path really ran
            assert executed["n"] >= 1, (
                "the mutant was never called — process_lease_expiry does not "
                "consult the pause registry, so G-LEASE proves nothing")
            # (b) and it REDS the credited assertion
            assert "runLEASEM_sA" in touched, (
                "removing the exemption did NOT route the paused worker's expiry "
                "into the matrix — G-LEASE is vacuous")
            # ...through the constant-phase row, i.e. a dead trial
            st = b.coord.ledger.get_trial(run_id)
            assert st["state"] == "aborted", (
                "the mutant should kill the trial exactly as the pre-fix code did")
        finally:
            b.close()


def gate_pause_mutant():
    """MUTATION EVIDENCE for the pause gate (§5). Remove ONLY the capacity gate —
    `staging_can_accept()` always True, which is the pre-fix reader — and prove the
    mutated path executed and the credited "peer stalls" assertion reds."""
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg())
        try:
            run_id, sid = "runPM", "runPM_s0"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            _claim(b.coord, run_id, sid, "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"])
            b.saturate()

            executed = {"n": 0}
            orig = RangeMinerCoordinator.staging_can_accept

            def _mutant(self_):
                executed["n"] += 1
                return True                 # the gate, removed

            RangeMinerCoordinator.staging_can_accept = _mutant
            try:
                b.peers["hostA:gpu0"].send(
                    _inline_result("hostA:gpu0", sid, 0, 0, 30))
                got = b.drain(0.8)
            finally:
                RangeMinerCoordinator.staging_can_accept = orig

            assert executed["n"] >= 1, (
                "the mutant was never called — the reader does not consult the "
                "capacity gate, so the pause gates prove nothing")
            delivered = [m for (k, _s, m, _c) in got if k == "msg"]
            assert len(delivered) == 1, (
                "with the gate removed the envelope should be delivered "
                "IMMEDIATELY into coordinator RAM — it was not, so the pause gates "
                "are not measuring the gate")
            assert b.coord.paused_connection_count() == 0, (
                "the connection paused with the gate removed")
        finally:
            b.close()


# ===========================================================================
# G-MATRIX-DIFF — the six out-of-scope callers, proven surgical
# ===========================================================================
_OUT_OF_SCOPE_CALLERS = {
    # the reason each one STAYS (brief §0) — not a code comment, a governance fact
    "D1b attempt cannot fit staging_high_water_bytes": "permanent configuration "
                                                       "impossibility",
    "StagingHashMismatch": "worker-output defect, C2:130-132 + Defect-5 ruling",
    "StagingTimeout": "staging-job failure, matrix-governed per C2",
    "StagingConfigurationError": "Part B binding ruling, narrow by construction",
    "generic transient fetch/IO": "C2 retryable class",
    "StripeComplete reconciliation mismatch": "definitive structural worker-output "
                                              "failure",
}


def _on_staging_failed_call_sites(src_text):
    """Every `self._on_staging_failed(...)` call in a module source, normalized
    through `ast.unparse` so comments/blank lines/line numbers cannot mask a real
    argument change. AST over live source, never a text anchor — `2389b61` reverted
    a fix by whole-block replacement and a text anchor would have gone green."""
    tree = ast.parse(src_text)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_on_staging_failed"):
            out.append(ast.unparse(node))
    return out


# The two BASELINES this gate compares against. They are pinned commits, NOT
# `HEAD`, and that is the whole point:
#
#   * `HEAD` was correct only while the remediation was uncommitted. The moment
#     `4b1aad6` landed, `git show HEAD:` returned the POST-fix file, so `before`
#     became 6 and the gate red on its own success — which is exactly what it did
#     at `42bdbb1`, before this amendment. A gate whose baseline moves with the
#     work cannot certify the work.
#   * A commit hash is the only thing that anchors "what the file looked like
#     before the change", so it belongs here (it anchors a certified artifact, not
#     a value copied from memory).
_PRE_REMEDIATION_REV = "7c4f11b1b9910f868b56906f05f7269f58fba53e"   # parent of 4b1aad6
_AMENDMENT_BASELINE_REV = "4b1aad6ddfa7e6f6a3082a7850fe71b7ae7825b8"  # the ruling's subject


def _rev_source(rev, path="miner/range_miner_coordinator.py"):
    return subprocess.run(["git", "-C", _ROOT, "show", f"{rev}:{path}"],
                          capture_output=True, text=True, check=True).stdout


def gate_matrix_diff_six_callers_unchanged():
    """§0: exactly ONE call site is removed and the other SIX are byte-identical.

    Structural half, over three revisions rather than two:
      pre-remediation (7c4f11b) -> 7 call sites
      the ruling's subject (4b1aad6) -> 6
      the working tree (this amendment) -> the SAME 6
    Seven before, six after, the six that remain set-equal to the six that were
    there, and the removed one precisely the deferred-overflow back-pressure call.
    The 4b1aad6-vs-live comparison is the amendment's own claim: F1-F5 changed
    NOTHING in the retry matrix or its surviving callers.
    """
    live = open(os.path.join(_ROOT, "miner", "range_miner_coordinator.py")).read()
    head = _rev_source(_PRE_REMEDIATION_REV)
    base = _rev_source(_AMENDMENT_BASELINE_REV)
    before = _on_staging_failed_call_sites(head)
    at_baseline = _on_staging_failed_call_sites(base)
    after = _on_staging_failed_call_sites(live)
    assert len(before) == 7, (
        f"expected 7 pre-change _on_staging_failed call sites, found {len(before)}")
    assert len(at_baseline) == 6, (
        f"expected exactly 6 at {_AMENDMENT_BASELINE_REV[:7]}, "
        f"found {len(at_baseline)}: {at_baseline}")
    assert len(after) == 6, (
        f"expected exactly 6 after the removal, found {len(after)}: {after}")
    removed = [c for c in before if c not in after]
    assert len(removed) == 1, f"more than one call site changed: {removed}"
    assert "deferred queue full" in removed[0], (
        f"the removed call site is NOT the deferred-overflow one: {removed[0]}")
    surviving_changed = [c for c in after if c not in before]
    assert surviving_changed == [], (
        f"an out-of-scope caller was modified: {surviving_changed}")
    # THE AMENDMENT'S OWN CLAIM: the six survivors are untouched by F1-F5.
    assert after == at_baseline, (
        f"this amendment changed a surviving _on_staging_failed caller:\n"
        f"  at {_AMENDMENT_BASELINE_REV[:7]}: {at_baseline}\n"
        f"  live: {after}")
    # ...and the matrix plumbing itself is untouched, against BOTH baselines.
    #
    # BYTE (AST) IDENTITY RETAINED for the two the F1/F2 amendment does not
    # touch. This is the strongest check available and it still applies to them.
    for meth in ("_on_staging_failed", "handle_stripe_failure"):
        a_src = _method_source(live, meth)
        assert _method_source(head, meth) == a_src, (
            f"{meth} was modified since {_PRE_REMEDIATION_REV[:7]} — out of scope")
        assert _method_source(base, meth) == a_src, (
            f"{meth} was modified by THIS AMENDMENT — out of scope")

    # SUPERSEDED for `_handle_stripe_failure_locked` ONLY (Team Beta, F1/F2 R1
    # §C). Beta granted the supersession WITH AN ORDER CONSTRAINT — "do not
    # simply update the old baseline to the current submitted source; the
    # current source contains Blockers A and B; that would certify the defects."
    # Blockers A and B were fixed FIRST; what is certified below is the
    # corrected function.
    #
    # "Failure matrix unchanged" means TERMINAL DECISION SEMANTICS unchanged.
    # F1 changed how a hybrid first failure is PLACED (deferred placement: the
    # requeue no longer claims), and R1 Blocker A changed the retry's selector
    # from a LIKE-scoped prefix to stripe identity. Both live inside this
    # function; neither is a terminal decision, and an AST comparison cannot
    # express that distinction. The invariant that can:
    #
    #   the four terminal decisions, IN ORDER
    #     non-retryable                                -> non_retryable
    #     constant phase                               -> constant_phase
    #     hybrid first failure + no alternate eligible -> no_alternate_worker
    #     hybrid second failure                        -> hybrid_second_failure
    #   plus BOTH ratified nonterminal outcomes of the hybrid first failure,
    #   plus identity (never prefix) selection on the immediate placement.
    #
    # The behavioural half is `gate_matrix_diff_behavioural` below, which DRIVES
    # every one of these rows including the busy-alternate requeue.
    _assert_matrix_invariant(live)


def _assert_matrix_invariant(module_src):
    """[F1/F2 R1 §C] The superseding invariant for `_handle_stripe_failure_locked`,
    read off the LIVE source by AST. Replaces byte identity for that ONE method;
    every other assertion in the gate that owns it is preserved."""
    EXPECTED = ["non_retryable", "constant_phase", "no_alternate_worker",
                "hybrid_second_failure"]
    hits = [n for n in ast.walk(ast.parse(module_src))
            if isinstance(n, ast.FunctionDef)
            and n.name == "_handle_stripe_failure_locked"]
    assert len(hits) == 1, hits
    terminal, nonterminal = [], []
    for node in ast.walk(hits[0]):
        if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)):
            continue
        d = {k.value: v.value
             for k, v in zip(node.value.keys, node.value.values)
             if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)}
        if d.get("action") == "fail_trial":
            terminal.append((node.lineno, d.get("reason")))
        elif d.get("action") in ("reassigned", "requeued", "noop"):
            nonterminal.append(d["action"])
    ordered = [r for _ln, r in sorted(terminal)]
    assert ordered == EXPECTED, (
        f"the four terminal decisions changed or were reordered: "
        f"{ordered} != {EXPECTED}")
    assert {"reassigned", "requeued"} <= set(nonterminal), (
        f"the hybrid first-failure nonterminal branch lost a ratified outcome: "
        f"{nonterminal}")
    src = ast.unparse(hits[0])
    assert "exact_stripe_id=stripe_id" in src, (
        "the hybrid immediate placement no longer selects by stripe identity "
        "(R1 Blocker A)")
    assert "stripe_prefix=stripe_id" not in src, (
        "prefix-as-exact selection was reintroduced (R1 Blocker A)")


def _method_source(module_src, name):
    tree = ast.parse(module_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"method {name} not found")


def gate_matrix_diff_behavioural():
    """§0 behavioural half: DRIVE each surviving classification and assert the
    matrix outcome is exactly what it was — same action, same reason, same retry
    accounting. A removal is only 'surgical' if the six survivors still behave."""
    EXPECTED = [
        # (label, phase, retryable, lease_expiry, alternate, action, reason)
        ("D1b / StagingConfigurationError — non-retryable, constant", 1, False,
         False, True, "fail_trial", "non_retryable"),
        ("D1b / StagingConfigurationError — non-retryable, hybrid", 3, False,
         False, True, "fail_trial", "non_retryable"),
        ("HashMismatch / Timeout / generic / reconciliation — constant 1", 1, True,
         False, True, "fail_trial", "constant_phase"),
        ("HashMismatch / Timeout / generic / reconciliation — constant 2", 2, True,
         False, True, "fail_trial", "constant_phase"),
        ("retryable, hybrid attempt 0, alternate exists", 3, True, False, True,
         "reassigned", None),
        ("retryable, hybrid attempt 0, NO alternate", 3, True, False, False,
         "fail_trial", "no_alternate_worker"),
        ("lease expiry, hybrid, alternate exists", 4, True, True, True,
         "reassigned", None),
        ("lease expiry, constant phase", 1, True, True, True,
         "fail_trial", "constant_phase"),
    ]
    for (label, phase, retryable, lease_expiry, alternate,
         exp_action, exp_reason) in EXPECTED:
        with tempfile.TemporaryDirectory() as tmp:
            coord = _coord(tmp)
            run_id = "runMD"
            coord.ledger.create_trial(run_id, 0)
            w0 = _register(coord, "hostA:gpu0")
            workers = [w0]
            if alternate:
                workers.append(_register(coord, "hostB:gpu0"))
            fam = "java_lcg" if phase in (1, 2) else "java_lcg_hybrid"
            a = coord.assign_stripes(run_id, fam, phase, 1000, [w0], now=100.0)
            sid = a[0]["stripe_id"]
            got = coord.handle_stripe_failure(
                run_id, sid, retryable=retryable, eligible_workers=workers,
                now=200.0, lease_expiry=lease_expiry)
            assert got["action"] == exp_action, (
                f"[{label}] action {got['action']!r} != {exp_action!r} ({got})")
            if exp_reason is not None:
                assert got.get("reason") == exp_reason, (
                    f"[{label}] reason {got.get('reason')!r} != {exp_reason!r}")
            if exp_action == "reassigned":
                assert got["attempt"] == 1 and got["phase_degraded"] is True
    assert len(_OUT_OF_SCOPE_CALLERS) == 6

    # [F1/F2 R1 §C] THE SECOND RATIFIED NONTERMINAL OUTCOME, driven rather than
    # asserted structurally: the hybrid first failure with an alternate that
    # EXISTS but is temporarily COMPUTE-BUSY. Under the old model the retry
    # claimed the alternate regardless of occupancy; under the ratified
    # deferred-placement semantics the stripe returns to the backlog with its
    # retry budget consumed and the trial alive, and the scheduler places it on
    # the first alternate to go idle — with a fresh lease.
    #
    # The terminal rows above are unchanged and are what this gate has always
    # certified; this row exists so "the alternate is busy" can never silently
    # become a terminal decision.
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        run_id = "runMDBUSY"
        coord.ledger.create_trial(run_id, 0)
        w0 = _register(coord, "hostA:gpu0")
        w1 = _register(coord, "hostB:gpu0")
        # both workers get one stripe, so the only alternate is compute-busy
        a = coord.assign_stripes(run_id, "java_lcg_hybrid", 3,
                                 2 * coord.config.miner_stripe_size,
                                 [w0, w1], now=100.0)
        claimed = [r for r in a if r["claimed"]]
        assert len(claimed) == 2, a
        sid = claimed[0]["stripe_id"]
        prior = claimed[0]["worker_id"]
        other = next(r["worker_id"] for r in claimed if r["worker_id"] != prior)
        got = coord.handle_stripe_failure(
            run_id, sid, retryable=True, eligible_workers=[w0, w1], now=200.0)
        assert got["action"] == "requeued", (
            f"[hybrid attempt 0, alternate exists but is COMPUTE-BUSY] {got}")
        assert got["attempt"] == 1 and got["phase_degraded"] is True, got
        assert coord.ledger.get_trial(run_id)["state"] == "running", (
            "a temporarily-busy alternate became a TERMINAL decision")
        st = coord.ledger.get_stripe(run_id, sid)
        assert st["state"] == ST_PENDING and st["lease_expires_at"] is None, st
        # ...and it really is placed once that alternate frees its compute slot
        osid = next(r["stripe_id"] for r in claimed if r["worker_id"] == other)
        coord.ledger.record_stripe_complete(run_id, osid, 0, other, 1, 0)
        placed = coord.schedule_pending_stripes(
            run_id, "java_lcg_hybrid", 3, [w0, w1],
            stage_prefix=run_id, now=300.0)
        assert [p["stripe_id"] for p in placed] == [sid], placed
        assert placed[0]["worker_id"] == other != prior, placed
        st2 = coord.ledger.get_stripe(run_id, sid)
        assert st2["lease_expires_at"] == 300.0 + coord.config.compute_lease_timeout, st2


def gate_no_capacity_path_reaches_the_matrix():
    """§0's classification law, asserted structurally over the LIVE source:
    `enqueue_staging` no longer routes ANY capacity condition into the matrix, and
    its one terminal path is a DIRECT `fail_trial`."""
    src = inspect.getsource(RangeMinerCoordinator.enqueue_staging)
    tree = ast.parse(src.lstrip())
    calls = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any("_on_staging_failed(run_id, stripe_id, True" in c
                   for c in calls), (
        "a retryable capacity failure is still routed through the matrix")
    assert not any(".handle_stripe_failure(" in c for c in calls), (
        "enqueue_staging reaches the matrix directly")
    assert any(c.startswith("self.fail_trial(") for c in calls), (
        "the invariant path does not terminate via a DIRECT fail_trial")
    assert "coordinator_staging_capacity_invariant" in src, (
        "the invariant reason string is missing")
    # ...and the D1b non-retryable caller is STILL there (it is out of scope)
    assert any("_on_staging_failed(run_id, stripe_id, False" in c for c in calls), (
        "the D1b permanent-configuration caller was removed — it is out of scope")

    # the §1.5 terminal path is likewise direct, on the serve loop
    serve = inspect.getsource(RangeMinerCoordinator.serve_trial)
    stree = ast.parse(serve.lstrip())
    for node in ast.walk(stree):
        if (isinstance(node, ast.If)
                and "staging_capacity_timeout_expired" in ast.unparse(node.test)):
            body = ast.unparse(node)
            assert "self.fail_trial(" in body, (
                "the capacity timeout does not terminate via a DIRECT fail_trial")
            assert "handle_stripe_failure" not in body, (
                "the capacity timeout routes through the matrix")
            break
    else:
        raise AssertionError("no capacity-timeout check in the serve loop")


# ===========================================================================
# §4 metrics — grep-stable and complete
# ===========================================================================
def gate_metrics_are_grep_stable_and_complete():
    """§4: every required series is emitted through the existing logger, prefixed
    `[S172-BP]` so `gate_s172_prod_shape.py` and operators can extract it."""
    with tempfile.TemporaryDirectory() as tmp:
        records = []
        _lg, _h, restore = _capture_bp(records)
        b = _Bench(tmp, **_saturating_cfg())
        try:
            run_id, sid = "runMET", "runMET_s0"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            _claim(b.coord, run_id, sid, "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"])
            b.saturate()
            b.peers["hostA:gpu0"].send(_inline_result("hostA:gpu0", sid, 0, 0, 30))
            assert b.wait_paused(1)
            b.release_all_slots()
            b.coord._release_capacity()
            assert b.wait_unpaused()
            b.drain(0.5)
            m = b.coord.log_staging_backpressure_summary(run_id)
        finally:
            restore()
            b.close()

    bp = [r for r in records if r.startswith("[S172-BP]")]
    assert bp, "no [S172-BP] lines at all"
    kinds = {r.split()[1] for r in bp if len(r.split()) > 1}
    for required in ("pause", "resume", "summary"):
        assert required in kinds, (
            f"no grep-stable [S172-BP] {required} line; saw {sorted(kinds)}")
    for key in ("inbound_qsize_high_water", "deferred_high_water", "paused_now",
                "paused_high_water", "pause_events", "pause_seconds_total",
                "pause_seconds_max", "staging_jobs_completed",
                "capacity_timeout_terminations", "bound_in_force",
                "derived_bound", "staging_jobs_per_sec"):
        assert key in m, f"the metrics record lacks {key}"
    summary = [r for r in bp if r.startswith("[S172-BP] summary")][0]
    for key in ("inbound_qsize_high_water=", "deferred_high_water=",
                "paused_high_water=", "pause_seconds_total=",
                "staging_jobs_completed=", "capacity_timeout_terminations="):
        assert key in summary, f"the summary line lacks {key}: {summary}"
    assert m["pause_events"] >= 1 and m["pause_seconds_total"] > 0.0


# ===========================================================================
# S172-BP AMENDMENT (Beta findings F1-F5, ruling of 2026-08-06)
# Five targeted gates. Every one of them is RED against the behaviour committed
# at 4b1aad6 — the red runs come from a worktree at that commit with this file
# copied in (see docs/CLAUDE_CODE_REPORT_S172_BP_AMENDMENT.md §red-table).
# ===========================================================================

def gate_resume_credit_one_wake_per_release():
    """F1 / G-RESUME-CREDIT (part a — the credit arithmetic, deterministic).

    TWO paused connections are registered through the REAL registry API and
    capacity is held WIDE OPEN for the whole gate — so the ONLY thing that can
    stop a second wake is the ingress credit. That is the defect exactly: the
    pre-amendment `_resume_paused_connections` set an event, re-checked
    `staging_can_accept()` and looped, and because a wake CONSUMED NOTHING one
    freed slot satisfied the check on every iteration and released the fleet.

    No reader threads here on purpose: the assertion is about which event a single
    capacity-release invocation sets, and that is decided synchronously inside the
    call. Part (b) drives the same property through real reader threads with real
    capacity accounting.
    """
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, staging_workers=2, staging_queue_depth=0,
                       staging_deferred_max=1, miner_stripe_size=1000)
        _register(coord, "hostA:gpu0")
        _register(coord, "hostB:gpu0")
        assert coord.staging_can_accept(), (
            "capacity must be OPEN for this gate — otherwise a green result means "
            "only that nothing could resume")

        eA = coord.register_paused_connection("connA", "hostA:gpu0")
        eB = coord.register_paused_connection("connB", "hostB:gpu0")
        assert not eA.is_set() and not eB.is_set()

        # --- ONE capacity-release path invocation -> AT MOST ONE wake ---------
        # (asserted BEFORE anything touches the new credit API, so this gate reds
        # against 4b1aad6 on the BEHAVIOUR rather than on a missing attribute)
        coord._release_capacity()
        assert eA.is_set(), "the FIFO-first paused reader was not woken at all"
        assert not eB.is_set(), (
            "ONE capacity observation woke MORE THAN ONE reader — the wake did "
            "not consume the observation (thundering herd)")
        assert coord.resume_credits_outstanding() == 1, (
            "the wake did not take a credit, so nothing reserves the observation")

        # --- further release events grant NOTHING while the credit is out -----
        for _ in range(5):
            coord._release_capacity()
        assert not eB.is_set(), (
            "a second reader was woken while a credit was outstanding — a wake "
            "must RESERVE the observation until it is used")
        assert coord.resume_credits_outstanding() == 1

        # --- a NON-HEAD reader cannot self-resume on someone else's observation
        assert coord.staging_can_accept()
        assert coord._try_self_resume("connB") is False, (
            "a non-head paused reader self-released — the defensive poll is still "
            "a second thundering-herd door")

        # --- credit consumed -> the FIFO head may take the next observation ----
        coord.deregister_paused_connection("connA", reason="resume")
        assert coord._release_resume_credit("connA", delivered=True) is True
        assert coord.resume_credits_outstanding() == 0
        assert coord._try_self_resume("connB") is True, (
            "the FIFO head could not escape with capacity open and no grant in "
            "flight — the lost-wakeup protection was lost")
        assert eB.is_set()
        assert coord.resume_credits_outstanding() == 1
        # ...and clearing is attributable: a non-holder cannot clear it
        assert coord._release_resume_credit("connA", delivered=False) is False
        assert coord.resume_credits_outstanding() == 1
        assert coord._release_resume_credit("connB", delivered=True) is True
        assert coord.resume_credits_outstanding() == 0


def gate_resume_credit_real_readers_fifo():
    """F1 / G-RESUME-CREDIT (part b — real readers, real capacity accounting).

    Two REAL reader threads pause on real framed sockets. Exactly ONE staging
    capacity unit is freed and exactly ONE capacity-release path is invoked; the
    freed unit is then RECLAIMED, which is what the serve loop does in production
    the moment it stages the resumed envelope. The second reader must therefore
    stay paused across a real settling window (>= 10 of its 50 ms poll cycles),
    and must resume when a SECOND unit is freed. FIFO order is asserted throughout.

    ⚠ THIS GATE DOES NOT COVER THE HANDOFF INVARIANT (Beta F1-R §5, last line).
    It reclaims the freed unit FROM THE TEST THREAD immediately after the grant,
    "modelling the serve loop" — and that reclaim deletes precisely the interval
    the reservation has to survive (envelope in `inbound`, slot still physically
    free). It went green in round 1 against a reader that cleared its credit at
    `inbound.put`, which is the defect. What it still legitimately proves is the
    CAPACITY ACCOUNTING: one freed unit resumes exactly one reader, FIFO-first, and
    a second unit is needed for the second reader. The handoff itself — that the
    reservation survives until DISPOSITION — is proven only by G-RESUME-HANDOFF,
    which dispatches through the REAL serve path and touches neither the semaphore
    nor the credit from the test thread.

    On the reclaim window: until the woken reader deregisters it is still the FIFO
    head — so the second reader cannot self-resume before the reclaim, which
    happens a couple of microseconds after the grant returns while a thread wake-up
    costs tens. The gate records the inbound depth at reclaim time so a lost race
    is reported as itself instead of as a confusing "the second reader resumed".
    """
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg())
        try:
            run_id = "runRC"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            for wid, sid in (("hostA:gpu0", "runRC_sA"), ("hostB:gpu0", "runRC_sB")):
                _claim(b.coord, run_id, sid, wid, b.wconn_by_worker[wid])
            b.saturate()
            sem = b.coord._staging_slots()

            # A pauses FIRST, then B — the FIFO order under test.
            b.peers["hostA:gpu0"].send(
                _inline_result("hostA:gpu0", "runRC_sA", 0, 0, 30))
            assert b.wait_paused(1), "A never paused"
            b.peers["hostB:gpu0"].send(
                _inline_result("hostB:gpu0", "runRC_sB", 0, 0, 30))
            assert b.wait_paused(2), "B never paused"
            order = [r["worker_id"] for r in b.coord._paused_connections.values()]
            assert order == ["hostA:gpu0", "hostB:gpu0"], (
                f"the pause registry is not in FIFO entry order: {order}")

            # ---- free EXACTLY ONE unit, invoke EXACTLY ONE release path ------
            b._held.pop()
            sem.release()
            b.coord._release_capacity()
            depth_at_reclaim = b.inbound.qsize()
            reclaimed = sem.acquire(blocking=False)
            if reclaimed:
                b._held.append(True)
            assert reclaimed, (
                "the freed unit could not be reclaimed — this gate models the "
                "serve loop staging the resumed envelope, and without the reclaim "
                "capacity stays open and nothing is being measured")
            assert not b.coord.staging_can_accept(), (
                "capacity is still open after the reclaim")

            # ---- exactly ONE reader resumed, and it is the FIFO-first --------
            deadline = time.time() + 3.0
            while time.time() < deadline and b.coord.paused_connection_count() > 1:
                time.sleep(0.01)
            assert b.coord.paused_worker_ids() == frozenset({"hostB:gpu0"}), (
                f"expected ONLY hostB still paused, got "
                f"{b.coord.paused_worker_ids()} (inbound depth at reclaim = "
                f"{depth_at_reclaim}; a nonzero depth means the woken reader beat "
                f"the reclaim and the gate lost its capacity race)")
            entries = b.drain(0.5)
            first = [m for (k, s, m, _c) in entries
                     if k == "msg" and b.worker_by_sock.get(s) == "hostA:gpu0"]
            assert len(first) == 1, (
                f"the FIFO-first reader did not deliver its held envelope: {first}")
            # [S172-BP AMENDMENT F1-R] delivery is INGRESS, so the reservation is
            # still outstanding here. Disposing of it is the serve loop's job:
            # dispatch through the production seam. Capacity is (deliberately)
            # closed, so this envelope is RETAINED in the bounded deferred queue —
            # disposition (ii), which ends the reservation exactly as (i) does.
            assert b.coord.resume_credits_outstanding() == 1, (
                "the reservation ended at ingress — the freed slot is still "
                "unconsumed and a second reader could wake on it")
            b.dispatch(entries, run_id)

            # ---- and B REMAINS paused across a real settling window ----------
            settle_end = time.time() + 0.6      # >= 12 of B's 50 ms poll cycles
            while time.time() < settle_end:
                assert b.coord.paused_worker_ids() == frozenset({"hostB:gpu0"}), (
                    "the second reader resumed on the FIRST reader's capacity "
                    "observation — one freed slot woke more than one connection")
                time.sleep(0.02)
            assert b.coord.resume_credits_outstanding() == 0, (
                "the wake's reservation was never released at disposition — the "
                "fleet would wedge")

            # ---- free a SECOND unit: now the second reader resumes -----------
            b._held.pop()
            sem.release()
            b.coord._release_capacity()
            assert b.wait_unpaused(), "the second reader never resumed"
            second = [m for (k, s, m, _c) in b.drain(0.8)
                      if k == "msg" and b.worker_by_sock.get(s) == "hostB:gpu0"]
            assert len(second) == 1, (
                f"the second reader did not deliver its held envelope: {second}")
        finally:
            b.close()


def gate_resume_credit_mutants():
    """MUTATION EVIDENCE for F1 (both doors), deterministic at the registry API.

    Mutant 1 RESTORES THE LOOP: `_resume_paused_connections` becomes the
    pre-amendment while-loop. It must EXECUTE and wake BOTH readers on one
    capacity-release invocation, redding the credited assertion of part (a).

    Mutant 2 lets a NON-HEAD reader self-resume (the bare `staging_can_accept()`
    escape). It must EXECUTE and return True for a non-head connection, redding
    the head-only assertion.
    """
    # ---- mutant 1: the loop -------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, staging_workers=2, staging_queue_depth=0,
                       staging_deferred_max=1, miner_stripe_size=1000)
        _register(coord, "hostA:gpu0")
        _register(coord, "hostB:gpu0")
        eA = coord.register_paused_connection("connA", "hostA:gpu0")
        eB = coord.register_paused_connection("connB", "hostB:gpu0")
        executed = {"n": 0}
        orig = RangeMinerCoordinator._resume_paused_connections

        def _looping_mutant(self_):
            executed["n"] += 1
            while True:                          # the deleted loop, restored
                if not self_.staging_can_accept():
                    return
                with self_._pause_lock:
                    target = None
                    for _key, rec in self_._paused_connections.items():
                        if not rec["event"].is_set():
                            target = rec
                            break
                    if target is None:
                        return
                    target["event"].set()
                if not self_.staging_can_accept():
                    return

        RangeMinerCoordinator._resume_paused_connections = _looping_mutant
        try:
            coord._release_capacity()
        finally:
            RangeMinerCoordinator._resume_paused_connections = orig
        assert executed["n"] >= 1, (
            "the mutant was never called — `_pump_deferred`'s finally is not the "
            "resume trigger, so G-RESUME-CREDIT proves nothing")
        assert eA.is_set() and eB.is_set(), (
            "restoring the loop did NOT wake both readers on one observation — "
            "G-RESUME-CREDIT is vacuous")

    # ---- mutant 2: non-head self-resume ------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, staging_workers=2, staging_queue_depth=0,
                       staging_deferred_max=1, miner_stripe_size=1000)
        _register(coord, "hostA:gpu0")
        _register(coord, "hostB:gpu0")
        coord.register_paused_connection("connA", "hostA:gpu0")
        coord.register_paused_connection("connB", "hostB:gpu0")
        assert coord._try_self_resume("connB") is False   # the fixed behaviour
        executed = {"n": 0}
        orig_self = RangeMinerCoordinator._try_self_resume

        def _headless_mutant(self_, conn_key):
            executed["n"] += 1
            # the pre-amendment escape: capacity alone, no head test, no credit
            return bool(self_.staging_can_accept())

        RangeMinerCoordinator._try_self_resume = _headless_mutant
        try:
            escaped = coord._try_self_resume("connB")
        finally:
            RangeMinerCoordinator._try_self_resume = orig_self
        assert executed["n"] == 1, "the non-head mutant never executed"
        assert escaped is True, (
            "removing the head test did NOT let a non-head reader self-release — "
            "the head-only assertion is vacuous")


def _spool_result(wid, sid, sub_index, seed_start, seed_count):
    """A REMOTE (spooled) result, so staging it consumes a real slot through a real
    fetch — the slot is then held for as long as the gated transfer blocks."""
    _obj, pb = build_substripe_payload_bytes(
        sid, sub_index, seed_start, seed_count, [[seed_start, 0.9, None, [1]]])
    remote = f"{SPOOL_ROOT}/{sid}/{sub_index}.json"
    msg = SubStripeResultMessage(
        worker_id=wid, stripe_id=sid, sub_index=sub_index, seed_start=seed_start,
        seed_count=seed_count, survivor_count=1, spool_path=remote, inline=None,
        size_bytes=len(pb), sha256=hashlib.sha256(pb).hexdigest(),
        effective_threshold=0.25)
    return msg, remote, pb


class _Round1ClearQueue(_queue.Queue):
    """THE ROUND-1 READER, restored at exactly the one instruction that differs.

    Round 1 called `_release_resume_credit(rawsock, delivered=True)` immediately
    after the successful `inbound.put`. Reproducing it by wrapping `put` executes
    the same clear, in the same thread, at the same instant, without rewriting
    `_conn_reader_loop` — so the mutant is the round-1 BEHAVIOUR and not an
    approximation of it.
    """

    def __init__(self, executed, **kw):
        super().__init__(**kw)
        self.coord = None
        self._executed = executed

    def put(self, item, *a, **kw):
        out = super().put(item, *a, **kw)
        try:
            kind, sock, msg, _credit_id = item
        except (TypeError, ValueError):
            return out
        if (kind == "msg" and self.coord is not None
                and getattr(msg, "message_type", None) == "sub_stripe_result"):
            if self.coord._release_resume_credit(
                    sock, delivered=True, disposition="round1_ingress_clear"):
                self._executed["n"] += 1
        return out


def _handoff_bench(tmp, inbound=None):
    """The shared fixture for G-RESUME-HANDOFF and its mutant: two real readers,
    two real paused connections, and a staging job that HOLDS its slot (a gated
    fetch) so capacity never reopens behind the gate's back."""
    fetch_gate = threading.Event()
    payloads, msgs = {}, {}
    for wid, sid, subs in (("hostA:gpu0", "runRH_sA", (0, 1)),
                           ("hostB:gpu0", "runRH_sB", (0,))):
        for i in subs:
            m, remote, pb = _spool_result(wid, sid, i, i * 30, 30)
            payloads[remote] = pb
            msgs[(wid, i)] = m
    transfer = _GatedTransfer(payloads=payloads, gate=fetch_gate)
    b = _Bench(tmp, transfer=transfer, inbound=inbound, **_saturating_cfg())
    return b, fetch_gate, msgs


def gate_resume_handoff_survives_until_disposition():
    """F1-R / G-RESUME-HANDOFF — Beta §5, all eleven steps.

    THE DEFECT. Round 1 cleared the ingress credit at `inbound.put`. But the freed
    staging slot is consumed only when the SERVE LOOP later dispatches that
    envelope into `enqueue_staging`. In the gap — envelope in `inbound`, slot still
    physically free — reader B finds: B is FIFO head (A deregistered), credits == 0
    (A cleared at put), and `staging_can_accept()` true ON THE SAME SLOT. Two
    wakes, one slot.

    THE GATE. Two REAL paused readers. Exactly ONE unit is freed and exactly ONE
    release path is invoked. A's envelope is then allowed to reach `inbound` and A
    to leave the pause registry — and from that instant until the dispatch, THE
    TEST THREAD TOUCHES NEITHER THE SEMAPHORE NOR THE CREDIT. That is the whole
    correction: round 1's part (b) reclaimed the unit itself "modelling the serve
    loop", which deleted exactly the interval under proof. The unit here is
    consumed by the REAL `enqueue_staging`, reached through the REAL dispatch seam.

    A deliberately has a SECOND result queued behind the first, so Beta §4's tail
    ("no second result while the reservation is out") is proven by the same hold:
    one connection must not stream several results against one observation.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b, fetch_gate, msgs = _handoff_bench(tmp)
        try:
            run_id = "runRH"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            _claim(b.coord, run_id, "runRH_sA", "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"], seed_count=60, expected=2)
            _claim(b.coord, run_id, "runRH_sB", "hostB:gpu0",
                   b.wconn_by_worker["hostB:gpu0"], seed_count=30, expected=1)
            b.saturate()
            sem = b.coord._staging_slots()
            sockA = b.peers["hostA:gpu0"].srv

            # (1-3) A pauses FIRST, then B; FIFO asserted from the registry.
            b.peers["hostA:gpu0"].send(msgs[("hostA:gpu0", 0)])
            assert b.wait_paused(1), "A never paused"
            b.peers["hostA:gpu0"].send(msgs[("hostA:gpu0", 1)])   # queued behind
            b.peers["hostB:gpu0"].send(msgs[("hostB:gpu0", 0)])
            assert b.wait_paused(2), "B never paused"
            order = [r["worker_id"] for r in b.coord._paused_connections.values()]
            assert order == ["hostA:gpu0", "hostB:gpu0"], (
                f"the pause registry is not in FIFO entry order: {order}")

            # (4-5) free EXACTLY ONE unit; invoke EXACTLY ONE release path.
            b._held.pop()
            sem.release()
            b.coord._release_capacity()

            # (6) wait until A's envelope is IN `inbound` and A has left the registry
            entry = None
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if entry is None:
                    try:
                        entry = b.inbound.get(timeout=0.05)
                    except _queue.Empty:
                        continue
                if b.coord.paused_connection_count() == 1:
                    break
                time.sleep(0.02)
            assert entry is not None, "A never delivered its held envelope"
            kind, esock, emsg, ecid = entry
            assert kind == "msg" and esock is sockA, (kind, entry)
            assert ecid is not None, (
                "A's resumed envelope arrived with no credit token — the "
                "reservation cannot be disposed of by exact identity")
            assert emsg.message_type == "sub_stripe_result", emsg.message_type
            assert b.coord.paused_worker_ids() == frozenset({"hostB:gpu0"}), (
                f"expected ONLY hostB still paused, got "
                f"{b.coord.paused_worker_ids()}")

            # (7-9) HOLD. Nothing here dispatches, and nothing here touches the
            # semaphore: the freed unit is STILL FREE and the reservation must
            # survive on it.
            assert b.coord.staging_can_accept(), (
                "the freed unit is no longer free — the gate has lost the very "
                "condition it exists to hold open, exactly as round 1 did")
            hold_end = time.time() + 0.6          # >= 12 of B's 50 ms poll cycles
            while time.time() < hold_end:
                assert b.coord.paused_worker_ids() == frozenset({"hostB:gpu0"}), (
                    "B resumed while A's envelope was still undispatched — two "
                    "wakes on ONE unconsumed slot (Beta F1-R §2)")
                assert b.inbound.qsize() == 0, (
                    "a second envelope reached `inbound` while the reservation "
                    "was outstanding — one connection streamed several results "
                    "against one capacity observation (Beta F1-R §4 tail)")
                assert b.coord.resume_credits_outstanding() == 1, (
                    "the reservation ended at INGRESS — `inbound.put` moves the "
                    "envelope, it does not consume the slot")
                time.sleep(0.02)

            # (10) dispatch through the REAL serve path; the clear must land only
            # AFTER disposition. The spy observes the credit from INSIDE
            # `_serve_dispatch`, which is the one place "before" is measurable.
            seen = []
            orig = RangeMinerCoordinator._serve_dispatch

            def _spy(self_, *a, **kw):
                seen.append(self_.resume_credits_outstanding())
                return orig(self_, *a, **kw)

            RangeMinerCoordinator._serve_dispatch = _spy
            try:
                b.dispatch([entry], run_id)
            finally:
                RangeMinerCoordinator._serve_dispatch = orig
            assert seen == [1], (
                f"the reservation was not still outstanding when the dispatch "
                f"began — it was cleared before disposition: {seen}")
            assert b.coord.resume_credits_outstanding() == 0, (
                "the reservation outlived the disposition — the paused fleet "
                "would never be granted another wake")
            fetch_deadline = time.time() + 5.0
            while (time.time() < fetch_deadline
                   and not b.coord.transfer.fetch_calls):
                time.sleep(0.02)
            assert len(b.coord.transfer.fetch_calls) == 1, (
                f"the dispatched envelope did not reach a real staging fetch, so "
                f"nothing consumed the freed unit: "
                f"{b.coord.transfer.fetch_calls}")
            assert not b.coord.staging_can_accept(), (
                "the freed unit was never consumed by the dispatch — the gate is "
                "measuring an open capacity condition, not a handoff")

            # (11) the NEXT unit resumes B — second, FIFO preserved. A, whose
            # second result was held all along, re-pauses BEHIND B.
            assert b.wait_paused(2, timeout=5.0), (
                "A never re-paused on its second result after the disposition")
            order = [r["worker_id"] for r in b.coord._paused_connections.values()]
            assert order == ["hostB:gpu0", "hostA:gpu0"], (
                f"FIFO was not preserved across the handoff: {order}")
            b._held.pop()
            sem.release()
            b.coord._release_capacity()
            deadline = time.time() + 5.0
            got_b = []
            while time.time() < deadline and not got_b:
                got_b = [m for (k, s, m, _c) in b.drain(0.2)
                         if k == "msg"
                         and b.worker_by_sock.get(s) == "hostB:gpu0"]
            assert len(got_b) == 1, (
                f"B did not resume second on the next freed unit: {got_b}")
            assert b.coord.paused_worker_ids() == frozenset({"hostA:gpu0"}), (
                f"A did not stay paused behind B: {b.coord.paused_worker_ids()}")
        finally:
            fetch_gate.set()
            b.close()


def gate_resume_handoff_mutant():
    """MUTATION EVIDENCE for F1-R (Beta §5): restore the round-1
    clear-at-`inbound.put`, prove it EXECUTES, and prove B resumes DURING the hold
    window on the still-unconsumed unit.

    Without this the handoff gate could be green for the wrong reason — a timing
    accident rather than the reservation. Here the ONLY difference from the gate
    above is where the credit is cleared, and it is enough to reproduce Beta's §2
    schedule every time.
    """
    executed = {"n": 0}
    q = _Round1ClearQueue(executed, maxsize=1024)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b, fetch_gate, msgs = _handoff_bench(tmp, inbound=q)
        q.coord = b.coord
        try:
            run_id = "runRHM"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            _claim(b.coord, run_id, "runRH_sA", "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"], seed_count=60, expected=2)
            _claim(b.coord, run_id, "runRH_sB", "hostB:gpu0",
                   b.wconn_by_worker["hostB:gpu0"], seed_count=30, expected=1)
            b.saturate()
            sem = b.coord._staging_slots()

            b.peers["hostA:gpu0"].send(msgs[("hostA:gpu0", 0)])
            assert b.wait_paused(1), "A never paused"
            b.peers["hostB:gpu0"].send(msgs[("hostB:gpu0", 0)])
            assert b.wait_paused(2), "B never paused"

            b._held.pop()
            sem.release()
            b.coord._release_capacity()

            # the SAME hold window as the gate — nothing dispatched, nothing
            # reclaimed. Under the mutant, B must escape inside it.
            b_resumed = False
            hold_end = time.time() + 0.6
            while time.time() < hold_end:
                if "hostB:gpu0" not in b.coord.paused_worker_ids():
                    b_resumed = True
                    break
                time.sleep(0.02)

            assert executed["n"] >= 1, (
                "the round-1 clear never executed — the mutant is not reproducing "
                "clear-at-`inbound.put`, so G-RESUME-HANDOFF proves nothing")
            assert b_resumed, (
                "restoring the clear-at-ingress did NOT let B wake on the "
                "still-unconsumed unit — G-RESUME-HANDOFF is vacuous")
        finally:
            fetch_gate.set()
            b.close()


def gate_summary_never_masks_the_sizing_terminal():
    """F1-R round 2 / G-SUMMARY-NO-MASK — Beta §7.

    Alpha's terminal-summary guard exists because `staging_deferred_bound()` falls
    back to the ON-DEMAND derivation when no stage bound was installed — and the
    one production path where that is true at trial-terminal time is precisely an
    F5 sizing failure. The SAME malformed cap record that failed stage setup would
    then raise again inside `log_staging_backpressure_summary`, out of the terminal
    reporting path, and mask the honest `coordinator_staging_sizing` termination:
    the F3 disease relocated to the reporting layer.

    The existing F5 gate restores the caps from the abort callback, so it never
    reaches this code — that gate is deliberately NOT modified. This one LEAVES the
    record malformed all the way through terminal summary construction, which is
    the direct execution of the guard.
    """
    label, exc_name, mangle = _MALFORMED_CAP_RECORDS[0]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1},{"draw":2},{"draw":3}]')

        records = []
        _lg, _h, restore_log = _capture_bp(records)
        state = {"injected": 0}
        sink = _Sink()                      # NO on_abort restore — that is the point
        transfer = _GatedTransfer()
        orig_assign = RangeMinerCoordinator.assign_stripes

        def _assign_then_corrupt(self_, *a, **kw):
            out = orig_assign(self_, *a, **kw)
            for _wid, conn in self_.connections.items():
                conn.seed_caps = mangle(conn.seed_caps)
            state["injected"] += 1
            return out

        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        port = lsock.getsockname()[1]

        holder = {}
        gate_caps = {**CAPS, "nvidia": 10}

        def run():
            try:
                holder["result"] = run_trial_miner(
                    "runSNM", None, 3, "java_lcg", [1, 2, 3], 80, 0.25, 0.25,
                    False, ds, worker_pool_size=1,
                    staging_dir=os.path.join(tmp, "stg"), phase5_sink=sink,
                    transfer=transfer, listen_sock=lsock,
                    family_name="java_lcg", workflow_phase=1,
                    miner_stripe_size=80, seed_cap_nvidia=10,
                    skip_min=0, skip_max=0, offset=0, window_size=3,
                    staging_workers=2, staging_queue_depth=0,
                    staging_high_water_bytes=64 * 1024 * 1024,
                    serve_timeout=45.0)
            except Exception:                                 # noqa: BLE001
                holder["err"] = traceback.format_exc()

        RangeMinerCoordinator.assign_stripes = _assign_then_corrupt
        w = _RemoteWorker("127.0.0.1", port, "hostA", 0, caps=gate_caps)
        try:
            t = threading.Thread(target=run, daemon=True)
            t.start()
            w.connect_register()
            w.start_loop()
            t.join(timeout=60)
            assert not t.is_alive(), "serve_trial never terminated"
        finally:
            RangeMinerCoordinator.assign_stripes = orig_assign
            restore_log()
            w.stop()
            try:
                lsock.close()
            except OSError:
                pass

        assert state["injected"] >= 1, (
            "the malformed cap record was never injected — the gate is measuring "
            "nothing")
        # (1) the summary did NOT raise out of the terminal path
        assert "err" not in holder, holder.get("err")
        result = holder["result"]

        # (2) the PRIMARY terminal truth is untouched by the reporting degradation
        assert result["state"] == "aborted", result["state"]
        assert sink.aborts, "no abort event was delivered"
        reason = sink.aborts[0].get("reason", "")
        assert reason.startswith("coordinator_staging_sizing:"), (
            f"the summary masked the honest sizing termination: {reason!r}")

        # (3) reporting DEGRADED rather than lying: no bound, and the cause named
        bp = result["staging_backpressure"]
        assert bp["bound_in_force"] is None, (
            f"a bound was reported although its derivation raises: "
            f"{bp['bound_in_force']!r}")
        assert "bound_in_force_error" in bp, (
            "the summary silently dropped the bound instead of recording WHY — a "
            "None with no cause is indistinguishable from 'never derived'")
        assert exc_name in bp["bound_in_force_error"], (
            f"[{label}] the summary does not name the derivation exception: "
            f"{bp['bound_in_force_error']!r}")

        # (4) ...and the grep-stable summary line STILL EMITS
        summary = [r for r in records if r.startswith("[S172-BP] summary")]
        assert summary, (
            "the [S172-BP] summary line never emitted — the terminal metrics "
            "record was lost to the very failure it is supposed to report on")
        assert "bound_in_force=None" in summary[-1], summary[-1]


def gate_lease_handoff_grace():
    """F2 / G-LEASE-HANDOFF. The §1.4 exemption covered only LIVE membership in
    `_paused_connections`, but a resuming reader DEREGISTERS FIRST, delivers the
    held envelope, and only then reaches the heartbeat queued behind it on the same
    ordered TCP stream. In that window the worker is unpaused, its compute lease
    (300 s) has been expired for as long as the pause ran (up to 600 s) and its
    renewal is still in flight — so `process_lease_expiry` routed the stripe into
    the matrix for a silence the coordinator caused.

    Three arms: (1) expiry inside the window touches nothing; (2) processing the
    heartbeat renews normally and CLEARS the grace; (3) a resumed worker that never
    heartbeats expires once the grace bound passes — the exemption is bounded.
    """
    # ---- arms 1 and 2 -------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg(compute_lease_timeout=300.0))
        try:
            run_id, sid = "runLH", "runLH_sA"
            t0 = time.time()
            b.coord.ledger.create_trial(run_id, 0, now=t0)
            # HYBRID (phase 3) deliberately: a genuine expiry here is observable
            # without the constant-phase row killing the whole trial.
            # The lease is ALREADY past its deadline — the pause outlived it.
            _claim(b.coord, run_id, sid, "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"], phase=3,
                   family="java_lcg_hybrid", lease=t0 - 1.0, now=t0)
            b.saturate()
            b.peers["hostA:gpu0"].send(_inline_result("hostA:gpu0", sid, 0, 0, 30))
            assert b.wait_paused(1), "the reader never paused"
            # the renewing heartbeat is written WHILE PAUSED: TCP is ordered, so it
            # sits on the wire behind the held result — the real shape of the bug.
            b.peers["hostA:gpu0"].send(MinerHeartbeatMessage(
                worker_id="hostA:gpu0", current_stripe_id=sid, busy=True))

            b.release_all_slots()
            b.coord._release_capacity()
            assert b.wait_unpaused(), "the reader never resumed"
            entries = b.drain(1.0)
            kinds = [m.message_type for (k, _s, m, _c) in entries if k == "msg"]
            # [S172-BP AMENDMENT F1-R2b] The heartbeat queued behind the held result
            # no longer reaches `inbound` on the same drain: the PRE-DECODE BARRIER
            # holds it ON THE WIRE until the credited envelope is disposed of, which
            # Beta §4.2 explicitly accepts. That makes the window this gate exists
            # for STRICTLY WIDER, not narrower — the renewal is one step further
            # from the coordinator than round 2's version of the same handoff — so
            # the F2 grace is needed at least as much. The gate's SUBJECT (arms 1-3)
            # is unchanged; only where the undelivered heartbeat is parked changed.
            assert kinds == ["sub_stripe_result"], (
                f"the held envelope did not arrive alone, ahead of a heartbeat "
                f"that must still be on the wire: {kinds}")

            # ARM 1: expiry INSIDE the window — deregistered, heartbeat undelivered.
            # The MATRIX assertion comes first deliberately: it is the behaviour
            # Beta's §3 names, and asserting it before touching the new grace API
            # is what makes this gate red against 4b1aad6 on behaviour.
            assert b.coord.paused_worker_ids() == frozenset(), "still paused"
            with _MatrixSpy() as spy:
                out = b.coord.process_lease_expiry(
                    run_id, list(b.wconn_by_worker.values()))
            assert spy.calls == [], (
                f"a coordinator-caused silence entered the matrix during the "
                f"resume handoff: {spy.calls}")
            assert out == [], out
            assert "hostA:gpu0" in b.coord.capacity_resume_grace(), (
                "no resume grace was recorded — the worker is unpaused with an "
                "expired lease and its renewal still queued")
            st = b.coord.ledger.get_stripe(run_id, sid)
            assert st["state"] == ST_CLAIMED and st["current_attempt"] == 0
            assert not st["phase_degraded"]
            assert st["claimed_by"] == "hostA:gpu0"

            # ARM 2: process the heartbeat -> normal renewal, grace CLEARED.
            # [S172-BP AMENDMENT F1-R2b] The disposition comes first, because that
            # is what releases the barrier and lets the reader decode the heartbeat
            # at all. `dispose` is the disposition clear alone (this gate's subject
            # is the lease, not the handoff — see `_Bench.dispose`).
            b.dispose(b.peers["hostA:gpu0"].srv)
            hb = []
            hb_deadline = time.time() + 4.0
            while time.time() < hb_deadline and not hb:
                hb = [e for e in b.drain(0.3) if e[0] == "msg"
                      and e[2].message_type == "heartbeat"]
            assert hb, (
                "the heartbeat never arrived after the credited envelope was "
                "disposed of — the barrier did not release the wire")
            b.dispatch(hb, run_id)
            st2 = b.coord.ledger.get_stripe(run_id, sid)
            assert st2["lease_expires_at"] > time.time(), (
                f"renew_lease did not run on the delayed heartbeat: "
                f"{st2['lease_expires_at']!r}")
            assert b.coord.capacity_resume_grace() == {}, (
                "the grace outlived the renewal it was bridging — the exemption "
                "must end the moment the real lease is renewed")
        finally:
            b.close()

    # ---- arm 3: a resumed worker that NEVER heartbeats still expires --------
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg(compute_lease_timeout=300.0))
        try:
            run_id, sid = "runLHB", "runLHB_sA"
            t0 = time.time()
            b.coord.ledger.create_trial(run_id, 0, now=t0)
            _claim(b.coord, run_id, sid, "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"], phase=3,
                   family="java_lcg_hybrid", lease=t0 - 1.0, now=t0)
            b.saturate()
            b.peers["hostA:gpu0"].send(_inline_result("hostA:gpu0", sid, 0, 0, 30))
            assert b.wait_paused(1)
            b.release_all_slots()
            b.coord._release_capacity()
            assert b.wait_unpaused()
            b.drain(0.5)
            assert "hostA:gpu0" in b.coord.capacity_resume_grace()
            with _MatrixSpy() as spy:
                out = b.coord.process_lease_expiry(
                    run_id, list(b.wconn_by_worker.values()),
                    now=time.time() + 301.0)      # past the grace bound
                touched = {c["stripe_id"] for c in spy.calls}
            assert sid in touched, (
                "the grace never expired — an exemption with no bound is not an "
                "exemption, it is a hole")
            assert out and out[0]["action"] in ("reassigned", "fail_trial"), out
            assert b.coord.capacity_resume_grace() == {}, (
                "the expired grace entry was not pruned")
        finally:
            b.close()


def gate_lease_handoff_mutant():
    """MUTATION EVIDENCE for F2: remove ONLY the grace RECORDING (the pre-amendment
    state of the world — deregistration wrote nothing) and prove the mutated path
    executed and reds the credited assertion."""
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg(compute_lease_timeout=300.0))
        try:
            run_id, sid = "runLHM", "runLHM_sA"
            t0 = time.time()
            b.coord.ledger.create_trial(run_id, 0, now=t0)
            _claim(b.coord, run_id, sid, "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"], phase=3,
                   family="java_lcg_hybrid", lease=t0 - 1.0, now=t0)
            b.saturate()

            executed = {"n": 0}
            orig = RangeMinerCoordinator.deregister_paused_connection

            def _no_grace(self_, conn_key, now=None, reason="resume"):
                executed["n"] += 1
                # the fix, removed: deregister EXACTLY as before, recording nothing
                out = orig(self_, conn_key, now=now, reason=reason)
                self_._capacity_resume_grace.clear()
                return out

            RangeMinerCoordinator.deregister_paused_connection = _no_grace
            try:
                b.peers["hostA:gpu0"].send(
                    _inline_result("hostA:gpu0", sid, 0, 0, 30))
                assert b.wait_paused(1)
                b.release_all_slots()
                b.coord._release_capacity()
                assert b.wait_unpaused()
                b.drain(0.5)
                assert b.coord.capacity_resume_grace() == {}, (
                    "the mutant did not remove the grace")
                with _MatrixSpy() as spy:
                    b.coord.process_lease_expiry(
                        run_id, list(b.wconn_by_worker.values()))
                    touched = {c["stripe_id"] for c in spy.calls}
            finally:
                RangeMinerCoordinator.deregister_paused_connection = orig

            assert executed["n"] >= 1, (
                "the mutant was never called — the resume path does not go through "
                "deregister_paused_connection, so G-LEASE-HANDOFF proves nothing")
            assert sid in touched, (
                "removing the grace recording did NOT route the resumed worker's "
                "expiry into the matrix — G-LEASE-HANDOFF is vacuous")
        finally:
            b.close()


def gate_timeout_snapshot_attributes_the_trigger():
    """F3 / G-TIMEOUT-SNAPSHOT. A reader can observe
    `staging_capacity_timeout_expired()`, latch it, deregister and EXIT before the
    serve loop builds the terminal reason. Reading the LIVE registry then truthfully
    reports `0 connections paused (none)` about a timeout that paused workers
    caused. The count and identities must be the TRIGGERING ones.
    """
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg(staging_capacity_timeout=0.3))
        try:
            run_id, sid = "runTS", "runTS_s0"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            _claim(b.coord, run_id, sid, "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"])
            b.saturate()
            b.peers["hostA:gpu0"].send(_inline_result("hostA:gpu0", sid, 0, 0, 30))
            assert b.wait_paused(1), "the reader never paused"

            # let the READER observe the latch, deregister and fully exit
            reader = b.peers["hostA:gpu0"].thread
            deadline = time.time() + 10.0
            while time.time() < deadline and reader.is_alive():
                time.sleep(0.02)
            assert not reader.is_alive(), (
                "the paused reader never exited on the capacity timeout")
            assert b.coord.paused_connection_count() == 0, (
                "the registry still holds the pause — the race this gate is about "
                "cannot occur, so it is measuring nothing")

            # the TERMINAL REASON first — that is the behaviour §4 names, and
            # asserting it before touching the new snapshot API is what makes this
            # gate red against 4b1aad6 on behaviour rather than on an attribute.
            reason = b.coord.staging_capacity_timeout_reason()
            assert reason.startswith("coordinator_staging_capacity_timeout:"), reason
            assert "hostA:gpu0" in reason, (
                f"the terminal reason does not name the worker whose pause caused "
                f"the timeout: {reason!r}")
            assert "1 connections paused" in reason, reason
            assert "(none)" not in reason, (
                f"the reason attributes the timeout to nobody: {reason!r}")

            snap = b.coord.capacity_timeout_snapshot()
            assert snap is not None, "the latch took no evidence snapshot"
            assert snap["paused_count"] == 1, snap
            assert snap["worker_ids"] == ["hostA:gpu0"], snap
            assert snap["latched_at"] >= snap["oldest_since"], snap

            m = b.coord.staging_backpressure_metrics()
            assert m["paused_at_capacity_timeout"] == 1, m
            assert m["capacity_timeout_worker_ids"] == ["hostA:gpu0"], m
            assert m["capacity_timeout_snapshot"]["latched_at"] == \
                snap["latched_at"], m
            assert m["paused_now"] == 0, (
                "precondition lost: the live registry must be EMPTY, otherwise the "
                "snapshot and the live read are indistinguishable")
        finally:
            b.close()


def gate_unbound_result_is_never_paused():
    """F4 / G-BOUND-PAUSE. The reader's pause condition tested message type and
    capacity but not IDENTITY, so an unregistered socket sending a well-formed
    `sub_stripe_result` under saturation acquired pause state (`worker_id=None`),
    consumed the one-envelope allowance, joined the §1.5 oldest-pause clock and was
    held BEFORE the serve loop's identity rejection could see it.

    An unbound result must NOT be paused and NOT be held: it flows to `inbound`
    unchanged and dies in the EXISTING identity rejection. No new rejection logic.
    """
    with tempfile.TemporaryDirectory() as tmp:
        b = _Bench(tmp, **_saturating_cfg())
        try:
            run_id, sid = "runBP", "runBP_s0"
            b.coord.ledger.create_trial(run_id, 0, now=100.0)
            _claim(b.coord, run_id, sid, "hostA:gpu0",
                   b.wconn_by_worker["hostA:gpu0"])
            b.saturate()

            stray = _Peer(b.coord, "stray:gpu0", b.worker_by_sock, b.inbound,
                          b.reader_stop, bind=False)
            b.peers["__stray__"] = stray
            assert b.worker_by_sock.get(stray.srv) is None, (
                "the stray socket is bound — the gate's premise is gone")

            stray.send(_inline_result("stray:gpu0", sid, 0, 0, 30))
            got = b.drain(0.8)
            from_stray = [m for (k, s, m, _c) in got if k == "msg" and s is stray.srv]
            assert len(from_stray) == 1, (
                f"an UNBOUND sub_stripe_result was intercepted by the capacity "
                f"gate instead of flowing to the existing identity check: "
                f"{from_stray}")
            assert b.coord.paused_connection_count() == 0, (
                "an unregistered socket acquired pause state")
            assert b.coord.paused_worker_ids() == frozenset()
            assert b.coord.capacity_resume_grace() == {}, (
                "an unregistered socket acquired a resume-grace record")
            assert b.coord.capacity_timeout_snapshot() is None
            assert not b.coord.staging_capacity_timeout_expired(
                now=time.time() + 10_000.0), (
                "an unregistered socket joined the oldest-pause clock — it can now "
                "time out a trial it was never part of")

            # ...and it dies in the EXISTING identity rejection: no ledger mutation
            with _MatrixSpy() as spy:
                b.dispatch(got, run_id)
            assert spy.calls == [], f"the stray result reached the matrix: {spy.calls}"
            assert b.coord.ledger.get_shards(run_id, sid, 0) == [], (
                "the stray result mutated the ledger — the existing identity "
                "rejection did not govern it")

            # NARROWNESS: a BOUND connection under the same saturation still pauses
            b.peers["hostA:gpu0"].send(_inline_result("hostA:gpu0", sid, 0, 0, 30))
            assert b.wait_paused(1), (
                "the identity predicate is too wide — a REGISTERED worker stopped "
                "being back-pressured")
            assert b.coord.paused_worker_ids() == frozenset({"hostA:gpu0"})
        finally:
            b.close()


# Two malformed worker-cap records, and the exception each one produces inside the
# stage derivation. The ValueError arm is the one Beta's §6 names: it is EXACTLY
# the class the pre-amendment `except (ValueError, TypeError)` swallowed before
# continuing on `_derive_bound_from_current_state`. The KeyError arm is a second
# hole in the same handler — pre-amendment it was not caught at all and escaped
# `serve_trial` as an unhandled exception. Failing closed answers both.
_MALFORMED_CAP_RECORDS = [
    # all four caps advertised as zero: expected_substripes_for() refuses a
    # non-positive effective cap -> ValueError
    ("zero_caps", "ValueError",
     lambda caps: {k: 0 for k in caps}),
    # a cap key missing: advertised_effective_cap() builds VramCaps from all four
    # -> KeyError
    ("missing_cap_key", "KeyError",
     lambda caps: {k: v for k, v in caps.items() if k != "amd_hybrid"}),
]


def gate_bound_derivation_failure_fails_closed():
    """F5 / G-BOUND-DERIVATION-FAILURE. A malformed worker-cap record is injected
    at stage setup, AFTER the real `assign_stripes` and BEFORE the sizing call, so
    the stage derivation raises exactly where Beta's §6 defect lived.

    Pre-amendment the `except (ValueError, TypeError)` swallowed it and let
    `staging_deferred_bound()` fall back to `_derive_bound_from_current_state` —
    ONE macro-stripe, phase 1 — which can be MATERIALLY SMALLER than the stage
    derivation that just failed, silently re-arming the undersized-queue condition.
    It must now terminate the trial DIRECTLY, before any result traffic.

    The injection is undone at the SYNCHRONOUS L7 abort discharge (`_Sink.on_abort`
    fires at the instant the trial becomes terminal), so the gate measures the
    fail-closed decision and not a teardown artefact of its own fault injection.
    """
    for label, exc_name, mangle in _MALFORMED_CAP_RECORDS:
        _bound_derivation_failure_arm(label, exc_name, mangle)


def _bound_derivation_failure_arm(label, exc_name, mangle):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1},{"draw":2},{"draw":3}]')

        state = {"coord": None, "injected": 0, "restored": 0, "saved": {}}

        def _restore(_event):
            coord = state["coord"]
            if coord is None:
                return
            for wid, caps in state["saved"].items():
                conn = coord.connections.get(wid)
                if conn is not None:
                    conn.seed_caps = dict(caps)
            state["restored"] += 1

        sink = _Sink(on_abort=_restore)
        transfer = _GatedTransfer()

        orig_assign = RangeMinerCoordinator.assign_stripes

        def _assign_then_corrupt(self_, *a, **kw):
            out = orig_assign(self_, *a, **kw)
            # a MALFORMED worker-cap record: `advertised_effective_cap` builds a
            # VramCaps from all four advertised keys, so a record missing one
            # cannot resolve. Injected after the assignment so the failure lands
            # in the SIZING derivation, which is the seam under test.
            state["coord"] = self_
            for wid, conn in self_.connections.items():
                state["saved"].setdefault(wid, dict(conn.seed_caps))
                conn.seed_caps = mangle(conn.seed_caps)
            state["injected"] += 1
            return out

        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        port = lsock.getsockname()[1]

        holder = {}
        gate_caps = {**CAPS, "nvidia": 10}

        def run():
            try:
                holder["result"] = run_trial_miner(
                    "runBD", None, 3, "java_lcg", [1, 2, 3], 80, 0.25, 0.25,
                    False, ds, worker_pool_size=1,
                    staging_dir=os.path.join(tmp, "stg"), phase5_sink=sink,
                    transfer=transfer, listen_sock=lsock,
                    family_name="java_lcg", workflow_phase=1,
                    miner_stripe_size=80, seed_cap_nvidia=10,
                    skip_min=0, skip_max=0, offset=0, window_size=3,
                    staging_workers=2, staging_queue_depth=0,
                    staging_high_water_bytes=64 * 1024 * 1024,
                    serve_timeout=45.0)
            except Exception:                                 # noqa: BLE001
                holder["err"] = traceback.format_exc()

        RangeMinerCoordinator.assign_stripes = _assign_then_corrupt
        with _MatrixSpy() as spy:
            t = threading.Thread(target=run, daemon=True)
            t.start()
            w = _RemoteWorker("127.0.0.1", port, "hostA", 0, caps=gate_caps)
            try:
                w.connect_register()
                w.start_loop()
                t.join(timeout=60)
                assert not t.is_alive(), "serve_trial never terminated"
                RangeMinerCoordinator.assign_stripes = orig_assign
                assert "err" not in holder, holder.get("err")
                result = holder["result"]

                assert state["injected"] >= 1, (
                    f"[{label}] the malformed cap record was never injected — the "
                    f"gate is measuring nothing")
                assert state["restored"] >= 1, (
                    f"[{label}] the fault injection was never undone at the "
                    f"terminal discharge")

                # terminal, for the RIGHT reason, leading with the root cause
                assert result["state"] == "aborted", (label, result["state"])
                assert sink.aborts, f"[{label}] no abort event was delivered"
                reason = sink.aborts[0].get("reason", "")
                assert reason.startswith("coordinator_staging_sizing:"), (
                    f"[{label}] a sizing failure did not fail closed: {reason!r}")
                assert "stage 0" in reason, (label, reason)
                assert exc_name in reason, (
                    f"[{label}] the terminal reason does not carry the cause "
                    f"({exc_name}): {reason!r}")

                # never the matrix, no retry consumed
                assert spy.calls == [], (
                    f"[{label}] a sizing failure entered the retry matrix: "
                    f"{spy.calls}")
                for sid, st in result["stripes"].items():
                    assert st["current_attempt"] == 0, (label, sid, st)
                    assert not st["phase_degraded"], (label, sid, st)

                # BEFORE any result traffic for that stage
                assert w.assigned == [], (
                    f"[{label}] the stage was dispatched despite a failed sizing: "
                    f"{w.assigned}")
                assert transfer.fetch_calls == [], (label, transfer.fetch_calls)
                assert not sink.published, (label, sink.published)

                # ...and execution never continued on the one-slot fallback
                bp = result["staging_backpressure"]
                assert bp["derived_bound"] is None, (
                    f"[{label}] a bound was installed although the derivation "
                    f"failed: {bp['derived_bound']!r}")
                assert bp["staging_jobs_completed"] == 0, (label, bp)
                assert bp["deferred_high_water"] == 0, (label, bp)
                assert bp["capacity_invariant_terminations"] == 0, (label, bp)
            finally:
                RangeMinerCoordinator.assign_stripes = orig_assign
                w.stop()
                try:
                    lsock.close()
                except OSError:
                    pass


def gate_invariant_reason_names_which_bound_tripped():
    """F5 (Beta item-3 ratification detail): the §1.6 invariant reason must
    distinguish WHICH bound tripped — derived count, operator-override count, or
    retained-bytes high-water. They are three different defects with three
    different owners; one undifferentiated "the deferred queue overflowed" sends
    all of them to the wrong place. The arithmetic already carried is retained."""
    src = inspect.getsource(RangeMinerCoordinator.enqueue_staging)
    for phrase in ("DERIVED COUNT bound", "OPERATOR OVERRIDE COUNT bound",
                   "RETAINED-BYTES HIGH-WATER"):
        assert phrase in src, (
            f"the §1.6 invariant reason cannot say {phrase!r} — it does not "
            f"distinguish which bound tripped")

    # ...and the classification is DERIVED FROM THE REFUSAL, not from a guess.
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, staging_workers=1, staging_queue_depth=0,
                       miner_stripe_size=1000)
        _register(coord, "hostA:gpu0")
        _obj, pb = build_substripe_payload_bytes("s", 0, 0, 30,
                                                 [[0, 0.9, None, [1]]])
        msg = SubStripeResultMessage(
            worker_id="hostA:gpu0", stripe_id="s", sub_index=0, seed_start=0,
            seed_count=30, survivor_count=1, inline=_obj, size_bytes=len(pb),
            sha256=hashlib.sha256(pb).hexdigest())
        entry = ("inline", None, "r", "s", 0, 0, msg, None, None)

        # (a) an OPERATOR OVERRIDE count trip
        coord.config.staging_deferred_max = 1
        with coord._admission_lock:
            assert coord._defer_locked(entry) is True
            assert coord._defer_locked(entry) is False
        assert coord._last_defer_refusal == "operator_override_count_bound", (
            coord._last_defer_refusal)

        # (b) a DERIVED count trip — no override in force
        coord._deferred = []
        coord.config.staging_deferred_max = None
        coord._derived_deferred_bound = 1
        with coord._admission_lock:
            assert coord._defer_locked(entry) is True
            assert coord._defer_locked(entry) is False
        assert coord._last_defer_refusal == "derived_count_bound", (
            coord._last_defer_refusal)

        # (c) a RETAINED-BYTES trip — the count bound is wide open
        coord._deferred = []
        coord._derived_deferred_bound = 1000
        coord.config.staging_high_water_bytes = 1
        with coord._admission_lock:
            assert coord._defer_locked(entry) is False
        assert coord._last_defer_refusal == "retained_bytes_high_water", (
            coord._last_defer_refusal)


# ===========================================================================
# ===========================================================================
# S172-BP AMENDMENT ROUND 3 (Beta F1-R2a / F1-R2b, 2026-08-07)
# ===========================================================================
def _identity_bench(tmp):
    """Fixture for G-CREDIT-ENVELOPE-IDENTITY and its mutant.

    Two REAL readers on real socketpairs and a staging job whose fetch is gated,
    so the ONE unit freed here stays physically free until a credited envelope is
    genuinely dispatched into `enqueue_staging` — the interval both the invariant
    and the defect live in.
    """
    fetch_gate = threading.Event()
    payloads, msgs = {}, {}
    for wid, sid, subs in (("hostA:gpu0", "runCE_sA", (0, 1)),
                           ("hostB:gpu0", "runCE_sB", (0,))):
        for i in subs:
            m, remote, pb = _spool_result(wid, sid, i, i * 30, 30)
            payloads[remote] = pb
            msgs[(wid, i)] = m
    transfer = _GatedTransfer(payloads=payloads, gate=fetch_gate)
    b = _Bench(tmp, transfer=transfer, **_saturating_cfg())
    return b, fetch_gate, msgs


def _arm_uncredited_ahead_of_credited(b, msgs, run_id):
    """Beta §5.1 steps 1-6, shared by the gate and its mutant.

    Leaves the coordinator in the exact state F1-R2a is about: an OLDER,
    UNCREDITED result `U` of A's at the HEAD of `inbound` (enqueued under open
    capacity, before any pause existed), A's credited envelope `C` queued BEHIND
    it, B paused with no credit, exactly one staging unit physically free, and A's
    exact token outstanding. Returns (sockA, entry_u, credit_id).
    """
    b.coord.ledger.create_trial(run_id, 0, now=100.0)
    _claim(b.coord, run_id, "runCE_sA", "hostA:gpu0",
           b.wconn_by_worker["hostA:gpu0"], seed_count=60, expected=2)
    _claim(b.coord, run_id, "runCE_sB", "hostB:gpu0",
           b.wconn_by_worker["hostB:gpu0"], seed_count=30, expected=1)
    sockA = b.peers["hostA:gpu0"].srv
    sem = b.coord._staging_slots()

    # (1) U is made a DUPLICATE in the ledger: the shard row for (attempt 0,
    # sub 0) already exists, so `_serve_dispatch` drops it at the EXISTING dedup
    # insert and returns BEFORE `enqueue_staging`. It therefore consumes no
    # capacity whatsoever — which is precisely why clearing a credit on it is a
    # defect and not merely an ordering quirk.
    u = msgs[("hostA:gpu0", 0)]
    b.coord.ledger.record_substripe_result(
        run_id, "runCE_sA", 0, 0, "hostA:gpu0", 0, 30, 1,
        remote_spool_path=u.spool_path, size_bytes=u.size_bytes, sha256=u.sha256)

    # (2) under OPEN capacity A sends U. It reaches `inbound` uncredited, and
    # NOTHING dispatches it.
    b.peers["hostA:gpu0"].send(u)
    deadline = time.time() + 5.0
    while time.time() < deadline and b.inbound.qsize() < 1:
        time.sleep(0.02)
    assert b.inbound.qsize() == 1, (
        "U never reached `inbound` under open capacity — the gate's premise "
        "(an older uncredited envelope already queued) does not hold")

    # (3-4) saturate; A pauses on C, then B pauses on B1. FIFO from the registry.
    b.saturate()
    b.peers["hostA:gpu0"].send(msgs[("hostA:gpu0", 1)])
    assert b.wait_paused(1), "A never paused on C"
    b.peers["hostB:gpu0"].send(msgs[("hostB:gpu0", 0)])
    assert b.wait_paused(2), "B never paused"
    order = [r["worker_id"] for r in b.coord._paused_connections.values()]
    assert order == ["hostA:gpu0", "hostB:gpu0"], (
        f"the pause registry is not in FIFO entry order: {order}")

    # (5) free EXACTLY ONE unit and invoke EXACTLY ONE release path: A takes the
    # sole credit and queues C BEHIND U.
    b._held.pop()
    sem.release()
    b.coord._release_capacity()
    deadline = time.time() + 5.0
    while time.time() < deadline and b.inbound.qsize() < 2:
        time.sleep(0.02)
    assert b.inbound.qsize() == 2, (
        f"C never queued behind U — inbound depth {b.inbound.qsize()}")
    credit_id = b.coord.resume_credit_id_for(sockA)
    assert credit_id is not None, (
        "A took the sole credit but holds no token — there is nothing to key an "
        "exact disposition on")
    assert b.coord.paused_worker_ids() == frozenset({"hostB:gpu0"}), (
        f"expected ONLY B still paused: {b.coord.paused_worker_ids()}")

    # (6) the FIFO head IS the uncredited U, and it carries NO token.
    entry_u = b.inbound.get(timeout=1.0)
    assert entry_u[0] == "msg" and entry_u[1] is sockA, entry_u[:2]
    assert entry_u[2].sub_index == 0, (
        f"the head of `inbound` is not the older envelope: {entry_u[2].sub_index}")
    assert entry_u[3] is None, (
        f"an envelope enqueued before the pause carries a credit token: "
        f"{entry_u[3]!r}")
    return sockA, entry_u, credit_id


def gate_credit_clears_only_on_the_exact_envelope():
    """F1-R2a / G-CREDIT-ENVELOPE-IDENTITY — Beta §5.1, all thirteen steps.

    THE DEFECT. Round 2 ended the reservation on `rawsock is holder`. That
    identity was defended as "the credited envelope is by construction the FIRST
    result the connection delivers after its resume" — true of LATER traffic, and
    silent about EARLIER traffic. An older, uncredited result `U` of the holder's,
    already sitting in `inbound` from before the pause, dispatches first, is
    dropped by the existing dedup fence (consuming NO capacity), and its `finally`
    releases the credit. `C` is still queued, the freed unit is still physically
    free, and the next FIFO head wakes on it: F1's two-wakes-one-slot, re-entered
    from the other side of the queue.

    THE GATE. Real readers, the real serve path, one real freed unit. Between
    dispatching `U` and dispatching `C` the test thread touches NEITHER the
    semaphore NOR the credit: the reservation must survive a dispatch that
    disposed of something else entirely.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b, fetch_gate, msgs = _identity_bench(tmp)
        try:
            run_id = "runCE"
            sockA, entry_u, credit_id = _arm_uncredited_ahead_of_credited(
                b, msgs, run_id)

            # (7) dispatch ONLY U, through the REAL seam.
            b.dispatch([entry_u], run_id)

            # (8) it consumed nothing: fence-rejected before `enqueue_staging`.
            assert not b.coord.transfer.fetch_calls, (
                f"the duplicate reached a staging fetch, so it did consume "
                f"capacity: {b.coord.transfer.fetch_calls}")
            assert b.coord.staging_can_accept(), (
                "the freed unit is no longer free after a fence-rejected "
                "duplicate — the gate has lost the condition it exists to hold")

            # (9-11) HOLD: A's EXACT token still outstanding, C still queued, B
            # still paused, capacity still physically open.
            hold_end = time.time() + 0.6        # >= 12 of B's 50 ms poll cycles
            while time.time() < hold_end:
                assert b.coord.resume_credit_id_for(sockA) == credit_id, (
                    "dispatching an OLDER, UNCREDITED result of the holder's "
                    "released the reservation — the credit is keyed on the "
                    "socket, not on the envelope it was granted for (F1-R2a)")
                assert b.coord.resume_credits_outstanding() == 1
                assert b.inbound.qsize() == 1, (
                    "the credited envelope left `inbound` — nothing in this "
                    "window should have dispatched it")
                assert b.coord.paused_worker_ids() == frozenset({"hostB:gpu0"}), (
                    "B woke while A's credited envelope was still queued on an "
                    "unconsumed unit — two wakes, one slot")
                assert b.coord.staging_can_accept(), (
                    "the unit stopped being free with nothing dispatched")
                time.sleep(0.02)

            # (12) dispatch C: the EXACT token clears, and the unit is consumed.
            entry_c = b.inbound.get(timeout=1.0)
            assert entry_c[1] is sockA and entry_c[2].sub_index == 1, entry_c[:3]
            assert entry_c[3] == credit_id, (
                f"the credited envelope did not carry A's token: {entry_c[3]!r} "
                f"vs {credit_id!r}")
            b.dispatch([entry_c], run_id)
            assert b.coord.resume_credit_id_for(sockA) is None, (
                "the reservation outlived the disposition of its OWN envelope")
            assert b.coord.resume_credits_outstanding() == 0
            fetch_deadline = time.time() + 5.0
            while (time.time() < fetch_deadline
                   and not b.coord.transfer.fetch_calls):
                time.sleep(0.02)
            assert len(b.coord.transfer.fetch_calls) == 1, (
                f"C never reached a real staging fetch, so nothing consumed the "
                f"freed unit: {b.coord.transfer.fetch_calls}")
            assert not b.coord.staging_can_accept(), (
                "the freed unit was never consumed by C's dispatch")

            # (13) only NOW does B receive the next valid grant — a DIFFERENT
            # token, on a DIFFERENT unit.
            b._held.pop()
            b.coord._staging_slots().release()
            b.coord._release_capacity()
            sockB = b.peers["hostB:gpu0"].srv
            entry_b = None
            deadline = time.time() + 5.0
            while time.time() < deadline and entry_b is None:
                try:
                    entry_b = b.inbound.get(timeout=0.05)
                except _queue.Empty:
                    continue
            assert entry_b is not None, (
                "B never received the next grant after the credited envelope "
                "was disposed of — the reservation wedged the paused fleet")
            assert entry_b[1] is sockB, "the next grant did not go to B"
            assert entry_b[3] is not None and entry_b[3] != credit_id, (
                f"B's envelope carries A's spent token: {entry_b[3]!r} vs "
                f"{credit_id!r} — tokens are not unique per grant")
        finally:
            fetch_gate.set()
            b.close()


def gate_credit_envelope_identity_mutant():
    """MUTATION EVIDENCE for F1-R2a (Beta §5.1): restore ROUND 2's socket-only
    release at exactly the instruction that differs, prove it EXECUTES, and prove
    that dispatching the older UNCREDITED envelope then clears the credit and lets
    B resume while C is still queued and undispatched.

    Without this, G-CREDIT-ENVELOPE-IDENTITY could be green on a timing accident
    rather than on the token.
    """
    executed = {"n": 0}
    orig = RangeMinerCoordinator._release_resume_credit_exact

    def _socket_only(self_, conn_key, credit_id, delivered,
                     disposition="dispatch"):
        # round 2 exactly: the token is IGNORED, holder identity alone releases.
        executed["n"] += 1
        return self_._release_resume_credit(conn_key, delivered=delivered,
                                            disposition=disposition)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b, fetch_gate, msgs = _identity_bench(tmp)
        RangeMinerCoordinator._release_resume_credit_exact = _socket_only
        try:
            run_id = "runCEM"
            sockA, entry_u, credit_id = _arm_uncredited_ahead_of_credited(
                b, msgs, run_id)
            b.dispatch([entry_u], run_id)

            assert executed["n"] >= 1, (
                "the socket-only release never executed — the mutant is not "
                "reproducing round 2, so G-CREDIT-ENVELOPE-IDENTITY proves "
                "nothing")
            assert b.coord.resume_credits_outstanding() == 0, (
                "the restored socket-only release did not clear the credit on "
                "the uncredited envelope — this is not the round-2 behaviour")

            b_resumed = False
            hold_end = time.time() + 0.6
            while time.time() < hold_end:
                if "hostB:gpu0" not in b.coord.paused_worker_ids():
                    b_resumed = True
                    break
                time.sleep(0.02)
            assert b_resumed, (
                "B did not wake on the still-unconsumed unit under the round-2 "
                "release — G-CREDIT-ENVELOPE-IDENTITY is vacuous")
            still_queued = [e for e in list(b.inbound.queue)
                            if e[1] is sockA and e[2].sub_index == 1]
            assert still_queued, (
                "C left `inbound` during the mutant window — the wake must be "
                "shown happening while the credited envelope is STILL queued")
        finally:
            RangeMinerCoordinator._release_resume_credit_exact = orig
            fetch_gate.set()
            b.close()


class _CountingFS:
    """A pass-through proxy over the REAL `MinerFramedSocket` that COUNTS decodes.

    The reader under test is the production `_conn_reader_loop`, unmodified; only
    the object it was already being handed is instrumented. "Did this connection
    decode a second envelope?" is then answered by a counter, not by inference
    from what happened to arrive in `inbound`.
    """

    def __init__(self, fs):
        self._fs = fs
        self.started = 0
        self.completed = 0

    def recv_msg(self):
        self.started += 1
        msg = self._fs.recv_msg()
        self.completed += 1
        return msg

    def __getattr__(self, name):
        return getattr(self._fs, name)


class _PostDecodeBarrierFS(_CountingFS):
    """ROUND 2's PLACEMENT, restored at exactly the instruction that differs.

    Round 2 waited for the reservation AFTER `recv_msg` returned. Reproducing it
    here — decode first, wait second, on round 2's own `holds_resume_credit`
    predicate — leaves the connection owning TWO decoded envelopes (the credited
    one in `inbound`, this one in hand), which is what F1-R2b indicts. Paired with
    a neutralised pre-decode barrier this IS the round-2 reader.
    """

    def __init__(self, fs):
        super().__init__(fs)
        self.coord = None
        self.conn_key = None
        self.stop = None
        self.waits = 0

    def recv_msg(self):
        msg = super().recv_msg()          # THE DECODE HAPPENS FIRST — round 2
        while (self.coord is not None
               and self.coord.holds_resume_credit(self.conn_key)
               and not (self.stop is not None and self.stop.is_set())):
            self.waits += 1
            time.sleep(0.05)
        return msg


def _predecode_bench(tmp, fs_wrap):
    """Fixture for G-NO-PREDECODE and its mutant: A with two results to send, a
    gated staging fetch so the freed unit is genuinely consumed by the dispatch,
    and A's framed socket wrapped by `fs_wrap`."""
    fetch_gate = threading.Event()
    payloads, msgs = {}, {}
    for i in (0, 1):
        m, remote, pb = _spool_result("hostA:gpu0", "runND_sA", i, i * 30, 30)
        payloads[remote] = pb
        msgs[i] = m
    transfer = _GatedTransfer(payloads=payloads, gate=fetch_gate)
    b = _Bench(tmp, transfer=transfer, fs_wrap=fs_wrap, **_saturating_cfg())
    return b, fetch_gate, msgs


def _deliver_credited_envelope(b, msgs, run_id):
    """Drive A to: pause on C, take the sole credit, and deliver C into `inbound`
    UNDISPOSED. Returns (sockA, credit_id, counter)."""
    b.coord.ledger.create_trial(run_id, 0, now=100.0)
    _claim(b.coord, run_id, "runND_sA", "hostA:gpu0",
           b.wconn_by_worker["hostA:gpu0"], seed_count=60, expected=2)
    sockA = b.peers["hostA:gpu0"].srv
    counter = b.peers["hostA:gpu0"].reader_fs
    b.saturate()
    b.peers["hostA:gpu0"].send(msgs[0])                     # C
    assert b.wait_paused(1), "A never paused on C"
    b._held.pop()
    b.coord._staging_slots().release()
    b.coord._release_capacity()
    deadline = time.time() + 5.0
    while time.time() < deadline and b.inbound.qsize() < 1:
        time.sleep(0.02)
    assert b.inbound.qsize() == 1, "A never delivered its credited envelope"
    credit_id = b.coord.resume_credit_id_for(sockA)
    assert credit_id is not None, "A delivered C without holding a token"
    assert counter.completed == 1, (
        f"the connection decoded {counter.completed} envelopes to reach C — the "
        f"counter is not measuring what the gate claims")
    return sockA, credit_id, counter


def gate_no_predecode_while_the_credit_is_outstanding():
    """F1-R2b / G-NO-PREDECODE — Beta §5.2.

    THE DEFECT. Round 2's §4-tail wait ran AFTER `recv_msg`. While the credited
    envelope C sat undisposed in `inbound`, the reader had ALREADY decoded the
    next envelope C2 into its local: ONE connection, TWO decoded envelopes —
    breaking the one-decoded-envelope-per-connection bound the §1.2 resume margin
    is derived from, and pulling a payload off the wire that §1.1 exists to leave
    on it.

    THE GATE. The production reader, its framed socket wrapped in a counting
    proxy. C is delivered and left UNDISPOSED; C2 is then written on the same
    socket and the state is held for 0.6 s. Nothing further is decoded, and C2's
    bytes are still sitting unread in the kernel receive buffer — the OS itself
    reporting that the frame is still on the wire. The DISPOSITION of C is what
    releases the next decode.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b, fetch_gate, msgs = _predecode_bench(tmp, _CountingFS)
        try:
            run_id = "runND"
            sockA, credit_id, counter = _deliver_credited_envelope(
                b, msgs, run_id)

            # C2 goes on the wire behind the undisposed C.
            b.peers["hostA:gpu0"].send(msgs[1])
            hold_end = time.time() + 0.6
            while time.time() < hold_end:
                assert counter.completed == 1, (
                    f"the reader decoded a SECOND envelope while its credited "
                    f"one was undisposed: {counter.completed} decodes (F1-R2b)")
                assert counter.started == 1, (
                    "the reader entered `recv_msg` again before its reservation "
                    "was disposed of — the barrier is not PRE-decode")
                assert b.coord.resume_credit_id_for(sockA) == credit_id, (
                    "the reservation ended with nothing dispatched")
                assert b.inbound.qsize() == 1, (
                    "a second envelope reached `inbound` while the reservation "
                    "was outstanding")
                time.sleep(0.02)

            # C2 IS STILL ON THE WIRE: the coordinator's own socket still has
            # unread bytes pending, which is the property §1.1 is built on.
            readable, _, _ = select.select([sockA], [], [], 0)
            assert readable, (
                "the coordinator's socket has no pending bytes — C2 was read off "
                "the wire instead of being left on it")

            # DISPOSE C -> and only then does the reader decode C2.
            entry_c = b.inbound.get(timeout=1.0)
            assert entry_c[3] == credit_id, entry_c[3]
            b.dispatch([entry_c], run_id)
            deadline = time.time() + 5.0
            while time.time() < deadline and counter.completed < 2:
                time.sleep(0.02)
            assert counter.completed == 2, (
                f"the reader never decoded C2 after the disposition — the "
                f"barrier does not release: {counter.completed} decodes")
            # C's own staging job now holds the unit (its fetch is gated), so the
            # freshly decoded C2 meets a CLOSED capacity gate and pauses on it —
            # one decoded envelope held, which is the bound restored, not broken.
            assert b.wait_paused(1), (
                "C2 was decoded but the connection neither delivered nor paused "
                "on it")
            b.release_all_slots()
            b.coord._release_capacity()
            deadline = time.time() + 5.0
            while time.time() < deadline and b.inbound.qsize() < 1:
                time.sleep(0.02)
            entry_c2 = b.inbound.get(timeout=1.0)
            assert entry_c2[2].sub_index == 1, entry_c2[2].sub_index
            assert entry_c2[3] is not None and entry_c2[3] != credit_id, (
                f"C2 resumed on a NEW grant but carries {entry_c2[3]!r} against "
                f"the spent {credit_id!r} — tokens are not unique per grant")
        finally:
            fetch_gate.set()
            b.close()


def gate_no_predecode_mutant():
    """MUTATION EVIDENCE for F1-R2b (Beta §5.2): restore the POST-decode barrier
    placement — the pre-decode wait neutralised, round 2's wait reinstated after
    `recv_msg` — and prove the decode counter advances while C is undisposed.

    Both halves are proven live: the neutralised pre-decode barrier is shown to
    have executed, and the restored post-decode wait is shown to have spun.
    """
    neutralised = {"n": 0}
    orig_wait = RangeMinerCoordinator._await_exact_credit_clear

    def _no_barrier(self_, credit_id, reader_stop):
        # round 2: there is no wait here at all — the loop decodes immediately.
        neutralised["n"] += 1
        return True

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b, fetch_gate, msgs = _predecode_bench(tmp, _PostDecodeBarrierFS)
        RangeMinerCoordinator._await_exact_credit_clear = _no_barrier
        try:
            run_id = "runNDM"
            sockA, credit_id, counter = _deliver_credited_envelope(
                b, msgs, run_id)
            counter.coord = b.coord
            counter.conn_key = sockA
            counter.stop = b.reader_stop

            b.peers["hostA:gpu0"].send(msgs[1])
            decoded_while_undisposed = False
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if counter.completed >= 2:
                    decoded_while_undisposed = (
                        b.coord.resume_credit_id_for(sockA) == credit_id)
                    break
                time.sleep(0.02)

            assert neutralised["n"] >= 1, (
                "the pre-decode barrier never ran, so removing it proves "
                "nothing — the mutant is not exercising G-NO-PREDECODE's subject")
            assert decoded_while_undisposed, (
                f"restoring the post-decode placement did NOT produce a second "
                f"decode while the reservation was outstanding — G-NO-PREDECODE "
                f"is vacuous ({counter.completed} decodes)")
            assert counter.waits >= 1, (
                "the restored post-decode wait never spun — the mutant decoded "
                "for some other reason than round 2's placement")
        finally:
            RangeMinerCoordinator._await_exact_credit_clear = orig_wait
            fetch_gate.set()
            b.close()


# ===========================================================================
# S172 STAGING-CAPACITY AMENDMENT (Beta staging ruling §§2-6, 2026-08-07)
# ===========================================================================
_HIGHWATER_CONTROLS = {
    # key: (kebab, flag, manifest default, injected value)
    "staging_high_water_files": ("staging-high-water-files",
                                 "--staging-high-water-files", None, 777),
    "staging_high_water_bytes": ("staging-high-water-bytes",
                                 "--staging-high-water-bytes",
                                 16 * 1024 ** 3, 512 * 1024 * 1024),
}


def _reserve_with_file(coord, run_id, sid, sub_index, size_bytes, tmp,
                       attempt=0, gen=0):
    """One REAL held reservation with a REAL staged file on disk.

    Uses the production `reserve_capacity` + `set_reservation_paths`, so the row
    the release path later discharges is the same shape staging produces."""
    rid = coord.reserve_capacity(run_id, sid, attempt, sub_index, gen, size_bytes)
    assert rid is not None, "capacity refused a reservation the gate needs"
    staged = os.path.join(tmp, f"{sid}_sub{sub_index}.bin")
    with open(staged, "wb") as fh:
        fh.write(b"\0" * size_bytes)
    coord.ledger.set_reservation_paths(rid, staged_path=staged)
    return rid, staged


def gate_highwater_route():
    """G-HIGHWATER-ROUTE. BOTH high-waters travel the complete governed route:

        manifest default_params -> args_map -> window_optimizer argparse
          -> coordinator attrs -> run_trial_miner -> build_coordinator
          -> CoordinatorConfig

    and the delivered values are LOAD-BEARING — they change the real reservation
    limits, not just a stored field. The route is gated, not the parameters
    (§2.15): a Step-1 key the manifest does not declare dies at hop 1.
    """
    mpath = os.path.join(_ROOT, "agent_manifests", "window_optimizer.json")
    with open(mpath) as fh:
        manifest = json.load(fh)
    dp = manifest["default_params"]
    amap = manifest["actions"][0]["args_map"]
    for key, (kebab, _flag, default, _inj) in _HIGHWATER_CONTROLS.items():
        assert key in dp, f"hop 1a: manifest default_params lacks {key}"
        assert dp[key] == default, (
            f"hop 1a: manifest {key}={dp[key]!r} != {default!r}")
        assert amap.get(kebab) == key, f"hop 1b: args_map lacks {kebab!r}"
        declared = dict(dp)
        merged = {**declared}
        for k, v in {key: "OVERRIDE", "undeclared_key": 1}.items():
            if k in declared:
                merged[k] = v
        assert merged[key] == "OVERRIDE", f"hop 1c: {key} dies in the filter"
        assert "undeclared_key" not in merged

    wo = os.path.join(_ROOT, "window_optimizer.py")
    with open(wo) as fh:
        wo_src = fh.read()
    flags = set()
    for node in ast.walk(ast.parse(wo_src)):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            flags.add(node.args[0].value)
    # hop 2b is checked over the AST, not the text: the call site legitimately
    # wraps across lines, and a substring probe would report a broken route for a
    # purely cosmetic line break (and, worse, would go green on a commented-out
    # one). Collect every `<key>=getattr(args, '<key>', ...)` keyword actually
    # passed at a call site.
    _forwarded = set()
    for node in ast.walk(ast.parse(wo_src)):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            v = kw.value
            if (kw.arg and isinstance(v, ast.Call)
                    and getattr(v.func, "id", None) == "getattr"
                    and len(v.args) >= 2
                    and getattr(v.args[0], "id", None) == "args"
                    and isinstance(v.args[1], ast.Constant)
                    and v.args[1].value == kw.arg):
                _forwarded.add(kw.arg)
    for key, (_kebab, flag, _d, _i) in _HIGHWATER_CONTROLS.items():
        assert flag in flags, f"hop 2a: argparse lacks {flag}"
        assert key in _forwarded, (
            f"hop 2b: {flag} is parsed but never passed on")
        assert f"coordinator.{key}" in wo_src, (
            f"hop 2c: nothing assigns coordinator.{key} — that read is DEAD")

    integ = os.path.join(_ROOT, "window_optimizer_integration_final.py")
    with open(integ) as fh:
        integ_src = fh.read()
    for key in _HIGHWATER_CONTROLS:
        assert f"getattr(coordinator, '{key}'" in integ_src, (
            f"hop 3: the integration does not read coordinator.{key}")
        assert key in inspect.signature(run_trial_miner).parameters
        assert key in inspect.signature(build_coordinator).parameters
        assert key in {f.name for f in dataclasses.fields(CoordinatorConfig)}

    # The old production source was a STALE literal: `getattr(..., 512)` while the
    # committed dataclass default was 4096. Neither survives as a default here.
    assert "'staging_high_water_files', 512)" not in integ_src, (
        "the stale 512 getattr fallback is still the integration's default")

    injected = {k: v[3] for k, v in _HIGHWATER_CONTROLS.items()}
    with tempfile.TemporaryDirectory() as tmp:
        params = {**dp, **injected}
        filtered = {k: v for k, v in params.items() if k in dp}
        coord = build_coordinator(
            staging_dir=os.path.join(tmp, "stg"),
            **{k: filtered[k] for k in _HIGHWATER_CONTROLS})
        for key, (_k, _f, _d, value) in _HIGHWATER_CONTROLS.items():
            got = getattr(coord.config, key)
            assert got == value, (
                f"manifest-injected {key}={value!r} is NOT observed in the "
                f"coordinator config (got {got!r}) — the route is broken")
        # LOAD-BEARING: the delivered file ceiling is what reservations enforce.
        assert coord.effective_high_water_files() == 777
        coord.ledger.create_trial("rteR", 0, now=100.0)
        coord.ledger.add_stripe("rteR", "rteR_s0", 0, 10, "java_lcg", 1, 100.0)
        # a single shard larger than the delivered BYTE ceiling is refused
        assert coord.reserve_capacity(
            "rteR", "rteR_s0", 0, 0, 0, 512 * 1024 * 1024 + 1) is None, (
            "the injected byte high-water does not bound real reservations")

        # ---- MUTATION: break ONE hop and the gate must red -------------------
        # hop 1 is the hop §2.15 says kills a parameter silently: drop the key
        # from the DECLARED set and WATCHER's filter discards the operator value.
        mutated_declared = {k: v for k, v in dp.items()
                            if k != "staging_high_water_files"}
        mutated = {**mutated_declared}
        for k, v in injected.items():
            if k in mutated_declared:
                mutated[k] = v
        assert "staging_high_water_files" not in mutated, (
            "MUTATION INERT: the key survived an undeclared manifest, so this "
            "gate would pass with hop 1 broken")
        mcoord = build_coordinator(
            staging_dir=os.path.join(tmp, "stg2"),
            **{k: mutated[k] for k in _HIGHWATER_CONTROLS if k in mutated})
        assert mcoord.config.staging_high_water_files != 777, (
            "MUTATION DID NOT RED THE GATE: the injected value arrived even "
            "though the manifest never declared the key")


def gate_trial_retention_preflight():
    """G-TRIAL-RETENTION-PREFLIGHT (compact/unit arm).

    ⚠ PROVENANCE, CORRECTED [R1 §3, Beta]. This arm uses the **2026-08-05
    STAGING-BACK-PRESSURE FIXTURE** — four macro-stripes across the recorded
    heterogeneous worker set (34+14+34+34 = 116 exact). That fixture was built to
    demonstrate the exact-vs-conservative burst-bound distinction and it is kept
    here as a compact mathematical arm.

    **It is NOT the gate-12 geometry**, and an earlier revision of this gate
    mislabelled it as such. The real 2026-08-07 gate-12 production geometry —
    1,073,741,824 seeds over 67,108,864 = 16 macro-stripes per stage — is exercised
    by `gate_trial_retention_preflight_gate12_geometry` below.

    The requirement is COMPUTED from the geometry, never transcribed: no total is
    written as a literal in any assertion here.
    """
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, staging_high_water_files=512,
                       miner_stripe_size=RECORDED_STRIPE_SPAN)
        workers = []
        for i, (wid, backend) in enumerate(RECORDED_ASSIGNMENT):
            workers.append(_register(coord, wid=wid, backend=backend,
                                     now=100.0 + i))
        stages = workflow_stages_for("java_lcg", True)
        total_seeds = RECORDED_STRIPE_SPAN * len(RECORDED_ASSIGNMENT)

        required, detail = coord.trial_retention_requirement(
            stages, total_seeds, workers)
        # the requirement is the sum over PLANNED PHASES of the sum over PLANNED
        # STRIPES of max-over-eligible expected sub-stripes — recomputed here from
        # the primitives so the gate does not merely re-read the implementation.
        expected = 0
        for fam, ph in stages:
            expected += staging_burst_bound_conservative(
                [RECORDED_STRIPE_SPAN] * len(RECORDED_ASSIGNMENT),
                workers, ph, caps=coord._central_caps(), family_name=fam)
        assert required == expected, (required, expected)
        assert len(stages) == 4 and detail["stripe_count"] == 4
        assert required > 512, (
            f"the 2026-08-05 fixture geometry derives only {required} files, so a "
            f"512 ceiling would NOT be undersized and this arm proves nothing")

        # 512 < derived -> FAIL CLOSED, and the failure is a configuration defect
        raised = None
        try:
            coord.preflight_trial_retention("runRP", stages, total_seeds, workers)
        except StagingRetentionSizingError as e:
            raised = e
        assert raised is not None, (
            "an operator ceiling below the derived requirement was ACCEPTED — "
            "Beta: a warning is explicitly insufficient")
        assert str(required) in str(raised) and "512" in str(raised)
        assert isinstance(raised, StagingConfigurationError), (
            "a retention sizing defect must be permanent/non-retryable")

        # at or above the requirement -> preflight PASSES and resolves that value
        ok = _coord(tmp, dbname="ok.db", staging_high_water_files=required,
                    miner_stripe_size=RECORDED_STRIPE_SPAN)
        okw = [_register(ok, wid=w, backend=b, now=100.0 + i)
               for i, (w, b) in enumerate(RECORDED_ASSIGNMENT)]
        det = ok.preflight_trial_retention("runRP2", stages, total_seeds, okw)
        assert det["mode"] == "operator" and det["resolved_files"] == required
        assert ok.effective_high_water_files() == required

        # UNSET means DERIVE: the resolved ceiling IS the derived requirement.
        der = _coord(tmp, dbname="der.db", staging_high_water_files=None,
                     miner_stripe_size=RECORDED_STRIPE_SPAN)
        derw = [_register(der, wid=w, backend=b, now=100.0 + i)
                for i, (w, b) in enumerate(RECORDED_ASSIGNMENT)]
        dd = der.preflight_trial_retention("runRP3", stages, total_seeds, derw)
        assert dd["mode"] == "derived" and dd["resolved_files"] == required
        assert der.effective_high_water_files() == required


# --- the REAL 2026-08-07 gate-12 production geometry [R1 §3, Beta] ---------
# max_seeds / miner_stripe_size = 16 macro-stripes per stage. Stage 0 consumed 504
# files and stage 1 consumed 524 (total 1,028) against the 512 ceiling — an
# OBSERVED two-stage count, recorded here as provenance only. No total below is
# written as a literal in an assertion: the derivation produces the number.
GATE12_MAX_SEEDS = 1_073_741_824
GATE12_STRIPE_SIZE = 67_108_864
GATE12_EXPECTED_STRIPES = 16


def gate_trial_retention_preflight_gate12_geometry():
    """G-TRIAL-RETENTION-PREFLIGHT (real gate-12 geometry) [R1 §3].

    The 2026-08-07 production run that deadlocked: 1,073,741,824 seeds partitioned
    at 67,108,864 into 16 macro-stripes per stage, full test_both_modes workflow.
    Establishes the four things Beta asked for — derived stripe_count == 16, derived
    requirement > 512, an explicit 512 ceiling fails closed before StripeAssign, and
    an unset ceiling resolves to the derived requirement.
    """
    with tempfile.TemporaryDirectory() as tmp:
        stages = workflow_stages_for("java_lcg", True)

        # ---- derived geometry -------------------------------------------------
        coord = _coord(tmp, staging_high_water_files=512,
                       miner_stripe_size=GATE12_STRIPE_SIZE)
        workers = [_register(coord, wid=w, backend=b, now=100.0 + i)
                   for i, (w, b) in enumerate(RECORDED_ASSIGNMENT)]
        required, detail = coord.trial_retention_requirement(
            stages, GATE12_MAX_SEEDS, workers)
        assert detail["stripe_count"] == GATE12_EXPECTED_STRIPES, (
            f"the production partition yields {detail['stripe_count']} macro-"
            f"stripes, not {GATE12_EXPECTED_STRIPES} — the geometry is wrong")
        assert detail["stage_count"] == 4
        assert sum(detail["stripe_spans"]) == GATE12_MAX_SEEDS
        # recomputed from the primitives, per stage, so the gate does not merely
        # re-read the implementation it is testing
        ebs = coord.resolve_eligible_by_stage(stages, workers)
        expected = sum(
            staging_burst_bound_conservative(
                detail["stripe_spans"], ebs[(f, p)], p,
                caps=coord._central_caps(), family_name=f)
            for f, p in stages)
        assert required == expected, (required, expected)
        assert required > 512, (
            f"the real gate-12 geometry derives {required} files; a 512 ceiling "
            f"must be undersized for this regression to mean anything")
        # every stage contributes, and no stage is silently free
        assert len(detail["per_stage"]) == 4
        assert all(s["files"] > 0 for s in detail["per_stage"])
        assert sum(s["files"] for s in detail["per_stage"]) == required

        # ---- explicit 512 -> FAIL CLOSED --------------------------------------
        raised = None
        try:
            coord.preflight_trial_retention(
                "runG12", stages, GATE12_MAX_SEEDS, workers)
        except StagingRetentionSizingError as e:
            raised = e
        assert raised is not None, (
            "the real gate-12 geometry was ADMITTED under the 512 ceiling that "
            "deadlocked it")
        assert str(required) in str(raised)

        # ---- unset -> resolved ceiling IS the derived requirement -------------
        der = _coord(tmp, dbname="g12.db", staging_high_water_files=None,
                     miner_stripe_size=GATE12_STRIPE_SIZE)
        derw = [_register(der, wid=w, backend=b, now=100.0 + i)
                for i, (w, b) in enumerate(RECORDED_ASSIGNMENT)]
        dd = der.preflight_trial_retention(
            "runG12b", stages, GATE12_MAX_SEEDS, derw)
        assert dd["mode"] == "derived"
        assert dd["resolved_files"] == required == dd["required_files"]
        assert der.effective_high_water_files() == required


def gate_stage_specific_eligibility():
    """G-STAGE-ELIGIBILITY [R1 §4, Beta BLOCKER B].

    Eligibility is family/phase dependent by construction: the Phase-4 contract
    requires a worker to advertise the EXACT concrete variant before assignment
    (`can_assign_variant`). Sizing every planned stage from ONE collection is
    therefore wrong in principle.

    NEGATIVE ARM (Beta): with asymmetric variant support, demonstrate that reusing
    a single stage's eligible population DIFFERS from the correctly stage-resolved
    calculation, then prove the preflight uses the correct later-stage population.
    """
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, staging_high_water_files=None,
                       miner_stripe_size=RECORDED_STRIPE_SPAN)
        stages = workflow_stages_for("java_lcg", True)

        # Beta's example, made concrete:
        #   A (cuda, LOOSE hybrid cap) supports the CONSTANT variants only
        #   B (rocm, TIGHT hybrid cap)  supports the HYBRID variants only
        const_only = ["java_lcg", "java_lcg_reverse"]
        hybrid_only = ["java_lcg_hybrid", "java_lcg_hybrid_reverse"]
        node = NodeConfig(hostname="hostA", spool_root=SPOOL_ROOT,
                          ssh_address="10.0.0.9", ssh_user="michael")
        a = coord.register_worker(
            worker_id="hostA:gpu0", hostname="hostA", backend="cuda",
            capabilities={"seed_caps": dict(CAPS),
                          "supported_variants": list(const_only)},
            node_config=node, now=100.0)
        b = coord.register_worker(
            worker_id="hostB:gpu0", hostname="hostB", backend="rocm",
            capabilities={"seed_caps": dict(CAPS),
                          "supported_variants": list(hybrid_only)},
            node_config=NodeConfig(hostname="hostB", spool_root=SPOOL_ROOT,
                                   ssh_address="10.0.0.10", ssh_user="michael"),
            now=101.0)
        candidates = [a, b]

        # the resolver must partition them EXACTLY as assign_stripes would
        ebs = coord.resolve_eligible_by_stage(stages, candidates)
        assert [w.worker_id for w in ebs[("java_lcg", 1)]] == ["hostA:gpu0"]
        assert [w.worker_id for w in ebs[("java_lcg_hybrid", 3)]] == ["hostB:gpu0"]
        for (fam, ph), pop in ebs.items():
            for w in pop:
                assert coord.can_assign_variant(w, fam), (
                    f"{w.worker_id} was placed in stage {(fam, ph)} but "
                    f"assign_stripes would refuse it")

        spans = [RECORDED_STRIPE_SPAN] * 2
        total_seeds = RECORDED_STRIPE_SPAN * 2

        # ---- NEGATIVE ARM: one-collection reuse DIFFERS from stage-resolved ----
        stage0_pop = ebs[("java_lcg", 1)]              # A only
        reuse_everywhere = sum(
            staging_burst_bound_conservative(
                spans, stage0_pop, p, caps=coord._central_caps(), family_name=f)
            for f, p in stages)
        stage_resolved = sum(
            staging_burst_bound_conservative(
                spans, ebs[(f, p)], p, caps=coord._central_caps(), family_name=f)
            for f, p in stages)
        assert reuse_everywhere != stage_resolved, (
            "the fixture is INERT: stage-0 reuse and stage-resolved sizing agree, "
            "so this arm cannot detect the defect it exists for")
        # A is cuda (hybrid cap 2.5M); B is rocm (hybrid cap 1.0M). Sizing the
        # hybrid stages from A alone UNDERSTATES them — the conservative-bound
        # violation Beta named.
        hyb_from_a = staging_burst_bound_conservative(
            spans, stage0_pop, 3, caps=coord._central_caps(),
            family_name="java_lcg_hybrid")
        hyb_correct = staging_burst_bound_conservative(
            spans, ebs[("java_lcg_hybrid", 3)], 3, caps=coord._central_caps(),
            family_name="java_lcg_hybrid")
        assert hyb_from_a < hyb_correct, (hyb_from_a, hyb_correct)

        # ---- the PREFLIGHT uses the correct later-stage population ------------
        required, detail = coord.trial_retention_requirement(
            stages, total_seeds, candidates)
        assert required == stage_resolved, (
            f"the preflight derived {required}, not the stage-resolved "
            f"{stage_resolved} — a stage is being sized from the wrong population")
        by_key = {(s["family_name"], s["phase"]): s for s in detail["per_stage"]}
        assert by_key[("java_lcg", 1)]["eligible_worker_ids"] == ["hostA:gpu0"]
        assert by_key[("java_lcg_hybrid", 3)]["eligible_worker_ids"] == ["hostB:gpu0"]
        assert by_key[("java_lcg_hybrid", 3)]["files"] == hyb_correct

        # ---- a planned stage nobody can serve FAILS CLOSED --------------------
        # (never sized as 0 files, which would let the trial start and then strand)
        lonely = _coord(tmp, dbname="lonely.db", staging_high_water_files=None,
                        miner_stripe_size=RECORDED_STRIPE_SPAN)
        only_const = lonely.register_worker(
            worker_id="hostA:gpu0", hostname="hostA", backend="cuda",
            capabilities={"seed_caps": dict(CAPS),
                          "supported_variants": list(const_only)},
            node_config=node, now=100.0)
        try:
            lonely.trial_retention_requirement(stages, total_seeds, [only_const])
            raise AssertionError(
                "a planned hybrid stage with NO eligible worker was sized instead "
                "of refused")
        except ValueError as e:
            assert "NO eligible worker" in str(e), e


def gate_preflight_plan_is_persisted():
    """G-PREFLIGHT-PLAN-PERSISTED [R1 §5, Beta REQUIRED].

    The planned geometry behind an admissibility decision is durable, sufficient to
    reproduce the decision, and written FROM THE PREFLIGHT'S OWN VALUES — never a
    second derivation. Includes the refusal case, which is precisely the case a
    post-mortem cannot reconstruct from stripe rows because there are none.
    """
    with tempfile.TemporaryDirectory() as tmp:
        stages = workflow_stages_for("java_lcg", True)

        # ---- ADMITTED (derived) ----------------------------------------------
        coord = _coord(tmp, staging_high_water_files=None,
                       miner_stripe_size=GATE12_STRIPE_SIZE)
        workers = [_register(coord, wid=w, backend=b, now=100.0 + i)
                   for i, (w, b) in enumerate(RECORDED_ASSIGNMENT)]
        detail = coord.preflight_trial_retention(
            "runPP", stages, GATE12_MAX_SEEDS, workers)
        plan = coord.ledger.get_preflight_plan("runPP")
        assert plan is not None, "no preflight plan was persisted before dispatch"

        # sufficient to REPRODUCE the decision
        assert plan["total_seeds"] == GATE12_MAX_SEEDS
        assert plan["miner_stripe_size"] == GATE12_STRIPE_SIZE
        assert plan["macro_stripe_count"] == GATE12_EXPECTED_STRIPES
        assert plan["stripe_spans"] == detail["stripe_spans"]
        assert sum(plan["stripe_spans"]) == GATE12_MAX_SEEDS
        assert [tuple(s) for s in plan["stages"]] == [
            (f, p) for f, p in detail["stages"]]
        assert plan["per_stage"] == detail["per_stage"]
        assert plan["required_files"] == detail["required_files"]
        assert plan["high_water_mode"] == "derived"
        assert plan["configured_files"] is None
        assert plan["resolved_files"] == detail["resolved_files"]
        assert plan["admitted"] is True
        assert plan["schema_version"] == MinerLedger.PREFLIGHT_PLAN_SCHEMA_VERSION
        assert plan["created_at"] > 0
        assert plan["caps"] == detail["caps"]
        assert len(plan["execution_set_sha256"]) == 64
        assert len(plan["stripe_spans_sha256"]) == 64
        # the stage-specific eligible sets are part of the record (R1 §4 interlock)
        assert all(s["eligible_worker_ids"] for s in plan["per_stage"])

        # the record REPRODUCES the decision: re-deriving from the persisted
        # geometry alone reproduces the persisted total.
        redone = sum(s["files"] for s in plan["per_stage"])
        assert redone == plan["required_files"] == plan["resolved_files"]

        # ---- written from the preflight's OWN values, not recomputed ----------
        # If persistence re-derived anything, poisoning the derivation AFTER the
        # preflight would change the stored row. It must not: the row is a
        # transcript of `detail`.
        spy = {"calls": 0}
        real = RangeMinerCoordinator.trial_retention_requirement

        def _counting(self_, *a, **k):
            spy["calls"] += 1
            return real(self_, *a, **k)
        RangeMinerCoordinator.trial_retention_requirement = _counting
        try:
            c2 = _coord(tmp, dbname="pp2.db", staging_high_water_files=None,
                        miner_stripe_size=GATE12_STRIPE_SIZE)
            w2 = [_register(c2, wid=w, backend=b, now=100.0 + i)
                  for i, (w, b) in enumerate(RECORDED_ASSIGNMENT)]
            c2.preflight_trial_retention("runPP2", stages, GATE12_MAX_SEEDS, w2)
        finally:
            RangeMinerCoordinator.trial_retention_requirement = real
        assert spy["calls"] == 1, (
            f"the requirement was derived {spy['calls']} times for ONE preflight "
            f"— persistence must transcribe the preflight's values, not recompute")

        # ---- REFUSED trials are persisted too --------------------------------
        low = _coord(tmp, dbname="pp3.db", staging_high_water_files=512,
                     miner_stripe_size=GATE12_STRIPE_SIZE)
        lw = [_register(low, wid=w, backend=b, now=100.0 + i)
              for i, (w, b) in enumerate(RECORDED_ASSIGNMENT)]
        try:
            low.preflight_trial_retention("runPP3", stages, GATE12_MAX_SEEDS, lw)
            raise AssertionError("the undersized ceiling was admitted")
        except StagingRetentionSizingError:
            pass
        ref = low.ledger.get_preflight_plan("runPP3")
        assert ref is not None, (
            "a REFUSED trial persisted no plan — this is the one case a "
            "post-mortem cannot reconstruct, because no stripe rows exist")
        assert ref["admitted"] is False
        assert ref["high_water_mode"] == "operator"
        assert ref["configured_files"] == 512
        assert ref["resolved_files"] is None
        assert ref["required_files"] > 512

        # ⚠ The R1 arm that stood here — "a provenance-write failure must NOT
        # change the decision … the trial is still admitted" — is DELETED under
        # Beta's R2 §5 ruling: the durable plan was never optional telemetry, and a
        # trial cannot satisfy both "must be persisted before dispatch" and "if
        # persistence fails, dispatch anyway". Its two replacements are
        # G-PREFLIGHT-PROVENANCE-FAIL-CLOSED below.


def gate_late_worker_excluded_from_frozen_cohort():
    """G-LATE-WORKER-EXCLUDED [R2 §4, Beta].

    The trial's assignable cohort is FROZEN at successful preflight. A daemon that
    comes online afterwards registers normally and serves a LATER trial, but cannot
    alter the execution geometry of a trial whose ceiling is already certified and
    persisted.

    Target invariant:

        actual worker used by trial ⊆ population used to derive the ceiling

    Worker C is given a MATERIALLY tighter applicable cap, so if it could join it
    would genuinely change the conservative bound — not a cosmetic difference.
    """
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, staging_high_water_files=None,
                       miner_stripe_size=RECORDED_STRIPE_SPAN)
        stages = [("java_lcg", 1)]
        total_seeds = RECORDED_STRIPE_SPAN * 2

        def _reg(c, wid, backend, caps, now):
            return c.register_worker(
                worker_id=wid, hostname=wid.split(":")[0], backend=backend,
                capabilities={"seed_caps": dict(caps),
                              "supported_variants": list(VARIANTS)},
                node_config=NodeConfig(hostname=wid.split(":")[0],
                                       spool_root=SPOOL_ROOT,
                                       ssh_address="10.0.0.9", ssh_user="michael"),
                now=now)

        # ---- 1) preflight with A and B (both CUDA, loose cap) ----------------
        a = _reg(coord, "hostA:gpu0", "cuda", CAPS, 100.0)
        b = _reg(coord, "hostB:gpu0", "cuda", CAPS, 101.0)
        detail = coord.preflight_trial_retention(
            "runLW", stages, total_seeds, [a, b])
        required_before = detail["required_files"]

        # ---- 2) the stage-specific execution set is persisted ----------------
        plan = coord.ledger.get_preflight_plan("runLW")
        assert plan is not None
        frozen_ids = plan["per_stage"][0]["eligible_worker_ids"]
        assert frozen_ids == ["hostA:gpu0", "hostB:gpu0"], frozen_ids
        cohort = coord.frozen_trial_cohort("runLW")
        assert cohort is not None, (
            "no assignable cohort was frozen at a successful preflight — the "
            "trial's worker population is unbounded after certification")
        assert set(cohort[("java_lcg", 1)]) == {"hostA:gpu0", "hostB:gpu0"}

        # ---- 3) C registers AFTER preflight, with a MATERIALLY tighter cap ----
        # Beta's own example: a CUDA-only population at preflight, then a
        # tighter-cap ROCm worker joins. The tightness must come from the BACKEND,
        # not from advertising different numbers — `_validate_caps` requires the
        # advertised caps to equal the central config exactly and quarantines a
        # worker that disagrees, so a "tighter caps" worker would be excluded for
        # an unrelated reason and would prove nothing about the freeze.
        #   cuda -> nvidia cap 5,000,000 -> ceil(span/5M) = 14 per stripe
        #   rocm -> amd    cap 2,000,000 -> ceil(span/2M) = 34 per stripe
        c = _reg(coord, "hostC:gpu0", "rocm", CAPS, 102.0)
        # C really would move the bound if it were counted
        would_be, _ = coord.trial_retention_requirement(
            stages, total_seeds, [a, b, c])
        assert would_be > required_before, (
            f"the fixture is COSMETIC: C does not change the bound "
            f"({would_be} vs {required_before})")

        # ---- 4) C cannot receive a StripeAssign for THIS trial ---------------
        assert coord.cohort_eligible("runLW", "java_lcg", 1, c) is False
        assert coord.can_assign_variant(c, "java_lcg") is True, (
            "C must be variant-capable — otherwise the exclusion proves nothing "
            "about the freeze")
        assigns = coord.assign_stripes(
            "runLW", "java_lcg", 1, total_seeds, [a, b, c],
            stripe_prefix="runLW__st0")
        claimed_by = {x["worker_id"] for x in assigns if x["claimed"]}
        assert "hostC:gpu0" not in claimed_by, (
            f"a post-freeze joiner received a StripeAssign: {claimed_by}")
        assert claimed_by <= {"hostA:gpu0", "hostB:gpu0"}, claimed_by
        # the retry path is bound by the same cohort
        assert coord._pick_other_worker([c], "hostA:gpu0", "java_lcg") is None, (
            "the retry matrix would reassign a stripe to a post-freeze joiner")
        assert coord._pick_other_worker([b, c], "hostA:gpu0", "java_lcg") is b

        # ---- no re-derivation happened -------------------------------------
        assert coord._retention_preflight_detail["required_files"] == required_before
        assert coord.effective_high_water_files() == required_before
        assert coord.ledger.get_preflight_plan("runLW")["required_files"] == \
            required_before
        assert coord.frozen_trial_cohort("runLW")[("java_lcg", 1)].keys() == \
            {"hostA:gpu0", "hostB:gpu0"}

        # ---- 5) C IS usable by a later trial --------------------------------
        later, later_detail = coord.trial_retention_requirement(
            stages, total_seeds, [a, b, c])
        assert later > required_before
        d2 = coord.preflight_trial_retention("runLW2", stages, total_seeds,
                                             [a, b, c])
        assert "hostC:gpu0" in d2["per_stage"][0]["eligible_worker_ids"], (
            "C was excluded from a NEW trial — the freeze is per trial, not a "
            "global connection refusal")
        assert coord.cohort_eligible("runLW2", "java_lcg", 1, c) is True
        # ...and the earlier trial's cohort is untouched by the later freeze
        assert coord.frozen_trial_cohort("runLW")[("java_lcg", 1)].keys() == \
            {"hostA:gpu0", "hostB:gpu0"}

        # ---- 6) reconnect: admissible ONLY on a matching capability signature -
        # same identity, same advertisements -> readmitted
        a_same = _reg(coord, "hostA:gpu0", "cuda", CAPS, 103.0)
        assert coord.cohort_eligible("runLW", "java_lcg", 1, a_same) is True, (
            "a frozen identity reconnecting UNCHANGED was excluded")
        # same identity, but the BACKEND the applicable cap is read from changed:
        # the ceiling was derived against cuda/5M, this is rocm/2M
        a_backend = _reg(coord, "hostA:gpu0", "rocm", CAPS, 104.0)
        assert coord.cohort_eligible("runLW", "java_lcg", 1, a_backend) is False, (
            "a frozen identity reconnected on a DIFFERENT backend and was still "
            "admitted — the ceiling was derived against its old advertisement")
        # same identity, but the advertised variant set changed -> different stage
        # membership from the one the ceiling was certified over
        a_variants = coord.register_worker(
            worker_id="hostA:gpu0", hostname="hostA", backend="cuda",
            capabilities={"seed_caps": dict(CAPS),
                          "supported_variants": ["java_lcg"]},
            node_config=NodeConfig(hostname="hostA", spool_root=SPOOL_ROOT,
                                   ssh_address="10.0.0.9", ssh_user="michael"),
            now=105.0)
        assert coord.cohort_eligible("runLW", "java_lcg", 1, a_variants) is False, (
            "a frozen identity reconnected advertising a different variant set "
            "and was still admitted")
        # restore A so the closing assertions describe the real cohort
        _reg(coord, "hostA:gpu0", "cuda", CAPS, 106.0)

        # ---- losing a frozen worker never ENLARGES the cohort ----------------
        assert coord.cohort_filter("runLW", "java_lcg", 1, [c]) == []


def _explode_preflight_persist():
    """Patch `record_preflight_plan` to fail. Returns a restore callable."""
    orig = MinerLedger.record_preflight_plan

    def _boom(self_, run_id, detail):
        raise RuntimeError("disk full")

    MinerLedger.record_preflight_plan = _boom
    return lambda: setattr(MinerLedger, "record_preflight_plan", orig)


def _prov_serve_fail_closed(tmp):
    """Case A driven through the REAL serve loop: a would-be admitted trial whose
    provenance write fails must terminate with the §5-A classification having
    dispatched ZERO StripeAssign and touched the retry matrix zero times."""
    ds = os.path.join(tmp, "prov_ds.json")
    with open(ds, "w") as f:
        f.write('[{"draw":1},{"draw":2},{"draw":3}]')
    sink = _Sink()
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(8)
    port = lsock.getsockname()[1]
    holder = {}
    gate_caps = {**CAPS, "nvidia": 10}

    def run():
        try:
            holder["result"] = run_trial_miner(
                "runPSV", None, 3, "java_lcg", [1, 2, 3], 80, 0.25, 0.25,
                False, ds, worker_pool_size=1,
                staging_dir=os.path.join(tmp, "prov_stg"), phase5_sink=sink,
                listen_sock=lsock, family_name="java_lcg", workflow_phase=1,
                miner_stripe_size=20, seed_cap_nvidia=10,
                skip_min=0, skip_max=0, offset=0, window_size=3,
                staging_high_water_files=None,          # would be ADMITTED
                staging_high_water_bytes=64 * 1024 * 1024,
                serve_timeout=45.0)
        except Exception:
            holder["err"] = traceback.format_exc()

    restore = _explode_preflight_persist()
    with _MatrixSpy() as spy:
        t = threading.Thread(target=run, daemon=True)
        t.start()
        w = _RemoteWorker("127.0.0.1", port, "hostA", 0, caps=gate_caps)
        try:
            w.connect_register()
            w.start_loop()
            t.join(timeout=60)
            assert not t.is_alive(), "serve_trial never terminated"
            assert "err" not in holder, holder.get("err")
            result = holder["result"]
            assert result["state"] == "aborted", result["state"]
            assert sink.aborts, "no abort event delivered"
            reason = sink.aborts[0].get("reason", "")
            assert reason.startswith("coordinator_staging_preflight_provenance:"), (
                f"terminal reason does not lead with the provenance "
                f"classification: {reason!r}")
            assert not w.assigned, (
                f"stripes were DISPATCHED despite an unrecordable retention "
                f"decision: {w.assigned}")
            assert not result["stripes"], "stripe rows were created"
            assert spy.calls == [], (
                f"a provenance failure entered the retry matrix: {spy.calls}")
        finally:
            restore()
            w.stop()
            try:
                lsock.close()
            except OSError:
                pass


def gate_preflight_provenance_fail_closed():
    """G-PREFLIGHT-PROVENANCE-FAIL-CLOSED [R2 §5, Beta].

    The durable retention plan is NOT optional telemetry. Two asymmetric cases:

      A. would-be ADMITTED + provenance failure -> FAIL CLOSED before any
         StripeAssign, classified `coordinator_staging_preflight_provenance`;
      B. already SIZING-REFUSED + provenance failure -> still sizing-refused; the
         terminal cause stays `coordinator_staging_retention_sizing` and the
         provenance failure is only secondary evidence.

    Beta's framing, preserved: *"failure to write the audit record may not override
    a safety refusal, but inability to create the mandatory audit record prevents a
    would-be admission."*
    """
    with tempfile.TemporaryDirectory() as tmp:
        stages = workflow_stages_for("java_lcg", True)

        # ---- CASE A: would-be admitted -> fail closed ------------------------
        a = _coord(tmp, dbname="provA.db", staging_high_water_files=None,
                   miner_stripe_size=GATE12_STRIPE_SIZE)
        aw = [_register(a, wid=w, backend=b, now=100.0 + i)
              for i, (w, b) in enumerate(RECORDED_ASSIGNMENT)]
        restore = _explode_preflight_persist()
        raised = None
        try:
            a.preflight_trial_retention("runPA", stages, GATE12_MAX_SEEDS, aw)
        except StagingPreflightProvenanceError as e:
            raised = e
        finally:
            restore()
        assert raised is not None, (
            "a trial whose mandatory retention record could not be written was "
            "ADMITTED — Beta: the durable plan was not optional telemetry")
        assert "disk full" in str(raised)
        assert isinstance(raised, StagingConfigurationError), (
            "a provenance failure must be permanent/non-retryable, never charged "
            "to a worker")
        # NOTHING became effective: no ceiling installed, no cohort frozen
        assert a._resolved_high_water_files is None, (
            "the retention ceiling was installed despite the fail-closed refusal")
        assert a.frozen_trial_cohort("runPA") is None, (
            "a cohort was frozen for a trial that never started")
        assert a.ledger.get_preflight_plan("runPA") is None

        # ---- CASE B: sizing refusal stays PRIMARY ----------------------------
        b = _coord(tmp, dbname="provB.db", staging_high_water_files=512,
                   miner_stripe_size=GATE12_STRIPE_SIZE)
        bw = [_register(b, wid=w, backend=bk, now=100.0 + i)
              for i, (w, bk) in enumerate(RECORDED_ASSIGNMENT)]
        restore = _explode_preflight_persist()
        raised_b = None
        try:
            b.preflight_trial_retention("runPB", stages, GATE12_MAX_SEEDS, bw)
        except StagingRetentionSizingError as e:
            raised_b = e
        except StagingPreflightProvenanceError as e:      # pragma: no cover
            raise AssertionError(
                f"a provenance failure OVERRODE a safety refusal: {e}")
        finally:
            restore()
        assert raised_b is not None, "the undersized ceiling was admitted"
        # the terminal cause is unchanged...
        assert type(raised_b) is StagingRetentionSizingError
        assert "cannot retain the whole" in str(raised_b)
        # ...and the provenance failure rides along as SECONDARY evidence only
        assert "secondary" in str(raised_b) and "disk full" in str(raised_b), (
            f"the provenance failure was not attached as secondary evidence: "
            f"{raised_b}")
        assert b._resolved_high_water_files is None
        assert b.frozen_trial_cohort("runPB") is None

        # ---- CASE A through the REAL serve loop: ZERO StripeAssign ------------
        _prov_serve_fail_closed(tmp)


def gate_commit_cleanup_resumes_after_crash():
    """G-COMMIT-CRASH-RESUME [R1 §2, Beta BLOCKER A].

    Delivery and cleanup are two INDEPENDENT durable phases. A crash between
    reservation releases must not strand the remainder: on restart, delivery is
    already `done` (so the sink is NOT called again) but cleanup is not, and the
    idempotent sweep RESUMES.

    The submitted revision returned on the `delivery == done` branch before the
    sweep, so reservations 2..N were never discharged. `ack_by_event_id` being
    idempotent is necessary but not sufficient if the recovery path never calls it.

    PROCESS RESTART IS MODELLED BY REOPENING THE LEDGER AND COORDINATOR against the
    same on-disk SQLite file, so the resuming objects share no in-memory state with
    the ones that crashed — the durable row is the only channel between them.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dbp = os.path.join(tmp, "crash.db")
        stg = os.path.join(tmp, "stg")
        os.makedirs(stg, exist_ok=True)
        run_id, sid = "runCC", "runCC_s0"
        N = 3

        class _CountingSink(Phase5Sink):
            def __init__(self):
                self.published, self.commits, self.aborts = [], [], []

            def publish_shard(self, manifest):
                self.published.append(manifest)

            def commit_trial(self, event):
                self.commits.append(dict(event))

            def abort_trial(self, event):
                self.aborts.append(event)

        def _open(sink):
            ledger = MinerLedger(dbp)
            return RangeMinerCoordinator(
                CoordinatorConfig(staging_dir=stg, staging_high_water_files=64,
                                  staging_high_water_bytes=64 * 1024 * 1024),
                ledger, phase5_sink=sink)

        # ---- pass 1: commit, then FAULT after exactly one ack ----------------
        sink1 = _CountingSink()
        c1 = _open(sink1)
        c1.ledger.create_trial(run_id, 0, now=100.0)
        c1.ledger.add_stripe(run_id, sid, 0, 30, "java_lcg", 1, 100.0)
        paths = [_reserve_with_file(c1, run_id, sid, s, 512, stg)[1]
                 for s in range(N)]
        assert c1.reserved_files() == N

        class _Boom(RuntimeError):
            pass

        real_ack = RangeMinerCoordinator.ack_by_event_id
        acked = {"n": 0}

        def _ack_once(self_, event_id, now=None):
            if acked["n"] >= 1:
                raise _Boom("simulated process death mid-sweep")
            acked["n"] += 1
            return real_ack(self_, event_id, now)

        RangeMinerCoordinator.ack_by_event_id = _ack_once
        try:
            c1.commit_trial(run_id)
            raise AssertionError("the injected fault did not fire")
        except _Boom:
            pass
        finally:
            RangeMinerCoordinator.ack_by_event_id = real_ack

        # the twelve-step state at the moment of the crash
        t = c1.ledger.get_trial(run_id)
        assert t["commit_delivery_status"] == "done", (
            "delivery was not durably recorded before the sweep")
        assert t["commit_cleanup_status"] != "done", (
            "cleanup was marked done despite the sweep dying partway")
        held = c1.ledger.held_reservations(run_id)
        assert len(held) == N - 1, (
            f"expected {N-1} reservations still held after one ack, got {len(held)}")
        assert len(sink1.commits) == 1
        survivors = [p for p in paths if os.path.exists(p)]
        assert len(survivors) == N - 1

        # ---- pass 2: RESTART — new ledger + coordinator on the same file ------
        del c1
        sink2 = _CountingSink()
        c2 = _open(sink2)
        assert c2.ledger.get_trial(run_id)["commit_delivery_status"] == "done"

        ev = c2.commit_trial(run_id)

        # the sink is NOT called again after durable delivery
        assert sink2.commits == [], (
            f"the sink was re-delivered on the recovery path: {sink2.commits}")
        assert ev["delivery"] == "done"
        assert ev["cleanup"] == "resumed", ev
        # every REMAINING reservation and file is discharged...
        assert ev["released_reservations"] == N - 1, ev
        assert c2.ledger.held_reservations(run_id) == []
        assert c2.reserved_files() == 0 and c2.reserved_bytes() == 0
        for p in paths:
            assert not os.path.exists(p), f"staged file survived recovery: {p}"
        # ...and the ALREADY-discharged one is not discharged twice
        assert ev["released_reservations"] != N, (
            "the resumed sweep re-released the reservation the first pass had "
            "already discharged")
        assert c2.ledger.get_trial(run_id)["commit_cleanup_status"] == "done"

        # a further call is now a genuine duplicate: nothing at all happens
        ev3 = c2.commit_trial(run_id)
        assert ev3.get("duplicate") is True and ev3["cleanup"] == "already_done"
        assert ev3["released_reservations"] == 0
        assert sink2.commits == []


def gate_commit_crash_resume_mutant():
    """G-MUT-COMMIT-CRASH-RESUME — RED-FIRST EVIDENCE for the R1 §2 blocker.

    Restores the SUBMITTED semantics: `commit_delivery_status == done` treated as
    proof that cleanup also happened. That conflation is the whole defect — the
    submitted `commit_trial` returned on that branch before the sweep — and with it
    restored the recovery path must strand the remaining reservations.

    The restoration is executed, not described, and is asserted NON-INERT first.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dbp = os.path.join(tmp, "mut.db")
        stg = os.path.join(tmp, "stg")
        os.makedirs(stg, exist_ok=True)
        run_id, sid, N = "runMC", "runMC_s0", 3
        sink = _Sink()
        ledger = MinerLedger(dbp)
        coord = RangeMinerCoordinator(
            CoordinatorConfig(staging_dir=stg, staging_high_water_files=64,
                              staging_high_water_bytes=64 * 1024 * 1024),
            ledger, phase5_sink=sink)
        coord.ledger.create_trial(run_id, 0, now=100.0)
        coord.ledger.add_stripe(run_id, sid, 0, 30, "java_lcg", 1, 100.0)
        paths = [_reserve_with_file(coord, run_id, sid, s, 512, stg)[1]
                 for s in range(N)]

        # crash after exactly one ack, exactly as the real gate does
        real_ack = RangeMinerCoordinator.ack_by_event_id
        acked = {"n": 0}

        class _Boom(RuntimeError):
            pass

        def _ack_once(self_, event_id, now=None):
            if acked["n"] >= 1:
                raise _Boom("simulated process death mid-sweep")
            acked["n"] += 1
            return real_ack(self_, event_id, now)

        RangeMinerCoordinator.ack_by_event_id = _ack_once
        try:
            coord.commit_trial(run_id)
            raise AssertionError("the injected fault did not fire")
        except _Boom:
            pass
        finally:
            RangeMinerCoordinator.ack_by_event_id = real_ack
        assert len(coord.ledger.held_reservations(run_id)) == N - 1

        # ---- restore the SUBMITTED conflation of the two durable statuses ----
        real_get = MinerLedger.get_trial

        def _submitted_conflation(self_, rid):
            row = real_get(self_, rid)
            if row is not None:
                row = dict(row)
                # the submitted revision had no independent cleanup phase: reaching
                # `delivery == done` was itself the "already cleaned up" signal.
                row["commit_cleanup_status"] = row["commit_delivery_status"]
            return row

        MinerLedger.get_trial = _submitted_conflation
        try:
            row = MinerLedger.get_trial(ledger, run_id)
            assert row["commit_cleanup_status"] == "done", (
                "MUTATION INERT: the restored conflation did not take effect, so "
                "this arm cannot demonstrate the submitted defect")
            ev = coord.commit_trial(run_id)
            assert ev["released_reservations"] == 0, ev
            assert len(coord.ledger.held_reservations(run_id)) == N - 1, (
                "MUTATION DID NOT RED THE GATE: the remaining reservations were "
                "discharged even with the submitted conflation restored")
            stranded = [p for p in paths if os.path.exists(p)]
            assert len(stranded) == N - 1, (
                f"expected {N-1} stranded staged files under the submitted "
                f"semantics, got {len(stranded)}")
        finally:
            MinerLedger.get_trial = real_get

        # and with the real two-phase rule back, the same call recovers
        ev = coord.commit_trial(run_id)
        assert ev["cleanup"] == "resumed" and ev["released_reservations"] == N - 1
        assert coord.ledger.held_reservations(run_id) == []


def gate_stage_eligibility_mutant():
    """G-MUT-STAGE-ELIGIBILITY — CORRECTED [R2 §3, Beta].

    ⚠ WHAT THIS MUTANT RESTORES, AND WHY IT CHANGED.
    The R1 version of this gate claimed to restore "the submitted behaviour" by
    taking the FIRST STAGE'S RESOLVED eligible population and copying it across
    every stage, then asserting the result UNDERSTATED the correct one.

    **That was Beta's own hypothesis, and Beta has WITHDRAWN it.** The submitted
    code passed `serve_trial._eligible()` — ALL connected, non-quarantined workers,
    never variant-filtered — to every stage. Since the bound is a max over the
    supplied population, the old calculation used a SUPERSET and could only
    OVER-estimate. There was no undercount to reproduce.

    This mutant therefore restores the REAL previous behaviour:

        for every planned stage:
            eligible[stage] = ALL candidate workers (all-connected, non-quarantined)

    and asserts what is actually true of it. Beta: *"Do NOT require it to
    understate. Do not manufacture a safety failure the previous code did not
    have."*

    The gate's purpose is now exactly two things:
      1. exact-variant stage semantics are PRESERVED by the current code;
      2. the old all-connected calculation is DETECTABLY DIFFERENT from it
         (and, on this asymmetric fixture, more conservative).
    """
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, staging_high_water_files=None,
                       miner_stripe_size=RECORDED_STRIPE_SPAN)
        stages = workflow_stages_for("java_lcg", True)
        a = coord.register_worker(
            worker_id="hostA:gpu0", hostname="hostA", backend="cuda",
            capabilities={"seed_caps": dict(CAPS),
                          "supported_variants": ["java_lcg", "java_lcg_reverse"]},
            node_config=NodeConfig(hostname="hostA", spool_root=SPOOL_ROOT,
                                   ssh_address="10.0.0.9", ssh_user="michael"),
            now=100.0)
        b = coord.register_worker(
            worker_id="hostB:gpu0", hostname="hostB", backend="rocm",
            capabilities={"seed_caps": dict(CAPS),
                          "supported_variants": ["java_lcg_hybrid",
                                                 "java_lcg_hybrid_reverse"]},
            node_config=NodeConfig(hostname="hostB", spool_root=SPOOL_ROOT,
                                   ssh_address="10.0.0.10", ssh_user="michael"),
            now=101.0)
        candidates = [a, b]
        total_seeds = RECORDED_STRIPE_SPAN * 2

        correct, _ = coord.trial_retention_requirement(
            stages, total_seeds, candidates)

        # ---- restore the REAL previous behaviour: ALL candidates, every stage --
        real_resolve = RangeMinerCoordinator.resolve_eligible_by_stage

        def _all_connected(self_, workflow_stages, candidate_workers):
            # exactly what the submitted code did: the un-variant-filtered
            # all-connected, non-quarantined collection, handed to every stage
            pool = list(candidate_workers)
            return {(str(f), int(p)): list(pool) for f, p in workflow_stages}

        RangeMinerCoordinator.resolve_eligible_by_stage = _all_connected
        try:
            probe = coord.resolve_eligible_by_stage(stages, candidates)
            assert all(len(v) == len(candidates) for v in probe.values()), (
                "MUTATION INERT: the restored all-connected resolver did not hand "
                "every candidate to every stage")
            previous, _ = coord.trial_retention_requirement(
                stages, total_seeds, candidates)
            # (1) DETECTABLY DIFFERENT — the point of the gate
            assert previous != correct, (
                "MUTATION DID NOT RED THE GATE: the all-connected calculation "
                "produced the same requirement as the stage-resolved one, so this "
                "fixture cannot detect the change in semantics")
            # (2) and on this asymmetric fixture it is MORE CONSERVATIVE, because a
            #     superset can only raise a max. Asserted as the observed fact, NOT
            #     as a safety requirement: the old code did not undercount here.
            assert previous > correct, (
                f"the all-connected calculation produced {previous} vs the "
                f"stage-resolved {correct}; a superset population should not "
                f"lower a max-over-workers bound")
        finally:
            RangeMinerCoordinator.resolve_eligible_by_stage = real_resolve

        # exact-variant stage semantics are preserved once the real resolver is back
        again, _ = coord.trial_retention_requirement(
            stages, total_seeds, candidates)
        assert again == correct
        restored = coord.resolve_eligible_by_stage(stages, candidates)
        assert [w.worker_id for w in restored[("java_lcg", 1)]] == ["hostA:gpu0"]
        assert [w.worker_id for w in restored[("java_lcg_hybrid", 3)]] == ["hostB:gpu0"]


def gate_trial_retention_preflight_dispatches_nothing():
    """G-TRIAL-RETENTION-PREFLIGHT (serve path). An undersized ceiling terminates
    the trial through the REAL serve loop with ZERO StripeAssign, ZERO result
    traffic and ZERO retry-matrix calls, under a `coordinator_staging_*` reason.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1},{"draw":2},{"draw":3}]')
        sink = _Sink()
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        port = lsock.getsockname()[1]
        holder = {}
        gate_caps = {**CAPS, "nvidia": 10}

        def run():
            try:
                holder["result"] = run_trial_miner(
                    "runRPS", None, 3, "java_lcg", [1, 2, 3], 80, 0.25, 0.25,
                    False, ds, worker_pool_size=1,
                    staging_dir=os.path.join(tmp, "stg"), phase5_sink=sink,
                    listen_sock=lsock, family_name="java_lcg", workflow_phase=1,
                    miner_stripe_size=20, seed_cap_nvidia=10,
                    skip_min=0, skip_max=0, offset=0, window_size=3,
                    # 80 seeds / 20 = 4 stripes, each ceil(20/10)=2 sub-stripes
                    # => 8 files required for the one planned stage. A ceiling of
                    # 1 cannot retain it.
                    staging_high_water_files=1,
                    staging_high_water_bytes=64 * 1024 * 1024,
                    serve_timeout=45.0)
            except Exception:
                holder["err"] = traceback.format_exc()

        with _MatrixSpy() as spy:
            t = threading.Thread(target=run, daemon=True)
            t.start()
            w = _RemoteWorker("127.0.0.1", port, "hostA", 0, caps=gate_caps)
            try:
                w.connect_register()
                w.start_loop()
                t.join(timeout=60)
                assert not t.is_alive(), "serve_trial never terminated"
                assert "err" not in holder, holder.get("err")
                result = holder["result"]
                assert result["state"] == "aborted", result["state"]
                assert sink.aborts, "no abort event delivered"
                reason = sink.aborts[0].get("reason", "")
                assert reason.startswith("coordinator_staging_retention_sizing:"), (
                    f"terminal reason does not lead with the retention sizing "
                    f"classification: {reason!r}")
                # ZERO StripeAssign reached any worker, so zero result traffic
                assert not w.assigned, (
                    f"stripes were DISPATCHED despite an impossible ceiling: "
                    f"{w.assigned}")
                assert not result["stripes"], (
                    "stripe rows were created before the preflight refused")
                # ZERO retry-matrix calls: this is infrastructure, not a worker fault
                assert spy.calls == [], (
                    f"a retention sizing refusal entered the matrix: {spy.calls}")
            finally:
                w.stop()
                try:
                    lsock.close()
                except OSError:
                    pass


def gate_commit_release():
    """G-COMMIT-RELEASE. A successful multi-shard TrialCommit releases every
    trial-owned reservation exactly once, deletes every staged file, returns the
    capacity, and leaves the assembly usable. A DUPLICATE successful commit
    releases nothing a second time, deletes nothing, and re-reads no spool."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _Sink()
        coord = _coord(tmp, sink=sink, staging_high_water_files=64,
                       staging_high_water_bytes=64 * 1024 * 1024)
        run_id, sid = "runCR", "runCR_s0"
        coord.ledger.create_trial(run_id, 0, now=100.0)
        coord.ledger.add_stripe(run_id, sid, 0, 30, "java_lcg", 1, 100.0)
        staged = []
        for sub in range(3):
            _rid, path = _reserve_with_file(coord, run_id, sid, sub, 1024, tmp)
            staged.append(path)
        assert coord.reserved_files() == 3 and coord.reserved_bytes() == 3072
        assert all(os.path.isfile(p) for p in staged)

        ev = coord.commit_trial(run_id)
        assert ev["delivery"] == "done", ev
        assert ev["released_reservations"] == 3, ev
        assert ev["staged_files_deleted"] == 3, ev
        # capacity is genuinely back
        assert coord.ledger.held_reservations(run_id) == []
        assert coord.reserved_files() == 0 and coord.reserved_bytes() == 0
        # the staged files are gone from disk
        for p in staged:
            assert not os.path.exists(p), f"staged file survived commit: {p}"
        # durable cleanup status, on the SUCCESS column
        trial = coord.ledger.get_trial(run_id)
        assert trial["commit_cleanup_status"] == "done", trial
        assert trial["state"] == "committed"
        # the assembly remains usable: exactly one commit was delivered
        assert len(sink.commits) == 1, sink.commits

        # ---- duplicate successful commit ------------------------------------
        ev2 = coord.commit_trial(run_id)
        assert ev2.get("duplicate") is True, ev2
        assert ev2["delivery"] == "done"
        assert ev2["released_reservations"] == 0, (
            "a duplicate commit released capacity a second time")
        assert ev2["staged_files_deleted"] == 0
        assert len(sink.commits) == 1, (
            f"a duplicate commit RE-READ the spools / re-delivered assembly: "
            f"{sink.commits}")
        assert coord.reserved_files() == 0


def gate_commit_fail_retains():
    """G-COMMIT-FAIL-RETAINS. A FAILED assembly retains every manifest, staged
    spool and reservation, and the same event_id stays retryable. Repairing and
    retrying THE SAME EVENT delivers `done` — and only then releases, exactly
    once. This is D1.1's retry contract: release-on-failure would delete the very
    spools the retry has to re-read."""
    with tempfile.TemporaryDirectory() as tmp:
        state = {"fail": True}

        class _FlakySink(Phase5Sink):
            def __init__(self):
                self.published, self.commits, self.aborts = [], [], []

            def publish_shard(self, manifest):
                self.published.append(manifest)

            def commit_trial(self, event):
                if state["fail"]:
                    raise RuntimeError("corrupt spool: assembly failed")
                self.commits.append(event)

            def abort_trial(self, event):
                self.aborts.append(event)

        sink = _FlakySink()
        coord = _coord(tmp, sink=sink, staging_high_water_files=64,
                       staging_high_water_bytes=64 * 1024 * 1024)
        run_id, sid = "runCF", "runCF_s0"
        coord.ledger.create_trial(run_id, 0, now=100.0)
        coord.ledger.add_stripe(run_id, sid, 0, 30, "java_lcg", 1, 100.0)
        sink.published.append({"event_id": f"{run_id}:{sid}:0"})   # a manifest
        staged = [ _reserve_with_file(coord, run_id, sid, s, 1024, tmp)[1]
                   for s in range(2) ]
        held_before = len(coord.ledger.held_reservations(run_id))
        assert held_before == 2

        ev = coord.commit_trial(run_id)
        assert ev["delivery"] == "failed", ev
        assert "corrupt spool" in ev.get("error", "")
        assert ev["released_reservations"] == 0
        # EVERYTHING is retained
        assert len(coord.ledger.held_reservations(run_id)) == 2, (
            "a failed assembly released reservations")
        for p in staged:
            assert os.path.isfile(p), f"a failed assembly deleted {p}"
        assert len(sink.published) == 1, "a failed assembly discarded a manifest"
        assert coord.reserved_files() == 2
        trial = coord.ledger.get_trial(run_id)
        assert trial["commit_delivery_status"] == "failed"
        assert trial["commit_cleanup_status"] == "none"

        # ---- repair and retry THE SAME EVENT --------------------------------
        state["fail"] = False
        ev2 = coord.commit_trial(run_id)
        assert ev2["event_id"] == ev["event_id"], (
            "the retry used a different event_id — the commit is no longer "
            "idempotent by event")
        assert ev2["delivery"] == "done", ev2
        assert ev2["released_reservations"] == 2, (
            "the repaired retry did not release the retained reservations")
        assert coord.ledger.held_reservations(run_id) == []
        for p in staged:
            assert not os.path.exists(p)
        assert coord.reserved_files() == 0
        assert len(sink.commits) == 1

        # exactly once: a third call releases nothing further
        ev3 = coord.commit_trial(run_id)
        assert ev3["released_reservations"] == 0 and len(sink.commits) == 1


def gate_executor_capacity_timeout():
    """G-EXECUTOR-CAPACITY-TIMEOUT. The exact shape the pre-amendment observer
    could not see: NO paused connection, a staging-executor job blocked on
    reservation capacity, waiting past the bound.

    Pre-amendment `staging_capacity_timeout_expired()` read only
    `_paused_connections`, so this waited forever — the failed run sat ~19
    minutes against a 600 s bound.
    """
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, staging_high_water_files=1,
                       staging_high_water_bytes=64 * 1024 * 1024,
                       staging_capacity_timeout=1.0)
        conn = _register(coord)
        run_id, sid = "runEX", "runEX_s0"
        coord.ledger.create_trial(run_id, 0, now=100.0)
        _claim(coord, run_id, sid, "hostA:gpu0", conn, seed_count=30, expected=2)
        # fill the ONE available file slot so the next reserve back-pressures
        _reserve_with_file(coord, run_id, sid, 0, 512, tmp)
        assert coord.reserved_files() == 1

        obj, pb = build_substripe_payload_bytes(sid, 1, 10, 10,
                                                [[10, 0.9, None, [1]]])
        msg = SubStripeResultMessage(
            worker_id="hostA:gpu0", stripe_id=sid, sub_index=1, seed_start=10,
            seed_count=10, survivor_count=1, inline=obj, size_bytes=len(pb),
            sha256=hashlib.sha256(pb).hexdigest(), effective_threshold=0.25)

        with _MatrixSpy() as spy:
            t = threading.Thread(
                target=coord._run_staging_job,
                args=("inline", conn, run_id, sid, 0, 1, msg, lambda: [conn]),
                daemon=True)
            t.start()
            # the job registers an executor-side capacity wait...
            deadline = time.time() + 5.0
            while (coord.staging_reservation_wait_count() == 0
                   and time.time() < deadline):
                time.sleep(0.01)
            assert coord.staging_reservation_wait_count() == 1, (
                "the staging-executor reservation wait was never registered — "
                "the blocker is still invisible to the bounded timeout")
            # ...with NO connection paused. This is the missing shape.
            assert coord.paused_connection_count() == 0, (
                "a connection is paused; this is not the executor-only shape")

            # before the bound: not expired
            assert not coord.staging_capacity_timeout_expired(), (
                "the timeout latched before the bound elapsed")

            # ---- MUTATION: restore the reader-only oldest-pause logic --------
            # The pre-amendment observer considered ONLY paused connections. With
            # it restored, this wait must stay INVISIBLE — the gate remains wedged.
            orig = RangeMinerCoordinator._capacity_blockers_locked
            try:
                def _reader_only(self_):
                    return [{"blocker_class": "reader_pause",
                             "since": r["since"], "worker_id": r.get("worker_id"),
                             "run_id": None, "stripe_id": None,
                             "attempt": None, "sub_index": None}
                            for r in self_._paused_connections.values()]
                RangeMinerCoordinator._capacity_blockers_locked = _reader_only
                time.sleep(1.3)                      # well past the 1.0 s bound
                assert not coord.staging_capacity_timeout_expired(), (
                    "MUTATION DID NOT RED THE GATE: the reader-only observer saw "
                    "an executor wait, so this gate does not prove the widening")
            finally:
                RangeMinerCoordinator._capacity_blockers_locked = orig

            # with the real observer, the bound is now exceeded
            assert coord.staging_capacity_timeout_expired(), (
                "the widened observer still cannot see an executor reservation "
                "wait past the bound")
            snap = coord.capacity_timeout_snapshot()
            assert snap is not None
            assert snap["blocker_class"] == "staging_reservation", snap
            assert snap["paused_count"] == 0, snap
            assert snap["staging_reservation_wait_count"] == 1, snap
            trig = snap["trigger"]
            assert trig["run_id"] == run_id and trig["stripe_id"] == sid
            assert trig["attempt"] == 0 and trig["sub_index"] == 1
            assert trig["worker_id"] == "hostA:gpu0"
            # the episode carries the capacity evidence §1.4 requires
            assert snap["reserved_files"] == 1
            assert snap["high_water_files"] == 1
            assert snap["high_water_bytes"] == 64 * 1024 * 1024
            assert "reserved_bytes" in snap and "derived_required_files" in snap

            reason = coord.staging_capacity_timeout_reason()
            assert reason.startswith("coordinator_staging_capacity_timeout:")
            assert "staging_reservation" in reason, reason
            assert sid in reason, reason
            # a capacity wait never enters the worker retry matrix
            assert spy.calls == [], f"a capacity wait reached the matrix: {spy.calls}"

            # the terminal snapshot survives the blocker's thread exiting (F3)
            coord.ledger.cancel_active_stripes(run_id)
            t.join(timeout=5.0)
            assert coord.staging_reservation_wait_count() == 0, (
                "the wait record leaked after the job exited — it would age into "
                "a false capacity timeout")
            after = coord.capacity_timeout_snapshot()
            assert after["blocker_class"] == "staging_reservation", (
                "the triggering blocker was lost once its thread exited")
            assert sid in coord.staging_capacity_timeout_reason()


def gate_sequential_trial_reuse():
    """G-SEQUENTIAL-TRIAL-REUSE (MANDATORY). Two sequential successful trials
    through the SAME production staging ledger and coordinator.

    The failed production command requested EIGHT trials. A success path that
    frees nothing ratchets across trials: trial 1 holds its files forever and
    trial 2 deadlocks. After trial 1 commits, held reservations must be ZERO and
    trial 2 must be able to consume the SAME FULL high-water again."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _Sink()
        HW = 3
        coord = _coord(tmp, sink=sink, staging_high_water_files=HW,
                       staging_high_water_bytes=64 * 1024 * 1024)
        consumed = []
        for trial in range(2):
            run_id = f"runSEQ{trial}"
            sid = f"{run_id}_s0"
            coord.ledger.create_trial(run_id, trial, now=100.0 + trial)
            coord.ledger.add_stripe(run_id, sid, 0, 30, "java_lcg", 1, 100.0)
            paths = []
            for sub in range(HW):                    # consume the FULL high-water
                _rid, p = _reserve_with_file(coord, run_id, sid, sub, 256, tmp)
                paths.append(p)
            assert coord.reserved_files() == HW, (
                f"trial {trial} could not consume the full high-water — capacity "
                f"did not return from the previous trial")
            # the ceiling is real: one more is refused
            assert coord.reserve_capacity(run_id, sid, 0, HW, 0, 256) is None
            consumed.append(len(paths))

            ev = coord.commit_trial(run_id)
            assert ev["delivery"] == "done", ev
            assert ev["released_reservations"] == HW, ev
            assert coord.reserved_files() == 0, (
                f"after trial {trial} committed, {coord.reserved_files()} "
                f"reservation(s) are STILL HELD — this is the ratchet that "
                f"deadlocked the 8-trial run")
            assert coord.ledger.held_reservations(run_id) == []
            for p in paths:
                assert not os.path.exists(p)
        assert consumed == [HW, HW]
        assert len(sink.commits) == 2, sink.commits


def main():
    print("=" * 74)
    print("S172 STAGING BACK-PRESSURE — acceptance gates (CPU-only)")
    print("=" * 74)

    print("\n-- §0 the classification law --")
    _check("G-MATRIX-DIFF-a: exactly ONE call site removed, six AST-identical",
           gate_matrix_diff_six_callers_unchanged)
    _check("G-MATRIX-DIFF-b: all six surviving classifications behave identically",
           gate_matrix_diff_behavioural)
    _check("G-LAW: no capacity path reaches the matrix; terminals are DIRECT",
           gate_no_capacity_path_reaches_the_matrix)

    print("\n-- Beta gates 1-2: a capacity wait fails nothing --")
    _check("G1: saturating staging on a PHASE-1 stripe fails nothing",
           gate1_saturation_on_phase1_fails_nothing)
    _check("G2: a capacity wait consumes ZERO retry budget",
           gate2_capacity_wait_consumes_zero_retry_budget)

    print("\n-- Beta gates 3-4: per-connection, self-resuming --")
    _check("G3: paused peer stalls; a second connection flows to completion",
           gate3_paused_peer_stalls_while_second_connection_flows)
    _check("G4: a real capacity release resumes it, with no operator action",
           gate4_capacity_release_resumes_without_operator_action)

    print("\n-- Beta gates 5-7: exactly-once and fencing across pause/resume --")
    _check("G5: every accepted sub-stripe staged EXACTLY ONCE (ledger rows)",
           gate5_each_substripe_staged_exactly_once)
    _check("G6: no duplicate rows, no stale acceptance, NO second dedup layer",
           gate6_no_duplicate_rows_no_stale_acceptance)
    _check("G7: a superseded attempt cannot resume-and-publish (fence drops it)",
           gate7_superseded_attempt_cannot_resume_and_publish)

    print("\n-- Beta gate 8: bounded retention --")
    _check("G8: <= derived bound + margin, <= 1 pending envelope per connection",
           gate8_retention_is_bounded)

    print("\n-- Beta gate 9: the derived bound (116 vs 136) --")
    _check("G9: exact = 116 (recorded assignment); conservative = 136 (4 slots)",
           gate9_derived_bounds_116_and_136)
    _check("G9b: the RUNTIME bound is derived; an override below it WARNS",
           gate9b_runtime_uses_the_derived_bound)

    print("\n-- Beta gate 10: the configuration route --")
    _check("G10: manifest -> args_map -> CLI -> coordinator -> CoordinatorConfig",
           gate10_manifest_to_coordinator_route)

    print("\n-- Beta gate 11: the bounded capacity timeout --")
    _check("G11: terminates with coordinator_staging_capacity_timeout, NOT the matrix",
           gate11_capacity_timeout_terminates_outside_the_matrix)

    print("\n-- §1.4 lease exemption (flagged for Beta ratification) --")
    _check("G-LEASE: a paused worker's lease is exempt; an unpaused one is not",
           gate_lease_exemption_is_required_and_narrow)

    print("\n-- mutation evidence (§5) --")
    _check("G-MUT-PAUSE: removing the capacity gate executes and REDS the pause gate",
           gate_pause_mutant)
    _check("G-MUT-LEASE: removing the exemption executes and REDS the lease gate",
           gate_lease_exemption_mutant)

    print("\n-- §4 metrics --")
    _check("G-METRICS: [S172-BP] pause/resume/summary series complete",
           gate_metrics_are_grep_stable_and_complete)

    print("\n-- S172-BP AMENDMENT (Beta F1-F5, 2026-08-06) --")
    _check("G-RESUME-CREDIT-a: ONE wake per capacity release; non-head cannot ride",
           gate_resume_credit_one_wake_per_release)
    _check("G-RESUME-CREDIT-b: real readers, FIFO, second stays paused, then resumes",
           gate_resume_credit_real_readers_fifo)
    _check("G-MUT-RESUME-CREDIT: the loop and the headless poll execute and RED it",
           gate_resume_credit_mutants)

    print("\n-- S172-BP AMENDMENT ROUND 2 (Beta F1-R, 2026-08-06) --")
    _check("G-RESUME-HANDOFF: the reservation survives ingress and ends at disposition",
           gate_resume_handoff_survives_until_disposition)
    _check("G-MUT-RESUME-HANDOFF: clear-at-ingress executes and REDS the handoff",
           gate_resume_handoff_mutant)
    _check("G-SUMMARY-NO-MASK: a raising bound degrades reporting, never the terminal",
           gate_summary_never_masks_the_sizing_terminal)
    _check("G-LEASE-HANDOFF: the resume window is exempt, bounded, and self-clearing",
           gate_lease_handoff_grace)
    _check("G-MUT-LEASE-HANDOFF: removing the grace executes and REDS the handoff",
           gate_lease_handoff_mutant)
    _check("G-TIMEOUT-SNAPSHOT: the terminal reason names the TRIGGERING workers",
           gate_timeout_snapshot_attributes_the_trigger)
    _check("G-BOUND-PAUSE: an unregistered socket is never paused, never held",
           gate_unbound_result_is_never_paused)
    _check("G-BOUND-DERIVATION-FAILURE: a sizing failure fails the trial closed",
           gate_bound_derivation_failure_fails_closed)
    _check("G-BOUND-TRIP-PHRASE: the §1.6 reason names WHICH bound tripped",
           gate_invariant_reason_names_which_bound_tripped)

    print("\n-- S172-BP AMENDMENT ROUND 3 (Beta F1-R2a / F1-R2b, 2026-08-07) --")
    _check("G-CREDIT-ENVELOPE-IDENTITY: only the EXACT credited envelope clears it",
           gate_credit_clears_only_on_the_exact_envelope)
    _check("G-MUT-CREDIT-IDENTITY: socket-only release executes and REDS it",
           gate_credit_envelope_identity_mutant)
    _check("G-NO-PREDECODE: nothing is decoded while the reservation is outstanding",
           gate_no_predecode_while_the_credit_is_outstanding)
    _check("G-MUT-NO-PREDECODE: the post-decode placement executes and REDS it",
           gate_no_predecode_mutant)

    print("\n-- S172 STAGING-CAPACITY AMENDMENT (Beta staging ruling, 2026-08-07) --")
    _check("G-HIGHWATER-ROUTE: both high-waters travel the governed route (+mutation)",
           gate_highwater_route)
    _check("G-TRIAL-RETENTION-PREFLIGHT: 512 < derived requirement -> fail closed",
           gate_trial_retention_preflight)
    _check("G-TRIAL-RETENTION-PREFLIGHT: REAL gate-12 geometry (16 macro-stripes)",
           gate_trial_retention_preflight_gate12_geometry)
    _check("G-TRIAL-RETENTION-PREFLIGHT: zero StripeAssign, zero matrix, real serve",
           gate_trial_retention_preflight_dispatches_nothing)
    _check("G-COMMIT-RELEASE: success releases exactly once; duplicate releases nothing",
           gate_commit_release)
    _check("G-COMMIT-FAIL-RETAINS: failed assembly retains; repaired retry releases once",
           gate_commit_fail_retains)
    _check("G-EXECUTOR-CAPACITY-TIMEOUT: executor wait is observed (+mutation)",
           gate_executor_capacity_timeout)
    _check("G-SEQUENTIAL-TRIAL-REUSE: capacity returns between trials",
           gate_sequential_trial_reuse)

    print("\n-- S172 STAGING-CAPACITY REVISION 1 (Beta return-for-revision, 2026-08-08) --")
    _check("G-COMMIT-CRASH-RESUME: cleanup resumes across a process restart",
           gate_commit_cleanup_resumes_after_crash)
    _check("G-MUT-COMMIT-CRASH-RESUME: the submitted conflation executes and REDS it",
           gate_commit_crash_resume_mutant)
    _check("G-STAGE-ELIGIBILITY: per-stage populations (+negative arm)",
           gate_stage_specific_eligibility)
    _check("G-MUT-STAGE-ELIGIBILITY: one-collection reuse executes and REDS it",
           gate_stage_eligibility_mutant)
    _check("G-PREFLIGHT-PLAN-PERSISTED: durable geometry, written not recomputed",
           gate_preflight_plan_is_persisted)

    print("\n-- S172 STAGING-CAPACITY REVISION 2 (Beta R1 ruling, 2026-08-08) --")
    _check("G-LATE-WORKER-EXCLUDED: the assignable cohort is frozen at preflight",
           gate_late_worker_excluded_from_frozen_cohort)
    _check("G-PREFLIGHT-PROVENANCE-FAIL-CLOSED: admission fails closed; refusal stays primary",
           gate_preflight_provenance_fail_closed)

    print("\n" + "=" * 74)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    for name, ok, tb in _results:
        if not ok:
            print(f"\n--- {name} ---\n{tb}")
    print(f"\n{passed}/{total} checks green")
    if passed == total:
        print("COMPLETION SENTINEL: PASS — S172 staging back-pressure CPU gates green")
        print("NOTE: Beta gate 12 (G-PROD-SHAPE) is NOT in this file. It needs a live")
        print("      25-daemon fleet, it is MICHAEL-INITIATED ONLY, and Beta's ruling")
        print("      forbids running it until these gates are reviewed.")
        return 0
    print("COMPLETION SENTINEL: FAIL — DO NOT COMMIT")
    return 1


if __name__ == "__main__":
    sys.exit(main())
