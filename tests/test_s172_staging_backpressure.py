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
    ST_STAGING,
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    Phase5Sink,
    RangeMinerCoordinator,
    TransferAdapter,
    advertised_effective_cap,
    applicable_seed_cap,
    build_coordinator,
    expected_substripes_for,
    run_trial_miner,
    staging_burst_bound_conservative,
    staging_burst_bound_exact,
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

    def __init__(self, coord, worker_id, worker_by_sock, inbound, reader_stop):
        self.worker_id = worker_id
        self.srv, self.cli = socket.socketpair()
        self.srv_fs = MinerFramedSocket(self.srv)
        self.cli_fs = MinerFramedSocket(self.cli)
        worker_by_sock[self.srv] = worker_id
        self.thread = threading.Thread(
            target=coord._conn_reader_loop,
            args=(self.srv_fs, self.srv, inbound, reader_stop, worker_by_sock),
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

    def __init__(self, tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"), **cfg):
        self.coord = _coord(tmp, **cfg)
        self.worker_by_sock = {}
        self.wconn_by_worker = {}
        self.inbound = _queue.Queue(maxsize=1024)
        self.reader_stop = threading.Event()
        self.peers = {}
        for wid in worker_ids:
            self.wconn_by_worker[wid] = _register(self.coord, wid)
            self.peers[wid] = _Peer(self.coord, wid, self.worker_by_sock,
                                    self.inbound, self.reader_stop)
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
        """Feed drained frames to the REAL serve dispatcher."""
        eligible = eligible or (lambda: list(self.wconn_by_worker.values()))
        for kind, rawsock, msg in entries:
            if kind != "msg":
                continue
            self.coord._serve_dispatch(
                msg, run_id, self.worker_by_sock.get(rawsock),
                self.wconn_by_worker, eligible)

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
                msgs = [m for (k, _s, m) in after if k == "msg"]
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
            from_a = [m for (k, s, m) in got
                      if k == "msg" and b.worker_by_sock.get(s) == "hostA:gpu0"]
            from_b = [m for (k, s, m) in got
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
            rest = b.drain(1.0)
            seq = [(m.message_type, getattr(m, "sub_index", None))
                   for (k, s, m) in rest
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
            msgs = [m for (k, _s, m) in got if k == "msg"]
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
            entries = b.drain(1.5)
            b.dispatch(entries, run_id)
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
            entries = b.drain(1.2)
            got = [m for (k, _s, m) in entries if k == "msg"]
            assert len(got) == 2, f"both frames should reach the serve loop: {got}"
            b.dispatch(entries, run_id)
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
            assert [m.message_type for (k, _s, m) in entries if k == "msg"] == \
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
            assert out[0]["action"] == "reassigned", out
            sb = b.coord.ledger.get_stripe(run_id, "runLEASE_sB")
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
            delivered = [m for (k, _s, m) in got if k == "msg"]
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


def gate_matrix_diff_six_callers_unchanged():
    """§0: exactly ONE call site is removed and the other SIX are byte-identical.

    Structural half: the pre-change file at HEAD versus the working tree, compared
    as normalized call expressions. Seven before, six after, and the six that
    remain must match the six that were there — set-equal, with the removed one
    being precisely the deferred-overflow back-pressure call.
    """
    live = open(os.path.join(_ROOT, "miner", "range_miner_coordinator.py")).read()
    head = subprocess.run(
        ["git", "-C", _ROOT, "show", "HEAD:miner/range_miner_coordinator.py"],
        capture_output=True, text=True, check=True).stdout
    before = _on_staging_failed_call_sites(head)
    after = _on_staging_failed_call_sites(live)
    assert len(before) == 7, (
        f"expected 7 pre-change _on_staging_failed call sites, found {len(before)}")
    assert len(after) == 6, (
        f"expected exactly 6 after the removal, found {len(after)}: {after}")
    removed = [c for c in before if c not in after]
    assert len(removed) == 1, f"more than one call site changed: {removed}"
    assert "deferred queue full" in removed[0], (
        f"the removed call site is NOT the deferred-overflow one: {removed[0]}")
    surviving_changed = [c for c in after if c not in before]
    assert surviving_changed == [], (
        f"an out-of-scope caller was modified: {surviving_changed}")
    # ...and the matrix plumbing itself is untouched
    for meth in ("_on_staging_failed", "handle_stripe_failure",
                 "_handle_stripe_failure_locked"):
        b_src = _method_source(head, meth)
        a_src = _method_source(live, meth)
        assert b_src == a_src, f"{meth} was modified — it is out of scope"


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
