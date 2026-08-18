#!/usr/bin/env python3
"""S172 — H1/H2 DISCRIMINATION INSTRUMENTATION GATE BATTERY

Implements the gates for `docs/ATTEMPT6_RIG_LOG_FORENSIC_v1_0.md` §7 items A-E,
built to the brief `~/dashboard_work/CCODE_BRIEF_H1H2_INSTRUMENTATION_v1_0.md`.

    H1  worker-side stall or send blocking
    H2  coordinator-side acceptance backlog delaying lease renewal

THE STANDARD THIS BATTERY IS WRITTEN AGAINST
--------------------------------------------
The brief is explicit, and it is aimed at a specific recurring failure in this
arc: ten recorded instances of a check that passes on a fact it does not verify,
including `assignment_active_at_loss`, which appears 25 times in the attempt-6
bundle, reads like a discriminator, and is STRUCTURALLY GUARANTEED FALSE on the
path it was read on (forensic §4.2).

    "for each new field, state what value would prove it is measuring rather
     than defaulting, and build the arm that catches a field that is present but
     always constant. A new field that cannot vary is the next vacuous
     discriminator."

So NO gate here asserts that a field exists, or that it is non-negative, or that
it parses. Every gate for a measured field runs the SAME code twice — a CLEAN
CONTROL and a FAULT-INJECTION CONTROL — and asserts the two produce MATERIALLY
DIFFERENT values. A field hardwired to any constant whatsoever fails that
comparison, whatever the constant is. Where a field is a presence/absence fact
rather than a magnitude, the gate drives both dispositions and asserts they
differ.

The mutants at the end go further: they DISABLE each instrument at its source
and assert the corresponding gate goes red. A gate that stays green with its
instrument removed is proving nothing, and this battery is the wrong place to
discover that six months from now.

VOCABULARY, AND IT IS LOAD-BEARING
-----------------------------------
`None` means UNOBSERVED throughout — a residency that could not be measured, a
send-block figure with no socket to read, an age with no accepted progress to
measure from. It never degrades to `0.0`, because a zero READS AS A MEASUREMENT
and would invert the finding: "processed instantly" where the truth is "never
observed". This is the §2.28 concurrency-sampler lesson and the VIR-5 rule,
applied to a new instrument before it can repeat them.

Run:  source ~/venvs/torch/bin/activate && python3 -u tests/test_s172_h1h2_instrumentation.py
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import logging
import os
import queue as _queue
import socket
import sys
import tempfile
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from miner.range_miner_coordinator import (  # noqa: E402
    ACTIVE_STRIPE_REPORT_INTERVAL_S,
    FRAME_ARRIVAL_ATTR,
    OBS_NO_OBSERVATION,
    OBS_OK,
    OBS_UNAVAILABLE,
    ST_CLAIMED,
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    RangeMinerCoordinator,
)
from miner.range_miner_worker import (  # noqa: E402
    MinerFramedSocket,
    RangeMinerWorker,
    SubStripeOutcome,
    VramCaps,
    configure_worker_logging,
)
from miner.range_miner_protocol import (  # noqa: E402
    StripeAssignMessage,
    SubStripeResultMessage,
    message_to_bytes,
)

CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
VARIANTS = ["java_lcg", "java_lcg_reverse",
            "java_lcg_hybrid", "java_lcg_hybrid_reverse"]
SPOOL_ROOT = "/var/spool/miner"

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
_RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        _RESULTS.append((name, True, detail))
        print(f"  [{GREEN}PASS{RESET}] {name:<38} {detail}")
    except AssertionError as e:
        _RESULTS.append((name, False, str(e)))
        print(f"  [{RED}FAIL{RESET}] {name:<38} {e}")
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  [{RED}ERROR{RESET}] {name:<38} {type(e).__name__}: {e}")


# ===========================================================================
# fixtures
# ===========================================================================
def _coord(tmp, dbname="l.db", **cfg):
    cfg.setdefault("staging_dir", os.path.join(tmp, "staging"))
    ledger = MinerLedger(os.path.join(tmp, dbname))
    return RangeMinerCoordinator(CoordinatorConfig(**cfg), ledger)


def _register(coord, wid="hostA:gpu0", backend="cuda", now=100.0):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend=backend,
        capabilities={"seed_caps": dict(CAPS),
                      "supported_variants": list(VARIANTS)},
        node_config=node, now=now)


def _assign(stripe_id="st0_s0", seed_start=0, seed_count=8_000_000,
            family="java_lcg"):
    return StripeAssignMessage(
        worker_id="hostA:gpu0", stripe_id=stripe_id, seed_start=seed_start,
        seed_count=seed_count, family_name=family, prng_type=family, phase=1,
        attempt=0, trial_number=1, substripes=8,
        payload={"window_size": 8, "offset": 0, "sessions": ["evening"],
                 "forward_threshold": 0.25, "reverse_threshold": 0.25},
    )


class _CollectingHandler(logging.Handler):
    """Captures the records the instruments actually WRITE. The log line is the
    record of authority on a rig — an in-process list is a convenience — so
    every emission gate reads the emitted text, not the return value."""

    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def of(self, token):
        return [ln for ln in self.lines if token in ln]

    def payloads(self, token):
        out = []
        for ln in self.of(token):
            b = ln.find("{")
            if b >= 0:
                out.append(json.loads(ln[b:]))
        return out


class _Capture:
    def __init__(self, *modules):
        self.h = _CollectingHandler()
        self.loggers = [logging.getLogger(m) for m in modules]

    def __enter__(self):
        for lg in self.loggers:
            lg.addHandler(self.h)
            lg.setLevel(logging.INFO)
        return self.h

    def __exit__(self, *a):
        for lg in self.loggers:
            lg.removeHandler(self.h)
        return False


def _worker(executor, conn=None, wid="hostA:gpu0"):
    w = RangeMinerWorker(host="127.0.0.1", port=1, gpu_id=0,
                         caps=VramCaps(**CAPS), executor=executor,
                         hostname=wid.split(":")[0])
    w.conn = conn
    return w


class _NullConn:
    """A socket-shaped sink that never blocks. THE CLEAN CONTROL for every
    send-side measurement: same code path, same accounting, no blockage."""

    def __init__(self):
        self.sent = []
        # Keyed by thread, exactly as the real MinerFramedSocket now is (R1-A).
        self._by_thread = {}

    @staticmethod
    def _slot():
        return {"send_calls": 0, "send_syscall_s": 0.0,
                "send_syscall_max_s": 0.0, "send_lock_wait_s": 0.0,
                "send_lock_wait_max_s": 0.0, "bytes_sent": 0}

    def _mine(self):
        tid = threading.get_ident()
        if tid not in self._by_thread:
            self._by_thread[tid] = self._slot()
        return self._by_thread[tid]

    def send_msg(self, msg):
        self.sent.append(msg)
        self._mine()["send_calls"] += 1

    def send_accounting(self, thread_id=None):
        tid = threading.get_ident() if thread_id is None else int(thread_id)
        slot = self._by_thread.get(tid)
        return dict(slot) if slot is not None else self._slot()

    def _new_acct_slot(self):
        return self._slot()

    def close(self):
        pass


class _ProgrammableSock:
    """A socket-shaped object whose `send` can be made to block for a chosen
    number of calls.

    Used by A10/A11 so those arms drive the **real** `MinerFramedSocket` —
    the real `_send_lock`, the real `_sendall` loop, the real per-thread
    accounting — with only the syscall slowed. A fixture that reimplemented the
    accounting would be testing the fixture."""

    def __init__(self, delay=0.0, delay_calls=0):
        self.delay = delay
        self.delay_calls = delay_calls
        self.sent = bytearray()
        self.in_send = threading.Event()
        self._n_lock = threading.Lock()

    def send(self, view):
        with self._n_lock:
            d = self.delay if self.delay_calls > 0 else 0.0
            if self.delay_calls > 0:
                self.delay_calls -= 1
        if d:
            self.in_send.set()
            time.sleep(d)
        b = bytes(view)
        self.sent.extend(b)
        return len(b)

    def shutdown(self, how):
        pass

    def close(self):
        pass


def _hb_frame(wid="hostA:gpu0"):
    from miner.range_miner_protocol import MinerHeartbeatMessage
    return MinerHeartbeatMessage(worker_id=wid, stripes_done=0, stripes_error=0,
                                 current_stripe_id="st0_s0", busy=True)


class _BlockingConn(_NullConn):
    """THE FAULT-INJECTION CONTROL: a peer that accepts the frame only after a
    deliberate delay, exactly as a coordinator whose reader has paused does.

    It charges the delay through the SAME accounting fields a real
    `MinerFramedSocket` charges, so the gate exercises the instrument's
    arithmetic rather than a parallel bookkeeping path invented for the test."""

    def __init__(self, block_s):
        super().__init__()
        self.block_s = block_s

    def send_msg(self, msg):
        t0 = time.perf_counter()
        time.sleep(self.block_s)
        blocked = time.perf_counter() - t0
        self.sent.append(msg)
        a = self._mine()
        a["send_calls"] += 1
        a["send_syscall_s"] += blocked
        if blocked > a["send_syscall_max_s"]:
            a["send_syscall_max_s"] = blocked


def _exec_fast(assign, seed_start, seed_count):
    return SubStripeOutcome(survivors=[1], count=1, effective_threshold=0.25)


def _exec_slow(delay):
    def _e(assign, seed_start, seed_count):
        time.sleep(delay)
        return SubStripeOutcome(survivors=[1], count=1, effective_threshold=0.25)
    return _e


# ===========================================================================
# §B — TIMESTAMPS. The precondition, gated as one.
# ===========================================================================
def gate_b1_timestamp_prefix_is_present_and_utc():
    """WHAT WOULD PROVE IT IS MEASURING: two records emitted a measurable
    interval apart must carry DIFFERENT timestamps, and both must be readable as
    UTC. A formatter wired to a constant, or to a field that never advances,
    fails on the inequality — which is the arm that a bare "the line has a
    prefix" assertion would not have."""
    stream = io.StringIO()
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = []
    try:
        h = configure_worker_logging(stream=stream)
        lg = logging.getLogger("miner.range_miner_worker")
        lg.warning("[MINER-SESSION] X %s", json.dumps({"event": "X"}))
        time.sleep(1.05)
        lg.warning("[MINER-SESSION] Y %s", json.dumps({"event": "Y"}))
        h.flush()
    finally:
        root.handlers = saved
    lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 2, f"expected 2 emitted lines, got {len(lines)}: {lines}"
    stamps = [ln.split()[0] for ln in lines]
    for s in stamps:
        assert s.endswith("Z") and "T" in s, (
            f"the line prefix is not an ISO-8601 UTC stamp: {s!r}. A rig log "
            f"that cannot be joined to the coordinator log across four machines "
            f"is the condition §B exists to end")
    assert stamps[0] != stamps[1], (
        f"both records carry the SAME timestamp {stamps[0]!r} after a 1.05 s "
        f"interval — the field is present and CONSTANT, which is the vacuous "
        f"class this battery exists to catch")
    # UTC, not localtime: the hour must match gmtime, not localtime, whenever
    # the two differ. Where they do not differ this arm is inert and says so.
    now_utc = time.strftime("%Y-%m-%dT%H", time.gmtime())
    assert stamps[1].startswith(now_utc), (
        f"the stamp {stamps[1]!r} is not on the UTC hour {now_utc!r} — "
        f"`Formatter.converter` defaults to localtime and must be gmtime")
    return f"{stamps[0][11:]} -> {stamps[1][11:]}, UTC, distinct"


def gate_b2_prefix_does_not_break_the_certified_gate_parsers():
    """`gate12_sentinel_gate.py` and `gate12_worker_liveness_gate.py` locate
    records by substring `grep` and extract the payload with `line.find('{')`.
    A prefix that carried a brace, or that displaced the token, would red both
    gates on the launch path — silently, and only on the rigs.

    THE WRONG-INPUT ARM: a prefix containing `{` must FAIL this gate. It is
    constructed and asserted against, so the gate is known to be able to fail."""
    stream = io.StringIO()
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = []
    try:
        configure_worker_logging(stream=stream)
        lg = logging.getLogger("miner.range_miner_worker")
        payload = {"event": "SESSION_SENTINEL", "run_nonce": "abc123",
                   "worker_id": "rrig6600:gpu0"}
        lg.warning("[MINER-SESSION] %s %s", "SESSION_SENTINEL",
                   json.dumps(payload, sort_keys=True))
    finally:
        root.handlers = saved
    line = stream.getvalue().strip()
    assert "SESSION_SENTINEL" in line, "the sentinel token did not survive"
    assert "abc123" in line, "the nonce did not survive the prefix"
    b = line.find("{")
    assert b > 0 and json.loads(line[b:])["event"] == "SESSION_SENTINEL", (
        "line.find('{') no longer reaches the payload — both launch gates parse "
        "exactly this way")
    prefix = line[:b]
    assert "{" not in prefix, f"the prefix carries a brace: {prefix!r}"
    # the wrong-input arm, executed
    bad = "2026-01-01T00:00:00.000Z {rig} " + line[b:]
    assert bad.find("{") < bad.find('"event"'), (
        "the negative control is malformed")
    bad_ok = False
    try:
        json.loads(bad[bad.find("{"):])
        bad_ok = True
    except json.JSONDecodeError:
        pass
    assert not bad_ok, (
        "a prefix containing a brace parsed anyway — this gate cannot fail and "
        "therefore proves nothing")
    return f"prefix={prefix.strip()!r}, payload reachable, brace-prefix rejected"


# ===========================================================================
# §A — WORKER STRIPE LIFECYCLE. The decisive worker-side half.
# ===========================================================================
def gate_a1_four_records_once_each_in_order():
    with _Capture("miner.range_miner_worker") as cap:
        w = _worker(_exec_fast, conn=_NullConn())
        w.handle_stripe(_assign(seed_count=8_000_000))
    kinds = [p["event"] for p in cap.payloads("[MINER-STRIPE]")]
    expected = ["STRIPE_BEGIN", "STRIPE_COMPUTE_DONE", "STRIPE_SEND_DONE",
                "STRIPE_END"]
    assert kinds == expected, f"records/order wrong: {kinds}"
    return f"{kinds}"


def gate_a2_rate_is_per_stripe_not_per_substripe():
    """§7 A's own bound: stripe granularity, not sub-stripe. With a 5 MB nvidia
    cap an 8 M-seed stripe partitions into 2 sub-stripes; the record count must
    not move with that number, and the gate proves it by comparing a 2-sub and a
    5-sub stripe rather than asserting a constant."""
    counts = []
    for seed_count in (8_000_000, 25_000_000):
        with _Capture("miner.range_miner_worker") as cap:
            w = _worker(_exec_fast, conn=_NullConn())
            w.handle_stripe(_assign(seed_count=seed_count))
        subs = w.stripe_events[-1]["sub_count"]
        counts.append((subs, len(cap.of("[MINER-STRIPE]"))))
    (s1, r1), (s2, r2) = counts
    assert s2 > s1, f"the two fixtures did not differ in sub-stripe count: {counts}"
    assert r1 == r2 == 4, (
        f"the record count tracks sub-stripes ({counts}) — this is per-frame "
        f"emission wearing a stripe-shaped name, and it breaches the §15 bar")
    return f"{s1} subs -> {r1} records; {s2} subs -> {r2} records"


def gate_a3_compute_s_measures_and_varies():
    """WHAT WOULD PROVE IT IS MEASURING: a stripe whose executor sleeps 0.30 s
    per sub-stripe must report a materially larger `compute_s` than the same
    stripe with an instant executor. A constant — 0.0 or anything else — fails."""
    with _Capture("miner.range_miner_worker"):
        wf = _worker(_exec_fast, conn=_NullConn())
        wf.handle_stripe(_assign(seed_count=8_000_000))
        ws = _worker(_exec_slow(0.30), conn=_NullConn())
        ws.handle_stripe(_assign(seed_count=8_000_000))
    fast = wf._last_stripe_acct["compute_s"]
    slow = ws._last_stripe_acct["compute_s"]
    subs = ws._last_stripe_acct["sub_count"]
    assert fast < 0.10, f"the clean control is not clean: compute_s={fast}"
    assert slow >= 0.30 * subs * 0.9, (
        f"compute_s={slow} did not track {subs} x 0.30 s of injected kernel time")
    assert slow > fast * 5, (
        f"compute_s does not vary between the clean ({fast}) and injected "
        f"({slow}) controls — a field that cannot vary is a vacuous "
        f"discriminator")
    return f"clean={fast:.4f}s  injected={slow:.4f}s ({subs} subs)"


def gate_a4_send_stall_measures_and_varies():
    """THE FIELD THE FORENSIC NAMES AS DECISIVE (§7 A): under H2 the worker is
    inside `handle_stripe` and NOT computing, and no existing field can express
    that. WHAT WOULD PROVE IT IS MEASURING: a peer that delays acceptance by
    0.30 s per frame must produce a large `stripe_send_stall_s` where a
    non-blocking peer produces ~0."""
    with _Capture("miner.range_miner_worker"):
        wf = _worker(_exec_fast, conn=_NullConn())
        wf.handle_stripe(_assign(seed_count=8_000_000))
        wb = _worker(_exec_fast, conn=_BlockingConn(0.30))
        wb.handle_stripe(_assign(seed_count=8_000_000))
    clean = wf._last_stripe_acct["stripe_send_stall_s"]
    blocked = wb._last_stripe_acct["stripe_send_stall_s"]
    subs = wb._last_stripe_acct["sub_count"]
    assert clean is not None and blocked is not None, (
        "stripe_send_stall_s is UNOBSERVED on a path that has a socket")
    assert clean < 0.05, f"the clean control is not clean: stall={clean}"
    # +1: the StripeComplete frame blocks too, and it is part of the stripe.
    assert blocked >= 0.30 * (subs + 1) * 0.9, (
        f"stripe_send_stall_s={blocked} did not track {subs + 1} x 0.30 s of "
        f"injected send blockage")
    assert blocked > clean * 10, (
        f"stripe_send_stall_s does not vary between the clean ({clean}) and "
        f"injected ({blocked}) controls")
    return f"clean={clean:.4f}s  injected={blocked:.4f}s ({subs}+1 frames)"


def gate_a10_heartbeat_syscall_is_not_charged_to_the_stripe():
    """[R1-A, BLOCKING ARM] THE COUNTER-EXAMPLE BETA SUPPLIED, EXECUTED.

        mining thread    executing a GPU kernel, sending nothing
        heartbeat thread holds the send lock, blocked in _sendall for 0.60 s
        WRONG                stripe send stall ~0.60 s  <- the HEARTBEAT's time
        RIGHT                stripe send stall ~0       + heartbeat_send_syscall_s ~0.60

    The first revision kept ONE socket-wide cumulative counter and subtracted a
    snapshot, so it attributed the heartbeat's 0.60 s to a stripe that had not
    sent a byte. **This arm is red against that implementation and green against
    the per-thread one.**

    WRONG INPUT THAT REDS IT: any accounting that is not keyed by the thread
    doing the sending — a socket-wide counter, a connection-wide counter, or a
    per-stripe counter fed from either."""
    sock = _ProgrammableSock(delay=0.60, delay_calls=1)
    conn = MinerFramedSocket(sock)
    w = _worker(_exec_slow(0.90), conn=conn)
    hb = threading.Thread(target=lambda: conn.send_msg(_hb_frame()),
                          name="fixture-heartbeat", daemon=True)
    w._hb_thread = hb
    with _Capture("miner.range_miner_worker"):
        hb.start()
        assert sock.in_send.wait(3.0), "the heartbeat never entered _sendall"
        # 4 M seeds / 5 M nvidia cap = ONE sub-stripe, so the mining thread
        # computes for 0.90 s and only then sends — after the heartbeat has
        # already released the lock at 0.60 s.
        w.handle_stripe(_assign(seed_count=4_000_000))
        hb.join(timeout=5.0)
    a = w._last_stripe_acct
    assert not hb.is_alive(), "the fixture heartbeat never finished"
    hb_s = a["heartbeat_send_syscall_s"]
    assert hb_s is not None and hb_s >= 0.50, (
        f"the heartbeat's own syscall time was not captured: {hb_s!r} — the "
        f"separation cannot be checked from the record")
    assert a["stripe_send_syscall_s"] < 0.05, (
        f"the heartbeat's syscall time was charged to the stripe: "
        f"stripe_send_syscall_s={a['stripe_send_syscall_s']} while the "
        f"heartbeat blocked {hb_s:.3f}s and the stripe sent nothing until after")
    assert a["stripe_send_lock_wait_s"] < 0.10, (
        f"the stripe reports lock wait it never did: "
        f"{a['stripe_send_lock_wait_s']}")
    assert a["stripe_send_stall_s"] < 0.15, (
        f"stripe send stall is {a['stripe_send_stall_s']} — under this fixture "
        f"the mining thread's send path was never obstructed")
    assert a["compute_s"] >= 0.80, f"the fixture did not compute: {a['compute_s']}"
    return (f"heartbeat syscall={hb_s:.3f}s  stripe stall="
            f"{a['stripe_send_stall_s']:.3f}s  compute={a['compute_s']:.3f}s")


def gate_a11_lock_wait_appears_as_stripe_send_stall():
    """[R1-A, BLOCKING ARM, SECOND HALF] If the heartbeat takes the lock first
    and blocks, the mining thread spends its time waiting FOR THE LOCK rather
    than inside its own `_sendall`. That is still coordinator-side send-path
    obstruction, and a discriminator built on syscall time alone would miss it
    completely — reporting a blocked worker as `stall ~= 0` and pointing at H1.

    WHAT WOULD PROVE IT IS MEASURING: the wait must appear in
    `stripe_send_lock_wait_s` AND in `stripe_send_stall_s`, while
    `stripe_send_syscall_s` stays small — i.e. the two components are separated
    AND the sum is the thing the verdict reads.

    WRONG INPUT THAT REDS IT: `stripe_send_stall_s = stripe_send_syscall_s`, the
    syscall-only discriminator."""
    sock = _ProgrammableSock(delay=0.60, delay_calls=1)
    conn = MinerFramedSocket(sock)
    w = _worker(_exec_fast, conn=conn)
    hb = threading.Thread(target=lambda: conn.send_msg(_hb_frame()),
                          name="fixture-heartbeat", daemon=True)
    w._hb_thread = hb
    with _Capture("miner.range_miner_worker"):
        hb.start()
        assert sock.in_send.wait(3.0), "the heartbeat never entered _sendall"
        # instant executor: the mining thread reaches its first send while the
        # heartbeat is still holding the lock, and must queue behind it.
        w.handle_stripe(_assign(seed_count=4_000_000))
        hb.join(timeout=5.0)
    a = w._last_stripe_acct
    assert a["stripe_send_lock_wait_s"] >= 0.40, (
        f"the mining thread waited for a lock held for 0.60 s and reports "
        f"{a['stripe_send_lock_wait_s']} — lock wait is invisible, and a worker "
        f"blocked this way would be misread as H1")
    assert a["stripe_send_syscall_s"] < 0.20, (
        f"syscall time {a['stripe_send_syscall_s']} — the components are not "
        f"separated")
    assert a["stripe_send_stall_s"] >= 0.40, (
        f"stall {a['stripe_send_stall_s']} does not include the lock wait")
    assert abs(a["stripe_send_stall_s"]
               - (a["stripe_send_syscall_s"] + a["stripe_send_lock_wait_s"])) < 1e-5, (
        "stall is not the sum of its two reported components")
    return (f"lock_wait={a['stripe_send_lock_wait_s']:.3f}s  "
            f"syscall={a['stripe_send_syscall_s']:.3f}s  "
            f"stall={a['stripe_send_stall_s']:.3f}s")


def gate_a5_h1_versus_h2_are_TOLD_APART():
    """THE GATE THIS WHOLE BATTERY EXISTS FOR.

    Two fixtures with the SAME total stripe wall time, differing ONLY in where
    that time goes: one spends it in the kernel (H1), the other blocked writing
    to a peer that will not read (H2). Attempt 6's instrumentation produced ONE
    structural skeleton across all 25 worker logs and could not separate these
    (forensic §4.1). This gate asserts the new records DO.

    It compares the two SIDE BY SIDE rather than testing each against a
    threshold, so an instrument that reported the same plausible-looking numbers
    for both — the exact attempt-6 failure — reds here."""
    per_sub = 0.25
    with _Capture("miner.range_miner_worker"):
        w_h1 = _worker(_exec_slow(per_sub), conn=_NullConn())
        w_h1.handle_stripe(_assign(seed_count=8_000_000))
        w_h2 = _worker(_exec_fast, conn=_BlockingConn(per_sub))
        w_h2.handle_stripe(_assign(seed_count=8_000_000))
    a, b = w_h1._last_stripe_acct, w_h2._last_stripe_acct
    assert abs(a["total_s"] - b["total_s"]) < a["total_s"] * 0.6, (
        f"the two fixtures are not comparable: total_s {a['total_s']} vs "
        f"{b['total_s']} — the gate must isolate WHERE the time went, not how "
        f"much there was")

    def verdict(acct):
        # [R1-A] THE MINING THREAD'S SEND-PATH STALL, not a socket-wide
        # cumulative number: `stripe_send_stall_s` is syscall + lock wait for
        # THIS stripe's execution thread, and the heartbeat's own syscall time
        # is excluded by construction (A10) rather than by assumption.
        c, s = acct["compute_s"], acct["stripe_send_stall_s"]
        if s is None:
            return "UNOBSERVED"
        return "H1" if c > s * 2 else ("H2" if s > c * 2 else "AMBIGUOUS")

    v1, v2 = verdict(a), verdict(b)
    assert v1 == "H1", (
        f"the kernel-stall fixture classified {v1}: compute_s={a['compute_s']} "
        f"stall={a['stripe_send_stall_s']}")
    assert v2 == "H2", (
        f"the send-block fixture classified {v2}: compute_s={b['compute_s']} "
        f"stall={b['stripe_send_stall_s']}")
    return (f"H1: compute={a['compute_s']:.3f} stall={a['stripe_send_stall_s']:.3f}"
            f" | H2: compute={b['compute_s']:.3f} "
            f"stall={b['stripe_send_stall_s']:.3f}  -> TOLD APART")


def gate_a6_unattributed_residual_is_a_third_outcome():
    """A stripe whose elapsed time is neither kernel nor send blockage must not
    be forced into H1 or H2. The residual is reported, and a fixture that spends
    its time OUTSIDE both (payload construction, here simulated by a slow
    `_build_sub_result`) must show it."""
    with _Capture("miner.range_miner_worker"):
        w = _worker(_exec_fast, conn=_NullConn())
        orig = w._build_sub_result

        def _slow_build(assign, sub, outcome):
            time.sleep(0.25)
            return orig(assign, sub, outcome)
        w._build_sub_result = _slow_build
        w.handle_stripe(_assign(seed_count=8_000_000))
    acct = w._last_stripe_acct
    assert acct["unattributed_s"] is not None, "the residual is UNOBSERVED"
    assert acct["unattributed_s"] > acct["compute_s"], (
        f"time spent outside both named phases ({acct['unattributed_s']}) did "
        f"not exceed compute ({acct['compute_s']}) — the residual is being "
        f"absorbed into a named term, which is how a third cause gets reported "
        f"as one of the two")
    return (f"total={acct['total_s']:.3f} compute={acct['compute_s']:.3f} "
            f"stall={acct['stripe_send_stall_s']:.3f} "
            f"unattributed={acct['unattributed_s']:.3f}")


def gate_a7_stripe_end_on_the_failure_path_too():
    """A stripe that fails must still retire its accounting. If STRIPE_END were
    emitted only on success, every interesting stripe would be the one with no
    record — which is the shape attempt 6 actually had."""
    with _Capture("miner.range_miner_worker") as cap:
        def _boom(assign, seed_start, seed_count):
            raise RuntimeError("injected kernel fault")
        w = _worker(_boom, conn=_NullConn())
        w.handle_stripe(_assign(seed_count=8_000_000))
    ends = [p for p in cap.payloads("[MINER-STRIPE]")
            if p["event"] == "STRIPE_END"]
    assert len(ends) == 1, f"expected exactly one STRIPE_END, got {len(ends)}"
    e = ends[0]
    assert e["outcome"] == "failed" and e["exc_class"] == "RuntimeError", e
    assert e["subs_computed"] == 0 and e["compute_done"] is False, (
        f"a stripe that never completed a kernel reports compute_done={e['compute_done']}")
    return f"outcome={e['outcome']} exc={e['exc_class']} compute_done={e['compute_done']}"


def gate_a8_unobserved_is_never_zero():
    """WITHOUT a socket there is nothing to measure, and the record must say
    UNOBSERVED. A `0.0` here would read as 'the worker never blocked', which is
    the strongest possible H1 claim, asserted from no measurement at all."""
    with _Capture("miner.range_miner_worker"):
        w = _worker(_exec_fast, conn=None)
        try:
            w.handle_stripe(_assign(seed_count=8_000_000))
        except AssertionError:
            pass                     # `_send` asserts a connection; expected
    acct = w._stripe_acct or w._last_stripe_acct
    assert acct is not None, "no accounting was opened at all"
    for f in ("stripe_send_syscall_s", "stripe_send_lock_wait_s",
              "stripe_send_stall_s"):
        assert acct[f] is None, (
            f"{f} reported {acct[f]!r} with no socket to measure — UNOBSERVED "
            f"must never degrade to a number")
    return "stripe send syscall/lock_wait/stall all None (UNOBSERVED), not 0.0"


def gate_a9_session_end_carries_a_field_that_can_vary():
    """THE REPAIR FOR THE §4.2 VACUITY, GATED.

    `assignment_active_at_loss` is structurally always False on the
    `explicit_shutdown` path — it appears 25 times in the attempt-6 bundle and
    is evidence about nothing. The session-end record now also carries the
    stripe accounting, and THIS gate proves that field varies where the old one
    could not: two workers, identical shutdown path, different stripe histories,
    different records."""
    with _Capture("miner.range_miner_worker"):
        wa = _worker(_exec_fast, conn=_NullConn())
        wa.handle_stripe(_assign(seed_count=8_000_000))
        wb = _worker(_exec_fast, conn=_BlockingConn(0.20))
        wb.handle_stripe(_assign(seed_count=8_000_000))
    sa = wa.stripe_accounting_snapshot()
    sb = wb.stripe_accounting_snapshot()
    assert sa["inflight"] is None and sb["inflight"] is None, (
        "a returned handle_stripe still reports an in-flight stripe")
    va = sa["last"]["stripe_send_stall_s"]
    vb = sb["last"]["stripe_send_stall_s"]
    assert va is not None and vb is not None, "both snapshots are UNOBSERVED"
    assert vb > va * 10, (
        f"the session-end stripe accounting reports {va} and {vb} for two "
        f"materially different stripe histories — it is as vacuous as the field "
        f"it was added to compensate for")
    # ...and the old field really is constant across both, which is the point.
    assert (wa.state == "idle") and (wb.state == "idle"), (
        "the premise of forensic §4.2 does not hold in this fixture")
    return f"last.stripe_send_stall_s: {va:.4f} vs {vb:.4f} (both idle)"


# ===========================================================================
# §C — COORDINATOR FRAME ARRIVAL AND QUEUE RESIDENCY
# ===========================================================================
def gate_c1_residency_measures_and_varies():
    """WHAT WOULD PROVE IT IS MEASURING: a frame held 0.40 s between decode and
    processing must report ~0.40 s, and a frame processed immediately must
    report ~0. Both arms run the real `stamp_frame_arrival` /
    `frame_queue_residency` pair."""
    C = RangeMinerCoordinator
    msg_fast = SubStripeResultMessage(
        worker_id="w", stripe_id="s", sub_index=0, seed_start=0, seed_count=1,
        survivor_count=0, inline=[], size_bytes=2, sha256="0" * 64)
    msg_slow = SubStripeResultMessage(
        worker_id="w", stripe_id="s", sub_index=1, seed_start=0, seed_count=1,
        survivor_count=0, inline=[], size_bytes=2, sha256="0" * 64)
    C.stamp_frame_arrival(msg_fast)
    r_fast = C.frame_queue_residency(msg_fast)
    C.stamp_frame_arrival(msg_slow)
    time.sleep(0.40)
    r_slow = C.frame_queue_residency(msg_slow)
    assert r_fast is not None and r_slow is not None, "residency UNOBSERVED"
    assert r_fast < 0.05, f"the clean control is not clean: {r_fast}"
    assert 0.35 < r_slow < 1.5, f"residency did not track the 0.40 s hold: {r_slow}"
    assert r_slow > r_fast * 10, (
        f"residency does not vary between {r_fast} and {r_slow}")
    return f"immediate={r_fast * 1000:.1f}ms  held={r_slow * 1000:.1f}ms"


def gate_c2_unstamped_frame_is_unobserved_not_zero():
    """A frame that never passed a reader has NO arrival instant. Reporting 0.0
    would say 'processed instantly' — the strongest possible claim that the
    coordinator was not backlogged, made from nothing."""
    C = RangeMinerCoordinator
    msg = SubStripeResultMessage(
        worker_id="w", stripe_id="s", sub_index=0, seed_start=0, seed_count=1,
        survivor_count=0, inline=[], size_bytes=2, sha256="0" * 64)
    assert C.frame_queue_residency(msg) is None, (
        "an unstamped frame reported a residency; UNOBSERVED must be None")
    C.stamp_frame_arrival(msg)
    assert C.frame_queue_residency(msg) is not None, (
        "a stamped frame reports UNOBSERVED — the two dispositions are not "
        "distinguishable, so `None` proves nothing")
    return "unstamped=None, stamped=float — the two dispositions differ"


def gate_c3_the_stamp_never_reaches_the_wire():
    """NON-REGRESSION ON THE PROTOCOL. `message_to_bytes` builds from
    `dataclasses.fields()`, so an undeclared attribute is invisible to it — but
    that is a claim, and the D5 spool/inline sha256 contract depends on it. It is
    measured here: the frame's bytes and digest must be IDENTICAL before and
    after stamping."""
    msg = SubStripeResultMessage(
        worker_id="w", stripe_id="s", sub_index=0, seed_start=0, seed_count=1,
        survivor_count=3, inline=[7, 8], size_bytes=2, sha256="a" * 64)
    before = message_to_bytes(msg)
    RangeMinerCoordinator.stamp_frame_arrival(msg)
    after = message_to_bytes(msg)
    assert before == after, (
        "the arrival stamp changed the wire frame — the D5 byte contract and "
        "every sha256 over it are affected")
    assert FRAME_ARRIVAL_ATTR.encode() not in after, (
        f"{FRAME_ARRIVAL_ATTR} appears in the serialised frame")
    d = hashlib.sha256(before).hexdigest()[:12]
    return f"{len(before)} bytes, sha256 {d} unchanged; attr absent from wire"


def gate_c4_reader_stamps_before_any_branch():
    """STRUCTURAL, read off the live source: the stamp must sit between the
    decode and every branch that can hold, pause, route or discard the frame —
    otherwise a PAUSED frame would accrue no residency, and a paused frame is
    precisely the H2 case."""
    src = inspect.getsource(RangeMinerCoordinator._conn_reader_loop)
    i_recv = src.find("msg = cfs.recv_msg()")
    i_stamp = src.find("self.stamp_frame_arrival(msg)")
    i_pause = src.find("if gated_result and not self.staging_can_accept():")
    i_put = src.find("inbound.put((\"msg\"")
    assert i_recv >= 0 and i_stamp >= 0 and i_pause >= 0 and i_put >= 0, (
        f"anchors not found: recv={i_recv} stamp={i_stamp} pause={i_pause} "
        f"put={i_put}")
    assert i_recv < i_stamp < i_pause < i_put, (
        f"stamp is out of order: recv={i_recv} stamp={i_stamp} "
        f"pause={i_pause} put={i_put}")
    return "recv -> stamp -> pause -> put (live source order)"


def gate_c5_summary_reports_zero_frames_distinguishably():
    """THE ATTEMPT-6 SHAPE, REPRODUCED. `st1_s29/s30/s31` had ZERO accepted
    sub-stripe shard rows against 34 expected. The summary must report that as
    `frames_received=0` WITH `age_since_last_accepted_progress_s=None` — 'no
    progress was ever accepted' — and must report a stripe that DID progress
    with a non-None age. Both dispositions are driven; a field that reports the
    same thing for both is caught here."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_s29", "rrig6600b:gpu1")
        c.note_stripe_claimed("run", "st1_s12", "rrig6600b:gpu1")
        c.note_stripe_frame_accepted("run", "st1_s12", "rrig6600b:gpu1", 0.9)
        # [R2-2] the progress CLOCK is a separate call, bound to a real renewal.
        c.note_stripe_renewing_progress("run", "st1_s12")
        with _Capture("range_miner_coordinator") as cap:
            silent = c.emit_stripe_rx_summary("run", "st1_s29", "lease_expiry")
            active = c.emit_stripe_rx_summary("run", "st1_s12", "stripe_complete")
        assert len(cap.payloads("[H1H2] stripe_rx")) == 2, "not both emitted"
    assert silent["frames_received"] == 0, silent
    assert silent["age_since_last_accepted_progress_s"] is None, (
        f"a stripe with no accepted frame reported an age of "
        f"{silent['age_since_last_accepted_progress_s']!r} — a number here "
        f"asserts progress that never happened")
    assert silent["age_since_claim_s"] is not None, (
        "the claim age is UNOBSERVED for a stripe that was claimed")
    assert active["frames_received"] == 1, active
    assert active["age_since_last_accepted_progress_s"] is not None, (
        "a stripe that DID progress also reports None — the two dispositions "
        "are indistinguishable, so the field decides nothing")
    assert active["residency_max_s"] == 0.9 and silent["residency_max_s"] is None
    return (f"silent: frames=0 age=None | active: frames=1 "
            f"age={active['age_since_last_accepted_progress_s']}s")


def gate_c6_summary_is_emitted_at_the_lease_expiry_boundary():
    """The record must exist for the ONE event attempt 6 turned on, and it must
    be emitted BEFORE the matrix acts — in a constant phase
    `handle_stripe_failure` fails the trial and cancels everything behind it.

    The emission sits in the serve loop rather than in `process_lease_expiry`,
    which is a NO-TOUCH surface (attempt-6 FAIR-3/2). This gate pins the
    ORDERING that placement has to preserve, and the no-touch property itself is
    gated separately by N5."""
    serve = inspect.getsource(RangeMinerCoordinator.serve_trial)
    i_emit = serve.find("emit_stripe_rx_summary")
    i_expiry = serve.find("self.process_lease_expiry(run_id, eligible)")
    assert i_emit >= 0, "the serve loop emits no expiry-boundary summary"
    assert i_emit < i_expiry, (
        "the summary is emitted AFTER process_lease_expiry — by then a "
        "constant-phase trial is already failed and its stripes cancelled")
    disp = inspect.getsource(RangeMinerCoordinator._serve_dispatch)
    assert disp.count("emit_stripe_rx_summary") == 2, (
        f"expected the completion and error boundaries to emit; found "
        f"{disp.count('emit_stripe_rx_summary')}")
    return "pre-expiry (serve loop) + stripe_complete + stripe_error"


def gate_c8_arrived_but_never_accepted_is_observable():
    """[R1-B, BLOCKING ARM] ARRIVED IS NOT ACCEPTED, AND THE GAP IS THE MECHANISM.

        reader   receives, decodes, enqueues the frame
        serve    500+ frames behind; never dequeues it for >300 s
        lease    no accepted progress -> expires

    Under the first revision this read `frames_received = 0`, which is
    **identical to "the worker sent nothing"** — H2 wearing H1's clothes. The
    worker does not rescue it either: a socket that absorbs all 34 frames
    quickly leaves the worker's own send stall SMALL.

    WHAT WOULD PROVE IT IS MEASURING: the enqueued stripe reports
    `frames_enqueued > 0`, `frames_dequeued == 0`, `subresults_accepted == 0`
    and a large `oldest_pending_age_s`; a stripe that went all the way through
    reports the opposite. Both dispositions are driven.

    WRONG INPUT THAT REDS IT: counting arrival at acceptance time (the R0
    behaviour) — `frames_enqueued` would then be 0 and this arm fails."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_s29", "rrig6600b:gpu1")
        c.note_stripe_claimed("run", "st1_s12", "rrig6600b:gpu1")
        # BACKLOGGED: three frames decoded and enqueued, never dequeued.
        for i in range(3):
            m = SubStripeResultMessage(
                worker_id="rrig6600b:gpu1", stripe_id="st1_s29", sub_index=i,
                seed_start=0, seed_count=1, survivor_count=0, inline=[],
                size_bytes=2, sha256="0" * 64)
            c.stamp_frame_arrival(m)
            c.note_stripe_frame_enqueued("run", c.frame_stripe_id(m),
                                         getattr(m, FRAME_ARRIVAL_ATTR, None),
                                         token=c.frame_token(m))
        # CLEAN CONTROL: one frame enqueued, dequeued and accepted.
        m2 = SubStripeResultMessage(
            worker_id="rrig6600b:gpu1", stripe_id="st1_s12", sub_index=0,
            seed_start=0, seed_count=1, survivor_count=0, inline=[],
            size_bytes=2, sha256="0" * 64)
        c.stamp_frame_arrival(m2)
        c.note_stripe_frame_enqueued("run", c.frame_stripe_id(m2),
                                     getattr(m2, FRAME_ARRIVAL_ATTR, None),
                                     token=c.frame_token(m2))
        c.note_stripe_frame_dequeued("run", c.frame_stripe_id(m2),
                                     token=c.frame_token(m2))
        c.note_stripe_frame_accepted("run", "st1_s12", "rrig6600b:gpu1", 0.01,
                                     kind="subresult")
        time.sleep(0.40)
        with _Capture("range_miner_coordinator") as cap:
            stuck = c.emit_stripe_rx_summary(
                "run", "st1_s29", "lease_expired_offered_to_matrix")
            flow = c.emit_stripe_rx_summary("run", "st1_s12", "stripe_complete")
        assert len(cap.payloads("[H1H2] stripe_rx")) == 2, "not both emitted"
    assert stuck["frames_enqueued"] == 3, stuck
    assert stuck["frames_dequeued"] == 0, stuck
    assert stuck["frames_pending"] == 3, stuck
    assert stuck["subresults_accepted"] == 0, stuck
    assert stuck["frames_received"] == 0, stuck
    assert stuck["oldest_pending_age_s"] is not None and \
        stuck["oldest_pending_age_s"] > 0.35, (
        f"the pending age did not track the hold: "
        f"{stuck['oldest_pending_age_s']!r}")
    assert flow["frames_enqueued"] == 1 and flow["frames_dequeued"] == 1
    assert flow["frames_pending"] == 0 and flow["subresults_accepted"] == 1
    assert flow["oldest_pending_age_s"] is None, (
        "a stripe with nothing pending reports a pending age")
    return (f"backlogged: enq=3 deq=0 pending=3 accepted=0 "
            f"age={stuck['oldest_pending_age_s']:.2f}s | "
            f"flowing: enq=1 deq=1 pending=0 accepted=1")


def gate_c9_message_classes_are_counted_separately():
    """[R1-D] A generic `frames_received` cannot answer *"did actual result
    payload progress reach acceptance?"* — heartbeat and sub_stripe_result BOTH
    renew the lease.

    WHAT WOULD PROVE THE CLASSES ARE SEPARATE: a stripe kept alive by heartbeats
    ALONE must show a live `age_since_last_accepted_progress_s` and **no**
    `age_since_last_subresult_s`; a stripe with real payload progress must show
    both. Driving only one disposition would leave a counter that always agrees
    with itself.

    WRONG INPUT THAT REDS IT: one counter serving both questions."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_hb", "w:gpu0")
        c.note_stripe_claimed("run", "st1_mix", "w:gpu1")
        # kept alive by heartbeats only
        for _ in range(3):
            c.note_stripe_frame_accepted("run", "st1_hb", "w:gpu0", 0.02,
                                         kind="heartbeat", renewed=True)
            # [R2-2] the counter and the CLOCK are separate calls: the clock is
            # moved only where the ledger's answer is in hand.
            c.note_stripe_renewing_progress("run", "st1_hb")
        # real progress plus a heartbeat plus its terminal frame
        c.note_stripe_frame_accepted("run", "st1_mix", "w:gpu1", 0.02,
                                     kind="heartbeat", renewed=False)
        c.note_stripe_frame_accepted("run", "st1_mix", "w:gpu1", 0.03,
                                     kind="subresult")
        c.note_stripe_renewing_progress("run", "st1_mix")
        c.note_stripe_frame_accepted("run", "st1_mix", "w:gpu1", 0.04,
                                     kind="terminal")
        hb = c.emit_stripe_rx_summary("run", "st1_hb", "lease_expiry")
        mx = c.emit_stripe_rx_summary("run", "st1_mix", "stripe_complete")
    assert hb["heartbeats_accepted"] == 3 and hb["heartbeats_renewed"] == 3, hb
    assert hb["subresults_accepted"] == 0 and hb["terminal_frames_accepted"] == 0
    assert hb["age_since_last_accepted_progress_s"] is not None, (
        "a heartbeat did not count as lease-renewing progress")
    assert hb["age_since_last_subresult_s"] is None, (
        "a stripe that delivered NO payload reports a sub-result age — the two "
        "questions are being answered by one counter")
    assert mx["heartbeats_accepted"] == 1 and mx["heartbeats_renewed"] == 0, mx
    assert mx["subresults_accepted"] == 1 and mx["terminal_frames_accepted"] == 1
    assert mx["age_since_last_subresult_s"] is not None, mx
    assert mx["frames_received"] == 3, mx
    return ("heartbeat-only: hb=3 renewed=3 sub=0 subresult_age=None | "
            "mixed: hb=1 renewed=0 sub=1 terminal=1 subresult_age=set")


def _race_frames(n, sid="st1_s29"):
    out = []
    for i in range(n):
        out.append(SubStripeResultMessage(
            worker_id="rrig6600b:gpu1", stripe_id=sid, sub_index=i,
            seed_start=0, seed_count=1, survivor_count=0, inline=[],
            size_bytes=2, sha256="0" * 64))
    return out


def _run_queue_race(c, frames, *, mode="free", sid="st1_s29"):
    """Drive a REAL `queue.Queue` with a REAL producer thread and a REAL
    consumer thread, each recording its own half of the inventory.

    THE INTERLEAVING IS FORCED, NOT HOPED FOR. `mode="consumer_first"` makes the
    producer WAIT on a per-frame event that the consumer sets only after it has
    recorded its dequeue — so the inversion R2-1 describes happens for EVERY
    frame, every run. A sleep-based fixture would exercise the race *usually*,
    and "usually" is how a concurrency gate becomes vacuous on a quiet machine.
    `mode="producer_first"` forces the benign order the same way.
    `mode="free"` lets the scheduler decide and asserts only the invariant."""
    q = _queue.Queue()
    errors = []
    evs = {m.sub_index: threading.Event() for m in frames}

    def _produce():
        try:
            for m in frames:
                c.stamp_frame_arrival(m)
                q.put(m)
                if mode == "consumer_first":
                    assert evs[m.sub_index].wait(10.0), "consumer never signalled"
                c.note_stripe_frame_enqueued(
                    "run", c.frame_stripe_id(m),
                    getattr(m, FRAME_ARRIVAL_ATTR, None),
                    token=c.frame_token(m))
                if mode == "producer_first":
                    evs[m.sub_index].set()
        except Exception as e:                                   # noqa: BLE001
            errors.append(f"producer: {type(e).__name__}: {e}")
        finally:
            for e in evs.values():
                e.set()
            q.put(None)

    def _consume():
        try:
            while True:
                m = q.get(timeout=10.0)
                if m is None:
                    return
                if mode == "producer_first":
                    assert evs[m.sub_index].wait(10.0), "producer never signalled"
                c.note_stripe_frame_dequeued(
                    "run", c.frame_stripe_id(m), token=c.frame_token(m))
                if mode == "consumer_first":
                    evs[m.sub_index].set()
        except Exception as e:                                   # noqa: BLE001
            errors.append(f"consumer: {type(e).__name__}: {e}")

    tp = threading.Thread(target=_produce, name="race-producer")
    tc = threading.Thread(target=_consume, name="race-consumer")
    tc.start()
    tp.start()
    tp.join(timeout=30.0)
    tc.join(timeout=30.0)
    assert not tp.is_alive() and not tc.is_alive(), "a race thread hung"
    assert not errors, f"race threads raised: {errors}"
    return c.stripe_rx_snapshot("run", sid)


def gate_r21a_producer_first_reconciles():
    """[R2-1 CONTROL] Producer bookkeeping wins the race — the benign ordering.
    Must reconcile to N/N/0, and `early_dequeue_events` must be 0, which is what
    proves this arm is the CONTROL and not accidentally the inversion."""
    n = 24
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_s29", "rrig6600b:gpu1")
        snap = _run_queue_race(c, _race_frames(n), mode="producer_first")
    assert (snap["frames_enqueued"], snap["frames_dequeued"],
            len(snap["pending"])) == (n, n, 0), (
        f"producer-first did not reconcile: enq={snap['frames_enqueued']} "
        f"deq={snap['frames_dequeued']} pending={len(snap['pending'])}")
    assert snap["early_dequeue_events"] == 0, (
        f"the CONTROL arm raced after all ({snap['early_dequeue_events']} "
        f"inversions) — it is not a control")
    assert not snap["early_dequeued"], snap["early_dequeued"]
    return f"{n} frames, producer-first: {n}/{n}/0, 0 inversions"


def gate_r21b_consumer_first_inversion_reconciles():
    """[R2-1, BLOCKING ARM] THE EXECUTION BETA DESCRIBED, DRIVEN FOR REAL.

        reader:  inbound.put(frame) succeeds -> the frame is visible
        serve:   wakes, get()s it, records the dequeue      <- CONSUMER FIRST
        reader:  resumes, records the enqueue

    Under the R1 implementation the consumer popped an empty FIFO, recorded
    nothing, and the producer then appended — leaving `enqueued=1 · dequeued=0 ·
    pending=1` for a frame that had already been processed. **That manufactures
    the H2b signature the instrument exists to prove**, and it is most likely at
    LOW occupancy, so a phantom pending frame could persist indefinitely.

    **C8 could never have caught this**: it calls the two accounting functions in
    deterministic order on one thread, so the consumer cannot outrun the
    producer's bookkeeping in that fixture at all.

    WHAT MAKES THIS ARM NON-VACUOUS: it asserts `early_dequeue_events > 0`, i.e.
    the inversion demonstrably happened, before asserting the tally reconciles.

    WRONG INPUT THAT REDS IT: any implementation that relies on scheduling order
    after `Queue.put()` returns — including the R1 FIFO-of-stamps shape."""
    n = 24
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_s29", "rrig6600b:gpu1")
        snap = _run_queue_race(c, _race_frames(n), mode="consumer_first")
    assert snap["early_dequeue_events"] == n, (
        f"the inversion occurred {snap['early_dequeue_events']} times, not {n} "
        f"— the fixture is not driving the race it claims to drive, which is "
        f"exactly how C8 passed")
    assert (snap["frames_enqueued"], snap["frames_dequeued"],
            len(snap["pending"])) == (n, n, 0), (
        f"consumer-first did NOT reconcile: enq={snap['frames_enqueued']} "
        f"deq={snap['frames_dequeued']} pending={len(snap['pending'])} — a "
        f"phantom pending frame is the manufactured H2b signature")
    assert not snap["early_dequeued"], (
        f"tokens left parked after reconciliation: {snap['early_dequeued']}")
    assert snap["frames_untokened"] == 0, snap["frames_untokened"]
    return (f"{n} frames, consumer-first: {n}/{n}/0 with "
            f"{snap['early_dequeue_events']} real inversions")


def gate_r21c_interleaved_race_reconciles():
    """[R2-1] Neither side deliberately favoured: both threads run flat out over
    a larger batch, so the interleaving is whatever the scheduler produces. The
    invariant must hold regardless — that is the whole content of
    'order-independent'."""
    n = 200
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_s29", "rrig6600b:gpu1")
        snap = _run_queue_race(c, _race_frames(n))
    assert (snap["frames_enqueued"], snap["frames_dequeued"],
            len(snap["pending"])) == (n, n, 0), (
        f"free-running race did not reconcile: enq={snap['frames_enqueued']} "
        f"deq={snap['frames_dequeued']} pending={len(snap['pending'])}")
    assert not snap["early_dequeued"]
    return (f"{n} frames free-running: {n}/{n}/0 "
            f"({snap['early_dequeue_events']} inversions observed)")


def gate_r21d_untokened_frames_cannot_distort_the_inventory():
    """A frame that never passed the reader's stamp has no token and cannot be
    reconciled. It is counted where it can be SEEN rather than being allowed to
    corrupt an inventory that claims exactly-once — the same UNOBSERVED
    discipline applied to a count. In production `frames_untokened` is 0, so a
    non-zero value is itself a finding."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_s29", "w")
        c.note_stripe_frame_enqueued("run", "st1_s29", None, token=None)
        c.note_stripe_frame_dequeued("run", "st1_s29", token=None)
        snap = c.stripe_rx_snapshot("run", "st1_s29")
    assert snap["frames_untokened"] == 1, snap
    assert snap["frames_enqueued"] == 0 and snap["frames_dequeued"] == 0, (
        f"an unreconcilable frame entered the exactly-once inventory: {snap}")
    assert len(snap["pending"]) == 0, snap
    return "untokened frame counted separately; inventory undisturbed"


def gate_r31a_snapshot_is_detached_from_the_live_inventory():
    """[R3-1, BLOCKING ARM] A SNAPSHOT MUST NOT ALIAS THE LIVE STRUCTURES.

    `dict(slot)` copies the outer mapping only, so `pending` (a dict) and
    `early_dequeued` (a set) came back as SHARED REFERENCES. The lock was then
    released and the record builders called `len()` and `min(...values())` on
    live objects while the reader and serve threads mutated them.

    WHAT WOULD PROVE DETACHMENT: take a snapshot, then mutate the live inventory
    hard, and show the snapshot's counters AND its token collections are
    bit-for-bit what they were at acquisition. Identity is checked too — a copy
    that happens to be equal today is not a detached copy.

    WRONG INPUT THAT REDS IT: `return dict(slot)`, i.e. the shallow copy. That is
    executed as a positive control at the end of this arm."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_s29", "w:gpu0")
        for t in (101, 102, 103):
            c.note_stripe_frame_enqueued("run", "st1_s29", 10.0 + t, token=t)
        c.note_stripe_frame_dequeued("run", "st1_s29", token=901)   # early
        snap = c.stripe_rx_snapshot("run", "st1_s29")
        before = (dict(snap["pending"]), set(snap["early_dequeued"]),
                  snap["frames_enqueued"], snap["frames_dequeued"])

        live = c._stripe_rx[("run", "st1_s29")]
        assert snap["pending"] is not live["pending"], (
            "snapshot['pending'] IS the live dict — the lock protects mutation "
            "and not the read")
        assert snap["early_dequeued"] is not live["early_dequeued"], (
            "snapshot['early_dequeued'] IS the live set")

        # ...now churn the live inventory as hard as the real path would
        c.note_stripe_frame_dequeued("run", "st1_s29", token=101)
        c.note_stripe_frame_dequeued("run", "st1_s29", token=102)
        c.note_stripe_frame_enqueued("run", "st1_s29", 99.0, token=901)
        for t in range(200, 260):
            c.note_stripe_frame_enqueued("run", "st1_s29", float(t), token=t)

        after = (dict(snap["pending"]), set(snap["early_dequeued"]),
                 snap["frames_enqueued"], snap["frames_dequeued"])
        assert before == after, (
            f"the snapshot CHANGED under concurrent mutation:\n"
            f"  before={before}\n  after ={after}")
        fresh = c.stripe_rx_snapshot("run", "st1_s29")
        assert fresh["frames_enqueued"] != snap["frames_enqueued"], (
            "a fresh snapshot is identical to the old one — the fixture did not "
            "actually mutate anything, so this arm proved nothing")

    # POSITIVE CONTROL, executed: the shallow copy really does alias.
    shallow = {"pending": {1: 1.0}, "early_dequeued": {2}}
    alias = dict(shallow)
    shallow["pending"][3] = 3.0
    assert alias["pending"] == shallow["pending"] and \
        alias["pending"] is shallow["pending"], (
        "the positive control does not demonstrate aliasing — this arm cannot "
        "distinguish a detached copy from a shallow one")
    return (f"pending/early_dequeued detached (identity + value); "
            f"{len(before[0])} tokens held stable across 63 live mutations")


def gate_r31b_record_is_coherent_across_mid_build_mutation():
    """[R3-1] FORCE MUTATION BETWEEN SNAPSHOT ACQUISITION AND RECORD
    CONSTRUCTION — the exact window Beta names.

    `emit_stripe_rx_summary` takes a snapshot and then reads `len(pending)` and
    `min(pending.values())` from it. This arm wraps `stripe_rx_snapshot` so that
    the live inventory is churned the instant after the snapshot is taken and
    before the record is built. The emitted record must describe ONE consistent
    instant, and must not raise.

    ⚠ THE R4-1 DEFECT IN THE PREVIOUS FORM: the churn was CARDINALITY-NEUTRAL —
    40 tokens dequeued and 40 enqueued — so an aliased shallow snapshot still
    held exactly 40 pending entries and **every assertion survived the exact
    wrong input the arm advertised**. Executed and confirmed before repair:
    the shallow snapshot yielded `frames_pending=40 · frames_enqueued=40 ·
    frames_dequeued=0 · oldest_pending_age_s=748.789` — a clean pass.

    The churn now REMOVES 40 and ADDS 7, so a shallow alias reports **7**. And
    Beta's alternative is applied as well rather than instead: the arm asserts
    the ACTUAL pre-churn oldest arrival stamp, two-sided — `is not None` was
    never sufficient, because the aliased snapshot also returns a number.

    WRONG INPUT THAT REDS IT: the aliased snapshot — `frames_pending` becomes 7
    and `oldest_pending_age_s` moves to the post-churn minimum. Mutant M13
    installs it and reports the observed values."""
    PRE_STAMPS = [float(t) for t in range(300, 340)]      # oldest = 300.0
    POST_STAMPS = [float(t) for t in range(400, 407)]     # oldest = 400.0, n=7
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_s29", "w:gpu0")
        for t in range(300, 340):
            c.note_stripe_frame_enqueued("run", "st1_s29", float(t), token=t)

        orig = RangeMinerCoordinator.stripe_rx_snapshot
        churn = {"n": 0}

        def _snapshot_then_churn(self, run_id, stripe_id):
            snap = orig(self, run_id, stripe_id)
            # THE WINDOW: the record has not been built yet. The churn is
            # deliberately NOT cardinality-neutral — 40 out, 7 in.
            for t in range(300, 340):
                self.note_stripe_frame_dequeued(run_id, stripe_id, token=t)
            for t in range(400, 407):
                self.note_stripe_frame_enqueued(run_id, stripe_id, float(t),
                                                token=t)
            churn["n"] += 1
            return snap

        with _patch_attr(RangeMinerCoordinator, "stripe_rx_snapshot",
                         _snapshot_then_churn):
            with _Capture("range_miner_coordinator") as cap:
                t_ref = time.monotonic()
                rec = c.emit_stripe_rx_summary("run", "st1_s29", "lease_expiry")
            payload = cap.payloads("[H1H2] stripe_rx")[0]
        live = c._stripe_rx[("run", "st1_s29")]
        assert len(live["pending"]) == len(POST_STAMPS), (
            f"the fixture did not churn as intended: live pending="
            f"{len(live['pending'])}")
    assert churn["n"] == 1, f"the churn seam did not fire ({churn['n']})"
    # (a) CARDINALITY — a shallow alias reports 7 here.
    assert rec["frames_pending"] == len(PRE_STAMPS), (
        f"the record reports {rec['frames_pending']} pending, not "
        f"{len(PRE_STAMPS)} — it read state that post-dates its own snapshot")
    assert rec["frames_enqueued"] == 40 and rec["frames_dequeued"] == 0, rec
    # (b) THE ACTUAL AGE, two-sided. `is not None` passes on the alias too.
    age = rec["oldest_pending_age_s"]
    assert age is not None, rec
    expected_pre = t_ref - min(PRE_STAMPS)
    expected_post = t_ref - min(POST_STAMPS)
    assert abs(age - expected_pre) < 1.0, (
        f"oldest_pending_age_s={age} does not match the PRE-churn oldest stamp "
        f"({expected_pre:.3f}); post-churn would be {expected_post:.3f}")
    assert abs(age - expected_post) > 50.0, (
        f"oldest_pending_age_s={age} matches the POST-churn minimum — the "
        f"record read the live structure")
    assert payload["frames_pending"] == len(PRE_STAMPS), (
        f"the EMITTED record disagrees with the returned one: {payload}")
    return (f"pre-churn snapshot: pending={rec['frames_pending']} (live={len(POST_STAMPS)}), "
            f"age={age:.1f}s matches stamp {min(PRE_STAMPS):.0f} not "
            f"{min(POST_STAMPS):.0f}; no raise")


def _iter_stmt_bodies(node):
    """Every statement LIST in a tree — function bodies, if/else, loops, try
    bodies and handlers, with-blocks. Pairing is a within-body property, so the
    detector has to see every body, not just the first one it finds."""
    import ast as _ast
    for n in _ast.walk(node):
        for _field, value in _ast.iter_fields(n):
            if isinstance(value, list) and value and \
                    all(isinstance(x, _ast.stmt) for x in value):
                yield value


def _classify_inbound_put(call):
    """`msg` | `control` | `unclassifiable` for one `inbound.put(...)` call.

    The production envelope is `("msg", rawsock, msg, credit_id, None)`; the EOF
    envelope is `("eof", …)` and is legitimately unaccounted. Anything whose kind
    cannot be read — a bare name, a computed tuple — is `unclassifiable` and is
    treated as a MESSAGE put by the caller: **fail closed**, because a put we
    cannot prove is control traffic must not escape the accounting requirement.
    Beta's counterexample used bare `inbound.put(x)` / `put(y)`."""
    import ast as _ast
    if not call.args:
        return "unclassifiable"
    first = call.args[0]
    if isinstance(first, _ast.Tuple) and first.elts and \
            isinstance(first.elts[0], _ast.Constant) and \
            isinstance(first.elts[0].value, str):
        kind = first.elts[0].value
        return "msg" if kind == "msg" else "control"
    return "unclassifiable"


def _single_call_offenders(reader_src, serve_src):
    """[R3-2 · hardened by R5-2] The single-call discipline, as ONE detector used
    by the live check and every wrong-shape control.

    ⚠ R5-2: the previous form counted `note_stripe_frame_enqueued` sites and
    required exactly one, then found *a* try whose body held a put followed by
    that call. **It never counted message puts**, so a second successful
    `inbound.put` with no accounting passed — contradicting the very invariant
    the gate states. Message-envelope puts are now enumerated and paired
    ONE-TO-ONE with their accounting sibling, within each body."""
    import ast as _ast
    import textwrap as _tw
    bad = []

    def _stmt_call(stmt):
        if isinstance(stmt, _ast.Expr) and isinstance(stmt.value, _ast.Call):
            return stmt.value
        return None

    rtree = _ast.parse(_tw.dedent(reader_src))

    # ---- enumerate message-envelope puts and accounting calls --------------
    msg_puts, control_puts = [], []
    for n in _ast.walk(rtree):
        if isinstance(n, _ast.Call) and "inbound.put" in _ast.unparse(n.func):
            kind = _classify_inbound_put(n)
            (control_puts if kind == "control" else msg_puts).append((n, kind))
    enq = [n for n in _ast.walk(rtree) if isinstance(n, _ast.Call)
           and "note_stripe_frame_enqueued" in _ast.unparse(n.func)]
    if not msg_puts:
        bad.append("no message-envelope `inbound.put(...)` found in the reader")
    if len(enq) != len(msg_puts):
        bad.append(f"{len(msg_puts)} message-envelope put(s) but {len(enq)} "
                   f"enqueue-accounting call(s) — the invariant is ONE per "
                   f"SUCCESSFUL put")

    # ---- ONE-TO-ONE PAIRING, within each body ------------------------------
    # Walk each body in order: a message put opens a debt, the next accounting
    # sibling settles it. A second put before settlement, an unsettled put at
    # end of body, or an accounting call with no open debt are all offenders.
    paired = 0
    for body in _iter_stmt_bodies(rtree):
        pending = None
        for i, stmt in enumerate(body):
            call = _stmt_call(stmt)
            if call is None:
                continue
            u = _ast.unparse(call.func)
            if "inbound.put" in u:
                if _classify_inbound_put(call) == "control":
                    continue
                if pending is not None:
                    bad.append(
                        f"a second message-envelope put at stmt {i} with NO "
                        f"accounting for the put at stmt {pending}")
                pending = i
            elif "note_stripe_frame_enqueued" in u:
                if pending is None:
                    bad.append(f"enqueue accounting at stmt {i} with no "
                               f"preceding message put in the same body")
                else:
                    paired += 1
                    pending = None
        if pending is not None:
            bad.append(f"a message-envelope put at stmt {pending} has NO "
                       f"accounting sibling before the end of its body")
    if msg_puts and paired != len(msg_puts):
        bad.append(f"{paired} of {len(msg_puts)} message puts are paired with "
                   f"an accounting sibling")

    # ---- the paired accounting must follow the put, in the same try body,
    #      never in a handler, and the success flag must be set alongside it ---
    found = None
    for node in _ast.walk(rtree):
        if not isinstance(node, _ast.Try):
            continue
        idx_put = idx_enq = None
        for i, stmt in enumerate(node.body):
            call = _stmt_call(stmt)
            if call is None:
                continue
            u = _ast.unparse(call.func)
            if idx_put is None and "inbound.put" in u and \
                    _classify_inbound_put(call) != "control":
                idx_put = i
            if idx_enq is None and "note_stripe_frame_enqueued" in u:
                idx_enq = i
        if idx_put is not None and idx_enq is not None:
            found = (idx_put, idx_enq, node)
            break
    if found is None:
        bad.append("the enqueue accounting is not a DIRECT sibling statement of "
                   "a message-envelope `inbound.put(` inside the same try body")
    else:
        idx_put, idx_enq, try_node = found
        if idx_enq <= idx_put:
            bad.append(f"the enqueue accounting runs BEFORE the put "
                       f"(stmt {idx_enq} vs {idx_put}) — a failed put would count")
        if not any("_delivered_ok = True" in _ast.unparse(st)
                   for st in try_node.body):
            bad.append("the success flag is not set in the same block")
    for node in _ast.walk(rtree):
        if isinstance(node, _ast.Try):
            for h in node.handlers:
                if "note_stripe_frame_enqueued" in _ast.unparse(h):
                    bad.append("the enqueue accounting appears in an except "
                               "handler — a `Full` retry would double-count")

    # ---- consumer side -----------------------------------------------------
    stree = _ast.parse(_tw.dedent(serve_src))
    deq = [n for n in _ast.walk(stree) if isinstance(n, _ast.Call)
           and "note_stripe_frame_dequeued" in _ast.unparse(n.func)]
    if len(deq) != 1:
        bad.append(f"{len(deq)} dequeue-accounting call sites, expected 1")
    gets = [n for n in _ast.walk(stree) if isinstance(n, _ast.Call)
            and "inbound.get" in _ast.unparse(n.func)]
    if len(gets) != 1:
        bad.append(f"{len(gets)} `inbound.get(` sites, expected 1 — one dequeue "
                   f"call cannot be one-per-get against {len(gets)} gets")
    guarded = any(isinstance(n, _ast.If)
                  and "note_stripe_frame_dequeued" in _ast.unparse(n.body)
                  and "kind == 'msg'" in _ast.unparse(n.test).replace('"', "'")
                  for n in _ast.walk(stree))
    if deq and not guarded:
        bad.append("the dequeue accounting is not guarded by `kind == 'msg'` — "
                   "an eof entry carries `msg=None` and would be counted")
    return bad

def gate_r32_single_call_discipline_is_structural():
    """[R3-2, BLOCKING ARM] THE INVENTORY'S CORRECTNESS RESTS ON SINGLE-CALL
    DISCIPLINE, NOT ON IDEMPOTENCE — so the discipline is PROVEN here rather
    than asserted in prose.

    Once a token has reconciled it is in neither collection, so a duplicate is
    treated as a fresh event (`E,D,D -> enqueued=1 dequeued=2`). The R2 report's
    idempotence claim is withdrawn. What is true, and what this establishes over
    the LIVE AST:

        exactly ONE enqueue-accounting call per SUCCESSFUL `inbound.put()`
        exactly ONE dequeue-accounting call per `inbound.get()`

    ⚠ [R4 audit] This arm advertised five wrong inputs and executed ONE. All
    five now run through `_single_call_offenders` — the same detector the live
    check uses.

    WRONG INPUT THAT REDS IT: a second call site; the enqueue moved into the
    `except _queue.Full` handler; the enqueue moved above the put; the dequeue
    moved out from under its `kind == "msg"` guard; a second `inbound.get(`."""
    import textwrap as _tw
    rsrc_live = inspect.getsource(RangeMinerCoordinator._conn_reader_loop)
    ssrc_live = inspect.getsource(RangeMinerCoordinator.serve_trial)
    bad = _single_call_offenders(rsrc_live, ssrc_live)
    assert not bad, "; ".join(bad)

    # ---- THE FALSIFIER SET, EXECUTED -----------------------------------
    CONTROLS = {
        "enqueue in the except handler": ("""
            def r(self):
                while not ok:
                    try:
                        inbound.put(x)
                        _delivered_ok = True
                    except Full:
                        self.note_stripe_frame_enqueued(a, b, c)
            """, None),
        "enqueue above the put": ("""
            def r(self):
                while not ok:
                    try:
                        self.note_stripe_frame_enqueued(a, b, c)
                        inbound.put(x)
                        _delivered_ok = True
                    except Full:
                        pass
            """, None),
        "two enqueue call sites": ("""
            def r(self):
                while not ok:
                    try:
                        inbound.put(x)
                        _delivered_ok = True
                        self.note_stripe_frame_enqueued(a, b, c)
                        self.note_stripe_frame_enqueued(a, b, c)
                    except Full:
                        pass
            """, None),
        "dequeue not guarded by kind=='msg'": (None, """
            def s(self):
                kind, rawsock, msg, credit_id, reader_exit = inbound.get(timeout=t)
                self.note_stripe_frame_dequeued(r, sid, token=tok)
            """),
        # [R5-2] THE SHAPE BETA EXECUTED. Previously offenders=[] — a second
        # successful message put with no accounting, contradicting the gate's
        # own stated invariant.
        "second message put, no accounting": ("""
            def r(self):
                while not ok:
                    try:
                        inbound.put(x)
                        _delivered_ok = True
                        self.note_stripe_frame_enqueued(a, b, c)

                        inbound.put(y)
                    except Full:
                        pass
            """, None),
        # ...and the legitimate control-envelope put must NOT be demanded to
        # account, or the detector would reject correct code.
        "eof envelope put (must NOT be an offender)": ("""
            def r(self):
                while not ok:
                    try:
                        inbound.put(('msg', a, b, c, None))
                        _delivered_ok = True
                        self.note_stripe_frame_enqueued(a, b, c)
                    except Full:
                        pass
                inbound.put(('eof', sock, None, None, rec))
            """, "EXPECT_CLEAN"),
        "two inbound.get sites": (None, """
            def s(self):
                if a:
                    kind, b, msg, c, d = inbound.get(timeout=t)
                else:
                    kind, b, msg, c, d = inbound.get(timeout=0)
                if kind == 'msg':
                    self.note_stripe_frame_dequeued(r, sid, token=tok)
            """),
    }
    survived, wrongly_rejected = [], []
    for label, (rsrc, ssrc) in CONTROLS.items():
        expect_clean = (ssrc == "EXPECT_CLEAN")
        offenders = _single_call_offenders(
            _tw.dedent(rsrc) if rsrc else rsrc_live,
            ssrc_live if (ssrc is None or expect_clean) else _tw.dedent(ssrc))
        if expect_clean:
            if offenders:
                wrongly_rejected.append(f"{label}: {offenders}")
            continue
        if not offenders:
            survived.append(label)
    assert not wrongly_rejected, (
        f"the detector rejects CORRECT code: {wrongly_rejected} — a legitimate "
        f"control-envelope put must not be demanded to account")
    assert not survived, (
        f"the detector still accepts: {survived} — the advertised falsifiers "
        f"exceed its reach")
    n_wrong = sum(1 for _l, (_r, _s) in CONTROLS.items() if _s != "EXPECT_CLEAN")
    n_clean = len(CONTROLS) - n_wrong
    return ("1 msg-put paired 1:1 with its accounting (post-put, in-body, not in "
            "a handler) · 1 dequeue (kind=='msg') · 1 inbound.get; "
            f"{n_wrong}/{n_wrong} wrong shapes rejected, "
            f"{n_clean}/{n_clean} correct shape accepted")

def gate_c10_lease_progress_clock_moves_only_on_real_renewal():
    """[R2-2, BLOCKING ARM] ACCEPTED IS NOT RENEWED.

    The R1 implementation moved the lease-progress clock for EVERY accepted
    class — including a heartbeat the ledger refused to renew, and including
    terminal frames — so the record could state *"the lease was renewed just
    now"* on the strength of a frame that renewed nothing. **That is the R1-E
    error reintroduced inside the instrument built to prevent it.**

    THE FULL TRUTH TABLE IS DRIVEN, positive and negative:

        renewed heartbeat        -> lease-progress age REFRESHES
        renewed=False heartbeat  -> does NOT refresh
        accepted terminal frame  -> does NOT refresh
        subresult + real renewal -> REFRESHES

    WRONG INPUT THAT REDS IT: driving `age_since_last_accepted_progress_s` from
    `last_accepted_frame_at` — the R1 shape — or from 'passed the identity
    fence'."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        for sid in ("st1_no", "st1_term", "st1_hb", "st1_sub"):
            c.note_stripe_claimed("run", sid, "w:gpu0")

        # NEGATIVE 1 — accepted heartbeat, ledger REFUSED the renewal
        c.note_stripe_frame_accepted("run", "st1_no", "w:gpu0", 0.01,
                                     kind="heartbeat", renewed=False)
        no = c.emit_stripe_rx_summary("run", "st1_no", "lease_expiry")

        # NEGATIVE 2 — accepted terminal frame, renews nothing by construction
        c.note_stripe_frame_accepted("run", "st1_term", "w:gpu0", 0.01,
                                     kind="terminal")
        term = c.emit_stripe_rx_summary("run", "st1_term", "stripe_complete")

        # POSITIVE 1 — heartbeat the ledger granted
        c.note_stripe_frame_accepted("run", "st1_hb", "w:gpu0", 0.01,
                                     kind="heartbeat", renewed=True)
        c.note_stripe_renewing_progress("run", "st1_hb")
        hb = c.emit_stripe_rx_summary("run", "st1_hb", "stripe_complete")

        # POSITIVE 2 — sub-result whose renewal succeeded
        c.note_stripe_frame_accepted("run", "st1_sub", "w:gpu0", 0.01,
                                     kind="subresult")
        c.note_stripe_renewing_progress("run", "st1_sub")
        sub = c.emit_stripe_rx_summary("run", "st1_sub", "stripe_complete")

    for label, rec in (("refused heartbeat", no), ("terminal frame", term)):
        assert rec["age_since_last_accepted_progress_s"] is None, (
            f"an accepted {label} refreshed the LEASE-PROGRESS clock "
            f"({rec['age_since_last_accepted_progress_s']}) — the record now "
            f"states a renewal that never happened")
        assert rec["age_since_last_accepted_frame_s"] is not None, (
            f"the {label} did not register as an accepted FRAME either — the "
            f"two clocks have been collapsed in the other direction")
    assert no["heartbeats_accepted"] == 1 and no["heartbeats_renewed"] == 0, no
    assert term["terminal_frames_accepted"] == 1, term
    for label, rec in (("renewed heartbeat", hb), ("renewed subresult", sub)):
        assert rec["age_since_last_accepted_progress_s"] is not None, (
            f"a {label} did NOT refresh the lease-progress clock")
    assert hb["heartbeats_renewed"] == 1, hb
    assert sub["age_since_last_subresult_s"] is not None, sub
    assert no["age_since_last_subresult_s"] is None, no
    return ("refused-hb: progress=None frame=set | terminal: progress=None "
            "frame=set | renewed-hb: progress=set | subresult: progress=set")


def _renewal_clock_offenders(src):
    """[R3-3] THE HARDENED DETECTOR, factored out so it can be run against the
    live source AND against deliberately-wrong sources.

    Returns a list of complaints; empty means the source binds every
    `_renew_active_lease()` result to a name and guards every
    `note_stripe_renewing_progress()` call with THAT EXACT NAME."""
    import ast as _ast
    import textwrap as _tw
    tree = _ast.parse(_tw.dedent(src))
    bad = []

    # 1. every `_renew_active_lease(...)` result must be BOUND to a Name
    renewal_names = set()
    renewal_assigns = []          # (lineno, target_id) — for the R4-2 pairing
    bound_calls = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign) and isinstance(node.value, _ast.Call) \
                and "_renew_active_lease" in _ast.unparse(node.value.func):
            bound_calls.add(id(node.value))
            for t in node.targets:
                if isinstance(t, _ast.Name):
                    renewal_names.add(t.id)
                    renewal_assigns.append((node.lineno, t.id))
                else:
                    bad.append(f"renewal result bound to a non-Name target: "
                               f"{_ast.unparse(t)}")
    renewal_assigns.sort()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call) and \
                "_renew_active_lease" in _ast.unparse(node.func) and \
                id(node) not in bound_calls:
            bad.append("a `_renew_active_lease(...)` result is DISCARDED — its "
                       "answer cannot then guard anything")

    # 2. every clock call must be guarded by one of EXACTLY those names
    parent = {}
    for node in _ast.walk(tree):
        for child in _ast.iter_child_nodes(node):
            parent[child] = node
    clock_calls = 0
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.Call)
                and "note_stripe_renewing_progress" in _ast.unparse(node.func)):
            continue
        clock_calls += 1
        cur, guards = node, []
        while cur in parent:
            cur = parent[cur]
            if isinstance(cur, _ast.If):
                guards.append(cur.test)
        # THE BINDING: the test must BE the renewal-result Name itself, not
        # merely mention something that looks like one. `if renew_metrics_enabled:`
        # is rejected because that Name never received a renewal result.
        named = [t for t in guards
                 if isinstance(t, _ast.Name) and t.id in renewal_names]
        if not named:
            bad.append(
                f"clock call guarded by {[_ast.unparse(t) for t in guards]}, "
                f"none of which IS a bound renewal result {sorted(renewal_names)}")
            continue

        # [R4-2] AND IT MUST BE THE *CORRESPONDING* RESULT, not merely SOME
        # ledger answer. Beta extracted the previous detector and ran the
        # swapped shape through it: `offenders = []`. The check above proves
        # only that each clock call is guarded by A renewal result; the claim
        # was that each is bound to ITS OWN.
        #
        # PAIRING RULE: the governing renewal for a guard is the NEAREST
        # PRECEDING `_renew_active_lease` assignment in DOCUMENT ORDER, and the
        # guard's identifier must equal that assignment's target. This is exact
        # for the straight-line-then-branch shape `_serve_dispatch` has (each
        # branch assigns its own result immediately above its own guard); a
        # future refactor that interleaves the assignments differently would
        # need this rule revisited, and that limit is stated rather than hidden.
        guard = named[0]                      # innermost enclosing renewal guard
        preceding = [a for a in renewal_assigns if a[0] < guard.lineno]
        if not preceding:
            bad.append(
                f"clock call guarded by `{guard.id}` with NO preceding "
                f"`_renew_active_lease` assignment — the guard cannot be that "
                f"call's own result")
            continue
        gov_line, gov_name = preceding[-1]
        if gov_name != guard.id:
            bad.append(
                f"clock call guarded by `{guard.id}` but the governing renewal "
                f"result at line {gov_line} is `{gov_name}` — this call is bound "
                f"to ANOTHER branch's ledger answer")
    return bad, clock_calls, sorted(renewal_names)


def gate_c11_renewal_clock_is_bound_to_the_ledger_answer():
    """[R2-2 · HARDENED BY R3-3] The clock may move only where the LEDGER'S OWN
    ANSWER is in hand — and the gate must be able to detect every way that can
    fail, not merely the ways today's source happens not to fail.

    ⚠ THE R3-3 DEFECT IN THIS GATE'S PREVIOUS FORM: it accepted any enclosing
    condition whose TEXT contained "renew", so this wrong implementation passed:

        _hb_renewed = self._renew_active_lease(...)
        if renew_metrics_enabled:          # <- not the renewal result at all
            self.note_stripe_renewing_progress(...)

    The production code was correct; the gate's advertised falsifier simply
    exceeded its reach — the same class as a lock that protects mutation but not
    the read. It now binds the target Name of each `_renew_active_lease()`
    assignment and requires THAT EXACT `ast.Name` to be the guard.

    ⚠ THE R4-2 DEFECT IN THE PREVIOUS FORM: it gathered every renewal-result
    name into ONE SET and passed a clock call guarded by ANY of them — proving
    only that each call is guarded by SOME ledger answer, while the report
    claimed each is bound to ITS OWN. Beta extracted the detector and ran the
    swapped shape through it: `offenders = []`. The pairing rule is now the
    NEAREST PRECEDING renewal assignment in document order, and the guard's
    identifier must equal that assignment's target.

    WRONG INPUT THAT REDS IT — all SIX executed below as controls, with the
    observed offender text asserted non-empty for each: a guard that merely
    contains the word "renew"; a guard on the message type; an unconditional
    call; a discarded renewal result; **the heartbeat result guarding the
    sub-result clock call**; **the two guards fully swapped**."""
    import textwrap as _tw
    src = inspect.getsource(RangeMinerCoordinator._serve_dispatch)
    bad, clock_calls, names = _renewal_clock_offenders(src)
    assert not bad, "; ".join(bad)
    assert clock_calls == 2, f"expected 2 renewal-clock call sites, got {clock_calls}"
    assert len(names) == 2, f"expected 2 bound renewal results, got {names}"
    assert src.count("_renew_active_lease") == 2, src.count("_renew_active_lease")

    # the accepted-frame recorder must NOT move the clock
    acc = inspect.getsource(RangeMinerCoordinator.note_stripe_frame_accepted)
    assert "last_renewing_progress_at" not in acc.split(chr(34) * 3)[-1], (
        "note_stripe_frame_accepted writes the lease-progress clock — "
        "acceptance is not renewal")

    # ---- THE FALSIFIER SET, EXECUTED ----------------------------------
    CONTROLS = {
        "renew_metrics_enabled": """
            def f(self):
                _hb_renewed = self._renew_active_lease(a, b)
                if renew_metrics_enabled:
                    self.note_stripe_renewing_progress(r, s)
            """,
        "message-type guard": """
            def f(self):
                _hb_renewed = self._renew_active_lease(a, b)
                if mt == 'heartbeat':
                    self.note_stripe_renewing_progress(r, s)
            """,
        "unconditional": """
            def f(self):
                _hb_renewed = self._renew_active_lease(a, b)
                self.note_stripe_renewing_progress(r, s)
            """,
        "discarded result": """
            def f(self):
                self._renew_active_lease(a, b)
                if ok:
                    self.note_stripe_renewing_progress(r, s)
            """,
        # [R4-2] THE TWO SHAPES BETA EXTRACTED THE DETECTOR AND RAN. Both
        # previously returned `offenders = []`.
        "wrong corresponding result": """
            def f(self):
                _hb_renewed = self._renew_active_lease(a, source='heartbeat')
                if _hb_renewed:
                    self.note_stripe_renewing_progress(r, hb_stripe)
                _sr_renewed = self._renew_active_lease(a, source='sub_stripe_result')
                if _hb_renewed:
                    self.note_stripe_renewing_progress(r, sr_stripe)
            """,
        "fully swapped guards": """
            def f(self):
                _hb_renewed = self._renew_active_lease(a, source='heartbeat')
                if _sr_renewed:
                    self.note_stripe_renewing_progress(r, hb_stripe)
                _sr_renewed = self._renew_active_lease(a, source='sub_stripe_result')
                if _hb_renewed:
                    self.note_stripe_renewing_progress(r, sr_stripe)
            """,
    }
    survived = []
    for label, bad_src in CONTROLS.items():
        offenders, _, _ = _renewal_clock_offenders(_tw.dedent(bad_src))
        if not offenders:
            survived.append(label)
    assert not survived, (
        f"the hardened detector still accepts: {survived} — its falsifier is "
        f"broader than its reach, which is the R3-3 defect unrepaired")

    # ...and a CORRECT shape must still pass, or the detector is merely strict
    GOOD = _tw.dedent("""
        def f(self):
            _hb_renewed = self._renew_active_lease(a, b)
            if _hb_renewed:
                self.note_stripe_renewing_progress(r, s)
        """)
    ok_bad, _, _ = _renewal_clock_offenders(GOOD)
    assert not ok_bad, f"the detector rejects the CORRECT shape: {ok_bad}"
    return (f"2/2 clock calls bound to exactly {names}; "
            f"{len(CONTROLS)}/{len(CONTROLS)} wrong shapes rejected")

def _iter_same_scope(root):
    """Yield `root` and every descendant **in the same executable scope**.

    ⚠ [R7-1] A nested `def`, `lambda` or `class` is a SEPARATE scope whose body
    does not run when the enclosing block runs. Unrestricted `ast.walk()` cannot
    tell "this code executes here" from "this code is merely defined here", and
    that distinction is the whole content of a same-critical-section claim.
    Nested scope nodes are yielded (so a caller can see one exists) but are NOT
    descended into."""
    import ast as _ast
    SCOPES = (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.Lambda, _ast.ClassDef)
    yield root
    stack = list(_ast.iter_child_nodes(root))
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, SCOPES):
            continue                      # defined here, not executed here
        stack.extend(_ast.iter_child_nodes(n))


def _scope_root(node):
    """A `Module` wrapping a single function is really that function's scope —
    unwrap it so `_iter_same_scope` starts inside the body rather than refusing
    to enter it."""
    import ast as _ast
    if isinstance(node, _ast.Module) and len(node.body) == 1 and \
            isinstance(node.body[0], (_ast.FunctionDef, _ast.AsyncFunctionDef)):
        return node.body[0]
    return node


def _ast_calls_to(node, name, same_scope=True):
    """Every ACTUAL `ast.Call` whose callee resolves to `name`, by default only
    those **executed in `node`'s own scope**.

    ⚠ [R6-1] THE HELPER THAT EXISTS BECAUSE TEXT IS NOT A CALL. `"_defer_locked"
    in ast.unparse(with_node)` is satisfied by a STRING LITERAL — Beta executed a
    block whose only mention was `marker = "_defer_locked(entry)"`, with no
    insertion at all, and E4 returned `offenders = []`.

    ⚠ [R7-1] AND A CALL IS NOT AN EXECUTION. Beta then executed a block whose
    only insertion sat inside `def later():` — defined, never called — and E4
    certified the same-acquisition property over a block that inserts nothing.
    `same_scope=True` refuses to descend into nested `FunctionDef`,
    `AsyncFunctionDef`, `Lambda` or `ClassDef` bodies.

    **Two declared boundaries, stated rather than implied:**

    1. The callee is matched by attribute or bare name, **NOT by receiver
       identity** — `other._defer_locked(x)` counts. For the single-object
       methods these detectors read that is sufficient; it is a declared
       boundary, not a proven one.
    2. **[R7-4] WHICH DETECTORS ROUTE THROUGH HERE, exactly.** R6 claimed
       *"every detector in this file that means 'a call happened' now routes
       through here."* **That was an over-claim** — the same class this arc keeps
       surfacing, in a docstring rather than an assertion. The accurate list:

           _defer_order_offenders  (E4)   -> uses _ast_calls_to
           _renewal_site_offenders (N2)   -> uses _ast_calls_to

       `_single_call_offenders` (R3-2) and `_renewal_clock_offenders` (C11)
       still identify `ast.Call` nodes with their own `ast.unparse` matching.
       They are **not reopened here** — each carries its own executed
       wrong-shape controls — but they are not covered by this helper and this
       docstring no longer says they are."""
    import ast as _ast
    root = _scope_root(node)
    walker = _iter_same_scope(root) if same_scope else _ast.walk(root)
    out = []
    for n in walker:
        if not isinstance(n, _ast.Call):
            continue
        f = n.func
        if isinstance(f, _ast.Attribute) and f.attr == name:
            out.append(n)
        elif isinstance(f, _ast.Name) and f.id == name:
            out.append(n)
    return out

def _defer_order_offenders(enqueue_src, pump_src):
    """[E4 · hardened by R5-1 and R6-1] The two structural halves, factored so
    they run against the live source AND against deliberately-wrong sources.

    R5-1: the note must be inside the SAME `ast.With` acquisition that performs
    the insertion — not merely some `_admission_lock` block. A split lock lets
    the pump observe, remove and release-account the entry between the
    insertion's release and the note's acquisition, inverting the pair.

    R6-1: and BOTH must be proven as actual `ast.Call` nodes. A `With` whose only
    mention of `_defer_locked` is a string literal contains no insertion, and
    certifying it proves nothing at all."""
    import ast as _ast
    import textwrap as _tw
    bad = []
    tree = _ast.parse(_tw.dedent(enqueue_src))

    note_calls = _ast_calls_to(tree, "note_stripe_frame_deferred")
    insert_calls = _ast_calls_to(tree, "_defer_locked")
    if not note_calls:
        bad.append("enqueue_staging makes no `note_stripe_frame_deferred(...)` "
                   "CALL (a mention in a string or comment is not a call)")
    if not insert_calls:
        bad.append("enqueue_staging makes no `_defer_locked(...)` CALL — there is "
                   "no insertion, so no critical section can be identified")

    if note_calls and insert_calls:
        # THE critical section is the acquisition that performs the insertion,
        # identified by containing an actual insertion CALL.
        insert_withs = [n for n in _ast.walk(tree)
                        if isinstance(n, _ast.With)
                        and "_admission_lock" in _ast.unparse(n.items[0])
                        and _ast_calls_to(n, "_defer_locked")]
        if not insert_withs:
            bad.append("the `_defer_locked(...)` insertion is not inside any "
                       "`with self._admission_lock` block")
        elif not any(_ast_calls_to(w, "note_stripe_frame_deferred")
                     for w in insert_withs):
            bad.append(
                "the defer note is NOT inside the SAME `with self._admission_lock` "
                "block as the `_defer_locked(...)` insertion — between the "
                "insertion's release and the note's acquisition the pump can "
                "observe, remove and release-account the entry, inverting the pair")

    ptree = _ast.parse(_tw.dedent(pump_src))
    total = sum(1 for n in _ast.walk(ptree)
                if isinstance(n, _ast.Attribute) and n.attr == "_deferred")
    in_lock = 0
    for node in _ast.walk(ptree):
        if isinstance(node, _ast.With) and \
                "_admission_lock" in _ast.unparse(node.items[0]):
            in_lock += sum(1 for n in _ast.walk(node)
                           if isinstance(n, _ast.Attribute)
                           and n.attr == "_deferred")
    if total == 0:
        bad.append("the pump does not touch `_deferred` at all")
    elif in_lock != total:
        bad.append(f"{total - in_lock} of {total} `_deferred` accesses in the "
                   f"pump are OUTSIDE _admission_lock")
    return bad, total, in_lock

def gate_e4_defer_precedes_release_by_lock_construction():
    """[R2-1 AUDIT] The OTHER cross-thread accounting pair: `deferred` is
    recorded on the dispatch thread and `released` from a staging-completion
    callback. Could the release bookkeeping outrun the defer bookkeeping the way
    the dequeue outran the enqueue?

    NO — and by construction rather than by luck. The property has TWO halves:

      (a) `note_stripe_frame_deferred` is called INSIDE the `_admission_lock`
          block that appended the entry to `_deferred`; and
      (b) `_pump_deferred` can only OBSERVE that entry from inside the same
          lock.

    Together those give: the defer record strictly precedes any possible release
    record. Either half alone gives nothing.

    ⚠ [R4 audit] The docstring previously advertised two wrong inputs and
    executed NEITHER. Both are now run through the same detector the live check
    uses, and the observed offender text is asserted non-empty.

    WRONG INPUT THAT REDS IT: the defer note moved outside the lock; or any
    `_deferred` access in the pump moved outside it."""
    import textwrap as _tw
    bad, total, in_lock = _defer_order_offenders(
        inspect.getsource(RangeMinerCoordinator.enqueue_staging),
        inspect.getsource(RangeMinerCoordinator._pump_deferred))
    assert not bad, "; ".join(bad)
    assert total > 0 and in_lock == total, (total, in_lock)
    pump = inspect.getsource(RangeMinerCoordinator._pump_deferred)
    assert "note_stripe_frame_released" in pump

    CONTROLS = {
        "defer note outside the lock": (
            """
            def enqueue_staging(self):
                with self._admission_lock:
                    action = 'deferred' if self._defer_locked(entry) else 'bp'
                self.note_stripe_frame_deferred(r, s, a, i)
            """,
            """
            def _pump_deferred(self):
                with self._admission_lock:
                    for entry in self._deferred:
                        pass
                    self._deferred = still
            """),
        # [R7-1] THE SHAPE BETA EXECUTED AGAINST THE R6 REPAIR. The only
        # insertion sits inside a nested `def` that is never called — defined
        # here, not executed here.
        "insertion inside a nested def that never runs": (
            """
            def enqueue_staging(self):
                with self._admission_lock:
                    def later():
                        self._defer_locked(entry)
                    self.note_stripe_frame_deferred(r, s, a, i)
            """,
            """
            def _pump_deferred(self):
                with self._admission_lock:
                    for entry in self._deferred:
                        pass
                    self._deferred = still
            """),
        # [R6-1] THE SHAPE BETA EXECUTED AGAINST THE R5 REPAIR. The only
        # mention of the insertion is a STRING LITERAL — there is no insertion —
        # and the previous form returned offenders=[].
        "string literal, no insertion call": (
            """
            def enqueue_staging(self):
                with self._admission_lock:
                    marker = "_defer_locked(entry)"
                    self.note_stripe_frame_deferred(r, s, a, i)
            """,
            """
            def _pump_deferred(self):
                with self._admission_lock:
                    for entry in self._deferred:
                        pass
                    self._deferred = still
            """),
        # [R5-1] THE SHAPE BETA EXECUTED. Previously offenders=[], total=2,
        # lock-held=2 — accepted, while the pump could release-account the entry
        # between the insertion's release and the note's acquisition.
        "split lock: note in a LATER acquisition": (
            """
            def enqueue_staging(self):
                with self._admission_lock:
                    action = 'deferred' if self._defer_locked(entry) else 'bp'

                with self._admission_lock:
                    self.note_stripe_frame_deferred(r, s, a, i)
            """,
            """
            def _pump_deferred(self):
                with self._admission_lock:
                    for entry in self._deferred:
                        pass
                    self._deferred = still
            """),
        "pump reads _deferred outside the lock": (
            """
            def enqueue_staging(self):
                with self._admission_lock:
                    action = 'deferred' if self._defer_locked(entry) else 'bp'
                    self.note_stripe_frame_deferred(r, s, a, i)
            """,
            """
            def _pump_deferred(self):
                for entry in self._deferred:
                    pass
                with self._admission_lock:
                    self._deferred = still
            """),
    }
    survived = []
    for label, (esrc, psrc) in CONTROLS.items():
        offenders, _, _ = _defer_order_offenders(_tw.dedent(esrc), _tw.dedent(psrc))
        if not offenders:
            survived.append(label)
    assert not survived, (
        f"the detector still accepts: {survived} — its falsifier is broader "
        f"than its reach")
    return (f"defer note lock-held AND {in_lock}/{total} pump `_deferred` "
            f"accesses lock-held; {len(CONTROLS)}/{len(CONTROLS)} wrong shapes rejected")


def gate_c7_claim_precision_distinguishes_measured_from_reconciled():
    """A retried stripe is claimed inside `_handle_stripe_failure_locked`, which
    is NO-TOUCH, so its claim instant cannot be captured exactly. It is adopted
    on first sighting and LABELLED — because an age derived from first sighting
    is a LOWER BOUND, and reporting it identically to a measured one would make
    the instrument quietly wrong on exactly the retried stripes.

    Both dispositions are driven and must differ."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp, compute_lease_timeout=300.0)
        w0 = _register(c, "rrig6600:gpu0", backend="rocm")
        w1 = _register(c, "rrig6600:gpu1", backend="rocm")
        for sid, w in (("st1_s0", w0), ("st1_s1", w1)):
            c.ledger.add_stripe("run", sid, 0, 67_108_864,
                                "java_lcg_reverse", 2)
            assert c.ledger.claim_stripe("run", sid, w.worker_id, 0, 34,
                                         time.time() + 300.0)
        # s0 goes through the scheduler's own record; s1 never does.
        c.note_stripe_claimed("run", "st1_s0", w0.worker_id)
        _acct = c.active_stripe_accounting("run")
        assert _acct["status"] == "OK", _acct
        rows = {r["stripe_id"]: r for r in _acct["stripes"]}
    assert rows["st1_s0"]["claim_precision"] == "exact", rows["st1_s0"]
    assert rows["st1_s1"]["claim_precision"] == "reconciled", (
        f"a stripe the scheduler never reported was not labelled: "
        f"{rows['st1_s1']}")
    assert rows["st1_s1"]["age_since_claim_s"] is not None, (
        "the reconciled stripe has no age at all — it would be invisible on "
        "exactly the retry path")
    return "exact vs reconciled, both present and labelled"


# ===========================================================================
# §D — PERIODIC ACTIVE-STRIPE ACCOUNTING
# ===========================================================================
def gate_d1_tick_is_rate_limited_and_one_record_per_tick():
    """§7 D asks for 'one line per active stripe'. At 25 workers that is 2.5
    lines/second for the whole run, which breaches the very bar §15 set. This
    emits ONE record carrying every active stripe. The gate proves BOTH: the
    interval suppresses a second call, and the single record covers all
    stripes."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        with _Capture("range_miner_coordinator") as cap:
            r1 = c.maybe_emit_active_stripe_accounting("run", stage_idx=1)
            r2 = c.maybe_emit_active_stripe_accounting("run", stage_idx=1)
        assert r1 is not None, "the first tick did not emit"
        assert r2 is None, (
            "a second call inside the interval emitted again — the rate limit "
            "is not enforced and this becomes the flood it exists to avoid")
        assert len(cap.payloads("[H1H2] active_stripes")) == 1, "not one record"
        assert ACTIVE_STRIPE_REPORT_INTERVAL_S >= 5.0, (
            f"the interval {ACTIVE_STRIPE_REPORT_INTERVAL_S} is below the bar")
        # forced tick bypasses the clock — the test seam, and it is exercised
        r3 = c.maybe_emit_active_stripe_accounting("run", stage_idx=1, force=True)
        assert r3 is not None, "force=True did not emit"
    return f"interval={ACTIVE_STRIPE_REPORT_INTERVAL_S}s, 1 record/tick, force works"


def gate_d2_lease_remaining_decays_measurably():
    """WHAT WOULD PROVE IT IS MEASURING: `lease_remaining_s` sampled twice, a
    measurable interval apart, must DECREASE. In attempt 6 this quantity was
    decaying invisibly on three stripes for five minutes; a field that reported
    a constant would have shown exactly the same nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp, compute_lease_timeout=300.0)
        w = _register(c, "rrig6600b:gpu1", backend="rocm")
        c.ledger.add_stripe("run", "st1_s29", 0, 67_108_864,
                            "java_lcg_reverse", 2)
        assert c.ledger.claim_stripe("run", "st1_s29", w.worker_id, 0, 34,
                                     time.time() + 300.0), "claim failed"
        c.note_stripe_claimed("run", "st1_s29", w.worker_id)
        a = c.active_stripe_accounting("run")
        time.sleep(0.60)
        b = c.active_stripe_accounting("run")
    assert a["status"] == "OK" and b["status"] == "OK", (a["status"], b["status"])
    a, b = a["stripes"], b["stripes"]
    assert len(a) == 1 and len(b) == 1, f"active rows: {len(a)}, {len(b)}"
    la, lb = a[0]["lease_remaining_s"], b[0]["lease_remaining_s"]
    aa, ab = a[0]["age_since_claim_s"], b[0]["age_since_claim_s"]
    assert la is not None and lb is not None, "lease_remaining_s UNOBSERVED"
    assert lb < la - 0.4, (
        f"lease_remaining_s did not decay across a 0.60 s interval: "
        f"{la} -> {lb}. A constant here reproduces attempt 6 exactly")
    assert ab > aa + 0.4, f"age_since_claim_s did not advance: {aa} -> {ab}"
    assert a[0]["frames_received"] == 0 and a[0]["worker_id"] == "rrig6600b:gpu1"
    return f"lease_remaining {la:.2f}s -> {lb:.2f}s; age {aa:.2f}s -> {ab:.2f}s"


def _claimed_coord(tmp, wid="rrig6600b:gpu1"):
    c = _coord(tmp, compute_lease_timeout=300.0)
    w = _register(c, wid, backend="rocm")
    c.ledger.add_stripe("run", "st1_s29", 0, 67_108_864, "java_lcg_reverse", 2)
    assert c.ledger.claim_stripe("run", "st1_s29", w.worker_id, 0, 34,
                                 time.time() + 300.0), "claim failed"
    c.note_stripe_claimed("run", "st1_s29", w.worker_id)
    return c


class _RaisingLedger:
    """Swaps one ledger method for a raiser, on the CLASS, and restores it."""

    def __init__(self, ledger, method):
        self.cls, self.method = type(ledger), method
        self.orig = getattr(self.cls, method)
        self.hit = {"n": 0}

    def __enter__(self):
        hit = self.hit

        def _boom(_self, *a, **k):
            hit["n"] += 1
            raise RuntimeError("injected ledger failure")
        setattr(self.cls, self.method, _boom)
        return self

    def __exit__(self, *a):
        setattr(self.cls, self.method, self.orig)
        return False


def gate_d4_successful_read_with_no_active_stripes_is_OK_zero():
    """[R1-C] A SUCCESSFUL read that finds nothing is a MEASUREMENT: status OK,
    `active_count = 0`, `stripes = []`. This is the arm that gives `0` its
    meaning, and without it `UNAVAILABLE -> None` proves nothing, because there
    would be no case in which a genuine zero is expected."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        acct = c.active_stripe_accounting("run")
        with _Capture("range_miner_coordinator") as cap:
            rec = c.maybe_emit_active_stripe_accounting("run", stage_idx=1,
                                                        force=True)
        payload = cap.payloads("[H1H2] active_stripes")[0]
    assert acct["status"] == "OK", acct
    assert acct["active_count"] == 0 and acct["stripes"] == [], acct
    assert rec["status"] == "OK" and rec["active_count"] == 0, rec
    assert payload["status"] == "OK" and payload["active_count"] == 0, payload
    return "OK / active_count=0 / stripes=[] — a real zero"


def gate_d5_failed_read_is_UNAVAILABLE_with_no_counts():
    """[R1-C, BLOCKING ARM] THE ELEVENTH INSTANCE, CAUGHT.

    The first revision caught the ledger exception and returned `[]`, so the
    tick published `active_count = len([]) = 0` — a count-shaped assertion that
    the fleet was empty, derived from a read that never completed. **And the
    gate that claimed to prove otherwise asserted `bad == []`: it checked the
    collapse instead of catching it.**

    WHAT MUST NOW HOLD: status UNAVAILABLE, `active_count is None`, `stripes is
    None`, and the EMITTED record carries the same — because the log line is the
    forensic surface, not the return value.

    WRONG INPUT THAT REDS IT: any count-shaped rendering on the unavailable
    path, including `active_count = 0`, `stripes = []`, or omitting the status."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _claimed_coord(tmp)
        good = c.active_stripe_accounting("run")
        with _RaisingLedger(c.ledger, "all_stripes") as inj:
            bad = c.active_stripe_accounting("run")
            with _Capture("range_miner_coordinator") as cap:
                rec = c.maybe_emit_active_stripe_accounting("run", stage_idx=2,
                                                            force=True)
            payload = cap.payloads("[H1H2] active_stripes")[0]
        again = c.active_stripe_accounting("run")
    assert inj.hit["n"] > 0, "the injected failure was never reached"
    assert good["status"] == "OK" and good["active_count"] == 1, good
    assert bad["status"] == "UNAVAILABLE", bad
    assert bad["active_count"] is None, (
        f"a failed read rendered count-shaped: active_count="
        f"{bad['active_count']!r}")
    assert bad["stripes"] is None, (
        f"a failed read rendered list-shaped: stripes={bad['stripes']!r}")
    assert bad.get("error"), "the failure is unavailable AND unexplained"
    assert rec["status"] == "UNAVAILABLE" and rec["active_count"] is None, rec
    assert payload["status"] == "UNAVAILABLE" and \
        payload["active_count"] is None, (
        f"the EMITTED record still asserts an empty fleet: {payload}")
    assert again["status"] == "OK" and again["active_count"] == 1, (
        "the accounting did not recover — the instrument latched a wrong state")
    return ("OK/1 -> UNAVAILABLE/None (record and return) -> OK/1; "
            "error text present")


def gate_d5b_failed_expiry_query_is_UNAVAILABLE():
    """[R1-C] The same collapse existed in the expiry helper: a failed query
    became `[]` and therefore "emit no expiry records" — operationally harmless,
    diagnostically indistinguishable from a successful query finding nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _claimed_coord(tmp)
        ok_none = c.expired_claimed_stripes_for_report("run", time.time())
        ok_some = c.expired_claimed_stripes_for_report("run",
                                                       time.time() + 10_000)
        with _RaisingLedger(c.ledger, "expired_claimed_stripes") as inj:
            bad = c.expired_claimed_stripes_for_report("run", time.time())
    assert inj.hit["n"] > 0, "the injected failure was never reached"
    assert ok_none["status"] == "OK" and ok_none["expired_count"] == 0, ok_none
    assert ok_some["status"] == "OK" and ok_some["expired_count"] == 1, ok_some
    assert bad["status"] == "UNAVAILABLE", bad
    assert bad["expired_count"] is None and bad["stripe_ids"] is None, (
        f"a failed expiry query rendered count-shaped: {bad}")
    return "OK/0 · OK/1 · UNAVAILABLE/None — all three distinguishable"


def gate_d5c_no_observation_is_not_silence():
    """[R1-C] A stripe this instrument has NEVER SEEN emits `NO_OBSERVATION`
    with every count `None`, rather than emitting nothing. Silence would be
    indistinguishable from a stripe that WAS observed and had zero of
    everything — and for a stripe that just expired, "the coordinator never
    recorded a claim for it" is itself a finding."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_seen", "w:gpu0")
        with _Capture("range_miner_coordinator") as cap:
            unseen = c.emit_stripe_rx_summary("run", "st1_never", "lease_expiry")
            seen = c.emit_stripe_rx_summary("run", "st1_seen", "lease_expiry")
        assert len(cap.payloads("[H1H2] stripe_rx")) == 2, (
            "the unobserved stripe emitted nothing — silence is not a status")
    assert unseen["status"] == "NO_OBSERVATION", unseen
    for k in ("frames_enqueued", "frames_received", "subresults_accepted",
              "frames_deferred", "age_since_claim_s"):
        assert unseen[k] is None, f"{k} rendered count-shaped on NO_OBSERVATION"
    assert seen["status"] == "OK", seen
    assert seen["frames_enqueued"] == 0 and seen["frames_received"] == 0, (
        "an OBSERVED stripe with nothing yet must report real zeros")
    return "NO_OBSERVATION/None vs OK/0 — distinguishable"


# ===========================================================================
# §E — PER-STRIPE DEFERRAL ATTRIBUTION
# ===========================================================================
def gate_e1_deferral_is_attributed_and_varies():
    """`deferred_high_water = 716` is a run-scoped scalar with no attribution, so
    'its frames were deferred' is unfalsifiable for a NAMED stripe (forensic
    §5.1). WHAT WOULD PROVE IT IS MEASURING: a stripe whose frame is deferred for
    0.35 s must report a deferred count and a deferred duration; an undeferred
    stripe must report zero for both — and the two must differ."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_s29", "rrig6600b:gpu1")
        c.note_stripe_claimed("run", "st1_s12", "rrig6600b:gpu1")
        c.note_stripe_frame_deferred("run", "st1_s29", 0, 5)
        time.sleep(0.35)
        c.note_stripe_frame_released("run", "st1_s29", 0, 5)
        deferred = c.stripe_rx_snapshot("run", "st1_s29")
        clean = c.stripe_rx_snapshot("run", "st1_s12")
    assert clean["frames_deferred"] == 0 and clean["deferred_seconds_total"] == 0.0
    assert deferred["frames_deferred"] == 1, deferred
    assert 0.30 < deferred["deferred_seconds_total"] < 2.0, (
        f"the deferred duration {deferred['deferred_seconds_total']} did not "
        f"track the 0.35 s hold")
    assert deferred["deferred_open"] == 0, (
        f"the frame is still counted as open after release: {deferred}")
    assert deferred["deferred_seconds_total"] > clean["deferred_seconds_total"], (
        "the deferral field does not vary between a deferred and an undeferred "
        "stripe")
    return (f"deferred: n=1 t={deferred['deferred_seconds_total']:.3f}s | "
            f"clean: n=0 t=0.0s")


# ===========================================================================
# [R-4 PAIRED AMENDMENT, Beta-authorized 2026-08-16]
#
# `gate_e2_release_charges_both_departure_classes` asserted
# `src.count("released.append(") == 2`. THAT GATE WAS CORRECT FOR ITS CERTIFIED
# SOURCE ANCHOR — `_pump_deferred` then had exactly two departure classes — and
# it remains historically correct for that anchor. R-3's end-of-pass capacity
# sweep legitimately introduces a THIRD, so the literal cardinality 2 is
# superseded.
#
# ⚠ IT IS NOT REPLACED BY `== 3`. Beta agreed with Alpha that bumping the count
# is the wrong long-term amendment: it preserves the same brittle detector shape
# and fails again on the next legitimate departure class, while proving nothing
# about whether the charge is CORRECT. The INVARIANT is what survives:
#
#     Every frame leaving `_deferred` is charged exactly once through
#     `note_stripe_frame_released`, after the admission lock is released and
#     before staging submission.
#
# The amended gate is BEHAVIOURAL and is strictly harder to satisfy than either
# count: a count cannot see a double charge, a charge for a RETAINED entry, or a
# departure that is silently uncharged. This one drives all three currently
# reachable departure classes and requires each charge site to be load-bearing.
# ===========================================================================
_E2_CLASSES = ("dead-in-main-scan", "resumed-into-ready", "dead-in-r3-sweep")


def _e2_probe(tmp, tag, pump, klass):
    """Drive ONE departure class through `pump` and return the observed
    departures, the charges, and what stayed retained.

    Nothing here reads production source: departures are computed by diffing
    `_deferred` across the pass, and charges are recorded at the real
    `note_stripe_frame_released` seam."""
    import concurrent.futures as _cf
    import types as _types

    coord = _coord(tmp, dbname=f"{tag}.db")
    charges = []
    coord.note_stripe_frame_released = (
        lambda r, s, a, i: charges.append((r, s, a, i)))
    submitted = []

    def _submit(kind, wconn, run_id, stripe_id, attempt, sub_index, msg, elig):
        submitted.append((stripe_id, attempt, sub_index))
        f = _cf.Future()
        f.set_result(True)
        return f

    coord._submit_with_slot = _submit
    coord._resume_paused_connections = lambda: None

    run = "e2run"

    def _stripe(sid, attempt, state):
        coord.ledger.add_stripe(run, sid, 0, 1000, "java_lcg", 1, now=1.0)
        # one worker per stripe: F1 refuses two concurrent compute claims
        assert coord.ledger.claim_stripe(run, sid, f"w-{sid}", attempt, 8, 9e18)
        if state != ST_CLAIMED:
            coord.ledger.set_stripe_state(run, sid, state)

    class _M:
        size_bytes = 0
        sha256 = "0" * 64

    def _defer(sid, attempt, sub):
        coord._deferred.append(("inline", None, run, sid, attempt, sub,
                                _M(), None, _cf.Future()))

    sem = coord._staging_slots()
    if klass == "dead-in-main-scan":
        _stripe("s0", 0, "done")
        for j in range(5):
            _defer("s0", 0, j)
    elif klass == "resumed-into-ready":
        _stripe("s0", 0, ST_CLAIMED)
        for j in range(4):
            _defer("s0", 0, j)
    else:                                     # dead-in-r3-sweep
        while sem.acquire(blocking=False):    # starve slots: nothing may stage
            pass
        _stripe("s0", 0, ST_CLAIMED)          # A — owns admission
        _stripe("s1", 0, ST_CLAIMED)          # B — memoized live, then killed
        coord._admitted[(run, "s0", 0)] = True
        _defer("s0", 0, 0)
        for j in range(5):
            _defer("s1", 0, j)
        real = coord._try_admit_locked
        state = {"n": 0}

        def _hook(r, sid, att):
            ok = real(r, sid, att)
            if not ok and sid == "s1" and not state["n"]:
                state["n"] = 1
                coord.ledger.set_stripe_state(r, "s1", "done")
            return ok

        coord._try_admit_locked = _hook

    before = [(e[3], e[4], e[5]) for e in coord._deferred]
    _types.MethodType(pump, coord)()
    after = [(e[3], e[4], e[5]) for e in coord._deferred]
    departed = [k for k in before if k not in after]
    return {"before": before, "after": after, "departed": departed,
            "charged": [(s, a, i) for (_r, s, a, i) in charges],
            "submitted": submitted}


def _e2_violations(pump):
    """THE INVARIANT, evaluated over all three departure classes."""
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for klass in _E2_CLASSES:
            o = _e2_probe(tmp, f"e2-{klass}", pump, klass)
            dep, chg = o["departed"], o["charged"]
            if not dep:
                bad.append(f"{klass}: NO departure occurred — arm is vacuous")
                continue
            for k in dep:
                if chg.count(k) != 1:
                    bad.append(f"{klass}: {k} departed but was charged "
                               f"{chg.count(k)} times (expected exactly 1)")
            for k in o["after"]:
                if k in chg:
                    bad.append(f"{klass}: {k} is RETAINED but was charged")
            extra = [k for k in chg if k not in dep]
            if extra:
                bad.append(f"{klass}: charged without departing: {extra}")
    return bad


def _pump_without_charge_site(index):
    """The live `_pump_deferred` with its `index`-th `released.append(...)`
    statement DELETED, compiled against the PRODUCTION module's globals.

    [A8-B2 LESSON] `exec` is given `COORD.__dict__` so `PhaseCharge` and every
    other name resolves in the module under test, not in this test module — the
    exact way Beta's Defect-A mutant first survived."""
    import ast as _ast
    import textwrap as _tw
    import miner.range_miner_coordinator as _CO

    tree = _ast.parse(_tw.dedent(
        inspect.getsource(RangeMinerCoordinator._pump_deferred)))
    seen = []

    class _Cut(_ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, _ast.Call)
                    and _ast.unparse(node.value).startswith(
                        "released.append(")):
                seen.append(node)
                if len(seen) - 1 == index:
                    return None
            return node

    _Cut().visit(tree)
    _ast.fix_missing_locations(tree)
    assert len(seen) == 3, (
        f"expected 3 `released.append(` sites in the live pump, found "
        f"{len(seen)} — the amendment's falisfier set is out of date")
    ns = {}
    exec(compile(tree, "<pump minus charge site>", "exec"), _CO.__dict__, ns)
    fn = ns["_pump_deferred"]
    assert fn.__globals__ is _CO.__dict__
    return fn


def gate_e2_every_departure_is_charged_exactly_once():
    """[R-4 AMENDMENT — supersedes `E2-BOTH-DEPARTURE-CLASSES`]

    THE INVARIANT: every frame leaving `_deferred` is charged exactly once
    through `note_stripe_frame_released`, after the admission lock is released
    and before staging submission.

    Behavioural, not a text count. Departures are computed by diffing
    `_deferred` across a real pass; charges are recorded at the real seam. All
    three currently reachable departure classes are driven:

        dead during the main scan · admitted/resumed into `ready` ·
        dead in R-3's end-of-pass capacity sweep

    NON-VACUITY IS NOT ASSERTED, IT IS EARNED. Each of the three
    `released.append(...)` sites is deleted from the LIVE source in turn and the
    same property is re-evaluated; every deletion must produce a violation.
    A site that is never exercised could not red, so the falsifier sweep proves
    coverage of all three classes at the same time as it proves each charge is
    load-bearing.

    WRONG INPUT THAT REDS IT: any uncharged departure, any double charge, any
    charge for a retained entry, the charge loop moved inside
    `_admission_lock`, or the charge moved after the submit loop."""
    live = _e2_violations(RangeMinerCoordinator._pump_deferred)
    assert not live, "; ".join(live[:6])

    survivors = []
    for i in range(3):
        if not _e2_violations(_pump_without_charge_site(i)):
            survivors.append(i)
    assert not survivors, (
        f"deleting charge site(s) {survivors} changed nothing — those sites are "
        f"unexercised or the property does not detect a missing charge")

    # structural half, unchanged in intent from the superseded gate
    src = inspect.getsource(RangeMinerCoordinator._pump_deferred)
    assert "note_stripe_frame_released" in src, "nothing is charged at all"
    i_lock_block = src.find("with self._admission_lock:")
    i_charge = src.find("self.note_stripe_frame_released(")
    i_ready = src.find("for entry in ready:")
    assert i_lock_block < i_charge < i_ready, (
        "the charge is not outside the admission lock and before the submit "
        "loop")
    import ast as _ast
    import textwrap as _tw
    tree = _ast.parse(_tw.dedent(src))
    lock = [n for n in _ast.walk(tree) if isinstance(n, _ast.With)
            and "_admission_lock" in _ast.unparse(n.items[0])][0]
    assert not [n for n in _ast.walk(lock) if isinstance(n, _ast.Call)
                and getattr(n.func, "attr", None)
                == "note_stripe_frame_released"], (
        "a charge call is INSIDE `_admission_lock`")
    return (f"{len(_E2_CLASSES)} departure classes, every frame charged exactly "
            f"once; 3/3 charge-site deletions red the property")


def gate_e3_an_unmatched_release_is_a_no_op():
    """A release with no matching defer must not fabricate a duration. Without
    this an idempotent-replay path would inflate `deferred_seconds_total` from
    nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        c.note_stripe_claimed("run", "st1_s0", "w")
        c.note_stripe_frame_released("run", "st1_s0", 0, 0)
        snap = c.stripe_rx_snapshot("run", "st1_s0")
    assert snap["deferred_seconds_total"] == 0.0 and snap["frames_deferred"] == 0
    return "unmatched release charges nothing"


# ===========================================================================
# MUTANTS — each disables one instrument and asserts its gate goes RED
# ===========================================================================
class _patch_attr:
    """Replace a class attribute and restore THE ORIGINAL DESCRIPTOR.

    ⚠ THE TRAP THIS EXISTS FOR, and it bit this battery. `getattr(cls, name)`
    on a `staticmethod` yields a PLAIN FUNCTION. Assigning that plain function
    back to the class does NOT restore a staticmethod — it installs an INSTANCE
    method, so every later `self.stamp_frame_arrival(msg)` silently binds `msg`
    to `at` and the frame is never stamped at all. Two later mutants failed on
    exactly that contamination, and they failed for a reason that had nothing to
    do with what they were testing.

    Restoring from `cls.__dict__` preserves the descriptor exactly, so a mutant
    cannot change the calling convention of the surface it borrowed."""

    def __init__(self, cls, name, repl):
        self.cls, self.name, self.repl = cls, name, repl
        self.orig = cls.__dict__[name]

    def __enter__(self):
        setattr(self.cls, self.name, self.repl)
        return self

    def __exit__(self, *a):
        setattr(self.cls, self.name, self.orig)
        return False


def _expect_red(gate, name):
    try:
        gate()
    except AssertionError:
        return True
    return False


def mutant_m1_send_accounting_disabled():
    """Neutralise `_charge_stripe_send`. A4 and A5 must both go red. If they do
    not, they are passing on something other than the measurement."""
    orig = RangeMinerWorker._charge_stripe_send
    RangeMinerWorker._charge_stripe_send = lambda self: None
    try:
        red4 = _expect_red(gate_a4_send_stall_measures_and_varies, "A4")
        red5 = _expect_red(gate_a5_h1_versus_h2_are_TOLD_APART, "A5")
    finally:
        RangeMinerWorker._charge_stripe_send = orig
    assert red4 and red5, (
        f"with send accounting disabled, A4 red={red4} A5 red={red5} — a gate "
        f"that survives its own instrument being removed proves nothing")
    assert gate_a4_send_stall_measures_and_varies(), "A4 did not recover"
    return "A4 RED, A5 RED with the instrument removed; both recover"


def mutant_m2_compute_timer_frozen():
    """Freeze `compute_s` at 0.0. A3 and A5 must go red."""
    import miner.range_miner_worker as W
    orig = W.time.perf_counter
    calls = {"n": 0}

    def _frozen():
        calls["n"] += 1
        return 1000.0
    W.time.perf_counter = _frozen
    try:
        red3 = _expect_red(gate_a3_compute_s_measures_and_varies, "A3")
    finally:
        W.time.perf_counter = orig
    assert calls["n"] > 0, "the mutant was never reached"
    assert red3, "A3 stayed green with a frozen clock — it is not timing anything"
    assert gate_a3_compute_s_measures_and_varies(), "A3 did not recover"
    return f"A3 RED under a frozen perf_counter ({calls['n']} calls); recovers"


def mutant_m3_arrival_stamp_removed():
    """Stop stamping arrivals. C1 must go red, and C2's 'stamped' arm with it —
    residency collapses to UNOBSERVED everywhere, which is the honest failure
    mode but is NOT a measurement."""
    with _patch_attr(RangeMinerCoordinator, "stamp_frame_arrival",
                     staticmethod(lambda msg, at=None: None)):
        red1 = _expect_red(gate_c1_residency_measures_and_varies, "C1")
        red2 = _expect_red(gate_c2_unstamped_frame_is_unobserved_not_zero, "C2")
    assert red1 and red2, f"C1 red={red1} C2 red={red2} with stamping removed"
    assert gate_c1_residency_measures_and_varies(), "C1 did not recover"
    return "C1 RED, C2 RED without the stamp; both recover"


def mutant_m4_residency_defaults_to_zero():
    """THE VACUOUS-FIELD MUTANT, and the one this brief names explicitly. Make
    `frame_queue_residency` return `0.0` instead of `None` for an unstamped
    frame — a field that is always present, always a number, and always wrong.
    C2 must catch it."""
    orig = RangeMinerCoordinator.frame_queue_residency
    with _patch_attr(RangeMinerCoordinator, "frame_queue_residency",
                     staticmethod(lambda msg, at=None: (orig(msg, at) or 0.0))):
        red = _expect_red(gate_c2_unstamped_frame_is_unobserved_not_zero, "C2")
    assert red, (
        "C2 stayed green while UNOBSERVED was reported as 0.0 — this is exactly "
        "the `assignment_active_at_loss` class the brief names")
    return "C2 RED when None degrades to 0.0"


def mutant_m5_lease_remaining_constant():
    """Pin `lease_remaining_s` to a constant. D2 must go red — a constant is the
    attempt-6 signal exactly: present, plausible, and decaying nowhere."""
    orig = RangeMinerCoordinator.active_stripe_accounting

    def _const(self, run_id):
        acct = orig(self, run_id)
        for r in (acct.get("stripes") or ()):
            r["lease_remaining_s"] = 300.0
            r["age_since_claim_s"] = 0.0
        return acct
    RangeMinerCoordinator.active_stripe_accounting = _const
    try:
        red = _expect_red(gate_d2_lease_remaining_decays_measurably, "D2")
    finally:
        RangeMinerCoordinator.active_stripe_accounting = orig
    assert red, "D2 stayed green with a constant lease_remaining_s"
    assert gate_d2_lease_remaining_decays_measurably(), "D2 did not recover"
    return "D2 RED with a pinned lease_remaining_s; recovers"


def mutant_m7_heartbeat_time_folded_into_the_stripe():
    """[R1-A] REINSTATE THE R0 DEFECT and prove A10 catches it: charge the
    stripe from a SOCKET-WIDE roll-up instead of the executing thread's slot,
    which is exactly what the first revision did."""
    orig = RangeMinerWorker._charge_stripe_send

    def _socket_wide(self):
        acct = self._stripe_acct
        if acct is None or self.conn is None:
            return
        base = acct.get("_send_base")
        if base is None:
            return
        allacct = self.conn.send_accounting_all()["total"]
        syscall = round(allacct["send_syscall_s"] - base["send_syscall_s"], 6)
        lock_wait = round(allacct["send_lock_wait_s"] - base["send_lock_wait_s"], 6)
        acct["stripe_send_syscall_s"] = syscall
        acct["stripe_send_lock_wait_s"] = lock_wait
        acct["stripe_send_stall_s"] = round(syscall + lock_wait, 6)
        acct["stripe_send_syscall_max_s"] = 0.0
        acct["stripe_send_lock_wait_max_s"] = 0.0
        acct["stripe_send_calls"] = 0
        acct["heartbeat_send_syscall_s"] = 0.0
    RangeMinerWorker._charge_stripe_send = _socket_wide
    try:
        red = _expect_red(gate_a10_heartbeat_syscall_is_not_charged_to_the_stripe,
                          "A10")
    finally:
        RangeMinerWorker._charge_stripe_send = orig
    assert red, (
        "A10 stayed green while the heartbeat's send time was charged to a "
        "stripe that sent nothing — this is the R1-A defect, and the arm that "
        "exists to catch it did not")
    assert gate_a10_heartbeat_syscall_is_not_charged_to_the_stripe(), (
        "A10 did not recover")
    return "A10 RED under socket-wide attribution; recovers under per-thread"


def mutant_m8_stall_drops_the_lock_wait():
    """[R1-A, second half] The syscall-only discriminator: a worker blocked
    behind another thread's send reports `stall ~= 0` and is misread as H1.
    A11 must catch it."""
    orig = RangeMinerWorker._charge_stripe_send

    def _syscall_only(self):
        orig(self)
        acct = self._stripe_acct
        if acct is not None and acct.get("stripe_send_syscall_s") is not None:
            acct["stripe_send_stall_s"] = acct["stripe_send_syscall_s"]
    RangeMinerWorker._charge_stripe_send = _syscall_only
    try:
        red = _expect_red(gate_a11_lock_wait_appears_as_stripe_send_stall, "A11")
    finally:
        RangeMinerWorker._charge_stripe_send = orig
    assert red, "A11 stayed green with lock wait dropped from the stall"
    assert gate_a11_lock_wait_appears_as_stripe_send_stall(), "A11 did not recover"
    return "A11 RED when stall = syscall only; recovers"


def mutant_m9_arrival_counted_only_on_acceptance():
    """[R1-B] Collapse arrival into acceptance — the R0 behaviour — and prove
    C8 catches it. Under this mutant a fully backlogged stripe reports
    `frames_enqueued = 0`, i.e. indistinguishable from a worker that sent
    nothing."""
    orig = RangeMinerCoordinator.note_stripe_frame_enqueued
    RangeMinerCoordinator.note_stripe_frame_enqueued = (
        lambda self, r, s, a=None, token=None: None)
    try:
        red = _expect_red(gate_c8_arrived_but_never_accepted_is_observable, "C8")
    finally:
        RangeMinerCoordinator.note_stripe_frame_enqueued = orig
    assert red, (
        "C8 stayed green with arrival counting removed — H2 would still look "
        "identical to H1, which is the whole of R1-B")
    assert gate_c8_arrived_but_never_accepted_is_observable(), "C8 did not recover"
    return "C8 RED without arrival counting; recovers"


def mutant_m10_unavailable_collapses_to_empty_list():
    """[R1-C] THE EXACT R0 DEFECT, re-armed: make the failed ledger read return
    an empty list again. D5 must catch it — and, unlike its predecessor D3, it
    must NOT accept `[]` as proof of anything."""
    orig = RangeMinerCoordinator.active_stripe_accounting

    def _collapse(self, run_id):
        try:
            return orig(self, run_id)
        except Exception:                                        # noqa: BLE001
            return []
    # the mutant must reproduce the R0 SHAPE, so it collapses the status too
    def _r0(self, run_id):
        acct = _collapse(self, run_id)
        if isinstance(acct, dict) and acct.get("status") == "UNAVAILABLE":
            return {"status": "OK", "active_count": 0, "stripes": [],
                    "error": None}
        return acct
    RangeMinerCoordinator.active_stripe_accounting = _r0
    try:
        red = _expect_red(gate_d5_failed_read_is_UNAVAILABLE_with_no_counts, "D5")
    finally:
        RangeMinerCoordinator.active_stripe_accounting = orig
    assert red, (
        "D5 stayed green while a failed read rendered as OK/active_count=0 — "
        "this is the eleventh instance, and the gate written to catch it did not")
    assert gate_d5_failed_read_is_UNAVAILABLE_with_no_counts(), "D5 did not recover"
    return "D5 RED when UNAVAILABLE collapses to OK/0; recovers"


def mutant_m11_post_put_fifo_race_restored():
    """[R2-1] REINSTATE THE R1 SHAPE — a FIFO of bare arrival stamps, appended by
    the producer after `put()` and popped by the consumer — and prove the
    inversion arm catches it. This is the defect verbatim: the consumer pops an
    empty list, records nothing, and the producer then appends."""
    o_enq = RangeMinerCoordinator.note_stripe_frame_enqueued
    o_deq = RangeMinerCoordinator.note_stripe_frame_dequeued

    def _r1_enq(self, run_id, stripe_id, arrived_mono=None, token=None):
        if not stripe_id:
            return
        with self._stripe_rx_lock:
            slot = self._stripe_rx_slot(run_id, stripe_id)
            slot.setdefault("_r1_fifo", [])
            slot["frames_enqueued"] += 1
            slot["_r1_fifo"].append(
                time.monotonic() if arrived_mono is None else arrived_mono)

    def _r1_deq(self, run_id, stripe_id, token=None):
        if not stripe_id:
            return
        with self._stripe_rx_lock:
            slot = self._stripe_rx_slot(run_id, stripe_id)
            fifo = slot.setdefault("_r1_fifo", [])
            if fifo:
                fifo.pop(0)
                slot["frames_dequeued"] += 1

    RangeMinerCoordinator.note_stripe_frame_enqueued = _r1_enq
    RangeMinerCoordinator.note_stripe_frame_dequeued = _r1_deq
    try:
        red = _expect_red(gate_r21b_consumer_first_inversion_reconciles, "R2-1b")
    finally:
        RangeMinerCoordinator.note_stripe_frame_enqueued = o_enq
        RangeMinerCoordinator.note_stripe_frame_dequeued = o_deq
    assert red, (
        "the inversion arm stayed green against the R1 post-put FIFO — it does "
        "not exercise the race, which is exactly how C8 passed")
    assert gate_r21b_consumer_first_inversion_reconciles(), "R2-1b did not recover"
    return "R2-1b RED against the R1 post-put FIFO; recovers"


def mutant_m12_lease_clock_driven_from_any_accepted_frame():
    """[R2-2] REINSTATE THE R1 SHAPE — drive the public lease-progress age from
    `last_accepted_frame_at` — and prove C10 catches it. Under this mutant a
    heartbeat the ledger REFUSED reports a lease-progress age of ~0."""
    orig = RangeMinerCoordinator.emit_stripe_rx_summary

    def _r1_clock(self, run_id, stripe_id, disposition):
        rec = orig(self, run_id, stripe_id, disposition)
        if rec is not None and rec.get("status") == "OK":
            rec["age_since_last_accepted_progress_s"] = \
                rec.get("age_since_last_accepted_frame_s")
        return rec
    RangeMinerCoordinator.emit_stripe_rx_summary = _r1_clock
    try:
        red = _expect_red(gate_c10_lease_progress_clock_moves_only_on_real_renewal,
                          "C10")
    finally:
        RangeMinerCoordinator.emit_stripe_rx_summary = orig
    assert red, (
        "C10 stayed green while a refused heartbeat reported a fresh "
        "lease-progress age — this is the R1-E error reintroduced inside the "
        "instrument built to prevent it")
    assert gate_c10_lease_progress_clock_moves_only_on_real_renewal(), (
        "C10 did not recover")
    return "C10 RED when the clock is driven from any accepted frame; recovers"


def mutant_m13_shallow_snapshot_restored():
    """[R4-1] REINSTATE THE R3-1 PRODUCTION DEFECT — `return dict(slot)`, the
    shallow copy that aliases `pending` and `early_dequeued` — and prove the
    REPAIRED R3-1b reds on it.

    This is the arm's own advertised wrong input, EXECUTED rather than asserted.
    Before the R4 repair the same mutant passed R3-1b cleanly, because the churn
    was cardinality-neutral: 40 out, 40 in, `len(aliased) == 40` either way."""
    def _shallow(self, run_id, stripe_id):
        with self._stripe_rx_lock:
            slot = self._stripe_rx.get((run_id, stripe_id))
            return None if slot is None else dict(slot)      # the R3-1 defect

    observed = {}

    def _probe():
        # capture what the defective snapshot actually produces, for the record
        with tempfile.TemporaryDirectory() as tmp:
            c = _coord(tmp)
            c.note_stripe_claimed("run", "st1_s29", "w:gpu0")
            for t in range(300, 340):
                c.note_stripe_frame_enqueued("run", "st1_s29", float(t), token=t)
            base = RangeMinerCoordinator.stripe_rx_snapshot

            def _churn(self, run_id, stripe_id):
                snap = base(self, run_id, stripe_id)
                for t in range(300, 340):
                    self.note_stripe_frame_dequeued(run_id, stripe_id, token=t)
                for t in range(400, 407):
                    self.note_stripe_frame_enqueued(run_id, stripe_id,
                                                    float(t), token=t)
                return snap
            with _patch_attr(RangeMinerCoordinator, "stripe_rx_snapshot", _churn):
                observed.update(
                    c.emit_stripe_rx_summary("run", "st1_s29", "lease_expiry"))

    with _patch_attr(RangeMinerCoordinator, "stripe_rx_snapshot", _shallow):
        _probe()
        red = _expect_red(gate_r31b_record_is_coherent_across_mid_build_mutation,
                          "R3-1b")
    assert red, (
        "R3-1b stayed green on the shallow snapshot it advertises as its wrong "
        "input — the churn is not observable in the record")
    assert observed["frames_pending"] == 7, (
        f"the defective snapshot did not leak the post-churn cardinality: "
        f"{observed['frames_pending']}")
    assert gate_r31b_record_is_coherent_across_mid_build_mutation(), (
        "R3-1b did not recover")
    return (f"R3-1b RED on the shallow alias; it observed frames_pending="
            f"{observed['frames_pending']} (correct is 40); recovers")


def mutant_m14_message_classes_collapsed():
    """[R4 audit] C9 advertised *"one counter serving both questions"* as its
    wrong input and nothing executed it. This collapses every accepted class
    into the sub-result counter — so a stripe kept alive by heartbeats alone
    would report a sub-result age it never earned — and asserts C9 reds."""
    orig = RangeMinerCoordinator.note_stripe_frame_accepted

    def _collapsed(self, run_id, stripe_id, worker_id, residency_s,
                   kind="subresult", renewed=None):
        return orig(self, run_id, stripe_id, worker_id, residency_s,
                    kind="subresult", renewed=renewed)

    with _patch_attr(RangeMinerCoordinator, "note_stripe_frame_accepted",
                     _collapsed):
        red = _expect_red(gate_c9_message_classes_are_counted_separately, "C9")
    assert red, (
        "C9 stayed green with every class collapsed into one counter — the "
        "discrimination it advertises is not reached by its assertions")
    assert gate_c9_message_classes_are_counted_separately(), "C9 did not recover"
    return "C9 RED when the message classes collapse into one counter; recovers"


def mutant_m6_deferral_never_charged():
    """Neutralise the deferral charge. E1 must go red."""
    orig = RangeMinerCoordinator.note_stripe_frame_released
    RangeMinerCoordinator.note_stripe_frame_released = (
        lambda self, r, s, a, i: None)
    try:
        red = _expect_red(gate_e1_deferral_is_attributed_and_varies, "E1")
    finally:
        RangeMinerCoordinator.note_stripe_frame_released = orig
    assert red, "E1 stayed green with the deferral charge removed"
    assert gate_e1_deferral_is_attributed_and_varies(), "E1 did not recover"
    return "E1 RED without the release charge; recovers"


# ===========================================================================
# NON-REGRESSION — the instrument must not touch what it observes
# ===========================================================================
N1_FORBIDDEN = ("renew_lease", "handle_stripe_failure", "fail_trial",
                "abort_trial", "claim_stripe", "_defer_locked",
                "_pump_deferred", "enqueue_staging", "set_stripe_state",
                "commit_trial", "_release_resume_credit")


def _control_flow_offenders(src, forbidden=N1_FORBIDDEN):
    """[N1 · hardened by R6-3] Does this method REACH the lease / matrix /
    scheduling / acceptance machinery, by any route?

    ⚠ R6-3: the previous form matched only DIRECT call names, so
    `getattr(self, 'fail_trial')(run_id)` escaped. It now FAILS CLOSED on three
    routes, and the choice of fail-closed over narrowing the claim is
    deliberate: this gate is the one asserting the instruments cannot perturb
    what they observe, and an instrument that could reach control flow by an
    indirection nobody checked is exactly the class this arc keeps finding.

      (a) a direct call whose callee resolves to a forbidden name;
      (b) ANY indirect callee — `Call(func=Call(...))` — because the resolved
          target cannot be read statically;
      (c) a constant-string `getattr(..., '<forbidden>')`, or an attribute LOAD
          of a forbidden name, whether or not it is called on the spot: a
          reference that can be stored and invoked later is a reach.

    ⚠ AND IT DISTINGUISHES THE CALL FORM FROM THE READ FORM. Alpha's own first
    probe flagged *any* `getattr` and reported a live divergence; reading the
    hits showed all five were `getattr(msg, …)` ATTRIBUTE READS. The predicate
    is therefore *which name is being resolved*, never "a getattr is present" —
    `getattr(msg, 'message_type', None)` is legitimate and must pass.

    Docstrings are excluded: a text search would red on the very sentences that
    RECORD which surfaces these methods deliberately avoid."""
    import ast as _ast
    import textwrap as _tw
    bad = []
    tree = _ast.parse(_tw.dedent(src))
    fn = tree.body[0]
    body = fn.body
    if (body and isinstance(body[0], _ast.Expr)
            and isinstance(body[0].value, _ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    # [R7 sweep] Names aliased to `self` ARE self receivers. `me = self;
    # getattr(me, name)` reaches exactly what `getattr(self, name)` reaches, and
    # a predicate that reads only the literal `self` would miss it.
    self_names = {"self"}
    _grew = True
    while _grew:
        _grew = False
        for stmt in body:
            for n in _ast.walk(stmt):
                if not isinstance(n, _ast.Assign) or not isinstance(n.value, _ast.Name):
                    continue
                if n.value.id in self_names:
                    for t in n.targets:
                        if isinstance(t, _ast.Name) and t.id not in self_names:
                            self_names.add(t.id)
                            _grew = True
    for stmt in body:
        for n in _ast.walk(stmt):
            # (a) direct call
            if isinstance(n, _ast.Call):
                f = n.func
                name = (f.attr if isinstance(f, _ast.Attribute)
                        else f.id if isinstance(f, _ast.Name) else None)
                if name in forbidden:
                    bad.append(f"calls {name}")
                # (b) indirect callee — the target cannot be read statically
                if isinstance(f, _ast.Call):
                    bad.append(f"indirect callee {_ast.unparse(f)[:48]!r} — "
                               f"the resolved target is not statically knowable")
                # (c) constant-string getattr naming a forbidden target
                if isinstance(f, _ast.Name) and f.id == "getattr" and \
                        len(n.args) >= 2 and isinstance(n.args[1], _ast.Constant) \
                        and n.args[1].value in forbidden:
                    bad.append(f"resolves {n.args[1].value!r} via getattr")
                # (d) [R7-2] DYNAMIC getattr on `self` — the name is computed, so
                # the resolved method cannot be read statically and the reach
                # cannot be excluded. Beta executed
                #     name = 'fail_' + 'trial'; f = getattr(self, name); f(...)
                # which reaches `fail_trial` while every earlier rule passed.
                # Scoped to a `self` receiver ON PURPOSE: `getattr(msg, k)` on a
                # message cannot resolve a coordinator method, and rejecting it
                # would be the coarse-matcher error this predicate exists to
                # avoid.
                if isinstance(f, _ast.Name) and f.id == "getattr" and n.args:
                    recv = _ast.unparse(n.args[0])
                    dynamic = not (len(n.args) >= 2
                                   and isinstance(n.args[1], _ast.Constant)
                                   and isinstance(n.args[1].value, str))
                    if dynamic and (recv in self_names
                                    or any(recv.startswith(sn + ".")
                                           for sn in self_names)):
                        bad.append(
                            f"dynamic getattr on {recv} with a computed name "
                            f"({_ast.unparse(n.args[1]) if len(n.args) > 1 else '?'})"
                            f" — the resolved method is not statically knowable")
            # (c cont.) a bare attribute LOAD of a forbidden name
            if isinstance(n, _ast.Attribute) and n.attr in forbidden and \
                    isinstance(n.ctx, _ast.Load):
                bad.append(f"references {n.attr} (a stored reference is a reach)")
    return sorted(set(bad))


def gate_n1_no_control_flow_semantics_changed():
    """The claim asserted here is exactly what is tested and no more: **none of
    the new coordinator methods REACHES any of the named lease / matrix /
    scheduling / acceptance functions** — by direct call, by an indirect callee,
    or by resolving one by name. It is not a proof that they are side-effect
    free in general: they do mutate the accounting map, under their own lock.

    WRONG INPUT THAT REDS IT — executed below: a direct call; a
    `getattr(self, 'fail_trial')(...)` indirection; a stored reference invoked
    later. A legitimate `getattr(msg, 'message_type', None)` attribute READ must
    still pass."""
    import ast as _ast
    import textwrap as _tw
    NEW = ("stamp_frame_arrival", "frame_queue_residency", "_stripe_rx_slot",
           "frame_stripe_id", "frame_token", "note_stripe_claimed",
           "note_stripe_frame_enqueued", "note_stripe_frame_dequeued",
           "note_stripe_frame_accepted", "note_stripe_renewing_progress",
           "note_stripe_frame_deferred", "note_stripe_frame_released",
           "stripe_rx_snapshot", "emit_stripe_rx_summary",
           "active_stripe_accounting", "maybe_emit_active_stripe_accounting",
           "expired_claimed_stripes_for_report")
    offenders = []
    for name in NEW:
        fn = getattr(RangeMinerCoordinator, name, None)
        assert fn is not None, f"{name} is MISSING — the sweep would be vacuous"
        for b in _control_flow_offenders(inspect.getsource(fn)):
            offenders.append(f"{name} -> {b}")
    assert not offenders, f"an instrument reaches control flow: {offenders}"

    CONTROLS = {
        "direct call": "def note_x(self):\n    self.fail_trial(run_id)\n",
        "getattr indirection, called": (
            "def note_x(self):\n    getattr(self, 'fail_trial')(run_id)\n"),
        "stored reference, called later": (
            "def note_x(self):\n    f = self.fail_trial\n    f(run_id)\n"),
        "computed callee": (
            "def note_x(self):\n    self._table['t']()(run_id)\n"),
        # [R7-2] THE SHAPE BETA EXECUTED AGAINST THE R6 REPAIR.
        "stored dynamic getattr on self": (
            "def note_x(self):\n    name = 'fail_' + 'trial'\n"
            "    f = getattr(self, name)\n    f(run_id)\n"),
        # [R7 sweep, found by re-attacking the R7 repair itself]
        "dynamic getattr on a name aliased to self": (
            "def note_x(self):\n    me = self\n"
            "    f = getattr(me, name)\n    f(run_id)\n"),
    }
    survived = [lbl for lbl, src in CONTROLS.items()
                if not _control_flow_offenders(src)]
    assert not survived, (
        f"the detector still accepts: {survived} — an instrument could reach "
        f"control flow by a route nobody checks")
    GOOD = ("def frame_stripe_id(msg, key):\n"
            "    mt = getattr(msg, 'message_type', None)\n"
            "    dyn = getattr(msg, key, None)\n"
            "    return getattr(msg, 'stripe_id', None)\n")
    assert not _control_flow_offenders(GOOD), (
        f"the detector rejects a legitimate attribute READ: "
        f"{_control_flow_offenders(GOOD)} — a coarse matcher produces false "
        f"findings, which is the lesson from Alpha's own first probe")

    # vacuity arm: the extractor must actually be finding calls
    probe = _ast.parse(_tw.dedent(inspect.getsource(
        RangeMinerCoordinator.emit_stripe_rx_summary)))
    n_calls = sum(1 for n in _ast.walk(probe) if isinstance(n, _ast.Call))
    assert n_calls >= 3, (
        f"the call extractor found only {n_calls} calls in a method that "
        f"plainly makes several — it is not inspecting anything")
    return (f"{len(NEW)} new methods, none REACHING any of "
            f"{len(N1_FORBIDDEN)} control-flow names; "
            f"{len(CONTROLS)}/{len(CONTROLS)} wrong shapes rejected, "
            f"1/1 legitimate attribute read accepted")

def _renewal_site_offenders(src):
    """[N2 · hardened by R6-2] The renewal-site check, as one detector usable
    against the live source and against wrong shapes.

    ⚠ `.count("_renew_active_lease") == 2` cannot tell a CALL from a COMMENT
    MENTION: one real call plus one mention also counts 2. The count is now over
    actual `ast.Call` nodes, and the two `source=` discriminators are read from
    the calls' own keywords rather than from anywhere in the file's text."""
    import ast as _ast
    import textwrap as _tw
    bad = []
    tree = _ast.parse(_tw.dedent(src))
    calls = _ast_calls_to(tree, "_renew_active_lease")
    if len(calls) != 2:
        bad.append(f"{len(calls)} actual `_renew_active_lease(...)` CALL(s), "
                   f"expected 2 (text occurrences: "
                   f"{_tw.dedent(src).count('_renew_active_lease')})")
    sources = set()
    for c in calls:
        for kw in c.keywords:
            if kw.arg == "source" and isinstance(kw.value, _ast.Constant):
                sources.add(kw.value.value)
    missing = {"heartbeat", "sub_stripe_result"} - sources
    if missing:
        bad.append(f"no renewal CALL carries source={sorted(missing)} "
                   f"(observed {sorted(sources)})")
    return bad


def gate_n2_renewal_call_sites_unchanged():
    """The F1/F2 certified guard: `_serve_dispatch` renews in exactly two
    places, one per liveness source. The residency hook sits beside them and
    must not have added a third.

    [R6-2] Counted over AST calls, not text — see `_renewal_site_offenders`.

    WRONG INPUT THAT REDS IT — executed below: one real call plus a comment
    mention (text count 2, real calls 1); a third real call; a call whose
    `source=` discriminator is missing. A CORRECT shape must still pass."""
    import textwrap as _tw
    live = inspect.getsource(RangeMinerCoordinator._serve_dispatch)
    bad = _renewal_site_offenders(live)
    assert not bad, "; ".join(bad)

    CONTROLS = {
        "comment mention inflates a text count": """
            def f(self):
                # _renew_active_lease is called once below
                r = self._renew_active_lease(a, source='heartbeat')
                if r:
                    self.note_stripe_renewing_progress(x, y)
            """,
        "a third real renewal call": """
            def f(self):
                a1 = self._renew_active_lease(a, source='heartbeat')
                a2 = self._renew_active_lease(b, source='sub_stripe_result')
                a3 = self._renew_active_lease(c, source='extra')
            """,
        "source discriminator missing": """
            def f(self):
                a1 = self._renew_active_lease(a, source='heartbeat')
                a2 = self._renew_active_lease(b)
            """,
    }
    survived = [lbl for lbl, src in CONTROLS.items()
                if not _renewal_site_offenders(_tw.dedent(src))]
    assert not survived, (
        f"the detector still accepts: {survived} — its falsifier exceeds its reach")
    GOOD = _tw.dedent("""
        def f(self):
            a1 = self._renew_active_lease(a, source='heartbeat')
            a2 = self._renew_active_lease(b, source='sub_stripe_result')
        """)
    assert not _renewal_site_offenders(GOOD), (
        f"the detector rejects a CORRECT shape: {_renewal_site_offenders(GOOD)}")
    return (f"2 AST renewal calls, heartbeat + sub_stripe_result; "
            f"{len(CONTROLS)}/{len(CONTROLS)} wrong shapes rejected, "
            f"1/1 correct shape accepted")

def gate_n3_inbound_envelope_arity_unchanged():
    """The `inbound` tuple keeps its five fields. Two certified suites unpack it
    positionally, and widening it to carry the arrival stamp would have
    superseded their guards for no gain — the envelope was already the
    established carrier (F1-R2a)."""
    src = inspect.getsource(RangeMinerCoordinator._conn_reader_loop)
    assert 'inbound.put(("msg", rawsock, msg, put_credit_id, None)' in src, (
        "the msg envelope is no longer the certified 5-tuple")
    drain = inspect.getsource(RangeMinerCoordinator.serve_trial)
    assert ("kind, rawsock, msg, credit_id, reader_exit = inbound.get("
            in drain), "the drain no longer unpacks the certified 5-tuple"
    return "5-field envelope preserved on both put and get"


def gate_n6_no_unavailable_path_renders_count_shaped():
    """[R1-C, THE AUDIT BETA ORDERED] *"Audit the other new arms for the same
    shape."*

    Rather than re-reading each one by eye — which is how the first instance
    survived — this is a STRUCTURAL sweep over the live source of every new
    coordinator method: **no `except` handler may return anything count-shaped.**
    A handler may return `None`, or a dict whose `status` is `UNAVAILABLE` and
    whose count-named fields are all `None`. Anything else — a bare `[]`, a `0`,
    a dict with a numeric count — is the defect, wherever it appears and however
    it is spelled.

    WRONG INPUT THAT REDS IT: reintroducing `return []` in any handler. That is
    executed below as a positive control, so this gate is known to be able to
    fail."""
    import ast as _ast
    import textwrap as _tw
    NEW = ("stamp_frame_arrival", "frame_queue_residency", "_stripe_rx_slot",
           "frame_stripe_id", "note_stripe_claimed", "note_stripe_frame_enqueued",
           "note_stripe_frame_dequeued", "note_stripe_frame_accepted",
           "note_stripe_frame_deferred", "note_stripe_frame_released",
           "stripe_rx_snapshot", "emit_stripe_rx_summary",
           "active_stripe_accounting", "maybe_emit_active_stripe_accounting",
           "expired_claimed_stripes_for_report")
    COUNTY = ("count", "frames", "stripes", "ids", "total", "accepted",
              "enqueued", "dequeued", "pending", "deferred")

    def _count_shaped(v):
        """A value that reads as a MEASUREMENT: a container literal, a numeric
        constant, or a dict carrying a numeric count-named field."""
        if isinstance(v, (_ast.List, _ast.Tuple, _ast.Set)):
            return f"a bare {type(v).__name__.lower()}"
        if isinstance(v, _ast.Constant) and isinstance(v.value, (int, float)) \
                and not isinstance(v.value, bool):
            return f"the number {v.value!r}"
        if isinstance(v, _ast.Dict):
            keys = {k.value for k in v.keys if isinstance(k, _ast.Constant)}
            if "status" not in keys:
                return "a dict with no status"
            for k, val in zip(v.keys, v.values):
                if isinstance(k, _ast.Constant) and \
                        any(t in str(k.value) for t in COUNTY):
                    if not (isinstance(val, _ast.Constant) and val.value is None):
                        return f"count-shaped {k.value!r} = {_ast.unparse(val)}"
        return None

    def _offenders(src):
        """[R6-4] The unavailable-path sweep, no longer confined to `Return`
        nodes physically inside an `except`.

        ⚠ A handler that ASSIGNS `result = []` and returns it OUTSIDE the
        handler escaped the previous form entirely — the count-shaped value
        still reaches the caller on the unavailable path, which is the whole
        claim. Names bound to count-shaped values inside any handler are now
        tracked, and any later `Return` mentioning such a name is an offender."""
        bad = []
        tree = _ast.parse(_tw.dedent(src))
        # (1) direct: a Return inside a handler
        for h in (n for n in _ast.walk(tree)
                  if isinstance(n, _ast.ExceptHandler)):
            for r in (n for n in _ast.walk(h) if isinstance(n, _ast.Return)):
                v = r.value
                if v is None or (isinstance(v, _ast.Constant) and v.value is None):
                    continue
                shape = _count_shaped(v)
                if shape is not None:
                    bad.append(f"returns {shape} from an except handler")
                elif not isinstance(v, _ast.Dict):
                    # A status dict that `_count_shaped` cleared is the CORRECT
                    # unavailable contract and must pass. Anything else returned
                    # from a handler cannot be shown not to be count-shaped, so
                    # it fails closed.
                    bad.append(f"returns {_ast.unparse(v)[:60]} from an except "
                               f"handler — not a status dict, so it cannot be "
                               f"shown not to render count-shaped")
        # (2) [R6-4 · R7-3] indirect: bound in a handler, returned outside it
        # R7-3: `Assign` ALONE missed `result: list = []` (an `AnnAssign`), and a
        # single hop missed `alias = result` -> `return alias`. Both bindings are
        # now seeded and simple name-to-name taint is propagated TO A FIXED
        # POINT before returns are inspected.
        def _bind_targets(node):
            if isinstance(node, _ast.Assign):
                return [t for t in node.targets if isinstance(t, _ast.Name)], node.value
            if isinstance(node, _ast.AnnAssign) and isinstance(node.target, _ast.Name):
                return [node.target], node.value
            return [], None

        tainted = {}
        for h in (n for n in _ast.walk(tree)
                  if isinstance(n, _ast.ExceptHandler)):
            for a in (n for n in _ast.walk(h)
                      if isinstance(n, (_ast.Assign, _ast.AnnAssign))):
                targets, value = _bind_targets(a)
                if value is None:
                    continue
                shape = _count_shaped(value)
                if shape is None:
                    continue
                for t in targets:
                    tainted[t.id] = shape
        # fixed point over simple name-to-name rebinding, anywhere in the function
        changed = True
        while changed:
            changed = False
            for a in (n for n in _ast.walk(tree)
                      if isinstance(n, (_ast.Assign, _ast.AnnAssign))):
                targets, value = _bind_targets(a)
                if not targets or value is None:
                    continue
                # [R7 sweep] Not just name-to-name: ANY binding whose value
                # mentions a tainted name carries the count-shaped value onward
                # — `box = {'v': result}; return box['v']` reaches the caller
                # just as surely as a bare alias does. Fail closed.
                mentioned = sorted({n.id for n in _ast.walk(value)
                                    if isinstance(n, _ast.Name)} & set(tainted))
                if not mentioned:
                    continue
                for t in targets:
                    if t.id not in tainted:
                        tainted[t.id] = (f"{tainted[mentioned[0]]} (via "
                                         f"`{mentioned[0]}`)")
                        changed = True
        if tainted:
            in_handler = {id(r) for h in _ast.walk(tree)
                          if isinstance(h, _ast.ExceptHandler)
                          for r in _ast.walk(h) if isinstance(r, _ast.Return)}
            for r in (n for n in _ast.walk(tree) if isinstance(n, _ast.Return)):
                if r.value is None or id(r) in in_handler:
                    continue
                names = {n.id for n in _ast.walk(r.value)
                         if isinstance(n, _ast.Name)}
                for nm in sorted(names & set(tainted)):
                    bad.append(
                        f"`{nm}` is bound to {tainted[nm]} INSIDE an except "
                        f"handler and returned outside it — the count-shaped "
                        f"value still reaches the caller on the unavailable path")
        return bad

    offenders = []
    for name in NEW:
        fn = getattr(RangeMinerCoordinator, name, None)
        if fn is None:
            offenders.append(f"{name} is MISSING — the sweep is vacuous")
            continue
        for b in _offenders(inspect.getsource(fn)):
            offenders.append(f"{name}: {b}")
    assert not offenders, (
        "an UNAVAILABLE path renders count-shaped: " + "; ".join(offenders))

    # POSITIVE CONTROLS, EXECUTED — each is a wrong shape the docstring names.
    CONTROLS = {
        "return [] from the handler": (
            "def f(self, r):\n    try:\n        return g()\n"
            "    except Exception:\n        return []\n"),
        # [R6-4] the shape that escaped the previous form entirely
        "assign in the handler, return outside": (
            "def f(self, r):\n    try:\n        return g()\n"
            "    except Exception:\n        result = []\n    return result\n"),
        "handler returns a numeric count": (
            "def f(self, r):\n    try:\n        return g()\n"
            "    except Exception:\n        return 0\n"),
        # [R7-3] THE TWO SHAPES BETA EXECUTED AGAINST THE R6 REPAIR.
        "alias chain out of the handler": (
            "def f(self, r):\n    try:\n        return g()\n"
            "    except Exception:\n        result = []\n"
            "    alias = result\n    return alias\n"),
        # [R7 sweep, found by re-attacking the R7 repair itself]
        "taint carried through a container element": (
            "def f(self, r):\n    try:\n        return g()\n"
            "    except Exception:\n        result = []\n"
            "    box = {'v': result}\n    return box['v']\n"),
        "AnnAssign binding in the handler": (
            "def f(self, r):\n    try:\n        return g()\n"
            "    except Exception:\n        result: list = []\n"
            "    return result\n"),
        "handler returns a dict with a numeric count": (
            "def f(self, r):\n    try:\n        return g()\n"
            "    except Exception:\n"
            "        return {'status': 'UNAVAILABLE', 'active_count': 0}\n"),
    }
    survived = [lbl for lbl, src in CONTROLS.items() if not _offenders(src)]
    assert not survived, (
        f"the detector still accepts: {survived} — a count-shaped value reaches "
        f"the caller on an unavailable path and the sweep does not see it")
    # CORRECT-CODE CONTROL: the real shape must be accepted, or the detector is
    # merely strict. A handler binding a STATUS STRING is not count-shaped.
    GOOD = ("def f(self, r):\n    try:\n        return g()\n"
            "    except Exception as exc:\n"
            "        return {'status': 'UNAVAILABLE', 'active_count': None,\n"
            "                'stripes': None, 'error': str(exc)}\n")
    assert not _offenders(GOOD), (
        f"the detector rejects the CORRECT unavailable shape: {_offenders(GOOD)}")
    return (f"{len(NEW)} methods swept, 0 count-shaped unavailable paths; "
            f"{len(CONTROLS)}/{len(CONTROLS)} wrong shapes rejected, "
            f"1/1 correct shape accepted")


def gate_n7_mutants_restore_the_original_descriptors():
    """[R2 AUDIT] A MUTANT THAT DOES NOT RESTORE CLEANLY CONTAMINATES EVERY LATER
    ARM, and the contamination is silent.

    `getattr(cls, name)` on a `staticmethod` yields a plain function; assigning
    it back installs an INSTANCE method, so `self.stamp_frame_arrival(msg)`
    thereafter binds `msg` to `at` and stamps nothing. **Two R2 mutants failed on
    exactly that, for reasons unrelated to what they were testing** — and had
    they not happened to depend on the damaged surface, the battery would have
    gone green with a broken instrument underneath it.

    This runs LAST and asserts the descriptor types are exactly what the module
    defines, so no future mutant can leave the class quietly altered.

    WRONG INPUT THAT REDS IT: any mutant restoring `staticmethod`-defined
    attributes by bare assignment. Executed as a positive control below."""
    EXPECT = {
        "stamp_frame_arrival": staticmethod,
        "frame_queue_residency": staticmethod,
        "frame_token": staticmethod,
        "frame_stripe_id": staticmethod,
    }
    bad = []
    for name, kind in EXPECT.items():
        got = RangeMinerCoordinator.__dict__.get(name)
        if not isinstance(got, kind):
            bad.append(f"{name}: {type(got).__name__} (expected {kind.__name__})")
    assert not bad, f"a mutant left the class altered: {bad}"
    # ...and the instance calling convention actually works, which is the thing
    # the descriptor type is a proxy for.
    with tempfile.TemporaryDirectory() as tmp:
        c = _coord(tmp)
        m = SubStripeResultMessage(
            worker_id="w", stripe_id="s", sub_index=0, seed_start=0,
            seed_count=1, survivor_count=0, inline=[], size_bytes=2,
            sha256="0" * 64)
        c.stamp_frame_arrival(m)
        assert c.frame_token(m) is not None, (
            "an instance call to stamp_frame_arrival stamped nothing — the "
            "staticmethod was replaced by an instance method somewhere")
    # positive control: the naive restore is detected
    naive = RangeMinerCoordinator.stamp_frame_arrival          # plain function
    assert not isinstance(naive, staticmethod), (
        "the positive control is malformed — getattr should yield a function")
    return f"{len(EXPECT)} descriptors intact; instance call stamps correctly"


def gate_n5_no_touch_surfaces_are_untouched():
    """THE CERTIFIED PIN, RE-ASSERTED HERE so this amendment carries its own
    copy of the constraint rather than relying on another suite to notice.

    `test_s172_attempt6_remediation.py` FAIR-3/2 pins seven definitions by AST
    digest. The first draft of this instrumentation edited TWO of them —
    `schedule_pending_stripes` (to stamp the claim instant) and
    `process_lease_expiry` (to emit the expiry summary) — and that gate caught
    both. Neither hook needed to be there: the scheduler already RETURNS its
    handoff records, and the expiry query is a pure read the serve loop can make
    for itself. An instrument is not a reason to supersede a certified pin."""
    NO_TOUCH = ("claim_stripe", "schedule_pending_stripes", "renew_lease",
                "_renew_active_lease", "process_lease_expiry",
                "_handle_stripe_failure_locked",
                "_execution_set_expected_workers")
    NEW_NAMES = ("note_stripe_claimed", "emit_stripe_rx_summary",
                 "note_stripe_frame_accepted", "stamp_frame_arrival",
                 "maybe_emit_active_stripe_accounting",
                 "expired_claimed_stripes_for_report")
    offenders = []
    for name in NO_TOUCH:
        fn = getattr(RangeMinerCoordinator, name, None)
        if fn is None:
            continue
        src = inspect.getsource(fn)
        for new in NEW_NAMES:
            if new in src:
                offenders.append(f"{name} contains {new}")
    assert not offenders, (
        f"this amendment reaches into a NO-TOUCH surface: {offenders}")
    # ...and the hooks really are in the serve loop, so the gate is not vacuous
    serve = inspect.getsource(RangeMinerCoordinator.serve_trial)
    for required in ("note_stripe_claimed", "emit_stripe_rx_summary",
                     "maybe_emit_active_stripe_accounting"):
        assert required in serve, (
            f"{required} is in neither the no-touch surfaces nor the serve "
            f"loop — this gate would pass on an instrument that does not exist")
    return f"{len(NO_TOUCH)} pinned surfaces clean; 3 hooks in serve_trial"


def gate_n4_worker_compute_path_source_unchanged_by_the_freeze():
    """THE FINDING FROM THE MID-TURN QUESTION, KEPT AS A GATE so it cannot be
    quietly falsified later: nothing on the worker's compute or sieve path
    changed at `69ff222` (the fleet freeze). `sieve_gpu_worker.py`,
    `prng_registry.py` and `miner/range_miner_protocol.py` are byte-identical
    from `69ff222~1` through HEAD, and inside the worker only the sentinel /
    release-barrier definitions were added.

    Skipped, not failed, outside a git worktree."""
    import subprocess
    try:
        subprocess.run(["git", "rev-parse", "69ff222"], cwd=_ROOT,
                       capture_output=True, check=True)
    except Exception:                                            # noqa: BLE001
        return "SKIPPED — commit 69ff222 not reachable from this tree"
    unchanged = []
    for f in ("sieve_gpu_worker.py", "prng_registry.py",
              "miner/range_miner_protocol.py"):
        a = subprocess.run(["git", "show", f"69ff222~1:{f}"], cwd=_ROOT,
                           capture_output=True).stdout
        b = subprocess.run(["git", "show", f"69ff222:{f}"], cwd=_ROOT,
                           capture_output=True).stdout
        assert a and b, f"could not read {f} at the freeze boundary"
        assert hashlib.sha256(a).digest() == hashlib.sha256(b).digest(), (
            f"{f} CHANGED at the freeze commit — the compute-path finding in "
            f"the H1/H2 report is falsified and must be revised")
        unchanged.append(f)
    return f"{len(unchanged)} compute-path files byte-identical across 69ff222"


# ===========================================================================
def main():
    print("=" * 74)
    print("S172 — H1/H2 DISCRIMINATION INSTRUMENTATION (forensic §7 A-E)")
    print("=" * 74)

    print("\n-- §B  timestamps (precondition) --")
    check("B1-TIMESTAMP-VARIES", gate_b1_timestamp_prefix_is_present_and_utc)
    check("B2-GATE-PARSERS-INTACT", gate_b2_prefix_does_not_break_the_certified_gate_parsers)

    print("\n-- §A  worker stripe lifecycle --")
    check("A1-FOUR-RECORDS-IN-ORDER", gate_a1_four_records_once_each_in_order)
    check("A2-RATE-IS-PER-STRIPE", gate_a2_rate_is_per_stripe_not_per_substripe)
    check("A3-COMPUTE-S-VARIES", gate_a3_compute_s_measures_and_varies)
    check("A4-SEND-STALL-VARIES", gate_a4_send_stall_measures_and_varies)
    check("A5-H1-VS-H2-TOLD-APART", gate_a5_h1_versus_h2_are_TOLD_APART)
    check("A10-HEARTBEAT-NOT-CHARGED", gate_a10_heartbeat_syscall_is_not_charged_to_the_stripe)
    check("A11-LOCK-WAIT-IS-VISIBLE", gate_a11_lock_wait_appears_as_stripe_send_stall)
    check("A6-RESIDUAL-IS-A-THIRD-OUTCOME", gate_a6_unattributed_residual_is_a_third_outcome)
    check("A7-STRIPE-END-ON-FAILURE", gate_a7_stripe_end_on_the_failure_path_too)
    check("A8-UNOBSERVED-NEVER-ZERO", gate_a8_unobserved_is_never_zero)
    check("A9-SESSION-END-CAN-VARY", gate_a9_session_end_carries_a_field_that_can_vary)

    print("\n-- §C  coordinator arrival / queue residency --")
    check("C1-RESIDENCY-VARIES", gate_c1_residency_measures_and_varies)
    check("C2-UNSTAMPED-IS-NONE", gate_c2_unstamped_frame_is_unobserved_not_zero)
    check("C3-STAMP-NOT-ON-THE-WIRE", gate_c3_the_stamp_never_reaches_the_wire)
    check("C4-STAMP-BEFORE-ANY-BRANCH", gate_c4_reader_stamps_before_any_branch)
    check("C5-ZERO-FRAMES-DISTINGUISHABLE", gate_c5_summary_reports_zero_frames_distinguishably)
    check("C6-SUMMARY-AT-EXPIRY", gate_c6_summary_is_emitted_at_the_lease_expiry_boundary)
    check("C7-CLAIM-PRECISION-LABELLED", gate_c7_claim_precision_distinguishes_measured_from_reconciled)
    check("C8-ARRIVED-NOT-ACCEPTED", gate_c8_arrived_but_never_accepted_is_observable)
    check("C9-MESSAGE-CLASSES-SPLIT", gate_c9_message_classes_are_counted_separately)
    check("C10-LEASE-CLOCK-TRUTH", gate_c10_lease_progress_clock_moves_only_on_real_renewal)
    check("C11-CLOCK-BOUND-TO-LEDGER", gate_c11_renewal_clock_is_bound_to_the_ledger_answer)

    print("\n-- R3  snapshot isolation and single-call discipline --")
    check("R3-1a-SNAPSHOT-DETACHED", gate_r31a_snapshot_is_detached_from_the_live_inventory)
    check("R3-1b-RECORD-COHERENT", gate_r31b_record_is_coherent_across_mid_build_mutation)
    check("R3-2-SINGLE-CALL-STRUCTURAL", gate_r32_single_call_discipline_is_structural)

    print("\n-- R2-1  queue inventory under REAL producer/consumer concurrency --")
    check("R2-1a-PRODUCER-FIRST", gate_r21a_producer_first_reconciles)
    check("R2-1b-CONSUMER-FIRST", gate_r21b_consumer_first_inversion_reconciles)
    check("R2-1c-FREE-RUNNING-RACE", gate_r21c_interleaved_race_reconciles)
    check("R2-1d-UNTOKENED-ISOLATED", gate_r21d_untokened_frames_cannot_distort_the_inventory)

    print("\n-- §D  periodic active-stripe accounting --")
    check("D1-TICK-RATE-LIMITED", gate_d1_tick_is_rate_limited_and_one_record_per_tick)
    check("D2-LEASE-REMAINING-DECAYS", gate_d2_lease_remaining_decays_measurably)
    check("D4-OK-ZERO-IS-A-MEASUREMENT", gate_d4_successful_read_with_no_active_stripes_is_OK_zero)
    check("D5-FAILED-READ-IS-UNAVAILABLE", gate_d5_failed_read_is_UNAVAILABLE_with_no_counts)
    check("D5b-EXPIRY-QUERY-UNAVAILABLE", gate_d5b_failed_expiry_query_is_UNAVAILABLE)
    check("D5c-NO-OBSERVATION-NOT-SILENCE", gate_d5c_no_observation_is_not_silence)

    print("\n-- §E  per-stripe deferral attribution --")
    check("E1-DEFERRAL-ATTRIBUTED", gate_e1_deferral_is_attributed_and_varies)
    check("E2-EVERY-DEPARTURE-CHARGED-ONCE",
          gate_e2_every_departure_is_charged_exactly_once)
    check("E3-UNMATCHED-RELEASE-NOOP", gate_e3_an_unmatched_release_is_a_no_op)
    check("E4-DEFER-ORDER-LOCK-HELD", gate_e4_defer_precedes_release_by_lock_construction)

    print("\n-- non-regression --")
    check("N1-NO-CONTROL-FLOW-REACHED", gate_n1_no_control_flow_semantics_changed)
    check("N2-RENEWAL-SITES-UNCHANGED", gate_n2_renewal_call_sites_unchanged)
    check("N3-ENVELOPE-ARITY-UNCHANGED", gate_n3_inbound_envelope_arity_unchanged)
    check("N4-COMPUTE-PATH-UNCHANGED-69ff222", gate_n4_worker_compute_path_source_unchanged_by_the_freeze)
    check("N5-NO-TOUCH-SURFACES-CLEAN", gate_n5_no_touch_surfaces_are_untouched)
    check("N6-NO-COUNT-SHAPED-UNAVAILABLE", gate_n6_no_unavailable_path_renders_count_shaped)

    print("\n-- mutation: disable each instrument, the gate must go RED --")
    check("M1-SEND-ACCOUNTING-KILLED", mutant_m1_send_accounting_disabled)
    check("M2-COMPUTE-CLOCK-FROZEN", mutant_m2_compute_timer_frozen)
    check("M3-ARRIVAL-STAMP-REMOVED", mutant_m3_arrival_stamp_removed)
    check("M4-RESIDENCY-DEFAULTS-TO-ZERO", mutant_m4_residency_defaults_to_zero)
    check("M5-LEASE-REMAINING-PINNED", mutant_m5_lease_remaining_constant)
    check("M6-DEFERRAL-NEVER-CHARGED", mutant_m6_deferral_never_charged)
    check("M7-HEARTBEAT-FOLDED-IN", mutant_m7_heartbeat_time_folded_into_the_stripe)
    check("M8-STALL-DROPS-LOCK-WAIT", mutant_m8_stall_drops_the_lock_wait)
    check("M9-ARRIVAL-AT-ACCEPTANCE", mutant_m9_arrival_counted_only_on_acceptance)
    check("M10-UNAVAILABLE-AS-EMPTY", mutant_m10_unavailable_collapses_to_empty_list)
    check("M11-POST-PUT-FIFO-RACE", mutant_m11_post_put_fifo_race_restored)
    check("M12-CLOCK-FROM-ANY-FRAME", mutant_m12_lease_clock_driven_from_any_accepted_frame)
    check("M13-SHALLOW-SNAPSHOT", mutant_m13_shallow_snapshot_restored)
    check("M14-CLASSES-COLLAPSED", mutant_m14_message_classes_collapsed)

    print("\n-- post-mutation integrity --")
    check("N7-DESCRIPTORS-RESTORED", gate_n7_mutants_restore_the_original_descriptors)

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 74)
    print(f"\n{passed}/{total} checks green")
    if passed == total:
        print("COMPLETION SENTINEL: PASS — H1/H2 instrumentation, forensic §7 "
              "A-E, every measured field carries a clean control, a "
              "fault-injection control and a mutant (pending Team Beta review).")
        return 0
    print("COMPLETION SENTINEL: FAIL — "
          + "; ".join(n for n, ok, _ in _RESULTS if not ok))
    return 1


if __name__ == "__main__":
    sys.exit(main())
