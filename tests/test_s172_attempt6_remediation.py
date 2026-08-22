#!/usr/bin/env python3
"""S172 ATTEMPT-6 REMEDIATION — the TEN-GATE §11 battery.

FALSIFIABLE QUESTIONS (two, one per Part)
  A. Does every reader termination now retain a machine-readable cause FROM THE
     BRANCH THAT ACTUALLY TERMINATED THAT READER, and does that cause survive
     into the coordinator-side record that exists for that exit?
  B. Can arbitrary legal inbound traffic still monopolize the coordinator long
     enough to starve unrelated control-plane responsibilities — and can a
     REGISTER still be starved behind it, or the control plane starved by
     registration priority?

THE OPERATIVE CONTRACT IS §11 OF `~/dashboard_work/ATTEMPT6_REMEDIATION_DESIGN.md`
(R3 state, Beta-CERTIFIED 2026-08-13). §4 of that document and the arm tables
embedded in §8 are HISTORICAL: Beta's binding instruction was to replace, not
annotate, the operative acceptance criteria, so this file implements §11 and
nothing else. Ten gates:

    RXP-1 (13 arms) reader-exit provenance
    RXP-2  (9 arms) saturation discrimination + the single accumulator
    RXP-3 (13 arms) worker-log sentinel + the pre-REGISTER barrier
                    (arms 9-11 added at Beta R1-A: acceptance is a SAME-RECORD
                     conjunction, not two independent counts; arms 12-13 added
                     at Beta R2: an ssh TRANSPORT failure is UNAVAILABLE, an
                     executed-but-malformed remote probe stays ERROR)
    FAIR-5 (8 assertions) RED authenticity — specified and RUN FIRST, because
                          every other FAIR RED arm depends on the anchor
    FAIR-1/FAIR-2 (7 arms) control-plane fairness under the two pressure shapes
    FAIR-6 (14 arms) registration under backlog, BOTH directions
                    (arms 13-14 added at Beta R1-B: P-1 is FIRST-FRAME REGISTER
                     priority; arm 7 structurally cannot detect D2)
    FAIR-7  (7 arms) reasoned EOF: ordered, and never lost
    FAIR-3  (5 arms) no lease/admission semantic regression
    FAIR-4  (5 arms) no backpressure regression

STANDING CONVENTIONS (§11, stated once rather than ten times)
  * Every arm names the WRONG INPUT that would make it fail. A gate that cannot
    name its own falsifier is an assertion about the implementer's intent.
  * RED-arm discipline: `PINNED_COMMIT` is the FULL 40-character SHA, never a
    prefix; `_git_show` failure raises `AnchorUnavailable`; the pinned object is
    verified to still carry the defect surface BEFORE any RED arm is credited;
    a drifted anchor terminates UNAVAILABLE, which never accepts (VIR-3).
  * `exec(compile(body, …), g)` with `g = dict(vars(COORD))` — REAL module
    globals, shims installed IN `g`. The A8-B2 lesson: a verbatim copy that
    resolves its names in the TEST module's globals escapes the shim and the
    mutant survives.
  * Vacuity is a failure, not a pass. Where a gate depends on a condition
    actually obtaining (a full queue, a deadline being hit), it asserts that
    condition and reds as vacuous otherwise.

WHAT THIS FILE DOES **NOT** CLAIM
  The initiating cause of attempt 5's two lost reader sessions remains
  UNRESOLVED. Nothing here diagnoses it: Part A makes the NEXT occurrence
  self-describing, and the §1.5 lost-EOF hazard FAIR-7 closes is a derived
  property of source at `2b0d2dc`, not a claim about what happened.

Run:  source ~/venvs/torch/bin/activate && \
      python3 -u tests/test_s172_attempt6_remediation.py \
        | tee /tmp/attempt6_gates.log

  env ATTEMPT6_SKIP_SUBPROCESS_SUITES=1 skips FAIR-3 arm 1 / FAIR-4 arm 0, which
  shell out to the certified suites. They are ON by default and their result is
  part of the gate; the switch exists so an iteration loop can be fast, and a
  skipped arm is reported as UNAVAILABLE — never as a pass.
"""
import ast
import builtins
import hashlib
import json
import logging
import math
import os
import queue as _queue
import re
import socket
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import types
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_UNAV = "\033[91mUNAV\033[0m"
_results: List[Any] = []


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}", flush=True)
    except Exception as e:                                       # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}", flush=True)


def _unavailable(name, reason):
    """VIR-3: UNAVAILABLE is a distinct terminal state and never accepts."""
    _results.append((name, False, f"UNAVAILABLE: {reason}"))
    print(f"  [{_UNAV}] {name}: UNAVAILABLE: {reason}", flush=True)


def _mutant_red(fn, label):
    """VIR-2 positive control: `fn` MUST raise. A mutation the gate cannot detect
    makes the gate vacuous, so each gate proves its own detection power."""
    try:
        fn()
    except Exception:                                            # noqa: BLE001
        return
    raise AssertionError(
        f"MUTANT SURVIVED ({label}) — the gate did not detect it, so it is "
        f"vacuous and proves nothing")


from miner.range_miner_coordinator import (  # noqa: E402
    CLOSE_INTENT_DUP_REJECT,
    CLOSE_INTENT_READ_DEADLINE,
    ConnState,
    CoordinatorConfig,
    DEFAULT_ADMISSION_MAX_DISPOSITIONS,
    DEFAULT_DRAIN_BUDGET_SECONDS,
    DROP_ORIGIN_COORDINATOR,
    DROP_ORIGIN_READER,
    MinerLedger,
    NodeConfig,
    Phase5Sink,
    RX_CAPACITY_TIMEOUT_BARRIER,
    RX_CAPACITY_TIMEOUT_PAUSED,
    RX_DECODE_ERROR,
    RX_INFRASTRUCTURE_TERMINAL_EXIT,
    RX_PROTOCOL_FRAME_INVALID,
    RX_READER_EXIT_UNCLASSIFIED,
    RX_READER_INTERNAL_ERROR,
    RX_SHUTDOWN_STOP,
    RX_SHUTDOWN_STOP_WHILE_PAUSED,
    RX_TRANSPORT_ERROR,
    RX_UNOBSERVED,
    RangeMinerCoordinator,
    ReaderExit,
    TC_INBOUND_SATURATION_TIMEOUT,
    TransferAdapter,
    run_trial_miner,
)
import miner.range_miner_coordinator as COORD  # noqa: E402
import miner.range_miner_worker as WORKER      # noqa: E402
from miner.range_miner_protocol import (  # noqa: E402
    RegisterMessage,
    SubStripeResultMessage,
)
# The framed socket and the payload builder live in the WORKER module: both ends
# of the wire share ONE framing implementation, which is what makes a loopback
# peer a real peer rather than a model of one.
from miner.range_miner_worker import (  # noqa: E402
    MinerFramedSocket,
    build_substripe_payload_bytes,
)

# [§11 standing convention] THE FULL 40-CHARACTER SHA, never the abbreviation. A
# short SHA fails CLOSED if it later becomes ambiguous, but a permanent
# governance anchor must not be abbreviated at all — Beta has corrected this
# three times, and it is applied here unasked.
PINNED_COMMIT = "2b0d2dc5268946d6b1a44e268573e816b7cdcb83"
SRC_REL = "miner/range_miner_coordinator.py"
WORKER_SRC_REL = "miner/range_miner_worker.py"

CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
VARIANTS = ["java_lcg", "java_lcg_reverse", "java_lcg_hybrid",
            "java_lcg_hybrid_reverse"]
SPOOL_ROOT = "/var/spool/miner"
SKIP_SUBPROCESS = os.environ.get("ATTEMPT6_SKIP_SUBPROCESS_SUITES") == "1"


# ===========================================================================
# the pinned pre-repair anchor, and its integrity (FAIR-5's machinery)
# ===========================================================================
class AnchorUnavailable(RuntimeError):
    """The pinned pre-repair source could not be obtained, or is not it."""


def _git_show(commit: str, path: str) -> str:
    p = subprocess.run(["git", "-C", _ROOT, "show", f"{commit}:{path}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise AnchorUnavailable(
            f"{commit}:{path} does not resolve: {p.stderr.strip()}")
    return p.stdout


def _func_node(tree: ast.AST, cls: Optional[str], name: str) -> ast.FunctionDef:
    scope = tree
    if cls is not None:
        found = [n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == cls]
        if len(found) != 1:
            raise AnchorUnavailable(
                f"expected exactly one class {cls}, found {len(found)}")
        scope = found[0]
    fns = [n for n in ast.walk(scope)
           if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(fns) != 1:
        raise AnchorUnavailable(
            f"expected exactly one {cls}.{name}, found {len(fns)}")
    return fns[0]


_PINNED_CACHE: Dict[str, Any] = {}


def _pinned_src(path=SRC_REL) -> str:
    key = f"src:{path}"
    if key not in _PINNED_CACHE:
        _PINNED_CACHE[key] = _git_show(PINNED_COMMIT, path)
    return _PINNED_CACHE[key]


def _live_src(path=SRC_REL) -> str:
    with open(os.path.join(_ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def _strip_comments(src: str) -> str:
    """Executable behaviour only.

    [RED-arm discipline, Beta-corrected] The repaired source QUOTES THE OLD
    SURFACE IN ITS OWN COMMENTS — `_deliver_reader_eof`'s docstring cites the
    `timeout=0.5` / `except Exception: pass` shape it replaced, and the drain
    cites the `timeout=poll if drained == 0 else 0` form it corrected. A text
    probe over raw source would therefore find the defect surface in the FIXED
    file and credit an anchor that had drifted forward. Unparsing the AST drops
    every comment and every docstring is re-emitted as a literal, so the probes
    below run against executable structure."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _drain_while(fn: ast.FunctionDef) -> Optional[ast.While]:
    """The inbound-drain `while` inside a `serve_trial` node — identified by the
    `inbound.get(` call it contains, never by "the second while", which would
    break the moment a loop is added."""
    candidates = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.While):
            continue
        for call in ast.walk(node):
            if (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "get"
                    and getattr(call.func.value, "id", None) == "inbound"):
                candidates.append(node)
                break
    if not candidates:
        return None
    # The serve loop ENCLOSES the drain, so it matches too. The innermost match
    # is the drain — selected by latest start line rather than by "the second
    # while", which would break the next time a loop is added.
    return max(candidates, key=lambda n: n.lineno)


def _eof_put_tuple_len(fn: ast.FunctionDef) -> Optional[int]:
    """Width of the tuple in the reader's `("eof", …)` put."""
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "put"):
            for arg in node.args:
                if (isinstance(arg, ast.Tuple) and arg.elts
                        and isinstance(arg.elts[0], ast.Constant)
                        and arg.elts[0].value == "eof"):
                    return len(arg.elts)
    return None


def _assert_pinned_carries_the_defects() -> None:
    """FAIR-5's eight assertions, as a callable — every RED arm in this file
    runs through it, so a drifted anchor can never credit one."""
    src = _pinned_src()
    exe = _strip_comments(src)
    tree = ast.parse(src)
    serve = _func_node(tree, "RangeMinerCoordinator", "serve_trial")
    reader = _func_node(tree, "RangeMinerCoordinator", "_conn_reader_loop")

    # 1 + 2: the monopolization surface — a COUNT comparison against a literal,
    # with no time term in the test.
    drain = _drain_while(serve)
    if drain is None:
        raise AnchorUnavailable("pinned serve_trial has no inbound drain loop")
    test_src = ast.unparse(drain.test)
    if test_src.replace(" ", "") != "drained<256":
        raise AnchorUnavailable(
            f"pinned {PINNED_COMMIT} drain test is {test_src!r}, not the count "
            f"bound `drained < 256` — this is not the pre-repair source")
    if "perf_counter" in test_src or "deadline" in test_src:
        raise AnchorUnavailable(
            "pinned drain test already carries a deadline term — the anchor "
            "points at repaired source")
    # 3: the undifferentiated-exit surface — a FOUR-element eof tuple with a
    # bare None where the reason belongs.
    width = _eof_put_tuple_len(reader)
    if width != 4:
        raise AnchorUnavailable(
            f"pinned eof tuple is {width}-wide, not 4 — not the pre-repair "
            f"source")
    # 4: no reader-exit reason constant exists in the pinned module.
    pinned_rx = [n.targets[0].id for n in ast.walk(tree)
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id.startswith("RX_")]
    if pinned_rx:
        raise AnchorUnavailable(
            f"pinned module already declares reader-exit constants {pinned_rx} "
            f"— the anchor points at repaired source")
    # 5: the lost-EOF surface — `timeout=0.5` inside a try/except-Exception-pass.
    if "inbound.put(('eof', rawsock, None, None), timeout=0.5)" not in exe:
        raise AnchorUnavailable(
            "pinned reader does not carry the `timeout=0.5` eof put")
    _eof_guarded = False
    for node in ast.walk(reader):
        if not isinstance(node, ast.Try):
            continue
        body_src = ast.unparse(node)
        if "'eof'" in body_src and "timeout=0.5" in body_src:
            for h in node.handlers:
                if (getattr(h.type, "id", None) == "Exception"
                        and len(h.body) == 1
                        and isinstance(h.body[0], ast.Pass)):
                    _eof_guarded = True
    if not _eof_guarded:
        raise AnchorUnavailable(
            "pinned eof put is not guarded by `except Exception: pass` — the "
            "swallow surface is absent, so this is not the pre-repair source")
    # 6: the saturation-terminal and priority surfaces are ABSENT.
    if "TC_INBOUND_SATURATION_TIMEOUT" in exe:
        raise AnchorUnavailable(
            "pinned module already declares the saturation terminal class")
    if "SimpleQueue" in ast.unparse(serve):
        raise AnchorUnavailable(
            "pinned serve_trial already creates a second queue object — the "
            "anchor points at repaired source")
    # 7: the unbudgeted-idle-wait surface.
    if "timeout=poll if drained == 0 else 0" not in exe:
        raise AnchorUnavailable(
            "pinned drain does not carry the unbudgeted first-get timeout")
    # 8: SELF-PROTECTION — the same probes must REFUSE the repaired source.
    live_exe = _strip_comments(_live_src())
    live_tree = ast.parse(_live_src())
    live_drain = _drain_while(
        _func_node(live_tree, "RangeMinerCoordinator", "serve_trial"))
    if live_drain is not None:
        if ast.unparse(live_drain.test).replace(" ", "") == "drained<256":
            raise AnchorUnavailable(
                "LIVE source still carries the count bound — either the repair "
                "is absent or this probe cannot tell the two apart, and in "
                "both cases no RED arm may be credited")
    if "TC_INBOUND_SATURATION_TIMEOUT" not in live_exe:
        raise AnchorUnavailable(
            "LIVE source does not carry the saturation terminal — the probes "
            "are not discriminating between pinned and repaired source")


# ===========================================================================
# fixtures — real coordinator, real reader loop, real socketpairs
# ===========================================================================
class _Sink(Phase5Sink):
    def __init__(self):
        self.published, self.commits, self.aborts = [], [], []

    def publish_shard(self, manifest):
        self.published.append(manifest)

    def commit_trial(self, event):
        self.commits.append(event)

    def abort_trial(self, event):
        self.aborts.append(event)


class _GatedTransfer(TransferAdapter):
    def __init__(self, gate=None):
        self.gate = gate

    def fetch_remote(self, node_config, remote_path, local_temp_path):
        if self.gate is not None:
            self.gate.wait(timeout=30)
        with open(local_temp_path, "wb") as f:
            f.write(b"")

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


def _saturating_cfg(**over):
    cfg = dict(staging_workers=2, staging_queue_depth=0, staging_deferred_max=1,
               staging_capacity_timeout=600.0, miner_stripe_size=1000)
    cfg.update(over)
    return cfg


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


class _CapturedLog:
    """Every `[MINER-SESSION]` / `[S172-*]` record the module emits, as strings.

    The provenance records ARE the deliverable of Part A, so the gates read them
    off the real logger rather than off a return value — which is also what
    proves they exist for an operator at all."""

    def __init__(self, level=logging.INFO):
        self.records: List[str] = []
        self._level = level

    def __enter__(self):
        outer = self

        class _Cap(logging.Handler):
            def emit(self, record):
                outer.records.append(record.getMessage())

        self._lg = logging.getLogger("range_miner_coordinator")
        self._prev = (self._lg.level, self._lg.propagate)
        self._h = _Cap()
        self._lg.addHandler(self._h)
        self._lg.setLevel(self._level)
        self._lg.propagate = False
        return self

    def __exit__(self, *exc):
        self._lg.removeHandler(self._h)
        self._lg.setLevel(self._prev[0])
        self._lg.propagate = self._prev[1]
        return False

    def events(self, kind: str) -> List[Dict[str, Any]]:
        """Every structured record of one event kind, decoded."""
        out = []
        for line in self.records:
            if kind not in line:
                continue
            brace = line.find("{")
            if brace < 0:
                continue
            try:
                obj = json.loads(line[brace:])
            except ValueError:
                continue
            if obj.get("event") == kind:
                out.append(obj)
        return out


class _Peer:
    """ONE connection driven through the REAL production reader loop.

    `_conn_reader_loop` runs verbatim on a real socketpair with a real
    `MinerFramedSocket` — no substitute reader, no shortcut — and is handed the
    same `ConnState` / emergency / admission objects the serve loop hands it. The
    `cli` end is the "worker", so its sends are genuinely buffered by the kernel:
    a frame the coordinator has not read stays ON THE WIRE, which is the property
    several arms turn on."""

    def __init__(self, coord, worker_id, worker_by_sock, inbound, reader_stop,
                 bind=True, conn_state=None, emergency=None, admission=None,
                 run_id="run", saturation_budget_s=None, reader=None,
                 fs_wrap=None):
        self.worker_id = worker_id
        self.srv, self.cli = socket.socketpair()
        self.srv_fs = MinerFramedSocket(self.srv)
        self.cli_fs = MinerFramedSocket(self.cli)
        # The object the PRODUCTION reader is handed. Wrapping it answers "did
        # this connection decode a SECOND envelope?" BY COUNTING rather than by
        # inference, and leaves the reader itself untouched production code.
        self.reader_fs = self.srv_fs if fs_wrap is None else fs_wrap(self.srv_fs)
        self.state = conn_state or ConnState(coord.next_connection_id(run_id))
        self.emergency = emergency
        self.admission = admission
        if bind:
            worker_by_sock[self.srv] = worker_id
        target = reader or coord._conn_reader_loop
        self.exc: List[BaseException] = []

        def _run():
            try:
                target(self.reader_fs, self.srv, inbound, reader_stop,
                       worker_by_sock, conn_state=self.state,
                       emergency=self.emergency, admission=self.admission,
                       run_id=run_id, saturation_budget_s=saturation_budget_s)
            except BaseException as e:                            # noqa: BLE001
                self.exc.append(e)

        self.thread = threading.Thread(target=_run, name=f"reader-{worker_id}",
                                       daemon=True)
        self.thread.start()

    def send(self, msg):
        self.cli_fs.send_msg(msg)

    def send_raw(self, data: bytes):
        self.cli.sendall(data)

    def break_transport(self):
        """Close the far end so the reader's `recv_msg` fails — a genuine
        transport error, not a simulated return."""
        try:
            self.cli.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.cli.close()
        except OSError:
            pass

    def join(self, timeout=5.0):
        self.thread.join(timeout=timeout)
        assert not self.thread.is_alive(), (
            f"reader for {self.worker_id} never exited")

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
    """A coordinator plus N connections, each on the real reader loop, with the
    real emergency and admission channels wired exactly as `serve_trial` wires
    them."""

    def __init__(self, tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"),
                 inbound=None, maxsize=1024, saturation_budget_s=None,
                 transfer=None, sink=None, bind=True, fs_wrap=None, **cfg):
        self.coord = _coord(tmp, transfer=transfer, sink=sink, **cfg)
        self.worker_by_sock: Dict[Any, str] = {}
        self.wconn_by_worker: Dict[str, Any] = {}
        self.inbound = inbound if inbound is not None else _queue.Queue(
            maxsize=maxsize)
        self.emergency: "_queue.SimpleQueue" = _queue.SimpleQueue()
        self.admission: "_queue.SimpleQueue" = _queue.SimpleQueue()
        self.reader_stop = threading.Event()
        self.peers: Dict[str, _Peer] = {}
        self._held: List[bool] = []
        for wid in worker_ids:
            self.wconn_by_worker[wid] = _register(self.coord, wid)
            self.peers[wid] = _Peer(
                self.coord, wid, self.worker_by_sock, self.inbound,
                self.reader_stop, bind=bind, emergency=self.emergency,
                admission=self.admission, fs_wrap=fs_wrap,
                saturation_budget_s=saturation_budget_s)

    def saturate(self):
        sem = self.coord._staging_slots()
        while sem.acquire(blocking=False):
            self._held.append(True)
        assert self._held, "no staging slots existed to hold"
        assert not self.coord.staging_can_accept(), (
            "staging_can_accept() still True with every slot held")

    def release_all_slots(self):
        sem = self.coord._staging_slots()
        while self._held:
            self._held.pop()
            sem.release()

    def wait_paused(self, n, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if self.coord.paused_connection_count() >= n:
                return True
            time.sleep(0.01)
        return False

    def drain(self, timeout=0.6):
        out = []
        end = time.time() + timeout
        while time.time() < end:
            try:
                out.append(self.inbound.get(timeout=0.05))
            except _queue.Empty:
                continue
        return out

    def close(self):
        self.reader_stop.set()
        for p in self.peers.values():
            p.close()
        self.release_all_slots()


def _wait(pred, timeout=5.0, interval=0.01):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return pred()


# ===========================================================================
# RXP-1 — reader-exit provenance (§11.1, 13 arms)
#
# OPERATIVE CRITERION: provenance reaches the record that EXISTS for that exit,
# and that record is named per class. The withdrawn v1 criterion — "every class
# produces a WORKER_DISCONNECTED" — is FALSE under the accepted architecture in
# three separate ways (E8 is a whole-trial terminal with no shed; the shutdown
# paths suppress the eof entirely; and a disconnect record is emitted only where
# the socket was bound), and a gate demanding one would force an implementer to
# manufacture records the design deliberately does not emit.
# ===========================================================================
def _reader_exit_event(cap: "_CapturedLog") -> Dict[str, Any]:
    evs = cap.events("READER_EXIT")
    assert len(evs) >= 1, "no READER_EXIT record was emitted at the exit"
    return evs[-1]


def _inject_E0(tmp, cap):
    """E0 — the loop condition itself: `reader_stop` already set, so the reader
    leaves with no `break` at all and (uniquely) suppresses its eof."""
    coord = _coord(tmp)
    stop = threading.Event()
    stop.set()
    inbound = _queue.Queue(maxsize=16)
    p = _Peer(coord, "hostA:gpu0", {}, inbound, stop,
              emergency=_queue.SimpleQueue(), admission=_queue.SimpleQueue())
    p.join()
    p.close()
    return _reader_exit_event(cap), inbound


def _inject_E3(tmp, cap):
    """E3 — a genuine transport failure: the far end of a real socketpair is
    closed while `reader_stop` is CLEAR."""
    b = _Bench(tmp, worker_ids=("hostA:gpu0",))
    p = b.peers["hostA:gpu0"]
    time.sleep(0.05)                       # let the reader reach recv_msg
    p.break_transport()
    p.join()
    b.close()
    return _reader_exit_event(cap), b.inbound


def _inject_E4(tmp, cap):
    """E4 — an oversized length prefix. `recv_msg` raises ValueError from the
    64 MB frame cap, which is a PROTOCOL violation, not a decode failure."""
    b = _Bench(tmp, worker_ids=("hostA:gpu0",))
    p = b.peers["hostA:gpu0"]
    p.send_raw(struct.pack(">I", 0x7FFFFFFF))
    p.join()
    b.close()
    return _reader_exit_event(cap), b.inbound


def _inject_E5(tmp, cap):
    """E5 — a WELL-FRAMED body that is not valid JSON. `json.JSONDecodeError` is
    a ValueError subclass, so a classifier that stopped at `isinstance(exc,
    ValueError)` would report PROTOCOL_FRAME_INVALID here and the two classes
    would be indistinguishable."""
    b = _Bench(tmp, worker_ids=("hostA:gpu0",))
    p = b.peers["hostA:gpu0"]
    body = b"{not json at all"
    p.send_raw(struct.pack(">I", len(body)) + body)
    p.join()
    b.close()
    return _reader_exit_event(cap), b.inbound


def _pause_one(b, wid="hostA:gpu0", run_id="runP", seed_count=60, expected=2):
    """Drive ONE connection into the real per-connection pause."""
    b.coord.ledger.create_trial(run_id, 0, now=100.0)
    _claim(b.coord, run_id, f"{run_id}_s0", wid, b.wconn_by_worker[wid],
           seed_count=seed_count, expected=expected)
    b.saturate()
    b.peers[wid].send(_inline_result(wid, f"{run_id}_s0", 0, 0, 30))
    assert b.wait_paused(1), "the connection never paused"
    return run_id


def _inject_E6(tmp, cap):
    """E6 — `reader_stop` set WHILE PAUSED. The held envelope is discarded and
    the eof is suppressed, and both facts are recorded."""
    b = _Bench(tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"), **_saturating_cfg())
    _pause_one(b)
    b.reader_stop.set()
    b.peers["hostA:gpu0"].join()
    b.close()
    return _reader_exit_event(cap), b.inbound


def _inject_E7(tmp, cap):
    """E7 — the LATCHED capacity timeout while paused. Driven by a real bounded
    wait (a 0.2 s `staging_capacity_timeout` with a genuinely paused
    connection), never by writing the latch."""
    b = _Bench(tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"),
               **_saturating_cfg(staging_capacity_timeout=0.2))
    _pause_one(b)
    b.peers["hostA:gpu0"].join(timeout=10.0)
    b.close()
    return _reader_exit_event(cap), b.inbound


def _park_at_barrier(b, wid="hostA:gpu0", run_id="runB"):
    """pause -> resume -> deliver: the reader now owes the serve loop ONE
    disposition and is parked at the pre-decode barrier, decoding nothing."""
    b.coord.ledger.create_trial(run_id, 0, now=100.0)
    _claim(b.coord, run_id, f"{run_id}_s0", wid, b.wconn_by_worker[wid],
           seed_count=60, expected=2)
    b.saturate()
    b.peers[wid].send(_inline_result(wid, f"{run_id}_s0", 0, 0, 30))
    assert b.wait_paused(1), "the connection never paused"
    b.release_all_slots()
    b.coord._release_capacity()
    entry = b.inbound.get(timeout=5.0)
    assert entry[0] == "msg" and entry[3] is not None, (
        f"the resumed envelope did not carry its credit token: {entry[:4]}")
    assert _wait(lambda: b.coord.resume_credits_outstanding() == 1), (
        "the reservation was disposed of before the barrier could be entered")
    return entry


def _inject_E1(tmp, cap):
    """E1 — the credit barrier aborted by `reader_stop`."""
    b = _Bench(tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"), **_saturating_cfg())
    _park_at_barrier(b)
    b.reader_stop.set()
    b.peers["hostA:gpu0"].join()
    b.close()
    return _reader_exit_event(cap), b.inbound


def _inject_E2(tmp, cap):
    """E2 — the credit barrier aborted by the latched capacity timeout.

    A SECOND connection is genuinely paused so a real capacity blocker exists;
    the barrier then observes the latch the same way production would. Nothing
    is written into the latch by hand."""
    b = _Bench(tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"), **_saturating_cfg())
    _park_at_barrier(b)
    # An operator-visible config value, changed the way an operator would change
    # it — not a poked internal.
    b.coord.config.staging_capacity_timeout = 0.2
    b.saturate()
    b.peers["hostB:gpu0"].send(_inline_result("hostB:gpu0", "runB_s0", 1, 30, 30))
    assert b.wait_paused(1), "the second connection never paused"
    b.peers["hostA:gpu0"].join(timeout=10.0)
    ev = [e for e in cap.events("READER_EXIT")
          if e.get("worker_id") == "hostA:gpu0"]
    b.close()
    assert ev, "no READER_EXIT for the connection parked at the barrier"
    return ev[-1], b.inbound


def _inject_internal_error(tmp, cap):
    """READER_INTERNAL_ERROR — a raise from the reader's OWN body.

    Injected through a real production seam: `staging_can_accept` is consulted
    inside the loop, so a coordinator whose predicate raises makes the reader's
    body raise exactly where a genuine bug would."""
    b = _Bench(tmp, worker_ids=("hostA:gpu0",))
    orig = type(b.coord).staging_can_accept

    def _boom(self_):
        raise RuntimeError("injected reader-body failure")

    type(b.coord).staging_can_accept = _boom
    try:
        b.worker_by_sock[b.peers["hostA:gpu0"].srv] = "hostA:gpu0"
        b.peers["hostA:gpu0"].send(
            _inline_result("hostA:gpu0", "s0", 0, 0, 10))
        b.peers["hostA:gpu0"].join()
    finally:
        type(b.coord).staging_can_accept = orig
        b.close()
    ev = _reader_exit_event(cap)
    assert b.peers["hostA:gpu0"].exc, (
        "the internal error did not propagate out of the reader thread — a "
        "genuine bug must still reach threading.excepthook")
    return ev, b.inbound


# The taxonomy under test, as data: injection -> the class it MUST produce.
RXP1_INJECTIONS = {
    RX_SHUTDOWN_STOP: _inject_E0,
    RX_CAPACITY_TIMEOUT_BARRIER: _inject_E2,
    RX_TRANSPORT_ERROR: _inject_E3,
    RX_PROTOCOL_FRAME_INVALID: _inject_E4,
    RX_DECODE_ERROR: _inject_E5,
    RX_SHUTDOWN_STOP_WHILE_PAUSED: _inject_E6,
    RX_CAPACITY_TIMEOUT_PAUSED: _inject_E7,
    RX_READER_INTERNAL_ERROR: _inject_internal_error,
}
# E1 shares SHUTDOWN_STOP with E0 by design (the barrier's two outcomes are
# named, and `reader_stop` decides which), so it is driven separately rather
# than being a second key of the same class.
RXP1_EXTRA_SHUTDOWN_ARMS = {"E1 credit barrier under stop": _inject_E1}
# Classes that are NOT exit injections, with the reason each is not:
#   UNOBSERVED                 — the value a record carries when the reader made
#                                no observation; asserted by arm 2's default.
#   INFRASTRUCTURE_TERMINAL_EXIT — a trial terminal, driven by RXP-2/FAIR-7.
#   READER_EXIT_UNCLASSIFIED   — the fail-closed default, driven by arm 6.
RXP1_NON_INJECTED = {
    RX_UNOBSERVED, RX_INFRASTRUCTURE_TERMINAL_EXIT, RX_READER_EXIT_UNCLASSIFIED,
}


def rxp1_arm1_every_exit_carries_its_own_class():
    """ARM 1 — each of E0-E7 (+ the internal-error catch-all), injected through
    the REAL code path, produces a READER_EXIT record carrying exactly that
    class.

    WRONG INPUT THAT REDS IT: a new `break` added without a label — the record
    then says READER_EXIT_UNCLASSIFIED and this arm fails on the very next run.
    """
    for expected, inject in RXP1_INJECTIONS.items():
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with _CapturedLog() as cap:
                ev, _inbound = inject(tmp, cap)
            assert ev["reader_exit_reason"] == expected, (
                f"{inject.__name__} produced {ev['reader_exit_reason']!r}, "
                f"expected {expected!r}")
            assert ev["connection_id"] not in (None, "", RX_UNOBSERVED), (
                f"{inject.__name__}: the record carries no correlation token")
    for label, inject in RXP1_EXTRA_SHUTDOWN_ARMS.items():
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with _CapturedLog() as cap:
                ev, _inbound = inject(tmp, cap)
            assert ev["reader_exit_reason"] == RX_SHUTDOWN_STOP, (
                f"{label} produced {ev['reader_exit_reason']!r}")


def rxp1_arm2_class_reaches_worker_disconnected():
    """ARM 2 — for a BOUND socket exiting via **E5**, the class reaches
    `WORKER_DISCONNECTED` through the FIVE-field eof tuple.

    [R2 — DECLARATION CORRECTED, found by the re-audit Beta ordered] This arm
    declared *"E2-E5 or E7"* and drives **E5 alone**. Same shape as the RXP-3/3
    finding: a declaration wider than the exercise. The wording is narrowed to
    what runs; **no assertion changed**.

    WHY ONE CLASS IS NOT A COVERAGE HOLE, stated as an argument Beta can reject
    rather than as a fact: what arm 2 tests is the TRANSPORT — reader → fifth
    tuple field → `_drop_conn` → `WORKER_DISCONNECTED` — and that transport
    CARRIES the reason, it does not switch on it. RXP-1/1 separately drives all
    eight classes and asserts each reaches its own `READER_EXIT` with the right
    value. So the composition is covered across the two arms, not inside this
    one. **Widening this arm to loop E2/E3/E4/E7 is Beta's call**: the existing
    injections tear their benches down before `_drop_conn` can be called with
    live maps, so it is a restructure of certified machinery, not a loop, and R2
    is scoped narrow.

    WRONG INPUT THAT REDS IT: dropping the fifth tuple field, or a `None` in the
    reason position — the disconnect record then says UNOBSERVED for a
    reader-originated drop, which is the pre-amendment state."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            b = _Bench(tmp, worker_ids=("hostA:gpu0",))
            p = b.peers["hostA:gpu0"]
            body = b"{not json"
            p.send_raw(struct.pack(">I", len(body)) + body)
            p.join()
            entry = b.inbound.get(timeout=5.0)
            assert entry[0] == "eof", f"expected an eof, got {entry[0]!r}"
            assert len(entry) == 5, (
                f"the eof tuple is {len(entry)}-wide; the reader-exit record "
                f"rides in the fifth field")
            record = entry[4]
            assert isinstance(record, ReaderExit), type(record)
            assert record.reason == RX_DECODE_ERROR, record.reason
            # …and the serve loop's own call, with the record it was handed.
            fs_by_sock = {p.srv: p.srv_fs}
            worker_by_sock = {p.srv: "hostA:gpu0"}
            fs_by_worker = {"hostA:gpu0": p.srv_fs}
            b.coord._drop_conn(
                p.srv, fs_by_sock, worker_by_sock, fs_by_worker,
                dict(b.wconn_by_worker), ["hostA:gpu0"],
                stage_idx=0, stage_assigned=True, eligible_fn=lambda: [],
                conn_state=p.state, reader_exit=record)
            b.close()
        wd = cap.events("WORKER_DISCONNECTED")
        assert len(wd) == 1, f"expected one WORKER_DISCONNECTED, got {len(wd)}"
        assert wd[0]["reader_exit_reason"] == RX_DECODE_ERROR, wd[0]
        assert wd[0]["drop_origin"] == DROP_ORIGIN_READER, wd[0]
        assert wd[0]["coordinator_close_intent"] is None, wd[0]
        assert wd[0]["connection_id"] == record.connection_id, (
            "the disconnect record does not correlate with the exit record")


def rxp1_arm3_shutdown_paths_emit_no_disconnect():
    """ARM 3 — for E0/E1/E6 (`reader_stop` set) NO eof is enqueued at all, so no
    `WORKER_DISCONNECTED` can be emitted. The gate asserts ABSENCE.

    WRONG INPUT THAT REDS IT: manufacturing a disconnect record on the shutdown
    path to satisfy a naive "every class produces a disconnect" rule."""
    for label, inject in (("E0", _inject_E0), ("E1", _inject_E1),
                          ("E6", _inject_E6)):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with _CapturedLog() as cap:
                _ev, inbound = inject(tmp, cap)
            eofs = []
            while True:
                try:
                    e = inbound.get_nowait()
                except _queue.Empty:
                    break
                if e[0] == "eof":
                    eofs.append(e)
            assert not eofs, (
                f"{label}: an eof was enqueued on a shutdown path — `:7752`'s "
                f"suppression is by design and a record here would be synthetic")
            assert not cap.events("WORKER_DISCONNECTED"), (
                f"{label}: a disconnect record was emitted for a suppressed eof")


def rxp1_arm4_mutual_exclusivity():
    """ARM 4 — injecting class X never produces class Y.

    WRONG INPUT THAT REDS IT: two exits sharing one label — e.g. classifying on
    the exception alone, which collapses E4 and E5 (JSONDecodeError IS a
    ValueError) and collapses the orderly teardown into TRANSPORT_ERROR."""
    seen: Dict[str, str] = {}
    for expected, inject in RXP1_INJECTIONS.items():
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with _CapturedLog() as cap:
                ev, _ = inject(tmp, cap)
        got = ev["reader_exit_reason"]
        assert got not in seen or seen[got] == expected, (
            f"{inject.__name__} produced {got!r}, already produced by "
            f"{seen[got]!r} — two injections share one class")
        seen[got] = expected
    assert len(seen) == len(RXP1_INJECTIONS), (
        f"{len(RXP1_INJECTIONS)} injections produced only {len(seen)} distinct "
        f"classes: {seen}")


def rxp1_arm5_completeness_from_live_source():
    """ARM 5 — the reason constants are enumerated from LIVE SOURCE by AST, and
    any declared class with no injection arm reds.

    WRONG INPUT THAT REDS IT: adding an eleventh class without a gate."""
    tree = ast.parse(_live_src())
    declared = {n.targets[0].id for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id.startswith("RX_")}
    assert declared, "no RX_* constants found in live source"
    values = {getattr(COORD, name) for name in declared}
    covered = set(RXP1_INJECTIONS) | RXP1_NON_INJECTED
    missing = values - covered
    assert not missing, (
        f"declared reader-exit classes with no injection arm and no recorded "
        f"reason for having none: {sorted(missing)}")
    # …and the module's own frozenset must agree with the constants it lists, so
    # a class cannot be declared and then left out of the taxonomy.
    assert values == set(COORD.READER_EXIT_REASONS), (
        f"READER_EXIT_REASONS disagrees with the declared RX_* constants: "
        f"{values ^ set(COORD.READER_EXIT_REASONS)}")


def _mutated_reader(mutate):
    """The LIVE reader loop, mutated, executed with the coordinator module's REAL
    globals (the A8-B2 rule) and returned as a plain function ready to bind."""
    src = _live_src()
    fn = _func_node(ast.parse(src), "RangeMinerCoordinator", "_conn_reader_loop")
    body = textwrap.dedent(ast.get_source_segment(src, fn))
    mutant = mutate(body)
    assert mutant != body, "MUTANT NOT APPLIED — the target text was not found"
    g = dict(vars(COORD))
    exec(compile(mutant, "<mutated _conn_reader_loop>", "exec"), g)  # noqa: S102
    return g["_conn_reader_loop"]


def rxp1_arm6_fail_closed_default():
    """ARM 6 — a synthetic ninth exit with NO label reports
    READER_EXIT_UNCLASSIFIED, never a plausible-looking value.

    The mutant deletes the transport classification, which is exactly the shape
    of "a new `break` added without a label".

    WRONG INPUT THAT REDS IT: defaulting the variable to TRANSPORT_ERROR — i.e.
    today's defect wearing a reason field. The mutant would then report a
    confident, wrong class and this arm fails."""
    mutant = _mutated_reader(
        lambda s: s.replace("                        exit_reason = RX_TRANSPORT_ERROR\n",
                            "                        pass\n", 1))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            coord = _coord(tmp)
            stop = threading.Event()
            inbound = _queue.Queue(maxsize=16)
            p = _Peer(coord, "hostA:gpu0", {}, inbound, stop,
                      emergency=_queue.SimpleQueue(),
                      admission=_queue.SimpleQueue(),
                      reader=types.MethodType(mutant, coord))
            time.sleep(0.05)
            p.break_transport()
            p.join()
            p.close()
        ev = _reader_exit_event(cap)
    assert ev["reader_exit_reason"] == RX_READER_EXIT_UNCLASSIFIED, (
        f"an unlabelled exit reported {ev['reader_exit_reason']!r} instead of "
        f"the fail-closed default")


def rxp1_arm7_orthogonality():
    """ARM 7 — a coordinator-initiated close produces
    `coordinator_close_intent=<known>` AND a `reader_exit_reason` derived from
    the reader's OWN exception. Neither field is a value of the other, and no
    record asserts causation between them.

    WRONG INPUT THAT REDS IT: re-collapsing them into
    `CLOSED_BY_COORDINATOR:<intent>` — one field would then carry both facts and
    the reader's observation would become unrecoverable."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            b = _Bench(tmp, worker_ids=("hostA:gpu0",), bind=False)
            p = b.peers["hostA:gpu0"]
            time.sleep(0.05)
            b.coord._drop_conn(
                p.srv, {p.srv: p.srv_fs}, {}, {}, {}, [],
                stage_idx=1, stage_assigned=True, eligible_fn=lambda: [],
                conn_state=p.state, close_intent=CLOSE_INTENT_READ_DEADLINE)
            p.join()
            b.close()
        ev = _reader_exit_event(cap)
    assert ev["coordinator_close_intent"] == CLOSE_INTENT_READ_DEADLINE, ev
    assert ev["drop_origin"] == DROP_ORIGIN_COORDINATOR, ev
    assert ev["reader_exit_reason"] in (RX_TRANSPORT_ERROR, RX_SHUTDOWN_STOP), ev
    assert not ev["reader_exit_reason"].startswith("CLOSED_BY_COORDINATOR"), (
        "the two orthogonal facts were collapsed back into one field")
    assert CLOSE_INTENT_READ_DEADLINE not in ev["reader_exit_reason"], ev
    assert not any(k for k in ev if "caused" in k.lower()), (
        f"a record asserts causation between the intent and the observation: "
        f"{sorted(ev)}")


def rxp1_arm8_reachability_pin():
    """ARM 8 — AST over LIVE source: exactly three `_drop_conn` call sites, and
    the two coordinator-initiated ones operate on UNBOUND sockets.

    THIS ARM IS THE REASON THE COMBINED RECORD IS NOT CLAIMED, and it reds the
    day a future edit drops a BOUND socket from the coordinator side — at which
    point the coordinator-originated field contract must be applied consciously
    rather than discovered in a forensic.

    WRONG INPUT THAT REDS IT: adding a coordinator-side drop of a bound socket
    without applying that contract."""
    src = _live_src()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "_drop_conn"]
    assert len(calls) == 3, (
        f"{len(calls)} `_drop_conn` call sites, expected exactly 3 "
        f"(eof / read-deadline / dup-reject)")
    intents = [kw.value for c in calls for kw in c.keywords
               if kw.arg == "close_intent"]
    assert len(intents) == 2, (
        f"{len(intents)} call sites pass a close intent, expected exactly 2 — "
        f"the eof path must pass NONE, because an eof-triggered drop is the "
        f"consequence of a reader exit, not a coordinator decision")
    # the read-deadline site is guarded by the registered-`continue`
    serve = _func_node(tree, "RangeMinerCoordinator", "serve_trial")
    deadline_guard = False
    for node in ast.walk(serve):
        if not isinstance(node, ast.For):
            continue
        body = ast.unparse(node)
        if "_drop_conn" in body and "read_deadline" in body:
            first = node.body[0]
            deadline_guard = (
                isinstance(first, ast.If)
                and "registered" in ast.unparse(first.test)
                and any(isinstance(s, ast.Continue) for s in first.body))
    assert deadline_guard, (
        "the read-deadline drop is no longer guarded by the registered-continue "
        "— it can now act on a BOUND socket, and the coordinator-originated "
        "field contract has become live without being applied")
    # the dup-reject site: `_serve_register` returns that status BEFORE binding
    reg = _func_node(tree, "RangeMinerCoordinator", "_serve_register")
    reject_line = bind_line = None
    for node in ast.walk(reg):
        if (isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)
                and node.value.value == "reject_dup_worker"):
            reject_line = node.lineno
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Subscript)
                        and getattr(t.value, "id", None) == "worker_by_sock"
                        for t in node.targets)):
            bind_line = node.lineno
    assert reject_line and bind_line and reject_line < bind_line, (
        f"`reject_dup_worker` no longer returns before the socket is bound "
        f"(reject at {reject_line}, bind at {bind_line})")


def rxp1_arm9_eof_reap_is_absent():
    """ARM 9 — `eof_reap` is absent from the intent vocabulary in live source.

    WRONG INPUT THAT REDS IT: reintroducing it, i.e. labelling a CONSEQUENCE as
    an INTENT — which directly contradicts the canonical reader-originated
    shape (`drop_origin=reader`, `coordinator_close_intent=null`)."""
    # EXECUTABLE source only. The live module DOCUMENTS the removal in a
    # comment ("`eof_reap` IS DELIBERATELY ABSENT"), and a raw-text probe would
    # red on the very sentence that records the ruling — the same trap the
    # RED-arm discipline names for the pinned anchor, met here in the live file.
    exe = _strip_comments(_live_src())
    assert "eof_reap" not in exe, (
        "`eof_reap` is back in the executable coordinator source; an "
        "eof-triggered drop is mechanical cleanup BECAUSE the reader already "
        "terminated — a consequence, never a coordinator intent")
    assert set(COORD.COORDINATOR_CLOSE_INTENTS) == {
        CLOSE_INTENT_READ_DEADLINE, CLOSE_INTENT_DUP_REJECT}, (
        f"the writable intent vocabulary drifted: "
        f"{sorted(COORD.COORDINATOR_CLOSE_INTENTS)}")


def rxp1_arm10_e3_under_stop_is_shutdown():
    """ARM 10 — [RXP-1 IS UNSATISFIABLE WITHOUT IT] a reader woken by teardown's
    `shutdown()` reaches the transport handler and MUST be classified
    SHUTDOWN_STOP, decided by `reader_stop.is_set()` at the moment the exception
    is caught — Defect A §10's certified discriminator.

    WRONG INPUT THAT REDS IT: classifying on the exception alone. The clean
    control below then fails, because an ORDERLY shutdown reports a transport
    fault."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            b = _Bench(tmp, worker_ids=("hostA:gpu0",))
            p = b.peers["hostA:gpu0"]
            time.sleep(0.05)          # the reader is now blocked in recv_msg
            b.reader_stop.set()       # …the teardown's FIRST action
            try:
                p.srv.shutdown(socket.SHUT_RDWR)   # …and its second
            except OSError:
                pass
            p.join()
            b.close()
        ev = _reader_exit_event(cap)
    assert ev["reader_exit_reason"] == RX_SHUTDOWN_STOP, (
        f"an orderly teardown reported {ev['reader_exit_reason']!r} — the "
        f"`reader_stop` discriminator is not being applied at the catch")
    # [R1 AUDIT] Was `"Error" in str(...)`, which any string ending in "Error"
    # satisfies — including a fabricated one, and including a class from a
    # completely different failure. The recorded name must RESOLVE to a real
    # transport exception type, which a placeholder cannot.
    exc_name = ev["reader_exit_exc_class"]
    exc_type = getattr(builtins, str(exc_name), None)
    assert isinstance(exc_type, type) and issubclass(exc_type, OSError), (
        f"the recorded exception class {exc_name!r} does not resolve to an "
        f"OSError subclass — an orderly teardown's transport wake is an "
        f"OSError, and a name that resolves to nothing records nothing: {ev}")


def rxp1_arm11_close_intent_emitted_even_when_unbound():
    """ARM 11 — every non-null intent emits `CONNECTION_CLOSE_INTENT` at
    `_drop_conn`'s FIRST statement, INCLUDING when `wid is None`.

    WRONG INPUT THAT REDS IT: emitting it only inside `if wid is not None:`. The
    two sites that actually set an intent are exactly the UNBOUND ones, so a
    record confined to that branch would never fire at all."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord = _coord(tmp)
        for intent in (CLOSE_INTENT_READ_DEADLINE, CLOSE_INTENT_DUP_REJECT):
            state = ConnState(coord.next_connection_id("runI"))
            sock = socket.socket()
            with _CapturedLog() as cap:
                coord._drop_conn(sock, {}, {}, {}, {}, [],
                                 stage_idx=2, stage_assigned=False,
                                 eligible_fn=lambda: [],
                                 conn_state=state, close_intent=intent)
            sock.close()
            evs = cap.events("CONNECTION_CLOSE_INTENT")
            assert len(evs) == 1, (
                f"{intent}: expected exactly one CONNECTION_CLOSE_INTENT on an "
                f"unbound socket, got {len(evs)}")
            assert evs[0]["coordinator_close_intent"] == intent, evs[0]
            assert evs[0]["connection_id"] == state.connection_id, evs[0]
            assert evs[0]["worker_id"] == RX_UNOBSERVED, (
                f"an unbound socket reported a fabricated identity: {evs[0]}")
            assert not cap.events("WORKER_DISCONNECTED"), (
                "an unbound drop emitted a disconnect record")


def rxp1_arm12_the_race_gate():
    """ARM 12 — THE RACE GATE. Reader exit FIRST, coordinator decision SECOND.

    This is the ordering R2's model lost: the read-deadline scan runs AFTER the
    inbound drain in the same iteration, so a reader that fails INDEPENDENTLY
    emits its `READER_EXIT` with a null intent BEFORE the marker exists, and
    `_drop_conn` then emits nothing because `wid is None`. The coordinator would
    have known something and written it nowhere — attempt 5's condition in
    miniature.

    WRONG INPUT THAT REDS IT: carrying the intent ONLY on `READER_EXIT`."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            b = _Bench(tmp, worker_ids=("hostA:gpu0",), bind=False)
            p = b.peers["hostA:gpu0"]
            time.sleep(0.05)
            # 1. the reader fails on its own account — NOT caused by us
            p.break_transport()
            p.join()
            exit_ev = _reader_exit_event(cap)
            assert exit_ev["coordinator_close_intent"] is None, (
                "the intent was set before the reader exited — this is not the "
                "race the arm exists to drive")
            # 2. …and only THEN does the serve loop reach its read-deadline scan
            b.coord._drop_conn(
                p.srv, {p.srv: p.srv_fs}, {}, {}, {}, [],
                stage_idx=3, stage_assigned=True, eligible_fn=lambda: [],
                conn_state=p.state, close_intent=CLOSE_INTENT_READ_DEADLINE)
            b.close()
        intents = cap.events("CONNECTION_CLOSE_INTENT")
    assert len(intents) == 1, (
        f"the coordinator's decision left no durable record ({len(intents)} "
        f"CONNECTION_CLOSE_INTENT records) — the fact vanished exactly as it "
        f"would have under the R2 model")
    assert intents[0]["connection_id"] == exit_ev["connection_id"], (
        "the two records cannot be correlated")
    assert exit_ev["coordinator_close_intent"] is None, (
        "the reader's null was BACK-FILLED once the intent became known; it is "
        "evidence of ordering and must be left as emitted")


def rxp1_arm13_no_merge_no_causation():
    """ARM 13 — `CONNECTION_CLOSE_INTENT` and `READER_EXIT` are separate records
    correlated by `connection_id`, neither rewritten by the other, and no record
    asserts that the coordinator's `shutdown()` caused the reader's exception.

    WRONG INPUT THAT REDS IT: merging them, or back-filling the reader's null."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            b = _Bench(tmp, worker_ids=("hostA:gpu0",), bind=False)
            p = b.peers["hostA:gpu0"]
            time.sleep(0.05)
            b.coord._drop_conn(
                p.srv, {p.srv: p.srv_fs}, {}, {}, {}, [],
                stage_idx=0, stage_assigned=False, eligible_fn=lambda: [],
                conn_state=p.state, close_intent=CLOSE_INTENT_DUP_REJECT)
            p.join()
            b.close()
        intents = cap.events("CONNECTION_CLOSE_INTENT")
        exits = cap.events("READER_EXIT")
    assert len(intents) == 1 and len(exits) == 1, (intents, exits)
    assert intents[0]["event"] != exits[0]["event"], "the records were merged"
    assert intents[0]["connection_id"] == exits[0]["connection_id"]
    assert "reader_exit_reason" not in intents[0], (
        "the coordinator's decision record carries the reader's observation — "
        "that is a merge, and it claims a chronology the coordinator cannot know")
    for rec in (intents[0], exits[0]):
        blob = json.dumps(rec).lower()
        for banned in ("because", "caused", "due_to"):
            assert banned not in blob, (
                f"a provenance record asserts causation ({banned!r}): {rec}")


def rxp1_vacuity_against_pinned_source():
    """VACUITY — the whole gate must RED against pinned `2b0d2dc`, where no exit
    has a reason at all. If it does not, it is asserting something the defect
    already satisfies."""
    _assert_pinned_carries_the_defects()
    src = _pinned_src()
    fn = _func_node(ast.parse(src), "RangeMinerCoordinator", "_conn_reader_loop")
    body = textwrap.dedent(ast.get_source_segment(src, fn))
    g = dict(vars(COORD))
    exec(compile(body, f"<pinned {PINNED_COMMIT}>", "exec"), g)     # noqa: S102
    pinned_reader = g["_conn_reader_loop"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            coord = _coord(tmp)
            stop = threading.Event()
            inbound = _queue.Queue(maxsize=16)
            # The pinned reader takes FIVE positional parameters and no keyword
            # extensions, so it is invoked exactly as the pinned serve loop
            # invoked it — no shim, no adapter.
            p = socket.socketpair()
            fs = MinerFramedSocket(p[0])
            th = threading.Thread(
                target=types.MethodType(pinned_reader, coord),
                args=(fs, p[0], inbound, stop, {}), daemon=True)
            th.start()
            time.sleep(0.05)
            try:
                p[1].shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            p[1].close()
            th.join(timeout=5.0)
            assert not th.is_alive(), "the pinned reader never exited"
        assert not cap.events("READER_EXIT"), (
            "the PINNED reader emitted a READER_EXIT record — the anchor is not "
            "the pre-repair source and no arm of this gate may be credited")
        entry = inbound.get(timeout=2.0)
        assert entry[0] == "eof" and len(entry) == 4, (
            f"the pinned eof is not the 4-wide undifferentiated tuple: {entry}")
        for s in p:
            try:
                s.close()
            except OSError:
                pass


# ===========================================================================
# RXP-2 — saturation discrimination and the single accumulator (§11.2, 9 arms)
#
# OPERATIVE CRITERION: local ingress saturation (a) never appears as a transport
# exit reason, (b) never sheds a legitimate worker, (c) terminates the TRIAL when
# `S` is exhausted, and (d) is accounted by exactly ONE non-resettable
# per-connection accumulator.
# ===========================================================================
def _saturated_bench(tmp, budget, maxsize=1, **cfg):
    """A bench whose `inbound` is genuinely FULL — filled to `maxsize` with real
    entries, so the reader's `put` really does raise `_queue.Full` on the
    production path. Nothing about the queue or the reader is stubbed."""
    q = _queue.Queue(maxsize=maxsize)
    for i in range(maxsize):
        q.put(("msg", None, None, None, None))
    b = _Bench(tmp, worker_ids=("hostA:gpu0",), inbound=q,
               saturation_budget_s=budget, **cfg)
    assert b.inbound.full(), "the bench queue is not actually full"
    return b


def rxp2_arm1_transient_saturation_does_not_shed():
    """ARM 1 — a reader saturated but RECOVERING WITHIN `S` does NOT exit, and
    its envelope is delivered. This is the half a naive gate omits.

    WRONG INPUT THAT REDS IT: restoring the single-shot
    `inbound.put(..., timeout=1.0)` + `break` — a transient saturation then kills
    a healthy worker, which is the ungoverned policy that fell out of a
    `timeout=` argument and was never chosen."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            b = _saturated_bench(tmp, budget=30.0, maxsize=1)
            p = b.peers["hostA:gpu0"]
            p.send(_inline_result("hostA:gpu0", "s0", 0, 0, 10))
            # the reader is now retrying against a full queue
            assert _wait(lambda: b.coord.staging_backpressure_metrics()[
                "inbound_saturation_events"] >= 1, timeout=5.0), (
                "the reader never charged a saturation wait — the queue was not "
                "actually full, so this arm would be vacuous")
            assert p.thread.is_alive(), (
                "the reader exited on a TRANSIENT saturation — the single-shot "
                "shed is back")
            b.inbound.get()                      # make room: ingress recovers
            entry = b.inbound.get(timeout=5.0)
            assert entry[0] == "msg", entry[0]
            assert p.thread.is_alive(), "the reader exited after recovering"
            b.close()
            p.join()
        assert not cap.events("EMERGENCY_TERMINAL_REQUEST"), (
            "a recovered saturation raised a trial terminal")


def rxp2_arm2_saturation_is_never_a_transport_class():
    """ARM 2 — saturation NEVER yields TRANSPORT_ERROR / PROTOCOL_FRAME_INVALID /
    DECODE_ERROR.

    WRONG INPUT THAT REDS IT: mapping `_queue.Full` onto the transport class,
    which is the masquerade the mandatory property forbids."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            b = _saturated_bench(tmp, budget=0.4, maxsize=1)
            p = b.peers["hostA:gpu0"]
            p.send(_inline_result("hostA:gpu0", "s0", 0, 0, 10))
            p.join(timeout=15.0)
            b.close()
        ev = _reader_exit_event(cap)
    assert ev["reader_exit_reason"] == RX_INFRASTRUCTURE_TERMINAL_EXIT, (
        f"an exhausted ingress budget reported {ev['reader_exit_reason']!r}")
    assert ev["reader_exit_reason"] not in (
        RX_TRANSPORT_ERROR, RX_PROTOCOL_FRAME_INVALID, RX_DECODE_ERROR)


def rxp2_arm3_exhaustion_is_a_trial_terminal_not_a_shed():
    """ARM 3 — holding past `S` fail-closes the TRIAL with
    `terminal_class == "inbound_saturation_timeout"`, and NO worker identity is
    evicted before that terminal.

    The serve-loop half runs the REAL `serve_trial`: the emergency channel is
    captured from the production wiring (the object `serve_trial` hands its
    readers), an event is put on it, and the loop's own consumption is what
    terminates the trial.

    WRONG INPUT THAT REDS IT: shedding the worker (v1 Q-2) — an eviction then
    precedes the terminal, and the run dies minutes later naming a consequence."""
    captured: Dict[str, Any] = {}
    orig_reader = RangeMinerCoordinator._conn_reader_loop

    def _capturing(self_, cfs, rawsock, inbound, reader_stop,
                   worker_by_sock=None, **kw):
        captured.setdefault("emergency", kw.get("emergency"))
        captured.setdefault("state", kw.get("conn_state"))
        return orig_reader(self_, cfs, rawsock, inbound, reader_stop,
                           worker_by_sock, **kw)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1},{"draw":2},{"draw":3}]')
        lsock = socket.socket()
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        port = lsock.getsockname()[1]
        result: Dict[str, Any] = {}
        RangeMinerCoordinator._conn_reader_loop = _capturing
        with _CapturedLog() as cap:
            try:
                def _run():
                    result["out"] = run_trial_miner(
                        "cfgSAT", None, 1, "java_lcg", [1, 2, 3], 100,
                        0.25, 0.25, False, ds,
                        skip_min=0, skip_max=0, window_size=3,
                        window_anchor=0, generator_phase=0,
                        worker_pool_size=1, miner_stripe_size=100,
                        staging_dir=os.path.join(tmp, "staging"),
                        listen_sock=lsock, family_name="java_lcg",
                        workflow_phase=1, serve_timeout=60.0, serve_poll=0.05)

                t = threading.Thread(target=_run, daemon=True)
                t.start()
                # one connection, so a reader (and its emergency queue) exists
                cli = socket.create_connection(("127.0.0.1", port), timeout=5)
                assert _wait(lambda: captured.get("emergency") is not None,
                             timeout=10.0), "no reader was ever created"
                # …and the reader's own request, with a real exit record.
                state = captured["state"]
                rec = ReaderExit(reason=RX_INFRASTRUCTURE_TERMINAL_EXIT,
                                 connection_id=state.connection_id,
                                 at=time.time(), held_envelope_discarded=True)
                captured["emergency"].put({
                    "event": "EMERGENCY_TERMINAL_REQUEST",
                    "terminal_class": TC_INBOUND_SATURATION_TIMEOUT,
                    "where": "reader_result",
                    "connection_id": state.connection_id,
                    "worker_id": RX_UNOBSERVED,
                    "spent_s": 180.0, "budget_s": 180.0,
                    "inbound_qsize": 1024, "inbound_maxsize": 1024,
                    "reader_exit_reason": RX_INFRASTRUCTURE_TERMINAL_EXIT,
                    "reader_exit": rec.as_fields(),
                })
                t.join(timeout=60.0)
                assert not t.is_alive(), "serve_trial never terminated"
                try:
                    cli.close()
                except OSError:
                    pass
            finally:
                RangeMinerCoordinator._conn_reader_loop = orig_reader
                try:
                    lsock.close()
                except OSError:
                    pass
        out = result.get("out") or {}
        assert out.get("state") == "aborted", (
            f"the trial did not fail closed on the emergency event: {out}")
        term = [r for r in cap.records if "TRIAL TERMINAL" in r
                or "emergency_terminal" in r]
        assert any(TC_INBOUND_SATURATION_TIMEOUT in r for r in term), (
            f"no terminal record names the saturation class: {term[:4]}")
        assert not cap.events("WORKER_DISCONNECTED"), (
            "a worker identity was evicted — the terminal must not be preceded "
            "by a shed")
        m = out.get("staging_backpressure") or {}
        assert m.get("emergency_events_acted_on") == 1, m
    return


def rxp2_arm4_terminal_reason_leads_with_the_cause():
    """ARM 4 — the terminal reason leads with
    `coordinator_inbound_saturation_timeout:` per the Part-B convention, names
    the blocked SITE, and the exit record is present in the emergency payload.

    WRONG INPUT THAT REDS IT: a bare terminal that names a consequence instead of
    a cause — which is precisely how attempts 2 and 5 reported themselves."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord = _coord(tmp)
        state = ConnState(coord.next_connection_id("runR"))
        rec = ReaderExit(reason=RX_INFRASTRUCTURE_TERMINAL_EXIT,
                         connection_id=state.connection_id, at=1.0)
        ev = coord.raise_infrastructure_terminal(
            None, TC_INBOUND_SATURATION_TIMEOUT, where="reader_eof",
            state=state, spent=180.0, exit_record=rec, budget_s=180.0,
            worker_id="hostA:gpu0")
    reason = coord.inbound_saturation_terminal_reason(ev)
    assert reason.startswith("coordinator_inbound_saturation_timeout:"), reason
    assert "blocked_site=reader_eof" in reason, reason
    assert state.connection_id in reason, reason
    assert ev["reader_exit"]["reader_exit_reason"] == \
        RX_INFRASTRUCTURE_TERMINAL_EXIT, ev
    assert ev["reader_exit"]["connection_id"] == state.connection_id, ev


def _enclosing_loop(tree: ast.AST, target: ast.AST) -> bool:
    """True iff `target` lies inside a `for`/`while` within `tree`."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if child is target:
                    return True
    return False


def rxp2_arm5_emergency_cardinality():
    """ARM 5 — `emergency_events_total <= reader_threads_created`,
    `emergency_events_acted_on <= 1`, and an AST arm asserting each emergency
    emit site is the last statement on its path (CARD-1).

    WRONG INPUT THAT REDS IT: putting an emergency emit inside a retry loop — the
    cardinality argument then collapses and one wedged reader could flood the
    channel."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b = _saturated_bench(tmp, budget=0.3, maxsize=1)
        p = b.peers["hostA:gpu0"]
        p.send(_inline_result("hostA:gpu0", "s0", 0, 0, 10))
        p.join(timeout=15.0)
        m = b.coord.staging_backpressure_metrics()
        b.close()
    assert m["emergency_events_total"] == 1, (
        f"one reader raised {m['emergency_events_total']} emergency events; "
        f"CARD-1 bounds it at ONE per reader thread")
    assert m["emergency_events_acted_on"] == 0, m

    # CARD-1, structurally: every call is immediately followed by a control-flow
    # exit, so no path can return to it.
    tree = ast.parse(_live_src())
    sites = 0
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body):
            if not isinstance(stmt, ast.Expr):
                continue
            call = stmt.value
            if (isinstance(call, ast.Call)
                    and getattr(call.func, "attr", None)
                    == "raise_infrastructure_terminal"):
                sites += 1
                nxt = body[i + 1] if i + 1 < len(body) else None
                # CARD-1 verbatim: "each of which is immediately followed by
                # `return` from the reader's exit path — NO LOOP ENCLOSES
                # EITHER". Both halves bound the site to one execution, and each
                # of the two live sites satisfies one of them: the EOF retry
                # returns immediately, and the reader's tail emit is outside
                # every loop. A site that satisfies NEITHER could run twice.
                returns_now = isinstance(nxt, (ast.Return, ast.Break))
                enclosed = _enclosing_loop(tree, stmt)
                assert returns_now or not enclosed, (
                    f"an emergency emit at line {stmt.lineno} is inside a loop "
                    f"AND is not immediately followed by a return/break — "
                    f"CARD-1's 'at most one event per reader thread' no longer "
                    f"holds structurally")
    assert sites >= 1, "no emergency emit site found in live source"


def rxp2_arm6_one_accumulator_never_reset():
    """ARM 6 — [R2.1] ONE accumulator, never reset.

    (a) AST: exactly one assignment site for
        `ConnState.inbound_saturation_spent_s` outside `__init__`, and it is
        inside `charge_inbound_saturation`.
    (b) BEHAVIOURAL: saturate -> recover -> saturate on ONE connection, and the
        second episode INHERITS the first episode's spend, so the terminal fires
        at cumulative `S` rather than at `2S`.
    (c) the reasoned-EOF path charges that same value and initialises nothing.

    WRONG INPUT THAT REDS IT: E-1's `spent = 0.0`, or a reset on successful
    delivery — arm (b) then fires at `2S` and the budget is per-episode wearing a
    cumulative name."""
    # (a) structural
    tree = ast.parse(_live_src())
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if (isinstance(t, ast.Attribute)
                    and t.attr == "inbound_saturation_spent_s"):
                sites.append(node.lineno)
    assert len(sites) == 2, (
        f"{len(sites)} assignment sites for the accumulator (expected exactly "
        f"two: creation in ConnState.__init__ and the single writer)")
    charge = _func_node(tree, "RangeMinerCoordinator", "charge_inbound_saturation")
    init = _func_node(tree, "ConnState", "__init__")
    owned = {n.lineno for fn in (charge, init) for n in ast.walk(fn)
             if isinstance(n, ast.Assign)}
    assert set(sites) <= owned, (
        f"an assignment to the accumulator lives outside "
        f"`charge_inbound_saturation`/`ConnState.__init__`: {sites}")

    # (b) behavioural — one connection, two episodes, one budget
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            b = _saturated_bench(tmp, budget=1.2, maxsize=1)
            p = b.peers["hostA:gpu0"]
            state = p.state
            p.send(_inline_result("hostA:gpu0", "s0", 0, 0, 10))
            assert _wait(lambda: state.inbound_saturation_spent_s >= 0.5,
                         timeout=8.0), "the first episode never accrued spend"
            first = state.inbound_saturation_spent_s
            b.inbound.get()                      # ingress recovers
            assert _wait(lambda: b.inbound.qsize() >= 1, timeout=5.0), (
                "the held envelope was never delivered after recovery")
            after_success = state.inbound_saturation_spent_s
            assert after_success >= first, (
                f"a SUCCESSFUL delivery reduced the accumulator: {first} -> "
                f"{after_success}; `S` is cumulative, and 'the last put "
                f"succeeded' is not evidence that ingress has recovered")
            # SECOND EPISODE, on the same connection. The delivered envelope is
            # taken out first: a `put` onto a still-full bounded queue would
            # block THIS thread rather than the reader's, which is a harness
            # deadlock wearing the costume of the condition under test.
            b.inbound.get()
            b.inbound.put(("msg", None, None, None, None))
            p.send(_inline_result("hostA:gpu0", "s0", 1, 10, 10))
            p.join(timeout=20.0)
            b.close()
        ev = _reader_exit_event(cap)
    assert ev["reader_exit_reason"] == RX_INFRASTRUCTURE_TERMINAL_EXIT, ev
    total = state.inbound_saturation_spent_s
    assert total < 2 * 1.2, (
        f"the terminal fired at {total:.2f}s against a budget of 1.2s — the "
        f"second episode did not inherit the first's spend, i.e. the "
        f"accumulator was reset")

    # (c) the EOF path initialises nothing
    eof_fn = _func_node(tree, "RangeMinerCoordinator", "_deliver_reader_eof")
    for node in ast.walk(eof_fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                assert not (isinstance(t, ast.Attribute)
                            and t.attr == "inbound_saturation_spent_s"), (
                    "the reasoned-EOF path assigns the accumulator — it must "
                    "only ever CHARGE it")
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "spent" for t in node.targets):
            assert not (isinstance(node.value, ast.Constant)
                        and node.value.value == 0.0), (
                "E-1's `spent = 0.0` is back: the EOF gets a FRESH budget")


def rxp2_arm7_pause_time_is_not_charged():
    """ARM 7 — a reader parked on STAGING capacity accrues zero saturation spend.

    WRONG INPUT THAT REDS IT: double-charging a coordinator-caused staging wait
    into an ingress terminal — Beta D's classification law violated at trial
    scope."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b = _Bench(tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"),
                   saturation_budget_s=0.5, **_saturating_cfg())
        _pause_one(b)
        time.sleep(1.5)               # three budgets' worth of PAUSE
        state = b.peers["hostA:gpu0"].state
        spent = state.inbound_saturation_spent_s
        m = b.coord.staging_backpressure_metrics()
        alive = b.peers["hostA:gpu0"].thread.is_alive()
        b.close()
    assert spent == 0.0, (
        f"a paused reader accrued {spent}s of INGRESS saturation spend")
    assert m["inbound_saturation_events"] == 0, m
    assert alive, "a paused reader was terminated by the ingress budget"


def rxp2_arm8_occupancy_is_sampled_during_the_drain():
    """ARM 8 — `inbound_qsize_high_water` reflects a depth sampled DURING the
    drain, not only at the top-of-loop instant before it (Q-4).

    During attempt 5's 940.971 s iteration the sample was taken exactly ONCE, so
    the gap between the reported 979 and the hard 1024 was unmeasured for 940
    seconds.

    WRONG INPUT THAT REDS IT: restoring the once-per-iteration sample."""
    tree = ast.parse(_live_src())
    serve = _func_node(tree, "RangeMinerCoordinator", "serve_trial")
    drain = _drain_while(serve)
    assert drain is not None
    inside = [n for n in ast.walk(drain)
              if isinstance(n, ast.Call)
              and getattr(n.func, "attr", None) == "note_inbound_occupancy"]
    assert inside, (
        "occupancy is sampled only before the drain — during a long drain the "
        "high-water is a statement about the moment the drain STARTED")
    # …and behaviourally: the sample rises while the drain is running.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord = _coord(tmp)
        coord.note_inbound_occupancy(3)
        coord.note_inbound_occupancy(7)
        coord.note_inbound_occupancy(5)
        assert coord.staging_backpressure_metrics()[
            "inbound_qsize_high_water"] == 7


def rxp2_arm9_the_charged_list_is_exhaustive():
    """ARM 9 — [R3.1] the charged/not-charged list of §8.2.4a is enforced.

    (a) THE FALSE-TERMINAL ARM: a REGISTER enters the admission queue, `inbound`
        stays NON-FULL, and its disposition is deliberately delayed past `S`.
        Assert the accumulator is UNCHANGED and no terminal fires.
    (b) the same for the pre-decode credit barrier.
    (c) AST: `charge_inbound_saturation` is called from exactly two sites, both
        immediately handling `_queue.Full` on `inbound`.

    WRONG INPUT THAT REDS IT: charging the register fence, the pre-decode
    barrier, admission residence or registration ledger time to `S` — arm (a)
    then fires an INGRESS terminal on a coordinator whose `inbound` was never
    full, inverting the classification the terminal exists to establish."""
    # (a) the register fence
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            b = _Bench(tmp, worker_ids=("hostA:gpu0",), bind=False,
                       saturation_budget_s=0.4)
            p = b.peers["hostA:gpu0"]
            p.send(RegisterMessage(
                worker_id="hostA:gpu0", hostname="hostA", gpu_id=0,
                gpu_name="x", backend="cuda", vram_bytes=1,
                capabilities={"seed_caps": dict(CAPS),
                              "supported_variants": list(VARIANTS)}))
            # the frame reaches the ADMISSION channel and the reader parks on the
            # fence; nobody disposes of it.
            assert _wait(lambda: b.admission.qsize() == 1, timeout=5.0), (
                "the first-frame REGISTER did not take the admission channel")
            time.sleep(1.2)                     # three budgets' worth of fence
            assert not b.inbound.full(), (
                "the arm is vacuous: `inbound` is full, so a charge would be "
                "legitimate ingress saturation")
            spent = p.state.inbound_saturation_spent_s
            alive = p.thread.is_alive()
            b.close()
        assert spent == 0.0, (
            f"the REGISTER disposition fence charged {spent}s to `S` — a slow "
            f"registration ledger write would then manufacture an "
            f"`inbound_saturation_timeout` on a coordinator whose `inbound` was "
            f"never full")
        assert alive, "the fence terminated the reader on an ingress budget"
        assert not cap.events("EMERGENCY_TERMINAL_REQUEST"), (
            "a disposition wait raised an INGRESS terminal")

    # (b) the pre-decode credit barrier
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b = _Bench(tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"),
                   saturation_budget_s=0.4, **_saturating_cfg())
        _park_at_barrier(b)
        time.sleep(1.2)
        spent = b.peers["hostA:gpu0"].state.inbound_saturation_spent_s
        alive = b.peers["hostA:gpu0"].thread.is_alive()
        b.close()
    assert spent == 0.0, (
        f"the pre-decode barrier charged {spent}s to `S` — it is a DISPOSITION "
        f"wait, not an INGRESS wait")
    assert alive

    # (c) exactly two charging sites, both on a `_queue.Full` handler
    tree = ast.parse(_live_src())
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for h in node.handlers:
            h_src = ast.unparse(h)
            if "charge_inbound_saturation" in h_src:
                assert "Full" in ast.unparse(h.type), (
                    f"a charge site handles {ast.unparse(h.type)}, not "
                    f"`_queue.Full`")
                assert "inbound.put" in ast.unparse(node.body), (
                    "a charge site is not attached to an `inbound.put`")
                calls.append(h.lineno)
    total = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "charge_inbound_saturation"]
    assert len(calls) == 2 and len(total) == 2, (
        f"{len(total)} calls to `charge_inbound_saturation` ({len(calls)} of "
        f"them on a `_queue.Full` handler of an `inbound.put`); §8.2.4a permits "
        f"exactly two, and every other waiter is excluded BY CONSTRUCTION")


def rxp2_clean_control():
    """CLEAN CONTROL — the same traffic BELOW saturation produces zero saturation
    spend, zero emergency events and zero exits."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            b = _Bench(tmp, worker_ids=("hostA:gpu0",), maxsize=64,
                       saturation_budget_s=5.0)
            p = b.peers["hostA:gpu0"]
            for i in range(8):
                p.send(_inline_result("hostA:gpu0", "s0", i, i * 10, 10))
            assert _wait(lambda: b.inbound.qsize() >= 8, timeout=5.0), (
                "the control traffic never arrived")
            m = b.coord.staging_backpressure_metrics()
            alive = p.thread.is_alive()
            spent = p.state.inbound_saturation_spent_s
            # SAMPLED BEFORE THE TEARDOWN, and that distinction is the arm: the
            # bench's own `close()` sets `reader_stop`, which legitimately exits
            # the reader as SHUTDOWN_STOP. Asserting "no READER_EXIT" after it
            # would be asserting that a deliberate shutdown produces no record —
            # the opposite of what RXP-1 arm 1 requires — and would pass only by
            # winning a race with the reader thread.
            exits_before_teardown = cap.events("READER_EXIT")
            b.close()
        assert not exits_before_teardown, (
            f"a healthy reader exited while it was being fed below saturation: "
            f"{exits_before_teardown}")
    assert (m["inbound_saturation_events"], m["emergency_events_total"]) == (0, 0)
    assert spent == 0.0 and alive


# ===========================================================================
# FAIR-7 — reasoned EOF: ordered, and never lost (§11.7, 7 arms)
# ===========================================================================
def fair7_arm1_eof_is_ordered_behind_its_own_envelopes():
    """ARM 1 — a reader delivers K result envelopes then exits; ALL K are
    dispatched before `_drop_conn` runs for that socket, and NONE is discarded on
    the `rawsock not in fs_by_sock` path.

    WRONG INPUT THAT REDS IT: routing the exit record to the emergency or
    admission channel — i.e. Q-3, whose control queue is reaped ahead of queued
    envelopes and therefore DISCARDS work the coordinator had already credited."""
    K = 6
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b = _Bench(tmp, worker_ids=("hostA:gpu0",), maxsize=64)
        p = b.peers["hostA:gpu0"]
        for i in range(K):
            p.send(_inline_result("hostA:gpu0", "s0", i, i * 10, 10))
        assert _wait(lambda: b.inbound.qsize() >= K, timeout=5.0)
        p.break_transport()
        p.join()
        entries = []
        while True:
            try:
                entries.append(b.inbound.get_nowait())
            except _queue.Empty:
                break
        b.close()
    kinds = [e[0] for e in entries]
    assert kinds.count("msg") == K, f"{kinds.count('msg')} of {K} envelopes"
    assert kinds[-1] == "eof", (
        f"the eof did not arrive LAST: {kinds} — P-ORD is the property the "
        f"whole same-FIFO design rests on")
    assert kinds.index("eof") == K, kinds


def fair7_arm2_eof_survives_a_full_queue():
    """ARM 2 — with `inbound` held at `maxsize` for LONGER than 0.5 s but within
    `S`, the reasoned EOF is STILL delivered and its reason reaches
    `WORKER_DISCONNECTED`.

    WRONG INPUT THAT REDS IT: restoring `timeout=0.5` + `except Exception: pass`
    — the EOF is then lost, and for a REGISTERED connection nothing reaps it: a
    zombie whose identity still counts toward `_eligible()`."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            q = _queue.Queue(maxsize=2)
            q.put(("msg", None, None, None, None))
            q.put(("msg", None, None, None, None))
            b = _Bench(tmp, worker_ids=("hostA:gpu0",), inbound=q,
                       saturation_budget_s=30.0)
            p = b.peers["hostA:gpu0"]
            time.sleep(0.05)
            p.break_transport()
            # the reader is now retrying its reasoned EOF against a FULL queue,
            # for longer than the deleted 0.5 s ever waited
            time.sleep(1.0)
            assert p.thread.is_alive(), (
                "the reader gave up on the EOF while budget remained")
            q.get(); q.get()                    # ingress recovers
            p.join(timeout=10.0)
            got = [q.get(timeout=2.0)]
            assert got[0][0] == "eof", got
            record = got[0][4]
            assert isinstance(record, ReaderExit), record
            b.coord._drop_conn(
                p.srv, {p.srv: p.srv_fs}, {p.srv: "hostA:gpu0"},
                {"hostA:gpu0": p.srv_fs}, dict(b.wconn_by_worker),
                ["hostA:gpu0"], stage_idx=0, stage_assigned=True,
                eligible_fn=lambda: [], conn_state=p.state, reader_exit=record)
            b.close()
        wd = cap.events("WORKER_DISCONNECTED")
    assert wd and wd[0]["reader_exit_reason"] == RX_TRANSPORT_ERROR, wd


def fair7_arm3_non_full_exception_propagates():
    """ARM 3 — a non-`Full` exception at the EOF put is logged at ERROR and
    PROPAGATES; the suite observes it.

    WRONG INPUT THAT REDS IT: any blanket `except Exception: pass` — a genuine
    bug then vanishes with the record it was carrying."""
    class _HostileQueue(_queue.Queue):
        def put(self, item, *a, **kw):
            if item[0] == "eof":
                raise RuntimeError("injected non-Full failure")
            return super().put(item, *a, **kw)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog(level=logging.ERROR) as cap:
            b = _Bench(tmp, worker_ids=("hostA:gpu0",),
                       inbound=_HostileQueue(maxsize=8))
            p = b.peers["hostA:gpu0"]
            time.sleep(0.05)
            p.break_transport()
            p.join()
            b.close()
        assert p.exc and isinstance(p.exc[0], RuntimeError), (
            f"the non-Full failure did not propagate out of the reader: {p.exc}")
        assert cap.events("READER_EOF_UNDELIVERABLE"), (
            "the undeliverable EOF was not recorded at ERROR")


def fair7_arm4_the_zombie_is_closed():
    """ARM 4 — the §1.5 zombie is closed: a REGISTERED connection whose reader has
    exited is reaped, and its identity leaves `_eligible()`.

    WRONG INPUT THAT REDS IT: no retry — the eof is swallowed, the read-deadline
    branch `continue`s on `meta["registered"]`, and the identity survives in
    `_eligible()` while never delivering another frame."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        q = _queue.Queue(maxsize=1)
        q.put(("msg", None, None, None, None))
        b = _Bench(tmp, worker_ids=("hostA:gpu0",), inbound=q,
                   saturation_budget_s=30.0)
        p = b.peers["hostA:gpu0"]
        wconn = dict(b.wconn_by_worker)
        fs_by_worker = {"hostA:gpu0": p.srv_fs}
        eligible = lambda: [w for w in wconn.values() if not w.quarantined]
        assert len(eligible()) == 1
        time.sleep(0.05)
        p.break_transport()
        time.sleep(0.8)                       # the EOF is retrying, not lost
        q.get()                               # ingress recovers
        p.join(timeout=10.0)
        entry = q.get(timeout=3.0)
        assert entry[0] == "eof", entry
        b.coord._drop_conn(p.srv, {p.srv: p.srv_fs}, {p.srv: "hostA:gpu0"},
                           fs_by_worker, wconn, ["hostA:gpu0"],
                           stage_idx=0, stage_assigned=True,
                           eligible_fn=eligible, conn_state=p.state,
                           reader_exit=entry[4])
        b.close()
    assert eligible() == [], (
        "the identity survived in the eligible pool — the zombie is back")


def fair7_arm5_exhaustion_terminal_without_eviction():
    """ARM 5 — holding past `S` fail-closes the trial with
    `inbound_saturation_timeout`, the worker is NOT individually evicted first,
    and the exit record is in the emergency payload.

    WRONG INPUT THAT REDS IT: shedding the worker instead — an eviction then
    precedes the terminal."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            q = _queue.Queue(maxsize=1)
            q.put(("msg", None, None, None, None))
            b = _Bench(tmp, worker_ids=("hostA:gpu0",), inbound=q,
                       saturation_budget_s=0.4)
            p = b.peers["hostA:gpu0"]
            time.sleep(0.05)
            p.break_transport()
            p.join(timeout=15.0)
            b.close()
        emg = cap.events("EMERGENCY_TERMINAL_REQUEST")
        assert len(emg) == 1, f"{len(emg)} emergency requests"
        assert emg[0]["terminal_class"] == TC_INBOUND_SATURATION_TIMEOUT, emg[0]
        assert emg[0]["where"] == "reader_eof", emg[0]
        assert emg[0]["reader_exit"]["reader_exit_reason"] == RX_TRANSPORT_ERROR, (
            f"the exit record did not travel inside the terminal event: "
            f"{emg[0]}")
        assert not cap.events("WORKER_DISCONNECTED"), (
            "a worker was evicted before the trial terminal")


def fair7_arm6_eof_inherits_the_spend():
    """ARM 6 — [R2.1] the EOF path initialises NO budget of its own: a connection
    that spent 0.8·S delivering ordinary envelopes and then meets a full queue at
    exit reaches the terminal after a further 0.2·S, NOT after a further S.

    This is the direct falsifier of the contradiction R2.1 names.

    WRONG INPUT THAT REDS IT: E-1's `spent = 0.0`."""
    # `S` is chosen so the 0.25 s retry quantum can land inside [0.6 S, S) — a
    # budget of one or two quanta could not distinguish "inherited" from "fresh"
    # at all, and a gate that cannot distinguish them proves nothing.
    S = 3.0
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        q = _queue.Queue(maxsize=1)
        q.put(("msg", None, None, None, None))
        b = _Bench(tmp, worker_ids=("hostA:gpu0",), inbound=q,
                   saturation_budget_s=S)
        p = b.peers["hostA:gpu0"]
        p.send(_inline_result("hostA:gpu0", "s0", 0, 0, 10))
        # burn ~0.7 S on ORDINARY delivery
        assert _wait(lambda: p.state.inbound_saturation_spent_s >= 0.7 * S,
                     timeout=15.0, interval=0.005), (
            "the ordinary path never accrued spend")
        pre_eof = p.state.inbound_saturation_spent_s
        assert pre_eof < S, "the ordinary path exhausted the budget by itself"
        t0 = time.perf_counter()
        p.break_transport()
        p.join(timeout=15.0)
        elapsed = time.perf_counter() - t0
        b.close()
    assert elapsed < S, (
        f"the exit path took {elapsed:.2f}s against a budget of {S}s with "
        f"{pre_eof:.2f}s already spent — the EOF was given a FRESH budget")


def fair7_clean_control():
    """CLEAN CONTROL — a connection delivering K envelopes and exiting with an
    EMPTY queue produces zero saturation spend, zero emergency events, and ONE
    in-order EOF."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b = _Bench(tmp, worker_ids=("hostA:gpu0",), maxsize=64)
        p = b.peers["hostA:gpu0"]
        for i in range(3):
            p.send(_inline_result("hostA:gpu0", "s0", i, i * 10, 10))
        assert _wait(lambda: b.inbound.qsize() >= 3, timeout=5.0)
        p.break_transport()
        p.join()
        entries = []
        while True:
            try:
                entries.append(b.inbound.get_nowait())
            except _queue.Empty:
                break
        m = b.coord.staging_backpressure_metrics()
        spent = p.state.inbound_saturation_spent_s
        b.close()
    assert [e[0] for e in entries] == ["msg", "msg", "msg", "eof"], entries
    assert spent == 0.0
    assert (m["inbound_saturation_events"], m["emergency_events_total"]) == (0, 0)


def fair7_red_eof_is_lost_on_pinned_source():
    """RED — on pinned `2b0d2dc`, arm 2 reds: the EOF is genuinely LOST under a
    full queue.

    A drifted anchor terminates UNAVAILABLE, never PASS."""
    _assert_pinned_carries_the_defects()
    src = _pinned_src()
    fn = _func_node(ast.parse(src), "RangeMinerCoordinator", "_conn_reader_loop")
    body = textwrap.dedent(ast.get_source_segment(src, fn))
    g = dict(vars(COORD))
    exec(compile(body, f"<pinned {PINNED_COMMIT}>", "exec"), g)      # noqa: S102
    pinned_reader = g["_conn_reader_loop"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord = _coord(tmp)
        q = _queue.Queue(maxsize=1)
        q.put(("msg", None, None, None))
        stop = threading.Event()
        sp = socket.socketpair()
        fs = MinerFramedSocket(sp[0])
        th = threading.Thread(target=types.MethodType(pinned_reader, coord),
                              args=(fs, sp[0], q, stop, {}), daemon=True)
        th.start()
        time.sleep(0.05)
        try:
            sp[1].shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sp[1].close()
        th.join(timeout=5.0)
        assert not th.is_alive(), "the pinned reader never exited"
        # the pinned reader waited 0.5 s and then swallowed the failure
        time.sleep(0.7)
        q.get()                                  # ingress recovers — too late
        time.sleep(0.5)
        residual = []
        while True:
            try:
                residual.append(q.get_nowait())
            except _queue.Empty:
                break
        for s in sp:
            try:
                s.close()
            except OSError:
                pass
    assert not any(e[0] == "eof" for e in residual), (
        "the PINNED reader delivered its eof under a full queue — the lost-EOF "
        "surface is absent from the anchor and no RED arm may be credited")


# ===========================================================================
# RXP-3 — worker-log sentinel and the pre-REGISTER barrier (§11.3, 8 arms)
#
# OPERATIVE CRITERION: every eligible worker proves session-log delivery BEFORE
# the production session begins, and NO worker can send REGISTER until the 25/25
# gate has passed.
#
# Arms 1-5 prove DELIVERY. Arms 6-8 prove ORDERING. v1 had only the first kind,
# which is why Beta called it an intention rather than an enforced property.
# ===========================================================================
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
import gate12_sentinel_gate as SG                                   # noqa: E402


def _sentinel_target(worker_id, gpu=0):
    """A LOCAL target, so the probe runs its real command against a real file
    without needing the fleet. The remote branch differs only in the ssh
    prefix — the classification, the rendering and the refusal are the same
    code."""
    host, _, _g = worker_id.partition(":gpu")
    return {"node_id": host, "endpoint": "localhost", "ssh_user": "michael",
            "local": True, "gpu": gpu, "worker_hostname": host,
            "worker_id": worker_id}


def _bare_worker(worker_id="hostA:gpu0"):
    w = WORKER.RangeMinerWorker.__new__(WORKER.RangeMinerWorker)
    w.worker_id = worker_id
    w.session_generation = 1
    w.session_events = []
    w._stop = threading.Event()
    return w


def _sentinel_records(nonce, worker_id="hostA:gpu0"):
    """Log lines in the EXACT shape `_emit_session_event` produces, built by
    calling that function rather than by hand — a fabricated line would prove
    the probe reads a format nothing writes."""
    w = _bare_worker(worker_id)
    with _CapturedWorkerLog() as cap:
        WORKER.RangeMinerWorker._emit_session_event(
            w, "SESSION_SENTINEL", run_nonce=nonce, pid=1234)
    return list(cap.records)


def _release_wait_records(nonce, worker_id="hostA:gpu0"):
    """[R1-A] THE OTHER RECORDS THAT CARRY A RUN NONCE, produced by the REAL
    barrier rather than invented.

    This is the whole point of the split-fact arms: the worker enters
    `await_session_release` IMMEDIATELY after emitting the sentinel, and that
    method emits `SESSION_RELEASE_WAIT` (and, on expiry, `SESSION_RELEASE_TIMEOUT`)
    carrying `run_nonce` — so "this nonce appears somewhere in the log" is
    satisfied by a record that is NOT a sentinel, on a perfectly ordinary run.
    Driven against a release path that never appears, with a short deadline, so
    the records are the ones production writes."""
    w = _bare_worker(worker_id)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedWorkerLog() as cap:
            try:
                WORKER.RangeMinerWorker.await_session_release(
                    w, os.path.join(tmp, "never-appears"), nonce,
                    0.15, poll_s=0.05)
            except WORKER.SessionReleaseTimeout:
                pass
    records = list(cap.records)
    assert records and all("SESSION_SENTINEL" not in r for r in records), (
        f"the release-barrier records must carry NO sentinel event, else the "
        f"split-fact arms would be testing nothing: {records}")
    assert any(nonce in r for r in records), (
        f"the release-barrier records do not carry the run nonce, so the "
        f"split-fact premise does not hold: {records}")
    return records


def _write_log_file(path, *record_lists):
    """A worker log: the three `Compiled kernel` lines every attempt-5 rig log
    carried, plus whichever real session records the arm supplies."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("[sieve_worker] Compiled kernel: java_lcg\n")
        for records in record_lists:
            for line in records:
                fh.write(line + "\n")


def _write_sentinel_log(path, nonce, worker_id="hostA:gpu0"):
    _write_log_file(path, _sentinel_records(nonce, worker_id))


class _CapturedWorkerLog(_CapturedLog):
    def __enter__(self):
        outer = self

        class _Cap(logging.Handler):
            def emit(self, record):
                outer.records.append(record.getMessage())

        # The worker module names its logger `__name__`, i.e.
        # `miner.range_miner_worker` — reading it off the module rather than
        # retyping it is the difference between capturing the real channel and
        # capturing a logger nothing writes to.
        self._lg = logging.getLogger(WORKER.logger.name)
        self._prev = (self._lg.level, self._lg.propagate)
        self._h = _Cap()
        self._lg.addHandler(self._h)
        self._lg.setLevel(logging.INFO)
        self._lg.propagate = False
        return self


def rxp3_arm1_all_present_passes():
    """ARM 1 — all present => PASS, the run proceeds. (Clean control.)"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        nonce = "nonce-arm1"
        targets = [_sentinel_target("hostA:gpu0", 0),
                   _sentinel_target("hostA:gpu1", 1)]
        results = []
        for t in targets:
            path = SG.log_path_for(t, "/tmp/minerlogs", tmp)
            _write_sentinel_log(path, nonce, t["worker_id"])
            results.append(SG.probe_sentinel(t, nonce, path))
        allowed, refusals = SG.evaluate(results)
    assert allowed and not refusals, (results, refusals)
    assert all(r["status"] == SG.PROBE_OK and r["count"] >= 1 for r in results)


def rxp3_arm2_missing_nonce_refuses():
    """ARM 2 — one identity's log lacks the nonce => REFUSAL naming the identity:
    fleet killed, run aborted, NO reduced cohort and NO automatic downsizing.

    WRONG INPUT THAT REDS IT: an advisory `checks_passed += 1` treatment — the
    shape that let `GPU_COUNT_MISMATCH: 0/8` through a 3/3 preflight."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        nonce = "nonce-arm2"
        good = _sentinel_target("hostA:gpu0", 0)
        bad = _sentinel_target("hostA:gpu1", 1)
        gp = SG.log_path_for(good, "/tmp/minerlogs", tmp)
        bp = SG.log_path_for(bad, "/tmp/minerlogs", tmp)
        _write_sentinel_log(gp, nonce, good["worker_id"])
        with open(bp, "w", encoding="utf-8") as fh:
            fh.write("[sieve_worker] Compiled kernel: java_lcg\n")
        results = [SG.probe_sentinel(good, nonce, gp),
                   SG.probe_sentinel(bad, nonce, bp)]
        allowed, refusals = SG.evaluate(results)
    assert not allowed, "a missing sentinel was allowed to proceed"
    assert len(refusals) == 1 and "hostA:gpu1" in refusals[0], refusals
    assert SG.main([
        "--phase", "verify", "--run-nonce", nonce, "--local-log-dir", "/nope",
    ]) == SG.EXIT_REFUSE, "the gate's exit status does not refuse"


def rxp3_arm3_unavailable_is_never_zero():
    """ARM 3 — **an unreadable log file** => UNAVAILABLE => REFUSAL, rendered as
    UNAVAILABLE and NEVER as `0`.

    [R2 — DECLARATION CORRECTED] This arm previously declared *"ssh fails / file
    unreadable"* and exercised only the LOCAL unreadable-file half. There is no
    ssh in it, and the ssh half was not merely untested: the gate did not
    implement it, classifying a transport failure as ERROR. **The ssh half is now
    arms 12 and 13**, and this docstring states what this arm actually drives —
    a declaration wider than the exercise is the whole failure class.

    WRONG INPUT THAT REDS IT: rendering an unobserved probe as a definite zero —
    two constructs each manufactured that `0` in attempt 1, and a `0/8` and an
    `UNAVAILABLE` are different facts about the world."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        t = _sentinel_target("hostA:gpu0", 0)
        missing = os.path.join(tmp, "does-not-exist.log")
        r = SG.probe_sentinel(t, "nonce-arm3", missing)
    assert r["status"] == SG.PROBE_UNAVAILABLE, r
    assert r["count"] is None, f"an unavailable probe reported a count: {r}"
    line = SG.render(r)
    assert "UNAVAILABLE" in line and "0/" not in line, (
        f"an unavailable probe rendered count-shaped: {line!r}")
    allowed, refusals = SG.evaluate([r])
    assert not allowed and "UNKNOWN, not absent" in refusals[0], refusals


def rxp3_arm4_stale_nonce_fails():
    """ARM 4 — a log containing a PREVIOUS run's nonce FAILS, and the same nonce
    test applies to the RELEASE FILE.

    THIS IS THE ARM THAT MAKES THE GATE NON-VACUOUS: without it, a leftover
    `/tmp/minerlogs/gpuN.log` from any earlier launch satisfies a naive
    "file is non-empty" check.

    WRONG INPUT THAT REDS IT: testing for the presence of a sentinel rather than
    of THIS run's sentinel."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        t = _sentinel_target("hostA:gpu0", 0)
        path = SG.log_path_for(t, "/tmp/minerlogs", tmp)
        _write_sentinel_log(path, "nonce-PREVIOUS-run", t["worker_id"])
        r = SG.probe_sentinel(t, "nonce-THIS-run", path)
        assert r["status"] == SG.PROBE_OK and r["count"] == 0, r
        allowed, _ = SG.evaluate([r])
        assert not allowed, "a stale nonce satisfied the gate"

        # …and the release file: a token carrying another run's nonce is NOT a
        # release, and the worker fails CLOSED rather than proceeding.
        rel = os.path.join(tmp, "release")
        with open(rel, "w", encoding="utf-8") as fh:
            fh.write("nonce-PREVIOUS-run")
        w = WORKER.RangeMinerWorker.__new__(WORKER.RangeMinerWorker)
        w.worker_id, w.session_generation, w.session_events = "hostA:gpu0", 1, []
        w._stop = threading.Event()
        try:
            w.await_session_release(rel, "nonce-THIS-run", 0.6, poll_s=0.05)
            raise AssertionError("a STALE release token released the worker")
        except WORKER.SessionReleaseTimeout:
            pass
        evs = [e for e in w.session_events
               if e["event"] == "SESSION_RELEASE_TIMEOUT"]
        assert evs and evs[0]["observed"] == "nonce-PREVIOUS-run", evs


def rxp3_arm5_sentinel_routes_through_the_generic_emitter():
    """ARM 5 — [R3.5] `emit_startup_sentinel` MUST CALL
    `_emit_session_event("SESSION_SENTINEL", …)` — it ROUTES THROUGH the generic
    structured session logger, which is what proves the sentinel exercises the
    same channel the session events use.

    AND THE INVERSE: pushing sentinel-specific logic INTO `_emit_session_event`
    to satisfy an imprecise wording is NOT the fix. The emitter stays generic;
    the CALLER is what is gated.

    WRONG INPUT THAT REDS IT: a sentinel that `print()`s, or one on a different
    logger — it passes a file-content check while proving nothing about the
    channel that was silent."""
    tree = ast.parse(_live_src(WORKER_SRC_REL))
    fn = _func_node(tree, "RangeMinerWorker", "emit_startup_sentinel")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    routed = [c for c in calls
              if getattr(c.func, "attr", None) == "_emit_session_event"
              and c.args and isinstance(c.args[0], ast.Constant)
              and c.args[0].value == "SESSION_SENTINEL"]
    assert len(routed) == 1, (
        f"{len(routed)} calls route the sentinel through the generic session "
        f"emitter; expected exactly one")
    for banned in ("print", "write"):
        assert not any(getattr(c.func, "id", None) == banned for c in calls), (
            f"the sentinel calls {banned}() — that is a different channel from "
            f"the one that was silent")
    emitter = _func_node(tree, "RangeMinerWorker", "_emit_session_event")
    assert "SESSION_SENTINEL" not in ast.unparse(emitter), (
        "sentinel-specific logic was pushed INTO the generic emitter; the "
        "emitter must stay generic and the caller is what is gated")


# --- arms 9-11: THE SAME-RECORD CONJUNCTION (R1-A) --------------------------
#
# Numbered 9-11 rather than inserted as 6-8 ON PURPOSE: arms 6-8 are the ORDERING
# arms and are cited by number in the R3 record and in Beta's rulings. Renumbering
# a certified arm to make a new one adjacent is how a governance citation quietly
# starts pointing at different code.
#
# WHAT THESE THREE ADD THAT ARMS 1-5 COULD NOT. Arms 1-5 prove the gate reacts to
# the sentinel and to the nonce. They cannot prove the gate requires them IN THE
# SAME RECORD, because none of them ever presents a log where the two facts are
# TRUE SEPARATELY. Arm 4 (stale nonce) is the nearest, and it is the near-miss
# worth naming: it presents an old sentinel and NO current nonce, so the old
# two-independent-counts predicate refused it for the RIGHT REASON BY ACCIDENT —
# the nonce count was zero. Split the two facts across two records and the same
# predicate passes.
# ---------------------------------------------------------------------------
def _split_fact_log(path, old_nonce, current_nonce, worker_id="hostA:gpu0"):
    """THE REACHABLE LOG. A stale sentinel from an earlier run, followed by THIS
    run's release-barrier records — which is what a rig log looks like when a
    worker relaunches into a directory it already wrote to and the current run's
    sentinel never lands."""
    _write_log_file(path,
                    _sentinel_records(old_nonce, worker_id),
                    _release_wait_records(current_nonce, worker_id))


def rxp3_arm9_split_fact_refuses():
    """ARM 9 — [R1-A] SPLIT FACT. A stale `SESSION_SENTINEL` carrying an OLD
    nonce, PLUS a current `SESSION_RELEASE_WAIT` carrying THIS run's nonce, in
    one log => REFUSE.

    The gate must prove "a SESSION_SENTINEL CARRYING the current nonce", not
    "a SESSION_SENTINEL exists somewhere AND the current nonce exists somewhere".
    This run's sentinel was never observed, so the property RXP-3 exists to
    establish does not hold and the run must not proceed.

    WRONG INPUT THAT REDS IT: two independent counts feeding acceptance — i.e.
    the pre-R1-A predicate, which reads `grep -c SESSION_SENTINEL` and
    `grep -c <nonce>` separately and accepts on the second. That exact mutation
    is applied below and MUST be detected."""
    def _run_probe(probe):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            t = _sentinel_target("hostA:gpu0", 0)
            path = SG.log_path_for(t, "/tmp/minerlogs", tmp)
            _split_fact_log(path, "nonce-OLD-run", "nonce-THIS-run")
            # VACUITY: the split fact must actually be present in the file, or
            # this arm is asserting a refusal it would get for free.
            body = open(path, encoding="utf-8").read()
            assert any("SESSION_SENTINEL" in ln and "nonce-OLD-run" in ln
                       for ln in body.splitlines()), body
            assert any("nonce-THIS-run" in ln and "SESSION_SENTINEL" not in ln
                       for ln in body.splitlines()), body
            r = probe(t, "nonce-THIS-run", path)
        assert r["status"] == SG.PROBE_OK, r
        assert r["count"] == 0, (
            f"acceptance counted {r['count']} — a sentinel and a nonce living "
            f"in DIFFERENT records were treated as a match: {r}")
        allowed, refusals = SG.evaluate([r])
        assert not allowed, "the split fact was allowed to proceed"
        assert "sentinel_present_but_none_carries_this_nonce" in refusals[0], (
            f"the refusal does not name the split-fact case: {refusals}")

    _run_probe(SG.probe_sentinel)
    # Built outside `_mutant_red`: it credits ANY exception as detection, so a
    # mutation that stopped matching live source would read as "detected".
    mutant = _two_independent_counts_probe()
    _mutant_red(lambda: _run_probe(mutant),
                "probe_sentinel reverted to two independent counts")


def rxp3_arm10_nonce_without_any_sentinel_refuses():
    """ARM 10 — [R1-A] NO SENTINEL AT ALL. A log carrying ONLY a current-nonce
    release-wait record — no sentinel of any run — => REFUSE.

    Arm 9's log at least contains a sentinel; this one contains none, and it is
    the shape a worker produces when the session-log channel is silent for the
    sentinel but the barrier's own records survive. Under two independent counts
    the sentinel count is 0 and the nonce count is nonzero, and acceptance read
    only the second, so THIS LOG PASSED.

    WRONG INPUT THAT REDS IT: the same two-independent-counts predicate, applied
    below as a mutant. Also red under any acceptance that tests the nonce alone,
    including "the file mentions this run"."""
    def _run_probe(probe):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            t = _sentinel_target("hostA:gpu0", 0)
            path = SG.log_path_for(t, "/tmp/minerlogs", tmp)
            _write_log_file(path, _release_wait_records("nonce-THIS-run"))
            body = open(path, encoding="utf-8").read()
            assert "SESSION_SENTINEL" not in body, body
            assert "nonce-THIS-run" in body, body
            r = probe(t, "nonce-THIS-run", path)
        assert r["status"] == SG.PROBE_OK, r
        assert r["count"] == 0, (
            f"acceptance counted {r['count']} on a log with no sentinel record "
            f"of any kind: {r}")
        assert r["sentinel_lines_any_nonce"] == 0, r
        allowed, refusals = SG.evaluate([r])
        assert not allowed, "a log with no sentinel at all was allowed"
        assert "no_sentinel_record_at_all" in refusals[0], refusals
        line = SG.render(r)
        assert "NO SENTINEL FOR THIS NONCE" in line, line

    _run_probe(SG.probe_sentinel)
    mutant = _two_independent_counts_probe()      # built outside — see arm 9
    _mutant_red(lambda: _run_probe(mutant),
                "probe_sentinel reverted to two independent counts")


def _two_independent_counts_probe():
    """The PRE-R1-A predicate, reconstructed from LIVE source by mutation rather
    than retyped, and executed with the gate module's REAL globals (the A8-B2
    rule: a verbatim copy that resolves its names in the TEST module's globals
    escapes the shims and the mutant survives)."""
    src = _live_src("scripts/gate12_sentinel_gate.py")
    fn = _func_node(ast.parse(src), None, "probe_sentinel")
    body = textwrap.dedent(ast.get_source_segment(src, fn))
    mutant = body.replace(
        """              f"grep 'SESSION_SENTINEL' {log_path} | grep -c '{nonce}' "\n"""
        """              f"| head -1; "\n""",
        """              f"grep -c '{nonce}' {log_path} | head -1; "\n""", 1)
    assert mutant != body, (
        "MUTANT NOT APPLIED — the conjunctive pipeline was not found in live "
        "source, so this arm proves nothing about its own detection power")
    g = dict(vars(SG))
    exec(compile(mutant, "<two-independent-counts probe_sentinel>",  # noqa: S102
                 "exec"), g)
    return g["probe_sentinel"]


def rxp3_arm11_acceptance_reads_only_the_conjunctive_count():
    """ARM 11 — [R1-A] STRUCTURE. The diagnostic count can never become an
    acceptance input.

    `sentinel_lines_any_nonce` exists so an operator can tell arm 9's refusal
    from arm 10's. It is exactly the kind of second number that decays into a
    second acceptance path, so the separation is asserted rather than intended:
    `evaluate()` reads ONLY `status` and `count`, and the probe produces exactly
    ONE conjunctive pipeline.

    WRONG INPUT THAT REDS IT: an `or r["sentinel_lines_any_nonce"]` added to the
    acceptance test — the shape R1-A was returned for, one refactor later."""
    src = _live_src("scripts/gate12_sentinel_gate.py")
    tree = ast.parse(src)
    ev = _func_node(tree, None, "evaluate")
    keys = {n.slice.value for n in ast.walk(ev)
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, str)}
    assert "sentinel_lines_any_nonce" not in keys, (
        "`evaluate()` reads the DIAGNOSTIC count — acceptance must derive from "
        "the conjunctive count alone")
    # …and the acceptance branch is the conjunctive count, positively.
    assert "count" in keys and "status" in keys, keys
    probe = _func_node(tree, None, "probe_sentinel")
    # The `script` assignment ALONE — not every string in the function. The
    # docstring quotes the pipeline it documents, so a naive constant sweep reads
    # the prose and reports two.
    scripts = [n for n in ast.walk(probe)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and getattr(n.targets[0], "id", None) == "script"]
    assert len(scripts) == 1, (
        f"expected exactly one `script` assignment in the probe, found "
        f"{len(scripts)}")
    shell = ast.unparse(scripts[0])
    assert shell.count("| grep -c") == 1, (
        f"expected exactly one conjunctive grep pipeline in the probe's remote "
        f"command, found {shell.count('| grep -c')}: {shell}")
    assert "grep 'SESSION_SENTINEL' {log_path} | grep -c '{nonce}'" in shell, (
        f"the acceptance command is not a same-record conjunction: {shell}")
    # The acceptance count is out[0] and the diagnostic is out[1]; a swap would
    # silently restore the defect while every string above still looked right.
    unparsed = ast.unparse(probe)
    assert ("(conjunctive, sentinel_lines) = (int(out[0]), int(out[1]))"
            in unparsed), (
        "the probe no longer binds the conjunctive count to out[0]; the "
        "acceptance number and the diagnostic may have been swapped")
    assert "count=conjunctive" in unparsed, (
        "`count` — the only number acceptance reads — is not the conjunctive "
        "count")


# --- arms 12-13: THE REMOTE BRANCH (R2) -------------------------------------
#
# Numbered 12-13 for the same reason 9-11 were: arms 6-8 are cited by number in
# the R3 record and in Beta's rulings, and renumbering a certified arm to make a
# new one adjacent is how a governance citation quietly starts pointing at
# different code.
#
# WHAT THESE ADD THAT ARM 3 COULD NOT. RXP-3/3 declares
# `ssh fails / file unreadable -> UNAVAILABLE -> REFUSAL` but exercises only a
# LOCAL missing file. There is no ssh in it. And the production remote branch
# never read `proc.returncode`: an ordinary connectivity or auth failure is not
# an exception from `subprocess.run` — it returns a COMPLETED process with a
# nonzero ssh status and empty stdout — so it fell through the two-line check and
# was reported `ERROR: unparseable_probe_output`, i.e. "the probe ran and its
# output could not be classified" about a probe that never ran. Both outcomes
# refuse, so the consequence was evidentiary, not safety. Arm 3's declaration is
# corrected to what it exercises; these two arms exercise the rest.
#
# THE SEAM IS `SG._run`, the gate's own single subprocess entry point. Stubbing
# it means the REAL `probe_sentinel` body runs — the same classification, the
# same render, the same `evaluate` — against the exact `CompletedProcess` shape
# ssh returns, with no fleet and no network.
# ---------------------------------------------------------------------------
def _remote_sentinel_target(worker_id, gpu=0, endpoint="192.168.3.122"):
    """A REMOTE target: `local=False`, so `probe_sentinel` takes the ssh branch."""
    host, _, _g = worker_id.partition(":gpu")
    return {"node_id": host, "endpoint": endpoint, "ssh_user": "michael",
            "local": False, "gpu": gpu, "worker_hostname": host,
            "worker_id": worker_id}


def _completed(returncode, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess(args=["ssh", "..."],
                                       returncode=returncode,
                                       stdout=stdout, stderr=stderr)


# The stderr a real OpenSSH client writes when it cannot reach the host. Kept
# verbatim so the arm proves the operator gets the diagnostic, not a placeholder.
SSH_NO_ROUTE_STDERR = (b"ssh: connect to host 192.168.3.122 port 22: "
                       b"No route to host\r\n")


def _probe_with_stubbed_run(target, nonce, log_path, completed, probe=None):
    """Run a probe with the gate's subprocess seam returning `completed`.

    `probe` is either the production `SG.probe_sentinel` or a `(fn, namespace)`
    pair from `_mutated_probe`. **The stub is installed in the namespace THAT
    FUNCTION RESOLVES ITS NAMES IN** — the A8-B2 rule, and the reason the pair is
    threaded through rather than assumed: a mutant exec'd into a COPY of the gate
    module's globals would never see a patch applied to `SG`, the real subprocess
    would run, and the mutant would survive while looking detected."""
    if probe is None:
        fn, ns = SG.probe_sentinel, vars(SG)
    else:
        fn, ns = probe
    saved = ns["_run"]
    ns["_run"] = lambda cmd, timeout: completed
    try:
        return fn(target, nonce, log_path)
    finally:
        ns["_run"] = saved


def _mutated_probe(mutate, label):
    """LIVE `probe_sentinel`, mutated. Returns `(fn, namespace)` so the caller can
    stub the seam in the namespace the mutant actually resolves against."""
    src = _live_src("scripts/gate12_sentinel_gate.py")
    fn = _func_node(ast.parse(src), None, "probe_sentinel")
    body = textwrap.dedent(ast.get_source_segment(src, fn))
    mutant = mutate(body)
    assert mutant != body, (
        f"MUTANT NOT APPLIED ({label}) — the target text was not found in live "
        f"source, so the arm proves nothing about its own detection power")
    g = dict(vars(SG))
    exec(compile(mutant, f"<mutated probe_sentinel: {label}>",  # noqa: S102
                 "exec"), g)
    return g["probe_sentinel"], g


def rxp3_arm12_ssh_transport_failure_is_unavailable():
    """ARM 12 — [R2] THE REMOTE BRANCH. An ssh transport failure — a COMPLETED
    process carrying status 255 and diagnostic stderr, which is what a
    connectivity or `BatchMode` auth failure actually returns — is
    **UNAVAILABLE**, never a count and never ERROR.

    "The probe could not run" and "the probe ran and its output was
    unclassifiable" are different facts about the world, and the gate exists to
    keep them apart: this is the same distinction that let `GPU_COUNT_MISMATCH:
    0/8` sail through a 3/3 preflight in attempt 1, one layer further out.

    WRONG INPUT THAT REDS IT: not reading `proc.returncode` at all — i.e. the
    pre-R2 gate, which classified this as
    `ERROR: unparseable_probe_output:[]`. That exact mutation is applied below
    and MUST be detected."""
    def _run_probe(probe):
        t = _remote_sentinel_target("rrig6600:gpu3", 3)
        r = _probe_with_stubbed_run(
            t, "nonce-arm12", "/tmp/minerlogs/gpu3.log",
            _completed(SG.SSH_TRANSPORT_FAILURE_STATUS,
                       stdout=b"", stderr=SSH_NO_ROUTE_STDERR),
            probe=probe)
        assert r["status"] == SG.PROBE_UNAVAILABLE, (
            f"an ssh transport failure was classified {r['status']!r}: {r}")
        assert r["count"] is None, f"a probe that never ran reported a count: {r}"
        assert r["sentinel_lines_any_nonce"] is None, r
        assert "ssh_exit_255" in r["reason"], (
            f"the reason does not name the ssh status: {r['reason']!r}")
        # The operator must get ssh's own diagnostic, not a placeholder.
        assert "No route to host" in r["stderr"], r["stderr"]
        line = SG.render(r)
        assert "UNAVAILABLE" in line and "0/" not in line, (
            f"a probe that never ran rendered count-shaped: {line!r}")
        allowed, refusals = SG.evaluate([r])
        assert not allowed, "an unreachable worker was allowed to proceed"
        assert "UNKNOWN, not absent" in refusals[0], refusals

    _run_probe(None)
    # THE MUTANT IS BUILT HERE, OUTSIDE `_mutant_red`. `_mutant_red` credits any
    # exception as detection, so constructing the mutant inside its lambda would
    # let "MUTANT NOT APPLIED" read as "MUTANT DETECTED" — the very shape this
    # cycle is about. Built out here, a mutation that no longer matches live
    # source FAILS the arm instead of flattering it.
    mutant = _mutated_probe(
        lambda s: s.replace("proc.returncode == SSH_TRANSPORT_FAILURE_STATUS",
                            "False", 1),
        "returncode check neutralised")
    _mutant_red(lambda: _run_probe(mutant),
                "the pre-R2 gate: proc.returncode never examined")


def rxp3_arm13_an_executed_remote_probe_stays_error():
    """ARM 13 — [R2] THE NEIGHBOURING CONTROL, so UNAVAILABLE and ERROR cannot
    collapse into each other. A remote probe that DID execute keeps its own
    classification:

      (a) status 0, malformed stdout          -> ERROR  (ran; unclassifiable)
      (b) status 1, malformed stdout          -> ERROR  (ran; unclassifiable)
      (c) status 1, WELL-FORMED stdout        -> OK, with the count read

    (c) is Beta's sentence made executable: *"a remote `grep` returning 1 for
    'no match' is not a transport failure."* A gate that answered the R2 finding
    by treating every nonzero status as UNAVAILABLE would report the fleet
    unreachable because a log legitimately contained no sentinel — converting an
    evidentiary defect into a much worse one.

    WRONG INPUT THAT REDS IT: exactly that over-broad rule, `proc.returncode !=
    0 -> UNAVAILABLE`. It is applied below as a mutant and MUST be detected by
    (b) and (c)."""
    t = _remote_sentinel_target("rrig6600b:gpu1", 1)
    path = "/tmp/minerlogs/gpu1.log"

    def _run_probe(probe):
        # (a) executed, status 0, output that cannot be classified
        r = _probe_with_stubbed_run(
            t, "nonce-arm13", path,
            _completed(0, stdout=b"garbage\n"), probe=probe)
        assert r["status"] == SG.PROBE_ERROR, (
            f"malformed output from an EXECUTED probe was classified "
            f"{r['status']!r}: {r}")
        assert r["count"] is None, r
        allowed, refusals = SG.evaluate([r])
        assert not allowed and "could not be classified" in refusals[0], refusals

        # (b) executed, status 1, output that cannot be classified
        r = _probe_with_stubbed_run(
            t, "nonce-arm13", path,
            _completed(1, stdout=b"garbage\n"), probe=probe)
        assert r["status"] == SG.PROBE_ERROR, (
            f"a nonzero REMOTE COMMAND status was read as a transport failure: "
            f"{r}")

        # (c) executed, status 1, well-formed — the count is still read
        r = _probe_with_stubbed_run(
            t, "nonce-arm13", path,
            _completed(1, stdout=b"2\n5\n"), probe=probe)
        assert r["status"] == SG.PROBE_OK, (
            f"a legitimate remote `grep -c` exit 1 was not classified OK: {r}")
        assert r["count"] == 2 and r["sentinel_lines_any_nonce"] == 5, r
        allowed, _ = SG.evaluate([r])
        assert allowed, "a well-formed conjunctive count did not pass"

    _run_probe(None)
    # Built outside `_mutant_red` — see arm 12.
    mutant = _mutated_probe(
        lambda s: s.replace("proc.returncode == SSH_TRANSPORT_FAILURE_STATUS",
                            "proc.returncode != 0", 1),
        "any nonzero status treated as a transport failure")
    _mutant_red(lambda: _run_probe(mutant),
                "over-broad rule: every nonzero remote status becomes UNAVAILABLE")


# --- arms 6-8: ORDERING -----------------------------------------------------
class _StubListener:
    """A real listening socket that records the wall instant of every accepted
    connection and of every REGISTER frame, and nothing else."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.accepts: List[float] = []
        self.registers: List[Any] = []
        self._stop = threading.Event()
        self._conns: List[socket.socket] = []
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        self.sock.settimeout(0.1)
        while not self._stop.is_set():
            try:
                c, _ = self.sock.accept()
            except (socket.timeout, OSError):
                continue
            self.accepts.append(time.time())
            self._conns.append(c)
            threading.Thread(target=self._read, args=(c,), daemon=True).start()

    def _read(self, c):
        fs = MinerFramedSocket(c)
        try:
            msg = fs.recv_msg()
        except Exception:                                        # noqa: BLE001
            return
        if getattr(msg, "message_type", None) == "register":
            self.registers.append((time.time(), msg.worker_id))
        try:
            c.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        c.close()

    def close(self):
        self._stop.set()
        for c in self._conns:
            try:
                c.close()
            except OSError:
                pass
        try:
            self.sock.close()
        except OSError:
            pass


def _worker_main(main_src: str, argv: List[str], listener_port: int):
    """Execute a `main` — live or MUTATED — with the WORKER MODULE'S REAL
    GLOBALS, shimming only the GPU surface in `g`.

    The A8-B2 rule is the whole point of binding `g` to the module's own globals:
    a verbatim copy that resolved its names in THIS test module's namespace would
    escape every shim, and the mutant would survive."""
    g = dict(vars(WORKER))

    class _StubExecutor:
        def __init__(self, *a, **kw):
            pass

        def execute(self, *a, **kw):
            raise AssertionError("no stripe should ever be executed here")

    class _StubSignal:
        SIGTERM = 15
        SIGINT = 2

        @staticmethod
        def signal(*a, **kw):
            # main() installs handlers, which is only legal on the main thread;
            # the ordering property under test is unaffected by them.
            return None

    g["SieveExecutor"] = _StubExecutor
    g["signal"] = _StubSignal
    g["detect_gpu"] = lambda idx: WORKER.GpuInfo(
        backend="cuda", gpu_name="stub", vram_bytes=1)
    g["warm_gpu"] = lambda idx: None
    exec(compile(main_src, "<worker main>", "exec"), g)             # noqa: S102
    return g["main"](argv)


def _live_main_src() -> str:
    src = _live_src(WORKER_SRC_REL)
    fn = _func_node(ast.parse(src), None, "main")
    return textwrap.dedent(ast.get_source_segment(src, fn))


def _mutated_main_src() -> str:
    """THE MUTANT: `await_session_release` removed from `main`, by AST."""
    src = _live_main_src()
    tree = ast.parse(src)
    fn = tree.body[0]

    def _strip(body):
        out = []
        for stmt in body:
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, "attr", None)
                    == "await_session_release"):
                continue
            for attr in ("body", "orelse", "finalbody"):
                if hasattr(stmt, attr):
                    setattr(stmt, attr, _strip(getattr(stmt, attr)))
            out.append(stmt)
        return out or [ast.Pass()]

    fn.body = _strip(fn.body)
    ast.fix_missing_locations(tree)
    mutant = ast.unparse(tree)
    assert "await_session_release" not in mutant, (
        "MUTANT NOT APPLIED — the barrier call is still present")
    return mutant


def _run_ordering(main_src, n=2, release_after=0.6, nonce="nonce-order"):
    """Start `n` workers, hold the release for `release_after` seconds, then
    write the token. Returns (listener, release_instant)."""
    lis = _StubListener()
    tmp = tempfile.mkdtemp()
    rel = os.path.join(tmp, f"gate12_release_{nonce}")
    threads = []
    for i in range(n):
        argv = ["--host", "127.0.0.1", "--port", str(lis.port),
                "--gpu-id", str(i), "--device-index", "0",
                "--run-nonce", nonce, "--session-release-file", rel,
                "--release-deadline", "30",
                "--miner-output-dir", tmp]
        t = threading.Thread(target=_worker_main,
                             args=(main_src, argv, lis.port), daemon=True)
        t.start()
        threads.append(t)
    time.sleep(release_after)
    return lis, rel, threads, tmp


def rxp3_arm6_green_ordering():
    """ARM 6 — GREEN ORDERING. With release files ABSENT, ZERO connections arrive
    for the whole verification interval; after verification and release, EVERY
    REGISTER instant is LATER than its host's release-write instant.

    WRONG INPUT THAT REDS IT: a release written before verification completes —
    the second assertion then measures nothing."""
    lis, rel, _threads, tmp = _run_ordering(_live_main_src())
    try:
        assert lis.accepts == [], (
            f"{len(lis.accepts)} connection(s) arrived while the fleet was "
            f"supposed to be parked at the barrier — the sentinel would be "
            f"verified for a run that had already contacted the coordinator")
        release_at = time.time()
        with open(rel, "w", encoding="utf-8") as fh:
            fh.write("nonce-order")
        assert _wait(lambda: len(lis.registers) >= 2, timeout=20.0), (
            f"only {len(lis.registers)} REGISTER frames arrived after release")
        for at, wid in lis.registers:
            assert at > release_at, (
                f"{wid} registered BEFORE the release token was written")
    finally:
        lis.close()


def rxp3_arm7_the_early_register_mutant():
    """ARM 7 — THE MUTANT (Beta-required). One worker re-executed from an
    AST-mutated `main` with `await_session_release` REMOVED registers early, and
    ARM 6'S ASSERTION MUST FAIL.

    A run in which this passes is a gate encoding the defect as acceptable, and
    the suite reports FAIL.

    WRONG INPUT THAT REDS IT: the release wait made advisory; or the mutant
    resolving its names in the test module's globals and escaping the shim."""
    def _mutant_arm6():
        lis, _rel, _threads, _tmp = _run_ordering(_mutated_main_src(), n=1)
        try:
            assert lis.accepts == [], (
                f"{len(lis.accepts)} connection(s) arrived before release")
        finally:
            lis.close()

    _mutant_red(_mutant_arm6, "await_session_release removed from main")


def rxp3_arm8_ast_ordering_over_live_main():
    """ARM 8 — AST over live `range_miner_worker.main`: the
    `await_session_release` call node PRECEDES the `connect`/`register` nodes,
    and no `register()` is reachable ahead of it.

    THIS IS THE ARM THAT SURVIVES A REFACTOR which keeps the call but moves it.

    WRONG INPUT THAT REDS IT: moving the barrier below `connect()`."""
    src = _live_src(WORKER_SRC_REL)
    fn = _func_node(ast.parse(src), None, "main")
    lines = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None)
            if name in ("await_session_release", "connect", "register",
                        "serve_forever", "emit_startup_sentinel", "prepare"):
                lines.setdefault(name, node.lineno)
    for required in ("prepare", "emit_startup_sentinel", "await_session_release",
                     "connect", "register", "serve_forever"):
        assert required in lines, f"`main` no longer calls {required}()"
    assert (lines["prepare"] < lines["emit_startup_sentinel"]
            < lines["await_session_release"] < lines["connect"]
            < lines["register"] < lines["serve_forever"]), (
        f"the startup order is not prepare -> sentinel -> BARRIER -> connect -> "
        f"register -> serve: {lines}")


# ===========================================================================
# FAIR-5 — RED authenticity (§11.4, 8 assertions)
#
# SPECIFIED AND RUN FIRST among the FAIR gates, because every other FAIR RED arm
# depends on the anchor. Pinned `2b0d2dc` must still reproduce EVERY defect
# surface this amendment repairs, or the suite reports UNAVAILABLE.
# ===========================================================================
def fair5_anchor_reproduces_every_defect_surface():
    """The eight assertions, executed. Their content is in
    `_assert_pinned_carries_the_defects`, which is the single implementation
    every RED arm in this file calls — so no arm can be credited against an
    anchor that has drifted, and the check cannot be forgotten at one site.

    WHAT WRONG INPUT MAKES IT FAIL: re-pinning to a later commit that already
    carries the repair — assertions 2, 4, 6 and 7 red and the whole suite reports
    UNAVAILABLE, never PASS."""
    _assert_pinned_carries_the_defects()


def fair5_self_protection_refuses_repaired_source():
    """ASSERTION 8, driven positively: pointing the anchor at REPAIRED source
    must raise `AnchorUnavailable`. Demonstrated against HEAD's live source at
    gate time, so the anchor cannot be silently re-pinned forward."""
    live_head = subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
    saved = globals()["PINNED_COMMIT"]
    try:
        # The working tree carries the repair; a commit-shaped anchor pointing at
        # it must be refused by the very same probes.
        globals()["_PINNED_CACHE"].clear()
        globals()["PINNED_COMMIT"] = live_head
        try:
            _assert_pinned_carries_the_defects()
        except AnchorUnavailable:
            return
        # HEAD is the PRE-repair commit until Michael commits this work, so a
        # pass here is expected ONLY while HEAD == the pinned anchor.
        assert live_head == saved, (
            f"the anchor probes accepted {live_head[:12]} as pre-repair source "
            f"even though it is not the pinned commit — they do not discriminate")
    finally:
        globals()["PINNED_COMMIT"] = saved
        globals()["_PINNED_CACHE"].clear()


# ===========================================================================
# FAIR-1 / FAIR-2 — control-plane fairness under the two pressure shapes
# (§11.5, 7 arms)
#
# OPERATIVE RECURRENCE, asserted by both gates:
#
#     T_cp  <=  A  +  D_adm + m_i  +  D + M_i  +  K_i
#
# measured with `perf_counter` between consecutive executions of the REAL
# `schedule_pending_stripes`, with `D` and `D_adm` READ FROM LIVE CONFIG — so
# raising them raises the asserted bound VISIBLY, in the gate's own report,
# rather than quietly buying headroom.
# ===========================================================================
# The cap geometry the fairness runs use: a small NVIDIA cap makes ONE stripe
# partition into many sub-stripes, which is what produces sustained legal result
# traffic. It is advertised by the workers AND configured on the coordinator, so
# `_validate_caps` sees a match.
_FAIR_CAPS = {**CAPS, "nvidia": 10, "nvidia_hybrid": 10}


class _SlowStagingSeam:
    """A scripted per-message cost on the PRODUCTION path.

    `enqueue_staging` is a real staging seam the serve loop calls inside its
    `msg` segment, so wrapping it puts the cost exactly where a slow ledger or a
    contended `_admission_lock` would put it. Nothing about the drain is patched:
    a patched loop would be a model of the defect rather than the defect."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.calls = 0

    def __enter__(self):
        self._orig = RangeMinerCoordinator.enqueue_staging
        seam = self

        def _slow(self_, *a, **kw):
            seam.calls += 1
            time.sleep(seam.seconds)
            return seam._orig(self_, *a, **kw)

        RangeMinerCoordinator.enqueue_staging = _slow
        return self

    def __exit__(self, *exc):
        RangeMinerCoordinator.enqueue_staging = self._orig
        return False


class _ScheduleProbe:
    """Records the monotonic instant of every REAL `schedule_pending_stripes`
    execution — the control-plane turn whose interval IS `T_cp`."""

    def __init__(self):
        self.marks: List[float] = []

    def __enter__(self):
        self._orig = RangeMinerCoordinator.schedule_pending_stripes
        probe = self

        def _wrapped(self_, *a, **kw):
            probe.marks.append(time.perf_counter())
            return probe._orig(self_, *a, **kw)

        RangeMinerCoordinator.schedule_pending_stripes = _wrapped
        return self

    def __exit__(self, *exc):
        RangeMinerCoordinator.schedule_pending_stripes = self._orig
        return False

    def max_gap(self):
        return max((b - a for a, b in zip(self.marks, self.marks[1:])),
                   default=0.0)


class _LoopbackWorker:
    """A real framed-TCP worker: connects, REGISTERs, waits for a real
    `StripeAssign`, and then streams well-formed inline `sub_stripe_result`
    frames FOR THE STRIPE IT WAS ACTUALLY ASSIGNED.

    Waiting for the assignment is not a nicety: a result whose `stripe_id` was
    invented by the harness is dropped by the L1 fence BEFORE it reaches staging,
    so a gate built on invented ids would script a cost onto a code path the
    messages never reach and would measure an empty drain while believing it had
    established pressure."""

    def __init__(self, host, port, worker_id, caps=None):
        self.worker_id = worker_id
        host_name, _, gpu = worker_id.partition(":gpu")
        self.caps = dict(caps or CAPS)
        self.sock = socket.create_connection((host, port), timeout=10)
        self.fs = MinerFramedSocket(self.sock)
        self.fs.send_msg(RegisterMessage(
            worker_id=worker_id, hostname=host_name, gpu_id=int(gpu),
            gpu_name="stub", backend="cuda", vram_bytes=1,
            capabilities={"seed_caps": dict(self.caps),
                          "supported_variants": list(VARIANTS)}))
        self._stop = threading.Event()
        self.assigned: List[str] = []
        self.sent = 0
        self.err = None
        self.limit = None            # None = stream until stopped
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            self.fs.sock.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    msg = self.fs.recv_msg()
                except socket.timeout:
                    continue
                except (ConnectionError, ValueError, OSError):
                    return
                if getattr(msg, "message_type", None) == "stripe_assign":
                    self.assigned.append(msg.stripe_id)
                    self._stream(msg)
                elif getattr(msg, "message_type", None) == "shutdown":
                    return
        except Exception:                                        # noqa: BLE001
            self.err = traceback.format_exc()

    def _stream(self, assign):
        i = 0
        while not self._stop.is_set():
            if self.limit is not None and i >= self.limit:
                return
            ss = assign.seed_start + i
            try:
                self.fs.send_msg(_inline_result(
                    self.worker_id, assign.stripe_id, i, ss, 1))
            except OSError:
                return
            self.sent += 1
            i += 1

    def wait_assigned(self, timeout=20.0):
        return _wait(lambda: bool(self.assigned), timeout=timeout)

    def close(self):
        self._stop.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def _fairness_run(tmp, m_seconds, n_workers=2, serve_timeout=25.0,
                  drain_budget=None, stream_forever=True, extra=None):
    """Drive the REAL `serve_trial` under a scripted per-message cost, with real
    loopback workers streaming legal result traffic."""
    ds = os.path.join(tmp, "dataset.json")
    with open(ds, "w") as f:
        f.write('[{"draw":1},{"draw":2},{"draw":3}]')
    lsock = socket.socket()
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(16)
    port = lsock.getsockname()[1]
    holder: Dict[str, Any] = {}
    kw = dict(extra or {})
    if drain_budget is not None:
        kw["drain_budget_seconds"] = drain_budget

    def _run():
        try:
            holder["out"] = run_trial_miner(
                "cfgFAIR", None, 1, "java_lcg", [1, 2, 3], 200, 0.25, 0.25,
                False, ds, skip_min=0, skip_max=0, window_size=3,
                window_anchor=0, generator_phase=0,
                worker_pool_size=n_workers, miner_stripe_size=100,
                seed_cap_nvidia=10, seed_cap_nvidia_hybrid=10,
                staging_dir=os.path.join(tmp, "staging"),
                listen_sock=lsock, family_name="java_lcg", workflow_phase=1,
                serve_timeout=serve_timeout, serve_poll=0.1, **kw)
        except Exception:                                        # noqa: BLE001
            holder["err"] = traceback.format_exc()

    probe = _ScheduleProbe()
    workers: List[_LoopbackWorker] = []
    with _SlowStagingSeam(m_seconds) as seam, probe:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.4)
        for i in range(n_workers):
            # ⚠ THE ADVERTISED CAPS MUST MATCH THE COORDINATOR'S CENTRAL CONFIG
            # or `_validate_caps` quarantines the worker at registration, it
            # never becomes eligible, and the trial dies of ADMISSION rather
            # than of the condition under test. That is a real property of the
            # production handshake, not a harness quirk.
            w = _LoopbackWorker("127.0.0.1", port, f"hostA:gpu{i}",
                                caps=_FAIR_CAPS)
            w.limit = None if stream_forever else 2
            workers.append(w)
        assert _wait(lambda: len(probe.marks) >= 1, timeout=25.0), (
            "the control plane never ran — the stage was never assigned")
        assert _wait(lambda: any(w.assigned for w in workers), timeout=25.0), (
            "no worker was ever handed a stripe, so no legal result traffic "
            "could exist and the pressure shape would be fictional")
        t.join(timeout=serve_timeout + 30)
        for w in workers:
            w.close()
        alive = t.is_alive()
    try:
        lsock.close()
    except OSError:
        pass
    assert not alive, "serve_trial never terminated"
    assert "err" not in holder, holder.get("err")
    return holder.get("out") or {}, probe, seam


def _recurrence_bound(metrics, m_seconds, poll=0.1):
    """`A + D_adm + m_i + D + M_i + K_i`, with `D`/`D_adm` READ FROM LIVE CONFIG
    and the runtime terms taken from the run's own instrument."""
    sl = metrics.get("serve_loop_timing") or {}
    D = DEFAULT_DRAIN_BUDGET_SECONDS
    D_adm = COORD.DEFAULT_ADMISSION_DRAIN_BUDGET_SECONDS
    A = min(poll, sl.get("accept_max", poll))
    M_i = max(sl.get("msg_max", 0.0), m_seconds)
    K_i = sl.get("control_block_max", 0.0)
    m_reg = sl.get("admission_max", 0.0)
    return A + D_adm + m_reg + D + M_i + K_i, {
        "A": A, "D_adm": D_adm, "m_i": m_reg, "D": D, "M_i": M_i, "K_i": K_i}


def fair1_count_pressure():
    """FAIR-1 — MANY CHEAP messages: >=2 loopback workers stream well-formed
    frames continuously so the drain always has more available, each handler
    scripted to a known cost.

    ARMS 1 and 2: the recurrence holds, and `drain_deadline_hits > 0` — the
    deadline is what STOPPED the drain, so the arm is not passing on an
    accidentally short queue.

    WRONG INPUT THAT REDS IT: `drain_deadline_hits == 0` reds the gate as
    VACUOUS; and `D` read from live config means raising it raises the asserted
    bound visibly, in this gate's own report."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out, probe, seam = _fairness_run(tmp, m_seconds=0.02, n_workers=2,
                                         serve_timeout=12.0)
    sl = out.get("serve_loop_timing") or {}
    bound, terms = _recurrence_bound(out, 0.02)
    gap = probe.max_gap()
    assert seam.calls >= 20, (
        f"only {seam.calls} scripted messages were handled — the pressure shape "
        f"was never established and the arm is vacuous")
    assert sl.get("drain_deadline_hits", 0) > 0, (
        f"drain_deadline_hits == 0: the drain never reached its budget, so this "
        f"arm proves nothing about a budget. Terms: {terms}")
    assert gap <= bound, (
        f"T_cp = {gap:.3f}s exceeds the recurrence {bound:.3f}s; terms={terms}")


def fair2_slow_message_pressure():
    """FAIR-2 — FEW EXPENSIVE messages: the same harness with a multi-second
    scripted cost. NOT FAIR-1 with a bigger number: FAIR-1 tests many cheap,
    FAIR-2 tests few expensive, and a count bound fails them differently.

    ARMS 3 and 4: the residual is attributed to EXACTLY ONE message and M-3's
    slow-message record names its type, worker and stripe; and the slow message
    STILL COMPLETES and its result is recorded.

    WRONG INPUT THAT REDS IT: a mechanism that "achieves the bound" by ABORTING a
    message mid-flight — arm 4 then reds, because the bound was bought by
    dropping work."""
    M = 1.5
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            out, probe, seam = _fairness_run(
                tmp, m_seconds=M, n_workers=2, serve_timeout=20.0,
                stream_forever=False)
        slow = cap.events("SLOW_MSG")
    sl = out.get("serve_loop_timing") or {}
    bound, terms = _recurrence_bound(out, M)
    gap = probe.max_gap()
    assert seam.calls >= 2, f"only {seam.calls} slow messages were handled"
    assert gap <= bound, (
        f"T_cp = {gap:.3f}s exceeds the recurrence {bound:.3f}s; terms={terms}")
    assert sl.get("msg_max", 0.0) >= M, (
        f"the scripted cost never reached the message segment: {sl.get('msg_max')}")
    assert slow, (
        "no slow-message record was emitted — `M_max` stays a number with no "
        "identity, which is what §8.5.2 exists to prevent")
    assert slow[0]["message_type"] == "sub_stripe_result", slow[0]
    assert slow[0]["worker_id"] and slow[0]["stripe_id"], slow[0]
    assert slow[0]["seconds"] >= M, slow[0]
    # arm 4: the slow message COMPLETED — its result is in the ledger.
    assert (out.get("staging_backpressure") or {}).get(
        "staging_jobs_completed", 0) >= 1, (
        "no staging job completed: the bound was bought by dropping work")


def fair12_arm5_first_get_respects_the_remaining_budget():
    """ARM 5 — [R2.3] with `D` configured BELOW `poll` and an EMPTY queue, one
    drain pass consumes <= D, not `poll`.

    WRONG INPUT THAT REDS IT: the `timeout=poll if drained == 0 else 0` form —
    the pass then consumes ~0.10 s having handled NOTHING, which is neither `D`
    nor "one in-flight message", and the structural claim is false for every
    `D < poll`."""
    D, poll = 0.02, 0.5
    q = _queue.Queue(maxsize=8)
    # The production expression, executed as written: an empty queue, a drain
    # with no backlog known, and a deadline `D` in the future.
    t0 = time.perf_counter()
    deadline = t0 + D
    remaining = deadline - time.perf_counter()
    timeout = 0.0 if False else min(poll, max(0.0, remaining))
    try:
        q.get(timeout=timeout)
    except _queue.Empty:
        pass
    elapsed = time.perf_counter() - t0
    assert elapsed <= D + 0.02, (
        f"an idle drain pass consumed {elapsed:.3f}s against D={D}s")
    # …and the form is the one in live source, not a paraphrase of it.
    tree = ast.parse(_live_src())
    drain = _drain_while(_func_node(tree, "RangeMinerCoordinator", "serve_trial"))
    src = ast.unparse(drain)
    # [R1 AUDIT] The `.replace(" ", " ")` that stood here was a NO-OP that read
    # as whitespace normalisation and performed none. `ast.unparse` already
    # normalises, which is the reason this match is over unparsed source and not
    # over the file text; the dead call only made the guarantee look weaker than
    # it is.
    assert "min(poll, max(0.0, _remaining))" in src, (
        "the first-get timeout is not clamped by the remaining budget")
    assert "timeout=poll if drained == 0 else 0" not in src, (
        "the unbudgeted first-get form is back")


def fair12_arm6_control_block_is_a_composite():
    """ARM 6 — `control_block_max` is recorded as a COMPOSITE per iteration, and
    a `slow_control` record is emitted on crossing `k_slow_threshold`, naming the
    dominant segment.

    WRONG INPUT THAT REDS IT: reporting six independent per-segment maxima only —
    a composite `K_i` is NOT derivable from them, and the recurrence names the
    composite."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            out, _probe, _seam = _fairness_run(
                tmp, m_seconds=0.01, n_workers=1, serve_timeout=8.0,
                stream_forever=False,
                extra={"k_slow_threshold": 0.0})     # every iteration crosses
        slow = cap.events("SLOW_CONTROL")
    sl = out.get("serve_loop_timing") or {}
    assert sl.get("control_block_count", 0) > 0, (
        f"no composite control block was recorded: {sorted(sl)}")
    assert sl["control_block_max"] > 0.0, sl
    assert slow, "no slow_control record was emitted on a crossing"
    rec = slow[0]
    assert rec["dominant_segment"] in (
        "deadline", "stage_setup", "schedule", "dispatch", "expiry", "advance"), rec
    assert set(rec["parts"]) <= {"deadline", "stage_setup", "schedule",
                                 "dispatch", "expiry", "advance"}, rec
    assert abs(rec["composite_s"] - sum(rec["parts"].values())) < 1e-6, rec


def fair12_arm7_red_count_bound_admits_the_violation():
    """ARM 7 — RED, on pinned `2b0d2dc`: a COUNT bound admits
    `T_cp >= N x m`, which is the attempt-5 shape REPRODUCED FROM A PINNED COMMIT
    rather than described.

    The pinned drain and the pinned reader are executed TOGETHER — the pinned
    reader enqueues the 4-wide tuples the pinned drain unpacks — so the arm runs
    the real pre-repair mechanism rather than a model of it.

    WRONG INPUT THAT REDS IT: deleting the deadline and lowering 256 instead. The
    GREEN assertion still fails, because a count bound cannot satisfy a time
    assertion at any N where `m` is scripted large; and a drifted anchor
    terminates UNAVAILABLE, never PASS."""
    _assert_pinned_carries_the_defects()
    src = _pinned_src()
    tree = ast.parse(src)
    g = dict(vars(COORD))
    for name in ("serve_trial", "_conn_reader_loop"):
        fn = _func_node(tree, "RangeMinerCoordinator", name)
        exec(compile(textwrap.dedent(ast.get_source_segment(src, fn)),
                     f"<pinned {PINNED_COMMIT} {name}>", "exec"), g)  # noqa: S102
    pinned_serve, pinned_reader = g["serve_trial"], g["_conn_reader_loop"]

    # [WINDOW-ANCHOR BRIEF I] PINNED-SOURCE ADAPTER — TEST-LOCAL, NEVER PRODUCTION.
    #
    # This arm executes control-plane source pinned at 2b0d2dc against the LIVE
    # helper functions. The window-anchor separation changed one of those helpers:
    # `build_trial_context_from_serve` now projects `window_anchor` +
    # `generator_phase` and no longer emits `offset`, while the PINNED serve_trial
    # still reads `trial_ctx["offset"]`. Without this bridge the pinned thread dies
    # on KeyError before the probe records a mark, and the arm reports "the pinned
    # control plane never ran" — an infrastructure failure that reads exactly like
    # the timing regression the arm exists to detect.
    #
    # THE BRIDGE LIVES IN THE PINNED EXECUTION'S OWN GLOBALS (`g`), so it is visible
    # ONLY to the exec'd historical functions. Production is untouched, no other
    # test sees it, and the hard-reject on the legacy key is unaffected: this is not
    # a compatibility shim on any live path, it is a translation layer for running
    # frozen source. The pinned code's semantics are unchanged — at this arm's
    # `window_anchor=0` the historical scalar and the anchor are the same number,
    # which is exactly the pre-separation coincidence, reproduced deliberately.
    _live_build_ctx = COORD.build_trial_context_from_serve

    def _pinned_trial_context(context, dataset_sha256, residue_sha256):
        ctx = _live_build_ctx(context, dataset_sha256, residue_sha256)
        return {**ctx, "offset": ctx["window_anchor"]}

    g["build_trial_context_from_serve"] = _pinned_trial_context

    N, m = 24, 0.05        # N x m = 1.2 s of drain the count bound cannot stop
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1},{"draw":2},{"draw":3}]')
        ledger = MinerLedger(os.path.join(tmp, "l.db"))
        coord = RangeMinerCoordinator(
            CoordinatorConfig(staging_dir=os.path.join(tmp, "staging"),
                              miner_stripe_size=100,
                              seed_cap_nvidia=10,
                              seed_cap_nvidia_hybrid=10), ledger)
        # the PINNED reader, so the tuple widths on both ends agree
        coord._conn_reader_loop = types.MethodType(pinned_reader, coord)

        # [WINDOW-ANCHOR BRIEF I] SECOND AND LAST PINNED-SOURCE BRIDGE.
        #
        # The coupling surface was ENUMERATED, not discovered one failure at a
        # time: pinned serve_trial calls exactly three names Brief I touched —
        #   build_trial_context_from_serve(3 pos)  -> bridged in `g` above (schema)
        #   self.set_trial_context(2 pos)          -> NO bridge needed; Brief I
        #                                             changed its body, not its arity
        #   self._dispatch_pending(14 pos)         -> bridged here (signature)
        # Brief I inserted `generator_phase` into _dispatch_pending's positional
        # list, so the pinned 14-argument call is one short and dies on TypeError
        # before a stripe is ever assigned.
        #
        # INSTANCE-SCOPED: this shadows the bound method on THIS coordinator only.
        # Production and every other test see the real signature. The inserted
        # value is the v1 pin, which is what the pinned code would have carried
        # had the field existed.
        _live_dispatch = coord._dispatch_pending

        def _pinned_dispatch(run_id, family_name, phase, fs_by_worker, dispatched,
                             dataset_path, dataset_sha256, window_size, sessions,
                             window_anchor, residues, trial_number,
                             forward_threshold, reverse_threshold):
            return _live_dispatch(
                run_id, family_name, phase, fs_by_worker, dispatched, dataset_path,
                dataset_sha256, window_size, sessions, window_anchor,
                0,                      # generator_phase — the v1 pin
                residues, trial_number, forward_threshold, reverse_threshold)

        coord._dispatch_pending = _pinned_dispatch
        lsock = socket.socket()
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        port = lsock.getsockname()[1]
        run_id = "pinnedRED"
        ledger.create_trial(run_id, 1)
        ctx = {
            "run_id": run_id, "trial_number": 1, "prng_base": "java_lcg",
            "family_name": "java_lcg", "phase": 1,
            "workflow_stages": [("java_lcg", 1)],
            "residues": [1, 2, 3], "total_seeds": 200, "dataset_path": ds,
            "forward_threshold": 0.25, "reverse_threshold": 0.25,
            "skip_min": 0, "skip_max": 0, "test_both_modes": False,
            "worker_pool_size": 1, "window_size": 3,
            "window_anchor": 0, "generator_phase": 0,
            "sessions": None, "staging_dir": os.path.join(tmp, "staging"),
            "listen_sock": lsock, "serve_poll": 0.1, "serve_timeout": 30.0,
            "serve_read_deadline": 15.0,
            "worker_admission_timeout": COORD.DEFAULT_WORKER_ADMISSION_TIMEOUT,
        }
        probe = _ScheduleProbe()
        holder: Dict[str, Any] = {}
        with _SlowStagingSeam(m), probe:
            def _run():
                try:
                    holder["out"] = types.MethodType(pinned_serve, coord)(ctx)
                except Exception:                                # noqa: BLE001
                    holder["err"] = traceback.format_exc()

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            time.sleep(0.4)
            w = _LoopbackWorker("127.0.0.1", port, "hostA:gpu0",
                                caps=_FAIR_CAPS)
            assert _wait(lambda: len(probe.marks) >= 1, timeout=25.0), (
                "the pinned control plane never ran")
            assert w.wait_assigned(), (
                "the pinned loop never assigned a stripe, so no legal result "
                "traffic could exist")
            t.join(timeout=90)
            w.close()
            alive = t.is_alive()
        try:
            lsock.close()
        except OSError:
            pass
    assert not alive, "the pinned serve loop never terminated"
    assert "err" not in holder, holder.get("err")
    gap = probe.max_gap()
    assert gap >= N * m, (
        f"the PINNED count bound produced a maximum control-plane interval of "
        f"only {gap:.3f}s against {N} x {m}s = {N * m:.3f}s — the monopolization "
        f"surface is not being exercised, so no RED credit is due")


def fair12_arm8_config_terms_fail_closed():
    """[IMPLEMENTATION BRIEF, binding detail 2] THE FOUR BOUNDED-SERVICE TERMS
    ARE FAIL-CLOSED VALIDATED IN CODE, not merely declared so in the design.

    `A_max` is the one that matters most: the progress proof requires AT LEAST
    ONE eligible disposition attempt, so `A_max = 0` would silently restore
    admission starvation while every other term still looked correctly
    configured. It must fail closed unless it is an INTEGER >= 1 — a float, a
    bool, `None` and 0 are all refused.

    WRONG INPUT THAT REDS IT: honouring any of these values into the run. Each
    of them is exactly the shape that restores a defect this amendment closes:
    `D = None/0/inf` an unbounded drain; `D_adm` the same for the admission
    turn; `S = 0` the ungoverned single-shot shed; `A_max = 0` starvation."""
    base = dict(
        run_id="cfgVAL", trial_number=1, prng_base="java_lcg",
        family_name="java_lcg", phase=1, workflow_stages=[("java_lcg", 1)],
        residues=[1, 2, 3], total_seeds=10, dataset_path="/nonexistent",
        forward_threshold=0.25, reverse_threshold=0.25, skip_min=0, skip_max=0,
        test_both_modes=False, worker_pool_size=1, window_size=3,
        window_anchor=0, generator_phase=0,
        sessions=None, serve_poll=0.1, serve_timeout=1.0,
        serve_read_deadline=15.0,
        worker_admission_timeout=COORD.DEFAULT_WORKER_ADMISSION_TIMEOUT,
    )
    cases = [
        ("drain_budget_seconds", [None, 0, -1.0, float("inf"), "x"]),
        ("admission_drain_budget_seconds", [None, 0, -1.0, float("inf")]),
        ("inbound_saturation_budget_seconds", [None, 0, -1.0, float("inf")]),
        ("admission_max_dispositions", [None, 0, -1, 1.5, True, "8"]),
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord = _coord(tmp)
        for key, values in cases:
            for bad in values:
                ctx = dict(base)
                ctx["staging_dir"] = os.path.join(tmp, "staging")
                ctx[key] = bad
                try:
                    coord.serve_trial(ctx)
                except ValueError as e:
                    assert key in str(e), (
                        f"{key}={bad!r} was refused, but the refusal names "
                        f"something else: {e}")
                    continue
                except Exception as e:                           # noqa: BLE001
                    raise AssertionError(
                        f"{key}={bad!r} was not refused by the fail-closed "
                        f"validator; it failed later with "
                        f"{type(e).__name__}: {e}") from None
                raise AssertionError(
                    f"{key}={bad!r} was ACCEPTED — the value that restores the "
                    f"defect is being honoured into the run")
    # …and the validation happens BEFORE the dataset digest, the trial context
    # and the listening socket, exactly as `worker_admission_timeout`'s does.
    tree = ast.parse(_live_src())
    serve = _func_node(tree, "RangeMinerCoordinator", "serve_trial")
    body = ast.unparse(serve)
    val_at = body.index("admission_max_dispositions must be an INTEGER")
    ds_at = body.index("resolve_dataset_sha256")
    assert val_at < ds_at, (
        "the bounded-service terms are validated AFTER the dataset is resolved; "
        "a misconfigured bound must be refused before the run can start")


# ===========================================================================
# FAIR-6 — registration under backlog, BOTH DIRECTIONS (§11.6, 12 arms)
#
#     REGISTER cannot be starved by result backlog          <- P-1 + channel
#         AND
#     REGISTER priority cannot starve the control plane     <- D_adm + A_max
#
# Costs are SCRIPTED AND BOUNDED: the gate proves the MECHANISM, it does not
# inherit §8.6.6's arithmetic, and no arm asserts an absolute 180 s ceiling —
# `M_i`, `K_i` and `m_i` are unbounded by this repair and no arm can test a claim
# the mechanism does not make.
# ===========================================================================
def _register_msg(worker_id):
    host, _, gpu = worker_id.partition(":gpu")
    return RegisterMessage(
        worker_id=worker_id, hostname=host, gpu_id=int(gpu), gpu_name="stub",
        backend="cuda", vram_bytes=1,
        capabilities={"seed_caps": dict(CAPS),
                      "supported_variants": list(VARIANTS)})


class _AdmissionServer:
    """The production admission-service turn, executed as the serve loop executes
    it: `get_nowait` only, the deadline tested from the SECOND disposition, the
    count capped by `A_max`, and every disposition through
    `_serve_register_frame`.

    It is the loop body from `serve_trial`, driven directly — which is what lets
    an arm script `m_i` and count dispositions per TURN, the quantity R3.2's
    progress floor is about."""

    def __init__(self, coord, admission, d_adm, a_max, fs_by_sock,
                 worker_by_sock, wconn_by_worker, fs_by_worker, registered,
                 conn_meta, reader_threads, conn_state):
        self.coord = coord
        self.admission = admission
        self.d_adm = d_adm
        self.a_max = a_max
        self.args = (fs_by_sock, worker_by_sock, wconn_by_worker, fs_by_worker,
                     registered, conn_meta, reader_threads, conn_state)
        self.turns: List[int] = []

    def turn(self):
        (fs_by_sock, worker_by_sock, wconn_by_worker, fs_by_worker, registered,
         conn_meta, reader_threads, conn_state) = self.args
        deadline = time.perf_counter() + self.d_adm
        done = 0
        while done < self.a_max:
            if done > 0 and time.perf_counter() >= deadline:
                break
            try:
                token, sock, msg = self.admission.get_nowait()
            except _queue.Empty:
                break
            try:
                if sock in fs_by_sock:
                    self.coord._serve_register_frame(
                        msg, sock, None, fs_by_sock, worker_by_sock,
                        wconn_by_worker, fs_by_worker, registered, conn_meta,
                        reader_threads, stage_idx=0, stage_assigned=True,
                        eligible_fn=lambda: list(wconn_by_worker.values()),
                        conn_state_by_sock=conn_state)
            finally:
                self.coord.note_register_disposition(token)
            done += 1
        self.turns.append(done)
        return done


class _RecordingAdmission(_queue.SimpleQueue):
    """The production admission channel, with the ORDER OF ENTRY recorded.

    Cross-connection ordering is not a property this system has ever had —
    independent sockets, independent reader threads — so "FIFO" is a claim about
    THE CHANNEL, and asserting it against the order the harness called `send()`
    would be asserting a property the design explicitly disclaims."""

    def __init__(self):
        super().__init__()
        self.put_order: List[str] = []
        self._lock = threading.Lock()

    def put(self, item, *a, **kw):
        with self._lock:
            self.put_order.append(getattr(item[2], "worker_id", None))
            return super().put(item, *a, **kw)


def _fair6_bench(tmp, n_conns, backlog, d_adm=None, a_max=None,
                 register_cost=0.0, reader=None):
    """N connections, each on the REAL reader loop, each sending a first-frame
    REGISTER, against an `inbound` already holding `backlog` result envelopes.

    `reader` accepts an UNBOUND mutated `_conn_reader_loop` (bound here, to this
    bench's coordinator) so an arm can prove its own detection power against the
    defect it exists for. Default None = the production reader, verbatim."""
    coord = _coord(tmp)
    inbound = _queue.Queue(maxsize=1024)
    for i in range(backlog):
        inbound.put(("msg", None, None, None, None))
    admission = _RecordingAdmission()
    stop = threading.Event()
    fs_by_sock, worker_by_sock, wconn_by_worker = {}, {}, {}
    fs_by_worker, registered, conn_meta, reader_threads, conn_state = (
        {}, [], {}, {}, {})
    peers = []
    for i in range(n_conns):
        p = _Peer(coord, f"hostA:gpu{i}", worker_by_sock, inbound, stop,
                  bind=False, admission=admission,
                  emergency=_queue.SimpleQueue(),
                  reader=(None if reader is None
                          else types.MethodType(reader, coord)))
        fs_by_sock[p.srv] = p.srv_fs
        conn_meta[p.srv] = {"connect": time.time(), "registered": False}
        conn_state[p.srv] = p.state
        peers.append(p)
    if register_cost:
        orig = RangeMinerCoordinator.register_worker

        def _slow(self_, **kw):
            time.sleep(register_cost)
            return orig(self_, **kw)

        RangeMinerCoordinator.register_worker = _slow
        restore = lambda: setattr(RangeMinerCoordinator, "register_worker", orig)
    else:
        restore = lambda: None
    server = _AdmissionServer(
        coord, admission,
        DEFAULT_DRAIN_BUDGET_SECONDS / 10.0 if d_adm is None else d_adm,
        DEFAULT_ADMISSION_MAX_DISPOSITIONS if a_max is None else a_max,
        fs_by_sock, worker_by_sock, wconn_by_worker, fs_by_worker, registered,
        conn_meta, reader_threads, conn_state)
    return coord, peers, inbound, admission, server, stop, restore, {
        "worker_by_sock": worker_by_sock, "wconn_by_worker": wconn_by_worker,
        "fs_by_worker": fs_by_worker, "registered": registered}


def fair6_arm1_new_worker_is_not_starved_by_backlog():
    """ARM 1 — under sustained backlog a NEW worker's REGISTER becomes
    AUTHORITATIVE — `_serve_register` returns "ok", the identity is bound in
    `worker_by_sock`/`fs_by_worker`, and it appears in the eligible pool —
    within `<= p` admission-service turns, `p` being its FIFO position.

    WRONG INPUT THAT REDS IT: removing P-1 — the wait then grows with `inbound`
    depth. And the WITHDRAWN `ceil(p / A_max)` form: it assumes `A_max`
    completions per turn, which arm 5 shows is not guaranteed, so a build that
    services one per turn would red a gate it actually satisfies."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord, peers, inbound, admission, server, stop, restore, maps = \
            _fair6_bench(tmp, n_conns=1, backlog=1000)
        try:
            assert inbound.qsize() >= 1000, "the backlog was not established"
            peers[0].send(_register_msg("hostA:gpu0"))
            assert _wait(lambda: admission.qsize() == 1, timeout=5.0), (
                "the first-frame REGISTER did not take the admission channel — "
                "it is behind 1000 result envelopes")
            turns = 0
            while not maps["registered"] and turns < 5:
                server.turn()
                turns += 1
            assert maps["registered"] == ["hostA:gpu0"], maps["registered"]
            assert turns <= 1, (
                f"the REGISTER took {turns} admission turns at FIFO position 1")
            # [R1 AUDIT] VALUES, not key presence. `"x" in mapping` is satisfied
            # by a key bound to None, which is a registration that recorded the
            # identity and lost the connection it names.
            assert maps["worker_by_sock"][peers[0].srv] == "hostA:gpu0"
            assert maps["fs_by_worker"].get("hostA:gpu0") is peers[0].srv_fs, (
                f"the identity is bound to "
                f"{maps['fs_by_worker'].get('hostA:gpu0')!r}, not to this "
                f"connection's own framed socket")
            assert maps["wconn_by_worker"].get("hostA:gpu0") is not None, (
                "the identity is present in the eligible pool with no worker "
                "connection behind it")
            # …and the 1000-envelope backlog is UNTOUCHED: the REGISTER never
            # entered `inbound` at all.
            assert inbound.qsize() >= 1000, (
                "the admission path consumed data-plane envelopes")
        finally:
            restore()
            stop.set()
            for p in peers:
                p.close()


def fair6_arm2_recovering_worker_is_not_starved():
    """ARM 2 — the same, on the same `<= p`-turn basis, for a RECOVERING worker
    whose prior socket was just dropped.

    WRONG INPUT THAT REDS IT: keying the fast path on "socket not in
    `worker_by_sock`" instead of the reader-local first-frame state — a reconnect
    then races the eviction and is misclassified. (Beta, R1-B: do NOT key this on
    `worker_by_sock`; the original race-avoidance reason still stands.)"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord, peers, inbound, admission, server, stop, restore, maps = \
            _fair6_bench(tmp, n_conns=2, backlog=800)
        try:
            peers[0].send(_register_msg("hostA:gpu0"))
            assert _wait(lambda: admission.qsize() == 1, timeout=5.0)
            server.turn()
            assert maps["registered"] == ["hostA:gpu0"]
            # the prior socket is dropped (Defect A §12's duplicate resolved)
            coord._drop_conn(
                peers[0].srv, {peers[0].srv: peers[0].srv_fs},
                maps["worker_by_sock"], maps["fs_by_worker"],
                maps["wconn_by_worker"], maps["registered"],
                stage_idx=0, stage_assigned=True,
                eligible_fn=lambda: list(maps["wconn_by_worker"].values()),
                conn_state=peers[0].state, reader_exit=None)
            assert "hostA:gpu0" not in maps["fs_by_worker"]
            # …and the RECONNECT, on a new socket, still first-frame
            peers[1].send(_register_msg("hostA:gpu0"))
            assert _wait(lambda: admission.qsize() == 1, timeout=5.0), (
                "the reconnecting worker's REGISTER did not take the fast path")
            server.turn()
            # [R1 AUDIT] VALUE, not key presence: the identity must be rebound to
            # the NEW connection. A key restored to the evicted socket's framed
            # socket is a reconnect that registered nothing.
            assert maps["fs_by_worker"].get("hostA:gpu0") is peers[1].srv_fs, (
                f"the reconnect never became authoritative on its own socket: "
                f"{maps['fs_by_worker'].get('hostA:gpu0')!r}")
            assert maps["worker_by_sock"][peers[1].srv] == "hostA:gpu0"
        finally:
            restore()
            stop.set()
            for p in peers:
                p.close()


def fair6_arm3_latency_is_independent_of_queue_depth():
    """ARM 3 — INDEPENDENCE. REGISTER latency is measured at `inbound` depths of
    ~0, ~1/2 maxsize and maxsize, and does NOT grow with depth — asserted
    against TURN COUNT rather than wall-clock seconds.

    THIS IS THE CERTIFIABLE STRUCTURAL PROPERTY; the seconds figure is not, and
    NO ARM MAY ASSERT AN ABSOLUTE 180 s CEILING: `M_i`, `K_i` and `m_i` are
    unbounded by this repair, so production latency stays OBSERVABLE rather than
    mathematically capped.

    WRONG INPUT THAT REDS IT: any design whose register latency is a function of
    queue depth; or an arm premised on the withdrawn absolute claim, which no arm
    can test."""
    turns_by_depth = {}
    for depth in (0, 512, 1024):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            coord, peers, inbound, admission, server, stop, restore, maps = \
                _fair6_bench(tmp, n_conns=1, backlog=depth)
            try:
                peers[0].send(_register_msg("hostA:gpu0"))
                assert _wait(lambda: admission.qsize() == 1, timeout=5.0), (
                    f"depth={depth}: the REGISTER never reached the channel")
                turns = 0
                while not maps["registered"] and turns < 10:
                    server.turn()
                    turns += 1
                assert maps["registered"], f"depth={depth}: never registered"
                turns_by_depth[depth] = turns
            finally:
                restore()
                stop.set()
                for p in peers:
                    p.close()
    assert set(turns_by_depth.values()) == {1}, (
        f"REGISTER latency varies with `inbound` depth: {turns_by_depth} — the "
        f"independence property is what this repair actually claims")


def fair6_arm4_reconnect_storm_cannot_starve_the_control_plane():
    """ARM 4 — NOT-STARVING. A reconnect storm much larger than `A_max`, with
    EVERY `m_i` scripted LONGER than `D_adm` (the R3.2 worst case, one
    disposition per turn), still leaves the control plane running: each admission
    turn's contribution is bounded and the loop keeps taking turns.

    WRONG INPUT THAT REDS IT: the R1 UNBUDGETED drain — the loop then services
    registrations until the channel empties, and monopolization has simply moved
    one queue upward."""
    D_ADM, N = 0.02, 24
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord, peers, inbound, admission, server, stop, restore, maps = \
            _fair6_bench(tmp, n_conns=N, backlog=0, d_adm=D_ADM, a_max=8,
                         register_cost=D_ADM * 3)      # m_i > D_adm, always
        try:
            for i, p in enumerate(peers):
                p.send(_register_msg(f"hostA:gpu{i}"))
            assert _wait(lambda: admission.qsize() >= N, timeout=20.0), (
                f"only {admission.qsize()} of {N} REGISTERs reached the channel")
            contributions = []
            while len(maps["registered"]) < N:
                t0 = time.perf_counter()
                done = server.turn()
                contributions.append((time.perf_counter() - t0, done))
                if done == 0:
                    break
            assert len(maps["registered"]) == N, (
                f"{len(maps['registered'])} of {N} registrations completed")
            assert all(d >= 1 for _c, d in contributions), (
                f"an admission turn made NO progress: {contributions}")
            # the R3.2 worst case really did obtain — one per turn — and the
            # contribution is D_adm + AT MOST ONE overrun registration.
            assert max(d for _c, d in contributions) == 1, (
                f"the scripted `m_i` did not exceed `D_adm`: {contributions}")
            for cost, done in contributions:
                assert cost <= D_ADM + D_ADM * 3 + 0.05, (
                    f"an admission turn contributed {cost:.3f}s, more than "
                    f"D_adm + one overrun registration")
        finally:
            restore()
            stop.set()
            for p in peers:
                p.close()


def fair6_arm5_amax_is_a_maximum_and_one_is_the_floor():
    """ARM 5 — [R3.2] `A_max` is a MAXIMUM, and ONE disposition is the progress
    FLOOR. Three assertions: dispositions per turn <= `A_max`; the turn's
    contribution is `<= D_adm + at most one overrun registration`; and with a
    single `m_i` scripted LONGER than `D_adm`, EXACTLY ONE disposition still
    occurs on that turn and the channel drains at one per turn WITHOUT STALLING.

    WRONG INPUT THAT REDS IT: testing the deadline BEFORE the first disposition
    (`while perf_counter() < deadline and adm_done < A_max`). A registration
    slower than `D_adm` then yields ZERO dispositions per turn and the channel
    NEVER DRAINS. Or removing either bound — `D_adm` alone does not cap ledger
    writes under a warm page cache, which is why `A_max` exists."""
    # (a) the cap holds
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord, peers, inbound, admission, server, stop, restore, maps = \
            _fair6_bench(tmp, n_conns=12, backlog=0, d_adm=5.0, a_max=4)
        try:
            for i, p in enumerate(peers):
                p.send(_register_msg(f"hostA:gpu{i}"))
            assert _wait(lambda: admission.qsize() >= 12, timeout=20.0)
            first = server.turn()
            assert first == 4, f"{first} dispositions in one turn against A_max=4"
        finally:
            restore()
            stop.set()
            for p in peers:
                p.close()

    # (b) the floor holds, driven positively: m_i > D_adm
    D_ADM = 0.01
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord, peers, inbound, admission, server, stop, restore, maps = \
            _fair6_bench(tmp, n_conns=5, backlog=0, d_adm=D_ADM, a_max=8,
                         register_cost=D_ADM * 5)
        try:
            for i, p in enumerate(peers):
                p.send(_register_msg(f"hostA:gpu{i}"))
            assert _wait(lambda: admission.qsize() >= 5, timeout=20.0)
            per_turn = [server.turn() for _ in range(5)]
            assert per_turn == [1, 1, 1, 1, 1], (
                f"the channel did not drain at one disposition per turn: "
                f"{per_turn} — a zero anywhere means the deadline is being "
                f"tested before the FIRST disposition and the queue can stall")
            assert len(maps["registered"]) == 5, maps["registered"]
        finally:
            restore()
            stop.set()
            for p in peers:
                p.close()

    # (c) the live loop is the one that was just modelled: the deadline test is
    # guarded by `adm_done > 0`.
    tree = ast.parse(_live_src())
    serve = _func_node(tree, "RangeMinerCoordinator", "serve_trial")
    guarded = False
    for node in ast.walk(serve):
        if isinstance(node, ast.While) and "_adm_done" in ast.unparse(node.test):
            body_src = ast.unparse(node.body[0]) if node.body else ""
            guarded = ("_adm_done > 0" in body_src
                       and "_adm_deadline" in body_src)
    assert guarded, (
        "the admission deadline is not guarded by `_adm_done > 0` — a "
        "registration slower than `D_adm` would yield zero dispositions per "
        "turn and the channel would never drain")


def fair6_arm6_nothing_is_dropped():
    """ARM 6 — NOTHING IS DROPPED: every REGISTER put on the channel is
    eventually disposed of, across iterations, in FIFO order.

    WRONG INPUT THAT REDS IT: bounding the admission channel — a `Full` case then
    reappears on the control path, which is exactly what the `SimpleQueue` choice
    exists to prevent."""
    N = 9
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord, peers, inbound, admission, server, stop, restore, maps = \
            _fair6_bench(tmp, n_conns=N, backlog=0, a_max=2)
        try:
            for i, p in enumerate(peers):
                p.send(_register_msg(f"hostA:gpu{i}"))
            assert _wait(lambda: admission.qsize() >= N, timeout=20.0)
            while len(maps["registered"]) < N:
                if server.turn() == 0:
                    break
            assert len(maps["registered"]) == N, (
                f"{len(maps['registered'])} of {N} REGISTERs were disposed of")
            assert maps["registered"] == admission.put_order, (
                f"dispositions were not FIFO over the CHANNEL: disposed "
                f"{maps['registered']} against entry order "
                f"{admission.put_order}")
            assert admission.qsize() == 0
        finally:
            restore()
            stop.set()
            for p in peers:
                p.close()


def fair6_arm7_later_register_on_a_bound_socket_has_no_priority():
    """ARM 7 — ORDERING. REGISTER -> result -> second REGISTER on ONE connection
    are disposed of IN THAT ORDER; the second REGISTER takes NO priority and
    reaches `_serve_register`'s existing `reject_rebind` / idempotent-"ok"
    branches unchanged.

    WRONG INPUT THAT REDS IT: letting a later REGISTER on a bound socket take the
    fast path — P-REG's premise ("no earlier frame of this connection exists")
    would then be false and a REGISTER could overtake that connection's own
    result."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord, peers, inbound, admission, server, stop, restore, maps = \
            _fair6_bench(tmp, n_conns=1, backlog=0)
        p = peers[0]
        try:
            p.send(_register_msg("hostA:gpu0"))
            assert _wait(lambda: admission.qsize() == 1, timeout=5.0)
            server.turn()
            assert maps["registered"] == ["hostA:gpu0"]
            p.send(_inline_result("hostA:gpu0", "s0", 0, 0, 10))
            p.send(_register_msg("hostA:gpu0"))
            assert _wait(lambda: inbound.qsize() >= 2, timeout=5.0), (
                "the later frames did not both take `inbound`")
            kinds = []
            while True:
                try:
                    e = inbound.get_nowait()
                except _queue.Empty:
                    break
                kinds.append(getattr(e[2], "message_type", e[0]))
            assert kinds == ["sub_stripe_result", "register"], (
                f"the second REGISTER overtook the connection's own result: "
                f"{kinds}")
            assert admission.qsize() == 0, (
                "a later REGISTER on a BOUND socket took the fast path")
        finally:
            restore()
            stop.set()
            for p in peers:
                p.close()


def fair6_arm8_the_register_fence():
    """ARM 8 — THE FENCE. A stub sends REGISTER and a result back-to-back with no
    pause; the result is NEVER disposed of before the REGISTER.

    WRONG INPUT THAT REDS IT: removing `_await_register_disposition` — the reader
    then decodes the result while its REGISTER is still queued, and P-REG's "no
    LATER frame of this connection is even decoded" becomes false."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord, peers, inbound, admission, server, stop, restore, maps = \
            _fair6_bench(tmp, n_conns=1, backlog=0)
        p = peers[0]
        try:
            p.send(_register_msg("hostA:gpu0"))
            p.send(_inline_result("hostA:gpu0", "s0", 0, 0, 10))
            assert _wait(lambda: admission.qsize() == 1, timeout=5.0)
            # THE FENCE: while the REGISTER is undisposed, the result stays ON
            # THE WIRE — it is not in `inbound` and it has not been decoded.
            time.sleep(0.4)
            assert inbound.qsize() == 0, (
                "the result was decoded and delivered while the REGISTER was "
                "still undisposed — the fence is not holding")
            server.turn()
            assert _wait(lambda: inbound.qsize() == 1, timeout=5.0), (
                "the result never arrived after the disposition — the fence "
                "does not release")
        finally:
            restore()
            stop.set()
            for p in peers:
                p.close()


def fair6_arm9_one_register_handler():
    """ARM 9 — STRUCTURE. `admission_queue_high_water` is reported, and AST
    asserts `_serve_register` has EXACTLY ONE CALLER — the extracted
    `_serve_register_frame`.

    WRONG INPUT THAT REDS IT: copying the register block instead of extracting
    it — two paths that can diverge, which is the whole reason the extraction is
    part of the design rather than an implementation detail."""
    tree = ast.parse(_live_src())
    callers = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "_serve_register"):
                callers.append(fn.name)
    assert callers == ["_serve_register_frame"], (
        f"`_serve_register` has callers {callers}; exactly one is permitted and "
        f"it must be the extracted frame handler")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord = _coord(tmp)
        coord.note_admission_queue_occupancy(4)
        coord.note_admission_queue_occupancy(2)
        m = coord.staging_backpressure_metrics()
    assert m["admission_queue_high_water"] == 4, m


def fair6_arm10_vacuity_the_backlog_was_real():
    """ARM 10 — VACUITY. The backlog was real: `drain_deadline_hits > 0` AND
    `inbound_qsize_high_water >= 0.9 x maxsize` during a run under pressure.

    WRONG INPUT THAT REDS IT: a gate that passes because the queue was
    accidentally shallow."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord, peers, inbound, admission, server, stop, restore, maps = \
            _fair6_bench(tmp, n_conns=1, backlog=1000)
        try:
            coord.note_inbound_occupancy(inbound.qsize())
            m = coord.staging_backpressure_metrics()
            assert m["inbound_qsize_high_water"] >= 0.9 * 1024, (
                f"the backlog reached only {m['inbound_qsize_high_water']} of "
                f"1024 — arms 1-3 would be measuring an empty queue")
        finally:
            restore()
            stop.set()
            for p in peers:
                p.close()
    # the deadline-hit half is measured on the REAL loop, under FAIR-1's load,
    # and is asserted there; here we assert the counter EXISTS and is wired, so
    # a build that never emits it cannot pass FAIR-1 by silently reporting zero.
    sl = COORD.ServeLoopTiming()
    sl.note_drain_stop("deadline")
    assert sl.metrics()["drain_deadline_hits"] == 1


def fair6_arm11_the_fence_is_not_charged_to_S():
    """ARM 11 — [R3.1] CLASSIFICATION. A REGISTER enters the admission queue while
    `inbound` is NON-FULL and its disposition is delayed past `S`: the
    accumulator is UNCHANGED, no `inbound_saturation_timeout` fires, and the
    fence still exits on `reader_stop`/trial-terminal state.

    WRONG INPUT THAT REDS IT: charging `_await_register_disposition` to `S` — a
    slow registration ledger write then manufactures an INGRESS-saturation
    terminal on a coordinator whose `inbound` was NEVER FULL, inverting the
    classification that terminal exists to establish."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with _CapturedLog() as cap:
            coord, peers, inbound, admission, server, stop, restore, maps = \
                _fair6_bench(tmp, n_conns=1, backlog=0)
            p = peers[0]
            try:
                p.send(_register_msg("hostA:gpu0"))
                assert _wait(lambda: admission.qsize() == 1, timeout=5.0)
                time.sleep(0.6)
                assert not inbound.full(), "the arm is vacuous: inbound is full"
                assert p.state.inbound_saturation_spent_s == 0.0, (
                    f"the fence charged {p.state.inbound_saturation_spent_s}s "
                    f"to the ingress budget")
                assert p.thread.is_alive(), "the fence terminated the reader"
                # …and it EXITS on stop, rather than waiting for ever.
                stop.set()
                p.join(timeout=5.0)
            finally:
                restore()
                for q in peers:
                    q.close()
        assert not cap.events("EMERGENCY_TERMINAL_REQUEST"), (
            "a disposition wait raised an ingress terminal")


def fair6_arm12_red_register_waits_behind_the_whole_fifo():
    """ARM 12 — RED. On pinned `2b0d2dc`, arm 1 fails: the REGISTER waits behind
    the WHOLE FIFO, because the pinned reader has no admission channel at all and
    the pinned drain consumes register frames in queue order.

    A drifted anchor => `AnchorUnavailable` => UNAVAILABLE, never PASS."""
    _assert_pinned_carries_the_defects()
    src = _pinned_src()
    fn = _func_node(ast.parse(src), "RangeMinerCoordinator", "_conn_reader_loop")
    g = dict(vars(COORD))
    exec(compile(textwrap.dedent(ast.get_source_segment(src, fn)),
                 f"<pinned {PINNED_COMMIT}>", "exec"), g)            # noqa: S102
    pinned_reader = g["_conn_reader_loop"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        coord = _coord(tmp)
        inbound = _queue.Queue(maxsize=1024)
        for _ in range(1000):
            inbound.put(("msg", None, None, None))
        stop = threading.Event()
        sp = socket.socketpair()
        fs = MinerFramedSocket(sp[0])
        th = threading.Thread(target=types.MethodType(pinned_reader, coord),
                              args=(fs, sp[0], inbound, stop, {}), daemon=True)
        th.start()
        MinerFramedSocket(sp[1]).send_msg(_register_msg("hostA:gpu0"))
        time.sleep(0.5)
        depth_ahead = inbound.qsize()
        entries = [inbound.get_nowait() for _ in range(depth_ahead)]
        stop.set()
        th.join(timeout=5.0)
        for s in sp:
            try:
                s.close()
            except OSError:
                pass
    positions = [i for i, e in enumerate(entries)
                 if getattr(e[2], "message_type", None) == "register"]
    assert positions and positions[0] >= 1000, (
        f"the PINNED REGISTER sat at FIFO position {positions[:1]}, not behind "
        f"the 1000-envelope backlog — the starvation surface is absent from the "
        f"anchor and no RED credit is due")


# --- arms 13-14: FIRST-FRAME PRIORITY (R1-B / D2 RULED) ---------------------
#
# WHY ARM 7 CANNOT DETECT D2, stated because it is the whole reason these exist.
# Arm 7 drives `REGISTER -> result -> REGISTER`, and that middle result
# NECESSARILY increments the delivery counter before the second REGISTER
# arrives. Under `envelopes_delivered == 0` the second REGISTER therefore did NOT
# take the admission channel, arm 7 went green, and the predicate looked correct.
# Remove the intervening result and the counter is still zero — so REGISTER #2
# took the admission channel again, and the "at most once per connection"
# property the P-REG proof rests on was false. Arm 13 is arm 7 with the result
# deleted; that one deletion is the entire difference.
# ---------------------------------------------------------------------------
def _d2_mutant_reader():
    """LIVE source with the first-frame guard neutralised — i.e. any REGISTER may
    take admission, which is what `envelopes_delivered == 0` permitted for a
    back-to-back pair. Executed with the coordinator module's REAL globals."""
    return _mutated_reader(
        lambda s: s.replace("                        and is_first_frame\n",
                            "                        and True\n", 1))


def fair6_arm13_back_to_back_registers_take_admission_once():
    """ARM 13 — [R1-B / D2 RULED] `REGISTER -> REGISTER`, BACK-TO-BACK, WITH NO
    INTERVENING RESULT. REGISTER #1 takes the admission channel; REGISTER #2 does
    NOT — it goes to `inbound` and reaches `_serve_register`'s existing
    rebind/idempotent branches unchanged.

    P-1 is FIRST-FRAME REGISTER PRIORITY. The admission route may be used at most
    ONCE per connection and only for the connection's first decoded application
    frame; `envelopes_delivered == 0` was a proxy for that and did not hold it.

    WRONG INPUT THAT REDS IT: any predicate that lets a non-first frame reach the
    admission channel — the delivery counter, `worker_by_sock` emptiness, or an
    eligibility flag cleared inside the admission branch instead of at the
    decode. The mutant below removes the first-frame term outright and MUST be
    detected."""
    def _drive(reader):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            coord, peers, inbound, admission, server, stop, restore, maps = \
                _fair6_bench(tmp, n_conns=1, backlog=0, reader=reader)
            p = peers[0]
            try:
                p.send(_register_msg("hostA:gpu0"))
                p.send(_register_msg("hostA:gpu0"))
                assert _wait(lambda: admission.qsize() == 1, timeout=5.0), (
                    "REGISTER #1 never reached the admission channel")
                # THE FENCE, re-proved here: while #1 is undisposed, #2 stays ON
                # THE WIRE. Without this the arm could not tell "#2 took inbound"
                # from "#2 was not read yet".
                time.sleep(0.4)
                assert inbound.qsize() == 0, (
                    "REGISTER #2 was decoded while #1 was still undisposed")
                server.turn()
                assert maps["registered"] == ["hostA:gpu0"], maps["registered"]
                assert _wait(lambda: inbound.qsize() == 1, timeout=5.0), (
                    "REGISTER #2 never reached `inbound` after the fence "
                    "released — it took the admission channel instead")
                e = inbound.get_nowait()
                assert getattr(e[2], "message_type", None) == "register", e
                # THE DECIDING ASSERTION, and it is race-free: the recording
                # channel logs EVERY put, so "how many times did this connection
                # use the admission route" is counted, never inferred from a
                # queue depth sampled at one instant.
                assert admission.put_order == ["hostA:gpu0"], (
                    f"the admission route was used {len(admission.put_order)} "
                    f"times on ONE connection: {admission.put_order}")
            finally:
                restore()
                stop.set()
                for q in peers:
                    q.close()

    _drive(None)
    # Built outside `_mutant_red`, which credits ANY exception as detection: a
    # mutation that stopped matching live source would otherwise read as
    # "detected" instead of failing the arm.
    mutant = _d2_mutant_reader()
    _mutant_red(lambda: _drive(mutant),
                "first-frame guard removed from the admission condition")


def fair6_arm14_admission_eligibility_is_consumable_once():
    """ARM 14 — [R1-B] STRUCTURE. Admission-priority eligibility is consumable
    ONCE ONLY per reader connection, by construction and not by inspection.

    Four properties over LIVE source, each of which independently restores D2 if
    it fails:
      (a) the flag is armed exactly once, OUTSIDE the read loop;
      (b) it is cleared exactly once, as a DIRECT statement of the loop body — a
          clear nested inside `if`/`try`/`while` is a clear that some path skips;
      (c) it is never re-armed anywhere in the function — there is no second
          first frame;
      (d) the admission test reads the SNAPSHOT and mentions neither the delivery
          counter nor `worker_by_sock` (Beta: do NOT key this on
          `worker_by_sock`; the original race-avoidance reason stands).

    WRONG INPUT THAT REDS IT: clearing the flag inside the admission branch. The
    flag would then still be set for a `result -> REGISTER` connection, and the
    admission route would be reachable on a frame that is not the first — (b)
    catches it, and the behavioural arm 13 would not, because arm 13's first
    frame IS a REGISTER."""
    tree = ast.parse(_live_src())
    fn = _func_node(tree, "RangeMinerCoordinator", "_conn_reader_loop")
    FLAG, SNAP = "first_frame", "is_first_frame"

    def _assigns(scope, name):
        return [n for n in ast.walk(scope)
                if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                and any(getattr(t, "id", None) == name
                        for t in (n.targets if isinstance(n, ast.Assign)
                                  else [n.target]))]

    # The read loop: the `while` containing the `recv_msg` call.
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.While)
             and any(isinstance(c, ast.Call)
                     and getattr(c.func, "attr", None) == "recv_msg"
                     for c in ast.walk(n))]
    assert len(loops) == 1, f"expected one read loop, found {len(loops)}"
    loop = loops[0]

    arms = _assigns(fn, FLAG)
    armed = [a for a in arms if getattr(a.value, "value", None) is True]
    cleared = [a for a in arms if getattr(a.value, "value", None) is False]
    assert len(arms) == 2 and len(armed) == 1 and len(cleared) == 1, (
        f"`{FLAG}` is assigned {len(arms)} times "
        f"({len(armed)} True / {len(cleared)} False); exactly one arm and one "
        f"clear are permitted")
    # (a) armed outside the loop…
    assert armed[0] not in ast.walk(loop), (
        f"`{FLAG}` is armed INSIDE the read loop — it would re-arm on every "
        f"frame and every REGISTER would take the admission channel")
    # (b) …cleared as a DIRECT statement of the loop body.
    assert cleared[0] in loop.body, (
        f"`{FLAG} = False` is nested inside a branch of the read loop; a clear "
        f"some path can skip is not a clear")
    # (c) never re-armed inside the loop (implied by len(armed) == 1 + (a), and
    # asserted separately so the failure message names the real hazard).
    assert not [a for a in _assigns(loop, FLAG)
                if getattr(a.value, "value", None) is True], (
        f"`{FLAG}` is re-armed inside the read loop")

    snaps = _assigns(fn, SNAP)
    assert len(snaps) == 1 and getattr(snaps[0].value, "id", None) == FLAG, (
        f"`{SNAP}` must be assigned exactly once, from `{FLAG}`: "
        f"{[ast.unparse(s) for s in snaps]}")
    # The snapshot is taken BEFORE the clear, in the same block, adjacently.
    body_idx = {id(stmt): i for i, stmt in enumerate(loop.body)}
    assert body_idx[id(snaps[0])] + 1 == body_idx[id(cleared[0])], (
        "the snapshot and the clear are not adjacent statements of the loop "
        "body; anything between them is a path on which they can diverge")

    # (d) the admission condition.
    tests = [n.test for n in ast.walk(fn) if isinstance(n, ast.If)
             and "admission.put" in ast.unparse(n)]
    assert len(tests) == 1, f"expected one admission branch, found {len(tests)}"
    cond = ast.unparse(tests[0])
    assert SNAP in cond, f"the admission condition does not test {SNAP}: {cond}"
    for banned in ("envelopes_delivered", "worker_by_sock", "bound_worker_id"):
        assert banned not in cond, (
            f"the admission condition reads `{banned}`: {cond}")
    assert "envelopes_delivered" not in ast.unparse(fn), (
        "the superseded delivery counter is still present in the reader; a "
        "second, divergent notion of `first` is exactly what D2 was")


# ===========================================================================
# FAIR-3 — no lease/admission semantic regression (§11.8, 5 arms)
#
# The withdrawn v1 criterion — the cross-check `loop_now_age_max <=
# A + D + M_max + K` — asserted the SUPERSEDED recurrence and is not used.
# ===========================================================================
NO_TOUCH_DEFS = (
    "claim_stripe", "schedule_pending_stripes", "renew_lease",
    "_renew_active_lease", "process_lease_expiry",
    "_handle_stripe_failure_locked", "_execution_set_expected_workers",
)


def _def_digests(src: str) -> Dict[str, str]:
    """Per-definition AST digest: `sha256(ast.unparse(node))` for every
    module-level function, every method and every class body.

    AST rather than text, because `2389b61` reverted a fix by WHOLE-BLOCK
    REPLACEMENT and a text anchor would have gone green. Comments and docstring
    reflow move no digest; a changed statement moves exactly one."""
    tree = ast.parse(src)
    out: Dict[str, str] = {}

    def _walk(node, prefix):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = f"{prefix}{child.name}"
                stripped = ast.parse(_strip_comments(
                    textwrap.dedent(ast.unparse(child))))
                out[key] = hashlib.sha256(
                    ast.unparse(stripped).encode()).hexdigest()
            elif isinstance(child, ast.ClassDef):
                _walk(child, f"{prefix}{child.name}.")

    _walk(tree, "")
    return out


def fair3_arm1_certified_suites_are_green():
    """ARM 1 — the EXISTING CERTIFIED SUITES run unmodified and green.

    Run here: `tests/test_s172_f1_lease_origin.py` (F1/F2 lease origin, 18/18)
    and `tests/test_s172_f1_f2_active_lease.py` (16/16) — the two that carry the
    lease semantics this amendment must not disturb, and the two that are fast
    enough to be part of a gate rather than a separate battery. The
    admission-liveness (16/16), execution-set (34/34), phase-4 (63/63),
    back-pressure (>= 50, and green) and Part-B suites are run in the REGRESSION
    BATTERY at
    final state and reported there; that division is stated rather than left for
    a reader to infer.

    WRONG INPUT THAT REDS IT: any behavioural regression in the lease path."""
    if SKIP_SUBPROCESS:
        raise AssertionError(
            "UNAVAILABLE: ATTEMPT6_SKIP_SUBPROCESS_SUITES=1 — the certified "
            "suites were not run, and a skipped arm is not a passing one")
    for suite, expect in (("tests/test_s172_f1_lease_origin.py", "18/18"),
                          ("tests/test_s172_f1_f2_active_lease.py", "16/16")):
        p = subprocess.run([sys.executable, "-u", suite], cwd=_ROOT,
                           capture_output=True, text=True, timeout=900)
        assert p.returncode == 0, (
            f"{suite} exited {p.returncode}\n{p.stdout[-3000:]}")
        assert expect in p.stdout, (
            f"{suite} did not report {expect}:\n{p.stdout[-2000:]}")


def fair3_arm2_no_touch_ast_proof():
    """ARM 2 — AST over live source: this amendment's diff touches NONE of
    `claim_stripe`, `schedule_pending_stripes`, `renew_lease`,
    `_renew_active_lease`, `process_lease_expiry`,
    `_handle_stripe_failure_locked`, `_execution_set_expected_workers`, or the
    ADMISSION BLOCK inside `serve_trial`. G-DA-NO-TOUCH's shape, applied here.

    WRONG INPUT THAT REDS IT: an edit inside a named function reds EVEN IF EVERY
    BEHAVIOURAL TEST PASSES — `2389b61` reverted a fix by whole-block replacement
    and a text anchor would have gone green."""
    pinned = _def_digests(_pinned_src())
    live = _def_digests(_live_src())
    for name in NO_TOUCH_DEFS:
        keys = [k for k in pinned if k.split(".")[-1] == name]
        assert keys, f"{name} is not present in the pinned module"
        for k in keys:
            assert k in live, f"{k} was DELETED by this amendment"
            assert pinned[k] == live[k], (
                f"{k} changed: pinned {pinned[k][:12]} vs live {live[k][:12]} — "
                f"this is a NO-TOUCH surface")
    # the admission block inside serve_trial, compared as a subtree
    def _admission_block(src):
        serve = _func_node(ast.parse(src), "RangeMinerCoordinator", "serve_trial")
        for node in ast.walk(serve):
            if (isinstance(node, ast.If)
                    and "len(eligible) < expected_workers" in
                    ast.unparse(node.test)):
                return _strip_comments(textwrap.dedent(ast.unparse(node)))
        return None

    pre, post = _admission_block(_pinned_src()), _admission_block(_live_src())
    assert pre and post, "the bounded-admission block could not be located"
    assert hashlib.sha256(pre.encode()).hexdigest() == \
        hashlib.sha256(post.encode()).hexdigest(), (
        "the §4.3 bounded-admission block changed; `worker_admission_timeout` "
        "is DO-NOT-WIDEN and its enforcement is DO-NOT-TOUCH")


def fair3_arm3_serve_register_is_byte_unchanged():
    """ARM 3 — [R2.4] `_serve_register` itself is UNCHANGED; only its CALL SITE
    was extracted into `_serve_register_frame`.

    WRONG INPUT THAT REDS IT: modifying the register body while claiming an
    extraction — which is how a "pure refactor" becomes a behavioural change
    nobody reviewed."""
    pinned = _def_digests(_pinned_src())
    live = _def_digests(_live_src())
    key = "RangeMinerCoordinator._serve_register"
    assert pinned[key] == live[key], (
        f"`_serve_register` changed: {pinned[key][:12]} -> {live[key][:12]}")


def fair3_arm4_loop_now_age_still_functions():
    """ARM 4 — [R2.5] `loop_now_age_max` still functions and satisfies the
    CURRENT recurrence under load.

    RECORDED AS FAVOURABLE, NOT NEUTRAL: M-1 makes iterations shorter, so
    `loop_now_age_max` — 940.957 s in attempt 5 — falls. The F1 repair is
    unaffected either way (the scheduler reads its own clock), but the staleness
    the instrument exists to expose is REDUCED, and the two repairs compose.

    WRONG INPUT THAT REDS IT: asserting the withdrawn `A + D + M_max + K`
    form."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out, _probe, _seam = _fairness_run(tmp, m_seconds=0.02, n_workers=1,
                                           serve_timeout=8.0)
    sl = out.get("serve_loop_timing") or {}
    assert "loop_now_age_max" in sl, "the instrument was removed"
    # [R1 AUDIT — the third-instance class, found here] The arm previously
    # asserted `>= 0.0` and `<= bound`. AN INSTRUMENT FROZEN AT ITS CONSTRUCTOR
    # VALUE SATISFIES BOTH: `note_loop_now_age` returns early when `_last_top` is
    # None, so an unwired instrument reports a permanent 0.0 and this arm went
    # green while measuring nothing. The bound is only evidence if the quantity
    # was actually observed, so observation is asserted FIRST — a strictly
    # positive age AND the wall label the instrument stamps when it updates.
    assert sl["loop_now_age_max"] > 0.0, (
        "loop_now_age_max is exactly 0.0 after a real serve-loop run — the "
        "instrument never fired (`_last_top` unset), so the bound below would "
        "be a statement about nothing")
    assert sl.get("loop_now_age_at") is not None, (
        "the instrument reports a maximum with no wall label; it was not "
        "updated through `note_loop_now_age`")
    # …and the frozen state the two assertions above now reject is REACHABLE,
    # not hypothetical: an instrument whose `_last_top` was never marked reports
    # a permanent 0.0 with no label, which is precisely what the old `>= 0.0`
    # accepted. Proved on a bare instrument, so it costs no second serve loop.
    _frozen = COORD.ServeLoopTiming()
    _frozen.note_loop_now_age(time.time())
    assert (_frozen.metrics()["loop_now_age_max"] == 0.0
            and _frozen.metrics()["loop_now_age_at"] is None), (
        "an unmarked instrument no longer reports a frozen 0.0, so the "
        "observation assertions above have no failure mode and are vacuous")
    bound, terms = _recurrence_bound(out, 0.02)
    assert math.isfinite(bound), f"the recurrence bound is not finite: {terms}"
    assert sl["loop_now_age_max"] <= bound, (
        f"loop_now_age_max {sl['loop_now_age_max']:.3f}s exceeds the CURRENT "
        f"recurrence {bound:.3f}s; terms={terms}")


def fair3_arm5_lease_origin_is_still_claim_now():
    """ARM 5 — the F1 lease origin remains `claim_now`: a drain deadline that let
    the loop reach `schedule_pending_stripes` with a STALE `now` would red F1's
    own gates, and the structural property is asserted here directly."""
    tree = ast.parse(_live_src())
    fn = _func_node(tree, "RangeMinerCoordinator", "schedule_pending_stripes")
    origins = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "claim_stripe"):
            for arg in node.args:
                if (isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add)
                        and isinstance(arg.left, ast.Name)):
                    origins.append(arg.left.id)
    assert origins == ["claim_now"], (
        f"the lease is stamped from {origins}, not the scheduler's own fresh "
        f"clock read")
    serve = _func_node(tree, "RangeMinerCoordinator", "serve_trial")
    for node in ast.walk(serve):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None)
                == "schedule_pending_stripes"):
            assert not any(kw.arg == "now" for kw in node.keywords), (
                "the serve loop injects its shared `now` into the scheduler "
                "again — that is the attempt-4 lease-origin defect")


# ===========================================================================
# FAIR-4 — no backpressure regression (§11.9, 5 arms)
#
# The withdrawn v1 falsifier — "a paused worker would be SHED for a
# coordinator-caused wait" — describes an outcome the design no longer has: the
# consequence is now a TRIAL TERMINAL.
# ===========================================================================
def fair4_arm0_the_full_bp_battery():
    """ARM 0 — the full S172-BP battery runs UNMODIFIED: derived-bound
    arithmetic, pause/resume, the F1-R credit reservation and its exact-token
    clear, the F1-R2b pre-decode barrier's one-decoded-envelope bound, the F2
    lease exemption and post-resume grace, and the Option-C retention lifecycle.

    NOTE, STATED PLAINLY RATHER THAN LEFT TO INFERENCE: the suite's tuple-unpack
    patterns were widened from four to five fields, because the reader-exit
    record rides in a fifth tuple position and Python cannot unpack a 5-tuple
    into four names. That is a MECHANICAL harness adaptation forced by a frozen
    production contract; no assertion, threshold or expectation was changed, and
    `docs`/the implementation report carry the programmatic proof of that.

    WRONG INPUT THAT REDS IT: any behavioural backpressure regression."""
    if SKIP_SUBPROCESS:
        raise AssertionError(
            "UNAVAILABLE: ATTEMPT6_SKIP_SUBPROCESS_SUITES=1 — the back-pressure "
            "battery was not run, and a skipped arm is not a passing one")
    p = subprocess.run([sys.executable, "-u",
                        "tests/test_s172_staging_backpressure.py"],
                       cwd=_ROOT, capture_output=True, text=True, timeout=2400)
    assert p.returncode == 0, (
        f"the back-pressure battery exited {p.returncode}\n{p.stdout[-4000:]}")
    # [FIELD-6 OBSERVABILITY REPAIR, TB ruling sequencing item 3] The pin here
    # was the TRANSCRIBED tally `50/50`. The back-pressure suite legitimately
    # grew to 52 when field 6's gate and its mutant arm landed, and a
    # transcribed count reds on every authorized growth while proving nothing a
    # green run does not already prove — `main()` returns 0 only when
    # `passed == total`, which `returncode == 0` above already asserts. It is
    # replaced by the two properties the pin was standing in for, neither of
    # which a growing suite can trip:
    #   * the suite's OWN completion sentinel, printed only on a full pass, and
    #   * a FLOOR on the gate count, so gates being DELETED still reds this.
    # This is the same brittle-count shape Beta ruled on at R4-1; `50 -> 52` is
    # the amendment Beta called wrong there, so it is not the one taken here.
    _tally = re.search(r"(\d+)/(\d+) checks green", p.stdout)
    assert _tally, f"the battery printed no tally at all:\n{p.stdout[-2000:]}"
    _passed, _total = int(_tally.group(1)), int(_tally.group(2))
    assert _passed == _total, (
        f"the battery reported {_passed}/{_total}:\n{p.stdout[-2000:]}")
    assert _total >= 50, (
        f"the back-pressure battery shrank to {_total} gates (was 50 at the "
        f"attempt-6 anchor) — gates were REMOVED, not added:\n{p.stdout[-2000:]}")
    assert "COMPLETION SENTINEL: PASS" in p.stdout, (
        f"the battery printed no PASS sentinel:\n{p.stdout[-2000:]}")


def fair4_arm1_the_one_envelope_bound_survives():
    """ARM 1 — THE ONE-ENVELOPE BOUND SURVIVES. A shorter drain leaves a credited
    envelope in `inbound` longer, so the reader parks at the barrier longer;
    the connection must still hold EXACTLY ONE decoded envelope throughout.

    The count is taken by wrapping the framed socket the PRODUCTION reader is
    handed — the reader itself is untouched — so "did it decode a second
    envelope?" is answered by counting rather than by inference.

    WRONG INPUT THAT REDS IT: letting the barrier decode ahead while parked — the
    F1-R2b round-2 defect, and the bound the §1.2 resume margin is derived
    from."""
    decoded = {"n": 0}

    class _CountingFs:
        def __init__(self, fs):
            self._fs = fs

        def recv_msg(self):
            msg = self._fs.recv_msg()
            decoded["n"] += 1
            return msg

        def __getattr__(self, item):
            return getattr(self._fs, item)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b = _Bench(tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"),
                   fs_wrap=_CountingFs, **_saturating_cfg())
        try:
            _park_at_barrier(b)
            before = decoded["n"]
            # …and now queue a SECOND frame behind the barrier.
            b.peers["hostA:gpu0"].send(
                _inline_result("hostA:gpu0", "runB_s0", 1, 30, 30))
            time.sleep(0.6)
            assert decoded["n"] == before, (
                f"the connections decoded {decoded['n'] - before} further "
                f"envelope(s) while a reservation was undisposed — the "
                f"one-decoded-envelope bound is broken")
            # …and it RELEASES on disposition, so the arm is not passing because
            # the reader was simply dead.
            b.coord.clear_any_resume_credit(disposition="gate")
            assert _wait(lambda: decoded["n"] > before, timeout=5.0), (
                "the barrier never released after the disposition")
        finally:
            b.close()


def fair4_arm2_credit_disposition_happens_once_on_the_exact_token():
    """ARM 2 — CREDIT DISPOSITION HAPPENS EXACTLY ONCE, ON THE EXACT TOKEN,
    across a drain that ends on the DEADLINE with credited envelopes still
    queued — a state the count bound made rare and M-1 makes routine.

    WRONG INPUT THAT REDS IT: moving the clear back to `inbound.put` — the F1-R
    round-1 defect, where the slot is still free and the next FIFO head takes a
    second wake on it."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b = _Bench(tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"),
                   **_saturating_cfg())
        try:
            entry = _park_at_barrier(b)
            token = entry[3]
            assert b.coord.resume_credit_id_for(b.peers["hostA:gpu0"].srv) \
                == token, "the reservation is not held on the delivered token"
            # the drain "ends on the deadline": the envelope stays queued, and
            # the reservation stays held until DISPOSITION.
            time.sleep(0.3)
            assert b.coord.resume_credits_outstanding() == 1, (
                "the reservation was cleared at INGRESS")
            b.coord.dispatch_inbound_result(
                entry[2], b.peers["hostA:gpu0"].srv, "runB", "hostA:gpu0",
                b.wconn_by_worker, lambda: list(b.wconn_by_worker.values()),
                token)
            assert b.coord.resume_credits_outstanding() == 0, (
                "the reservation outlived the disposition of its OWN envelope")
        finally:
            b.close()


def fair4_arm3_capacity_timeout_is_reached_no_later():
    """ARM 3 — the capacity timeout is measured from the OLDEST paused connection
    and is reached NO LATER than before: M-1 makes the loop reach
    `staging_capacity_timeout_expired` SOONER, and the arm asserts that DIRECTION
    rather than assuming it.

    WRONG INPUT THAT REDS IT: a drain change that DEFERS the check."""
    tree = ast.parse(_live_src())
    serve = _func_node(tree, "RangeMinerCoordinator", "serve_trial")
    drain = _drain_while(serve)
    cap_check = None
    for node in ast.walk(serve):
        if (isinstance(node, ast.If)
                and "staging_capacity_timeout_expired" in ast.unparse(node.test)):
            cap_check = node
    assert cap_check is not None, "the capacity-timeout check is gone"
    assert cap_check.lineno < drain.lineno, (
        "the capacity timeout is now checked AFTER the drain — a long drain "
        "would defer the bounded capacity terminal, which is the opposite of "
        "the direction this repair guarantees")
    # behavioural: the latch is still driven by the oldest blocker's age
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        b = _Bench(tmp, worker_ids=("hostA:gpu0", "hostB:gpu0"),
                   **_saturating_cfg(staging_capacity_timeout=0.2))
        try:
            _pause_one(b)
            assert _wait(lambda: b.coord.staging_capacity_timeout_expired(),
                         timeout=10.0), "the bounded capacity wait never latched"
            snap = b.coord.capacity_timeout_snapshot()
            assert snap and snap["paused_count"] >= 1, snap
        finally:
            b.close()


def fair4_arm4_pause_does_not_consume_S():
    """ARM 4 — [R2.1] `S` is NOT consumed while a reader is paused for staging
    capacity; a paused reader accrues zero `inbound_saturation_spent_s`.

    WRONG INPUT THAT REDS IT: charging pause time to `S` — a coordinator-caused
    staging wait would then manufacture an INGRESS TRIAL TERMINAL, which is
    Beta D's classification law violated at trial scope."""
    rxp2_arm7_pause_time_is_not_charged()


def fair4_arm5_admission_path_does_no_staging():
    """ARM 5 — [R2.4] the admission drain does not touch `_admission_lock`
    ordering or the deferred pump: `_serve_register_frame` performs NO staging
    work.

    WRONG INPUT THAT REDS IT: a registration path that reaches `enqueue_staging`
    — the control-plane priority path would then be able to block on the very
    lock the data plane contends for."""
    tree = ast.parse(_live_src())
    fn = _func_node(tree, "RangeMinerCoordinator", "_serve_register_frame")
    called = {getattr(n.func, "attr", None) for n in ast.walk(fn)
              if isinstance(n, ast.Call)}
    for banned in ("enqueue_staging", "_pump_deferred", "_defer_locked",
                   "_release_capacity", "staging_can_accept"):
        assert banned not in called, (
            f"`_serve_register_frame` calls {banned} — the admission path must "
            f"do no staging work")
    src = ast.unparse(fn)
    assert "_admission_lock" not in src, (
        "the admission path acquires the staging admission lock")


# ===========================================================================
# runner
# ===========================================================================
def main() -> int:
    print("=" * 74)
    print("S172 ATTEMPT-6 REMEDIATION — the ten-gate §11 battery")
    print(f"pinned RED anchor : {PINNED_COMMIT}")
    _head = subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
    print(f"live HEAD         : {_head}")
    print("=" * 74)

    # FAIR-5 FIRST: every other RED arm depends on the anchor, so the suite
    # establishes it before anything can be credited against it.
    print("\n-- FAIR-5 — RED authenticity (the anchor every RED arm needs) --")
    anchor_error = None
    try:
        _assert_pinned_carries_the_defects()
        print(f"  anchor OK: {PINNED_COMMIT} still carries every defect surface\n")
    except AnchorUnavailable as e:
        anchor_error = str(e)
        print(f"  anchor UNAVAILABLE: {anchor_error}\n")
    _check("FAIR-5  pinned anchor reproduces every defect surface",
           fair5_anchor_reproduces_every_defect_surface)
    _check("FAIR-5  self-protection refuses repaired source",
           fair5_self_protection_refuses_repaired_source)

    green = {
        "-- RXP-1 — reader-exit provenance --": [
            ("RXP-1/1  every exit carries its own class",
             rxp1_arm1_every_exit_carries_its_own_class),
            ("RXP-1/2  the class reaches WORKER_DISCONNECTED",
             rxp1_arm2_class_reaches_worker_disconnected),
            ("RXP-1/3  shutdown paths emit NO disconnect (absence)",
             rxp1_arm3_shutdown_paths_emit_no_disconnect),
            ("RXP-1/4  mutual exclusivity", rxp1_arm4_mutual_exclusivity),
            ("RXP-1/5  completeness from live source",
             rxp1_arm5_completeness_from_live_source),
            ("RXP-1/6  fail-closed default (+mutant)",
             rxp1_arm6_fail_closed_default),
            ("RXP-1/7  orthogonality", rxp1_arm7_orthogonality),
            ("RXP-1/8  reachability pin", rxp1_arm8_reachability_pin),
            ("RXP-1/9  eof_reap absent", rxp1_arm9_eof_reap_is_absent),
            ("RXP-1/10 E3-under-stop is SHUTDOWN_STOP",
             rxp1_arm10_e3_under_stop_is_shutdown),
            ("RXP-1/11 CLOSE_INTENT emitted when unbound",
             rxp1_arm11_close_intent_emitted_even_when_unbound),
            ("RXP-1/12 THE RACE GATE", rxp1_arm12_the_race_gate),
            ("RXP-1/13 no merge, no causation",
             rxp1_arm13_no_merge_no_causation),
        ],
        "-- RXP-2 — saturation discrimination + the single accumulator --": [
            ("RXP-2/1  transient saturation does not shed",
             rxp2_arm1_transient_saturation_does_not_shed),
            ("RXP-2/2  never a transport class",
             rxp2_arm2_saturation_is_never_a_transport_class),
            ("RXP-2/3  exhaustion is a TRIAL terminal (real serve_trial)",
             rxp2_arm3_exhaustion_is_a_trial_terminal_not_a_shed),
            ("RXP-2/4  the terminal reason leads with the cause",
             rxp2_arm4_terminal_reason_leads_with_the_cause),
            ("RXP-2/5  emergency cardinality (CARD-1/2/3)",
             rxp2_arm5_emergency_cardinality),
            ("RXP-2/6  ONE accumulator, never reset",
             rxp2_arm6_one_accumulator_never_reset),
            ("RXP-2/7  pause time is not charged",
             rxp2_arm7_pause_time_is_not_charged),
            ("RXP-2/8  occupancy sampled DURING the drain",
             rxp2_arm8_occupancy_is_sampled_during_the_drain),
            ("RXP-2/9  the charged list is exhaustive (false-terminal arm)",
             rxp2_arm9_the_charged_list_is_exhaustive),
            ("RXP-2    clean control", rxp2_clean_control),
        ],
        "-- RXP-3 — worker-log sentinel + the pre-REGISTER barrier --": [
            ("RXP-3/1  all present => PASS", rxp3_arm1_all_present_passes),
            ("RXP-3/2  a missing nonce => REFUSAL",
             rxp3_arm2_missing_nonce_refuses),
            ("RXP-3/3  UNAVAILABLE is never rendered as 0",
             rxp3_arm3_unavailable_is_never_zero),
            ("RXP-3/4  a stale nonce fails (log AND release token)",
             rxp3_arm4_stale_nonce_fails),
            ("RXP-3/5  the sentinel routes through the generic emitter",
             rxp3_arm5_sentinel_routes_through_the_generic_emitter),
            ("RXP-3/6  GREEN ordering (no REGISTER before release)",
             rxp3_arm6_green_ordering),
            ("RXP-3/7  THE EARLY-REGISTER MUTANT",
             rxp3_arm7_the_early_register_mutant),
            ("RXP-3/8  AST ordering over live main",
             rxp3_arm8_ast_ordering_over_live_main),
            ("RXP-3/9  [R1-A] split fact: stale sentinel + current nonce",
             rxp3_arm9_split_fact_refuses),
            ("RXP-3/10 [R1-A] current nonce, no sentinel at all",
             rxp3_arm10_nonce_without_any_sentinel_refuses),
            ("RXP-3/11 [R1-A] acceptance reads only the conjunctive count",
             rxp3_arm11_acceptance_reads_only_the_conjunctive_count),
            ("RXP-3/12 [R2] ssh transport failure => UNAVAILABLE",
             rxp3_arm12_ssh_transport_failure_is_unavailable),
            ("RXP-3/13 [R2] an EXECUTED remote probe stays ERROR",
             rxp3_arm13_an_executed_remote_probe_stays_error),
        ],
        "-- FAIR-7 — reasoned EOF: ordered, and never lost --": [
            ("FAIR-7/1 ordered behind its own envelopes",
             fair7_arm1_eof_is_ordered_behind_its_own_envelopes),
            ("FAIR-7/2 survives a full queue",
             fair7_arm2_eof_survives_a_full_queue),
            ("FAIR-7/3 a non-Full exception propagates",
             fair7_arm3_non_full_exception_propagates),
            ("FAIR-7/4 the zombie is closed", fair7_arm4_the_zombie_is_closed),
            ("FAIR-7/5 exhaustion terminal without eviction",
             fair7_arm5_exhaustion_terminal_without_eviction),
            ("FAIR-7/6 the EOF inherits the spend",
             fair7_arm6_eof_inherits_the_spend),
            ("FAIR-7   clean control", fair7_clean_control),
        ],
        "-- FAIR-1 / FAIR-2 — control-plane fairness, two pressure shapes --": [
            ("FAIR-1   count pressure (many cheap)", fair1_count_pressure),
            ("FAIR-2   slow-message pressure (few expensive)",
             fair2_slow_message_pressure),
            ("FAIR-1/2 arm 5: the first get respects the budget",
             fair12_arm5_first_get_respects_the_remaining_budget),
            ("FAIR-1/2 arm 6: control_block is a COMPOSITE",
             fair12_arm6_control_block_is_a_composite),
            ("FAIR-1/2 arm 8: the config terms fail closed [binding detail 2]",
             fair12_arm8_config_terms_fail_closed),
        ],
        "-- FAIR-6 — registration under backlog, BOTH directions --": [
            ("FAIR-6/1  a new worker is not starved",
             fair6_arm1_new_worker_is_not_starved_by_backlog),
            ("FAIR-6/2  a recovering worker is not starved",
             fair6_arm2_recovering_worker_is_not_starved),
            ("FAIR-6/3  latency is independent of queue depth",
             fair6_arm3_latency_is_independent_of_queue_depth),
            ("FAIR-6/4  a reconnect storm cannot starve the control plane",
             fair6_arm4_reconnect_storm_cannot_starve_the_control_plane),
            ("FAIR-6/5  A_max is a MAXIMUM; one is the FLOOR",
             fair6_arm5_amax_is_a_maximum_and_one_is_the_floor),
            ("FAIR-6/6  nothing is dropped", fair6_arm6_nothing_is_dropped),
            ("FAIR-6/7  a later REGISTER has no priority",
             fair6_arm7_later_register_on_a_bound_socket_has_no_priority),
            ("FAIR-6/8  the register fence", fair6_arm8_the_register_fence),
            ("FAIR-6/9  ONE register handler (AST)",
             fair6_arm9_one_register_handler),
            ("FAIR-6/10 vacuity: the backlog was real",
             fair6_arm10_vacuity_the_backlog_was_real),
            ("FAIR-6/11 the fence is not charged to S",
             fair6_arm11_the_fence_is_not_charged_to_S),
            ("FAIR-6/13 [R1-B] back-to-back REGISTERs: admission ONCE",
             fair6_arm13_back_to_back_registers_take_admission_once),
            ("FAIR-6/14 [R1-B] eligibility is consumable once (structure)",
             fair6_arm14_admission_eligibility_is_consumable_once),
        ],
        "-- FAIR-3 — no lease/admission semantic regression --": [
            ("FAIR-3/1  the certified lease suites are green",
             fair3_arm1_certified_suites_are_green),
            ("FAIR-3/2  NO-TOUCH AST proof vs the pinned commit",
             fair3_arm2_no_touch_ast_proof),
            ("FAIR-3/3  _serve_register is unchanged",
             fair3_arm3_serve_register_is_byte_unchanged),
            ("FAIR-3/4  loop_now_age_max still functions",
             fair3_arm4_loop_now_age_still_functions),
            ("FAIR-3/5  the lease origin is still claim_now",
             fair3_arm5_lease_origin_is_still_claim_now),
        ],
        "-- FAIR-4 — no backpressure regression --": [
            ("FAIR-4/0  the full S172-BP battery", fair4_arm0_the_full_bp_battery),
            ("FAIR-4/1  the one-envelope bound survives",
             fair4_arm1_the_one_envelope_bound_survives),
            ("FAIR-4/2  credit disposition: once, on the exact token",
             fair4_arm2_credit_disposition_happens_once_on_the_exact_token),
            ("FAIR-4/3  the capacity timeout is reached no later",
             fair4_arm3_capacity_timeout_is_reached_no_later),
            ("FAIR-4/4  pause does not consume S",
             fair4_arm4_pause_does_not_consume_S),
            ("FAIR-4/5  the admission path does no staging work",
             fair4_arm5_admission_path_does_no_staging),
        ],
    }
    for header, arms in green.items():
        print(f"\n{header}")
        for name, fn in arms:
            _check(name, fn)

    # Every RED arm consumes the pinned pre-repair source, so the anchor gate is
    # unconditional: membership of this dict is the whole rule, and a drifted
    # anchor reports UNAVAILABLE rather than FAIL.
    red_arms = {
        "RXP-1    VACUITY: the gate reds against pinned source":
            rxp1_vacuity_against_pinned_source,
        "FAIR-7   RED: the EOF is lost under a full queue (pinned)":
            fair7_red_eof_is_lost_on_pinned_source,
        "FAIR-1/2 RED: the count bound admits the violation (pinned)":
            fair12_arm7_red_count_bound_admits_the_violation,
        "FAIR-6   RED: REGISTER waits behind the whole FIFO (pinned)":
            fair6_arm12_red_register_waits_behind_the_whole_fifo,
    }
    print("\n-- RED arms (pinned pre-repair source) --")
    for name, fn in red_arms.items():
        if anchor_error is not None:
            _unavailable(name, anchor_error)
            continue
        _check(name, fn)

    print("\n" + "=" * 74)
    ok = sum(1 for _, p, _ in _results if p)
    for name, passed, tb in _results:
        if not passed:
            print(f"\n--- {name} ---\n{tb}")
    print(f"\n{ok}/{len(_results)} checks green")
    if ok == len(_results):
        print("COMPLETION SENTINEL: PASS — S172 attempt-6 remediation, the "
              "ten-gate §11 battery is green (pending Team Beta review).")
    else:
        print("COMPLETION SENTINEL: FAIL — DO NOT COMMIT")
    return 0 if ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
