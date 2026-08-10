#!/usr/bin/env python3
"""
test_s172_admission_liveness.py — S172 §4.3 admission-liveness acceptance harness

Subject: the repair Team Beta ruled a Phase-7 blocker (Ruling 1), described in
docs/FLEET_STATE_REQUIREMENTS_v1.md §4.3 and
docs/CLAUDE_CODE_INSTRUCTIONS_ADMISSION_LIVENESS_REPAIR.md.

    Before the repair, `assign_stripes`, `_dispatch_pending`,
    `process_lease_expiry` AND the stage advance all sat behind one guard,
        if len(eligible) >= expected_workers and stage_idx < len(workflow_stages):
    while `serve_timeout` defaults to None (correctly — a multi-billion-seed scan
    exceeds any wall clock). So a worker loss that crossed the threshold stopped
    lease expiry from being processed at all: the dead worker's stripes stayed
    `claimed` with an expired lease nobody looked at, and the trial neither
    completed nor failed. The Blocker-3 failure matrix was unreachable in exactly
    the situation it exists for.

The repair separates ADMISSION LIVENESS (bounded) from EXECUTION MAINTENANCE
(unbounded). This harness proves both halves, and proves it is not asserting into
a vacuum.

WHY A NEW HARNESS RATHER THAN GATES APPENDED TO PHASE 4
    tests/test_s172_phase4_coordinator.py's tally (63/63) is cited as a
    non-regression figure by the deliverable's own §6 and by several docs. Growing
    it would silently move a number other artifacts pin. This harness is
    registered in that file's gate-22 whitelist with a rationale instead.

HOW A HANG IS TESTED — READ THIS BEFORE TRUSTING A GREEN (VIR-1, VIR-2)
    Testing a hang is the hard part, because the obvious harness ("wait, then give
    up") produces a green for the exact defect it is meant to catch. Three separate
    controls keep that from happening here:

    1. NO WALL CLOCK EXISTS IN THE LIVE ARM. Every live-source run passes
       `serve_timeout=None` — the production default. There is therefore no
       injected timeout that could end the run. The serve thread can only finish by
       reaching a terminal decision of its own (`fail_trial` / `commit_trial`). A
       run that ends is, structurally, a run that decided.
    2. THE HARNESS BUDGET IS A FAILURE, NEVER A PASS. `_drive` joins with a budget
       ~10-20x the behaviour under test. Still alive at the budget = `still_hung`,
       which every live gate asserts False. A gate cannot pass by timing out; it
       can only fail that way (VIR-3: PASS | FAIL | UNAVAILABLE | INCOMPLETE).
    3. THE TERMINATION IS CLASSIFIED, NOT ASSUMED. `_RunOutcome.ended_by` is
       derived from the abort reason the code under test produced:
         own-decision            — a terminal reason this repair is responsible for
         harness-injected-clock  — the reason is "serve_trial timeout" (mutant arm
                                   only; impossible in the live arm, see 1)
         still-hung              — no terminal decision inside the budget
       Live gates additionally assert the reason TEXT (run id, stage, expected,
       eligible / "constant-phase failure" / ...), so "it stopped" is never
       confused with "it stopped for the right reason".

    G-MUTANT is what makes the rest mean anything, and Beta named it explicitly.
    Every scenario is run a SECOND time against a MUTANT of the live module in
    which the outer threshold guard is restored — one line, located by AST (not by
    text match, per VERIFICATION_INTEGRITY_STANDARD; 2389b61 reverted a fix by
    whole-block replacement and a text anchor would have gone green). The five
    hang scenarios must all go red under the mutant, and the healthy control must
    stay green, or this harness is not measuring what it claims.
    The mutant arm — and ONLY the mutant arm — injects a finite `serve_timeout`,
    purely as a thread reaper so a deliberately-hung run cannot leak a thread or
    outlive its tempdir. It is set well above the observation window, and a mutant
    that terminates by it is recorded as `harness-injected-clock`, i.e. RED.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 tests/test_s172_admission_liveness.py
Exit 0 = all green. Exit 1 = a check failed (DO NOT COMMIT).
"""
import ast
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import miner.range_miner_coordinator as COORD           # noqa: E402
from miner.range_miner_coordinator import (              # noqa: E402
    DEFAULT_WORKER_ADMISSION_TIMEOUT,
    ST_DONE,
    TC_COMPUTE_LEASE_EXPIRY,
    TC_SERVE_TIMEOUT,
    TC_STRIPE_ERROR,
    TC_WORKER_ADMISSION_TIMEOUT,
)

# Reuse the Phase-4 harness's REAL-wire worker and Phase-5 sink stubs rather than
# reimplementing them: gate 37 already certifies that `_FakeWorker` speaks the
# genuine MinerFramedSocket protocol, and a second copy would be free to drift.
from test_s172_phase4_coordinator import _FakeWorker, _StubSink   # noqa: E402

_COORD_PATH = os.path.abspath(COORD.__file__)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"

_results: List[tuple] = []
# Arm outcomes, keyed "<gate>/<arm>", consumed by the G-MUTANT summary gate.
_arm_verdicts: Dict[str, str] = {}


def _check(name, fn):
    try:
        detail = fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}" + (f" — {detail}" if detail else ""))
    except Exception as e:                                   # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


# ===========================================================================
# The mutant: restore the outer threshold guard (AST-located, single line)
# ===========================================================================
# The repaired guard reads exactly
#     if stage_idx < len(workflow_stages):
# and the pre-repair guard read
#     if len(eligible) >= expected_workers and stage_idx < len(workflow_stages):
# Restoring that one line reverts the WHOLE repair semantically: below the
# threshold the block is skipped entirely, so the admission wait is never
# evaluated (hang before assignment) and dispatch / lease expiry / stage advance
# never run (hang after assignment). That is why one line is a sufficient mutant.
_MUTANT_GUARD = ("if len(eligible) >= expected_workers "
                 "and stage_idx < len(workflow_stages):")
_REPAIRED_GUARD_AST = ast.dump(
    ast.parse("stage_idx < len(workflow_stages)", mode="eval").body)


def _locate_guard_line(src: str) -> int:
    """1-based line number of the repaired staged-assignment guard inside
    serve_trial, located structurally. Raises if it is not uniquely identifiable —
    an ambiguous or missing anchor is INCOMPLETE, never a silent pass."""
    tree = ast.parse(src)
    serve = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == "serve_trial"]
    if len(serve) != 1:
        raise AssertionError(
            f"expected exactly one serve_trial definition, found {len(serve)}")
    hits = [n for n in ast.walk(serve[0])
            if isinstance(n, ast.If) and ast.dump(n.test) == _REPAIRED_GUARD_AST]
    if len(hits) != 1:
        raise AssertionError(
            "the repaired staged-assignment guard `if stage_idx < "
            f"len(workflow_stages):` is not uniquely locatable in serve_trial "
            f"(found {len(hits)}). The mutant cannot be built, so the gates it "
            "validates are UNPROVEN — this is INCOMPLETE, not a pass.")
    return hits[0].lineno


def _build_mutant_module() -> types.ModuleType:
    """Compile a copy of the LIVE coordinator source with the outer threshold
    guard restored, and exec it as its own module."""
    with open(_COORD_PATH, "r", encoding="utf-8") as fh:
        src = fh.read()
    lineno = _locate_guard_line(src)
    lines = src.splitlines(keepends=True)
    original = lines[lineno - 1]
    stripped = original.strip()
    if stripped != "if stage_idx < len(workflow_stages):":
        raise AssertionError(
            f"guard line {lineno} is {stripped!r}, not the single-line form the "
            "mutant rewrites. Refusing to guess (INCOMPLETE).")
    indent = original[:len(original) - len(original.lstrip())]
    lines[lineno - 1] = f"{indent}{_MUTANT_GUARD}\n"
    mutant_src = "".join(lines)
    if mutant_src == src:
        raise AssertionError("mutant source is identical to live source")

    name = "range_miner_coordinator__threshold_guard_mutant"
    mod = types.ModuleType(name)
    mod.__file__ = _COORD_PATH + " [MUTANT: outer threshold guard restored]"
    mod.__package__ = ""
    sys.modules[name] = mod
    exec(compile(mutant_src, mod.__file__, "exec"), mod.__dict__)
    return mod


# ===========================================================================
# Workers: real framed sockets (subclassing the Phase-4 gate-37 worker)
# ===========================================================================
class _LiveWorker(_FakeWorker):
    """Adds the lifecycle behaviours a liveness test needs, on top of the real
    wire behaviour gate 37 already certifies.

      complete            (inherited) — one inline sub-stripe + StripeComplete
      silent                          — take the assignment, never answer, so the
                                        COMPUTE LEASE expires (the §4.2/§4.3 path)
      assign_then_die                 — take the assignment, then drop the socket:
                                        a mid-run worker loss holding a stripe
      gated_complete                  — take the assignment, wait for the harness
                                        to permit it, then complete and drop.
                                        Used to order "work finished" strictly
                                        AFTER "pool dropped below threshold".
      slow_complete                   — complete after `work_delay` seconds, to
                                        prove execution is not bounded by the
                                        admission window.
    """

    def __init__(self, host, port, hostname, gpu_id, behavior,
                 work_delay: float = 0.0):
        super().__init__(host, port, hostname, gpu_id, behavior)
        self.work_delay = work_delay
        self.assigned = threading.Event()
        self.completed = threading.Event()
        self.permit = threading.Event()

    def _respond(self, assign):
        self.assigned.set()
        if self.behavior == "silent":
            return
        if self.behavior == "assign_then_die":
            self._hard_close()
            return
        if self.behavior == "gated_complete":
            self.permit.wait(timeout=60.0)
            super()._respond(assign)
            self.completed.set()
            # The disconnect the ruling's fourth row is about: the work is done,
            # THEN the worker goes away.
            time.sleep(0.3)
            self._hard_close()
            return
        if self.work_delay:
            time.sleep(self.work_delay)
        super()._respond(assign)
        self.completed.set()

    def _loop(self):
        """Whole-frame blocking reads, overriding the inherited loop.

        FIXTURE DEFECT THIS AVOIDS (found while stabilising this harness; it is a
        TEST-SIDE defect, not a production one): `_FakeWorker._loop` puts a 0.5s
        timeout on the socket and `continue`s on `socket.timeout`. But
        `MinerFramedSocket._recvall` (range_miner_worker.py:1128) loops on
        `sock.recv`, so a frame whose body is split across TCP segments can time
        out MID-FRAME — the bytes already consumed are discarded with the
        abandoned `chunks` buffer, and the next `recv_msg` reads a "header" from
        the middle of that body. The stream desynchronises, the worker dies
        silently on the bogus length, and its stripe then sits `claimed` until the
        compute lease expires. That is indistinguishable from the §4.3 hang this
        harness is measuring, so the fixture must not be able to produce it.
        The real coordinator reader (`_conn_reader_loop`) does exactly what this
        does: blocking full-frame reads, unblocked by shutdown() at teardown.
        """
        try:
            self.fs.sock.settimeout(None)
            while not self._stop.is_set():
                try:
                    msg = self.fs.recv_msg()
                except (ConnectionError, ValueError, OSError):
                    break
                if msg.message_type == "stripe_assign":
                    self.assigns_received.append((msg.stripe_id, msg.attempt))
                    self._respond(msg)
                elif msg.message_type == "shutdown":
                    break
        except Exception:                                    # noqa: BLE001
            self.err = traceback.format_exc()

    def stop(self):
        # shutdown() (not just close()) is what reliably wakes a thread blocked in
        # recv, now that the read is blocking.
        self._hard_close()

    def _hard_close(self):
        self._stop.set()
        try:
            self.fs.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.fs.close()
        except Exception:                                    # noqa: BLE001
            pass


# ===========================================================================
# The driver
# ===========================================================================
@dataclass
class _RunOutcome:
    still_hung: bool
    elapsed: float
    state: str
    committed: bool
    abort_reasons: List[str] = field(default_factory=list)
    # [F1/F2 R2] The WHOLE abort event, not just its prose. F2's canonicalization
    # made `reason` derive from the TerminalRecord, so the caller's short prose
    # ("serve_trial timeout", "…: constant-phase failure") is no longer on the
    # wire — the machine-readable answer is `terminal_class`, which is what this
    # harness should have been reading all along. See `ended_by` below.
    abort_events: List[Dict[str, Any]] = field(default_factory=list)
    run_id: Optional[str] = None
    stripes: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    serve_timeout: Optional[float] = None
    workers: List[_LiveWorker] = field(default_factory=list)

    @property
    def abort_classes(self) -> List[str]:
        return [str(e.get("terminal_class", "")) for e in self.abort_events]

    @property
    def ended_by(self) -> str:
        if self.still_hung:
            return "still-hung"
        # [F1/F2 R2] CLASSIFY ON THE STRUCTURED CLASS, NOT ON PROSE.
        #
        # This branch used to match the literal "serve_trial timeout" — the
        # CALLER's prose. F2's reason canonicalization replaced the event's
        # `reason` with the TerminalRecord's own text ("serve_trial exceeded its
        # configured serve_timeout of 20.0s (elapsed 20.1s)"), which does not
        # contain that literal, so the branch went permanently false and the
        # harness would have silently lost its ability to distinguish a
        # harness-injected clock from the code's own decision. `TC_SERVE_TIMEOUT`
        # is the field that exists to answer this and cannot drift with wording.
        if TC_SERVE_TIMEOUT in self.abort_classes:
            return "harness-injected-clock"
        if self.error:
            return "raised"
        return "own-decision"

    def summary(self) -> str:
        return (f"ended_by={self.ended_by} state={self.state} "
                f"elapsed={self.elapsed:.1f}s")


def _drive(mod, *, worker_specs, expected_workers, admission_timeout,
           lease_timeout, serve_timeout, budget, family="java_lcg", phase=1,
           both_modes=False, total_seeds=30, orchestrate=None,
           reap_extra=25.0) -> _RunOutcome:
    """Drive the REAL `serve_trial` of `mod` (VIR-1 execution proof: the gates
    never reimplement its logic) against loopback workers on real framed sockets.

    `budget` is the OBSERVATION WINDOW. Exceeding it is `still_hung` — a FAILURE
    signal for the live arm, never a pass. `serve_timeout` is None for every live
    run, so nothing but the code's own terminal decision can end it.
    """
    tmp = tempfile.mkdtemp(prefix="s172_admission_")
    ds = os.path.join(tmp, "dataset.json")
    with open(ds, "w") as fh:
        fh.write('[{"draw":1},{"draw":2},{"draw":3}]')
    sink = _StubSink()

    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(16)
    port = lsock.getsockname()[1]

    holder: Dict[str, Any] = {}

    def run():
        try:
            holder["result"] = mod.run_trial_miner(
                "run-admission", None, 5, "java_lcg", [1, 2, 3], total_seeds,
                0.25, 0.25, both_modes, ds,
                worker_pool_size=expected_workers,
                staging_dir=os.path.join(tmp, "stg"),
                phase5_sink=sink, listen_sock=lsock,
                skip_min=0, skip_max=0, offset=0, window_size=3,
                compute_lease_timeout=lease_timeout,
                # PRODUCTION DEFAULT in the live arm: None. Nothing here can end
                # the run except serve_trial's own terminal decision.
                serve_timeout=serve_timeout,
                worker_admission_timeout=admission_timeout,
                # family=None means "no override": the trial runs the REAL
                # resolved workflow (workflow_stages_for), i.e. more than one
                # stage, which is the only way to exercise a genuine stage
                # boundary. A concrete family collapses the workflow to one stage,
                # exactly as the Phase-4 serve-path gate does.
                **({} if family is None else
                   {"family_name": family, "workflow_phase": phase}))
        except Exception:                                    # noqa: BLE001
            holder["err"] = traceback.format_exc()

    workers: List[_LiveWorker] = []
    stop_orch = threading.Event()
    orch_thread = None
    t0 = time.time()
    t = threading.Thread(target=run, name="serve-under-test", daemon=True)
    t.start()
    try:
        for (hostname, gpu_id, behavior, delay) in worker_specs:
            w = _LiveWorker("127.0.0.1", port, hostname, gpu_id, behavior, delay)
            w.connect_register()
            w.start_loop()
            workers.append(w)
        if orchestrate is not None:
            orch_thread = threading.Thread(
                target=orchestrate, args=(workers, port, stop_orch),
                name="orchestrator", daemon=True)
            orch_thread.start()

        t.join(timeout=budget)
        still_hung = t.is_alive()
        if still_hung and serve_timeout is not None:
            # Mutant arm only: let the injected reaper collect the hung thread so
            # nothing leaks and the tempdir can be removed. The verdict was
            # already recorded above.
            t.join(timeout=serve_timeout + reap_extra)
        elapsed = time.time() - t0
    finally:
        stop_orch.set()
        if orch_thread is not None:
            orch_thread.join(timeout=5.0)
        for w in workers:
            try:
                w.stop()
            except Exception:                                # noqa: BLE001
                pass
        try:
            lsock.close()
        except Exception:                                    # noqa: BLE001
            pass

    if still_hung:
        # VIR-1: a hang must be DIAGNOSABLE, not just reported. Dump the durable
        # ledger so a red says WHERE it stopped (which stripes, in which state,
        # claimed by whom) instead of only that it stopped.
        try:
            import sqlite3
            db = os.path.join(tmp, "stg", "miner_ledger.db")
            con = sqlite3.connect(db)
            print("      [hang diagnostic] trials:",
                  con.execute("select run_id,state from trials").fetchall())
            print("      [hang diagnostic] stripes:",
                  con.execute("select stripe_id,state,claimed_by,current_attempt "
                              "from stripes").fetchall())
            cols = [r[1] for r in con.execute("PRAGMA table_info(shards)")]
            print("      [hang diagnostic] shard cols:", cols)
            print("      [hang diagnostic] shards:",
                  con.execute("select * from shards").fetchall())
            print("      [hang diagnostic] workers seen:",
                  [(w.worker_id, w.assigns_received, w.err is not None)
                   for w in workers])
            con.close()
        except Exception as e:                               # noqa: BLE001
            print(f"      [hang diagnostic] unavailable: {e}")

    result = holder.get("result") or {}
    outcome = _RunOutcome(
        still_hung=still_hung,
        elapsed=elapsed,
        state=result.get("state", "unknown"),
        committed=bool(result.get("committed")),
        abort_reasons=[str(e.get("reason", "")) for e in sink.aborts],
        abort_events=[dict(e) for e in sink.aborts],
        run_id=result.get("run_id"),
        stripes=result.get("stripes", {}),
        error=holder.get("err"),
        serve_timeout=serve_timeout,
        workers=workers,
    )
    if not t.is_alive():
        shutil.rmtree(tmp, ignore_errors=True)
    return outcome


# ===========================================================================
# Shared assertions
# ===========================================================================
def _assert_not_vacuous(o: _RunOutcome):
    """The live arm's anti-vacuity wall (VIR-2): a pass must be a decision."""
    assert not o.still_hung, (
        f"THE CODE UNDER TEST HUNG: no terminal decision within the "
        f"{o.elapsed:.1f}s observation window. This is the §4.3 defect, and it is "
        f"a FAILURE — the harness budget is never a pass.")
    assert o.serve_timeout is None, (
        "live-arm invariant violated: a finite serve_timeout was injected, so "
        "termination could not be attributed to the repair")
    assert o.ended_by == "own-decision", (
        f"termination was not the code's own decision: {o.summary()}")


def _run_id_diagnostics(o: _RunOutcome, stage: int, expected: int):
    """Beta's message requirement: run ID, stage, expected and eligible counts.
    The eligible count is READ OUT of the message and checked for consistency
    (0 <= eligible < expected) rather than pinned to a literal — under churn the
    pool legitimately holds a different number at the instant the window closes,
    and pinning it would make the gate assert the fixture instead of the
    contract."""
    assert o.abort_reasons, "no abort event reached the Phase-5 sink"
    ev = o.abort_events[0]
    reason = o.abort_reasons[0]
    # [F1/F2 R2] Beta's four required facts are UNCHANGED; two of them now live in
    # structured event fields rather than in the prose, because F2 canonicalized
    # `reason` onto the TerminalRecord. WHAT is asserted is identical — the
    # terminal cause and the run identity — and reading them from
    # `terminal_class` / `run_id` is stronger than a substring of a message.
    assert ev.get("terminal_class") == TC_WORKER_ADMISSION_TIMEOUT, (
        f"the terminal cause is not the admission window: {ev}")
    assert o.run_id and ev.get("run_id") == o.run_id, (o.run_id, ev)
    assert f"stage {stage}" in reason, reason
    assert f"expected {expected}" in reason, reason
    m = re.search(r"(\d+) admitted", reason)
    assert m, f"the message carries no eligible count: {reason}"
    eligible = int(m.group(1))
    assert 0 <= eligible < expected, (
        f"reported eligible={eligible} is not a shortage against "
        f"expected={expected}: {reason}")
    return reason


# ===========================================================================
# Scenarios — each runs against an arbitrary module (live source or mutant)
# ===========================================================================
LIVE_BUDGET = 45.0          # >= 10x every behaviour under test
MUT_OBSERVE = 12.0          # mutant observation window
MUT_REAPER = 20.0           # mutant-only serve_timeout (strictly > MUT_OBSERVE)


def _arm(mod, live: bool, **kw) -> _RunOutcome:
    if live:
        return _drive(mod, serve_timeout=None, budget=LIVE_BUDGET, **kw)
    return _drive(mod, serve_timeout=MUT_REAPER, budget=MUT_OBSERVE, **kw)


def scn_admission_timeout(mod, live: bool) -> _RunOutcome:
    """Fewer daemons than expected_workers ever register, BEFORE assignment."""
    return _arm(mod, live,
                worker_specs=[("hostA", 0, "complete", 0.0)],
                expected_workers=2, admission_timeout=2.0, lease_timeout=60.0)


def _churn(workers, port, stop):
    """Connect and drop a second worker continuously, so the pool oscillates
    1 <-> 2 but never reaches 3. Runs until the driver stops it: an
    implementation that reset the window on churn would therefore NEVER fire,
    which is precisely what this must be able to detect."""
    i = 0
    while not stop.is_set():
        i += 1
        w = None
        try:
            w = _LiveWorker("127.0.0.1", port, "hostChurn", i, "complete", 0.0)
            w.connect_register()
            w.start_loop()
        except Exception:                                    # noqa: BLE001
            return
        if stop.wait(0.6):
            try:
                w.stop()
            except Exception:                                # noqa: BLE001
                pass
            return
        try:
            w._hard_close()
        except Exception:                                    # noqa: BLE001
            pass
        stop.wait(0.4)


def scn_no_reset_on_churn(mod, live: bool) -> _RunOutcome:
    """Same shortage, but with continuous connect/disconnect churn under the
    threshold. The window must expire measured from the stage boundary."""
    return _arm(mod, live,
                worker_specs=[("hostA", 0, "complete", 0.0)],
                expected_workers=3, admission_timeout=5.0, lease_timeout=60.0,
                orchestrate=_churn)


def scn_cross_constant(mod, live: bool) -> _RunOutcome:
    """Constant phase (TFM's java_lcg path). The stripe-holding worker dies after
    assignment, dropping the pool below expected_workers. Its compute lease must
    still be processed -> Blocker-3 row 'constant phase' -> immediate trial
    failure."""
    return _arm(mod, live,
                worker_specs=[("hostA", 0, "assign_then_die", 0.0),
                              ("hostB", 0, "complete", 0.0)],
                expected_workers=2, admission_timeout=30.0, lease_timeout=2.0,
                family="java_lcg", phase=1)


def scn_cross_hybrid(mod, live: bool) -> _RunOutcome:
    """Hybrid phase. Same loss, below the ORIGINAL threshold (2 eligible of an
    expected 3) -> the existing one-reassignment policy must execute against a
    DIFFERENT worker, and the trial must then commit."""
    return _arm(mod, live,
                worker_specs=[("hostA", 0, "assign_then_die", 0.0),
                              ("hostB", 0, "complete", 0.0),
                              ("hostC", 0, "complete", 0.0)],
                expected_workers=3, admission_timeout=30.0, lease_timeout=2.0,
                family="java_lcg_hybrid", phase=3)


def _drop_idle_then_permit(workers, port, stop):
    """Order the fourth row of the ruling's table deterministically:
      1. the working worker has the assignment;
      2. the IDLE worker is dropped -> eligible (1) < expected (2);
      3. only THEN is the work allowed to finish.
    So the stage advance and commit necessarily happen below the threshold."""
    worker, idle = workers[0], workers[1]
    if not worker.assigned.wait(timeout=20.0):
        return
    idle._hard_close()
    # let the serve loop drain the EOF and evict it from the eligible pool
    if stop.wait(1.5):
        return
    worker.permit.set()


def scn_final_stage_below_threshold(mod, live: bool) -> _RunOutcome:
    """Final-stage work completes while the pool is below threshold, and the
    worker disconnects afterwards. The trial must still commit."""
    return _arm(mod, live,
                worker_specs=[("hostA", 0, "gated_complete", 0.0),
                              ("hostB", 0, "complete", 0.0)],
                expected_workers=2, admission_timeout=30.0, lease_timeout=60.0,
                orchestrate=_drop_idle_then_permit)


HEALTHY_ADMISSION = 1.5
HEALTHY_WORK_DELAY = 3.0


def scn_long_healthy(mod, live: bool) -> _RunOutcome:
    """CLEAN CONTROL (VIR-2). A healthy full pool whose WORK takes several times
    longer than the admission window, with serve_timeout=None. Nothing may fire,
    and no ceiling may be imposed on the run's duration.

    SCOPE NOTE (VIR-5 — unobservable is not clean). This control drives ONE stage.
    A multi-stage variant was built first and is NOT used, because the real
    two-stage workflow intermittently stalls in the Phase-5 STAGING ADMISSION path
    — the second stage's shard sits at staging_status='pending' forever — and that
    stall reproduces at byte-HEAD, without this repair, and is therefore neither
    caused by nor in scope for it (see the session report). A control that flaked
    for an unrelated reason would be worse than no control: it would put a red next
    to this repair for someone else's defect. The stage-boundary re-arm rule this
    would have exercised is instead proven structurally by G-REARM-STRUCTURE and
    behaviourally (its negative half) by G-ADMISSION-NO-RESET-ON-CHURN."""
    specs = [("hostA", 0, "slow_complete", HEALTHY_WORK_DELAY),
             ("hostB", 0, "slow_complete", HEALTHY_WORK_DELAY)]
    if live:
        return _drive(mod, serve_timeout=None, budget=LIVE_BUDGET,
                      worker_specs=specs, expected_workers=2,
                      admission_timeout=HEALTHY_ADMISSION, lease_timeout=120.0)
    # The mutant arm of the control is EXPECTED GREEN (a full pool never trips the
    # restored guard), so it gets a generous reaper purely as a leak guard.
    return _drive(mod, serve_timeout=90.0, budget=45.0,
                  worker_specs=specs, expected_workers=2,
                  admission_timeout=HEALTHY_ADMISSION, lease_timeout=120.0)


# ===========================================================================
# LIVE-ARM GATES
# ===========================================================================
def g_admission_timeout():
    o = scn_admission_timeout(COORD, live=True)
    _arm_verdicts["G-ADMISSION-TIMEOUT/live"] = o.ended_by
    _assert_not_vacuous(o)
    assert o.state == "aborted" and not o.committed, o.summary()
    reason = _run_id_diagnostics(o, stage=0, expected=2)
    # It fired because the window closed, not early and not late.
    assert 2.0 <= o.elapsed < 2.0 + 12.0, f"fired at {o.elapsed:.1f}s"
    assert "serve_trial timeout" not in reason, reason
    return f"terminal at {o.elapsed:.1f}s (budget {LIVE_BUDGET}s, no wall clock)"


def g_no_reset_on_churn():
    o = scn_no_reset_on_churn(COORD, live=True)
    _arm_verdicts["G-ADMISSION-NO-RESET-ON-CHURN/live"] = o.ended_by
    _assert_not_vacuous(o)
    assert o.state == "aborted", o.summary()
    _run_id_diagnostics(o, stage=0, expected=3)
    # Churn ran continuously at ~1s intervals for the whole window. An
    # implementation that re-armed on connect/disconnect could not have fired at
    # all, so an on-time expiry IS the evidence the window is stage-scoped.
    assert 5.0 <= o.elapsed < 5.0 + 12.0, (
        f"expired at {o.elapsed:.1f}s — the window is not anchored to the stage "
        f"boundary")
    return f"expired at {o.elapsed:.1f}s under continuous sub-threshold churn"


def g_cross_constant():
    o = scn_cross_constant(COORD, live=True)
    _arm_verdicts["G-CROSS-CONSTANT/live"] = o.ended_by
    _assert_not_vacuous(o)
    assert o.state == "aborted" and not o.committed, o.summary()
    joined = " | ".join(o.abort_reasons)
    # [F1/F2 R2] Same two facts, read off the canonical surfaces. The constant
    # row is identified by its terminal CLASS (a lease expiry or a reported
    # stripe error routed through the phase policy) plus the phase policy the
    # record states in full; the negative check becomes "the admission window is
    # not the terminal cause", asserted on the class rather than on wording.
    assert set(o.abort_classes) <= {TC_COMPUTE_LEASE_EXPIRY, TC_STRIPE_ERROR}, (
        f"unexpected terminal class for the constant row: {o.abort_classes}")
    assert "CONSTANT-MODE" in joined, joined
    assert TC_WORKER_ADMISSION_TIMEOUT not in o.abort_classes, (
        "the failure came from the admission window, not from lease expiry "
        f"reaching the matrix: {o.abort_classes} :: {joined}")
    return f"Blocker-3 constant row reached below threshold ({o.elapsed:.1f}s)"


def g_cross_hybrid():
    o = scn_cross_hybrid(COORD, live=True)
    _arm_verdicts["G-CROSS-HYBRID/live"] = o.ended_by
    _assert_not_vacuous(o)
    assert o.state == "committed" and o.committed, (o.summary(), o.abort_reasons)
    assert o.stripes, "no stripes reported"
    degraded = [s for s in o.stripes.values() if s["phase_degraded"]]
    assert degraded, f"no stripe was reassigned: {o.stripes}"
    for s in degraded:
        assert s["current_attempt"] == 1, s
        assert s["claimed_by"] != "hostA:gpu0", s
        assert s["state"] == ST_DONE, s
    return (f"one reassignment to {degraded[0]['claimed_by']} with 2 of an "
            f"expected 3 eligible")


def g_final_stage():
    o = scn_final_stage_below_threshold(COORD, live=True)
    _arm_verdicts["G-FINAL-STAGE/live"] = o.ended_by
    _assert_not_vacuous(o)
    assert o.state == "committed" and o.committed, (o.summary(), o.abort_reasons)
    assert not o.abort_reasons, o.abort_reasons
    assert all(s["state"] == ST_DONE for s in o.stripes.values()), o.stripes
    return "committed with the pool below threshold and the worker gone"


def g_long_healthy():
    o = scn_long_healthy(COORD, live=True)
    _arm_verdicts["G-LONG-HEALTHY/live"] = o.ended_by
    _assert_not_vacuous(o)
    assert o.state == "committed" and o.committed, (o.summary(), o.abort_reasons)
    assert not o.abort_reasons, (
        f"a healthy full-pool run must not abort: {o.abort_reasons}")
    # The run outlived its admission window several times over — the window does
    # not become a duration ceiling once a stage is assigned.
    assert o.elapsed > 2 * HEALTHY_ADMISSION, (
        f"control is too short ({o.elapsed:.1f}s) to demonstrate anything about "
        f"the {HEALTHY_ADMISSION}s admission window")
    assert o.elapsed > HEALTHY_WORK_DELAY, o.elapsed
    assert o.stripes and all(s["state"] == ST_DONE for s in o.stripes.values()), \
        o.stripes
    return (f"{o.elapsed:.1f}s run vs a {HEALTHY_ADMISSION}s admission window, "
            f"serve_timeout=None, no abort")


# ===========================================================================
# CONFIGURATION AND FORBIDDEN-CHANGE GATES
# ===========================================================================
def g_config_fail_closed():
    """The admission window may not be disabled back into the defect. None, 0,
    negative and inf are all refused before any worker is admitted."""
    assert DEFAULT_WORKER_ADMISSION_TIMEOUT == 180.0, DEFAULT_WORKER_ADMISSION_TIMEOUT
    tmp = tempfile.mkdtemp(prefix="s172_admission_cfg_")
    try:
        coord = COORD.build_coordinator(staging_dir=tmp)
        for bad in (None, 0, -1.0, float("inf")):
            try:
                # The context is deliberately minimal: the refusal must land
                # BEFORE the dataset digest, the trial context or the listening
                # socket, so a context this thin never gets that far.
                coord.serve_trial({"run_id": "r", "worker_admission_timeout": bad})
            except ValueError as e:
                assert "worker_admission_timeout" in str(e), e
            else:
                raise AssertionError(f"{bad!r} was accepted as an admission window")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return "None/0/negative/inf refused; default 180.0s (PWC readiness window)"


def g_rearm_structure():
    """"Reset ONLY at a genuine new-stage boundary" — proven structurally against
    LIVE source, because it is a rule about what CANNOT happen and no finite set of
    behavioural runs can establish that. Two properties, both read off the AST:

      1. the window is re-armed under exactly one condition, and that condition
         compares the window's owning stage to the current stage — not a worker
         count, not a registration event, not a clock;
      2. `admission_started_at` is written in exactly that one place, so nothing on
         the connect / register / drop / quarantine paths can extend it.

    G-ADMISSION-NO-RESET-ON-CHURN is the behavioural half (churn does not extend
    the window); this is the half that says why."""
    src = open(_COORD_PATH, encoding="utf-8").read()
    serve = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.FunctionDef) and n.name == "serve_trial"]
    assert len(serve) == 1
    writes = []
    for n in ast.walk(serve[0]):
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id == "admission_started_at":
                writes.append(n)
    # One initialisation to None plus exactly one re-arm.
    assert len(writes) == 2, (
        f"admission_started_at is written in {len(writes)} places; every extra "
        f"write is a place the window could be reset by something that is not a "
        f"stage boundary: {[ast.get_source_segment(src, w) for w in writes]}")
    init, rearm = writes
    assert ast.get_source_segment(src, init).endswith("None"), \
        ast.get_source_segment(src, init)
    # The re-arm must sit inside `if admission_stage_idx != stage_idx:`.
    guards = [n for n in ast.walk(serve[0])
              if isinstance(n, ast.If)
              and any(w is rearm for b in n.body for w in ast.walk(b))
              and ast.dump(n.test) == ast.dump(ast.parse(
                  "admission_stage_idx != stage_idx", mode="eval").body)]
    assert guards, (
        "the re-arm is not guarded by `admission_stage_idx != stage_idx`, so it is "
        "not anchored to a stage boundary")
    return ("admission_started_at written exactly twice (init + one stage-keyed "
            "re-arm); no churn/registration path can touch it")


def _fn_src(src: str, qualname: str) -> str:
    tree = ast.parse(src)
    hits = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == qualname]
    assert len(hits) == 1, f"{qualname}: {len(hits)} definitions"
    return ast.get_source_segment(src, hits[0])


def g_forbidden_changes_absent():
    """Beta's four explicit prohibitions, checked against HEAD rather than
    asserted in prose (§7 asks for exactly this confirmation)."""
    head = subprocess.run(
        ["git", "show", "HEAD:miner/range_miner_coordinator.py"],
        cwd=_ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace")
    assert head.returncode == 0, head.stderr
    old, new = head.stdout, open(_COORD_PATH, encoding="utf-8").read()

    # 1. The Blocker-3 failure matrix is REACHABLE AND UNREWRITTEN.
    #
    #    BYTE IDENTITY RETAINED for three of the four. Nothing in F1/F2 touches
    #    the entry point, the alternate-selection predicate or the expiry sweep,
    #    so the strongest available check still applies to them unchanged.
    for fn in ("handle_stripe_failure", "_pick_other_worker",
               "process_lease_expiry"):
        assert _fn_src(old, fn) == _fn_src(new, fn), (
            f"{fn} changed — the Blocker-3 matrix must be untouched")

    #    SUPERSEDED for `_handle_stripe_failure_locked` ONLY (Team Beta, F1/F2 R1
    #    §C — granted on Alpha's §3 request, and granted with an ORDER
    #    CONSTRAINT that was honoured: Blockers A and B were fixed BEFORE this
    #    baseline moved, so what is certified below is the corrected function,
    #    never the defective one).
    #
    #    WHAT CHANGED AND WHY THE OLD FORM NO LONGER FITS. "Failure matrix
    #    unchanged" means TERMINAL DECISION SEMANTICS unchanged — not that the
    #    function may never change a byte. F1 changed how a hybrid first failure
    #    is PLACED (the requeue no longer claims; the scheduler hands the stripe
    #    to the first idle alternate with a fresh lease), and R1 Blocker A
    #    changed the retry's selector from a LIKE-scoped prefix to stripe
    #    identity. Both live inside this function; neither is a terminal
    #    decision. A byte comparison cannot express that distinction, so it is
    #    replaced by the invariant that can.
    #
    #    THE SUPERSEDING INVARIANT — the four terminal decisions, IN ORDER:
    #        non-retryable                                -> non_retryable
    #        constant phase                               -> constant_phase
    #        hybrid first failure + no alternate eligible -> no_alternate_worker
    #        hybrid second failure                        -> hybrid_second_failure
    #    Read off the LIVE source by AST, in source order, so a reordering (which
    #    WOULD be a semantic change: the non-retryable test must precede the
    #    phase test, and the no-alternate test must precede the second-failure
    #    row) reds this gate.
    EXPECTED_TERMINAL_ORDER = ["non_retryable", "constant_phase",
                               "no_alternate_worker", "hybrid_second_failure"]
    fn_ast = [n for n in ast.walk(ast.parse(new))
              if isinstance(n, ast.FunctionDef)
              and n.name == "_handle_stripe_failure_locked"]
    assert len(fn_ast) == 1, fn_ast
    terminal_reasons, nonterminal_actions = [], []
    for node in ast.walk(fn_ast[0]):
        if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)):
            continue
        d = {}
        for k, v in zip(node.value.keys, node.value.values):
            if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                d[k.value] = v.value
        if d.get("action") == "fail_trial":
            terminal_reasons.append((node.lineno, d.get("reason")))
        elif d.get("action") in ("reassigned", "requeued", "noop"):
            nonterminal_actions.append(d["action"])
    ordered = [r for _ln, r in sorted(terminal_reasons)]
    assert ordered == EXPECTED_TERMINAL_ORDER, (
        f"the four terminal decisions changed or were reordered: {ordered} != "
        f"{EXPECTED_TERMINAL_ORDER}")
    #    ...and the hybrid first-failure NONTERMINAL branch still has BOTH of its
    #    ratified outcomes available — immediate reassignment when an eligible
    #    alternate is truly idle, OR pending/requeued when alternate capacity is
    #    temporarily busy. (Behaviourally certified in
    #    tests/test_s172_staging_backpressure.py::gate_matrix_diff_behavioural and
    #    tests/test_s172_f1_f2_active_lease.py G-F1-HYBRID-MATRIX /
    #    G-F1-EXACT-STRIPE-COLLISION; asserted structurally here so this gate
    #    cannot go green on a function that lost one of them.)
    assert {"reassigned", "requeued"} <= set(nonterminal_actions), (
        f"the hybrid nonterminal branch lost an outcome: {nonterminal_actions}")
    #    The selector the retry path uses is IDENTITY, never a prefix (R1
    #    Blocker A): `run__st0_s1%` also matches `run__st0_s10 … s19`.
    body_src = ast.unparse(fn_ast[0])
    assert "exact_stripe_id=stripe_id" in body_src, (
        "the hybrid immediate placement no longer selects by stripe identity")
    assert "stripe_prefix=stripe_id" not in body_src, (
        "prefix-as-exact selection was reintroduced (R1 Blocker A)")

    # 2. expected_workers is never reduced dynamically: it is bound exactly once,
    #    in serve_trial's preamble, from the requested worker_pool_size.
    #
    #    AMENDED for the ADMISSION-BINDING repair (Beta: AUTHORIZED and REQUIRED,
    #    docs/CLAUDE_CODE_INSTRUCTIONS_ADMISSION_BINDING_REPAIR.md §1). The
    #    binding used to be compared BYTE-FOR-BYTE against HEAD, which is the
    #    right check while the value's source is frozen and the wrong one once
    #    Beta rules that source must change: the set recorded an admission count
    #    while this line derived expected_workers independently from
    #    context["worker_pool_size"] — two frozen run facts about one run, free
    #    to disagree, so a local two-GPU set still waited for eight.
    #
    #    What this gate protects is UNCHANGED and is what §4.3 actually needs:
    #    ONE binding, in the preamble (never inside a loop or a branch, i.e.
    #    never reduced dynamically as workers come and go), derived from the
    #    requested pool size. What it no longer asserts is that the requested
    #    value is the FINAL authority — that is precisely what the repair moved
    #    to the frozen execution set. `tests/test_s172_admission_binding.py`
    #    gates where the number now comes from.
    def _binds(src):
        serve = [n for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.FunctionDef) and n.name == "serve_trial"][0]
        out = []
        for n in ast.walk(serve):
            targets = []
            if isinstance(n, ast.Assign):
                targets = n.targets
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
                targets = [n.target]
            for tgt in targets:
                names = ([e for e in tgt.elts if isinstance(e, ast.Name)]
                         if isinstance(tgt, ast.Tuple)
                         else [tgt] if isinstance(tgt, ast.Name) else [])
                if any(x.id == "expected_workers" for x in names):
                    out.append(ast.get_source_segment(src, n))
        return out
    assert len(_binds(old)) == 1 and len(_binds(new)) == 1, (
        f"expected_workers must be bound exactly once: {_binds(old)} -> "
        f"{_binds(new)}")
    assert "worker_pool_size" in _binds(new)[0], _binds(new)[0]

    #    …and it is still bound in the PREAMBLE, at statement level — a binding
    #    that migrated inside the serve loop would be a dynamic reduction even
    #    if it still mentioned worker_pool_size.
    serve_new = [n for n in ast.walk(ast.parse(new))
                 if isinstance(n, ast.FunctionDef) and n.name == "serve_trial"][0]
    top_level_binds = [
        s for s in serve_new.body
        if isinstance(s, ast.Assign)
        and any(any(e.id == "expected_workers"
                    for e in (t.elts if isinstance(t, ast.Tuple) else [t])
                    if isinstance(e, ast.Name))
                for t in s.targets)]
    assert len(top_level_binds) == 1, (
        "expected_workers is no longer bound exactly once at serve_trial's top "
        "level — a binding inside the loop is a dynamic reduction")

    # 3. worker_pool_size keeps its numerical interpretation (no unit change).
    #    Counted over CODE ONLY (string/identifier AST nodes), because this
    #    deliverable's comments legitimately discuss the name — a raw text count
    #    would red on prose and green on a real edit hidden behind one.
    def _wps_sites(src):
        out = []
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Constant) and n.value == "worker_pool_size":
                out.append("const")
            elif isinstance(n, ast.Name) and n.id == "worker_pool_size":
                out.append("name")
            elif isinstance(n, ast.arg) and n.arg == "worker_pool_size":
                out.append("arg")
            elif isinstance(n, ast.keyword) and n.arg == "worker_pool_size":
                out.append("kw")
            elif isinstance(n, ast.Attribute) and n.attr == "worker_pool_size":
                out.append("attr")
        return sorted(out)
    assert _wps_sites(new) == _wps_sites(old), (
        f"worker_pool_size code sites changed {_wps_sites(old)} -> "
        f"{_wps_sites(new)} — its unit semantics are deferred to the later "
        f"fleet-authority work")

    # 4. NO finite serve_timeout was introduced: the default stays None in both
    #    the serve loop and the runner's context.
    assert '_timeout_raw = context.get("serve_timeout", None)' in new
    assert '"serve_timeout": kwargs.get("serve_timeout", None),' in new
    return ("matrix: 3/4 byte-identical, _handle_stripe_failure_locked "
            "certified by the superseding invariant (4 terminal decisions in "
            "order + both nonterminal outcomes + identity selector); "
            "expected_workers bound once in the preamble from worker_pool_size; "
            "serve_timeout default still None")


# ===========================================================================
# G-MUTANT ARM
# ===========================================================================
def _mutant_must_be_red(gate: str, scn: Callable, mutant) -> str:
    o = scn(mutant, live=False)
    _arm_verdicts[f"{gate}/mutant"] = o.ended_by
    assert o.ended_by != "own-decision", (
        f"MUTANT SURVIVED: with the outer threshold guard restored, {gate} still "
        f"reached a correct terminal decision ({o.summary()}). The live gate is "
        f"therefore NOT evidence that this repair does anything.")
    assert o.state != "committed", f"mutant committed: {o.summary()}"
    return (f"red under the restored guard: {o.ended_by} "
            f"(reasons={o.abort_reasons or 'none'})")


def g_mutant_summary():
    """The whole point (Beta named this gate explicitly): the five hang scenarios
    must flip red under the one-line revert, and the healthy control must NOT."""
    reds = ["G-ADMISSION-TIMEOUT", "G-ADMISSION-NO-RESET-ON-CHURN",
            "G-CROSS-CONSTANT", "G-CROSS-HYBRID", "G-FINAL-STAGE"]
    for gate in reds:
        live = _arm_verdicts.get(f"{gate}/live")
        mut = _arm_verdicts.get(f"{gate}/mutant")
        assert live == "own-decision", f"{gate} live arm did not run/pass: {live}"
        assert mut in ("still-hung", "harness-injected-clock"), (
            f"{gate} mutant arm was not red: {mut}")
    ctrl_live = _arm_verdicts.get("G-LONG-HEALTHY/live")
    ctrl_mut = _arm_verdicts.get("G-LONG-HEALTHY/mutant")
    assert ctrl_live == "own-decision" and ctrl_mut == "own-decision", (
        f"the clean control must be unaffected by the mutation: live={ctrl_live} "
        f"mutant={ctrl_mut}. A mutant that also breaks the healthy path is not "
        f"isolating the repair.")
    return (f"{len(reds)}/5 hang gates red under the restored guard; "
            f"healthy control unaffected")


# ===========================================================================
def main():
    print("=" * 70)
    print("S172 §4.3 ADMISSION LIVENESS — acceptance harness")
    print("live arm: serve_timeout=None (production default) — no wall clock can")
    print("          end a run; the harness budget is a FAILURE signal only")
    print("=" * 70)

    print("\n-- live source ------------------------------------------------")
    _check("G-ADMISSION-TIMEOUT: shortage before assignment -> terminal failure",
           g_admission_timeout)
    _check("G-ADMISSION-NO-RESET-ON-CHURN: window is stage-scoped, not churn-scoped",
           g_no_reset_on_churn)
    _check("G-CROSS-CONSTANT: loss below threshold -> Blocker-3 immediate failure",
           g_cross_constant)
    _check("G-CROSS-HYBRID: one reassignment executes below the original threshold",
           g_cross_hybrid)
    _check("G-FINAL-STAGE: completion below threshold still commits",
           g_final_stage)
    _check("G-LONG-HEALTHY: healthy unbounded run, admission window never fires",
           g_long_healthy)
    _check("G-REARM-STRUCTURE: the window is re-armed only at a stage boundary",
           g_rearm_structure)
    _check("G-CONFIG-FAILCLOSED: the admission window cannot be disabled",
           g_config_fail_closed)
    _check("G-FORBIDDEN-ABSENT: matrix / expected_workers / worker_pool_size / "
           "serve_timeout unchanged", g_forbidden_changes_absent)

    print("\n-- G-MUTANT: outer threshold guard restored (one line, AST-located) --")
    try:
        mutant = _build_mutant_module()
    except Exception as e:                                   # noqa: BLE001
        print(f"  [{_FAIL}] mutant could not be built: {e}")
        _results.append(("G-MUTANT: build", False, traceback.format_exc()))
        mutant = None

    if mutant is not None:
        for gate, scn in (
            ("G-ADMISSION-TIMEOUT", scn_admission_timeout),
            ("G-ADMISSION-NO-RESET-ON-CHURN", scn_no_reset_on_churn),
            ("G-CROSS-CONSTANT", scn_cross_constant),
            ("G-CROSS-HYBRID", scn_cross_hybrid),
            ("G-FINAL-STAGE", scn_final_stage_below_threshold),
        ):
            _check(f"G-MUTANT/{gate} must go red",
                   lambda g=gate, s=scn: _mutant_must_be_red(g, s, mutant))

        def _ctrl():
            o = scn_long_healthy(mutant, live=False)
            _arm_verdicts["G-LONG-HEALTHY/mutant"] = o.ended_by
            assert o.state == "committed", (
                f"the mutation must be TARGETED: a healthy full-pool run is "
                f"unaffected by the threshold guard, but got {o.summary()}")
            return "healthy control still green under the mutant (targeted)"
        _check("G-MUTANT/G-LONG-HEALTHY must stay green (targeted mutation)", _ctrl)

    _check("G-MUTANT: summary — the live gates are not vacuous", g_mutant_summary)

    print("=" * 70)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} S172 admission-liveness checks green")
    if passed != total:
        print("\nFAILURES (DO NOT COMMIT):")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n--- {name} ---\n{tb}")
        print("\nRESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
