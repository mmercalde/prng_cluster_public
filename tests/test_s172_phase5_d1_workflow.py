#!/usr/bin/env python3
"""
test_s172_phase5_d1_workflow.py — S172 Phase-5 Deliverable D1.0 acceptance harness.

D1.0 is the pair of narrow Phase-4 producer corrections that D1.1's assembly
engine depends on (docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D1.md REV5 §2):

  1. §2.1 [TB-D1-B1] `workflow_stages_for(base, False)` returned the
     forward-constant stage ALONE, so a real `test_both_modes=False` trial ran
     no P2 reverse pass and could never produce a constant bidirectional
     population. Constant is ALWAYS bidirectional (legacy Step 1 "PART 1:
     CONSTANT SKIP TEST (Always runs)"); only the hybrid pair is gated.
  2. §2.2 [TB-D1-C1] `abort_trial` conflated `False-because-already-aborted`
     (retry the discharge idempotently) with `False-because-committed` (refuse):
     it pre-read the trial state WITHOUT a lock, early-returned on `committed`
     only from that possibly STALE read, and then discharged the abort — sink
     tombstone + staged-file deletion — regardless of which terminal transition
     actually won the ledger's atomic `state='running'` race. The fix is
     CAS-result disambiguation plus a terminal-state re-read, with NO
     `_lifecycle_lock` acquisition inside `abort_trial` [TB-D1-DL].

Gates (each constructed to FAIL on the wrong behavior, not merely on a stub):

  W1  workflow_stages_for shape, both branches, ≥2 distinct bases.
  W2  producer gate: a REAL `test_both_modes=False` run over the real
      serve/publish surface publishes P1 AND P2 manifests, zero P3/P4.
  W3  commit reaches the sink with a COMPLETE constant bidirectional input
      (both directional populations non-empty, intersecting).
  W4  non-regression: Phase 4 63/63, Phase 3 17/17, D0 harness green.
  W5  terminal-race fencing — W5-A (ordinary commit-first), W5-B (ordinary
      abort-first), W5-R (the discriminating stale-read race exclusion).
  W6  a locked retry-matrix failure reaches the SYNCHRONOUS abort without
      deadlock — run in an isolated, timeout-terminable subprocess.

The harness drives the REAL lifecycle: real framed-socket workers, real staged
spool files on disk, real `Phase5Sink.publish_shard` from the real publish
surface, real `coordinator.commit_trial` / `coordinator.abort_trial` /
`coordinator.handle_stripe_failure`. No sleeps are used for synchronization —
every rendezvous is a `threading.Event`; timeouts are failure detectors only.

This module is standalone: it must run BEFORE `miner/range_miner_npz_writer.py`
(D1.1) exists and therefore never imports it.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 tests/test_s172_phase5_d1_workflow.py
Exit 0 = all green. Exit 1 = a check failed (DO NOT COMMIT).
"""
import dataclasses
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from miner.range_miner_coordinator import (  # noqa: E402
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    Phase5Sink,
    RangeMinerCoordinator,
    TrialAborted,
    run_trial_miner,
    workflow_phase_semantics,
    workflow_stages_for,
)
from miner.range_miner_worker import (  # noqa: E402
    MinerFramedSocket,
    VramCaps,
    build_substripe_payload_bytes,
    supported_variants,
)
from miner.range_miner_protocol import (  # noqa: E402
    RegisterMessage,
    StripeCompleteMessage,
    SubStripeResultMessage,
)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_results = []

# Every rendezvous below is an Event; these bounds are FAILURE DETECTORS, never
# synchronization (a correct implementation never reaches them).
_EV_TIMEOUT = 30.0
_JOIN_TIMEOUT = 60.0

CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
SPOOL_ROOT = "/var/spool/miner"

# §6.8 phase -> (family suffix for java_lcg, direction, skip_mode, prng_type)
PHASE_TABLE = {
    1: ("java_lcg",                "forward", "constant", "java_lcg"),
    2: ("java_lcg_reverse",        "reverse", "constant", "java_lcg"),
    3: ("java_lcg_hybrid",         "forward", "variable", "java_lcg_hybrid"),
    4: ("java_lcg_hybrid_reverse", "reverse", "variable", "java_lcg_hybrid"),
}

# Hand-chosen constant-pass populations with a NON-EMPTY intersection plus a
# forward-only and a reverse-only seed (W3). Seeds stay inside [0, 30).
PHASE_SURVIVORS = {
    1: [(3, 0.91), (5, 0.83), (7, 0.72)],   # forward / constant
    2: [(5, 0.61), (7, 0.55), (9, 0.44)],   # reverse / constant
}
EXPECT_FORWARD = {3, 5, 7}
EXPECT_REVERSE = {5, 7, 9}


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:  # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


# ---------------------------------------------------------------------------
# Shared real-lifecycle scaffolding (D0 harness pattern)
# ---------------------------------------------------------------------------
class _RecordingSink(Phase5Sink):
    """The REAL Phase-5 interface the coordinator drives. Records every publish /
    commit / abort delivery and can PAUSE inside commit_trial or abort_trial so a
    gate can hold the post-durable-transition window open deterministically."""

    def __init__(self):
        self.published = []
        self.commits = []
        self.aborts = []
        self.pause_commit = False
        self.pause_abort = False
        self.entered_commit = threading.Event()
        self.release_commit = threading.Event()
        self.entered_abort = threading.Event()
        self.release_abort = threading.Event()
        self.commit_release_timed_out = False
        self.abort_release_timed_out = False

    def publish_shard(self, manifest):
        self.published.append(manifest)

    def commit_trial(self, event):
        # Recorded BEFORE pausing so call counts are observable from the gate.
        self.commits.append(event)
        if self.pause_commit:
            self.entered_commit.set()
            if not self.release_commit.wait(timeout=_EV_TIMEOUT):
                self.commit_release_timed_out = True

    def abort_trial(self, event):
        self.aborts.append(event)
        if self.pause_abort:
            self.entered_abort.set()
            if not self.release_abort.wait(timeout=_EV_TIMEOUT):
                self.abort_release_timed_out = True


def _coord(tmp, sink, dbname="l.db"):
    ledger = MinerLedger(os.path.join(tmp, dbname))
    cfg = CoordinatorConfig(staging_dir=os.path.join(tmp, "staging"))
    return RangeMinerCoordinator(cfg, ledger, phase5_sink=sink)


def _register(coord, wid="hostA:gpu0"):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend="cuda",
        capabilities={"seed_caps": dict(CAPS),
                      "supported_variants": list(supported_variants())},
        node_config=node, now=100.0)


def _full_ctx(**over):
    ctx = dict(
        trial_number=7, window_size=5, offset=2,
        sessions=["midday", "evening"], skip_min=1, skip_max=9,
        prng_base="java_lcg", forward_threshold=0.40, reverse_threshold=0.45,
        dataset_sha256="d" * 64, residue_sha256="r" * 64,
    )
    ctx.update(over)
    return ctx


def _stage_attempt(coord, run_id, sid, conn, attempt=0, seed_start=0, seed_count=30,
                   now=100.0):
    """Record + stage a real inline sub-stripe so its shard is verified on disk."""
    survivors = [[seed_start, 0.91, None, [1]], [seed_start + 5, 0.83, None, [2, 3]]]
    _, pb = build_substripe_payload_bytes(sid, 0, seed_start, seed_count, survivors)
    size, sha = len(pb), hashlib.sha256(pb).hexdigest()
    coord.ledger.record_substripe_result(
        run_id, sid, attempt, 0, conn.worker_id, seed_start, seed_count,
        len(survivors), remote_spool_path=None, size_bytes=size, sha256=sha, now=now)
    res = coord.stage_inline_shard(run_id, sid, attempt, 0, seed_start, seed_count,
                                   survivors, size, sha, now=now)
    assert res["status"] == "verified", res
    assert os.path.isfile(res["staged_path"]), "real staged spool must exist on disk"
    return res["staged_path"]


def _running_trial_with_staged_files(tmp, sink, run_id, dbname="l.db", phase=1,
                                     publish=True):
    """Build a REAL running trial: durable context, a real assigned stripe, a real
    staged+verified spool file on disk, optionally published through the REAL
    publish surface. Returns (coord, stripe_id, staged_path)."""
    coord = _coord(tmp, sink, dbname)
    coord.ledger.create_trial(run_id, 7, now=100.0)
    coord.ledger.set_trial_context(run_id, _full_ctx())
    conn = _register(coord)
    family = PHASE_TABLE[phase][0]
    recs = coord.assign_stripes(run_id, family, phase, 30, [conn],
                                stripe_prefix=f"{run_id}__p{phase}", now=100.0)
    sid = recs[0]["stripe_id"]
    staged = _stage_attempt(coord, run_id, sid, conn, now=100.0)
    if publish:
        coord.ledger.record_stripe_complete(run_id, sid, 0, conn.worker_id, 1, 2)
        coord.finalize_stripe(run_id, sid, now=100.0)
        assert len(sink.published) == 1, "exactly one manifest must publish"
        assert sink.published[0]["local_spool_path"] == staged
    return coord, sid, staged


# ---------------------------------------------------------------------------
# W1 — workflow_stages_for: constant is bidirectional; hybrid pair gated
# ---------------------------------------------------------------------------
def w1_workflow_stages_shape():
    # Never hardcode one family: assert the shape for several distinct bases.
    for base in ("java_lcg", "pcg32", "xorshift128"):
        off = workflow_stages_for(base, False)
        assert off == [(base, 1), (f"{base}_reverse", 2)], \
            f"test_both_modes=False must run BOTH constant directions: {off}"
        # (this is exactly what fails against pre-D1.0 HEAD, which returns [(base, 1)])
        assert len(off) == 2, off

        on = workflow_stages_for(base, True)
        assert on == [(base, 1), (f"{base}_reverse", 2),
                      (f"{base}_hybrid", 3), (f"{base}_hybrid_reverse", 4)], on
        assert len(on) == 4, "the test_both_modes=True branch must be unchanged"

        # the False branch is a strict PREFIX of the True branch (same P1/P2 pair)
        assert on[:2] == off, (on, off)
        # and it contains NO variable-skip (hybrid) stage
        assert all(ph in (1, 2) for _fam, ph in off), off
        assert all(workflow_phase_semantics(ph)[1] == "constant" for _f, ph in off), off
        # both directions present, exactly once each
        assert sorted(workflow_phase_semantics(ph)[0] for _f, ph in off) == \
            ["forward", "reverse"], off


# ---------------------------------------------------------------------------
# W2/W3 — the REAL serve/publish surface with test_both_modes=False
# ---------------------------------------------------------------------------
class _FakeWorker:
    """Minimal framed-socket worker speaking the REAL MinerFramedSocket wire:
    register -> receive assign -> reply with one inline sub-stripe covering the
    whole stripe + StripeComplete. Survivor populations are phase-specific so the
    forward and reverse constant passes intersect but are not identical."""

    def __init__(self, host, port, hostname="hostA", gpu_id=0):
        self.host, self.port = host, port
        self.hostname, self.gpu_id = hostname, gpu_id
        self.worker_id = f"{hostname}:gpu{gpu_id}"
        self.assigns_received = []
        self.phases_seen = []
        self.err = None
        self._stop = threading.Event()
        self._t = None
        self.fs = None

    def connect_register(self):
        sock = socket.create_connection((self.host, self.port))
        self.fs = MinerFramedSocket(sock)
        self.fs.send_msg(RegisterMessage(
            worker_id=self.worker_id, hostname=self.hostname, gpu_id=self.gpu_id,
            gpu_name="fake", backend="cuda", vram_bytes=12 * 1024 ** 3,
            capabilities={"supported_variants": supported_variants(),
                          "seed_caps": dataclasses.asdict(VramCaps())}))

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
                    self.assigns_received.append(
                        (msg.stripe_id, msg.attempt, msg.phase, msg.family_name))
                    self.phases_seen.append(int(msg.phase))
                    self._respond(msg)
                elif msg.message_type == "shutdown":
                    break
        except Exception:  # noqa: BLE001
            self.err = traceback.format_exc()

    def _respond(self, assign):
        # Constant passes emit (seed, rate, null, [best_skip]) — worker:881-899.
        pop = PHASE_SURVIVORS.get(int(assign.phase))
        if pop is None:                       # an unexpected phase (pre/post-fix drift)
            pop = [(int(assign.seed_start), 0.5)]
        survivors = [[int(assign.seed_start) + s, r, None, [1]] for (s, r) in pop]
        payload_obj, pb = build_substripe_payload_bytes(
            assign.stripe_id, 0, assign.seed_start, assign.seed_count, survivors)
        # [S172 D6] Echo the assignment's resolved threshold as the EFFECTIVE
        # value, exactly as the real worker does off its executor (the parent's
        # fail-closed provenance gate requires it on every D6 assignment).
        eff = (assign.payload or {}).get("min_match_threshold")
        self.fs.send_msg(SubStripeResultMessage(
            worker_id=self.worker_id, stripe_id=assign.stripe_id, sub_index=0,
            seed_start=assign.seed_start, seed_count=assign.seed_count,
            survivor_count=len(survivors), inline=payload_obj, size_bytes=len(pb),
            sha256=hashlib.sha256(pb).hexdigest(), effective_threshold=eff))
        self.fs.send_msg(StripeCompleteMessage(
            worker_id=self.worker_id, stripe_id=assign.stripe_id,
            substripes_done=1, survivors_total=len(survivors),
            effective_threshold=eff))

    def stop(self):
        self._stop.set()
        try:
            self.fs.close()
        except Exception:  # noqa: BLE001
            pass


def _drive_two_modes_false_run(tmp, sink):
    """Run a REAL `test_both_modes=False` trial to a terminal state through the
    REAL default serve path (run_trial_miner with NO _serve and NO
    family_name/workflow_phase override, so `workflow_stages_for(prng_base, False)`
    is the producer authority) driving a real framed-socket worker.
    Returns (result, worker)."""
    ds = os.path.join(tmp, "dataset.json")
    with open(ds, "w") as f:
        f.write('[{"draw":1},{"draw":2},{"draw":3}]')

    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(8)
    port = lsock.getsockname()[1]

    holder = {}

    def run():
        try:
            holder["result"] = run_trial_miner(
                "run-d1", None, 7, "java_lcg", [1, 2, 3], 30, 0.40, 0.45,
                False,                                  # test_both_modes=False
                ds, worker_pool_size=1,
                staging_dir=os.path.join(tmp, "stg"), phase5_sink=sink,
                listen_sock=lsock,
                # NO family_name / workflow_phase override: the workflow STAGES
                # come from workflow_stages_for(prng_base, test_both_modes).
                skip_min=1, skip_max=9, window_size=5, offset=2,
                sessions=["midday", "evening"], serve_timeout=45.0)
        except Exception:  # noqa: BLE001
            holder["err"] = traceback.format_exc()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    w = _FakeWorker("127.0.0.1", port)
    try:
        w.connect_register()
        w.start_loop()
        t.join(timeout=_JOIN_TIMEOUT)
        assert not t.is_alive(), "serve_trial did not terminate in time"
        assert "err" not in holder, holder.get("err")
        assert w.err is None, w.err
        return holder["result"], w
    finally:
        w.stop()
        try:
            lsock.close()
        except Exception:  # noqa: BLE001
            pass


def _manifests_by_phase(sink):
    out = {}
    for m in sink.published:
        out.setdefault(int(m["workflow_phase"]), []).append(m)
    return out


def w2_producer_publishes_both_constant_directions():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _RecordingSink()
        result, w = _drive_two_modes_false_run(tmp, sink)

        assert result["state"] == "committed", result["state"]
        by_phase = _manifests_by_phase(sink)

        # (1) BOTH constant directions published. Pre-D1.0 HEAD publishes phase 1
        # only, so this assertion fails against it — the producer defect.
        assert set(by_phase) == {1, 2}, \
            f"a test_both_modes=False trial must publish P1 AND P2 only: {sorted(by_phase)}"
        assert by_phase[1] and by_phase[2]

        # (2) ZERO variable-skip manifests, from either the sink or the worker.
        assert 3 not in by_phase and 4 not in by_phase, sorted(by_phase)
        assert sorted(set(w.phases_seen)) == [1, 2], \
            f"only the two constant stages may be dispatched: {w.phases_seen}"

        # (3) EXPLICIT phase identities on the published manifests (D0 semantics).
        for phase, (family, direction, skip_mode, prng_type) in PHASE_TABLE.items():
            if phase not in by_phase:
                continue
            for m in by_phase[phase]:
                meta = m["trial_metadata"]
                assert meta["direction"] == direction, (phase, meta["direction"])
                assert meta["skip_mode"] == skip_mode, (phase, meta["skip_mode"])
                assert meta["family_name"] == family, (phase, meta["family_name"])
                assert meta["prng_type"] == prng_type, (phase, meta["prng_type"])
                assert meta["workflow_phase"] == phase == int(m["workflow_phase"])
                assert meta["threshold_used"] == (0.40 if direction == "forward"
                                                  else 0.45), meta
        # explicit identity strings, both directions, constant only
        dirs = {m["trial_metadata"]["direction"] for m in sink.published}
        modes = {m["trial_metadata"]["skip_mode"] for m in sink.published}
        assert dirs == {"forward", "reverse"}, dirs
        assert modes == {"constant"}, modes

        # (4) the run's stage set IS the producer authority's answer
        assert [(fam, ph) for (fam, ph) in workflow_stages_for("java_lcg", False)] == \
            [("java_lcg", 1), ("java_lcg_reverse", 2)]


def w3_commit_carries_complete_bidirectional_constant():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _RecordingSink()
        result, _w = _drive_two_modes_false_run(tmp, sink)
        run_id = result["run_id"]

        # commit reached the REAL sink exactly once for this run
        assert result["committed"] is True, result
        assert len(sink.commits) == 1, sink.commits
        ev = sink.commits[0]
        assert ev["event_type"] == "trial_commit" and ev["run_id"] == run_id
        assert ev["event_id"] == f"{run_id}:commit", ev
        assert ev.get("delivery") == "done", ev
        assert sink.aborts == [], sink.aborts

        # manifest-coverage: BOTH constant directional populations are present and
        # non-empty, and they intersect (full assembly is D1.1).
        by_dir = {"forward": set(), "reverse": set()}
        for m in sink.published:
            meta = m["trial_metadata"]
            assert meta["skip_mode"] == "constant", meta
            assert m["run_id"] == run_id, (m["run_id"], run_id)
            path = m["local_spool_path"]
            assert os.path.isfile(path), f"staged spool must survive commit: {path}"
            with open(path) as f:
                payload = json.load(f)
            assert payload["schema_version"] == "s172_substripe_v1", payload
            assert payload["stripe_id"] == m["stripe_id"]
            assert payload["sub_index"] == m["sub_index"]
            for entry in payload["survivors"]:
                by_dir[meta["direction"]].add(int(entry[0]))

        assert by_dir["forward"], "forward constant population must be non-empty"
        assert by_dir["reverse"], "reverse constant population must be non-empty"
        assert by_dir["forward"] == EXPECT_FORWARD, by_dir["forward"]
        assert by_dir["reverse"] == EXPECT_REVERSE, by_dir["reverse"]
        inter = by_dir["forward"] & by_dir["reverse"]
        assert inter == {5, 7}, inter
        # a genuine bidirectional fixture: each direction also has an exclusive seed
        assert by_dir["forward"] - by_dir["reverse"] == {3}
        assert by_dir["reverse"] - by_dir["forward"] == {9}


# ---------------------------------------------------------------------------
# W4 — non-regression: Phase 4 63/63, Phase 3 17/17, D0 harness green
# ---------------------------------------------------------------------------
def _run_suite(rel_path, expect_substr, timeout=900):
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, os.path.join(_ROOT, rel_path)],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-25:])
    assert proc.returncode == 0, \
        f"{rel_path} exited {proc.returncode}\n{tail}\n{(proc.stderr or '')[-2000:]}"
    assert expect_substr in (proc.stdout or ""), \
        f"{rel_path}: expected {expect_substr!r} in output\n{tail}"


def w4_non_regression_suites():
    _run_suite("tests/test_s172_phase4_coordinator.py", "63/63 checks green")
    _run_suite("tests/test_s172_phase3_worker.py", "17/17 gates green")
    _run_suite("tests/test_s172_phase5_d0.py", "12/12 D0 gate checks green")


# ---------------------------------------------------------------------------
# W5 — abort/commit terminal-race fencing [TB-D1-C1, TB-D1-W5R2]
# ---------------------------------------------------------------------------
def w5a_ordinary_commit_first():
    """Commit's durable terminal transition has already happened and its sink
    delivery is IN FLIGHT when a real abort arrives: the abort must refuse,
    deliver nothing to the sink, and delete no staged spool."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _RecordingSink()
        coord, _sid, staged = _running_trial_with_staged_files(tmp, sink, "w5a")
        sink.pause_commit = True

        holder = {}

        def commit():
            try:
                holder["event"] = coord.commit_trial("w5a", now=200.0)
            except Exception:  # noqa: BLE001
                holder["err"] = traceback.format_exc()

        t = threading.Thread(target=commit, daemon=True)
        t.start()
        assert sink.entered_commit.wait(timeout=_EV_TIMEOUT), \
            "sink.commit_trial was never entered"
        # the durable committed transition has already occurred
        assert coord.ledger.get_trial("w5a")["state"] == "committed"

        res = coord.abort_trial("w5a", reason="race", now=201.0)
        assert res["cleanup"] == "refused", res
        assert res["refused"] == "already_committed", res
        assert res["first"] is False, res
        assert len(sink.aborts) == 0, "sink.abort_trial must NOT be called"
        assert os.path.isfile(staged), "a committed trial's staged spool must survive"

        sink.release_commit.set()
        t.join(timeout=_JOIN_TIMEOUT)
        assert not t.is_alive(), "commit_trial did not return"
        assert "err" not in holder, holder.get("err")
        assert holder["event"]["delivery"] == "done", holder["event"]
        assert sink.commit_release_timed_out is False
        assert len(sink.commits) == 1 and len(sink.aborts) == 0
        assert coord.ledger.get_trial("w5a")["state"] == "committed"
        assert coord.ledger.get_trial("w5a")["commit_delivery_status"] == "done"


def w5b_ordinary_abort_first():
    """Abort's durable terminal transition has already happened and its sink
    delivery is IN FLIGHT when a real commit arrives: commit must raise
    TrialAborted at the coordinator and never reach the sink."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _RecordingSink()
        coord, _sid, staged = _running_trial_with_staged_files(tmp, sink, "w5b")
        sink.pause_abort = True

        holder = {}

        def abort():
            try:
                holder["res"] = coord.abort_trial("w5b", reason="race", now=200.0)
            except Exception:  # noqa: BLE001
                holder["err"] = traceback.format_exc()

        t = threading.Thread(target=abort, daemon=True)
        t.start()
        assert sink.entered_abort.wait(timeout=_EV_TIMEOUT), \
            "sink.abort_trial was never entered"
        assert coord.ledger.get_trial("w5b")["state"] == "aborted"

        raised = False
        try:
            coord.commit_trial("w5b", now=201.0)
        except TrialAborted:
            raised = True
        assert raised, "commit after a durable abort must raise TrialAborted"
        assert len(sink.commits) == 0, "sink.commit_trial must NOT be called"

        sink.release_abort.set()
        t.join(timeout=_JOIN_TIMEOUT)
        assert not t.is_alive(), "abort_trial did not return"
        assert "err" not in holder, holder.get("err")
        assert holder["res"]["cleanup"] == "done", holder["res"]
        assert holder["res"]["first"] is True, holder["res"]
        assert sink.abort_release_timed_out is False
        assert len(sink.aborts) == 1 and len(sink.commits) == 0
        # L7: staged files are deleted only AFTER the sink returned successfully
        assert not os.path.exists(staged), "abort discharge must remove staged spools"
        assert coord.ledger.get_trial("w5b")["abort_cleanup_status"] == "done"


def w5r_stale_read_race_exclusion():
    """THE DISCRIMINATING GATE [TB-D1-W5R2]. Hold abort in the exact vulnerable
    interval — it has observed `running` but has NOT yet called
    mark_trial_aborted — then let a real commit win the atomic race. CAS
    semantics: commit is NOT expected to block. The corrected abort must
    disambiguate its False from mark_trial_aborted by RE-READING the terminal
    state and refusing. Pre-fix HEAD instead discharges the abort: it fires the
    sink abort AND deletes the committed trial's staged spools."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _RecordingSink()
        coord, _sid, staged = _running_trial_with_staged_files(tmp, sink, "w5r")

        abort_read_running = threading.Event()
        release_abort_read = threading.Event()
        state = {"intercepted": False, "abort_tid": None, "read_state": None,
                 "release_ok": None}
        real_get_trial = coord.ledger.get_trial

        def wrapped_get_trial(rid, *a, **kw):
            # the REAL method runs first; only the abort worker thread's FIRST
            # relevant call is intercepted (the pre-read), never commit's.
            row = real_get_trial(rid, *a, **kw)
            if (rid == "w5r" and not state["intercepted"]
                    and threading.get_ident() == state["abort_tid"]):
                state["intercepted"] = True
                state["read_state"] = None if row is None else row["state"]
                abort_read_running.set()
                state["release_ok"] = release_abort_read.wait(timeout=_EV_TIMEOUT)
            return row

        coord.ledger.get_trial = wrapped_get_trial
        holder = {}

        def abort():
            state["abort_tid"] = threading.get_ident()
            try:
                holder["res"] = coord.abort_trial("w5r", reason="race", now=200.0)
            except Exception:  # noqa: BLE001
                holder["err"] = traceback.format_exc()

        commit_holder = {}

        def commit():
            try:
                commit_holder["event"] = coord.commit_trial("w5r", now=201.0)
            except Exception:  # noqa: BLE001
                commit_holder["err"] = traceback.format_exc()

        ta = threading.Thread(target=abort, daemon=True)
        ta.start()
        try:
            # (3) abort has read `running` and is paused BEFORE mark_trial_aborted
            assert abort_read_running.wait(timeout=_EV_TIMEOUT), \
                "abort's pre-read was never intercepted"
            assert state["read_state"] == "running", \
                f"the intercepted pre-read must observe 'running': {state['read_state']}"

            # (4/5) commit must COMPLETE while abort stays paused (no lock held)
            tb = threading.Thread(target=commit, daemon=True)
            tb.start()
            tb.join(timeout=_JOIN_TIMEOUT)
            assert not tb.is_alive(), \
                "commit_trial blocked while abort held its window open (no lock is " \
                "permitted in abort_trial — TB-D1-DL)"
            assert "err" not in commit_holder, commit_holder.get("err")
            assert commit_holder["event"]["delivery"] == "done", commit_holder["event"]
            assert coord.ledger.get_trial("w5r")["state"] == "committed"
            assert len(sink.commits) == 1, sink.commits
            assert len(sink.aborts) == 0, sink.aborts
            assert not abort_read_running.is_set() or ta.is_alive(), \
                "the abort worker must still be paused"

            # (6) release the stale read; abort now loses the CAS and must refuse
            release_abort_read.set()
            ta.join(timeout=_JOIN_TIMEOUT)
            assert not ta.is_alive(), "abort_trial did not return"
            assert "err" not in holder, holder.get("err")
            assert state["release_ok"] is True
        finally:
            coord.ledger.get_trial = real_get_trial
            release_abort_read.set()
            ta.join(timeout=5.0)

        # (10) the corrected abort re-read `committed` and refused
        res = holder["res"]
        assert res["cleanup"] == "refused", res
        assert res["refused"] == "already_committed", res
        assert res["first"] is False, res
        assert len(sink.aborts) == 0, \
            "the losing abort must NOT deliver to the sink (pre-fix HEAD fires both)"
        assert len(sink.commits) == 1, sink.commits
        assert os.path.isfile(staged), \
            "the committed trial's staged spool must NOT be deleted by the losing abort"
        # the committed result stays intact and retrievable
        trial = real_get_trial("w5r")
        assert trial["state"] == "committed", trial["state"]
        assert trial["commit_delivery_status"] == "done", trial
        assert trial["commit_event_id"] == "w5r:commit", trial


# ---------------------------------------------------------------------------
# W6 — locked retry-matrix failure -> synchronous abort, no deadlock
#      [TB-D1-W6]. Run in an ISOLATED, TIMEOUT-TERMINABLE SUBPROCESS: a plain
#      future.result(timeout=) can report a timeout while leaving the caller
#      thread blocked holding _lifecycle_lock and the cleanup-executor thread
#      blocked waiting for it, and those surviving threads can prevent the test
#      process from exiting. The parent kills the child instead.
# ---------------------------------------------------------------------------
_W6_CASES = {
    # case -> (phase, retryable, expected reason)
    "non_retryable": (1, False, "non_retryable"),
    "constant_phase": (2, True, "constant_phase"),
}


def _w6_child(case):
    """Child-process body (see __main__ dispatch). Prints a JSON verdict."""
    import concurrent.futures
    phase, retryable, expect_reason = _W6_CASES[case]
    with tempfile.TemporaryDirectory() as tmp:
        sink = _RecordingSink()
        run_id = f"w6_{case}"
        coord, sid, staged = _running_trial_with_staged_files(
            tmp, sink, run_id, phase=phase, publish=False)

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(coord.handle_stripe_failure, run_id, sid,
                        retryable, [], 300.0)
        try:
            # bounded completion timeout = the FAILURE DETECTOR, not a rendezvous
            action = fut.result(timeout=45.0)
        except concurrent.futures.TimeoutError:
            print(json.dumps({"ok": False, "case": case,
                              "error": "handle_stripe_failure DEADLOCKED "
                                       "(no completion within 45s)"}), flush=True)
            # surviving blocked threads must not stop the child from dying
            os._exit(3)

        trial = coord.ledger.get_trial(run_id)
        verdict = {
            "ok": True,
            "case": case,
            "action": action.get("action"),
            "reason": action.get("reason"),
            "trial_state": trial["state"],
            "abort_cleanup_status": trial["abort_cleanup_status"],
            "sink_aborts": len(sink.aborts),
            "sink_commits": len(sink.commits),
            "abort_event_ids": [e.get("event_id") for e in sink.aborts],
            "expect_reason": expect_reason,
            "staged_removed": not os.path.exists(staged),
        }
        try:
            assert verdict["action"] == "fail_trial", verdict
            assert verdict["reason"] == expect_reason, verdict
            assert verdict["trial_state"] == "aborted", verdict
            assert verdict["sink_aborts"] == 1, verdict
            assert verdict["sink_commits"] == 0, verdict
            assert verdict["abort_cleanup_status"] == "done", verdict
            assert verdict["abort_event_ids"] == [f"{run_id}:abort"], verdict
            assert verdict["staged_removed"] is True, verdict
        except AssertionError as e:
            verdict["ok"] = False
            verdict["error"] = str(e)
        print(json.dumps(verdict), flush=True)
        ex.shutdown(wait=False)
        return 0 if verdict["ok"] else 4


def w6_locked_matrix_failure_no_deadlock():
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    for case in _W6_CASES:
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--w6-child", case],
                cwd=_ROOT, env=env, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            # The child (killed by subprocess.run) never completed: the caller
            # thread holds _lifecycle_lock while waiting on the cleanup executor's
            # Future.result() and the executor thread cannot acquire it — the exact
            # deadlock the REV4 locked-abort design would introduce.
            raise AssertionError(
                f"W6[{case}]: the isolated child process DEADLOCKED and was "
                f"terminated on timeout (handle_stripe_failure -> fail_trial -> "
                f"abort_trial never completed)")
        out = (proc.stdout or "").strip().splitlines()
        assert out, f"W6[{case}]: child produced no verdict\n{proc.stderr[-2000:]}"
        verdict = json.loads(out[-1])
        assert proc.returncode == 0 and verdict["ok"], \
            f"W6[{case}]: rc={proc.returncode} verdict={verdict}\n{proc.stderr[-2000:]}"
        assert verdict["action"] == "fail_trial", verdict
        assert verdict["reason"] == _W6_CASES[case][2], verdict
        assert verdict["trial_state"] == "aborted", verdict
        assert verdict["sink_aborts"] == 1, verdict
        assert verdict["abort_cleanup_status"] == "done", verdict


def main():
    print("=" * 74)
    print("S172 Phase-5 D1.0 — workflow + terminal-race acceptance harness")
    print("=" * 74)
    _check("W1: workflow_stages_for — constant bidirectional, hybrid gated",
           w1_workflow_stages_shape)
    _check("W2: producer publishes P1+P2, zero P3/P4 (test_both_modes=False)",
           w2_producer_publishes_both_constant_directions)
    _check("W3: commit carries a complete constant bidirectional input",
           w3_commit_carries_complete_bidirectional_constant)
    _check("W5-A: ordinary commit-first ordering — abort refuses",
           w5a_ordinary_commit_first)
    _check("W5-B: ordinary abort-first ordering — commit raises TrialAborted",
           w5b_ordinary_abort_first)
    _check("W5-R: stale-read race exclusion (the discriminating gate)",
           w5r_stale_read_race_exclusion)
    _check("W6: locked retry-matrix failure -> synchronous abort, no deadlock",
           w6_locked_matrix_failure_no_deadlock)
    _check("W4: non-regression — Phase 4 63/63, Phase 3 17/17, D0 green",
           w4_non_regression_suites)

    print("=" * 74)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} D1.0 gate checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All D1.0 gate checks green — workflow + terminal-race corrections are "
          "contract-validated (pending Team Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--w6-child":
        sys.exit(_w6_child(sys.argv[2]))
    sys.exit(main())
