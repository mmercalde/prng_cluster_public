#!/usr/bin/env python3
"""
test_s172_phase5_d2_directional_uniqueness.py — S172 Phase-5 Deliverable D2.

Directional uniqueness at BOTH enforcement layers
(docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D2.md, absorbing the Team Beta
corrected two-gate brief [TB-D2-COR]).

The invariant: within ONE (run_id, workflow_phase, accepted_attempt, family) a
seed appears at most once. A duplicate is a producer/coverage defect; Phase 5
must raise DirectionalDuplicateError and must NEVER keep the max match rate or
otherwise deduplicate.

The originally requested single fixture — two overlapping shards travelling
through a legitimately accepted/reconciled attempt into the sink — is IMPOSSIBLE
by design at HEAD b9c6120 (assign_stripes disjoint tiles; _coverage_exact walks a
cursor so any gap/overlap fails coverage; finalize_stripe publishes ONLY on a
reconciled+verified attempt). D2 therefore proves the two barriers SEPARATELY:

  D2-A  the REAL producer rejects the overlap BEFORE Phase 5 ever sees it — a
        misbehaving framed-socket worker declares overlapping sub-stripe ranges
        over the real MinerFramedSocket wire; the definitive reconciliation
        failure routes through the real matrix (constant-phase -> fail_trial)
        and NOTHING is published. This fires ONLY on the FULL serve path
        (eligible_provider is not None, coordinator:1897-1905) — a bare
        finalize_stripe() call would leave the stripe parked in staging, so D2-A
        drives run_trial_miner's real serve loop.

  D2-B  Phase 5 STILL fails closed if that upstream barrier is ever bypassed or
        regresses — a labeled `DIRECT SINK INVARIANT-BREAK PROBE` delivers a
        complete, individually-valid manifest set (two shards in one directional
        population sharing one seed at different match rates) through the PUBLIC
        sink.publish_shard surface, then the REAL coordinator.commit_trial
        assembles and DirectionalDuplicateError is raised with all 13 structured
        attributes. Delivery is "failed"; no assembly installs; no silent dedup.

Plus four negative controls (the invariant is SCOPED, not global) and the
blocking non-regression runner.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 tests/test_s172_phase5_d2_directional_uniqueness.py
Exit 0 = all green. Exit 1 = a check failed (DO NOT COMMIT).

Single-gate dispatch (used by the §6 mutation proof, which must NOT run the NR
runner because the D1.1 g14 subprocess would also red under a mutated writer):
    PYTHONPATH=. python3 tests/test_s172_phase5_d2_directional_uniqueness.py --gate d2a
    PYTHONPATH=. python3 tests/test_s172_phase5_d2_directional_uniqueness.py --gate d2b
"""
import copy
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
    SH_VERIFIED,
    evaluate_stripe_completion,
    run_trial_miner,
    workflow_stages_for,
)
from miner.range_miner_worker import (  # noqa: E402
    MinerFramedSocket,
    build_substripe_payload_bytes,
    supported_variants,
)
from miner.range_miner_protocol import (  # noqa: E402
    RegisterMessage,
    StripeCompleteMessage,
    SubStripeResultMessage,
)
from miner.range_miner_npz_writer import (  # noqa: E402
    AssemblingPhase5Sink,
    DirectionalDuplicateError,
    ManifestReplayConflict,
    MinerTrialAssembly,
    assemble_trial,
)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_results = []

# Every rendezvous below is a threading.Event or thread.join; these bounds are
# FAILURE DETECTORS, never synchronization (a correct implementation never hits
# them).
_JOIN_TIMEOUT = 60.0
_SERVE_TIMEOUT = 45.0

PRNG_BASE = "java_lcg"
SPOOL_ROOT = "/var/spool/miner"

# §6.8 phase -> (family_name, direction, skip_mode, prng_type)
PHASE_TABLE = {
    1: ("java_lcg",                "forward", "constant", "java_lcg"),
    2: ("java_lcg_reverse",        "reverse", "constant", "java_lcg"),
    3: ("java_lcg_hybrid",         "forward", "variable", "java_lcg_hybrid"),
    4: ("java_lcg_hybrid_reverse", "reverse", "variable", "java_lcg_hybrid"),
}

# The immutable trial-global context (all _SERVE_CONTEXT_REQUIRED fields present).
CTX = dict(trial_number=7, window_size=5, offset=2,
           sessions=["midday", "evening"], skip_min=1, skip_max=9,
           prng_base=PRNG_BASE, forward_threshold=0.40, reverse_threshold=0.45,
           dataset_sha256="d" * 64, residue_sha256="r" * 64)

# Small caps + a small macro size so the REAL producer yields controllable
# single-stripe / two-sub-stripe runs for the manifest-generation helper
# (D2-B + negative controls): 20 seeds / macro 20 = 1 stripe/phase; 20/cap 10 =
# 2 sub-stripes ([0,10), [10,20)).
GEN_TOTAL = 20
GEN_MACRO = 20
GEN_CAP = 10


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:  # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


def _raises(exc, fn, *a, **kw):
    """Assert `fn` raises `exc` (exactly that class or a subclass) and return it."""
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    except Exception as other:  # noqa: BLE001
        raise AssertionError(
            f"expected {exc.__name__}, got {type(other).__name__}: {other}")
    raise AssertionError(f"expected {exc.__name__}, nothing was raised")


# ===========================================================================
# Shared real-lifecycle scaffolding (D0/D1 harness pattern)
# ===========================================================================
class _RecordingSink(Phase5Sink):
    """A minimal non-asserting Phase-5 interface — used only to GENERATE valid
    manifests through the real producer (its publish path)."""

    def __init__(self):
        self.published = []
        self.commits = []
        self.aborts = []

    def publish_shard(self, manifest):
        self.published.append(manifest)

    def commit_trial(self, event):
        self.commits.append(event)

    def abort_trial(self, event):
        self.aborts.append(event)


def _mk_coord(tmp, sink, dbname, *, seed_cap=GEN_CAP, macro=GEN_MACRO):
    ledger = MinerLedger(os.path.join(tmp, dbname))
    cfg = CoordinatorConfig(
        staging_dir=os.path.join(tmp, "stg_" + dbname),
        miner_stripe_size=macro,
        seed_cap_amd=seed_cap, seed_cap_nvidia=seed_cap,
        seed_cap_amd_hybrid=seed_cap, seed_cap_nvidia_hybrid=seed_cap)
    return RangeMinerCoordinator(cfg, ledger, phase5_sink=sink)


def _register(coord, seed_cap=GEN_CAP, wid="hostA:gpu0"):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    caps = {"amd": seed_cap, "nvidia": seed_cap,
            "amd_hybrid": seed_cap, "nvidia_hybrid": seed_cap}
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend="cuda",
        capabilities={"seed_caps": caps,
                      "supported_variants": list(supported_variants())},
        node_config=node, now=100.0)


def _entries(pop, variable, s_start, s_count):
    """The worker's canonical survivor tuples for one sub-stripe range
    (miner/range_miner_worker.py:881-899). Constant passes emit
    (seed, rate, null, [best_skip]); hybrid passes emit
    (seed, rate, strategy_id, skip_sequence). Every seed MUST sit in-range."""
    out = []
    for seed, rate in pop:
        assert s_start <= seed < s_start + s_count, (seed, s_start, s_count)
        if variable:
            out.append([seed, rate, (seed % 3) + 1, [1, 2, 3]])
        else:
            out.append([seed, rate, None, [(seed % 4) + 1]])
    return out


def _gen_manifests(coord, run_id, phases, pops):
    """Drive the REAL producer surface to publish a set of individually-valid
    manifests (durable context, real assigned stripes, real staged+verified spool
    files on disk, real publish_attempt -> publish_shard) up to but NOT including
    the terminal commit.

    `pops` = {phase: {sub_index: [(seed, rate), ...]}} — exact-tiling populations,
    so every stripe reconciles and publishes. Returns the coordinator's own
    published-manifest list (coordinator.enqueued), independent of sink type."""
    coord.ledger.create_trial(run_id, CTX["trial_number"], now=100.0)
    coord.ledger.set_trial_context(run_id, dict(CTX))
    conn = _register(coord)
    for phase in phases:
        family = PHASE_TABLE[phase][0]
        variable = PHASE_TABLE[phase][2] == "variable"
        recs = coord.assign_stripes(run_id, family, phase, GEN_TOTAL, [conn],
                                    stripe_prefix=f"{run_id}__p{phase}", now=100.0)
        assert len(recs) == GEN_TOTAL // GEN_MACRO, recs
        for rec in recs:
            assert rec["claimed"], rec
            assert rec["expected_substripes"] == GEN_MACRO // GEN_CAP, rec
            sid = rec["stripe_id"]
            surv_total = 0
            for sub_index in range(rec["expected_substripes"]):
                s_start = rec["seed_start"] + sub_index * GEN_CAP
                entries = _entries(pops.get(phase, {}).get(sub_index, []),
                                   variable, s_start, GEN_CAP)
                _, pb = build_substripe_payload_bytes(
                    sid, sub_index, s_start, GEN_CAP, entries)
                size, sha = len(pb), hashlib.sha256(pb).hexdigest()
                coord.ledger.record_substripe_result(
                    run_id, sid, 0, sub_index, conn.worker_id, s_start, GEN_CAP,
                    len(entries), remote_spool_path=None, size_bytes=size,
                    sha256=sha, now=100.0)
                res = coord.stage_inline_shard(run_id, sid, 0, sub_index, s_start,
                                               GEN_CAP, entries, size, sha, now=100.0)
                assert res["status"] == "verified", res
                assert os.path.isfile(res["staged_path"]), res
                surv_total += len(entries)
            assert coord.ledger.record_stripe_complete(
                run_id, sid, 0, conn.worker_id, rec["expected_substripes"],
                surv_total), "StripeComplete transition failed"
            coord.finalize_stripe(run_id, sid, now=100.0)
    return list(coord.enqueued)


def _repoint(manifest, tmp, payload_obj, tag):
    """Write a mutated payload to a NEW staged path and repoint a COPY of the
    manifest at it, RECOMPUTING expected_size + expected_sha256 so the assembler
    exercises the validator, not merely the digest check. The manifest's
    trial_metadata / stripe_id / sub_index / event_id are preserved."""
    raw = json.dumps(payload_obj, separators=(",", ":"),
                     sort_keys=True).encode("utf-8")
    path = os.path.join(tmp, f"mutated_{tag}.json")
    with open(path, "wb") as f:
        f.write(raw)
    out = copy.deepcopy(manifest)
    out["local_spool_path"] = path
    out["expected_size"] = len(raw)
    out["expected_sha256"] = hashlib.sha256(raw).hexdigest()
    return out


# ===========================================================================
# D2-A — REAL producer overlap rejection (the misbehaving framed-socket worker)
# ===========================================================================
# Isolated-coverage construction over a stripe assigned [0, 30) (Team Beta
# preferred): three sub-stripes whose declared COUNTS sum to 30 (so seed_sum
# stays True) but whose ranges overlap at 9 and leave a compensating gap at 19,
# so coverage_ok is the PROVABLY ONLY red predicate.
D2A_STRIPE_SEEDS = 30
D2A_CAP = 10                         # ceil(30/10) = 3 expected sub-stripes
# (sub_index, seed_start, seed_count, survivors[[seed,rate,strategy,skips]])
D2A_SUBS = [
    (0,  0, 10, [[2, 0.90, None, [1]], [9, 0.80, None, [2]]]),   # [0,10)
    (1,  9, 10, [[9, 0.70, None, [3]], [15, 0.60, None, [4]]]),  # [9,19) overlap @9
    (2, 20, 10, [[25, 0.50, None, [1]]]),                        # [20,30) gap @19
]
D2A_SURVIVORS_TOTAL = 5              # 2 + 2 + 1
D2A_SEED_SUM = 30                    # 10 + 10 + 10 == stripe seed_count


class _CountingAssemblingSink(AssemblingPhase5Sink):
    """The REAL Phase-5 sink under test, instrumented with call counters + a
    record of every abort event_id. No behavior is overridden — each method
    delegates to the production implementation."""

    def __init__(self):
        super().__init__()
        self.publish_calls = 0
        self.commit_calls = 0
        self.abort_calls = 0
        self.abort_event_ids = []
        self._lock2 = threading.Lock()

    def publish_shard(self, manifest):
        with self._lock2:
            self.publish_calls += 1
        return super().publish_shard(manifest)

    def commit_trial(self, event):
        with self._lock2:
            self.commit_calls += 1
        return super().commit_trial(event)

    def abort_trial(self, event):
        with self._lock2:
            self.abort_calls += 1
            self.abort_event_ids.append(event.get("event_id"))
        return super().abort_trial(event)


class _OverlapWorker:
    """A MISBEHAVING framed-socket worker speaking the REAL MinerFramedSocket
    wire: register -> receive assign -> reply with THREE overlapping inline
    sub-stripes + a StripeComplete whose totals reconcile on everything EXCEPT
    coverage. This is the realistic producer-defect vector (§3)."""

    def __init__(self, host, port, seed_cap=D2A_CAP, hostname="hostA", gpu_id=0):
        self.host, self.port = host, port
        self.seed_cap = seed_cap
        self.hostname, self.gpu_id = hostname, gpu_id
        self.worker_id = f"{hostname}:gpu{gpu_id}"
        self.assigns_received = []
        self.err = None
        self._stop = threading.Event()
        self._t = None
        self.fs = None

    def connect_register(self):
        sock = socket.create_connection((self.host, self.port))
        self.fs = MinerFramedSocket(sock)
        caps = {"amd": self.seed_cap, "nvidia": self.seed_cap,
                "amd_hybrid": self.seed_cap, "nvidia_hybrid": self.seed_cap}
        self.fs.send_msg(RegisterMessage(
            worker_id=self.worker_id, hostname=self.hostname, gpu_id=self.gpu_id,
            gpu_name="fake", backend="cuda", vram_bytes=12 * 1024 ** 3,
            capabilities={"supported_variants": supported_variants(),
                          "seed_caps": caps}))

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
                        (msg.stripe_id, int(msg.phase), msg.family_name))
                    self._respond(msg)
                elif msg.message_type == "shutdown":
                    break
        except Exception:  # noqa: BLE001
            self.err = traceback.format_exc()

    def _respond(self, assign):
        total_surv = 0
        for sub_index, s_start, s_count, survivors in D2A_SUBS:
            payload_obj, pb = build_substripe_payload_bytes(
                assign.stripe_id, sub_index, s_start, s_count, survivors)
            # [S172 D6] Echo the assignment's resolved threshold as the
            # EFFECTIVE value, exactly as the real worker does off its executor
            # (the parent's fail-closed provenance gate requires it on every D6
            # assignment, and requires all sub-stripes to agree).
            self.fs.send_msg(SubStripeResultMessage(
                worker_id=self.worker_id, stripe_id=assign.stripe_id,
                sub_index=sub_index, seed_start=s_start, seed_count=s_count,
                survivor_count=len(survivors), inline=payload_obj,
                size_bytes=len(pb), sha256=hashlib.sha256(pb).hexdigest(),
                effective_threshold=(assign.payload or {}).get(
                    "min_match_threshold")))
            total_surv += len(survivors)
        # substripes_done == expected_substripes == 3; survivors_total == the true
        # sum -> coverage is the ONLY red predicate.
        self.fs.send_msg(StripeCompleteMessage(
            worker_id=self.worker_id, stripe_id=assign.stripe_id,
            substripes_done=len(D2A_SUBS), survivors_total=total_surv,
            effective_threshold=(assign.payload or {}).get(
                "min_match_threshold")))

    def stop(self):
        self._stop.set()
        try:
            self.fs.close()
        except Exception:  # noqa: BLE001
            pass


def _install_probes(coord, rec):
    """Wrap the coordinator's real seams to OBSERVE (never alter) the lifecycle:
    every stage_inline_shard result, every finalize CompletionCheck, and every
    retry-matrix routing. Each wrapper delegates to the real bound method."""
    real_stage = coord.stage_inline_shard
    real_final = coord.finalize_stripe
    real_matrix = coord.handle_stripe_failure
    lock = threading.Lock()

    def stage(run_id, stripe_id, attempt, sub_index, *a, **kw):
        res = real_stage(run_id, stripe_id, attempt, sub_index, *a, **kw)
        with lock:
            rec["stage"].append({
                "sub_index": sub_index, "status": res.get("status"),
                "exists": os.path.isfile(res.get("staged_path") or "")})
        return res

    def final(*a, **kw):
        chk = real_final(*a, **kw)
        with lock:
            rec["checks"].append(chk)
        return chk

    def matrix(*a, **kw):
        retryable = kw.get("retryable", a[2] if len(a) > 2 else None)
        out = real_matrix(*a, **kw)
        with lock:
            rec["matrix"].append({"retryable": retryable, "out": out})
        return out

    coord.stage_inline_shard = stage
    coord.finalize_stripe = final
    coord.handle_stripe_failure = matrix


def _drive_overlap_trial(tmp, sink):
    """Run a REAL trial to a terminal state through run_trial_miner's default
    serve path (the FULL real serve/dispatch lifecycle, so eligible_provider is
    not None and the definitive reconciliation failure can route through the
    matrix), driving the misbehaving _OverlapWorker. Returns (holder, worker)."""
    ds = os.path.join(tmp, "dataset.json")
    with open(ds, "w") as f:
        f.write('[{"draw":1},{"draw":2},{"draw":3}]')

    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(8)
    port = lsock.getsockname()[1]

    holder = {"stage": [], "checks": [], "matrix": []}

    def _serve(coord, context):
        holder["coord"] = coord
        holder["run_id"] = context["run_id"]
        _install_probes(coord, holder)
        return coord.serve_trial(context)

    def run():
        try:
            holder["result"] = run_trial_miner(
                "run-d2a", None, 7, "java_lcg", [1, 2, 3], D2A_STRIPE_SEEDS,
                0.40, 0.45,
                False,                                  # test_both_modes=False
                ds, worker_pool_size=1,
                staging_dir=os.path.join(tmp, "stg"), phase5_sink=sink,
                listen_sock=lsock,
                # config caps == the worker's advertised caps (else quarantine);
                # java_lcg/cuda effective cap = 10 -> expected_substripes = 3.
                seed_cap_nvidia=D2A_CAP, seed_cap_amd=D2A_CAP,
                seed_cap_nvidia_hybrid=D2A_CAP, seed_cap_amd_hybrid=D2A_CAP,
                skip_min=1, skip_max=9, window_size=5, offset=2,
                sessions=["midday", "evening"], serve_timeout=_SERVE_TIMEOUT,
                _serve=_serve)
        except Exception:  # noqa: BLE001
            holder["err"] = traceback.format_exc()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    w = _OverlapWorker("127.0.0.1", port)
    try:
        w.connect_register()
        w.start_loop()
        t.join(timeout=_JOIN_TIMEOUT)
        assert not t.is_alive(), "serve_trial did not terminate in time"
        assert "err" not in holder, holder.get("err")
        assert w.err is None, w.err
        return holder, w
    finally:
        w.stop()
        try:
            lsock.close()
        except Exception:  # noqa: BLE001
            pass


def d2a_real_producer_rejects_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _CountingAssemblingSink()
        holder, worker = _drive_overlap_trial(tmp, sink)
        coord = holder["coord"]
        run_id = holder["run_id"]
        result = holder["result"]

        # the worker really did dispatch exactly the phase-1 java_lcg assignment
        assert worker.assigns_received, "worker received no assignment"
        assert all(ph == 1 and fam == "java_lcg"
                   for _sid, ph, fam in worker.assigns_received), \
            worker.assigns_received

        # (1) all three shard payloads staged + hash-verified INDIVIDUALLY.
        assert len(holder["stage"]) == 3, holder["stage"]
        assert {s["sub_index"] for s in holder["stage"]} == {0, 1, 2}, holder["stage"]
        for s in holder["stage"]:
            assert s["status"] == "verified", s
            assert s["exists"] is True, s

        # (2a) evaluate_stripe_completion — a DETERMINISTIC, race-free direct
        #      reconstruction over the fixture with all shards verified: coverage
        #      is the PROVABLY ONLY red predicate.
        synth_stripe = {
            "seed_start": 0, "seed_count": D2A_STRIPE_SEEDS,
            "expected_substripes": len(D2A_SUBS),
            "substripes_done": len(D2A_SUBS),
            "stripe_complete_seen": True,
            "survivors_total": D2A_SURVIVORS_TOTAL,
        }
        synth_shards = [
            {"sub_index": si, "seed_start": ss, "seed_count": sc,
             "survivor_count": len(sv), "staging_status": SH_VERIFIED}
            for (si, ss, sc, sv) in D2A_SUBS]
        d = evaluate_stripe_completion(synth_stripe, synth_shards)
        assert d.substripes_match is True, d
        assert d.seed_sum_match is True, d          # 10+10+10 == 30
        assert d.survivor_sum_match is True, d      # 2+2+1 == 5
        assert d.coverage_ok is False, d            # overlap @9 / gap @19
        assert d.all_verified is True, d
        assert d.is_complete is False and d.reconciled is False, d
        assert any("coverage" in r and "gap or overlap" in r for r in d.reasons), \
            d.reasons
        # coverage is the SOLE failing predicate (the other three accounting
        # invariants hold and every shard is verified).
        assert (d.substripes_match and d.seed_sum_match and d.survivor_sum_match
                and d.all_verified and not d.coverage_ok), d

        # (2b) the REAL run actually routed on that predicate: at least one live
        #      finalize CompletionCheck had every expected shard present but did
        #      NOT reconcile, with coverage the failing accounting invariant.
        red = [c for c in holder["checks"]
               if c.substripes_match and not c.reconciled]
        assert red, "the definitive reconciliation-failure check never occurred"
        for c in red:
            assert c.seed_sum_match is True, c
            assert c.survivor_sum_match is True, c
            assert c.coverage_ok is False, c
            assert any("coverage" in r and "gap or overlap" in r
                       for r in c.reasons), c.reasons

        # (3) the attempt NEVER reached Phase5Sink.publish_shard.
        assert sink.publish_calls == 0, sink.publish_calls
        assert sink.commit_calls == 0, sink.commit_calls
        assert coord.enqueued == [], coord.enqueued
        assert result["manifests"] == [], result["manifests"]
        assert sink._runs == {}, "no Phase-5 run state may accumulate"

        # (4) the failure routed through the real matrix EXACTLY ONCE, retryable,
        #     constant-phase -> fail_trial.
        assert len(holder["matrix"]) == 1, holder["matrix"]
        m = holder["matrix"][0]
        assert m["retryable"] is True, m
        assert m["out"]["action"] == "fail_trial", m
        assert m["out"]["reason"] == "constant_phase", m

        # (5) final trial state aborted; sink abort discharged EXACTLY ONCE.
        assert result["state"] == "aborted", result["state"]
        assert result["committed"] is False, result
        trial = coord.ledger.get_trial(run_id)
        assert trial["state"] == "aborted", trial["state"]
        assert trial["abort_cleanup_status"] == "done", trial
        assert sink.abort_calls == 1, sink.abort_calls
        assert sink.abort_event_ids == [f"{run_id}:abort"], sink.abort_event_ids

        # (6) no canonical assembly anywhere.
        assert sink.get_assembly(run_id) is None, "no assembly may exist"


# ===========================================================================
# D2-B — DIRECT SINK INVARIANT-BREAK PROBE (Phase-5 defense in depth)
# ===========================================================================
# The post-D1.0 producer CANNOT legitimately emit this input (§0); the probe
# deliberately bypasses the upstream reconciliation barrier to prove the SECOND
# layer fails closed.

_SORT_KEY_FIELDS = ("workflow_phase", "stripe_id", "sub_index", "attempt",
                    "event_id")


def _sort_key(manifest):
    """The engine's insertion order key (range_miner_npz_writer.assemble_trial):
    (workflow_phase, stripe_id, sub_index, attempt, event_id)."""
    return (int(manifest["workflow_phase"]), str(manifest.get("stripe_id")),
            int(manifest.get("sub_index", 0)), int(manifest.get("attempt", 0)),
            str(manifest.get("event_id")))


class _DirectSinkProbe(AssemblingPhase5Sink):
    """DIRECT SINK INVARIANT-BREAK PROBE observing sink. A thin subclass whose
    commit_trial calls super().commit_trial(), captures ONLY the FIRST
    DirectionalDuplicateError (stored under a lock so the gate reads a stable
    reference), and re-raises it UNCHANGED — the lifecycle is not altered."""

    def __init__(self):
        super().__init__()
        self.captured = None
        self.commit_calls = 0
        self._probe_lock = threading.Lock()

    def commit_trial(self, event):
        with self._probe_lock:
            self.commit_calls += 1
        try:
            return super().commit_trial(event)
        except DirectionalDuplicateError as e:
            with self._probe_lock:
                if self.captured is None:
                    self.captured = e
            raise


def _build_d2b_manifests(tmp):
    """Generate a complete, individually-valid {1,2} manifest set, then repoint
    the phase-1 sub_index-1 shard at an overlapping payload so exactly ONE seed
    (5) is shared inside phase-1 forward/constant, at a DIFFERENT match rate.

    Returns (manifests, first_shard, dup_shard)."""
    gen = _mk_coord(tmp, AssemblingPhase5Sink(), "d2b_gen.db")
    pops = {
        1: {0: [(5, 0.90)],   1: [(15, 0.80)]},     # forward / constant
        2: {0: [(5, 0.50)],   1: [(16, 0.40)]},     # reverse / constant
    }
    manifests = _gen_manifests(gen, "d2b", (1, 2), pops)
    assert len(manifests) == 4, manifests

    first = next(m for m in manifests
                 if int(m["workflow_phase"]) == 1 and m["sub_index"] == 0)
    p1_sub1 = next(m for m in manifests
                   if int(m["workflow_phase"]) == 1 and m["sub_index"] == 1)

    # The overlapping payload: declared range [5, 15) contains the shared seed 5
    # (at a DIFFERENT rate, 0.11) plus its own seed 12 — each duplicate occurrence
    # inside its own spool's declared range (permitted only because this probe
    # bypasses reconciliation). stripe_id/sub_index match the manifest so §5.3's
    # identity checks pass and the DUPLICATE invariant is what fires.
    dup_payload = {
        "schema_version": "s172_substripe_v1",
        "stripe_id": p1_sub1["stripe_id"],
        "sub_index": p1_sub1["sub_index"],
        "seed_start": 5,
        "seed_count": 10,
        "survivors": [[5, 0.11, None, [2]], [12, 0.30, None, [3]]],
    }
    dup = _repoint(p1_sub1, tmp, dup_payload, "d2b_dup")

    manifests = [dup if m["event_id"] == p1_sub1["event_id"] else m
                 for m in manifests]

    # Determinism pin: the intended-FIRST shard sorts strictly before the dup
    # under the EXACT engine key (guards against a future sort-key change silently
    # swapping first/dup provenance).
    assert _sort_key(first) < _sort_key(dup), (_sort_key(first), _sort_key(dup))
    assert first["event_id"] != dup["event_id"], "distinct event_ids required"
    assert (first["stripe_id"], first["sub_index"]) != \
           (dup["stripe_id"], dup["sub_index"]), "distinct logical slots required"
    return manifests, first, dup


def d2b_direct_sink_invariant_break_probe():
    """DIRECT SINK INVARIANT-BREAK PROBE."""
    with tempfile.TemporaryDirectory() as tmp:
        manifests, first, dup = _build_d2b_manifests(tmp)

        sink = _DirectSinkProbe()
        # Deliver through the PUBLIC publish surface (never sink internals; never
        # assemble_trial directly — that is D1.1's G14, already green).
        for m in manifests:
            sink.publish_shard(m)

        # A REAL coordinator drives the commit; run_id trial is running.
        co = _mk_coord(tmp, sink, "d2b_commit.db")
        co.ledger.create_trial("d2b", CTX["trial_number"], now=100.0)
        ev = co.commit_trial("d2b", now=200.0)

        # the observer did NOT alter the lifecycle
        assert ev.get("delivery") == "failed", ev
        assert ev["event_id"] == "d2b:commit", ev
        assert ev["run_id"] == "d2b", ev

        e = sink.captured
        assert e is not None, "no DirectionalDuplicateError was captured"
        assert type(e) is DirectionalDuplicateError, type(e)
        assert sink.commit_calls == 1, sink.commit_calls

        # ALL 13 structured attributes — asserted directly, never via message text.
        assert e.run_id == "d2b", e.run_id
        assert e.workflow_phase == 1, e.workflow_phase
        assert e.direction == "forward", e.direction
        assert e.skip_mode == "constant", e.skip_mode
        assert e.seed == 5, e.seed
        assert e.first_stripe == first["stripe_id"], e.first_stripe
        assert e.first_sub_index == 0, e.first_sub_index
        assert e.first_attempt == 0, e.first_attempt
        assert e.first_match_rate == 0.90, e.first_match_rate
        assert e.dup_stripe == dup["stripe_id"], e.dup_stripe
        assert e.dup_sub_index == 1, e.dup_sub_index
        assert e.dup_attempt == 0, e.dup_attempt
        assert e.dup_match_rate == 0.11, e.dup_match_rate
        for attr in ("run_id", "workflow_phase", "direction", "skip_mode", "seed",
                     "first_stripe", "first_sub_index", "first_attempt",
                     "first_match_rate", "dup_stripe", "dup_sub_index",
                     "dup_attempt", "dup_match_rate"):
            assert getattr(e, attr) is not None, f"{attr} must be populated"

        # State: no assembly installed, no consumed commit marker, manifests
        # retained (same-instance retry per §4.0/§4.3), staged files intact.
        assert sink.get_assembly("d2b") is None, "no assembly may install"
        state = sink._runs.get("d2b")
        assert state is not None and state.result is None, "no result installed"
        assert state.consumed_commits == set(), "no consumed commit marker"
        assert len(state.manifests) == 4, "accumulated manifests must be retained"
        for m in manifests:
            assert os.path.isfile(m["local_spool_path"]), \
                f"staged file must NOT be deleted merely because delivery failed: {m}"

        # No NPZ was written anywhere, and no assembly ever carried a path (the
        # load-bearing assertion TODAY: no installation, no silent dedup).
        for root, _dirs, files in os.walk(tmp):
            for fn in files:
                assert not fn.endswith(".npz"), os.path.join(root, fn)

        # the dup is NEVER resolved by max/keep-first at this boundary
        assert not isinstance(e, ManifestReplayConflict)


# ===========================================================================
# §5 — Negative controls: the invariant is SCOPED, not global
# ===========================================================================
def nc1_same_seed_across_directions_is_legitimate():
    """The SAME seed in P1 forward AND P2 reverse is legitimate — it lands in both
    directional maps AND appears in bidirectional_constant (the invariant is keyed
    on (direction, skip_mode); there is no cross-direction leak)."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = AssemblingPhase5Sink()
        coord = _mk_coord(tmp, sink, "nc1.db")
        pops = {1: {0: [(5, 0.90)], 1: [(15, 0.80)]},
                2: {0: [(5, 0.50)], 1: [(16, 0.40)]}}
        _gen_manifests(coord, "nc1", (1, 2), pops)
        ev = coord.commit_trial("nc1", now=200.0)
        assert ev.get("delivery") == "done", ev
        a = sink.get_assembly("nc1")
        assert isinstance(a, MinerTrialAssembly), a
        assert 5 in a.forward_map_constant, a.forward_map_constant
        assert 5 in a.reverse_map_constant, a.reverse_map_constant
        assert 5 in a.bidirectional_constant, a.bidirectional_constant
        assert a.forward_map_constant[5] == 0.90
        assert a.reverse_map_constant[5] == 0.50


def nc2_two_disjoint_shards_one_population_commits():
    """Two DISJOINT shards in one directional population: commit succeeds."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = AssemblingPhase5Sink()
        coord = _mk_coord(tmp, sink, "nc2.db")
        # forward/constant carries two disjoint sub-stripes (seeds 5 and 15).
        pops = {1: {0: [(5, 0.90)], 1: [(15, 0.80)]},
                2: {0: [(5, 0.50)], 1: []}}
        _gen_manifests(coord, "nc2", (1, 2), pops)
        ev = coord.commit_trial("nc2", now=200.0)
        assert ev.get("delivery") == "done", ev
        a = sink.get_assembly("nc2")
        assert set(a.forward_map_constant) == {5, 15}, a.forward_map_constant
        assert a.directional_counts["forward_constant"] == 2, a.directional_counts


def nc3_same_seed_constant_and_variable_commits():
    """The SAME seed in constant (P1) AND variable (P3), four-phase fixture: not a
    directional duplicate (keyed on skip_mode too) — commit succeeds."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = AssemblingPhase5Sink()
        coord = _mk_coord(tmp, sink, "nc3.db")
        pops = {
            1: {0: [(5, 0.90)], 1: []},     # forward / constant
            2: {0: [(5, 0.50)], 1: []},     # reverse / constant
            3: {0: [(5, 0.95)], 1: []},     # forward / variable
            4: {0: [(5, 0.55)], 1: []},     # reverse / variable
        }
        _gen_manifests(coord, "nc3", (1, 2, 3, 4), pops)
        ev = coord.commit_trial("nc3", now=200.0)
        assert ev.get("delivery") == "done", ev
        a = sink.get_assembly("nc3")
        assert 5 in a.forward_map_constant, a.forward_map_constant
        assert 5 in a.forward_map_variable, a.forward_map_variable
        assert a.forward_map_constant[5] == 0.90
        assert a.forward_map_variable[5] == 0.95


def nc4_same_slot_different_event_id_is_replay_conflict():
    """A DIFFERENT event_id claiming the SAME (stripe_id, sub_index) slot raises
    ManifestReplayConflict BEFORE assembly (intentional overlap with D1.1's G7;
    retained per Team Beta so D2 reads standalone)."""
    with tempfile.TemporaryDirectory() as tmp:
        gen = _mk_coord(tmp, AssemblingPhase5Sink(), "nc4.db")
        manifests = _gen_manifests(gen, "nc4", (1, 2),
                                   {1: {0: [(5, 0.90)], 1: [(15, 0.80)]},
                                    2: {0: [(5, 0.50)], 1: [(16, 0.40)]}})
        original = manifests[0]

        sink = AssemblingPhase5Sink()
        sink.publish_shard(original)
        clash = copy.deepcopy(original)
        clash["event_id"] = original["event_id"] + "__other"   # NEW event id...
        # ...same (stripe_id, sub_index) slot.
        assert (clash["stripe_id"], clash["sub_index"]) == \
               (original["stripe_id"], original["sub_index"])
        e = _raises(ManifestReplayConflict, sink.publish_shard, clash)
        assert "logical shard slot" in str(e), str(e)
        # and it is NOT a directional duplicate (it never reached assembly)
        assert not isinstance(e, DirectionalDuplicateError)


# ===========================================================================
# §7 — blocking non-regression
# ===========================================================================
def _run_suite(rel_path, expect_substr, timeout=1800):
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


def nonregression_suites():
    _run_suite("tests/test_s172_phase5_d1_engine.py", "18/18 D1.1 gate checks green")
    _run_suite("tests/test_s172_phase5_d1_workflow.py", "8/8 D1.0 gate checks green")
    _run_suite("tests/test_s172_phase5_d0.py", "12/12 D0 gate checks green")
    _run_suite("tests/test_s172_phase4_coordinator.py", "63/63 checks green")
    _run_suite("tests/test_s172_phase3_worker.py", "17/17 gates green")


# ===========================================================================
# dispatch
# ===========================================================================
GATES = {
    "d2a": ("D2-A: REAL producer overlap rejection (misbehaving worker)",
            d2a_real_producer_rejects_overlap),
    "d2b": ("D2-B: DIRECT SINK INVARIANT-BREAK PROBE (fail-closed assembly)",
            d2b_direct_sink_invariant_break_probe),
    "nc1": ("NC1: same seed across directions is legitimate",
            nc1_same_seed_across_directions_is_legitimate),
    "nc2": ("NC2: two disjoint shards in one population commit",
            nc2_two_disjoint_shards_one_population_commits),
    "nc3": ("NC3: same seed constant AND variable commits",
            nc3_same_seed_constant_and_variable_commits),
    "nc4": ("NC4: same slot / different event_id -> ManifestReplayConflict",
            nc4_same_slot_different_event_id_is_replay_conflict),
}


def main():
    print("=" * 74)
    print("S172 Phase-5 D2 — directional uniqueness at BOTH enforcement layers")
    print("=" * 74)
    for _key, (name, fn) in GATES.items():
        _check(name, fn)
    _check("NR:  D1.1 18/18, D1.0 8/8, D0 12/12, Phase 4 63/63, Phase 3 17/17",
           nonregression_suites)

    print("=" * 74)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} D2 gate checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All D2 gate checks green — directional uniqueness is enforced at BOTH "
          "the producer overlap-rejection layer and the Phase-5 fail-closed "
          "assembly layer (pending Team Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--gate":
        key = sys.argv[2]
        if key not in GATES:
            print(f"unknown gate {key!r}; choices: {sorted(GATES)}")
            sys.exit(2)
        name, fn = GATES[key]
        try:
            fn()
            print(f"[{_PASS}] {name}")
            sys.exit(0)
        except Exception:  # noqa: BLE001
            print(f"[{_FAIL}] {name}")
            traceback.print_exc()
            sys.exit(1)
    sys.exit(main())
