#!/usr/bin/env python3
"""
test_s172_phase5_d0.py — S172 Phase-5 Deliverable D0 acceptance harness.

D0 is the Phase-4 metadata-seam + durable-context correction (docs/
CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5.md §D0). It makes every ShardReadyManifest
published to Phase 5 carry a COMPLETE, IMMUTABLE, RESTART-DURABLE trial_metadata
projection — reconstructed from the durable ledger (the write-once `trial_context`
row + each stripe's own persisted phase/family_name), never from spool contents —
so `trial_metadata` is no longer always `{}`.

Seven D0 gate checks (§ "Gate D0") plus five Beta-correction gates (B1/B2 = REV2,
B3/B4 = REV3, B4-corrected + B5 = REV4), each constructed to FAIL on the wrong
behavior (i.e. it would fail against the pre-D0 `{}`-emitting code, not just a stub):

  1. every published shard carries all mandatory metadata fields (non-empty);
  2. metadata identical where trial-global, correct where phase-specific
     (P1 forward/constant, P2 reverse/constant, P3 forward/variable,
     P4 reverse/variable, with matching family_name);
  3. forward/reverse and constant/variable are EXPLICIT strings, not inferred
     from the numeric phase by the consumer;
  4. metadata cannot change after trial creation (mutate the source post-creation
     -> published manifests unaffected) + a restart-recovery reconstruction
     produces an IDENTICAL manifest;
  5. retry attempt 1 carries the same SEMANTIC context as attempt 0
     (same phase/family/direction/mode/thresholds);
  6. a manifest missing any mandatory field FAILS CLOSED before Phase 5
     publication (no `{}` leak) — at persist time AND at publish time;
  7. commit / abort / acknowledgement behavior is UNCHANGED (terminal
     exclusivity + ack paths still hold).

The harness exercises the REAL lifecycle — real staged spool files on disk, real
`Phase5Sink.publish_shard` calls from the real coordinator publish surface, real
attempt bumps through the ledger. Nothing under test is monkeypatched away.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 tests/test_s172_phase5_d0.py
Exit 0 = all green. Exit 1 = a check failed (DO NOT COMMIT).
"""
import hashlib
import os
import socket
import sys
import tempfile
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from miner.range_miner_coordinator import (  # noqa: E402
    ST_FAILED,
    CoordinatorConfig,
    MinerLedger,
    MinerMetadataError,
    NodeConfig,
    Phase5Sink,
    RangeMinerCoordinator,
    TrialAborted,
    MANDATORY_MANIFEST_METADATA,
    build_trial_context_from_serve,
    derive_trial_metadata,
    run_trial_miner,
    workflow_phase_semantics,
)
from miner.range_miner_worker import build_substripe_payload_bytes  # noqa: E402

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_results = []

# All four §6.8 java-family variants so a single worker can serve every phase.
VARIANTS = ["java_lcg", "java_lcg_reverse", "java_lcg_hybrid",
            "java_lcg_hybrid_reverse"]
CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
SPOOL_ROOT = "/var/spool/miner"

# §6.8 phase -> (family suffix, direction, skip_mode, prng_type suffix)
PHASE_TABLE = {
    1: ("java_lcg",                "forward", "constant", "java_lcg"),
    2: ("java_lcg_reverse",        "reverse", "constant", "java_lcg"),
    3: ("java_lcg_hybrid",         "forward", "variable", "java_lcg_hybrid"),
    4: ("java_lcg_hybrid_reverse", "reverse", "variable", "java_lcg_hybrid"),
}


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:  # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


class _StubSink(Phase5Sink):
    """Records the real coordinator publish surface: published manifests, commit
    and abort events. This is the actual Phase-5 interface the coordinator drives
    (not a monkeypatch of the code under test)."""

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
                      "supported_variants": list(VARIANTS)},
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


def _stage_attempt(coord, run_id, sid, conn, attempt, seed_start=0, seed_count=30,
                   now=100.0):
    """Record + stage a real inline sub-stripe (one sub covering the stripe) for the
    given attempt so its shard is verified on disk. Returns (size, sha)."""
    survivors = [[seed_start, 0.91, None, [1]], [seed_start + 5, 0.83, None, [2, 3]]]
    _, pb = build_substripe_payload_bytes(sid, 0, seed_start, seed_count, survivors)
    size, sha = len(pb), hashlib.sha256(pb).hexdigest()
    coord.ledger.record_substripe_result(
        run_id, sid, attempt, 0, conn.worker_id, seed_start, seed_count,
        len(survivors), remote_spool_path=None, size_bytes=size, sha256=sha, now=now)
    res = coord.stage_inline_shard(run_id, sid, attempt, 0, seed_start, seed_count,
                                   survivors, size, sha, now=now)
    assert res["status"] == "verified", res
    staged = res["staged_path"]
    assert os.path.isfile(staged), "real staged spool file must exist on disk"
    return size, sha


def _run_phase_to_publish(coord, run_id, sink, phase, prefix, now=100.0):
    """Assign a stripe for the §6.8 `phase`, stage a real inline shard, complete +
    finalize -> the attempt is published through the REAL publish surface. Returns
    the published manifest for that stripe."""
    family = PHASE_TABLE[phase][0]
    conn = _register(coord)
    recs = coord.assign_stripes(run_id, family, phase, 30, [conn],
                                stripe_prefix=prefix, now=now)
    sid = recs[0]["stripe_id"]
    before = len(sink.published)
    _stage_attempt(coord, run_id, sid, conn, attempt=0, now=now)
    coord.ledger.record_stripe_complete(run_id, sid, 0, conn.worker_id, 1, 2)
    coord.finalize_stripe(run_id, sid, now=now)
    assert len(sink.published) == before + 1, "exactly one manifest must publish"
    return sink.published[-1], sid


# ---------------------------------------------------------------------------
# GATE 1 — every published shard carries all mandatory metadata (non-empty)
# ---------------------------------------------------------------------------
def gate1_mandatory_fields_present():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord(tmp, sink)
        run_id = "run1"
        coord.ledger.set_trial_context(run_id, _full_ctx())
        m, _ = _run_phase_to_publish(coord, run_id, sink, phase=1, prefix=run_id + "__p1")

        meta = m["trial_metadata"]
        # The pre-D0 code published `{}` here — this assertion fails against it.
        assert meta != {}, "trial_metadata must not be empty (pre-D0 regression)"
        for k in MANDATORY_MANIFEST_METADATA:
            assert k in meta, f"missing mandatory field {k!r}"
            assert meta[k] is not None, f"mandatory field {k!r} is None"
            if isinstance(meta[k], str):
                assert meta[k].strip() != "", f"identity field {k!r} is empty"
        # provenance retained (non-NPZ), both in trial_metadata and top-level
        for k in ("dataset_sha256", "residue_sha256"):
            assert meta.get(k), f"provenance {k!r} missing from trial_metadata"
            assert m.get(k) == meta[k], f"provenance {k!r} not lifted to manifest"
        # concrete trial-global values propagated from the ledger (not defaults)
        assert meta["window_size"] == 5 and meta["offset"] == 2
        assert meta["skip_min"] == 1 and meta["skip_max"] == 9
        assert meta["sessions"] == ["midday", "evening"]
        assert meta["trial_number"] == 7
        assert meta["forward_threshold"] == 0.40 and meta["reverse_threshold"] == 0.45


# ---------------------------------------------------------------------------
# GATE 2 — trial-global identical across P1..P4; phase-specific correct per phase
# ---------------------------------------------------------------------------
def gate2_identical_global_correct_phase():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord(tmp, sink)
        run_id = "run2"
        coord.ledger.set_trial_context(run_id, _full_ctx())

        metas = {}
        for phase in (1, 2, 3, 4):
            m, _ = _run_phase_to_publish(coord, run_id, sink, phase,
                                         prefix=f"{run_id}__p{phase}")
            metas[phase] = m["trial_metadata"]

        # trial-global fields IDENTICAL across every phase's manifest
        global_fields = ("trial_number", "window_size", "offset", "sessions",
                         "skip_min", "skip_max", "prng_base", "forward_threshold",
                         "reverse_threshold", "dataset_sha256", "residue_sha256")
        for gf in global_fields:
            vals = {phase: metas[phase][gf] for phase in (1, 2, 3, 4)}
            assert len(set(map(repr, vals.values()))) == 1, \
                f"trial-global {gf!r} differs across phases: {vals}"

        # phase-specific fields CORRECT per §6.8
        for phase, (family, direction, skip_mode, prng_type) in PHASE_TABLE.items():
            meta = metas[phase]
            assert meta["workflow_phase"] == phase, (phase, meta["workflow_phase"])
            assert meta["family_name"] == family, (phase, meta["family_name"])
            assert meta["direction"] == direction, (phase, meta["direction"])
            assert meta["skip_mode"] == skip_mode, (phase, meta["skip_mode"])
            assert meta["prng_type"] == prng_type, (phase, meta["prng_type"])
            expect_thresh = 0.40 if direction == "forward" else 0.45
            assert meta["threshold_used"] == expect_thresh, (phase, meta)


# ---------------------------------------------------------------------------
# GATE 3 — direction/skip_mode are EXPLICIT strings, not phase-number inference
# ---------------------------------------------------------------------------
def gate3_explicit_string_identities():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord(tmp, sink)
        run_id = "run3"
        coord.ledger.set_trial_context(run_id, _full_ctx())
        m, _ = _run_phase_to_publish(coord, run_id, sink, phase=4,
                                     prefix=run_id + "__p4")
        meta = m["trial_metadata"]
        # A consumer reads the identity as strings and needs NO numeric-phase logic.
        assert isinstance(meta["direction"], str) and meta["direction"] == "reverse"
        assert isinstance(meta["skip_mode"], str) and meta["skip_mode"] == "variable"
        assert meta["direction"] in ("forward", "reverse")
        assert meta["skip_mode"] in ("constant", "variable")
        # The resolver itself returns explicit strings from the §6.8 table and
        # hard-fails on an unknown phase (fail-closed, not a silent inference).
        assert workflow_phase_semantics(1) == ("forward", "constant")
        assert workflow_phase_semantics(2) == ("reverse", "constant")
        assert workflow_phase_semantics(3) == ("forward", "variable")
        assert workflow_phase_semantics(4) == ("reverse", "variable")
        raised = False
        try:
            workflow_phase_semantics(9)
        except MinerMetadataError:
            raised = True
        assert raised, "unknown phase must hard-fail, never be inferred"


# ---------------------------------------------------------------------------
# GATE 4 — immutability post-creation + restart-recovery identical reconstruction
# ---------------------------------------------------------------------------
def gate4_immutable_and_restart_durable():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        db = os.path.join(tmp, "durable.db")
        coord = _coord(tmp, sink, dbname="durable.db")
        run_id = "run4"
        coord.ledger.set_trial_context(run_id, _full_ctx())
        m0, sid = _run_phase_to_publish(coord, run_id, sink, phase=3,
                                        prefix=run_id + "__p3")
        meta0 = m0["trial_metadata"]

        # (a) IMMUTABILITY (Blocker 1): a second set_trial_context with MUTATED values
        # is now a fail-closed CONFLICT — it RAISES MinerMetadataError and leaves the
        # original row unchanged (compare-and-insert, no silent divergence). An
        # IDENTICAL replay is still an idempotent no-op (no raise).
        raised = False
        try:
            coord.ledger.set_trial_context(
                run_id, _full_ctx(window_size=999, offset=888, skip_min=777,
                                  trial_number=666, prng_base="pcg32"))
        except MinerMetadataError:
            raised = True
        assert raised, "a conflicting immutable trial_context must raise (Blocker 1)"
        coord.ledger.set_trial_context(run_id, _full_ctx())   # identical replay: no-op
        ctx_after = coord.ledger.get_trial_context(run_id)
        assert ctx_after["window_size"] == 5 and ctx_after["offset"] == 2, \
            "trial_context must be immutable (original row unchanged after conflict)"
        assert ctx_after["prng_base"] == "java_lcg" and ctx_after["skip_min"] == 1
        assert ctx_after["trial_number"] == 7

        # (b) RESTART RECOVERY: a fresh coordinator over the SAME db reconstructs a
        # byte-identical trial_metadata from the durable ledger alone (context row +
        # persisted stripe phase/family) — no in-memory state carried over.
        del coord
        sink2 = _StubSink()
        ledger2 = MinerLedger(db)
        coord2 = RangeMinerCoordinator(
            CoordinatorConfig(staging_dir=os.path.join(tmp, "staging")),
            ledger2, phase5_sink=sink2)
        stripe = coord2.ledger.get_stripe(run_id, sid)
        ctx = coord2.ledger.get_trial_context(run_id)
        assert ctx is not None, "trial_context must survive restart"
        meta_recovered = derive_trial_metadata(ctx, stripe)
        for k in list(MANDATORY_MANIFEST_METADATA) + ["dataset_sha256",
                                                       "residue_sha256",
                                                       "threshold_used"]:
            assert meta_recovered[k] == meta0[k], \
                f"restart-reconstructed {k!r} differs: {meta_recovered[k]!r} != {meta0[k]!r}"


# ---------------------------------------------------------------------------
# GATE 5 — retry attempt 1 carries the same SEMANTIC context as attempt 0
# ---------------------------------------------------------------------------
def gate5_retry_semantic_context_stable():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord(tmp, sink)
        run_id = "run5"
        coord.ledger.set_trial_context(run_id, _full_ctx())
        conn = _register(coord)
        recs = coord.assign_stripes(run_id, "java_lcg_hybrid", 3, 30, [conn],
                                    stripe_prefix=run_id + "__p3", now=100.0)
        sid = recs[0]["stripe_id"]

        # attempt 0 published via the real publish surface
        _stage_attempt(coord, run_id, sid, conn, attempt=0, now=100.0)
        m0 = coord.publish_attempt(run_id, sid, 0, now=100.0)[0]

        # real retry: fail the stripe, re-claim it as attempt 1, stage + publish
        coord.ledger.set_stripe_state(run_id, sid, ST_FAILED)
        assert coord.ledger.claim_stripe(run_id, sid, conn.worker_id, 1,
                                         expected_substripes=1,
                                         lease_expires_at=100.0 + 300), "reclaim failed"
        _stage_attempt(coord, run_id, sid, conn, attempt=1, now=101.0)
        m1 = coord.publish_attempt(run_id, sid, 1, now=101.0)[0]

        assert m0["attempt"] == 0 and m1["attempt"] == 1
        a, b = m0["trial_metadata"], m1["trial_metadata"]
        for k in ("workflow_phase", "family_name", "direction", "skip_mode",
                  "prng_type", "prng_base", "forward_threshold",
                  "reverse_threshold", "threshold_used", "window_size", "offset",
                  "skip_min", "skip_max", "sessions", "trial_number",
                  "dataset_sha256", "residue_sha256"):
            assert a[k] == b[k], f"retry changed semantic field {k!r}: {a[k]!r} != {b[k]!r}"
        assert b["direction"] == "forward" and b["skip_mode"] == "variable"


# ---------------------------------------------------------------------------
# GATE 6 — fail closed on missing mandatory metadata (no `{}` leak)
# ---------------------------------------------------------------------------
def gate6_fail_closed_no_empty_leak():
    with tempfile.TemporaryDirectory() as tmp:
        # (a) fail closed at PERSIST time: an incomplete trial-global context is
        # refused before any stripe work (so it can never surface a `{}` later).
        sink = _StubSink()
        coord = _coord(tmp, sink)
        bad_global = _full_ctx()
        del bad_global["skip_max"]
        raised = False
        try:
            coord.ledger.set_trial_context("bad", bad_global)
        except MinerMetadataError:
            raised = True
        assert raised, "incomplete trial-global context must fail closed at persist"
        assert coord.ledger.get_trial_context("bad") is None

        # (b) fail closed at PUBLISH time: a verified shard on a stripe whose
        # phase-specific identity is corrupted (empty family_name) must RAISE and
        # publish NOTHING — no `{}` reaches the sink.
        sink2 = _StubSink()
        coord2 = _coord(tmp, sink2, dbname="l2.db")
        run_id = "run6"
        coord2.ledger.set_trial_context(run_id, _full_ctx())
        conn = _register(coord2)
        recs = coord2.assign_stripes(run_id, "java_lcg", 1, 30, [conn],
                                     stripe_prefix=run_id + "__p1", now=100.0)
        sid = recs[0]["stripe_id"]
        _stage_attempt(coord2, run_id, sid, conn, attempt=0, now=100.0)
        # corrupt the stripe's persisted identity (producer/coverage defect analogue)
        with coord2.ledger._conn() as c:
            c.execute("UPDATE stripes SET family_name='' WHERE run_id=? AND stripe_id=?",
                      (run_id, sid))
            c.commit()
        raised = False
        try:
            coord2.publish_attempt(run_id, sid, 0, now=100.0)
        except MinerMetadataError:
            raised = True
        assert raised, "empty family_name must fail closed at publish"
        assert sink2.published == [], "no manifest (and no `{}`) may reach Phase 5"

        # negative control: with a VALID context + stripe, publish succeeds (proves
        # the gate is not merely always-raising).
        sink3 = _StubSink()
        coord3 = _coord(tmp, sink3, dbname="l3.db")
        coord3.ledger.set_trial_context("ok", _full_ctx())
        m, _ = _run_phase_to_publish(coord3, "ok", sink3, phase=1, prefix="ok__p1")
        assert m["trial_metadata"]["family_name"] == "java_lcg"


# ---------------------------------------------------------------------------
# GATE 7 — commit / abort / acknowledgement behavior UNCHANGED
# ---------------------------------------------------------------------------
def gate7_commit_abort_ack_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord(tmp, sink)
        run_id = "run7"
        coord.ledger.create_trial(run_id, 7, now=100.0)
        coord.ledger.set_trial_context(run_id, _full_ctx())
        m, sid = _run_phase_to_publish(coord, run_id, sink, phase=1,
                                       prefix=run_id + "__p1")

        # ack path unchanged: enqueued -> acked
        assert coord.ledger.get_shard(run_id, sid, 0, 0)["phase5_status"] == "enqueued"
        coord.ledger.mark_shard_acked(run_id, sid, 0, 0, now=100.0)
        assert coord.ledger.get_shard(run_id, sid, 0, 0)["phase5_status"] == "acked"

        # commit delivers a survivor-free event to the sink; carries only the
        # {event_type, run_id, event_id} contract (+ delivery bookkeeping).
        ev = coord.commit_trial(run_id, now=200.0)
        assert ev["event_type"] == "trial_commit" and ev["run_id"] == run_id
        assert len(sink.commits) == 1
        published_event = sink.commits[0]
        # The commit event carries only the {event_type, run_id, event_id} contract
        # (+ the coordinator's own delivery bookkeeping added to the same dict
        # afterward) — NEVER survivor maps / manifests / assembly. Assert the
        # contract keys are present and no survivor-bearing key leaked.
        assert {"event_type", "run_id", "event_id"} <= set(published_event), published_event
        assert set(published_event) <= {"event_type", "run_id", "event_id",
                                        "delivery", "duplicate", "error"}, \
            f"commit event must carry no survivor data: {published_event}"
        assert coord.ledger.get_trial(run_id)["state"] == "committed"

        # terminal exclusivity unchanged: a committed trial cannot be aborted, and a
        # duplicate commit is an idempotent no-op.
        assert coord.ledger.mark_trial_aborted(run_id, f"{run_id}:abort", now=201.0) is False
        assert coord.ledger.get_trial(run_id)["state"] == "committed"
        ev2 = coord.commit_trial(run_id, now=202.0)
        assert ev2.get("duplicate") is True
        assert len(sink.commits) == 1, "duplicate commit must not re-deliver"

        # and abort remains terminal-exclusive on a fresh trial (raises on commit)
        r2 = "run7b"
        coord.ledger.create_trial(r2, 8, now=100.0)
        assert coord.ledger.mark_trial_aborted(r2, f"{r2}:abort", now=100.0) is True
        raised = False
        try:
            coord.commit_trial(r2, now=101.0)
        except TrialAborted:
            raised = True
        assert raised, "commit after abort must raise (terminal exclusivity)"


# ---------------------------------------------------------------------------
# GATE B1 (Beta correction, Blocker 1) — set_trial_context compare-and-insert:
#   first insert succeeds; identical replay is an idempotent no-op; a one-field
#   mutation RAISES with the original row unchanged and ZERO stripes/manifests.
# ---------------------------------------------------------------------------
def gateB1_compare_and_insert_conflict():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord(tmp, sink)
        run_id = "runB1"

        # (1) FIRST INSERTION → succeeds, row present with the persisted values.
        coord.ledger.set_trial_context(run_id, _full_ctx())
        row = coord.ledger.get_trial_context(run_id)
        assert row is not None and row["window_size"] == 5 and row["prng_base"] == "java_lcg"

        # (2) IDENTICAL REPLAY → idempotent no-op (no raise; row unchanged). A
        # re-encoded-but-equal ctx (sessions rebuilt, thresholds as equal floats)
        # must still compare equal through the canonicalizer.
        coord.ledger.set_trial_context(
            run_id, _full_ctx(sessions=["midday", "evening"],
                              forward_threshold=0.40, reverse_threshold=0.45))
        row2 = coord.ledger.get_trial_context(run_id)
        assert row2 == row, f"identical replay must be a no-op: {row2} != {row}"

        # (3) ONE-FIELD MUTATION → raises MinerMetadataError; original row unchanged;
        # zero stripes assigned; zero manifests published. Test window_size,
        # dataset_sha256, and prng_base mutations (Beta's named minimum set).
        for field, bad in (("window_size", 40),
                           ("dataset_sha256", "e" * 64),
                           ("prng_base", "pcg32")):
            raised = False
            try:
                coord.ledger.set_trial_context(run_id, _full_ctx(**{field: bad}))
            except MinerMetadataError:
                raised = True
            assert raised, f"mutating {field!r} must raise a conflict (Blocker 1)"
            after = coord.ledger.get_trial_context(run_id)
            assert after == row, f"original row must be unchanged after {field!r} conflict"
        assert coord.ledger.all_stripes(run_id) == [], "zero stripes on conflict"
        assert sink.published == [], "zero manifests published on conflict"


# ---------------------------------------------------------------------------
# GATE B2 (Beta correction, Blocker 2) — the serve context-builder fails closed
#   on every missing mandatory field BEFORE stripe creation (no fallback), and
#   threshold_used maps to the direction-correct threshold.
# ---------------------------------------------------------------------------
def gateB2_no_fallback_fail_closed():
    DS, RES = "d" * 64, "r" * 64

    def _serve_ctx(**over):
        ctx = dict(trial_number=7, window_size=5, offset=2,
                   sessions=["midday", "evening"], skip_min=1, skip_max=9,
                   prng_base="java_lcg", forward_threshold=0.40,
                   reverse_threshold=0.45)
        ctx.update(over)
        return ctx

    # sanity: a complete serve context builds a full projection (proves the gate is
    # not merely always-raising).
    ok = build_trial_context_from_serve(_serve_ctx(), DS, RES)
    assert ok["prng_base"] == "java_lcg" and ok["skip_min"] == 1 and ok["skip_max"] == 9
    assert ok["forward_threshold"] == 0.40 and ok["reverse_threshold"] == 0.45
    assert ok["dataset_sha256"] == DS and ok["residue_sha256"] == RES

    # each of these must fail closed (MinerMetadataError) BEFORE any stripe creation,
    # with NO trial context inserted / stripe assigned / manifest published.
    cases = [
        ("missing prng_base",        {k: v for k, v in _serve_ctx().items() if k != "prng_base"}),
        ("empty prng_base",          _serve_ctx(prng_base="")),
        ("missing skip_min",         {k: v for k, v in _serve_ctx().items() if k != "skip_min"}),
        ("missing skip_max",         {k: v for k, v in _serve_ctx().items() if k != "skip_max"}),
        ("missing forward_threshold", {k: v for k, v in _serve_ctx().items() if k != "forward_threshold"}),
        ("missing reverse_threshold", {k: v for k, v in _serve_ctx().items() if k != "reverse_threshold"}),
    ]
    for label, bad_ctx in cases:
        with tempfile.TemporaryDirectory() as tmp:
            sink = _StubSink()
            coord = _coord(tmp, sink)
            run_id = "runB2"
            raised = False
            try:
                # the REAL serve seam: build the immutable projection, then persist it.
                projection = build_trial_context_from_serve(bad_ctx, DS, RES)
                coord.ledger.set_trial_context(run_id, projection)
            except MinerMetadataError:
                raised = True
            assert raised, f"{label} must fail closed (no fallback substitute)"
            assert coord.ledger.get_trial_context(run_id) is None, \
                f"{label}: no trial context may be inserted"
            assert coord.ledger.all_stripes(run_id) == [], f"{label}: no stripe assigned"
            assert sink.published == [], f"{label}: no manifest published"

    # threshold_used correctness: forward phases (1, 3) -> forward_threshold;
    # reverse phases (2, 4) -> reverse_threshold. Derived from the real projection.
    ctx = build_trial_context_from_serve(_serve_ctx(), DS, RES)
    for phase, (family, direction, _skip, _prng) in PHASE_TABLE.items():
        stripe = {"phase": phase, "family_name": family}
        meta = derive_trial_metadata(ctx, stripe)
        expect = 0.40 if direction == "forward" else 0.45
        assert meta["threshold_used"] == expect, \
            f"phase {phase} ({direction}): threshold_used {meta['threshold_used']} != {expect}"
        assert meta["threshold_used"] == (
            meta["forward_threshold"] if direction == "forward"
            else meta["reverse_threshold"])


# ---------------------------------------------------------------------------
# GATE B3 (Beta REV3, Blocker 1) — a COMPLETELY ABSENT trial_context row must
#   fail closed at the REAL publish surface. REV2 tested incomplete-context at
#   set_trial_context and a corrupted family_name AFTER a valid context existed —
#   never publication with the durable row absent, where the old
#   `if trial_ctx is not None else None` fallback leaked `trial_metadata: {}`.
# ---------------------------------------------------------------------------
def gateB3_missing_context_publish_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord(tmp, sink)
        run_id = "runB3"

        # DELIBERATELY no set_trial_context(run_id, ...): the durable trial_context
        # row is COMPLETELY ABSENT. Drive the REAL staging + publish path (mirror
        # _run_phase_to_publish) up to a verified shard on disk, then publish.
        conn = _register(coord)
        recs = coord.assign_stripes(run_id, "java_lcg", 1, 30, [conn],
                                    stripe_prefix=run_id + "__p1", now=100.0)
        sid = recs[0]["stripe_id"]
        _stage_attempt(coord, run_id, sid, conn, attempt=0, now=100.0)
        coord.ledger.record_stripe_complete(run_id, sid, 0, conn.worker_id, 1, 2)

        # publish MUST fail closed — no `{}` manifest reaches Phase 5. (Against the
        # pre-REV3 code this publishes `published_count 1, trial_metadata {}`, so this
        # gate FAILS on the old behavior, as required.)
        raised = False
        try:
            coord.publish_attempt(run_id, sid, 0, now=100.0)
        except MinerMetadataError:
            raised = True
        assert raised, "absent trial_context row must raise MinerMetadataError at publish"
        assert sink.published == [], "no manifest (and no `{}`) may reach Phase 5"
        assert coord.enqueued == [], "no shard may be recorded enqueued on fail-closed publish"
        # the shard must NOT be marked enqueued in the durable ledger either
        shard = coord.ledger.get_shard(run_id, sid, 0, 0)
        assert shard["phase5_status"] != "enqueued", \
            f"shard must not be enqueued after fail-closed publish: {shard['phase5_status']!r}"

        # negative control: with the durable context PRESENT, the same path publishes
        # exactly one complete manifest (proves the gate is not merely always-raising).
        sink_ok = _StubSink()
        coord_ok = _coord(tmp, sink_ok, dbname="b3ok.db")
        coord_ok.ledger.set_trial_context("b3ok", _full_ctx())
        m, _ = _run_phase_to_publish(coord_ok, "b3ok", sink_ok, phase=1, prefix="b3ok__p1")
        assert m["trial_metadata"] != {} and m["trial_metadata"]["family_name"] == "java_lcg"


# ---------------------------------------------------------------------------
# Shared driver for the entry-point fail-closed gates (B4/B5): omit exactly ONE
# mandatory field at the REAL run_trial_miner() entry point, supplying every OTHER
# mandatory field, and assert the guard fails closed (no context, no stripe, no
# manifest) BEFORE serving. Isolating one omission is what makes the gate prove the
# specific field — REV3's B4 omitted skip AND window_size AND offset, so once the
# latter two became mandatory the assertion passed for the wrong reason (Beta's
# "accidentally vacuous" catch).
# ---------------------------------------------------------------------------
# The full mandatory-metadata arg set for a run_trial_miner entry-point call.
_MANDATORY_KW = dict(skip_min=0, skip_max=0, window_size=5, offset=0)


def _assert_entrypoint_fails_closed(tmp, ds, label, kw):
    """Call run_trial_miner with `kw` (one mandatory field omitted) and assert it
    fails closed synchronously — before the socket bind / any stripe / any publish.

    The serve path is BOUNDED (pre-bound ephemeral loopback socket + 0.25s serve /
    read deadlines, single worker slot) so that if the window_size/offset fabrication
    regression is reintroduced the fail-closed guard no longer fires and this helper
    reaches serve(): the bound path then TIMES OUT quickly and the `raised` assertion
    fails cleanly, instead of hanging forever (Beta reproduced exit 124 on the
    unbounded call). Correct code still raises MinerMetadataError before serving, so
    the socket is never actually served."""
    stg = os.path.join(tmp, "stg_" + label)
    sink = _StubSink()
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(8)
    raised = False
    try:
        run_trial_miner("runFC", None, 5, "java_lcg", [1, 2, 3], 30, 0.25, 0.25,
                        False, ds, staging_dir=stg, phase5_sink=sink,
                        listen_sock=lsock, worker_pool_size=1,
                        serve_timeout=0.25, serve_read_deadline=0.25, **kw)
    except MinerMetadataError:
        raised = True
    finally:
        lsock.close()
    assert raised, f"omitted {label} must fail closed at the entry point"
    assert sink.published == [], f"no manifest published on omitted {label}"
    ledger = MinerLedger(os.path.join(stg, "miner_ledger.db"))
    with ledger._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM trial_context").fetchone()[0] == 0, \
            f"no trial_context may be inserted on omitted {label}"
        assert c.execute("SELECT COUNT(*) FROM stripes").fetchone()[0] == 0, \
            f"no stripe may be assigned on omitted {label}"


def _assert_entrypoint_persists(tmp, ds, run_id, kw):
    """Call run_trial_miner with a COMPLETE mandatory arg set on a pre-bound socket
    with a short timeout + no workers: it passes the guard, persists the durable
    context, and aborts (no MinerMetadataError). Assert the context row exists."""
    stg = os.path.join(tmp, "stg_ok_" + run_id)
    sink = _StubSink()
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(8)
    try:
        result = run_trial_miner(
            run_id, None, 5, "java_lcg", [1, 2, 3], 30, 0.25, 0.25, False, ds,
            staging_dir=stg, phase5_sink=sink, listen_sock=lsock,
            worker_pool_size=1, serve_timeout=1.0, serve_read_deadline=1.0, **kw)
    finally:
        lsock.close()
    assert isinstance(result, dict), "complete mandatory args must pass the guard and serve"
    ledger = MinerLedger(os.path.join(stg, "miner_ledger.db"))
    with ledger._conn() as c:
        assert c.execute("SELECT COUNT(*) FROM trial_context").fetchone()[0] == 1, \
            "a complete mandatory arg set must persist the durable trial_context"


# ---------------------------------------------------------------------------
# GATE B4 (Beta REV3 Blocker 2, corrected REV4) — omitted skip_min / skip_max at
#   the ACTUAL run_trial_miner() entry point fails closed. window_size/offset ARE
#   supplied so the ONLY missing mandatory field is the skip under test (fixes the
#   REV3 vacuity where all three were omitted). Explicit zero still succeeds.
# ---------------------------------------------------------------------------
def gateB4_omitted_skip_via_entrypoint_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw": 1}]')

        # (a) omit exactly skip_min, then exactly skip_max (all other mandatory fields
        # supplied) → each fails closed for the RIGHT reason.
        _assert_entrypoint_fails_closed(
            tmp, ds, "skip_min", {k: v for k, v in _MANDATORY_KW.items() if k != "skip_min"})
        _assert_entrypoint_fails_closed(
            tmp, ds, "skip_max", {k: v for k, v in _MANDATORY_KW.items() if k != "skip_max"})

        # (b) companion: skip_min=0, skip_max=0 EXPLICIT (with window_size/offset
        # supplied) passes the guard and persists the context — zero is legitimate.
        _assert_entrypoint_persists(tmp, ds, "runB4ok", dict(_MANDATORY_KW))


# ---------------------------------------------------------------------------
# GATE B5 (Beta REV4, window_size/offset Blocker) — omitted window_size / offset at
#   the ACTUAL run_trial_miner() entry point fails closed (REV3 fabricated 1/0).
#   skip_min/skip_max supplied so the only missing field is the one under test.
#   Explicit offset=0 still succeeds — proving 0 is a legitimate value, only omission
#   fails. Non-vacuous against a reverted copy (old code persists 1/0 with no raise).
# ---------------------------------------------------------------------------
def gateB5_omitted_window_offset_via_entrypoint_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw": 1}]')

        # (a) omit exactly window_size, then exactly offset → each fails closed.
        _assert_entrypoint_fails_closed(
            tmp, ds, "window_size", {k: v for k, v in _MANDATORY_KW.items() if k != "window_size"})
        _assert_entrypoint_fails_closed(
            tmp, ds, "offset", {k: v for k, v in _MANDATORY_KW.items() if k != "offset"})

        # (b) companion: explicit offset=0 (window_size + skip supplied) still persists
        # a valid context — proving 0 is a legitimate explicit offset, only omission
        # fails.
        _assert_entrypoint_persists(tmp, ds, "runB5ok", dict(_MANDATORY_KW))


def main():
    print("=" * 70)
    print("S172 Phase-5 D0 — metadata-seam + durable-context acceptance harness")
    print("=" * 70)
    _check("Gate D0-1: mandatory metadata present + non-empty", gate1_mandatory_fields_present)
    _check("Gate D0-2: trial-global identical / phase-specific correct", gate2_identical_global_correct_phase)
    _check("Gate D0-3: explicit string direction/skip_mode identities", gate3_explicit_string_identities)
    _check("Gate D0-4: immutable + restart-durable reconstruction", gate4_immutable_and_restart_durable)
    _check("Gate D0-5: retry attempt-1 same semantic context", gate5_retry_semantic_context_stable)
    _check("Gate D0-6: fail closed, no `{}` leak", gate6_fail_closed_no_empty_leak)
    _check("Gate D0-7: commit/abort/ack unchanged", gate7_commit_abort_ack_unchanged)
    _check("Gate B1: compare-and-insert conflict (Blocker 1)", gateB1_compare_and_insert_conflict)
    _check("Gate B2: no-fallback fail-closed + threshold_used (Blocker 2)", gateB2_no_fallback_fail_closed)
    _check("Gate B3: absent trial_context fails closed at publish (REV3 Blocker 1)", gateB3_missing_context_publish_fails_closed)
    _check("Gate B4: omitted skip via run_trial_miner fails closed (Blocker 2, REV4-corrected)", gateB4_omitted_skip_via_entrypoint_fails_closed)
    _check("Gate B5: omitted window_size/offset via run_trial_miner fails closed (REV4 Blocker)", gateB5_omitted_window_offset_via_entrypoint_fails_closed)

    print("=" * 70)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} D0 gate checks green\n")
    if passed != total:
        print("FAILURES (DO NOT COMMIT):\n")
        for name, ok, tb in _results:
            if not ok:
                print(f"--- {name} ---\n{tb}")
        return 1
    print("All D0 gate checks green — metadata seam is contract-validated "
          "(pending Team Alpha + Team Beta review).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
